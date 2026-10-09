#pragma once
// TF32 and 3xTF32 products on Hopper wgmma, built from CUTLASS 3 warp-specialized kernels with TMA
// loads and a persistent tile scheduler. In the TF32 GEMM the two consumer warp groups compute the
// whole 128 x 128 output tile from the two halves of every k-tile of 32 columns and add their
// partials in the order p0 + p1 through shared memory before the epilogue commits the tile; the
// four warps of a group hold its rows, so the block is the carrier (4, 1, 2). W can also cut the
// rows into cg contiguous slices across blocks, cg^2 <= tiles within the 2.5D bound, written as
// partial slices and summed in a fixed order. 3xTF32 writes every operand as big + small TF32
// parts: D runs the TF32 kernel on operands pre-split into three K segments (small big, big small,
// big big), and W splits X in the registers of the eight warps of its two groups, which hold the
// rows of the tile (the carrier (8, 1, 1) in the block), and promotes the tensor-core accumulator
// to FP32 every few k-tiles.
//
// Operands are read K-major through TMA: W = V^T X takes V and X column-major as they are, D takes
// a row-major copy of V that `launch_pack_vr` writes for every product.
#include "carrier_cutlass.cuh"
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized.hpp"
namespace qr_omega {
namespace gmma {
using namespace cute;
// Round to nearest (ties away) at the TF32 mantissa, as cvt.rna.tf32.f32 does for finite values.
__device__ __forceinline__ float tf32_rna(float x) {
    return __uint_as_float((__float_as_uint(x) + 0x1000u) & 0xFFFFE000u);
}
// Base is the mainloop the CUTLASS builder picks for the tile.
template <class Base> struct KSplitMainloop : Base {
    using typename Base::MainloopPipeline;
    using typename Base::PipelineState;
    using typename Base::SmemLayoutA;
    using typename Base::SmemLayoutB;
    using typename Base::TensorStorage;
    using typename Base::TiledMma;
    using typename Base::TileShape;
    static constexpr int BM = size<0>(TileShape{}), BN = size<1>(TileShape{});
    static constexpr int WG = 128;
    // One warp group covering the whole tile: the same GMMA atom, no M split across groups.
    using TiledMmaFull =
        decltype(make_tiled_mma(typename TiledMma::Atom{}, Layout<Shape<_1, _1, _1>>{}));
    static_assert(size(TiledMma{}) == 2 * WG, "cooperative (two consumer warp groups) only");
    static_assert(size(TiledMmaFull{}) == WG, "one warp group per full-tile partial");
    // Each group parks the half of its partial that the other group owns, thread-major (fragment
    // coordinates are the same in both groups for the same thread index). The exchange uses the
    // shared memory of the tile's last stage, which the group holds until the combine is done.
    static constexpr int FullVals = BM * BN / WG, HalfVals = FullVals / 2;
    static constexpr int BK_ = size<2>(TileShape{});
    static_assert(BK_ * (BM + BN) >= HalfVals * WG,
                  "a held stage must hold one half-tile exchange");
    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                            FrgTensorC &accum, int k_tile_count, int thread_idx, TensorStorage &st,
                            typename Base::Params const &) {
        Tensor sA = make_tensor(make_smem_ptr(st.smem_A.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(st.smem_B.data()), SmemLayoutB{});
        const int wg = __shfl_sync(0xFFFFFFFF, thread_idx / WG, 0), t = thread_idx % WG;
        TiledMmaFull full;
        auto thr = full.get_slice(t);
        Tensor tCsA = thr.partition_A(sA);
        Tensor tCsB = thr.partition_B(sB);
        Tensor tCrA = thr.make_fragment_A(tCsA);
        Tensor tCrB = thr.make_fragment_B(tCsB);
        Tensor accF = partition_fragment_C(full, take<0, 2>(TileShape{})); // (MMA, BM/64, BN/atomN)
        static_assert(decltype(size(accF))::value == FullVals, "full-tile partial");
        if (k_tile_count < 1) {
            clear(accF);
        }
        // Within every k-tile, warp group w contracts the k-blocks [w KB/2, (w+1) KB/2) (one TF32
        // k-block is 8 columns). Both groups consume every stage together, so the pipeline, its
        // release lag and the producer are the base's.
        constexpr int KB = decltype(size<2>(tCrA))::value;
        static_assert(KB % 2 == 0, "two K halves per k-tile");
        const int kb0 = wg * (KB / 2);
        auto issue = [&](int stage) {
            warpgroup_arrive();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < KB / 2; ++kb) {
                cute::gemm(full, tCrA(_, _, kb0 + kb, stage), tCrB(_, _, kb0 + kb, stage), accF);
                full.accumulate_ = GMMA::ScaleOut::One;
            }
        };
        // Fixed-order combine p0 + p1 over the owned half (M atoms [wg MA/2, (wg+1) MA/2)), through
        // one half-tile exchange buffer used twice (group 1 -> group 0, then group 0 -> group 1).
        // Indices are compile-time so the partial stays in registers.
        constexpr int MA = decltype(size<1>(accF))::value;
        static_assert(MA % 2 == 0, "two M halves");
        constexpr int NA = decltype(size<2>(accF))::value, VA = decltype(size<0>(accF))::value;
        float *xa = nullptr;
        float *xb = nullptr;
        auto slot = [&](int e) -> float & { return e < BM * BK_ ? xa[e] : xb[e - BM * BK_]; };
        auto put = [&](auto half) {
            constexpr int H0 = decltype(half)::value * (MA / 2);
#pragma unroll
            for (int m = 0; m < MA / 2; ++m) {
#pragma unroll
                for (int n = 0; n < NA; ++n) {
#pragma unroll
                    for (int i = 0; i < VA; ++i)
                        slot(((m * NA + n) * VA + i) * WG + t) = accF(i, H0 + m, n);
                }
            }
        };
        auto take = [&](auto half, bool own_first) {
            constexpr int H0 = decltype(half)::value * (MA / 2);
#pragma unroll
            for (int m = 0; m < MA / 2; ++m) {
#pragma unroll
                for (int n = 0; n < NA; ++n) {
#pragma unroll
                    for (int i = 0; i < VA; ++i) {
                        const float other = slot(((m * NA + n) * VA + i) * WG + t),
                                    own = accF(i, H0 + m, n);
                        accum(i, m, n) =
                            own_first ? own + other : other + own; // p0 + p1 either way
                    }
                }
            }
        };
        PipelineState smem_pipe_release = smem_pipe_read;
        constexpr int K_PIPE_MMAS = 1;
        const int prologue = cute::min(K_PIPE_MMAS, k_tile_count);
        full.accumulate_ = GMMA::ScaleOut::Zero;
        warpgroup_fence_operand(accF);
        for (int p = 0; p < prologue; ++p) {
            auto tok = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, tok);
            issue(smem_pipe_read.index());
            warpgroup_commit_batch();
            ++smem_pipe_read;
        }
        warpgroup_fence_operand(accF);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = prologue; k < k_tile_count; ++k) {
            auto tok = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, tok);
            warpgroup_fence_operand(accF);
            issue(smem_pipe_read.index());
            warpgroup_commit_batch();
            warpgroup_wait<K_PIPE_MMAS>();
            warpgroup_fence_operand(accF);
            pipeline.consumer_release(smem_pipe_release);
            ++smem_pipe_read;
            ++smem_pipe_release;
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(accF);
        // The last stage is the exchange buffer; it is released after the combine.
        const int held = smem_pipe_release.index();
        xa = reinterpret_cast<float *>(st.smem_A.data()) + size_t(held) * BM * BK_;
        xb = reinterpret_cast<float *>(st.smem_B.data()) + size_t(held) * BN * BK_;
        cutlass::arch::NamedBarrier::sync(2 * WG, 8); // both groups retired their GMMAs
        if (wg == 1)
            put(Int<0>{}); // p1 of the half group 0 owns
        cutlass::arch::NamedBarrier::sync(2 * WG, 8);
        if (wg == 0)
            take(Int<0>{}, true); // p0 (own) + p1
        cutlass::arch::NamedBarrier::sync(2 * WG, 8);
        if (wg == 0)
            put(Int<1>{}); // p0 of the half group 1 owns
        cutlass::arch::NamedBarrier::sync(2 * WG, 8);
        if (wg == 1)
            take(Int<1>{}, false); // p0 + p1 (own)
        // The generic-proxy use of the stage ends before TMA refills it.
        cutlass::arch::fence_view_async_shared();
        pipeline.consumer_release(smem_pipe_release);
    }
    CUTLASS_DEVICE void mma_tail(MainloopPipeline, PipelineState, int) {
        warpgroup_wait<0>();
    }
};
// C(M x N) = alpha A(M x K) B(K x N) + beta C in TF32 from FP32 storage: A row-major and B
// column-major (both K-major), C in layout LC. The TMA rounds the operands to TF32 in flight.
template <int BM, int BN, class LC> struct KSplitGemm {
    using TS = Shape<Int<BM>, Int<BN>, _32>;
    using CS = Shape<_1, _1, _1>;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, float, LC, 4, float, LC, 4,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
    using BaseMain = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, float, cutlass::layout::RowMajor, 4,
        float, cutlass::layout::ColumnMajor, 4, float, TS, CS,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Main = KSplitMainloop<BaseMain>;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    static constexpr int BK = 32;
};
// Pipeline stages of the register-split W: four stages of A and two B would exceed shared memory
// next to the FP32 epilogue.
inline constexpr int x3rs_stages = 3;
// k-tiles between promotions of its tensor-core accumulator to FP32. Across GPUs every other k-tile:
// with four the errors grow to about 1.4 times the published ones.
#ifdef QR_OMEGA_MULTI_GPU
inline constexpr int x3rs_promote = 2;
#else
inline constexpr int x3rs_promote = 4;
#endif
template <class Base> struct X3RSMainloop : Base {
    using typename Base::DispatchPolicy;
    using typename Base::MainloopPipeline;
    using typename Base::PipelineState;
    using typename Base::SmemLayoutA;
    using typename Base::SmemLayoutB;
    using typename Base::TiledMma;
    using typename Base::TileShape;
    using ClusterShape = typename DispatchPolicy::ClusterShape;
    static constexpr int BM = size<0>(TileShape{}), BN = size<1>(TileShape{}),
                         BK = size<2>(TileShape{});
    static constexpr int PromoteKTiles = x3rs_promote;
    static_assert(size(TiledMma{}) == 256, "cooperative: two consumer warp groups split M");
    struct TensorStorage : Base::TensorStorage {
        cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>,
                            Base::SmemAlignmentB>
            smem_Bs;
    };
    struct SharedStorage {
        TensorStorage tensors;
        typename Base::PipelineStorage pipeline;
    };
    using PipelineStorage = typename Base::PipelineStorage;
    static constexpr uint32_t TmaBytes =
        Base::TmaTransactionBytesMK + 2 * Base::TmaTransactionBytesNK;
    struct Arguments : Base::Arguments {
        typename Base::ElementB const *ptr_Bs = nullptr;
    };
    using TMA_B = typename Base::Params::TMA_B;
    struct Params : Base::Params {
        TMA_B tma_load_bs;
    };
    template <class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const &ps, Arguments const &args, void *ws) {
        Params p{Base::to_underlying_arguments(
                     ps, static_cast<typename Base::Arguments const &>(args), ws),
                 {}};
        auto [M, N, K, L] = append<4>(ps, 1);
        (void)M;
        Tensor tb =
            make_tensor(reinterpret_cast<typename Base::InternalElementB const *>(args.ptr_Bs),
                        make_layout(make_shape(N, K, L), args.dB));
        p.tma_load_bs =
            make_tma_copy_B_sm90(typename Base::GmemTiledCopyB{}, tb,
                                 SmemLayoutB{}(_, _, cute::Int<0>{}), TileShape{}, ClusterShape{});
        p.tma_transaction_bytes = TmaBytes;
        return p;
    }
    CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const &p) {
        cute::prefetch_tma_descriptor(p.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(p.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(p.tma_load_bs.get_tma_descriptor());
    }
    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const &ps, Params const &p) const {
        using X = Underscore;
        auto [M, N, K, L] = ps;
        Tensor mA = p.tma_load_a.get_tma_tensor(make_shape(M, K, L));
        Tensor mB = p.tma_load_b.get_tma_tensor(make_shape(N, K, L));
        Tensor mBs = p.tma_load_bs.get_tma_tensor(make_shape(N, K, L));
        Tensor gA = local_tile(mA, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor gB = local_tile(mB, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
        Tensor gBs = local_tile(mBs, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
        return cute::make_tuple(gA, gB, gBs);
    }
    template <class TensorA, class TensorB, class TensorBs, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void
    load(Params const &p, MainloopPipeline pipeline, PipelineState smem_pipe_write,
         cute::tuple<TensorA, TensorB, TensorBs> const &in, BlockCoord const &blk_coord,
         KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
         uint32_t block_rank_in_cluster, TensorStorage &st) {
        if (cute::elect_one_sync()) {
            Tensor sA = as_position_independent_swizzle_tensor(
                make_tensor(make_smem_ptr(st.smem_A.data()), SmemLayoutA{}));
            Tensor sB = as_position_independent_swizzle_tensor(
                make_tensor(make_smem_ptr(st.smem_B.data()), SmemLayoutB{}));
            Tensor sBs = as_position_independent_swizzle_tensor(
                make_tensor(make_smem_ptr(st.smem_Bs.data()), SmemLayoutB{}));
            auto [m, n, k, l] = blk_coord;
            Tensor gA = get<0>(in)(_, _, m, _, l);
            Tensor gB = get<1>(in)(_, _, n, _, l);
            Tensor gBs = get<2>(in)(_, _, n, _, l);
            auto ta = p.tma_load_a.get_slice(0);
            auto tb = p.tma_load_b.get_slice(0);
            auto tbs = p.tma_load_bs.get_slice(0);
            Tensor tAgA = ta.partition_S(gA);
            Tensor tAsA = ta.partition_D(sA);
            Tensor tBgB = tb.partition_S(gB);
            Tensor tBsB = tb.partition_D(sB);
            Tensor tBgBs = tbs.partition_S(gBs);
            Tensor tBsBs = tbs.partition_D(sBs);
            CUTLASS_PRAGMA_NO_UNROLL
            for (; k_tile_count > 0; --k_tile_count) {
                pipeline.producer_acquire(smem_pipe_write);
                auto *bar = pipeline.producer_get_barrier(smem_pipe_write);
                const int s = smem_pipe_write.index();
                copy(p.tma_load_a.with(*bar, 0), tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, s));
                copy(p.tma_load_b.with(*bar, 0), tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, s));
                copy(p.tma_load_bs.with(*bar, 0), tBgBs(_, _, _, *k_tile_iter), tBsBs(_, _, _, s));
                ++k_tile_iter;
                ++smem_pipe_write;
            }
        }
    }
    // accum is the warp group's 64 x BN accumulator, accF the tensor-core accumulator of a chunk.
    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                            FrgTensorC &accum, int k_tile_count, int thread_idx, TensorStorage &st,
                            Params const &) {
        Tensor sA = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(st.smem_A.data()), SmemLayoutA{}));
        Tensor sB = make_tensor(make_smem_ptr(st.smem_B.data()), SmemLayoutB{});
        Tensor sBs = make_tensor(make_smem_ptr(st.smem_Bs.data()), SmemLayoutB{});
        TiledMma tiled_mma;
        const int wg = __shfl_sync(0xFFFFFFFF, thread_idx / 128, 0);
        auto thr = tiled_mma.get_thread_slice(thread_idx);
        auto wgs = tiled_mma.get_slice(make_layout(Int<2>{}, Int<128>{})(wg));
        Tensor tCrA = thr.partition_fragment_A(
            sA(_, _, Int<0>{}));                 // (MMA,MMA_M,MMA_K) raw bits, then A_big in place
        Tensor tCrAs = make_fragment_like(tCrA); // A_small
        Tensor tCrB =
            wgs.make_fragment_B(wgs.partition_B(sB)); // (MMA,MMA_N,MMA_K,PIPE) descriptors
        Tensor tCrBs = wgs.make_fragment_B(wgs.partition_B(sBs));
        auto cpA = make_tiled_copy_A(typename Base::InternalSmemCopyAtomA{}, tiled_mma);
        auto thr_cpA = cpA.get_thread_slice(thread_idx);
        Tensor tCsA_v = thr_cpA.partition_S(sA);
        Tensor tCrA_v = thr_cpA.retile_D(tCrA);
        Tensor accF = make_fragment_like(accum);
        constexpr int KB = size<2>(tCrA);
        auto split = [&](int kb) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCrA(_, _, kb)); ++i) {
                const float x = __uint_as_float(tCrA(_, _, kb)(i).raw());
                const float b = tf32_rna(x);
                tCrA(_, _, kb)(i) = cutlass::tfloat32_t::bitcast(__float_as_uint(b));
                tCrAs(_, _, kb)(i) = cutlass::tfloat32_t::bitcast(__float_as_uint(tf32_rna(x - b)));
            }
        };
        PipelineState release = smem_pipe_read;
        bool fresh = true; // next GMMA starts a chunk (ScaleOut::Zero)
        bool first_chunk = true;
        warpgroup_fence_operand(accF);
        warpgroup_fence_operand(accum);
        for (int t = 0; t < k_tile_count; ++t) {
            auto tok = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, tok);
            const int s = smem_pipe_read.index();
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                if (t > 0) {
                    warpgroup_wait<KB - 1>();
                    if (kb == KB - 1) {
                        pipeline.consumer_release(release);
                        ++release;
                    }
                } // (t-1,kb) retired
                copy(cpA, tCsA_v(_, _, kb, s), tCrA_v(_, _, kb));
                split(kb);
                warpgroup_fence_operand(accF);
                warpgroup_arrive();
                tiled_mma.accumulate_ = fresh ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
                fresh = false;
                cute::gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb, s), accF);
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
                cute::gemm(tiled_mma, tCrAs(_, _, kb), tCrB(_, _, kb, s), accF);
                cute::gemm(tiled_mma, tCrA(_, _, kb), tCrBs(_, _, kb, s), accF);
                warpgroup_commit_batch();
                warpgroup_fence_operand(accF);
            }
            ++smem_pipe_read;
            if ((t + 1) % PromoteKTiles == 0 && t + 1 < k_tile_count) {
                warpgroup_wait<0>();
                warpgroup_fence_operand(accF);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(accF); ++i)
                    accum(i) = first_chunk ? accF(i) : accum(i) + accF(i);
                first_chunk = false;
                fresh = true;
                warpgroup_fence_operand(accum);
            }
        }
        warpgroup_wait<0>();
        warpgroup_fence_operand(accF);
        if (k_tile_count > 0) {
            pipeline.consumer_release(release);
            ++release;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accF); ++i)
            accum(i) = k_tile_count < 1 ? 0.f : (first_chunk ? accF(i) : accum(i) + accF(i));
        warpgroup_fence_operand(accum);
    }
    CUTLASS_DEVICE void mma_tail(MainloopPipeline, PipelineState, int) {
        warpgroup_wait<0>();
    }
};
// C(M x N) = A(M x K) B(N x K)^T with both operands K-major; C is W^T, row-major.
template <int BM = 128, int BN = 128> struct X3RSGemm {
    using E = cutlass::tfloat32_t;
    using TS = Shape<Int<BM>, Int<BN>, _32>;
    using CS = Shape<_1, _1, _1>;
    using LC = cutlass::layout::RowMajor;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, float, LC, 4, float, LC, 4,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
    using Atom = cute::SM90::GMMA::MMA_64x128x8_F32TF32TF32_RS_TN<>;
    static_assert(BN == 128, "RS atom is 64x128x8");
    using TiledMmaRS = decltype(make_tiled_mma(Atom{}, Layout<Shape<_2, _1, _1>>{}));
    using SmemAtom = GMMA::Layout_K_SW128_Atom<E>;
    using StrideAB = cute::tuple<int64_t, cute::Int<1>, int64_t>;
    using BaseMain = cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecialized<
            x3rs_stages, CS, cutlass::gemm::KernelTmaWarpSpecializedCooperative>,
        TS, E, StrideAB, E, StrideAB, TiledMmaRS, cute::SM90_TMA_LOAD, SmemAtom,
        cute::Copy_Atom<cute::AutoVectorizingCopy, E>, cute::identity, cute::SM90_TMA_LOAD,
        SmemAtom, void, cute::identity>;
    using Main = X3RSMainloop<BaseMain>;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
} // namespace gmma

// TMA needs a 16-byte aligned base and a leading dimension that is a multiple of four floats.
template <class P> bool gmma_aligned(const P *p, long long ld) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (ld % 4 == 0);
}
template <class T> bool gmma_tf32() {
    return std::is_same_v<T, float> && fp32_math() == Fp32Math::TF32;
}
template <class T> bool gmma_x3() {
    return std::is_same_v<T, float> && fp32_math() == Fp32Math::X3;
}
template <class T> bool gmma_enabled() {
    return gmma_tf32<T>() || gmma_x3<T>();
}
// K extent of one 3xTF32 segment: h rounded up to whole k-tiles.
inline int gmma_x3_seg(int h) {
    return (h + 31) / 32 * 32;
}
// Rows of one of the cg GPU-level slices of W: whole 32-row k-tiles. The remainder rows - cg ks
// belongs to the last slice.
inline int gmma_w_slice(int rows, int cg) {
    return cg > 1 ? rows / (32 * cg) * 32 : rows;
}
// The 128 x 128 tiles of an M x N output, the GPU-level fibres of the wgmma products.
inline long long gmma_tiles(int m, int n) {
    return (long long)ceildiv(m, 128) * ceildiv(n, 128);
}
// Carriers of the TF32 GEMM over an M x N output with cg slices of K, and of the 3xTF32 W.
inline Carrier gmma_carrier(int m, int n, int cg) {
    return Carrier{}.set(GpuLevel, ceildiv(m, 128), ceildiv(n, 128), cg).set(BlockLevel, 4, 1, 2);
}
inline Carrier gmma_w3rs_carrier(int h, int q, int c) {
    return Carrier{}.set(GpuLevel, ceildiv(h, 128), ceildiv(q, 128), c).set(BlockLevel, 1, 8, 1);
}
inline bool gmma_w_supported(int c, int rows) {
    return c >= 2 && c % 2 == 0 && rows >= 16 * c;
}
// Z3[n*ld3 + {0,hs,2hs} + k] = {big, small, big} of Z[k + n*ldz] (zero for k >= h), 3hs x q
// column-major.
__global__ void pack_z3(const float *__restrict__ Z, int ldz, int h, int hs, int q,
                        float *__restrict__ Z3, int ld3) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)q * hs;
         i += (long long)gridDim.x * blockDim.x) {
        const long long n = i / hs;
        const int k = int(i % hs);
        const float z = k < h ? Z[k + n * ldz] : 0.f, b = gmma::tf32_rna(z),
                    m = gmma::tf32_rna(z - b);
        float *o = Z3 + n * ld3;
        o[k] = b;
        o[hs + k] = m;
        o[2 * hs + k] = b;
    }
}
inline int pack_vr_grid(long long n) {
    return int(std::max(32LL, std::min(4096LL, (n + 1023) / 1024)));
}
// VR[r*ldvr + k] = V[r + k*ldv] through a padded shared tile, so both the loads of V and the stores
// of VR are coalesced. X3 writes the three segments {small, big, big} of every value instead.
template <bool X3>
__global__ void pack_vr_tiled(const float *__restrict__ V, int ldv, int rows, int h, int hs,
                              float *__restrict__ VR, int ldvr) {
    __shared__ float tile[32][33];
    const int nr = (rows + 31) / 32, nk = ((X3 ? hs : h) + 31) / 32;
    for (long long ti = blockIdx.x; ti < (long long)nr * nk; ti += gridDim.x) {
        const int r0 = int(ti % nr) * 32, k0 = int(ti / nr) * 32;
        const int r = r0 + threadIdx.x;
#pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int k = k0 + threadIdx.y + j;
            tile[threadIdx.y + j][threadIdx.x] =
                (r < rows && k < h) ? V[r + (long long)k * ldv] : 0.f;
        }
        __syncthreads();
        const int k = k0 + threadIdx.x;
#pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int rr = r0 + threadIdx.y + j;
            if (rr < rows && k < (X3 ? hs : h)) {
                const float v = tile[threadIdx.x][threadIdx.y + j];
                float *o = VR + (long long)rr * ldvr;
                if constexpr (X3) {
                    const float b = gmma::tf32_rna(v), s = gmma::tf32_rna(v - b);
                    o[k] = s;
                    o[hs + k] = b;
                    o[2 * hs + k] = b;
                } else
                    o[k] = v;
            }
        }
        __syncthreads();
    }
}
inline void launch_pack_vr(const float *v, int ldv, int rows, int h, int hs, float *vr, int ldvr,
                           bool x3, cudaStream_t st) {
    const int grid = int(std::min(4096LL, (long long)ceildiv(rows, 32) * ceildiv(x3 ? hs : h, 32)));
    if (x3)
        pack_vr_tiled<true><<<grid, dim3(32, 8), 0, st>>>(v, ldv, rows, h, hs, vr, ldvr);
    else
        pack_vr_tiled<false><<<grid, dim3(32, 8), 0, st>>>(v, ldv, rows, h, h, vr, ldvr);
    CU(cudaGetLastError());
}
namespace gmma_detail {
using LCM = cutlass::layout::ColumnMajor;
using GemmD = gmma::KSplitGemm<128, 128, LCM>::Gemm;
using GemmW3RS = gmma::X3RSGemm<>::Gemm;

// tiles_per_block == 0 launches the persistent grid, which holds every SM for the whole product. A
// positive value launches one block per that many output tiles instead, so the product releases its
// SMs tile by tile to a panel running at a higher priority.
template <class Gemm> void run(typename Gemm::Arguments &a, cudaStream_t st, int tiles_per_block) {
    if (tiles_per_block > 0) {
        using K = typename Gemm::GemmKernel;
        using Scheduler = typename K::TileScheduler;
        static_assert(cute::size(typename K::ClusterShape{}) == 1, "one block per cluster");
        CU(cudaGetDevice(&a.hw_info.device_id));
        a.hw_info.sm_count = 1 << 30; // the scheduler then sizes its grid to the tile count
        const auto grid = Scheduler::get_grid_shape(
            typename Scheduler::Params{}, a.problem_shape, typename K::TileShape{},
            typename K::ClusterShape{}, a.hw_info, a.scheduler);
        const unsigned long long tiles = uint64_t(grid.x) * grid.y * grid.z,
                                 blocks = (tiles + tiles_per_block - 1) / tiles_per_block;
        // CUTLASS rasterizes a tall D along grid.y, which CUDA caps at 65535.
        a.hw_info.sm_count =
            int(std::max(1ULL, std::min(blocks, grid.y > 1 ? 65535ULL : 2147483647ULL)));
    }
    Gemm g;
    if (Gemm::get_workspace_size(a) || g.can_implement(a) != cutlass::Status::kSuccess ||
        g.initialize(a, nullptr, st) != cutlass::Status::kSuccess ||
        g.run(st) != cutlass::Status::kSuccess)
        throw std::runtime_error("wgmma kernel launch failed");
    CU(cudaGetLastError());
}
template <class Gemm> bool implementable(const typename Gemm::Arguments &a) {
    return Gemm::can_implement(a) == cutlass::Status::kSuccess;
}
inline void combine_slices(const float *part, long long sp, int cg, float *w, int ldw, int q, int h,
                           cudaStream_t st) {
    carrier_g_combine<float>
        <<<std::min(1024, ceildiv(h * q, 256)), 256, 0, st>>>(part, sp, cg, w, ldw, q, h, 1, 0);
    CU(cudaGetLastError());
}
} // namespace gmma_detail
// Arguments of the products, shared by the admission tests and the launches so that what is
// admitted is exactly what runs.
inline gmma_detail::GemmD::Arguments gmma_d_args(const float *vr, int ldvr, const float *z, int ldz,
                                                 float *x, int ldx, int rows, int h, int q) {
    using G = gmma_detail::GemmD;
    typename G::Arguments a;
    a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {rows, q, h, 1};
    a.mainloop.ptr_A = reinterpret_cast<decltype(a.mainloop.ptr_A)>(vr);
    a.mainloop.dA = typename G::GemmKernel::StrideA{int64_t(ldvr), cute::Int<1>{}, int64_t(0)};
    a.mainloop.ptr_B = reinterpret_cast<decltype(a.mainloop.ptr_B)>(z);
    a.mainloop.dB = typename G::GemmKernel::StrideB{int64_t(ldz), cute::Int<1>{}, int64_t(0)};
    a.epilogue.thread.alpha = -1.f;
    a.epilogue.thread.beta = 1.f;
    a.epilogue.ptr_C = x;
    a.epilogue.dC = typename G::GemmKernel::StrideC{cute::Int<1>{}, int64_t(ldx), int64_t(0)};
    a.epilogue.ptr_D = x;
    a.epilogue.dD = a.epilogue.dC;
    return a;
}
inline gmma_detail::GemmD::Arguments gmma_w_args(int c, const float *v, int ldv, const float *x,
                                                 int ldx, float *w, int ldw, int rows, int h, int q,
                                                 float *part, long long sp) {
    using G = gmma_detail::GemmD;
    const int cg = c / 2, ks = gmma_w_slice(rows, cg);
    typename G::Arguments a;
    a.mode = cg > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                    : cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {h, q, ks, cg};
    a.mainloop.ptr_A = v;
    a.mainloop.dA = typename G::GemmKernel::StrideA{int64_t(ldv), cute::Int<1>{}, int64_t(ks)};
    a.mainloop.ptr_B = x;
    a.mainloop.dB = typename G::GemmKernel::StrideB{int64_t(ldx), cute::Int<1>{}, int64_t(ks)};
    a.epilogue.thread.alpha = 1.f;
    a.epilogue.thread.beta = 0.f;
    float *o = cg > 1 ? part : w;
    a.epilogue.ptr_C = o;
    a.epilogue.dC =
        typename G::GemmKernel::StrideC{cute::Int<1>{}, int64_t(ldw), int64_t(cg > 1 ? sp : 0)};
    a.epilogue.ptr_D = o;
    a.epilogue.dD = a.epilogue.dC;
    return a;
}
inline gmma_detail::GemmW3RS::Arguments gmma_w3rs_args(const float *x, int ldx, const float *vb,
                                                       const float *vs, int ldvs, float *o, int ldw,
                                                       long long sp, int q, int h, int ks, int L,
                                                       float beta) {
    using G = gmma_detail::GemmW3RS;
    using E = cutlass::tfloat32_t;
    typename G::Arguments a;
    a.mode = L > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                   : cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {q, h, ks, L};
    a.mainloop.ptr_A = reinterpret_cast<const E *>(x);
    a.mainloop.dA = {int64_t(ldx), cute::Int<1>{}, int64_t(ks)};
    a.mainloop.ptr_B = reinterpret_cast<const E *>(vb);
    a.mainloop.dB = {int64_t(ldvs), cute::Int<1>{}, int64_t(ks)};
    a.mainloop.ptr_Bs = reinterpret_cast<const E *>(vs);
    a.epilogue.thread.alpha = 1.f;
    a.epilogue.thread.beta = beta;
    a.epilogue.ptr_C = o;
    a.epilogue.dC = {int64_t(ldw), cute::Int<1>{}, int64_t(L > 1 ? sp : 0)};
    a.epilogue.ptr_D = o;
    a.epilogue.dD = a.epilogue.dC;
    return a;
}
// V (rows x h, col-major ldv) -> V_big, V_small (col-major, ld ldvs), tf32 round-to-nearest split.
__global__ void split_v_rn(const float *__restrict__ V, int ldv, int rows, int h,
                           float *__restrict__ Vb, float *__restrict__ Vs, int ldvs) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)rows * h;
         i += (long long)gridDim.x * blockDim.x) {
        const int r = int(i % rows);
        const long long k = i / rows;
        const float v = V[r + k * ldv], b = gmma::tf32_rna(v);
        Vb[r + k * ldvs] = b;
        Vs[r + k * ldvs] = gmma::tf32_rna(v - b);
    }
}
// Leading dimension of the split V (V_big then V_small, column-major).
inline int gmma_vsplit_ld(int rows) {
    return (rows + 3) / 4 * 4;
}

// X(rows x q) -= VR Z.
inline bool gmma_d_admits(const float *vr, int ldvr, const float *z, int ldz, float *x, int ldx,
                          int rows, int h, int q) {
    return h >= 32 && rows && q && gmma_aligned(vr, ldvr) && gmma_aligned(z, ldz) &&
           gmma_aligned(x, ldx) &&
           gmma_detail::implementable<gmma_detail::GemmD>(
               gmma_d_args(vr, ldvr, z, ldz, x, ldx, rows, h, q));
}
inline void launch_gmma_d(const float *vr, int ldvr, const float *z, int ldz, float *x, int ldx,
                          int rows, int h, int q, cudaStream_t st, int tiles_per_block) {
    auto a = gmma_d_args(vr, ldvr, z, ldz, x, ldx, rows, h, q);
    gmma_detail::run<gmma_detail::GemmD>(a, st, tiles_per_block);
}
// Z(h x q) = T^T W in TF32, with the K halves of D.
inline bool gmma_z_admits(const float *t, int ldt, const float *w, int ldw, float *z, int ldz,
                          int h, int q) {
    return gmma_tf32<float>() && h >= 32 && q && gmma_aligned(t, ldt) && gmma_aligned(w, ldw) &&
           gmma_aligned(z, ldz) &&
           gmma_detail::implementable<gmma_detail::GemmD>(
               gmma_w_args(2, t, ldt, w, ldw, z, ldz, h, h, q, nullptr, 0));
}
inline void launch_gmma_z(const float *t, int ldt, const float *w, int ldw, float *z, int ldz,
                          int h, int q, cudaStream_t st, int tiles_per_block) {
    auto a = gmma_w_args(2, t, ldt, w, ldw, z, ldz, h, h, q, nullptr, 0);
    gmma_detail::run<gmma_detail::GemmD>(a, st, tiles_per_block);
}
// W(h x q, ldw) = V^T X in TF32 with c = 2 cg. With cg > 1 the slices go to `part` (stride sp >=
// ldw q) and the remainder rows accumulate into the last slice before the combine.
inline bool gmma_w_admits(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                          int ldw, int rows, int h, int q, float *part, long long sp) {
    using G = gmma_detail::GemmD;
    if (!gmma_w_supported(c, rows) || !h || !q || !gmma_aligned(v, ldv) || !gmma_aligned(x, ldx) ||
        !gmma_aligned(w, ldw) || (c > 2 && !gmma_aligned(part, sp)) ||
        !gmma_detail::implementable<G>(
            gmma_w_args(c, v, ldv, x, ldx, w, ldw, rows, h, q, part, sp)))
        return false;
    const int cg = c / 2, ks = gmma_w_slice(rows, cg), rem = rows - cg * ks;
    if (!rem)
        return true;
    const size_t o = size_t(cg) * ks;
    return gmma_aligned(v + o, ldv) && gmma_aligned(x + o, ldx) &&
           gmma_detail::implementable<G>(
               gmma_w_args(2, v + o, ldv, x + o, ldx, w, ldw, rem, h, q, nullptr, 0));
}
inline void launch_gmma_w(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                          int ldw, int rows, int h, int q, cudaStream_t st, float *part,
                          long long sp, int tiles_per_block) {
    using G = gmma_detail::GemmD;
    auto a = gmma_w_args(c, v, ldv, x, ldx, w, ldw, rows, h, q, part, sp);
    gmma_detail::run<G>(a, st, tiles_per_block);
    const int cg = c / 2, ks = gmma_w_slice(rows, cg), rem = rows - cg * ks;
    if (cg == 1)
        return;
    if (rem) {
        const size_t o = size_t(cg) * ks;
        auto r = gmma_w_args(2, v + o, ldv, x + o, ldx, part + size_t(cg - 1) * size_t(sp), ldw,
                             rem, h, q, nullptr, 0);
        r.epilogue.thread.beta = 1.f;
        gmma_detail::run<G>(r, st, tiles_per_block);
    }
    gmma_detail::combine_slices(part, sp, cg, w, ldw, q, h, st);
}
// W in 3xTF32 with X split in registers: c contiguous row slices at the GPU level (the two warp
// groups split the output rows here), slice g written at w + g sp, the remainder rows accumulated
// into the last slice, then the fixed-order combine commits W.
inline bool gmma_w3rs_admits(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                             int ldw, int rows, int h, int q, const float *vsplit) {
    using G = gmma_detail::GemmW3RS;
    if (c < 1 || !h || !q || rows < 32 * c || !vsplit)
        return false;
    const int ldvs = gmma_vsplit_ld(rows);
    const float *vb = vsplit, *vs = vsplit + size_t(ldvs) * h;
    const long long sp = (long long)ldw * q;
    if (!gmma_aligned(v, ldv) || !gmma_aligned(x, ldx) || !gmma_aligned(w, ldw) ||
        !gmma_aligned(vb, ldvs) || !gmma_aligned(vs, ldvs) || !gmma_aligned(w, sp))
        return false;
    const int ks = gmma_w_slice(rows, c), rem = rows - c * ks;
    if (!gmma_detail::implementable<G>(
            gmma_w3rs_args(x, ldx, vb, vs, ldvs, w, ldw, sp, q, h, ks, c, 0.f)))
        return false;
    if (!rem)
        return true;
    const size_t o = size_t(c) * ks;
    return gmma_aligned(x + o, ldx) && gmma_aligned(vb + o, ldvs) && gmma_aligned(vs + o, ldvs) &&
           gmma_detail::implementable<G>(gmma_w3rs_args(x + o, ldx, vb + o, vs + o, ldvs,
                                                        w + size_t(c - 1) * sp, ldw, sp, q, h, rem,
                                                        1, 1.f));
}
inline void launch_gmma_w3rs(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                             int ldw, int rows, int h, int q, cudaStream_t st, float *vsplit,
                             int tiles_per_block) {
    using G = gmma_detail::GemmW3RS;
    const int ldvs = gmma_vsplit_ld(rows);
    float *vb = vsplit, *vs = vsplit + size_t(ldvs) * h;
    split_v_rn<<<pack_vr_grid((long long)rows * h), 256, 0, st>>>(v, ldv, rows, h, vb, vs, ldvs);
    CU(cudaGetLastError());
    const long long sp = (long long)ldw * q;
    const int ks = gmma_w_slice(rows, c), rem = rows - c * ks;
    auto a = gmma_w3rs_args(x, ldx, vb, vs, ldvs, w, ldw, sp, q, h, ks, c, 0.f);
    gmma_detail::run<G>(a, st, tiles_per_block);
    if (rem) {
        const size_t o = size_t(c) * ks;
        auto r = gmma_w3rs_args(x + o, ldx, vb + o, vs + o, ldvs, w + size_t(c - 1) * sp, ldw, sp,
                                q, h, rem, 1, 1.f);
        gmma_detail::run<G>(r, st, tiles_per_block);
    }
    if (c > 1)
        gmma_detail::combine_slices(w, sp, c, w, ldw, q, h, st);
}
// G(h x q) = V1^T V2 of an aggregated T in TF32: cg GPU-level row slices of whole k-tiles, the last
// one also taking the remainder, each with the two warp-group peers of W.
inline bool gmma_g_admits(int cg, const float *v, int ldv, const float *x, int ldx, float *g,
                          int ldg, int rows, int h, int q, float *part, size_t part_words) {
    using G = gmma_detail::GemmD;
    if (!gmma_tf32<float>() || cg < 1 || cg > 128 || rows < 32 * cg || h < 1 || h > 128 || q < 1 ||
        q > 128 || !gmma_aligned(v, ldv) || !gmma_aligned(x, ldx) || !gmma_aligned(g, ldg) ||
        (cg > 1 &&
         (!part || part_words < size_t(cg) * ldg * q || !gmma_aligned(part, (long long)ldg * q))))
        return false;
    const long long sp = (long long)ldg * q;
    const int ks = gmma_w_slice(rows, cg), rem = rows - cg * ks;
    if (!gmma_detail::implementable<G>(
            gmma_w_args(2 * cg, v, ldv, x, ldx, g, ldg, rows, h, q, part, sp)))
        return false;
    if (!rem)
        return true;
    const size_t o = size_t(cg) * ks;
    auto tail = gmma_w_args(2, v + o, ldv, x + o, ldx, part + size_t(cg - 1) * sp, ldg, rem, h, q,
                            nullptr, 0);
    tail.epilogue.thread.beta = 1.f;
    return gmma_detail::implementable<G>(tail);
}
inline void launch_gmma_g(int cg, const float *v, int ldv, const float *x, int ldx, float *g,
                          int ldg, int rows, int h, int q, float *part, cudaStream_t st) {
    using G = gmma_detail::GemmD;
    const long long sp = (long long)ldg * q;
    const int ks = gmma_w_slice(rows, cg), rem = rows - cg * ks;
    auto a = gmma_w_args(2 * cg, v, ldv, x, ldx, g, ldg, rows, h, q, part, sp);
    gmma_detail::run<G>(a, st, 0);
    if (cg == 1)
        return;
    if (rem) {
        const size_t o = size_t(cg) * ks;
        auto tail = gmma_w_args(2, v + o, ldv, x + o, ldx, part + size_t(cg - 1) * sp, ldg, rem, h,
                                q, nullptr, 0);
        tail.epilogue.thread.beta = 1.f;
        gmma_detail::run<G>(tail, st, 0);
    }
    gmma_detail::combine_slices(part, sp, cg, g, ldg, q, h, st);
}
} // namespace qr_omega
