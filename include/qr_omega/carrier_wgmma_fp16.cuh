#pragma once
// 3xFP16 W and D on Hopper wgmma: every FP32 operand x is written as big + small FP16 parts
// b = fp16(s x), m = fp16(s x - b) of a power-of-two prescaled value, the three products b b +
// b m + m b run on FP16 tensor cores with FP32 accumulation, and the epilogue's alpha undoes the
// scales. The prescale puts an operand's largest magnitude near 2^14, so that the small part of
// every value within 2^-17 of it stays a normal FP16 number (FP16 has 5 exponent bits; without it
// the small parts of typical Householder entries, about m^-1/2, would be subnormal):
//   X (W's register operand): |x| <= ||A(:, j)|| <= sqrt(rows) after the columns were scaled to
//     unit maximum, a fixed bound;
//   V: a fixed 2^12 (Householder entries are at most about 1);
//   Z (D's): its exact maximum, reduced on the device, with alpha read from device memory.
// W keeps the structure of the 3xTF32 W (X split in registers, V pre-split, tensor-core
// accumulator promoted to FP32 every x3rs_promote k-tiles of 32); D runs a plain FP16 GEMM on
// operands pre-split into three K segments, as the 3xTF32 D does.
#include "carrier_wgmma.cuh"
#include <cuda_fp16.h>
namespace qr_omega {
namespace gmma {
using namespace cute;
inline constexpr float x3h_v_scale = 4096.f; // 2^12
__device__ __forceinline__ void split_h(float x, cutlass::half_t &b, cutlass::half_t &m) {
    const __half hb = __float2half_rn(x);
    b = cutlass::half_t::bitcast(__half_as_ushort(hb));
    m = cutlass::half_t::bitcast(__half_as_ushort(__float2half_rn(x - __half2float(hb))));
}

template <class Base> struct X3HRSMainloop : Base {
    using typename Base::DispatchPolicy;
    using typename Base::MainloopPipeline;
    using typename Base::PipelineState;
    using typename Base::SmemLayoutA;
    using typename Base::SmemLayoutB;
    using typename Base::TiledMma;
    using typename Base::TileShape;
    using ClusterShape = typename DispatchPolicy::ClusterShape;
    static constexpr int PromoteKTiles = x3rs_promote;
    static_assert(size(TiledMma{}) == 256, "cooperative: two consumer warp groups split M");
    // A (X) stays FP32 in shared memory; B (V) and its small part are FP16.
    struct TensorStorage : cute::aligned_struct<128, _0> {
        cute::array_aligned<float, cute::cosize_v<SmemLayoutA>, Base::SmemAlignmentA> smem_A;
        cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutB>, Base::SmemAlignmentB>
            smem_B;
        cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutB>, Base::SmemAlignmentB>
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
        float x_scale = 1.f;
    };
    using TMA_B = typename Base::Params::TMA_B;
    struct Params : Base::Params {
        TMA_B tma_load_bs;
        float x_scale;
    };
    template <class ProblemShape>
    static Params to_underlying_arguments(ProblemShape const &ps, Arguments const &args, void *ws) {
        Params p{Base::to_underlying_arguments(
                     ps, static_cast<typename Base::Arguments const &>(args), ws),
                 {}, args.x_scale};
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
            using IA = typename Base::InternalElementA;
            using IB = typename Base::InternalElementB;
            Tensor sA = as_position_independent_swizzle_tensor(make_tensor(
                make_smem_ptr(reinterpret_cast<IA *>(st.smem_A.data())), SmemLayoutA{}));
            Tensor sB = as_position_independent_swizzle_tensor(make_tensor(
                make_smem_ptr(reinterpret_cast<IB *>(st.smem_B.data())), SmemLayoutB{}));
            Tensor sBs = as_position_independent_swizzle_tensor(make_tensor(
                make_smem_ptr(reinterpret_cast<IB *>(st.smem_Bs.data())), SmemLayoutB{}));
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
    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                            FrgTensorC &accum, int k_tile_count, int thread_idx, TensorStorage &st,
                            Params const &p) {
        Tensor sA = as_position_independent_swizzle_tensor(
            make_tensor(make_smem_ptr(st.smem_A.data()), SmemLayoutA{}));
        Tensor sB = make_tensor(make_smem_ptr(st.smem_B.data()), SmemLayoutB{});
        Tensor sBs = make_tensor(make_smem_ptr(st.smem_Bs.data()), SmemLayoutB{});
        TiledMma tiled_mma;
        const int wg = __shfl_sync(0xFFFFFFFF, thread_idx / 128, 0);
        auto thr = tiled_mma.get_thread_slice(thread_idx);
        auto wgs = tiled_mma.get_slice(make_layout(Int<2>{}, Int<128>{})(wg));
        Tensor tCrA = thr.partition_fragment_A(sA(_, _, Int<0>{})); // (MMA,MMA_M,MMA_K) A_big
        Tensor tCrAs = make_fragment_like(tCrA);                    // A_small
        Tensor tCrAf = make_fragment_like<float>(tCrA);             // A as loaded
        Tensor tCrB = wgs.make_fragment_B(wgs.partition_B(sB));
        Tensor tCrBs = wgs.make_fragment_B(wgs.partition_B(sBs));
        auto cpA = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, float>{}, tiled_mma);
        auto thr_cpA = cpA.get_thread_slice(thread_idx);
        Tensor tCsA_v = thr_cpA.partition_S(sA);
        Tensor tCrA_v = thr_cpA.retile_D(tCrAf);
        Tensor accF = make_fragment_like(accum);
        constexpr int KB = size<2>(tCrA);
        const float xs = p.x_scale;
        auto split = [&](int kb) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCrA(_, _, kb)); ++i)
                split_h(tCrAf(_, _, kb)(i) * xs, tCrA(_, _, kb)(i), tCrAs(_, _, kb)(i));
        };
        PipelineState release = smem_pipe_read;
        bool fresh = true;
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
// C(M x N) = alpha A(M x K) B(N x K)^T, A = X in FP32 and B = V in FP16 parts, both K-major; C is
// W^T, row-major.
struct X3HRSGemm {
    using EA = cutlass::tfloat32_t; // 32-bit storage copied raw by the TMA
    using EB = cutlass::half_t;
    using TS = Shape<_128, _128, _32>;
    using CS = Shape<_1, _1, _1>;
    using LC = cutlass::layout::RowMajor;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, float, LC, 4, float, LC, 4,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
    using Atom = cute::SM90::GMMA::MMA_64x128x16_F32F16F16_RS<GMMA::Major::K, GMMA::Major::K>;
    using TiledMmaRS = decltype(make_tiled_mma(Atom{}, Layout<Shape<_2, _1, _1>>{}));
    using StrideAB = cute::tuple<int64_t, cute::Int<1>, int64_t>;
    using BaseMain = cutlass::gemm::collective::CollectiveMma<
        cutlass::gemm::MainloopSm90TmaGmmaRmemAWarpSpecialized<
            x3rs_stages, CS, cutlass::gemm::KernelTmaWarpSpecializedCooperative>,
        TS, EA, StrideAB, EB, StrideAB, TiledMmaRS, cute::SM90_TMA_LOAD,
        GMMA::Layout_K_SW128_Atom<EA>, cute::Copy_Atom<cute::AutoVectorizingCopy, EA>,
        cute::identity, cute::SM90_TMA_LOAD, GMMA::Layout_K_SW64_Atom<EB>, void, cute::identity>;
    using Main = X3HRSMainloop<BaseMain>;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
// C(M x N) = alpha A(M x K) B(K x N) + C in FP16 with FP32 accumulation: A row-major, B
// column-major (both K-major), C column-major; alpha read from device memory.
struct HGemm {
    using TS = Shape<_128, _128, _64>;
    using CS = Shape<_1, _1, _1>;
    using LC = cutlass::layout::ColumnMajor;
    using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TS, CS,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, float, LC, 4, float, LC, 4,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
    using Main = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, cutlass::half_t,
        cutlass::layout::RowMajor, 8, cutlass::half_t, cutlass::layout::ColumnMajor, 8, float, TS,
        CS,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename Epi::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
} // namespace gmma

// Scale of W's register operand X: 2^13 / 2^e with 2^e >= sqrt(rows) >= |x|.
inline float x3h_x_scale(long long rows) {
    int e = 0;
    while ((1LL << (2 * e)) < rows)
        ++e;
    return std::ldexp(1.f, 13 - e);
}
// TMA needs a 16-byte aligned base and a leading dimension that is a multiple of eight halves.
inline bool gmma_aligned_h(const void *p, long long ld) {
    return (reinterpret_cast<uintptr_t>(p) % 16 == 0) && (ld % 8 == 0);
}
// Leading dimension of the FP16 split V of W.
inline int gmma_vsplit_ld_h(int rows) {
    return (rows + 7) / 8 * 8;
}
// Largest magnitude whose FP16 rounding stays finite is just below 65520; the check keeps a
// margin and also catches NaN.
__device__ __forceinline__ bool fp16_range(float scaled) {
    return fabsf(scaled) < 65504.f;
}
// V (rows x h, col-major ldv) -> big, small FP16 parts of x3h_v_scale V (col-major, ld ldvs). An
// entry outside the FP16 range sets FP16_RANGE in `status` (when given).
__global__ void split_v_h(const float *__restrict__ V, int ldv, int rows, int h,
                          cutlass::half_t *__restrict__ Vb, cutlass::half_t *__restrict__ Vs,
                          int ldvs, int *status) {
    bool bad = false;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)rows * h;
         i += (long long)gridDim.x * blockDim.x) {
        const int r = int(i % rows);
        const long long k = i / rows;
        const float x = V[r + k * ldv] * gmma::x3h_v_scale;
        bad |= !fp16_range(x);
        gmma::split_h(x, Vb[r + k * ldvs], Vs[r + k * ldvs]);
    }
    if (bad && status)
        atomicCAS(status, 0, FP16_RANGE);
}
// VR[r*ldvr + {0,hs,2hs} + k] = {small, big, big} of x3h_v_scale V[r + k*ldv] (zero for k >= h).
__global__ void pack_vr_h(const float *__restrict__ V, int ldv, int rows, int h, int hs,
                          cutlass::half_t *__restrict__ VR, int ldvr, int *status) {
    __shared__ float tile[32][33];
    bool bad = false;
    const int nr = (rows + 31) / 32, nk = (hs + 31) / 32;
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
            if (rr < rows && k < hs) {
                cutlass::half_t b, s;
                const float x = tile[threadIdx.x][threadIdx.y + j] * gmma::x3h_v_scale;
                bad |= !fp16_range(x);
                gmma::split_h(x, b, s);
                cutlass::half_t *o = VR + (long long)rr * ldvr;
                o[k] = s;
                o[hs + k] = b;
                o[2 * hs + k] = b;
            }
        }
        __syncthreads();
    }
    if (bad && status)
        atomicCAS(status, 0, FP16_RANGE);
}
inline void launch_pack_vr_h(const float *v, int ldv, int rows, int h, int hs, cutlass::half_t *vr,
                             int ldvr, cudaStream_t st, int *status = nullptr) {
    const int grid = int(std::min(4096LL, (long long)ceildiv(rows, 32) * ceildiv(hs, 32)));
    pack_vr_h<<<grid, dim3(32, 8), 0, st>>>(v, ldv, rows, h, hs, vr, ldvr, status);
    CU(cudaGetLastError());
}
// max |Z| as the bits of a non-negative float (their order is the float order).
__global__ void abs_max_bits(const float *__restrict__ Z, int ldz, int h, int q,
                             unsigned *__restrict__ out) {
    float m = 0.f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)q * h;
         i += (long long)gridDim.x * blockDim.x)
        m = fmaxf(m, fabsf(Z[i % h + (i / h) * ldz]));
    for (int o = 16; o; o >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    if ((threadIdx.x & 31) == 0)
        atomicMax(out, __float_as_uint(m));
}
// Scale of Z: 2^(14 - e) with max |Z| < 2^e. Non-finite Z flags FP16_RANGE.
__device__ __forceinline__ float x3h_z_scale(unsigned maxbits, int *status) {
    const float m = __uint_as_float(maxbits);
    if (!isfinite(m) && status && blockIdx.x == 0 && threadIdx.x == 0)
        atomicCAS(status, 0, FP16_RANGE);
    if (!(m > 0.f) || !isfinite(m))
        return 1.f;
    int e;
    frexpf(m, &e);
    return ldexpf(1.f, 14 - e);
}
// Z3[n*ld3 + {0,hs,2hs} + k] = {big, small, big} of sz Z[k + n*ldz] (zero for k >= h); alpha =
// -1 / (x3h_v_scale sz).
__global__ void pack_z3_h(const float *__restrict__ Z, int ldz, int h, int hs, int q,
                          cutlass::half_t *__restrict__ Z3, int ld3,
                          const unsigned *__restrict__ zmax, float *__restrict__ alpha,
                          int *status) {
    const float sz = x3h_z_scale(*zmax, status);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *alpha = -1.f / (gmma::x3h_v_scale * sz);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (long long)q * hs;
         i += (long long)gridDim.x * blockDim.x) {
        const long long n = i / hs;
        const int k = int(i % hs);
        cutlass::half_t b, s;
        gmma::split_h(k < h ? Z[k + n * ldz] * sz : 0.f, b, s);
        cutlass::half_t *o = Z3 + n * ld3;
        o[k] = b;
        o[hs + k] = s;
        o[2 * hs + k] = b;
    }
}

namespace gmma_detail {
using GemmDH = gmma::HGemm::Gemm;
using GemmW3H = gmma::X3HRSGemm::Gemm;
} // namespace gmma_detail
inline gmma_detail::GemmDH::Arguments gmma_dh_args(const cutlass::half_t *vr, int ldvr,
                                                   const cutlass::half_t *z, int ldz, float *x,
                                                   int ldx, int rows, int k, int q,
                                                   const float *alpha) {
    using G = gmma_detail::GemmDH;
    typename G::Arguments a;
    a.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {rows, q, k, 1};
    a.mainloop.ptr_A = vr;
    a.mainloop.dA = typename G::GemmKernel::StrideA{int64_t(ldvr), cute::Int<1>{}, int64_t(0)};
    a.mainloop.ptr_B = z;
    a.mainloop.dB = typename G::GemmKernel::StrideB{int64_t(ldz), cute::Int<1>{}, int64_t(0)};
    a.epilogue.thread.alpha_ptr = alpha;
    a.epilogue.thread.beta = 1.f;
    a.epilogue.ptr_C = x;
    a.epilogue.dC = typename G::GemmKernel::StrideC{cute::Int<1>{}, int64_t(ldx), int64_t(0)};
    a.epilogue.ptr_D = x;
    a.epilogue.dD = a.epilogue.dC;
    return a;
}
// X(rows x q) -= V Z from the FP16 segments: VR (rows x k row-major) and Z3 (k x q column-major).
inline bool gmma_dh_admits(const cutlass::half_t *vr, int ldvr, const cutlass::half_t *z3, int ld3,
                           float *x, int ldx, int rows, int k, int q, const float *alpha) {
    return rows && q && gmma_aligned_h(vr, ldvr) && gmma_aligned_h(z3, ld3) &&
           gmma_aligned(x, ldx) &&
           gmma_detail::implementable<gmma_detail::GemmDH>(
               gmma_dh_args(vr, ldvr, z3, ld3, x, ldx, rows, k, q, alpha));
}
inline void launch_gmma_dh(const cutlass::half_t *vr, int ldvr, const cutlass::half_t *z3, int ld3,
                           float *x, int ldx, int rows, int k, int q, const float *alpha,
                           cudaStream_t st, int tiles_per_block) {
    auto a = gmma_dh_args(vr, ldvr, z3, ld3, x, ldx, rows, k, q, alpha);
    gmma_detail::run<gmma_detail::GemmDH>(a, st, tiles_per_block);
}
// Splits Z into the FP16 segments of D: zmax and alpha are two words of device scratch.
inline void pack_z3_fp16(const float *z, int ldz, int h, int hs, int q, cutlass::half_t *z3,
                         int ld3, unsigned *zmax, float *alpha, cudaStream_t st,
                         int *status = nullptr) {
    CU(cudaMemsetAsync(zmax, 0, sizeof(unsigned), st));
    abs_max_bits<<<std::min(1024, ceildiv(h * q, 256)), 256, 0, st>>>(z, ldz, h, q, zmax);
    CU(cudaGetLastError());
    pack_z3_h<<<pack_vr_grid((long long)q * hs), 256, 0, st>>>(z, ldz, h, hs, q, z3, ld3, zmax,
                                                                 alpha, status);
    CU(cudaGetLastError());
}

inline gmma_detail::GemmW3H::Arguments
gmma_w3h_args(const float *x, int ldx, const cutlass::half_t *vb, const cutlass::half_t *vs,
              int ldvs, float *o, int ldw, long long sp, int q, int h, int ks, int L, float beta,
              float x_scale) {
    using G = gmma_detail::GemmW3H;
    using EA = gmma::X3HRSGemm::EA;
    typename G::Arguments a;
    a.mode = L > 1 ? cutlass::gemm::GemmUniversalMode::kBatched
                   : cutlass::gemm::GemmUniversalMode::kGemm;
    a.problem_shape = {q, h, ks, L};
    a.mainloop.ptr_A = reinterpret_cast<const EA *>(x);
    a.mainloop.dA = {int64_t(ldx), cute::Int<1>{}, int64_t(ks)};
    a.mainloop.ptr_B = vb;
    a.mainloop.dB = {int64_t(ldvs), cute::Int<1>{}, int64_t(ks)};
    a.mainloop.ptr_Bs = vs;
    a.mainloop.x_scale = x_scale;
    a.epilogue.thread.alpha = 1.f / (gmma::x3h_v_scale * x_scale);
    a.epilogue.thread.beta = beta;
    a.epilogue.ptr_C = o;
    a.epilogue.dC = {int64_t(ldw), cute::Int<1>{}, int64_t(L > 1 ? sp : 0)};
    a.epilogue.ptr_D = o;
    a.epilogue.dD = a.epilogue.dC;
    return a;
}
// W in 3xFP16 with X split in registers, c GPU-level row slices as in launch_gmma_w3rs; `vsplit`
// holds the FP16 parts of V.
inline bool gmma_w3h_admits(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                            int ldw, int rows, int h, int q, void *vsplit, float x_scale) {
    using G = gmma_detail::GemmW3H;
    if (c < 1 || !h || !q || rows < 32 * c || !vsplit)
        return false;
    const int ldvs = gmma_vsplit_ld_h(rows);
    auto *vb = static_cast<cutlass::half_t *>(vsplit), *vs = vb + size_t(ldvs) * h;
    const long long sp = (long long)ldw * q;
    if (!gmma_aligned(v, ldv) || !gmma_aligned(x, ldx) || !gmma_aligned(w, ldw) ||
        !gmma_aligned_h(vb, ldvs) || !gmma_aligned_h(vs, ldvs) || !gmma_aligned(w, sp))
        return false;
    const int ks = gmma_w_slice(rows, c), rem = rows - c * ks;
    if (!gmma_detail::implementable<G>(
            gmma_w3h_args(x, ldx, vb, vs, ldvs, w, ldw, sp, q, h, ks, c, 0.f, x_scale)))
        return false;
    if (!rem)
        return true;
    const size_t o = size_t(c) * ks;
    return gmma_aligned(x + o, ldx) && gmma_aligned_h(vb + o, ldvs) &&
           gmma_aligned_h(vs + o, ldvs) &&
           gmma_detail::implementable<G>(gmma_w3h_args(x + o, ldx, vb + o, vs + o, ldvs,
                                                       w + size_t(c - 1) * sp, ldw, sp, q, h, rem,
                                                       1, 1.f, x_scale));
}
inline void launch_gmma_w3h(int c, const float *v, int ldv, const float *x, int ldx, float *w,
                            int ldw, int rows, int h, int q, cudaStream_t st, void *vsplit,
                            float x_scale, int tiles_per_block, int *status = nullptr) {
    using G = gmma_detail::GemmW3H;
    const int ldvs = gmma_vsplit_ld_h(rows);
    auto *vb = static_cast<cutlass::half_t *>(vsplit), *vs = vb + size_t(ldvs) * h;
    split_v_h<<<pack_vr_grid((long long)rows * h), 256, 0, st>>>(v, ldv, rows, h, vb, vs, ldvs,
                                                                   status);
    CU(cudaGetLastError());
    const long long sp = (long long)ldw * q;
    const int ks = gmma_w_slice(rows, c), rem = rows - c * ks;
    auto a = gmma_w3h_args(x, ldx, vb, vs, ldvs, w, ldw, sp, q, h, ks, c, 0.f, x_scale);
    gmma_detail::run<G>(a, st, tiles_per_block);
    if (rem) {
        const size_t o = size_t(c) * ks;
        auto r = gmma_w3h_args(x + o, ldx, vb + o, vs + o, ldvs, w + size_t(c - 1) * sp, ldw, sp,
                               q, h, rem, 1, 1.f, x_scale);
        gmma_detail::run<G>(r, st, tiles_per_block);
    }
    if (c > 1)
        gmma_detail::combine_slices(w, sp, c, w, ldw, q, h, st);
}
} // namespace qr_omega
