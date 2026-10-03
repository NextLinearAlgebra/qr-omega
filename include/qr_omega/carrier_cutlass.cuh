#pragma once
// Carriers of W, Z and D built from CUTLASS 2.x mainloops (FP64 DMMA, FP32 FMA, and TF32 or 3xTF32
// tensor-core MMA). The warps of a block form c slices that share one multistage pipeline and split
// every k-tile of BK indices into slices of WK = BK / c; with r = K mod BK the residue tile comes
// first, so slice w owns {k < r : k / WK = w} and {k >= r : ((k - r) mod BK) / WK = w}. After the
// mainloop every slice parks its partial tile in shared memory and the partials are added in the
// order w = 0..c-1 before the tile is committed once. W and Z can also span the blocks of a cluster
// (each owns a contiguous range of K, combined through distributed shared memory) and groups of
// blocks of the GPU (their partial tiles summed in HBM in a fixed order).
#include <cutlass/cutlass.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/mma_multistage.h>
#include "group_mma_multistage.h"
#include "kernels.cuh"
#include <cooperative_groups.h>
#include <type_traits>
namespace tqr {
template <class M> struct CgToGroup;
template <class S, class IA, class SIA, cutlass::arch::CacheOperation::Kind CA, class IB, class SIB,
          cutlass::arch::CacheOperation::Kind CBk, class EC, class LC, class P, int St,
          cutlass::gemm::SharedMemoryClearOption SC, class En>
struct CgToGroup<cutlass::gemm::threadblock::MmaMultistage<S, IA, SIA, CA, IB, SIB, CBk, EC, LC, P,
                                                           St, SC, En>> {
    using type = p2c::GroupMmaMultistage<S, IA, SIA, CA, IB, SIB, CBk, EC, LC, P, St, SC, En>;
};

// Arithmetic of a carrier: FP64 DMMA, FP32 FMA on CUDA cores, or FP32 storage on TF32 tensor cores
// with one TF32 product (MathTF32) or three (MathX3: a = big + small, big big + big small +
// small big, accumulated in FP32).
struct MathTF32 {};
struct MathX3 {};
template <class T> struct CarrierOp;
template <> struct CarrierOp<double> {
    using Element = double;
    using Class = cutlass::arch::OpClassTensorOp;
    using Arch = cutlass::arch::Sm90;
    using Inst = cutlass::gemm::GemmShape<16, 8, 4>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static constexpr int Pad = 8;
};
template <> struct CarrierOp<float> {
    using Element = float;
    using Class = cutlass::arch::OpClassSimt;
    using Arch = cutlass::arch::Sm80;
    using Inst = cutlass::gemm::GemmShape<1, 1, 1>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static constexpr int Pad = 4;
};
template <> struct CarrierOp<MathTF32> {
    using Element = float;
    using Class = cutlass::arch::OpClassTensorOp;
    using Arch = cutlass::arch::Sm80;
    using Inst = cutlass::gemm::GemmShape<16, 8, 8>;
    using Operator = cutlass::arch::OpMultiplyAdd;
    static constexpr int Pad = 4;
};
template <> struct CarrierOp<MathX3> {
    using Element = float;
    using Class = cutlass::arch::OpClassTensorOp;
    using Arch = cutlass::arch::Sm80;
    using Inst = cutlass::gemm::GemmShape<16, 8, 8>;
    using Operator = cutlass::arch::OpMultiplyAddFastF32;
    static constexpr int Pad = 4;
};
template <class T, int BM_, int BN_, int BK_, int WM_, int WN_, int WK_, int Stages_, int MinB_>
struct CarrierDCfg {
    using Element = typename CarrierOp<T>::Element;
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WK = WK_, Stages = Stages_, MinB = MinB_;
    using DM = cutlass::gemm::threadblock::DefaultMma<
        Element, cutlass::layout::ColumnMajor, 1, Element, cutlass::layout::RowMajor, 1, Element,
        cutlass::layout::RowMajor, typename CarrierOp<T>::Class, typename CarrierOp<T>::Arch,
        cutlass::gemm::GemmShape<BM, BN, BK>, cutlass::gemm::GemmShape<WM_, WN_, WK_>,
        typename CarrierOp<T>::Inst, Stages, typename CarrierOp<T>::Operator, false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using Mma = typename CgToGroup<typename DM::ThreadblockMma>::type;
    using WarpCount = typename Mma::Base::WarpCount;
    static constexpr int C = WarpCount::kK; // block-level K peers
    static constexpr int WMN = WarpCount::kM * WarpCount::kN;
    static constexpr int SliceThreads = WMN * 32, Threads = SliceThreads * C;
    // Parked tile rows padded so fragment row stores hit distinct banks (fp64: 64 B mod 128 B).
    static constexpr int LDP = BN + CarrierOp<T>::Pad;
    static constexpr size_t PartElems = size_t(BM) * LDP;
    static constexpr size_t MainSmem = sizeof(typename Mma::SharedStorage);
    static constexpr size_t PartSmem = size_t(C) * PartElems * sizeof(Element);
    static constexpr size_t Smem = MainSmem > PartSmem ? MainSmem : PartSmem;
    static constexpr int PerThread = BM * BN / Threads, U = PerThread < 8 ? PerThread : 8;
    static_assert(BM * BN % Threads == 0 && PerThread % U == 0, "");
    static_assert(WK * C == BK, "slices tile each k-tile exactly");
};
// D splits every k-tile of BK reflectors between two slices of WK = BK / 2.
using CarrierD2 = CarrierDCfg<double, 64, 64, 16, 32, 32, 8, 3, 2>;
using CarrierS2s = CarrierDCfg<float, 64, 64, 16, 32, 32, 8, 3, 2>;
using CarrierT2 = CarrierDCfg<MathTF32, 128, 128, 32, 64, 64, 16, 3, 1>;
using CarrierX2 = CarrierDCfg<MathX3, 128, 64, 32, 64, 32, 16, 3, 1>;

// Column tiles per raster group once Z^T no longer fits in half the L2.
inline constexpr int carrier_d_raster = 16;
template <class Cfg>
__global__ void __launch_bounds__(Cfg::Threads, Cfg::MinB)
    carrier_d_kernel(const typename Cfg::Element *__restrict__ V, int ldv,
                     const typename Cfg::Element *__restrict__ Zt, int ldzt,
                     typename Cfg::Element *X, int ldx, int rows, int h, int q, int raster) {
    using T = typename Cfg::Element;
    extern __shared__ __align__(128) char carrier_d_smem[];
    char *smem = carrier_d_smem;
    using Mma = typename Cfg::Mma;
    constexpr int BM = Cfg::BM, BN = Cfg::BN;
    const int tid = threadIdx.x + Cfg::SliceThreads * threadIdx.y; // threadIdx.y = K peer (slice)
    // `raster` column tiles per group, or all of them when Z^T fits in half the L2.
    int tcol, trow;
    raster_tile(blockIdx.x + gridDim.x * blockIdx.y, gridDim.x, gridDim.y, raster, tcol, trow);
    const int n0 = tcol * BM, m0 = trow * BN; // X columns / X rows of the tile
    {                                         // X tile -> L2 while the mainloop runs
        const int rr = min(BN, rows - m0), cc = min(BM, q - n0);
        constexpr int LPC = (BN * int(sizeof(T)) + 127) / 128, RPL = 128 / int(sizeof(T));
        for (int l = tid; l < cc * LPC; l += Cfg::Threads) {
            const int col = l / LPC, r = (l % LPC) * RPL;
            if (r < rr)
                asm volatile(
                    "prefetch.global.L2 [%0];" ::"l"(X + (m0 + r) + (long long)(n0 + col) * ldx));
        }
    }
    typename Mma::IteratorA itA(typename Mma::IteratorA::Params(cutlass::layout::ColumnMajor(ldzt)),
                                const_cast<T *>(Zt), {q, h}, tid, {n0, 0});
    typename Mma::IteratorB itB(typename Mma::IteratorB::Params(cutlass::layout::RowMajor(ldv)),
                                const_cast<T *>(V), {h, rows}, tid, {0, m0});
    const int iters = (h + Cfg::BK - 1) / Cfg::BK;
    const int warp = __shfl_sync(0xffffffffu, tid / 32, 0), lane = threadIdx.x % 32;
    typename Mma::FragmentC acc;
    acc.clear();
    {
        typename Mma::SharedStorage &ss = *reinterpret_cast<typename Mma::SharedStorage *>(smem);
        Mma mma(ss, tid, warp, lane, 0);
        if (iters > 0)
            mma(iters, acc, itA, itB, acc);
    }
    __syncthreads(); // pipeline SMEM is dead
    T *part = reinterpret_cast<T *>(smem);
    {
        const int wmn = warp % Cfg::WMN, wk = warp / Cfg::WMN; // wk == threadIdx.y
        typename Mma::Operator::IteratorC itC(
            {part + size_t(wk) * Cfg::PartElems, cutlass::layout::RowMajor(Cfg::LDP)}, lane);
        itC.add_tile_offset({wmn % Cfg::WarpCount::kM, wmn / Cfg::WarpCount::kM});
        itC.store(acc);
    }
    __syncthreads();
    const bool full = (m0 + BN <= rows) && (n0 + BM <= q);
#pragma unroll 1
    for (int b0 = 0; b0 < Cfg::PerThread; b0 += Cfg::U) {
        T xv[Cfg::U];
        int off[Cfg::U];
        long long gx[Cfg::U];
        bool in[Cfg::U];
#pragma unroll
        for (int u = 0; u < Cfg::U; ++u) {
            const int e = tid + (b0 + u) * Cfg::Threads, i = e / BN,
                      j = e % BN; // i: tile column (X column), j: X row
            off[u] = i * Cfg::LDP + j;
            gx[u] = (m0 + j) + (long long)(n0 + i) * ldx;
            in[u] = full || ((n0 + i) < q && (m0 + j) < rows);
            xv[u] = in[u] ? X[gx[u]] : T(0);
        }
#pragma unroll
        for (int u = 0; u < Cfg::U; ++u) {
            T s = 0;
#pragma unroll
            for (int w = 0; w < Cfg::C; ++w)
                s += part[w * Cfg::PartElems + off[u]]; // fixed order w = 0..c-1
            xv[u] -= s;
        }
#pragma unroll
        for (int u = 0; u < Cfg::U; ++u)
            if (in[u])
                X[gx[u]] = xv[u]; // owner commits once
    }
}
// k-tile and slice widths (BK, WK) of the D carrier of the arithmetic in use.
template <class T> void carrier_d_law(int &bk, int &wk) {
    const bool wide = !std::is_same_v<T, double> && fp32_math() != Fp32Math::IEEE;
    bk = wide ? CarrierT2::BK : CarrierD2::BK;
    wk = wide ? CarrierT2::WK : CarrierD2::WK;
}
// Both slices own reflectors when h covers the first slice of the residue k-tile.
template <class T> bool carrier_d_supported(int h) {
    int bk, wk;
    carrier_d_law<T>(bk, wk);
    return h >= bk - wk + 1;
}
template <class Cfg>
void launch_carrier_d_cfg(const typename Cfg::Element *v, int ldv, const typename Cfg::Element *zt,
                          int ldzt, typename Cfg::Element *x, int ldx, int rows, int h, int q,
                          cudaStream_t st) {
    reserve_shared_memory(carrier_d_kernel<Cfg>, Cfg::Smem, true);
    const int nx = ceildiv(q, Cfg::BM);
    const bool spills =
        size_t(q) * h * sizeof(typename Cfg::Element) * 2 > size_t(device_properties().l2CacheSize);
    carrier_d_kernel<Cfg>
        <<<dim3(nx, ceildiv(rows, Cfg::BN)), dim3(Cfg::SliceThreads, Cfg::C), Cfg::Smem, st>>>(
            v, ldv, zt, ldzt, x, ldx, rows, h, q, spills ? carrier_d_raster : nx);
    CU(cudaGetLastError());
}
// X(rows x q) -= V Z with Z given transposed (q x h), on the carrier of the arithmetic in use.
template <class T>
void launch_carrier_d(const T *v, int ldv, const T *zt, int ldzt, T *x, int ldx, int rows, int h,
                      int q, cudaStream_t st) {
    if (!carrier_d_supported<T>(h))
        throw std::runtime_error("too few reflectors for the two slices of D");
    if (!rows || !q)
        return;
    if constexpr (std::is_same_v<T, double>)
        launch_carrier_d_cfg<CarrierD2>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st);
    else if (fp32_math() == Fp32Math::TF32)
        launch_carrier_d_cfg<CarrierT2>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st);
    else if (fp32_math() == Fp32Math::X3)
        launch_carrier_d_cfg<CarrierX2>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st);
    else
        launch_carrier_d_cfg<CarrierS2s>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st);
}

// W and Z: Out = A B in CUTLASS's transposed frame Out^T = B^T A^T (layouts LA, LB). CL blocks of a
// cluster split K into contiguous ranges, SK slices split every k-tile, and block z sums its rows
// of the tile over the pieces (z', w) in the order z' = 0..CL-1, w = 0..SK-1 through distributed
// shared memory before it stores them: c = CL SK.
template <class T, class LA_, class LB_, int BM_, int BN_, int BK_, int WM_, int WN_, int WK_,
          int Stages_, int MinB_, class Instruction_ = typename CarrierOp<T>::Inst>
struct CarrierGCfg {
    using Element = typename CarrierOp<T>::Element;
    using LA = LA_;
    using LB = LB_;
    static constexpr int BM = BM_, BN = BN_, BK = BK_, WK = WK_, Stages = Stages_, MinB = MinB_;
    using DM = cutlass::gemm::threadblock::DefaultMma<
        Element, LA, 1, Element, LB, 1, Element, cutlass::layout::RowMajor,
        typename CarrierOp<T>::Class, typename CarrierOp<T>::Arch,
        cutlass::gemm::GemmShape<BM, BN, BK>, cutlass::gemm::GemmShape<WM_, WN_, WK_>, Instruction_,
        Stages, typename CarrierOp<T>::Operator, false,
        cutlass::gemm::SharedMemoryClearOption::kNone>;
    using Mma = typename CgToGroup<typename DM::ThreadblockMma>::type;
    using WarpCount = typename Mma::Base::WarpCount;
    static constexpr int SK = WarpCount::kK, WMN = WarpCount::kM * WarpCount::kN;
    static constexpr int SliceThreads = WMN * 32, Threads = SliceThreads * SK;
    static constexpr int LDP = BN + CarrierOp<T>::Pad;
    static constexpr size_t PartElems = size_t(BM) * LDP;
    static constexpr size_t MainSmem = sizeof(typename Mma::SharedStorage);
    static constexpr size_t PartSmem = size_t(SK) * PartElems * sizeof(Element);
    static constexpr size_t Smem = MainSmem > PartSmem ? MainSmem : PartSmem;
    static_assert(WK * SK == BK, "slices tile each k-tile exactly");
};
// CG > 1 groups of blocks split K again (peer g CL + z owns [K pid / (CG CL), K (pid + 1) / (CG
// CL))); group g writes its partial to the slice P + g sp, and carrier_g_combine adds the slices in
// the order g = 0..CG-1 and commits O once.
template <class T> struct CarrierGArgs {
    const T *A;
    int lda;
    const T *B;
    int ldb;
    T *O;
    int ldo;
    int M, N, K;
    int CG = 1;
    T *P = nullptr;
    long long sp = 0;
};
template <class Cfg, int CL>
__global__ void __launch_bounds__(Cfg::Threads, Cfg::MinB)
    carrier_g_kernel(const __grid_constant__ CarrierGArgs<typename Cfg::Element> a) {
    using T = typename Cfg::Element;
    using Mma = typename Cfg::Mma;
    constexpr int BM = Cfg::BM, BN = Cfg::BN;
    static_assert(BM % CL == 0, "cluster row slices");
    extern __shared__ __align__(128) char carrier_g_smem[];
    char *smem = carrier_g_smem;
    const int tid = threadIdx.x + Cfg::SliceThreads * threadIdx.y;
    int z = 0;
    if constexpr (CL > 1)
        z = int(cooperative_groups::this_cluster().block_rank());
    // The N tiles of an M tile run back to back and share the strip of X in L2.
    const int linear = blockIdx.x + gridDim.x * blockIdx.y, tm = linear / gridDim.y,
              tn = linear % gridDim.y;
    const int gz = blockIdx.z / CL, peers = CL * a.CG, pid = gz * CL + z, m0 = tm * BM,
              n0 = tn * BN;
    const T *Ap = a.A, *Bp = a.B;
    T *O = a.CG > 1 ? a.P + gz * a.sp : a.O;
    const int k0 = int((long long)a.K * pid / peers), k1 = int((long long)a.K * (pid + 1) / peers);
    typename Mma::IteratorA itA(typename Mma::IteratorA::Params(typename Cfg::LA(a.lda)),
                                const_cast<T *>(Ap), {a.M, k1}, tid, {m0, k0});
    typename Mma::IteratorB itB(typename Mma::IteratorB::Params(typename Cfg::LB(a.ldb)),
                                const_cast<T *>(Bp), {k1, a.N}, tid, {k0, n0});
    const int iters = (k1 - k0 + Cfg::BK - 1) / Cfg::BK;
    const int warp = __shfl_sync(0xffffffffu, tid / 32, 0), lane = threadIdx.x % 32;
    typename Mma::FragmentC acc;
    acc.clear();
    {
        auto &ss = *reinterpret_cast<typename Mma::SharedStorage *>(smem);
        Mma mma(ss, tid, warp, lane, 0);
        if (iters > 0)
            mma(iters, acc, itA, itB, acc);
    }
    __syncthreads();
    T *part = reinterpret_cast<T *>(smem);
    {
        const int wmn = warp % Cfg::WMN, wk = warp / Cfg::WMN;
        typename Mma::Operator::IteratorC itC(
            {part + size_t(wk) * Cfg::PartElems, cutlass::layout::RowMajor(Cfg::LDP)}, lane);
        itC.add_tile_offset({wmn % Cfg::WarpCount::kM, wmn / Cfg::WarpCount::kM});
        itC.store(acc);
    }
    if constexpr (CL > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else
        __syncthreads();
    constexpr int SL = BM / CL;
    const int r0 = z * SL;
    const bool full = (m0 + BM <= a.M) && (n0 + BN <= a.N);
    for (int e = tid; e < SL * BN; e += Cfg::Threads) {
        const int i = r0 + e / BN, j = e % BN;
        const long long go = (n0 + j) + (long long)(m0 + i) * a.ldo;
        const bool in = full || ((m0 + i) < a.M && (n0 + j) < a.N);
        T s = 0;
#pragma unroll
        for (int r = 0; r < CL; ++r) { // fixed order z'=0..CL-1, w=0..SK-1
            const T *peer = part;
            if constexpr (CL > 1)
                peer = cooperative_groups::this_cluster().map_shared_rank(part, r);
#pragma unroll
            for (int w = 0; w < Cfg::SK; ++w)
                s += peer[w * Cfg::PartElems + i * Cfg::LDP + j];
        }
        if (in)
            O[go] = s; // the owner commits once (or its slice)
    }
    if constexpr (CL > 1) {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    }
}
// GPU-level fixed-order combine of CG partial slices sp apart: O = sum over g = 0..CG-1 of P_g, for
// `lines` lines of `width` elements, ldo apart. Blocks walk the lines, threads the elements.
template <class T>
__global__ void carrier_g_combine(const T *P, long long sp, int CG, T *O, int ldo, int lines,
                                  int width) {
    for (int i = blockIdx.x; i < lines; i += gridDim.x) {
        const long long base = (long long)i * ldo;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            const T *src = P + base + j;
            T s = 0;
            for (int g = 0; g < CG; ++g)
                s += src[size_t(g) * sp];
            O[base + j] = s;
        }
    }
}
template <class Cfg, int CL>
void launch_carrier_g(const CarrierGArgs<typename Cfg::Element> &a, cudaStream_t st) {
    if (!a.M || !a.N)
        return;
    if (a.CG > 1 && (!a.P || a.sp < (long long)a.M * a.ldo))
        throw std::runtime_error("no room for the partial slices of the GPU groups");
    reserve_shared_memory(carrier_g_kernel<Cfg, CL>, Cfg::Smem, true);
    launch_clustered(carrier_g_kernel<Cfg, CL>,
                     dim3(ceildiv(a.M, Cfg::BM), ceildiv(a.N, Cfg::BN), CL * a.CG),
                     dim3(Cfg::SliceThreads, Cfg::SK), Cfg::Smem, st, dim3(1, 1, CL), a);
    if (a.CG > 1) {
        carrier_g_combine<typename Cfg::Element>
            <<<std::min(1024, ceildiv(a.M * a.N, 256)), 256, 0, st>>>(a.P, a.sp, a.CG, a.O, a.ldo,
                                                                      a.M, a.N);
        CU(cudaGetLastError());
    }
}
using CRM = cutlass::layout::RowMajor;
using CCM = cutlass::layout::ColumnMajor;
// The carrier of the arithmetic in use, as a tag: double, MathTF32 or MathX3. IEEE FP32 computes W
// and Z with cuBLAS peers and has none.
template <class T, class F> decltype(auto) with_carrier_policy(F &&f) {
    if constexpr (std::is_same_v<T, double>)
        return f(double{});
    else {
        if (fp32_math() == Fp32Math::IEEE)
            throw std::logic_error("IEEE FP32 has no CUTLASS carrier for W and Z");
        return fp32_math() == Fp32Math::TF32 ? f(MathTF32{}) : f(MathX3{});
    }
}
// W = V^T X in the transposed frame W^T = X^T V. FP64 tiles the output 64 x 128 (the whole width of
// a panel) once the tiles fill the GPU, else 64 x 64; the tensor-core modes always use 64 x 64.
template <class P> struct CarrierW {
    using Big = CarrierGCfg<P, CRM, CCM, 64, 64, 32, 32, 32, 16, 3, 1>;
    using Small = Big;
};
template <> struct CarrierW<double> {
    using Big = CarrierGCfg<double, CRM, CCM, 64, 128, 32, 32, 64, 16, 3, 1>;
    using Small = CarrierGCfg<double, CRM, CCM, 64, 64, 32, 32, 32, 16, 3, 2>;
};
constexpr int carrier_w_sk = 2;
inline bool carrier_w_supported(int c, int rows) {
    return (c == 2 || c == 4 || c == 8) && rows >= (c / carrier_w_sk) * 32;
}
// The 64 x 128 tile once it gives about 96 blocks, else 64 x 64 for more of them.
inline bool carrier_w_big(int c, int q) {
    return (long long)ceildiv(q, CarrierW<double>::Big::BM) * (c / carrier_w_sk) >= 96;
}
template <class T>
void launch_carrier_w(int c, const T *v, int ldv, const T *x, int ldx, T *w, int ldw, int rows,
                      int h, int q, cudaStream_t st, int cg = 1, T *part = nullptr) {
    if (!carrier_w_supported(c, rows))
        throw std::runtime_error("the W carrier needs c = 2, 4 or 8 and 32 rows per block");
    // cg > 1 groups of blocks split the rows, with partial slices of q ldw words in `part`.
    const CarrierGArgs<T> a{x, ldx, v, ldv, w, ldw, q, h, rows, cg, part, (long long)q * ldw};
    auto go = [&](auto cfg) {
        using Cfg = decltype(cfg);
        if (c == 2)
            launch_carrier_g<Cfg, 1>(a, st);
        else if (c == 4)
            launch_carrier_g<Cfg, 2>(a, st);
        else
            launch_carrier_g<Cfg, 4>(a, st);
    };
    with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        if (carrier_w_big(c, q))
            go(typename CarrierW<P>::Big{});
        else
            go(typename CarrierW<P>::Small{});
    });
}
// Groups of blocks for a small W with many rows (a Gram block of T): enough blocks for two waves,
// at least 256 rows per peer, at most 128 groups.
template <class T> int carrier_w_gpu_split(int c, int q, int h, int rows) {
    const int cl = c / carrier_w_sk;
    int bm = 64, bn = 64;
    with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        if (carrier_w_big(c, q)) {
            bm = CarrierW<P>::Big::BM;
            bn = CarrierW<P>::Big::BN;
        } else {
            bm = CarrierW<P>::Small::BM;
            bn = CarrierW<P>::Small::BN;
        }
    });
    const long long tiles = (long long)ceildiv(q, bm) * ceildiv(h, bn) * cl;
    long long cg = std::max(1LL, 2LL * device_properties().multiProcessorCount / tiles);
    cg = std::min<long long>({cg, 128, std::max(1, rows / (256 * cl))});
    return int(std::max(1LL, cg));
}
// Z = op(T) W (K = h), or Z^T = W^T op(T)^T for the D carriers, which read Z transposed. FP64
// splits every k-tile of 8 SK indices between SK slices of 8; the tensor-core modes keep two slices
// of 16 (a 32-bit operand transposed through shared memory holds one 128-byte line per k-tile) and
// reach c = 4 with two blocks of a cluster.
template <class P, class LA, class LB, int SK> struct CarrierZSel;
template <class LA, class LB, int SK> struct CarrierZSel<double, LA, LB, SK> {
    using type = CarrierGCfg<double, LA, LB, 64, 64, 8 * SK, 32, 32, 8, 3, (SK == 2 ? 2 : 1)>;
};
template <class LA, class LB, int SK> struct CarrierZSel<MathTF32, LA, LB, SK> {
    using type = CarrierGCfg<MathTF32, LA, LB, 64, 64, 32, 32, 32, 16, 3, 1>;
};
template <class LA, class LB, int SK> struct CarrierZSel<MathX3, LA, LB, SK> {
    using type = CarrierGCfg<MathX3, LA, LB, 64, 64, 32, 32, 32, 16, 3, 1>;
};
template <class P, class LA, class LB, int SK>
using CarrierZCfg = typename CarrierZSel<P, LA, LB, SK>::type;
template <class T> void carrier_z_law(int c, int &bk, int &wk) {
    with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        if (c == 2) {
            bk = CarrierZCfg<P, CRM, CCM, 2>::BK;
            wk = CarrierZCfg<P, CRM, CCM, 2>::WK;
        } else {
            bk = CarrierZCfg<P, CRM, CCM, 4>::BK;
            wk = CarrierZCfg<P, CRM, CCM, 4>::WK;
        }
    });
}
// Slices SK of a block and blocks CL of a cluster with SK CL = c.
template <class T> void carrier_z_split(int c, int &sk, int &cl) {
    sk = c;
    cl = 1;
    if constexpr (std::is_same_v<T, float>) {
        if (fp32_math() != Fp32Math::IEEE) {
            sk = 2;
            cl = c / 2;
        }
    }
}
// Every piece non-empty: each cluster CTA's range floor(h/CL) >= BK - WK + 1 (residue tile first).
template <class T> bool carrier_z_supported(int c, int h) {
    if (c != 2 && c != 4)
        return false;
    int bk, wk, sk, cl;
    carrier_z_law<T>(c, bk, wk);
    carrier_z_split<T>(c, sk, cl);
    return h / cl >= bk - wk + 1;
}
template <class T>
void launch_carrier_z(int c, const T *t, int ldt, const T *w, int ldw, T *z, int ldz, int h, int q,
                      bool transpose, bool zt_out, cudaStream_t st) {
    if (!carrier_z_supported<T>(c, h))
        throw std::runtime_error("the Z carrier needs c = 2 or 4 and a reflector per slice");
    with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        int zsk, zcl;
        carrier_z_split<T>(c, zsk, zcl);
        auto go = [&](auto sk, auto la, auto lb, const T *A, int lda, const T *B, int ldb, int M,
                      int N) {
            using Cfg = CarrierZCfg<P, decltype(la), decltype(lb), decltype(sk)::value>;
            const CarrierGArgs<T> a{A, lda, B, ldb, z, ldz, M, N, h};
            if (zcl == 2)
                launch_carrier_g<Cfg, 2>(a, st);
            else
                launch_carrier_g<Cfg, 1>(a, st);
        };
        auto dispatch = [&](auto sk) {
            if (zt_out) {
                if (transpose)
                    go(sk, CRM{}, CCM{}, t, ldt, w, ldw, h, q);
                else
                    go(sk, CCM{}, CCM{}, t, ldt, w, ldw, h, q);
            } else {
                if (transpose)
                    go(sk, CRM{}, CCM{}, w, ldw, t, ldt, q, h);
                else
                    go(sk, CRM{}, CRM{}, w, ldw, t, ldt, q, h);
            }
        };
        if (c == 2)
            dispatch(std::integral_constant<int, 2>{});
        else
            dispatch(std::integral_constant<int, 4>{});
    });
}
} // namespace tqr
