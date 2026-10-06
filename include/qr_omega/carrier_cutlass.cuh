#pragma once
// Carriers of W, Z and D built from CUTLASS 2.x mainloops (FP64 DMMA, FP32 FMA, and TF32 or 3xTF32
// tensor-core MMA). The warps of a block form a 2 x 2 grid over the output tile and two slices that
// share one multistage pipeline and split every k-tile of BK indices into slices of WK = BK / 2:
// the block is the carrier (2, 2, 2). With r = K mod BK the residue tile comes first, so slice w
// owns {k < r : k / WK = w} and {k >= r : ((k - r) mod BK) / WK = w}. After the mainloop every
// slice parks its partial tile in shared memory and the partials are added in the order w = 0, 1
// before the tile is committed once. W can also split K across groups of blocks of the GPU, whose
// partial tiles are summed in HBM in a fixed order; the number of groups CG keeps the GPU-level
// carrier (tiles_i, tiles_j, CG) within the 2.5D bound CG^2 <= tiles_i tiles_j.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/mma_multistage.h>
#include "group_mma_multistage.h"
#include "carrier.hpp"
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
// The rows of X as the D carrier walks them. Identity, or the h-row blocks of the children of TT
// eliminations: row v lies in block w / inner of node n = v / outer (w = v mod outer), at
// n outer_stride + (w / inner) inner_stride + w mod inner.
struct RowMap {
    int inner = 0, outer = 0;
    long long inner_stride = 0, outer_stride = 0;
    __host__ __device__ long long operator()(int v) const {
        if (!inner)
            return v;
        const int n = v / outer, w = v % outer;
        return n * outer_stride + (w / inner) * inner_stride + w % inner;
    }
};
template <class Cfg>
__global__ void __launch_bounds__(Cfg::Threads, Cfg::MinB)
    carrier_d_kernel(const typename Cfg::Element *__restrict__ V, int ldv,
                     const typename Cfg::Element *__restrict__ Zt, int ldzt,
                     typename Cfg::Element *X, int ldx, int rows, int h, int q, int raster,
                     int seg_rows, int segs, long long zt_seg, RowMap map) {
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
    if (seg_rows) // the Z of the segment that holds the tile (the last one may be longer)
        Zt += min(m0 / seg_rows, segs - 1) * zt_seg;
    { // X tile -> L2 while the mainloop runs
        const int rr = min(BN, rows - m0), cc = min(BM, q - n0);
        constexpr int LPC = (BN * int(sizeof(T)) + 127) / 128, RPL = 128 / int(sizeof(T));
        for (int l = tid; l < cc * LPC; l += Cfg::Threads) {
            const int col = l / LPC, r = (l % LPC) * RPL;
            if (r < rr)
                asm volatile("prefetch.global.L2 [%0];" ::"l"(X + map(m0 + r) +
                                                              (long long)(n0 + col) * ldx));
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
            gx[u] = map(m0 + j) + (long long)(n0 + i) * ldx;
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
// Rows of X in a row tile of the D carrier of the arithmetic in use: segments of D hold whole tiles.
template <class T> int carrier_d_row_tile() {
    if constexpr (std::is_same_v<T, double>)
        return CarrierD2::BN;
    else
        return fp32_math() == Fp32Math::TF32 ? CarrierT2::BN
               : fp32_math() == Fp32Math::X3 ? CarrierX2::BN
                                             : CarrierS2s::BN;
}
// Both slices own reflectors when h covers the first slice of the residue k-tile.
template <class T> bool carrier_d_supported(int h) {
    int bk, wk;
    carrier_d_law<T>(bk, wk);
    return h >= bk - wk + 1;
}
// D in its own frame X(rows x q): block tiles BN x BM, the warps kN x kM over them and C slices.
template <class Cfg> Carrier carrier_d_cfg_carrier(int rows, int q) {
    return Carrier{}
        .set(GpuLevel, ceildiv(rows, Cfg::BN), ceildiv(q, Cfg::BM), 1)
        .set(BlockLevel, Cfg::WarpCount::kN, Cfg::WarpCount::kM, Cfg::C);
}
template <class Cfg>
Carrier launch_carrier_d_cfg(const typename Cfg::Element *v, int ldv,
                             const typename Cfg::Element *zt, int ldzt, typename Cfg::Element *x,
                             int ldx, int rows, int h, int q, cudaStream_t st, int seg_rows = 0,
                             int segs = 1, long long zt_seg = 0, RowMap map = {}) {
    if (seg_rows % Cfg::BN)
        throw std::runtime_error("segments of D must hold whole row tiles");
    reserve_shared_memory(carrier_d_kernel<Cfg>, Cfg::Smem, true);
    const int nx = ceildiv(q, Cfg::BM);
    const bool spills =
        size_t(q) * h * sizeof(typename Cfg::Element) * 2 > size_t(device_properties().l2CacheSize);
    carrier_d_kernel<Cfg>
        <<<dim3(nx, ceildiv(rows, Cfg::BN)), dim3(Cfg::SliceThreads, Cfg::C), Cfg::Smem, st>>>(
            v, ldv, zt, ldzt, x, ldx, rows, h, q, spills ? carrier_d_raster : nx, seg_rows, segs,
            zt_seg, map);
    CU(cudaGetLastError());
    return carrier_d_cfg_carrier<Cfg>(rows, q);
}
// X(rows x q) -= V Z with Z given transposed (q x h), on the carrier of the arithmetic in use.
// With seg_rows, every seg_rows rows of V and X form a segment with its own Z^T, zt_seg elements
// after the previous one, the last of `segs` taking the remaining rows; `map` places the rows of X.
template <class T>
Carrier launch_carrier_d(const T *v, int ldv, const T *zt, int ldzt, T *x, int ldx, int rows, int h,
                         int q, cudaStream_t st, int seg_rows = 0, int segs = 1,
                         long long zt_seg = 0, RowMap map = {}) {
    if (!carrier_d_supported<T>(h))
        throw std::runtime_error("too few reflectors for the two slices of D");
    if constexpr (std::is_same_v<T, double>) {
        if (!rows || !q)
            return carrier_d_cfg_carrier<CarrierD2>(rows, q);
        return launch_carrier_d_cfg<CarrierD2>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st, seg_rows,
                                               segs, zt_seg, map);
    } else {
        auto run = [&](auto cfg) {
            using Cfg = decltype(cfg);
            if (!rows || !q)
                return carrier_d_cfg_carrier<Cfg>(rows, q);
            return launch_carrier_d_cfg<Cfg>(v, ldv, zt, ldzt, x, ldx, rows, h, q, st, seg_rows,
                                             segs, zt_seg, map);
        };
        if (fp32_math() == Fp32Math::TF32)
            return run(CarrierT2{});
        if (fp32_math() == Fp32Math::X3)
            return run(CarrierX2{});
        return run(CarrierS2s{});
    }
}

// W and Z: Out = A B in CUTLASS's transposed frame Out^T = B^T A^T (layouts LA, LB). SK slices of
// the block split every k-tile, and block z sums the SK partials of its tile in the order
// w = 0..SK-1 before it stores the tile.
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
// CG > 1 groups of blocks split K again (group g owns [K g / CG, K (g + 1) / CG)); group g writes
// its partial to the slice P + g sp, and carrier_g_combine adds the slices in the order
// g = 0..CG-1 and commits O once. S > 1 segments are independent products side by side, such as
// the eliminations of the domains of a GPU: segment s reads A and B a_seg and b_seg elements
// further and writes its output o_seg elements further (its partials take the slices s CG..);
// k_last, when nonzero, is the K of a shorter last segment. With a_grp >= 0 the contraction of a
// segment is instead `blocks` blocks of K indices each, a_grp and b_grp elements apart in A and B
// (the children of a TT elimination, whose rows lie apart), dealt to the CG groups in runs of
// ceil(blocks / CG); the last segment has blocks_last of them.
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
    int S = 1;
    long long a_seg = 0, b_seg = 0, o_seg = 0;
    int k_last = 0;
    long long a_grp = -1, b_grp = 0;
    int blocks = 1, blocks_last = 1;
};
template <class Cfg>
__global__ void __launch_bounds__(Cfg::Threads, Cfg::MinB)
    carrier_g_kernel(const __grid_constant__ CarrierGArgs<typename Cfg::Element> a) {
    using T = typename Cfg::Element;
    using Mma = typename Cfg::Mma;
    constexpr int BM = Cfg::BM, BN = Cfg::BN;
    extern __shared__ __align__(128) char carrier_g_smem[];
    char *smem = carrier_g_smem;
    const int tid = threadIdx.x + Cfg::SliceThreads * threadIdx.y;
    // The N tiles of an M tile run back to back and share the strip of X in L2.
    const int linear = blockIdx.x + gridDim.x * blockIdx.y, tm = linear / gridDim.y,
              tn = linear % gridDim.y;
    const int seg = blockIdx.z / a.CG, gz = blockIdx.z % a.CG, m0 = tm * BM, n0 = tn * BN;
    const T *Ap = a.A + seg * a.a_seg, *Bp = a.B + seg * a.b_seg;
    T *O = a.CG > 1 ? a.P + (seg * a.CG + gz) * a.sp : a.O + seg * a.o_seg;
    const int K = seg == a.S - 1 && a.k_last ? a.k_last : a.K;
    // The contraction of this group: one range of K, or a run of blocks of K indices.
    int k0 = 0, k1 = K, b0 = 0, b1 = 1;
    if (a.a_grp >= 0) {
        const int blocks = seg == a.S - 1 ? a.blocks_last : a.blocks,
                  run = (a.blocks + a.CG - 1) / a.CG;
        b0 = min(blocks, gz * run);
        b1 = min(blocks, b0 + run);
    } else {
        k0 = int((long long)K * gz / a.CG);
        k1 = int((long long)K * (gz + 1) / a.CG);
    }
    const int warp = __shfl_sync(0xffffffffu, tid / 32, 0), lane = threadIdx.x % 32;
    typename Mma::FragmentC acc;
    acc.clear();
    for (int blk = b0; blk < b1; ++blk) {
        const T *Ab = Ap + (a.a_grp >= 0 ? blk * a.a_grp : 0),
                *Bb = Bp + (a.a_grp >= 0 ? blk * a.b_grp : 0);
        typename Mma::IteratorA itA(typename Mma::IteratorA::Params(typename Cfg::LA(a.lda)),
                                    const_cast<T *>(Ab), {a.M, k1}, tid, {m0, k0});
        typename Mma::IteratorB itB(typename Mma::IteratorB::Params(typename Cfg::LB(a.ldb)),
                                    const_cast<T *>(Bb), {k1, a.N}, tid, {k0, n0});
        const int iters = (k1 - k0 + Cfg::BK - 1) / Cfg::BK;
        auto &ss = *reinterpret_cast<typename Mma::SharedStorage *>(smem);
        Mma mma(ss, tid, warp, lane, 0);
        if (iters > 0)
            mma(iters, acc, itA, itB, acc);
        if (blk + 1 < b1)
            __syncthreads(); // the next block refills the pipeline
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
    __syncthreads();
    const bool full = (m0 + BM <= a.M) && (n0 + BN <= a.N);
    for (int e = tid; e < BM * BN; e += Cfg::Threads) {
        const int i = e / BN, j = e % BN;
        const long long go = (n0 + j) + (long long)(m0 + i) * a.ldo;
        const bool in = full || ((m0 + i) < a.M && (n0 + j) < a.N);
        T s = 0;
#pragma unroll
        for (int w = 0; w < Cfg::SK; ++w) // fixed order w = 0..SK-1
            s += part[w * Cfg::PartElems + i * Cfg::LDP + j];
        if (in)
            O[go] = s; // the owner commits once (or its slice)
    }
}
// GPU-level fixed-order combine of CG partial slices sp apart: O = sum over g = 0..CG-1 of P_g, for
// `lines` lines of `width` elements, ldo apart, in each of S segments (output o_seg apart). Blocks
// walk the lines, threads the elements.
template <class T>
__global__ void carrier_g_combine(const T *P, long long sp, int CG, T *O, int ldo, int lines,
                                  int width, int S, long long o_seg) {
    for (int l = blockIdx.x; l < S * lines; l += gridDim.x) {
        const int seg = l / lines;
        const long long base = (long long)(l % lines) * ldo;
        const T *src = P + size_t(seg) * CG * sp + base;
        T *dst = O + seg * o_seg + base;
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            T s = 0;
            for (int g = 0; g < CG; ++g)
                s += src[size_t(g) * sp + j];
            dst[j] = s;
        }
    }
}
// Carrier of a launch in the frame of the product Out^T (N x M), or of Out (M x N) when `direct`;
// S segments stack their outputs along the rows of the product.
template <class Cfg>
Carrier carrier_g_carrier(int M, int N, int cg, bool direct = false, int S = 1) {
    const int tm = ceildiv(M, Cfg::BM), tn = ceildiv(N, Cfg::BN);
    Carrier c;
    if (direct)
        return c.set(GpuLevel, tm * S, tn, cg)
            .set(BlockLevel, Cfg::WarpCount::kM, Cfg::WarpCount::kN, Cfg::SK);
    return c.set(GpuLevel, tn * S, tm, cg)
        .set(BlockLevel, Cfg::WarpCount::kN, Cfg::WarpCount::kM, Cfg::SK);
}
template <class Cfg>
Carrier launch_carrier_g(const CarrierGArgs<typename Cfg::Element> &a, cudaStream_t st,
                         bool direct = false) {
    const Carrier carrier = carrier_g_carrier<Cfg>(a.M, a.N, a.CG, direct, a.S);
    if (!a.M || !a.N)
        return carrier;
    if (a.CG > 1 && (!a.P || a.sp < (long long)a.M * a.ldo))
        throw std::runtime_error("no room for the partial slices of the GPU groups");
    if ((long long)a.CG * a.CG > (long long)ceildiv(a.M, Cfg::BM) * ceildiv(a.N, Cfg::BN))
        throw std::runtime_error("GPU groups beyond the 2.5D bound of the product");
    reserve_shared_memory(carrier_g_kernel<Cfg>, Cfg::Smem, true);
    carrier_g_kernel<Cfg><<<dim3(ceildiv(a.M, Cfg::BM), ceildiv(a.N, Cfg::BN), a.S * a.CG),
                            dim3(Cfg::SliceThreads, Cfg::SK), Cfg::Smem, st>>>(a);
    CU(cudaGetLastError());
    if (a.CG > 1) {
        carrier_g_combine<typename Cfg::Element>
            <<<std::min(1024, ceildiv(a.S * a.M * a.N, 256)), 256, 0, st>>>(
                a.P, a.sp, a.CG, a.O, a.ldo, a.M, a.N, a.S, a.o_seg);
        CU(cudaGetLastError());
    }
    return carrier;
}
using CRM = cutlass::layout::RowMajor;
using CCM = cutlass::layout::ColumnMajor;
// The carrier of the arithmetic in use, as a tag: double, float (IEEE FP32 on CUDA cores),
// MathTF32 or MathX3.
template <class T, class F> decltype(auto) with_carrier_policy(F &&f) {
    if constexpr (std::is_same_v<T, double>)
        return f(double{});
    else {
        if (fp32_math() == Fp32Math::IEEE)
            return f(float{});
        return fp32_math() == Fp32Math::TF32 ? f(MathTF32{}) : f(MathX3{});
    }
}
// W = V^T X in the transposed frame W^T = X^T V on one of three tiles: 64 x 128 for FP64 (the whole
// width of a panel), 64 x 64, and 32 x 32 for the narrow products, whose few output tiles would
// otherwise admit too few groups of blocks within the 2.5D bound. Every tile keeps the (2, 2, 2)
// warps of the block.
template <class P> struct CarrierW {
    using Big = CarrierGCfg<P, CRM, CCM, 64, 64, 32, 32, 32, 16, 3, 1>;
    using Small = Big;
    using Tiny = CarrierGCfg<P, CRM, CCM, 32, 32, 32, 16, 16, 16, 3, 2>;
};
template <> struct CarrierW<double> {
    using Big = CarrierGCfg<double, CRM, CCM, 64, 128, 32, 32, 64, 16, 3, 1>;
    using Small = CarrierGCfg<double, CRM, CCM, 64, 64, 32, 32, 32, 16, 3, 2>;
    using Tiny = CarrierGCfg<double, CRM, CCM, 32, 32, 16, 16, 16, 8, 3, 2>;
};
template <> struct CarrierW<float> {
    using Big = CarrierGCfg<float, CRM, CCM, 64, 64, 16, 32, 32, 8, 3, 2>;
    using Small = Big;
    using Tiny = CarrierGCfg<float, CRM, CCM, 32, 32, 16, 16, 16, 8, 3, 2>;
};
// Rows of one k-tile of W: every group of blocks needs one so that both slices own rows.
constexpr int carrier_w_rows = 32;
// The tile of a W launch and its groups of blocks, each owning a contiguous range of rows.
struct WPlan {
    int tile = 1, groups = 1;
};
template <class T, class F> decltype(auto) with_carrier_w(int tile, F &&f) {
    return with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        return tile == 0   ? f(typename CarrierW<P>::Big{})
               : tile == 1 ? f(typename CarrierW<P>::Small{})
                           : f(typename CarrierW<P>::Tiny{});
    });
}
// The largest tile whose groups give about three quarters of a wave, else the one with the most
// blocks. Groups: at least c / 2 (the two slices of a block give the rest of the requested c),
// enough for two waves of the GPU, at least 256 rows and one k-tile each, at most `room`, and
// within the 2.5D bound CG^2 <= tiles.
template <class T> WPlan carrier_w_plan(int c, int q, int h, int rows, int room) {
    const int sms = device_properties().multiProcessorCount;
    WPlan best;
    long long best_blocks = 0;
    for (int tile = 0; tile < 3; ++tile) {
        const auto [tiles, groups] = with_carrier_w<T>(tile, [&](auto cfg) {
            using Cfg = decltype(cfg);
            const long long t = (long long)ceildiv(q, Cfg::BM) * ceildiv(h, Cfg::BN);
            long long g = std::max<long long>(c / 2, 2LL * sms / std::max(1LL, t));
            g = std::min<long long>({g, 128, std::max(1, room), std::max(1, rows / 256),
                                     std::max(1, rows / carrier_w_rows)});
            return std::pair<long long, int>(t, bounded_groups(t, int(std::max(1LL, g))));
        });
        if (tiles * groups >= 3 * sms / 4)
            return WPlan{tile, groups};
        if (tiles * groups > best_blocks) {
            best = WPlan{tile, groups};
            best_blocks = tiles * groups;
        }
    }
    return best;
}
// W(h x q, ldw) = V^T X on the tile and groups of `plan`; partial slices of q ldw words go to
// `part`.
template <class T>
Carrier launch_carrier_w(const T *v, int ldv, const T *x, int ldx, T *w, int ldw, int rows, int h,
                         int q, cudaStream_t st, WPlan plan, T *part = nullptr) {
    if (rows < plan.groups * carrier_w_rows)
        throw std::runtime_error("the W carrier needs a k-tile of rows per group of blocks");
    const CarrierGArgs<T> a{x, ldx, v,    ldv,         w,    ldw,
                            q, h,   rows, plan.groups, part, (long long)q * ldw};
    return with_carrier_w<T>(plan.tile,
                             [&](auto cfg) { return launch_carrier_g<decltype(cfg)>(a, st); });
}
// W from prepared arguments (segments, blocks) on a tile of with_carrier_w.
template <class T>
Carrier launch_carrier_w_args(const CarrierGArgs<T> &a, int tile, cudaStream_t st) {
    return with_carrier_w<T>(tile,
                             [&](auto cfg) { return launch_carrier_g<decltype(cfg)>(a, st); });
}
// Z = op(T) W (K = h), or Z^T = W^T op(T)^T for the D carriers, which read Z transposed, with the
// two slices of every block: FP64 and IEEE FP32 split every k-tile of 16 indices into slices of 8,
// the tensor-core modes every k-tile of 32 into slices of 16 (a 32-bit operand transposed through
// shared memory holds one 128-byte line per k-tile).
template <class P, class LA, class LB> struct CarrierZSel;
template <class LA, class LB> struct CarrierZSel<double, LA, LB> {
    using type = CarrierGCfg<double, LA, LB, 64, 64, 16, 32, 32, 8, 3, 2>;
};
template <class LA, class LB> struct CarrierZSel<float, LA, LB> {
    using type = CarrierGCfg<float, LA, LB, 64, 64, 16, 32, 32, 8, 3, 2>;
};
template <class LA, class LB> struct CarrierZSel<MathTF32, LA, LB> {
    using type = CarrierGCfg<MathTF32, LA, LB, 64, 64, 32, 32, 32, 16, 3, 1>;
};
template <class LA, class LB> struct CarrierZSel<MathX3, LA, LB> {
    using type = CarrierGCfg<MathX3, LA, LB, 64, 64, 32, 32, 32, 16, 3, 1>;
};
template <class P, class LA, class LB> using CarrierZCfg = typename CarrierZSel<P, LA, LB>::type;
template <class T> void carrier_z_law(int &bk, int &wk) {
    with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        bk = CarrierZCfg<P, CRM, CCM>::BK;
        wk = CarrierZCfg<P, CRM, CCM>::WK;
    });
}
// Both slices own reflectors when h covers the first slice of the residue k-tile.
template <class T> bool carrier_z_supported(int h) {
    int bk, wk;
    carrier_z_law<T>(bk, wk);
    return h >= bk - wk + 1;
}
// With S > 1, segment s computes its own Z from the T and W t_seg and w_seg elements further and
// writes it z_seg elements further. With groups > 1 (one segment), groups of blocks split the
// contraction and their partial slices, `slice` elements apart from `parts` on, meet in a fixed-order
// combine: the replication of Z at the GPU level.
template <class T>
Carrier launch_carrier_z(const T *t, int ldt, const T *w, int ldw, T *z, int ldz, int h, int q,
                         bool transpose, bool zt_out, cudaStream_t st, int S = 1,
                         long long t_seg = 0, long long w_seg = 0, long long z_seg = 0,
                         int groups = 1, T *parts = nullptr, long long slice = 0) {
    if (!carrier_z_supported<T>(h))
        throw std::runtime_error("the Z carrier needs a reflector per slice");
    return with_carrier_policy<T>([&](auto tag) {
        using P = decltype(tag);
        auto go = [&](auto la, auto lb, const T *A, int lda, const T *B, int ldb, int M, int N) {
            using Cfg = CarrierZCfg<P, decltype(la), decltype(lb)>;
            CarrierGArgs<T> a{A, lda, B, ldb, z, ldz, M, N, h};
            a.CG = groups;
            a.P = parts;
            a.sp = slice;
            a.S = S;
            a.a_seg = A == t ? t_seg : w_seg;
            a.b_seg = A == t ? w_seg : t_seg;
            a.o_seg = z_seg;
            return launch_carrier_g<Cfg>(a, st, zt_out);
        };
        if (zt_out)
            return transpose ? go(CRM{}, CCM{}, t, ldt, w, ldw, h, q)
                             : go(CCM{}, CCM{}, t, ldt, w, ldw, h, q);
        return transpose ? go(CRM{}, CCM{}, w, ldw, t, ldt, q, h)
                         : go(CRM{}, CRM{}, w, ldw, t, ldt, q, h);
    });
}
} // namespace tqr
