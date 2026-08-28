#pragma once
// LRQR provides CUDA implementations of layered-reduction Householder QR.
// Panels are factored with TSQR-style reductions, reconstructed as compact-WY
// transforms, and applied through tiered trailing updates with look-ahead.
// Uses the in-tree blocked QR implementation rather than cuSolverDx.
#ifndef LRQR_APPLY_PHASE1_WMMA
#define LRQR_APPLY_PHASE1_WMMA 1
#endif
#ifndef LRQR_APPLY_PHASE3_WMMA
#define LRQR_APPLY_PHASE3_WMMA 1
#endif
// Set LRQR_NVTX=1 at compile time to emit host-side NVTX ranges.
#ifndef LRQR_NVTX
#define LRQR_NVTX 0
#endif

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "tqr_repl_model.cuh"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <mma.h>
#include <type_traits>
#include <thread>
#include <atomic>
#include <chrono>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while (0)

// Host-side profiling ranges; no-ops unless LRQR_NVTX is enabled.
#if LRQR_NVTX
  #if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
  #else
    #include <nvToolsExt.h>
  #endif
  #define NVTX_RANGE(name) nvtxRangePushA(name)
  #define NVTX_POP()       nvtxRangePop()
#else
  #define NVTX_RANGE(name) ((void)0)
  #define NVTX_POP()       ((void)0)
#endif

namespace lrqr {

// Options for Plan::create. Disable validation_copy to omit Q-reconstruction
// buffers and reduce production memory use. The default supports validation.
struct lrqr_opts_t {
    bool validation_copy = true;
};

// Register-tiled smem microkernel: 256 threads compute C (MR×NC) = op(A)·B (+optional).
// Thread (tx,ty) of a 16×16 grid owns a (MR/16)×(NC/16) register tile.
// TRANSA: C[r,c] = Σ_k A[k + r*lda]·B[k + c*ldb]   (A is K×MR, col-major)
// else:   C[r,c] = Σ_k A[r + k*lda]·B[k + c*ldb]   (A is MR×K)
template<typename Real, int MR, int NC, int K, bool TRANSA, int TM = MR / 16, int TN = NC / 16>
__device__ __forceinline__ void mmtile(const Real* A, int lda, const Real* B, int ldb,
                                       Real acc[TM][TN]) {
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int r0 = tx * TM, c0 = ty * TN;
#pragma unroll 1
    for (int k = 0; k < K; ++k) {
        Real a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) a[i] = TRANSA ? A[k + (r0 + i) * lda] : A[(r0 + i) + k * lda];
#pragma unroll
        for (int j = 0; j < TN; ++j) b[j] = B[k + (c0 + j) * ldb];
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
}

// TF32 tensor-core GEMM for 128-square tiles. Output is written to shared memory;
// callers synchronize before consuming it. Only instantiated for float.
template<bool TRANSA, int NB>
__device__ __forceinline__ void wmma_gemm128(const float* A, int lda, const float* B, int ldb,
                                             float* C, int ldc) {
    namespace w = nvcuda::wmma;
    const int warp = threadIdx.x >> 5, wr = warp >> 1, wc = warp & 1;
    using AL = std::conditional_t<TRANSA, w::row_major, w::col_major>;
    w::fragment<w::accumulator, 16, 16, 8, float> cf[2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) w::fill_fragment(cf[i][j], 0.0f);
    for (int kt = 0; kt < NB / 8; ++kt) {
        const int k0 = 8 * kt;
        w::fragment<w::matrix_a, 16, 16, 8, w::precision::tf32, AL> af[2];
        w::fragment<w::matrix_b, 16, 16, 8, w::precision::tf32, w::col_major> bf[4];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r0 = 32 * wr + 16 * i;
            const float* ap = TRANSA ? (A + r0 * lda + k0) : (A + r0 + k0 * lda);
            w::load_matrix_sync(af[i], ap, lda);
#pragma unroll
            for (int t = 0; t < af[i].num_elements; ++t) af[i].x[t] = w::__float_to_tf32(af[i].x[t]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w::load_matrix_sync(bf[j], B + k0 + (64 * wc + 16 * j) * ldb, ldb);
#pragma unroll
            for (int t = 0; t < bf[j].num_elements; ++t) bf[j].x[t] = w::__float_to_tf32(bf[j].x[t]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j) w::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 4; ++j)
        w::store_matrix_sync(C + (32 * wr + 16 * i) + (64 * wc + 16 * j) * ldc, cf[i][j], ldc, w::mem_col_major);
}

__device__ int g_lrqr_tree_wmma = 1;   // runtime A/B toggle (host sets via LRQR_NO_WMMA)

// ---------------------------------------------------------------- in-block blocked QR
// Blocked Householder QR of an M×NB smem tile: warp 0 factors 16-column mini-panels
// (shuffle reductions, no block syncs on the latency path), then all threads apply the
// compound WY update to the trailing columns. Replaces the latency-bound one-column-at-
// a-time vendor path (cuSolverDx merge measured ~1 ms/node; this targets ~100 µs).
// Layout: A col-major lda=M in smem; V below diag (unit implicit), R on/above; tau[NB].
// Scratch: Vp (M×16), W (16×NB), G/T16 (16×16 each).
template<typename Real, int M, int NB>
__device__ void d_bqr_step(Real* A, Real* tau, Real* Vp, Real* W, Real* W2, Real* G16,
                           Real* T16, int p) {
    constexpr int PW = 16;
    constexpr int LDA = M + 1;
    const int tid = threadIdx.x, nth = blockDim.x, lane = tid & 31;
    const int warp = tid >> 5;
    {
        const int pe = min(p + PW, NB), mp = M - p, ldv = mp + 1;
        // ---- mini-panel factorization: EVERY warp builds each reflector ----
        //
        // 10:S6.2 gives rung 0 Split = "fragments and k slices" and Combine =
        // "warp-level partial sums and local Householder reductions". This used to be
        // `if (tid < 32)`: one warp of eight owned the whole reflector -- the norm, the
        // scalars and the scaling -- while the other seven sat at the __syncthreads
        // below. ncu named the cost exactly (job 9709):
        //
        //   k_merge_geqrf  61.1% of all stall cycles are Stall Barrier, est. speedup
        //                  61.06%, "commonly caused by diverging code paths before a
        //                  barrier"; 24.06 of 32 threads active per warp
        //   k_leaf_geqrf   est. speedup 40.91%; 0.12 inst/cycle issued
        //
        // An analytic count of loop ITERATIONS said k-slicing was worth 0-4%, which is
        // why it is measured and not modelled: the cost was never the iterations, it
        // was seven warps idling at a barrier.
        //
        // So: k-slice the column over all `nth` threads, reduce within each warp by
        // shuffle, then combine the per-warp partials through G16 -- which is dead here
        // (it is written after this loop) and is PW*PW = 256 words, comfortably more
        // than the <=32 warps a 1024-thread block can have.
        //
        // The reduction order changes (warp-partial tree instead of one warp's
        // shuffle). 10:S1.2 permits it -- "the tree is allowed to change the
        // floating-point result within the standard backward-stable Householder
        // envelope" -- and the kappa-flatness gate (<10x over kappa = 1e2..1e14) is
        // what holds it to that envelope.
        for (int j = p; j < pe; ++j) {
            {
                Real sig = Real(0);
                for (int i = j + 1 + tid; i < M; i += nth) {
                    const Real v = A[i + j * LDA];
                    sig += v * v;
                }
                #pragma unroll
                for (int o = 16; o; o >>= 1) sig += __shfl_xor_sync(0xffffffffu, sig, o);
                if (lane == 0) G16[warp] = sig;
            }
            __syncthreads();
            if (warp == 0) {
                const int NWr = (nth + 31) >> 5;
                Real s = (lane < NWr) ? G16[lane] : Real(0);
                #pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
                __syncwarp();          // every lane has read G16 before lane 0 rewrites it
                if (lane == 0) {
                    const Real alpha = A[j + j * LDA];
                    Real tj = Real(0), vs = Real(0), beta = alpha;
                    if (s != Real(0)) {
                        const Real mu = sqrt(alpha * alpha + s);
                        beta = (alpha >= Real(0)) ? -mu : mu;  // |beta| = ‖x‖, v0 = α−β never cancels
                        tj = (beta - alpha) / beta;
                        vs = Real(1) / (alpha - beta);
                    }
                    tau[j] = tj; A[j + j * LDA] = beta;
                    G16[0] = vs;       // broadcast the scale to the whole block
                }
            }
            __syncthreads();
            {   // scale v, also k-sliced
                const Real vs = G16[0];
                for (int i = j + 1 + tid; i < M; i += nth) A[i + j * LDA] *= vs;
            }
            __syncthreads();
            {   // apply H_j to columns j+1..pe-1, one warp per column (round-robin)
                Real tj = tau[j];
                for (int c = j + 1 + warp; c < pe; c += (int)(blockDim.x >> 5)) {
                    Real d = (lane == 0) ? A[j + c * LDA] : Real(0);
                    for (int i = j + 1 + lane; i < M; i += 32) d += A[i + j * LDA] * A[i + c * LDA];
                    #pragma unroll
                    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                    Real w = tj * d;
                    if (lane == 0) A[j + c * LDA] -= w;
                    for (int i = j + 1 + lane; i < M; i += 32) A[i + c * LDA] -= A[i + j * LDA] * w;
                }
            }
            __syncthreads();
        }
        const int wt = NB - pe;
        // ---- explicit mini-panel V (unit diag, zeros above) rows p..M
        for (int idx = tid; idx < mp * PW; idx += nth) {
            int c = idx / mp, r = idx % mp;             // r,c relative: row p+r, col p+c
            Real v = Real(0);
            if (p + c < pe) {
                if (r > c) v = A[(p + r) + (p + c) * LDA];
                else if (r == c) v = Real(1);
            }
            Vp[r + c * ldv] = v;
        }
        __syncthreads();
        if (wt == 0) return;
        // ---- G16 = Vp^T Vp (strict upper used)
        for (int idx = tid; idx < PW * PW; idx += nth) {
            int c = idx / PW, r = idx % PW;
            Real acc = Real(0);
            for (int k = 0; k < mp; ++k) acc += Vp[k + r * ldv] * Vp[k + c * ldv];
            G16[r + c * PW] = acc;
        }
        __syncthreads();
        // ---- T16 recurrence (warp 0; forward columnwise)
        if (tid < 32) {
            for (int j = 0; j < PW; ++j) {
                Real tj = (p + j < pe) ? tau[p + j] : Real(0);
                Real acc = Real(0);
                if (lane < j) {
                    for (int k = lane; k < j; ++k) acc += T16[lane + k * PW] * G16[k + j * PW];
                }
                __syncwarp();
                if (lane < j) T16[lane + j * PW] = -tj * acc;
                else if (lane == j) T16[j + j * PW] = tj;
                else if (lane < PW) T16[lane + j * PW] = Real(0);
                __syncwarp();
            }
        }
        __syncthreads();
        // ---- W = Vp^T · A[p.., pe..NB]  (register-tiled; static acc indexing — no spills)
        {
            // The column-group stride was the literal 16, i.e. nth/PW at nth = 256.
            // That pinned the kernel to a 256-thread block, and ncu (job 9709) measured
            // the price: Block Limit Shared Mem = 1, so ONE block of 256 threads is
            // resident per SM -- 8 warps, 12.5% occupancy, 0.12 inst/cycle issued, and
            // an estimated 87.5% left on the table. CT stays sized for the SMALLEST
            // supported block (256) so `acc` is still static; a wider block simply has
            // a larger stride and guards the extra iterations off.
            constexpr int CT = NB / (256 / 16);
            const int NG = nth >> 4;                  // = nth / PW, PW == 16 here
            const int r = tid & 15, cw = tid >> 4;
            Real acc[CT];
            #pragma unroll
            for (int ci = 0; ci < CT; ++ci) acc[ci] = Real(0);
            for (int k = 0; k < mp; ++k) {
                Real vk = Vp[k + r * ldv];
                #pragma unroll
                for (int ci = 0; ci < CT; ++ci) {
                    int c = cw + NG * ci;
                    if (c < wt) acc[ci] += vk * A[(p + k) + (pe + c) * LDA];
                }
            }
            #pragma unroll
            for (int ci = 0; ci < CT; ++ci) {
                int c = cw + NG * ci;
                if (c < wt) W[r + c * PW] = acc[ci];
            }
        }
        __syncthreads();
        // ---- W2 = T16^T · W (Q^T apply: I − V T^T V^T)
        for (int idx = tid; idx < PW * wt; idx += nth) {
            int c = idx / PW, r = idx % PW;
            Real acc = Real(0);
            for (int k = 0; k <= r; ++k) acc += T16[k + r * PW] * W[k + c * PW];   // T^T lower
            W2[r + c * PW] = acc;
        }
        __syncthreads();
        // ---- A[p.., pe..] −= Vp · W2  (one row per thread; static acc indexing — no spills)
        for (int r = tid; r < mp; r += nth) {
            for (int cb = 0; cb < wt; cb += 8) {
                Real acc[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] = Real(0);
                #pragma unroll 4
                for (int k = 0; k < PW; ++k) {
                    Real vk = Vp[r + k * ldv];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        if (cb + j < wt) acc[j] += vk * W2[k + (cb + j) * PW];
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (cb + j < wt) A[(p + r) + (pe + cb + j) * LDA] -= acc[j];
            }
        }
        __syncthreads();
    }
}

template<typename Real, int M, int NB>
__device__ void bqr_core(Real* A, Real* tau, Real* Vp, Real* W, Real* W2, Real* G16, Real* T16) {
    for (int p = 0; p < NB; p += 16)
        d_bqr_step<Real, M, NB>(A, tau, Vp, W, W2, G16, T16, p);
}

// smem bytes for a bqr kernel with M rows
template<typename Real, int NB>
constexpr int bqr_smem(int M) {
    return ((M + 1) * NB + (M + 1) * 16 + 2 * 16 * NB + 2 * 16 * 16 + NB) * (int)sizeof(Real);
}

// Leaf QR: in-place blocked QR of a SLABxNB tile of the panel buffer.
template<typename Real, int SLAB, int NB>
__global__ void k_leaf_geqrf(Real* P, int ldp, Real* tauL, int nleaves) {
    extern __shared__ __align__(16) unsigned char smraw[];
    Real* As  = reinterpret_cast<Real*>(smraw);
    Real* Vp  = As + (SLAB + 1) * NB;
    Real* W   = Vp + (SLAB + 1) * 16;
    Real* W2  = W + 16 * NB;
    Real* G16 = W2 + 16 * NB;
    Real* T16 = G16 + 16 * 16;
    Real* tau = T16 + 16 * 16;
    const int leaf = blockIdx.x;
    if (leaf >= nleaves) return;
    Real* src = P + size_t(leaf) * SLAB;
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int idx = tid; idx < SLAB * NB; idx += nth) {
        int c = idx / SLAB, r = idx % SLAB;
        As[c * (SLAB + 1) + r] = src[size_t(c) * ldp + r];
    }
    __syncthreads();
    bqr_core<Real, SLAB, NB>(As, tau, Vp, W, W2, G16, T16);
    for (int idx = tid; idx < SLAB * NB; idx += nth) {
        int c = idx / SLAB, r = idx % SLAB;
        src[size_t(c) * ldp + r] = As[c * (SLAB + 1) + r];
    }
    for (int i = tid; i < NB; i += nth) tauL[leaf * NB + i] = tau[i];
}

// Merge: Householder QR of two stacked NBxNB upper triangles (the ⊕ of spec §3), fully
// structure-exploiting (spec §7 "triangle-on-triangle"): every reflector's top part is
// exactly e_j, and the bottom support of reflector j is rows 0..j of the second triangle.
// All loops run over the ~ (pe+16)-row live range instead of 2NB dense rows.
// smem: Rt (NB×NB upper, becomes R) | Bt (NB×NB, triangle j → becomes V_bot) | W/W2/G16/T16/tau.
template<typename Real, int SLAB, int NB, int PW = 16>
__device__ __noinline__ void d_merge_body(unsigned char* smraw, Real* P, int ldp, int2 nd,
                             Real* Vout, Real* tauOut) {
    Real* Rt  = reinterpret_cast<Real*>(smraw);      // NB×NB (top triangle / R out)
    Real* Bt  = Rt + (NB + 1) * NB;                        // NB×NB (bottom triangle / V_bot out)
    Real* W   = Bt + (NB + 1) * NB;                        // PW×NB
    Real* W2  = W + PW * NB;                         // PW×NB
    Real* G16 = W2 + PW * NB;
    Real* T16 = G16 + PW * PW;
    Real* tau = T16 + PW * PW;
    constexpr int LDS = NB + 1;
    Real* Ri = P + size_t(nd.x) * SLAB;
    Real* Rj = P + size_t(nd.y) * SLAB;
    const int tid = threadIdx.x, nth = blockDim.x, lane = tid & 31, warp = tid >> 5;
    for (int idx = tid; idx < NB * NB; idx += nth) {
        int c = idx / NB, r = idx % NB;
        Rt[r + c * LDS] = (r <= c) ? Ri[size_t(c) * ldp + r] : Real(0);
        Bt[r + c * LDS] = (r <= c) ? Rj[size_t(c) * ldp + r] : Real(0);
    }
    __syncthreads();
    for (int p = 0; p < NB; p += PW) {
        const int pe = p + PW;
        // ---- mini-panel: reflector j lives in {Rt[j,j]} ∪ Bt[0..j, j]
        // Same rung-0 k-slice as d_bqr_step, and this kernel is where ncu measured the
        // cost: k_merge_geqrf spends 61.1% of its stall cycles waiting at the barrier
        // below, because `if (tid < 32)` left seven of eight warps with nothing to do.
        // It is 48.2% of the whole TSQR panel, so this is the single largest term.
        for (int j = p; j < pe; ++j) {
            {
                Real sig = Real(0);
                for (int i = tid; i <= j; i += nth) {
                    const Real v = Bt[i + j * LDS];
                    sig += v * v;
                }
                #pragma unroll
                for (int o = 16; o; o >>= 1) sig += __shfl_xor_sync(0xffffffffu, sig, o);
                if (lane == 0) G16[warp] = sig;
            }
            __syncthreads();
            if (warp == 0) {
                const int NWr = (nth + 31) >> 5;
                Real s = (lane < NWr) ? G16[lane] : Real(0);
                #pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
                __syncwarp();          // every lane has read G16 before lane 0 rewrites it
                if (lane == 0) {
                    const Real alpha = Rt[j + j * LDS];
                    Real tj = Real(0), vs = Real(0), beta = alpha;
                    if (s != Real(0)) {
                        const Real mu = sqrt(alpha * alpha + s);
                        beta = (alpha >= Real(0)) ? -mu : mu;
                        tj = (beta - alpha) / beta;
                        vs = Real(1) / (alpha - beta);
                    }
                    tau[j] = tj; Rt[j + j * LDS] = beta;
                    G16[0] = vs;
                }
            }
            __syncthreads();
            {
                const Real vs = G16[0];
                for (int i = tid; i <= j; i += nth) Bt[i + j * LDS] *= vs;
            }
            __syncthreads();
            {   // apply H_j to mini-panel columns c = j+1..pe-1 (one warp per column)
                Real tj = tau[j];
                for (int c = j + 1 + warp; c < pe; c += (int)(blockDim.x >> 5)) {
                    Real d = (lane == 0) ? Rt[j + c * LDS] : Real(0);
                    for (int i = lane; i <= j; i += 32) d += Bt[i + j * LDS] * Bt[i + c * LDS];
                    #pragma unroll
                    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                    Real w = tj * d;
                    if (lane == 0) Rt[j + c * LDS] -= w;
                    for (int i = lane; i <= j; i += 32) Bt[i + c * LDS] -= Bt[i + j * LDS] * w;
                }
            }
            __syncthreads();
        }
        const int wt = NB - pe;
        if (wt == 0) break;
        // ---- G16[r,c] = w_r^T w_c over bottom rows 0..pe (V columns p+r, p+c)
        for (int idx = tid; idx < PW * PW; idx += nth) {
            int c = idx / PW, r = idx % PW;
            Real acc = Real(0);
            for (int k = 0; k <= p + r; ++k) acc += Bt[k + (p + r) * LDS] * Bt[k + (p + c) * LDS];
            G16[r + c * PW] = acc;
        }
        __syncthreads();
        if (tid < 32) {   // T16 forward recurrence
            for (int j = 0; j < PW; ++j) {
                Real tj = tau[p + j];
                Real acc = Real(0);
                if (lane < j) for (int k = lane; k < j; ++k) acc += T16[lane + k * PW] * G16[k + j * PW];
                __syncwarp();
                if (lane < j) T16[lane + j * PW] = -tj * acc;
                else if (lane == j) T16[j + j * PW] = tj;
                else if (lane < PW) T16[lane + j * PW] = Real(0);
                __syncwarp();
            }
        }
        __syncthreads();
        // ---- W = V^T A_tr = Rt[p..pe, tr] + V_bot^T Bt[0..pe, tr]
        {
            constexpr int NGmin = 256 / PW;            // column groups at the smallest block
            constexpr int CT = (NB + NGmin - 1) / NGmin;
            const int NG = nth / PW;                   // actual, so the block size is free
            const int r = tid % PW, cw = tid / PW;
            Real acc[CT];
            #pragma unroll
            for (int ci = 0; ci < CT; ++ci) acc[ci] = Real(0);
            for (int k = 0; k <= p + r; ++k) {         // bottom support of v_{p+r}: rows 0..p+r
                Real vk = Bt[k + (p + r) * LDS];
                #pragma unroll
                for (int ci = 0; ci < CT; ++ci) {
                    int c = cw + NG * ci;
                    if (c < wt) acc[ci] += vk * Bt[k + (pe + c) * LDS];
                }
            }
            #pragma unroll
            for (int ci = 0; ci < CT; ++ci) {
                int c = cw + NG * ci;
                if (c < wt) W[r + c * PW] = acc[ci] + Rt[(p + r) + (pe + c) * LDS];
            }
        }
        __syncthreads();
        // ---- W2 = T16^T · W
        for (int idx = tid; idx < PW * wt; idx += nth) {
            int c = idx / PW, r = idx % PW;
            Real acc = Real(0);
            for (int k = 0; k <= r; ++k) acc += T16[k + r * PW] * W[k + c * PW];
            W2[r + c * PW] = acc;
        }
        __syncthreads();
        // ---- A_tr update: Rt rows p..pe get −W2; Bt rows 0..pe get −V_bot·W2
        for (int idx = tid; idx < PW * wt; idx += nth) {
            int c = idx / PW, r = idx % PW;
            Rt[(p + r) + (pe + c) * LDS] -= W2[r + c * PW];
        }
        for (int r = tid; r < pe; r += nth) {
            for (int cb = 0; cb < wt; cb += 8) {
                Real acc[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] = Real(0);
                #pragma unroll 4
                for (int k = 0; k < PW; ++k) {
                    if (p + k >= r) {                  // v_{p+k} bottom support: rows 0..p+k
                        Real vk = Bt[r + (p + k) * LDS];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            if (cb + j < wt) acc[j] += vk * W2[k + (cb + j) * PW];
                    }
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (cb + j < wt) Bt[r + (pe + cb + j) * LDS] -= acc[j];
            }
        }
        __syncthreads();
    }
    for (int idx = tid; idx < NB * NB; idx += nth) {
        int c = idx / NB, r = idx % NB;
        if (r <= c) Ri[size_t(c) * ldp + r] = Rt[r + c * LDS];
        Vout[idx] = Bt[r + c * LDS];
    }
    for (int i = tid; i < NB; i += nth) tauOut[i] = tau[i];
}

template<typename Real, int SLAB, int NB, int PW = 16, int NT = 256>
__global__ void k_merge_geqrf(Real* P, int ldp, const int2* nodes, int nnodes, int nodeBase,
                              Real* Vm, Real* taum) {
    extern __shared__ __align__(16) unsigned char smraw[];
    const int nid = blockIdx.x;
    if (nid >= nnodes) return;
    d_merge_body<Real, SLAB, NB, PW>(smraw, P, ldp, nodes[nid],
                                     Vm + size_t(nodeBase + nid) * NB * NB,
                                     taum + size_t(nodeBase + nid) * NB);
}

// smem bytes for the structured merge kernel
template<typename Real, int NB, int PW = 16>
constexpr int merge_smem() {
    return (2 * (NB + 1) * NB + 2 * PW * NB + 2 * PW * PW + NB) * (int)sizeof(Real);
}

// larft: T (upper tri) from Householder vectors + taus (forward columnwise recurrence).
// MERGE: V = Vm[node] (NB×NB upper-tri; effective Gram = V^T V since tops are e_j's).
// LEAF:  V = explicit unit-lower SLAB×NB from panel tile.
// Dynamic smem: V (VR×NB) | G (NB×NB) | T (NB×NB) | tmp (NB).
template<typename Real, int SLAB, int NB, bool MERGE>
__device__ __noinline__ void d_larft_body(unsigned char* smraw, const Real* P, int ldp, const Real* Vm,
                             const Real* tauAll, Real* Tout, int item, int nodeBase) {
    constexpr int VR = MERGE ? NB : SLAB;
    Real* Vs  = reinterpret_cast<Real*>(smraw);
    Real* Gs  = Vs + (VR + 1) * NB;
    Real* Ts  = Gs + NB * NB;
    Real* tmp = Ts + NB * NB;
    const int tid = threadIdx.x, nth = blockDim.x;
    const Real* tau;
    if (MERGE) {
        const Real* Vsrc = Vm + size_t(nodeBase + item) * NB * NB;
        for (int idx = tid; idx < NB * NB; idx += nth) {
            int c = idx / NB, r = idx % NB;
            Vs[r + c * (VR + 1)] = Vsrc[idx];
        }
        tau = tauAll + size_t(nodeBase + item) * NB;
    } else {
        const Real* tile = P + size_t(item) * SLAB;
        for (int idx = tid; idx < VR * NB; idx += nth) {
            int c = idx / VR, r = idx % VR;
            Real v = Real(0);
            if (r > c) v = tile[size_t(c) * ldp + r];
            else if (r == c) v = Real(1);
            Vs[r + c * (VR + 1)] = v;
        }
        tau = tauAll + size_t(item) * NB;
    }
    __syncthreads();
    {   // G = V^T V (register-tiled)
        constexpr int TM = NB / 16, TN = NB / 16;
        Real acc[TM][TN] = {};
        mmtile<Real, NB, NB, VR, true>(Vs, VR + 1, Vs, VR + 1, acc);
        const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                Gs[(tx * TM + i) + (ty * TN + j) * NB] = acc[i][j];
    }
    for (int idx = tid; idx < NB * NB; idx += nth) Ts[idx] = Real(0);
    __syncthreads();
    for (int j = 0; j < NB; ++j) {
        Real tj = tau[j];
        for (int i = tid; i < j; i += nth) {
            Real acc = Real(0);
            for (int k = i; k < j; ++k) acc += Ts[i + k * NB] * Gs[k + j * NB];
            tmp[i] = acc;
        }
        __syncthreads();
        for (int i = tid; i < j; i += nth) Ts[i + j * NB] = -tj * tmp[i];
        if (tid == 0) Ts[j + j * NB] = tj;
        __syncthreads();
    }
    Real* Td = Tout + size_t((MERGE ? nodeBase : 0) + item) * NB * NB;
    for (int idx = tid; idx < NB * NB; idx += nth) Td[idx] = Ts[idx];
}

template<typename Real, int SLAB, int NB, bool MERGE>
__global__ void k_larft(const Real* P, int ldp, const Real* Vm, const Real* tauAll,
                        Real* Tout, int count, int nodeBase) {
    extern __shared__ __align__(16) unsigned char smraw[];
    if (blockIdx.x >= count) return;
    d_larft_body<Real, SLAB, NB, MERGE>(smraw, P, ldp, Vm, tauAll, Tout, blockIdx.x, nodeBase);
}

// Fused down-sweep (per merge node, top→bottom): W = T·Q_in ; QS[i] = Q_in − W ;
// QS[j] = −V_bot·W. One kernel, no global W round-trip; W overwrites T's smem.
template<typename Real, int NB>
__device__ __noinline__ void d_dsweep_body(unsigned char* smraw, Real* QS, const Real* TmM, const Real* Vm,
                              int2 nd, int nodeIdx) {
    Real* Ts = reinterpret_cast<Real*>(smraw);      // T, later W
    Real* Qs = Ts + NB * NB;
    Real* Vs = Qs + NB * NB;
    const int tid = threadIdx.x, nth = blockDim.x;
    const Real* T = TmM + size_t(nodeIdx) * NB * NB;
    const Real* V = Vm + size_t(nodeIdx) * NB * NB;
    Real* Qi = QS + size_t(nd.x) * NB * NB;
    Real* Qj = QS + size_t(nd.y) * NB * NB;
    for (int idx = tid; idx < NB * NB; idx += nth) {
        Ts[idx] = T[idx]; Qs[idx] = Qi[idx]; Vs[idx] = V[idx];
    }
    __syncthreads();
    constexpr int TM = NB / 16, TN = NB / 16;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    {   // W = T·Q_in (regs) → write Q_i, then W over T's smem
        Real acc[TM][TN] = {};
        mmtile<Real, NB, NB, NB, false>(Ts, NB, Qs, NB, acc);
        __syncthreads();               // all reads of T complete
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j) {
                int r = tx * TM + i, c = ty * TN + j;
                Qi[r + c * NB] = Qs[r + c * NB] - acc[i][j];
                Ts[r + c * NB] = acc[i][j];
            }
    }
    __syncthreads();
    {   // Q_j = −V_bot·W
        Real acc[TM][TN] = {};
        mmtile<Real, NB, NB, NB, false>(Vs, NB, Ts, NB, acc);
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                Qj[(tx * TM + i) + (ty * TN + j) * NB] = -acc[i][j];
    }
}

template<typename Real, int NB>
__global__ void k_dsweep(Real* QS, const Real* TmM, const Real* Vm,
                         const int2* nodes, int nnodes, int nodeBase) {
    extern __shared__ __align__(16) unsigned char smraw[];
    const int nid = blockIdx.x;
    if (nid >= nnodes) return;
    d_dsweep_body<Real, NB>(smraw, QS, TmM, Vm, nodes[nid], nodeBase + nid);
}

// Leaf apply (fused): thinQ rows of leaf = (I − V T V^T)·[Q_in; 0].
// Dyn smem: Vhat (SLAB×NB) | B1 (NB²) | B2 (NB²).
template<typename Real, int SLAB, int NB>
__device__ __noinline__ void d_leaf_apply_body(unsigned char* smraw, const Real* P, int ldp, const Real* TmL,
                                  const Real* QS, Real* QT, int ldq, int leaf) {
    Real* Vs = reinterpret_cast<Real*>(smraw);
    Real* B1 = Vs + (SLAB + 1) * NB;
    Real* B2 = B1 + NB * NB;
    const int tid = threadIdx.x, nth = blockDim.x;
    const Real* tile = P + size_t(leaf) * SLAB;
    const Real* Qin  = QS + size_t(leaf) * NB * NB;
    for (int idx = tid; idx < SLAB * NB; idx += nth) {
        int c = idx / SLAB, r = idx % SLAB;
        Real v = Real(0);
        if (r > c) v = tile[size_t(c) * ldp + r];
        else if (r == c) v = Real(1);
        Vs[r + c * (SLAB + 1)] = v;
    }
    for (int idx = tid; idx < NB * NB; idx += nth) B1[idx] = Qin[idx];
    __syncthreads();
    constexpr int TM = NB / 16, TN = NB / 16;
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    {   // X = Vhat_top^T · Qin → B2  (transposed GEMM: 3.4× on TF32 tensor cores — Lever A)
      bool did = false;
      if constexpr (std::is_same<Real, float>::value) {
        if (g_lrqr_tree_wmma) { wmma_gemm128<true, NB>(Vs, SLAB + 1, B1, NB, B2, NB); did = true; }
      }
      if (!did) {
        Real acc[TM][TN] = {};
        mmtile<Real, NB, NB, NB, true>(Vs, SLAB + 1, B1, NB, acc);
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                B2[(tx * TM + i) + (ty * TN + j) * NB] = acc[i][j];
      }
    }
    __syncthreads();
    { const Real* T = TmL + size_t(leaf) * NB * NB;
      for (int idx = tid; idx < NB * NB; idx += nth) B1[idx] = T[idx]; }
    __syncthreads();
    {   // Y = T·X over B2 (all threads hold full acc before any write; one sync barrier)
        Real acc[TM][TN] = {};
        mmtile<Real, NB, NB, NB, false>(B1, NB, B2, NB, acc);
        __syncthreads();
        for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
                B2[(tx * TM + i) + (ty * TN + j) * NB] = acc[i][j];
    }
    __syncthreads();
    {   // out = [Qin;0] − Vhat·Y → QT rows [leaf*SLAB, +SLAB)
        constexpr int TMo = SLAB / 16;
        Real acc[TMo][TN] = {};
        mmtile<Real, SLAB, NB, NB, false>(Vs, SLAB + 1, B2, NB, acc);
        Real* out = QT + size_t(leaf) * SLAB;
        for (int i = 0; i < TMo; ++i)
            for (int j = 0; j < TN; ++j) {
                int r = tx * TMo + i, c = ty * TN + j;
                Real q = (r < NB) ? Qin[r + c * NB] : Real(0);
                out[size_t(c) * ldq + r] = q - acc[i][j];
            }
    }
}

template<typename Real, int SLAB, int NB>
__global__ void k_leaf_apply(const Real* P, int ldp, const Real* TmL, const Real* QS,
                             Real* QT, int ldq, int nleaves) {
    extern __shared__ __align__(16) unsigned char smraw[];
    if (blockIdx.x >= nleaves) return;
    d_leaf_apply_body<Real, SLAB, NB>(smraw, P, ldp, TmL, QS, QT, ldq, blockIdx.x);
}

// Extract R (upper triangle, NB×NB) from A's slab into P at the given slab offset.
// Used by the c_L-lane path to collect per-lane R-states for the cross-lane ⊕ merge.
// nrows: number of valid rows in the slab (≤ NB for short slabs — remaining rows zeroed).
template<typename Real, int NB>
__global__ void k_extract_R(const Real* A, size_t lda, int slab_start, int nrows,
                            Real* P, int ldp, int p_off) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NB * NB) return;
    int c = idx / NB, r = idx % NB;
    P[size_t(p_off + r) + size_t(c) * ldp] =
        (r <= c && r < nrows) ? A[size_t(slab_start + r) + size_t(c) * lda] : Real(0);
}

// All cL lanes' R extracted in ONE launch. panel_25d used to call k_extract_R
// once per lane -- 8 launches of 16 blocks each at c_L=8, on the panel path, per
// panel. The per-lane geometry needs no upload to fold them together: panel_25d
// lays the lanes out as slabStart[j] = j*base and slabSize[j] = base for every
// lane but the last, which takes the remainder, so a block can recover its own
// lane's offsets from (base, cL, mk) arithmetically. Grid is cL * the old block
// count; lane = blockIdx.x / blocksPerLane.
template<typename Real, int NB>
__global__ void k_extract_R_lanes(const Real* A, size_t lda, int base, int cL,
                                  int mk, Real* P, int ldp, int slab) {
    const int bpl = gridDim.x / cL;                 // blocks per lane
    const int j   = blockIdx.x / bpl;               // this block's lane
    const int idx = (blockIdx.x - j * bpl) * blockDim.x + threadIdx.x;
    if (idx >= NB * NB) return;
    const int slab_start = j * base;
    const int nrows_lane = (j < cL - 1) ? base : (mk - slab_start);
    const int nrows = nrows_lane < NB ? nrows_lane : NB;
    const int c = idx / NB, r = idx % NB;
    P[size_t(j * slab + r) + size_t(c) * ldp] =
        (r <= c && r < nrows) ? A[size_t(slab_start + r) + size_t(c) * lda] : Real(0);
}

// Copy upper triangle (NB×NB) from src to dst (different leading dims).
// Used by panel_25d Phase 4a: copy root R from Pbuf to A's top-NB rows.
template<typename Real, int NB>
__global__ void k_copy_upper_tri(Real* dst, size_t ldd, const Real* src, int lds) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NB * NB) return;
    const int c = idx / NB, r = idx % NB;
    if (r <= c)
        dst[size_t(c) * ldd + r] = src[size_t(c) * lds + r];
}

// Element-wise ops for the merge apply (Stage 2 of two-level WY).
// op=0: W[r,c] += A[row+r, colA+c]  (W1 = B_top + W_μ^T·B_bot, after the GEMM)
// op=1: A[row+r, colA+c] -= W[r,c]  (B_top -= W2)
template<typename Real, int NB>
__global__ void k_merge_elem(Real* A, size_t lda, int row, int colA,
                             Real* W, int ldw, int nc, int op) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NB * nc) return;
    const int c = idx / NB, r = idx % NB;
    if (op == 0)
        W[c * ldw + r] += A[size_t(colA + c) * lda + row + r];
    else
        A[size_t(colA + c) * lda + row + r] -= W[c * ldw + r];
}

// Unpivoted LU of the top NB×NB of thinQ with on-the-fly sign S (BDGJNS recon, spec §11):
// step c: S_c = −sgn(M[c,c]); M[c,c] −= S_c; eliminate. In-place Y1 (unit lower) + U.
template<typename Real, int NB>
__device__ __noinline__ void d_lu_sign_body(unsigned char* smraw, Real* QT, int ldq, Real* Sg) {
    Real* M   = reinterpret_cast<Real*>(smraw);
    Real* Ssh = M + NB * NB;
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int idx = tid; idx < NB * NB; idx += nth) {
        int c = idx / NB, r = idx % NB;
        M[idx] = QT[size_t(c) * ldq + r];
    }
    __syncthreads();
    // row-owner elimination: thread r scales its L entry and updates its whole row —
    // one barrier per step, no cross-thread hazards (pivot row c is read-only this step).
    for (int c = 0; c < NB; ++c) {
        const Real mcc = M[c + c * NB];
        __syncthreads();   // all threads read M[c,c] before thread c overwrites it
        const Real s = (mcc >= Real(0)) ? Real(-1) : Real(1);
        const Real p = mcc - s;
        const Real inv = Real(1) / p;
        if (tid == c) { Ssh[c] = s; M[c + c * NB] = p; }
        const int r = tid;
        if (r > c && r < NB) {
            Real L = M[r + c * NB] * inv;
            M[r + c * NB] = L;
            for (int j = c + 1; j < NB; ++j) M[r + j * NB] -= L * M[c + j * NB];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < NB * NB; idx += nth) {
        int c = idx / NB, r = idx % NB;
        QT[size_t(c) * ldq + r] = M[idx];
    }
    for (int i = tid; i < NB; i += nth) Sg[i] = Ssh[i];
}

template<typename Real, int NB>
__global__ void k_lu_sign(Real* QT, int ldq, Real* Sg) {
    extern __shared__ __align__(16) unsigned char smraw[];
    d_lu_sign_body<Real, NB>(smraw, QT, ldq, Sg);
}

// ================================================================ fused panel (spec §11)
// One cooperative kernel = the whole intermediary factorization of a panel: leaf QR,
// ⊕-merge tree, T factors, thin-Q tree-apply, BDGJNS reconstruction (LU + both triangular
// solves in-kernel) and output assembly, with grid.sync() replacing kernel boundaries.
// The tree state never leaves L2; the O(1)-per-panel expensive-level episode count of
// §8/§12 is realized literally: one launch per panel. Used for the small/mid-n regime
// (a cooperative grid needs full residency, which would serialize against the far-rest
// GEMM overlap that matters at large n — there the per-level path + look-ahead wins).



// ---------------- wavefront up-sweep step (left-looking ⊕ node, one mini-panel) ----------------
// Node state across steps: Bt (accumulating bottom V, smem-persistent), tau, T16s (one 16×16
// per mini-panel). Incoming 16-column block is ingested from the children's published R
// triangles, receives the node's previous reflectors (left-looking), is factored, and the
// finished R columns are published in place over child_i's triangle (the parent reads them
// next step). The monoid ⊕ is unchanged — this is §3's scheduling freedom at sub-merge grain.
template<typename Real, int SLAB, int NB, int PW = 16>
__device__ void d_wave_step(Real* Bt /*(NB+1)×NB persistent*/, Real* Ct /*(NB+1)×PW*/,
                            Real* tau, Real* T16s /*8·PW²*/, Real* Wsc /*2·PW²+PW²*/,
                            Real* P, int ldp, int2 nd, int s) {
    constexpr int LDS = NB + 1;
    const int tid = threadIdx.x, nth = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const int p = s * PW, pe = p + PW;
    Real* Ri = P + size_t(nd.x) * SLAB;
    Real* Rj = P + size_t(nd.y) * SLAB;
    Real* W  = Wsc;               // PW×PW
    Real* W2 = Wsc + PW * PW;     // PW×PW
    Real* G  = Wsc + 2 * PW * PW; // PW×PW
    // ---- ingest child columns [p,pe): top from child_i (full triangle rows), bottom from child_j
    for (int idx = tid; idx < NB * PW; idx += nth) {
        int c = idx / NB, r = idx % NB;
        Ct[r + c * LDS] = (r <= p + c) ? Ri[size_t(p + c) * ldp + r] : Real(0);
        Bt[r + (p + c) * LDS] = (r <= p + c) ? Rj[size_t(p + c) * ldp + r] : Real(0);
    }
    __syncthreads();
    // ---- left-looking: apply previous mini-panels q = 0..s-1 (Q^T = Π (I − V_q T_q^T V_q^T))
    for (int q = 0; q < s; ++q) {
        const int qb = q * PW, qe = qb + PW;
        const Real* Tq = T16s + q * PW * PW;
        // W = V_q^T C = Ct rows [qb,qe) + Bt_q^T · C_bot(rows 0..qe)
        for (int idx = tid; idx < PW * PW; idx += nth) {
            int b = idx / PW, a = idx % PW;
            Real acc = Ct[(qb + a) + b * LDS];
            for (int k = 0; k <= qb + a; ++k)
                acc += Bt[k + (qb + a) * LDS] * Bt[k + (p + b) * LDS];
            W[a + b * PW] = acc;
        }
        __syncthreads();
        // W2 = T_q^T W   (T upper → T^T lower)
        for (int idx = tid; idx < PW * PW; idx += nth) {
            int b = idx / PW, a = idx % PW;
            Real acc = Real(0);
            for (int k = 0; k <= a; ++k) acc += Tq[k + a * PW] * W[k + b * PW];
            W2[a + b * PW] = acc;
        }
        __syncthreads();
        // C −= V_q W2 : Ct rows [qb,qe) −= W2 ; C_bot rows 0..qe −= Bt_q·W2
        for (int idx = tid; idx < PW * PW; idx += nth) {
            int b = idx / PW, a = idx % PW;
            Ct[(qb + a) + b * LDS] -= W2[a + b * PW];
        }
        for (int idx = tid; idx < qe * PW; idx += nth) {
            int b = idx / qe, r = idx % qe;
            Real acc = Real(0);
            for (int k = (r > qb ? r - qb : 0); k < PW; ++k)   // v_{qb+k} support rows ≤ qb+k
                acc += Bt[r + (qb + k) * LDS] * W2[k + b * PW];
            Bt[r + (p + b) * LDS] -= acc;
        }
        __syncthreads();
    }
    // ---- factor the 16 columns (reflector j: pivot Ct[j], bottom Bt rows 0..j)
    for (int jj = 0; jj < PW; ++jj) {
        const int j = p + jj;
        if (tid < 32) {
            Real sig = Real(0);
            for (int i = lane; i <= j; i += 32) sig += Bt[i + j * LDS] * Bt[i + j * LDS];
            #pragma unroll
            for (int o = 16; o; o >>= 1) sig += __shfl_xor_sync(0xffffffffu, sig, o);
            Real alpha = Ct[j + jj * LDS], tj = Real(0), vs = Real(0), beta = alpha;
            if (sig != Real(0)) {
                Real mu = sqrt(alpha * alpha + sig);
                beta = (alpha >= Real(0)) ? -mu : mu;
                tj = (beta - alpha) / beta;
                vs = Real(1) / (alpha - beta);
            }
            __syncwarp();
            if (lane == 0) { tau[j] = tj; Ct[j + jj * LDS] = beta; }
            for (int i = lane; i <= j; i += 32) Bt[i + j * LDS] *= vs;
        }
        __syncthreads();
        {
            Real tj = tau[j];
            for (int c = jj + 1 + warp; c < PW; c += (int)(blockDim.x >> 5)) {
                Real d = (lane == 0) ? Ct[j + c * LDS] : Real(0);
                for (int i = lane; i <= j; i += 32) d += Bt[i + j * LDS] * Bt[i + (p + c) * LDS];
                #pragma unroll
                for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
                Real w = tj * d;
                if (lane == 0) Ct[j + c * LDS] -= w;
                for (int i = lane; i <= j; i += 32) Bt[i + (p + c) * LDS] -= Bt[i + j * LDS] * w;
            }
        }
        __syncthreads();
    }
    // ---- T16 for this mini-panel (Gram of new bottom columns + forward recurrence)
    for (int idx = tid; idx < PW * PW; idx += nth) {
        int c = idx / PW, r = idx % PW;
        Real acc = Real(0);
        for (int k = 0; k <= p + r; ++k) acc += Bt[k + (p + r) * LDS] * Bt[k + (p + c) * LDS];
        G[r + c * PW] = acc;
    }
    __syncthreads();
    Real* Ts = T16s + s * PW * PW;
    if (tid < 32) {
        for (int j = 0; j < PW; ++j) {
            Real tj = tau[p + j];
            Real acc = Real(0);
            if (lane < j) for (int k = lane; k < j; ++k) acc += Ts[lane + k * PW] * G[k + j * PW];
            __syncwarp();
            if (lane < j) Ts[lane + j * PW] = -tj * acc;
            else if (lane == j) Ts[j + j * PW] = tj;
            else if (lane < PW) Ts[lane + j * PW] = Real(0);
            __syncwarp();
        }
    }
    __syncthreads();
    // ---- publish R columns [p,pe) over child_i's triangle (parent consumes next step)
    for (int idx = tid; idx < NB * PW; idx += nth) {
        int c = idx / NB, r = idx % NB;
        if (r <= p + c) Ri[size_t(p + c) * ldp + r] = Ct[r + c * LDS];
    }
    __syncthreads();
}


// Top-of-tree wavefront (per-level path, large panels): the last levels of a big ⊕-tree
// have few nodes and are pure latency — run them as one small cooperative wavefront
// (≤63 blocks, so the concurrent far-rest GEMM keeps most of the machine).

// Grid barrier that works in BOTH launch modes: cooperative (cg) or a normal launch with
// guaranteed residency (persistent kernel launched first on an empty device). The manual
// path is the classic arrive-generation barrier; cooperative launches on this driver
// (570.x) block ALL concurrently-submitted work, so the persistent pipeline uses normal
// launches + manual barriers to genuinely co-run with update GEMMs.
struct GridBar {
    cooperative_groups::grid_group* cg;
    volatile unsigned *cnt, *gen;
    unsigned nb;
    __device__ void sync() {
        if (cg) { cg->sync(); return; }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned g = *gen;
            unsigned a = atomicAdd((unsigned*)cnt, 1u);
            if (a == nb - 1) {
                *cnt = 0;
                __threadfence();
                atomicAdd((unsigned*)gen, 1u);
            } else {
                while (atomicAdd((unsigned*)gen, 0u) == g) __nanosleep(1);
            }
        }
        __syncthreads();
    }
};

template<typename Real>
struct WaveTopArgs {
    Real* Pbuf; int mpad;
    const int2* nodes;            // panel node array (levels concatenated)
    int2 lvl[20]; int nlv, l0;    // wavefront covers levels l0..nlv-1
    Real *Vm, *taum;
};

template<typename Real, int SLAB, int NB, int PW = 16>
__global__ void k_wave_top(WaveTopArgs<Real> a) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char smraw[];
    constexpr int LDS = NB + 1;
    Real* Bt   = reinterpret_cast<Real*>(smraw);
    Real* Ct   = Bt + LDS * NB;
    Real* wtau = Ct + LDS * 16;
    Real* T16s = wtau + NB;
    Real* Wsc  = T16s + (NB / 16) * 16 * 16;
    const int tid = threadIdx.x, nth = blockDim.x, bid = blockIdx.x;
    const int base = a.lvl[a.l0].x;
    const int myNode = base + bid;
    const int total = a.lvl[a.nlv - 1].x + a.lvl[a.nlv - 1].y;
    int myLvl = -1;
    if (myNode < total)
        for (int l = a.l0; l < a.nlv; ++l)
            if (myNode >= a.lvl[l].x && myNode < a.lvl[l].x + a.lvl[l].y) { myLvl = l; break; }
    const int TSTEPS = (a.nlv - a.l0) + NB / 16 - 1;
    for (int t = 0; t < TSTEPS; ++t) {
        if (myLvl >= 0) {
            int s = t - (myLvl - a.l0);
            if (s >= 0 && s < NB / 16)
                d_wave_step<Real, SLAB, NB>(Bt, Ct, wtau, T16s, Wsc,
                                            a.Pbuf, a.mpad, a.nodes[myNode], s);
        }
        grid.sync();
    }
    if (myLvl >= 0) {
        Real* Vout = a.Vm + size_t(myNode) * NB * NB;
        for (int idx = tid; idx < NB * NB; idx += nth) {
            int c = idx / NB, r = idx % NB;
            Vout[idx] = Bt[r + c * LDS];
        }
        for (int i = tid; i < NB; i += nth) a.taum[myNode * NB + i] = wtau[i];
    }
}

template<typename Real>
struct PanelArgs {
    Real* A; size_t lda; int row0, mk, mpad, N;
    Real *Pbuf, *QT, *QS, *Vm, *TmM, *TmL, *tauL, *taum;
    const int2* nodes;            // this panel's nodes, levels concatenated
    int2 lvl[20];                 // per level: (offset within panel nodes, count)
    int nlv;
    Real *Sg, *Tpan, *SPV; size_t ldspv; int spvcol;
    // The panel's R, saved out of Pbuf before phase 5 overwrites it. Needed whenever
    // QT aliases Pbuf -- 31:415 puts V in tril(A), so the compact caller has no second
    // m x b to spare and aliases them. NB*NB words. May be null when QT is distinct.
    Real *Rroot;
};

template<typename Real, int SLAB, int NB, int PW = 16>
__device__ void d_panel_body(GridBar gb, PanelArgs<Real>& a,
                             unsigned char* smraw) {
    const int tid = threadIdx.x, nth = blockDim.x, lane = tid & 31, warp = tid >> 5;
    const int bid = blockIdx.x, nbl = gridDim.x;
    const int mpad = a.mpad, N = a.N;
    const size_t gth = size_t(nbl) * nth;

    // ---- phase 0: load panel (zero-padded)
    {
        Real* Acol = a.A;
        for (size_t g = size_t(bid) * nth + tid; g < size_t(mpad) * NB; g += gth) {
            int c = int(g / mpad), r = int(g % mpad);
            a.Pbuf[size_t(c) * mpad + r] = (r < a.mk) ? Acol[size_t(c) * a.lda + r] : Real(0);
        }
    }
    gb.sync();

    const int nmergesAll = a.nlv ? (a.lvl[a.nlv - 1].x + a.lvl[a.nlv - 1].y) : 0;
    int l0 = a.nlv;
    for (int l = 0; l < a.nlv; ++l)
        if (nmergesAll - a.lvl[l].x <= nbl) { l0 = l; break; }
    const bool leafInWave = (l0 == 0) && (nmergesAll + N <= nbl) && nmergesAll > 0;
    // ---- phase 1: leaves (skipped when leaves ride the wavefront as level −1)
    if (!leafInWave)
    for (int leaf = bid; leaf < N; leaf += nbl) {
        Real* As  = reinterpret_cast<Real*>(smraw);
        Real* Vp  = As + (SLAB + 1) * NB;
        Real* W   = Vp + (SLAB + 1) * 16;
        Real* W2  = W + 16 * NB;
        Real* G16 = W2 + 16 * NB;
        Real* T16 = G16 + 16 * 16;
        Real* tau = T16 + 16 * 16;
        Real* srcp = a.Pbuf + size_t(leaf) * SLAB;
        for (int idx = tid; idx < SLAB * NB; idx += nth) {
            int c = idx / SLAB, r = idx % SLAB;
            As[c * (SLAB + 1) + r] = srcp[size_t(c) * mpad + r];
        }
        __syncthreads();
        bqr_core<Real, SLAB, NB>(As, tau, Vp, W, W2, G16, T16);
        for (int idx = tid; idx < SLAB * NB; idx += nth) {
            int c = idx / SLAB, r = idx % SLAB;
            srcp[size_t(c) * mpad + r] = As[c * (SLAB + 1) + r];
        }
        for (int i = tid; i < NB; i += nth) a.tauL[leaf * NB + i] = tau[i];
        __syncthreads();
    }
    gb.sync();


    // ---- phase 3: ⊕ tree — wavefront up-sweep (spec §3 schedule freedom): persistent
    // block↔node binding, left-looking nodes, node at level l runs mini-panel s at
    // wavefront time t = l+s; ~ (levels+8) steps instead of levels×(full node).
    const int nmerges = nmergesAll;
    for (int l = 0; l < l0; ++l) {          // batched lower levels (plenty of nodes)
        const int off = a.lvl[l].x, cnt = a.lvl[l].y;
        for (int nid = bid; nid < cnt; nid += nbl) {
            d_merge_body<Real, SLAB, NB, PW>(smraw, a.Pbuf, mpad, a.nodes[off + nid],
                                             a.Vm + size_t(off + nid) * NB * NB,
                                             a.taum + size_t(off + nid) * NB);
            __syncthreads();
        }
        gb.sync();
    }
    if (l0 < a.nlv) {
        constexpr int LDS = NB + 1;
        Real* Bt   = reinterpret_cast<Real*>(smraw);            // merge state (persistent)
        Real* Ct   = Bt + LDS * NB;
        Real* wtau = Ct + LDS * 16;
        Real* T16s = wtau + NB;
        Real* Wsc  = T16s + (NB / 16) * 16 * 16;
        Real* lAs  = reinterpret_cast<Real*>(smraw);            // leaf state (persistent)
        Real* lVp  = lAs + (SLAB + 1) * NB;
        Real* lW   = lVp + (SLAB + 1) * 16;
        Real* lW2  = lW + 16 * NB;
        Real* lG   = lW2 + 16 * NB;
        Real* lT   = lG + 16 * 16;
        Real* ltau = lT + 16 * 16;
        const int wbase = a.lvl[l0].x, wcount = nmerges - wbase;
        const int myNode = wbase + bid;
        int myLvl = -1, myLeaf = -1;
        if (bid < wcount) {
            for (int l = l0; l < a.nlv; ++l)
                if (myNode >= a.lvl[l].x && myNode < a.lvl[l].x + a.lvl[l].y) { myLvl = l; break; }
        } else if (leafInWave && bid < wcount + N) {
            myLeaf = bid - wcount;
        }
        if (myLeaf >= 0) {
            Real* srcp = a.Pbuf + size_t(myLeaf) * SLAB;
            for (int idx = tid; idx < SLAB * NB; idx += nth) {
                int c = idx / SLAB, r = idx % SLAB;
                lAs[c * (SLAB + 1) + r] = srcp[size_t(c) * mpad + r];
            }
            __syncthreads();
        }
        const int shift = leafInWave ? 1 : 0;
        const int TSTEPS = (a.nlv - l0) + shift + NB / 16 - 1;
        for (int t = 0; t < TSTEPS; ++t) {
            if (myLeaf >= 0 && t < NB / 16) {
                const int p = t * 16;
                d_bqr_step<Real, SLAB, NB>(lAs, ltau, lVp, lW, lW2, lG, lT, p);
                __syncthreads();
                Real* srcp = a.Pbuf + size_t(myLeaf) * SLAB;
                for (int idx = tid; idx < 16 * NB; idx += nth) {
                    int c = idx / NB, r = idx % NB;
                    if (r <= p + c) srcp[size_t(p + c) * mpad + r] = lAs[(p + c) * (SLAB + 1) + r];
                }
                __syncthreads();
            }
            if (myLvl >= 0) {
                int s = t - (myLvl - l0) - shift;
                if (s >= 0 && s < NB / 16)
                    d_wave_step<Real, SLAB, NB>(Bt, Ct, wtau, T16s, Wsc,
                                                a.Pbuf, mpad, a.nodes[myNode], s);
            }
            gb.sync();
        }
        if (myLvl >= 0) {
            Real* Vout = a.Vm + size_t(myNode) * NB * NB;
            for (int idx = tid; idx < NB * NB; idx += nth) {
                int c = idx / NB, r = idx % NB;
                Vout[idx] = Bt[r + c * LDS];
            }
            for (int i = tid; i < NB; i += nth) a.taum[myNode * NB + i] = wtau[i];
        }
        if (myLeaf >= 0) {   // V below diag + taus (R already published)
            Real* srcp = a.Pbuf + size_t(myLeaf) * SLAB;
            for (int idx = tid; idx < SLAB * NB; idx += nth) {
                int c = idx / SLAB, r = idx % SLAB;
                if (r > c) srcp[size_t(c) * mpad + r] = lAs[c * (SLAB + 1) + r];
            }
            for (int i = tid; i < NB; i += nth) a.tauL[myLeaf * NB + i] = ltau[i];
        }
        gb.sync();
    }
    // ---- leaf larft (after the up-sweep: in wave mode leaf V lands in Pbuf at its end)
    for (int leaf = bid; leaf < N; leaf += nbl) {
        d_larft_body<Real, SLAB, NB, false>(smraw, a.Pbuf, mpad, nullptr, a.tauL, a.TmL, leaf, 0);
        __syncthreads();
    }
    gb.sync();
    {   // merge T factors: fast if the wavefront stored T16s, else the recurrence
        for (int nid = bid; nid < nmerges; nid += nbl) {
            d_larft_body<Real, SLAB, NB, true>(smraw, nullptr, 0, a.Vm, a.taum, a.TmM, nid, 0);
            __syncthreads();
        }
        gb.sync();
    }

    // ---- phase 4: down-sweep (root identity, then levels top→bottom)
    if (bid == 0)
        for (int idx = tid; idx < NB * NB; idx += nth)
            a.QS[idx] = (idx % NB == idx / NB) ? Real(1) : Real(0);
    gb.sync();
    for (int l = a.nlv - 1; l >= 0; --l) {
        const int off = a.lvl[l].x, cnt = a.lvl[l].y;
        for (int nid = bid; nid < cnt; nid += nbl) {
            d_dsweep_body<Real, NB>(smraw, a.QS, a.TmM, a.Vm, a.nodes[off + nid], off + nid);
            __syncthreads();
        }
        gb.sync();
    }

    // ---- phase 4b: save the panel's R before phase 5 writes the thin Q over it ----
    // The merge tree leaves R in Pbuf's top NB rows. Phase 9 reads it back for the
    // R = S * R_tsqr output, but phase 5 writes QT there when QT aliases Pbuf. NB*NB
    // words copied once per panel, against the m x b the alias saves.
    if (a.Rroot) {
        for (int idx = size_t(bid) * nth + tid; idx < size_t(NB) * NB; idx += gth) {
            const int c = int(idx / NB), r = int(idx % NB);
            a.Rroot[size_t(c) * NB + r] = a.Pbuf[size_t(c) * mpad + r];
        }
        gb.sync();
    }

    // ---- phase 5: leaf apply → thin Q
    for (int leaf = bid; leaf < N; leaf += nbl) {
        d_leaf_apply_body<Real, SLAB, NB>(smraw, a.Pbuf, mpad, a.TmL, a.QS, a.QT, mpad, leaf);
        __syncthreads();
    }
    gb.sync();

    // ---- phase 6: BDGJNS — sign LU of thinQ top (block 0)
    if (bid == 0) d_lu_sign_body<Real, NB>(smraw, a.QT, mpad, a.Sg);
    gb.sync();

    // ---- phase 7: Y_below = M_below · U^{-1} (row-parallel forward substitution)
    {
        Real* Us  = reinterpret_cast<Real*>(smraw);            // (NB+1)×NB padded
        Real* inv = Us + (NB + 1) * NB;
        for (int idx = tid; idx < NB * NB; idx += nth) {
            int c = idx / NB, r = idx % NB;
            Us[r + c * (NB + 1)] = (r <= c) ? a.QT[size_t(c) * mpad + r] : Real(0);
        }
        __syncthreads();
        for (int j = tid; j < NB; j += nth) inv[j] = Real(1) / Us[j + j * (NB + 1)];
        __syncthreads();
        Real x[NB];
        for (size_t rg = size_t(bid) * nth + tid; rg < size_t(mpad - NB); rg += gth) {
            Real* Mr = a.QT + NB + rg;
            #pragma unroll 4
            for (int j = 0; j < NB; ++j) {
                Real acc = Mr[size_t(j) * mpad];
                for (int k = 0; k < j; ++k) acc -= x[k] * Us[k + j * (NB + 1)];
                x[j] = acc * inv[j];
            }
            for (int j = 0; j < NB; ++j) Mr[size_t(j) * mpad] = x[j];
        }
    }
    gb.sync();

    // ---- phase 8: T_panel = −U·S·Y1^{-T} (block 0; row-parallel forward substitution)
    if (bid == 0) {
        Real* Ls = reinterpret_cast<Real*>(smraw);             // Y1 unit-lower, padded
        Real* Bs = Ls + (NB + 1) * NB;                         // RHS −U·diag(S), padded
        for (int idx = tid; idx < NB * NB; idx += nth) {
            int c = idx / NB, r = idx % NB;
            Real q = a.QT[size_t(c) * mpad + r];
            Ls[r + c * (NB + 1)] = (r > c) ? q : (r == c ? Real(1) : Real(0));
            Bs[r + c * (NB + 1)] = (r <= c) ? -q * a.Sg[c] : Real(0);
        }
        __syncthreads();
        // per row i: solve T[i,:]·Y1^T = B[i,:]  ⇒  T[i,c] = B[i,c] − Σ_{j<c} T[i,j]·Y1[c,j]
        if (tid < NB) {
            Real x[NB];
            for (int c = 0; c < NB; ++c) {
                Real acc = Bs[tid + c * (NB + 1)];
                for (int j = 0; j < c; ++j) acc -= x[j] * Ls[c + j * (NB + 1)];
                x[c] = acc;
            }
            for (int c = 0; c < NB; ++c) a.Tpan[tid + c * NB] = x[c];
        }
        __syncthreads();
    }
    gb.sync();

    // ---- phase 9: outputs — Y into SPV, V below diag and R = S·R_tsqr into A
    for (size_t g = size_t(bid) * nth + tid; g < size_t(a.mk) * NB; g += gth) {
        const int c = int(g / a.mk), r = int(g % a.mk);
        Real y;
        if (r < NB) y = (r > c) ? a.QT[size_t(c) * mpad + r] : (r == c ? Real(1) : Real(0));
        else        y = a.QT[size_t(c) * mpad + r];
        // SPV is the masked-Y side output. The compact caller has none -- 31:415 puts
        // V in tril(A) -- and passing nullptr here faulted at exactly this store
        // (compute-sanitizer: invalid 8-byte write to 0x1b00, i.e. null + offset).
        if (a.SPV) a.SPV[size_t(a.spvcol + c) * a.ldspv + a.row0 + r] = y;
        Real* Ac = a.A + size_t(c) * a.lda;
        if (r > c) Ac[r] = y;
        else       Ac[r] = a.Sg[r] * (a.Rroot ? a.Rroot[size_t(c) * NB + r]
                                              : a.Pbuf[size_t(c) * mpad + r]);
    }
}

template<typename Real, int SLAB, int NB, int PW = 16>
__global__ void k_panel_fused(PanelArgs<Real> a) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char smraw[];
    GridBar gb{&grid, nullptr, nullptr, 0};
    d_panel_body<Real, SLAB, NB, PW>(gb, a, smraw);
}

// Persistent panel pipeline (spec §11 "persistent kernels own the SMs for the whole
// factorization"): ONE cooperative launch executes every panel; the host signals panel k
// runnable (ready[k], after narrow-update k−1) and consumes fin[k] via a spin kernel.
// Update GEMMs stream through the remaining SM slots at all times — φ-hiding by structure.
template<typename Real, int SLAB, int NB, int PW = 16>
__global__ void k_panel_persist(PanelArgs<Real>* args, int npanels,
                                volatile int* ready, int* fin,
                                volatile unsigned* bcnt, volatile unsigned* bgen) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();       // cooperative launch → all blocks co-resident
    extern __shared__ __align__(16) unsigned char smraw[];
    GridBar gb{&grid, nullptr, nullptr, 0};      // real grid barrier (was fragile manual barrier)
    (void)bcnt; (void)bgen;
    if (threadIdx.x == 0) {                     // alive doorbell: every block reports entry
        __threadfence_system();
        atomicAdd(&fin[npanels], 1);            // fin[npanels] = resident-block count
    }
    for (int k = 0; k < npanels; ++k) {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            while (ready[k] == 0) __nanosleep(1);   // volatile mapped read (no PCIe atomics)
        gb.sync();
        PanelArgs<Real> a = args[k];
        d_panel_body<Real, SLAB, NB, PW>(gb, a, smraw);
        gb.sync();
        __threadfence_system();
        if (blockIdx.x == 0 && threadIdx.x == 0) fin[k] = 1;
    }
}

// ================================================================ domino panel (spec §5)
// ONE cooperative kernel factors the whole m_k×NB panel column-by-column — the degenerate
// sequential ⊕-tree, licensed by the monoid's schedule freedom (§3) and the serial
// schedule (§5); pure Householder, never Gram. At square n the layered tree is dormant
// (§8.4, c_L=1) and its per-node latency (merges + larfts + BDGJNS reconstruction ≈ 55%
// of small-n time) is pure overhead; this kernel removes ALL of it: native (V,T) out,
// nothing to reconstruct. Critical path = 1 grid-wide event per column (batched dots,
// ~1 µs grid.sync) + 4 events per 16-col WY boundary ≈ 160 events/panel.
//
// Reflectors are kept UNSCALED in place: u = a_j with implicit pivot f = α−β; v = u/f,
// τ̃ = τ/f² = −1/(β·f), so H·a_c = a_c − (τ̃·d_c)·ũ with d_c = f·a[j,c] + Σ_{r>j} u_r·a_rc.
// No per-column scaling pass; V is scaled once at writeout. T̃ = D·T·D (D = diag(1/f))
// obeys the SAME forward recurrence with (τ̃, G̃ = ṼᵀṼ) and compounds across minipanels
// as T̃_off = −T̃_pre·G̃c·T̃_16; Tpan = D⁻¹T̃D⁻¹ recovered at writeout.
//
// Cross-block protocol per column j (ONE grid.sync):
//   inputs, all finalized in region j−1: sig[j] (fresh Σ u², not downdated), α = A[j,j],
//   dslot[jj][c] (u-part dots), prow[j&1][c] (pivot-row snapshot — avoids reading A[j,c]
//   which the owner overwrites this region). Region j: everyone redundantly computes
//   β, f, τ̃, s_c; owner rows apply reflector j to minipanel cols j+1..pe−1; the col-j+1
//   pass stages v_{j+1} in smem, the remaining cols accumulate d_c for column j+1 plus
//   σ_{j+1}; row-(j+1) owner snapshots prow[(j+1)&1].
template<typename Real>
struct DomArgs {
    Real* A; size_t lda; int mk, rpb, row0;
    Real* SPV; size_t ldspv; int spvcol;
    Real* Tpan;                       // NB×NB out
    Real *tauT, *fv, *bet, *sig;      // NB each: τ̃, f=α−β, β, σ
    Real *dslot;                      // PW×PW   [jj*PW + c]
    double *sigD, *dslotD;            // fp64 shadows (see dcolP: these are no longer atomic)
    // A6 (31:246) requires DETERMINISTIC W reductions and 31:434's rung-1 Combine is
    // "stack-HH R, deterministic W". The per-column dots and sigma used to be summed
    // with device-scope fp64 atomicAdd, whose order is undefined: job 9715 measured
    // 3.8M of 4.2M words differing BITWISE run to run at 2048^2 (max rel 1.2e-9) and
    // 63M of 67M at 8192^2 (1.62e-7) -- far above rounding noise, because the
    // nondeterministic dots compound through 64 sequential reflectors.
    //
    // Instead each block publishes its own partial here and every block reduces them
    // in FIXED block order b = 0..nbl-1. Same mechanism the kernel already uses for its
    // boundary partials (Pp -> B2a). Double-buffered by column parity: region jj reads
    // generation jj&1 and writes (jj+1)&1, and there is already a gsync between
    // regions, so no block can overwrite a generation another block is still reading.
    //   layout: dcolP[gen][block][slot],  slot 0..PW-1 = dots, slot PW = sigma
    double *dcolP;
    int dcolStride;                   // block capacity: dcolP[gen][slot][block], stride here
    Real *prow;                       // 2×PW    pivot-row snapshots (parity)
    Real *Pp;                         // nbl×NB×PW  per-block boundary partials [b][t][q]
    Real *Psum;                       // NB×PW      reduced [t][q]
    Real *T16g;                       // PW×PW      T̃16 broadcast (block 0 → all)
    unsigned long long* tphase;       // optional [8]: init,cols,B1,B2a,B2b,B3,writeout
    int nslab;                        // true slab count (cdiv(mk, rpb)); > gridDim.x when grid-stride
};

// v3: the minipanel lives in shared memory, double-buffered across minipanels — the 16
// column regions are all-LDS (gmem carries only the prow/dslot/sig scalar protocol);
// B3 stages the next minipanel into the other buffer while applying the WY update; B1
// flushes the finished one and masks it in place into the ũ slab. Block 0 keeps the
// running T̃ in smem. Phase bodies are __noinline__ so one phase's register appetite
// doesn't spill the others.
template<typename Real, int NB, int PW>
struct DomCtx {
    DomArgs<Real> a;
    Real *msm, *msmN;   // current / next minipanel buffer (rpb×PW each, ping-pong)
    Real *W2, *Ts, *ssm, *fsm, *gsm;
    // scalar protocol (dslot/prow/σ/f/β/τ̃): block 0's smem via DSMEM in cluster mode
    // (~30ns vs ~300ns L2), the gmem buffers in cooperative mode
    Real *dslot, *prow, *sig, *fv, *bet, *tauT;
    // A6's per-block column partials, [2][PW+1] at a FIXED smem offset in every block
    // so cluster peers can read them by rank. Non-null only under the cluster carrier;
    // the gmem carrier keeps them in DomArgs::dcolP. 10:S6.3: "row-partial W blocks
    // reduce deterministically in DSM" -- reducing them out of global memory, as the
    // first cut did, is the same computation on the slow side of the boundary the
    // carrier exists to exploit (DSMEM ~30 ns against L2 ~300 ns).
    // MEASURED NEGATIVE, recorded rather than kept. 10:S6.3 says "row-partial W blocks
    // reduce deterministically in DSM", and A6's reduction reads its partials from
    // GLOBAL memory -- on the cluster carrier that is the same computation on the slow
    // side of the boundary the carrier exists for (DSMEM ~30 ns vs L2 ~300 ns), and
    // 2048^2 / 4096^2, the cells still under cuSOLVER, are exactly the ones that select
    // it. Publishing partials at a fixed smem offset and reducing them across peers via
    // map_shared_rank (same fixed-order butterfly, so A6 held bitwise at all four
    // shapes) measured geomean 1.2208 against 1.2336 -- nothing at the cluster cells,
    // a small loss at the gmem cells from 272 B of extra smem. Job 9757.
    //
    // Same answer as every other placement: the per-column cost is the barrier and the
    // serial reflector chain, not where the partials live.
    // [r0,r1) is the resident shared-memory row interval.  A capacity-overflow
    // tail, when present, lives coherently in A over [r1,r1all).  Keeping the
    // distinction explicit lets one resident CTA carry more rows than fit in
    // shared memory without grid-striding the complete 16-column state.
    int tid, nth, lane, warp, NW, bid, nbl, r0, r1, r1all;
};

// one column region: form reflector j from the published (σ, α = prow[jj], dots), apply
// it to the rest of the minipanel in smem, publish next column's (dots, σ, prow incl. α).
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_dom_region(DomCtx<Real, NB, PW>& c, int p, int jj) {
    const int j = p + jj, rpb = c.a.rpb, r0 = c.r0, r1 = c.r1;
    const int r1all = c.r1all;
    const int tid = c.tid, nth = c.nth, lane = c.lane, warp = c.warp, NW = c.NW;
    const size_t lda = c.a.lda;
    Real* A = c.a.A;
    Real* msm = c.msm;
    Real* ssm = c.ssm;
    double* dslot = c.a.dslotD;
    double* sig = c.a.sigD;
    const Real* pr = c.prow + (j & 1) * PW;
    Real* prN = c.prow + ((j + 1) & 1) * PW;
    // ---- A6: deterministic reduction of the previous region's partials ----
    // Fixed order over blocks, computed redundantly by every block into its own smem,
    // so no cross-block write and no extra barrier: the gsync that already separates
    // regions is what makes the partials visible.
    // ---- A6's fixed-order reduction: ONE WARP PER SLOT ----
    //
    // Four placements were measured (geomean over 10 cells, against 1.3424 for the
    // A6-VIOLATING fp64 atomics this replaces):
    //     per-slot serial, one thread          1.2074   132-deep dependent chain
    //     warp per slot, shared staging        1.2281   <- shipped
    //     folded into the barrier's last arriver 1.2199 delays the release for all
    //     per-thread, no staging               1.0934   512x redundant loads
    // The ordering is stable and the reason is the same each time: this kernel is
    // latency-bound on a serial reflector chain, so cost tracks DEPENDENCY DEPTH times
    // REDUNDANCY, and the warp-cooperative form minimises their product.
    //
    // Layout is [gen][slot][block], so the lanes striding over blocks read CONSECUTIVE
    // doubles. Determinism: a fixed lane stride and a fixed butterfly over fixed data,
    // so every block computes the same sum bitwise on every run. A6 (31:246) asks for
    // determinism, not for a particular summation order.
    {
        const double* Pc = c.a.dcolP
                         + (size_t)(jj & 1) * (PW + 1) * c.a.dcolStride;
        for (int e = warp; e < PW + 1; e += NW) {
            double acc = 0.0;
            for (int b = lane; b < c.nbl; b += 32)
                acc += Pc[(size_t)e * c.a.dcolStride + b];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) c.gsm[e] = Real(acc);
        }
        __syncthreads();
    }
    const Real sg = c.gsm[PW], al = pr[jj];
    Real beta = al, f = Real(1), tt = Real(0);
    if (sg > Real(0)) {
        Real mu = sqrt(al * al + sg);
        beta = (al >= Real(0)) ? -mu : mu;
        f = al - beta;
        tt = Real(-1) / (beta * f);
    }
    if (tid == 0) { c.bet[j] = beta; c.fv[j] = f; c.tauT[j] = tt; }
    if (tid > jj && tid < PW)
        ssm[tid] = tt * (f * pr[tid] + c.gsm[tid]);
    __syncthreads();
    if (jj + 1 >= PW) return;               // last column: scalars only, WY carries it
    Real* mj  = msm + size_t(jj) * rpb;     // indexed by rr = r − r0
    Real* mc1 = msm + size_t(jj + 1) * rpb;
    const Real s1 = ssm[jj + 1];
    // phase 1: apply reflector j to col jj+1; publish α_{j+1} = new A[j+1,j+1]
    for (int r = max(r0, j) + tid; r < r1; r += nth) {
        Real vj = (r == j) ? f : mj[r - r0];
        Real nv = mc1[r - r0] - s1 * vj;
        mc1[r - r0] = nv;
        if (r == j + 1) prN[jj + 1] = nv;
    }
    // Capacity-overflow rows remain in their compact in-place A storage.  They
    // are owned by this CTA, so the same Householder region can update them
    // directly while the capacity-sized prefix stays resident in shared memory.
    for (int r = max(r1, j) + tid; r < r1all; r += nth) {
        Real vj = A[r + size_t(p + jj) * lda];
        Real nv = A[r + size_t(p + jj + 1) * lda] - s1 * vj;
        A[r + size_t(p + jj + 1) * lda] = nv;
    }
    __syncthreads();
    // phase 2: warp-per-column apply to cols jj+2..15 + dots with v_{j+1};
    // row-(j+1) owner snapshots prow for the next region.
    for (int cc = jj + 2 + warp; cc < PW; cc += NW) {
        Real* mcc = msm + size_t(cc) * rpb;
        const Real sc = ssm[cc];
        Real acc = Real(0);
        for (int r = max(r0, j) + lane; r < r1; r += 32) {
            Real vj = (r == j) ? f : mj[r - r0];
            Real nv = mcc[r - r0] - sc * vj;
            mcc[r - r0] = nv;
            if (r == j + 1) prN[cc] = nv;
            if (r > j + 1) acc += mc1[r - r0] * nv;
        }
        for (int r = max(r1, j + 2) + lane; r < r1all; r += 32) {
            const Real vj = A[r + size_t(p + jj) * lda];
            Real* arc = A + r;
            const Real nv = arc[size_t(p + cc) * lda] - sc * vj;
            arc[size_t(p + cc) * lda] = nv;
            acc += arc[size_t(p + jj + 1) * lda] * nv;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0)
            c.a.dcolP[(size_t)((jj + 1) & 1) * (PW + 1) * c.a.dcolStride
                      + (size_t)cc * c.a.dcolStride + c.bid] = double(acc);
    }
    if (warp == 0) {   // σ_{j+1}
        Real sa = Real(0);
        for (int r = max(r0, j + 2) + lane; r < r1; r += 32) {
            Real v = mc1[r - r0];
            sa += v * v;
        }
        for (int r = max(r1, j + 2) + lane; r < r1all; r += 32) {
            Real v = A[r + size_t(p + jj + 1) * lda];
            sa += v * v;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
        if (lane == 0)
            c.a.dcolP[(size_t)((jj + 1) & 1) * (PW + 1) * c.a.dcolStride
                      + (size_t)PW * c.a.dcolStride + c.bid] = double(sa);
    }
}

// boundary B1: flush the finished minipanel's live rows to gmem, mask it in place into
// the ũ slab (f on pivot, 0 above), then the paired-target dot GEMM
// Pp[b][t][q] = Σ_{r own} ũ_q[r]·src_t[r].
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_dom_b1(DomCtx<Real, NB, PW>& c, int p,
                                      unsigned long long tck) {
    const int pe = p + PW, wt = NB - pe;
    const int rpb = c.a.rpb, r0 = c.r0, r1 = c.r1;
    const int r1all = c.r1all;
    const int tid = c.tid, nth = c.nth, lane = c.lane, warp = c.warp, NW = c.NW;
    const int bid = c.bid;
    const size_t lda = c.a.lda;
    Real* A = c.a.A;
    Real* msm = c.msm;
    Real* ssm = c.ssm;
    Real* Pp = c.a.Pp;
    if (tid < PW) ssm[tid] = c.fv[p + tid];
    const int rb = max(r0, p), span = r1 - rb;   // rows this block touched this minipanel
    for (int e = tid; e < span * PW; e += nth) {
        int q = e / span, r = rb + e % span;
        A[r + size_t(p + q) * lda] = msm[(r - r0) + q * rpb];
    }
    __syncthreads();
    const int re = min(r1, pe);                  // triangle/pivot rows: straddlers only
    if (re > rb) {
        const int spanM = re - rb;
        for (int e = tid; e < spanM * PW; e += nth) {
            int q = e / spanM, r = rb + e % spanM, cq = p + q;
            Real* slot = &msm[(r - r0) + q * rpb];
            Real v = *slot;
            *slot = (r > cq) ? v : (r == cq ? ssm[q] : Real(0));
        }
    }
    __syncthreads();
    if (c.a.tphase && bid == 0 && tid == 0)
        { unsigned long long now = clock64(); atomicAdd(&c.a.tphase[7], now - tck); }
    // two targets per warp pass: ũ smem loads amortized over both
    Real* msmb = msm + (rb - r0);
    for (int t = 2 * warp; t < NB; t += 2 * NW) {
        const Real* srcp[2];
        for (int h = 0; h < 2; ++h) {
            int th = t + h, col; bool own = false;
            if (th < wt) col = pe + th;
            else if (th < wt + p) col = th - wt;
            else { col = p + (th - wt - p); own = true; }
            srcp[h] = own ? (msmb + size_t(col - p) * rpb) : (A + size_t(col) * lda + rb);
        }
        Real ac0[PW], ac1[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) { ac0[q] = Real(0); ac1[q] = Real(0); }
        for (int rr = lane; rr < span; rr += 32) {
            Real sv0 = srcp[0][rr], sv1 = srcp[1][rr];
            #pragma unroll
            for (int q = 0; q < PW; ++q) {
                Real u = msmb[rr + q * rpb];
                ac0[q] += u * sv0;
                ac1[q] += u * sv1;
            }
        }
        // The overflow suffix is already the finalized, unscaled Householder
        // state in A.  Accumulate it into the same per-CTA boundary partial so
        // B2a still reduces exactly one state per resident CTA.
        for (int r = r1 + lane; r < r1all; r += 32) {
            const Real sv0 = A[r + size_t((t < wt) ? (pe + t)
                                      : (t < wt + p ? t - wt : p + (t - wt - p))) * lda];
            const int th1 = t + 1;
            const Real sv1 = A[r + size_t((th1 < wt) ? (pe + th1)
                                      : (th1 < wt + p ? th1 - wt
                                                       : p + (th1 - wt - p))) * lda];
            #pragma unroll
            for (int q = 0; q < PW; ++q) {
                const Real u = A[r + size_t(p + q) * lda];
                ac0[q] += u * sv0;
                ac1[q] += u * sv1;
            }
        }
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) {
                ac0[q] += __shfl_xor_sync(0xffffffffu, ac0[q], o);
                ac1[q] += __shfl_xor_sync(0xffffffffu, ac1[q], o);
            }
        }
        // static-index selection: ac0[lane] would demote the whole array to local memory
        Real out0 = Real(0), out1 = Real(0);
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            out0 = (lane == q) ? ac0[q] : out0;
            out1 = (lane == q) ? ac1[q] : out1;
        }
        if (lane < PW && (span > 0 || r1all > r1)) {
            Pp[(size_t(bid) * NB + t) * PW + lane] = out0;
            Pp[(size_t(bid) * NB + t + 1) * PW + lane] = out1;
        }
    }
    // dslot zeroing moved to k_panel_domino (after b1 slab loop, before B2a) —
    // in grid-stride mode, zeroing inside b1 would corrupt dslot for later slabs.
}

// boundary B2b (block 0): build T̃16 via the τ̃-recurrence over G̃16, place it into the
// running smem T̃ diag, broadcast T̃16 through its gmem slab for B3. The off-diagonal
// compound T̃_off = −T̃_pre·G̃c·T̃16 is DEFERRED to writeout (d_dom_tcompound) — the
// Psum/T̃16 slabs persist per boundary, so nothing else waits on it here.
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_dom_b2b(DomCtx<Real, NB, PW>& c, int p) {
    const int wt = NB - p - PW, tid = c.tid, nth = c.nth, lane = c.lane, warp = c.warp;
    Real *Gs = c.gsm, *Ts = c.Ts, *ssm = c.ssm;
    const Real* Psum = c.a.Psum + size_t(p / PW) * NB * PW;
    Real* T16g = c.a.T16g + size_t(p / PW) * PW * PW;
    for (int e = tid; e < PW * PW; e += nth)
        Gs[e] = Psum[size_t(wt + p + e / PW) * PW + (e % PW)];
    if (tid < PW) ssm[tid] = c.tauT[p + tid];
    __syncthreads();
    if (warp == 0)   // one-warp recurrence: lane i owns row i, __syncwarp only
        for (int jT = 0; jT < PW; ++jT) {
            const Real tj = ssm[jT];
            Real acc = Real(0);
            if (lane < jT)
                for (int k = lane; k < jT; ++k) acc += Ts[lane + k * PW] * Gs[k + jT * PW];
            __syncwarp();
            if (lane < jT) Ts[lane + jT * PW] = -tj * acc;
            else if (lane == jT) Ts[jT + jT * PW] = tj;
            __syncwarp();
        }
    __syncthreads();
    // Ts is block-local and every block now builds it (see the call site): b3 reads it
    // straight out of shared memory instead of round-tripping through T16g and a
    // device barrier. T16g is still published, by block 0 only, because the replicated
    // T assembly at the end of the kernel reads it back.
    if (blockIdx.x == 0)
        for (int e = tid; e < PW * PW; e += nth) {
            int cc = e / PW, r = e % PW;
            T16g[e] = (r <= cc) ? Ts[e] : Real(0);
        }
}

// boundary B3: WY-apply reflectors p..pe−1 to the trailing columns; the next minipanel's
// 16 columns land in the other smem buffer (rows ≥ pe; final R rows go to gmem), the
// rest go to gmem. Publishes col-pe dots/σ and the full prow row (incl. α at c = 0).
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_dom_b3(DomCtx<Real, NB, PW>& c, int p) {
    const int pe = p + PW, wt = NB - pe;
    const int rpb = c.a.rpb, r0 = c.r0, r1 = c.r1;
    const int r1all = c.r1all;
    const int tid = c.tid, nth = c.nth, lane = c.lane, warp = c.warp, NW = c.NW;
    const size_t lda = c.a.lda;
    Real* A = c.a.A;
    Real *Ts = c.Ts, *W2 = c.W2, *msmN = c.msmN;  // msmN=nullptr → niter=1 W2 gmem path
    const Real* Psum = c.a.Psum + size_t(p / PW) * NB * PW;
    double* dslot = c.a.dslotD;
    Real* prow = c.prow;
    // Ts already holds THIS block's T16 for minipanel p: d_dom_b2b builds it locally in
    // every block now (31:429 rung 1 Replicate, "compact R,T,W in cluster lanes"). It
    // used to be reloaded from the global T16g that block 0 published, which is what
    // made the broadcast barrier necessary. Only the upper triangle is read below
    // (k <= q), and that is exactly what b2b writes, so no masking pass is needed.
    __syncthreads();
    // W2[q + c·PW] = Σ_{k≤q} T̃16[k,q]·W[k,c]   (W[k,c] = Psum[c·PW + k])
    for (int e = tid; e < PW * wt; e += nth) {
        int cc = e / PW, q = e % PW;
        Real acc = Real(0);
        for (int k = 0; k <= q; ++k) acc += Ts[k + q * PW] * Psum[size_t(cc) * PW + k];
        W2[q + cc * PW] = acc;
    }
    __syncthreads();
    const int rb = max(r0, p), span = r1 - rb;
    Real* msmb = c.msm + (rb - r0);
    for (int cc = 2 * warp; cc < wt; cc += 2 * NW) {
        const bool two = (cc + 1 < wt);
        Real w0[PW], w1[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            w0[q] = W2[q + cc * PW];
            w1[q] = two ? W2[q + (cc + 1) * PW] : Real(0);
        }
        Real* col0 = A + size_t(pe + cc) * lda;
        Real* col1 = A + size_t(pe + cc + (two ? 1 : 0)) * lda;
        for (int rr = lane; rr < span; rr += 32) {
            const int r = rb + rr;
            Real a0 = Real(0), a1 = Real(0);
            #pragma unroll
            for (int q = 0; q < PW; ++q) {
                Real u = msmb[rr + q * rpb];
                a0 += u * w0[q];
                a1 += u * w1[q];
            }
            Real nv0 = col0[r] - a0;
            Real nv1 = two ? col1[r] - a1 : Real(0);
            col0[r] = nv0;
            if (msmN && cc < PW && r >= pe) msmN[(r - r0) + cc * rpb] = nv0;
            if (two) {
                col1[r] = nv1;
                if (msmN && cc + 1 < PW && r >= pe) msmN[(r - r0) + (cc + 1) * rpb] = nv1;
            }
            if (r == pe) {
                if (cc < PW) prow[cc] = nv0;
                if (two && cc + 1 < PW) prow[cc + 1] = nv1;
            }
        }
        for (int r = r1 + lane; r < r1all; r += 32) {
            Real a0 = Real(0), a1 = Real(0);
            #pragma unroll
            for (int q = 0; q < PW; ++q) {
                const Real u = A[r + size_t(p + q) * lda];
                a0 += u * w0[q];
                a1 += u * w1[q];
            }
            col0[r] -= a0;
            if (two) col1[r] -= a1;
        }
    }
    __syncthreads();
    if (msmN) {
        // niter>1: dots/σ of v_pe with next minipanel — all from the staged smem buffer
        // (coherent after __syncthreads; no fence needed). This is the G5 fix.
        for (int cc = 1 + warp; cc < PW; cc += NW) {
            Real acc = Real(0);
            for (int r = max(r0, pe + 1) + lane; r < r1; r += 32)
                acc += msmN[r - r0] * msmN[(r - r0) + cc * rpb];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            // generation 0: this feeds the NEXT minipanel's region jj = 0
            if (lane == 0)
                c.a.dcolP[(size_t)cc * c.a.dcolStride + c.bid] = double(acc);
        }
        if (warp == 0) {
            Real sa = Real(0);
            for (int r = max(r0, pe + 1) + lane; r < r1; r += 32) {
                Real v = msmN[r - r0];
                sa += v * v;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
            if (lane == 0)
                c.a.dcolP[(size_t)PW * c.a.dcolStride + c.bid] = double(sa);
        }
    } else {
        // niter=1 W2 tailfix path: b3 wrote next-mp cols to gmem; dots/σ read from gmem.
        __threadfence_block();   // make b3's gmem writes visible within block for dots/σ
        for (int cc = 1 + warp; cc < PW; cc += NW) {
            Real acc = Real(0);
            for (int r = max(r0, pe + 1) + lane; r < r1all; r += 32)
                acc += A[r + size_t(pe) * lda] * A[r + size_t(pe + cc) * lda];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0)
                c.a.dcolP[(size_t)cc * c.a.dcolStride + c.bid] = double(acc);
        }
        if (warp == 0) {
            Real sa = Real(0);
            for (int r = max(r0, pe + 1) + lane; r < r1all; r += 32) {
                Real v = A[r + size_t(pe) * lda];
                sa += v * v;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
            if (lane == 0)
                c.a.dcolP[(size_t)PW * c.a.dcolStride + c.bid] = double(sa);
        }
    }
}

// Non-coop barrier: atomicAdd arrival + generation-counter departure.
// Release: __syncthreads + __threadfence (all threads) → writes reach L2 before atomic.
// Acquire: __threadfence (leader) + __syncthreads — leader sees cross-block writes; broadcast via smem.
// MEASURED NEGATIVE, kept as a note rather than code. Folding A6's fixed-order column
// reduction into this barrier's last arriver looked free -- every other block is already
// spinning there -- and measured WORSE: geomean 1.2199 against 1.2318 for reducing in
// the reader (job 9720 vs 9719). The blocks are not idle-with-slack, they are waiting
// for THE RELEASE, and reducing before the generation bump delays the release for all
// of them. Three variants were tried and the ordering never changed:
//     atomics (A6-violating)          1.3424
//     reader-side warp reduction      1.2361
//     + transposed dcolP layout       1.2318   <- shipped
//     barrier last-arriver fold       1.2199
// The atomics were cheap because they were OFF the critical path: fire-and-forget
// accumulation that overlapped the next region's work. Every deterministic scheme puts
// the sum ON the dependency chain, and this kernel pays for that roughly 1:1. Closing
// the remaining ~8% would need an order-independent accumulator (fixed-point /
// Kulisch-style), which cannot pick a scale here without a fitted constant.
__device__ __forceinline__ void gmem_barrier(volatile unsigned* bar, int nbl) {
    __syncthreads();
    __threadfence();  // release: all threads publish to L2
    if (threadIdx.x == 0) {
        unsigned my_gen = bar[1];
        unsigned old = atomicAdd((unsigned*)&bar[0], 1);
        if (old == (unsigned)(nbl - 1)) {
            bar[0] = 0;
            __threadfence();
            bar[1] = my_gen + 1;
        } else {
            while (bar[1] == my_gen) { }
        }
    }
    __syncthreads();  // broadcast leader's view to all threads
}

// ============================================================ mbarrier helpers
// Hopper mbarrier (PTX §9.7.14.16). Key rule (§9.7.14.16.11): a thread may
// *arrive* on a .shared::cluster mbarrier (remote CTA, via mapa) but may only
// *try_wait* on a .shared::cta (LOCAL) mbarrier. So the fan-out pattern (1
// producer → M consumers) uses M per-consumer LOCAL mbarriers; the producer
// remote-arrives on each. mbarrier.arrive is non-blocking; try_wait.parity
// polls a phase flip (auto-resets the arrival count on flip, §9.7.14.16.8).
// Used by k_tqr_mega (T5 megakernel, replaces the gmem-counter + __nanosleep
// that deadlocks at n≥16384 in k_tqr_resident).
__device__ __forceinline__ void mbar_init(uint64_t* addr, uint32_t count) {
    uint64_t sh;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh) : "l"(addr));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "l"(sh), "r"(count) : "memory");
}
// Arrive on a REMOTE (cluster-shared) mbarrier via a mapa'd address. count=1.
__device__ __forceinline__ void mbar_arrive_cluster(uint64_t addr) {
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n"
                 :: "l"(addr) : "memory");
}
// Arrive on a LOCAL mbarrier (CTA-scope, no remote mapa needed).
__device__ __forceinline__ void mbar_arrive_local(uint64_t* addr) {
    uint64_t sh;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh) : "l"(addr));
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
                 :: "l"(sh) : "memory");
}
// Map a local shared address to peer CTA `rank` in the cluster → remote addr.
__device__ __forceinline__ uint64_t mbar_mapa(uint64_t* local_addr, uint32_t rank) {
    uint64_t sh, r;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh) : "l"(local_addr));
    asm volatile("mapa.shared::cluster.u64 %0, %1, %2;\n"
                 : "=l"(r) : "l"(sh), "r"(rank));
    return r;
}
// Try-wait (non-blocking) on a LOCAL mbarrier for the phase with given parity.
// Returns 1 if that phase has completed (count hit 0), 0 otherwise. Poll loop.
__device__ __forceinline__ uint32_t mbar_try_wait_parity(uint64_t* addr, uint32_t parity) {
    uint64_t sh; uint32_t done;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh) : "l"(addr));
    asm volatile(
        "{ .reg .pred %%p;\n"
        "  mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %%p, [%1], %2;\n"
        "  selp.u32 %0, 1, 0, %%p; }\n"
        : "=r"(done) : "l"(sh), "r"(parity) : "memory");
    return done;
}

// ---- TMA bulk async copy (PTX §9.7.9.26.4.1 cp.async.bulk) ----
// Declares expected transaction bytes on a LOCAL mbarrier (§9.7.14.16.6.1).
__device__ __forceinline__ void mbar_expect_tx(uint64_t* addr, uint32_t bytes) {
    uint64_t sh;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh) : "l"(addr));
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;\n"
                 :: "l"(sh), "r"(bytes) : "memory");
}
// cp.async.bulk gmem→smem with mbarrier completion (§9.7.9.26.4.1).
__device__ __forceinline__ void cp_async_bulk_g2s(void* smem_dst, const void* gmem_src,
                                                   uint32_t bytes, uint64_t* mbar) {
    uint64_t sh_dst, sh_mbar;
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh_dst) : "l"(smem_dst));
    asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(sh_mbar) : "l"(mbar));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "l"(sh_dst), "l"(gmem_src), "r"(bytes), "l"(sh_mbar) : "memory");
}
// Fence to make async-proxy smem writes visible to compute proxy (§9.7.9.26.2).
__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---- TMA tensor load (cp.async.bulk.tensor.2d) + expect_tx ----
// W3' carrier: B32 swizzle (CU_TENSOR_MAP_SWIZZLE_32B) proven viable for KT=8 TF32
// (w3-swizzle-prove daceda7: max_err=5.96e-08 = 1 ULP). SBO=256 mandatory; LBO=16
// canonical ("assumed 1" for K-major swizzled, PTX §9.7.16.5.1.2.1.3 Fig 167).
__device__ __forceinline__ void tma_load_2d(void* smem_dst, const CUtensorMap* desc,
                                            int crd0, int crd1, uint64_t* mbar) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0],[%1,{%2,%3}],[%4];"
        :: "r"(smem_addr), "l"(desc), "r"(crd0), "r"(crd1), "r"(mbar_addr)
        : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t sh = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(sh), "r"(tx_bytes) : "memory");
}

// ---- wgmma helpers — PTX §9.7.16 (sm_90a mandatory) ----
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template<int N>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
__device__ __forceinline__ void wgmma_fence_acc(float* acc, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) asm volatile("" : "+f"(acc[i]) :: "memory");
}
// GMMA descriptor: layout_type at bits 62-63.
//   0 = SWIZZLE_NONE, 3 = SWIZZLE_32B (CU_TENSOR_MAP_SWIZZLE_32B).
// For SWIZZLE_NONE: LBO = contiguous-dim byte stride, SBO = 8 * LBO.
// For SWIZZLE_32B (KT=8): LBO = 16 (half swizzle group), SBO = 256 (8 swizzle groups).
__device__ __forceinline__ uint64_t make_gmma_desc(const void* smem_ptr, int lbo, int sbo, int layout_type) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(addr >> 4) & 0x3FFF);
    desc |= ((uint64_t)(lbo >> 4) & 0x3FFF) << 16;
    desc |= ((uint64_t)(sbo >> 4) & 0x3FFF) << 32;
    desc |= ((uint64_t)layout_type) << 62;
    return desc;
}
// wgmma m64n64k8 SS TF32 — 32 accumulators per M-half.
// scale_d: 0=zero accum, 1=accumulate (immediate — ptxas rejects predicate-folded 0).
// a_maj/b_maj: -1=M/N-major (row-major), 1=K-major (col-major) — immediate.
#define WGMMA_M64N64K8_SS(acc, da, db, scale_d, a_maj, b_maj) \
    do { \
        asm volatile( \
        "wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 " \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}," \
        " %32, %33, %34, %35, %36;\n" \
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]), \
          "+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]), \
          "+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]), \
          "+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]), \
          "+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]), \
          "+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]), \
          "+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]), \
          "+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]) \
        : "l"(da), "l"(db), "n"(scale_d), "n"(a_maj), "n"(b_maj)); \
    } while(0)

template<typename Real, int NB, int PW = 16, bool CL = false, bool NOCOOP = false,
         bool PERSIST = false>
// LRQR_DOM_BPSM: minimum blocks per SM the compiler must fit. With no second argument
// __launch_bounds__ lets it take 128 registers/thread, and 512 x 128 is exactly the
// 65,536-register file -- ONE block per SM, 16 warps, 25% occupancy. ncu at 2048^2 on
// the cluster carrier (job 9766): barrier stall 53%, 0.21 instructions issued per
// cycle, 30.49 of 32 threads active. The block is not idle and it is not divergent; it
// is latency-bound with too few warps, and the cap is self-inflicted. 2 forces <= 64
// registers and 2 blocks/SM = 50% occupancy, paid for in spills -- which is exactly the
// trade to measure, not to assume.
#ifndef LRQR_DOM_BPSM
#define LRQR_DOM_BPSM 1
#endif
__global__ __launch_bounds__(512, LRQR_DOM_BPSM) void k_panel_domino(DomArgs<Real> a, unsigned* gbar) {
    namespace cg = cooperative_groups;
    // Grid-stride non-cooperative domino: launch R = gridDim.x resident blocks,
    // each grid-striding over nslab slabs. The gmem barrier counts R arrivals
    // (all co-resident → deadlock-free). When nslab <= R (niter=1), this is
    // identical to the original one-block-per-slab path.
    const int R = gridDim.x;
    const int nslab = a.nslab;
    const int niter = PERSIST ? 1 : (nslab + R - 1) / R;
    auto gsync = [gbar, R] {
        if constexpr (NOCOOP) { gmem_barrier(gbar, R); }
        else if constexpr (CL) { cg::this_cluster().sync(); }
        else                   { cg::this_grid().sync(); }
    };
    extern __shared__ __align__(16) unsigned char smraw[];
    DomCtx<Real, NB, PW> c;
    c.a = a;
    // G5 fix: double-buffer (msm0+msm1) when niter>1 — b3 stages next-mp dots/σ into
    // smem msmN (coherent after __syncthreads). Single-buffer (msm0 only) when niter==1
    // — W2 tailfix preserved (b3 reads dots from gmem, larger rpb → tall win).
    Real* msm0 = reinterpret_cast<Real*>(smraw);
    Real* msm1 = msm0 + size_t(a.rpb) * PW;          // second buffer (niter>1 only)
    const bool db = (niter > 1);
    c.W2  = db ? (msm1 + size_t(a.rpb) * PW) : (msm0 + size_t(a.rpb) * PW);
    c.Ts  = c.W2 + PW * (NB - PW);
    c.ssm = c.Ts + PW * PW;
    c.fsm = c.ssm + PW;
    c.gsm = c.fsm + NB;
    Real* psm = c.gsm + PW * PW;
    c.fv = psm; c.bet = psm + NB; c.tauT = psm + 2 * NB;
    c.dslot = a.dslot; c.prow = a.prow; c.sig = a.sig;
    c.tid = threadIdx.x; c.nth = blockDim.x;
    c.lane = c.tid & 31; c.warp = c.tid >> 5; c.NW = c.nth >> 5;
    c.nbl = R;
    // Persistent overflow mode gives each of the R resident CTAs one logical
    // contiguous row interval.  Its capacity-sized prefix resides in msm; only
    // the suffix that cannot fit in shared memory remains in compact in-place A.
    // The standard path keeps the historical rpb-sized grid-stride intervals.
    auto set_slab = [&](int slab, bool act) {
        c.bid = slab;
        if (!act) {
            c.r0 = c.r1 = c.r1all = a.mk;
        } else if constexpr (PERSIST) {
            const int logicalRpb = (a.mk + R - 1) / R;
            c.r0 = slab * logicalRpb;
            c.r1all = min(a.mk, c.r0 + logicalRpb);
            c.r1 = min(c.r1all, c.r0 + a.rpb);
        } else {
            c.r0 = slab * a.rpb;
            c.r1 = min(a.mk, c.r0 + a.rpb);
            c.r1all = c.r1;
        }
    };
    unsigned long long tck = 0;
    auto stamp = [&](int slot) {
        if (a.tphase && blockIdx.x == 0 && c.tid == 0) {
            unsigned long long now = clock64();
            atomicAdd(&a.tphase[slot], now - tck);
            tck = now;
        }
    };
    if (a.tphase && blockIdx.x == 0 && c.tid == 0) tck = clock64();

    // ---- init phase: per-slab load + dots of col 0 (grid-stride over slabs).
    //      Inactive blocks (slab >= nslab) get an empty row range so the
    //      __syncthreads inside d_dom_region etc. are reached by ALL threads.
    for (int s = 0; s < niter; ++s) {
        int slab = blockIdx.x + s * R;
        bool act = (slab < nslab);
        set_slab(slab, act);
        if (act) {
            if (slab == 0) {
                for (int i = c.tid; i < NB; i += c.nth) c.a.sigD[i] = 0.0;
                for (int i = c.tid; i < PW * PW; i += c.nth) c.a.dslotD[i] = 0.0;
            }
            const int span0 = c.r1 - c.r0;
            for (int e = c.tid; e < span0 * PW; e += c.nth) {
                int q = e / span0, rr = e % span0;
                msm0[rr + q * a.rpb] = a.A[(c.r0 + rr) + size_t(q) * a.lda];
            }
        }
        gsync();
        // dots of col 0 (rows > 0) with cols 1..15; σ0; prow[0][c] = A[0,c]
        for (int cc = 1 + c.warp; cc < PW; cc += c.NW) {
            Real acc = Real(0);
            for (int r = max(c.r0, 1) + c.lane; r < c.r1; r += 32)
                acc += msm0[r - c.r0] * msm0[(r - c.r0) + cc * a.rpb];
            for (int r = max(c.r1, 1) + c.lane; r < c.r1all; r += 32)
                acc += a.A[r] * a.A[r + size_t(cc) * a.lda];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            // generation 0: this feeds region jj = 0 of the first minipanel
            if (c.lane == 0)
                c.a.dcolP[(size_t)cc * c.a.dcolStride + c.bid] = double(acc);
        }
        if (c.warp == 0) {
            Real sa = Real(0);
            for (int r = max(c.r0, 1) + c.lane; r < c.r1; r += 32) {
                Real v = msm0[r - c.r0];
                sa += v * v;
            }
            for (int r = max(c.r1, 1) + c.lane; r < c.r1all; r += 32) {
                Real v = a.A[r];
                sa += v * v;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
            if (c.lane == 0)
                c.a.dcolP[(size_t)PW * c.a.dcolStride + c.bid] = double(sa);
        }
        if (act && c.r0 == 0 && c.tid < PW) c.prow[c.tid] = msm0[size_t(c.tid) * a.rpb];
        gsync();               // init produced generation 0, for region jj = 0
    }
    stamp(0);

    // ---- column phase: per-minipanel region+b1 (slab loop) / B2a / b2b / b3 (slab loop)
    for (int p = 0; p < NB; p += PW) {
        const int mp = p / PW;
        if (db) {
            // niter>1: ping-pong double-buffer (pre-W2). b3 stages next-mp into msmN.
            c.msm  = (mp & 1) ? msm1 : msm0;
            c.msmN = (mp & 1) ? msm0 : msm1;
        } else {
            c.msm  = msm0;                  // single buffer (W2 tailfix: no ping-pong)
            c.msmN = nullptr;               // b3 reads dots from gmem (niter=1 shortcut)
        }
        // --- niter==1 single-buffer per-mp load: for niter==1 and p>0, b3 no longer
        //     stages the next minipanel into msmN. Load it from gmem (b3 wrote it there).
        //     For niter>1 the per-slab loads in the region/b1 loops handle this. ---
        if (p > 0 && niter == 1) {
            const int span0 = c.r1 - c.r0;
            for (int e = c.tid; e < span0 * PW; e += c.nth) {
                int q = e / span0, rr = e % span0;
                msm0[rr + q * a.rpb] = a.A[(c.r0 + rr) + size_t(p + q) * a.lda];
            }
            __syncthreads();
        }
        // --- region (jj outer, slabs inner) + b1 (slab loop) ---
        // For niter>1 the jj loop MUST be outer: d_dom_region(jj) reads
        // sig[j]/dslot[jj*PW+*] which are atomicAdd'd by ALL nslab slabs.
        // The gsync after the s-loop syncs R co-resident blocks; niter
        // iterations cover all nslab slabs.  msm is saved/restored to
        // gmem A between slabs within each jj pass.
        for (int jj = 0; jj < PW; ++jj) {
            for (int s = 0; s < niter; ++s) {
                int slab = blockIdx.x + s * R;
                bool act = (slab < nslab);
                set_slab(slab, act);
                if (act && niter > 1) {
                    const int span0 = c.r1 - c.r0;
                    for (int e = c.tid; e < span0 * PW; e += c.nth) {
                        int q = e / span0, rr = e % span0;
                        c.msm[rr + q * a.rpb] = a.A[(c.r0 + rr) + size_t(p + q) * a.lda];
                    }
                    __syncthreads();
                }
                d_dom_region<Real, NB, PW>(c, p, jj);
                if (act && niter > 1) {
                    const int rb = max(c.r0, p), span = c.r1 - rb;
                    for (int e = c.tid; e < span * PW; e += c.nth) {
                        int q = e / span, r = rb + e % span;
                        a.A[r + size_t(p + q) * a.lda] = c.msm[(r - c.r0) + q * a.rpb];
                    }
                    __syncthreads();
                }
            }
            gsync();           // region jj produced generation (jj+1)&1
        }
        stamp(1);
        for (int s = 0; s < niter; ++s) {
            int slab = blockIdx.x + s * R;
            bool act = (slab < nslab);
            set_slab(slab, act);
            if (act && niter > 1) {
                const int span0 = c.r1 - c.r0;
                for (int e = c.tid; e < span0 * PW; e += c.nth) {
                    int q = e / span0, rr = e % span0;
                    c.msm[rr + q * a.rpb] = a.A[(c.r0 + rr) + size_t(p + q) * a.lda];
                }
                __syncthreads();
            }
            d_dom_b1<Real, NB, PW>(c, p, tck);
            __syncthreads();
        }
        gsync();
        stamp(2);
        // --- dslot zero (was in b1; moved here for grid-stride safety) ---
        if (blockIdx.x == 0)
            for (int i = c.tid; i < PW * PW; i += c.nth) c.a.dslotD[i] = 0.0;
        // --- B2a: reduce boundary partials over ALL nslab slabs (fp64 accumulator; uniform) ---
        for (int e = blockIdx.x * c.nth + c.tid; e < NB * PW; e += R * c.nth) {
            double sacc = 0.0;
            for (int b = 0; b < nslab; ++b) sacc += a.Pp[size_t(b) * NB * PW + e];
            a.Psum[size_t(mp) * NB * PW + e] = Real(sacc);
        }
        gsync();
        stamp(3);
        // --- b2b: REPLICATED, not broadcast ---
        // 31:429 rung 1 Replicate is "compact R,T,W in cluster lanes". This 16x16 T
        // recurrence ran in block 0 and was published to every other block through a
        // device-wide barrier -- but its inputs are already everywhere: Psum is global
        // and visible after B2a's barrier, and tauT is block-local and computed
        // redundantly by every block in d_dom_region. So every block can build it, and
        // the barrier whose only job was to broadcast it goes. 4 of the panel's rounds.
        d_dom_b2b<Real, NB, PW>(c, p);
        __syncthreads();          // block-scope: Ts is block-local now, ~50 ns not 1.35 us
        stamp(4);
        // --- b3 (grid-stride over slabs; niter>1 stages next-mp into smem msmN) ---
        for (int s = 0; s < niter; ++s) {
            int slab = blockIdx.x + s * R;
            bool act = (slab < nslab);
            set_slab(slab, act);
            if (act && niter > 1 && NB - (p + PW) > 0) {
                // grid-stride: reload masked minipanel from gmem for this slab
                const int rb = max(c.r0, p), span = c.r1 - rb;
                for (int e = c.tid; e < span * PW; e += c.nth) {
                    int q = e / span, rr = e % span, r = rb + rr, cq = p + q;
                    Real v = a.A[r + size_t(p + q) * a.lda];
                    if (r == cq) v = c.fv[p + q];
                    else if (r < cq) v = Real(0);
                    c.msm[(rb - c.r0) + rr + q * a.rpb] = v;
                }
            }
            if (NB - (p + PW) > 0) __syncthreads();
            if (NB - (p + PW) > 0) d_dom_b3<Real, NB, PW>(c, p);
            if (NB - (p + PW) > 0) __syncthreads();
        }
        gsync();               // b3 produced generation 0 for the next minipanel
        stamp(5);
    }
    // ---- T assembly: REPLICATED, not distributed ----
    //
    // 31:429 gives rung 1's Replicate as "compact R,T,W in cluster lanes" and rung 0's
    // Combine as "warp partials / local HH". The compact WY T is exactly that state, and
    // it was being built by a grid-stride recurrence across every block behind SEVEN
    // device-wide barriers -- 1 + 2*(NB/PW - 1) of the panel's 89 rounds -- to produce
    // about 8,700 element updates. With 66 blocks x 512 threads that is 0.26 elements
    // per thread per barrier: the barriers cost far more than the arithmetic they order.
    //
    // One block does the whole thing instead, with __syncthreads in place of gsync.
    // Nothing inside the kernel reads Tpan afterwards -- its consumer is larfb, after
    // the launch -- so no barrier is needed to publish it, and the other blocks go
    // straight on to the writeout. c.fv is block-local and identical in every block
    // (d_dom_region computes the reflector scalars redundantly), so block 0 has the
    // same inputs the distributed version had.
    //
    // __syncthreads() carries a block-scope memory fence, so the M1g round-trip through
    // a.Pp stays visible within the block exactly as the gsync made it visible across
    // the grid.
    if (blockIdx.x == 0) {
        const int gtid = c.tid, gth = c.nth;
        for (int e = gtid; e < NB * NB; e += gth) {
            const int cc = e / NB, r = e % NB, sr = r / PW, sc = cc / PW;
            if (sr == sc)
                a.Tpan[e] = a.T16g[size_t(sr) * PW * PW + (r - sr * PW) + (cc - sc * PW) * PW];
            else if (sr > sc)
                a.Tpan[e] = Real(0);
        }
        __syncthreads();
        Real* M1g = a.Pp;                     // boundary-partials buffer is free by now
        for (int s = 1; s < NB / PW; ++s) {
            const int p = s * PW, wt = NB - p - PW;
            const Real* Ps = a.Psum + size_t(s) * NB * PW;
            const Real* Tg = a.T16g + size_t(s) * PW * PW;
            for (int e = gtid; e < p * PW; e += gth) {
                const int i = e % p, q = e / p;
                Real acc = Real(0);
                for (int k = 0; k <= q; ++k)
                    acc += Ps[size_t(wt + i) * PW + k] * Tg[k + q * PW];
                M1g[e] = acc;
            }
            __syncthreads();
            for (int e = gtid; e < p * PW; e += gth) {
                const int i = e % p, q = e / p;
                Real acc = Real(0);
                for (int k = i; k < p; ++k)
                    acc += a.Tpan[i + size_t(k) * NB] * M1g[k + q * p];
                a.Tpan[i + size_t(p + q) * NB] = -acc;
            }
            __syncthreads();
        }
        for (int e = gtid; e < NB * NB; e += gth) {
            const int cc = e / NB, r = e % NB;
            if (r <= cc) a.Tpan[e] *= c.fv[r] * c.fv[cc];
        }
    }
    // ---- writeout: scale V in place (v = u/f), β on diag, Y into SPV.
    // Stage 1/f and β into THIS block's smem: DSMEM reads must not outlive block 0.
    if (c.tid < NB) { c.fsm[c.tid] = Real(1) / c.fv[c.tid]; c.gsm[c.tid] = c.bet[c.tid]; }
    __syncthreads();
    for (int s = 0; s < niter; ++s) {
        int slab = blockIdx.x + s * R;
        if (slab >= nslab) break;
        set_slab(slab, true);
        for (int r = c.r0 + c.tid; r < c.r1all; r += c.nth) {
            // 31:403 stores V below the diagonal of A and nothing else; this Y
            // copy is a convenience for callers that want a pre-masked V for plain
            // GEMMs, and it costs m*b words -- O(mb), not the O(nb) the compactness
            // theorem (31:415) allows. At 131072x16384 that measured as 8x the
            // ENTIRE nb auxiliary budget. A caller that applies V straight out of
            // tril(A) (TRMM(LOWER,UNIT)+GEMM) needs none of it and passes SPV=null.
            // With SPV non-null this is bit-identical to before: one predicate, and
            // the store it guards is the expensive part.
            Real* SPVr = a.SPV + size_t(a.spvcol) * a.ldspv + a.row0 + r;
            for (int cc = 0; cc < NB; ++cc) {
                Real y;
                if (r > cc) {
                    Real v = a.A[r + size_t(cc) * a.lda] * c.fsm[cc];
                    a.A[r + size_t(cc) * a.lda] = v;
                    y = v;
                } else if (r == cc) {
                    a.A[r + size_t(cc) * a.lda] = c.gsm[cc];
                    y = Real(1);
                } else {
                    y = Real(0);
                }
                if (a.SPV) SPVr[size_t(cc) * a.ldspv] = y;
            }
        }
    }
    stamp(6);
}

// ================================================================ pipelined domino
// (spec §5 sequential ⊕-schedule + §9(iv) look-ahead applied one level down): for
// mk ≤ pipeRowMax the WHOLE minipanel (rows p..mk × 16 cols) fits in ONE block's smem,
// so the column microloop runs at __syncthreads latency with ZERO cross-block protocol
// (no cluster/grid sync, no L2 scalar round-trips — the lockstep kernel pays ~2.2µs per
// column for exactly that). Block 0 races ahead on the critical path; updater blocks
// trail one boundary behind on gmem doorbells: on uflag[s] they stage block 0's NEXT
// minipanel first (priority; stCnt), then WY-apply the boundary to their statically-
// owned trailing columns (owner = col % U → per-column boundary order is preserved
// within one block, no cross-updater races) and emit the Psum dot slabs for the
// deferred T̃ compound. This is cuSOLVER's panel∥trailing domino replicated inside the
// panel. Cooperative launch for co-residency only — no grid.sync anywhere.

// one column region, single-block, all operands in registers/smem: form reflector j
// from (σ, α, dots), apply it to the rest of the minipanel, leave next column's
// (dots, σ) behind. Fully inlined — no ctx struct (a by-reference ctx demotes every
// field access to a local-memory load on the critical path).
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_region(
        Real* msm, Real* dsl, Real* ssm, Real* sgL, Real* fvL, Real* betL, Real* ttL,
        int ldm, int p, int jj, int h, int tid, int nth, int lane, int warp, int NW) {
    const Real sg = sgL[jj], al = msm[jj + jj * ldm];
    Real beta = al, f = Real(1), tt = Real(0);
    if (sg > Real(0)) {
        Real mu = sqrt(al * al + sg);
        beta = (al >= Real(0)) ? -mu : mu;
        f = al - beta;
        tt = Real(-1) / (beta * f);
    }
    if (tid == 0) {
        const int j = p + jj;
        betL[j] = beta; fvL[j] = f; ttL[j] = tt;
    }
    if (tid > jj && tid < PW)
        ssm[tid] = tt * (f * msm[jj + tid * ldm] + dsl[jj * PW + tid]);
    if (jj + 1 < PW) {                    // zero next column's accumulators (atomics)
        if (tid == 0) sgL[jj + 1] = Real(0);
        if (tid > jj + 1 && tid < PW) dsl[(jj + 1) * PW + tid] = Real(0);
    }
    __syncthreads();
    if (jj + 1 >= PW) return;
    Real* mj  = msm + jj * ldm;
    Real* mc1 = msm + (jj + 1) * ldm;
    const Real s1 = ssm[jj + 1];
    for (int rr = jj + tid; rr < h; rr += nth) {
        Real vj = (rr == jj) ? f : mj[rr];
        mc1[rr] -= s1 * vj;
    }
    __syncthreads();
    // phase 2, 2D warp map: njob = apply+dot columns + one σ job; stripes give every
    // warp a row slice, 4-way unrolled for ILP.
    const int ncols = PW - (jj + 2), njob = ncols + 1;
    const int S = max(1, NW / njob);
    const int job = warp % njob, st = warp / njob;
    if (st < S) {
        const int step = S * 32;
        if (job < ncols) {
            const int cc = jj + 2 + job;
            Real* mcc = msm + cc * ldm;
            const Real sc = ssm[cc];
            Real acc = Real(0);
            int rr = jj + st * 32 + lane;
            for (; rr + 3 * step < h; rr += 4 * step) {
                const int r1_ = rr + step, r2_ = rr + 2 * step, r3_ = rr + 3 * step;
                Real v0 = (rr == jj) ? f : mj[rr];
                Real v1 = mj[r1_], v2 = mj[r2_], v3 = mj[r3_];
                Real n0 = mcc[rr]  - sc * v0, n1 = mcc[r1_] - sc * v1;
                Real n2 = mcc[r2_] - sc * v2, n3 = mcc[r3_] - sc * v3;
                mcc[rr] = n0; mcc[r1_] = n1; mcc[r2_] = n2; mcc[r3_] = n3;
                if (rr > jj + 1) acc += mc1[rr] * n0;
                acc += mc1[r1_] * n1 + mc1[r2_] * n2 + mc1[r3_] * n3;
            }
            for (; rr < h; rr += step) {
                Real vj = (rr == jj) ? f : mj[rr];
                Real nv = mcc[rr] - sc * vj;
                mcc[rr] = nv;
                if (rr > jj + 1) acc += mc1[rr] * nv;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) atomicAdd(&dsl[(jj + 1) * PW + cc], acc);
        } else {
            Real sa = Real(0);
            int rr = jj + 2 + st * 32 + lane;
            for (; rr + 3 * step < h; rr += 4 * step) {
                Real v0 = mc1[rr], v1 = mc1[rr + step];
                Real v2 = mc1[rr + 2 * step], v3 = mc1[rr + 3 * step];
                sa += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
            for (; rr < h; rr += step) { Real v = mc1[rr]; sa += v * v; }
            #pragma unroll
            for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
            if (lane == 0) atomicAdd(&sgL[jj + 1], sa);
        }
    }
    __syncthreads();
}

// boundary, block 0: local ũ Gram (120 pairs, unrolled) + one-warp τ̃-recurrence → T̃16.
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_gram(
        Real* msm, Real* Gs, Real* Ts, Real* fvL, Real* ttL,
        int ldm, int p, int h, int lane, int warp, int NW) {
    for (int pr_ = warp; pr_ < PW * (PW - 1) / 2; pr_ += NW) {
        int jT = 1, rem = pr_;
        while (rem >= jT) { rem -= jT; ++jT; }
        const int k = rem;
        const Real* uk = msm + k * ldm;
        const Real* uj = msm + jT * ldm;
        Real acc = (lane == 0) ? fvL[p + jT] * uk[jT] : Real(0);
        int rr = jT + 1 + lane;
        for (; rr + 96 < h; rr += 128)
            acc += uk[rr] * uj[rr] + uk[rr + 32] * uj[rr + 32]
                 + uk[rr + 64] * uj[rr + 64] + uk[rr + 96] * uj[rr + 96];
        for (; rr < h; rr += 32) acc += uk[rr] * uj[rr];
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) Gs[k + jT * PW] = acc;
    }
    __syncthreads();
    if (warp == 0)
        for (int jT = 0; jT < PW; ++jT) {
            const Real tj = ttL[p + jT];
            Real acc = Real(0);
            if (lane < jT)
                for (int k = lane; k < jT; ++k) acc += Ts[lane + k * PW] * Gs[k + jT * PW];
            __syncwarp();
            if (lane < jT) Ts[lane + jT * PW] = -tj * acc;
            else if (lane == jT) Ts[jT + jT * PW] = tj;
            __syncwarp();
        }
    __syncthreads();
}

// boundary, block 0: flush minipanel (raw: R above pivots, u below) + T̃16 slab +
// scalars to gmem, then ring the boundary doorbell.
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_flush(
        Real* msm, Real* Ts, Real* fvL, Real* betL, Real* ttL,
        Real* A, size_t lda, Real* T16g, Real* fvG, Real* betG, Real* ttG, int* pf,
        int ldm, int p, int h, int s, int tid, int nth) {
    for (int e = tid; e < h * PW; e += nth) {
        int q = e / h, rr = e % h;
        A[(p + rr) + size_t(p + q) * lda] = msm[rr + q * ldm];
    }
    for (int e = tid; e < PW * PW; e += nth) {
        int cc = e / PW, r = e % PW;
        T16g[e] = (r <= cc) ? Ts[e] : Real(0);
    }
    if (tid < PW) {
        fvG[p + tid]  = fvL[p + tid];
        betG[p + tid] = betL[p + tid];
        ttG[p + tid]  = ttL[p + tid];
    }
    __syncthreads();
    __threadfence();
    if (tid == 0) atomicExch(&pf[s], 1);
}

// block 0: seed a freshly staged minipanel's first-column (dots, σ) from smem.
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_seed(
        Real* msm, Real* dsl, Real* sgL,
        int ldm, int h2, int tid, int lane, int warp, int NW) {
    if (tid < PW) { dsl[tid] = Real(0); if (tid == 0) sgL[0] = Real(0); }
    __syncthreads();
    const int S = max(1, NW / PW);
    const int job = warp % PW, st = warp / PW;
    if (st < S) {
        const int step = S * 32;
        const Real* m0 = msm;
        const Real* mcJ = msm + job * ldm;     // job 0 = σ, jobs 1..15 = dots
        Real acc = Real(0);
        int rr = 1 + st * 32 + lane;
        for (; rr + 3 * step < h2; rr += 4 * step)
            acc += m0[rr] * mcJ[rr] + m0[rr + step] * mcJ[rr + step]
                 + m0[rr + 2 * step] * mcJ[rr + 2 * step]
                 + m0[rr + 3 * step] * mcJ[rr + 3 * step];
        for (; rr < h2; rr += step) acc += m0[rr] * mcJ[rr];
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            if (job == 0) atomicAdd(&sgL[0], acc);
            else          atomicAdd(&dsl[job], acc);
        }
    }
    __syncthreads();
}

// block 0: initial stage — load minipanel 0 raw from gmem, then seed.
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_gather(
        Real* msm, Real* dsl, Real* sgL, const Real* A, size_t lda,
        int ldm, int h2, int tid, int nth, int lane, int warp, int NW) {
    for (int e = tid; e < h2 * PW; e += nth) {
        int q = e / h2, rr = e % h2;
        msm[rr + q * ldm] = A[rr + size_t(q) * lda];
    }
    __syncthreads();
    d_pipe_seed<Real, PW>(msm, dsl, sgL, ldm, h2, tid, lane, warp, NW);
}

// block 0: SELF-stage minipanel s+1 — apply this boundary's WY to the next 16 columns
// locally (ũ and T̃16 are already in smem), shifting the minipanel buffer 16 rows up
// in place via compute-then-shift row tiles. The only cross-block dependency is the
// updaters' boundary-(s−1) pass over these columns (stCnt[s−1]) — rung one full
// boundary period earlier, so the wait is ~zero in steady state.
template<typename Real, int PW>
__device__ __forceinline__ void d_pipe_selfstage(
        Real* msm, Real* Ds, Real* W2, Real* Ts, Real* fvL,
        Real* A, size_t lda, int* pf,
        int ldm, int p, int h, int s, int tid, int nth, int lane, int warp, int NW) {
    const int pe = p + PW;
    if (s >= 1) {
    if (tid == 0)
        while (atomicAdd(&pf[s], 0) == 0) __nanosleep(1);
    __syncthreads();
    __threadfence();
    }
    // dots d[k][c] = ũ_kᵀ·A[·, pe+c]: warp per column, masked ũ on the triangle rows
    for (int cq = warp; cq < PW; cq += NW) {
        const Real* col = A + size_t(pe + cq) * lda + p;
        Real dv[PW];
        #pragma unroll
        for (int k = 0; k < PW; ++k) dv[k] = Real(0);
        for (int rr = lane; rr < h; rr += 32) {
            Real sv = col[rr];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k) {
                    Real uk = (rr < k) ? Real(0)
                                       : (rr == k ? fvL[p + k] : msm[rr + k * ldm]);
                    dv[k] += uk * sv;
                }
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) dv[k] += msm[rr + k * ldm] * sv;
            }
        }
        #pragma unroll
        for (int k = 0; k < PW; ++k) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) dv[k] += __shfl_xor_sync(0xffffffffu, dv[k], o);
        }
        Real out = Real(0);
        #pragma unroll
        for (int k = 0; k < PW; ++k) out = (lane == k) ? dv[k] : out;
        if (lane < PW) Ds[lane + cq * PW] = out;
    }
    __syncthreads();
    if (tid < PW * PW) {          // W2[k + c·PW] = Σ_{k'≤k} T̃16[k',k]·d[k'][c]
        const int cq = tid / PW, k = tid % PW;
        Real acc = Real(0);
        for (int k2 = 0; k2 <= k; ++k2) acc += Ts[k2 + k * PW] * Ds[k2 + cq * PW];
        W2[k + cq * PW] = acc;
    }
    __syncthreads();
    // apply + shift: each thread holds one row's 16 new values in registers, then
    // writes them 16 rows up (rows < PW are final R rows of the next columns → gmem)
    for (int t0 = 0; t0 < h; t0 += nth) {
        const int rr = t0 + tid;
        const bool live = rr < h;
        Real val[PW];
        if (live) {
            #pragma unroll
            for (int q = 0; q < PW; ++q) val[q] = A[(p + rr) + size_t(pe + q) * lda];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k) {
                    Real uk = (rr < k) ? Real(0)
                                       : (rr == k ? fvL[p + k] : msm[rr + k * ldm]);
                    #pragma unroll
                    for (int q = 0; q < PW; ++q) val[q] -= uk * W2[k + q * PW];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) {
                    Real uk = msm[rr + k * ldm];
                    #pragma unroll
                    for (int q = 0; q < PW; ++q) val[q] -= uk * W2[k + q * PW];
                }
            }
        }
        __syncthreads();
        if (live) {
            if (rr < PW) {
                #pragma unroll
                for (int q = 0; q < PW; ++q) A[(p + rr) + size_t(pe + q) * lda] = val[q];
            } else {
                #pragma unroll
                for (int q = 0; q < PW; ++q) msm[(rr - PW) + q * ldm] = val[q];
            }
        }
        __syncthreads();
    }
}

// updater block, boundary s: on the doorbell, cache masked ũ + T̃16, then per owned
// column (owner = col % U, stable across boundaries → per-column boundary order holds
// within one block): dot → w = T̃ᵀd → WY apply. The 16 next-minipanel columns come
// first (each bumps stCnt); prev-column Psum dots (deferred T̃ compound) come last.
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_pipe_upd(
        Real* A, size_t lda, Real* usm, Real* Ts, Real* fvL, Real* d, Real* w,
        const Real* T16gAll, const Real* fvG, Real* PsumAll, int* pf,
        int ldm, int mk, int s, int u, int U, int tid, int nth, int lane) {
    const int p = s * PW, pe = p + PW, wt = NB - pe;
    const int h = mk - p;
    if (tid == 0)
        while (atomicAdd(&pf[s], 0) == 0) __nanosleep(1);
    __syncthreads();
    __threadfence();
    for (int e = tid; e < PW * PW; e += nth) Ts[e] = T16gAll[size_t(s) * PW * PW + e];
    if (tid < PW) fvL[tid] = fvG[p + tid];
    __syncthreads();
    for (int e = tid; e < h * PW; e += nth) {
        int q = e / h, rr = e % h;
        Real v = A[(p + rr) + size_t(p + q) * lda];
        usm[rr + q * ldm] = (rr > q) ? v : (rr == q ? fvL[q] : Real(0));
    }
    __syncthreads();
    const int c0 = pe + PW;          // block 0 self-stages [pe, pe+PW); we start after
    for (int cc = c0 + ((u - c0 % U + U) % U); cc < NB; cc += U) {
        if (tid < PW) d[tid] = Real(0);
        __syncthreads();
        Real dloc[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) dloc[q] = Real(0);
        Real* col = A + size_t(cc) * lda + p;
        for (int rr = tid; rr < h; rr += nth) {
            Real sv = col[rr];
            #pragma unroll
            for (int q = 0; q < PW; ++q) dloc[q] += usm[rr + q * ldm] * sv;
        }
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) dloc[q] += __shfl_xor_sync(0xffffffffu, dloc[q], o);
        }
        Real out = Real(0);
        #pragma unroll
        for (int q = 0; q < PW; ++q) out = (lane == q) ? dloc[q] : out;
        if (lane < PW) atomicAdd(&d[lane], out);
        __syncthreads();
        if (tid < PW) {
            Real acc = Real(0);
            for (int k = 0; k <= tid; ++k) acc += Ts[k + tid * PW] * d[k];
            w[tid] = acc;
        }
        __syncthreads();
        for (int rr = tid; rr < h; rr += nth) {
            Real acc = Real(0);
            #pragma unroll
            for (int q = 0; q < PW; ++q) acc += usm[rr + q * ldm] * w[q];
            col[rr] -= acc;
        }
        if (cc < pe + 2 * PW) {      // a column block 0 will self-stage next boundary
            __syncthreads();
            __threadfence();
            if (tid == 0) atomicAdd(&pf[8 + s], 1);
        }
        __syncthreads();
    }
    Real* Psum = PsumAll + size_t(s) * NB * PW;
    for (int i = u; i < p; i += U) {
        if (tid < PW) d[tid] = Real(0);
        __syncthreads();
        Real dloc[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) dloc[q] = Real(0);
        const Real* col = A + size_t(i) * lda + p;
        for (int rr = tid; rr < h; rr += nth) {
            Real sv = col[rr];
            #pragma unroll
            for (int q = 0; q < PW; ++q) dloc[q] += usm[rr + q * ldm] * sv;
        }
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) dloc[q] += __shfl_xor_sync(0xffffffffu, dloc[q], o);
        }
        Real out = Real(0);
        #pragma unroll
        for (int q = 0; q < PW; ++q) out = (lane == q) ? dloc[q] : out;
        if (lane < PW) atomicAdd(&d[lane], out);
        __syncthreads();
        if (tid < PW) Psum[size_t(wt + i) * PW + tid] = d[tid];
        __syncthreads();
    }
}

// block 0 tail: T̃ diag blocks + deferred off-diagonal compound + D⁻¹TD⁻¹ scaling —
// alone, __syncthreads only (updaters do the V/SPV writeout concurrently).
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_pipe_tassemble(
        Real* Tpan, const Real* T16gAll, const Real* PsumAll, Real* M1, const Real* fvL,
        int* pf, int nbl, int tid, int nth) {
    if (tid == 0)
        while (atomicAdd(&pf[16], 0) < nbl - 1) __nanosleep(1);
    __syncthreads();
    __threadfence_system();
    for (int e = tid; e < NB * NB; e += nth) {
        const int cc = e / NB, r = e % NB, sr = r / PW, sc = cc / PW;
        if (sr == sc)
            Tpan[e] = T16gAll[size_t(sr) * PW * PW + (r - sr * PW) + (cc - sc * PW) * PW];
        else if (sr > sc)
            Tpan[e] = Real(0);
    }
    __syncthreads();
    for (int s = 1; s < NB / PW; ++s) {
        const int p = s * PW, wt = NB - p - PW;
        const Real* Ps = PsumAll + size_t(s) * NB * PW;
        const Real* Tg = T16gAll + size_t(s) * PW * PW;
        for (int e = tid; e < p * PW; e += nth) {
            const int i = e % p, q = e / p;
            Real acc = Real(0);
            for (int k = 0; k <= q; ++k) acc += Ps[size_t(wt + i) * PW + k] * Tg[k + q * PW];
            M1[e] = acc;
        }
        __syncthreads();
        for (int e = tid; e < p * PW; e += nth) {
            const int i = e % p, q = e / p;
            Real acc = Real(0);
            for (int k = i; k < p; ++k) acc += Tpan[i + size_t(k) * NB] * M1[k + q * p];
            Tpan[i + size_t(p + q) * NB] = -acc;
        }
        __syncthreads();
    }
    for (int e = tid; e < NB * NB; e += nth) {
        const int cc = e / NB, r = e % NB;
        if (r <= cc) Tpan[e] *= fvL[r] * fvL[cc];
    }
}

// updater tail: scale V in place (v = u/f), β on the diagonal, Y into SPV — rows
// grid-strided over the U updater blocks while block 0 assembles T.
template<typename Real, int NB>
__device__ __noinline__ void d_pipe_writeout(
        Real* A, size_t lda, Real* SPV0, size_t ldspv, Real* fvL, Real* betL,
        const Real* fvG, const Real* betG, int mk, int u, int U, int tid, int nth) {
    if (tid < NB) {
        fvL[tid]  = Real(1) / fvG[tid];
        betL[tid] = betG[tid];
    }
    __syncthreads();
    for (int r = u * nth + tid; r < mk; r += U * nth) {
        Real* SPVr = SPV0 + r;
        for (int cc = 0; cc < NB; ++cc) {
            Real y;
            if (r > cc) {
                Real v = A[r + size_t(cc) * lda] * fvL[cc];
                A[r + size_t(cc) * lda] = v;
                y = v;
            } else if (r == cc) {
                A[r + size_t(cc) * lda] = betL[cc];
                y = Real(1);
            } else {
                y = Real(0);
            }
            if (SPV0) SPVr[size_t(cc) * ldspv] = y;
        }
    }
}

template<typename Real, int NB, int PW = 16>
__global__ __launch_bounds__(512) void k_panel_domino_pipe(DomArgs<Real> a, int* pf) {
    extern __shared__ __align__(16) unsigned char smraw[];
    const int ldm = a.mk + 1;
    Real* base = reinterpret_cast<Real*>(smraw);
    Real* msm = base;
    Real* scr = base + size_t(ldm) * PW;
    Real* Gs = scr;         Real* Ts = scr + 256;    Real* dsl = scr + 512;
    Real* ssm = scr + 768;  Real* sgL = scr + 784;
    Real* fvL = scr + 800;  Real* betL = fvL + NB;   Real* ttL = betL + NB;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = nth >> 5;
    const int bid = blockIdx.x, nbl = gridDim.x;
    const int mk = a.mk;
    Real* A = a.A;
    const size_t lda = a.lda;
    if (bid == 0) {
        unsigned long long tck = 0;
        auto stamp = [&](int slot) {
            if (a.tphase && tid == 0) {
                unsigned long long now = clock64();
                atomicAdd(&a.tphase[slot], now - tck);
                tck = now;
            }
        };
        if (a.tphase && tid == 0) tck = clock64();
        d_pipe_gather<Real, PW>(msm, dsl, sgL, A, lda, ldm, mk,
                                tid, nth, lane, warp, NW);
        stamp(0);
        for (int s = 0; s < NB / PW; ++s) {
            const int p = s * PW, h = mk - p;
            for (int jj = 0; jj < PW; ++jj)
                d_pipe_region<Real, PW>(msm, dsl, ssm, sgL, fvL, betL, ttL,
                                        ldm, p, jj, h, tid, nth, lane, warp, NW);
            stamp(1);
            d_pipe_gram<Real, PW>(msm, Gs, Ts, fvL, ttL, ldm, p, h, lane, warp, NW);
            stamp(2);
            d_pipe_flush<Real, PW>(msm, Ts, fvL, betL, ttL, A, lda,
                                   a.T16g + size_t(s) * PW * PW, a.fv, a.bet, a.tauT,
                                   pf, ldm, p, h, s, tid, nth);
            stamp(3);
            if (s + 1 < NB / PW) {
                d_pipe_selfstage<Real, PW>(msm, Gs, dsl, Ts, fvL, A, lda, pf,
                                           ldm, p, h, s, tid, nth, lane, warp, NW);
                d_pipe_seed<Real, PW>(msm, dsl, sgL, ldm, h - PW, tid, lane, warp, NW);
                stamp(4);
            }
        }
        d_pipe_tassemble<Real, NB, PW>(a.Tpan, a.T16g, a.Psum, msm, fvL, pf, nbl,
                                       tid, nth);
        stamp(6);
    } else {
        const int u = bid - 1, U = nbl - 1;
        for (int s = 0; s < NB / PW; ++s)
            d_pipe_upd<Real, NB, PW>(A, lda, msm, Ts, fvL, dsl, dsl + PW, a.T16g,
                                     a.fv, a.Psum, pf, ldm, mk, s, u, U, tid, nth, lane);
        __threadfence();
        if (tid == 0) {
            atomicAdd(&pf[16], 1);
            while (atomicAdd(&pf[16], 0) < nbl - 1) __nanosleep(1);
        }
        __syncthreads();
        d_pipe_writeout<Real, NB>(A, lda,
                                  a.SPV + size_t(a.spvcol) * a.ldspv + a.row0, a.ldspv,
                                  fvL, betL, a.fv, a.bet, mk, u, U, tid, nth);
    }
}

// ============================================================ de-fused domino
// (roofline path, ROADMAP §5): the fused domino serializes against the far-update
// GEMM waves (coop co-residency) and does its GEMM-shaped boundary work on CUDA
// cores at ~6 TF. De-fuse at WY boundaries: per minipanel, a SMALL column-loop
// kernel (≤16-block cluster / small coop grid — leaves the machine to concurrent
// far GEMMs) alternates with cuBLAS tensor-core boundary GEMMs:
//   k_dom_cols(s)  — 16-column microloop on smem slices (d_dom_region + protocol),
//                    epilogue: flush raw→A, masked ũ→Ug, local Gram→Gs, scalars.
//   GEMM1          — W_s[16×NB] = Ugᵀ·A[p:, :]   (Psum prev-cols + dots, natural ld 16)
//   k_dom_small(s) — T̃16 from (Gs, τ̃) one-warp recurrence; W2 = T̃ᵀ·W[:, pe:]
//   GEMM2          — A[p:, pe:] −= Ug·W2         (rank-16 apply, tensor cores)
//   k_dom_tail     — T̃ compound from W slabs + D⁻¹TD⁻¹ + V/SPV writeout.
template<typename Real, int NB, int PW = 16, bool CL = false>
__global__ void k_dom_cols(DomArgs<Real> a, int s, Real* Ug, Real* Gs) {
    namespace cg = cooperative_groups;
    auto gsync = [] {
        if constexpr (CL) cg::this_cluster().sync();
        else              cg::this_grid().sync();
    };
    extern __shared__ __align__(16) unsigned char smraw[];
    const int p = s * PW, pe = p + PW;
    DomCtx<Real, NB, PW> c;
    c.a = a;
    c.msm = reinterpret_cast<Real*>(smraw);
    c.msmN = nullptr;
    c.W2 = nullptr; c.Ts = nullptr; c.gsm = nullptr;
    c.ssm = c.msm + size_t(a.rpb) * PW;
    Real* psm = c.ssm + PW;
    c.fv = psm; c.bet = psm + NB; c.tauT = psm + 2 * NB;
    c.fsm = nullptr;
    c.dslot = a.dslot; c.prow = a.prow; c.sig = a.sig;
    c.tid = threadIdx.x; c.nth = blockDim.x;
    c.lane = c.tid & 31; c.warp = c.tid >> 5; c.NW = c.nth >> 5;
    c.bid = blockIdx.x; c.nbl = gridDim.x;
    c.r0 = p + c.bid * a.rpb; c.r1 = min(a.mk, c.r0 + a.rpb); c.r1all = c.r1;
    const int rpb = a.rpb, r0 = c.r0, r1 = c.r1, span = r1 - r0;
    const int tid = c.tid, nth = c.nth, lane = c.lane, warp = c.warp, NW = c.NW;
    // ---- prologue: zero protocol, stage slice, seed col-p (dots, σ, prow)
    if (c.bid == 0) {
        for (int i = tid; i < PW * PW; i += nth) c.dslot[i] = Real(0);
        if (tid < PW) { c.sig[p + tid] = Real(0); }
    }
    for (int e = tid; e < span * PW; e += nth) {
        int q = e / span, rr = e % span;
        c.msm[rr + q * rpb] = a.A[(r0 + rr) + size_t(p + q) * a.lda];
    }
    gsync();
    {
        for (int cc = 1 + warp; cc < PW; cc += NW) {
            Real acc = Real(0);
            for (int r = max(r0, p + 1) + lane; r < r1; r += 32)
                acc += c.msm[r - r0] * c.msm[(r - r0) + cc * rpb];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) atomicAdd(&c.dslot[cc], acc);
        }
        if (warp == 0) {
            Real sa = Real(0);
            for (int r = max(r0, p + 1) + lane; r < r1; r += 32) {
                Real v = c.msm[r - r0];
                sa += v * v;
            }
            #pragma unroll
            for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
            if (lane == 0) atomicAdd(&c.sig[p], sa);
        }
        if (r0 <= p && p < r1 && tid < PW)
            c.prow[(p & 1) * PW + tid] = c.msm[(p - r0) + size_t(tid) * rpb];
    }
    gsync();
    // ---- 16 column regions (existing lockstep machinery, one gsync per column)
    for (int jj = 0; jj < PW; ++jj) {
        d_dom_region<Real, NB, PW>(c, p, jj);
        gsync();
    }
    // ---- epilogue (v2): flush raw values only — ũ is never materialized; the
    // boundary GEMMs run on raw A and k_dom_small corrects the 16 triangle rows
    // algebraically (W_true = W_raw − Mᵀ·A[p:p+16,:], G̃ from W_own − Lᵀ·M).
    for (int e = tid; e < span * PW; e += nth) {
        int q = e / span, rr = e % span;
        a.A[(r0 + rr) + size_t(p + q) * a.lda] = c.msm[rr + q * rpb];
    }
    if (c.bid == 0 && tid < PW) {
        a.fv[p + tid]   = c.fv[p + tid];
        a.bet[p + tid]  = c.bet[p + tid];
        a.tauT[p + tid] = c.tauT[p + tid];
    }
    (void)Ug; (void)Gs; (void)lane; (void)warp; (void)NW;
}

// v2 boundary brain (one block): correct the raw-A dot GEMM on the 16 triangle rows
// (W_true = W_raw − Mᵀ·Arow, M = R-strict-upper + β diag), derive the Gram
// (G̃ = W_true_own − Lᵀ·M, L = masked lower triangle), run the T̃16 recurrence, form
// W2 = T̃ᵀ·W_true[:, pe:], and stash M for the post-apply fixup (k_dom_fix).
template<typename Real, int NB, int PW = 16>
__global__ void k_dom_small(DomArgs<Real> a, int s, Real* Mout, Real* W, Real* W2) {
    __shared__ Real Arow[PW * NB], M[PW * PW], L[PW * PW], G[PW * PW],
                    Ts[PW * PW], tts[PW];
    const int p = s * PW, pe = p + PW, wt = NB - pe;
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5;
    const Real* A = a.A;
    const size_t lda = a.lda;
    for (int e = tid; e < PW * NB; e += nth) {       // Arow[i + t·PW] = A[p+i, t]
        int t = e / PW, i = e % PW;
        Arow[e] = A[(p + i) + size_t(t) * lda];
    }
    if (tid < PW) tts[tid] = a.tauT[p + tid];
    __syncthreads();
    for (int e = tid; e < PW * PW; e += nth) {       // M[i + k·PW], L[i + k·PW]
        int k = e / PW, i = e % PW;
        Real tri = Arow[i + size_t(p + k) * PW];
        M[e] = (i < k) ? tri : (i == k ? a.bet[p + k] : Real(0));
        L[e] = (i > k) ? tri : (i == k ? a.fv[p + k] : Real(0));
    }
    __syncthreads();
    for (int e = tid; e < PW * NB; e += nth) {       // W ← W_raw − Mᵀ·Arow (in place)
        int t = e / PW, k = e % PW;
        Real acc = Real(0);
        #pragma unroll
        for (int i = 0; i < PW; ++i) acc += M[i + k * PW] * Arow[i + size_t(t) * PW];
        W[k + size_t(t) * PW] -= acc;
    }
    __syncthreads();
    for (int e = tid; e < PW * PW; e += nth) {       // G̃[k + j·PW] (k < j used)
        int j = e / PW, k = e % PW;
        Real acc = W[k + size_t(p + j) * PW];
        #pragma unroll
        for (int i = 0; i < PW; ++i) acc -= L[i + k * PW] * M[i + j * PW];
        G[e] = acc;
    }
    if (tid < PW * PW) Mout[tid] = M[tid];
    __syncthreads();
    if (warp == 0)
        for (int jT = 0; jT < PW; ++jT) {
            const Real tj = tts[jT];
            Real acc = Real(0);
            if (lane < jT)
                for (int k = lane; k < jT; ++k) acc += Ts[lane + k * PW] * G[k + jT * PW];
            __syncwarp();
            if (lane < jT) Ts[lane + jT * PW] = -tj * acc;
            else if (lane == jT) Ts[jT + jT * PW] = tj;
            __syncwarp();
        }
    __syncthreads();
    Real* T16g = a.T16g + size_t(s) * PW * PW;
    for (int e = tid; e < PW * PW; e += nth) {
        int cc = e / PW, r = e % PW;
        T16g[e] = (r <= cc) ? Ts[e] : Real(0);
    }
    for (int e = tid; e < PW * wt; e += nth) {
        int c2 = e / PW, k = e % PW;
        Real acc = Real(0);
        for (int k2 = 0; k2 <= k; ++k2)
            acc += Ts[k2 + k * PW] * W[k2 + size_t(pe + c2) * PW];
        W2[k + size_t(c2) * PW] = acc;
    }
}

// v2 post-apply fixup: GEMM2 applied raw·W2; undo the triangle-row excess:
// A[p:p+16, pe:] += M·W2 (E = M on those rows).
template<typename Real, int NB, int PW = 16>
__global__ void k_dom_fix(DomArgs<Real> a, int s, const Real* Min, const Real* W2) {
    __shared__ Real M[PW * PW];
    const int p = s * PW, pe = p + PW, wt = NB - pe;
    const int tid = threadIdx.x, nth = blockDim.x;
    for (int e = tid; e < PW * PW; e += nth) M[e] = Min[e];
    __syncthreads();
    Real* A = a.A;
    const size_t lda = a.lda;
    for (int e = tid; e < PW * wt; e += nth) {
        int c2 = e / PW, i = e % PW;
        Real acc = Real(0);
        #pragma unroll
        for (int k = 0; k < PW; ++k) acc += M[i + k * PW] * W2[k + size_t(c2) * PW];
        A[(p + i) + size_t(pe + c2) * lda] += acc;
    }
}

// panel tail: block 0 assembles T̃ (diag blocks + deferred compound from the natural-
// layout W slabs) and scales D⁻¹TD⁻¹; the other blocks scale V and write SPV/Y.
template<typename Real, int NB, int PW = 16>
__global__ void k_dom_tail(DomArgs<Real> a) {
    __shared__ Real fsm[NB], M1s[(NB - PW) * PW];
    const int tid = threadIdx.x, nth = blockDim.x;
    if (blockIdx.x == 0) {
        if (tid < NB) fsm[tid] = a.fv[tid];
        Real* Tpan = a.Tpan;
        for (int e = tid; e < NB * NB; e += nth) {
            const int cc = e / NB, r = e % NB, sr = r / PW, sc = cc / PW;
            if (sr == sc)
                Tpan[e] = a.T16g[size_t(sr) * PW * PW + (r - sr * PW) + (cc - sc * PW) * PW];
            else if (sr > sc)
                Tpan[e] = Real(0);
        }
        __syncthreads();
        for (int s = 1; s < NB / PW; ++s) {
            const int p = s * PW;
            const Real* Ws = a.Psum + size_t(s) * NB * PW;   // natural: W[k + t·PW]
            const Real* Tg = a.T16g + size_t(s) * PW * PW;
            for (int e = tid; e < p * PW; e += nth) {
                const int i = e % p, q = e / p;
                Real acc = Real(0);
                for (int k = 0; k <= q; ++k) acc += Ws[k + size_t(i) * PW] * Tg[k + q * PW];
                M1s[e] = acc;
            }
            __syncthreads();
            for (int e = tid; e < p * PW; e += nth) {
                const int i = e % p, q = e / p;
                Real acc = Real(0);
                for (int k = i; k < p; ++k) acc += Tpan[i + size_t(k) * NB] * M1s[k + q * p];
                Tpan[i + size_t(p + q) * NB] = -acc;
            }
            __syncthreads();
        }
        for (int e = tid; e < NB * NB; e += nth) {
            const int cc = e / NB, r = e % NB;
            if (r <= cc) Tpan[e] *= fsm[r] * fsm[cc];
        }
    } else {
        __shared__ Real bsm[NB];
        if (tid < NB) { fsm[tid] = Real(1) / a.fv[tid]; bsm[tid] = a.bet[tid]; }
        __syncthreads();
        const int u = blockIdx.x - 1, U = gridDim.x - 1;
        Real* A = a.A;
        const size_t lda = a.lda;
        for (int r = u * nth + tid; r < a.mk; r += U * nth) {
            Real* SPVr = a.SPV + size_t(a.spvcol) * a.ldspv + a.row0 + r;
            for (int cc = 0; cc < NB; ++cc) {
                Real y;
                if (r > cc) {
                    Real v = A[r + size_t(cc) * lda] * fsm[cc];
                    A[r + size_t(cc) * lda] = v;
                    y = v;
                } else if (r == cc) {
                    A[r + size_t(cc) * lda] = bsm[cc];
                    y = Real(1);
                } else {
                    y = Real(0);
                }
                SPVr[size_t(cc) * a.ldspv] = y;
            }
        }
    }
}

// ============================================================ DSM-pipe domino
// (ROADMAP §6; spec §11 "cluster-DSM as the layer dimension", realized): ONE
// cooperative+cluster launch per panel with a SMALL grid — a C-block hardware
// cluster owns the minipanel column loop (row slices in each block's smem,
// per-column cross-block reduction via DSM reads of published partials, ONE
// cluster.sync per column), while U updater blocks in the same kernel saturate
// the boundary WY work behind gmem doorbells (pipe protocol). Total C+U ≈ 30-60
// blocks: the far-update GEMM waves fill the remaining SMs while the kernel
// stays resident, so far never gains exclusive residency (no launch ping-pong).

// Cache-bypassing load: forces ld.global.cv which bypasses L1 and invalidates
// stale L1 entries. Required for cross-block gmem reads where another block
// wrote the data between this block's prior access and current read. On Hopper,
// L1 is NOT coherent for global memory — __threadfence only publishes writes to
// L2, it does NOT invalidate remote L1 caches.
template<typename T>
__device__ __forceinline__ T load_fresh(const T* addr) {
    return *reinterpret_cast<const volatile T*>(addr);
}

template<typename Real>
struct DsmPub {                      // DSM-readable, fixed offset in every block
    Real dslP[2][16];                // partial dots (parity jj&1)
    Real sgP[2];                     // partial σ (parity)
    Real prow[2][16];                // pivot-row snapshot (row-owner publishes)
    Real GsP[256];                   // Gram pair partials [k + jT·16], k<jT
    Real DsP[256];                   // self-stage dot partials [k + q·16]
    Real Ts[256];                    // rank 0: T̃16 after recurrence
    Real W2[256];                    // rank 0: W2 = T̃ᵀ·Ds
};

// one column region across the col cluster: read the published (σ, dots, prow),
// form reflector j redundantly, apply to own rows, publish next column's partials.
template<typename Real, int NB, int PW, typename PubAt>
__device__ __forceinline__ void d_dsm_region(
        Real* msm, DsmPub<Real>* pub, PubAt pubAt, Real* ssm,
        Real* fvL, Real* betL, Real* ttL,
        int rpb, int r0, int r1, int C, int p, int jj,
        int tid, int nth, int lane, int warp, int NW) {
    const int j = p + jj, par = jj & 1, parN = (jj + 1) & 1;
    const int owner = j / rpb;
    // step 1a: warp w pulls rank w's partials (parallel DSM reads) into the
    // staging slots ssm[18..34] (zeroed by the previous region's step 1b / init)
    for (int rk = warp; rk < C; rk += NW) {
        DsmPub<Real>* pr = pubAt(rk);
        Real v = (lane < PW) ? pr->dslP[par][lane]
               : (lane == 16 ? pr->sgP[par] : Real(0));
        if (lane <= 16) atomicAdd(&ssm[PW + 2 + lane], v);
    }
    __syncthreads();
    if (warp == 0) {                 // step 1b: scalars from the staged sums
        Real dsum = (lane < PW) ? ssm[PW + 2 + lane] : Real(0);
        Real sg = (lane == 16) ? ssm[PW + 2 + 16] : Real(0);
        if (lane <= 16) ssm[PW + 2 + lane] = Real(0);   // rezero for next region
        Real prc = (lane < PW) ? pubAt(owner)->prow[par][lane] : Real(0);
        const Real al = __shfl_sync(0xffffffffu, prc, jj);
        const Real sgv = __shfl_sync(0xffffffffu, sg, 16);
        Real beta = al, f = Real(1), tt = Real(0);
        if (sgv > Real(0)) {
            Real mu = sqrt(al * al + sgv);
            beta = (al >= Real(0)) ? -mu : mu;
            f = al - beta;
            tt = Real(-1) / (beta * f);
        }
        if (lane == 0) { betL[j] = beta; fvL[j] = f; ttL[j] = tt; }
        if (lane < PW) {
            if (lane > jj) ssm[lane] = tt * (f * prc + dsum);
            pub->dslP[parN][lane] = Real(0);
        }
        if (lane == 16) pub->sgP[parN] = Real(0);
    }
    __syncthreads();
    if (jj + 1 >= PW) return;        // last column: scalars only
    const Real f = fvL[j], s1 = ssm[jj + 1];
    Real* mj  = msm + jj * rpb;
    Real* mc1 = msm + (jj + 1) * rpb;
    const int rb = max(r0, j);
    // phase 1: col jj+1 on own rows (publish its pivot-row value as α_{j+1})
    for (int rr = rb - r0 + tid; rr < r1 - r0; rr += nth) {
        const int r = r0 + rr;
        Real vj = (r == j) ? f : mj[rr];
        Real nv = mc1[rr] - s1 * vj;
        mc1[rr] = nv;
        if (r == j + 1) pub->prow[parN][jj + 1] = nv;
    }
    __syncthreads();
    // phase 2: warp-per-column over own rows, 4x unrolled; the first 32-row
    // stride handles the pivot rows (j, j+1) and the prow snapshot, the rest is
    // a clean latency-pipelined loop.
    for (int cc = jj + 2 + warp; cc < PW; cc += NW) {
        Real* mcc = msm + cc * rpb;
        const Real sc = ssm[cc];
        Real acc = Real(0);
        int rr = rb - r0 + lane;
        if (rr < r1 - r0) {                        // prologue stride (may hold j/j+1)
            const int r = r0 + rr;
            Real vj = (r == j) ? f : mj[rr];
            Real nv = mcc[rr] - sc * vj;
            mcc[rr] = nv;
            if (r > j + 1) acc += mc1[rr] * nv;
            if (r == j + 1) pub->prow[parN][cc] = nv;
            rr += 32;
        }
        const int hh = r1 - r0;
        for (; rr + 96 < hh; rr += 128) {
            Real v0 = mj[rr],       v1 = mj[rr + 32];
            Real v2 = mj[rr + 64],  v3 = mj[rr + 96];
            Real n0 = mcc[rr]      - sc * v0, n1 = mcc[rr + 32] - sc * v1;
            Real n2 = mcc[rr + 64] - sc * v2, n3 = mcc[rr + 96] - sc * v3;
            mcc[rr] = n0; mcc[rr + 32] = n1; mcc[rr + 64] = n2; mcc[rr + 96] = n3;
            acc += mc1[rr] * n0 + mc1[rr + 32] * n1
                 + mc1[rr + 64] * n2 + mc1[rr + 96] * n3;
        }
        for (; rr < hh; rr += 32) {
            Real nv = mcc[rr] - sc * mj[rr];
            mcc[rr] = nv;
            acc += mc1[rr] * nv;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) atomicAdd(&pub->dslP[parN][cc], acc);
    }
    if (warp == NW - 1) {            // σ_{j+1} over own rows > j+1
        Real sa = Real(0);
        for (int rr = max(rb, j + 2) - r0 + lane; rr < r1 - r0; rr += 32) {
            Real v = mc1[rr];
            sa += v * v;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) sa += __shfl_xor_sync(0xffffffffu, sa, o);
        if (lane == 0) atomicAdd(&pub->sgP[parN], sa);
    }
    // Every caller immediately executes cluster_group::sync().  That barrier
    // includes all threads in this CTA and supplies the required shared-memory
    // ordering, so a preceding CTA-only barrier merely paid twice per column.
}

// boundary across the col cluster (3 cluster.syncs): Gram+Ds partials + flush →
// rank-0 reduce/T̃16/W2 + doorbell → all: self-stage next minipanel into own rows
// (no row shift — slices are absolute) + seed partials.
template<typename Real, int NB, int PW, typename PubAt, typename Cl>
__device__ __forceinline__ void d_dsm_boundary(
        DomArgs<Real>& a, Real* msm, DsmPub<Real>* pub, PubAt pubAt,
        Real* fvL, Real* betL, Real* ttL,
        int rpb, int r0, int r1, int C, int rank, int s, int* pf, Cl& cl,
        int tid, int nth, int lane, int warp, int NW) {
    const int p = s * PW, pe = p + PW;
    const bool last = (s + 1 >= NB / PW);
    Real* A = a.A;
    const size_t lda = a.lda;
    // stCnt gate: cols [pe, pe+PW) must have the updaters' boundary-(s−1) applied
    // BEFORE we read them (Ds dots + self-stage). Rung one boundary earlier.
    if (!last && s >= 1) {
        if (tid == 0)
            while (atomicAdd(&pf[8 + (s - 1)], 0) < PW) __nanosleep(1);
        __syncthreads();
        __threadfence();
    }
    // ---- phase A: Gram partials (pairs k<jT over own rows ≥ p+jT), Ds partials,
    //      flush own slice rows ≥ p to gmem
    for (int pr_ = warp; pr_ < PW * (PW - 1) / 2; pr_ += NW) {
        int jT = 1, rem = pr_;
        while (rem >= jT) { rem -= jT; ++jT; }
        const int k = rem;
        const Real* uk = msm + k * rpb;
        const Real* uj = msm + jT * rpb;
        const int rb = max(r0, p + jT);
        Real acc = Real(0);
        for (int rr = rb - r0 + lane; rr < r1 - r0; rr += 32) {
            const int r = r0 + rr;
            Real vj = (r == p + jT) ? fvL[p + jT] : uj[rr];
            acc += uk[rr] * vj;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) pub->GsP[k + jT * PW] = acc;
    }
    if (!last)
        for (int q = warp; q < PW; q += NW) {   // Ds partials: warp per next column
            const Real* col = A + size_t(pe + q) * lda;
            Real dv[PW];
            #pragma unroll
            for (int k = 0; k < PW; ++k) dv[k] = Real(0);
            for (int rr = max(r0, p) - r0 + lane; rr < r1 - r0; rr += 32) {
                const int r = r0 + rr;
                const Real sv = load_fresh(&col[r]);
                if (r < pe) {
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) {
                        Real uk = (r < p + k) ? Real(0)
                                 : (r == p + k ? fvL[p + k] : msm[rr + k * rpb]);
                        dv[k] += uk * sv;
                    }
                } else {
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) dv[k] += msm[rr + k * rpb] * sv;
                }
            }
            #pragma unroll
            for (int k = 0; k < PW; ++k) {
                #pragma unroll
                for (int o = 16; o; o >>= 1) dv[k] += __shfl_xor_sync(0xffffffffu, dv[k], o);
            }
            Real out = Real(0);
            #pragma unroll
            for (int k = 0; k < PW; ++k) out = (lane == k) ? dv[k] : out;
            if (lane < PW) pub->DsP[lane + q * PW] = out;
        }
    {   // flush own slice (rows ≥ p only are live for this minipanel)
        const int rb = max(r0, p), span = r1 - rb;
        for (int e = tid; e < span * PW; e += nth) {
            int q = e / span, rr = e % span;
            A[(rb + rr) + size_t(p + q) * lda] = msm[(rb - r0 + rr) + q * rpb];
        }
    }
    cl.sync();
    // ---- phase B: rank 0 reduces, T̃16 recurrence, W2; gmem scalars; doorbell
    if (rank == 0) {
        for (int e = tid; e < PW * PW; e += nth) {
            Real acc = Real(0);
            for (int rk = 0; rk < C; ++rk) acc += pubAt(rk)->GsP[e];
            pub->GsP[e] = acc;                  // rank-0 slot becomes the total
        }
        if (!last)
            for (int e = tid; e < PW * PW; e += nth) {
                Real acc = Real(0);
                for (int rk = 0; rk < C; ++rk) acc += pubAt(rk)->DsP[e];
                pub->DsP[e] = acc;
            }
        __syncthreads();
        if (warp == 0)
            for (int jT = 0; jT < PW; ++jT) {
                const Real tj = ttL[p + jT];
                Real acc = Real(0);
                if (lane < jT)
                    for (int k = lane; k < jT; ++k)
                        acc += pub->Ts[lane + k * PW] * pub->GsP[k + jT * PW];
                __syncwarp();
                if (lane < jT) pub->Ts[lane + jT * PW] = -tj * acc;
                else if (lane == jT) pub->Ts[jT + jT * PW] = tj;
                __syncwarp();
            }
        __syncthreads();
        if (!last)
            for (int e = tid; e < PW * PW; e += nth) {   // W2 = T̃ᵀ·Ds
                const int q = e / PW, k = e % PW;
                Real acc = Real(0);
                for (int k2 = 0; k2 <= k; ++k2)
                    acc += pub->Ts[k2 + k * PW] * pub->DsP[k2 + q * PW];
                pub->W2[k + q * PW] = acc;
            }
        Real* T16g = a.T16g + size_t(s) * PW * PW;
        for (int e = tid; e < PW * PW; e += nth) {
            int cc = e / PW, r = e % PW;
            T16g[e] = (r <= cc) ? pub->Ts[e] : Real(0);
        }
        if (tid < PW) {
            a.fv[p + tid]   = fvL[p + tid];
            a.bet[p + tid]  = betL[p + tid];
            a.tauT[p + tid] = ttL[p + tid];
        }
        __syncthreads();
        __threadfence();
        if (tid == 0) atomicExch(&pf[s], 1);
    }
    cl.sync();
    if (last) return;
    // ---- phase C: self-stage minipanel s+1 into own rows; publish seeds
    {
        DsmPub<Real>* p0 = pubAt(0);
        // copy W2 to local ssm-free smem? read via DSM per use — W2 read PW times
        // per row; cache in registers per thread: w2r[q] needed per (k,q)... read
        // via DSM directly (L1-cached? no) — stage into own pub->W2 first.
        for (int e = tid; e < PW * PW; e += nth)
            if (rank != 0) pub->W2[e] = p0->W2[e];
    }
    __syncthreads();
    {
        const int rb = max(r0, p);
        for (int rr = rb - r0 + tid; rr < r1 - r0; rr += nth) {
            const int r = r0 + rr;
            Real val[PW];
            #pragma unroll
            for (int q = 0; q < PW; ++q) val[q] = load_fresh(&A[r + size_t(pe + q) * lda]);
            if (r < pe) {
                #pragma unroll
                for (int k = 0; k < PW; ++k) {
                    Real uk = (r < p + k) ? Real(0)
                             : (r == p + k ? fvL[p + k] : msm[rr + k * rpb]);
                    #pragma unroll
                    for (int q = 0; q < PW; ++q) val[q] -= uk * pub->W2[k + q * PW];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) {
                    const Real uk = msm[rr + k * rpb];
                    #pragma unroll
                    for (int q = 0; q < PW; ++q) val[q] -= uk * pub->W2[k + q * PW];
                }
            }
            if (r < pe) {          // final R rows of the next minipanel's columns
                #pragma unroll
                for (int q = 0; q < PW; ++q) A[r + size_t(pe + q) * lda] = val[q];
            } else {
                #pragma unroll
                for (int q = 0; q < PW; ++q) msm[rr + q * rpb] = val[q];
            }
        }
    }
    __syncthreads();
    {   // seeds for region 0 of minipanel s+1 (parity slot 0, zeroed by region 15)
        Real dp[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) dp[q] = Real(0);
        const int rb = max(r0, pe + 1);
        for (int rr = rb - r0 + tid; rr < r1 - r0; rr += nth) {
            const Real v0 = msm[rr];
            dp[0] += v0 * v0;
            #pragma unroll
            for (int q = 1; q < PW; ++q) dp[q] += v0 * msm[rr + q * rpb];
        }
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) dp[q] += __shfl_xor_sync(0xffffffffu, dp[q], o);
        }
        Real out = Real(0);
        #pragma unroll
        for (int q = 0; q < PW; ++q) out = (lane == q) ? dp[q] : out;
        if (lane < PW) {
            if (lane == 0) atomicAdd(&pub->sgP[0], out);
            else           atomicAdd(&pub->dslP[0][lane], out);
        }
        if (r0 <= pe && pe < r1 && tid < PW)
            pub->prow[0][tid] = msm[(pe - r0) + size_t(tid) * rpb];
    }
    cl.sync();
}

// in-kernel updater, large-h variant: no full-ũ cache (mk up to ~56K rows) —
// direct gmem ũ reads amortized over column GROUPS of 4 (dloc[4][16] registers).
template<typename Real, int NB, int PW>
__device__ __noinline__ void d_dsm_upd(
        Real* A, size_t lda, Real* Ts, Real* fvT, Real* dsh, Real* wsh,
        const Real* T16gAll, const Real* fvG, Real* PsumAll, int* pf,
        int mk, int s, int u, int U, int tid, int nth, int lane) {
    const int p = s * PW, pe = p + PW, wt = NB - pe;
    const int h = mk - p;
    if (tid == 0)
        while (atomicAdd(&pf[s], 0) == 0) __nanosleep(1);
    __syncthreads();
    __threadfence();
    for (int e = tid; e < PW * PW; e += nth) Ts[e] = load_fresh(&T16gAll[size_t(s) * PW * PW + e]);
    if (tid < PW) fvT[tid] = load_fresh(&fvG[p + tid]);
    __syncthreads();
    // owned trailing columns: contiguous chunk of [pe+PW, NB)
    const int nc = NB - (pe + PW);
    const int chunk = nc > 0 ? (nc + U - 1) / U : 0;
    const int c0 = pe + PW + u * chunk, c1 = min(NB, c0 + chunk);
    for (int g0 = c0; g0 < c1; g0 += 4) {
        const int G = min(4, c1 - g0);
        if (tid < 64) dsh[tid] = Real(0);
        __syncthreads();
        Real dloc[4][PW];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            #pragma unroll
            for (int k = 0; k < PW; ++k) dloc[g][k] = Real(0);
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    const Real sv = A[r + size_t(g0 + g) * lda];
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) dloc[g][k] += uk[k] * sv;
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int k = 0; k < PW; ++k) {
                #pragma unroll
                for (int o = 16; o; o >>= 1)
                    dloc[g][k] += __shfl_xor_sync(0xffffffffu, dloc[g][k], o);
            }
        }
        {   // lane e < 16 owns k=e for each g: atomicAdd block partials
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                Real out = Real(0);
                #pragma unroll
                for (int k = 0; k < PW; ++k) out = (lane == k) ? dloc[g][k] : out;
                if (lane < PW && g < G) atomicAdd(&dsh[g * PW + lane], out);
            }
        }
        __syncthreads();
        if (tid < 64) {
            const int g = tid / PW, k = tid % PW;
            Real acc = Real(0);
            for (int k2 = 0; k2 <= k; ++k2) acc += Ts[k2 + k * PW] * dsh[g * PW + k2];
            wsh[g * PW + k] = acc;
        }
        __syncthreads();
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    Real acc = Real(0);
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) acc += uk[k] * wsh[g * PW + k];
                    A[r + size_t(g0 + g) * lda] -= acc;
                }
            }
        }
        __syncthreads();
        const int lo = max(g0, pe + PW), hi = min(g0 + G, pe + 2 * PW);
        if (hi > lo) {
            __threadfence();
            if (tid == 0) atomicAdd(&pf[8 + s], hi - lo);
        }
        __syncthreads();
    }
    // prev-column Psum dot slabs for the deferred T̃ compound
    Real* Psum = PsumAll + size_t(s) * NB * PW;
    const int chunkP = p > 0 ? (p + U - 1) / U : 0;
    const int i0 = u * chunkP, i1 = min(p, i0 + chunkP);
    for (int g0 = i0; g0 < i1; g0 += 4) {
        const int G = min(4, i1 - g0);
        if (tid < 64) dsh[tid] = Real(0);
        __syncthreads();
        Real dloc[4][PW];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            #pragma unroll
            for (int k = 0; k < PW; ++k) dloc[g][k] = Real(0);
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    const Real sv = load_fresh(&A[r + size_t(g0 + g) * lda]);
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) dloc[g][k] += uk[k] * sv;
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int k = 0; k < PW; ++k) {
                #pragma unroll
                for (int o = 16; o; o >>= 1)
                    dloc[g][k] += __shfl_xor_sync(0xffffffffu, dloc[g][k], o);
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            Real out = Real(0);
            #pragma unroll
            for (int k = 0; k < PW; ++k) out = (lane == k) ? dloc[g][k] : out;
            if (lane < PW && g < G) atomicAdd(&dsh[g * PW + lane], out);
        }
        __syncthreads();
        if (tid < PW * G) {
            const int g = tid / PW, k = tid % PW;
            Psum[size_t(wt + g0 + g) * PW + k] = dsh[g * PW + k];
        }
        __syncthreads();
    }
}

// DSM column/updater carrier.  A 16-column minipanel maps to exactly 16 warps;
// one 512-thread CTA is therefore the uniform work quantum for every tier.
template<typename Real, int NB, int PW = 16>
__global__ __launch_bounds__(512, 1)
void k_panel_dsm(DomArgs<Real> a, int* pf, int C) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(16) unsigned char smraw[];
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = nth >> 5;
    const int bid = blockIdx.x, nbl = gridDim.x;
    const int mk = a.mk, rpb = a.rpb;
    Real* A = a.A;
    const size_t lda = a.lda;
    Real* msm = reinterpret_cast<Real*>(smraw);
    DsmPub<Real>* pub = reinterpret_cast<DsmPub<Real>*>(msm + size_t(rpb) * PW);
    Real* fvL  = reinterpret_cast<Real*>(pub + 1);
    Real* betL = fvL + NB;
    Real* ttL  = betL + NB;
    Real* ssm  = ttL + NB;                     // 16 + spare

    if (bid >= C) {   // ------------------------------------------------ updaters
        const int u = bid - C, U = nbl - C;
        Real* Ts  = msm;                       // scratch layout for updaters
        Real* dsh = msm + 256;
        Real* wsh = msm + 320;
        Real* fvT = msm + 384;
        for (int s = 0; s < NB / PW; ++s)
            d_dsm_upd<Real, NB, PW>(A, lda, Ts, fvT, dsh, wsh, a.T16g, a.fv,
                                    a.Psum, pf, mk, s, u, U, tid, nth, lane);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            atomicAdd(&pf[16], 1);
            while (atomicAdd(&pf[16], 0) < nbl - C) { __nanosleep(1); }
        }
        __syncthreads();
        d_pipe_writeout<Real, NB>(A, lda,
                                  a.SPV + size_t(a.spvcol) * a.ldspv + a.row0, a.ldspv,
                                  fvL, betL, a.fv, a.bet, mk, u, U, tid, nth);
        return;
    }
    // ---------------------------------------------------------------- col cluster
    cg::cluster_group cl = cg::this_cluster();
    const int rank = cl.block_rank();
    const int r0 = rank * rpb, r1 = min(mk, r0 + rpb);
    auto pubAt = [&](int rk) -> DsmPub<Real>* {
        return reinterpret_cast<DsmPub<Real>*>(cl.map_shared_rank((void*)pub, rk));
    };
    for (int e = tid; e < (r1 - r0) * PW; e += nth) {
        int q = e / (r1 - r0), rr = e % (r1 - r0);
        msm[rr + q * rpb] = A[(r0 + rr) + size_t(q) * lda];
    }
    if (tid < PW) { pub->dslP[0][tid] = Real(0); if (tid == 0) pub->sgP[0] = Real(0); }
    if (tid >= PW && tid < PW + 17) ssm[PW + 2 + (tid - PW)] = Real(0);  // staging
    __syncthreads();
    {   // seed col-0 partials: dots vs cols 1..15 and σ over own rows ≥ 1
        Real dp[PW];
        #pragma unroll
        for (int q = 0; q < PW; ++q) dp[q] = Real(0);
        for (int rr = max(r0, 1) - r0 + tid; rr < r1 - r0; rr += nth) {
            const Real v0 = msm[rr];
            dp[0] += v0 * v0;
            #pragma unroll
            for (int q = 1; q < PW; ++q) dp[q] += v0 * msm[rr + q * rpb];
        }
        #pragma unroll
        for (int q = 0; q < PW; ++q) {
            #pragma unroll
            for (int o = 16; o; o >>= 1) dp[q] += __shfl_xor_sync(0xffffffffu, dp[q], o);
        }
        Real out = Real(0);
        #pragma unroll
        for (int q = 0; q < PW; ++q) out = (lane == q) ? dp[q] : out;
        if (lane < PW) {
            if (lane == 0) atomicAdd(&pub->sgP[0], out);
            else           atomicAdd(&pub->dslP[0][lane], out);
        }
        if (r0 == 0 && tid < PW) pub->prow[0][tid] = msm[size_t(tid) * rpb];
    }
    cl.sync();
    for (int s = 0; s < NB / PW; ++s) {
        const int p = s * PW;
        for (int jj = 0; jj < PW; ++jj) {
            d_dsm_region<Real, NB, PW>(msm, pub, pubAt, ssm, fvL, betL, ttL,
                                       rpb, r0, r1, C, p, jj,
                                       tid, nth, lane, warp, NW);
            cl.sync();
        }
        d_dsm_boundary<Real, NB, PW>(a, msm, pub, pubAt, fvL, betL, ttL,
                                     rpb, r0, r1, C, rank, s, pf, cl,
                                     tid, nth, lane, warp, NW);
    }
    if (rank == 0)
        d_pipe_tassemble<Real, NB, PW>(a.Tpan, a.T16g, a.Psum, msm, fvL, pf,
                                       nbl - C + 1, tid, nth);
}

// ============================================================ STAGE 2: k_super_dsm
// (ROADMAP §6.2): ONE resident coop+cluster kernel per SUPER-PANEL — the col
// cluster and U updaters live across all npan panels, so cuBLAS storms (far
// chunks on s1, Tsp on sP, rest-slices on sR) can never steal exclusive
// residency between panels. The updaters' trailing range extends into the NEXT
// panel's block (in-kernel narrow = the look-ahead tree-apply; s=7 starts at pe
// so the next panel's first minipanel is covered and rings nextCnt). The wider
// intra-super-panel "rest" updates stay on host cuBLAS (sR), gated by in-kernel
// flags; panel k+2 gates on restDone[k] (classic d=1 pipeline, flag-mapped).
//
// pf layout per panel k (base k·20): [0..7] uflag, [8..15] stCnt, [16] donePan,
// [17] nextCnt, [18] tassDone (host Tsp/rest gate), [19] writeoutDone counter.
// pf[320 + k] = restDone[k] (set by k_set_flag on sR after the rest slice).
__global__ inline void k_wait_cnt(volatile int* f, int tgt) {
    while (atomicAdd((int*)f, 0) < tgt) __nanosleep(1);
}

template<typename Real, int NB, int PW>
__device__ __noinline__ void d_super_upd(
        Real* A, size_t lda, Real* Ts, Real* fvT, Real* dsh, Real* wsh,
        const Real* T16g, const Real* fvG, Real* Psum, int* pfk, int* restGate,
        int mk, int s, int u, int U, int nb2, int ncTotal, int tid, int nth, int lane) {
    const int p = s * PW, pe = p + PW, wt = NB - pe;
    const int h = mk - p;
    if (tid == 0)
        while (atomicAdd(&pfk[s], 0) == 0) __nanosleep(1);
    __syncthreads();
    __threadfence();
    for (int e = tid; e < PW * PW; e += nth) Ts[e] = load_fresh(&T16g[e]);
    if (tid < PW) fvT[tid] = load_fresh(&fvG[p + tid]);
    __syncthreads();
    const int cs = (s + 1 < NB / PW) ? pe + PW : pe;   // s=7 covers [pe, nb2)
    const int FB = 2 * PW;                              // fixed base = min cs (s=0)
    // FIXED strided column ownership (bugfix, rpb>512 race): updater u owns the
    // 4-col groups gi≡u (mod U) of [FB,nb2) for ALL minipanels s, so each trailing
    // column receives reflectors s=0..7 in order from ONE block. The prior scheme
    // (contiguous [cs,nb2) re-partitioned every s) shifted a column's owner between
    // minipanels, letting block B apply reflector s before block A applied s−1 to
    // the same wide [NB,2NB) look-ahead column — a cross-block ordering race whose
    // window widened with panel height. cs (mult of 4) never straddles a 4-col group.
    if (s == 0 && restGate && nb2 > NB) {   // cols ≥ NB need the k−1 rest slice
        if (tid == 0)
            while (atomicAdd(restGate, 0) == 0) __nanosleep(1);
        __syncthreads();
        __threadfence();
    }
    for (int gi = u; ; gi += U) {
        const int g0 = FB + gi * 4;
        if (g0 >= ncTotal) break;          // rest internalized: cover [FB, ncTotal)
        if (g0 < cs) continue;              // group finalized at an earlier minipanel
        const int G = min(4, ncTotal - g0);
        if (tid < 64) dsh[tid] = Real(0);
        __syncthreads();
        Real dloc[4][PW];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            #pragma unroll
            for (int k = 0; k < PW; ++k) dloc[g][k] = Real(0);
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    const Real sv = A[r + size_t(g0 + g) * lda];
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) dloc[g][k] += uk[k] * sv;
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int k = 0; k < PW; ++k) {
                #pragma unroll
                for (int o = 16; o; o >>= 1)
                    dloc[g][k] += __shfl_xor_sync(0xffffffffu, dloc[g][k], o);
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            Real out = Real(0);
            #pragma unroll
            for (int k = 0; k < PW; ++k) out = (lane == k) ? dloc[g][k] : out;
            if (lane < PW && g < G) atomicAdd(&dsh[g * PW + lane], out);
        }
        __syncthreads();
        if (tid < 64) {
            const int g = tid / PW, k = tid % PW;
            Real acc = Real(0);
            for (int k2 = 0; k2 <= k; ++k2) acc += Ts[k2 + k * PW] * dsh[g * PW + k2];
            wsh[g * PW + k] = acc;
        }
        __syncthreads();
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    Real acc = Real(0);
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) acc += uk[k] * wsh[g * PW + k];
                    A[r + size_t(g0 + g) * lda] -= acc;
                }
            }
        }
        __syncthreads();
        {
            const int lo = max(g0, pe + PW), hi = min(g0 + G, pe + 2 * PW);
            if (hi > lo) {
                __threadfence();
                if (tid == 0) atomicAdd(&pfk[8 + s], hi - lo);
            }
        }
        if (s + 1 == NB / PW) {
            const int lo = max(g0, NB), hi = min(g0 + G, NB + PW);
            if (hi > lo) {
                __threadfence();
                if (tid == 0) atomicAdd(&pfk[17], hi - lo);
            }
        }
        __syncthreads();
    }
    Real* PsumS = Psum;
    const int chunkP = p > 0 ? (p + U - 1) / U : 0;
    const int i0 = u * chunkP, i1 = min(p, i0 + chunkP);
    for (int g0 = i0; g0 < i1; g0 += 4) {
        const int G = min(4, i1 - g0);
        if (tid < 64) dsh[tid] = Real(0);
        __syncthreads();
        Real dloc[4][PW];
        #pragma unroll
        for (int g = 0; g < 4; ++g)
            #pragma unroll
            for (int k = 0; k < PW; ++k) dloc[g][k] = Real(0);
        for (int rr = tid; rr < h; rr += nth) {
            const int r = p + rr;
            Real uk[PW];
            if (rr < PW) {
                #pragma unroll
                for (int k = 0; k < PW; ++k)
                    uk[k] = (rr < k) ? Real(0)
                           : (rr == k ? fvT[k] : load_fresh(&A[r + size_t(p + k) * lda]));
            } else {
                #pragma unroll
                for (int k = 0; k < PW; ++k) uk[k] = load_fresh(&A[r + size_t(p + k) * lda]);
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                if (g < G) {
                    const Real sv = load_fresh(&A[r + size_t(g0 + g) * lda]);
                    #pragma unroll
                    for (int k = 0; k < PW; ++k) dloc[g][k] += uk[k] * sv;
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int k = 0; k < PW; ++k) {
                #pragma unroll
                for (int o = 16; o; o >>= 1)
                    dloc[g][k] += __shfl_xor_sync(0xffffffffu, dloc[g][k], o);
            }
        }
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            Real out = Real(0);
            #pragma unroll
            for (int k = 0; k < PW; ++k) out = (lane == k) ? dloc[g][k] : out;
            if (lane < PW && g < G) atomicAdd(&dsh[g * PW + lane], out);
        }
        __syncthreads();
        if (tid < PW * G) {
            const int g = tid / PW, k = tid % PW;
            PsumS[size_t(wt + g0 + g) * PW + k] = dsh[g * PW + k];
        }
        __syncthreads();
    }
}

template<typename Real, int NB, int PW = 16>
__global__ __launch_bounds__(512) void k_super_dsm(DomArgs<Real> a, int* pf, int C,
                                                 int npan) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(16) unsigned char smraw[];
    const int tid = threadIdx.x, nth = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, NW = nth >> 5;
    const int bid = blockIdx.x, nbl = gridDim.x;
    const size_t lda = a.lda;
    const int rpb0 = a.rpb;
    Real* msm = reinterpret_cast<Real*>(smraw);
    DsmPub<Real>* pub = reinterpret_cast<DsmPub<Real>*>(msm + size_t(rpb0) * PW);
    Real* fvL  = reinterpret_cast<Real*>(pub + 1);
    Real* betL = fvL + NB;
    Real* ttL  = betL + NB;
    Real* ssm  = ttL + NB;
    Real* m1s  = ssm + 64;                     // rank 0 T-assembly scratch (1792)

    if (bid >= C) {   // ------------------------------------------------ updaters
        const int u = bid - C, U = nbl - C;
        Real* Ts  = msm;
        Real* dsh = msm + 256;
        Real* wsh = msm + 320;
        Real* fvT = msm + 384;
        for (int k = 0; k < npan; ++k) {
            int* pfk = pf + k * 20;
            Real* Ak = a.A + size_t(k) * NB * lda + size_t(k) * NB;
            const int mkk = a.mk - k * NB;
            const int nb2 = (k + 1 < npan) ? 2 * NB : NB;
            const Real* T16gk = a.T16g + size_t(k & 1) * 256 * (NB / PW);
            Real* Psumk = a.Psum + size_t(k & 1) * size_t(NB) * PW * (NB / PW);
            if (k >= 2) {   // rest-slice(k−2) must be applied before panel-k reads
                if (tid == 0)
                    while (atomicAdd(&pf[320 + (k - 2)], 0) == 0) __nanosleep(1);
                __syncthreads();
                __threadfence();
            }
            int* restGate = (k >= 1) ? pf + 320 + (k - 1) : nullptr;
            const int ncTotal = (npan - k) * NB;   // rest internalized: full trailing extent
            for (int s = 0; s < NB / PW; ++s)
                d_super_upd<Real, NB, PW>(Ak, lda, Ts, fvT, dsh, wsh,
                                          T16gk + size_t(s) * 256, a.fv,
                                          Psumk + size_t(s) * NB * PW, pfk, restGate,
                                          mkk, s, u, U, nb2, ncTotal, tid, nth, lane);
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicAdd(&pfk[16], 1);
                while (atomicAdd(&pfk[16], 0) < U) __nanosleep(1);
            }
            __syncthreads();
            // restDone[k] set IN-KERNEL: the internalized rest columns are now
            // applied by the updaters above (d_super_upd covers [FB,ncTotal)),
            // so the host rest slice is gone. Set after the donePan barrier so
            // all updater blocks have fenced their rest writes; panels k+2 (col
            // cluster + updaters) spin on this and proceed without host help.
            if (tid == 0 && u == 0) { __threadfence(); atomicExch(&pf[320 + k], 1); }
            __syncthreads();
            d_pipe_writeout<Real, NB>(Ak, lda,
                a.SPV + size_t(a.spvcol + k * NB) * a.ldspv + a.row0 + k * NB,
                a.ldspv, fvL, betL, a.fv, a.bet, mkk, u, U, tid, nth);
            __syncthreads();
            __threadfence();
            if (tid == 0) atomicAdd(&pfk[19], 1);
        }
        return;
    }
    // ---------------------------------------------------------------- col cluster
    cg::cluster_group cl = cg::this_cluster();
    const int rank = cl.block_rank();
    auto pubAt = [&](int rk) -> DsmPub<Real>* {
        return reinterpret_cast<DsmPub<Real>*>(cl.map_shared_rank((void*)pub, rk));
    };
    for (int k = 0; k < npan; ++k) {
        int* pfk = pf + k * 20;
        Real* Ak = a.A + size_t(k) * NB * lda + size_t(k) * NB;
        const int mkk = a.mk - k * NB;
        const int rpb = 32 * (((mkk + C - 1) / C + 31) / 32);
        const int r0 = rank * rpb, r1 = min(mkk, r0 + rpb);
        Real* T16gk = a.T16g + size_t(k & 1) * 256 * (NB / PW);
        Real* Psumk = a.Psum + size_t(k & 1) * size_t(NB) * PW * (NB / PW);
        if (k > 0) {   // ALL of panel k−1's updater work must be applied (the
            // boundary-0 self-stage reads cols [16,32) which only donePan covers)
            if (tid == 0)
                while (atomicAdd(&(pf + (k - 1) * 20)[16], 0) < nbl - C)
                    __nanosleep(1);
            __syncthreads();
            __threadfence();
        }
        if (k >= 2) {  // and the rest-slice(k−2) for columns beyond the next panel
            if (tid == 0)
                while (atomicAdd(&pf[320 + (k - 2)], 0) == 0) __nanosleep(1);
            __syncthreads();
            __threadfence();
        }
        if (tid < PW) { pub->dslP[0][tid] = Real(0); if (tid == 0) pub->sgP[0] = Real(0); }
        if (tid >= PW && tid < PW + 17) ssm[PW + 2 + (tid - PW)] = Real(0);
        __syncthreads();
        for (int e = tid; e < (r1 - r0) * PW; e += nth) {
            int q = e / (r1 - r0), rr = e % (r1 - r0);
            msm[rr + q * rpb] = Ak[(r0 + rr) + size_t(q) * lda];
        }
        __syncthreads();
        {
            Real dp[PW];
            #pragma unroll
            for (int q = 0; q < PW; ++q) dp[q] = Real(0);
            for (int rr = max(r0, 1) - r0 + tid; rr < r1 - r0; rr += nth) {
                const Real v0 = msm[rr];
                dp[0] += v0 * v0;
                #pragma unroll
                for (int q = 1; q < PW; ++q) dp[q] += v0 * msm[rr + q * rpb];
            }
            #pragma unroll
            for (int q = 0; q < PW; ++q) {
                #pragma unroll
                for (int o = 16; o; o >>= 1)
                    dp[q] += __shfl_xor_sync(0xffffffffu, dp[q], o);
            }
            Real out = Real(0);
            #pragma unroll
            for (int q = 0; q < PW; ++q) out = (lane == q) ? dp[q] : out;
            if (lane < PW) {
                if (lane == 0) atomicAdd(&pub->sgP[0], out);
                else           atomicAdd(&pub->dslP[0][lane], out);
            }
            if (r0 == 0 && tid < PW) pub->prow[0][tid] = msm[size_t(tid) * rpb];
        }
        // rank 0: previous panel's T-assembly (parity slabs; fvL still holds k−1's
        // f values — this panel's regions haven't overwritten them yet)
        if (k > 0 && rank == 0) {
            int* pfp = pf + (k - 1) * 20;
            d_pipe_tassemble<Real, NB, PW>(
                a.Tpan + size_t(k - 1) * NB * NB,
                a.T16g + size_t((k - 1) & 1) * 256 * (NB / PW),
                a.Psum + size_t((k - 1) & 1) * size_t(NB) * PW * (NB / PW),
                m1s, fvL, pfp, (nbl - C) + 1, tid, nth);
            __threadfence();
            if (tid == 0) atomicExch(&pfp[18], 1);
            __syncthreads();
        }
        cl.sync();
        DomArgs<Real> ak = a;
        ak.A = Ak; ak.mk = mkk; ak.rpb = rpb;
        ak.T16g = T16gk; ak.Psum = Psumk;
        for (int s = 0; s < NB / PW; ++s) {
            const int p = s * PW;
            for (int jj = 0; jj < PW; ++jj) {
                d_dsm_region<Real, NB, PW>(msm, pub, pubAt, ssm, fvL, betL, ttL,
                                           rpb, r0, r1, C, p, jj,
                                           tid, nth, lane, warp, NW);
                cl.sync();
            }
            d_dsm_boundary<Real, NB, PW>(ak, msm, pub, pubAt, fvL, betL, ttL,
                                         rpb, r0, r1, C, rank, s, pfk, cl,
                                         tid, nth, lane, warp, NW);
        }
    }
    if (rank == 0) {
        int* pfp = pf + (npan - 1) * 20;
        d_pipe_tassemble<Real, NB, PW>(
            a.Tpan + size_t(npan - 1) * NB * NB,
            a.T16g + size_t((npan - 1) & 1) * 256 * (NB / PW),
            a.Psum + size_t((npan - 1) & 1) * size_t(NB) * PW * (NB / PW),
            m1s, fvL, pfp, (nbl - C) + 1, tid, nth);
        __threadfence();
        if (tid == 0) atomicExch(&pfp[18], 1);
    }
}

__global__ inline void k_set_flag(int* f) { atomicExch(f, 1); }
__global__ inline void k_wait_flag(volatile int* f) {
    while (atomicAdd((int*)f, 0) == 0) __nanosleep(1);
}

// Tp = −U·diag(S) (upper incl diag), zeros below — pre-trsm RHS for T_panel = −U·S·Y1^{-T}.
template<typename Real, int NB>
__global__ void k_negUS(const Real* QT, int ldq, const Real* Sg, Real* Tp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NB * NB) return;
    int c = idx / NB, r = idx % NB;
    Tp[idx] = (r <= c) ? -QT[size_t(c) * ldq + r] * Sg[c] : Real(0);
}

// Assemble: SPV column block ← explicit-unit Y; A panel ← V below diag, R = S·R_tsqr above.
template<typename Real, int NB>
__global__ void k_writeYRV(const Real* QT, int ldq, const Real* P, int ldp, const Real* Sg,
                           Real* A, size_t lda, int row0, int mk,
                           Real* SPV, size_t ldspv, int spvcol) {
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(mk) * NB) return;
    const int c = int(gid / mk), r = int(gid % mk);
    Real y;
    if (r < NB) y = (r > c) ? QT[size_t(c) * ldq + r] : (r == c ? Real(1) : Real(0));
    else        y = QT[size_t(c) * ldq + r];
    // SPV is the masked-Y side output. TQR-Omega's compact path does not have one --
    // 31:415 puts V in tril(A), so a second m x b copy of the same vectors is exactly
    // the O(mb) staging that made this carrier fail its own O(nb) ledger. Guard it
    // rather than demand a buffer nobody reads.
    if (SPV) SPV[size_t(spvcol + c) * ldspv + row0 + r] = y;
    Real* Ac = A + size_t(c) * lda + row0;
    if (r > c) Ac[r] = y;
    else       Ac[r] = Sg[r] * P[size_t(c) * ldp + r];
}

template<typename Real>
__global__ void k_load_panel(const Real* A, size_t lda, int mk, int nb, Real* P, int ldp, int mpad) {
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(mpad) * nb) return;
    const int c = int(gid / mpad), r = int(gid % mpad);
    P[size_t(c) * ldp + r] = (r < mk) ? A[size_t(c) * lda + r] : Real(0);
}

template<typename Real, int NB>
__global__ void k_set_identity(Real* QS) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NB * NB) return;
    QS[idx] = (idx % NB == idx / NB) ? Real(1) : Real(0);
}

template<typename Real>
__global__ void k_eye(Real* Q, size_t ldq, int m, int n) {
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(m) * n) return;
    const int c = int(gid / m), r = int(gid % m);
    Q[size_t(c) * ldq + r] = (r == c) ? Real(1) : Real(0);
}

// W6-P3 single-copy validation: seed an (m×b) column-major buffer X with the
// columns [i0:i1] of the m×m identity (X[r,c] = 1 iff r == i0+c, else 0). This is
// the seed for Plan::orgqr_rows, which applies Q^T to X to form Q[i0:i1,:]^T
// (rows [i0:i1] of thin Q, transposed). See SINGLECOPY_VALIDATION_DESIGN.md §2.1.
template<typename Real>
__global__ void k_eye_rows(Real* X, size_t ldx, int m, int i0, int b) {
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(m) * b) return;
    const int c = int(gid / m), r = int(gid % m);
    X[size_t(c) * ldx + r] = (r == i0 + c) ? Real(1) : Real(0);
}

template<typename Real>
__global__ void k_extractY(const Real* A, size_t lda, int row0, int mk, int nb, Real* Y, int ldy) {
    const size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(mk) * nb) return;
    const int c = int(gid / mk), r = int(gid % mk);
    const Real* Ac = A + size_t(c) * lda + row0;
    Y[size_t(c) * ldy + r] = (r > c) ? Ac[r] : (r == c ? Real(1) : Real(0));
}

// ---------------------------------------------------------------- GEMM wrappers (math tiers)
inline void gemm(cublasHandle_t h, bool tf32, cublasOperation_t opA, cublasOperation_t opB,
                 int m, int n, int k, float alpha, const float* A, int lda,
                 const float* B, int ldb, float beta, float* C, int ldc) {
    CUBLAS_CHECK(cublasGemmEx(h, opA, opB, m, n, k, &alpha, A, CUDA_R_32F, lda,
                              B, CUDA_R_32F, ldb, &beta, C, CUDA_R_32F, ldc,
                              tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

// cuBLASLt GEMM with SM_COUNT_TARGET — limits GEMM to ltSmTarget SMs so the
// cooperative panel kernel gets the rest without SM contention.
template<typename Real>
inline void gemmLt(cublasLtHandle_t ltH, cublasLtMatmulDesc_t mmDesc,
                   cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, Real alpha,
                   const Real* A, int lda, const Real* B, int ldb,
                   Real beta, Real* C, int ldc,
                   void* workspace, size_t wsSize, cudaStream_t stream) {
    constexpr cudaDataType_t dtype = std::is_same_v<Real, double> ? CUDA_R_64F : CUDA_R_32F;
    int aRows = (opA == CUBLAS_OP_T) ? k : m;
    int aCols = (opA == CUBLAS_OP_T) ? m : k;
    int bRows = (opB == CUBLAS_OP_T) ? n : k;
    int bCols = (opB == CUBLAS_OP_T) ? k : n;

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, dtype, aRows, aCols, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, dtype, bRows, bCols, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, dtype, m, n, ldc);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &wsSize, sizeof(wsSize));
    cublasLtMatmulHeuristicResult_t result;
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltH, mmDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                   pref, 1, &result, &nResults);
    if (nResults == 0) {
        cublasLtMatmul(ltH, mmDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                       C, Cdesc, C, Cdesc, nullptr, workspace, wsSize, stream);
    } else {
        cublasLtMatmul(ltH, mmDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                       C, Cdesc, C, Cdesc, &result.algo, workspace, result.workspaceSize, stream);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}
inline void gemm(cublasHandle_t h, bool, cublasOperation_t opA, cublasOperation_t opB,
                 int m, int n, int k, double alpha, const double* A, int lda,
                 const double* B, int ldb, double beta, double* C, int ldc) {
    CUBLAS_CHECK(cublasDgemm(h, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
}
inline void trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t up,
                 cublasOperation_t op, cublasDiagType_t dg, int m, int n, float alpha,
                 const float* A, int lda, float* B, int ldb) {
    CUBLAS_CHECK(cublasStrsm(h, side, up, op, dg, m, n, &alpha, A, lda, B, ldb));
}
inline void trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t up,
                 cublasOperation_t op, cublasDiagType_t dg, int m, int n, double alpha,
                 const double* A, int lda, double* B, int ldb) {
    CUBLAS_CHECK(cublasDtrsm(h, side, up, op, dg, m, n, &alpha, A, lda, B, ldb));
}
// ---- scale-safety helpers for geqrf's wrapper (see the comment there) --------
// Block-strided max|a_ij| in fp64 regardless of Real, so the decision to rescale a
// tf32-tier matrix is not itself made in low precision.
template<typename Real>
__global__ void k_absmax_blk(const Real* A, size_t lda, int m, int n, double* part) {
    __shared__ double sh[256];
    double mx = 0.0;
    const size_t total = (size_t)m * n;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(t / m), i = (int)(t % m);
        const double v = fabs((double)A[(size_t)j * lda + i]);
        if (v > mx) mx = v;
    }
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int o = blockDim.x >> 1; o; o >>= 1) {
        if (threadIdx.x < o && sh[threadIdx.x + o] > sh[threadIdx.x]) sh[threadIdx.x] = sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sh[0];
}
template<typename Real>
__global__ void k_scale_block(Real* A, size_t lda, int m, int n, Real s) {
    const size_t total = (size_t)m * n;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(t / m), i = (int)(t % m);
        A[(size_t)j * lda + i] *= s;
    }
}
// Scale only the R region: rows [0,k), the upper triangle (i <= j). V (i > j) and
// the T blocks are left alone on purpose -- they carry the Q of the scaled matrix,
// which is the Q of the original.
template<typename Real>
__global__ void k_scale_upper(Real* A, size_t lda, int k, int n, Real s) {
    const size_t total = (size_t)k * n;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(t / k), i = (int)(t % k);
        if (i <= j) A[(size_t)j * lda + i] *= s;
    }
}

// trmm, to apply R from the compact factored form. cuBLAS trmm is out-of-place in
// its C argument, so B is passed as both source and destination -- which cuBLAS
// permits and which keeps the operation storage-free, the whole point of applying R
// out of triu(A) rather than copying it somewhere first.
inline void trmm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t up,
                 cublasOperation_t op, cublasDiagType_t dg, int m, int n, float alpha,
                 const float* A, int lda, float* B, int ldb) {
    CUBLAS_CHECK(cublasStrmm(h, side, up, op, dg, m, n, &alpha, A, lda, B, ldb, B, ldb));
}
inline void trmm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t up,
                 cublasOperation_t op, cublasDiagType_t dg, int m, int n, double alpha,
                 const double* A, int lda, double* B, int ldb) {
    CUBLAS_CHECK(cublasDtrmm(h, side, up, op, dg, m, n, &alpha, A, lda, B, ldb, B, ldb));
}

// ---------------------------------------------------------------- custom (non-cuBLAS) compact-WY apply
// C[mr x nc] -= Y[mr x W] * (Tw^T[W x W] * (Y^T[W x mr] * C[mr x nc])) — the exact math applyQt's 3
// cuBLAS GEMMs compute, realized as a plain __global__ function instead.
//
// WHY: a cuBLAS GEMM's first-ever dispatch of a given (op-pattern, math-mode) combination in a CUDA
// context BLOCKS THE HOST THREAD ITSELF while any cooperative(+cluster) kernel is resident (ROADMAP
// §1's exact original wording: "the host cuBLAS call hangs" — reconfirmed 2026-07-07 for STAGE-2:
// this is not merely GPU-side stream serialization, since even Tsp-compound GEMMs with no data
// dependency back into k_super_dsm's progress also hung the host outright, blocking it from ever
// reaching the code that issues the genuinely-dependent rest-slice work on another stream). NVIDIA's
// docs corroborate part of the mechanism (without cublasSetWorkspace, cuBLAS's lazy default-workspace
// path uses cudaMallocAsync, documented unsafe around cooperative kernels) but an explicit workspace
// alone did not fix it, and only a real — not zero-extent — GEMM dispatch warms the path; the
// remaining, dominant cost is cuBLAS's own lazy kernel/JIT dispatch, which has no public preload API.
// A plain __global__ function has none of this: it is EAGER-loaded at context init exactly like every
// other kernel in this file (k_wait_flag included, verified to launch concurrently with a resident
// STAGE-2 kernel with zero issue) — no lazy dispatch state, no cold start, no warmup call anywhere.
// Used for BOTH the rest-slice apply (stream sR — on k_super_dsm's critical path: restDone gates panel
// k+1) and the Tsp-compound (stream sP/cbT — not on that path, but the host-blocking nature of a cold
// cuBLAS call makes it fatal regardless). The proven-working default/STAGE-1 path's OWN Tsp-compound
// (stream sP, `!superOK` loop) is untouched — its coop kernel drains per-panel (not per-super-panel),
// giving the host-blocking JIT enough gaps to land without contention; only STAGE-2's much
// longer-resident kernel exposes this.
template<typename Real, int W, int BW = 32>
__global__ __launch_bounds__(256) void k_custom_apply(
        Real* C, size_t ldc, const Real* Y, int ldy, const Real* Tw, int ldT,
        int mr, int nc) {
    __shared__ Real W1[W][BW];
    __shared__ Real W2[W][BW];
    const int tid = threadIdx.x, nth = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, NW = nth >> 5;
    const int c0 = blockIdx.x * BW;
    const int ncLoc = min(BW, nc - c0);
    if (ncLoc <= 0) return;

    // Phase 1 (TF32 tensor-core, WMMA): W1[wi][cj] = sum_r Y[r,wi] * C[r,c0+cj]. K=mr is the ONLY
    // large-K contraction in this kernel (up to tens of thousands for large panels), so a tensor
    // core's per-fragment setup amortizes well here — unlike the project's earlier (Julia-based)
    // history (memory/lrqr-goal, 9 independent measurements) which found WMMA does NOT help the
    // K~16-128 "thin/shallow" leaf-apply shapes; K=mr is a genuinely different, much larger regime.
    // Requires mr%8==0 and W%16==0/BW%16==0 (hold for every NB=128-aligned shape this driver uses;
    // not a general-purpose kernel). Y/C are loaded as plain fp32 — nvcuda::wmma's tf32 fragments
    // accept float pointers directly and round to tf32 in hardware, no explicit cast needed.
#if LRQR_APPLY_PHASE1_WMMA
    if constexpr (std::is_same<Real, float>::value) {
        using namespace nvcuda;
        static_assert(W % 16 == 0 && BW % 16 == 0, "WMMA tile constraints");
        constexpr int RT = W / 16, CT = BW / 16;
        for (int t = warp; t < RT * CT; t += NW) {
            const int rt = t % RT, ct = t / RT;
            const int wi0 = rt * 16, cj0 = ct * 16;
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fragA;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fragB;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> fragC;
            wmma::fill_fragment(fragC, 0.0f);
            for (int r0 = 0; r0 < mr; r0 += 8) {
                wmma::load_matrix_sync(fragA, Y + size_t(wi0) * ldy + r0, ldy);
                wmma::load_matrix_sync(fragB, C + r0 + size_t(c0 + cj0) * ldc, ldc);
                wmma::mma_sync(fragC, fragA, fragB, fragC);
            }
            wmma::store_matrix_sync(&W1[wi0][cj0], fragC, BW, wmma::mem_row_major);
        }
    } else
#endif
    {
        for (int pair = warp; pair < W * ncLoc; pair += NW) {
            const int wi = pair % W, cj = pair / W;
            const Real* Yc = Y + size_t(wi) * ldy;
            const Real* Cc = C + size_t(c0 + cj) * ldc;
            Real acc = Real(0);
            for (int r = lane; r < mr; r += 32) acc += Yc[r] * Cc[r];
            #pragma unroll
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) W1[wi][cj] = acc;
        }
    }
    __syncthreads();

    // Phase 2: W2[wi][cj] = sum_k Tw^T[wi,k] * W1[k,cj], Tw^T[wi,k] = Tw[k,wi] — small (W x W), one
    // thread per (wi,cj) output with a plain W-length loop.
    // When Phase 3 WMMA is enabled, negate the result so Phase 3 can use beta=1 (C += Y@W2_neg = C - Y@W2).
    for (int idx = tid; idx < W * ncLoc; idx += nth) {
        const int wi = idx % W, cj = idx / W;
        Real acc = Real(0);
        #pragma unroll 4
        for (int k = 0; k < W; ++k) acc += Tw[k + size_t(wi) * ldT] * W1[k][cj];
#if LRQR_APPLY_PHASE3_WMMA
        W2[wi][cj] = -acc;
#else
        W2[wi][cj] = acc;
#endif
    }
    __syncthreads();

    // Phase 3 (WMMA + scalar add): W2 is already negated, so fragC = Y @ W2_neg = -(Y @ W2_orig).
    // Store fragC to per-warp smem scratch, then scalar C += fragC = C - Y @ W2_orig.
    // Reuses W1's smem (no longer needed after Phase 2) as 8 per-warp 16×16 scratch tiles.
#if LRQR_APPLY_PHASE3_WMMA
    if constexpr (std::is_same<Real, float>::value) {
        using namespace nvcuda;
        static_assert(W % 8 == 0 && BW % 16 == 0, "WMMA Phase 3 tile constraints");
        constexpr int CT = BW / 16;
        const int RT = mr / 16;   // mr is always a multiple of 16 (NB-aligned)
        const int totalTiles = RT * CT;
        // Zero out-of-bounds W2 columns (partial BW blocks)
        if (ncLoc < BW) {
            for (int idx = tid; idx < W * (BW - ncLoc); idx += nth) {
                int wi = idx % W, cj = ncLoc + idx / W;
                W2[wi][cj] = Real(0);
            }
            __syncthreads();
        }
        Real (*tmp)[16] = reinterpret_cast<Real(*)[16]>(&W1[0][0]);
        for (int t = warp; t < totalTiles; t += NW) {
            const int rt = t % RT, ct = t / RT;
            const int r0 = rt * 16, cj0 = ct * 16;
            if (r0 >= mr) continue;
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fragA;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fragB;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> fragC;
            wmma::fill_fragment(fragC, 0.0f);
            for (int wi0 = 0; wi0 < W; wi0 += 8) {
                wmma::load_matrix_sync(fragA, Y + r0 + size_t(wi0) * ldy, ldy);
                wmma::load_matrix_sync(fragB, &W2[wi0][cj0], BW);
                wmma::mma_sync(fragC, fragA, fragB, fragC);
            }
            wmma::store_matrix_sync(&tmp[warp * 16][0], fragC, 16, wmma::mem_row_major);
            // Scalar add: each lane handles 256/32 = 8 elements
            for (int i = lane; i < 256; i += 32) {
                const int r = i % 16, c = i / 16;
                if (cj0 + c < ncLoc)
                    C[(r0 + r) + size_t(c0 + cj0 + c) * ldc] += tmp[warp * 16 + r][c];
            }
        }
    } else
#endif
    {
        // Phase 3 (scalar): C[r,c0+cj] -= sum_wi Y[r,wi] * W2[wi,cj] — one thread per (r,cj) output.
        for (int idx = tid; idx < mr * ncLoc; idx += nth) {
            const int r = idx % mr, cj = idx / mr;
            Real acc = Real(0);
            #pragma unroll 4
            for (int wi = 0; wi < W; ++wi) acc += Y[r + size_t(wi) * ldy] * W2[wi][cj];
            C[r + size_t(c0 + cj) * ldc] -= acc;
        }
    }
}

// Transpose Y[mr×w, lda] into Yt[w×mr, ldt]. Tiled for coalesced access.
// Called once per super-panel before the far update — Y is constant across all far chunks.
template<typename Real, int BM = 64, int BN = 64>
__global__ void k_transpose_y(const Real* __restrict__ Y, int lda,
                               Real* __restrict__ Yt, int ldt, int mr, int w) {
    __shared__ Real tile[BM][BN + 1];
    const int bi = blockIdx.x * BM, bj = blockIdx.y * BN;
    // Each thread handles a 2×2 sub-tile (256 threads = 16×16, each covers 4×4 = 64×64 total)
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    #pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            int ti = bi + tx * 4 + dx, tj = bj + ty * 4 + dy;
            if (ti < mr && tj < w)
                tile[tx * 4 + dx][ty * 4 + dy] = Y[ti + size_t(tj) * lda];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        #pragma unroll
        for (int dx = 0; dx < 4; ++dx) {
            int dj = bj + tx * 4 + dx, di = bi + ty * 4 + dy;
            if (dj < w && di < mr)
                Yt[dj + size_t(di) * ldt] = tile[ty * 4 + dy][tx * 4 + dx];
        }
    }
}

// 3×TF32 Ozaki split (Ootomo-Yokota 2022/2023): hi = round-to-nearest-even to TF32 (10 mantissa
// bits); lo = (x - hi) * 2^11, scaling the exact Sterbenz residual into TF32's normal range to avoid
// underflow / lost precision in the correction GEMMs. The 2^-11 is folded back into those GEMMs'
// alpha. Used to split Y/Yt (once per super-panel) and C/W2 (per far-rest chunk).
template<typename Real>
__global__ void k_split_3xtf32_2d(const Real* __restrict__ in, int ldin,
                                   Real* __restrict__ hi, int ldhi,
                                   Real* __restrict__ lo, int ldlo,
                                   int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    float x = (float)in[size_t(c) * ldin + r];
    uint32_t bits = __float_as_uint(x);
    uint32_t lsb    = (bits >> 13) & 1u;
    uint32_t roundb = (bits >> 12) & 1u;
    uint32_t sticky = (bits & 0xFFFu) != 0u;
    uint32_t hi_bits = (bits & 0xFFFFE000u) + ((roundb & (sticky | lsb)) << 13);
    float h = __uint_as_float(hi_bits);
    hi[size_t(c) * ldhi + r] = (Real)h;
    lo[size_t(c) * ldlo + r] = (Real)((x - h) * 2048.0f);
}

// Sum batched split-K outputs: out[w*nc] = sum_{b=0}^{numBatches-1} batch[b * w * nc + idx].
// Each thread handles one output element by iterating over batches.
template<typename Real>
__global__ void k_sum_w1_batch(const Real* __restrict__ batch, Real* __restrict__ out,
                               int w, int nc, int numBatches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * nc;
    if (idx >= total) return;
    Real sum = Real(0);
    const Real* p = batch + idx;
    for (int b = 0; b < numBatches; b++)
        sum += p[size_t(b) * total];
    out[idx] = sum;
}

// General non-cuBLAS GEMM: C[M,N] = alpha*op(A)[M,K]@op(B)[K,N] + beta*C[M,N]. TA/TB select whether
// A/B are stored transposed (matches cuBLAS's opA/opB=T convention). Same rationale as k_custom_apply
// above — used for STAGE-2's Tsp-compound (the T-matrix assembly, structurally different math from the
// compact-WY apply, hence its own kernel rather than reusing k_custom_apply).
template<typename Real, bool TA, bool TB, int BM = 32, int BN = 32>
__global__ __launch_bounds__(256) void k_custom_gemm(
        int M, int N, int K, Real alpha, const Real* A, int lda,
        const Real* Bm, int ldb, Real beta, Real* C, int ldc) {
    const int tid = threadIdx.x, nth = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, NW = nth >> 5;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int mLoc = min(BM, M - m0), nLoc = min(BN, N - n0);
    if (mLoc <= 0 || nLoc <= 0) return;
#if LRQR_APPLY_PHASE1_WMMA
    // TA && !TB is this kernel's ONLY large-K caller (Tsp-compound's Zb = SPV^T@SPV, K=mr, up to
    // tens of thousands) — same tensor-core rationale as k_custom_apply's Phase 1. The other
    // (TA=false) caller has small fixed K (j0 or NB=128), matching the project's earlier findings
    // that WMMA does NOT help thin/shallow shapes, so it keeps the CUDA-core warp-reduction path.
    if constexpr (TA && !TB && std::is_same<Real, float>::value) {
        using namespace nvcuda;
        static_assert(BM % 16 == 0 && BN % 16 == 0, "WMMA tile constraints");
        constexpr int RT = BM / 16, CT = BN / 16;
        __shared__ Real tileTmp[8][16][16];   // one scratch tile per warp (<=8 warps @ 256 threads)
        for (int t = warp; t < RT * CT; t += NW) {
            const int rt = t % RT, ct = t / RT;
            const int mi0 = m0 + rt * 16, nj0 = n0 + ct * 16;
            if (mi0 >= M || nj0 >= N) continue;
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fragA;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fragB;
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> fragC;
            wmma::fill_fragment(fragC, 0.0f);
            for (int k0 = 0; k0 < K; k0 += 8) {
                wmma::load_matrix_sync(fragA, A + size_t(mi0) * lda + k0, lda);
                wmma::load_matrix_sync(fragB, Bm + k0 + size_t(nj0) * ldb, ldb);
                wmma::mma_sync(fragC, fragA, fragB, fragC);
            }
            wmma::store_matrix_sync(&tileTmp[warp][0][0], fragC, 16, wmma::mem_row_major);
            for (int e = lane; e < 256; e += 32) {
                const int ii = e >> 4, jj = e & 15;
                Real* c = &C[(mi0 + ii) + size_t(nj0 + jj) * ldc];
                const Real v = tileTmp[warp][ii][jj];
                *c = (beta == Real(0)) ? alpha * v : alpha * v + beta * (*c);
            }
        }
        return;
    }
#endif
    for (int pair = warp; pair < mLoc * nLoc; pair += NW) {
        const int mi = pair % mLoc, nj = pair / mLoc;
        Real acc = Real(0);
        for (int k = lane; k < K; k += 32) {
            Real a = TA ? A[k + size_t(m0 + mi) * lda] : A[(m0 + mi) + size_t(k) * lda];
            Real b = TB ? Bm[(n0 + nj) + size_t(k) * ldb] : Bm[k + size_t(n0 + nj) * ldb];
            acc += a * b;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            // BLAS convention: beta==0 means C is logically uninitialized and must NOT be read —
            // *c can genuinely be stale NaN/Inf bits from a previous Plan's freed device memory
            // (reused by the allocator), and IEEE 754 gives 0*NaN=NaN, 0*Inf=NaN, silently
            // poisoning an intended plain overwrite. cuBLAS already special-cases this; matching
            // it here (bug found 2026-07-07 via a 2-size-in-one-process repro: n=4096 then
            // n=16384 deterministically produced NaN at 16384, n=16384 alone never did).
            Real* c = &C[(m0 + mi) + size_t(n0 + nj) * ldc];
            *c = (beta == Real(0)) ? alpha * acc : alpha * acc + beta * (*c);
        }
    }
}

// ---------------------------------------------------- resident warp-spec QR
// Multi-block cooperative kernel (spec §9/§11): block 0 factorizes panels
// (consumer, FMA pipe), blocks 1..N update the trailing matrix (producer,
// wgmma tensor cores). V/T in gmem (not smem) → supports arbitrary m.
// Double-buffered: block 0 factorizes panel k+1 while blocks 1..N update k.
// Synchronization via gmem generation counters (no cuBLAS → no deadlock).
// Normalized Householder vectors (LAPACK convention: v[0]=1 implicit, tau=beta*v0²).
template<typename Real, int NB, int TILE_N, int CHUNK_M>
__global__ void k_tqr_resident(
    Real* A, int m, int n, int lda, int n_panels, Real* Tpan,
    Real* Vg,                      // [2 * m * NB] double-buffered V (col-major, ld=m)
    Real* Tg,                      // [2 * NB * NB] double-buffered T (col-major, ld=NB)
    volatile unsigned* panel_gen,  // [2] generation: block 0 writes k+1 when panel k ready
    volatile unsigned* update_cnt, // [2] count: blocks 1..N atomicAdd when tile done
    int n_update_blocks)           // gridDim.x - 1
{
    extern __shared__ __align__(16) unsigned char smem_raw[];
    // Smem layout (shared between consumer and producer code paths):
    Real* T_sm  = reinterpret_cast<Real*>(smem_raw);                  // NB * NB
    Real* Rd    = T_sm + NB * NB;                                     // NB
    Real* W     = Rd + NB;                                            // NB * TILE_N
    Real* W2    = W + NB * TILE_N;                                   // NB * TILE_N
    Real* V_ch  = W2 + NB * TILE_N;                                  // CHUNK_M * NB
    Real* C_ch  = V_ch + CHUNK_M * NB;                               // CHUNK_M * TILE_N
    volatile Real* red = C_ch + CHUNK_M * TILE_N;                   // 8 (reduction scratch)

    const unsigned tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nth = blockDim.x;   // 256
    const int nup = n_update_blocks;

    // ---- reduction across 8 warps (256 threads) ----
    auto reduce8 = [&](Real val) -> Real {
        for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xFFFFFFFF, val, o);
        unsigned warp = tid / 32, lane = tid % 32;
        if (lane == 0) red[warp] = val;
        __syncthreads();
        if (warp == 0) {
            Real sum = (lane < 8) ? red[lane] : Real(0);
            for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFF, sum, o);
            if (lane == 0) red[0] = sum;
        }
        __syncthreads();
        return red[0];
    };

    for (int k = 0; k < n_panels; k++) {
        int buf = k & 1;
        int row0 = k * NB, col0 = k * NB;
        int mk = m - row0;
        int Bcur = (n - col0 < NB) ? (n - col0) : NB;
        int ts = (k + 1) * NB;
        int tn = n - ts;

        Real* Vb = Vg + (size_t)buf * m * NB;      // V for this buffer
        Real* Tb = Tg + (size_t)buf * NB * NB;     // T for this buffer

        if (bid == 0) {
            // ============ CONSUMER: Householder panel factorization ============
            // Wait for previous update on this buf (double-buffering for Vg/Tg
            // reuse: panel k writes Vg/Tg[buf], panel k-2's producer still reads
            // it).
            if (k >= 2) {
                if (tid == 0) {
                    unsigned expected = (unsigned)(k / 2) * (unsigned)nup;
                    while (update_cnt[buf] < expected) __nanosleep(8);
                }
                __syncthreads();
            }
            // Wait for panel k-1's trailing update to finish writing to A.
            // The consumer for panel k reads A[row0:m, col0:col0+Bcur], which is
            // the trailing matrix updated by the producer of panel k-1 (which
            // uses buffer (k-1)&1, a different buffer).  Without this wait the
            // consumer reads stale pre-update data → incorrect factorization.
            if (k >= 1) {
                int buf_prev = (k - 1) & 1;
                if (tid == 0) {
                    unsigned expected_prev =
                        (unsigned)((k + 1) / 2) * (unsigned)nup;
                    while (update_cnt[buf_prev] < expected_prev)
                        __nanosleep(8);
                }
                __syncthreads();
                __threadfence();
            }

            // Load panel A[row0:m, col0:col0+Bcur] → Vg[buf]
            // Vg layout: Vb[j * m + i] = V[i][j], column-major, ld=m
            for (int j = 0; j < Bcur; j++) {
                Real* Vcol = Vb + (size_t)j * m;
                const Real* Acol = A + (size_t)(col0 + j) * lda + row0;
                for (int i = (int)tid; i < mk; i += nth)
                    Vcol[i] = Acol[i];
            }
            for (int idx = (int)tid; idx < NB * NB; idx += nth)
                T_sm[idx] = Real(0);
            __syncthreads();

            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;   // V[:,j]

                // sigma = ||V[j+1:mk, j]||²
                Real sigma = 0;
                for (int i = j + (int)tid; i < mk; i += nth) {
                    Real x = Vj[i];
                    sigma = fma(x, x, sigma);
                }
                sigma = reduce8(sigma);

                if (tid == 0) {
                    Real x0 = Vj[j];
                    Real nm = sqrt(sigma);
                    Real alpha = (x0 >= 0) ? -nm : nm;
                    Real v0 = x0 - alpha;
                    Real beta = Real(2) / (fma(v0, v0, sigma - x0 * x0));
                    Real tau = beta * v0 * v0;
                    red[0] = alpha; red[1] = v0; red[2] = tau;
                }
                __syncthreads();
                Real alpha = red[0], v0 = red[1], tau = red[2];

                // Normalize: V[i,j] /= v0 for i > j
                for (int i = j + 1 + (int)tid; i < mk; i += nth)
                    Vj[i] /= v0;
                if (tid == 0) { Rd[j] = alpha; T_sm[j + j * NB] = tau; }
                __syncthreads();

                // Update trailing columns within panel
                for (int kk = j + 1; kk < Bcur; kk++) {
                    Real* Vkk = Vb + (size_t)kk * m;
                    Real dot = 0;
                    for (int i = j + (int)tid; i < mk; i += nth) {
                        Real vj = (i == j) ? Real(1) : Vj[i];
                        dot = fma(vj, Vkk[i], dot);
                    }
                    dot = reduce8(dot);
                    Real w = tau * dot;
                    for (int i = j + (int)tid; i < mk; i += nth) {
                        Real vj = (i == j) ? Real(1) : Vj[i];
                        Vkk[i] = fma(-w, vj, Vkk[i]);
                    }
                    __syncthreads();
                }

                // T recurrence (compact WY, normalized vectors)
                if (j > 0) {
                    if ((int)tid < j) {
                        int c = (int)tid;
                        Real* Vc = Vb + (size_t)c * m;
                        Real z = 0;
                        for (int row = j; row < mk; row++) {
                            Real vk = (row == c) ? Real(1) : Vc[row];
                            Real vj = (row == j) ? Real(1) : Vj[row];
                            z = fma(vk, vj, z);
                        }
                        T_sm[c + j * NB] = z;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        for (int i = 0; i < j; i++) {
                            Real w = 0;
                            for (int kk = i; kk < j; kk++)
                                w += T_sm[i + kk * NB] * T_sm[kk + j * NB];
                            T_sm[i + j * NB] = -tau * w;
                        }
                    }
                    __syncthreads();
                }
            }

            // Store panel to HBM: A[i,j] = (i==j) ? Rd[j] : V[i,j]
            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;
                Real* Acol = A + (size_t)(col0 + j) * lda + row0;
                for (int i = (int)tid; i < mk; i += nth)
                    Acol[i] = (i == j) ? Rd[j] : Vj[i];
            }
            // Store Tpan and Tg
            for (int idx = (int)tid; idx < Bcur * Bcur; idx += nth) {
                int j = idx / Bcur, i = idx % Bcur;
                Real t = T_sm[i + j * NB];
                Tpan[(size_t)k * NB * NB + i + (size_t)j * NB] = t;
                Tb[i + (size_t)j * NB] = t;
            }
            // Clean Vg for update blocks: zero upper triangle, set diagonal to 1
            // (so update blocks can load V without branching)
            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;
                for (int i = (int)tid; i < j; i += nth)
                    Vj[i] = Real(0);
                if (tid == 0) Vj[j] = Real(1);
            }
            __syncthreads();
            __threadfence();
            if (tid == 0) panel_gen[buf] = (unsigned)(k + 1);

        } else {
            // ============ PRODUCER: trailing update ============
            // Wait for panel k to be ready
            if (tid == 0) {
                while (panel_gen[buf] < (unsigned)(k + 1)) __nanosleep(8);
            }
            __syncthreads();
            __threadfence();

            if (tn > 0) {
                // Load T from Tg[buf] into smem
                for (int idx = (int)tid; idx < NB * NB; idx += nth)
                    T_sm[idx] = Tb[idx];
                __syncthreads();

                // Column strip for this block (ceil division to avoid losing columns)
                int cols_per_block = (tn + nup - 1) / nup;
                int my_start = ts + (bid - 1) * cols_per_block;
                int my_end   = ts + bid * cols_per_block;
                if (my_end > ts + tn) my_end = ts + tn;

                for (int col = my_start; col < my_end; col += TILE_N) {
                    int tnn = (my_end - col < TILE_N) ? (my_end - col) : TILE_N;

                    // ---- GEMM1: W = V^T * C (NB × tnn, K=mk) via FFMA ----
                    {
                        for (int idx = (int)tid; idx < NB * TILE_N; idx += nth)
                            W[idx] = Real(0);
                        __syncthreads();
                        for (int i0 = 0; i0 < mk; i0 += CHUNK_M) {
                            int iend = (i0 + CHUNK_M < mk) ? (i0 + CHUNK_M) : mk;
                            int ch = iend - i0;
                            for (int j = 0; j < NB; j++) {
                                const Real* Vcol = Vb + (size_t)j * m + i0;
                                for (int i = (int)tid; i < CHUNK_M; i += nth)
                                    V_ch[i + j * CHUNK_M] = (i < ch) ? Vcol[i] : Real(0);
                            }
                            for (int nn = 0; nn < tnn; nn++) {
                                const Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                                for (int i = (int)tid; i < CHUNK_M; i += nth)
                                    C_ch[i + nn * CHUNK_M] = (i < ch) ? Acol[i] : Real(0);
                            }
                            __syncthreads();
                            for (int idx = (int)tid; idx < NB * tnn; idx += nth) {
                                int nn = idx / NB, b = idx % NB;
                                int istart = (b > i0) ? (b - i0) : 0;
                                Real a = 0;
                                for (int i = istart; i < ch; i++)
                                    a = fma(V_ch[i + b * CHUNK_M], C_ch[i + nn * CHUNK_M], a);
                                W[b + nn * NB] += a;
                            }
                            __syncthreads();
                        }
                    }
                    __syncthreads();

                    // ---- GEMM2: W2 = T * W (NB × tnn), T upper-triangular (FFMA) ----
                    for (int idx = (int)tid; idx < NB * TILE_N; idx += nth) {
                        int nn = idx / NB, b = idx % NB;
                        if (nn >= tnn) { W2[idx] = Real(0); continue; }
                        Real a = 0;
                        for (int kk = 0; kk <= b; kk++)
                            a = fma(T_sm[kk + b * NB], W[kk + nn * NB], a);
                        W2[b + nn * NB] = a;
                    }
                    // Zero padding columns of W2
                    for (int nn = tnn; nn < TILE_N; nn++)
                        for (int i = (int)tid; i < NB; i += nth)
                            W2[i + nn * NB] = Real(0);
                    __syncthreads();

                    // ---- GEMM3: C -= V * W2 (mk × tnn, K=NB) via wmma TF32 ----
                    for (int i0 = 0; i0 < mk; i0 += CHUNK_M) {
                        int iend = (i0 + CHUNK_M < mk) ? (i0 + CHUNK_M) : mk;
                        int ch = iend - i0;
                        // Load V_ch (Vg already clean)
                        for (int j = 0; j < NB; j++) {
                            const Real* Vcol = Vb + (size_t)j * m + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                V_ch[i + j * CHUNK_M] = (i < ch) ? Vcol[i] : Real(0);
                        }
                        // Load C_ch
                        for (int nn = 0; nn < tnn; nn++) {
                            const Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                C_ch[i + nn * CHUNK_M] = (i < ch) ? Acol[i] : Real(0);
                        }
                        for (int nn = tnn; nn < TILE_N; nn++)
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                C_ch[i + nn * CHUNK_M] = Real(0);
                        __syncthreads();

                        // GEMM3: C -= V * W2 (mk × tnn, K=NB) via FFMA
                        for (int idx = (int)tid; idx < CHUNK_M * TILE_N; idx += nth)
                            W[idx] = Real(0);
                        __syncthreads();
                        for (int j = 0; j < NB; j++) {
                            for (int i = (int)tid; i < CHUNK_M; i += nth) {
                                Real vj = V_ch[i + j * CHUNK_M];
                                for (int nn = 0; nn < TILE_N; nn++)
                                    W[i + nn * CHUNK_M] = fma(vj, W2[j + nn * NB], W[i + nn * CHUNK_M]);
                            }
                            __syncthreads();
                        }
                        // C_ch -= W
                        for (int idx = (int)tid; idx < CHUNK_M * TILE_N; idx += nth)
                            C_ch[idx] -= W[idx];
                        __syncthreads();

                        // Store C_ch back to A (only valid rows/cols)
                        for (int nn = 0; nn < tnn; nn++) {
                            Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                            for (int i = (int)tid; i < ch; i += nth)
                                Acol[i] = C_ch[i + nn * CHUNK_M];
                        }
                        __syncthreads();
                    }
                }
            }

            __threadfence();
            if (tid == 0) atomicAdd((unsigned int*)&update_cnt[buf], 1);
        }
    }
}

template<typename Real, int NB> struct ResidentSmem {
    static constexpr int TILE_N = 16;
    static constexpr int CHUNK_M = 128;
    static constexpr size_t T_SZ  = NB * NB;
    static constexpr size_t RD_SZ = NB;
    static constexpr size_t W_SZ  = NB * TILE_N;
    static constexpr size_t W2_SZ = NB * TILE_N;
    static constexpr size_t VCH_SZ = CHUNK_M * NB;
    static constexpr size_t CCH_SZ = CHUNK_M * TILE_N;
    static constexpr size_t RED_SZ = 8;
    static constexpr size_t total_floats = T_SZ + RD_SZ + W_SZ + W2_SZ + VCH_SZ + CCH_SZ + RED_SZ;
    static constexpr size_t bytes = ((total_floats * sizeof(Real) + 15) & ~size_t(15)) + 32;
};

// ============================================================ T5 megakernel
// k_tqr_mega: warp-specialized resident QR with cluster-scope mbarrier dataflow.
// Replaces k_tqr_resident's gmem-counter + __nanosleep (which deadlocks at
// n≥16384, lrqr.cuh:4067) with PTX §9.7.14.16 mbarrier (fan-out via per-consumer
// LOCAL mbarriers + producer remote-arrive; §9.7.14.16.11). Cluster of 8 CTAs:
//   rank 0 = panel (Householder, fp32 CUDA cores — the L-carrier, NO TF32)
//   rank 1..7 = trailing update (the B-carrier; scalar FFMA initially, wgmma
//     TF32 upgrade planned). Double-buffered V/T in gmem (rVg/rTg).
//
// The mbarrier pair (panel_ready, update_done) replaces panel_gen/update_cnt.
// Phase tracking: double-buffer buf=k&1, phase=k/2, parity=phase&1. The
// producer arrives (flips parity); consumers try_wait for the completed phase.
//
// Spec refs: §5 (tower schedule), §7 (on-chip chain), §10 (Pipeline=domino),
// §6 T5 (REQUIRED megakernel deliverable). Route B (mbarrier megakernel +
// standalone CUTLASS update engine) — see RESULT.md.
template<typename Real, int NB>
struct MegaSmem {
    static constexpr int TILE_N = 16;
    static constexpr int CHUNK_M = 128;
    static constexpr size_t MBAR_SZ = 4;  // 4 × uint64_t: panel_ready[2] + update_done[2]
    static constexpr size_t T_SZ  = NB * NB;
    static constexpr size_t RD_SZ = NB;
    static constexpr size_t W_SZ  = NB * TILE_N;
    static constexpr size_t W2_SZ = NB * TILE_N;
    static constexpr size_t VCH_SZ = CHUNK_M * NB;
    static constexpr size_t CCH_SZ = CHUNK_M * TILE_N;
    static constexpr size_t RED_SZ = 8;
    static constexpr size_t total_floats = T_SZ + RD_SZ + W_SZ + W2_SZ + VCH_SZ + CCH_SZ + RED_SZ;
    static constexpr size_t bytes = MBAR_SZ * 8 + ((total_floats * sizeof(Real) + 15) & ~size_t(15)) + 32;
};

// Smem for k_tqr_mega_wgmma: wgmma TF32 consumer variant (W3' carrier sub-unit a).
// Layout: 8 mbarriers + T_sm(NB²) + Rd(NB) + W(NB·TILE_N) + W2(NB·TILE_N)
//         + 128B align + smem_A(max(STAGES·KT·NB, M_TILE·TILE_N)) + smem_B(STAGES·KT·TILE_N) + red(8).
// B32 swizzle (SBO=256) proven viable (w3-swizzle-prove daceda7). float-only (TF32 wgmma).
// smem ≤196 KiB hard (carveout cliff starves TMA/wgmma above) — asserted in create().
template<typename Real, int NB, int TILE_N, int KT = 8, int STAGES = 2>
struct MegaSmemWgmma {
    static constexpr int M_TILE = 64;
    static constexpr size_t MBAR_SZ = 8;  // 8 × uint64_t: panel_ready[2]+update_done[2]+full[2]+empty[2]
    static constexpr size_t T_SZ  = NB * NB;
    static constexpr size_t RD_SZ = NB;
    static constexpr size_t W_SZ  = NB * TILE_N;
    static constexpr size_t W2_SZ = NB * TILE_N;
    static constexpr size_t SA_G1 = STAGES * KT * NB;
    static constexpr size_t SA_G3 = (size_t)M_TILE * TILE_N;
    static constexpr size_t SA_SZ = SA_G1 > SA_G3 ? SA_G1 : SA_G3;
    static constexpr size_t SB_SZ = STAGES * KT * TILE_N;
    static constexpr size_t RED_SZ = 8;
    // pre-align bytes = barriers + T + Rd + W + W2 (before 128B alignment)
    static constexpr size_t pre_align = MBAR_SZ * 8 + (T_SZ + RD_SZ + W_SZ + W2_SZ) * sizeof(Real);
    static constexpr size_t aligned = (pre_align + 127) & ~size_t(127);
    static constexpr size_t total_floats = T_SZ + RD_SZ + W_SZ + W2_SZ + SA_SZ + SB_SZ + RED_SZ;
    static constexpr size_t bytes = aligned + (SA_SZ + SB_SZ + RED_SZ) * sizeof(Real) + 32;
};

template<typename Real, int NB, int TILE_N, int CHUNK_M, int CLUSTER_SIZE>
__global__ __launch_bounds__(256)
void k_tqr_mega(
    Real* A, int m, int n, int lda, int n_panels, Real* Tpan,
    Real* Vg,                      // [2 * m * NB] double-buffered V (col-major, ld=m)
    Real* Tg)                      // [2 * NB * NB] double-buffered T (col-major, ld=NB)
{
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    const int crank = cluster.block_rank();   // 0..CLUSTER_SIZE-1
    const unsigned tid = threadIdx.x;
    const int nth = blockDim.x;               // 256
    const int nup = CLUSTER_SIZE - 1;          // 7 update blocks

    extern __shared__ __align__(16) unsigned char smem_raw[];
    // ---- smem layout: mbarriers FIRST (same offset in all CTAs for mapa) ----
    uint64_t* panel_ready = reinterpret_cast<uint64_t*>(smem_raw);     // [2]
    uint64_t* update_done = panel_ready + 2;                            // [2]
    Real* T_sm  = reinterpret_cast<Real*>(update_done + 2);             // NB*NB
    Real* Rd    = T_sm + NB * NB;                                       // NB
    Real* W     = Rd + NB;                                              // NB*TILE_N
    Real* W2    = W + NB * TILE_N;                                      // NB*TILE_N
    Real* V_ch  = W2 + NB * TILE_N;                                     // CHUNK_M*NB
    Real* C_ch  = V_ch + CHUNK_M * NB;                                  // CHUNK_M*TILE_N
    volatile Real* red = C_ch + CHUNK_M * TILE_N;                       // 8

    // ---- mbarrier init (rank 0 owns update_done; ranks 1..7 own panel_ready) ----
    if (crank == 0) {
        mbar_init(&update_done[0], (uint32_t)nup);
        mbar_init(&update_done[1], (uint32_t)nup);
    } else {
        mbar_init(&panel_ready[0], 1u);
        mbar_init(&panel_ready[1], 1u);
    }
    cluster.sync();   // publish mbarrier smem across the cluster

    // ---- warp-level reduction (8 warps, 256 threads) ----
    auto reduce8 = [&](Real val) -> Real {
        for (int o = 16; o > 0; o >>= 1) val += __shfl_xor_sync(0xFFFFFFFF, val, o);
        unsigned warp = tid / 32, lane = tid % 32;
        if (lane == 0) red[warp] = val;
        __syncthreads();
        if (warp == 0) {
            Real sum = (lane < 8) ? red[lane] : Real(0);
            for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFF, sum, o);
            if (lane == 0) red[0] = sum;
        }
        __syncthreads();
        return red[0];
    };

    for (int k = 0; k < n_panels; k++) {
        int buf = k & 1;
        int row0 = k * NB, col0 = k * NB;
        int mk = m - row0;
        int Bcur = (n - col0 < NB) ? (n - col0) : NB;
        int ts = (k + 1) * NB;
        int tn = n - ts;
        int phase = k / 2;
        uint32_t parity = (uint32_t)(phase & 1);

        Real* Vb = Vg + (size_t)buf * m * NB;
        Real* Tb = Tg + (size_t)buf * NB * NB;

        if (crank == 0) {
            // ============ PANEL: Householder factorization (L-carrier, fp32) ============
            // Wait for update of panel k-2 to finish (buffer reuse: Vg/Tg[buf]
            // was used by panel k-2's update). Parity = (phase-1)&1 (the
            // COMPLETED phase for panel k-2's update).
            if (k >= 2) {
                if (tid == 0) {
                    uint32_t wpar = (uint32_t)((phase - 1) & 1);
                    while (!mbar_try_wait_parity(&update_done[buf], wpar)) {}
                }
                __syncthreads();
            }
            // Wait for panel k-1's trailing update to finish writing to A.
            // Panel k reads A[row0:m, col0:col0+Bcur], which is the trailing
            // matrix updated by panel k-1's update (different buffer).
            // Without this wait, panel k reads stale pre-update data → wrong.
            if (k >= 1) {
                int buf_prev = (k - 1) & 1;
                int phase_prev = (k - 1) / 2;
                if (tid == 0) {
                    uint32_t wpar_prev = (uint32_t)(phase_prev & 1);
                    while (!mbar_try_wait_parity(&update_done[buf_prev], wpar_prev)) {}
                }
                __syncthreads();
                __threadfence();  // acquire: gmem writes visible to all panel threads
            }

            // Load panel A[row0:m, col0:col0+Bcur] → Vg[buf]
            for (int j = 0; j < Bcur; j++) {
                Real* Vcol = Vb + (size_t)j * m;
                const Real* Acol = A + (size_t)(col0 + j) * lda + row0;
                for (int i = (int)tid; i < mk; i += nth)
                    Vcol[i] = Acol[i];
            }
            for (int idx = (int)tid; idx < NB * NB; idx += nth)
                T_sm[idx] = Real(0);
            __syncthreads();

            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;
                Real sigma = 0;
                for (int i = j + (int)tid; i < mk; i += nth) {
                    Real x = Vj[i];
                    sigma = fma(x, x, sigma);
                }
                sigma = reduce8(sigma);
                if (tid == 0) {
                    Real x0 = Vj[j];
                    Real nm = sqrt(sigma);
                    Real alpha = (x0 >= 0) ? -nm : nm;
                    Real v0 = x0 - alpha;
                    Real beta = Real(2) / (fma(v0, v0, sigma - x0 * x0));
                    Real tau = beta * v0 * v0;
                    red[0] = alpha; red[1] = v0; red[2] = tau;
                }
                __syncthreads();
                Real alpha = red[0], v0 = red[1], tau = red[2];
                for (int i = j + 1 + (int)tid; i < mk; i += nth)
                    Vj[i] /= v0;
                if (tid == 0) { Rd[j] = alpha; T_sm[j + j * NB] = tau; }
                __syncthreads();

                for (int kk = j + 1; kk < Bcur; kk++) {
                    Real* Vkk = Vb + (size_t)kk * m;
                    Real dot = 0;
                    for (int i = j + (int)tid; i < mk; i += nth) {
                        Real vj = (i == j) ? Real(1) : Vj[i];
                        dot = fma(vj, Vkk[i], dot);
                    }
                    dot = reduce8(dot);
                    Real w = tau * dot;
                    for (int i = j + (int)tid; i < mk; i += nth) {
                        Real vj = (i == j) ? Real(1) : Vj[i];
                        Vkk[i] = fma(-w, vj, Vkk[i]);
                    }
                    __syncthreads();
                }
                if (j > 0) {
                    if ((int)tid < j) {
                        int c = (int)tid;
                        Real* Vc = Vb + (size_t)c * m;
                        Real z = 0;
                        for (int row = j; row < mk; row++) {
                            Real vk = (row == c) ? Real(1) : Vc[row];
                            Real vj = (row == j) ? Real(1) : Vj[row];
                            z = fma(vk, vj, z);
                        }
                        T_sm[c + j * NB] = z;
                    }
                    __syncthreads();
                    if (tid == 0) {
                        for (int i = 0; i < j; i++) {
                            Real w = 0;
                            for (int kk = i; kk < j; kk++)
                                w += T_sm[i + kk * NB] * T_sm[kk + j * NB];
                            T_sm[i + j * NB] = -tau * w;
                        }
                    }
                    __syncthreads();
                }
            }

            // Store panel to HBM: A[i,j] = (i==j) ? Rd[j] : V[i,j]
            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;
                Real* Acol = A + (size_t)(col0 + j) * lda + row0;
                for (int i = (int)tid; i < mk; i += nth)
                    Acol[i] = (i == j) ? Rd[j] : Vj[i];
            }
            for (int idx = (int)tid; idx < Bcur * Bcur; idx += nth) {
                int j = idx / Bcur, i = idx % Bcur;
                Real t = T_sm[i + j * NB];
                Tpan[(size_t)k * NB * NB + i + (size_t)j * NB] = t;
                Tb[i + (size_t)j * NB] = t;
            }
            // Clean Vg for update blocks: zero upper triangle, set diagonal to 1
            for (int j = 0; j < Bcur; j++) {
                Real* Vj = Vb + (size_t)j * m;
                for (int i = (int)tid; i < j; i += nth)
                    Vj[i] = Real(0);
                if (tid == 0) Vj[j] = Real(1);
            }
            __syncthreads();
            __threadfence();  // release: gmem writes visible to update blocks

            // Signal: panel ready → remote-arrive on each update block's panel_ready
            if (tid == 0) {
                for (int b = 1; b <= nup; b++) {
                    uint64_t remote = mbar_mapa(&panel_ready[buf], (uint32_t)b);
                    mbar_arrive_cluster(remote);
                }
            }

        } else {
            // ============ UPDATE: trailing matrix update (B-carrier) ============
            // Wait for panel k to be ready (parity = phase&1, the COMPLETED
            // phase for panel k's signal).
            if (tid == 0) {
                while (!mbar_try_wait_parity(&panel_ready[buf], parity)) {}
            }
            __syncthreads();

            if (tn > 0) {
                // Load T from Tg[buf] into smem
                for (int idx = (int)tid; idx < NB * NB; idx += nth)
                    T_sm[idx] = Tb[idx];
                __syncthreads();

                // Column strip for this update block
                int cols_per_block = (tn + nup - 1) / nup;
                int my_start = ts + (crank - 1) * cols_per_block;
                int my_end   = ts + crank * cols_per_block;
                if (my_end > ts + tn) my_end = ts + tn;

                for (int col = my_start; col < my_end; col += TILE_N) {
                    int tnn = (my_end - col < TILE_N) ? (my_end - col) : TILE_N;

                    // ---- GEMM1: W = V^T * C (NB × tnn, K=mk) via FFMA ----
                    for (int idx = (int)tid; idx < NB * TILE_N; idx += nth)
                        W[idx] = Real(0);
                    __syncthreads();
                    for (int i0 = 0; i0 < mk; i0 += CHUNK_M) {
                        int iend = (i0 + CHUNK_M < mk) ? (i0 + CHUNK_M) : mk;
                        int ch = iend - i0;
                        for (int j = 0; j < NB; j++) {
                            const Real* Vcol = Vb + (size_t)j * m + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                V_ch[i + j * CHUNK_M] = (i < ch) ? Vcol[i] : Real(0);
                        }
                        for (int nn = 0; nn < tnn; nn++) {
                            const Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                C_ch[i + nn * CHUNK_M] = (i < ch) ? Acol[i] : Real(0);
                        }
                        __syncthreads();
                        for (int idx = (int)tid; idx < NB * tnn; idx += nth) {
                            int nn = idx / NB, b = idx % NB;
                            int istart = (b > i0) ? (b - i0) : 0;
                            Real a = 0;
                            for (int i = istart; i < ch; i++)
                                a = fma(V_ch[i + b * CHUNK_M], C_ch[i + nn * CHUNK_M], a);
                            W[b + nn * NB] += a;
                        }
                        __syncthreads();
                    }
                    __syncthreads();

                    // ---- GEMM2: W2 = T * W (NB × tnn), T upper-triangular (FFMA) ----
                    for (int idx = (int)tid; idx < NB * TILE_N; idx += nth) {
                        int nn = idx / NB, b = idx % NB;
                        if (nn >= tnn) { W2[idx] = Real(0); continue; }
                        Real a = 0;
                        for (int kk = 0; kk <= b; kk++)
                            a = fma(T_sm[kk + b * NB], W[kk + nn * NB], a);
                        W2[b + nn * NB] = a;
                    }
                    for (int nn = tnn; nn < TILE_N; nn++)
                        for (int i = (int)tid; i < NB; i += nth)
                            W2[i + nn * NB] = Real(0);
                    __syncthreads();

                    // ---- GEMM3: C -= V * W2 (mk × tnn, K=NB) via FFMA ----
                    for (int i0 = 0; i0 < mk; i0 += CHUNK_M) {
                        int iend = (i0 + CHUNK_M < mk) ? (i0 + CHUNK_M) : mk;
                        int ch = iend - i0;
                        for (int j = 0; j < NB; j++) {
                            const Real* Vcol = Vb + (size_t)j * m + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                V_ch[i + j * CHUNK_M] = (i < ch) ? Vcol[i] : Real(0);
                        }
                        for (int nn = 0; nn < tnn; nn++) {
                            const Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                C_ch[i + nn * CHUNK_M] = (i < ch) ? Acol[i] : Real(0);
                        }
                        for (int nn = tnn; nn < TILE_N; nn++)
                            for (int i = (int)tid; i < CHUNK_M; i += nth)
                                C_ch[i + nn * CHUNK_M] = Real(0);
                        __syncthreads();

                        for (int idx = (int)tid; idx < CHUNK_M * TILE_N; idx += nth)
                            W[idx] = Real(0);
                        __syncthreads();
                        for (int j = 0; j < NB; j++) {
                            for (int i = (int)tid; i < CHUNK_M; i += nth) {
                                Real vj = V_ch[i + j * CHUNK_M];
                                for (int nn = 0; nn < TILE_N; nn++)
                                    W[i + nn * CHUNK_M] = fma(vj, W2[j + nn * NB], W[i + nn * CHUNK_M]);
                            }
                            __syncthreads();
                        }
                        for (int idx = (int)tid; idx < CHUNK_M * TILE_N; idx += nth)
                            C_ch[idx] -= W[idx];
                        __syncthreads();

                        for (int nn = 0; nn < tnn; nn++) {
                            Real* Acol = A + (size_t)(col + nn) * lda + row0 + i0;
                            for (int i = (int)tid; i < ch; i += nth)
                                Acol[i] = C_ch[i + nn * CHUNK_M];
                        }
                        __syncthreads();
                    }
                }
            }

            // Signal: update done → remote-arrive on block 0's update_done.
            // ALL update blocks must arrive (even if tn=0 or no columns assigned),
            // otherwise block 0 hangs at panel k+2.
            __syncthreads();
            __threadfence();  // release: gmem writes visible to panel block
            if (tid == 0) {
                uint64_t remote = mbar_mapa(&update_done[buf], 0u);
                mbar_arrive_cluster(remote);
            }
        }
    }
}

// ============================================================ k_tqr_mega_wgmma
// W3' carrier (sub-unit a): wgmma TF32 consumer. The kernel body lives ENTIRELY in
// wgmma_carrier.cu (compiled WITHOUT LTO — nvcc 13.0's LTO device-link ptxas defaults to
// sm_90, rejecting wgmma/setmaxnreg). bench.cu NEVER references the kernel directly; it
// calls the HOST LAUNCHER functions below (default visibility, the cusolverdx pattern).
// This dodges both the hidden-stub error (non-separable kernel) AND LTO poisoning (the
// non-separable fatbin doesn't enter nvlink's LTO pipeline). Sole path: float, NB=128,
// TILE_N=64, CHUNK_M=128, cluster=8. B32 swizzle (SBO=256) proven (daceda7, 5.96e-08).
//
// wgmma_carrier_setattr: cudaFuncSetAttribute (smem + cluster). Call once in create().
// wgmma_carrier_smem:    returns MegaSmemWgmma<float,128,64>::bytes (for assert/print).
// wgmma_carrier_launch:  builds TMA descriptors (SWIZZLE_32B) + cudaLaunchKernelExC
//                        (cluster=8, block=256). Async — caller does watchdog poll + sync.
bool wgmma_carrier_setattr();
size_t wgmma_carrier_smem();
void wgmma_carrier_launch(float* A, int m, int n, int lda, int npanels, float* Tpan,
                          float* rVg, float* rTg, cudaStream_t st,
                          int* prog_k, int* upd_phase, int wd);


// ---------------------------------------------------------------- TqrSchedule
// Per-rung ν-ladder / S-ladder parameters (spec §2/§4/§7, Lemma N / Cor. S).
// Tracks ν_λ, S_λ=n/ν_λ, W_λ (pipeline width), r_λ (fan-out), c_λ (replicate) at each of
// the 4 rungs, recording BOTH the code value and the spec target so the deviation is
// visible (T2 RESEARCH_MEMO §2/§6). This is the Lemma-N/Cor.-S code instantiation —
// scaffolding for the F5 W-sweep + Track-B F2. Print via LRQR_SCHEDULE=1.
struct TqrRung {
    size_t capacity_bytes; // measured effective capacity M-hat at this boundary
    int nu;       // ν_λ: panel width at this rung (CODE value)
    int S;        // S_λ = n/ν_λ: phase count (CODE value, Cor. S floor)
    int W;        // W_λ: pipeline width / domino depth (CODE value)
    int r;        // r_λ: fan-out (CODE value)
    int c;        // c_λ: replicate / buffer count (CODE value)
    int nu_spec;  // ν_λ: spec target (√M̂_λ or b)
    int S_spec;   // S_λ: spec target (n/ν_spec)
    int W_spec;   // W_λ: spec target (super-domino 15)
    int r_spec;   // r_λ: spec target (tiered 33/438)
    int c_spec;   // c_λ: spec target
    // PHASE 2: the width the proof prescribes, nu* = min{sqrt(M), B(c)/n, chi}
    // (docs/31:262), and which of the three terms binds. Emitted beside the CODE
    // value so the two can be compared per cell instead of assumed equal.
    int nu_star = 0;
    int nu_star_binding = 0;   // 0 = capacity sqrt(M), 1 = panel traffic B/n, 2 = tree depth chi
    // Step 4: the two proved quantities, kept so they can be CHECKED rather than
    // asserted. K = S (ordered frontiers at this boundary) and D = tree depth, so
    // the synchronization theorem (31:345) predicts S_lambda = Theta(K + D); B_words
    // is B_lambda(c) from 31:221, the communication budget Q_lambda must match to
    // within a constant (31:297). Both were already computed here for nu* and then
    // thrown away.
    int D = 1;
    double B_words = 0.0;
};
struct TqrSchedule {
    TqrRung rung[4];  // [0]=reg→shared(leaf) [1]=shared→DSM [2]=DSM↔L2 [3]=L2→HBM
    int b = 0;        // spec target b=√(M̂₁/wordsize) (169 fp64 / 240 fp32)
    int NB = 0;       // actual leaf block (128/64) — DEVIATION recorded
    int n = 0;        // matrix n (S_λ evaluated at this n)
};

// ---------------------------------------------------------------- plan
template<typename Real, int NB, int SLAB>
struct Plan {
    int m = 0, n = 0, B = 0;
    int kmax = 0;             // min(m,n): the number of reflectors the factorization has.
                              // Equals n on every m>=n cell, so no existing path changes.
    size_t waxElemsCached = 0;  // Wax/Wbx capacity, so ormqr can refuse rather than overrun
    // Scale-probe reduction buffer. Allocated ONCE in create(), not per geqrf call:
    // a cudaMalloc plus a device sync on every call cost 6-14% on the 1048576x64
    // extreme cell, where the whole factorization is ~20 ms. The probe's ALGORITHMIC
    // cost is one read of A (~0.9% there, ~0.1% at n=65536); the rest was mine.
    // 2048, not 256. At 256 blocks the probe gets 2 blocks/SM, and a
    // bandwidth-bound kernel at that occupancy runs far under the measured
    // 4.0 TB/s -- worth ~6% of the 1048576x64 extreme cell, where the whole
    // factorization is ~20 ms. The launch uses min(kAbsmaxBlocks, work/256) so small
    // matrices do not pay for blocks with nothing to do.
    static constexpr int kAbsmaxBlocks = 2048;
    // Domino minipanel width, host side. The four-move witness counts minipanel
    // GENERATIONS (NB/PW of them per lane) and so needs the same width the device
    // kernels use, but Plan is Plan<Real,NB,SLAB> and carries no PW of its own --
    // so mv_combine/mv_pipeline referenced an identifier that was never in scope.
    // It went unnoticed because bench.cu's and wgmma_carrier.cu's objects were only
    // stale, not correct: the first edit to lrqr.cuh that forced a full rebuild
    // turned it into six hard errors.
    //
    // 16 is not a choice made here. Every domino launch on the Plan path
    // instantiates the kernels at their default -- k_panel_fused<Real,SLAB,NB>,
    // d_panel_body<Real,SLAB,NB,PW> with PW defaulted -- so this must track the
    // `int PW = 16` default on those templates. A default template argument cannot
    // be static_assert'd against from here; if that default ever moves, this moves
    // with it and the witness counts go wrong silently otherwise.
    static constexpr int PW = 16;
    double* absmaxPart = nullptr;
    bool tf32far = true;      // §13.4 far-update tier
    bool lookahead = true;    // §9 d_look = 1
    cublasHandle_t cb0 = nullptr, cb1 = nullptr;   // one handle per stream (never rebound)
    cublasHandle_t cbT = nullptr;                  // Tsp compound chain on sP
    // cuBLASLt with SM_COUNT_TARGET (LRQR_LT_SM env var): limits GEMM SM footprint
    // to leave room for the cooperative panel kernel, reducing SM-sharing degradation.
    cublasLtHandle_t lt0 = nullptr, lt1 = nullptr, ltT = nullptr;
    cublasLtMatmulDesc_t ltDescTN = nullptr, ltDescNN = nullptr;
    cublasLtMatmulDesc_t ltDescTN_plain = nullptr, ltDescNN_plain = nullptr;
    cublasLtMatmulDesc_t ltDescTN_f32 = nullptr;  // fp32 compute for 3×TF32 GEMM2
    void* ltWorkspace = nullptr;
    int ltSmTarget = 0;  // 0 = disabled, >0 = SM count target
    int ltSplitK = 0;    // 0 = disabled, >0 = split-K factor for far-rest GEMMs
    cudaEvent_t evTsp = nullptr;
    cudaStream_t sR = nullptr;                     // super-DSM rest slices (k_custom_apply, no cuBLAS)
    cudaStream_t sN = nullptr;                     // domino REST applyQt (cuBLAS, separate from s1 far-rest)
    cudaEvent_t evRst = nullptr;
    bool useSuper = false;
    cudaStream_t sP = nullptr;                     // persistent panel pipeline (own stream)
    void *cbws0 = nullptr, *cbws1 = nullptr, *cbwsT = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    cudaEvent_t evFar[3] = {nullptr, nullptr, nullptr}, evSlice = nullptr, evIn = nullptr;
    bool evFarValid[3] = {false, false, false};

    int mpadMax = 0, Nmax = 0, npanels = 0;
    // TqrSchedule: per-rung ν/S/W/r/c ladder (T2 RESEARCH_MEMO §2). Computed in create(),
    // printed when LRQR_SCHEDULE=1. Code values track the actual NB/B/fch; spec targets
    // record the spec's √M̂_λ ladder so deviations are visible.
    TqrSchedule sched;
    // L2 access-policy window (spec §10 Replicate at L2 = partition pair c=2):
    // V6 W1 — opt-in pending far-rest regression box-bench (memo R5). set-aside
    // ladder default = min(box-max 37.5MB, B²·wordsize), computed in create().
    // hitRatio=0.5 (concurrent-stream guidance, S1). LRQR_L2HIT/L2SETASIDE override.
    bool useL2win = false;
    float l2hit = 0.5f;
    size_t l2setaside = size_t(37.5 * 1024.0 * 1024.0);
    int fusedGrid = 0, fusedSmem = 0, smCount = 0;
    int waveTopSmem = 0; bool waveTopOK = false;
    bool usePersist = false; int pgrid = 96;
    bool gemmsWarmed = false;   // cuBLAS lazy-loads its kernels on first use, ignoring
                                // CUDA_MODULE_LOADING=EAGER (verified). A load that fires
                                // WHILE the resident persist grid runs deadlocks. So the very
                                // first geqrf runs the normal (non-persist) path — identical
                                // GEMM shapes — to load every cuBLAS kernel once; persist
                                // (which never triggers a load) engages from the 2nd call.
    PanelArgs<Real>* dArgs = nullptr;
    int *dReady = nullptr, *dFin = nullptr;          // device views of mapped doorbells
    volatile int *hReady = nullptr, *hFin = nullptr; // host views (pinned, zero-copy)
    unsigned *dBcnt = nullptr, *dBgen = nullptr;
    std::vector<PanelArgs<Real>> hArgs;
    // green-context SM partition (spec §9 overlap): panel on the big partition,
    // update GEMMs confined to the small one, so both are genuinely co-resident.
    CUgreenCtx gcB = nullptr, gcS = nullptr;
    cudaStream_t sCols = nullptr;                 // de-fused: reserved cols partition
    cudaEvent_t evC0 = nullptr, evC1 = nullptr;   // cols ↔ boundary handoff
    CUcontext ctxB = nullptr, ctxS = nullptr;
    bool green = false;
    int smallSMs = 0;
    bool useResident = false;     // resident warp-spec kernel (spec §9/§11, LRQR_RESIDENT=1)
    int residentMmax = 0;         // M_MAX for resident kernel (computed in create)
    Real* rVg = nullptr;          // [2*m*NB] double-buffered V in gmem
    Real* rTg = nullptr;          // [2*NB*NB] double-buffered T in gmem
    volatile unsigned* rPanelGen = nullptr;  // [2] panel generation flags
    volatile unsigned* rUpdateCnt = nullptr; // [2] update completion counters
    int rGrid = 132;              // cooperative grid size (all SMs)
    // ---- T5 megakernel (k_tqr_mega, cluster-scope mbarrier, LRQR_MEGA=1) ----
    bool useMega = false;         // T5 megakernel: mbarrier resident QR
    bool useMegaWgmma = false;    // W3' carrier: wgmma TF32 consumer variant (capacity-ladder engaged)
    // Deadlock-proof watchdog (memo §5): host-mapped prog_k/upd_phase let the host poll
    // forward progress without syncing the kernel. prog_k = panels started (rank 0 writes
    // at top of each panel); upd_phase = updates completed (rank 0 mirrors update_done).
    // Stalled prog_k (>2s) OR upd_phase lagging prog_k by >STAGES+1 → deadlock verdict.
    int* dProgK = nullptr;        // device view of host-mapped prog_k
    volatile int* hProgK = nullptr;  // host view (pinned, zero-copy)
    int* dUpdPhase = nullptr;     // device view of host-mapped upd_phase
    volatile int* hUpdPhase = nullptr;
    int megaWd = 1;               // watchdog on (1) for shakedown; tiny overhead (<0.1%)
    static constexpr int megaCluster = 8;  // H200 max cluster (T1-verified)
    bool useFused = true;         // fused cooperative panel for the small/mid-n regime
    bool useDomino = false;       // single-kernel sequential panel (spec §5) — no recon
    unsigned* gbar = nullptr;     // gmem barrier for non-cooperative domino (count + generation)
    Real *domScal = nullptr, *domSlot = nullptr, *domProw = nullptr;
    double *domSigD = nullptr, *domDslotD = nullptr;  // fp64 shadows
    double *domDcolP = nullptr;   // A6: per-block column partials, 2 generations
    double *domDcolR = nullptr;   // A6: reduced [2][PW+1], from the barrier
    Real *domPp = nullptr, *domPsum = nullptr, *domT16 = nullptr;
    static constexpr int cLmax = 8;
    // Per-lane domino buffers for concurrent execution (c_L>1)
    Real *domScalL[cLmax] = {}, *domSlotL[cLmax] = {}, *domProwL[cLmax] = {};
    Real *domPpL[cLmax] = {}, *domPsumL[cLmax] = {}, *domT16L[cLmax] = {};
    double *domSigDL[cLmax] = {}, *domDslotDL[cLmax] = {};  // per-lane fp64 shadows (lanes run concurrently)
    unsigned* gbarL[cLmax] = {};
    cudaEvent_t evDom[cLmax] = {};
    cudaEvent_t evPrePanel = nullptr;
    cudaStream_t domSt[cLmax] = {};  // per-lane domino streams (s0, sN, sP, sR, ...)
    unsigned long long* domT = nullptr;
    int* pipeF = nullptr;         // pipelined-domino doorbells: 8 uflag + 8 stCnt + done
    bool usePipe = false;
    int pipeNbl = 49;             // 1 driver + 48 updaters (leave SMs for far GEMMs)
    Real *domUg = nullptr, *domW2 = nullptr, *domGs = nullptr;   // de-fused domino
    bool useDefuse = false;
    bool corFC = false;           // T4 Cor.FC: fuse Tsp-compound onto s0 (spec §6)
    bool useDsm = false;          // DSM-pipe (ROADMAP §6)
    int dsmU = 28;                // updater blocks in the DSM kernel
    // (G6 takeover: the LRQR_GMEM_DOMINO experimental rung was removed — measured
    // 2.1× slower, finding preserved in G2_PANEL_V13_REPORT.md; the batched-panel
    // kernel class is the G2 follow-up. See the dispatch-site comment.)
    // ---- c_L lanes (spec §7.2): split-K fan-out of the latency carrier ----
    // Each lane runs its own domino on a disjoint row-slab; the c_L partial R-states
    // are merged by ONE cross-lane ⊕ (k_merge_geqrf, Householder-on-stack — spec §3/§14,
    // NEVER Gram). The (V,T) update is a two-level WY apply (no reconstruction on the
    // chain — LD-QR §5 Principle S). c_L* is selected from the live capacity
    // ladder and reduction depth (§7.2: lanes cut chain latency by ≈c_L).
    int cLused = 0;
    Real *domTpanL = nullptr;     // c_L * NB*NB per-lane compound T
    // c_L merge-tree nodes, as a TABLE indexed by c_L rather than one list.
    // The tree for a given c_L is a pure function of c_L, and c_L has only
    // cLmax+1 possible values -- so the whole table is 63 int2 and belongs on the
    // device once, not rebuilt and re-uploaded on the panel path. Row c starts at
    // d_nodes2d + c*kNodeStride and holds that tree's nodes in bottom-up level
    // order; a row is at most cLmax-1 long because a binary merge of c leaves has
    // c-1 internal nodes.
    int2* d_nodes2d = nullptr;
    static constexpr int kNodeStride = cLmax - 1;
    int nnodes2d = 0;
    bool use25d = false;          // live when the capacity/depth formula selects c_L>1
    // Two-level WY apply state (set by panel_25d, read by applyQt_25d/mergeApplyQt).
    // Valid immediately after panel_25d returns — used by near/rest/far updates.
    int last_cL = 1;
    int last_slabStart[cLmax] = {0};
    int last_slabSize[cLmax] = {0};
    int last_nm = 0;              // number of merge nodes (= c_L - 1)
    int2 last_nodes[cLmax] = {};  // merge tree nodes (bottom-up level order)
    // Deferred-far save buffers: double-buffered so panel k's far (s1) doesn't block
    // panel k+1's copy (s0). Sync only at panel k+2 (same slot reuse).
    Real *VmSav[2] = {nullptr, nullptr}, *TmMSav[2] = {nullptr, nullptr}, *domTpanLSav[2] = {nullptr, nullptr};
    int sav_cL[2] = {1, 1}, sav_slabStart[2][cLmax] = {}, sav_slabSize[2][cLmax] = {};
    int sav_nm[2] = {0, 0}; int2 sav_nodes[2][cLmax] = {};
    cudaEvent_t evMergeCpy[2] = {nullptr, nullptr}, evFar25[2] = {nullptr, nullptr};
    bool far25Pending[2] = {false, false};
    // ---- THE FAR ROUTE IS A SUPER-PANEL PROPERTY, NOT A PER-PANEL ONE ----
    // A super-panel updates its far region [sp+Bcur, n) by exactly one of two routes:
    //
    //   Tsp route (c_L==1):  every panel folds its transform into the running Tsp
    //                        compound, and ONE combined rank-Bcur far apply runs at
    //                        the super-panel end.
    //   2.5D route (c_L>1):  the Tsp compound is invalid (the panel's reflectors are
    //                        per-slab plus merge nodes, not a single contiguous V),
    //                        so each panel applies its OWN far immediately, and the
    //                        combined far is skipped -- `if (any25d) continue`.
    //
    // Mixing them inside one super-panel LOSES far updates. c_L is chosen per panel
    // from mk = m - row0, which decreases down the super-panel, so the last panels of
    // a super-panel whose mk falls below 2*NB drop to c_L=1, take the `else` branch,
    // fold into Tsp -- and then `any25d` (set by their c_L>1 predecessors) suppresses
    // the combined far that would have applied them. Their transform never reaches
    // the far columns.
    //
    // Signature, and it is unmistakable: ||I-Q^T Q|| stays clean while ||A-QR||
    // explodes -- a missed trailing update, not a bad Householder. Measured at
    // c_L>=2 on 512x4096 (RESULT(1)=4.0e+12, RESULT(2)=0.91) and 2048x3000
    // (4.5e+11, 0.73), which are precisely the two --validate-lapack classes with
    // n > kmax=min(m,n), i.e. wfar_super>0. Every square and every tall cell with
    // n<=B has wfar_super==0 and never reaches the far path at all, which is why
    // both accuracy suites certified this for as long as they have.
    //
    // FIX: the route is decided ONCE per super-panel and every panel honours it.
    //   far25Route  -- this super-panel is on the 2.5D route; a c_L==1 panel inside
    //                  it must still issue its own far (degenerate 1-lane form).
    //   sp_c1Locked -- a c_L==1 panel already folded into Tsp with a far region
    //                  pending, so no LATER panel may switch to the 2.5D route.
    // Both reset at each super-panel head.
    bool far25Route = false, sp_c1Locked = false;
    Real *Pbuf = nullptr, *QT = nullptr, *QS = nullptr;
    Real *Vm = nullptr, *TmM = nullptr, *TmL = nullptr, *tauL = nullptr, *taum = nullptr;
    // W6(G7) compact-storage footprint telemetry: sum of every device allocation made by
    // create() (cuBLAS workspaces, all ralloc'd Real arrays, domT/pipeF/gbar/... misc).
    // Reported on the [TQR_TELEMETRY] line alongside the LAPACK baseline m*n + n + n*NB.
    size_t footprintBytes = 0;
    // Per-panel merge data save for orgqr Q reconstruction (cL>1 panels).
    // Vm/TmM/domTpanL are overwritten each panel during geqrf; these save a copy
    // per panel so orgqr can reconstruct Q = Q_merge · diag(Q_j) for every panel.
    Real *domTpanL_pan = nullptr, *Vm_pan = nullptr, *TmM_pan = nullptr;
    // W6(G7) Phase-2: the per-panel save buffers (3·npanels·cLmax·NB²·w, ~1.5 GiB
    // at fp64@131072²) are written every panel during geqrf but read ONLY by
    // orgqr (the validation/bench Q-reconstruction path). In the production path
    // (LRQR_ONLY — no orgqr call) they are pure waste. saveForOrgqr gates both
    // the allocation and the per-panel save copies; when false, geqrf skips
    // ~npanels D2D copies (a small speed win) and frees the footprint for the
    // G7(b) OOM recession (fp64@131072² prod: 141.34→~139.8 GiB, under 141).
    // orgqr asserts saveForOrgqr (it is only called in the bench path where the
    // flag is true). RESEARCH_MEMO §4.1 (Vm_pan/TmM_pan/domTpanL_pan).
    bool saveForOrgqr = true;
    int *cL_pan = nullptr, *nm_pan = nullptr;
    int *slabStart_pan = nullptr, *slabSize_pan = nullptr;   // [npanels*cLmax]
    int2 *nodes_pan = nullptr;                                // [npanels*cLmax]
    Real *SPVx[3] = {nullptr, nullptr, nullptr}, *Tspx[3] = {nullptr, nullptr, nullptr};
    Real *Wax[3] = {nullptr, nullptr, nullptr}, *Wbx[3] = {nullptr, nullptr, nullptr};
    Real *Yt = nullptr;  // pre-transposed Y for far-update TN GEMM 3
    Real *Ytx[3] = {nullptr, nullptr, nullptr};  // triple-buffered Yt
    Real *Tpan = nullptr, *Sg = nullptr;
    Real *Zb = nullptr, *M1b = nullptr, *Wn1 = nullptr, *Wn2 = nullptr;
    Real *Wr1 = nullptr, *Wr2 = nullptr;
    cudaEvent_t evPD = nullptr, evRest[2] = {nullptr, nullptr};
    bool evRestValid[2] = {false, false};
    // F8 Step-1 per-stage timer (LRQR_STAGET): span of s0 (panel+slice carrier) and
    // s1 (far-rest bandwidth carrier) + exact far-rest flop accumulator. Overlap φ and
    // far-rest throughput disambiguate compute-bound (~47 TF) from HBM-bound (the
    // falsification gate for the aspect-aware fix). See RESEARCH_MEMO §(b)/SINGLE HIGHEST-RISK.
    cudaEvent_t evSt0a = nullptr, evSt0b = nullptr, evSt1a = nullptr, evSt1b = nullptr;
    double stagetFarFlops = 0.0;
    bool staget = false;
    // 3×TF32: fp32-accurate far-update via 3 TF32-TC GEMMs (§13.4). Split each fp32 operand
    // into hi+lo TF32 (Sterbenz-exact), do 3 GEMMs: A_hi·B_hi + A_hi·B_lo + A_lo·B_hi.
    // Y/Yt are constant across all far chunks → pre-split once per super-panel (amortized).
    // C/W2 change per chunk → split per applyQt call. GEMM2 (T^T·W1) uses fp32 CUDA cores.
    // Split-K on the dominant A_hi·B_hi term chunks the K-reduction for fp32-class accuracy
    // at large K (single TF32 GEMM accumulates ~linearly in K; chunked + fp32 beta=1 cancels).
    // W4a: tuned 128→256. At 256, fp32-OoM holds at ALL n (16/16 PASS, worst 1.16× cuSOLVER @n=32768)
    // and n=65536 speed is +14.3% (59.2 vs 51.8 TF) from halving split-K dispatch+reduce overhead.
    // 512 FAILS at n=32768 (resid 5.99e-4 = 202× cuSOLVER: RZ-in-TC bias threshold crossed).
    // 0 FAILS at n=32768 (no C1 mitigation: C2+C3 alone insufficient at large K). Continuous param.
    bool use3xtf32 = false;
    int  splitK3x = 256;        // K-chunk for A_hi·B_hi split-K (0 disables; env LRQR_3X_SPLITK)
    // W6-P4 (G7 Phase-3): max K-slice for the in-place Ozaki C-split. Chi/Clo are sized
    // maxSk3x·maxFarchunk3x (not m·maxFarchunk3x) and re-split per K-slice inside the
    // GEMM1 K-loop (OZAKI_PHASE3_DESIGN §3). 2048 = upper branch of the sk ternary below.
    static constexpr int maxSk3x = 2048;
    Real *Yhi[3] = {nullptr, nullptr, nullptr}, *Ylo[3] = {nullptr, nullptr, nullptr};
    Real *Ythi[3] = {nullptr, nullptr, nullptr}, *Ytlo[3] = {nullptr, nullptr, nullptr};
    // Takeover: Chi/Clo DOUBLE-BUFFERED so the K-slice re-split (producer kernel)
    // pipelines on a side stream against the consumer GEMMs — recovers the
    // OZAKI_PHASE3_DESIGN §6 "7% launch overhead" (split serialized on st before).
    Real *Chi[2] = {nullptr, nullptr}, *Clo[2] = {nullptr, nullptr};
    cudaStream_t s3xsp = nullptr;            // split-producer stream (3×TF32 pipeline)
    cudaEvent_t evSp3x[2] = {}, evG3x[2] = {};  // split-done / gemms-done per buffer parity
    Real *W2hi = nullptr, *W2lo = nullptr;   // per-chunk W2 split (B × maxFarchunk3x)
    int maxFarchunk3x = 0;                    // max far-rest chunk for 3×TF32 scratch sizing
    // ENGAGEMENT TELEMETRY (spec §10 Theorem-U; AUDIT_V6 §6; RESEARCH_MEMO): observability-only
    // counters proving which mechanisms engaged in the default path. NEVER gate any dispatch
    // (Theorem U: telemetry is a witness, not a switch). Reset per geqrf; printed once per
    // process (first geqrf) on stderr so bench stdout parsing is unaffected.
    // ---- the four 2.5D moves, per rung. Host counters at the real dispatch sites.
    // Observability ONLY -- never read by a dispatch (Theorem U: telemetry is a
    // witness, not a switch). Without these the question "is this 2.5D?" can only be
    // answered by reading source, which is how it went unanswered for so long.
    //   Split     disjoint row intervals a panel is partitioned into
    //   Replicate lanes each carrying their own copy of the panel state
    //   Combine   ordered Householder merge-tree nodes reducing those lanes
    //   Pipeline  overlapping generations within a lane
    long long mv_split[4] = {0, 0, 0, 0};
    long long mv_replicate[4] = {0, 0, 0, 0};
    long long mv_combine[4] = {0, 0, 0, 0};
    long long mv_pipeline[4] = {0, 0, 0, 0};
    // Widest replica wave actually co-resident, from panel_25d's occupancy
    // calculation. The cost model needs it: lanes past this one are serialized
    // into later waves, so they add merge depth without shortening any chain.
    int mv_lanes_wave = 0;
    // S_lambda instrumentation at rung 1. mv_merge_levels is D summed over panels
    // -- the number of ordered combine stages the algorithm owes. mv_merge_launches
    // is how many DEVICE-VISIBLE rounds were spent paying them. The theorem
    // (31:345) says the tree costs Theta(K+D), so the per-level launch loop
    // (launches == levels) is the K*D form and the wavefront (launches == K) is the
    // K+D form. Their ratio is the compliance number, emitted rather than claimed.
    long long mv_merge_launches = 0, mv_merge_levels = 0;
    // ---- Step 3: the diagonal readiness edge X(k-1,j+1) -> X(k,j) (31:329) ----
    // Stage-2 (mergeApplyQt) writes only the top NB rows of the two slabs per merge
    // node, and slab s is consumed as a "bot" at level ctz(s) and never touched
    // again. So epoch k+1's lane j does NOT need epoch k's whole tree -- only the
    // levels whose top bands intersect that lane's rows. evM2lvl[l] marks Stage-2
    // level l complete; evS1done marks Stage 1 complete (needed by a lane whose rows
    // no top band touches at all). Recording per level is what lets the release be
    // per lane instead of one blanket barrier.
    // Levels the GLOBAL near apply is allowed to perform. -1 = all (default).
    // When the diagonal edge is armed, the near apply stops at releaseLevel and the
    // NEXT epoch's lanes apply the remaining levels themselves, on their own rows,
    // from replicated state -- so every node is applied exactly once, by whoever the
    // invariant says owns it. This is the "required update occurs locally at that
    // stage" clause of 31:329, which is the enabling condition the three earlier
    // release-timing attempts were missing.
    int m2_max_level = -1;
    int m2_near_cap  = -1;   // cap for the NEAR apply only; see panel_25d
    // Levels the previous epoch's tree HAS, vs levels the global near apply actually
    // performed. Capping the apply shrank last_m2_levels, so the next epoch computed
    // fullTree = cap and its self-apply guard (pv_lv > release+1) was always false --
    // the skipped levels were applied by nobody, reproducing the uncapped failure
    // byte for byte. The two quantities are now distinct.
    int last_m2_applied = 0;
    // Per-lane NBxNB scratch for the local apply. Small and lane-private so two
    // lanes servicing different skipped nodes cannot collide; Pbuf is not usable
    // here because it is written by k_extract_R_lanes later in the same panel.
    Real* laneW1[cLmax] = {}; Real* laneW2[cLmax] = {};
    cudaEvent_t evM2lvl[cLmax] = {};
    cudaEvent_t evS1done = nullptr;
    // The previous panel's rest-slice completion, handed down from the panel loop
    // because the parity index is loop-local. Lanes released by the diagonal edge
    // must still wait on it: the rest slice runs on sN and writes rows the next
    // panel reads, and the per-level events are recorded on s0 only.
    cudaEvent_t restEv = nullptr;
    bool restEvValid = false;
    // p_r chosen jointly with c by replicaGridArgmin. 0 = not selected (uncalibrated
    // model, or no feasible tuple), in which case laneDsmGrid keeps its own breadth
    // rule. This is the only channel by which the planner's row-partition decision
    // reaches the DSM launch, so a nonzero value here IS the 2.5D grid being executed
    // rather than merely computed -- the failure mode this plan opened by naming.
    int gridPr = 0;
    int last_m2_levels = 0;      // Stage-2 levels recorded for the PREVIOUS panel
    int last_row0 = -1;          // that panel's row0, to place its slab bands
    // Opt-in while it is measured, exactly like LRQR_CL. Off, the panel loop keeps
    // its blanket evPrePanel barrier and the schedule is bit-identical; on, lanes
    // are released per the diagonal edge. Ships as default only behind 256/256.
    // A function-local static, not a const data member: a const member would delete
    // Plan's implicit copy-assignment, and read-once semantics are what the other
    // measurement knobs (LRQR_CL, LRQR_LANEDEPTH) already use.
    // ---- Compose replication WITH the DSM rung, instead of opting out of it ----
    // The profile said k_panel_domino's duration is flat in c and I first read that
    // as the panel being floor-bound. It is not. nsys at 16384^2 fp64 lists
    // k_panel_dsm 930x/359ms on the c_L=1 arm and NOT AT ALL on the c_L=2 and
    // c_L=4 arms, where only k_panel_domino appears. panel() dispatches c_L>1 to
    // panel_25d and returns (:7092 ff) BEFORE reaching the DSM branch at :7229, so
    // replication silently drops off the cluster/DSM carrier onto the older
    // global-memory domino -- and every c>1 measurement so far has been comparing
    // replication against a different, faster kernel.
    //
    // That is also where the two mechanisms collide. The DSM branch already
    // realizes rung-1 replication in hardware: its cluster breadth is generated
    // from the shared->DSM replica capacity (clusterFloor = rung[1].c + rung[0].r),
    // and the C blocks of a cluster reduce through DSM rather than through global
    // Pp/Psum. So the spec's rung-1 replication and the c_L lane mechanism are two
    // implementations of the same move, and only one of them was on the fast path.
    // Composing them is the point (the four moves are meant to hold together, not
    // to be alternatives).
    //
    // RESIDENCY IS THE CATCH. k_panel_dsm spins on pipeF doorbells across clusters,
    // so its blocks must be co-resident; per lane G ~ C + dsmU (dsmU = 28 updater
    // blocks), and c lanes of that oversubscribe 132 SMs quickly. The caller
    // therefore checks the TOTAL grid across lanes and declines rather than
    // deadlocking. Opt-in while it is measured.
    static bool laneDsm() {
        static const bool on = [] {
            const char* e = getenv("LRQR_LANEDSM"); return e && atoi(e) != 0;
        }();
        return on;
    }
    // Blocks k_panel_dsm would need for a slab of mkLane rows, or 0 if it does not
    // apply. Pure arithmetic, so the caller can total it across lanes first.
    int laneDsmGrid(int mkLane, int* Cout, int* rpbOut) const {
        const int pipeMaxL = clRpbMaxDB + NB;
        if (!(useDsm && mkLane > std::min(pipeMaxL, (int)pipeRowMax) &&
              mkLane <= dsm2RowMax)) return 0;
        const int replicaCap = std::max(1, sched.rung[1].c);
        const int clusterQuantum = 16 + replicaCap + 1;
        const int clusterFloor = replicaCap + std::max(1, sched.rung[0].r);
        // Planner-chosen p_r wins over the shape heuristic when it exists. The
        // heuristic (sqrt(m/clusterQuantum) floored at clusterFloor) is a guess at
        // the same quantity the joint argmin computes; using both would be two
        // answers to one question, and the argmin is the one bound by (A5).
        int C = (gridPr > 0)
              ? std::min(16, std::max(1, gridPr))
              : std::min(16, std::max(clusterFloor, (int)std::ceil(
                    std::sqrt(double(m) / double(clusterQuantum)))));
        int rpb = 32 * cdiv(cdiv(mkLane, C), 32);
        if (rpb > (int)dsmRpbMax) return 0;      // smem was configured for dsmRpbMax
        C = cdiv(mkLane, rpb);
        if (C < 1 || C > 16) return 0;
        *Cout = C; *rpbOut = rpb;
        return C * cdiv(C + dsmU, C);
    }
    static bool pipe25d() {
        static const bool on = [] {
            const char* e = getenv("LRQR_PIPE25D"); return e && atoi(e) != 0;
        }();
        return on;
    }
    long long mv_pipe_released = 0, mv_pipe_lanes = 0;  // lanes freed early / total
    // Q_lambda, split by boundary. mv_words_upd is rung 3 (L2/HBM): the far
    // updates that stream the trailing matrix. mv_words_near is rung 2 (DSM/L2):
    // the near updates inside the L2-resident super-panel column block. Keeping
    // them apart is required, not cosmetic -- summing them reports rung 2's
    // traffic against rung 3's budget and manufactures a 14.6x optimality gap.
    double mv_words_upd = 0.0, mv_words_near = 0.0;
    // Which rungs are INSTRUMENTED at all. Without this a zero is ambiguous between
    // "the move does not happen here" and "nobody counted it" -- the absent-vs-zero
    // confusion that has bitten this project before. Rungs 2/3 carry no replica lane
    // and no reduction tree in this build: the far update is a chunked GEMM sequence,
    // so their zeros are STRUCTURAL, not missing measurement.
    // Which rungs the four-move witness actually observes. Rung 3 is now 1: on
    // inspection its carriers were NOT missing, only uncounted -- the far pipeline
    // already splits the frontier into stripes, runs them against a second saved
    // copy of the compact transform state (sav_*/VmSav/TmMSav at parity fslot), and
    // overlaps them on s1 with the next panel's near work on s0, which is Split,
    // Replicate and Pipeline of 31:429's L2/HBM row. COMBINE at rung 3 is still
    // uncounted, so this stays honest rather than rounding up to [1,1,1,1].
    // Rung 2 starts at 0 and is RAISED AT RUNTIME by panel_25d when the lanes run on
    // the DSM carrier, because that is the only configuration in which a rung-2
    // carrier exists at all. A lane on the gmem domino is a single boundary and
    // belongs to rung 1; a lane that is a hardware cluster nests two, and the outer
    // one -- the lanes' compact R slots in Pbuf plus the device-scope merge over
    // them -- is rung 2's row of 31:429. So this initializer is a floor, and the
    // emitted value says which carriers actually ran for that cell.
    int mv_instrumented[4] = {1, 1, 0, 1};

    // Rung 0 moves, read off the domino dispatch geometry: the lane's rows are split
    // into nslab slabs, row state is register double-buffered (sched.rung[0].c), and
    // the B2a phase reduces boundary partials across ALL nslab slabs once per
    // minipanel -- so Combine is (nslab-way reduction) x (NB/PW minipanels).
    void noteRung0(int nslab) {
        if (nslab <= 0) return;
        mv_split[0]     += nslab;
        mv_replicate[0] += std::max(1, sched.rung[0].c);
        mv_combine[0]   += (long long)nslab * std::max(1, NB / PW);
    }
    int tel_far_chunks = 0;   // far-region applyQt calls (lookahead slice + far-rest chunks)
    int tel_split_path = 0;   // far calls that took the 3×TF32 Ozaki split (applyQt 6202 branch)
    int last_fch = 0;         // far-rest chunk size last used (0 = no chunked far-rest issued)
    int2* d_nodes = nullptr;
    struct PTab { int N; int nodesOff; std::vector<std::pair<int,int>> lv; };
    std::vector<PTab> tab;

    static int cdiv(int a, int b) { return (a + b - 1) / b; }
    static constexpr int TPB = 256;

    static int fanoutDepth(int states, int fanout) {
        int depth = 0;
        states = std::max(1, states);
        fanout = std::max(2, fanout);
        while (states > 1) {
            states = cdiv(states, fanout);
            ++depth;
        }
        return depth;
    }

    // Capacity inputs for the four local rungs currently materialized here.
    // M-hat is read from the active device: opt-in shared; that capacity across
    // the live carrier cluster; persisting L2; and the same cache on its
    // HBM-facing side. Integer floor/ceil is explicit. The saturated HBM/NVLink
    // rung is added separately when the full G1 ladder is materialized.
    void computeSchedule() {
        int sharedOpt = 0, l2Persist = 0, regsPerSm = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sharedOpt,
                   cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(&l2Persist,
                   cudaDevAttrMaxPersistingL2CacheSize, 0));
        CUDA_CHECK(cudaDeviceGetAttribute(&regsPerSm,
                   cudaDevAttrMaxRegistersPerMultiprocessor, 0));
        const size_t capBytes[4] = {
            (size_t)sharedOpt,
            (size_t)sharedOpt * (size_t)megaCluster,
            (size_t)l2Persist,
            (size_t)l2Persist
        };
        const size_t leafWords = std::max<size_t>(1, capBytes[0] / sizeof(Real));
        const int bCapacity = std::max(1, (int)std::floor(std::sqrt((double)leafWords)));
        const size_t payloadWords = size_t(bCapacity) * bCapacity;
        const size_t payloadBytes = payloadWords * sizeof(Real);
        constexpr int b_spec = (sizeof(Real) == 8) ? 169 : 240;
        sched.NB = NB;
        sched.b = bCapacity;
        sched.n = n;
        const int nu_spec[4]  = { b_spec, 683, 2500, n };
        const int r_spec_arr[4] = { 2, 33, 438, 438 };
        const int c_spec_arr[4] = { 2, 32, 2, 3 };
        for (int i = 0; i < 4; ++i) {
            const size_t words = std::max<size_t>(1, capBytes[i] / sizeof(Real));
            const int nu = std::max(1, (int)std::floor(std::sqrt((double)words)));
            const int phases = std::max(1, cdiv(n, nu));
            const size_t rawR = (2 * words) / payloadWords;
            const size_t rawW = words / (2 * payloadWords);
            const size_t rawC = words / payloadWords;
            int replicas;
            if (i == 0) {
                const size_t regBytes = size_t(regsPerSm) * sizeof(unsigned);
                replicas = std::min(2, 1 + (int)(regBytes / std::max<size_t>(1, payloadBytes)));
            } else if (i == 1) {
                replicas = std::min(32, std::max(1, (int)rawC));
            } else if (i == 2) {
                replicas = std::min(2, std::max(1, (int)rawC));
            } else {
                replicas = std::min(3, std::max(1, (int)rawC));
            }
            sched.rung[i].capacity_bytes = capBytes[i];
            sched.rung[i].nu      = nu;
            sched.rung[i].S       = phases;
            sched.rung[i].W       = std::max(1, std::min(phases, (int)std::min<size_t>(rawW, 1u << 30)));
            sched.rung[i].r       = std::max(2, (int)std::min<size_t>(rawR, 1u << 30));
            sched.rung[i].c       = replicas;
            sched.rung[i].nu_spec = nu_spec[i];
            sched.rung[i].S_spec  = cdiv(n, nu_spec[i]);
            sched.rung[i].W_spec  = (i == 0) ? 1 : (i == 1 ? 16 : (i == 2 ? 109 : 15));
            sched.rung[i].r_spec  = r_spec_arr[i];
            sched.rung[i].c_spec  = c_spec_arr[i];
            // PHASE 2: nu*, the width the proof actually prescribes. docs/31:262
            //     chi   = sqrt(B(c)/D)                  (+inf when P == 1)
            //     nu*   = min{ sqrt(M), B(c)/n, chi }
            //     B(c)  = I + F/(P sqrt(M(c))) + Q_setup(c)          [docs/31:221]
            // B is a WORD COUNT from a formula, so nu* needs no measurement. On one
            // device P=1 at this boundary, and compact-state on-chip replication
            // "does not claim the full-operand 1/sqrt c volume reduction"
            // (docs/31:234), so M(c)=M and Q_setup=0 -- c does not enter here, which
            // is why c=1 in the ledger is forced by the geometry rather than a defect.
            //
            // MEASURED CONSEQUENCE, recorded so this is not mistaken for a lever:
            // B/n binds only for n <= 1024, and at those sizes capacity_superpanel_B
            // sets B = n (n^2 <= 6.25e6 words), so the far region does not exist and
            // this boundary is never crossed. From n = 2048 up, capacity binds and
            // nu* equals the shipped floor(sqrt(M/w)) exactly. The proof VALIDATES
            // the shipped width everywhere it applies; the recoverable time is in the
            // GEMM-shape term, not the width.
            {
                const double I_lam = (double)m * (double)n;          // A read at least once
                // SHAPE-GENERAL flop count. (4/3)n^3 is the SQUARE case and is wrong
                // for m != n: the QR of an m x n matrix does 2k^2*lng - (2/3)k^3
                // flops with k = min(m,n), lng = max(m,n). At 131072x4096 the square
                // formula gives 9.16e10 against a true 4.35e12 -- 47x low -- so
                // B_lambda was underestimated by 47x and every Q/B ratio on a tall
                // cell was inflated by the same factor. That is what made tall cells
                // read 15-19x "over budget" while squares read 1.9-3.7: the schedule
                // was fine, the budget was wrong. Identical bug class to bench.cu's
                // flops_qr, fixed earlier this session; the schedule carried it too.
                // Reduces to (4/3)n^3 exactly when m == n, so square numbers do not move.
                const double k_lam   = (double)std::min(m, n);
                const double lng_lam = (double)std::max(m, n);
                const double F_lam = 2.0 * k_lam * k_lam * lng_lam
                                   - (2.0 / 3.0) * k_lam * k_lam * k_lam;
                const double sqrtM = (double)std::max(1, nu);        // nu == floor(sqrt(M/w))
                const double B_lam = I_lam + F_lam / sqrtM;          // P_lambda = 1, Q_setup = 0
                const double bn    = B_lam / std::max(1.0, (double)n);
                const int D_lam    = std::max(1, fanoutDepth(phases, std::max(2, (int)rawR)));
                const double chi   = std::sqrt(B_lam / (double)D_lam);
                double ns = std::min((double)nu, bn);
                ns = std::min(ns, chi);
                sched.rung[i].D = D_lam;
                sched.rung[i].B_words = B_lam;
                sched.rung[i].nu_star = std::max(1, (int)std::floor(ns));
                sched.rung[i].nu_star_binding =
                    (bn  <= (double)nu && bn  <= chi) ? 1 :   // panel traffic B/n
                    (chi <= (double)nu)               ? 2 :   // tree depth chi
                                                        0;    // capacity sqrt(M)
            }
        }
    }
    void printSchedule() const {
        const char* name[4] = { "reg→shared(leaf)", "shared→DSM", "DSM↔L2", "L2→HBM" };
        fprintf(stderr, "[TqrSchedule] n=%d NB=%d b_capacity=%d B=%d cLmax=%d fp%d\n",
                sched.n, sched.NB, sched.b, B, cLmax, (int)(sizeof(Real) * 8));
        fprintf(stderr, "  %-18s %9s %6s %6s %7s %-9s | %6s %6s | %4s %4s | %4s %4s | %4s %4s\n",
                "rung", "MhatKiB", "nu", "nu_sp", "nu*", "binds", "S", "S_sp", "W", "W_sp", "r", "r_sp", "c", "c_sp");
        for (int i = 0; i < 4; ++i) {
            const auto& rg = sched.rung[i];
            const char* binds = rg.nu_star_binding == 1 ? "B/n"
                              : rg.nu_star_binding == 2 ? "chi" : "sqrt(M)";
            fprintf(stderr, "  %-18s %9zu %6d %6d %7d %-9s | %6d %6d | %4d %4d | %4d %4d | %4d %4d\n",
                    name[i], rg.capacity_bytes / 1024, rg.nu, rg.nu_spec,
                    rg.nu_star, binds, rg.S, rg.S_spec,
                    rg.W, rg.W_spec, rg.r, rg.r_spec, rg.c, rg.c_spec);
        }
        fprintf(stderr, "  leaf tile NB=%d; schedule columns are measured-capacity formulas\n",
                sched.NB);
    }

    void create(int m_, int n_, int B_, bool tf32_, bool la_,
                lrqr_opts_t opts = lrqr_opts_t{}) {
        m = m_; n = n_; B = B_; tf32far = tf32_; lookahead = la_;
        // W6-P3 (G6 hygiene): validation_copy is now an explicit API parameter
        // (lrqr_opts_t). saveForOrgqr is derived from opts.validation_copy, with
        // the LRQR_ONLY env var retained ONLY as a temporary production override
        // that forces validation_copy=false (the G7(b) OOM-recession prod toggle).
        // When LRQR_ONLY is unset, the API param alone controls the behavior, so
        // callers that pass opts explicitly are env-read-free (AUDIT_V9 §G6: the
        // "LRQR_ONLY 0→10 getenv reads" become an API parameter). Default = true
        // for parity with the historical LRQR_ONLY-unset bench path.
        saveForOrgqr = opts.validation_copy && !getenv("LRQR_ONLY");
        // WMMA tree-apply: validated (Lever A) but measured NO end-to-end speedup (the swapped
        // matmul is negligible vs the panel+recon pipeline) and it adds ~100× orth error at
        // small n → never selected (honest parameter→0, AUDIT_V6 §6 / Theorem U: the env gate
        // LRQR_WMMA is removed; the mechanism stays as dead code). See ROADMAP_ROOFLINE.md.
        { int w = 0;  // capacity-dispatched or removed: never beneficial → parameter→0
          CUDA_CHECK(cudaMemcpyToSymbol(g_lrqr_tree_wmma, &w, sizeof(int))); }
        // A2 PROBE (uncommitted): m >= n is dropped from the assert. The
        // factorization has min(m,n) reflectors, so the panel walk -- which
        // anchors panel k at row0 = k*NB on the diagonal -- must stop at
        // kmax = min(m,n); columns [kmax, n) are updated by the far apply but
        // never factored, leaving R upper TRAPEZOIDAL. n % NB == 0 is retained
        // for now: padding n up to a multiple of NB can push n_pad above m,
        // which is this same wide case, so m<n has to work first.
        assert(n % NB == 0 && B % NB == 0);
        kmax = std::min(m, n);
        npanels = cdiv(kmax, NB);
        Nmax = cdiv(m, SLAB);
        mpadMax = Nmax * SLAB;
        int lo, hi; CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s0, cudaStreamNonBlocking, hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&s1, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sP, cudaStreamNonBlocking, hi));
        const size_t wsbytes = size_t(64) << 20;
        CUBLAS_CHECK(cublasCreate(&cbT));
        CUBLAS_CHECK(cublasSetMathMode(cbT, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
        CUDA_CHECK(cudaMalloc(&cbwsT, wsbytes));
        footprintBytes += wsbytes;
        CUBLAS_CHECK(cublasSetWorkspace(cbT, cbwsT, wsbytes));
        CUBLAS_CHECK(cublasSetStream(cbT, sP));
        CUDA_CHECK(cudaEventCreateWithFlags(&evTsp, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sR, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sN, cudaStreamNonBlocking, lo));
        CUDA_CHECK(cudaEventCreateWithFlags(&evRst, cudaEventDisableTiming));
        // cbT is the Tsp-compound handle (stream sP), used by the proven-working
        // default/STAGE-1 path (`!superOK` loop below) — that path's coop kernel
        // drains per-panel, giving cuBLAS's host-blocking cold dispatch (see
        // k_custom_apply's comment) enough gaps to land without contention.
        // STAGE-2's own Tsp-compound and rest-slice (streams sP/sR) use
        // k_custom_gemm/k_custom_apply instead — no handle/workspace needed there.
        // cb0/cb1 avoid cuBLAS's lazy default-workspace path (cudaMallocAsync,
        // documented unsafe around cooperative kernels) via explicit workspaces.
        CUBLAS_CHECK(cublasCreate(&cb0));
        CUBLAS_CHECK(cublasSetMathMode(cb0, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
        CUDA_CHECK(cudaMalloc(&cbws0, wsbytes));
        footprintBytes += wsbytes;
        CUBLAS_CHECK(cublasSetWorkspace(cb0, cbws0, wsbytes));
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        CUBLAS_CHECK(cublasCreate(&cb1));
        CUBLAS_CHECK(cublasSetMathMode(cb1, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
        CUDA_CHECK(cudaMalloc(&cbws1, wsbytes));
        footprintBytes += wsbytes;
        CUBLAS_CHECK(cublasSetWorkspace(cb1, cbws1, wsbytes));
        CUBLAS_CHECK(cublasSetStream(cb1, s1));
        // Set SM_COUNT_TARGET on s1's cuBLAS handle so far-rest GEMMs leave SMs
        // free for the concurrent panel kernel on s0 (spec §11 look-ahead overlap).
        if (const char* e = getenv("LRQR_CUBLAS_SM")) {
            int sm = atoi(e);
            CUBLAS_CHECK(cublasSetSmCountTarget(cb1, sm));
            fprintf(stderr, "[LRQR] cublasSetSmCountTarget(cb1, %d)\n", sm);
        }
        for (int p = 0; p < 3; ++p)
            CUDA_CHECK(cudaEventCreateWithFlags(&evFar[p], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evSlice, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evIn, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPD, cudaEventDisableTiming));
        for (int p = 0; p < 2; ++p)
            CUDA_CHECK(cudaEventCreateWithFlags(&evRest[p], cudaEventDisableTiming));
        // F8 Step-1 stage timer events (timing ENABLED — default flags, not DisableTiming)
        staget = getenv("LRQR_STAGET") && atoi(getenv("LRQR_STAGET"));
        if (staget) {
            CUDA_CHECK(cudaEventCreateWithFlags(&evSt0a, 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&evSt0b, 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&evSt1a, 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&evSt1b, 0));
        }
        // W6(G7): ralloc accumulates into footprintBytes for the G7 telemetry gate.
        // (also tallies the small non-Real cudaMallocs below via footprintBytes += ...).
        auto ralloc = [&](Real** p, size_t elems) {
            CUDA_CHECK(cudaMalloc(p, elems * sizeof(Real)));
            footprintBytes += elems * sizeof(Real);
        };
        auto ralloc64 = [&](double** p, size_t elems) {
            CUDA_CHECK(cudaMalloc(p, elems * sizeof(double)));
            footprintBytes += elems * sizeof(double);
        };
        ralloc(&Pbuf, size_t(mpadMax) * NB);
        ralloc(&QT,   size_t(mpadMax) * NB);
        ralloc(&QS,   size_t(Nmax) * NB * NB);
        // W6(G7): WS (Nmax*NB*NB) removed — dead allocation (declared/alloc'd/freed but
        // never read or written by any kernel; verified by grep \bWS\b across cuda/).
        // First concrete ELIMINATE-list win from the Stage-R memo (RESEARCH_MEMO §4.1).
        ralloc(&Vm,   size_t(Nmax) * NB * NB);
        ralloc(&TmM,  size_t(Nmax) * NB * NB);
        ralloc(&TmL,  size_t(Nmax) * NB * NB);
        ralloc(&tauL, size_t(Nmax) * NB);
        ralloc(&taum, size_t(Nmax) * NB);
        // W6(G7) Phase-2: Wax/Wbx were over-allocated at B·max(n,NB) (≈1 GiB each
        // at fp64@131072², 6 GiB total). Actual peak element count across ALL
        // consumers (verified at every applyQt + mergeApplyQt + orgqr call site):
        //   orgqr           : W1 = NB×n  (ld=NB) → n·NB elems   (bench path only)
        //   applyQt far-rest: W1 = B×fch (ld=B)  → B·fch elems  (fch = far chunk)
        //   applyQt slice   : W1 = B×wnext (ld=B) → B·B elems   (lookahead: wnext≤B)
        //   applyQt near/rest: W1 = NB×nc (ld=NB) → ≤ NB·NB elems
        //   mergeApplyQt    : W1 = NB×nc (ld=NB) → ≤ NB·fch elems
        // Shrink to max(n·NB, B·fchCap, B·B). This is still O(n·b) (spec §3 O2 STAY),
        // just the true constant. Safe for bench AND prod when lookahead=true (wnext≤B).
        // --no-la keeps B·n (slice wnext=wfar≤n needs the full B·n). RESEARCH_MEMO §4.2.
        const bool is3x = std::is_same<Real, float>::value && getenv("LRQR_3XTF32");
        size_t fchCap;
        if (const char* e = getenv("LRQR_FARCHUNK")) {
            fchCap = size_t(std::max(1, atoi(e)));
        } else {
            // Workspace cap = the same far-chunk law the far-rest sites use
            // (farChunkCore; use3xtf32 isn't set yet in create() → pass is3x).
            fchCap = size_t(farChunkCore(is3x, B, n, m));
        }
        fchCap = std::min(size_t(n), fchCap);
        const size_t waxElems = lookahead
            ? std::max({size_t(n) * NB, size_t(B) * fchCap, size_t(B) * size_t(B)})
            : size_t(B) * std::max(n, NB);
        waxElemsCached = waxElems;
        CUDA_CHECK(cudaMalloc(&absmaxPart, kAbsmaxBlocks * sizeof(double)));
        for (int p = 0; p < 3; ++p) {
            ralloc(&SPVx[p], size_t(m) * B);
            ralloc(&Tspx[p], size_t(B) * B);
            ralloc(&Wax[p],  waxElems);
            ralloc(&Wbx[p],  waxElems);
            // W6(G7) Phase-2: Ytx (pre-transposed Y for TN GEMM3, 3·B·m·w ≈ 3 GiB
            // at fp64@131072²) is skipped in the LRQR_ONLY production path. The
            // far-update GEMM3 then takes the existing Yt==nullptr NN fallback
            // (lrqr.cuh:6479: C -= Y·W2 reads Y directly from SPV). TN is ~1.5×
            // faster for GEMM3 (comment :6476) but GEMM3 runs on s1 (far-rest, the
            // non-critical stream) — the s0 panel stream is unaffected. The ≤3%
            // speed gate (G7d) is measured on the BENCH path where saveForOrgqr
            // is true and Ytx/TN stays; production (LRQR_ONLY) trades GEMM3 speed
            // for the footprint needed to clear the G7(b) OOM line. Combined with
            // the per-panel-save skip + Wax/Wbx shrink, this removes ~10 GiB at
            // fp64@131072² (141.34→~131 GiB, well under 140.4 GiB H200 FB total).
            if (saveForOrgqr)
                ralloc(&Ytx[p], size_t(B) * m);
        }
        Yt = Ytx[0];  // alias for compatibility (nullptr in LRQR_ONLY)
        ralloc(&Tpan, size_t(npanels) * NB * NB);
        ralloc(&Sg,   size_t(npanels) * NB);
        ralloc(&Zb,   size_t(B) * NB);
        ralloc(&M1b,  size_t(B) * NB);
        ralloc(&Wn1,  size_t(NB) * B);
        ralloc(&Wn2,  size_t(NB) * B);
        ralloc(&Wr1,  size_t(NB) * B);
        ralloc(&Wr2,  size_t(NB) * B);
        ralloc(&domScal, size_t(4) * NB);
        ralloc(&domSlot, 16 * 16);
        ralloc(&domProw, 2 * 16);
        ralloc(&domPp,   size_t(std::max(132, cdiv(m, (int)clRpbMaxDB))) * NB * 16);
        ralloc(&domPsum, size_t(2) * NB * 16 * (NB / 16));  // ×2: super-DSM parity
        ralloc(&domT16,  size_t(2) * 16 * 16 * (NB / 16));
        ralloc64(&domSigD, NB);          // fp64 shadow of sig (domScal+3*NB)
        ralloc64(&domDslotD, 16 * 16);   // fp64 shadow of dslot (domSlot)
        // A6 (31:246) deterministic W: k_panel_domino now reduces the per-column dots
        // and sigma in fixed block order out of this buffer instead of by fp64
        // atomicAdd. Sized for the widest grid this build launches (one row of PW+1
        // slots per block, two column-parity generations).
        ralloc64(&domDcolP, size_t(2) * 132 * 17);
        ralloc64(&domDcolR, size_t(2) * 17);
        CUDA_CHECK(cudaMalloc(&domT, 8 * sizeof(unsigned long long)));
        footprintBytes += 8 * sizeof(unsigned long long);
        CUDA_CHECK(cudaMemset(domT, 0, 8 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&pipeF, 512 * sizeof(int)));
        footprintBytes += 512 * sizeof(int);
        CUDA_CHECK(cudaMalloc(&gbar, 2 * sizeof(unsigned)));  // gmem barrier: count + generation
        footprintBytes += 2 * sizeof(unsigned);
        CUDA_CHECK(cudaMemset(gbar, 0, 2 * sizeof(unsigned)));
        // Per-lane domino buffers for concurrent c_L>1 execution
        for (int j = 0; j < cLmax; ++j) {
            ralloc(&domScalL[j], size_t(4) * NB);
            ralloc(&domSlotL[j], 16 * 16);
            ralloc(&domProwL[j], 2 * 16);
            ralloc(&domPpL[j],   size_t(std::max(132, cdiv(m, (int)clRpbMaxDB))) * NB * 16);
            ralloc(&domPsumL[j], size_t(2) * NB * 16 * (NB / 16));
            ralloc(&domT16L[j],  size_t(2) * 16 * 16 * (NB / 16));
            ralloc(&laneW1[j], size_t(NB) * NB);
            ralloc(&laneW2[j], size_t(NB) * NB);
            ralloc64(&domSigDL[j], NB);          // per-lane fp64 shadow of sig
            ralloc64(&domDslotDL[j], 16 * 16);   // per-lane fp64 shadow of dslot
            CUDA_CHECK(cudaMalloc(&gbarL[j], 2 * sizeof(unsigned)));
            footprintBytes += 2 * sizeof(unsigned);
            CUDA_CHECK(cudaMemset(gbarL[j], 0, 2 * sizeof(unsigned)));
            CUDA_CHECK(cudaEventCreateWithFlags(&evDom[j], cudaEventDisableTiming));
        }
        for (int j = 0; j < cLmax; ++j)
            CUDA_CHECK(cudaStreamCreateWithFlags(&domSt[j], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPrePanel, cudaEventDisableTiming));
        // ---- resident multi-block kernel scratch (spec §9/§11) ----
        ralloc(&rVg, size_t(2) * m * NB);                    // double-buffered V in gmem
        ralloc(&rTg, size_t(2) * NB * NB);                   // double-buffered T in gmem
        for (int l = 0; l < cLmax; ++l)
            CUDA_CHECK(cudaEventCreateWithFlags(&evM2lvl[l], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evS1done, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc((void**)&rPanelGen, 2 * sizeof(unsigned)));
        footprintBytes += 2 * sizeof(unsigned);
        CUDA_CHECK(cudaMemset((void*)rPanelGen, 0, 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaMalloc((void**)&rUpdateCnt, 2 * sizeof(unsigned)));
        footprintBytes += 2 * sizeof(unsigned);
        CUDA_CHECK(cudaMemset((void*)rUpdateCnt, 0, 2 * sizeof(unsigned)));
        // ---- c_L-lane per-layer buffers (spec §7.2) ----
        // Only domTpanL is lane-private (each lane's compound T differs). The domino
        // itself reuses the plan's domScal/domSlot/domPp/... buffers (run sequentially
        // per lane in the fork-join path; A1's resident kernel will run them concurrently
        // as consumer warp-groups with DSM-local buffers — A3's ≤96-reg panel removes the
        // gmem Pp/Psum path that caused Agent 8's per-layer buffer bug).
        ralloc(&domTpanL, size_t(cLmax) * NB * NB);
        // Deferred-far save buffers (double-buffered, same sizes as originals)
        for (int s = 0; s < 2; ++s) {
            ralloc(&VmSav[s],       size_t(Nmax) * NB * NB);
            ralloc(&TmMSav[s],      size_t(Nmax) * NB * NB);
            ralloc(&domTpanLSav[s], size_t(cLmax) * NB * NB);
            CUDA_CHECK(cudaEventCreateWithFlags(&evMergeCpy[s], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evFar25[s],    cudaEventDisableTiming));
        }
        // Per-panel merge data save for orgqr (cL>1 Q reconstruction).
        // W6(G7) Phase-2: skip when orgqr will not be called (LRQR_ONLY production
        // path) — eliminates 3·npanels·cLmax·NB²·w of device footprint + the
        // per-panel D2D save copies. saveForOrgqr is set at top of create().
        if (saveForOrgqr) {
            const size_t perPan = size_t(cLmax) * NB * NB;
            ralloc(&domTpanL_pan, size_t(npanels) * perPan);
            ralloc(&Vm_pan,       size_t(npanels) * perPan);
            ralloc(&TmM_pan,      size_t(npanels) * perPan);
            CUDA_CHECK(cudaMallocHost(&cL_pan, npanels * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&nm_pan, npanels * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&slabStart_pan, npanels * cLmax * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&slabSize_pan, npanels * cLmax * sizeof(int)));
            CUDA_CHECK(cudaMallocHost(&nodes_pan, npanels * cLmax * sizeof(int2)));
            memset(cL_pan, 0, npanels * sizeof(int));
            memset(nm_pan, 0, npanels * sizeof(int));
        }
        {
            // One row per possible c_L, uploaded once. This used to be a single
            // list for cLmax that panel_25d then OVERWROTE on every panel with a
            // blocking cudaMemcpy -- so the upload here was dead and the panel path
            // paid a host-device round trip per panel to reproduce a table that
            // never changes. At 16384^2 that is 256 pipeline drains per
            // factorization, charged to the merge, which is the term deciding
            // whether replication is worth doing at all.
            std::vector<int2> tab((size_t)(cLmax + 1) * kNodeStride, int2{0, 0});
            for (int c = 0; c <= cLmax; ++c) {
                size_t w = (size_t)c * kNodeStride;
                for (int step = 1; step < c; step *= 2)
                    for (int i = 0; i + step < c; i += 2 * step)
                        tab[w++] = {i, i + step};
            }
            nnodes2d = (int)tab.size();
            CUDA_CHECK(cudaMalloc(&d_nodes2d, tab.size() * sizeof(int2)));
            footprintBytes += tab.size() * sizeof(int2);
            CUDA_CHECK(cudaMemcpy(d_nodes2d, tab.data(), tab.size() * sizeof(int2),
                                  cudaMemcpyHostToDevice));
        }
        if (const char* e = getenv("LRQR_PIPENBL")) pipeNbl = atoi(e);
        pipeNbl = std::max(2, std::min(pipeNbl, 132));
        // V6 W1 (spec §10 Theorem U): domino+pipe are the §5 Pipeline — always
        // live. The OFF escape hatches (LRQR_NO_DOMINO/NO_PIPE) are removed;
        // pipeNbl (4944) and pipeMax (5666) remain as the continuous selectors.
        useDomino = true;
        use25d = useDomino;   // panel() engages it when the live formula selects c_L>1
        usePipe = useDomino;
        useDefuse = false;  // env gate LRQR_DEFUSE removed (AUDIT_V6 §6 / Theorem U);
                            // de-fused domino never beneficial on fork-join → parameter→0.
                            // The mechanism stays as dead code; never selected.
        // T4 Cor.FC (spec §6): fuse the Tsp-compound (⊕-hop) onto the panel stream s0,
        // eliminating the standalone sP/cbT stream+handle. V6 W1: the fused ⊕-hop is
        // the headline Combine mechanism — always live (LRQR_CORFC opt-in removed).
        // The far-rest on s1 reads Tsp via evSlice (s0→s1), which transitively includes Tsp
        // after fusion (Tsp is in-stream on s0 before the far-update that precedes evSlice).
        corFC = true;
        // W3''' step 3: capacity-ladder dispatch (spec §10 Theorem U). Replaces the
        // former compile-time engagement flags and the step-1/2 bring-up env overrides
        // for the carrier — all four symbols are DEAD (grep-proof verifier gate). The
        // ladder measures on-device capacity and decides carrier engagement from:
        //   (1) smem carveout: carrier smem ≤ 196 KiB cliff (TMA/wgmma starve above)
        //   (2) SM count: ≥ megaCluster (8) for single-cluster residency
        //   (3) n-divisibility: panel stream requires n % NB == 0
        //   (4) speed gate: the carrier engages only when beneficial at (m,n). The
        //       single-cluster carrier (8 SMs, ~92-111 TF @ n=65536, W3STEP2_MEMO §3)
        //       is 0.01× cuSOLVER (224 TF) — NOT beneficial. Multi-cluster (step 3
        //       priority 3) lifts this. Until then the ladder falls back to fork-join:
        //       the mechanism is live, the engagement threshold is honest about current
        //       carrier speed. Default path = fork-join = byte-identical to pre-step-3
        //       (no sentinel regression: validate-all 69/71, TF32@65536 ~222 TF).
        if (const char* e = getenv("LRQR_RGRID")) rGrid = atoi(e);
        {
            const bool isFloat = std::is_same_v<Real, float>;
            const bool nDiv = (n % NB == 0);
            int dev = 0, smCount = 0, smemPerBlockOptin = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
            cudaDeviceGetAttribute(&smemPerBlockOptin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            // Carrier capacity: smem ≤196 KiB cliff, SMs ≥ cluster, n%NB==0, float+NB=128.
            const bool capOk = isFloat && (NB == 128) && nDiv &&
                               (smCount >= Plan::megaCluster) &&
                               (smemPerBlockOptin >= (int)(196 * 1024));
            // Speed gate (step 4 — carrier-hosted engagement by formula). The carrier
            // engages only when its estimated TF exceeds the fork-join TF, computed from
            // MEASURED capacity (SM count + cluster span), not a hardcoded constant.
            // Model: carrier_TF ≈ (engaged_SMs / total_SMs) · forkjoin_TF · wgmma_gain,
            //   where engaged_SMs = megaCluster (single-cluster) and wgmma_gain ≈ 5× is
            //   the wgmma m64n64k8 vs scalar-FFMA throughput advantage (measured W3STEP2).
            // Engage iff (engaged_SMs/total_SMs)·wgmma_gain > 1.
            //   single-cluster (8/132 SMs): 0.30 → NOT beneficial (honest, matches W3STEP2
            //     §3: 92-111 TF carrier vs 222 TF fork-join = 0.01× cuSOLVER).
            //   multi-cluster (132/132 SMs): 5.0 → beneficial (the step-3 multi-cluster
            //     resume point flips this; the formula is ready — only the kernel span
            //     changes, not the gate).
            // The c_L latency carrier (2.5D domino, panel_25d) is a separate
            // capacity/depth-derived path and does not depend on this carrier model.
            constexpr float wgmma_gain = 5.0f;
            const float carrier_frac = (float)Plan::megaCluster / (float)smCount;
            const bool speedOk = (carrier_frac * wgmma_gain) > 1.0f;
            useResident = false;   // scalar FFMA skeleton (0.02 TF) — never beneficial
            useMega = false;       // scalar mega skeleton (0.02 TF) — never beneficial
            useMegaWgmma = capOk && speedOk;
            (void)m;
        }
        if (useResident) {
            residentMmax = m;  // V in gmem, no smem limit on m
            constexpr int RN = ResidentSmem<Real, NB>::TILE_N;
            constexpr int RC = ResidentSmem<Real, NB>::CHUNK_M;
            size_t rs = ResidentSmem<Real, NB>::bytes;
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)k_tqr_resident<Real, NB, RN, RC>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, rs));
            fprintf(stderr, "[LRQR] resident kernel: NB=%d TILE_N=%d CHUNK_M=%d smem=%zuKB grid=%d\n",
                    NB, RN, RC, rs/1024, rGrid);
        }
        if (useMega) {
            constexpr int MN = MegaSmem<Real, NB>::TILE_N;
            constexpr int MC = MegaSmem<Real, NB>::CHUNK_M;
            size_t ms = MegaSmem<Real, NB>::bytes;
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)k_tqr_mega<Real, NB, MN, MC, Plan::megaCluster>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, ms));
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)k_tqr_mega<Real, NB, MN, MC, Plan::megaCluster>,
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            fprintf(stderr, "[LRQR] T5 megakernel: NB=%d TILE_N=%d CHUNK_M=%d smem=%zuKB cluster=%d\n",
                    NB, MN, MC, ms/1024, Plan::megaCluster);
        }
        // wgmma carrier (W3' sub-unit a, float+NB=128): B32-swizzle TF32 consumer.
        // Engaged by the capacity ladder above (smem/SM/nDiv + speed gate). smem ≤196 KiB.
        if constexpr (std::is_same_v<Real, float> && NB == 128) {
            if (useMegaWgmma) {
                constexpr int WN = 64;   // TILE_N for wgmma variant
                size_t ws = wgmma_carrier_smem();
                static_assert(MegaSmemWgmma<Real, NB, WN>::bytes <= 196 * 1024,
                              "MegaSmemWgmma exceeds 196 KiB carveout cliff");
                if (!wgmma_carrier_setattr()) {
                    fprintf(stderr, "[LRQR] W3' wgmma_carrier_setattr FAILED — disabling\n");
                    useMegaWgmma = false;
                } else {
                    // Host-mapped watchdog signals (zero-copy: device writes, host polls).
                    // Plain mapped (NOT WriteCombined): WC is for host-write/device-read; here
                    // the device writes and the host reads, so uncacheable-mapped is correct.
                    CUDA_CHECK(cudaHostAlloc((void**)&hProgK, sizeof(int),
                        cudaHostAllocMapped));
                    CUDA_CHECK(cudaHostGetDevicePointer((void**)&dProgK, (void*)hProgK, 0));
                    CUDA_CHECK(cudaHostAlloc((void**)&hUpdPhase, sizeof(int),
                        cudaHostAllocMapped));
                    CUDA_CHECK(cudaHostGetDevicePointer((void**)&dUpdPhase, (void*)hUpdPhase, 0));
                    fprintf(stderr, "[LRQR] W3' wgmma mega: NB=%d TILE_N=%d smem=%zuKB cluster=%d wd=%d\n",
                            NB, WN, ws/1024, Plan::megaCluster, megaWd);
                }
            }
        }
        // DSM col-cluster + in-kernel updaters (ROADMAP §6): measured to beat the
        // fused champion for square/wide mid-n (4096–32768, m≤3n) where mk fits a
        // ≤16-block cluster; fused wins at 65536 (top panels exceed the cluster cap)
        // and tall panels. Auto-select that regime; LRQR_DSM forces on, LRQR_NO_DSM off.
        // V6 W1 (spec §10 Theorem U): DSM cluster + super-domino are always live.
        // The n>=3072/m<=40000/m<=3*n size box and the LRQR_DSM/LRQR_NO_DSM/
        // LRQR_SUPER opt-in/out hatches are removed. The continuous capacity test
        // at :5669 (mk <= dsmRowMax) dispatches DSM per-panel; the test at :6309
        // (mk <= 16*superRpbMax) dispatches super. dsmU (5006) stays as the
        // ladder-derived cluster-U parameter.
        useDsm = useDomino;
        // V6 W1 (spec §10 Theorem U): super-domino is LIVE-BY-DEFAULT. The
        // previous LRQR_SUPER env gate is removed. The 92% regression was
        // caused NOT by the cluster launch (already non-cooperative via
        // cudaLaunchKernelEx + cluster-dim attr) but by the host-launched
        // k_custom_apply rest slices on sR: k_super_dsm busy-spun (158ms/call)
        // on restDone flags set by those slow, host-serialized slices. Re-route:
        // the rest columns [nb2,Bcur) are now INTERNALIZED into d_super_upd's
        // update loop (same minipanel-by-minipanel WY apply, extended column
        // range to ncTotal=(npan-k)*NB), and restDone[k] is set IN-KERNEL after
        // the donePan barrier — eliminating the host rest slices entirely so the
        // cluster kernel never stalls. The minipanel-by-minipanel apply is
        // mathematically identical to the compound-T apply it replaces. The
        // continuous capacity test at :6562 (mk <= 16*superRpbMax) dispatches
        // super per-panel; panels exceeding the cluster smem cap fall back.
        useSuper = useDomino;
        if (useSuper) useDsm = true;
        if (const char* e = getenv("LRQR_DSMU")) dsmU = std::max(1, atoi(e));
        // fp64: the d_dsm_region warp/rank mapping bug (warp<C guard dropped ranks
        // when NW<C, i.e. fp64 blockDim=256) is fixed via a cooperative strided loop;
        // fp64 now runs the full domino tower (DSM + super). k_panel_domino_pipe and
        // k_panel_domino remain correct fallbacks (relRR ≤1.7e-14).
        ralloc(&domUg, size_t(m) * 16);
        ralloc(&domW2, size_t(NB) * 16);
        ralloc(&domGs, 16 * 16);
        // 3×TF32 scratch allocation (float-only; activated by LRQR_3XTF32 env var set by bench.cu --3xtf32)
        use3xtf32 = std::is_same<Real, float>::value && getenv("LRQR_3XTF32");
        if (use3xtf32) {
            // G6c tier-cap (W4'', see RESID_BASE_CURVE.md): the prior magic 8192 is replaced
            // by a derived 3-way cap. The EC accuracy floor is the measured resid_base curve
            // (slurm-7576, GPU-8ac4d888, reps=7 median): resid_ec(K) = 2.40e-9 * K^0.99 ~= 2.4e-9*K,
            // where K is the GEMM accumulation depth (= mr for the TN far GEMM1). This floor is
            // K-driven, NOT fch-driven, so ACCURACY DOES NOT CAP fch (the N=chunk dimension).
            // At the max operating K=mr<=65536 the floor is <=1.5e-4 (2.2x below the 1xTF32 floor
            // 3.3e-4, and within the 3xTF32 tier fp32-OoM bar -- validate-all PASSES every size).
            // => fch cap = min(memory_cap, determinism_cap, n):
            size_t memFree3x = 0, memTotal3x = 0;
            cudaMemGetInfo(&memFree3x, &memTotal3x);
            // 4 limb buffers (Chi,Clo,W2hi,W2lo). W6-P4 (G7 Phase-3): Chi/Clo are now
            // maxSk3x·fch (in-place per-K-slice re-split, OZAKI_PHASE3_DESIGN §3), not m·fch.
            // W2hi/W2lo remain B·fch. budget = 1/8 of free HBM for the limb buffers.
            int mem_cap = (int)(memFree3x / (8u * 2u * size_t(maxSk3x + B) * sizeof(float)));
            if (mem_cap < 256) mem_cap = 256;            // floor for GEMM efficiency
            const int DET_CAP_3X = 8192;                  // cuBLAS-algo determinism floor (MEASURED,
                                                          // AUDIT_V6 §3 / G6C_CLOSURE.md §4.#1): cuBLAS
                                                          // picks non-deterministic TF32-TC algos for
                                                          // the 3x Ozaki-split GEMMs at N>8192 —
                                                          // between-process r_ratio spread 0.72x..81x
                                                          // (112x); 9/9 launches bit-identical at the
                                                          // cap. NOT an accuracy cap (resid_base is
                                                          // K-driven). Remove via the in-kernel wgmma
                                                          // EC (commit c — a single wgmma kernel has
                                                          // no algo selection).
            maxFarchunk3x = std::min(n, std::min(mem_cap, DET_CAP_3X));
            if (const char* e = getenv("LRQR_3XTF32_CHUNK")) { int v = atoi(e); if (v > 0) maxFarchunk3x = v; }
            if (maxFarchunk3x <= 0) maxFarchunk3x = std::min(n, std::min(mem_cap, DET_CAP_3X));
            if (const char* e = getenv("LRQR_3X_SPLITK")) splitK3x = atoi(e);
            for (int p = 0; p < 3; ++p) {
                ralloc(&Yhi[p],  size_t(m) * B);
                ralloc(&Ylo[p],  size_t(m) * B);
                ralloc(&Ythi[p], size_t(B) * m);
                ralloc(&Ytlo[p], size_t(B) * m);
            }
            // W6-P4 (G7 Phase-3): Chi/Clo shrunk from m·maxFarchunk3x to maxSk3x·maxFarchunk3x
            // (8.0 GiB → 0.125 GiB @131072²). The GEMM1 consumer re-splits C per K-slice into
            // these buffers with ld=maxSk3x (see applyQt 3×TF32 path). 5b1802c bug avoided:
            // BOTH the alloc AND the producer/consumer ld move to maxSk3x together.
            for (int p2 = 0; p2 < 2; ++p2) {   // double-buffered K-slice split (pipeline)
                ralloc(&Chi[p2],  size_t(maxSk3x) * maxFarchunk3x);
                ralloc(&Clo[p2],  size_t(maxSk3x) * maxFarchunk3x);
                CUDA_CHECK(cudaEventCreateWithFlags(&evSp3x[p2], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&evG3x[p2], cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaStreamCreateWithFlags(&s3xsp, cudaStreamNonBlocking));
            ralloc(&W2hi, size_t(B) * maxFarchunk3x);
            ralloc(&W2lo, size_t(B) * maxFarchunk3x);
            fprintf(stderr, "[LRQR] 3xTF32 far tier enabled: far GEMM1+GEMM3 split into "
                    "3 TF32-TC GEMMs each, Y/Yt pre-split per super-panel. "
                    "maxFarchunk=%d splitK=%d\n", maxFarchunk3x, splitK3x);
            // Force DEFAULT_MATH on all handles so CUBLAS_COMPUTE_32F is truly fp32 (not TF32).
            // TF32_TENSOR_OP_MATH would promote COMPUTE_32F→TF32, leaking error into fp32 paths
            // (orgqr Q-reconstruction, Tsp compound, panel TRSM, near/rest applyQt). The 3xTF32
            // split GEMMs in applyQt toggle TF32 on explicitly for the 3 split GEMMs only.
            CUBLAS_CHECK(cublasSetMathMode(cb0, CUBLAS_DEFAULT_MATH));
            CUBLAS_CHECK(cublasSetMathMode(cb1, CUBLAS_DEFAULT_MATH));
            CUBLAS_CHECK(cublasSetMathMode(cbT, CUBLAS_DEFAULT_MATH));
        }
        std::vector<int2> nodes;
        tab.resize(npanels);
        for (int k = 0; k < npanels; ++k) {
            int N = cdiv(m - k * NB, SLAB);
            tab[k].N = N;
            tab[k].nodesOff = (int)nodes.size();
            for (int step = 1; step < N; step *= 2) {
                int off = (int)nodes.size();
                for (int i = 0; i + step < N; i += 2 * step) nodes.push_back({i, i + step});
                tab[k].lv.push_back({off, (int)nodes.size() - off});
            }
        }
        CUDA_CHECK(cudaMalloc(&d_nodes, std::max<size_t>(nodes.size(), 1) * sizeof(int2)));
        footprintBytes += std::max<size_t>(nodes.size(), 1) * sizeof(int2);
        if (!nodes.empty())
            CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(int2),
                                  cudaMemcpyHostToDevice));
        setAttrs();
        {
   // fused cooperative panel: smem = max over phases, grid = residency limit
            fusedSmem = std::max({bqr_smem<Real, NB>(SLAB), merge_smem<Real, NB>(),
                                  smLarft(false), smLarft(true),
                                  3 * NB * NB * (int)sizeof(Real),
                                  ((SLAB + 1) * NB + 2 * NB * NB) * (int)sizeof(Real),
                                  ((NB + 1) * NB + NB) * (int)sizeof(Real),
                                  2 * (NB + 1) * NB * (int)sizeof(Real)});
            CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_fused<Real, SLAB, NB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, fusedSmem));
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
            smCount = prop.multiProcessorCount;
            int perSM = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM,
                k_panel_fused<Real, SLAB, NB>, TPB, fusedSmem));
            fusedGrid = perSM * smCount;
            if (fusedGrid < 1 || !prop.cooperativeLaunch) useFused = false;
            // V6 W1 (spec §10 Theorem U): fused panel live-by-default; the
            // LRQR_NO_FUSED OFF-hatch and the n<=16384 cap are removed. NOTE: the
            // fused path (:5861) is unreachable while useDomino is true (the domino
            // block at :5685 handles every panel and returns), so this is currently
            // inert — kept for the future non-cooperative fused dispatch (Phase C).
            else useFused = (cdiv(m, SLAB) - 1 <= fusedGrid);
        }
        {   // top-of-tree wavefront kernel (per-level path)
            waveTopSmem = ((NB + 1) * NB + (NB + 1) * 16 + NB + (NB / 16 + 3) * 16 * 16)
                          * (int)sizeof(Real);
            cudaError_t e = cudaFuncSetAttribute((const void*)k_wave_top<Real, SLAB, NB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, waveTopSmem);
            cudaDeviceProp pr; CUDA_CHECK(cudaGetDeviceProperties(&pr, 0));
            // V6 W1 (spec §10 Theorem U): tree-top merge always live.
            // LRQR_NO_WAVETOP hatch removed; cooperative-launch is the capacity test.
            waveTopOK = (e == cudaSuccess) && pr.cooperativeLaunch;
        }
        {   // persistent panel pipeline (spec §11): the cooperative single-launch variant
            // (k_panel_persist — one coop grid for all panels + host doorbell) deadlocks on
            // driver 570. Per spec §10 Theorem U the mechanism is re-routed to NON-COOPERATIVE
            // MULTI-LAUNCH: the panel kernel (k_panel_domino via panel()) re-launches per panel
            // on s0 with the gmem barrier, overlapping the far-rest on s1 — the same §11
            // pipeline, live-by-default, no env gate. The cooperative k_panel_persist + host
            // doorbell is removed; usePersist stays false (the cooperative carrier is gone).
            // LRQR_PGRID remains a tuning knob for the (now inert) cooperative grid sizing.
            if (const char* e = getenv("LRQR_PGRID")) pgrid = atoi(e);
            pgrid = std::min(pgrid, fusedGrid > 0 ? fusedGrid : 0);
            usePersist = false;
        }
        // GREEN-ROUTING = spec §9 overlap band-partition (W_band), materialized to its
        // →1 limit here (G1 spec-faithfulness; spec §10 Theorem U + the presence-vs-
        // parameter honesty clause: a ladder parameter smoothly →1 on real capacity,
        // like HBM→NVLink saturating at 1 GPU). The green-context partition splits the
        // SMs into a big partition (panel/latency carrier) + a small partition (trailing
        // bandwidth carrier), overlapping the two rungs. On THIS materialization the band
        // width W_band → 1 (a single unified context, no partition) for TWO spec-predicted
        // reasons, not an exclusion: (1) single-GPU has no cross-context band to route
        // over — the overlap the partition would buy is already achieved by the on-device
        // stream overlap (s0 panel ∥ s1 far) + the resident cluster-domino, so the
        // partition's marginal Combine budget →0; (2) HARDWARE: thread-block clusters
        // (the tower's core Split/Combine substrate) cannot launch inside a green
        // partition — the driver hangs — so green is incompatible with the cluster path
        // at any width. Both drive W_band→1 continuously (a parameter limit, not a gate):
        // the generator is live, its band parameter is at its single-GPU floor. greenInit()
        // is retained for a future non-cluster cols reservation; LRQR_GREEN is not read
        // (G6: no env gate). Multi-GPU would lift W_band>1 via the same formula.
        green = false;
        // cuBLASLt with SM_COUNT_TARGET (opt-in via LRQR_LT_SM env var)
        if (const char* e = getenv("LRQR_LT_SM")) ltSmTarget = atoi(e);
        if (const char* e = getenv("LRQR_SPLITK")) ltSplitK = atoi(e);
        if (ltSmTarget > 0 || use3xtf32) {
            cublasLtCreate(&lt0); cublasLtCreate(&lt1); cublasLtCreate(&ltT);
            int opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
            cublasComputeType_t compute;
            cudaDataType_t dtype;
            if constexpr (std::is_same<Real, double>::value) {
                compute = CUBLAS_COMPUTE_64F;
                dtype = CUDA_R_64F;
            } else {
                compute = tf32_ ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
                dtype = CUDA_R_32F;
            }
            if (ltSmTarget > 0) {
            cublasLtMatmulDescCreate(&ltDescTN, compute, dtype);
            cublasLtMatmulDescSetAttribute(ltDescTN, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescTN, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescTN, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                           &ltSmTarget, sizeof(ltSmTarget));
            cublasLtMatmulDescCreate(&ltDescNN, compute, dtype);
            cublasLtMatmulDescSetAttribute(ltDescNN, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescNN, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescNN, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
                                           &ltSmTarget, sizeof(ltSmTarget));
            }
            cublasLtMatmulDescCreate(&ltDescTN_plain, compute, dtype);
            cublasLtMatmulDescSetAttribute(ltDescTN_plain, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescTN_plain, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int));
            cublasLtMatmulDescCreate(&ltDescNN_plain, compute, dtype);
            cublasLtMatmulDescSetAttribute(ltDescNN_plain, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(int));
            cublasLtMatmulDescSetAttribute(ltDescNN_plain, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int));
            // fp32 descriptor for 3×TF32 GEMM2 (T^T·W1 — fp32 CUDA cores for accuracy)
            if (use3xtf32) {
                cublasLtMatmulDescCreate(&ltDescTN_f32, CUBLAS_COMPUTE_32F, dtype);
                cublasLtMatmulDescSetAttribute(ltDescTN_f32, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(int));
                cublasLtMatmulDescSetAttribute(ltDescTN_f32, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(int));
            }
            if (!ltWorkspace) CUDA_CHECK(cudaMalloc(&ltWorkspace, 128 * 1024 * 1024));
            if (ltWorkspace) footprintBytes += 128 * 1024 * 1024;
            fprintf(stderr, "[LRQR] cuBLASLt enabled SM_COUNT_TARGET=%d splitK=%d 3xTF32=%d\n",
                    ltSmTarget, ltSplitK, (int)use3xtf32);
        }
        // T2: compute the TqrSchedule (ν/S/W/r/c per rung, code + spec targets)
        computeSchedule();
        if (getenv("LRQR_SCHEDULE") && atoi(getenv("LRQR_SCHEDULE"))) printSchedule();
        // T3: L2 access-policy window (spec §10 Replicate at L2 = partition pair c=2).
        // V6 W1 (spec §10 Theorem U): live-by-default. The prior ~10% tf32 regression
        // came from the 37.5MB device reservation claiming L2 from the TC far-rest
        // streaming set. Re-route: size the persisting reservation to the Tsp working
        // set itself (B²·sizeof ≤ 2MB) so the far-rest L2 budget is left intact — a
        // continuous parameter, not a switch. LRQR_L2HIT/L2SETASIDE override.
        useL2win = true;
        l2setaside = std::min(size_t(B) * size_t(B) * sizeof(Real),
                              size_t(4 * 1024 * 1024));
        if (const char* e = getenv("LRQR_L2HIT")) l2hit = float(atof(e));
        if (const char* e = getenv("LRQR_L2SETASIDE")) l2setaside = size_t(atoll(e)) * 1024 * 1024;
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2setaside));
        fprintf(stderr, "[LRQR] L2 window ON: Tsp pin hitRatio=%.2f setaside=%zuMB\n",
                l2hit, l2setaside / (1024 * 1024));
    }
    void setAttrs() {
        auto setsm = [](const void* f, int bytes) {
            CUDA_CHECK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        };
        setsm((const void*)k_leaf_geqrf<Real, SLAB, NB>, bqr_smem<Real, NB>(SLAB));
        setsm((const void*)k_merge_geqrf<Real, SLAB, NB>, merge_smem<Real, NB>());
        setsm((const void*)k_larft<Real, SLAB, NB, false>, smLarft(false));
        setsm((const void*)k_larft<Real, SLAB, NB, true>,  smLarft(true));
        setsm((const void*)k_dsweep<Real, NB>, 3 * NB * NB * (int)sizeof(Real));
        setsm((const void*)k_leaf_apply<Real, SLAB, NB>,
              ((SLAB + 1) * NB + 2 * NB * NB) * (int)sizeof(Real));
        setsm((const void*)k_lu_sign<Real, NB>, (NB * NB + NB) * (int)sizeof(Real));
        if (fusedSmem > 0)
            CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_fused<Real, SLAB, NB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, fusedSmem));
        if (useDomino) {
            setsm((const void*)k_panel_domino<Real, NB>, domSmem(domRpb(m)));
            setsm((const void*)k_panel_domino<Real, NB, 16, true>,
                  domSmem(clRpb(std::min(m, clRowMax))));
            setsm((const void*)k_panel_domino<Real, NB, 16, false, true>,
                  domSmem(clRpbMax));  // G2-PANEL V13: gmem-domino uses clRpbMax (large rpb)
            setsm((const void*)k_panel_domino<Real, NB, 16, false, true, true>,
                  domSmem(clRpbMax));  // capacity-overflow persistent-row carrier
            CUDA_CHECK(cudaFuncSetAttribute(
                (const void*)k_panel_domino<Real, NB, 16, true>,
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            if (usePipe)
                setsm((const void*)k_panel_domino_pipe<Real, NB>,
                      pipeSmem(std::min(m, pipeRowMax)));
            if (useDsm) {
                setsm((const void*)k_panel_dsm<Real, NB>, dsmSmem(dsmRpbMax));
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)k_panel_dsm<Real, NB>,
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
                setsm((const void*)k_super_dsm<Real, NB>, dsmSmem2(superRpbMax));
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)k_super_dsm<Real, NB>,
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            }
            if (useDefuse) {
                setsm((const void*)k_dom_cols<Real, NB, 16, false>, colsSmem(colsRpbMax));
                setsm((const void*)k_dom_cols<Real, NB, 16, true>,  colsSmem(colsRpbMax));
                CUDA_CHECK(cudaFuncSetAttribute(
                    (const void*)k_dom_cols<Real, NB, 16, true>,
                    cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            }
        }
    }
    // smem dynamic cap (H200: 232448 bytes usable per block)
    static constexpr int SMEM_CAP = 232448;
    // The c_L dial is evaluated in panel() from row-state work, the measured
    // shared→DSM fan-out and phase depth, and the capacity-derived replica cap.
    static constexpr int roundDown32(int x) { return (x / 32) * 32; }
    // Round to the nearest power of two (ties round up). Used by the continuous
    // ν-ladder (HBM far-rest chunk) formulas that replace the former n/aspect
    // case splits (spec §10 Theorem U: parameters vary continuously, not by case).
    static int roundPow2(int x) {
        if (x <= 1) return 1;
        int lo = 1, hi = 1;
        while (hi < x) { lo = hi; hi <<= 1; }
        return (x - lo < hi - x) ? lo : hi;
    }
    // Far-rest chunk width — ONE definition for every far site (c_L=1, c_L>1,
    // and the workspace cap), so they can never diverge.
    //   • fp32-accurate composed tiers (tf32/fp32/fp64): a shape law fitted on the
    //     measured per-tier optima (GF_FCH_7979 + GF_FCH2_7980, n=65536, reps 5,
    //     two GPUs): tf32/fp32 want FINE chunks (2048 — a fat chunk's s1 GEMM spans
    //     several panel periods and the 3-slot evFar pipeline stalls s0), fp64 wants
    //     FAT (launch/Yt amortization + far-GEMM shape). B·(w/4)^4 reproduces both
    //     with existing parameters, no branch.
    //   • 3×TF32 (Ozaki split-K EC): the chunk is an ACCURACY parameter, not a perf
    //     knob — it sets the RZ-error accumulation length of the split GEMMs. Kept at
    //     its proven sizing (relRR ≤10× gate, G5); the shape-law value regressed the
    //     κ-conditioned 65536 cell to 12.4× (GRAPHFREE_V2_7982). Spec-honest per the
    //     presence-vs-parameter clause: a per-tier arithmetic parameter, not a gate.
    // mr = the far GEMM's M-extent (the trailing block's row count). It enters the
    // 3xTF32 branch because that branch was SHAPE-BLIND: it returned
    // max(roundPow2(B*word), roundPow2(12288)), a constant independent of the
    // matrix, and job 9464 measured the cost of that on thin-m cells.
    //
    // WHY A BOUND IN mr, and why this is a law rather than a fit. Far traffic is
    //     Q_far(w) = 2*mr*w_far + mr*NB*w_far/w
    // whose only w-dependent term falls monotonically, so COMMUNICATION alone says
    // "w as large as possible". Measurement disagrees on thin m: at mr=1024 the
    // 3xTF32 curve is a genuine bowl with its floor at 2048 and the shipped 8192
    // sitting 15.7% above it. That is the rate term R(shape) opposing the
    // communication term -- the tension this cost model exists to arbitrate -- and
    // the measured optimum tracks mr (1024->2048, 2048->16384, 4096->16384). So the
    // chunk is bounded by a constant multiple of the M-extent, floored at the value
    // the workspace already reserves:
    //     w*_3x = min( w_far, max(2048, roundPow2(2*mr)) )
    // Measured against the swept curves (job 9464): +13.58 / +12.09 / +8.79% at
    // mr=1024, -1.56 / -0.44% at mr=2048, 0.00% at mr=4096, -0.19% on the tall
    // control. Large gains where the old law was worst, under 1.6% cost where it was
    // already near optimal.
    //
    // The non-3x branch is UNCHANGED: its B*w4^4 form already tracks shape through B
    // and the sweep found it at or within ~1% of the argmin on every fp32/fp64/tf32
    // control, so there is nothing measured to justify touching it.
    static int farChunkCore(bool is3x, int B, int wfar, int mr) {
        if (is3x)
            return std::min(wfar, std::max(2048, roundPow2(2 * std::max(1, mr))));
        const int w4 = (int)sizeof(Real) / 4;   // 1 (fp32-class) / 2 (fp64)
        return std::min(wfar, std::max(512, roundPow2(B * w4 * w4 * w4 * w4)));
    }
    int farChunkFor(int B, int wfar) const { return farChunkCore(use3xtf32, B, wfar, m); }
    // Largest power of two ≤ x (c_L binary merge-tree fanout, spec §7).
    static int floorPow2(int x) {
        if (x < 1) return 1;
        int p = 1;
        while ((p << 1) <= x) p <<= 1;
        return p;
    }

    // super-DSM: + 1792-elem T-assembly scratch for rank 0
    static constexpr int dsmOverhead = (int)(sizeof(DsmPub<Real>) / sizeof(Real)) + 3 * NB + 64;
    static constexpr int superOverhead = dsmOverhead + (NB - 16) * 16;
    static constexpr int superRpbMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - superOverhead) / 16);
    static int dsmSmem2(int rpb) {
        return (rpb * 16 + (int)(sizeof(DsmPub<Real>) / sizeof(Real)) + 3 * NB + 64
                + (NB - 16) * 16) * (int)sizeof(Real);
    }
    // DSM-pipe: slice smem = rpb×16 + DsmPub + f/β/τ̃ + ssm
    static constexpr int dsmRpbMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - dsmOverhead) / 16);
    static constexpr int dsmRowMax = 16 * dsmRpbMax;
    // DSM-2x: per-CTA smem budget = SMEM_CAP / minBlocksDsm so the launch-bounds
    // residency target and the row cap derive from ONE capacity formula.
    static constexpr int minBlocksDsm = sizeof(Real) == 8 ? 2 : 1;
    static constexpr int dsmRpb2Max = roundDown32((SMEM_CAP / minBlocksDsm / (int)sizeof(Real) - dsmOverhead) / 16);
    static constexpr int dsm2RowMax = 16 * dsmRpb2Max;
    static int dsmSmem(int rpb) {
        return (rpb * 16 + (int)(sizeof(DsmPub<Real>) / sizeof(Real)) + 3 * NB + 64)
               * (int)sizeof(Real);
    }
    // de-fused column-loop kernel: slice smem = rpb×16 minipanel + ssm + f/β/τ̃
    static constexpr int colsRpbMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - (16 + 3 * NB)) / 16);
    static int colsSmem(int rpb) {
        return (rpb * 16 + 16 + 3 * NB) * (int)sizeof(Real);
    }

    // domino kernel geometry: rows per block (128, widened only past 132 blocks) and the
    // dynamic-smem footprint. G5 fix: smem sizing dispatches on niter (the runtime grid-
    // stride loop count = ceil(nbl/smCount), derived from mk/rpb/smCount here).
    //   niter==1 (nbl ≤ smCount): ONE rpb×16 minipanel buffer (W2 tailfix preserved —
    //     b3 writes next mp to gmem, reloads from gmem; clRpbMax stays /16 → tall win).
    //   niter>1  (nbl >  smCount): TWO rpb×16 minipanel buffers (msmN restored — b3 stages
    //     next mp into smem; dots/σ read from smem, not gmem). clRpbMaxDB is /32.
    static int domRpb(int mk, bool db = false) {
        int cap = 132;               // LRQR_DOMNBL: leave SMs free for concurrent GEMMs
        if (const char* e = getenv("LRQR_DOMNBL")) cap = std::max(2, atoi(e));
        int rpb = 128;
        if (cdiv(mk, rpb) > cap) rpb = 32 * cdiv(cdiv(mk, cap), 32);
        rpb = std::min(rpb, db ? (int)clRpbMaxDB : (int)clRpbMax);
        return rpb;
    }
    // largest 16-block cluster panel whose smem fits the dynamic smem cap:
    // domSmem(rpb) ≤ SMEM_CAP. rpb rounded to mult of 32.
    static constexpr int domOverhead = 16 * (NB - 16) + 2 * 16 * 16 + 16 + NB + 8 + 3 * NB;
    // niter==1 path (single minipanel buffer, W2 tailfix): /16 → fp64 1696, fp32 3520.
    static constexpr int clRpbMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - domOverhead) / 16);
    static constexpr int clRowMax = 16 * clRpbMax;
    // niter>1 path (double-buffered minipanel, msmN restored): /32 → fp64 832, fp32 1728.
    static constexpr int clRpbMaxDB = roundDown32((SMEM_CAP / (int)sizeof(Real) - domOverhead) / 32);
    static int clRpb(int mk) { return 32 * cdiv(cdiv(mk, 16), 32); }
    // pipelined domino: whole minipanel (ldm=mk+1 × 16) in ONE block's smem + scratch
    // overhead = Gs(256)+Ts(256)+dsl(256)+ssm(16)+sgL(16)+fvL(NB)+betL(NB)+ttL(NB) = 992+pad
    // +1024 static smem; 1280 elems pad ensures total <= SMEM_CAP for all Real/sizeof
    static constexpr int pipeRowMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - 1280) / 16 - 1);
    static int pipeSmem(int mk) {
        return ((mk + 1) * 16 + 1280) * (int)sizeof(Real);
    }
    static int domSmem(int rpb, bool db = false) {
        return (rpb * (db ? 32 : 16) + 16 * (NB - 16) + 2 * 16 * 16 + 16 + NB + 8
                + 3 * NB)                             // block-local f/β/τ̃
               * (int)sizeof(Real);
    }

    bool greenInit() {
        // V6 W1: LRQR_NO_GREEN kill-hatch removed (spec §10 Theorem U). Green is
        // still opt-in via LRQR_GREEN (:5110, Phase C will flip that default);
        // the (!useFused && !useDomino) capacity guard stays as a graceful fallback.
        if (!useFused && !useDomino) return false;
        auto ok = [](CUresult r) { return r == CUDA_SUCCESS; };
        CUdevice dev;
        int rtdev = 0; CUDA_CHECK(cudaGetDevice(&rtdev));
        if (!ok(cuDeviceGet(&dev, rtdev))) return false;
        CUdevResource all{};
        if (!ok(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM))) return false;
        CUdevResource grp{}, rem{};
        unsigned ng = 1;
        int smallCnt = 16;      // s1 partition size; defuse wants ~88 (far there)
        if (const char* e = getenv("LRQR_GREEN_SMALL")) smallCnt = atoi(e);
        if (!ok(cuDevSmResourceSplitByCount(&grp, &ng, &all, &rem, 0, smallCnt)) || ng < 1)
            return false;
        CUdevResourceDesc dS = nullptr, dB = nullptr;
        if (!ok(cuDevResourceGenerateDesc(&dS, &grp, 1))) return false;
        if (!ok(cuDevResourceGenerateDesc(&dB, &rem, 1))) return false;
        if (!ok(cuGreenCtxCreate(&gcS, dS, dev, CU_GREEN_CTX_DEFAULT_STREAM))) return false;
        if (!ok(cuGreenCtxCreate(&gcB, dB, dev, CU_GREEN_CTX_DEFAULT_STREAM))) {
            cuGreenCtxDestroy(gcS); gcS = nullptr; return false;
        }
        if (!ok(cuCtxFromGreenCtx(&ctxB, gcB)) || !ok(cuCtxFromGreenCtx(&ctxS, gcS))) return false;
        CUstream sb = nullptr, ss = nullptr;
        if (!ok(cuGreenCtxStreamCreate(&sb, gcB, CU_STREAM_NON_BLOCKING, 0))) return false;
        // de-fused mode: ONLY the panel column-loop kernels get the small reserved
        // partition (sCols); s0 AND s1 (narrow/slice/boundaries + far) share the big
        // one where non-coop kernels wave-interleave freely. Otherwise: original
        // fused mapping (s1 small).
        if (useDefuse) {
            int loPri = 0, hiPri = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&loPri, &hiPri));
            CUstream sb2 = nullptr;
            // boundary chain (s0) at highest priority: its blocks jump the dispatch
            // queue as far-GEMM blocks retire (block, not wave, granularity)
            cuStreamDestroy((CUstream)sb);
            if (!ok(cuGreenCtxStreamCreate(&sb, gcB, CU_STREAM_NON_BLOCKING, hiPri)))
                return false;
            if (!ok(cuGreenCtxStreamCreate(&sb2, gcB, CU_STREAM_NON_BLOCKING, loPri)))
                return false;
            if (!ok(cuGreenCtxStreamCreate(&ss, gcS, CU_STREAM_NON_BLOCKING, hiPri)))
                return false;
            cublasDestroy(cb0); cublasDestroy(cb1);
            cudaStreamDestroy(s0); cudaStreamDestroy(s1);
            s0 = (cudaStream_t)sb; s1 = (cudaStream_t)sb2;
            sCols = (cudaStream_t)ss;
            const size_t wsb = size_t(64) << 20;
            cuCtxPushCurrent(ctxB);
            CUBLAS_CHECK(cublasCreate(&cb0));
            CUBLAS_CHECK(cublasSetMathMode(cb0, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
            CUBLAS_CHECK(cublasSetWorkspace(cb0, cbws0, wsb));
            CUBLAS_CHECK(cublasSetStream(cb0, s0));
            CUBLAS_CHECK(cublasCreate(&cb1));
            CUBLAS_CHECK(cublasSetMathMode(cb1, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
            CUBLAS_CHECK(cublasSetWorkspace(cb1, cbws1, wsb));
            CUBLAS_CHECK(cublasSetStream(cb1, s1));
            cudaEventDestroy(evPD); cudaEventDestroy(evSlice);
            CUDA_CHECK(cudaEventCreateWithFlags(&evPD, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evSlice, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evC0, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evC1, cudaEventDisableTiming));
            for (int p = 0; p < 2; ++p) {
                cudaEventDestroy(evFar[p]); cudaEventDestroy(evRest[p]);
                CUDA_CHECK(cudaEventCreateWithFlags(&evFar[p], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&evRest[p], cudaEventDisableTiming));
            }
            setAttrs();
            {   // warm the exact de-fused GEMM kernels in this fresh context
                cudaFuncAttributes fa;
                cudaFuncGetAttributes(&fa, (const void*)k_dom_small<Real, NB>);
                cudaFuncGetAttributes(&fa, (const void*)k_dom_fix<Real, NB>);
                cudaFuncGetAttributes(&fa, (const void*)k_dom_tail<Real, NB>);
                for (int t32 = 0; t32 < 2; ++t32) {
                    gemm(cb0, t32 != 0, CUBLAS_OP_T, CUBLAS_OP_N, 16, NB, m - NB,
                         Real(1), domUg, (int)m, SPVx[0], (int)m, Real(0), domPsum, 16);
                    gemm(cb0, t32 != 0, CUBLAS_OP_N, CUBLAS_OP_N, m - NB, NB - 16, 16,
                         Real(-1), domUg, (int)m, domW2, 16, Real(1), SPVx[0], (int)m);
                }
                CUDA_CHECK(cudaStreamSynchronize(s0));
            }
            cuCtxPopCurrent(nullptr);
            cuCtxPushCurrent(ctxS);
            setAttrs();                 // cols kernels launch in this context
            cuCtxPopCurrent(nullptr);
            smallSMs = grp.sm.smCount;
            fusedGrid = rem.sm.smCount;
            return true;
        }
        if (!ok(cuGreenCtxStreamCreate(&ss, gcS, CU_STREAM_NON_BLOCKING, 0))) return false;
        // replace streams/handles/events with partition-bound ones
        cublasDestroy(cb0); cublasDestroy(cb1);
        cudaStreamDestroy(s0); cudaStreamDestroy(s1);
        s0 = (cudaStream_t)sb; s1 = (cudaStream_t)ss;
        const size_t wsbytes = size_t(64) << 20;
        cuCtxPushCurrent(ctxB);
        CUBLAS_CHECK(cublasCreate(&cb0));
        CUBLAS_CHECK(cublasSetMathMode(cb0, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
        CUBLAS_CHECK(cublasSetWorkspace(cb0, cbws0, wsbytes));
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        cudaEventDestroy(evPD); cudaEventDestroy(evSlice);
        CUDA_CHECK(cudaEventCreateWithFlags(&evPD, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evSlice, cudaEventDisableTiming));
        setAttrs();                     // per-context function attributes!
        if (useDefuse) {
            // lazy-load gotcha (§1): warm the EXACT de-fused GEMM kernels in this
            // fresh context before any coop cols kernel can be resident beside a
            // blocking module load; preload our own kernels via attribute queries.
            cudaFuncAttributes fa;
            cudaFuncGetAttributes(&fa, (const void*)k_dom_small<Real, NB>);
            cudaFuncGetAttributes(&fa, (const void*)k_dom_fix<Real, NB>);
            cudaFuncGetAttributes(&fa, (const void*)k_dom_tail<Real, NB>);
            for (int t32 = 0; t32 < 2; ++t32) {
                gemm(cb0, t32 != 0, CUBLAS_OP_T, CUBLAS_OP_N, 16, NB, m - NB,
                     Real(1), domUg, (int)m, SPVx[0], (int)m, Real(0), domPsum, 16);
                gemm(cb0, t32 != 0, CUBLAS_OP_N, CUBLAS_OP_N, m - NB, NB - 16, 16,
                     Real(-1), domUg, (int)m, domW2, 16, Real(1), SPVx[0], (int)m);
            }
            CUDA_CHECK(cudaStreamSynchronize(s0));
        }
        cuCtxPopCurrent(nullptr);
        cuCtxPushCurrent(ctxS);
        CUBLAS_CHECK(cublasCreate(&cb1));
        CUBLAS_CHECK(cublasSetMathMode(cb1, (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH : ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH))));
        CUBLAS_CHECK(cublasSetWorkspace(cb1, cbws1, wsbytes));
        CUBLAS_CHECK(cublasSetStream(cb1, s1));
        for (int p = 0; p < 2; ++p) {
            cudaEventDestroy(evFar[p]); cudaEventDestroy(evRest[p]);
            CUDA_CHECK(cudaEventCreateWithFlags(&evFar[p], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&evRest[p], cudaEventDisableTiming));
        }
        if (!evFar[2]) { CUDA_CHECK(cudaEventCreateWithFlags(&evFar[2], cudaEventDisableTiming)); }
        if (!evRest[2]) { CUDA_CHECK(cudaEventCreateWithFlags(&evRest[2], cudaEventDisableTiming)); }
        cuCtxPopCurrent(nullptr);
        smallSMs = grp.sm.smCount;
        fusedGrid = rem.sm.smCount;     // 1 block/SM at our smem footprint
        return true;
    }

    static int smLarft(bool merge) {
        int vr = merge ? NB : SLAB;
        return ((vr + 1) * NB + 2 * NB * NB + NB) * (int)sizeof(Real);
    }

    void destroy() {
        // All work is synchronized by geqrf()/orgqr() before teardown.
        for (Real* p : {Pbuf, QT, QS, Vm, TmM, TmL, tauL, taum, Tpan, Sg, Zb, M1b, Wn1, Wn2,
                        Wr1, Wr2, domScal, domSlot, domProw, domPp, domPsum, domT16,
                        domUg, domW2, domGs, rVg, rTg, domTpanL})
            if (p) cudaFree(p);
        if (domT) cudaFree(domT);
        if (domSigD) cudaFree(domSigD);
        if (domDcolP) { cudaFree(domDcolP); domDcolP = nullptr; }
        if (domDcolR) { cudaFree(domDcolR); domDcolR = nullptr; }
        if (domDslotD) cudaFree(domDslotD);
        if (pipeF) cudaFree(pipeF);
        if (gbar) cudaFree(gbar);
        if (rPanelGen) cudaFree((void*)rPanelGen);
        if (rUpdateCnt) cudaFree((void*)rUpdateCnt);
        if (d_nodes2d) cudaFree(d_nodes2d);
        for (int j = 0; j < cLmax; ++j) {
            for (Real* p : {domScalL[j], domSlotL[j], domProwL[j], domPpL[j],
                            domPsumL[j], domT16L[j]})
                if (p) cudaFree(p);
            if (domSigDL[j]) cudaFree(domSigDL[j]);
            if (domDslotDL[j]) cudaFree(domDslotDL[j]);
            if (gbarL[j]) cudaFree(gbarL[j]);
        }
        for (int s = 0; s < 2; ++s) {
            for (Real* p : {VmSav[s], TmMSav[s], domTpanLSav[s]})
                if (p) cudaFree(p);
        }
        for (int p = 0; p < 3; ++p) {
            if (SPVx[p]) cudaFree(SPVx[p]);
            if (Tspx[p]) cudaFree(Tspx[p]);
            if (Wax[p]) cudaFree(Wax[p]);
            if (Wbx[p]) cudaFree(Wbx[p]);
            if (Ytx[p]) cudaFree(Ytx[p]);
            if (use3xtf32) {
                if (Yhi[p])  cudaFree(Yhi[p]);
                if (Ylo[p])  cudaFree(Ylo[p]);
                if (Ythi[p]) cudaFree(Ythi[p]);
                if (Ytlo[p]) cudaFree(Ytlo[p]);
            }
        }
        if (absmaxPart) { cudaFree(absmaxPart); absmaxPart = nullptr; }
        if (use3xtf32) {
            for (int p2 = 0; p2 < 2; ++p2) {
                if (Chi[p2]) cudaFree(Chi[p2]);
                if (Clo[p2]) cudaFree(Clo[p2]);
                if (evSp3x[p2]) cudaEventDestroy(evSp3x[p2]);
                if (evG3x[p2]) cudaEventDestroy(evG3x[p2]);
            }
            if (s3xsp) cudaStreamDestroy(s3xsp);
            if (W2hi) cudaFree(W2hi);
            if (W2lo) cudaFree(W2lo);
        }
        if (d_nodes) cudaFree(d_nodes);
        if (domTpanL_pan) cudaFree(domTpanL_pan);
        if (Vm_pan) cudaFree(Vm_pan);
        if (TmM_pan) cudaFree(TmM_pan);
        if (cL_pan) cudaFreeHost(cL_pan);
        if (nm_pan) cudaFreeHost(nm_pan);
        if (slabStart_pan) cudaFreeHost(slabStart_pan);
        if (slabSize_pan) cudaFreeHost(slabSize_pan);
        if (nodes_pan) cudaFreeHost(nodes_pan);
        if (dArgs) cudaFree(dArgs);
        if (hReady) cudaFreeHost((void*)hReady);
        if (hFin) cudaFreeHost((void*)hFin);
        if (dBcnt) cudaFree(dBcnt);
        if (dBgen) cudaFree(dBgen);
        if (hProgK) cudaFreeHost((void*)hProgK);
        if (hUpdPhase) cudaFreeHost((void*)hUpdPhase);
        if (cb0) cublasDestroy(cb0);
        if (cb1) cublasDestroy(cb1);
        if (cbT) cublasDestroy(cbT);
        if (lt0) cublasLtDestroy(lt0);
        if (lt1) cublasLtDestroy(lt1);
        if (ltT) cublasLtDestroy(ltT);
        if (ltDescTN) cublasLtMatmulDescDestroy(ltDescTN);
        if (ltDescNN) cublasLtMatmulDescDestroy(ltDescNN);
        if (ltDescTN_plain) cublasLtMatmulDescDestroy(ltDescTN_plain);
        if (ltDescNN_plain) cublasLtMatmulDescDestroy(ltDescNN_plain);
        if (ltDescTN_f32) cublasLtMatmulDescDestroy(ltDescTN_f32);
        if (ltWorkspace) cudaFree(ltWorkspace);
        if (cbws0) cudaFree(cbws0);
        if (cbws1) cudaFree(cbws1);
        if (cbwsT) cudaFree(cbwsT);
        if (s0) cudaStreamDestroy(s0);
        if (s1) cudaStreamDestroy(s1);
        if (sP) cudaStreamDestroy(sP);
        if (sR) cudaStreamDestroy(sR);
        if (sN) cudaStreamDestroy(sN);
        if (sCols) cudaStreamDestroy(sCols);
        for (int j = 0; j < cLmax; ++j) {
            if (domSt[j]) cudaStreamDestroy(domSt[j]);
            if (evDom[j]) cudaEventDestroy(evDom[j]);
            if (evM2lvl[j]) cudaEventDestroy(evM2lvl[j]);
        }
        if (evPrePanel) cudaEventDestroy(evPrePanel);
        if (evS1done) cudaEventDestroy(evS1done);
        if (evTsp) cudaEventDestroy(evTsp);
        if (evRst) cudaEventDestroy(evRst);
        for (int p = 0; p < 3; ++p) if (evFar[p]) cudaEventDestroy(evFar[p]);
        if (evSlice) cudaEventDestroy(evSlice);
        if (evIn) cudaEventDestroy(evIn);
        if (evPD) cudaEventDestroy(evPD);
        for (int p = 0; p < 2; ++p) if (evRest[p]) cudaEventDestroy(evRest[p]);
        for (int s = 0; s < 2; ++s) {
            if (evMergeCpy[s]) cudaEventDestroy(evMergeCpy[s]);
            if (evFar25[s]) cudaEventDestroy(evFar25[s]);
        }
        if (evSt0a) cudaEventDestroy(evSt0a);
        if (evSt0b) cudaEventDestroy(evSt0b);
        if (evSt1a) cudaEventDestroy(evSt1a);
        if (evSt1b) cudaEventDestroy(evSt1b);
        if (evC0) cudaEventDestroy(evC0);
        if (evC1) cudaEventDestroy(evC1);
    }

    void buildArgs(Real* A, size_t lda) {
        hArgs.resize(npanels);
        for (int k = 0; k < npanels; ++k) {
            const PTab& t = tab[k];
            const int row0 = k * NB, mk = m - row0, N = t.N, mpad = N * SLAB;
            const int sp = (k * NB) / B * B, j0 = k * NB - sp, spi = (sp / B) & 1;
            PanelArgs<Real>& pa = hArgs[k];
            pa.A = A + size_t(k) * NB * lda + row0;
            pa.lda = lda; pa.row0 = row0; pa.mk = mk; pa.mpad = mpad; pa.N = N;
            pa.Pbuf = Pbuf; pa.QT = QT; pa.QS = QS; pa.Vm = Vm; pa.TmM = TmM; pa.TmL = TmL;
            pa.tauL = tauL; pa.taum = taum;
            pa.nodes = d_nodes + t.nodesOff;
            pa.nlv = (int)t.lv.size();
            for (int l = 0; l < pa.nlv && l < 20; ++l)
                pa.lvl[l] = {t.lv[l].first - t.nodesOff, t.lv[l].second};
            pa.Sg = Sg + size_t(k) * NB;
            pa.Tpan = Tpan + size_t(k) * NB * NB;
            pa.SPV = SPVx[spi % 3]; pa.ldspv = m; pa.spvcol = j0;
        }
        CUDA_CHECK(cudaMemcpyAsync(dArgs, hArgs.data(), sizeof(PanelArgs<Real>) * npanels,
                                   cudaMemcpyHostToDevice, s0));
    }

    // ------------------------------------------------------------ c_L-lane panel (spec §7.2)
    // Split panel rows into c_L slabs, run per-lane dominoes, merge R-states by ONE
    // cross-lane ⊕ (k_merge_geqrf — Householder-on-stack, spec §3/§14, NEVER Gram).
    // No reconstruction on the chain (LD-QR §5 Principle S): the trailing update is a
    // two-level WY apply (per-lane V_j/Tpan_j + merge Vm/TmM).
    //
    // c_L=1 (validated, relRR 7.28e-04 @8192 TF32): runs the standard domino with the
    //   plan's regular buffers — no per-layer-buffer bug, no broken Phase-4 recon.
    // c_L>1: per-lane dominoes + ⊕ merge + two-level WY apply (applyQt_25d: Stage 1
    //   per-lane applyQt + Stage 2 mergeApplyQt). Each layer owns a stream; measured
    //   occupancy determines how many replicas may execute in one wave.
    void panel_25d(Real* A, size_t lda, int k, Real* SPV, int spvcol, cudaStream_t st,
                   int cL) {
        const int row0 = k * NB, mk = m - row0;
        Real* Acol = A + size_t(k) * NB * lda + row0;
        // The kernel is launch-bounded at 512 threads and its 128-register
        // footprint fits exactly once in the measured 64K-register SM file.
        // Use that capacity-derived block width uniformly for every storage tier.
        const int dtpb = 512;
        // ---- c_L=1 fast path: standard domino, plan buffers (validated) ----
        if (cL <= 1) {
            DomArgs<Real> da;
            da.A = Acol; da.lda = lda; da.mk = mk; da.row0 = row0;
            da.SPV = SPV; da.ldspv = m; da.spvcol = spvcol;
            da.Tpan = Tpan + size_t(k) * NB * NB;
            da.tauT = domScal; da.fv = domScal + NB; da.bet = domScal + 2 * NB;
            da.sig = domScal + 3 * NB;
            da.dslot = domSlot; da.prow = domProw; da.Pp = domPp; da.Psum = domPsum;
            da.sigD = domSigD; da.dslotD = domDslotD; da.dcolP = domDcolP; da.dcolStride = 132;
            da.T16g = domT16; da.tphase = nullptr;
            int rpb = domRpb(mk, false);
            int nbl = cdiv(mk, rpb);
            const bool overflow = (nbl > smCount);
            if (overflow) { rpb = clRpbMax; nbl = smCount; }
            da.rpb = rpb;
            da.nslab = nbl; noteRung0(da.nslab);
            {   // grid-stride non-coop domino (handles nbl > smCount)
                if (overflow)
                    k_panel_domino<Real, NB, 16, false, true, true>
                        <<<nbl, dtpb, domSmem(rpb, false), st>>>(da, gbar);
                else
                    k_panel_domino<Real, NB, 16, false, true>
                        <<<nbl, dtpb, domSmem(rpb, false), st>>>(da, gbar);
            }
            return;
        }
        // ---- c_L>1: per-lane dominoes + cross-lane ⊕ merge + two-level WY apply ----
        // RESEARCH_MEMO §1-§3: Q_total = diag(Q_0,…,Q_{c_L-1}) · Q_merge. The trailing
        // update is a two-level WY apply (Stage 1 per-lane applyQt + Stage 2 merge apply).
        // No reconstruction on the chain (LD-QR §5 Principle S): per-lane V_j are applied
        // directly as bulk GEMMs; only the small merge tree (c_L·NB rows) is folded.
        // Ensure each slab has ≥ NB rows: the merge tree's R-rows (top NB per slab)
        // must fit within the slab. Reduce cL if mk is too small.
        while (cL > 1 && mk < cL * NB) cL /= 2;
        if (cL <= 1) {
            // Fall back to c_L=1 path
            last_cL = 1;
            last_row0 = -1; last_m2_levels = 0;   // no Stage-2 events for this panel
            DomArgs<Real> da;
            da.A = Acol; da.lda = lda; da.mk = mk; da.row0 = row0;
            da.SPV = SPV; da.ldspv = m; da.spvcol = spvcol;
            da.Tpan = Tpan + size_t(k) * NB * NB;
            da.tauT = domScal; da.fv = domScal + NB; da.bet = domScal + 2 * NB;
            da.sig = domScal + 3 * NB;
            da.dslot = domSlot; da.prow = domProw; da.Pp = domPp; da.Psum = domPsum;
            da.sigD = domSigD; da.dslotD = domDslotD; da.dcolP = domDcolP; da.dcolStride = 132;
            da.T16g = domT16; da.tphase = nullptr;
            int rpb = domRpb(mk, false);
            int nbl = cdiv(mk, rpb);
            const bool overflow = (nbl > smCount);
            if (overflow) { rpb = clRpbMax; nbl = smCount; }
            da.rpb = rpb;
            da.nslab = nbl; noteRung0(da.nslab);
            {   // grid-stride non-coop domino (handles nbl > smCount)
                if (overflow)
                    k_panel_domino<Real, NB, 16, false, true, true>
                        <<<nbl, dtpb, domSmem(rpb, false), st>>>(da, gbar);
                else
                    k_panel_domino<Real, NB, 16, false, true>
                        <<<nbl, dtpb, domSmem(rpb, false), st>>>(da, gbar);
            }
            return;
        }
        int base = (mk / cL / 32) * 32;
        if (base < 32) base = 32;
        int slabStart[cLmax], slabSize[cLmax], acc = 0;
        for (int j = 0; j < cL; ++j) {
            slabStart[j] = acc;
            slabSize[j] = (j < cL - 1) ? base : (mk - acc);
            acc += slabSize[j];
        }
        // All leaf grids in a wave are individually all-resident (niter=1).
        // Derive the number of simultaneous replicas from measured kernel
        // occupancy and the largest leaf grid; additional replicas form later
        // waves.  This exercises replication without ever time-slicing a grid
        // barrier or returning to the inaccurate grid-stride leaf state.
        // ONE row-block size for every lane, derived from the WHOLE panel.
        //
        // domRpb targets a ~132-block grid, i.e. the whole device. Called per lane
        // with slabSize[j], each lane independently demands all 132 SMs, and since
        // the domino runs at one block per SM only ONE lane can be resident:
        // measured lanes_wave=1 at c_L=8, eight replica lanes executing strictly one
        // at a time. A 2.5D replica dimension PARTITIONS the processor set
        // (p_r p_c c = P, docs/10 §7.1); it does not let each replica claim all of
        // it. Sizing from mk gives lane j about 132/c blocks so the lanes tile the
        // device, and shortens each lane's chain to ~132/c blocks, which is the
        // D_lambda reduction the ordered-domino theorem is about.
        //
        // THEN GROW IT UNTIL THE LANES ACTUALLY FIT. Every lane holds its own grid
        // barrier gbarL[j], so with the lanes concurrent ALL of their blocks must be
        // co-resident or the barriers deadlock. The requirement is
        // sum_j nbl_j <= smCount, and cdiv rounds up PER LANE, so
        // sum_j nbl_j <= cdiv(mk,rpb) + cL can exceed smCount by up to cL even when
        // the unsplit grid fits exactly. That overrun is a deadlock introduced by
        // making the lanes concurrent, so it is closed here rather than discovered
        // in a hang. At c_L=1 this loop cannot execute (sum == cdiv(mk,rpb) <= 132),
        // so the unreplicated path is bit-identical.
        // GROW rpb ONLY IF IT ACTUALLY BUYS CO-RESIDENCY. Fattening the row block
        // trades intra-lane parallelism (fewer blocks) for the chance to fit more
        // lanes at once. That trade is worth making when it lets the lanes run
        // together and is pure loss when it does not -- and it often does not,
        // because rpb is capped at clRpbMax. Measured when this grew rpb
        // unconditionally: 262144x1024 at c=2 went 53.37 -> 64.62 ms and
        // 1048576x64 at c=8 went 8.12 -> 10.68 ms, both cells where residency was
        // unreachable anyway, so the fatter blocks bought nothing and cost the
        // parallelism. So: search for the smallest rpb that achieves
        // sum_j nbl_j <= smCount, and if none exists at or below clRpbMax, keep the
        // whole-panel size and let lanesPerWave serialize the lanes as before.
        // TWO REGIMES, and the fallback must be the per-lane size, not the panel one.
        //
        //   whole-panel sizing, domRpb(mk): lane j gets ~132/c blocks, so the lanes
        //     TILE the device and can be co-resident. Right when that is reachable.
        //   per-lane sizing, domRpb(slabSize[j]): each lane asks for a full ~132
        //     block grid, so the lanes serialize -- but each one is internally well
        //     parallelized while it runs.
        //
        // Measured the hard way: at 262144x1024, c=2, residency is unreachable
        // either way (2 x 78 = 156 blocks > 132). Whole-panel sizing there gives 78
        // blocks per lane instead of 128 and costs 53.4 -> 64.6 ms -- serialized
        // AND under-parallelized. My first fix "fell back" to domRpb(mk), which is
        // already the coarser size, so it fixed nothing; the fallback has to be the
        // per-lane size. At c_L=1 both expressions are domRpb(mk), so the
        // unreplicated path is bit-identical under either branch.
        auto totalBlocks = [&](int rpb) {
            long long t = 0;
            for (int j = 0; j < cL; ++j) t += cdiv(slabSize[j], rpb);
            return t;
        };
        const int rpbTile = domRpb(mk, false);          // lanes tile the device
        int rpbLanes = rpbTile;
        if (totalBlocks(rpbTile) > smCount) {
            int found = 0;
            for (int r = rpbTile; r <= (int)clRpbMax; r = 32 * (r / 32 + 1))
                if (totalBlocks(r) <= smCount) { found = r; break; }
            // No rpb <= clRpbMax makes the lanes fit, so they will serialize
            // whatever we do. Give each serialized lane the widest grid it can use.
            rpbLanes = found ? found : domRpb(slabSize[0], false);
        }
        int maxLeafBlocks = 1;
        int leafBlocksPerSm = smCount;
        for (int j = 0; j < cL; ++j) {
            const int rpbj = rpbLanes;
            maxLeafBlocks = std::max(maxLeafBlocks, cdiv(slabSize[j], rpbj));
            int occ = 1;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,
                k_panel_domino<Real, NB, 16, false, true>, dtpb,
                (size_t)domSmem(rpbj, false)));
            leafBlocksPerSm = std::min(leafBlocksPerSm, std::max(1, occ));
        }
        const int replicaCap = std::max(1, sched.rung[1].c);
        const int lanesPerWave = std::max(1, std::min({cL, replicaCap,
            leafBlocksPerSm * smCount / maxLeafBlocks}));
        // Reported for the cost model: replication only shortens the per-lane
        // Householder chain for lanes that actually run AT THE SAME TIME. Lanes
        // past this width form later waves, so they add merge levels without
        // shortening any chain, and a model using c rather than min(c,lanesWave)
        // would credit them with a saving they never delivered.
        //
        // It is NOT, on the evidence, what separates squares from tall cells. That
        // separation is chain DEPTH against panel COUNT, both already in the model:
        // the saving is A*rowStates/c and the cost B*npanels*log2(c), and rowStates
        // = cdiv(mk,SLAB) is 8192 at 1048576x64 against 32 at 4096^2 while npanels
        // runs the other way, 1 against 64. Measured: c=8 is worth 2.47x at
        // 1048576x64 (20.03 -> 8.10 ms), c=2 wins 9.6% at 262144x1024, and every
        // square loses monotonically (4096^2: 24.5 -> 45.3 -> 58.9 ms). Recorded as
        // a max because the wave width, like c_L itself, varies panel to panel.
        mv_lanes_wave = std::max(mv_lanes_wave, lanesPerWave);
        // ---- THE DIAGONAL EDGE: per-lane release, computed before last_* is clobbered ----
        // The panel loop's blanket `for j: cudaStreamWaitEvent(domSt[j], evPrePanel)`
        // makes every lane of epoch k+1 wait for ALL of epoch k. That is the surplus
        // edge X(k,D-1) -> X(k+1,0): a legal topological order of the domino DAG, but
        // one the readiness invariant never asked for, and it is exactly the
        // difference between Theta(K+D) and Theta(K*D) (31:345).
        //
        // What lane j actually needs, from the row-support argument: Stage 1 of the
        // previous panel (its own Q over full slabs), plus only those Stage-2 levels
        // whose top bands intersect lane j's rows. Stage 2's node (top,bot) writes NB
        // rows at each of two slab heads and nothing else (mergeApplyQt), and slab s
        // is consumed at level ctz(s), so most lanes need only the earliest levels.
        //
        // The previous panel's slabs are NOT this panel's: row0 advances by NB and
        // base shrinks, so lane j of epoch k+1 straddles different epoch-k slabs.
        // Hence the release level is a max over the epoch-k slab heads that actually
        // land inside this lane's rows -- computed here, not assumed.
        // ---- THE INVARIANT, VERBATIM (31:329) ----
        //     X_{k,j-1}   -> X_{k,j}                 vertical
        //     X_{k-1,j+1} -> X_{k,j}   (j < D-1)     diagonal
        //     X_{k-1,D-1} -> X_{k,D-1}               top-serial
        // The leaves are stage j = 0, so the rule is X_{k-1,1} -> X_{k,0}: release on
        // the PREVIOUS epoch's stage 1, UNIFORMLY across lanes. The proof says why --
        // "the only prior-epoch transformation not already committed on those rows is
        // the nearest ancestor-stage effect" -- and the nearest ancestor of stage j is
        // j+1. When D < 2 there is no stage 1 and the top-serial edge applies, which
        // is the full tree.
        //
        // WHAT WAS HERE BEFORE, and why it was wrong. A per-lane level from row-band
        // intersection and ctz(s): for odd lanes ctz = 0, releasing ONE LEVEL EARLIER
        // than the invariant permits. That is a read-before-write, and it produced
        // exactly the observed signature -- error compounding with n (1.1e-1 @4096,
        // 1e8 @16384, 1e31 @32768, nan @65536), 13 fp64 failures, 36/71. It was an
        // optimisation the spec never granted, and it was then patched around with an
        // evRest wait instead of being rechecked against the invariant.
        //
        // LADDER (LRQR_PIPEMODE), so the three hypotheses are separable in one job:
        //   0  full tree      D-1        must be clean; if not, the defect is NOT the
        //                                level math but the cross-stream dependency set
        //   1  invariant      min(1,D-1) the target
        //   2  ctz(j)         known-bad  positive control; a ladder that cannot
        //                                reproduce the known failure proves nothing
        static const int pipeMode = [] {
            const char* e = getenv("LRQR_PIPEMODE"); return e ? atoi(e) : 1;
        }();
        int laneRelease[cLmax];
        const bool pipeOK = pipe25d() && last_row0 >= 0 && last_cL > 1 &&
                            last_m2_applied > 0;   // events exist only for APPLIED levels
        // Release after the last level the global apply actually performed. Events
        // exist only for those, so this is also the largest index evM2lvl[] can take.
        const int fullTree = std::max(0, last_m2_applied - 1);
        for (int j = 0; j < cL; ++j) {
            int lev = fullTree;                                  // mode 0 and default
            if (pipeMode == 1 || pipeMode == 3) lev = std::min(1, fullTree);  // the invariant
            else if (pipeMode == 2) {                             // known-bad control
                const int r0 = row0 + slabStart[j], r1 = r0 + slabSize[j];
                int need = -1;
                for (int s = 0; s < last_cL; ++s) {
                    const int h0 = last_row0 + last_slabStart[s], h1 = h0 + NB;
                    if (h1 <= r0 || h0 >= r1) continue;
                    int e2 = 0;
                    if (s == 0) e2 = fullTree;
                    else { int t = s; while ((t & 1) == 0) { t >>= 1; ++e2; } }
                    if (e2 > need) need = e2;
                }
                lev = (need < 0) ? -1 : std::min(need, fullTree);
            }
            laneRelease[j] = pipeOK ? lev : fullTree;
        }
        // Arm the global near apply to stop at the same boundary the next epoch's
        // lanes will resume from. Only mode 3 splits the work; every other mode
        // leaves it at -1 (apply all levels), which is the shipped behaviour.
        // Stash the cap; do NOT arm it here. m2_max_level is a member, so arming it
        // in panel_25d capped EVERY mergeApplyQt call -- near, rest AND far. Only the
        // NEAR columns are self-applied by the next epoch's lanes, so the rest and far
        // ranges lost levels >= 2 outright, with nobody applying them. That is why
        // mode 3 measured 8/71, worse than mode 1's race: mode 1 read stale rows,
        // this DROPPED transformations across most of the trailing matrix. The cap is
        // armed around the near call only, by the panel loop.
        m2_max_level = -1;
        m2_near_cap = -1;
        if (pipeMode == 3 && pipe25d()) {
            int nlvv = 0; for (int step = 1; step < cL; step *= 2) ++nlvv;
            m2_near_cap = std::min(1, std::max(0, nlvv - 1));
        }
        // Store slab/merge info for applyQt_25d (valid until the next panel_25d call).
        last_cL = cL;
        for (int j = 0; j < cL; ++j) {
            last_slabStart[j] = slabStart[j];
            last_slabSize[j] = slabSize[j];
        }
        // Phase 1: per-lane dominoes — concurrent on separate streams with per-lane buffers.
        // Each lane gets its own domScalL/domPpL/gbarL/etc. so no cross-lane races.
        // After all launches, sync back to st before the merge tree.
        // Can every lane run the DSM-pipe domino instead of the gmem one? Decided
        // for the whole panel, not per lane, because k_panel_dsm spins on pipeF
        // doorbells across its clusters and so needs its blocks resident; a mixed
        // or oversubscribed launch would hang rather than run slowly.
        int dsmC[cLmax] = {}, dsmRpb[cLmax] = {};
        bool laneDsmOK = laneDsm();
        if (laneDsmOK) {
            long long tot = 0;
            for (int j = 0; j < cL && laneDsmOK; ++j) {
                const int g = laneDsmGrid(slabSize[j], &dsmC[j], &dsmRpb[j]);
                if (g <= 0) laneDsmOK = false; else tot += g;
            }
            if (tot > smCount) laneDsmOK = false;
        }
        // Snapshot the previous epoch's merge layout for the LOCAL apply below.
        // Vm/TmM still hold epoch k-1's data here: epoch k's merge runs later on st,
        // after the lane join, so the join orders it and no save buffer is needed.
        const int  pv_cL = last_cL, pv_nm = last_nm, pv_row0 = last_row0;
        const int  pv_lv = last_m2_levels, pv_applied = last_m2_applied;
        int  pv_slabStart[cLmax]; int2 pv_nodes[cLmax];
        for (int t = 0; t < cLmax; ++t) { pv_slabStart[t] = last_slabStart[t]; pv_nodes[t] = last_nodes[t]; }
        // CONSUME-AND-INVALIDATE. laneRelease[] above has now read the previous
        // panel's Stage-2 events; clear them so no LATER panel can release against a
        // panel two back. applyQt_25d is called only `if (wnear > 0)`, so on the last
        // leaf panel of each super-panel it never runs and never refreshes these --
        // leaving them stale exactly where the super-panel boundary is. This panel's
        // own applyQt_25d will set them again if it runs; if it does not, the next
        // panel correctly falls back to the evPrePanel barrier.
        last_m2_levels = 0; last_m2_applied = 0; last_row0 = -1;
        // STREAMS ABOVE cL MUST STILL BE ORDERED. domSt[] are PERSISTENT streams, and
        // the panel loop's blanket barrier waited ALL cLmax of them on evPrePanel
        // every panel. The per-lane release below only issues waits for j < cL, so a
        // stream that carried a lane of an EARLIER panel with a larger c_L is left
        // unordered -- its domino can still be writing A and domTpanL[j] while later
        // panels read them. That is what the blanket barrier was silently doing, and
        // removing it without replacing this is why mode 0 (full-tree release, which
        // is semantically the whole previous Stage-2 apply) still corrupted: the
        // defect was never the release LEVEL, it was the dependency SET.
        if (pipe25d())
            for (int j = cL; j < cLmax; ++j)
                CUDA_CHECK(cudaStreamWaitEvent(domSt[j], evPrePanel, 0));
        NVTX_RANGE("lane_merge");
        for (int j = 0; j < cL; ++j) {
            DomArgs<Real> da;
            da.A = Acol + size_t(slabStart[j]); da.lda = lda;
            da.mk = slabSize[j]; da.row0 = row0 + slabStart[j];
            da.SPV = SPV; da.ldspv = m; da.spvcol = spvcol;
            da.Tpan = domTpanL + size_t(j) * NB * NB;  // per-lane compound T
            da.tauT = domScalL[j]; da.fv = domScalL[j] + NB; da.bet = domScalL[j] + 2 * NB;
            da.sig = domScalL[j] + 3 * NB;
            da.dslot = domSlotL[j]; da.prow = domProwL[j];
            da.Pp = domPpL[j]; da.Psum = domPsumL[j];
            da.sigD = domSigDL[j]; da.dslotD = domDslotDL[j]; da.dcolP = domDcolP; da.dcolStride = 132;
            da.T16g = domT16L[j]; da.tphase = nullptr;
            // Each leaf must finish as a self-consistent Householder state before
            // it participates in the cross-lane ⊕.  Give it the capacity-sized
            // single-buffer row block and an all-resident grid (niter=1).
            const bool db = false;
            // rpbLanes: one row-block size for all lanes, sized from the whole panel
            // and grown until sum_j nbl_j <= smCount. See its derivation above.
            int rpb = rpbLanes;
            int nbl = cdiv(slabSize[j], rpb);
            // A lane's grid must be ALL-RESIDENT or its grid barrier deadlocks:
            // blocks past the SM residency limit never launch, and the resident
            // ones wait on them forever. The c_L=1 path has always guarded this
            // (overflow -> the grid-stride kernel); panel_25d never did, because
            // the shipped rule happens to pick a c_L large enough that every lane
            // fits -- at 1048576x64 it picks c_L=8, giving 131072 rows per lane.
            // That made the omission unobservable and turned c_L into a parameter
            // with a silent LOWER bound: forcing c_L=2 there gives 524288 rows per
            // lane, four times the blocks, and the factorization hangs. Measured:
            // job 9484 wedged on exactly that cell until it was cancelled.
            // Same fallback as the c_L=1 path, so the sweep can ask for any c.
            const bool overflow = (nbl > smCount);
            if (overflow) { rpb = clRpbMax; nbl = smCount; }
            da.rpb = rpb;
            da.nslab = nbl; noteRung0(da.nslab);
            cudaStream_t dst = domSt[j];
            // Release this lane at its own readiness point rather than at the
            // previous panel's completion. With the gate off, the panel loop's
            // blanket wait on evPrePanel still stands and nothing here fires.
            // THE DEPENDENCY THE ROW-SUPPORT ARGUMENT MISSED, and the reason the
            // first version of this corrupted every c>1 cell (residual 0.18 against
            // a 3.6e-15 baseline). The trailing update is carried on TWO streams:
            // the near slice on s0 and the rest slice on sN. The blanket barrier
            // covered both, because the panel loop waits s0 on evRest[pprev] BEFORE
            // recording evPrePanel, so evPrePanel transitively encodes the rest.
            // evM2lvl/evS1done are recorded on s0 only, so they encode the near
            // update alone -- a lane released on them races the previous panel's
            // rest slice, which is still writing the rows it is about to read.
            //
            // The diagonal edge is about WHICH MERGE LEVELS a lane must wait for,
            // and says nothing about the rest slice; so the rest stays a hard
            // dependency and is waited on unconditionally. That keeps the edge
            // (the tree no longer has to drain) without inventing an overlap the
            // invariant never licensed.
            if (pipe25d() && restEvValid && restEv)
                CUDA_CHECK(cudaStreamWaitEvent(dst, restEv, 0));
            if (pipe25d() && !pipeOK) {
                // No usable previous-panel Stage-2 events (first panel, or the
                // previous panel ran c_L=1 and never called applyQt_25d). The panel
                // loop skipped its blanket wait because the gate is on, so the
                // conservative barrier has to be reinstated HERE or the lane races
                // the previous panel's trailing update.
                CUDA_CHECK(cudaStreamWaitEvent(dst, evPrePanel, 0));
            }
            // LOCAL APPLY (31:329). The lane was released before epoch k-1's levels
            // above releaseLevel were globally applied, so it applies the ones that
            // touch ITS rows itself, from the replicated compact state. m2_max_level
            // stopped the global near apply at the same boundary, so every node is
            // applied exactly once. A node is owned by the lane whose rows contain a
            // band of it; at D=3 the single skipped node (0,4) has its top band at
            // pv_row0, below this epoch's first row, so exactly one lane owns it.
            if (pipeMode == 3 && pipeOK && pv_lv > pv_applied) {
                const int r0 = row0 + slabStart[j], r1 = r0 + slabSize[j];
                int own = 0;
                for (int nd = 0; nd < pv_nm && !own; ++nd) {
                    const int bb = pv_row0 + pv_slabStart[pv_nodes[nd].y];
                    if (bb + NB > r0 && bb < r1) own = 1;
                }
                if (own)
                    mergeApplyLevels(A, lda, pv_row0, k * NB, NB, dst,
                                     laneW1[j], laneW2[j], pv_cL, pv_slabStart,
                                     Vm, TmM, pv_nm, pv_nodes,
                                     pv_applied, pv_lv);
            }
            if (pipeOK) {
                ++mv_pipe_lanes;
                // A lane released before the previous panel's tree drained is one
                // "cluster on a successive epoch" -- rung 2's Pipeline move in
                // 31:429's table, and the only one of the four that the diagonal
                // edge (not the lane structure) provides.
                if (laneRelease[j] < 0) {
                    CUDA_CHECK(cudaStreamWaitEvent(dst, evS1done, 0));
                    ++mv_pipe_released; ++mv_pipeline[2];
                } else {
                    CUDA_CHECK(cudaStreamWaitEvent(dst, evM2lvl[laneRelease[j]], 0));
                    if (laneRelease[j] < last_m2_levels - 1) {
                        ++mv_pipe_released; ++mv_pipeline[2];
                    }
                }
            }
            if (j >= lanesPerWave) {
                const int prev0 = ((j / lanesPerWave) - 1) * lanesPerWave;
                const int prev1 = std::min(cL, prev0 + lanesPerWave);
                for (int q = prev0; q < prev1; ++q)
                    CUDA_CHECK(cudaStreamWaitEvent(dst, evDom[q], 0));
            }
            if (laneDsmOK) {
                // The lane runs the SAME carrier the c_L=1 path uses: a C-block
                // hardware cluster reducing through DSM instead of through global
                // Pp/Psum. This is what makes replication compose with the rung-1
                // move rather than opt out of it -- panel() reaches its DSM branch
                // only when c_L == 1, so until now every replicated lane silently
                // fell back to the older gmem domino.
                da.rpb = dsmRpb[j];
                int* pf = pipeF + j * 32;          // pipeF is 512 ints; 32 per lane
                CUDA_CHECK(cudaMemsetAsync(pf, 0, 32 * sizeof(int), dst));
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(dsmC[j] * cdiv(dsmC[j] + dsmU, dsmC[j]));
                cfg.blockDim = dim3(32 * 16);      // one warp per live minipanel column
                cfg.dynamicSmemBytes = (size_t)dsmSmem(dsmRpb[j]);
                cfg.stream = dst;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim = {(unsigned)dsmC[j], 1u, 1u};
                cfg.attrs = at; cfg.numAttrs = 1;
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_panel_dsm<Real, NB>,
                                              da, pf, dsmC[j]));
            } else if (overflow)
                k_panel_domino<Real, NB, 16, false, true, true>
                    <<<nbl, dtpb, domSmem(rpb, db), dst>>>(da, gbarL[j]);
            else
                k_panel_domino<Real, NB, 16, false, true>
                    <<<nbl, dtpb, domSmem(rpb, db), dst>>>(da, gbarL[j]);
            CUDA_CHECK(cudaEventRecord(evDom[j], dst));
        }
        // Sync: the merge stream waits for every replicated leaf.
        for (int j = 0; j < cL; ++j)
            CUDA_CHECK(cudaStreamWaitEvent(st, evDom[j], 0));
        NVTX_POP();
        // Phase 2: extract per-lane R into Pbuf for the merge
        NVTX_RANGE("stitch");
        const int mpad2d = cL * SLAB;
        k_extract_R_lanes<Real, NB><<<cL * cdiv(NB * NB, TPB), TPB, 0, st>>>(
            Acol, lda, base, cL, mk, Pbuf, mpad2d, SLAB);
        // Phase 3: cross-lane ⊕ (Householder-on-stack, spec §3/§14 — validated via tree path)
        // A merge of cL leaves has at most cLmax-1 internal nodes, known at compile
        // time, so this is a stack array rather than a std::vector: the vector meant
        // a heap allocation and free on the panel path, 256 of each per
        // factorization at 16384^2, for a list that is never longer than seven.
        int2 nodes_cL[cLmax];
        int nm = 0;
        for (int step = 1; step < cL; step *= 2)
            for (int i = 0; i + step < cL; i += 2 * step)
                nodes_cL[nm++] = {i, i + step};
        last_nm = nm;
        for (int i = 0; i < nm; ++i) last_nodes[i] = nodes_cL[i];
        if (nm > 0) {
            // No upload here: row c_L of the device table already holds exactly
            // these nodes, in the same bottom-up level order the loop below walks.
            // The host-side nodes_cL is still built because applyQt_25d and the
            // orgqr save path read last_nodes.
            const int2* nodesDev = d_nodes2d + size_t(cL) * kNodeStride;
            // Level table: lvl[l] = (first node index, node count) for tree level l,
            // bottom-up. Built once here and consumed by whichever path runs below;
            // the ORDER is the A6 obligation and is identical either way.
            int2 lvl[16]; int nlv = 0, off = 0;
            for (int step = 1; step < cL; step *= 2) {
                int cnt = 0;
                for (int i = 0; i + step < cL; i += 2 * step) ++cnt;
                lvl[nlv++] = {off, cnt};
                off += cnt;
            }
            // ---- the K*D -> K+D step, at the level of one panel's tree ----
            // The per-level launch loop below costs D kernel launches per panel, and
            // a launch is a pipeline drain: at 16384^2 that is 256 panels x D drains
            // charged to the merge, which is the term that decides whether
            // replication is worth doing at all (31:345 -- CP is Theta(K+D), and the
            // per-level loop is the K*D form).
            //
            // k_wave_top (:963) already solves exactly this and is validated on the
            // reconstruction tree path at :7264 -- one cooperative launch that walks
            // levels l0..nlv-1 internally with grid.sync(), its own step count being
            // TSTEPS = (nlv-l0) + NB/16 - 1, the K+D-1 wavefront form. panel_25d
            // simply never adopted it. Mirrors the launch at :7287.
            //
            // Residency: grid.sync() requires every block resident or the launch
            // fails with CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (and a silent
            // overrun hangs). Grid here is nm <= cLmax-1 = 7 blocks against 132 SMs,
            // so it is safe by construction -- asserted, not assumed.
            const bool waveOK = waveTopOK && nlv > 1 && nm <= smCount;
            if (waveOK) {
                WaveTopArgs<Real> wa;
                wa.Pbuf = Pbuf; wa.mpad = mpad2d;
                wa.nodes = nodesDev;
                wa.nlv = nlv; wa.l0 = 0;
                for (int l = 0; l < nlv; ++l) wa.lvl[l] = lvl[l];
                wa.Vm = Vm; wa.taum = taum;
                void* args[] = { &wa };
                dim3 grid(nm), block(TPB);
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)k_wave_top<Real, SLAB, NB>, grid, block, args,
                    (size_t)waveTopSmem, st));
            } else {
                // D == 1 (c_L == 2): one level, so a cooperative launch is pure
                // overhead and the single k_merge_geqrf IS the wavefront.
                for (int l = 0; l < nlv; ++l)
                    k_merge_geqrf<Real, SLAB, NB><<<lvl[l].y, TPB, merge_smem<Real, NB>(), st>>>(
                        Pbuf, mpad2d, nodesDev + lvl[l].x, lvl[l].y, lvl[l].x, Vm, taum);
            }
            mv_merge_launches += waveOK ? 1 : nlv;
            mv_merge_levels   += nlv;
            NVTX_RANGE("recon");
            k_larft<Real, SLAB, NB, true><<<nm, TPB, smLarft(true), st>>>(
                nullptr, 0, Vm, taum, TmM, nm, 0);
            NVTX_POP();
        }
        NVTX_POP();
        // Phase 4a: copy root R from Pbuf[0] to A's top-NB rows (upper triangle only —
        // the lower triangle holds V_0 from slab 0's domino, which must be preserved).
        // Root R is in Pbuf at rows [0,NB), cols [0,NB), leading dimension mpad2d.
        // A's panel column is Acol[r + c*lda] for r in [0,NB), c in [0,NB).
        k_copy_upper_tri<Real, NB><<<cdiv(NB * NB, TPB), TPB, 0, st>>>(
            Acol, lda, Pbuf, mpad2d);
        // Save per-panel merge data for orgqr Q reconstruction.
        // Vm/TmM/domTpanL are overwritten by the next panel; snapshot them now.
        // W6(G7) Phase-2: skipped in the LRQR_ONLY production path (saveForOrgqr
        // == false) — orgqr is never called there, so the saves are pure waste.
        // 2.5D WITNESS, rung 1 -- counted here because this is where the lane
        // structure is final and in scope. The lanes are shared->DSM lanes: laneCap
        // is min(cLmax, sched.rung[1].c), so replication is bounded by that rung's
        // capacity-derived replica count. Split = the cL disjoint row intervals,
        // Replicate = the cL lanes each holding panel state, Combine = the nm ordered
        // merge-tree nodes. Host counters, unconditional (NOT inside the
        // saveForOrgqr guard -- the moves happen whether or not orgqr will be called).
        // WHICH RUNG DO THE LANE MOVES BELONG TO? It depends on what a lane IS.
        //
        // With the lanes on the gmem domino, the lane boundary is the only replica
        // structure there is, so it is credited to rung 1 (shared->DSM) as before.
        //
        // With the lanes on the DSM carrier, there are TWO nested boundaries and
        // crediting both to rung 1 hides one of them -- which is precisely why rung
        // 2 has been reading as "no carrier":
        //   rung 1, shared->DSM : the C-block hardware cluster INSIDE one lane.
        //       Split = the C row slices; Replicate = each block publishing its
        //       partial for DSM peer reads; Combine = the cross-block reduction;
        //       Pipeline = the NB/PW minipanel generations.
        //   rung 2, DSM<->L2    : the c_L lanes THEMSELVES. 31:429 names this rung's
        //       carriers as "cluster output slots; a constant number of compact
        //       global slots; device-scope reduction slots; clusters on successive
        //       epochs" -- which is exactly each lane's R block staged in Pbuf, the
        //       cross-lane merge tree reducing over them, and the diagonal edge.
        // So the carrier for rung 2 is not missing; it appears the moment a lane
        // becomes a cluster, and the counters simply have to be attributed to the
        // boundary the structure actually sits on.
        if (laneDsmOK) {
            long long cblocks = 0;
            for (int j = 0; j < cL; ++j) cblocks += dsmC[j];
            mv_split[1]     += cblocks;                 // row slices within clusters
            mv_replicate[1] += cblocks;                 // partials published for DSM peers
            for (int j = 0; j < cL; ++j)
                mv_combine[1] += std::max(0, dsmC[j] - 1);   // cross-block reduction
            mv_pipeline[1]  += (long long)cL * std::max(0, NB / PW - 1);
            mv_split[2]     += cL;                      // cluster output slots
            mv_replicate[2] += cL;                      // compact global R slots in Pbuf
            mv_combine[2]   += nm;                      // device-scope reduction tree
            mv_instrumented[1] = 1;
            mv_instrumented[2] = 1;
        } else {
            mv_split[1]     += cL;
            mv_replicate[1] += cL;
            mv_combine[1]   += nm;
        }
        // PIPELINE for the minipanel generations, booked ONCE, at the boundary the
        // overlap actually crosses. Within each lane the domino runs NB/PW
        // generations and the look-ahead overlaps generation g's trailing update
        // with g+1's factorization, so the number of OVERLAPS is (NB/PW - 1) per
        // lane. Where that overlap lives depends on the carrier, and 31:429 keeps
        // the two apart: rung 0's Pipeline is "async copy/MMA stages", rung 1's is
        // "cluster wavefront". On the DSM carrier the generations advance by
        // cluster.sync with DSM peer reads, so the overlap is rung 1's and was
        // credited above; on the gmem domino it stays at the register/shared
        // boundary. Crediting both would count one physical overlap twice.
        if (!laneDsmOK)
            mv_pipeline[0]  += (long long)cL * std::max(0, NB / PW - 1);
        if (saveForOrgqr) {
            const size_t perPan = size_t(cLmax) * NB * NB;
            CUDA_CHECK(cudaMemcpyAsync(domTpanL_pan + size_t(k) * perPan, domTpanL,
                                       size_t(cL) * NB * NB * sizeof(Real),
                                       cudaMemcpyDeviceToDevice, st));
            if (nm > 0) {
                CUDA_CHECK(cudaMemcpyAsync(Vm_pan + size_t(k) * perPan, Vm,
                                           size_t(nm) * NB * NB * sizeof(Real),
                                           cudaMemcpyDeviceToDevice, st));
                CUDA_CHECK(cudaMemcpyAsync(TmM_pan + size_t(k) * perPan, TmM,
                                           size_t(nm) * NB * NB * sizeof(Real),
                                           cudaMemcpyDeviceToDevice, st));
            }
            cL_pan[k] = cL;  nm_pan[k] = nm;
            for (int j = 0; j < cL; ++j) {
                slabStart_pan[k * cLmax + j] = slabStart[j];
                slabSize_pan[k * cLmax + j]  = slabSize[j];
            }
            for (int i = 0; i < nm; ++i) nodes_pan[k * cLmax + i] = nodes_cL[i];
        }
    }

    // ------------------------------------------------------------ panel factorization
    void panel(Real* A, size_t lda, int k, Real* SPV, int spvcol, cudaStream_t st) {
        const PTab& t = tab[k];
        const int row0 = k * NB, mk = m - row0, N = t.N, mpad = N * SLAB;
        Real* Acol = A + size_t(k) * NB * lda + row0;
        if (useDomino) {
            // ---- c_L-lane dispatch (spec §7.2) ----
            // Accuracy-latency dial from the measured shared→DSM rung.  A lane may
            // carry fanout^(phaseDepth+2) row states before it adds an unabsorbed
            // reduction level.  Replicate just enough lanes to keep each lane within
            // that depth, rounded up to the binary Householder-on-stack merge tree and
            // capped only by the capacity-derived replica count.  There is no size,
            // aspect, or precision dispatch here: all inputs are generated schedule
            // parameters or live work counts.
            const int rowStates = cdiv(mk, SLAB);
            const int fanout = std::max(2, sched.rung[1].r);
            const int phaseDepth = fanoutDepth(std::max(1, sched.rung[1].S), fanout);
            // MEASUREMENT KNOB (LRQR_LANEDEPTH), default = the shipped +2. This
            // exponent is the ONLY un-derived number in the lane formula and it is
            // exactly the replication dial: a smaller exponent packs fewer row states
            // per lane, so more lanes are needed -> shorter per-lane Householder chain
            // (less synchronization depth) but a deeper ordered merge tree (more
            // Combine). That is the 2.5D walltime/sync trade in one integer, and it is
            // swept before any law is fitted to it. Read once, never a shipping
            // dispatch: with the variable unset the arithmetic is bit-identical.
            static const int laneDepthBias = [] {
                const char* e = getenv("LRQR_LANEDEPTH");
                return e ? atoi(e) : 2;
            }();
            size_t statesPerLane = 1;
            for (int d = 0; d < phaseDepth + laneDepthBias && statesPerLane < (size_t)rowStates; ++d)
                statesPerLane = std::min((size_t)rowStates, statesPerLane * (size_t)fanout);
            const int laneNeed = cdiv(rowStates, (int)statesPerLane);
            int cL = 1;
            // Slab-split floor, stated where c_L is chosen rather than left implicit.
            // panel_25d divides mk rows as base = max(32, (mk/cL/32)*32) for the
            // first cL-1 lanes and mk - base*(cL-1) for the last, so once mk/cL < 32
            // the floor of 32 wins and the last lane's size would go negative.
            //
            // panel_25d ALREADY prevents that, at :6672 --
            //     while (cL > 1 && mk < cL * NB) cL /= 2;
            // -- which is strictly tighter (it demands mk/cL >= NB = 64, not 32), so
            // this cap never binds and is documentation plus a floor for any future
            // caller that reaches the split without going through :6672. Recorded
            // because the earlier reading of a c_L=8 accuracy break as a negative
            // slab was WRONG: that run's binary was stale (the PW compile errors
            // meant old objects were reused), and the first clean build shows healthy
            // residuals at c_L=8 on every square cell.
            const int slabCap = std::max(1, mk / 32);
            const int laneCap = std::max(1, std::min({(int)cLmax, sched.rung[1].c, slabCap}));
            while (cL < laneNeed && cL < laneCap) cL <<= 1;
            cL = std::min(cL, laneCap);
            // THE SELECTION, once the cost model has constants. c_L becomes the
            // argmin of T_panel(c) = A*(rowStates/c) + B*ceil(log2 c) + C*(c-1)
            // over the legal set {1,2,...,laneCap} -- wall time first, with the
            // ordered merge (synchronization) and replica setup (communication)
            // priced against it rather than assumed away. rowStates is the LIVE
            // per-panel count, so c* is a trajectory down the matrix.
            //
            // if constexpr, so while kReplCostCalibrated is false this compiles to
            // nothing at all and the exponent rule above is bit-identically what
            // ships. An uncalibrated cost model does not get to touch a decision
            // here: that is precisely how the epsilon/delta certification stayed
            // RED for five sessions.
            // TWO-CAP LAW (job 9526, 48-point aspect x rowStates grid):
            //     c* = min(rowStates/1024, 4096/n)
            // rowStates alone is a function of m (~m/128) and shipped a regression
            // -- 262144x8192 fell to 0.83x -- because the fit that produced it had
            // one cell per rowStates and could not see n. The second cap is the
            // Stage-2 apply bound: mergeApplyQt runs on c*NB rows by the trailing
            // width, so its cost grows like c*n and bounds c ~ 1/n. 12 of 12 grid
            // cells, and it returns c=1 on exactly the two cells that regressed.
            if constexpr (kTwoCapCalibrated) {
                cL = std::max(1, std::min(replicaTwoCap(rowStates, n, laneCap), laneCap));
            }
            if constexpr (kReplCostCalibrated) {
                // JOINT selection, not two heuristics. c and p_r are chosen together
                // under c*(p_r+U) <= P -- assumption (A5), docs/10 §7.1's
                // p_r p_c c <= P_lambda, instantiated on one GPU with P = smCount.
                // p_r is the DSM cluster breadth (reduction depth INSIDE a lane, so
                // part of D_lambda, and its DSM reduction is what keeps that traffic
                // off L2/HBM); c is the replica count and the cross-lane tree depth.
                // They spend one processor budget, so S_lambda = Theta(K+D) and
                // Q_lambda = Theta(B_lambda(c)) cannot be satisfied by optimizing
                // either alone -- which is why this is one argmin over the legal
                // family (31:486) rather than a c law and a separate cluster rule.
                // c ONLY, for now. replicaGridArgmin also returns p_r and would
                // drive the DSM cluster breadth through gridPr -- but that path runs
                // through LRQR_LANEDSM, which gate 9499 shows is defective, so
                // enabling it here would ship an untested change under the cover of a
                // calibrated c. The joint argmin stays available and unused until the
                // DSM lane path passes the c-matrix; see task #8.
                cL = replicaArgmin(rowStates, laneCap, kReplCost);
            }
            // ---- LIVENESS FLOOR (Theorem 31:441) ----
            // "executes AT LEAST TWO REPLICA LANES and two pipeline generations when
            //  the matrix has nonzero work" -- that is the theorem's HYPOTHESIS, so
            // c_L = 1 leaves every conclusion about four-move liveness inapplicable
            // to this build. The shipped exponent rule gives c_L = 1 on essentially
            // everything (c_L>=2 only for n <= ~482 or m in the millions), and its
            // n<=482 cliff is an artifact of fanoutDepth(cdiv(n,nu1),16) with no
            // basis in the spec's c selection.
            //
            // c_L on ONE GPU is not a communication lever: 31:230 says compact-state
            // replication at on-chip rungs "keeps the mechanism live but does not
            // claim the full-operand 1/sqrt(c) volume reduction". Its job is
            // liveness, at O(nb) storage and O(nbL) work, which 31:441 proves is
            // lower-order. Choosing c by argmin of wall time was optimising a
            // quantity the spec fixes -- which is why every fitted c-law came out
            // arbitrary and needed a compensating cap.
            //
            // Raised only where LEGAL: laneCap already carries min(cLmax, rung[1].c,
            // slabCap), and the mk < c*NB guard below still applies. Never above
            // legality -- 31:230 is explicit that c is optimized, not maximized.
            // The knob is for A/B measurement only; default is the spec's floor.
            static const bool liveFloor = [] {
                const char* e = getenv("LRQR_LIVENESS"); return !e || atoi(e) != 0;
            }();
            if (liveFloor && laneCap >= 2 && mk >= 2 * NB) cL = std::max(cL, 2);
            // Route lock: an earlier panel of this super-panel took the Tsp route
            // with a far region outstanding, so the compound is live and this panel
            // must join it. Switching to c_L>1 here would set any25d and suppress the
            // combined far that the earlier panel depends on. See far25Route.
            if (sp_c1Locked) cL = 1;
            // MEASUREMENT INSTRUMENT (LRQR_CL) -- the reason LANEDEPTH is not enough.
            // On this device fanout = rung[1].r = 16 and laneCap = min(cLmax,
            // rung[1].c) = 8, so c_L can only ever take {1,2,4,8} while ONE step of
            // laneDepthBias moves statesPerLane by x16 -- four doublings of c_L at a
            // time. The bias sweep therefore steps straight over the interior of the
            // feasible set: measured, 1048576x64 reads c_L = 8,8,8,8,1 across bias
            // 0..4 and 16384^2 reads 4,1,1,1,1. No choice of cells repairs that,
            // because the step size, not the cell, is what is too coarse. Fitting
            //     T_panel(c) = A*(rowStates/c) + B*ceil(log2 c) + C*(c-1)
            // needs c varied one doubling at a time, so this knob sets c_L directly
            // and every cell -- squares included -- yields the full four-point T(c).
            // Rounded DOWN to a power of two (the merge tree is binary) and clamped
            // to laneCap so the request can never exceed the capacity-derived replica
            // count. Read once; unset the arithmetic above is bit-identical, so this
            // is an instrument and never a shipping dispatch (G6).
            static const int cLForce = [] {
                const char* e = getenv("LRQR_CL");
                return e ? atoi(e) : 0;
            }();
            if (cLForce > 0) {
                int f = 1;
                while (f * 2 <= cLForce) f *= 2;
                cL = std::max(1, std::min(f, laneCap));
            }
            if (cL > 1) {
                cLused = std::max(cLused, cL);
                panel_25d(A, lda, k, SPV, spvcol, st, cL);
                return;
            }
            // No 2.5D Stage-2 events will be recorded for this panel, so invalidate
            // the previous panel's: a later panel must not release its lanes against
            // events describing a panel that is no longer its predecessor.
            last_row0 = -1; last_m2_levels = 0;
            // c_L=1: one lane, so Split/Replicate/Combine at rung 1 are 1/1/0 --
            // recorded rather than left blank, because "one lane" and "not measured"
            // are different states and only one of them is a liveness failure.
            mv_split[1]     += 1;
            mv_replicate[1] += 1;
            mv_pipeline[0]  += std::max(0, NB / PW - 1);
            last_cL = 1;  // c_L=1 path: no merge, standard applyQt
            DomArgs<Real> da;
            da.A = Acol; da.lda = lda; da.mk = mk; da.row0 = row0;
            da.SPV = SPV; da.ldspv = m; da.spvcol = spvcol;
            da.Tpan = Tpan + size_t(k) * NB * NB;
            da.tauT = domScal; da.fv = domScal + NB; da.bet = domScal + 2 * NB;
            da.sig = domScal + 3 * NB;
            da.dslot = domSlot; da.prow = domProw; da.Pp = domPp; da.Psum = domPsum;
            da.sigD = domSigD; da.dslotD = domDslotD; da.dcolP = domDcolP; da.dcolStride = 132;
            da.T16g = domT16;
            da.tphase = getenv("LRQR_PHASET") ? domT : nullptr;
            da.nslab = 0;  // set per-dispatch for k_panel_domino paths
            int dtpb = sizeof(Real) == 8 ? 256 : 512;   // fp64: 256 threads → 2 blocks/SM (128 regs each); fp32: 512 → 1 block
            if (const char* e = getenv("LRQR_DOMTPB")) dtpb = atoi(e);
            // pipelined domino: block 0 owns the whole minipanel on-chip, updaters
            // trail on doorbells — no per-column cross-block sync at all.
            // Pipe wins only while its complete minipanel fits within one
            // double-buffered cluster row-state plus the look-ahead panel.
            int pipeMax = clRpbMaxDB + NB; // double-buffered row-state + one look-ahead panel
            // G2-PANEL V13: gmem-domino capacity-ladder rung. mk > smem-variant max
            // G6 takeover: the LRQR_GMEM_DOMINO(_RPB) experimental rung is REMOVED.
            // It measured 2.1× SLOWER than k_panel_dsm at fp64@4096 (slurm 7792,
            // GPU-35469031) — the gmem atomic barrier (~1µs) is ~5× slower than
            // cl.sync and fewer blocks cut parallelism 3.3×. cuSOLVER's
            // geqr2_gmem_domino<9> wins by batching 4 panels/launch (a NEW kernel
            // class the tower lacks — the G2 batched-panel workstream), NOT by the
            // gmem barrier. A parked-off env mechanism is exactly the pattern G6
            // kills: the finding lives in G2_PANEL_V13_REPORT.md + this comment;
            // the k_panel_domino<NOCOOP> gbar path itself remains live below as the
            // mk>dsmRowMax capacity rung (its honest home).
            // DSM-pipe (ROADMAP §6): one small coop+cluster kernel per panel —
            // C-block DSM col cluster + U in-kernel updaters; far waves fill the rest.
            if (useDsm && mk > std::min(pipeMax, (int)pipeRowMax) && mk <= dsm2RowMax) {
                // Cluster breadth is generated from the shared→DSM replica capacity:
                // the floor exposes every replica lane plus the register fanout, while
                // sqrt(m/(PW+c+pad)) grows continuously to Hopper's 16-block cap.
                const int replicaCap = std::max(1, sched.rung[1].c);
                const int clusterQuantum = 16 + replicaCap + 1; // minipanel + replicas + pad
                const int clusterFloor = replicaCap + std::max(1, sched.rung[0].r);
                int C = std::min(16, std::max(clusterFloor, (int)std::ceil(
                    std::sqrt(double(m) / double(clusterQuantum)))));
                int rpb = 32 * cdiv(cdiv(mk, C), 32);   //   per block in phase 2
                C = cdiv(mk, rpb);
                const int G = C * cdiv(C + dsmU, C);
                da.rpb = rpb;
                CUDA_CHECK(cudaMemsetAsync(pipeF, 0, 32 * sizeof(int), st));
                cudaLaunchConfig_t cfg = {};
                int dsmTpb = 32 * 16; // one warp per live minipanel column
                cfg.gridDim = dim3(G); cfg.blockDim = dim3(dsmTpb);
                cfg.dynamicSmemBytes = (size_t)dsmSmem(rpb); cfg.stream = st;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim = {(unsigned)C, 1u, 1u};
                cfg.attrs = at; cfg.numAttrs = 1;
                NVTX_RANGE("chain_step");
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_panel_dsm<Real, NB>,
                                              da, pipeF, C));
                NVTX_POP();
                return;
            }
            // de-fused domino (ROADMAP §5): small column-loop kernels alternate with
            // cuBLAS tensor-core boundary GEMMs; far GEMMs genuinely overlap.
            if (useDefuse && (green || mk > std::min(pipeMax, (int)pipeRowMax))) {
                cublasHandle_t cbp = (st == s1) ? cb1 : cb0;
                // boundary GEMMs feed T̃/V (latency carrier) → fp32-accurate compute
                // (TF32 here costs ~100× orthogonality; verified). 3×TF32 is the
                // fp32-accurate fast tier if these ever bind (spec §13.4).
                // env gate LRQR_DEFUSE_TF32 removed (AUDIT_V6 §6 / Theorem U); defuse
                // is never selected (useDefuse=false) → btf32 dead code, parameter→0.
                const bool btf32 = false;
                int colsNbl = 16;
                if (const char* e = getenv("LRQR_COLSNBL")) colsNbl = atoi(e);
                if (green) colsNbl = std::min(colsNbl, smallSMs);
                for (int s = 0; s < NB / 16; ++s) {
                    NVTX_RANGE("chain_step");
                    const int p = s * 16, pe = p + 16, hs = mk - p, w2 = NB - pe;
                    // under green the cols kernel runs on the reserved partition
                    // (sCols/ctxS), event-chained with the big-partition boundaries
                    cudaStream_t sc = green ? sCols : st;
                    if (green) {
                        CUDA_CHECK(cudaEventRecord(evC0, st));
                        CUDA_CHECK(cudaStreamWaitEvent(sCols, evC0, 0));
                        cuCtxPushCurrent(ctxS);
                    }
                    int rpb = 32 * cdiv(cdiv(hs, colsNbl), 32);
                    // clusters cannot be placed inside a green partition (hangs)
                    if (!green && colsNbl <= 16 && rpb <= colsRpbMax) {
                        const int nbl = cdiv(hs, rpb);
                        da.rpb = rpb;
                        cudaLaunchConfig_t cfg = {};
                        cfg.gridDim = dim3(nbl); cfg.blockDim = dim3(512);
                        cfg.dynamicSmemBytes = (size_t)colsSmem(rpb); cfg.stream = sc;
                        cudaLaunchAttribute at[1];
                        at[0].id = cudaLaunchAttributeClusterDimension;
                        at[0].val.clusterDim = {(unsigned)nbl, 1u, 1u};
                        cfg.attrs = at; cfg.numAttrs = 1;
                        CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                            k_dom_cols<Real, NB, 16, true>, da, s, domUg, domGs));
                    } else {                            // small cooperative grid
                        rpb = std::min(rpb, (int)colsRpbMax);
                        const int nbl = cdiv(hs, rpb);
                        da.rpb = rpb;
                        int s_ = s;
                        void* cargs[] = { &da, &s_, &domUg, &domGs };
                        CUDA_CHECK(cudaLaunchCooperativeKernel(
                            (void*)k_dom_cols<Real, NB, 16, false>, dim3(nbl), dim3(512),
                            cargs, (size_t)colsSmem(rpb), sc));
                    }
                    if (green) {
                        CUDA_CHECK(cudaEventRecord(evC1, sCols));
                        cuCtxPopCurrent(nullptr);
                        CUDA_CHECK(cudaStreamWaitEvent(st, evC1, 0));
                    }
                    // v2: GEMMs on RAW A (no ũ materialization); k_dom_small corrects
                    // the 16 triangle rows, k_dom_fix undoes the raw-apply excess.
                    Real* Ws = domPsum + size_t(s) * NB * 16;
                    const Real* rawU = Acol + p + size_t(p) * lda;
                    gemm(cbp, btf32, CUBLAS_OP_T, CUBLAS_OP_N, 16, NB, hs,
                         Real(1), rawU, (int)lda, Acol + p, (int)lda, Real(0), Ws, 16);
                    k_dom_small<Real, NB><<<1, 256, 0, st>>>(da, s, domGs, Ws, domW2);
                    if (w2 > 0) {
                        gemm(cbp, btf32, CUBLAS_OP_N, CUBLAS_OP_N, hs, w2, 16,
                             Real(-1), rawU, (int)lda, domW2, 16, Real(1),
                             Acol + p + size_t(pe) * lda, (int)lda);
                        k_dom_fix<Real, NB><<<1, 256, 0, st>>>(da, s, domGs, domW2);
                    }
                    NVTX_POP();
                }
                k_dom_tail<Real, NB><<<33, 256, 0, st>>>(da);
                return;
            }
            if (usePipe && mk <= std::min(pipeMax, (int)pipeRowMax)) {
                da.rpb = 0;
                int ptpb = 512;   // 16 warps: 2D warp map needs NW>=15 (NB/PW+1) jobs
                if (const char* e = getenv("LRQR_PIPETPB")) ptpb = atoi(e);
                CUDA_CHECK(cudaMemsetAsync(pipeF, 0, 32 * sizeof(int), st));
                void* pargs[] = { &da, &pipeF };
                const int pn = green ? std::min(pipeNbl, fusedGrid) : pipeNbl;
                NVTX_RANGE("chain_step");
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)k_panel_domino_pipe<Real, NB>, dim3(pn), dim3(ptpb),
                    pargs, (size_t)pipeSmem(mk), st));
                NVTX_POP();
                return;
            }
            // cluster mode: whole panel as ONE ≤16-block cluster → hw barrier (~0.23µs
            // vs ~1.0µs coop grid.sync). Ceiling: smem 32·rpb+18840 floats ≤ 227KB.
            int clmax = 4096;
            if (const char* e = getenv("LRQR_CLMAX")) clmax = atoi(e);
            if (mk <= clmax && mk <= clRowMax) {
                const int rpb = clRpb(mk), nbl = cdiv(mk, rpb);
                da.rpb = rpb;
                da.nslab = nbl; noteRung0(da.nslab);
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(nbl); cfg.blockDim = dim3(dtpb);
                cfg.dynamicSmemBytes = (size_t)domSmem(rpb); cfg.stream = st;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim = {(unsigned)nbl, 1u, 1u};
                cfg.attrs = at; cfg.numAttrs = 1;
                NVTX_RANGE("chain_step");
                cudaError_t err = cudaLaunchKernelEx(&cfg,
                    k_panel_domino<Real, NB, 16, true>, da, (unsigned*)nullptr);
                NVTX_POP();
                if (err == cudaSuccess) return;
                (void)cudaGetLastError();       // cluster didn't fit → cooperative path
            }
            int rpb = domRpb(mk, false), nbl = cdiv(mk, rpb);
            const bool overflow = (nbl > smCount);
            // Once the capacity-sized single-buffer grid exceeds residency, keep
            // exactly one CTA resident per SM and give it a contiguous logical row
            // interval.  The clRpbMax prefix remains on chip across every column;
            // only the formula-derived overflow suffix is updated in compact A.
            // This replaces the old niter>1 ping-pong path (which reloaded the full
            // 16-column state once per Householder column) without an aspect/size
            // gate: engagement is precisely the shared-memory capacity overflow.
            if (overflow && gbar) {
                const int R = smCount;
                rpb = clRpbMax;
                da.rpb = rpb;
                da.nslab = R; noteRung0(da.nslab);
                if (getenv("LRQR_DEBUG_LAUNCH"))
                    fprintf(stderr, "[domino-persist] mk=%d rpb=%d logical=%d R=%d spill=%d smem=%d\n",
                            mk, rpb, cdiv(mk, R), R, std::max(0, cdiv(mk, R) - rpb),
                            (int)domSmem(rpb, false));
                NVTX_RANGE("chain_step");
                k_panel_domino<Real, NB, 16, false, true, true>
                    <<<R, dtpb, domSmem(rpb, false), st>>>(da, gbar);
                NVTX_POP();
                return;
            }
            da.rpb = rpb;
            da.nslab = nbl; noteRung0(da.nslab);
            // Non-cooperative all-resident path.  Capacity overflow has already
            // been absorbed by the persistent-row carrier above, hence nbl is a
            // residency-safe grid and the gmem barrier cannot time-slice.  The
            // former FORCE_COOP/DOM_R environment gates are removed: production
            // geometry is generated solely from the same capacity calculation.
            if (gbar) {
                const int R = nbl;
                if (getenv("LRQR_DEBUG_LAUNCH"))
                    fprintf(stderr, "[domino] mk=%d rpb=%d nbl=%d R=%d niter=%d dtpb=%d db=%d smem=%d\n",
                            mk, rpb, nbl, R, 1, dtpb, 0, (int)domSmem(rpb, false));
                NVTX_RANGE("chain_step");
                k_panel_domino<Real, NB, 16, false, true><<<R, dtpb, domSmem(rpb, false), st>>>(
                    da, gbar);
                NVTX_POP();
                return;
            }
            {   // cooperative launch — may fail if nbl > max cooperative grid
                // (very tall panels where smem caps rpb, making nbl > smCount×blocks/SM).
                // Fall through to multi-kernel path on failure (same pattern as cluster fallback above).
                // Cooperative is always niter=1 → single-buffer smem (recompute best rpb).
                int rpb_co = domRpb(mk, false);
                int nbl_co = cdiv(mk, rpb_co);
                da.rpb = rpb_co; da.nslab = nbl_co;
                void* args[] = { &da, &gbar };
                NVTX_RANGE("chain_step");
                cudaError_t err = cudaLaunchCooperativeKernel((void*)k_panel_domino<Real, NB>,
                    dim3(nbl_co), dim3(dtpb), args, (size_t)domSmem(rpb_co, false), st);
                NVTX_POP();
                if (err == cudaSuccess) return;
                (void)cudaGetLastError();
            }
        }
        if (useFused && (int)t.lv.size() <= 20) {
            PanelArgs<Real> pa;
            pa.A = Acol; pa.lda = lda; pa.row0 = row0; pa.mk = mk; pa.mpad = mpad; pa.N = N;
            pa.Pbuf = Pbuf; pa.QT = QT; pa.QS = QS; pa.Vm = Vm; pa.TmM = TmM; pa.TmL = TmL;
            pa.tauL = tauL; pa.taum = taum;
            pa.nodes = d_nodes + t.nodesOff;
            pa.nlv = (int)t.lv.size();
            for (int l = 0; l < pa.nlv; ++l)
                pa.lvl[l] = {t.lv[l].first - t.nodesOff, t.lv[l].second};
            pa.Sg = Sg + size_t(k) * NB;
            pa.Tpan = Tpan + size_t(k) * NB * NB;
            pa.SPV = SPV; pa.ldspv = m; pa.spvcol = spvcol;
            void* args[] = { &pa };
            const int nmg = t.lv.empty() ? 0 : (t.lv.back().first - t.nodesOff + t.lv.back().second);
            int g = fusedGrid - 24;                 // headroom for concurrent update GEMMs
            if (nmg > g) g = fusedGrid;             // wavefront must fit; sacrifice overlap
            dim3 grid(g), block(TPB);
            NVTX_RANGE("chain_step");
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)k_panel_fused<Real, SLAB, NB>, grid, block, args,
                (size_t)fusedSmem, st));
            NVTX_POP();
            return;
        }
        k_load_panel<Real><<<cdiv(mpad * NB, TPB), TPB, 0, st>>>(Acol, lda, mk, NB, Pbuf, mpad, mpad);
        NVTX_RANGE("chain_step");
        k_leaf_geqrf<Real, SLAB, NB><<<N, TPB, bqr_smem<Real, NB>(SLAB), st>>>(Pbuf, mpad, tauL, N);
        NVTX_POP();
        NVTX_RANGE("recon");
        k_larft<Real, SLAB, NB, false><<<N, TPB, smLarft(false), st>>>(
            Pbuf, mpad, nullptr, tauL, TmL, N, 0);
        NVTX_POP();
        {   // batched lower levels; top-of-tree (≤63 nodes) as one small wavefront kernel
            const int nlvv = (int)t.lv.size();
            const int totalN = nlvv ? (t.lv.back().first - t.nodesOff + t.lv.back().second) : 0;
            int l0 = nlvv;
            if (waveTopOK && nlvv > 1 && nlvv <= 20)
                for (int l = 0; l < nlvv; ++l)
                    if (totalN - (t.lv[l].first - t.nodesOff) <= 63) { l0 = l; break; }
            if (l0 >= nlvv - 1) l0 = nlvv;         // need ≥2 wavefront levels to pay off
            for (int l = 0; l < l0; ++l) {
                int base = t.lv[l].first - t.nodesOff;
                k_merge_geqrf<Real, SLAB, NB><<<t.lv[l].second, TPB, merge_smem<Real, NB>(), st>>>(
                    Pbuf, mpad, d_nodes + t.lv[l].first, t.lv[l].second, base, Vm, taum);
            }
            if (l0 < nlvv) {
                WaveTopArgs<Real> wa;
                wa.Pbuf = Pbuf; wa.mpad = mpad;
                wa.nodes = d_nodes + t.nodesOff;
                wa.nlv = nlvv; wa.l0 = l0;
                for (int l = 0; l < nlvv; ++l)
                    wa.lvl[l] = {t.lv[l].first - t.nodesOff, t.lv[l].second};
                wa.Vm = Vm; wa.taum = taum;
                void* args[] = { &wa };
                dim3 grid(totalN - (t.lv[l0].first - t.nodesOff)), block(TPB);
                CUDA_CHECK(cudaLaunchCooperativeKernel(
                    (void*)k_wave_top<Real, SLAB, NB>, grid, block, args,
                    (size_t)waveTopSmem, st));
            }
        }
        const int nmerges = t.lv.empty() ? 0
            : (t.lv.back().first - t.nodesOff + t.lv.back().second);
        if (nmerges > 0) {
            NVTX_RANGE("recon");
            k_larft<Real, SLAB, NB, true><<<nmerges, TPB, smLarft(true), st>>>(
                nullptr, 0, Vm, taum, TmM, nmerges, 0);
            NVTX_POP();
        }
        k_set_identity<Real, NB><<<cdiv(NB * NB, TPB), TPB, 0, st>>>(QS);
        for (int l = (int)t.lv.size() - 1; l >= 0; --l) {
            int base = t.lv[l].first - t.nodesOff;
            k_dsweep<Real, NB><<<t.lv[l].second, TPB, 3 * NB * NB * sizeof(Real), st>>>(
                QS, TmM, Vm, d_nodes + t.lv[l].first, t.lv[l].second, base);
        }
        k_leaf_apply<Real, SLAB, NB><<<N, TPB, ((SLAB + 1) * NB + 2 * NB * NB) * sizeof(Real), st>>>(
            Pbuf, mpad, TmL, QS, QT, mpad, N);
        // BDGJNS reconstruction
        Real* Sk = Sg + size_t(k) * NB;
        k_lu_sign<Real, NB><<<1, TPB, (NB * NB + NB) * sizeof(Real), st>>>(QT, mpad, Sk);
        CUBLAS_CHECK(cublasSetStream(cb0, st));   // ensure trsm runs on st (race fix: cb0 may be on another stream from a prior applyQt)
        if (mpad > NB)
            trsm(cb0, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                 mpad - NB, NB, Real(1), QT, mpad, QT + NB, mpad);
        Real* Tk = Tpan + size_t(k) * NB * NB;
        k_negUS<Real, NB><<<cdiv(NB * NB, TPB), TPB, 0, st>>>(QT, mpad, Sk, Tk);
        trsm(cb0, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
             NB, NB, Real(1), QT, mpad, Tk, NB);
        k_writeYRV<Real, NB><<<cdiv(mk * NB, TPB), TPB, 0, st>>>(QT, mpad, Pbuf, mpad, Sk,
            A + size_t(k) * NB * lda, lda, row0, mk, SPV, m, spvcol);
    }

    // Apply Q^T (Y in SPV cols [c0,c0+w) rows rows0.., compound T w×w) to A[rows0.., colA:+nc).
    // When Yt != nullptr, GEMM 3 uses TN (opA=T on pre-transposed Yt) instead of NN — ~1.5× faster
    // on H200 (TN at ~106% TF32 ceiling vs NN at ~66%). Yt must be w×mr col-major (lda=w).
    // par3x >= 0 AND use3xtf32: activates the 3×TF32 Ozaki-split path (fp32-accurate, §13.4).
    //   Y/Yt must be pre-split into Yhi/Ylo/Ythi/Ytlo[par3x] by the caller (once per super-panel).
    //   C/W2 are split per-call. GEMM2 runs fp32. Split-K chunks GEMM1's A_hi·B_hi for accuracy.
    void applyQt(Real* A, size_t lda, const Real* SPV, int rows0, int c0, int w,
                 const Real* Tw, int ldT, int colA, int nc, bool tf32, cudaStream_t st,
                 Real* W1, Real* W2, const Real* Yt = nullptr, bool smLimit = false,
                 int par3x = -1, int mr_override = -1) {
        if (nc <= 0 || w <= 0) return;
        NVTX_RANGE("update_wave");
        const int mr = (mr_override > 0) ? mr_override : (m - rows0);
        // Q_lambda at the L2/HBM boundary, counted where the dimensions are known
        // rather than modelled afterwards. A WY apply over an mr x nc trailing block
        // reads and writes C and reads the mr x w reflector block; the 2*mr*nc term
        // dominates and is exactly the update traffic B_lambda's F/(P sqrt M) term
        // bounds. Counting every applyQt (near and far alike) is deliberate: the
        // trailing update IS the boundary traffic the theorem is about, and
        // splitting it by stream would only hide part of it.
        // ATTRIBUTE THE TRAFFIC TO THE RIGHT BOUNDARY. Counting every applyQt as
        // L2/HBM traffic conflates two rungs and inflates Q_3 by the near updates,
        // which is exactly the cross-rung mixing 31:371 forbids. A near update runs
        // inside the super-panel's column block, which is L2-resident by
        // construction -- that is what the super-panel is FOR -- so it crosses
        // DSM/L2 (rung 2). Only the far updates stream the trailing matrix from
        // HBM (rung 3), and smLimit is set exactly on those call sites.
        //
        // Measured consequence at 8192^2 fp64: counting both gave Q3 = 5.82e9
        // against B_3 = 3.98e8, a ratio of 14.6 that looked like gross
        // communication sub-optimality. It was the instrument. 5.82e9 is
        // 2n^3/(3*NB) -- the near updates at NB=64 granularity -- while far-only is
        // 2n^3/(3*B) = 3.6e8 at B=1024, i.e. ratio ~0.9 against the budget.
        if (smLimit) mv_words_upd  += 2.0 * (double)mr * (double)nc + (double)mr * (double)w;
        else         mv_words_near += 2.0 * (double)mr * (double)nc + (double)mr * (double)w;
        // Rung 3 COMBINE, the one move of 31:429's L2/HBM row that was never counted.
        // That row reads "W and ordered commits", and this is precisely where both
        // happen: the apply forms W = V^T C -- a reduction over mr rows into a w x nc
        // partial -- and then commits it back into C in panel order. Split (far
        // stripes), Replicate (the saved compact transform state at parity fslot) and
        // Pipeline (s1 overlapped with the next panel on s0) were already counted at
        // the far dispatch; only the reduction itself was missing, which is why the
        // witness read instrumented=[1,1,0,1] with combine[3]=0 rather than a genuine
        // absence of the carrier.
        if (smLimit) ++mv_combine[3];
        const Real* Y = SPV + size_t(c0) * m + rows0;
        Real* C = A + size_t(colA) * lda + rows0;
        if (st == sR) {   // STAGE-2 rest-slice: custom kernel, no cuBLAS cold-dispatch risk (see above)
            assert(w == NB);
            const int gridN = cdiv(nc, 32);
            k_custom_apply<Real, NB, 32><<<gridN, 256, 0, st>>>(C, lda, Y, m, Tw, ldT, mr, nc);
            NVTX_POP();
            return;
        }
        cublasHandle_t cb = (st == s1) ? cb1 : (st == sP ? cbT : cb0);
        CUBLAS_CHECK(cublasSetStream(cb, st));
        const bool small = green && (st == s1);
        if (small) cuCtxPushCurrent(ctxS);

        // ---- 3×TF32 path: fp32-accurate far-update via 3 TF32-TC GEMMs (§13.4) ----
        // Split each fp32 operand into hi+lo TF32 (Sterbenz-exact, bits & 0xFFFFE000).
        // C = A·B ≈ A_hi·B_hi + A_hi·B_lo + A_lo·B_hi  (omits A_lo·B_lo, O(2^-26)).
        // Y/Yt are pre-split once per super-panel (amortized over all far chunks).
        // C/W2 are split per call. GEMM2 (T^T·W1) uses fp32 CUDA cores for accuracy.
        // Split-K on the dominant A_hi·B_hi term chunks the K-reduction: per-chunk TF32-TC
        // error is random and cancels across the fp32 beta=1 cross-chunk accumulation.
        if (use3xtf32 && par3x >= 0 && nc <= maxFarchunk3x) {
            tel_split_path++;   // ENGAGEMENT TELEMETRY: 3×TF32 Ozaki-split path engaged
            Real* Yhi_p = Yhi[par3x];
            Real* Ylo_p = Ylo[par3x];
            // GEMM1: W1 = sum over K-slices of
            //          Yhi^T·Chi  +  oys·Yhi^T·Clo  +  oys·Ylo^T·Chi
            // All three terms are K-reductions → per-slice accumulation with beta=1
            // is bit-identical to the full-K form (OZAKI_PHASE3_DESIGN §5).
            // Toggle TF32 on for the split GEMMs, off for GEMM2 (fp32). DEFAULT_MATH is set
            // at create time; cublasGemmEx with FAST_TF32 compute + TF32 math mode uses TF32 TC.
            CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TF32_TENSOR_OP_MATH));
            const Real oys = Real(1.0f / 2048.0f);
            // Adaptive splitK: small K→fine split (accuracy), large K→coarse split (throughput).
            // RZ error ~ sqrt(K*splitK); quantization floor dominates at large K.
            // K<=1024: 32, K=4096: 256, K>4096: 2048 (W4 EC-collective microbench, slurm 7120).
            // MEASURED (W4a 6a0acd2 / G6C_CLOSURE.md §4.#4): sk=512 FAILS @n=32768 (resid 5.99e-4
            // = 202x cuSOLVER, RZ-in-TC bias threshold); sk=256 PASSES 16/16 (worst 1.16x @32768);
            // sk=0 FAILS (no C1 mitigation). Handles RZ (separate from the EC floor splitK
            // cannot reduce — RESID_BASE_CURVE.md §5).
            int sk = (mr <= 1024) ? 32 : (mr <= 4096) ? 256 : 2048;
            // W6-P4 (G7 Phase-3): Chi/Clo are sized maxSk3x·nc (in-place, OZAKI_PHASE3_DESIGN
            // §3.2). The producer re-splits C per K-slice into these buffers with ld=maxSk3x;
            // the consumer GEMMs read kl ≤ maxSk3x rows. Both branches use ldCsplit=maxSk3x
            // (the 5b1802c bug was allocating shrunk but reading with ld=m — avoided here).
            const int ldCsplit = maxSk3x;
            if (splitK3x > 0 && mr > sk) {
                // Zero W1 implicitly via the first main GEMM (beta=0); subsequent GEMMs
                // accumulate with beta=1. No separate memset (OZAKI_PHASE3_DESIGN §6 row 2).
                // Takeover pipeline: the slice re-split runs on s3xsp into buffer parity
                // i&1 while the consumer GEMMs read parity (i-1)&1 on st — the split cost
                // (~kl·nc·3·4B of traffic per slice, formerly serialized before the GEMMs)
                // hides behind the previous slice's GEMMs. Math, order, and accumulation
                // are BIT-IDENTICAL to the serial form (same GEMM sequence on st; the
                // split output for slice i is identical regardless of when it runs).
                // Seed both parities' "last reader" events at the current st frontier so
                // (a) slice 0/1 splits see C's latest contents (C last written on st) and
                // (b) any previous chunk's buffer use is fully ordered.
                CUDA_CHECK(cudaEventRecord(evG3x[0], st));
                CUDA_CHECK(cudaEventRecord(evG3x[1], st));
                int nsl = 0;
                for (int k0 = 0; k0 < mr; k0 += sk, ++nsl) {
                    int kl = std::min(sk, mr - k0);
                    const int pb = nsl & 1;
                    // Producer: wait for the GEMMs that last read this parity.
                    CUDA_CHECK(cudaStreamWaitEvent(s3xsp, evG3x[pb], 0));
                    { dim3 bt(32, 8); dim3 gt(cdiv(kl, 32), cdiv(nc, 8));
                      k_split_3xtf32_2d<Real><<<gt, bt, 0, s3xsp>>>(
                          C + k0, (int)lda, Chi[pb], ldCsplit, Clo[pb], ldCsplit, kl, nc); }
                    CUDA_CHECK(cudaEventRecord(evSp3x[pb], s3xsp));
                    // Consumer: GEMMs for slice nsl wait its split.
                    CUDA_CHECK(cudaStreamWaitEvent(st, evSp3x[pb], 0));
                    Real beta = (k0 == 0) ? Real(0) : Real(1);
                    // Main term (was split-K before — same as the prior code's main loop).
                    gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, kl, Real(1),
                         Yhi_p + k0, m, Chi[pb], ldCsplit, beta, W1, w);
                    // Cross terms (were full-K before — now per-slice, beta=1). Per-slice
                    // TF32 rounding is STRICTLY more accurate than full-K (smaller K/round).
                    gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, kl, Real(oys),
                         Yhi_p + k0, m, Clo[pb], ldCsplit, Real(1), W1, w);
                    gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, kl, Real(oys),
                         Ylo_p + k0, m, Chi[pb], ldCsplit, Real(1), W1, w);
                    CUDA_CHECK(cudaEventRecord(evG3x[pb], st));
                }
            } else {
                // Small-mr path: mr ≤ sk, no split. Use Chi[0]/Clo[0] as a mr×nc tile.
                { dim3 bt(32, 8); dim3 gt(cdiv(mr, 32), cdiv(nc, 8));
                  k_split_3xtf32_2d<Real><<<gt, bt, 0, st>>>(
                      C, (int)lda, Chi[0], ldCsplit, Clo[0], ldCsplit, mr, nc); }
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, mr, Real(1),
                     Yhi_p, m, Chi[0], ldCsplit, Real(0), W1, w);
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, mr, Real(oys),
                     Yhi_p, m, Clo[0], ldCsplit, Real(1), W1, w);
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, mr, Real(oys),
                     Ylo_p, m, Chi[0], ldCsplit, Real(1), W1, w);
            }
            // GEMM2: W2 = T^T·W1 (fp32 — T is small, TF32 error would propagate)
            CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, w, Real(1), Tw, ldT, W1, w,
                 Real(0), W2, w);
            // GEMM3: 3×TF32 split (C -= Yt_hi^T·W2_hi + Yt_hi^T·W2_lo + Yt_lo^T·W2_hi)
            CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TF32_TENSOR_OP_MATH));
            { dim3 bt(32, 8); dim3 gt(cdiv(w, 32), cdiv(nc, 8));
              k_split_3xtf32_2d<Real><<<gt, bt, 0, st>>>(W2, w, W2hi, w, W2lo, w, w, nc); }
            if (Yt) {
                Real* Ythi_p = Ythi[par3x];
                Real* Ytlo_p = Ytlo[par3x];
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, w, Real(-1),
                     Ythi_p, w, W2hi, w, Real(1), C, (int)lda);
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, w, Real(-oys),
                     Ythi_p, w, W2lo, w, Real(1), C, (int)lda);
                gemm(cb, true, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, w, Real(-oys),
                     Ytlo_p, w, W2hi, w, Real(1), C, (int)lda);
            } else {
                gemm(cb, true, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, w, Real(-1),
                     Yhi_p, m, W2hi, w, Real(1), C, (int)lda);
                gemm(cb, true, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, w, Real(-oys),
                     Yhi_p, m, W2lo, w, Real(1), C, (int)lda);
                gemm(cb, true, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, w, Real(-oys),
                     Ylo_p, m, W2hi, w, Real(1), C, (int)lda);
            }
            CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
            if (small) cuCtxPopCurrent(nullptr);
            NVTX_POP();
            return;
        }

        // ---- non-3xTF32 path ----
        if (ltSmTarget > 0 && smLimit) {
            // cuBLASLt path: far-rest only (smLimit=true). Near/rest/slice use regular cuBLAS.
            cublasLtHandle_t ltH = (st == s1) ? lt1 : (st == sP ? ltT : lt0);
            cublasLtMatmulDesc_t descTN = ltDescTN;
            cublasLtMatmulDesc_t descNN = ltDescNN;
            size_t wsSize = 128 * 1024 * 1024;
            if (smLimit && ltSplitK > 0 && mr > 4096) {
                // Manual split-K for GEMM 1: W1 = Y^T @ C (K=mr, large)
                // Keeps Y chunks in L2 (~K_chunk*B*4 bytes)
                int sk = ltSplitK, klen = (mr + sk - 1) / sk;
                for (int ks = 0; ks < sk; ++ks) {
                    int k0 = ks * klen, k1 = std::min(mr, k0 + klen), kl = k1 - k0;
                    if (kl <= 0) break;
                    Real beta = (ks == 0) ? Real(0) : Real(1);
                    gemmLt(ltH, descTN, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, kl, Real(1),
                           Y + k0, m, C + k0, (int)lda, beta, W1, w, ltWorkspace, wsSize, st);
                }
            } else {
                gemmLt(ltH, descTN, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, mr, Real(1),
                       Y, m, C, (int)lda, Real(0), W1, w, ltWorkspace, wsSize, st);
            }
            gemmLt(ltH, descTN, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, w, Real(1),
                     Tw, ldT, W1, w, Real(0), W2, w, ltWorkspace, wsSize, st);
            if (smLimit && ltSplitK > 0 && w > 256) {
                // Manual split-K for GEMM 3: C -= Yt^T @ W2 (K=w)
                int sk = ltSplitK, klen = (w + sk - 1) / sk;
                for (int ks = 0; ks < sk; ++ks) {
                    int k0 = ks * klen, k1 = std::min(w, k0 + klen), kl = k1 - k0;
                    if (kl <= 0) break;
                    if (Yt) {
                        gemmLt(ltH, descTN, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, kl, Real(-1),
                                Yt + k0, w, W2 + k0, w, Real(1), C, (int)lda,
                                ltWorkspace, wsSize, st);
                    } else {
                        gemmLt(ltH, descNN, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, kl, Real(-1),
                                Y + size_t(k0) * m, m, W2 + k0, w, Real(1), C, (int)lda,
                               ltWorkspace, wsSize, st);
                    }
                }
            } else {
                if (Yt) {
                    gemmLt(ltH, descTN, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, w, -1.0f,
                            Yt, w, W2, w, 1.0f, C, (int)lda, ltWorkspace, wsSize, st);
                } else {
                    gemmLt(ltH, descNN, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, w, -1.0f,
                            Y, m, W2, w, 1.0f, C, (int)lda, ltWorkspace, wsSize, st);
                }
            }
        } else {
        gemm(cb, tf32, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, mr, Real(1), Y, m, C, (int)lda,
             Real(0), W1, w);
        gemm(cb, tf32, CUBLAS_OP_T, CUBLAS_OP_N, w, nc, w, Real(1), Tw, ldT, W1, w,
             Real(0), W2, w);
        if (Yt) {
            // TN: C[mr×nc] -= Yt^T[w×mr→mr×w] @ W2[w×nc] — 1.5× faster than NN on H200
            gemm(cb, tf32, CUBLAS_OP_T, CUBLAS_OP_N, mr, nc, w, Real(-1), Yt, w, W2, w,
                 Real(1), C, (int)lda);
        } else {
            gemm(cb, tf32, CUBLAS_OP_N, CUBLAS_OP_N, mr, nc, w, Real(-1), Y, m, W2, w,
                 Real(1), C, (int)lda);
        }
        }
        if (small) cuCtxPopCurrent(nullptr);
        NVTX_POP();
    }

    // ------------------------------------------------------------ Stage 2: merge apply
    // Applies Q_merge^T to B (the trailing matrix) bottom-up over merge-tree nodes.
    // Each node μ = (topIdx, botIdx) has V_μ = [I_NB; W_μ] (W_μ = Vm[node], NB×NB, ld=NB),
    // T_μ = TmM[node] (NB×NB upper-tri, ld=NB). The apply on each node's 2·NB rows:
    //   W1 = B_top + W_μ^T · B_bot ;  W2 = T_μ^T · W1 ;  B_top −= W2 ;  B_bot −= W_μ · W2
    // where B_top = A[row0+slabStart[topIdx], colA:], B_bot = A[row0+slabStart[botIdx], colA:].
    // fp32 (no TF32) per the memo §5: merge reflectors are small, TF32 risks precision drift.
    void mergeApplyQt(Real* A, size_t lda, int row0, int colA, int nc,
                      cudaStream_t st, Real* W1, Real* W2) {
        if (nc <= 0 || last_nm == 0) return;
        cublasHandle_t cb = (st == s1) ? cb1 : (st == sP ? cbT : cb0);
        CUBLAS_CHECK(cublasSetStream(cb, st));
        // Merge reflectors are small (NB×NB); run in true fp32 (no TF32) per memo §5.
        // The handle may carry CUBLAS_TF32_TENSOR_OP_MATH from the bulk-apply tier; on
        // some cuBLAS builds that downgrades CUBLAS_COMPUTE_32F to TF32, eroding the
        // merge accuracy and widening the c_L>1 path-depth error. Force DEFAULT_MATH
        // (no-op for fp64, which is already DEFAULT/PEDANTIC) and restore after.
        cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
        CUBLAS_CHECK(cublasGetMathMode(cb, &saveMath));
        const bool forceFp32 = !std::is_same<Real, double>::value &&
                               saveMath != CUBLAS_DEFAULT_MATH;
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
        const int dtpb = 256;
        // Level boundaries, rebuilt from c_L exactly as panel_25d built the tree, so
        // an event can be recorded after each ORDERED level. The node loop order is
        // unchanged -- this only observes it (A6: the merge order is the obligation).
        int lvlEnd[cLmax], nlv2 = 0, acc2 = 0;
        for (int step = 1; step < last_cL; step *= 2) {
            int cnt = 0;
            for (int i = 0; i + step < last_cL; i += 2 * step) ++cnt;
            acc2 += cnt; lvlEnd[nlv2++] = acc2;
        }
        int lvlPtr = 0;
        // Node index at which the global apply stops when the cap is armed. Levels
        // beyond it belong to the next epoch's lanes.
        const int nodeStop = (m2_max_level >= 0 && m2_max_level < nlv2)
                           ? lvlEnd[m2_max_level] : last_nm;
        for (int node = 0; node < nodeStop; ++node) {
            const int topIdx = last_nodes[node].x;
            const int botIdx = last_nodes[node].y;
            const int top = row0 + last_slabStart[topIdx];
            const int bot = row0 + last_slabStart[botIdx];
            Real* B_top = A + size_t(colA) * lda + top;
            Real* B_bot = A + size_t(colA) * lda + bot;
            const Real* Wmu = Vm + size_t(node) * NB * NB;
            const Real* Tmu = TmM + size_t(node) * NB * NB;
            // GEMM1: W1 = W_μ^T · B_bot (beta=0)
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                 Wmu, NB, B_bot, (int)lda, Real(0), W1, NB);
            // W1 += B_top (element-wise)
            { int ngr = cdiv(NB * nc, dtpb);
              k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W1, NB, nc, 0); }
            // GEMM2: W2 = T_μ^T · W1 (beta=0)
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                 Tmu, NB, W1, NB, Real(0), W2, NB);
            // B_top -= W2 (element-wise)
            { int ngr = cdiv(NB * nc, dtpb);
              k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W2, NB, nc, 1); }
            // GEMM3: B_bot -= W_μ · W2 (beta=1)
            gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nc, NB, Real(-1),
                 Wmu, NB, W2, NB, Real(1), B_bot, (int)lda);
            // Record only on s0, the NEAR update. applyQt_25d also runs for the rest
            // slice and for the deferred far region (on s1), and each call would
            // otherwise re-record these events; a later panel waiting on them would
            // then be waiting on the far update of the previous panel -- still
            // correct, but strictly more conservative than the blanket barrier it
            // replaces, which would hide the pipeline entirely. The next panel's
            // columns live in the near region of the same super-panel, so the near
            // update is the one that governs its readiness.
            if (st == s0 && lvlPtr < nlv2 && node + 1 == lvlEnd[lvlPtr])
                CUDA_CHECK(cudaEventRecord(evM2lvl[lvlPtr++], st));
        }
        // Only the near update (s0) records the level events, so only it may
        // advertise them to the next panel. lvlPtr is how many were actually
        // recorded; publishing nlv2 unconditionally would let a lane wait on a
        // stale event from an older panel.
        if (st == s0) { last_m2_levels = nlv2; last_m2_applied = lvlPtr; }
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, saveMath));
    }

    // ------------------------------------------------------------ two-level WY apply
    // Q_total^T = Q_merge^T · diag(Q_0^T,…,Q_{c_L-1}^T)  (RESEARCH_MEMO §1)
    // Stage 1: c_L per-lane applyQt (row-bounded, bulk GEMM — reuses applyQt verbatim)
    // Stage 2: merge apply on c_L·NB rows (bottom-up, small GEMMs — mergeApplyQt)
    // W1/W2 are reused for both stages (sequential on the same stream, no conflict).
    void applyQt_25d(Real* A, size_t lda, Real* SPV, int row0, int j0,
                     int colA, int nc, bool tf32, cudaStream_t st,
                     Real* W1, Real* W2) {
        if (nc <= 0) return;
        NVTX_RANGE("applyQt_25d");
        for (int j = 0; j < last_cL; ++j) {
            applyQt(A, lda, SPV, row0 + last_slabStart[j], j0, NB,
                    domTpanL + size_t(j) * NB * NB, NB,
                    colA, nc, tf32, st, W1, W2, nullptr, false, -1,
                    last_slabSize[j]);
        }
        // Stage 1 complete: every lane's own Q has been applied over its full slab.
        // A next-panel lane whose rows no merge-node top band touches needs only this.
        if (st == s0) CUDA_CHECK(cudaEventRecord(evS1done, st));
        mergeApplyQt(A, lda, row0, colA, nc, st, W1, W2);
        // Remember which panel these Stage-2 level events describe -- and only for
        // the near update, which is the one that recorded them. last_slabStart /
        // last_cL still hold THIS panel's layout, and the next panel reads them
        // before overwriting, so row0 is the only missing piece.
        if (st == s0) last_row0 = row0;
        NVTX_POP();
    }
    // Far-deferred variant: merge data passed explicitly from save buffers.
    // Eliminates the swap/sync race — member variables are never modified.
    // ---- the invariant's LOCAL apply (31:329) ----
    // "Replicated compact transform state allows the required update to occur
    //  locally at that stage; no stale matrix value is used."
    //
    // Applies merge nodes in the level range [lvFrom, lvTo) of the PREVIOUS epoch to
    // a single column block, from state passed EXPLICITLY (the replicated copy), on
    // the caller's stream. Paired with m2_max_level, which stops the global near
    // apply at lvFrom, so each node is applied exactly ONCE -- by the lane the
    // invariant says owns it rather than by a global pass the lane must wait for.
    //
    // WHY A NODE IS NOT SPLIT ACROSS LANES. A node (top,bot) reads AND writes both
    // bands, so the algebra does not separate. It does not need to: at D=3 the only
    // skipped node is (0,4), whose top band sits at prow0 -- OUTSIDE the next epoch's
    // rows, which begin at prow0+NB -- while its bot band is inside. So exactly one
    // lane needs it and applies the whole node. The caller passes the node's own
    // bands; ownership is decided there.
    void mergeApplyLevels(Real* A, size_t lda, int prow0, int colA, int nc,
                          cudaStream_t st, Real* W1, Real* W2,
                          int p_cL, const int* p_slabStart,
                          const Real* p_Vm, const Real* p_TmM,
                          int p_nm, const int2* p_nodes, int lvFrom, int lvTo) {
        if (nc <= 0 || p_nm == 0 || p_cL <= 1 || lvFrom >= lvTo) return;
        int lvlEnd[cLmax], nlv = 0, acc = 0;
        for (int step = 1; step < p_cL; step *= 2) {
            int cnt = 0;
            for (int i = 0; i + step < p_cL; i += 2 * step) ++cnt;
            acc += cnt; lvlEnd[nlv++] = acc;
        }
        if (lvFrom >= nlv) return;
        const int n0 = (lvFrom == 0) ? 0 : lvlEnd[lvFrom - 1];
        const int n1 = lvlEnd[std::min(lvTo, nlv) - 1];
        cublasHandle_t cb = (st == s1) ? cb1 : (st == sP ? cbT : cb0);
        CUBLAS_CHECK(cublasSetStream(cb, st));
        cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
        CUBLAS_CHECK(cublasGetMathMode(cb, &saveMath));
        const bool forceFp32 = !std::is_same<Real, double>::value &&
                               saveMath != CUBLAS_DEFAULT_MATH;
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
        const int dtpb = 256;
        for (int node = n0; node < n1; ++node) {
            const int top = prow0 + p_slabStart[p_nodes[node].x];
            const int bot = prow0 + p_slabStart[p_nodes[node].y];
            const Real* Wmu = p_Vm  + size_t(node) * NB * NB;
            const Real* Tmu = p_TmM + size_t(node) * NB * NB;
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                 Wmu, NB, A + size_t(colA) * lda + bot, (int)lda, Real(0), W1, NB);
            { int ngr = cdiv(NB * nc, dtpb);
              k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W1, NB, nc, 0); }
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                 Tmu, NB, W1, NB, Real(0), W2, NB);
            { int ngr = cdiv(NB * nc, dtpb);
              k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W2, NB, nc, 1); }
            gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nc, NB, Real(-1),
                 Wmu, NB, W2, NB, Real(1), A + size_t(colA) * lda + bot, (int)lda);
        }
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, saveMath));
    }

    void applyQt_25d_far(Real* A, size_t lda, Real* SPV, int row0, int j0,
                         int colA, int nc, bool tf32, cudaStream_t st,
                         Real* W1, Real* W2,
                         int s_cL, const int* s_slabStart, const int* s_slabSize,
                         const Real* s_domTpanL,
                         Real* s_Vm, Real* s_TmM, int s_nm, const int2* s_nodes) {
        if (nc <= 0) return;
        NVTX_RANGE("applyQt_25d_far");
        for (int j = 0; j < s_cL; ++j) {
            applyQt(A, lda, SPV, row0 + s_slabStart[j], j0, NB,
                    s_domTpanL + size_t(j) * NB * NB, NB,
                    colA, nc, tf32, st, W1, W2, nullptr, false, -1,
                    s_slabSize[j]);
        }
        if (s_nm > 0) {
            cublasHandle_t cb = (st == s1) ? cb1 : (st == sP ? cbT : cb0);
            CUBLAS_CHECK(cublasSetStream(cb, st));
            cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
            CUBLAS_CHECK(cublasGetMathMode(cb, &saveMath));
            const bool forceFp32 = !std::is_same<Real, double>::value &&
                                   saveMath != CUBLAS_DEFAULT_MATH;
            if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
            const int dtpb = 256;
            for (int node = 0; node < s_nm; ++node) {
                const int topIdx = s_nodes[node].x;
                const int botIdx = s_nodes[node].y;
                const int top = row0 + s_slabStart[topIdx];
                const int bot = row0 + s_slabStart[botIdx];
                Real* B_top = A + size_t(colA) * lda + top;
                Real* B_bot = A + size_t(colA) * lda + bot;
                const Real* Wmu = s_Vm + size_t(node) * NB * NB;
                const Real* Tmu = s_TmM + size_t(node) * NB * NB;
                gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                     Wmu, NB, B_bot, (int)lda, Real(0), W1, NB);
                { int ngr = cdiv(NB * nc, dtpb);
                  k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W1, NB, nc, 0); }
                gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nc, NB, Real(1),
                     Tmu, NB, W1, NB, Real(0), W2, NB);
                { int ngr = cdiv(NB * nc, dtpb);
                  k_merge_elem<Real, NB><<<ngr, dtpb, 0, st>>>(A, lda, top, colA, W2, NB, nc, 1); }
                gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nc, NB, Real(-1),
                     Wmu, NB, W2, NB, Real(1), B_bot, (int)lda);
            }
            if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb, saveMath));
        }
        NVTX_POP();
    }

    // ------------------------------------------------------------ full factorization
    // ========================================================================
    // geqrf — scale-safe wrapper around the factorization.
    //
    // WHY THIS EXISTS. The Householder step at lrqr.cuh:1338 computes
    // mu = sqrt(al*al + sg) with sg a plain SUM OF SQUARES. On a matrix scaled
    // near the underflow threshold (LAPACK's dlatb4 IMAT 7 puts ANORM at
    // 0.25*safmin/eps ~ 2.5e-293 for fp64) every term of sg underflows to zero,
    // the `if (sg > 0)` guard goes false, and EVERY REFLECTOR BECOMES THE
    // IDENTITY: nothing is factored, Q = I, R = triu(A). --validate-lapack caught
    // this on its first run -- 21 cells, RESULT(1) ~ 1/(m*eps) and RESULT(2)
    // exactly 0. The guard is right; it is dlarfg's convention for a genuinely
    // zero column. The defect is that squaring without scaling makes a merely
    // SMALL column indistinguishable from a zero one.
    //
    // WHY SCALE THE MATRIX INSTEAD OF THE INNER LOOP. sg is accumulated at three
    // separate sites inside the hot column sweep (:1633, :1654, :1926). A scaled
    // two-pass norm at each would rewrite the critical path for a case no shipping
    // cell hits. QR is scale-equivariant instead: cA = Q(cR) for c > 0, so the
    // factorization of a rescaled matrix has THE SAME Q, and only R scales. So:
    // scale A up by a power of two (exact in binary FP, no rounding introduced),
    // factor, and scale R back down. Nothing in the kernels changes.
    //
    // ONLY R IS UNSCALED, AND THAT IS DELIBERATE. V and the T blocks are left as
    // the factorization produced them. They represent the Q of the scaled matrix,
    // which IS the Q of the original -- so orgqr and ormqr are correct with no
    // correction, whatever internal normalization V and T happen to use. Trying to
    // unscale them would require knowing that normalization; this does not.
    //
    // NO SHIPPING CELL IS TOUCHED. The band is wide enough that any matrix with
    // entries in a normal engineering range takes the identity path: the only cost
    // there is one max-abs reduction, and the factorization is bit-identical.
    double absmax_matrix(const Real* A, size_t lda, int mm, int nn) {
        // One kernel + one async copy + ONE stream sync. No allocation, no
        // cudaDeviceSynchronize: both were pure overhead on fast cells.
        double h[kAbsmaxBlocks];
        const int blocks = (int)std::min<size_t>(
            kAbsmaxBlocks, std::max<size_t>(1, ((size_t)mm * nn + 255) / 256));
        k_absmax_blk<Real><<<blocks, 256, 0, s0>>>(A, lda, mm, nn, absmaxPart);
        CUDA_CHECK(cudaMemcpyAsync(h, absmaxPart, (size_t)blocks * sizeof(double),
                                   cudaMemcpyDeviceToHost, s0));
        CUDA_CHECK(cudaStreamSynchronize(s0));
        double mx = 0.0;
        for (int i = 0; i < blocks; ++i) if (h[i] > mx) mx = h[i];
        return mx;
    }
    void scale_block(Real* A, size_t lda, int mm, int nn, Real s) {
        k_scale_block<Real><<<256, 256, 0, s0>>>(A, lda, mm, nn, s);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }
    void scale_upper(Real* A, size_t lda, int kk, int nn, Real s) {
        k_scale_upper<Real><<<256, 256, 0, s0>>>(A, lda, kk, nn, s);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }

    void geqrf(Real* A, size_t lda) {
        const int kk = (kmax > 0) ? kmax : std::min(m, n);
        // Safe band. Below the low end, amax*amax underflows; above the high end it
        // could overflow once summed over a column. Chosen far from every normal
        // matrix so the guarded path is genuinely never taken in production.
        const double LO = std::is_same<Real, double>::value ? 1e-120 : 1e-15;
        const double HI = std::is_same<Real, double>::value ? 1e+120 : 1e+15;
        const double amax = absmax_matrix(A, lda, m, n);
        int expo = 0;
        if (amax > 0.0 && (amax < LO || amax > HI)) {
            // Land amax in [1,2): a power of two, so scaling is exact both ways.
            std::frexp(amax, &expo);
            expo = -(expo - 1);
        }
        if (expo != 0) {
            const double s = std::ldexp(1.0, expo);
            scale_block(A, lda, m, n, (Real)s);
            geqrf_unscaled(A, lda);
            scale_upper(A, lda, kk, n, (Real)std::ldexp(1.0, -expo));
            return;
        }
        geqrf_unscaled(A, lda);
    }

    void geqrf_unscaled(Real* A, size_t lda) {
        // W3' carrier (sub-unit a): wgmma TF32 consumer mega kernel. Replaces scalar
        // FFMA GEMM1/GEMM3 with wgmma m64n64k8 SS TF32. TMA 2D loads (B32 swizzle) for V
        // and C. Timeout-wrapped launch + two-signal (prog_k/upd_phase) forward-progress
        // watchdog (memo §5). float-only. Must prove deadlock-free at n≤8192 before n≥16384.
        if constexpr (std::is_same_v<Real, float> && NB == 128) {
        if (useMegaWgmma) {
            CUDA_CHECK(cudaEventRecord(evIn, 0));
            CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
            // L2 persistence: pin V in L2
            cudaStreamAttrValue stream_attr;
            stream_attr.accessPolicyWindow.base_ptr = rVg;
            stream_attr.accessPolicyWindow.num_bytes =
                std::min<size_t>(2 * size_t(m) * NB * sizeof(Real), 48 * 1024 * 1024);
            stream_attr.accessPolicyWindow.hitRatio = 1.0;
            stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            // Launch via the host launcher (builds TMA desc SWIZZLE_32B + cudaLaunchKernelExC;
            // the kernel + TMA wiring live in wgmma_carrier.cu). Async — watchdog polls below.
            *hProgK = 0; *hUpdPhase = 0;
            wgmma_carrier_launch(A, m, n, lda, npanels, Tpan, rVg, rTg, s0,
                                 dProgK, dUpdPhase, megaWd);
            // ---- Timeout-wrapped watchdog (memo §5): 50ms poll, 30s hard cap ----
            // Two-signal forward-progress proof: prog_k (panels started) must be monotonic;
            // upd_phase (updates completed) must track prog_k within STAGES+1 lag. A stall
            // of either >2s while the kernel is still running → deadlock verdict.
            if (megaWd) {
                auto now_us = [] { return std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count(); };
                const int64_t hard_cap_us = 600000000;      // 600s (10 min) — allows n=65536
                const int64_t stall_cap_us = 5000000;        // 5s no-progress → deadlock
                int64_t t0 = now_us();
                int last_prog = 0, last_upd = 0;
                int64_t prog_stall_t = t0, upd_stall_t = t0;
                for (;;) {
                    cudaError_t qs = cudaStreamQuery(s0);
                    if (qs == cudaSuccess) break;
                    if (qs != cudaErrorNotReady) { CUDA_CHECK(qs); break; }
                    int cur_prog = __atomic_load_n((int*)hProgK, __ATOMIC_ACQUIRE);
                    int cur_upd  = __atomic_load_n((int*)hUpdPhase, __ATOMIC_ACQUIRE);
                    // Step 6 — device-side bounded-wait assert: a carrier CTA that exceeded
                    // its BOUND_WAIT spin count writes DEADLOCK_FLAG (-1) to prog_k. Detect
                    // it immediately (no 5s stall wait) and abort — the device CTA is
                    // permanently stuck and the cluster cannot recover.
                    if (cur_prog == -1) {
                        fprintf(stderr, "[W3' WATCHDOG] DEADLOCK (device bounded-wait assert) "
                                "prog_k=-1 upd_phase=%d npanels=%d — carrier CTA stuck\n",
                                cur_upd, npanels);
                        cudaDeviceReset();
                        std::abort();
                    }
                    int64_t tn = now_us();
                    if (cur_prog != last_prog) { last_prog = cur_prog; prog_stall_t = tn; }
                    if (cur_upd  != last_upd)  { last_upd  = cur_upd;  upd_stall_t  = tn; }
                    int64_t prog_stall = tn - prog_stall_t;
                    int64_t upd_stall  = tn - upd_stall_t;
                    bool starved = (cur_prog - cur_upd) > (2 /*STAGES*/ + 1);
                    if (tn - t0 > hard_cap_us) {
                        fprintf(stderr, "[W3' WATCHDOG] DEADLOCK (30s hard cap) "
                                "prog_k=%d upd_phase=%d npanels=%d\n", cur_prog, cur_upd, npanels);
                        cudaDeviceReset();
                        fprintf(stderr, "[W3' WATCHDOG] cudaDeviceReset done (cooperative kernel "
                                "cannot be cleanly cancelled). ABORT.\n");
                        std::abort();
                    }
                    if ((tn - t0) > stall_cap_us &&
                        (prog_stall > stall_cap_us || (starved && upd_stall > stall_cap_us))) {
                        fprintf(stderr, "[W3' WATCHDOG] DEADLOCK (forward-progress stall) "
                                "prog_k=%d upd_phase=%d stall_prog=%lldms stall_upd=%lldms\n",
                                cur_prog, cur_upd, (long long)prog_stall/1000,
                                (long long)upd_stall/1000);
                        cudaDeviceReset();
                        std::abort();
                    }
                    usleep(50000);  // 50ms poll
                }
            } else {
                CUDA_CHECK(cudaStreamSynchronize(s0));
            }
            fprintf(stderr, "[W3' CARRIER] done prog_k=%d upd_phase=%d npanels=%d\n",
                    *hProgK, *hUpdPhase, npanels);
            // Reset L2 persistence window
            stream_attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            gemmsWarmed = true;
            return;
        }
        }
        // T5 megakernel (spec §6 T5, LRQR_MEGA=1): cluster-scope mbarrier
        // resident QR. Block 0 = panel (Householder fp32), blocks 1-7 = trailing
        // update. mbarrier handoff (NOT gmem-counter + __nanosleep). Cluster
        // launch (cudaLaunchKernelEx, NOT cooperative). Double-buffered V/T.
        if (useMega) {
            CUDA_CHECK(cudaEventRecord(evIn, 0));
            CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
            constexpr int MN = MegaSmem<Real, NB>::TILE_N;
            constexpr int MC = MegaSmem<Real, NB>::CHUNK_M;
            size_t ms = MegaSmem<Real, NB>::bytes;
            // L2 persistence: pin V in L2 (spec §8: cudaAccessPolicyWindow)
            cudaStreamAttrValue stream_attr;
            stream_attr.accessPolicyWindow.base_ptr = rVg;
            stream_attr.accessPolicyWindow.num_bytes =
                std::min<size_t>(2 * size_t(m) * NB * sizeof(Real), 48 * 1024 * 1024);
            stream_attr.accessPolicyWindow.hitRatio = 1.0;
            stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            // Cluster launch: grid = 1 cluster of 8 CTAs (1 panel + 7 update)
            dim3 grid(Plan::megaCluster), block(256);
            void* args[] = {&A, &m, &n, &lda, &npanels, &Tpan, &rVg, &rTg};
            cudaLaunchConfig_t config = {};
            config.gridDim = grid;
            config.blockDim = block;
            config.dynamicSmemBytes = ms;
            config.stream = s0;
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim.x = Plan::megaCluster;
            attrs[0].val.clusterDim.y = 1;
            attrs[0].val.clusterDim.z = 1;
            config.attrs = attrs;
            config.numAttrs = 1;
            cudaError_t le = cudaLaunchKernelExC(&config,
                (const void*)k_tqr_mega<Real, NB, MN, MC, Plan::megaCluster>, args);
            CUDA_CHECK(le);
            CUDA_CHECK(cudaStreamSynchronize(s0));
            // Reset L2 persistence window
            stream_attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            gemmsWarmed = true;
            return;
        }
        // Resident multi-block cooperative kernel (spec §9/§11): block 0 does
        // panel factorization (FMA pipe), blocks 1..N do trailing update (tensor).
        // Double-buffered: panel k+1 overlaps update k. No cuBLAS → no deadlock.
        if (useResident) {
            CUDA_CHECK(cudaEventRecord(evIn, 0));
            CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
            constexpr int RN = ResidentSmem<Real, NB>::TILE_N;
            constexpr int RC = ResidentSmem<Real, NB>::CHUNK_M;
            size_t rs = ResidentSmem<Real, NB>::bytes;
            // Reset double-buffer flags
            CUDA_CHECK(cudaMemsetAsync((void*)rPanelGen, 0, 2 * sizeof(unsigned), s0));
            CUDA_CHECK(cudaMemsetAsync((void*)rUpdateCnt, 0, 2 * sizeof(unsigned), s0));
            // L2 persistence: pin V in L2 (spec §8: cudaAccessPolicyWindow)
            // V is read by all 131 update blocks per panel → high reuse
            // T3 FIX: was capped at 48MB > box max-persisting 37.5MB (guaranteed evictions
            // at hitRatio=1.0). Clamp to l2setaside (default 24MB, ≤ 37.5MB box max).
            cudaStreamAttrValue stream_attr;
            stream_attr.accessPolicyWindow.base_ptr = rVg;
            stream_attr.accessPolicyWindow.num_bytes =
                std::min<size_t>(2 * size_t(m) * NB * sizeof(Real), l2setaside);
            stream_attr.accessPolicyWindow.hitRatio = 1.0;
            stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            dim3 grid(rGrid), block(256);
            int nup = rGrid - 1;  // 131 update blocks
            void* args[] = {&A, &m, &n, &lda, &npanels, &Tpan,
                            &rVg, &rTg, &rPanelGen, &rUpdateCnt, &nup};
            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (const void*)k_tqr_resident<Real, NB, RN, RC>,
                grid, block, args, rs, s0));
            CUDA_CHECK(cudaStreamSynchronize(s0));
            // Reset L2 persistence window (don't affect subsequent kernels)
            stream_attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
            gemmsWarmed = true;
            return;
        }
        // Re-assert cuBLAS math mode on all handles to prevent TF32 state leak between
        // validate-all tiers (TF32 tier leaves handles in TF32_TENSOR_OP_MATH; fp32 tier
        // creates new handles but cuBLAS may cache TF32 kernel plans at context level).
        // Cor.FC (cbTsp=cb0) makes cb0 carry Tsp-compound GEMMs, widening the exposure.
        { const cublasMath_t m = (std::is_same<Real, double>::value ? CUBLAS_DEFAULT_MATH :
              ((!use3xtf32 && tf32far) ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
          if (cb0) CUBLAS_CHECK(cublasSetMathMode(cb0, m));
          if (cb1) CUBLAS_CHECK(cublasSetMathMode(cb1, m));
          if (cbT) CUBLAS_CHECK(cublasSetMathMode(cbT, m)); }
        // order after the caller's legacy-default-stream work; evIn lives in the primary
        // context — record BEFORE entering the green (big-partition) context.
        CUDA_CHECK(cudaEventRecord(evIn, 0));
        if (green) cuCtxPushCurrent(ctxB);
        CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
        CUDA_CHECK(cudaStreamWaitEvent(s1, evIn, 0));
        evFarValid[0] = evFarValid[1] = evFarValid[2] = false;
        if (gbar) CUDA_CHECK(cudaMemsetAsync(gbar, 0, 2 * sizeof(unsigned), s0));
        const bool persist = usePersist && gemmsWarmed;
        static bool ytDisabled = getenv("LRQR_NO_YT") && atoi(getenv("LRQR_NO_YT"));
        // Deferred far-rest: save state and issue at the START of the next iteration,
        // so the next super-panel's panel 0 launches BEFORE the far-rest GEMMs.
        // This eliminates the host-overhead gap where s0 is idle while the host
        // issues far-rest cuBLAS calls (spec §11: "streams run panel k+1's on-chip
        // reduction concurrently with panel k's HBM update").
        struct DeferredFarRest {
            Real* A; size_t lda; Real* SPV; int sp; int Bcur;
            Real* Tsp; int B; int farc; int wnext; int wfar; int fch;
            Real* Wa; Real* Wb; const Real* Yt_ptr;
            int par; int prev; bool valid; bool tf32far; int par3x;
        } dfr = {};
        // F8 Step-1: record stream-span start events (s0=panel/slice carrier, s1=far-rest)
        if (staget) {
            stagetFarFlops = 0.0;
            CUDA_CHECK(cudaEventRecord(evSt0a, s0));
            CUDA_CHECK(cudaEventRecord(evSt1a, s1));
        }
        // T4 Cor.FC: Tsp-compound stream/handle selectors. When corFC is set,
        // Tsp rides s0/cb0 (in-stream after the panel kernel); otherwise sP/cbT.
        cudaStream_t sTsp = corFC ? s0 : sP;
        cublasHandle_t cbTsp = corFC ? cb0 : cbT;
        tel_far_chunks = 0;   // ENGAGEMENT TELEMETRY: reset per geqrf (printed once/process)
        tel_split_path = 0;
        last_fch = 0;
        // A2: the super-panel walk stops at kmax = min(m,n) as well. Past kmax
        // there are no reflectors left to form, and the columns beyond it are
        // already covered -- each super-panel's far apply spans
        // wfar_super = n - (sp + Bcur), i.e. all the way to n, so the trailing
        // trapezoidal block still receives every reflector. On m>=n cells
        // kmax == n and the bound is the historical one.
        for (int sp = 0, spi = 0; sp < kmax; sp += B, ++spi) {
            const int par = spi % 3;
            const int Bcur = std::min(B, n - sp);
            // G2-PANEL V13: wfar_super — is there a far region for this super-panel?
            // The Tsp compound (3 GEMMs + Tpan→Tsp copy per panel) feeds ONLY the
            // super-panel-end far WY apply. When wfar_super==0 (last super-panel, or
            // n==B), the far apply is skipped entirely → the Tsp compound is wasted
            // work. Skipping it saves 3 GEMMs + 1 copy per panel (SMALLN_ADVERSARIAL_V12
            // §1.3: tower boundary GEMMs 0.585 vs cuSOLVER 0.233 ms/rep @1024; the Tsp
            // compound is a significant fraction of that 0.352 ms gap). This is NOT a
            // size gate — it's the same "is there far work" condition as the existing
            // wfar>0 check at the far-apply site (:7658).
            const int wfar_super = n - (sp + Bcur);
            Real *SPV = SPVx[par], *Tsp = Tspx[par], *Wa = Wax[par], *Wb = Wbx[par];
            // Triple-buffered: far-rest of spi-2 reads SPV[(spi-2)%3] = SPVx[par],
            // but that far-rest completed by now (waited via evFar at the END of spi-2's loop).
            // No evFar wait needed — the 3rd buffer is free.
            CUDA_CHECK(cudaMemsetAsync(SPV, 0, size_t(m) * B * sizeof(Real), s0));
            CUDA_CHECK(cudaMemsetAsync(Tsp, 0, size_t(B) * B * sizeof(Real), s0));
            evRestValid[0] = evRestValid[1] = false;
            far25Route = false; sp_c1Locked = false;   // far route is per super-panel
            // T3: L2 access-policy window — pin Tsp (B², 4MB fp32/8MB fp64) on s0+s1.
            // hitRatio=0.5 (S1 concurrent-stream guidance); opt-in via LRQR_L2WIN.
            if (useL2win) {
                cudaStreamAttrValue lw{};
                lw.accessPolicyWindow.base_ptr  = Tsp;
                lw.accessPolicyWindow.num_bytes = size_t(B) * B * sizeof(Real);
                lw.accessPolicyWindow.hitRatio  = l2hit;
                lw.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
                lw.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
                cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &lw);
                cudaStreamSetAttribute(s1, cudaStreamAttributeAccessPolicyWindow, &lw);
            }
            // Issue deferred far-rest from PREVIOUS super-panel on s1 (async).
            // This runs concurrently with this super-panel's panels on s0.
            // s1 already has the evSlice + evFar[prev] dependencies from when
            // the far-rest was deferred at the end of the previous iteration.
            if (dfr.valid) {
                if (dfr.wfar > dfr.wnext) {
                    for (int fc0 = dfr.farc + dfr.wnext; fc0 < dfr.farc + dfr.wfar; fc0 += dfr.fch) {
                        tel_far_chunks++;   // ENGAGEMENT TELEMETRY: far-rest chunk dispatched
                        mv_split[3]++;      // rung 3 SPLIT: one far stripe of the L2/HBM frontier
                        applyQt(dfr.A, dfr.lda, dfr.SPV, dfr.sp, 0, dfr.Bcur, dfr.Tsp, dfr.B,
                                fc0, std::min(dfr.fch, dfr.farc + dfr.wfar - fc0),
                                dfr.tf32far, s1, dfr.Wa, dfr.Wb, dfr.Yt_ptr, /*smLimit=*/true,
                                dfr.par3x);
                    }
                    CUDA_CHECK(cudaEventRecord(evFar[dfr.par], s1));
                    evFarValid[dfr.par] = true;
                } else {
                    evFarValid[dfr.par] = false;
                }
                dfr.valid = false;
            }
            // V6 W1 (spec §10 Theorem U): continuous overlap predicate replaces the
            // fp64 aspect>=12 case split. Wait when the panel is a large fraction of
            // the remaining work (panel_frac = Bcur / (n - sp)); overlap otherwise.
            // CORRECTNESS GUARD (kept): (useDsm && !tf32far) — non-coop DSM cluster
            // kernel races with s1 far-rest (comment above); fp32+DSM needs the wait
            // (box microbenchmark 2026-07-15: non-deterministic relRR 9e-7→2e-4 spike
            // at fp32@8192 without it). tf32+DSM was validated safe without the wait.
            const int far_cols = n - (sp + Bcur);
            const float panel_frac = float(Bcur) / float(std::max(Bcur + far_cols, 1));
            bool needFarWait = (panel_frac > 0.3f) || (useDsm && !tf32far);
            if (needFarWait) {
                const int fprev1 = (par + 2) % 3;  // spi-1
                if (evFarValid[fprev1]) CUDA_CHECK(cudaStreamWaitEvent(s0, evFar[fprev1], 0));
            }
            // STAGE 2 (ROADMAP §6.2): one resident coop+cluster kernel per super-panel;
            // host does only Tsp (sP), rest slices (sR), and the far update — all
            // flag-gated so no cuBLAS storm can steal exclusive residency mid-panel.
            // STAGE-2 super-domino capacity test. Two continuous conditions:
            //   (1) (m - sp) <= 16*superRpbMax  — panel height fits the cluster smem.
            //   (2) Bcur <= 2*NB (npan <= 2)     — no rest slices: the rest columns
            //       [2NB,Bcur) cost O(npan^2) per super-panel. With npan>2 that cost
            //       (whether host k_custom_apply or the internalized in-kernel apply)
            //       plus the resident cluster kernel blocking the s1 far-update overlap
            //       makes the resident kernel slower than the overlapped regular
            //       domino path (measured: npan=8 → 19 TF vs 142 TF baseline). npan<=2
            //       is the no-rest regime where the cluster kernel's full in-kernel
            //       look-ahead has no rest overhead. For npan>2 the !superOK loop
            //       (regular domino, cuBLAS far overlap) runs instead. This is NOT a
            //       presence-gate (useSuper=useDomino, no env var) — it is the per-
            //       panel capacity dispatch (spec §10 Theorem U), same class as the
            //       height test above.
            // npan cap stays 2: takeover probe (job 7903, GPU-8ac4d888, reps=9) measured
            // npan=4 at small n (B=512, n=1024/2048/4096) FLAT vs baseline — the small-n
            // fp64 gap (0.86-0.88x) is not launch-count-bound at this granularity either
            // (joins the V12 env-knob sweep, the B-sweep, and super-engagement, all flat).
            // The remaining lever is the batched-panel kernel class (G2 roadmap).
            const bool superOK = useSuper && !persist && Bcur % NB == 0 &&
                                 (m - sp) <= 16 * superRpbMax &&
                                 Bcur <= 2 * NB;
            if (superOK) {
                const int npan = Bcur / NB;
                int Cc = std::min(16, cdiv(m - sp, 512));
                int rpb0 = 32 * cdiv(cdiv(m - sp, Cc), 32);
                Cc = cdiv(m - sp, rpb0);
                const int G = Cc * cdiv(Cc + dsmU, Cc);   // multiple of cluster size
                DomArgs<Real> da;
                da.A = A + size_t(sp) * lda + sp; da.lda = lda;
                da.mk = m - sp; da.rpb = rpb0; da.row0 = sp;
                da.SPV = SPV; da.ldspv = m; da.spvcol = 0;
                da.Tpan = Tpan + size_t(sp / NB) * NB * NB;   // kernel uses local k
                da.tauT = domScal; da.fv = domScal + NB;
                da.bet = domScal + 2 * NB; da.sig = domScal + 3 * NB;
                da.dslot = domSlot; da.prow = domProw; da.Pp = domPp;
                da.Psum = domPsum; da.T16g = domT16; da.tphase = nullptr;
                da.nslab = 0;  // k_super_dsm, not k_panel_domino
                CUDA_CHECK(cudaMemsetAsync(pipeF, 0, 512 * sizeof(int), s0));
                CUDA_CHECK(cudaEventRecord(evPD, s0));
                if (!corFC) CUDA_CHECK(cudaStreamWaitEvent(sP, evPD, 0));
                CUDA_CHECK(cudaStreamWaitEvent(sR, evPD, 0));
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(G); cfg.blockDim = dim3(sizeof(Real) == 8 ? 256 : 512);
                cfg.dynamicSmemBytes = (size_t)dsmSmem2(rpb0); cfg.stream = s0;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim = {(unsigned)Cc, 1u, 1u};
                cfg.attrs = at; cfg.numAttrs = 1;
                NVTX_RANGE("chain_step");
                CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_super_dsm<Real, NB>,
                                              da, pipeF, Cc, npan));
                NVTX_POP();
                const int Uu = G - Cc;
                const bool sdbg = getenv("LRQR_SUPERDBG") != nullptr;
                std::vector<int> hpf, restv;
                if (sdbg) {
                    fprintf(stderr, "[superdbg] npan=%d Cc=%d Uu=%d rpb0=%d G=%d\n",
                            npan, Cc, Uu, rpb0, G);
                    hpf.resize(20 * npan);
                    restv.resize(std::max(1, npan));
                }
                auto dbgPoll = [&](const char* tag) {
                    if (!sdbg) return;
                    CUDA_CHECK(cudaMemcpy(hpf.data(), pipeF, hpf.size() * sizeof(int),
                                           cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(restv.data(), pipeF + 320,
                                           restv.size() * sizeof(int), cudaMemcpyDeviceToHost));
                    fprintf(stderr, "[%s]", tag);
                    for (int k = 0; k < npan; ++k) {
                        int* p = &hpf[k * 20];
                        fprintf(stderr, " k%d[u:%d,%d,%d,%d,%d,%d,%d,%d st:%d,%d,%d,%d,%d,%d,%d,%d done:%d next:%d tass:%d wr:%d]",
                                k, p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],
                                p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15],
                                p[16],p[17],p[18],p[19]);
                    }
                    fprintf(stderr, " rest:");
                    for (int k = 0; k < npan; ++k) fprintf(stderr, "%d,", restv[k]);
                    fprintf(stderr, "\n");
                    fflush(stderr);
                };
                dbgPoll("pre-loop");
                for (int k = 0; k < npan; ++k) {
                    int* pfk = pipeF + k * 20;
                    const int j0 = k * NB, mr = m - sp;
                    // Cor.FC: k_wait_flag not needed — Tsp is in-stream after k_super_dsm on s0,
                    // so all tassDone flags are already set when Tsp runs.
                    if (!corFC) k_wait_flag<<<1, 1, 0, sP>>>((volatile int*)(pfk + 18));
                    if (sdbg) {
                        char tag[32]; snprintf(tag, 32, "k%d-waitflag-%s", k, corFC ? "s0" : "sP");
                        fprintf(stderr, "[sdbg] issuing %s, syncing...\n", tag);
                        fflush(stderr);
                        CUDA_CHECK(cudaStreamSynchronize(sTsp));
                        dbgPoll(tag);
                    }
                    if (j0 > 0) {
                        // Tsp-compound: custom kernels, not cuBLAS — see k_custom_apply's comment.
                        // Same host-blocking cold-dispatch issue applies here even though Tsp isn't
                        // read by k_super_dsm; the host call itself hangs, which blocks the host from
                        // ever reaching the (genuinely critical-path) rest-slice code below.
                        NVTX_RANGE("Tsp_compound");
                        const dim3 g(cdiv(NB, 32), cdiv(j0, 32));
                        k_custom_gemm<Real, true, false><<<g, 256, 0, sTsp>>>(
                            j0, NB, mr, Real(1), SPV + sp, m, SPV + size_t(j0) * m + sp, m,
                            Real(0), Zb, j0);
                        k_custom_gemm<Real, false, false><<<g, 256, 0, sTsp>>>(
                            j0, NB, j0, Real(1), Tsp, B, Zb, j0, Real(0), M1b, j0);
                        k_custom_gemm<Real, false, false><<<g, 256, 0, sTsp>>>(
                            j0, NB, NB, Real(-1), M1b, j0, Tpan + size_t(sp / NB + k) * NB * NB,
                            NB, Real(0), Tsp + size_t(j0) * B, B);
                        NVTX_POP();
                    }
                    CUDA_CHECK(cudaMemcpy2DAsync(Tsp + size_t(j0) * B + j0,
                        B * sizeof(Real), Tpan + size_t(sp / NB + k) * NB * NB,
                        NB * sizeof(Real), NB * sizeof(Real), NB,
                        cudaMemcpyDeviceToDevice, sTsp));
                    if (sdbg) {
                        char tag[32]; snprintf(tag, 32, "post-k%d-wait+gemm", k);
                        CUDA_CHECK(cudaStreamSynchronize(sTsp));
                        dbgPoll(tag);
                    }
                    // rest slice on sR: REMOVED — the rest columns [2NB,Bcur) are now
                    // applied in-kernel by d_super_upd (extended loop covers [FB,ncTotal)),
                    // and restDone[k]=pf[320+k] is set in-kernel after the donePan barrier.
                    // This eliminates the host-serialized k_custom_apply rest slices that
                    // caused k_super_dsm to busy-spin 158ms/call on restDone flags.
                }
                CUDA_CHECK(cudaEventRecord(evRst, sR));
                CUDA_CHECK(cudaStreamWaitEvent(s0, evRst, 0));
                // Cor.FC: evTsp on s0 (in-stream after Tsp); s0 proceeds without cross-stream wait.
                // Baseline: evTsp on sP; s0 waits cross-stream.
                CUDA_CHECK(cudaEventRecord(evTsp, sTsp));
                if (!corFC) CUDA_CHECK(cudaStreamWaitEvent(s0, evTsp, 0));
                if (sdbg) {
                    CUDA_CHECK(cudaStreamSynchronize(s0));
                    dbgPoll("post-super-panel");
                    fprintf(stderr, "[superdbg] super-panel done, sp=%d\n", sp);
                }
            }
            bool any25d = false;
            // A2: stop the diagonal walk at kmax = min(m,n). Past that the panel
            // would be anchored at row0 >= m and have mk <= 0 rows. On every
            // m>=n cell kmax == n and this bound never binds.
            for (int j0 = 0; !superOK && j0 < Bcur && (sp + j0) < kmax; j0 += NB) {
                const int k = (sp + j0) / NB, row0 = sp + j0, pj = (j0 / NB) & 1;
                const int pprev = pj ^ 1;
                // evRest sync BEFORE panel: the c_L>1 merge writes Vm/TmM (global,
                // reused per panel) which the previous rest's mergeApplyQt reads on sN.
                // Without this wait the merge and rest race on Vm/TmM (non-deterministic
                // accuracy corruption at npan≥6 where the rest outlasts the panel).
                if (evRestValid[pprev])
                    CUDA_CHECK(cudaStreamWaitEvent(s0, evRest[pprev], 0));
                // Hand the rest-slice event down: with the blanket barrier removed,
                // panel_25d's per-lane release is the only place that can wait on it,
                // and evPrePanel (which encoded it transitively) is no longer used.
                restEv = evRest[pprev];
                restEvValid = evRestValid[pprev];
                CUDA_CHECK(cudaEventRecord(evPrePanel, s0));
                // The blanket barrier IS the surplus edge X(k,D-1) -> X(k+1,0). With
                // the pipeline on, panel_25d instead releases each lane at its own
                // readiness point (Stage 1, or the Stage-2 level whose top band
                // touches that lane's rows), which is the diagonal edge. Off, this
                // stands and the schedule is unchanged.
                if (!pipe25d())
                    for (int j = 0; j < cLmax; ++j)
                        CUDA_CHECK(cudaStreamWaitEvent(domSt[j], evPrePanel, 0));
                panel(A, lda, k, SPV, j0, s0);
                const int wnear = Bcur - (j0 + NB);
                if (last_cL > 1) {
                    any25d = true;
                    far25Route = true;   // every panel of this super-panel now owes its own far
                    // ---- c_L>1 two-level WY apply (RESEARCH_MEMO §1-§3) ----
                    // Stage 1 (per-lane applyQt) + Stage 2 (mergeApplyQt) per update.
                    // Near/rest on s0/sN (original merge buffers). Far deferred to s1
                    // (save-buffered) — overlapped with next panel's near/rest on s0/sN.
                    CUDA_CHECK(cudaEventRecord(evPD, s0));
                    // Copy merge data to save buffers for deferred far on s1.
                    // Double-buffered: slot = k%2, so we only wait for the far from
                    // panel k-2 (same slot), not k-1. This gives 2×panel_time overlap.
                    const int farc_sp = sp + Bcur, wfar_sp = n - farc_sp;
                    const bool doFar25 = (wfar_sp > 0);
                    const int fslot = k % 2;
                    if (doFar25) {
                        if (far25Pending[fslot]) {
                            CUDA_CHECK(cudaStreamWaitEvent(s0, evFar25[fslot], 0));
                            far25Pending[fslot] = false;
                        }
                        const size_t vmBytes = size_t(std::max(last_nm, 1)) * NB * NB * sizeof(Real);
                        const size_t dtpBytes = size_t(last_cL) * NB * NB * sizeof(Real);
                        CUDA_CHECK(cudaMemcpyAsync(VmSav[fslot], Vm, vmBytes, cudaMemcpyDeviceToDevice, s0));
                        CUDA_CHECK(cudaMemcpyAsync(TmMSav[fslot], TmM, vmBytes, cudaMemcpyDeviceToDevice, s0));
                        CUDA_CHECK(cudaMemcpyAsync(domTpanLSav[fslot], domTpanL, dtpBytes, cudaMemcpyDeviceToDevice, s0));
                        sav_cL[fslot] = last_cL; sav_nm[fslot] = last_nm;
                        for (int j = 0; j < last_cL; ++j) {
                            sav_slabStart[fslot][j] = last_slabStart[j];
                            sav_slabSize[fslot][j] = last_slabSize[j];
                        }
                        for (int j = 0; j < last_nm; ++j) sav_nodes[fslot][j] = last_nodes[j];
                        CUDA_CHECK(cudaEventRecord(evMergeCpy[fslot], s0));
                        CUDA_CHECK(cudaStreamWaitEvent(s1, evMergeCpy[fslot], 0));
                    }
                    if (wnear > 0) {
                        m2_max_level = m2_near_cap;      // near only
                        applyQt_25d(A, lda, SPV, row0, j0,
                                    sp + j0 + NB, std::min(NB, wnear),
                                    use3xtf32 ? false : tf32far, s0, Wn1, Wn2);
                        m2_max_level = -1;               // rest/far apply ALL levels
                        const int wrest = wnear - NB;
                        if (wrest > 0) {
                            CUDA_CHECK(cudaStreamWaitEvent(sN, evPD, 0));
                            if (evRestValid[pprev])
                                CUDA_CHECK(cudaStreamWaitEvent(sN, evRest[pprev], 0));
                            applyQt_25d(A, lda, SPV, row0, j0,
                                        sp + j0 + 2 * NB, wrest,
                                        use3xtf32 ? false : tf32far, sN, Wr1, Wr2);
                            CUDA_CHECK(cudaEventRecord(evRest[pj], sN));
                            evRestValid[pj] = true;
                        } else {
                            evRestValid[pj] = false;
                        }
                    }
                    // FAR on s1 (deferred, overlapped with next panel on s0).
                    // Far columns [farc_sp,n) don't conflict with near [sp+j0+NB,sp+j0+2*NB)
                    // or rest [sp+j0+2*NB,sp+Bcur) — safe to overlap.
                    if (doFar25) {
                        int fch = 0;
                        if (const char* e = getenv("LRQR_FARCHUNK")) fch = atoi(e);
                        if (fch <= 0) fch = farChunkFor(B, wfar_sp);
                        if (use3xtf32 && maxFarchunk3x > 0)
                            fch = std::min(fch, maxFarchunk3x);
                        last_fch = fch;   // ENGAGEMENT TELEMETRY: far-rest chunk size (c_L>1 path)
                        for (int fc0 = farc_sp; fc0 < farc_sp + wfar_sp; fc0 += fch) {
                            tel_far_chunks++;   // ENGAGEMENT TELEMETRY: 2.5D far chunk dispatched
                            // Rung 3 (L2/HBM), the carriers named in 31:429.
                            // SPLIT: this far stripe of the frontier.
                            // REPLICATE: the state it runs against is the SAVED copy
                            //   (sav_*/VmSav/TmMSav/domTpanLSav at parity fslot), a
                            //   second compact transform-state replica held so the
                            //   far update of panel k can proceed while panel k+1
                            //   overwrites the live buffers -- exactly "compact
                            //   transform state and in-flight tiles".
                            // PIPELINE: it is issued on s1 concurrently with the next
                            //   panel's near/rest on s0 -- "streamed batched WY
                            //   applies" across successive epochs.
                            mv_split[3]++;
                            mv_replicate[3]++;
                            mv_pipeline[3]++;
                            applyQt_25d_far(A, lda, SPV, row0, j0,
                                        fc0, std::min(fch, farc_sp + wfar_sp - fc0),
                                        use3xtf32 ? false : tf32far, s1, Wa, Wb,
                                        sav_cL[fslot], sav_slabStart[fslot], sav_slabSize[fslot],
                                        domTpanLSav[fslot],
                                        VmSav[fslot], TmMSav[fslot], sav_nm[fslot], sav_nodes[fslot]);
                        }
                        CUDA_CHECK(cudaEventRecord(evFar25[fslot], s1));
                        far25Pending[fslot] = true;
                    }
                } else {
                    // ---- c_L=1 original path ----
                    // Pre-transpose Y for TN GEMM 3 in NEAR/REST applyQt (Lever 3: NN→TN 1.5× on H200).
                // Yt buffer reused: far Yt transpose happens later (after j0 loop), no conflict.
                // Must wait for previous REST (s1) to finish reading Yt before overwriting it.
                const Real* Yt_near = nullptr;
                if (wnear > 0) {
                    if (evRestValid[pprev]) CUDA_CHECK(cudaStreamWaitEvent(s0, evRest[pprev], 0));
                    // Triple-buffered Yt: no need to wait for previous far-rest —
                    // Ytx[par] is distinct from Ytx[(par+2)%3] which far-rest reads.
                    // G2-PANEL V13: skip the Yt transpose when wfar_super==0 (no far
                    // region in this super-panel). The Yt feeds ONLY the NEAR/REST
                    // TN-GEMM3 (1.5x faster than NN) and the far GEMM3. When wfar==0
                    // the far GEMM3 is skipped, and for the NEAR/REST the transpose
                    // launch cost (~2-2.3us on s0 critical path × npan) exceeds the
                    // TN-vs-NN savings (mr×64×64 GEMM3, ~0.13us/panel at mr=1024).
                    // Net s0 savings: ~31us at fp64@1024 (15 panels × ~2.1us transpose).
                    if (!ytDisabled && saveForOrgqr && wfar_super > 0) {
                        // env gate LRQR_YT_NEAR removed (AUDIT_V6 §6 / Theorem U);
                        // Yt transpose for near applyQt is beneficial (Lever 3: NN→TN
                        // 1.5× on H200, measured) → default-ON. The Yt transpose
                        // mechanism stays live; the far path already defaults ON.
                        static bool ytNear = true;
                        if (ytNear) {
                            const int mr_near = m - row0;
                            dim3 bt_y(256, 1, 1);
                            dim3 gt_y(cdiv(mr_near, 64), cdiv(NB, 64), 1);
                            k_transpose_y<Real, 64, 64><<<gt_y, bt_y, 0, s0>>>(
                                SPV + size_t(j0) * m + row0, m, Ytx[par], NB, mr_near, NB);
                            Yt_near = Ytx[par];
                        }
                    }
                }
                CUDA_CHECK(cudaEventRecord(evPD, s0));
                // Tsp compound: feeds ONLY the super-panel-end far WY — run the
                // whole chain on sP (3rd stream) so the panel-critical s0 and the
                // rest-slice s1 never wait behind these k=mr GEMMs (nsys: 330ms of
                // the 65536 wall). TF32: the far tier's own accuracy class (§13.4).
                // Cor.FC (spec §6): Tsp rides s0 in-stream (no separate sP/cbT hop).
                // G2-PANEL V13: skip when wfar_super==0 — the compound T feeds ONLY
                // the far apply, which is skipped when there's no far region. Saves
                // 3 GEMMs + 1 copy per panel at the last super-panel (fp64@1024: 1
                // super-panel, wfar_super==0 → all 15 Tsp compounds skipped).
                if (!corFC) CUDA_CHECK(cudaStreamWaitEvent(sP, evPD, 0));
                if (wfar_super > 0 && j0 > 0) {
                    const int mr = m - sp;
                    CUBLAS_CHECK(cublasSetStream(cbTsp, sTsp));
                    if (use3xtf32) CUBLAS_CHECK(cublasSetMathMode(cbTsp, CUBLAS_DEFAULT_MATH));
                    NVTX_RANGE("Tsp_compound");
                    gemm(cbTsp, use3xtf32 ? false : tf32far, CUBLAS_OP_T, CUBLAS_OP_N, j0, NB, mr, Real(1),
                         SPV + sp, m, SPV + size_t(j0) * m + sp, m, Real(0), Zb, j0);
                    gemm(cbTsp, false, CUBLAS_OP_N, CUBLAS_OP_N, j0, NB, j0, Real(1),
                         Tsp, B, Zb, j0, Real(0), M1b, j0);
                    gemm(cbTsp, false, CUBLAS_OP_N, CUBLAS_OP_N, j0, NB, NB, Real(-1),
                         M1b, j0, Tpan + size_t(k) * NB * NB, NB,
                         Real(0), Tsp + size_t(j0) * B, B);
                    NVTX_POP();
                }
                if (wfar_super > 0)
                    CUDA_CHECK(cudaMemcpy2DAsync(Tsp + size_t(j0) * B + j0, B * sizeof(Real),
                        Tpan + size_t(k) * NB * NB, NB * sizeof(Real),
                        NB * sizeof(Real), NB, cudaMemcpyDeviceToDevice, sTsp));
                if (wnear > 0) {
                    // narrow: only the next panel's block, on the critical stream (§9 d=1);
                    // evRest[pprev] already waited above (before Yt transpose)
                    applyQt(A, lda, SPV, row0, j0, NB, Tpan + size_t(k) * NB * NB, NB,
                            sp + j0 + NB, std::min(NB, wnear), use3xtf32 ? false : tf32far, s0, Wn1, Wn2, Yt_near);
                    // rest: concurrent with the NEXT panel's cooperative kernel (headroom SMs)
                    const int wrest = wnear - NB;
                    if (wrest > 0) {
                        // REST on sN (separate from s1 far-rest) so REST dispatches
                        // concurrently with far-rest
                        CUDA_CHECK(cudaStreamWaitEvent(sN, evPD, 0));
                        if (evRestValid[pprev])
                            CUDA_CHECK(cudaStreamWaitEvent(sN, evRest[pprev], 0));
                        applyQt(A, lda, SPV, row0, j0, NB, Tpan + size_t(k) * NB * NB, NB,
                                sp + j0 + 2 * NB, wrest, use3xtf32 ? false : tf32far, sN, Wr1, Wr2, Yt_near);
                        CUDA_CHECK(cudaEventRecord(evRest[pj], sN));
                        evRestValid[pj] = true;
                    } else {
                        evRestValid[pj] = false;
                    }
                }
                // ---- c_L==1 panel inside a 2.5D super-panel: it owes its OWN far ----
                // Degenerate one-lane form of the 2.5D far: one compact-WY apply of
                // THIS panel's (V,T) over the far stripe. Without it the panel's
                // transform reaches the near region and never the far one, because
                // any25d suppresses the combined Tsp far. Tpan[k] and SPV column j0
                // are stable until the next super-panel, so no save-buffer replica is
                // needed here -- there is only one lane, hence nothing to replicate;
                // mv_replicate[3] is deliberately NOT raised, because it is not.
                // s1 ordering and the super-panel-end drain match the c_L>1 path.
                if (far25Route && wfar_super > 0) {
                    const int farc_sp = sp + Bcur;
                    const int fslot = k % 2;
                    if (far25Pending[fslot]) {
                        CUDA_CHECK(cudaStreamWaitEvent(s0, evFar25[fslot], 0));
                        far25Pending[fslot] = false;
                    }
                    CUDA_CHECK(cudaEventRecord(evPD, s0));
                    CUDA_CHECK(cudaStreamWaitEvent(s1, evPD, 0));
                    int fch = 0;
                    if (const char* e = getenv("LRQR_FARCHUNK")) fch = atoi(e);
                    if (fch <= 0) fch = farChunkFor(B, wfar_super);
                    if (use3xtf32 && maxFarchunk3x > 0) fch = std::min(fch, maxFarchunk3x);
                    for (int fc0 = farc_sp; fc0 < farc_sp + wfar_super; fc0 += fch) {
                        tel_far_chunks++;
                        mv_split[3]++; mv_pipeline[3]++;
                        applyQt(A, lda, SPV, row0, j0, NB, Tpan + size_t(k) * NB * NB, NB,
                                fc0, std::min(fch, farc_sp + wfar_super - fc0),
                                use3xtf32 ? false : tf32far, s1, Wa, Wb, nullptr, false, -1);
                    }
                    CUDA_CHECK(cudaEventRecord(evFar25[fslot], s1));
                    far25Pending[fslot] = true;
                } else if (wfar_super > 0) {
                    // Folded into the Tsp compound with a far region outstanding:
                    // lock the route so no later panel switches away and strands it.
                    // Reachable only from the FIRST panel of the super-panel (mk is
                    // maximal there, so if c_L>1 is legal anywhere it is legal there),
                    // hence the lock never costs liveness that was otherwise available.
                    sp_c1Locked = true;
                }
                }
            }
            // join: far update (and Tsp/SPV reuse) needs all rest slices complete
            for (int p = 0; p < 2; ++p)
                if (evRestValid[p]) CUDA_CHECK(cudaStreamWaitEvent(s0, evRest[p], 0));
            // c_L>1: wait for deferred far (s1) at super-panel boundary — next super-panel's
            // panel_25d reads A columns that the far writes.
            for (int s = 0; s < 2; ++s) {
                if (far25Pending[s]) {
                    CUDA_CHECK(cudaStreamWaitEvent(s0, evFar25[s], 0));
                    far25Pending[s] = false;
                }
            }
            // c_L>1: far update done per-panel inside the j0 loop — skip combined far.
            if (any25d) continue;
            // Cor.FC: evTsp on s0 (in-stream); s0 proceeds without cross-stream wait.
            CUDA_CHECK(cudaEventRecord(evTsp, sTsp));
            if (!corFC) CUDA_CHECK(cudaStreamWaitEvent(s0, evTsp, 0));
                // far update: one rank-Bcur pass (§4.3 Cor. 2)
            const int farc = sp + Bcur, wfar = n - farc;
            if (wfar > 0) {
                // regions ≥ farc may still be written by the previous far-rest on s1
                const int prev = (par + 2) % 3;
                if (evFarValid[prev]) CUDA_CHECK(cudaStreamWaitEvent(s0, evFar[prev], 0));
                // 3×TF32: also wait for far-rest from par-3 (same triple-buffer slot) to
                // complete before overwriting Yhi/Ylo/Ythi/Ytlo/Ytx[par]. With B=NB (1
                // panel/super-panel), the 3-super-panel pipeline slack is only 3 panels —
                // not enough for large far-rests to finish before the next pre-split.
                if (use3xtf32 && evFarValid[par])
                    CUDA_CHECK(cudaStreamWaitEvent(s0, evFar[par], 0));
                // Pre-transpose Y for TN GEMM 3 (1.5× faster than NN on H200).
                // Y is constant across all far chunks — transpose once, reuse everywhere.
                // Toggle via LRQR_NO_YT=1 (default: on).
                const Real* Yt_ptr = nullptr;
                if (!ytDisabled && saveForOrgqr) {
                    const int mr_far = m - sp;
                    dim3 bt(256, 1, 1);
                    dim3 gt(cdiv(mr_far, 64), cdiv(Bcur, 64), 1);
                    k_transpose_y<Real, 64, 64><<<gt, bt, 0, s0>>>(
                        SPV + sp, m, Ytx[par], Bcur, mr_far, Bcur);
                    Yt_ptr = Ytx[par];
                }
                // 3×TF32: pre-split Y and Yt once per super-panel (amortized over all far chunks).
                // Y is (mr_far × Bcur, lda=m); Yt is (Bcur × mr_far, lda=Bcur).
                int par3x = -1;
                if (use3xtf32) {
                    const int mr_far = m - sp;
                    { dim3 bt3(32, 8); dim3 gt3(cdiv(mr_far, 32), cdiv(Bcur, 8));
                      k_split_3xtf32_2d<Real><<<gt3, bt3, 0, s0>>>(
                          SPV + sp, m, Yhi[par], m, Ylo[par], m, mr_far, Bcur); }
                    if (!ytDisabled && saveForOrgqr) {
                        dim3 bt3(32, 8); dim3 gt3(cdiv(Bcur, 32), cdiv(mr_far, 8));
                        k_split_3xtf32_2d<Real><<<gt3, bt3, 0, s0>>>(
                            Ytx[par], Bcur, Ythi[par], Bcur, Ytlo[par], Bcur, Bcur, mr_far);
                    }
                    par3x = par;
                }
                const int wnext = lookahead ? std::min(B, wfar) : wfar;
                // slice covering the next super-panel (critical stream): 3×TF32 tier at large n
                // (slice GEMMs are huge there — roofline §13.4), exact tier below
                tel_far_chunks++;   // ENGAGEMENT TELEMETRY: lookahead far slice dispatched
                applyQt(A, lda, SPV, sp, 0, Bcur, Tsp, B, farc, wnext,
                        use3xtf32 ? false : tf32far,
                        s0, Wa, Wb, Yt_ptr, /*smLimit=*/false, par3x);
                if (lookahead && wfar > wnext) {
                    CUDA_CHECK(cudaEventRecord(evSlice, s0));
                    CUDA_CHECK(cudaStreamWaitEvent(s1, evSlice, 0));
                    if (evFarValid[prev]) CUDA_CHECK(cudaStreamWaitEvent(s1, evFar[prev], 0));
                    // Defer far-rest to next iteration: save state and issue after
                    // the next super-panel's panel 0 launch (spec §11 overlap).
                    int fch = 0;
                    if (const char* e = getenv("LRQR_FARCHUNK")) fch = atoi(e);
                    if (fch <= 0) fch = farChunkFor(B, wfar);
                    if (use3xtf32 && maxFarchunk3x > 0) fch = std::min(fch, maxFarchunk3x);
                    last_fch = fch;   // ENGAGEMENT TELEMETRY: far-rest chunk size (c_L=1 path)
                    dfr = {A, lda, SPV, sp, Bcur, Tsp, B, farc, wnext, wfar, fch,
                           Wa, Wb, Yt_ptr, par, prev, true, use3xtf32 ? false : tf32far, par3x};
                    // F8 Step-1: accumulate exact far-rest flops (2 big GEMMs + 1 small).
                    // GEMM1: 2·B·nc·mr, GEMM2: 2·B·nc·B, GEMM3: 2·mr·nc·B → sum over chunks.
                    if (staget) {
                        const int mr_f = m - sp;
                        stagetFarFlops += 2.0 * (double)Bcur * (double)(wfar - wnext)
                                          * (2.0 * (double)mr_f + (double)Bcur);
                    }
                    // Don't set evFarValid yet — it's set when the far-rest is actually issued
                    evFarValid[par] = false;
                } else {
                    evFarValid[par] = false;
                }
            }
        }
        // Issue final deferred far-rest
        if (dfr.valid && dfr.wfar > dfr.wnext) {
            if (evFarValid[dfr.prev]) CUDA_CHECK(cudaStreamWaitEvent(s1, evFar[dfr.prev], 0));
            for (int fc0 = dfr.farc + dfr.wnext; fc0 < dfr.farc + dfr.wfar; fc0 += dfr.fch) {
                tel_far_chunks++;   // ENGAGEMENT TELEMETRY: final far-rest chunk dispatched
                applyQt(dfr.A, dfr.lda, dfr.SPV, dfr.sp, 0, dfr.Bcur, dfr.Tsp, dfr.B,
                        fc0, std::min(dfr.fch, dfr.farc + dfr.wfar - fc0),
                        dfr.tf32far, s1, dfr.Wa, dfr.Wb, dfr.Yt_ptr, /*smLimit=*/true,
                        dfr.par3x);
            }
            CUDA_CHECK(cudaEventRecord(evFar[dfr.par], s1));
            evFarValid[dfr.par] = true;
        }
        // F8 Step-1: record stream-span stop events BEFORE the syncs (captures all async work)
        if (staget) {
            CUDA_CHECK(cudaEventRecord(evSt0b, s0));
            CUDA_CHECK(cudaEventRecord(evSt1b, s1));
        }
        // T3: reset L2 access-policy window so subsequent kernels are unaffected
        if (useL2win) {
            cudaStreamAttrValue lw{};
            lw.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(s0, cudaStreamAttributeAccessPolicyWindow, &lw);
            cudaStreamSetAttribute(s1, cudaStreamAttributeAccessPolicyWindow, &lw);
            // Reset device-level persisting cache size (set in create() via
            // cudaDeviceSetLimit). Without this, the limit persists into the next
            // validate-all tier, causing TF32-level accuracy in fp32 tests.
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        }
        CUDA_CHECK(cudaStreamSynchronize(s0));
        CUDA_CHECK(cudaStreamSynchronize(s1));
        CUDA_CHECK(cudaStreamSynchronize(sN));
        if (usePersist) CUDA_CHECK(cudaStreamSynchronize(sP));
        if (green) cuCtxPopCurrent(nullptr);
        if (use25d && cLused > 1)
            fprintf(stderr, "[2.5D] c_L=%d used (m=%d n=%d)\n", cLused, m, n);
        gemmsWarmed = true;   // all cuBLAS kernels now loaded → persist safe from next call
        // F8 Step-1: report per-stage span (s0=panel+slice carrier, s1=far-rest carrier).
        // overlap = s0_span + s1_span - wall; far_rest_TF vs ~47 TF fp64 ceiling ⇒ compute/HBM verdict.
        if (staget) {
            float s0ms = 0, s1ms = 0;
            cudaEventElapsedTime(&s0ms, evSt0a, evSt0b);
            cudaEventElapsedTime(&s1ms, evSt1a, evSt1b);
            const double s0d = s0ms, s1d = s1ms;
            const double wall = std::max(s0d, s1d);  // span max ≈ wall (sync covers both)
            const double overlap = s0d + s1d - wall;
            const double phi = (s0d > 0 && s1d > 0) ? overlap / std::min(s0d, s1d) : 0.0;
            const double farTF = (s1d > 1e-9) ? stagetFarFlops / (s1d * 1e-3) / 1e12 : 0.0;
            const double farAI = 0.0;  // reported by ncu (see RESULT.md Step 1d)
            fprintf(stderr, "[staget] m=%d n=%d B=%d la=%d tf32far=%d | "
                    "s0_span=%.1fms s1_span=%.1fms wall≈%.1fms | "
                    "overlap=%.1fms phi=%.2f | far_flops=%.3e far_TF=%.1f (%.0f%% of 53.6 ceiling) "
                    "exposed_panel≈%.1fms\n",
                    m, n, B, (int)lookahead, (int)tf32far,
                    s0d, s1d, wall, overlap, phi, stagetFarFlops, farTF,
                    (farTF / 53.6) * 100.0, std::max(0.0, wall - s1d));
            (void)farAI;
        }
        if (useDomino && getenv("LRQR_PHASET")) {
            unsigned long long h[8];
            CUDA_CHECK(cudaMemcpy(h, domT, sizeof(h), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemset(domT, 0, sizeof(h)));
            const double us = 1.0 / 1.98e3;   // ~1.98 GHz
            fprintf(stderr, "[domino phases µs/call: init %.0f cols %.0f B1 %.0f (stage %.0f) "
                    "B2a %.0f B2b %.0f B3 %.0f wout %.0f]\n", h[0] * us, h[1] * us, h[2] * us,
                    h[7] * us, h[3] * us, h[4] * us, h[5] * us, h[6] * us);
        }
        // ENGAGEMENT TELEMETRY (spec §10 Theorem-U; AUDIT_V6 §6; RESEARCH_MEMO):
        // one stderr line per process (first geqrf) witnessing which mechanisms engaged
        // in the default path. Observability-only — NEVER gates any dispatch (Theorem U).
        // Proves "live by default" by engagement counts, not grep absence.
        {
            static bool telemetry_printed = false;
            if (!telemetry_printed) {
                const char* tier = std::is_same<Real, double>::value ? "fp64"
                                 : use3xtf32 ? "3xtf32"
                                 : tf32far ? "tf32" : "fp32";
                const int raw_path = tel_far_chunks - tel_split_path;
                const bool noCoop = (getenv("LRQR_FORCE_COOP") == nullptr);
                const bool far25en = true;
                // Step 5 — c_ch re-fit. The former c_ch = ceil(log2(Nmax)) was a crude
                // chain-LENGTH upper bound (8-10 at n=65536) that conflated the domino
                // block count with the chain-HOP overhead. The spec (§4 Lemma N, §7) says
                // the chain is pipelined ("dominos overlap the latency terms down to the
                // ∑_λ ≤ L·max_λ envelope") so the exposed chain overhead is L (hierarchy
                // depth) × per-rung overhead, NOT log2(chain length). The re-fit c_ch =
                // the number of STRICTLY NESTING ν-ladder rungs above the materialized
                // leaf (nu[i] > nu[i-1]): a ⊕-chain hop fires exactly where a rung's
                // panel spans multiple lower-rung panels. The leaf rung IS the domino
                // (flat — its capacity-ideal nu is the b that materializes to NB, not a
                // hop), and equal-capacity rungs (DSM↔L2 / L2→HBM both bounded by the
                // persisting-L2 window on this part) have equal ν → they collapse to ONE
                // hop, a §10 Theorem-U parameter→ limit, not an exclusion. This lands in
                // [1,3] (typically 2: shared→DSM + DSM↔L2/HBM) and is validated by
                // fitting E(n)=1/(1+0.75·c_ch·n_crit/n) from fresh carrier-path timings
                // (see W3STEP456_MEMO.md §5). This is observability-only (Theorem U).
                int c_ch_refit = 0;
                for (int i = 1; i < 4; ++i)
                    if (sched.rung[i].nu > sched.rung[i - 1].nu) c_ch_refit++;
                c_ch_refit = std::max(1, c_ch_refit);
                const int c_ch = c_ch_refit;
                // W6(G7) footprint telemetry: actual Plan device bytes (footprintBytes,
                // summed over every cudaMalloc in create()) vs the LAPACK compact baseline
                // m*n (in-place A: V⊗R) + n (tau) + n*NB (block-T WY workspace). The gap
                // = spec-mandated triple-buffered SPVx + 2.5D saves + resident rVg that
                // pure LAPACK lacks (RESEARCH_MEMO §5). G7 gate (a): telemetry added +
                // shows the reduction vs LAPACK baseline.
                const size_t lapack_bytes = (size_t(m) * n + size_t(n) + size_t(npanels) * NB * NB)
                                            * sizeof(Real);
                const size_t A_bytes = size_t(m) * n * sizeof(Real);
                fprintf(stderr,
                    "[TQR_TELEMETRY] tier=%s n=%d m=%d B=%d c_L=%d fch=%d splitK=%d "
                    "far_chunks=%d split_path=%d raw_path=%d domino=%s dsm=%s l2win=%s "
                    "corFC=%s far25=%s noCoop=%s Wfront=%d r=%d c_ch=%d "
                    "footprint_bytes=%zu A_bytes=%zu plan_bytes=%zu lapack_bytes=%zu "
                    "plan_over_lapack=%.2fx\n",
                    tier, n, m, B, last_cL, last_fch, splitK3x,
                    tel_far_chunks, tel_split_path, raw_path,
                    useDomino ? "true" : "false", useDsm ? "true" : "false",
                    useL2win ? "true" : "false", corFC ? "true" : "false",
                    far25en ? "true" : "false", noCoop ? "true" : "false",
                    sched.rung[3].W, sched.rung[3].r, c_ch,
                    footprintBytes + A_bytes, A_bytes, footprintBytes, lapack_bytes,
                    lapack_bytes > 0 ? double(footprintBytes) / double(lapack_bytes) : 0.0);
                // 2.5D move witnesses and the replication actually used, so a run
                // can be scored on whether the four moves fired rather than on a
                // reading of the source.
                fprintf(stderr,
                    "[TQR_MOVES] c_L=%d lanedepth=%d lanes_wave=%d "
                    "merge_levels=%lld merge_rounds=%lld pipe=%d early=%lld/%lld "
                    "split=[%lld,%lld,%lld,%lld] replicate=[%lld,%lld,%lld,%lld] "
                    "combine=[%lld,%lld,%lld,%lld] pipeline=[%lld,%lld,%lld,%lld] "
                    "c_rung=[%d,%d,%d,%d] instrumented=[%d,%d,%d,%d] "
                    // THE COMPLIANCE NUMBERS (31:345, 31:390). At rung 1 the ordered
                    // frontiers are the panels and the tree depth is log2(c_L), so
                    // the theorem predicts S_1 = Theta(K_1 + D_1).
                    //
                    // S1_upper is an UPPER BOUND, not S_1: it is the panel count
                    // plus every merge ROUND issued, i.e. what the schedule costs if
                    // nothing overlaps. It is honest about two different effects.
                    // The wavefront (Step 2) lowers it directly, by paying D levels
                    // in one round -- visible as merge_rounds < merge_levels. The
                    // diagonal edge (Step 3) does NOT lower it, because it changes
                    // when a round is RELEASED, not how many are issued; its effect
                    // shows up in early=released/lanes and in wall time. Reporting
                    // this as if it were S_1 would credit Step 3 with nothing and
                    // Step 2 with everything, so it is named for what it measures.
                    // Q_3 against B_3(c) from 31:221. The theorem (31:297) says
                    // Q_lambda = Theta(Q_lambda^LB) with B_lambda the budget, so this
                    // ratio should sit at a small constant. B_words is the formula's
                    // value, computed in computeSchedule for nu*; Q3 is counted in
                    // applyQt from the actual trailing-block dimensions.
                    "Q3far=%.4g Q2near=%.4g B3=%.4g B2=%.4g Q3_over_B3=%.2f Q2_over_B2=%.2f "
                    "S1_upper=%lld S1_KplusD=%lld S1_ratio=%.2f\n",
                    cLused, [] { const char* e = getenv("LRQR_LANEDEPTH"); return e ? atoi(e) : 2; }(),
                    mv_lanes_wave, mv_merge_levels, mv_merge_launches,
                    (int)pipe25d(), mv_pipe_released, mv_pipe_lanes,
                    mv_split[0], mv_split[1], mv_split[2], mv_split[3],
                    mv_replicate[0], mv_replicate[1], mv_replicate[2], mv_replicate[3],
                    mv_combine[0], mv_combine[1], mv_combine[2], mv_combine[3],
                    mv_pipeline[0], mv_pipeline[1], mv_pipeline[2], mv_pipeline[3],
                    sched.rung[0].c, sched.rung[1].c, sched.rung[2].c, sched.rung[3].c,
                    mv_instrumented[0], mv_instrumented[1], mv_instrumented[2], mv_instrumented[3],
                    // Per-boundary, both rungs. 31:297 requires Q_lambda =
                    // Theta(Q_lambda^LB) SIMULTANEOUSLY for every lambda, so
                    // reporting rung 3 alone can call the schedule optimal while
                    // rung 2 is 17x over. On this part capBytes[2]==capBytes[3]
                    // (both the persisting-L2 window), so nu_2 == nu_3 and the two
                    // budgets coincide -- the rungs collapse to one hop -- but the
                    // TRAFFIC does not, and that is the point of emitting both.
                    mv_words_upd, mv_words_near,
                    sched.rung[3].B_words, sched.rung[2].B_words,
                    mv_words_upd  / std::max(1.0, sched.rung[3].B_words),
                    mv_words_near / std::max(1.0, sched.rung[2].B_words),
                    (long long)(npanels + mv_merge_launches),
                    (long long)(npanels + (cLused > 1 ? fanoutDepth(cLused, 2) : 0)),
                    (double)(npanels + mv_merge_launches) /
                        std::max(1.0, (double)(npanels +
                            (cLused > 1 ? fanoutDepth(cLused, 2) : 0))));
                telemetry_printed = true;
            }
        }
    }

    // Explicit thin Q (m×n) from factored A + stored Tpan (validation).
    void orgqr(const Real* A, size_t lda, Real* Q, size_t ldq) {
        // W6(G7) Phase-2: the per-panel merge save (Vm_pan/TmM_pan/domTpanL_pan)
        // is gated on saveForOrgqr (false in LRQR_ONLY production). orgqr is only
        // called in the bench/validation path where saveForOrgqr is true.
        // G7-WRAP: the c_L=1 path (lines below, the else branch) uses ONLY
        // Wax/Wbx/Pbuf/Tpan — all unconditionally allocated. The c_L>1 path uses
        // Vm_pan/TmM_pan/domTpanL_pan (saveForOrgqr-gated). The blanket assert is
        // relaxed to a per-panel check so the single-copy path (validation_copy=false)
        // can call orgqr at c_L=1 without the 5+ GiB per-panel-save + Ytx overhead.
        CUDA_CHECK(cudaEventRecord(evIn, 0));
        if (green) cuCtxPushCurrent(ctxB);
        CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        // A2: every column count inside this function is Q's WIDTH, which is
        // min(m,n) -- not the matrix n. On m>=n cells nq == n, so no existing
        // measurement moves. With m<n and nq left as n, the three reflector GEMMs
        // below write past the end of Q; that is the illegal access probe 9436 hit
        // at what was line 8180.
        const int nq = kmax;
        {
            // The thin Q is m x min(m,n), not m x n. For m<n the factorization has
            // only m reflectors and a wider Q would have undefined columns.
            size_t total = size_t(m) * kmax;
            unsigned grid = (unsigned)((total + TPB - 1) / TPB);
            k_eye<Real><<<grid, TPB, 0, s0>>>(Q, ldq, m, kmax);
        }
        Real *Wa = Wax[0], *Wb = Wbx[0];
        for (int k = npanels - 1; k >= 0; --k) {
            const int row0 = k * NB, mk = m - row0;
            if (cL_pan && cL_pan[k] > 1) {
                const int cL = cL_pan[k], nm = nm_pan[k];
                assert(saveForOrgqr && "orgqr c_L>1 needs per-panel merge saves "
                       "(saveForOrgqr=false — Vm_pan/TmM_pan/domTpanL_pan not allocated)");
                const size_t perPan = size_t(cLmax) * NB * NB;
                // Merge + per-lane WY reconstruction GEMMs use CUBLAS_COMPUTE_32F (fp32);
                // force DEFAULT_MATH so the handle's TF32 mode (set for the bulk tier)
                // cannot downgrade them. Mirrors mergeApplyQt (memo §5).
                cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
                CUBLAS_CHECK(cublasGetMathMode(cb0, &saveMath));
                const bool forceFp32 = !std::is_same<Real, double>::value &&
                                       saveMath != CUBLAS_DEFAULT_MATH;
                if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, CUBLAS_DEFAULT_MATH));
                // Q_k = diag(Q_{k,j}) · Q_merge.  Apply Q_merge first (reverse node
                // order, T_μ not transposed — applying Q, not Q^T), then per-lane WY.
                for (int node = nm - 1; node >= 0; --node) {
                    const int topIdx = nodes_pan[k * cLmax + node].x;
                    const int botIdx = nodes_pan[k * cLmax + node].y;
                    const int top = row0 + slabStart_pan[k * cLmax + topIdx];
                    const int bot = row0 + slabStart_pan[k * cLmax + botIdx];
                    const Real* Wmu = Vm_pan + size_t(k) * perPan + size_t(node) * NB * NB;
                    const Real* Tmu = TmM_pan + size_t(k) * perPan + size_t(node) * NB * NB;
                    gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nq, NB, Real(1),
                         Wmu, NB, Q + bot, (int)ldq, Real(0), Wa, NB);
                    { int ngr = cdiv(NB * n, TPB);
                      k_merge_elem<Real, NB><<<ngr, TPB, 0, s0>>>(Q, ldq, top, 0, Wa, NB, nq, 0); }
                    gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nq, NB, Real(1),
                         Tmu, NB, Wa, NB, Real(0), Wb, NB);
                    { int ngr = cdiv(NB * n, TPB);
                      k_merge_elem<Real, NB><<<ngr, TPB, 0, s0>>>(Q, ldq, top, 0, Wb, NB, nq, 1); }
                    gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nq, NB, Real(-1),
                         Wmu, NB, Wb, NB, Real(1), Q + bot, (int)ldq);
                }
                // Per-lane WY: Q_{k,j} = I - Y_j T_j Y_j^T (disjoint rows, any order)
                for (int j = 0; j < cL; ++j) {
                    const int sStart = slabStart_pan[k * cLmax + j];
                    const int sSize  = slabSize_pan[k * cLmax + j];
                    const Real* Tj = domTpanL_pan + size_t(k) * perPan + size_t(j) * NB * NB;
                    k_extractY<Real><<<cdiv(sSize * NB, TPB), TPB, 0, s0>>>(
                        A + size_t(k) * NB * lda, lda, row0 + sStart, sSize, NB, Pbuf, sSize);
                    gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nq, sSize, Real(1),
                         Pbuf, sSize, Q + row0 + sStart, (int)ldq, Real(0), Wa, NB);
                    gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nq, NB, Real(1),
                         Tj, NB, Wa, NB, Real(0), Wb, NB);
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, sSize, nq, NB, Real(-1),
                     Pbuf, sSize, Wb, NB, Real(1), Q + row0 + sStart, (int)ldq);
                }
                if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, saveMath));
            } else {
                // c_L=1: standard WY apply using Tpan[k]
                k_extractY<Real><<<cdiv(mk * NB, TPB), TPB, 0, s0>>>(
                    A + size_t(k) * NB * lda, lda, row0, mk, NB, Pbuf, mk);
                gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, nq, mk, Real(1), Pbuf, mk,
                     Q + row0, (int)ldq, Real(0), Wa, NB);
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, nq, NB, Real(1),
                     Tpan + size_t(k) * NB * NB, NB, Wa, NB, Real(0), Wb, NB);
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, mk, nq, NB, Real(-1), Pbuf, mk,
                     Wb, NB, Real(1), Q + row0, (int)ldq);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(s0));
        if (green) cuCtxPopCurrent(nullptr);
    }

    // ========================================================================
    // ormqr — apply Q or Q^T to a general C, from either side, WITHOUT FORMING Q.
    //
    // This is the capability that makes the compact representation worth having.
    // orgqr is already this operation applied to the identity: it extracts Y per
    // panel from tril(A) with k_extractY and pairs it with the retained Tpan block,
    // so nothing new is stored -- Tpan is npanels*NB*NB and survives geqrf, and Y
    // lives in A in LAPACK layout. Peak extra memory here is Pbuf (mk x NB) plus
    // two NB x nc panels, against m x k for an explicit Q.
    //
    // Q = Q_0 Q_1 ... Q_{p-1} with Q_k = I - Y_k T_k Y_k^T, so:
    //   'L','N'  C := Q C     apply Q_{p-1} first  -> k DESCENDING, T as-is
    //   'L','T'  C := Q^T C   apply Q_0^T first    -> k ASCENDING,  T TRANSPOSED
    //   'R','N'  C := C Q     apply Q_0 first      -> k ASCENDING,  T as-is
    //   'R','T'  C := C Q^T   apply Q_{p-1}^T first-> k DESCENDING, T TRANSPOSED
    // Left:  C[row0:,:] -= Y (op(T) (Y^T C[row0:,:]))
    // Right: C[:,row0:] -= ((C[:,row0:] Y) op(T)) Y^T
    //
    // RETURNS FALSE, rather than computing something wrong, when a panel used
    // c_L>1: those panels carry a per-lane merge tree (domTpanL_pan/nodes_pan) and
    // the single-T WY form below does not represent them. c_L=1 holds on every
    // square cell by construction -- NOT by an assert (I previously wrote that there
    // was an m==n assert in panel(); there is none on this tree, that was carried
    // over from a deleted branch). It holds because the lane formula reduces, on
    // square geometry, to laneNeed = nu_1 / (SLAB * fanout^bias): the problem size
    // cancels, and with the shipped bias=2 the threshold is 128*16^2 = 32768 against
    // nu_1 = 482, so c_L is 1 at EVERY square size. This covers the
    // square manifest; tall cells that select c_L=2 report unsupported and say so.
    // ========================================================================
    bool ormqr(const Real* A, size_t lda, char side, char trans,
               Real* C, size_t ldc, int rows, int cols) {
        if (cols <= 0 || rows <= 0) return true;
        const bool left = (side == 'L' || side == 'l');
        const bool tr   = (trans == 'T' || trans == 't');
        if (left  && rows != m) return false;   // C must be m x cols
        if (!left && cols != m) return false;   // C must be rows x m
        // c_L>1 panels carry a per-lane merge tree. Supported for the LEFT cases
        // below (Q_k = diag(Q_{k,j})·Q_merge, mirroring orgqr, with the transpose
        // flipping only T_mu and reversing the two stages). The RIGHT cases would
        // need the mirrored merge scatter as well and are refused rather than
        // approximated -- returning false is a reportable outcome, a wrong apply is not.
        bool any_multi = false;
        if (cL_pan) for (int k = 0; k < npanels; ++k) if (cL_pan[k] > 1) any_multi = true;
        if (any_multi && !left) return false;
        if (any_multi && !saveForOrgqr) return false;   // merge saves not allocated

        const int nc = left ? cols : rows;                  // the "other" extent
        if ((size_t)nc * NB > waxElemsCached) return false;  // Wa/Wb capacity

        CUDA_CHECK(cudaEventRecord(evIn, 0));
        if (green) cuCtxPushCurrent(ctxB);
        CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        Real *Wa = Wax[0], *Wb = Wbx[0];
        const cublasOperation_t opT = tr ? CUBLAS_OP_T : CUBLAS_OP_N;
        const bool descending = (left && !tr) || (!left && tr);

        // c_L>1 left-side panel: Q_k = diag(Q_{k,j}) · Q_merge. Applying Q_k means
        // Q_merge first (nodes in REVERSE order) then the per-lane WY blocks;
        // applying Q_k^T reverses both -- per-lane first, then the nodes FORWARD
        // with T_mu transposed. The node block form is
        //   Y_mu^T C = C_top + Wmu^T C_bot ; W = op(T_mu)(Y_mu^T C)
        //   C_top -= W ; C_bot -= Wmu W
        // which is exactly what k_merge_elem's op=0/op=1 modes implement.
        auto panel_multi_left = [&](int k, int row0, int cols_) {
            const int cL = cL_pan[k], nm = nm_pan[k];
            const size_t perPan = size_t(cLmax) * NB * NB;
            cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
            CUBLAS_CHECK(cublasGetMathMode(cb0, &saveMath));
            const bool forceFp32 = !std::is_same<Real, double>::value &&
                                   saveMath != CUBLAS_DEFAULT_MATH;
            if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, CUBLAS_DEFAULT_MATH));
            auto per_lane = [&]() {
                for (int j = 0; j < cL; ++j) {
                    const int sStart = slabStart_pan[k * cLmax + j];
                    const int sSize  = slabSize_pan[k * cLmax + j];
                    const Real* Tj = domTpanL_pan + size_t(k) * perPan + size_t(j) * NB * NB;
                    k_extractY<Real><<<cdiv(sSize * NB, TPB), TPB, 0, s0>>>(
                        A + size_t(k) * NB * lda, lda, row0 + sStart, sSize, NB, Pbuf, sSize);
                    gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, cols_, sSize, Real(1),
                         Pbuf, sSize, C + row0 + sStart, (int)ldc, Real(0), Wa, NB);
                    gemm(cb0, false, opT, CUBLAS_OP_N, NB, cols_, NB, Real(1),
                         Tj, NB, Wa, NB, Real(0), Wb, NB);
                    gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, sSize, cols_, NB, Real(-1),
                         Pbuf, sSize, Wb, NB, Real(1), C + row0 + sStart, (int)ldc);
                }
            };
            auto merge_nodes = [&](bool forward) {
                for (int q = 0; q < nm; ++q) {
                    const int node = forward ? q : (nm - 1 - q);
                    const int topIdx = nodes_pan[k * cLmax + node].x;
                    const int botIdx = nodes_pan[k * cLmax + node].y;
                    const int top = row0 + slabStart_pan[k * cLmax + topIdx];
                    const int bot = row0 + slabStart_pan[k * cLmax + botIdx];
                    const Real* Wmu = Vm_pan  + size_t(k) * perPan + size_t(node) * NB * NB;
                    const Real* Tmu = TmM_pan + size_t(k) * perPan + size_t(node) * NB * NB;
                    gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, cols_, NB, Real(1),
                         Wmu, NB, C + bot, (int)ldc, Real(0), Wa, NB);
                    { int ngr = cdiv(NB * cols_, TPB);
                      k_merge_elem<Real, NB><<<ngr, TPB, 0, s0>>>(C, ldc, top, 0, Wa, NB, cols_, 0); }
                    gemm(cb0, false, opT, CUBLAS_OP_N, NB, cols_, NB, Real(1),
                         Tmu, NB, Wa, NB, Real(0), Wb, NB);
                    { int ngr = cdiv(NB * cols_, TPB);
                      k_merge_elem<Real, NB><<<ngr, TPB, 0, s0>>>(C, ldc, top, 0, Wb, NB, cols_, 1); }
                    gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, NB, cols_, NB, Real(-1),
                         Wmu, NB, Wb, NB, Real(1), C + bot, (int)ldc);
                }
            };
            if (!tr) { merge_nodes(/*forward=*/false); per_lane(); }
            else     { per_lane(); merge_nodes(/*forward=*/true); }
            if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, saveMath));
        };

        for (int idx = 0; idx < npanels; ++idx) {
            const int k = descending ? (npanels - 1 - idx) : idx;
            const int row0 = k * NB, mk = m - row0;
            if (mk <= 0) continue;
            if (cL_pan && cL_pan[k] > 1) { panel_multi_left(k, row0, cols); continue; }
            k_extractY<Real><<<cdiv(mk * NB, TPB), TPB, 0, s0>>>(
                A + size_t(k) * NB * lda, lda, row0, mk, NB, Pbuf, mk);
            const Real* Tk = Tpan + size_t(k) * NB * NB;
            if (left) {
                gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, cols, mk, Real(1),
                     Pbuf, mk, C + row0, (int)ldc, Real(0), Wa, NB);
                gemm(cb0, false, opT, CUBLAS_OP_N, NB, cols, NB, Real(1),
                     Tk, NB, Wa, NB, Real(0), Wb, NB);
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, mk, cols, NB, Real(-1),
                     Pbuf, mk, Wb, NB, Real(1), C + row0, (int)ldc);
            } else {
                Real* Cc = C + size_t(row0) * ldc;
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, rows, NB, mk, Real(1),
                     Cc, (int)ldc, Pbuf, mk, Real(0), Wa, rows);
                gemm(cb0, false, CUBLAS_OP_N, opT, rows, NB, NB, Real(1),
                     Wa, rows, Tk, NB, Real(0), Wb, rows);
                gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_T, rows, mk, NB, Real(-1),
                     Wb, rows, Pbuf, mk, Real(1), Cc, (int)ldc);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(s0));
        if (green) cuCtxPopCurrent(nullptr);
        return true;
    }

    // Apply R (trmm) or solve with R (trsm) from the compact factored form. R is
    // in triu(A) in LAPACK layout, so these are thin wrappers -- but they are the
    // difference between "the factorization is in there somewhere" and a usable
    // factored operator, and they cost no extra storage at all.
    //   solve=false : B := alpha * R * B        (R is k x k, upper, non-unit)
    //   solve=true  : B := alpha * R^{-1} * B
    void trm_R(const Real* A, size_t lda, Real* Bm, size_t ldb, int nrhs,
               bool solve, Real alpha = Real(1)) {
        const int kk = kmax;
        if (kk <= 0 || nrhs <= 0) return;
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        if (solve)
            trsm(cb0, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                 CUBLAS_DIAG_NON_UNIT, kk, nrhs, alpha, A, (int)lda, Bm, (int)ldb);
        else
            trmm(cb0, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                 CUBLAS_DIAG_NON_UNIT, kk, nrhs, alpha, A, (int)lda, Bm, (int)ldb);
        CUDA_CHECK(cudaStreamSynchronize(s0));
    }

    // W6-P3 single-copy validation: form a ROW-TILE of thin Q — X (m×b, col-major
    // ld=ldx) = Q^T · [e_{i0},...,e_{i1-1}], so that X[0:n, :] = Q[i0:i1, :]^T
    // (n×b). The caller seeds X with k_eye_rows (columns [i0:i1] of the identity)
    // and this method applies the panel WY transforms in FORWARD order with the
    // block-T TRANSPOSED (H_panel^T = I - Y·T^T·Y^T), computing Q^T·X. This is the
    // memory-tiled counterpart to orgqr(): the full orgqr forms the whole (m×n) Q
    // and OOMs at fp64@98304² (A + Q + Plan > 140.4 GiB FB); orgqr_rows forms one
    // (m×b) tile at a time (peak A + m·b + Plan ≈ 84 GiB at 98304², b=1024).
    //
    // PROTOTYPE scope: c_L=1 path only (the large-square-n regime — telemetry
    // confirms c_L=1 at fp64@98304²). For c_L>1 panels the method aborts (the
    // merge-path WY transpose is deferred — see SINGLECOPY_VALIDATION_DESIGN.md
    // §4). The GEMMs mirror orgqr's c_L=1 branch (lrqr.cuh:7521-7531) with three
    // changes: (1) panel loop FORWARD (k=0..npanels-1) since Q^T = H_P^T..H_1^T
    // applies H_1^T first; (2) the block-T GEMM uses OP_T (T^T) for H_panel^T;
    // (3) the Q-dimension is b (tile width) instead of n. Cost = O(m·n·b) per
    // tile × (m/b) tiles = O(m·n²) — identical flop count to full orgqr.
    void orgqr_rows(const Real* A, size_t lda, Real* X, size_t ldx, int i0, int i1) {
        // G7-WRAP: orgqr_rows uses ONLY the c_L=1 path (aborts at c_L>1 below), which
        // needs Wax[0]/Wbx[0]/Pbuf/Tpan — all unconditionally allocated in create().
        // The per-panel saves (Vm_pan/TmM_pan/domTpanL_pan, gated by saveForOrgqr) are
        // NOT needed. This allows the single-copy path to use validation_copy=false
        // (no Ytx/per-panel saves → ~5 GiB smaller plan → fits at fp64@131072²).
        assert(Wax[0] && Wbx[0] && Pbuf && Tpan && "orgqr_rows needs Wax/Wbx/Pbuf/Tpan");
        const int b = i1 - i0;
        assert(b > 0 && i0 >= 0 && i1 <= m);
        CUDA_CHECK(cudaEventRecord(evIn, 0));
        if (green) cuCtxPushCurrent(ctxB);
        CUDA_CHECK(cudaStreamWaitEvent(s0, evIn, 0));
        CUBLAS_CHECK(cublasSetStream(cb0, s0));
        // Seed X (m×b) = columns [i0:i1] of the m×m identity.
        {
            size_t total = size_t(m) * b;
            unsigned grid = (unsigned)((total + TPB - 1) / TPB);
            k_eye_rows<Real><<<grid, TPB, 0, s0>>>(X, ldx, m, i0, b);
        }
        Real *Wa = Wax[0], *Wb = Wbx[0];   // (NB×n) -> use first b cols (NB×b)
        cublasMath_t saveMath = CUBLAS_DEFAULT_MATH;
        CUBLAS_CHECK(cublasGetMathMode(cb0, &saveMath));
        const bool forceFp32 = !std::is_same<Real, double>::value && saveMath != CUBLAS_DEFAULT_MATH;
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, CUBLAS_DEFAULT_MATH));
        // FORWARD panel order with T^T (OP_T): Q^T = H_P^T·...·H_1^T applies H_1^T
        // first; each H_panel^T = I - Y·T^T·Y^T. Validated vs full orgqr at fp64@4096
        // (row-tiled resid 3.45e-15 vs full-Q 3.18e-15, MATCH — commit 2 self-check).
        for (int k = 0; k < npanels; ++k) {
            const int row0 = k * NB, mk = m - row0;
            if (cL_pan && cL_pan[k] > 1) {
                // c_L>1 merge path: the WY transpose (T^T) + forward merge-node
                // order is deferred (SINGLECOPY_VALIDATION_DESIGN.md §4). Abort so
                // the prototype does not silently produce a wrong Q row-tile.
                fprintf(stderr, "[LRQR] orgqr_rows: c_L=%d>1 at panel %d (n=%d) — "
                        "single-copy row-tiled Q not implemented for c_L>1; abort.\n",
                        cL_pan[k], k, n);
                fflush(stderr);
                abort();
            }
            // c_L=1 WY apply (H_panel^T = I - Y·T^T·Y^T):
            //   Wa = Y^T · X[row0:m, :]      (NB×b)
            //   Wb = T^T · Wa                (NB×b)
            //   X[row0:m, :] -= Y · Wb       (mk×b)
            k_extractY<Real><<<cdiv(mk * NB, TPB), TPB, 0, s0>>>(
                A + size_t(k) * NB * lda, lda, row0, mk, NB, Pbuf, mk);
            gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, b, mk, Real(1), Pbuf, mk,
                 X + row0, (int)ldx, Real(0), Wa, NB);
            gemm(cb0, false, CUBLAS_OP_T, CUBLAS_OP_N, NB, b, NB, Real(1),
                 Tpan + size_t(k) * NB * NB, NB, Wa, NB, Real(0), Wb, NB);
            gemm(cb0, false, CUBLAS_OP_N, CUBLAS_OP_N, mk, b, NB, Real(-1), Pbuf, mk,
                 Wb, NB, Real(1), X + row0, (int)ldx);
        }
        if (forceFp32) CUBLAS_CHECK(cublasSetMathMode(cb0, saveMath));
        CUDA_CHECK(cudaStreamSynchronize(s0));
        if (green) cuCtxPopCurrent(nullptr);
    }
};

}  // namespace lrqr
