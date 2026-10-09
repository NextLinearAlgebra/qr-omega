#pragma once
// Register-resident Householder QR, adapted from gau.nernst's QR-v2 kernels for the GPU MODE
// leaderboard: a block or a warp owns a group of columns with all their rows in registers and
// publishes each reflector to the owners on its right. Two kernels share this scheme: the fused
// kernel (gx_body), in which the warps of one block hand reflectors over through shared memory and
// also form V^T V and T -- every GE/TS/TT node of a GPU's tree and the TT merge across GPUs -- and a
// chain of blocks that hand them over through global memory, for merge stacks taller than a block.
// Their arithmetic is LAPACK's: dlarfg reflectors, the norm downdates of xGEQPF, and T built by
// dlarft from the overlaps of the reflectors, never from A^T A.
#include "carrier.hpp"
#include "kernels.cuh"
#include <cooperative_groups.h>
#include <array>
#include <type_traits>
#include <vector>
namespace qr_omega {
template <class T> struct GfRange;
template <> struct GfRange<float> {
    static constexpr float lo = 0x1p-100f, hi = 0x1p100f, xmax = 0x1p50f;
};
template <> struct GfRange<double> {
    static constexpr double lo = 0x1p-900, hi = 0x1p900, xmax = 0x1p450;
};
__device__ __forceinline__ float gf_sqrt(float x) {
    return __fsqrt_rn(x);
}
__device__ __forceinline__ double gf_sqrt(double x) {
    return __dsqrt_rn(x);
}
__device__ __forceinline__ float gf_div(float a, float b) {
    return __fdiv_rn(a, b);
}
__device__ __forceinline__ double gf_div(double a, double b) {
    return __ddiv_rn(a, b);
}
__device__ __forceinline__ float gf_fma(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
__device__ __forceinline__ double gf_fma(double a, double b, double c) {
    return __fma_rn(a, b, c);
}

template <int N> struct GfLog {
    static constexpr int v = N <= 1 ? 0 : N <= 2 ? 1 : N <= 4 ? 2 : N <= 8 ? 3 : 4;
};
// Recursive halving: at the level with xor offset O each lane keeps half of its N live partials and
// receives the partner's matching half.
template <class T, int M, int N, int O>
__device__ __forceinline__ void gf_halve(T (&cur)[M], int lane) {
    if constexpr (N > 1) {
        constexpr int H = N / 2;
        const bool hi = (lane & O) != 0;
#pragma unroll
        for (int q = 0; q < H; ++q) {
            const T send = hi ? cur[q] : cur[q + H];
            const T keep = hi ? cur[q + H] : cur[q];
            cur[q] = keep + __shfl_xor_sync(0xffffffffu, send, O);
        }
        gf_halve<T, M, H, O / 2>(cur, lane);
    }
}
// Warp sum of N (1,2,4,8,16) partials per lane. Afterwards lane L holds the full sum of entry L >>
// (5 - log2 N); d is destroyed. 2N - 1 + (5 - log2 N) shuffles, fixed summation order.
template <class T, int N> __device__ __forceinline__ T gf_reduce_scatter(T (&d)[N], int lane) {
    gf_halve<T, N, N, 16>(d, lane);
    T s = d[0];
#pragma unroll
    for (int o = 16 >> GfLog<N>::v; o; o >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}
// Every lane receives all N sums.
template <class T, int N> __device__ __forceinline__ void gf_allreduce(T (&d)[N], int lane) {
    const T s = gf_reduce_scatter<T, N>(d, lane);
    if constexpr (N == 1)
        d[0] = s;
    else {
#pragma unroll
        for (int q = 0; q < N; ++q)
            d[q] = __shfl_sync(0xffffffffu, s, q << (5 - GfLog<N>::v));
    }
}

// Nonzero test on the bit pattern, ignoring the sign of -0.
__device__ __forceinline__ unsigned gf_nz(float y) {
    return __float_as_uint(y) << 1;
}
__device__ __forceinline__ unsigned gf_nz(double y) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(y) << 1;
    return unsigned(b >> 32) | unsigned(b);
}
// 1/sqrt(q) and 1/s for q, s well inside the normal range (the fast-path bounds): hardware estimate
// + one Newton step (fp32: <= 1 ulp more than correctly rounded; fp64: CUDA's rsqrt / IEEE
// reciprocal). LAPACK's own dlapy2 / dscal sequence is not correctly rounded either; the
// Householder backward error analysis only needs O(u) relative errors.
__device__ __forceinline__ float gf_rsqrt(float q) {
    const float y = rsqrtf(q), e = __fmaf_rn(-q * y, y, 1.0f);
    return __fmaf_rn(0.5f * y, e, y);
}
__device__ __forceinline__ double gf_rsqrt(double q) {
    return rsqrt(q);
}
__device__ __forceinline__ float gf_rcp(float s) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
    return __fmaf_rn(r, __fmaf_rn(-s, r, 1.0f), r);
}
__device__ __forceinline__ double gf_rcp(double s) {
    return __drcp_rn(s);
}

__device__ __forceinline__ void gau_store_release(int *p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int gau_load_acquire(const int *p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
template <class A> __device__ __forceinline__ A gau_warp_sum(A v) {
#pragma unroll
    for (int o = 16; o; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Block all-reduce of C partials per thread: recursive halving inside each warp, one hop through
// shared memory, a short shuffle tree over the warps, and a broadcast of the C sums to every lane.
// The summation order is fixed. part: [C][WARPS], double-buffered by the callers across barriers.
template <class T, int C, int WARPS>
__device__ __forceinline__ void gau_block_allreduce(T (&d)[C], T *part, int lane, int warp) {
    constexpr int SH = 5 - GfLog<C>::v, G = 32 / C; // lanes per column group in stage 2
    const T s = gf_reduce_scatter<T, C>(d, lane);
    if ((lane & ((1 << SH) - 1)) == 0)
        part[(lane >> SH) * WARPS + warp] = s;
    __syncthreads();
    const int q = lane % C;
    T t = T(0);
#pragma unroll
    for (int wv = lane / C; wv < WARPS; wv += G)
        t += part[q * WARPS + wv];
#pragma unroll
    for (int o = C; o < 32; o <<= 1)
        t += __shfl_xor_sync(0xffffffffu, t, o);
#pragma unroll
    for (int j = 0; j < C; ++j)
        d[j] = __shfl_sync(0xffffffffu, t, j);
}
// Owner chain: block o owns C consecutive columns with all their rows in registers. It first
// applies, in order, every reflector published by the owners to its left, then generates its own
// reflectors one by one and publishes each (its V column and tau) behind a release flag that the
// owners to its right acquire, so the panel advances as a wavefront without grid barriers. flags[k]
// == epoch once column k is published; the host bumps the epoch per launch, so the flags are never
// reset. A TT stack runs the same chain, and its structural zeros stay exact zeros.
template <class T, int C, int ITEMS, int THREADS>
__global__ void __launch_bounds__(THREADS, 1)
    gau_ge_kernel(T *A, int lda, const TilePacket *packet, T *V, int ldv, T *tau, int *flags,
                  int epoch, int *status) {
    constexpr int WARPS = THREADS / 32;
    // Keep the consumed reflector in registers only while it fits beside the owned columns;
    // otherwise the update re-reads it (an L1/L2 hit: the same CTA read it one reduction earlier).
    constexpr bool KEEP = ITEMS <= 16;
    const TilePacket pk = *packet;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, rows = pk.rows, h = pk.h,
              c0 = blockIdx.x * C;
    T *a = A + pk.row + size_t(pk.col) * lda;
    __shared__ T part[2][C * WARPS];
    __shared__ T tailt[2][WARPS];
    __shared__ int nzw[2][WARPS];
    __shared__ T diag[2];
    __shared__ double dred[WARPS];
    T col[ITEMS][C];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * THREADS + tid;
#pragma unroll
        for (int j = 0; j < C; ++j)
            col[it][j] = (r < rows && c0 + j < h) ? a[r + size_t(c0 + j) * lda] : T(0);
    }
    int par = 0;
    // ---- consume every reflector to the left, in order ----
    for (int k = 0; k < c0; ++k) {
        if (tid == 0) {
            while (gau_load_acquire(flags + k) != epoch)
                __nanosleep(20);
        }
        __syncthreads();
        const T tk = tau[k];
        const T *vk = V + size_t(k) * ldv;
        T d[C];
#pragma unroll
        for (int j = 0; j < C; ++j)
            d[j] = T(0);
        T vr[KEEP ? ITEMS : 1];
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * THREADS + tid;
            const T v = (r < rows && r >= k) ? vk[r] : T(0);
            if constexpr (KEEP)
                vr[it] = v;
#pragma unroll
            for (int j = 0; j < C; ++j)
                d[j] += v * col[it][j];
        }
        gau_block_allreduce<T, C, WARPS>(d, part[par], lane, warp);
#pragma unroll
        for (int j = 0; j < C; ++j)
            d[j] *= tk;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * THREADS + tid;
            T v;
            if constexpr (KEEP)
                v = vr[it];
            else
                v = (r < rows && r >= k) ? vk[r] : T(0);
#pragma unroll
            for (int j = 0; j < C; ++j)
                col[it][j] -= v * d[j];
        }
        par ^= 1;
    }
#pragma unroll
    for (int i = 0; i < C; ++i) {
        // ---- generate and publish this owner's reflectors ----
        const int c = c0 + i;
        if (c >= h)
            break;
        // dlarfg over the block: range-checked fast path in the working precision (IEEE-grade
        // rsqrt/rcp with one Newton step), LAPACK-scaled path otherwise (fp64-accumulated for fp32
        // storage, power-of-two scaled for fp64). Every quantity below is block-uniform, so the
        // branches are uniform.
        T tl = T(0);
        unsigned nz = 0;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * THREADS + tid;
            const T y = col[it][i];
            if (r > c && r < rows) {
                tl = gf_fma(y, y, tl);
                nz |= gf_nz(y);
            }
            if (r == c)
                diag[par] = y;
        }
        tl = gau_warp_sum(tl);
        {
            const bool wz = __any_sync(0xffffffffu, nz != 0u);
            if (lane == 0) {
                tailt[par][warp] = tl;
                nzw[par][warp] = wz;
            }
        }
        __syncthreads();
        const T tail = gau_warp_sum(lane < WARPS ? tailt[par][lane] : T(0));
        const bool has = __any_sync(0xffffffffu, lane < WARPS && nzw[par][lane] != 0);
        const T x0 = diag[par];
        T beta = x0, tk = T(0), inv = T(0);
        int e = 0;
        bool slow = false;
        double invd = 0;
        if (has) {
            if (tail >= GfRange<T>::lo && tail <= GfRange<T>::hi && fabs(x0) <= GfRange<T>::xmax) {
                const T q = gf_fma(x0, x0, tail), y = gf_rsqrt(q), nrm = q * y, sa = fabs(x0) + nrm,
                        rs = gf_rcp(sa);
                beta = x0 < T(0) ? nrm : -nrm;
                tk = sa * y;
                inv = x0 < T(0) ? -rs : rs;
            } else {
                slow = true;
                if constexpr (sizeof(T) == 8) {
                    double m = fabs(double(x0));
#pragma unroll
                    for (int it = 0; it < ITEMS; ++it) {
                        const int r = it * THREADS + tid;
                        if (r > c && r < rows)
                            m = fmax(m, fabs(double(col[it][i])));
                    }
#pragma unroll
                    for (int o = 16; o; o >>= 1)
                        m = fmax(m, __shfl_xor_sync(0xffffffffu, m, o));
                    if (lane == 0)
                        dred[warp] = m;
                    __syncthreads();
                    m = lane < WARPS ? dred[lane] : 0.0;
#pragma unroll
                    for (int o = 16; o; o >>= 1)
                        m = fmax(m, __shfl_xor_sync(0xffffffffu, m, o));
                    __syncthreads();
                    frexp(m, &e);
                }
                double t2 = 0;
#pragma unroll
                for (int it = 0; it < ITEMS; ++it) {
                    const int r = it * THREADS + tid;
                    if (r > c && r < rows) {
                        const double y = scalbn(double(col[it][i]), -e);
                        t2 = fma(y, y, t2);
                    }
                }
                t2 = gau_warp_sum(t2);
                if (lane == 0)
                    dred[warp] = t2;
                __syncthreads();
                t2 = gau_warp_sum(lane < WARPS ? dred[lane] : 0.0);
                __syncthreads();
                const double a = scalbn(double(x0), -e), nrm = sqrt(fma(a, a, t2)),
                             bs = a < 0 ? nrm : -nrm;
                beta = T(scalbn(bs, e));
                tk = T((bs - a) / bs);
                invd = 1.0 / (a - bs);
            }
        }
        if (tid == 0) {
            tau[c] = tk;
            if (!isfinite(beta))
                atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
        }
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * THREADS + tid;
            const T x = col[it][i];
            const T v = r == c ? T(1)
                        : (has && r > c && r < rows)
                            ? (slow ? T(scalbn(double(x), -e) * invd) : x * inv)
                            : T(0);
            if (r == c)
                col[it][i] = beta;
            else if (r > c)
                col[it][i] = has ? v : x;
            if (r < ldv)
                V[r + size_t(c) * ldv] = r < rows ? v : T(0);
        }
        for (int r = ITEMS * THREADS + tid; r < ldv; r += THREADS)
            V[r + size_t(c) * ldv] = T(0);
        // Every row writer of V column c and tau[c] precedes this barrier; the release below is
        // cumulative over it.
        T d[C];
#pragma unroll
        for (int j = 0; j < C; ++j)
            d[j] = T(0);
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * THREADS + tid;
            const T v = r == c ? T(1) : (r > c && r < rows) ? col[it][i] : T(0);
#pragma unroll
            for (int j = 0; j < C; ++j)
                if (j > i)
                    d[j] += v * col[it][j];
        }
        if (i + 1 < C) {
            const T s = gf_reduce_scatter<T, C>(d, lane);
            constexpr int SH = 5 - GfLog<C>::v;
            if ((lane & ((1 << SH) - 1)) == 0)
                part[par][(lane >> SH) * WARPS + warp] = s;
        }
        __syncthreads();
        if (tid == 0)
            gau_store_release(flags + c, epoch);
        if (i + 1 < C) {
            constexpr int G = 32 / C;
            const int q = lane % C;
            T t = T(0);
#pragma unroll
            for (int wv = lane / C; wv < WARPS; wv += G)
                t += part[par][q * WARPS + wv];
#pragma unroll
            for (int o = C; o < 32; o <<= 1)
                t += __shfl_xor_sync(0xffffffffu, t, o);
#pragma unroll
            for (int j = 0; j < C; ++j)
                d[j] = __shfl_sync(0xffffffffu, t, j) * tk;
#pragma unroll
            for (int it = 0; it < ITEMS; ++it) {
                const int r = it * THREADS + tid;
                const T v = r == c ? T(1) : (r > c && r < rows) ? col[it][i] : T(0);
#pragma unroll
                for (int j = 0; j < C; ++j)
                    if (j > i)
                        col[it][j] -= v * d[j];
            }
        }
        par ^= 1;
    }
    // ---- the owner commits its columns once: R above, beta on, v below the diagonal ----
    bool bad = false;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * THREADS + tid;
        if (r >= rows)
            continue;
#pragma unroll
        for (int j = 0; j < C; ++j)
            if (c0 + j < h) {
                const T y = col[it][j];
                bad |= !isfinite(y);
                a[r + size_t(c0 + j) * lda] = y;
            }
    }
    if (bad)
        atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
}

// T of Q = I - V T V^T from the overlaps O = V^T V and tau, as LAPACK dlarft forms it: the
// recurrence on 8-column blocks, then T12 = -T11 (O12 T22) level by level. H bounds h at compile
// time; columns h..H-1 carry tau = 0 and zero overlaps. Shared: S (ld H + 1) holds T in its upper
// triangle and O transposed in its lower one; M holds O12 T22 for every pair of blocks of the
// level.
template <class T, int H, int S>
__device__ __forceinline__ void gau_t_compose_level(T *Sm, T *M, int tid) {
    constexpr int LD = H + 1, P = H / (2 * S), OUT = P * S * S, NT = 512;
    constexpr int PER = OUT >= NT ? OUT / NT : 1;         // outputs per thread (power of two)
    constexpr int TC = PER >= 4 ? 4 : PER, TR = PER / TC; // TR x TC register tile (rows x columns)
    constexpr int TILES_R = S / TR, TILES_C = S / TC, TILES = P * TILES_R * TILES_C;
    // phase A: M = O12 T22  (T22 upper: q <= j)
    for (int t = tid; t < TILES; t += NT) {
        const int pr = t / (TILES_R * TILES_C), rem = t - pr * TILES_R * TILES_C,
                  i0 = (rem % TILES_R) * TR, j0 = (rem / TILES_R) * TC, a0 = 2 * S * pr,
                  a1 = a0 + S;
        T acc[TR][TC];
#pragma unroll
        for (int u = 0; u < TR; ++u)
#pragma unroll
            for (int v = 0; v < TC; ++v)
                acc[u][v] = T(0);
#pragma unroll 8
        for (int q = 0; q < S; ++q) {
            T x[TR], y[TC];
#pragma unroll
            for (int u = 0; u < TR; ++u)
                x[u] = Sm[(a1 + q) + (a0 + i0 + u) * LD]; // O(a0+i, a1+q), stored transposed
#pragma unroll
            for (int v = 0; v < TC; ++v)
                y[v] = q <= j0 + v ? Sm[(a1 + q) + (a1 + j0 + v) * LD] : T(0); // T(a1+q, a1+j)
#pragma unroll
            for (int u = 0; u < TR; ++u)
#pragma unroll
                for (int v = 0; v < TC; ++v)
                    acc[u][v] += x[u] * y[v];
        }
#pragma unroll
        for (int u = 0; u < TR; ++u)
#pragma unroll
            for (int v = 0; v < TC; ++v)
                M[pr * S * S + (i0 + u) + (j0 + v) * S] = acc[u][v];
    }
    __syncthreads();
    // phase B: T12 = -T11 M  (T11 upper: q >= i)
    for (int t = tid; t < TILES; t += NT) {
        const int pr = t / (TILES_R * TILES_C), rem = t - pr * TILES_R * TILES_C,
                  i0 = (rem % TILES_R) * TR, j0 = (rem / TILES_R) * TC, a0 = 2 * S * pr,
                  a1 = a0 + S;
        T acc[TR][TC];
#pragma unroll
        for (int u = 0; u < TR; ++u)
#pragma unroll
            for (int v = 0; v < TC; ++v)
                acc[u][v] = T(0);
#pragma unroll 8
        for (int q = 0; q < S; ++q) {
            T x[TR], y[TC];
#pragma unroll
            for (int u = 0; u < TR; ++u)
                x[u] = q >= i0 + u ? Sm[(a0 + i0 + u) + (a0 + q) * LD] : T(0); // T(a0+i, a0+q)
#pragma unroll
            for (int v = 0; v < TC; ++v)
                y[v] = M[pr * S * S + q + (j0 + v) * S];
#pragma unroll
            for (int u = 0; u < TR; ++u)
#pragma unroll
                for (int v = 0; v < TC; ++v)
                    acc[u][v] += x[u] * y[v];
        }
#pragma unroll
        for (int u = 0; u < TR; ++u)
#pragma unroll
            for (int v = 0; v < TC; ++v)
                Sm[(a0 + i0 + u) + (a1 + j0 + v) * LD] = -acc[u][v];
    }
    __syncthreads();
}
// S on entry: tau on the diagonal and the overlaps of the reflectors [base, base + h) transposed
// below it, zeros beyond h. One block of 512 threads.
template <class T, int H>
__device__ __forceinline__ void gau_load_overlaps(T *Sm, const T *G, int ldg, const T *tau,
                                                  int base, int h, int tid) {
    constexpr int LD = H + 1;
    for (int e = tid; e < H * H; e += 512) {
        const int i = e % H, j = e / H;
        const T g = (i < h && j < h) ? G[base + i + size_t(base + j) * ldg] : T(0);
        if (i < j)
            Sm[j + i * LD] = g;
        else if (i == j)
            Sm[i + i * LD] = i < h ? tau[base + i] : T(0);
    }
    __syncthreads();
}
// T in the upper triangle of S: dlarft inside each 8-column block (one warp per block, lanes 0..7
// own the rows), then the blocks joined level by level.
template <class T, int H> __device__ __forceinline__ void gau_build_t(T *Sm, T *M, int tid) {
    constexpr int LD = H + 1;
    const int lane = tid & 31, warp = tid >> 5;
    if (warp < H / 8) {
        const int b0 = warp * 8, r = b0 + lane;
#pragma unroll
        for (int c = 1; c < 8; ++c) {
            T v = T(0);
            if (lane < c) {
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (q >= lane && q < c)
                        v += Sm[r + (b0 + q) * LD] * Sm[(b0 + c) + (b0 + q) * LD];
                v *= -Sm[(b0 + c) + (b0 + c) * LD];
            }
            __syncwarp();
            if (lane < c)
                Sm[r + (b0 + c) * LD] = v;
            __syncwarp();
        }
    }
    __syncthreads();
    if constexpr (H > 8)
        gau_t_compose_level<T, H, 8>(Sm, M, tid);
    if constexpr (H > 16)
        gau_t_compose_level<T, H, 16>(Sm, M, tid);
    if constexpr (H > 32)
        gau_t_compose_level<T, H, 32>(Sm, M, tid);
    if constexpr (H > 64)
        gau_t_compose_level<T, H, 64>(Sm, M, tid);
}
template <class T, int H>
__global__ __launch_bounds__(512) void gau_build_t_h(const T *G, int ldg, const T *tau,
                                                     const TilePacket *packet, T *Ts, int b) {
    extern __shared__ __align__(16) unsigned char gau_smem[];
    T *Sm = reinterpret_cast<T *>(gau_smem), *M = Sm + H * (H + 1);
    const TilePacket pk = *packet;
    gau_load_overlaps<T, H>(Sm, G, ldg, tau, 0, pk.h, threadIdx.x);
    gau_build_t<T, H>(Sm, M, threadIdx.x);
    T *tri = Ts + size_t(pk.tile) * b * b;
    for (int e = threadIdx.x; e < b * b; e += 512) {
        const int i = e % b, j = e / b;
        tri[e] = (i < pk.h && j < pk.h && i <= j) ? Sm[i + j * (H + 1)] : T(0);
    }
}
template <class T, int H> inline size_t gau_build_t_h_smem() {
    return (size_t(H) * (H + 1) + size_t(H) * H / 4 + 8) * sizeof(T);
}
template <class T, int H>
void launch_build_t_h(const T *G, int ldg, const T *tau, const TilePacket *pk, T *Ts, int b,
                      cudaStream_t st) {
    reserve_shared_memory(gau_build_t_h<T, H>, gau_build_t_h_smem<T, H>());
    gau_build_t_h<T, H><<<1, 512, gau_build_t_h_smem<T, H>(), st>>>(G, ldg, tau, pk, Ts, b);
    CU(cudaGetLastError());
}
// T of the h <= 128 reflectors of a packet from their overlaps G = V^T V and tau, into its tile of
// Ts.
template <class T>
void launch_build_t(const T *G, int ldg, const T *tau, const TilePacket *pk, int h, T *Ts, int b,
                    cudaStream_t st) {
    if (h <= 32)
        launch_build_t_h<T, 32>(G, ldg, tau, pk, Ts, b, st);
    else if (h <= 64)
        launch_build_t_h<T, 64>(G, ldg, tau, pk, Ts, b, st);
    else
        launch_build_t_h<T, 128>(G, ldg, tau, pk, Ts, b, st);
}

// Operands of a register-resident panel: the rows x h packet of A and where its reflectors go.
template <class T> struct RegisterPanel {
    T *a;
    int lda;
    const TilePacket *packet;
    int rows, h;
    T *v;
    int ldv;
    T *tau;
    int *status;
    Carrier *carrier = nullptr; // receives the carrier of the launch
};
// Launches the chain in which every block owns C columns and each of its THREADS threads holds
// ITEMS rows of them. False when the owners are not all resident or the kernel spills registers.
template <class T, int C, int ITEMS, int THREADS>
bool launch_chain(const RegisterPanel<T> &p, int *flags, int epoch, cudaStream_t st) {
    auto kernel = gau_ge_kernel<T, C, ITEMS, THREADS>;
    static int resident = -1;
    static bool usable = false;
    if (resident < 0) {
        int per_sm = 0;
        CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, 0));
        cudaFuncAttributes attr;
        CU(cudaFuncGetAttributes(&attr, kernel));
        usable = device_properties().cooperativeLaunch && attr.localSizeBytes <= 128;
        resident = per_sm * device_properties().multiProcessorCount;
    }
    int owners = (p.h + C - 1) / C;
    if (!usable || owners > resident)
        return false;
    RegisterPanel<T> q = p;
    void *args[] = {&q.a, &q.lda, &q.packet, &q.v, &q.ldv, &q.tau, &flags, &epoch, &q.status};
    CU(cudaLaunchCooperativeKernel((const void *)kernel, dim3(owners), dim3(THREADS), args, 0, st));
    if (p.carrier)
        *p.carrier = Carrier{}.set(GpuLevel, 1, owners, 1).set(BlockLevel, 1, 1, THREADS / 32);
    return true;
}
__device__ __forceinline__ unsigned gau_smem_addr(const void *p) {
    return unsigned(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void gau_mbar_init(unsigned a, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void gau_mbar_arrive(unsigned a) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ bool gau_mbar_test(unsigned a, int phase) {
    unsigned ready;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t}"
                 : "=r"(ready)
                 : "r"(a), "r"(phase)
                 : "memory");
    return ready != 0;
}
__device__ __forceinline__ void gau_mbar_wait(unsigned a, int phase) {
    asm volatile("{\n\t.reg .pred ready;\n\tGAU_WAIT_%=:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1, %2;\n\t"
                 "@!ready bra.uni GAU_WAIT_%=;\n\t}" ::"r"(a),
                 "r"(phase), "r"(0x989680)
                 : "memory");
}

// dlarfg on column i of a warp's registers (row r = it*32 + lane), pivot row c. On return the
// column holds beta at the pivot and v (unit pivot implied) below it; returns tau (0 and the column
// untouched when the tail is zero: H = I). Fast path (the unscaled sum of squares q is a normal
// number well inside the range, so it equals the scaled one to working accuracy): with y =
// 1/sqrt(q), s = |alpha| + ||x|| (no cancellation): beta = -sign(alpha) ||x||, tau = (beta -
// alpha)/beta = s y, 1/(alpha - beta) = sign(alpha)/s. Otherwise the fp64-accumulated (fp32
// storage) or power-of-two scaled (fp64 storage) LAPACK path. Every branch is warp-uniform.
template <class T, int ITEMS, int VEC>
__device__ __forceinline__ T gf_reflector(T (&col)[ITEMS][VEC], int i, int c, int rows, int lane,
                                          int *status) {
    T tl = T(0), xc = T(0);
    unsigned nz = 0;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * 32 + lane;
        const T y = col[it][i];
        if (r > c && r < rows) {
            tl = gf_fma(y, y, tl);
            nz |= gf_nz(y);
        }
        if (r == c)
            xc = y;
    }
    const bool has = __any_sync(0xffffffffu, nz != 0u);
    tl = gau_warp_sum(tl);
    const T x0 = __shfl_sync(0xffffffffu, xc, c & 31);
    if (!has)
        return T(0);
    T beta, tau;
    if (tl >= GfRange<T>::lo && tl <= GfRange<T>::hi && fabs(x0) <= GfRange<T>::xmax) {
        const T q = gf_fma(x0, x0, tl), y = gf_rsqrt(q), nrm = q * y, sa = fabs(x0) + nrm,
                rs = gf_rcp(sa);
        beta = x0 < T(0) ? nrm : -nrm;
        tau = sa * y;
        const T inv = x0 < T(0) ? -rs : rs;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r == c)
                col[it][i] = beta;
            else if (r > c && r < rows)
                col[it][i] *= inv;
        }
    } else if constexpr (sizeof(T) == 4) {
        // FP32 storage: an FP32 square can neither overflow nor underflow in FP64, so the FP64 sum
        // needs no scaling.
        double t2 = 0.0;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r > c && r < rows) {
                const double y = double(col[it][i]);
                t2 = fma(y, y, t2);
            }
        }
        t2 = gau_warp_sum(t2);
        const double a = double(x0), nrm = sqrt(fma(a, a, t2)), b = a < 0 ? nrm : -nrm,
                     inv = 1.0 / (a - b);
        beta = T(b);
        tau = T((b - a) / b);
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r == c)
                col[it][i] = beta;
            else if (r > c && r < rows)
                col[it][i] = T(double(col[it][i]) * inv);
        }
    } else {
        // FP64 storage: scale exactly by the power of two of the largest magnitude, which keeps
        // the sum in range as LAPACK's dnrm2 and dlapy2 do.
        double m = fabs(double(x0));
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r > c && r < rows)
                m = fmax(m, fabs(double(col[it][i])));
        }
#pragma unroll
        for (int o = 16; o; o >>= 1)
            m = fmax(m, __shfl_xor_sync(0xffffffffu, m, o));
        int e;
        frexp(m, &e);
        double t2 = 0.0;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r > c && r < rows) {
                const double y = scalbn(double(col[it][i]), -e);
                t2 = fma(y, y, t2);
            }
        }
        t2 = gau_warp_sum(t2);
        const double a = scalbn(double(x0), -e), nrm = sqrt(fma(a, a, t2)), bs = a < 0 ? nrm : -nrm,
                     inv = 1.0 / (a - bs);
        beta = T(scalbn(bs, e));
        tau = T((bs - a) / bs);
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r == c)
                col[it][i] = beta;
            else if (r > c && r < rows)
                col[it][i] = T(scalbn(double(col[it][i]), -e) * inv);
        }
    }
    if (lane == 0 && !isfinite(beta))
        atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
    return tau;
}
// Stride of a reflector slot in shared memory: 16 bytes past the rows keep bulk alignment and move
// consecutive slots to different banks.
template <class T> __host__ __device__ constexpr int gf_slot(int rows) {
    return rows + 16 / int(sizeof(T));
}
// ---- the fused kernel ---------------------------------------------------------------------------
// One block factors a node: warp w owns columns [w VEC, (w + 1) VEC) with all rows in registers
// (row it 32 + lane) and publishes each reflector to the warps on its right through shared memory. With u = x_c - beta e_c the unnormalized
// Householder vector, H_c = I - tau_u u u^T, tau_u = 1 / (|beta| (|alpha| + |beta|)). The dots of u
// with the owner's later columns are G_q - beta x_q(c), G_q = sum_{r >= c} x_c(r) x_q(r), so their
// warp reduction starts with the step and is interleaved, level by level, with the reflector's
// square root and reciprocals. The next column's pivot and tail square sum follow by the norm
// downdate of LAPACK's xGEQPF, recomputed directly when it cancels. Consumers apply (u, tau_u);
// V is normalized to a unit diagonal, with LAPACK's tau, before it is stored or overlapped for T.
// Pivots are columns c < h, so only the first ceil(h / 32) items of a lane hold pivot rows: PIV.
template <class T> __device__ __forceinline__ T gx_rsqrt_approx(T q);
template <> __device__ __forceinline__ double gx_rsqrt_approx<double>(double q) {
    double y;
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(q));
    return y;
}
template <> __device__ __forceinline__ float gx_rsqrt_approx<float>(float q) {
    return rsqrtf(q);
}
template <class T> __device__ __forceinline__ T gx_rcp_approx(T s);
template <> __device__ __forceinline__ double gx_rcp_approx<double>(double s) {
    double r;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(s));
    return r;
}
template <> __device__ __forceinline__ float gx_rcp_approx<float>(float s) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
    return r;
}
// The downdated square sum is kept when it retains at least this fraction of the square sum it
// came from, so that its relative error stays within a few units of roundoff.
template <class T> constexpr T gx_keep = T(0.25);
// Warp all-reduce of any N <= 16 values: gf_allreduce on the next power of two.
template <class T, int N> __device__ __forceinline__ void gx_allreduce(T (&d)[N], int lane) {
    constexpr int P = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : 16;
    if constexpr (P == N)
        gf_allreduce<T, N>(d, lane);
    else {
        T x[P];
#pragma unroll
        for (int q = 0; q < P; ++q)
            x[q] = q < N ? d[q] : T(0);
        gf_allreduce<T, P>(x, lane);
#pragma unroll
        for (int q = 0; q < N; ++q)
            d[q] = x[q];
    }
}
template <class T> struct GxReflector {
    T beta, tau_u, tau, inv;
};
// A butterfly all-reduce of N values (bitwise identical on every lane) and B broadcasts from the
// lanes src, interleaved level by level with the fast-path reflector arithmetic of (alpha, s2).
// Newton steps follow the hardware estimates: two in FP64, one in FP32.
// Broadcasts 0 .. B - 3 come from lane `src0`, the last two from lane `src1`.
template <class T, int N, int B>
__device__ __forceinline__ GxReflector<T> gx_reduce_reflect(T (&d)[N > 0 ? N : 1],
                                                            T (&bv)[B > 0 ? B : 1], int src0,
                                                            int src1, T alpha, T s2) {
    auto level = [&](int o) {
        T t[N > 0 ? N : 1];
#pragma unroll
        for (int q = 0; q < N; ++q)
            t[q] = __shfl_xor_sync(0xffffffffu, d[q], o);
#pragma unroll
        for (int q = 0; q < N; ++q)
            d[q] += t[q];
    };
    auto broadcast = [&](auto b) {
        constexpr int i = decltype(b)::value;
        if constexpr (i < B)
            bv[i] = __shfl_sync(0xffffffffu, bv[i], i + 2 < B ? src0 : src1);
    };
    constexpr bool wide = sizeof(T) == 8;
    const T q = gf_fma(alpha, alpha, s2);
    broadcast(std::integral_constant<int, 0>{});
    level(16);
    T y = gx_rsqrt_approx<T>(q);
    broadcast(std::integral_constant<int, 1>{});
    level(8);
    const T half = T(0.5) * q;
    y = y * gf_fma(-half * y, y, T(1.5));
    broadcast(std::integral_constant<int, 2>{});
    level(4);
    if constexpr (wide)
        y = y * gf_fma(-half * y, y, T(1.5));
    const T nrm = q * y, sa = fabs(alpha) + nrm, ns = nrm * sa;
    broadcast(std::integral_constant<int, 3>{});
    level(2);
    T r = gx_rcp_approx<T>(sa), t = gx_rcp_approx<T>(ns);
    r = gf_fma(r, gf_fma(-sa, r, T(1)), r);
    t = gf_fma(t, gf_fma(-ns, t, T(1)), t);
    broadcast(std::integral_constant<int, 4>{});
    level(1);
    if constexpr (wide) {
        r = gf_fma(r, gf_fma(-sa, r, T(1)), r);
        t = gf_fma(t, gf_fma(-ns, t, T(1)), t);
    }
    broadcast(std::integral_constant<int, 5>{});
    broadcast(std::integral_constant<int, 6>{});
    broadcast(std::integral_constant<int, 7>{});
    broadcast(std::integral_constant<int, 8>{});
    static_assert(B <= 9, "at most nine broadcasts");
    return {alpha < T(0) ? nrm : -nrm, t, sa * y, alpha < T(0) ? -r : r};
}
// a when p, else b. Opaque to the compiler, which would otherwise fold a chain of such selections
// over the items of a lane into an indexed load and demote the register array to local memory.
__device__ __forceinline__ double gx_select(bool p, double a, double b) {
    double r;
    asm("{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %3, 0;\n\tselp.f64 %0, %1, %2, q;\n\t}"
        : "=d"(r)
        : "d"(a), "d"(b), "r"(unsigned(p)));
    return r;
}
__device__ __forceinline__ float gx_select(bool p, float a, float b) {
    float r;
    asm("{\n\t.reg .pred q;\n\tsetp.ne.u32 q, %3, 0;\n\tselp.f32 %0, %1, %2, q;\n\t}"
        : "=f"(r)
        : "f"(a), "f"(b), "r"(unsigned(p)));
    return r;
}
// Entry of column q at row `row` held by this lane, when the lane owns that row (row < 32 PIV).
template <class T, int ITEMS, int VEC, int PIV>
__device__ __forceinline__ T gx_row(const T (&x)[ITEMS][VEC], int q, int row) {
    T value = x[0][q];
#pragma unroll
    for (int it = 1; it < PIV; ++it)
        value = gx_select((row >> 5) == it, x[it][q], value);
    return value;
}
// The warp's direct pivot (row c) and tail square sum (rows > c) of column q.
template <class T, int ITEMS, int VEC, int PIV>
__device__ __forceinline__ void gx_direct(const T (&x)[ITEMS][VEC], int q, int c, int lane,
                                          T &alpha, T &s2) {
    T s = T(0);
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const T y = x[it][q];
        s = it >= PIV || it * 32 + lane > c ? gf_fma(y, y, s) : s;
    }
    alpha = __shfl_sync(0xffffffffu, gx_row<T, ITEMS, VEC, PIV>(x, q, c), c & 31);
    s2 = gau_warp_sum(s);
}
// Pivot state of the column whose reflector forms next: its entry at the pivot row and the square
// sum below it; `fresh` when they must be computed directly.
template <class T> struct GxPivot {
    T alpha = T(0), s2 = T(0);
    bool fresh = true;
};
// The downdated pivot state of column `next` after the reflector with pivot k: S is its pre-update
// square sum from row k on, xk and xk1 its pre-update rows k and k + 1, f the multiplier of u in its
// update, uk and uk1 the reflector's rows k and k + 1.
template <class T>
__device__ __forceinline__ void gx_downdate(GxPivot<T> &p, T S, T xk, T xk1, T f, T uk, T uk1) {
    const T rk = gf_fma(f, uk, xk); // an entry of R
    p.alpha = gf_fma(f, uk1, xk1);
    p.s2 = S - gf_fma(rk, rk, p.alpha * p.alpha);
    p.fresh = !(p.s2 >= gx_keep<T> * S);
}

// Applies the published reflector k (u in the shared slot, tau_u) to the warp's columns. LAST: the
// warp's column 0 forms the next reflector, so its pivot state is downdated.
template <class T, int ITEMS, int VEC, int PIV, bool LAST>
__device__ __forceinline__ void gx_consume(T (&x)[ITEMS][VEC], const T *slot, T tu, int k, int lane,
                                           GxPivot<T> &p) {
    T u[ITEMS];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        // T construction reuses rows above the pivot concurrently. A volatile conditional load
        // prevents speculative reads of that storage even when their values would be discarded.
        u[it] = it >= PIV || it * 32 + lane >= k
                    ? static_cast<const volatile T *>(slot)[it * 32 + lane]
                    : T(0);
    }
    constexpr int N = VEC + (LAST ? 1 : 0);
    T d[N];
#pragma unroll
    for (int q = 0; q < N; ++q)
        d[q] = T(0);
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
#pragma unroll
        for (int q = 0; q < VEC; ++q)
            d[q] = gf_fma(u[it], x[it][q], d[q]);
        if constexpr (LAST) {
            const T y = x[it][0];
            d[VEC] = it >= PIV || it * 32 + lane >= k ? gf_fma(y, y, d[VEC]) : d[VEC];
        }
    }
    T xk = T(0), xk1 = T(0), uk = T(0), uk1 = T(0);
    if constexpr (LAST) {
        xk = __shfl_sync(0xffffffffu, gx_row<T, ITEMS, VEC, PIV>(x, 0, k), k & 31);
        xk1 = __shfl_sync(0xffffffffu, gx_row<T, ITEMS, VEC, PIV>(x, 0, k + 1), (k + 1) & 31);
        T uu[ITEMS][1];
#pragma unroll
        for (int it = 0; it < PIV; ++it)
            uu[it][0] = u[it];
        uk = __shfl_sync(0xffffffffu, gx_row<T, ITEMS, 1, PIV>(uu, 0, k), k & 31);
        uk1 = __shfl_sync(0xffffffffu, gx_row<T, ITEMS, 1, PIV>(uu, 0, k + 1), (k + 1) & 31);
    }
    gx_allreduce<T, N>(d, lane);
#pragma unroll
    for (int q = 0; q < VEC; ++q) {
        const T f = -(d[q] * tu);
#pragma unroll
        for (int it = 0; it < ITEMS; ++it)
            x[it][q] = gf_fma(u[it], f, x[it][q]);
    }
    if constexpr (LAST)
        gx_downdate(p, d[VEC], xk, xk1, -(d[0] * tu), uk, uk1);
}

// Applies the VEC reflectors k0 .. k0 + VEC - 1 of a finished block to the warp's columns at once
// with the block's diagonal T: x -= V T^T V^T x, v_k = u_k invs[k] (forward dlarft order). A warp
// whose columns arrive after these reflectors were published catches up a block per warp reduction
// instead of a reduction per reflector.
template <class T, int ITEMS, int VEC, int PIV>
__device__ __forceinline__ void gx_consume_block(T (&x)[ITEMS][VEC], const T *slots, const T *invs,
                                                 const T *Tp, int k0, int lane) {
    constexpr int SL = gf_slot<T>(ITEMS * 32);
    T d[VEC * VEC];
#pragma unroll
    for (int e = 0; e < VEC * VEC; ++e)
        d[e] = T(0);
#pragma unroll
    for (int it = 0; it < ITEMS; ++it)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const T u =
                it >= PIV || it * 32 + lane >= k0 + i
                    ? static_cast<const volatile T *>(slots)[size_t(i) * SL + it * 32 + lane]
                    : T(0);
#pragma unroll
            for (int q = 0; q < VEC; ++q)
                d[i * VEC + q] = gf_fma(u, x[it][q], d[i * VEC + q]);
        }
    gx_allreduce<T, VEC * VEC>(d, lane);
    // diag(invs) T^T diag(invs) (U^T x), in place from the last reflector down
#pragma unroll
    for (int i = VEC - 1; i >= 0; --i) {
        const T *ti = Tp + size_t(k0 + i) * (k0 + i + 1) / 2 + k0; // T(k0 .., k0 + i)
#pragma unroll
        for (int q = 0; q < VEC; ++q) {
            T acc = T(0);
#pragma unroll
            for (int j = 0; j <= i; ++j)
                acc = gf_fma(ti[j] * invs[j], d[j * VEC + q], acc);
            d[i * VEC + q] = invs[i] * acc;
        }
    }
#pragma unroll
    for (int it = 0; it < ITEMS; ++it)
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
            const T u =
                it >= PIV || it * 32 + lane >= k0 + i
                    ? static_cast<const volatile T *>(slots)[size_t(i) * SL + it * 32 + lane]
                    : T(0);
#pragma unroll
            for (int q = 0; q < VEC; ++q)
                x[it][q] = gf_fma(-u, d[i * VEC + q], x[it][q]);
        }
}

// Shared memory of gx_body: the reflector slots (u_c on rows >= c; the overlaps O(k, c) of the T
// construction on rows k < c), tau_u, 1 / u_c(c) and LAPACK's tau per column, mbarriers (reflector
// published, overlaps of a block complete, diagonal T block ready), packed T and per-warp scratch.
template <class T> inline size_t gx_smem(int items, int vec, int warps) {
    const size_t cols = size_t(warps) * vec, sl = size_t(gf_slot<T>(items * 32));
    return (cols * sl + 3 * cols + 2) * sizeof(T) + (cols + 2 * size_t(warps)) * 8 +
           (cols * (cols + 1) / 2 + size_t(warps) * vec * vec) * sizeof(T) + 64;
}

// The owner's column step I (pivot c0 + I) and the steps after it.
template <class T, int ITEMS, int VEC, int PIV, int I>
__device__ __forceinline__ void gx_steps(T (&x)[ITEMS][VEC], int c0, int h, int rows, int lane,
                                         T *refl, T *tauu, T *invs, T *taus, unsigned long long *mb,
                                         T *tau_out, int *status, GxPivot<T> &p) {
    if constexpr (I < VEC) {
        constexpr int SL = gf_slot<T>(ITEMS * 32), L = VEC - 1 - I; // L: the warp's later columns
        const int c = c0 + I;
        if (c >= h)
            return;
        if (p.fresh)
            gx_direct<T, ITEMS, VEC, PIV>(x, I, c, lane, p.alpha, p.s2);
        // G_q (rows >= c) for the later columns and the next column's square sum S (rows >= c),
        // and their rows c (later columns) and c + 1 (this and the next column) from their lanes
        T g[L > 0 ? L + 1 : 1], bc[L > 0 ? L + 2 : 1];
#pragma unroll
        for (int q = 0; q <= L; ++q)
            g[q] = T(0);
        if constexpr (L > 0) {
#pragma unroll
            for (int it = 0; it < ITEMS; ++it) {
                const bool on = it >= PIV || it * 32 + lane >= c;
                const T uv = on ? x[it][I] : T(0), y = on ? x[it][I + 1] : T(0);
#pragma unroll
                for (int q = 0; q < L; ++q)
                    g[q] = gf_fma(uv, x[it][I + 1 + q], g[q]);
                g[L] = gf_fma(y, y, g[L]);
            }
#pragma unroll
            for (int q = 0; q < L; ++q)
                bc[q] = gx_row<T, ITEMS, VEC, PIV>(x, I + 1 + q, c);
            bc[L] = gx_row<T, ITEMS, VEC, PIV>(x, I, c + 1);
            bc[L + 1] = gx_row<T, ITEMS, VEC, PIV>(x, I + 1, c + 1);
        }
        const bool safe =
            p.s2 >= GfRange<T>::lo && p.s2 <= GfRange<T>::hi && fabs(p.alpha) <= GfRange<T>::xmax;
        GxReflector<T> rf;
        if constexpr (L > 0)
            rf = gx_reduce_reflect<T, L + 1, L + 2>(g, bc, c & 31, (c + 1) & 31, p.alpha, p.s2);
        else
            rf = gx_reduce_reflect<T, 0, 0>(g, bc, 0, 0, p.alpha, p.s2);
        T uc = p.alpha - rf.beta, shift = rf.beta;
        if (!safe) {
            // The scaled LAPACK path normalizes the column in place (beta at row c, v below):
            // then u = v, u(c) = 1, tau_u = tau, and the dots are formed again with u.
            rf.tau = rf.tau_u = gf_reflector<T, ITEMS, VEC>(x, I, c, rows, lane, status);
            rf.inv = uc = T(1);
            shift = T(0);
            if constexpr (L > 0) {
#pragma unroll
                for (int q = 0; q <= L; ++q)
                    g[q] = T(0);
#pragma unroll
                for (int it = 0; it < ITEMS; ++it) {
                    const int r = it * 32 + lane;
                    const T uv = it >= PIV || r > c ? x[it][I] : r == c ? T(1) : T(0);
                    const T y = it >= PIV || r >= c ? x[it][I + 1] : T(0);
#pragma unroll
                    for (int q = 0; q < L; ++q)
                        g[q] = gf_fma(uv, x[it][I + 1 + q], g[q]);
                    g[L] = gf_fma(y, y, g[L]);
                }
                gx_allreduce<T, L + 1>(g, lane);
                bc[L] =
                    __shfl_sync(0xffffffffu, gx_row<T, ITEMS, VEC, PIV>(x, I, c + 1), (c + 1) & 31);
            }
        }
        // publish u (zero above the pivot, uc at it) and the reflector's scalars
        T *slot = refl + size_t(c) * SL + lane;
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            slot[it * 32] = it >= PIV || r > c ? x[it][I] : r == c ? uc : T(0);
        }
        if (lane == 0) {
            tauu[c] = rf.tau_u;
            invs[c] = rf.inv;
            taus[c] = rf.tau;
            if (tau_out)
                tau_out[c] = rf.tau;
        }
        gau_mbar_arrive(gau_smem_addr(mb + c));
        if constexpr (L > 0) {
            // later columns: d_q = G_q - beta x_q(c), x_q += f_q u
#pragma unroll
            for (int q = 0; q < L; ++q) {
                const T f = -((g[q] - shift * bc[q]) * rf.tau_u);
#pragma unroll
                for (int it = 0; it < ITEMS; ++it) {
                    const int r = it * 32 + lane;
                    const T uv = it >= PIV || r > c ? x[it][I] : r == c ? uc : T(0);
                    x[it][I + 1 + q] = gf_fma(uv, f, x[it][I + 1 + q]);
                }
            }
            gx_downdate(p, g[L], bc[0], bc[L + 1], -((g[0] - shift * bc[0]) * rf.tau_u), uc, bc[L]);
        }
        // the finished column: beta at the pivot, v = u / u(c) below
        if (safe) {
#pragma unroll
            for (int it = 0; it < ITEMS; ++it) {
                const int r = it * 32 + lane;
                x[it][I] = it >= PIV || r > c ? x[it][I] * rf.inv : r == c ? rf.beta : x[it][I];
            }
        }
        gx_steps<T, ITEMS, VEC, PIV, I + 1>(x, c0, h, rows, lane, refl, tauu, invs, taus, mb,
                                            tau_out, status, p);
    }
}

template <class T, int ITEMS, int VEC, int PIV, class Source>
__device__ __forceinline__ void gx_body(const Source &source, int rows, int h, T *vo, int ldv,
                                        int vrows, T *tau_out, T *tri, int b, int *status,
                                        unsigned char *smem) {
    constexpr int ROWS = ITEMS * 32, SL = gf_slot<T>(ROWS);
    static_assert(PIV >= 1 && PIV <= ITEMS, "pivot rows lie in the first PIV items");
    const int nt = blockDim.x, WARPS = nt >> 5, COLS = WARPS * VEC;
    T *refl = reinterpret_cast<T *>(smem);
    T *tauu = refl + size_t(COLS) * SL, *invs = tauu + COLS, *taus = invs + COLS;
    unsigned long long *mb =
        reinterpret_cast<unsigned long long *>(taus + COLS + ((3 * COLS * sizeof(T)) % 8 ? 1 : 0));
    // oc: overlaps of a block complete; td: its diagonal T block
    unsigned long long *oc = mb + COLS, *td = oc + WARPS;
    T *Tp = reinterpret_cast<T *>(td + WARPS);   // T(i,j), i <= j, at j(j+1)/2 + i
    T *ysc = Tp + size_t(COLS) * (COLS + 1) / 2; // [WARPS][VEC][VEC]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c0 = warp * VEC,
              nb = (h + VEC - 1) / VEC;
    if (tid == 0) {
        for (int i = 0; i < COLS; ++i)
            gau_mbar_init(gau_smem_addr(mb + i), 32);
        for (int bb = 0; bb < WARPS; ++bb) {
            gau_mbar_init(gau_smem_addr(oc + bb), (bb + 1) * 32);
            gau_mbar_init(gau_smem_addr(td + bb), 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    source.acquire(warp, lane);
    // Every row of a lane lives in one piece of the source: its column 0 and its first stored column.
    T col[ITEMS][VEC];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        int first;
        const T *rp = source.row(it * 32 + lane, rows, first);
#pragma unroll
        for (int q = 0; q < VEC; ++q) {
            const int c = c0 + q;
            col[it][q] = rp && c < h && c >= first ? rp[size_t(c) * source.lda] : T(0);
        }
    }
    GxPivot<T> pivot;
    // The reflectors of the warps to the left, in order: whole blocks while their diagonal T is
    // already formed, then one at a time as they are published. The last one is applied alone, so
    // that it also downdates the pivot of the warp's first column.
    const int kend = min(c0, h);
    int k = 0;
    if constexpr (Source::late)
        for (; k + VEC < kend && gau_mbar_test(gau_smem_addr(td + k / VEC), 0); k += VEC)
            gx_consume_block<T, ITEMS, VEC, PIV>(col, refl + size_t(k) * SL, invs + k, Tp, k, lane);
    for (; k < kend; ++k) {
        gau_mbar_wait(gau_smem_addr(mb + k), 0);
        if (k + 1 < kend)
            gx_consume<T, ITEMS, VEC, PIV, false>(col, refl + size_t(k) * SL, tauu[k], k, lane,
                                                  pivot);
        else
            gx_consume<T, ITEMS, VEC, PIV, true>(col, refl + size_t(k) * SL, tauu[k], k, lane,
                                                 pivot);
    }
    gx_steps<T, ITEMS, VEC, PIV, 0>(col, c0, h, rows, lane, refl, tauu, invs, taus, mb, tau_out,
                                    status, pivot);
    // commit: R above, beta on, v below the diagonal (the owner writes its columns once), and the
    // packed V with its unit diagonal. The rows that can hold R (the first PIV items) go first and
    // are published at once: a consumer of this node needs only its triangle.
    bool bad = false;
    auto commit = [&](auto first, auto last) {
#pragma unroll
        for (int it = decltype(first)::value; it < decltype(last)::value; ++it) {
            int from;
            T *rp = source.row(it * 32 + lane, rows, from);
#pragma unroll
            for (int q = 0; q < VEC; ++q) {
                const int c = c0 + q;
                if (rp && c < h && c >= from) {
                    const T y = col[it][q];
                    bad |= !isfinite(y);
                    rp[size_t(c) * source.lda] = y;
                }
            }
        }
    };
    commit(std::integral_constant<int, 0>{}, std::integral_constant<int, PIV>{});
    source.publish(warp, lane);
    commit(std::integral_constant<int, PIV>{}, std::integral_constant<int, ITEMS>{});
    if (bad)
        atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * 32 + lane;
#pragma unroll
        for (int q = 0; q < VEC; ++q) {
            const int c = c0 + q;
            col[it][q] = (c < h && r >= c) ? (r == c ? T(1) : col[it][q]) : T(0);
            if (c < h && r < vrows)
                vo[r + size_t(c) * ldv] = col[it][q];
        }
    }
    for (int q = 0; q < VEC; ++q)
        if (c0 + q < h)
            for (int r = ROWS + lane; r < vrows; r += 32)
                vo[r + size_t(c0 + q) * ldv] = T(0);
    // T by dlarft's left-looking blocked recurrence while the chain runs: once the overlaps of block
    // bb are complete, its owner forms the diagonal block T22 and every earlier warp w forms its rows
    // of T12 = -T11 O12 T22 as -Y T22 with Y = T(rows_w, :) O(:, block). The slots hold u, scaled to
    // v by invs on reading.
    constexpr int SH = 5 - GfLog<VEC>::v;
    constexpr int NP = VEC * VEC, PPL = NP >= 32 ? NP / 32 : 1, LPP = NP >= 32 ? 1 : 32 / NP;
    T *ys = ysc + size_t(warp) * NP;
    if (c0 < h)
        for (int bb = warp; bb < nb; ++bb) {
            const int jb = bb * VEC, j0 = max(jb, c0 + 1), j1 = min(jb + VEC, h);
            for (int j = j0; j < j1; ++j) {
                if (bb > warp)
                    gau_mbar_wait(gau_smem_addr(mb + j), 0);
                const T sj = invs[j];
                T d[VEC];
#pragma unroll
                for (int q = 0; q < VEC; ++q)
                    d[q] = T(0);
#pragma unroll
                for (int it = 0; it < ITEMS; ++it) {
                    const int r = it * 32 + lane;
                    if (it >= PIV || r >= j) {
                        const T y = r == j ? T(1) : refl[size_t(j) * SL + r] * sj;
#pragma unroll
                        for (int q = 0; q < VEC; ++q)
                            d[q] = gf_fma(col[it][q], y, d[q]);
                    }
                }
                const T s = gf_reduce_scatter<T, VEC>(d, lane);
                const int q = lane >> SH;
                if ((lane & ((1 << SH) - 1)) == 0 && c0 + q < j)
                    refl[size_t(j) * SL + c0 + q] = s;
            }
            __syncwarp();
            gau_mbar_arrive(gau_smem_addr(oc + bb));
            if (bb == warp) {
                // diagonal block: dlarft inside the block, lane = row
                if (lane < VEC && c0 + lane < h) {
                    const int r = c0 + lane;
                    T t[VEC];
#pragma unroll
                    for (int cq = 0; cq < VEC; ++cq) {
                        const int c = c0 + cq;
                        t[cq] = T(0);
                        if (c < h) {
                            if (cq == lane)
                                t[cq] = taus[c];
                            else if (cq > lane) {
                                T acc = T(0);
#pragma unroll
                                for (int p = 0; p < cq; ++p)
                                    if (p >= lane)
                                        acc = gf_fma(t[p], refl[size_t(c) * SL + c0 + p], acc);
                                t[cq] = -taus[c] * acc;
                            }
                        }
                    }
#pragma unroll
                    for (int cq = 0; cq < VEC; ++cq) {
                        const int c = c0 + cq;
                        if (c < h && cq >= lane)
                            Tp[size_t(c) * (c + 1) / 2 + r] = t[cq];
                    }
                }
                __syncwarp();
                gau_mbar_arrive(gau_smem_addr(td + warp));
            } else {
                // rows c0.. of T12 for block bb: Y = T(rows, jb-1 ..) O(.., block), then
                // T(rows, block) = -Y T22
                gau_mbar_wait(gau_smem_addr(oc + bb), 0);
                gau_mbar_wait(gau_smem_addr(td + bb), 0);
#pragma unroll
                for (int m = 0; m < PPL; ++m) {
                    const int idx = (lane / LPP) + m * (32 / LPP), ri = idx % VEC, p = idx / VEC,
                              part = lane % LPP, i = c0 + ri;
                    T y = T(0);
                    if (i < h)
                        for (int qq = i + part; qq < jb; qq += LPP)
                            y = gf_fma(Tp[size_t(qq) * (qq + 1) / 2 + i],
                                       refl[size_t(jb + p) * SL + qq], y);
#pragma unroll
                    for (int o = LPP / 2; o; o >>= 1)
                        y += __shfl_xor_sync(0xffffffffu, y, o);
                    if (part == 0)
                        ys[p * VEC + ri] = y;
                }
                __syncwarp();
#pragma unroll
                for (int m = 0; m < PPL; ++m) {
                    const int idx = (lane / LPP) + m * (32 / LPP), ri = idx % VEC, jq = idx / VEC,
                              part = lane % LPP, i = c0 + ri, j = jb + jq;
                    if (part == 0 && i < h && j < h) {
                        T acc = T(0);
#pragma unroll
                        for (int p = 0; p < VEC; ++p)
                            if (p <= jq)
                                acc = gf_fma(ys[p * VEC + ri], Tp[size_t(j) * (j + 1) / 2 + jb + p],
                                             acc);
                        Tp[size_t(j) * (j + 1) / 2 + i] = -acc;
                    }
                }
                __syncwarp();
            }
        }
    __syncthreads();
    for (int j = warp; j < b; j += WARPS)
        for (int i = lane; i < b; i += 32)
            tri[i + size_t(j) * b] = (j < h && i <= j) ? Tp[size_t(j) * (j + 1) / 2 + i] : T(0);
}

template <class T> struct ContiguousRegisterSource {
    static constexpr bool late = false; // every column is there from the start
    T *a;
    int lda;
    __device__ void acquire(int, int) const {}
    // Row r (of `rows`) at column 0 and its first stored column; null past the block.
    __device__ T *row(int r, int rows, int &first) const {
        first = 0;
        return r < rows ? a + r : nullptr;
    }
    __device__ void publish(int, int) const {}
};
template <class T, int ITEMS, int VEC, int MAXW, int PIV>
__global__ void __launch_bounds__(MAXW * 32, 1)
    gx_sm1_kernel(T *a, int lda, const TilePacket *packet, T *vo, int ldv, T *tau_out, T *ts, int b,
                  int *status) {
    extern __shared__ __align__(16) unsigned char storage[];
    const TilePacket pk = *packet;
    const ContiguousRegisterSource<T> source{a + pk.row + size_t(pk.col) * lda, lda};
    gx_body<T, ITEMS, VEC, PIV>(source, pk.rows, pk.h, vo, ldv, ldv, tau_out,
                                ts + size_t(pk.tile) * b * b, b, status, storage);
}
// Launches gx_sm1_kernel: warps own VEC columns each, threads hold ITEMS rows, pivot rows lie in the
// first PIV items (h <= 32 PIV). False when the packet exceeds its registers or shared memory.
template <class T, int ITEMS, int VEC, int MAXW, int PIV>
bool launch_fused_gx(const RegisterPanel<T> &p, T *Ts, int b, cudaStream_t st) {
    auto kernel = gx_sm1_kernel<T, ITEMS, VEC, MAXW, PIV>;
    static bool checked = false, spill = false;
    static size_t cap = 0;
    if (!checked) {
        cudaFuncAttributes attr;
        CU(cudaFuncGetAttributes(&attr, kernel));
        spill = attr.localSizeBytes > 256; // a cold commit may keep row pointers on the stack
        cap = device_properties().sharedMemPerBlockOptin - attr.sharedSizeBytes - 1024;
        checked = true;
    }
    const int warps = (p.h + VEC - 1) / VEC;
    const size_t smem = gx_smem<T>(ITEMS, VEC, warps);
    if (spill || ITEMS * 32 < p.rows || warps > MAXW || p.h > 32 * PIV || smem > cap)
        return false;
    reserve_shared_memory(kernel, smem);
    kernel<<<1, warps * 32, smem, st>>>(p.a, p.lda, p.packet, p.v, p.ldv, p.tau, Ts, b, p.status);
    CU(cudaGetLastError());
    if (p.carrier)
        *p.carrier = Carrier{}.set(BlockLevel, 1, warps, 1);
    return true;
}
} // namespace qr_omega
