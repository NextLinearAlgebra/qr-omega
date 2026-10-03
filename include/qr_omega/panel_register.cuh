#pragma once
// Register-resident panels for the last panels of the factorization and the TT merges, adapted from
// gau.nernst's QR-v2 kernels for the GPU MODE leaderboard: a block or a warp owns a group of
// columns with all their rows in registers and publishes each reflector to the owners on its right.
// Two kernels share this scheme: a chain of blocks that hand reflectors over through global memory,
// and a fused kernel in which the warps of one block hand them over through shared memory and also
// form V^T V and T. Their arithmetic is LAPACK's: dlarfg reflectors with IEEE square roots and
// divisions, FP64 sums of squares for FP32 storage, and T built by dlarft from the overlaps of the
// reflectors, never from A^T A.
#include "kernels.cuh"
#include <cooperative_groups.h>
#include <array>
#include <vector>
namespace tqr {
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
    return true;
}
// Factors the packet with a chain of blocks, each consuming the reflectors of the owners to its
// left before it generates its own. The configurations hold ITEMS x THREADS rows; false, before
// anything is written, when the packet is taller than the largest. `wide` admits up to 512 columns.
template <class T>
bool launch_chain_panel(const RegisterPanel<T> &p, int *flags, int epoch, bool wide,
                        cudaStream_t st) {
    if (p.h < 1 || p.h > (wide ? 512 : 128) || p.rows < p.h)
        return false;
    const int rows = p.rows;
    if constexpr (sizeof(T) == 4)
        return rows <= 512    ? launch_chain<T, 8, 2, 256>(p, flags, epoch, st)
               : rows <= 1024 ? launch_chain<T, 8, 4, 256>(p, flags, epoch, st)
               : rows <= 2048 ? launch_chain<T, 8, 8, 256>(p, flags, epoch, st)
               : rows <= 4096 ? launch_chain<T, 8, 16, 256>(p, flags, epoch, st)
               : rows <= 8192 ? launch_chain<T, 4, 16, 512>(p, flags, epoch, st)
                              : false;
    else
        return rows <= 256    ? launch_chain<T, 8, 2, 128>(p, flags, epoch, st)
               : rows <= 512  ? launch_chain<T, 8, 2, 256>(p, flags, epoch, st)
               : rows <= 1024 ? launch_chain<T, 8, 4, 256>(p, flags, epoch, st)
               : rows <= 2048 ? launch_chain<T, 8, 8, 256>(p, flags, epoch, st)
               : rows <= 4096 ? launch_chain<T, 4, 16, 256>(p, flags, epoch, st)
               : rows <= 8192 ? launch_chain<T, 2, 16, 512>(p, flags, epoch, st)
                              : false;
}

// T of up to 512 reflectors (the TT merge across GPUs): the 128-column diagonal blocks by the same
// recurrence, then the off-diagonal blocks joined pairwise at growing spans.
template <class T>
__global__ __launch_bounds__(512) void multi_wide_t_diagonal(const T *G, int ldg, const T *tau,
                                                             const TilePacket *packet, T *Ts,
                                                             int b) {
    constexpr int H = 128, LD = H + 1;
    extern __shared__ __align__(16) unsigned char wide_smem[];
    T *Sm = reinterpret_cast<T *>(wide_smem), *M = Sm + H * LD;
    const TilePacket pk = *packet;
    const int base = blockIdx.x * H, h = min(H, pk.h - base);
    if (h <= 0)
        return;
    gau_load_overlaps<T, H>(Sm, G, ldg, tau, base, h, threadIdx.x);
    gau_build_t<T, H>(Sm, M, threadIdx.x);
    T *tri = Ts + size_t(pk.tile) * b * b;
    for (int e = threadIdx.x; e < H * H; e += 512) {
        const int i = e % H, j = e / H;
        if (i < h && j < h && i <= j)
            tri[base + i + size_t(base + j) * b] = Sm[i + j * LD];
    }
}

template <class T> __global__ void multi_wide_t_zero(const TilePacket *packet, T *Ts, int b) {
    T *tri = Ts + size_t(packet[0].tile) * b * b;
    for (int i = threadIdx.x; i < b * b; i += blockDim.x)
        tri[i] = T(0);
}

// T12 = -(T1 G12) T2 with G12 = V1^T V2. Every output sums two disjoint halves of its contraction,
// which meet in shared memory and are added in a fixed order.
template <class T, bool Right>
__global__ void multi_wide_t_join(const T *G, int ldg, const TilePacket *packet, T *Ts, int b,
                                  T *temporary, int span) {
    const int base = int(blockIdx.y) * 2 * span, h = packet[0].h;
    const int left = min(span, h - base), right = min(span, h - base - left);
    if (left <= 0 || right <= 0)
        return;
    const int element = blockIdx.x * blockDim.x + threadIdx.x, peer = threadIdx.y;
    const bool active = element < left * right;
    const int i = element % left, j = element / left, contraction = Right ? right : left;
    const int carriers = min(2, contraction), first = contraction * peer / carriers,
              last = contraction * (peer + 1) / carriers;
    T *tri = Ts + size_t(packet[0].tile) * b * b;
    T sum = T(0);
    if (active && peer < carriers)
        for (int k = first; k < last; ++k) {
            if constexpr (Right) {
                if (k <= j)
                    sum += temporary[base + i + size_t(base + left + k) * b] *
                           tri[base + left + k + size_t(base + left + j) * b];
            } else {
                if (k >= i)
                    sum += tri[base + i + size_t(base + k) * b] *
                           G[base + k + size_t(base + left + j) * ldg];
            }
        }
    __shared__ T partial[2][128];
    partial[peer][threadIdx.x] = sum;
    __syncthreads();
    if (peer == 0 && active) {
        const T value = partial[0][threadIdx.x] + partial[1][threadIdx.x];
        if constexpr (Right)
            tri[base + i + size_t(base + left + j) * b] = -value;
        else
            temporary[base + i + size_t(base + left + j) * b] = value;
    }
}

// Whether the 512-thread T builder fits for a packet of h > 128 columns; it also sizes its shared
// memory.
template <class T> bool wide_t_fits(int h, int b) {
    if (h <= 128 || h > 512 || b < h)
        return false;
    cudaFuncAttributes attr;
    CU(cudaFuncGetAttributes(&attr, multi_wide_t_diagonal<T>));
    const size_t bytes = gau_build_t_h_smem<T, 128>();
    if (bytes > device_properties().sharedMemPerBlockOptin - attr.sharedSizeBytes)
        return false;
    reserve_shared_memory(multi_wide_t_diagonal<T>, bytes);
    int occupancy = 0;
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, multi_wide_t_diagonal<T>, 512,
                                                     bytes));
    return occupancy > 0;
}

template <class T>
void launch_multi_wide_t(const T *G, int ldg, const T *tau, const TilePacket *packet, int h, T *Ts,
                         int b, T *temporary, cudaStream_t st) {
    multi_wide_t_zero<T><<<1, 256, 0, st>>>(packet, Ts, b);
    multi_wide_t_diagonal<T>
        <<<ceildiv(h, 128), 512, gau_build_t_h_smem<T, 128>(), st>>>(G, ldg, tau, packet, Ts, b);
    for (int span = 128; span < h; span *= 2) {
        const dim3 grid(ceildiv(span * span, 128), ceildiv(h, 2 * span)), threads(128, 2);
        multi_wide_t_join<T, false>
            <<<grid, threads, 0, st>>>(G, ldg, packet, Ts, b, temporary, span);
        multi_wide_t_join<T, true>
            <<<grid, threads, 0, st>>>(G, ldg, packet, Ts, b, temporary, span);
    }
    CU(cudaGetLastError());
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
// x_q <- x_q - v tau (v^T x_q) for the warp's columns q >= first (all rows in registers, v in
// registers).
template <class T, int ITEMS, int VEC>
__device__ __forceinline__ void gf_apply(T (&col)[ITEMS][VEC], const T (&v)[ITEMS], T tk, int first,
                                         int lane) {
    T d[VEC];
#pragma unroll
    for (int q = 0; q < VEC; ++q) {
        T s = T(0);
        if (q >= first) {
#pragma unroll
            for (int it = 0; it < ITEMS; ++it)
                s = gf_fma(v[it], col[it][q], s);
        }
        d[q] = s;
    }
    gf_allreduce<T, VEC>(d, lane);
#pragma unroll
    for (int q = 0; q < VEC; ++q)
        if (q >= first) {
            const T f = -(d[q] * tk);
#pragma unroll
            for (int it = 0; it < ITEMS; ++it)
                col[it][q] = gf_fma(v[it], f, col[it][q]);
        }
}

// ---- the fused kernel ---------------------------------------------------------------------------
// One block per packet; warp w owns columns [w VEC, (w + 1) VEC) with all ROWS = 32 ITEMS rows in
// registers, and the same rows of T. Shared memory: refl [COLS][SL] (slot c holds v_c on rows >= c
// and O(k, c) = v_k^T v_c on rows k < c; the stride SL = ROWS + 16 bytes keeps bulk alignment and
// moves consecutive slots to different banks), the taus, mbarriers (reflector published, overlaps
// of a block complete, diagonal T block ready), the packed upper T and per-warp scratch. T follows
// dlarft's left-looking blocked recurrence while the chain runs: once the overlaps of block bb are
// complete, its owner forms the diagonal block T22 and every earlier warp w forms its rows of
// T12 = -T11 O12 T22 as -Y T22 with Y = T(rows_w, :) O(:, block).
template <class T> __host__ __device__ constexpr int gf_slot(int rows) {
    return rows + 16 / int(sizeof(T));
}
template <class T> inline size_t gf_sm1_smem(int items, int vec, int warps) {
    const size_t cols = size_t(warps) * vec, sl = size_t(gf_slot<T>(items * 32));
    return (cols * sl + cols + (cols & 1)) * sizeof(T) + (cols + 2 * size_t(warps)) * 8 +
           (cols * (cols + 1) / 2 + size_t(warps) * vec * vec) * sizeof(T) + 64;
}
template <class T, int ITEMS, int VEC, int MAXW>
__global__ void __launch_bounds__(MAXW * 32, 1)
    gf_sm1_kernel(T *A, int lda, const TilePacket *packet, T *vo, int ldv, T *tau_out, T *Ts, int b,
                  int *status) {
    constexpr int ROWS = ITEMS * 32, SL = gf_slot<T>(ROWS);
    const int nt = blockDim.x, WARPS = nt >> 5, COLS = WARPS * VEC;
    extern __shared__ __align__(16) unsigned char gau_smem[];
    const TilePacket pk = *packet;
    const int rows = pk.rows, h = pk.h;
    T *refl = reinterpret_cast<T *>(gau_smem);
    T *taus = refl + size_t(COLS) * SL;
    unsigned long long *mb = reinterpret_cast<unsigned long long *>(taus + COLS + (COLS & 1));
    unsigned long long *oc = mb + COLS, *td = oc + WARPS;
    T *Tp = reinterpret_cast<T *>(td + WARPS);   // T(i,j), i <= j, at j(j+1)/2 + i
    T *ysc = Tp + size_t(COLS) * (COLS + 1) / 2; // [WARPS][VEC][VEC]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c0 = warp * VEC,
              nb = (h + VEC - 1) / VEC;
    T *a = A + pk.row + size_t(pk.col) * lda;
    if (tid == 0) {
        for (int i = 0; i < COLS; ++i)
            gau_mbar_init(gau_smem_addr(mb + i), 32);
        for (int bb = 0; bb < WARPS; ++bb) {
            gau_mbar_init(gau_smem_addr(oc + bb), bb + 1);
            gau_mbar_init(gau_smem_addr(td + bb), 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
    T col[ITEMS][VEC];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * 32 + lane;
#pragma unroll
        for (int q = 0; q < VEC; ++q)
            col[it][q] = (r < rows && c0 + q < h) ? a[r + size_t(c0 + q) * lda] : T(0);
    }
    // consume every reflector of the warps to the left, in order
    const int kend = min(c0, h);
    for (int k = 0; k < kend; ++k) {
        gau_mbar_wait(gau_smem_addr(mb + k), 0);
        T v[ITEMS];
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            v[it] = r >= k ? refl[size_t(k) * SL + r] : T(0);
        }
        gf_apply<T, ITEMS, VEC>(col, v, taus[k], 0, lane);
    }
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        // Generate, publish (shared slot, V column and tau in global memory) and apply this warp's
        // reflectors.
        const int c = c0 + i;
        if (c >= h)
            break;
        const T tk = gf_reflector<T, ITEMS, VEC>(col, i, c, rows, lane, status);
        T v[ITEMS];
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            v[it] = r == c ? T(1) : r > c ? col[it][i] : T(0);
            if (r >= c)
                refl[size_t(c) * SL + r] = v[it];
        }
        if (lane == 0)
            taus[c] = tk;
        gau_mbar_arrive(gau_smem_addr(mb + c));
        if (i + 1 < VEC)
            gf_apply<T, ITEMS, VEC>(col, v, tk, i + 1, lane);
#pragma unroll
        for (int it = 0; it < ITEMS; ++it) {
            const int r = it * 32 + lane;
            if (r < ldv)
                vo[r + size_t(c) * ldv] = v[it];
        }
        for (int r = ROWS + lane; r < ldv; r += 32)
            vo[r + size_t(c) * ldv] = T(0);
        if (lane == 0)
            tau_out[c] = tk;
    }
    // commit: R above, beta on, v below the diagonal (the owner writes its columns once)
    bool bad = false;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * 32 + lane;
        if (r < rows) {
#pragma unroll
            for (int q = 0; q < VEC; ++q)
                if (c0 + q < h) {
                    const T y = col[it][q];
                    bad |= !isfinite(y);
                    a[r + size_t(c0 + q) * lda] = y;
                }
        }
    }
    if (bad)
        atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
        const int r = it * 32 + lane;
#pragma unroll
        for (int q = 0; q < VEC; ++q) {
            const int c = c0 + q;
            col[it][q] = (c < h && r >= c) ? (r == c ? T(1) : col[it][q]) : T(0);
        }
    }
    constexpr int SH = 5 - GfLog<VEC>::v;
    constexpr int NP = VEC * VEC, PPL = NP >= 32 ? NP / 32 : 1,
                  LPP = NP >= 32 ? 1 : 32 / NP; // (row, column) pairs per lane / lanes per pair
    T *ys = ysc + size_t(warp) * NP;
    if (c0 < h)
        for (int bb = warp; bb < nb; ++bb) {
            const int jb = bb * VEC, j0 = max(jb, c0 + 1), j1 = min(jb + VEC, h);
            // overlaps O(c0+q, j) of this warp's reflectors with block bb's
            for (int j = j0; j < j1; ++j) {
                if (bb > warp)
                    gau_mbar_wait(gau_smem_addr(mb + j), 0);
                T d[VEC];
#pragma unroll
                for (int q = 0; q < VEC; ++q)
                    d[q] = T(0);
#pragma unroll
                for (int it = 0; it < ITEMS; ++it) {
                    const int r = it * 32 + lane;
                    if (r >= j) {
                        const T y = refl[size_t(j) * SL + r];
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
            if (lane == 0)
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
                if (lane == 0)
                    gau_mbar_arrive(gau_smem_addr(td + warp));
            } else {
                // rows c0.. of T12 for block bb: Y = T(rows, jb-1 ..) O(.., block), then T(rows,
                // block) = -Y T22
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
    T *tri = Ts + size_t(pk.tile) * b * b;
    for (int j = warp; j < b; j += WARPS)
        for (int i = lane; i < b; i += 32)
            tri[i + size_t(j) * b] = (j < h && i <= j) ? Tp[size_t(j) * (j + 1) / 2 + i] : T(0);
}

// Launches the kernel whose warps own VEC columns each and whose threads hold ITEMS rows of them.
// False when the packet exceeds its registers or its shared memory.
template <class T, int ITEMS, int VEC, int MAXW>
bool launch_fused(const RegisterPanel<T> &p, T *Ts, int b, cudaStream_t st) {
    auto kernel = gf_sm1_kernel<T, ITEMS, VEC, MAXW>;
    static bool checked = false, spill = false;
    static size_t cap = 0;
    if (!checked) {
        cudaFuncAttributes attr;
        CU(cudaFuncGetAttributes(&attr, kernel));
        spill = attr.localSizeBytes > 128;
        cap = device_properties().sharedMemPerBlockOptin - attr.sharedSizeBytes - 1024;
        checked = true;
    }
    const int warps = (p.h + VEC - 1) / VEC;
    const size_t smem = gf_sm1_smem<T>(ITEMS, VEC, warps);
    if (spill || ITEMS * 32 < p.rows || warps > MAXW || smem > cap)
        return false;
    reserve_shared_memory(kernel, smem);
    kernel<<<1, warps * 32, smem, st>>>(p.a, p.lda, p.packet, p.v, p.ldv, p.tau, Ts, b, p.status);
    CU(cudaGetLastError());
    return true;
}
// Factors the packet inside one block, which also forms V^T V and T. False, before anything is
// written, when the packet has more than 128 rows (FP64) or 256 rows (FP32) or 128 columns.
template <class T>
bool launch_fused_panel(const RegisterPanel<T> &p, T *Ts, int b, cudaStream_t st) {
    if (p.h < 1 || p.h > 128 || p.rows < p.h || b < p.h)
        return false;
    if (launch_fused<T, 4, 8, 16>(p, Ts, b, st))
        return true;
    if constexpr (sizeof(T) == 4)
        return launch_fused<T, 8, 8, 16>(p, Ts, b, st);
    return false;
}
} // namespace tqr
