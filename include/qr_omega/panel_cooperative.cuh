#pragma once
// GE of a tall panel split among resident thread blocks (one cooperative launch per window of
// columns). Every block holds its share of the rows of the window in shared memory. Per column the
// blocks form partial norms and dot products over their rows, combine them in a fixed order after a
// grid barrier, and apply the reflector to their rows. Inside the window the columns advance in
// mini-panels of PW columns: dots and rank-1 updates stay inside the mini-panel, and at its
// boundary one product P = V^T A over the rows updates the rest of the window; the part of P
// against earlier columns also gives the off-diagonal block of T, T12 = -T1 (V1^T V2) T2. Between
// launches the host updates the columns beyond the window and joins the T factors of the windows.
#include "kernels.cuh"
#include <cooperative_groups.h>
namespace tqr {
// Largest number of blocks a panel is split among.
inline constexpr int coop_partition_limit = 128;
// Shared memory of a block: its rows of the window, the staged column and PW columns of P.
inline size_t coop_mini_shared_bytes(int local_height, int h, int pw, size_t word) {
    return (size_t(local_height) * (size_t(h) + 1) + size_t(pw) * size_t(h)) * word;
}
inline size_t coop_mini_partial_slots(int groups, int h, int pw) {
    // Two generations of 2 + 2 PW slots per block; the column scaling uses h slots of the same
    // region.
    const size_t generations = size_t(2) * (size_t(2) + size_t(2) * pw);
    return size_t(groups) * (generations > size_t(h) ? generations : size_t(h));
}
// The partials of the boundary product and their combination.
inline size_t coop_mini_boundary_slots(int groups, int h, int pw) {
    return size_t(groups) * size_t(pw) * size_t(h) + size_t(pw) * size_t(h);
}
inline size_t coop_mini_total_slots(int groups, int h, int pw) {
    return coop_mini_partial_slots(groups, h, pw) + coop_mini_boundary_slots(groups, h, pw);
}
// FP32 storage accumulates sums of squares in FP64: a squared FP32 value can neither overflow nor
// underflow, so its norm needs no scaling pass.
template <class T> constexpr bool coop_ssq64() {
    return std::is_same_v<T, float>;
}
__device__ __forceinline__ double coop_block_sum_d(double v, double *redd) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
#pragma unroll
    for (int o = 16; o; o >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0)
        redd[warp] = v;
    __syncthreads();
    double t = 0;
    if (warp == 0) {
        t = lane < warps ? redd[lane] : 0.0;
#pragma unroll
        for (int o = 16; o; o >>= 1)
            t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0)
            redd[32] = t;
    }
    __syncthreads();
    t = redd[32];
    __syncthreads();
    return t;
}
__device__ __forceinline__ void coop_put_d(float *lo, float *hi, double d) {
    const unsigned long long u = __double_as_longlong(d);
    *lo = __uint_as_float(unsigned(u));
    *hi = __uint_as_float(unsigned(u >> 32));
}
__device__ __forceinline__ double coop_get_d(float lo, float hi) {
    return __longlong_as_double(
        (long long)((unsigned long long)__float_as_uint(hi) << 32 | __float_as_uint(lo)));
}
// The FP64 norm partial: raw bits in two float slots for FP32 storage, the double itself for FP64.
template <class T> __device__ __forceinline__ void coop_put_norm(T *lo, T *hi, double d) {
    if constexpr (std::is_same_v<T, double>) {
        *lo = d;
        *hi = 0;
    } else
        coop_put_d((float *)lo, (float *)hi, d);
}
template <class T> __device__ __forceinline__ double coop_get_norm(T lo, T hi) {
    if constexpr (std::is_same_v<T, double>)
        return double(lo);
    else
        return coop_get_d(lo, hi);
}
// A block's partials for column k of the window, into the slots `out`: [0] scale, [1] sum of
// squares, [2 + jj] dot product with column j0 + jj, [2 + nw + jj] the pivot-row entry of that
// column.
template <class T>
__device__ __noinline__ void cooperative_prepare_window(const T *a, int ld, int local_rows,
                                                        int first, int w0, int j0, int nw, int k,
                                                        T *out, int outs, T *red, T *xs) {
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, warps = blockDim.x / 32;
    // The pivot is matrix row w0 + k and column k of the staged window.
    const int krow = w0 + k;
    int start = max(0, krow + 1 - first);
    T mx = 0;
    if constexpr (std::is_same_v<T, double>) {
        __shared__ double redd[33];
        double d = 0;
        for (int r = start + tid; r < local_rows; r += blockDim.x) {
            const double x = a[r + size_t(k) * ld];
            xs[r] = x;
            d += x * x;
        }
        d = coop_block_sum_d(d, redd);
        // Far from overflow and underflow the unscaled sum is used; otherwise the column is
        // rescaled by its largest magnitude, as LAPACK does. The branch is uniform in the block.
        if (d >= 0x1p-970 && d <= 0x1p970) {
            mx = sqrt(d);
            for (int r = start + tid; r < local_rows; r += blockDim.x)
                xs[r] /= mx;
            __syncthreads();
            if (tid == 0) {
                out[0] = mx;
                out[outs] = 1;
            }
        } else {
            for (int r = start + tid; r < local_rows; r += blockDim.x)
                mx = max(mx, absval(xs[r]));
            mx = tile_reduce<T, true>(mx, red);
            T sum = 0;
            if (mx)
                for (int r = start + tid; r < local_rows; r += blockDim.x) {
                    T x = xs[r] / mx;
                    xs[r] = x;
                    sum += x * x;
                }
            sum = tile_reduce<T, false>(sum, red);
            if (tid == 0) {
                out[0] = mx;
                out[outs] = sum;
            }
        }
    } else {
        __shared__ double redd[33];
        double d = 0;
        for (int r = start + tid; r < local_rows; r += blockDim.x) {
            const T x = a[r + size_t(k) * ld];
            xs[r] = x;
            d += double(x) * double(x);
        }
        d = coop_block_sum_d(d, redd);
        mx = T(sqrt(d));
        if (tid == 0)
            coop_put_norm<T>(&out[0], &out[outs], d);
        if (mx) {
            for (int r = start + tid; r < local_rows; r += blockDim.x)
                xs[r] = xs[r] / mx;
            __syncthreads();
        }
    }
    for (int jj = warp; jj < nw; jj += warps) {
        int j = j0 + jj;
        T dot = 0;
        if (j != k && mx)
            for (int r = start + lane; r < local_rows; r += 32)
                dot += xs[r] * a[r + size_t(j) * ld];
        dot = warp_add(dot);
        if (lane == 0) {
            out[size_t(2 + jj) * outs] = dot;
            out[size_t(2 + nw + jj) * outs] = (krow >= first && krow < first + local_rows)
                                                  ? a[krow - first + size_t(j) * ld]
                                                  : T(0);
        }
    }
}
// Warp sum whose only consumer is lane 0.
template <class T> __device__ __forceinline__ T panel_boundary_sum(T x) {
    for (int d = 16; d; d /= 2)
        x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}
// The kernel runs with 512 threads per block and every block resident at once.
template <class T, int PW>
__global__ __launch_bounds__(512) void cooperative_ge_mini(T *A, int lda, const TilePacket *packets,
                                                           int groups, int local_height, int sw,
                                                           int w0, T *Ts, int b, T *partial,
                                                           T *bpart, T *bfull, int *status) {
    auto grid = cooperative_groups::this_grid();
    const int part = blockIdx.x;
    auto p = packets[0];
    const int first = part * local_height, rows = max(0, min(local_height, p.rows - first));
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, warps = blockDim.x / 32;
    T *global = A + p.row + size_t(p.col + w0) * lda, *tri = Ts + size_t(p.tile) * b * b;
    extern __shared__ __align__(16) unsigned char memory[];
    T *a = reinterpret_cast<T *>(memory);
    T *xs = a + size_t(local_height) * sw;
    T *P = xs + local_height;
    __shared__ T red[32], scales[128], g[PW], betas[128], ratio[coop_partition_limit], t2[PW * PW];
    __shared__ T gm[PW * PW],
        taus_s[PW]; // combined dots v_j^T v_k (j < k) and tau of the current mini-panel
    const size_t stride = 2 + 2 * PW, generation = size_t(groups) * stride;
    T *base = partial, *mine = base + part;
    for (int i = tid; i < rows * sw; i += blockDim.x)
        a[i % rows + size_t(i / rows) * local_height] =
            global[first + i % rows + size_t(i / rows) * lda];
    // Only the first window clears T: later windows must keep the blocks their
    // predecessors wrote (the diagonal blocks and the composed off-diagonals).
    if (!w0 && part == 0)
        for (int i = tid; i < b * b; i += blockDim.x)
            tri[i] = 0;
    __syncthreads();
    for (int j = warp; j < sw; j += warps) {
        T mx = 0;
        for (int r = lane; r < rows; r += 32)
            mx = max(mx, absval(a[r + size_t(j) * local_height]));
        mx = warp_maximum(mx);
        if (lane == 0)
            mine[size_t(j) * groups] = mx;
    }
    grid.sync();
    for (int j = warp; j < sw; j += warps) {
        T mx = 0;
        for (int i = lane; i < groups; i += 32)
            mx = max(mx, base[size_t(j) * groups + i]);
        mx = warp_maximum(mx);
        if (lane == 0)
            scales[j] = mx;
        for (int r = lane; r < rows; r += 32)
            if (mx)
                a[r + size_t(j) * local_height] /= mx;
    }
    // Every reader of the scaling generation retires before it becomes partials.
    grid.sync();
    for (int m0 = 0; m0 < sw; m0 += PW) {
        const int m1 = min(m0 + PW, sw), nw = m1 - m0;
        cooperative_prepare_window(a, local_height, rows, first, w0, m0, nw, m0, mine, groups, red,
                                   xs);
        for (int k = m0; k < m1; ++k) {
            const int kk = k - m0, krow = w0 + k;
            grid.sync();
            T *read = base + size_t(kk % 2) * generation;
            const T *sig = read, *ssq = read + groups;
            const int owner = krow / local_height;
            // All the partials of this column are loaded at once, one per thread.
            const bool pf = groups <= int(blockDim.x) && groups <= 128;
            T pf_sig = 0, pf_ssq = 0, pf_dot[4] = {0, 0, 0, 0}, pf_row = 0;
            if (pf) {
                if (tid < groups) {
                    pf_sig = sig[tid];
                    pf_ssq = ssq[tid];
                }
                if (warp < nw) {
#pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        const int i = lane + 32 * m;
                        if (i < groups)
                            pf_dot[m] = read[size_t(2 + warp) * groups + i];
                    }
                    pf_row = read[size_t(2 + nw + warp) * groups + owner];
                }
            }
            const T alpha = read[size_t(2 + nw + kk) * groups + owner];
            T tail = 0, scale = 1, s = 0;
            double sd = 0;
            (void)s;
            if constexpr (coop_ssq64<T>()) {
                __shared__ double redd[33];
                if (pf) {
                    if (tid < groups) {
                        sd = coop_get_norm<T>(pf_sig, pf_ssq);
                        ratio[tid] = T(1);
                    }
                } else
                    for (int i = tid; i < groups; i += blockDim.x) {
                        sd += coop_get_norm<T>(sig[i], ssq[i]);
                        ratio[i] = T(1);
                    }
                sd = coop_block_sum_d(sd, redd);
                tail = sd > 0 ? T(1) : T(0);
                // The norm needs no max pass, but dots still need a scaled column:
                // unscaled fp32 products underflow for nearly dependent columns. Match
                // prepare's local norm and combine the scaled dots in the original law.
                scale = max(T(sqrt(sd)), absval(alpha));
                for (int i = tid; i < groups; i += blockDim.x) {
                    const double d = coop_get_norm<T>(sig[i], ssq[i]);
                    ratio[i] = scale ? T(sqrt(d)) / scale : T(0);
                }
                __syncthreads();
            } else {
                for (int i = tid; i < groups; i += blockDim.x)
                    tail = max(tail, pf && i == tid ? pf_sig : sig[i]);
                tail = tile_reduce<T, true>(tail, red);
                scale = max(tail, absval(alpha));
                if (tail)
                    for (int i = tid; i < groups; i += blockDim.x) {
                        T r = sig[i] / scale;
                        ratio[i] = r;
                        s += ssq[i] * r * r;
                    }
            }
            if constexpr (!coop_ssq64<T>())
                s = tile_reduce<T, false>(s, red);
            T beta = alpha, tau = 0, den = 1;
            if constexpr (coop_ssq64<T>()) {
                if (tail) {
                    const double an = double(alpha);
                    double bn = -sqrt(an * an + sd);
                    if (an < 0)
                        bn = -bn;
                    beta = T(bn);
                    den = T((an - bn) / double(scale));
                    tau = T(1.0 - an / bn);
                }
            } else if (tail) {
                const T an = alpha / scale;
                T sum = an * an + s, bn = -sqrt(sum);
                if (an < 0)
                    bn = -bn;
                beta = scale * bn;
                den = an - bn;
                tau = 1 - an / bn;
            }
            if (tid == 0)
                betas[k] = beta;
            if (tid == 0)
                taus_s[kk] = tau;
            if (tau)
                for (int r = max(0, krow + 1 - first) + tid; r < rows; r += blockDim.x)
                    a[r + size_t(k) * local_height] =
                        (a[r + size_t(k) * local_height] / scale) / den;
            __syncthreads();
            for (int jj = warp; jj < nw; jj += warps) {
                const int j = m0 + jj;
                // The warp of the column being factored has no update to do: it extends T2 by
                // column kk - 1, whose dots and tau are complete since the last barrier. The last
                // column is left to the boundary.
                if (j == k) {
                    if (kk > 0) {
                        const int c = kk - 1;
                        const T tc = taus_s[c];
                        T sum = 0;
                        if (lane < c) {
                            for (int q = lane; q < c; ++q)
                                sum += t2[lane + size_t(q) * nw] * gm[c * PW + q];
                        }
                        if (lane < nw)
                            t2[lane + size_t(c) * nw] =
                                lane < c ? -tc * sum : (lane == c ? tc : T(0));
                    }
                    continue;
                }
                T dot = 0;
                const bool mine = pf && jj == warp;
                if (tau) {
                    if (mine) {
#pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            const int i = lane + 32 * m;
                            if (i < groups)
                                dot += pf_dot[m] * ratio[i];
                        }
                    } else
                        for (int i = lane; i < groups; i += 32)
                            dot += read[size_t(2 + jj) * groups + i] * ratio[i];
                }
                dot = warp_add(dot);
                T value = (mine ? pf_row : read[size_t(2 + nw + jj) * groups + owner]) +
                          (tau ? dot / den : T(0));
                if (j < k) {
                    if (lane == 0)
                        gm[kk * PW + jj] = value;
                } else {
                    T update = tau * value;
                    for (int r = max(0, krow + 1 - first) + lane; r < rows; r += 32) {
                        T v = tau ? a[r + size_t(k) * local_height] : T(0);
                        a[r + size_t(j) * local_height] -= v * update;
                    }
                    if (part == owner && lane == 0)
                        a[krow - first + size_t(j) * local_height] -= update;
                }
            }
            __syncthreads();
            // V has a unit diagonal; beta rejoins A at the writeback. Nothing reads the pivot
            // before the next barrier.
            if (part == owner && tid == 0)
                a[krow - first + size_t(k) * local_height] = T(1);
            if (k + 1 < m1)
                cooperative_prepare_window(a, local_height, rows, first, w0, m0, nw, k + 1,
                                           mine + size_t((kk + 1) % 2) * generation, groups, red,
                                           xs);
        }
        // The last column of T2: warp 0 of every block runs the recursion (lane i owns row i).
        if (warp == 0) {
            for (int kk = nw - 1; kk < nw; ++kk) {
                const T tk = taus_s[kk];
                T sum = 0;
                if (lane < kk) {
                    for (int jj = lane; jj < kk; ++jj)
                        sum += t2[lane + size_t(jj) * nw] * gm[kk * PW + jj];
                }
                __syncwarp();
                if (lane < nw)
                    t2[lane + size_t(kk) * nw] = lane < kk ? -tk * sum : (lane == kk ? tk : T(0));
                __syncwarp();
            }
            if (part == 0)
                for (int i = lane; i < nw * nw; i += 32)
                    tri[w0 + m0 + i % nw + size_t(w0 + m0 + i / nw) * b] = t2[i];
        }
        if (nw >= sw)
            break;
        // Mini-panel boundary: P = V^T A[:, columns outside the mini-panel], summed over the row
        // sets of the blocks. Every warp accumulates JT columns of P in registers.
        {
            constexpr int JT0 = 4 * 16 / PW / (sizeof(T) == 8 ? 2 : 1), JT = JT0 > 0 ? JT0 : 1;
            const bool below = first >= w0 + m1;
            for (int j0 = warp * JT; j0 < sw; j0 += warps * JT) {
                T acc[PW][JT];
                bool live[JT];
                bool any = false;
#pragma unroll
                for (int jt = 0; jt < JT; ++jt) {
                    const int j = j0 + jt;
                    live[jt] = j < sw && (j < m0 || j >= m1);
                    any |= live[jt];
#pragma unroll
                    for (int c = 0; c < PW; ++c)
                        acc[c][jt] = 0;
                }
                // Columns inside the mini-panel publish zeros without reading the rows.
                if (any)
                    for (int r = lane; r < rows; r += 32) {
                        const int gr = first + r;
                        T x[JT];
#pragma unroll
                        for (int jt = 0; jt < JT; ++jt)
                            x[jt] = live[jt] ? a[r + size_t(j0 + jt) * local_height] : T(0);
#pragma unroll
                        for (int c = 0; c < PW; ++c)
                            if (c < nw) {
                                const T v = (below || gr >= w0 + m0 + c)
                                                ? a[r + size_t(m0 + c) * local_height]
                                                : T(0);
#pragma unroll
                                for (int jt = 0; jt < JT; ++jt)
                                    acc[c][jt] += v * x[jt];
                            }
                    }
                // Transposed butterfly: after 31 shuffles lane l holds the sum of partial l over
                // the lanes.
                static_assert((PW * JT) % 32 == 0, "whole 32-value groups");
                T *flat = &acc[0][0];
#pragma unroll
                for (int g0 = 0; g0 < PW * JT; g0 += 32) {
                    T *v = flat + g0;
#pragma unroll
                    for (int s = 16; s >= 1; s >>= 1) {
                        const bool up = (lane & s) != 0;
#pragma unroll
                        for (int i = 0; i < s; ++i) {
                            const T send = up ? v[i] : v[i + s], keep = up ? v[i + s] : v[i];
                            v[i] = keep + __shfl_xor_sync(0xffffffffu, send, s);
                        }
                    }
                    const int e = g0 + lane, c = e / JT, jt = e % JT, j = j0 + jt;
                    if (c < nw && j < sw)
                        bpart[size_t(c * sw + j) * groups + part] =
                            (j < m0 || j >= m1) ? v[0] : T(0);
                }
            }
        }
        grid.sync();
        // Each block combines a disjoint slice of P and commits it once.
        for (int idx = part + warp * groups; idx < nw * sw; idx += warps * groups) {
            T acc = 0;
            for (int i = lane; i < groups; i += 32)
                acc += bpart[size_t(idx) * groups + i];
            acc = panel_boundary_sum(acc);
            if (lane == 0)
                bfull[idx] = acc;
        }
        grid.sync();
        for (int i = tid; i < nw * sw; i += blockDim.x)
            P[i] = bfull[i];
        __syncthreads();
        // P <- T2^T P, in registers so the triangular sweep needs no extra barrier.
        for (int j = tid; j < sw; j += blockDim.x) {
            if (j >= m0 && j < m1)
                continue;
            T col[PW];
#pragma unroll
            for (int c = 0; c < PW; ++c)
                col[c] = (c < nw) ? P[size_t(c) * sw + j] : T(0);
            // Unrolled over the compile-time width so col[] keeps static indices and
            // stays in registers; a dynamic bound here put it in local memory.
            for (int c = nw - 1; c >= 0; --c) {
                T acc = 0;
#pragma unroll
                for (int cp = 0; cp < PW; ++cp)
                    if (cp <= c)
                        acc += t2[cp + size_t(c) * nw] * col[cp];
                P[size_t(c) * sw + j] = acc;
            }
        }
        __syncthreads();
        // A[:, j >= m1] -= V P, four columns per thread so that each V entry feeds four FMAs.
        const int ntr = sw - m1;
        if constexpr (sizeof(T) == 4) // FP32 keeps four columns of P in registers
            if (ntr > 0) {
                const int ng = (ntr + 3) / 4;
                int gg = ng, ww = warps;
                while (ww) {
                    const int t = gg % ww;
                    gg = ww;
                    ww = t;
                } // gg = gcd(ng, warps)
                const int rs = min(8, warps / gg),
                          items = ng * rs; // row slices balance the items over the warps
                for (int it = warp; it < items; it += warps) {
                    const int jg = it % ng, sl = it / ng, j0 = m1 + jg * 4;
                    T pr[PW][4];
#pragma unroll
                    for (int c = 0; c < PW; ++c) {
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            pr[c][u] = (c < nw && j0 + u < sw) ? P[size_t(c) * sw + j0 + u] : T(0);
                    }
                    for (int r = sl * 32 + lane; r < rows; r += 32 * rs) {
                        const int gr = first + r;
                        T acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
                        for (int c = 0; c < PW; ++c)
                            if (c < nw) {
                                const T vc = (gr >= w0 + m0 + c)
                                                 ? a[r + size_t(m0 + c) * local_height]
                                                 : T(0);
                                acc0 += vc * pr[c][0];
                                acc1 += vc * pr[c][1];
                                acc2 += vc * pr[c][2];
                                acc3 += vc * pr[c][3];
                            }
                        a[r + size_t(j0) * local_height] -= acc0;
                        if (j0 + 1 < sw)
                            a[r + size_t(j0 + 1) * local_height] -= acc1;
                        if (j0 + 2 < sw)
                            a[r + size_t(j0 + 2) * local_height] -= acc2;
                        if (j0 + 3 < sw)
                            a[r + size_t(j0 + 3) * local_height] -= acc3;
                    }
                }
            }
        if constexpr (!(sizeof(T) == 4 && PW <= 16)) {
            if (ntr > 0) {
                const int jt = (ntr + 3) / 4;
                for (int idx = tid; idx < rows * jt; idx += blockDim.x) {
                    const int r = idx % rows, jg = idx / rows, j0 = m1 + jg * 4, gr = first + r;
                    T v[PW];
#pragma unroll
                    for (int c = 0; c < PW; ++c)
                        v[c] = (c < nw && gr >= w0 + m0 + c) ? a[r + size_t(m0 + c) * local_height]
                                                             : T(0);
                    T acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
#pragma unroll
                    for (int c = 0; c < PW; ++c)
                        if (c < nw) {
                            const T vc = v[c];
                            const T *pc = P + size_t(c) * sw + j0;
                            acc0 += vc * pc[0];
                            if (j0 + 1 < sw)
                                acc1 += vc * pc[1];
                            if (j0 + 2 < sw)
                                acc2 += vc * pc[2];
                            if (j0 + 3 < sw)
                                acc3 += vc * pc[3];
                        }
                    a[r + size_t(j0) * local_height] -= acc0;
                    if (j0 + 1 < sw)
                        a[r + size_t(j0 + 1) * local_height] -= acc1;
                    if (j0 + 2 < sw)
                        a[r + size_t(j0 + 2) * local_height] -= acc2;
                    if (j0 + 3 < sw)
                        a[r + size_t(j0 + 3) * local_height] -= acc3;
                }
            }
        }
        // T12 = -T1 (V1^T V2) T2, and (V1^T V2) T2 is the j < m0 part of P.
        for (int gi = part + tid * groups; gi < m0 * nw; gi += size_t(blockDim.x) * groups) {
            const int i = gi / nw, c = gi - i * nw;
            T acc = 0;
            for (int j = i; j < m0; ++j)
                acc += tri[w0 + i + size_t(w0 + j) * b] * P[size_t(c) * sw + j];
            tri[w0 + i + size_t(w0 + m0 + c) * b] = -acc;
        }
        __syncthreads();
    }
    for (int i = tid; i < rows * sw; i += blockDim.x) {
        int r = i % rows, j = i / rows, gj = w0 + j;
        T v = (first + r == gj) ? betas[j] : a[r + size_t(j) * local_height];
        if (first + r <= gj) {
            v *= scales[j];
            if (!isfinite(v))
                atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
        }
        global[first + r + size_t(j) * lda] = v;
    }
}
// Launches the kernel on the window [w0, w0 + sw) of a rows x h packet, its rows split among
// `groups` resident blocks; with `dry` it only checks. False when the blocks do not all fit on the
// GPU or a block's share of the window exceeds its shared memory.
template <class T, int PW>
bool launch_cooperative_kernel(T *A, int ld, const TilePacket *packets, int rows, int h, int w0,
                               int sw, int groups, int threads, T *tri, int b, T *partial,
                               int *status, cudaStream_t stream, bool dry, int max_h) {
    const cudaDeviceProp &prop = device_properties();
    if (!prop.cooperativeLaunch || groups < 1 || groups > coop_partition_limit || h < 1 ||
        h > max_h || max_h > 512 || rows < h || threads < 32 || threads > 1024 || threads % 32)
        return false;
    if (sw < 1 || sw > 128 || sw > h || w0 < 0 || w0 + sw > h)
        return false;
    int local_height = ceildiv(rows, groups);
    size_t dynamic = coop_mini_shared_bytes(local_height, sw, PW, sizeof(T));
    cudaFuncAttributes attr;
    CU(cudaFuncGetAttributes(&attr, cooperative_ge_mini<T, PW>));
    if (dynamic > prop.sharedMemPerBlockOptin - attr.sharedSizeBytes)
        return false;
    reserve_shared_memory(cooperative_ge_mini<T, PW>, dynamic);
    int occupancy = 0;
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, cooperative_ge_mini<T, PW>,
                                                     threads, dynamic));
    if (size_t(groups) > size_t(occupancy) * prop.multiProcessorCount)
        return false;
    if (dry)
        return true;
    T *bpart = partial + coop_mini_partial_slots(groups, sw, PW),
      *bfull = bpart + size_t(groups) * PW * sw;
    void *args[] = {&A,   &ld, &packets, &groups, &local_height, &sw,    &w0,
                    &tri, &b,  &partial, &bpart,  &bfull,        &status};
    CU(cudaLaunchCooperativeKernel((const void *)cooperative_ge_mini<T, PW>, dim3(groups),
                                   dim3(threads), args, dynamic, stream));
    return true;
}
// `pw` is the mini-panel width, 8 or 16 columns.
template <class T>
bool launch_cooperative_window(T *A, int ld, const TilePacket *packets, int rows, int h, int w0,
                               int sw, int pw, int groups, int threads, T *tri, int b, T *partial,
                               int *status, cudaStream_t stream, bool dry, int max_h) {
    if (pw == 8)
        return launch_cooperative_kernel<T, 8>(A, ld, packets, rows, h, w0, sw, groups, threads,
                                               tri, b, partial, status, stream, dry, max_h);
    if (pw == 16)
        return launch_cooperative_kernel<T, 16>(A, ld, packets, rows, h, w0, sw, groups, threads,
                                                tri, b, partial, status, stream, dry, max_h);
    return false;
}
} // namespace tqr
