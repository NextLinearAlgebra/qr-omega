#pragma once
// Stable Householder reconstruction from the thin orthonormal Q of an HQR panel.
// Ballard et al., EECS-2013-175, Algorithms 5/6; LAPACK xORHR_COL.
// Modified LU factors Q - [S; 0] = Y U, with S_j = -sign(pivot_j).
// Then T Y_1^T = -U S and (I - Y T Y^T)[:, :h] = Q S.
// Multiplying the rows of the tree's R by S preserves the panel factorization.
// This procedure operates on Q, never on the input Gram matrix or A - R.
#include "tree_update.cuh"

namespace tqr {
constexpr int reconstruction_width = 64;

// The correctly rounded reciprocal: the value of 1 / x, without the general division.
__device__ __forceinline__ double exact_reciprocal(double x) {
    return __drcp_rn(x);
}
__device__ __forceinline__ float exact_reciprocal(float x) {
    return __frcp_rn(x);
}
// Modified LU of the h x h block at q (leading dimension ldq, global or shared memory), held in
// registers by 256 threads, blocked by warps: warp w owns the eight consecutive columns
// [8w, 8w + 8) and each lane two rows of each. The owner of a block factors it with shuffles only,
// choosing every sign and pivot as the unblocked algorithm does, and publishes its L columns
// through alternating shared buffers (lblock, 2 x 8 x B); one barrier per block lets the later
// warps apply them to their columns (forming U12 and updating the trailing rows). Eight barriers
// instead of 64. lu receives the factors (leading dimension B) and then S.
template <class T>
__device__ void modified_lu(const T *q, int ldq, int h, T *lu, int *status, T *lblock, T *signs) {
    constexpr int B = reconstruction_width, W = 8;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    T x[W][2];
#pragma unroll
    for (int c = 0; c < W; ++c) {
        const int j = warp * W + c;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int i = lane + r * 32;
            x[c][r] = i < h && j < h ? q[i + size_t(j) * ldq] : T(0);
        }
    }
    for (int b = 0; b * W < h; ++b) {
        T *lb = lblock + (b & 1) * W * B;
        if (warp == b) {
#pragma unroll
            for (int kk = 0; kk < W; ++kk) {
                const int k = b * W + kk;
                if (k < h) {
                    const T alpha = __shfl_sync(0xffffffff, k < 32 ? x[kk][0] : x[kk][1], k % 32);
                    const T sign = alpha >= T(0) ? T(-1) : T(1);
                    const T pivot = alpha - sign, inverse = exact_reciprocal(pivot);
                    if (lane == 0) {
                        signs[k] = sign;
                        if (!isfinite(alpha))
                            atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
                    }
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int i = lane + r * 32;
                        if (i == k)
                            x[kk][r] = pivot;
                        else if (i > k)
                            x[kk][r] *= inverse;
                    }
#pragma unroll
                    for (int c = kk + 1; c < W; ++c) {
                        const T u = __shfl_sync(0xffffffff, k < 32 ? x[c][0] : x[c][1], k % 32);
#pragma unroll
                        for (int r = 0; r < 2; ++r)
                            if (lane + r * 32 > k)
                                x[c][r] -= x[kk][r] * u;
                    }
                }
            }
#pragma unroll
            for (int kk = 0; kk < W; ++kk)
#pragma unroll
                for (int r = 0; r < 2; ++r)
                    lb[kk * B + lane + r * 32] = x[kk][r];
        }
        __syncthreads();
        if (warp > b) {
#pragma unroll
            for (int kk = 0; kk < W; ++kk) {
                const int k = b * W + kk;
                if (k < h) {
                    const T l0 = lb[kk * B + lane], l1 = lb[kk * B + lane + 32];
#pragma unroll
                    for (int c = 0; c < W; ++c) {
                        const T u = __shfl_sync(0xffffffff, k < 32 ? x[c][0] : x[c][1], k % 32);
                        if (lane > k)
                            x[c][0] -= l0 * u;
                        if (lane + 32 > k)
                            x[c][1] -= l1 * u;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < h)
        lu[B * B + threadIdx.x] = signs[threadIdx.x];
#pragma unroll
    for (int c = 0; c < W; ++c)
#pragma unroll
        for (int r = 0; r < 2; ++r)
            lu[lane + r * 32 + (warp * W + c) * B] = x[c][r];
}
// Y_1 (the unit lower factor) in q and below R in a, and R's rows scaled by S, from the modified LU
// in lu (written by this block).
template <class T>
__device__ void modified_lu_finish(T *q, int ldq, T *a, int lda, int h, int columns, const T *lu) {
    constexpr int B = reconstruction_width;
    for (int e = threadIdx.x; e < B * B; e += blockDim.x) {
        const int i = e % B, j = e / B;
        if (i < h && j < h) {
            const T v = i < j ? T(0) : i == j ? T(1) : lu[i + j * B];
            q[i + size_t(j) * ldq] = v;
            if (i > j)
                a[i + size_t(j) * lda] = v;
        }
    }
    for (int e = threadIdx.x; e < h * columns; e += blockDim.x) {
        const int i = e % h, j = e / h;
        if (i <= j)
            a[i + size_t(j) * lda] *= lu[B * B + i];
    }
}
template <class T>
__global__ __launch_bounds__(256) void reconstruct_top_kernel(T *q, int ldq, T *a, int lda, int h,
                                                              int columns, T *lu, int *status) {
    __shared__ T lblock[2 * 8 * reconstruction_width], signs[reconstruction_width];
    modified_lu(q, ldq, h, lu, status, lblock, signs);
    __syncthreads();
    modified_lu_finish(q, ldq, a, lda, h, columns, lu);
}

// Row `row` of T by one warp, from T Y_1^T = -U S: y holds the modified LU (leading dimension B)
// and sign its signs (zero past h), in shared memory. Right-looking substitution keeps the row in
// two registers per lane; shuffles broadcast the next solution. The rows are independent, so they
// spread over warps (and SMs) instead of a serial dot product per entry.
template <class T>
__device__ void reconstruct_t_row(const T *y, const T *sign, int h, int row, T *t, int ldt) {
    constexpr int B = reconstruction_width;
    const int lane = threadIdx.x % 32;
    T lo = lane >= row && lane < h ? -y[row + lane * B] * sign[lane] : T(0);
    T hi = lane + 32 >= row && lane + 32 < h ? -y[row + (lane + 32) * B] * sign[lane + 32] : T(0);
    for (int j = row; j < h; ++j) {
        const T pivot = __shfl_sync(0xffffffff, j < 32 ? lo : hi, j % 32);
        if (lane > j && lane < h)
            lo -= pivot * y[lane + j * B];
        if (lane + 32 > j && lane + 32 < h)
            hi -= pivot * y[lane + 32 + j * B];
    }
    if (lane < h)
        t[row + size_t(lane) * ldt] = lo;
    if (lane + 32 < h)
        t[row + size_t(lane + 32) * ldt] = hi;
}
template <class T>
__global__ __launch_bounds__(32) void reconstruct_t_kernel(const T *lu, int h, T *t, int ldt) {
    constexpr int B = reconstruction_width;
    __shared__ T y[B * B], sign[B];
    const int lane = threadIdx.x;
    for (int e = lane; e < B * B; e += 32)
        y[e] = lu[e];
    sign[lane] = lane < h ? lu[B * B + lane] : T(0);
    sign[lane + 32] = lane + 32 < h ? lu[B * B + lane + 32] : T(0);
    __syncwarp();
    reconstruct_t_row(y, sign, h, blockIdx.x, t, ldt);
}

// U and the reciprocals of its diagonal from the modified LU in lu.
template <class T> __device__ void load_u(const T *lu, int h, T *u, T *dinv) {
    constexpr int B = reconstruction_width;
    for (int e = threadIdx.x; e < B * B; e += blockDim.x)
        u[e] = __ldcg(lu + e);
    for (int e = threadIdx.x; e < B; e += blockDim.x)
        dinv[e] = e < h ? T(1) / __ldcg(lu + e * (B + 1)) : T(0);
}
// Independent rows [lo, hi) of Y U = Q, in place in q and in a: a row per thread, eight columns at a
// time in registers, the thread's previous solutions in ys (blockDim.x x B), U (leading dimension B)
// and the reciprocals of its diagonal in shared memory. No unbounded replication is hidden in the
// solve: each thread owns a complete row contraction.
template <class T>
__device__ void solve_rows(T *q, long long ldq, T *a, long long lda, int lo, int hi, int h,
                           const T *u, const T *dinv, T *ys) {
    constexpr int B = reconstruction_width, K = 8;
    const int R = blockDim.x;
    for (int r = lo + int(threadIdx.x); r < hi; r += R)
        for (int j0 = 0; j0 < h; j0 += K) {
            T x[K];
#pragma unroll
            for (int j = 0; j < K; ++j)
                x[j] = j0 + j < h ? q[r + (j0 + j) * ldq] : T(0);
            for (int k = 0; k < j0; ++k) {
                const T previous = ys[threadIdx.x + k * R];
#pragma unroll
                for (int j = 0; j < K; ++j)
                    x[j] -= previous * u[k + (j0 + j) * B];
            }
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if (j0 + k < h) {
                    x[k] *= dinv[j0 + k];
#pragma unroll
                    for (int j = k + 1; j < K; ++j)
                        x[j] -= x[k] * u[j0 + k + (j0 + j) * B];
                }
            }
#pragma unroll
            for (int j = 0; j < K; ++j)
                if (j0 + j < h) {
                    ys[threadIdx.x + (j0 + j) * R] = x[j];
                    q[r + (j0 + j) * ldq] = x[j];
                    a[r + (j0 + j) * lda] = x[j];
                }
        }
}
template <class T>
__global__ __launch_bounds__(128) void reconstruct_tail_kernel(T *q, int ldq, T *a, int lda,
                                                               int rows, int h, const T *lu,
                                                               int first) {
    constexpr int B = reconstruction_width;
    extern __shared__ __align__(16) unsigned char raw[];
    T *u = reinterpret_cast<T *>(raw), *dinv = u + B * B, *ys = dinv + B;
    load_u(lu, h, u, dinv);
    __syncthreads();
    const int lo = first + blockIdx.x * blockDim.x;
    solve_rows(q, ldq, a, lda, lo, min(rows, lo + int(blockDim.x)), h, u, dinv, ys);
}

// Y = Q U^-1 for rows [first, rows) of q, with U from the modified LU in lu: in place in q and in
// a. A GPU that holds the rows of a panel below another GPU's diagonal block solves all of them.
template <class T>
void reconstruct_rows(T *q, int ldq, T *a, int lda, int rows, int h, const T *lu, int first,
                      cudaStream_t stream) {
    if (rows <= first)
        return;
    constexpr int B = reconstruction_width;
    const size_t bytes = (B * B + B + 128 * B) * sizeof(T);
    reserve_shared_memory(reconstruct_tail_kernel<T>, bytes);
    reconstruct_tail_kernel<T>
        <<<ceildiv(rows - first, 128), 128, bytes, stream>>>(q, ldq, a, lda, rows, h, lu, first);
    CU(cudaGetLastError());
}
// Q contains an explicitly formed thin orthonormal matrix, in native precision.
// A contains the HQR panel; only its upper triangle is read. Workspace has
// reconstruction_width * (reconstruction_width + 1) entries and retains S for
// diagnostics. On return Q holds Y, A holds Y below R, and T is upper triangular.
template <class T>
void reconstruct_householder(T *q, int ldq, T *a, int lda, int rows, int h, int columns, T *t,
                             int ldt, T *workspace, int *status, cudaStream_t stream) {
    if (h < 1 || h > reconstruction_width || rows < h || columns < h || ldq < rows || lda < rows ||
        ldt < h)
        throw std::runtime_error("invalid Householder reconstruction shape");
    reconstruct_top_kernel<T><<<1, 256, 0, stream>>>(q, ldq, a, lda, h, columns, workspace, status);
    reconstruct_t_kernel<T><<<h, 32, 0, stream>>>(workspace, h, t, ldt);
    reconstruct_rows(q, ldq, a, lda, rows, h, workspace, h, stream);
}

// ---- the thin Q of a tree and its reconstruction in one launch ----------------------------------
// Blocks of B x B in shared memory (leading dimension B + 1, zero-padded): E, an operand, W, Z and
// a chunk of V for the thin Q; later U, its diagonal's reciprocals, S and the solve's rows.
template <class T> struct ThinQ {
    static constexpr int B = reconstruction_width, LD = B + 1, threads = 256;
    static constexpr int per_thread = B * B / threads;
    static constexpr size_t block = size_t(B) * LD;
    static constexpr size_t words =
        std::max(5 * block, size_t(B) * B + 2 * B + size_t(threads) * B);
    static constexpr size_t smem = words * sizeof(T);
};
enum class Shape { Full, Upper, UnitLower, UnitLowerT };
// Rows [r0, r0 + B) of a block at x (leading dimension ldx; `rows` x `cols`, zero beyond) into s,
// shaped: Full, Upper (row <= column), UnitLower (unit diagonal, zero above) or UnitLowerT (the
// transpose of the B x B top of a UnitLower block). All loads are in flight before the first
// store; `l2` loads what other blocks of the launch wrote.
template <class T>
__device__ void thin_q_stage(const T *x, long long ldx, int r0, int rows, int cols, Shape shape,
                             bool l2, T *s) {
    using C = ThinQ<T>;
    const bool transposed = shape == Shape::UnitLowerT,
               unit = shape == Shape::UnitLower || shape == Shape::UnitLowerT;
    T v[C::per_thread];
#pragma unroll
    for (int it = 0; it < C::per_thread; ++it) {
        const int e = threadIdx.x + it * C::threads;
        const int i = transposed ? e / C::B : e % C::B, j = transposed ? e % C::B : e / C::B;
        const int row = transposed ? j : r0 + i, col = transposed ? i : j;
        v[it] = T(0);
        if (row < rows && col < cols) {
            if (shape == Shape::Full || (shape == Shape::Upper && row <= col) ||
                (unit && row > col))
                v[it] = l2 ? __ldcg(x + row + col * ldx) : x[row + col * ldx];
            else if (unit && row == col)
                v[it] = T(1);
        }
    }
#pragma unroll
    for (int it = 0; it < C::per_thread; ++it) {
        const int e = threadIdx.x + it * C::threads;
        const int i = transposed ? e / C::B : e % C::B, j = transposed ? e % C::B : e / C::B;
        s[i + j * C::LD] = v[it];
    }
}
// A thread's 4 x 4 tile of a B x B block: rows t % 16 + 16 i, columns 4 (t / 16) + j.
template <class T> using ThinQTile = T[4][4];
__device__ __forceinline__ int thin_q_row(int i) {
    return int(threadIdx.x % 16) + 16 * i;
}
__device__ __forceinline__ int thin_q_col(int j) {
    return int(threadIdx.x / 16) * 4 + j;
}
// c = A X for B x B blocks in shared memory (A(i, k) at a[i + k LD]); per k, eight shared loads
// feed sixteen FMAs.
template <class T> __device__ void thin_q_product(const T *a, const T *x, ThinQTile<T> &c) {
    using C = ThinQ<T>;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = T(0);
#pragma unroll 4
    for (int k = 0; k < C::B; ++k) {
        T av[4], xv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            av[i] = a[thin_q_row(i) + k * C::LD];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            xv[j] = x[k + thin_q_col(j) * C::LD];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                c[i][j] = fma(av[i], xv[j], c[i][j]);
    }
}
// Tiles to and from shared memory (leading dimension LD) or global memory (leading dimension ld,
// rows < rows and columns < cols).
template <class T> __device__ void thin_q_put(const ThinQTile<T> &c, T *s) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            s[thin_q_row(i) + thin_q_col(j) * ThinQ<T>::LD] = c[i][j];
}
template <class T> __device__ void thin_q_get(const T *s, ThinQTile<T> &c) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = s[thin_q_row(i) + thin_q_col(j) * ThinQ<T>::LD];
}
template <class T>
__device__ void thin_q_store(const ThinQTile<T> &c, int rows, int cols, T *x, long long ld) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (thin_q_row(i) < rows && thin_q_col(j) < cols)
                x[thin_q_row(i) + thin_q_col(j) * ld] = c[i][j];
}
// Publication of a block's global writes to the other blocks of the launch.
__device__ __forceinline__ void thin_q_signal(int *flag, int epoch) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        gau_store_release(flag, epoch);
}
__device__ __forceinline__ void thin_q_wait(const int *flag, int epoch) {
    if (threadIdx.x == 0)
        ge_wait(flag, epoch);
    __syncthreads();
}
// Rows [r0, r0 + B) of x -= V Z, V (leading dimension ldv, `rows` rows) shaped and staged in vs, Z
// in zs; E (the thread's tile e) is kept on the rows below `keep`.
template <class T>
__device__ void thin_q_chunk(const T *v, long long ldv, Shape shape, int r0, int rows, int h, int q,
                             const T *zs, const ThinQTile<T> &e, int keep, T *x, long long ldx,
                             T *vs) {
    thin_q_stage(v, ldv, r0, rows, h, shape, false, vs);
    __syncthreads();
    ThinQTile<T> c;
    thin_q_product(vs, zs, c);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            c[i][j] = (r0 + thin_q_row(i) < keep ? e[i][j] : T(0)) - c[i][j];
    thin_q_store(c, rows - r0, q, x + r0, ldx);
    __syncthreads();
}
// An element copied to shared memory asynchronously (cp.async), or `fill` written there.
template <class T>
__device__ __forceinline__ void thin_q_async(T *dst, const T *src, bool copy, T fill) {
    if (copy)
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(
                         unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "n"(sizeof(T))
                     : "memory");
    else
        *dst = fill;
}

// The thin Q of a panel's tree t (q <= 64 columns) and, with `solve`, its Householder
// reconstruction, in one launch. The tree's Q applied to [E; 0] (E the h x q block on the first
// triangle, already in q: I on one GPU, the merge's block on a member of several) only fills what
// the steps above have filled: at a TT node with Z = T E, E on its first child's triangle becomes
// E - Z and the zeros on child j's triangle -V_j Z; a TS step does the same for its tile; a
// domain's GE turns [E; 0] into [E; 0] - V T V_1^T E. A block per domain, admitted in ticket order,
// follows the first-child path of the nodes whose first domain it is: it enters at -V_j Z of its
// parent (published in zs_out, one block per node), publishes the Z of each node, keeps E in
// registers and writes its domain's rows of Q to q. The first domain's path, root to leaf, holds
// the critical chain: one product per level, then its first B rows of Q, the modified LU of
// Q_top - S = Y U, and U for every block; with `solve` the blocks form rows of T and their rows of
// Y = Q U^-1, in q and a. Waits only target blocks with earlier tickets.
template <class T>
__global__ __launch_bounds__(256, 1) void thin_q_reconstruct_kernel(
    DomainTree t, int qc, const T *vp, int ldp, const T *vstack, const T *tt, const T *slot, int sb,
    bool owner, int height, T *q, int ldq, T *a, int lda, T *zs_out, T *lu, T *tout, int ldt,
    int tb, int *status, int *flags, unsigned *ticket, unsigned base, int epoch, bool solve) {
    using C = ThinQ<T>;
    extern __shared__ __align__(16) unsigned char thin_q_storage[];
    T *es = reinterpret_cast<T *>(thin_q_storage), *as = es + C::block, *ws = as + C::block,
      *zs = ws + C::block, *vs = zs + C::block;
    __shared__ int ticket_s;
    if (threadIdx.x == 0)
        ticket_s = int(atomicAdd(ticket, 1u) - base);
    __syncthreads();
    const int h = t.h, D = t.count[0], F = t.fan_in;
    const long long sq = (long long)h * h;
    int *u_ready = flags + t.nodes(), *top_ready = u_ready + 1;
    if (solve && ticket_s == 1) {
        // The block of the modified LU, admitted right after the first domain's: Q_top - S = Y U
        // as soon as the first domain has formed its rows, U for every block, then Y_1, S on R
        // and T's padding past qc.
        thin_q_wait(top_ready, epoch);
        thin_q_stage(q, ldq, 0, qc, qc, Shape::Full, true, ws);
        __syncthreads();
        modified_lu(ws, C::LD, qc, lu, status, es, zs);
        thin_q_signal(u_ready, epoch);
        modified_lu_finish(q, ldq, a, lda, qc, qc, lu);
        for (int x = threadIdx.x; x < tb * tb; x += C::threads)
            if (x % tb >= qc || x / tb >= qc)
                tout[x % tb + size_t(x / tb) * ldt] = T(0);
        return;
    }
    const int d = solve && ticket_s > 1 ? ticket_s - 1 : ticket_s;
    // The GE's V_1^T (unit lower, transposed) into vs and its T (upper) into zs, asynchronously:
    // they do not depend on the levels above.
    const T *v = vp + (long long)d * t.leaf_stride();
    for (int x = threadIdx.x; x < C::B * C::B; x += C::threads) {
        const int i = x / C::B, k = x % C::B;
        thin_q_async(vs + i + k * C::LD, v + k + size_t(i) * ldp, i < h && k < h && k > i,
                     i < h && k == i ? T(1) : T(0));
        const int r = x % C::B, col = x / C::B;
        thin_q_async(zs + r + col * C::LD, tt + d * sq + r + size_t(col) * h,
                     r < h && col < h && r <= col, T(0));
    }
    tu_commit();
    // the highest level of the nodes whose first domain is d (the root's for d = 0)
    int top = 0;
    long long span = 1;
    while (top + 1 < t.levels && d % (span * F) == 0) {
        span *= F;
        ++top;
    }
    ThinQTile<T> e, c;
    if (top + 1 == t.levels) {
        // E: I in its first `height` rows on one GPU or on the owner of a merge across GPUs,
        // less V_i T of the merge (slot: its b x b block of V then T, the identity on the owner)
        for (int x = threadIdx.x; x < C::B * C::B; x += C::threads) {
            const int i = x % C::B, j = x / C::B;
            T val = T(0);
            if (i < height && i < t.rows && j < qc) {
                if (!slot)
                    val = i == j ? T(1) : T(0);
                else {
                    const T *tm = slot + size_t(sb) * sb;
                    T sum = T(0);
                    for (int k = 0; k <= j; ++k)
                        sum = fma(slot[i + size_t(k) * sb], tm[k + size_t(j) * sb], sum);
                    val = (owner && i == j ? T(1) : T(0)) - sum;
                }
            }
            es[i + j * C::LD] = val;
        }
        __syncthreads();
        thin_q_get(es, e);
    } else {
        // child j of node n of level top + 1: E = -V_j Z_n
        const int l = top + 1, n = int(d / (span * F)), j = int(d / span % F);
        thin_q_stage(vstack + t.vstack_offset(l) + (long long)n * F * h + j * h, t.vstack_rows(l),
                     0, h, h, Shape::Upper, false, as);
        thin_q_wait(flags + t.first[l] + n, epoch);
        thin_q_stage(zs_out + size_t(t.first[l] + n) * C::B * C::B, C::B, 0, C::B, C::B,
                     Shape::Full, true, ws);
        __syncthreads();
        thin_q_product(as, ws, c);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int jj = 0; jj < 4; ++jj)
                e[i][jj] = -c[i][jj];
        __syncthreads();
    }
    for (int l = top; l >= 1; --l, span /= F) {
        const int n = int(d / span), node = t.first[l] + n;
        thin_q_put(e, es);
        thin_q_stage(tt + node * sq, h, 0, h, h, Shape::Upper, false, as);
        __syncthreads();
        thin_q_product(as, es, c);
        thin_q_store(c, C::B, C::B, zs_out + size_t(node) * C::B * C::B, C::B);
        thin_q_signal(flags + node, epoch);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                e[i][j] -= c[i][j];
    }
    T *xd = q + (long long)d * t.domain;
    for (int p = t.chain - 1; p >= 1; --p) {
        const int tail = t.ts_rows(p, d);
        if (!tail)
            continue;
        thin_q_put(e, es);
        thin_q_stage(tt + ((long long)p * D + d) * sq, h, 0, h, h, Shape::Upper, false, as);
        __syncthreads();
        thin_q_product(as, es, c);
        thin_q_put(c, ws);
        __syncthreads();
        const ThinQTile<T> none = {};
        for (int r0 = 0; r0 < tail; r0 += C::B)
            thin_q_chunk(vstack + t.ts_offset(p) + (long long)d * DomainConfig::rows + h,
                         t.ts_vrows(), Shape::Full, r0, tail, h, qc, ws, none, 0,
                         xd + (long long)p * t.tile, ldq, as);
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                e[i][j] -= c[i][j];
    }
    // the GE's V_1^T and T, staged while the levels above are formed
    tu_wait<0>();
    __syncthreads();
    thin_q_put(e, es);
    __syncthreads();
    thin_q_product(vs, es, c);
    thin_q_put(c, ws);
    __syncthreads();
    thin_q_product(zs, ws, c);
    thin_q_put(c, as);
    __syncthreads();
    const int rows = t.ge_rows(d);
    thin_q_chunk(v, ldp, Shape::UnitLower, 0, rows, h, qc, as, e, h, xd, ldq, vs);
    if (solve && d == 0)
        thin_q_signal(top_ready, epoch);
    for (int r0 = C::B; r0 < rows; r0 += C::B)
        thin_q_chunk(v, ldp, Shape::UnitLower, r0, rows, h, qc, as, e, h, xd, ldq, vs);
    if (!solve)
        return;
    thin_q_wait(u_ready, epoch);
    T *u = es, *dinv = u + C::B * C::B, *sign = dinv + C::B, *ys = sign + C::B;
    load_u(lu, qc, u, dinv);
    for (int i = threadIdx.x; i < C::B; i += C::threads)
        sign[i] = i < qc ? __ldcg(lu + C::B * C::B + i) : T(0);
    __syncthreads();
    const int warps = C::threads / 32;
    for (int row = d * warps + int(threadIdx.x) / 32; row < qc; row += D * warps)
        reconstruct_t_row(u, sign, qc, row, tout, ldt);
    solve_rows(xd, ldq, a + (long long)d * t.domain, lda, d == 0 ? qc : 0, t.domain_rows(d), qc, u,
               dinv, ys);
}
// Flops of the thin Q of tree t with q columns.
inline double thin_q_flops(const DomainTree &t, int q) {
    const double hh = double(t.h) * t.h * q;
    double f = 0;
    for (int l = 1; l < t.levels; ++l)
        for (int n = 0; n < t.count[l]; ++n)
            f += hh * t.children(l, n);
    for (int i = 0; i < t.count[0]; ++i) {
        for (int p = 1; p < t.chain; ++p)
            if (t.ts_rows(p, i))
                f += hh + 2.0 * t.ts_rows(p, i) * t.h * q;
        f += 2 * hh + 2.0 * t.ge_rows(i) * t.h * q;
    }
    return f;
}
// The thin Q of tree t (q columns) in q (leading dimension ldq) from the tree's GE reflectors vp,
// stacked V and T factors, starting from [E; 0] (E: see thin_q_reconstruct_kernel; slot null on
// one GPU); with `solve` also its reconstruction: Y in q and in a (leading dimension lda), U and S
// in lu, T in tout (tb x tb, leading dimension ldt). zs holds a B x B block per node.
template <class T>
void thin_q_reconstruct(const DomainTree &t, int qc, const T *vp, int ldp, const T *vstack,
                        const T *tt, const T *slot, int sb, bool owner, int height, T *q, int ldq,
                        T *a, int lda, T *zs, T *lu, T *tout, int tb, int *status, TreeSync &sync,
                        bool solve, cudaStream_t stream) {
    if (t.h > reconstruction_width || qc < 1 || qc > reconstruction_width ||
        (solve && t.ge_rows(0) < qc))
        throw std::runtime_error("invalid thin Q reconstruction shape");
    if (sync.flags.n < size_t(t.nodes()) + 2)
        throw std::runtime_error("a thin Q beyond its reserved flags");
    ++sync.epoch;
    const int blocks = t.count[0] + (solve ? 1 : 0);
    reserve_shared_memory(thin_q_reconstruct_kernel<T>, ThinQ<T>::smem);
    thin_q_reconstruct_kernel<T><<<blocks, ThinQ<T>::threads, ThinQ<T>::smem, stream>>>(
        t, qc, vp, ldp, vstack, tt, slot, sb, owner, height, q, ldq, a, lda, zs, lu, tout, tb, tb,
        status, sync.flags.p, sync.ticket.p, sync.base, sync.epoch, solve);
    sync.base += unsigned(blocks);
    CU(cudaGetLastError());
}
} // namespace tqr
