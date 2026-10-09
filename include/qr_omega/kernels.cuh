#pragma once
// Small device kernels: reductions, column scaling, packing of reflectors, composition of T
// factors, the copies of a TT merge, and the input and error kernels shared with the reference
// adapters.
#include "common.hpp"
namespace qr_omega {
// What a factorization reports: success, a non-finite input, or a result FP32 cannot represent.
enum DeviceStatus { OK = 0, NONFINITE_INPUT = 1, UNREPRESENTABLE_RESULT = 2 };

// Rows [row, row + rows) of one GPU below the diagonal of a panel, reduced by h reflectors whose T
// factor is `tile`.
struct TilePacket {
    int row, col, rows, h, tile;
};

// Launch order of the output tiles of a product that keeps the operands of one wave in L2: groups
// of G column tiles, all row tiles of a group before the next. L is the linear block index of an
// nx x ny grid of tiles.
__device__ __forceinline__ void raster_tile(int L, int nx, int ny, int G, int &tx, int &ty) {
    const int span = G * ny, group = L / span, in = L % span, width = min(G, nx - group * G);
    tx = group * G + in % width;
    ty = in / width;
}
template <class T> __device__ T absval(T x) {
    return x < T(0) ? -x : x;
}
// Block reductions through s; the leading barrier lets a block reuse s for consecutive reductions.
template <class T> __device__ T block_sum(T x, T *s) {
    int t = threadIdx.x;
    __syncthreads();
    s[t] = x;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d /= 2) {
        if (t < d)
            s[t] += s[t + d];
        __syncthreads();
    }
    return s[0];
}
template <class T> __device__ T block_max(T x, T *s) {
    int t = threadIdx.x;
    __syncthreads();
    s[t] = x;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d /= 2) {
        if (t < d)
            s[t] = max(s[t], s[t + d]);
        __syncthreads();
    }
    return s[0];
}
template <class T> __device__ T warp_add(T x) {
    for (int d = 16; d; d /= 2)
        x += __shfl_down_sync(0xffffffff, x, d);
    return __shfl_sync(0xffffffff, x, 0);
}
template <class T> __device__ T warp_maximum(T x) {
    for (int d = 16; d; d /= 2)
        x = max(x, __shfl_down_sync(0xffffffff, x, d));
    return __shfl_sync(0xffffffff, x, 0);
}
template <class T, bool Maximum> __device__ T tile_reduce(T x, T *scratch) {
    if (blockDim.x == 32)
        return Maximum ? warp_maximum(x) : warp_add(x);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    x = Maximum ? warp_maximum(x) : warp_add(x);
    if (lane == 0)
        scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        x = lane < warps ? scratch[lane] : T(0);
        x = Maximum ? warp_maximum(x) : warp_add(x);
        if (lane == 0)
            scratch[0] = x;
    }
    __syncthreads();
    return scratch[0];
}
// Every column is scaled to unit maximum before the factorization so that FP32 cannot overflow, and
// R is scaled back afterwards: A D^-1 = Q R' gives A = Q (R' D), and V and T never change.
template <class T> __global__ void tiled_column_max(const T *a, int nr, int n, int ld, T *scale) {
    __shared__ T red[256];
    int col = blockIdx.x;
    T mx = 0;
    for (int row = threadIdx.x; row < nr; row += blockDim.x)
        mx = max(mx, absval(a[row + size_t(col) * ld]));
    mx = block_max(mx, red);
    if (threadIdx.x == 0)
        scale[col] = mx;
}
template <class T> __global__ void tiled_scale_max(T *scale, const T *other, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        scale[i] = max(scale[i], other[i]);
}
// Divides (or, restoring, multiplies) the columns by their scales. UpperOnly visits only the rows
// of R (global row <= global column, with the local rows and columns of this GPU distributed by
// `rows` and `cols` at grid position (r, c)), CheckAll also checks the other elements for
// finiteness.
template <class T, bool Restore, bool UpperOnly = false, bool CheckAll = false>
__global__ void tiled_column_scale(T *a, int nr, int n, int ld, const T *scale, int *status,
                                   Cyclic rows = Cyclic{}, int r = 0, Cyclic cols = Cyclic{},
                                   int c = 0) {
    bool bad = false;
    const int stride = blockDim.x * gridDim.x, r0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int col = blockIdx.y; col < n; col += gridDim.y) {
        T *v = a + size_t(col) * ld;
        const T s = scale[col];
        const long long gcol = cols.global(c, col);
        for (int row = r0; row < nr; row += stride) {
            if constexpr (UpperOnly) {
                if (rows.global(r, row) > gcol) {
                    if constexpr (CheckAll)
                        bad |= !isfinite(v[row]);
                    continue;
                }
            }
            const T x = v[row];
            T y;
            if constexpr (Restore)
                y = x * s;
            else
                y = s ? x / s : T(0);
            v[row] = y;
            bad |= !isfinite(y);
        }
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0)
        atomicCAS(status, 0, UNREPRESENTABLE_RESULT);
}
// Rows over up to 8 blocks of 256 threads, columns over the second grid dimension.
inline dim3 column_pass_grid(int nr, int n) {
    return dim3(std::max(1, std::min(8, ceildiv(std::max(nr, 1), 1024))),
                std::max(1, std::min(n, 8192)));
}
template <class T>
__global__ void finite_scan_2d(const T *a, int rows, int n, int ld, int *status) {
    bool bad = false;
    const int stride = blockDim.x * gridDim.x, r0 = blockIdx.x * blockDim.x + threadIdx.x;
    for (int col = blockIdx.y; col < n; col += gridDim.y) {
        const T *c = a + size_t(col) * ld;
        for (int row = r0; row < rows; row += stride)
            bad |= !isfinite(c[row]);
    }
    if (__syncthreads_or(bad) && threadIdx.x == 0)
        atomicCAS(status, 0, NONFINITE_INPUT);
}
// The reflectors of a packet with their unit diagonal and the zeros above it, leading dimension
// `leaf`.
template <class T>
__global__ void pack_ge_v(const T *A, int ld, const TilePacket *packet, T *V, int leaf, int b) {
    const TilePacket p = *packet;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < leaf * b; i += blockDim.x * gridDim.x) {
        int r = i % leaf, j = i / leaf;
        V[i] = (r >= p.rows || j >= p.h || r < j) ? T(0)
               : r == j                           ? T(1)
                                                  : A[p.row + r + size_t(p.col + j) * ld];
    }
}
// All the rows of a panel's reflectors on a GPU whose rows lie below the panel's diagonal block.
template <class T>
__global__ void pack_rows(const T *A, int ld, const TilePacket *packet, T *V, int leaf, int h) {
    const TilePacket p = *packet;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < leaf * h; i += blockDim.x * gridDim.x) {
        const int r = i % leaf, j = i / leaf;
        V[i] = r < p.rows ? A[p.row + r + size_t(p.col + j) * ld] : T(0);
    }
}
// Enough 128-thread blocks for about four elements per thread, between 32 and 2048.
inline int pack_v_grid(int leaf, int w) {
    const long long e = (long long)leaf * std::max(1, w);
    return int(std::max(32LL, std::min(2048LL, (e + 511) / 512)));
}
// The same for the panel columns [j0, j1), packed as soon as a window of the panel is factored.
template <class T>
__global__ void pack_ge_v_cols(const T *A, int ld, const TilePacket *packet, T *V, int leaf, int j0,
                               int j1) {
    const TilePacket p = *packet;
    const int w = j1 - j0;
    if (w <= 0)
        return;
    // One item is a column and a chunk of rows: a single 32-bit division per item.
    const int CH = int(blockDim.x) * 8, rch = (leaf + CH - 1) / CH;
    for (int it = blockIdx.x; it < w * rch; it += gridDim.x) {
        const int jj = it / rch, rc = it - jj * rch, j = j0 + jj, r1 = min(leaf, (rc + 1) * CH);
        const T *src = A + p.row + size_t(p.col + j) * ld;
        T *dst = V + size_t(j) * leaf;
        for (int r = rc * CH + int(threadIdx.x); r < r1; r += blockDim.x)
            dst[r] = (r >= p.rows || j >= p.h || r < j) ? T(0) : r == j ? T(1) : src[r];
    }
}
// T12 = -T1 (V1^T V2) T2 joins the T of the windows already factored to the T of the window just
// factored, from G = V1^T V2. One block per column; S = G T2 stays in shared memory.
template <class T> __global__ void tile_compose_offdiag(T *tri, int b, const T *G, int w0, int sw) {
    const int c = blockIdx.x;
    if (c >= sw)
        return;
    extern __shared__ __align__(16) unsigned char raw[];
    T *S = reinterpret_cast<T *>(raw);
    for (int i = threadIdx.x; i < w0; i += blockDim.x) {
        T acc = 0;
        for (int j = 0; j <= c; ++j)
            acc += G[size_t(i) + size_t(j) * w0] * tri[w0 + j + size_t(w0 + c) * b];
        S[i] = acc;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < w0; i += blockDim.x) {
        T acc = 0;
        for (int j = i; j < w0; ++j)
            acc += tri[i + size_t(j) * b] * S[j];
        tri[i + size_t(w0 + c) * b] = -acc;
    }
}
// Append a reconstructed panel after its row broadcast. Every column peer owns
// the same group factors, including panels whose matrix columns live elsewhere.
// The panel's b columns go to group columns [column, column + b), its rows below
// the first `offset` rows of the group, which are zero.
template <class T>
__global__ void append_group_panel(const T *v, int ldv, const T *t, int b, int rows, int offset,
                                   int column, T *vg, int ldvg, T *diagonal) {
    const int group_rows = rows + offset;
    for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < size_t(group_rows) * b;
         e += size_t(gridDim.x) * blockDim.x) {
        const int r = int(e % group_rows), j = int(e / group_rows);
        vg[r + size_t(column + j) * ldvg] = r < offset ? T(0) : v[r - offset + size_t(j) * ldv];
    }
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < b * b; e += gridDim.x * blockDim.x)
        diagonal[e] = t[e];
}
// V_g of a group of panels in the reflector order `perm`: panel j is zero above its own diagonal.
template <class T>
__global__ void pack_v_block_perm(const T *A, int ld, int row, int col, int rows, int H,
                                  const int *perm, T *V, int ldv) {
    // One item is a column and a chunk of rows: a single order lookup and division per item.
    const int CH = int(blockDim.x) * 8, rch = (rows + CH - 1) / CH;
    for (int it = blockIdx.x; it < H * rch; it += gridDim.x) {
        const int p = it / rch, rc = it - p * rch, j = perm[p], r1 = min(rows, (rc + 1) * CH);
        const T *src = A + row + size_t(col + j) * ld;
        T *dst = V + size_t(p) * ldv;
        for (int r = rc * CH + int(threadIdx.x); r < r1; r += blockDim.x)
            dst[r] = r < j ? T(0) : r == j ? T(1) : src[r];
    }
}
// The Gram blocks G_ij = V_i^T V_j (i < j) of panel j in the natural order of the group, from the
// Gram Gp (H x H) of its reflectors in the group order: block i at G + i b b, leading dimension b.
template <class T>
__global__ void gather_gram_blocks(const T *Gp, int H, const int *inverse, int j, int b, T *G) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < j * b * b;
         e += gridDim.x * blockDim.x) {
        const int i = e / (b * b), r = e % b, c = (e / b) % b;
        G[e] = Gp[inverse[i * b + r] + size_t(inverse[j * b + c]) * H];
    }
}
// T_g of the group in the same order.
template <class T> __global__ void permute_t(const T *Tn, int H, const int *perm, T *Tp) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H * H; i += blockDim.x * gridDim.x) {
        const int r = i % H, c = i / H;
        Tp[i] = Tn[perm[r] + size_t(perm[c]) * H];
    }
}
// Appends panel j to the T_g of the group: T_g(0:jb, jb:jb + b) = -T_prev G_j T_j, with the Gram
// blocks G_j = V_prev^T V_j, and T_j on the diagonal. One block per column.
template <class T>
__global__ void compose_tg_step(T *Tg, int H, int j, int b, const T *G, const T *Tn) {
    const int c = blockIdx.x;
    if (c >= b)
        return;
    const int jb = j * b, col = jb + c;
    extern __shared__ __align__(16) unsigned char raw[];
    T *S = reinterpret_cast<T *>(raw);
    for (int i = threadIdx.x; i < b; i += blockDim.x)
        Tg[jb + i + size_t(col) * H] = (i <= c) ? Tn[i + size_t(c) * b] : T(0);
    if (!j)
        return;
    for (int i = threadIdx.x; i < jb; i += blockDim.x) {
        const T *gi = G + size_t(i / b) * b * b + (i % b);
        T acc = 0;
        for (int l = 0; l <= c; ++l)
            acc += gi[size_t(l) * b] * Tn[l + size_t(c) * b];
        S[i] = acc;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < jb; i += blockDim.x) {
        T acc = 0;
        for (int l = i; l < jb; ++l)
            acc += Tg[i + size_t(l) * H] * S[l];
        Tg[i + size_t(col) * H] = -acc;
    }
}
// The same append as two 32 x 32 tiled products, S = G_j T_j and T_g(0:jb, jb + c) = -T_prev S, and
// a copy of the diagonal block. Every sum runs in 32-wide chunks of ascending index.
constexpr int kComposeTile = 32;
template <class T>
__global__ void compose_s_tiled(const T *__restrict__ G, int b, int jb, const T *__restrict__ Tn,
                                T *__restrict__ S) {
    constexpr int CT = kComposeTile;
    __shared__ T Gs[CT][CT + 1], Ts[CT][CT + 1];
    const int i0 = blockIdx.x * CT, c0 = blockIdx.y * CT, tx = threadIdx.x, ty = threadIdx.y;
    T acc[4] = {T(0), T(0), T(0), T(0)};
    const int lmax = min(b, c0 + CT); // Tn upper triangular: l <= c < c0+CT
    for (int l0 = 0; l0 < lmax; l0 += CT) {
        for (int r = ty; r < CT; r += 8) {
            const int l = l0 + r, i = i0 + tx;
            Gs[r][tx] = (i < jb && l < b) ? G[size_t(i / b) * b * b + (i % b) + size_t(l) * b]
                                          : T(0); // Gs[l][i]
            const int c = c0 + r, ll = l0 + tx;
            Ts[r][tx] = (c < b && ll < b && ll <= c) ? Tn[ll + size_t(c) * b] : T(0);
        } // Ts[c][l]
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < CT; ++k) {
            const T g = Gs[k][tx];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[q] += g * Ts[ty + 8 * q][k];
        }
        __syncthreads();
    }
    const int i = i0 + tx;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c = c0 + ty + 8 * q;
        if (i < jb && c < b)
            S[i + size_t(c) * jb] = acc[q];
    }
}
template <class T>
__global__ void compose_n_tiled(T *__restrict__ Tg, int H, int jb, int b, const T *__restrict__ S) {
    constexpr int CT = kComposeTile;
    __shared__ T Ps[CT][CT + 1], Ss[CT][CT + 1];
    const int i0 = blockIdx.x * CT, c0 = blockIdx.y * CT, tx = threadIdx.x, ty = threadIdx.y;
    T acc[4] = {T(0), T(0), T(0), T(0)};
    const int i = i0 + tx;
    for (int l0 = i0; l0 < jb; l0 += CT) { // T_prev upper triangular: l >= i >= i0
        for (int r = ty; r < CT; r += 8) {
            const int l = l0 + r, ii = i0 + tx;
            Ps[r][tx] = (ii < jb && l < jb && l >= ii) ? Tg[ii + size_t(l) * H] : T(0); // Ps[l][i]
            const int c = c0 + r, ll = l0 + tx;
            Ss[r][tx] = (c < b && ll < jb) ? S[ll + size_t(c) * jb] : T(0);
        } // Ss[c][l]
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < CT; ++k) {
            const T p = Ps[k][tx];
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[q] += p * Ss[ty + 8 * q][k];
        }
        __syncthreads();
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c = c0 + ty + 8 * q;
        if (i < jb && c < b)
            Tg[i + size_t(jb + c) * H] = -acc[q];
    }
}
template <class T> __global__ void compose_diag_copy(T *Tg, int H, int j, int b, const T *Tn) {
    const int jb = j * b;
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < b * b; e += gridDim.x * blockDim.x) {
        const int i = e % b, c = e / b;
        Tg[jb + i + size_t(jb + c) * H] = (i <= c) ? Tn[i + size_t(c) * b] : T(0);
    }
}
// TT merge: the triangle of a member, its scatter back after the merge, the member's block of
// reflectors below the stacked triangle, and the owner's identity block: its W is a copy of X and
// its D subtracts Z (or Z^T) directly.
template <class T>
__global__ void extract_r(const T *A, int ld, int row, int col, int rows, int h, T *R, int ldr) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * h; i += blockDim.x * gridDim.x) {
        int r = i % rows, j = i / rows;
        R[r + size_t(j) * ldr] = r <= j ? A[row + r + size_t(col + j) * ld] : T(0);
    }
}
template <class T>
__global__ void scatter_upper(const T *R, int ldr, T *A, int ld, int row, int col, int rows,
                              int h) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * h; i += blockDim.x * gridDim.x) {
        int r = i % rows, j = i / rows;
        if (r <= j)
            A[row + r + size_t(col + j) * ld] = R[r + size_t(j) * ldr];
    }
}
template <class T>
__global__ void tile_bottom_v(const T *A, int ld, int row, int col, int hb, int h, T *V, int b) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < b * b; i += blockDim.x * gridDim.x) {
        int r = i % b, j = i / b;
        V[i] = (r < hb && j < h && r <= j) ? A[row + r + size_t(col + j) * ld] : T(0);
    }
}
// The pr blocks of rows x q (leading dimension rows), one after the other, as one (pr rows) x q
// matrix with leading dimension pr rows.
template <class T> __global__ void restack(const T *blocks, T *stack, int pr, int rows, int q) {
    const size_t n = size_t(pr) * rows * q;
    for (size_t e = blockIdx.x * size_t(blockDim.x) + threadIdx.x; e < n;
         e += size_t(gridDim.x) * blockDim.x) {
        const size_t rest = e / rows;
        const int r = int(e % rows), j = int(rest % q), block = int(rest / q);
        stack[size_t(block) * rows + r + size_t(j) * pr * rows] = blocks[e];
    }
}
// The b x b block with the identity in its leading h x h part and zeros elsewhere.
template <class T> __global__ void tile_identity(T *a, int b, int h) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < b * b; i += blockDim.x * gridDim.x)
        a[i] = i % b == i / b && i % b < h ? T(1) : T(0);
}
template <class T>
__global__ void tile_copy_identity(const T *A, int ld, int row, int col, int rows, int q, T *W,
                                   int ldw) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * q; i += blockDim.x * gridDim.x) {
        int r = i % rows, j = i / rows;
        W[r + size_t(j) * ldw] = A[row + r + size_t(col + j) * ld];
    }
}
template <class T>
__global__ void tile_add_identity(T *A, int ld, int row, int col, int rows, int q, const T *Z,
                                  int ldz) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * q; i += blockDim.x * gridDim.x) {
        int r = i % rows, j = i / rows;
        A[row + r + size_t(col + j) * ld] -= Z[r + size_t(j) * ldz];
    }
}
template <class T>
__global__ void tile_add_identity_t(T *A, int ld, int row, int col, int rows, int q, const T *Zt,
                                    int ldzt) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * q; i += blockDim.x * gridDim.x) {
        int r = i % rows, j = i / rows;
        A[row + r + size_t(col + j) * ld] -= Zt[j + size_t(r) * ldzt];
    }
}
// out = the sum of the c partials of a product, n elements apart in `in`, in a fixed order; `out`
// may be the first partial.
template <class T> __global__ void tile_combine_partials(T *out, const T *in, size_t n, int c) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(blockDim.x) * gridDim.x) {
        T s = in[i];
        for (int z = 1; z < c; ++z)
            s += in[i + size_t(z) * n];
        out[i] = s;
    }
}

// Entry generator of the inputs and the relative error ||a - b||_F / ||b||_F, scaled by the largest
// magnitude to stay in range.
__device__ inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
template <class T>
__global__ void relative_error(const T *a, const T *b, int nr, int n, int lda, int ldb, T *out) {
    __shared__ T red[256];
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t step = size_t(blockDim.x) * gridDim.x;
    T mx = 0;
    for (size_t i = first; i < size_t(nr) * n; i += step) {
        T x = b[i % nr + (i / nr) * ldb];
        mx = max(mx, absval(x));
        mx = max(mx, absval(a[i % nr + (i / nr) * lda]));
    }
    mx = block_max(mx, red);
    T diff = 0, den = 0;
    if (mx)
        for (size_t i = first; i < size_t(nr) * n; i += step) {
            T x = b[i % nr + (i / nr) * ldb];
            T v = a[i % nr + (i / nr) * lda] / mx;
            T y = x / mx;
            diff += (v - y) * (v - y);
            den += y * y;
        }
    diff = block_sum(diff, red);
    den = block_sum(den, red);
    if (threadIdx.x == 0) {
        out += 3 * blockIdx.x;
        out[0] = sqrt(diff);
        out[1] = sqrt(den);
        out[2] = mx;
    }
}

// Combine independently scaled block norms without squaring their original magnitudes.
template <class T> double relative_error_ratio(const std::vector<T> &partials) {
    double scale = 0;
    for (size_t i = 0; i < partials.size(); i += 3) {
        if (!std::isfinite(partials[i]) || !std::isfinite(partials[i + 1]) ||
            !std::isfinite(partials[i + 2]))
            return INFINITY;
        scale = std::max(scale, double(partials[i + 2]));
    }
    if (!scale)
        return 0;
    double diff = 0, den = 0;
    for (size_t i = 0; i < partials.size(); i += 3) {
        const double weight = double(partials[i + 2]) / scale;
        diff = std::hypot(diff, double(partials[i]) * weight);
        den = std::hypot(den, double(partials[i + 1]) * weight);
    }
    return den ? diff / den : diff * scale;
}
} // namespace qr_omega
