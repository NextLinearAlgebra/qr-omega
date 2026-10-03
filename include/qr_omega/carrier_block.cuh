#pragma once
// Block carriers on CUDA cores for the products the tensor-core carriers do not take (few
// reflectors, short or ragged panels): the CB groups of 64 threads of a block split the contraction
// K into balanced intervals [K z / CB, K (z + 1) / CB), keep their partial tile in registers, and
// group 0 adds the partials in a fixed order and commits the tile once.
#include "kernels.cuh"
#include <type_traits>
namespace tqr {
inline constexpr int carried_group_threads = 64;
template <int CB> inline constexpr int carried_tile_m = CB == 8 ? 16 : 64;
template <class T, int CB, int TM, int TN>
inline constexpr size_t carried_shared_bytes =
    size_t(CB) * size_t(TM * TN > 2 * (TM + TN) * 8 ? TM * TN : 2 * (TM + TN) * 8) * sizeof(T);

// C(m x n) (-)= op(A) B over the group's interval of K, with op(A) = A^T when Transpose.
template <class T, int CB, int TM, int TN, bool Transpose, bool Subtract, bool TransC = false,
          bool TransB = false>
__device__ __forceinline__ void carried_apply_body(const T *a, int lda, const T *b, int ldb, T *c,
                                                   int ldc, int m, int n, int k) {
    constexpr int BK = 8, RM = TM / 8, RN = TN / 8;
    const int z = threadIdx.y, t = threadIdx.x, ri = t % 8, cj = t / 8;
    const int i0 = blockIdx.y * TM, j0 = blockIdx.x * TN;
    extern __shared__ __align__(16) unsigned char carried_smem[];
    T *scratch = reinterpret_cast<T *>(carried_smem);
    // Every group stages its slabs of A and B in two slots of shared memory, which the partials
    // reuse after the last slab.
    constexpr size_t SLOT = (TM + TN) * BK;
    T *slot0 = scratch + size_t(z) * 2 * SLOT, *slot1 = slot0 + SLOT;
    T acc[RM][RN] = {};
    const int begin = int((static_cast<long long>(k) * z) / CB);
    const int end = int((static_cast<long long>(k) * (z + 1)) / CB);
    // Every group passes the same number of barriers, short intervals included.
    const int steps = (k + CB * BK - 1) / (CB * BK);
    // The loads of the next slab are issued before the arithmetic on the current one.
    auto load_slab = [&](int kb, T *as, T *bs) {
        for (int ix = t; ix < TM * BK; ix += carried_group_threads) {
            const int i = Transpose ? ix / BK : ix % TM, kk = Transpose ? ix % BK : ix / TM;
            const int row = i0 + i, inner = kb + kk;
            as[kk * TM + i] = (row < m && inner < end) ? (Transpose ? a[inner + size_t(row) * lda]
                                                                    : a[row + size_t(inner) * lda])
                                                       : T(0);
        }
        for (int ix = t; ix < BK * TN; ix += carried_group_threads) {
            const int kk = ix % BK, j = ix / BK, inner = kb + kk;
            bs[j * BK + kk] =
                (j0 + j < n && inner < end)
                    ? (TransB ? b[j0 + j + size_t(inner) * ldb] : b[inner + size_t(j0 + j) * ldb])
                    : T(0);
        }
    };
    auto compute_slab = [&](T *as, T *bs) {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            T av[RM], bv[RN];
#pragma unroll
            for (int u = 0; u < RM; ++u)
                av[u] = as[kk * TM + ri + 8 * u];
#pragma unroll
            for (int v = 0; v < RN; ++v)
                bv[v] = bs[(cj + 8 * v) * BK + kk];
#pragma unroll
            for (int u = 0; u < RM; ++u) {
#pragma unroll
                for (int v = 0; v < RN; ++v)
                    acc[u][v] += av[u] * bv[v];
            }
        }
    };
    T *as = slot0, *bs = slot0 + TM * BK;
    load_slab(begin, as, bs);
    __syncthreads();
    for (int step = 0; step < steps; ++step) {
        T *ns = step % 2 ? slot0 : slot1, *nbs = ns + TM * BK;
        if (step + 1 < steps)
            load_slab(begin + (step + 1) * BK, ns, nbs);
        __syncthreads();
        compute_slab(as, bs);
        __syncthreads();
        as = ns;
        bs = nbs;
    }
    // The staging slots are free: the partials go to the same shared memory.
#pragma unroll
    for (int u = 0; u < RM; ++u) {
#pragma unroll
        for (int v = 0; v < RN; ++v)
            scratch[size_t(z) * TM * TN + ri + 8 * u + size_t(cj + 8 * v) * TM] = acc[u][v];
    }
    __syncthreads();
    if (z == 0) {
#pragma unroll
        for (int u = 0; u < RM; ++u) {
#pragma unroll
            for (int v = 0; v < RN; ++v) {
                const int i = ri + 8 * u, j = cj + 8 * v;
                T sum = scratch[i + size_t(j) * TM];
#pragma unroll
                for (int peer = 1; peer < CB; ++peer)
                    sum += scratch[size_t(peer) * TM * TN + i + size_t(j) * TM];
                if (i0 + i < m && j0 + j < n) {
                    T &out = TransC ? c[j0 + j + size_t(i0 + i) * ldc]
                                    : c[i0 + i + size_t(j0 + j) * ldc];
                    if constexpr (Subtract)
                        out -= sum;
                    else
                        out = sum;
                }
            }
        }
    }
}

template <class T, int CB, int TM = carried_tile_m<CB>, int TN = 64>
__global__ __launch_bounds__(carried_group_threads *CB) void tile_carried_W(const T *v, int ldv,
                                                                            const T *x, int ldx,
                                                                            T *w, int ldw, int rows,
                                                                            int h, int q) {
    carried_apply_body<T, CB, TM, TN, true, false>(v, ldv, x, ldx, w, ldw, h, q, rows);
}
template <class T, int CB, bool TransC = false, int TM = carried_tile_m<CB>, int TN = 64>
__global__ __launch_bounds__(carried_group_threads *CB) void tile_carried_Z(const T *t, int ldt,
                                                                            const T *w, int ldw,
                                                                            T *z, int ldz, int h,
                                                                            int q, bool transpose) {
    if (transpose)
        carried_apply_body<T, CB, TM, TN, true, false, TransC>(t, ldt, w, ldw, z, ldz, h, q, h);
    else
        carried_apply_body<T, CB, TM, TN, false, false, TransC>(t, ldt, w, ldw, z, ldz, h, q, h);
}
template <class T, int CB, int TM = carried_tile_m<CB>, int TN = 64>
__global__ __launch_bounds__(carried_group_threads *CB) void tile_carried_D(const T *v, int ldv,
                                                                            const T *z, int ldz,
                                                                            T *x, int ldx, int rows,
                                                                            int h, int q) {
    // (m, n, k) = (rows, q, h): X(rows x q) -= V(rows x h) Z(h x q).
    carried_apply_body<T, CB, TM, TN, false, true>(v, ldv, z, ldz, x, ldx, rows, q, h);
}
// Runs f with the number of groups as a constant: two, or one when the contraction has a single
// index.
template <class F> void dispatch_block_carrier(int contraction, F &&f) {
    if (contraction < 2)
        f(std::integral_constant<int, 1>{});
    else
        f(std::integral_constant<int, 2>{});
}
template <class T, int CB>
inline constexpr size_t carried_bytes = carried_shared_bytes<T, CB, carried_tile_m<CB>, 64>;
template <class T>
void launch_carried_W(const T *v, int ldv, const T *x, int ldx, T *w, int ldw, int rows, int h,
                      int q, cudaStream_t st) {
    if (!rows || !h || !q)
        return;
    dispatch_block_carrier(rows, [&](auto tag) {
        constexpr int CB = decltype(tag)::value;
        reserve_shared_memory(tile_carried_W<T, CB>, carried_bytes<T, CB>);
        tile_carried_W<T, CB><<<dim3(ceildiv(q, 64), ceildiv(h, carried_tile_m<CB>)),
                                dim3(carried_group_threads, CB), carried_bytes<T, CB>, st>>>(
            v, ldv, x, ldx, w, ldw, rows, h, q);
    });
}
// Z = op(T) W, or Z^T (q x h, leading dimension ldz) when zt_out, as the D carriers read it.
template <class T>
void launch_carried_Z(const T *t, int ldt, const T *w, int ldw, T *z, int ldz, int h, int q,
                      bool transpose, cudaStream_t st, bool zt_out = false) {
    if (!h || !q)
        return;
    dispatch_block_carrier(h, [&](auto tag) {
        constexpr int CB = decltype(tag)::value;
        const dim3 grid(ceildiv(q, 64), ceildiv(h, carried_tile_m<CB>)),
            block(carried_group_threads, CB);
        if (zt_out) {
            reserve_shared_memory(tile_carried_Z<T, CB, true>, carried_bytes<T, CB>);
            tile_carried_Z<T, CB, true><<<grid, block, carried_bytes<T, CB>, st>>>(
                t, ldt, w, ldw, z, ldz, h, q, transpose);
        } else {
            reserve_shared_memory(tile_carried_Z<T, CB>, carried_bytes<T, CB>);
            tile_carried_Z<T, CB><<<grid, block, carried_bytes<T, CB>, st>>>(t, ldt, w, ldw, z, ldz,
                                                                             h, q, transpose);
        }
    });
}
template <class T>
void launch_carried_D(const T *v, int ldv, const T *z, int ldz, T *x, int ldx, int rows, int h,
                      int q, cudaStream_t st) {
    if (!rows || !h || !q)
        return;
    dispatch_block_carrier(h, [&](auto tag) {
        constexpr int CB = decltype(tag)::value;
        reserve_shared_memory(tile_carried_D<T, CB>, carried_bytes<T, CB>);
        tile_carried_D<T, CB><<<dim3(ceildiv(q, 64), ceildiv(rows, carried_tile_m<CB>)),
                                dim3(carried_group_threads, CB), carried_bytes<T, CB>, st>>>(
            v, ldv, z, ldz, x, ldx, rows, h, q);
    });
}

// The same carrier with C groups for the short contractions of the tensor-core modes (see
// short_carrier_path in update.cuh).
template <class T, int C, bool D, bool TransC = false>
__global__ __launch_bounds__(carried_group_threads *C) void short_carrier_kernel(
    const T *a, int lda, const T *b, int ldb, T *out, int ldo, int rows, int h, int q,
    bool transpose) {
    if constexpr (D)
        carried_apply_body<T, C, 64, 64, false, true, false, true>(a, lda, b, ldb, out, ldo, rows,
                                                                   q, h);
    else if (transpose)
        carried_apply_body<T, C, 64, 64, true, false, TransC>(a, lda, b, ldb, out, ldo, h, q, h);
    else
        carried_apply_body<T, C, 64, 64, false, false, TransC>(a, lda, b, ldb, out, ldo, h, q, h);
}
template <class T, bool D, bool TransC = false>
void launch_short_carrier(int c, const T *a, int lda, const T *b, int ldb, T *out, int ldo,
                          int rows, int h, int q, bool transpose, cudaStream_t st) {
    if (!rows || !q)
        return;
    auto launch = [&](auto tag) {
        constexpr int C = decltype(tag)::value;
        constexpr size_t bytes = carried_shared_bytes<T, C, 64, 64>;
        reserve_shared_memory(short_carrier_kernel<T, C, D, TransC>, bytes);
        short_carrier_kernel<T, C, D, TransC>
            <<<dim3(ceildiv(q, 64), ceildiv(D ? rows : h, 64)), dim3(carried_group_threads, C),
               bytes, st>>>(a, lda, b, ldb, out, ldo, rows, h, q, transpose);
        CU(cudaGetLastError());
    };
    if (c == 2)
        launch(std::integral_constant<int, 2>{});
    else
        launch(std::integral_constant<int, 4>{});
}
} // namespace tqr
