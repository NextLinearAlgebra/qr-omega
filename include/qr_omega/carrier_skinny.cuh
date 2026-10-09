#pragma once
// Products C = A^T B with few outputs over a long contraction: the Gram blocks and the updates
// inside a panel. Every warp owns a small output tile and one of `groups` consecutive ranges of
// the rows; its lanes walk the range row by row with coalesced loads and sum their partials with
// shuffles, and the warps of a block own neighbouring tiles. The small tiles leave enough of them
// for groups within the 2.5D bound groups^2 <= tiles: the carrier is (row tiles, column tiles,
// groups) at the GPU level and (1, warps, 1) at the block level. Several groups write partial
// tiles that are summed in a fixed order.
#include "carrier.hpp"
#include "kernels.cuh"

namespace qr_omega {
inline constexpr int skinny_warps = 4;
// Rows a group of the skinny carrier walks at least.
inline constexpr int skinny_min_rows = 256;

// Lane l ends with the sums over the warp of values [l N / 32, (l + 1) N / 32) when N >= 32, else
// lanes l and its 32 / N - 1 neighbours hold the sum of value l N / 32.
template <class T, int N> __device__ __forceinline__ void skinny_reduce(T (&v)[N], int lane) {
    constexpr int halvings = N >= 32 ? 5 : N >= 16 ? 4 : N >= 8 ? 3 : N >= 4 ? 2 : N >= 2 ? 1 : 0;
#pragma unroll
    for (int level = 0; level < halvings; ++level) {
        const int o = 16 >> level, half = N >> (level + 1);
        const bool hi = lane & o;
#pragma unroll
        for (int q = 0; q < half; ++q) {
            const T send = hi ? v[q] : v[q + half];
            const T keep = hi ? v[q + half] : v[q];
            v[q] = keep + __shfl_xor_sync(0xffffffffu, send, o);
        }
    }
#pragma unroll
    for (int o = 16 >> halvings; o >= 1; o /= 2)
        v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
}

template <class T, int TM, int TN>
__global__ __launch_bounds__(skinny_warps *
                             32) void skinny_atb_kernel(const T *__restrict__ a, int lda,
                                                        const T *__restrict__ b, int ldb, T *out,
                                                        int ldo, long long slice, int m, int n,
                                                        int k, int groups) {
    constexpr int N = TM * TN, P = N >= 32 ? N / 32 : 1, spread = N >= 32 ? 1 : 32 / N;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int i0 = blockIdx.y * TM, j0 = (blockIdx.x * skinny_warps + warp) * TN, z = blockIdx.z;
    if (j0 >= n)
        return;
    const int k0 = int((long long)k * z / groups), k1 = int((long long)k * (z + 1) / groups);
    T acc[N] = {};
    for (int r = k0 + lane; r < k1; r += 32) {
        T av[TM], bv[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i)
            av[i] = i0 + i < m ? a[r + size_t(i0 + i) * lda] : T(0);
#pragma unroll
        for (int j = 0; j < TN; ++j)
            bv[j] = j0 + j < n ? b[r + size_t(j0 + j) * ldb] : T(0);
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int i = 0; i < TM; ++i)
                acc[i + TM * j] += av[i] * bv[j];
    }
    skinny_reduce<T, N>(acc, lane);
    if (lane % spread)
        return;
    T *o = out + size_t(z) * slice;
#pragma unroll
    for (int s = 0; s < P; ++s) {
        const int f = lane / spread * P + s, i = f % TM, j = f / TM;
        if (i0 + i < m && j0 + j < n)
            o[i0 + i + size_t(j0 + j) * ldo] = acc[s];
    }
}

// Output tiles of a warp: wide for the larger outputs, narrow ones for more tiles and so more
// groups within the bound.
template <class T> struct SkinnyTiles;
template <> struct SkinnyTiles<double> {
    static constexpr int wide_m = 8, wide_n = 8, narrow_m = 4, narrow_n = 4;
};
template <> struct SkinnyTiles<float> {
    static constexpr int wide_m = 16, wide_n = 8, narrow_m = 8, narrow_n = 4;
};
// The skinny carrier takes products with at most this many outputs.
inline constexpr long long skinny_max_outputs = 64 * 128;

// C (m x n, leading dimension ldc) = A^T B over k rows of A (k x m) and B (k x n). Partial tiles
// of several groups go to `parts`, `slice` elements apart, room for `room` of them.
template <class T>
Carrier launch_skinny_atb(const T *a, int lda, const T *b, int ldb, T *c, int ldc, T *parts,
                          long long slice, int room, int k, int m, int n, cudaStream_t st) {
    using Tiles = SkinnyTiles<T>;
    const int target = 8 * device_properties().multiProcessorCount;
    const int most = std::max(1, std::min(room, k / skinny_min_rows));
    // The wide tile when its warps fill the GPU, else the one with more warps.
    auto plan = [&](int tm, int tn) {
        const long long tiles = (long long)ceildiv(m, tm) * ceildiv(n, tn);
        return std::pair<long long, int>(tiles, bounded_groups(tiles, most));
    };
    const auto [wide_tiles, wide_groups] = plan(Tiles::wide_m, Tiles::wide_n);
    const auto [narrow_tiles, narrow_groups] = plan(Tiles::narrow_m, Tiles::narrow_n);
    const bool wide = wide_tiles * wide_groups >= target ||
                      wide_tiles * wide_groups >= narrow_tiles * narrow_groups;
    const int tm = wide ? Tiles::wide_m : Tiles::narrow_m,
              tn = wide ? Tiles::wide_n : Tiles::narrow_n;
    const int groups = wide ? wide_groups : narrow_groups;
    T *out = groups > 1 ? parts : c;
    const dim3 grid(ceildiv(ceildiv(n, tn), skinny_warps), ceildiv(m, tm), groups);
    if (wide)
        skinny_atb_kernel<T, Tiles::wide_m, Tiles::wide_n>
            <<<grid, skinny_warps * 32, 0, st>>>(a, lda, b, ldb, out, ldc, slice, m, n, k, groups);
    else
        skinny_atb_kernel<T, Tiles::narrow_m, Tiles::narrow_n>
            <<<grid, skinny_warps * 32, 0, st>>>(a, lda, b, ldb, out, ldc, slice, m, n, k, groups);
    CU(cudaGetLastError());
    if (groups > 1) {
        tile_combine_partials<T>
            <<<std::max(1, std::min(264, ceildiv(int(slice), 256))), 256, 0, st>>>(
                c, parts, size_t(slice), groups);
        CU(cudaGetLastError());
    }
    return Carrier{}
        .set(GpuLevel, ceildiv(m, tm), ceildiv(n, tn), groups)
        .set(BlockLevel, 1, skinny_warps, 1);
}
} // namespace qr_omega
