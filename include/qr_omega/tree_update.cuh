#pragma once
// The update of a strip by one step of a GPU's elimination tree, fused into one kernel: a block owns
// a column tile of one segment (a domain, or the children of a TT elimination), streams the
// segment's rows through shared memory twice and commits X once:
//     W = V^T X,   Z = op(T) W,   X -= V Z.
// The block is the 2.5D carrier of KAMI at the block level: its warps form a 2D grid over every
// output and `layers` layers split the contraction of W (the rows of a domain, or the children of a
// TT elimination), whose partial products meet in shared memory in a fixed order. At the GPU level
// the blocks form the 2D grid (segments, column tiles). Chunks of 64 rows of V and X stream through
// two stages of shared memory with cp.async, so the loads of one chunk overlap the tensor-core
// products (m16n8k16, the fragment map of carrier_kami.cuh) of the previous one.
#include "domains.cuh"
#include "tree_math.cuh"

namespace tqr {
// The rows a step of a tree updates, as chunks of up to 64 rows: segment s of a domain step is the
// first tile at s domain in X (the GE reflectors packed at s tile in V). A TS step joins the
// domain head and its next dense tile. Segment s of a TT step is node s, whose chunk k is
// the triangle of child k, at row s outer + k inner of X and (s fan_in + k) h of the stacked V.
struct TreeStep {
    int h = 0, segments = 0;
    int domain = 0, last = 0;          // domain step: rows of a segment and of the last one
    int tile = 0, phase = 0;           // GE height cap, or TS tile height and chain position
    long long outer = 0, inner = 0;    // TT step: rows between nodes and between children
    int fan_in = 0, last_children = 0; // TT step: children of a node and of the last one
    __host__ __device__ bool merges() const {
        return outer > 0;
    }
    __host__ __device__ int tail(int s) const {
        return max(0, min(tile, (s + 1 < segments ? domain : last) - phase * tile));
    }
    __host__ __device__ int ge_rows(int s) const {
        const int rows = s + 1 < segments ? domain : last;
        return tile ? min(tile, rows) : rows;
    }
    __host__ __device__ int chunks(int s) const {
        if (phase)
            return 1 + (tail(s) + 63) / 64;
        if (merges())
            return s + 1 < segments ? fan_in : last_children;
        return (ge_rows(s) + 63) / 64;
    }
    __host__ __device__ int chunk_rows(int s, int k) const {
        if (phase)
            return k == 0 ? h : min(64, tail(s) - 64 * (k - 1));
        if (merges())
            return h;
        return min(64, ge_rows(s) - 64 * k);
    }
    __host__ __device__ long long x_row(int s, int k) const {
        if (phase)
            return (long long)s * domain + (k == 0 ? 0 : phase * tile + 64 * (k - 1));
        return merges() ? s * outer + k * inner : (long long)s * domain + 64 * k;
    }
    __host__ __device__ long long v_row(int s, int k) const {
        if (phase)
            return (long long)s * DomainConfig::rows + (k == 0 ? 0 : h + 64 * (k - 1));
        return merges() ? (long long)(s * fan_in + k) * h
                        : (long long)s * (tile ? min(tile, domain) : domain) + 64 * k;
    }
    bool aligned_pairs() const {
        return phase      ? domain % 2 == 0 && tile % 2 == 0 && h % 2 == 0
               : merges() ? h % 2 == 0 && outer % 2 == 0 && inner % 2 == 0
                          : domain % 2 == 0;
    }
};

// Pack the disjoint head and tail of every TS update into contiguous segments for the
// segmented FP32/TF32 carriers. Padding is zero, including absent tails in short domains.
template <class T, bool GATHER>
__global__ void tree_ts_copy(TreeStep st, T *x, int ldx, T *packed, int q) {
    const int height = DomainConfig::rows, ld = st.segments * height;
    const long long total = (long long)ld * q;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < total;
         e += (long long)gridDim.x * blockDim.x) {
        const int row = int(e % ld), col = int(e / ld), s = row / height, r = row % height;
        const bool valid = r < st.h || r - st.h < st.tail(s);
        const long long xr =
            (long long)s * st.domain + (r < st.h ? r : st.phase * st.tile + r - st.h);
        if constexpr (GATHER)
            packed[e] = valid ? x[xr + (long long)col * ldx] : T(0);
        else if (valid)
            x[xr + (long long)col * ldx] = packed[e];
    }
}

// Shared memory, all column-major: two stages of a V chunk (64 x 64) and an X chunk (64 x BN), T,
// and Z; the layers' partials of W reuse the stages once the last chunk of W is consumed.
template <class T, int BN, int LAYERS, int THREADS = 256, int STAGES = 2> struct TreeUpdateCfg {
    static constexpr int threads = THREADS, warps = THREADS / 32, H = 64, LD = H + 4;
    static constexpr int stages = STAGES, layer_warps = warps / LAYERS;
    static_assert(warps % LAYERS == 0 && (H / 16) % LAYERS == 0, "layers split the k-steps");
    static_assert(LAYERS * LAYERS * LAYERS <= warps, "2.5D layers must satisfy c^3 <= P");
    static constexpr size_t v_words = size_t(H) * LD, x_words = size_t(BN) * LD,
                            stage = v_words + x_words, t_words = size_t(H) * LD;
    static constexpr size_t smem = (STAGES * stage + t_words + x_words) * sizeof(T);
    static_assert(LAYERS * x_words <= STAGES * stage && x_words <= stage,
                  "the partials fit the stages");
};

// Issues the copies of chunk k of segment s into one stage: V (rows x h, zero beyond) and the
// column tile of X (rows x BN), column-major with leading dimension LD. PAIRS copies two rows at a
// time (16 bytes), for operands whose columns start at even rows.
template <class T, class Cfg, int BN, bool PAIRS>
__device__ __forceinline__ void tu_issue(const TreeStep &st, int s, int k, const T *v, int ldv,
                                         const T *x, int ldx, int col0, int q, T *stage) {
    const int rows = st.chunk_rows(s, k);
    const T *vc = v + st.v_row(s, k), *xc = x + st.x_row(s, k) + (long long)col0 * ldx;
    T *vs = stage, *xs = stage + Cfg::v_words;
    if constexpr (PAIRS) {
        constexpr int R2 = Cfg::H / 2;
        for (int e = threadIdx.x; e < R2 * Cfg::H; e += Cfg::threads) {
            const int r = 2 * (e % R2), c = e / R2;
            const int n = c < st.h ? max(0, min(2, rows - r)) : 0;
            tree_copy<T, 2>(vs + c * Cfg::LD + r, n ? vc + r + size_t(c) * ldv : v, n);
        }
        for (int e = threadIdx.x; e < R2 * BN; e += Cfg::threads) {
            const int r = 2 * (e % R2), j = e / R2;
            const int n = col0 + j < q ? max(0, min(2, rows - r)) : 0;
            tree_copy<T, 2>(xs + j * Cfg::LD + r, n ? xc + r + size_t(j) * ldx : x, n);
        }
    } else {
        for (int e = threadIdx.x; e < Cfg::H * Cfg::H; e += Cfg::threads) {
            const int r = e % Cfg::H, c = e / Cfg::H;
            const bool in = r < rows && c < st.h;
            tree_copy<T, 1>(vs + c * Cfg::LD + r, in ? vc + r + size_t(c) * ldv : v, in);
        }
        for (int e = threadIdx.x; e < Cfg::H * BN; e += Cfg::threads) {
            const int r = e % Cfg::H, j = e / Cfg::H;
            const bool in = r < rows && col0 + j < q;
            tree_copy<T, 1>(xs + j * Cfg::LD + r, in ? xc + r + size_t(j) * ldx : x, in);
        }
    }
}

// Warp tiling of a 64 x BN output among `warps` warps: each warp TM x TN tiles of 16 x 8.
template <int BN, int WARPS> struct TreeTiling {
    static constexpr int MT = 4, NT = BN / 8, per = MT * NT / WARPS, TM = per >= 2 ? 2 : 1,
                         TN = per / TM, WM = MT / TM, WN = NT / TN;
    static_assert(per >= 1 && WM * WN == WARPS && TM * WM == MT && TN * WN == NT, "warp tiling");
};
// The fragments of the m16n8k16 products take k-slot (t, u) of a lane as index 4 t + u in both
// operands, a permutation of the hardware's t + 4 u that leaves every product unchanged and makes
// each lane's four indices consecutive: every fragment comes from shared memory in 16-byte loads.
// a[2u], a[2u + 1] are rows m, m + 1 at index 4 t + u; b[u] is index 4 t + u of column n.
template <class T>
__device__ __forceinline__ void tu_frag_a_cols(const T *col_m, const T *col_m1, T (&a)[8]) {
    // A(m, k) = col_m[k] (two columns m, m + 1, k consecutive)
    const TreePair<T> p0 = *reinterpret_cast<const TreePair<T> *>(col_m),
                      p1 = *reinterpret_cast<const TreePair<T> *>(col_m + 2),
                      q0 = *reinterpret_cast<const TreePair<T> *>(col_m1),
                      q1 = *reinterpret_cast<const TreePair<T> *>(col_m1 + 2);
    a[0] = p0.x, a[2] = p0.y, a[4] = p1.x, a[6] = p1.y;
    a[1] = q0.x, a[3] = q0.y, a[5] = q1.x, a[7] = q1.y;
}
template <class T>
__device__ __forceinline__ void tu_frag_a_rows(const T *pairs, int ld, T (&a)[8]) {
    // A(m, k): the rows m, m + 1 adjacent in column k (pairs + k ld), k consecutive
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const TreePair<T> p = *reinterpret_cast<const TreePair<T> *>(pairs + u * ld);
        a[2 * u] = p.x, a[2 * u + 1] = p.y;
    }
}
template <class T> __device__ __forceinline__ void tu_frag_b(const T *col_n, T (&b)[4]) {
    const TreePair<T> p0 = *reinterpret_cast<const TreePair<T> *>(col_n),
                      p1 = *reinterpret_cast<const TreePair<T> *>(col_n + 2);
    b[0] = p0.x, b[1] = p0.y, b[2] = p1.x, b[3] = p1.y;
}

// One step of a tree on the columns [0, q) of x: grid (column tiles, segments).
template <class T, class Math, int BN, int LAYERS, int THREADS, int STAGES, bool PAIRS>
__global__ __launch_bounds__(THREADS) void tree_update_kernel(TreeStep st, const T *v, int ldv,
                                                              const T *t, T *x, int ldx, int q,
                                                              bool transpose) {
    using Cfg = TreeUpdateCfg<T, BN, LAYERS, THREADS, STAGES>;
    constexpr int S = Cfg::stages, H = Cfg::H, LD = Cfg::LD;
    extern __shared__ __align__(16) unsigned char tu_smem[];
    T *stages = reinterpret_cast<T *>(tu_smem), *ts = stages + S * Cfg::stage,
      *zs = ts + Cfg::t_words;
    const int s = blockIdx.y, col0 = blockIdx.x * BN, h = st.h, chunks = st.chunks(s);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int p = 0; p < S - 1; ++p) {
        if (p < chunks)
            tu_issue<T, Cfg, BN, PAIRS>(st, s, p, v, ldv, x, ldx, col0, q, stages + p * Cfg::stage);
        tu_commit();
    }
    // T of the segment, T(i, j) at ts[j LD + i].
    const T *tseg = t + size_t(s) * h * h;
    for (int e = threadIdx.x; e < H * H; e += Cfg::threads) {
        const int i = e % H, j = e / H;
        ts[j * LD + i] = i < h && j < h ? tseg[i + size_t(j) * h] : T(0);
    }
    // W = V^T X (m: reflector, n: column, k: row): layer `layer` takes the k-steps layer,
    // layer + LAYERS, ... of every chunk, its warps tiled by TreeTiling.
    using WT = TreeTiling<BN, Cfg::layer_warps>;
    const int layer = warp / Cfg::layer_warps, lw = warp % Cfg::layer_warps, wm = lw % WT::WM,
              wn = lw / WT::WM;
    T acc[WT::TM][WT::TN][4] = {};
    for (int k = 0; k < chunks; ++k) {
        if (k + S - 1 < chunks)
            tu_issue<T, Cfg, BN, PAIRS>(st, s, k + S - 1, v, ldv, x, ldx, col0, q,
                                        stages + ((k + S - 1) % S) * Cfg::stage);
        tu_commit();
        tu_wait<S - 1>();
        __syncthreads();
        const T *vs = stages + (k % S) * Cfg::stage, *xs = vs + Cfg::v_words;
#pragma unroll
        for (int ks = layer; ks < H / 16; ks += LAYERS) {
            const int kk = ks * 16 + 4 * tq;
            T a[WT::TM][8];
#pragma unroll
            for (int i = 0; i < WT::TM; ++i) {
                const int m = (wm * WT::TM + i) * 16 + 2 * gq;
                tu_frag_a_cols(vs + m * LD + kk, vs + (m + 1) * LD + kk, a[i]);
            }
#pragma unroll
            for (int jn = 0; jn < WT::TN; ++jn) {
                T b[4];
                tu_frag_b(xs + ((wn * WT::TN + jn) * 8 + gq) * LD + kk, b);
#pragma unroll
                for (int i = 0; i < WT::TM; ++i)
                    Math::mma(acc[i][jn], a[i], b);
            }
        }
        __syncthreads();
    }
    // The layers' partials W(m, n) at wp[n LD + m] in the stages, summed in the order 0..LAYERS-1.
    T *wp = stages;
#pragma unroll
    for (int i = 0; i < WT::TM; ++i)
#pragma unroll
        for (int jn = 0; jn < WT::TN; ++jn) {
            const int m = (wm * WT::TM + i) * 16 + 2 * gq, n = (wn * WT::TN + jn) * 8 + 2 * tq;
            T *o = wp + layer * Cfg::x_words + n * LD + m;
            *reinterpret_cast<TreePair<T> *>(o) = tree_make_pair(acc[i][jn][0], acc[i][jn][2]);
            *reinterpret_cast<TreePair<T> *>(o + LD) = tree_make_pair(acc[i][jn][1], acc[i][jn][3]);
        }
    __syncthreads();
    if constexpr (LAYERS > 1) {
        for (int e = threadIdx.x; e < H * BN; e += Cfg::threads) {
            const int r = e % H, j = e / H;
            T sum = wp[j * LD + r];
#pragma unroll
            for (int l = 1; l < LAYERS; ++l)
                sum += wp[l * Cfg::x_words + j * LD + r];
            wp[j * LD + r] = sum;
        }
        __syncthreads();
    }
    // The first chunks of X -= V Z stream into stages 1.. while Z forms; chunk k of the D pass
    // lives in stage (k + 1) mod S.
#pragma unroll
    for (int p = 0; p < S - 1; ++p) {
        if (p < chunks)
            tu_issue<T, Cfg, BN, PAIRS>(st, s, p, v, ldv, x, ldx, col0, q,
                                        stages + ((p + 1) % S) * Cfg::stage);
        tu_commit();
    }
    // -Z = -op(T) W, kept as zs[n LD + m] (m: reflector), all warps.
    using AT = TreeTiling<BN, Cfg::warps>;
    const int am = warp % AT::WM, an = warp / AT::WM;
    {
        T z[AT::TM][AT::TN][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < H; k0 += 16) {
            const int kk = k0 + 4 * tq;
            T a[AT::TM][8];
#pragma unroll
            for (int i = 0; i < AT::TM; ++i) {
                const int m = (am * AT::TM + i) * 16 + 2 * gq;
                if (transpose) // op(T)(m, k) = T(k, m): column m of T, k consecutive
                    tu_frag_a_cols(ts + m * LD + kk, ts + (m + 1) * LD + kk, a[i]);
                else // T(m, k): row m of T
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        a[i][2 * u] = ts[(kk + u) * LD + m];
                        a[i][2 * u + 1] = ts[(kk + u) * LD + m + 1];
                    }
            }
#pragma unroll
            for (int jn = 0; jn < AT::TN; ++jn) {
                T b[4];
                tu_frag_b(wp + ((an * AT::TN + jn) * 8 + gq) * LD + kk, b);
#pragma unroll
                for (int i = 0; i < AT::TM; ++i)
                    Math::mma(z[i][jn], a[i], b);
            }
        }
#pragma unroll
        for (int i = 0; i < AT::TM; ++i)
#pragma unroll
            for (int jn = 0; jn < AT::TN; ++jn) {
                const int m = (am * AT::TM + i) * 16 + 2 * gq, n = (an * AT::TN + jn) * 8 + 2 * tq;
                *reinterpret_cast<TreePair<T> *>(zs + n * LD + m) =
                    tree_make_pair(-z[i][jn][0], -z[i][jn][2]);
                *reinterpret_cast<TreePair<T> *>(zs + (n + 1) * LD + m) =
                    tree_make_pair(-z[i][jn][1], -z[i][jn][3]);
            }
    }
    // X += V (-Z) chunk by chunk: X enters the accumulators and leaves them for global memory.
    for (int k = 0; k < chunks; ++k) {
        const int at = (k + 1) % S;
        __syncthreads(); // -Z is formed; the stage of chunk k - 1 is free again
        if (k + S - 1 < chunks)
            tu_issue<T, Cfg, BN, PAIRS>(st, s, k + S - 1, v, ldv, x, ldx, col0, q,
                                        stages + ((k + S) % S) * Cfg::stage);
        tu_commit();
        tu_wait<S - 1>();
        __syncthreads();
        const T *vs = stages + at * Cfg::stage, *xs = vs + Cfg::v_words;
        // V (-Z) accumulates from zero and joins X in one rounded addition: tensor cores that
        // accumulate TF32 products truncate the sum, which would otherwise erode X at every step.
        T d[AT::TM][AT::TN][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < H; k0 += 16) {
            const int kk = k0 + 4 * tq;
            T a[AT::TM][8];
#pragma unroll
            for (int i = 0; i < AT::TM; ++i)
                tu_frag_a_rows(vs + kk * LD + (am * AT::TM + i) * 16 + 2 * gq, LD, a[i]);
#pragma unroll
            for (int jn = 0; jn < AT::TN; ++jn) {
                T b[4];
                tu_frag_b(zs + ((an * AT::TN + jn) * 8 + gq) * LD + kk, b);
#pragma unroll
                for (int i = 0; i < AT::TM; ++i)
                    Math::mma(d[i][jn], a[i], b);
            }
        }
        const int rows = st.chunk_rows(s, k);
        T *xc = x + st.x_row(s, k) + (long long)col0 * ldx;
#pragma unroll
        for (int i = 0; i < AT::TM; ++i)
#pragma unroll
            for (int jn = 0; jn < AT::TN; ++jn) {
                const int m = (am * AT::TM + i) * 16 + 2 * gq, n = (an * AT::TN + jn) * 8 + 2 * tq;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = m + (e >> 1), j = n + (e & 1);
                    if (r < rows && col0 + j < q)
                        xc[r + size_t(j) * ldx] = xs[j * LD + r] + d[i][jn][e];
                }
            }
    }
}

// The X-resident variant: a block owns a column tile of BN = 32 columns of a segment, reads it into
// shared memory once (padded to whole chunks), forms W = V^T X and -Z = -op(T) W there, adds V (-Z) in
// the accumulators and writes X once. V and T come straight from L2 (read-only loads into the fragments),
// so two blocks fit an SM and one's loads hide behind the other's tensor-core products.
template <class T, int LAYERS> struct TreeUpdateX {
    static constexpr int BN = 32, H = 64, threads = 256, rows = 256, LDX = rows + 2, LDW = H + 4;
    static constexpr size_t smem = (size_t(BN) * LDX + LAYERS * size_t(BN) * LDW) * sizeof(T);
};
template <class T> __device__ __forceinline__ TreePair<T> tux_ld2(const T *p) {
    return tree_pair(p);
}
// The column tile [col0, col0 + BN) of segment s by one group of C::threads threads (thread `tid`
// of the group, shared memory `smem` of C::smem bytes). Every barrier is the whole block's, so the
// groups of a block run it together on the same segment.
template <class T, class Math, int LAYERS, bool PAIRS>
__device__ __forceinline__ void tree_update_x_tile(const TreeStep &st, int s, int col0, const T *v,
                                                   int ldv, const T *t, T *x, int ldx, int q,
                                                   bool transpose, unsigned char *smem, int tid) {
    using C = TreeUpdateX<T, LAYERS>;
    constexpr int BN = C::BN, H = C::H, LDX = C::LDX, LDW = C::LDW;
    T *xs = reinterpret_cast<T *>(smem), *ws = xs + BN * LDX;
    const int h = st.h, chunks = st.chunks(s), r16 = 64 * chunks;
    const int lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
    // X tile, chunk k at rows [64 k, 64 k + 64) of xs, zero past the chunk's rows and the columns
    for (int k = 0; k < chunks; ++k) {
        const int rows = st.chunk_rows(s, k);
        const T *xc = x + st.x_row(s, k) + (long long)col0 * ldx;
        if constexpr (PAIRS) {
            for (int e = tid; e < 32 * BN; e += C::threads) {
                const int r = 2 * (e % 32), j = e / 32;
                const int n = col0 + j < q ? max(0, min(2, rows - r)) : 0;
                tree_copy<T, 2>(xs + j * LDX + 64 * k + r, n ? xc + r + size_t(j) * ldx : x, n);
            }
        } else {
            for (int e = tid; e < 64 * BN; e += C::threads) {
                const int r = e % 64, j = e / 64;
                const bool in = r < rows && col0 + j < q;
                tree_copy<T, 1>(xs + j * LDX + 64 * k + r, in ? xc + r + size_t(j) * ldx : x, in);
            }
        }
    }
    tu_commit();
    tu_wait<0>();
    __syncthreads();
    // V(r, c) of chunk k (rows past the chunk read as zero; only X rows within it are nonzero, so a
    // clamped row suffices)
    auto vcol = [&](int k, int c) { return v + st.v_row(s, k) + size_t(c) * ldv; };
    // W: a 2D warp grid per layer. Each layer owns alternating 16-row
    // contraction slices, retaining its partials until the fixed-order sum.
    {
        constexpr int WM = 4 / LAYERS, LW = 8 / LAYERS;
        const int layer = warp / LW, lw = warp % LW, mt = (lw % WM) * LAYERS, n0 = (lw / WM) * 2;
        T acc[LAYERS][2][4] = {};
        for (int k = 0; k < chunks; ++k) {
            const int rows = st.chunk_rows(s, k);
#pragma unroll
            for (int ks = layer; ks < 4; ks += LAYERS) {
                const int kk = ks * 16 + 4 * tq;
#pragma unroll
                for (int i = 0; i < LAYERS; ++i) {
                    T a[8];
                    const int m = (mt + i) * 16 + 2 * gq;
                    const T *pa = vcol(k, min(m, h - 1)), *pb = vcol(k, min(m + 1, h - 1));
                    if (PAIRS && kk + 3 < rows)
                        tu_frag_a_cols(pa + kk, pb + kk, a);
                    else {
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            a[2 * u] = kk + u < rows ? __ldg(pa + kk + u) : T(0);
                            a[2 * u + 1] = kk + u < rows ? __ldg(pb + kk + u) : T(0);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 8; ++u)
                        if (m + (u & 1) >= h)
                            a[u] = T(0);
#pragma unroll
                    for (int jn = 0; jn < 2; ++jn) {
                        T b[4];
                        tu_frag_b(xs + ((n0 + jn) * 8 + gq) * LDX + 64 * k + kk, b);
                        Math::mma(acc[i][jn], a, b);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < LAYERS; ++i)
#pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                const int m = (mt + i) * 16 + 2 * gq, n = (n0 + jn) * 8 + 2 * tq;
                T *out = ws + size_t(layer) * BN * LDW + n * LDW + m;
                *reinterpret_cast<TreePair<T> *>(out) =
                    tree_make_pair(acc[i][jn][0], acc[i][jn][2]);
                *reinterpret_cast<TreePair<T> *>(out + LDW) =
                    tree_make_pair(acc[i][jn][1], acc[i][jn][3]);
            }
    }
    __syncthreads();
    if constexpr (LAYERS > 1) {
        for (int e = tid; e < H * BN; e += C::threads) {
            const int r = e % H, j = e / H;
            T sum = ws[j * LDW + r];
#pragma unroll
            for (int layer = 1; layer < LAYERS; ++layer)
                sum += ws[size_t(layer) * BN * LDW + j * LDW + r];
            ws[j * LDW + r] = sum;
        }
        __syncthreads();
    }
    // -Z = -op(T) W into the registers, then over W
    {
        const int mt = warp & 3, n0 = (warp >> 2) * 2, m = mt * 16 + 2 * gq;
        const T *tseg = t + size_t(s) * h * h;
        T z[2][4] = {};
#pragma unroll
        for (int k0 = 0; k0 < H; k0 += 16) {
            const int kk = k0 + 4 * tq;
            T a[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int kr = kk + u;
                // op(T)(m, kr): T(kr, m) when transposed, T(m, kr) otherwise; zero outside h x h
                const T x0 = m < h && kr < h ? __ldg(transpose ? tseg + kr + size_t(m) * h
                                                               : tseg + m + size_t(kr) * h)
                                             : T(0);
                const T x1 = m + 1 < h && kr < h ? __ldg(transpose ? tseg + kr + size_t(m + 1) * h
                                                                   : tseg + m + 1 + size_t(kr) * h)
                                                 : T(0);
                a[2 * u] = x0;
                a[2 * u + 1] = x1;
            }
#pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                T b[4];
                tu_frag_b(ws + ((n0 + jn) * 8 + gq) * LDW + kk, b);
                Math::mma(z[jn], a, b);
            }
        }
        __syncthreads();
#pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            const int n = (n0 + jn) * 8 + 2 * tq;
            *reinterpret_cast<TreePair<T> *>(ws + n * LDW + m) =
                tree_make_pair(-z[jn][0], -z[jn][2]);
            *reinterpret_cast<TreePair<T> *>(ws + (n + 1) * LDW + m) =
                tree_make_pair(-z[jn][1], -z[jn][3]);
        }
    }
    __syncthreads();
    // X += V (-Z): each warp owns a 16 x 32 output tile. Sharing V across
    // all four column tiles was faster than reducing the accumulator count.
    for (int rt = warp; rt < r16 / 16; rt += C::threads / 32) {
        const int k = rt / 4, r0 = (rt % 4) * 16 + 2 * gq, rows = st.chunk_rows(s, k);
        // from zero, then one rounded addition to X (see tree_update_kernel)
        T d[4][4] = {};
        const int rr = min(r0, max(rows - 2, 0));
#pragma unroll
        for (int k0 = 0; k0 < H; k0 += 16) {
            const int kk = k0 + 4 * tq;
            T a[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int c = kk + u;
                TreePair<T> p = tree_make_pair(T(0), T(0));
                if (c < h && r0 < rows) {
                    const T *pv = vcol(k, c) + rr;
                    if (PAIRS && r0 + 1 < rows)
                        p = tux_ld2(pv);
                    else {
                        p.x = __ldg(vcol(k, c) + r0);
                        p.y = r0 + 1 < rows ? __ldg(vcol(k, c) + r0 + 1) : T(0);
                    }
                }
                a[2 * u] = p.x, a[2 * u + 1] = p.y;
            }
#pragma unroll
            for (int jn = 0; jn < 4; ++jn) {
                T b[4];
                tu_frag_b(ws + (jn * 8 + gq) * LDW + kk, b);
                Math::mma(d[jn], a, b);
            }
        }
        T *xc = x + st.x_row(s, k) + (long long)col0 * ldx;
#pragma unroll
        for (int jn = 0; jn < 4; ++jn) {
            const int n = jn * 8 + 2 * tq;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = r0 + (e >> 1), j = n + (e & 1);
                if (r < rows && col0 + j < q)
                    xc[r + size_t(j) * ldx] = xs[j * LDX + 64 * k + r] + d[jn][e];
            }
        }
    }
}
template <class T, class Math, int LAYERS, bool PAIRS>
__global__ __launch_bounds__(TreeUpdateX<T, LAYERS>::threads,
                             2) void tree_update_x_kernel(TreeStep st, const T *v, int ldv,
                                                          const T *t, T *x, int ldx, int q,
                                                          bool transpose) {
    extern __shared__ __align__(16) unsigned char tux_smem[];
    tree_update_x_tile<T, Math, LAYERS, PAIRS>(st, blockIdx.y,
                                               blockIdx.x * TreeUpdateX<T, LAYERS>::BN, v, ldv, t,
                                               x, ldx, q, transpose, tux_smem, threadIdx.x);
}
template <class T, class Math, int LAYERS>
Carrier launch_tree_update_x_cfg(const TreeStep &st, const T *v, int ldv, const T *t, T *x, int ldx,
                                 int q, bool transpose, cudaStream_t s) {
    using C = TreeUpdateX<T, LAYERS>;
    const bool pairs = ldv % 2 == 0 && ldx % 2 == 0 &&
                       reinterpret_cast<uintptr_t>(v) % (2 * sizeof(T)) == 0 &&
                       reinterpret_cast<uintptr_t>(x) % (2 * sizeof(T)) == 0 && st.aligned_pairs();
    auto go = [&](auto kernel) {
        reserve_shared_memory(kernel, C::smem);
        kernel<<<dim3(ceildiv(q, C::BN), st.segments), C::threads, C::smem, s>>>(st, v, ldv, t, x,
                                                                                 ldx, q, transpose);
        CU(cudaGetLastError());
    };
    if (pairs)
        go(tree_update_x_kernel<T, Math, LAYERS, true>);
    else
        go(tree_update_x_kernel<T, Math, LAYERS, false>);
    return Carrier{}
        .set(GpuLevel, st.segments, ceildiv(q, C::BN), 1)
        .set(BlockLevel, 4 / LAYERS, 2, LAYERS);
}

template <class T>
Carrier launch_tree_update_x(const TreeStep &st, const T *v, int ldv, const T *t, T *x, int ldx,
                             int q, bool transpose, int layers, cudaStream_t stream) {
    Carrier result;
    with_tree_math<T>([&](auto math) {
        using Math = decltype(math);
        result =
            layers >= 2
                ? launch_tree_update_x_cfg<T, Math, 2>(st, v, ldv, t, x, ldx, q, transpose, stream)
                : launch_tree_update_x_cfg<T, Math, 1>(st, v, ldv, t, x, ldx, q, transpose, stream);
    });
    return result;
}

template <class T, class Math, int L>
Carrier launch_tree_update_cfg(const TreeStep &st, const T *v, int ldv, const T *t, T *x, int ldx,
                               int q, bool transpose, cudaStream_t s, bool pairs) {
    using Cfg = TreeUpdateCfg<T, 64, L>;
    auto go = [&](auto kernel) {
        reserve_shared_memory(kernel, Cfg::smem);
        kernel<<<dim3(ceildiv(q, 64), st.segments), Cfg::threads, Cfg::smem, s>>>(
            st, v, ldv, t, x, ldx, q, transpose);
        CU(cudaGetLastError());
    };
    if (pairs)
        go(tree_update_kernel<T, Math, 64, L, 256, 2, true>);
    else
        go(tree_update_kernel<T, Math, 64, L, 256, 2, false>);
    using WT = TreeTiling<64, Cfg::layer_warps>;
    return Carrier{}
        .set(GpuLevel, st.segments, ceildiv(q, 64), 1)
        .set(BlockLevel, WT::WM, WT::WN, L);
}
// All arithmetic modes use the same tiling and bounded warp replication.
template <class T>
Carrier launch_tree_update(const TreeStep &st, const T *v, int ldv, const T *t, T *x, int ldx,
                           int q, bool transpose, int layers, cudaStream_t s) {
    const bool pairs = ldv % 2 == 0 && ldx % 2 == 0 &&
                       reinterpret_cast<uintptr_t>(v) % (2 * sizeof(T)) == 0 &&
                       reinterpret_cast<uintptr_t>(x) % (2 * sizeof(T)) == 0 && st.aligned_pairs();
    Carrier result;
    with_tree_math<T>([&](auto math) {
        using Math = decltype(math);
        result =
            layers >= 2
                ? launch_tree_update_cfg<T, Math, 2>(st, v, ldv, t, x, ldx, q, transpose, s, pairs)
                : launch_tree_update_cfg<T, Math, 1>(st, v, ldv, t, x, ldx, q, transpose, s, pairs);
    });
    return result;
}
// ---- the tree as one kernel ---------------------------------------------------------------------
// A block per GE/TS/TT node, admitted in ticket order (producers before consumers). A node factors
// its stack with its columns in registers (gx_body). Given q release columns (the next panel's,
// right of the panel in a), it then applies its elimination to its rows of them: the update that
// the next panel waits for, done as soon as the nodes below it have updated those rows, so that only
// the root's update follows the root's factorization. Release flags follow the hand-off flags.
// Before releasing, a node waits until *ready reaches ready_value: the release columns' update by
// the previous panel, which another stream applies meanwhile. The waiting blocks leave the stream's
// kernels free SMs as long as the tree has fewer nodes than the GPU has SMs.
template <class T, int LAYERS = 2> struct TreeRelease {
    static constexpr int layers = LAYERS, groups = 2,
                         threads = groups * TreeUpdateX<T, layers>::threads;
    static constexpr size_t group_smem = TreeUpdateX<T, layers>::smem;
};
// The update step of one node of t on its rows (segment 0 at the node's base).
inline __host__ __device__ TreeStep tree_node_step(const DomainTree &t, int level, int n,
                                                   int phase) {
    TreeStep st;
    st.h = t.h;
    st.segments = 1;
    if (level == 0) {
        st.domain = t.domain;
        st.last = t.domain_rows(n);
        st.tile = t.tile;
        st.phase = phase;
    } else {
        st.outer = t.stride[level];
        st.inner = t.stride[level - 1];
        st.fan_in = t.fan_in;
        st.last_children = t.children(level, n);
    }
    return st;
}
// A node's release of q columns at x (leading dimension ldx) by its V (leading dimension ldv) and T,
// out of line so that it does not take registers from the factorization.
template <class T, class Math, int LAYERS>
__device__ __noinline__ void tree_release(TreeStep st, const T *v, int ldv, const T *t, T *x,
                                          int ldx, int q, unsigned char *smem) {
    // the groups take column tiles in turn, all of them the same number of tiles
    using R = TreeRelease<T, LAYERS>;
    using X = TreeUpdateX<T, R::layers>;
    const int group = threadIdx.x / X::threads, tid = threadIdx.x % X::threads;
    for (int i = 0; i * R::groups * X::BN < q; ++i)
        tree_update_x_tile<T, Math, R::layers, false>(st, 0, (i * R::groups + group) * X::BN, v,
                                                      ldv, t, x, ldx, q, true,
                                                      smem + group * R::group_smem, tid);
}
template <class T, class Math>
__global__ __launch_bounds__(512, 1) void register_domain_tree_kernel(
    T *a, int lda, DomainTree t, T *vp, int ldp, T *vstack, T *tt, int *status, int *flags,
    unsigned *ticket, unsigned base, int epoch, T *retained_tau, int q, const int *ready,
    int ready_value, int domain_layers, int merge_layers) {
    extern __shared__ __align__(16) unsigned char register_domain_storage[];
    __shared__ int node_s;
    __shared__ RegisterTreeSource<T> src;
    if (threadIdx.x == 0)
        node_s = int(atomicAdd(ticket, 1u) - base);
    __syncthreads();
    const int node = node_s, h = t.h, blocks = (h + 3) / 4;
    const int domain_nodes = t.chain * t.count[0];
    int level = 0, n = node % t.count[0], phase = node / t.count[0];
    if (node >= domain_nodes) {
        level = 1;
        while (level + 1 < t.levels && node >= t.first[level + 1])
            ++level;
        n = node - t.first[level];
    }
    const bool leaf = level == 0 && phase == 0;
    const int tail = level == 0 && phase ? t.ts_rows(phase, n) : 0;
    int *released = flags + size_t(t.nodes()) * blocks;
    T *tn = tt + size_t(node) * h * h;
    T *taus = retained_tau ? retained_tau + size_t(node) * h : nullptr;
    const int ldv = leaf ? ldp : level ? t.vstack_rows(level) : t.ts_vrows();
    const int vrows = leaf ? t.ge_rows(n) : level ? t.fan_in * h : DomainConfig::rows;
    T *vn = leaf    ? vp + (long long)n * t.leaf_stride()
            : level ? vstack + t.vstack_offset(level) + size_t(n) * t.fan_in * h
                    : vstack + t.ts_offset(phase) + size_t(n) * DomainConfig::rows;
    if (level == 0 && phase && !tail) {
        for (int e = threadIdx.x; e < vrows * h; e += blockDim.x)
            vn[e % vrows + size_t(e / vrows) * ldv] = T(0);
        for (int e = threadIdx.x; e < h * h; e += blockDim.x)
            tn[e] = T(0);
        if (taus)
            for (int e = threadIdx.x; e < h; e += blockDim.x)
                taus[e] = T(0);
        // the R and the release of the domain stay with its last phase that factors rows
        if (threadIdx.x == 0)
            for (int p = 0; p < blocks; ++p) {
                ge_wait(flags + (node - t.count[0]) * blocks + p, epoch);
                ge_signal(flags + node * blocks + p, epoch);
            }
        return;
    }
    const int rows = leaf ? t.ge_rows(n) : level ? t.children(level, n) * h : h + tail;
    if (threadIdx.x == 0) {
        src = {};
        src.lda = lda;
        src.h = h;
        src.epoch = epoch;
        src.flags = flags + node * blocks;
        if (leaf) {
            src.pieces = 1;
            src.x[0] = a + (long long)n * t.domain;
            src.rows[0] = rows;
        } else if (level == 0) {
            src.pieces = 2;
            src.x[0] = a + (long long)n * t.domain;
            src.x[1] = src.x[0] + phase * t.tile;
            src.rows[0] = h;
            src.rows[1] = tail;
            src.triangle[0] = true;
            src.wait[0] = flags + (node - t.count[0]) * blocks;
        } else {
            src.pieces = t.children(level, n);
            const int first =
                (level == 1 ? (t.chain - 1) * t.count[0] : t.first[level - 1]) + n * t.fan_in;
            for (int p = 0; p < src.pieces; ++p) {
                src.x[p] = a + n * t.stride[level] + p * t.stride[level - 1];
                src.rows[p] = h;
                src.triangle[p] = true;
                src.wait[p] = flags + (first + p) * blocks;
            }
        }
    }
    __syncthreads();
    gx_body<T, DomainConfig::items, DomainConfig::vec, DomainConfig::width / 32>(
        src, rows, h, vn, ldv, vrows, taus, tn, h, status, register_domain_storage);
    if (q == 0)
        return;
    // Release: the rows this node updates were last updated by the release of the previous phase
    // (TS) or of the children (TT), whose V, T and rows are in global memory.
    if (threadIdx.x == 0 && ready)
        while (gau_load_acquire(ready) < ready_value)
            __nanosleep(64);
    if (threadIdx.x < (level ? t.children(level, n) : leaf ? 0 : 1)) {
        const int child = level == 0   ? node - t.count[0]
                          : level == 1 ? t.last_phase(n * t.fan_in + threadIdx.x) * t.count[0] +
                                             n * t.fan_in + threadIdx.x
                                       : t.first[level - 1] + n * t.fan_in + threadIdx.x;
        while (gau_load_acquire(released + child) != epoch)
            __nanosleep(32);
    }
    __syncthreads();
    const auto step = tree_node_step(t, level, n, phase);
    T *x = (level ? a + n * t.stride[level] : a + (long long)n * t.domain) + size_t(h) * lda;
    if ((level ? merge_layers : domain_layers) >= 2)
        tree_release<T, Math, 2>(step, vn, ldv, tn, x, lda, q, register_domain_storage);
    else
        tree_release<T, Math, 1>(step, vn, ldv, tn, x, lda, q, register_domain_storage);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        gau_store_release(released + node, epoch);
    }
}

// Ticket and flags of the tree kernel (hand-off flags of every warp of every node, then a release
// flag per node), and the epochs and ticket bases of its launches.
__global__ void tree_signal(int *flag, int value) {
    gau_store_release(flag, value);
}
struct TreeSync {
    Buffer<int> flags;
    Buffer<unsigned> ticket;
    unsigned base = 0;
    int epoch = 0;
    static size_t words(int nodes) {
        return size_t(nodes) * (DomainConfig::width / DomainConfig::vec + 1);
    }
    void reserve(int nodes, cudaStream_t stream) {
        if (flags.n < words(nodes)) {
            flags.alloc(words(nodes));
            // Initialize on the consumer's stream: a default-stream memset does not order
            // accesses from the nonblocking streams used by factorization and updates.
            flags.zero(stream);
            ticket.alloc(1);
            ticket.zero(stream);
            base = 0;
            epoch = 0;
        }
    }
};

// Factors the panel block of t.rows x t.h at a (leading dimension lda) as the tree t: the factors in
// place, the domains' reflectors also in vp (leading dimension ldp), the TT levels' stacked V in
// vstack (t.vstack_words()) and every T in tt (t.t_words()). With release > 0 it also applies the
// panel's Q^T to the next `release` columns of a, right of the panel (h = 64 only), once *ready
// (when given) reaches ready_value.
template <class T>
Carrier factor_domains(T *a, int lda, const DomainTree &t, T *vp, int ldp, T *vstack, T *tt,
                       int *status, TreeSync &sync, cudaStream_t st, T *retained_tau = nullptr,
                       int release = 0, const int *ready = nullptr, int ready_value = 0,
                       int domain_layers = 2, int merge_layers = 2) {
    if (t.h > DomainConfig::width)
        throw std::runtime_error("tree panels are at most 64 columns wide");
    const int nodes = t.nodes();
    if (sync.flags.n < TreeSync::words(nodes))
        throw std::runtime_error("a domain tree beyond its reserved hand-off flags");
    const int warps = ceildiv(t.h, DomainConfig::vec);
    if (release && warps * 32 != TreeRelease<T>::threads)
        throw std::runtime_error("a release needs a full-width tree panel");
    ++sync.epoch;
    size_t bytes = gx_smem<T>(DomainConfig::items, DomainConfig::vec, warps);
    if (release) {
        const int layers = std::max(domain_layers, t.levels > 1 ? merge_layers : 1);
        const size_t release_bytes =
            layers >= 2 ? TreeRelease<T, 2>::group_smem : TreeRelease<T, 1>::group_smem;
        bytes = std::max(bytes, TreeRelease<T>::groups * release_bytes);
    }
    with_tree_math<T>([&](auto math) {
        auto kernel = register_domain_tree_kernel<T, decltype(math)>;
        reserve_shared_memory(kernel, bytes);
        kernel<<<nodes, warps * 32, bytes, st>>>(
            a, lda, t, vp, ldp, vstack, tt, status, sync.flags.p, sync.ticket.p, sync.base,
            sync.epoch, retained_tau, release, ready, ready_value, domain_layers, merge_layers);
    });
    sync.base += unsigned(nodes);
    CU(cudaGetLastError());
    return Carrier{}.set(GpuLevel, t.count[0], 1, 1).set(BlockLevel, 1, warps, 1);
}
} // namespace tqr
