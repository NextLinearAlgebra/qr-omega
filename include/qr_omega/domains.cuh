#pragma once
// Hierarchical QR inside one GPU: Algorithm 1 at the GPU level. The rows of a panel are split into
// domains; a GE reduces the first tile, a flat TS chain absorbs the remaining dense tiles, and TT
// eliminations merge fan_in domain triangles at a time up a tree to R. Every elimination is a block
// of Householder reflectors with its own V and T, stored in place as PLASMA stores them: the V of a
// domain below its diagonal, the V of a TT elimination in the triangles it consumed, R in the
// triangle of the first domain. The trailing matrix is updated along the same tree
// (Updater::apply_tree): one segmented product per step updates the rows of every domain, and the W
// of the TT eliminations of a level sums the partial products of their children, which are the
// replication layers c of its carrier.
//
// The tree runs as one kernel, a block per GE/TS/TT node, each node factored with its columns in
// registers (gx_body): an elimination starts on a warp's columns as soon as its predecessors publish
// them. Ticket order keeps producers ahead of consumers.
#include "carrier_kami.cuh"
#include "panel_register.cuh"
#include <climits>

namespace tqr {
// A node holds at most `rows` = 32 items rows of a panel of at most `width` columns; its warps own
// `vec` columns each. A TT elimination merges at most `max_children` triangles.
struct DomainConfig {
    static constexpr int width = 64, items = 8, vec = 4;
    static constexpr int rows = 32 * items, max_children = 4;
};
inline constexpr int domain_max_levels = 16;

// ---- elimination tree ---------------------------------------------------------------------------
// Level 0 holds count[0] domains of `domain` rows, the last taking the remainder. Within a domain,
// phase 0 factors its first `tile` rows, and later phases absorb another tile by TSQRT. T factors
// are ordered by (phase, domain), followed by the TT levels. Node n of level l >= 1
// merges children [n fan_in, n fan_in + children(l, n)) of level l - 1; its triangle lies at row
// n stride[l], its children stride[l - 1] apart. first[l] numbers the T factors of the nodes.
struct DomainTree {
    int rows = 0, h = 0, domain = 0, fan_in = 2, levels = 1;
    int tile = 0, chain = 1; // GE of the first tile, then chain - 1 TS eliminations per domain
    int count[domain_max_levels] = {}, first[domain_max_levels] = {};
    long long stride[domain_max_levels] = {};
    __host__ __device__ int domain_rows(int i) const {
        return i + 1 < count[0] ? domain : rows - i * domain;
    }
    __host__ __device__ int children(int l, int n) const {
        return min(fan_in, count[l - 1] - n * fan_in);
    }
    __host__ __device__ int ge_rows(int n) const {
        return min(tile, domain_rows(n));
    }
    __host__ __device__ int leaf_stride() const {
        return min(tile, domain);
    }
    __host__ __device__ int ts_rows(int phase, int n) const {
        return max(0, min(tile, domain_rows(n) - phase * tile));
    }
    // The last phase of domain n that factors rows: its node holds the domain's R.
    __host__ __device__ int last_phase(int n) const {
        int p = 0;
        while (p + 1 < chain && ts_rows(p + 1, n) > 0)
            ++p;
        return p;
    }
    __host__ __device__ int ts_vrows() const {
        return count[0] * DomainConfig::rows;
    }
    __host__ __device__ long long ts_offset(int phase) const {
        return (long long)(phase - 1) * ts_vrows() * h;
    }
    __host__ __device__ int nodes() const {
        return levels == 1 ? chain * count[0] : first[levels - 1] + count[levels - 1];
    }
    // Words of the T factors and of the stacked V of the TT levels.
    size_t t_words() const {
        return size_t(nodes()) * h * h;
    }
    __host__ __device__ long long vstack_offset(int l) const {
        long long words = (long long)(chain - 1) * ts_vrows() * h;
        for (int i = 1; i < l; ++i)
            words += (long long)count[i] * fan_in * h * h;
        return words;
    }
    size_t vstack_words() const {
        return size_t(vstack_offset(levels));
    }
    // Rows of the stacked V of level l, its leading dimension.
    __host__ __device__ int vstack_rows(int l) const {
        return count[l] * fan_in * h;
    }
};
// The domain height for D carriers whose row tiles hold `tile` rows: the largest multiple of the
// tile that leaves room in a block for a remainder of up to h - 1 rows.
inline int domain_height(int tile, int h) {
    return (DomainConfig::rows - (h - 1)) / tile * tile;
}
inline DomainTree domain_tree(int rows, int h, int domain, int fan_in, int tiles = 1) {
    if (h < 1 || h > DomainConfig::width || rows < h || domain < h || domain > DomainConfig::rows ||
        fan_in < 2 || fan_in > DomainConfig::max_children || tiles < 1 ||
        (long long)domain * tiles > INT_MAX || (tiles > 1 && domain + h > DomainConfig::rows))
        throw std::runtime_error("invalid domain tree");
    if (fan_in * h > DomainConfig::rows)
        throw std::runtime_error("a TT elimination of " + std::to_string(fan_in) +
                                 " triangles of " + std::to_string(h) +
                                 " rows exceeds the rows of one block");
    DomainTree t;
    t.rows = rows;
    t.h = h;
    t.fan_in = fan_in;
    int d = tiles == 1 && rows <= DomainConfig::rows ? rows : domain * tiles;
    int D = ceildiv(rows, d);
    if (D > 1 && rows - (D - 1) * d < h)
        --D; // the last domain takes the short remainder
    t.domain = d;
    t.count[0] = D;
    t.tile = tiles == 1 ? std::max(d, t.domain_rows(D - 1)) : domain;
    t.chain = tiles == 1 ? 1 : ceildiv(std::max(std::min(rows, d), t.domain_rows(D - 1)), domain);
    if (tiles == 1 && t.domain_rows(D - 1) > DomainConfig::rows)
        throw std::runtime_error("the final domain and its remainder exceed the rows of one block");
    t.stride[0] = d;
    for (t.levels = 1; t.count[t.levels - 1] > 1; ++t.levels) {
        if (t.levels == domain_max_levels)
            throw std::runtime_error("too many levels in a domain tree");
        const int l = t.levels;
        t.count[l] = ceildiv(t.count[l - 1], fan_in);
        t.first[l] = l == 1 ? D * t.chain : t.first[l - 1] + t.count[l - 1];
        t.stride[l] = t.stride[l - 1] * fan_in;
    }
    return t;
}

// ---- asynchronous copies to shared memory --------------------------------------------------------
__device__ __forceinline__ void tu_cp8(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(
                     unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src), "r"(valid ? 8 : 0)
                 : "memory");
}
__device__ __forceinline__ void tu_cp16(void *dst, const void *src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                     unsigned(__cvta_generic_to_shared(dst))),
                 "l"(src), "r"(bytes)
                 : "memory");
}
__device__ __forceinline__ void tu_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N> __device__ __forceinline__ void tu_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Hand-off of R between the GEs of a tree: every node publishes its blocks of R columns with an epoch.
__device__ __forceinline__ void ge_signal(int *flag, int epoch) {
    __threadfence();
    gau_store_release(flag, epoch);
}
__device__ __forceinline__ void ge_wait(const int *flag, int epoch) {
    while (gau_load_acquire(flag) != epoch)
        __nanosleep(32);
}

// ---- the tree as one kernel ---------------------------------------------------------------------
// The pieces of a node's stack (a domain's tile, a TS step's triangle and tile, a TT elimination's
// triangles) with the flags of their producers, which publish four columns at a time, the columns
// of a warp. The ticket scheduler admits GE producers before TS/TT consumers.
template <class T> struct RegisterTreeSource {
    static constexpr bool late = true; // a TS/TT node's columns arrive as its producers finish them
    T *x[DomainConfig::max_children];
    int rows[DomainConfig::max_children];
    bool triangle[DomainConfig::max_children];
    const int *wait[DomainConfig::max_children];
    int pieces, lda, h, epoch;
    int *flags;
    __device__ void acquire(int warp, int lane) const {
        if (lane == 0)
            for (int p = 0; p < pieces; ++p)
                if (wait[p])
                    ge_wait(wait[p] + warp, epoch);
        __syncwarp();
    }
    // Row r of the stack at column 0 and its first stored column (a triangle holds row r from
    // column r on); null past the stack.
    __device__ T *row(int r, int stack_rows, int &first) const {
        first = 0;
        if (r >= stack_rows)
            return nullptr;
#pragma unroll
        for (int p = 0; p < DomainConfig::max_children; ++p) {
            if (p < pieces && r < rows[p]) {
                first = triangle[p] ? r : 0;
                return x[p] + r;
            }
            r -= p < pieces ? rows[p] : 0;
        }
        return nullptr;
    }
    // The release store orders the warp's writes, which __syncwarp orders before it.
    __device__ void publish(int warp, int lane) const {
        __syncwarp();
        if (lane == 0)
            gau_store_release(flags + warp, epoch);
    }
};

// Rebuild a panel's compact WY factors from the in-place Householder vectors and
// retained taus. Only the two panels in flight need full T/V workspaces. This is
// dlarft on V^T V, not a Gram-based factorization of the input matrix.
template <class T>
__global__ __launch_bounds__(512) void restore_tree_kernel(const T *a, int lda, DomainTree t,
                                                           const T *tau, T *vp, int ldp, T *vstack,
                                                           T *tt) {
    constexpr int H = DomainConfig::width, R = DomainConfig::rows, LD = H + 1;
    extern __shared__ __align__(16) unsigned char storage[];
    T *v = reinterpret_cast<T *>(storage), *sm = v + R * H, *scratch = sm + H * LD;
    const int node = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, h = t.h;
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
    const int rows = leaf ? t.ge_rows(n) : level ? t.children(level, n) * h : tail ? h + tail : 0;
    const int packed_rows = leaf ? rows : level ? t.fan_in * h : R;
    const int ldout = leaf ? ldp : level ? t.vstack_rows(level) : t.ts_vrows();
    T *out = leaf    ? vp + (long long)n * t.leaf_stride()
             : level ? vstack + t.vstack_offset(level) + (long long)n * t.fan_in * h
                     : vstack + t.ts_offset(phase) + (long long)n * R;
    for (int e = tid; e < R * h; e += 512) {
        const int r = e % R, c = e / R;
        T y = T(0);
        if (r < rows) {
            if (leaf)
                y = r < c ? T(0) : r == c ? T(1) : a[(long long)n * t.domain + r + size_t(c) * lda];
            else if (r < h)
                y = r == c ? T(1) : T(0);
            else if (level) {
                const int child = r / h, rr = r % h;
                if (rr <= c)
                    y = a[n * t.stride[level] + child * t.stride[level - 1] + rr + size_t(c) * lda];
            } else
                y = a[(long long)n * t.domain + phase * t.tile + r - h + size_t(c) * lda];
        }
        v[r + size_t(c) * R] = y;
        if (r < packed_rows)
            out[r + size_t(c) * ldout] = y;
    }
    for (int e = tid; e < H * LD; e += 512)
        sm[e] = T(0);
    __syncthreads();
    if (tid < h)
        sm[tid * LD + tid] = tau[size_t(node) * h + tid];
    __syncthreads();
    // A warp owns one overlap. The contraction stays within the warp; all
    // sixteen warps own distinct output entries, without replication.
    for (int e = warp; e < h * h; e += 16) {
        const int i = e % h, j = e / h;
        if (i < j) {
            double sum = 0;
            for (int r = lane; r < rows; r += 32)
                sum = fma(double(v[r + size_t(i) * R]), double(v[r + size_t(j) * R]), sum);
            sum = gau_warp_sum(sum);
            if (lane == 0)
                sm[j + i * LD] = T(sum);
        }
    }
    __syncthreads();
    gau_build_t<T, H>(sm, scratch, tid);
    T *tn = tt + size_t(node) * h * h;
    for (int e = tid; e < h * h; e += 512) {
        const int i = e % h, j = e / h;
        tn[e] = i <= j ? sm[i + j * LD] : T(0);
    }
}
template <class T>
void restore_tree(const T *a, int lda, const DomainTree &tree, const T *tau, T *vp, int ldp,
                  T *vstack, T *tt, cudaStream_t stream) {
    const size_t bytes = DomainConfig::rows * DomainConfig::width * sizeof(T) +
                         gau_build_t_h_smem<T, DomainConfig::width>();
    reserve_shared_memory(restore_tree_kernel<T>, bytes);
    restore_tree_kernel<T>
        <<<tree.nodes(), 512, bytes, stream>>>(a, lda, tree, tau, vp, ldp, vstack, tt);
    CU(cudaGetLastError());
}

// Flops of the Householder QR of a rows x h block.
inline double ge_flops(int rows, int h) {
    return 2.0 * rows * h * h - 2.0 / 3.0 * h * h * h;
}
// Flops of the eliminations of a tree: its domains and its TT stacks.
inline double tree_flops(const DomainTree &t) {
    double f = 0;
    for (int i = 0; i < t.count[0]; ++i) {
        f += ge_flops(t.ge_rows(i), t.h);
        for (int phase = 1; phase < t.chain; ++phase)
            if (t.ts_rows(phase, i))
                f += ge_flops(t.h + t.ts_rows(phase, i), t.h);
    }
    for (int l = 1; l < t.levels; ++l)
        for (int n = 0; n < t.count[l]; ++n)
            f += ge_flops(t.children(l, n) * t.h, t.h);
    return f;
}
} // namespace tqr
