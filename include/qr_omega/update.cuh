#pragma once
// The update of Algorithm 2 on one strip of columns: W = V^T X, Z = T^T W and X -= V Z. Every
// product runs on a carrier within the 2.5D bound c^2 <= p_i p_j at every level of the GPU and
// records it, with its flops, in the tally.
#include "carrier_block.cuh"
#include "carrier_kami.cuh"
#include "carrier_simt.cuh"
#include "carrier_skinny.cuh"
#include "carrier_vendor.cuh"
#include "carrier_wgmma.cuh"
#include "domains.cuh"
#include "tree_update.cuh"
#include <functional>

namespace tqr {
// Requested replications c of the products, each lowered per product to the 2.5D bound. W reaches
// its c with the two slices of a block and c / 2 groups of blocks; Z and D always split the
// contraction between the two slices of a block. Inside a GPU's elimination tree the W of the
// domains splits each domain's rows among `domain` layers, and the W of a TT elimination deals its
// children to `merge` layers: warp layers of the fused kernel, groups of blocks on the
// segmented carriers.
struct Replication {
    int w = 2, z = 2, domain = 2, merge = 2, tree_groups = 1;
    static constexpr int d = 2;
};

// The CUTLASS carriers of the TF32 modes split a k-tile of 32 between two slices of 16, so a
// contraction shorter than a k-tile would leave a slice empty. Those products run on the block
// carrier, whose two groups can own intervals as short as one index.
template <class T> bool short_carrier_path(int h, bool d) {
    if constexpr (!std::is_same_v<T, float>)
        return false;
    else
        return fp32_math() != Fp32Math::IEEE && h >= 2 &&
               !(d ? carrier_d_supported<T>(h) : carrier_z_supported<T>(h));
}
template <class T> bool native_d_admits(int h) {
    return carrier_d_supported<T>(h) || short_carrier_path<T>(h, true);
}

// Flops of an m x n product with a contraction of k.
inline double product_flops(int m, int n, int k) {
    return 2.0 * m * n * k;
}

// Stream and buffers of one strip in flight.
template <class T> struct Lane {
    cudaStream_t stream = nullptr;
    T *w = nullptr, *z = nullptr; // W followed by its partial slices, and Z
    size_t w_words = 0;
    T *vr = nullptr; // FP32: the row-major or split copy of V read by the wgmma kernels
    // Far update under look-ahead: its wgmma products release their SMs tile by tile to the panel.
    bool far = false;
    // What `vr` holds when it is the row-major copy of an aggregated V.
    const T *packed = nullptr;
    int packed_rows = 0, packed_h = 0;
    uint64_t packed_version = 0;
    // Updates along an elimination tree: the W of every segment (domain or TT elimination), the
    // partials of its groups and its Z^T.
    T *tw = nullptr, *tp = nullptr, *tz = nullptr, *tx = nullptr;
    // W summed over the GPUs of a grid column, when the reflectors' rows are spread over them.
    T *reduced = nullptr;
};

template <class T> class Updater {
    int b, aggregate, strip, max_rows, d_tiles;
    Replication c;
    bool resident_tree;
    Tally &tally;
    // The node-level cuts of the products being issued: of W and every other product, and of Z
    // and D, which differ from W's when the rows of the reflectors are spread over a grid column.
    Cut node, node_z, node_d;
    Buffer<T> w_, z_, vr_, tree_w_, tree_p_, tree_z_, tree_x_;
    // Bounds of the tree buffers: segments, partial slices and columns of one pass.
    int tree_segments = 0, tree_slices = 0, tree_strip = 0;
    std::vector<Lane<T>> lanes_;
    VendorProduct<T> vendor;

    // wgmma products with fewer output tiles x slices than this run on the CUTLASS carriers,
    // whose smaller tiles admit more groups within the bound.
    static constexpr int gmma_min_blocks = 16;
    // FP64 and FP32 products A^T B with at least this many outputs run as cuBLASLt peers.
    static constexpr long long vendor_min_outputs = 128 * 1024;

    // Reflector columns one update can carry, the leading dimension of T, W and Z.
    int width() const {
        return b * aggregate;
    }
    size_t vr_words() const {
        const size_t packed = size_t(max_rows) * width();
        return fp32_math() == Fp32Math::X3 ? 3 * packed + 3 * size_t(width()) * strip : packed;
    }
    static bool vendor_admits(int m, int n) {
        return (std::is_same_v<T, double> || fp32_math() == Fp32Math::IEEE) &&
               (long long)m * n >= vendor_min_outputs;
    }

    // Whether D runs on wgmma, which reads V row-major from the lane and Z as it is.
    bool d_on_wgmma(const Lane<T> &lane, T *x, int ldx, int rows, int h, int q, int ld) const {
        if constexpr (std::is_same_v<T, float>) {
            if (!lane.vr || !native_d_admits<T>(h))
                return false;
            if (fp32_math() == Fp32Math::TF32)
                return h <= width() &&
                       gmma_d_admits(lane.vr, width(), lane.z, ld, x, ldx, rows, h, q);
            if (fp32_math() == Fp32Math::X3) {
                const int hs = gmma_x3_seg(h);
                return hs <= width() && q <= strip &&
                       gmma_d_admits(lane.vr, 3 * hs, lane.vr + 3 * size_t(max_rows) * width(),
                                     3 * hs, x, ldx, rows, 3 * hs, q);
            }
        }
        return false;
    }
    void pack_v(Lane<T> &lane, const T *v, int ldv, int rows, int h, int hs, int ldvr, bool x3,
                uint64_t version) {
        if (version && lane.packed == v && lane.packed_rows == rows && lane.packed_h == h &&
            lane.packed_version == version)
            return;
        if constexpr (std::is_same_v<T, float>)
            launch_pack_vr(v, ldv, rows, h, hs, lane.vr, ldvr, x3, lane.stream);
        lane.packed = v;
        lane.packed_rows = rows;
        lane.packed_h = h;
        lane.packed_version = version;
    }
    // W of a strip of the trailing matrix on the lane's buffers. TF32 and 3xTF32 run on wgmma when
    // the shape admits it with enough blocks within the bound; the rest on the carriers of
    // stage_w, with partial slices in the lane's W buffer.
    Carrier strip_w(Lane<T> &lane, const T *v, int ldv, const T *x, int ldx, int rows, int h, int q,
                    int ld) {
        T *w = lane.w;
        cudaStream_t st = lane.stream;
        const long long slice = (long long)ld * q;
        const int room = int(lane.w_words / std::max<size_t>(1, size_t(slice))) - 1;
        Carrier carrier;
        if (vendor_admits(h, q) && vendor.atb(st, v, ldv, x, ldx, w, ld, w, slice, rows, h, q,
                                              std::min(c.w, room + 1), carrier))
            return carrier;
        if constexpr (std::is_same_v<T, float>) {
            const int tiles_per_block = lane.far ? 1 : 0;
            const long long tiles = gmma_tiles(h, q);
            float *part = w + slice;
            if (fp32_math() == Fp32Math::TF32) {
                const int cg = bounded_groups(tiles, std::max(1, c.w / 2));
                if (tiles * cg >= gmma_min_blocks &&
                    gmma_w_admits(2 * cg, v, ldv, x, ldx, w, ld, rows, h, q, part, slice)) {
                    launch_gmma_w(2 * cg, v, ldv, x, ldx, w, ld, rows, h, q, st, part, slice,
                                  tiles_per_block);
                    return gmma_carrier(h, q, cg);
                }
            }
            if (fp32_math() == Fp32Math::X3) {
                const int cg = bounded_groups(tiles, std::max(1, c.w));
                if (tiles * cg >= gmma_min_blocks &&
                    gmma_w3rs_admits(cg, v, ldv, x, ldx, w, ld, rows, h, q, lane.vr)) {
                    // The split V shares the lane's buffer with the row-major copy D packs
                    // afterwards.
                    lane.packed_version = 0;
                    launch_gmma_w3rs(cg, v, ldv, x, ldx, w, ld, rows, h, q, st, lane.vr,
                                     tiles_per_block);
                    return gmma_w3rs_carrier(h, q, cg);
                }
            }
        }
        return stage_w(lane, v, ldv, x, ldx, w, ld, rows, h, q, w + slice, room);
    }

  public:
    // `rows` bounds the rows of an update and `b aggregate` its reflectors; `streams` are the
    // lanes.
    Updater(int b_, int aggregate_, int strip_, int rows, Replication c_, int d_tiles_,
            const std::vector<cudaStream_t> &streams, int first_far, Tally &tally_,
            bool resident_tree_ = false)
        : b(b_), aggregate(aggregate_), strip(strip_), max_rows(rows), d_tiles(d_tiles_), c(c_),
          resident_tree(resident_tree_), tally(tally_), lanes_(streams.size()) {
        const size_t n = streams.size(), block = size_t(width()) * strip;
        w_.alloc(n * c.w * block);
        z_.alloc(n * c.z * block);
        if (gmma_enabled<T>())
            vr_.alloc(n * vr_words());
        for (size_t i = 0; i < n; ++i) {
            Lane<T> &lane = lanes_[i];
            lane.stream = streams[i];
            lane.w = w_.p + i * c.w * block;
            lane.w_words = c.w * block;
            lane.z = z_.p + i * c.z * block;
            lane.vr = vr_.n ? vr_.p + i * vr_words() : nullptr;
            lane.far = first_far >= 0 && int(i) >= first_far;
        }
    }
    Lane<T> &lane(int i) {
        return lanes_[i];
    }
    json workspace() const {
        const size_t w = w_.n * sizeof(T), z = z_.n * sizeof(T), packed = vr_.n * sizeof(T),
                     tw = tree_w_.n * sizeof(T), tp = tree_p_.n * sizeof(T),
                     tz = tree_z_.n * sizeof(T), tx = tree_x_.n * sizeof(T);
        return {{"strip_lanes", lanes_.size()},
                {"requested_replication",
                 {{"w", c.w},
                  {"z", c.z},
                  {"domain", c.domain},
                  {"merge", c.merge},
                  {"tree_groups", c.tree_groups}}},
                {"w_bytes", w},
                {"z_bytes", z},
                {"packed_operand_bytes", packed},
                {"tree_w_bytes", tw},
                {"tree_partials_bytes", tp},
                {"tree_z_bytes", tz},
                {"tree_rows_bytes", tx},
                {"total_bytes", w + z + packed + tw + tp + tz + tx}};
    }
    // Buffers of updates along elimination trees of at most `segments` segments of h reflectors.
    // A pass covers at most the columns whose W, partials and Z^T fit `budget` bytes per buffer.
    void reserve_trees(int segments, int h, int ts_height = 0, size_t budget = size_t(256) << 20) {
        const int slices = segments * std::max({1, c.domain, c.merge, c.tree_groups}) + 8;
        const size_t per_column =
            std::max(size_t(slices) * h, size_t(segments) * ts_height) * sizeof(T);
        tree_strip = int(std::clamp<size_t>(budget / per_column / 64 * 64, 64, size_t(strip)));
        tree_segments = segments;
        tree_slices = slices;
        const size_t n = lanes_.size(), words = size_t(slices) * h * tree_strip;
        tree_w_.alloc(n * words);
        tree_p_.alloc(n * words);
        tree_z_.alloc(n * words);
        const size_t xwords = size_t(segments) * ts_height * tree_strip;
        if (xwords)
            tree_x_.alloc(n * xwords);
        for (size_t i = 0; i < n; ++i) {
            lanes_[i].tw = tree_w_.p + i * words;
            lanes_[i].tp = tree_p_.p + i * words;
            lanes_[i].tz = tree_z_.p + i * words;
            lanes_[i].tx = xwords ? tree_x_.p + i * xwords : nullptr;
        }
    }
    // The node-level cut of the products issued from now on.
    void at_node(Cut cut) {
        node = node_z = node_d = cut;
    }
    // The same, with Z and D on cuts of their own.
    void at_node(Cut w, Cut z, Cut d) {
        node = w;
        node_z = z;
        node_d = d;
    }
    void record(Product p, Carrier carrier, double flops) {
        carrier.at[NodeLevel] = p == ProductZ ? node_z : p == ProductD ? node_d : node;
        tally.add(p, carrier, flops);
    }
    // The three products with explicit operands. `ld` is the leading dimension of T, W and Z. W
    // can split its rows across groups of blocks when `part` has room for `room` partial slices.
    Carrier stage_w(Lane<T> &lane, const T *v, int ldv, const T *x, int ldx, T *w, int ld, int rows,
                    int h, int q, T *part = nullptr, int room = 0) {
        cudaStream_t st = lane.stream;
        if (rows < 2 * carrier_w_rows)
            return launch_carried_W<T>(v, ldv, x, ldx, w, ld, rows, h, q, st);
        if ((long long)h * q <= skinny_max_outputs)
            return launch_skinny_atb<T>(v, ldv, x, ldx, w, ld, part, (long long)ld * q,
                                        part ? room : 0, rows, h, q, st);
        return launch_carrier_w<T>(v, ldv, x, ldx, w, ld, rows, h, q, st,
                                   carrier_w_plan<T>(c.w, q, h, rows, part ? room : 1), part);
    }
    // Z = op(T) W, written transposed (q x h) when `transposed`. Z reaches its c with the two
    // slices of a block and c / 2 groups of blocks splitting the contraction, bounded by the 2.5D
    // bound of its tiles and by one k-tile per group; their partial slices follow Z in its buffer.
    Carrier stage_z(Lane<T> &lane, const T *t, const T *w, T *z, int ld, int h, int q,
                    bool transpose, bool transposed) {
        cudaStream_t st = lane.stream;
        const int ldz = transposed ? q : ld;
        if (carrier_z_supported<T>(h)) {
            int bk, wk;
            carrier_z_law<T>(bk, wk);
            const long long slice = (long long)ldz * (transposed ? h : q);
            const long long tiles = (long long)ceildiv(h, 64) * ceildiv(q, 64);
            const int groups =
                bounded_groups(tiles, std::max(1, std::min(c.z / 2, h / std::max(1, bk))));
            return launch_carrier_z<T>(t, ld, w, ld, z, ldz, h, q, transpose, transposed, st, 1, 0,
                                       0, 0, groups, groups > 1 ? z + slice : nullptr, slice);
        }
        if (short_carrier_path<T>(h, false))
            return transposed ? launch_short_carrier<T, false, true>(t, ld, w, ld, z, ldz, h, h, q,
                                                                     transpose, st)
                              : launch_short_carrier<T, false>(t, ld, w, ld, z, ldz, h, h, q,
                                                               transpose, st);
        return launch_carried_Z<T>(t, ld, w, ld, z, ldz, h, q, transpose, st, transposed);
    }
    // X -= V Z. Z is read transposed whenever the contraction admits two slices.
    Carrier stage_d(Lane<T> &lane, const T *v, int ldv, const T *z, int ld, T *x, int ldx, int rows,
                    int h, int q) {
        cudaStream_t st = lane.stream;
        if (!native_d_admits<T>(h))
            return launch_carried_D<T>(v, ldv, z, ld, x, ldx, rows, h, q, st);
        if (short_carrier_path<T>(h, true))
            return launch_short_carrier<T, true>(v, ldv, z, q, x, ldx, rows, h, q, false, st);
        if constexpr (std::is_same_v<T, double>) {
            // Aggregated updates: the (2, 2, 2) block carrier with FP64 tensor-core MMA.
            if (h >= 384 && h % 32 == 0 && kami::d_admits(v, ldv, z, q, x, ldx, h))
                return kami::launch_dnp16<kami::DN128x64K16>(v, ldv, z, q, x, ldx, rows, h, q, st);
        } else if (fp32_math() == Fp32Math::IEEE) {
            // The two warp layers of a block own the contraction halves.
            if (h >= 256 && kami::simt_d2_admits<kami::SF2K16>(v, ldv, z, q, x, ldx, h))
                return kami::launch_simt_d2<kami::SF2K16>(v, ldv, z, q, x, ldx, rows, h, q, st);
            if (kami::simt_d2_admits<kami::SF3>(v, ldv, z, q, x, ldx, h))
                return kami::launch_simt_d2<kami::SF3>(v, ldv, z, q, x, ldx, rows, h, q, st);
        }
        return launch_carrier_d<T>(v, ldv, z, q, x, ldx, rows, h, q, st);
    }
    // X(rows x q) <- (I - V op(T) V^T) X for V (rows x h) and its triangular factor T, whose
    // leading dimension is max(b, h). `version`, when nonzero, identifies the contents of an
    // aggregated V so that its row-major copy is packed once per lane.
    // When the rows of V are spread over the GPUs of a grid column, `reduce` sums the partial W of
    // the lane over them (in place or into a buffer it returns) between W and Z; a GPU without rows
    // of V contributes zeros and stops there.
    using Reduce = std::function<const T *(Lane<T> &, T *, size_t)>;
    void apply(Lane<T> &lane, T *x, int ldx, const T *v, int ldv, const T *t, int rows, int h,
               int q, bool transpose, uint64_t version = 0, const Reduce &reduce = {}) {
        const int ld = std::max(b, h);
        cudaStream_t st = lane.stream;
        if (rows <= 0) {
            if (reduce) {
                CU(cudaMemsetAsync(lane.w, 0, size_t(ld) * q * sizeof(T), st));
                reduce(lane, lane.w, size_t(ld) * q);
            }
            return;
        }
        record(ProductW, strip_w(lane, v, ldv, x, ldx, rows, h, q, ld), product_flops(h, q, rows));
        const T *w = reduce ? reduce(lane, lane.w, size_t(ld) * q) : lane.w;
        const bool wgmma = d_on_wgmma(lane, x, ldx, rows, h, q, ld);
        const bool transposed = !wgmma && native_d_admits<T>(h);
        bool z_done = false;
        if constexpr (std::is_same_v<T, float>) {
            // The wgmma Z has no groups: it serves when Z's replication is the block's own.
            if (wgmma && transpose && c.z <= 2 && gmma_z_admits(t, ld, w, ld, lane.z, ld, h, q)) {
                launch_gmma_z(t, ld, w, ld, lane.z, ld, h, q, st, lane.far ? 1 : 0);
                record(ProductZ, gmma_carrier(h, q, 1), product_flops(h, q, h) / 2);
                z_done = true;
            }
        }
        if (!z_done)
            record(ProductZ, stage_z(lane, t, w, lane.z, ld, h, q, transpose, transposed),
                   product_flops(h, q, h) / 2);
        if (!wgmma) {
            record(ProductD, stage_d(lane, v, ldv, lane.z, ld, x, ldx, rows, h, q),
                   product_flops(rows, q, h));
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            const int tiles = lane.far ? d_tiles : 0;
            if (fp32_math() == Fp32Math::X3) {
                const int hs = gmma_x3_seg(h);
                float *z3 = lane.vr + 3 * size_t(max_rows) * width();
                pack_v(lane, v, ldv, rows, h, hs, 3 * hs, true, version);
                pack_z3<<<pack_vr_grid((long long)q * hs), 256, 0, st>>>(lane.z, ld, h, hs, q, z3,
                                                                         3 * hs);
                CU(cudaGetLastError());
                launch_gmma_d(lane.vr, 3 * hs, z3, 3 * hs, x, ldx, rows, 3 * hs, q, st, tiles);
            } else {
                pack_v(lane, v, ldv, rows, h, h, width(), false, version);
                launch_gmma_d(lane.vr, width(), lane.z, ld, x, ldx, rows, h, q, st, tiles);
            }
            record(ProductD, gmma_carrier(rows, q, 1), product_flops(rows, q, h));
        }
    }
    // G(h1 x h2) = V1^T V2 over `rows` rows, the product behind every composed T, with its rows
    // split across groups of blocks within the bound; their slices go to `partials`.
    void gram(cudaStream_t st, const T *v1, int ld1, const T *v2, int ld2, T *g, int ldg, int rows,
              int h1, int h2, T *partials, size_t partial_words) {
        const long long slice = (long long)ldg * h2;
        const int room = int(std::min<size_t>(128, partial_words / size_t(slice)));
        Carrier carrier;
        bool done = false;
        if (rows < 2 * carrier_w_rows) {
            carrier = launch_carried_W<T>(v1, ld1, v2, ld2, g, ldg, rows, h1, h2, st);
            done = true;
        }
        if (!done && (long long)h1 * h2 <= skinny_max_outputs) {
            carrier = launch_skinny_atb<T>(v1, ld1, v2, ld2, g, ldg, partials, slice, room, rows,
                                           h1, h2, st);
            done = true;
        }
        if (!done && vendor_admits(h1, h2))
            done = vendor.atb(st, v1, ld1, v2, ld2, g, ldg, partials, slice, rows, h1, h2,
                              std::min(c.w, room), carrier);
        if constexpr (std::is_same_v<T, float>) {
            if (!done && fp32_math() == Fp32Math::TF32) {
                const long long tiles = gmma_tiles(h1, h2);
                const int cg =
                    std::min(std::max(1, room), bounded_groups(tiles, std::max(1, c.w / 2)));
                if (tiles * cg >= gmma_min_blocks && gmma_w_admits(2 * cg, v1, ld1, v2, ld2, g, ldg,
                                                                   rows, h1, h2, partials, slice)) {
                    launch_gmma_w(2 * cg, v1, ld1, v2, ld2, g, ldg, rows, h1, h2, st, partials,
                                  slice, 0);
                    carrier = gmma_carrier(h1, h2, cg);
                    done = true;
                }
            }
        }
        if (!done)
            carrier = launch_carrier_w<T>(v1, ld1, v2, ld2, g, ldg, rows, h1, h2, st,
                                          carrier_w_plan<T>(c.w, h2, h1, rows, room), partials);
        record(ProductGram, carrier, product_flops(h1, h2, rows));
    }

    // ---- elimination trees inside the GPU ------------------------------------------------------
    // Q^T X (transpose) or Q X for the tree t of a panel, whose domains' reflectors are vp (leading
    // dimension ldp), TS/TT stacked V vstack and T factors tt: first GE, then the TS chain,
    // then TT levels on their children's triangles (the reverse order for Q X).
    void apply_tree(Lane<T> &lane, T *x, int ldx, const DomainTree &t, const T *vp, int ldp,
                    const T *vstack, const T *tt, int q, bool transpose) {
        if (t.count[0] > tree_segments)
            throw std::runtime_error("an elimination tree beyond the reserved segments");
        for (int c0 = 0; c0 < q; c0 += tree_strip) {
            const int qq = std::min(tree_strip, q - c0);
            T *xc = x + size_t(c0) * ldx;
            if (transpose)
                tree_domains(lane, xc, ldx, t, vp, ldp, tt, qq, true);
            if (transpose)
                for (int phase = 1; phase < t.chain; ++phase)
                    tree_ts(lane, xc, ldx, t, phase, vstack, tt, qq, true);
            for (int i = 1; i < t.levels; ++i)
                tree_level(lane, xc, ldx, t, transpose ? i : t.levels - i, vstack, tt, qq,
                           transpose);
            if (!transpose) {
                for (int phase = t.chain - 1; phase > 0; --phase)
                    tree_ts(lane, xc, ldx, t, phase, vstack, tt, qq, false);
                tree_domains(lane, xc, ldx, t, vp, ldp, tt, qq, false);
            }
        }
    }

    // The release has two column groups per block. Domain and TT nodes use their requested warp
    // layers, bounded to two as in the separate update. Count only active TS nodes.
    void record_tree_release(const DomainTree &t, int q) {
        double rows[2] = {};
        int nodes[2] = {t.count[0], 0};
        for (int i = 0; i < t.count[0]; ++i) {
            rows[0] += t.ge_rows(i);
            for (int phase = 1; phase < t.chain; ++phase)
                if (t.ts_rows(phase, i)) {
                    rows[0] += t.h + t.ts_rows(phase, i);
                    ++nodes[0];
                }
        }
        for (int l = 1; l < t.levels; ++l)
            for (int n = 0; n < t.count[l]; ++n) {
                rows[1] += double(t.children(l, n)) * t.h;
                ++nodes[1];
            }
        for (int kind = 0; kind < 2; ++kind) {
            if (!nodes[kind])
                continue;
            const int layers = std::clamp(kind ? c.merge : c.domain, 1, 2);
            constexpr int groups = TreeRelease<T>::groups;
            Carrier carrier = Carrier{}
                                  .set(GpuLevel, t.nodes(), 1, 1)
                                  .set(BlockLevel, 4 / layers, 2 * groups, layers);
            record(ProductW, carrier, 2.0 * rows[kind] * q * t.h);
            carrier.set(BlockLevel, 4, 2 * groups, 1);
            record(ProductZ, carrier, nodes[kind] * product_flops(t.h, q, t.h) / 2);
            carrier.set(BlockLevel, 8, groups, 1);
            record(ProductD, carrier, 2.0 * rows[kind] * q * t.h);
        }
    }

  private:
    void tree_step(const TreeStep &step, const T *v, int ldv, const T *t, T *x, int ldx, int rows,
                   int q, bool transpose, int layers, cudaStream_t st) {
        Carrier carrier =
            resident_tree ? launch_tree_update_x(step, v, ldv, t, x, ldx, q, transpose, layers, st)
                          : launch_tree_update(step, v, ldv, t, x, ldx, q, transpose, layers, st);
        record(ProductW, carrier, product_flops(rows, q, step.h));
        // Warp layers split W's row contraction. Z and D tile their outputs without replication;
        // credit only the product that actually executes a replicated contraction.
        carrier.set(BlockLevel, resident_tree ? 4 : 2, resident_tree ? 2 : 4, 1);
        record(ProductZ, carrier, step.segments * product_flops(step.h, q, step.h) / 2);
        if (resident_tree)
            carrier.set(BlockLevel, 8, 1, 1);
        record(ProductD, carrier, product_flops(rows, q, step.h));
    }
    // Whether the segmented carriers take h reflectors (both slices of Z and D own some).
    static bool tree_carriers_admit(int h) {
        return carrier_z_supported<T>(h) && carrier_d_supported<T>(h);
    }
    // The domains' update: W_i = V_i^T X_i with each domain's rows among c.domain groups of
    // blocks, Z_i = op(T_i) W_i, X_i -= V_i Z_i, one segmented product each for all domains.
    void tree_domains(Lane<T> &lane, T *x, int ldx, const DomainTree &t, const T *vp, int ldp,
                      const T *tt, int q, bool transpose) {
        const int h = t.h, D = t.count[0];
        const int stride = t.leaf_stride(), rows = (D - 1) * stride + t.ge_rows(D - 1);
        const long long ws = (long long)h * q, zs = (long long)q * h, sq = (long long)h * h;
        cudaStream_t st = lane.stream;
        if (c.tree_groups == 1) {
            TreeStep step;
            step.h = h;
            step.segments = D;
            step.domain = t.domain;
            step.last = t.domain_rows(D - 1);
            step.tile = t.tile;
            tree_step(step, vp, ldp, tt, x, ldx, rows, q, transpose, c.domain, st);
            return;
        }
        if (!tree_carriers_admit(h) || stride % carrier_d_row_tile<T>()) {
            for (int i = 0; i < D; ++i)
                apply(lane, x + size_t(i) * t.domain, ldx, vp + size_t(i) * stride, ldp,
                      t_at_b(lane, tt + i * sq, h), t.ge_rows(i), h, q, transpose);
            return;
        }
        const long long tiles = (long long)ceildiv(q, 64) * ceildiv(h, 64);
        const int groups = bounded_groups(tiles, std::min(std::max(c.tree_groups, c.domain),
                                                          std::max(1, stride / carrier_w_rows)));
        CarrierGArgs<T> w{x, ldx, vp, ldp, lane.tw, h, q, h, t.ge_rows(0), groups, lane.tp, ws};
        w.S = D;
        w.a_seg = t.domain;
        w.b_seg = stride;
        w.o_seg = ws;
        if (t.ge_rows(D - 1) != t.ge_rows(0))
            w.k_last = t.ge_rows(D - 1);
        record(ProductW, launch_carrier_w_args(w, 1, st), product_flops(h, q, rows));
        record(ProductZ,
               launch_carrier_z<T>(tt, h, lane.tw, h, lane.tz, q, h, q, transpose, true, st, D, sq,
                                   ws, zs),
               D * product_flops(h, q, h) / 2);
        record(ProductD,
               launch_carrier_d<T>(
                   vp, ldp, lane.tz, q, x, ldx, rows, h, q, st, D > 1 ? stride : 0, D, zs,
                   D > 1 && stride != t.domain ? RowMap{stride, stride, t.domain, t.domain}
                                               : RowMap{}),
               product_flops(rows, q, h));
    }
    // TSMQR: apply the triangle/dense-tile elimination to each domain's head and tail.
    void tree_ts(Lane<T> &lane, T *x, int ldx, const DomainTree &t, int phase, const T *vstack,
                 const T *tt, int q, bool transpose) {
        const int h = t.h, D = t.count[0], height = DomainConfig::rows, ld = t.ts_vrows();
        const long long sq = (long long)h * h, ws = (long long)h * q;
        const T *v = vstack + t.ts_offset(phase), *ts = tt + size_t(phase * D) * sq;
        TreeStep step;
        step.h = h;
        step.segments = D;
        step.domain = t.domain;
        step.last = t.domain_rows(D - 1);
        step.tile = t.tile;
        step.phase = phase;
        cudaStream_t st = lane.stream;
        if (c.tree_groups == 1) {
            int rows = 0;
            for (int i = 0; i < D; ++i)
                rows += h + t.ts_rows(phase, i);
            tree_step(step, v, ld, ts, x, ldx, rows, q, transpose, c.domain, st);
            return;
        }
        {
            tree_ts_copy<T, true><<<256, 256, 0, st>>>(step, x, ldx, lane.tx, q);
            if (!tree_carriers_admit(h) || height % carrier_d_row_tile<T>()) {
                for (int i = 0; i < D; ++i)
                    apply(lane, lane.tx + size_t(i) * height, ld, v + size_t(i) * height, ld,
                          t_at_b(lane, ts + i * sq, h), height, h, q, transpose);
            } else {
                const long long tiles = (long long)ceildiv(q, 64) * ceildiv(h, 64);
                const int groups =
                    bounded_groups(tiles, std::min(std::max(c.tree_groups, c.domain),
                                                   std::max(1, height / carrier_w_rows)));
                CarrierGArgs<T> w{lane.tx, ld, v,      ld,     lane.tw, h,
                                  q,       h,  height, groups, lane.tp, ws};
                w.S = D;
                w.a_seg = w.b_seg = height;
                w.o_seg = ws;
                record(ProductW, launch_carrier_w_args(w, 1, st), product_flops(h, q, ld));
                record(ProductZ,
                       launch_carrier_z<T>(ts, h, lane.tw, h, lane.tz, q, h, q, transpose, true, st,
                                           D, sq, ws, ws),
                       D * product_flops(h, q, h) / 2);
                record(ProductD,
                       launch_carrier_d<T>(v, ld, lane.tz, q, lane.tx, ld, ld, h, q, st, height, D,
                                           ws),
                       product_flops(ld, q, h));
            }
            tree_ts_copy<T, false><<<256, 256, 0, st>>>(step, x, ldx, lane.tx, q);
            CU(cudaGetLastError());
        }
    }
    // The update of TT level l: W_n = sum over the children j of V_nj^T X_j, the children dealt to
    // c.merge layers of blocks whose partials are summed (the replication of the elimination);
    // Z_n = op(T_n) W_n; X_j -= V_nj Z_n on the rows of every child's triangle.
    void tree_level(Lane<T> &lane, T *x, int ldx, const DomainTree &t, int l, const T *vstack,
                    const T *tt, int q, bool transpose) {
        const int h = t.h, nodes = t.count[l], F = t.fan_in, last = t.children(l, nodes - 1);
        const int ldv = t.vstack_rows(l), vrows = (nodes - 1) * F * h + last * h;
        const long long ws = (long long)h * q, zs = (long long)q * h, sq = (long long)h * h;
        const T *vl = vstack + t.vstack_offset(l), *tl = tt + size_t(t.first[l]) * sq;
        const RowMap map{h, F * h, t.stride[l - 1], t.stride[l]};
        cudaStream_t st = lane.stream;
        if (c.tree_groups == 1) {
            TreeStep step;
            step.h = h;
            step.segments = nodes;
            step.outer = t.stride[l];
            step.inner = t.stride[l - 1];
            step.fan_in = F;
            step.last_children = last;
            tree_step(step, vl, ldv, tl, x, ldx, vrows, q, transpose, c.merge, st);
            return;
        }
        if (!tree_carriers_admit(h) || F * h % carrier_d_row_tile<T>()) {
            for (int n = 0; n < nodes; ++n)
                tree_node_by_copy(lane, x + n * t.stride[l], ldx, t.children(l, n), t.stride[l - 1],
                                  vl + size_t(n) * F * h, ldv, tl + n * sq, h, q, transpose);
            return;
        }
        const long long tiles = (long long)ceildiv(q, 64) * ceildiv(h, 64);
        const int layers = bounded_groups(tiles, std::min(std::max(c.tree_groups, c.merge), F));
        CarrierGArgs<T> w{x, ldx, vl, ldv, lane.tw, h, q, h, h, layers, lane.tp, ws};
        w.S = nodes;
        w.a_seg = t.stride[l];
        w.b_seg = (long long)F * h;
        w.o_seg = ws;
        w.a_grp = t.stride[l - 1];
        w.b_grp = h;
        w.blocks = F;
        w.blocks_last = last;
        record(ProductW, launch_carrier_w_args(w, 1, st), product_flops(h, q, vrows));
        record(ProductZ,
               launch_carrier_z<T>(tl, h, lane.tw, h, lane.tz, q, h, q, transpose, true, st, nodes,
                                   sq, ws, zs),
               nodes * product_flops(h, q, h) / 2);
        record(ProductD,
               launch_carrier_d<T>(vl, ldv, lane.tz, q, x, ldx, vrows, h, q, st, F * h, nodes, zs,
                                   map),
               product_flops(vrows, q, h));
    }
    // A tree's T (leading dimension h) copied to the leading dimension of apply.
    const T *t_at_b(Lane<T> &lane, const T *t, int h) {
        const int ld = std::max(b, h);
        CU(cudaMemcpy2DAsync(lane.tz, size_t(ld) * sizeof(T), t, size_t(h) * sizeof(T),
                             size_t(h) * sizeof(T), h, cudaMemcpyDeviceToDevice, lane.stream));
        return lane.tz;
    }
    // One TT elimination through a copy of its children's rows, for reflectors too few for the
    // segmented carriers: the stack of k triangles' rows (k h x q) is gathered, updated with the
    // stacked V (leading dimension ldv) and scattered back.
    void tree_node_by_copy(Lane<T> &lane, T *x, int ldx, int k, long long step, const T *v, int ldv,
                           const T *t, int h, int q, bool transpose) {
        cudaStream_t st = lane.stream;
        T *stack = lane.tp;
        for (int j = 0; j < k; ++j)
            CU(cudaMemcpy2DAsync(stack + size_t(j) * h, size_t(k) * h * sizeof(T), x + j * step,
                                 size_t(ldx) * sizeof(T), size_t(h) * sizeof(T), q,
                                 cudaMemcpyDeviceToDevice, st));
        apply(lane, stack, k * h, v, ldv, t_at_b(lane, t, h), k * h, h, q, transpose);
        for (int j = 0; j < k; ++j)
            CU(cudaMemcpy2DAsync(x + j * step, size_t(ldx) * sizeof(T), stack + size_t(j) * h,
                                 size_t(k) * h * sizeof(T), size_t(h) * sizeof(T), q,
                                 cudaMemcpyDeviceToDevice, st));
    }
};
} // namespace tqr
