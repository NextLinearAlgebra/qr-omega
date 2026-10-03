#pragma once
// The QR-Omega engine. It runs the elimination list of plan.hpp with the schedule of Algorithm 1:
// every panel is factored by one GE, its reflectors are applied to the trailing columns in strips
// (update.cuh), the strips that release the next panel go first while the far ones overlap its
// factorization, and across GPUs one TT elimination per panel merges the local triangles.
#include "plan.hpp"
#include "transport.cuh"
#include "kernels.cuh"
#include "panel_cooperative.cuh"
#include "panel_register.cuh"
#include "update.cuh"
#include <functional>

namespace tqr {
struct Options {
    int b = 128;            // panel width
    int strip = 4096;       // columns of one strip of the trailing update
    int depth = 1;          // strips in flight
    int aggregate = 1;      // panels composed into one far update
    bool lookahead = false; // factor the next panel while the far update runs
    Replication c;          // contraction replication of W and Z
    // A tall panel is split among `groups` thread blocks and advances in windows of `window`
    // columns and mini-panels of `minipanel`; panels with fewer than `late_rows` rows use
    // `late_groups`.
    int groups = 32, minipanel = 16, window = 64, late_groups = 0, late_rows = 0;
    int compose_groups = 0; // TF32: GPU-level groups of the Gram products of an aggregated T
    int d_tiles = 1;        // 3xTF32: output tiles per thread block of the far D
};

template <class T> class Engine {
    // Panels at least this tall can be split among thread blocks of this many threads.
    static constexpr int cooperative_panel_rows = 1024, panel_threads = 512;

    Context &ctx;
    Options opt;
    Plan plan;
    const int n, b, ldv; // ldv: leading dimension of the packed reflectors
    Tally tally;

    Stream stream, compose_stream;
    std::vector<std::unique_ptr<Stream>> side_streams;
    std::vector<std::unique_ptr<Event>> lane_joined;
    Event lane_fork, far_done, merge_factored, compose_fork, compose_joined;
    std::unique_ptr<Updater<T>> update;

    Buffer<int> status;
    Buffer<TilePacket> packets;
    Buffer<T> tri, V, column_scale, scale_gather;
    // Panel workspaces: partials of the cooperative panel, the Gram block of a window boundary, the
    // GPU-level partial slices of a Gram product, tau and overlaps of a register panel.
    Buffer<T> panel_partials, window_gram, gram_partials, tau, overlaps;
    Buffer<int> chain_flags;
    int chain_epoch = 0;
    std::vector<int> panel_groups, panel_window; // cooperative configuration of each panel
    // Reflectors of the panels in flight live in V, one half per panel.
    size_t vhalf, vbase = 0;
    // Aggregated transform: V_g and T_g in up to two generations, the Gram blocks, and the
    // reflector order that makes each contraction half of D a union of whole panels.
    Buffer<T> group_v, group_t, group_t_ordered, group_gram, group_s;
    Buffer<int> group_order;
    std::vector<int> order_offset;
    int generation = 0;
    uint64_t group_version[2] = {1, 1};
    // TT merge across GPUs.
    std::unique_ptr<Transport> transport;
    Buffer<T> stack, stack_v, reduced, gathered, member_v;

    Matrix<T> *factors = nullptr;
    bool factored = false;

    T *reflectors() const {
        return V.p + vbase;
    }
    T *t_of(int tile) const {
        return tri.p + size_t(tile) * b * b;
    }
    cudaStream_t lane_stream(int lane) const {
        return update->lane(lane).stream;
    }
    T *group_v_at(int gen) const {
        return group_v.p + size_t(gen) * ldv * b * opt.aggregate;
    }
    T *group_t_at(Buffer<T> const &buffer, int gen) const {
        const size_t width = size_t(b) * opt.aggregate;
        return buffer.p + size_t(gen) * width * width;
    }
    int local_status() {
        return ctx.max(status.download()[0]);
    }

    // ---- column scaling --------------------------------------------------------------------------
    // Columns are scaled to unit maximum so that FP32 cannot overflow; only R is scaled back.
    void normalize_columns(Matrix<T> &x, T *scales) {
        tiled_column_max<<<x.n, 256, 0, stream>>>(x.a.p, x.rows, x.n, x.ld, scales);
        if (transport)
            for (int col = 0; col < x.n; col += opt.strip) {
                const int q = std::min(opt.strip, x.n - col);
                for (int root = 0; root < ctx.size; ++root) {
                    transport->broadcast(root, scales + col, scale_gather.p, q);
                    tiled_scale_max<<<ceildiv(q, 128), 128, 0, stream>>>(scales + col,
                                                                         scale_gather.p, q);
                }
            }
        if (x.rows)
            tiled_column_scale<T, false><<<column_pass_grid(x.rows, x.n), 256, 0, stream>>>(
                x.a.p, x.rows, x.n, x.ld, scales, status.p);
    }

    // ---- strips ---------------------------------------------------------------------------------
    void fork_lanes(int first, int count) {
        if (count < 2)
            return;
        lane_fork.record(lane_stream(first));
        for (int i = 1; i < count; ++i)
            lane_fork.wait(lane_stream(first + i));
    }
    void join_lanes(int first, int count) {
        for (int i = 1; i < count; ++i) {
            lane_joined[first + i]->record(lane_stream(first + i));
            lane_joined[first + i]->wait(lane_stream(first));
        }
    }
    // Runs issue(lane, col, q) for the strips of [lo, hi), opt.depth of them in flight on the lanes
    // from `first` on.
    using Issue = std::function<void(Lane<T> &, int, int)>;
    void for_strips(int lo, int hi, int first, const Issue &issue) {
        fork_lanes(first, opt.depth);
        for (int col = lo, strip = 0; col < hi; ++strip) {
            const int q = std::min(opt.strip, hi - col);
            issue(update->lane(first + strip % opt.depth), col, q);
            col += q;
        }
        join_lanes(first, opt.depth);
    }

    // ---- panel ----------------------------------------------------------------------------------
    // G = V1^T V2 between reflectors packed with leading dimension ld.
    void gram(cudaStream_t st, const T *v1, const T *v2, T *g, int ldg, int rows, int h1, int h2,
              int ld, int tf32_groups = 0) {
        update->gram(st, v1, ld, v2, ld, g, ldg, rows, h1, h2, gram_partials.p, gram_partials.n,
                     tf32_groups);
    }
    // Householder QR of a rows x h block whose columns are held in registers: one block that also
    // builds T when the block fits in it, else a chain of blocks, the overlaps V^T V and T from
    // them. V goes to v, T to the tile of the packet. Returns false, before writing anything, when
    // the block is too large.
    bool register_factor(T *a, int lda, const TilePacket *pk, int rows, int h, T *v, int ld) {
        const RegisterPanel<T> panel{a, lda, pk, rows, h, v, ld, tau.p, status.p};
        const bool wide = h > 128;
        if (launch_fused_panel(panel, tri.p, b, stream))
            return true;
        if (wide && !wide_t_fits<T>(h, b))
            return false;
        if (!launch_chain_panel(panel, chain_flags.p, chain_epoch + 1, wide, stream))
            return false;
        ++chain_epoch;
        gram(stream, v, v, overlaps.p, h, rows, h, h, ld);
        if (wide)
            launch_multi_wide_t(overlaps.p, h, tau.p, pk, h, tri.p, b, window_gram.p, stream);
        else
            launch_build_t(overlaps.p, h, tau.p, pk, h, tri.p, b, stream);
        return true;
    }
    // Window [w0, w0 + sw) of a cooperative panel is factored: pack its reflectors, join its T to
    // the T of the earlier windows (on a side stream), and update the rest of the panel.
    void window_boundary(Matrix<T> &a, const TilePacket &pk, int w0, int sw) {
        const TilePacket *dev = packets.p + pk.tile;
        const int w1 = w0 + sw;
        pack_ge_v_cols<<<pack_v_grid(ldv, sw), 128, 0, stream>>>(a.a.p, a.ld, dev, reflectors(),
                                                                 ldv, w0, w1);
        if (w0 > 0) {
            cudaStream_t st = stream;
            if (has_native_wz<T>()) {
                compose_fork.record(stream);
                compose_fork.wait(compose_stream);
                st = compose_stream;
            }
            gram(st, reflectors() + w0, reflectors() + size_t(w0) * ldv + w0, window_gram.p, w0,
                 pk.rows - w0, w0, sw, ldv);
            tile_compose_offdiag<T>
                <<<sw, 128, size_t(w0) * sizeof(T), st>>>(t_of(pk.tile), b, window_gram.p, w0, sw);
        }
        for (int col = pk.col + w1; col < pk.col + pk.h;) {
            const int q = std::min(opt.strip, pk.col + pk.h - col);
            update->apply(update->lane(0), a.a.p + pk.row + size_t(col) * a.ld, a.ld,
                          reflectors() + size_t(w0) * ldv, ldv, t_of(pk.tile) + w0 + size_t(w0) * b,
                          pk.rows, sw, q, true);
            col += q;
        }
    }
    // The rows of a tall panel split among thread blocks, one launch per window of columns.
    void cooperative_panel(Matrix<T> &a, const TilePacket &pk, int panel) {
        const TilePacket *dev = packets.p + pk.tile;
        const int groups = panel_groups[panel], sw0 = panel_window[panel],
                  pw = std::min(opt.minipanel, sw0), max_h = std::max(128, b);
        int windows = 0, last = 0;
        for (int w0 = 0; w0 < pk.h; w0 += sw0, ++windows) {
            const int sw = std::min(sw0, pk.h - w0);
            if (!launch_cooperative_window(a.a.p, a.ld, dev, pk.rows, pk.h, w0, sw, pw, groups,
                                           panel_threads, tri.p, b, panel_partials.p, status.p,
                                           stream.s, false, max_h))
                throw std::runtime_error("a panel window no longer fits its thread blocks");
            if (w0 + sw < pk.h)
                window_boundary(a, pk, w0, sw);
            last = w0;
        }
        if (windows > 1) {
            window_boundary(a, pk, last, pk.h - last);
            if (has_native_wz<T>()) {
                compose_joined.record(compose_stream);
                compose_joined.wait(stream);
            }
        } else
            pack_ge_v<<<pack_v_grid(ldv, b), 128, 0, stream>>>(a.a.p, a.ld, dev, reflectors(), ldv,
                                                               b);
    }
    // GE of the local rows of a panel: R and V in place, the packed V in reflectors(), T in tri.
    void factor_ge(Matrix<T> &a, int panel) {
        const TilePacket &pk = plan.panels[panel].ge[ctx.rank];
        if (!pk.rows)
            return;
        if (pk.h <= 128 &&
            register_factor(a.a.p, a.ld, packets.p + pk.tile, pk.rows, pk.h, reflectors(), ldv))
            tally.add(1);
        else if (panel_groups[panel]) {
            cooperative_panel(a, pk, panel);
            tally.add(panel_groups[panel]);
        } else
            throw std::runtime_error("no panel kernel for " + std::to_string(pk.rows) + " x " +
                                     std::to_string(pk.h));
    }
    // Applies the reflectors of a panel to the columns [lo, hi) of x.
    void apply_ge(Matrix<T> &x, int panel, int lo, int hi, int first_lane, bool transpose) {
        const TilePacket &pk = plan.panels[panel].ge[ctx.rank];
        if (!pk.rows)
            return;
        for_strips(lo, hi, first_lane, [&](Lane<T> &lane, int col, int q) {
            update->apply(lane, x.a.p + pk.row + size_t(col) * x.ld, x.ld, reflectors(), ldv,
                          t_of(pk.tile), pk.rows, pk.h, q, transpose);
        });
    }
    // Cooperative configuration of a panel: the widest window whose blocks fit in shared memory.
    void configure_panels() {
        panel_groups.assign(plan.panels.size(), 0);
        panel_window.assign(plan.panels.size(), 0);
        if (!opt.groups)
            return;
        for (size_t k = 0; k < plan.panels.size(); ++k) {
            const TilePacket &pk = plan.panels[k].ge[ctx.rank];
            const bool tall = pk.rows >= cooperative_panel_rows && pk.h >= opt.minipanel;
            if (!pk.rows || !(tall || pk.h > 128))
                continue;
            const int groups =
                opt.late_groups > 1 && pk.rows < opt.late_rows ? opt.late_groups : opt.groups;
            const int widest = std::min(pk.h, opt.window ? opt.window : pk.h);
            std::vector<int> candidates{widest};
            for (int w : {64, 32, 16, 8})
                if (w < widest && w >= opt.minipanel)
                    candidates.push_back(w);
            for (int sw : candidates) {
                bool fits = true;
                for (int w0 = 0; w0 < pk.h && fits; w0 += sw)
                    fits = launch_cooperative_window<T>(
                        nullptr, pk.rows, nullptr, pk.rows, pk.h, w0, std::min(sw, pk.h - w0),
                        std::min(opt.minipanel, sw), groups, panel_threads, nullptr, b, nullptr,
                        nullptr, nullptr, true, std::max(128, b));
                if (fits) {
                    panel_groups[k] = groups;
                    panel_window[k] = sw;
                    break;
                }
            }
            if (!panel_groups[k])
                throw std::runtime_error("the panel does not fit in shared memory with " +
                                         std::to_string(groups) + " thread blocks");
        }
    }

    // ---- aggregation ----------------------------------------------------------------------------
    // Whether D cuts the reflectors of g aggregated panels into two halves of whole panels.
    bool halves_are_panels(int g) const {
        int bk, wk;
        carrier_d_law<T>(bk, wk);
        return g % 2 == 0 && bk == 2 * wk && (g * b) % bk == 0 && native_d_admits<T>(g * b);
    }
    // Panels k .. k + g - 1 whose far update is applied as one composed transform. A group must
    // leave columns to its right.
    int group_size(size_t k) const {
        if (opt.aggregate < 2 || transport || plan.panels[k].h != b)
            return 1;
        int g = 1;
        while (g < opt.aggregate && k + g < plan.panels.size() && plan.panels[k + g].h == b)
            ++g;
        while (g > 1 && (plan.panels[k + g - 1].col + b >= n || !halves_are_panels(g)))
            --g;
        return g;
    }
    // First column after the group of panels that starts at panel k.
    int unit_end(size_t k) const {
        if (k >= plan.panels.size())
            return n;
        const Panel &last = plan.panels[k + group_size(k) - 1];
        return std::min(n, last.col + last.h);
    }
    // Reflector order of a group of g panels: every k-tile of D alternates between the first and
    // the second half of the reflectors, so each of its two peers contracts whole panels.
    void build_group_orders() {
        int bk, wk;
        carrier_d_law<T>(bk, wk);
        std::vector<int> all;
        order_offset.assign(opt.aggregate + 1, 0);
        for (int g = 2; g <= opt.aggregate; g += 2) {
            const int H = g * b;
            order_offset[g] = int(all.size());
            for (int q = 0; q < H; ++q)
                all.push_back((q % bk) / wk * (H / 2) + (q / bk) * wk + q % wk);
        }
        group_order.alloc(all.size());
        CU(cudaMemcpy(group_order.p, all.data(), all.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    // T_g of the group by the compose recurrence, one panel at a time, and V_g in the group order.
    void compose_group(Matrix<T> &a, size_t k, int g) {
        const int H = g * b, rows = plan.panels[k].ge[0].rows;
        T *tg = group_t_at(group_t, generation);
        CU(cudaMemsetAsync(tg, 0, size_t(H) * H * sizeof(T), stream));
        for (int j = 0; j < g; ++j) {
            const T *tj = t_of(plan.panels[k + j].ge[0].tile);
            for (int i = 0; i < j; ++i)
                gram(stream, V.p + size_t(i) * vhalf + size_t(j - i) * b, V.p + size_t(j) * vhalf,
                     group_gram.p + size_t(i) * b * b, b, rows - j * b, b, b, ldv,
                     opt.compose_groups);
            if (j > 0 && b % kComposeTile == 0) {
                const int jb = j * b;
                const dim3 grid(ceildiv(jb, kComposeTile), b / kComposeTile),
                    block(kComposeTile, 8);
                compose_s_tiled<T><<<grid, block, 0, stream>>>(group_gram.p, b, jb, tj, group_s.p);
                compose_n_tiled<T><<<grid, block, 0, stream>>>(tg, H, jb, b, group_s.p);
                compose_diag_copy<T><<<ceildiv(b * b, 256), 256, 0, stream>>>(tg, H, j, b, tj);
            } else
                compose_tg_step<T><<<b, 128, size_t(std::max(1, j) * b) * sizeof(T), stream>>>(
                    tg, H, j, b, group_gram.p, tj);
            CU(cudaGetLastError());
        }
        const TilePacket &first = plan.panels[k].ge[0];
        const int *order = group_order.p + order_offset[g];
        ++group_version[generation];
        pack_v_block_perm<<<256, 256, 0, stream>>>(a.a.p, a.ld, first.row, first.col, first.rows, H,
                                                   order, group_v_at(generation), ldv);
        permute_t<T><<<64, 256, 0, stream>>>(tg, H, order, group_t_at(group_t_ordered, generation));
    }
    // The composed transform of the group applied to the columns [lo, hi).
    void apply_group(Matrix<T> &a, size_t k, int g, int lo, int hi, int first_lane) {
        const TilePacket &first = plan.panels[k].ge[0];
        const T *vg = group_v_at(generation), *tg = group_t_at(group_t_ordered, generation);
        const uint64_t version = group_version[generation];
        for_strips(lo, hi, first_lane, [&](Lane<T> &lane, int col, int q) {
            update->apply(lane, a.a.p + first.row + size_t(col) * a.ld, a.ld, vg, ldv, tg,
                          first.rows, g * b, q, true, version);
        });
    }

    // ---- TT merge across GPUs -------------------------------------------------------------------
    int copy_blocks(int rows, int columns) const {
        return std::max(1, std::min(512, ceildiv(rows * columns, 128)));
    }
    // Gathers the triangles of the members on the owner, factors their stack with the register
    // chain, and returns R to the owner's rows and each member's block of V to its own rows.
    void factor_merge(const Merge &e, Matrix<T> &a) {
        const int h = e.h, K = e.k, owner = e.owner(), mine = e.member(ctx.rank);
        const size_t block = size_t(b) * b;
        cudaStream_t st = stream;
        transport->bind(st);
        if (mine == 0) {
            stack.zero(st);
            extract_r<<<8, 128, 0, st>>>(a.a.p, a.ld, e.row[0], e.col, e.height[0], h, stack.p,
                                         K * b);
        } else if (mine > 0) {
            T *r = gathered.p + (mine - 1) * block;
            CU(cudaMemsetAsync(r, 0, block * sizeof(T), st));
            extract_r<<<8, 128, 0, st>>>(a.a.p, a.ld, e.row[mine], e.col, e.height[mine], h, r, b);
        }
        for (int i = 1; i < K; ++i) {
            T *r = gathered.p + (i - 1) * block;
            transport->publish(e.rank[i], owner, r, r, size_t(b) * h * sizeof(T));
        }
        if (mine == 0) {
            int off = e.height[0];
            for (int i = 1; i < K; off += e.height[i++])
                CU(cudaMemcpy2DAsync(stack.p + off, K * b * sizeof(T), gathered.p + (i - 1) * block,
                                     b * sizeof(T), e.height[i] * sizeof(T), h,
                                     cudaMemcpyDeviceToDevice, st));
            if (!register_factor(stack.p, K * b, packets.p + e.tile, K * b, h, stack_v.p, K * b))
                throw std::runtime_error("no kernel for the stack of triangles");
            tally.add(1);
            scatter_upper<<<8, 128, 0, st>>>(stack.p, K * b, a.a.p, a.ld, e.row[0], e.col, h, h);
            off = e.height[0];
            for (int i = 1; i < K; off += e.height[i++]) {
                T *r = gathered.p + (i - 1) * block;
                CU(cudaMemsetAsync(r, 0, block * sizeof(T), st));
                CU(cudaMemcpy2DAsync(r, b * sizeof(T), stack.p + off, K * b * sizeof(T),
                                     e.height[i] * sizeof(T), h, cudaMemcpyDeviceToDevice, st));
            }
        }
        for (int i = 1; i < K; ++i) {
            T *r = gathered.p + (i - 1) * block;
            transport->publish(owner, e.rank[i], r, r, size_t(b) * h * sizeof(T));
        }
        if (mine > 0)
            scatter_upper<<<8, 128, 0, st>>>(gathered.p + (mine - 1) * block, b, a.a.p, a.ld,
                                             e.row[mine], e.col, e.height[mine], h);
    }
    // Applies the TT elimination to the columns [lo, hi) of x. Its reflector block on the owner is
    // the identity, so W = sum of the members' V_i^T X_i is one all-reduce, Z = op(T) W is computed
    // on every GPU from the same W, and each member commits X_i -= V_i Z on its own rows. With
    // fewer members than GPUs the partials are reduced to, and Z gathered from, the owners of its
    // columns.
    void apply_merge(const Merge &e, Matrix<T> &x, int lo, int hi, int first_lane, bool transpose) {
        const int h = e.h, K = e.k, owner = e.owner(), mine = e.member(ctx.rank);
        const size_t block = size_t(b) * b;
        cudaStream_t st = lane_stream(first_lane);
        transport->bind(st);
        if (mine > 0)
            tile_bottom_v<<<8, 128, 0, st>>>(factors->a.p, factors->ld, e.row[mine], e.col,
                                             e.height[mine], h, member_v.p, b);
        for (int i = 1; i < K; ++i)
            transport->publish(owner, e.rank[i], t_of(e.tile), gathered.p, block * sizeof(T));
        const T *t = mine == 0 ? t_of(e.tile) : gathered.p;
        const int depth =
            std::min(opt.depth, std::max(1, ceildiv(std::max(0, hi - lo), opt.strip)));
        ColumnTeam team;
        team.size = K;
        std::copy(e.rank, e.rank + K, team.rank);
        const bool everyone = K == ctx.size;
        // The arithmetic runs on the lanes and the collectives on the transport's own stream, in
        // column order; events hand every strip over and back.
        fork_lanes(first_lane, depth);
        lane_fork.record(st);
        transport->bind(nullptr);
        lane_fork.wait(transport->bound());
        std::vector<std::unique_ptr<Event>> handoff;
        for (int i = 0; i < depth; ++i)
            handoff.push_back(std::make_unique<Event>());
        auto to_transport = [&](int i, cudaStream_t lane) {
            handoff[i]->record(lane);
            handoff[i]->wait(transport->bound());
        };
        auto to_lane = [&](int i, cudaStream_t lane) {
            handoff[i]->record(transport->bound());
            handoff[i]->wait(lane);
        };
        int strips = 0;
        for (int col = lo; col < hi; ++strips) {
            const int i = strips % depth, q = std::min(opt.strip, hi - col);
            Lane<T> &lane = update->lane(first_lane + i);
            cudaStream_t ls = lane.stream;
            T *w = lane.w, *z = lane.z, *wsum = reduced.p + size_t(first_lane + i) * b * opt.strip;
            T *xi = mine >= 0 ? x.a.p + e.row[mine] + size_t(col) * x.ld : nullptr;
            if (mine >= 0) {
                CU(cudaMemsetAsync(w, 0, size_t(opt.c.w) * b * q * sizeof(T), ls));
                CU(cudaMemsetAsync(z, 0, size_t(opt.c.z) * b * q * sizeof(T), ls));
            }
            // Columns of Z this GPU computes: all of them after an all-reduce, its share otherwise.
            const int zq = everyone   ? q
                           : mine < 0 ? 0
                                      : team.begin(mine + 1, q) - team.begin(mine, q);
            if (zq)
                tally.add(K);
            if (mine == 0) {
                // W and D of the owner's identity block are a copy and a subtraction.
                tally.add(1);
                tally.add(1);
                tile_copy_identity<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0], col, h,
                                                                      q, w, b);
            } else if (mine > 0)
                tally.add(
                    update->stage_w(lane, member_v.p, b, xi, x.ld, w, b, e.height[mine], h, q));
            to_transport(i, ls);
            if (everyone)
                transport->allreduce(w, wsum, size_t(b) * q);
            else
                transport->template columns<T, true>(team, w, wsum, b, q, false);
            to_lane(i, ls);
            const bool transposed = native_d_admits<T>(h);
            if (zq)
                tally.add(update->stage_z(lane, t, wsum, z, b, h, zq, transpose, transposed));
            const T *zfull = z;
            if (!everyone) {
                to_transport(i, ls);
                transport->template columns<T, false>(team, z, wsum, b, q, transposed);
                to_lane(i, ls);
                zfull = wsum;
            }
            if (mine == 0) {
                if (transposed)
                    tile_add_identity_t<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0],
                                                                           col, h, q, zfull, q);
                else
                    tile_add_identity<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0], col,
                                                                         h, q, zfull, b);
            } else if (mine > 0)
                tally.add(
                    update->stage_d(lane, member_v.p, b, zfull, b, xi, x.ld, e.height[mine], h, q));
            col += q;
        }
        join_lanes(first_lane, depth);
        transport->bind(st);
    }

    // ---- schedule -------------------------------------------------------------------------------
    void factor_panels(Matrix<T> &a) {
        const int far_lane = opt.lookahead ? opt.depth : 0;
        // A far update in flight, and whether it reads the reflectors of a single panel in V.
        bool far_pending = false, far_reads_v = false;
        auto wait_far = [&] {
            if (far_pending)
                far_done.wait(stream);
            far_pending = far_reads_v = false;
            if (transport)
                transport->bind(stream);
        };
        auto fork_far = [&] {
            lane_fork.record(stream);
            lane_fork.wait(lane_stream(far_lane));
        };
        for (size_t k = 0; k < plan.panels.size(); ++k) {
            const Panel &panel = plan.panels[k];
            if (const int g = group_size(k); g > 1) {
                // The panels of the group, each updating the columns of the group, then one
                // composed update of the columns to its right.
                if (far_reads_v)
                    wait_far();
                const int group_end = unit_end(k);
                for (int j = 0; j < g; ++j) {
                    const Panel &pj = plan.panels[k + j];
                    vbase = j * vhalf;
                    factor_ge(a, int(k) + j);
                    apply_ge(a, int(k) + j, pj.col + pj.h, group_end, 0, true);
                }
                vbase = 0;
                const int release = opt.lookahead ? unit_end(k + g) : n;
                if (opt.lookahead)
                    generation ^= 1;
                compose_group(a, k, g);
                wait_far();
                apply_group(a, k, g, group_end, release, 0);
                if (release < n) {
                    fork_far();
                    apply_group(a, k, g, release, n, far_lane);
                    far_done.record(lane_stream(far_lane));
                    far_pending = true;
                }
                k += g - 1;
                continue;
            }
            const int end = panel.col + panel.h;
            // Columns the next panel (or group of panels) needs before it can be factored.
            const int release =
                opt.lookahead && k + 1 < plan.panels.size() && end < n ? unit_end(k + 1) : n;
            const Merge &merge = panel.merge;
            vbase = opt.lookahead ? (k & 1) * vhalf : 0;
            factor_ge(a, int(k));
            wait_far();
            apply_ge(a, int(k), end, release, 0, true);
            if (release < n) {
                fork_far();
                apply_ge(a, int(k), release, n, far_lane, true);
            }
            if (merge.k) {
                // The far GE update is already running: the gather, the TT and its near update
                // overlap it.
                factor_merge(merge, a);
                apply_merge(merge, a, end, release, 0, true);
            }
            if (release < n) {
                if (merge.k) {
                    merge_factored.record(stream);
                    merge_factored.wait(lane_stream(far_lane));
                    apply_merge(merge, a, release, n, far_lane, true);
                }
                far_done.record(lane_stream(far_lane));
                far_pending = far_reads_v = true;
            }
        }
        wait_far();
        vbase = 0;
    }

  public:
    Engine(Context &c, int m, int n_, const Options &o)
        : ctx(c), opt(o), plan(m, n_, c.size, o.b), n(n_), b(o.b),
          ldv((plan.rows.local_rows(c.rank) + 127) / 128 * 128),
          stream(o.lookahead ? Stream::greatest_priority() : 0),
          compose_stream(Stream::greatest_priority()) {
        if (m < n || b > (ctx.size > 1 ? 512 : 128) || opt.strip < 1 || opt.depth < 1 ||
            opt.c.w < 2 || opt.c.z < 2 || opt.aggregate < 1 || opt.aggregate > 16)
            throw std::runtime_error("invalid schedule options");
        if (ctx.size > 1 && (opt.aggregate != 1 || !opt.lookahead))
            throw std::runtime_error("several GPUs need look-ahead and no aggregation");
        const size_t tiles = plan.tiles[ctx.rank], block = size_t(b) * b;
        status.alloc(1);
        packets.alloc(tiles);
        std::vector<TilePacket> host(tiles);
        for (const Panel &panel : plan.panels) {
            const TilePacket &pk = panel.ge[ctx.rank];
            if (pk.rows)
                host[pk.tile] = pk;
            const Merge &e = panel.merge;
            if (e.k && e.owner() == ctx.rank) {
                int rows = 0;
                for (int i = 0; i < e.k; ++i)
                    rows += e.height[i];
                host[e.tile] = {0, 0, rows, e.h, e.tile};
            }
        }
        packets.upload(host, stream);
        status.zero(stream);
        stream.sync();

        // Lanes 0 .. depth - 1 carry the strips that release the next panel; under look-ahead the
        // lanes depth .. 2 depth - 1 carry the far update at the lowest priority.
        const int lanes = opt.depth * (opt.lookahead ? 2 : 1);
        std::vector<cudaStream_t> streams{stream.s};
        lane_joined.resize(lanes);
        for (int i = 1; i < lanes; ++i) {
            side_streams.push_back(opt.lookahead && i < opt.depth
                                       ? std::make_unique<Stream>(Stream::greatest_priority())
                                       : std::make_unique<Stream>());
            streams.push_back(side_streams.back()->s);
            lane_joined[i] = std::make_unique<Event>();
        }
        update = std::make_unique<Updater<T>>(b, opt.aggregate, opt.strip, ldv, opt.c, opt.d_tiles,
                                              streams, opt.lookahead ? opt.depth : -1, tally);
        column_scale.alloc(n);
        scale_gather.alloc(opt.strip);
        tri.alloc(tiles * block);
        vhalf = size_t(ldv) * b;
        V.alloc(std::max(opt.lookahead ? 2 : 1, opt.aggregate) * vhalf);
        configure_panels();
        if (opt.groups)
            panel_partials.alloc(
                coop_mini_total_slots(std::max(opt.groups, opt.late_groups), b, opt.minipanel));
        window_gram.alloc(block);
        gram_partials.alloc(128 * block);
        tau.alloc(b);
        overlaps.alloc(block);
        chain_flags.alloc(b);
        CU(cudaMemset(chain_flags.p, 0, b * sizeof(int)));
        if (opt.aggregate > 1) {
            const size_t generations = opt.lookahead ? 2 : 1, width = size_t(b) * opt.aggregate;
            group_v.alloc(generations * ldv * width);
            group_t.alloc(generations * width * width);
            group_t_ordered.alloc(generations * width * width);
            group_gram.alloc(opt.aggregate * block);
            group_s.alloc(opt.aggregate * block);
            build_group_orders();
        }
        if (ctx.size > 1) {
            stack.alloc(ctx.size * block);
            stack_v.alloc(ctx.size * block);
            reduced.alloc(size_t(lanes) * b * opt.strip);
            gathered.alloc((ctx.size - 1) * block);
            member_v.alloc(block);
            transport = std::make_unique<Transport>(
                ctx, std::max(block, size_t(b) * opt.strip) * sizeof(T), block * sizeof(T));
            transport->bind(stream);
        }
    }
    const RowMap &rows() const {
        return plan.rows;
    }
    // Products issued by the last factorization or application of Q.
    const Tally &products() const {
        return tally;
    }
    // Factors a in place: R in its upper triangle, the reflectors below it, their T factors inside
    // the engine. Returns a DeviceStatus.
    int factor(Matrix<T> &a) {
        factors = &a;
        factored = false;
        tally = Tally{};
        status.zero(stream);
        finite_scan_2d<<<column_pass_grid(a.rows, a.n), 256, 0, stream>>>(a.a.p, a.rows, a.n, a.ld,
                                                                          status.p);
        stream.sync();
        if (int st = local_status())
            return st;
        normalize_columns(a, column_scale.p);
        factor_panels(a);
        tiled_column_scale<T, true, true, true><<<column_pass_grid(a.rows, a.n), 256, 0, stream>>>(
            a.a.p, a.rows, a.n, a.ld, column_scale.p, status.p, ctx.size > 1 ? b : 0, ctx.size,
            ctx.rank);
        stream.sync();
        const int st = local_status();
        factored = st == OK;
        return st;
    }
    // x <- Q x or Q^T x with the factors of the last factorization.
    int apply_q(Matrix<T> &x, bool transpose) {
        if (!factored || x.m != plan.rows.m)
            throw std::runtime_error("apply_q needs a factorization of matching height");
        tally = Tally{};
        Buffer<T> scales(x.n);
        status.zero(stream);
        finite_scan_2d<<<column_pass_grid(x.rows, x.n), 256, 0, stream>>>(x.a.p, x.rows, x.n, x.ld,
                                                                          status.p);
        stream.sync();
        if (int st = local_status())
            return st;
        normalize_columns(x, scales.p);
        const int panels = int(plan.panels.size());
        for (int i = 0; i < panels; ++i) {
            const int k = transpose ? i : panels - 1 - i;
            const Panel &panel = plan.panels[k];
            auto ge = [&] {
                const TilePacket &pk = panel.ge[ctx.rank];
                if (!pk.rows)
                    return;
                pack_ge_v<<<pack_v_grid(ldv, b), 128, 0, stream>>>(
                    factors->a.p, factors->ld, packets.p + pk.tile, reflectors(), ldv, b);
                apply_ge(x, k, 0, x.n, 0, transpose);
            };
            auto merge = [&] {
                if (panel.merge.k)
                    apply_merge(panel.merge, x, 0, x.n, 0, transpose);
            };
            if (transpose) {
                ge();
                merge();
            } else {
                merge();
                ge();
            }
        }
        tiled_column_scale<T, true><<<column_pass_grid(x.rows, x.n), 256, 0, stream>>>(
            x.a.p, x.rows, x.n, x.ld, scales.p, status.p);
        stream.sync();
        return local_status();
    }
};
} // namespace tqr
