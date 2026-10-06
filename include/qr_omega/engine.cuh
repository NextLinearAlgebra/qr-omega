#pragma once
// The QR-Omega engine. It runs the elimination list of plan.hpp with the schedule of Algorithm 1 on
// a grid of GPUs. The GPUs of the grid column that holds a panel factor their rows of it with one
// GE each, and one TT elimination merges their triangles. The reflectors of every GE are shared
// along its grid row and applied to the trailing columns in strips (update.cuh); the strips that
// release the next panel go first and the far ones overlap its factorization. The update of a TT
// merge sums the partial W of the grid rows within every grid column.
#include "plan.hpp"
#include "transport.cuh"
#include "kernels.cuh"
#include "panel_register.cuh"
#include "update.cuh"
#include "householder_reconstruct.cuh"
#include <functional>

namespace tqr {
struct Options {
    int b = 64;                 // GE/TS/TT width, at most 64
    int strip = 4096;           // columns of one strip of the trailing update
    int depth = 1;              // strips in flight
    int aggregate = 1;          // panels composed into one far update
    bool lookahead = false;     // factor the next panel while the far update runs
    bool resident_tree = false; // keep X in shared memory for tree steps
    bool reconstruct = false;   // reconstruct compact WY from the HQR panel's thin Q
    Replication c;              // contraction replication of the products (update.cuh)
    int grid_rows = 0;          // grid rows of the GPUs; 0 takes the most within the 2.5D bound
    int d_tiles = 1;            // 3xTF32: output tiles per thread block of the far D
    int fan_in = 4;             // triangles one TT elimination inside a GPU merges
    int domain_tiles = 1;       // GE tile plus TS tiles per GPU domain
    // One GPU: the columns from tail_from on (a multiple of b aggregate) form a trailing block that
    // its own engine factors in the tree format with the schedule `tail` (this one when absent).
    int tail_from = 0;
    std::shared_ptr<const Options> tail;
};

template <class T> class Engine {
    // Share channels between GPUs: the reflectors of a GE followed by its T, and the TT block of a
    // merge member followed by the T of the merge.
    enum { GeChannel, MergeChannel };

    Context &ctx;
    Options opt;
    Plan plan;
    int head = 0; // panels this engine factors; a tail engine factors the rest
    std::unique_ptr<Engine<T>> tail;
    std::unique_ptr<Matrix<T>> tail_a;
    const int n, b, my_row, my_col;
    const int ldv; // leading dimension of the packed reflectors, a bound on several GPUs
    int vld;       // leading dimension of the reflectors of the panel in flight
    Tally tally;

    Stream stream;
    std::vector<std::unique_ptr<Stream>> side_streams;
    std::vector<std::unique_ptr<Event>> lane_joined, handoff;
    Event lane_fork, far_done, merge_factored, tree_done, released;
    std::unique_ptr<Updater<T>> update;

    Buffer<int> status;
    Buffer<TilePacket> packets;
    Buffer<T> tri, V, column_scale;
    // Workspaces of the register-resident TT merge between GPUs.
    Buffer<T> gram_partials, tau, overlaps;
    Buffer<int> chain_flags;
    int chain_epoch = 0;
    // Reflectors of the panels in flight, one slot of vhalf per panel: V on one GPU, the slots of
    // the GE channel on several.
    T *vslots = nullptr;
    size_t vhalf = 0, vbase = 0;
    // TT merge across GPUs: the stack of triangles, the blocks that travel between its members, the
    // summed W of every lane, and the slots of the merge channel.
    std::unique_ptr<Transport> transport;
    Buffer<T> stack, stack_v, gathered, reduced;
    T *merge_slots = nullptr;
    // Merges that gather rows: per lane the TT blocks and rows of the grid rows as gathered and
    // restacked.
    Buffer<T> merge_blocks, merge_vstack, merge_rows, merge_stack;

    // Retain only the taus of owned panels; V stays in the factored matrix.
    // T and packed V occupy two reusable workspaces and are rebuilt for Q replay.
    std::vector<DomainTree> trees;
    std::vector<size_t> tree_tau_at;
    Buffer<T> tree_t, tree_v, tree_tau;
    size_t tree_t_stride = 1, tree_v_stride = 1;
    TreeSync tree_sync, thin_q_sync;
    // The panel-after-next application of the released schedule: its stream (the last lane), the
    // flag it raises when done and the value of its last raise.
    bool released_trees = false;
    Buffer<int> applied_flag;
    int applied_value = 0;
    Buffer<T> thin_q, thin_q_z, reconstruct_work;
    // Composed WY transforms use separate generations so their far updates can
    // overlap the next group's GE/TS/TT factorizations and reconstruction.
    Buffer<T> group_v, group_t, group_gram, group_gram_sum, group_blocks, group_s, group_diagonal;
    Buffer<int> group_order;
    int group_generation = 0;
    uint64_t group_version = 0;

    Matrix<T> *factors = nullptr;
    bool factored = false;

    // On one GPU with retained tree factors, each tree kernel also releases the next panel and the
    // factors of three panels are in flight (factor_released); otherwise two.
    bool releases() const {
        return released_trees;
    }
    int slots() const {
        return releases() ? 3 : 2;
    }
    const T *tree_t_of(int k) const {
        if (my_col != plan.panels[k].owner)
            return reflectors() + size_t(vld) * b;
        return tree_t.p + (k % slots()) * tree_t_stride;
    }
    T *tree_t_of(int k) {
        if (my_col != plan.panels[k].owner)
            return reflectors() + size_t(vld) * b;
        return tree_t.p + (k % slots()) * tree_t_stride;
    }
    T *tree_v_of(int k) {
        if (my_col != plan.panels[k].owner)
            return reflectors() + size_t(vld) * b + trees[k].t_words();
        return tree_v.p + (k % slots()) * tree_v_stride;
    }

    size_t block() const {
        return size_t(b) * b;
    }
    T *reflectors() const {
        return vslots + vbase;
    }
    T *t_of(int tile) const {
        return tri.p + size_t(tile) * block();
    }
    T *ge_t_of(int k) const {
        return my_col == plan.panels[k].owner ? t_of(plan.panels[k].ge[ctx.rank].tile)
                                              : reflectors() + size_t(vld) * b;
    }
    T *merge_slot(int k) const {
        return merge_slots + (k & 1) * 2 * block();
    }
    cudaStream_t lane_stream(int lane) const {
        return update->lane(lane).stream;
    }
    int local_status() {
        return ctx.max(status.download()[0]);
    }
    // Local columns of this GPU before the global column `col`.
    int local_col(int col) const {
        return plan.cols.before(my_col, col);
    }

    // ---- column scaling --------------------------------------------------------------------------
    // Columns are scaled to unit maximum so that FP32 cannot overflow; only R is scaled back.
    void normalize_columns(Matrix<T> &x, T *scales) {
        if (!x.cols)
            return;
        tiled_column_max<<<x.cols, 256, 0, stream>>>(x.a.p, x.rows, x.cols, x.ld, scales);
        if (transport) {
            transport->bind(stream);
            transport->column_max(scales, scales, x.cols);
        }
        if (x.rows)
            tiled_column_scale<T, false><<<column_pass_grid(x.rows, x.cols), 256, 0, stream>>>(
                x.a.p, x.rows, x.cols, x.ld, scales, status.p);
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
              int ld) {
        update->gram(st, v1, ld, v2, ld, g, ldg, rows, h1, h2, gram_partials.p, gram_partials.n);
    }
    // Flops of the Householder QR of a rows x h block.
    static double ge_flops(int rows, int h) {
        return 2.0 * rows * h * h - 2.0 / 3.0 * h * h * h;
    }
    // Householder QR of a rows x h block whose columns are held in registers: the node kernel of the
    // GPU tree when the block fits it, else a chain of blocks, the overlaps V^T V and T from them.
    // V goes to v, T to the tile of the packet. Returns false, before writing anything, when the
    // block is too large.
    bool register_factor(T *a, int lda, const TilePacket *pk, int rows, int h, T *v, int ld) {
        Carrier carrier;
        const RegisterPanel<T> panel{a, lda, pk, rows, h, v, ld, tau.p, status.p, &carrier};
        if (rows <= DomainConfig::rows &&
            launch_fused_gx<T, DomainConfig::items, DomainConfig::vec,
                            DomainConfig::width / DomainConfig::vec, DomainConfig::width / 32>(
                panel, tri.p, b, stream)) {
            update->record(ProductPanel, carrier, ge_flops(rows, h));
            return true;
        }
        if (rows > 512 ||
            !launch_chain<T, 4, 16, 32>(panel, chain_flags.p, chain_epoch + 1, stream))
            return false;
        update->record(ProductPanel, carrier, ge_flops(rows, h));
        ++chain_epoch;
        gram(stream, v, v, overlaps.p, h, rows, h, h, ld);
        launch_build_t(overlaps.p, h, tau.p, pk, h, tri.p, b, stream);
        return true;
    }
    // HQR of this GPU's rows of its grid-column panel: R and GE reflectors in place,
    // packed GE reflectors in tree_reflectors(), and TS/TT V and T in the current workspace.
    // With fewer rows than the panel width, GE reduces as many columns as there are rows;
    // its reflectors update the rest of the panel, leaving R trapezoidal.
    void factor_ge(Matrix<T> &a, int panel, int release = 0, int ready_value = 0) {
        const TilePacket &pk = plan.panels[panel].ge[ctx.rank];
        if (!pk.rows)
            return;
        const DomainTree &t = trees[panel];
        update->record(ProductPanel,
                       factor_domains(a.a.p + pk.row + size_t(pk.col) * a.ld, a.ld, t,
                                      tree_reflectors(), vld, tree_v_of(panel), tree_t_of(panel),
                                      status.p, tree_sync, stream,
                                      opt.reconstruct ? nullptr : tree_tau.p + tree_tau_at[panel],
                                      release, ready_value ? applied_flag.p : nullptr, ready_value,
                                      opt.c.domain, opt.c.merge),
                       tree_flops(t));
        if (release)
            update->record_tree_release(t, release);
        const int h = plan.panels[panel].h;
        if (pk.h < h) {
            T *tail = a.a.p + pk.row + size_t(pk.col + pk.h) * a.ld;
            update->apply_tree(update->lane(0), tail, a.ld, trees[panel], tree_reflectors(), vld,
                               tree_v_of(panel), tree_t_of(panel), h - pk.h, true);
        }
    }
    // The GE reflectors of the tree: in place of the panel's operand slot, or, when the panel is
    // reconstructed, in the scratch that the reconstruction reads while it writes Y to the slot.
    T *tree_reflectors() {
        return opt.reconstruct ? thin_q.p : reflectors();
    }
    // Householder reconstruction of panel k on the GPUs of its grid column, after its TT merge.
    // Every member forms its rows of the panel's thin Q in native precision: its block E of the
    // merge's Q [I; 0] (I - T on the owner, -V_i T on the others, I without a merge) through its
    // own tree, in one launch with the owner's reconstruction: the owner factors Q_top - S = Y U
    // (modified LU) as soon as its top rows exist, scales R by S, forms T and solves its rows of
    // Y = Q U^-1, then publishes U, S and T to the other members, which solve all their rows. Y
    // replaces the factors in the panel columns: the panel becomes one compact-WY transform over
    // the rows of the grid column, whose updates sum W over it.
    void reconstruct(Matrix<T> &a, int k) {
        constexpr int B = reconstruction_width;
        const Panel &panel = plan.panels[k];
        const Merge &e = panel.merge;
        const TilePacket &pk = panel.ge[ctx.rank];
        const int mine = e.k ? e.member(ctx.rank) : 0, h = panel.h;
        T *ap = pk.rows ? a.a.p + pk.row + size_t(pk.col) * a.ld : nullptr;
        if (pk.rows) {
            const DomainTree &t = trees[k];
            thin_q_reconstruct(
                t, h, thin_q.p, vld, tree_v_of(k), tree_t_of(k), e.k ? merge_slot(k) : nullptr, b,
                mine == 0, e.k ? e.height[mine] : h, reflectors(), vld, ap, a.ld, thin_q_z.p,
                reconstruct_work.p, ge_t_of(k), b, status.p, thin_q_sync, mine == 0, stream);
            update->record(ProductPanel,
                           Carrier{}.set(GpuLevel, t.count[0], 1, 1).set(BlockLevel, 2, 4, 1),
                           thin_q_flops(t, h));
            if (mine == 0) {
                update->record(ProductPanel, Carrier{}.set(BlockLevel, 1, 8, 1),
                               2.0 / 3.0 * h * h * h);
                update->record(
                    ProductPanel,
                    Carrier{}.set(GpuLevel, ceildiv(h, 8), 1, 1).set(BlockLevel, 8, 1, 1),
                    double(h) * h * h / 3);
                if (pk.rows > h)
                    update->record(
                        ProductPanel,
                        Carrier{}.set(GpuLevel, t.count[0], 1, 1).set(BlockLevel, 8, 1, 1),
                        double(pk.rows - h) * h * h);
            }
        }
        if (transport) {
            transport->bind(stream);
            for (int i = 1; i < e.k; ++i) {
                // the h columns of U (leading dimension B) that the members' row solves read
                transport->publish(e.owner(), e.rank[i], reconstruct_work.p, reconstruct_work.p,
                                   size_t(B) * h * sizeof(T));
                transport->publish(e.owner(), e.rank[i], ge_t_of(k), ge_t_of(k),
                                   block() * sizeof(T));
            }
        }
        if (!pk.rows || !mine)
            return;
        reconstruct_rows(reflectors(), vld, ap, a.ld, pk.rows, h, reconstruct_work.p, 0, stream);
        update->record(
            ProductPanel,
            Carrier{}.set(GpuLevel, ceildiv(pk.rows, 128), 1, 1).set(BlockLevel, 4, 1, 1),
            double(pk.rows) * h * h);
    }
    // Whether the panels are compact-WY transforms over several GPUs of a grid column.
    bool spread_wy() const {
        return opt.reconstruct && plan.grid.pr > 1;
    }
    // The node cuts of such an update: W sums the partials of the grid rows, (1, pc, pr); every
    // GPU of the column forms Z, (1, pc, 1); D commits the rows of each, (pr, pc, 1).
    void at_spread_node() {
        const int pr = plan.grid.pr, pc = plan.grid.pc;
        update->at_node(Cut{1, pc, pr}, Cut{1, pc, 1}, Cut{pr, pc, 1});
    }
    // The update of a strip by a compact-WY panel whose rows are spread over the grid column: W
    // summed over its GPUs on the transport's stream between W and Z.
    typename Updater<T>::Reduce column_reduce() {
        if (!spread_wy())
            return {};
        return [this](Lane<T> &lane, T *w, size_t words) -> const T * {
            const int i = int(&lane - &update->lane(0));
            handoff[i]->record(lane.stream);
            handoff[i]->wait(transport->bound());
            transport->column_allreduce(w, lane.reduced, words);
            handoff[i]->record(transport->bound());
            handoff[i]->wait(lane.stream);
            return lane.reduced;
        };
    }
    // Leading dimension of the reflectors of panel k: fixed on one GPU; on several, the rows of
    // the GE of this grid row, so that a share moves only them.
    int v_leading(int k) const {
        return transport ? std::max(128, (plan.row_ge(k, ctx.rank).rows + 127) / 128 * 128) : ldv;
    }
    // The slots of panel k. On several GPUs a GPU writes them only after the copies of its earlier
    // shares of them have landed.
    void begin_panel(int k) {
        vbase = (transport || opt.lookahead ? k % slots() : 0) * vhalf;
        vld = v_leading(k);
        if (!transport)
            return;
        transport->bind(stream);
    }
    // The GPU hierarchy's GE reflectors, all its T factors, and its TS/TT reflectors, from the
    // panel's grid column to its peers. The slot capacity is the same on every rank even when
    // row counts differ. Peers read these factors in place until the far update releases them.
    void share_ge(int k) {
        const TilePacket &pk = plan.row_ge(k, ctx.rank);
        if (!transport || !pk.rows)
            return;
        const int owner = plan.panels[k].owner;
        const auto &tree = trees[k];
        const size_t vt = size_t(vld) * b, tw = opt.reconstruct ? block() : tree.t_words(),
                     vw = opt.reconstruct ? 0 : tree.vstack_words();
        if (my_col == owner) {
            CU(cudaMemcpyAsync(reflectors() + vt, opt.reconstruct ? ge_t_of(k) : tree_t_of(k),
                               tw * sizeof(T), cudaMemcpyDeviceToDevice, stream));
            if (vw)
                CU(cudaMemcpyAsync(reflectors() + vt + tw, tree_v_of(k), vw * sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream));
        }
        transport->bind(stream);
        transport->row_share(GeChannel, k & 1, owner, (vt + tw + vw) * sizeof(T));
    }
    // Applies the reflectors of the GE of a panel on this grid row to the columns [lo, hi) of x.
    // Every grid row applies its own GE, with the columns cut among the grid columns.
    void apply_ge(Matrix<T> &x, int panel, int lo, int hi, int first_lane, bool transpose) {
        const TilePacket &pk = plan.row_ge(panel, ctx.rank);
        if (lo >= hi || (!pk.rows && !spread_wy()))
            return;
        const int h = plan.panels[panel].h;
        const DomainTree &tree = trees[panel];
        if (spread_wy()) {
            at_spread_node();
            transport->bind(nullptr);
            const auto reduce = column_reduce();
            for_strips(lo, hi, first_lane, [&](Lane<T> &lane, int col, int q) {
                T *xi = pk.rows ? x.a.p + pk.row + size_t(col) * x.ld : nullptr;
                update->apply(lane, xi, x.ld, reflectors(), vld, ge_t_of(panel), pk.rows, h, q,
                              transpose, 0, reduce);
            });
            transport->bind(stream);
            update->at_node(Cut{});
            return;
        }
        update->at_node(Cut{1, plan.grid.pc, 1});
        for_strips(lo, hi, first_lane, [&](Lane<T> &lane, int col, int q) {
            T *xi = x.a.p + pk.row + size_t(col) * x.ld;
            if (opt.reconstruct)
                update->apply(lane, xi, x.ld, reflectors(), vld, ge_t_of(panel), pk.rows, pk.h, q,
                              transpose);
            else
                update->apply_tree(lane, xi, x.ld, tree, reflectors(), vld, tree_v_of(panel),
                                   tree_t_of(panel), q, transpose);
        });
        update->at_node(Cut{});
    }

    // ---- composition of reconstructed HQR panels -----------------------------------------------
    int group_size(size_t k) const {
        if (opt.aggregate < 2 || plan.panels[k].h != b)
            return 1;
        int g = 1;
        while (g < opt.aggregate && k + g < plan.panels.size() && plan.panels[k + g].h == b)
            ++g;
        // A group pays for composition only when there are columns to its right.
        while (g > 1 && plan.panels[k + g - 1].col + b >= n)
            --g;
        return g;
    }
    int unit_end(size_t k) const {
        if (k >= plan.panels.size())
            return n;
        const Panel &last = plan.panels[k + group_size(k) - 1];
        return last.col + last.h;
    }
    T *group_vectors() const {
        return group_v.p + size_t(group_generation) * ldv * b * opt.aggregate;
    }
    T *group_triangular() const {
        const size_t width = size_t(b) * opt.aggregate;
        return group_t.p + group_generation * width * width;
    }
    void compose_group(size_t k, int g) {
        const TilePacket &first = plan.row_ge(int(k), ctx.rank);
        const int h = g * b;
        T *vg = group_vectors(), *tg = group_triangular();
        // Natural reflector order keeps T triangular. Each panel was appended
        // after its row broadcast, so this also works across grid columns; with the rows of the
        // reflectors over the grid rows, V^T V sums over the grid column.
        T *gg = group_gram.p;
        if (first.rows)
            gram(stream, vg, vg, gg, h, first.rows, h, h, ldv);
        else
            CU(cudaMemsetAsync(gg, 0, size_t(h) * h * sizeof(T), stream));
        if (spread_wy()) {
            transport->bind(stream);
            transport->column_allreduce(gg, group_gram_sum.p, size_t(h) * h);
            gg = group_gram_sum.p;
        }
        CU(cudaMemsetAsync(tg, 0, size_t(h) * h * sizeof(T), stream));
        for (int j = 0; j < g; ++j) {
            const T *tj = group_diagonal.p + size_t(j) * b * b;
            if (j)
                gather_gram_blocks<<<ceildiv(j * b * b, 256), 256, 0, stream>>>(
                    gg, h, group_order.p, j, b, group_blocks.p);
            if (j && b % kComposeTile == 0) {
                const int jb = j * b;
                const dim3 grid(ceildiv(jb, kComposeTile), b / kComposeTile),
                    threads(kComposeTile, 8);
                compose_s_tiled<<<grid, threads, 0, stream>>>(group_blocks.p, b, jb, tj, group_s.p);
                compose_n_tiled<<<grid, threads, 0, stream>>>(tg, h, jb, b, group_s.p);
                compose_diag_copy<<<ceildiv(b * b, 256), 256, 0, stream>>>(tg, h, j, b, tj);
                const Carrier c =
                    Carrier{}.set(GpuLevel, grid.x, grid.y, 1).set(BlockLevel, 1, 8, 1);
                update->record(ProductGram, c, double(jb) * b * b);
                update->record(ProductGram, c, double(jb) * jb * b);
            } else {
                compose_tg_step<<<b, 128, size_t(std::max(1, j) * b) * sizeof(T), stream>>>(
                    tg, h, j, b, group_blocks.p, tj);
                if (j)
                    update->record(ProductGram,
                                   Carrier{}.set(GpuLevel, 1, b, 1).set(BlockLevel, 4, 1, 1),
                                   double(j * b) * b * b + double(j * b) * (j * b) * b);
            }
        }
        ++group_version;
        CU(cudaGetLastError());
    }
    void apply_group(Matrix<T> &a, size_t k, int g, int lo, int hi, int first_lane) {
        if (lo >= hi)
            return;
        const TilePacket &first = plan.row_ge(int(k), ctx.rank);
        if (spread_wy()) {
            at_spread_node();
            transport->bind(nullptr);
        } else
            update->at_node(Cut{1, plan.grid.pc, 1});
        const auto reduce = column_reduce();
        for_strips(lo, hi, first_lane, [&](Lane<T> &lane, int col, int q) {
            update->apply(lane, a.a.p + first.row + size_t(col) * a.ld, a.ld, group_vectors(), ldv,
                          group_triangular(), first.rows, g * b, q, true, group_version, reduce);
        });
        if (spread_wy())
            transport->bind(stream);
        update->at_node(Cut{});
    }

    // ---- TT merge across GPUs -------------------------------------------------------------------
    int copy_blocks(int rows, int columns) const {
        return std::max(1, std::min(512, ceildiv(rows * columns, 128)));
    }
    // Gathers the triangles of the members on the owner, factors their stack with the register
    // chain and returns R to the owner's rows. Every other member receives its block of V, followed
    // by the T of the merge, in its merge slot, and the block replaces its triangle.
    void factor_merge(const Merge &e, Matrix<T> &a, int k) {
        const int h = e.h, K = e.k, owner = e.owner(), mine = e.member(ctx.rank);
        const size_t block = this->block(), stride = 2 * block;
        T *slot = merge_slot(k);
        cudaStream_t st = stream;
        transport->bind(st);
        if (mine == 0) {
            stack.zero(st);
            extract_r<<<8, 128, 0, st>>>(a.a.p, a.ld, e.row[0], e.col, e.height[0], h, stack.p,
                                         K * b);
        } else if (mine > 0) {
            T *r = gathered.p + (mine - 1) * stride;
            CU(cudaMemsetAsync(r, 0, block * sizeof(T), st));
            extract_r<<<8, 128, 0, st>>>(a.a.p, a.ld, e.row[mine], e.col, e.height[mine], h, r, b);
        }
        for (int i = 1; i < K; ++i) {
            T *r = gathered.p + (i - 1) * stride;
            // The members produce these triangles independently into disjoint owner slots.
            // NVSHMEM's signal orders each receive before the stack consumes it; credits
            // protect the sender's staging buffer without imposing an order across senders.
            transport->publish_unordered(e.rank[i], owner, r, r, size_t(b) * h * sizeof(T));
        }
        if (mine == 0) {
            int off = e.height[0];
            for (int i = 1; i < K; off += e.height[i++])
                CU(cudaMemcpy2DAsync(stack.p + off, K * b * sizeof(T),
                                     gathered.p + (i - 1) * stride, b * sizeof(T),
                                     e.height[i] * sizeof(T), h, cudaMemcpyDeviceToDevice, st));
            if (!register_factor(stack.p, K * b, packets.p + e.tile, K * b, h, stack_v.p, K * b))
                throw std::runtime_error("no kernel for the stack of triangles");
            scatter_upper<<<8, 128, 0, st>>>(stack.p, K * b, a.a.p, a.ld, e.row[0], e.col, h, h);
            tile_identity<<<copy_blocks(b, b), 128, 0, st>>>(slot, b, h);
            CU(cudaMemcpyAsync(slot + block, t_of(e.tile), block * sizeof(T),
                               cudaMemcpyDeviceToDevice, st));
            off = e.height[0];
            for (int i = 1; i < K; off += e.height[i++]) {
                T *r = gathered.p + (i - 1) * stride;
                CU(cudaMemsetAsync(r, 0, block * sizeof(T), st));
                CU(cudaMemcpy2DAsync(r, b * sizeof(T), stack.p + off, K * b * sizeof(T),
                                     e.height[i] * sizeof(T), h, cudaMemcpyDeviceToDevice, st));
                CU(cudaMemcpyAsync(r + block, slot + block, block * sizeof(T),
                                   cudaMemcpyDeviceToDevice, st));
            }
        }
        for (int i = 1; i < K; ++i) {
            T *r = gathered.p + (i - 1) * stride;
            transport->publish(owner, e.rank[i], r, slot, stride * sizeof(T));
        }
        if (mine > 0)
            scatter_upper<<<8, 128, 0, st>>>(slot, b, a.a.p, a.ld, e.row[mine], e.col,
                                             e.height[mine], h);
    }
    // The merge slot again, from the factors: the T of the owner and the TT block of every member,
    // read back from its triangle rows.
    void restore_merge_slot(const Merge &e, int k) {
        const int mine = e.member(ctx.rank);
        T *t = merge_slot(k) + block();
        transport->bind(stream);
        if (mine == 0) {
            tile_identity<<<copy_blocks(b, b), 128, 0, stream>>>(merge_slot(k), b, e.h);
            CU(cudaMemcpyAsync(t, t_of(e.tile), block() * sizeof(T), cudaMemcpyDeviceToDevice,
                               stream));
        }
        for (int i = 1; i < e.k; ++i)
            transport->publish(e.owner(), e.rank[i], t, t, block() * sizeof(T));
        if (mine > 0)
            tile_bottom_v<<<8, 128, 0, stream>>>(factors->a.p, factors->ld, e.row[mine], e.col,
                                                 e.height[mine], e.h, merge_slot(k), b);
    }
    // The merge slot of every member, from the member to the other GPUs of its grid row; the GPUs
    // of grid rows without rows of the panel hold a zero block.
    void share_merge(int k) {
        if (plan.row_member(k, ctx.rank) < 0) {
            CU(cudaMemsetAsync(merge_slot(k), 0, block() * sizeof(T), stream));
            return;
        }
        transport->bind(stream);
        transport->row_share(MergeChannel, k & 1, plan.panels[k].owner, 2 * block() * sizeof(T));
    }
    // Applies the TT elimination of panel k to the columns [lo, hi) of x. The reflector block of
    // its first member is the identity, so W sums a copy of that member's rows and V_i^T X_i of
    // the others. Where the bound admits it (pr^2 <= pc) every member forms its V_i^T X_i and an
    // all-reduce within the grid column sums them: a carrier (1, pc, K) at the node. Otherwise the
    // GPUs of the column gather the rows of every grid row and each contracts the whole stack with
    // the stacked TT blocks: (1, pc, 1). Every member then computes Z = op(T) W and commits
    // X_i -= V_i Z on its own rows.
    void apply_merge(const Merge &e, int k, Matrix<T> &x, int lo, int hi, int first_lane,
                     bool transpose) {
        if (lo >= hi)
            return;
        const int h = e.h, K = e.k, pr = plan.grid.pr, pc = plan.grid.pc, rows = pr * b;
        const int mine = plan.row_member(k, ctx.rank);
        const bool reduce = plan.grid.column_reduces();
        const T *vi = merge_slot(k), *t = vi + block();
        const bool transposed = native_d_admits<T>(h);
        const int depth = std::min(opt.depth, ceildiv(hi - lo, opt.strip));
        // The arithmetic runs on the lanes and the collectives on the transport's own stream, in
        // column order; events hand every strip over and back.
        cudaStream_t first = lane_stream(first_lane);
        auto hand = [&](int lane, cudaStream_t from, cudaStream_t to) {
            handoff[lane]->record(from);
            handoff[lane]->wait(to);
        };
        transport->bind(nullptr);
        hand(first_lane, first, transport->bound());
        T *vstack = nullptr;
        if (!reduce) {
            // The TT blocks of the grid rows, stacked: the identity for the first member and
            // zeros where a grid row holds no rows of the panel.
            T *blocks = merge_blocks.p + size_t(first_lane) * rows * b;
            vstack = merge_vstack.p + size_t(first_lane) * rows * b;
            transport->column_allgather(vi, blocks, block());
            hand(first_lane, transport->bound(), first);
            restack<<<copy_blocks(rows, b), 128, 0, first>>>(blocks, vstack, pr, b, b);
        }
        fork_lanes(first_lane, depth);
        for (int col = lo, strips = 0; col < hi; ++strips) {
            const int i = first_lane + strips % depth, q = std::min(opt.strip, hi - col);
            Lane<T> &lane = update->lane(i);
            cudaStream_t ls = lane.stream;
            T *w = lane.w, *z = lane.z;
            T *xi = mine >= 0 ? x.a.p + e.row[mine] + size_t(col) * x.ld : nullptr;
            if (reduce) {
                T *partial = w;
                w = lane.reduced;
                if (mine < 0 || h < b)
                    CU(cudaMemsetAsync(partial, 0, size_t(b) * q * sizeof(T), ls));
                if (mine == 0)
                    tile_copy_identity<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0],
                                                                          col, h, q, partial, b);
                else if (mine > 0) {
                    update->at_node(Cut{1, pc, K});
                    update->record(
                        ProductW,
                        update->stage_w(lane, vi, b, xi, x.ld, partial, b, e.height[mine], h, q),
                        product_flops(h, q, e.height[mine]));
                }
                hand(i, ls, transport->bound());
                transport->column_allreduce(partial, w, size_t(b) * q);
                hand(i, transport->bound(), ls);
            } else {
                T *gathered = merge_rows.p + size_t(i) * rows * opt.strip,
                  *stack = merge_stack.p + size_t(i) * rows * opt.strip,
                  *own = gathered + size_t(my_row) * b * q;
                const int height = mine >= 0 ? e.height[mine] : 0;
                if (height < b)
                    CU(cudaMemsetAsync(own, 0, size_t(b) * q * sizeof(T), ls));
                if (height)
                    CU(cudaMemcpy2DAsync(own, size_t(b) * sizeof(T), xi, size_t(x.ld) * sizeof(T),
                                         size_t(height) * sizeof(T), q, cudaMemcpyDeviceToDevice,
                                         ls));
                hand(i, ls, transport->bound());
                transport->column_allgather(own, gathered, size_t(b) * q);
                hand(i, transport->bound(), ls);
                if (mine >= 0) {
                    restack<<<copy_blocks(rows, q), 128, 0, ls>>>(gathered, stack, pr, b, q);
                    update->at_node(Cut{1, pc, 1});
                    update->record(
                        ProductW,
                        update->stage_w(lane, vstack, rows, stack, rows, w, b, rows, h, q),
                        product_flops(h, q, rows));
                }
            }
            if (mine >= 0) {
                update->at_node(Cut{1, pc, 1});
                update->record(ProductZ,
                               update->stage_z(lane, t, w, z, b, h, q, transpose, transposed),
                               product_flops(h, q, h) / 2);
            }
            if (mine == 0) {
                if (transposed)
                    tile_add_identity_t<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0],
                                                                           col, h, q, z, q);
                else
                    tile_add_identity<<<copy_blocks(h, q), 128, 0, ls>>>(x.a.p, x.ld, e.row[0], col,
                                                                         h, q, z, b);
            } else if (mine > 0) {
                update->at_node(Cut{K, pc, 1});
                update->record(ProductD,
                               update->stage_d(lane, vi, b, z, b, xi, x.ld, e.height[mine], h, q),
                               product_flops(e.height[mine], q, h));
            }
            col += q;
        }
        join_lanes(first_lane, depth);
        update->at_node(Cut{});
        transport->bind(stream);
    }

    // ---- schedule -------------------------------------------------------------------------------
    // One GPU, retained tree factors: tree kernel k factors panel k and releases panel k + 1 (its
    // update by panel k); the edge lane then applies panel k to panel k + 2 once the far update of
    // panel k - 1 has passed it, and the far lane applies panel k to the rest. Tree kernel k + 1
    // factors meanwhile and waits for that application only before its own release, so the far
    // updates trail by two panels.
    void factor_released(Matrix<T> &a) {
        const int panels = int(plan.panels.size()), far_lane = opt.depth, edge_lane = 2 * opt.depth;
        cudaStream_t edge = lane_stream(edge_lane);
        bool far_pending = false;
        int ready = 0; // value of the flag the next tree's release waits for
        for (int k = 0; k < panels; ++k) {
            const Panel &panel = plan.panels[k];
            const int end = panel.col + panel.h, next = std::min(n, unit_end(k + 1)),
                      after = std::min(n, unit_end(k + 2));
            begin_panel(k);
            factor_ge(a, k, panel.h == b ? next - end : 0, ready);
            tree_done.record(stream);
            ready = 0;
            if (after > next) {
                tree_done.wait(edge);
                if (far_pending)
                    far_done.wait(edge);
                apply_ge(a, k, local_col(next), local_col(after), edge_lane, true);
                tree_signal<<<1, 1, 0, edge>>>(applied_flag.p, ++applied_value); // after the join
                CU(cudaGetLastError());
                ready = applied_value;
            }
            if (after < n) {
                tree_done.wait(lane_stream(far_lane));
                apply_ge(a, k, local_col(after), a.cols, far_lane, true);
                far_done.record(lane_stream(far_lane));
                far_pending = true;
            }
        }
        released.record(edge);
        released.wait(stream);
        if (far_pending)
            far_done.wait(stream);
        vbase = 0;
    }
    void factor_panels(Matrix<T> &a) {
        if (releases()) {
            factor_released(a);
            return;
        }
        const int far_lane = opt.lookahead ? opt.depth : 0;
        bool far_pending = false, far_reads_v = false;
        auto wait_far = [&] {
            if (far_pending)
                far_done.wait(stream);
            far_pending = far_reads_v = false;
        };
        auto fork_far = [&] {
            lane_fork.record(stream);
            lane_fork.wait(lane_stream(far_lane));
        };
        for (size_t k = 0; k < size_t(head); ++k) {
            const Panel &panel = plan.panels[k];
            if (const int g = group_size(k); g > 1) {
                if (far_reads_v)
                    wait_far();
                const int end = unit_end(k);
                const TilePacket &first = plan.row_ge(int(k), ctx.rank);
                if (opt.lookahead)
                    group_generation ^= 1;
                for (int j = 0; j < g; ++j) {
                    const Panel &pj = plan.panels[k + j];
                    begin_panel(int(k) + j);
                    factor_ge(a, int(k) + j);
                    if (pj.merge.k)
                        factor_merge(pj.merge, a, int(k) + j);
                    reconstruct(a, int(k) + j);
                    share_ge(int(k) + j);
                    // Panel k + j starts below the first panel's local rows by the local rows of
                    // this grid row in between (j b on one GPU); zero where it has none here.
                    const TilePacket &pkj = plan.row_ge(int(k) + j, ctx.rank);
                    const int offset = pkj.rows ? pkj.row - first.row : first.rows;
                    append_group_panel<<<pack_v_grid(first.rows, b), 128, 0, stream>>>(
                        reflectors(), vld, ge_t_of(int(k) + j), b, pkj.rows, offset, j * b,
                        group_vectors(), ldv, group_diagonal.p + size_t(j) * b * b);
                    apply_ge(a, int(k) + j, local_col(pj.col + pj.h), local_col(end), 0, true);
                }
                compose_group(k, g);
                wait_far();
                const int release = opt.lookahead ? unit_end(k + g) : n;
                apply_group(a, k, g, local_col(end), local_col(release), 0);
                if (local_col(release) < a.cols) {
                    fork_far();
                    apply_group(a, k, g, local_col(release), a.cols, far_lane);
                    far_done.record(lane_stream(far_lane));
                    far_pending = true;
                }
                k += g - 1;
                continue;
            }
            const int end = panel.col + panel.h;
            // Columns the next panel needs before it can be factored, and
            // this GPU's trailing columns before and after them.
            const int release =
                opt.lookahead && k + 1 < plan.panels.size() && end < n ? unit_end(k + 1) : n;
            const int lo = local_col(end), mid = local_col(release), hi = a.cols;
            const Merge &merge = panel.merge;
            // A compact-WY panel absorbs its TT merge: reconstruction follows the merge and its
            // updates sum W over the grid column instead of applying the merge separately.
            const bool merged = merge.k && !opt.reconstruct;
            begin_panel(int(k));
            factor_ge(a, int(k));
            if (opt.reconstruct) {
                if (merge.k)
                    factor_merge(merge, a, int(k));
                reconstruct(a, int(k));
            }
            share_ge(int(k));
            if (merged) {
                // The gather, the TT and the shares overlap the far update of the previous panel.
                factor_merge(merge, a, int(k));
                share_merge(int(k));
            }
            wait_far();
            apply_ge(a, int(k), lo, mid, 0, true);
            if (mid < hi) {
                fork_far();
                apply_ge(a, int(k), mid, hi, far_lane, true);
            }
            if (merged)
                apply_merge(merge, int(k), a, lo, mid, 0, true);
            if (mid < hi) {
                if (merged) {
                    merge_factored.record(stream);
                    merge_factored.wait(lane_stream(far_lane));
                    apply_merge(merge, int(k), a, mid, hi, far_lane, true);
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
        : ctx(c), opt(o), plan(m, n_, make_grid(c.size, o.grid_rows), o.b), n(n_), b(o.b),
          my_row(plan.grid.row(c.rank)), my_col(plan.grid.col(c.rank)),
          ldv((plan.rows.local(0) + 127) / 128 * 128), vld(ldv),
          stream(o.lookahead ? Stream::greatest_priority() : 0) {
        if (m < n || b < 1 || b > DomainConfig::width || opt.strip < 1 || opt.depth < 1 ||
            opt.c.w < 2 || opt.c.z < 2 || opt.aggregate < 1 || opt.aggregate > 16)
            throw std::runtime_error("invalid schedule options");
        if (opt.domain_tiles < 1 || opt.c.domain < 1 || opt.c.merge < 1 || opt.c.tree_groups < 1)
            throw std::runtime_error("domain tiles and tree replications must be positive");
        if (opt.aggregate > 1 && !opt.reconstruct)
            throw std::runtime_error(
                "aggregation requires reconstructed HQR panels (--panel-format wy)");
        if (ctx.size > 1 && !opt.lookahead)
            throw std::runtime_error("several GPUs need look-ahead");
        if (opt.reconstruct && plan.grid.pr > 1 && !plan.grid.column_reduces())
            throw std::runtime_error(
                "compact-WY panels over grid rows need pr^2 <= pc: their W sums over the column");
        const size_t tiles = plan.tiles[ctx.rank], block = this->block();
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
        // lanes depth .. 2 depth - 1 carry the far update at the lowest priority, and the released
        // schedule adds edge lanes 2 depth .. 3 depth - 1 at the highest.
        released_trees = ctx.size == 1 && opt.lookahead && !opt.reconstruct && opt.aggregate == 1 &&
                         b == DomainConfig::width;
        const int lanes = opt.depth * (opt.lookahead ? 2 : 1) + (released_trees ? opt.depth : 0);
        std::vector<cudaStream_t> streams{stream.s};
        lane_joined.resize(lanes);
        for (int i = 1; i < lanes; ++i) {
            side_streams.push_back(opt.lookahead && (i < opt.depth || i >= 2 * opt.depth)
                                       ? std::make_unique<Stream>(Stream::greatest_priority())
                                       : std::make_unique<Stream>());
            streams.push_back(side_streams.back()->s);
            lane_joined[i] = std::make_unique<Event>();
        }
        update = std::make_unique<Updater<T>>(b, opt.aggregate, opt.strip, ldv, opt.c, opt.d_tiles,
                                              streams, opt.lookahead ? opt.depth : -1, tally,
                                              opt.resident_tree);
        column_scale.alloc(n);
        tri.alloc(tiles * block);
        gram_partials.alloc(128 * block);
        tau.alloc(b);
        overlaps.alloc(block);
        if (opt.reconstruct) {
            thin_q.alloc(size_t(ldv) * b);
            reconstruct_work.alloc(reconstruction_width * (reconstruction_width + 1));
        }
        if (opt.aggregate > 1) {
            const size_t width = size_t(b) * opt.aggregate, generations = opt.lookahead ? 2 : 1;
            group_v.alloc(generations * ldv * width);
            group_t.alloc(generations * width * width);
            group_gram.alloc(width * width);
            if (plan.grid.pr > 1)
                group_gram_sum.alloc(width * width);
            group_blocks.alloc(width * b);
            group_s.alloc(width * b);
            group_diagonal.alloc(width * b);
            std::vector<int> order(width);
            for (size_t i = 0; i < width; ++i)
                order[i] = int(i);
            group_order.alloc(width);
            group_order.upload(order, stream);
        }
        chain_flags.alloc(b);
        CU(cudaMemset(chain_flags.p, 0, b * sizeof(int)));
        size_t tree_channel_words = 0;
        {
            if (opt.fan_in < 2)
                throw std::runtime_error("tree fan-in must be at least two");
            const int tile = 64;
            const int domain = opt.domain_tiles == 1 ? domain_height(tile, b)
                                                     : (DomainConfig::rows - b) / tile * tile;
            size_t tau_words = 0;
            int segments = 1, nodes = 1;
            auto make_tree = [&](const TilePacket &pk) {
                return !pk.rows ? DomainTree{}
                                : domain_tree(pk.rows, pk.h, domain, opt.fan_in, opt.domain_tiles);
            };
            for (size_t k = 0; k < plan.panels.size(); ++k) {
                const TilePacket &pk = plan.row_ge(int(k), ctx.rank);
                trees.push_back(make_tree(pk));
                tree_tau_at.push_back(tau_words);
                if (my_col == plan.panels[k].owner) {
                    if (!opt.reconstruct)
                        tau_words += size_t(trees.back().nodes()) * trees.back().h;
                    tree_t_stride = std::max(tree_t_stride, trees.back().t_words());
                    tree_v_stride = std::max(tree_v_stride, trees.back().vstack_words());
                }
                segments = std::max(segments, trees.back().count[0]);
                nodes = std::max(nodes, trees.back().nodes());
                // NCCL window capacities must match across all ranks, including rows
                // with a shorter last tile and columns that never own a particular panel.
                for (int r = 0; r < plan.grid.pr; ++r) {
                    const auto &other = plan.row_ge(int(k), plan.grid.rank(r, 0));
                    if (!other.rows)
                        continue;
                    const auto tree = make_tree(other);
                    tree_channel_words =
                        std::max(tree_channel_words,
                                 opt.reconstruct ? block : tree.t_words() + tree.vstack_words());
                }
            }
            tree_sync.reserve(nodes, stream);
            if (opt.reconstruct) {
                thin_q_sync.reserve(nodes + 1, stream);
                thin_q_z.alloc(size_t(nodes) * reconstruction_width * reconstruction_width);
            }
            // Released trees must leave the other streams free SMs while their blocks wait.
            if (nodes >= device_properties().multiProcessorCount)
                released_trees = false;
            if (released_trees) {
                applied_flag.alloc(1);
                applied_flag.zero(stream);
            }
            tree_tau.alloc(std::max<size_t>(1, tau_words));
            tree_t.alloc(slots() * tree_t_stride);
            tree_v.alloc(slots() * tree_v_stride);
            update->reserve_trees(segments, b,
                                  opt.domain_tiles > 1 && opt.c.tree_groups > 1 ? DomainConfig::rows
                                                                                : 0);
        }
        head = int(plan.panels.size());
        if (opt.tail_from) {
            const int s = opt.tail_from;
            if (ctx.size > 1 || !opt.reconstruct || s % (b * opt.aggregate) || s >= n)
                throw std::runtime_error("a tree-format tail needs one GPU, compact-WY panels and "
                                         "a first column inside the matrix at a group boundary");
            head = 0;
            while (head < int(plan.panels.size()) && plan.panels[head].col < s)
                ++head;
            Options t = opt.tail ? *opt.tail : opt;
            t.reconstruct = false;
            t.aggregate = 1;
            t.tail_from = 0;
            t.tail = nullptr;
            tail = std::make_unique<Engine<T>>(ctx, m - s, n - s, t);
            tail_a = std::make_unique<Matrix<T>>(m - s, n - s, m - s, n - s);
        }
        if (ctx.size == 1) {
            vhalf = size_t(ldv) * b;
            V.alloc((opt.lookahead ? slots() : 1) * vhalf);
            vslots = V.p;
            return;
        }
        // On several GPUs NCCL broadcasts fill the local operand slots: a slot of
        // the GE channel holds the reflectors and T, one of the merge channel a TT block and T.
        const int pr = plan.grid.pr;
        vhalf = (size_t(ldv) * b + tree_channel_words + 63) / 64 * 64;
        stack.alloc(pr * block);
        stack_v.alloc(pr * block);
        gathered.alloc(std::max(1, pr - 1) * 2 * block);
        // Over several grid rows, the partial W summed over the grid column: of a TT merge (b
        // columns) or of a compact-WY panel or group (b aggregate); other merges gather rows.
        const size_t wide = size_t(b) * (opt.reconstruct ? opt.aggregate : 1),
                     column_words = pr > 1 ? wide * std::max<size_t>(opt.strip, wide) : 0;
        if (pr > 1 && plan.grid.column_reduces()) {
            reduced.alloc(size_t(lanes) * wide * opt.strip);
            for (int i = 0; i < lanes; ++i)
                update->lane(i).reduced = reduced.p + size_t(i) * wide * opt.strip;
        } else if (pr > 1) {
            merge_blocks.alloc(size_t(lanes) * pr * block);
            merge_vstack.alloc(size_t(lanes) * pr * block);
            merge_rows.alloc(size_t(lanes) * pr * b * opt.strip);
            merge_stack.alloc(size_t(lanes) * pr * b * opt.strip);
        }
        for (int i = 0; i < lanes; ++i)
            handoff.push_back(std::make_unique<Event>());
        // Publications: a TT block and T, or the U of a panel's reconstruction.
        const size_t published =
            std::max(2 * block, opt.reconstruct ? size_t(reconstruction_width) * b : size_t(0));
        transport = std::make_unique<Transport>(
            ctx, published * sizeof(T),
            std::max({size_t(plan.cols.local(0)), block, column_words}) * sizeof(T), my_row, my_col,
            plan.grid.pc, std::vector<size_t>{vhalf * sizeof(T), 2 * block * sizeof(T)});
        vslots = static_cast<T *>(transport->slot(GeChannel, 0));
        merge_slots = static_cast<T *>(transport->slot(MergeChannel, 0));
        transport->bind(stream);
    }
    // The distribution of the matrix: its rows over the grid rows, its columns over the grid
    // columns.
    const Plan &layout() const {
        return plan;
    }
    // Products issued by the last factorization or application of Q.
    const Tally &products() const {
        return tally;
    }
    json communication() const {
        return transport ? transport->report() : json::object();
    }
    json update_workspace() const {
        json memory = update->workspace();
        size_t bytes = memory["total_bytes"].get<size_t>();
        if (tail) {
            memory["tail"] = tail->update_workspace();
            memory["tail_matrix_bytes"] = tail_a->a.n * sizeof(T);
            bytes += memory["tail"]["inclusive_bytes"].get<size_t>() + tail_a->a.n * sizeof(T);
        }
        memory["inclusive_bytes"] = bytes;
        return memory;
    }
    // Factors a in place: R in its upper triangle, the reflectors below it, their T factors inside
    // the engine. Returns a DeviceStatus.
    int factor(Matrix<T> &a) {
        factors = &a;
        factored = false;
        tally = Tally{};
        status.zero(stream);
        finite_scan_2d<<<column_pass_grid(a.rows, a.cols), 256, 0, stream>>>(a.a.p, a.rows, a.cols,
                                                                             a.ld, status.p);
        stream.sync();
        if (int st = local_status())
            return st;
        normalize_columns(a, column_scale.p);
        factor_panels(a);
        if (tail)
            if (int st = through_tail(a, nullptr))
                return st;
        tiled_column_scale<T, true, true, true>
            <<<column_pass_grid(a.rows, a.cols), 256, 0, stream>>>(
                a.a.p, a.rows, a.cols, a.ld, column_scale.p, status.p, plan.rows, my_row, plan.cols,
                my_col);
        stream.sync();
        const int st = local_status();
        factored = st == OK;
        return st;
    }
    // x <- Q x or Q^T x with the factors of the last factorization. x has the rows of the factored
    // matrix on this GPU and any number of columns, which the GPUs of a grid column share.
    int apply_q(Matrix<T> &x, bool transpose) {
        if (!factored || x.m != plan.rows.n || x.rows != factors->rows)
            throw std::runtime_error("apply_q needs a factorization of matching rows");
        tally = Tally{};
        // Q = Q_head diag(I, Q_tail)
        if (tail && !transpose)
            if (int st = through_tail(x, &transpose))
                return st;
        const int st = apply_head(x, transpose);
        return st || !tail || !transpose ? st : through_tail(x, &transpose);
    }

  private:
    void copy_block(T *dst, int ldd, const T *src, int lds, int rows, int cols) {
        CU(cudaMemcpy2DAsync(dst, size_t(ldd) * sizeof(T), src, size_t(lds) * sizeof(T),
                             size_t(rows) * sizeof(T), size_t(cols), cudaMemcpyDeviceToDevice,
                             stream));
    }
    // The rows from tail_from on of x through the tail engine: its factorization of the trailing
    // block of the factored matrix (transpose null), or its Q or Q^T applied to those rows of x.
    int through_tail(Matrix<T> &x, const bool *transpose) {
        const int s = opt.tail_from, cols = transpose ? x.cols : x.cols - s;
        T *rows = x.a.p + s + (transpose ? 0 : size_t(s) * x.ld);
        std::unique_ptr<Matrix<T>> applied;
        if (transpose)
            applied = std::make_unique<Matrix<T>>(x.m - s, x.n, x.rows - s, x.cols);
        Matrix<T> &y = transpose ? *applied : *tail_a;
        copy_block(y.a.p, y.ld, rows, x.ld, x.rows - s, cols);
        stream.sync();
        const int st = transpose ? tail->apply_q(y, *transpose) : tail->factor(y);
        for (const auto &e : tail->products().entries)
            tally.entries.push_back(e);
        copy_block(rows, x.ld, y.a.p, y.ld, x.rows - s, cols);
        stream.sync();
        return st;
    }
    int apply_head(Matrix<T> &x, bool transpose) {
        Buffer<T> scales(std::max(1, x.cols));
        status.zero(stream);
        finite_scan_2d<<<column_pass_grid(x.rows, x.cols), 256, 0, stream>>>(x.a.p, x.rows, x.cols,
                                                                             x.ld, status.p);
        stream.sync();
        if (int st = local_status())
            return st;
        normalize_columns(x, scales.p);
        const int panels = head;
        for (int i = 0; i < panels; ++i) {
            const int k = transpose ? i : panels - 1 - i;
            const Panel &panel = plan.panels[k];
            begin_panel(k);
            auto ge = [&] {
                const TilePacket &pk = panel.ge[ctx.rank];
                if (pk.rows) {
                    // Below another GPU's diagonal block a GPU holds whole rows of the panel's Y.
                    const bool below =
                        opt.reconstruct && panel.merge.k && panel.merge.member(ctx.rank) > 0;
                    if (below)
                        pack_rows<<<pack_v_grid(vld, b), 128, 0, stream>>>(
                            factors->a.p, factors->ld, packets.p + pk.tile, reflectors(), vld,
                            panel.h);
                    else if (opt.reconstruct)
                        pack_ge_v<<<pack_v_grid(vld, b), 128, 0, stream>>>(
                            factors->a.p, factors->ld, packets.p + pk.tile, reflectors(), vld, b);
                    else
                        restore_tree(factors->a.p + pk.row + size_t(pk.col) * factors->ld,
                                     factors->ld, trees[k], tree_tau.p + tree_tau_at[k],
                                     reflectors(), vld, tree_v_of(k), tree_t_of(k), stream);
                }
                share_ge(k);
                apply_ge(x, k, 0, x.cols, 0, transpose);
            };
            auto merge = [&] {
                if (!panel.merge.k || opt.reconstruct)
                    return;
                restore_merge_slot(panel.merge, k);
                share_merge(k);
                apply_merge(panel.merge, k, x, 0, x.cols, 0, transpose);
            };
            if (transpose) {
                ge();
                merge();
            } else {
                merge();
                ge();
            }
        }
        vbase = 0;
        if (x.cols)
            tiled_column_scale<T, true><<<column_pass_grid(x.rows, x.cols), 256, 0, stream>>>(
                x.a.p, x.rows, x.cols, x.ld, scales.p, status.p);
        stream.sync();
        return local_status();
    }
};
} // namespace tqr
