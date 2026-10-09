#pragma once
// The elimination list (Algorithm 1) on a pr x pc grid of GPUs. The rows of the matrix are
// block-cyclic over the grid rows and its columns over the grid columns, in blocks of b. Panel k
// lives on the grid column that owns its columns: every GPU of that column reduces its rows of the
// panel with one GE, and one TT elimination of arity pr merges their triangles. The reflectors of
// each GE travel along its grid row to the GPUs that hold the trailing columns of those rows.
#include "kernels.cuh"

namespace qr_omega {
inline constexpr int max_gpus = 8;

// The GPUs of the node as a pr x pc grid, row-major: GPU `rank` sits in grid row rank / pc and
// grid column rank % pc.
struct Grid {
    int pr = 1, pc = 1;
    int row(int rank) const {
        return rank / pc;
    }
    int col(int rank) const {
        return rank % pc;
    }
    int rank(int r, int c) const {
        return r * pc + c;
    }
    int size() const {
        return pr * pc;
    }
    // Whether a W may sum the partials of the grid rows of a grid column, a carrier (1, pc, pr) at
    // the node, within the 2.5D bound: the W of a TT merge or of a compact-WY panel over the rows.
    bool column_reduces() const {
        return pr * pr <= pc;
    }
};

// The grid of `gpus` GPUs with `rows` grid rows or, by default, the one with the most grid rows
// whose square does not exceed the grid columns, so that the W of a TT merge can sum the partials of
// the grid rows of a grid column (a carrier (1, pc, pr) at the node, replicated) within the 2.5D
// bound pr^2 <= pc: 2 x 4 on eight GPUs. Without such a grid the most square one with at least as
// many grid rows as columns, whose merges contract gathered rows instead (c = 1): 2 x 2 on four
// GPUs, P x 1 for a prime P.
inline Grid make_grid(int gpus, int rows) {
    if (rows <= 0) {
        rows = 0;
        for (int r = 2; r * r * r <= gpus; ++r)
            if (gpus % r == 0)
                rows = r;
        if (!rows) {
            rows = gpus;
            for (int r = 1; r * r <= gpus; ++r)
                if (gpus % r == 0)
                    rows = gpus / r;
        }
    }
    if (rows < 1 || gpus % rows)
        throw std::runtime_error("the grid rows must divide the GPUs");
    return Grid{rows, gpus / rows};
}

// TT elimination of the triangles of k GPUs of one grid column. Member 0 owns the diagonal block of
// the panel and receives the merged R.
struct Merge {
    int k = 0, col = 0, h = 0, tile = 0;
    int rank[max_gpus], row[max_gpus], height[max_gpus];
    int owner() const {
        return rank[0];
    }
    int member(int gpu) const {
        for (int i = 0; i < k; ++i)
            if (rank[i] == gpu)
                return i;
        return -1;
    }
};

struct Panel {
    int col, h;                 // global columns [col, col + h)
    int owner;                  // the grid column that holds them
    int local_col;              // their first local column on the GPUs of that grid column
    std::vector<TilePacket> ge; // one per GPU; rows == 0 where the GPU holds no row of the panel
    Merge merge;                // k == 0 when a single GPU holds the panel
};

struct Plan {
    Grid grid;
    Cyclic rows, cols;
    int n = 0, b = 0;
    std::vector<Panel> panels;
    std::vector<int> tiles; // T factors retained per GPU

    Plan(int m, int n_, Grid g, int b_)
        : grid(g), rows{m, g.pr, b_}, cols{n_, g.pc, b_}, n(n_), b(b_), tiles(g.size(), 0) {
        if (m < 1 || n < 1 || g.pr < 1 || g.pc < 1 || g.size() > max_gpus || b < 1)
            throw std::runtime_error("invalid matrix, grid or panel dimensions");
        for (int col = 0; col < std::min(m, n); col += b) {
            const int owner = cols.owner(col);
            Panel panel{col, std::min(b, std::min(m, n) - col), owner, cols.before(owner, col), {},
                        {}};
            Merge &merge = panel.merge;
            merge.col = panel.local_col;
            merge.h = panel.h;
            // The owner of the diagonal block comes first; the other grid rows follow in cyclic
            // order.
            for (int i = 0; i < g.pr; ++i) {
                const int r = (rows.owner(col) + i) % g.pr, gpu = g.rank(r, owner);
                const int first = rows.before(r, col), count = rows.local(r) - first;
                if (!count)
                    continue;
                merge.rank[merge.k] = gpu;
                merge.row[merge.k] = first;
                merge.height[merge.k] = std::min(panel.h, count);
                ++merge.k;
            }
            panel.ge.assign(g.size(), TilePacket{0, panel.local_col, 0, 0, 0});
            for (int i = 0; i < merge.k; ++i) {
                const int gpu = merge.rank[i];
                const int count = rows.local(g.row(gpu)) - merge.row[i];
                panel.ge[gpu] = {merge.row[i], panel.local_col, count, merge.height[i],
                                 tiles[gpu]++};
            }
            if (merge.k > 1)
                merge.tile = tiles[merge.owner()]++;
            else
                merge.k = 0;
            panels.push_back(std::move(panel));
        }
    }
    // The GE of panel `k` whose reflectors act on the rows of `gpu`: the one of its grid row.
    const TilePacket &row_ge(int k, int gpu) const {
        return panels[k].ge[grid.rank(grid.row(gpu), panels[k].owner)];
    }
    // The member of the TT merge of panel `k` on the grid row of `gpu`, or -1.
    int row_member(int k, int gpu) const {
        const Merge &e = panels[k].merge;
        for (int i = 0; i < e.k; ++i)
            if (grid.row(e.rank[i]) == grid.row(gpu))
                return i;
        return -1;
    }
};
} // namespace qr_omega
