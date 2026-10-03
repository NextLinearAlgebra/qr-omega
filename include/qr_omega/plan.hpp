#pragma once
// The elimination list (Algorithm 1): every GPU reduces its rows of a panel with one GE and, across
// GPUs, one TT elimination of arity P merges the resulting triangles.
#include "kernels.cuh"

namespace tqr {
inline constexpr int max_gpus = 8;

// TT elimination of the triangles of k GPUs. Member 0 owns the diagonal block of the panel and
// receives the merged R.
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
    int col, h;
    std::vector<TilePacket> ge; // one per GPU; rows == 0 where the GPU holds no row of the panel
    Merge merge;                // k == 0 when a single GPU holds the panel
};

struct Plan {
    RowMap rows;
    int n = 0, b = 0;
    std::vector<Panel> panels;
    std::vector<int> tiles; // T factors retained per GPU

    Plan(int m, int n_, int gpus, int b_) : rows{m, gpus, b_}, n(n_), b(b_), tiles(gpus, 0) {
        if (m < 1 || n < 1 || gpus < 1 || gpus > max_gpus || b < 1)
            throw std::runtime_error("invalid matrix or panel dimensions");
        for (int col = 0; col < std::min(m, n); col += b) {
            Panel panel{col, std::min(b, std::min(m, n) - col), {}, {}};
            Merge &merge = panel.merge;
            merge.col = col;
            merge.h = panel.h;
            // The owner of the diagonal block comes first; the others follow in cyclic order.
            for (int i = 0; i < gpus; ++i) {
                const int gpu = (rows.owner(col) + i) % gpus;
                const int first = rows.rows_before(gpu, col), count = rows.local_rows(gpu) - first;
                if (!count)
                    continue;
                merge.rank[merge.k] = gpu;
                merge.row[merge.k] = first;
                merge.height[merge.k] = std::min(panel.h, count);
                ++merge.k;
            }
            panel.ge.assign(gpus, TilePacket{0, col, 0, 0, 0});
            for (int gpu = 0; gpu < gpus; ++gpu) {
                const int i = merge.member(gpu);
                if (i < 0)
                    continue;
                const int count = rows.local_rows(gpu) - merge.row[i];
                panel.ge[gpu] = {merge.row[i], col, count, merge.height[i], tiles[gpu]++};
            }
            if (merge.k > 1)
                merge.tile = tiles[merge.owner()]++;
            else
                merge.k = 0;
            panels.push_back(std::move(panel));
        }
    }
};
} // namespace tqr
