#pragma once
// Carriers of the products. A product C = A B is split at four levels of the machine: the GPUs of
// the node, the blocks (or clusters) of a GPU, the blocks of a cluster and the warps of a block.
// At each level its p_i p_j c peers split the rows I, the columns J and the contraction K of C; a
// 2.5D carrier keeps 1 <= c <= (p_i p_j c)^(1/3), that is c^2 <= p_i p_j, and replicates when
// c >= 2. Every launch records its carrier and flop count so that a run can report how much of
// its arithmetic is carried within the bound.
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace qr_omega {
enum Level { NodeLevel, GpuLevel, ClusterLevel, BlockLevel, Levels };
inline const char *level_name(int l) {
    static const char *names[Levels] = {"node", "gpu", "cluster", "block"};
    return names[l];
}

struct Cut {
    int pi = 1, pj = 1, c = 1;
    bool bounded() const {
        return c >= 1 && int64_t(c) * c <= int64_t(pi) * pj;
    }
    bool operator<(const Cut &o) const {
        return std::tie(pi, pj, c) < std::tie(o.pi, o.pj, o.c);
    }
};

struct Carrier {
    Cut at[Levels];
    Carrier &set(Level l, int pi, int pj, int c) {
        at[l] = Cut{pi, pj, c};
        return *this;
    }
    bool bounded() const {
        for (const Cut &cut : at)
            if (!cut.bounded())
                return false;
        return true;
    }
    // Whether some level replicates the product (c >= 2).
    bool replicated() const {
        for (const Cut &cut : at)
            if (cut.c > 1)
                return true;
        return false;
    }
    bool operator<(const Carrier &o) const {
        for (int l = 0; l < Levels; ++l) {
            if (at[l] < o.at[l])
                return true;
            if (o.at[l] < at[l])
                return false;
        }
        return false;
    }
};

// Kinds of products: the three of an update, the Gram products of composed T factors, and the
// column steps of a panel GE (or of the TT merge of a stack of triangles).
enum Product { ProductW, ProductZ, ProductD, ProductGram, ProductPanel, Products };
inline const char *product_name(int p) {
    static const char *names[Products] = {"W", "Z", "D", "Gram", "panel"};
    return names[p];
}

// Products issued by a factorization: every launch with its carrier and flops.
struct Tally {
    struct Entry {
        Product product;
        Carrier carrier;
        double flops;
    };
    std::vector<Entry> entries;
    void add(Product product, const Carrier &carrier, double flops) {
        entries.push_back({product, carrier, flops});
    }
    struct Summary {
        uint64_t products = 0, carried = 0, bounded = 0;
        double flops = 0, carried_flops = 0, bounded_flops = 0;
    };
    Summary summary() const {
        Summary s;
        for (const Entry &e : entries) {
            const bool carried = e.carrier.replicated(), bounded = e.carrier.bounded();
            ++s.products;
            s.carried += carried;
            s.bounded += bounded;
            s.flops += e.flops;
            s.carried_flops += carried ? e.flops : 0;
            s.bounded_flops += bounded ? e.flops : 0;
        }
        return s;
    }
    // The distinct carriers of each kind of product, with their counts and flops.
    struct Use {
        uint64_t count = 0;
        double flops = 0;
    };
    std::map<std::pair<int, Carrier>, Use> uses() const {
        std::map<std::pair<int, Carrier>, Use> u;
        for (const Entry &e : entries) {
            Use &x = u[{int(e.product), e.carrier}];
            ++x.count;
            x.flops += e.flops;
        }
        return u;
    }
};
// The largest number of groups, at most `wanted`, that splits the contraction of a product with
// `tiles` output tiles within the 2.5D bound groups^2 <= tiles.
inline int bounded_groups(long long tiles, int wanted) {
    int g = wanted > 1 ? wanted : 1;
    while (g > 1 && (long long)g * g > tiles)
        --g;
    return g;
}
inline std::string describe(const Carrier &c) {
    std::string s;
    for (int l = 0; l < Levels; ++l) {
        const Cut &k = c.at[l];
        s += std::string(l ? " " : "") + level_name(l) + "(" + std::to_string(k.pi) + "," +
             std::to_string(k.pj) + "," + std::to_string(k.c) + ")";
    }
    return s;
}
} // namespace qr_omega
