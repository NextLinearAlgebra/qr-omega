#pragma once
// machine.cuh -- the TQR-Omega machine model (Definition 2.1 of the proof package).
//
// A machine is a nest of peer memory regions separated by boundaries. Region 0 is
// the fastest and smallest, region L the slowest and largest; boundary l separates
// region l from region l+1, so an (L+1)-region machine carries exactly L boundaries:
//
//     [region 0] --bnd 0-- [region 1] --bnd 1-- ... --bnd L-1-- [region L]
//      fastest                                                    slowest
//
// Definition 2.1 states the hierarchy as
//     H = < L, { (P_l, M_l, B_l, alpha_l, r_l) }_{l=0}^{L-1} >.
// This file splits that tuple the way the hardware does. M_l is the capacity of one
// level-l domain and belongs to the region; (P_l, B_l, alpha_l, r_l) describe the
// interface and belong to the boundary. P_l is stored once, on the boundary, because
// it is the fan-out of a level-(l+1) domain into its level-l children -- a property
// of the pair, not of either region alone.
//
// Every quantity is in WORDS (one matrix element), exactly as the proof states them.
// machine::word_bytes records how wide a word is, so byte-denominated spec-sheet
// numbers can be converted at the door; no bound below ever sees a byte.
//
// Nothing here names a vendor, a device or a product. Registers, scratchpads, caches,
// device memory and interconnects are *instantiations* of the same handful of numbers
// (Section 2): "a register file is a boundary with small M, large P, tiny alpha; an
// interconnect is a boundary with large M, moderate P, large alpha."

#include <cstddef>

#if defined(__CUDACC__)
#define TQR_HD __host__ __device__
#else
#define TQR_HD
#endif

namespace tqr {

// A machine is a hierarchy H in the sense of Definition 2.1: an ordered chain of peer
// memory regions, fastest first, with one boundary between each adjacent pair.
//
// Usage -- a five-region GPU-plus-host machine. The names are generic and the numbers
// below are illustrative placeholders; substitute measured values for a real target.
//
//     tqr::machine m("gpu + host", /*word_bytes=*/8);   // fp64
//     m.root({"registers",  64 * 1024 / 8, machine::kRegisterFile});
//     m.attach({/*P=*/32, /*B=*/..., /*alpha=*/..., /*r=*/2}, {"smem/L1", ...});
//     m.attach({/*P=*/132, ...},                              {"L2",      ...});
//     m.attach({/*P=*/1,   ...},                              {"hbm",     ...});
//     m.attach({/*P=*/1,   ...},                              {"host ram", ...});
//     const char* err = m.validate();   // nullptr when well formed
//
// The region count is bounded at compile time (kMaxRegions) so a machine is a plain
// value: no allocation, trivially copyable to the device, and constexpr-constructible.
struct machine {
    // Real hierarchies have L <= 5 or so. 16 regions is far past any of them, and the
    // proof requires only that L be a machine constant independent of m and n.
    static constexpr int kMaxRegions   = 16;
    static constexpr int kMaxBoundaries = kMaxRegions - 1;

    // ---------------------------------------------------------------------------
    // A peer memory region: one level of the hierarchy.
    // ---------------------------------------------------------------------------

    // Purely descriptive. Section 2 is explicit that the model refers to none of these
    // -- they are instantiations of (P, M, B, alpha, r), not inputs to any bound. The
    // tag exists so a hierarchy reads like the machine it describes; no bound reads it.
    enum kind : int {
        kUnspecified = 0,
        kRegisterFile,
        kScratchpad,     // software-managed, e.g. shared memory
        kCache,          // hardware-managed
        kDeviceMemory,   // e.g. HBM
        kHostMemory,     // e.g. CPU RAM
        kRemoteMemory,   // reached across an interconnect
    };

    struct peer_memory_region {
        // Free-form label, e.g. "registers", "smem/L1", "L2", "hbm", "host ram".
        // Must outlive the machine; a string literal is the intended case.
        const char* name = "";

        // M_l: the physical capacity, in words, of ONE level-l domain -- not of all
        // P_l of them together. Workspace is NOT subtracted (Corollary 12.2 charges it
        // inside the capacity constraints rather than against M_l).
        double capacity = 0.0;

        // Descriptive tag only; see `kind` above.
        kind tag = kUnspecified;

        TQR_HD constexpr bool valid() const { return capacity > 0.0; }
    };

    // Short alias for the common case.
    using region = peer_memory_region;

    // ---------------------------------------------------------------------------
    // A boundary: the interface between region l and region l+1.
    // ---------------------------------------------------------------------------
    // Which transport carries a crossing. This is not a preference ordering: the move
    // alphabet of Definition 4.3 splits by it. Combine(b) row-partial addition over a
    // declared disjoint row partition, Replicate broadcasts of Y_s/W_s, and the deferred
    // 2.5D layer reduction are commutative and associative, so a collective can carry
    // them. Combine(a), the ordered stack-Householder merge, is associative once signs
    // and leaf order are fixed but is NOT commutative (Counterexample 16.1), so no
    // collective reduction can express it -- it, and arbitrary adjacent transport, need
    // a one-sided point-to-point path. The two cost differently and are carried
    // separately, because the panel-merge depth D_l is paid at one-sided latency while
    // the W reduction and the broadcasts are paid at collective latency.
    enum transport : int {
        kToParent = 0,      // across the boundary, into the level-(l+1) domain
        kPeerCollective,    // peer to peer at level l, via a collective
        kPeerOneSided,      // peer to peer at level l, point to point
    };

    struct boundary {
        // P_l >= 1: the number of peer domains at level l that share the level-(l+1)
        // side. A local branching factor, NOT the total number of level-l domains in
        // the machine -- see machine::total_domains() for the latter and Remark 2.2
        // for why the two must never be confused.
        int peers = 1;

        // --- the path to the parent: the proof's (B_l, alpha_l, r_l) -----------------
        // B_l: sustainable bandwidth across the boundary, in words per unit time, per
        // domain.
        double bandwidth = 0.0;

        // alpha_l: exposed latency of one dependency-bearing round across the boundary,
        // in the same time unit as `bandwidth`.
        double latency = 0.0;

        // r_l >= 2: maximum fan-in of a reduction tree at the boundary. Only consulted
        // when peers > 1; a boundary with a single peer runs no tree.
        int fanin = 2;

        // --- the peer paths ---------------------------------------------------------
        // Definition 2.1 gives a boundary one (B, alpha) pair, and Lemma 4.11 charges
        // peer traffic at level l to boundary l -- "Replicate places a pebble in a peer
        // domain at the same colour, which by Lemma 4.9 is a single crossing of that
        // boundary". One pair is physically right wherever the peer path IS the parent
        // path: two warps exchange through shared memory, two SMs through L2. It is
        // wrong exactly where a dedicated peer fabric bypasses the parent, as NVLink
        // bypasses host memory. Left at zero, the parent numbers stand in, which is the
        // correct reading for a store-and-forward peer path.
        double peer_bandwidth = 0.0;      // collective path
        double peer_latency   = 0.0;
        int    peer_fanin     = 2;        // a hardware all-to-all can beat a binary tree

        double onesided_bandwidth = 0.0;  // point-to-point path
        double onesided_latency   = 0.0;

        TQR_HD constexpr bool valid() const {
            return peers >= 1 && bandwidth > 0.0 && latency >= 0.0
                && (peers == 1 || fanin >= 2);
        }

        TQR_HD constexpr bool has(transport t) const {
            return (t == kPeerCollective) ? peer_bandwidth > 0.0
                 : (t == kPeerOneSided)   ? onesided_bandwidth > 0.0
                                          : bandwidth > 0.0;
        }

        // The (B, alpha, r) triple actually carried by a given transport.
        TQR_HD constexpr double B(transport t = kToParent) const {
            if (t == kPeerCollective && peer_bandwidth > 0.0)     return peer_bandwidth;
            if (t == kPeerOneSided && onesided_bandwidth > 0.0)   return onesided_bandwidth;
            return bandwidth;
        }
        // Bandwidth and latency fall back independently: a peer path whose latency is
        // measured but whose bandwidth is not still contributes the latency, rather than
        // silently reverting both to the parent. The L2 crossbar is exactly that case --
        // the far-hit latency is measurable by pointer chase, the partition-to-partition
        // bandwidth is not.
        TQR_HD constexpr double alpha(transport t = kToParent) const {
            if (t == kPeerCollective && peer_latency > 0.0)     return peer_latency;
            if (t == kPeerOneSided && onesided_latency > 0.0)   return onesided_latency;
            return latency;
        }
        TQR_HD constexpr int fan_in(transport t = kToParent) const {
            return (t == kPeerCollective && peer_bandwidth > 0.0) ? peer_fanin : fanin;
        }

        // D_l = ceil(log_{r_l} P_l), the depth of a reduction tree at this boundary,
        // and 0 when P_l = 1. Computed by integer accumulation rather than through
        // logarithms, so D_l is exact -- log(8)/log(2) is not reliably 3.
        // Returns -1 if the boundary is malformed.
        TQR_HD constexpr int tree_depth(transport t = kToParent) const {
            const int r = fan_in(t);
            if (peers < 1) return -1;
            if (peers == 1) return 0;
            if (r < 2) return -1;
            int d = 0;
            long long reach = 1;
            while (reach < (long long)peers) { reach *= (long long)r; ++d; }
            return d;
        }

        // The per-boundary term of the time model (1): Q_l/B_l + Scrit_l * alpha_l,
        // where Q is words crossing this boundary at the maximally loaded domain and
        // Scrit is the number of those crossings lying on one longest directed chain.
        TQR_HD constexpr double time(double words, double critical_rounds,
                                     transport t = kToParent) const {
            return words / B(t) + critical_rounds * alpha(t);
        }
    };

    // ---------------------------------------------------------------------------
    // Regimes (Definition 2.4).
    // ---------------------------------------------------------------------------
    // A boundary is `resident` for a subproblem when the (replicated) active matrix
    // fits in the aggregate fast side, and `streaming` otherwise. The two regimes have
    // genuinely different optimal constants (Theorems 6.3 and 6.13), so the test is
    // part of the model rather than a heuristic.
    enum regime : int { kStreaming = 0, kResident = 1 };

    // ---------------------------------------------------------------------------
    // The machine itself.
    // ---------------------------------------------------------------------------
    const char* name = "";

    // Bytes per word, i.e. per matrix element: 8 for fp64, 4 for fp32. Used only to
    // convert spec-sheet figures; the model is stated entirely in words.
    int word_bytes = 8;

    // R, the sustained machine-wide arithmetic rate in operations per unit time. This
    // is an OUTPUT of the theory, not an input to it: Section 6.5 derives it from the
    // innermost boundary (Corollary 6.27, Proposition 6.28). Left at 0 until derived,
    // and no member of this file reads it.
    double sustained_rate = 0.0;

    region   regions[kMaxRegions]     = {};
    boundary boundaries[kMaxBoundaries] = {};
    int      n_regions = 0;

    constexpr machine() = default;   // TQR_HD omitted: nvcc ignores it on a defaulted ctor
    TQR_HD constexpr explicit machine(const char* nm, int wb = 8)
        : name(nm), word_bytes(wb) {}

    // ---- shape -----------------------------------------------------------------

    // L + 1, the number of levels.
    TQR_HD constexpr int region_count() const { return n_regions; }
    // L, the number of boundaries. One fewer than the region count.
    TQR_HD constexpr int boundary_count() const { return n_regions > 0 ? n_regions - 1 : 0; }
    // The proof's L.
    TQR_HD constexpr int depth() const { return boundary_count(); }

    TQR_HD constexpr const region&   region_at(int l)   const { return regions[l]; }
    TQR_HD constexpr const boundary& boundary_at(int l) const { return boundaries[l]; }
    TQR_HD constexpr region&         region_at(int l)         { return regions[l]; }
    TQR_HD constexpr boundary&       boundary_at(int l)       { return boundaries[l]; }

    // ---- construction ----------------------------------------------------------

    // Install region 0, the innermost and fastest. Resets any existing chain.
    TQR_HD constexpr bool root(const region& r) {
        n_regions = 0;
        regions[n_regions++] = r;
        return true;
    }

    // Extend the chain outward: reach a new, slower region across a new boundary.
    // The boundary becomes boundary (region_count() - 1) and separates the current
    // outermost region from `r`. Returns false if the machine is full or has no root.
    TQR_HD constexpr bool attach(const boundary& b, const region& r) {
        if (n_regions < 1 || n_regions >= kMaxRegions) return false;
        boundaries[n_regions - 1] = b;
        regions[n_regions++] = r;
        return true;
    }

    // Convert a byte-denominated capacity or bandwidth into words.
    TQR_HD constexpr double words(double bytes) const {
        return bytes / (double)word_bytes;
    }

    // ---- derived quantities ----------------------------------------------------

    // M_l: capacity in words of one level-l domain.
    TQR_HD constexpr double capacity(int l) const { return regions[l].capacity; }

    // P_l: peers sharing the level-(l+1) side of boundary l. The outermost level L has
    // no boundary above it and thus a single domain, so P_L = 1 by convention.
    TQR_HD constexpr int peers(int l) const {
        return l < boundary_count() ? boundaries[l].peers : 1;
    }

    // Pbar_l = prod_{l' = l}^{L-1} P_l', the TOTAL number of level-l domains in the
    // machine, with Pbar_L = 1 (Remark 2.2). This is the normalization that divides
    // total work or total traffic among all domains of a level -- the per-domain Q_l,
    // the rate R/Pbar_l, and the right-hand sides of Corollary 6.4, Theorem 6.13 and
    // Theorem 7.9. It is NOT the normalization for anything describing a collective at
    // a single boundary (the tree depth D_l, the p x p x c grid of Definition 10.1);
    // those are governed by P_l. The two coincide at the outermost boundary.
    TQR_HD constexpr long long total_domains(int l) const {
        long long prod = 1;
        for (int lp = l; lp < boundary_count(); ++lp) prod *= (long long)boundaries[lp].peers;
        return prod;
    }

    // Mtilde_l, the aggregate red capacity of Definition 4.8: contracting levels
    // 0..l into a single red side gives each level-l domain the capacity of everything
    // nested inside it,
    //     Mtilde_l = sum_{l'=0}^{l} ( prod_{t=l'}^{l-1} P_t ) M_l'.
    // Lower bounds obtained by projection (Lemma 4.9) are stated against this, not
    // against M_l; see xi() and Remark 4.10 for when the distinction matters.
    TQR_HD constexpr double aggregate_capacity(int l) const {
        double acc = regions[l].capacity;
        long long prod = 1;
        for (int lp = l - 1; lp >= 0; --lp) {
            prod *= (long long)boundaries[lp].peers;
            acc += (double)prod * regions[lp].capacity;
        }
        return acc;
    }

    // xi_l = Mtilde_l / M_l - 1, the fraction by which the memory nested below
    // boundary l inflates its capacity (Definition 4.8, Remark 4.10). Every lower bound
    // scales as M^{-1/2}, so writing M_l where Mtilde_l is meant overstates it by
    // sqrt(1 + xi_l). State bounds with M_l only where xi_l = o(1) and carry Mtilde_l
    // otherwise: at the outermost boundary xi is typically 1e-2 or less and the
    // distinction is immaterial, but at an inner boundary an aggregate register file
    // can be a constant fraction of the next level and Mtilde is the honest quantity.
    TQR_HD constexpr double xi(int l) const {
        const double m = regions[l].capacity;
        return m > 0.0 ? aggregate_capacity(l) / m - 1.0 : 0.0;
    }

    // D_l = ceil(log_{r_l} P_l), the collective depth at boundary l.
    TQR_HD constexpr int tree_depth(int l, transport t = kToParent) const {
        return boundaries[l].tree_depth(t);
    }

    // Lhat_l = 2 * alpha_l * B_l / M_l, the sole dimensionless parameter of a boundary
    // (equation (23)): the ratio of the latency-bandwidth product to the capacity. It
    // is what selects the optimal tile shape, moving it continuously from the
    // bandwidth-limited square at Lhat -> 0 to the 1:2:1 latency-limited split as
    // Lhat -> infinity. Vendor-agnostic by construction, and dimensionless whatever
    // unit B and M share.
    TQR_HD constexpr double l_hat(int l, transport t = kToParent) const {
        const double m = regions[l].capacity;
        return m > 0.0 ? 2.0 * boundaries[l].alpha(t) * boundaries[l].B(t) / m : 0.0;
    }

    // Definition 2.4. Boundary l is resident for an m' x n' subproblem at replication
    // factor c when P_l * M_l >= c * m' * n', i.e. the replicated active matrix fits in
    // the aggregate fast side; otherwise it is streaming. A hierarchy typically has
    // streaming boundaries at the fast, small-capacity end and resident boundaries at
    // the interconnect, where the aggregate capacity P_l * M_l is large.
    TQR_HD constexpr regime regime_at(int l, double m, double n, double c = 1.0) const {
        const double aggregate = (double)boundaries[l].peers * regions[l].capacity;
        return aggregate >= c * m * n ? kResident : kStreaming;
    }

    // ---- validation ------------------------------------------------------------

    // Returns nullptr when the machine is a well-formed hierarchy, or a description of
    // the first violation found. Cheap enough to call on every constructed machine.
    TQR_HD constexpr const char* validate() const {
        if (n_regions < 1) return "machine has no regions";
        if (n_regions > kMaxRegions) return "region count exceeds kMaxRegions";
        if (word_bytes < 1) return "word_bytes must be at least 1";
        for (int l = 0; l < n_regions; ++l) {
            if (!regions[l].valid()) return "a region has non-positive capacity";
        }
        for (int l = 0; l < boundary_count(); ++l) {
            const boundary& b = boundaries[l];
            if (b.peers < 1) return "a boundary has peers < 1";
            if (!(b.bandwidth > 0.0)) return "a boundary has non-positive bandwidth";
            if (b.latency < 0.0) return "a boundary has negative latency";
            if (b.peers > 1 && b.fanin < 2) return "a boundary with peers > 1 has fanin < 2";
        }
        // Levels run fastest to slowest, so capacity must not shrink going outward.
        // This is a modelling convention, not a theorem: it is what makes "region 0 is
        // innermost" meaningful and what the nesting argument of Lemma 11.1 assumes
        // when it takes M_{l-1}/M_l to be small.
        for (int l = 1; l < n_regions; ++l) {
            if (regions[l].capacity < regions[l - 1].capacity) {
                return "regions are not ordered fastest-to-slowest by capacity";
            }
        }
        return nullptr;
    }

    TQR_HD constexpr bool is_valid() const { return validate() == nullptr; }
};

}  // namespace tqr
