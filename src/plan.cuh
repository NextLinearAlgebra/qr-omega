#pragma once
// plan.cuh -- Algorithm 4, Plan(m, n, lambda): derive the optimal geometry of a boundary.
//
// Pure arithmetic over a tqr::machine. No measurement, no search: every width is either
// a closed form or the root of a scalar equation that is proved to have exactly one.
// Theorem 9.4 is what makes that possible -- the objective has a strictly monotone
// logarithmic derivative, so the optimizer is unique and bisection cannot land on the
// wrong stationary point.
//
// The listing, verbatim from the proof package:
//
//   Require: (P_l, M_l, B_l, alpha_l, r_l) only
//    1: Lhat <- 2 alpha_l B_l / M_l                       . dimensionless; the only knob
//    2: if mn <= M_l then                                 . resident regime
//    3:    c <- min{(2/3)^{2/3} P_l^{1/3}, P_l M_l/(mn)};  p <- sqrt(P_l/c)
//    4:    nu <- n/sqrt(c P_l);  gamma <- mu <- sqrt(M_l)  . Theorem 10.3
//    5: else                                              . streaming regime
//    6:    z <- RootOf(phi(.) = 0) on (0,1) by bisection   . phi from (24)
//    7:    psi <- Lhat/z;  u <- 1 - z;  t <- (1+4psi)/(1+2psi)
//    8:    nu <- sqrt(u/(2t)) sqrt(M_l);  gamma <- t nu;  mu <- z M_l/(2(gamma + 2nu))
//    9:    c <- 1;  p <- sqrt(P_l)
//   10: end if
//   11: nu <- nu_{l-1} ceil(nu/nu_{l-1})                   . nesting; Lemma 11.1
//   12: if l = 1 then                                      . fix the innermost geometry once
//   13:    u <- root in (0,1) of (beta-1)u^2 + (beta+2)u = 1          . (17)
//   14:    mu_0 <- gamma_0 <- sqrt(u M_0);  nu_0 <- mu_0 (1-u)/(2 beta u)   . (18)
//   15:    rho_0 <- sqrt(u)(1-u) sqrt(M_0) / (2((beta-1)u + 1))
//   16:    b <- max{2 rho_0, nu_0}                                    . leaf width, (20)
//   17:    R_dom <- min{R_peak,dom, 2 rho_0 B_0}                       . roofline, (19)
//   18: end if
//   19: return (nu, gamma, mu, c, p, b)
//
// Three places where this file does something the listing does not say outright. Each is
// forced by another statement in the same document, and each is flagged at the point of
// use so the deviation is visible rather than folded in:
//
//   (a) Line 2's test reads `mn <= M_l`, but Definition 2.4 defines resident as
//       `P_l M_l >= c m'n'`, Theorem 10.3 opens with `mn <= P_l M_l`, and line 3's own
//       `c <- min{..., P_l M_l/(mn)}` is only meaningful when `mn <= P_l M_l`. We use the
//       aggregate. plan_options::literal_regime_test restores the listing's form; at a
//       boundary with P_l = 8 the two disagree by a factor of 8 on where the regime
//       switches.
//   (b) Line 11 nests nu upward onto a multiple of nu_{l-1}, which breaks the capacity
//       identity (22) unless something gives. Lemma 11.1 says gamma is "obtained by
//       rescaling so that (22) still holds ... the rounded point stays feasible because
//       gamma is rescaled downward". We therefore re-derive gamma and mu from the nested
//       nu, holding both halves of (22) exact.
//   (c) Line 11 needs nu_{l-1} while line 14 defines nu_0, so at l = 1 the listing reads
//       out of order. plan_hierarchy resolves it the only self-consistent way: fix the
//       innermost geometry first, then nest outward through l = 1, 2, ..., L-1.

#include "machine.cuh"

#include <cmath>

namespace tqr {

struct plan_options {
    // beta in (17): buffering depth at the innermost boundary. 1 gives a cube of side
    // sqrt(M_0/3); 2 gives the golden-ratio tile of Corollary 6.21, at a cost of about
    // 1.28x in intensity. Theorem 11.7 requires two live generations, so 2 is the
    // default the schedule actually runs.
    int buffering = 2;

    // theta_R in (29), the constant of the layer reduction, in [1,2]. 3/2 is the
    // deferred reduce-scatter / all-gather pair of Lemma 10.2 and reproduces the
    // (2/3)^{2/3} P^{1/3} of line 3 exactly.
    double theta_R = 1.5;

    // R_peak,dom for the roofline (19). Zero means unknown, and R_dom then reports the
    // bandwidth half 2 rho_0 B_0 alone.
    double peak_rate_per_domain = 0.0;

    // Use line 2 as written (mn <= M_l) rather than Definition 2.4 (mn <= P_l M_l).
    bool literal_regime_test = false;

    // Theorem 10.3 (28) needs c | P_l with sqrt(P_l/c) integral. When the c of (29)
    // is inadmissible the theorem says to "take instead the largest admissible c <= cL,
    // which costs a factor at most sqrt(cL/c) <= sqrt(2)". Off returns the continuous c.
    bool snap_replication = true;

    // Which transport carries which move. This is not a tuning knob: it follows the move
    // alphabet of Definition 4.3 and the traffic accounting of Lemma 9.1.
    //
    //   Combine (all four variants) -> ONE-SIDED. Combine(a), the ordered stack-
    //     Householder merge, is associative only once signs and leaf order are fixed and
    //     is NOT commutative (Counterexample 16.1), so no collective reduction can carry
    //     it; the others inherit the same carrier so that ordering and version stamps are
    //     handled uniformly. This is the transport the panel-tree depth is paid at.
    //   Everything else -> COLLECTIVE. Lemma 9.1's message count is tile visits times
    //     "c_S = 4 contiguous transfers (load A, store A, load Y_s, load Y_{s+1})", plus
    //     the Replicate broadcasts of Y_s and W_s. That is bulk movement, and it is what
    //     S_lambda in (21) counts -- so it is the collective's (B, alpha) that enters
    //     Lhat and therefore the tile shape.
    //
    // Set either to kToParent to model a boundary with no dedicated peer fabric.
    machine::transport combine_via = machine::kPeerOneSided;
    machine::transport move_via    = machine::kPeerCollective;

    // Bisection on (0,1) for (24). phi is strictly decreasing with phi(0+) = +inf and
    // phi(1-) = -inf, so the bracket is the whole interval and the root is unique.
    int    bisect_iters = 200;
    double bisect_tol   = 1e-15;
};

struct plan {
    // The geometry, in words. Line 19's return tuple.
    double nu    = 0.0;   // superpanel width
    double gamma = 0.0;   // column-block width
    double mu    = 0.0;   // row-strip height
    double c     = 1.0;   // replication factor
    double p     = 1.0;   // grid side, p x p x c
    double b     = 0.0;   // leaf width, set only where the innermost geometry is fixed

    // Intermediates worth keeping: these are what make a plan auditable.
    double L_hat = 0.0;   // 2 alpha B / M, the sole dimensionless knob (23)
    double z     = 0.0;   // fraction of capacity given to in-flight data
    double u     = 0.0;   // 1 - z, the fraction holding resident W blocks
    double t     = 0.0;   // gamma/nu at the optimum
    double psi   = 0.0;   // Lhat/z
    double F     = 0.0;   // the optimal objective value (26)
    double c_continuous = 0.0;  // c of (29) before snapping to an admissible grid
    double K     = 0.0;   // epochs, ceil(n/nu)
    int    D     = 0;     // collective depth at this boundary
    double S_crit = 0.0;  // K + D - 1, Theorem 11.4

    // Innermost quantities, set when the level-0 geometry is fixed.
    double rho0  = 0.0;   // intensity at the innermost boundary
    double R_dom = 0.0;   // derived sustained rate per domain (19)

    // The cost law of Lemma 9.1, per domain, for a given problem. F = 2mn^2 - (2/3)n^3.
    double flops = 0.0;   // F
    double Q     = 0.0;   // words crossing this boundary, per domain
    double S     = 0.0;   // transfers
    // Resident split (Theorem 10.3); zero at a streaming boundary, where Q is (21).
    double Q_broadcast = 0.0;   // mn / sqrt(c P)
    double Q_reduce    = 0.0;   // theta_R mn c / P, the deferred layer reduction

    // Latency split by transport. Theorem 11.4 gives Scrit = K + D - 1, never K*D: the
    // epoch-to-epoch link runs through the ordered merge, and the remaining D - 1 stages
    // are the pipeline fill of one epoch's broadcasts. Charging them to one alpha would
    // hide a factor that is measurable -- on this machine the two differ by 29%.
    double alpha_combine = 0.0;   // one-sided, pays the merge chain
    double alpha_move    = 0.0;   // collective, pays the broadcast fill and the transfers
    double B_move        = 0.0;   // collective bandwidth, the B in Q/B
    int    D_combine     = 0;     // ceil(log_r P) over the one-sided fan-in
    int    D_move        = 0;     // ceil(log_r P) over the collective fan-in
    double t_bandwidth   = 0.0;   // Q / B_move
    double t_lat_combine = 0.0;   // K * alpha_combine
    double t_lat_move    = 0.0;   // (D_move - 1) * alpha_move
    double t_total       = 0.0;   // their sum: this boundary's term of the time model (1)

    machine::regime  regime = machine::kStreaming;
    machine::transport via  = machine::kToParent;
    int  level = -1;
    bool nested = false;          // line 11 actually moved nu
    bool innermost_fixed = false;
    bool ok = false;
    const char* note = "";
};

namespace detail {

// (24): phi(z) = 2/z - 1/(1-z) - 1/(z+2Lhat) - 1/(z+4Lhat).
// Strictly decreasing on (0,1): d/dz[1/z - 1/(z+a)] = -1/z^2 + 1/(z+a)^2 < 0 for a > 0,
// and -1/(1-z) is strictly decreasing. phi(0+) = +inf, phi(1-) = -inf.
TQR_HD inline double phi(double z, double Lhat) {
    return 2.0 / z - 1.0 / (1.0 - z) - 1.0 / (z + 2.0 * Lhat) - 1.0 / (z + 4.0 * Lhat);
}

// The unique root of (24) in (0,1), by bisection. Monotonicity is what licenses this:
// there is exactly one sign change, so no bracketing heuristic is needed.
TQR_HD inline double root_z(double Lhat, int iters, double tol) {
    double lo = 1e-300, hi = 1.0 - 1e-16;
    for (int i = 0; i < iters; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (hi - lo < tol * (mid > 0.0 ? mid : 1.0)) break;
        if (phi(mid, Lhat) > 0.0) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// (17): the root in (0,1) of (beta-1)u^2 + (beta+2)u = 1. The left-hand side is strictly
// increasing on (0,1) from 0 to beta+1 > 1, so the root exists and is unique.
// beta = 1 degenerates to 3u = 1; otherwise take the positive branch of the quadratic.
TQR_HD inline double innermost_u(double beta) {
    if (beta == 1.0) return 1.0 / 3.0;
    const double a = beta - 1.0, b = beta + 2.0;
    return (-b + std::sqrt(b * b + 4.0 * a)) / (2.0 * a);
}

// Largest admissible replication factor <= c: Theorem 10.3 (28) needs c | P and
// sqrt(P/c) integral. Returns 1 if nothing better is admissible.
TQR_HD inline double snap_c(long long P, double c) {
    long long best = 1;
    for (long long k = 1; k <= P; ++k) {
        if (P % k) continue;                       // c | P
        const long long q = P / k;                 // need sqrt(q) integral
        long long r = (long long)(std::sqrt((double)q) + 0.5);
        if (r * r != q) continue;
        if ((double)k <= c && k > best) best = k;
    }
    return (double)best;
}

}  // namespace detail

// ---------------------------------------------------------------------------------
// Plan(m, n, lambda) -- Algorithm 4 for one boundary.
// ---------------------------------------------------------------------------------
// `inner_nu` is nu_{lambda-1} for the nesting of line 11; pass 0 to skip it (there is
// no inner width below the innermost boundary).
TQR_HD inline plan Plan(const machine& mach, int lambda, double m, double n,
                 double inner_nu = 0.0, const plan_options& opt = plan_options()) {
    plan pl;
    pl.level = lambda;
    if (lambda < 0 || lambda >= mach.boundary_count()) { pl.note = "boundary out of range"; return pl; }
    if (!(m > 0.0) || !(n > 0.0))                      { pl.note = "m and n must be positive"; return pl; }

    const machine::boundary& bnd = mach.boundary_at(lambda);
    const double M = mach.capacity(lambda);
    const long long P = bnd.peers;
    if (!(M > 0.0) || P < 1) { pl.note = "boundary has no capacity or no peers"; return pl; }

    // Line 2 / Definition 2.4. Decided before line 1 because it selects the transport,
    // and therefore which (B, alpha) pair line 1 is even asking about: a resident
    // boundary carries all-peer-to-peer traffic (Theorem 10.3), a streaming one goes out
    // to the parent.
    const double aggregate = opt.literal_regime_test ? M : (double)P * M;
    const bool resident = (m * n <= aggregate);
    pl.regime = resident ? machine::kResident : machine::kStreaming;

    // Which transports actually carry this boundary. A resident boundary is
    // all-peer-to-peer (Theorem 10.3), so both the bulk movement and the merge use the
    // peer fabric; a streaming one goes out to the parent for everything, since there is
    // no peer copy of the data to reach.
    const machine::transport tmove =
        resident ? opt.move_via    : machine::kToParent;
    const machine::transport tcomb =
        resident ? opt.combine_via : machine::kToParent;
    pl.via = tmove;

    // Line 1. Lhat is built from the MOVE transport, not the merge one: S_lambda in (21)
    // counts tile-visit transfers (load A, store A, load Y_s, load Y_{s+1}), so it is the
    // bulk path whose latency trades against the widths in the objective of Theorem 9.4.
    pl.alpha_move    = bnd.alpha(tmove);
    pl.alpha_combine = bnd.alpha(tcomb);
    pl.B_move        = bnd.B(tmove);
    pl.D_move        = bnd.tree_depth(tmove);
    pl.D_combine     = bnd.tree_depth(tcomb);
    pl.L_hat = 2.0 * pl.alpha_move * pl.B_move / M;

    if (resident) {
        // Lines 3-4, Theorem 10.3 (29). The replication factor is derived, not tuned:
        // the first term is where deferred layer reduction stops being lower-order, the
        // second is the capacity ceiling (27).
        const double by_reduction = std::cbrt((double)P / (opt.theta_R * opt.theta_R));
        const double by_capacity  = (double)P * M / (m * n);
        pl.c_continuous = by_reduction < by_capacity ? by_reduction : by_capacity;
        pl.c = opt.snap_replication ? detail::snap_c(P, pl.c_continuous) : pl.c_continuous;
        if (!(pl.c >= 1.0)) pl.c = 1.0;
        pl.p     = std::sqrt((double)P / pl.c);
        pl.nu    = n / std::sqrt(pl.c * (double)P);
        pl.gamma = std::sqrt(M);
        pl.mu    = std::sqrt(M);
        // z, u, t do not enter the resident geometry; Counterexample 16.3 is the reason
        // the regime test exists at all -- with the matrix resident the far update moves
        // no words and the optimizer of Theorem 9.4 is vacuous.
    } else {
        // Lines 6-9, Theorem 9.4 (24) and (25).
        pl.z   = detail::root_z(pl.L_hat, opt.bisect_iters, opt.bisect_tol);
        pl.psi = pl.L_hat / pl.z;
        pl.u   = 1.0 - pl.z;
        pl.t   = (1.0 + 4.0 * pl.psi) / (1.0 + 2.0 * pl.psi);
        pl.nu    = std::sqrt(pl.u / (2.0 * pl.t)) * std::sqrt(M);
        pl.gamma = pl.t * pl.nu;
        pl.mu    = pl.z * M / (2.0 * (pl.gamma + 2.0 * pl.nu));
        pl.c     = 1.0;
        pl.p     = std::sqrt((double)P);
        // (26).
        pl.F = std::sqrt(2.0 * (pl.z + 2.0 * pl.L_hat) * (pl.z + 4.0 * pl.L_hat)
                         / (pl.z * pl.z * (1.0 - pl.z)));
    }

    // Line 11, Lemma 11.1. The penalty is second order because the minimizer is
    // interior, so the gradient vanishes there: F(nu_hat) = F* (1 + O((nu_{l-1}/nu*)^2)).
    if (inner_nu > 0.0 && pl.nu > 0.0) {
        const double nested = inner_nu * std::ceil(pl.nu / inner_nu);
        if (nested != pl.nu) {
            pl.nested = true;
            pl.nu = nested;
            // (b) above: hold (22) exact. 2 nu gamma = u M is the resident half and
            // 2 mu (gamma + 2 nu) = z M the in-flight half, so gamma follows from the
            // nested nu and mu from gamma. gamma moves downward, which keeps the point
            // feasible, exactly as Lemma 11.1 requires.
            if (!resident && pl.u > 0.0) {
                pl.gamma = pl.u * M / (2.0 * pl.nu);
                pl.mu    = pl.z * M / (2.0 * (pl.gamma + 2.0 * pl.nu));
                pl.t     = pl.gamma / pl.nu;
            }
        }
    }

    // Epochs and the critical-path round count. Theorem 11.4: the domino DAG's longest
    // path is K + D - 1 exactly, never K*D -- provided the near column block is
    // scheduled first, which is a requirement on Algorithm 1, not an optimization.
    pl.K      = std::ceil(n / pl.nu);
    pl.D      = pl.D_move;
    pl.S_crit = pl.K + (double)pl.D - 1.0;

    // The cost law. F = 2mn^2 - (2/3)n^3 is the flop count of Lemma 3.2.
    //
    // Which law applies depends on the regime, and mixing them is exactly the error
    // Counterexample 16.3 warns about. Lemma 9.1 opens "let a STREAMING boundary lambda
    // execute Algorithm 1", and its S counts tile visits; at a resident boundary the
    // matrix never leaves the level, there is no streaming of the far region, and
    // applying (21) there yields fewer than one transfer per domain -- a number with no
    // meaning. Theorem 10.3 gives the resident traffic instead.
    pl.flops = 2.0 * m * n * n - (2.0 / 3.0) * n * n * n;
    if (pl.flops < 0.0) pl.flops = 0.0;

    if (resident) {
        // Theorem 10.3: a broadcast term and a deferred layer-reduction term, the second
        // capped at the first by the choice of c in (29).
        const double cP = pl.c * (double)P;
        if (cP > 0.0) {
            pl.Q_broadcast = m * n / std::sqrt(cP);
            pl.Q_reduce    = opt.theta_R * m * n * pl.c / (double)P;
            pl.Q           = pl.Q_broadcast + pl.Q_reduce;
        }
        // S = Theta(sqrt(c P) log P); the constant is not claimed, so this is indicative.
        pl.S = (P > 1) ? std::sqrt(cP) * std::log((double)P) / std::log(2.0) : 0.0;
    } else {
        // Lemma 9.1 (21), ignoring the O(mn + n^2) panel, near-block and W-re-read terms
        // that Lemma 9.3 bounds separately.
        if (pl.nu > 0.0 && pl.gamma > 0.0)
            pl.Q = pl.flops / (2.0 * (double)P) * (1.0 / pl.nu + 1.0 / pl.gamma);
        if (pl.nu > 0.0 && pl.gamma > 0.0 && pl.mu > 0.0)
            pl.S = pl.flops / ((double)P * pl.nu * pl.mu * pl.gamma);
    }

    // The boundary's term of the time model (1), split by who carries what. The K epochs
    // are linked by the ordered merge, so they are paid at the one-sided latency; the
    // remaining D - 1 stages are the pipeline fill of the broadcasts and are paid at the
    // collective's. Summing them at a single alpha would be wrong by the ratio between
    // the two transports.
    if (pl.B_move > 0.0) pl.t_bandwidth = pl.Q / pl.B_move;
    pl.t_lat_combine = pl.K * pl.alpha_combine;
    pl.t_lat_move    = (pl.D_move > 1 ? (double)(pl.D_move - 1) : 0.0) * pl.alpha_move;
    pl.t_total       = pl.t_bandwidth + pl.t_lat_combine + pl.t_lat_move;

    pl.ok = true;
    return pl;
}

// ---------------------------------------------------------------------------------
// The innermost geometry (lines 12-18), fixed once for level 0.
// ---------------------------------------------------------------------------------
// Separated out because it is not a per-boundary quantity: it fixes the leaf width b and
// the sustained rate R, both of which Section 6.5 makes outputs of the theory rather than
// inputs to it. Needs M_0 and B_0, so it reads region 0 and boundary 0.
TQR_HD inline plan PlanInnermost(const machine& mach, const plan_options& opt = plan_options()) {
    plan pl;
    pl.level = 0;
    pl.innermost_fixed = true;
    if (mach.region_count() < 1) { pl.note = "machine has no regions"; return pl; }
    const double M0 = mach.capacity(0);
    if (!(M0 > 0.0))             { pl.note = "region 0 has no capacity"; return pl; }
    const double beta = (double)opt.buffering;
    if (!(beta >= 1.0))          { pl.note = "buffering depth must be at least 1"; return pl; }

    // Line 13, (17).
    const double u = detail::innermost_u(beta);
    pl.u = u;
    pl.z = 1.0 - u;

    // Line 14, (18). All three operand families cross at lambda = 0 and none is
    // amortized, which is why this tile is isotropic in (nu, gamma) where the outer
    // ones are not -- and why the sqrt(M/8) ceiling of Theorem 6.13 does not apply here
    // (Remark 6.24: using it at lambda = 0 would be an error of regime).
    pl.mu    = std::sqrt(u * M0);
    pl.gamma = pl.mu;
    pl.nu    = pl.mu * (1.0 - u) / (2.0 * beta * u);
    pl.t     = pl.gamma / pl.nu;

    // Line 15.
    pl.rho0 = std::sqrt(u) * (1.0 - u) * std::sqrt(M0)
            / (2.0 * ((beta - 1.0) * u + 1.0));

    // Line 16, (20).
    pl.b = (2.0 * pl.rho0 > pl.nu) ? 2.0 * pl.rho0 : pl.nu;

    // Line 17, (19). The roofline: an intensity of rho0 words per flop against B_0 caps
    // the rate at 2 rho0 B_0 however fast the arithmetic units are.
    if (mach.boundary_count() > 0) {
        const double B0 = mach.boundary_at(0).B();
        const double roof = 2.0 * pl.rho0 * B0;
        pl.R_dom = (opt.peak_rate_per_domain > 0.0 && opt.peak_rate_per_domain < roof)
                 ? opt.peak_rate_per_domain : roof;
        pl.L_hat = mach.l_hat(0);
    }
    pl.regime = machine::kStreaming;
    pl.ok = true;
    return pl;
}

// ---------------------------------------------------------------------------------
// The whole hierarchy: innermost first, then nested outward.
// ---------------------------------------------------------------------------------
// Writes plans for levels 0 .. L-1 into `out` (which must hold machine::kMaxBoundaries
// entries) and returns how many were written. Level 0's entry is the innermost geometry
// of lines 12-18; levels 1..L-1 are Plan() with line 11 threaded from the level below,
// which is deviation (c) above: the listing needs nu_{l-1} at line 11 but only defines
// nu_0 at line 14, so the innermost geometry has to be fixed before the outward sweep
// rather than during it.
TQR_HD inline int plan_hierarchy(const machine& mach, double m, double n, plan* out,
                          const plan_options& opt = plan_options()) {
    if (!out || mach.boundary_count() < 1) return 0;
    int k = 0;
    out[k] = PlanInnermost(mach, opt);
    double inner_nu = out[k].ok ? out[k].nu : 0.0;
    ++k;
    for (int l = 1; l < mach.boundary_count(); ++l) {
        out[k] = Plan(mach, l, m, n, inner_nu, opt);
        if (out[k].ok && out[k].nu > 0.0) inner_nu = out[k].nu;
        ++k;
    }
    return k;
}

}  // namespace tqr
