#pragma once
// Replication cost model for panel lanes.
//
// T_panel(c) = A * rowStates / c + B * ceil(log2(c)) + C * (c - 1).
// It balances per-lane chain work, ordered merge latency, and replica setup.
// The model remains disabled until a representative calibration is available.
#include <cmath>

namespace lrqr {

struct ReplCost {
    double A;   // ms per per-lane chain state
    double B;   // ms per ordered merge-tree level
    double C;   // ms per extra replica (setup)
};

// These provisional coefficients are only meaningful for internal comparisons.
constexpr ReplCost kReplCost = {1.41e-3, 1.0, 0.0};
// Disabled: a row-state-only fit did not generalize across aspect ratios.
constexpr bool kReplCostCalibrated = false;

// Predicted time for a panel with the current number of row states.
inline double replPanelTime(int rowStates, int c, const ReplCost& k) {
    const double cc = (double)(c < 1 ? 1 : c);
    return k.A * ((double)rowStates / cc)
         + k.B * std::ceil(std::log2(cc))
         + k.C * (cc - 1.0);
}

// Select the best legal power-of-two replication factor.
inline int replicaArgmin(int rowStates, int cap, const ReplCost& k) {
    int best = 1;
    double bestT = replPanelTime(rowStates, 1, k);
    for (int c = 2; c <= cap; c <<= 1) {
        const double t = replPanelTime(rowStates, c, k);
        if (t < bestT) { bestT = t; best = c; }
    }
    return best;
}

// Continuous optimum, used only for telemetry.
inline double replicaCStar(int rowStates, const ReplCost& k) {
    if (!(k.A > 0.0) || !(k.B > 0.0)) return 1.0;
    return k.A * (double)rowStates * 0.6931471805599453 / k.B;
}

// Joint replication and cluster-row choice. Candidates must fit c * (p_r + updaters)
// within the processor budget; exhaustive enumeration keeps the decision uniform.
struct GridChoice { int c; int p_r; double t; bool feasible; };

inline GridChoice replicaGridArgmin(int rowStates, int cCap, int prMax,
                                    int updaters, int P, const ReplCost& k) {
    GridChoice best{1, 1, 0.0, false};
    for (int c = 1; c <= cCap; c <<= 1) {
        for (int pr = 1; pr <= prMax; ++pr) {
            if ((long long)c * (pr + updaters) > P) continue;   // (A5)
            // Both factors reduce chain work; only replication incurs setup cost.
            const double chain = k.A * ((double)rowStates / (double)(c * pr));
            const double merge = k.B * std::ceil(std::log2((double)(c < 1 ? 1 : c)));
            const double clust = k.B * std::ceil(std::log2((double)(pr < 1 ? 1 : pr)));
            const double setup = k.C * (double)(c - 1);
            const double t = chain + merge + clust + setup;
            if (!best.feasible || t < best.t) best = GridChoice{c, pr, t, true};
        }
    }
    return best;
}

// Accuracy cap for ordered cross-lane merges. The model assumes independent
// rounding errors and limits replication to the available residual headroom.
inline int replicaAccuracyCap(double resid_c1, double thresh, int hardCap) {
    if (!(resid_c1 > 0.0) || !(thresh > resid_c1)) return 1;   // no headroom at all
    const double r = thresh / resid_c1;
    const double logc = r * r - 1.0;                            // log2 of the cap
    if (logc <= 0.0) return 1;
    int c = 1;
    while (c * 2 <= hardCap && std::log2((double)(c * 2)) <= logc) c *= 2;
    return c;
}

// Historical two-cap candidate retained for diagnostics; it is not enabled.
struct TwoCapLaw { int rowStatesDiv; int nMul; };
constexpr TwoCapLaw kTwoCap = {1024, 4096};
// OFF. REMOVED BY OWNER DIRECTIVE, and the reason is sound.
//
// Cap 2 (4096/n) is a REMEDY FOR AN IMPLEMENTATION DEFECT, not a cost term. I
// justified it as the Stage-2 apply bound -- mergeApplyQt runs on c*NB rows by the
// trailing width, so cost ~ c*n and c <= const/n. That does not survive arithmetic:
// Stage 2 touches c*NB rows against Stage 1's m rows, which at m=262144, c=2 is
// 0.0488% of the trailing work and 0.07% of total flops -- and it is the SAME
// 0.0488% at n=1024, 4096 and 8192. A term that is CONSTANT in n cannot explain a
// c limit that scales as 1/n. The number was fitted from the argmin table and the
// mechanism was reverse-engineered afterwards to justify it.
//
// What the 1/n behaviour actually reflects is the defects Steps 3 and 3.5 exist to
// fix: lanes that serialize (at m=262144, rpb clamps to clRpbMax so nbl=78 per lane
// and 2*78 > 132 SMs, giving lanesPerWave=1), the K*D merge/apply structure, and
// B/NB = 16 passes over each super-panel. Shipping a law that compensates for those
// would bake them in and make them harder to see.
//
// COST OF REMOVING IT, recorded so the trade is explicit: the exponent rule gives
// 65536x8192 1.43x (vs 1.51x), 131072x4096 1.37x (1.46x), 262144x8192 1.09x (1.16x),
// extreme fp64 1.15x (1.22x). Real performance is being given up now to keep the
// defect visible and fix it properly.
//
// FALSIFIABLE PREDICTION: once Steps 3 and 3.5 land, re-running the 48-point grid
// should show c* no longer falling with n, and cap 2 should relax or vanish. If it
// does not, cap 2 was real and this note is wrong.
constexpr bool kTwoCapCalibrated = false;

inline int replicaTwoCap(int rowStates, int n, int cap) {
    const double v = std::min((double)rowStates / (double)kTwoCap.rowStatesDiv,
                              (double)kTwoCap.nMul / (double)(n > 0 ? n : 1));
    int c = 1;
    while (c * 2 <= cap && (double)(c * 2) <= v) c *= 2;
    return c;
}

}  // namespace lrqr
