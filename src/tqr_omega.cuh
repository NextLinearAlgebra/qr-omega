#pragma once
// TQR-Omega implements a single-GPU hierarchical Householder QR schedule.
// It builds compact-WY transforms from LRQR panel primitives, selects a legal
// four-level execution plan, and applies each committed transform once per level.
// The planner balances frontier count, workspace, and boundary traffic; measured
// hardware limits remain inputs rather than proof of performance. Contract details
// are maintained in contracts/tqr-omega/.
#include "lrqr.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace tqr {

// Reuse the validated panel primitives and BLAS wrappers from LRQR.
using lrqr::DomArgs;
using lrqr::k_panel_domino;
using lrqr::PanelArgs;
using lrqr::k_panel_fused;
using lrqr::gemm;
using lrqr::trmm;
using lrqr::trsm;
// TSQR and reconstruction primitives used by the hierarchical schedule.
using lrqr::k_load_panel;
using lrqr::k_leaf_geqrf;
using lrqr::k_larft;
using lrqr::k_merge_geqrf;
using lrqr::k_set_identity;
using lrqr::k_dsweep;
using lrqr::k_leaf_apply;
using lrqr::k_lu_sign;
using lrqr::k_negUS;
using lrqr::k_writeYRV;
using lrqr::bqr_smem;
using lrqr::merge_smem;

static constexpr int kRungs = 4;   // register/shared, shared/DSM, DSM/L2, and L2/HBM

// Precision tiers affect only bulk trailing GEMMs; transform construction and
// application remain in the tier's base precision. TF32X3 uses explicit TF32
// splitting for the bulk update.
enum class Tier { FP64, FP32, TF32, TF32X3 };

// TF32 round-to-nearest-even used to make correction terms match tensor-core
// operand rounding. The split stores hi = round_tf32(a) and lo = a - hi.
__device__ __forceinline__ float tf32_hi_rne(float a) {
    unsigned u = __float_as_uint(a);
    const unsigned rem = u & 0x1FFFu;          // the 13 bits tf32 discards
    u &= 0xFFFFE000u;
    // round half to even: strictly-greater rounds up; exactly-half rounds up only
    // when the surviving lsb is 1.
    if (rem > 0x1000u || (rem == 0x1000u && (u & 0x2000u))) u += 0x2000u;
    return __uint_as_float(u);
}

// Split an operand into the TF32-rounded value and its residual. Both are stored
// so correction GEMMs use the same rounded operands as the primary GEMM.
template <typename Real>
__global__ void k_split_hilo(const Real* A, size_t lda, Real* Hi, Real* Lo,
                             size_t ldh, int rows, int cols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    const float a  = (float)A[(size_t)c * lda + r];
    const float hi = tf32_hi_rne(a);
    Hi[(size_t)c * ldh + r] = (Real)hi;
    Lo[(size_t)c * ldh + r] = (Real)(a - hi);
}

// Split two operands in one launch; columns outside either operand are skipped.
template <typename Real>
__global__ void k_split_hilo2(const Real* A, size_t lda, Real* AHi, Real* ALo, int colsA,
                              const Real* B, size_t ldb, Real* BHi, Real* BLo, int colsB,
                              size_t ldh, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    if (c < colsA) {
        const float a  = (float)A[(size_t)c * lda + r];
        const float hi = tf32_hi_rne(a);
        AHi[(size_t)c * ldh + r] = (Real)hi;
        ALo[(size_t)c * ldh + r] = (Real)(a - hi);
    }
    if (c < colsB) {
        const float bb = (float)B[(size_t)c * ldb + r];
        const float hi = tf32_hi_rne(bb);
        BHi[(size_t)c * ldh + r] = (Real)hi;
        BLo[(size_t)c * ldh + r] = (Real)(bb - hi);
    }
}

inline Tier tier_from_env() {
    const char* e = getenv("TQR_TIER");
    if (!e) return Tier::FP64;
    if (!std::strcmp(e, "fp32"))   return Tier::FP32;
    if (!std::strcmp(e, "tf32"))   return Tier::TF32;
    if (!std::strcmp(e, "3xtf32")) return Tier::TF32X3;
    return Tier::FP64;
}
inline const char* tier_name(Tier t) {
    switch (t) { case Tier::FP32: return "fp32"; case Tier::TF32: return "tf32";
                 case Tier::TF32X3: return "3xtf32"; default: return "fp64"; }
}

// Materialize the unit-lower portion of V from the packed QR matrix. This avoids
// repeated triangular operations for narrow epoch updates.
template <typename Real>
__global__ void k_mask_unit_lower(const Real* A, size_t lda, Real* Y, int w) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= w || c >= w) return;
    Y[(size_t)c * w + r] = (r > c) ? A[(size_t)c * lda + r]
                                   : (r == c ? Real(1) : Real(0));
}

static inline int ceil_div(long long a, long long b) { return (int)((a + b - 1) / b); }

// ============================================================================
//  1. THE HIERARCHY  (10:S2.2)
//
//  "At level lambda define: P_lambda peer execution/locality domains visible at
//   the boundary; M_lambda usable fast capacity PER DOMAIN, in words, after
//   mandatory resident state; B_lambda sustainable boundary bandwidth in
//   words/s; alpha_lambda exposed latency per dependency-bearing round."
//
//  M is the FAST SIDE of the boundary. This is where the previous schedule was
//  wrong in a way that changed a cap: it used the L2 capacity for BOTH the
//  DSM<->L2 and the L2->HBM rung, and gave the L2/HBM rung P>1 so that chi
//  bound it. 31:82 and 10:S3.4 are explicit that D=0 and chi=+inf at a
//  single-peer boundary, and 31:553 names this exact rung: "For the illustrative
//  H200 L2/HBM boundary ... that particular single-peer rung".
// ============================================================================
struct Boundary {
    const char* name;
    int    P;          // peer domains at the boundary
    int    P_dev = 132;// device SM count, for the panel's occupancy fraction
    size_t M_words;    // usable fast-side capacity PER DOMAIN, in words
    int    r;          // legal combine fan-in chosen by the plan (>= 2, 10:S4.4)
    double alpha_s;    // exposed latency per dependency-bearing round, seconds
    double B_wps;      // sustainable bandwidth, words/s
};

// Probe the four H200 rungs. Latency/bandwidth are declared calibration inputs
// (31:511 assumption: the certificate covers rate/model error in epsilon), not
// derived quantities -- they enter only T_hat, never legality.
inline void probe_hierarchy(Boundary h[kRungs], int wordBytes, int smemOptinBytes,
                            int smCount, size_t l2Bytes) {
    const double W = (double)wordBytes;
    const int    ctasPerCluster = 8;                       // sm_90 max cluster dim
    const int    residentClusters = std::max(1, smCount / ctasPerCluster);

    // rung 0, registers -> shared. Fast side is the register file addressable by
    // one CTA (64K 32-bit registers), peers are its warps at 512 threads/CTA.
    h[0] = {"reg->shared",  16, smCount, (size_t)(65536.0 * 4.0 / W), 2, 0.2e-6, 5.0e13};
    // rung 1, shared -> DSM. Fast side is one CTA's shared memory; peers are the
    // CTAs of the cluster, which is what a distributed-shared reduction spans.
    h[1] = {"shared->DSM",  ctasPerCluster, smCount, (size_t)(smemOptinBytes / W), 2, 0.5e-6, 2.0e13};
    // rung 2, DSM <-> L2. Fast side is the cluster's aggregate DSM; peers are the
    // concurrently resident clusters.
    h[2] = {"DSM<->L2",     residentClusters, smCount,
            (size_t)((double)ctasPerCluster * smemOptinBytes / W), 2, 2.0e-6, 8.0e12};
    // rung 3, L2 -> HBM. ONE peer domain: there is one L2 and one HBM. Hence
    // D_3 = 0 and chi_3 = +inf, by 31:82 / 10:S3.4 / 31:553.
    h[3] = {"L2->HBM",      1, smCount, (size_t)((double)l2Bytes / W), 2, 5.0e-6, 6.0e11};
}

// ============================================================================
//  2. THE PLAN TUPLE  (10:S4.2 and the required structures of 10:S12)
// ============================================================================
struct LevelPlan {
    int level = 0;
    // --- the tuple pi_lambda = (b, nu, f, p_r, p_c, c, r, d) ---
    int leaf_b = 0;
    int epoch_nu = 0;
    int eager_frontier = 0;
    int p_rows = 1, p_cols = 1;
    int replica_layers = 1;
    int combine_fanin = 2;
    int pipeline_depth = 2;
    // --- derived from the boundary ---
    int    P = 1;
    size_t M_words = 0;
    int    D = 0;                 // ceil(log_r P), 0 at P=1
    double B_budget = 0;          // B_lambda(c)
    double chi = 0;               // sqrt(B/D) or +inf
    int    nu_star = 0;           // min{sqrt(M), B/n, chi}
    int    nu_cap = 0;            // largest nu the workspace constraint admits
    int    nu_binding = 0;        // 0 sqrt(M) | 1 B/n | 2 chi | 3 workspace
    int    K = 0;                 // ceil(n/nu)
    // --- proof record (10:S12) ---
    size_t resident_words = 0, workspace_words = 0;
    unsigned long long predicted_words_moved = 0;
    unsigned long long predicted_messages = 0;
    unsigned long long predicted_sync_rounds = 0;
    double predicted_compute_s = 0, predicted_bandwidth_s = 0;
    double predicted_latency_s = 0, predicted_chain_s = 0;
};

struct ProofRecord {
    LevelPlan levels[kRungs];
    bool householder_only = true;
    bool compact_global_storage = false;
    bool all_versions_legal = true;
    bool every_rung_live = false;
    bool no_cuda_graphs = true;
    double predicted_qr_over_cfar = 0;
    double That_s = 0;                       // the planner's own predicted time
    // false => the legal family came back EMPTY and the degenerate nu=[b,b,b,b]
    // plan is running. 10:S12: "A plan whose model error exceeds the declared
    // tolerance is not silently trusted" -- neither is one that does not exist.
    bool plan_enumerated = true;
    double pred_bandwidth_s = 0, pred_latency_s = 0;   // the tie-break keys
    int    panel_pr = 0;                               // A5 panel row partition
    // Which rung-1 carrier the panel runs on. This is a PLAN choice, not a
    // fallback: a cluster has cheap DSMEM barriers but at most 16 CTAs, while the
    // gmem carrier has ~30x costlier barriers and up to P blocks. Short panels want
    // the cluster; tall ones want the width. Hardcoding the cluster whenever it was
    // legal forced p_r ~ 16 everywhere and cost 24% at 16384x2048 -- the model has
    // to pick, on measurement, which is what a finite planner is for (31:486).
    bool   panel_cluster = false;
    double tpanel_s = 0;                               // measured per-panel cost at this b

    // ---- observed counters (31:640 items 3,5,6,7) ----
    long long split[kRungs]     = {0,0,0,0};
    long long replicate[kRungs] = {0,0,0,0};
    long long combine[kRungs]   = {0,0,0,0};
    long long pipeline[kRungs]  = {0,0,0,0};
    long long words[kRungs]     = {0,0,0,0};   // words crossing the boundary
    long long messages[kRungs]  = {0,0,0,0};
    long long rounds[kRungs]    = {0,0,0,0};   // dependency-bearing frontier rounds
    long long merges[kRungs]    = {0,0,0,0};   // ordered stack-Householder merges
    long long gram_calls   = 0;                // MUST be 0
    long long stale_reads  = 0;                // MUST be 0
    size_t    aux_words    = 0;                // globally allocated auxiliary
    size_t    aux_words_nonconforming = 0;     // the part outside the O(nb) ledger
    size_t    w_leafT = 0, w_ledger = 0, w_protocol = 0;   // by object type
    // DEVICE-WIDE dependency rounds actually executed, per rung. rounds[] counts
    // 31:329's X_{k,j} epoch nodes, which is the right granularity for the domino
    // DAG -- and is blind to everything BELOW an epoch. The rung-1 panel executes a
    // whole barrier schedule inside one epoch node, so S_1 was reported at ~1.0 x
    // (K+D) while the hardware was paying Theta(K_1 * b). 31:640 item 6 asks for
    // "observed frontier rounds"; this is the observation that was missing.
    long long dev_rounds[kRungs] = {0, 0, 0, 0};
};

// ============================================================================
//  3. THE FINITE EXACT PLANNER  (31:486, 10:S4.3-S4.5)
//
//  "The implementation must not use hand-written thresholds such as
//   `if (n <= 4096)` or aspect-specific algorithm branches. It enumerates a
//   compact admissible plan set and minimizes an exact model."
//
//  Legality, all of 10:S4.4 -- none of it is a preference:
//      M_resident <= M_lambda
//      p_r * p_c * c <= P_lambda                                       (A5)
//      nu = q*b, q in N;  f >= nu;  d*nu <= f + pending-log budget
//      Sum_lambda d(nu^2 + nu*f) <= kappa*n*b,  kappa = 3
//      q >= 1, c >= 2, r >= 2, d >= 2 at every on-chip rung          (liveness)
// ============================================================================
// The allocator-side knobs that change what the workspace ledger costs. At
// namespace scope rather than nested in Planner: a defaulted `LedgerCfg lcfg = {}`
// argument on a Planner member needs the type's default member initializers to be
// complete, which they are not while Planner itself is still being defined.
struct LedgerCfg {
    bool wpathTrmm = false;   // trmm path spends the W budget on one buffer
    bool wpathWide = false;   // wide path spends it twice (uncompliant, A/B only)
    bool tf32x3    = false;   // Ootomo-Yokota hi/lo split scratch
    int  mRows     = 1 << 30; // caps the 3xTF32 row chunk, as create() does
    // The PANEL CARRIER's own staging. 10:S6.3's row-slab carrier needs per-leaf and
    // per-merge-node state; the domino needs none (its partials are protocol, not
    // ledger). It depends only on (m, SLAB, NB, n/b) and NOT on the nu-ladder, so it
    // is a constant offset the planner can subtract -- which is the whole point of
    // putting it here rather than testing it after the fact. Testing it afterwards is
    // what made the carrier unreachable: create() planned a ladder that consumed
    // 2.81 x nb of a 3 x nb budget and only THEN asked whether the carrier fit.
    // Inside the ledger, enumerate() simply picks a narrower ladder and argmin T_hat
    // decides whether that trade is worth it -- no threshold, no special case.
    double carrierWords = 0;
};

struct Planner {
    static constexpr double kKappa = 3.0;   // pinned by the bundle, 31:409

    // Total conventional QR work for an m x n matrix. Shape-general: the square
    // (4/3)n^3 form is 47x low on tall cells and inflated every tall Q/B by that
    // factor for an entire session before it was caught.
    static double flops(int m, int n) {
        const double k = (double)std::min(m, n), lng = (double)std::max(m, n);
        return 2.0 * k * k * lng - (2.0 / 3.0) * k * k * k;
    }

    // B_lambda(c) = I + F/(P sqrt(M(c))) + Q_setup(c)                   31:221
    //
    // M(c): 31:224's M = Theta(c n^2 / P) is the DISTRIBUTED full-operand form.
    // 31:230 is explicit that it does not apply here -- "Compact-state
    // replication at on-chip rungs keeps the mechanism live but does not claim
    // the full-operand 1/sqrt(c) volume reduction" -- so on one GPU M(c) is the
    // hardware capacity and c buys no bandwidth. It is a LIVENESS parameter.
    // Q_setup is then the compact-state setup actually paid: (c-1) copies of a
    // nu x nu state per epoch, i.e. the O(nbL) term of 31:441, not Omega((c-1)n^2/P).
    static double budget(const Boundary& h, int m, int n, int c, int nu) {
        const double I = 2.0 * (double)m * (double)n / (double)h.P;   // compulsory in+out
        const double F = flops(m, n);
        const double upd = F / ((double)h.P * std::sqrt((double)std::max<size_t>(h.M_words, 1)));
        const double K = std::ceil((double)n / std::max(1, nu));
        const double setup = (double)(c - 1) * (double)nu * (double)nu * K / (double)h.P;
        return I + upd + setup;
    }

    // nu*_lambda = min{ sqrt(M), B/n, chi },  chi = sqrt(B/D) at P>=2 else +inf 31:266
    static void widths(LevelPlan& L, const Boundary& h, int m, int n, int c) {
        L.P = h.P; L.M_words = h.M_words; L.combine_fanin = h.r;
        // D = ceil(log_r P), and EXACTLY zero at P = 1 (31:82). The previous
        // schedule gave rung 3 a P > 1, so chi bound a cap the spec says does
        // not exist at a single-peer boundary.
        L.D = (h.P <= 1) ? 0 : (int)std::ceil(std::log((double)h.P) / std::log((double)std::max(2, h.r)));
        const double Bl = budget(h, m, n, c, std::max(1, L.epoch_nu ? L.epoch_nu : 1));
        L.B_budget = Bl;
        const double sqrtM = std::sqrt((double)std::max<size_t>(h.M_words, 1));
        const double bn    = Bl / std::max(1, n);
        L.chi = (L.D <= 0) ? std::numeric_limits<double>::infinity()
                           : std::sqrt(Bl / (double)L.D);
        double v = std::min(sqrtM, std::min(bn, L.chi));
        L.nu_star = std::max(1, (int)std::floor(v));
        L.nu_binding = (bn <= sqrtM && bn <= L.chi) ? 1 : (L.chi <= sqrtM ? 2 : 0);
    }

    // The workspace ledger, 31:270 / 31:409, charged exactly as written:
    //     Sum_lambda d_lambda ( nu_lambda^2 + nu_lambda * f_lambda ) <= kappa n b
    static double workspace(const LevelPlan L[kRungs]) {
        double s = 0;
        for (int i = 0; i < kRungs; ++i)
            s += (double)L[i].pipeline_depth *
                 ((double)L[i].epoch_nu * L[i].epoch_nu +
                  (double)L[i].epoch_nu * L[i].eager_frontier);
        return s;
    }

    // ========================================================================
    //  THE WORKSPACE LEDGER, COMPUTED ONCE
    //
    //  31:270 makes  Sum_lambda d_lambda(nu^2 + nu*f) <= kappa*n*b  a LEGALITY
    //  constraint on the plan, and 31:415 makes mn + O(nb) a theorem. Both were
    //  being checked against a formula that did not describe what create() then
    //  allocated, so enumerate() admitted plans the allocator overspent:
    //
    //      workspace() scored   2,695,168 = 2.57 x nb   at 16384^2
    //      create()  allocated  3,411,968 = 3.25 x nb   (measured 3.32)
    //
    //  and the run reported ledger = 3.33 with compact = 0 at 4096^2 and 16384^2 --
    //  a live 31:415 violation in the SHIPPING configuration. The gap decomposed as
    //  Y3/T3 double-charge 0.50 x nb (the model charges one nu^2 per generation, the
    //  allocator places both a compact T and a materialised Y) plus Zb/Mb 0.12 x nb,
    //  which the model did not mention at all.
    //
    //  This is the same defect shape as panelSmemBytes(): one rule, two copies,
    //  drift. So there is now ONE function. enumerate() rejects on it; create()
    //  asserts its own allocation-time tally against it. A divergence is a loud bug
    //  rather than a silent overspend that only surfaces as compact=0.
    //
    //  It mirrors create()'s allocations term for term, deliberately -- a ledger
    //  derived from anything other than the allocations is a claim about a formula.
    // ========================================================================
    static double ledger_words(const LevelPlan L[kRungs], int b, const LedgerCfg& cfg) {
        // d_lambda = 2 exactly: two replica layers, two live pipeline generations
        // (31:441's liveness hypothesis, and create() allocates in pairs to match).
        // If the planner ever enumerates a different depth, both sides move together
        // or the assert in create() fires -- which is the point of having one function.
        const double dg  = 2.0;
        const double nu1 = L[1].epoch_nu, nu2 = L[2].epoch_nu, nu3 = L[3].epoch_nu;
        const double f3  = std::max(L[3].eager_frontier, L[3].epoch_nu);
        const double f2  = L[2].eager_frontier;
        double w = cfg.carrierWords;
        // T and the materialised unit-lower Y, per generation, at rungs 1..3
        w += dg * (2.0 * nu1 * nu1 + 2.0 * nu2 * nu2);      // T1,Y1,T2,Y2
        w += dg * (2.0 * nu3 * nu3);                        // Y3,T3
        w += (double)b * b;                                 // Y0
        // streamed W. WcapWords is the per-generation budget; wf splits it.
        const double wcap  = std::max<double>(nu3, b) * std::max<double>(f3, b);
        const double wnear = std::max<double>(nu2, b) * std::max<double>(f2, b);
        const double wf    = cfg.wpathWide ? 1.0 : 2.0;
        const double nW    = cfg.wpathTrmm ? 1.0 : 2.0;     // W only, or W and W2
        w += dg * nW * (wcap / wf + 1.0);
        w += nW * (wnear / wf + 1.0);
        // compound scratch, (j x w) bounded by the parent x child widths
        w += 2.0 * nu3 * std::max<double>(nu2, b);
        // 3xTF32 hi/lo split scratch, chunked so the footprint is m-independent
        if (cfg.tf32x3) {
            for (int g = 0; g < 3; ++g) {
                const double wmax = (g == 2) ? std::max<double>(nu2, b) : std::max<double>(nu3, b);
                const double cmax = (g == 2) ? std::max<double>(nu2, b) : std::max<double>(f3 / 2, b);
                const double bud  = (g == 2) ? wnear : wcap;
                double sr = std::max(128.0, bud / (2.0 * (wmax + cmax)));
                sr = std::min(sr, (double)cfg.mRows);
                w += 2.0 * sr * wmax;                            // Ahi + Alo
                w += 2.0 * std::max(sr, wmax) * cmax;            // Bhi + Blo
            }
        }
        return w;
    }

    // Exact trailing-update word traffic of the 4-level ladder, from loop
    // dimensions rather than asymptotics (10:S4.3: "compute from tile counts").
    // A rung-lambda commit reads and writes its trailing block once per epoch.
    static double update_words(int m, int n, int b, int nu1, int nu2, int nu3, int rung) {
        const int kmax = std::min(m, n);
        double w = 0;
        if (rung == 3) {                       // far region, once per nu3 epoch
            for (int c3 = 0; c3 < kmax; c3 += nu3) {
                const int w3 = std::min(nu3, kmax - c3);
                const int nc = n - (c3 + w3);
                if (nc > 0) w += 2.0 * (double)(m - c3) * nc + (double)w3 * nc;
            }
        } else if (rung == 2) {                // inside a nu3 block, per nu2 epoch
            for (int c3 = 0; c3 < kmax; c3 += nu3) {
                const int w3 = std::min(nu3, kmax - c3);
                for (int o2 = 0; o2 < w3; o2 += nu2) {
                    const int w2 = std::min(nu2, w3 - o2), c2 = c3 + o2;
                    const int nc = (c3 + w3) - (c2 + w2);
                    if (nc > 0) w += 2.0 * (double)(m - c2) * nc;
                }
            }
        } else if (rung == 1) {                // inside a nu2 block, per nu1 epoch
            for (int c2 = 0; c2 < kmax; c2 += nu2) {
                const int w2 = std::min(nu2, kmax - c2);
                for (int o1 = 0; o1 < w2; o1 += nu1) {
                    const int w1 = std::min(nu1, w2 - o1), c1 = c2 + o1;
                    const int nc = (c2 + w2) - (c1 + w1);
                    if (nc > 0) w += 2.0 * (double)(m - c1) * nc;
                }
            }
        } else {                               // inside a nu1 block, per leaf panel
            for (int c1 = 0; c1 < kmax; c1 += nu1) {
                const int w1 = std::min(nu1, kmax - c1);
                for (int ob = 0; ob < w1; ob += b) {
                    const int wb = std::min(b, w1 - ob), cb = c1 + ob;
                    const int nc = (c1 + w1) - (cb + wb);
                    if (nc > 0) w += 2.0 * (double)(m - cb) * nc;
                }
            }
        }
        return w;
    }

    // T_hat(pi) = max{ T_compute, max Q/B, max alpha*S, T_chain } + T_unoverlappable
    //                                                                    10:S4.3
    // measured_out, when non-null, receives the part of the returned T_hat that is
    // a MEASUREMENT rather than a model -- see the tie-break in enumerate().
    static double That(const Boundary h[kRungs], const LevelPlan L[kRungs],
                       int m, int n, double peak_flops, double tpanel_s = 0.0,
                       double* measured_out = nullptr) {
        const double Tcomp = flops(m, n) / peak_flops;
        double Tbw = 0, Tlat = 0;
        for (int i = 0; i < kRungs; ++i) {
            Tbw  = std::max(Tbw,  (double)L[i].predicted_words_moved / h[i].B_wps);
            Tlat = std::max(Tlat, h[i].alpha_s * (double)L[i].predicted_sync_rounds);
        }
        // T_chain: the ordered panel frontier cannot be removed by replication
        // (10:S3.5). One leaf panel per b columns, each a dependency-bearing round.
        const double Tchain = (double)ceil_div(std::min(m, n), L[0].leaf_b) * h[1].alpha_s;
        // T_hat = max{...} + T_unoverlappable   (10:S4.3, verbatim shape).
        //
        // The panel chain IS the unoverlappable term: 10:S3.5 proves replication
        // "does not permit two dependent panels to be committed out of order", so
        // no amount of trailing work hides it. tpanel_s is a MEASURED per-panel
        // cost at this b -- 10:S4.3: "Use measured kernel-rate tables, not
        // theoretical peaks" -- and it is the only term in the model that can
        // separate two admissible leaf widths, because the two obvious candidates
        // cannot: total minipanel barrier steps are K0*(b/PW) = n/PW, INDEPENDENT
        // of b, and panel launches (n/b) are ~0.4% of wall at n=2048.
        //
        // This is a measurement, not a fitted constant: nothing is regressed, no
        // law is proposed, and if the probe is unavailable the term is zero and the
        // planner degrades to the max-form it had before.
        // ---- panel vs far: a TWO-RESOURCE overlap, not an occupancy discount ----
        //
        // The first version of this priced the panel only by the SMs it denies to
        // far work, (p_r/P)*t. That is half the story and it measured as such: the
        // planner chose p_r = 16 of 132 everywhere, which won +8.9% at 16384^2 and
        // lost 22% at 32768x4096. The missing half is that the panel is ON THE
        // CRITICAL PATH (10:S3.5: replication "does not permit two dependent panels
        // to be committed out of order"), so lengthening it costs wall time 1:1 --
        // the far work it frees was going to be done anyway.
        //
        // Both effects, as a max over two resources:
        //     chain    = K0 * t_panel(p_r)                  the serial spine
        //     far_time = F_far/peak + chain*(p_r/P)         far work, minus the
        //                                                   slice the panel occupies
        // Small p_r wins only when the chain it adds is cheaper than the far time
        // it recovers. Nothing fitted: t_panel is measured per p_r by the probe,
        // p_r/P is counted, and F_far comes from 10:S9.1's own identities below.
        const double K0    = (double)ceil_div(std::min(m, n), L[0].leaf_b);
        const double chain = K0 * tpanel_s;
        const double occ   = std::min(1.0, (double)L[1].p_rows / (double)std::max(1, h[1].P_dev));
        // 10:S9.1, exact for square QR and the right order otherwise:
        //     F_panel/F = (3/4)(b/n),  F_near/F = (3/2)(f/n),  phi = 1 - both.
        // This is also where f finally pays for itself: S9.1 calls f "a first-class
        // wall-time variable, not only a legality constraint" and requires T_hat to
        // "explicitly score this coupling". Wider f buys GEMM efficiency in the
        // issue term and is charged near-update flops here.
        const double nn    = (double)std::max(1, n);
        const double fpan  = 0.75 * (double)L[0].leaf_b / nn;
        const double fnear = 1.50 * (double)L[3].eager_frontier / nn;
        const double phi   = std::max(0.05, 1.0 - fpan - fnear);
        const double Tfar  = flops(m, n) * phi / peak_flops + chain * occ;

        // f enters T_hat through the ISSUE COUNT, which is where it actually acts.
        // The trailing update is streamed in chunks of f/2 columns (two live W
        // buffers), so the number of dispatched GEMMs at rung lambda is
        // Q_lambda-shaped / chunk, and each carries a fixed issue cost alpha. A
        // model that priced only bytes would rank every f identically and the
        // enumeration would be decorative.
        double Tissue = 0;
        for (int i = 0; i < kRungs; ++i) {
            const int chunk = std::max(1, L[i].eager_frontier / 2);
            const double nchunk = (double)L[i].predicted_words_moved /
                                  std::max(1.0, 2.0 * (double)m * chunk);
            Tissue += nchunk * h[3].alpha_s;
        }
        const double core = std::max(std::max(Tcomp, Tbw),
                            std::max(std::max(Tlat, Tchain), std::max(chain, Tfar)));
        // ---- how much of `core` is measurement, not model ----
        //
        // Everything above is a model except `chain = K0 * t_panel(p_r)`, whose
        // t_panel is probed on this device at this b and this p_r. The two branches
        // that carry it are `chain` itself and `Tfar` (which contains chain*occ).
        // enumerate() needs this split because the calibration band it applies to
        // ties is a statement about MODEL error, and must not be allowed to erase a
        // measured difference.
        if (measured_out) {
            *measured_out = (core == chain)             ? chain
                          : (core == Tfar)              ? chain * occ
                                                        : 0.0;
        }
        return core + Tissue;
    }

    // Exhaustive enumeration over the declared legal family; returns argmin T_hat.
    // The family is small by construction: b from the kernel's admissible set,
    // then each nu a multiple of its child (legal nesting, A2), each bounded above
    // by min(nu*, workspace cap).
    static bool enumerate(const Boundary h[kRungs], int m, int n, int b_fixed,
                          double peak_flops, ProofRecord& out, double tpanel_s = 0.0,
                          int smDev = 132, const double* tpanel_pr = nullptr,
                          int panel_smem_bytes = 0, int smemCapBytes = 1 << 30,
                          LedgerCfg lcfg = {}) {
        // tpanel_pr, when supplied, has one entry per prCand slot.
        // panel_smem_bytes / smemCapBytes carry A3's resident-state budget for the
        // leaf width: the caller knows the instantiated kernel's footprint.
        const int kmax = std::min(m, n);
        double best = std::numeric_limits<double>::infinity();
        double bestQ = std::numeric_limits<double>::infinity();
        double bestS = std::numeric_limits<double>::infinity();
        double bestMeas = 0.0;   // measured part of the incumbent's T_hat
        // ---- MEASURED NEGATIVE: the calibration band does not pay ----
        //
        // The idea was sound on its face. 31:511 certifies T <= (1+eps)T_hat + delta,
        // T_hat measures 66% off at 16384^2, and the audit showed ladders being
        // separated by 0.3% of T_hat -- so banding ties and falling through to
        // sum Q/B, the quantity 31:281 is actually about, should have been free
        // width. It was not. Three arms, one job, one device, cuSOLVER per arm
        // (job 9679, geomean speedup over 10 cells):
        //
        //     off  (exact argmin)                 1.3358   <-- ships
        //     model(band the modelled residue)    1.3295
        //     flat (band all of T_hat)            1.3034
        //
        // The whole difference is at 32768^2 (1.31 / 1.22 / 1.23), where all three
        // arms pick the SAME b, p_r and carrier and differ only in the nu-ladder:
        // the band's Q/B fallthrough takes a ladder that T_hat ranked marginally
        // worse and that is in fact 6.7% slower. So T_hat's ABSOLUTE error being
        // 66% does not make its RANKING noise at 0.3% -- the two are different
        // claims, and only the first was ever measured.
        //
        // Reverting is also the more literal spec: 31:486 says pi_hat = argmin T_hat,
        // with no band. The knob stays because the arms must remain reproducible,
        // and the default is the exact argmin.
        static const int bandMode = []{
            const char* e = getenv("TQR_BAND");
            if (!e) return 2;                       // exact argmin, 31:486 as written
            if (!std::strcmp(e, "flat"))  return 1;
            if (!std::strcmp(e, "model")) return 0;
            return 2;
        }();
        bool found = false;
        struct Cand { int b, nu1, nu2, nu3, f, pr; bool cluster;
                      double That, q, s; };
        Cand cands[8]{};
        int  nCand = 0;
        // The winner needs its OWN slot. The kept set is pruned by raw T_hat, but
        // the argmin is the BANDED lexicographic key, so the selected plan can be --
        // and was -- absent from its own audit: job 9676 printed eight candidates at
        // nu_3 = 512/448 and then shipped nu_3 = 832. An audit that cannot show the
        // winner cannot explain the choice, which is the whole point of roadmap
        // gate #5.
        Cand winner{};
        bool haveWinner = false;

        out.tpanel_s = tpanel_s;
        // b is selected from the kernel's admissible register/shared set, NOT
        // from sqrt(M) (10:S6.2). Here the domino kernel is instantiated at one
        // NB, so the admissible set is the singleton {NB} and the enumeration
        // over b is degenerate -- recorded rather than hidden, because widening
        // it is a real and separate lever (a bigger b raises the whole workspace
        // ledger kappa*n*b and therefore every nu cap above it).
        const int b = b_fixed;

        LevelPlan L[kRungs];
        for (int i = 0; i < kRungs; ++i) {
            L[i].level = i; L[i].leaf_b = b;
            L[i].pipeline_depth = 2;          // d >= 2, liveness
            L[i].replica_layers = 2;          // c >= 2, liveness
            L[i].combine_fanin = h[i].r;      // r >= 2, liveness
        }
        L[0].epoch_nu = b; L[0].eager_frontier = b;
        widths(L[0], h[0], m, n, L[0].replica_layers);

        // Provisional nu* at rungs 1..3 to bound the search (B depends weakly on nu
        // through the setup term only, so one pass is enough to fix the ranges).
        for (int i = 1; i < kRungs; ++i) { L[i].epoch_nu = b; widths(L[i], h[i], m, n, 2); }
        const int cap1 = std::min(L[1].nu_star, kmax);
        const int cap2 = std::min(L[2].nu_star, kmax);
        const int cap3 = std::min(L[3].nu_star, kmax);

        // 31:78 defines the width against the LEAF: "nu_lambda = q_lambda b,
        // q_lambda in N" -- for every lambda, not each against its child. A2 asks
        // only that block sizes be "nested UP TO ROUNDING", which the epoch loops
        // already realise (w2 = min(nu2, w3 - o2) truncates the last child).
        //
        // Requiring nu3 = q3*nu2 exactly, as this first did, is strictly stronger
        // than the spec and it bites: at n=16384 the best nu3 is 832 = 13*64, whose
        // only divisors that are multiples of 64 are 64 and 832, and 832 exceeds the
        // A7 capacity cap at rung 2 -- so nu2 was FORCED to 64 and rung 2 collapsed
        // to the leaf width. The ladder disappeared for a reason that was mine.
        for (int q1 = 1; q1 * b <= std::max(b, cap1); ++q1) {
          const int nu1 = q1 * b;
          for (int q2 = q1; q2 * b <= std::max(nu1, cap2); ++q2) {
            const int nu2 = q2 * b;
            for (int q3 = q2; q3 * b <= std::max(nu2, cap3); ++q3) {
              const int nu3 = q3 * b;
            // ---- f_lambda is ENUMERATED, not assumed ----
            //
            // 10:S4.2 puts f in the plan tuple and 10:S4.3 says "f is SCORED, not
            // legality-only". Pinning f = nu, as this first did, is a hidden
            // decision with a large measured price: the live streamed-W storage is
            // two buffers (W and T^T W) of nu x chunk, so two buffers at
            // chunk = f/2 IS f = nu, and full-width chunks are f = 2*nu. Fixing
            // f = nu therefore forced half-width chunks everywhere, which measured
            // 21% slower at 8192^2 and 16384^2 than the same schedule with the
            // storage bound simply ignored.
            //
            // The trade is real and belongs to the planner: larger f buys GEMM
            // efficiency, spends workspace, and so shrinks the nu it can afford.
            // Enumerating it is what turns a 21% "cost of compliance" into a
            // decision the model makes on the exact counts.
            for (int fm = 1; fm <= 4; fm *= 2) {
            // 16 is exactly a full non-portable cluster and 8 a portable one, so
            // these two candidates are what make the cluster carrier reachable
            // through the plan rather than only as an incidental fallback.
            const int prCand[6] = {smDev, smDev / 2, smDev / 4, smDev / 8, 16, 8};
            for (int prIdx = 0; prIdx < 6; ++prIdx) {
              const int pr = prCand[prIdx];
              if (pr < 1) continue;

              LevelPlan C[kRungs];
              for (int i = 0; i < kRungs; ++i) C[i] = L[i];
              C[0].epoch_nu = b;   C[0].eager_frontier = b * fm;
              C[1].epoch_nu = nu1; C[1].eager_frontier = nu1 * fm;
              C[2].epoch_nu = nu2; C[2].eager_frontier = nu2 * fm;
              C[3].epoch_nu = nu3; C[3].eager_frontier = nu3 * fm;
              for (int i = 0; i < kRungs; ++i) widths(C[i], h[i], m, n, C[i].replica_layers);

              // --- legality ---
              // workspace ledger, 31:270, charged against what the ALLOCATOR will
              // actually place -- see ledger_words(). The old test used a formula
              // that under-counted by 0.68 x nb, so plans measuring 3.33 x nb were
              // admitted against a kappa = 3 bound and shipped with compact = 0.
              if (ledger_words(C, b, lcfg) > kKappa * (double)n * (double)b) continue;
              // A5, read as written. 31:243: "a DISTRIBUTED 2.5D rung uses
              // p_r p_c c <= P_lambda", and 10:S7.1 states it under "At a
              // distributed rung choose p_r p_c = P". The constraint partitions
              // PROCESSORS; c counts replica layers, which at a distributed rung
              // are processor groups and so join the product.
              //
              // At a single-peer rung (P=1: the L2/HBM boundary of one GPU) there
              // is nothing to partition, and c is not a processor count at all --
              // 31:429 names its replicas "compact transform state and in-flight
              // tiles", i.e. GENERATIONS of compact state, which cost storage and
              // not processors. Folding c into the product there asserts that one
              // HBM must be split into two peers to hold two in-flight tiles,
              // which is false, and it rejected every legal ladder: the planner
              // returned nothing at all and the fallback shipped nu=[b,b,b,b].
              //
              // So: p_r*p_c <= P always; c joins the product only where P >= 2.
              bool a5 = true;
              for (int i = 0; i < kRungs; ++i) {
                  const bool distributed = (h[i].P >= 2);
                  C[i].p_rows = distributed ? std::max(1, h[i].P / C[i].replica_layers) : 1;
                  C[i].p_cols = 1;
                  const long long prod = (long long)C[i].p_rows * C[i].p_cols *
                                         (distributed ? C[i].replica_layers : 1);
                  if (prod > h[i].P) a5 = false;
              }
              if (!a5) continue;
              // The PANEL's row partition is a separate, device-wide quantity: the
              // panel grid spans SMs, not the cluster peers rung 1 counts. It is
              // enumerated over the device's processor budget so the split between
              // frontier and far is chosen by the model (10:S6.7 forbids fixing it).
              C[1].p_rows = pr;
              // ---- A7 applies to every rung. Only the LEAF WIDTH is exempt ----
              //
              // 31:243's A7 is nu_lambda^2 <= theta*M_lambda, and 31:405 says what
              // that nu^2 IS: "a collapsed width-nu_lambda compact-WY transform has
              // a triangular block of size Theta(nu_lambda^2)" -- the WY block that
              // must be RESIDENT on that boundary's fast side.
              //
              // RUNG 0 IS EXEMPT, and the spec says so in as many words. 10:S6.2:
              // "The leaf width b is selected from the kernel's admissible
              // register/shared-memory set. It is NOT derived from the entire SM
              // register-file capacity by a single sqrt(M) formula." So b is bounded
              // by A3's declared resident-state budget -- does the panel kernel's
              // shared-memory footprint actually fit -- and not by sqrt(M_0).
              //
              // RUNGS 1..3 ARE NOT EXEMPT, and an earlier cut of this code exempted
              // rung 1 as well. The argument was that rung 1's T is nu_1 x nu_1 in
              // GLOBAL memory, so charging it against per-CTA shared memory prices a
              // block that is not there. That argument is backwards: 10:S6.3 says
              // "compact transform state is replicated to cluster CTAs", so rung 1's
              // compact state is SUPPOSED to sit on the shared/DSM side, and A7 is
              // the constraint that keeps it able to. An implementation that parks it
              // in gmem is a departure to fix, not a licence to drop the constraint --
              // relaxing a spec bound because the code does not currently honour it is
              // exactly backwards.
              //
              // It also buys nothing: A7 at rung 1 gives nu_1 <= sqrt(0.9*29056) = 161,
              // and every shipping ladder measured has nu_1 in {64, 128}. The only
              // thing the exemption unlocked was b=256 via nesting -- and b=256 is
              // independently dead, because t_panel measures SUPER-linear in b
              // (2.30x for 2x, 5.89x for 4x, job 9685), so the panel chain
              // (n/b)*t_panel(b) GROWS 15-47%. Restoring A7 here re-blocks b=256
              // through the honest mechanism instead of a special case on NB.
              constexpr double kTheta = 0.9;
              bool cap_ok = true;
              for (int i = 1; i < kRungs; ++i)
                  if ((double)C[i].epoch_nu * C[i].epoch_nu > kTheta * (double)C[i].M_words)
                      cap_ok = false;
              // rung 0: A3's resident-state budget for this leaf width (10:S6.2).
              if (panel_smem_bytes > smemCapBytes) cap_ok = false;
              if (!cap_ok) continue;
              // f >= nu and d*nu <= f + pending-log budget (10:S4.4). With
              // f = nu the pending-log budget must cover (d-1)*nu, which the
              // T compound of the parent rung supplies exactly.
              for (int i = 0; i < kRungs; ++i)
                  if (C[i].eager_frontier < C[i].epoch_nu) cap_ok = false;
              if (!cap_ok) continue;

              // --- exact predicted counters ---
              for (int i = 0; i < kRungs; ++i) {
                  C[i].K = ceil_div(kmax, C[i].epoch_nu);
                  C[i].predicted_sync_rounds = (unsigned long long)(C[i].K + C[i].D);
                  C[i].predicted_words_moved =
                      (unsigned long long)update_words(m, n, b, nu1, nu2, nu3, i);
                  C[i].predicted_messages = (unsigned long long)C[i].K * std::max(1, C[i].D);
                  C[i].workspace_words = (size_t)C[i].pipeline_depth *
                      ((size_t)C[i].epoch_nu * C[i].epoch_nu +
                       (size_t)C[i].epoch_nu * C[i].eager_frontier);
                  C[i].resident_words = (size_t)C[i].epoch_nu * C[i].epoch_nu;
                  C[i].nu_cap = (int)std::floor(std::sqrt(kKappa * (double)n * b /
                                    (2.0 * (double)C[i].pipeline_depth)));
                  if (C[i].nu_cap < C[i].nu_star) C[i].nu_binding = 3;
              }
              // ---- argmin, with the tie broken by the two quantities ----
              //
              // T_hat is the max-form of 10:S4.3, and on this device T_compute
              // dominates it for EVERY legal ladder at once: at 4096^2 fp64 it is
              // 1.367e-3 s while the widest and the narrowest ladder differ only
              // in a bandwidth term below it. So every candidate ties, the first
              // enumerated one wins, and the planner shipped nu = [b,b,b,b] -- the
              // degenerate ladder -- while reporting an exact argmin. It was.
              //
              // The fix is not to invent an overlap model. 10:S2.5 is explicit
              // that the model "does not assume that all boundary times add", so
              // adding them would be a fabrication. Instead the argmin is made
              // LEXICOGRAPHIC on (T_hat, sum Q/B, sum alpha*S): among plans whose
              // predicted time is genuinely equal, prefer the one with less
              // communication, then less synchronization. Those are the two
              // quantities the theorem is about, so this is a refinement of an
              // exactly-tied optimum, not a second cost model.
              // predicted counters first: T_hat's issue term reads them.
              for (int i = 0; i < kRungs; ++i) {
                  C[i].K = ceil_div(kmax, C[i].epoch_nu);
                  C[i].predicted_sync_rounds = (unsigned long long)(C[i].K + C[i].D);
                  C[i].predicted_words_moved =
                      (unsigned long long)update_words(m, n, b, nu1, nu2, nu3, i);
              }
              // the probe's measurement AT THIS p_r, not a single global one
              const double tp = tpanel_pr ? tpanel_pr[prIdx] : tpanel_s;
              double tmeas = 0.0;
              const double t = That(h, C, m, n, peak_flops, tp, &tmeas);
              double qsum = 0, ssum = 0;
              for (int i = 0; i < kRungs; ++i) {
                  qsum += (double)C[i].predicted_words_moved / h[i].B_wps;
                  ssum += h[i].alpha_s * (double)C[i].predicted_sync_rounds;
              }
              // ---- ties are decided at the MODEL'S accuracy, and ONLY there ----
              //
              // 31:511's calibration certificate is T(pi) <= (1+eps)*T_hat(pi) + delta:
              // the MODEL has a declared error band, and ranking two plans whose
              // modelled cost differs by less than eps is not a measurement, it is
              // noise with a tie-break attached. Measured here, T_hat = 0.128 s
              // against 0.212 s actual at 16384^2 -- 66% off -- while the audit
              // shows the ladder being selected on 0.3% differences in T_hat.
              //
              // So candidates within eps of the best are TIED and the lexicographic
              // key falls through to sum Q/B -- the quantity the theorem is actually
              // about (31:281). That is what widened nu_3 from 512 to 832 while
              // Q_3/B_3 sat at 1.80.
              //
              // BUT THE BAND MUST NOT COVER THE MEASURED TERM. T_hat's chain term is
              // K0 * t_panel(p_r), probed on this device (10:S4.3: "use measured
              // kernel-rate tables"); model error says nothing about it. Applying one
              // flat eps to the whole T_hat tied two plans whose measured panel
              // chains genuinely differed, and sum Q/B -- which prefers fewer row
              // partitions because they move fewer words -- then picked the slower
              // one. Measured cost, same GPU, job 9660 -> 9676: p_r 33 -> 16 at
              // 32768^2 and 1.31x -> 1.23x; 16384^2 1.49x -> 1.44x; 32768x4096
              // 2.12x -> 2.03x. Nothing improved.
              //
              // Banding only (T_hat - measured) separates the two cases exactly:
              // candidates differing in nu/f differ only in modelled terms and stay
              // tied, so Q/B still decides; candidates differing in p_r differ in the
              // measured chain and are ranked on the measurement, as they must be.
              // TQR_BAND selects WHAT the band covers, so the three positions can be
              // A/B'd against each other on one device in one job rather than across
              // rebuilds. Default is the spec-faithful one.
              //   model (default) : band the modelled residue, keep measurement exact
              //   flat            : band all of T_hat        (what job 9676 shipped)
              //   off             : no band, exact argmin     (what job 9660 shipped)
              constexpr double kEpsCalib = 0.05;      // declared, like kappa and theta
              const double rel = (bandMode == 2) ? 0.0
                               : (bandMode == 1) ? kEpsCalib * std::max(1e-12, best)
                               : kEpsCalib * std::max(1e-12, best - bestMeas);
              const bool better = (t < best - rel) ||
                                  (std::abs(t - best) <= rel &&
                                   (qsum < bestQ - 1e-15 ||
                                    (std::abs(qsum - bestQ) <= 1e-15 && ssum < bestS)));
              // Keep the top few candidates so the SELECTION is auditable, not just
              // the winner. Roadmap gate #5 asks for this and the runners-up have
              // never been printed -- which is why "nu_3 = 640 although 1024 is
              // legal" could not be explained by reading the code.
              if (nCand < 8) {
                  Cand& cd = cands[nCand++];
                  cd.b = b; cd.nu1 = nu1; cd.nu2 = nu2; cd.nu3 = nu3;
                  cd.f = C[3].eager_frontier; cd.pr = pr; cd.cluster = (prIdx >= 4);
                  cd.That = t; cd.q = qsum; cd.s = ssum;
              } else {
                  int worst = 0;
                  for (int z = 1; z < 8; ++z)
                      if (cands[z].That > cands[worst].That ||
                          (cands[z].That == cands[worst].That && cands[z].q > cands[worst].q))
                          worst = z;
                  if (t < cands[worst].That ||
                      (t == cands[worst].That && qsum < cands[worst].q)) {
                      Cand& cd = cands[worst];
                      cd.b = b; cd.nu1 = nu1; cd.nu2 = nu2; cd.nu3 = nu3;
                      cd.f = C[3].eager_frontier; cd.pr = pr; cd.cluster = (prIdx >= 4);
                      cd.That = t; cd.q = qsum; cd.s = ssum;
                  }
              }
              if (better) {
                  best = std::min(best, t); bestQ = qsum; bestS = ssum;
                  bestMeas = tmeas; found = true;
                  winner.b = b; winner.nu1 = nu1; winner.nu2 = nu2; winner.nu3 = nu3;
                  winner.f = C[3].eager_frontier; winner.pr = pr;
                  winner.cluster = (prIdx >= 4);
                  winner.That = t; winner.q = qsum; winner.s = ssum;
                  haveWinner = true;
                  out.panel_pr = pr;
                  out.panel_cluster = (prIdx >= 4);   // slots 4,5 are the cluster carrier
                  for (int i = 0; i < kRungs; ++i) out.levels[i] = C[i];
                  out.That_s = t;
                  out.pred_bandwidth_s = qsum;
                  out.pred_latency_s = ssum;
              }
            }
            }
            }
          }
        }
        // emit the audit trail: winner first, then the closest runners-up
        if (found && getenv("TQR_AUDIT")) {
            std::printf("  [plan audit] m=%d n=%d -- winner, then closest runners-up "
                        "(T_hat, measured part, sum Q/B, sum aS)\n", m, n);
            const Cand* rows[9]; int nRow = 0;
            if (haveWinner) rows[nRow++] = &winner;
            for (int z = 0; z < nCand; ++z) rows[nRow++] = &cands[z];
            for (int z = 0; z < nRow; ++z) {
                const Cand& c = *rows[z];
                const bool sel = (z == 0 && haveWinner);
                std::printf("    b=%-4d nu=[%d,%d,%d] f3=%-5d p_r=%-4d %-8s "
                            "That=%.6g  Q/B=%.6g  aS=%.6g%s\n",
                            c.b, c.nu1, c.nu2, c.nu3, c.f, c.pr,
                            c.cluster ? "cluster" : "gmem", c.That, c.q, c.s,
                            sel ? "   <== SELECTED" : "");
            }
            std::printf("    [band] mode=%s  T_hat=%.6g measured=%.6g band=%.4g\n",
                        bandMode == 1 ? "flat" : bandMode == 2 ? "off(argmin)" : "model",
                        best, bestMeas,
                        bandMode == 2 ? 0.0
                      : bandMode == 1 ? 0.05 * best
                                      : 0.05 * std::max(0.0, best - bestMeas));
        }
        return found;
    }
};

}  // namespace tqr

namespace tqr {

// geam is the only BLAS3 op lrqr.cuh does not already wrap file-scope. It is used
// for the two places dlarfb needs a strided copy or an in-place subtract, both of
// which would otherwise cost a bespoke kernel.
inline void geam(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                 int m, int n, float a, const float* A, int lda,
                 float bta, const float* B, int ldb, float* C, int ldc) {
    CUBLAS_CHECK(cublasSgeam(h, opA, opB, m, n, &a, A, lda, &bta, B, ldb, C, ldc));
}
inline void geam(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                 int m, int n, double a, const double* A, int lda,
                 double bta, const double* B, int ldb, double* C, int ldc) {
    CUBLAS_CHECK(cublasDgeam(h, opA, opB, m, n, &a, A, lda, &bta, B, ldb, C, ldc));
}

// ============================================================================
//  4. THE SCHEDULER
//
//  One nested execution realising the four moves at all four rungs over the
//  planned nu-ladder. Correctness first, by explicit instruction: this cut runs
//  a single stream, so rung-2/3 Pipeline counters are honestly zero and
//  every_rung_live reports false until the streamed generations land. Rung-0/1
//  Pipeline is real and nonzero -- it is the domino's own NB/PW minipanel
//  wavefront -- and is counted from the kernel geometry, not asserted.
// ============================================================================
template<typename Real, int NB>
struct Omega {
    // ---- problem ----
    int m = 0, n = 0, kmax = 0;
    size_t lda = 0;
    // ---- plan ----
    ProofRecord rec;
    Boundary hier[kRungs];
    int b = NB, nu1 = NB, nu2 = NB, nu3 = NB, f3 = NB;
    // ---- device state ----
    // d_lambda = 2 pipeline generations (10:S4.4 liveness, 31:441 hypothesis).
    // Generation 0 is the frontier: panels, and the near commits the next epoch
    // depends on. Generation 1 is the deferred far commit, which no epoch reads
    // until two epochs later -- 31:429's rung-3 Pipeline carrier, "streamed
    // batched WY applies". Each generation owns its own streamed W, because they
    // are concurrent; sharing one W buffer would be a race, not a saving.
    cublasHandle_t cb = nullptr, cbDef = nullptr, cbNear = nullptr;
    cudaStream_t   st = nullptr, sDef = nullptr, sNear = nullptr;
    cudaEvent_t    evEpoch = nullptr, evDef[2] = {nullptr, nullptr};
    cudaEvent_t    evFront = nullptr, evNearLast = nullptr;
    bool           defPending[2] = {false, false};
    bool           nearLive = false;
    bool           pipeOn = true;
    // TQR_WPATH selects how the streamed W is realised. All three compute the same
    // arithmetic; they differ only in storage and in which BLAS3 kernel runs.
    //   gemm (default) : two nu x (f/2) buffers -- total nu*f, EXACTLY the ledger
    //   trmm           : one nu x f buffer, in-place triangular -- also within ledger
    //   wide           : two nu x f buffers -- 2x the ledger's W term, NOT compliant
    // 'wide' exists to price the storage bound: without it, "compliance is free" or
    // "compliance costs 20%" would both be assertions.
    bool           wpathTrmm = false, wpathWide = false;
    Tier           tier = Tier::FP64;
    // True only for the K-large trailing GEMMs. Every other GEMM in this file --
    // the T-apply, the Yd legs, and all of compound() -- passes false, so the tier
    // cannot reach the transform state even if the handle's math mode drifts.
    bool bulk_tf32() const {
        return !std::is_same<Real, double>::value &&
               (tier == Tier::TF32 || tier == Tier::TF32X3);
    }
    int  smCount = 132, smemOptin = 232448;
    int  panelPr = 0;          // A5 row partition for the panel, from the plan
    bool panelCluster = false; // rung-1 carrier chosen by the plan

    Real *Tleaf = nullptr;                       // (kmax/b) leaf T blocks : n*b words
    // ---- c_lambda = 2 replica layers of compact state at rungs 1 and 2 ----
    // 31:429: rung 1 replicates "compact R,T,W in cluster lanes", rung 2 "a
    // constant number of compact global slots". Same role as T3's two layers and
    // for the same reason: the deferred near generation reads T1 (or T2) for epoch
    // j on sNear while the frontier is already zeroing it for epoch j+1.
    //
    // This was diagnosed the hard way. Adding the near look-ahead with single
    // buffers failed 11 of 19 gate cells; I attributed it to the event-parity
    // bookkeeping, fixed that, and measured EXACTLY 11 again -- the count did not
    // move, so the bookkeeping was never the variable. The third time this session
    // that Pipeline turned out to be illegal without Replicate under it, which is
    // what 31:329 and the 31:429 carrier table say and what "the four moves as a
    // whole" means operationally.
    Real *T1[2] = {nullptr, nullptr};
    Real *T2[2] = {nullptr, nullptr};
    // Layer-release events: evT[r][p] fires when the deferred generation that read
    // rung-r layer p is done, so the layer can be rebuilt. Two layers, two live
    // generations: d_lambda = 2 exactly, not a queue.
    cudaEvent_t evT[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    bool        evTValid[2][2] = {{false, false}, {false, false}};
    // The materialised unit-lower V block, one per live layer, same lifetime and
    // same release discipline as the T layer it accompanies.
    Real *Y1[2] = {nullptr, nullptr};
    Real *Y2[2] = {nullptr, nullptr};
    Real *Y0 = nullptr;                 // rung 0: frontier only, never deferred
    // Rung 3's mask. The profile named TRMM as the second category at small/mid n
    // -- 13.4% of GPU time and 540 instances at 2048^2 -- and every one of them is
    // a rung-3 commit still on the storage-free TRMM path. Materialising the mask
    // there turns 2 TRMM + 2 GEAM into 2 GEMM per chunk.
    //
    // TWO layers, parity-indexed like T3. The mask is read-only once built, so the
    // eager and deferred generations may share a layer -- but epoch e+1 REBUILDS it
    // while epoch e's deferred generation is still reading, and that is the same
    // race that has now bitten four times. Replicate first, measure second.
    Real *Y3[2] = {nullptr, nullptr};
    // ---- rung-3 REPLICATE: c_3 = 2 layers of compact transform state ----
    // 31:429 gives rung 3's Replicate carrier as "compact transform state and
    // in-flight tiles". This is that carrier, and it is not decoration: the
    // deferred far generation READS T3 for epoch e while the frontier generation
    // is already zeroing and rebuilding T3 for epoch e+1. With one buffer that is
    // a race, and it measured as one -- 18 of 19 gate cells failed with the
    // pipeline on and 0 with it off, the two arms differing in nothing else.
    //
    // So REPLICATE is the enabling condition for PIPELINE at rung 3, exactly as
    // it is for the diagonal edge at rung 2 (31:329: "Replicated compact transform
    // state allows the required update to occur locally at that stage"). The four
    // moves are one mechanism; taking Pipeline without Replicate is not a partial
    // implementation of the spec, it is an incorrect program.
    //
    // The storage ledger already paid for this: the planner charges
    // d_3 (nu_3^2 + nu_3 f_3) with d_3 = 2, i.e. two generations of the width-nu_3
    // compact state. One buffer was under-spending its own budget.
    Real *T3[2] = {nullptr, nullptr};
    // Three W generations, not three copies of the same size. Generation 2 serves
    // the NEAR commits, whose width is bounded by nu2, so charging it nu3*f3 would
    // spend the storage ledger on space the near path can never address.
    // Two half-width buffers per generation: 2 * nu * (f/2) == the nu*f the ledger
    // charges. Splitting the budget, not exceeding it.
    Real *Wbuf[3] = {nullptr, nullptr, nullptr};
    Real *W2buf[3] = {nullptr, nullptr, nullptr};
    size_t WcapWords = 0, WnearWords = 0;
    Real *Zb = nullptr, *Mb = nullptr;           // compound scratch : Theta(nu3^2)
    // 3xTF32 low-part scratch. K-chunked so the footprint is independent of m and
    // sits inside the same streaming budget as W; see split_rows().
    // PER GENERATION. This is the fourth time in this file that state shared
    // across concurrent pipeline generations turned into a data race: T3, then T1
    // and T2, and now the three-product split scratch. Generations 0/1/2 run on
    // st/sDef/sNear simultaneously, so one set of hi/lo buffers is one set too few.
    // 31:429 lists rung 3's Replicate carrier as "compact transform state AND
    // IN-FLIGHT TILES" -- these buffers are in-flight tiles, and the spec has been
    // saying so the whole time.
    // ---- TSQR panel state ----
    // Pbuf/QT are mpad x NB and QS/Vm/TmM/TmL are O(N*b^2): the staging this
    // algorithm needs is O(mb), which 31:415 excludes. That is the SELECTION rule,
    // not an obstacle -- on squares it is ~2-3x nb and affordable, exactly where the
    // panel is 89-99% of wall; on tall cells it is ~20x nb and inadmissible, exactly
    // where the panel is only 16-33% and the domino is already fine. It is charged
    // through the LEDGER tally so the workspace constraint does the gating.
    Real *Pbuf = nullptr, *QT = nullptr, *QS = nullptr;
    Real *Vm = nullptr, *TmM = nullptr, *TmL = nullptr, *tauL = nullptr, *taum = nullptr;
    Real *Sg = nullptr, *Rroot = nullptr;
    int   domPpBlocks = 0;         // block count domPp was sized for; asserted at launch
    int2 *dNodes = nullptr;
    struct PTab { int N, nodesOff, nlv; int2 lvl[24]; };
    std::vector<PTab> ptab;
    int  mpadMax = 0, NmaxLeaf = 0;
    bool   tsqrReady = false;
    size_t tsqrNeedWords = 0;      // what the carrier asked for, legal or not

    // Which carrier the caller asked for: 1 = tsqr, 0 = domino. Read once, used by
    // create() (to charge the ledger) and by leaf_panel (to dispatch), so the plan is
    // costed for the carrier that actually runs.
    static int panelSelStatic() {
        static const int v = [] {
            const char* e = getenv("TQR_PANEL");
            if (e && !std::strcmp(e, "tsqr"))  return 1;
            if (e && !std::strcmp(e, "fused")) return 2;   // 10:S6.6 resident packet
            return 0;
        }(); return v;
    }

    // Which rung-1 panel ALGORITHM is live. The per-cell line used to print only
    // carrier=gmem|cluster, which names the BARRIER, so a TSQR arm that never
    // engaged still printed a plausible-looking carrier and read as a pass.
    // 31:640 wants the plan that ran to be readable off the record; this is the
    // field that distinguishes 10:S6.3's row-slab carrier from the distributed
    // per-column domino.
    const char* panel_algo() const {
        // One source for "which carrier was asked for" -- this used to carry its own
        // copy of the getenv that only knew about "tsqr", so a fused run reported
        // panel=domino while the fused kernel was in fact executing (job 9733).
        const int sel = panelSelStatic();
        if (!tsqrReady) return "domino";
        return sel == 2 ? "fused" : (sel == 1 ? "tsqr" : "domino");
    }

    Real *Ahi[3] = {nullptr,nullptr,nullptr}, *Alo[3] = {nullptr,nullptr,nullptr};
    Real *Bhi[3] = {nullptr,nullptr,nullptr}, *Blo[3] = {nullptr,nullptr,nullptr};
    size_t aloW[3] = {0,0,0}, bloW[3] = {0,0,0};
    int    splitRowsG[3] = {0,0,0};
    // 10:S8.3's telemetry invariant: F_EC,far / F_far == 1 for every cell with
    // nonzero far work. "No silent fallback of eligible far tiles to ordinary TF32
    // or fp32 is permitted in the 3xTF32 tier" -- so the ratio is COUNTED, and a
    // fallback would show up as a ratio below 1 rather than as nothing at all.
    double ecFarFlops = 0, farFlops = 0;

    // domino scalar/partial protocol (its own interface, not part of the ladder)
    Real *domScal = nullptr, *domSlot = nullptr, *domProw = nullptr;
    Real *domPp = nullptr, *domPsum = nullptr, *domT16 = nullptr;
    double *domSigD = nullptr, *domDslotD = nullptr;
    double *domDcolP = nullptr;   // A6: per-block column partials, 2 generations
    double *domDcolR = nullptr;   // A6: reduced [2][PW+1], written by the barrier
    Real  *QTf = nullptr;         // fused carrier: QT un-aliased from Pbuf
    int    fusedSmem = 0, fusedGrid = 0;   // 10:S6.6's resident execution packet
    unsigned *gbar = nullptr;
    // Yscr is GONE. k_panel_domino's masked-Y write is now guarded on a null SPV
    // (lrqr.cuh:2133/2599), so this scheduler -- which reads V straight out of
    // tril(A) -- allocates no m x b copy at all. That buffer was the single item
    // keeping the storage claim false: at 131072x16384 it measured 8x the whole
    // O(nb) auxiliary budget, because it is O(mb) and grows with aspect ratio.

    // -------------------------------------------------- domino geometry
    static constexpr int SMEM_CAP = 232448;
    static constexpr int roundDown32(int x) { return (x / 32) * 32; }
    static constexpr int domOverhead = 16 * (NB - 16) + 2 * 16 * 16 + 16 + NB + 8 + 3 * NB;
    static constexpr int clRpbMax = roundDown32((SMEM_CAP / (int)sizeof(Real) - domOverhead) / 16);
    // The smallest rows-per-block the panel kernel is known to execute: domRpb's
    // own base value. Anything below this is untested territory, not a tuning knob.
    static constexpr int kRpbFloor = 128;
    // Cluster geometry, mirroring lrqr.cuh:6521. A non-portable cluster on sm_90
    // is at most 16 CTAs, so a panel fits the cluster carrier iff its rows fit in
    // 16 slabs of at most clRpbMax each. That is a CAPACITY bound, which is what
    // 10:S4.3 permits to gate a dispatch -- unlike the old build's LRQR_CLMAX=4096,
    // a tunable constant that is exactly the kind of threshold S4.3 forbids.
    // Leaf slab height for the TSQR panel. Each slab factors independently -- no
    // cross-block sync for the whole b columns -- and the slab R states then merge
    // up an ordered stack-Householder tree of depth log2(nslab). That is 10:S6.3's
    // rung-1 structure verbatim, against the domino's NB sequential cross-block
    // reductions, and the difference is the measured ~6us x NB panel floor.
    // SLAB is the leaf slab height. Taller slabs shrink the O(m*b^2/SLAB) tree
    // buffers and shorten the merge tree -- but the leaf QR holds a SLAB x NB tile
    // in shared memory, so SLAB is capped by capacity, not by preference:
    //   bqr_smem(S) = ((S+1)*NB + (S+1)*16 + 2*16*NB + 2*256 + NB) * sizeof(Real)
    // must fit SMEM_CAP = 232448, giving S <= 329 at NB=64 and S <= 167 at NB=128.
    // SLAB=512 measured as `invalid argument` on the attribute call (349 KB), which
    // is the capacity constraint speaking.
    //
    // ---- SLAB is DERIVED, per (Real, NB). It must not be a written-down number ----
    //
    // It was 128: "the largest value legal at BOTH instantiated widths". One number
    // doing two jobs, and it cost the carrier its existence -- the tree state is
    // 4*N*b^2 with N = ceil(m/SLAB), so halving SLAB doubles it, and at 128 the
    // staging came to 1.7x the whole kappa*n*b ledger at 2048^2. tsqrReady was false
    // at every size and 10:S6.3's carrier never ran once.
    //
    // Replacing it with a per-width CONSTANT was the same mistake one level up: the
    // tile is (S+1)*(NB+16) + const WORDS, so its byte size scales with sizeof(Real),
    // and an fp64-derived constant gives the fp32/TF32/3xTF32 tiers less than half
    // their legal slab (256 against 592). Solve the capacity inequality instead:
    //
    //     bqr_smem(S) = ((S+1)*NB + (S+1)*16 + 2*16*NB + 2*16*16 + NB) * sizeof(Real)
    //                 = (S+1)*(NB+16)*sizeof(Real) + const
    //
    // for the largest S, rounded down to a multiple of the leaf QR's minipanel step
    // PW=16. Yields 272 / 128 for fp64 at NB=64/128 and 592 / 304 for fp32.
    //
    // The budget is the 196 KiB carveout, NOT the 227 KiB opt-in maximum: dynamic
    // smem above 200,704 B on sm_90a starves L1, which this kernel needs for the
    // global loads that stage each tile. That is a measured hardware bound.
    //
    // This is the CAP, not the choice. Two pressures oppose inside [16, slabCap()]:
    // a taller slab shrinks the 4*N*b^2 tree state and the merge depth, a shorter one
    // raises the leaf phase's CTA count, which is m/SLAB. Scoring that trade is A5's
    // row-slab partition -- SLAB *is* p_r -- and belongs to the planner (Stage 4).
    // Until then the cap is taken and the measurement decides whether the resulting
    // CTA count starves the leaf phase.
    static constexpr int kSmemCarveout = 200704;         // 196 KiB, sm_90a L1 cliff
    // SLAB must clear the carveout for EVERY kernel whose shared memory scales with it,
    // not just the leaf QR. This bounded bqr_smem alone and missed smLarft, which is
    // LARGER at the same slab:
    //     SLAB=272   bqr_smem 195,712   smLarft 205,824  -> fusedSmem 205,824  OVER
    //     SLAB=256   bqr_smem 185,472   smLarft 197,632  -> fusedSmem 197,632  fits
    // The fused packet takes the max over its phases, so those 5,120 B put it over the
    // cliff and pinned it at one block per SM. Measured there (job 9747, 16384^2, grid
    // 121 of 132 so parallelism was NOT the limit): occupancy 12.50%, barrier stall
    // 22.52 cycles/inst against the domino's 10.02, and 0.06 instructions issued per
    // cycle. Same class of defect as the fp64-derived SLAB constant earlier: a bound
    // written for one formula and applied to several.
    //   bqr_smem(S) = ((S+1)*(NB+16) + 2*16*NB + 2*256 + NB) * sizeof(Real)
    //   smLarft(S)  = ((S+1)*NB      + 2*NB*NB          + NB) * sizeof(Real)
    static constexpr int slabCapBqr() {
        return (((kSmemCarveout / (int)sizeof(Real)
                  - (2 * 16 * NB + 2 * 16 * 16 + NB)) / (NB + 16)) - 1);
    }
    static constexpr int slabCapLarft() {
        return (((kSmemCarveout / (int)sizeof(Real)
                  - (2 * NB * NB + NB)) / NB) - 1);
    }
    static constexpr int slabCap() {
        const int c = (slabCapBqr() < slabCapLarft() ? slabCapBqr() : slabCapLarft()) / 16 * 16;
        return c < 16 ? 16 : c;
    }
    // At fp64/NB=128 smLarft's FIXED term alone (2*NB^2 = 32,768 words = 262 KB) exceeds
    // the carveout, so no slab height makes the row-slab carrier fit there: its cap
    // comes out negative. That is a derived property of the width, not a whitelist --
    // tsqrReady gates on it below instead of on an enumerated (NB == 64 || NB == 128).
    static constexpr bool tsqrSmemOk() {
        return slabCapBqr() >= 16 && slabCapLarft() >= 16;
    }
    // ---- and SLAB wants to be SMALL, not large ----
    //
    // slabCap() is the capacity ceiling. It is not the choice, and taking it was wrong
    // for the carrier that matters. The leaf count N = ceil(m/SLAB) is simultaneously
    //   the fused packet's PARALLELISM  (its cooperative grid is N + nmerges ~= 2N)
    //   and its STORAGE                 (tree state 4*N*b^2, charged to 31:415)
    // so 31:415 bounds it from one side and the device from the other:
    //     mpad*b + 4*N*b^2 <= kappa*n*b   =>   for square m = n,  SLAB >= 2b
    // A LARGER slab shrinks storage and starves the grid; a smaller one does the
    // reverse. Taking the capacity ceiling (272) gave 15 blocks of 132 at 2048^2 --
    // 11% of the device -- and the fused packet measured 0.34x there against the
    // domino's 0.69x (job 9740). 2.5b clears the square bound with margin:
    //     grid   15 -> 25 (2048^2), 31 -> 51 (4096^2), 61 -> 103 (8192^2)
    //
    // Note the hard ceiling this exposes: grid <= 2N <= 3n/(2b), so at 2048^2 the
    // COMPLIANT carrier can never exceed 48 of 132 SMs without breaking 31:415. The
    // storage theorem caps the parallelism of the carrier the synchronization theorem
    // prescribes. That tension is a property of the spec, not of this implementation.
    // MEASURED: taking 2.5b instead (SLAB 272 -> 160, grid 15 -> 25 at 2048^2) made the
    // fused packet WORSE, 0.38 -> 0.19 at 4096^2 and 0.53 -> 0.43 at 8192^2, with
    // S_dev/(K+D) rising 8.75 -> 10.31 (job 9741 vs 9740). More leaves means a deeper
    // merge tree, and one extra level of serial 64-reflector passes costs more than the
    // leaf parallelism buys. Depth beats width here, so the capacity ceiling is also
    // the right choice -- for the reason opposite to the one first written down.
    static constexpr int SLAB = slabCap();
    static_assert(SLAB >= 16, "no legal leaf slab height at this (Real, NB)");
    static_assert(bqr_smem<Real, NB>(SLAB) <= kSmemCarveout, "leaf QR tile over budget");

    static constexpr int TPB  = 256;
    // ---- block size for the two single-CTA QR kernels, DERIVED ----
    //
    // k_leaf_geqrf and k_merge_geqrf are 67% of the TSQR panel and both are one CTA
    // per node -- at the merge tree's tail, one CTA is the entire GPU. ncu (job 9709)
    // measured them at 12.5% occupancy with `Block Limit Shared Mem = 1`: a single
    // 256-thread block per SM, 8 warps, 0.12 instructions issued per cycle, estimated
    // 87.5% headroom. When only one block can be resident, threads-per-block IS the
    // latency hiding, and their shared-memory footprint does not depend on it.
    //
    // Bound it by what the kernels can actually use rather than a written-down number:
    // the reflector is k-sliced over `nth`, and the W stage groups threads as
    // (PW rows) x (nth/PW columns), so nth beyond PW*NB buys nothing on the widest
    // stage. Cap at the hardware maximum block and keep it a multiple of PW.
    //   NB=64  -> min(1024, 16*64) = 1024   (32 warps, 50% occupancy at 1 block/SM)
    //   NB=128 -> min(1024, 16*128) = 1024
    // The ALGORITHMIC ceiling: the W stage groups threads (PW rows) x (nth/PW columns),
    // so nth beyond PW*NB buys nothing on the widest stage. Hardware caps it at 1024.
    static constexpr int TPB_QR_WANT =
        (16 * NB) < 1024 ? (16 * NB < 256 ? 256 : 16 * NB) : 1024;
    static_assert(TPB_QR_WANT % 16 == 0, "TPB_QR_WANT must be a multiple of PW");
    // ...but the REGISTER budget is the real bound and only the compiled kernel knows
    // it: at TPB_QR_WANT = 1024 both kernels returned "too many resources requested
    // for launch" (job 9712). Ask them, do not guess -- cudaFuncGetAttributes reports
    // the largest block their register footprint admits. Resolved in create().
    int tpbQR = TPB;
    static int smLarft(bool merge) {
        const int vr = merge ? NB : SLAB;
        return ((vr + 1) * NB + 2 * NB * NB + NB) * (int)sizeof(Real);
    }

    static constexpr int kClusterMax = 16;
    static constexpr int clRowMax = kClusterMax * clRpbMax;
    static int clRpb(int mk) { return 32 * ceil_div(ceil_div(mk, kClusterMax), 32); }
    static int domSmem(int rpb) {
        return (rpb * 16 + 16 * (NB - 16) + 2 * 16 * 16 + 16 + NB + 8 + 3 * NB) * (int)sizeof(Real);
    }
    int domRpb(int mk) const {
        int rpb = 128;
        if (ceil_div(mk, rpb) > smCount) rpb = 32 * ceil_div(ceil_div(mk, smCount), 32);
        return std::min(rpb, (int)clRpbMax);
    }

    // ---- the panel launch geometry, derived ONCE ----
    //
    // leaf_panel launches this grid; create() sizes domPp -- the cross-block boundary
    // partials, indexed Pp[(bid*NB + t)*PW + lane] -- from it. domPp used to be sized
    // max(132, ceil_div(m, clRpbMax)) * NB * PW instead: a literal whole device, when
    // the launcher never asks for that many blocks. Measured, that single literal was
    // the ENTIRE protocol overage -- 1.03 x nb at 2048^2 and 2.06 x nb at 8192x1024 --
    // and the only reason compact=0 survived the workspace-ledger fix.
    //
    // TQR_PR / TQR_LIVENESS are read here rather than at the call site so the sizing
    // and the launch see the same arms; an A/B that changed the grid but not the
    // buffer would overrun it.
    static bool prFromPlan() {
        static const bool v = [] {
            const char* e = getenv("TQR_PR"); return !e || std::strcmp(e, "full") != 0;
        }(); return v;
    }
    static bool liveFloor() {
        static const bool v = [] {
            const char* e = getenv("TQR_LIVENESS"); return !e || atoi(e) != 0;
        }(); return v;
    }
    // 10:S6.3's row-slab carrier: per-leaf V/R/T and per-merge-node V/T. Depends only
    // on the geometry, never on the nu-ladder, so the planner can subtract it up front
    // (LedgerCfg::carrierWords) instead of discovering afterwards that no room is left.
    static size_t tsqrStagingWords(int m_, int kmax_, int b_) {
        const size_t N    = (size_t)std::max(1, ceil_div(m_, SLAB));
        const size_t mpad = N * SLAB;
        const size_t npan = (size_t)ceil_div(kmax_, b_) + 1;
        return mpad * NB                 // Pbuf  (QT aliases it)
             + 4 * N * NB * NB           // QS + Vm + TmM + TmL
             + 2 * N * NB                // tauL + taum
             + npan * NB                 // Sg
             + (size_t)NB * NB;          // Rroot
    }

    void panelGrid(int mk, int& rpb, int& nbl, bool* overflowed = nullptr) const {
        rpb = domRpb(mk);
        if (prFromPlan() && panelPr > 0) {
            const int want = ceil_div(mk, panelPr);
            rpb = std::min((int)clRpbMax, std::max(kRpbFloor, roundDown32(want)));
            if (rpb < kRpbFloor) rpb = kRpbFloor;
        }
        if (liveFloor() && ceil_div(mk, rpb) < 2 && mk >= 2 * kRpbFloor) {
            const int half = roundDown32(mk / 2);
            if (half >= kRpbFloor && ceil_div(mk, half) >= 2) rpb = half;
        }
        nbl = ceil_div(mk, rpb);
        // More slabs than SMs: fall back to the widest resident slab and one wave.
        // The cluster carrier is not legal in that regime, hence the out-param.
        const bool ovf = (nbl > smCount);
        if (ovf) { rpb = clRpbMax; nbl = smCount; }
        if (overflowed) *overflowed = ovf;
    }

    // ---- A3's resident-state budget for THIS leaf width, in ONE place ----
    //
    // 10:S6.2: b is "selected from the kernel's ADMISSIBLE register/shared-memory
    // set", NOT from a sqrt(M) formula. This is that admissibility test, and it is
    // exactly A3: does the panel kernel's widest slab fit shared memory at this NB.
    //
    // It lives here rather than at the call sites because it HAD two call sites and
    // they drifted. create() excluded NB>128; predict_That -- which is what --autob
    // ranks b on -- passed the raw domSmem(clRpbMax). So a width was inadmissible
    // when factorizing and admissible when SELECTING: at 131072x16384 b-select saw
    // That(256)=1.00864 against That(128)=1.00902, a 0.04% "win", shipped b=256/p_r=1
    // through the unchecked FALLBACK, and the cell went 1.24x -> 0.97x.
    //
    // NB=256 is now excluded by A7 at rung 1 instead (nesting forces nu_1 >= b = 256
    // against nu_1 <= 161), which is the honest mechanism -- this function used to
    // return SMEM_CAP+1 for NB>128, i.e. it lied about a footprint in order to force
    // inadmissibility. Two facts about NB=256 are kept here for whoever revisits it:
    // it FITS shared memory (231,616 <= 232,448), and the kernel does not compute it
    // correctly (job 9675, 8 of 19 dqrt01 cells; the discriminator is panel HEIGHT,
    // not block count -- 512^2 and 1024^2 failed at p_r=1 while 4096x1024 and
    // 8192x512 passed at p_r=1). Neither matters while t_panel stays super-linear
    // in b (2.30x for 2x, 5.89x for 4x, job 9685).
    static int panelSmemBytes() { return domSmem((int)clRpbMax); }

    // ================================================================= setup
    double tpanelSec = 0.0;          // measured, set by the caller before create()
    const double* tpanelPrTab = nullptr;   // optional t_panel at each candidate p_r

    void create(int m_, int n_, size_t lda_, cudaStream_t stream) {
        m = m_; n = n_; lda = lda_; kmax = std::min(m, n);
        st = stream;
        CUBLAS_CHECK(cublasCreate(&cb));
        CUBLAS_CHECK(cublasSetStream(cb, st));
        // ---- 10:S6.7's priority discipline, expressed ----
        //   1 dependency-ready frontier tasks   -> st     (highest)
        //   2 combine tasks unblocking frontier -> sNear  (middle)
        //   3 far-update tiles                  -> sDef   (lowest)
        // Stream priority is work-conserving by construction: low-priority far
        // work runs on every SM the frontier is not using, which is exactly what
        // S6.7 asks for and is NOT the fixed reservation it forbids.
        //
        // TQR_PRIO=0 is the control arm.
        int prLo = 0, prHi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLo, &prHi));   // hi is numerically LOWER
        const char* pe = getenv("TQR_PRIO");
        const bool prioOn = (!pe || atoi(pe) != 0);
        const int pFrontier = prioOn ? prHi : 0;
        const int pNear     = prioOn ? (prHi + prLo) / 2 : 0;
        const int pFar      = prioOn ? prLo : 0;
        CUDA_CHECK(cudaStreamCreateWithPriority(&sDef, cudaStreamNonBlocking, pFar));
        CUBLAS_CHECK(cublasCreate(&cbDef));
        CUBLAS_CHECK(cublasSetStream(cbDef, sDef));
        CUDA_CHECK(cudaEventCreateWithFlags(&evEpoch, cudaEventDisableTiming));
        CUDA_CHECK(cudaStreamCreateWithPriority(&sNear, cudaStreamNonBlocking, pNear));
        (void)pFrontier;   // st is supplied by the caller; see note in the test harness
        CUBLAS_CHECK(cublasCreate(&cbNear));
        CUBLAS_CHECK(cublasSetStream(cbNear, sNear));
        CUDA_CHECK(cudaEventCreateWithFlags(&evFront, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evNearLast, cudaEventDisableTiming));
        for (int r = 0; r < 2; ++r)
            for (int g = 0; g < 2; ++g)
                CUDA_CHECK(cudaEventCreateWithFlags(&evT[r][g], cudaEventDisableTiming));
        for (int g = 0; g < 2; ++g)
            CUDA_CHECK(cudaEventCreateWithFlags(&evDef[g], cudaEventDisableTiming));
        // TQR_PIPE=0 is the control arm: one stream, no deferred generation.
        { const char* e = getenv("TQR_PIPE"); pipeOn = (!e || atoi(e) != 0); }
        tier = tier_from_env();
        if (std::is_same<Real, double>::value) tier = Tier::FP64;
        { const char* e = getenv("TQR_WPATH");
          wpathTrmm = (e && std::strcmp(e, "trmm") == 0);
          wpathWide = (e && std::strcmp(e, "wide") == 0); }

        cudaDeviceProp prop{}; int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        smCount = prop.multiProcessorCount;
        CUDA_CHECK(cudaDeviceGetAttribute(&smemOptin,
                   cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        // SMEM_CAP and everything derived from it at COMPILE time -- clRpbMax, SLAB --
        // assume this device provides that much opt-in shared memory. The planner
        // takes min() below and so degrades gracefully, but the compile-time constants
        // cannot, and a device with less would surface as a bare `invalid argument`
        // from cudaFuncSetAttribute with nothing pointing at the cause. Say it here.
        if (smemOptin < SMEM_CAP) {
            std::fprintf(stderr,
                "[TQR-OMEGA] device %d offers %d B opt-in shared memory but this build's "
                "compile-time constants assume SMEM_CAP=%d (clRpbMax=%d, SLAB=%d were "
                "derived from it). Rebuild with SMEM_CAP <= %d.\n",
                dev, smemOptin, (int)SMEM_CAP, (int)clRpbMax, (int)SLAB, smemOptin);
            std::abort();
        }
        smemOptin = std::min(smemOptin, SMEM_CAP);

        probe_hierarchy(hier, (int)sizeof(Real), smemOptin, smCount, (size_t)prop.l2CacheSize);

        // ---- the plan: exhaustive argmin over the declared legal family ----
        const double peak = std::is_same<Real, double>::value ? 6.7e13 : 6.0e14;
        // t_panel at each candidate p_r. 10:S4.3 wants measured kernel-rate
        // tables; this is that table's one entry that actually discriminates.
        double tprTab[6] = {tpanelSec, tpanelSec, tpanelSec, tpanelSec, tpanelSec, tpanelSec};
        if (tpanelPrTab) for (int i = 0; i < 6; ++i) tprTab[i] = tpanelPrTab[i];
        // A3's resident-state budget for THIS instantiation: the widest slab the
        // panel kernel can hold at this NB. If it does not fit, no plan at this leaf
        // width is legal -- which is the correct, kernel-derived form of "b from the
        // admissible set" (10:S6.2), replacing the sqrt(M) cap that used to bind it.
        // The admissible-set rule lives in panelSmemBytes(), so that this call and
        // predict_That's cannot drift apart again (they did; see that comment).
        // The allocator's own configuration decides the ledger, so the planner must
        // see it: the trmm path drops W2, the wide path doubles W, and the 3xTF32
        // tier adds hi/lo split scratch. Enumerating against a different config than
        // create() then allocates under is exactly the drift ledger_words() exists
        // to prevent.
        LedgerCfg lcfg{wpathTrmm, wpathWide, tier == Tier::TF32X3, m};
        // The panel carrier is part of the plan, so its storage is part of the plan's
        // ledger. With TSQR selected the ladder must be planned around it; with the
        // domino it costs nothing here (its boundary partials are PROTOCOL).
        if (panelSelStatic() >= 1) {
            lcfg.carrierWords = (double)tsqrStagingWords(m, kmax, b);
        }
        if (!Planner::enumerate(hier, m, n, NB, peak, rec, tpanelSec, smCount, tprTab,
                                panelSmemBytes(), SMEM_CAP, lcfg)) {
            rec.plan_enumerated = false;
            // ---- the fallback is not a licence to run an ILLEGAL plan ----
            //
            // The degenerate nu = [b,b,b,b] plan is load-bearing at tiny n: at
            // n=256 the workspace ledger admits nothing (4*sum(nu^2) = 65536 against
            // kappa*n*b = 49152), and the factorization must still happen. Relaxing
            // that ledger is defensible -- it is a planner-side budget, and the run
            // reports plan_enumerated=0 and compact=0 so the record shows it.
            //
            // A7 is different in kind: nu^2 <= theta*M is a CAPACITY constraint, and
            // violating it does not merely overspend, it computes the wrong answer.
            // Measured: b=256 has no legal plan (A7 caps rung 1 at sqrt(M)=170 and
            // nesting forces nu1 >= b), the fallback ran it anyway, and the result
            // was wrong by 1e13 on 7 of 10 cells while every other gate stayed
            // green. A fallback that bypasses the check that would have stopped it
            // is not a fallback, it is a silent downgrade.
            // The fallback must refuse on the SAME predicate the planner now uses.
            // It used to refuse on A7-at-rung-1 (b^2 > theta*M_1), which is the very
            // constraint corrected above; keeping it would reject a b the planner
            // considers legal. The real, kernel-derived bound is whether the panel's
            // shared-memory footprint fits -- violate that and the launch fails or
            // computes garbage, which is what b=256 did as an unchecked fallback.
            if (domSmem((int)clRpbMax) > SMEM_CAP) {
                std::fprintf(stderr,
                    "[TQR-OMEGA] REFUSING to run: the panel kernel at b=%d needs %d "
                    "bytes of shared memory against a %d cap, so no legal plan exists "
                    "at this leaf width on this device (A3 resident-state budget).\n",
                    NB, domSmem((int)clRpbMax), (int)SMEM_CAP);
                std::abort();
            }
            // No legal ladder: fall back to the degenerate single-level plan,
            // which is still legal (q=1 everywhere) and still correct. Recorded,
            // never silent -- a plan the enumerator could not populate is exactly
            // the case 10:S12 says must not be trusted quietly.
            for (int i = 0; i < kRungs; ++i) {
                rec.levels[i].level = i; rec.levels[i].leaf_b = NB;
                rec.levels[i].epoch_nu = NB; rec.levels[i].eager_frontier = NB;
                rec.levels[i].replica_layers = 2; rec.levels[i].combine_fanin = 2;
                rec.levels[i].pipeline_depth = 2;
                Planner::widths(rec.levels[i], hier[i], m, n, 2);
                rec.levels[i].K = ceil_div(kmax, NB);
            }
        }
        // rung 1 owns the panel's row slabs (31:429 "panel row slabs"), so its
        // p_r is the panel's processor partition.
        panelPr = std::max(1, rec.panel_pr ? rec.panel_pr : rec.levels[1].p_rows);
        panelCluster = rec.panel_cluster;
        b   = rec.levels[0].epoch_nu;
        nu1 = rec.levels[1].epoch_nu;
        nu2 = rec.levels[2].epoch_nu;
        nu3 = rec.levels[3].epoch_nu;
        f3  = rec.levels[3].eager_frontier;   // planner-chosen, may exceed nu3
        // f3 comes from the enumeration now; the only floor is legality f >= nu.
        f3 = std::max(f3, nu3);

        // 31:640 item 2 asks for "resident bytes BY OBJECT/TYPE/lineage". Tallied
        // at the point of allocation, never re-derived from a formula: the
        // subtractive/formula version drifted from what was actually allocated
        // twice, once absorbing the streamed-W buffers into "protocol" and once
        // reporting a ledger that omitted them entirely. A claim about storage that
        // is computed from anything other than the allocations themselves is a
        // claim about the formula.
        enum Cat { LEAF_T, LEDGER, PROTOCOL };
        auto ralloc = [&](Real** p, size_t words, Cat cat) {
            CUDA_CHECK(cudaMalloc(p, words * sizeof(Real)));
            CUDA_CHECK(cudaMemsetAsync(*p, 0, words * sizeof(Real), st));
            rec.aux_words += words;
            if      (cat == LEAF_T)   rec.w_leafT    += words;
            else if (cat == LEDGER)   rec.w_ledger   += words;
            else                      rec.w_protocol += words;
        };
        const int npan = ceil_div(kmax, b) + 1;
        ralloc(&Tleaf, (size_t)npan * b * b, LEAF_T);                 // == n*b, the 31:415 ledger
        for (int g = 0; g < 2; ++g) {
            ralloc(&T1[g], (size_t)nu1 * nu1, LEDGER);
            ralloc(&T2[g], (size_t)nu2 * nu2, LEDGER);
            ralloc(&Y1[g], (size_t)nu1 * nu1, LEDGER);
            ralloc(&Y2[g], (size_t)nu2 * nu2, LEDGER);
        }
        ralloc(&Y0, (size_t)b * b, LEDGER);
        for (int g = 0; g < 2; ++g) ralloc(&Y3[g], (size_t)nu3 * nu3, LEDGER);
        for (int g = 0; g < 2; ++g) ralloc(&T3[g], (size_t)nu3 * nu3, LEDGER);
        // W is streamed, so its capacity -- not the caller's chunk -- is what bounds
        // every apply. larfb clamps its own chunk against this; without that the
        // Q-rebuild (whose natural chunk is all k columns) walks straight off the
        // end of a buffer sized for the far apply.
        // WcapWords is the per-generation BUDGET. 'gemm' splits it across two
        // buffers; 'trmm' spends it on one; 'wide' spends it TWICE (uncompliant).
        WcapWords = (size_t)std::max(nu3, b) * std::max(f3, b);
        WnearWords = (size_t)std::max(nu2, b) * std::max(rec.levels[2].eager_frontier, b);
        const size_t wf = wpathWide ? 1 : 2;             // divisor for each buffer
        for (int g = 0; g < 2; ++g) {
            ralloc(&Wbuf[g],  WcapWords / wf + 1, LEDGER);
            if (!wpathTrmm) ralloc(&W2buf[g], WcapWords / wf + 1, LEDGER);
        }
        ralloc(&Wbuf[2],  WnearWords / wf + 1, LEDGER);
        if (!wpathTrmm) ralloc(&W2buf[2], WnearWords / wf + 1, LEDGER);
        // The compound scratch is (j x w) with j < the PARENT width and w <= the
        // CHILD width, so its true bound is nu3*nu2, not nu3^2. Sizing it nu3^2 put
        // the storage ledger 12% over kappa*n*b at n=16384 for space never used.
        const size_t zw = (size_t)nu3 * std::max(nu2, b);
        ralloc(&Zb, zw, LEDGER);
        ralloc(&Mb, zw, LEDGER);
        if (tier == Tier::TF32X3) {
            // Chunked so the footprint is independent of m: a tf32x3 backend that
            // pre-split whole operands would need Theta(m*nu) and break 31:415.
            //
            // Sizing is over the ACTUAL extents each call writes, not over the
            // chunk that bounds the other operand. gemm3x_NN splits B as a K x N
            // block with K = w, which reaches nu and has nothing to do with the row
            // chunk; sizing it by the chunk overran 3x and produced nan.
            for (int g = 0; g < 3; ++g) {
                const int wmax  = (g == 2) ? std::max(nu2, b) : std::max(nu3, b);
                const int cmax  = (g == 2) ? std::max(nu2, b) : std::max(f3 / 2, b);
                const size_t bud = (g == 2) ? WnearWords : WcapWords;
                int sr = std::max(128, (int)(bud / (size_t)(2 * (wmax + cmax))));
                sr = std::min(sr, m);
                splitRowsG[g] = sr;
                aloW[g] = (size_t)sr * wmax;
                bloW[g] = (size_t)std::max(sr, wmax) * cmax;
                ralloc(&Ahi[g], aloW[g], LEDGER); ralloc(&Alo[g], aloW[g], LEDGER);
                ralloc(&Bhi[g], bloW[g], LEDGER); ralloc(&Blo[g], bloW[g], LEDGER);
            }
        }
        // domino protocol buffers -- interface of the leaf kernel, sizes copied
        // verbatim from the validated allocation so the kernel contract holds.
        ralloc(&domScal, (size_t)4 * NB, PROTOCOL);
        ralloc(&domSlot, 16 * 16, PROTOCOL);
        ralloc(&domProw, 2 * 16, PROTOCOL);
        // Max over EVERY panel, not the tallest one. nbl is NOT monotone in mk: rpb
        // comes from roundDown32(ceil_div(mk, p_r)), so a shorter panel can round down
        // to a smaller slab and want MORE blocks. Assuming monotonicity sized 16 at
        // mk=4032 while a later panel asked for 18, which the launch-time assert caught.
        // n/b panels, evaluated once at setup -- cheaper than being wrong.
        domPpBlocks = 0;
        for (int r0 = 0; r0 < kmax; r0 += b) {
            int rpb0 = 0, nbl0 = 0;
            panelGrid(m - r0, rpb0, nbl0);
            domPpBlocks = std::max(domPpBlocks, nbl0);
        }
        domPpBlocks = std::max(1, domPpBlocks);
        ralloc(&domPp,   (size_t)domPpBlocks * NB * 16, PROTOCOL);
        ralloc(&domPsum, (size_t)2 * NB * 16 * (NB / 16), PROTOCOL);
        ralloc(&domT16,  (size_t)2 * 16 * 16 * (NB / 16), PROTOCOL);
        CUDA_CHECK(cudaMalloc(&domSigD, NB * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&domDslotD, 16 * 16 * sizeof(double)));
        // A6 (31:246): the per-column dots and sigma are reduced in FIXED block order
        // instead of by fp64 atomicAdd, whose order is undefined -- job 9715 measured
        // 90-94% of the factored words differing bitwise run to run. Two generations
        // (column parity), one row of PW+1 slots per block. At 132 blocks that is
        // 2*132*17 = 4488 words, ~0.03 x nb at 2048^2.
        CUDA_CHECK(cudaMalloc(&domDcolP,
                   (size_t)2 * std::max(1, domPpBlocks) * 17 * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(domDcolP, 0,
                   (size_t)2 * std::max(1, domPpBlocks) * 17 * sizeof(double), st));
        CUDA_CHECK(cudaMalloc(&domDcolR, (size_t)2 * 17 * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(domDcolR, 0, (size_t)2 * 17 * sizeof(double), st));
        CUDA_CHECK(cudaMemsetAsync(domSigD, 0, NB * sizeof(double), st));
        CUDA_CHECK(cudaMemsetAsync(domDslotD, 0, 16 * 16 * sizeof(double), st));
        CUDA_CHECK(cudaMalloc(&gbar, 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaMemsetAsync(gbar, 0, 2 * sizeof(unsigned), st));
        rec.aux_words_nonconforming = 0;

        // ---------------- TSQR panel: buffers, merge tree, attributes ----------------
        // Built only when its staging fits the workspace ledger. That single test is
        // what confines the algorithm to the shapes where it pays; there is no size
        // threshold anywhere (10:S4.3).
        {
            NmaxLeaf = ceil_div(m, SLAB);
            mpadMax  = NmaxLeaf * SLAB;
            const int npan = ceil_div(kmax, b) + 1;
            // ---- two of the three O(m*b) staging buffers are gone ----
            //
            // This asked for THREE full m x b buffers -- Pbuf, QT, Yspv -- against an
            // O(nb) ledger, which is 60% of the request and why tsqrReady was false
            // everywhere (2048^2: 659,520 words against a 393,216 budget).
            // 31:415 is mn + O(nb); O(mb) staging is not in it.
            //
            //  Yspv  DELETED. It is the masked-Y side output, a second copy of the
            //        very vectors 31:415 puts in tril(A). k_writeYRV now guards its
            //        SPV store, so the compact path passes nullptr.
            //  QT    ALIASED onto Pbuf. d_leaf_apply_body loads its slab's V into
            //        shared memory and __syncthreads() BEFORE it writes, and every
            //        block owns its slab exclusively, so writing the thin Q over the
            //        V it just consumed is safe. The one thing that does not survive
            //        is the root R in rows 0..NB-1, which leaf 0 overwrites -- so it
            //        is saved first into Rroot, NB*NB words.
            //  Pbuf  KEPT for now. Removing it needs an mk bound threaded through
            //        four kernels that lrqr.cuh's own path also calls; Stage 2's
            //        fused packet removes it as a side effect of reading tril(A)
            //        directly. Measured consequence of keeping it: SLAB is forced to
            //        256 rather than 128, halving the leaf phase's CTA count.
            const size_t need   = tsqrStagingWords(m, kmax, b);   // same helper the
                                                                  // planner was charged
            tsqrNeedWords = need;      // so the readiness line states a real number
            const double budget = Planner::kKappa * (double)n * b;
            // enumerate() was charged carrierWords up front, so a plan exists only if
            // the ladder it picked leaves room for exactly this. Re-testing the sum
            // here is the belt to that braces -- and it is what fails when no legal
            // ladder was narrow enough, in which case the carrier is genuinely
            // unaffordable at this shape and the domino is the legal answer.
            tsqrReady = (panelSelStatic() >= 1)
                     && ((double)(need + rec.w_ledger) <= budget)
                     && tsqrSmemOk();
            if (tsqrReady) {
                ralloc(&Pbuf, (size_t)mpadMax * NB, LEDGER);
                QT = Pbuf;                       // aliased, see the ledger note above
                ralloc(&Rroot, (size_t)NB * NB, LEDGER);
                ralloc(&QS,   (size_t)NmaxLeaf * NB * NB, LEDGER);
                ralloc(&Vm,   (size_t)NmaxLeaf * NB * NB, LEDGER);
                ralloc(&TmM,  (size_t)NmaxLeaf * NB * NB, LEDGER);
                ralloc(&TmL,  (size_t)NmaxLeaf * NB * NB, LEDGER);
                ralloc(&tauL, (size_t)NmaxLeaf * NB, LEDGER);
                ralloc(&taum, (size_t)NmaxLeaf * NB, LEDGER);
                ralloc(&Sg,   (size_t)npan * NB, LEDGER);
                // Per-panel ordered merge tree. A6 (31:239) fixes the leaf order and
                // the tree walk; this is the same construction lrqr.cuh:6220 uses, so
                // the order is the validated one rather than a fresh derivation.
                std::vector<int2> nodes;
                ptab.resize(npan);
                for (int k2 = 0; k2 < npan; ++k2) {
                    const int N = std::max(1, ceil_div(m - k2 * b, SLAB));
                    ptab[k2].N = N; ptab[k2].nodesOff = (int)nodes.size(); ptab[k2].nlv = 0;
                    for (int step = 1; step < N && ptab[k2].nlv < 24; step *= 2) {
                        const int off = (int)nodes.size();
                        for (int i = 0; i + step < N; i += 2 * step) nodes.push_back({i, i + step});
                        ptab[k2].lvl[ptab[k2].nlv++] =
                            {off - ptab[k2].nodesOff, (int)nodes.size() - off};
                    }
                }
                CUDA_CHECK(cudaMalloc(&dNodes, std::max<size_t>(nodes.size(), 1) * sizeof(int2)));
                if (!nodes.empty())
                    CUDA_CHECK(cudaMemcpy(dNodes, nodes.data(), nodes.size() * sizeof(int2),
                                          cudaMemcpyHostToDevice));
                auto setsm = [](const void* f, int bytes) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
                };
                setsm((const void*)k_leaf_geqrf<Real, SLAB, NB>, bqr_smem<Real, NB>(SLAB));
                setsm((const void*)k_merge_geqrf<Real, SLAB, NB>, merge_smem<Real, NB>());
                setsm((const void*)k_larft<Real, SLAB, NB, false>, smLarft(false));
                setsm((const void*)k_larft<Real, SLAB, NB, true>,  smLarft(true));
                setsm((const void*)k_dsweep<Real, NB>, 3 * NB * NB * (int)sizeof(Real));
                setsm((const void*)k_leaf_apply<Real, SLAB, NB>,
                      ((SLAB + 1) * NB + 2 * NB * NB) * (int)sizeof(Real));
                setsm((const void*)k_lu_sign<Real, NB>, (NB * NB + NB) * (int)sizeof(Real));
                if (panelSelStatic() == 2) {
                    QTf = Pbuf;        // aliased, as the multi-kernel path does
                    // 10:S6.6: "long-lived cooperative packets where they improve
                    // measured wall time". k_panel_fused does leaf QR, the ordered
                    // merge tree, T factors, the thin-Q sweep and BDGJNS in ONE
                    // launch with grid.sync() replacing 25 kernel boundaries.
                    fusedSmem = std::max({bqr_smem<Real, NB>(SLAB), merge_smem<Real, NB>(),
                                          smLarft(false), smLarft(true),
                                          3 * NB * NB * (int)sizeof(Real),
                                          ((SLAB + 1) * NB + 2 * NB * NB) * (int)sizeof(Real),
                                          ((NB + 1) * NB + NB) * (int)sizeof(Real),
                                          2 * (NB + 1) * NB * (int)sizeof(Real)});
                    CUDA_CHECK(cudaFuncSetAttribute(
                        (const void*)k_panel_fused<Real, SLAB, NB>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, fusedSmem));
                    int perSM = 0;
                    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM,
                        k_panel_fused<Real, SLAB, NB>, TPB, fusedSmem));
                    fusedGrid = perSM * smCount;
                    cudaDeviceProp pr2{}; int dv = 0;
                    CUDA_CHECK(cudaGetDevice(&dv));
                    CUDA_CHECK(cudaGetDeviceProperties(&pr2, dv));
                    if (fusedGrid < 1 || !pr2.cooperativeLaunch) {
                        std::fprintf(stderr, "[TQR-OMEGA] TQR_PANEL=fused: no cooperative "
                            "launch on this device (grid=%d)\n", fusedGrid);
                        std::abort();
                    }
                }
                // Largest block the two single-CTA QR kernels' registers admit, capped
                // by the algorithmic ceiling and rounded to a warp. This is the
                // occupancy lever ncu priced at 87.5%: with Block Limit Shared Mem = 1,
                // threads-per-block is the only way to add resident warps.
                {
                    cudaFuncAttributes fa{};
                    CUDA_CHECK(cudaFuncGetAttributes(&fa,
                               (const void*)k_leaf_geqrf<Real, SLAB, NB>));
                    int cap = fa.maxThreadsPerBlock;
                    CUDA_CHECK(cudaFuncGetAttributes(&fa,
                               (const void*)k_merge_geqrf<Real, SLAB, NB>));
                    cap = std::min(cap, fa.maxThreadsPerBlock);
                    tpbQR = std::max(TPB, std::min(TPB_QR_WANT, (cap / 32) * 32));
                    tpbQR = (tpbQR / 16) * 16;          // the W stage needs nth % PW == 0
                }
            }
        }

        // ---- the two sides must agree, and now they are checked ----
        //
        // Everything above this line is the LEDGER category. ledger_words() predicted
        // it and enumerate() rejected plans on that prediction, so if the prediction
        // is wrong the plan that shipped was never actually vetted. Compare them.
        // Tolerance is the handful of "+1" slack words create() adds to each W buffer.
        {
            // If the carrier was charged to the plan but then refused (it did not fit
            // the ladder the planner could reach), the allocator legitimately placed
            // none of it. Subtract it rather than reporting a divergence -- the real
            // diagnostic for that case is leaf_panel's refusal, which prints the
            // budget arithmetic.
            LedgerCfg vcfg = lcfg;
            if (!tsqrReady) vcfg.carrierWords = 0;
            const double predicted = Planner::ledger_words(rec.levels, b, vcfg);
            const double actual    = (double)rec.w_ledger;
            if (std::abs(predicted - actual) > 8.0) {
                std::fprintf(stderr,
                    "[TQR-OMEGA] LEDGER DIVERGENCE m=%d n=%d b=%d: planner vetted "
                    "%.0f words, allocator placed %.0f (%.2f vs %.2f x nb). The plan "
                    "that shipped was admitted against the wrong number.\n",
                    m, n, b, predicted, actual,
                    predicted / ((double)n * b), actual / ((double)n * b));
                std::abort();
            }
        }

        // Raise the domino's dynamic-smem limit to the LARGEST rpb any plan can
        // select, not to the one domRpb happens to return. Now that p_r sizes the
        // grid, a smaller p_r means a taller slab and MORE dynamic smem: at
        // p_r = 16 and mk = 16384 the plan asks for rpb = 1024 and 143 KB, against
        // an attribute set for rpb = 128 and 28 KB, and the launch fails with a
        // bare "invalid argument". clRpbMax is defined so domSmem(clRpbMax) fits
        // SMEM_CAP by construction, so this is the tight legal bound, not a guess.
        for (const void* fp : {(const void*)k_panel_domino<Real, NB, 16, false, true>,
                               (const void*)k_panel_domino<Real, NB, 16, false, true, true>,
                               (const void*)k_panel_domino<Real, NB, 16, false, false>,
                               (const void*)k_panel_domino<Real, NB, 16, false, false, true>})
            CUDA_CHECK(cudaFuncSetAttribute(fp, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            domSmem((int)clRpbMax)));
        // The cluster variant, sized for the widest slab a legal cluster can hold,
        // plus the opt-in for clusters above the portable 8 (lrqr.cuh:6384).
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_domino<Real, NB, 16, true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   domSmem(clRpb(std::min(m, (int)clRowMax)))));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_domino<Real, NB, 16, true>,
                   cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        // ---- storage verdict (31:415, 31:640 item 2) ----
        // w_leafT / w_ledger / w_protocol were tallied at each allocation above.
        // Nothing is recomputed here, so the verdict cannot disagree with what was
        // actually allocated -- which is how this reported compact=1 while holding
        // an O(mb) buffer, and later compact=0 for buffers the ledger had simply
        // failed to include.
        const double nb = (double)n * b;
        rec.compact_global_storage =
            ((double)rec.w_leafT  <= 1.5 * nb) &&
            ((double)rec.w_ledger <= Planner::kKappa * nb) &&
            ((double)rec.w_protocol <= nb);
    }

    void destroy() {
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(Tleaf); f(Zb); f(Mb);
        for (int g = 0; g < 2; ++g) { f(T1[g]); f(T2[g]); f(T3[g]); f(Y1[g]); f(Y2[g]); f(Y3[g]); }
        f(Y0);
        for (int r = 0; r < 2; ++r)
            for (int g = 0; g < 2; ++g) if (evT[r][g]) cudaEventDestroy(evT[r][g]);
        for (int g = 0; g < 3; ++g) { f(Wbuf[g]); f(W2buf[g]); }
        if (evFront) cudaEventDestroy(evFront);
        if (evNearLast) cudaEventDestroy(evNearLast);
        if (sNear) cudaStreamDestroy(sNear);
        if (cbNear) cublasDestroy(cbNear);
        if (evEpoch) cudaEventDestroy(evEpoch);
        for (int g = 0; g < 2; ++g) if (evDef[g]) cudaEventDestroy(evDef[g]);
        if (sDef) cudaStreamDestroy(sDef);
        if (cbDef) cublasDestroy(cbDef);
        f(domScal); f(domSlot); f(domProw); f(domPp); f(domPsum); f(domT16);
        f(domSigD); f(domDslotD); f(domDcolP); f(domDcolR); f(gbar);
        // QT aliases Pbuf -- one allocation, so exactly one free.
        f(Pbuf); f(QS); f(Vm); f(TmM); f(TmL); f(tauL); f(taum); f(Sg); f(Rroot);
        if (dNodes) cudaFree(dNodes);
        Pbuf = QT = QTf = QS = Vm = TmM = TmL = tauL = taum = Sg = Rroot = nullptr;
        dNodes = nullptr;
        for (int g = 0; g < 3; ++g) { f(Ahi[g]); f(Alo[g]); f(Bhi[g]); f(Blo[g]); }
        if (cb) cublasDestroy(cb);
        Tleaf = Zb = Mb = nullptr;
        for (int g = 0; g < 2; ++g) { T1[g] = T2[g] = T3[g] = Y1[g] = Y2[g] = Y3[g] = nullptr; }
        Y0 = nullptr;
        for (int r = 0; r < 2; ++r) for (int g = 0; g < 2; ++g) evT[r][g] = nullptr;
        for (int g = 0; g < 3; ++g) { Wbuf[g] = nullptr; W2buf[g] = nullptr; }
        cb = cbDef = cbNear = nullptr; sDef = sNear = nullptr;
        evEpoch = evFront = nullptr;
        evDef[0] = evDef[1] = nullptr; evNearLast = nullptr;
    }

    // ==================================================== the leaf panel (rung 0/1)
    //
    // k_panel_domino factors A[r0:m, r0:r0+b) in place: v scaled into tril with
    // beta on the diagonal (exact LAPACK storage, which is what lets every apply
    // above read V straight out of A), plus a b x b T.
    //
    // Its nslab row slabs reducing through the grid barrier ARE rung 1's Split /
    // Replicate / Combine (31:429: "panel row slabs", "compact R,T,W in cluster
    // lanes", "stack-HH R, deterministic W"), and its NB/PW minipanel chain is
    // rung 1's Pipeline. c_1 >= 2 is therefore satisfied by geometry, not by a
    // separate replication layer -- and unlike a second row split, its output is
    // a single WY that the rungs above can collapse.
    // ============ rung 1 as 10:S6.3 prescribes it: TSQR + BDGJNS ============
    //
    //   "Panel row slabs produce local R states. R states merge by ordered
    //    Householder-on-stack in DSM."
    //
    // Each slab factors independently -- NO cross-block dependency across the b
    // columns -- and the slab R states then merge up an ordered tree of depth
    // log2(nslab). k_panel_domino instead takes a cross-block reduction on EVERY one
    // of the NB columns: 64 sequential dependencies where this structure has 4, and
    // that difference is the ~6us x NB flat floor measured at 89-99% of wall.
    //
    // TSQR alone would not compose: its factor is Q_merge . diag(Q_slab), not a
    // single WY, and the ladder above needs a single collapsed WY. BDGJNS
    // reconstruction (sign LU + two triangular solves) recovers the standard
    // Householder (V, T), which is why this is usable here at all.
    //
    // Sequence copied from the validated lrqr.cuh:7861-7925 rather than re-derived;
    // in particular the merge-tree walk order is the one A6 (31:239) fixes.
    // ---- 10:S6.6's resident execution packet: the whole panel in one launch ----
    //
    // "Reduce launch overhead by the schedule: larger resident execution packets ...
    // persistent work queues or long-lived cooperative packets where they improve
    // measured wall time." k_panel_fused (lrqr.cuh:1326) is exactly that -- leaf QR,
    // the ordered stack-Householder merge tree, T factors, the thin-Q down-sweep and
    // BDGJNS reconstruction, with grid.sync() in place of 25 kernel boundaries and the
    // tree state never leaving L2.
    //
    // It has NEVER EXECUTED. lrqr.cuh:6417 records it as "currently inert -- the fused
    // path is unreachable while useDomino is true". So correctness is the open question
    // here, not speed, and the gate runs before any timing is read.
    void leaf_panel_fused(Real* A, int cb_col) {
        const int r0 = cb_col, mk = m - r0;
        const int k  = cb_col / b;
        const PTab& t = ptab[std::min<size_t>(k, ptab.size() - 1)];
        const int N = std::max(1, ceil_div(mk, SLAB));

        PanelArgs<Real> pa;
        pa.A = A + (size_t)cb_col * lda + r0; pa.lda = lda;
        pa.row0 = r0; pa.mk = mk; pa.mpad = N * SLAB; pa.N = N;
        pa.Pbuf = Pbuf; pa.QT = QTf; pa.QS = QS;
        pa.Vm = Vm; pa.TmM = TmM; pa.TmL = TmL; pa.tauL = tauL; pa.taum = taum;
        pa.nodes = dNodes + t.nodesOff;
        pa.nlv = t.nlv;
        for (int l = 0; l < t.nlv && l < 20; ++l) pa.lvl[l] = t.lvl[l];
        pa.Sg = Sg + (size_t)k * NB;
        pa.Tpan = Tleaf + (size_t)k * b * b;
        pa.SPV = nullptr; pa.ldspv = 0; pa.spvcol = 0;
        pa.Rroot = Rroot;      // QT aliases Pbuf, so R must be saved first

        void* args[] = { &pa };
        // ---- SIZE THE GRID TO THE WORK, not to the device ----
        //
        // This was `fusedGrid - 24` -- essentially the whole GPU -- and measured 2-4.4x
        // slower than the domino (job 9733) while being correct. The cause is shape,
        // not launches: phase 1 has N leaves (8 at 2048^2) and the merge tree tapers to
        // one node, but a cooperative grid is sized once and EVERY block must enter
        // EVERY grid.sync(). ~100 blocks were synchronising to do 8 blocks of work.
        //
        // N + nmerges is the exact width the kernel can use: it is also the condition
        // d_panel_body tests for `leafInWave` (nmergesAll + N <= nbl), which lets the
        // leaves ride the wavefront as level -1 and removes a whole phase plus its
        // barrier. So sizing to the work does not merely stop wasting blocks, it
        // selects the kernel's better schedule.
        //
        // The blocks not taken here are not idle: 10:S6.7 wants the far update to have
        // every SM the frontier is not using, and a narrower cooperative grid leaves
        // more of them free -- the opposite of the headroom subtraction it replaces.
        const int nmg = t.nlv ? (t.lvl[t.nlv - 1].x + t.lvl[t.nlv - 1].y) : 0;
        int g = std::min(fusedGrid, std::max(1, N + nmg));
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)k_panel_fused<Real, SLAB, NB>, dim3(g), dim3(TPB), args,
            (size_t)fusedSmem, st));
        {
            cudaError_t le = cudaGetLastError();
            // TQR_FUSEDBG syncs after every panel so an EXECUTION fault is attributed
            // to the panel that caused it. Without it the first symptom is a cuBLAS
            // INTERNAL_ERROR several calls later, which says nothing.
            static const bool dbg = getenv("TQR_FUSEDBG") != nullptr;
            if (le == cudaSuccess && dbg) le = cudaStreamSynchronize(st);
            if (le != cudaSuccess) {
                std::fprintf(stderr, "[TQR-OMEGA] fused panel failed at col=%d mk=%d "
                             "N=%d nlv=%d nmg=%d grid=%d smem=%d mpad=%d : %s\n",
                             cb_col, mk, N, t.nlv, nmg, g, fusedSmem, pa.mpad,
                             cudaGetErrorString(le));
                std::abort();
            }
        }
        // one launch, so the rung-1 device rounds are the tree depth, not 89
        rec.split[1]     += N;
        rec.replicate[1] += N;
        rec.combine[1]   += nmg;
        rec.pipeline[1]  += std::max(0, t.nlv - 1);
        rec.rounds[0]    += 1;
        rec.dev_rounds[1] += 1 + 2LL * t.nlv;
        rec.words[1]     += (long long)N * NB * NB;
        rec.messages[1]  += nmg;
        rec.merges[1]    += nmg;
    }

    void leaf_panel_tsqr(Real* A, int cb_col) {
        const int r0 = cb_col, mk = m - r0;
        const int k  = cb_col / b;
        Real* Acol = A + (size_t)cb_col * lda + r0;
        const PTab& t = ptab[std::min<size_t>(k, ptab.size() - 1)];
        const int N = std::max(1, ceil_div(mk, SLAB));
        const int mpad = N * SLAB;

        k_load_panel<Real><<<ceil_div(mpad * NB, TPB), TPB, 0, st>>>(
            Acol, lda, mk, NB, Pbuf, mpad, mpad);
        k_leaf_geqrf<Real, SLAB, NB><<<N, tpbQR, bqr_smem<Real, NB>(SLAB), st>>>(
            Pbuf, mpad, tauL, N);
        k_larft<Real, SLAB, NB, false><<<N, TPB, smLarft(false), st>>>(
            Pbuf, mpad, nullptr, tauL, TmL, N, 0);
        // ordered stack-Householder merge, bottom-up, fixed leaf order (A6)
        for (int l = 0; l < t.nlv; ++l) {
            const int base = t.lvl[l].x, cnt = t.lvl[l].y;
            if (cnt <= 0) continue;
            k_merge_geqrf<Real, SLAB, NB><<<cnt, tpbQR, merge_smem<Real, NB>(), st>>>(
                Pbuf, mpad, dNodes + t.nodesOff + base, cnt, base, Vm, taum);
            rec.merges[1] += cnt;
        }
        const int nmerges = t.nlv ? (t.lvl[t.nlv - 1].x + t.lvl[t.nlv - 1].y) : 0;
        if (nmerges > 0)
            k_larft<Real, SLAB, NB, true><<<nmerges, TPB, smLarft(true), st>>>(
                nullptr, 0, Vm, taum, TmM, nmerges, 0);
        // ---- save the root R before the thin-Q overwrites it ----
        //
        // QT aliases Pbuf, so k_leaf_apply's write of leaf 0's thin-Q lands on rows
        // 0..SLAB-1 of Pbuf -- which is where the merge tree left the panel's R.
        // k_writeYRV needs that R (it emits Sg[r]*R for r<=c), so it is copied out
        // first. NB x NB words, against the m x b buffer the alias saves.
        CUDA_CHECK(cudaMemcpy2DAsync(Rroot, NB * sizeof(Real),
                                     Pbuf,  (size_t)mpad * sizeof(Real),
                                     NB * sizeof(Real), NB,
                                     cudaMemcpyDeviceToDevice, st));
        // thin-Q down-sweep, then leaf apply
        k_set_identity<Real, NB><<<ceil_div(NB * NB, TPB), TPB, 0, st>>>(QS);
        for (int l = t.nlv - 1; l >= 0; --l) {
            const int base = t.lvl[l].x, cnt = t.lvl[l].y;
            if (cnt <= 0) continue;
            k_dsweep<Real, NB><<<cnt, TPB, 3 * NB * NB * sizeof(Real), st>>>(
                QS, TmM, Vm, dNodes + t.nodesOff + base, cnt, base);
        }
        k_leaf_apply<Real, SLAB, NB><<<N, TPB,
            ((SLAB + 1) * NB + 2 * NB * NB) * sizeof(Real), st>>>(
            Pbuf, mpad, TmL, QS, QT, mpad, N);
        // ---- BDGJNS reconstruction: standard Householder (V, T) out of thin Q ----
        Real* Sk = Sg + (size_t)k * NB;
        Real* Tk = Tleaf + (size_t)k * b * b;
        k_lu_sign<Real, NB><<<1, TPB, (NB * NB + NB) * sizeof(Real), st>>>(QT, mpad, Sk);
        CUBLAS_CHECK(cublasSetStream(cb, st));
        if (mpad > NB)
            trsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                 CUBLAS_DIAG_NON_UNIT, mpad - NB, NB, Real(1), QT, mpad, QT + NB, mpad);
        k_negUS<Real, NB><<<ceil_div(NB * NB, TPB), TPB, 0, st>>>(QT, mpad, Sk, Tk);
        trsm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
             CUBLAS_DIAG_UNIT, NB, NB, Real(1), QT, mpad, Tk, NB);
        // k_writeYRV adds row0 ITSELF (`Ac = A + c*lda + row0`), so it takes the
        // COLUMN base, not the panel corner. Passing Acol -- which already carries
        // +r0 -- double-counts the row offset and writes the panel at row 2*r0.
        // It surfaced only on wide cells because those are the only ones where the
        // ledger admitted this carrier at all; every square silently ran the domino
        // and passed, which is why "16/19 PASS" was not the reassurance it looked.
        // R from Rroot (ld = NB) rather than Pbuf, which the thin-Q has overwritten;
        // SPV = nullptr because V goes to tril(A) per 31:415 and k_writeYRV guards it.
        k_writeYRV<Real, NB><<<ceil_div(mk * NB, TPB), TPB, 0, st>>>(
            QT, mpad, Rroot, NB, Sk,
            A + (size_t)cb_col * lda, lda, r0, mk, (Real*)nullptr, 0, 0);
        {
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess) {
                std::fprintf(stderr, "[TQR-OMEGA] TSQR panel failed at col=%d mk=%d N=%d "
                             "mpad=%d : %s\n", cb_col, mk, N, mpad, cudaGetErrorString(le));
                std::abort();
            }
        }
        // ---- four-move witness, rungs 0 and 1, from this structure ----
        rec.split[0]     += NB / 16;
        rec.replicate[0] += 2;
        rec.combine[0]   += (long long)N * (NB / 16);
        rec.pipeline[0]  += NB / 16 - 1;
        rec.split[1]     += N;                 // panel row slabs (31:429)
        rec.replicate[1] += N;                 // each slab's own local R state
        rec.combine[1]   += nmerges;           // ordered stack-Householder merges
        rec.pipeline[1]  += std::max(0, t.nlv - 1);
        rec.rounds[0]    += 1;
        // Device-wide rounds for 10:S6.3's carrier: one after the leaf phase (which
        // itself has NONE -- each slab factors its own rows with __syncthreads and
        // warp shuffles), then one per merge level up and one per down-sweep level,
        // plus the reconstruction's global steps. O(log N) per panel against the
        // domino's 2 + (NB/PW)(PW+4) + 1 + 2(NB/PW - 1).
        rec.dev_rounds[1] += 3 + 2LL * t.nlv + 5;
        rec.words[1]     += (long long)N * NB * NB;
        rec.messages[1]  += nmerges;
    }

    void leaf_panel(Real* A, int cb_col) {
        // TQR_PANEL selects the rung-1 panel algorithm: tsqr (10:S6.3's structure)
        // or domino (distributed Householder). Forced for now so the A/B isolates it
        // on one device; it becomes a probed plan choice once measured.
        // ONE selector. This was a second copy of the getenv that knew only "tsqr"
        // and "domino", so TQR_PANEL=fused dispatched to the DOMINO while panel_algo()
        // -- which reads panelSelStatic() -- printed "fused". Every fused measurement
        // taken before this was the domino wearing the wrong label: the third
        // one-rule-two-copies defect this session, after panelSmemBytes() and
        // ledger_words(). The rule now exists exactly once.
        const int panelSel = panelSelStatic();
        // ---- a requested carrier that cannot run must ABORT, never fall through ----
        //
        // This line used to read `if (panelSel == 1 && tsqrReady)` and otherwise fell
        // silently into the domino below. tsqrReady was false at EVERY size (its
        // staging asked 1.7x-13x the whole kappa*n*b ledger), so TQR_PANEL=tsqr has
        // never once run TSQR -- and two verdicts were issued on that basis: the
        // roadmap's "TSQR is not faster (-0.5% to -14.6%)" and job 9687's phase
        // split, whose tsqr profile came back byte-identical to the domino's
        // (k_panel_domino 49.2%, zero k_leaf_geqrf instances).
        //
        // An arm that silently runs the other arm is worse than no arm. Refuse
        // loudly, with the numbers that made it illegal, exactly as create()'s
        // fallback refuses a plan that violates a physical constraint.
        if (panelSel >= 1 && !tsqrReady) {
            std::fprintf(stderr,
                "[TQR-OMEGA] TQR_PANEL=tsqr requested but the rung-1 TSQR carrier is "
                "NOT storage-legal for m=%d n=%d b=%d SLAB=%d: needs %zu words against "
                "kappa*n*b = %.0f (ledger already holds %.0f). Refusing to run the "
                "domino under a tsqr label.\n",
                m, n, b, SLAB, tsqrNeedWords,
                Planner::kKappa * (double)n * b, (double)rec.w_ledger);
            std::abort();
        }
        if (panelSel == 2) { leaf_panel_fused(A, cb_col); return; }
        if (panelSel == 1) { leaf_panel_tsqr(A, cb_col); return; }
        const int r0 = cb_col, mk = m - r0;
        Real* Acol = A + (size_t)cb_col * lda + r0;
        const int k = cb_col / b;

        // ---- A5's processor partition, finally reaching a launch ----
        //
        // Planner::enumerate chooses p_r under p_r*p_c*c <= P_lambda and this code
        // used to ignore it entirely: the grid came from domRpb(mk), which targets
        // ~132 blocks, i.e. the WHOLE device. So every panel evicted the deferred
        // far generation from the GPU for its whole duration.
        //
        // 10:S9.1 is explicit that this is where the time goes -- the exposed
        // non-far cost is the inter-task gap sum INSIDE the far-stream span,
        // 3.7-16.3% of wall, and "scheduling acts on duty". A prior measurement on
        // the old build put a number on it: the panel took ~48/132 SMs and the far
        // update got 84/132 = 0.68, which was the entire tf32 C_far gap.
        //
        // This is NOT the fixed panel-SM reservation 10:S6.7 and S9.3 forbid: p_r
        // is enumerated per cell by the planner against measured panel cost, so
        // the split is a model decision, and the far stream still takes every SM
        // the panel is not using.
        //
        // TQR_PR=full restores the old whole-device grid; TQR_LIVENESS=0 drops the
        // rung-1 two-lane floor. Both are read inside panelGrid(), which is also what
        // create() sized domPp against -- one derivation, so an A/B arm cannot move
        // the grid out from under the buffer.
        int rpb = 0, nbl = 0; bool overflow = false;
        panelGrid(mk, rpb, nbl, &overflow);
        // domPp is indexed by block id, so a grid wider than the one it was sized for
        // is a silent out-of-bounds write. It cannot happen while both come from
        // panelGrid(), and this says so if it ever does.
        if (nbl > domPpBlocks) {
            std::fprintf(stderr, "[TQR-OMEGA] panel grid %d blocks exceeds the %d "
                         "domPp was sized for (mk=%d)\n", nbl, domPpBlocks, mk);
            std::abort();
        }

        DomArgs<Real> da;
        da.A = Acol; da.lda = lda; da.mk = mk; da.row0 = r0; da.rpb = rpb;
        da.SPV = nullptr; da.ldspv = m; da.spvcol = 0;   // V stays in tril(A); no copy
        da.Tpan = Tleaf + (size_t)k * b * b;
        da.tauT = domScal; da.fv = domScal + NB; da.bet = domScal + 2 * NB;
        da.sig = domScal + 3 * NB;
        da.dslot = domSlot; da.prow = domProw; da.Pp = domPp; da.Psum = domPsum;
        da.sigD = domSigD; da.dslotD = domDslotD;
        da.dcolP = domDcolP; da.dcolStride = std::max(1, domPpBlocks);
        da.T16g = domT16; da.tphase = nullptr; da.nslab = nbl;

        // ================= rung 1's PRESCRIBED carrier =================
        //
        // 10:S6.3, verbatim: "Shared -> DSM rung: use thread-block clusters as the
        // explicit peer domain ... Cluster barriers are used only for true cluster
        // dependencies." The panel's row-slab reduction IS rung 1's Combine, so it
        // is exactly such a dependency -- and running it on a global-memory barrier
        // is falling off the carrier the spec names.
        //
        // It is also the whole floor. k_panel_domino runs ~7 grid barriers per
        // minipanel step, ~58 per panel at b=128, and gsync() in the NOCOOP variant
        // is gmem_barrier: __threadfence + atomicAdd + spin, through L2. Measured
        // 389us per panel / 58 = 6.7us each. The CL variant's gsync() is
        // cg::this_cluster().sync(), measured in this codebase at ~0.23us against
        // ~1.0us for a cooperative grid.sync (lrqr.cuh:7757) -- cluster members
        // share a DSMEM address space over an SM-to-SM NoC inside the GPC, with no
        // global round-trip. In cluster mode the domino's scalar protocol (sigma,
        // dots, pivot row) moves to DSMEM too: ~30ns against ~300ns.
        //
        // ENGAGEMENT IS CAPACITY LEGALITY, NOT A SIZE THRESHOLD. The cluster runs
        // when the panel's rows fit 16 slabs that each fit shared memory. The old
        // build gated this on LRQR_CLMAX=4096, a tunable constant; 10:S4.3 forbids
        // exactly that ("must not use hand-written thresholds such as if (n<=4096)")
        // and clRowMax is the real bound.
        static const bool clusterOn = [] {
            const char* e = getenv("TQR_CLUSTER"); return !e || atoi(e) != 0;
        }();
        bool launched = false;
        if (clusterOn && panelCluster && !overflow && mk <= clRowMax) {
            // Width from the plan, capped by the hardware cluster limit.
            const int cw   = std::min(std::max(1, panelPr), (int)kClusterMax);
            int crpb = std::max(kRpbFloor, roundDown32(ceil_div(mk, cw)));
            crpb = std::min(crpb, (int)clRpbMax);
            const int cnbl = ceil_div(mk, crpb);
            if (cnbl >= 1 && cnbl <= kClusterMax && domSmem(crpb) <= SMEM_CAP) {
                DomArgs<Real> cda = da;
                cda.rpb = crpb; cda.nslab = cnbl;
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(cnbl); cfg.blockDim = dim3(512);
                cfg.dynamicSmemBytes = (size_t)domSmem(crpb);
                cfg.stream = st;
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim = {(unsigned)cnbl, 1u, 1u};
                cfg.attrs = at; cfg.numAttrs = 1;
                const cudaError_t err = cudaLaunchKernelEx(
                    &cfg, k_panel_domino<Real, NB, 16, true>, cda, (unsigned*)nullptr);
                if (err == cudaSuccess) { launched = true; nbl = cnbl; rpb = crpb; }
                else (void)cudaGetLastError();   // cluster did not fit -> fall through
            }
        }
        static const bool cooperativeSync = [] {
            const char* mode = getenv("TQR_PANEL_SYNC");
            if (mode && std::strcmp(mode, "cooperative") && std::strcmp(mode, "legacy")) {
                std::fprintf(stderr, "TQR_PANEL_SYNC expects legacy|cooperative\n"); std::abort();
            }
            return mode && !std::strcmp(mode, "cooperative");
        }();
        if (!launched && cooperativeSync) {
            const void* fn = overflow
                ? (const void*)k_panel_domino<Real, NB, 16, false, false, true>
                : (const void*)k_panel_domino<Real, NB, 16, false, false>;
            int blocksPerSm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSm, fn, 512, domSmem(rpb)));
            if (nbl > blocksPerSm * smCount) {
                std::fprintf(stderr, "cooperative panel does not fit resident grid\n"); std::abort();
            }
            unsigned* unused = nullptr;
            void* args[] = {&da, &unused};
            CUDA_CHECK(cudaLaunchCooperativeKernel(fn, dim3(nbl), dim3(512), args, domSmem(rpb), st));
            launched = true;
        }
        if (!launched) {
            if (overflow)
                k_panel_domino<Real, NB, 16, false, true, true>
                    <<<nbl, 512, domSmem(rpb), st>>>(da, gbar);
            else
                k_panel_domino<Real, NB, 16, false, true>
                    <<<nbl, 512, domSmem(rpb), st>>>(da, gbar);
        }
        // A bad panel launch otherwise surfaces later as an opaque cuBLAS
        // EXECUTION_FAILED from whichever GEMM happens to run next, which points
        // at the wrong line entirely.
        {
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess) {
                std::fprintf(stderr, "[TQR-OMEGA] panel launch failed at col=%d "
                             "mk=%d rpb=%d nbl=%d smem=%d : %s\n",
                             cb_col, mk, rpb, nbl, domSmem(rpb), cudaGetErrorString(le));
                std::abort();
            }
        }

        // ---- four-move witness, rungs 0 and 1, from the launch geometry ----
        rec.split[0]     += NB / 16;              // PW-wide minipanel fragments
        rec.replicate[0] += 2;                    // msm0/msm1 ping-pong generations
        rec.combine[0]   += (long long)nbl * (NB / 16);   // warp/block partial reductions
        rec.pipeline[0]  += NB / 16 - 1;          // minipanel stages overlapped
        rec.split[1]     += nbl;                  // panel row slabs
        rec.replicate[1] += nbl;                  // each slab holds its own compact partial
        rec.combine[1]   += (long long)(nbl - 1) * (NB / 16);  // ordered barrier reduction
        rec.merges[1]    += nbl - 1;
        rec.pipeline[1]  += NB / 16 - 1;          // cluster wavefront over minipanels
        rec.rounds[0]    += 1;   // one X_{k,.} node per leaf epoch (31:329)
        // ---- the rounds the hardware actually pays, counted from the kernel ----
        //
        // k_panel_domino's gsync() sites, at niter == 1 (lrqr.cuh:1925..2096):
        //     init                                    2
        //     per minipanel, NB/PW of them:
        //         the jj column loop                  PW   <-- ONE BARRIER PER COLUMN
        //         after b1 / B2a / b2b / b3           4
        //     T-build tail                            1 + 2*(NB/PW - 1)
        // At NB=64, PW=16 that is 89 device-wide rounds for a 64-column panel.
        //
        // This is the Theta(b) term that makes S_1 = Theta(K_1 * b) instead of
        // 31:371's Theta(K_1 + D_1). It is not an artefact of counting: each of
        // those barriers is dependency-bearing -- the next column's reflector cannot
        // be formed until every row slab has published its partial norm and dot.
        // 10:S6.3's carrier removes them by construction rather than making them
        // cheaper, which is why the carrier and not the barrier is the lever.
        {
            // Kept in step with k_panel_domino's actual gsync() sites -- 31:640: a
            // counter that no longer describes the kernel is worse than no counter.
            //   init                                    2
            //   per minipanel (NB/PW of them):
            //     the jj column loop                    PW   <- one per column
            //     after b1 / B2a / b3                   3    (b2b's broadcast barrier
            //                                                 removed: it is replicated)
            //   T assembly                              0    (replicated into block 0)
            // NB=64, PW=16: 2 + 4*19 = 78, down from 89.
            constexpr int PWk = 16;
            constexpr int mp  = NB / PWk;
            rec.dev_rounds[1] += 2 + (long long)mp * (PWk + 3);
        }
        // words crossing shared->DSM: each slab publishes NB x PW partials per minipanel
        rec.words[1]     += (long long)nbl * NB * 16 * (NB / 16);
        rec.messages[1]  += (long long)(nbl - 1) * (NB / 16);
    }

    // ============================================ collapsed wide WY: the T compound
    //
    //  T = [ T1   -T1 (V1^T V2) T2 ]      (31:405: a width-nu compact-WY block has
    //      [  0          T2      ]         a triangular block of Theta(nu^2))
    //
    //  V1^T V2 is formed straight out of tril(A). V2's leading j rows are zero, so
    //      V1^T V2 = B1^T (I + strictlower(D)) + B2^T C2
    //  and (I + strictlower(D)) is exactly what TRMM(RIGHT,LOWER,N,UNIT) applies --
    //  so the unit-diagonal masking costs a transpose-copy and nothing else, and no
    //  materialised copy of V is needed anywhere in this file.
    //
    //  A6 (31:239, ordered reduction) is preserved by construction: children are
    //  compounded in increasing column order and the composition is never reordered.
    void compound(Real* A, Real* T, int ldT, int cE, int joff, int wchild,
                  const Real* Tchild, int ldTc) {
        // T[joff:joff+wchild, joff:joff+wchild] := Tchild
        CUDA_CHECK(cudaMemcpy2DAsync(T + (size_t)joff * ldT + joff, (size_t)ldT * sizeof(Real),
                                     Tchild, (size_t)ldTc * sizeof(Real),
                                     wchild * sizeof(Real), wchild,
                                     cudaMemcpyDeviceToDevice, st));
        if (joff == 0) return;                     // first child: nothing to couple

        const int j = joff, w = wchild;
        const int rE = cE;                          // epoch row origin == its first column
        const int mr = m - rE;
        Real* B1 = A + (size_t)cE * lda + (rE + j);            // w x j
        Real* D  = A + (size_t)(cE + j) * lda + (rE + j);      // w x w, unit lower
        Real* B2 = A + (size_t)cE * lda + (rE + j + w);        // (mr-j-w) x j
        Real* C2 = A + (size_t)(cE + j) * lda + (rE + j + w);

        // Z := B1^T                       (j x w)
        geam(cb, CUBLAS_OP_T, CUBLAS_OP_N, j, w, Real(1), B1, (int)lda,
             Real(0), Zb, j, Zb, j);
        // Z := Z * (I + strictlower(D))
        trmm(cb, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
             CUBLAS_DIAG_UNIT, j, w, Real(1), D, (int)lda, Zb, j);
        // Z += B2^T * C2
        const int tail = mr - j - w;
        if (tail > 0)
            gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, j, w, tail, Real(1),
                 B2, (int)lda, C2, (int)lda, Real(1), Zb, j);
        // M := T1 * Z        (T's leading j x j block; zeros below the diagonal
        //                     are real zeros, so a plain GEMM is the triangular op)
        gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, j, w, j, Real(1),
             T, ldT, Zb, j, Real(0), Mb, j);
        // T[0:j, j:j+w] := -M * Tchild
        gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, j, w, w, Real(-1),
             Mb, j, Tchild, ldTc, Real(0), T + (size_t)j * ldT, ldT);

        rec.combine[2] += 1;                        // ordered transform composition
    }

    // ================================================ the commit: dlarfb from tril(A)
    //
    //  C := (I - V T^T V^T) C  over C = A[rE:m, c0:c0+nc), V = A[rE:m, cE:cE+w).
    //  V1 (top w x w) is unit lower triangular BY CONSTRUCTION: panel p's column
    //  jl sits at epoch row p*b+jl, so the epoch's leading w x w block of V is
    //  exactly unit lower and its upper triangle holds R, which TRMM(LOWER,UNIT)
    //  never reads. That is why "V overwrites tril(A)" (31:403) is implementable
    //  with no masked copy, and it is what makes the storage claim true.
    //
    //  W is streamed over f columns at a time: live W storage Theta(nu*f), the
    //  quantity 31:405/31:409 charge, rather than a materialised nu x N object.
    //
    //  TRANS selects the factor: false applies Q = I - V T V^T (used to rebuild Q
    //  for the orthogonality gate), true applies Q^T = I - V T^T V^T (the
    //  factorization's own trailing update). Same five BLAS3 calls either way.
    void larfb(const Real* Vm, size_t ldv, int rE, int cE, int w,
               const Real* T, int ldT, bool trans,
               Real* C, size_t ldc, int c0, int nc, int fchunk, int rung, int gen = 0,
               const Real* Yd = nullptr) {
        if (nc <= 0 || w <= 0) return;
        cublasHandle_t cb = (gen == 0) ? this->cb : (gen == 1 ? this->cbDef : this->cbNear);
        Real* Wbuf  = this->Wbuf[gen];
        Real* W2buf = this->W2buf[gen];
        const int mr = m - rE, tail = mr - w;
        const Real* V1 = Vm + (size_t)cE * ldv + rE;
        const Real* V2 = Vm + (size_t)cE * ldv + (rE + w);
        // Theta(nu * f) live W storage is the quantity 31:405 charges, so the
        // stream width is bounded by the buffer, never by the request.
        // The chunk is bounded by ONE buffer's capacity, which depends on the arm.
        const size_t cap = ((gen == 2) ? WnearWords : WcapWords) / (wpathWide ? 1 : 2);
        long long chunkL = std::min<long long>(fchunk, (long long)(cap / (size_t)std::max(1, w)));
        if (tier == Tier::TF32X3) {
            // The three-product split scratch is streamed state too, so it bounds
            // the stream width exactly as W does. Missing this let a NARROW-w
            // commit (rung 0/1, w = b) take chunk = cap/w = 1152 columns while Blo
            // was sized for the f3/2 = 192 that only the WIDE rung-3 commits reach
            // -- a 6x overrun. The bound must be taken over every caller's w, not
            // over the widest one.
            const long long bcap = (long long)bloW[gen] /
                                   std::max<long long>(1, std::max(w, splitRowsG[gen]));
            chunkL = std::min(chunkL, bcap);
        }
        const int chunk = (int)std::max<long long>(1, chunkL);
        for (int cc = c0; cc < c0 + nc; cc += chunk) {
            const int ncc = std::min(chunk, c0 + nc - cc);
            Real* C1 = C + (size_t)cc * ldc + rE;
            Real* C2 = C + (size_t)cc * ldc + (rE + w);
            // W := V1^T C1
            if (Yd) {
                gemm(cb, false, CUBLAS_OP_T, CUBLAS_OP_N, w, ncc, w, Real(1),
                     Yd, w, C1, (int)ldc, Real(0), Wbuf, w);
            } else {
                geam(cb, CUBLAS_OP_N, CUBLAS_OP_N, w, ncc, Real(1), C1, (int)ldc,
                     Real(0), Wbuf, w, Wbuf, w);
                trmm(cb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                     CUBLAS_DIAG_UNIT, w, ncc, Real(1), V1, (int)ldv, Wbuf, w);
            }
            // W += V2^T C2   -- K = tail, the large trailing dimension: BULK
            if (tail > 0) {
                if (tier == Tier::TF32X3)
                    gemm3x_TN(gen, cb, (gen == 0) ? st : (gen == 1 ? sDef : sNear),
                              w, ncc, tail, Real(1), V2, (int)ldv, C2, (int)ldc,
                              Real(1), Wbuf, w);
                else
                    gemm(cb, bulk_tf32(), CUBLAS_OP_T, CUBLAS_OP_N, w, ncc, tail,
                         Real(1), V2, (int)ldv, C2, (int)ldc, Real(1), Wbuf, w);
            }
            // W2 := T^(T) W, as a GEMM into a SECOND buffer.
            //
            // T is triangular, so a TRMM computes this in half the flops -- and
            // measured 2-11% SLOWER on every cell (8192^2 lost 11%, 16384^2 9.5%),
            // because cuBLAS TRMM at these shapes is far less efficient than GEMM
            // and the in-place aliasing costs it further. Fewer flops, more time.
            //
            // The storage bound is kept regardless, because the bound is the spec
            // feature and the TRMM was only one way to meet it. 31:409 charges
            // d(nu^2 + nu*f) where nu*f is TOTAL live streamed-W storage per
            // generation, not one buffer of that shape: two buffers of nu x (f/2)
            // satisfy it exactly. So W and W2 are each half-width and the stream
            // chunk halves with them -- same live storage, same total work, GEMM
            // efficiency retained.
            // TQR_WPATH=trmm selects the in-place triangular form -- same storage
            // bound, half the flops, and measured SLOWER on every cell. Kept as a
            // switchable arm so the comparison can be re-run in ONE job on ONE
            // device: the first time it was measured across two jobs, Slurm handed
            // out a different GPU and cuSOLVER (unchanged) moved 20-25%, which made
            // both arms uncomparable in either direction.
            Real* Wt = W2buf;
            if (wpathTrmm) {
                trmm(cb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                     trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                     w, ncc, Real(1), T, ldT, Wbuf, w);
                Wt = Wbuf;
            } else {
                gemm(cb, false, trans ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                     w, ncc, w, Real(1), T, ldT, Wbuf, w, Real(0), W2buf, w);
            }
            // C2 -= V2 Wt   -- M = tail, the large trailing dimension: BULK.
            // (before the C1 leg, which may consume Wt in place)
            if (tail > 0) {
                if (tier == Tier::TF32X3)
                    gemm3x_NN(gen, cb, (gen == 0) ? st : (gen == 1 ? sDef : sNear),
                              tail, ncc, w, Real(-1), V2, (int)ldv, Wt, w,
                              Real(1), C2, (int)ldc);
                else
                    gemm(cb, bulk_tf32(), CUBLAS_OP_N, CUBLAS_OP_N, tail, ncc, w,
                         Real(-1), V2, (int)ldv, Wt, w, Real(1), C2, (int)ldc);
            }
            // C1 -= V1 Wt
            if (Yd) {
                gemm(cb, false, CUBLAS_OP_N, CUBLAS_OP_N, w, ncc, w, Real(-1),
                     Yd, w, Wt, w, Real(1), C1, (int)ldc);
            } else {
                W2buf = Wt;
                geam(cb, CUBLAS_OP_N, CUBLAS_OP_N, w, ncc, Real(1), W2buf, w,
                     Real(0), W2buf, w, Wbuf, w);
                trmm(cb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                     CUBLAS_DIAG_UNIT, w, ncc, Real(1), V1, (int)ldv, Wbuf, w);
                geam(cb, CUBLAS_OP_N, CUBLAS_OP_N, w, ncc, Real(1), C1, (int)ldc,
                     Real(-1), Wbuf, w, C1, (int)ldc);
            }
            if (rung >= 0) {
                rec.split[rung]    += 1;
                rec.pipeline[rung] += 1;
                rec.words[rung]    += 2LL * mr * ncc + (long long)w * ncc;
                rec.messages[rung] += 1;
            }
        }
        if (rung >= 0) {
            rec.combine[rung]   += 1;
            rec.replicate[rung] += rec.levels[rung].replica_layers;
        }
        // S_lambda is NOT counted here. 10:S2.5 defines it as "dependency-bearing
        // synchronization rounds ON THE CRITICAL FRONTIER", and the theorem counts
        // X_{k,j} as ONE node -- "child state ready, stage-j ordered combine
        // complete, and all stage-j transformations applied" (31:329). A panel and
        // its trailing commit are two LAUNCHES inside one node, and a deferred
        // generation is by construction not on the frontier at all.
        //
        // Counting launches here made S_3/(K+D) read 1.85 purely because each
        // rung-3 epoch issues an eager and a deferred commit. That is the same
        // mistake as measuring K against n/NB instead of n/nu: an instrument
        // reporting a quantity the theorem does not define. Rounds are raised once
        // per epoch by the geqrf loop instead.
    }

    // A split that does not fit its buffer must stop, not scribble. The overrun
    // this guards against produced nan at 4096^2 while the routing invariant
    // F_EC/F_far still read a healthy 1.0000 -- a reminder that the invariant
    // proves routing and nothing else (10:S8.2).
    void require_split(int g, int ar, int ac, int br, int bc, const char* tagn) const {
        if ((size_t)ar * ac > aloW[g] || (size_t)br * bc > bloW[g]) {
            std::fprintf(stderr, "[TQR-OMEGA] 3xTF32 split scratch too small in %s "
                         "gen=%d: A %dx%d vs %zu, B %dx%d vs %zu (splitRows=%d)\n",
                         tagn, g, ar, ac, aloW[g], br, bc, bloW[g], splitRowsG[g]);
            std::abort();
        }
    }

    // ---- Ootomo-Yokota three products, K-chunked (10:S8.3) ----
    //
    //   C := beta*C + alpha * A^T B      (opA = T; K is the long dimension)
    // realised as, per K-slice,
    //   hi_A hi_B : plain TF32 GEMM on the ORIGINAL operands (no buffer)
    //   hi_A lo_B : TF32 GEMM of the original A against materialised lo_B
    //   lo_A hi_B : TF32 GEMM of materialised lo_A against the original B
    // accumulating in fp32 outside the tensor-core rounding path (beta=1 on the
    // fp32 output), and omitting only lo*lo as the paper prescribes.
    void gemm3x_TN(int gen, cublasHandle_t cbh, cudaStream_t stm, int M, int N, int K,
                   Real alpha, const Real* Amat, int lda_,
                   const Real* Bmat, int ldb_, Real beta, Real* C, int ldc_) {
        dim3 tb(32, 8);
        const int splitRows = splitRowsG[gen];
        Real *Ahi = this->Ahi[gen], *Alo = this->Alo[gen];
        Real *Bhi = this->Bhi[gen], *Blo = this->Blo[gen];
        require_split(gen, std::min(splitRows, K), M, std::min(splitRows, K), N, "TN");
        for (int k0 = 0; k0 < K; k0 += splitRows) {
            const int kc = std::min(splitRows, K - k0);
            const Real bta = (k0 == 0) ? beta : Real(1);
            const Real* Ak = Amat + k0;
            const Real* Bk = Bmat + k0;
            k_split_hilo2<Real><<<dim3((kc + 31) / 32, (std::max(M, N) + 7) / 8),
                                  tb, 0, stm>>>(
                Ak, lda_, Ahi, Alo, M, Bk, ldb_, Bhi, Blo, N, kc, kc);
            gemm(cbh, true, CUBLAS_OP_T, CUBLAS_OP_N, M, N, kc, alpha,
                 Ahi, kc, Bhi, kc, bta,      C, ldc_);          // hi_A hi_B
            gemm(cbh, true, CUBLAS_OP_T, CUBLAS_OP_N, M, N, kc, alpha,
                 Ahi, kc, Blo, kc, Real(1),  C, ldc_);          // hi_A lo_B
            gemm(cbh, true, CUBLAS_OP_T, CUBLAS_OP_N, M, N, kc, alpha,
                 Alo, kc, Bhi, kc, Real(1),  C, ldc_);          // lo_A hi_B
        }
        ecFarFlops += 3.0 * 2.0 * M * N * K;
        farFlops   += 2.0 * (double)M * N * K;
    }

    //   C := beta*C + alpha * A B        (opA = N; M is the long dimension)
    // Same three products, chunked on M instead of K because K = w is already
    // small; B is w x N and its lo part costs almost nothing.
    void gemm3x_NN(int gen, cublasHandle_t cbh, cudaStream_t stm, int M, int N, int K,
                   Real alpha, const Real* Amat, int lda_,
                   const Real* Bmat, int ldb_, Real beta, Real* C, int ldc_) {
        dim3 tb(32, 8);
        const int splitRows = splitRowsG[gen];
        Real *Ahi = this->Ahi[gen], *Alo = this->Alo[gen];
        Real *Bhi = this->Bhi[gen], *Blo = this->Blo[gen];
        require_split(gen, std::min(splitRows, M), K, K, N, "NN");
        k_split_hilo<Real><<<dim3((K + 31) / 32, (N + 7) / 8), tb, 0, stm>>>(
            Bmat, ldb_, Bhi, Blo, K, K, N);
        for (int m0 = 0; m0 < M; m0 += splitRows) {
            const int mc = std::min(splitRows, M - m0);
            const Real* Am = Amat + m0;
            Real* Cm = C + m0;
            k_split_hilo<Real><<<dim3((mc + 31) / 32, (K + 7) / 8), tb, 0, stm>>>(
                Am, lda_, Ahi, Alo, mc, mc, K);
            gemm(cbh, true, CUBLAS_OP_N, CUBLAS_OP_N, mc, N, K, alpha,
                 Ahi, mc, Bhi, K, beta,     Cm, ldc_);          // hi_A hi_B
            gemm(cbh, true, CUBLAS_OP_N, CUBLAS_OP_N, mc, N, K, alpha,
                 Ahi, mc, Blo, K, Real(1),  Cm, ldc_);          // hi_A lo_B
            gemm(cbh, true, CUBLAS_OP_N, CUBLAS_OP_N, mc, N, K, alpha,
                 Alo, mc, Bhi, K, Real(1),  Cm, ldc_);          // lo_A hi_B
        }
        ecFarFlops += 3.0 * 2.0 * M * N * K;
        farFlops   += 2.0 * (double)M * N * K;
    }

    void commit(Real* A, int rung, int cE, int w, const Real* T, int ldT,
                int c0, int nc, int fchunk, int gen = 0, const Real* Yd = nullptr) {
        larfb(A, lda, cE, cE, w, T, ldT, /*trans=*/true, A, lda, c0, nc, fchunk,
              rung, gen, Yd);
    }

    // Build the epoch's unit-lower V block once, before its commits stream over it.
    void build_mask(const Real* A, int cE, int w, Real* Y) {
        dim3 tb(16, 16), gr((w + 15) / 16, (w + 15) / 16);
        k_mask_unit_lower<Real><<<gr, tb, 0, st>>>(A + (size_t)cE * lda + cE, lda, Y, w);
    }

    // ---------------- NEAR look-ahead: rung-1/2 Pipeline, d = 2 ----------------
    //
    //  The serial spine of the schedule is the panel chain (10:S3.5: replication
    //  "does not permit two dependent panels to be committed out of order"), so
    //  the only thing worth overlapping with a panel is trailing work that panel
    //  does not read. Split each near commit at the NEXT child epoch's boundary:
    //
    //     eager  [c0, c0+eagerW)   the next child factors these -- cannot defer
    //     defer  [c0+eagerW, ...)  first read one child later -- runs on sNear,
    //                              concurrent with that next child's panel
    //
    //  Ordering: child j's eager commit writes columns that child j-1's deferred
    //  commit also writes, and ordered composition (31:120(c)) fixes their order,
    //  so the eager issue waits on the previous deferred. That wait is what makes
    //  it exactly two live generations -- d_lambda = 2 -- rather than a queue.
    //
    //  Child j's PANEL needs no wait: its columns were covered by child j-1's
    //  EAGER commit, which ran on the frontier stream and is stream-ordered ahead
    //  of it. That asymmetry is the whole win.
    void commit_lookahead(Real* A, int rung, int cE, int w, const Real* T, int ldT,
                          int c0, int nc, int childW, int layerPar, const Real* Yd) {
        if (nc <= 0) return;
        if (!pipeOn) { commit(A, rung, cE, w, T, ldT, c0, nc, nc, 0, Yd); return; }
        const int eager = std::min(nc, std::max(childW, 1));
        // ONE event, re-recorded, rather than a parity rotation. sNear is a single
        // FIFO stream, so the most recent deferred completion dominates every
        // earlier one and a single wait is exactly the whole dependency.
        //
        // The parity scheme this replaces was wrong, and instructively so: two
        // rungs shared one 2-slot rotation, so when their calls interleaved --
        // which they do, rung 1 inside the o1 loop and rung 2 around it -- slot^1
        // no longer named the generation being depended on. It measured as 11 of
        // 19 gate cells failing while buying no wall time at all. Parity is only
        // correct when one producer advances it.
        if (nearLive) {
            CUDA_CHECK(cudaStreamWaitEvent(st, evNearLast, 0));
            nearLive = false;
        }
        commit(A, rung, cE, w, T, ldT, c0, eager, eager, /*gen=*/0, Yd);
        const int rest = nc - eager;
        if (rest > 0) {
            CUDA_CHECK(cudaEventRecord(evFront, st));
            CUDA_CHECK(cudaStreamWaitEvent(sNear, evFront, 0));
            commit(A, rung, cE, w, T, ldT, c0 + eager, rest, rest, /*gen=*/2, Yd);
            CUDA_CHECK(cudaEventRecord(evNearLast, sNear));
            // The deferred generation is still reading rung-`rung` layer
            // `layerPar`; this is when that layer becomes rebuildable.
            CUDA_CHECK(cudaEventRecord(evT[rung - 1][layerPar], sNear));
            evTValid[rung - 1][layerPar] = true;
            nearLive = true;
            rec.pipeline[rung] += 1;
        }
    }

    // Every deferred near generation must land before anything reads those columns
    // outside the loop that issued it.
    // Take rung-r layer p for writing. Blocks only on the generation that last
    // READ this layer -- one generation back, never the most recent -- which is
    // what keeps two of them live instead of serialising.
    void claim_layer(int rung, int par) {
        if (evTValid[rung - 1][par]) {
            CUDA_CHECK(cudaStreamWaitEvent(st, evT[rung - 1][par], 0));
            evTValid[rung - 1][par] = false;
        }
        rec.replicate[rung] += 1;
    }

    void drain_near() {
        if (nearLive) {
            CUDA_CHECK(cudaStreamWaitEvent(st, evNearLast, 0));
            nearLive = false;
        }
    }

    // Rebuild the m x k orthogonal factor for the dqrt01 gates: Q := H_0 H_1 ... H_{p-1}
    // applied to I in DESCENDING panel order, which is the ordered composition of
    // 31:120(c) read backwards -- the same leaf transforms, never re-derived.
    // Apply Q (trans=false) or Q^T (trans=true) to an m x nc block, in the panel
    // order the ordered composition requires: Q = H_0 H_1 ... H_{p-1}, so applying
    // Q sweeps DESCENDING and Q^T sweeps ASCENDING. Used by orgqr and by the
    // sampled residual, which needs Q applied to a single vector without ever
    // forming it.
    void apply_Q(const Real* A, Real* C, size_t ldc, int nc, bool trans) {
        const int npan = ceil_div(kmax, b);
        if (trans) {
            for (int p = 0; p < npan; ++p) {
                const int cbc = p * b;
                larfb(A, lda, cbc, cbc, b, Tleaf + (size_t)p * b * b, b, /*trans=*/true,
                      C, ldc, 0, nc, nc, /*rung=*/-1);
            }
        } else {
            for (int p = npan - 1; p >= 0; --p) {
                const int cbc = p * b;
                larfb(A, lda, cbc, cbc, b, Tleaf + (size_t)p * b * b, b, /*trans=*/false,
                      C, ldc, 0, nc, nc, /*rung=*/-1);
            }
        }
    }

    void orgqr(const Real* A, Real* Q, size_t ldq, int kcols) {
        const int npan = ceil_div(kmax, b);
        for (int p = npan - 1; p >= 0; --p) {
            const int cbc = p * b;
            if (cbc >= kcols) continue;
            larfb(A, lda, cbc, cbc, b, Tleaf + (size_t)p * b * b, b, /*trans=*/false,
                  Q, ldq, 0, kcols, kcols, /*rung=*/-1);
        }
    }


    // =========================================================== the nu-ladder
    //
    //  for e3 : nu3 epochs                         K3 = ceil(n/nu3) rung-3 frontiers
    //    for e2 : nu2 epochs inside                K2                rung-2
    //      for e1 : nu1 epochs inside              K1                rung-1
    //        for p : leaf panels of width b        K0                rung-0
    //          factor; compound into T1; commit to the rest of the nu1 epoch
    //        commit T1 to the rest of the nu2 epoch;  compound into T2
    //      commit T2 to the rest of the nu3 epoch;    compound into T3
    //    commit T3 to the far region [c3+w3, n)
    //
    //  Update words are m*n*(n/nu3 + nu3/nu2 + nu2/nu1 + nu1/b), minimised by the
    //  geometric ladder. Two levels give m*n*(n/B + B/b): the measured 16x.
    void geqrf(Real* A) {
        CUBLAS_CHECK(cublasSetStream(cb, st));
        CUBLAS_CHECK(cublasSetStream(cbDef, sDef));
        defPending[0] = defPending[1] = false;
        nearLive = false;
        for (int r = 0; r < 2; ++r) evTValid[r][0] = evTValid[r][1] = false;
        // ---- observed counters are PER FACTORIZATION ----
        // They were cumulative, so a harness that calls geqrf twice (time it, then
        // re-run it for the gate because cuSOLVER overwrote A) reported S_obs and
        // Q_obs at exactly 2x. Caught by comparing the same cell across two jobs:
        // rung-0 S_obs 374 -> 748 with no scheduling change between them. The move
        // counts survive that as liveness evidence, but the two RATIOS the whole
        // plan is judged on do not, so they reset here rather than being divided by
        // a call count afterwards -- a correction factor is what hid it.
        for (int i = 0; i < kRungs; ++i) {
            rec.split[i] = rec.replicate[i] = rec.combine[i] = rec.pipeline[i] = 0;
            rec.words[i] = rec.messages[i] = rec.rounds[i] = rec.merges[i] = 0;
            rec.dev_rounds[i] = 0;
        }
        rec.gram_calls = 0; rec.stale_reads = 0;
        ecFarFlops = farFlops = 0;
        int e3 = 0;
        for (int c3 = 0; c3 < kmax; c3 += nu3) {
            const int w3 = std::min(nu3, kmax - c3);
            // The replica layer this epoch owns. Epoch e-1's deferred far is still
            // reading layer (e-1)&1 on sDef; this epoch writes the other one.
            const int tpar = e3 & 1;
            Real* T3c = T3[tpar];
            rec.rounds[3] += 1;                      // one X_{k,.} node per nu3 epoch
            // ...but epoch e-2 used THIS layer, so its deferred far must be done.
            // Two layers give exactly two live generations, which is d_3 = 2.
            if (defPending[tpar]) {
                CUDA_CHECK(cudaStreamWaitEvent(st, evDef[tpar], 0));
                defPending[tpar] = false;
            }
            // Zero only the LIVE w x w block, not the whole nu x nu allocation: the
            // apply reads T[0:w,0:w] and nothing else, and at nu3 = 832 fp64 the
            // full clear is 5.5 MB per epoch on the FRONTIER stream -- the one
            // stream whose latency is the critical path (10:S3.5).
            CUDA_CHECK(cudaMemset2DAsync(T3c, (size_t)nu3 * sizeof(Real), 0,
                                         (size_t)w3 * sizeof(Real), w3, st));
            rec.replicate[3] += 1;

            int e2 = 0;
            for (int o2 = 0; o2 < w3; o2 += nu2, ++e2) {
                const int w2 = std::min(nu2, w3 - o2), c2 = c3 + o2;
                const int p2 = e2 & 1; Real* T2c = T2[p2];
                rec.rounds[2] += 1;                  // one X_{k,.} node per nu2 epoch
                claim_layer(2, p2);
                // Zero only the LIVE w x w block, not the whole nu x nu allocation: the
            // apply reads T[0:w,0:w] and nothing else, and at nu3 = 832 fp64 the
            // full clear is 5.5 MB per epoch on the FRONTIER stream -- the one
            // stream whose latency is the critical path (10:S3.5).
            CUDA_CHECK(cudaMemset2DAsync(T2c, (size_t)nu2 * sizeof(Real), 0,
                                         (size_t)w2 * sizeof(Real), w2, st));

                int e1 = 0;
                for (int o1 = 0; o1 < w2; o1 += nu1, ++e1) {
                    const int w1 = std::min(nu1, w2 - o1), c1 = c2 + o1;
                    const int p1 = e1 & 1; Real* T1c = T1[p1];
                    rec.rounds[1] += 1;              // one X_{k,.} node per nu1 epoch
                    claim_layer(1, p1);
                    // Zero only the LIVE w x w block, not the whole nu x nu allocation: the
            // apply reads T[0:w,0:w] and nothing else, and at nu3 = 832 fp64 the
            // full clear is 5.5 MB per epoch on the FRONTIER stream -- the one
            // stream whose latency is the critical path (10:S3.5).
            CUDA_CHECK(cudaMemset2DAsync(T1c, (size_t)nu1 * sizeof(Real), 0,
                                         (size_t)w1 * sizeof(Real), w1, st));

                    for (int ob = 0; ob < w1; ob += b) {
                        const int wb = std::min(b, w1 - ob), cbc = c1 + ob;
                        leaf_panel(A, cbc);
                        compound(A, T1c, nu1, c1, ob, wb, Tleaf + (size_t)(cbc / b) * b * b, b);
                        const int rest = (c1 + w1) - (cbc + wb);
                        if (rest > 0) {
                            build_mask(A, cbc, wb, Y0);
                            commit(A, 0, cbc, wb, Tleaf + (size_t)(cbc / b) * b * b, b,
                                   cbc + wb, rest, rest, 0, Y0);
                        }
                    }
                    // ---- rung-1 frontier: ONE collapsed width-nu1 commit ----
                    const int rest2 = (c2 + w2) - (c1 + w1);
                    if (rest2 > 0) build_mask(A, c1, w1, Y1[p1]);
                    commit_lookahead(A, 1, c1, w1, T1c, nu1, c1 + w1, rest2, nu1, p1, Y1[p1]);
                    compound(A, T2c, nu2, c2, o1, w1, T1c, nu1);
                }
                // The rung-2 compound reads V over the whole nu2 block, whose tail
                // columns the deferred near generation writes.
                drain_near();
                // ---- rung-2 frontier ----
                const int rest3 = (c3 + w3) - (c2 + w2);
                if (rest3 > 0) build_mask(A, c2, w2, Y2[p2]);
                commit_lookahead(A, 2, c2, w2, T2c, nu2, c2 + w2, rest3, nu2, p2, Y2[p2]);
                compound(A, T3c, nu3, c3, o2, w2, T2c, nu2);
            }
            drain_near();      // same, one level up: the rung-3 compound reads it all
            // ================= rung-3 frontier, in TWO generations =================
            //
            //  31:429 gives rung 3's Pipeline carrier as "streamed batched WY
            //  applies", and 10:S4.4 requires d_3 >= 2 generations live. The split
            //  is forced by the dependency structure, not chosen:
            //
            //    EAGER  [c3+w3, c3+2*w3)  -- epoch e+1's OWN columns. It factors
            //                                them next, so this cannot be deferred.
            //    DEFER  [c3+2*w3, n)      -- first read by epoch e+2. It overlaps
            //                                epoch e+1's entire panel chain, which
            //                                is the serial part of the schedule.
            //
            //  Two orderings must hold and both are enforced, not hoped for:
            //   (a) transform composition is ordered (31:120(c)), so epoch e+1's
            //       far must follow epoch e's on the SAME columns -- hence the wait
            //       on evDef[par^1] before e+1 touches [c3+2*w3, ...);
            //   (b) the deferred apply reads V and T of epoch e out of A, which
            //       epoch e+1's panels do not write (disjoint columns), so no copy
            //       of the transform state is needed to hold the generation open.
            //
            //  This is the ONE place the schedule is concurrent, so it is also the
            //  only place a data race could hide. TQR_PIPE=0 collapses it to a
            //  single stream for the control arm.
            const int far = n - (c3 + w3);
            if (far > 0) {
                const int par = e3 & 1;
                const int eager = pipeOn ? std::min(far, w3) : far;
                // (a): epoch e-1's deferred far wrote these columns and must be
                // composed before this one.
                if (defPending[par ^ 1]) {
                    CUDA_CHECK(cudaStreamWaitEvent(st, evDef[par ^ 1], 0));
                    defPending[par ^ 1] = false;
                }
                build_mask(A, c3, w3, Y3[tpar]);
                commit(A, 3, c3, w3, T3c, nu3, c3 + w3, eager, f3, /*gen=*/0, Y3[tpar]);
                const int rest = far - eager;
                if (rest > 0) {
                    CUDA_CHECK(cudaEventRecord(evEpoch, st));
                    CUDA_CHECK(cudaStreamWaitEvent(sDef, evEpoch, 0));
                    commit(A, 3, c3, w3, T3c, nu3, c3 + w3 + eager, rest, f3, /*gen=*/1,
                           Y3[tpar]);
                    CUDA_CHECK(cudaEventRecord(evDef[par], sDef));
                    defPending[par] = true;
                    rec.pipeline[3] += 1;      // one deferred generation actually live
                }
            }
            ++e3;
        }
        // drain both generations before the caller may read A
        for (int g = 0; g < 2; ++g)
            if (defPending[g]) {
                CUDA_CHECK(cudaStreamWaitEvent(st, evDef[g], 0));
                defPending[g] = false;
            }
        // liveness verdict, computed from the counters actually raised
        rec.every_rung_live = true;
        for (int i = 0; i < kRungs; ++i)
            if (!(rec.split[i] && rec.replicate[i] && rec.combine[i] && rec.pipeline[i]))
                rec.every_rung_live = false;
    }

    // ============================================ 5. THE RUNTIME PROOF RECORD
    //  31:640, all nine items. "A plan whose measured counters violate the model
    //  is not covered by the theorem, even if it is fast."
    void emit(const char* tag) const {
        std::printf("\n[TQR-OMEGA] %s  m=%d n=%d  b=%d  nu=[%d,%d,%d,%d]  tier=%s\n",
                    tag, m, n, b, b, nu1, nu2, nu3, tier_name(tier));
        std::printf("  rung  boundary       P     M(w)   D   nu     nu*   nu_cap binds"
                    "    K    S=K+D   c  r  d   p_r\n");
        static const char* bindname[4] = {"sqrt(M)", "B/n", "chi", "workspace"};
        for (int i = 0; i < kRungs; ++i) {
            const LevelPlan& L = rec.levels[i];
            std::printf("  %-4d  %-12s %4d %8zu %3d %5d %7d %7d %-9s %5d %6llu %3d %2d %2d %5d\n",
                        i, hier[i].name, L.P, L.M_words, L.D, L.epoch_nu, L.nu_star,
                        L.nu_cap, bindname[L.nu_binding], L.K,
                        (unsigned long long)L.predicted_sync_rounds,
                        L.replica_layers, L.combine_fanin, L.pipeline_depth, L.p_rows);
        }
        std::printf("  panel p_r=%d of %d SMs (occupancy %.2f) carrier=%s -- A5 partition "
                    "at the launch\n", panelPr, smCount,
                    (double)panelPr / std::max(1, smCount),
                    panelCluster ? "CLUSTER(DSMEM barrier)" : "gmem barrier");
        // 10:S6.3's carrier and WHY it is or is not available, in the record itself.
        std::printf("  panel tpbQR=%d (block size the QR kernels' registers admit)\n", tpbQR);
        std::printf("  panel algorithm=%s   tsqr_ready=%d  (SLAB=%d needs %zu words vs "
                    "kappa*n*b=%.0f, ledger holds %.0f)\n",
                    panel_algo(), (int)tsqrReady, SLAB, tsqrNeedWords,
                    Planner::kKappa * (double)n * b, (double)rec.w_ledger);
        {
        }
        std::printf("  --- moves (Split/Replicate/Combine/Pipeline), observed ---\n");
        for (int i = 0; i < kRungs; ++i)
            std::printf("  rung %d: S=%-10lld R=%-10lld C=%-10lld P=%-10lld  live=%s\n",
                        i, rec.split[i], rec.replicate[i], rec.combine[i], rec.pipeline[i],
                        (rec.split[i] && rec.replicate[i] && rec.combine[i] && rec.pipeline[i])
                            ? "yes" : "NO");
        std::printf("  --- the two quantities ---\n");
        for (int i = 0; i < kRungs; ++i) {
            const LevelPlan& L = rec.levels[i];
            const double S_obs = (double)rec.rounds[i];
            const double S_tgt = (double)(L.K + L.D);
            const double Q_obs = (double)rec.words[i];
            const double S_dev = (double)rec.dev_rounds[i];
            std::printf("  rung %d:  S_obs=%-8.0f  K+D=%-8.0f  S/(K+D)=%-7.2f   "
                        "Q_obs=%-12.4g  B_lambda=%-12.4g  Q/B=%-7.2f\n",
                        i, S_obs, S_tgt, S_tgt > 0 ? S_obs / S_tgt : 0.0,
                        Q_obs, L.B_budget, L.B_budget > 0 ? Q_obs / L.B_budget : 0.0);
            // 31:371 is S_lambda = Theta(K_lambda + D_lambda) over DEPENDENCY-BEARING
            // rounds, not over DAG nodes. S_obs above counts 31:329's X_{k,j} epoch
            // nodes and is therefore blind to any barrier schedule running inside one
            // node -- which is exactly where the rung-1 panel hides Theta(b) of them.
            // Reported separately so the claim is falsifiable: 31:520 is explicit that
            // a plan whose measured counters violate the model is not covered by the
            // theorem, even if it is fast.
            if (S_dev > 0)
                std::printf("           S_dev=%-8.0f (device-wide rounds actually "
                            "executed)  S_dev/(K+D)=%-7.2f  %s\n",
                            S_dev, S_tgt > 0 ? S_dev / S_tgt : 0.0,
                            (S_tgt > 0 && S_dev / S_tgt > 2.0)
                                ? "<== VIOLATES 31:371" : "");
        }
        const double nb = (double)n * b;
        std::printf("  --- proof record ---\n");
        std::printf("  householder_only=%d  gram_calls=%lld  stale_reads=%lld  no_cuda_graphs=%d\n",
                    (int)rec.householder_only, rec.gram_calls, rec.stale_reads,
                    (int)rec.no_cuda_graphs);
        std::printf("  storage/nb:  leafT=%.2f (<=1.5)  ledger=%.2f (<=kappa=3)  "
                    "protocol=%.2f (<=1)  nonconforming=%.2f   => compact=%d\n",
                    rec.w_leafT / nb, rec.w_ledger / nb, rec.w_protocol / nb,
                    rec.aux_words_nonconforming / nb, (int)rec.compact_global_storage);
        // 31:398: communication optimality survives a sub-nu* width, but
        // synchronization optimality LAPSES, by the factor nu*/nu in frontier
        // rounds. Stated per rung rather than left to be inferred from the table.
        // Rung 0's width IS the leaf b, and 10:S6.2 is explicit that b comes from
        // the kernel's admissible register/shared set and "is NOT derived from the
        // entire SM register-file capacity by a single sqrt(M) formula". Comparing
        // b against sqrt(M_0) therefore reports a lapse that the spec does not
        // claim, so rung 0 is excluded from this note rather than flagged forever.
        for (int i = 1; i < kRungs; ++i) {
            const LevelPlan& L = rec.levels[i];
            const double slack = (double)L.nu_star / std::max(1, L.epoch_nu);
            if (slack > 2.0)
                std::printf("  NOTE rung %d: nu=%d is %.1fx below nu*=%d -- "
                            "width-maximality fails, so S-optimality lapses here (31:398)\n",
                            i, L.epoch_nu, slack, L.nu_star);
        }
        if (farFlops > 0)
            std::printf("  F_EC,far/F_far = %.4f   (10:S8.3 requires exactly 1 in the "
                        "3xTF32 tier; below 1 means a far tile silently fell back)\n",
                        farFlops > 0 ? (ecFarFlops / 3.0) / farFlops : 0.0);
        std::printf("  every_rung_live=%d   plan_enumerated=%d   That=%.6g s"
                    "  (sum Q/B=%.6g s, sum alpha*S=%.6g s)\n",
                    (int)rec.every_rung_live, (int)rec.plan_enumerated, rec.That_s,
                    rec.pred_bandwidth_s, rec.pred_latency_s);
    }
};

// ---------------------------------------------------------------------------
//  Panel-rate probe: measure one leaf panel at this b, on this device, at a
//  representative height. This is the measured entry of 10:S4.3's kernel-rate
//  table -- the only thing that lets T_hat rank two admissible leaf widths.
//
//  It allocates its own minimal scratch and frees it, so the loser of a b
//  comparison never holds the full working set. Cost is a few hundred
//  microseconds once per factorization.
// ---------------------------------------------------------------------------
template <typename Real, int NB>
inline double probe_panel_s(int m, int reps = 3, int prSel = 0, bool wantCluster = false) {
    using O = Omega<Real, NB>;
    // Representative height, bounded. The 8192 cap was too aggressive: at
    // m = 131072 the real panel clamps rpb against clRpbMax and behaves nothing
    // like an 8192-row probe, so the model ranked p_r on a measurement that did
    // not describe the launch -- and picked one that cost 5.7% there. Probing
    // nearer the true height costs a few extra launches once per factorization
    // and is bounded by min(), so small cells stay cheap.
    const int mk = std::max(NB, std::min(m, 65536));
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const int smCount = prop.multiProcessorCount;
    int rpb = 128;
    if (ceil_div(mk, rpb) > smCount) rpb = 32 * ceil_div(ceil_div(mk, smCount), 32);
    rpb = std::min(rpb, (int)O::clRpbMax);
    // Probe the grid the plan would actually launch, so the measurement ranks the
    // real candidate rather than a nominal one.
    if (prSel > 0) {
        const int want = ceil_div(mk, prSel);
        rpb = std::min((int)O::clRpbMax, std::max((int)O::kRpbFloor, O::roundDown32(want)));
    }
    int nbl = ceil_div(mk, rpb);
    const bool overflow = (nbl > smCount);
    if (overflow) { rpb = O::clRpbMax; nbl = smCount; }

    Real *A = nullptr, *Y = nullptr, *Tp = nullptr, *sc = nullptr, *sl = nullptr,
         *pr = nullptr, *Pp = nullptr, *Ps = nullptr, *T16 = nullptr;
    double *sd = nullptr, *dd = nullptr, *dcp = nullptr, *dcr = nullptr; unsigned* gb = nullptr;
    auto al = [&](Real** p, size_t w) { CUDA_CHECK(cudaMalloc(p, w * sizeof(Real)));
                                        CUDA_CHECK(cudaMemset(*p, 0, w * sizeof(Real))); };
    al(&A, (size_t)mk * NB); al(&Y, (size_t)mk * NB); al(&Tp, (size_t)NB * NB);
    al(&sc, 4 * NB); al(&sl, 256); al(&pr, 32);
    al(&Pp, (size_t)std::max(132, nbl) * NB * 16);
    al(&Ps, (size_t)2 * NB * 16 * (NB / 16));
    al(&T16, (size_t)2 * 16 * 16 * (NB / 16));
    CUDA_CHECK(cudaMalloc(&sd, NB * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dd, 256 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gb, 2 * sizeof(unsigned)));
    // A6's per-block column partials. The probe builds its OWN DomArgs, so it needs
    // its own buffer -- leaving dcolP unset here is what produced "misaligned address"
    // in job 9716: an uninitialized device pointer dereferenced by the panel kernel.
    // Sized for the widest grid the probe can launch (2 generations x blocks x PW+1).
    CUDA_CHECK(cudaMalloc(&dcp, (size_t)2 * 132 * 17 * sizeof(double)));
    CUDA_CHECK(cudaMemset(sd, 0, NB * sizeof(double)));
    CUDA_CHECK(cudaMemset(dd, 0, 256 * sizeof(double)));
    CUDA_CHECK(cudaMemset(dcp, 0, (size_t)2 * 132 * 17 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dcr, (size_t)2 * 17 * sizeof(double)));
    CUDA_CHECK(cudaMemset(dcr, 0, (size_t)2 * 17 * sizeof(double)));
    CUDA_CHECK(cudaMemset(gb, 0, 2 * sizeof(unsigned)));
    // A must not be identically zero or the reflectors degenerate; a cheap
    // deterministic fill keeps the probe representative without an RNG.
    {
        std::vector<Real> h((size_t)mk * NB);
        for (size_t i = 0; i < h.size(); ++i) h[i] = (Real)(((i * 2654435761u) % 1000) + 1) / 500 - 1;
        CUDA_CHECK(cudaMemcpy(A, h.data(), h.size() * sizeof(Real), cudaMemcpyHostToDevice));
    }
    for (const void* fp : {(const void*)k_panel_domino<Real, NB, 16, false, true>,
                           (const void*)k_panel_domino<Real, NB, 16, false, true, true>})
        CUDA_CHECK(cudaFuncSetAttribute(fp, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        O::domSmem((int)O::clRpbMax)));

    DomArgs<Real> da;
    da.A = A; da.lda = mk; da.mk = mk; da.row0 = 0; da.rpb = rpb;
    da.SPV = Y; da.ldspv = mk; da.spvcol = 0; da.Tpan = Tp;
    da.tauT = sc; da.fv = sc + NB; da.bet = sc + 2 * NB; da.sig = sc + 3 * NB;
    da.dslot = sl; da.prow = pr; da.Pp = Pp; da.Psum = Ps;
    da.sigD = sd; da.dslotD = dd; da.dcolP = dcp; da.dcolStride = 132;
    da.T16g = T16; da.tphase = nullptr; da.nslab = nbl;

    // The probe must launch the SAME WAY leaf_panel will, or the model ranks a
    // launch it will not perform. That mistake has already been paid for once: the
    // min(m,8192) probe height misrepresented the rpb clamp at m=131072 and cost
    // 5.7% there. With the cluster carrier in play the gap would be far larger --
    // a gmem-barrier measurement standing in for a DSMEM-barrier launch.
    const bool clusterOn = [] {
        const char* e = getenv("TQR_CLUSTER"); return !e || atoi(e) != 0;
    }();
    int crpb = 0, cnbl = 0;
    bool useCluster = false;
    if (clusterOn && wantCluster && !overflow && mk <= (int)O::clRowMax) {
        const int cw = std::min(std::max(1, prSel), (int)O::kClusterMax);
        crpb = std::max((int)O::kRpbFloor, O::roundDown32(ceil_div(mk, cw)));
        crpb = std::min(crpb, (int)O::clRpbMax);
        cnbl = ceil_div(mk, crpb);
        useCluster = (cnbl >= 1 && cnbl <= (int)O::kClusterMax &&
                      O::domSmem(crpb) <= O::SMEM_CAP);
    }
    // A cluster candidate that is not legal here must not be measured as if it were
    // the gmem carrier -- that would let the model select a carrier it cannot run.
    if (wantCluster && !useCluster) return std::numeric_limits<double>::infinity();
    if (useCluster) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_domino<Real, NB, 16, true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize, O::domSmem(crpb)));
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k_panel_domino<Real, NB, 16, true>,
                   cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    }

    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    double best = 1e30;
    for (int r = 0; r < reps + 1; ++r) {
        CUDA_CHECK(cudaEventRecord(e0));
        if (useCluster) {
            DomArgs<Real> cda = da; cda.rpb = crpb; cda.nslab = cnbl;
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(cnbl); cfg.blockDim = dim3(512);
            cfg.dynamicSmemBytes = (size_t)O::domSmem(crpb); cfg.stream = 0;
            cudaLaunchAttribute at[1];
            at[0].id = cudaLaunchAttributeClusterDimension;
            at[0].val.clusterDim = {(unsigned)cnbl, 1u, 1u};
            cfg.attrs = at; cfg.numAttrs = 1;
            if (cudaLaunchKernelEx(&cfg, k_panel_domino<Real, NB, 16, true>,
                                   cda, (unsigned*)nullptr) != cudaSuccess) {
                (void)cudaGetLastError();
                useCluster = false;
                k_panel_domino<Real, NB, 16, false, true>
                    <<<nbl, 512, O::domSmem(rpb)>>>(da, gb);
            }
        }
        else if (overflow) k_panel_domino<Real, NB, 16, false, true, true>
                          <<<nbl, 512, O::domSmem(rpb)>>>(da, gb);
        else          k_panel_domino<Real, NB, 16, false, true>
                          <<<nbl, 512, O::domSmem(rpb)>>>(da, gb);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        if (r > 0) best = std::min(best, (double)ms);   // drop the warmup
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    for (void* p : {(void*)A,(void*)Y,(void*)Tp,(void*)sc,(void*)sl,(void*)pr,
                    (void*)Pp,(void*)Ps,(void*)T16,(void*)sd,(void*)dd,(void*)dcp,(void*)dcr,(void*)gb})
        cudaFree(p);
    // Returned as measured, deliberately unscaled. The probe runs at ONE height
    // while the real chain sweeps mk from m down to b, so this is not the absolute
    // per-panel cost and must never be read as one. It is used only to RANK two
    // leaf widths at the same m, where both are probed identically and the height
    // bias cancels in the comparison. Scaling it by a modelled average-height
    // factor would be a fitted constant standing in for a mechanism, which is the
    // failure mode this project has already paid for twice (the sqrt(2) accuracy
    // "law" and the 4096/n cap); the ranking does not need it, so it is not there.
    return best * 1e-3;
}

// Run the planner alone, with a measured panel probe, and return its T_hat. This
// is what makes the enumeration over b real rather than degenerate: b is fixed at
// compile time by the panel kernel's instantiation, so the family is enumerated by
// calling this once per instantiated width and taking the argmin -- the same exact
// enumeration 31:486 prescribes, just spread across template instantiations.
template <typename Real, int NB>
inline double predict_That(int m, int n) {
    Boundary h[kRungs];
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int smemOptin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smemOptin,
               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    smemOptin = std::min(smemOptin, Omega<Real, NB>::SMEM_CAP);
    probe_hierarchy(h, (int)sizeof(Real), smemOptin, prop.multiProcessorCount,
                    (size_t)prop.l2CacheSize);
    const double peak = std::is_same<Real, double>::value ? 6.7e13 : 6.0e14;
    const double tp = probe_panel_s<Real, NB>(m);
    ProofRecord r;
    // SAME admissibility rule the factorization uses. Passing the raw
    // domSmem(clRpbMax) here is what let --autob select a width that create()
    // would then refuse to plan for and run as an unchecked FALLBACK.
    if (!Planner::enumerate(h, m, n, NB, peak, r, tp, prop.multiProcessorCount, nullptr,
                            Omega<Real, NB>::panelSmemBytes(),
                            Omega<Real, NB>::SMEM_CAP))
        return std::numeric_limits<double>::infinity();
    return r.That_s;
}

}  // namespace tqr
