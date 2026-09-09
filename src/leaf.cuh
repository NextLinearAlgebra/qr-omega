#pragma once
// leaf.cuh -- LeafHouseholderQR, Algorithm 3 line 1.
//
//   1: function LeafHouseholderQR(A in F^{m'xn'}, b)   . m'n' <= M_l: the block is resident
//   2:   for j = 1, b+1, 2b+1, ... do
//   3:     factor the b-wide panel A(j:m', j:j+b-1) by b applications of hhqr of (3),
//          storing each v^(k) in the annihilated positions
//   4:     T_j <- the b x b compact-WY factor of those b reflectors
//   5:     A(j:m', j+b:n') -= Y_j T_j^* Y_j^* A(j:m', j+b:n')
//   6:   end for
//   7:   return (Y, T, R) in place            . b = Theta(sqrt(M_0)) by Prop. 6.28
//   8: end function
//
// This is the base case of the whole recursion (Algorithm 1 line 1 returns here whenever
// lambda = 0 or the block fits), so it is the only place real arithmetic happens. Two
// properties of the document rest on it and neither is negotiable:
//
//  * THE SIGN CONVENTION. Section 3.1 defines hhqr "with a fixed diagonal sign
//    convention", and Lemma 4.4 makes uniqueness of a pebble's denotation depend on
//    "fixed ownership, fixed leaf order, the sign convention". The ordered stack merge
//    (2) is associative only once signs are fixed, so a convention that varied with, say,
//    rounding would make Combine(a) ill-defined and the merge non-deterministic. We take
//    beta = -sign(alpha)*||x||, a deterministic function of the input alone.
//
//  * BACKWARD STABILITY. Theorem 13.1 asks for a constant independent of the replication
//    factor, the tile geometry and the tree shapes. That is inherited from the leaf only
//    if the leaf is itself backward stable, so the reflector generation below uses the
//    cancellation-free branch and a scaled norm; see the notes at each step.
//
// Storage, in place, as line 7 requires: v^(k) occupies the annihilated positions
// A(k+1:m, k) with v(k) = 1 implicit, R occupies the upper triangle including the
// diagonal, and tau is returned separately. T factors go to a caller-supplied workspace,
// which is where Algorithm 1's "{T_s} in O(nb) workspace" comes from.

#include "machine.cuh"

#include <cmath>

#if defined(__CUDACC__)
#include <mma.h>   // fp64 tensor cores for line 5 (see do_update_tc)
#endif

namespace tqr {

// =================================================================================
// WHAT SHAPES THIS ACTUALLY SEES, AND WHICH FORM TO USE
// =================================================================================
// Derived from the algorithms rather than assumed, because the answer is not square and
// tuning against square matrices tunes against the wrong thing.
//
//   Alg 1 line 6 : K = ceil(n/nu) superpanels, each m x nu
//   Alg 1 line 7 : (Y_s, T_s) <- PanelQR(A(:, nu_s), lambda - 1)     panel is m x nu
//   Alg 2 line 1 : Split P into p row blocks
//   Alg 2 line 3 : (V_t, T_t, R_t) <- TQR-Omega(P_t, lambda - 1)     leaf is (m/p) x nu
//   Alg 1 line 1 : leaf entered when lambda = 0 or m'n' <= M_lambda
//
// So n is NOT free: it is nu, fixed by Plan for that boundary. Only the row count varies,
// and the leaf is tall and narrow. On the machine measured here (Plan, 262144^2 problem):
//
//   boundary     n = nu      b     M_lambda     tallest leaf that fits
//   registers         9      8          512                        36
//   smem/L1          89      8       29184                       324
//   L2             1513      8      7.9e+06                      5195
//
// The level-1 leaf is the one that runs on an SM: n = 89, b = 8, m anywhere up to 324.
//
// MEASURED, 2048 leaves across the GPU, n = 89, b = 8 (GFLOP/s, fp64 peak 33.5 TFLOP/s):
//
//     m      warp form          block form
//    89      658  (3/SM)        316
//   128      632  (2/SM)        423
//   192      391  (1/SM)        561
//   256      431  (1/SM)        640
//
// Two rules follow, and both are worth real throughput:
//
//   1. CHOOSE THE FORM BY WARP SUPPLY, not by m: use the block form once the warp form
//      can no longer keep ~8 warps resident (see leaf_prefers_block for the measurements).
//      Below it the warp form wins because several leaves fit per SM and their panels
//      hide each other's latency; above it only one leaf fits, occupancy is gone, and the
//      block form -- which puts every thread on that single leaf -- wins instead. Picking
//      the wrong side costs about 2x either way.
//
//   2. SET THE OTHER THREE KNOBS FROM THE SHAPE. Each has its own measured crossover and
//      a helper that encodes it; none of them is a guess:
//        LOOKAHEAD  leaf_prefers_lookahead(m, n)        squarer than m ~ 1.5n
//        PW         leaf_panel_warps(m, n)              1 / 2 / 4 as m crosses 128, 256
//        TC         leaf_prefers_tensor_cores<T>(m,n,b) m >= 128 fp64, m >= 256 fp32
//      Together they are worth 1.3x to 2.5x over the plain block form on the real shapes.
//
//   3. TAKE THE LEAF AS BIG AS IT FITS. RECURSE ONLY WHEN IT DOES NOT.
//      The occupancy cliff that used to punish tall leaves is gone: with the tensor-core
//      update, 256x89 runs at 1149 GF/s against 1129 at 128x89 -- flat, where the scalar
//      path fell from 851 to 714. So there is no longer an occupancy argument for
//      splitting a leaf that fits, and splitting one measures much WORSE:
//
//        256x89 monolithic          6.39 ms                    1149 GF/s
//        256x89 as 2x(128x89)       5.65 ms leaf + 4.80 merge   702 GF/s
//
//      Halving the leaf does buy 12% on the factorization itself, exactly as the
//      occupancy model predicts, and then the merge costs 6.5x what that saved. A merge
//      is pure overhead when the rows are already resident in one SM's reach; it earns
//      its keep in Algorithm 2 only because there the row blocks live in DIFFERENT
//      domains and the data movement is happening regardless.
//
//      Recursion is still mandatory above the capacity: 512x89 needs 365 KiB and cannot
//      launch at all, and splitting it two ways gives 892 GF/s (four ways gives 611 --
//      recurse only as deep as the fit requires, never deeper). This is Algorithm 1
//      line 1 read literally: descend when m'n' > M_lambda, and not before.
//
// =================================================================================
// WHERE THIS DEVIATES FROM ALGORITHM 3, AND WHY
// =================================================================================
// Algorithm 3 lines 1-8 specify: for each j, factor the b-wide panel by b applications of
// hhqr, form the b x b compact-WY factor T_j, then apply Y_j T_j* Y_j* to the trailing
// columns. Everything below computes exactly that. The deviations are in SCHEDULE,
// SUMMATION ORDER and STORAGE, and each is recorded here because two of them are visible
// in the low-order bits.
//
//  1. SCHEDULE (LOOKAHEAD = true). Line 5's trailing update is split into the near block
//     (the next panel's own columns) and the far block, and panel j+1 is factored on the
//     panel warps while the others run the far sweep of panel j. This is Algorithm 1
//     lines 10-12 applied one level down, and it is the same reflectors applied in the
//     same order: measured BIT-IDENTICAL to the plain schedule (0.00e+00) on every real
//     shape. Costs one extra b x b of workspace for the double-buffered T.
//
//  2. SUMMATION ORDER (PW > 1). house_multi reduces the norm across PW warps through
//     shared memory instead of inside one warp, so the additions happen in a different
//     order and the reflectors differ in the last bits -- 3e-14 relative at 89x89 against
//     8e-16 for PW = 1. This is NOT the order the theory constrains: Proposition 13.2 and
//     Counterexample 16.1 fix the order of the MERGES in Algorithm 2 line 9, which
//     LeafMergeQR_warp honours exactly, not the order of a summation inside one leaf's
//     norm. It does mean PW > 1 is not bitwise reproducible against PW = 1; if that is
//     ever required, pin PW = 1.
//
//  3. SUMMATION ORDER (TC > 0). The MMA accumulates line 5's two GEMMs in its own order.
//     Products are exact fp64, so this stays at 8e-16, but it is the same class of
//     deviation as (2). For T = float the deviation is larger and deliberate: the update
//     runs as 3xTF32, giving ||A-QR||/||A|| of 2.1-2.5e-06 against 1.3-1.6e-06 scalar.
//     Plain TF32 (2.89e-04) is rejected precisely because it would break the backward
//     stability that makes Householder the right primitive here.
//
//  4. STORAGE. Algorithm 1's postcondition asks for {T} in O(nb) workspace. This leaf
//     also stages W for line 5 and, on the tensor-core path, one 8 x 8 fragment tile per
//     warp; leaf_capacity_words() reports the true total. The leading dimension is padded
//     (leaf_pad_for) for bank behaviour, so the resident footprint exceeds m'n' by
//     n*pad words. THIS MATTERS FOR ALGORITHM 1 LINE 1: the test is m'n' <= M_lambda, but
//     what must actually fit is leaf_capacity_words(), so callers must size against that
//     and not against m'n'.
//
//  5. THE MERGE. Algorithm 2 line 9 says hhqr on the stacked [R_t1* ... R_tq*]*.
//     LeafMergeQR_warp skips the structural zeros of those upper-triangular blocks --
//     the same factorization, 2.9x to 5.0x fewer flops, and the fixed leaf order of line 1
//     is preserved exactly as Proposition 13.2 requires.
//
//  6. THE SHAPE ITSELF. This is a deviation from Algorithm 4's geometry, not just from
//     Algorithm 3's schedule, and it is recorded because it is a real departure.
//
//     Plan puts the boundary-1 leaf at nu = 88.94, so n = 89. The register-resident form
//     cannot take that width: it needs n = NW*b with b = 8, so n must be a multiple of 8,
//     and the register file caps NW at 16 -- beyond that the per-thread allocation spills,
//     measured at n = 176, NW = 22, where ptxas cuts to 40 registers and throughput falls
//     to 635 GF/s. Measured, fp64, |R| verified against an independent host QR:
//
//       shape      NW   warps/SM  blocks/SM   GF/s   |R| vs host QR
//        16x16      2      12         6        123      3.1e-16
//        32x32      4      12         3        236      7.7e-16
//        64x64      8      16         2        450      7.2e-16
//        96x96      8       8         1        257      1.8e-15   <- b = 12, see below
//       128x128    16      16         1        560      2.4e-15   <- KAMI's largest square
//       128x88     11      11         1        489      4.6e-16
//       178x88     11      11         1        603      4.4e-16
//       256x88     11      11         1        795      5.5e-16   <- nearest the theory's nu
//       324x88     11      11         1        755      4.9e-16
//       512x88     11      11         1        759      3.0e-15
//       256x128    16      16         1       1000      5.5e-16
//       256x64      8      16         2       1115      5.5e-16   <- best
//
//     Two hard limits sit in that table. b MUST stay 8: at b = 12 (the 96x96 row) the
//     register allocation goes to 228 and only one block of eight warps survives, which
//     costs half the throughput. And NW <= 16: at NW = 22 the per-thread allocation
//     spills, ptxas cuts to 40 registers, and 256x176 falls to 635 GF/s.
//
//     n = 64 rather than 89 is worth 40% (1115 against 795), and the table shows why: at
//     n = 64 the block needs only 8 warps, so TWO blocks fit per SM and the warp count
//     doubles. The theory's nu is derived from a shared-memory capacity M_1; once the
//     operand budget is the register file instead, the capacity that fixes the width is a
//     different one, and it gives 64. Against KAMI's own square block sizes the
//     tall-skinny shape is 2.0x better (1115 at 256x64 against 560 at 128x128).
//
//     Prefer m tall and n = 64. Nesting (Lemma 11.1) then wants nu_1 = 64 rather than 89,
//     which changes K = ceil(n/nu) upstream; that is the caller's business, but it is a
//     genuine change to Algorithm 4's output and must not be made silently.
//
// =================================================================================

// LAUNCH CONSTRAINT. Registers cap the block, not just shared memory: threads * regs must
// stay under 65536 per SM, and the tensor-core paths raise the register count (71 scalar,
// 96 with m8n8k4, 128 with m16n8k8), so the largest launchable block shrinks accordingly.
// Size groups with leaf_groups_per_block().
//
// BLOCK SIZE for the block form: measured with the tensor-core update and look-ahead,
// throughput rises with the thread count until it runs out of registers or shared memory,
// so take the largest block that still launches -- 640 threads for m >= 178 (782 GF/s at
// 178x89, 1037 at 256x89). The exception is m = 128, which has a hard cliff between 320
// and 352 threads (1010 -> 552 GF/s) when the second block stops fitting on the SM; there
// 320 is the right answer. Note the leaf does NOT want warps merely matched to the column
// tile count: at 178x89 and 256x89 there are 11 tiles and performance still improves out
// to 20 warps.
//
// =================================================================================
// Block-cooperative device implementation.
// =================================================================================
// Same mathematics, same sign convention, same storage; a whole thread block does the
// arithmetic. A profile of the naive cooperative version put the three phases of
// Algorithm 3 at roughly 35% / 27% / 38% (panel / form T / trailing update), so all
// three are optimised rather than just the one with the flops:
//
//  * The PANEL and FORM T are reduction-bound, not flop-bound: the naive form issues one
//    block-wide reduction per (reflector, column) pair, which is O(b^2) reductions each
//    costing a log(nt) tree of __syncthreads. Both phases here BATCH their independent
//    dot products into a single multi-value reduction, taking the count to O(b), and the
//    reduction itself is warp-shuffle based so it costs one __syncthreads rather than
//    log(nt) of them.
//  * The TRAILING UPDATE is the flops and parallelises over COLUMNS with no reduction at
//    all: a thread that owns column c does the whole of w = Y^* A(:,c), w <- T^* w and
//    A(:,c) -= Y w by itself. It is restructured to sweep rows in the outer loop so the
//    column is read once instead of nb times, and w lives in registers rather than in
//    the caller's workspace.
//
// PERFORMANCE NOTE ON `lda`. In shared memory a column-major block with lda a multiple of
// 32 puts every thread's column on the SAME bank -- thread t touches r + lda*t, and
// lda = 128 gives bank (r + 128t) % 32 = r % 32 for every t, a 32-way conflict on the
// hottest loop in the routine. Pass an odd lda (m + 1 is the usual choice) and it
// vanishes. The routine cannot do this for the caller because it does not own the
// allocation, so it is the caller's job and it is worth real time.
//
// Every thread of the block must call these: the reductions and the __syncthreads are
// collective, so an early return by any thread hangs the block. blockDim.x must be a
// multiple of the warp size, which the shuffles require.
#if defined(__CUDACC__)

// =================================================================================
// LeafHouseholderQR -- register-resident, engineered on KAMI's three techniques
// =================================================================================
// The matrix never enters shared memory. It is loaded from global straight into
// registers and stays there for the whole factorization; shared holds only the one
// operand that has to cross warp boundaries.
//
//   DISTRIBUTION. Warp w owns columns [w*COLS, (w+1)*COLS). Inside a warp a column is
//   spread over the 32 lanes BY ROW: lane l holds rows l, l+32, l+64, ... So a whole
//   column lives inside one warp, and the panel width is exactly one warp's column
//   count, b = COLS.
//
//   (2) GLOBAL -> REGISTER, per KAMI. `a[COLS][MC]` is the matrix. MC = ceil(m/32)
//       doubles per lane per column; the register cost is COLS*MC doubles = 2*COLS*MC
//       registers, and that -- not shared memory -- is what bounds the leaf.
//
//   (3) NO CROSS-WARP REDUCTION, per KAMI. Because a column is warp-local, every norm
//       and every dot product in the panel is a shuffle. The panel therefore runs from
//       start to finish with NO barrier: reflector generation, the b-1 in-panel applies
//       and the compact-WY T are all warp-local. The only two barriers per block column
//       are the ones that publish the broadcast.
//
//   (1) FRAGMENT-ORDER STAGING, per KAMI. Shared carries only Y (m x b) and T (b x b) --
//       the workspace is m*b + b*b words and nothing else. The owning warp writes Y in
//       the order its lanes hold it and every reader reads it the same way, so lane l
//       always touches row l + 32i: contiguous, and at the 2-wavefront fp64 floor with
//       no bank conflicts, without any padding trick.
//
// WHAT IT COSTS. Pinning columns to warps means a warp whose columns are already
// factored has nothing left to do, so utilisation falls as the sweep advances -- with
// one panel per warp the k-th block column leaves k warps idle. That is the price of
// technique 2 for a factorization: KAMI's GEMM never pays it because every output tile
// is live throughout. It is why the shapes below want to be TALL: the taller the leaf,
// the more work each surviving warp has to cover the loss.
//
//   T    : element type
//   NW   : warps in the block; n must equal NW*COLS
//   COLS : columns per warp, and the panel width b
//   MC   : ceil(m/32), doubles per lane per column
//
// Every thread of the block must call this: the barriers are collective.
template <typename T, int NW, int COLS, int MC>
__device__ inline void LeafHouseholderQR(const T* __restrict__ Ain, T* __restrict__ Aout,
                                         int m, int n, T* Y, T* Tout) {
    constexpr int B = COLS;
    const int lane = (int)(threadIdx.x & 31), warp = (int)(threadIdx.x >> 5);

    T a[COLS][MC];                                   // the matrix lives here
    #pragma unroll
    for (int c = 0; c < COLS; ++c)
        #pragma unroll
        for (int i = 0; i < MC; ++i) {
            const int r = lane + 32 * i, gc = warp * COLS + c;
            a[c][i] = (r < m && gc < n) ? Ain[r + (size_t)gc * m] : T(0);
        }

    // Factor block column `pjb` (columns pj..pj+COLS-1), staging Y into `Yb`
    // and T into its own slot of the output. Warp-local: no barrier inside.
    auto do_panel = [&](int pjb, T* Yb) {
        const int j = pjb * COLS;
        T* Y = Yb;
        T* Tm = Tout + (size_t)pjb * COLS * COLS;
                T tv[COLS];
                #pragma unroll
                for (int k = 0; k < COLS; ++k) {
                    const int d = j + k;
                    tv[k] = T(0);
                    if (d >= m || d >= n) continue;
                    T ss = T(0);
                    #pragma unroll
                    for (int i = 0; i < MC; ++i) {
                        const int r = lane + 32 * i;
                        if (r > d && r < m) ss += a[k][i] * a[k][i];
                    }
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
                    const int dl = d & 31, di = d >> 5;
                    T alpha = (di < MC) ? a[k][di] : T(0);
                    alpha = __shfl_sync(0xffffffffu, alpha, dl);
                    T beta = alpha, tau = T(0);
                    if (ss > T(0)) {
                        const T nr = std::sqrt(alpha * alpha + ss);
                        beta = (alpha > T(0)) ? -nr : nr;   // sign convention of section 3.1
                        tau  = (beta - alpha) / beta;
                        const T sc = T(1) / (alpha - beta);
                        #pragma unroll
                        for (int i = 0; i < MC; ++i) {
                            const int r = lane + 32 * i;
                            if (r > d && r < m) a[k][i] *= sc;
                        }
                    }
                    if (lane == dl && di < MC) a[k][di] = beta;
                    tv[k] = tau;
                    if (tau != T(0)) {
                        #pragma unroll
                        for (int c = 0; c < COLS; ++c) {
                            if (c <= k) continue;
                            T w = T(0);
                            #pragma unroll
                            for (int i = 0; i < MC; ++i) {
                                const int r = lane + 32 * i;
                                if (r == d)              w += a[c][i];      // implicit unit head
                                else if (r > d && r < m) w += a[k][i] * a[c][i];
                            }
                            #pragma unroll
                            for (int o = 16; o > 0; o >>= 1) w += __shfl_xor_sync(0xffffffffu, w, o);
                            w *= tau;
                            #pragma unroll
                            for (int i = 0; i < MC; ++i) {
                                const int r = lane + 32 * i;
                                if (r == d)              a[c][i] -= w;
                                else if (r > d && r < m) a[c][i] -= w * a[k][i];
                            }
                        }
                    }
                }
                // stage Y in fragment order: each lane writes exactly the rows it holds
                #pragma unroll
                for (int c = 0; c < COLS; ++c)
                    #pragma unroll
                    for (int i = 0; i < MC; ++i) {
                        const int r = lane + 32 * i, d = j + c;
                        if (r < m) Y[r + (size_t)c * m] = (r == d) ? T(1)
                                                        : ((r > d) ? a[c][i] : T(0));
                    }
                __syncwarp();
                #pragma unroll
                for (int k = 0; k < COLS; ++k) {          // compact-WY T, warp-local
                    const T tk = tv[k];
                    if (lane == 0) for (int i = 0; i <= k; ++i) Tm[i + (size_t)k * B] = T(0);
                    __syncwarp();
                    if (tk == T(0)) continue;
                    T z[COLS];
                    #pragma unroll
                    for (int i = 0; i < COLS; ++i) z[i] = T(0);
                    #pragma unroll
                    for (int i2 = 0; i2 < MC; ++i2) {
                        const int r = lane + 32 * i2;
                        if (r < m) {
                            const T yk = Y[r + (size_t)k * m];
                            #pragma unroll
                            for (int i = 0; i < COLS; ++i)
                                if (i < k) z[i] += Y[r + (size_t)i * m] * yk;
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < COLS; ++i) {
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) z[i] += __shfl_xor_sync(0xffffffffu, z[i], o);
                    }
                    if (lane == 0) {
                        #pragma unroll
                        for (int i = 0; i < COLS; ++i)
                            if (i < k) Tm[i + (size_t)k * B] = -tk * z[i];
                        for (int i = 0; i < k; ++i) {      // ascending: later entries read earlier
                            T acc = T(0);
                            for (int p = i; p < k; ++p)
                                acc += Tm[i + (size_t)p * B] * Tm[p + (size_t)k * B];
                            Tm[i + (size_t)k * B] = acc;
                        }
                        Tm[k + (size_t)k * B] = tk;
                    }
                    __syncwarp();
                }
    };

    // Apply (Y_jb, T_jb) to this warp's own columns. Warp-local, no reduction.
    auto do_update = [&](int ujb, const T* Yb) {
        const T* Y = Yb;
        const T* Tm = Tout + (size_t)ujb * COLS * COLS;
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    T wv[COLS];
                    #pragma unroll
                    for (int i = 0; i < COLS; ++i) wv[i] = T(0);
                    #pragma unroll
                    for (int i2 = 0; i2 < MC; ++i2) {
                        const int r = lane + 32 * i2;
                        if (r < m) {
                            const T av = a[c][i2];
                            #pragma unroll
                            for (int i = 0; i < COLS; ++i) wv[i] += Y[r + (size_t)i * m] * av;
                        }
                    }
                    #pragma unroll
                    for (int i = 0; i < COLS; ++i) {
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) wv[i] += __shfl_xor_sync(0xffffffffu, wv[i], o);
                    }
                    T wt[COLS];
                    #pragma unroll
                    for (int i = 0; i < COLS; ++i) {
                        T acc = T(0);
                        #pragma unroll
                        for (int p = 0; p < COLS; ++p)
                            if (p <= i) acc += Tm[p + (size_t)i * B] * wv[p];
                        wt[i] = acc;
                    }
                    #pragma unroll
                    for (int i2 = 0; i2 < MC; ++i2) {
                        const int r = lane + 32 * i2;
                        if (r < m) {
                            T acc = T(0);
                            #pragma unroll
                            for (int i = 0; i < COLS; ++i) acc += Y[r + (size_t)i * m] * wt[i];
                            a[c][i2] -= acc;
                        }
                    }
                }
    };

    // ---- LOOK-AHEAD --------------------------------------------------------------
    // Without it every warp waits at the barrier while one warp factors its panel, and
    // Nsight Compute puts 77% of this kernel's stall samples on barriers for exactly
    // that (302296 of ~391000, against 32787 'wait' and 51501 scoreboard). So panel
    // jb+1 is factored by ITS owner while every later warp is still running the update
    // for jb. The owner does its own update first -- it needs Y_jb applied to its
    // columns before it can factor -- which is Algorithm 1 lines 10-12 one level down.
    // Y is double buffered for this; T already is, each block column owning its own
    // slot of the output array.
    T* Yc = Y;
    T* Yn = Y + (size_t)m * COLS;
    if (warp == 0 && m > 0 && n > 0) do_panel(0, Yc);
    __syncthreads();

    for (int jb = 0; jb + 1 < NW; ++jb) {
        if (jb * COLS >= m || jb * COLS >= n) break;
        const int nxt = jb + 1;
        if (warp == nxt) {                 // near: update my columns, then factor them
            do_update(jb, Yc);
            if (nxt * COLS < m && nxt * COLS < n) do_panel(nxt, Yn);
        } else if (warp > nxt) {           // far: overlapped with the panel above
            do_update(jb, Yc);
        }
        __syncthreads();
        T* sw = Yc; Yc = Yn; Yn = sw;

    }

    #pragma unroll
    for (int c = 0; c < COLS; ++c)
        #pragma unroll
        for (int i = 0; i < MC; ++i) {
            const int r = lane + 32 * i, gc = warp * COLS + c;
            if (r < m && gc < n) Aout[r + (size_t)gc * m] = a[c][i];
        }
}

// Workspace, in words -- the whole of it, there is nothing else.
//
// Y is scratch: the broadcast staging. It is DOUBLE buffered, because look-ahead has the
// owner of panel jb+1 writing its Y while the far update of jb is still reading Y_jb, so
// this is 2*m*b and not m*b. (Sizing it at m*b puts the second buffer on top of the T
// output array; that corrupts T rather than failing loudly, so it is worth stating.)
//
// The T array is an OUTPUT, one b x b per block column, n*b words, and must survive the
// call -- it is Algorithm 1's {T} in O(nb).
TQR_HD inline size_t leaf_broadcast_words(int m, int b) { return (size_t)2 * m * b; }
TQR_HD inline size_t leaf_tfactor_words(int n, int b)   { return (size_t)n * b; }
TQR_HD inline size_t leaf_workspace_words(int m, int n, int b) {
    return leaf_broadcast_words(m, b) + leaf_tfactor_words(n, b);
}

// Doubles of register file per lane the matrix itself occupies. The leaf is register
// bound, so this is the sizing test that matters; 2x this many registers per thread go
// to data, and NW*32*(that + working set) must stay under 65536 for one block per SM.
TQR_HD inline int leaf_regs_per_lane(int m, int cols) {
    return cols * ((m + 31) / 32);
}

#endif  // __CUDACC__

}  // namespace tqr
