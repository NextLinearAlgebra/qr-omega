// TQR-Omega correctness and performance gate.
// It reports separate LAPACK-normalized reconstruction and orthogonality metrics,
// rebuilding Q from stored transforms to verify their ordered composition.
#include "tqr_omega.cuh"
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <climits>
#include <vector>
#include <string>
#include <cstring>

// --probe measures global and cluster barrier latency by participant count.
__global__ __launch_bounds__(512) void k_bar_gmem(unsigned* bar, int nbl, int iters) {
    for (int i = 0; i < iters; ++i) lrqr::gmem_barrier(bar, nbl);
}
__global__ __launch_bounds__(512) void k_bar_cluster(int iters) {
    namespace cg = cooperative_groups;
    for (int i = 0; i < iters; ++i) cg::this_cluster().sync();
}
static void barrier_cost_table() {
    const int iters = 2000;
    unsigned* bar = nullptr;
    CUDA_CHECK(cudaMalloc(&bar, 2 * sizeof(unsigned)));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    std::printf("\n=== barrier cost vs participants (us per barrier) ===\n");
    std::printf("  %10s %14s\n", "blocks", "gmem_us");
    for (int nbl : {2, 4, 8, 16, 32, 66, 132}) {
        CUDA_CHECK(cudaMemset(bar, 0, 2 * sizeof(unsigned)));
        k_bar_gmem<<<nbl, 512>>>(bar, nbl, 50);            // warm
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(bar, 0, 2 * sizeof(unsigned)));
        CUDA_CHECK(cudaEventRecord(e0));
        k_bar_gmem<<<nbl, 512>>>(bar, nbl, iters);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  %10d %14.3f\n", nbl, ms * 1e3 / iters);
    }
    // cluster barrier at the sizes a panel would actually use
    std::printf("  %10s %14s\n", "clusterCTAs", "dsm_us");
    for (int cs : {2, 4, 8, 16}) {
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = dim3(cs, 1, 1); cfg.blockDim = dim3(512, 1, 1);
        cfg.dynamicSmemBytes = 0; cfg.stream = 0;
        // cluster size 16 is non-portable and must be opted into on the FUNCTION,
        // not the launch (cudaFuncAttribute..., as tqr_omega.cuh:1549 does).
        CUDA_CHECK(cudaFuncSetAttribute((const void*)k_bar_cluster,
                   cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        cudaLaunchAttribute at[1]{};
        at[0].id = cudaLaunchAttributeClusterDimension;
        at[0].val.clusterDim.x = cs; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.attrs = at; cfg.numAttrs = 1;
        if (cudaLaunchKernelEx(&cfg, k_bar_cluster, 50) != cudaSuccess) {
            std::printf("  %10d %14s\n", cs, "unsupported"); cudaGetLastError(); continue;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaLaunchKernelEx(&cfg, k_bar_cluster, iters));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        std::printf("  %10d %14.3f\n", cs, ms * 1e3 / iters);
    }
    std::printf("  prize: a two-level barrier pays dsm(16) + gmem(nbl/16) per round\n"
                "  instead of gmem(nbl). Compare that sum against the gmem(132) row.\n");
    CUDA_CHECK(cudaFree(bar));
}


using lrqr::gemm;
using lrqr::trmm;
using tqr::geam;

template <typename Real>
__global__ void k_omega_triu(const Real* A, size_t lda, Real* R, size_t ldr, int k, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= n || r >= k) return;
    R[(size_t)c * ldr + r] = (r <= c) ? A[(size_t)c * lda + r] : Real(0);
}
// Canonical-sign R in double precision. QR factors are equivalent up to row signs,
// so normalize each row to a nonnegative diagonal before comparing tiers.
template <typename Src>
__global__ void k_canon_triu(const Src* A, size_t lda, double* R, size_t ldr,
                             int k, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= n || r >= k) return;
    double v = (r <= c) ? (double)A[(size_t)c * lda + r] : 0.0;
    if ((double)A[(size_t)r * lda + r] < 0.0) v = -v;
    R[(size_t)c * ldr + r] = v;
}

template <typename Real>
__global__ void k_omega_eye(Real* Q, size_t ldq, int m, int k) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= k || r >= m) return;
    Q[(size_t)c * ldq + r] = (r == c) ? Real(1) : Real(0);
}

template <typename Real>
static double frob(cublasHandle_t h, const Real* X, size_t n) {
    double out = 0;
    if constexpr (std::is_same<Real, double>::value) {
        double r; CUBLAS_CHECK(cublasDnrm2(h, (int)std::min(n, (size_t)INT_MAX), X, 1, &r)); out = r;
    } else {
        float r; CUBLAS_CHECK(cublasSnrm2(h, (int)std::min(n, (size_t)INT_MAX), X, 1, &r)); out = r;
    }
    return out;
}

// Best-of-repetitions cuSOLVER baseline on the identical input.
template <typename Real>
static double cusolver_ms(int m, int n, size_t lda, const Real* A0, Real* A, int reps) {
    cusolverDnHandle_t h; if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) return -1;
    int lwork = 0;
    Real* tau = nullptr; Real* work = nullptr; int* info = nullptr;
    const int k = std::min(m, n);
    CUDA_CHECK(cudaMalloc(&tau, (size_t)k * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    if constexpr (std::is_same<Real, double>::value)
        cusolverDnDgeqrf_bufferSize(h, m, n, A, (int)lda, &lwork);
    else
        cusolverDnSgeqrf_bufferSize(h, m, n, A, (int)lda, &lwork);
    CUDA_CHECK(cudaMalloc(&work, (size_t)std::max(1, lwork) * sizeof(Real)));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemcpy(A, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        if constexpr (std::is_same<Real, double>::value)
            cusolverDnDgeqrf(h, m, n, A, (int)lda, tau, work, lwork, info);
        else
            cusolverDnSgeqrf(h, m, n, A, (int)lda, tau, work, lwork, info);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, (double)ms);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(tau); cudaFree(work); cudaFree(info); cusolverDnDestroy(h);
    return best;
}

// Probabilistic residual check for sizes where explicit-Q validation is impractical.
// It applies Q and Q^T to random vectors without materializing Q.
template <typename Real, int NB>
static int sampled_residual(tqr::Omega<Real, NB>& om, cublasHandle_t hb, cudaStream_t st,
                            const Real* A, const Real* A0, size_t lda, int m, int n,
                            int nvec, double& out_r1, double& out_r2) {
    const int k = std::min(m, n);
    const double eps = std::is_same<Real, double>::value ? 1.11e-16 : 5.96e-8;
    Real *x = nullptr, *y = nullptr, *z = nullptr, *u = nullptr;
    CUDA_CHECK(cudaMalloc(&x, (size_t)std::max(m, n) * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&y, (size_t)m * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&z, (size_t)m * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&u, (size_t)m * sizeof(Real)));

    // ||A0||_F, chunked: m*n can approach INT_MAX and nrm2 takes an int length.
    double nA2 = 0;
    {
        const int chunk = std::max(1, (int)(1u << 28) / std::max(1, m));
        for (int c = 0; c < n; c += chunk) {
            const int cc = std::min(chunk, n - c);
            double r = frob(hb, A0 + (size_t)c * lda, (size_t)cc * lda);
            nA2 += r * r;
        }
    }
    const double nA = std::sqrt(nA2);
    const double denom = eps * (double)std::max(m, n);
    out_r1 = 0; out_r2 = 0;

    std::mt19937_64 rng(9001 + m + n);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Real> hx((size_t)std::max(m, n));

    for (int t = 0; t < nvec; ++t) {
        for (auto& v : hx) v = (Real)nd(rng);
        CUDA_CHECK(cudaMemcpyAsync(x, hx.data(), (size_t)n * sizeof(Real),
                                   cudaMemcpyHostToDevice, st));
        // y[0:k] = R x  =  triu(A[:k,:k]) x[:k]  +  A[:k, k:n] x[k:n]
        CUDA_CHECK(cudaMemcpyAsync(y, x, (size_t)k * sizeof(Real),
                                   cudaMemcpyDeviceToDevice, st));
        trmm(hb, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
             CUBLAS_DIAG_NON_UNIT, k, 1, Real(1), A, (int)lda, y, k);
        if (n > k)
            gemm(hb, false, CUBLAS_OP_N, CUBLAS_OP_N, k, 1, n - k, Real(1),
                 A + (size_t)k * lda, (int)lda, x + k, n - k, Real(1), y, k);
        // z = Q [y;0]
        CUDA_CHECK(cudaMemsetAsync(z, 0, (size_t)m * sizeof(Real), st));
        CUDA_CHECK(cudaMemcpyAsync(z, y, (size_t)k * sizeof(Real),
                                   cudaMemcpyDeviceToDevice, st));
        om.apply_Q(A, z, m, 1, /*trans=*/false);
        // z -= A0 x
        gemm(hb, false, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, n, Real(-1),
             A0, (int)lda, x, n, Real(1), z, m);
        const double r1 = frob(hb, z, (size_t)m) / (nA * frob(hb, x, (size_t)n) * denom + 1e-300);
        out_r1 = std::max(out_r1, r1);

        // orthogonality: || Q^T (Q u) - u || / (eps max(m,n)).
        // Q is m x k with k <= m, so Q Q^T is a PROJECTOR, not the identity -- but
        // Q^T Q = I is what orthonormal columns mean, and applying Q then Q^T to a
        // vector composes exactly that when the vector is first projected. Taking
        // u = Q w for a random w keeps the test inside the column space, so the
        // identity being checked is the true one rather than a projector confused
        // for it.
        for (auto& v : hx) v = (Real)nd(rng);
        CUDA_CHECK(cudaMemcpyAsync(u, hx.data(), (size_t)m * sizeof(Real),
                                   cudaMemcpyHostToDevice, st));
        if (m > k) CUDA_CHECK(cudaMemsetAsync(u + k, 0, (size_t)(m - k) * sizeof(Real), st));
        om.apply_Q(A, u, m, 1, /*trans=*/false);          // u := Q w, in range(Q)
        CUDA_CHECK(cudaMemcpyAsync(z, u, (size_t)m * sizeof(Real),
                                   cudaMemcpyDeviceToDevice, st));
        om.apply_Q(A, z, m, 1, /*trans=*/true);           // z := Q^T u
        om.apply_Q(A, z, m, 1, /*trans=*/false);          // z := Q Q^T u  == u if orthonormal
        geam(hb, CUBLAS_OP_N, CUBLAS_OP_N, m, 1, Real(1), z, m,
             Real(-1), u, m, z, m);
        const double nu_ = frob(hb, u, (size_t)m);
        out_r2 = std::max(out_r2, frob(hb, z, (size_t)m) / (nu_ * denom + 1e-300));
    }
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaFree(x); cudaFree(y); cudaFree(z); cudaFree(u);
    return 0;
}

// Factor the SAME input in fp64 and in the float tier, then compare canonical R.
// Returns relative Frobenius and relative max-elementwise disagreement.
template <int NB>
static void mixed_tier_R(const float* Afac, size_t lda, int m, int n,
                         const float* A0f, double& relF, double& relMax) {
    const int k = std::min(m, n);
    cudaStream_t st; CUDA_CHECK(cudaStreamCreate(&st));
    cublasHandle_t hb; CUBLAS_CHECK(cublasCreate(&hb)); CUBLAS_CHECK(cublasSetStream(hb, st));
    double *Ad = nullptr, *Rref = nullptr, *Rtier = nullptr;
    CUDA_CHECK(cudaMalloc(&Ad,    lda * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Rref,  (size_t)k * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Rtier, (size_t)k * n * sizeof(double)));
    {   // promote the ORIGINAL input, so the reference factors the same matrix
        std::vector<float> h((size_t)m * n);
        CUDA_CHECK(cudaMemcpy(h.data(), A0f, lda * n * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<double> hd(h.begin(), h.end());
        CUDA_CHECK(cudaMemcpy(Ad, hd.data(), lda * n * sizeof(double), cudaMemcpyHostToDevice));
    }
    tqr::Omega<double, 64> ref;
    ref.tpanelSec = tqr::probe_panel_s<double, 64>(m);
    ref.create(m, n, lda, st);
    ref.geqrf(Ad);
    CUDA_CHECK(cudaStreamSynchronize(st));

    dim3 tb(16, 16), gr((n + 15) / 16, (k + 15) / 16);
    k_canon_triu<double><<<gr, tb, 0, st>>>(Ad,   lda, Rref,  k, k, n);
    k_canon_triu<float> <<<gr, tb, 0, st>>>(Afac, lda, Rtier, k, k, n);
    const double nref = frob(hb, Rref, (size_t)k * n);
    int imax = 0; CUBLAS_CHECK(cublasIdamax(hb, k * n, Rref, 1, &imax));
    double refmax = 0;
    CUDA_CHECK(cudaMemcpy(&refmax, Rref + (imax - 1), sizeof(double), cudaMemcpyDeviceToHost));
    refmax = std::fabs(refmax);
    const double one = 1.0, mone = -1.0;
    CUBLAS_CHECK(cublasDaxpy(hb, k * n, &mone, Rref, 1, Rtier, 1));   // Rtier -= Rref
    const double ndif = frob(hb, Rtier, (size_t)k * n);
    CUBLAS_CHECK(cublasIdamax(hb, k * n, Rtier, 1, &imax));
    double dmax = 0;
    CUDA_CHECK(cudaMemcpy(&dmax, Rtier + (imax - 1), sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(st));
    relF   = (nref   > 0) ? ndif / nref : 0.0;
    relMax = (refmax > 0) ? std::fabs(dmax) / refmax : 0.0;
    (void)one;
    ref.destroy(); cublasDestroy(hb);
    CUDA_CHECK(cudaFree(Ad)); CUDA_CHECK(cudaFree(Rref)); CUDA_CHECK(cudaFree(Rtier));
    CUDA_CHECK(cudaStreamDestroy(st));
}

template <typename Real, int NB>
static int run_cell(int m, int n, int reps, bool verbose, bool check = true) {
    // dqrt01 normalises by the WORKING precision's eps. A tf32 trailing update on
    // fp32 storage is not fp32 arithmetic, so scoring it at fp32 eps would fail it
    // for being tf32 rather than for being wrong. Each tier is scored against the
    // eps of the arithmetic it actually ran, which is what 10:S8.2 means by a
    // precision contract -- and the tier is printed beside the number so the two
    // can never be read apart.
    const tqr::Tier tr = std::is_same<Real, double>::value
                       ? tqr::Tier::FP64 : tqr::tier_from_env();
    // 10:S8.2's evidence column, verbatim: TF32 is scored at u_TF32 = 4.88e-4,
    // while 3xTF32's required evidence is "residual ... in fp32 order, historically
    // ~1e-6" -- the SAME bar as fp32. Scoring 3xTF32 at tf32 eps would let a
    // silently-degraded error-corrected path pass, which is exactly the failure
    // 10:S8.3's no-silent-fallback clause exists to catch.
    const double eps = (tr == tqr::Tier::FP64)   ? 1.11e-16
                     : (tr == tqr::Tier::TF32)   ? 4.88e-4
                                                 : 5.96e-8;    // fp32 AND 3xTF32
    const int k = std::min(m, n);
    const size_t lda = (size_t)m;

    Real *A = nullptr, *A0 = nullptr, *Q = nullptr, *R = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A,  lda * n * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&A0, lda * n * sizeof(Real)));
    if (check) {
        // Q, R and the residual buffer are the gate's cost, not the solver's.
        // Skipping them is what lets a size be TIMED that cannot be CHECKED here.
        CUDA_CHECK(cudaMalloc(&Q,  lda * k * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&R,  (size_t)k * n * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&C,  lda * n * sizeof(Real)));
    }

    std::vector<Real> host((size_t)m * n);
    std::mt19937_64 rng(12345 + m * 7919ull + n);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& x : host) x = (Real)nd(rng);
    CUDA_CHECK(cudaMemcpy(A0, host.data(), lda * n * sizeof(Real), cudaMemcpyHostToDevice));

    // The frontier stream is created HERE, so it carries S6.7's top priority.
    int prLo = 0, prHi = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prLo, &prHi));
    const char* pe = getenv("TQR_PRIO");
    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking,
               (!pe || atoi(pe) != 0) ? prHi : 0));
    // t_panel at each candidate p_r, so the planner can rank the frontier/far
    // device split instead of defaulting to "panel takes everything".
    int smc = 0; { cudaDeviceProp pp{}; int d = 0; CUDA_CHECK(cudaGetDevice(&d));
                   CUDA_CHECK(cudaGetDeviceProperties(&pp, d)); smc = pp.multiProcessorCount; }
    // Slots 0-3 probe the gmem carrier at decreasing width; slots 4-5 probe the
    // CLUSTER carrier. Both are measured so T_hat ranks the carrier, rather than
    // the cluster being taken whenever it happens to be legal.
    const int  prCand[6] = {smc, smc / 2, smc / 4, smc / 8, 16, 8};
    const bool clCand[6] = {false, false, false, false, true, true};
    double tprTab[6];
    for (int i = 0; i < 6; ++i)
        tprTab[i] = tqr::probe_panel_s<Real, NB>(m, 3, std::max(1, prCand[i]), clCand[i]);
    tqr::Omega<Real, NB> om;
    om.tpanelPrTab = tprTab;
    om.tpanelSec = tprTab[0];
    om.create(m, n, lda, st);
    if (verbose) om.emit("plan (pre-factor)");

    // ---- factor ----
    CUDA_CHECK(cudaMemcpyAsync(A, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        CUDA_CHECK(cudaMemcpyAsync(A, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaEventRecord(e0, st));
        om.geqrf(A);
        CUDA_CHECK(cudaEventRecord(e1, st));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::min(best, ms);
    }

    const double cus = cusolver_ms<Real>(m, n, lda, A0, A, reps);
    if (!check) {
        // cuSOLVER overwrote A; redo ours so the sampled residual scores OUR result.
        CUDA_CHECK(cudaMemcpyAsync(A, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        om.geqrf(A);
        cublasHandle_t hb; CUBLAS_CHECK(cublasCreate(&hb));
        CUBLAS_CHECK(cublasSetStream(hb, st));
        double r1 = 0, r2 = 0;
        sampled_residual<Real, NB>(om, hb, st, A, A0, lda, m, n, 2, r1, r2);
        cublasDestroy(hb);
        const bool ok = (r1 < 30.0) && (r2 < 30.0) && std::isfinite(r1) && std::isfinite(r2);
        const double F = tqr::Planner::flops(m, n);
        std::printf("%6d %6d  %-6s  %8.2f ms %8.1f GF/s   cuSOLVER %8.2f ms %8.1f GF/s"
                    "   speedup %5.2fx   sampled r1=%-10.4g r2=%-10.4g %s\n",
                    m, n, tqr::tier_name(tr),
                    best, F / (best * 1e6), cus, F / (cus * 1e6), cus / best,
                    r1, r2, ok ? "OK" : "**RESIDUAL FAIL**");
        // Panel share, from the measurement already taken. 10:S9.1 wants the
        // non-far budget by category and S11 Phase 4 says reduce the largest one
        // first; this is that number for the panel category, computed as
        // K0 * t_panel(chosen p_r) / wall -- no profiler required, and it tracks
        // the carrier the plan actually selected.
        {
            const int K0 = (std::min(m, n) + om.b - 1) / om.b;
            const double panel_ms = K0 * om.tpanelSec * 1e3;
            std::printf("            panel: K0=%d x t_panel=%.1fus = %.2f ms = %.0f%% of wall"
                        "   [panel=%s carrier=%s p_r=%d]\n",
                        K0, om.tpanelSec * 1e6, panel_ms, 100.0 * panel_ms / best,
                        om.panel_algo(), om.panelCluster ? "cluster" : "gmem", om.panelPr);
        }
        if (verbose) om.emit("plan");
        om.destroy(); CUDA_CHECK(cudaStreamDestroy(st));
        CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(A0));
        return ok ? 0 : 1;
    }
    // A was overwritten by cuSOLVER; redo our factorization for the gate.
    CUDA_CHECK(cudaMemcpyAsync(A, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    om.geqrf(A);
    CUDA_CHECK(cudaStreamSynchronize(st));

    // ---- A6 determinism probe (TQR_DETERM=1) ----
    //
    // 31:246 (A6) requires "deterministic W reductions" and 31:434's rung-1 Combine is
    // "stack-HH R, DETERMINISTIC W". The domino accumulates every column's sig and
    // dslot with device-scope atomicAdd over all row slabs (lrqr.cuh:1491, :1505,
    // :1723, :1733), and fp64 atomicAdd has no defined summation order. The comment at
    // lrqr.cuh:1392 calls the fp64 shadows "deterministic atomicAdd", which only holds
    // where the partials sum exactly -- not for the fp64 tier.
    //
    // Printed residuals carry four digits and cannot see this. Factor the SAME matrix
    // twice in the same process and compare raw bits; report the first differing
    // element and how many differ. This decides whether A6 is met, before any
    // performance argument about replacing the atomics is made.
    if (getenv("TQR_DETERM")) {
        Real* A2 = nullptr;
        CUDA_CHECK(cudaMalloc(&A2, lda * n * sizeof(Real)));
        CUDA_CHECK(cudaMemcpyAsync(A2, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        om.geqrf(A2);
        CUDA_CHECK(cudaStreamSynchronize(st));
        std::vector<Real> h1((size_t)lda * n), h2((size_t)lda * n);
        CUDA_CHECK(cudaMemcpy(h1.data(), A,  lda * n * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h2.data(), A2, lda * n * sizeof(Real), cudaMemcpyDeviceToHost));
        size_t ndiff = 0, first = 0; double maxrel = 0;
        for (size_t i = 0; i < h1.size(); ++i) {
            if (std::memcmp(&h1[i], &h2[i], sizeof(Real)) != 0) {
                if (!ndiff) first = i;
                ++ndiff;
                const double a = (double)h1[i], b = (double)h2[i];
                const double d = std::abs(a - b) / std::max(1e-300, std::abs(a));
                if (d > maxrel) maxrel = d;
            }
        }
        std::printf("            [A6 determinism] %s : %zu of %zu words differ bitwise "
                    "run-to-run (first idx %zu, max rel %.3g)  panel=%s\n",
                    ndiff ? "NONDETERMINISTIC" : "bitwise identical",
                    ndiff, h1.size(), first, maxrel, om.panel_algo());
        CUDA_CHECK(cudaFree(A2));
    }

    // ---- rebuild Q and score the dqrt01 criteria ----
    cublasHandle_t hb; CUBLAS_CHECK(cublasCreate(&hb)); CUBLAS_CHECK(cublasSetStream(hb, st));
    dim3 tb(16, 16);
    k_omega_eye<Real><<<dim3((k + 15) / 16, (m + 15) / 16), tb, 0, st>>>(Q, lda, m, k);
    om.orgqr(A, Q, lda, k);
    k_omega_triu<Real><<<dim3((n + 15) / 16, (k + 15) / 16), tb, 0, st>>>(A, lda, R, k, k, n);

    // C := Q*R - A0
    CUDA_CHECK(cudaMemcpyAsync(C, A0, lda * n * sizeof(Real), cudaMemcpyDeviceToDevice, st));
    gemm(hb, false, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, Real(1), Q, (int)lda, R, k,
         Real(-1), C, (int)lda);
    const double nA  = frob(hb, A0, lda * n);
    const double nR1 = frob(hb, C,  lda * n);

    // Qt*Q - I  (k x k)
    Real* G = nullptr; CUDA_CHECK(cudaMalloc(&G, (size_t)k * k * sizeof(Real)));
    k_omega_eye<Real><<<dim3((k + 15) / 16, (k + 15) / 16), tb, 0, st>>>(G, k, k, k);
    gemm(hb, false, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m, Real(1), Q, (int)lda, Q, (int)lda,
         Real(-1), G, k);
    const double nR2 = frob(hb, G, (size_t)k * k);
    CUDA_CHECK(cudaStreamSynchronize(st));

    const double denom = eps * (double)std::max(m, n);
    const double RES1 = nR1 / (nA * denom);
    const double RES2 = nR2 / denom;
    bool pass = (RES1 < 30.0) && (RES2 < 30.0) && std::isfinite(RES1) && std::isfinite(RES2);

    const double F = tqr::Planner::flops(m, n);
    // ---- 10:S8.2 mixed-tier evidence: canonical-sign R against fp64 ----
    // Required IN ADDITION to the residual, and it is the only check here that can
    // discriminate the float tiers from each other: 3xTF32 must land in fp32 order
    // while plain TF32 lands in tf32 order. If an error-corrected path silently
    // degraded to ordinary TF32, the residual and the orthogonality would both
    // still look fine -- and F_EC/F_far would still read 1.0000.
    double relF = -1, relMax = -1;
    bool Rok = true;
    if constexpr (!std::is_same<Real, double>::value) {
        mixed_tier_R<NB>((const float*)A, lda, m, n, (const float*)A0, relF, relMax);
        // Thresholds are the "required discriminating evidence" column of 10:S8.2:
        // fp32 and 3xTF32 agree in fp32 order (~1e-6 historically), TF32 in its own
        // band (~5e-4..2e-3). Set ~100x above the historical value so the gate
        // catches a tier that fell back or degraded, not ordinary rounding spread.
        const double thr = (tr == tqr::Tier::TF32) ? 1e-2 : 1e-4;
        Rok = (relF < thr) && std::isfinite(relF);
    }
    pass = pass && Rok;
    std::printf("%6d %6d  %-6s  RES1=%-11.4g RES2=%-11.4g  %7.2f ms %7.1f GF/s"
                "  cuSOLVER %7.2f ms  %5.2fx  dR_F=%-10.3g dR_max=%-10.3g %s\n",
                m, n, tqr::tier_name(tr),
                RES1, RES2, best, F / (best * 1e6), cus, cus / best,
                relF, relMax, pass ? "PASS" : (Rok ? "FAIL" : "FAIL[R vs fp64]"));
    // State the configuration that ACTUALLY RAN, per cell. Two green gates in this
    // project meant less than they looked because the code under test was silently
    // not the code running: the wfar_super>0 blindness, and TSQR's "16/19" where the
    // 16 passes were the domino. A pass is only evidence about what executed.
    std::printf("            ran: b=%d nu=[%d,%d,%d] panel=%s carrier=%s p_r=%d plan=%s\n",
                om.b, om.nu1, om.nu2, om.nu3, om.panel_algo(),
                om.panelCluster ? "cluster" : "gmem", om.panelPr,
                om.rec.plan_enumerated ? "enumerated" : "FALLBACK");
    if (verbose) om.emit("plan");

    CUDA_CHECK(cudaFree(G));
    cublasDestroy(hb);
    om.destroy();
    CUDA_CHECK(cudaStreamDestroy(st));
    CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(A0)); CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(R)); CUDA_CHECK(cudaFree(C));
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    int only_m = 0, only_n = 0, reps = 1, bisect = 0, bsel = 64;
    bool verbose = false, speed = false, probe = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) only_m = atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) only_n = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "-v") verbose = true;
        // Bisect on the panel count: for a fixed m, sweep n in leaf-panel steps.
        // The first failing n names the panel index where the factorization first
        // goes wrong, which a pass/fail on one size cannot.
        else if (a == "--bisect" && i + 1 < argc) bisect = atoi(argv[++i]);
        else if (a == "--speed") speed = true;
        // 10:S6.2: b comes from the KERNEL's admissible register/shared set, so the
        // family is whatever widths the panel kernel is instantiated at. Sweeping it
        // is the only way the enumeration over b stops being degenerate -- and b is
        // the one knob that moves BOTH open problems: it halves the serial panel
        // chain and it doubles kappa*n*b, which is what caps nu_3 (31:409).
        else if (a == "--b" && i + 1 < argc) bsel = atoi(argv[++i]);
        // --autob: let the planner pick b from the kernel's admissible set using
        // the MEASURED panel probe, which is the only spec-legal way to choose it
        // (10:S4.3 forbids size thresholds and mandates measured rate tables).
        else if (a == "--autob") bsel = 0;
        // --probe: print the RAW measured panel cost per leaf width, with no model
        // on top. The b=256 program rests entirely on one claim -- that panel cost
        // is sub-linear in b -- which was read off cuSOLVER's panel, not ours. This
        // measures OUR kernel: if t_panel(2b)/t_panel(b) == 2, then the chain
        // K0*t_panel = (n/b)*(c*b) = c*n is invariant and b is not a lever at all.
        else if (a == "--probe") probe = true;
    }
    int bad = 0;
    if (probe) {
        // The chain is K0*t_panel = (n/b)*t_panel(b). Measured t_panel ~ C*b^1.22, so
        // the CHAIN scales as b^0.22 -- it grows with b, and the same exponent says
        // SMALLER b shortens it. The earlier sweep went upward only (64 -> 128 -> 256),
        // found it worse, and closed the "b program" on that. b < 64 had never been
        // instantiated. cuSOLVER's own kernel at this cell is templated on 8 columns.
        std::printf("=== raw panel probe: t_panel(b, m), and the CHAIN (n/b)*t_panel ===\n");
        std::printf("  %8s %10s %10s %10s %10s   %9s %9s %9s\n",
                    "m", "t32_us", "t64_us", "t128_us", "t256_us",
                    "chain32", "chain64", "chain128");
        for (int mm : {2048, 4096, 8192, 16384, 32768}) {
            const double a32  = tqr::probe_panel_s<double,  32>(mm, 5) * 1e6;
            const double a64  = tqr::probe_panel_s<double,  64>(mm, 5) * 1e6;
            const double a128 = tqr::probe_panel_s<double, 128>(mm, 5) * 1e6;
            const double a256 = tqr::probe_panel_s<double, 256>(mm, 5) * 1e6;
            std::printf("  %8d %10.1f %10.1f %10.1f %10.1f   %9.0f %9.0f %9.0f\n",
                        mm, a32, a64, a128, a256,
                        (mm / 32.0) * a32, (mm / 64.0) * a64, (mm / 128.0) * a128);
        }
        std::printf("  the chain columns are what the frontier actually pays; the\n"
                    "  smallest one wins, and the planner picks it from this probe.\n");
        // the carrier table the planner actually ranks on, printed raw
        std::printf("\n=== t_panel per (p_r, carrier) slot, b=64 -- what the planner ranks ===\n");
        std::printf("  %8s %10s %10s %10s %10s %10s %10s\n", "m",
                    "g/132", "g/66", "g/33", "g/16", "CL/16", "CL/8");
        for (int mm : {2048, 4096, 8192, 16384, 32768}) {
            const int  prC[6] = {132, 66, 33, 16, 16, 8};
            const bool clC[6] = {false, false, false, false, true, true};
            std::printf("  %8d", mm);
            for (int i = 0; i < 6; ++i)
                std::printf(" %10.1f",
                            tqr::probe_panel_s<double, 64>(mm, 5, prC[i], clC[i]) * 1e6);
            std::printf("\n");
        }
        barrier_cost_table();
        return 0;
    }
    if (speed && bsel == 0) {
        std::printf("=== speed, b chosen by the planner from the measured probe ===\n");
        const int sizes[] = {2048, 4096, 8192, 16384, 32768};
        auto go = [&](int mm, int nn) {
            // 10:S6.2: b comes from the KERNEL's admissible register/shared set.
            // That set is whatever widths the panel kernel is instantiated at, so
            // enumerating b means calling the planner once per instantiation.
            // NB=256 is added because the profile says the panel IS the dominant
            // category at small/mid n (56% at 2048^2, 47% at 4096^2) and its cost
            // is sub-linear in b -- cuSOLVER's 256-wide panel runs 631us against
            // our 128-wide at 389us, i.e. 2x the columns for 1.6x the time.
            const double t32  = tqr::predict_That<double,  32>(mm, nn);
            const double t64  = tqr::predict_That<double,  64>(mm, nn);
            const double t128 = tqr::predict_That<double, 128>(mm, nn);
            const double t256 = tqr::predict_That<double, 256>(mm, nn);
            const double tb = std::min(std::min(t32, t64), std::min(t128, t256));
            const int bpick = (tb == t256) ? 256 : (tb == t128) ? 128 : (tb == t32) ? 32 : 64;
            std::printf("  [b-select] m=%d n=%d  That(32)=%.6g That(64)=%.6g That(128)=%.6g "
                        "That(256)=%.6g -> b=%d\n", mm, nn, t32, t64, t128, t256, bpick);
            if (bpick == 256)      bad += run_cell<double, 256>(mm, nn, reps, verbose, false);
            else if (bpick == 128) bad += run_cell<double, 128>(mm, nn, reps, verbose, false);
            else if (bpick == 32)  bad += run_cell<double,  32>(mm, nn, reps, verbose, false);
            else                   bad += run_cell<double,  64>(mm, nn, reps, verbose, false);
        };
        // --m/--n restrict the sweep to one cell. Without this, an A/B arm whose
        // carrier is not legal at every shape aborts partway and the whole arm
        // silently produces nothing -- which is what happened to the first tsqr
        // speed run: it died on the tall cells and printed an empty arm.
        if (only_m && only_n) { go(only_m, only_n); }
        else {
            for (int sz : sizes) go(sz, sz);
            for (int sz : sizes) go(sz * 4, sz / 2);
        }
        std::printf("=== %s : %d timed cell(s) failed the sampled residual ===\n",
                    bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (speed) {
        std::printf("=== speed only (no Q/R gate) : TQR-Omega vs cuSOLVER, b=%d ===\n", bsel);
        const int sizes[] = {2048, 4096, 8192, 16384, 32768};
        // --m/--n restrict the sweep to one cell; same reason as the autob path above.
        auto one = [&](int mm, int nn) {
            bad += (bsel == 128) ? run_cell<double, 128>(mm, nn, reps, false, false)
                                 : run_cell<double,  64>(mm, nn, reps, false, false);
        };
        if (only_m && only_n) { one(only_m, only_n); }
        else {
            for (int sz : sizes) one(sz, sz);
            for (int sz : sizes) one(sz * 4, sz / 2);
        }
        std::printf("=== %s : %d timed cell(s) failed the sampled residual ===\n",
                    bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (bsel == 32) {   // --b 32 selects the FLOAT instantiation (tier from TQR_TIER)
        std::printf("=== TQR-Omega gate, float storage, tier=%s ===\n",
                    tqr::tier_name(tqr::tier_from_env()));
        const int cells[][2] = {
            {256,256},{512,512},{1024,1024},{2048,2048},{4096,4096},
            {4096,1024},{8192,512},{512,4096},{1024,2048},{2048,3072},{3072,2048},
        };
        for (auto& c : cells) bad += run_cell<float, 128>(c[0], c[1], reps, verbose);
        std::printf("=== %s : %d cell(s) failed ===\n", bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (bsel == 256) {
        std::printf("=== TQR-Omega gate, b=256 ===\n");
        const int cells[][2] = {
            {512,512},{1024,1024},{2048,2048},{4096,4096},
            {4096,1024},{8192,512},{512,4096},{1024,2048},{2048,3072},{3072,2048},
        };
        for (auto& c : cells) bad += run_cell<double, 256>(c[0], c[1], reps, verbose);
        std::printf("=== %s : %d cell(s) failed ===\n", bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (bsel == 128) {
        std::printf("=== TQR-Omega gate, b=128 ===\n");
        const int cells[][2] = {
            {256,256},{512,512},{1024,1024},{2048,2048},{4096,4096},
            {4096,1024},{8192,512},{512,4096},{1024,2048},{2048,3072},{3072,2048},
        };
        for (auto& c : cells) bad += run_cell<double, 128>(c[0], c[1], reps, verbose);
        std::printf("=== %s : %d cell(s) failed ===\n", bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (bsel == 32) {
        std::printf("=== TQR-Omega gate, b=32 ===\n");
        const int cells32[][2] = {
            {256,256},{512,512},{1024,1024},{2048,2048},{4096,4096},
            {4096,1024},{8192,512},{512,4096},{1024,2048},{2048,3072},{3072,2048},
        };
        for (auto& c : cells32) bad += run_cell<double, 32>(c[0], c[1], reps, verbose);
        std::printf("=== %s : %d cell(s) failed ===\n", bad ? "FAIL" : "PASS", bad);
        return bad;
    }
    if (bisect) {
        std::printf("=== bisect: m=%d, n swept in leaf-panel steps ===\n", bisect);
        for (int nn = 512; nn <= bisect; nn += 512)
            bad += run_cell<double, 64>(bisect, nn, reps, false);
        std::printf("=== bisect done : %d failed ===\n", bad);
        return bad;
    }
    if (only_m && only_n) {
        // Honour --b here too. Without it a single-cell debug of a float tier
        // silently ran fp64 and "passed", which is worse than not running.
        if (bsel == 32)       bad += run_cell<float,  128>(only_m, only_n, reps, true);
        else if (bsel == 256) bad += run_cell<double, 256>(only_m, only_n, reps, true);
        else if (bsel == 128) bad += run_cell<double, 128>(only_m, only_n, reps, true);
        else if (bsel == 32)  bad += run_cell<double,  32>(only_m, only_n, reps, true);
        else                  bad += run_cell<double,  64>(only_m, only_n, reps, true);
        return bad;
    }
    std::printf("=== TQR-Omega gate : dqrt01 criteria, THRESH=30 ===\n");
    // square, tall, wide, and n>kmax -- the geometry class that hid a real defect
    const int cells[][2] = {
        {256, 256}, {512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096},
        {4096, 1024}, {8192, 512}, {16384, 256},
        {512, 4096}, {1024, 2048}, {256, 8192},
        {2048, 3008}, {3008, 2048},
        // bracket around the one cell that failed the first gate, to separate
        // "a size" from "a property of that size"
        {3072, 3072}, {3584, 3584}, {4032, 4032}, {4160, 4160},
        {4096, 4032}, {4032, 4096},
    };
    for (auto& c : cells) bad += run_cell<double, 64>(c[0], c[1], reps, verbose);
    std::printf("=== %s : %d cell(s) failed ===\n", bad ? "FAIL" : "PASS", bad);
    return bad;
}
