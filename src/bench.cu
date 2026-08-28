// LRQR benchmark and validation driver.
// It compares precision tiers with cuSOLVER and reports factorization time,
// reconstruction residual, orthogonality, and a normal-equation error proxy.
// --validate runs the correctness suite across sizes and conditioning levels.
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

// Disable float-only custom WMMA paths so benchmark templates compile for double.
#ifndef LRQR_APPLY_PHASE1_WMMA
#define LRQR_APPLY_PHASE1_WMMA 0
#endif
#ifndef LRQR_APPLY_PHASE3_WMMA
#define LRQR_APPLY_PHASE3_WMMA 0
#endif

// Compile-time cuBLASLt overload for double precision. The LRQR runtime path does not
// select it unless explicitly configured.
inline void gemmLt(cublasLtHandle_t ltH, cublasLtMatmulDesc_t mmDesc,
                   cublasOperation_t opA, cublasOperation_t opB,
                   int m, int n, int k, double alpha,
                   const double* A, int lda, const double* B, int ldb,
                   double beta, double* C, int ldc,
                   void* workspace, size_t wsSize, cudaStream_t stream) {
    int aRows = (opA == CUBLAS_OP_T) ? k : m;
    int aCols = (opA == CUBLAS_OP_T) ? m : k;
    int bRows = (opB == CUBLAS_OP_T) ? n : k;
    int bCols = (opB == CUBLAS_OP_T) ? k : n;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, aRows, aCols, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, bRows, bCols, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &wsSize, sizeof(wsSize));
    cublasLtMatmulHeuristicResult_t result;
    int nResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltH, mmDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                   pref, 1, &result, &nResults);
    if (nResults == 0) {
        cublasLtMatmul(ltH, mmDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                       C, Cdesc, C, Cdesc, nullptr, workspace, wsSize, stream);
    } else {
        cublasLtMatmul(ltH, mmDesc, &alpha, A, Adesc, B, Bdesc, &beta,
                       C, Cdesc, C, Cdesc, &result.algo, workspace, result.workspaceSize, stream);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
}

#include "lrqr.cuh"
#include <cusolverDn.h>
#include <curand.h>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <cstdio>
#include <type_traits>
#include <chrono>
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>

#define CUSOLVER_CHECK(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while (0)
#define CURAND_CHECK(x) do { curandStatus_t s_ = (x); if (s_ != CURAND_STATUS_SUCCESS) { \
    fprintf(stderr, "cuRAND error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1);} } while (0)

// Any sanity failure invalidates the run and returns a nonzero status.
static int g_sanity_invalid = 0;

// Householder QR work for either tall or wide matrices.
// Expressing it through min(m, n) avoids invalid negative values for wide inputs.
static double flops_qr(double m, double n) {
    const double k = (m < n) ? m : n;
    const double lng = (m < n) ? n : m;
    return 2.0 * k * k * lng - (2.0 / 3.0) * k * k * k;
}

// ---- device kernels
__global__ void k_f2d(const float* s, double* d, size_t nelem) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < nelem) d[i] = (double)s[i];
}
__global__ void k_upper2d(const float* s, size_t lds, double* d, size_t ldd, int n) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(n) * n) return;
    int c = int(gid / n), r = int(gid % n);
    d[size_t(c) * ldd + r] = (r <= c) ? (double)s[size_t(c) * lds + r] : 0.0;
}
__global__ void k_upper2d_d(const double* s, size_t lds, double* d, size_t ldd, int n) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(n) * n) return;
    int c = int(gid / n), r = int(gid % n);
    d[size_t(c) * ldd + r] = (r <= c) ? s[size_t(c) * lds + r] : 0.0;
}
__global__ void k_scale_cols(float* A, size_t lda, int m, int n, const float* s) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(m) * n) return;
    int c = int(gid / m), r = int(gid % m);
    A[size_t(c) * lda + r] *= s[c];
}
__global__ void k_scale_cols_d(double* A, size_t lda, int m, int n, const double* s) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(m) * n) return;
    int c = int(gid / m), r = int(gid % m);
    A[size_t(c) * lda + r] *= s[c];
}
template<typename Real>
__global__ void k_extract_upper(const Real* A, size_t lda, Real* R, size_t ldr, int m, int n) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(n) * n) return;
    int c = int(gid / n), r = int(gid % n);
    R[size_t(c) * ldr + r] = (r <= c) ? A[size_t(c) * lda + r] : Real(0);
}

// Extract upper-triangular block R[0:nrows, j0:j1] from Afac (zeros below diagonal)
template<typename Real>
__global__ void k_extract_upper_block(const Real* Afac, size_t lda, Real* R, size_t ldr,
                                       int nrows, int j0, int j1) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int bw = j1 - j0;
    size_t total = size_t(nrows) * bw;
    if (gid >= total) return;
    int c = int(gid / nrows), r = int(gid % nrows);
    int gc = j0 + c;
    R[size_t(c) * ldr + r] = (r <= gc) ? Afac[size_t(gc) * lda + r] : Real(0);
}

// Deterministic normal generator for single-copy validation. Values depend only on
// global coordinates, allowing input tiles to be regenerated without storing A0.
__device__ __forceinline__ unsigned long long w6p3_splitmix64(unsigned long long z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
template<typename Real>
__global__ void k_hash_normal_T(Real* A, size_t lda, int m, int n,
                                 int i0, int j0, int h, int w) {
    // Generate this local tile from global coordinates; the local leading dimension is h.
    (void)lda; (void)m; (void)n;
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = size_t(h) * w;
    if (gid >= total) return;
    int c = int(gid / h), r = int(gid % h);
    int i = i0 + r, j = j0 + c;
    unsigned long long key = (unsigned long long)i * 1000003ULL + (unsigned long long)j * 2654435761ULL + 42ULL;
    unsigned long long h1 = w6p3_splitmix64(key);
    unsigned long long h2 = w6p3_splitmix64(h1 + 0x9e3779b97f4a7c15ULL);
    double u1 = ((h1 >> 11) + 1) * (1.0 / 9007199254740992.0);   // (0,1), 53-bit
    double u2 = ((h2 >> 11) + 1) * (1.0 / 9007199254740992.0);
    double nrm = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);  // Box-Muller
    A[size_t(c) * h + r] = Real(nrm);
}

// Strided float->double submatrix copy: src[rows×cols, lds] -> dst[rows×cols, ldd]
__global__ void k_f2d_strided(const float* src, int lds, double* dst, int ldd,
                               int rows, int cols) {
    size_t gid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= size_t(rows) * cols) return;
    int c = int(gid / rows), r = int(gid % rows);
    dst[size_t(c) * ldd + r] = (double)src[size_t(c) * lds + r];
}

// fp64 Frobenius norm squared: sum of x_i^2 accumulated in double
template<typename Real>
__global__ void k_frob_sq(const Real* X, size_t total, double* result) {
    __shared__ double sdata[256];
    size_t tid = threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    double sum = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + tid; i < total; i += stride) {
        double v = (double)X[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

template<typename Real>
static double frob_sq_T(const Real* X, size_t total) {
    if (total == 0) return 0;
    double* result_d;
    CUDA_CHECK(cudaMalloc(&result_d, sizeof(double)));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0, sizeof(double)));
    int threads = 256;
    int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 4096);
    k_frob_sq<Real><<<blocks, threads>>>(X, total, result_d);
    double result;
    CUDA_CHECK(cudaMemcpy(&result, result_d, sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(result_d);
    return result;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

// LAPACK-style QR validation uses one-norm residuals and absolute normalization.
// It is intentionally separate from the tiled Frobenius-norm parity metrics.

// One-norm of an m x n column-major matrix = max over columns of sum |a_ij|.
// One block per column; block-level reduction in fp64 regardless of Real, so the
// norm of a tf32-tier matrix is not itself computed in low precision.
template<typename Real>
__global__ void k_colsum_abs(const Real* A, size_t lda, int m, int n, double* out) {
    const int j = blockIdx.x;
    if (j >= n) return;
    __shared__ double sh[256];
    double s = 0.0;
    for (int i = threadIdx.x; i < m; i += blockDim.x)
        s += fabs((double)A[(size_t)j * lda + i]);
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x >> 1; o; o >>= 1) {
        if (threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[j] = sh[0];
}

template<typename Real>
static double norm1_T(const Real* A, size_t lda, int m, int n) {
    if (m <= 0 || n <= 0) return 0.0;
    double* d; CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(double)));
    k_colsum_abs<Real><<<n, 256>>>(A, lda, m, n, d);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d);
    double mx = 0.0;
    for (int j = 0; j < n; ++j) if (h[j] > mx) mx = h[j];
    return mx;
}

// ---- relRR: always fp64 (k-invariant backward-error proxy, S14)
// For Real=float: converts A0 and upper(Afac) to double.
// For Real=double: uses A0 directly, converts upper(Afac) via k_upper2d_d.
template<typename Real>
static double relRR_vec_T(cublasHandle_t cb, const Real* A0, const Real* Afac, int m, int n) {
    // A2: this proxy extracts R as n x n (k_upper2d(Afac, m, Rd, n, n)). For m<n
    // R is m x n and TRAPEZOIDAL, so an n x n extraction would read rows that do
    // not exist in Afac. Return -1 (the record's "not computed" value) rather than
    // a number read out of bounds. relRR is an S14 kappa-invariance proxy, not one
    // of the gated LAPACK residuals, and no PASS/FAIL depends on it.
    if (m < n) return -1.0;
    size_t mn = size_t(m) * n, nn = size_t(n) * n;
    double *A0d = nullptr, *Rd, *x, *y, *z1, *z2;
    CUDA_CHECK(cudaMalloc(&Rd, nn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&y, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&z1, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&z2, n * sizeof(double)));
    bool ownA0d;
    if constexpr (std::is_same_v<Real, double>) {
        A0d = const_cast<double*>(A0); ownA0d = false;
        k_upper2d_d<<<(unsigned)((nn + 255) / 256), 256>>>(Afac, m, Rd, n, n);
    } else {
        CUDA_CHECK(cudaMalloc(&A0d, mn * sizeof(double))); ownA0d = true;
        k_f2d<<<(unsigned)((mn + 255) / 256), 256>>>(A0, A0d, mn);
        k_upper2d<<<(unsigned)((nn + 255) / 256), 256>>>(Afac, m, Rd, n, n);
    }
    std::vector<double> hx(n);
    srand(1234);
    for (int i = 0; i < n; ++i) hx[i] = 2.0 * rand() / RAND_MAX - 1.0;
    CUDA_CHECK(cudaMemcpy(x, hx.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemv(cb, CUBLAS_OP_N, m, n, &one, A0d, m, x, 1, &zero, y, 1));
    CUBLAS_CHECK(cublasDgemv(cb, CUBLAS_OP_T, m, n, &one, A0d, m, y, 1, &zero, z1, 1));
    CUDA_CHECK(cudaMemcpy(z2, x, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDtrmv(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, Rd, n, z2, 1));
    CUBLAS_CHECK(cublasDtrmv(cb, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, Rd, n, z2, 1));
    double neg = -1.0;
    CUBLAS_CHECK(cublasDaxpy(cb, n, &neg, z1, 1, z2, 1));
    double n1, n2;
    CUBLAS_CHECK(cublasDnrm2(cb, n, z1, 1, &n1));
    CUBLAS_CHECK(cublasDnrm2(cb, n, z2, 1, &n2));
    if (ownA0d) cudaFree(A0d);
    cudaFree(Rd); cudaFree(x); cudaFree(y); cudaFree(z1); cudaFree(z2);
    return n2 / n1;
}

// ---- tiled fp64-accumulated orthogonality: ||Q^T Q - I||_F (works at any n)
// For float Q: converts tiles to double, uses Dgemm for fp64 Gramian accumulation.
// For double Q: uses Dgemm directly.
// Blocked over n columns of the Gramian (b=2048); for float, also tiled over m rows.
template<typename Real>
static double orth_tiled_T(cublasHandle_t cb, const Real* Q, int m, int n) {
    const int b = 2048;
    const int tile_m = 4096;
    double orth_sq = 0;

    double* Gblock_d;
    CUDA_CHECK(cudaMalloc(&Gblock_d, (size_t)n * b * sizeof(double)));

    double *Qtile_d = nullptr, *Qjb_d = nullptr;
    if constexpr (std::is_same_v<Real, float>) {
        CUDA_CHECK(cudaMalloc(&Qtile_d, (size_t)tile_m * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&Qjb_d, (size_t)tile_m * b * sizeof(double)));
    }

    double one_d = 1.0, zero_d = 0.0;
    for (int j0 = 0; j0 < n; j0 += b) {
        int j1 = std::min(j0 + b, n);
        int bw = j1 - j0;
        CUDA_CHECK(cudaMemsetAsync(Gblock_d, 0, (size_t)n * bw * sizeof(double)));

        if constexpr (std::is_same_v<Real, double>) {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, bw, m,
                                     &one_d, Q, m, Q + (size_t)j0 * m, m,
                                     &zero_d, Gblock_d, n));
        } else {
            for (int i0 = 0; i0 < m; i0 += tile_m) {
                int i1 = std::min(i0 + tile_m, m);
                int tm = i1 - i0;
                k_f2d_strided<<<(unsigned)((size_t)tm * n + 255) / 256, 256>>>(
                    Q + i0, m, Qtile_d, tile_m, tm, n);
                k_f2d_strided<<<(unsigned)((size_t)tm * bw + 255) / 256, 256>>>(
                    Q + (size_t)j0 * m + i0, m, Qjb_d, tile_m, tm, bw);
                CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, n, bw, tm,
                                         &one_d, Qtile_d, tile_m, Qjb_d, tile_m,
                                         &one_d, Gblock_d, n));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> hG((size_t)n * bw);
        CUDA_CHECK(cudaMemcpy(hG.data(), Gblock_d, hG.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int c = 0; c < bw; ++c)
            for (int r = 0; r < n; ++r) {
                double v = hG[(size_t)c * n + r] - (r == j0 + c ? 1.0 : 0.0);
                orth_sq += v * v;
            }
    }
    if constexpr (std::is_same_v<Real, float>) { cudaFree(Qtile_d); cudaFree(Qjb_d); }
    cudaFree(Gblock_d);
    return std::sqrt(orth_sq);
}

// ---- tiled fp64-accumulated residual: ||A0 - Q*R||_F / ||A0||_F (works at any n)
// R is extracted from Afac's upper triangle per column-block. GEMM in Real precision
// (fp32 for float tiers — the residual is dominated by factorization error, not GEMM
// rounding); Frobenius norm reduced in fp64 via k_frob_sq.
template<typename Real>
static double residual_tiled_T(cublasHandle_t cb, const Real* A0, const Real* Q,
                                const Real* Afac, int m, int n) {
    const int b = 2048;
    double normA_sq = frob_sq_T(A0, (size_t)m * n);
    if (normA_sq == 0) return 0;

    Real* Eblock; CUDA_CHECK(cudaMalloc(&Eblock, (size_t)m * b * sizeof(Real)));
    Real* Rblock; CUDA_CHECK(cudaMalloc(&Rblock, (size_t)m * b * sizeof(Real)));
    double resid_sq = 0;
    Real oner = Real(1), negr = Real(-1);

    // A2: R has min(m,n) rows, not n. For m>=n this is n and nothing changes; for
    // m<n, R is upper TRAPEZOIDAL and clamping is what keeps the GEMM's k dimension
    // equal to Q's actual column count. Without the clamp cuBLAS rejects the call
    // ("DGEMM parameter number 10 had an illegal value"), which is how the wide
    // case first announced itself.
    const int kq = std::min(m, n);
    for (int j0 = 0; j0 < n; j0 += b) {
        int j1 = std::min(j0 + b, n);
        int bw = j1 - j0;
        int nrows = std::min(j1, kq);  // R[0:min(j1,k), j0:j1]

        // Eblock = A0[:, j0:j1]
        CUDA_CHECK(cudaMemcpy(Eblock, A0 + (size_t)j0 * m,
                              (size_t)m * bw * sizeof(Real), cudaMemcpyDeviceToDevice));
        // Rblock = upper(Afac)[0:nrows, j0:j1] (zeros below diagonal)
        {
            size_t total = size_t(nrows) * bw;
            k_extract_upper_block<Real><<<(unsigned)((total + 255) / 256), 256>>>(
                Afac, m, Rblock, m, nrows, j0, j1);
        }
        // Eblock -= Q[:, 0:nrows] * Rblock[0:nrows, 0:bw]
        if constexpr (std::is_same_v<Real, float>) {
            CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, m, bw, nrows, &negr,
                                      Q, CUDA_R_32F, m, Rblock, CUDA_R_32F, m,
                                      &oner, Eblock, CUDA_R_32F, m,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        } else {
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, m, bw, nrows,
                                     &negr, Q, m, Rblock, m, &oner, Eblock, m));
        }
        resid_sq += frob_sq_T(Eblock, (size_t)m * bw);
    }
    cudaFree(Eblock); cudaFree(Rblock);
    return std::sqrt(resid_sq) / std::sqrt(normA_sq);
}

// Tiled LRQR residual and orthogonality metrics. Form Q with default math to
// prevent TF32 promotion during validation.
template<class PlanT, typename Real>
static void residual_orth_T(PlanT& plan, cublasHandle_t cb, const Real* A0, const Real* Afac,
                            int m, int n, double* resid, double* orth) {
    // A2: the thin Q is m x min(m,n). Equals m x n on every m>=n cell, so no
    // existing measurement moves; for m<n it is what makes the metric defined.
    const int kq = std::min(m, n);
    Real* Q; CUDA_CHECK(cudaMalloc(&Q, size_t(m) * kq * sizeof(Real)));

    // Keep Q formation in base fp32.
    cublasMath_t oldM0 = CUBLAS_DEFAULT_MATH, oldM1 = CUBLAS_DEFAULT_MATH, oldMT = CUBLAS_DEFAULT_MATH;
    if (plan.cb0) { cublasGetMathMode(plan.cb0, &oldM0); cublasSetMathMode(plan.cb0, CUBLAS_DEFAULT_MATH); }
    if (plan.cb1) { cublasGetMathMode(plan.cb1, &oldM1); cublasSetMathMode(plan.cb1, CUBLAS_DEFAULT_MATH); }
    if (plan.cbT) { cublasGetMathMode(plan.cbT, &oldMT); cublasSetMathMode(plan.cbT, CUBLAS_DEFAULT_MATH); }

    plan.orgqr(Afac, m, Q, m);

    if (plan.cb0) cublasSetMathMode(plan.cb0, oldM0);
    if (plan.cb1) cublasSetMathMode(plan.cb1, oldM1);
    if (plan.cbT) cublasSetMathMode(plan.cbT, oldMT);

    *orth = orth_tiled_T<Real>(cb, Q, m, kq);   // I - Q'Q is kq x kq
    *resid = residual_tiled_T<Real>(cb, A0, Q, Afac, m, n);
    cudaFree(Q);
}

// Single-copy residual validation regenerates input tiles and streams Q row tiles,
// avoiding full A0 and Q allocations. Orthogonality is reported as unavailable.
template<class PlanT, typename Real>
static void residual_single_copy_T(PlanT& plan, cublasHandle_t cb, const Real* Afac,
                                   int m, int n, double* resid, double* orth,
                                   const Real* A0_full_for_check) {
    const int b = 1024;          // row-tile height (Q row-block)
    const int bw = 2048;         // R column-block width
    *orth = -1.0;                // deferred (§2.4): row-tiled Gramian needs n×n fp64
    double resid_sq = 0, normA_sq = 0;
    Real* X;      CUDA_CHECK(cudaMalloc(&X, size_t(m) * b * sizeof(Real)));      // m×b
    Real* A0tile; CUDA_CHECK(cudaMalloc(&A0tile, size_t(b) * n * sizeof(Real))); // b×n (ld=b)
    Real* E;      CUDA_CHECK(cudaMalloc(&E, size_t(b) * n * sizeof(Real)));      // b×n (ld=b)
    Real* Rblock; CUDA_CHECK(cudaMalloc(&Rblock, size_t(n) * bw * sizeof(Real))); // nrows×bw
    Real oner = Real(1), negr = Real(-1);

    // Force fp32 Q-formation (parity with residual_orth_T P0.b).
    cublasMath_t oldM0 = CUBLAS_DEFAULT_MATH;
    if (plan.cb0) { cublasGetMathMode(plan.cb0, &oldM0); cublasSetMathMode(plan.cb0, CUBLAS_DEFAULT_MATH); }

    for (int i0 = 0; i0 < m; i0 += b) {
        int i1 = std::min(i0 + b, m);
        int bcur = i1 - i0;
        // 1. Regenerate A0[i0:i1, 0:n] (bcur×n, ld=bcur) via the position-independent hash.
        {
            size_t total = size_t(bcur) * n;
            unsigned grid = (unsigned)((total + 255) / 256);
            k_hash_normal_T<Real><<<grid, 256>>>(A0tile, bcur, bcur, n, i0, 0, bcur, n);
        }
        normA_sq += frob_sq_T(A0tile, (size_t)bcur * n);
        // 2. Form Q[i0:i1,:]^T as X[0:n,:] (m×b, ld=m) via orgqr_rows.
        plan.orgqr_rows(Afac, m, X, m, i0, i1);
        // 3. E = A0_tile copied into the b×n (ld=b) buffer; pre-zero so the last tile's
        //    unused rows [bcur:b] contribute zero to frob_sq (the GEMM writes only bcur rows).
        CUDA_CHECK(cudaMemsetAsync(E, 0, size_t(b) * n * sizeof(Real)));
        if (bcur == b) {
            CUDA_CHECK(cudaMemcpy(E, A0tile, size_t(b) * n * sizeof(Real), cudaMemcpyDeviceToDevice));
        } else {
            for (int j = 0; j < n; ++j)
                CUDA_CHECK(cudaMemcpy(E + size_t(j)*b, A0tile + size_t(j)*bcur,
                                      size_t(bcur)*sizeof(Real), cudaMemcpyDeviceToDevice));
        }
        // E[:, j0:j1] -= X[0:nrows,:]^T · Rblock  (Q[i0:i1,0:nrows]·R[0:nrows,j0:j1])
        for (int j0 = 0; j0 < n; j0 += bw) {
            int j1 = std::min(j0 + bw, n);
            int nrows = j1;
            int wcur = j1 - j0;
            size_t total = size_t(nrows) * wcur;
            k_extract_upper_block<Real><<<(unsigned)((total + 255) / 256), 256>>>(
                Afac, m, Rblock, nrows, nrows, j0, j1);
            // C(b, wcur) = X[0:nrows,:]^T (b×nrows) · Rblock (nrows×wcur)
            if constexpr (std::is_same_v<Real, float>) {
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, bcur, wcur, nrows, &negr,
                                          X, CUDA_R_32F, m, Rblock, CUDA_R_32F, nrows,
                                          &oner, E + size_t(j0)*b, CUDA_R_32F, b,
                                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            } else {
                CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, bcur, wcur, nrows,
                                         &negr, X, m, Rblock, nrows, &oner, E + size_t(j0)*b, b));
            }
        }
        resid_sq += frob_sq_T(E, (size_t)b * n);   // full b×n (unused rows are zero)
    }
    if (plan.cb0) cublasSetMathMode(plan.cb0, oldM0);
    cudaFree(X); cudaFree(A0tile); cudaFree(E); cudaFree(Rblock);
    (void)A0_full_for_check;  // reserved for a future curand-offset self-check
    *resid = (normA_sq > 0) ? std::sqrt(resid_sq) / std::sqrt(normA_sq) : -1.0;
}

// ---- GPU reduction: sum of (G[r,c] - delta(r, j0+c))^2 for one Gramian column-block.
// G is (n×wcur, ld=n); the identity subtraction targets column j0+c of the full n×n Gramian.
__global__ void k_orth_block_sq(const double* G, int n, int j0, int wcur, double* result) {
    __shared__ double sdata[256];
    size_t tid = threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    double sum = 0;
    size_t total = size_t(n) * wcur;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + tid; i < total; i += stride) {
        int c = int(i / n), r = int(i % n);
        double v = G[size_t(c) * n + r] - (r == j0 + c ? 1.0 : 0.0);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// Single-copy orthogonality validation streams Q row tiles and accumulates the
// Gramian in fp64 blocks, avoiding a materialized Q.
template<class PlanT, typename Real>
static double orth_single_copy_T(PlanT& plan, cublasHandle_t cb, const Real* Afac, int m, int n) {
    if (n <= 0) return 0.0;
    const int bw = 2048;         // Gramian column-block width
    const int nblocks = (n + bw - 1) / bw;

    // Force fp32 Q-formation (parity with residual_orth_T P0.b / residual_single_copy_T).
    cublasMath_t oldM0 = CUBLAS_DEFAULT_MATH;
    if (plan.cb0) { cublasGetMathMode(plan.cb0, &oldM0); cublasSetMathMode(plan.cb0, CUBLAS_DEFAULT_MATH); }

    // Adaptive row-tile height b: start at 1024, reduce to 512/256 if the Gblock
    // batch size k is too small (too many passes over Q). At fp64@131072² (A=128 GiB),
    // b=1024 leaves room for only k=1 Gblock (64 passes — too slow); b=512 fits k=3
    // (22 passes — feasible). At 98304² / 3×tf32@131072², b=1024 already gives k≥30.
    int b = 1024;
    const size_t per_block = size_t(n) * bw * sizeof(double);
    const size_t headroom = size_t(1) << 30;   // 1 GiB for cuBLAS ws + GEMM internals
    Real* X = nullptr;
    int k = 0;
    for (int b_try : {1024, 512, 256, 128}) {
        size_t x_bytes = size_t(m) * b_try * sizeof(Real);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        size_t avail = (free_bytes > headroom + x_bytes) ? free_bytes - headroom - x_bytes : 0;
        int k_try = std::max(1, (int)(avail / per_block));
        k_try = std::min(k_try, nblocks);
        if (k_try >= 4 || b_try == 128) {   // k≥4 → ≤ceil(nblocks/4) passes, or min b
            b = b_try;
            k = k_try;
            break;
        }
    }
    CUDA_CHECK(cudaMalloc(&X, size_t(m) * b * sizeof(Real)));   // m×b

    double* Gblock_d;  // k × (n×bw) contiguous fp64 Gramian column-blocks
    CUDA_CHECK(cudaMalloc(&Gblock_d, (size_t)k * per_block));

    // For float Q: double conversion buffers (X[0:n,:] and X[j0:j1,:] to double).
    double *Xd_d = nullptr, *Xjd_d = nullptr;
    if constexpr (std::is_same_v<Real, float>) {
        CUDA_CHECK(cudaMalloc(&Xd_d, size_t(n) * b * sizeof(double)));   // n×b
        CUDA_CHECK(cudaMalloc(&Xjd_d, size_t(bw) * b * sizeof(double)));  // bw×b
    }

    double* orth_sq_d;  // fp64 scalar on device for the GPU reduction
    CUDA_CHECK(cudaMalloc(&orth_sq_d, sizeof(double)));

    double one_d = 1.0;
    int threads = 256;
    int max_blocks = 4096;

    for (int blk0 = 0; blk0 < nblocks; blk0 += k) {
        int blk1 = std::min(blk0 + k, nblocks);
        int kcur = blk1 - blk0;
        for (int bi = 0; bi < kcur; ++bi)
            CUDA_CHECK(cudaMemsetAsync(Gblock_d + (size_t)bi * n * bw, 0, per_block));

        // Stream Q row-tiles, accumulating into all kcur Gblock buffers (one pass).
        for (int i0 = 0; i0 < m; i0 += b) {
            int i1 = std::min(i0 + b, m);
            // X[0:n, 0:b] = Q[i0:i1, :]^T  (n×b, ld=m). Extra cols (bcur..b) are zero.
            plan.orgqr_rows(Afac, m, X, m, i0, i1);

            if constexpr (std::is_same_v<Real, double>) {
                // Gblock_bi += X[0:n,:] · X[j0:j1,:]^T  (n×wcur += n×b · b×wcur)
                for (int bi = 0; bi < kcur; ++bi) {
                    int j0 = (blk0 + bi) * bw;
                    int j1 = std::min(j0 + bw, n);
                    int wcur = j1 - j0;
                    CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T,
                                             n, wcur, b, &one_d,
                                             X, m, X + (size_t)j0, m,
                                             &one_d, Gblock_d + (size_t)bi * n * bw, n));
                }
            } else {
                // Convert X[0:n, 0:b] (n×b, ld=m, float) → Xd_d (n×b, ld=n, double).
                {
                    size_t total = size_t(n) * b;
                    k_f2d_strided<<<(unsigned)((total + 255) / 256), 256>>>(
                        X, m, Xd_d, n, n, b);
                }
                for (int bi = 0; bi < kcur; ++bi) {
                    int j0 = (blk0 + bi) * bw;
                    int j1 = std::min(j0 + bw, n);
                    int wcur = j1 - j0;
                    // Convert X[j0:j1, 0:b] (wcur×b, ld=m, float) → Xjd_d (wcur×b, ld=wcur, double).
                    {
                        size_t total = size_t(wcur) * b;
                        k_f2d_strided<<<(unsigned)((total + 255) / 256), 256>>>(
                            X + (size_t)j0, m, Xjd_d, wcur, wcur, b);
                    }
                    // Gblock_bi += Xd_d · Xjd_d^T  (n×wcur += n×b · b×wcur)
                    CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T,
                                             n, wcur, b, &one_d,
                                             Xd_d, n, Xjd_d, wcur,
                                             &one_d, Gblock_d + (size_t)bi * n * bw, n));
                }
            }
        }

        // GPU-reduce (G - I)^2 for each Gblock in this batch.
        CUDA_CHECK(cudaMemsetAsync(orth_sq_d, 0, sizeof(double)));
        for (int bi = 0; bi < kcur; ++bi) {
            int j0 = (blk0 + bi) * bw;
            int j1 = std::min(j0 + bw, n);
            int wcur = j1 - j0;
            size_t total = size_t(n) * wcur;
            int blocks = (int)std::min<size_t>((total + threads - 1) / threads, max_blocks);
            k_orth_block_sq<<<blocks, threads>>>(
                Gblock_d + (size_t)bi * n * bw, n, j0, wcur, orth_sq_d);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double orth_sq;
    CUDA_CHECK(cudaMemcpy(&orth_sq, orth_sq_d, sizeof(double), cudaMemcpyDeviceToHost));

    if constexpr (std::is_same_v<Real, float>) { cudaFree(Xd_d); cudaFree(Xjd_d); }
    cudaFree(X); cudaFree(Gblock_d); cudaFree(orth_sq_d);
    if (plan.cb0) cublasSetMathMode(plan.cb0, oldM0);
    return std::sqrt(orth_sq);
}

// ---- cuSOLVER residual + orth (tiled, works at any n)
// Forms explicit Q via orgqr, then computes tiled resid and orth (same as LR-QR path).
// P0.b/P0.c: parity baseline exists at every n.
template<typename Real>
static void cusolver_resid_orth_T(cusolverDnHandle_t cs, cublasHandle_t cb,
                                   const Real* A0, const Real* Afac, const Real* tau,
                                   int m, int n, double* resid, double* orth) {
    // A2: cuSOLVER's thin Q is m x min(m,n) too -- ORGQR's K argument is the
    // reflector count. The comparison arm has to be generalized in step with the
    // LR-QR arm or the m<n cells would compare different objects.
    const int kq = std::min(m, n);
    Real* Q; CUDA_CHECK(cudaMalloc(&Q, size_t(m) * kq * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy2D(Q, size_t(m) * sizeof(Real), Afac, size_t(m) * sizeof(Real),
                            size_t(m) * sizeof(Real), (size_t)kq, cudaMemcpyDeviceToDevice));
    int lw = 0;
    if constexpr (std::is_same_v<Real, float>)
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cs, m, kq, kq, Q, m, tau, &lw));
    else
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs, m, kq, kq, Q, m, tau, &lw));
    Real* work; CUDA_CHECK(cudaMalloc(&work, size_t(lw) * sizeof(Real)));
    int* info; CUDA_CHECK(cudaMalloc(&info, 4));
    if constexpr (std::is_same_v<Real, float>)
        CUSOLVER_CHECK(cusolverDnSorgqr(cs, m, kq, kq, Q, m, tau, work, lw, info));
    else
        CUSOLVER_CHECK(cusolverDnDorgqr(cs, m, kq, kq, Q, m, tau, work, lw, info));
    CUDA_CHECK(cudaDeviceSynchronize());

    *orth = orth_tiled_T<Real>(cb, Q, m, kq);
    *resid = residual_tiled_T<Real>(cb, A0, Q, Afac, m, n);
    cudaFree(Q); cudaFree(work); cudaFree(info);
}

// --bw-probe measures streaming global-memory throughput across working-set sizes.
// The result calibrates timing rates, not communication word-count bounds.
template<typename Real>
__global__ static void k_stream_rw(const Real* __restrict__ src, Real* __restrict__ dst,
                                   size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i] * Real(1.0000001);
}

template<typename Real>
static int bandwidth_probe(const char* tier_name) {
    // Working sets from comfortably L2-resident to far past it. H200 carries ~50 MB
    // of L2 with a 39 MB persisting window, so the sweep straddles the boundary
    // rather than assuming where it is.
    const size_t mbs[] = {2, 8, 24, 48, 128, 512, 2048, 8192};
    printf("\n=== bandwidth probe beta_lambda (%s, %zu-byte words) ===\n",
           tier_name, sizeof(Real));
    printf("  %10s %14s %14s %16s  %s\n", "set_MB", "GB/s", "Gword/s", "words_moved", "class");
    for (size_t mb : mbs) {
        const size_t bytes = mb << 20;
        const size_t nelem = bytes / sizeof(Real);
        Real *src = nullptr, *dst = nullptr;
        if (cudaMalloc(&src, bytes) != cudaSuccess) { printf("  %10zu  (alloc failed)\n", mb); break; }
        if (cudaMalloc(&dst, bytes) != cudaSuccess) { cudaFree(src); printf("  %10zu  (alloc failed)\n", mb); break; }
        CUDA_CHECK(cudaMemset(src, 1, bytes));
        const int threads = 256;
        const int blocks = (int)std::min<size_t>(65535, (nelem + threads - 1) / threads);
        // warm up, then time enough repeats that a small set is not launch-dominated
        k_stream_rw<Real><<<blocks, threads>>>(src, dst, nelem);
        CUDA_CHECK(cudaDeviceSynchronize());
        const int reps = (mb <= 48) ? 200 : 20;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) k_stream_rw<Real><<<blocks, threads>>>(src, dst, nelem);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        const double secs = (double)ms * 1e-3 / reps;
        const double words = 2.0 * (double)nelem;          // one read + one write
        const double gbs = (2.0 * (double)bytes) / secs / 1e9;
        const double gws = words / secs / 1e9;
        printf("  %10zu %14.1f %14.3f %16.4e  %s\n", mb, gbs, gws, words,
               (mb <= 24) ? "L2-resident" : (mb <= 48 ? "L2 boundary" : "HBM"));
        fflush(stdout);
        cudaFree(src); cudaFree(dst);
    }
    printf("  NOTE: rungs 0/1 (register, shared/DSM) are NOT measured here -- a global\n"
           "  stream cannot see them, so their beta stays uncalibrated.\n");
    return 0;
}

// ============================================================================
// LAPACK DLATB4/DLATMS matrix types for the 'QR' path (--validate-lapack).
//
// CONSTANTS ARE THE REFERENCE ONES, not invented here. From
// Reference-LAPACK/lapack TESTING/LIN/dlatb4.f, the QR/LQ/QL/RQ group sets
// TYPE='N', DIST='S', MODE=3 throughout, and per IMAT:
//
//   IMAT  KL          KU          CNDNUM   ANORM
//   1     0           0           TWO      ONE     -> diagonal
//   2     0           max(N-1,0)  TWO      ONE     -> upper triangular
//   3     max(M-1,0)  0           TWO      ONE     -> lower triangular
//   4     max(M-1,0)  max(N-1,0)  TWO      ONE     -> full
//   5     full                    BADC1    ONE
//   6     full                    BADC2    ONE
//   7     full                    TWO      SMALL
//   8     full                    TWO      LARGE
//
//   EPS   = DLAMCH('Precision')          (= 2u for IEEE)
//   BADC2 = TENTH / EPS                  (TENTH = 0.1)
//   BADC1 = sqrt(BADC2)
//   SMALL = SHRINK * (safmin / EPS)      (SHRINK = 0.25)
//   LARGE = ONE / SMALL
//
// MODE=3 is a GEOMETRIC singular-value sequence, d_i = CNDNUM^(-(i-1)/(k-1)),
// which is what this project's existing gen_svd_T already produces -- so types
// 4-6 are the generator already here, at three condition numbers.
//
// TYPES 2 AND 3 ARE FAITHFUL, NOT APPROXIMATED, and the reason is worth stating:
// DLATMS builds a banded matrix with prescribed singular values via
// band-preserving rotations, which would be a real port. But QR preserves
// singular values -- A = QR with Q orthogonal gives R = Q'A and sv(R) = sv(A) --
// so factoring a full matrix that has the prescribed geometric spectrum and
// taking R yields an upper triangular matrix with EXACTLY that spectrum.
// Transposing it gives the lower triangular case. Both properties the test
// depends on (triangular structure, prescribed conditioning) hold exactly.
//
// SHAPE GENERALITY: U is m x k and V is n x k with k = min(m,n), both TALL, so
// the cuSOLVER geqrf/orgqr that build them work for m<n as well as m>=n. The
// pre-existing gen_svd_T cannot do m<n (it orgqr's an m x n basis).
//
// WHICH EPS FOR A TF32 TIER. SMALL/LARGE are about REPRESENTABILITY, so they use
// the storage type's safmin/eps (float for tf32 and 3xTF32). CNDNUM and the
// residual normalization are about ACCURACY, so they use the tier's own u. Using
// tf32's u for safmin scaling would put type 7 above float's underflow threshold.
// ============================================================================

__global__ static void k_zero_d(double* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0;
}
template<typename Real>
__global__ static void k_zero_T(Real* p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = Real(0);
}
// A[i,i] = d[i] on a zeroed m x n buffer (IMAT 1)
template<typename Real>
__global__ static void k_set_diag_T(Real* A, size_t lda, int k, const double* d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) A[(size_t)i * lda + i] = (Real)d[i];
}
// zero the strict lower triangle (keep upper) -- used to take R out of a factored A
template<typename Real>
__global__ static void k_keep_upper_T(Real* A, size_t lda, int m, int n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)m * n) return;
    int j = (int)(t / m), i = (int)(t % m);
    if (i > j) A[(size_t)j * lda + i] = Real(0);
}
template<typename Real>
__global__ static void k_transpose_T(const Real* S, size_t lds, Real* D, size_t ldd,
                                     int rows, int cols) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)rows * cols) return;
    int j = (int)(t / rows), i = (int)(t % rows);
    D[(size_t)i * ldd + j] = S[(size_t)j * lds + i];
}
template<typename Real>
__global__ static void k_scal_T(Real* A, size_t n, double s) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i] = (Real)((double)A[i] * s);
}

// dlatb4 'QR' parameters for one IMAT, at a given storage/accuracy precision.
struct LapackQRParams {
    int imat; const char* name;
    bool diagonal, upper_tri, lower_tri;   // KL/KU structure
    double cndnum;                          // CNDNUM
    double anorm;                           // ANORM (<0 means "leave at ONE")
};
static LapackQRParams lapack_qr_params(int imat, double eps_precision, double safmin) {
    const double TENTH = 0.1, SHRINK = 0.25;
    const double BADC2 = TENTH / eps_precision;
    const double BADC1 = std::sqrt(BADC2);
    const double SMALL = SHRINK * (safmin / eps_precision);
    const double LARGE = 1.0 / SMALL;
    switch (imat) {
        case 1: return {1, "diagonal",        true,  false, false, 2.0,   1.0};
        case 2: return {2, "upper-triangular",false, true,  false, 2.0,   1.0};
        case 3: return {3, "lower-triangular",false, false, true,  2.0,   1.0};
        case 4: return {4, "full-cnd2",       false, false, false, 2.0,   1.0};
        case 5: return {5, "full-badc1",      false, false, false, BADC1, 1.0};
        case 6: return {6, "full-badc2",      false, false, false, BADC2, 1.0};
        case 7: return {7, "full-underflow",  false, false, false, 2.0,   SMALL};
        default:return {8, "full-overflow",   false, false, false, 2.0,   LARGE};
    }
}

// An orthonormal basis: rows x k, k <= rows, from QR of a random matrix.
template<typename Real>
static void gen_orth_basis_T(cusolverDnHandle_t cs, curandGenerator_t rg,
                             Real* Qb, int rows, int k) {
    size_t rk = (size_t)rows * k;
    if constexpr (std::is_same_v<Real, float>)
        CURAND_CHECK(curandGenerateNormal(rg, Qb, rk, 0.0f, 1.0f));
    else
        CURAND_CHECK(curandGenerateNormalDouble(rg, Qb, rk, 0.0, 1.0));
    Real* tau; CUDA_CHECK(cudaMalloc(&tau, (size_t)k * sizeof(Real)));
    int* info; CUDA_CHECK(cudaMalloc(&info, 4));
    int lw1 = 0, lw2 = 0;
    if constexpr (std::is_same_v<Real, float>) {
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cs, rows, k, Qb, rows, &lw1));
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cs, rows, k, k, Qb, rows, tau, &lw2));
    } else {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs, rows, k, Qb, rows, &lw1));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs, rows, k, k, Qb, rows, tau, &lw2));
    }
    int lw = std::max(lw1, lw2);
    Real* work; CUDA_CHECK(cudaMalloc(&work, (size_t)std::max(lw, 1) * sizeof(Real)));
    if constexpr (std::is_same_v<Real, float>) {
        CUSOLVER_CHECK(cusolverDnSgeqrf(cs, rows, k, Qb, rows, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnSorgqr(cs, rows, k, k, Qb, rows, tau, work, lw, info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgeqrf(cs, rows, k, Qb, rows, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnDorgqr(cs, rows, k, k, Qb, rows, tau, work, lw, info));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(tau); cudaFree(info); cudaFree(work);
}

// Generate the IMAT-th LAPACK 'QR' test matrix into A (m x n, ld = m).
// Works for m >= n and m < n.
template<typename Real>
static void gen_lapack_qr_type_T(int imat, cusolverDnHandle_t cs, cublasHandle_t cb,
                                 curandGenerator_t rg, Real* A, int m, int n,
                                 double eps_precision, double safmin) {
    const LapackQRParams P = lapack_qr_params(imat, eps_precision, safmin);
    const int k = std::min(m, n);
    const size_t mn = (size_t)m * n;

    // MODE=3: geometric singular values, d_1 = 1 down to d_k = 1/CNDNUM.
    std::vector<double> hd(k);
    for (int i = 0; i < k; ++i)
        hd[i] = (k == 1) ? 1.0 : std::pow(P.cndnum, -(double)i / (double)(k - 1));
    double* d; CUDA_CHECK(cudaMalloc(&d, (size_t)k * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d, hd.data(), (size_t)k * sizeof(double), cudaMemcpyHostToDevice));

    k_zero_T<Real><<<(unsigned)((mn + 255) / 256), 256>>>(A, mn);

    if (P.diagonal) {
        k_set_diag_T<Real><<<(unsigned)((k + 255) / 256), 256>>>(A, m, k, d);
    } else {
        // A = U * diag(d) * V^T with U (m x k) and V (n x k) orthonormal.
        Real *U, *V, *W;
        CUDA_CHECK(cudaMalloc(&U, (size_t)m * k * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&V, (size_t)n * k * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&W, (size_t)m * k * sizeof(Real)));
        gen_orth_basis_T<Real>(cs, rg, U, m, k);
        gen_orth_basis_T<Real>(cs, rg, V, n, k);
        // W = U * diag(d)
        {
            std::vector<Real> hdr(k);
            for (int i = 0; i < k; ++i) hdr[i] = (Real)hd[i];
            Real* dr; CUDA_CHECK(cudaMalloc(&dr, (size_t)k * sizeof(Real)));
            CUDA_CHECK(cudaMemcpy(dr, hdr.data(), (size_t)k * sizeof(Real), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(W, U, (size_t)m * k * sizeof(Real), cudaMemcpyDeviceToDevice));
            if constexpr (std::is_same_v<Real, float>)
                CUBLAS_CHECK(cublasSdgmm(cb, CUBLAS_SIDE_RIGHT, m, k, W, m, dr, 1, W, m));
            else
                CUBLAS_CHECK(cublasDdgmm(cb, CUBLAS_SIDE_RIGHT, m, k, W, m, dr, 1, W, m));
            cudaFree(dr);
        }
        // A = W * V^T
        Real one = Real(1), zero = Real(0);
        if constexpr (std::is_same_v<Real, float>)
            CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                     &one, W, m, V, n, &zero, A, m));
        else
            CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
                                     &one, W, m, V, n, &zero, A, m));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(U); cudaFree(V); cudaFree(W);

        if (P.upper_tri || P.lower_tri) {
            // QR preserves singular values, so R from a QR of A is upper
            // triangular WITH THE PRESCRIBED SPECTRUM. Transpose for the lower case.
            const int kk = std::min(m, n);
            Real* tau; CUDA_CHECK(cudaMalloc(&tau, (size_t)kk * sizeof(Real)));
            int* info; CUDA_CHECK(cudaMalloc(&info, 4));
            int lw = 0;
            if constexpr (std::is_same_v<Real, float>)
                CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cs, m, n, A, m, &lw));
            else
                CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs, m, n, A, m, &lw));
            Real* work; CUDA_CHECK(cudaMalloc(&work, (size_t)std::max(lw, 1) * sizeof(Real)));
            if constexpr (std::is_same_v<Real, float>)
                CUSOLVER_CHECK(cusolverDnSgeqrf(cs, m, n, A, m, tau, work, lw, info));
            else
                CUSOLVER_CHECK(cusolverDnDgeqrf(cs, m, n, A, m, tau, work, lw, info));
            CUDA_CHECK(cudaDeviceSynchronize());
            k_keep_upper_T<Real><<<(unsigned)((mn + 255) / 256), 256>>>(A, m, m, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (P.lower_tri && m == n) {
                Real* Tb; CUDA_CHECK(cudaMalloc(&Tb, mn * sizeof(Real)));
                k_transpose_T<Real><<<(unsigned)((mn + 255) / 256), 256>>>(A, m, Tb, m, m, n);
                CUDA_CHECK(cudaMemcpy(A, Tb, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
                cudaFree(Tb);
            }
            // m != n: a transpose would change the shape, so the lower-triangular
            // type is emitted as upper for non-square cells and labelled as such by
            // the caller. Recorded rather than silently substituted.
            cudaFree(tau); cudaFree(info); cudaFree(work);
        }
    }

    // ANORM: DLATMS scales so the largest absolute entry is ANORM.
    if (P.anorm != 1.0) {
        double mx = norm1_T<Real>(A, m, m, n);   // upper bound on max|a_ij|
        if (mx > 0) {
            double s = P.anorm / mx;
            k_scal_T<Real><<<(unsigned)((mn + 255) / 256), 256>>>(A, mn, s);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    cudaFree(d);
}

// first column where |R| diag diverges from reference by > tol (host-side; float only)
static int first_diag_divergence(const float* dA, const float* dRef, int m, int n, double tol,
                                  double* worst) {
    std::vector<float> a(n), b(n);
    for (int j = 0; j < n; ++j) {
        CUDA_CHECK(cudaMemcpy(&a[j], dA + size_t(j) * m + j, 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&b[j], dRef + size_t(j) * m + j, 4, cudaMemcpyDeviceToHost));
    }
    int first = -1; *worst = 0;
    for (int j = 0; j < n; ++j) {
        double d = std::abs(std::abs((double)a[j]) - std::abs((double)b[j])) /
                   std::max(std::abs((double)b[j]), 1e-30);
        if (d > *worst) *worst = d;
        if (first < 0 && d > tol) first = j;
    }
    return first;
}

// ---- SVD-based exact-kappa generator: A = U diag(s) V^T, U/V random orthogonal, s log-spaced 1..1/kappa
template<typename Real>
static void gen_svd_T(cusolverDnHandle_t cs, cublasHandle_t cb, curandGenerator_t rg,
                      Real* A, int m, int n, double kappa, Real* Ubuf, Real* Vbuf) {
    size_t mn = size_t(m) * n, nn = size_t(n) * n;
    Real* tau; CUDA_CHECK(cudaMalloc(&tau, n * sizeof(Real)));
    int* info; CUDA_CHECK(cudaMalloc(&info, 4));
    int lw1=0, lw2=0, lw3=0, lw4=0;
    if constexpr (std::is_same_v<Real, float>) {
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cs, m, n, Ubuf, m, &lw1));
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cs, m, n, n, Ubuf, m, tau, &lw2));
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cs, n, n, Vbuf, n, &lw3));
        CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(cs, n, n, n, Vbuf, n, tau, &lw4));
    } else {
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs, m, n, Ubuf, m, &lw1));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs, m, n, n, Ubuf, m, tau, &lw2));
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs, n, n, Vbuf, n, &lw3));
        CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs, n, n, n, Vbuf, n, tau, &lw4));
    }
    int lw = std::max({lw1, lw2, lw3, lw4});
    Real* work; CUDA_CHECK(cudaMalloc(&work, size_t(lw) * sizeof(Real)));
    if constexpr (std::is_same_v<Real, float>) {
        CURAND_CHECK(curandGenerateNormal(rg, Ubuf, mn, 0.0f, 1.0f));
        CUSOLVER_CHECK(cusolverDnSgeqrf(cs, m, n, Ubuf, m, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnSorgqr(cs, m, n, n, Ubuf, m, tau, work, lw, info));
        CURAND_CHECK(curandGenerateNormal(rg, Vbuf, nn, 0.0f, 1.0f));
        CUSOLVER_CHECK(cusolverDnSgeqrf(cs, n, n, Vbuf, n, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnSorgqr(cs, n, n, n, Vbuf, n, tau, work, lw, info));
    } else {
        CURAND_CHECK(curandGenerateNormalDouble(rg, Ubuf, mn, 0.0, 1.0));
        CUSOLVER_CHECK(cusolverDnDgeqrf(cs, m, n, Ubuf, m, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnDorgqr(cs, m, n, n, Ubuf, m, tau, work, lw, info));
        CURAND_CHECK(curandGenerateNormalDouble(rg, Vbuf, nn, 0.0, 1.0));
        CUSOLVER_CHECK(cusolverDnDgeqrf(cs, n, n, Vbuf, n, tau, work, lw, info));
        CUSOLVER_CHECK(cusolverDnDorgqr(cs, n, n, n, Vbuf, n, tau, work, lw, info));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<Real> hs(n);
    for (int j = 0; j < n; ++j) hs[j] = (Real)std::pow(kappa, -double(j) / (n - 1));
    Real* ds; CUDA_CHECK(cudaMalloc(&ds, n * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(ds, hs.data(), n * sizeof(Real), cudaMemcpyHostToDevice));
    if constexpr (std::is_same_v<Real, float>) {
        k_scale_cols<<<(unsigned)((mn + 255) / 256), 256>>>(Ubuf, m, m, n, ds);
    } else {
        k_scale_cols_d<<<(unsigned)((mn + 255) / 256), 256>>>(Ubuf, m, m, n, ds);
    }
    cudaFree(ds);
    Real one = Real(1), zero = Real(0);
    if constexpr (std::is_same_v<Real, float>) {
        CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &one, Ubuf, m, Vbuf, n, &zero, A, m));
    } else {
        CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &one, Ubuf, m, Vbuf, n, &zero, A, m));
    }
    cudaFree(tau); cudaFree(info); cudaFree(work);
}

// Reference GEMM ceilings used to normalize performance metrics.
static double get_ceiling_tflops(const char* tier, int n) {
    struct Ceil { int n; double tf32, fp32, fp64, x3tf32; };
    static const Ceil ceils[] = {
        {8192,   451.3, 50.6, 64.2, 146.7},
        {16384,  447.1, 54.1, 64.9, 147.9},
        {32768,  435.9, 54.7, 65.1, 143.5},
        {65536,  423.8, 54.9, 65.2, 141.2},
        {98304,  429.2, 54.7, 65.1, 143.1},
        {131072, 427.8, 54.1, 65.1, 142.6},
    };
    auto pick = [&](const Ceil& cc) -> double {
        if (!strcmp(tier, "tf32")) return cc.tf32;
        if (!strcmp(tier, "fp32")) return cc.fp32;
        if (!strcmp(tier, "fp64")) return cc.fp64;
        if (!strcmp(tier, "3xtf32")) return cc.x3tf32;
        return 0;
    };
    for (auto& cc : ceils) if (cc.n == n) return pick(cc);
    // nearest size
    const Ceil* best = &ceils[0];
    for (auto& cc : ceils) if (std::abs(n - cc.n) < std::abs(n - best->n)) best = &cc;
    return pick(*best);
}

// ---- benchmark context
struct BenchCtx {
    int B, reps;
    bool doOrth, kappaCol, kappaSvd;
    cublasHandle_t cb;
    cusolverDnHandle_t cs;
    curandGenerator_t rg;
    bool single, lonly, conly;
};

// Literal L2→HBM superpanel rung shared by benchmarks, validation, stage probes,
// and the single-copy wrapper.  M̂ is in words.  Once n² exceeds L2, the live
// front is the spec's W_HBM=15 leaf panels; there are no size/aspect/tier tables.
static int capacity_superpanel_B(int n, int nb, int overrideB = 0) {
    auto roundPow2 = [](int x) {
        if (x <= 1) return 1;
        int lo = 1, hi = 1;
        while (hi < x) { lo = hi; hi <<= 1; }
        return (x - lo < hi - x) ? lo : hi;
    };
    constexpr double M_L2_words = 6.25e6;
    // W_HBM is not a free constant: it is W_spec[3], the SPEC frontier width at the
    // L2->HBM rung, so B = roundPow2(W_HBM * NB) is already a plan-derived width --
    // just hardcoded rather than read from the schedule. The MEASURED frontier width
    // at that rung (sched.rung[3].W) is smaller: 6 for fp32 at n=16384 against the
    // spec's 15. Driving B from the measured plan instead of the spec constant is the
    // "make nu drive the decomposition" step, so it is exposed as a measurement knob
    // and swept before anything is changed. Read once; unset reproduces the shipped
    // B exactly.
    static const int W_HBM = [] {
        const char* e = getenv("LRQR_WHBM");
        return e ? std::max(1, atoi(e)) : 15;
    }();
    const bool oneL2State = double(n) * double(n) <= M_L2_words;
    int b = overrideB ? overrideB : (oneL2State ? n : roundPow2(W_HBM * nb));
    b = std::min(b, n);
    if (b >= nb) b = (b / nb) * nb;
    if (b < nb) b = std::min(n, nb);
    return b;
}

// ---- per-size benchmark (templated on Real, NB, SLAB)
template<typename Real, int NB, int SLAB>
static void run_size(int m, int n, bool tf32, bool la, const BenchCtx& ctx) {
    size_t mn = size_t(m) * n;
    Real *A, *A0 = nullptr;
    CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
    if (ctx.single) {
        if constexpr (std::is_same_v<Real, float>)
            CURAND_CHECK(curandGenerateNormal(ctx.rg, A, mn, 0.0f, 1.0f));
        else
            CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A, mn, 0.0, 1.0));
    } else {
        CUDA_CHECK(cudaMalloc(&A0, mn * sizeof(Real)));
        if constexpr (std::is_same_v<Real, float>)
            CURAND_CHECK(curandGenerateNormal(ctx.rg, A0, mn, 0.0f, 1.0f));
        else
            CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A0, mn, 0.0, 1.0));
    }

    // ---- Continuous B-ladder (spec §2 ν-ladder at the L2/HBM rung; AUDIT_V6 §6
    // dodge (c); Theorem U: parameters vary by capacity, not input-size cases).
    // If the n×n column state fits M̂_L2=6.25e6 words, S_L2 saturates to one
    // and the whole state is one superpanel. Otherwise B is the spec's W_HBM=15
    // panel live front, rounded to a power of two and then to the actual NB leaf.
    // M̂ is already in words, so it must not be divided by sizeof(Real). Precision
    // enters only through the capacity-derived leaf NB; aspect does not enter.
    int Bsel = capacity_superpanel_B(n, NB, ctx.B);
    lrqr::Plan<Real, NB, SLAB> plan;
    if (!ctx.conly) plan.create(m, n, Bsel, tf32, la);

    // ---- LR-QR timing (V5 §2 asserts ③/④)
    // ③ Timed region ends after a device sync covering EVERY deferred stream
    //    (s0 / s1 far-rest / sP Tsp / sR rest-slice / sN domino). cudaEventRecord on the
    //    default stream captures only default-stream work (NVIDIA CUDA Runtime Event Mgmt
    //    docs: "Captures in event the contents of stream"), so without a device sync the
    //    stop event can fire while s1/sP/sR work is still in flight → undercounted time →
    //    inflated TF (the 7054-TF mechanism). geqrf's internal syncs (lrqr.cuh:6011-6014)
    //    cover s0/s1/sN/cond.-sP but NOT sR unconditionally; the explicit
    //    cudaDeviceSynchronize() below is the structural guard (matches the cuSOLVER path).
    std::vector<double> lr_ms_vec, lr_wc_vec;
    bool timing_mismatch = false;
    // Two untimed calls: the first resolves cuBLAS lazy loading, the second lets
    // the persist path (gemmsWarmed-gated) reach its steady state, so timed samples
    // share one code path and one deterministic outlier stays out of the spread.
    constexpr int warmupReps = 2;
    for (int r = 0; !ctx.conly && r < ctx.reps + warmupReps; ++r) {
        if (!ctx.single) CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
        auto wc_start = std::chrono::high_resolution_clock::now();
        Timer t; t.start();
        plan.geqrf(A, m);
        CUDA_CHECK(cudaDeviceSynchronize());   // ③ covers s0/s1/sP/sR/sN before the stop event
        float ms = t.stop();
        auto wc_end = std::chrono::high_resolution_clock::now();
        double wc_ms = std::chrono::duration<double, std::milli>(wc_end - wc_start).count();
        if (r >= warmupReps) {
            lr_ms_vec.push_back((double)ms);
            lr_wc_vec.push_back(wc_ms);
            // cross-check cudaEvent vs wall-clock (>10% ⇒ flag)
            if (ms > 1e29 || wc_ms > 1e29 || std::abs((double)ms - wc_ms) / std::max(wc_ms, 1.0) > 0.10)
                timing_mismatch = true;
        }
    }
    // ④ median-of-≥5 + min/max/spread; >10% spread ⇒ unstable (not PASS)
    std::sort(lr_ms_vec.begin(), lr_ms_vec.end());
    std::sort(lr_wc_vec.begin(), lr_wc_vec.end());
    auto med = [](const std::vector<double>& v) -> double {
        if (v.empty()) return 1e30;
        size_t n = v.size();
        return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };
    double tlr = med(lr_ms_vec);
    double chrono_lr = med(lr_wc_vec);
    double tlr_min = lr_ms_vec.empty() ? 1e30 : lr_ms_vec.front();
    double tlr_max = lr_ms_vec.empty() ? 1e30 : lr_ms_vec.back();
    double tlr_spread = (tlr > 1e29 || tlr <= 0) ? 0.0 : (tlr_max - tlr_min) / tlr;
    bool unstable = (tlr > 1e29) ? false : (tlr_spread > 0.10);
    if (!ctx.conly && (int)lr_ms_vec.size() < 5)
        printf("  WARN: only %d timed reps (V5 §2④ requires >=5); re-run with --reps 5\n",
               (int)lr_ms_vec.size());

    // relRR uses fp64 copies -- skip at huge n or via LRQR_NOCHK
    bool chk = !ctx.single && !getenv("LRQR_NOCHK") &&
               mn * sizeof(double) <= (size_t(60) << 30);
    double relrr = chk ? relRR_vec_T<Real>(ctx.cb, A0, A, m, n) : -1.0;
    double resid = -1, orth = -1;
    if (ctx.doOrth && !ctx.single && !ctx.conly)
        residual_orth_T(plan, ctx.cb, A0, A, m, n, &resid, &orth);

    // ---- cuSOLVER baseline (sgeqrf for float, dgeqrf for double)
    double tcs = 1e30, csRelrr = -1.0, csOrth = -1.0, csResid = -1.0;
    if (!ctx.lonly) {
        int lw = 0;
        if constexpr (std::is_same_v<Real, float>)
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
        else
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
        Real* work; CUDA_CHECK(cudaMalloc(&work, size_t(lw) * sizeof(Real)));
        Real* tau; CUDA_CHECK(cudaMalloc(&tau, size_t(n) * sizeof(Real)));
        int* info; CUDA_CHECK(cudaMalloc(&info, 4));
        std::vector<double> cs_ms_vec;
        for (int r = 0; r < ctx.reps + 1; ++r) {
            if (!ctx.single) CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            Timer t; t.start();
            if constexpr (std::is_same_v<Real, float>)
                CUSOLVER_CHECK(cusolverDnSgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
            else
                CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
            CUDA_CHECK(cudaDeviceSynchronize());
            float ms = t.stop();
            if (r > 0) cs_ms_vec.push_back((double)ms);
        }
        std::sort(cs_ms_vec.begin(), cs_ms_vec.end());
        tcs = cs_ms_vec.empty() ? 1e30
            : (cs_ms_vec.size() % 2 ? cs_ms_vec[cs_ms_vec.size()/2]
                                    : 0.5*(cs_ms_vec[cs_ms_vec.size()/2 - 1] + cs_ms_vec[cs_ms_vec.size()/2]));
        if (chk) csRelrr = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
        if (ctx.doOrth && !ctx.single)
            cusolver_resid_orth_T<Real>(ctx.cs, ctx.cb, A0, A, tau, m, n, &csResid, &csOrth);
        cudaFree(work); cudaFree(tau); cudaFree(info);
    }

    // DIAG: compare LR-QR vs cuSOLVER R-diag divergence (float only)
    if (getenv("LRQR_DIAG") && !ctx.conly && !ctx.lonly && !ctx.single) {
        if constexpr (std::is_same_v<Real, float>) {
            Real* A2; CUDA_CHECK(cudaMalloc(&A2, mn * sizeof(Real)));
            CUDA_CHECK(cudaMemcpy(A2, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A2, m);
            double worst; int fj = first_diag_divergence(A2, A, m, n, 1e-2, &worst);
            printf("  DIAG: first |Rjj| divergence >1e-2 at col %d (panel %d); worst rel %.2e\n",
                   fj, fj / NB, worst);
            cudaFree(A2);
        }
    }

    double fl = flops_qr(m, n);
    auto fmt_tf = [](double tf, double ms) -> std::string {
        if (ms > 1e29) return "             n/a";
        char b[48]; snprintf(b, sizeof(b), "%8.1f (%6.1f)", tf, ms); return b;
    };
    auto fmt_e = [](double v) -> std::string {
        if (v < 0) return "         -";
        char b[24]; snprintf(b, sizeof(b), "%10.2e", v); return b;
    };
    double speedup = (tlr > 1e29 || tcs > 1e29) ? 0.0 : tcs / tlr;
    double lr_tf = (tlr > 1e29) ? 0 : fl / (tlr * 1e-3) / 1e12;
    double cs_tf = (tcs > 1e29) ? 0 : fl / (tcs * 1e-3) / 1e12;

    // P0.d: determine tier name for sanity ceiling check
    const char* tier_name = nullptr;
    if constexpr (std::is_same_v<Real, double>) tier_name = "fp64";
    else tier_name = getenv("LRQR_3XTF32") ? "3xtf32" : (tf32 ? "tf32" : "fp32");

    // V5 §2①: achieved_TF > 1.05×C_tier(n) ⇒ auto-FAIL / INVALID (catches the 7054-TF bug)
    // V5 §2②: the value PRINTED in the ms column must yield the authoritative TF — catches the
    // green-ctx TF/ms column-swap. `print_tf`/`print_ms` are exactly the values emitted in the
    // LR-QR TF(ms) cell by the printf below. The assert recomputes TF from `print_ms` (the
    // printed ms) and compares to `lr_tf` (the TF from the immutable cudaEvent time `tlr`).
    // The prior version recomputed from the same internal `tlr` that `lr_tf` is derived from and
    // compared to `lr_tf` — tautological (rel_err always 0, could never fire). Comparing to
    // `lr_tf` rather than to `print_tf` is deliberate: printed_TF ≈ flops/(printed_ms·1e9) is
    // symmetric under a swap (both sides come from the same swapped pair) and cannot fire;
    // anchoring one side on the immutable measurement breaks the symmetry — a column swap
    // reassigns print_ms to the TF value, so flops/(print_ms·1e9) becomes `tlr` ≠ `lr_tf`.
    double print_tf = lr_tf;
    double print_ms = tlr;
    bool sanity_fail = false;
    double ceil_c = get_ceiling_tflops(tier_name, n);
    if (ceil_c > 0 && lr_tf > 1.05 * ceil_c) {
        sanity_fail = true;
        g_sanity_invalid++;
        printf("  SANITY-FAIL-1: TF=%.1f > 1.05xC=%.1f (%s@%d) — timing bug (missing device sync?)\n",
               lr_tf, ceil_c, tier_name, n);
    }
    if (tlr < 1e29 && tlr > 0.0 && lr_tf > 0.0 && print_ms > 0.0) {
        // recompute TF from the PRINTED ms (print_ms) via the analytic flop count; compare to
        // the authoritative lr_tf (TF from the measured cudaEvent time tlr).
        // A2: use the shared shape-general count, not a second inline copy of the
        // m>=n formula -- a self-consistency check computed with a different flop
        // model than the value it checks is not a check.
        double flops_check = flops_qr((double)m, (double)n);
        double tf_from_printed_ms = flops_check / (print_ms * 1e-3) / 1e12;
        double rel_err = std::abs(tf_from_printed_ms - lr_tf) / std::max(lr_tf, 1e-12);
        if (rel_err > 0.01) {
            sanity_fail = true;
            g_sanity_invalid++;
            printf("  SANITY-FAIL-2: TF/ms mismatch — printed TF=%.3f, printed ms=%.3f, "
                   "flops/(printed_ms*1e9)=%.3f vs authoritative TF=%.3f (rel %.2f%%) — TF/ms column swap?\n",
                   print_tf, print_ms, tf_from_printed_ms, lr_tf, rel_err * 100.0);
        }
    }
    // V5 §2③: device-sync coverage is enforced structurally in the timing loop (see ③ above);
    // it covers s0/s1/sP/sR/sN — a primitive/GEMM rate (no deferred streams) cannot pose as the QR.
    // V5 §2④: spread > 10% ⇒ unstable, re-measure (not PASS)
    if (unstable) {
        sanity_fail = true;
        g_sanity_invalid++;
        printf("  SANITY-FAIL-4: unstable — spread=%.1f%% (min=%.2f med=%.2f max=%.2f ms over %d reps); re-measure\n",
               tlr_spread * 100.0, tlr_min, tlr, tlr_max, (int)lr_ms_vec.size());
    }
    // cross-check cudaEvent vs wall-clock (>10% ⇒ flag)
    if (timing_mismatch) {
        sanity_fail = true;
        g_sanity_invalid++;
        printf("  SANITY-FAIL-3: cudaEvent=%.2fms vs wallclock=%.2fms (>10%%) — timer not wrapping whole QR\n",
               tlr, chrono_lr);
    }

    // Parity ratios
    auto fmt_ratio = [](double lr, double cs) -> std::string {
        if (lr < 0 || cs < 0 || cs == 0) return "   n/a";
        char b[16]; snprintf(b, sizeof(b), "%6.2fx", lr / cs); return b;
    };
    printf("  %7d  %s  %s  %7.2fx  %s  %s  %s  %s  %s  %s  %s",
           n,
           fmt_tf(print_tf, print_ms).c_str(),
           fmt_tf(cs_tf, tcs).c_str(),
           speedup,
           fmt_e(relrr).c_str(),
           fmt_e(csRelrr).c_str(),
           fmt_e(orth).c_str(),
           fmt_e(csOrth).c_str(),
           fmt_e(resid).c_str(),
           fmt_ratio(orth, csOrth).c_str(),
           fmt_ratio(resid, csResid).c_str());
    if (sanity_fail) printf("  SANITY-FAIL");
    if (unstable) printf("  UNSTABLE");
    if (timing_mismatch) printf("  TIMING-MISMATCH");
    if (sanity_fail || unstable || timing_mismatch) printf("  INVALID");
    printf("\n");
    // V5 §2④: report median/min/max/spread on every timed LR-QR row
    if (tlr < 1e29 && !ctx.conly)
        printf("    reps=%d  median=%.2fms  min=%.2fms  max=%.2fms  spread=%.1f%%  %s\n",
               (int)lr_ms_vec.size(), tlr, tlr_min, tlr_max, tlr_spread * 100.0,
               unstable ? "UNSTABLE-re-measure" : "stable");
    fflush(stdout);

    // ---- kappa column-scaling sweep (cheap conditioning proxy)
    if (ctx.kappaCol && n <= 16384 && !ctx.single && !ctx.conly) {
        for (double lk : {2.0, 6.0, 10.0, 14.0}) {
            std::vector<Real> hs(n);
            for (int j = 0; j < n; ++j) hs[j] = (Real)std::pow(10.0, -lk * j / (n - 1));
            Real* ds; CUDA_CHECK(cudaMalloc(&ds, n * sizeof(Real)));
            CUDA_CHECK(cudaMemcpy(ds, hs.data(), n * sizeof(Real), cudaMemcpyHostToDevice));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
            if constexpr (std::is_same_v<Real, float>)
                CURAND_CHECK(curandGenerateNormal(ctx.rg, A0, mn, 0.0f, 1.0f));
            else
                CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A0, mn, 0.0, 1.0));
            if constexpr (std::is_same_v<Real, float>)
                k_scale_cols<<<(unsigned)((mn + 255) / 256), 256>>>(A0, m, m, n, ds);
            else
                k_scale_cols_d<<<(unsigned)((mn + 255) / 256), 256>>>(A0, m, m, n, ds);
            CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A, m);
            printf("    kappa~1e%-3.0f relRR=%.2e\n", lk, relRR_vec_T<Real>(ctx.cb, A0, A, m, n));
            fflush(stdout);
            cudaFree(ds);
        }
    }

    // ---- kappa SVD sweep (exact kappa via U diag(s) V^T)
    if (ctx.kappaSvd && n <= 16384 && !ctx.single && !ctx.conly) {
        Real *Ubuf, *Vbuf;
        CUDA_CHECK(cudaMalloc(&Ubuf, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&Vbuf, size_t(n) * n * sizeof(Real)));
        printf("    [SVD k-sweep, n=%d]\n", n);
        double kr_min = 1e30, kr_max = 0;
        for (double kappa : {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14}) {
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
            gen_svd_T<Real>(ctx.cs, ctx.cb, ctx.rg, A0, m, n, kappa, Ubuf, Vbuf);
            CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A, m);
            double kr = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
            kr_min = std::min(kr_min, kr);
            kr_max = std::max(kr_max, kr);
            printf("    kappa=1e%-9.0f relRR=%.3e\n", std::log10(kappa), kr);
            fflush(stdout);
        }
        double flat_ratio = kr_max / std::max(kr_min, 1e-300);
        printf("    kappa-flatness: max/min=%.2fx (%.3e / %.3e)\n", flat_ratio, kr_max, kr_min);
        fflush(stdout);
        cudaFree(Ubuf); cudaFree(Vbuf);
    }

    if (!ctx.conly) plan.destroy();
    cudaFree(A); if (A0) cudaFree(A0);
}

// ---- validation targets per precision tier (V7: order-of-magnitude gates)
// fp64: resid & orth within 10x cuSOLVER Dgeqrf (same OoM, ~1e-15..1e-14)
// fp32: resid & orth within 10x cuSOLVER Sgeqrf (same OoM, ~1e-6)
// 3xTF32: resid within 10x cuSOLVER Sgeqrf (fp32-OoM at every n)
// TF32: resid < 1e-3 (order 1e-4) AND orth within 10x cuSOLVER Sgeqrf
enum GateMode { GATE_OOM_BOTH, GATE_OOM_RESID, GATE_HARD_RESID_OOM_ORTH };
struct ValTargets {
    const char* name;
    double u;                 // unit roundoff (informational)
    GateMode gate;            // OoM gate mode per tier
    double oom;               // max LR/cuSOLVER ratio for same OoM (10x)
    double hard_resid;        // hard residual limit (TF32 only; -1 otherwise)
};

// --validate-lapack applies absolute, one-norm QR tests. It complements the
// comparative --validate-all suite and reports thin-Q metrics for wide matrices.

// R_thin (k x n) from the factored A: upper triangle, zeros below.
template<typename Real>
__global__ static void k_extract_R_thin(const Real* Afac, size_t lda, Real* R, size_t ldr,
                                        int k, int n) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)k * n) return;
    int j = (int)(t / k), i = (int)(t % k);
    R[(size_t)j * ldr + i] = (i <= j) ? Afac[(size_t)j * lda + i] : Real(0);
}
// I - G, in place, on a k x k buffer
template<typename Real>
__global__ static void k_eye_minus(Real* G, size_t ldg, int k) {
    size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)k * k) return;
    int j = (int)(t / k), i = (int)(t % k);
    Real v = G[(size_t)j * ldg + i];
    G[(size_t)j * ldg + i] = (i == j) ? (Real(1) - v) : (Real(0) - v);
}

struct LapackShape { int m, n; const char* cls; };

template<typename Real, int NB, int SLAB>
static int run_validate_lapack_T(bool tf32, bool la, const BenchCtx& ctx,
                                 const ValTargets& tgt, int* out_pass, int* out_total) {
    const double THRESH = 30.0;                 // dchkqr's default THRESH
    const double eps = tgt.u;                   // DLAMCH('Epsilon')
    const double eps_prec = 2.0 * tgt.u;        // DLAMCH('Precision') = eps*base
    // SMALL/LARGE are about REPRESENTABILITY, so they use the STORAGE type's
    // safmin/eps -- float for the tf32 tiers, whose u is far larger than fp32's.
    const double safmin = std::is_same_v<Real, double> ? 2.2250738585072014e-308
                                                       : 1.1754943508222875e-38;
    const double eps_store = std::is_same_v<Real, double> ? 2.220446049250313e-16
                                                          : 1.1920928955078125e-07;

    const LapackShape shapes[] = {
        {1024, 1024, "square"},
        {2048, 1024, "tall"},
        {4096,  512, "tall"},
        {1024, 2048, "wide"},        // needs A2
        { 512, 4096, "wide"},        // needs A2
        {1024, 1000, "n%NB!=0"},     // needs A1
        {2048, 3000, "n%NB!=0"},     // needs A1
        {65536, 512, "extreme"},     // aspect 128:1 -- selects c_L>1, exercising
                                     // ormqr's merge-tree branch rather than the
                                     // single-T WY path every square cell takes
    };

    int pass = 0, total = 0;
    printf("\n=== LAPACK QR test suite (dqrt01 criteria, THRESH=%.0f) ===\n", THRESH);
    printf("Tier: %-6s  eps=%.3e  eps_precision=%.3e\n", tgt.name, eps, eps_prec);
    printf("  %-7s %-7s %-9s %-18s %12s %12s %12s %12s  %s\n",
           "m", "n", "class", "type", "RESULT(1)", "RESULT(2)", "ormqr", "Xfwd", "verdict");

    for (const auto& S : shapes) {
        const int m = S.m, n = S.n, k = std::min(m, n);
        // A1: an n that is not a multiple of NB is handled by ZERO-PADDING, at the
        // caller, with no kernel change. Householder proceeds left to right, so the
        // reflectors of the first n columns are computed from those columns alone --
        // factoring [A | 0] leaves them untouched, and lrqr.cuh:1338's `sg > 0` guard
        // gives the padded columns tau=0 identity reflectors, which is dlarfg's own
        // convention. So Q[:, 0:k] and R[0:k, 0:n] are exactly A's, and every figure
        // below is reported on the USER's n, never the padded one.
        //
        // Padding is deliberately NOT put inside Plan: it costs an extra <= NB-1
        // columns of A, and burying that in create() would make G7's compact-storage
        // accounting quietly include padding as payload. The one caller that needs it
        // pays for it visibly.
        const int n_pad = ((n + NB - 1) / NB) * NB;
        const bool padded = (n_pad != n);
        for (int imat = 1; imat <= 8; ++imat) {
            const LapackQRParams P = lapack_qr_params(imat, eps_prec, safmin);
            const char* tname = P.name;
            bool lower_became_upper = (imat == 3 && m != n);

            // Storage is m x n_pad; the LOGICAL matrix stays m x n.
            const size_t mn_pad = (size_t)m * n_pad, mn = (size_t)m * n;
            Real *A0, *A;
            CUDA_CHECK(cudaMalloc(&A0, mn_pad * sizeof(Real)));
            CUDA_CHECK(cudaMalloc(&A,  mn_pad * sizeof(Real)));
            k_zero_T<Real><<<(unsigned)((mn_pad + 255) / 256), 256>>>(A0, mn_pad);
            CUDA_CHECK(cudaDeviceSynchronize());
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42 + imat));
            gen_lapack_qr_type_T<Real>(imat, ctx.cs, ctx.cb, ctx.rg, A0, m, n,
                                       eps_prec, safmin * (eps_prec / eps_store));

            // Norms and residuals are over the USER's m x n block throughout.
            const double anorm = norm1_T<Real>(A0, m, m, n);

            int Bsel = capacity_superpanel_B(n_pad, NB, ctx.B);
            lrqr::Plan<Real, NB, SLAB> plan;
            plan.create(m, n_pad, Bsel, tf32, la);
            CUDA_CHECK(cudaMemcpy(A, A0, mn_pad * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A, m);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Q_thin. orgqr forms m x min(m, n_pad); only the first k = min(m,n)
            // columns belong to A -- the padded columns' reflectors are identities
            // and cannot change columns to their left, so Q[:, 0:k] is exactly A's.
            const int k_pad = std::min(m, n_pad);
            Real* Q; CUDA_CHECK(cudaMalloc(&Q, (size_t)m * k_pad * sizeof(Real)));
            plan.orgqr(A, m, Q, m);
            CUDA_CHECK(cudaDeviceSynchronize());

            // RESULT(1): E = R_thin - Q' A0, one-norm over k x n
            Real* E; CUDA_CHECK(cudaMalloc(&E, (size_t)k * n * sizeof(Real)));
            k_extract_R_thin<Real><<<(unsigned)(((size_t)k * n + 255) / 256), 256>>>(
                A, m, E, k, k, n);
            {
                Real one = Real(1), neg = Real(-1);
                if constexpr (std::is_same_v<Real, float>)
                    CUBLAS_CHECK(cublasSgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
                                             &neg, Q, m, A0, m, &one, E, k));
                else
                    CUBLAS_CHECK(cublasDgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
                                             &neg, Q, m, A0, m, &one, E, k));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            const double resid1 = norm1_T<Real>(E, k, k, n);
            const double r1 = (anorm > 0.0)
                ? ((resid1 / (double)std::max(1, m)) / anorm) / eps : 0.0;

            // RESULT(2): I - Q'Q, one-norm over k x k
            Real* G; CUDA_CHECK(cudaMalloc(&G, (size_t)k * k * sizeof(Real)));
            {
                Real one = Real(1), zero = Real(0);
                if constexpr (std::is_same_v<Real, float>)
                    CUBLAS_CHECK(cublasSgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m,
                                             &one, Q, m, Q, m, &zero, G, k));
                else
                    CUBLAS_CHECK(cublasDgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m,
                                             &one, Q, m, Q, m, &zero, G, k));
                k_eye_minus<Real><<<(unsigned)(((size_t)k * k + 255) / 256), 256>>>(G, k, k);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            const double r2 = (norm1_T<Real>(G, k, k, k) / (double)std::max(1, m)) / eps;

            // ---- dqrt03 analogue: ormqr against the explicitly-formed Q --------
            // LAPACK's dqrt03 checks DORMQR by comparing it to an explicit Q
            // multiply, in all four side/trans combinations:
            //     norm( op(Q)*C - op(Q_explicit)*C ) / (max(m,n) * norm(C) * eps)
            // Q here is already formed for RESULT(1)/(2), so it is the reference and
            // ormqr -- which never forms Q -- is the thing under test. That is
            // exactly the property the compact representation is supposed to have.
            double r3_worst = 0.0; int r3_done = 0; bool r3_unsupported = false;
            {
                const int nrhs = 64;
                // C_L is m x nrhs (left cases); C_R is nrhs x m (right cases)
                Real *CL, *CL_ref, *CR, *CR_ref;
                CUDA_CHECK(cudaMalloc(&CL, (size_t)m * nrhs * sizeof(Real)));
                CUDA_CHECK(cudaMalloc(&CL_ref, (size_t)m * nrhs * sizeof(Real)));
                CUDA_CHECK(cudaMalloc(&CR, (size_t)nrhs * m * sizeof(Real)));
                CUDA_CHECK(cudaMalloc(&CR_ref, (size_t)nrhs * m * sizeof(Real)));
                if constexpr (std::is_same_v<Real, float>)
                    CURAND_CHECK(curandGenerateNormal(ctx.rg, CL, (size_t)m * nrhs, 0.f, 1.f));
                else
                    CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, CL, (size_t)m * nrhs, 0., 1.));
                CUDA_CHECK(cudaMemcpy(CL_ref, CL, (size_t)m * nrhs * sizeof(Real),
                                      cudaMemcpyDeviceToDevice));
                const double cnorm = norm1_T<Real>(CL, m, m, nrhs);
                const double den = (double)std::max(m, n) * (cnorm > 0 ? cnorm : 1.0) * eps;

                // 'L','T': ormqr gives Q'C over the first k rows; the reference is
                // Q_explicit' * C, a k x nrhs GEMM.
                Real* refT; CUDA_CHECK(cudaMalloc(&refT, (size_t)k_pad * nrhs * sizeof(Real)));
                Real one = Real(1), zero = Real(0), neg = Real(-1);
                if constexpr (std::is_same_v<Real, float>)
                    CUBLAS_CHECK(cublasSgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k_pad, nrhs, m,
                                             &one, Q, m, CL_ref, m, &zero, refT, k_pad));
                else
                    CUBLAS_CHECK(cublasDgemm(ctx.cb, CUBLAS_OP_T, CUBLAS_OP_N, k_pad, nrhs, m,
                                             &one, Q, m, CL_ref, m, &zero, refT, k_pad));
                if (!plan.ormqr(A, m, 'L', 'T', CL, m, m, nrhs)) {
                    r3_unsupported = true;
                } else {
                    // compare the leading k_pad rows
                    Real* D; CUDA_CHECK(cudaMalloc(&D, (size_t)k_pad * nrhs * sizeof(Real)));
                    CUDA_CHECK(cudaMemcpy2D(D, (size_t)k_pad * sizeof(Real),
                                            CL, (size_t)m * sizeof(Real),
                                            (size_t)k_pad * sizeof(Real), (size_t)nrhs,
                                            cudaMemcpyDeviceToDevice));
                    if constexpr (std::is_same_v<Real, float>)
                        CUBLAS_CHECK(cublasSaxpy(ctx.cb, (size_t)k_pad * nrhs, &neg, refT, 1, D, 1));
                    else
                        CUBLAS_CHECK(cublasDaxpy(ctx.cb, (size_t)k_pad * nrhs, &neg, refT, 1, D, 1));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    r3_worst = std::max(r3_worst, norm1_T<Real>(D, k_pad, k_pad, nrhs) / den);
                    ++r3_done;
                    cudaFree(D);
                }
                // 'L','N': tested as a ROUND TRIP, Q(Q^T C) == C, rather than
                // against a thin-Q reference. Q_thin * C[0:k,:] is NOT Q_full * C
                // unless m == k, so the thin reference is only valid for square
                // cells -- it is what made 4096x512 "fail" while 1024x1024 passed.
                // The round trip needs no explicit Q at all, which is the point.
                if (!r3_unsupported) {
                    CUDA_CHECK(cudaMemcpy(CL, CL_ref, (size_t)m * nrhs * sizeof(Real),
                                          cudaMemcpyDeviceToDevice));
                    if (plan.ormqr(A, m, 'L', 'T', CL, m, m, nrhs) &&
                        plan.ormqr(A, m, 'L', 'N', CL, m, m, nrhs)) {
                        if constexpr (std::is_same_v<Real, float>)
                            CUBLAS_CHECK(cublasSaxpy(ctx.cb, (size_t)m * nrhs, &neg, CL_ref, 1, CL, 1));
                        else
                            CUBLAS_CHECK(cublasDaxpy(ctx.cb, (size_t)m * nrhs, &neg, CL_ref, 1, CL, 1));
                        CUDA_CHECK(cudaDeviceSynchronize());
                        r3_worst = std::max(r3_worst, norm1_T<Real>(CL, m, m, nrhs) / den);
                        ++r3_done;
                    }
                }
                // 'R','N' / 'R','T': round trip (C Q) Q^T == C, on an nrhs x m C.
                // Also needs no explicit Q. c_L>1 cells refuse the right side, and a
                // refusal is recorded by r3_done not advancing, never as a pass.
                {
                    if constexpr (std::is_same_v<Real, float>)
                        CURAND_CHECK(curandGenerateNormal(ctx.rg, CR, (size_t)nrhs * m, 0.f, 1.f));
                    else
                        CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, CR, (size_t)nrhs * m, 0., 1.));
                    CUDA_CHECK(cudaMemcpy(CR_ref, CR, (size_t)nrhs * m * sizeof(Real),
                                          cudaMemcpyDeviceToDevice));
                    const double crn = norm1_T<Real>(CR, nrhs, nrhs, m);
                    const double denR = (double)std::max(m, n) * (crn > 0 ? crn : 1.0) * eps;
                    if (plan.ormqr(A, m, 'R', 'N', CR, nrhs, nrhs, m) &&
                        plan.ormqr(A, m, 'R', 'T', CR, nrhs, nrhs, m)) {
                        if constexpr (std::is_same_v<Real, float>)
                            CUBLAS_CHECK(cublasSaxpy(ctx.cb, (size_t)nrhs * m, &neg, CR_ref, 1, CR, 1));
                        else
                            CUBLAS_CHECK(cublasDaxpy(ctx.cb, (size_t)nrhs * m, &neg, CR_ref, 1, CR, 1));
                        CUDA_CHECK(cudaDeviceSynchronize());
                        r3_worst = std::max(r3_worst, norm1_T<Real>(CR, nrhs, nrhs, m) / denR);
                        ++r3_done;
                    }
                }
                cudaFree(refT);
                cudaFree(CL); cudaFree(CL_ref); cudaFree(CR); cudaFree(CR_ref);
            }

            // ---- dgeqrs analogue: least-squares solve off the compact form -------
            //     norm( B - A*X ) / ( norm(A) * norm(X) * eps ),  X = R^{-1} Q' B
            // Uses ormqr then trm_R, so neither step forms Q and R is never copied
            // out of triu(A). Only meaningful when R is nonsingular, so it is skipped
            // for the underflow type, whose R is numerically zero.
            double r4 = 0.0, xfwd = -1.0; bool r4_done = false;
            if (!r3_unsupported && k == n && imat != 7) {
                const int nrhs = 8;
                Real *Bv, *Bref, *X;
                CUDA_CHECK(cudaMalloc(&Bv, (size_t)m * nrhs * sizeof(Real)));
                CUDA_CHECK(cudaMalloc(&Bref, (size_t)m * nrhs * sizeof(Real)));
                CUDA_CHECK(cudaMalloc(&X, (size_t)m * nrhs * sizeof(Real)));
                // dgeqrs' test builds a CONSISTENT system: X_true is drawn, then
                // B = A*X_true, so the least-squares residual is zero up to rounding.
                // A random B would leave the true LS residual -- O(||B||) whenever
                // m > n -- and read as a solver failure on every tall cell.
                Real* Xtrue; CUDA_CHECK(cudaMalloc(&Xtrue, (size_t)n * nrhs * sizeof(Real)));
                if constexpr (std::is_same_v<Real, float>)
                    CURAND_CHECK(curandGenerateNormal(ctx.rg, Xtrue, (size_t)n * nrhs, 0.f, 1.f));
                else
                    CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, Xtrue, (size_t)n * nrhs, 0., 1.));
                {
                    Real one1 = Real(1), zero1 = Real(0);
                    if constexpr (std::is_same_v<Real, float>)
                        CUBLAS_CHECK(cublasSgemm(ctx.cb, CUBLAS_OP_N, CUBLAS_OP_N, m, nrhs, n,
                                                 &one1, A0, m, Xtrue, n, &zero1, Bv, m));
                    else
                        CUBLAS_CHECK(cublasDgemm(ctx.cb, CUBLAS_OP_N, CUBLAS_OP_N, m, nrhs, n,
                                                 &one1, A0, m, Xtrue, n, &zero1, Bv, m));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                if (plan.ormqr(A, m, 'L', 'T', Bv, m, m, nrhs)) {
                    CUDA_CHECK(cudaMemcpy(X, Bv, (size_t)m * nrhs * sizeof(Real),
                                          cudaMemcpyDeviceToDevice));
                    plan.trm_R(A, m, X, m, nrhs, /*solve=*/true);
                    // residual B - A*X  (A is m x n, X is n x nrhs stored at ld=m)
                    Real one = Real(1), neg = Real(-1);
                    if constexpr (std::is_same_v<Real, float>)
                        CUBLAS_CHECK(cublasSgemm(ctx.cb, CUBLAS_OP_N, CUBLAS_OP_N, m, nrhs, n,
                                                 &neg, A0, m, X, m, &one, Bref, m));
                    else
                        CUBLAS_CHECK(cublasDgemm(ctx.cb, CUBLAS_OP_N, CUBLAS_OP_N, m, nrhs, n,
                                                 &neg, A0, m, X, m, &one, Bref, m));
                    CUDA_CHECK(cudaDeviceSynchronize());
                    // Forward error against the known X_true. This DISCRIMINATES:
                    // if X == X_true the solve chain is right and any large residual
                    // is a defect in the residual computation, not in trm_R.
                    {
                        Real* XD; CUDA_CHECK(cudaMalloc(&XD, (size_t)n * nrhs * sizeof(Real)));
                        CUDA_CHECK(cudaMemcpy2D(XD, (size_t)n * sizeof(Real),
                                                X, (size_t)m * sizeof(Real),
                                                (size_t)n * sizeof(Real), (size_t)nrhs,
                                                cudaMemcpyDeviceToDevice));
                        Real negl = Real(-1);
                        if constexpr (std::is_same_v<Real, float>)
                            CUBLAS_CHECK(cublasSaxpy(ctx.cb, (size_t)n * nrhs, &negl, Xtrue, 1, XD, 1));
                        else
                            CUBLAS_CHECK(cublasDaxpy(ctx.cb, (size_t)n * nrhs, &negl, Xtrue, 1, XD, 1));
                        CUDA_CHECK(cudaDeviceSynchronize());
                        const double xt = norm1_T<Real>(Xtrue, n, n, nrhs);
                        xfwd = (xt > 0) ? norm1_T<Real>(XD, n, n, nrhs) / xt : -1.0;
                        cudaFree(XD);
                    }
                    const double xn = norm1_T<Real>(X, m, n, nrhs);
                    const double bn = norm1_T<Real>(Bref, m, m, nrhs);
                    // LAPACK's solve residual carries the DIMENSION factor as well
                    // (DGET02 form): norm(B-AX) / (max(m,n)*norm(A)*norm(X)*eps).
                    // Omitting it read 1.481e+03 on a cell whose absolute relative
                    // residual is 1.6e-13 -- the solve was right and the divisor wrong.
                    const double dfac = (double)std::max(m, n);
                    if (anorm > 0 && xn > 0) { r4 = bn / (dfac * anorm * xn * eps); r4_done = true; }
                }
                cudaFree(Bv); cudaFree(Bref); cudaFree(X); cudaFree(Xtrue);
            }

            // Xfwd GATES. The forward error ||X - X_true||_1/||X_true||_1 against the
            // known solution of the consistent system settled what the residual form
            // could not: it reads 4e-17..8e-16 on every cell, so the solve chain
            // (ormqr then trm_R) is correct to machine precision and the earlier ~1e12
            // "solve" figures were a defect in MY residual computation, not in trm_R.
            // The forward error is also the stronger test for a consistent system --
            // it cannot be passed by a wrong X that happens to have a small residual --
            // so it replaces the residual rather than sitting beside it.
            const bool ok = (r1 < THRESH) && (r2 < THRESH)
                         && (r3_done == 0 || r3_worst < THRESH)
                         && (!r4_done || (xfwd >= 0.0 && xfwd < THRESH * P.cndnum * eps));
            // The Xfwd bound is THRESH * CNDNUM * eps, not THRESH * eps. A solve's
            // FORWARD error legitimately scales with the condition number -- that is
            // the standard perturbation bound -- which is precisely why LAPACK's
            // dgeqrs test uses a BACKWARD (residual) test instead: the residual is
            // O(eps) whatever the conditioning, and LAPACK does not know kappa at
            // test time. This suite GENERATES the matrix, so CNDNUM is known exactly
            // and the sharper bound is available. Gating at O(eps) failed all 15
            // full-badc1/full-badc2 cells by construction: badc2 is kappa = 0.1/eps,
            // so kappa*eps ~ 0.05 and the measured 1.2e-2 was correct behaviour
            // being reported as a failure.
            char b3[24], b4[24];
            if (r3_unsupported) snprintf(b3, sizeof b3, "%12s", "c_L>1 n/s");
            else if (r3_done)   snprintf(b3, sizeof b3, "%12.3e", r3_worst);
            else                snprintf(b3, sizeof b3, "%12s", "-");
            if (r4_done) snprintf(b4, sizeof b4, "%12.3e", xfwd);
            else         snprintf(b4, sizeof b4, "%12s", "-");
            printf("  %-7d %-7d %-9s %-18s %12.3e %12.3e %s %s  %s%s%s%s\n",
                   m, n, S.cls, tname, r1, r2, b3, b4, ok ? "PASS" : "FAIL",
                   (!ok && r1 >= THRESH) ? " [resid]" : "",
                   (!ok && r2 >= THRESH) ? " [orth]" : "",
                   padded ? "  (n zero-padded to NB multiple; figures on the user n)" : "");
            if (lower_became_upper)
                printf("        (imat 3 on a non-square cell: emitted UPPER triangular -- a "
                       "transpose would change the shape)\n");
            fflush(stdout);
            ++total; if (ok) ++pass;

            plan.destroy();
            cudaFree(A0); cudaFree(A); cudaFree(Q); cudaFree(E); cudaFree(G);
        }
    }
    if (out_pass) *out_pass = pass;
    if (out_total) *out_total = total;
    printf("\n=== LAPACK suite summary: %d/%d PASS (THRESH=%.0f) ===\n", pass, total, THRESH);
    printf("    Xfwd = ||X - X_true||_1 / ||X_true||_1 for the consistent system\n"
           "    B = A*X_true, solved as trm_R(ormqr(B)) -- neither step forms Q.\n");
    fflush(stdout);
    return pass == total ? 0 : 1;
}

// ---- tall-skinny validation at multiple aspect ratios (panel-bound regime)
// Tests m >> n where the factorization is update-bound and the panel/leaf path
// dominates. Three aspect ratios per spec acceptance:
//   16384x1024   (  16:1, mid-tall)
//   100000x256   (~390:1, tall)
//   1048576x64   (16384:1, extreme tall-skinny / TSQR regime)
// NB is dispatched per shape: float tiers use NB=128 where n%128==0, else NB=64
// (so the 1Mx64 shape runs at NB=64); double uses NB=64 throughout.
template<typename Real>
static void run_tall_skinny_T(bool tf32, bool la, const BenchCtx& ctx, const ValTargets& tgt,
                              int& total_tests, int& total_pass) {
    printf("\n--- Tall-skinny correctness (m >> n) ---\n");
    printf("  %7s  %7s  %10s  %10s  %10s  %10s  %10s  %10s  %7s  %7s  %s\n",
           "m", "n", "LR resid", "CS resid", "r/ratio", "LR orth", "CS orth", "o/ratio", "relRR_r", "Result", "tag");
    struct TS { int m, n; const char* tag; };
    for (auto s : {TS{16384, 1024, "16:1"}, TS{100000, 256, "390:1"}, TS{1048576, 64, "16384:1"}}) {
        int m = s.m, n = s.n;
        size_t mn = size_t(m) * n;
        int nb;
        if constexpr (std::is_same_v<Real, double>) nb = 64;
        else nb = (n % 128 == 0) ? 128 : 64;
        if (n % nb != 0) {
            printf("  %7d  %7d  (skip: n%%NB!=0)\n", m, n);
            continue;
        }

        Real *A0, *A;
        CUDA_CHECK(cudaMalloc(&A0, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
        if constexpr (std::is_same_v<Real, float>)
            CURAND_CHECK(curandGenerateNormal(ctx.rg, A0, mn, 0.0f, 1.0f));
        else
            CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A0, mn, 0.0, 1.0));

        int Bsel = capacity_superpanel_B(n, nb, ctx.B);

        // dispatch on the compile-time NB/SLAB template params of Plan
        auto run_one = [&](auto plan_dummy) {
            using PlanT = decltype(plan_dummy);
            PlanT plan;
            plan.create(m, n, Bsel, tf32, la);
            CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A, m);
            double lr_relRR = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
            double lr_resid, lr_orth;
            residual_orth_T(plan, ctx.cb, A0, A, m, n, &lr_resid, &lr_orth);
            plan.destroy();

            // cuSOLVER baseline (sgeqrf for float, dgeqrf for double)
            CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            int lw = 0;
            if constexpr (std::is_same_v<Real, float>)
                CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
            else
                CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
            Real* work; CUDA_CHECK(cudaMalloc(&work, size_t(lw) * sizeof(Real)));
            Real* tau; CUDA_CHECK(cudaMalloc(&tau, size_t(n) * sizeof(Real)));
            int* info; CUDA_CHECK(cudaMalloc(&info, 4));
            if constexpr (std::is_same_v<Real, float>)
                CUSOLVER_CHECK(cusolverDnSgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
            else
                CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
            CUDA_CHECK(cudaDeviceSynchronize());
            double cs_relRR = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
            double cs_resid, cs_orth;
            cusolver_resid_orth_T<Real>(ctx.cs, ctx.cb, A0, A, tau, m, n, &cs_resid, &cs_orth);
            cudaFree(work); cudaFree(tau); cudaFree(info);

            // V7: order-of-magnitude gate (resid + orth ratio vs cuSOLVER)
            double rr_ratio = (cs_resid > 0) ? lr_resid / cs_resid : 0;
            double ro_ratio = (cs_orth > 0) ? lr_orth / cs_orth : 0;
            double relrr_ratio = (cs_relRR > 0) ? lr_relRR / cs_relRR : 0;
            bool pass;
            if (tgt.gate == GATE_OOM_BOTH) {
                pass = rr_ratio <= tgt.oom && ro_ratio <= tgt.oom;
            } else if (tgt.gate == GATE_OOM_RESID) {
                pass = rr_ratio <= tgt.oom;
            } else {
                pass = lr_resid < tgt.hard_resid && ro_ratio <= tgt.oom;
            }
            printf("  %7d  %7d  %10.2e  %10.2e  %7.2fx  %10.2e  %10.2e  %7.2fx  %7.2fx  %s  [%s]\n",
                   m, n, lr_resid, cs_resid, rr_ratio, lr_orth, cs_orth, ro_ratio, relrr_ratio,
                   pass ? "PASS" : "FAIL", s.tag);
            if (!pass && m == 1048576 && n == 64)
                printf("    DOCUMENTED spec-predicted degradation -- see G5_ERROR_MODEL_MEMO.md\n");
            fflush(stdout);
            total_tests++;
            if (pass) total_pass++;
        };

        if constexpr (std::is_same_v<Real, double>)
            run_one(lrqr::Plan<double, 64, 128>{});
        else if (nb == 128)
            run_one(lrqr::Plan<float, 128, 128>{});
        else
            run_one(lrqr::Plan<float, 64, 128>{});

        cudaFree(A0); cudaFree(A);
    }
}

// Comprehensive validation suite using tiled metrics at every supported size.
template<typename Real, int NB, int SLAB>
static int run_validate_T(bool tf32, bool la, const BenchCtx& ctx, const ValTargets& tgt,
                          int* out_pass = nullptr, int* out_total = nullptr) {
    int total_pass = 0, total_tests = 0;
    // AUDIT_V6 §3: validate-all stopped at 32768 — the 2.49e-4@65536 3xTF32
    // regression shipped invisible. Extend to the largest fitting size per tier.
    // float tiers: 98304^2*4 = 38.7 GB/matrix (fits H200 141 GB with Q + orth bufs).
    // fp64: 65536^2*8 = 34.4 GB/matrix (98304^2*8 = 77.4 GB does NOT fit with Q).
    std::vector<int> val_sizes;
    if constexpr (std::is_same_v<Real, double>)
        val_sizes = {1024, 4096, 8192, 16384, 32768, 65536};
    else
        val_sizes = {1024, 4096, 8192, 16384, 32768, 65536, 98304};
    double val_kappas[] = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14};

    printf("=== LR-QR Validation Suite (V7 OoM gates) ===\n");
    if (tgt.gate == GATE_OOM_BOTH)
        printf("Tier: %-6s  u=%.2e  OoM: resid&orth <= %.0fx cuSOLVER\n",
               tgt.name, tgt.u, tgt.oom);
    else if (tgt.gate == GATE_OOM_RESID)
        printf("Tier: %-6s  u=%.2e  OoM: resid <= %.0fx cuSOLVER\n",
               tgt.name, tgt.u, tgt.oom);
    else
        printf("Tier: %-6s  u=%.2e  resid < %.1e AND orth <= %.0fx cuSOLVER\n",
               tgt.name, tgt.u, tgt.hard_resid, tgt.oom);
    printf("\n");

    // --- Random matrix correctness test (LR-QR vs cuSOLVER, parity gate) ---
    printf("--- Random matrix correctness (LR-QR vs cuSOLVER, OoM gate) ---\n");
    printf("  %7s  %10s  %10s  %7s  %10s  %10s  %7s  %10s  %10s  %s\n",
           "n", "LR resid", "CS resid", "r_ratio", "LR orth", "CS orth", "o_ratio", "LR relRR", "CS relRR", "Result");

    for (int n : val_sizes) {
        int m = n;
        size_t mn = size_t(m) * n;
        Real *A0, *A;
        CUDA_CHECK(cudaMalloc(&A0, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
        if constexpr (std::is_same_v<Real, float>)
            CURAND_CHECK(curandGenerateNormal(ctx.rg, A0, mn, 0.0f, 1.0f));
        else
            CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A0, mn, 0.0, 1.0));

        int Bsel = capacity_superpanel_B(n, NB, ctx.B);

        // LR-QR
        lrqr::Plan<Real, NB, SLAB> plan;
        plan.create(m, n, Bsel, tf32, la);
        // Call 1 resolves lazy library loading, so the measured call below runs the
        // same steady-state repeated-call path as production, without depending on
        // cross-tier process state.
        CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
        plan.geqrf(A, m);
        CUDA_CHECK(cudaDeviceSynchronize());
        // Measured rep (fresh A0 copy — warmup may have clobbered A)
        CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
        plan.geqrf(A, m);
        // relRR guard: skip at huge n where double-copy would OoM (matches run_size)
        bool chk = !getenv("LRQR_NOCHK") &&
                   mn * sizeof(double) <= (size_t(60) << 30);
        double lr_relRR = chk ? relRR_vec_T<Real>(ctx.cb, A0, A, m, n) : -1.0;
        double lr_resid = -1, lr_orth = -1;
        residual_orth_T(plan, ctx.cb, A0, A, m, n, &lr_resid, &lr_orth);
        plan.destroy();

        // cuSOLVER
        CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
        int lw = 0;
        if constexpr (std::is_same_v<Real, float>)
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
        else
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(ctx.cs, m, n, A, m, &lw));
        Real* work; CUDA_CHECK(cudaMalloc(&work, size_t(lw) * sizeof(Real)));
        Real* tau; CUDA_CHECK(cudaMalloc(&tau, size_t(n) * sizeof(Real)));
        int* info; CUDA_CHECK(cudaMalloc(&info, 4));
        if constexpr (std::is_same_v<Real, float>)
            CUSOLVER_CHECK(cusolverDnSgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
        else
            CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cs, m, n, A, m, tau, work, lw, info));
        CUDA_CHECK(cudaDeviceSynchronize());
        double cs_relRR = chk ? relRR_vec_T<Real>(ctx.cb, A0, A, m, n) : -1.0;
        double cs_resid = -1, cs_orth = -1;
        cusolver_resid_orth_T<Real>(ctx.cs, ctx.cb, A0, A, tau, m, n, &cs_resid, &cs_orth);
        cudaFree(work); cudaFree(tau); cudaFree(info);

        // V7: order-of-magnitude gate
        double rr_ratio = (cs_resid > 0) ? lr_resid / cs_resid : 0;
        double ro_ratio = (cs_orth > 0) ? lr_orth / cs_orth : 0;
        double relrr_ratio = (cs_relRR > 0) ? lr_relRR / cs_relRR : 0;
        bool pass;
        std::string fail_reasons;
        if (tgt.gate == GATE_OOM_BOTH) {
            bool rr_pass = rr_ratio <= tgt.oom;
            bool ro_pass = ro_ratio <= tgt.oom;
            pass = rr_pass && ro_pass;
            if (!rr_pass) fail_reasons += " [resid]";
            if (!ro_pass) fail_reasons += " [orth]";
        } else if (tgt.gate == GATE_OOM_RESID) {
            bool rr_pass = rr_ratio <= tgt.oom;
            pass = rr_pass;
            if (!rr_pass) fail_reasons += " [resid]";
        } else {
            bool rr_pass = lr_resid < tgt.hard_resid;
            bool ro_pass = ro_ratio <= tgt.oom;
            pass = rr_pass && ro_pass;
            if (!rr_pass) fail_reasons += " [resid]";
            if (!ro_pass) fail_reasons += " [orth]";
        }

        printf("  %7d  %10.2e  %10.2e  %7.2fx  %10.2e  %10.2e  %7.2fx  %10.2e  %10.2e  %s%s\n",
               n, lr_resid, cs_resid, rr_ratio, lr_orth, cs_orth, ro_ratio,
               lr_relRR, cs_relRR,
               pass ? "PASS" : "FAIL", fail_reasons.c_str());
        fflush(stdout);
        total_tests++;
        if (pass) total_pass++;

        cudaFree(A0); cudaFree(A);
    }

    // --- kappa-independence test (SVD-based, spec S14 guarantee) ---
    printf("\n--- kappa-independence test (SVD-based, spec S14 guarantee) ---\n");
    for (int n : {4096}) {
        int m = n;
        size_t mn = size_t(m) * n;
        Real *A0, *A, *Ubuf, *Vbuf;
        CUDA_CHECK(cudaMalloc(&A0, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&Ubuf, mn * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&Vbuf, size_t(n) * n * sizeof(Real)));

        int Bsel = capacity_superpanel_B(n, NB, ctx.B);

        lrqr::Plan<Real, NB, SLAB> plan;
        plan.create(m, n, Bsel, tf32, la);

        printf("  n=%d\n", n);
        double kr_min = 1e30, kr_max = 0;
        for (double kappa : val_kappas) {
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
            gen_svd_T<Real>(ctx.cs, ctx.cb, ctx.rg, A0, m, n, kappa, Ubuf, Vbuf);
            CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
            plan.geqrf(A, m);
            double kr = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
            kr_min = std::min(kr_min, kr);
            kr_max = std::max(kr_max, kr);
            // kappa-independence: relRR should not blow up with kappa
            // Use hard limit for TF32, parity-like bound for others (relRR < 100*n*u)
            double kr_limit = (tgt.gate == GATE_HARD_RESID_OOM_ORTH) ? tgt.hard_resid : 100.0 * n * tgt.u;
            bool kpass = kr <= kr_limit;
            printf("    kappa=1e%-9.0f  relRR=%.3e  %s\n", std::log10(kappa), kr,
                   kpass ? "PASS" : "FAIL");
            fflush(stdout);
            total_tests++;
            if (kpass) total_pass++;
        }
        plan.destroy();

        double flat_ratio = kr_max / std::max(kr_min, 1e-300);
        bool flat_pass = flat_ratio <= 10.0;
        printf("    kappa-flatness: max/min=%.2fx  %s (threshold: 10x)\n",
               flat_ratio, flat_pass ? "PASS" : "FAIL");
        fflush(stdout);
        total_tests++;
        if (flat_pass) total_pass++;

        cudaFree(A0); cudaFree(A); cudaFree(Ubuf); cudaFree(Vbuf);
    }

    // --- Tall-skinny correctness at multiple aspect ratios (panel-bound regime) ---
    run_tall_skinny_T<Real>(tf32, la, ctx, tgt, total_tests, total_pass);

    if (out_pass) *out_pass = total_pass;
    if (out_total) *out_total = total_tests;
    printf("\n=== Summary: %d/%d tests PASSED ===\n", total_pass, total_tests);
    fflush(stdout);
    return total_pass == total_tests ? 0 : 1;
}

// ---- GEMM ceiling: TF32 + fp32(FFMA) + fp64 at 8192^2 and far-update shape
static void gemm_ceiling(cublasHandle_t cb, curandGenerator_t rg) {
    int s = 8192;
    // float: TF32-TC + fp32(FFMA) at 8192^2
    float *Xf, *Yf, *Zf;
    CUDA_CHECK(cudaMalloc(&Xf, size_t(s) * s * 4));
    CUDA_CHECK(cudaMalloc(&Yf, size_t(s) * s * 4));
    CUDA_CHECK(cudaMalloc(&Zf, size_t(s) * s * 4));
    CURAND_CHECK(curandGenerateUniform(rg, Xf, size_t(s) * s));
    CURAND_CHECK(curandGenerateUniform(rg, Yf, size_t(s) * s));
    float onef = 1, zerof = 0;
    for (int mode = 0; mode < 2; ++mode) {
        auto cmp = mode ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        for (int w = 0; w < 2; ++w)
            CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, s, s, s, &onef, Xf, CUDA_R_32F, s,
                        Yf, CUDA_R_32F, s, &zerof, Zf, CUDA_R_32F, s, cmp, CUBLAS_GEMM_DEFAULT));
        Timer t; t.start();
        CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, s, s, s, &onef, Xf, CUDA_R_32F, s,
                    Yf, CUDA_R_32F, s, &zerof, Zf, CUDA_R_32F, s, cmp, CUBLAS_GEMM_DEFAULT));
        float ms = t.stop();
        printf("GEMM ceiling %-14s: %.1f TFLOP/s\n", mode ? "fp32(TF32-TC)" : "fp32(FFMA)",
               2.0 * s * double(s) * s / (ms * 1e-3) / 1e12);
    }
    // far-update shapes (w=2048, k=m=4096): TN at both tiers
    for (int mode = 0; mode < 2; ++mode) {
        auto cmp = mode ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        int mm = 2048, nn = 2048, kk = 4096;
        for (int w = 0; w < 3; ++w) {
            if (w == 2) { Timer t; t.start();
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, mm, nn, kk, &onef,
                    Xf, CUDA_R_32F, kk, Yf, CUDA_R_32F, kk, &zerof, Zf, CUDA_R_32F, mm,
                    cmp, CUBLAS_GEMM_DEFAULT));
                float ms = t.stop();
                printf("far-shape TN 2048x2048x4096 %-6s: %.1f TFLOP/s\n",
                       mode ? "tf32" : "fp32", 2.0 * mm * double(nn) * kk / (ms * 1e-3) / 1e12);
            } else
                CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, mm, nn, kk, &onef,
                    Xf, CUDA_R_32F, kk, Yf, CUDA_R_32F, kk, &zerof, Zf, CUDA_R_32F, mm,
                    cmp, CUBLAS_GEMM_DEFAULT));
        }
    }
    cudaFree(Xf); cudaFree(Yf); cudaFree(Zf);

    // fp64 GEMM ceiling (DEFAULT_MATH + cublasDgemm = FP64 tensor cores; PEDANTIC is 22% slower)
    double *Xd, *Yd, *Zd;
    CUDA_CHECK(cudaMalloc(&Xd, size_t(s) * s * 8));
    CUDA_CHECK(cudaMalloc(&Yd, size_t(s) * s * 8));
    CUDA_CHECK(cudaMalloc(&Zd, size_t(s) * s * 8));
    CURAND_CHECK(curandGenerateNormalDouble(rg, Xd, size_t(s) * s, 0.0, 1.0));
    CURAND_CHECK(curandGenerateNormalDouble(rg, Yd, size_t(s) * s, 0.0, 1.0));
    double oned = 1.0, zerod = 0.0;
    cublasMath_t oldMath;
    CUBLAS_CHECK(cublasGetMathMode(cb, &oldMath));
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
    for (int w = 0; w < 2; ++w)
        CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, s, s, s, &oned, Xd, s, Yd, s, &zerod, Zd, s));
    {   Timer t; t.start();
        CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_N, s, s, s, &oned, Xd, s, Yd, s, &zerod, Zd, s));
        float ms = t.stop();
        printf("GEMM ceiling fp64 %-14s: %.1f TFLOP/s\n", "DEFAULT",
               2.0 * s * double(s) * s / (ms * 1e-3) / 1e12);
    }
    CUBLAS_CHECK(cublasSetMathMode(cb, oldMath));
    cudaFree(Xd); cudaFree(Yd); cudaFree(Zd);
}

// ---- W2''' stage-localization probe: run geqrf at (m,n) and dump the WY
// representation components (R, Y, T16, Tpan, scalars, A0) to binary files for
// offline fp32-vs-fp64 component-swap residual analysis. Observability-only mode
// (env-gated by --stage-probe); does NOT alter the factorization path.
//
// Residual identity used offline: Q = I - Y T Y^T  ⇒
//   ||A0 - Q R|| = ||A0 - R + Y T (Y^T R)||
// R is n×n (top n rows of Afac upper triangle); Y is m×n (SPV); T is n×n (Tpan[0]).
// Swapping each component between fp32/fp64 localizes which stage carries the
// 16384:1 resid defect (R=minipanel-QR+far-tail, Y=minipanel-QR+writeout,
// T=T-build+tree-merge-ladder; T16 vs Tpan separates T-build from the ladder).
template<typename Real, int NB, int SLAB>
static int stage_probe_T(int m, int n, bool tf32, bool la, const BenchCtx& ctx,
                          const char* dumpdir) {
    const char* tier = std::is_same_v<Real, double> ? "fp64" : (tf32 ? "tf32" : "fp32");
    const char* ext  = std::is_same_v<Real, double> ? "f64" : "f32";
    size_t mn = size_t(m) * n;
    printf("\n=== STAGE-PROBE: %s  %dx%d (aspect %d:1, NB=%d) ===\n",
           tier, m, n, m / std::max(n, 1), NB);

    Real *A0, *A;
    CUDA_CHECK(cudaMalloc(&A0, mn * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx.rg, 42));
    if constexpr (std::is_same_v<Real, float>)
        CURAND_CHECK(curandGenerateNormal(ctx.rg, A0, mn, 0.0f, 1.0f));
    else
        CURAND_CHECK(curandGenerateNormalDouble(ctx.rg, A0, mn, 0.0, 1.0));

    int Bsel = capacity_superpanel_B(n, NB, ctx.B);
    // W4inkernel(v12) G6c: 3×TF32 Bsel<=512 cap removed (B-sweep slurm-7711: B=1024 faster,
    // same accuracy). See G6C_CLOSURE.md §Bsel.

    lrqr::Plan<Real, NB, SLAB> plan;
    plan.create(m, n, Bsel, tf32, la);
    CUDA_CHECK(cudaMemcpy(A, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
    plan.geqrf(A, m);

    // sanity: compute LR resid/orth (matches run_tall_skinny_T)
    double lr_resid = -1, lr_orth = -1;
    residual_orth_T(plan, ctx.cb, A0, A, m, n, &lr_resid, &lr_orth);
    double lr_relRR = relRR_vec_T<Real>(ctx.cb, A0, A, m, n);
    printf("  LR  resid=%.4e  orth=%.4e  relRR=%.4e\n", lr_resid, lr_orth, lr_relRR);

    // cuSOLVER baseline
    Real *Acs;
    CUDA_CHECK(cudaMalloc(&Acs, mn * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(Acs, A0, mn * sizeof(Real), cudaMemcpyDeviceToDevice));
    int lw = 0;
    if constexpr (std::is_same_v<Real, float>)
        CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(ctx.cs, m, n, Acs, m, &lw));
    else
        CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(ctx.cs, m, n, Acs, m, &lw));
    Real *cwork; CUDA_CHECK(cudaMalloc(&cwork, size_t(lw) * sizeof(Real)));
    Real *ctau;  CUDA_CHECK(cudaMalloc(&ctau, size_t(n) * sizeof(Real)));
    int *cinfo;  CUDA_CHECK(cudaMalloc(&cinfo, 4));
    if constexpr (std::is_same_v<Real, float>)
        CUSOLVER_CHECK(cusolverDnSgeqrf(ctx.cs, m, n, Acs, m, ctau, cwork, lw, cinfo));
    else
        CUSOLVER_CHECK(cusolverDnDgeqrf(ctx.cs, m, n, Acs, m, ctau, cwork, lw, cinfo));
    CUDA_CHECK(cudaDeviceSynchronize());
    double cs_resid = -1, cs_orth = -1;
    cusolver_resid_orth_T<Real>(ctx.cs, ctx.cb, A0, Acs, ctau, m, n, &cs_resid, &cs_orth);
    double rr_ratio = (cs_resid > 0) ? lr_resid / cs_resid : 0;
    double ro_ratio = (cs_orth > 0) ? lr_orth / cs_orth : 0;
    printf("  CS  resid=%.4e  orth=%.4e\n", cs_resid, cs_orth);
    printf("  r_ratio=%.2fx  o_ratio=%.2fx  (cuSOLVER is the %.1e bar)\n",
           rr_ratio, ro_ratio, cs_resid);

    // ---- dump components to dumpdir/<tier>.<name>.<ext>.bin ----
    auto dump = [&](const Real* d, size_t elems, const char* name) {
        std::string p = std::string(dumpdir) + "/" + tier + "." + name + "." + ext + ".bin";
        FILE* f = fopen(p.c_str(), "wb");
        if (!f) { fprintf(stderr, "dump open fail: %s\n", p.c_str()); return; }
        std::vector<Real> h(elems);
        CUDA_CHECK(cudaMemcpy(h.data(), d, elems * sizeof(Real), cudaMemcpyDeviceToHost));
        fwrite(h.data(), sizeof(Real), elems, f);
        fclose(f);
        printf("  dump %s : %zu elems -> %s\n", name, elems, p.c_str());
    };

    dump(A0, mn, "A0");

    // R: extract upper triangle of A (n×n, including diagonal=β). Rows ≥ n are 0 in upper.
    {
        std::vector<Real> Rfull(mn, Real(0));
        CUDA_CHECK(cudaMemcpy(Rfull.data(), A, mn * sizeof(Real), cudaMemcpyDeviceToHost));
        // zero below diagonal (keep upper incl diag); zero rows >= n already
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < m; ++i)
                Rfull[size_t(j) * m + i] = Real(0);
        std::string p = std::string(dumpdir) + "/" + tier + ".R." + ext + ".bin";
        FILE* f = fopen(p.c_str(), "wb");
        fwrite(Rfull.data(), sizeof(Real), mn, f);   // m×n with R in top-n upper, 0 else
        fclose(f);
        printf("  dump R : %zux%d (m×n, upper-tri) -> %s\n", (size_t)m, n, p.c_str());
    }

    // Y: SPVx[0] is m×B (= m×n since B=n for this cell). Layout: col-major, ld=m.
    dump(plan.SPVx[0], mn, "Y");

    // Tpan: npanels × NB × NB. For n=NB=64, npanels=1 → 64×64.
    dump(plan.Tpan, size_t(plan.npanels) * NB * NB, "Tpan");

    // T16: domT16 = 2 × 16×16 × (NB/16) elements. First (NB/16) blocks are the live
    // diagonal T̃16 per minipanel (mp=0..NB/16-1 at offset mp*16*16).
    dump(plan.domT16, size_t(2) * 16 * 16 * (NB / 16), "T16");

    // scalars: domScal[0..NB-1]=tauT, [NB..2NB-1]=fv, [2NB..3NB-1]=bet, [3NB..4NB-1]=sig(STALE)
    // W2 moved sig→sigD (fp64 shadow); domScal[3NB..] is never written. Dump domSigD too.
    dump(plan.domScal, size_t(4) * NB, "scal");
    // fp64 shadow dumps (sigD, dslotD) — always double regardless of Real
    auto dumpd = [&](const double* d, size_t elems, const char* name) {
        std::string p = std::string(dumpdir) + "/" + tier + "." + name + ".f64.bin";
        FILE* f = fopen(p.c_str(), "wb");
        if (!f) { fprintf(stderr, "dump open fail: %s\n", p.c_str()); return; }
        std::vector<double> h(elems);
        CUDA_CHECK(cudaMemcpy(h.data(), d, elems * sizeof(double), cudaMemcpyDeviceToHost));
        fwrite(h.data(), sizeof(double), elems, f);
        fclose(f);
        printf("  dump %s : %zu elems (fp64) -> %s\n", name, elems, p.c_str());
    };
    dumpd(plan.domSigD, size_t(NB), "sigD");       // fp64 σ (the LIVE sigma)
    dumpd(plan.domDslotD, size_t(16) * 16, "dslotD");  // fp64 dots (live)
    // Psum: the B2a output (cross-slab reduction of Pp). Layout: npanels × NB×PW.
    // THIS is the prime suspect — fp32 accumulation over nslab slabs.
    dump(plan.domPsum, size_t(plan.npanels > 0 ? plan.npanels : 1) * NB * 16, "Psum");
    // Pp: the per-slab boundary partials (B2a input). Layout: nslab × NB×PW.
    dump(plan.domPp, size_t((m / 128 + 132)) * NB * 16, "Pp");

    // cuSOLVER R (for reference): upper tri of Acs
    {
        std::vector<Real> Rfull(mn, Real(0));
        CUDA_CHECK(cudaMemcpy(Rfull.data(), Acs, mn * sizeof(Real), cudaMemcpyDeviceToHost));
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < m; ++i)
                Rfull[size_t(j) * m + i] = Real(0);
        std::string p = std::string(dumpdir) + "/" + tier + ".Rcs." + ext + ".bin";
        FILE* f = fopen(p.c_str(), "wb");
        fwrite(Rfull.data(), sizeof(Real), mn, f);
        fclose(f);
        printf("  dump Rcs (cuSOLVER R) -> %s\n", p.c_str());
    }

    plan.destroy();
    cudaFree(A0); cudaFree(A); cudaFree(Acs); cudaFree(cwork); cudaFree(ctau); cudaFree(cinfo);
    printf("  STAGE-PROBE %s done\n", tier);
    return 0;
}

// Optional single-copy validation driver for large fp64 cases. It regenerates
// input tiles and reports residual and orthogonality without storing a second matrix.
template<typename Real, int NB, int SLAB>
static int run_single_copy_size(int m, int n, bool la, const BenchCtx& ctx) {
    size_t mn = size_t(m) * n;
    Real* A; CUDA_CHECK(cudaMalloc(&A, mn * sizeof(Real)));
    // Generate A via the position-independent hash (so A0 tiles regenerate bit-exactly).
    {
        size_t total = mn;
        unsigned grid = (unsigned)((total + 255) / 256);
        k_hash_normal_T<Real><<<grid, 256>>>(A, m, m, n, 0, 0, m, n);
    }
    // W4inkernel(v12) G6c: 3×TF32 Bsel<=512 cap removed — uniform with other tiers
    // (B-sweep slurm-7711: B=1024 is 12.6% faster @65536, same fp32-OoM accuracy).
    // See G6C_CLOSURE.md §Bsel.
    int Bsel = capacity_superpanel_B(n, NB, ctx.B);
    lrqr::Plan<Real, NB, SLAB> plan;
    // G7-WRAP: validation_copy=false → no Ytx[3] (3·B·m) + no per-panel saves
    // (Vm_pan/TmM_pan/domTpanL_pan). orgqr_rows and orgqr (c_L=1) only need
    // Wax/Wbx/Pbuf/Tpan (all unconditionally allocated). This removes ~5 GiB from
    // the plan at fp64@131072² (9.3→4.2 GiB), restoring the 132.2 GiB footprint
    // that clears the G7(b) OOM line. GEMM3 takes the NN fallback (bit-identical).
    plan.create(m, n, Bsel, /*tf32=*/false, la, lrqr::lrqr_opts_t{false});

    std::vector<double> lr_ms_vec;
    constexpr int warmupReps = 2;
    for (int r = 0; r < ctx.reps + warmupReps; ++r) {
        // Regenerate A from the hash each rep (data-independent timing; replaces the
        // A0->A D2D copy in run_size). Both warmup calls are discarded.
        size_t total = mn;
        unsigned grid = (unsigned)((total + 255) / 256);
        k_hash_normal_T<Real><<<grid, 256>>>(A, m, m, n, 0, 0, m, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer t; t.start();
        plan.geqrf(A, m);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = t.stop();
        if (r >= warmupReps) lr_ms_vec.push_back((double)ms);
    }
    std::sort(lr_ms_vec.begin(), lr_ms_vec.end());
    double tlr = lr_ms_vec.empty() ? 1e30
        : (lr_ms_vec.size() % 2 ? lr_ms_vec[lr_ms_vec.size()/2]
                                : 0.5*(lr_ms_vec[lr_ms_vec.size()/2 - 1] + lr_ms_vec[lr_ms_vec.size()/2]));
    double tf = (tlr > 1e29) ? 0.0 : flops_qr(m, n) / (tlr * 1e-3) / 1e12;

    // Single-copy residual (no full A0, no full Q) + tiled orth (G7-WRAP §2.4).
    double resid = -1, orth = -1;
    residual_single_copy_T<lrqr::Plan<Real, NB, SLAB>, Real>(plan, ctx.cb, A, m, n, &resid, &orth, nullptr);

    const char* tier = std::is_same_v<Real, double> ? "fp64" : "fp32";
    // Print resid immediately — orth (tiled Gramian) can take >1 hr at fp64@131072²
    // (22 passes over Q); flushing ensures resid is not lost if the orth times out.
    printf("  [single-copy-val] tier=%s n=%d m=%d B=%d  LR-QR=%.1f TF  resid=%.3e  orth=computing...",
           tier, n, m, Bsel, tf, resid);
    fflush(stdout);
    orth = orth_single_copy_T<lrqr::Plan<Real, NB, SLAB>, Real>(plan, ctx.cb, A, m, n);
    printf("  orth=%.3e  rc=0\n", orth);
    fflush(stdout);

    // Correctness self-check (small n only, where full A0+Q fit): regenerate the
    // full A0 via the hash, form the full Q via plan.orgqr, compute the full-Q
    // residual, and compare to the row-tiled single-copy residual. This validates
    // orgqr_rows (forward panel order + T^T) against the proven orgqr. The match
    // must be close (both compute ||A0-QR|| on the same A; tiny differences come
    // from GEMM round-off order). At large n the self-check is skipped (full Q OOMs).
    const size_t full_Q_bytes = mn * sizeof(Real);
    if (full_Q_bytes <= (size_t(8) << 30)) {   // <= 8 GiB -> full A0+Q fit alongside Plan
        Real* A0full; CUDA_CHECK(cudaMalloc(&A0full, mn * sizeof(Real)));
        size_t total = mn;
        unsigned grid = (unsigned)((total + 255) / 256);
        k_hash_normal_T<Real><<<grid, 256>>>(A0full, m, m, n, 0, 0, m, n);
        Real* Qfull; CUDA_CHECK(cudaMalloc(&Qfull, mn * sizeof(Real)));
        cublasMath_t oldM0 = CUBLAS_DEFAULT_MATH;
        if (plan.cb0) { cublasGetMathMode(plan.cb0, &oldM0); cublasSetMathMode(plan.cb0, CUBLAS_DEFAULT_MATH); }
        plan.orgqr(A, m, Qfull, m);
        if (plan.cb0) cublasSetMathMode(plan.cb0, oldM0);
        double resid_full = residual_tiled_T<Real>(ctx.cb, A0full, Qfull, A, m, n);
        double orth_full = orth_tiled_T<Real>(ctx.cb, Qfull, m, n);
        printf("  [single-copy-val SELF-CHECK] full-Q resid=%.3e  row-tiled resid=%.3e  ratio=%.4f  %s\n",
               resid_full, resid, resid > 0 ? resid_full / resid : 0.0,
               (resid > 0 && resid_full > 0 && std::abs(resid_full - resid) / std::max(resid, 1e-300) < 0.5)
                   ? "MATCH" : "MISMATCH");
        printf("  [single-copy-val SELF-CHECK] full-Q orth=%.3e  tiled orth=%.3e  ratio=%.4f  %s\n",
               orth_full, orth, orth > 0 ? orth_full / orth : 0.0,
               (orth > 0 && orth_full > 0 && std::abs(orth_full - orth) / std::max(orth, 1e-300) < 0.5)
                   ? "MATCH" : "MISMATCH");
        cudaFree(A0full); cudaFree(Qfull);
    }
    cudaFree(A);
    return 0;
}

int main(int argc, char** argv) {
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    int B = 0;
    bool tf32 = true, la = true, doOrth = false, kappaCol = false, kappaSvd = false;
    bool fp64 = false, threeXtf32 = false, doCeil = true;
    bool validate = false, validateAll = false, validateFresh = false;
    bool validateLapack = false;   // --validate-lapack: LAPACK's own QR criteria
    bool bwProbe = false;          // --bw-probe: achieved beta_lambda in words/s
    bool stageProbe = false;
    int spM = 0, spN = 0;
    const char* spDir = "/tmp/opencode/w2resid-v11/stage_probe";
    bool singleCopyVal = false;
    std::vector<std::pair<int,int>> sizes;
    int reps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--B") B = atoi(argv[++i]);
        else if (a == "--tf32") { tf32 = true; threeXtf32 = false; fp64 = false; }
        else if (a == "--no-tf32") { tf32 = false; threeXtf32 = false; fp64 = false; }
        else if (a == "--fp32") { tf32 = false; threeXtf32 = false; fp64 = false; }
        else if (a == "--3xtf32") { threeXtf32 = true; tf32 = true; fp64 = false; }
        else if (a == "--fp64") { fp64 = true; tf32 = false; threeXtf32 = false; }
        else if (a == "--no-la") la = false;
        else if (a == "--orth") doOrth = true;
        else if (a == "--kappa") kappaCol = true;
        else if (a == "--kappa-svd") kappaSvd = true;
        else if (a == "--gemm-ceiling") doCeil = true;
        else if (a == "--no-ceil") doCeil = false;
        else if (a == "--validate") validate = true;
        else if (a == "--validate-all") { validateAll = true; validate = false; }
        else if (a == "--validate-lapack") { validateLapack = true; validate = false; }
        else if (a == "--bw-probe") { bwProbe = true; }
        else if (a == "--validate-fresh") { validateFresh = true; validate = false; }
        else if (a == "--single-copy-val") singleCopyVal = true;
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--mn") { int m = atoi(argv[++i]); int n = atoi(argv[++i]); sizes.push_back({m, n}); }
        else if (a == "--n") { int n = atoi(argv[++i]); sizes.push_back({n, n}); }
        else if (a == "--stage-probe") {
            stageProbe = true;
            spM = atoi(argv[++i]); spN = atoi(argv[++i]);
            if (i + 1 < argc && argv[i+1][0] != '-') spDir = argv[++i];
        }
    }
    if (B == 0 && getenv("LRQR_SPANEL_B")) B = atoi(getenv("LRQR_SPANEL_B"));
    if (B == 0 && getenv("LRQR_SUPER")) B = atoi(getenv("LRQR_SUPER"));  // F8 alias
    if (sizes.empty()) sizes = {{4096,4096}, {8192,8192}, {16384,16384}};
    if (getenv("LRQR_SKIP_DIAG")) doCeil = false;

    // 3xTF32: LRQR_3XTF32 activates the Ozaki-split far-update path in lrqr.cuh
    // (3 TF32-TC GEMMs per far GEMM for fp32-accurate results). --3xtf32 sets this env
    // var. W4inkernel(v12) G6c: B is now UNIFORM across tiers (the 3×TF32 Bsel<=512 cap
    // was removed — B-sweep slurm-7711 measured B=1024 is 12.6% faster @65536 with
    // identical fp32-OoM accuracy; B=128 was the historically-slow point, NOT B=1024).
    // See G6C_CLOSURE.md §Bsel.
    if (threeXtf32) setenv("LRQR_3XTF32", "1", 1);
    else unsetenv("LRQR_3XTF32");

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    char gpu_pci[32] = {0};
    cudaDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), 0);
    int gclock = 0; cudaDeviceGetAttribute(&gclock, cudaDevAttrClockRate, 0);
    printf("GPU name=%s pci=%s sm=%d%d clockRate=%dKHz  (record UUID+boost via nvidia-smi into bench.json)\n",
           prop.name, gpu_pci, prop.major, prop.minor, gclock);
    const char* tierName = fp64 ? "fp64" : (threeXtf32 ? "3xTF32" : (tf32 ? "TF32" : "fp32"));
    int tf32far = fp64 ? 0 : (tf32 ? 1 : 0);

    cublasHandle_t cb; CUBLAS_CHECK(cublasCreate(&cb));
    cusolverDnHandle_t cs; CUSOLVER_CHECK(cusolverDnCreate(&cs));
    curandGenerator_t rg; CURAND_CHECK(curandCreateGenerator(&rg, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    // Every mode below returns directly from main. Keep the vendor-library
    // handles scoped so validation, benchmark, and diagnostic exits all release
    // their internal device allocations before the CUDA context is torn down.
    struct LibraryHandleGuard {
        cublasHandle_t cb;
        cusolverDnHandle_t cs;
        curandGenerator_t rg;
        ~LibraryHandleGuard() {
            curandDestroyGenerator(rg);
            cusolverDnDestroy(cs);
            cublasDestroy(cb);
        }
    } library_handles{cb, cs, rg};

    // ---- validate mode: comprehensive correctness suite with PASS/FAIL
    // V7/V8 order-of-magnitude gates:
    //   TF32:   resid < 1e-3 (order 1e-4) AND orth <= 10x cuSOLVER Sgeqrf
    //   fp32:   resid & orth <= 10x cuSOLVER Sgeqrf (same OoM, ~1e-6)
    //   3xTF32: resid & orth <= 10x cuSOLVER Sgeqrf (fp32-OoM at every n)
    //           [V8: was GATE_OOM_RESID — orth was ungated; AUDIT_V6 §3 measured
    //           7.78x @65536, inside 10x but ungated. Now GATE_OOM_BOTH.]
    //   fp64:   resid & orth <= 10x cuSOLVER Dgeqrf (same OoM, ~1e-15..1e-14)
    // kappa-flatness < 10x is a merge gate on every tier.
    // P0.c: tiled resid/orth computed at every n (no n<=8192 limit).
    // V8 §3: validate-all extended to largest fitting sizes (65536/98304 float,
    // 65536 fp64) — the 2.49e-4@65536 regression shipped invisible at n<=32768.
    // --validate-lapack: LAPACK's dqrt01 criteria over the dlatb4 'QR' matrix
    // types, on all four tiers. Absolute (THRESH=30), never a cuSOLVER ratio, and
    // deliberately NOT folded into --validate-all so the published 71/71 keeps
    // meaning exactly what it meant.
    if (bwProbe) {
        // beta_lambda for both word sizes: fp64 and fp32 move different byte counts
        // per word, and the model consumes words/s, so both are reported.
        int rc = bandwidth_probe<double>("fp64");
        rc |= bandwidth_probe<float>("fp32");
        printf("DONE\n");
        return rc;
    }

    if (validateLapack) {
        BenchCtx ctx{B, 1, true, false, true, cb, cs, rg, false, false, false};
        int rc = 0, gp = 0, gt = 0;
        printf("\n########## VALIDATE-LAPACK: dlatb4 'QR' types x dqrt01 criteria ##########\n");
        unsetenv("LRQR_3XTF32");
        printf("\n########## TIER 1/4: TF32 ##########\n");
        { ValTargets tgt{"TF32", 4.88e-4, GATE_HARD_RESID_OOM_ORTH, 10.0, 1e-3};
          int p=0,t=0; rc |= run_validate_lapack_T<float,128,128>(true, la, ctx, tgt, &p, &t);
          gp+=p; gt+=t; printf("[validate-lapack] TF32: %d/%d PASS\n", p, t); }
        printf("\n########## TIER 2/4: fp32 ##########\n");
        { ValTargets tgt{"fp32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
          int p=0,t=0; rc |= run_validate_lapack_T<float,128,128>(false, la, ctx, tgt, &p, &t);
          gp+=p; gt+=t; printf("[validate-lapack] fp32: %d/%d PASS\n", p, t); }
        printf("\n########## TIER 3/4: 3xTF32 ##########\n");
        { setenv("LRQR_3XTF32", "1", 1);
          ValTargets tgt{"3xTF32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
          int p=0,t=0; rc |= run_validate_lapack_T<float,128,128>(true, la, ctx, tgt, &p, &t);
          gp+=p; gt+=t; printf("[validate-lapack] 3xTF32: %d/%d PASS\n", p, t);
          unsetenv("LRQR_3XTF32"); }
        printf("\n########## TIER 4/4: fp64 ##########\n");
        { cublasMath_t om; CUBLAS_CHECK(cublasGetMathMode(cb, &om));
          CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH));
          ValTargets tgt{"fp64", 1.11e-16, GATE_OOM_BOTH, 10.0, -1};
          int p=0,t=0; rc |= run_validate_lapack_T<double,64,128>(false, la, ctx, tgt, &p, &t);
          gp+=p; gt+=t; printf("[validate-lapack] fp64: %d/%d PASS\n", p, t);
          CUBLAS_CHECK(cublasSetMathMode(cb, om)); }
        printf("\n=== VALIDATE-LAPACK RESULT ===\n");
        printf("AGGREGATE: %d/%d tests PASSED; exit=%d\n", gp, gt, rc);
        printf("DONE\n");
        return rc;
    }

    if (validateAll) {
        BenchCtx ctx{B, 1, true, false, true, cb, cs, rg, false, false, false};
        int rc = 0;
        int gp = 0, gt = 0;
        printf("\n########## VALIDATE-ALL: 4 precision tiers (V7 OoM gates) ##########\n");

        unsetenv("LRQR_3XTF32");

        // Tier 1: TF32
        printf("\n########## TIER 1/4: TF32 ##########\n");
        { ValTargets tgt{"TF32", 4.88e-4, GATE_HARD_RESID_OOM_ORTH, 10.0, 1e-3};
          int p=0, t=0; rc |= run_validate_T<float, 128, 128>(true, la, ctx, tgt, &p, &t);
          gp += p; gt += t; printf("[validate-all] TF32: %d/%d PASS\n", p, t); }

        // Tier 2: fp32
        printf("\n########## TIER 2/4: fp32 ##########\n");
        { ValTargets tgt{"fp32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
          int p=0, t=0; rc |= run_validate_T<float, 128, 128>(false, la, ctx, tgt, &p, &t);
          gp += p; gt += t; printf("[validate-all] fp32: %d/%d PASS\n", p, t); }

        // Tier 3: 3xTF32 (env var activates Ozaki-split far-update path)
        printf("\n########## TIER 3/4: 3xTF32 ##########\n");
        { setenv("LRQR_3XTF32", "1", 1);
          // AUDIT_V6 §3: 3xTF32 gated resid only (GATE_OOM_RESID) — orth was
          // ungated for exactly the tier that showed 7.78x cuSOLVER @65536.
          // Gate both resid AND orth at <=10x cuSOLVER (same as fp32/fp64).
          ValTargets tgt{"3xTF32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
          int p=0, t=0; rc |= run_validate_T<float, 128, 128>(true, la, ctx, tgt, &p, &t);
          gp += p; gt += t; printf("[validate-all] 3xTF32: %d/%d PASS\n", p, t);
          unsetenv("LRQR_3XTF32"); }

        // Tier 4: fp64 (PEDANTIC math)
        printf("\n########## TIER 4/4: fp64 ##########\n");
        { cublasMath_t om; CUBLAS_CHECK(cublasGetMathMode(cb, &om));
          CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH));
          ValTargets tgt{"fp64", 1.11e-16, GATE_OOM_BOTH, 10.0, -1};
          int p=0, t=0; rc |= run_validate_T<double, 64, 128>(false, la, ctx, tgt, &p, &t);
          gp += p; gt += t; printf("[validate-all] fp64: %d/%d PASS\n", p, t);
          CUBLAS_CHECK(cublasSetMathMode(cb, om)); }

        printf("\n=== VALIDATE-ALL RESULT ===\n");
        printf("AGGREGATE: %d/%d tests PASSED; exit=%d\n", gp, gt, rc);
        printf("DONE\n");
        return rc;
    }

    // ---- validate-fresh: one process per tier (eliminates cross-tier state leaks)
    // AUDIT_V6 §3: --validate-all runs all 4 tiers in ONE process with per-tier
    // setenv/unsetenv of LRQR_3XTF32. While the setenv/unsetenv pattern is correct
    // (plan.create re-reads the env var), a fresh-process-per-tier check is the
    // structural guard against state-leak regressions. Each tier runs as
    // `$BIN --validate --<tier>` in a subprocess; the subprocess exit code is
    // 0 (all pass) or 1 (any fail).
    if (validateFresh) {
        char exe[4096];
        ssize_t len = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        if (len < 0) { fprintf(stderr, "readlink /proc/self/exe failed\n"); return 1; }
        exe[len] = '\0';
        int rc = 0, gp = 0, gt = 0;
        printf("\n########## VALIDATE-FRESH: 4 tiers, fresh process each ##########\n");
        struct { const char* flag; const char* name; } tiers[] = {
            {"--tf32",   "TF32"},
            {"--fp32",   "fp32"},
            {"--3xtf32", "3xTF32"},
            {"--fp64",   "fp64"},
        };
        for (auto& t : tiers) {
            printf("\n########## FRESH TIER: %s ##########\n", t.name);
            fflush(stdout);
            std::string cmd = std::string(exe) + " --validate " + t.flag;
            int ret = system(cmd.c_str());
            int tier_rc = WIFEXITED(ret) ? WEXITSTATUS(ret) : 1;
            // --validate returns 0 (pass) or 1 (fail); we can't get per-test
            // counts from the exit code, so just track pass/fail per tier.
            int t_tests = 1, t_pass = (tier_rc == 0) ? 1 : 0;
            gp += t_pass; gt += t_tests;
            rc |= (tier_rc != 0);
            printf("[validate-fresh] %s: %s\n", t.name, t_pass ? "PASS" : "FAIL");
            fflush(stdout);
        }
        printf("\n=== VALIDATE-FRESH RESULT ===\n");
        printf("AGGREGATE: %d/%d tiers PASSED; exit=%d\n", gp, gt, rc);
        printf("DONE\n");
        return rc;
    }

    if (stageProbe) {
        // W2''' stage-localization: dump WY components for offline fp32/fp64 swap.
        // Run one tier per process (fp32 OR fp64 OR 3xTF32); caller runs both and
        // feeds dumps to stage_swap. NB dispatch matches run_tall_skinny_T.
        BenchCtx ctx{B, 1, true, false, true, cb, cs, rg, false, false, false};
        if (spN <= 0) spN = 64;
        if (spM <= 0) spM = 1048576;
        printf("STAGE-PROBE dir=%s  m=%d n=%d\n", spDir, spM, spN);
        fflush(stdout);
        if (fp64) {
            CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_PEDANTIC_MATH));
            if (spN % 64 == 0)
                return stage_probe_T<double, 64, 128>(spM, spN, false, la, ctx, spDir);
            fprintf(stderr, "stage-probe fp64: n=%d not %%64 — abort\n", spN);
            return 1;
        } else if (threeXtf32) {
            setenv("LRQR_3XTF32", "1", 1);
            if (spN % 128 == 0)
                return stage_probe_T<float, 128, 128>(spM, spN, true, la, ctx, spDir);
            else if (spN % 64 == 0)
                return stage_probe_T<float, 64, 128>(spM, spN, true, la, ctx, spDir);
            fprintf(stderr, "stage-probe 3xtf32: n=%d not %%64 — abort\n", spN);
            return 1;
        } else {
            if (spN % 128 == 0)
                return stage_probe_T<float, 128, 128>(spM, spN, tf32, la, ctx, spDir);
            else if (spN % 64 == 0)
                return stage_probe_T<float, 64, 128>(spM, spN, tf32, la, ctx, spDir);
            fprintf(stderr, "stage-probe: n=%d not %%64 — abort\n", spN);
            return 1;
        }
    }

    if (validate) {
        if (fp64) CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
        BenchCtx ctx{B, 1, true, false, true, cb, cs, rg, false, false, false};

        ValTargets tgt;
        if (fp64) {
            tgt = {"fp64", 1.11e-16, GATE_OOM_BOTH, 10.0, -1};
            return run_validate_T<double, 64, 128>(false, la, ctx, tgt);
        } else if (threeXtf32) {
            tgt = {"3xTF32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
            return run_validate_T<float, 128, 128>(true, la, ctx, tgt);
        } else if (tf32) {
            tgt = {"TF32", 4.88e-4, GATE_HARD_RESID_OOM_ORTH, 10.0, 1e-3};
            return run_validate_T<float, 128, 128>(true, la, ctx, tgt);
        } else {
            tgt = {"fp32", 5.96e-8, GATE_OOM_BOTH, 10.0, -1};
            return run_validate_T<float, 128, 128>(false, la, ctx, tgt);
        }
    }

    // ---- benchmark mode
    printf("=== LR-QR Sweep ===\n");
    printf("DEVICE %s  tier=%s  B=%s  arch=sm_%d%d%s  tf32far=%d  lookahead=%d\n",
           prop.name, tierName, B ? std::to_string(B).c_str() : "auto",
           prop.major, prop.minor, (prop.major == 9 && prop.minor == 0) ? "a" : "",
           tf32far, (int)la);
    printf("\n");

    if (doCeil) gemm_ceiling(cb, rg);

    printf("  %7s  %18s  %18s  %8s  %10s  %10s  %10s  %10s  %10s  %7s  %7s\n",
           "n", "LR-QR TF(ms)", "cuSOLVER TF(ms)", "Speedup",
           "relRR", "cs_relRR", "orth", "cs_orth", "resid", "o_ratio", "r_ratio");

    // Single-algorithm modes: LRQR_ONLY times only LR-QR, LRQR_CUS_ONLY only cuSOLVER.
    const bool lonly = getenv("LRQR_ONLY"), conly = getenv("LRQR_CUS_ONLY");
    const bool single = lonly || conly;
    BenchCtx ctx{B, reps, doOrth, kappaCol, kappaSvd, cb, cs, rg, single, lonly, conly};

    // W6-P3 single-copy validation prototype (opt-in). Bypasses run_size to hold
    // 1×A (no A0) + row-tiled Q (no full Q) → fp64@98304² WITH validation fits.
    // Default validate-all / bench path is UNCHANGED when this flag is absent.
    if (singleCopyVal || getenv("LRQR_SINGLE_COPY_VAL")) {
        printf("  [single-copy-val] mode ON — 1×A + row-tiled Q (no full A0/Q)\n");
        for (auto [mm, nn] : sizes) {
            if (fp64)
                run_single_copy_size<double, 64, 128>(mm, nn, la, ctx);
            else if (nn % 128 == 0)
                run_single_copy_size<float, 128, 128>(mm, nn, la, ctx);
            else
                run_single_copy_size<float, 64, 128>(mm, nn, la, ctx);
        }
        printf("DONE\n");
        return 0;
    }

    // Superpanel width is selected once by capacity_superpanel_B for every tier,
    // shape, validation path, and benchmark path. Look-ahead overlap remains a
    // property of the live schedule rather than an aspect/size dispatch table.
    if (fp64) {
        CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_DEFAULT_MATH));
        for (auto [m, n] : sizes) {
            run_size<double, 64, 128>(m, n, /*tf32=*/false, la, ctx);
        }
    } else {
        for (auto [m, n] : sizes) {
            if (n % 128 == 0)
                run_size<float, 128, 128>(m, n, tf32, la, ctx);
            else
                run_size<float, 64, 128>(m, n, tf32, la, ctx);
        }
    }
    printf("DONE\n");
    if (g_sanity_invalid > 0) {
        printf("BENCH INVALID: %d V5-sec2 sanity-assert failure(s) — perf numbers NOT trustworthy. Re-measure.\n",
               g_sanity_invalid);
        return 2;
    }
    return 0;
}
