#pragma once
// Reference-only host orchestration. Numerical operations remain in existing
// GPU kernels or the named vendor library, outside the native QR implementation.
#include "common.hpp"
#include "reference_gpu.hpp"
#include <cusolverDn.h>

namespace tqr {
inline int reference_validation_width(int n) {
    int width = 512;
    if (const char* value = std::getenv("TQR_REFERENCE_VALIDATION_WIDTH")) {
        width = std::stoi(value);
        if (width < 512 || width > 8192 || width % 512)
            throw std::runtime_error("reference_validation_width_multiple_of_512_up_to_8192");
    }
    return std::min(std::max(n, 1), width);
}
inline void reference_solver_check(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cusolver_status_" + std::to_string(status));
}

template<class T> class ReferenceOrmqr {
    cusolverDnHandle_t handle;
    Stream& stream;
    Buffer<T> workspace;
    Buffer<int> info{1};
public:
    // Q = Q_0 Q_1 ...; Q X applies the rightmost block first, while Q^T X applies the leftmost
    // block first. These calls are validation only, after the factorization timing interval.
    static constexpr int reflector_block = 1024;
    ReferenceOrmqr(cusolverDnHandle_t h, Stream& s) : handle(h), stream(s) {}
    void apply(int m, int q, int k, const T* a, int lda, const T* tau,
               T* x, int ldx, cublasOperation_t operation) {
        if (!m || !q || !k) return;
        if (k > reflector_block) {
            auto panel = [&](int first) {
                apply(m - first, q, std::min(reflector_block, k - first),
                      a + first + size_t(first) * lda, lda, tau + first,
                      x + first, ldx, operation);
            };
            if (operation == CUBLAS_OP_N)
                for (int first = (k - 1) / reflector_block * reflector_block;
                     first >= 0; first -= reflector_block) panel(first);
            else
                for (int first = 0; first < k; first += reflector_block) panel(first);
            return;
        }
        int count = 0;
        if constexpr (sizeof(T) == 4)
            reference_solver_check(cusolverDnSormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
                operation, m, q, k, a, lda, tau, x, ldx, &count), "Sormqr_bufferSize");
        else
            reference_solver_check(cusolverDnDormqr_bufferSize(handle, CUBLAS_SIDE_LEFT,
                operation, m, q, k, a, lda, tau, x, ldx, &count), "Dormqr_bufferSize");
        if (count < 0) throw std::runtime_error("negative_ormqr_workspace");
        if (size_t(count) > workspace.n) {
            // Queries and allocation are validation costs, never QR timings.
            workspace = Buffer<T>();
            workspace.alloc(count);
        }
        if constexpr (sizeof(T) == 4)
            reference_solver_check(cusolverDnSormqr(handle, CUBLAS_SIDE_LEFT, operation,
                m, q, k, a, lda, tau, x, ldx, workspace.p, int(workspace.n), info.p), "Sormqr");
        else
            reference_solver_check(cusolverDnDormqr(handle, CUBLAS_SIDE_LEFT, operation,
                m, q, k, a, lda, tau, x, ldx, workspace.p, int(workspace.n), info.p), "Dormqr");
        stream.sync();
        if (int status = info.download()[0])
            throw std::runtime_error("ormqr_info_" + std::to_string(status));
    }
    size_t workspace_bytes() const { return workspace.n * sizeof(T); }
};

template<class T, class ReadR, class ApplyQ>
json validate_conventional_reference(int m, int n, int block, ReadR read_r, ApplyQ apply_q) {
    double started = seconds(), residual = 0;
    int ld = std::max(m, 1), width = reference_validation_width(n);
    Buffer<T> x(checked_mul(size_t(ld), size_t(width))), original(x.n);
    int columns = 0;
    for (int col = 0; col < n; col += width) {
        int q = std::min(width, n - col);
        read_r(x.p, ld, col, q);
        apply_q(x.p, ld, q, CUBLAS_OP_N);
        reference_generate(original.p, m, q, ld, m, 1024, 1, 1, 0, 0, col, sizeof(T));
        for (int first = 0; first < q; first += 512) {
            double r = reference_relative(x.p + size_t(first) * ld,
                original.p + size_t(first) * ld, m, std::min(512, q - first), ld, ld, sizeof(T));
            residual = !std::isfinite(r) ? INFINITY : std::max(residual, r);
        }
        columns += q;
    }
    int vectors = std::min(width, 16);
    reference_generate(x.p, m, vectors, ld, m, 1024, 1, 1, 0, 0, 17, sizeof(T));
    if (m) CU(cudaMemcpy(original.p, x.p, size_t(ld) * vectors * sizeof(T), cudaMemcpyDeviceToDevice));
    apply_q(x.p, ld, vectors, CUBLAS_OP_N);
    apply_q(x.p, ld, vectors, CUBLAS_OP_T);
    double inverse = reference_relative(x.p, original.p, m, vectors, ld, ld, sizeof(T));
    double u = working_unit_roundoff<T>();
    double tolerance = std::min(0.01, 64 * u * (std::sqrt(double(std::max(m, 1))) +
        std::sqrt(double(std::max(n, 1))) + block) * (1 + std::log2(std::max(1, ceildiv(m, block)))));
    return {{"performed", true}, {"pass", std::isfinite(residual) && std::isfinite(inverse) &&
                residual <= tolerance && inverse <= tolerance},
        {"input_columns_checked", columns}, {"full_residual_coverage", columns == n},
        {"residual_max_column_block", residual}, {"Q_QT_inverse_relative", inverse},
        {"inverse_vectors", vectors}, {"engineering_tolerance", tolerance},
        {"validation_block_columns", width},
        {"residual_statistic_block_columns", 512},
        {"unit_roundoff", u}, {"validation_s", seconds() - started},
        {"buffer_bytes", 2 * x.n * sizeof(T)},
        {"scope", "GPU reconstruction of every input column and a finite Q/Q-transpose inverse sketch; engineering gate, not a uniform stability proof"}};
}
} // namespace tqr
