#pragma once
// Independent extended-precision reference for GPU Householder panel tests.
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace panel_reference {
inline void require(bool condition, const std::string &message) {
    if (!condition)
        throw std::runtime_error(message);
}

// Q(rows,:) <- (I - V T V^T) Q(rows,:) in extended precision on the CPU. V and T
// come from the GPU, but this calculation shares no update code with the implementation.
template <class T>
void apply_node(std::vector<long double> &q, int ldq, int h, const std::vector<int> &rows,
                const T *v, int ldv, const T *t, int columns = -1, bool transpose = false) {
    if (columns < 0)
        columns = h;
    std::vector<long double> w(h), z(h);
    for (int j = 0; j < columns; ++j) {
        std::fill(w.begin(), w.end(), 0);
        std::fill(z.begin(), z.end(), 0);
        for (int k = 0; k < h; ++k)
            for (size_t r = 0; r < rows.size(); ++r)
                w[k] +=
                    static_cast<long double>(v[r + size_t(k) * ldv]) * q[rows[r] + size_t(j) * ldq];
        for (int i = 0; i < h; ++i)
            for (int k = transpose ? 0 : i; k < (transpose ? i + 1 : h); ++k)
                z[i] +=
                    static_cast<long double>(t[transpose ? k + size_t(i) * h : i + size_t(k) * h]) *
                    w[k];
        for (size_t r = 0; r < rows.size(); ++r)
            for (int k = 0; k < h; ++k)
                q[rows[r] + size_t(j) * ldq] -=
                    static_cast<long double>(v[r + size_t(k) * ldv]) * z[k];
    }
}

template <class T> std::vector<T> input(int rows, int h, int pattern) {
    std::vector<T> a(size_t(rows) * h);
    std::mt19937_64 random(73);
    for (int c = 0; c < h; ++c)
        for (int r = 0; r < rows; ++r) {
            const int value = int(random() % 2001) - 1000;
            T x = T(value) / T(1000);
            if (pattern == 1) // diagonal: every reflector has a zero tail
                x = r == c ? T(c % 2 ? -1 : 1) : T(0);
            if (pattern == 2) // rank deficient, with exactly zero columns
                x = c % 3 == 0 ? T(0) : T(r % 7 - 3) / T(4);
            if (pattern == 3) // nearly diagonal: exercise cancellation in norm downdates
                x = r == c ? T(1) : x * std::numeric_limits<T>::epsilon();
            if (pattern == 4) // independently scaled columns, far below/above a safe squared norm
                x = std::ldexp(x, (c % 3 - 1) * (sizeof(T) == 8 ? 500 : 60));
            a[r + size_t(c) * rows] = x;
        }
    return a;
}

template <class T>
std::pair<double, double> check_qr(const std::vector<T> &original, const std::vector<T> &factors,
                                   const std::vector<long double> &q, int rows, int h, int lda,
                                   double tolerance) {
    long double worst_residual = 0, orth2 = 0;
    for (int j = 0; j < h; ++j) {
        long double scale = 0, err2 = 0, norm2 = 0;
        for (int r = 0; r < rows; ++r)
            scale =
                std::max(scale, std::abs(static_cast<long double>(original[r + size_t(j) * rows])));
        if (scale == 0)
            scale = 1;
        for (int r = 0; r < rows; ++r) {
            long double got = 0;
            for (int k = 0; k <= j; ++k)
                got += q[r + size_t(k) * rows] * (factors[k + size_t(j) * lda] / scale);
            const long double expected = original[r + size_t(j) * rows] / scale;
            err2 += (got - expected) * (got - expected);
            norm2 += expected * expected;
        }
        worst_residual = std::max(worst_residual, std::sqrt(err2 / std::max(norm2, 1.L)));
        require(std::isfinite(err2), "nonfinite reconstructed column");
        for (int k = 0; k < h; ++k) {
            long double dot = 0;
            for (int r = 0; r < rows; ++r)
                dot += q[r + size_t(j) * rows] * q[r + size_t(k) * rows];
            dot -= j == k ? 1 : 0;
            orth2 += dot * dot;
        }
        for (int r = rows; r < lda; ++r)
            require(std::isnan(factors[r + size_t(j) * lda]), "panel padding was overwritten");
    }
    const long double orth = std::sqrt(orth2 / h);
    require(std::isfinite(orth) && worst_residual <= tolerance && orth <= tolerance,
            "residual or orthogonality exceeds the panel tolerance");
    return {double(worst_residual), double(orth)};
}

} // namespace panel_reference
