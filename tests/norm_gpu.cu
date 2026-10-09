#include "kernels.cuh"
#include <iomanip>
using namespace qr_omega;

template <class T> void check_norms() {
    const int rows = 259, cols = 129, lda = rows + 5, ldb = rows + 7;
    std::vector<T> a(size_t(lda) * cols, T(99)), b(size_t(ldb) * cols, T(-99));
    const T large = std::ldexp(T(1), std::numeric_limits<T>::max_exponent / 2);
    const T small = std::ldexp(T(1), std::numeric_limits<T>::min_exponent / 2);
    for (int kind = 0; kind < 4; ++kind) {
        long double diff = 0, den = 0;
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const T scale = kind == 2 ? ((i + j) % 3 ? small : large) : T(1);
                const T y = kind == 0 || kind == 3 ? T(0) : scale * T((i + j) % 7 - 3);
                const T x = kind == 0 ? T(0) : y + scale * T(0.125);
                a[i + size_t(j) * lda] = x;
                b[i + size_t(j) * ldb] = y;
                diff = std::hypot(diff, static_cast<long double>(x) - y);
                den = std::hypot(den, static_cast<long double>(y));
            }
        const double expected = double(den ? diff / den : diff);
        Buffer<T> da(a.size()), db(b.size());
        da.upload(a, nullptr);
        db.upload(b, nullptr);
        for (int blocks : {1, 3, 64, 256, 512}) {
            Buffer<T> output(size_t(3) * blocks);
            relative_error<<<blocks, 256>>>(da.p, db.p, rows, cols, lda, ldb, output.p);
            CU(cudaDeviceSynchronize());
            const double got = relative_error_ratio(output.download());
            const double tolerance =
                64 * std::numeric_limits<T>::epsilon() * std::max(1.0, expected);
            if (!std::isfinite(got) || std::abs(got - expected) > tolerance)
                throw std::runtime_error(
                    "scaled norm differs from independent extended-precision norm");
        }
    }
    // Analytic large-array norm and launch measurements for the common reduction path.
    const int side = 4096;
    std::vector<T> ones(size_t(side) * side, T(1));
    Buffer<T> da(ones.size()), db(ones.size());
    db.upload(ones, nullptr);
    std::fill(ones.begin(), ones.end(), T(1.125));
    da.upload(ones, nullptr);
    for (int blocks : {1, 64, 128, 256, 512}) {
        Buffer<T> output(size_t(3) * blocks);
        relative_error<<<blocks, 256>>>(da.p, db.p, side, side, side, side, output.p);
        CU(cudaDeviceSynchronize());
        Event start, stop;
        start.record(nullptr);
        for (int rep = 0; rep < 3; ++rep)
            relative_error<<<blocks, 256>>>(da.p, db.p, side, side, side, side, output.p);
        stop.record(nullptr);
        CU(cudaEventSynchronize(stop.e));
        float milliseconds;
        CU(cudaEventElapsedTime(&milliseconds, start.e, stop.e));
        const double error = relative_error_ratio(output.download());
        if (std::abs(error - 0.125) > 64 * std::numeric_limits<T>::epsilon())
            throw std::runtime_error("large-array norm differs from analytic result");
        std::cout << precision<T>() << " blocks " << blocks << " milliseconds " << milliseconds / 3
                  << " norm " << std::setprecision(15) << error << '\n';
    }
}
int main() {
    try {
        check_norms<double>();
        check_norms<float>();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
