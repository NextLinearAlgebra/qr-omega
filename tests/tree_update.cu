// Check the fused update against extended-precision CPU W/Z/D, including untouched matrix rows.
#include "tree_update.cuh"
#include "panel_reference.hpp"
#include <iostream>
#include <numeric>

using namespace qr_omega;
using panel_reference::require;

namespace {
template <class T>
void check(int h, int q, int kind, bool aligned, bool resident, int layers, bool transpose) {
    TreeStep step;
    step.h = h;
    step.segments = 3;
    std::vector<std::vector<int>> xr(3), vr(3);
    int rows = 0, vrows = 0;
    auto append = [&](int s, int x0, int v0, int n) {
        for (int r = 0; r < n; ++r) {
            xr[s].push_back(x0 + r);
            vr[s].push_back(v0 + r);
        }
        rows = std::max(rows, x0 + n);
        vrows = std::max(vrows, v0 + n);
    };
    if (kind == 0) {
        step.domain = step.tile = 192;
        step.last = 137;
        for (int s = 0; s < 3; ++s)
            append(s, s * 192, s * 192, s == 2 ? 137 : 192);
    } else if (kind == 1) {
        step.domain = 768;
        step.tile = 192;
        step.phase = 2;
        step.last = 385;
        for (int s = 0; s < 3; ++s) {
            append(s, s * 768, s * 256, h);
            append(s, s * 768 + 384, s * 256 + h, s == 2 ? 1 : 192);
        }
    } else {
        step.outer = 768;
        step.inner = 192;
        step.fan_in = 4;
        step.last_children = 3;
        for (int s = 0; s < 3; ++s)
            for (int child = 0; child < (s == 2 ? 3 : 4); ++child)
                append(s, s * 768 + child * 192, (s * 4 + child) * h, h);
    }
    const int ldx = (rows + 1) / 2 * 2 + (aligned ? 0 : 1),
              ldv = (vrows + 1) / 2 * 2 + (aligned ? 0 : 1);
    auto x = panel_reference::input<T>(ldx, q, 0);
    std::vector<T> v(size_t(ldv) * h, std::numeric_limits<T>::quiet_NaN()), t(size_t(3) * h * h);
    for (int s = 0; s < 3; ++s) {
        for (int c = 0; c < h; ++c)
            for (size_t r = 0; r < vr[s].size(); ++r)
                v[vr[s][r] + size_t(c) * ldv] = T(int((r * 7 + c * 11 + s) % 31) - 15) / T(128) +
                                                T(int((r * 13 + c) % 97)) / T(99991);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i <= j; ++i)
                t[size_t(s) * h * h + i + size_t(j) * h] =
                    i == j ? T(0.75) + T(j) / T(1009) : T((i * 3 + j * 7) % 7 - 3) / T(253);
    }
    std::vector<long double> expected(x.begin(), x.end());
    for (int s = 0; s < 3; ++s)
        for (int j = 0; j < q; ++j) {
            std::vector<long double> w(h, 0), z(h, 0);
            for (int k = 0; k < h; ++k)
                for (size_t r = 0; r < xr[s].size(); ++r)
                    w[k] +=
                        (long double)v[vr[s][r] + size_t(k) * ldv] * x[xr[s][r] + size_t(j) * ldx];
            for (int i = 0; i < h; ++i)
                for (int k = 0; k < h; ++k)
                    z[i] += (long double)t[size_t(s) * h * h +
                                           (transpose ? k + size_t(i) * h : i + size_t(k) * h)] *
                            w[k];
            for (size_t r = 0; r < xr[s].size(); ++r)
                for (int k = 0; k < h; ++k)
                    expected[xr[s][r] + size_t(j) * ldx] -=
                        (long double)v[vr[s][r] + size_t(k) * ldv] * z[k];
        }
    Buffer<T> dx(x.size()), dv(v.size()), dt(t.size());
    Stream stream;
    dx.upload(x, stream);
    dv.upload(v, stream);
    dt.upload(t, stream);
    auto launch = resident ? launch_tree_update_x<T> : launch_tree_update<T>;
    const auto carrier = launch(step, dv.p, ldv, dt.p, dx.p, ldx, q, transpose, layers, stream);
    require(carrier.bounded(), "unbounded tree update");
    require(carrier.at[BlockLevel].c == std::min(layers, 2), "warp replication was not executed");
    stream.sync();
    const auto actual = dx.download();
    long double error = 0, scale = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "tree update consumed poisoned padding");
        error = std::max(error, std::abs(actual[i] - expected[i]));
        scale = std::max(scale, std::abs(expected[i]));
    }
    // Operands with full mantissas: FP32 and 3xTF32 must stay FP32-class, so that a running sum
    // left inside the TF32 tensor cores (which truncate it) fails here.
    const double tolerance = std::is_same_v<T, double>       ? 2e-13
                             : fp32_math() == Fp32Math::TF32 ? 2e-3
                                                             : 1.5e-6;
    if (error > tolerance * std::max(scale, 1.L)) {
        std::cerr << precision<T>() << " h=" << h << " q=" << q << " kind=" << kind
                  << " aligned=" << aligned << " resident=" << resident << " layers=" << layers
                  << " transpose=" << transpose << " relative error=" << double(error / scale)
                  << '\n';
        throw std::runtime_error("incorrect tree update");
    }
}
template <class T> void suite(bool quick) {
    for (bool resident : {false, true})
        for (int kind : {0, 1, 2})
            for (int h : quick ? std::vector<int>{33, 64} : std::vector<int>{1, 7, 33, 64})
                for (bool aligned : {false, true})
                    for (int layers : {1, 2})
                        for (bool transpose : {false, true})
                            check<T>(h, quick ? 35 : 67, kind, aligned, resident, layers,
                                     transpose);
}
} // namespace

int main(int argc, char **) {
    try {
        suite<double>(argc > 1);
        for (auto math : {Fp32Math::IEEE, Fp32Math::TF32, Fp32Math::X3}) {
            fp32_math() = math;
            suite<float>(argc > 1);
        }
        std::cout << "All tree updates passed extended-precision reference checks.\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
