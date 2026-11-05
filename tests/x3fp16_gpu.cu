// Check the 3xFP16 W = V^T X and D: X -= V Z against extended-precision CPU products, beside the
// 3xTF32 kernels on the same operands, and the FP16 range check of the prescaled reflectors.
//
// Operands mimic the update: V is a Householder block (unit diagonal, entries about rows^-1/2
// below it), the columns of X and Z have magnitudes from 1 down to 1e-7, as the trailing columns
// of a graded matrix do. The error of every entry must stay within an FP32-class componentwise
// bound plus the absolute floor of the prescaled FP16 parts (the spacing of FP16 subnormals,
// unscaled), which is what the column-wise backward error of the factorization needs.
#include "carrier_wgmma_fp16.cuh"
#include "panel_reference.hpp"
#include <cmath>
#include <iostream>
#include <vector>

using namespace qr_omega;
using panel_reference::require;

namespace {
struct Rng {
    unsigned long long s;
    double uniform() { // [-1, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return double(s >> 11) * 0x1.0p-52 - 1.0;
    }
};
// Componentwise tolerance of the FP32-class products; TF32 alone is about 5e-4.
constexpr double tolerance = 4e-6;
// One FP16 subnormal spacing (2^-24) of a prescaled part, doubled for the two parts.
constexpr double fp16_floor = 0x1.0p-23;

std::vector<float> householder(int rows, int h, Rng &g) {
    std::vector<float> v(size_t(rows) * h, 0.f);
    const double s = 1.0 / std::sqrt(double(rows));
    for (int k = 0; k < h; ++k)
        for (int r = k; r < rows; ++r)
            v[r + size_t(k) * rows] = r == k ? 1.f : float(s * g.uniform() * 1.7);
    return v;
}
double column_scale(int j) {
    static const double s[] = {1.0, 1e-3, 1e-7, 0.25};
    return s[j % 4];
}

// W(h x q) = V^T X with rows split over c slices.
double check_w(int rows, int h, int q, int c, bool x3h) {
    Rng g{0x5eedULL + rows * 31 + h * 7 + q + c};
    const int ldx = (rows + 3) / 4 * 4, ldw = (h + 3) / 4 * 4;
    const auto v = householder(rows, h, g);
    std::vector<float> x(size_t(ldx) * q, 0.f);
    for (int j = 0; j < q; ++j)
        for (int r = 0; r < rows; ++r)
            x[r + size_t(j) * ldx] = float(g.uniform() * column_scale(j));
    Stream st;
    Buffer<float> dv(v.size()), dx(x.size()), dw(size_t(std::max(c, 1)) * ldw * q);
    // The split V: 2 ldvs h halves (FP16) or floats (TF32).
    Buffer<float> split(2 * size_t(gmma_vsplit_ld(rows)) * h + 64);
    Buffer<int> status(1);
    dv.upload(v, st);
    dx.upload(x, st);
    status.zero(st);
    const float xs = x3h_x_scale(rows);
    if (x3h) {
        require(gmma_w3h_admits(c, dv.p, rows, dx.p, ldx, dw.p, ldw, rows, h, q, split.p, xs),
                "3xFP16 W not admitted");
        launch_gmma_w3h(c, dv.p, rows, dx.p, ldx, dw.p, ldw, rows, h, q, st, split.p, xs, 0,
                        status.p);
    } else {
        require(gmma_w3rs_admits(c, dv.p, rows, dx.p, ldx, dw.p, ldw, rows, h, q, split.p),
                "3xTF32 W not admitted");
        launch_gmma_w3rs(c, dv.p, rows, dx.p, ldx, dw.p, ldw, rows, h, q, st, split.p, 0);
    }
    st.sync();
    require(status.download()[0] == OK, "range check fired on in-range reflectors");
    const auto w = dw.download();
    double worst = 0;
    for (int j = 0; j < q; ++j) {
        long double xn1 = 0;
        for (int r = 0; r < rows; ++r)
            xn1 += std::fabs(x[r + size_t(j) * ldx]);
        for (int i = 0; i < h; ++i) {
            long double ref = 0, mag = 0, vn1 = 0;
            for (int r = 0; r < rows; ++r) {
                const long double a = v[r + size_t(i) * rows], b = x[r + size_t(j) * ldx];
                ref += a * b;
                mag += std::fabs(a * b);
                vn1 += std::fabs(a);
            }
            const double err = double(std::fabs(w[i + size_t(j) * ldw] - ref));
            const double bound = tolerance * double(mag) +
                                 fp16_floor * (double(vn1) / xs + double(xn1) / gmma::x3h_v_scale);
            worst = std::max(worst, err / bound);
        }
    }
    return worst;
}

// X(rows x q) -= V Z over h reflectors.
double check_d(int rows, int h, int q, bool x3h) {
    Rng g{0xd00dULL + rows * 13 + h * 5 + q};
    const int ldx = (rows + 3) / 4 * 4, ldz = (h + 3) / 4 * 4, hs = gmma_x3_seg(h);
    const auto v = householder(rows, h, g);
    std::vector<float> z(size_t(ldz) * q, 0.f), x(size_t(ldx) * q, 0.f);
    for (int j = 0; j < q; ++j) {
        for (int k = 0; k < h; ++k)
            z[k + size_t(j) * ldz] = float(g.uniform() * column_scale(j) * 3);
        for (int r = 0; r < rows; ++r)
            x[r + size_t(j) * ldx] = float(g.uniform() * column_scale(j));
    }
    Stream st;
    Buffer<float> dv(v.size()), dz(z.size()), dx(x.size());
    Buffer<float> vr(3 * size_t(rows) * hs + 64), z3(3 * size_t(hs) * q + 64), scratch(32);
    Buffer<int> status(1);
    dv.upload(v, st);
    dz.upload(z, st);
    dx.upload(x, st);
    status.zero(st);
    if (x3h) {
        auto *hvr = reinterpret_cast<cutlass::half_t *>(vr.p);
        auto *hz3 = reinterpret_cast<cutlass::half_t *>(z3.p);
        auto *zmax = reinterpret_cast<unsigned *>(scratch.p);
        float *alpha = scratch.p + 1;
        require(gmma_dh_admits(hvr, 3 * hs, hz3, 3 * hs, dx.p, ldx, rows, 3 * hs, q, alpha),
                "3xFP16 D not admitted");
        launch_pack_vr_h(dv.p, rows, rows, h, hs, hvr, 3 * hs, st, status.p);
        pack_z3_fp16(dz.p, ldz, h, hs, q, hz3, 3 * hs, zmax, alpha, st, status.p);
        launch_gmma_dh(hvr, 3 * hs, hz3, 3 * hs, dx.p, ldx, rows, 3 * hs, q, alpha, st, 0);
    } else {
        require(gmma_d_admits(vr.p, 3 * hs, z3.p, 3 * hs, dx.p, ldx, rows, 3 * hs, q),
                "3xTF32 D not admitted");
        launch_pack_vr(dv.p, rows, rows, h, hs, vr.p, 3 * hs, true, st);
        pack_z3<<<pack_vr_grid((long long)q * hs), 256, 0, st>>>(dz.p, ldz, h, hs, q, z3.p,
                                                                 3 * hs);
        launch_gmma_d(vr.p, 3 * hs, z3.p, 3 * hs, dx.p, ldx, rows, 3 * hs, q, st, 0);
    }
    st.sync();
    require(status.download()[0] == OK, "range check fired on in-range operands");
    const auto out = dx.download();
    double zmax = 0;
    for (float e : z)
        zmax = std::max(zmax, double(std::fabs(e)));
    double worst = 0;
    for (int j = 0; j < q; ++j) {
        long double zn1 = 0;
        for (int k = 0; k < h; ++k)
            zn1 += std::fabs(z[k + size_t(j) * ldz]);
        for (int r = 0; r < rows; ++r) {
            long double ref = x[r + size_t(j) * ldx], mag = std::fabs(ref), vn1 = 0;
            for (int k = 0; k < h; ++k) {
                const long double a = v[r + size_t(k) * rows], b = z[k + size_t(j) * ldz];
                ref -= a * b;
                mag += std::fabs(a * b);
                vn1 += std::fabs(a);
            }
            const double err = double(std::fabs(out[r + size_t(j) * ldx] - ref));
            // Z is prescaled by about 2^14 / max |Z|.
            const double bound = tolerance * double(mag) +
                                 fp16_floor * (double(vn1) * zmax * 0x1.0p-13 +
                                               double(zn1) / gmma::x3h_v_scale);
            worst = std::max(worst, err / bound);
        }
    }
    return worst;
}

// A reflector entry beyond the range x3h_v_scale admits must set FP16_RANGE.
void check_range() {
    const int rows = 512, h = 64, q = 128, ldx = 512, ldw = 64;
    Rng g{7};
    auto v = householder(rows, h, g);
    v[100 + size_t(3) * rows] = 1e5f;
    std::vector<float> x(size_t(ldx) * q, 0.5f);
    Stream st;
    Buffer<float> dv(v.size()), dx(x.size()), dw(size_t(ldw) * q);
    Buffer<float> split(2 * size_t(rows) * h), vr(3 * size_t(rows) * h);
    Buffer<int> status(1);
    dv.upload(v, st);
    dx.upload(x, st);
    status.zero(st);
    launch_gmma_w3h(1, dv.p, rows, dx.p, ldx, dw.p, ldw, rows, h, q, st, split.p,
                    x3h_x_scale(rows), 0, status.p);
    st.sync();
    require(status.download()[0] == FP16_RANGE, "the split of W missed an out-of-range entry");
    status.zero(st);
    launch_pack_vr_h(dv.p, rows, rows, h, h, reinterpret_cast<cutlass::half_t *>(vr.p), 3 * h, st,
                     status.p);
    st.sync();
    require(status.download()[0] == FP16_RANGE, "the packing of D missed an out-of-range entry");
}
} // namespace

int main() {
    try {
        struct W {
            int rows, h, q, c;
        };
        for (const W &t : {W{1000, 64, 128, 1}, W{4096, 200, 300, 2}, W{16384, 512, 128, 4},
                           W{65536, 128, 128, 4}}) {
            const double h16 = check_w(t.rows, t.h, t.q, t.c, true),
                         t32 = check_w(t.rows, t.h, t.q, t.c, false);
            std::cout << "W rows=" << t.rows << " h=" << t.h << " q=" << t.q << " c=" << t.c
                      << ": error / bound 3xfp16 " << h16 << ", 3xtf32 " << t32 << '\n';
            require(h16 <= 1, "3xFP16 W beyond its bound");
        }
        struct D {
            int rows, h, q;
        };
        for (const D &t : {D{1000, 64, 128}, D{4096, 200, 300}, D{16384, 512, 128},
                           D{65536, 128, 128}}) {
            const double h16 = check_d(t.rows, t.h, t.q, true),
                         t32 = check_d(t.rows, t.h, t.q, false);
            std::cout << "D rows=" << t.rows << " h=" << t.h << " q=" << t.q
                      << ": error / bound 3xfp16 " << h16 << ", 3xtf32 " << t32 << '\n';
            require(h16 <= 1, "3xFP16 D beyond its bound");
        }
        check_range();
        std::cout << "All 3xFP16 products passed extended-precision reference checks.\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
