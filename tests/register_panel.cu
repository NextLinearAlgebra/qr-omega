// Exercise the register node kernel (the layout of every GPU tree node and node-level merge) and
// its shared-memory publications.
// CPU Q/QR checks are independent of the GPU's factorization and T construction.
#include "panel_register.cuh"
#include "panel_reference.hpp"
#include <numeric>

using namespace tqr;
using namespace panel_reference;

namespace {
template <class T, int Items, int Columns, int Warps> void check(int rows, int h, int pattern) {
    const auto original = input<T>(rows, h, pattern);
    const int lda = rows + 3, ldv = rows + 5;
    std::vector<T> padded(size_t(lda) * h, std::numeric_limits<T>::quiet_NaN());
    for (int j = 0; j < h; ++j)
        std::copy_n(original.data() + size_t(j) * rows, rows, padded.data() + size_t(j) * lda);
    Buffer<T> a(padded.size()), v(size_t(ldv) * h), t(size_t(h) * h), tau(h);
    Buffer<int> status(1);
    Buffer<TilePacket> packet(1);
    Stream stream;
    packet.upload(std::vector<TilePacket>{{0, 0, rows, h, 0}}, stream);
    Carrier carrier;
    RegisterPanel<T> panel{a.p, lda, packet.p, rows, h, v.p, ldv, tau.p, status.p, &carrier};
    for (int rep = 0; rep < 3; ++rep) {
        a.upload(padded, stream);
        status.zero(stream);
        require(
            launch_fused_gx<T, Items, Columns, Warps, Columns * Warps / 32>(panel, t.p, h, stream),
            "production register layout does not fit");
    }
    stream.sync();
    require(status.download()[0] == 0 && carrier.bounded(), "register panel status or carrier");
    std::vector<long double> q(size_t(rows) * h, 0);
    for (int j = 0; j < h; ++j)
        q[j + size_t(j) * rows] = 1;
    std::vector<int> indices(rows);
    std::iota(indices.begin(), indices.end(), 0);
    const auto vectors = v.download(), triangular = t.download();
    apply_node(q, rows, h, indices, vectors.data(), ldv, triangular.data());
    const auto error =
        check_qr(original, a.download(), q, rows, h, lda, sizeof(T) == 8 ? 1e-10 : 3e-5);
    std::cout << (sizeof(T) == 8 ? "fp64" : "fp32") << " rows=" << rows << " h=" << h
              << " pattern=" << pattern << " residual=" << error.first
              << " orthogonality=" << error.second << '\n';
}
template <class T> void suite() {
    for (int h : {1, 7, 8, 9, 31, 32, 33, 63, 64})
        check<T, 8, 4, 16>(256, h, 0);
    for (int rows : {64, 65, 100, 129, 200, 255})
        check<T, 8, 4, 16>(rows, 64, 0);
    for (int pattern = 1; pattern <= 4; ++pattern)
        check<T, 8, 4, 16>(256, 64, pattern);
}
} // namespace

int main() {
    try {
        suite<double>();
        suite<float>();
        std::cout << "register panel checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "register panel check failed: " << e.what() << '\n';
        return 1;
    }
}
