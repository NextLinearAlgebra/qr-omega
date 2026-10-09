// GPU panel regression with an independent CPU application of the returned compact-WY factors.
// Covers sparse/zero tails, rank deficiency, partial mini-panels, uneven domains and TT levels.
#include "tree_update.cuh"
#include "householder_reconstruct.cuh"
#include <numeric>
#include "panel_reference.hpp"

using namespace qr_omega;
using namespace panel_reference;

namespace {

void invalid_trees() {
    for (const auto &shape : std::vector<std::vector<int>>{{0, 1, 192, 4},
                                                           {31, 32, 192, 4},
                                                           {300, 65, 192, 4},
                                                           {300, 8, 192, 8},
                                                           {300, 64, 256, 4},
                                                           {300, 32, 512, 4}}) {
        bool rejected = false;
        try {
            domain_tree(shape[0], shape[1], shape[2], shape[3]);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "an unsupported tree shape was accepted");
    }
}

template <class T>
void check(int rows, int h, int fan_in, int pattern, int repeats, int tiles = 1, int release = 0,
           int domain_layers = 2, int merge_layers = 2) {
    const int tile = tiles == 1 ? domain_height(64, h) : (DomainConfig::rows - h) / 64 * 64;
    const auto tree = domain_tree(rows, h, tile, fan_in, tiles);
    const auto original = input<T>(rows, h + release, pattern);
    // Odd leading dimensions and poisoned padding expose accidental reads beyond the panel.
    const int lda = rows + 3, ldv = rows + 5;
    std::vector<T> padded(size_t(lda) * (h + release), std::numeric_limits<T>::quiet_NaN());
    for (int j = 0; j < h + release; ++j)
        std::copy_n(original.data() + size_t(j) * rows, rows, padded.data() + size_t(j) * lda);
    Buffer<T> a(padded.size()), v(size_t(ldv) * h), vs(std::max<size_t>(1, tree.vstack_words()));
    Buffer<T> t(tree.t_words()), tau(size_t(tree.nodes()) * h);
    Buffer<int> status(1);
    Stream stream;
    TreeSync sync;
    sync.reserve(tree.nodes(), stream);
    // Repeated launches reuse the flags and ticket. Check the final launch after every
    // predecessor has overwritten the same R locations and advanced the publication epoch.
    for (int rep = 0; rep < repeats; ++rep) {
        a.upload(padded, stream);
        status.zero(stream);
        const auto carrier =
            factor_domains(a.p, lda, tree, v.p, ldv, vs.p, t.p, status.p, sync, stream, tau.p,
                           release, nullptr, 0, domain_layers, merge_layers);
        require(carrier.bounded(), "unbounded panel carrier");
    }
    // Discard all packed factors: Q must remain reconstructible from the matrix
    // and retained scalar taus, including zero-tail and rank-deficient cases.
    CU(cudaMemsetAsync(v.p, 0xff, v.n * sizeof(T), stream));
    CU(cudaMemsetAsync(vs.p, 0xff, vs.n * sizeof(T), stream));
    CU(cudaMemsetAsync(t.p, 0xff, t.n * sizeof(T), stream));
    restore_tree(a.p, lda, tree, tau.p, v.p, ldv, vs.p, t.p, stream);
    stream.sync();
    require(status.download()[0] == 0, "unexpected panel status");
    const auto factors = a.download(), leaves = v.download(), merges = vs.download(),
               ts = t.download();
    std::vector<long double> q(size_t(rows) * h, 0);
    struct Node {
        std::vector<int> rows;
        const T *v;
        int ldv;
        const T *t;
    };
    std::vector<Node> reverse_nodes;
    auto apply = [&](const std::vector<int> &map, const T *vn, int stride, const T *tn) {
        apply_node(q, rows, h, map, vn, stride, tn);
        if (release)
            reverse_nodes.push_back({map, vn, stride, tn});
    };
    for (int j = 0; j < h; ++j)
        q[j + size_t(j) * rows] = 1;
    for (int l = tree.levels - 1; l > 0; --l)
        for (int n = 0; n < tree.count[l]; ++n) {
            std::vector<int> map;
            for (int child = 0; child < tree.children(l, n); ++child)
                for (int r = 0; r < h; ++r)
                    map.push_back(int(n * tree.stride[l] + child * tree.stride[l - 1]) + r);
            apply(map, merges.data() + tree.vstack_offset(l) + size_t(n) * fan_in * h,
                  tree.vstack_rows(l), ts.data() + size_t(tree.first[l] + n) * h * h);
        }
    for (int phase = tree.chain - 1; phase > 0; --phase)
        for (int n = 0; n < tree.count[0]; ++n) {
            std::vector<int> map(h + tree.ts_rows(phase, n));
            std::iota(map.begin(), map.begin() + h, n * tree.domain);
            std::iota(map.begin() + h, map.end(), n * tree.domain + phase * tree.tile);
            apply(map, merges.data() + tree.ts_offset(phase) + size_t(n) * DomainConfig::rows,
                  tree.ts_vrows(), ts.data() + size_t(phase * tree.count[0] + n) * h * h);
        }
    for (int n = 0; n < tree.count[0]; ++n) {
        std::vector<int> map(tree.ge_rows(n));
        std::iota(map.begin(), map.end(), n * tree.domain);
        apply(map, leaves.data() + n * tree.leaf_stride(), ldv, ts.data() + size_t(n) * h * h);
    }
    const double tolerance =
        64 * h * (tree.levels + tree.chain - 1) * std::numeric_limits<T>::epsilon();
    const auto [residual, orth] = check_qr(original, factors, q, rows, h, lda, tolerance);
    if (release) {
        std::vector<long double> expected(original.begin() + size_t(rows) * h, original.end());
        for (auto node = reverse_nodes.rbegin(); node != reverse_nodes.rend(); ++node)
            apply_node(expected, rows, h, node->rows, node->v, node->ldv, node->t, release, true);
        const double update_tolerance = std::is_same_v<T, float> && fp32_math() == Fp32Math::TF32
                                            ? 2e-3 * (tree.levels + tree.chain)
                                            : tolerance;
        for (int j = 0; j < release; ++j) {
            long double error = 0, norm = 0;
            for (int r = 0; r < rows; ++r) {
                const long double actual = factors[r + size_t(h + j) * lda];
                const long double want = expected[r + size_t(j) * rows];
                require(std::isfinite(actual), "nonfinite released column");
                error += (actual - want) * (actual - want);
                norm += want * want;
            }
            require(std::sqrt(error / std::max(norm, 1.L)) <= update_tolerance,
                    "released columns disagree with independent CPU Q transpose");
            for (int r = rows; r < lda; ++r)
                require(std::isnan(factors[r + size_t(h + j) * lda]),
                        "released-column padding was overwritten");
        }
    }
    // Reconstruct from an independently formed thin Q. This isolates modified LU
    // and both triangular solves from the GPU tree application used by the engine.
    std::vector<T> thin(size_t(ldv) * h, std::numeric_limits<T>::quiet_NaN());
    for (int j = 0; j < h; ++j)
        for (int r = 0; r < rows; ++r)
            thin[r + size_t(j) * ldv] = T(q[r + size_t(j) * rows]);
    v.upload(thin, stream);
    Buffer<T> work(reconstruction_width * (reconstruction_width + 1));
    reconstruct_householder(v.p, ldv, a.p, lda, rows, h, h, t.p, h, work.p, status.p, stream);
    stream.sync();
    require(status.download()[0] == 0, "unexpected reconstruction status");
    const auto y = v.download(), compact_t = t.download(), signed_r = a.download(),
               lu = work.download();
    std::vector<long double> qhr(size_t(rows) * h, 0);
    for (int j = 0; j < h; ++j)
        qhr[j + size_t(j) * rows] = 1;
    std::vector<int> all_rows(rows);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    apply_node(qhr, rows, h, all_rows, y.data(), ldv, compact_t.data());
    const auto [hr_residual, hr_orth] = check_qr(original, signed_r, qhr, rows, h, lda, tolerance);
    for (int j = 0; j < h; ++j) {
        const T sign = lu[reconstruction_width * reconstruction_width + j];
        require(sign == T(1) || sign == T(-1), "invalid reconstruction sign");
        for (int r = 0; r < rows; ++r)
            require(std::abs(qhr[r + size_t(j) * rows] - sign * q[r + size_t(j) * rows]) <=
                        tolerance,
                    "reconstruction changed thin Q beyond column signs");
        for (int r = rows; r < ldv; ++r)
            require(std::isnan(y[r + size_t(j) * ldv]), "thin Q padding was overwritten");
    }
    std::cout << precision<T>() << " rows=" << rows << " h=" << h << " fan_in=" << fan_in
              << " tiles=" << tiles << " pattern=" << pattern << " release=" << release
              << " residual=" << residual << " orthogonality=" << orth
              << " hr_residual=" << hr_residual << " hr_orthogonality=" << hr_orth << '\n';
}

template <class T> void suite(bool quick) {
    const auto run = [](int rows, int h, int fan, int pattern, int repeats, int tiles = 1) {
        check<T>(rows, h, fan, pattern, repeats, tiles);
    };
    for (int h : {1, 7, 8, 9, 31, 32, 33, 63, 64})
        run(257, h, 4, 0, 3);
    for (int pattern = 1; pattern <= 4; ++pattern)
        run(385, 33, 4, pattern, 3);
    for (int fan_in : {2, 3, 4})
        run(1025, 64, fan_in, 0, 3);
    run(64, 64, 4, 0, 3);
    for (int tiles : {2, 4, 8}) {
        run(1025, 64, 4, 0, 3, tiles);
        run(769, 33, 3, 0, 3, tiles);
    }
    for (int pattern = 1; pattern <= 4; ++pattern)
        run(777, 33, 4, pattern, 3, 2);
    if (!quick)
        // More nodes than SMs, so the producer ticket order must also make progress
        // when some consumers cannot become resident until earlier blocks finish.
        run(32769, 64, 4, 0, 8);
}

template <class T> void releases(bool quick) {
    // A partial second column group, empty final TS stages, and changing publication epochs.
    for (int layers : {1, 2}) {
        check<T>(257, 64, 3, 0, 3, 1, 35, layers, 3 - layers);
        check<T>(1025, 64, 4, 0, 3, 4, 64, layers, layers);
        if (!quick)
            check<T>(769, 64, 3, 2, 3, 8, 1, layers, 3 - layers);
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        bool quick = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) != "--quick")
                throw std::runtime_error("usage: qr_omega_domain_tests [--quick]");
            quick = true;
        }
        invalid_trees();
        suite<double>(quick);
        suite<float>(quick);
        releases<double>(quick);
        for (auto math : {Fp32Math::IEEE, Fp32Math::TF32, Fp32Math::X3}) {
            fp32_math() = math;
            releases<float>(quick);
        }
        std::cout << "domain panel checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "domain panel check failed: " << e.what() << '\n';
        return 1;
    }
}
