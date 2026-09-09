// discover_main.cu -- print the machine tqr::discover() finds on this node.
//
//   nvcc -std=c++17 -arch=sm_90a -O2 -o tqr_discover src/discover_main.cu -lcudart
//   ./tqr_discover [--fp32] [--devices N] [--no-links] [--quick]

#include "discover.cuh"
#include "plan.cuh"
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    tqr::discovery_options opt;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (!std::strcmp(a, "--fp32"))     opt.word_bytes = 4;
        else if (!std::strcmp(a, "--fp64"))     opt.word_bytes = 8;
        else if (!std::strcmp(a, "--no-links")) opt.measure_links = false;
        else if (!std::strcmp(a, "--quick"))  { opt.repeats = 3; opt.warmups = 1;
                                                opt.latency_repeats = 3;
                                                opt.latency_max_repeats = 6; }
        else if (!std::strcmp(a, "--devices") && i + 1 < argc)
            opt.device_count = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--first-device") && i + 1 < argc)
            opt.first_device = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--l2-partitions") && i + 1 < argc)
            opt.l2_partitions_override = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--ranks") && i + 1 < argc)
            opt.nranks = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr,
                "usage: %s [--fp32|--fp64] [--first-device N] [--devices N]"
                " [--no-links] [--quick]\n"
                "                 [--l2-partitions N]\n"
                "  Probe an IDLE device: a co-tenant evicting the L2 working set will\n"
                "  stop the L2 latency probe converging, and discovery will say so.\n",
                argv[0]);
            return 2;
        }
    }

    tqr::machine m;
    tqr::discovery_report rep;
    if (!tqr::discover(m, rep, opt)) {
        std::fprintf(stderr, "discover failed: %s\n", rep.error);
        return 1;
    }
    if (rep.rank == 0) rep.print();
    if (!rep.converged)
        std::fprintf(stderr,
            "\nwarning: %s did not settle. Check for another process on this device\n"
            "         (nvidia-smi) and probe an idle one with --first-device.\n",
            rep.warning);

    if (rep.rank != 0) return 0;
    std::printf("\nmachine %s   L = %d   %d bytes/word\n", m.name, m.depth(), m.word_bytes);
    std::printf("%-10s %13s %13s %8s %9s %5s %3s %13s %12s %11s\n",
                "region", "M (words)", "Mtilde", "xi", "Pbar", "P", "D",
                "B (words/s)", "alpha (s)", "Lhat");
    for (int l = 0; l < m.region_count(); ++l) {
        std::printf("%-10s %13.6g %13.6g %8.4f %9lld", m.region_at(l).name,
                    m.capacity(l), m.aggregate_capacity(l), m.xi(l), m.total_domains(l));
        if (l < m.boundary_count()) {
            const tqr::machine::boundary& b = m.boundary_at(l);
            std::printf(" %5d %3d %13.5g %12.5g %11.5g\n",
                        b.peers, b.tree_depth(), b.bandwidth, b.latency, m.l_hat(l));
        } else {
            std::printf(" %5s %3s %13s %12s %11s\n", "-", "-", "-", "-", "-");
        }
    }

    // Definition 2.4 decides which intensity ceiling governs each boundary, and it
    // depends on the problem, so it is reported per size rather than baked in.
    const double sizes[] = {4096.0, 32768.0, 262144.0};
    std::printf("\nregime by problem size (Definition 2.4, c = 1)\n");
    std::printf("%-26s", "boundary");
    for (double n : sizes) std::printf(" %12.0f", n);
    std::printf("\n");
    for (int l = 0; l < m.boundary_count(); ++l) {
        char lbl[48];
        std::snprintf(lbl, sizeof(lbl), "%d  %s -> %s", l,
                      m.region_at(l).name, m.region_at(l + 1).name);
        std::printf("%-26s", lbl);
        for (double n : sizes)
            std::printf(" %12s", m.regime_at(l, n, n) == tqr::machine::kResident
                                 ? "resident" : "streaming");
        std::printf("\n");
    }
    // Algorithm 4 on the machine just discovered. The geometry is not searched for:
    // every width is a closed form or the unique root of a scalar equation.
    tqr::plan pl[tqr::machine::kMaxBoundaries];
    const int np = tqr::plan_hierarchy(m, 32768.0, 32768.0, pl);
    std::printf("\nAlgorithm 4 geometry for a 32768 x 32768 problem "
                "(widths in words; beta = 2)\n");
    std::printf("%-10s %10s %10s %10s %10s %6s %5s %9s\n",
                "boundary", "regime", "nu", "gamma", "mu", "c", "p", "Lhat");
    for (int i = 0; i < np; ++i) {
        const tqr::plan& q = pl[i];
        if (!q.ok) { std::printf("%-10d  (not planned: %s)\n", i, q.note); continue; }
        char lbl[32];
        std::snprintf(lbl, sizeof lbl, "%d %s", q.level, m.region_at(q.level).name);
        std::printf("%-10s %10s %10.4g %10.4g %10.4g %6.3g %5.3g %9.4g\n",
                    lbl, q.innermost_fixed ? "innermost"
                       : (q.regime == tqr::machine::kResident ? "resident" : "streaming"),
                    q.nu, q.gamma, q.mu, q.c, q.p, q.L_hat);
    }

    // Who carries what, and what each leg costs. Combine goes one-sided because the
    // ordered stack-Householder merge is not commutative; bulk movement goes collective.
    std::printf("\n%-10s %8s %9s %8s %9s %6s %11s %11s %11s\n",
                "boundary", "K", "a_combine", "D_comb", "a_move", "D_move",
                "t_bw", "t_lat_comb", "t_total");
    for (int i = 0; i < np; ++i) {
        const tqr::plan& q = pl[i];
        if (!q.ok || q.innermost_fixed) continue;
        char lbl[32];
        std::snprintf(lbl, sizeof lbl, "%d %s", q.level, m.region_at(q.level).name);
        std::printf("%-10s %8.0f %7.3f us %8d %7.3f us %6d %11.4g %11.4g %11.4g\n",
                    lbl, q.K, q.alpha_combine * 1e6, q.D_combine,
                    q.alpha_move * 1e6, q.D_move,
                    q.t_bandwidth, q.t_lat_combine, q.t_total);
    }
    if (np > 0 && pl[0].ok)
        std::printf("  leaf width b = %.4g words, rho_0 = %.4g, R_dom <= %.4g flop/s\n",
                    pl[0].b, pl[0].rho0, pl[0].R_dom);
    return 0;
}
