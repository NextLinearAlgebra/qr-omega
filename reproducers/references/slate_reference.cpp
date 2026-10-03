// SLATE geqrf benchmark adapter (one MPI rank per GPU): times the factorization and validates it.
// Library-owned input tiles also occupy SLATE's existing memory pool, avoiding
// a second full input-sized reservation in geqrf's reserveDeviceWorkspace().
#include "common.hpp"
#include "reference_gpu.hpp"
#include "reference_duration.hpp"
#include <slate/slate.hh>
#include <mpi.h>
#include <omp.h>

static_assert(sizeof(omp_nest_lock_t) == 16, "Use the supplied GCC-ABI SLATE build");
using namespace tqr;
static const char *reference_phase = "initialization";
#ifdef TQR_SLATE_WORKSPACE_ENVELOPE
extern "C" void tqr_slate_workspace_domain(int64_t, int64_t, int64_t);
#endif

int local_count(int n, int nb, int coord, int parts) {
    int count = 0;
    for (int j = coord * nb; j < n; j += parts * nb)
        count += std::min(nb, n - j);
    return count;
}

template <class T> void mark_modified(slate::Matrix<T> &a) {
    for (int64_t j = 0; j < a.nt(); ++j)
        for (int64_t i = 0; i < a.mt(); ++i)
            if (a.tileIsLocal(i, j))
                a.tileModified(i, j, 0);
}

template <class T> void generate_tiles(slate::Matrix<T> &a, int m, int nb) {
    for (int64_t j = 0; j < a.nt(); ++j) {
        for (int64_t i = 0; i < a.mt(); ++i) {
            if (!a.tileIsLocal(i, j))
                continue;
            a.tileGetForWriting(i, j, 0, slate::LayoutConvert::ColMajor);
            auto tile = a(i, j, 0);
            // The existing generator permits explicit global block coordinates.
            reference_generate(tile.data(), int(tile.mb()), int(tile.nb()), int(tile.stride()), m,
                               nb, 1, 1, int(i), 0, int(j) * nb, sizeof(T));
            a.tileModified(i, j, 0);
        }
    }
    CU(cudaDeviceSynchronize());
}

// Stream a block column of R into the established compact row-cyclic buffer.
// All data movement is device-to-device; diagonal masking uses bounded copies.
template <class T>
void r_block(slate::Matrix<T> &a, T *x, int ld, int nr, int local_width, int first, int nb, int pr,
             int pc, int rr, int rc) {
    if (!nr || !local_width)
        return;
    CU(cudaMemset2D(x, size_t(ld) * sizeof(T), 0, size_t(nr) * sizeof(T), local_width));
    if (first % (nb * pc) != 0)
        throw std::runtime_error("unaligned_reference_R_block");
    for (int local_col = 0; local_col < local_width; local_col += nb) {
        int j = first / nb + rc + (local_col / nb) * pc;
        int width = std::min(nb, local_width - local_col);
        if (j >= a.nt())
            throw std::runtime_error("reference_R_column_out_of_range");
        for (int i = rr; i < a.mt() && i <= j; i += pr) {
            if (!a.tileIsLocal(i, j))
                throw std::runtime_error("reference_tile_ownership");
            a.tileGetForReading(i, j, 0, slate::LayoutConvert::ColMajor);
            CU(cudaDeviceSynchronize());
            auto tile = a(i, j, 0);
            if (tile.nb() != width)
                throw std::runtime_error("reference_R_block_width");
            T *destination = x + size_t(local_col) * ld + size_t(i / pr) * nb;
            if (i < j) {
                CU(cudaMemcpy2D(destination, size_t(ld) * sizeof(T), tile.data(),
                                size_t(tile.stride()) * sizeof(T), size_t(tile.mb()) * sizeof(T),
                                width, cudaMemcpyDeviceToDevice));
            } else {
                for (int col = 0; col < width; ++col) {
                    size_t rows = std::min(int(tile.mb()), col + 1);
                    CU(cudaMemcpy(destination + size_t(col) * ld,
                                  tile.data() + size_t(col) * tile.stride(), rows * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
                }
            }
        }
    }
}

template <class T> int bench(int m, int n, int nb, int reps, int p, int rank) {
    int pr = p == 4 ? 2 : p;
    if (const char *value = std::getenv("TQR_SLATE_GRID_ROWS"))
        pr = std::stoi(value);
    if (pr < 1 || pr > p || p % pr)
        throw std::runtime_error("invalid_SLATE_process_grid");
    int pc = p / pr, rr = rank % pr, rc = rank / pr;
    int nr = local_count(m, nb, rr, pr), ld = std::max(1, nr);
#ifdef TQR_SLATE_WORKSPACE_ENVELOPE
    tqr_slate_workspace_domain(nr, nb, n);
#endif
    size_t initial_free, total, input_free, factor_free;
    CU(cudaMemGetInfo(&initial_free, &total));
    double setup_start = seconds();
    reference_phase = "allocate_owned_input_tiles";
    slate::Matrix<T> a(m, n, nb, pr, pc, MPI_COMM_WORLD);
    a.insertLocalTiles(slate::Target::Devices);
    CU(cudaMemGetInfo(&input_free, &total));
    double setup_s = seconds() - setup_start;
    slate::TriangularFactors<T> factors;
    slate::Options opts = {{slate::Option::Target, slate::Target::Devices}};
    json tuning = {{"defaults", "library defaults unless explicitly overridden"}};
    auto option = [&](const char *name, slate::Option key, int64_t minimum = 1) {
        if (const char *value = std::getenv(name)) {
            int64_t v = std::stoll(value);
            if (v < minimum)
                throw std::runtime_error("invalid_SLATE_tuning");
            opts[key] = v;
            tuning[name] = v;
        }
    };
    option("TQR_SLATE_LOOKAHEAD", slate::Option::Lookahead, 0);
    option("TQR_SLATE_PANEL_THREADS", slate::Option::MaxPanelThreads);
    option("TQR_SLATE_INNER_BLOCK", slate::Option::InnerBlocking);
    std::vector<double> times;
    ReferenceDurationLimit duration;
    double first_s = 0;
    for (int rep = 0; rep <= reps; ++rep) {
        reference_phase = "regenerate_input";
        generate_tiles(a, m, nb);
        factors.clear();
        reference_phase = "geqrf";
        if (rank == 0)
            std::cerr << json{{"phase", reference_phase}, {"rep", rep}}.dump() << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        double start = seconds();
        if (m && n)
            slate::geqrf(a, factors, opts);
        CU(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        double end = seconds(), release, completion;
        MPI_Allreduce(&start, &release, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&end, &completion, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rep)
            times.push_back(completion - release);
        else
            first_s = completion - release;
        if (duration.observe(completion - release))
            break;
    }
    CU(cudaMemGetInfo(&factor_free, &total));
    // Paper timing campaign: TQR_REFERENCE_GEQRF_ONLY skips the streamed validation; the winning
    // configuration is validated in a separate full run.
    if (std::getenv("TQR_REFERENCE_GEQRF_ONLY")) {
        auto sorted_t = times;
        std::sort(sorted_t.begin(), sorted_t.end());
        if (rank == 0)
            std::cout
                << json{{"reference", "SLATE 2025.05.28 geqrf, owned device tiles"},
                        {"precision", precision<T>()},
                        {"m", m},
                        {"n", n},
                        {"gpus", p},
                        {"block", nb},
                        {"grid", {pr, pc}},
                        {"OMP_NUM_THREADS", omp_get_max_threads()},
                        {"tuning", tuning},
                        {"setup_s", setup_s},
                        {"first_s", first_s},
                        {"raw_s", times},
                        {"median_s",
                         sorted_t.empty() ? json(nullptr) : json(sorted_t[sorted_t.size() / 2])},
                        {"factor_time_limit", duration.record()},
                        {"rank0_memory",
                         {{"initial_free_bytes", initial_free},
                          {"after_input_free_bytes", input_free},
                          {"after_factor_free_bytes", factor_free}}},
                        {"boundary",
                         "ready library-owned device tiles to retained R/Q factors, all-rank completion; factor allocation included"},
                        {"layout_conversion",
                         "excluded; factor-only boundary; allocation/setup reported separately"},
                        {"validation",
                         {{"performed", false},
                          {"pass", nullptr},
                          {"scope", "timing-only run (TQR_REFERENCE_GEQRF_ONLY)"}}}}
                       .dump()
                << '\n';
        return duration.exceeded() ? 4 : 0;
    }
    reference_phase = "streamed_validation";
    double validation_start = seconds(), worst = 0, inverse = 0;
    int qmax = nb * pc;
    if (const char *value = std::getenv("TQR_REFERENCE_VALIDATION_WIDTH")) {
        int wanted = std::stoi(value);
        if (wanted < 1 || wanted > 8192)
            throw std::runtime_error("reference_validation_width_1_to_8192");
        qmax *= std::max(1, wanted / qmax);
    }
    const int allocation_columns = std::max(std::min(n, qmax), std::min(qmax, 16));
    Buffer<T> x(size_t(ld) * local_count(allocation_columns, nb, rc, pc)), original(x.n);
    for (int col = 0; col < n; col += qmax) {
        int width = std::min(qmax, n - col);
        int local_width = local_count(width, nb, rc, pc);
        T *xp = x.p;
        auto X = slate::Matrix<T>::fromDevices(m, width, &xp, 1, ld, nb, pr, pc, MPI_COMM_WORLD);
        r_block(a, x.p, ld, nr, local_width, col, nb, pr, pc, rr, rc);
        mark_modified(X);
        if (m && width)
            slate::unmqr(slate::Side::Left, slate::Op::NoTrans, a, factors, X, opts);
        CU(cudaDeviceSynchronize());
        reference_generate(original.p, nr, local_width, ld, m, nb, pr, pc, rr, rc, col, sizeof(T));
        // Preserve the original per-rank, one-native-block-column norm.
        for (int first = 0; first < local_width; first += nb) {
            double residual =
                reference_relative(x.p + size_t(first) * ld, original.p + size_t(first) * ld, nr,
                                   std::min(nb, local_width - first), ld, ld, sizeof(T));
            worst = !std::isfinite(residual) ? INFINITY : std::max(worst, residual);
        }
    }
    {
        int width = std::min(qmax, 16), local_width = local_count(width, nb, rc, pc);
        T *xp = x.p;
        auto X = slate::Matrix<T>::fromDevices(m, width, &xp, 1, ld, nb, pr, pc, MPI_COMM_WORLD);
        reference_generate(x.p, nr, local_width, ld, m, nb, pr, pc, rr, rc, nb * pc, sizeof(T));
        if (local_width)
            CU(cudaMemcpy(original.p, x.p, size_t(ld) * local_width * sizeof(T),
                          cudaMemcpyDeviceToDevice));
        mark_modified(X);
        if (m && n) {
            slate::unmqr(slate::Side::Left, slate::Op::NoTrans, a, factors, X, opts);
            slate::unmqr(slate::Side::Left, slate::Op::Trans, a, factors, X, opts);
        }
        CU(cudaDeviceSynchronize());
        inverse = reference_relative(x.p, original.p, nr, local_width, ld, ld, sizeof(T));
    }
    double global_residual, global_inverse;
    MPI_Allreduce(&worst, &global_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&inverse, &global_inverse, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double u = std::numeric_limits<T>::epsilon() / 2;
    double tolerance = std::min(
        0.01, 64 * u *
                  (std::sqrt(double(std::max(1, m))) + std::sqrt(double(std::max(1, n))) + nb) *
                  (1 + std::log2(std::max(1, ceildiv(m, nb)))));
    bool pass = std::isfinite(global_residual) && std::isfinite(global_inverse) &&
                global_residual <= tolerance && global_inverse <= tolerance;
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    if (rank == 0)
        std::cout
            << json{{"reference", "SLATE 2025.05.28 geqrf, owned device tiles"},
                    {"precision", precision<T>()},
                    {"m", m},
                    {"n", n},
                    {"gpus", p},
                    {"block", nb},
                    {"grid", {pr, pc}},
                    {"OMP_NUM_THREADS", omp_get_max_threads()},
                    {"tuning", tuning},
                    {"setup_s", setup_s},
                    {"first_s", first_s},
                    {"raw_s", times},
                    {"median_s", sorted.empty() ? json(nullptr) : json(sorted[sorted.size() / 2])},
                    {"factor_time_limit", duration.record()},
                    {"rank0_memory",
                     {{"initial_free_bytes", initial_free},
                      {"after_input_free_bytes", input_free},
                      {"after_factor_free_bytes", factor_free}}},
                    {"boundary",
                     "ready library-owned device tiles to retained R/Q factors, all-rank completion; factor allocation included"},
                    {"layout_conversion",
                     "excluded; factor-only boundary; allocation/setup reported separately"},
                    {"validation",
                     {{"performed", true},
                      {"pass", pass},
                      {"full_residual_max_rank_block", global_residual},
                      {"input_columns_checked", n},
                      {"validation_block_columns", qmax},
                      {"residual_statistic_local_block_columns", nb},
                      {"Q_QT_16_vector_inverse", global_inverse},
                      {"engineering_tolerance", tolerance},
                      {"validation_s", seconds() - validation_start}}}}
                   .dump()
            << '\n';
    return !pass ? 1 : duration.exceeded() ? 4 : 0;
}

#ifdef TQR_REFERENCE_MODULE
extern "C" int tqr_slate_reference_entry(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    int provided, rank, p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    try {
        if (provided < MPI_THREAD_MULTIPLE)
            throw std::runtime_error("MPI_THREAD_MULTIPLE_required");
        if (argc != 6)
            throw std::runtime_error("usage: slate_tiles_reference fp32|fp64 m n block reps");
        std::string dtype = argv[1];
        int m = std::stoi(argv[2]), n = std::stoi(argv[3]), nb = std::stoi(argv[4]),
            reps = std::stoi(argv[5]);
        if ((dtype != "fp32" && dtype != "fp64") || m < 0 || n < 0 || nb < 1 || reps < 1 || p > 4)
            throw std::runtime_error("descriptor");
        int count;
        CU(cudaGetDeviceCount(&count));
        if (count != 1)
            throw std::runtime_error("use one-gpu-per-rank.sh with exactly one visible device");
        CU(cudaSetDevice(0));
        if (!std::getenv("OMP_NUM_THREADS"))
            omp_set_num_threads(std::max(1, omp_get_num_procs() / p));
        int result = dtype == "fp32" ? bench<float>(m, n, nb, reps, p, rank)
                                     : bench<double>(m, n, nb, reps, p, rank);
        MPI_Finalize();
        return result;
    } catch (const std::exception &e) {
        std::cerr << json{{"reference", "SLATE owned tiles"},
                          {"rank", rank},
                          {"phase", reference_phase},
                          {"status", "failed"},
                          {"reason", e.what()}}
                         .dump()
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
