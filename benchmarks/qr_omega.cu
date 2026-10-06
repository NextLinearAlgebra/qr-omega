// QR-Omega benchmark: factors a generated m x n matrix on one GPU or, built with MPI, on a grid of
// the GPUs of a node (one rank per GPU), times the factorization and checks the factors.
#include "engine.cuh"
#include <cuda_profiler_api.h>
#include <map>

using namespace tqr;

namespace {
struct Arguments {
    std::string mode = "fp64", output;
    int m = 4096, n = 4096, reps = 3;
    double kappa = 0; // > 0: the conditioned input of that condition number
    bool profile_factor = false;
    bool check = true; // validate the factors (off only for schedule searches)
    Options schedule;
};

// A schedule option of s (key without its leading dashes); false if there is none of that name.
bool schedule_option(Options &s, const std::string &key, const std::string &value) {
    const std::map<std::string, int *> integers{{"b", &s.b},
                                                {"grid-rows", &s.grid_rows},
                                                {"strip", &s.strip},
                                                {"depth", &s.depth},
                                                {"aggregate", &s.aggregate},
                                                {"wc", &s.c.w},
                                                {"zc", &s.c.z},
                                                {"dc", &s.c.domain},
                                                {"mc", &s.c.merge},
                                                {"fan-in", &s.fan_in},
                                                {"domain-tiles", &s.domain_tiles},
                                                {"tree-groups", &s.c.tree_groups},
                                                {"d-tiles", &s.d_tiles},
                                                {"tail-from", &s.tail_from}};
    if (key == "lookahead")
        s.lookahead = value != "0";
    else if (key == "panel-format") {
        if (value != "tree" && value != "wy")
            throw std::runtime_error("--panel-format is tree or wy");
        s.reconstruct = value == "wy";
    } else if (key == "tree-update") {
        if (value != "streaming" && value != "resident")
            throw std::runtime_error("--tree-update is streaming or resident");
        s.resident_tree = value == "resident";
    } else if (auto it = integers.find(key); it != integers.end())
        *it->second = std::stoi(value);
    else
        return false;
    return true;
}

Arguments parse(int argc, char **argv) {
    Arguments a;
    Options &s = a.schedule;
    const std::map<std::string, int *> integers{{"--m", &a.m}, {"--n", &a.n}, {"--reps", &a.reps}};
    // --tail-<option>: the tree-format tail's schedule, otherwise this one's
    std::vector<std::pair<std::string, std::string>> tail;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--profile-factor") {
            a.profile_factor = true;
            continue;
        }
        if (key == "--no-check") {
            a.check = false;
            continue;
        }
        if (key == "--lookahead" || key == "--tail-lookahead") {
            if (key == "--lookahead")
                s.lookahead = true;
            else
                tail.push_back({"lookahead", "1"});
            continue;
        }
        if (key == "--no-lookahead" || key == "--tail-no-lookahead") {
            if (key == "--no-lookahead")
                s.lookahead = false;
            else
                tail.push_back({"lookahead", "0"});
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("missing value of " + key);
        const std::string value = argv[++i];
        if (key == "--mode")
            a.mode = value;
        else if (key == "--kappa")
            a.kappa = std::stod(value);
        else if (key == "--output")
            a.output = value;
        else if (auto it = integers.find(key); it != integers.end())
            *it->second = std::stoi(value);
        else if (key.rfind("--tail-", 0) == 0 && key != "--tail-from")
            tail.push_back({key.substr(7), value});
        else if (key.rfind("--", 0) != 0 || !schedule_option(s, key.substr(2), value))
            throw std::runtime_error("unknown option " + key);
    }
    if (!tail.empty()) {
        auto t = std::make_shared<Options>(s);
        t->reconstruct = false;
        t->aggregate = 1;
        t->tail_from = 0;
        for (const auto &[key, value] : tail)
            if (!schedule_option(*t, key, value))
                throw std::runtime_error("unknown option --tail-" + key);
        s.tail = t;
    }
    const std::map<std::string, Fp32Math> math{{"fp64", Fp32Math::IEEE},
                                               {"fp32", Fp32Math::IEEE},
                                               {"tf32", Fp32Math::TF32},
                                               {"3xtf32", Fp32Math::X3}};
    if (!math.count(a.mode))
        throw std::runtime_error("--mode is one of fp64, fp32, tf32, 3xtf32");
    if (a.m < a.n || a.n < 1 || a.reps < 1)
        throw std::runtime_error("the matrix needs m >= n >= 1 and at least one repetition");
    if (a.kappa && (a.kappa < 1 || a.m != a.n || a.n & (a.n - 1)))
        throw std::runtime_error(
            "--kappa needs kappa >= 1 and a square matrix of power-of-two order");
    fp32_math() = math.at(a.mode);
    return a;
}

// Entry (row, col) of the input: a pseudorandom multiple of 1/1000 in [-1, 1]. With the table g of
// a conditioned input (below), the entry of S1 H diag(sigma) H S2 / n instead: H is the Hadamard
// matrix of order n, S1 and S2 pseudorandom signs, so that the singular values are sigma, and
// (H diag(sigma) H)(row, col) = g(row xor col) for the Walsh-Hadamard transform g of sigma.
template <class T>
__device__ T input_entry(long long row, long long col, int m, const double *g = nullptr) {
    if (!g)
        return T(int(mix64(uint64_t(row) + uint64_t(col) * m + 73) % 2001) - 1000) / T(1000);
    const bool flip = (mix64(uint64_t(row) + 977) ^ mix64(uint64_t(col) + 7919)) & 1;
    const double v = g[(row ^ col) & (m - 1)] / m;
    return T(flip ? -v : v);
}
// The table g of a conditioned square input of order n (a power of two) and condition number
// kappa: the Walsh-Hadamard transform of the singular values sigma_k = kappa^(-k / (n - 1)).
std::vector<double> conditioned_table(int n, double kappa) {
    std::vector<double> g(n);
    for (int k = 0; k < n; ++k)
        g[k] = n > 1 ? std::pow(kappa, -double(k) / (n - 1)) : 1;
    for (int h = 1; h < n; h *= 2)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double x = g[j], y = g[j + h];
                g[j] = x + y;
                g[j + h] = x - y;
            }
    return g;
}
// Where the local block of a GPU sits in the matrix: the maps of its rows and columns and its place
// in the grid.
struct Place {
    Cyclic rows, cols;
    int r = 0, c = 0;
    __device__ long long row(int i) const {
        return rows.global(r, i);
    }
    __device__ long long col(int j) const {
        return cols.global(c, j);
    }
};
// Local columns [first, first + cols) of the input (`g`: of the conditioned input).
template <class T>
__global__ void generate(T *a, int rows, int cols, int ld, Place place, int first,
                         const double *g = nullptr) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(rows) * cols;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i % rows), j = int(i / rows);
        a[row + size_t(j) * ld] =
            input_entry<T>(place.row(row), place.col(first + j), place.rows.n, g);
    }
}
// Local columns [first, first + cols) of R, from the upper triangle of the factored matrix.
template <class T>
__global__ void copy_r(const T *a, int lda, T *r, int ldr, int rows, int cols, Place place,
                       int first) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(rows) * cols;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i % rows), j = int(i / rows);
        r[row + size_t(j) * ldr] =
            place.row(row) <= place.col(first + j) ? a[row + size_t(first + j) * lda] : T(0);
    }
}
// ||x - y||_F / ||y||_F over the local block.
template <class T> double relative_difference(const Matrix<T> &x, const Matrix<T> &y) {
    if (!x.rows || !x.cols)
        return 0;
    const size_t elements = size_t(x.rows) * x.cols;
    const int blocks = int(std::min<size_t>(256, (elements + 65535) / 65536));
    Buffer<T> out(size_t(3) * blocks);
    relative_error<<<blocks, 256>>>(x.a.p, y.a.p, x.rows, x.cols, x.ld, y.ld, out.p);
    CU(cudaDeviceSynchronize());
    return relative_error_ratio(out.download());
}

// Columns checked per round: 512, doubled up to 4096 (the width of the reference adapters' checks)
// while the two blocks of every GPU fit in half of its free memory. Wider blocks take fewer passes
// over the retained factors.
template <class T> int validation_width(const Context &ctx, int rows) {
    size_t free = 0, total = 0;
    CU(cudaMemGetInfo(&free, &total));
    int block = 512;
    while (block < 4096 && 4 * size_t(rows) * block * sizeof(T) <= free / 2)
        block *= 2;
    return -ctx.max(-block);
}

struct Checks {
    int status = 0;
    double residual = 0, orthogonality = 0;
};
// The residual is the largest ||A_J - (QR)_J|| / ||A_J|| over blocks J of `block` local columns and
// over the GPUs, every grid column checking its own blocks; the orthogonality error is
// ||Q (Q^T X) - X|| / ||X|| for 16 pseudorandom vectors X on every grid column.
template <class T>
Checks validate(Engine<T> &engine, const Matrix<T> &a, const Context &ctx, const Place &place,
                int block, const double *g) {
    Matrix<T> x(a.m, block, a.rows, block), reference(a.m, block, a.rows, block);
    Stream st;
    Checks checks;
    const int rounds = ctx.max(ceildiv(a.cols, block));
    for (int round = 0; round < rounds; ++round) {
        const int first = round * block;
        x.cols = reference.cols = std::clamp(a.cols - first, 0, block);
        if (x.rows && x.cols) {
            copy_r<<<256, 128, 0, st>>>(a.a.p, a.ld, x.a.p, x.ld, x.rows, x.cols, place, first);
            generate<<<256, 128, 0, st>>>(reference.a.p, x.rows, x.cols, x.ld, place, first, g);
        }
        st.sync();
        checks.status = std::max(checks.status, engine.apply_q(x, false));
        const double error = relative_difference(x, reference);
        checks.residual = std::isfinite(error) ? std::max(checks.residual, error) : INFINITY;
    }
    Place vectors = place;
    vectors.cols = Cyclic{};
    x.cols = reference.cols = std::min(block, 16);
    if (x.rows) {
        generate<<<256, 128, 0, st>>>(x.a.p, x.rows, x.cols, x.ld, vectors, 19);
        CU(cudaMemcpyAsync(reference.a.p, x.a.p, size_t(x.ld) * x.cols * sizeof(T),
                           cudaMemcpyDeviceToDevice, st));
    }
    st.sync();
    checks.status = std::max(checks.status, engine.apply_q(x, false));
    checks.status = std::max(checks.status, engine.apply_q(x, true));
    checks.residual = ctx.max(checks.residual);
    checks.orthogonality = ctx.max(relative_difference(x, reference));
    return checks;
}

// How the products of the last factorization were carried: on every GPU, how many and how much of
// their arithmetic replicated (c >= 2 at some level) and stayed within the 2.5D bound at every
// level, and the distinct carriers of each kind of product on the first GPU.
json carrier_report(const Tally &tally, const Context &ctx) {
    const Tally::Summary s = tally.summary();
    json uses = json::array();
    for (const auto &[key, use] : tally.uses())
        uses.push_back({{"product", product_name(key.first)},
                        {"carrier", describe(key.second)},
                        {"bounded", key.second.bounded()},
                        {"count", use.count},
                        {"flops", use.flops}});
    return {{"products", ctx.sum(s.products)},
            {"replicated", ctx.sum(s.carried)},
            {"bounded", ctx.sum(s.bounded)},
            {"flops", ctx.sum(s.flops)},
            {"replicated_flops", ctx.sum(s.carried_flops)},
            {"bounded_flops", ctx.sum(s.bounded_flops)},
            {"uses", uses}};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <class T> int run(const Arguments &args, Context &ctx) {
    const Options &schedule = args.schedule;
    Engine<T> engine(ctx, args.m, args.n, schedule);
    const Plan &layout = engine.layout();
    const Grid &grid = layout.grid;
    const Place place{layout.rows, layout.cols, grid.row(ctx.rank), grid.col(ctx.rank)};
    Matrix<T> a(args.m, args.n, layout.rows.local(place.r), layout.cols.local(place.c));
    Stream st;
    Buffer<double> table;
    if (args.kappa) {
        const std::vector<double> host = conditioned_table(args.n, args.kappa);
        table.alloc(host.size());
        table.upload(host, st);
        st.sync();
    }
    const double *g = args.kappa ? table.p : nullptr;
    // The first factorization warms up the kernels and is not timed.
    std::vector<double> times;
    double warmup = 0;
    int status = 0;
    for (int rep = 0; rep <= args.reps && !status; ++rep) {
        generate<<<256, 128, 0, st>>>(a.a.p, a.rows, a.cols, a.ld, place, 0, g);
        st.sync();
        ctx.barrier();
        if (args.profile_factor && rep == 1)
            CU(cudaProfilerStart());
        const double start = seconds();
        status = engine.factor(a);
        ctx.barrier();
        const double elapsed = ctx.max(seconds()) - ctx.min(start);
        if (args.profile_factor && rep == 1)
            CU(cudaProfilerStop());
        if (rep)
            times.push_back(elapsed);
        else
            warmup = elapsed;
    }
    json record = {{"mode", args.mode},
                   {"m", args.m},
                   {"n", args.n},
                   {"kappa", args.kappa},
                   {"gpus", ctx.size},
                   {"grid", {grid.pr, grid.pc}},
                   {"replication",
                    {{"w", schedule.c.w},
                     {"z", schedule.c.z},
                     {"domain", schedule.c.domain},
                     {"merge", schedule.c.merge},
                     {"fan_in", schedule.fan_in},
                     {"domain_tiles", schedule.domain_tiles}}},
                   {"tree_groups", schedule.c.tree_groups},
                   {"tree_update", schedule.resident_tree ? "resident" : "streaming"},
                   {"panel_algorithm", "hqr"},
                   {"panel_format", schedule.reconstruct ? "wy" : "tree"},
                   {"aggregate", schedule.aggregate},
                   {"tail_from", schedule.tail_from},
                   {"status", status},
                   {"warmup_s", warmup},
                   {"times_s", times},
                   {"pass", false}};
    if (schedule.tail_from) {
        const Options &t = schedule.tail ? *schedule.tail : schedule;
        record["tail"] = {
            {"strip", t.strip},
            {"depth", t.depth},
            {"domain_tiles", t.domain_tiles},
            {"fan_in", t.fan_in},
            {"replication",
             {{"w", t.c.w}, {"z", t.c.z}, {"domain", t.c.domain}, {"merge", t.c.merge}}},
            {"tree_groups", t.c.tree_groups},
            {"tree_update", t.resident_tree ? "resident" : "streaming"},
            {"lookahead", t.lookahead}};
    }
    if (!status) {
        const Tally products = engine.products();
        record["carriers"] = carrier_report(products, ctx);
        record["update_workspace"] = engine.update_workspace();
        record["median_s"] = median(times);
        record["checked"] = args.check;
    }
    if (!status && args.check) {
        const Checks checks = validate(engine, a, ctx, place, validation_width<T>(ctx, a.rows), g);
        const double tolerance = std::min(
            0.01, 32 * working_unit_roundoff<T>() *
                      (std::sqrt(double(args.m)) + std::sqrt(double(args.n)) + schedule.b));
        record["residual"] = checks.residual;
        record["orthogonality"] = checks.orthogonality;
        record["tolerance"] = tolerance;
        record["pass"] =
            !checks.status && checks.residual <= tolerance && checks.orthogonality <= tolerance;
    }
    if (ctx.rank == 0) {
        record["communication"] = engine.communication();
        if (args.output.empty())
            std::cout << record.dump(2) << '\n';
        else
            std::ofstream(args.output) << record.dump(2) << '\n';
    }
    // Unchecked runs only time a schedule: they succeed when the factorization completes.
    return record["pass"] || (!args.check && !status) ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parse(argc, argv);
        Context ctx;
        return args.mode == "fp64" ? run<double>(args, ctx) : run<float>(args, ctx);
    } catch (const std::exception &e) {
        std::cerr << "qr_omega: " << e.what() << '\n';
        Context::abort();
        return 2;
    }
}
