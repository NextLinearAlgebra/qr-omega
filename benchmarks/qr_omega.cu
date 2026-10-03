// QR-Omega benchmark: factors a generated m x n matrix on one GPU or, built with MPI, on the GPUs
// of a node (one rank per GPU), times the factorization and checks the factors.
#include "engine.cuh"
#include <map>

using namespace tqr;

namespace {
struct Arguments {
    std::string mode = "fp64", output;
    int m = 4096, n = 4096, reps = 3;
    Options schedule;
};

Arguments parse(int argc, char **argv) {
    Arguments a;
    Options &s = a.schedule;
    const std::map<std::string, int *> integers{{"--m", &a.m},
                                                {"--n", &a.n},
                                                {"--reps", &a.reps},
                                                {"--b", &s.b},
                                                {"--strip", &s.strip},
                                                {"--depth", &s.depth},
                                                {"--aggregate", &s.aggregate},
                                                {"--wc", &s.c.w},
                                                {"--zc", &s.c.z},
                                                {"--groups", &s.groups},
                                                {"--minipanel", &s.minipanel},
                                                {"--window", &s.window},
                                                {"--late-groups", &s.late_groups},
                                                {"--late-rows", &s.late_rows},
                                                {"--compose-groups", &s.compose_groups},
                                                {"--d-tiles", &s.d_tiles}};
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--lookahead") {
            s.lookahead = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("missing value of " + key);
        const std::string value = argv[++i];
        if (key == "--mode")
            a.mode = value;
        else if (key == "--output")
            a.output = value;
        else if (auto it = integers.find(key); it != integers.end())
            *it->second = std::stoi(value);
        else
            throw std::runtime_error("unknown option " + key);
    }
    const std::map<std::string, Fp32Math> math{{"fp64", Fp32Math::IEEE},
                                               {"fp32", Fp32Math::IEEE},
                                               {"tf32", Fp32Math::TF32},
                                               {"3xtf32", Fp32Math::X3}};
    if (!math.count(a.mode))
        throw std::runtime_error("--mode is one of fp64, fp32, tf32, 3xtf32");
    if (a.m < a.n || a.n < 1 || a.reps < 1)
        throw std::runtime_error("the matrix needs m >= n >= 1 and at least one repetition");
    fp32_math() = math.at(a.mode);
    return a;
}

// Entry (row, col) of the input: a pseudorandom multiple of 1/1000 in [-1, 1].
template <class T> __device__ T input_entry(long long row, int col, int m) {
    return T(int(mix64(uint64_t(row) + uint64_t(col) * m + 73) % 2001) - 1000) / T(1000);
}
// Columns [first, first + n) of the local rows of the input.
template <class T>
__global__ void generate(T *a, int rows, int n, int ld, RowMap map, int rank, int first) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(rows) * n;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i % rows), j = int(i / rows);
        a[row + size_t(j) * ld] = input_entry<T>(map.global(rank, row), first + j, map.m);
    }
}
// Columns [first, first + n) of R, from the upper triangle of the factored matrix.
template <class T>
__global__ void copy_r(const T *a, int lda, T *r, int ldr, int rows, int n, RowMap map, int rank,
                       int first) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(rows) * n;
         i += size_t(blockDim.x) * gridDim.x) {
        const int row = int(i % rows), j = int(i / rows);
        r[row + size_t(j) * ldr] =
            map.global(rank, row) <= first + j ? a[row + size_t(first + j) * lda] : T(0);
    }
}
// ||x - y||_F / ||y||_F over the local rows.
template <class T> double relative_difference(const Matrix<T> &x, const Matrix<T> &y) {
    if (!x.rows || !x.n)
        return 0;
    Buffer<T> out(3);
    relative_error<<<1, 256>>>(x.a.p, y.a.p, x.rows, x.n, x.ld, y.ld, out.p);
    CU(cudaDeviceSynchronize());
    const auto v = out.download();
    return v[1] != 0 ? double(v[0]) / double(v[1]) : double(v[0]) * double(v[2]);
}

struct Checks {
    int status = 0;
    double residual = 0, orthogonality = 0;
};
// The residual is the largest ||A_J - (QR)_J|| / ||A_J|| over blocks J of `block` columns and over
// the GPUs; the orthogonality error is ||Q (Q^T X) - X|| / ||X|| for 16 pseudorandom vectors X.
template <class T>
Checks validate(Engine<T> &engine, const Matrix<T> &a, const Context &ctx, int block) {
    const RowMap &map = engine.rows();
    Matrix<T> x(a.m, block, a.rows), reference(a.m, block, a.rows);
    Stream st;
    Checks checks;
    for (int col = 0; col < a.n; col += block) {
        x.n = reference.n = std::min(block, a.n - col);
        copy_r<<<256, 128, 0, st>>>(a.a.p, a.ld, x.a.p, x.ld, x.rows, x.n, map, ctx.rank, col);
        generate<<<256, 128, 0, st>>>(reference.a.p, x.rows, x.n, x.ld, map, ctx.rank, col);
        st.sync();
        checks.status = std::max(checks.status, engine.apply_q(x, false));
        const double error = relative_difference(x, reference);
        checks.residual = std::isfinite(error) ? std::max(checks.residual, error) : INFINITY;
    }
    x.n = reference.n = std::min(block, 16);
    generate<<<256, 128, 0, st>>>(x.a.p, x.rows, x.n, x.ld, map, ctx.rank, 19);
    if (x.rows)
        CU(cudaMemcpyAsync(reference.a.p, x.a.p, size_t(x.ld) * x.n * sizeof(T),
                           cudaMemcpyDeviceToDevice, st));
    st.sync();
    checks.status = std::max(checks.status, engine.apply_q(x, false));
    checks.status = std::max(checks.status, engine.apply_q(x, true));
    checks.residual = ctx.max(checks.residual);
    checks.orthogonality = ctx.max(relative_difference(x, reference));
    return checks;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <class T> int run(const Arguments &args, Context &ctx) {
    const Options &schedule = args.schedule;
    Engine<T> engine(ctx, args.m, args.n, schedule);
    const RowMap &map = engine.rows();
    Matrix<T> a(args.m, args.n, map.local_rows(ctx.rank));
    Stream st;
    // The first factorization warms up the kernels and is not timed.
    std::vector<double> times;
    double warmup = 0;
    int status = 0;
    for (int rep = 0; rep <= args.reps && !status; ++rep) {
        generate<<<256, 128, 0, st>>>(a.a.p, a.rows, a.n, a.ld, map, ctx.rank, 0);
        st.sync();
        ctx.barrier();
        const double start = seconds();
        status = engine.factor(a);
        ctx.barrier();
        const double elapsed = ctx.max(seconds()) - ctx.min(start);
        if (rep)
            times.push_back(elapsed);
        else
            warmup = elapsed;
    }
    json record = {{"mode", args.mode}, {"m", args.m},      {"n", args.n},
                   {"gpus", ctx.size},  {"status", status}, {"warmup_s", warmup},
                   {"times_s", times},  {"pass", false}};
    if (!status) {
        const Tally products = engine.products();
        const Checks checks = validate(engine, a, ctx, std::min(schedule.strip, 512));
        const double tolerance = std::min(
            0.01, 32 * working_unit_roundoff<T>() *
                      (std::sqrt(double(args.m)) + std::sqrt(double(args.n)) + schedule.b));
        record["median_s"] = median(times);
        record["products"] = ctx.sum(products.products);
        record["products_carried"] = ctx.sum(products.carried);
        record["residual"] = checks.residual;
        record["orthogonality"] = checks.orthogonality;
        record["tolerance"] = tolerance;
        record["pass"] =
            !checks.status && checks.residual <= tolerance && checks.orthogonality <= tolerance;
    }
    if (ctx.rank == 0) {
        if (args.output.empty())
            std::cout << record.dump(2) << '\n';
        else
            std::ofstream(args.output) << record.dump(2) << '\n';
    }
    return record["pass"] ? 0 : 1;
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
