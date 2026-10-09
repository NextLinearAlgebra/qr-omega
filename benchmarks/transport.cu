// Measure the engine's actual communication adapters, with and without an
// independent GEMM. CUDA-event intervals include host launch gaps; setup,
// warmup and payload validation are outside every measured interval.
#include "plan.hpp"
#include "transport.cuh"
#include <cublas_v2.h>
#include <functional>

using namespace qr_omega;

namespace {
void blas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("transport benchmark GEMM failed: " + std::to_string(status));
}
template <class T> __global__ void constant(T *x, size_t count, T value) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        x[i] = value;
}
template <class T>
__global__ void verify(const T *x, size_t count, int kind, int pr, int pc, int col, T value,
                       int *errors) {
    const size_t total = kind == 3 ? count * pr : count;
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total;
         i += size_t(gridDim.x) * blockDim.x) {
        T expected = value;
        if (kind == 1)
            expected = T(pr * (col + 1) + pc * pr * (pr - 1) / 2);
        else if (kind == 2)
            expected = T((pr - 1) * pc + col + 1);
        else if (kind == 3)
            expected = T((i / count) * pc + col + 1);
        if (x[i] != expected)
            atomicAdd(errors, 1);
    }
}
struct Arguments {
    int grid_rows = 0, reps = 200, samples = 5, compute_n = 1024;
    size_t bytes = 65536;
    std::string mode = "fp64", output;
};
Arguments parse(int argc, char **argv) {
    Arguments a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc)
            throw std::runtime_error("missing value of " + key);
        const std::string value = argv[++i];
        if (key == "--grid-rows")
            a.grid_rows = std::stoi(value);
        else if (key == "--bytes")
            a.bytes = std::stoull(value);
        else if (key == "--reps")
            a.reps = std::stoi(value);
        else if (key == "--samples")
            a.samples = std::stoi(value);
        else if (key == "--compute-n")
            a.compute_n = std::stoi(value);
        else if (key == "--mode")
            a.mode = value;
        else if (key == "--output")
            a.output = value;
        else
            throw std::runtime_error("unknown option " + key);
    }
    if (a.bytes < 8 || a.bytes % 8 || a.bytes > size_t(1) << 30 || a.reps < 1 || a.samples < 1 ||
        a.compute_n < 1 || a.compute_n > 4096 || (a.mode != "fp32" && a.mode != "fp64"))
        throw std::runtime_error(
            "require aligned bytes in [8, 2^30], positive repetitions/samples, "
            "compute-n in [1,4096], and mode fp32 or fp64");
    return a;
}

template <class T> class Gemm {
    cublasHandle_t handle = nullptr;
    const int n;
    Buffer<T> a, b, c;

  public:
    explicit Gemm(int size) : n(size), a(size_t(n) * n), b(a.n), c(a.n) {
        blas(cublasCreate(&handle));
        blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        constant<<<256, 128>>>(a.p, a.n, T(1));
        constant<<<256, 128>>>(b.p, b.n, T(1));
        CU(cudaDeviceSynchronize());
    }
    ~Gemm() {
        if (handle)
            cublasDestroy(handle);
    }
    void operator()(cudaStream_t stream) {
        blas(cublasSetStream(handle, stream));
        const T one = T(1), zero = T(0);
        if constexpr (std::is_same_v<T, double>)
            blas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a.p, n, b.p, n, &zero,
                             c.p, n));
        else
            blas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a.p, n, b.p, n, &zero,
                             c.p, n));
    }
    void check(int *errors, cudaStream_t stream) {
        verify<<<256, 128, 0, stream>>>(c.p, c.n, 0, 1, 1, 0, T(n), errors);
    }
};

double middle(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
}
template <class T> void run(Context &ctx, const Arguments &a) {
    if (ctx.size < 2)
        throw std::runtime_error("transport measurements require at least two GPUs");
    const Grid grid = make_grid(ctx.size, a.grid_rows);
    const int row = grid.row(ctx.rank), col = grid.col(ctx.rank);
    const size_t count = a.bytes / sizeof(T);
    Transport transport(ctx, a.bytes, a.bytes, row, col, grid.pc, {a.bytes});
    Buffer<T> source(count), destination(count * grid.pr);
    Buffer<int> errors(1);
    Stream comm(Stream::greatest_priority()), compute;
    Event start, compute_done, stop;
    Gemm<T> gemm(a.compute_n);
    errors.zero(comm);
    constant<<<256, 128, 0, comm>>>(source.p, count, T(ctx.rank + 1));
    T *slot = static_cast<T *>(transport.slot(0, 0));
    constant<<<256, 128, 0, comm>>>(slot, count, T(ctx.rank + 1));
    transport.bind(comm);
    const std::vector<std::string> names{"row_broadcast",    "column_sum",   "column_max",
                                         "column_allgather", "ordered_pair", "unordered_pair"};
    json measurements = json::array();
    for (int op = 0; op < int(names.size()); ++op) {
        if ((op == 0 && grid.pc == 1) || (op >= 1 && op <= 3 && grid.pr == 1))
            continue;
        auto communicate = [&] {
            if (op == 0)
                transport.row_share(0, 0, 0, a.bytes);
            else if (op == 1)
                transport.column_allreduce(source.p, destination.p, count);
            else if (op == 2)
                transport.column_max(source.p, destination.p, count);
            else if (op == 3)
                transport.column_allgather(source.p, destination.p, count);
            else if (op == 4)
                transport.publish(0, ctx.size - 1, source.p, destination.p, a.bytes);
            else
                transport.publish_unordered(0, ctx.size - 1, source.p, destination.p, a.bytes);
        };
        for (int i = 0; i < 20; ++i) {
            communicate();
            gemm(compute);
        }
        comm.sync();
        compute.sync();
        json result = {{"operation", names[op]},
                       {"payload_bytes", a.bytes},
                       {"participants", op == 0  ? grid.pc
                                        : op < 4 ? grid.pr
                                                 : 2},
                       {"backend", op == 5 ? "NVSHMEM" : "low-latency-nccl-LLBuffer"}};
        const std::vector<std::string> phases{"communication", "compute", "serial", "overlap"};
        for (int phase = 0; phase < 4; ++phase) {
            std::vector<double> device_us, host_us;
            for (int sample = 0; sample < a.samples; ++sample) {
                comm.sync();
                compute.sync();
                ctx.barrier();
                const double begun = seconds();
                start.record(comm);
                if (phase == 1 || phase == 3)
                    start.wait(compute);
                for (int rep = 0; rep < a.reps; ++rep) {
                    if (phase != 1)
                        communicate();
                    if (phase == 2)
                        gemm(comm);
                    else if (phase == 1 || phase == 3)
                        gemm(compute);
                }
                if (phase == 1 || phase == 3) {
                    compute_done.record(compute);
                    compute_done.wait(comm);
                }
                stop.record(comm);
                CU(cudaEventSynchronize(stop.e));
                const double wall = seconds() - begun;
                float elapsed;
                CU(cudaEventElapsedTime(&elapsed, start.e, stop.e));
                device_us.push_back(ctx.max(double(elapsed) * 1000 / a.reps));
                host_us.push_back(ctx.max(wall * 1e6 / a.reps));
                if (phase != 1 && (op < 4 || ctx.rank == ctx.size - 1))
                    verify<<<256, 128, 0, comm>>>(op == 0 ? slot : destination.p, count,
                                                  op < 4 ? op : 0, grid.pr, grid.pc, col,
                                                  T(op == 0 ? row * grid.pc + 1 : 1), errors.p);
                if (phase != 0)
                    gemm.check(errors.p, comm);
                comm.sync();
                if (ctx.max(errors.download()[0]))
                    throw std::runtime_error("payload or GEMM validation failed");
            }
            result[phases[phase]] = {{"device_us_per_iteration", middle(device_us)},
                                     {"host_us_per_iteration", middle(host_us)},
                                     {"device_samples_us", device_us},
                                     {"host_samples_us", host_us}};
        }
        result["serial_over_overlap"] = result["serial"]["device_us_per_iteration"].get<double>() /
                                        result["overlap"]["device_us_per_iteration"].get<double>();
        measurements.push_back(result);
    }
    if (ctx.rank == 0) {
        json result = {
            {"gpus", ctx.size},
            {"grid", {grid.pr, grid.pc}},
            {"mode", a.mode},
            {"reps", a.reps},
            {"samples", a.samples},
            {"warmup", 20},
            {"compute", {{"operation", "native GEMM"}, {"n", a.compute_n}}},
            {"timing", "maximum rank CUDA event and host intervals; includes launch gaps"},
            {"pass", true},
            {"communication", transport.report()},
            {"measurements", measurements}};
        if (a.output.empty())
            std::cout << result.dump(2) << '\n';
        else
            std::ofstream(a.output) << result.dump(2) << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        const Arguments args = parse(argc, argv);
        Context ctx;
        if (args.mode == "fp64")
            run<double>(ctx, args);
        else
            run<float>(ctx, args);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "transport benchmark: " << e.what() << '\n';
        Context::abort();
        return 2;
    }
}
