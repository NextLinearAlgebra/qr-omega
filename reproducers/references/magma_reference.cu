// MAGMA geqrf2_gpu / geqrf2_mgpu benchmark adapter: times the factorization and validates it.
#include "kernels.cuh"
#include <magma_v2.h>
#include <cstdlib>
#include "reference_duration.hpp"
#ifdef TQR_MAGMA_TUNING_HOOK
extern "C" unsigned long long tqr_magma_nb_calls(int);
#endif
#ifdef TQR_REFERENCE_VALIDATE
#include "reference_validation.hpp"
#endif
using namespace tqr;
template <class T>
__global__ void magma_input(T *a, int m, int local_n, int ld, int rank, int p, int block) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < size_t(m) * local_n;
         index += size_t(blockDim.x) * gridDim.x) {
        int r = index % m, j = index / m, g = (j / block) * p * block + rank * block + j % block;
        a[r + size_t(j) * ld] =
            T(int(mix64(uint64_t(r) + uint64_t(g) * std::max(m, 1) + 73) % 2001) - 1000) / T(1000);
    }
}
#ifdef TQR_REFERENCE_VALIDATE
template <class T>
json validate_magma_factors(int m, int n, int p, int block, int ld, const std::vector<T *> &a,
                            const std::vector<T> &tau) {
    CU(cudaSetDevice(0));
    Stream stream;
    cusolverDnHandle_t handle = nullptr;
    reference_solver_check(cusolverDnCreate(&handle), "validation handle");
    try {
        reference_solver_check(cusolverDnSetStream(handle, stream), "validation stream");
        reference_solver_check(cusolverDnSetMathMode(handle, CUSOLVER_DEFAULT_MATH),
                               "validation math");
        ReferenceOrmqr<T> apply(handle, stream);
        Buffer<T> device_tau(tau.size()), panel(p > 1 ? checked_mul(size_t(ld), size_t(block)) : 0);
        device_tau.upload(tau, stream);
        stream.sync();
        Buffer<T> gathered;
        const bool full_gather = std::getenv("TQR_MAGMA_VALIDATION_GATHER") != nullptr;
        if (full_gather) {
            size_t bytes = checked_mul(checked_mul(size_t(ld), size_t(n)), sizeof(T));
            if (bytes > size_t(1024) * 1024 * 1024)
                throw std::runtime_error("diagnostic_gather_limit");
            gathered.alloc(size_t(ld) * n);
            for (int j = 0; j < n; ++j) {
                int owner = (j / block) % p, local = j / (p * block) * block + j % block;
                const T *source = a[owner] + size_t(local) * ld;
                if (owner == 0)
                    CU(cudaMemcpy(gathered.p + size_t(j) * ld, source, size_t(m) * sizeof(T),
                                  cudaMemcpyDeviceToDevice));
                else
                    CU(cudaMemcpyPeer(gathered.p + size_t(j) * ld, 0, source, owner,
                                      size_t(m) * sizeof(T)));
            }
            CU(cudaDeviceSynchronize());
        }
        auto result = validate_conventional_reference<T>(
            m, n, block,
            [&](T *x, int ldx, int col, int q) {
                if (!m)
                    return;
                CU(cudaMemset2DAsync(x, size_t(ldx) * sizeof(T), 0, size_t(m) * sizeof(T), q,
                                     stream));
                for (int j = 0; j < q; ++j) {
                    int global = col + j, owner = (global / block) % p;
                    int local = global / (p * block) * block + global % block;
                    size_t bytes = size_t(std::min(m, global + 1)) * sizeof(T);
                    const T *source = full_gather ? gathered.p + size_t(global) * ld
                                                  : a[owner] + size_t(local) * ld;
                    if (full_gather)
                        owner = 0;
                    if (owner == 0)
                        CU(cudaMemcpyAsync(x + size_t(j) * ldx, source, bytes,
                                           cudaMemcpyDeviceToDevice, stream));
                    else
                        CU(cudaMemcpyPeerAsync(x + size_t(j) * ldx, 0, source, owner, bytes,
                                               stream));
                }
            },
            [&](T *x, int ldx, int q, cublasOperation_t operation) {
                int k = std::min(m, n);
                if (!m || !q || !k)
                    return;
                if (full_gather) {
                    apply.apply(m, q, k, gathered.p, ld, device_tau.p, x, ldx, operation);
                    return;
                }
                auto apply_panel = [&](int first) {
                    int width = std::min(block, k - first), owner = (first / block) % p;
                    int local = first / (p * block) * block;
                    const T *v = a[owner] + size_t(local) * ld;
                    if (owner) {
                        CU(cudaMemcpyPeerAsync(panel.p, 0, v, owner, size_t(ld) * width * sizeof(T),
                                               stream));
                        v = panel.p;
                    }
                    apply.apply(m - first, q, width, v + first, ld, device_tau.p + first, x + first,
                                ldx, operation);
                };
                // Q = Q_0 Q_1 ...: rightmost block acts first on X.
                if (operation == CUBLAS_OP_N)
                    for (int first = (k - 1) / block * block; first >= 0; first -= block)
                        apply_panel(first);
                else
                    for (int first = 0; first < k; first += block)
                        apply_panel(first);
            });
        result["copy_visibility"] =
            "peer/local copies and Ormqr execute on the same validation stream; explicit completion before slot reuse";
        result["Q_application"] =
            "conventional MAGMA A/tau, blockwise cuSOLVER Ormqr on GPU0; peer copies and validation excluded from factor timing";
        result["diagnostic_gathered_bytes"] = gathered.n * sizeof(T);
        result["panel_buffer_bytes"] = panel.n * sizeof(T);
        result["ormqr_workspace_bytes"] = apply.workspace_bytes();
        reference_solver_check(cusolverDnDestroy(handle), "validation handle destroy");
        return result;
    } catch (...) {
        cusolverDnDestroy(handle);
        throw;
    }
}
#endif
template <class T> int bench(int m, int n, int p, int reps) {
    static_assert(sizeof(magma_int_t) == 8, "benchmark ABI must match supplied ILP64 dependency");
    int block = sizeof(T) == 4 ? magma_get_sgeqrf_nb(m, n) : magma_get_dgeqrf_nb(m, n),
        ld = std::max(1, ceildiv(m, 32) * 32);
    std::vector<int> cols(p);
    for (int j = 0; j < n; j += block)
        cols[(j / block) % p] += std::min(block, n - j);
    std::vector<T *> a(p, nullptr);
    std::vector<T> tau(std::min(m, n));
    std::vector<size_t> memory;
    for (int rank = 0; rank < p; ++rank) {
        CU(cudaSetDevice(rank));
        if (cols[rank])
            CU(cudaMalloc(&a[rank], size_t(ld) * cols[rank] * sizeof(T)));
        memory.push_back(size_t(ld) * cols[rank] * sizeof(T));
    }
    std::vector<double> times;
    magma_int_t info = 0;
    double first = 0;
    ReferenceDurationLimit duration;
#ifdef TQR_MAGMA_TUNING_HOOK
    std::vector<unsigned long long> tuning_calls;
#endif
    for (int rep = 0; rep <= reps; ++rep) {
        for (int rank = 0; rank < p; ++rank) {
            CU(cudaSetDevice(rank));
            if (m && cols[rank])
                magma_input<<<256, 128>>>(a[rank], m, cols[rank], ld, rank, p, block);
            CU(cudaDeviceSynchronize());
        }
        CU(cudaSetDevice(0));
        double start = seconds();
#ifdef TQR_MAGMA_TUNING_HOOK
        unsigned long long calls_before = tqr_magma_nb_calls(sizeof(T) == 4 ? 0 : 1);
#endif
        if constexpr (sizeof(T) == 4) {
            if (p == 1)
                magma_sgeqrf2_gpu(m, n, a[0], ld, tau.data(), &info);
            else
                magma_sgeqrf2_mgpu(p, m, n, a.data(), ld, tau.data(), &info);
        } else {
            if (p == 1)
                magma_dgeqrf2_gpu(m, n, a[0], ld, tau.data(), &info);
            else
                magma_dgeqrf2_mgpu(p, m, n, a.data(), ld, tau.data(), &info);
        }
        for (int rank = 0; rank < p; ++rank) {
            CU(cudaSetDevice(rank));
            CU(cudaDeviceSynchronize());
        }
        double elapsed = seconds() - start;
        if (rep)
            times.push_back(elapsed);
        else
            first = elapsed;
#ifdef TQR_MAGMA_TUNING_HOOK
        auto observed = tqr_magma_nb_calls(sizeof(T) == 4 ? 0 : 1) - calls_before;
        tuning_calls.push_back(observed);
        if (!observed)
            throw std::runtime_error("MAGMA_native_call_did_not_observe_tuning_function");
#endif
        if (duration.observe(elapsed) || info)
            break;
    }
    json validation = "info checked; independent numerical reference validation still required";
    bool numerical_pass = true;
#ifdef TQR_REFERENCE_VALIDATE
    if (info == 0) {
        try {
            validation = validate_magma_factors(m, n, p, block, ld, a, tau);
            numerical_pass = validation.value("pass", false);
        } catch (const std::exception &e) {
            validation = {
                {"performed", false}, {"completed", false}, {"pass", nullptr}, {"error", e.what()}};
            numerical_pass = false;
        }
    }
#endif
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    json record = {
        {"reference", "MAGMA 2.10.0 geqrf2_gpu/geqrf2_mgpu"},
        {"precision", precision<T>()},
        {"m", m},
        {"n", n},
        {"gpus", p},
        {"block", block},
        {"ld", ld},
        {"status", info},
        {"matrix_bytes_per_gpu", memory},
        {"first_s", first},
        {"raw_s", times},
        {"median_s", sorted.empty() ? 0 : sorted[sorted.size() / 2]},
        {"input", "GPU-regenerated deterministic original matrix; 1D block-column cyclic"},
        {"output", "in-place R and conventional A/tau full-Q handle"},
        {"boundary",
         "ready library-native distributed input through all device completion; internal per-call allocation included"},
        {"layout_conversion",
         "not included; report factor-only comparison separately until common-layout conversion is measured"},
        {"math", "FP32/FP64; NVIDIA_TF32_OVERRIDE=0; CPU panel arithmetic is reference-only"},
        {"host_threads", std::getenv("MKL_NUM_THREADS") ? std::getenv("MKL_NUM_THREADS") : "unset"},
        {"validation", validation}};
#ifdef TQR_MAGMA_TUNING_HOOK
    record["block_tuning"] = {
        {"mechanism",
         "isolated public get_geqrf_nb function override; native geqrf2_mgpu unchanged"},
        {"requested",
         std::getenv("TQR_MAGMA_QR_NB") ? std::getenv("TQR_MAGMA_QR_NB") : "native default"},
        {"effective", block},
        {"native_calls_per_factor", tuning_calls}};
#endif
    record["factor_time_limit"] = duration.record();
    if (times.empty())
        record["median_s"] = nullptr;
    for (int rank = 0; rank < p; ++rank) {
        CU(cudaSetDevice(rank));
        if (a[rank])
            CU(cudaFree(a[rank]));
    }
    std::cout << record.dump() << '\n';
    return info || !numerical_pass ? 1 : duration.exceeded() ? 4 : 0;
}
int main(int argc, char **argv) {
    try {
        if (argc != 6)
            throw std::runtime_error("usage: magma_reference fp32|fp64 m n gpus reps");
        int m = std::stoi(argv[2]), n = std::stoi(argv[3]), p = std::stoi(argv[4]),
            reps = std::stoi(argv[5]);
        if (m < 0 || n < 0 || p < 1 || p > 4 || reps < 1)
            throw std::runtime_error("descriptor");
        magma_init();
        int status = std::string(argv[1]) == "fp32" ? bench<float>(m, n, p, reps)
                                                    : bench<double>(m, n, p, reps);
        magma_finalize();
        return status;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
