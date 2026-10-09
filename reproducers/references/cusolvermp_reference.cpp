// cuSOLVERMp pgeqrf benchmark adapter (one MPI rank per GPU): times the factorization and
// validates it.
#include "common.hpp"
#include "reference_gpu.hpp"
#include "reference_duration.hpp"
#include "reference_validation.hpp"
#include <cusolverMp.h>
#include <nccl.h>
#include <mpi.h>
#include <dlfcn.h>
using namespace qr_omega;
void mp_check_impl(cusolverStatus_t s, const char *expression) {
    if (s != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverMp_status_" + std::to_string(s) + " at " + expression);
}
#define mp_check(expression) mp_check_impl((expression), #expression)
void nc_check_ref(ncclResult_t s) {
    if (s != ncclSuccess)
        throw std::runtime_error(ncclGetErrorString(s));
}

#include "reference_mp_gather.inc"
template <class T>
int bench(int m, int n, int block, int reps, int p, int rank, int device, ncclComm_t comm) {
    constexpr cudaDataType type = sizeof(T) == 4 ? CUDA_R_32F : CUDA_R_64F;
    int pr = p;
    if (const char *requested = std::getenv("QR_OMEGA_CUSOLVERMP_GRID_ROWS"))
        pr = std::stoi(requested);
    if (pr < 1 || p % pr)
        throw std::runtime_error("process_grid_rows_must_divide_P");
    const int pc = p / pr, rr = rank % pr, rc = rank / pr;
    const int nr = mp_local_length(m, block, pr, rr), nc = mp_local_length(n, block, pc, rc),
              ld = std::max(1, nr);
    Stream st;
    cusolverMpHandle_t h = nullptr;
    cusolverMpGrid_t grid = nullptr;
    cusolverMpMatrixDescriptor_t desc = nullptr, xdesc = nullptr;
    mp_check(cusolverMpCreate(&h, device, st));
    auto setmath = reinterpret_cast<cusolverStatus_t (*)(cusolverMpHandle_t, cusolverMathMode_t)>(
        dlsym(RTLD_DEFAULT, "cusolverMpSetMathMode"));
    if (setmath)
        mp_check(setmath(h, CUSOLVER_DEFAULT_MATH));
    mp_check(cusolverMpCreateDeviceGrid(h, &grid, comm, pr, pc, CUSOLVERMP_GRID_MAPPING_COL_MAJOR));
    Buffer<T> a(size_t(ld) * std::max(1, nc)), tau(std::max(1, nc));
    Buffer<int> info(1);
    mp_check(cusolverMpCreateMatrixDesc(&desc, grid, type, m, n, block, block, 0, 0, ld));
    size_t d = 0, s = 0;
    if (m && n)
        mp_check(cusolverMpGeqrf_bufferSize(h, m, n, a.p, 1, 1, desc, type, &d, &s));
    Buffer<unsigned char> work(d);
    std::vector<unsigned char> host(s);
    std::vector<double> times;
    double first = 0;
    int error = 0;
    ReferenceDurationLimit duration;
    for (int rep = 0; rep <= reps; ++rep) {
        reference_generate(a.p, nr, nc, ld, m, block, pr, pc, rr, rc, 0, sizeof(T));
        MPI_Barrier(MPI_COMM_WORLD);
        double start = seconds();
        if (m && n)
            mp_check(cusolverMpGeqrf(h, m, n, a.p, 1, 1, desc, tau.p, type, work.p, work.n,
                                     host.data(), host.size(), info.p));
        st.sync();
        int local = m && n ? info.download()[0] : 0;
        MPI_Allreduce(&local, &error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double end = seconds(), first_release = 0, last_completion = 0;
        MPI_Allreduce(&start, &first_release, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&end, &last_completion, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double maximum = last_completion - first_release;
        if (rep)
            times.push_back(maximum);
        else
            first = maximum;
        if (duration.observe(maximum) || error)
            break;
    }
    bool validate = std::getenv("QR_OMEGA_REFERENCE_TIMING_ONLY") == nullptr;
    json validation = {
        {"performed", false}, {"pass", nullptr}, {"scope", "no independent validation requested"}};
    if (validate && !error && m && n)
        validation =
            validate_mp_gathered<T>(m, n, block, pr, pc, rank, nr, ld, a.p, tau.p, comm, st);
    bool pass = error == 0 && validate && validation.value("pass", false);
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    int version;
    mp_check(cusolverMpGetVersion(h, &version));
    int nv = 0;
    nc_check_ref(ncclGetVersion(&nv));
    Dl_info ni{}, mi{};
    dladdr(reinterpret_cast<void *>(&ncclGetVersion), &ni);
    dladdr(reinterpret_cast<void *>(&cusolverMpGetVersion), &mi);
    if (rank == 0)
        std::cout
            << json{{"reference", "cuSOLVERMp Geqrf"},
                    {"version", version},
                    {"precision", precision<T>()},
                    {"m", m},
                    {"n", n},
                    {"gpus", p},
                    {"block", block},
                    {"grid", {pr, pc}},
                    {"status", error},
                    {"first_s", first},
                    {"raw_s", times},
                    {"median_s", sorted.empty() ? json(nullptr) : json(sorted[sorted.size() / 2])},
                    {"factor_time_limit", duration.record()},
                    {"device_workspace_bytes", d},
                    {"host_workspace_bytes", s},
                    {"math",
                     setmath
                         ? "DEFAULT_MATH; selected precision; TF32 override0"
                         : "installed legacy default; selected precision; TF32 override0; setter absent"},
                    {"nccl_version", nv},
                    {"nccl_library", ni.dli_fname ? ni.dli_fname : "unknown"},
                    {"cusolvermp_library", mi.dli_fname ? mi.dli_fname : "unknown"},
                    {"backend",
                     "installed cuSOLVERMp with recorded NCCL runtime; no custom numerical offload"},
                    {"boundary",
                     "ready library-native 2D block-cyclic device input through all-rank completion; in-place R and full-Q conventional handle"},
                    {"layout_conversion", "not included; factor-only comparison"},
                    {"validation", validation}}
                   .dump()
            << '\n';
    mp_check(cusolverMpDestroyMatrixDesc(desc));
    mp_check(cusolverMpDestroyGrid(grid));
    mp_check(cusolverMpDestroy(h));
    return !pass ? 1 : duration.exceeded() ? 4 : 0;
}
int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    try {
        if (argc != 6)
            throw std::runtime_error("usage: cusolvermp_reference fp32|fp64 m n block reps");
        int count;
        CU(cudaGetDeviceCount(&count));
        int device = count == 1 ? 0 : rank;
        if (device >= count)
            throw std::runtime_error("GPU_mapping");
        CU(cudaSetDevice(device));
        int m = std::stoi(argv[2]), n = std::stoi(argv[3]), b = std::stoi(argv[4]),
            reps = std::stoi(argv[5]);
        if (m < 0 || n < 0 || b < 1 || reps < 1)
            throw std::runtime_error("descriptor");
        ncclUniqueId id;
        if (rank == 0)
            nc_check_ref(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t comm;
        nc_check_ref(ncclCommInitRank(&comm, p, id, rank));
        int result = std::string(argv[1]) == "fp32"
                         ? bench<float>(m, n, b, reps, p, rank, device, comm)
                         : bench<double>(m, n, b, reps, p, rank, device, comm);
        nc_check_ref(ncclCommDestroy(comm));
        MPI_Finalize();
        return result;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
