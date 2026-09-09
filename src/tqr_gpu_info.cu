// Small environment probe: intentionally independent of the QR implementation.
#include <cuda_runtime.h>
#include <cstdio>

static bool checked(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

int main() {
    int count = 0;
    if (!checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount") || count == 0) {
        std::fprintf(stderr, "No accessible CUDA device.\n");
        return 1;
    }
    // The existing QR test uses logical device 0. Select a physical GPU through
    // CUDA_VISIBLE_DEVICES before launching either executable.
    cudaDeviceProp p{};
    int optin = 0, cluster = 0, driver = 0, runtime = 0;
    if (!checked(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties") ||
        !checked(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0), "shared memory query") ||
        !checked(cudaDeviceGetAttribute(&cluster, cudaDevAttrClusterLaunch, 0), "cluster query") ||
        !checked(cudaDriverGetVersion(&driver), "driver query") ||
        !checked(cudaRuntimeGetVersion(&runtime), "runtime query")) return 1;

    std::printf("device=0 name=%s cc=%d.%d SMs=%d visible_devices=%d\n",
                p.name, p.major, p.minor, p.multiProcessorCount, count);
    std::printf("memory_GiB=%.2f L2_bytes=%d shared_optin_bytes=%d\n",
                p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), p.l2CacheSize, optin);
    std::printf("cooperative_launch=%d cluster_launch=%d driver_api=%d runtime=%d\n",
                p.cooperativeLaunch, cluster, driver, runtime);
    if (p.major != 9 || p.minor != 0 || optin < 232448 ||
        !p.cooperativeLaunch || !cluster) {
        std::fprintf(stderr, "This gate expects Hopper 9.0, 232448 B opt-in shared memory, "
                             "and cooperative/cluster launch support.\n");
        return 1;
    }
    std::puts("Environment prerequisites PASS; QR kernels and cluster sizes still require execution tests.");
    return 0;
}
