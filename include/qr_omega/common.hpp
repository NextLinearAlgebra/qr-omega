#pragma once
// Host utilities shared by the engine, the drivers and the reference adapters.
#include <cuda_runtime.h>
#include <json.hpp>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tqr {
using json = nlohmann::json;

inline void cuda_check(cudaError_t e, const char *where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
#define CU(x) ::tqr::cuda_check((x), #x)

inline double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline size_t checked_mul(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b)
        throw std::runtime_error("size overflow");
    return a * b;
}
inline size_t checked_add(size_t a, size_t b) {
    if (a > SIZE_MAX - b)
        throw std::runtime_error("size overflow");
    return a + b;
}
inline int ceildiv(int a, int b) {
    return a / b + (a % b != 0);
}

// Arithmetic of the products when the matrix is stored in FP32: IEEE single precision, TF32 tensor
// cores, or error-compensated 3xTF32. FP64 storage ignores it.
enum class Fp32Math { IEEE, TF32, X3 };
inline Fp32Math &fp32_math() {
    static Fp32Math math = Fp32Math::IEEE;
    return math;
}
template <class T> std::string precision() {
    return sizeof(T) == 4 ? "fp32" : "fp64";
}
// Unit roundoff of the arithmetic, from which the validation tolerances are derived.
template <class T> double working_unit_roundoff() {
    if constexpr (std::is_same_v<T, float>) {
        if (fp32_math() == Fp32Math::TF32)
            return std::ldexp(1.0, -11);
    }
    return std::numeric_limits<T>::epsilon() / 2;
}

// Properties of the current device, queried once.
inline const cudaDeviceProp &device_properties() {
    static std::map<int, cudaDeviceProp> known;
    int device;
    CU(cudaGetDevice(&device));
    auto [it, fresh] = known.try_emplace(device);
    if (fresh)
        CU(cudaGetDeviceProperties(&it->second, device));
    return it->second;
}
// Lets `kernel` launch with `bytes` of dynamic shared memory; `prefer_shared` also asks for the
// largest shared-memory carveout. The attributes are set again only when the request changes.
template <class Kernel>
void reserve_shared_memory(Kernel *kernel, size_t bytes, bool prefer_shared = false) {
    static std::map<std::pair<Kernel *, int>, size_t> reserved;
    int device;
    CU(cudaGetDevice(&device));
    auto [it, fresh] = reserved.try_emplace({kernel, device}, bytes);
    if (!fresh && it->second == bytes)
        return;
    it->second = bytes;
    CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
    if (prefer_shared)
        CU(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                int(cudaSharedmemCarveoutMaxShared)));
}

// Launches `kernel` in clusters of `cluster` blocks.
template <class... Params, class... Args>
void launch_clustered(void (*kernel)(Params...), dim3 grid, dim3 block, size_t shared_bytes,
                      cudaStream_t stream, dim3 cluster, Args &&...args) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    cudaLaunchAttribute attribute;
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = cluster.x;
    attribute.val.clusterDim.y = cluster.y;
    attribute.val.clusterDim.z = cluster.z;
    config.attrs = &attribute;
    config.numAttrs = cluster.x * cluster.y * cluster.z > 1;
    CU(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

struct Stream {
    cudaStream_t s = nullptr;
    Stream() {
        CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    }
    explicit Stream(int priority) {
        CU(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority));
    }
    Stream(const Stream &) = delete;
    ~Stream() {
        if (s)
            cudaStreamDestroy(s);
    }
    operator cudaStream_t() const {
        return s;
    }
    void sync() const {
        CU(cudaStreamSynchronize(s));
    }
    // Streams are created with the least priority unless one is given.
    static int greatest_priority() {
        int least = 0, greatest = 0;
        CU(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        return greatest;
    }
};

struct Event {
    cudaEvent_t e = nullptr;
    Event() {
        CU(cudaEventCreate(&e));
    }
    Event(const Event &) = delete;
    ~Event() {
        if (e)
            cudaEventDestroy(e);
    }
    void record(cudaStream_t s) {
        CU(cudaEventRecord(e, s));
    }
    void wait(cudaStream_t s) {
        CU(cudaStreamWaitEvent(s, e, 0));
    }
};

// Device array of n elements.
template <class T> struct Buffer {
    T *p = nullptr;
    size_t n = 0;
    Buffer() = default;
    explicit Buffer(size_t count) {
        alloc(count);
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept : p(other.p), n(other.n) {
        other.p = nullptr;
        other.n = 0;
    }
    Buffer &operator=(Buffer &&other) noexcept {
        std::swap(p, other.p);
        std::swap(n, other.n);
        return *this;
    }
    ~Buffer() {
        if (p)
            cudaFree(p);
    }
    void alloc(size_t count) {
        if (p)
            throw std::runtime_error("buffer already allocated");
        n = count;
        if (n)
            CU(cudaMalloc(&p, checked_mul(n, sizeof(T))));
    }
    void zero(cudaStream_t s) {
        if (n)
            CU(cudaMemsetAsync(p, 0, n * sizeof(T), s));
    }
    void upload(const std::vector<T> &v, cudaStream_t s) {
        if (v.size() > n)
            throw std::runtime_error("upload exceeds the buffer");
        if (!v.empty())
            CU(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
    }
    std::vector<T> download() const {
        std::vector<T> v(n);
        if (n)
            CU(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }
};

// One-dimensional block-cyclic distribution of m rows over p GPUs in blocks of `block` rows. With
// one GPU every row is local and local and global indices coincide.
struct RowMap {
    int m = 0, p = 1, block = 1;
    int owner(int row) const {
        return (row / block) % p;
    }
    // Global index of a local row.
    __host__ __device__ long long global(int rank, int row) const {
        return (long long)(row / block) * block * p + (long long)rank * block + row % block;
    }
    // Number of rows of `rank` with a global index below `row`.
    int rows_before(int rank, int row) const {
        const int64_t cycle = int64_t(block) * p;
        return int((row / cycle) * block +
                   std::clamp<int64_t>(row % cycle - int64_t(rank) * block, 0, block));
    }
    int local_rows(int rank) const {
        return rows_before(rank, m);
    }
};

// The local rows of an m x n matrix on one GPU, column-major.
template <class T> struct Matrix {
    int m, n, rows, ld;
    Buffer<T> a;
    Matrix(int m_, int n_, int local_rows)
        : m(m_), n(n_), rows(local_rows), ld(std::max(1, local_rows)),
          a(checked_mul(size_t(ld), size_t(n_))) {}
};
} // namespace tqr
