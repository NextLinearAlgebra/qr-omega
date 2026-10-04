// Lean micro-benchmarks of one GPU and its node. The result is a machine profile (JSON): for every level of the
// memory hierarchy its capacity, access latency and bandwidth, the arithmetic rate of every precision, and the
// latency of the runtime operations that order work (launch, barrier, stream join). Nothing here depends on an
// algorithm; the profile is the input of the cost models in machine/.
//
//   machine_probe [--device 0] [--peer 1] [--output machine.json] [--quick]
//   machine_probe --identify        print the name, UUID and driver of the device and exit
//   machine_probe --product tf32 m n k   time one vendor product C(m x n) = A(m x k) B(k x n) and exit
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
namespace {
constexpr const char* probe_version = "machine-probe/1";

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cublas_status_" + std::to_string(int(s)));
}
#define CK(x) check((x), #x)
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
double minimum(const std::vector<double>& v) { return v.empty() ? 0 : *std::min_element(v.begin(), v.end()); }

template<class T> struct Device {
  T* p = nullptr; size_t n = 0;
  explicit Device(size_t count) : n(count) { if (n) CK(cudaMalloc(&p, n * sizeof(T))); }
  Device(const Device&) = delete;
  ~Device() { if (p) cudaFree(p); }
};
struct Stream {
  cudaStream_t s;
  Stream() { CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(s); }
};
struct Event {
  cudaEvent_t e;
  Event() { CK(cudaEventCreate(&e)); }
  ~Event() { cudaEventDestroy(e); }
};
// Device time of f on one stream, bracketed by events.
template<class F> double event_time(F f, cudaStream_t s) {
  Event a, b;
  CK(cudaEventRecord(a.e, s)); f(); CK(cudaEventRecord(b.e, s)); CK(cudaEventSynchronize(b.e));
  float ms = 0; CK(cudaEventElapsedTime(&ms, a.e, b.e));
  return double(ms) * 1e-3;
}
template<class F> std::vector<double> repeat(F f, cudaStream_t s, int reps, int warm = 2) {
  for (int i = 0; i < warm; ++i) f();
  CK(cudaStreamSynchronize(s));
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) t.push_back(event_time(f, s));
  return t;
}

// In-kernel clocks: SM cycles and device nanoseconds.
struct Tick { unsigned long long cycles, ns; };
__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ Tick tick() { Tick t; t.cycles = (unsigned long long)clock64(); t.ns = global_ns(); return t; }

__global__ void k_empty() {}

// ---- arithmetic -----------------------------------------------------------------------------------------------------
// Dependent chain of fused multiply-adds in registers: ILP = 1 gives the latency of one operation, ILP = 8 with every
// warp resident gives the scalar (non-tensor) arithmetic rate.
template<class T, int ILP> __global__ void k_fma(T* out, T a, T b, long iters, Tick* ticks) {
  T x[ILP];
  for (int j = 0; j < ILP; ++j) x[j] = T(threadIdx.x + j) * T(1e-3);
  const Tick t0 = tick();
  for (long i = 0; i < iters; ++i)
    for (int j = 0; j < ILP; ++j) x[j] = x[j] * a + b;
  const Tick t1 = tick();
  T s = 0;
  for (int j = 0; j < ILP; ++j) s += x[j];
  if (threadIdx.x == 0) {
    out[blockIdx.x] = s;
    if (ticks && blockIdx.x == 0) { ticks[0] = t0; ticks[1] = t1; }
  }
}

// ---- access latency -------------------------------------------------------------------------------------------------
// next[i] = (a i + c) mod n is a single cycle through all n = 2^k entries (Hull-Dobell), so a chase visits distinct
// pseudo-random locations and every load depends on the previous one.
__global__ void k_cycle_init(uint32_t* next, uint32_t n) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
    next[i] = (1664525u * uint32_t(i) + 1013904223u) & (n - 1);
}
__global__ void k_chase(const uint32_t* next, long warm, long iters, Tick* ticks, uint32_t* sink) {
  uint32_t i = 0;
  for (long k = 0; k < warm; ++k) i = next[i];
  const Tick t0 = tick();
  for (long k = 0; k < iters; ++k) i = next[i];
  const Tick t1 = tick();
  ticks[0] = t0; ticks[1] = t1; *sink = i;
}
__global__ void k_chase_shared(long iters, Tick* ticks, uint32_t* sink) {
  constexpr uint32_t n = 4096;
  __shared__ uint32_t next[n];
  for (uint32_t i = 0; i < n; ++i) next[i] = (1664525u * i + 1013904223u) & (n - 1);
  uint32_t i = 0;
  for (uint32_t k = 0; k < n; ++k) i = next[i];
  const Tick t0 = tick();
  for (long k = 0; k < iters; ++k) i = next[i];
  const Tick t1 = tick();
  ticks[0] = t0; ticks[1] = t1; *sink = i;
}
__global__ void k_barrier(long iters, Tick* ticks, int* sink) {
  __shared__ int cell;
  if (threadIdx.x == 0) cell = 0;
  __syncthreads();
  const Tick t0 = tick();
  for (long k = 0; k < iters; ++k) {
    if (threadIdx.x == unsigned(k) % blockDim.x) cell += 1;
    __syncthreads();
  }
  const Tick t1 = tick();
  if (threadIdx.x == 0) { ticks[0] = t0; ticks[1] = t1; *sink = cell; }
}

// ---- bandwidth ------------------------------------------------------------------------------------------------------
// Every block owns a contiguous chunk and its warps read or write 512 consecutive bytes per step, so the grid size is
// the number of SMs that drive the memory. `passes` repeats the sweep inside the kernel for cache-resident sizes.
struct Word16 { unsigned long long a, b; };
__global__ void k_read(const Word16* x, size_t n, int passes, unsigned long long* out) {
  const size_t chunk = n / gridDim.x, first = size_t(blockIdx.x) * chunk;
  unsigned long long acc = 0;
  for (int p = 0; p < passes; ++p)
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) { const Word16 w = x[first + i]; acc += w.a ^ w.b; }
  if (acc == 0x9e3779b97f4a7c15ULL) out[blockIdx.x] = acc;
}
__global__ void k_write(Word16* x, size_t n, int passes) {
  const size_t chunk = n / gridDim.x, first = size_t(blockIdx.x) * chunk;
  for (int p = 0; p < passes; ++p)
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) x[first + i] = Word16{i + p, first};
}
__global__ void k_copy(Word16* y, const Word16* x, size_t n, int passes) {
  const size_t chunk = n / gridDim.x, first = size_t(blockIdx.x) * chunk;
  for (int p = 0; p < passes; ++p)
    for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) y[first + i] = x[first + i];
}
__global__ void k_read_shared(int passes, unsigned long long* out) {
  constexpr int n = 2048;   // 32 KiB of shared memory
  __shared__ Word16 s[n];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = Word16{(unsigned long long)i, 1};
  __syncthreads();
  unsigned long long acc = 0;
  for (int p = 0; p < passes; ++p)
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const Word16 w = s[i]; acc += w.a ^ w.b; }
  if (acc == 0x9e3779b97f4a7c15ULL) out[blockIdx.x] = acc;
}

struct Options { int device = 0, peer = -1; std::string output, arithmetic; bool quick = false, identify = false; int shape[3] = {0, 0, 0}; };

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() { if (++i == argc) throw std::runtime_error("missing value for " + k); return std::string(argv[i]); };
    if (k == "--device") o.device = std::stoi(val());
    else if (k == "--peer") o.peer = std::stoi(val());
    else if (k == "--output") o.output = val();
    else if (k == "--quick") o.quick = true;
    else if (k == "--identify") o.identify = true;
    else if (k == "--product") { o.arithmetic = val(); for (int& x : o.shape) x = std::stoi(val()); }
    else throw std::runtime_error("unknown option " + k);
  }
  return o;
}

std::string first_line(const char* path, const char* key) {
  std::ifstream f(path); std::string line;
  while (std::getline(f, line))
    if (line.rfind(key, 0) == 0) { auto v = line.substr(line.find(':') + 1); v.erase(0, v.find_first_not_of(" \t")); return v; }
  return "";
}

// Two-point fit t(x) = alpha + beta x from the smallest and the largest sample.
json alpha_beta(const std::vector<std::pair<double, double>>& samples) {
  const auto& lo = samples.front(); const auto& hi = samples.back();
  const double beta = (hi.second - lo.second) / (hi.first - lo.first);
  json raw = json::array();
  for (auto& s : samples) raw.push_back({{"bytes", s.first}, {"seconds", s.second}});
  return {{"latency_s", std::max(0.0, lo.second - beta * lo.first)}, {"seconds_per_byte", beta},
          {"bandwidth_bytes_per_s", beta > 0 ? 1.0 / beta : 0.0}, {"samples", raw}};
}

struct Probe {
  Options o; cudaDeviceProp prop{}; Stream st; int sms = 0, blocks_per_sm = 1; double clock_hz = 0;
  json out;

  double cycles_to_s(double cycles) const { return cycles / clock_hz; }

  void device() {
    size_t free_b = 0, total_b = 0; CK(cudaMemGetInfo(&free_b, &total_b));
    int runtime = 0, driver = 0; CK(cudaRuntimeGetVersion(&runtime)); CK(cudaDriverGetVersion(&driver));
    int count = 0; CK(cudaGetDeviceCount(&count));
    sms = prop.multiProcessorCount; blocks_per_sm = std::max(1, prop.maxThreadsPerMultiProcessor / 1024);
    char uuid[40]; const unsigned char* u = (const unsigned char*)prop.uuid.bytes;
    snprintf(uuid, sizeof uuid, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]);
    out["device"] = {{"name", prop.name}, {"uuid", uuid}, {"compute_capability", {prop.major, prop.minor}},
      {"sm_count", sms}, {"warp_size", prop.warpSize}, {"max_threads_per_sm", prop.maxThreadsPerMultiProcessor},
      {"max_threads_per_block", prop.maxThreadsPerBlock}, {"registers_per_sm", prop.regsPerMultiprocessor},
      {"registers_per_block", prop.regsPerBlock}, {"shared_bytes_per_sm", prop.sharedMemPerMultiprocessor},
      {"shared_bytes_per_block", prop.sharedMemPerBlock}, {"shared_bytes_per_block_optin", prop.sharedMemPerBlockOptin},
      {"l2_bytes", prop.l2CacheSize}, {"memory_total_bytes", total_b}, {"memory_free_bytes", free_b},
      {"cuda_runtime", runtime}, {"cuda_driver", driver}};
    json peers = json::array();
    for (int d = 0; d < count; ++d) { int a = d == o.device; if (d != o.device) CK(cudaDeviceCanAccessPeer(&a, o.device, d)); peers.push_back(a); }
    out["node"] = {{"gpu_count", count}, {"peer_access", peers}, {"cpu_model", first_line("/proc/cpuinfo", "model name")},
      {"cpu_affinity", first_line("/proc/self/status", "Cpus_allowed_list")},
      {"host_threads", std::thread::hardware_concurrency()}};
  }

  // SM clock under load: cycles per device nanosecond over a register-only loop.
  void clock() {
    Device<float> sink(1); Device<Tick> ticks(2); std::vector<double> hz;
    for (int r = 0; r < 5; ++r) {
      k_fma<float, 1><<<1, 32, 0, st.s>>>(sink.p, 1.0000001f, 1e-7f, 1L << 22, ticks.p);
      Tick t[2]; CK(cudaMemcpyAsync(t, ticks.p, sizeof t, cudaMemcpyDeviceToHost, st.s)); CK(cudaStreamSynchronize(st.s));
      if (r) hz.push_back(double(t[1].cycles - t[0].cycles) / (double(t[1].ns - t[0].ns) * 1e-9));
    }
    clock_hz = median(hz);
    out["device"]["clock_hz"] = clock_hz;
  }

  template<class T> json simt(const char* name) {
    Device<T> sink(size_t(sms) * blocks_per_sm); Device<Tick> ticks(2);
    const long iters = o.quick ? 1L << 16 : 1L << 18;
    // Latency: one thread, one accumulator.
    std::vector<double> lat;
    for (int r = 0; r < 5; ++r) {
      k_fma<T, 1><<<1, 1, 0, st.s>>>(sink.p, T(1.0000001), T(1e-7), iters, ticks.p);
      Tick t[2]; CK(cudaMemcpyAsync(t, ticks.p, sizeof t, cudaMemcpyDeviceToHost, st.s)); CK(cudaStreamSynchronize(st.s));
      lat.push_back(double(t[1].cycles - t[0].cycles) / double(iters));
    }
    // Rate: every warp resident, eight independent chains per thread; one SM and the whole device.
    auto rate = [&](int blocks) {
      auto t = repeat([&]() { k_fma<T, 8><<<blocks, 1024, 0, st.s>>>(sink.p, T(1.0000001), T(1e-7), iters / 8, nullptr); }, st.s, 5);
      return 2.0 * 8.0 * 1024.0 * blocks * double(iters / 8) / minimum(t);
    };
    const double one = rate(blocks_per_sm), all = rate(sms * blocks_per_sm);
    return {{"unit", name}, {"fma_latency_cycles", median(lat)}, {"fma_latency_s", cycles_to_s(median(lat))},
            {"flops_per_s_per_sm", one}, {"flops_per_s", all}, {"seconds_per_flop", 1.0 / all}};
  }

  void latency() {
    json l;
    l["launch_sync_s"] = median(repeat([&]() { k_empty<<<1, 1, 0, st.s>>>(); }, st.s, 31));
    const int K = 256;
    const double many = median(repeat([&]() { for (int i = 0; i < K; ++i) k_empty<<<1, 1, 0, st.s>>>(); }, st.s, 15));
    l["launch_async_s"] = many / K;
    std::vector<double> enqueue, sync, event;
    for (int i = 0; i < 31; ++i) {
      double t = now(); k_empty<<<1, 1, 0, st.s>>>(); enqueue.push_back(now() - t);
      CK(cudaStreamSynchronize(st.s));
      t = now(); k_empty<<<1, 1, 0, st.s>>>(); CK(cudaStreamSynchronize(st.s)); sync.push_back(now() - t);
      Event a; t = now(); CK(cudaEventRecord(a.e, st.s)); CK(cudaEventSynchronize(a.e)); event.push_back(now() - t);
    }
    l["host_enqueue_s"] = median(enqueue); l["host_launch_and_wait_s"] = median(sync); l["host_event_wait_s"] = median(event);
    // One more block in a grid: the slope between a grid that fills the SMs once and a much larger one.
    const int g0 = sms, g1 = sms * 512;
    const double t0 = median(repeat([&]() { k_empty<<<g0, 32, 0, st.s>>>(); }, st.s, 15));
    const double t1 = median(repeat([&]() { k_empty<<<g1, 32, 0, st.s>>>(); }, st.s, 15));
    l["block_s"] = std::max(0.0, (t1 - t0) / double(g1 - g0));
    l["launch_full_grid_s"] = t0;
    // Joining two streams: fork an empty kernel on a second stream and wait for it.
    { Stream other; Event fork, join;
      l["stream_fork_join_s"] = median(repeat([&]() {
        CK(cudaEventRecord(fork.e, st.s)); CK(cudaStreamWaitEvent(other.s, fork.e, 0));
        k_empty<<<1, 1, 0, other.s>>>(); CK(cudaEventRecord(join.e, other.s)); CK(cudaStreamWaitEvent(st.s, join.e, 0));
      }, st.s, 15)); }
    Device<Tick> ticks(2); Device<int> sink(1); json barrier = json::array();
    for (int threads : {32, 256, 1024}) {
      if (threads > prop.maxThreadsPerBlock) continue;
      const long iters = 1L << 14; std::vector<double> c;
      for (int r = 0; r < 5; ++r) {
        k_barrier<<<1, threads, 0, st.s>>>(iters, ticks.p, sink.p);
        Tick t[2]; CK(cudaMemcpyAsync(t, ticks.p, sizeof t, cudaMemcpyDeviceToHost, st.s)); CK(cudaStreamSynchronize(st.s));
        c.push_back(double(t[1].cycles - t[0].cycles) / double(iters));
      }
      barrier.push_back({{"threads", threads}, {"cycles", median(c)}, {"seconds", cycles_to_s(median(c))}});
    }
    l["barrier"] = barrier;
    out["latency"] = l;
  }

  // Dependent loads over a growing footprint: the plateaus are the latencies of L1, L2 and device memory, and the
  // steps between them are the capacities.
  void access() {
    Device<Tick> ticks(2); Device<uint32_t> sink(1);
    const size_t l2 = size_t(prop.l2CacheSize), cap = std::min<size_t>(o.quick ? 4 * l2 : 32 * l2, size_t(2) << 30);
    json sweep = json::array(); std::vector<std::pair<double, double>> pts;
    for (size_t bytes = 4096; bytes <= cap; bytes *= 2) {
      const uint32_t n = uint32_t(bytes / 4);
      Device<uint32_t> next(n);
      k_cycle_init<<<std::min<size_t>(size_t(sms) * 4, (n + 255) / 256), 256, 0, st.s>>>(next.p, n);
      const long warm = std::min<long>(n, 1L << 18), iters = std::max<long>(1L << 14, std::min<long>(n, 1L << 17));
      std::vector<double> cyc, ns;
      for (int r = 0; r < 4; ++r) {
        k_chase<<<1, 1, 0, st.s>>>(next.p, warm, iters, ticks.p, sink.p);
        Tick t[2]; CK(cudaMemcpyAsync(t, ticks.p, sizeof t, cudaMemcpyDeviceToHost, st.s)); CK(cudaStreamSynchronize(st.s));
        if (r) { cyc.push_back(double(t[1].cycles - t[0].cycles) / double(iters)); ns.push_back(double(t[1].ns - t[0].ns) / double(iters)); }
      }
      sweep.push_back({{"footprint_bytes", bytes}, {"cycles", median(cyc)}, {"seconds", median(ns) * 1e-9}});
      pts.push_back({double(bytes), median(cyc)});
    }
    // Plateaus: L1 is the smallest footprint; a level ends where the latency first exceeds 1.25x its plateau, so an
    // observed capacity is the largest power of two that is still fully resident.
    auto plateau_end = [&](size_t from, double level) { size_t i = from; while (i + 1 < pts.size() && pts[i + 1].second <= 1.25 * level) ++i; return i; };
    const double l1 = pts.front().second; const size_t e1 = plateau_end(0, l1);
    size_t s2 = std::min(e1 + 2, pts.size() - 1);
    // The L2 plateau is read at footprints well inside the reported L2 size, past the L1 transition.
    std::vector<double> mid;
    for (size_t i = e1 + 1; i < pts.size(); ++i) if (pts[i].first >= 4 * pts[e1].first && pts[i].first <= double(l2) / 4) mid.push_back(pts[i].second);
    const double lat2 = mid.empty() ? pts[s2].second : median(mid);
    size_t e2 = s2; for (size_t i = s2; i < pts.size(); ++i) if (pts[i].second <= 1.25 * lat2) e2 = i;
    // At the largest footprint a share l2/footprint of the loads still hits L2; remove it.
    const double hit = std::min(0.5, double(l2) / pts.back().first), mem = (pts.back().second - hit * lat2) / (1 - hit);
    { std::vector<double> c;
      for (int r = 0; r < 5; ++r) {
        k_chase_shared<<<1, 1, 0, st.s>>>(1L << 16, ticks.p, sink.p);
        Tick t[2]; CK(cudaMemcpyAsync(t, ticks.p, sizeof t, cudaMemcpyDeviceToHost, st.s)); CK(cudaStreamSynchronize(st.s));
        c.push_back(double(t[1].cycles - t[0].cycles) / double(1L << 16));
      }
      out["access"]["shared"] = {{"latency_cycles", median(c)}, {"latency_s", cycles_to_s(median(c))}}; }
    out["access"]["l1"] = {{"latency_cycles", l1}, {"latency_s", cycles_to_s(l1)}, {"capacity_bytes_observed", pts[e1].first}};
    out["access"]["l2"] = {{"latency_cycles", lat2}, {"latency_s", cycles_to_s(lat2)}, {"capacity_bytes_observed", pts[e2].first}, {"capacity_bytes_reported", l2}};
    out["access"]["memory"] = {{"latency_cycles", mem}, {"latency_s", cycles_to_s(mem)}, {"footprint_bytes", pts.back().first}};
    out["access"]["sweep"] = sweep;
    out["access"]["method"] = "single-thread dependent loads over a full-period cycle; cycles from clock64, in-kernel";
  }

  void bandwidth() {
    size_t free_b = 0, total_b = 0; CK(cudaMemGetInfo(&free_b, &total_b));
    const size_t l2 = size_t(prop.l2CacheSize);
    const size_t big = std::min<size_t>(o.quick ? size_t(512) << 20 : size_t(2) << 30, free_b / 8) / 16;   // elements
    Device<Word16> x(big), y(big); Device<unsigned long long> sink(size_t(sms) * 8);
    CK(cudaMemsetAsync(x.p, 1, big * 16, st.s)); CK(cudaMemsetAsync(y.p, 1, big * 16, st.s));
    std::vector<int> grids;
    for (int g = 1; g < sms; g *= 2) grids.push_back(g);
    grids.push_back(sms); grids.push_back(2 * sms); grids.push_back(4 * sms);
    json scale = json::array(); double best_r = 0, best_w = 0, best_c = 0, one_r = 0, one_w = 0, one_c = 0;
    for (int g : grids) {
      // Small grids sweep a share of the buffer that still exceeds L2 several times.
      const size_t n = std::min(big, std::max<size_t>(8 * l2 / 16, big / std::max(1, sms / g / 4)));
      const double bytes = double(n / g * g) * 16;
      const double r = bytes / minimum(repeat([&]() { k_read<<<g, 1024, 0, st.s>>>(x.p, n, 1, sink.p); }, st.s, 3, 1));
      const double w = bytes / minimum(repeat([&]() { k_write<<<g, 1024, 0, st.s>>>(y.p, n, 1); }, st.s, 3, 1));
      const double c = 2 * bytes / minimum(repeat([&]() { k_copy<<<g, 1024, 0, st.s>>>(y.p, x.p, n, 1); }, st.s, 3, 1));
      scale.push_back({{"blocks", g}, {"read_bytes_per_s", r}, {"write_bytes_per_s", w}, {"copy_bytes_per_s", c}});
      if (g == 1) { one_r = r; one_w = w; one_c = c; }
      best_r = std::max(best_r, r); best_w = std::max(best_w, w); best_c = std::max(best_c, c);
    }
    auto saturation = [&](const char* key, double best) { for (auto& s : scale) if (s[key].get<double>() >= 0.9 * best) return s["blocks"].get<int>(); return sms; };
    out["bandwidth"]["memory"] = {{"read_bytes_per_s", best_r}, {"write_bytes_per_s", best_w}, {"copy_bytes_per_s", best_c},
      {"read_bytes_per_s_per_sm", one_r}, {"write_bytes_per_s_per_sm", one_w}, {"copy_bytes_per_s_per_sm", one_c},
      {"read_saturation_blocks", saturation("read_bytes_per_s", best_r)}, {"copy_saturation_blocks", saturation("copy_bytes_per_s", best_c)},
      {"scaling", scale}, {"buffer_bytes", big * 16},
      {"method", "16-byte loads and stores, 1024 threads per block, contiguous chunk per block; copy counts read plus write"}};
    // L2-resident: a quarter of the cache, swept repeatedly inside the kernel.
    { const size_t n = std::max<size_t>(1024, l2 / 4 / 16); const int passes = 64;
      k_read<<<sms, 1024, 0, st.s>>>(x.p, n, 1, sink.p);
      auto bw = [&](int g) { return double(n / g * g) * 16 * passes / minimum(repeat([&]() { k_read<<<g, 1024, 0, st.s>>>(x.p, n, passes, sink.p); }, st.s, 3, 1)); };
      out["bandwidth"]["l2"] = {{"read_bytes_per_s", bw(sms)}, {"read_bytes_per_s_per_sm", bw(1)}, {"footprint_bytes", n * 16}}; }
    // L1-resident and shared memory: one block on one SM.
    { const size_t n = 1024; const int passes = 4096;   // 16 KiB
      const double l1 = double(n) * 16 * passes / minimum(repeat([&]() { k_read<<<1, 1024, 0, st.s>>>(x.p, n, passes, sink.p); }, st.s, 3, 1));
      const double sh = 2048.0 * 16 * passes / minimum(repeat([&]() { k_read_shared<<<1, 1024, 0, st.s>>>(passes, sink.p); }, st.s, 3, 1));
      out["bandwidth"]["l1"] = {{"read_bytes_per_s_per_sm", l1}, {"footprint_bytes", n * 16}};
      out["bandwidth"]["shared"] = {{"read_bytes_per_s_per_sm", sh}, {"footprint_bytes", 2048 * 16}}; }
    // Device-to-device copy through the runtime.
    { std::vector<std::pair<double, double>> s;
      for (size_t bytes : {size_t(4) << 10, size_t(1) << 20, std::min(big * 16, size_t(256) << 20)})
        s.push_back({double(bytes), median(repeat([&]() { CK(cudaMemcpyAsync(y.p, x.p, bytes, cudaMemcpyDeviceToDevice, st.s)); }, st.s, 5))});
      out["bandwidth"]["device_copy"] = alpha_beta(s); }
  }

  void transfers() {
    const size_t big = size_t(256) << 20;
    Device<char> dev(big); char* host = nullptr; CK(cudaMallocHost(&host, big)); std::memset(host, 1, big);
    for (bool up : {true, false}) {
      std::vector<std::pair<double, double>> s;
      for (size_t bytes : {size_t(4) << 10, size_t(1) << 20, big})
        s.push_back({double(bytes), median(repeat([&]() {
          if (up) CK(cudaMemcpyAsync(dev.p, host, bytes, cudaMemcpyHostToDevice, st.s));
          else CK(cudaMemcpyAsync(host, dev.p, bytes, cudaMemcpyDeviceToHost, st.s)); }, st.s, 5))});
      out["bandwidth"][up ? "host_to_device" : "device_to_host"] = alpha_beta(s);
    }
    CK(cudaFreeHost(host));
    int count = 0; CK(cudaGetDeviceCount(&count));
    int peer = o.peer >= 0 ? o.peer : (count > 1 ? (o.device + 1) % count : -1);
    if (peer < 0 || peer == o.device) { out["bandwidth"]["peer"] = nullptr; return; }
    void* remote = nullptr;
    CK(cudaSetDevice(peer)); CK(cudaMalloc(&remote, big)); CK(cudaMemset(remote, 1, big)); CK(cudaSetDevice(o.device));
    int access = 0; CK(cudaDeviceCanAccessPeer(&access, o.device, peer));
    if (access) { cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0); if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) check(e, "cudaDeviceEnablePeerAccess"); }
    std::vector<std::pair<double, double>> s;
    for (size_t bytes : {size_t(4) << 10, size_t(1) << 20, big})
      s.push_back({double(bytes), median(repeat([&]() { CK(cudaMemcpyPeerAsync(remote, peer, dev.p, o.device, bytes, st.s)); }, st.s, 5))});
    out["bandwidth"]["peer"] = alpha_beta(s);
    out["bandwidth"]["peer"]["devices"] = {o.device, peer}; out["bandwidth"]["peer"]["direct_access"] = access != 0;
    CK(cudaSetDevice(peer)); CK(cudaFree(remote)); CK(cudaSetDevice(o.device));
  }

  // Vendor matrix products: the square product is the arithmetic capacity, the rank-k update (large output, short
  // contraction) and the inner product (small output, long contraction) show how the rate depends on the shape.
  struct Mode { const char* name; cudaDataType type; cublasComputeType_t compute; size_t word; };
  // Operations per second of one product C(m x n) = A B with A (m x k), B (k x n); with `inner`, C = A^T B with
  // A (k x m). Zero when the library does not support the arithmetic.
  double product(cublasHandle_t h, const Mode& md, int m, int n, int k, bool inner) {
    {
      const size_t rows_a = inner ? k : m, cols_a = inner ? m : k;
      Device<char> a(rows_a * cols_a * md.word), b(size_t(k) * n * md.word), c(size_t(m) * n * md.word);
      CK(cudaMemsetAsync(a.p, 0, a.n, st.s)); CK(cudaMemsetAsync(b.p, 0, b.n, st.s)); CK(cudaMemsetAsync(c.p, 0, c.n, st.s));
      const double one_d = 1, zero_d = 0; const float one_f = 1, zero_f = 0; const uint16_t one_h = 0x3c00, zero_h = 0;
      const void* one = md.word == 8 ? (const void*)&one_d : md.word == 4 ? (const void*)&one_f : (const void*)&one_h;
      const void* zero = md.word == 8 ? (const void*)&zero_d : md.word == 4 ? (const void*)&zero_f : (const void*)&zero_h;
      auto call = [&]() { return cublasGemmEx(h, inner ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, one, a.p, md.type, int(rows_a),
                                             b.p, md.type, k, zero, c.p, md.type, m, md.compute, CUBLAS_GEMM_DEFAULT); };
      if (call() != CUBLAS_STATUS_SUCCESS) return 0.0;
      CK(cudaStreamSynchronize(st.s));
      const double t = minimum(repeat([&]() { CK(call()); }, st.s, 3, 1));
      return 2.0 * m * n * double(k) / t;
    }
  }
  json gemm_mode(cublasHandle_t h, const Mode& md) {
    auto run = [&](int m, int n, int k, bool inner) { return product(h, md, m, n, k, inner); };
    const int n = o.quick ? 4096 : 8192;
    json j; const double square = run(n, n, n, false);
    if (square == 0) return nullptr;
    j["square"] = {{"n", n}, {"flops_per_s", square}};
    // 1/rate = gamma + delta/k: gamma is the time per operation of a compute-bound product and delta/k the traffic of
    // the output, which a short contraction cannot amortize.
    json rank = json::array(); double sx = 0, sy = 0, sxx = 0, sxy = 0; int cnt = 0, best_k = 0; double best = 0;
    for (int k : {32, 64, 128, 256, 512, 1024, 2048}) {
      const double r = run(n, n, k, false); if (r == 0) continue;
      rank.push_back({{"k", k}, {"flops_per_s", r}});
      const double x = 1.0 / k, y = 1.0 / r; sx += x; sy += y; sxx += x * x; sxy += x * y; ++cnt;
      if (r > best) { best = r; best_k = k; }
    }
    const double delta = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx), gamma = (sy - delta * sx) / cnt;
    j["rank_k"] = {{"m", n}, {"n", n}, {"samples", rank}, {"gamma_s_per_flop", gamma}, {"delta_s_per_flop_times_k", delta},
                   {"k_half", gamma > 0 ? delta / gamma : 0.0}, {"best_k", best_k}};
    json inner = json::array();
    for (int k : {128, 512, 1024}) {
      const int rows = o.quick ? 16384 : 32768, q = 4096;
      const double r = run(k, q, rows, true); if (r == 0) continue;
      inner.push_back({{"k", k}, {"q", q}, {"rows", rows}, {"flops_per_s", r}});
    }
    j["inner"] = inner;
    j["peak_flops_per_s"] = std::max(square, best); j["seconds_per_flop"] = 1.0 / std::max(square, best);
    // A product too small to occupy the device costs its call latency.
    { const double r = run(32, 32, 32, false); j["call_latency_s"] = r > 0 ? 2.0 * 32 * 32 * 32 / r : 0.0; }
    return j;
  }
  static constexpr Mode modes[] = {{"fp64", CUDA_R_64F, CUBLAS_COMPUTE_64F, 8}, {"fp32", CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, 4},
                                   {"tf32", CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4}, {"fp16", CUDA_R_16F, CUBLAS_COMPUTE_16F, 2}};
  // One product of a requested shape, for checking the product law of a profile on shapes the probe did not fit.
  json single(const std::string& arithmetic, int m, int n, int k) {
    CK(cudaSetDevice(o.device)); CK(cudaGetDeviceProperties(&prop, o.device));
    cublasHandle_t h = nullptr; CK(cublasCreate(&h)); CK(cublasSetStream(h, st.s));
    for (auto& md : modes)
      if (arithmetic == md.name) {
        const double rate = product(h, md, m, n, k, false);
        CK(cublasDestroy(h));
        return {{"arithmetic", arithmetic}, {"m", m}, {"n", n}, {"k", k}, {"flops_per_s", rate},
                {"seconds", rate > 0 ? 2.0 * m * n * double(k) / rate : 0.0}};
      }
    throw std::runtime_error("unknown arithmetic " + arithmetic);
  }
  void gemm() {
    cublasHandle_t h = nullptr; CK(cublasCreate(&h)); CK(cublasSetStream(h, st.s));
    json g;
    for (auto& md : modes) g[md.name] = gemm_mode(h, md);
    const char* tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
    g["tf32_override"] = tf32 ? tf32 : "unset";
    g["method"] = "cuBLAS GemmEx; minimum of three timed calls after a warm-up; fp32 is IEEE (pedantic), tf32 allows TF32 tensor operations";
    out["arithmetic"]["gemm"] = g;
    CK(cublasDestroy(h));
  }

  // Two kernels on two streams, each on half of the SMs, against one of them alone: 1 means they ran side by side.
  void concurrency() {
    Device<float> a(size_t(sms) * blocks_per_sm), b(size_t(sms) * blocks_per_sm); Stream other; Event fork, join;
    const long iters = 1L << 16; const int half = std::max(1, sms / 2), full = sms * blocks_per_sm;
    auto pair = [&](int blocks) {
      const double single = minimum(repeat([&]() { k_fma<float, 8><<<blocks, 1024, 0, st.s>>>(a.p, 1.0000001f, 1e-7f, iters, nullptr); }, st.s, 5));
      const double both = minimum(repeat([&]() {
        CK(cudaEventRecord(fork.e, st.s)); CK(cudaStreamWaitEvent(other.s, fork.e, 0));
        k_fma<float, 8><<<blocks, 1024, 0, st.s>>>(a.p, 1.0000001f, 1e-7f, iters, nullptr);
        k_fma<float, 8><<<blocks, 1024, 0, other.s>>>(b.p, 1.0000001f, 1e-7f, iters, nullptr);
        CK(cudaEventRecord(join.e, other.s)); CK(cudaStreamWaitEvent(st.s, join.e, 0)); }, st.s, 5));
      return both / single;
    };
    out["concurrency"] = {{"two_streams_half_sms_multiplier", pair(half)}, {"two_streams_all_sms_multiplier", pair(full)},
      {"method", "compute-bound kernels on two streams; elapsed time of the pair over one kernel alone. half: one block "
                 "per SM on half of the SMs each (1 = disjoint SMs run side by side); all: every warp slot taken by each"}};
  }

  json run() {
    const double start = now();
    CK(cudaSetDevice(o.device)); CK(cudaGetDeviceProperties(&prop, o.device));
    out["schema"] = "machine-profile/1"; out["probe"] = probe_version;
    device();
    if (o.identify) return {{"name", out["device"]["name"]}, {"uuid", out["device"]["uuid"]},
                            {"cuda_driver", out["device"]["cuda_driver"]}, {"probe", probe_version}};
    clock();
    out["arithmetic"]["simt"] = {{"fp32", simt<float>("fp32")}, {"fp64", simt<double>("fp64")}};
    latency(); access(); bandwidth(); transfers(); gemm(); concurrency();
    clock();   // again after load: a capped or throttled GPU shows here
    out["duration_s"] = now() - start;
    return out;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse(argc, argv);
    CK(cudaSetDevice(o.device));      // before the probe creates its stream
    Probe p; p.o = o;
    const json j = p.o.arithmetic.empty() ? p.run() : p.single(p.o.arithmetic, p.o.shape[0], p.o.shape[1], p.o.shape[2]);
    if (p.o.output.empty()) std::cout << j.dump(2) << '\n';
    else { std::ofstream f(p.o.output); if (!f) throw std::runtime_error("cannot write " + p.o.output); f << j.dump(2) << '\n'; }
    return 0;
  } catch (const std::exception& e) { std::cerr << "machine_probe: " << e.what() << '\n'; return 2; }
}
