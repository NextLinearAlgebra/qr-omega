#pragma once
// Fused per-panel local TT tree.
//
// The elimination list and its ordered QR combines
// are unchanged: the same merges run in the same level order with the same expression identities.
// What changes is below the moves, at the machine-model level: one cooperative launch
// carries a whole panel's local tree instead of one launch per level plus separate gather, scatter,
// children-check and history-begin launches. Each merge keeps the block-local reduction tree and
// the HBM-as-global discipline; levels are separated by grid barriers. No operand is replicated
// (c=1), no pipeline depth is added (d=1): this is launch fusion, not a new carrier.
//
// The factor arithmetic is adapted from tiled_ge (kind=2 path) in tiled_kernels.cuh in this
// repository: same Householder algebra, same column scaling and R rescaling, same T recurrence,
// same status codes. No code is copied from the qr-omega domino panel or from KAMI; the minipanel,
// warp-specialization and tensor-core techniques studied there are not used here. Differences from
// tiled_ge are compilation details only: single block per merge, fixed 256 threads, stack passed
// explicitly.
//
// Ordering (same as local_level in tiled_engine.cuh): children lineage check (reads only), gather
// stack, factor, scatter R parts, publish history entry. Publication uses the same CAS as
// tile_history_begin; kernel completion orders all writes before any later same-stream consumer
// observes hist.
#include "tiled_kernels.cuh"
#include <cooperative_groups.h>

namespace tqr {

// The device forms prefix offsets.
struct TTLevels {
  int n = 0;
  int cnt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

// Compact serial Householder factorization of one TT stack (S=ht+hb rows,
// h columns, leading dimension 2b), mirroring tiled_ge's kind=2 path:
// positive column prescaling, reflector loop with block reductions, T factor
// recurrence, R rescaling. tri is b*b; red/scales/g are block-shared scratch
// (32/128/128). Witness counting is left to the caller.
template<class T> __device__ __forceinline__ void tt_stack_factor(
    T* st, int S, int h, T* tri, int b, T* red, T* scales, T* g, T* tau,
    int* status) {
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int warps = blockDim.x / 32, ld = 2 * b;
  for (int i = tid; i < b * b; i += blockDim.x) tri[i] = 0;
  for (int j = warp; j < h; j += warps) {
    T mx = 0;
    for (int r = lane; r < S; r += 32) mx = max(mx, absval(st[r + size_t(j) * ld]));
    mx = warp_maximum(mx);
    if (lane == 0) scales[j] = mx;
    for (int r = lane; r < S; r += 32) if (mx) st[r + size_t(j) * ld] /= mx;
  }
  __syncthreads();
  for (int k = 0; k < h; ++k) {
    T tail = 0;
    for (int r = k + 1 + tid; r < S; r += blockDim.x) tail = max(tail, absval(st[r + size_t(k) * ld]));
    tail = tile_reduce<T, true>(tail, red);
    const T alpha = st[k + size_t(k) * ld], ns = max(tail, absval(alpha));
    T bet = alpha, tu = 0, dn = 1;
    if (tail) {
      T sum = 0;
      for (int r = k + tid; r < S; r += blockDim.x) { T x = st[r + size_t(k) * ld] / ns; sum += x * x; }
      sum = tile_reduce<T, false>(sum, red);
      const T an = alpha / ns;
      T bn = -sqrt(sum);
      if (an < 0) bn = -bn;
      bet = ns * bn; dn = an - bn; tu = 1 - an / bn;
    }
    if (tid == 0) { *tau = tu; if (tail) { st[k + size_t(k) * ld] = bet; tri[k + size_t(k) * b] = tu; } }
    if (tail) for (int r = k + 1 + tid; r < S; r += blockDim.x) st[r + size_t(k) * ld] = (st[r + size_t(k) * ld] / ns) / dn;
    __syncthreads();
    for (int j = warp; j < k; j += warps) {
      T dot = lane == 0 ? st[k + size_t(j) * ld] : T(0);
      for (int r = k + 1 + lane; r < S; r += 32) dot += st[r + size_t(j) * ld] * st[r + size_t(k) * ld];
      dot = warp_add(dot);
      if (lane == 0) g[j] = dot;
    }
    __syncthreads();
    if (tid < k) { T sum = 0; for (int j = tid; j < k; ++j) sum += tri[tid + size_t(j) * b] * g[j]; tri[tid + size_t(k) * b] = -(*tau) * sum; }
    for (int j = k + 1 + warp; j < h; j += warps) {
      T dot = lane == 0 ? st[k + size_t(j) * ld] : T(0);
      for (int r = k + 1 + lane; r < S; r += 32) dot += st[r + size_t(k) * ld] * st[r + size_t(j) * ld];
      dot = warp_add(dot) * (*tau);
      if (lane == 0) st[k + size_t(j) * ld] -= dot;
      for (int r = k + 1 + lane; r < S; r += 32) st[r + size_t(j) * ld] -= st[r + size_t(k) * ld] * dot;
    }
    __syncthreads();
  }
  for (int i = tid; i < h * h; i += blockDim.x) {
    int r = i % h, j = i / h;
    if (r <= j) { T y = st[r + size_t(j) * ld] * scales[j]; st[r + size_t(j) * ld] = y; if (!isfinite(y)) atomicCAS(status, 0, UNREPRESENTABLE_RESULT); }
  }
  __syncthreads();
}

// One cooperative launch for a panel's whole local TT tree. merges holds all levels contiguously;
// loff[l] is the first merge index of level l (loff[nlevels] = total). stack holds maxc slots of
// 2b*b words. hist entries hist_n is the expected children value; hist_next is the published value
// (col+h). Every block reaches every grid.sync (no early returns); slots without a merge in a level
// stay idle but synchronized.
template<class T> __global__ void tiled_tt_panel(
    T* A, int ld, int begin, const TileMerge* lbase, TTLevels lv,
    int maxc, T* stack, T* tri, int b, bool use_shared,
    uint64_t* hist, int hist_n, int hist_next, int* status, Witness* w) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  const int s = blockIdx.x, tid = threadIdx.x;
  __shared__ T red[32], scales[128], g[128], tau;
  extern __shared__ __align__(16) unsigned char smem[];
  const size_t slot_elems = size_t(2) * b * b;
  // Single-level launches need no cross-block sync: blocks own disjoint
  // merges, slots, tri/history entries and A regions (tree property), and
  // later kernels observe via stream ordering. Multi-level keeps grid syncs.
  const bool need_sync = (lv.n > 1);
  int off = 0;
  for (int l = 0; l < lv.n; ++l) {
    const int cnt = lv.cnt[l];
    const bool active = s < cnt;
    // Lineage check (reads only), same predicate as tile_history_children.
    // Scoped copy: the descriptor must not stay live across the factor loop
    // (register budget: 1024 threads x regs must fit 64K/SM like tiled_ge).
    if (active && tid == 0) {
      const TileMerge ec = lbase[off + s];
      if (hist[ec.top_child()] != uint64_t(hist_n) || hist[ec.bottom_child()] != uint64_t(hist_n))
        atomicCAS(status, 0, HISTORY_ERROR);
    }
    if (need_sync) grid.sync();
    // Gather stack with the tile_stack mapping, into block-shared staging
    // when selected (same data, same layout/leading dimension either way).
    // Only three scalars cross into factor; the struct dies at brace end.
    T* st_slot = use_shared ? reinterpret_cast<T*>(smem) : stack + size_t(s) * 2 * b * b;
    int Sh = 0, Hh = 0, Etile = 0;
    if (active) {
      const TileMerge e = lbase[off + s];
      for (int i = tid; i < 2 * b * b; i += blockDim.x) {
        int r = i % (2 * b), j = i / (2 * b);
        T v = 0;
        if (j < e.h) {
          if (r < e.ht() && r <= j) v = A[e.top() - begin + r + size_t(e.col + j) * ld];
          else if (r >= e.ht() && r < e.ht() + e.hb() && r - e.ht() <= j) v = A[e.bottom() - begin + r - e.ht() + size_t(e.col + j) * ld];
        }
        st_slot[i] = v;
      }
      Sh = e.rows();
      Hh = e.h;
      Etile = e.tile;
    }
    if (need_sync) grid.sync(); else __syncthreads();
    // Factor this slot's stack.
    if (active) tt_stack_factor(st_slot, Sh, Hh, tri + size_t(Etile) * b * b, b, red, scales, g, &tau, status);
    if (need_sync) grid.sync(); else __syncthreads();
    // Scatter R parts + publish history + witness, same mapping as
    // tile_scatter_factors and tile_history_begin. Fresh reload (read-only,
    // L1-resident) keeps the descriptor out of the factor live range.
    if (active) {
      const TileMerge e = lbase[off + s];
      const T* in = st_slot;
      for (int i = tid; i < e.h * e.h; i += blockDim.x) {
        int r = i % e.h, j = i / e.h;
        if (r <= j) {
          A[e.top() - begin + r + size_t(e.col + j) * ld] = in[r + size_t(j) * 2 * b];
          if (r < e.hb()) A[e.bottom() - begin + r + size_t(e.col + j) * ld] = in[e.ht() + r + size_t(j) * 2 * b];
        }
      }
      if (tid == 0) {
        if (atomicCAS((unsigned long long*)(hist + e.tile), 0ULL, (unsigned long long)hist_next) != 0ULL)
          atomicCAS(status, 0, HISTORY_ERROR);
        atomicAdd(&w->tt, 1ULL);
        atomicAdd(&w->reflectors, (unsigned long long)e.h);
      }
    }
    if (need_sync) grid.sync();
    off += cnt;
  }
}

template<class T> void launch_tt_panel(
    T* A, int ld, int begin, const TileMerge* lbase, TTLevels lv,
    int maxc, T* stack, T* tri, int b, bool use_shared, int threads,
    uint64_t* hist, int hist_n, int hist_next, int* status, Witness* w,
    cudaStream_t stream) {
  if (lv.n < 1 || lv.n > 8 || maxc < 1 || b < 1 || b > 128) throw std::runtime_error("tt_panel_descriptor_before_modify");
  if (threads != 256 && threads != 512 && threads != 1024) throw std::runtime_error("tt_panel_threads_before_modify");
  size_t dynamic = use_shared ? size_t(2) * b * b * sizeof(T) : 0;
  // Driver queries + func-attribute programming are synchronous host-side calls (~10s of us each):
  // cache per (dynamic, threads) instead of paying per level.
  struct Entry { bool used = false; size_t dynamic = 0; int threads = 0; int occupancy = 0; int smCount = 0; bool coop = false; };
  static Entry entries[12];
  Entry* found = nullptr;
  for (auto& e : entries) if (e.used && e.dynamic == dynamic && e.threads == threads) { found = &e; break; }
  if (!found) {
    for (auto& e : entries) if (!e.used) { found = &e; break; }
    if (!found) throw std::runtime_error("tt_panel_cache_exhausted_before_modify");
    int device;
    CU(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CU(cudaGetDeviceProperties(&prop, device));
    cudaFuncAttributes attr;
    CU(cudaFuncGetAttributes(&attr, tiled_tt_panel<T>));
    if (dynamic + attr.sharedSizeBytes > size_t(prop.sharedMemPerBlockOptin))
      throw std::runtime_error("tt_panel_shared_capacity_before_modify");
    if (dynamic) CU(cudaFuncSetAttribute(tiled_tt_panel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(dynamic + attr.sharedSizeBytes)));
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&found->occupancy, tiled_tt_panel<T>, threads, dynamic));
    found->dynamic = dynamic;
    found->threads = threads;
    found->smCount = prop.multiProcessorCount;
    found->coop = prop.cooperativeLaunch;
    found->used = true;
  }
  int occupancy = found->occupancy;
  if (occupancy < 1) throw std::runtime_error("tt_panel_occupancy_before_modify");
  if (lv.n > 1) {
    // Multi-level needs grid syncs: cooperative launch with residency check.
    if (!found->coop) throw std::runtime_error("tt_panel_cooperative_unsupported");
    if (size_t(maxc) > size_t(occupancy) * found->smCount)
      throw std::runtime_error("tt_panel_grid_residency_before_modify");
    void* args[] = {&A, &ld, &begin, &lbase, &lv, &maxc, &stack, &tri, &b, &use_shared, &hist, &hist_n, &hist_next, &status, &w};
    CU(cudaLaunchCooperativeKernel((const void*)tiled_tt_panel<T>, dim3(maxc), dim3(threads), args, dynamic, stream));
  } else {
    // Single level: blocks are fully independent (disjoint merges, slots,
    // tri/history entries and A regions), later kernels observe via stream
    // ordering, so an ordinary launch suffices -- no coop requirement.
    tiled_tt_panel<T><<<dim3(maxc), dim3(threads), dynamic, stream>>>(
        A, ld, begin, lbase, lv, maxc, stack, tri, b, use_shared,
        hist, hist_n, hist_next, status, w);
    CU(cudaGetLastError());
  }
}

}  // namespace tqr
