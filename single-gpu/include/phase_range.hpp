#pragma once
// Profiler ranges for the phases of a run (probe, warm-up, timed factorization, validation).
#include <string>
// C0 PROFILER PHASE RANGES. Launch counts reconcile against device
// observations per invocation and phase (probe, warmup factorization, timed
// factorization, validation Q application, teardown) without inferring
// boundaries from incidental validation kernels. NVTX ranges are
// host-side markers: they add no device work, no synchronization, and no
// numerical change. Header-only (nvtx3); when the header is unavailable the
// guard compiles to nothing.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
namespace tqr {
struct PhaseRange {
  explicit PhaseRange(const char* name){nvtxRangePushA(name);}
  explicit PhaseRange(const std::string& name){nvtxRangePushA(name.c_str());}
  ~PhaseRange(){nvtxRangePop();}
  PhaseRange(const PhaseRange&)=delete;PhaseRange& operator=(const PhaseRange&)=delete;
};
}  // namespace tqr
#else
namespace tqr {
struct PhaseRange {
  explicit PhaseRange(const char*){}
  explicit PhaseRange(const std::string&){}
};
}  // namespace tqr
#endif
