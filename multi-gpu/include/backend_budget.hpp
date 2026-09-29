#pragma once
// Memory accounting of the communication backends (NCCL windows, NVSHMEM heap) in the per-GPU budget.
#include "common.hpp"
namespace tqr {
// Each NCCL VMM allocation is rounded independently. NVSHMEM's requested
// payloads are inside its reserved heap, not allocations to count twice.
inline size_t backend_arena_budget(size_t capacity,int ranks,const json&profile){
 if(ranks==1)return 0;capacity=std::max(capacity,size_t(64));
 size_t gran=profile.at("hardware").at("nccl_vmm_granularity");if(!gran)throw std::runtime_error("invalid_VMM_granularity");
 auto rounded=[&](size_t bytes){return checked_mul(checked_add(bytes,gran-1)/gran,gran);};
 return checked_add(size_t(64)*1024*1024,checked_add(rounded(checked_mul(capacity,size_t(ranks)*4)),checked_mul(2,rounded(capacity))));
}
}
