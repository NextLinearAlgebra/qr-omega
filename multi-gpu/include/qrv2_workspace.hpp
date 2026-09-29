#pragma once
// Storage geometry of the register-resident panel backend.
#include "common.hpp"
namespace tqr {
// Host-only storage geometry, shared by admission and allocation. Reserve the
// QR-v2 backend even for the cooperative control so a replay's inventory does
// not depend on an environment switch. Counts below are elements, not bytes.
struct Qrv2Workspace {
 size_t count=0,flags=0,tau=0,overlaps=0,stack_v=0;
 size_t bytes(size_t word)const{
  if(word!=4&&word!=8)throw std::runtime_error("invalid_qrv2_workspace_precision");
  const size_t data=checked_add(checked_add(tau,overlaps),stack_v);
  return checked_add(checked_mul(flags,sizeof(int)),checked_mul(data,word));
 }
};
// Native reflector overlaps share one GPART arena across the whole batch.
// A GPU group writes h*h words PER packet; 128 arena slices are not 128
// groups for every packet. With one group launch_carrier_w writes straight
// to the final O, retaining its c>=2 block/cluster contraction split.
inline int qrv2_overlap_gpu_groups(size_t arena_words,int h,size_t count,int requested){
 if(h<1||count<1||requested<1)throw std::runtime_error("invalid_qrv2_overlap_shape");
 const size_t per_group=checked_mul(count,checked_mul(size_t(h),size_t(h)));
 return int(std::max<size_t>(1,std::min(size_t(requested),arena_words/per_group)));
}
inline Qrv2Workspace qrv2_workspace(int leaf,int b,size_t batch,int radix,int gradix,int apply_c){
 if(leaf<1||b<1||batch<1||radix<1||gradix<1||apply_c<1)
  throw std::runtime_error("invalid_qrv2_workspace_shape");
 Qrv2Workspace x;x.count=std::max(checked_mul(batch,size_t(radix)),size_t(gradix));
 x.flags=x.tau=checked_mul(x.count,size_t(b));
 x.stack_v=checked_mul(x.flags,size_t(b));
 x.overlaps=checked_mul(size_t(apply_c),x.stack_v);
 return x;
}
}
