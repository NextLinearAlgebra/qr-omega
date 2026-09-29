#pragma once
// Shared-memory capacity checks per kernel specialization, used to admit carriers.
#include "tiled_engine.cuh"
namespace tqr {
// The capacity that rules a carrier in or out is therefore the capacity of THE KERNEL THAT WILL
// RUN: the opt-in shared memory per block minus that kernel's own static shared storage.
//
// The selector used one number for every cooperative specialization -- the plain tiled_ge<T,true>
// limit -- while each launcher subtracts its own static allocation. At fp64 rows=55296 g128 PW8
// sw64 (need 228736 B) the selector admitted 230128 B and the launcher 228544 B, so a stamped
// carrier was refused at launch and the panel silently ran the block path with a false priced
// certificate.
//
// occupancy_at_max_dynamic is cudaOccupancyMaxActiveBlocksPerMultiprocessor at the kernel's full
// dynamic capacity: occupancy only falls as dynamic storage grows, so >=1 there proves >=1 at every
// admitted footprint, which is the launcher's residency rule for every group count <= the SM count.
template<class T> json kernel_resources(int device){
 cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
 auto one=[&](const void*fn){
  cudaFuncAttributes a;CU(cudaFuncGetAttributes(&a,fn));
  const size_t optin=prop.sharedMemPerBlockOptin;
  const size_t max_dynamic=optin>a.sharedSizeBytes?optin-a.sharedSizeBytes:0;
  CU(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,int(max_dynamic)));
  json occ=json::object();
  for(int t:{128,256,512,1024}){if(t>a.maxThreadsPerBlock)continue;int o=0;
   CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o,fn,t,max_dynamic));occ[std::to_string(t)]=o;}
  return json{{"static_shared",size_t(a.sharedSizeBytes)},{"max_dynamic",max_dynamic},{"regs",a.numRegs},
   {"max_threads",a.maxThreadsPerBlock},{"local_bytes",size_t(a.localSizeBytes)},{"occupancy_at_max_dynamic",occ}};};
 json k=json::object();
 k["coop_ge"]=one((const void*)cooperative_ge<T>);
 k["coop_mini_pw8"]=one((const void*)cooperative_ge_mini<T,8>);
 k["coop_mini_pw16"]=one((const void*)cooperative_ge_mini<T,16>);
 k["coop_mini_pw32"]=one((const void*)cooperative_ge_mini<T,32>);
 k["tiled_ge_shared"]=one((const void*)tiled_ge<T,true>);
 return json{{"schema",1},{"precision",sizeof(T)==4?"fp32":"fp64"},{"sm_count",prop.multiProcessorCount},
  {"shared_optin_per_block",size_t(prop.sharedMemPerBlockOptin)},{"shared_per_sm",size_t(prop.sharedMemPerMultiprocessor)},
  {"source","cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor on this binary; no timing"},
  {"kernels",k}};
}
// The selector's view of the table.
struct CoopResources {
 bool present=false;int sms=0;
 size_t cap[4]={0,0,0,0};              // pw index: 0->coop_ge, 1->pw8, 2->pw16, 3->pw32
 int occ256[4]={0,0,0,0},occ512[4]={0,0,0,0};
 static int index_of(int pw){return pw==0?0:pw==8?1:pw==16?2:pw==32?3:-1;}
 static CoopResources from(const json&profile){
  CoopResources r;if(!profile.contains("kernel_resources"))return r;
  const json&t=profile.at("kernel_resources");const json&k=t.at("kernels");
  const char*names[4]={"coop_ge","coop_mini_pw8","coop_mini_pw16","coop_mini_pw32"};
  for(int i=0;i<4;++i){const json&e=k.at(names[i]);r.cap[i]=e.at("max_dynamic").get<size_t>();
   const json&o=e.at("occupancy_at_max_dynamic");r.occ256[i]=o.value("256",0);r.occ512[i]=o.value("512",0);}
  r.sms=t.at("sm_count").get<int>();r.present=true;return r;}
 // Exactly the launchers' rules (cooperative_packet.cuh launch_cooperative_ge,
 // cooperative_minipanel.cuh launch_cooperative_ge_mini_pw): dynamic storage
 // within this kernel's opt-in capacity, and the grid co-resident.
 bool fits(int pw,int groups,int threads,size_t need,size_t legacy)const{
  if(!present)return need<=legacy;
  const int i=index_of(pw);if(i<0)return false;
  if(need>cap[i])return false;
  const int occ=threads==256?occ256[i]:threads==512?occ512[i]:0;
  return occ>=1&&size_t(groups)<=size_t(occ)*size_t(sms);}
 // The largest capacity among the specializations: a certificate stating that
 // the cheapest candidate's need exceeds it proves that no candidate fits.
 size_t max_capacity(size_t legacy)const{
  if(!present)return legacy;size_t m=0;for(size_t c:cap)m=std::max(m,c);return m;}
};
}
