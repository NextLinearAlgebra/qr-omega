#pragma once
// Device-side timestamps and counters that record the ordering and overlap of issued work.
#include "common.hpp"
#include "kernels.cuh"
namespace tqr {
// The carrier witness counts what the HOST ISSUED.
//
// Each packet is bracketed, on its own stream, by two one-thread kernels. Stream order alone
// makes them sound: the open cannot run before the previous work on that slot has retired, and the
// close cannot run before the packet's own products have completed. Each records %globaltimer, a
// device-wide nanosecond clock that is comparable across CTAs and across streams, so the intervals
// of two slots can be compared directly. The close also carries the packet's per-kind product
// counts, which gives a device-side completion count to reconcile against the host's issued count.
//
// Cost: ONE extra launch per packet (the close is folded into the existing per-strip history commit
// wherever a factoring pass already issues one), plus two per panel factor.
struct ExecInterval {
 unsigned long long start_ns,end_ns;   // %globaltimer at open and close
 unsigned int slot,kind;               // pipeline slot; kind below
 unsigned int elimination,packet;      // which elimination, which packet of it
 unsigned int products[execution_product_kinds];             // per CarrierWitness::Kind, this packet
 unsigned int live_at_open,live_at_close;
};
// kind
enum ExecKind{EXEC_PACKET=0,EXEC_PANEL_FACTOR=1,EXEC_GLOBAL_PACKET=2,EXEC_TT_FACTOR=3,EXEC_KINDS};
__device__ inline unsigned long long exec_now(){
 unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ inline void exec_open(ExecInterval*log,unsigned cap,unsigned idx,
  unsigned slot,unsigned kind,unsigned elimination,unsigned packet,Witness*w){
 if(threadIdx.x||blockIdx.x)return;
 const unsigned long long t=exec_now();
 const unsigned long long live=atomicAdd(&w->credits_live,1ULL)+1ULL;
 atomicMax(&w->credits_max,live);
 if(idx<cap){ExecInterval&e=log[idx];e.start_ns=t;e.end_ns=0;e.slot=slot;e.kind=kind;
  e.elimination=elimination;e.packet=packet;e.live_at_open=(unsigned)live;e.live_at_close=0;
  for(int k=0;k<execution_product_kinds;++k)e.products[k]=0;}
 else atomicAdd(&w->evidence_overflow,1ULL);
}
// Closing releases the credit and certifies completion: this kernel is ordered
// after the packet's own products on the same stream, so the counts it adds
// were produced by work the device had finished.
__global__ inline void exec_close(ExecInterval*log,unsigned cap,unsigned idx,Witness*w,
  unsigned p0,unsigned p1,unsigned p2,unsigned p3,unsigned p4,unsigned p5,unsigned p6){
 if(threadIdx.x||blockIdx.x)return;
 const unsigned long long t=exec_now();
 const unsigned long long live=atomicAdd(&w->credits_live,~0ULL);   // -1
 const unsigned c[execution_product_kinds]={p0,p1,p2,p3,p4,p5,p6};
 for(int k=0;k<execution_product_kinds;++k)atomicAdd(&w->device_products[k],(unsigned long long)c[k]);
 if(idx<cap){ExecInterval&e=log[idx];e.end_ns=t;e.live_at_close=(unsigned)live;
  for(int k=0;k<execution_product_kinds;++k)e.products[k]=c[k];}
}
// HOST REDUCTION.
struct ExecSummary {
 uint64_t sampled=0,closed=0,overflow=0;
 uint64_t busy_ns=0,union_ns=0,overlap_ns=0;
 uint64_t packet_overlap_ns=0;      // between packets of ONE elimination on different slots
 uint64_t max_concurrent=0;         // from the interval sweep, independent of credits_max
 std::map<unsigned,unsigned> per_elimination_max;
 uint64_t slot_reuse_violations=0;
 bool have_release=false; long long next_panel_release_ns=0;
 uint64_t release_panels=0,released_early=0;
 json release_record()const{
  return {{"median_gap_ns",next_panel_release_ns},{"panels",release_panels},{"panels_released_early",released_early},
   {"fraction_early",release_panels?double(released_early)/double(release_panels):0.0},
   {"definition","panel i is early when some EXEC_PACKET interval of an elimination < i ends after panel i's factor starts (Alg 1: panel k+1 starts from ready tiles while earlier trailing continues); median_gap_ns: start minus the latest end of the intervals issued since the previous panel started"}};}
 json record()const{
  json j={{"sampled_intervals",sampled},{"closed_intervals",closed},{"dropped_intervals",overflow},
   {"busy_ns",busy_ns},{"union_ns",union_ns},{"overlap_ns",overlap_ns},
   {"within_elimination_overlap_ns",packet_overlap_ns},
   {"max_concurrent_intervals",max_concurrent},
   {"per_elimination_max_concurrent",per_elimination_max},
   {"slot_reuse_violations",slot_reuse_violations},
   {"clock","device %globaltimer, nanoseconds, comparable across CTAs and streams WITHIN one device only; no cross-device correlation"},
   {"scope","one interval per issued packet and per panel factor, opened and closed by one-thread kernels on the packet's own stream; a dropped interval is counted, never estimated"}};
  j["next_panel_release_ns"]=have_release?json(next_panel_release_ns):json(nullptr);
  return j;}
};
inline ExecSummary exec_reduce(const std::vector<ExecInterval>&log,size_t used,uint64_t overflow){
 ExecSummary s;s.overflow=overflow;
 std::vector<const ExecInterval*> iv;
 for(size_t i=0;i<used&&i<log.size();++i){const ExecInterval&e=log[i];
  if(!e.start_ns)continue;++s.sampled;if(!e.end_ns)continue;++s.closed;
  if(e.end_ns>e.start_ns)iv.push_back(&e);}
 if(iv.empty())return s;
 std::sort(iv.begin(),iv.end(),[](const ExecInterval*a,const ExecInterval*b){return a->start_ns<b->start_ns;});
 // Sum, union and the running concurrency, in one sweep over the sorted starts.
 unsigned long long cur_start=iv[0]->start_ns,cur_end=iv[0]->end_ns;
 for(const ExecInterval*e:iv){
  s.busy_ns+=e->end_ns-e->start_ns;
  if(e->start_ns>cur_end){s.union_ns+=cur_end-cur_start;cur_start=e->start_ns;cur_end=e->end_ns;}
  else cur_end=std::max(cur_end,e->end_ns);}
 s.union_ns+=cur_end-cur_start;
 s.overlap_ns=s.busy_ns>s.union_ns?s.busy_ns-s.union_ns:0;
 // Concurrency by sweeping endpoints.
 {std::vector<std::pair<uint64_t,int>>ev;ev.reserve(iv.size()*2);
  for(const ExecInterval*e:iv){ev.push_back({e->start_ns,1});ev.push_back({e->end_ns,-1});}
  std::sort(ev.begin(),ev.end());long long live=0;
  for(auto&p:ev){live+=p.second;s.max_concurrent=std::max<uint64_t>(s.max_concurrent,(uint64_t)std::max(0LL,live));}}
  // Only packets of the SAME elimination on DIFFERENT slots count, so cross-elimination concurrency
  // (a different move, {Pipeline} across panels) cannot be read as depth.
  {std::map<unsigned,std::vector<const ExecInterval*>>by;
   for(const ExecInterval*e:iv)if(e->kind==EXEC_PACKET)by[e->elimination].push_back(e);
   for(auto&kv:by){auto&v=kv.second;if(v.size()<2)continue;
    unsigned long long sum=0,un=0;std::sort(v.begin(),v.end(),[](const ExecInterval*a,const ExecInterval*b){return a->start_ns<b->start_ns;});
    unsigned long long cs=v[0]->start_ns,ce=v[0]->end_ns;bool multi=false;
    for(const ExecInterval*e:v){sum+=e->end_ns-e->start_ns;if(e->slot!=v[0]->slot)multi=true;
     if(e->start_ns>ce){un+=ce-cs;cs=e->start_ns;ce=e->end_ns;}else ce=std::max(ce,e->end_ns);}
    un+=ce-cs;if(multi&&sum>un)s.packet_overlap_ns+=sum-un;}
   // Per-elimination concurrency (all kinds, endpoints sweep per elim).
   for(auto&kv:by){auto&v=kv.second;
    std::vector<std::pair<uint64_t,int>>ev;ev.reserve(v.size()*2);
    for(const ExecInterval*e:v){ev.push_back({e->start_ns,1});ev.push_back({e->end_ns,-1});}
    std::sort(ev.begin(),ev.end());long long live=0;unsigned mx=0;
    for(auto&p:ev){live+=p.second;mx=std::max<unsigned>(mx,(unsigned)std::max(0LL,live));}
    s.per_elimination_max[kv.first]=mx;}
   // Same-slot overlap among PACKET intervals only (panel/global/TT factor
   // intervals share slot 0 across phases by construction): the close of a
   // packet interval releases its slot credit, so a second packet interval
   // on the same slot overlapping it means the slot was re-entered before
   // release -- a protocol fault, and the fault-injection suite proves the
   // gate fails it.
   {std::map<unsigned,std::vector<const ExecInterval*>>byslot;
    for(const ExecInterval*e:iv)if(e->kind==EXEC_PACKET)byslot[e->slot].push_back(e);
    for(auto&kv:byslot){auto&v=kv.second;std::sort(v.begin(),v.end(),
      [](const ExecInterval*a,const ExecInterval*b){return a->start_ns<b->start_ns;});
     for(size_t i=1;i<v.size();++i)
      if(v[i]->start_ns<v[i-1]->end_ns)s.slot_reuse_violations++;}}}
 // Alg 1's last line: how long after the previous panel's trailing ended did
 // the next panel's factor start. Negative means it started BEFORE, which is
 // the cross-panel pipeline actually happening.
 {std::vector<const ExecInterval*>panels;
  for(const ExecInterval*e:iv)if(e->kind==EXEC_PANEL_FACTOR)panels.push_back(e);
  if(panels.size()>1){
   std::sort(panels.begin(),panels.end(),[](const ExecInterval*a,const ExecInterval*b){return a->start_ns<b->start_ns;});
   std::vector<long long>gaps;
   for(size_t i=1;i<panels.size();++i){unsigned long long prev_end=0;
    for(const ExecInterval*e:iv)if(e->kind!=EXEC_PANEL_FACTOR&&e->start_ns>=panels[i-1]->start_ns&&e->start_ns<panels[i]->start_ns)
     prev_end=std::max(prev_end,e->end_ns);
    if(prev_end)gaps.push_back((long long)panels[i]->start_ns-(long long)prev_end);}
   if(!gaps.empty()){std::sort(gaps.begin(),gaps.end());s.have_release=true;s.next_panel_release_ns=gaps[gaps.size()/2];}
   for(size_t i=1;i<panels.size();++i){++s.release_panels;const ExecInterval*p=panels[i];
    for(const ExecInterval*e:iv)if(e->kind==EXEC_PACKET&&e->elimination<p->elimination&&e->end_ns>p->start_ns){++s.released_early;break;}}}}
 return s;
}
}
