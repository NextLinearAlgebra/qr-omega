#pragma once
// Distributed matrix storage and the execution context shared by the engine.
#include "plan.hpp"
#include "transport.cuh"
#include <memory>
namespace tqr {
template<class T> struct Matrix {
 int m,n,p,rank,begin,nr,ld;Buffer<T> a;
 Matrix(int m_,int n_,int p_,int rank_,int pad=0):m(descriptor_dimension(m_)),n(descriptor_dimension(n_)),p(descriptor_parts(p_)),rank(descriptor_dimension(rank_)),begin(rank<p?row_begin(m,p,rank):m),nr(rank<p?row_begin(m,p,rank+1)-begin:0),ld(descriptor_ld(nr,pad)),a(checked_mul(size_t(ld),size_t(n))){}
 Matrix(int m_,int n_,int p_,int rank_,T* data,int stride,size_t extent):m(descriptor_dimension(m_)),n(descriptor_dimension(n_)),p(descriptor_parts(p_)),rank(descriptor_dimension(rank_)),begin(rank<p?row_begin(m,p,rank):m),nr(rank<p?row_begin(m,p,rank+1)-begin:0),ld(stride),a(data,extent){
  if(m<0||n<0||p<1||rank<0||ld<std::max(1,nr)||(n&&nr&&(!data||extent<checked_add(checked_mul(size_t(n-1),size_t(ld)),nr)))||(uintptr_t(data)%alignof(T)))throw std::runtime_error("matrix_view_descriptor_before_modify");
 }
};
template<class T> struct Factor {
 FactorEvent event;Buffer<T> v,tri;Buffer<int> rows;
 Factor(const FactorEvent& e,int rank,cudaStream_t stream):event(e),v(rank==e.owner?e.rows.size()*e.h:0),tri(rank==e.owner?e.h*e.h:0),rows(e.rows.size()){rows.upload(e.rows,stream);}
};
template<class T> struct Slot {
 Stream stream;Event begun,done;bool timed=false;int event_id=0,column=0;
 Buffer<T> v,x,partial,w,z,scales,tri;int maxs,maxh,q,c;
 Slot(int s,int h,int q_,int c_):v(size_t(s)*h*c_),x(size_t(s)*std::max(h,q_)*c_),partial(size_t(h)*q_*c_),w(h*q_),z(h*q_),scales(q_),tri(h*h),maxs(s),maxh(h),q(q_),c(c_){}
};
template<class T> class Engine {
 Context& ctx;Plan plan;bool trace;size_t physical_delta=0;Stream main;std::unique_ptr<Transport> transport;
 Buffer<T> panel,packing;Buffer<int> status;Buffer<Witness> witness;
 Buffer<uint64_t> hist,expected;std::vector<uint64_t> previous;
 std::vector<Factor<T>> factors;std::vector<std::unique_ptr<Slot<T>>> slots;
 int maxs=1,maxh=1;size_t reserved=0;uint64_t invocation=0,expression=0;Event origin;json intervals=json::array(),lineage=json::array();
 void retire(Slot<T>& slot){slot.stream.sync();if(slot.timed){float a,b;CU(cudaEventElapsedTime(&a,origin.e,slot.begun.e));CU(cudaEventElapsedTime(&b,origin.e,slot.done.e));if(trace)intervals.push_back({{"event",slot.event_id},{"column",slot.column},{"begin_s",a*1e-3},{"end_s",b*1e-3}});slot.timed=false;}}
 Buffer<T> scalar_tau;const T* scalar_vectors=nullptr;int scalar_ld=0;bool has_factor=false;
 std::unique_ptr<Matrix<T>> active_matrix;
 void check_status(){main.sync();int x=status.download()[0];if(ctx.max_status(x))throw std::runtime_error("device_status_"+std::to_string(ctx.max_status(x)));}
 std::vector<std::pair<int,int>> runs(const FactorEvent&e,int rank)const {
  std::vector<std::pair<int,int>> r;for(int i=0;i<int(e.rows.size());){if(owner_of(e.rows[i],plan.m,plan.p)!=rank){++i;continue;}int j=i+1;while(j<int(e.rows.size())&&owner_of(e.rows[j],plan.m,plan.p)==rank)++j;r.push_back({i,j-i});i=j;}return r;
 }
 void gather(const Factor<T>& f,const Matrix<T>& a,int col,int q,T* output){
  const auto&e=f.event;int s=e.rows.size();
  for(int rank=0;rank<ctx.size;++rank)for(auto [pos,nr]:runs(e,rank)){
   if(ctx.rank==rank){gather_rows<<<ceildiv(nr*q,128),128,0,main>>>(a.a.p,a.ld,a.begin,f.rows.p+pos,nr,col,q,packing.p,nr);main.sync();}
   transport->publish(rank,e.owner,packing.p,panel.p,size_t(nr)*q*sizeof(T),++expression,1);
   if(ctx.rank==e.owner){CU(cudaMemcpy2DAsync(output+pos,size_t(s)*sizeof(T),panel.p,size_t(nr)*sizeof(T),size_t(nr)*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
  }
 }
 void scatter(const Factor<T>& f,Matrix<T>& a,int col,int q,const T* input){
  const auto&e=f.event;int s=e.rows.size();
  for(int rank=0;rank<ctx.size;++rank)for(auto [pos,nr]:runs(e,rank)){
   if(ctx.rank==e.owner){CU(cudaMemcpy2DAsync(packing.p,size_t(nr)*sizeof(T),input+pos,size_t(s)*sizeof(T),size_t(nr)*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
   transport->publish(e.owner,rank,packing.p,panel.p,size_t(nr)*q*sizeof(T),++expression,2);
   if(ctx.rank==rank){scatter_rows<<<ceildiv(nr*q,128),128,0,main>>>(panel.p,nr,0,a.a.p,a.ld,a.begin,f.rows.p+pos,nr,col,q);main.sync();}
  }
 }
 void advance_history(const Factor<T>& f,const Matrix<T>& a,int col,int q,uint64_t next,cudaStream_t stream){
  for(auto [pos,nr]:runs(f.event,ctx.rank))versions<<<ceildiv(nr*q,128),128,0,stream>>>(hist.p,a.nr,a.begin,f.rows.p,expected.p,pos,nr,col,q,next,status.p);
 }
 void apply_strip(const Factor<T>& f,Matrix<T>& a,int col,int q,bool transpose,uint64_t next,Slot<T>& slot){
  const auto&e=f.event;int s=e.rows.size(),h=e.h,layers=std::min(plan.c,s);auto st=slot.stream.s;
  if(ctx.size==1){
   if(trace)slot.begun.record(st);slot.timed=trace;slot.event_id=e.id;slot.column=col;
   gather_rows<<<ceildiv(s*q,128),128,0,st>>>(a.a.p,a.ld,a.begin,f.rows.p,s,col,q,slot.x.p,s);
   // Full immutable copies per logical contraction layer are deliberately
   // explicit. Their total bytes and repeated setup are charged.
   for(int c=0;c<layers;++c){CU(cudaMemcpyAsync(slot.v.p+size_t(c)*s*h,f.v.p,s*h*sizeof(T),cudaMemcpyDeviceToDevice,st));
    if(c)CU(cudaMemcpyAsync(slot.x.p+size_t(c)*s*q,slot.x.p,s*q*sizeof(T),cudaMemcpyDeviceToDevice,st));}
   record_replica<<<1,1,0,st>>>(witness.p,size_t(layers)*s*h*sizeof(T)+size_t(layers-1)*s*q*sizeof(T));
   column_scales<<<q,plan.threads,0,st>>>(slot.x.p,s,q,slot.scales.p);
   for(int c=0;c<layers;++c)w_partial<<<ceildiv(h*q,128),128,0,st>>>(slot.v.p+size_t(c)*s*h,slot.x.p+size_t(c)*s*q,slot.scales.p,s,h,q,s*c/layers,s*(c+1)/layers,slot.partial.p+size_t(c)*h*q,witness.p);
   sum_partials<<<ceildiv(h*q,128),128,0,st>>>(slot.partial.p,slot.w.p,h*q,layers,witness.p);
   triangular_apply<<<ceildiv(h*q,128),128,0,st>>>(f.tri.p,slot.w.p,slot.z.p,h,q,transpose);
   commit_update<<<ceildiv(s*q,128),128,0,st>>>(f.v.p,slot.x.p,slot.z.p,slot.scales.p,s,h,q,a.a.p,a.ld,a.begin,f.rows.p,0,s,col,status.p,witness.p);
   advance_history(f,a,col,q,next,st);slot.done.record(st);return;
  }
  // Serialized hybrid episodes: gather the exact old support, then freeze it.
  gather(f,a,col,q,slot.x.p);
  if(layers==1){
   if(ctx.rank==e.owner){column_scales<<<q,plan.threads,0,st>>>(slot.x.p,s,q,slot.scales.p);
    w_partial<<<ceildiv(h*q,128),128,0,st>>>(f.v.p,slot.x.p,slot.scales.p,s,h,q,0,s,slot.w.p,witness.p);
    triangular_apply<<<ceildiv(h*q,128),128,0,st>>>(f.tri.p,slot.w.p,slot.z.p,h,q,transpose);
   }
   // Packed update has no global-row map; use the dedicated packed kernel.
   if(ctx.rank==e.owner){packed_update(f.v.p,slot, s,h,q,st);slot.stream.sync();}
   scatter(f,a,col,q,slot.x.p);advance_history(f,a,col,q,next,main);main.sync();return;
  }
  layers=std::min(layers,plan.p);
  uint64_t expr=++expression;
  transport->template paper<T,false>(e.owner,f.v.p,slot.v.p,s*h,expr);
  transport->template paper<T,false>(e.owner,f.tri.p,slot.tri.p,h*h,expr);
  transport->template paper<T,false>(e.owner,slot.x.p,slot.x.p,s*q,expr);
  // These drains reproduce exactly the completion the transport used to perform inside every
  // episode; the layered member is the reference representation, not the performance path, so it
  // keeps the simple form.
  transport->drain();
  if(ctx.rank<plan.p)column_scales<<<q,plan.threads,0,st>>>(slot.x.p,s,q,slot.scales.p);
  int lo=ctx.rank<layers?s*ctx.rank/layers:0,hi=ctx.rank<layers?s*(ctx.rank+1)/layers:0;
  if(ctx.rank<plan.p)w_partial<<<ceildiv(h*q,128),128,0,st>>>(slot.v.p,slot.x.p,slot.scales.p,s,h,q,lo,hi,slot.partial.p,witness.p);slot.stream.sync();
  transport->template paper<T,true>(e.owner,slot.partial.p,slot.w.p,h*q,expr);transport->drain();
  if(ctx.rank==e.owner){triangular_apply<<<ceildiv(h*q,128),128,0,st>>>(slot.tri.p,slot.w.p,slot.z.p,h,q,transpose);slot.stream.sync();}
  transport->template paper<T,false>(e.owner,slot.z.p,slot.z.p,h*q,expr);transport->drain();
  for(auto [pos,nr]:runs(e,ctx.rank))commit_update<<<ceildiv(nr*q,128),128,0,st>>>(slot.v.p,slot.x.p,slot.z.p,slot.scales.p,s,h,q,a.a.p,a.ld,a.begin,f.rows.p,pos,nr,col,status.p,witness.p);
  advance_history(f,a,col,q,next,st);slot.stream.sync();ctx.barrier();
 }
 void packed_update(const T* v,Slot<T>& slot,int s,int h,int q,cudaStream_t st){packed_commit<<<ceildiv(s*q,128),128,0,st>>>(v,slot.z.p,slot.scales.p,slot.x.p,s,h,q,status.p,witness.p);}
 void begin_event(const Factor<T>&f){std::vector<uint64_t> v;for(int r:f.event.rows)v.push_back(previous[r]);expected.upload(v,main);main.sync();}
 void end_event(const Factor<T>&f,uint64_t next){for(auto&slot:slots)retire(*slot);main.sync();
  // Canonical history follows completed owner commits, separately from inbox retirement.
  for(int rank=0;rank<plan.p;++rank)if(rank!=f.event.owner&&!runs(f.event,rank).empty())transport->publish(rank,f.event.owner,nullptr,nullptr,0,++expression,4);
  for(int r:f.event.rows)previous[r]=next;CU(cudaGetLastError());}
public:
 Engine(Context& c,Plan p,int ncols,size_t budget,bool trace_=true):ctx(c),plan(std::move(p)),trace(trace_),status(1),witness(1),previous(plan.tree=="scalar_inplace"?0:plan.m,0){
  size_t free_before,total_memory;CU(cudaMemGetInfo(&free_before,&total_memory));
  if(plan.p>ctx.size)throw std::runtime_error("active_subset_out_of_range");
  if(plan.tree=="scalar_inplace"){reserved=size_t(std::min(plan.m,plan.n))*sizeof(T)+sizeof(int)+sizeof(Witness)+4096;
   if(reserved+size_t(plan.m)*plan.n*sizeof(T)>budget)throw std::runtime_error("insufficient_storage_before_modify");scalar_tau.alloc(std::min(plan.m,plan.n));return;}
  for(const auto&e:plan.events){maxs=std::max(maxs,int(e.rows.size()));maxh=std::max(maxh,e.h);}
  size_t maxpayload=checked_mul(size_t(maxs)*std::max({maxh,plan.strip,1}),sizeof(T));
  // Runtime checks exact explicit allocation demand before factorization/A writes.
  reserved=sizeof(int)+sizeof(Witness)+size_t(plan.m)*sizeof(uint64_t);
  int nr=ctx.rank<plan.p?row_begin(plan.m,plan.p,ctx.rank+1)-row_begin(plan.m,plan.p,ctx.rank):0;
  reserved+=size_t(nr)*ncols*sizeof(uint64_t)+size_t(maxs)*(sizeof(uint64_t)+sizeof(int));
  reserved+=2*maxpayload;
  for(const auto&e:plan.events){reserved+=e.rows.size()*sizeof(int);if(e.owner==ctx.rank)reserved+=(e.rows.size()*e.h+e.h*e.h)*sizeof(T);}
  reserved+=size_t(plan.d)*(size_t(plan.c)*(maxs*maxh+maxs*std::max(maxh,plan.strip)+maxh*plan.strip)+2*maxh*plan.strip+plan.strip+maxh*maxh)*sizeof(T);
  if(ctx.size>1)reserved+=maxpayload*(4*ctx.size+4)+65536;
  if(plan.p<ctx.size)reserved+=size_t(std::max(1,nr))*ncols*sizeof(T);
  if(reserved+size_t(nr)*ncols*sizeof(T)>budget)throw std::runtime_error("insufficient_storage_before_modify");
  hist.alloc(size_t(nr)*ncols);expected.alloc(maxs);panel.alloc(maxpayload/sizeof(T));packing.alloc(maxpayload/sizeof(T));
  for(const auto&e:plan.events)factors.emplace_back(e,ctx.rank,main);
  for(int i=0;i<plan.d;++i)slots.push_back(std::make_unique<Slot<T>>(maxs,maxh,plan.strip,plan.c));
  // Bound to this engine's stream: the transport is stream-ordered now, so the
  // staging copies above and the publication below have to be in ONE queue.
  // The surrounding main.sync() calls become redundant rather than load-bearing.
  main.sync();transport=std::make_unique<Transport>(ctx,maxpayload,plan.p);transport->bind(main.s);
  if(plan.p<ctx.size)active_matrix=std::make_unique<Matrix<T>>(plan.m,ncols,plan.p,ctx.rank);
  size_t free_after;CU(cudaMemGetInfo(&free_after,&total_memory));physical_delta=free_before>free_after?free_before-free_after:0;
  size_t public_A=plan.scope.contains("allocation_lifetimes")?plan.scope["allocation_lifetimes"][ctx.rank]["caller_A"].template get<size_t>():size_t(plan.m)*plan.n*sizeof(T);
  if(checked_add(physical_delta,public_A)>budget)throw std::runtime_error("physical_storage_before_modify");
 }
 int factor_internal(Matrix<T>&a){
  if(a.m!=plan.m||a.n!=plan.n||a.p!=plan.p)throw std::runtime_error("factor_descriptor_before_modify");has_factor=false;intervals=json::array();lineage=json::array();origin.record(main);main.sync();
  ++invocation;status.zero(main);witness.zero(main);hist.zero(main);std::fill(previous.begin(),previous.end(),0);
  if(a.nr&&a.n)finite_scan<<<128,128,0,main>>>(a.a.p,a.nr,a.n,a.ld,status.p);main.sync();
  int initial=ctx.max_status(status.download()[0]);if(initial)return initial;
  if(plan.tree=="scalar_inplace"){scalar_inplace<<<1,plan.threads,0,main>>>(a.a.p,a.m,a.n,a.ld,scalar_tau.p,status.p,witness.p);main.sync();scalar_vectors=a.a.p;scalar_ld=a.ld;has_factor=true;return status.download()[0];}
  for(auto&f:factors){const auto&e=f.event;int s=e.rows.size();begin_event(f);
   // gather uses panel as transport inbox scratch; factor input must be distinct.
   gather(f,a,e.col,e.h,slots[0]->x.p);
   if(ctx.rank==e.owner){native_hh<<<1,plan.threads,0,main>>>(slots[0]->x.p,s,e.h,f.v.p,f.tri.p,e.kind,status.p,witness.p);main.sync();if(trace)lineage.push_back({{"event",e.id},{"kind",e.kind},{"children",e.children},{"reflectors",e.h}});}
   scatter(f,a,e.col,e.h,slots[0]->x.p);advance_history(f,a,e.col,e.h,e.id+1,main);main.sync();
   int packet=0;for(int col=e.col+e.h;col<a.n;col+=plan.strip){auto& slot=*slots[packet++%plan.d];retire(slot);apply_strip(f,a,col,std::min(plan.strip,a.n-col),true,e.id+1,slot);}
   end_event(f,e.id+1);
  }
  main.sync();ctx.barrier();has_factor=true;return ctx.max_status(status.download()[0]);
 }
 int apply_q_internal(Matrix<T>&a,bool transpose){
  if(!has_factor)throw std::runtime_error("Q_handle_not_ready");
  if(plan.tree=="scalar_inplace"){if(a.m!=plan.m||a.a.p==scalar_vectors)throw std::runtime_error("Q_apply_descriptor_or_alias");status.zero(main);if(a.n)scalar_apply<<<a.n,plan.threads,0,main>>>(scalar_vectors,scalar_ld,scalar_tau.p,a.m,std::min(plan.m,plan.n),a.a.p,a.n,a.ld,transpose,status.p);main.sync();return status.download()[0];}
  if(a.m!=plan.m||a.n*size_t(a.nr)>hist.n)throw std::runtime_error("Q_apply_workspace_descriptor");
  ++invocation;status.zero(main);hist.zero(main);std::fill(previous.begin(),previous.end(),0);main.sync();
  for(int step=0;step<int(factors.size());++step){int index=transpose?step:int(factors.size())-1-step;auto&f=factors[index];begin_event(f);
   int packet=0;for(int col=0;col<a.n;col+=plan.strip){auto& slot=*slots[packet++%plan.d];retire(slot);apply_strip(f,a,col,std::min(plan.strip,a.n-col),transpose,step+1,slot);}end_event(f,step+1);
  }main.sync();ctx.barrier();return ctx.max_status(status.download()[0]);
 }
 json record(){auto v=witness.download()[0];return {{"rank",ctx.rank},{"strip_intervals_same_device",intervals},{"completed_factor_lineage",lineage},{"executed_GE",v.ge},{"executed_TS",v.ts},{"executed_TT",v.tt},{"reflectors",v.reflectors},{"partial_kernels",v.partials},{"join_kernels",v.joins},{"commit_kernels",v.commits},{"replica_bytes_software_count",v.replica_bytes},{"explicit_workspace_bytes",reserved},{"runtime_device_allocation_delta",physical_delta},{"backend",transport?transport->evidence:json{{"standalone",true}}},{"histories",plan.tree=="scalar_inplace"?"one finite block packet, strict internal program order":"device CAS against complete per-row predecessor; entry histories for every committed column"}};}
 const Plan& selected_plan()const{return plan;}
#ifdef TQR_MULTI
 ncclComm_t reference_comm()const{return transport->reference_comm();}
#endif
 // Reference-only conversion, charged in its timing boundary. Public balanced
 // row slabs <-> P-by-1 block-cyclic storage, GPU bytes via the same NV path.
 void reference_cyclic(Matrix<T>& pub,T* cyclic,int ld,int block,bool to_cyclic){
  for(int row=0;row<pub.m;){int slab=owner_of(row,pub.m,ctx.size),owner=(row/block)%ctx.size;
   int nr=std::min({maxs,pub.m-row,row_begin(pub.m,ctx.size,slab+1)-row,block-row%block});
   int lr=(row/(block*ctx.size))*block+row%block;
   for(int col=0;col<pub.n;col+=plan.strip){int q=std::min(plan.strip,pub.n-col),src=to_cyclic?slab:owner,dst=to_cyclic?owner:slab;
    if(ctx.rank==src){const T* from=to_cyclic?pub.a.p+(row-pub.begin)+size_t(col)*pub.ld:cyclic+lr+size_t(col)*ld;int stride=to_cyclic?pub.ld:ld;CU(cudaMemcpy2DAsync(packing.p,nr*sizeof(T),from,stride*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
    transport->publish(src,dst,packing.p,panel.p,size_t(nr)*q*sizeof(T),++expression,10);
    if(ctx.rank==dst){T* to=to_cyclic?cyclic+lr+size_t(col)*ld:pub.a.p+(row-pub.begin)+size_t(col)*pub.ld;int stride=to_cyclic?ld:pub.ld;CU(cudaMemcpy2DAsync(to,stride*sizeof(T),panel.p,nr*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
   }row+=nr;
  }
 }
 void redistribute(const Matrix<T>&source,Matrix<T>&dest){
  if(source.m!=dest.m||source.n!=dest.n)throw std::runtime_error("redistribution_descriptor");
  for(int src=0;src<source.p;++src)for(int dst=0;dst<dest.p;++dst){
   int first=std::max(row_begin(source.m,source.p,src),row_begin(dest.m,dest.p,dst)),last=std::min(row_begin(source.m,source.p,src+1),row_begin(dest.m,dest.p,dst+1));
   for(int row=first;row<last;row+=maxs){int nr=std::min(maxs,last-row);for(int col=0;col<source.n;col+=plan.strip){int q=std::min(plan.strip,source.n-col);
    if(ctx.rank==src){CU(cudaMemcpy2DAsync(packing.p,nr*sizeof(T),source.a.p+(row-source.begin)+size_t(col)*source.ld,source.ld*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
    transport->publish(src,dst,packing.p,panel.p,size_t(nr)*q*sizeof(T),++expression,9);
    if(ctx.rank==dst){CU(cudaMemcpy2DAsync(dest.a.p+(row-dest.begin)+size_t(col)*dest.ld,dest.ld*sizeof(T),panel.p,nr*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
   }}
  }
 }
 int factor(Matrix<T>&a){if(!active_matrix)return factor_internal(a);active_matrix->n=a.n;redistribute(a,*active_matrix);int s=factor_internal(*active_matrix);if(!s)redistribute(*active_matrix,a);return s;}
 int apply_q(Matrix<T>&a,bool transpose){if(!active_matrix)return apply_q_internal(a,transpose);active_matrix->n=a.n;redistribute(a,*active_matrix);int s=apply_q_internal(*active_matrix,transpose);if(!s)redistribute(*active_matrix,a);return s;}
 Buffer<T> gather_full(const Matrix<T>& a){
  Buffer<T> result(ctx.rank==0?size_t(a.m)*a.n:0);
  if(ctx.size==1){if(a.m&&a.n)CU(cudaMemcpy2D(result.p,a.m*sizeof(T),a.a.p,a.ld*sizeof(T),a.m*sizeof(T),a.n,cudaMemcpyDeviceToDevice));return result;}
  for(int rank=0;rank<ctx.size;++rank){int first=row_begin(a.m,ctx.size,rank),last=row_begin(a.m,ctx.size,rank+1);
   for(int row=first;row<last;row+=maxs){int nr=std::min(maxs,last-row);
    for(int col=0;col<a.n;col+=plan.strip){int q=std::min(plan.strip,a.n-col);
     if(ctx.rank==rank){CU(cudaMemcpy2DAsync(packing.p,nr*sizeof(T),a.a.p+(row-first)+size_t(col)*a.ld,a.ld*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
     transport->publish(rank,0,packing.p,panel.p,size_t(nr)*q*sizeof(T),++expression,3);
     if(ctx.rank==0){CU(cudaMemcpy2DAsync(result.p+row+size_t(col)*a.m,a.m*sizeof(T),panel.p,nr*sizeof(T),nr*sizeof(T),q,cudaMemcpyDeviceToDevice,main));main.sync();}
    }
   }
  }return result;
 }
};
}
