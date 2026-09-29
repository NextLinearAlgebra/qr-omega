#pragma once
#include "tiled_engine.cuh"
#include "kernel_resources.cuh"
#include "protocol.cuh"
#include <unordered_map>
#include <set>
#include "packet_service.hpp"
namespace tqr {
template<class T> json tiled_probe(Context&ctx,const json&hardware){
 json p=probe<T>(ctx,hardware);p["catalog"]=tiled_catalog;p["execution_format"]=execution_format_version;p["tiled_samples"]=json::array();p["tiled_probe_workspace_limit_bytes"]=size_t(2)*1024*1024*1024;
 const size_t limit=p["tiled_probe_workspace_limit_bytes"];
 // Eq pipe's pace needs d DISJOINT X strips (see the pace loop), which is d
 // times the operand the other samples need, so it carries its own allowance.
 // Recorded in the profile so a reader can see which limit shaped which sample.
 p["tiled_pace_workspace_limit_bytes"]=size_t(12)*1024*1024*1024;
 const size_t pace_limit=p["tiled_pace_workspace_limit_bytes"];Stream st;cublasHandle_t bh;tiled_blas_check(cublasCreate(&bh));tiled_blas_check(cublasSetStream(bh,st));tiled_blas_check(cublasSetMathMode(bh,CUBLAS_PEDANTIC_MATH));tiled_blas_check(cublasSetAtomicsMode(bh,CUBLAS_ATOMICS_NOT_ALLOWED));
 // Pipeline slots for the PACE measurement below. The formula stays the paper's; t_max becomes an
 // observation instead of an assumption. Bind the menu ONCE: pipeline_depth_menu() returns by
 // value, so calling it twice for begin() and end() gives iterators into two different temporaries.
 const std::vector<int> pipe_depths=pipeline_depth_menu();
 const int pipe_max_depth=*std::max_element(pipe_depths.begin(),pipe_depths.end());
 std::vector<Stream> pipe_st(pipe_max_depth);std::vector<cublasHandle_t> pipe_bh(pipe_max_depth);
 for(int i=0;i<pipe_max_depth;++i){tiled_blas_check(cublasCreate(&pipe_bh[i]));tiled_blas_check(cublasSetStream(pipe_bh[i],pipe_st[i].s));
  tiled_blas_check(cublasSetMathMode(pipe_bh[i],CUBLAS_PEDANTIC_MATH));tiled_blas_check(cublasSetAtomicsMode(pipe_bh[i],CUBLAS_ATOMICS_NOT_ALLOWED));}
 Event pipe_fork;std::vector<Event> pipe_join(pipe_max_depth);

 Buffer<unsigned char>blas_workspace(tiled_blas_workspace_bytes);tiled_blas_check(cublasSetWorkspace(bh,blas_workspace.p,blas_workspace.n));p["blas_workspace_bytes"]=blas_workspace.n;
 Buffer<int>status(1);Buffer<Witness>witness(1);status.zero(st);witness.zero(st);
 auto save=[&](std::string op,int rows,int h,int q,int batches,int threads,const std::vector<double>&v){auto sorted=v;std::sort(sorted.begin(),sorted.end());p["tiled_samples"].push_back({{"op",op},{"rows",rows},{"h",h},{"q",q},{"batch",batches},{"threads",threads},{"raw_s",v},{"median_s",median(v)},{"p10_s",sorted[sorted.size()/10]},{"p90_s",sorted[9*sorted.size()/10]},{"class","measurement; no uniform duration bounds"}});};
 if(ctx.size==1){
  Buffer<T>a(128*128),orig(a.n),tau(128);int*host=nullptr;CU(cudaMallocHost(&host,sizeof(int)));std::vector<double>copy;
  for(int i=0;i<15;++i){double start=seconds();CU(cudaMemcpyAsync(host,status.p,sizeof(int),cudaMemcpyDeviceToHost,st));st.sync();copy.push_back(seconds()-start);}p["scalar_status_host_s"]=median(copy);p["scalar_status_host_raw_s"]=copy;CU(cudaFreeHost(host));
  for(int m:{1,2,4,8,16,32,64,128})for(int n:{1,2,4,8,16,32,64,128})for(int threads:{1,32,128,256}){
   make_input<<<128,128,0,st>>>(orig.p,m,n,m,0,m,0,T(1));st.sync();std::vector<double>raw;Event begin,end;
   for(int i=0;i<12;++i){CU(cudaMemcpyAsync(a.p,orig.p,size_t(m)*n*sizeof(T),cudaMemcpyDeviceToDevice,st));begin.record(st);scalar_inplace<T,true><<<1,threads,0,st>>>(a.p,m,n,m,tau.p,status.p,witness.p);end.record(st);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,begin.e,end.e));if(i>=3)raw.push_back(ms*1e-3);}save("scalar_fused",m,std::min(m,n),n,1,threads,raw);
  }p["scalar_packet_catalog"]={{"maximum_rows",128},{"maximum_columns",128},{"threads",{1,32,128,256}},{"empty","identity-Q degeneration without a numerical launch"}};
  cudaDeviceProp prop;cudaFuncAttributes attr;CU(cudaGetDeviceProperties(&prop,ctx.device));CU(cudaFuncGetAttributes(&attr,small_shared_packet<T>));int capacity=prop.sharedMemPerBlockOptin-attr.sharedSizeBytes;
  CU(cudaFuncSetAttribute(small_shared_packet<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,capacity));p["small_packet_shared_capacity_bytes"]=capacity;p["small_packet_resources"]=json::array();
  for(int m:{1,2,4,8,16,32,64,128})for(int n:{1,2,4,8,16,32,64,128})for(int threads:{32,128,256}){
   size_t dynamic=size_t(m)*n*sizeof(T);if(dynamic>size_t(capacity))continue;
   make_input<<<128,128,0,st>>>(orig.p,m,n,m,0,m,0,T(1));st.sync();std::vector<double>raw;Event begin,end;
   for(int i=0;i<12;++i){CU(cudaMemcpyAsync(a.p,orig.p,size_t(m)*n*sizeof(T),cudaMemcpyDeviceToDevice,st));begin.record(st);small_shared_packet<<<1,threads,dynamic,st>>>(a.p,m,n,m,tau.p,status.p,witness.p);end.record(st);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,begin.e,end.e));if(i>=3)raw.push_back(ms*1e-3);}save("small_shared",m,std::min(m,n),n,1,threads,raw);
   int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,small_shared_packet<T>,threads,dynamic));p["small_packet_resources"].push_back({{"m",m},{"n",n},{"threads",threads},{"dynamic_shared_bytes",dynamic},{"static_shared_bytes",attr.sharedSizeBytes},{"registers_per_thread",attr.numRegs},{"local_bytes_per_thread",attr.localSizeBytes},{"active_blocks_per_SM",occupancy}});
  }
 }
 for(int rows:{32,128,1024,4096,16384})for(int q:{16,128,512,4096}){
  Buffer<T>x(size_t(rows)*q),scale(q),other(q);make_input<<<256,128,0,st>>>(x.p,rows,q,rows,0,rows,0,T(1));other.zero(st);st.sync();
  save("column_max",rows,1,q,1,256,time_primitive([&](){tiled_column_max<<<q,256,0,st>>>(x.p,rows,q,rows,scale.p);},st,9));
  save("column_normalize",rows,1,q,1,256,time_primitive([&](){tiled_column_scale<T,false><<<256,256,0,st>>>(x.p,rows,q,rows,0,scale.p,status.p);},st,9));
  save("upper_restore",rows,1,q,1,256,time_primitive([&](){tiled_column_scale<T,true,true><<<256,256,0,st>>>(x.p,rows,q,rows,0,scale.p,status.p);},st,9));
  save("scale_max",q,1,1,1,128,time_primitive([&](){tiled_scale_max<<<ceildiv(q,128),128,0,st>>>(scale.p,other.p,q);},st,9));
 }
 p["global_panel_resources"]=json::array();
 for(int h:{16,32,64,128})for(int rows:{32,64,128,256,512,1024,4096,16384})for(int count:{1,8,64,256}){
  if(rows<h)continue;
  size_t matrix=size_t(rows)*count*h;if((2*matrix+size_t(count)*h*h)*sizeof(T)>limit)continue;
  Buffer<T>a(matrix),orig(matrix),tri(size_t(count)*h*h);Buffer<TilePacket>packets(count);std::vector<TilePacket>desc;
  for(int i=0;i<count;++i)desc.push_back({i*rows,0,rows,h,i});packets.upload(desc,st);make_input<<<256,128,0,st>>>(orig.p,rows*count,h,rows*count,0,rows*count,0,T(1));st.sync();
  for(int threads:{128,256,512,1024}){std::vector<double> times;Event start,end;for(int i=0;i<12;++i){CU(cudaMemcpyAsync(a.p,orig.p,matrix*sizeof(T),cudaMemcpyDeviceToDevice,st));start.record(st);tiled_ge<<<count,threads,0,st>>>(a.p,rows*count,packets.p,count,tri.p,h,0,status.p,witness.p);end.record(st);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,start.e,end.e));if(i>=3)times.push_back(ms*1e-3);}save("GE",rows,h,1,count,threads,times);
   int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,tiled_ge<T>,threads,0));p["global_panel_resources"].push_back({{"kind",0},{"rows",rows},{"h",h},{"threads",threads},{"active_blocks_per_SM",occupancy}});}
 }
 for(int h:{16,32,64,128})for(int radix:{2,4,8})for(int count:{1,8,64,256}){
  int rows=radix*h;if((2*size_t(rows)*h*count+size_t(count)*h*h)*sizeof(T)>limit)continue;
  Buffer<T>a(size_t(rows)*h*count),orig(a.n),tri(size_t(count)*h*h);Buffer<TilePacket>packets(count);std::vector<TilePacket>desc;for(int i=0;i<count;++i)desc.push_back({0,i*h,rows,h,i});packets.upload(desc,st);
  for(int i=0;i<count;++i)structural_input<<<32,128,0,st>>>(orig.p+size_t(i)*rows*h,rows,h,2);st.sync();
  for(int threads:{128,256,512,1024}){std::vector<double> times;Event start,end;for(int i=0;i<12;++i){CU(cudaMemcpyAsync(a.p,orig.p,a.n*sizeof(T),cudaMemcpyDeviceToDevice,st));start.record(st);tiled_ge<<<count,threads,0,st>>>(a.p,rows,packets.p,count,tri.p,h,2,status.p,witness.p);end.record(st);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,start.e,end.e));if(i>=3)times.push_back(ms*1e-3);}save("TT",rows,h,1,count,threads,times);
   int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,tiled_ge<T>,threads,0));p["global_panel_resources"].push_back({{"kind",2},{"rows",rows},{"h",h},{"threads",threads},{"active_blocks_per_SM",occupancy}});}
 }
 // One packet spans the whole panel height, so the local tree disappears entirely -- that is the
 // only regime where this carrier pays. Sampled only where the engine can dispatch it (count==1,
 // rows>=4096); launch_cooperative_ge refuses on residency or shared capacity and those points are
 // simply absent from the profile, which makes them unselectable rather than mispriced. Rows below
 // 1024 are sampled now. The engine's dispatch floor used to be the constant coop_min_rows=1024,
 // and the carrier was never SAMPLED below it, so the floor could not be checked: "the block path
 // wins under 1024" was an assumption with no number attached, and the merge stacks (K*b = 256 rows
 // at b=128) fell through it silently. Rows beyond 16384 are sampled because the panel-tall leaf
 // makes them reachable: the coop_GE interpolation carries NO rows term (the kernel is
 // barrier-dominated and nearly flat in rows up to 16384 -- 2.72 ms at rows=128 against 3.10 at
 // 16384 for g128/p0/h128 fp64), so beyond the sampled range the estimate would simply stop
 // growing.
 for(int h:{16,32,64,128})for(int rows:{128,256,512,1024,2048,4096,8192,16384,32768,65536}){
  if(rows<h)continue;size_t matrix=size_t(rows)*h;
  if((2*matrix+size_t(h)*h)*sizeof(T)>limit)continue;
  // Third copy of the group rule, now the engine's own (8e: a constant written beside a parameter
  // outlives it). Sample the WHOLE carrier menu: contraction replication c (groups) x minipanel
  // width (0 = unblocked) x threads.
  size_t slots=0;for(int g:coop_group_menu())for(int pw:coop_width_menu())
   slots=std::max(slots,pw?coop_mini_total_slots(g,h,pw):size_t(2)*g*(2+2*h));
  Buffer<T>a(matrix),orig(matrix),tri(size_t(h)*h),partial(slots);
  Buffer<TilePacket>packets(1);packets.upload({{0,0,rows,h,0}},st);
  make_input<<<256,128,0,st>>>(orig.p,rows,h,rows,0,rows,0,T(1));st.sync();
  for(int groups:coop_group_menu())for(int pw:coop_width_menu())for(int threads:{256,512}){
   if(pw>h)continue;
   std::vector<double>times;Event start,end;bool ok=true;
   for(int i=0;i<12&&ok;++i){
    CU(cudaMemcpyAsync(a.p,orig.p,matrix*sizeof(T),cudaMemcpyDeviceToDevice,st));
    start.record(st);
    // A refusal is a real answer (the launcher enforces residency and shared
    // capacity), but it is LOGGED rather than swallowed -- section 21.
    try{if(pw)launch_cooperative_ge_mini(a.p,rows,packets.p,rows,h,0,h,pw,groups,threads,tri.p,h,partial.p,status.p,witness.p,st);
        else launch_cooperative_ge(a.p,rows,packets.p,1,rows,h,groups,threads,tri.p,h,partial.p,status.p,witness.p,st);}
    catch(const std::exception&e){if(!i)fprintf(stderr,"coop_probe_refused rows=%d h=%d g=%d pw=%d t=%d: %s\n",rows,h,groups,pw,threads,e.what());ok=false;break;}
    end.record(st);CU(cudaEventSynchronize(end.e));
    float ms;CU(cudaEventElapsedTime(&ms,start.e,end.e));if(i>=3)times.push_back(ms*1e-3);
   }
   // NOTE: deliberately NOT recorded in global_panel_resources. That array is
   // consumed by PacketServices, which reads active_blocks_per_SM from every
   // entry and maps kind!=0 to "TT"; a carrier entry there both breaks the
   // read and pollutes the TT residency table. The carrier's residency is
   // enforced at launch by launch_cooperative_ge, and a sample existing in
   // tiled_samples is itself proof that the launch was accepted.
   if(ok&&times.size()>=9)save(coop_op(groups,pw),rows,h,1,1,threads,times);
  }
  // THE STAGED COLUMN WINDOW, SAMPLED. Without this the selector can only be TOLD a window by
  // --csw; it can never choose one, which is why the carrier is lost at exactly the shapes where
  // capacity refuses the unwindowed form. What is timed is the whole window SEQUENCE the engine
  // issues -- ceil(h/sw) minipanel launches -- so the sample is the thing that runs.
  for(int sw:coop_window_menu()){
   if(sw>=h)continue;
   for(int groups:coop_group_menu())for(int pw:coop_width_menu())for(int threads:{256,512}){
    if(!pw||pw>sw)continue;            // the window only exists for the minipanel kernel
    std::vector<double>times;Event start,end;bool ok=true;
    for(int i=0;i<12&&ok;++i){
     CU(cudaMemcpyAsync(a.p,orig.p,matrix*sizeof(T),cudaMemcpyDeviceToDevice,st));
     start.record(st);
     try{for(int w0=0;w0<h;w0+=sw)
          launch_cooperative_ge_mini(a.p,rows,packets.p,rows,h,w0,std::min(sw,h-w0),pw,groups,threads,tri.p,h,partial.p,status.p,witness.p,st);}
     catch(const std::exception&e){if(!i)fprintf(stderr,"coop_window_probe_refused rows=%d h=%d sw=%d g=%d pw=%d t=%d: %s\n",rows,h,sw,groups,pw,threads,e.what());ok=false;break;}
     end.record(st);CU(cudaEventSynchronize(end.e));
     float ms;CU(cudaEventElapsedTime(&ms,start.e,end.e));if(i>=3)times.push_back(ms*1e-3);
    }
    if(ok&&times.size()>=9)save(coop_op(groups,pw,sw),rows,h,1,1,threads,times);
   }
  }
 }
 // THE WINDOW BOUNDARY'S OWN SHAPES. coop_weigh charges every staged window an Eq-apply of its sw
 // reflectors over the panel columns beyond it, and asks the service for apply(ge_rows, sw, q) with
 // q <= h <= 128.
 for(int h:{8,16,32,64})for(int rows:{4096,16384,32768,65536}){
  if(rows<h)continue;
  const size_t matrix=size_t(rows)*h,xw=size_t(rows)*128;
  if((2*matrix+xw+size_t(h)*h)*sizeof(T)>limit)continue;
  Buffer<T>a(matrix),v(matrix),t(size_t(h)*h),x(xw),w(size_t(h)*128),z(size_t(h)*128);
  Buffer<TilePacket>packets(1);packets.upload({{0,0,rows,h,0}},st);
  make_input<<<256,128,0,st>>>(a.p,rows,h,rows,0,rows,0,T(1));
  tiled_ge<<<1,256,0,st>>>(a.p,rows,packets.p,1,t.p,h,0,status.p,witness.p);
  pack_ge_v<<<dim3(pack_v_grid(rows,h),1),128,0,st>>>(a.p,rows,packets.p,1,v.p,rows,h);
  make_input<<<256,128,0,st>>>(x.p,rows,128,rows,0,rows,0,T(1));st.sync();
  for(int q:{16,32,64,96,128}){
   // The boundary runs the ENGINE's dispatch at the reference block arm
   // (rep=1, CB=2), exactly as the main grid's composite does: the probe runs
   // what the engine dispatches, not a copy of it.
   auto one=[&](){
    apply_stage_W<T>(bh,st,1,v.p,rows,0,x.p,rows,0,w.p,h,0,rows,h,q,1,nullptr,nullptr,2);
    apply_stage_Z<T>(bh,st,1,t.p,h,0,w.p,h,0,z.p,h,0,h,q,1,true,nullptr,2);
    apply_stage_D<T>(bh,st,1,v.p,rows,0,z.p,h,0,x.p,rows,0,rows,h,q,1);};
   save("apply",rows,h,q,1,0,time_primitive(one,st,9));
  }
 }
 {cudaFuncAttributes attr;cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,ctx.device));CU(cudaFuncGetAttributes(&attr,tiled_ge<T,true>));int capacity=prop.sharedMemPerBlockOptin-attr.sharedSizeBytes;CU(cudaFuncSetAttribute(tiled_ge<T,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,capacity));p["tiled_shared_capacity_bytes"]=capacity;p["kernel_resources"]=kernel_resources<T>(ctx.device);p["shared_panel_resources"]=json::array();
  for(int kind:{0,2})for(int h:{16,32,64,128})for(int rows:{32,64,128,256,512,1024,4096}){
   if(rows<h||(kind==2&&(rows%h||rows/h<2||rows/h>TILE_MAX_RADIX||((rows/h)&(rows/h-1)))))continue;size_t dynamic=size_t(rows)*h*sizeof(T);if(dynamic>size_t(capacity))continue;
   for(int count:{1,8,64,256}){Buffer<T>a(size_t(rows)*h*count),orig(a.n),tri(size_t(count)*h*h);Buffer<TilePacket>packets(count);std::vector<TilePacket>desc;for(int i=0;i<count;++i)desc.push_back({0,i*h,rows,h,i});packets.upload(desc,st);for(int i=0;i<count;++i)structural_input<<<32,128,0,st>>>(orig.p+size_t(i)*rows*h,rows,h,kind);st.sync();
    for(int threads:{128,256,512,1024}){int occupancy;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,tiled_ge<T,true>,threads,dynamic));std::vector<double>times;Event start,end;for(int i=0;i<12;++i){CU(cudaMemcpyAsync(a.p,orig.p,a.n*sizeof(T),cudaMemcpyDeviceToDevice,st));start.record(st);tiled_ge<T,true><<<count,threads,dynamic,st>>>(a.p,rows,packets.p,count,tri.p,h,kind,status.p,witness.p);end.record(st);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,start.e,end.e));if(i>=3)times.push_back(ms*1e-3);}save(kind==0?"GE_shared":"TT_shared",rows,h,1,count,threads,times);p["shared_panel_resources"].push_back({{"kind",kind},{"rows",rows},{"h",h},{"batch",count},{"threads",threads},{"dynamic_shared_bytes",dynamic},{"static_shared_bytes",attr.sharedSizeBytes},{"registers_per_thread",attr.numRegs},{"local_bytes_per_thread",attr.localSizeBytes},{"active_blocks_per_SM",occupancy}});}
   }
  }
 }
  // rows reaches the engine's leaf sizes (leaf runs to 80000) and q reaches the
  // descriptor's maximum strip (tiled_max_strip 32768): the model used to
  // extrapolate off the edge of its samples on both axes (Entry 35's lesson,
  // third bite). Shapes that exceed the sample memory guard are still skipped.
  for(int h:{16,32,64,128})for(int rows:{64,128,256,1024,4096,16384,32768,65536}){
  if(rows<h)continue;
  for(int count:{1,8,64})for(int q:{256,1024,4096,16384,32768}){
   // W must also hold the c partials of the apply carrier side by side, so it
   // is sized for the widest sampled replication, not for the batch alone.
   size_t matrix=size_t(rows)*count*q,backing=size_t(rows)*count*h,small=size_t(count)*h*q;
   const std::vector<int> rep_menu=apply_rep_menu();
   size_t wslots=std::max(size_t(count),size_t(rep_menu.back()))*size_t(h)*q;
   if((matrix+2*backing+small+wslots+size_t(count)*h*h)*sizeof(T)>limit)continue;
   Buffer<T>a(backing),v(backing),t(size_t(count)*h*h),x(matrix),w(wslots),z(small);Buffer<TilePacket>packets(count);std::vector<TilePacket>desc;for(int i=0;i<count;++i)desc.push_back({i*rows,0,rows,h,i});packets.upload(desc,st);
   make_input<<<256,128,0,st>>>(a.p,rows*count,h,rows*count,0,rows*count,0,T(1));tiled_ge<<<count,256,0,st>>>(a.p,rows*count,packets.p,count,t.p,h,0,status.p,witness.p);pack_ge_v<<<dim3(pack_v_grid(rows,h),count),128,0,st>>>(a.p,rows*count,packets.p,count,v.p,rows,h);make_input<<<256,128,0,st>>>(x.p,rows*count,q,rows*count,0,rows*count,0,T(1));st.sync();
    // The unreplicated baseline, sampled through the NATIVE BLOCK-CARRIED kernels the engine
    // dispatches at GPU-level c=1. Every sampler below goes through apply_stage_*, the one
    // implementation the engine dispatches, at the block carrier it names.
    auto stage_w=[&](int rep,T*out,T**pp,int bc=2){apply_stage_W<T>(bh,st,rep,v.p,rows,(long long)rows*h,x.p,rows*count,(long long)rows,out,h,size_t(h)*q,rows,h,q,count,pp,nullptr,bc);};
    auto stage_z=[&](int zr,const T*in,T*out,T**pp,int bc=2){apply_stage_Z<T>(bh,st,zr,t.p,h,(long long)h*h,in,h,size_t(h)*q,out,h,size_t(h)*q,h,q,count,true,pp,bc);};
    // rep<=1 only (composite reference arm and per-CB block samples): the carried path ignores the
    // partial credit, so it stays null with the block carrier in the block_c position.
    auto stage_d=[&](int dr,const T*in,T*dst,int bc=2){apply_stage_D<T>(bh,st,dr,v.p,rows,size_t(rows)*h,in,h,size_t(h)*q,dst,rows*count,(long long)rows,rows,h,q,count,bc);};
    // The composite at the REFERENCE block arm (menu front). The per-stage
    // block samples below price the other arms exactly; the composite prices
    // the serial fall-back, the tail, the across-GPU update and the panel
    // boundary at the arm the probe names, never at an unpriced one.
    auto product=[&](){stage_w(1,w.p,nullptr,2);stage_z(1,w.p,z.p,nullptr,2);stage_d(1,z.p,x.p,2);};
    save("apply",rows,h,q,count,0,time_primitive(product,st,9));
   // PER-STAGE BLOCK-CARRIER SAMPLES FOR Eq pipe AND FOR THE BLOCK SELECTOR. The depth d overlaps
   // the three products of Eq apply, so C(d) needs t_W, t_Z and t_D separately -- their sum alone
   // cannot tell d anything, which is why the model was blind to d.
    // Each lambda is TiledEngine::product's corresponding stage at the block
    // carrier it names, so the model prices what the engine dispatches, not
    // an idealization of it. The GPU-level c>1 stages below run the cuBLAS
    // peer machinery (c>1 regime).
    for(int wb:block_rep_menu()){
     auto sWb=[&](){stage_w(1,w.p,nullptr,wb);};
     save(apply_w_block_op(wb),rows,h,q,count,0,time_primitive(sWb,st,9));}
    for(int zb:block_rep_menu()){
     auto sZb=[&](){stage_z(1,w.p,z.p,nullptr,zb);};
     save(apply_z_block_op(zb),rows,h,q,count,0,time_primitive(sZb,st,9));}
    for(int db:block_rep_menu()){
     auto sDb=[&](){stage_d(1,z.p,x.p,db);};
     save(apply_d_block_op(db),rows,h,q,count,0,time_primitive(sDb,st,9));}
   // The apply carrier's contraction replication. Only c changes W; Z contracts over h and D
   // commits its own rows, so neither is resampled per c.
   //
   // The count>1 arm is sampled through the SAME pointer-array peers the engine dispatches,
   // pointer-fill kernel included, because peer (i,z) sits at i*s_batch + z*part and that is two
   // strides, not one.
    for(int rep:rep_menu){
     // rep_menu holds strictly >1 candidates; rep==1 is covered by the native
     // apply_W sample above, so a c1 carrier sample would be probe time spent
     // on a name nothing reads.
     if(rep<2||rows<rep)continue;
    const int peers=count*rep;
    const size_t wneed=size_t(peers)*size_t(h)*q;
    if((matrix+2*backing+small+wslots+wneed+size_t(count)*h*h)*sizeof(T)+size_t(3*peers)*sizeof(T*)>limit)continue;
    Buffer<T>wc(wneed);Buffer<T*>pp(size_t(3)*peers);
    // THE PROBE RUNS THE ENGINE'S DISPATCH, not a copy of it: apply_stage_W is
    // the same function TiledEngine::product and the across-GPU merge call.
    auto sWc=[&](){apply_stage_W<T>(bh,st,rep,v.p,rows,(long long)rows*h,x.p,rows*count,(long long)rows,
      wc.p,h,size_t(h)*q,rows,h,q,count,pp.p,nullptr);};
    save(apply_rep_op(rep),rows,h,q,count,0,time_primitive(sWc,st,9));
   }
   // Z'S AND D'S OWN CARRIERS (Alg 2). Their [K] is the reflector index h, so the cut is over
   // reflector ranges rather than row ranges. Sampled through the SAME dispatch the engine uses --
   // pointer-array peers plus the additive combine for Z, in-place accumulation for D -- so a c
   // that loses here loses on the kernel that would actually run. Sampling both, and recording the
   // loss, is what turns "Z and D carry c=1" from an assumption into a priced refusal.
   for(int zc:z_rep_menu()){
    if(h<zc)continue;
    const int peers=count*zc;
    const size_t zneed=size_t(peers)*size_t(h)*q;
    if((matrix+2*backing+small+wslots+zneed+size_t(count)*h*h)*sizeof(T)+size_t(3*peers)*sizeof(T*)>limit)continue;
    Buffer<T>zc_buf(zneed);Buffer<T*>pp(size_t(3)*peers);
    auto sZc=[&](){apply_stage_Z<T>(bh,st,zc,t.p,h,(long long)h*h,w.p,h,size_t(h)*q,
      zc_buf.p,h,size_t(h)*q,h,q,count,true,pp.p);};
    save(apply_z_rep_op(zc),rows,h,q,count,0,time_primitive(sZc,st,9));
   }
   // D'S OWN CARRIER, SAMPLED THROUGH THE ENGINE'S DISPATCH (Alg 2): drep sequential in-place
   // passes over X.
   for(int dc:d_rep_menu()){
    if(h<dc)continue;
    // The engine runs the native carried D wherever native_d_path holds (Z then arrives as Zt, ld
    // q); the sample runs exactly that arm.
    const bool nd=native_d_path<T>(dc,h);
    auto sDc=[&](){apply_stage_D<T>(bh,st,dc,v.p,rows,size_t(rows)*h,z.p,nd?q:h,size_t(h)*q,
      x.p,rows*count,(long long)rows,rows,h,q,count,2,nullptr,nd);};
    save(apply_d_rep_op(dc),rows,h,q,count,0,time_primitive(sDc,st,9));
   }
   // Eq pipe says that once d is deep enough the packets leave at the pace of the slowest stage,
   // t_max.
   //   pace(d,c) = (T(d,c) - t_sum) / (R-1)
   // is Eq pipe's t_max with the contention already in it. A shape whose buffers do not fit gets no
   // pace sample, and the model then credits that depth with NOTHING (serial) rather than guessing
   // -- an unmeasured overlap is never paid for. THE CAVEAT THAT USED TO BE HERE WAS THE BUG. All R
   // packets swept the SAME strip of X, while the engine gives every in-flight packet a DISJOINT
   // strip.
   //
   // d disjoint strips at (16384,4096) fp64 is 2.1 GB against the 2 GB sample limit, i.e. precisely
   // the shapes that matter would be skipped, so the pace loop carries its own larger allowance.
   // Every other sample keeps the original limit, so nothing else about the profile moves.
    if(count==1){
     const int R=tiled_pipe_pace_packets;
     // rep 1 is the native-executed baseline; the carried arms run the cuBLAS
     // peer machinery. Both are paced because the model must price the depth
     // on the kernels that will actually run in each regime.
     std::vector<int>reps{1};for(int r:rep_menu)if(r>1)reps.push_back(r);
     // AND THE PACE NOW CARRIES Z AND D TOO. The old composite took the carried W through cuBLAS
     // and then ran UNSPLIT full-contraction Z and D -- kernels the engine never runs when zc/dc>1
     // -- while t_Sigma came from the SPLIT stage samples. pace=(comp-t_Sigma)/(R-1) therefore
     // subtracted mismatched arms, landed above t_Sigma, and the model's clamp into [t_max,t_Sigma]
     // returned t_Sigma: the SERIAL pace, i.e. ZERO depth credit at every shape. The key now names
     // the WHOLE carrier, so a (d,c,zc,dc) arm is priced on its own measurement and never on
     // another arm's.
     std::vector<int>zs{1};for(int r:z_rep_menu())zs.push_back(r);
     std::vector<int>ds{1};for(int r:d_rep_menu())ds.push_back(r);
    // The (zc,dc) grid multiplies this loop by nine, and the X strips it needs
    // are the expensive part -- d*matrix words allocated and initialised. They
    // depend on d alone and the W credit on (d,rep), so both are hoisted and
    // only the Z credit is per zc. Without this the grid pays its allocation
    // cost nine times over for no extra measurement.
    for(int d:pipe_depths){
     const size_t xneed=size_t(d)*matrix;
     if((2*backing+small+wslots+xneed+size_t(count)*h*h)*sizeof(T)>pace_limit)continue;
     Buffer<T>px(xneed);
     make_input<<<256,128,0,st>>>(px.p,rows*count,d*q,rows*count,0,rows*count,0,T(1));st.sync();
     for(int rep:reps){
      if(rep>1&&rows<rep)continue;
      const size_t wneed=size_t(d)*rep*size_t(h)*q;
      if((2*backing+small+wslots+xneed+wneed+size_t(count)*h*h)*sizeof(T)>pace_limit)continue;
      Buffer<T>pw(wneed);
      for(int zc:zs){
       if(zc>1&&h<zc)continue;
       const size_t zneed=size_t(d)*size_t(zc)*size_t(h)*q;
       if((2*backing+small+wslots+xneed+wneed+zneed+size_t(count)*h*h)*sizeof(T)>pace_limit)continue;
       Buffer<T>pz(zneed);
       const size_t peer_words=size_t(3)*std::max(rep,zc);
       Buffer<T*>ppp(size_t(d)*peer_words);
       for(int dc:ds){
        if(dc>1&&h<dc)continue;
        auto composite=[&](){
         pipe_fork.record(st);for(int i=0;i<d;++i)pipe_fork.wait(pipe_st[i].s);
         for(int k=0;k<R;++k){const int slot=k%d;cublasHandle_t sbh=pipe_bh[slot];cudaStream_t ss=pipe_st[slot].s;
          T*w=pw.p+size_t(slot)*rep*size_t(h)*q,*z=pz.p+size_t(slot)*size_t(zc)*size_t(h)*q;
          T**pp=ppp.p+size_t(slot)*peer_words;
          // The slot's OWN strip of X. Two packets in flight never touch the
          // same words, which is the engine's arrangement and was not this
          // probe's.
          T*xk=px.p+size_t(slot)*matrix;
          // The SAME three functions TiledEngine::product calls, at the SAME
          // carrier, on the slot's own stream and cuBLAS handle.
          apply_stage_W<T>(sbh,ss,rep,v.p,rows,(long long)rows*h,xk,rows*count,(long long)rows,w,h,size_t(h)*q,rows,h,q,count,pp,nullptr,2);
          const bool nd=native_d_path<T>(dc,h);
          apply_stage_Z<T>(sbh,ss,zc,t.p,h,(long long)h*h,w,h,size_t(h)*q,z,nd?q:h,size_t(h)*q,h,q,count,true,pp,2,nullptr,nd);
          apply_stage_D<T>(sbh,ss,dc,v.p,rows,size_t(rows)*h,z,nd?q:h,size_t(h)*q,xk,rows*count,(long long)rows,rows,h,q,count,2,nullptr,nd);
         }
         for(int i=0;i<d;++i){pipe_join[i].record(pipe_st[i].s);pipe_join[i].wait(st);}
        };
        auto raw=time_primitive(composite,st,9);
        // Saved on the SAME (rows,h,q,count,threads=0) signature the stage
        // samples use, so the service matches it the way it matches every other
        // op. R is fixed by the model's pace formula, not carried per sample.
        save(apply_pace_op(d,rep,zc,dc),rows,h,q,count,0,raw);
       }   // dc
      }    // zc
     }     // rep
    }      // depth
   }
   if(q==256)save("pack_V",rows,h,1,count,0,time_primitive([&](){pack_ge_v<<<dim3(pack_v_grid(rows,h),count),128,0,st>>>(a.p,rows*count,packets.p,count,v.p,rows,h);},st,9));
  }
 }
 for(int count:{1024,1048576,67108864}){Buffer<T>x(count);make_input<<<256,128,0,st>>>(x.p,count,1,count,0,count,0,T(1));st.sync();save("finite_scan",count,1,1,1,256,time_primitive([&](){finite_scan<<<256,256,0,st>>>(x.p,count,1,count,status.p);},st,9));}
 st.sync();int err=status.download()[0];p["tiled_probe_status"]=ctx.max_status(err);tiled_blas_check(cublasDestroy(bh));
 // Concurrent ranks measure their actual same-node local service. Preserve
 // each raw rank table; prediction uses the slowest rank's median per signature.
 auto ranks=ctx.all_records(p["tiled_samples"]);p["tiled_rank_samples"]=ranks;
 for(size_t i=0;i<p["tiled_samples"].size();++i){double mx=0;for(auto&r:ranks)mx=std::max(mx,r[i]["median_s"].template get<double>());p["tiled_samples"][i]["median_s"]=mx;}
 // The publish arena is now per-sender, so it is charged separately from the
 // collective capacity. The model only ever asks NV_publish for b*h <= 16384
 // elements (one R block) and for 0, so a 131072-element publish arena covers
 // every sample the selector can request while keeping the symmetric heap at a
 // few MB instead of ranks x 16 MB.
 if(ctx.size>1){size_t cap=size_t(128)*16384*sizeof(T),pubcap=size_t(131072)*sizeof(T);Transport transport(ctx,cap,0,pubcap);Buffer<T>in(cap/sizeof(T)),out(in.n);make_input<<<256,128,0,st>>>(in.p,int(in.n),1,int(in.n),0,int(in.n),0,T(1));st.sync();
  // DENSER COUNTS ACROSS THE CLIFF.
  for(int count:{0,1,31,1024,16384,131071,262144,393216,524288,786432,1048576,1572864,2097152}){if(size_t(count)>in.n)continue;for(int op=0;op<3;++op){if(op==0&&size_t(count)*sizeof(T)>pubcap)continue;std::vector<double>times;for(int rep=0;rep<12;++rep){ctx.barrier();double start=seconds();if(op==0)transport.publish(0,ctx.size-1,in.p,out.p,size_t(count)*sizeof(T),count,800);else if(op==1)transport.template paper<T,false>(0,in.p,out.p,count,count);else transport.template paper<T,true>(0,in.p,out.p,count,count);transport.drain();ctx.barrier();if(rep>=3)times.push_back(seconds()-start);}save(op==0?"NV_publish":op==1?"LL_copy":"LL_reduce",count,1,1,ctx.size,128,times);}}
  // THE EPISODE AS THE ENGINE RUNS IT. Above, twelve identical collectives run back to back on an
  // idle machine. The engine instead issues reduce-then-broadcast pairs SEPARATED BY COMPUTE, on the
  // stream the transport is bound to, so each episode's barrier waits on the slowest rank's
  // arithmetic. Pricing the engine's episodes with the isolated number is the same error as the pace
  // probe's unsplit Z/D, and it has the same remedy: measure the pattern that runs. One sample per
  // count, R pairs deep, with a representative apply kernel between pairs so the critical path is in
  // the number.
  {const int R=8;Buffer<T>cw(size_t(128)*1024),cx(size_t(128)*1024);
   make_input<<<256,128,0,st>>>(cx.p,128,1024,128,0,128,0,T(1));st.sync();
   for(int count:{16384,131071,262144,393216,524288,786432,1048576,1572864,2097152}){
    if(size_t(count)>in.n)continue;
    std::vector<double>times;
    for(int rep=0;rep<9;++rep){ctx.barrier();double start=seconds();
      for(int k=0;k<R;++k){
       transport.template paper<T,true>(0,in.p,out.p,count,count);
       // representative local work between the two halves of the episode: the
       // native block-carried W at the reference arm, which is what the engine
       // runs there now (the serial native oracle it replaced never runs).
       launch_carried_W<T>(cx.p,128,0,cx.p,128,0,cw.p,128,0,128,128,64,1,2,st);
       transport.template paper<T,false>(0,in.p,out.p,count,count);
      // PER-EPISODE synchronization, as the engine pays it.
      transport.drain();ctx.barrier();
     }
     if(rep>=3)times.push_back((seconds()-start)/R);}
    save("LL_episode",count,1,1,ctx.size,128,times);
   }
  }
  p["tiled_backend_evidence"]=transport.evidence;p["tiled_backend_capacity_bytes"]=cap;
 }
 // Deterministic split by operation after measurement, before any held-out QR.
 p["tiled_heldout_samples"]=json::array();json fit=json::array();std::map<std::string,int>ordinal;for(auto&s:p["tiled_samples"]){std::string op=s["op"];if(++ordinal[op]%5==0)p["tiled_heldout_samples"].push_back(s);else fit.push_back(s);}p["tiled_samples"]=std::move(fit);
 p["fit_boundary"]="every fifth operation signature reserved before matrix execution; raw per-rank observations retained";p["tiled_cache_condition"]="ordinary repeated operand accesses; footprint and prior-touch conditional, no residency guarantee";
 p["packet_service_model"]="measured batch curves and row interpolation; queried CTA residency for waves beyond measured counts; fixed packet cost retained below minimum rows; predictive only";
 p.erase("id");p["id"]=digest(p.dump());return p;
}
struct CoopChoice {int c=0,pw=0,threads=0,sw=0;double cost=INFINITY;};
class TiledServices {
 struct Sample {std::string op;int rows,h,q,count,threads;double seconds;};std::vector<Sample> samples;
 // Struct-keyed service cache and coop_weigh memo. Pure memoization -- every returned value is the
 // one the uncached computation returns.
 struct ServiceKey {std::string op;int64_t rows;int h,q,count,threads;
  bool operator==(const ServiceKey& x)const{return op==x.op&&rows==x.rows&&h==x.h&&q==x.q&&count==x.count&&threads==x.threads;}};
 struct ServiceHash {size_t operator()(const ServiceKey& x)const{
  size_t z=std::hash<std::string>{}(x.op);auto mix=[&](size_t v){z^=v+0x9e3779b9+(z<<6)+(z>>2);};
  mix(std::hash<int64_t>{}(x.rows));mix(x.h);mix(x.q);mix(x.count);mix(x.threads);return z;}};
 std::unordered_map<ServiceKey,double,ServiceHash>cache;double launch;PacketServices packets;
public:
 // Per-kernel shared capacity and residency: the capacity that admits a cooperative carrier is that
 // of the specialization that will run.
 CoopResources coop_res;
 std::map<std::tuple<int,int,size_t,size_t,int>,CoopChoice> coop_choices;
 size_t coop_calls=0,coop_hits=0;
 // The floor under an interpolated estimate is the ASYNCHRONOUS launch gap,
 // not the synchronous round trip: the engine issues and runs ahead. Old
 // profiles without the sample fall back to launch_s, so they stay selectable
 // and keep their old numbers exactly.
 explicit TiledServices(const json&p):launch(p.value("gpu_launch_gap_s",p.value("launch_s",0.0))),packets(p),coop_res(CoopResources::from(p)){for(auto&s:p.at("tiled_samples"))samples.push_back({s.at("op"),s.at("rows"),s.at("h"),s.at("q"),s.at("batch"),s.at("threads"),s.at("median_s")});}
 // Below it every estimate is a linear extrapolation clamped only at the launch latency, so the
 // selector treats it as the floor of the strip menu.
 int min_sampled_apply_q()const{int q=0;for(auto&s:samples){if(s.op!="apply"&&s.op.rfind("apply_",0)!=0)continue;if(!q||s.q<q)q=s.q;}return q?q:256;}
 double get(const std::string&op,int64_t rows,int h,int q,int count,int threads=0){if(!count)return 0;ServiceKey key{op,rows,h,q,count,threads};auto it=cache.find(key);if(it!=cache.end())return it->second;
  if(packets.supports(op))return cache[key]=std::max(launch,packets.get(op,int(rows),h,count,threads));
  double distance=INFINITY,estimate=INFINITY;for(auto&s:samples){if(s.op!=op||s.threads!=threads)continue;double d=abs(log(double(std::max<int64_t>(1,rows))/std::max(1,s.rows)))+2*abs(log(double(h)/s.h))+abs(log(double(std::max(1,q))/s.q))+abs(log(double(count)/s.count));if(d>=distance)continue;distance=d;
   double ratio;if(op.rfind("coop_GE",0)==0)ratio=double(h)/s.h*double(count)/s.count;
   else if(op=="GE"||op=="TT"||op=="GE_shared"||op=="TT_shared")ratio=double(rows)*h*h/(double(s.rows)*s.h*s.h)*double(count)/s.count;else if(op=="apply")ratio=(4.0*rows*h+2.0*h*h)*q*count/((4.0*s.rows*s.h+2.0*s.h*s.h)*s.q*s.count);
   // Eq apply's stages scale differently, so they cannot share one rule.
   // W = V^T X and D = V Z both contract or emit over the ROW index, so they
   // scale as rows*h*q; Z = T^T W contracts over the h reflectors and never
   // touches a matrix row, so it scales as h*h*q and must NOT carry rows.
    else if(op=="apply_Z"||op.rfind("apply_Z_c",0)==0||op.rfind("apply_Z_b",0)==0)ratio=double(h)*h*q*count/(double(s.h)*s.h*s.q*s.count);
    else if(op=="apply_W"||op=="apply_D"||op.rfind("apply_W_c",0)==0||op.rfind("apply_D_c",0)==0||op.rfind("apply_W_b",0)==0||op.rfind("apply_D_b",0)==0)ratio=double(rows)*h*q*count/(double(std::max(1,s.rows))*s.h*s.q*s.count);
   else ratio=double(std::max<int64_t>(1,rows))*h*std::max(1,q)*count/(double(std::max(1,s.rows))*s.h*s.q*s.count);
   estimate=std::max(launch,s.seconds*ratio);
  }if(cache.size()>262144)cache.clear();cache[key]=estimate;return estimate;
 }
};
// The panel product's carrier, weighed from measurement. Enumerates the sampled menu -- contraction
// replication c, minipanel width, thread count -- applies the LAUNCHER's own feasibility test so
// the selector can never price a kernel the engine would refuse, and returns the cheapest. Both
// tiled_model and tiled_select call this, so what is priced is what is dispatched. The LAUNCHER
// sizes its slab by that window -- coop_mini_shared_bytes( local_height, sw, PW, word) -- and this
// test sized it by h, so the two disagreed exactly where it matters. At h=128 fp64 with 128 groups
// the unwindowed slab is local_height*(h+1)*8, which passes the 227 KB optin at rows>28160, so
// every panel taller than that was refused HERE even when a window was forced and the launcher
// would have accepted it. That refusal is what keeps the panel carrier unreachable at n>=32768 for
// b=128. THE PRICE IS NOW THE WINDOWED SAMPLE. The SMALLEST shared footprint any admitted carrier
// candidate would need at this shape.
inline size_t coop_cheapest_shared(int ge_rows,int h,size_t word,int window=0){
 size_t best=0;std::vector<int> windows;
 if(window>0&&window<h)windows.push_back(window);
 else{windows.push_back(h);for(int w:coop_window_menu())if(w<h)windows.push_back(w);}
 for(int sw:windows)for(int g:coop_group_menu())for(int pw:coop_width_menu()){
  if(pw>sw)continue;if(sw<h&&!pw)continue;
  const int local_height=ceildiv(ge_rows,g);
  const size_t need=pw?coop_mini_shared_bytes(local_height,sw,pw,word):coop_shared_bytes(local_height,h,word);
  if(!best||need<best)best=need;}
 return best;
}
// THE WINDOW IS NOW A CHOICE, NOT A FLAG. `window>0` still forces one; window==0 lets the selector
// weigh the unwindowed carrier against every admitted staged window from its own samples.
inline CoopChoice coop_weigh(TiledServices&service,int ge_rows,int h,size_t coop_capacity,size_t word,int window=0){
 CoopChoice out;if(!coop_capacity||!word)return out;
 const auto memo_key=std::make_tuple(ge_rows,h,coop_capacity,word,window);
 ++service.coop_calls;{auto memo=service.coop_choices.find(memo_key);
  if(memo!=service.coop_choices.end()){++service.coop_hits;return memo->second;}}
 std::vector<int> windows;
 if(window>0&&window<h)windows.push_back(window);
 else{windows.push_back(h);for(int w:coop_window_menu())if(w<h)windows.push_back(w);}
 for(int sw:windows)for(int g:coop_group_menu())for(int pw:coop_width_menu())for(int t:{256,512}){
  const bool windowed=sw<h;
  if(pw>sw)continue;
  if(windowed&&!pw)continue;          // the staged window exists only for the minipanel kernel
  const int local_height=ceildiv(ge_rows,g);
  const size_t need=pw?coop_mini_shared_bytes(local_height,sw,pw,word):coop_shared_bytes(local_height,h,word);
  if(!service.coop_res.fits(pw,g,t,need,coop_capacity))continue;
  double cost=service.get(coop_op(g,pw,windowed?sw:0),ge_rows,h,1,1,t);
  if(!std::isfinite(cost))continue;
  if(windowed){
   // One boundary per window except the last: an Eq-apply of this window's sw
   // reflectors over the panel columns beyond it. Summed at the real shapes,
   // on the composite at the reference block arm (same documented
   // approximation as the global-tree update).
   double b=0;bool priced=true;
   for(int w0=0;w0+sw<h;w0+=sw){
    const double e=service.get("apply",ge_rows,sw,h-w0-sw,1);
    if(!std::isfinite(e)){priced=false;break;}
    b+=e;}
   if(!priced)continue;               // an unpriced boundary is never credited as free
   cost+=b;
  }
  if(cost<out.cost)out={g,pw,t,sw,cost};
 }
 service.coop_choices.emplace(memo_key,out);
 return out;
}
// Stamp the weighed panel carrier onto every batch that will dispatch it.
inline int tiled_carry_panel_carriers(TiledPlan&plan,TiledServices&service,size_t shared_capacity,size_t word){
 if(plan.scalar)return 0;
 // csw_forced is the window forced by the caller (0 = unforced); csw is the window selected in the
 // immutable schedule. Weigh with the forced value; record the selected one.
 const int forced_window = plan.csw_forced;
 int carried=0;std::map<int,int> widths,windows_chosen;
 for(auto&pan:plan.panels)for(auto&batch:pan.ranks){
   // A RAGGED TAIL (h < b, the last panel when n is not a multiple of b) is carried too. It used to be skipped here,
   // so on a tall matrix the last panel -- e.g. 45056 x 56 at 48000x3000 -- fell to the ONE-CTA block path and cost
   // ~40 ms. The cooperative kernel and the engine's window loop already take h < b; the price comes from the nearest
   // sampled width (coop_GE ratio h/s.h).
   if(batch.ge.size()!=1||batch.h>plan.b||plan.b>coop_max_b)continue;
   CoopChoice pick=coop_weigh(service,batch.ge[0].rows,batch.h,shared_capacity,word,forced_window);
   batch.cshared_cap=service.coop_res.max_capacity(shared_capacity);
   batch.cshared_need=coop_cheapest_shared(batch.ge[0].rows,batch.h,word,forced_window);
   batch.ccoop_s=std::isfinite(pick.cost)?pick.cost:0.0;
   {const double blk=service.get(plan.ge_shared?"GE_shared":"GE",batch.ge[0].rows,batch.h,1,1,plan.threads);
    batch.cblock_s=std::isfinite(blk)?blk:0.0;}
   if(!std::isfinite(pick.cost))continue;
   // It was the constant coop_min_rows=1024, which stood in for "below this the block path wins"
   // with no number behind it. Compare the weighed carrier against the block GE the engine would
   // run on the SAME packet instead, and stamp only where the carrier is priced to win. A shape
   // with no carrier sample leaves pick non-finite and is skipped above, so an unsampled carrier is
   // still never credited.
   if(batch.cblock_s>0&&batch.cblock_s<=pick.cost)continue;
   // Panel c is CHOSEN from measurement among >1 candidates (the menu is
   // {32,64,128}; 1 is never offered, nothing is fixed). pw/threads ride
   // along from the same weighed choice.
   // The window the WEIGHER chose (0 = unwindowed), so the engine dispatches the
   // shape that was priced. Forcing --csw still pins it through coop_weigh.
   batch.cc=pick.c;batch.cpw=pick.pw;batch.cthreads=pick.threads;
   batch.csw=(pick.sw>0&&pick.sw<batch.h)?pick.sw:0;
   ++windows_chosen[batch.csw];
   plan.c=std::max(plan.c,pick.c);plan.cthreads=pick.threads;++widths[pick.pw];++carried;
 }
 if(!widths.empty())plan.pw=std::max_element(widths.begin(),widths.end(),[](const std::pair<const int,int>&a,const std::pair<const int,int>&b){return a.second<b.second;})->first;
 // REPORT THE WINDOW THAT WAS ACTUALLY STAMPED. If unforced, record the weigher's most-chosen
 // window as the selected one; if forced, the selected stays the forced value (coop_weigh was
 // pinned to it, so every stamped batch already carries it). Summaries never become replay inputs:
 // replay restores csw from the schedule.
 if(!windows_chosen.empty()){
  if(!forced_window)
   plan.csw=std::max_element(windows_chosen.begin(),windows_chosen.end(),[](const std::pair<const int,int>&a,const std::pair<const int,int>&b){return a.second<b.second;})->first;
  else plan.csw=forced_window;
 } else if(forced_window) plan.csw=forced_window;
 plan.c=std::max(plan.c,1);return carried;
}
inline double tiled_model(int m,int n,int p,int b,int leaf,int strip,int threads,TiledServices&service,double host,bool ge_shared=false,bool tt_shared=false,int radix=2,size_t coop_capacity=0,size_t word=0,int depth=1,int apply_c=1,int gradix=2,int zrep=1,int drep=1,int wb=2,int zb=2,int db=2,double*terms=nullptr,int row_block=0){
 // The b axis is the one the DP now ranks backwards at n=1024-4096 -- it prices b=16 at 10.511 ms
 // where it measures 7.22 -- and an aggregate number cannot say which term carries the 45%. Exact
 // at p=1; at p>1 the per-column maximum is taken over ranks, so the split is indicative.
 const RowMap map(m,p,row_block);
 double total=0;auto add=[&](int k,double v){if(terms)terms[k]+=v;return v;};
 if(m&&n){int rows=ceildiv(m,p);total+=add(0,service.get("column_max",rows,1,n,1,256)+service.get("column_normalize",rows,1,n,1,256)+service.get("upper_restore",rows,1,n,1,256)+3*host);
  if(p>1)for(int col=0;col<n;col+=strip){int q=std::min(strip,n-col);total+=p*(service.get("LL_copy",q,1,1,p,128)+service.get("scale_max",q,1,1,1,128)+host);}
 }
 for(int col=0;col<std::min(m,n);){int root=map.owner(col),h=std::min({b,std::min(m,n)-col,map.run_end(col)-col}),trailing=n-col-h;double local=0;int roots=0;
  for(int rank=0;rank<p;++rank){int rows=map.local_rows(rank)-map.lb(rank,col);if(!rows)continue;++roots;int count=ceildiv(rows,leaf);const int ge_rows=std::min(rows,leaf);
   double ge=service.get(ge_shared?"GE_shared":"GE",ge_rows,h,1,count,threads);
   // The engine dispatches it exactly when count==1 (the panel is one packet, so no local tree
   // remains), ge_rows>=coop_min_rows and h==b, so price that, not the block path it will not run.
   // An unsampled point returns non-finite and simply leaves the block cost in place, which keeps
   // old profiles selectable. Apply the SAME feasibility test here so the selector cannot price a
   // kernel the engine will not run.
    if(count==1&&h==b&&b<=coop_max_b){
     // Panel c is CHOSEN from measurement among >1 candidates (standing
     // order: never 1, never fixed). Price what the engine will dispatch, and
     // dispatch only where the carrier BEATS the block path at this shape --
     // the same comparison tiled_carry_panel_carriers makes, so the model and
     // the stamp cannot disagree about which kernel runs.
     const CoopChoice pick=coop_weigh(service,ge_rows,h,coop_capacity,word);
     if(std::isfinite(pick.cost)&&pick.cost<ge)ge=pick.cost;
    }
   double cost=add(1,ge+service.get("pack_V",ge_rows,h,1,count)+host*3);
   int full=trailing/strip,tail=trailing%strip;
   // The r strips are the packets, the three products of Eq apply are the stages, and depth d
   // admits d of them at once under their own W/Z credits: C(d) = t_sum + (r-1) t_max +
   // floor((r-1)/d) (t_sum - d t_max)_+ . Z and D carry their own selected replication (Alg 2),
   // admitted exactly where TiledEngine::carrier_c_z / carrier_c_d admit it: h must hold the cut.
   const int zr=(zrep>1&&h>=zrep)?zrep:1,dr=(drep>1&&h>=drep)?drep:1;
   // The GPU-level c>1 arms keep the cuBLAS peer samples, whose block geometry is vendor-internal.
   auto stages=[&](int arows,int acount,int aq,int rep,double*tsum,double*tmax)->bool{
    double tw=rep>1?service.get(apply_rep_op(rep),arows,h,aq,acount):service.get(apply_w_block_op(wb),arows,h,aq,acount);
    double tz=zr>1?service.get(apply_z_rep_op(zr),arows,h,aq,acount):service.get(apply_z_block_op(zb),arows,h,aq,acount);
    double td=dr>1?service.get(apply_d_rep_op(dr),arows,h,aq,acount):service.get(apply_d_block_op(db),arows,h,aq,acount);
    if(!std::isfinite(tw)||!std::isfinite(tz)||!std::isfinite(td))return false;
    *tsum=tw+tz+td;*tmax=std::max(tw,std::max(tz,td));return true;};
   // It is clamped into [t_max, t_sum]: a pipeline cannot beat its slowest stage, and cannot be
   // slower than running the packets serially.
   auto pace_of=[&](int arows,int acount,int aq,int dd,int rep,double tsum,double tmax)->double{
    const int R=tiled_pipe_pace_packets;
    // BLOCK-LEVEL MATCHING: the pace composite runs every GPU-level c=1 stage at the REFERENCE
    // block carrier (menu front), while tsum above is priced at the SELECTED triple.
    const int bref=block_rep_menu().front();
    const bool match=(rep>1||wb==bref)&&(zr>1||zb==bref)&&(dr>1||db==bref);
    double comp=match?service.get(apply_pace_op(dd,rep,zr,dr),arows,h,aq,acount)
                     :std::numeric_limits<double>::quiet_NaN();
    if(!std::isfinite(comp))return tsum;
    double pace=(comp-tsum)/double(R-1);
    return std::min(tsum,std::max(tmax,pace));};
   // The SERIAL fused form, for every apply the engine does not pipeline. Only the leaf
   // trailing-apply loop in FamilyEngine::ge_batch runs on pipeline slots; the tree-level applies
   // go through tile_merge_apply, a single fused kernel on the default stream with no W/Z credits
   // and no [K] cut, so crediting them with a depth or a replication would price a schedule the
   // engine does not run. The tree-level applies run through TiledEngine::local_trailing. So the
   // serial form prices the carrier exactly where the engine will dispatch it and the fused sample
   // everywhere else.
   const bool fused_merge=b<=TQR_MERGE_APPLY_MAX_B&&256%b==0&&
     merge_apply_shared(b,256/b,16,word)<=size_t(48)*1024;
   auto serial_strip=[&](int arows,int acount,int aq){
    const int rep=(!fused_merge&&apply_c>1&&arows>=apply_c)?apply_c:1;
    // Both regimes go through the stages first: rep>1 sums the cuBLAS peer samples, rep==1 sums the
    // selected block-carrier samples. The composite is the fall-back when a stage sample is missing
    // -- and, at fused shapes where the engine runs the one-kernel pentagonal apply, its stand-in,
    // as before.
    {double ts=0,tm=0;if(stages(arows,acount,aq,rep,&ts,&tm))return ts;}
    return service.get("apply",arows,h,aq,acount);};
   auto update=[&](int arows,int acount){return full*(serial_strip(arows,acount,strip)+host)+(tail?serial_strip(arows,acount,tail)+host:0);};
   auto pipelined_update=[&](int arows,int acount){
    const int rep=(apply_c>1&&arows>=apply_c)?apply_c:1;
    const int dd=std::max(1,depth);double out=0,tsum=0,tmax=0;
    if(full){ if(stages(arows,acount,strip,rep,&tsum,&tmax)){
      const double pace=pace_of(arows,acount,strip,dd,rep,tsum,tmax);
      out+=double(full)*host+tsum+double(full-1)*pace;
     } else out+=full*(service.get("apply",arows,h,strip,acount)+host); }
    if(tail){ double ts2=0,tm2=0; if(stages(arows,acount,tail,rep,&ts2,&tm2))out+=ts2+host;
      else out+=service.get("apply",arows,h,tail,acount)+host; }
    return out;};
   cost+=add(2,pipelined_update(std::min(rows,leaf),count));int short_rows=rows%leaf;if(short_rows&&short_rows<h){int extra=h-short_rows;cost+=service.get("apply",short_rows,short_rows,extra,1)+host*ceildiv(extra,strip);}
   // Same grouping as tiled_instantiate: k children per node, singletons
   // carried forward, so levels fall as log_k while a level costs C(k*h).
   for(int nodes=count;nodes>1;){int merges=0,next=0;
    for(int i=0;i<nodes;i+=radix){int g=std::min(radix,nodes-i);if(g>1)++merges;++next;}
    cost+=add(3,service.get(tt_shared?"TT_shared":"TT",radix*h,h,1,merges,threads)+service.get("pack_V",radix*h,h,1,merges)+host*5+update(radix*h,merges));nodes=next;}
   local=std::max(local,cost);
  }total+=local;
   // The global tree is k-ary with arity `gradix`, the ACROSS-GPU carrier's c.
  // Same grouping as tiled_instantiate. A node of g children costs one TT over
  // g*h rows, g-1 publications each way, and ONE strip loop whose apply spans
  // g*h rows -- so raising g trades a wider per-episode apply against strictly
  // fewer episodes, which is the trade the measurement has to settle.
  for(int nodes=roots;nodes>1;){int next=0;
   for(int i=0;i<nodes;i+=gradix){int g=std::min(gradix,nodes-i);++next;if(g<2)continue;
    total+=add(4,service.get(tt_shared?"TT_shared":"TT",g*h,h,1,1,threads)+2*(g-1)*service.get("NV_publish",b*h,1,1,p,128)+host*5);
    // ONE ACROSS-GPU EPISODE PER PACKET, priced as the engine runs it. LL_reduce+LL_copy are
    // isolated collectives on an idle machine; the engine pays the episode ON THE CRITICAL PATH,
    // its barrier waiting for the slowest rank. LL_episode measures the pattern. The isolated pair
    // remains the fall-back for profiles taken before it existed, so old profiles stay selectable
    // and keep their old numbers.
    int full=trailing/strip,tail=trailing%strip;
    auto update=[&](int q){
     const double ep=service.get("LL_episode",b*q,1,1,p,128);
     const double transport=std::isfinite(ep)
       ? ep
       : service.get("LL_reduce",b*q,1,1,p,128)+service.get("LL_copy",b*q,1,1,p,128);
     return transport+service.get("apply",g*h,h,q,1)+host*2;};
    if(full)total+=full*update(strip);if(tail)total+=update(tail);total+=double(g-1)*service.get("NV_publish",0,1,1,p,128);}
   nodes=next;}
  col+=h;
 }return total+(m&&n?2*service.get("finite_scan",int64_t(ceildiv(m,p))*n,1,1,1,256):0)+host*2;
}
inline TiledPlan tiled_select_for_map(const RunOptions&o,int ranks,size_t word,const json&profile,int row_block){
 double start=seconds();TiledServices services(profile);double best=INFINITY;int bb=0,ll=0,ss=0,tt=0,brad=2,evaluated=0;
 // The selected depth and apply-carrier replication. Both default to the
 // first admitted menu point, never to 1.
 int bdepth=pipeline_depth_menu().front(),barep=apply_rep_menu().front(),bgrad=o.gradix?o.gradix:global_radix_menu(ranks).front();bool bgs=false,bts=false,bscalar=false,bsmall=false;json menu=json::array();size_t shared_capacity=profile.value("tiled_shared_capacity_bytes",size_t(0));
 auto values=[](int forced,std::initializer_list<int>defaults){return forced?std::vector<int>{forced}:std::vector<int>(defaults);};
 auto menu_values=[](int forced,std::vector<int>defaults){return forced?std::vector<int>{forced}:defaults;};
 // SELECTED JOINTLY, NOT IN TWO STAGES. Z's and D's carriers used to be weighed AFTER the shape, on
 // a shape whose whole menu had been priced at zc=dc=1. The fix is alternating minimisation to a
 // FIXED POINT: price the entire shape menu at the current (zc,dc) and block triple (wcb,zcb,dcb),
 // weigh (zc,dc) on the winner, weigh the block triple jointly on the same winner, repeat until
 // nothing moves. Only the final pass is recorded, so the menu that is reported IS the objective
 // that was minimised -- the property whose absence made every earlier menu ranking untrustworthy.
 // Both provisionals start at their menu fronts, never at 1, exactly as depth and apply_c do.
 int bzc=o.zc?o.zc:z_rep_menu().front(),bdc=o.dc?o.dc:d_rep_menu().front();
 // THE BLOCK-LEVEL APPLY CARRIERS, SELECTED. One CB per product, weighed in the same alternating
 // fixed-point pass that resolves (zc,dc): price the shape menu at the current triple, weigh the
 // triple on the winner, repeat until nothing moves. Starts at the menu fronts, never at 1. Costs
 // no HBM (register/shared partials), so no capacity fall-back exists for it -- only the forced arm
 // pins it. stable menu object (temporary-vector iterator fix). Calling menu().begin()/menu().end()
 // on two different temporaries is UB; the old code did exactly that for the forced block-carrier
 // check.
 const std::vector<int> block_menu_stable = block_rep_menu();
 for(int forced:{o.wcb,o.zcb,o.dcb})
  if(forced&&std::find(block_menu_stable.begin(),block_menu_stable.end(),forced)==block_menu_stable.end())
   throw std::runtime_error("unsupported_block_carrier_force");
 int bwcb=o.wcb?o.wcb:block_rep_menu().front(),bzcb=o.zcb?o.zcb:block_rep_menu().front(),bdcb=o.dcb?o.dcb:block_rep_menu().front();
 json zdmenu=json::array(),blockmenu=json::array();bool zc_capacity_fallback=false,dc_capacity_fallback=false,capacity_fallback=false;
 bool z_priced_selected=false,d_priced_selected=false;
 double z_taken_selected=0,z_refused_selected=0,d_taken_selected=0,d_refused_selected=0;
 const int strip_floor=services.min_sampled_apply_q();
 int passes=0,inert_refusals=0,bpackets=0;bool bdepth_structure=false;double menu_argmin=INFINITY;
 // Best predicted cost among admitted candidates that issue ONE packet at the
 // widest elimination, against the best among those that issue more. An r=1
 // schedule is admissible only if it WON that comparison, and the engine
 // has to carry both numbers to say so.
 double best_r1=INFINITY,best_rmulti=INFINITY;   // reset per pass, below
 for(int pass=0;pass<4;++pass){
 ++passes;const int pzc=bzc,pdc=bdc,pwb=bwcb,pzb=bzcb,pdb=bdcb;
 best=INFINITY;bb=0;ll=0;ss=0;tt=0;brad=2;evaluated=0;inert_refusals=0;bpackets=0;bdepth_structure=false;best_r1=INFINITY;best_rmulti=INFINITY;
 bdepth=pipeline_depth_menu().front();barep=apply_rep_menu().front();bgrad=o.gradix?o.gradix:global_radix_menu(ranks).front();
 bgs=false;bts=false;bscalar=false;bsmall=false;menu=json::array();
 for(int b:values(o.b,{16,32,64,128}))for(int leaf:menu_values(o.leaf,leaf_candidates(o.m,ranks)))for(int strip:menu_values(o.strip,strip_candidates(o.m,o.n,b,strip_floor))){
  if(leaf<b||(row_block&&(row_block<b||row_block%b)))continue;
  // r at the widest elimination. Recomputed per strip because it is what
  // decides whether a depth above one can do anything at all (Eq pipe).
  const int rwide=packets_at_widest_elimination(o.m,o.n,b,strip);
  // EQ PIPE'S OWN ADMISSION RULE. At r=1 the formula reads C(d)=t_Sigma for
  // every d: the depth is inert, it buys nothing, and it still reserves d
  // slots of W/Z credits. An inert depth counts as NOT met under the standing
  // rule, so where this strip leaves the widest elimination a single packet
  // the depth menu collapses to 1 -- a STRUCTURE fall-back, counted and named,
  // not a silent d>1 the engine cannot use. It touches only d: the apply, Z
  // and D carriers keep their own menus, because the width of a packet has
  // nothing to do with how a product cuts [K].
  std::vector<int> depths=menu_values(o.d,pipeline_depth_menu());
  bool depth_inert_here=false;
  if(!o.d&&rwide<2){depth_inert_here=true;inert_refusals+=int(depths.size());depths.assign(1,1);}
  else if(!o.d){
   // A depth above r cannot admit a packet that does not exist -- it reserves d slots of W/Z
   // credits and leaves d-r of them idle. Refused and counted, not ranked on a tie the cost model
   // cannot see.
   const size_t before=depths.size();
   depths.erase(std::remove_if(depths.begin(),depths.end(),[&](int d){return d>rwide;}),depths.end());
   inert_refusals+=int(before-depths.size());
   if(depths.empty())depths.assign(1,std::min(rwide,pipeline_depth_menu().front()));
  }
  for(int radix:values(o.radix,{2,4,8})){
  // The ACROSS-GPU carrier joins them as a third menu dimension: the arity of the global TT tree is
  // the c of the cross-rank W reduce, so it is priced and chosen here rather than fixed at 2 by the
  // shape of the tree.
  for(int depth:depths)for(int arep:menu_values(o.c,apply_rep_menu()))for(int grad:menu_values(o.gradix,global_radix_menu(ranks))){
   auto inventory=tiled_inventory(o.m,o.n,ranks,b,leaf,strip,word,o.pad,radix,depth,arep,grad,pzc,pdc,1,0,row_block);size_t owned=*std::max_element(inventory.owned.begin(),inventory.owned.end());
  for(int threads:values(o.threads,{128,256,512,1024}))for(bool ge_shared:{false,true})for(bool tt_shared:{false,true}){
   if((ge_shared&&size_t(std::min(leaf,ceildiv(o.m,ranks)))*b*word>shared_capacity)||(tt_shared&&size_t(std::max(radix,ranks>1?grad:2))*b*b*word>shared_capacity))continue;
   size_t total=checked_add(owned,checked_add(backend_arena_budget(std::max(size_t(b)*b,size_t(b)*strip)*word,ranks,profile),ranks>1?profile.value("library_overhead_allowance_bytes",size_t(0)):0));bool fits=total<=o.budget;double host=profile.value("host_enqueue_s",0.0);
   double cost=fits?tiled_model(o.m,o.n,ranks,b,leaf,strip,threads,services,host,ge_shared,tt_shared,radix,shared_capacity,word,depth,arep,grad,pzc,pdc,pwb,pzb,pdb,nullptr,row_block):INFINITY;
   if(fits)++evaluated;
   menu.push_back({{"b",b},{"leaf",leaf},{"strip",strip},{"threads",threads},{"radix",radix},{"global_radix",grad},{"depth",depth},{"apply_c",arep},{"Z_c",pzc},{"D_c",pdc},{"W_b",pwb},{"Z_b",pzb},{"D_b",pdb},{"packets_per_elimination",rwide},{"depth_structure_fallback",depth_inert_here},{"ge_shared",ge_shared},{"tt_shared",tt_shared},{"predicted_s",std::isfinite(cost)?json(cost):json(nullptr)},{"owned_bytes",total},{"fits",fits}});
   if(fits){if(rwide<=1)best_r1=std::min(best_r1,cost);else best_rmulti=std::min(best_rmulti,cost);}
   if(fits&&cost<best){best=cost;bb=b;ll=leaf;ss=strip;tt=threads;bgs=ge_shared;bts=tt_shared;brad=radix;bdepth=depth;barep=arep;bgrad=grad;bpackets=rwide;bdepth_structure=depth_inert_here;}
  }}}
 }
 if(ranks==1&&o.m<=128&&o.n<=128&&!o.b&&!o.leaf&&!o.strip&&profile.contains("scalar_packet_catalog"))for(int threads:values(o.threads,{1,32,128,256}))for(bool shared:{false,true}){
  if(shared&&(threads<32||!profile.contains("small_packet_shared_capacity_bytes")||size_t(o.m)*o.n*word>profile.at("small_packet_shared_capacity_bytes").get<size_t>()))continue;
  auto candidate=tiled_scalar_instantiate(o.m,o.n,threads,word,o.pad);bool fits=candidate.owned[0]<=o.budget;double cost=(!o.m||!o.n)?0:services.get("scalar_fused",o.m,std::min(o.m,o.n),o.n,1,threads)+profile.value("scalar_status_host_s",0.0)+profile.value("host_enqueue_s",0.0);
  if(shared&&o.m&&o.n)cost=services.get("small_shared",o.m,std::min(o.m,o.n),o.n,1,threads)+profile.value("scalar_status_host_s",0.0)+profile.value("host_enqueue_s",0.0);
  menu.push_back({{"scalar",true},{"small_shared",shared},{"threads",threads},{"owned_bytes",candidate.owned[0]},{"fits",fits},{"predicted_s",cost}});++evaluated;if(fits&&cost<best){best=cost;bscalar=true;bsmall=shared;tt=threads;}
 }
  // CAPACITY FALL-BACK. Reached only when every >1 (depth, apply_c) candidate exceeds the budget
  // and the scalar path is unavailable; skipped for forced arms, where an infeasible shape must
  // error rather than change meaning. The unreplicated products then run the native kernels, never
  // cuBLAS.
  capacity_fallback=false;
  if(!bb&&!bscalar&&!o.d&&!o.c){
   for(int b:values(o.b,{16,32,64,128}))for(int leaf:menu_values(o.leaf,leaf_candidates(o.m,ranks)))// The fall-back may also take the single-packet width, which the derived menu
  for(int strip:menu_values(o.strip,[&]{auto v=strip_candidates(o.m,o.n,b,strip_floor);int full=strip_for_packets(o.m,o.n,b,1);if(full>0&&std::find(v.begin(),v.end(),full)==v.end())v.push_back(full);return v;}())){
    if(leaf<b||(row_block&&(row_block<b||row_block%b)))continue;
    for(int radix:values(o.radix,{2,4,8}))for(int grad:menu_values(o.gradix,global_radix_menu(ranks))){
     auto inventory=tiled_inventory(o.m,o.n,ranks,b,leaf,strip,word,o.pad,radix,1,1,grad,1,1,1,0,row_block);size_t owned=*std::max_element(inventory.owned.begin(),inventory.owned.end());
    for(int threads:values(o.threads,{128,256,512,1024}))for(bool ge_shared:{false,true})for(bool tt_shared:{false,true}){
     if((ge_shared&&size_t(std::min(leaf,ceildiv(o.m,ranks)))*b*word>shared_capacity)||(tt_shared&&size_t(std::max(radix,ranks>1?grad:2))*b*b*word>shared_capacity))continue;
     size_t total=checked_add(owned,checked_add(backend_arena_budget(std::max(size_t(b)*b,size_t(b)*strip)*word,ranks,profile),ranks>1?profile.value("library_overhead_allowance_bytes",size_t(0)):0));bool fits=total<=o.budget;double host=profile.value("host_enqueue_s",0.0);
     double cost=fits?tiled_model(o.m,o.n,ranks,b,leaf,strip,threads,services,host,ge_shared,tt_shared,radix,shared_capacity,word,1,1,grad,1,1,block_rep_menu().front(),block_rep_menu().front(),block_rep_menu().front(),nullptr,row_block):INFINITY;
     if(fits)++evaluated;
     menu.push_back({{"b",b},{"leaf",leaf},{"strip",strip},{"threads",threads},{"radix",radix},{"global_radix",grad},{"depth",1},{"apply_c",1},{"Z_c",1},{"D_c",1},{"ge_shared",ge_shared},{"tt_shared",tt_shared},{"predicted_s",std::isfinite(cost)?json(cost):json(nullptr)},{"owned_bytes",total},{"fits",fits},{"role","capacity fall-back: no c>1/d>1 candidate fits"}});
     if(fits&&cost<best){best=cost;bb=b;ll=leaf;ss=strip;tt=threads;bgs=ge_shared;bts=tt_shared;brad=radix;bdepth=1;barep=1;bgrad=grad;bpackets=packets_at_widest_elimination(o.m,o.n,b,strip);bdepth_structure=false;capacity_fallback=true;}
    }}}
  }
  if(!bb&&!bscalar)throw std::runtime_error("no_profiled_feasible_tiled_candidate");
  // Z'S AND D'S OWN CARRIERS, WEIGHED (Alg 2; Every >1 candidate is priced with the full model,
  // capacity included; c=1 is priced alongside as the capacity-fallback reference row and is
  // SELECTED only when no >1 candidate is finite (capacity) or the forcing flag names it. The whole
  // table is recorded in selection.zd_carrier.
  zdmenu=json::array();zc_capacity_fallback=false;dc_capacity_fallback=false;
  // The argmin of the menu just priced, kept so the record can show that the
  // number reported IS the number minimised (objective_gap_s below).
  menu_argmin=best;
  if(capacity_fallback){
   // The shape search has just reported that nothing with c>1 or d>1 fits, so Z and D go to 1 with
   // the rest and the next pass reprices the whole menu there -- one cost function, still.
   bzc=1;bdc=1;zc_capacity_fallback=true;dc_capacity_fallback=true;
   zdmenu.push_back({{"product","Z"},{"c",1},{"role","capacity fall-back: no c>1/d>1 candidate fits"}});
   zdmenu.push_back({{"product","D"},{"c",1},{"role","capacity fall-back: no c>1/d>1 candidate fits"}});
  }else if(bscalar){bzc=1;bdc=1;}
  else{
   auto price=[&](int zc,int dc,int wb,int zb,int db)->double{
     auto inv=tiled_inventory(o.m,o.n,ranks,bb,ll,ss,word,o.pad,brad,bdepth,barep,bgrad,zc,dc,1,0,row_block);
    size_t owned=*std::max_element(inv.owned.begin(),inv.owned.end());
    size_t total=checked_add(owned,checked_add(backend_arena_budget(std::max(size_t(bb)*bb,size_t(bb)*ss)*word,ranks,profile),ranks>1?profile.value("library_overhead_allowance_bytes",size_t(0)):0));
    if(total>o.budget)return INFINITY;
    return tiled_model(o.m,o.n,ranks,bb,ll,ss,tt,services,profile.value("host_enqueue_s",0.0),bgs,bts,brad,shared_capacity,word,bdepth,barep,bgrad,zc,dc,wb,zb,db,nullptr,row_block);};
   // THE PAIR IS WEIGHED JOINTLY, not Z first and then D. The grid is 3x3 = nine tiled_model calls
   // on the selected shape, which is free next to the 16 800-entry shape menu, so there is no reason
   // to separate them.
   const std::vector<int> zmenu=menu_values(o.zc,z_rep_menu()),dmenu=menu_values(o.dc,d_rep_menu());
   double bestpair=INFINITY,bestone=INFINITY,bestnone=INFINITY;int oz=1,od=1,nz=1,nd=1;
   for(int zc:std::vector<int>{1}) for(int dc:std::vector<int>{1}){
    double t=price(zc,dc,bwcb,bzcb,bdcb);
    zdmenu.push_back({{"product","ZxD"},{"Z_c",zc},{"D_c",dc},{"predicted_s",std::isfinite(t)?json(t):json(nullptr)},{"role","capacity-fallback reference: admissible only where no c>1 fits for either product"}});
    if(t<bestnone){bestnone=t;nz=zc;nd=dc;}}
   for(int zc:zmenu){double t=price(zc,1,bwcb,bzcb,bdcb);
    zdmenu.push_back({{"product","ZxD"},{"Z_c",zc},{"D_c",1},{"predicted_s",std::isfinite(t)?json(t):json(nullptr)},{"role","partial capacity fall-back: D unreplicated"}});
    if(t<bestone){bestone=t;oz=zc;od=1;}}
   for(int dc:dmenu){double t=price(1,dc,bwcb,bzcb,bdcb);
    zdmenu.push_back({{"product","ZxD"},{"Z_c",1},{"D_c",dc},{"predicted_s",std::isfinite(t)?json(t):json(nullptr)},{"role","partial capacity fall-back: Z unreplicated"}});
    if(t<bestone){bestone=t;oz=1;od=dc;}}
   for(int zc:zmenu)for(int dc:dmenu){double t=price(zc,dc,bwcb,bzcb,bdcb);
    zdmenu.push_back({{"product","ZxD"},{"Z_c",zc},{"D_c",dc},{"predicted_s",std::isfinite(t)?json(t):json(nullptr)}});
    if(t<bestpair){bestpair=t;bzc=zc;bdc=dc;}}
   if(std::isfinite(bestpair)){
    // Ties stay with bestpair (status quo).
    if((bestone<bestpair&&bestone<=bestnone)||(bestnone<bestpair&&bestnone<bestone)){
     if(bestnone<=bestone){bzc=1;bdc=1;
      z_priced_selected=true;d_priced_selected=true;
      z_taken_selected=bestnone;z_refused_selected=bestpair;
      d_taken_selected=bestnone;d_refused_selected=bestpair;}
     else if(od==1){bzc=oz;bdc=1;d_priced_selected=true;
      d_taken_selected=bestone;d_refused_selected=bestpair;}
     else{bzc=1;bdc=od;z_priced_selected=true;
      z_taken_selected=bestone;z_refused_selected=bestpair;}
    }
   }
   if(!std::isfinite(bestpair)){
    // Nothing with both replicated fits. Take the best pair with exactly one
    // at 1, then the unreplicated pair, and flag which product capacity ruled.
    if(std::isfinite(bestone)){bzc=oz;bdc=od;}else{bzc=nz;bdc=nd;}
    zc_capacity_fallback=(bzc==1);dc_capacity_fallback=(bdc==1);
   }
   if(o.zc)bzc=o.zc;if(o.dc)bdc=o.dc;
   // THE BLOCK TRIPLE, WEIGHED JOINTLY ON THE SELECTED SHAPE. The 27 triples are full-model prices
   // at the just-weighed (zc,dc) -- the t_max coupling stays inside the number, so no product is
   // chosen on another product's measurement. No capacity roles: CB costs no HBM, so every triple
   // is admissible and the argmin is pure time. An unsampled triple prices non-finite and is never
   // taken. Forced arms pin their axis.
   {
    const std::vector<int> wmenu=menu_values(o.wcb,block_rep_menu()),
     zmenu_b=menu_values(o.zcb,block_rep_menu()),dmenu_b=menu_values(o.dcb,block_rep_menu());
    double bestblock=INFINITY;int ow=2,ozb=2,odb=2;bool first=true;
    for(int wb:wmenu)for(int zb:zmenu_b)for(int db:dmenu_b){
     double t=price(bzc,bdc,wb,zb,db);
     blockmenu.push_back({{"product","WxZxD"},{"W_b",wb},{"Z_b",zb},{"D_b",db},
      {"predicted_s",std::isfinite(t)?json(t):json(nullptr)}});
     if(std::isfinite(t)&&(first||t<bestblock)){bestblock=t;ow=wb;ozb=zb;odb=db;first=false;}}
    if(!first){bwcb=ow;bzcb=ozb;bdcb=odb;}
    if(o.wcb)bwcb=o.wcb;if(o.zcb)bzcb=o.zcb;if(o.dcb)bdcb=o.dcb;
   }
   double chosen=price(bzc,bdc,bwcb,bzcb,bdcb);if(std::isfinite(chosen))best=chosen;
  }
  // FIXED POINT.
  if(bzc==pzc&&bdc==pdc&&bwcb==pwb&&bzcb==pzb&&bdcb==pdb)break;
 }
 auto plan=bscalar?tiled_scalar_instantiate(o.m,o.n,tt,word,o.pad):tiled_instantiate(o.m,o.n,ranks,bb,ll,ss,tt,word,o.pad,brad,bdepth,barep,bgrad,bzc,bdc,bwcb,bzcb,bdcb,1,0,row_block);plan.small_shared=bsmall;plan.ge_shared=bscalar?false:bgs;plan.tt_shared=bscalar?false:bts;
 if(!bscalar){plan.z_priced=z_priced_selected;plan.d_priced=d_priced_selected;
  plan.z_taken_s=z_taken_selected;plan.z_refused_s=z_refused_selected;
  plan.d_taken_s=d_taken_selected;plan.d_refused_s=d_refused_selected;}
 tiled_charge_backend(plan,word,profile);
 if(*std::max_element(plan.owned.begin(),plan.owned.end())>o.budget)throw std::runtime_error("exact_tiled_inventory_before_modify");
 // Record the carrier the model just priced, per batch, so the engine dispatches exactly what was
 // costed. The panel carrier's staged column window.
 plan.csw_forced=o.csw;plan.csw=0;
 plan.forcing={{"b",o.b},{"leaf",o.leaf},{"c",o.c},{"d",o.d},{"threads",o.threads},{"strip",o.strip},{"pad",o.pad},{"radix",o.radix},{"gradix",o.gradix},{"zc",o.zc},{"dc",o.dc},{"wcb",o.wcb},{"zcb",o.zcb},{"dcb",o.dcb},{"csw",o.csw},{"tree",o.tree},{"active",o.active}};
 if(!bscalar)tiled_carry_panel_carriers(plan,services,shared_capacity,word);
 // THE MERGE-STACK ELIMINATION'S CARRIER, PRICED AND REFUSED. TT_GE never executes c>1 -- the
 // witness reports 0% at every rank count -- and the refusal has TWO parts, one structural and one
 // numeric, so both are recorded rather than asserted. Structural: the block-level carrier
 // (cooperative_ge) is a GE. A merge stack is [R_0; R_1; ...], a stack of upper triangles, and the
 // TT kernel both exploits that structure and lays V/T out the way tile_scatter_factors reads them;
 // a general GE over the same stack is the same QR mathematically but discards the structure and
 // the layout. So it is not a carrier for THIS product, it is a different kernel. Numeric: the two
 // costs at the actual stack shape are recorded below, so the refusal carries the numbers even
 // though the structural half already settles it.
 json ttcarrier=json::array();
 if(!bscalar)for(auto node:std::vector<std::pair<int,std::string>>{{brad,"local_TT"},{ranks>1?bgrad:0,"global_TT"}}){
  if(node.first<2)continue;const int srows=node.first*bb;
  const double blk=services.get(bts?"TT_shared":"TT",srows,bb,1,1,tt);
  const CoopChoice pick=coop_weigh(services,srows,bb,shared_capacity,word);
  ttcarrier.push_back({{"node",node.second},{"stack_rows",srows},{"h",bb},{"arity",node.first},
   {"block_TT_s",std::isfinite(blk)?json(blk):json(nullptr)},
   {"cooperative_GE_s",std::isfinite(pick.cost)?json(pick.cost):json(nullptr)},
   {"cooperative_c",pick.c},{"cooperative_minipanel_width",pick.pw},
   {"carrier_would_win",std::isfinite(blk)&&std::isfinite(pick.cost)&&pick.cost<blk}});
 }
 // One extra tiled_model call.
 json breakdown=nullptr;
 if(!bscalar){double tv[5]={0,0,0,0,0};
  const double whole=tiled_model(o.m,o.n,ranks,bb,ll,ss,tt,services,profile.value("host_enqueue_s",0.0),bgs,bts,brad,shared_capacity,word,bdepth,barep,bgrad,bzc,bdc,bwcb,bzcb,bdcb,tv,row_block);
  breakdown={{"setup_s",tv[0]},{"panel_s",tv[1]},{"leaf_apply_s",tv[2]},{"local_tree_s",tv[3]},{"global_tree_s",tv[4]},{"total_s",whole},
   {"note","exact at p=1; at p>1 the per-column cost is a maximum over ranks so the split is indicative. leaf_apply_s is Eq apply over the trailing strips of the leaf packets; local_tree_s includes the tree levels' own applies"}};}
 // NUMBERS FOR THE REFUSAL CERTIFICATES (execution_format 24).
 plan.budget_bytes=o.budget;
 plan.strip_floor_q=strip_floor;
 plan.widest_packets=bpackets;
 plan.depth_price_r1_s=std::isfinite(best_r1)?best_r1:0.0;
 plan.depth_price_rmulti_s=std::isfinite(best_rmulti)?best_rmulti:0.0;
 if(!bscalar){
  // (1) The merge-stack elimination. The block-level carrier is a GE and a
  for(auto&row:ttcarrier)if(row.value("node",std::string())=="local_TT"){
   if(row["block_TT_s"].is_number())plan.tt_block_s=row["block_TT_s"].get<double>();
   if(row["cooperative_GE_s"].is_number())plan.tt_coop_s=row["cooperative_GE_s"].get<double>();}
  // (2) The fused pentagonal merge apply, which does W, Z and D in ONE kernel
  // over the support rows and therefore carries no [K] cut at all.
  const int mrows=brad*bb,mq=std::min(ss,std::max(1,o.n-bb));
  plan.merge_fused_s=services.get("apply",mrows,bb,mq,1);
  // The carried sequence at the SELECTED carriers, GPU-level and block-level
  // alike, so the R_FUSED_MERGE refusal prices the kernels that would run.
  {const int wr=(barep>1&&mrows>=barep)?barep:1,zr=(bzc>1&&bb>=bzc)?bzc:1,dr=(bdc>1&&bb>=bdc)?bdc:1;
   const double tw=wr>1?services.get(apply_rep_op(wr),mrows,bb,mq,1):services.get(apply_w_block_op(bwcb),mrows,bb,mq,1);
   const double tz=zr>1?services.get(apply_z_rep_op(zr),mrows,bb,mq,1):services.get(apply_z_block_op(bzcb),mrows,bb,mq,1);
   const double td=dr>1?services.get(apply_d_rep_op(dr),mrows,bb,mq,1):services.get(apply_d_block_op(bdcb),mrows,bb,mq,1);
   if(std::isfinite(tw)&&std::isfinite(tz)&&std::isfinite(td))plan.merge_carried_s=tw+tz+td;}
  if(!std::isfinite(plan.merge_fused_s))plan.merge_fused_s=0;
 }
 plan.profile_id=profile.at("id");plan.selection={{"status","completed_predictive_menu"},{"term_breakdown",breakdown},{"tt_carrier",{{"products",ttcarrier},{"structural_refusal","the block-level carrier is a GE; a merge stack is a stack of upper triangles whose structure the TT kernel exploits and whose V/T layout tile_scatter_factors depends on, so the cooperative GE is a different kernel for this product rather than a carrier for it"},{"numeric_note","block_TT_s and cooperative_GE_s are measured at the actual stack shape (arity*b rows, h=b) so the refusal carries numbers as well as structure"}}},{"evaluated",evaluated},{"selection_s",seconds()-start},{"predicted_s",best},{"model_gap",nullptr},{"physical_bound",nullptr},{"menu",menu},{"objective","reused ready-input through R/native-Q completion; first-use allocation/planning reported separately"},{"cost_scope","local rank maxima; ordered global TT and LL W joins; setup/pack/commit service and host kernel submission; interpolation remains heuristic"},{"selected_depth",bdepth},{"selected_apply_replication",barep},{"selected_global_radix",bgrad},{"selected_Z_replication",bzc},{"selected_D_replication",bdc},{"selected_W_block_carrier",bwcb},{"selected_Z_block_carrier",bzcb},{"selected_D_block_carrier",bdcb},{"capacity_fallback",capacity_fallback},{"selection_passes",passes},{"depth_inert_refusals",inert_refusals},{"strip_floor_q",strip_floor},{"strip_menu_basis","the strip is chosen as a packet count r (packets_menu) and the width derived from it, floored at the narrowest packet the profile measured and capped at the descriptor's maximum. Widths below the floor have no measured anchor: the service extrapolates them linearly in q and clamps only at the launch latency, which prices a tiny packet as nearly free"},{"leaf_menu",leaf_candidates(o.m,ranks)},{"depth_refusal_reasons","counts two Sec 3 refusals together: a depth above one where the strip leaves the widest elimination a single packet (C(d)=t_Sigma for every d), and a depth above r, which reserves slots for packets that do not exist"},{"selected_packets_per_elimination",bpackets},{"depth_structure_fallback",bdepth_structure},{"selected_depth_reason",bdepth_structure?"d=1: the selected strip issues a single packet, so Eq pipe gives C(d)=t_Sigma for every depth. Counted in depth_inert_refusals; the >1 depths were priced against it and lost":"selected among strictly >1 candidates by measurement, capped at r (Sec 3: depths up to the number of packets)"},{"depth_admission","Sec 3 Eq pipe: a (strip,depth>1) pair whose widest elimination issues r=1 packets is REFUSED and counted in depth_inert_refusals, because C(d)=t_Sigma at r=1 makes the depth inert while still reserving d slots of W/Z credits. Every admitted row carries the packets_per_elimination it was priced at; the engine's carrier_dispatch witness reports the r it actually issued"},{"objective_gap_s",(std::isfinite(best)&&std::isfinite(menu_argmin))?json(std::fabs(best-menu_argmin)):json(nullptr)},{"joint_selection","Sec 4.4: b, the panel widths, the tree at each level, EVERY carrier (panel, apply W, Z, D, across-GPU) and the depth are selected jointly. The shape menu, the (Z,D) carriers and the (W,Z,D) block carriers are alternated to a fixed point, so the recorded menu is the objective that was minimised: objective_gap_s is |predicted_s - argmin(menu)| and must read 0"},{"Z_capacity_fallback",zc_capacity_fallback},{"D_capacity_fallback",dc_capacity_fallback},{"Z_priced",z_priced_selected},{"D_priced",d_priced_selected},{"Z_refused_s",z_refused_selected},{"Z_taken_s",z_taken_selected},{"D_refused_s",d_refused_selected},{"D_taken_s",d_taken_selected},{"block_carrier",{{"menu",blockmenu},{"W_menu",block_rep_menu()},{"Z_menu",block_rep_menu()},{"D_menu",block_rep_menu()},{"basis","Sec 3 block level, one CB per Eq-apply product, each cutting its own [K] (rows for W, reflectors for Z and D) with register partials summed in shared scratch. The 27 triples are weighed jointly by the full model on the selected shape; t_max stays inside the number, while overlap off the reference block arm is priced serially (the pace sample names only the GPU-level carrier, so an unmatched arm gets no overlap credit). CB costs no HBM partials, so every triple is admissible and the argmin is pure time; an unsampled triple is never taken"}}},{"zd_carrier",{{"menu",zdmenu},{"Z_menu",z_rep_menu()},{"D_menu",d_rep_menu()},{"basis","Alg 2 gives Z and D their own carriers; [K] for both is the reflector index h. Z cuts it into disjoint reflector ranges whose partials are summed (zc partials of a b x strip block, charged in the inventory); D cuts the same index and accumulates each peer into X in place. Candidates above one are selected from measured latency and bandwidth; c=1 is the capacity-fallback reference, admissible only where no c>1 fits, and a priced candidate (Z_priced/D_priced with both times) where it measures faster than every c>1 pair"}}},{"depth_menu",pipeline_depth_menu()},{"apply_replication_menu",apply_rep_menu()},{"global_radix_menu",global_radix_menu(ranks)},{"across_gpu_carrier","the global TT tree arity IS the across-GPU contraction replication c: a node of k ranks reduces W = sum_i V_i^T X_i over k disjoint row sets {Combine: additive}. Selected from the measured LL_reduce/LL_copy/NV_publish/apply samples over a menu whose smallest candidate is 2, never fixed by the shape of the tree"},{"depth_model","Sec 3 Eq pipe over the strips of one elimination: r packets, three measured stages (W, Z, D), C(d)=t_sum+(r-1)*pace(d,c) where pace is Eq pipe t_max MEASURED at that depth (R=8 packets through the real slot machinery) and clamped into [t_max,t_sum]; a shape with no pace sample is credited with nothing (pace=t_sum, serial). Capacity charged per candidate through the d x c W/Z buffer credits"},{"admitted_member","tiled extension with ordinary CUDA launches, a selected local tree radix, a panel-product carrier with contraction replication c CHOSEN from measurement among >1 candidates plus measured minipanel width and thread count, an apply-product carrier whose replication is CHOSEN from measurement among >1 candidates, and a pipeline depth d CHOSEN from measurement among >1 candidates by Eq pipe (standing user order: never 1, never fixed)"}};plan.selection["kernel_resources"]=services.coop_res.present?json("per-kernel shared capacity and residency from the profile (plan P0.2)"):json("absent: legacy single tiled_ge capacity (profile predates P0.2; engine enforcement refuses a stamped carrier its launcher rejects)");plan.selection["coop_weigh_memo"]={{"calls",services.coop_calls},{"hits",services.coop_hits},{"unique",services.coop_choices.size()}};plan.key=digest(plan.record().dump());return plan;
}
// Missing measurements are unavailable, never modeled as zero-cost traffic.
inline TiledPlan tiled_select(const RunOptions&o,int ranks,size_t word,const json&profile){
 auto best=tiled_select_for_map(o,ranks,word,profile,0);
 double cost=best.selection.at("predicted_s");
 json menu=json::array({{{"kind","contiguous"},{"blk",0},{"predicted_s",cost},{"redistribution_s",0.0}}});
 std::set<int> seen;
 if(ranks>1)for(const auto&sample:profile.value("row_distribution_samples",json::array())){
  if(sample.value("m",-1)!=o.m||sample.value("n",-1)!=o.n||sample.value("p",-1)!=ranks||sample.value("word",0)!=int(word))continue;
  const int blk=sample.value("blk",0);const double rd=sample.value("median_s",-1.0);
  if(blk<1||!std::isfinite(rd)||rd<=0||!sample.value("validated",false)||sample.value("scope",std::string())!="slab_to_cyclic_and_back"||!seen.insert(blk).second)
   throw std::runtime_error("invalid_row_distribution_sample");
  try{
   auto candidate=tiled_select_for_map(o,ranks,word,profile,blk);
   const double compute=candidate.selection.at("predicted_s");const double total=compute+rd;
   menu.push_back({{"kind","block_cyclic"},{"blk",blk},{"predicted_s",total},{"geometry_s",compute},{"redistribution_s",rd}});
   if(total<cost){cost=total;best=std::move(candidate);best.selection["predicted_s"]=total;best.selection["redistribution_s"]=rd;}
  }catch(const std::runtime_error&e){menu.push_back({{"kind","block_cyclic"},{"blk",blk},{"refused",e.what()}});}
 }
 best.selection["row_distribution_menu"]=menu;
 best.selection["row_distribution_policy"]="minimum measured-service geometry cost plus matching complete redistribution; absent cyclic samples are unavailable";
 best.key=digest(best.record().dump());return best;
}

}
