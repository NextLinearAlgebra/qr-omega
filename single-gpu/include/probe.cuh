#pragma once
#include "validation.cuh"
#include <set>
#include <cuda.h>
namespace tqr {
inline std::string file_key(const std::string& path){
 std::ifstream f(path,std::ios::binary);uint64_t h=1469598103934665603ULL;char b[65536];while(f){f.read(b,sizeof(b));for(std::streamsize i=0;i<f.gcount();++i){h^=(unsigned char)b[i];h*=1099511628211ULL;}}return std::to_string(h);
}
inline std::string binary_key(){return file_key("/proc/self/exe");}
inline json discover(Context&ctx){
 cudaDeviceProp p;CU(cudaGetDeviceProperties(&p,ctx.device));size_t free,total;CU(cudaMemGetInfo(&free,&total));int rt,dr;CU(cudaRuntimeGetVersion(&rt));CU(cudaDriverGetVersion(&dr));
 json j={{"schema",1},{"mapping",ctx.mapping},{"ranks",ctx.size},{"binary_key",binary_key()},{"catalog",catalog_id},{"driver_api",dr},{"runtime",rt},{"cc",{p.major,p.minor}},{"sm_count",p.multiProcessorCount},{"warp_size",p.warpSize},{"registers_per_sm",p.regsPerMultiprocessor},{"shared_per_sm",p.sharedMemPerMultiprocessor},{"shared_per_block",p.sharedMemPerBlockOptin},{"l2_bytes",p.l2CacheSize},{"hbm_total_bytes",total},{"hbm_free_observed_bytes",free},{"managed_L2",false},{"numerical_offload",false},{"tf32",false}};
 std::ifstream proc_status("/proc/self/status");std::string cpu_line;while(std::getline(proc_status,cpu_line))if(cpu_line.rfind("Cpus_allowed_list:",0)==0){auto v=cpu_line.substr(cpu_line.find(':')+1);v.erase(0,v.find_first_not_of(" \t"));j["cpu_affinity"]=v;}
 j["measurement_class"]=std::getenv("TQR_MEASUREMENT_CLASS")?std::getenv("TQR_MEASUREMENT_CLASS"):"exclusive-pinned";
 j["host_worker_pools"]={{"OMP_NUM_THREADS",std::getenv("OMP_NUM_THREADS")?std::getenv("OMP_NUM_THREADS"):"unset"},{"MKL_NUM_THREADS",std::getenv("MKL_NUM_THREADS")?std::getenv("MKL_NUM_THREADS"):"unset"}};
 int count;CU(cudaGetDeviceCount(&count));j["peer_access"]=json::array();for(int d=0;d<count;++d){int a=1;if(d!=ctx.device)CU(cudaDeviceCanAccessPeer(&a,ctx.device,d));j["peer_access"].push_back(a);}
 std::ifstream maps("/proc/self/maps");std::set<std::string> libs;std::string line;while(std::getline(maps,line)){auto pos=line.find('/');if(pos!=std::string::npos&&(line.find("libnccl")!=std::string::npos||line.find("libnvshmem")!=std::string::npos||line.find("libcusolverMp")!=std::string::npos))libs.insert(line.substr(pos));}j["loaded_libraries"]=libs;json libkeys=json::object();for(auto&lib:libs)libkeys[lib]=file_key(lib);j["loaded_library_keys"]=libkeys;
 j["backend_policy"]="LL flags/custom-unicast; host reference Ring/Simple; NVSHMEM64M; no numerical offload";
 CUmemAllocationProp mp{};mp.type=CU_MEM_ALLOCATION_TYPE_PINNED;mp.location.type=CU_MEM_LOCATION_TYPE_DEVICE;mp.location.id=ctx.device;int fabric=0,rdma=0;cuDeviceGetAttribute(&fabric,CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,ctx.device);cuDeviceGetAttribute(&rdma,CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,ctx.device);mp.requestedHandleTypes=(CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR|(fabric?CU_MEM_HANDLE_TYPE_FABRIC:0));mp.allocFlags.gpuDirectRDMACapable=rdma?1:0;size_t gran=0;if(cuMemGetAllocationGranularity(&gran,&mp,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)!=CUDA_SUCCESS)throw std::runtime_error("VMM_granularity_query");j["nccl_vmm_granularity"]=gran;
 j["group_mapping"]=ctx.all_records(ctx.mapping);
 j["stable_key"]=digest(json{{"mapping",j["group_mapping"]},{"driver",dr},{"runtime",rt},{"binary",j["binary_key"]},{"catalog",catalog_id},{"libraries",libkeys},{"policy",j["backend_policy"]},{"ranks",ctx.size},{"cc",j["cc"]},{"tf32",false},{"cpu_affinity",j["cpu_affinity"]},{"measurement_class",j["measurement_class"]},{"host_worker_pools",j["host_worker_pools"]}}.dump());return j;
}
template<class F> std::vector<double> time_primitive(F f,cudaStream_t stream,int reps=15){
 for(int i=0;i<3;++i)f();CU(cudaStreamSynchronize(stream));std::vector<double> v;Event begin,end;
 for(int i=0;i<reps;++i){begin.record(stream);f();end.record(stream);CU(cudaEventSynchronize(end.e));float ms;CU(cudaEventElapsedTime(&ms,begin.e,end.e));v.push_back(ms*1e-3);}return v;
}
inline double median(std::vector<double> v){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[v.size()/2];}
template<class T> __global__ void structural_input(T*a,int s,int h,int kind){for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<s*h;i+=blockDim.x*gridDim.x){int r=i%s,j=i/s;T v=T(int(mix64(i+17)%2001)-1000)/T(1000);if(kind==1&&r<h&&r>j)v=0;if(kind==2&&r%h>j)v=0;a[i]=v;}}
template<class T> json probe(Context&ctx,const json& hardware){
 double start=seconds();json p={{"schema",1},{"precision",precision<T>()},{"hardware",hardware},{"catalog",catalog_id},{"samples",json::array()},{"bounds","unknown; measured predictions only"},{"cache_preparation","fresh primitive input then warm repeats; no residency guarantee"},{"timing","single primitive CUDA event interval; input setup excluded, output and completion included"}};
 Stream st;Buffer<T>a(256*256),saved(256*256),v(256*32),tri(32*32),x(256*256),w(32*256),z(32*256),scales(256),norm(256),partials(32*256*4);Buffer<int> status(1);Buffer<Witness>wit(1);status.zero(st);wit.zero(st);
 make_input<<<128,128,0,st>>>(saved.p,256,256,256,0,256,0,T(1));st.sync();
 auto save=[&](std::string op,int s,int h,int q,int threads,const std::vector<double>&t){p["samples"].push_back({{"op",op},{"s",s},{"h",h},{"q",q},{"threads",threads},{"raw_s",t},{"median_s",median(t)},{"evidence_class","hardware observation"}});};
 p["launch_s"]=median(time_primitive([&](){empty_kernel<<<1,1,0,st>>>();},st));
 // THE MARGINAL COST OF ONE MORE KERNEL ON A STREAM THE HOST IS ALREADY AHEAD OF. `launch_s` above
 // is a round trip -- issue one kernel and wait for it -- and the model used it as the floor under
 // every interpolated estimate, summed serially.
 {const int K=256;
  const double many=median(time_primitive([&](){for(int i=0;i<K;++i)empty_kernel<<<1,1,0,st>>>();},st));
  p["gpu_launch_gap_s"]=std::max(0.0,(many-p["launch_s"].get<double>())/double(K-1));
  p["gpu_launch_gap_basis"]="slope of K=256 empty kernels issued back to back on one stream, measured with the same CUDA-event interval as every other primitive; this is the floor an ASYNCHRONOUS launch deserves, launch_s is the floor a SYNCHRONOUS one deserves";}
 for(int threads:{1,32,128,256})for(int s:{1,8,16,32,64,128,256})for(int h:{1,8,16,32}){
  if(h>s)continue;
  for(int kind=0;kind<3;++kind){if(kind==1&&s<=h)continue;if(kind==2&&s!=2*h)continue;structural_input<<<128,128,0,st>>>(saved.p,s,h,kind);st.sync();
   std::vector<double> t;Event first,last;for(int i=0;i<18;++i){CU(cudaMemcpyAsync(a.p,saved.p,s*h*sizeof(T),cudaMemcpyDeviceToDevice,st));first.record(st);native_hh<<<1,threads,0,st>>>(a.p,s,h,v.p,tri.p,kind,status.p,wit.p);last.record(st);CU(cudaEventSynchronize(last.e));float ms;CU(cudaEventElapsedTime(&ms,first.e,last.e));if(i>=3)t.push_back(ms*1e-3);}
   save(kind==0?"ge":kind==1?"ts":"tt",s,h,1,threads,t);
  }
 }
 for(int s:{8,32,64,128,256})for(int h:{1,8,16,32})for(int q:{1,16,64,128}){
  if(h>s)continue;structural_input<<<128,128,0,st>>>(a.p,s,h,0);native_hh<<<1,128,0,st>>>(a.p,s,h,v.p,tri.p,0,status.p,wit.p);make_input<<<128,128,0,st>>>(saved.p,s,q,s,0,s,0,T(1));
  for(int threads:{1,32,128,256}){
   auto t=time_primitive([&](){CU(cudaMemcpyAsync(x.p,saved.p,s*q*sizeof(T),cudaMemcpyDeviceToDevice,st));column_scales<<<q,threads,0,st>>>(x.p,s,q,scales.p);w_partial<<<ceildiv(h*q,128),128,0,st>>>(v.p,x.p,scales.p,s,h,q,0,s,w.p,wit.p);triangular_apply<<<ceildiv(h*q,128),128,0,st>>>(tri.p,w.p,z.p,h,q,true);packed_commit<<<ceildiv(s*q,128),128,0,st>>>(v.p,z.p,scales.p,x.p,s,h,q,status.p,wit.p);},st);
   save("apply",s,h,q,threads,t);
  }
 }
 for(int count:{1,8,32,128,512,2048,8192,32768,65536}){
  save("copy",count,1,1,128,time_primitive([&](){CU(cudaMemcpyAsync(x.p,saved.p,count*sizeof(T),cudaMemcpyDeviceToDevice,st));},st));
  save("norm",count,1,1,256,time_primitive([&](){norm_probe<<<1,256,0,st>>>(saved.p,count,norm.p);},st));
 }
 partials.zero(st);for(int h:{1,8,16,32})for(int q:{1,16,64,128})for(int c:{1,2,4})save("join",h,q,c,128,time_primitive([&](){sum_partials<<<ceildiv(h*q,128),128,0,st>>>(partials.p,w.p,h*q,c,wit.p);},st));
 // Scalar member is a complete GPU HH packet, with input reset outside timing.
 for(int threads:{1,32,128,256})for(auto shape:std::vector<std::pair<int,int>>{{1,1},{8,8},{32,16},{16,32},{64,64},{128,32},{32,128},{128,128}}){int m=shape.first,n=shape.second;
  make_input<<<128,128,0,st>>>(saved.p,m,n,m,0,m,0,T(1));std::vector<double> times;Event first,last;
  for(int i=0;i<18;++i){CU(cudaMemcpyAsync(a.p,saved.p,m*n*sizeof(T),cudaMemcpyDeviceToDevice,st));first.record(st);scalar_inplace<<<1,threads,0,st>>>(a.p,m,n,m,norm.p,status.p,wit.p);last.record(st);CU(cudaEventSynchronize(last.e));float ms;CU(cudaEventElapsedTime(&ms,first.e,last.e));if(i>=3)times.push_back(ms*1e-3);}save("scalar_ge",m,std::min(m,n),n,threads,times);
 }
 // Host queue, event and descriptor service are separate from kernel intervals.
 std::vector<double> launch,enqueue,event,descriptor;
 for(int i=0;i<30;++i){double t=seconds();empty_kernel<<<1,1,0,st>>>();double mid=seconds();st.sync();launch.push_back(seconds()-t);enqueue.push_back(mid-t);t=seconds();Event a,b;a.record(st);b.record(st);CU(cudaEventSynchronize(b.e));event.push_back(seconds()-t);t=seconds();auto plan=instantiate(129,65,1,16,32,1,1,128,64,"binary_tt",sizeof(T));volatile size_t z=plan.events.size();(void)z;descriptor.push_back(seconds()-t);}
 p["host_launch_sync_s"]=median(launch);p["host_enqueue_s"]=median(enqueue);p["host_event_s"]=median(event);p["descriptor_129x65_s"]=median(descriptor);p["host_raw"]={{"launch_sync",launch},{"enqueue",enqueue},{"event",event},{"descriptor",descriptor}};
 p["resources"]=json::array();
 auto resource=[&](const char*name,const void*function){cudaFuncAttributes f;CU(cudaFuncGetAttributes(&f,function));for(int t:{1,32,128,256}){int blocks=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,t,0));p["resources"].push_back({{"kernel",name},{"threads",t},{"registers_per_thread",f.numRegs},{"static_shared_bytes",f.sharedSizeBytes},{"local_bytes_per_thread",f.localSizeBytes},{"active_blocks_per_sm",blocks},{"register_accounting","physical thread registers once; warp views add no capacity"}});}};
 resource("native_hh",(const void*)native_hh<T>);resource("scalar_inplace",(const void*)scalar_inplace<T>);resource("w_partial",(const void*)w_partial<T>);resource("packed_commit",(const void*)packed_commit<T>);
 // Cache-state probes are observations after explicit access, never residency guarantees.
 size_t cold_count=std::max(size_t(1),size_t(hardware.at("l2_bytes"))*2/sizeof(T));Buffer<T>cold(cold_count),coldout(cold_count);cold.zero(st);st.sync();
 for(size_t count:{size_t(4096),size_t(65536),cold_count}){save("cache_warm_copy",int(count),1,1,128,time_primitive([&](){CU(cudaMemcpyAsync(coldout.p,cold.p,count*sizeof(T),cudaMemcpyDeviceToDevice,st));},st));
  std::vector<double> raw;for(int i=0;i<15;++i){CU(cudaMemsetAsync(coldout.p,0,cold_count*sizeof(T),st));Event a,b;a.record(st);CU(cudaMemcpyAsync(coldout.p,cold.p,count*sizeof(T),cudaMemcpyDeviceToDevice,st));b.record(st);CU(cudaEventSynchronize(b.e));float ms;CU(cudaEventElapsedTime(&ms,a.e,b.e));raw.push_back(ms*1e-3);}save("cache_after_sweep_copy",int(count),1,1,128,raw);}
 p["cache_conditions"]="warm repeats and destination sweep of twice nominal L2; no guaranteed eviction, hit rate or owned cache bytes";
 // Two distinct operands and streams, joined on one device clock. These are
 // the only admitted overlapping numerical episodes; hybrid backends serialize.
 {Stream other;Buffer<T>x2(128*64),w2(16*64),z2(16*64),sc2(64);structural_input<<<128,128,0,st>>>(a.p,128,16,0);native_hh<<<1,128,0,st>>>(a.p,128,16,v.p,tri.p,0,status.p,wit.p);make_input<<<128,128,0,st>>>(x.p,128,64,128,0,128,0,T(1));make_input<<<128,128,0,st>>>(x2.p,128,64,128,0,128,0,T(1));st.sync();
  auto apply=[&](cudaStream_t stream,T*xx,T*ww,T*zz,T*ss){column_scales<<<64,128,0,stream>>>(xx,128,64,ss);w_partial<<<8,128,0,stream>>>(v.p,xx,ss,128,16,64,0,128,ww,wit.p);triangular_apply<<<8,128,0,stream>>>(tri.p,ww,zz,16,64,true);packed_commit<<<64,128,0,stream>>>(v.p,zz,ss,xx,128,16,64,status.p,wit.p);};
  auto serial=time_primitive([&](){apply(st,x.p,w.p,z.p,scales.p);apply(st,x2.p,w2.p,z2.p,sc2.p);},st);
  Event gate,done;auto concurrent=time_primitive([&](){gate.record(st);gate.wait(other);apply(st,x.p,w.p,z.p,scales.p);apply(other,x2.p,w2.p,z2.p,sc2.p);done.record(other);done.wait(st);},st);
  p["concurrent_apply_multiplier"]=2*median(concurrent)/median(serial);p["concurrency"]={{"shape",{128,16,64}},{"depth",2},{"serial_pair_raw_s",serial},{"concurrent_pair_raw_s",concurrent},{"service_scope","two disjoint local strips; geometry extrapolation predictive"},{"backend_concurrency","not admitted; common serialized episode order"}};
 }
 st.sync();int unstable=0;for(auto&sample:p["samples"]){auto raw=sample["raw_s"].template get<std::vector<double>>();std::sort(raw.begin(),raw.end());double med=median(raw),lo=raw[raw.size()/10],hi=raw[(raw.size()*9)/10];sample["p10_s"]=lo;sample["p90_s"]=hi;sample["relative_p90_p10_spread"]=med?(hi-lo)/med:0;sample["stable_by_50_percent_spread"]=!med||(hi-lo)/med<=0.5;if(med&&(hi-lo)/med>0.5)++unstable;}p["unstable_signatures"]=unstable;p["probe_policy"]="3 warmups, 15 measured observations; record instability without claiming confidence bounds";
 st.sync();p["probe_status"]=status.download()[0];p["duration_s"]=seconds()-start;p["id"]=digest(p.dump());return p;
}
}
