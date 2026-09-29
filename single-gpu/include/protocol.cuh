#pragma once
// Self-tests of the inter-GPU transport protocol.
#include "probe.cuh"
#include <thread>
namespace tqr {
template<class T> __device__ T protocol_value(int rank,int index,int mode){
 if(mode==0){if constexpr(sizeof(T)==8)return __longlong_as_double(0x3ff00000fffafffaULL);else return T(1.25);}
 if(mode==1)return (rank%2?-T(1):T(1))*T(1+index%7);
 if(mode==2)return rank%2?-T(0):T(0);
 if(mode==3)return std::sqrt(std::numeric_limits<T>::max())/T(128)*(rank%2?-T(1):T(1));
 return T(rank+1)+T(index%13)/T(16);
}
template<class T> __global__ void protocol_input(T* x,int count,int rank,int mode){for(int i=threadIdx.x;i<count;i+=blockDim.x)x[i]=protocol_value<T>(rank,i,mode);}
template<class T> __global__ void protocol_verify(const T* x,int count,int root,int ranks,int mode,bool sum,int* status){
 for(int i=threadIdx.x;i<count;i+=blockDim.x){T expected=protocol_value<T>(root,i,mode);if(sum){expected=0;for(int r=0;r<ranks;++r)expected+=protocol_value<T>(r,i,mode);}
  T diff=absval(x[i]-expected);if(!isfinite(x[i])||diff>T(16)*std::numeric_limits<T>::epsilon()*max(T(1),absval(expected)))atomicCAS(status,0,PROTOCOL_ERROR);
  if(!sum){const unsigned char* a=(const unsigned char*)(x+i);const unsigned char*b=(const unsigned char*)&expected;for(int j=0;j<sizeof(T);++j)if(a[j]!=b[j])atomicCAS(status,0,PROTOCOL_ERROR);}
 }
}
__global__ inline void compare_bytes(const unsigned char*a,const unsigned char*b,int n,int*status){for(int i=threadIdx.x;i<n;i+=blockDim.x)if(a[i]!=b[i])atomicCAS(status,0,PROTOCOL_ERROR);}
template<class T> json qualify_protocol(Context&ctx){
 if(ctx.size<2)throw std::runtime_error("protocol_requires_multiple_gpus");
 constexpr int capacity=131073;const int sizes[]={0,1,31,127,128,129,1023,4095,4096,4097,131071,131072,131073};
 // The transport is bound to this test's stream, so every episode is ordered
 // against the input/verify kernels exactly as it is in the engine, and the
 // per-call drain below makes each recorded interval a COMPLETE episode rather
 // than an enqueue.
 double setup_start=seconds();Stream stream;Transport tr(ctx,capacity*sizeof(T));tr.bind(stream.s);double setup_s=seconds()-setup_start;Buffer<T> input(capacity),pub(capacity),out(capacity),expected(capacity);Buffer<int> status(1);status.zero(stream);std::vector<double> copies,reduces,publications,hybrid;
 double start=seconds();
 for(int iter=0;iter<600;++iter){int count=sizes[iter%13],root=iter%ctx.size,mode=iter%5;
  protocol_input<<<1,128,0,stream>>>(input.p,capacity,ctx.rank,mode);stream.sync();
  if(count&&iter%79==0)tr.delay_next_paper_producer(root,5);
  ctx.barrier();double t=seconds();tr.template paper<T,false>(root,input.p,out.p,count,iter);tr.drain();copies.push_back(seconds()-t);
  protocol_verify<<<1,128,0,stream>>>(out.p,count,root,ctx.size,mode,false,status.p);stream.sync();
  if(count&&iter%79==0)tr.delay_next_paper_producer((root+1)%ctx.size,5);
  t=seconds();tr.template paper<T,true>(root,input.p,out.p,count,iter);tr.drain();reduces.push_back(seconds()-t);
  if(ctx.rank==root)protocol_verify<<<1,128,0,stream>>>(out.p,count,root,ctx.size,mode,true,status.p);stream.sync();
  int receiver=(root+1)%ctx.size;t=seconds();tr.publish(root,receiver,input.p,pub.p,count*sizeof(T),iter,7);tr.drain();publications.push_back(seconds()-t);
  // The publication receive/copy is complete before becoming a frozen NCCL source. NCCL output
  // completes before publishing.
  t=seconds();tr.template paper<T,false>(receiver,pub.p,out.p,count,iter);
  tr.publish(receiver,root,out.p,pub.p,count*sizeof(T),iter,8);tr.drain();hybrid.push_back(seconds()-t);
  if(ctx.rank==root)protocol_verify<<<1,128,0,stream>>>(pub.p,count,root,ctx.size,mode,false,status.p);stream.sync();
  int ragged=1+(iter*37)%1021;tr.publish(root,receiver,input.p,pub.p,ragged,iter,11);tr.drain();
  if(ctx.rank==receiver){protocol_input<<<1,128,0,stream>>>(expected.p,capacity,root,mode);compare_bytes<<<1,128,0,stream>>>((const unsigned char*)pub.p,(const unsigned char*)expected.p,ragged,status.p);}stream.sync();
  if(ctx.max_status(status.download()[0]))throw std::runtime_error("protocol_numerical_mismatch_at_"+std::to_string(iter));
 }
 bool mismatch_rejected=false,publication_rejected=false,stale_rejected=false,zero_rejected=false,self_rejected=false,mixed_rejected=false;
#ifdef TQR_MULTI
 // Normal traffic above exercises deferred agreement. Reject deliberately
 // divergent descriptors below before they can issue mismatched device work.
 tr.set_eager_agreement(true);
 try{tr.template paper<T,false>(0,input.p,out.p,ctx.rank==0?1:2,987654);}catch(const std::runtime_error&e){mismatch_rejected=std::string(e.what())=="collective_signature";}
 try{tr.publish(0,1,input.p,pub.p,8,ctx.rank,12);}catch(const std::runtime_error&e){publication_rejected=std::string(e.what())=="publication_signature_before_issue";}
 try{tr.template paper<T,false>(0,input.p,out.p,ctx.rank==0?0:1,987655);}catch(const std::runtime_error&e){zero_rejected=std::string(e.what())=="collective_signature";}
 try{tr.publish(0,ctx.rank==0?0:1,input.p,pub.p,8,987656,12);}catch(const std::runtime_error&e){self_rejected=std::string(e.what())=="publication_signature_before_issue";}
 try{if(ctx.rank==0)tr.publish(0,0,input.p,pub.p,0,987657,0);else tr.template paper<T,false>(0,input.p,out.p,0,987657);}catch(const std::runtime_error&e){mixed_rejected=std::string(e.what())==(ctx.rank==0?"publication_signature_before_issue":"collective_signature");}
 PublicationHeader wanted{100,10,8,1,0,1},old=wanted;old.generation=99;stale_rejected=!publication_matches(old,wanted);old=wanted;old.expression=9;stale_rejected=stale_rejected&&!publication_matches(old,wanted);
#endif
 if(!mismatch_rejected||!publication_rejected||!stale_rejected||!zero_rejected||!self_rejected||!mixed_rejected)throw std::runtime_error("protocol_fault_guard");
 return {{"zero_mismatch_rejected",zero_rejected},{"self_mismatch_rejected",self_rejected},{"mixed_backend_mismatch_rejected",mixed_rejected},{"tested_counts",sizes},{"delay_location","after common staging/release, immediately before paper kernel launch"},{"mismatched_collective_rejected",mismatch_rejected},{"mismatched_publication_rejected",publication_rejected},{"stale_header_rejected",stale_rejected},{"ragged_bytes_bit_exact",true},{"precision",precision<T>()},{"gpus",ctx.size},{"pass",true},{"calls",600},{"finite_fp64_poison_collision",sizeof(T)==8},{"ragged_and_zero_counts",true},{"delayed_peers",true},{"both_handoff_directions",true},{"epoch_wraps",">=9 over the 600 iterations; the window is cleared on epoch-value reuse (every 254 sequence numbers) and on first use, and each recorded interval is drained"},{"status",status.download()[0]},{"paper_copy_raw_s",copies},{"paper_reduce_raw_s",reduces},{"nvshmem_raw_s",publications},{"hybrid_raw_s",hybrid},{"paper_episode_s",median(reduces)},{"nvshmem_publication_s",median(publications)},{"setup_s",setup_s},{"duration_s",seconds()-start},{"evidence",tr.evidence}};
}
}
