#pragma once
// Carriers for short contractions (few reflectors) that keep c > 1 with SIMT peers.
#include "carried_apply.cuh"
#include "carrier_gemm.cuh"
namespace tqr {
// Short reflector contractions still execute the scheduled c. Tensor-op
// warp slices need 16 values each; balanced SIMT peers can own smaller,
// nonempty intervals without changing c or allocating output replicas.
// This covers only the tensor-mode shapes the existing sliced kernels refuse.
template<class T> bool short_carrier_path(int c,int h,bool d){
 if constexpr(!std::is_same_v<T,float>)return false;
 else return fp32_math()!=Fp32Math::IEEE&&(c==2||c==4)&&h>=c
   &&!(d?carrier_d_supported<T>(c,h):carrier_z_supported<T>(c,h));
}
template<class T,int C,bool D,bool TransC=false>
__global__ __launch_bounds__(carried_group_threads*C) void short_carrier_kernel(
 const T*a,int lda,long long sa,const T*b,int ldb,long long sb,
 T*out,int ldo,long long so,int rows,int h,int q,bool transpose,Witness*wit){
 if constexpr(D)
  carried_apply_body<T,C,64,64,false,true,false,true>(a,lda,sa,b,ldb,sb,out,ldo,so,rows,q,h);
 else if(transpose)
  carried_apply_body<T,C,64,64,true,false,TransC>(a,lda,sa,b,ldb,sb,out,ldo,so,h,q,h);
 else
  carried_apply_body<T,C,64,64,false,false,TransC>(a,lda,sa,b,ldb,sb,out,ldo,so,h,q,h);
 if(wit&&threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0){
  constexpr int kind=D?4:3;
  atomicAdd(&wit->device_peer_partials[kind],(unsigned long long)C);
  atomicAdd(&wit->device_block_peers[kind],(unsigned long long)C);
  witness_levels(wit,kind,C,1,1,1);
  atomicAdd(&wit->device_membership_reports,1ULL);
  if constexpr(D)atomicAdd(&wit->device_physical_commits,1ULL);
  else if(blockIdx.z==0)atomicAdd(&wit->device_combines,1ULL); // one per carried Z launch, not per batch member
 }
}
template<class T,bool D,bool TransC=false> int launch_short_carrier(int c,
 const T*a,int lda,long long sa,const T*b,int ldb,long long sb,T*out,int ldo,long long so,
 int rows,int h,int q,int count,bool transpose,cudaStream_t st,Witness*wit){
 if(!short_carrier_path<T>(c,h,D))throw std::runtime_error("short_carrier_shape");
 if(!rows||!q||!count)return c;
 dispatch_block_carrier(c,h,[&](auto tag){constexpr int C=decltype(tag)::value;
  if constexpr(C==2||C==4){
   constexpr size_t bytes=carried_shared_bytes<T,C,64,64>;
   static int last_device=-1;int device;CU(cudaGetDevice(&device));
   if(last_device!=device){
    CU(cudaFuncSetAttribute(short_carrier_kernel<T,C,D,TransC>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    last_device=device;
   }
   short_carrier_kernel<T,C,D,TransC><<<dim3(ceildiv(q,64),ceildiv(D?rows:h,64),count),
     dim3(carried_group_threads,C),bytes,st>>>(a,lda,sa,b,ldb,sb,out,ldo,so,rows,h,q,transpose,wit);
   CU(cudaGetLastError());
  }
 });
 return c;
}
}
