#pragma once
// Triangular factors T wider than 128 columns for the multi-GPU merges.
#include "gau_ge.cuh"

namespace tqr {
// The diagonal 128-column blocks use the existing dlarft recurrence. Keeping
// shared memory bounded by 128 avoids the quadratic shared-memory overflow
// that a direct H=256/512 instantiation of gau_build_t_h would cause.
template<class T> __global__ __launch_bounds__(512)
void multi_wide_t_diagonal(const T*G,int ldg,const T*tau,const TilePacket*packet,T*Ts,int b){
 constexpr int H=128,LD=129;
 extern __shared__ __align__(16) unsigned char wide_smem[];
 T*Sm=reinterpret_cast<T*>(wide_smem);T*M=Sm+H*LD;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,base=blockIdx.x*H;
 const TilePacket pk=packet[0];const int h=min(H,pk.h-base);
 if(h<=0)return;
 T*tri=Ts+size_t(pk.tile)*b*b;
 for(int e=tid;e<H*H;e+=512){const int i=e%H,j=e/H;
  const T g=(i<h&&j<h)?G[base+i+size_t(base+j)*ldg]:T(0);
  if(i<j)Sm[j+i*LD]=g;else if(i==j)Sm[i+i*LD]=i<h?tau[base+i]:T(0);
 }
 __syncthreads();
 if(warp<H/8){const int b0=warp*8,r=b0+lane;
  for(int c=1;c<8;++c){T value=T(0);
   if(lane<c){for(int q=0;q<8;++q)if(q>=lane&&q<c)
    value+=Sm[r+(b0+q)*LD]*Sm[(b0+c)+(b0+q)*LD];
    value*=-Sm[(b0+c)+(b0+c)*LD];}
   __syncwarp();if(lane<c)Sm[r+(b0+c)*LD]=value;__syncwarp();
  }
 }
 __syncthreads();
 gau_t_compose_level<T,H,8>(Sm,M,tid);
 gau_t_compose_level<T,H,16>(Sm,M,tid);
 gau_t_compose_level<T,H,32>(Sm,M,tid);
 gau_t_compose_level<T,H,64>(Sm,M,tid);
 for(int e=tid;e<H*H;e+=512){const int i=e%H,j=e/H;
  if(i<h&&j<h&&i<=j)tri[base+i+size_t(base+j)*b]=Sm[i+j*LD];
 }
}

template<class T> __global__ void multi_wide_t_zero(const TilePacket*packet,T*Ts,int b){
 T*tri=Ts+size_t(packet[0].tile)*b*b;
 for(int i=threadIdx.x;i<b*b;i+=blockDim.x)tri[i]=T(0);
}

// T12 = -(T1 G12) T2, G12 = V1^T V2 of ALREADY FORMED reflectors.
// Each output has two concurrent, disjoint contraction slices and one owner
// write. The operand needed by an output group is read into each lane's
// registers; partials meet in shared memory in fixed slice order. A unit
// contraction (possible only in a ragged terminal block) is explicitly c=1.
template<class T,bool Right> __global__ void multi_wide_t_join(const T*G,int ldg,
 const TilePacket*packet,T*Ts,int b,T*temporary,int span){
 const int base=int(blockIdx.y)*2*span,h=packet[0].h;
 const int left=min(span,h-base),right=min(span,h-base-left);
 if(left<=0||right<=0)return;
 const int element=blockIdx.x*blockDim.x+threadIdx.x,peer=threadIdx.y;
 const bool active=element<left*right;
 const int i=element%left,j=element/left,contraction=Right?right:left;
 const int carriers=min(2,contraction),first=contraction*peer/carriers,last=contraction*(peer+1)/carriers;
 T*tri=Ts+size_t(packet[0].tile)*b*b;T sum=T(0);
 if(active&&peer<carriers)for(int k=first;k<last;++k){
  if constexpr(Right){if(k<=j)sum+=temporary[base+i+size_t(base+left+k)*b]*tri[base+left+k+size_t(base+left+j)*b];}
  else {if(k>=i)sum+=tri[base+i+size_t(base+k)*b]*G[base+k+size_t(base+left+j)*ldg];}
 }
 __shared__ T partial[2][128];partial[peer][threadIdx.x]=sum;__syncthreads();
 if(peer==0&&active){const T value=partial[0][threadIdx.x]+partial[1][threadIdx.x];
  if constexpr(Right)tri[base+i+size_t(base+left+j)*b]=-value;
  else temporary[base+i+size_t(base+left+j)*b]=value;
 }
}

template<class T> void multi_wide_t_preflight(int h,int b,int count){
 if(h<=128||h>512||b<h||count!=1)throw std::runtime_error("multi_wide_T_descriptor_before_modify");
 int device;CU(cudaGetDevice(&device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
 cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,multi_wide_t_diagonal<T>));
 const size_t bytes=gau_build_t_h_smem<T,128>();
 if(bytes>size_t(prop.sharedMemPerBlockOptin-attr.sharedSizeBytes))
  throw std::runtime_error("multi_wide_T_shared_capacity_before_modify");
 CU(cudaFuncSetAttribute(multi_wide_t_diagonal<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
 int occupancy=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,multi_wide_t_diagonal<T>,512,bytes));
 if(!occupancy)throw std::runtime_error("multi_wide_T_launch_before_modify");
}

template<class T> void launch_multi_wide_t(const T*G,int ldg,const T*tau,
 const TilePacket*packet,int h,T*Ts,int b,T*temporary,cudaStream_t st){
 multi_wide_t_zero<T><<<1,256,0,st>>>(packet,Ts,b);
 multi_wide_t_diagonal<T><<<ceildiv(h,128),512,gau_build_t_h_smem<T,128>(),st>>>(G,ldg,tau,packet,Ts,b);
 for(int span=128;span<h;span*=2){
  const dim3 grid(ceildiv(span*span,128),ceildiv(h,2*span)),threads(128,2);
  multi_wide_t_join<T,false><<<grid,threads,0,st>>>(G,ldg,packet,Ts,b,temporary,span);
  multi_wide_t_join<T,true><<<grid,threads,0,st>>>(G,ldg,packet,Ts,b,temporary,span);
 }
 CU(cudaGetLastError());
}
}
