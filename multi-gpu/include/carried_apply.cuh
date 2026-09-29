#pragma once
// Block-level carried products: the warp groups of a thread block split the contraction and combine
// their partials in shared memory.
#include "tiled_kernels.cuh"
#include <type_traits>
namespace tqr {
// Each CTA owns one output tile. Its blockDim.y groups partition [K] into balanced, disjoint
// intervals [floor(K*z/CB),floor(K*(z+1)/CB)). Each group holds partial outputs in registers;
// shared memory combines them. D commits X once. No HBM partials, numerical atomics, TF32, or extra
// launches. CB=8 uses a shorter output tile to keep its 512-thread register allocation below the SM
// register file. The geometry is part of the compiled sample.
inline constexpr int carried_group_threads=64;
inline int carried_block_c(int requested,int contraction){
 if(requested!=2&&requested!=4&&requested!=8)throw std::runtime_error("unsupported_block_carrier");
 if(contraction<1)return 1;
 while(requested>contraction)requested/=2;
 return requested;
}
template<int CB> inline constexpr int carried_tile_m=CB==8?16:64;
template<class T,int CB,int TM,int TN> inline constexpr size_t carried_shared_bytes=
 size_t(CB)*size_t(TM*TN>2*(TM+TN)*8?TM*TN:2*(TM+TN)*8)*sizeof(T);

// Arithmetic and order are unchanged.
template<class T,int CB,int TM,int TN,bool Transpose,bool Subtract,bool TransC=false,bool TransB=false>
__device__ __forceinline__ void carried_apply_body(
 const T*a,int lda,long long sa,const T*b,int ldb,long long sb,
 T*c,int ldc,long long sc,int m,int n,int k){
 constexpr int BK=8,RM=TM/8,RN=TN/8;
 const int z=threadIdx.y,t=threadIdx.x,ri=t%8,cj=t/8;
 const int i0=blockIdx.y*TM,j0=blockIdx.x*TN;
 a+=static_cast<long long>(blockIdx.z)*sa;
 b+=static_cast<long long>(blockIdx.z)*sb;
 c+=static_cast<long long>(blockIdx.z)*sc;
 extern __shared__ __align__(16) unsigned char carried_smem[];
 T*scratch=reinterpret_cast<T*>(carried_smem);
 // No behavior change vs the gated sf-b/sf-c kernel. Software-pipelined staging, v2: two operand
 // slots per group and cp.async double-buffering. The next slab's global loads issue BEFORE the
 // current slab's math; the math that follows is independent of them, so the Long Scoreboard
 // latency (5.39, dominant in ncu) overlaps compute instead of stalling it (0.19 eligible warps had
 // nothing to switch to). The 8x8 per-thread tile is KEPT (cutting ILP buys warps at the cost of
 // the independent work that hides latency -- wrong direction here). cp.async-8B is alignment-safe
 // for every double array (8B elements, 256B-aligned bases); OOB lanes are predicated OFF over a
 // zero-filled slot, so tails stay exact. Same __syncthreads count per step (two), same algebra,
 // same shared footprint class (slots live inside the partials' region, reused after the last
 // compute, fenced by the existing barrier). Two staging slots per group: 2*SLOT words starting at
 // z*2*SLOT, inside the carried_shared_bytes allocation (which covers max(TM*TN, 2*SLOT) per group
 // -- the CB=8 tile needs the second term). Staging may overlap the partials' region; sequential
 // reuse fenced by the existing barriers makes that safe, exactly as the serial form reused one
 // slot the same way.
 constexpr size_t SLOT=(TM+TN)*BK;
 T*slot0=scratch+size_t(z)*2*SLOT,*slot1=slot0+SLOT;
 T acc[RM][RN]={};
 const int begin=int((static_cast<long long>(k)*z)/CB);
 const int end=int((static_cast<long long>(k)*(z+1))/CB);
 // All groups take the same number of block barriers, including short tails.
 const int steps=(k+CB*BK-1)/(CB*BK);
 // Hoisted-load double buffer: the next slab's staging loads issue BEFORE the current slab's math
 // and complete during it (regular loads have no alignment constraints, unlike cp.async which ptxas
 // restricts to 16B here). __syncthreads orders shared reuse but does not wait for in-flight global
 // loads; the scoreboard stalls only genuinely late data. Same barrier count, same algebra, same
 // 8x8 tile.
 auto load_slab=[&](int kb,T*as,T*bs){
  for(int ix=t;ix<TM*BK;ix+=carried_group_threads){
   const int i=Transpose?ix/BK:ix%TM,kk=Transpose?ix%BK:ix/TM;
   const int row=i0+i,inner=kb+kk;
   as[kk*TM+i]=(row<m&&inner<end)
     ?(Transpose?a[inner+size_t(row)*lda]:a[row+size_t(inner)*lda]):T(0);
  }
  for(int ix=t;ix<BK*TN;ix+=carried_group_threads){
   const int kk=ix%BK,j=ix/BK,inner=kb+kk;
   bs[j*BK+kk]=(j0+j<n&&inner<end)
     ?(TransB?b[j0+j+size_t(inner)*ldb]:b[inner+size_t(j0+j)*ldb]):T(0);
  }};
 auto compute_slab=[&](T*as,T*bs){
  #pragma unroll
  for(int kk=0;kk<BK;++kk){
   T av[RM],bv[RN];
   #pragma unroll
   for(int u=0;u<RM;++u)av[u]=as[kk*TM+ri+8*u];
   #pragma unroll
   for(int v=0;v<RN;++v)bv[v]=bs[(cj+8*v)*BK+kk];
   #pragma unroll
   for(int u=0;u<RM;++u){
    #pragma unroll
    for(int v=0;v<RN;++v)acc[u][v]+=av[u]*bv[v];
   }
  }};
 T*as=slot0,*bs=slot0+TM*BK;
 load_slab(begin,as,bs);
 __syncthreads();
 for(int step=0;step<steps;++step){
  T*ns=step%2?slot0:slot1,*nbs=ns+TM*BK;
  if(step+1<steps)load_slab(begin+(step+1)*BK,ns,nbs);
  __syncthreads();
  compute_slab(as,bs);
  __syncthreads();
  as=ns;bs=nbs;
 }
 // Operand staging is dead; reuse its shared allocation for the partials.
 #pragma unroll
 for(int u=0;u<RM;++u){
  #pragma unroll
  for(int v=0;v<RN;++v)scratch[size_t(z)*TM*TN+ri+8*u+size_t(cj+8*v)*TM]=acc[u][v];
 }
 __syncthreads();
 if(z==0){
  #pragma unroll
  for(int u=0;u<RM;++u){
   #pragma unroll
   for(int v=0;v<RN;++v){
    const int i=ri+8*u,j=cj+8*v;
    T sum=scratch[i+size_t(j)*TM];
    #pragma unroll
    for(int peer=1;peer<CB;++peer)sum+=scratch[size_t(peer)*TM*TN+i+size_t(j)*TM];
    if(i0+i<m&&j0+j<n){
     T&out=TransC?c[j0+j+size_t(i0+i)*ldc]:c[i0+i+size_t(j0+j)*ldc];
     if constexpr(Subtract)out-=sum;else out=sum;
    }
   }
  }
 }
}

template<class T,int CB,int TM=carried_tile_m<CB>,int TN=64>
__global__ __launch_bounds__(carried_group_threads*CB) void tile_carried_W(
 const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
 T*w,int ldw,long long sw,int rows,int h,int q,Witness*wit=nullptr){
 carried_apply_body<T,CB,TM,TN,true,false>(v,ldv,sv,x,ldx,sx,w,ldw,sw,h,q,rows);
 // One report per slice; gridDim.z==count (the launch helper) makes slices==members exactly.
 if(wit&&threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0){
  atomicAdd(&wit->device_peer_partials[2],1ULL);
  atomicAdd(&wit->device_block_peers[2],(unsigned long long)blockDim.y);witness_levels(wit,2,int(blockDim.y),1,1,1);
  atomicAdd(&wit->device_membership_reports,1ULL);}
}
template<class T,int CB,bool TransC=false,int TM=carried_tile_m<CB>,int TN=64>
__global__ __launch_bounds__(carried_group_threads*CB) void tile_carried_Z(
 const T*t,int ldt,long long stn,const T*w,int ldw,long long sw,
 T*z,int ldz,long long sz,int h,int q,bool transpose,Witness*wit=nullptr){
 if(transpose)carried_apply_body<T,CB,TM,TN,true,false,TransC>(t,ldt,stn,w,ldw,sw,z,ldz,sz,h,q,h);
 else carried_apply_body<T,CB,TM,TN,false,false,TransC>(t,ldt,stn,w,ldw,sw,z,ldz,sz,h,q,h);
 if(wit&&threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0){
  atomicAdd(&wit->device_peer_partials[3],1ULL);
  atomicAdd(&wit->device_block_peers[3],(unsigned long long)blockDim.y);witness_levels(wit,3,int(blockDim.y),1,1,1);
  atomicAdd(&wit->device_membership_reports,1ULL);}
}
template<class T,int CB,int TM=carried_tile_m<CB>,int TN=64>
__global__ __launch_bounds__(carried_group_threads*CB) void tile_carried_D(
 const T*v,int ldv,long long sv,const T*z,int ldz,long long sz,
 T*x,int ldx,long long sx,int rows,int h,int q,Witness*wit=nullptr){
  // (m,n,k) = (rows,q,h): D = V(rows,h) Z(h,q) into X(rows,q). The kernel parameter order
  // (rows,h,q) is NOT the body order -- transposing n/k here silently computes the wrong product.
  carried_apply_body<T,CB,TM,TN,false,true>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,q,h);
  // Peer partials as above; the physical commit is the owner-commits-once evidence for the native
  // path. The cuBLAS drep>1 path instead performs drep sequential beta=1 writes -- counted on the
  // host as library commits, never as one commit.
  if(wit&&threadIdx.x==0&&threadIdx.y==0&&blockIdx.x==0&&blockIdx.y==0){
   atomicAdd(&wit->device_peer_partials[4],1ULL);
   atomicAdd(&wit->device_block_peers[4],(unsigned long long)blockDim.y);witness_levels(wit,4,int(blockDim.y),1,1,1);
   atomicAdd(&wit->device_membership_reports,1ULL);
   atomicAdd(&wit->device_physical_commits,1ULL);}
}
// Configure opt-in shared memory once per instantiation and device. This
// initialization is outside the event-timed samples after their warmup.
template<class T,int CB,int Kind> void prepare_carried(){
 constexpr int TM=carried_tile_m<CB>,TN=64;
 constexpr size_t bytes=carried_shared_bytes<T,CB,TM,TN>;
 static int last_device=-1;int device;CU(cudaGetDevice(&device));
 if(device==last_device)return;
 const void*kernel=Kind==0?(const void*)tile_carried_W<T,CB>:
   (Kind==1?(const void*)tile_carried_Z<T,CB>:(const void*)tile_carried_D<T,CB>);
 CU(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
 last_device=device;
}
template<class T,int CB> void prepare_carried_zt(){
 constexpr size_t bytes=carried_shared_bytes<T,CB,carried_tile_m<CB>,64>;
 static int last_device=-1;int device;CU(cudaGetDevice(&device));
 if(device==last_device)return;
 CU(cudaFuncSetAttribute((const void*)tile_carried_Z<T,CB,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
 last_device=device;
}
template<class F> inline int dispatch_block_carrier(int requested,int contraction,F&&f){
 int cb=carried_block_c(requested,contraction);
 switch(cb){
  case 8:f(std::integral_constant<int,8>{});break;
  case 4:f(std::integral_constant<int,4>{});break;
  case 2:f(std::integral_constant<int,2>{});break;
  default:f(std::integral_constant<int,1>{});break;
 }
 return cb;
}
template<class T> int launch_carried_W(const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
 T*w,int ldw,long long sw,int rows,int h,int q,int count,int block_c,cudaStream_t st,Witness*wit=nullptr){
 if(!rows||!h||!q||!count)return 0;
 return dispatch_block_carrier(block_c,rows,[&](auto tag){constexpr int CB=decltype(tag)::value;
  prepare_carried<T,CB,0>();
  tile_carried_W<T,CB><<<dim3(ceildiv(q,64),ceildiv(h,carried_tile_m<CB>),count),
   dim3(carried_group_threads,CB),carried_shared_bytes<T,CB,carried_tile_m<CB>,64>,st>>>
   (v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,wit);});
}
// zt_out: write Zt = Z^T (q x h, ld = ldz) for the native carried D.
template<class T> int launch_carried_Z(const T*t,int ldt,long long stn,const T*w,int ldw,long long sw,
 T*z,int ldz,long long sz,int h,int q,int count,bool transpose,int block_c,cudaStream_t st,Witness*wit=nullptr,bool zt_out=false){
 if(!h||!q||!count)return 0;
 return dispatch_block_carrier(block_c,h,[&](auto tag){constexpr int CB=decltype(tag)::value;
  prepare_carried<T,CB,1>();
  if(zt_out)prepare_carried_zt<T,CB>();
  const dim3 grid(ceildiv(q,64),ceildiv(h,carried_tile_m<CB>),count),block(carried_group_threads,CB);
  constexpr size_t bytes=carried_shared_bytes<T,CB,carried_tile_m<CB>,64>;
  if(zt_out)tile_carried_Z<T,CB,true><<<grid,block,bytes,st>>>(t,ldt,stn,w,ldw,sw,z,ldz,sz,h,q,transpose,wit);
  else tile_carried_Z<T,CB><<<grid,block,bytes,st>>>(t,ldt,stn,w,ldw,sw,z,ldz,sz,h,q,transpose,wit);});
}
template<class T> int launch_carried_D(const T*v,int ldv,long long sv,const T*z,int ldz,long long sz,
 T*x,int ldx,long long sx,int rows,int h,int q,int count,int block_c,cudaStream_t st,Witness*wit=nullptr){
 if(!rows||!h||!q||!count)return 0;
 return dispatch_block_carrier(block_c,h,[&](auto tag){constexpr int CB=decltype(tag)::value;
  prepare_carried<T,CB,2>();
  tile_carried_D<T,CB><<<dim3(ceildiv(q,64),ceildiv(rows,carried_tile_m<CB>),count),
   dim3(carried_group_threads,CB),carried_shared_bytes<T,CB,carried_tile_m<CB>,64>,st>>>
   (v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,wit);});
}
}
