#pragma once
// Dispatch of the three products of an update, W = V^T X, Z = T^T W and D = V Z, to their carriers:
// concurrent vendor GEMM peers, native carried kernels, or tensor-core (wgmma) kernels.
#include "common.hpp"
#include "carried_apply.cuh"
#include "carrier_gemm.cuh"
#include "carrier_short.cuh"
#include "kami_carrier.cuh"
#include "kami_dnp16.cuh"
#include "simt_carrier.cuh"   // IEEE fp32 skinny W_ONLY products (compose_G) on the SIMT W carrier
#include <cublas_v2.h>
namespace tqr {
// The three stages of an update, W = V^T X, Z = T^T W and D = V Z, each with its own carrier (Algorithm 2).
// The engine, the across-GPU merge and the probe call the same stages, so the probe measures exactly what the
// engine runs. With c=1 the native kernels run; batched BLAS is used only where c>1.
inline void tiled_blas_check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("tiled_cublas_"+std::to_string(s));}
// Math mode of the engine's vendor handles. DEFAULT keeps the selected precision (fp64: DMMA, IEEE-exact products;
// fp32: FFMA -- TF32 is used only under CUBLAS_TF32_TENSOR_OP_MATH or NVIDIA_TF32_OVERRIDE=1, both off) and lets
// cuBLAS pick its fast Hopper kernels. TQR_BLAS_MATH=pedantic.
inline cublasMath_t tiled_blas_math(){static const cublasMath_t m=[]{const char*e=std::getenv("TQR_BLAS_MATH");
  return (e&&std::string(e)=="pedantic")?CUBLAS_PEDANTIC_MATH:CUBLAS_DEFAULT_MATH;}();return m;}
template<class T> void tile_gemm2(cublasHandle_t handle,cublasOperation_t op,cublasOperation_t opb,int m,int n,int k,T alpha,const T*A,int lda,long long sa,const T*B,int ldb,long long sb,T beta,T*C,int ldc,long long sc,int batches){
 if(!m||!n||!k||!batches)return;
 if constexpr(sizeof(T)==4)tiled_blas_check(cublasSgemmStridedBatched(handle,op,opb,m,n,k,&alpha,A,lda,sa,B,ldb,sb,&beta,C,ldc,sc,batches));
 else tiled_blas_check(cublasDgemmStridedBatched(handle,op,opb,m,n,k,&alpha,A,lda,sa,B,ldb,sb,&beta,C,ldc,sc,batches));
}
template<class T> void tile_gemm(cublasHandle_t handle,cublasOperation_t op,int m,int n,int k,T alpha,const T*A,int lda,long long sa,const T*B,int ldb,long long sb,T beta,T*C,int ldc,long long sc,int batches){
 tile_gemm2<T>(handle,op,CUBLAS_OP_N,m,n,k,alpha,A,lda,sa,B,ldb,sb,beta,C,ldc,sc,batches);
}
// Pointer-array form of the same product. A carrier with c>1 over a BATCH of count leaves needs
// peer (i,z) at operand offset i*s_batch + z*part, which is two strides and therefore not
// expressible as one strided-batched call.
template<class T> void tile_gemm_ptr2(cublasHandle_t handle,cublasOperation_t op,cublasOperation_t opb,int m,int n,int k,T alpha,const T*const*A,int lda,const T*const*B,int ldb,T beta,T*const*C,int ldc,int batches){
 if(!m||!n||!k||!batches)return;
 if constexpr(sizeof(T)==4)tiled_blas_check(cublasSgemmBatched(handle,op,opb,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc,batches));
 else tiled_blas_check(cublasDgemmBatched(handle,op,opb,m,n,k,&alpha,A,lda,B,ldb,&beta,C,ldc,batches));
}
template<class T> void tile_gemm_ptr(cublasHandle_t handle,cublasOperation_t op,int m,int n,int k,T alpha,const T*const*A,int lda,const T*const*B,int ldb,T beta,T*const*C,int ldc,int batches){
 tile_gemm_ptr2<T>(handle,op,CUBLAS_OP_N,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc,batches);
}
// Native Eq-apply products for the c=1 regime. Kept compiled and readable so the carried kernels
// stay checkable against an independent summation order. One thread per output element with the
// column index fastest: reads of the column-major right operand coalesce and reads of the left
// operand broadcast across each warp.
template<class T> __global__ void tile_native_W(const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,T*w,int ldw,long long sw,int rows,int h,int q,int count){
 for(long long bb=(long long)blockIdx.z;bb<count;bb+=(long long)gridDim.z){
  const T*vv=v+size_t(bb)*sv,*xx=x+size_t(bb)*sx;T*ww=w+size_t(bb)*sw;
  for(int k=blockIdx.y*blockDim.y+threadIdx.y;k<h;k+=gridDim.y*blockDim.y)
   for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<q;j+=gridDim.x*blockDim.x){
    T s=T(0);for(int r=0;r<rows;++r)s+=vv[size_t(r)+size_t(k)*ldv]*xx[size_t(r)+size_t(j)*ldx];
    ww[size_t(k)+size_t(j)*ldw]=s;}
 }
}
template<class T> __global__ void tile_native_Z(const T*t,int ldt,long long st,const T*w,int ldw,long long sw,T*z,int ldz,long long sz,int h,int q,int count,bool transpose){
 for(long long bb=(long long)blockIdx.z;bb<count;bb+=(long long)gridDim.z){
  const T*tt=t+size_t(bb)*st,*ww=w+size_t(bb)*sw;T*zz=z+size_t(bb)*sz;
  for(int k=blockIdx.y*blockDim.y+threadIdx.y;k<h;k+=gridDim.y*blockDim.y)
   for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<q;j+=gridDim.x*blockDim.x){
    T s=T(0);for(int i=0;i<h;++i)s+=(transpose?tt[size_t(i)+size_t(k)*ldt]:tt[size_t(k)+size_t(i)*ldt])*ww[size_t(i)+size_t(j)*ldw];
    zz[size_t(k)+size_t(j)*ldz]=s;}
 }
}
template<class T> __global__ void tile_native_D(const T*v,int ldv,long long sv,const T*z,int ldz,long long sz,T*x,int ldx,long long sx,int rows,int h,int q,int count){
 for(long long bb=(long long)blockIdx.z;bb<count;bb+=(long long)gridDim.z){
  const T*vv=v+size_t(bb)*sv,*zz=z+size_t(bb)*sz;T*xx=x+size_t(bb)*sx;
  for(int r=blockIdx.y*blockDim.y+threadIdx.y;r<rows;r+=gridDim.y*blockDim.y)
   for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<q;j+=gridDim.x*blockDim.x){
    T s=T(0);for(int i=0;i<h;++i)s+=vv[size_t(r)+size_t(i)*ldv]*zz[size_t(i)+size_t(j)*ldz];
    xx[size_t(r)+size_t(j)*ldx]-=s;}
 }
}
// Launch helper: 16x16 CTAs over (q columns, h-or-rows), one z-slice per batch.
template<class T> void launch_native_W(const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,T*w,int ldw,long long sw,int rows,int h,int q,int count,cudaStream_t st){
 if(!rows||!h||!q||!count)return;
 tile_native_W<T><<<dim3(ceildiv(q,16),ceildiv(h,16),count),dim3(16,16),0,st>>>(v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count);}
template<class T> void launch_native_Z(const T*t,int ldt,long long stn,const T*w,int ldw,long long sw,T*z,int ldz,long long sz,int h,int q,int count,bool transpose,cudaStream_t st){
 if(!h||!q||!count)return;
 tile_native_Z<T><<<dim3(ceildiv(q,16),ceildiv(h,16),count),dim3(16,16),0,st>>>(t,ldt,stn,w,ldw,sw,z,ldz,sz,h,q,count,transpose);}
template<class T> void launch_native_D(const T*v,int ldv,long long sv,const T*z,int ldz,long long sz,T*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st){
 if(!rows||!h||!q||!count)return;
 tile_native_D<T><<<dim3(ceildiv(q,16),ceildiv(rows,16),count),dim3(16,16),0,st>>>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count);}
// One peer per (leaf i, contraction part z). The output slices are z-major so
// that the additive combine leaves the result where a plain count-batched
// product would have left it, and the Z product that consumes it needs no
// knowledge of the carrier.
template<class T> __global__ void tile_carrier_pointers(T**pp,T*a,long long sa,long long pa,T*b,long long sb,long long pb,T*c,long long sc,long long slice,int count,int rep){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count*rep)return;
 int i=j/rep,z=j-i*rep;const int peers=count*rep;
 pp[j]=a+(long long)i*sa+(long long)z*pa;
 pp[peers+j]=b+(long long)i*sb+(long long)z*pb;
 pp[2*peers+j]=c+(long long)z*slice+(long long)i*sc;
}
// Additive combine of the c partials of W, in place over the first slice. Admissible because the
// row ranges are disjoint and their union is the whole contraction set.
template<class T> __global__ void tile_combine_partials(T*w,size_t n,int c,Witness*wit=nullptr){
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x){
  T s=w[i];for(int z=1;z<c;++z)s+=w[i+size_t(z)*n];w[i]=s;}
 if(wit&&threadIdx.x==0&&blockIdx.x==0)atomicAdd(&wit->device_combines,1ULL);
}
// --------------------------------------------------------------------------- W = V^T X. [K] is the ROW index, cut
// into `rep` disjoint ranges whose partials are summed by the additive combine. `w` holds the rep partials z-major,
// so partial z=0 is already where an unreplicated product would have left it and the Z stage needs no knowledge of
// the carrier.
//
// `block_c` is the BLOCK-level carrier. The rep<=1 branch runs the native block-carried kernel, never cuBLAS. Returns
// the block carrier the native kernel actually executed (0 when the GPU-level cuBLAS path ran, whose block geometry
// is vendor-internal and is left empty, never invented), so the caller can record the launch-derived partition proof.
//
// TQR_SIMT_COMPOSE=0 : IEEE fp32 compose_G on the cuBLAS c>1 intermediary instead of the SIMT W carrier. TQR_PIPE=1
// (default): compose G into the second generation BEFORE retiring B(G-1), then window A(G) alone, then
//                         +0.5% at fp64 24000^2 and +3.1% at 16384^2, the far CTAs occupy SMs A needs; mode 2).
// TQR_PIPE=2 : compose early AND fork B(G) beside A(G) (history cursor chained).
inline int pipe_mode(){static const int v=[]{const char*e=std::getenv("TQR_PIPE");return e?std::atoi(e):1;}();return v;}
inline bool pipe_enabled(){return pipe_mode()!=0;}
inline bool simt_compose_enabled(){static const bool v=[]{const char*e=std::getenv("TQR_SIMT_COMPOSE");return !(e&&std::string(e)=="0");}();return v;}
template<class T> constexpr bool native_carrier_precision(){return std::is_same_v<T,double>||std::is_same_v<T,float>;}
template<class T> constexpr bool native_wz_precision(){return std::is_same_v<T,double>||std::is_same_v<T,float>;}
// Runtime: fp32 W/Z are native only on tensor cores (fp32-math tf32 or x3); IEEE fp32 keeps the
// cuBLAS c>1 intermediary.
template<class T> bool native_wz_enabled(){
 if constexpr(std::is_same_v<T,double>)return true;
 else if constexpr(std::is_same_v<T,float>)return fp32_math()!=Fp32Math::IEEE;
 else return false;}
template<class T> bool native_w_path(int rep,int rows){
 if constexpr(native_wz_precision<T>())return native_wz_enabled<T>()&&rep>1&&carrier_w_supported(rep,rows);
 else return false;
}
template<class T> bool native_z_path(int zrep,int h){
 if constexpr(native_wz_precision<T>())return native_wz_enabled<T>()&&zrep>1&&(carrier_z_supported<T>(zrep,h)||short_carrier_path<T>(zrep,h,false));
 else return false;
}
// fp64 Z = T^T W on the vendor intermediary for wide strips: the zrep peers (disjoint K_z of the reflector index h)
// are ONE batched default-math GEMM (all peers' CTAs in one grid, i.e. concurrent), then the fixed-order additive
// combine and one commit (tile_combine_partials) -- the path IEEE fp32 already takes. nsys fp64 65536: the native
// CUTLASS Z (16x8x4 DMMA, 64x64 tiles) ran at ~12 TF, 0.135 s. Ablation TQR_Z_VENDOR=0; applies from h >= 256 and q
// >= TQR_Z_VENDOR_MINQ (default 4096).
template<class T> bool vendor_z(int q){
 if constexpr(!std::is_same_v<T,double>)return false;
 static const bool on=[]{const char*e=std::getenv("TQR_Z_VENDOR");return e&&std::string(e)=="1";}();
 static const int minq=[]{const char*e=std::getenv("TQR_Z_VENDOR_MINQ");return e?std::atoi(e):4096;}();
 return on&&q>=minq;}
template<class T> bool native_z_path_q(int zrep,int h,int q){return native_z_path<T>(zrep,h)&&!(h>=256&&vendor_z<T>(q));}
// W's c>1 contraction carrier with CONCURRENT vendor peers.
struct PeerLanes{const cudaStream_t*s;const cublasHandle_t*h;cudaEvent_t fork;const cudaEvent_t*join;int n;};
inline bool d_peers_enabled(){static const bool v=[]{const char*e=std::getenv("TQR_D_PEERS");return e&&std::string(e)=="1";}();return v;}
inline int d_peers_minh(){static const int v=[]{const char*e=std::getenv("TQR_D_PEERS_MINH");return e?std::max(2,std::stoi(e)):384;}();return v;}
template<class T> __global__ void d_combine_commit(const T*partial,T*x,int ldx,int rows,int q,Witness*wit){
 const size_t extent=size_t(rows)*q;
 for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<extent;i+=size_t(gridDim.x)*blockDim.x){
  const int row=i%rows,col=i/rows;
  const T d=partial[i]+partial[extent+i];
  x[row+size_t(col)*ldx]-=d;
 }
 if(wit&&blockIdx.x==0&&threadIdx.x==0)atomicAdd(&wit->device_physical_commits,1ULL);
}
// GPU-level c=2, immutable input versions, each partial in a distinct lane credit.
// Both peers are launched on independent streams before the join; neither peer
// writes X. The only X write is the fixed-order P0+P1 owner commit after the join.
template<class T> void peer_stage_D(cublasHandle_t bh,cudaStream_t st,const T*v,int ldv,
 const T*zt,int ldz,T*x,int ldx,int rows,int h,int q,T*partial,const PeerLanes&lanes,Witness*wit){
 if(h<2||lanes.n<1)throw std::runtime_error("D_peer_contract");
 const int cut=h/2;const size_t extent=size_t(rows)*q;
 CU(cudaEventRecord(lanes.fork,st));CU(cudaStreamWaitEvent(lanes.s[0],lanes.fork,0));
 tile_gemm2<T>(lanes.h[0],CUBLAS_OP_N,CUBLAS_OP_T,rows,q,h-cut,T(1),v+size_t(cut)*ldv,ldv,0,
               zt+size_t(cut)*ldz,ldz,0,T(0),partial+extent,rows,0,1);
 CU(cudaEventRecord(lanes.join[0],lanes.s[0]));
 tile_gemm2<T>(bh,CUBLAS_OP_N,CUBLAS_OP_T,rows,q,cut,T(1),v,ldv,0,zt,ldz,0,T(0),partial,rows,0,1);
 CU(cudaStreamWaitEvent(st,lanes.join[0],0));
 d_combine_commit<T><<<264,256,0,st>>>(partial,x,ldx,rows,q,wit);CU(cudaGetLastError());
}
template<class T> int peer_stage_W(cublasHandle_t bh,cudaStream_t st,int rep,const T*v,int ldv,long long sv,
  const T*x,int ldx,long long sx,T*w,int ldw,long long sw,int rows,int h,int q,int count,const PeerLanes&lanes,Witness*wit=nullptr){
 if(rep<2)throw std::runtime_error("peer_stage_W_needs_c_gt_1");
 if(rep-1>lanes.n)throw std::runtime_error("peer_stage_W_lanes");
 if(!rows||!h||!q||!count)return 0;
 const int part=rows/rep;
 if(part<1)throw std::runtime_error("peer_stage_W_empty_peer");
 const long long slice=(long long)count*sw;
 CU(cudaEventRecord(lanes.fork,st));
 for(int z=1;z<rep;++z)CU(cudaStreamWaitEvent(lanes.s[z-1],lanes.fork,0));
 for(int z=rep-1;z>=0;--z){const int k0=z*part,kz=(z==rep-1)?rows-k0:part;
  tile_gemm<T>(z?lanes.h[z-1]:bh,CUBLAS_OP_T,h,q,kz,T(1),v+k0,ldv,sv,x+k0,ldx,sx,T(0),w+size_t(z)*slice,ldw,sw,count);
  if(z){CU(cudaEventRecord(lanes.join[z-1],lanes.s[z-1]));}}
 for(int z=1;z<rep;++z)CU(cudaStreamWaitEvent(st,lanes.join[z-1],0));
 tile_combine_partials<T><<<264,256,0,st>>>(w,size_t(slice),rep,wit);
 CU(cudaGetLastError());
 return 0;
}
template<class T> int apply_stage_W(cublasHandle_t bh,cudaStream_t st,int rep,
  const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
  T*w,int ldw,long long sw,int rows,int h,int q,int count,T**pp,uint64_t*remainder,int block_c=2,Witness*wit=nullptr){
 if(rep<=1)return launch_carried_W<T>(v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count,block_c,st,wit);
 if constexpr(native_wz_precision<T>()){
  if(native_w_path<T>(rep,rows))return launch_carrier_w<T>(rep,v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count,st,wit);
 }
 const int part=rows/rep;
 // count==1: peer z is one contraction range, expressible as a single
 // strided-batched GEMM with the part as the batch stride.
 if(count==1)
  tile_gemm<T>(bh,CUBLAS_OP_T,h,q,part,1,v,ldv,(long long)part,x,ldx,(long long)part,0,w,ldw,sw,rep);
 else{
  // count>1: peer (i,z) sits at operand offset i*s_batch + z*part, which is TWO
  // strides and therefore not a strided-batched call. The peers go to cuBLAS as
  // one batch of pointers so that they run concurrently; issuing them as
  // separate calls on one stream serialises them and buys nothing but c times
  // the output writes.
  const int peers=count*rep;
  tile_carrier_pointers<T><<<ceildiv(peers,128),128,0,st>>>(pp,
    const_cast<T*>(v),sv,(long long)part,const_cast<T*>(x),sx,(long long)part,
    w,sw,(long long)count*sw,count,rep);
  tile_gemm_ptr<T>(bh,CUBLAS_OP_T,h,q,part,T(1),(const T*const*)pp,ldv,(const T*const*)(pp+peers),ldx,T(0),pp+2*peers,ldw,peers);
 }
 // A rep that does not divide rows leaves a remainder range; one extra
 // unreplicated peer covers it, so the union of the ranges is still the whole
 // contraction set, which is the additive combine's admission condition.
 if(part*rep<rows){if(remainder)++*remainder;
  tile_gemm<T>(bh,CUBLAS_OP_T,h,q,rows-part*rep,1,v+size_t(part)*rep,ldv,sv,x+size_t(part)*rep,ldx,sx,1,w,ldw,sw,count);}
 tile_combine_partials<T><<<256,256,0,st>>>(w,size_t(count)*size_t(sw),rep,wit);
 return 0;
}
// Z = T^T W with its own carrier. [K] is the reflector index h, cut into zrep disjoint ranges. Both
// operand offsets are a single stride from the batch base, so this is the pointer-array form.
// Block-level carrier and return contract as in apply_stage_W.
//
// Same contraction cut, same peers, same combine; only the output layout changes: Zt = W^T op(T)^T,
// so A = W (op T) and B = T (op N when Z = T^T W, op T when Z = T W).
template<class T> int apply_stage_Z(cublasHandle_t bh,cudaStream_t st,int zrep,
  const T*t,int ldt,long long stride_t,const T*w,int ldw,long long sw,
  T*z,int ldz,long long sz,int h,int q,int count,bool transpose,T**pp,int block_c=2,Witness*wit=nullptr,bool zt_out=false){
 if(zrep<=1)return launch_carried_Z<T>(t,ldt,stride_t,w,ldw,sw,z,ldz,sz,h,q,count,transpose,block_c,st,wit,zt_out);
 if(short_carrier_path<T>(zrep,h,false)){
  if(zt_out)return launch_short_carrier<T,false,true>(zrep,t,ldt,stride_t,w,ldw,sw,z,ldz,sz,h,h,q,count,transpose,st,wit);
  return launch_short_carrier<T,false>(zrep,t,ldt,stride_t,w,ldw,sw,z,ldz,sz,h,h,q,count,transpose,st,wit);
 }
 if constexpr(native_wz_precision<T>()){
  if(native_z_path_q<T>(zrep,h,q))return launch_carrier_z<T>(zrep,t,ldt,stride_t,w,ldw,sw,z,ldz,sz,h,q,count,transpose,zt_out,st,wit);
 }
 const int zpart=h/zrep,peers=count*zrep;
 if(zt_out){
  tile_carrier_pointers<T><<<ceildiv(peers,128),128,0,st>>>(pp,
    const_cast<T*>(w),sw,(long long)zpart,
    const_cast<T*>(t),stride_t,transpose?(long long)zpart:(long long)zpart*ldt,
    z,sz,(long long)count*sz,count,zrep);
  const cublasOperation_t opb=transpose?CUBLAS_OP_N:CUBLAS_OP_T;
  tile_gemm_ptr2<T>(bh,CUBLAS_OP_T,opb,q,h,zpart,T(1),(const T*const*)pp,ldw,(const T*const*)(pp+peers),ldt,T(0),pp+2*peers,ldz,peers);
  if(zpart*zrep<h)tile_gemm2<T>(bh,CUBLAS_OP_T,opb,q,h,h-zpart*zrep,1,w+size_t(zpart)*zrep,ldw,sw,
    t+(transpose?size_t(zpart)*zrep:size_t(zpart)*zrep*ldt),ldt,stride_t,1,z,ldz,sz,count);
  tile_combine_partials<T><<<256,256,0,st>>>(z,size_t(count)*size_t(sz),zrep,wit);
  return 0;
 }
 tile_carrier_pointers<T><<<ceildiv(peers,128),128,0,st>>>(pp,
   const_cast<T*>(t),stride_t,transpose?(long long)zpart:(long long)zpart*ldt,
   const_cast<T*>(w),sw,(long long)zpart,
   z,sz,(long long)count*sz,count,zrep);
 tile_gemm_ptr<T>(bh,transpose?CUBLAS_OP_T:CUBLAS_OP_N,h,q,zpart,T(1),(const T*const*)pp,ldt,(const T*const*)(pp+peers),ldw,T(0),pp+2*peers,ldz,peers);
 if(zpart*zrep<h)tile_gemm<T>(bh,transpose?CUBLAS_OP_T:CUBLAS_OP_N,h,q,h-zpart*zrep,1,
   t+(transpose?size_t(zpart)*zrep:size_t(zpart)*zrep*ldt),ldt,stride_t,w+size_t(zpart)*zrep,ldw,sw,1,z,ldz,sz,count);
 tile_combine_partials<T><<<256,256,0,st>>>(z,size_t(count)*size_t(sz),zrep,wit);
 return 0;
}
// D = V Z with its own carrier, cutting the same [K]=h. Block-level carrier and return contract as
// in apply_stage_W (0 = cuBLAS path, vendor-internal block geometry left empty).
//
// The drep>1 arm with zt_in runs the NATIVE carried D (include/carrier_gemm.cuh): block-level c =
// drep warp slices, fixed-order shared-memory combine, ONE commit, device counters. `z` is then Zt
// (q x h, ld ldz). Returns the executed block c. The sequential library arm below remains only
// where native_d_path() is false (h too small for c non-empty slices is refused before reaching
// here; other precisions), and is recorded as library commits.
template<class T> constexpr bool native_d_precision(){return native_carrier_precision<T>();}
template<class T> bool native_d_path(int drep,int h){
 if constexpr(native_d_precision<T>())return drep>1&&(carrier_d_supported<T>(drep,h)||short_carrier_path<T>(drep,h,true));
 else return false;
}
inline int& last_d_cluster(){static thread_local int v=0;return v;}
// All choices retain disjoint concurrent contraction halves, replicated operands, fixed-order
// shared-memory combine and one X owner commit.
inline int kami_d_tile(){static const int tile=[]{
 const char* e=std::getenv("TQR_KAMI_D_TILE");if(!e||std::string(e)=="64x128")return 0;
 const std::string s=e;
 if(s=="64x64")return 1;if(s=="64x64s2")return 2;
 if(s=="64x128nx")return 3;if(s=="64x128k32")return 4;if(s=="64x128sp")return 5;
 if(s=="128x64k16")return 6;
 if(s=="128x64k16sb")return 7;
 throw std::runtime_error("invalid_d_tile");}();return tile;}
inline int kami_d_raster(){static const int raster=[]{
 const char* e=std::getenv("TQR_KAMI_D_RASTER");const int n=e?std::stoi(e):TQR_KAMI_RASTER;
 if(n<1||n>64)throw std::runtime_error("invalid_d_raster");return n;}();return raster;}
inline int simt_raster(){static const int raster=[]{
 const char*e=std::getenv("TQR_SIMT_RASTER");const int n=e?std::stoi(e):TQR_KAMI_RASTER;
 if(n<1||n>64)throw std::runtime_error("invalid_simt_raster");return n;}();return raster;}
template<class T> int apply_stage_D(cublasHandle_t bh,cudaStream_t st,int drep,
  const T*v,int ldv,long long sv,const T*z,int ldz,long long sz,
  T*x,int ldx,long long sx,int rows,int h,int q,int count,int block_c=2,Witness*wit=nullptr,bool zt_in=false){
 last_d_cluster()=0;
 if(zt_in){
  if(!native_d_path<T>(drep,h))throw std::runtime_error("native_carried_D_requested_where_unsupported");
  if(short_carrier_path<T>(drep,h,true))return launch_short_carrier<T,true>(drep,v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,false,st,wit);
  if constexpr(std::is_same_v<T,double>){
   // KAMI-informed block carrier (p_i,p_j,c) = (2,2,2), 32x64 register blocks, DMMA m16n8k8, X staged early,
   // fixed-order combine, one commit (include/kami_carrier.cuh). Wins from h >= 384.
   static const int kami_minh=[]{const char*e=std::getenv("TQR_KAMI_D");if(e&&std::string(e)=="0")return 1<<30;
     const char*m=std::getenv("TQR_KAMI_D_MINH");return m?std::atoi(m):384;}();
   if(drep==2&&h>=kami_minh&&kami::d_admits(v,ldv,sv,z,ldz,sz,x,ldx,sx,h,count)){
    const int raster=kami_d_raster();
    switch(kami_d_tile()){
     case 1:return kami::launch_dnp<kami::DN64x64>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 2:return kami::launch_dnp<kami::DN64x64S2>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 3:return kami::launch_dnp<kami::DN64x128nx>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 4:return kami::launch_dnp<kami::DN64x128K32>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 5:return kami::launch_dsp<kami::DSP64x128>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 6:if(h%32==0)return kami::launch_dnp16<kami::DN128x64K16>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
       return kami::launch_dnp<kami::DN64x128>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     case 7:if(h%32==0)return kami::launch_dnp16<kami::DN128x64K16,true>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
       return kami::launch_dnp<kami::DN64x128>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
     default:return kami::launch_dnp<kami::DN64x128>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,raster);
    }
   }
  }
  if constexpr(std::is_same_v<T,float>){
   // IEEE fp32 D on the CUDA cores as a CLUSTER-PAIR carrier (p_i,p_j,c) = (tiles,tiles,2): the 2 CTAs of a
   // cluster own the balanced contraction halves, st.async+mbarrier additive combine into the owner's half, one
   // commit (include/kami_carrier.cuh simt_dcl_kernel; cuBLAS's 128x128x8 / 16x8-per-thread SIMT tile).
   static const bool simt_d=[]{const char*e=std::getenv("TQR_SIMT_D");return !(e&&std::string(e)=="0");}();
   // BK=16 / 2 stages from h >= 256; BK=8 below.
   if(simt_d&&drep==2&&fp32_math()!=Fp32Math::TF32&&fp32_math()!=Fp32Math::X3){
    if(h>=256&&kami::simt_dcl_admits<kami::SF2K16>(v,ldv,sv,z,ldz,sz,x,ldx,sx,h,count)){
     last_d_cluster()=kami::SF2K16::C;return kami::launch_simt_dcl<kami::SF2K16>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,simt_raster());}
    if(kami::simt_dcl_admits<kami::SF3>(v,ldv,sv,z,ldz,sz,x,ldx,sx,h,count)){
     last_d_cluster()=kami::SF3::C;return kami::launch_simt_dcl<kami::SF3>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit,simt_raster());}
   }
  }
  if constexpr(native_d_precision<T>())return launch_carrier_d<T>(drep,v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,st,wit);
 }
 if(drep<=1)return launch_carried_D<T>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,count,block_c,st,wit);
 if(!rows||!h||!q||!count)return 0;
 const int dpart=h/drep;
 for(int y=0;y<drep;++y)
  tile_gemm<T>(bh,CUBLAS_OP_N,rows,q,dpart,-1,v+size_t(y)*dpart*ldv,ldv,sv,z+size_t(y)*dpart,ldz,sz,1,x,ldx,sx,count);
 if(dpart*drep<h)tile_gemm<T>(bh,CUBLAS_OP_N,rows,q,h-dpart*drep,-1,v+size_t(dpart)*drep*ldv,ldv,sv,z+size_t(dpart)*drep,ldz,sz,1,x,ldx,sx,count);
 return 0;
}
}
