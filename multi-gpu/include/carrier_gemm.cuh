#pragma once
// Native carried GEMM kernels for W, Z and D (CUDA cores and FP64 tensor cores).
// NATIVE CARRIED D.
//  X(rows,q) -= V(rows,h) Z(h,q)  with Z supplied TRANSPOSED (Zt = Z^T, q x h, column-major).
// Paper: a carrier cuts [K] into c balanced parts; partials are added only over disjoint index sets
// whose union is the contraction (590); Alg 2: "D <- V_e Z with its own carrier", "sum the c partials
// over disjoint K_z", "owner commits X <- X - D exactly once".
//
// BLOCK-level carrier: c warp slices of one CTA (launch geometry dim3(slice_threads, c): threadIdx.y is
// the K peer) share one CUTLASS 2.x fp64 multistage cp.async pipeline (DMMA 16x8x4). With r = K mod BK,
// slice w owns
//    K_w = { k < r : floor(k/WK) = w }  u  { k >= r : floor(((k - r) mod BK)/WK) = w },
// WK = BK/c (residue tile first, CUTLASS 2.x convention): disjoint, union [0,K), balanced to within the
// residue tile. After the mainloop every slice parks its register partial in shared memory
// (column-major, rows padded to 64 B mod 128 B), the partials are summed in the fixed order w = 0..c-1
// (bitwise deterministic) and the owner commits X -= sum ONCE per element, coalesced. No HBM partials,
// no atomics into X, no library call. Device evidence per batch member: c peer partials, c block peers,
// one membership report, one physical commit (the same law as the native c=1 carried kernels).
#include <cutlass/cutlass.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/gemm/threadblock/mma_multistage.h>
#include "group_mma_multistage.h"
#include "carried_apply.cuh"
#include <cooperative_groups.h>
#include <type_traits>
namespace tqr {
template<class M> struct CgToGroup;
template<class S,class IA,class SIA,cutlass::arch::CacheOperation::Kind CA,class IB,class SIB,
         cutlass::arch::CacheOperation::Kind CBk,class EC,class LC,class P,int St,
         cutlass::gemm::SharedMemoryClearOption SC,class En>
struct CgToGroup<cutlass::gemm::threadblock::MmaMultistage<S,IA,SIA,CA,IB,SIB,CBk,EC,LC,P,St,SC,En>>{
  using type=p2c::GroupMmaMultistage<S,IA,SIA,CA,IB,SIB,CBk,EC,LC,P,St,SC,En>;};

// Tile BM x BN in CUTLASS's transposed frame (BM along X columns, BN along X rows): X^T(q,rows) -=
// Zt(q,h) V^T(h,rows). fp64: tensor-op DMMA 16x8x4. fp32: SIMT FFMA (TF32 never; Math policies: the
// element type plus the MMA path. `double`/`float` keep their usual meaning (fp64 DMMA; fp32
// IEEE SIMT FFMA); MathTF32 and MathX3 run fp32 storage on TF32 tensor cores (m16n8k8): one TF32
// product, or 3xTF32 (OpMultiplyAddFastF32: a = big + small, big*big + big*small + small*big, fp32
// accumulation).
struct MathTF32{};struct MathX3{};
template<class T> struct CarrierOp;
template<> struct CarrierOp<double>{using Element=double;using Class=cutlass::arch::OpClassTensorOp;using Arch=cutlass::arch::Sm90;using Inst=cutlass::gemm::GemmShape<16,8,4>;using Operator=cutlass::arch::OpMultiplyAdd;static constexpr int Pad=8;};
template<> struct CarrierOp<float>{using Element=float;using Class=cutlass::arch::OpClassSimt;using Arch=cutlass::arch::Sm80;using Inst=cutlass::gemm::GemmShape<1,1,1>;using Operator=cutlass::arch::OpMultiplyAdd;static constexpr int Pad=4;};
template<> struct CarrierOp<MathTF32>{using Element=float;using Class=cutlass::arch::OpClassTensorOp;using Arch=cutlass::arch::Sm80;using Inst=cutlass::gemm::GemmShape<16,8,8>;using Operator=cutlass::arch::OpMultiplyAdd;static constexpr int Pad=4;};
template<> struct CarrierOp<MathX3>{using Element=float;using Class=cutlass::arch::OpClassTensorOp;using Arch=cutlass::arch::Sm80;using Inst=cutlass::gemm::GemmShape<16,8,8>;using Operator=cutlass::arch::OpMultiplyAddFastF32;static constexpr int Pad=4;};
template<class T,int BM_,int BN_,int BK_,int WM_,int WN_,int WK_,int Stages_,int MinB_>
struct CarrierDCfg {
  using Element=typename CarrierOp<T>::Element;
  static constexpr int BM=BM_,BN=BN_,BK=BK_,WK=WK_,Stages=Stages_,MinB=MinB_;
  using DM=cutlass::gemm::threadblock::DefaultMma<
     Element,cutlass::layout::ColumnMajor,1,Element,cutlass::layout::RowMajor,1,Element,cutlass::layout::RowMajor,
     typename CarrierOp<T>::Class,typename CarrierOp<T>::Arch,
     cutlass::gemm::GemmShape<BM,BN,BK>,cutlass::gemm::GemmShape<WM_,WN_,WK_>,typename CarrierOp<T>::Inst,
     Stages,typename CarrierOp<T>::Operator,false,cutlass::gemm::SharedMemoryClearOption::kNone>;
  using Mma=typename CgToGroup<typename DM::ThreadblockMma>::type;
  using WarpCount=typename Mma::Base::WarpCount;
  static constexpr int C=WarpCount::kK;                      // block-level K peers
  static constexpr int WMN=WarpCount::kM*WarpCount::kN;
  static constexpr int SliceThreads=WMN*32,Threads=SliceThreads*C;
  // Parked tile rows padded so fragment row stores hit distinct banks (fp64: 64 B mod 128 B).
  static constexpr int LDP=BN+CarrierOp<T>::Pad;
  static constexpr size_t PartElems=size_t(BM)*LDP;
  static constexpr size_t MainSmem=sizeof(typename Mma::SharedStorage);
  static constexpr size_t PartSmem=size_t(C)*PartElems*sizeof(Element);
  static constexpr size_t Smem=MainSmem>PartSmem?MainSmem:PartSmem;
  static constexpr int PerThread=BM*BN/Threads,U=PerThread<8?PerThread:8;
  static_assert(BM*BN%Threads==0&&PerThread%U==0,"");
  static_assert(WK*C==BK,"slices tile each k-tile exactly");
};
// Every configuration of one (precision, c) uses the SAME partition law (BK = 8c, WK = 8), so
// the choice between output tilings (p_i x p_j) never changes the K cut or the combine order.
using CarrierD2=CarrierDCfg<double,64,64,16,32,32,8,3,2>;    // c=2: 256 threads, 2 CTAs/SM
using CarrierD4=CarrierDCfg<double,64,64,32,32,32,8,3,1>;    // c=4: 512 threads
using CarrierS2=CarrierDCfg<float,128,128,16,32,64,8,3,1>;   // c=2 fp32, saturated shapes
using CarrierS2s=CarrierDCfg<float,64,64,16,32,32,8,3,2>;    // c=2 fp32, short/narrow shapes
using CarrierS4=CarrierDCfg<float,64,64,32,32,32,8,3,1>;     // c=4 fp32
using CarrierT2=CarrierDCfg<MathTF32,128,128,32,64,64,16,3,1>;
using CarrierT4=CarrierDCfg<MathTF32,128,64,64,64,32,16,3,1>;
using CarrierX2=CarrierDCfg<MathX3,128,64,32,64,32,16,3,1>;
using CarrierX4=CarrierDCfg<MathX3,128,64,64,64,32,16,3,1>;
static_assert(CarrierD2::BK==CarrierS2::BK&&CarrierS2::BK==CarrierS2s::BK&&CarrierD2::WK==CarrierS2::WK,"one law per c");
static_assert(CarrierD4::BK==CarrierS4::BK&&CarrierD4::WK==CarrierS4::WK,"one law per c");

// Output-tile rasterization (output tiling only -- never the K cut, the combine order or the commit):
// the linear launch order L = x + nx*y is remapped so one wave's operand working set stays in L2.
// carrier_raster_x: groups of G x-tiles, all y-tiles of a group before the next group (x fastest
// inside a group). carrier_raster_y: y fastest (all y-tiles of an x-tile run back to back).
__device__ __forceinline__ void carrier_raster_x(int G,int&tx,int&ty){
  const int nx=gridDim.x,ny=gridDim.y,L=blockIdx.x+nx*blockIdx.y,span=G*ny,grp=L/span,in=L%span,gw=min(G,nx-grp*G);
  tx=grp*G+in%gw;ty=in/gw;}
__device__ __forceinline__ void carrier_raster_y(int&tx,int&ty){
  const int nx=gridDim.x,ny=gridDim.y,L=blockIdx.x+nx*blockIdx.y;tx=L/ny;ty=L%ny;}
#ifndef TQR_D_RASTER
#define TQR_D_RASTER 16
#endif
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,Cfg::MinB)
carrier_d_kernel(const typename Cfg::Element* __restrict__ V,int ldv,long long sv,const typename Cfg::Element* __restrict__ Zt,int ldzt,long long szt,
                 typename Cfg::Element* X,int ldx,long long sx,int rows,int h,int q,Witness* wit,int raster){
  using T=typename Cfg::Element;
  extern __shared__ __align__(128) char carrier_d_smem[];
  char* smem=carrier_d_smem;
  using Mma=typename Cfg::Mma;
  constexpr int BM=Cfg::BM,BN=Cfg::BN;
  const int tid=threadIdx.x+Cfg::SliceThreads*threadIdx.y;   // threadIdx.y = K peer (slice)
  const int batch=blockIdx.z;
  V+=batch*sv;Zt+=batch*szt;X+=batch*sx;
  // raster = column tiles per group (launcher: TQR_D_RASTER when Zt exceeds half the L2, else all
  // column tiles = the plain order). At q*K beyond L2 (e.g. 16384 x 512 fp64 = 64 MB) the plain order
  // re-read the whole Zt from HBM for every row block; below it the plain order is faster (K=128).
  int tcol,trow;carrier_raster_x(raster,tcol,trow);
  const int n0=tcol*BM,m0=trow*BN;                            // X columns / X rows of the tile
  {                                                          // X tile -> L2 while the mainloop runs
    const int rr=min(BN,rows-m0),cc=min(BM,q-n0);
    constexpr int LPC=(BN*int(sizeof(T))+127)/128,RPL=128/int(sizeof(T));
    for(int l=tid;l<cc*LPC;l+=Cfg::Threads){const int col=l/LPC,r=(l%LPC)*RPL;
      if(r<rr)asm volatile("prefetch.global.L2 [%0];"::"l"(X+(m0+r)+(long long)(n0+col)*ldx));}
  }
  typename Mma::IteratorA itA(typename Mma::IteratorA::Params(cutlass::layout::ColumnMajor(ldzt)),
     const_cast<T*>(Zt),{q,h},tid,{n0,0});
  typename Mma::IteratorB itB(typename Mma::IteratorB::Params(cutlass::layout::RowMajor(ldv)),
     const_cast<T*>(V),{h,rows},tid,{0,m0});
  const int iters=(h+Cfg::BK-1)/Cfg::BK;
  const int warp=__shfl_sync(0xffffffffu,tid/32,0),lane=threadIdx.x%32;
  typename Mma::FragmentC acc;acc.clear();
  {
    typename Mma::SharedStorage& ss=*reinterpret_cast<typename Mma::SharedStorage*>(smem);
    Mma mma(ss,tid,warp,lane,0);
    if(iters>0)mma(iters,acc,itA,itB,acc);
  }
  __syncthreads();                                           // pipeline SMEM is dead
  T* part=reinterpret_cast<T*>(smem);
  {
    const int wmn=warp%Cfg::WMN,wk=warp/Cfg::WMN;            // wk == threadIdx.y
    typename Mma::Operator::IteratorC itC({part+size_t(wk)*Cfg::PartElems,cutlass::layout::RowMajor(Cfg::LDP)},lane);
    itC.add_tile_offset({wmn%Cfg::WarpCount::kM,wmn/Cfg::WarpCount::kM});
    itC.store(acc);
  }
  __syncthreads();
  const bool full=(m0+BN<=rows)&&(n0+BM<=q);
  #pragma unroll 1
  for(int b0=0;b0<Cfg::PerThread;b0+=Cfg::U){
    T xv[Cfg::U];int off[Cfg::U];long long gx[Cfg::U];bool in[Cfg::U];
    #pragma unroll
    for(int u=0;u<Cfg::U;++u){
      const int e=tid+(b0+u)*Cfg::Threads,i=e/BN,j=e%BN;    // i: tile column (X column), j: X row
      off[u]=i*Cfg::LDP+j;gx[u]=(m0+j)+(long long)(n0+i)*ldx;
      in[u]=full||((n0+i)<q&&(m0+j)<rows);
      xv[u]=in[u]?X[gx[u]]:T(0);
    }
    #pragma unroll
    for(int u=0;u<Cfg::U;++u){
      T s=0;
      #pragma unroll
      for(int w=0;w<Cfg::C;++w)s+=part[w*Cfg::PartElems+off[u]];   // fixed order w = 0..c-1
      xv[u]-=s;
    }
    #pragma unroll
    for(int u=0;u<Cfg::U;++u)if(in[u])X[gx[u]]=xv[u];         // owner commits once
  }
  if(wit&&tid==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],(unsigned long long)Cfg::C);
    witness_levels(wit,4,int(blockDim.y),1,1,1);    // executed geometry: block c = slices, one CTA per tile
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
template<class Cfg> void prepare_carrier_d(){
  static int last_device=-1;int device;CU(cudaGetDevice(&device));
  if(device==last_device)return;
  CU(cudaFuncSetAttribute(carrier_d_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(carrier_d_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last_device=device;
}
// Smallest h for which every one of the c slices owns a nonempty K_w (same law for both
// precisions: BK = 8c, WK = 8).
// The partition law (BK, WK) of the configuration that will run: fp64 and IEEE fp32 BK=8c, WK=8;
// tf32 c=2 (32,16), c=4 (64,16); x3 c=2 (32,16), c=4 (64,16).
template<class T> void carrier_d_law(int c,int&bk,int&wk){
  if constexpr(std::is_same_v<T,double>){bk=c==2?CarrierD2::BK:CarrierD4::BK;wk=c==2?CarrierD2::WK:CarrierD4::WK;}
  else{switch(fp32_math()){
    case Fp32Math::TF32:bk=c==2?CarrierT2::BK:CarrierT4::BK;wk=c==2?CarrierT2::WK:CarrierT4::WK;break;
    case Fp32Math::X3:bk=c==2?CarrierX2::BK:CarrierX4::BK;wk=c==2?CarrierX2::WK:CarrierX4::WK;break;
    default:bk=c==2?CarrierS2::BK:CarrierS4::BK;wk=c==2?CarrierS2::WK:CarrierS4::WK;}}
}
template<class T> bool carrier_d_supported(int c,int h){if(c!=2&&c!=4)return false;int bk,wk;carrier_d_law<T>(c,bk,wk);return h>=bk-wk+1;}
template<class Cfg> void launch_carrier_d_cfg(const typename Cfg::Element*v,int ldv,long long sv,const typename Cfg::Element*zt,int ldzt,long long szt,
  typename Cfg::Element*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit){
  prepare_carrier_d<Cfg>();
  static int l2=0;if(!l2){int dev=0;CU(cudaGetDevice(&dev));CU(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev));}
  const int nx=ceildiv(q,Cfg::BM);
  const int raster=size_t(q)*h*sizeof(typename Cfg::Element)*2>size_t(l2)?TQR_D_RASTER:nx;
  carrier_d_kernel<Cfg><<<dim3(nx,ceildiv(rows,Cfg::BN),count),dim3(Cfg::SliceThreads,Cfg::C),Cfg::Smem,st>>>
    (v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,wit,raster);
  CU(cudaGetLastError());
}
// Output tiling only (p_i x p_j), never the K cut: fp32 c=2 uses 128x128 CTAs (1/SM) when they fill
// two waves, else 64x64.
inline int carrier_d_sms(){static int sms=0;if(!sms){int dev=0;CU(cudaGetDevice(&dev));CU(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));}return sms;}
template<class T> bool carrier_d_big_tile(int c,int rows,int q,int count){
  if constexpr(!std::is_same_v<T,float>)return false;
  else return c==2&&(long long)ceildiv(q,CarrierS2::BM)*ceildiv(rows,CarrierS2::BN)*count>=2LL*carrier_d_sms();
}
template<class T> int carrier_d_slice_threads(int c,int rows,int q,int count){
  if constexpr(std::is_same_v<T,double>)return c==2?CarrierD2::SliceThreads:CarrierD4::SliceThreads;
  else{
    if(fp32_math()==Fp32Math::TF32)return c==2?CarrierT2::SliceThreads:CarrierT4::SliceThreads;
    if(fp32_math()==Fp32Math::X3)return c==2?CarrierX2::SliceThreads:CarrierX4::SliceThreads;
    return c==4?CarrierS4::SliceThreads:carrier_d_big_tile<T>(c,rows,q,count)?CarrierS2::SliceThreads:CarrierS2s::SliceThreads;}
}
// Launch the native carried D; returns the executed block-level c.
template<class T> int launch_carrier_d(int c,const T*v,int ldv,long long sv,const T*zt,int ldzt,long long szt,
  T*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit){
  if(!carrier_d_supported<T>(c,h))throw std::runtime_error("carrier_d_unsupported_c_or_h");
  if(!rows||!q||!count)return c;
  if constexpr(std::is_same_v<T,double>){
    if(c==2)launch_carrier_d_cfg<CarrierD2>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
    else launch_carrier_d_cfg<CarrierD4>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
  }else if(fp32_math()==Fp32Math::TF32){
    if(c==2)launch_carrier_d_cfg<CarrierT2>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
    else launch_carrier_d_cfg<CarrierT4>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
  }else if(fp32_math()==Fp32Math::X3){
    if(c==2)launch_carrier_d_cfg<CarrierX2>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
    else launch_carrier_d_cfg<CarrierX4>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
  }else{
    if(c==4)launch_carrier_d_cfg<CarrierS4>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
    else if(carrier_d_big_tile<T>(c,rows,q,count))launch_carrier_d_cfg<CarrierS2>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
    else launch_carrier_d_cfg<CarrierS2s>(v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,h,q,count,st,wit);
  }
  return c;
}

// ============================================================================================
// GENERIC native carried product (W and Z): Out (col-major) = Aorig * Borig, computed in CUTLASS's
// transposed frame C' = Out^T = A' B' (A' = Borig^T, B' = Aorig^T; layouts LA, LB). Alg 2 "W, Z each
// with its own carrier"):
//  * CLUSTER level: CL CTAs of one thread-block cluster share an output tile; CTA z owns the
//    contiguous balanced K range [floor(K z/CL), floor(K (z+1)/CL)).
//  * BLOCK level: SK warp slices of each CTA share one multistage pipeline; inside the CTA's range
//    (r = |K_z| mod BK, residue tile first) slice w owns
//    {k < r : floor(k/WK) = w} u {k >= r : floor(((k - r) mod BK)/WK) = w}.
//  Combine: partials parked in SMEM; after barrier.cluster arrive.release / wait.acquire, CTA z
//  sums its row slice of the tile over pieces (z', w) in the fixed order z' = 0..CL-1, w = 0..SK-1
//  through distributed shared memory and commits it once (store). No HBM partials, no atomics
//  on data, bitwise deterministic. The executed product c = CL * SK.
// Device evidence per batch member: CL*SK peer partials, SK block peers, one membership report; one
// additive combine per launch.
enum CarrierEpi{CEPI_STORE=0,CEPI_SUB=1};
template<class T,class LA_,class LB_,int BM_,int BN_,int BK_,int WM_,int WN_,int WK_,int Stages_,int MinB_,int Epi_,class Instruction_=typename CarrierOp<T>::Inst>
struct CarrierGCfg {
  using Element=typename CarrierOp<T>::Element;using LA=LA_;using LB=LB_;
  static constexpr int BM=BM_,BN=BN_,BK=BK_,WK=WK_,Stages=Stages_,MinB=MinB_,Epi=Epi_;
  using DM=cutlass::gemm::threadblock::DefaultMma<Element,LA,1,Element,LB,1,Element,cutlass::layout::RowMajor,
     typename CarrierOp<T>::Class,typename CarrierOp<T>::Arch,
     cutlass::gemm::GemmShape<BM,BN,BK>,cutlass::gemm::GemmShape<WM_,WN_,WK_>,Instruction_,
     Stages,typename CarrierOp<T>::Operator,false,cutlass::gemm::SharedMemoryClearOption::kNone>;
  using Mma=typename CgToGroup<typename DM::ThreadblockMma>::type;
  using WarpCount=typename Mma::Base::WarpCount;
  static constexpr int SK=WarpCount::kK,WMN=WarpCount::kM*WarpCount::kN;
  static constexpr int SliceThreads=WMN*32,Threads=SliceThreads*SK;
  static constexpr int LDP=BN+CarrierOp<T>::Pad;
  static constexpr size_t PartElems=size_t(BM)*LDP;
  static constexpr size_t MainSmem=sizeof(typename Mma::SharedStorage);
  static constexpr size_t PartSmem=size_t(SK)*PartElems*sizeof(Element);
  static constexpr size_t Smem=MainSmem>PartSmem?MainSmem:PartSmem;
  static_assert(WK*SK==BK,"slices tile each k-tile exactly");
};
// GPU level: CG > 1 splits [K] further into CG x CL contiguous balanced ranges (peer pid = g*CL + z
// owns [floor(K pid/(CG CL)), floor(K (pid+1)/(CG CL)))); group g writes its tile sum into the HBM
// partial slice P + (batch*CG + g)*sp (same indexing as O), and carrier_g_combine sums the slices in
// the fixed order g = 0..CG-1 and commits O once. No atomics on data.
template<class T> struct CarrierGArgs{const T*A;int lda;long long sa;const T*B;int ldb;long long sb;T*O;int ldo;long long so;int M,N,K;Witness*wit;int kind;int CG=1;T*P=nullptr;long long sp=0;};
template<class Cfg,int CL> __global__ void __launch_bounds__(Cfg::Threads,Cfg::MinB)
carrier_g_kernel(const __grid_constant__ CarrierGArgs<typename Cfg::Element> a){
  using T=typename Cfg::Element;using Mma=typename Cfg::Mma;constexpr int BM=Cfg::BM,BN=Cfg::BN;
  static_assert(BM%CL==0,"cluster row slices");
  extern __shared__ __align__(128) char carrier_g_smem[];char*smem=carrier_g_smem;
  const int tid=threadIdx.x+Cfg::SliceThreads*threadIdx.y;
  int z=0;if constexpr(CL>1)z=int(cooperative_groups::this_cluster().block_rank());
  // N (= the product's h side for W) fastest: the N tiles of one M tile run back to back and share
  // their A' (X strip) stream in L2 instead of re-reading X once per N tile at composed h = g b.
  int tm_,tn_;carrier_raster_y(tm_,tn_);
  const int zz=blockIdx.z/CL,batch=zz/a.CG,gz=zz%a.CG,peers=CL*a.CG,pid=gz*CL+z,m0=tm_*BM,n0=tn_*BN;
  const T*Ap=a.A+batch*a.sa;const T*Bp=a.B+batch*a.sb;
  T*O=a.CG>1?a.P+(size_t(batch)*a.CG+gz)*a.sp:a.O+batch*a.so;
  const bool sub=Cfg::Epi==CEPI_SUB&&a.CG==1;                 // with CG>1 the combine kernel applies the epilogue
  const int k0=int((long long)a.K*pid/peers),k1=int((long long)a.K*(pid+1)/peers);
  typename Mma::IteratorA itA(typename Mma::IteratorA::Params(typename Cfg::LA(a.lda)),const_cast<T*>(Ap),{a.M,k1},tid,{m0,k0});
  typename Mma::IteratorB itB(typename Mma::IteratorB::Params(typename Cfg::LB(a.ldb)),const_cast<T*>(Bp),{k1,a.N},tid,{k0,n0});
  const int iters=(k1-k0+Cfg::BK-1)/Cfg::BK;
  const int warp=__shfl_sync(0xffffffffu,tid/32,0),lane=threadIdx.x%32;
  typename Mma::FragmentC acc;acc.clear();
  {auto&ss=*reinterpret_cast<typename Mma::SharedStorage*>(smem);Mma mma(ss,tid,warp,lane,0);if(iters>0)mma(iters,acc,itA,itB,acc);}
  __syncthreads();
  T*part=reinterpret_cast<T*>(smem);
  {const int wmn=warp%Cfg::WMN,wk=warp/Cfg::WMN;
   typename Mma::Operator::IteratorC itC({part+size_t(wk)*Cfg::PartElems,cutlass::layout::RowMajor(Cfg::LDP)},lane);
   itC.add_tile_offset({wmn%Cfg::WarpCount::kM,wmn/Cfg::WarpCount::kM});itC.store(acc);}
  if constexpr(CL>1){asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");}
  else __syncthreads();
  constexpr int SL=BM/CL;
  const int r0=z*SL;const bool full=(m0+BM<=a.M)&&(n0+BN<=a.N);
  for(int e=tid;e<SL*BN;e+=Cfg::Threads){
    const int i=r0+e/BN,j=e%BN;const long long go=(n0+j)+(long long)(m0+i)*a.ldo;
    const bool in=full||((m0+i)<a.M&&(n0+j)<a.N);
    T v=(sub&&in)?O[go]:T(0);T s=0;
    #pragma unroll
    for(int r=0;r<CL;++r){                                   // fixed order z'=0..CL-1, w=0..SK-1
      const T*peer=part;
      if constexpr(CL>1)peer=cooperative_groups::this_cluster().map_shared_rank(part,r);
      #pragma unroll
      for(int w=0;w<Cfg::SK;++w)s+=peer[w*Cfg::PartElems+i*Cfg::LDP+j];
    }
    if(in)O[go]=sub?v-s:s;                                   // owner commits once (or its slice)
  }
  if constexpr(CL>1){asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");asm volatile("barrier.cluster.wait.aligned;":::"memory");}
  if(a.wit&&tid==0&&z==0&&gz==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&a.wit->device_peer_partials[a.kind],(unsigned long long)(CL*Cfg::SK*a.CG));
    atomicAdd(&a.wit->device_block_peers[a.kind],(unsigned long long)Cfg::SK);
    unsigned cl_dim=1;if constexpr(CL>1)cl_dim=cooperative_groups::this_cluster().dim_blocks().z;
    // executed geometry: slices (blockDim.y), cluster CTAs (cluster dim), GPU-level groups (CG, which cut [K])
    witness_levels(a.wit,a.kind,int(blockDim.y),int(cl_dim),a.CG,1);
    atomicAdd(&a.wit->device_membership_reports,1ULL);
    if(batch==0)atomicAdd(&a.wit->device_combines,1ULL);    // one additive combine per carried issue
  }
}
// GPU-level fixed-order combine: O (+)= sum_{g=0..CG-1} P_g, one commit per element.
template<class T> __global__ void carrier_g_combine(const T*P,long long sp,int CG,T*O,int ldo,long long so,int M,int N,int count,bool sub){
  // blocks walk (batch, i) lines, threads walk j: the divides are per line, not per element. Each
  // output sums its CG partials in the same g = 0..CG-1 order: bit-identical combine.
  const long long lines=(long long)M*count;
  for(long long L=blockIdx.x;L<lines;L+=gridDim.x){
    const int batch=int(L/M),i=int(L-(long long)batch*M);const long long base=(long long)i*ldo;
    const T*srcb=P+size_t(batch)*CG*sp;T*dstb=O+batch*so;
    for(int j=threadIdx.x;j<N;j+=blockDim.x){const long long go=j+base;
      const T*src=srcb+go;T s=0;
      for(int g=0;g<CG;++g)s+=src[size_t(g)*sp];
      T*dst=dstb+go;*dst=sub?*dst-s:s;}}
}
template<class Cfg,int CL> void prepare_carrier_g(){
  static int last_device=-1;int device;CU(cudaGetDevice(&device));
  if(device==last_device)return;
  CU(cudaFuncSetAttribute(carrier_g_kernel<Cfg,CL>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(carrier_g_kernel<Cfg,CL>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last_device=device;
}
template<class Cfg,int CL> void launch_carrier_g(const CarrierGArgs<typename Cfg::Element>&a,int count,cudaStream_t st){
  if(!a.M||!a.N||!count)return;
  prepare_carrier_g<Cfg,CL>();
  if(a.CG>1&&(!a.P||a.sp<(long long)a.M*a.ldo))throw std::runtime_error("carrier_g_gpu_partials");
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(ceildiv(a.M,Cfg::BM),ceildiv(a.N,Cfg::BN),count*CL*a.CG);
  cfg.blockDim=dim3(Cfg::SliceThreads,Cfg::SK,1);cfg.dynamicSmemBytes=Cfg::Smem;cfg.stream=st;
  cudaLaunchAttribute at[1];int na=0;
  if constexpr(CL>1){at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=1;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=CL;na=1;}
  cfg.attrs=at;cfg.numAttrs=na;
  CU(cudaLaunchKernelEx(&cfg,carrier_g_kernel<Cfg,CL>,a));
  if(a.CG>1){carrier_g_combine<typename Cfg::Element><<<std::min(1024,ceildiv(a.M*a.N*count,256)),256,0,st>>>(a.P,a.sp,a.CG,a.O,a.ldo,a.so,a.M,a.N,count,Cfg::Epi==CEPI_SUB);CU(cudaGetLastError());}
}
using CRM=cutlass::layout::RowMajor;using CCM=cutlass::layout::ColumnMajor;
// ---- W = V^T X: C' = W^T (q x h) = X^T V, A' = X^T RowMajor(ldx), B' = V ColumnMajor(ldv).
// Runtime fp32 policy dispatch: f(tag) with tag in {float (IEEE SIMT), MathTF32, MathX3}.
template<class T,class F> decltype(auto) with_carrier_policy(F&&f){
  if constexpr(std::is_same_v<T,double>)return f(double{});
  else{switch(fp32_math()){case Fp32Math::TF32:return f(MathTF32{});case Fp32Math::X3:return f(MathX3{});default:return f(float{});}}
}
template<class P> struct CarrierW;
template<> struct CarrierW<double>{
  using Big=CarrierGCfg<double,CRM,CCM,64,128,32,32,64,16,3,1,CEPI_STORE>;   // q x full h, SK=2
  using Small=CarrierGCfg<double,CRM,CCM,64,64,32,32,32,16,3,2,CEPI_STORE>;  // q x h/2 tiles, SK=2
};
#ifdef TQR_MULTI_FIXED_DRIVER
// BK=32/WK=16, two concurrent peers, scalar load layout and the final fixed-order owner combine are
// unchanged.
using MultiMma8WBig=CarrierGCfg<double,CRM,CCM,64,128,32,32,64,16,3,1,CEPI_STORE,cutlass::gemm::GemmShape<16,8,8>>;
using MultiMma8WSmall=CarrierGCfg<double,CRM,CCM,64,64,32,32,32,16,3,2,CEPI_STORE,cutlass::gemm::GemmShape<16,8,8>>;
static_assert(MultiMma8WBig::Threads==CarrierW<double>::Big::Threads&&
              MultiMma8WSmall::Threads==CarrierW<double>::Small::Threads&&
              MultiMma8WBig::SK==2&&MultiMma8WSmall::SK==2,"unchanged W peer geometry");
// The SM80 64-bit crosswise iterator's XOR/exchange schedule is in K4
// groups. Merely selecting an m16n8k8 instruction does NOT adapt that
// schedule. Keep the original iterator and assemble two successive K4
// fragments in each K8 instruction's register order instead.
template<class Iterator4,int OperandRegisters,bool IsA>
class MultiWPairedIterator {
 Iterator4 iterator_;
public:
 using TensorRef=typename Iterator4::TensorRef;
 using TensorCoord=typename Iterator4::TensorCoord;
 using Fragment4=typename Iterator4::Fragment;
 using Fragment=cutlass::Array<double,2*Fragment4::kElements>;
 static_assert(Fragment4::kElements%OperandRegisters==0,"whole instruction operands");
 CUTLASS_DEVICE MultiWPairedIterator()=default;
 CUTLASS_DEVICE MultiWPairedIterator(TensorRef ref,int lane):iterator_(ref,lane){}
 CUTLASS_DEVICE void add_tile_offset(TensorCoord offset){
  if constexpr(IsA)offset.column()*=2;else offset.row()*=2;
  iterator_.add_tile_offset(offset);
 }
 CUTLASS_DEVICE void set_kgroup_index(int k){iterator_.set_kgroup_index(2*k);}
 CUTLASS_DEVICE MultiWPairedIterator& operator++(){++iterator_;++iterator_;return *this;}
 CUTLASS_DEVICE void load(Fragment& fragment)const{
  Fragment4 lo,hi;auto next=iterator_;next.load(lo);++next;next.load(hi);
  CUTLASS_PRAGMA_UNROLL
  for(int op=0;op<Fragment4::kElements/OperandRegisters;++op){
   CUTLASS_PRAGMA_UNROLL
   for(int reg=0;reg<OperandRegisters;++reg){
    fragment[op*2*OperandRegisters+reg]=lo[op*OperandRegisters+reg];
    fragment[op*2*OperandRegisters+OperandRegisters+reg]=hi[op*OperandRegisters+reg];
   }
  }
 }
};
template<class Warp4,class Warp8> struct MultiWPairedWarp:Warp8 {
 using IteratorA=MultiWPairedIterator<typename Warp4::IteratorA,2,true>;
 using IteratorB=MultiWPairedIterator<typename Warp4::IteratorB,1,false>;
 static_assert(std::is_same_v<typename IteratorA::Fragment,typename Warp8::FragmentA>&&
               std::is_same_v<typename IteratorB::Fragment,typename Warp8::FragmentB>,"K8 fragment types");
 static_assert(std::is_same_v<typename Warp4::LayoutA,typename Warp8::LayoutA>&&
               std::is_same_v<typename Warp4::LayoutB,typename Warp8::LayoutB>,"original shared layouts");
};
template<class Cfg4,class Cfg8> struct MultiWPairedCfg:Cfg4 {
 using Mma4=typename Cfg4::Mma;
 using Warp=MultiWPairedWarp<typename Mma4::Operator,typename Cfg8::Mma::Operator>;
 using Policy4=typename Mma4::Policy;
 using Policy=cutlass::gemm::threadblock::MmaPolicy<Warp,typename Policy4::SmemPaddingA,
   typename Policy4::SmemPaddingB,Policy4::kPartitionsK>;
 using Mma=p2c::GroupMmaMultistage<typename Mma4::Shape,typename Mma4::IteratorA,
   typename Mma4::SmemIteratorA,Mma4::kCacheOpA,typename Mma4::IteratorB,
   typename Mma4::SmemIteratorB,Mma4::kCacheOpB,typename Mma4::ElementC,
   typename Mma4::LayoutC,Policy,Cfg4::Stages>;
 static_assert(sizeof(typename Mma::SharedStorage)==Cfg4::MainSmem,"original shared allocation");
 static_assert(Mma::Base::WarpCount::kK==2&&Mma::Base::kWarpGemmIterations==2,"two K peers, two K8 steps");
};
using MultiPairedWBig=MultiWPairedCfg<CarrierW<double>::Big,MultiMma8WBig>;
using MultiPairedWSmall=MultiWPairedCfg<CarrierW<double>::Small,MultiMma8WSmall>;
#endif
template<> struct CarrierW<float>{   // IEEE SIMT: kept for reference; fp32 ieee W is not native
  using Big=CarrierGCfg<float,CRM,CCM,64,128,16,32,64,8,3,1,CEPI_STORE>;
  using Small=CarrierGCfg<float,CRM,CCM,64,64,16,32,32,8,3,1,CEPI_STORE>;
};
template<> struct CarrierW<MathTF32>{
  // Same law (BK=32, WK=16). 3xTF32 keeps 64x128: the 128x128 tile collapses to 21-23 TF there
  // (split fragments).
  using Big=CarrierGCfg<MathTF32,CRM,CCM,128,128,32,64,64,16,3,1,CEPI_STORE>;
  using Small=CarrierGCfg<MathTF32,CRM,CCM,64,64,32,32,32,16,3,1,CEPI_STORE>;
};
template<> struct CarrierW<MathX3>{
  using Big=CarrierGCfg<MathX3,CRM,CCM,64,128,32,32,64,16,3,1,CEPI_STORE>;
  using Small=CarrierGCfg<MathX3,CRM,CCM,64,64,32,32,32,16,3,1,CEPI_STORE>;
};
constexpr int carrier_w_sk=2;
inline bool carrier_w_supported(int c,int rows){return (c==2||c==4||c==8)&&rows>=(c/carrier_w_sk)*32;}
// The W partition law of the configuration that runs (BK, WK).
template<class T> void carrier_w_law(int&bk,int&wk){
  with_carrier_policy<T>([&](auto tag){using P=decltype(tag);bk=CarrierW<P>::Big::BK;wk=CarrierW<P>::Big::WK;});}
// Output tiling only: the 64 x 128 (full h) tile from ~96 CTAs up, else 64 x 64 for more tiles.
template<class T> bool carrier_w_big(int c,int q,int count){
  return (long long)ceildiv(q,CarrierW<double>::Big::BM)*(c/carrier_w_sk)*count>=96;}
template<class T> int carrier_w_slice_threads(int c,int q,int count){
  return with_carrier_policy<T>([&](auto tag){using P=decltype(tag);
    return carrier_w_big<T>(c,q,count)?CarrierW<P>::Big::SliceThreads:CarrierW<P>::Small::SliceThreads;});}
template<class T> int launch_carrier_w(int c,const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
  T*w,int ldw,long long sw,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int kind=2,int cg=1,T*part=nullptr){
  if(!carrier_w_supported(c,rows))throw std::runtime_error("carrier_w_unsupported_c_or_rows");
  // kind: the product the device reports (2 = apply_W; 6 = compose_G, G = V1^T V2, same W_ONLY shape).
  // cg > 1: GPU-level groups over the rows with HBM partial slices in `part` (q*ldw each).
  CarrierGArgs<T> a{x,ldx,sx,v,ldv,sv,w,ldw,sw,q,h,rows,wit,kind,cg,part,(long long)q*ldw};
  auto go=[&](auto cfg){using Cfg=decltype(cfg);
    if(c==2)launch_carrier_g<Cfg,1>(a,count,st);else if(c==4)launch_carrier_g<Cfg,2>(a,count,st);else launch_carrier_g<Cfg,4>(a,count,st);};
  with_carrier_policy<T>([&](auto tag){using P=decltype(tag);
    if(carrier_w_big<T>(c,q,count))go(typename CarrierW<P>::Big{});else go(typename CarrierW<P>::Small{});});
  return c;
}
// GPU-level group count for a skinny W_ONLY product (compose_G: output <= b x b, K = rows): enough
// CTAs for two waves, each peer keeping >= 256 rows, at most 128 groups (the partial buffer bound).
template<class T> int carrier_w_gpu_split(int c,int q,int h,int rows,int count){
  const int cl=c/carrier_w_sk;int bm=64,bn=64;
  with_carrier_policy<T>([&](auto tag){using P=decltype(tag);
    if(carrier_w_big<T>(c,q,count)){bm=CarrierW<P>::Big::BM;bn=CarrierW<P>::Big::BN;}else{bm=CarrierW<P>::Small::BM;bn=CarrierW<P>::Small::BN;}});
  const long long tiles=(long long)ceildiv(q,bm)*ceildiv(h,bn)*count*cl;
  long long cg=std::max(1LL,2LL*carrier_d_sms()/std::max(1LL,tiles));
  cg=std::min<long long>({cg,128,std::max(1,rows/(256*cl))});
  return int(std::max(1LL,cg));
}
// ---- Z with its own carrier (K = h): Zt = W^T op(T)^T (zt_out, for the native D) or Z = op(T) W.
//  Zt: C' = op(T) W (h x q): A' = op(T) [transpose: T^T RowMajor(ldt); else T ColumnMajor(ldt)],
//      B' = W ColumnMajor(ldw); Out = Zt, ldo = ldz (>= q).
//  Z : C' = W^T op(T)^T (q x h): A' = W^T RowMajor(ldw), B' = op(T)^T [transpose: T ColumnMajor;
//      else T^T RowMajor]; Out = Z, ldo = ldz (>= h).
// Laws: fp64 BK=8c, WK=8; IEEE fp32 c=2 (16,8), c=4 (128,32) (CUTLASS SIMT cores require
// (BK/32) % lane-tile == 0 when an operand is transposed through SMEM); TF32/3xTF32 BK=16c, WK=16.
template<class P,class LA,class LB,int SK> struct CarrierZSel;
template<class LA,class LB,int SK> struct CarrierZSel<double,LA,LB,SK>{using type=CarrierGCfg<double,LA,LB,64,64,8*SK,32,32,8,3,(SK==2?2:1),CEPI_STORE>;};
template<class LA,class LB,int SK> struct CarrierZSel<float,LA,LB,SK>{using type=std::conditional_t<SK==2,
  CarrierGCfg<float,LA,LB,64,64,16,32,32,8,3,1,CEPI_STORE>,CarrierGCfg<float,LA,LB,64,64,128,32,32,32,3,1,CEPI_STORE>>;};
// TF32/3xTF32: a crosswise 32-bit SMEM operand holds at most one 128-byte line per k-tile, so BK<=32
// and the warp slices stay at 2 (BK=32, WK=16); c=4 is 2 cluster CTAs x 2 slices (carrier_z_split).
template<class LA,class LB,int SK> struct CarrierZSel<MathTF32,LA,LB,SK>{using type=CarrierGCfg<MathTF32,LA,LB,64,64,32,32,32,16,3,1,CEPI_STORE>;};
template<class LA,class LB,int SK> struct CarrierZSel<MathX3,LA,LB,SK>{using type=CarrierGCfg<MathX3,LA,LB,64,64,32,32,32,16,3,1,CEPI_STORE>;};
template<class P,class LA,class LB,int SK> using CarrierZCfg=typename CarrierZSel<P,LA,LB,SK>::type;
template<class T> void carrier_z_law(int c,int&bk,int&wk){
  with_carrier_policy<T>([&](auto tag){using P=decltype(tag);
    if(c==2){bk=CarrierZCfg<P,CRM,CCM,2>::BK;wk=CarrierZCfg<P,CRM,CCM,2>::WK;}
    else{bk=CarrierZCfg<P,CRM,CCM,4>::BK;wk=CarrierZCfg<P,CRM,CCM,4>::WK;}});}
// (block slices SK, cluster CTAs CL) with SK*CL = c.
template<class T> void carrier_z_split(int c,int&sk,int&cl){
  sk=c;cl=1;
  if constexpr(std::is_same_v<T,float>){if(fp32_math()!=Fp32Math::IEEE){sk=2;cl=c/2;}}}
// Every piece non-empty: each cluster CTA's range floor(h/CL) >= BK - WK + 1 (residue tile first).
template<class T> bool carrier_z_supported(int c,int h){
  if(c!=2&&c!=4)return false;int bk,wk,sk,cl;carrier_z_law<T>(c,bk,wk);carrier_z_split<T>(c,sk,cl);return h/cl>=bk-wk+1;}
template<class T> int launch_carrier_z(int c,const T*t,int ldt,long long stn,const T*w,int ldw,long long sw,
  T*z,int ldz,long long sz,int h,int q,int count,bool transpose,bool zt_out,cudaStream_t st,Witness*wit){
  if(!carrier_z_supported<T>(c,h))throw std::runtime_error("carrier_z_unsupported_c_or_h");
  with_carrier_policy<T>([&](auto tag){using P=decltype(tag);
    int zsk,zcl;carrier_z_split<T>(c,zsk,zcl);
    auto go=[&](auto sk,auto la,auto lb,const T*A,int lda,long long sa,const T*B,int ldb,long long sb,int M,int N){
      using Cfg=CarrierZCfg<P,decltype(la),decltype(lb),decltype(sk)::value>;
      CarrierGArgs<T> a{A,lda,sa,B,ldb,sb,z,ldz,sz,M,N,h,wit,3};
      if(zcl==2)launch_carrier_g<Cfg,2>(a,count,st);else launch_carrier_g<Cfg,1>(a,count,st);};
    auto dispatch=[&](auto sk){
      if(zt_out){if(transpose)go(sk,CRM{},CCM{},t,ldt,stn,w,ldw,sw,h,q);else go(sk,CCM{},CCM{},t,ldt,stn,w,ldw,sw,h,q);}
      else{if(transpose)go(sk,CRM{},CCM{},w,ldw,sw,t,ldt,stn,q,h);else go(sk,CRM{},CRM{},w,ldw,sw,t,ldt,stn,q,h);}};
    if(c==2)dispatch(std::integral_constant<int,2>{});else dispatch(std::integral_constant<int,4>{});});
  return c;
}
template<class T> int carrier_z_slice_threads(int c){
  return with_carrier_policy<T>([&](auto tag){using P=decltype(tag);return CarrierZCfg<P,CRM,CCM,2>::SliceThreads;});}
}
