#pragma once
// KAMI-informed native fp64 carried D (paper eq:apply X' = X - V Z, "D <- V_e Z with its own carrier", "sum the c
// partials over disjoint K_z", "owner commits X <- X - D exactly once").
//
// KAMI (SC'25, Wang et al.) maps the same hierarchy onto communication-avoiding GEMM: tensor cores compute,
// registers hold the local blocks, shared memory is only the communication medium, and the 1D/2D/3D layouts are
// the cuts of [I], [J], [K]. Here the block carrier is (p_i, p_j, c) = (PI, PJ, C) over PI*PJ*C warps:
// * cut [I] (X rows) into PI and [J] (X columns) into PJ: every warp keeps a WM x WN block of X in registers
//   (32 x 64 fp64 = the register capacity of a 255-register thread with double-buffered fragments; ncu on the
// * cut [K] (the h Householder vectors) into C contiguous balanced ranges K_z (8-element granularity): layer z
//   runs its own cp.async ring over K_z only (own named barrier, no cross-layer lockstep);
// * Replicate: V(I_x, K_z) goes to the PJ warps of row x, Z(K_z, J_y) to the PI warps of column y, through the
// * Combine (additive): one layer parks its partial in shared memory, the other forms D = P_0 + P_1 and the
//   owner commits X - D exactly once (for C = 2 the sum is commutative bit for bit, so the fixed order holds
//   whichever layer adds). The finalizing layer alternates per tile, so one layer's MMAs cover the other's
//   combine and commit.
// Persistence (Pipeline move inside one product): each CTA walks its tiles in raster order and each layer's copy
// ring runs across tile boundaries, so the operands of the next tile stream in during the current tile's last
// k-tiles and its commit. The X block of a tile is prefetched into L2 when the tile starts. Tensor core: DMMA
// m16n8k8 (fp64, sm_90). The MMA row index is relabelled so that MMA rows g and g+8 are physical rows 2g and 2g+1:
// the A fragment pairs and the accumulator pairs are then adjacent in memory (128-bit shared loads/stores, 128-bit
// X accesses). The relabelling only renames which lane computes which element; the contraction sum of every
// element is untouched. Requirements (else the caller keeps the CUTLASS carrier): 16-byte aligned bases, even
// leading dimensions.
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include "common.hpp"
#include "carried_apply.cuh"
#include "carrier_gemm.cuh"
namespace tqr { namespace kami {

__device__ __forceinline__ void dmma8(double (&d)[4],const double (&a)[4],const double (&b)[2]){
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    :"+d"(d[0]),"+d"(d[1]),"+d"(d[2]),"+d"(d[3]):"d"(a[0]),"d"(a[1]),"d"(a[2]),"d"(a[3]),"d"(b[0]),"d"(b[1]));}
__device__ __forceinline__ uint32_t smem_u32(const void*p){return uint32_t(__cvta_generic_to_shared(p));}
// 16-byte async copy; src_bytes < 16 zero-fills the tail (bounds).
__device__ __forceinline__ void cp16(void*dst,const void*src,int src_bytes){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(smem_u32(dst)),"l"(src),"r"(src_bytes):"memory");}
__device__ __forceinline__ void cp_commit(){asm volatile("cp.async.commit_group;\n"::: "memory");}
template<int N> __device__ __forceinline__ void cp_wait(){asm volatile("cp.async.wait_group %0;\n"::"n"(N): "memory");}
__device__ __forceinline__ void bar_sync(int id,int n){asm volatile("bar.sync %0, %1;\n"::"r"(id),"r"(n): "memory");}
__device__ __forceinline__ void bar_arrive(int id,int n){asm volatile("bar.arrive %0, %1;\n"::"r"(id),"r"(n): "memory");}
__device__ __forceinline__ void prefetch_l2(const void*p){asm volatile("prefetch.global.L2 [%0];\n"::"l"(p));}
// Plain C++ shared-memory accesses: the compiler may schedule fragment loads ahead of the MMAs (the barriers
// above carry memory clobbers, so no load crosses a stage hand-off).
__device__ __forceinline__ void lds128(double&x,double&y,const double*p){const double2 v=*reinterpret_cast<const double2*>(p);x=v.x;y=v.y;}
__device__ __forceinline__ double lds64(const double*p){return *p;}
__device__ __forceinline__ void sts128(double*p,double x,double y){*reinterpret_cast<double2*>(p)=make_double2(x,y);}
// Ordered 128-bit shared loads for explicitly software-pipelined mainloops (stay where they are written
// relative to the MMAs and the stage barriers, which are volatile too).
__device__ __forceinline__ void ldsv(double&x,double&y,const double*p){
  asm volatile("ld.shared.v2.f64 {%0,%1}, [%2];\n":"=d"(x),"=d"(y):"r"(smem_u32(p)));}

template<int BM_,int BN_,int BK_,int PI_,int PJ_,int S_>
struct DCfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=PI_,PJ=PJ_,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8;
  static constexpr int LayerThreads=PI*PJ*32,Threads=LayerThreads*C;
  static constexpr int LDA=BM+4;            // As[k][m]: (2t+g) mod 8 distinct 16-B groups for the paired A loads
  static constexpr int LDB=BN+4;            // Bs[k][n]: 8t+2g distinct banks for the scalar B loads
  static constexpr int LDP=BM+2;            // Ps[n][m]: t*LDP+g distinct 16-B groups for accumulator pairs
  static constexpr int StageElems=BK*LDA+BK*LDB;
  static constexpr int LayerElems=S*StageElems;
  static constexpr int ParkElems=BN*LDP;
  static constexpr size_t Smem=size_t(C*LayerElems+ParkElems)*sizeof(double);
  static_assert(WM%16==0&&WN%8==0&&BK%8==0,"tile");
};
using D64=DCfg<64,128,16,2,2,3>;

// Raster: groups of G column tiles, all row tiles of a group before the next (Zt of a group stays in L2).
__device__ __forceinline__ void raster_tile(long long T,int nx,int ny,int G,int&tx,int&ty,int&tb){
  const long long per=(long long)nx*ny;tb=int(T/per);const int L=int(T-(long long)tb*per);
  const int span=G*ny,grp=L/span,in=L%span,gw=min(G,nx-grp*G);tx=grp*G+in%gw;ty=in/gw;}

template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,1)
kami_d_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
              double* X,int ldx,long long sx,int rows,int h,int q,int count,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDP=Cfg::LDP;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double kd_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int nx=(q+BN-1)/BN,ny=(rows+BM-1)/BM;
  const long long ntiles=(long long)nx*ny*count;
  double* const Ls=kd_smem+size_t(layer)*Cfg::LayerElems;
  double* const Ps=kd_smem+size_t(Cfg::C)*Cfg::LayerElems;
  // K_z: contiguous balanced ranges in 8-element units (paper: [K] cut into c balanced parts).
  const int nk8=(h+7)/8,per=(nk8+Cfg::C-1)/Cfg::C;
  const int kb=min(h,layer*per*8),ke=min(h,(layer+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(ltid%VCH)*2,vkk=ltid/VCH,znn=(ltid%ZCH)*2,zkk=ltid/ZCH;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  // Producer state of this layer's ring (advanced incrementally: tile coordinates are recomputed once per
  // tile, pointers advance by one k-tile per issue, the stage index wraps).
  long long iT=blockIdx.x;int ikt=0,istage=0,ik0=kb,ivnb=0,iznb=0;const double*iv=V;const double*iz=Zt;
  auto set_tile=[&](){
    if(iT<ntiles){int tx,ty,tb;raster_tile(iT,nx,ny,raster,tx,ty,tb);const int m0=ty*BM,n0=tx*BN;
      ivnb=min(2,max(0,rows-(m0+vmm)))*8;iznb=min(2,max(0,q-(n0+znn)))*8;
      iv=V+tb*sv+(ivnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;iz=Zt+tb*sz+(iznb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;
      ik0=kb;}};
  set_tile();
  const size_t vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  auto issue=[&](){
    if(iT<ntiles){
      double* St=Ls+istage*Cfg::StageElems;
      if(ik0+BK<=ke){
        #pragma unroll
        for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,iv+i*vstep,ivnb);
        #pragma unroll
        for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,iz+i*zstep,iznb);
      }else{
        #pragma unroll
        for(int i=0;i<VPT;++i){const bool in=ik0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?iv+i*vstep:V,in?ivnb:0);}
        #pragma unroll
        for(int i=0;i<ZPT;++i){const bool in=ik0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?iz+i*zstep:Zt,in?iznb:0);}
      }
      istage=(istage+1==S)?0:istage+1;
      if(++ikt==nkt){ikt=0;iT+=gridDim.x;set_tile();}else{iv+=vtile;iz+=ztile;ik0+=BK;}
    }
    cp_commit();
  };
  int cstage=0;                                 // consumer stage of this layer's ring
  #pragma unroll
  for(int s=0;s<S-1;++s)issue();
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  int it=0;
  for(long long T=blockIdx.x;T<ntiles;T+=gridDim.x,++it){
    int tx,ty,tb;raster_tile(T,nx,ny,raster,tx,ty,tb);
    const int m0=ty*BM,n0=tx*BN;
    double* Xb=X+tb*sx;
    {                                           // X block -> L2 while the contraction runs
      for(int l=tid;l<BN*(BM/16);l+=Cfg::Threads){const int col=n0+l/(BM/16),r=m0+(l%(BM/16))*16;
        if(col<q&&r<rows)prefetch_l2(Xb+size_t(r)+size_t(col)*ldx);}
    }
    double acc[MT][NT][4];
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
    for(int kt=0;kt<nkt;++kt){
      cp_wait<S-2>();
      bar_sync(1+layer,LT);                     // stage pc%S landed for all; stage (pc-1)%S free
      issue();
      const double* As=Ls+cstage*Cfg::StageElems;const double* Bs=As+BK*LDA;cstage=(cstage+1==S)?0:cstage+1;
      #pragma unroll
      for(int kk=0;kk<BK;kk+=8){
        double a[MT][4],b[NT][2];
        #pragma unroll
        for(int i=0;i<MT;++i){const double* p=As+(kk+t)*LDA+am+i*16+2*g;
          lds128(a[i][0],a[i][1],p);lds128(a[i][2],a[i][3],p+4*LDA);}
        #pragma unroll
        for(int j=0;j<NT;++j){const double* p=Bs+(kk+t)*LDB+bn+j*8+g;b[j][0]=lds64(p);b[j][1]=lds64(p+4*LDB);}
        #pragma unroll
        for(int i=0;i<MT;++i)
          #pragma unroll
          for(int j=0;j<NT;++j)dmma8(acc[i][j],a[i],b[j]);
      }
    }
    // Additive combine, split between the layers: the tile's column halves y < PJ/2 are finalized by layer 0,
    // the others by layer 1; each warp (wx,wy) either parks its partial (the half its layer does not finalize)
    // or, after the park barrier, adds the partner's parked partial to its own: D = P_0 + P_1 (commutative bit
    // for bit at c = 2, so the order is the fixed one whichever layer adds). D lands in the park buffer, and
    // after a second barrier all threads commit X - D exactly once, coalesced along X columns.
    __syncthreads();                            // the previous tile's commit has consumed the park buffer
    const bool mine=(wy<Cfg::PJ/2)==(layer==0);
    if(!mine){
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j){double* p=Ps+(bn+j*8+2*t)*LDP+am+i*16+2*g;
          sts128(p,acc[i][j][0],acc[i][j][2]);sts128(p+LDP,acc[i][j][1],acc[i][j][3]);}
    }
    __syncthreads();
    if(mine){
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j){double* p=Ps+(bn+j*8+2*t)*LDP+am+i*16+2*g;double u0,u1,w0,w1;
          lds128(u0,u1,p);lds128(w0,w1,p+LDP);
          sts128(p,u0+acc[i][j][0],u1+acc[i][j][2]);sts128(p+LDP,w0+acc[i][j][1],w1+acc[i][j][3]);}
    }
    __syncthreads();
    {
      constexpr int CH=BM/2,PER=BN*CH/Cfg::Threads,STEP=Cfg::Threads/CH,U=8;
      static_assert((BN*CH)%Cfg::Threads==0&&Cfg::Threads%CH==0&&PER%U==0,"commit plan");
      const int mm=(tid%CH)*2,cc=tid/CH,r=m0+mm;
      const bool fullr=(m0+BM<=rows);
      #pragma unroll 1
      for(int b0=0;b0<PER;b0+=U){
        double2 xv[U];
        #pragma unroll
        for(int u=0;u<U;++u){const int col=n0+cc+(b0+u)*STEP;const double* src=Xb+size_t(r)+size_t(col)*ldx;
          if(fullr&&col<q)xv[u]=__ldcg(reinterpret_cast<const double2*>(src));
          else{xv[u].x=(col<q&&r<rows)?src[0]:0.0;xv[u].y=(col<q&&r+1<rows)?src[1]:0.0;}}
        #pragma unroll
        for(int u=0;u<U;++u){const int cl=cc+(b0+u)*STEP,col=n0+cl;double d0,d1;lds128(d0,d1,Ps+cl*LDP+mm);
          const double x0=xv[u].x-d0,x1=xv[u].y-d1;double* dst=Xb+size_t(r)+size_t(col)*ldx;
          if(fullr&&col<q){asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(x0),"d"(x1):"memory");}
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}}
      }
    }
  }
  cp_wait<0>();
  if(wit&&tid==0&&blockIdx.x==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],(unsigned long long)Cfg::C);
    witness_levels(wit,4,Cfg::C,1,1,1);
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
#ifndef TQR_KAMI_RASTER
#define TQR_KAMI_RASTER 16
#endif

// Non-persistent form (one output tile per CTA; MinB CTAs per SM overlap each other's combine and commit).
// XS=true stages the X block into shared memory with cp.async at the start of the tile; XS=false reads it in
// the commit (prefetched into L2 at the start).
template<int BM_,int BN_,int BK_,int PI_,int PJ_,int S_,bool XS_,int MinB_>
struct NCfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=PI_,PJ=PJ_,C=2,S=S_,MinB=MinB_;static constexpr bool XS=XS_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8;
  static constexpr int LayerThreads=PI*PJ*32,Threads=LayerThreads*C;
  static constexpr int LDA=BM+4,LDB=BN+4,LDX=BM+2;
  static constexpr int StageElems=BK*LDA+BK*LDB,LayerElems=S*StageElems,XElems=XS?BN*LDX:0;
  static constexpr size_t Smem=size_t(C*LayerElems+XElems)*sizeof(double);
  static_assert(WM%16==0&&WN%8==0&&BK%8==0,"tile");
  static_assert(BN*LDX<=LayerElems,"a parked partial fits in its layer's dead stages");
};
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,Cfg::MinB)
kami_dnp_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double kn_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=kn_smem+size_t(layer)*Cfg::LayerElems;
  double* const Xs=kn_smem+size_t(Cfg::C)*Cfg::LayerElems;
  const int nk8=(h+7)/8,per=(nk8+Cfg::C-1)/Cfg::C;
  const int kb=min(h,layer*per*8),ke=min(h,(layer+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(ltid%VCH)*2,vkk=ltid/VCH,znn=(ltid%ZCH)*2,zkk=ltid/ZCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8,znb=min(2,max(0,q-(n0+znn)))*8;
  const double* pV=V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;
  const double* pZ=Zt+(znb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz,vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  auto load_tile=[&](int kt,int stage){
    double* St=Ls+stage*Cfg::StageElems;const int k0=kb+kt*BK;
    const double* v=pV+size_t(kt)*vtile;const double* z=pZ+size_t(kt)*ztile;
    if(k0+BK<=ke){
      #pragma unroll
      for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,v+i*vstep,vnb);
      #pragma unroll
      for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,z+i*zstep,znb);
    }else{
      #pragma unroll
      for(int i=0;i<VPT;++i){const bool in=k0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?v+i*vstep:V,in?vnb:0);}
      #pragma unroll
      for(int i=0;i<ZPT;++i){const bool in=k0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?z+i*zstep:Zt,in?znb:0);}
    }};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  if constexpr(Cfg::XS){
    constexpr int XPT=BN*VCH/Cfg::Threads,XKS=Cfg::Threads/VCH;
    static_assert((BN*VCH)%Cfg::Threads==0&&Cfg::Threads%VCH==0,"x plan");
    const int xmm=(tid%VCH)*2,xnn=tid/VCH,xnb=min(2,max(0,rows-(m0+xmm)))*8;
    const double* px=X+(xnb?size_t(m0+xmm):0)+size_t(n0+xnn)*ldx;
    #pragma unroll 4
    for(int i=0;i<XPT;++i){const bool in=n0+xnn+i*XKS<q;cp16(Xs+(xnn+i*XKS)*LDX+xmm,in?px+size_t(i*XKS)*ldx:X,in?xnb:0);}
  }else{
    for(int l=tid;l<BN*(BM/16);l+=Cfg::Threads){const int col=n0+l/(BM/16),r=m0+(l%(BM/16))*16;
      if(col<q&&r<rows)prefetch_l2(X+size_t(r)+size_t(col)*ldx);}
  }
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  for(int kt=0;kt<nkt;++kt){
    cp_wait<S-2>();
    bar_sync(1+layer,LT);
    {const int nt=kt+S-1;if(nt<nkt)load_tile(nt,nt%S);cp_commit();}
    const double* As=Ls+(kt%S)*Cfg::StageElems;const double* Bs=As+BK*LDA;
    #pragma unroll
    for(int kk=0;kk<BK;kk+=8){
      double a[MT][4],b[NT][2];
      #pragma unroll
      for(int i=0;i<MT;++i){const double* p=As+(kk+t)*LDA+am+i*16+2*g;
        lds128(a[i][0],a[i][1],p);lds128(a[i][2],a[i][3],p+4*LDA);}
      #pragma unroll
      for(int j=0;j<NT;++j){const double* p=Bs+(kk+t)*LDB+bn+j*8+g;b[j][0]=lds64(p);b[j][1]=lds64(p+4*LDB);}
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j)dmma8(acc[i][j],a[i],b[j]);
    }
  }
  cp_wait<0>();
  bar_sync(1+layer,LT);
  if(layer>0){
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j){double* p=Ls+(bn+j*8+2*t)*LDX+am+i*16+2*g;
        sts128(p,acc[i][j][0],acc[i][j][2]);sts128(p+LDX,acc[i][j][1],acc[i][j][3]);}
  }
  __syncthreads();
  if(layer==0){
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    #pragma unroll
    for(int i=0;i<MT;++i){
      double2 xv[NT][2];
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int ml=am+i*16+2*g,nl=bn+j*8+2*t+e,r=m0+ml,col=n0+nl;const double* src=X+size_t(r)+size_t(col)*ldx;
          if constexpr(Cfg::XS)lds128(xv[j][e].x,xv[j][e].y,Xs+nl*LDX+ml);
          else if(full)xv[j][e]=__ldcg(reinterpret_cast<const double2*>(src));
          else{xv[j][e].x=(col<q&&r<rows)?src[0]:0.0;xv[j][e].y=(col<q&&r+1<rows)?src[1]:0.0;}
        }
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int ml=am+i*16+2*g,nl=bn+j*8+2*t+e;
          double d0=acc[i][j][e],d1=acc[i][j][2+e];
          {double p0,p1;lds128(p0,p1,kn_smem+size_t(Cfg::LayerElems)+nl*LDX+ml);d0+=p0;d1+=p1;}
          const int r=m0+ml,col=n0+nl;double* dst=X+size_t(r)+size_t(col)*ldx;
          const double x0=xv[j][e].x-d0,x1=xv[j][e].y-d1;
          if(full)*reinterpret_cast<double2*>(dst)=make_double2(x0,x1);
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
        }
    }
  }
  if(wit&&tid==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],(unsigned long long)Cfg::C);
    witness_levels(wit,4,Cfg::C,1,1,1);
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
// SPLIT-PHASE semi-persistent c = 2 D. ncu at the in-situ fp64 24000^2 shape (rows 24000, q 16384, h 512): cuBLAS's
// 64x128x16 DMMA16x8x8 kernel (4 warps of 32x64, 2 CTAs/SM) keeps the fp64 tensor pipe 88.7% busy; the one-tile c = 2
// kernel above (8 warps of 32x64 in ONE CTA) 71.5%, because both layers reach the combine + commit together and the
// pipe idles for the whole epilogue (cuBLAS's second CTA covers it). Here a CTA walks `tpc` consecutive raster tiles;
// the two contraction layers still own the balanced halves K_z of EVERY tile (Split on [K], c = 2 concurrent peers),
// but the epilogue of tile i belongs to ONE layer, F(i) = i mod 2:
//  * the other layer parks its full-tile partial in the park buffer, arrives on the tile's named barrier and goes
//    straight on to tile i+1 (its cp.async ring runs across tile boundaries);
//  * F(i) syncs on that barrier, forms D = P_own + P_parked (c = 2: commutative bit for bit, so the fixed order
//    holds) at its own fragment positions, and the owner commits X - D exactly once -- while the other layer's MMAs
//    keep the tensor pipe busy.
// Tiles per CTA stay small so the grid still releases SMs to the higher-priority panel quickly (the NP lesson of the
// TF32 GMMA far update).
template<int BM_,int BN_,int BK_,int PI_,int PJ_,int S_>
struct SPCfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=PI_,PJ=PJ_,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8;
  static constexpr int LayerThreads=PI*PJ*32,Threads=LayerThreads*C;
  static constexpr int LDA=BM+4,LDB=BN+4,LDX=BM+2;
  static constexpr int StageElems=BK*LDA+BK*LDB,LayerElems=S*StageElems,ParkElems=BN*LDX;
  static constexpr size_t Smem=size_t(C*LayerElems+ParkElems)*sizeof(double);
  static_assert(WM%16==0&&WN%8==0&&BK%8==0,"tile");
};
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,1)
kami_dsp_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                double* X,int ldx,long long sx,int rows,int h,int q,int count,int raster,int tpc,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double ks_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int nx=(q+BN-1)/BN,ny=(rows+BM-1)/BM;
  const long long ntiles=(long long)nx*ny*count,T0=(long long)blockIdx.x*tpc,T1=min(ntiles,T0+tpc);
  const int ntl=int(T1-T0);
  double* const Ls=ks_smem+size_t(layer)*Cfg::LayerElems;
  double* const Ps=ks_smem+size_t(Cfg::C)*Cfg::LayerElems;
  const int nk8=(h+7)/8,per=(nk8+Cfg::C-1)/Cfg::C;
  const int kb=min(h,layer*per*8),ke=min(h,(layer+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(ltid%VCH)*2,vkk=ltid/VCH,znn=(ltid%ZCH)*2,zkk=ltid/ZCH;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz,vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  // Producer state of this layer's ring (advanced incrementally as in kami_d_kernel: tile coordinates are computed
  // once per tile, pointers advance by one k-tile per load, the stage index wraps).
  int lti=0,lkt=0,lstage=0,lvnb=0,lznb=0,lk0=kb;const double* lv=V;const double* lz=Zt;
  auto set_load_tile=[&](){
    if(lti<ntl){int tx,ty,tb;raster_tile(T0+lti,nx,ny,raster,tx,ty,tb);const int m0=ty*BM,n0=tx*BN;
      lvnb=min(2,max(0,rows-(m0+vmm)))*8;lznb=min(2,max(0,q-(n0+znn)))*8;
      lv=V+tb*sv+(lvnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;lz=Zt+tb*sz+(lznb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;lk0=kb;}};
  set_load_tile();
  auto load=[&](){
    if(lti<ntl){
      double* St=Ls+lstage*Cfg::StageElems;
      if(lk0+BK<=ke){
        #pragma unroll
        for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,lv+i*vstep,lvnb);
        #pragma unroll
        for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,lz+i*zstep,lznb);
      }else{
        #pragma unroll
        for(int i=0;i<VPT;++i){const bool in=lk0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?lv+i*vstep:V,in?lvnb:0);}
        #pragma unroll
        for(int i=0;i<ZPT;++i){const bool in=lk0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?lz+i*zstep:Zt,in?lznb:0);}
      }
      lstage=(lstage+1==S)?0:lstage+1;
      if(++lkt==nkt){lkt=0;++lti;set_load_tile();}else{lv+=vtile;lz+=ztile;lk0+=BK;}
    }
    cp_commit();};
  #pragma unroll
  for(int s=0;s<S-1;++s)load();
  int cstage=0;
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  for(int i=0;i<ntl;++i){
    int tx,ty,tb;raster_tile(T0+i,nx,ny,raster,tx,ty,tb);
    const int m0=ty*BM,n0=tx*BN;double* Xb=X+tb*sx;
    const bool fin=(i&1)==layer;
    if(fin)for(int l=ltid;l<BN*(BM/16);l+=LT){const int col=n0+l/(BM/16),r=m0+(l%(BM/16))*16;   // X block -> L2
      if(col<q&&r<rows)prefetch_l2(Xb+size_t(r)+size_t(col)*ldx);}
    double acc[MT][NT][4];
    #pragma unroll
    for(int a=0;a<MT;++a)
      #pragma unroll
      for(int j=0;j<NT;++j){acc[a][j][0]=acc[a][j][1]=acc[a][j][2]=acc[a][j][3]=0.0;}
    for(int kt=0;kt<nkt;++kt){
      cp_wait<S-2>();
      bar_sync(1+layer,LT);                     // stage cstage landed for the layer; the previous stage is free
      load();
      const double* As=Ls+cstage*Cfg::StageElems;const double* Bs=As+BK*LDA;cstage=(cstage+1==S)?0:cstage+1;
      #pragma unroll
      for(int kk=0;kk<BK;kk+=8){
        double a[MT][4],b[NT][2];
        #pragma unroll
        for(int u=0;u<MT;++u){const double* p=As+(kk+t)*LDA+am+u*16+2*g;
          lds128(a[u][0],a[u][1],p);lds128(a[u][2],a[u][3],p+4*LDA);}
        #pragma unroll
        for(int j=0;j<NT;++j){const double* p=Bs+(kk+t)*LDB+bn+j*8+g;b[j][0]=lds64(p);b[j][1]=lds64(p+4*LDB);}
        #pragma unroll
        for(int u=0;u<MT;++u)
          #pragma unroll
          for(int j=0;j<NT;++j)dmma8(acc[u][j],a[u],b[j]);
      }
    }
    const int bid=3+(i&1);
    if(!fin){
      // Park the full-tile partial (each thread at its own fragment positions), then hand off and move on.
      #pragma unroll
      for(int u=0;u<MT;++u)
        #pragma unroll
        for(int j=0;j<NT;++j){double* p=Ps+(bn+j*8+2*t)*LDX+am+u*16+2*g;
          sts128(p,acc[u][j][0],acc[u][j][2]);sts128(p+LDX,acc[u][j][1],acc[u][j][3]);}
      bar_arrive(bid,2*LT);
    }else{
      bar_sync(bid,2*LT);
      const bool full=(m0+BM<=rows)&&(n0+BN<=q);
      #pragma unroll
      for(int u=0;u<MT;++u){
        double2 xv[NT][2];
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){
            const int ml=am+u*16+2*g,nl=bn+j*8+2*t+e,r=m0+ml,col=n0+nl;const double* src=Xb+size_t(r)+size_t(col)*ldx;
            if(full)xv[j][e]=__ldcg(reinterpret_cast<const double2*>(src));
            else{xv[j][e].x=(col<q&&r<rows)?src[0]:0.0;xv[j][e].y=(col<q&&r+1<rows)?src[1]:0.0;}
          }
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){
            const int ml=am+u*16+2*g,nl=bn+j*8+2*t+e;
            double d0=acc[u][j][e],d1=acc[u][j][2+e];
            {double p0,p1;lds128(p0,p1,Ps+nl*LDX+ml);d0+=p0;d1+=p1;}
            const int r=m0+ml,col=n0+nl;double* dst=Xb+size_t(r)+size_t(col)*ldx;
            const double x0=xv[j][e].x-d0,x1=xv[j][e].y-d1;
            if(full)*reinterpret_cast<double2*>(dst)=make_double2(x0,x1);
            else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
          }
      }
    }
  }
  cp_wait<0>();
  if(wit&&tid==0&&blockIdx.x==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],(unsigned long long)Cfg::C);
    witness_levels(wit,4,Cfg::C,1,1,1);
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
using DSP64x128=SPCfg<64,128,16,2,2,3>;
// FRAGMENT-NATIVE non-persistent c = 2 D. Same carrier as kami_dnp_kernel (two contraction layers own the balanced
// halves K_z, fixed-order combine P_0 + P_1, one owner commit X - D) and the same 64x128 tile with 32x64 warp blocks;
// only operand staging differs:
//  * Z enters k-contiguous (Z: h x q, ld ldz -- the natural layout of Z = T^T W, no transpose pass);
//  * inside every 8-wide k block the MMA's k index t / t+4 is relabelled to physical k 2t / 2t+1 (the contraction is a
//    sum over k, so a consistent relabelling of A and B leaves every product and its accumulation order per k-block
//    unchanged in value set; the DMMA sums 8 terms in hardware either way). A thread's two B values are then adjacent:
//    one 128-bit shared load per n8 tile instead of two 64-bit loads (16 -> 8 B loads per k8 step per warp; the
//  * B is stored [n][k] with 16 doubles per n row and the 16-byte chunk XOR-swizzled by (n & 1) << 2, so each
//    quarter-warp's 128-bit loads hit 8 distinct bank groups without padding.
template<int BM_,int BN_,int BK_,int PI_,int PJ_,int S_>
struct FNCfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=PI_,PJ=PJ_,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8;
  static constexpr int LayerThreads=PI*PJ*32,Threads=LayerThreads*C;
  static constexpr int LDA=BM+4,LDX=BM+2;
  static constexpr int StageElems=BK*LDA+BN*BK,LayerElems=S*StageElems,XElems=BN*LDX;
  static constexpr size_t Smem=size_t(C*LayerElems+XElems)*sizeof(double);
  static_assert(WM%16==0&&WN%8==0&&BK==16,"tile (B rows are 16 doubles = 8 swizzled 16-byte chunks)");
  static_assert(BN*LDX<=LayerElems,"a parked partial fits in its layer's dead stages");
};
__device__ __forceinline__ int fn_chunk(int n,int c){return c^((n&1)<<2);}   // swizzled 16-byte chunk of B row n
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,1)
kami_dnpf_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Z,int ldz,long long sz,
                 double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDX=Cfg::LDX;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double kf_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int batch=blockIdx.z;V+=batch*sv;Z+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=kf_smem+size_t(layer)*Cfg::LayerElems;
  double* const Xs=kf_smem+size_t(Cfg::C)*Cfg::LayerElems;
  const int nk8=(h+7)/8,per=(nk8+Cfg::C-1)/Cfg::C;
  const int kb=min(h,layer*per*8),ke=min(h,(layer+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  // A (V, m contiguous) exactly as kami_dnp_kernel.
  constexpr int VCH=BM/2;static_assert(LT%VCH==0&&(BK*VCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH;
  const int vmm=(ltid%VCH)*2,vkk=ltid/VCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8;
  const double* pV=V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;
  const size_t vstep=size_t(VKS)*ldv,vtile=size_t(BK)*ldv;const int vso=vkk*LDA+vmm;
  // B (Z, k contiguous): thread ltid copies 16-byte chunk (ltid % 8) of rows n = ltid/8 + i*LT/8.
  constexpr int ZROWS=LT/8,ZPT=BN/ZROWS;static_assert(LT%8==0&&BN%ZROWS==0,"z plan");
  const int zc=ltid%8,zr=ltid/8;
  auto load_tile=[&](int kt,int stage){
    double* St=Ls+stage*Cfg::StageElems;double* Sb=St+BK*LDA;const int k0=kb+kt*BK;
    const double* v=pV+size_t(kt)*vtile;
    if(k0+BK<=ke){
      #pragma unroll
      for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,v+i*vstep,vnb);
    }else{
      #pragma unroll
      for(int i=0;i<VPT;++i){const bool in=k0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?v+i*vstep:V,in?vnb:0);}
    }
    const int kz=k0+2*zc;const int kbytes=max(0,min(2,ke-kz))*8;   // this chunk's valid k (0, 1 or 2 doubles)
    #pragma unroll
    for(int i=0;i<ZPT;++i){const int n=zr+i*ZROWS,col=n0+n;const bool in=col<q&&kbytes>0;
      cp16(Sb+n*BK+2*fn_chunk(n,zc),in?Z+size_t(col)*ldz+kz:Z,in?kbytes:0);}};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  {
    constexpr int XPT=BN*VCH/Cfg::Threads,XKS=Cfg::Threads/VCH;
    static_assert((BN*VCH)%Cfg::Threads==0&&Cfg::Threads%VCH==0,"x plan");
    const int xmm=(tid%VCH)*2,xnn=tid/VCH,xnb=min(2,max(0,rows-(m0+xmm)))*8;
    const double* px=X+(xnb?size_t(m0+xmm):0)+size_t(n0+xnn)*ldx;
    #pragma unroll 4
    for(int i=0;i<XPT;++i){const bool in=n0+xnn+i*XKS<q;cp16(Xs+(xnn+i*XKS)*LDX+xmm,in?px+size_t(i*XKS)*ldx:X,in?xnb:0);}
  }
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  for(int kt=0;kt<nkt;++kt){
    cp_wait<S-2>();
    bar_sync(1+layer,LT);
    {const int nt=kt+S-1;if(nt<nkt)load_tile(nt,nt%S);cp_commit();}
    const double* As=Ls+(kt%S)*Cfg::StageElems;const double* Bs=As+BK*LDA;
    #pragma unroll
    for(int kk=0;kk<BK;kk+=8){
      double a[MT][4],b[NT][2];
      #pragma unroll
      for(int i=0;i<MT;++i){const double* p=As+(kk+2*t)*LDA+am+i*16+2*g;   // MMA k t / t+4 -> physical kk+2t / kk+2t+1
        lds128(a[i][0],a[i][1],p);lds128(a[i][2],a[i][3],p+LDA);}
      #pragma unroll
      for(int j=0;j<NT;++j){const int n=bn+j*8+g;lds128(b[j][0],b[j][1],Bs+n*BK+2*fn_chunk(n,kk/2+t));}
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j)dmma8(acc[i][j],a[i],b[j]);
    }
  }
  cp_wait<0>();
  bar_sync(1+layer,LT);
  if(layer>0){
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j){double* p=Ls+(bn+j*8+2*t)*LDX+am+i*16+2*g;
        sts128(p,acc[i][j][0],acc[i][j][2]);sts128(p+LDX,acc[i][j][1],acc[i][j][3]);}
  }
  __syncthreads();
  if(layer==0){
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    #pragma unroll
    for(int i=0;i<MT;++i){
      double2 xv[NT][2];
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){const int ml=am+i*16+2*g,nl=bn+j*8+2*t+e;lds128(xv[j][e].x,xv[j][e].y,Xs+nl*LDX+ml);}
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int ml=am+i*16+2*g,nl=bn+j*8+2*t+e;
          double d0=acc[i][j][e],d1=acc[i][j][2+e];
          {double p0,p1;lds128(p0,p1,kf_smem+size_t(Cfg::LayerElems)+nl*LDX+ml);d0+=p0;d1+=p1;}
          const int r=m0+ml,col=n0+nl;double* dst=X+size_t(r)+size_t(col)*ldx;
          const double x0=xv[j][e].x-d0,x1=xv[j][e].y-d1;
          if(full)*reinterpret_cast<double2*>(dst)=make_double2(x0,x1);
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
        }
    }
  }
  if(wit&&tid==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],(unsigned long long)Cfg::C);
    witness_levels(wit,4,Cfg::C,1,1,1);
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
using DF64x128=FNCfg<64,128,16,2,2,3>;
template<class Cfg> void prepare_dnpf(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dnpf_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dnpf_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
// Z: h x q, ld ldz >= h (k contiguous). Requires 16-byte aligned bases and even leading dimensions (as d_admits).
template<class Cfg=DF64x128> int launch_dnpf(const double*v,int ldv,long long sv,const double*z,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_dnpf<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  kami_dnpf_kernel<Cfg><<<dim3(nx,ny,count),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,z,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit);
  CU(cudaGetLastError());
  return Cfg::C;}
   // 8 warps (2 layers x 2 x 2 of 32x64), 1 CTA/SM, park buffer, X from L2
template<class Cfg> void prepare_dsp(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dsp_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dsp_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
inline int d_tpc(){static const int n=[]{const char*e=std::getenv("TQR_D_TPC");return e?std::max(1,std::atoi(e)):4;}();return n;}
template<class Cfg> int launch_dsp(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0,int tpc=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_dsp<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  const long long ntiles=(long long)nx*ny*count;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  if(tpc<=0)tpc=d_tpc();
  const long long grid=(ntiles+tpc-1)/tpc;
  kami_dsp_kernel<Cfg><<<unsigned(grid),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,count,std::min(raster,nx),tpc,wit);
  CU(cudaGetLastError());
  return Cfg::C;}
using DN64x128=NCfg<64,128,16,2,2,3,true,1>;   // 8 warps, 1 CTA/SM, X staged (the v3 kernel)
using DN64x64=NCfg<64,64,16,2,1,3,false,2>;    // 4 warps, 2 CTAs/SM, X from L2 in the commit
using DN64x64S2=NCfg<64,64,16,2,1,2,true,2>;   // 4 warps, 2 CTAs/SM, 2 stages, X staged
using DN64x128K32=NCfg<64,128,32,2,2,2,false,1>;
using DN64x128nx=NCfg<64,128,16,2,2,3,false,1>;
template<class Cfg> void prepare_dnp(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dnp_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dnp_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
template<class Cfg> int launch_dnp(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_dnp<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  kami_dnp_kernel<Cfg><<<dim3(nx,ny,count),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit);
  CU(cudaGetLastError());
  return Cfg::C;}

// ================================================================================================
// W = V^T X (paper eq:apply), K = the support rows. Block carrier (p_i, p_j, c) = (PI, PJ, 2): two contraction
// layers of PI x PJ warps, each warp a 32 x 64 register block of W; layer z owns the contiguous half K_z of the
// CTA's rows. GPU level: cg groups cut [K] again (balanced, 16-row units) into HBM partial slices summed by
// carrier_g_combine in fixed order (the existing additive combine), else the CTA commits W once.
// Both operands are K-major (V and X are column-major), so the DMMA m16n8k16 k-slot relabelling
// (slot t + 4v of lane t = physical k 4t + v) is free at copy time: every fragment is two 128-bit shared loads.
__device__ __forceinline__ void dmma16(double (&d)[4],const double (&a)[8],const double (&b)[4]){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%0,%1,%2,%3};\n"
    :"+d"(d[0]),"+d"(d[1]),"+d"(d[2]),"+d"(d[3])
    :"d"(a[0]),"d"(a[1]),"d"(a[2]),"d"(a[3]),"d"(a[4]),"d"(a[5]),"d"(a[6]),"d"(a[7]),"d"(b[0]),"d"(b[1]),"d"(b[2]),"d"(b[3]));}
template<int BM_,int BN_,int PI_,int PJ_,int S_>
struct WCfg{
  static constexpr int BM=BM_,BN=BN_,BK=16,PI=PI_,PJ=PJ_,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8;
  static constexpr int LayerThreads=PI*PJ*32,Threads=LayerThreads*C;
  static constexpr int LD=BK+2;              // rows of 16 k + 2: (g + 2t) mod 8 distinct 16-B groups
  static constexpr int StageElems=(BM+BN)*LD,LayerElems=S*StageElems;
  static constexpr int LDP=BM+1;             // parked partial [n][m]
  static constexpr size_t Smem=size_t(C*LayerElems)*sizeof(double);
  static_assert(BN*LDP<=LayerElems,"parked partial fits in the dead stages of layer 1");
  static_assert(WM%16==0&&WN%8==0,"tile");
};
using W64x128=WCfg<64,128,2,2,4>;
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,1)
kami_w_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ X,int ldx,long long sx,
              double* W,int ldw,long long sw,double* P,long long sp,int rows,int h,int q,int cg,int kind,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LD=Cfg::LD,LDP=Cfg::LDP,MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double kw_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int m0=blockIdx.x*BM,n0=blockIdx.y*BN,batch=blockIdx.z/cg,gz=blockIdx.z%cg;
  V+=batch*sv;X+=batch*sx;
  double* Out=cg>1?P+(size_t(batch)*cg+gz)*sp:W+batch*sw;
  // K cut: GPU groups then layers, balanced in 16-row units.
  const int n16=(rows+15)/16,peers=cg*Cfg::C,pid=gz*Cfg::C+layer;
  const int kb=min(rows,int((long long)n16*pid/peers)*16),ke=min(rows,int((long long)n16*(pid+1)/peers)*16);
  const int nkt=(ke-kb+BK-1)/BK;
  double* const Ls=kw_smem+size_t(layer)*Cfg::LayerElems;
  // A rows = W rows (h), B rows = X columns (q)
  constexpr int RPP=LT/8,APT=BM/RPP,BPT=BN/RPP;
  static_assert(LT%8==0&&BM%RPP==0&&BN%RPP==0,"copy plan");
  const int kc=(ltid%8)*2,rr=ltid/8;
  // one base pointer per operand; rows rr + i*RPP of this thread are at base + i*stride
  const double* const pA=V+size_t(m0+rr)*ldv+kb+kc;const double* const pB=X+size_t(n0+rr)*ldx+kb+kc;
  const size_t sA=size_t(RPP)*ldv,sB=size_t(RPP)*ldx;
  const int arows=h-(m0+rr),brows=q-(n0+rr);            // row i valid iff i*RPP < arows (resp. brows)
  auto load_tile=[&](int kt,int stage){
    double* As=Ls+stage*Cfg::StageElems;double* Bs=As+BM*LD;const int k=kb+kt*BK+kc;
    const int kn=k+2<=ke?16:(k<ke?8:0);const size_t ko=size_t(kt)*BK;
    #pragma unroll
    for(int i=0;i<APT;++i){const int nb=(i*RPP<arows)?kn:0;cp16(As+(rr+i*RPP)*LD+kc,nb?pA+i*sA+ko:V,nb);}
    #pragma unroll
    for(int i=0;i<BPT;++i){const int nb=(i*RPP<brows)?kn:0;cp16(Bs+(rr+i*RPP)*LD+kc,nb?pB+i*sB+ko:X,nb);}
  };
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  // Software pipeline (fragments one k-tile ahead): tile kt+1's stage is made visible before tile kt's MMAs,
  // its A fragments load during tile kt's MMAs; B fragments rotate one n-tile ahead.
  static_assert(S>=3,"pipelined mainloop needs S >= 3");
  auto loadA=[&](double (&A)[MT][8],const double* As){
    #pragma unroll
    for(int i=0;i<MT;++i){const double* p=As+(am+i*16+g)*LD+4*t;
      ldsv(A[i][0],A[i][2],p);ldsv(A[i][4],A[i][6],p+2);ldsv(A[i][1],A[i][3],p+8*LD);ldsv(A[i][5],A[i][7],p+8*LD+2);}};
  auto loadB=[&](double (&B)[4],const double* Bs,int j){const double* p=Bs+(bn+j*8+g)*LD+4*t;ldsv(B[0],B[1],p);ldsv(B[2],B[3],p+2);};
  double A0[MT][8],A1[MT][8],Bc[4],Bn[4];
  if(nkt>0){cp_wait<S-2>();bar_sync(1+layer,LT);loadA(A0,Ls);loadB(Bc,Ls+BM*LD,0);}
  int cs=0;
  #pragma unroll 1
  for(int kt=0;kt<nkt;kt+=2){
    // two k-tiles per trip so the A buffers alternate by name (A0 then A1), never by copy
    #pragma unroll
    for(int half=0;half<2;++half){
      const int k=kt+half;if(k>=nkt)break;
      double (&Ac)[MT][8]=half?A1:A0;double (&An)[MT][8]=half?A0:A1;
      cp_wait<S-3>();
      bar_sync(1+layer,LT);
      {const int nt=k+S-1;if(nt<nkt)load_tile(nt,nt%S);cp_commit();}
      const double* Bs=Ls+cs*Cfg::StageElems+BM*LD;
      const int ns=(cs+1==S)?0:cs+1;const double* Nx=Ls+ns*Cfg::StageElems;const bool more=k+1<nkt;
      #pragma unroll
      for(int j=0;j<NT;++j){
        if(j+1<NT)loadB(Bn,Bs,j+1);else if(more)loadB(Bn,Nx+BM*LD,0);
        if(j==NT/2&&more)loadA(An,Nx);
        #pragma unroll
        for(int i=0;i<MT;++i)dmma16(acc[i][j],Ac[i],Bc);
        #pragma unroll
        for(int e=0;e<4;++e)Bc[e]=Bn[e];
      }
      cs=ns;
    }
  }
  cp_wait<0>();
  bar_sync(1+layer,LT);
  // block additive combine (fixed order: layer 0 + layer 1), one commit of W (or of this group's slice)
  double* Ps=kw_smem+size_t(Cfg::LayerElems);
  if(layer==1){
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j){const int m=am+i*16+g,n=bn+j*8+2*t;
        Ps[n*LDP+m]=acc[i][j][0];Ps[(n+1)*LDP+m]=acc[i][j][1];Ps[n*LDP+m+8]=acc[i][j][2];Ps[(n+1)*LDP+m+8]=acc[i][j][3];}
  }
  __syncthreads();
  if(layer==0){
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j){const int m=am+i*16+g,n=bn+j*8+2*t;
        #pragma unroll
        for(int e=0;e<4;++e){const int mm=m+(e>>1)*8,nn=n+(e&1);
          if(m0+mm<h&&n0+nn<q)Out[size_t(m0+mm)+size_t(n0+nn)*ldw]=acc[i][j][e]+Ps[nn*LDP+mm];}}
  }
  if(wit&&tid==0&&gz==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[kind],(unsigned long long)(Cfg::C*cg));
    atomicAdd(&wit->device_block_peers[kind],(unsigned long long)Cfg::C);
    witness_levels(wit,kind,Cfg::C,1,cg,1);
    atomicAdd(&wit->device_membership_reports,1ULL);
    if(batch==0)atomicAdd(&wit->device_combines,1ULL);
  }
}
template<class Cfg> void prepare_w(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_w_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_w_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
inline bool aligned16(const void*p){return (reinterpret_cast<uintptr_t>(p)&15)==0;}
// The native KAMI D admits a product when its operands allow 16-byte copies of every column start and each
// of the C layers owns a nonempty K_z.
template<class Cfg=D64> bool d_admits(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,const double*x,int ldx,long long sx,int h,int count){
  const int nk8=(h+7)/8;
  if(nk8<Cfg::C)return false;
  if(!aligned16(v)||!aligned16(zt)||!aligned16(x))return false;
  if((ldv|ldz|ldx)&1)return false;
  if(count>1&&((sv|sz|sx)&1))return false;
  return true;}
template<class Cfg> void prepare_d(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_d_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_d_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
inline int sm_count(){static int n=0;if(!n){int dev=0;CU(cudaGetDevice(&dev));CU(cudaDeviceGetAttribute(&n,cudaDevAttrMultiProcessorCount,dev));}return n;}
// Returns the executed block-level c. `ctas` caps the persistent grid (0: one CTA per SM).
template<class Cfg=D64> int launch_d(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0,int ctas=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_d<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  const long long ntiles=(long long)nx*ny*count;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  if(ctas<=0)ctas=sm_count();
  const int grid=int(std::min<long long>(ntiles,ctas));
  kami_d_kernel<Cfg><<<grid,Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,count,std::min(raster,nx),wit);
  CU(cudaGetLastError());
  return Cfg::C;}
template<class Cfg=W64x128> bool w_admits(const double*v,int ldv,long long sv,const double*x,int ldx,long long sx,int rows,int count){
  if(!aligned16(v)||!aligned16(x))return false;
  if((ldv|ldx)&1)return false;
  if(count>1&&((sv|sx)&1))return false;
  return rows>=2*16;}
// W = V^T X with block c = 2 and cg GPU groups (partials in `part`, q*ldw words each, combined by
// carrier_g_combine). Returns the executed c (block layers x GPU groups).
template<class Cfg=W64x128> int launch_w(const double*v,int ldv,long long sv,const double*x,int ldx,long long sx,
  double*w,int ldw,long long sw,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int kind=2,int cg=1,double*part=nullptr){
  if(!h||!q||!count)return Cfg::C*cg;
  if(cg>1&&!part)throw std::runtime_error("kami_w_gpu_partials");
  prepare_w<Cfg>();
  const long long sp=(long long)q*ldw;
  kami_w_kernel<Cfg><<<dim3((h+Cfg::BM-1)/Cfg::BM,(q+Cfg::BN-1)/Cfg::BN,count*cg),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,x,ldx,sx,w,ldw,sw,part,sp,rows,h,q,cg,kind,wit);
  CU(cudaGetLastError());
  if(cg>1){carrier_g_combine<double><<<std::min(1024,(h*q*count+255)/256),256,0,st>>>(part,sp,cg,w,ldw,sw,q,h,count,false);CU(cudaGetLastError());}
  return Cfg::C*cg;}

// Machine level: the GPU's thread-block cluster (Hopper). The peers are the 2 CTAs of a cluster, their shared
// memories are joined by distributed shared memory. Carrier (p_i, p_j, c) = (tile rows, tile columns, 2):
//  * Split: each cluster owns one BM x BN output tile; CTA z owns the balanced contraction range K_z (8-element
//    units) of the h Householder vectors. Inside a CTA, 2 x 2 warps each hold a 32 x 64 register block (the same
//    shared memory, 2 CTAs per SM (their prologues/epilogues overlap each other's MMAs, which the 8-warp in-CTA
//    (2,2,2) block carrier at 1 CTA/SM could not).
//  * Replicate: V(I_x, K_z) and Z(K_z, J_y) stream through the CTA's own cp.async ring.
//  * Combine (additive, disjoint K_z): CTA z owns tile columns [z BN/2, (z+1) BN/2). After both mainloops
//    (cluster barrier: the peer's ring is dead), the warps holding the peer's columns push their partial into the
//    peer's ring through DSM; after a second cluster barrier each CTA forms D = P_0 + P_1 (fixed order) for its
//    own half and the owner commits X - D exactly once. The owned half of X is staged by cp.async with the ring.
//  * The mainloop software-pipelines the fragments: B fragments of the next k8 step are loaded right after the
//    MMAs that last read their registers, A fragments are double-buffered; one CTA barrier per k-tile.
template<int BM_,int BN_,int BK_,int S_>
struct CCfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=2,PJ=2,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8,HN=BN/2;
  static constexpr int Threads=PI*PJ*32;
  static constexpr int LDA=BM+4,LDB=BN+4,LDX=BM+2;
  static constexpr int StageElems=BK*LDA+BK*LDB,RingElems=S*StageElems,XElems=HN*LDX,RecvElems=HN*LDX;
  static constexpr size_t Smem=size_t(RingElems+XElems)*sizeof(double);
  static_assert(BK==16,"two k8 steps per k-tile (the A fragment buffers alternate statically)");
  static_assert(RecvElems<=RingElems,"the peer partial lands in the dead ring");
  static_assert(WN==HN,"a warp column block is exactly one owned half");
  static_assert(WM%16==0&&WN%8==0,"tile");
};
__device__ __forceinline__ void cluster_sync_all(){
  asm volatile("barrier.cluster.arrive.release.aligned;\n"::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;\n"::: "memory");}
__device__ __forceinline__ uint32_t cluster_rank(){uint32_t r;asm volatile("mov.u32 %0, %%cluster_ctarank;\n":"=r"(r));return r;}
__device__ __forceinline__ uint32_t map_peer(uint32_t saddr,uint32_t rank){
  uint32_t r;asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n":"=r"(r):"r"(saddr),"r"(rank));return r;}
__device__ __forceinline__ void st_cluster128(uint32_t addr,double x,double y){
  asm volatile("st.shared::cluster.v2.f64 [%0], {%1,%2};\n"::"r"(addr),"d"(x),"d"(y):"memory");}
// One k8 step with lagged fragment loads (one B buffer, no WAR stall on the short scoreboard): on entry b[0..NT-3]
// hold this step's B fragments; the step loads its own tail b[NT-2..NT-1] after the first two MMA columns (the old
// values were last read >= 4 DMMAs earlier), then runs `mid` (the k-tile hand-off: every read of the current stage
// by this thread is issued before it), then, with NEXT, loads the next A fragments into an and the next step's
// b[j-2] after MMA column j (its last reader was issued 4 DMMAs earlier). cB/nA/nB point at row (kk+t) of this
// step's and the next step's stages.
template<class Cfg,bool NEXT,class Mid> __device__ __forceinline__ void dcl_step(double (&acc)[Cfg::MT][Cfg::NT][4],const double (&ac)[Cfg::MT][4],
    double (&an)[Cfg::MT][4],double (&b)[Cfg::NT][2],const double* cB,const double* nA,const double* nB,Mid&& mid){
  constexpr int NT=Cfg::NT,MT=Cfg::MT,LDB=Cfg::LDB,LDA=Cfg::LDA;
  static_assert(NT>=4,"lag of two columns");
  #pragma unroll
  for(int j=0;j<NT;++j){
    #pragma unroll
    for(int i=0;i<MT;++i)dmma8(acc[i][j],ac[i],b[j]);
    if(j==1){
      #pragma unroll
      for(int u=NT-2;u<NT;++u){b[u][0]=lds64(cB+u*8);b[u][1]=lds64(cB+u*8+4*LDB);}
      mid();
      if constexpr(NEXT){
        #pragma unroll
        for(int i=0;i<MT;++i){lds128(an[i][0],an[i][1],nA+i*16);lds128(an[i][2],an[i][3],nA+i*16+4*LDA);}
      }
    }
    if constexpr(NEXT){if(j>=2){b[j-2][0]=lds64(nB+(j-2)*8);b[j-2][1]=lds64(nB+(j-2)*8+4*LDB);}}
  }
}
template<class Cfg> __global__ void __cluster_dims__(2,1,1) __launch_bounds__(Cfg::Threads,2)
kami_dcl_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX,HN=Cfg::HN;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::Threads;
  extern __shared__ __align__(128) double kc_smem[];
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int z=int(cluster_rank());
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;raster_tile((blockIdx.x>>1)+(long long)(gridDim.x>>1)*blockIdx.y,gridDim.x>>1,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=kc_smem;
  double* const Xs=kc_smem+Cfg::RingElems;
  const int nk8=(h+7)/8,per=(nk8+1)/2;
  const int kb=min(h,z*per*8),ke=min(h,(z+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(tid%VCH)*2,vkk=tid/VCH,znn=(tid%ZCH)*2,zkk=tid/ZCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8,znb=min(2,max(0,q-(n0+znn)))*8;
  const double* pV=V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;
  const double* pZ=Zt+(znb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz,vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  auto load_tile=[&](int kt,int stage){
    double* St=Ls+stage*Cfg::StageElems;const int k0=kb+kt*BK;
    const double* v=pV+size_t(kt)*vtile;const double* zp=pZ+size_t(kt)*ztile;
    if(k0+BK<=ke){
      #pragma unroll
      for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,v+i*vstep,vnb);
      #pragma unroll
      for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,zp+i*zstep,znb);
    }else{
      #pragma unroll
      for(int i=0;i<VPT;++i){const bool in=k0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?v+i*vstep:V,in?vnb:0);}
      #pragma unroll
      for(int i=0;i<ZPT;++i){const bool in=k0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?zp+i*zstep:Zt,in?znb:0);}
    }};
  // prologue: tiles 0..S-2, then (after tile 0 landed) tile S-1 together with the owned half of X
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  cp_wait<S-2>();
  __syncthreads();
  if(S-1<nkt)load_tile(S-1,S-1);
  {constexpr int XPT=HN*VCH/LT,XKS=LT/VCH;
   static_assert((HN*VCH)%LT==0,"x plan");
   const int xmm=(tid%VCH)*2,xnn=tid/VCH,xnb=min(2,max(0,rows-(m0+xmm)))*8,nh=n0+z*HN;
   const double* px=X+(xnb?size_t(m0+xmm):0)+size_t(nh+xnn)*ldx;
   #pragma unroll 4
   for(int i=0;i<XPT;++i){const bool in=nh+xnn+i*XKS<q;cp16(Xs+(xnn+i*XKS)*LDX+xmm,in?px+size_t(i*XKS)*ldx:X,in?xnb:0);}}
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  const int aoff=t*LDA+am+2*g,boff=BK*LDA+t*LDB+bn+g;       // row (kk+t) fragment offsets inside a stage
  double a0[MT][4],a1[MT][4],b[NT][2];
  if(nkt>0){
    const double* As=Ls+aoff;const double* Bs=Ls+boff;
    #pragma unroll
    for(int i=0;i<MT;++i){lds128(a0[i][0],a0[i][1],As+i*16);lds128(a0[i][2],a0[i][3],As+i*16+4*LDA);}
    #pragma unroll
    for(int j=0;j<NT-2;++j){b[j][0]=lds64(Bs+j*8);b[j][1]=lds64(Bs+j*8+4*LDB);}
  }
  // rotating stage pointers (no modulo on the critical path): cur = stage of tile kt, nxt = tile kt+1
  double* cur=Ls;double* nxt=Ls+Cfg::StageElems;int cur_stage=0;
  auto none=[]{};
  for(int kt=0;kt<nkt;++kt){
    dcl_step<Cfg,true>(acc,a0,a1,b,cur+boff,cur+aoff+8*LDA,cur+boff+8*LDB,none);   // k8 step 0; next: (kt, kk=8)
    if(kt+1<nkt){
      auto handoff=[&]{cp_wait<S-2>();
        __syncthreads();                                                      // tile kt+1 landed; stage cur is dead
        const int nt=kt+S;if(nt<nkt)load_tile(nt,cur_stage);cp_commit();};
      dcl_step<Cfg,true>(acc,a1,a0,b,cur+boff+8*LDB,nxt+aoff,nxt+boff,handoff);  // k8 step 1; next: (kt+1, kk=0)
      cur=nxt;cur_stage=cur_stage+1==S?0:cur_stage+1;nxt=(nxt+Cfg::StageElems==Ls+Cfg::RingElems)?Ls:nxt+Cfg::StageElems;
    }else dcl_step<Cfg,false>(acc,a1,a0,b,cur+boff+8*LDB,nullptr,nullptr,none);
  }
  cp_wait<0>();
  __syncthreads();
  cluster_sync_all();                                                     // both mainloops done: rings are dead
  if(wy!=z){                                                              // this warp's columns belong to the peer
    const uint32_t rbase=map_peer(smem_u32(Ls),uint32_t(z^1));
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
          st_cluster128(rbase+uint32_t(off)*8u,acc[i][j][e],acc[i][j][2+e]);}
  }
  cluster_sync_all();                                                     // the peer's partial of our half landed
  if(wy==z){
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
          double p0,p1;lds128(p0,p1,Ls+off);                              // peer partial
          double d0,d1;
          if(z==0){d0=acc[i][j][e]+p0;d1=acc[i][j][2+e]+p1;}              // D = P_0 + P_1 (fixed order)
          else{d0=p0+acc[i][j][e];d1=p1+acc[i][j][2+e];}
          double x0,x1;lds128(x0,x1,Xs+off);
          const int r=m0+am+i*16+2*g,col=n0+z*HN+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
          x0-=d0;x1-=d1;
          if(full){asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(x0),"d"(x1):"memory");}
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
        }
  }
  if(wit&&tid==0&&z==0&&blockIdx.x<2&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],1ULL);
    witness_levels(wit,4,1,Cfg::C,1,1);                                   // block c 1, cluster c 2
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
using DC64x128=CCfg<64,128,16,3>;
template<class Cfg> void prepare_dcl(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dcl_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dcl_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
// Returns the executed c (the cluster pair).
template<class Cfg> int launch_dcl(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_dcl<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  kami_dcl_kernel<Cfg><<<dim3(2*nx,ny,count),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit);
  CU(cudaGetLastError());
  return Cfg::C;}

// Same carrier and combine law as kami_dcl_kernel; two changes from its ncu source profile:
// * DMMA m16n8k4 with FULLY double-buffered fragments (24 doubles per buffer): the loads of step s+1 never write a
//   register a DMMA of step s reads, so no LDS waits on the short scoreboard behind a DMMA (the k8 form needs 96
//   fragment registers to do this and spills). The k-tile hand-off sits at the start of the last k4 step, when
//   every read of the current stage has been issued.
// * Combine without cluster barriers on the critical path: each CTA's receive buffer has an mbarrier armed with the
//   byte count of the peer's half; the peer pushes with st.async ... mbarrier::complete_tx (no fence, no wait on
//   the sender side) and the owner waits on its own mbarrier. The only cluster barrier is the split one that
//   publishes the mbarrier initialisation (arrive right after init, wait after the mainloop).
// * X of the owned half is prefetched into L2 at tile start and read in the commit (no smem for it), so the
//   receive buffer is dedicated and never aliases the ring.
__device__ __forceinline__ void dmma4(double (&d)[4],const double (&a)[2],double b){
  asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
    :"+d"(d[0]),"+d"(d[1]),"+d"(d[2]),"+d"(d[3]):"d"(a[0]),"d"(a[1]),"d"(b));}
__device__ __forceinline__ void mbar_init(uint32_t bar,uint32_t count){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"::"r"(bar),"r"(count):"memory");}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar,uint32_t bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"::"r"(bar),"r"(bytes):"memory");}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar,uint32_t parity){
  asm volatile("{\n .reg .pred P;\n W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, [%0], %1;\n @!P bra W;\n}\n"::"r"(bar),"r"(parity):"memory");}
__device__ __forceinline__ void st_async128(uint32_t raddr,double x,double y,uint32_t rbar){
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f64 [%0], {%1,%2}, [%3];\n"::"r"(raddr),"d"(x),"d"(y),"r"(rbar):"memory");}
template<int BM_,int BN_,int BK_,int S_>
struct C4Cfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=2,PJ=2,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8,HN=BN/2,KS=BK/4;
  static constexpr int Threads=PI*PJ*32;
  static constexpr int LDA=BM+4,LDB=BN+4,LDX=BM+2;
  static constexpr int StageElems=BK*LDA+BK*LDB,RingElems=S*StageElems,RecvElems=HN*LDX;
  static constexpr uint32_t RecvBytes=uint32_t(BM)*HN*8u;                 // the peer's half: every element once
  static constexpr size_t Smem=size_t(RingElems+RecvElems)*sizeof(double)+16;
  static_assert(BK%4==0&&KS>=2,"k4 steps");
  static_assert(WN==HN,"a warp column block is exactly one owned half");
};
template<class Cfg> __device__ __forceinline__ void dcl4_load(double (&a)[Cfg::MT][2],double (&b)[Cfg::NT],const double* pa,const double* pb){
  #pragma unroll
  for(int i=0;i<Cfg::MT;++i)lds128(a[i][0],a[i][1],pa+i*16);
  #pragma unroll
  for(int j=0;j<Cfg::NT;++j)b[j]=lds64(pb+j*8);}
template<class Cfg> __device__ __forceinline__ void dcl4_mma(double (&acc)[Cfg::MT][Cfg::NT][4],const double (&a)[Cfg::MT][2],const double (&b)[Cfg::NT]){
  #pragma unroll
  for(int j=0;j<Cfg::NT;++j)
    #pragma unroll
    for(int i=0;i<Cfg::MT;++i)dmma4(acc[i][j],a[i],b[j]);}
template<class Cfg> __global__ void __cluster_dims__(2,1,1) __launch_bounds__(Cfg::Threads,2)
kami_dcl4_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                 double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX,HN=Cfg::HN,KS=Cfg::KS;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::Threads;
  extern __shared__ __align__(128) double kc4_smem[];
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int z=int(cluster_rank());
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;raster_tile((blockIdx.x>>1)+(long long)(gridDim.x>>1)*blockIdx.y,gridDim.x>>1,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=kc4_smem;
  double* const Rs=kc4_smem+Cfg::RingElems;
  const uint32_t mbar=smem_u32(kc4_smem+Cfg::RingElems+Cfg::RecvElems);
  if(tid==0){mbar_init(mbar,1);asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");mbar_expect_tx(mbar,Cfg::RecvBytes);}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n":::"memory");  // split: waited on after the mainloop
  const int nk8=(h+7)/8,per=(nk8+1)/2;
  const int kb=min(h,z*per*8),ke=min(h,(z+1)*per*8);
  const int nkt=(ke-kb+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(tid%VCH)*2,vkk=tid/VCH,znn=(tid%ZCH)*2,zkk=tid/ZCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8,znb=min(2,max(0,q-(n0+znn)))*8;
  const double* pV=V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;
  const double* pZ=Zt+(znb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz,vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  auto load_tile=[&](int kt,int stage){
    double* St=Ls+stage*Cfg::StageElems;const int k0=kb+kt*BK;
    const double* v=pV+size_t(kt)*vtile;const double* zp=pZ+size_t(kt)*ztile;
    if(k0+BK<=ke){
      #pragma unroll
      for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,v+i*vstep,vnb);
      #pragma unroll
      for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,zp+i*zstep,znb);
    }else{
      #pragma unroll
      for(int i=0;i<VPT;++i){const bool in=k0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?v+i*vstep:V,in?vnb:0);}
      #pragma unroll
      for(int i=0;i<ZPT;++i){const bool in=k0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?zp+i*zstep:Zt,in?znb:0);}
    }};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  {const int nh=n0+z*HN;                                                   // owned half of X -> L2
   for(int l=tid;l<HN*(BM/16);l+=LT){const int col=nh+l/(BM/16),r=m0+(l%(BM/16))*16;
     if(col<q&&r<rows)prefetch_l2(X+size_t(r)+size_t(col)*ldx);}}
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  const int aoff=t*LDA+am+2*g,boff=BK*LDA+t*LDB+bn+g;                    // row (kk+t) fragment offsets in a stage
  double fa[2][MT][2],fb[2][NT];
  cp_wait<S-2>();
  __syncthreads();                                                         // tile 0 landed
  if(S-1<nkt)load_tile(S-1,S-1);
  cp_commit();
  if(nkt>0)dcl4_load<Cfg>(fa[0],fb[0],Ls+aoff,Ls+boff);
  int cur_stage=0;
  for(int kt=0;kt<nkt;++kt){
    const double* cur=Ls+cur_stage*Cfg::StageElems;
    #pragma unroll
    for(int s=0;s<KS;++s){
      if(s<KS-1){dcl4_load<Cfg>(fa[(s+1)&1],fb[(s+1)&1],cur+aoff+(s+1)*4*LDA,cur+boff+(s+1)*4*LDB);}
      else if(kt+1<nkt){
        cp_wait<S-2>();
        __syncthreads();                                                   // tile kt+1 landed; every read of stage cur issued
        {const int nt=kt+S;if(nt<nkt)load_tile(nt,cur_stage);cp_commit();}
        const int ns=cur_stage+1==S?0:cur_stage+1;const double* nxt=Ls+ns*Cfg::StageElems;
        dcl4_load<Cfg>(fa[(s+1)&1],fb[(s+1)&1],nxt+aoff,nxt+boff);
      }
      dcl4_mma<Cfg>(acc,fa[s&1],fb[s&1]);
    }
    cur_stage=cur_stage+1==S?0:cur_stage+1;
  }
  asm volatile("barrier.cluster.wait.aligned;\n":::"memory");               // the peer's mbarrier is initialised
  if(wy!=z){                                                               // push the peer's half (no fence, no wait)
    const uint32_t rbase=map_peer(smem_u32(Rs),uint32_t(z^1)),rbar=map_peer(mbar,uint32_t(z^1));
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
          st_async128(rbase+uint32_t(off)*8u,acc[i][j][e],acc[i][j][2+e],rbar);}
  }else{
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    mbar_wait_cluster(mbar,0);                                             // the peer's partial of our half landed
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
          double p0,p1;lds128(p0,p1,Rs+off);
          double d0,d1;
          if(z==0){d0=acc[i][j][e]+p0;d1=acc[i][j][2+e]+p1;}               // D = P_0 + P_1 (fixed order)
          else{d0=p0+acc[i][j][e];d1=p1+acc[i][j][2+e];}
          const int r=m0+am+i*16+2*g,col=n0+z*HN+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
          double x0,x1;
          if(full){const double2 v=__ldcg(reinterpret_cast<const double2*>(dst));x0=v.x;x1=v.y;}
          else{x0=(col<q&&r<rows)?dst[0]:0.0;x1=(col<q&&r+1<rows)?dst[1]:0.0;}
          x0-=d0;x1-=d1;
          if(full){asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(x0),"d"(x1):"memory");}
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
        }
  }
  if(wit&&tid==0&&z==0&&blockIdx.x<2&&blockIdx.y==0){
    atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
    atomicAdd(&wit->device_block_peers[4],1ULL);
    witness_levels(wit,4,1,Cfg::C,1,1);                                    // block c 1, cluster c 2
    atomicAdd(&wit->device_membership_reports,1ULL);
    atomicAdd(&wit->device_physical_commits,1ULL);
  }
}
using DC4x64x128=C4Cfg<64,128,16,3>;
template<class Cfg> void prepare_dcl4(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dcl4_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dcl4_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
template<class Cfg> int launch_dcl4(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return Cfg::C;
  prepare_dcl4<Cfg>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  kami_dcl4_kernel<Cfg><<<dim3(2*nx,ny,count),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit);
  CU(cudaGetLastError());
  return Cfg::C;}

// LAB DIAGNOSTIC ONLY (c = 1, never dispatched by the engine): the kami_dcl mainloop with the full K per CTA and a
// direct commit, to separate mainloop quality from the cost of the c = 2 combine.
template<class Cfg> __global__ void __launch_bounds__(Cfg::Threads,2)
kami_dsolo_kernel(const double* __restrict__ V,int ldv,const double* __restrict__ Zt,int ldz,double* X,int ldx,int rows,int h,int q,int raster){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::Threads;
  extern __shared__ __align__(128) double ks_smem[];
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  int tcol,trow,tb0;raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=ks_smem;
  const int kb=0,ke=h,nkt=(h+BK-1)/BK;
  constexpr int VCH=BM/2,ZCH=BN/2;
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(tid%VCH)*2,vkk=tid/VCH,znn=(tid%ZCH)*2,zkk=tid/ZCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8,znb=min(2,max(0,q-(n0+znn)))*8;
  const double* pV=V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv;
  const double* pZ=Zt+(znb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz;
  const size_t vstep=size_t(VKS)*ldv,zstep=size_t(ZKS)*ldz,vtile=size_t(BK)*ldv,ztile=size_t(BK)*ldz;
  const int vso=vkk*LDA+vmm,zso=BK*LDA+zkk*LDB+znn;
  auto load_tile=[&](int kt,int stage){
    double* St=Ls+stage*Cfg::StageElems;const int k0=kb+kt*BK;
    const double* v=pV+size_t(kt)*vtile;const double* zp=pZ+size_t(kt)*ztile;
    if(k0+BK<=ke){
      #pragma unroll
      for(int i=0;i<VPT;++i)cp16(St+vso+i*VKS*LDA,v+i*vstep,vnb);
      #pragma unroll
      for(int i=0;i<ZPT;++i)cp16(St+zso+i*ZKS*LDB,zp+i*zstep,znb);
    }else{
      #pragma unroll
      for(int i=0;i<VPT;++i){const bool in=k0+vkk+i*VKS<ke;cp16(St+vso+i*VKS*LDA,in?v+i*vstep:V,in?vnb:0);}
      #pragma unroll
      for(int i=0;i<ZPT;++i){const bool in=k0+zkk+i*ZKS<ke;cp16(St+zso+i*ZKS*LDB,in?zp+i*zstep:Zt,in?znb:0);}
    }};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(s,s);cp_commit();}
  for(int l=tid;l<BN*(BM/16);l+=LT){const int col=n0+l/(BM/16),r=m0+(l%(BM/16))*16;if(col<q&&r<rows)prefetch_l2(X+size_t(r)+size_t(col)*ldx);}
  cp_wait<S-2>();
  __syncthreads();
  if(S-1<nkt)load_tile(S-1,S-1);
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  const int aoff=t*LDA+am+2*g,boff=BK*LDA+t*LDB+bn+g;
  double a0[MT][4],a1[MT][4],b[NT][2];
  {const double* As=Ls+aoff;const double* Bs=Ls+boff;
   #pragma unroll
   for(int i=0;i<MT;++i){lds128(a0[i][0],a0[i][1],As+i*16);lds128(a0[i][2],a0[i][3],As+i*16+4*LDA);}
   #pragma unroll
   for(int j=0;j<NT-2;++j){b[j][0]=lds64(Bs+j*8);b[j][1]=lds64(Bs+j*8+4*LDB);}}
  double* cur=Ls;double* nxt=Ls+Cfg::StageElems;int cur_stage=0;
  auto none=[]{};
  for(int kt=0;kt<nkt;++kt){
    dcl_step<Cfg,true>(acc,a0,a1,b,cur+boff,cur+aoff+8*LDA,cur+boff+8*LDB,none);
    if(kt+1<nkt){
      auto handoff=[&]{cp_wait<S-2>();__syncthreads();const int nt=kt+S;if(nt<nkt)load_tile(nt,cur_stage);cp_commit();};
      dcl_step<Cfg,true>(acc,a1,a0,b,cur+boff+8*LDB,nxt+aoff,nxt+boff,handoff);
      cur=nxt;cur_stage=cur_stage+1==S?0:cur_stage+1;nxt=(nxt+Cfg::StageElems==Ls+Cfg::RingElems)?Ls:nxt+Cfg::StageElems;
    }else dcl_step<Cfg,false>(acc,a1,a0,b,cur+boff+8*LDB,nullptr,nullptr,none);
  }
  const bool full=(m0+BM<=rows)&&(n0+BN<=q);
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j)
      #pragma unroll
      for(int e=0;e<2;++e){
        const int r=m0+am+i*16+2*g,col=n0+bn+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
        if(full){const double2 v=__ldcg(reinterpret_cast<const double2*>(dst));
          asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(v.x-acc[i][j][e]),"d"(v.y-acc[i][j][2+e]):"memory");}
        else if(col<q){if(r<rows)dst[0]-=acc[i][j][e];if(r+1<rows)dst[1]-=acc[i][j][2+e];}
      }
}
template<class Cfg> void launch_dsolo(const double*v,int ldv,const double*zt,int ldz,double*x,int ldx,int rows,int h,int q,cudaStream_t st){
  static bool once=false;if(!once){CU(cudaFuncSetAttribute(kami_dsolo_kernel<Cfg>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::RingElems*sizeof(double))));
    CU(cudaFuncSetAttribute(kami_dsolo_kernel<Cfg>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));once=true;}
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  kami_dsolo_kernel<Cfg><<<dim3(nx,ny,1),Cfg::Threads,Cfg::RingElems*sizeof(double),st>>>(v,ldv,zt,ldz,x,ldx,rows,h,q,std::min(TQR_KAMI_RASTER,nx));
  CU(cudaGetLastError());}

// Carrier and combine law of kami_dcl_kernel (cluster peers z = 0,1 over disjoint balanced K_z, st.async + mbarrier
// additive combine into the owner's half, one commit). The mainloop is rebuilt for register economy so that the k8
// fragments are FULLY double-buffered (the ncu profile of the solo c=1 twin showed the previous mainloop, not the
// combine, at 40 TF against cuBLAS's 50 TF on the same 64x128 / 2x2-warp / DMMA16x8x8 shape):
// * shared memory is addressed with 32-bit offsets and explicit ld.shared (volatile: the written order -- the next
//   step's loads interleaved between the current step's DMMAs -- reaches ptxas unchanged, and no load crosses a
//   stage hand-off barrier);
//   |K_z| % 16 == 0, else the caller keeps kami_dnp);
// * the k-tile hand-off (wait tile kt+1, barrier, refill the stage of tile kt) sits between the two k8 steps, after
//   the reads of stage kt were issued.
// SOLO: c = 1, full K per CTA, direct commit, no cluster.
__device__ __forceinline__ void ldsa128(double&x,double&y,uint32_t a){asm volatile("ld.shared.v2.f64 {%0,%1}, [%2];\n":"=d"(x),"=d"(y):"r"(a));}
__device__ __forceinline__ void ldsa64(double&x,uint32_t a){asm volatile("ld.shared.f64 %0, [%1];\n":"=d"(x):"r"(a));}
__device__ __forceinline__ void cp16a(uint32_t dst,const void*src,int src_bytes){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(dst),"l"(src),"r"(src_bytes):"memory");}
template<class Cfg,bool NEXT> __device__ __forceinline__ void dcl2_step(double (&acc)[Cfg::MT][Cfg::NT][4],
    const double (&ac)[Cfg::MT][4],const double (&bc)[Cfg::NT][2],double (&an)[Cfg::MT][4],double (&bn)[Cfg::NT][2],
    uint32_t na,uint32_t nb){
  constexpr int MT=Cfg::MT,NT=Cfg::NT;constexpr uint32_t A4=4u*Cfg::LDA*8u,B4=4u*Cfg::LDB*8u;
  #pragma unroll
  for(int j=0;j<NT;++j){
    #pragma unroll
    for(int i=0;i<MT;++i)dmma8(acc[i][j],ac[i],bc[j]);
    if constexpr(NEXT){
      ldsa64(bn[j][0],nb+uint32_t(j)*64u);ldsa64(bn[j][1],nb+uint32_t(j)*64u+B4);
      if(j<MT){ldsa128(an[j][0],an[j][1],na+uint32_t(j)*128u);ldsa128(an[j][2],an[j][3],na+uint32_t(j)*128u+A4);}
    }
  }
}
template<class Cfg,bool SOLO> __global__ void __launch_bounds__(Cfg::Threads,2)
kami_dcl2_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                 double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX,HN=Cfg::HN;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::Threads;
  constexpr uint32_t StageB=uint32_t(Cfg::StageElems)*8u;
  extern __shared__ __align__(128) double kc2_smem[];
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int z=SOLO?0:int(cluster_rank());
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;
  if constexpr(SOLO)raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  else raster_tile((blockIdx.x>>1)+(long long)(gridDim.x>>1)*blockIdx.y,gridDim.x>>1,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  const uint32_t sbase=smem_u32(kc2_smem);
  const uint32_t rbuf=sbase+uint32_t(Cfg::RingElems)*8u;
  const uint32_t mbar=rbuf+uint32_t(Cfg::RecvElems)*8u;
  if constexpr(!SOLO){
    if(tid==0){mbar_init(mbar,1);asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");mbar_expect_tx(mbar,Cfg::RecvBytes);}
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n":::"memory");
  }
  int kb=0,ke=h;
  if constexpr(!SOLO){const int nk8=(h+7)/8,per=(nk8+1)/2;kb=min(h,z*per*8);ke=min(h,(z+1)*per*8);}
  const int nkt=(ke-kb)/BK;                                                // admission: (ke-kb) % BK == 0
  constexpr int VCH=BM/2,ZCH=BN/2;
  static_assert(LT%VCH==0&&LT%ZCH==0&&(BK*VCH)%LT==0&&(BK*ZCH)%LT==0,"copy plan");
  constexpr int VPT=BK*VCH/LT,VKS=LT/VCH,ZPT=BK*ZCH/LT,ZKS=LT/ZCH;
  const int vmm=(tid%VCH)*2,vkk=tid/VCH,znn=(tid%ZCH)*2,zkk=tid/ZCH;
  const int vnb=min(2,max(0,rows-(m0+vmm)))*8,znb=min(2,max(0,q-(n0+znn)))*8;
  const char* pv=reinterpret_cast<const char*>(V+(vnb?size_t(m0+vmm):0)+size_t(kb+vkk)*ldv);
  const char* pz=reinterpret_cast<const char*>(Zt+(znb?size_t(n0+znn):0)+size_t(kb+zkk)*ldz);
  const uint32_t vstep=uint32_t(VKS)*uint32_t(ldv)*8u,zstep=uint32_t(ZKS)*uint32_t(ldz)*8u;   // < 2^31 (admission)
  const uint32_t vtile=uint32_t(BK)*uint32_t(ldv)*8u,ztile=uint32_t(BK)*uint32_t(ldz)*8u;
  const uint32_t vso=uint32_t(vkk*LDA+vmm)*8u,zso=uint32_t(BK*LDA+zkk*LDB+znn)*8u;
  auto load_tile=[&](uint32_t st){                                         // tile at the cursors -> stage base st
    #pragma unroll
    for(int i=0;i<VPT;++i)cp16a(st+vso+uint32_t(i*VKS*LDA)*8u,pv+size_t(i)*vstep,vnb);
    #pragma unroll
    for(int i=0;i<ZPT;++i)cp16a(st+zso+uint32_t(i*ZKS*LDB)*8u,pz+size_t(i)*zstep,znb);
    pv+=vtile;pz+=ztile;};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(sbase+uint32_t(s)*StageB);cp_commit();}
  {const int nh=n0+(SOLO?0:z*HN),nw=SOLO?BN:HN;                            // X block this CTA commits -> L2
   for(int l=tid;l<nw*(BM/16);l+=LT){const int col=nh+l/(BM/16),r=m0+(l%(BM/16))*16;
     if(col<q&&r<rows)prefetch_l2(X+size_t(r)+size_t(col)*ldx);}}
  cp_wait<S-2>();
  __syncthreads();                                                         // tile 0 landed
  if(S-1<nkt)load_tile(sbase+uint32_t(S-1)*StageB);
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  const uint32_t aoff=uint32_t(t*LDA+am+2*g)*8u,boff=uint32_t(BK*LDA+t*LDB+bn+g)*8u;
  constexpr uint32_t K8A=8u*LDA*8u,K8B=8u*LDB*8u;
  double fa0[MT][4],fb0[NT][2],fa1[MT][4],fb1[NT][2];
  if(nkt>0){
    #pragma unroll
    for(int i=0;i<MT;++i){ldsa128(fa0[i][0],fa0[i][1],sbase+aoff+uint32_t(i)*128u);ldsa128(fa0[i][2],fa0[i][3],sbase+aoff+uint32_t(i)*128u+4u*LDA*8u);}
    #pragma unroll
    for(int j=0;j<NT;++j){ldsa64(fb0[j][0],sbase+boff+uint32_t(j)*64u);ldsa64(fb0[j][1],sbase+boff+uint32_t(j)*64u+4u*LDB*8u);}
  }
  uint32_t cur=sbase,nxt=sbase+StageB;
  const uint32_t ring_end=sbase+uint32_t(S)*StageB;
  for(int kt=0;kt<nkt;++kt){
    dcl2_step<Cfg,true>(acc,fa0,fb0,fa1,fb1,cur+aoff+K8A,cur+boff+K8B);    // k8 step 0; loads (kt, kk=8)
    if(kt+1<nkt){
      cp_wait<S-2>();
      __syncthreads();                                                     // tile kt+1 landed; stage cur fully read
      if(kt+S<nkt)load_tile(cur);
      cp_commit();
      dcl2_step<Cfg,true>(acc,fa1,fb1,fa0,fb0,nxt+aoff,nxt+boff);          // k8 step 1; loads (kt+1, kk=0)
      cur=nxt;nxt=nxt+StageB==ring_end?sbase:nxt+StageB;
    }else dcl2_step<Cfg,false>(acc,fa1,fb1,fa0,fb0,0u,0u);
  }
  if constexpr(SOLO){
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    #pragma unroll
    for(int i=0;i<MT;++i)
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){
          const int r=m0+am+i*16+2*g,col=n0+bn+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
          if(full){const double2 v=__ldcg(reinterpret_cast<const double2*>(dst));
            asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(v.x-acc[i][j][e]),"d"(v.y-acc[i][j][2+e]):"memory");}
          else if(col<q){if(r<rows)dst[0]-=acc[i][j][e];if(r+1<rows)dst[1]-=acc[i][j][2+e];}
        }
    return;
  }else{
    asm volatile("barrier.cluster.wait.aligned;\n":::"memory");              // the peer's mbarrier is initialised
    if(wy!=z){                                                             // push the peer's half (no fence, no wait)
      const uint32_t rb=map_peer(rbuf,uint32_t(z^1)),rbar=map_peer(mbar,uint32_t(z^1));
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
            st_async128(rb+uint32_t(off)*8u,acc[i][j][e],acc[i][j][2+e],rbar);}
    }else{
      const bool full=(m0+BM<=rows)&&(n0+BN<=q);
      mbar_wait_cluster(mbar,0);
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){
            const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
            double p0,p1;ldsa128(p0,p1,rbuf+uint32_t(off)*8u);
            double d0,d1;
            if(z==0){d0=acc[i][j][e]+p0;d1=acc[i][j][2+e]+p1;}             // D = P_0 + P_1 (fixed order)
            else{d0=p0+acc[i][j][e];d1=p1+acc[i][j][2+e];}
            const int r=m0+am+i*16+2*g,col=n0+z*HN+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
            double x0,x1;
            if(full){const double2 v=__ldcg(reinterpret_cast<const double2*>(dst));x0=v.x;x1=v.y;}
            else{x0=(col<q&&r<rows)?dst[0]:0.0;x1=(col<q&&r+1<rows)?dst[1]:0.0;}
            x0-=d0;x1-=d1;
            if(full){asm volatile("st.global.v2.f64 [%0], {%1,%2};\n"::"l"(dst),"d"(x0),"d"(x1):"memory");}
            else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
          }
    }
    if(wit&&tid==0&&z==0&&blockIdx.x<2&&blockIdx.y==0){
      atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
      atomicAdd(&wit->device_block_peers[4],1ULL);
      witness_levels(wit,4,1,Cfg::C,1,1);                                  // block c 1, cluster c 2
      atomicAdd(&wit->device_membership_reports,1ULL);
      atomicAdd(&wit->device_physical_commits,1ULL);
    }
  }
}
using DC2x64x128=C4Cfg<64,128,16,3>;                                       // same smem law as dcl4 (ring + recv + mbarrier)
template<class Cfg> bool dcl2_admits(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,const double*x,int ldx,long long sx,int h,int count){
  const int nk8=(h+7)/8,per=(nk8+1)/2;
  if(nk8<2)return false;
  const int k0=per*8,k1=h-k0;                                              // |K_0|, |K_1|
  if(k0%Cfg::BK||k1%Cfg::BK||k1<=0)return false;
  if(!aligned16(v)||!aligned16(zt)||!aligned16(x))return false;
  if((ldv|ldz|ldx)&1)return false;
  if(count>1&&((sv|sz|sx)&1))return false;
  if((long long)Cfg::BK*ldv*8>=(1LL<<31)||(long long)Cfg::BK*ldz*8>=(1LL<<31))return false;
  return true;}
template<class Cfg,bool SOLO> void prepare_dcl2(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dcl2_kernel<Cfg,SOLO>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dcl2_kernel<Cfg,SOLO>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
template<class Cfg,bool SOLO=false> int launch_dcl2(const double*v,int ldv,long long sv,const double*zt,int ldz,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return SOLO?1:Cfg::C;
  prepare_dcl2<Cfg,SOLO>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(SOLO?nx:2*nx,ny,count);cfg.blockDim=dim3(Cfg::Threads);cfg.dynamicSmemBytes=Cfg::Smem;cfg.stream=st;
  cudaLaunchAttribute at[1];int na=0;
  if constexpr(!SOLO){at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=2;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;na=1;}
  cfg.attrs=at;cfg.numAttrs=na;
  CU(cudaLaunchKernelEx(&cfg,kami_dcl2_kernel<Cfg,SOLO>,v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit));
  return SOLO?1:Cfg::C;}

// Same carrier / combine law as kami_dcl_kernel. Differences:
// * Z enters k-contiguous (Z col-major, h x q; KAMI's "fragment-native" operand: the two B values a lane feeds to
//   DMMA m16n8k8 are adjacent). With the k-slot relabelling (slot t <-> physical k 2t, slot t+4 <-> 2t+1, applied to
//   A and B alike, so every product sum is unchanged) every B fragment is ONE 128-bit shared load and every A
//   fragment two (rows 2g, 2g+1 relabelled as before): 12 LDS per k8 step instead of 20. B stages carry an XOR
//   swizzle (unit ^ 4*(n&1)) instead of padding; A stages pad to LDA = 66 (both conflict-free for the fragment reads).
// * Register budget: A fragments double-buffered (32 regs), B fragments in a 4-slot ring loaded two MMA columns
//   ahead (16 regs; a slot is rewritten 4 DMMAs after its last reader) -- about 176 registers with the accumulator,
//   so ptxas has room not to reuse a register an in-flight DMMA still reads. The k-tile hand-off sits after the last
//   read of the current stage was issued (MMA column NT-2 of the tile's second k8 step).
// SOLO: c = 1, full K per CTA, direct commit, no cluster.
template<int BM_,int BN_,int BK_,int S_>
struct C3Cfg{
  static constexpr int BM=BM_,BN=BN_,BK=BK_,PI=2,PJ=2,C=2,S=S_;
  static constexpr int WM=BM/PI,WN=BN/PJ,MT=WM/16,NT=WN/8,HN=BN/2;
  static constexpr int Threads=PI*PJ*32;
  static constexpr int LDA=BM+2,LDX=BM+2;                                  // LDA = 2 mod 8 (16-B units) for (2t+g)
  static constexpr int AElems=BK*LDA,BElems=BN*BK,StageElems=AElems+BElems,RingElems=S*StageElems,RecvElems=HN*LDX;
  static constexpr uint32_t RecvBytes=uint32_t(BM)*HN*8u;
  static constexpr size_t Smem=size_t(RingElems+RecvElems)*sizeof(double)+16;
  static_assert(BK==16&&NT==8&&MT==2,"the pipeline below is written for 2 k8 steps, 8 MMA columns, 2 MMA rows");
  static_assert(WN==HN,"a warp column block is exactly one owned half");
};
template<class Cfg,bool SOLO> __global__ void __launch_bounds__(Cfg::Threads,2)
kami_dcl3_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Z,int ldzk,long long sz,
                 double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDX=Cfg::LDX,HN=Cfg::HN;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::Threads;
  constexpr uint32_t StageB=uint32_t(Cfg::StageElems)*8u,ABytes=uint32_t(Cfg::AElems)*8u;
  extern __shared__ __align__(128) double kc3_smem[];
  const int tid=threadIdx.x,warp=tid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int z=SOLO?0:int(cluster_rank());
  const int batch=blockIdx.z;V+=batch*sv;Z+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;
  if constexpr(SOLO)raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  else raster_tile((blockIdx.x>>1)+(long long)(gridDim.x>>1)*blockIdx.y,gridDim.x>>1,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  const uint32_t sbase=smem_u32(kc3_smem);
  const uint32_t rbuf=sbase+uint32_t(Cfg::RingElems)*8u;
  const uint32_t mbar=rbuf+uint32_t(Cfg::RecvElems)*8u;
  if constexpr(!SOLO){
    if(tid==0){mbar_init(mbar,1);asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");mbar_expect_tx(mbar,Cfg::RecvBytes);}
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n":::"memory");
  }
  int kb=0,ke=h;
  if constexpr(!SOLO){const int nk8=(h+7)/8,per=(nk8+1)/2;kb=min(h,z*per*8);ke=min(h,(z+1)*per*8);}
  const int nkt=(ke-kb)/BK;                                                // admission: (ke-kb) % BK == 0
  // A = V (col-major, rows contiguous): 32 units of 16 B per k, 4 k-rows per pass, 4 passes.
  const int au=tid&31,ak=tid>>5;
  const int anb=min(2,max(0,rows-(m0+2*au)))*8;
  const char* pa=reinterpret_cast<const char*>(V+(anb?size_t(m0+2*au):0)+size_t(kb+ak)*ldv);
  const uint32_t astep=4u*uint32_t(ldv)*8u,atile=uint32_t(BK)*uint32_t(ldv)*8u;
  const uint32_t aso=uint32_t(ak*LDA+2*au)*8u;
  // B = Z (col-major, k contiguous): 8 units of 16 B per column n, 16 columns per pass, 8 passes; swizzle u^(4(n&1)).
  const int bu=tid&7,bnr=tid>>3;
  const int nb_ok=(n0+bnr<q)?16:0;                                         // column bnr+16i: in range iff n0+bnr+16i<q
  const char* pb=reinterpret_cast<const char*>(Z+size_t(kb+2*bu)+size_t(min(n0+bnr,q-1))*ldzk);
  const uint32_t bstep=16u*uint32_t(ldzk)*8u;
  const uint32_t bso=ABytes+uint32_t(bnr*BK+((bu^((bnr&1)<<2))*2))*8u;       // (bnr+16i)&1 == bnr&1
  auto load_tile=[&](uint32_t st){
    #pragma unroll
    for(int i=0;i<4;++i)cp16a(st+aso+uint32_t(i*4*LDA)*8u,pa+size_t(i)*astep,anb);
    #pragma unroll
    for(int i=0;i<8;++i){const bool in=n0+bnr+16*i<q;cp16a(st+bso+uint32_t(i*16*BK)*8u,in?pb+size_t(i)*bstep:pb,in?nb_ok:0);}
    pa+=atile;pb+=size_t(BK)*8u;};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(sbase+uint32_t(s)*StageB);cp_commit();}
  {const int nh=n0+(SOLO?0:z*HN),nw=SOLO?BN:HN;
   for(int l=tid;l<nw*(BM/16);l+=LT){const int col=nh+l/(BM/16),r=m0+(l%(BM/16))*16;
     if(col<q&&r<rows)prefetch_l2(X+size_t(r)+size_t(col)*ldx);}}
  cp_wait<S-2>();
  __syncthreads();
  if(S-1<nkt)load_tile(sbase+uint32_t(S-1)*StageB);
  cp_commit();
  double acc[MT][NT][4];
  #pragma unroll
  for(int i=0;i<MT;++i)
    #pragma unroll
    for(int j=0;j<NT;++j){acc[i][j][0]=acc[i][j][1]=acc[i][j][2]=acc[i][j][3]=0.0;}
  const int am=wx*Cfg::WM,bn=wy*Cfg::WN;
  // fragment addresses inside a stage: A (kk + 2t, am + 16i + 2g) and (+1 row); B (n = bn + 8j + g, unit kk/2 + t)
  const uint32_t aoff=uint32_t(2*t*LDA+am+2*g)*8u;
  const uint32_t boff=ABytes+uint32_t((bn+g)*BK)*8u;
  const uint32_t bsw=uint32_t((g&1)<<2);                                   // (bn+8j+g)&1 == g&1
  auto bunit=[&](int kk)->uint32_t{return ((uint32_t(kk/2+t))^bsw)*16u;};
  const uint32_t bu0=bunit(0),bu8=bunit(8);
  double fa[2][MT][4],fb[4][2];
  auto ldA=[&](double (&a)[MT][4],uint32_t st,int kk){
    #pragma unroll
    for(int i=0;i<MT;++i){const uint32_t ad=st+aoff+uint32_t(kk*LDA+i*16)*8u;ldsa128(a[i][0],a[i][1],ad);ldsa128(a[i][2],a[i][3],ad+uint32_t(LDA)*8u);}};
  auto ldB=[&](double (&b)[2],uint32_t st,int kk,int j){ldsa128(b[0],b[1],st+boff+uint32_t(j*8*BK)*8u+(kk?bu8:bu0));};
  uint32_t cur=sbase,nxt=sbase+StageB;
  const uint32_t ring_end=sbase+uint32_t(S)*StageB;
  if(nkt>0){ldA(fa[0],cur,0);ldB(fb[0],cur,0,0);ldB(fb[1],cur,0,1);}
  for(int kt=0;kt<nkt;++kt){
    const bool more=kt+1<nkt;
    // ---- k8 step 0 (A in fa[0]); prefetch: B two columns ahead, A of step 1 (same stage) at column 0
    #pragma unroll
    for(int j=0;j<NT;++j){
      #pragma unroll
      for(int i=0;i<MT;++i)dmma8(acc[i][j],fa[0][i],fb[j&3]);
      if(j==0)ldA(fa[1],cur,8);
      if(j+2<NT)ldB(fb[(j+2)&3],cur,0,j+2);else ldB(fb[(j+2)&3],cur,8,j+2-NT);
    }
    // ---- k8 step 1 (A in fa[1]); after column NT-2 the stage is fully read: hand-off, then prefetch from the next stage
    #pragma unroll
    for(int j=0;j<NT;++j){
      #pragma unroll
      for(int i=0;i<MT;++i)dmma8(acc[i][j],fa[1][i],fb[j&3]);
      if(j+2<NT)ldB(fb[(j+2)&3],cur,8,j+2);
      else if(more){
        if(j==NT-2){
          cp_wait<S-2>();
          __syncthreads();                                                 // tile kt+1 landed; stage cur fully read
          if(kt+S<nkt)load_tile(cur);
          cp_commit();
          ldA(fa[0],nxt,0);
        }
        ldB(fb[(j+2)&3],nxt,0,j+2-NT);
      }
    }
    cur=nxt;nxt=nxt+StageB==ring_end?sbase:nxt+StageB;
  }
  if constexpr(SOLO){
    const bool full=(m0+BM<=rows)&&(n0+BN<=q);
    // X in batches of one MMA row (16 pairs): all loads of a batch are in flight together (plain stores, so the
    // compiler may hoist the next batch's loads above this batch's stores)
    #pragma unroll
    for(int i=0;i<MT;++i){
      double2 xv[NT][2];
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){const int r=m0+am+i*16+2*g,col=n0+bn+j*8+2*t+e;const double* src=X+size_t(r)+size_t(col)*ldx;
          if(full)xv[j][e]=__ldcg(reinterpret_cast<const double2*>(src));
          else{xv[j][e].x=(col<q&&r<rows)?src[0]:0.0;xv[j][e].y=(col<q&&r+1<rows)?src[1]:0.0;}}
      #pragma unroll
      for(int j=0;j<NT;++j)
        #pragma unroll
        for(int e=0;e<2;++e){const int r=m0+am+i*16+2*g,col=n0+bn+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
          const double x0=xv[j][e].x-acc[i][j][e],x1=xv[j][e].y-acc[i][j][2+e];
          if(full)*reinterpret_cast<double2*>(dst)=make_double2(x0,x1);
          else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}}
    }
    return;
  }else{
    asm volatile("barrier.cluster.wait.aligned;\n":::"memory");
    if(wy!=z){
      const uint32_t rb=map_peer(rbuf,uint32_t(z^1)),rbar=map_peer(mbar,uint32_t(z^1));
      #pragma unroll
      for(int i=0;i<MT;++i)
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
            st_async128(rb+uint32_t(off)*8u,acc[i][j][e],acc[i][j][2+e],rbar);}
    }else{
      const bool full=(m0+BM<=rows)&&(n0+BN<=q);
      #pragma unroll
      for(int i=0;i<MT;++i){
        double2 xv[NT][2];                                                 // X batch in flight before the wait
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){const int r=m0+am+i*16+2*g,col=n0+z*HN+j*8+2*t+e;const double* src=X+size_t(r)+size_t(col)*ldx;
            if(full)xv[j][e]=__ldcg(reinterpret_cast<const double2*>(src));
            else{xv[j][e].x=(col<q&&r<rows)?src[0]:0.0;xv[j][e].y=(col<q&&r+1<rows)?src[1]:0.0;}}
        if(i==0)mbar_wait_cluster(mbar,0);                                 // the peer's partial of our half landed
        #pragma unroll
        for(int j=0;j<NT;++j)
          #pragma unroll
          for(int e=0;e<2;++e){
            const int off=(j*8+2*t+e)*LDX+am+i*16+2*g;
            double p0,p1;ldsa128(p0,p1,rbuf+uint32_t(off)*8u);
            double d0,d1;
            if(z==0){d0=acc[i][j][e]+p0;d1=acc[i][j][2+e]+p1;}             // D = P_0 + P_1 (fixed order)
            else{d0=p0+acc[i][j][e];d1=p1+acc[i][j][2+e];}
            const int r=m0+am+i*16+2*g,col=n0+z*HN+j*8+2*t+e;double* dst=X+size_t(r)+size_t(col)*ldx;
            const double x0=xv[j][e].x-d0,x1=xv[j][e].y-d1;
            if(full)*reinterpret_cast<double2*>(dst)=make_double2(x0,x1);
            else if(col<q){if(r<rows)dst[0]=x0;if(r+1<rows)dst[1]=x1;}
          }
      }
    }
    if(wit&&tid==0&&z==0&&blockIdx.x<2&&blockIdx.y==0){
      atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
      atomicAdd(&wit->device_block_peers[4],1ULL);
      witness_levels(wit,4,1,Cfg::C,1,1);
      atomicAdd(&wit->device_membership_reports,1ULL);
      atomicAdd(&wit->device_physical_commits,1ULL);
    }
  }
}
using DC3x64x128=C3Cfg<64,128,16,3>;
template<class Cfg,bool SOLO> void prepare_dcl3(){
  static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev==last)return;
  CU(cudaFuncSetAttribute(kami_dcl3_kernel<Cfg,SOLO>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dcl3_kernel<Cfg,SOLO>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));
  last=dev;}
// Z is k-contiguous (col-major h x q, leading dimension ldzk >= h).
template<class Cfg,bool SOLO=false> int launch_dcl3(const double*v,int ldv,long long sv,const double*zk,int ldzk,long long sz,
  double*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return SOLO?1:Cfg::C;
  prepare_dcl3<Cfg,SOLO>();
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(SOLO?nx:2*nx,ny,count);cfg.blockDim=dim3(Cfg::Threads);cfg.dynamicSmemBytes=Cfg::Smem;cfg.stream=st;
  cudaLaunchAttribute at[1];int na=0;
  if constexpr(!SOLO){at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=2;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;na=1;}
  cfg.attrs=at;cfg.numAttrs=na;
  CU(cudaLaunchKernelEx(&cfg,kami_dcl3_kernel<Cfg,SOLO>,v,ldv,sv,zk,ldzk,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit));
  return SOLO?1:Cfg::C;}

// Fragments are double-buffered in registers (asm-ordered loads of step k+1 between the FFMAs of step k); 3-stage
// cp.async ring of BK = 8; the owned half of X is staged by cp.async. SOLO: c = 1, full K, 128x128 commit.
template<int S_,int BK_=8,int XPAD_=4>
struct FCfg{
  static constexpr int BM=128,BN=128,BK=BK_,S=S_,C=2,HN=64,Threads=128;
  static_assert(BK%4==0&&BK>=8,"copy plan: 4 k-rows per pass");
  static constexpr int AElems=BK*BM,BElems=BK*BN,StageElems=AElems+BElems,RingElems=S*StageElems;
  static constexpr int XLD=BM+XPAD_;                                       // X staging [col][row] (pad 4: fewer conflicts)
  static constexpr int XElemsHalf=HN*XLD,XElemsFull=BN*XLD;
  static constexpr int RecvElems=128*128/2;                                // the peer's half: 64 lanes-slots x 128
  static constexpr uint32_t RecvBytes=uint32_t(RecvElems)*4u;
  static constexpr size_t SmemPair=size_t(RingElems+XElemsHalf+RecvElems)*4+16;
  static constexpr size_t SmemSolo=size_t(RingElems+XElemsFull)*4;
};
__device__ __forceinline__ void ldsf4(float (&v)[4],uint32_t a){asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];\n":"=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]):"r"(a));}
__device__ __forceinline__ void st_async_f4(uint32_t raddr,const float (&v)[4],uint32_t rbar){
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];\n"
    ::"r"(raddr),"f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]),"r"(rbar):"memory");}
template<class Cfg,bool SOLO> __global__ void __launch_bounds__(128,2)
simt_dcl_kernel(const float* __restrict__ V,int ldv,long long sv,const float* __restrict__ Zt,int ldz,long long sz,
                float* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,HN=Cfg::HN,XLD=Cfg::XLD;
  constexpr uint32_t StageB=uint32_t(Cfg::StageElems)*4u,ABytes=uint32_t(Cfg::AElems)*4u;
  extern __shared__ __align__(128) float sf_smem[];
  const int tid=threadIdx.x,warp=tid>>5,lane=tid&31,wm=warp>>1,wn=warp&1,r=lane>>3,sc=lane&7;
  const int z=SOLO?0:int(cluster_rank());
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;
  if constexpr(SOLO)raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  else raster_tile((blockIdx.x>>1)+(long long)(gridDim.x>>1)*blockIdx.y,gridDim.x>>1,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  const uint32_t sbase=smem_u32(sf_smem);
  const uint32_t xs=sbase+uint32_t(Cfg::RingElems)*4u;
  const uint32_t rbuf=xs+uint32_t(Cfg::XElemsHalf)*4u,mbar=rbuf+Cfg::RecvBytes;
  if constexpr(!SOLO){
    if(tid==0){mbar_init(mbar,1);asm volatile("fence.mbarrier_init.release.cluster;\n":::"memory");mbar_expect_tx(mbar,Cfg::RecvBytes);}
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n":::"memory");
  }
  int kb=0,ke=h;
  if constexpr(!SOLO){const int nk=(h+2*BK-1)/(2*BK);kb=min(h,z*nk*BK);ke=min(h,(z+1)*nk*BK);}
  const int nkt=(ke-kb)/BK;                                                // admission: h % 16 == 0
  const int cu=tid&31,ck=tid>>5;
  const int avb=min(4,max(0,rows-(m0+4*cu)))*4,bvb=min(4,max(0,q-(n0+4*cu)))*4;
  const char* pa=reinterpret_cast<const char*>(V+(avb?size_t(m0+4*cu):0)+size_t(kb+ck)*ldv);
  const char* pb=reinterpret_cast<const char*>(Zt+(bvb?size_t(n0+4*cu):0)+size_t(kb+ck)*ldz);
  const uint32_t astep=4u*uint32_t(ldv)*4u,bstep=4u*uint32_t(ldz)*4u,atile=uint32_t(BK)*uint32_t(ldv)*4u,btile=uint32_t(BK)*uint32_t(ldz)*4u;
  const uint32_t aso=uint32_t(ck*BM+4*cu)*4u,bso=ABytes+uint32_t(ck*BN+4*cu)*4u;
  auto load_tile=[&](uint32_t st){
    #pragma unroll
    for(int i=0;i<BK/4;++i){cp16a(st+aso+uint32_t(i*4*BM)*4u,pa+size_t(i)*astep,avb);cp16a(st+bso+uint32_t(i*4*BN)*4u,pb+size_t(i)*bstep,bvb);}
    pa+=atile;pb+=btile;};
  #pragma unroll
  for(int s=0;s<S-1;++s){if(s<nkt)load_tile(sbase+uint32_t(s)*StageB);cp_commit();}
  cp_wait<S-2>();
  __syncthreads();
  if(S-1<nkt)load_tile(sbase+uint32_t(S-1)*StageB);
  {// X block this CTA commits -> smem [col][row] (SOLO: 128 cols, pair: the owned 64 cols)
   const int nh=n0+(SOLO?0:z*HN),nw=SOLO?BN:HN;const int xr=4*(tid&31),xvb=min(4,max(0,rows-(m0+xr)))*4;
   const float* px=X+(xvb?size_t(m0+xr):0)+size_t(nh)*ldx;
   for(int c=tid>>5;c<nw;c+=4){const bool in=nh+c<q;cp16a(xs+uint32_t(c*XLD+xr)*4u,in?px+size_t(c)*ldx:X,in?xvb:0);}}
  cp_commit();
  float acc[16][8];
  #pragma unroll
  for(int i=0;i<16;++i)
    #pragma unroll
    for(int j=0;j<8;++j)acc[i][j]=0.f;
  // fragment addresses inside a stage (k-row k): A at k*BM + 64wm + 4r + 16c ; B at k*BN + 64wn + 4sc + 32d
  const uint32_t aoff=uint32_t(64*wm+4*r)*4u,boff=ABytes+uint32_t(64*wn+4*sc)*4u;
  float fa[2][16],fb[2][8];
  auto ldfrag=[&](float (&a)[16],float (&b)[8],uint32_t st,int k){
    const uint32_t ak=st+aoff+uint32_t(k*BM)*4u,bk=st+boff+uint32_t(k*BN)*4u;
    #pragma unroll
    for(int c=0;c<4;++c){float v[4];ldsf4(v,ak+uint32_t(16*c)*4u);a[4*c]=v[0];a[4*c+1]=v[1];a[4*c+2]=v[2];a[4*c+3]=v[3];}
    #pragma unroll
    for(int d=0;d<2;++d){float v[4];ldsf4(v,bk+uint32_t(32*d)*4u);b[4*d]=v[0];b[4*d+1]=v[1];b[4*d+2]=v[2];b[4*d+3]=v[3];}};
  auto fma_step=[&](const float (&a)[16],const float (&b)[8]){
    #pragma unroll
    for(int i=0;i<16;++i)
      #pragma unroll
      for(int j=0;j<8;++j)acc[i][j]=fmaf(a[i],b[j],acc[i][j]);};
  uint32_t cur=sbase,nxt=sbase+StageB;int cur_stage=0;
  const uint32_t ring_end=sbase+uint32_t(S)*StageB;
  if(nkt>0)ldfrag(fa[0],fb[0],cur,0);
  for(int kt=0;kt<nkt;++kt){
    #pragma unroll
    for(int k=0;k<BK;++k){
      if(k<BK-1)ldfrag(fa[(k+1)&1],fb[(k+1)&1],cur,k+1);
      else if(kt+1<nkt){
        cp_wait<S-2>();
        __syncthreads();                                                   // tile kt+1 landed; stage cur fully read
        if(kt+S<nkt)load_tile(cur);
        cp_commit();
        ldfrag(fa[(k+1)&1],fb[(k+1)&1],nxt,0);
      }
      fma_step(fa[k&1],fb[k&1]);
    }
    cur=nxt;cur_stage=cur_stage+1==S?0:cur_stage+1;nxt=nxt+StageB==ring_end?sbase:nxt+StageB;
  }
  cp_wait<0>();
  __syncthreads();                                                         // X staged
  const bool full=(m0+BM<=rows)&&(n0+BN<=q);
  // thread element (i, j): row 64wm + 4r + 16(i/4) + i%4, column 64wn + 4sc + 32(j/4) + j%4
  auto commit=[&](int colbase,auto&& dval){
    #pragma unroll
    for(int j=0;j<8;++j){
      const int cl=64*wn+4*sc+32*(j>>2)+(j&3)-colbase;                     // column inside the staged X block
      const int col=n0+colbase+cl;
      #pragma unroll
      for(int c=0;c<4;++c){
        const int rl=64*wm+4*r+16*c;float x[4];ldsf4(x,xs+uint32_t(cl*XLD+rl)*4u);
        float o[4];
        #pragma unroll
        for(int e=0;e<4;++e)o[e]=x[e]-dval(4*c+e,j);
        float* dst=X+size_t(m0+rl)+size_t(col)*ldx;
        if(full)*reinterpret_cast<float4*>(dst)=make_float4(o[0],o[1],o[2],o[3]);
        else if(col<q){
          #pragma unroll
          for(int e=0;e<4;++e)if(m0+rl+e<rows)dst[e]=o[e];}
      }
    }};
  if constexpr(SOLO){
    commit(0,[&](int i,int j){return acc[i][j];});
  }else{
    asm volatile("barrier.cluster.wait.aligned;\n":::"memory");
    // recv layout: slot (c, j) of lane-slot L = 32 wm + lane: float4 at ((c*8 + j)*64 + L)*4 floats (conflict-free)
    const int L=32*wm+lane;
    if(wn!=z){
      const uint32_t rb=map_peer(rbuf,uint32_t(z^1)),rbar=map_peer(mbar,uint32_t(z^1));
      #pragma unroll
      for(int c=0;c<4;++c)
        #pragma unroll
        for(int j=0;j<8;++j){float v[4]={acc[4*c][j],acc[4*c+1][j],acc[4*c+2][j],acc[4*c+3][j]};
          st_async_f4(rb+uint32_t(((c*8+j)*64+L)*4)*4u,v,rbar);}
    }else{
      mbar_wait_cluster(mbar,0);
      #pragma unroll
      for(int j=0;j<8;++j){                                               // stream: peer float4 + X float4 per (c, j)
        const int cl=64*wn+4*sc+32*(j>>2)+(j&3)-z*HN,col=n0+z*HN+cl;
        #pragma unroll
        for(int c=0;c<4;++c){
          const int rl=64*wm+4*r+16*c;float x[4],pp[4];
          ldsf4(pp,rbuf+uint32_t(((c*8+j)*64+L)*4)*4u);ldsf4(x,xs+uint32_t(cl*XLD+rl)*4u);
          float o[4];
          #pragma unroll
          for(int e=0;e<4;++e){const float d=z==0?acc[4*c+e][j]+pp[e]:pp[e]+acc[4*c+e][j];o[e]=x[e]-d;}   // D = P_0 + P_1
          float* dst=X+size_t(m0+rl)+size_t(col)*ldx;
          if(full)*reinterpret_cast<float4*>(dst)=make_float4(o[0],o[1],o[2],o[3]);
          else if(col<q){
            #pragma unroll
            for(int e=0;e<4;++e)if(m0+rl+e<rows)dst[e]=o[e];}
        }
      }
    }
    if(wit&&tid==0&&z==0&&blockIdx.x<2&&blockIdx.y==0){
      atomicAdd(&wit->device_peer_partials[4],(unsigned long long)Cfg::C);
      atomicAdd(&wit->device_block_peers[4],1ULL);
      witness_levels(wit,4,1,Cfg::C,1,1);
      atomicAdd(&wit->device_membership_reports,1ULL);
      atomicAdd(&wit->device_physical_commits,1ULL);
    }
  }
}
using SF3=FCfg<3>;
using SF4=FCfg<4>;          // 4-stage ring
using SF2K16=FCfg<2,16>;    // BK=16, 2 stages
using SF3K16=FCfg<3,16,0>;  // BK=16, 3 stages, unpadded X staging (2 CTAs/SM fit)
template<class Cfg> bool simt_dcl_admits(const float*v,int ldv,long long sv,const float*zt,int ldz,long long sz,const float*x,int ldx,long long sx,int h,int count){
  if(h<2*Cfg::BK||h%(2*Cfg::BK))return false;
  if((reinterpret_cast<uintptr_t>(v)|reinterpret_cast<uintptr_t>(zt)|reinterpret_cast<uintptr_t>(x))&15)return false;
  if((ldv|ldz|ldx)&3)return false;
  if(count>1&&((sv|sz|sx)&3))return false;
  if((long long)Cfg::BK*ldv*4>=(1LL<<31)||(long long)Cfg::BK*ldz*4>=(1LL<<31))return false;
  return true;}
template<class Cfg,bool SOLO=false> int launch_simt_dcl(const float*v,int ldv,long long sv,const float*zt,int ldz,long long sz,
  float*x,int ldx,long long sx,int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=0){
  if(!rows||!q||!count)return SOLO?1:Cfg::C;
  const size_t smem=SOLO?Cfg::SmemSolo:Cfg::SmemPair;
  static int last=-1;int dev;CU(cudaGetDevice(&dev));
  if(dev!=last){CU(cudaFuncSetAttribute(simt_dcl_kernel<Cfg,SOLO>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(smem)));
    CU(cudaFuncSetAttribute(simt_dcl_kernel<Cfg,SOLO>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));last=dev;}
  const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
  if(raster<=0)raster=TQR_KAMI_RASTER;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(SOLO?nx:2*nx,ny,count);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute at[1];int na=0;
  if constexpr(!SOLO){at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=2;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;na=1;}
  cfg.attrs=at;cfg.numAttrs=na;
  CU(cudaLaunchKernelEx(&cfg,simt_dcl_kernel<Cfg,SOLO>,v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit));
  return SOLO?1:Cfg::C;}
}}  // namespace tqr::kami
