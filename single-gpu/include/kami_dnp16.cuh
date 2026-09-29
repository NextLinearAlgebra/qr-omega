#pragma once
// FP64 carried D with 128x64 tiles and m16n8k16 DMMA: two warp layers own disjoint contraction halves
// and combine them in shared memory (a KAMI-style (2,2,2) block carrier, see kami_carrier.cuh).
// Peers own disjoint contraction halves.
#include "kami_carrier.cuh"
namespace tqr { namespace kami {
template<class Cfg,bool StreamB=false> __global__ void __launch_bounds__(Cfg::Threads,Cfg::MinB)
kami_dnp16_kernel(const double* __restrict__ V,int ldv,long long sv,const double* __restrict__ Zt,int ldz,long long sz,
                double* X,int ldx,long long sx,int rows,int h,int q,int raster,Witness* wit){
  constexpr int BM=Cfg::BM,BN=Cfg::BN,BK=Cfg::BK,S=Cfg::S,LDA=Cfg::LDA,LDB=Cfg::LDB,LDX=Cfg::LDX;
  constexpr int MT=Cfg::MT,NT=Cfg::NT,LT=Cfg::LayerThreads;
  extern __shared__ __align__(128) double kn16_smem[];
  const int tid=threadIdx.x,layer=tid/LT,ltid=tid%LT,warp=ltid/32,lane=tid%32,g=lane>>2,t=lane&3;
  const int wx=warp/Cfg::PJ,wy=warp%Cfg::PJ;
  const int batch=blockIdx.z;V+=batch*sv;Zt+=batch*sz;X+=batch*sx;
  int tcol,trow,tb0;raster_tile(blockIdx.x+(long long)gridDim.x*blockIdx.y,gridDim.x,gridDim.y,raster,tcol,trow,tb0);
  const int n0=tcol*BN,m0=trow*BM;
  double* const Ls=kn16_smem+size_t(layer)*Cfg::LayerElems;
  double* const Xs=kn16_smem+size_t(Cfg::C)*Cfg::LayerElems;
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
    for(int kk=0;kk<BK;kk+=16){
      double a[MT][8];
      #pragma unroll
      for(int i=0;i<MT;++i){const double* p=As+(kk+t)*LDA+am+i*16+2*g;
        #pragma unroll
        for(int u=0;u<4;++u)lds128(a[i][2*u],a[i][2*u+1],p+4*u*LDA);}
      if constexpr(StreamB){
        #pragma unroll
        for(int j=0;j<NT;++j){double bj[4];const double* p=Bs+(kk+t)*LDB+bn+j*8+g;
          #pragma unroll
          for(int u=0;u<4;++u)bj[u]=lds64(p+4*u*LDB);
          #pragma unroll
          for(int i=0;i<MT;++i)dmma16(acc[i][j],a[i],bj);}
      }else{
        double bv[NT][4];
        #pragma unroll
        for(int j=0;j<NT;++j){const double* p=Bs+(kk+t)*LDB+bn+j*8+g;
          #pragma unroll
          for(int u=0;u<4;++u)bv[j][u]=lds64(p+4*u*LDB);}
        #pragma unroll
        for(int i=0;i<MT;++i)
          #pragma unroll
          for(int j=0;j<NT;++j)dmma16(acc[i][j],a[i],bv[j]);
      }
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
          {double p0,p1;lds128(p0,p1,kn16_smem+size_t(Cfg::LayerElems)+nl*LDX+ml);d0+=p0;d1+=p1;}
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

using DN128x64K16=NCfg<128,64,16,2,2,3,true,1>;
template<class Cfg,bool StreamB=false> int launch_dnp16(const double*v,int ldv,long long sv,
 const double*zt,int ldz,long long sz,double*x,int ldx,long long sx,
 int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int raster=16){
 static_assert(Cfg::C==2&&Cfg::BK%16==0,"two contraction peers with k16 tiles");
 if(!rows||!h||!q||!count)return 0;
 if(h%32)throw std::runtime_error("dnp16_requires_two_k16_aligned_halves");
 static int last=-1;int dev;CU(cudaGetDevice(&dev));if(dev!=last){
  CU(cudaFuncSetAttribute(kami_dnp16_kernel<Cfg,StreamB>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(Cfg::Smem)));
  CU(cudaFuncSetAttribute(kami_dnp16_kernel<Cfg,StreamB>,cudaFuncAttributePreferredSharedMemoryCarveout,int(cudaSharedmemCarveoutMaxShared)));last=dev;}
 const int nx=(q+Cfg::BN-1)/Cfg::BN,ny=(rows+Cfg::BM-1)/Cfg::BM;
 kami_dnp16_kernel<Cfg,StreamB><<<dim3(nx,ny,count),Cfg::Threads,Cfg::Smem,st>>>(v,ldv,sv,zt,ldz,sz,x,ldx,sx,rows,h,q,std::min(raster,nx),wit);
 CU(cudaGetLastError());return Cfg::C;
}
}}
