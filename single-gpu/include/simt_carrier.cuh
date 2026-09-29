#pragma once
// IEEE fp32 SIMT carrier for the two large products of Eq apply, W(h,q) = V^T X (both operands K-contiguous:
// register-staged float4 loads, transposed into SMEM) X(rows,q) -= V Z (V M-contiguous, Z given as Zt = Z^T,
// N-contiguous: float4 loads straight into SMEM) at cuBLAS SGEMM speed.
//
// Carriers, the same levels and laws as carrier_g_kernel / carrier_d_kernel:
// * BLOCK level: SK = 2 warp slices of one CTA (threadIdx.y is the K peer) share one double-buffered SMEM
//   pipeline; each 16-deep k-tile is cut 8 / 8 (BK = 16, WK = 8: slice w owns positions
//   floor(((k - min K_z) mod 16)/8) = w; the ragged residue tile, if any, is the LAST one).
// * CLUSTER / GPU levels (W only): peer pid = g CL + z of P = CL CG owns the 16-aligned balanced range
//   K_z = [16 floor(floor(K/16) pid/P), 16 floor(floor(K/16)(pid+1)/P)), the last peer ending at K
//   (aligned so every full k-tile is a float4 load; balanced to within one k-tile).
// * GPU level (W only): CG groups write HBM partial slices summed by carrier_g_combine (fixed order).
// Combine: every slice parks its register partial in SMEM; after the cluster barrier CTA z sums its row slice of the
// tile over pieces (z', w) in the fixed order z' = 0..CL-1, w = 0..SK-1 through DSM and commits it ONCE. No atomics
// on data, bitwise deterministic. Executed product c = CL * SK (* CG at the GPU level). Device evidence: identical
// to the kernels it replaces (peer partials, block peers, levels, membership, commits).
#include "carrier_gemm.cuh"
namespace tqr {
namespace simt {
constexpr int BM=128,BN=128,BK=16,WK=8,SK=2,SLICE=256,THREADS=SLICE*SK;
constexpr int LDS=BM+4;                         // SMEM row stride of a k-row (floats): 16-byte aligned rows
// Each slice runs its OWN pipeline over its own 8-deep k-blocks (block j of K_z goes to slice j mod 2, i.e.
// position floor(((k - min K_z) mod 16)/8) = w -- the same law), with its own SMEM stages and named barrier,
// so the two K peers behave like two independent CTAs on the SM and meet only at the combine.
constexpr int TILE_FLOATS=WK*LDS;               // one operand tile (8 k-rows) of one stage of one slice
constexpr int SLICE_FLOATS=2*2*TILE_FLOATS;     // 2 stages x (A, B) per slice
constexpr int PARK_LD=BM+4;                     // parked partial: column-major [n][m]
constexpr size_t PARK_FLOATS=size_t(BN)*PARK_LD;
constexpr size_t SMEM_MAIN=size_t(SK)*SLICE_FLOATS*sizeof(float);   // per slice: 2 stages x (A, B)
constexpr size_t SMEM_PARK=size_t(SK)*PARK_FLOATS*sizeof(float);
constexpr size_t SMEM=SMEM_MAIN>SMEM_PARK?SMEM_MAIN:SMEM_PARK;
enum Epi{STORE=0,SUB=1};
// Operand tile loader (one slice, 128 threads). KC: the operand is K-contiguous (element (i,k) at p[k + i ld]);
// else I-contiguous (element (i,k) at p[i + k ld]). The tile holds I in [i0, i0+128) x K in [k0, k0+8) (clipped to
// [0,I) x [k0,ke)), written to SMEM as s[k*LDS + i]. Each thread (256 per slice) moves 1 float4 per tile.
template<bool KC> struct Loader{
  float4 r;
  const float* ptr;long long step;int i_ok;int kbase;   // this thread's element at the slice's first block
  // Set up once: the thread's row (KC) or k-row (!KC) and its base address at k = k0 (the slice's first block).
  __device__ __forceinline__ void init(const float* __restrict__ p,long long ld,int I,int i0,int k0,int tid){
    if constexpr(KC){const int i=tid>>1,kq=(tid&1)*4,gi=i0+i;i_ok=gi<I;kbase=k0+kq;
      ptr=p+(long long)(i_ok?gi:0)*ld+kbase;step=2*WK;}           // next block of this slice: +16 k
    else{const int k=tid>>5,iq=(tid&31)*4,gi=i0+iq;i_ok=gi+3<I?4:(gi<I?I-gi:0);kbase=k0+k;
      ptr=p+(long long)kbase*ld+(i_ok?gi:0);step=2*WK*ld;}
  }
  // Block t of this slice: k offset 16 t from kbase. `full`: the whole 8-deep block lies below ke.
  __device__ __forceinline__ void load(int t,int ke,const float* __restrict__ p0,long long ld,int i0,int tid){
    const float* q=ptr+t*step;
    if constexpr(KC){const int gk=kbase+t*2*WK;
      if(i_ok&&gk+3<ke)r=__ldg(reinterpret_cast<const float4*>(q));
      else{float v[4];
        #pragma unroll
        for(int e=0;e<4;++e)v[e]=(i_ok&&gk+e<ke)?__ldg(q+e):0.f;
        r=make_float4(v[0],v[1],v[2],v[3]);}}
    else{const int gk=kbase+t*2*WK;
      if(i_ok==4&&gk<ke)r=__ldg(reinterpret_cast<const float4*>(q));
      else{float v[4];
        #pragma unroll
        for(int e=0;e<4;++e)v[e]=(gk<ke&&e<i_ok)?__ldg(q+e):0.f;
        r=make_float4(v[0],v[1],v[2],v[3]);}}
  }
  __device__ __forceinline__ void store(float* s,int tid)const{
    if constexpr(KC){const int i=tid>>1,kq=(tid&1)*4;
      s[(kq+0)*LDS+i]=r.x;s[(kq+1)*LDS+i]=r.y;s[(kq+2)*LDS+i]=r.z;s[(kq+3)*LDS+i]=r.w;}
    else{const int k=tid>>5,iq=(tid&31)*4;*reinterpret_cast<float4*>(s+k*LDS+iq)=r;}
  }
};
struct Args{
  const float*A;long long lda,sa;       // A operand (M side): W: V (K-contiguous); D: V (M-contiguous)
  const float*B;long long ldb,sb;       // B operand (N side): W: X (K-contiguous); D: Zt (N-contiguous)
  float*O;long long ldo,so;             // output, column-major (element (m,n) at O[m + n ldo])
  int M,N,K;Witness*wit;int kind;int CG;float*P;long long sp;int raster;
};
// One kernel for both products. AKC/BKC: operand K-contiguity. EPI: STORE (W) or SUB (D: X -= sum).
template<bool AKC,bool BKC,int EPI,int CL>
__global__ void __launch_bounds__(THREADS,1) simt_carrier_kernel(const __grid_constant__ Args a){
  extern __shared__ __align__(128) float sm[];
  const int lt=threadIdx.x,sl=threadIdx.y,tid=lt+SLICE*sl;
  int z=0;if constexpr(CL>1)z=int(cooperative_groups::this_cluster().block_rank());
  int tm_,tn_;
  if(a.raster>0)carrier_raster_x(a.raster,tm_,tn_);else{tm_=blockIdx.x;tn_=blockIdx.y;}
  const int zz=blockIdx.z/CL,batch=zz/a.CG,gz=zz%a.CG,peers=CL*a.CG,pid=gz*CL+z;
  const int m0=tm_*BM,n0=tn_*BN;
  const float*Ap=a.A+batch*a.sa;const float*Bp=a.B+batch*a.sb;
  // K_z of peer pid (P = CL*CG peers): 16-aligned balanced ranges, the last peer ending at K:
  //   K_z = [16 floor(floor(K/16) pid/P), 16 floor(floor(K/16)(pid+1)/P)), pid = P-1 -> [.., K).
  // k-tiles of 16 run from min K_z; slice w owns positions floor(((k - min K_z) mod 16)/8) = w (residue tile last).
  const int K16=a.K/BK;
  const int kb=BK*int((long long)K16*pid/peers),ke=(pid==peers-1)?a.K:BK*int((long long)K16*(pid+1)/peers);
  Loader<AKC> la;Loader<BKC> lb;
  float acc[8][8];
  #pragma unroll
  for(int i=0;i<8;++i)
    #pragma unroll
    for(int j=0;j<8;++j)acc[i][j]=0.f;
  // Slice sl's SMEM: stage g at sm + sl*SLICE_FLOATS + g*2*TILE_FLOATS (A), + TILE_FLOATS (B). Named barrier 1+sl.
  float* ss=sm+size_t(sl)*SLICE_FLOATS;
  const int nblk=ke>kb?(ke-kb+WK-1)/WK:0;          // 8-deep k-blocks of K_z; slice sl takes j = sl, sl+2, ...
  const int mine=nblk>sl?(nblk-sl+1)/2:0;
  auto bar=[&]{asm volatile("bar.sync %0, %1;"::"r"(1+sl),"r"(SLICE):"memory");};
  la.init(Ap,a.lda,a.M,m0,kb+sl*WK,lt);lb.init(Bp,a.ldb,a.N,n0,kb+sl*WK,lt);
  if(mine>0){la.load(0,ke,Ap,a.lda,m0,lt);lb.load(0,ke,Bp,a.ldb,n0,lt);la.store(ss,lt);lb.store(ss+TILE_FLOATS,lt);}
  bar();
  const int tm=lt&15,tn=lt>>4;   // 16 x 16 threads of the slice; rows tm*4+{0..3}, 64+..; cols tn*4+{0..3}, 64+..
  float4 fa[2][2],fb[2][2];
  auto frag=[&](int f,const float* as,int k){const float* bs=as+TILE_FLOATS;
    fa[f][0]=*reinterpret_cast<const float4*>(as+k*LDS+tm*4);fa[f][1]=*reinterpret_cast<const float4*>(as+k*LDS+64+tm*4);
    fb[f][0]=*reinterpret_cast<const float4*>(bs+k*LDS+tn*4);fb[f][1]=*reinterpret_cast<const float4*>(bs+k*LDS+64+tn*4);};
  auto fma_step=[&](int f){
    const float av[8]={fa[f][0].x,fa[f][0].y,fa[f][0].z,fa[f][0].w,fa[f][1].x,fa[f][1].y,fa[f][1].z,fa[f][1].w};
    const float bv[8]={fb[f][0].x,fb[f][0].y,fb[f][0].z,fb[f][0].w,fb[f][1].x,fb[f][1].y,fb[f][1].z,fb[f][1].w};
    #pragma unroll
    for(int i=0;i<8;++i)
      #pragma unroll
      for(int j=0;j<8;++j)acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);};
  if(mine>0)frag(0,ss,0);
  for(int t=0;t<mine;++t){
    const int cur=t&1;
    if(t+1<mine){la.load(t+1,ke,Ap,a.lda,m0,lt);lb.load(t+1,ke,Bp,a.ldb,n0,lt);}
    const float* as=ss+cur*2*TILE_FLOATS;
    #pragma unroll
    for(int k=0;k<WK;++k){
      if(k+1<WK)frag((k+1)&1,as,k+1);
      fma_step(k&1);
    }
    if(t+1<mine){float* nas=ss+(cur^1)*2*TILE_FLOATS;la.store(nas,lt);lb.store(nas+TILE_FLOATS,lt);}
    bar();
    if(t+1<mine)frag(0,ss+(cur^1)*2*TILE_FLOATS,0);
  }
  __syncthreads();   // both slices done: the park region overlaps both pipelines
  // park: slice sl writes its partial column-major into part[sl]
  {float* part=sm+size_t(sl)*PARK_FLOATS;
   #pragma unroll
   for(int jj=0;jj<8;++jj){const int n=(jj<4?tn*4+jj:64+tn*4+(jj-4));
     *reinterpret_cast<float4*>(part+size_t(n)*PARK_LD+tm*4)=make_float4(acc[0][jj],acc[1][jj],acc[2][jj],acc[3][jj]);
     *reinterpret_cast<float4*>(part+size_t(n)*PARK_LD+64+tm*4)=make_float4(acc[4][jj],acc[5][jj],acc[6][jj],acc[7][jj]);}}
  if constexpr(CL>1){asm volatile("barrier.cluster.arrive.release.aligned;":::"memory");asm volatile("barrier.cluster.wait.acquire.aligned;":::"memory");}
  else __syncthreads();
  constexpr int SL=BM/CL;
  const int r0=z*SL;
  float*O=a.CG>1?a.P+(size_t(batch)*a.CG+gz)*a.sp:a.O+batch*a.so;
  const bool sub=EPI==SUB&&a.CG==1;
  const bool full=(m0+BM<=a.M)&&(n0+BN<=a.N);
  constexpr int U=8;
  for(int e0=tid;e0<SL*BN;e0+=THREADS*U){
    float xv[U],sv[U];long long go[U];bool in[U];
    #pragma unroll
    for(int u=0;u<U;++u){const int e=e0+u*THREADS;const int i=r0+e%SL,j=e/SL;go[u]=(m0+i)+(long long)(n0+j)*a.ldo;
      in[u]=e<SL*BN&&(full||((m0+i)<a.M&&(n0+j)<a.N));
      float s=0.f;
      #pragma unroll
      for(int r=0;r<CL;++r){                                 // fixed order z' = 0..CL-1, w = 0..SK-1
        const float*peer=sm;
        if constexpr(CL>1)peer=cooperative_groups::this_cluster().map_shared_rank(sm,r);
        #pragma unroll
        for(int w=0;w<SK;++w)s+=(e<SL*BN)?peer[size_t(w)*PARK_FLOATS+size_t(j)*PARK_LD+i]:0.f;}
      sv[u]=s;xv[u]=(sub&&in[u])?O[go[u]]:0.f;}
    #pragma unroll
    for(int u=0;u<U;++u)if(in[u])O[go[u]]=sub?xv[u]-sv[u]:sv[u];  // owner commits once (or its GPU-group slice)
  }
  if constexpr(CL>1){asm volatile("barrier.cluster.arrive.relaxed.aligned;":::"memory");asm volatile("barrier.cluster.wait.aligned;":::"memory");}
  if(a.wit&&tid==0&&z==0&&gz==0&&blockIdx.x==0&&blockIdx.y==0){
    atomicAdd(&a.wit->device_peer_partials[a.kind],(unsigned long long)(CL*SK*a.CG));
    atomicAdd(&a.wit->device_block_peers[a.kind],(unsigned long long)SK);
    unsigned cl_dim=1;if constexpr(CL>1)cl_dim=cooperative_groups::this_cluster().dim_blocks().z;
    witness_levels(a.wit,a.kind,SK,int(cl_dim),a.CG,1);
    atomicAdd(&a.wit->device_membership_reports,1ULL);
    if constexpr(EPI==SUB)atomicAdd(&a.wit->device_physical_commits,1ULL);
    else if(batch==0)atomicAdd(&a.wit->device_combines,1ULL);
  }
}
template<bool AKC,bool BKC,int EPI,int CL> void launch(const Args&a,int count,cudaStream_t st){
  if(!a.M||!a.N||!count)return;
  static int last=-1;int dev;CU(cudaGetDevice(&dev));
  if(dev!=last){CU(cudaFuncSetAttribute(simt_carrier_kernel<AKC,BKC,EPI,CL>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(SMEM)));last=dev;}
  if(a.CG>1&&(!a.P||a.sp<(long long)a.ldo*a.N))throw std::runtime_error("simt_carrier_gpu_partials");
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(ceildiv(a.M,BM),ceildiv(a.N,BN),count*CL*a.CG);
  cfg.blockDim=dim3(SLICE,SK,1);cfg.dynamicSmemBytes=SMEM;cfg.stream=st;
  cudaLaunchAttribute at[1];int na=0;
  if constexpr(CL>1){at[0].id=cudaLaunchAttributeClusterDimension;at[0].val.clusterDim.x=1;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=CL;na=1;}
  cfg.attrs=at;cfg.numAttrs=na;
  CU(cudaLaunchKernelEx(&cfg,simt_carrier_kernel<AKC,BKC,EPI,CL>,a));
  if(a.CG>1){carrier_g_combine<float><<<std::min(1024,ceildiv(a.M*a.N*count,256)),256,0,st>>>(a.P,a.sp,a.CG,a.O,int(a.ldo),a.so,a.N,a.M,count,EPI==SUB);CU(cudaGetLastError());}
}
inline bool aligned16(const void*p,long long ld){return (reinterpret_cast<uintptr_t>(p)%16==0)&&(ld%4==0);}
}  // namespace simt
// W(h x q, ldw) = V^T X with c = SK x CL (SK = 2 slices, CL = c/2 cluster CTAs), optional GPU groups cg.
inline bool simt_w_admits(int c,const float*v,int ldv,const float*x,int ldx,const float*w,int ldw){
  return (c==2||c==4||c==8)&&simt::aligned16(v,ldv)&&simt::aligned16(x,ldx)&&w;}
inline int simt_w_gpu_split(int c,int q,int h,int rows,int count){
  const int cl=c/simt::SK;const long long tiles=(long long)ceildiv(h,simt::BM)*ceildiv(q,simt::BN)*count*cl;
  long long cg=std::max(1LL,2LL*carrier_d_sms()/std::max(1LL,tiles));
  cg=std::min<long long>({cg,128,std::max(1,rows/(256*cl))});return int(std::max(1LL,cg));}
inline int launch_simt_w(int c,const float*v,int ldv,long long sv,const float*x,int ldx,long long sx,float*w,int ldw,long long sw,
                         int rows,int h,int q,int count,cudaStream_t st,Witness*wit,int kind,int cg,float*part){
  simt::Args a{v,ldv,sv,x,ldx,sx,w,ldw,sw,h,q,rows,wit,kind,std::max(1,cg),part,(long long)ldw*q,0};
  if(c==2)simt::launch<true,true,simt::STORE,1>(a,count,st);else if(c==4)simt::launch<true,true,simt::STORE,2>(a,count,st);
  else simt::launch<true,true,simt::STORE,4>(a,count,st);
  return c;}
// X(rows x q) -= V Z with Zt = Z^T (q x h); block c = 2 (two warp slices, BK = 16, WK = 8: the CarrierS2 law).
inline bool simt_d_admits(int c,const float*v,int ldv,const float*zt,int ldzt,const float*x,int ldx){
  return c==2&&simt::aligned16(v,ldv)&&simt::aligned16(zt,ldzt)&&(reinterpret_cast<uintptr_t>(x)%4==0);}
inline int launch_simt_d(const float*v,int ldv,long long sv,const float*zt,int ldzt,long long szt,float*x,int ldx,long long sx,
                         int rows,int h,int q,int count,cudaStream_t st,Witness*wit){
  static int l2=0;if(!l2){int dev=0;CU(cudaGetDevice(&dev));CU(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,dev));}
  const int nx=ceildiv(rows,simt::BM);
  // Raster groups of row tiles when V's row stripes for all column tiles exceed half the L2 (the carrier_d rule).
  const int raster=size_t(q)*h*sizeof(float)*2>size_t(l2)?16:0;
  simt::Args a{v,ldv,sv,zt,ldzt,szt,x,ldx,sx,rows,q,h,wit,4,1,nullptr,0,raster>0&&raster<nx?raster:0};
  simt::launch<false,false,simt::SUB,1>(a,count,st);return 2;}
}  // namespace tqr
