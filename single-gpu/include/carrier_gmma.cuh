#pragma once
// TF32 and 3xTF32 carriers for W and D on Hopper wgmma.
// TF32 W and D carriers on Hopper GMMA (wgmma) with the block-level K split of gmma_ksplit.cuh. The same
// carriers in 3xTF32 (fp32-math x3). D pre-splits V and Z into three K segments (pack_vr3 / pack_z3:
// small*big, big*small, big*big) and runs the TF32 K-split kernel at K' = 3 hs; W runs the X3 K-split
// kernel (in-kernel split of each peer's K_z columns, chunk promotion).
//  D : X(rows x q) -= VR(rows x h) Z(h x q)    VR = V row-major (K-major A), Z column-major (K-major B)
//  W : W(h x q)     = V^T(h x rows) X(rows x q) V column-major = V^T row-major (K-major A), X col-major
// TF32 GMMA reads both operands K-major from shared memory, so D needs V row-major: `pack_vr` writes it
// per product into the slot's VR buffer. Carriers: D c = 2 at the block level. W c = 2 CG: block 2 x GPU
// CG, the GPU level cutting [K] = rows into CG equal contiguous slices (L-batched GEMM into HBM partial
// slices, summed by the fixed-order combine g = 0..CG-1 that commits W once); CG > 1 requires rows % (32
// CG) == 0 so every slice is whole k-tiles.
#include "gmma_ksplit.cuh"
#include "carrier_gemm.cuh"
#include "gmma_x3rs.cuh"
namespace tqr {
inline constexpr int gmma_wg_threads=128;   // threads of one consumer warp group (a block-level peer)
// TMA addressability: 16-byte base and a leading dimension that is a multiple of 4 floats.
template<class P> bool gmma_aligned(const P*p,long long ld){return (reinterpret_cast<uintptr_t>(p)%16==0)&&(ld%4==0);}
// Mode predicates: fp32 storage with TF32 math or error-compensated 3xTF32 math.
template<class T> bool gmma_tf32(){if constexpr(std::is_same_v<T,float>)return fp32_math()==Fp32Math::TF32;else return false;}
template<class T> bool gmma_x3(){if constexpr(std::is_same_v<T,float>)return fp32_math()==Fp32Math::X3;else return false;}
template<class T> bool gmma_enabled(){return gmma_tf32<T>()||gmma_x3<T>();}
inline int gmma_x3_seg(int h){return (h+31)/32*32;}
inline bool gmma_d_supported(int c,int h,int count){return c==2&&h>=32&&count==1;}
// The GPU level cuts [K] = rows into CG slices of ks whole 32-row k-tiles each (ks = 32 floor(rows/(32 CG))) plus
// a ragged remainder of rem = rows - CG ks < 32 CG rows that belongs to the LAST peer's K_z.
inline bool gmma_w_supported(int c,int rows,int count){
 if(count!=1||(c!=2&&c!=4&&c!=8))return false;const int cg=c/2;return rows>=32*cg;}
inline int gmma_w_slice(int rows,int cg){return cg>1?rows/(32*cg)*32:rows;}
// W (+)= sum_{g<CG} P_g + V_rem^T X_rem in the fixed order g = 0..CG-1, then r = 0..rem-1; one commit.
// V_rem = V + r0 (rem x h, ld ldv), X_rem = X + r0 (rem x q, ld ldx); fp32 FMA (at least the product's precision).
__global__ void gmma_w_combine_rem(const float* __restrict__ P,long long sp,int CG,float* __restrict__ W,int ldw,int q,int h,
                                   const float* __restrict__ Vr,int ldv,const float* __restrict__ Xr,int ldx,int rem){
 extern __shared__ float xs[];                      // rem x 8 columns of X_rem
 const int i0=blockIdx.x*8;
 for(int e=threadIdx.x;e<rem*8;e+=blockDim.x){const int r=e%rem,c=e/rem;xs[r*8+c]=(i0+c<q)?Xr[r+(long long)(i0+c)*ldx]:0.f;}
 __syncthreads();
 for(int e=threadIdx.x;e<8*h;e+=blockDim.x){
  const int j=e%h,c=e/h,i=i0+c;if(i>=q)continue;const long long go=j+(long long)i*ldw;
  float s=0.f;for(int g=0;g<CG;++g)s+=P[size_t(g)*sp+go];
  const float*vj=Vr+(long long)j*ldv;float t=0.f;for(int r=0;r<rem;++r)t=fmaf(vj[r],xs[r*8+c],t);
  W[go]=s+t;}
}
inline void gmma_w_combine(const float*part,long long sp,int cg,float*w,int ldw,int q,int h,const float*v,int ldv,const float*x,int ldx,int rows,cudaStream_t st){
 const int ks=gmma_w_slice(rows,cg),rem=rows-cg*ks;
 if(!rem){carrier_g_combine<float><<<std::min(1024,ceildiv(h*q,256)),256,0,st>>>(part,sp,cg,w,ldw,0,q,h,1,false);CU(cudaGetLastError());return;}
 gmma_w_combine_rem<<<ceildiv(q,8),256,size_t(rem)*8*sizeof(float),st>>>(part,sp,cg,w,ldw,q,h,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,rem);CU(cudaGetLastError());}
// VR[r*ldvr + k] = V[r + k*ldv], r < rows, k < h.
__global__ void pack_vr(const float* __restrict__ V,int ldv,int rows,int h,float* __restrict__ VR,int ldvr){
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<(long long)rows*h;i+=(long long)gridDim.x*blockDim.x){
  const long long r=i/h;const int k=int(i%h);VR[r*ldvr+k]=V[r+(long long)k*ldv];}}
// VR3[r*ld3 + {0,hs,2hs} + k] = {small, big, big} of V[r + k*ldv] (zero for k >= h), rows x 3hs.
__global__ void pack_vr3(const float* __restrict__ V,int ldv,int rows,int h,int hs,float* __restrict__ VR3,int ld3){
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<(long long)rows*hs;i+=(long long)gridDim.x*blockDim.x){
  const long long r=i/hs;const int k=int(i%hs);const float v=k<h?V[r+(long long)k*ldv]:0.f,b=gmma::tf32_rna(v),m=gmma::tf32_rna(v-b);
  float*o=VR3+r*ld3;o[k]=m;o[hs+k]=b;o[2*hs+k]=b;}}
// Z3[n*ld3 + {0,hs,2hs} + k] = {big, small, big} of Z[k + n*ldz] (zero for k >= h), 3hs x q column-major.
__global__ void pack_z3(const float* __restrict__ Z,int ldz,int h,int hs,int q,float* __restrict__ Z3,int ld3){
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<(long long)q*hs;i+=(long long)gridDim.x*blockDim.x){
  const long long n=i/hs;const int k=int(i%hs);const float z=k<h?Z[k+n*ldz]:0.f,b=gmma::tf32_rna(z),m=gmma::tf32_rna(z-b);
  float*o=Z3+n*ld3;o[k]=b;o[hs+k]=m;o[2*hs+k]=b;}}
inline int pack_vr_grid(long long n){return int(std::max(32LL,std::min(4096LL,(n+1023)/1024)));}
// transpose through a padded shared tile, coalescing both V loads and VR stores. The x3 split is
// unchanged and occurs after the transpose. Each slot's existing generation/cache checks still
// decide when this packing is needed.
template<bool X3>
__global__ void pack_vr_tiled(const float* __restrict__ V,int ldv,int rows,int h,int hs,
                             float* __restrict__ VR,int ldvr){
 __shared__ float tile[32][33];
 const int nr=(rows+31)/32,nk=((X3?hs:h)+31)/32;
 for(long long ti=blockIdx.x;ti<(long long)nr*nk;ti+=gridDim.x){
  const int r0=int(ti%nr)*32,k0=int(ti/nr)*32;
  const int r=r0+threadIdx.x;
  #pragma unroll
  for(int j=0;j<32;j+=8){
   const int k=k0+threadIdx.y+j;
   tile[threadIdx.y+j][threadIdx.x]=(r<rows&&k<h)?V[r+(long long)k*ldv]:0.f;
  }
  __syncthreads();
  const int k=k0+threadIdx.x;
  #pragma unroll
  for(int j=0;j<32;j+=8){
   const int rr=r0+threadIdx.y+j;
   if(rr<rows&&k<(X3?hs:h)){
    const float v=tile[threadIdx.x][threadIdx.y+j];float*o=VR+(long long)rr*ldvr;
    if constexpr(X3){const float b=gmma::tf32_rna(v),s=gmma::tf32_rna(v-b);o[k]=s;o[hs+k]=b;o[2*hs+k]=b;}
    else o[k]=v;
   }
  }
  __syncthreads();
 }
}
inline bool pack_vr_tiled_enabled(){static const bool on=[]{const char*e=std::getenv("TQR_PACK_VR_TILED");return e&&std::atoi(e)==1;}();return on;}
inline unsigned long long& pack_vr_tiled_launches(){static unsigned long long n=0;return n;}
inline void launch_pack_vr(const float*v,int ldv,int rows,int h,int hs,float*vr,int ldvr,bool x3,cudaStream_t st){
 if(pack_vr_tiled_enabled()){
  const int grid=int(std::min(4096LL,(long long)ceildiv(rows,32)*ceildiv(x3?hs:h,32)));
  if(x3)pack_vr_tiled<true><<<grid,dim3(32,8),0,st>>>(v,ldv,rows,h,hs,vr,ldvr);
  else pack_vr_tiled<false><<<grid,dim3(32,8),0,st>>>(v,ldv,rows,h,h,vr,ldvr);
  ++pack_vr_tiled_launches();
 }else if(x3)pack_vr3<<<pack_vr_grid((long long)rows*hs),256,0,st>>>(v,ldv,rows,h,hs,vr,ldvr);
 else pack_vr<<<pack_vr_grid((long long)rows*h),256,0,st>>>(v,ldv,rows,h,vr,ldvr);
 CU(cudaGetLastError());
}
namespace gmma_detail {
using LCM=cutlass::layout::ColumnMajor;
using GemmD=gmma::KSplitGemm<128,128,LCM>::Gemm;
using GemmD2CTA=gmma::KSplitGemm2CTA<LCM>::Gemm;
using GemmW3=gmma::KSplitGemm<128,128,LCM,void,1>::Gemm;   // In-kernel 3xTF32 split, chunk promotion
// 3xTF32 W in two launches over a pre-split V: mode 2 (A = V_big, B = X split in the peer's K_z) writes, mode
// 3 (A = V_small, B = X raw, truncated) adds with beta = 1. Same carrier, K_z sets and commit.
using GemmW3a=gmma::KSplitGemm<128,128,LCM,void,2>::Gemm;
using GemmW3b=gmma::KSplitGemm<128,128,LCM,void,3>::Gemm;
// A positive cap limits the persistent grid to that many SMs (KernelHardwareInfo::sm_count), leaving the
// rest to the concurrent panel. Set by the engine for the look-ahead window-B products only
// (GmmaSmCapScope) from the schedule's lookahead_free_sms; 0 = the device's SM count.
inline int& gmma_sm_cap(){static int c=0;return c;}
// non-persistent GMMA grids (TQR_GMMA_NP). The persistent tile scheduler launched with an unbounded sm_count
// truncates its grid to the tile count: one CTA per output tile, so a product releases SMs tile by tile (~10 us) to
// the higher-priority panel instead of holding every SM for its whole duration. 1 = look-ahead window-B products
// only, 2 = every GMMA product. The carrier (block c = 2 K halves, fixed-order combine, one commit) is unchanged;
// only the grid size differs.
inline int gmma_np_mode(){static const int m=[]{const char*e=std::getenv("TQR_GMMA_NP");return e?std::atoi(e):0;}();return m;}
inline constexpr int gmma_np_cap=1<<30;
inline bool gmma_d_2cta(){static const bool on=[]{const char*e=std::getenv("TQR_GMMA_D_2CTA");return e&&std::atoi(e)==1;}();return on&&gmma_sm_cap()==gmma_np_cap;}
inline unsigned long long& gmma_d_2cta_launches(){static unsigned long long n=0;return n;}
// register-split x3 W switch (see launch_gmma_w3rs).
inline bool x3rs_enabled(){static const bool on=[]{const char*e=std::getenv("TQR_X3RS");return e&&std::atoi(e)==1;}();return on;}
inline unsigned long long& x3rs_launches(){static unsigned long long n=0;return n;}
inline bool gmma_z_enabled(){static const bool on=[]{const char*e=std::getenv("TQR_GMMA_Z");return e&&std::atoi(e)==1;}();return on;}
inline unsigned long long& gmma_z_launches(){static unsigned long long n=0;return n;}
inline int gmma_g_groups(){static const int cg=[]{const char*e=std::getenv("TQR_GMMA_G_GROUPS");const int v=e?std::atoi(e):0;
 if(v<0||v>128)throw std::runtime_error("gmma_G_groups_requires_0_to_128");return v;}();return cg;}
inline unsigned long long& gmma_g_launches(){static unsigned long long n=0;return n;}
// explicit tiles per CTA for the finite GMMA grids. Reuse the existing static scheduler, so each
// CTA processes at most ceil(tiles/grid) whole output tiles before releasing its SM to the panel. K
// peers, partials and the owner epilogue are unchanged. This is a launch parameter, not a runtime
// selector.
inline int gmma_np_tpc(bool d){
 static const int dt=[](){const char*e=std::getenv("TQR_GMMA_D_TPC");return e?std::atoi(e):1;}();
 static const int wt=[](){const char*e=std::getenv("TQR_GMMA_W_TPC");return e?std::atoi(e):1;}();
 const int t=d?dt:wt;if(t<1||t>64)throw std::runtime_error("gmma_tpc_requires_1_to_64");return t;
}
struct GmmaFiniteGridStats {unsigned long long launches=0,tiles=0,ctas=0,limited=0,max_tiles_per_cta=0;};
inline GmmaFiniteGridStats& gmma_finite_stats(bool d){static GmmaFiniteGridStats s[2];return s[d?1:0];}
inline json gmma_finite_record(){
 json r={{"D_tiles_per_cta",gmma_np_tpc(true)},{"W_tiles_per_cta",gmma_np_tpc(false)},
         {"D_2cta_launches",gmma_d_2cta_launches()},{"Z_enabled",gmma_z_enabled()},{"x3rs_enabled",x3rs_enabled()},{"x3rs_launches",x3rs_launches()},{"Z_launches",gmma_z_launches()},
         {"VR_tiled_enabled",pack_vr_tiled_enabled()},{"VR_tiled_launches",pack_vr_tiled_launches()},
         {"G_requested_groups",gmma_g_groups()},{"G_launches",gmma_g_launches()},
         {"scope","GMMA products selected by TQR_GMMA_NP; static whole-output-tile scheduling; W counters include Z"}};
 for(bool d:{false,true}){const auto&s=gmma_finite_stats(d);r[d?"D":"W"]={{"launches",s.launches},{"tiles",s.tiles},
   {"ctas",s.ctas},{"grid_y_limit_launches",s.limited},{"max_tiles_per_cta",s.max_tiles_per_cta}};}
 return r;
}
inline unsigned long long& gmma_capped_launches(){static unsigned long long n=0;return n;}   // GMMA grids actually capped
struct GmmaSmCapScope{int old;explicit GmmaSmCapScope(int c):old(gmma_sm_cap()){gmma_sm_cap()=c;}~GmmaSmCapScope(){gmma_sm_cap()=old;}};
template<class Gemm> void run(Gemm&g,typename Gemm::Arguments&a,cudaStream_t st){
 if(gmma_sm_cap()>0){int dev=0;CU(cudaGetDevice(&dev));a.hw_info.device_id=dev;a.hw_info.sm_count=gmma_sm_cap();++gmma_capped_launches();}
 unsigned long long tiles=0,ctas=0;bool limited=false;
 if(gmma_sm_cap()==gmma_np_cap){
  using K=typename Gemm::GemmKernel;using Scheduler=typename K::TileScheduler;
  static_assert(cute::size(typename K::ClusterShape{})==1,"finite grid currently uses 1-CTA clusters");
  const auto grid=Scheduler::get_grid_shape(typename Scheduler::Params{},a.problem_shape,
                   typename K::TileShape{},typename K::ClusterShape{},a.hw_info,a.scheduler);
  tiles=uint64_t(grid.x)*grid.y*grid.z;
  const int tpc=gmma_np_tpc(a.mainloop.kind==4);
  const auto requested=(tiles+tpc-1)/tpc;
  // CUTLASS rasterizes tall D into grid.y. CUDA caps that dimension at 65535:
  // the first 131072 x 8192 far strip otherwise launches 65536 CTAs and fails.
  // Keeping the same scheduler with a smaller grid covers the remaining tile.
  const auto limit=grid.y>1?65535ULL:2147483647ULL;
  ctas=std::max(1ULL,std::min(requested,limit));limited=requested>limit;
  a.hw_info.sm_count=int(ctas);
 }
 const size_t ws=Gemm::get_workspace_size(a);
 if(ws)throw std::runtime_error("gmma_unexpected_workspace");
 if(g.can_implement(a)!=cutlass::Status::kSuccess)throw std::runtime_error("gmma_cannot_implement");
 if(g.initialize(a,nullptr,st)!=cutlass::Status::kSuccess)throw std::runtime_error("gmma_initialize");
 if(g.run(st)!=cutlass::Status::kSuccess){
  fprintf(stderr,"gmma_run kind=%d shape=%d,%d,%d,%d grid_budget=%d\n",a.mainloop.kind,
    int(cute::get<0>(a.problem_shape)),int(cute::get<1>(a.problem_shape)),
    int(cute::get<2>(a.problem_shape)),int(cute::get<3>(a.problem_shape)),a.hw_info.sm_count);
  throw std::runtime_error("gmma_run");}
 CU(cudaGetLastError());
 if(tiles){auto&s=gmma_finite_stats(a.mainloop.kind==4);++s.launches;s.tiles+=tiles;s.ctas+=ctas;s.limited+=limited;
  s.max_tiles_per_cta=std::max(s.max_tiles_per_cta,(tiles+ctas-1)/ctas);}
}
}
// Arguments of the two products (shared by the admission test and the launch, so what is admitted is
// exactly what runs).
template<class G=gmma_detail::GemmD>
inline typename G::Arguments gmma_d_args(const float*vr,int ldvr,const float*z,int ldz,float*x,int ldx,int rows,int h,int q,Witness*wit){
 typename G::Arguments a;a.mode=cutlass::gemm::GemmUniversalMode::kGemm;a.problem_shape={rows,q,h,1};
 a.mainloop.ptr_A=vr;a.mainloop.dA=typename G::GemmKernel::StrideA{int64_t(ldvr),cute::Int<1>{},int64_t(0)};
 a.mainloop.ptr_B=z;a.mainloop.dB=typename G::GemmKernel::StrideB{int64_t(ldz),cute::Int<1>{},int64_t(0)};
 a.mainloop.wit=wit;a.mainloop.kind=4;a.mainloop.commit=1;a.mainloop.gpu=1;
 a.epilogue.thread.alpha=-1.f;a.epilogue.thread.beta=1.f;
 a.epilogue.ptr_C=x;a.epilogue.dC=typename G::GemmKernel::StrideC{cute::Int<1>{},int64_t(ldx),int64_t(0)};
 a.epilogue.ptr_D=x;a.epilogue.dD=a.epilogue.dC;return a;}
inline gmma_detail::GemmD::Arguments gmma_w_args(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,
                                                 Witness*wit,int kind,float*part,long long sp){
 using G=gmma_detail::GemmD;const int cg=c/2,ks=gmma_w_slice(rows,cg);typename G::Arguments a;
 a.mode=cg>1?cutlass::gemm::GemmUniversalMode::kBatched:cutlass::gemm::GemmUniversalMode::kGemm;a.problem_shape={h,q,ks,cg};
 a.mainloop.ptr_A=v;a.mainloop.dA=typename G::GemmKernel::StrideA{int64_t(ldv),cute::Int<1>{},int64_t(ks)};
 a.mainloop.ptr_B=x;a.mainloop.dB=typename G::GemmKernel::StrideB{int64_t(ldx),cute::Int<1>{},int64_t(ks)};
 a.mainloop.wit=wit;a.mainloop.kind=kind;a.mainloop.commit=0;a.mainloop.gpu=cg;
 a.epilogue.thread.alpha=1.f;a.epilogue.thread.beta=0.f;float*o=cg>1?part:w;
 a.epilogue.ptr_C=o;a.epilogue.dC=typename G::GemmKernel::StrideC{cute::Int<1>{},int64_t(ldw),int64_t(cg>1?sp:0)};
 a.epilogue.ptr_D=o;a.epilogue.dD=a.epilogue.dC;return a;}
template<class G=gmma_detail::GemmW3>
inline typename G::Arguments gmma_w3_args(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,
                                          Witness*wit,int kind,float*part,long long sp,float beta=0.f){
 using E=cutlass::tfloat32_t;const int cg=c/2,ks=gmma_w_slice(rows,cg);typename G::Arguments a;
 a.mode=cg>1?cutlass::gemm::GemmUniversalMode::kBatched:cutlass::gemm::GemmUniversalMode::kGemm;a.problem_shape={h,q,ks,cg};
 a.mainloop.ptr_A=reinterpret_cast<const E*>(v);a.mainloop.dA=typename G::GemmKernel::StrideA{int64_t(ldv),cute::Int<1>{},int64_t(ks)};
 a.mainloop.ptr_B=reinterpret_cast<const E*>(x);a.mainloop.dB=typename G::GemmKernel::StrideB{int64_t(ldx),cute::Int<1>{},int64_t(ks)};
 a.mainloop.wit=wit;a.mainloop.kind=kind;a.mainloop.commit=0;a.mainloop.gpu=cg;
 a.epilogue.thread.alpha=1.f;a.epilogue.thread.beta=beta;float*o=cg>1?part:w;
 a.epilogue.ptr_C=o;a.epilogue.dC=typename G::GemmKernel::StrideC{cute::Int<1>{},int64_t(ldw),int64_t(cg>1?sp:0)};
 a.epilogue.ptr_D=o;a.epilogue.dD=a.epilogue.dC;return a;}
// V (rows x h, col-major ldv) -> V_big, V_small (col-major, ld ldvs), tf32 round-to-nearest split.
__global__ void split_v_rn(const float* __restrict__ V,int ldv,int rows,int h,float* __restrict__ Vb,float* __restrict__ Vs,int ldvs){
 for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<(long long)rows*h;i+=(long long)gridDim.x*blockDim.x){
  const int r=int(i%rows);const long long k=i/rows;const float v=V[r+k*ldv],b=gmma::tf32_rna(v);Vb[r+k*ldvs]=b;Vs[r+k*ldvs]=gmma::tf32_rna(v-b);}}
inline int gmma_vsplit_ld(int rows){return (rows+3)/4*4;}
inline size_t gmma_vsplit_words(int rows,int h){return 2*size_t(gmma_vsplit_ld(rows))*size_t(h);}
// Admission: the law's shape conditions, 16-byte base addresses, and CUTLASS's own TMA alignment test
// on exactly these arguments (contiguous extents and strides in whole 16-byte units).
inline bool gmma_d_admits(const float*vr,int ldvr,const float*z,int ldz,float*x,int ldx,int rows,int h,int q,int c,int count){
 if(!gmma_d_supported(c,h,count)||!rows||!q||!gmma_aligned(vr,ldvr)||!gmma_aligned(z,ldz)||!gmma_aligned(x,ldx))return false;
 if(gmma_detail::gmma_d_2cta())return gmma_detail::GemmD2CTA::can_implement(gmma_d_args<gmma_detail::GemmD2CTA>(vr,ldvr,z,ldz,x,ldx,rows,h,q,nullptr))==cutlass::Status::kSuccess;
 return gmma_detail::GemmD::can_implement(gmma_d_args(vr,ldvr,z,ldz,x,ldx,rows,h,q,nullptr))==cutlass::Status::kSuccess;}
inline bool gmma_w_admits(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,int count,float*part,long long sp){
 if(!gmma_w_supported(c,rows,count)||!h||!q||!gmma_aligned(v,ldv)||!gmma_aligned(x,ldx)||!gmma_aligned(w,ldw)||(c>2&&!gmma_aligned(part,sp)))return false;
 return gmma_detail::GemmD::can_implement(gmma_w_args(c,v,ldv,x,ldx,w,ldw,rows,h,q,nullptr,2,part,sp))==cutlass::Status::kSuccess;}
inline bool gmma_w3_admits(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,int count,float*part,long long sp){
 if(!gmma_w_supported(c,rows,count)||!h||!q||!gmma_aligned(v,ldv)||!gmma_aligned(x,ldx)||!gmma_aligned(w,ldw)||(c>2&&!gmma_aligned(part,sp)))return false;
 const int ldvs=gmma_vsplit_ld(rows);if(c>2&&ldvs!=rows)return false;
 return gmma_detail::GemmW3a::can_implement(gmma_w3_args<gmma_detail::GemmW3a>(c,v,ldvs,x,ldx,w,ldw,rows,h,q,nullptr,2,part,sp))==cutlass::Status::kSuccess&&
        gmma_detail::GemmW3b::can_implement(gmma_w3_args<gmma_detail::GemmW3b>(c,v,ldvs,x,ldx,w,ldw,rows,h,q,nullptr,2,part,sp,1.f))==cutlass::Status::kSuccess;}
// X(rows x q) -= VR Z; returns the executed block c (2).
inline int launch_gmma_d(const float*vr,int ldvr,const float*z,int ldz,float*x,int ldx,int rows,int h,int q,cudaStream_t st,Witness*wit){
 if(!rows||!q||!h)return 2;
 if(gmma_detail::gmma_d_2cta()){
  gmma_detail::GemmD2CTA g;auto a=gmma_d_args<gmma_detail::GemmD2CTA>(vr,ldvr,z,ldz,x,ldx,rows,h,q,wit);gmma_detail::run(g,a,st);++gmma_detail::gmma_d_2cta_launches();
 }else{gmma_detail::GemmD g;auto a=gmma_d_args(vr,ldvr,z,ldz,x,ldx,rows,h,q,wit);gmma_detail::run(g,a,st);}return 2;}
// Z = T^T W uses the same c=2 K halves as D, but produces a fresh Z. For x3
// both operands split inside each peer's own K slice, with chunk promotion.
// Q application with T (rather than T^T), batching, and transposed Z retain
// the incumbent path; no operand transpose or additional scratch is needed.
inline bool gmma_z_admits(const float*t,int ldt,const float*w,int ldw,float*z,int ldz,int h,int q,int c,int count,bool transpose,bool zt){
 if(!gmma_detail::gmma_z_enabled()||!transpose||zt||c!=2||count!=1||h<32||!q||
    !gmma_aligned(t,ldt)||!gmma_aligned(w,ldw)||!gmma_aligned(z,ldz))return false;
 if(gmma_x3<float>())return gmma_detail::GemmW3::can_implement(gmma_w3_args(2,t,ldt,w,ldw,z,ldz,h,h,q,nullptr,3,nullptr,0))==cutlass::Status::kSuccess;
 if(gmma_tf32<float>())return gmma_detail::GemmD::can_implement(gmma_w_args(2,t,ldt,w,ldw,z,ldz,h,h,q,nullptr,3,nullptr,0))==cutlass::Status::kSuccess;
 return false;
}
inline void launch_gmma_z(const float*t,int ldt,const float*w,int ldw,float*z,int ldz,int h,int q,cudaStream_t st,Witness*wit){
 if(gmma_x3<float>()){gmma_detail::GemmW3 g;auto a=gmma_w3_args(2,t,ldt,w,ldw,z,ldz,h,h,q,wit,3,nullptr,0);gmma_detail::run(g,a,st);}
 else{gmma_detail::GemmD g;auto a=gmma_w_args(2,t,ldt,w,ldw,z,ldz,h,h,q,wit,3,nullptr,0);gmma_detail::run(g,a,st);}
 ++gmma_detail::gmma_z_launches();
}
// W(h x q, ldw) = V^T X with c = 2 CG; part: CG slices of h x q (stride sp >= ldw q) when CG > 1.
inline int launch_gmma_w(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,
                         cudaStream_t st,Witness*wit,int kind,float*part,long long sp){
 if(!rows||!q||!h)return c;
 const int cg=c/2;if(!gmma_w_supported(c,rows,1))throw std::runtime_error("gmma_w_unsupported");
 if(cg>1&&(!part||sp<(long long)ldw*q))throw std::runtime_error("gmma_w_gpu_partials");
 gmma_detail::GemmD g;auto a=gmma_w_args(c,v,ldv,x,ldx,w,ldw,rows,h,q,wit,kind,part,sp);gmma_detail::run(g,a,st);
 if(cg>1){
  // The ragged remainder rows (rem < 32 CG, the tail of the LAST peer's K_z) run as a small GMMA that accumulates
  // (beta = 1) into that peer's partial slice, so the fixed-order combine below sums CG whole partials. TMA
  // zero-fills the ragged k-tile. Same carried product: no witness report. The SIMT remainder (gmma_w_combine_rem)
  // is kept only as the fallback when TMA cannot address the remainder.
  const int ks=gmma_w_slice(rows,cg),rem=rows-cg*ks;bool done=rem==0;
  if(rem>0){float*last=part+size_t(cg-1)*size_t(sp);
   auto r=gmma_w_args(2,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,last,ldw,rem,h,q,nullptr,kind,nullptr,0);r.epilogue.thread.beta=1.f;
   if(gmma_aligned(v+size_t(cg)*ks,ldv)&&gmma_aligned(x+size_t(cg)*ks,ldx)&&gmma_detail::GemmD::can_implement(r)==cutlass::Status::kSuccess){gmma_detail::GemmD gr;gmma_detail::run(gr,r,st);done=true;}}
  if(done){carrier_g_combine<float><<<std::min(1024,ceildiv(h*q,256)),256,0,st>>>(part,sp,cg,w,ldw,0,q,h,1,false);CU(cudaGetLastError());}
  else gmma_w_combine(part,sp,cg,w,ldw,q,h,v,ldv,x,ldx,rows,st);}
 return c;}
// W(h x q) = V^T X in 3xTF32, c = 2 CG, same GPU-level slices and fixed-order combine. With a split
// buffer (vsplit, gmma_vsplit_words) the product runs as two launches over the pre-split V; the second
// launch is arithmetic inside the same carried product and does not report to the witness.
inline int launch_gmma_w3(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,
                          cudaStream_t st,Witness*wit,int kind,float*part,long long sp,float*vsplit=nullptr){
 if(!rows||!q||!h)return c;
 const int cg=c/2;if(!gmma_w_supported(c,rows,1))throw std::runtime_error("gmma_w_unsupported");
 if(cg>1&&(!part||sp<(long long)ldw*q))throw std::runtime_error("gmma_w_gpu_partials");
 if(vsplit){const int ldvs=gmma_vsplit_ld(rows);float*vb=vsplit,*vs=vsplit+size_t(ldvs)*h;
  split_v_rn<<<pack_vr_grid((long long)rows*h),256,0,st>>>(v,ldv,rows,h,vb,vs,ldvs);CU(cudaGetLastError());
  {gmma_detail::GemmW3a g;auto a=gmma_w3_args<gmma_detail::GemmW3a>(c,vb,ldvs,x,ldx,w,ldw,rows,h,q,wit,kind,part,sp);gmma_detail::run(g,a,st);}
  {gmma_detail::GemmW3b g;auto a=gmma_w3_args<gmma_detail::GemmW3b>(c,vs,ldvs,x,ldx,w,ldw,rows,h,q,nullptr,kind,part,sp,1.f);gmma_detail::run(g,a,st);}
  // ragged remainder as the same two launches over the remainder rows, both accumulating (beta = 1) into the last
  // peer's partial (see launch_gmma_w).
  if(cg>1){const int ks=gmma_w_slice(rows,cg),rem=rows-cg*ks;bool done=rem==0;
   if(rem>0){float*last=part+size_t(cg-1)*size_t(sp);const size_t o=size_t(cg)*ks;
    auto ra=gmma_w3_args<gmma_detail::GemmW3a>(2,vb+o,ldvs,x+o,ldx,last,ldw,rem,h,q,nullptr,kind,nullptr,0,1.f);
    auto rb=gmma_w3_args<gmma_detail::GemmW3b>(2,vs+o,ldvs,x+o,ldx,last,ldw,rem,h,q,nullptr,kind,nullptr,0,1.f);
    if(gmma_aligned(vb+o,ldvs)&&gmma_aligned(vs+o,ldvs)&&gmma_aligned(x+o,ldx)&&gmma_detail::GemmW3a::can_implement(ra)==cutlass::Status::kSuccess&&gmma_detail::GemmW3b::can_implement(rb)==cutlass::Status::kSuccess){
     {gmma_detail::GemmW3a g;gmma_detail::run(g,ra,st);}{gmma_detail::GemmW3b g;gmma_detail::run(g,rb,st);}done=true;}}
   if(done){carrier_g_combine<float><<<std::min(1024,ceildiv(h*q,256)),256,0,st>>>(part,sp,cg,w,ldw,0,q,h,1,false);CU(cudaGetLastError());}
   else gmma_w_combine(part,sp,cg,w,ldw,q,h,v,ldv,x,ldx,rows,st);}
  return c;
 }else{gmma_detail::GemmW3 g;auto a=gmma_w3_args(c,v,ldv,x,ldx,w,ldw,rows,h,q,wit,kind,part,sp);gmma_detail::run(g,a,st);}
 if(cg>1)gmma_w_combine(part,sp,cg,w,ldw,q,h,v,ldv,x,ldx,rows,st);
 return c;}
// x3 W with the X split in registers (include/gmma_x3rs.cuh). TQR_X3RS=1 admits it for x3 W products with c in
// {2,4,8}: CG = c GPU-level contiguous K slices (block level unsplit: the warp groups split M), slice 0 written in
// place of W, slices 1..c-1 at w + g*sp, the ragged K tail accumulated (beta = 1) into the last slice, then the
// fixed-order combine g = 0..c-1 commits W once. V_big / V_small from split_v_rn in the slot's split region.
namespace gmma_detail {
using GemmW3RS=gmma::X3RSGemm<>::Gemm;
}
inline gmma_detail::GemmW3RS::Arguments gmma_w3rs_args(const float*x,int ldx,const float*vb,const float*vs,int ldvs,
    float*o,int ldw,long long sp,int q,int h,int ks,int L,Witness*wit,int kind,int gpu,float beta){
 using G=gmma_detail::GemmW3RS;using E=cutlass::tfloat32_t;typename G::Arguments a;
 a.mode=L>1?cutlass::gemm::GemmUniversalMode::kBatched:cutlass::gemm::GemmUniversalMode::kGemm;a.problem_shape={q,h,ks,L};
 a.mainloop.ptr_A=reinterpret_cast<const E*>(x);a.mainloop.dA={int64_t(ldx),cute::Int<1>{},int64_t(ks)};
 a.mainloop.ptr_B=reinterpret_cast<const E*>(vb);a.mainloop.dB={int64_t(ldvs),cute::Int<1>{},int64_t(ks)};
 a.mainloop.ptr_Bs=reinterpret_cast<const E*>(vs);a.mainloop.wit=wit;a.mainloop.kind=kind;a.mainloop.gpu=gpu;
 a.epilogue.thread.alpha=1.f;a.epilogue.thread.beta=beta;
 a.epilogue.ptr_C=o;a.epilogue.dC={int64_t(ldw),cute::Int<1>{},int64_t(L>1?sp:0)};a.epilogue.ptr_D=o;a.epilogue.dD=a.epilogue.dC;return a;}
inline bool gmma_w3rs_admits(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,int count,const float*vsplit){
 if(!gmma_detail::x3rs_enabled()||!gmma_x3<float>()||count!=1||(c!=2&&c!=4&&c!=8)||!h||!q||rows<32*c||!vsplit)return false;
 const int ldvs=gmma_vsplit_ld(rows);const float*vb=vsplit,*vs=vsplit+size_t(ldvs)*h;
 if(!gmma_aligned(v,ldv)||!gmma_aligned(x,ldx)||!gmma_aligned(w,ldw)||!gmma_aligned(vb,ldvs)||!gmma_aligned(vs,ldvs))return false;
 const long long sp=(long long)ldw*q;if(!gmma_aligned(w,sp))return false;
 const int ks=gmma_w_slice(rows,c),rem=rows-c*ks;
 if(gmma_detail::GemmW3RS::can_implement(gmma_w3rs_args(x,ldx,vb,vs,ldvs,w,ldw,sp,q,h,ks,c,nullptr,2,c,0.f))!=cutlass::Status::kSuccess)return false;
 if(rem){const size_t o=size_t(c)*ks;
  if(!gmma_aligned(x+o,ldx)||!gmma_aligned(vb+o,ldvs)||!gmma_aligned(vs+o,ldvs))return false;
  if(gmma_detail::GemmW3RS::can_implement(gmma_w3rs_args(x+o,ldx,vb+o,vs+o,ldvs,w+size_t(c-1)*sp,ldw,sp,q,h,rem,1,nullptr,2,1,1.f))!=cutlass::Status::kSuccess)return false;}
 return true;}
inline int launch_gmma_w3rs(int c,const float*v,int ldv,const float*x,int ldx,float*w,int ldw,int rows,int h,int q,
                            cudaStream_t st,Witness*wit,int kind,float*vsplit){
 if(!rows||!q||!h)return c;
 const int ldvs=gmma_vsplit_ld(rows);float*vb=vsplit,*vs=vsplit+size_t(ldvs)*h;
 split_v_rn<<<pack_vr_grid((long long)rows*h),256,0,st>>>(v,ldv,rows,h,vb,vs,ldvs);CU(cudaGetLastError());
 const long long sp=(long long)ldw*q;const int ks=gmma_w_slice(rows,c),rem=rows-c*ks;
 {gmma_detail::GemmW3RS g;auto a=gmma_w3rs_args(x,ldx,vb,vs,ldvs,w,ldw,sp,q,h,ks,c,wit,kind,c,0.f);gmma_detail::run(g,a,st);}
 if(rem){const size_t o=size_t(c)*ks;gmma_detail::GemmW3RS g;   // the last peer's K_z also owns the ragged tail
  auto a=gmma_w3rs_args(x+o,ldx,vb+o,vs+o,ldvs,w+size_t(c-1)*sp,ldw,sp,q,h,rem,1,nullptr,kind,1,1.f);gmma_detail::run(g,a,st);}
 carrier_g_combine<float><<<std::min(1024,ceildiv(h*q,256)),256,0,st>>>(w,sp,c,w,ldw,0,q,h,1,false);CU(cudaGetLastError());
 ++gmma_detail::x3rs_launches();return c;}
// explicit GPU K groups for skinny aggregate compose_G products. The same two concurrent GMMA
// warp-group peers run inside every GPU group. Whole 32-row slices belong to groups 0..cg-1; the
// final group also owns the ragged suffix. Its suffix accumulates into that group's partial before
// the single fixed-order g=0..cg-1 combine. Existing GPART storage owns every partial.
inline bool gmma_g_admits(int cg,const float*v,int ldv,const float*x,int ldx,float*g,int ldg,
                         int rows,int h,int q,float*part,size_t part_words){
 if(cg<1||cg>128||rows<32*cg||h<1||h>128||q<1||q>128||!gmma_enabled<float>()||
    !gmma_aligned(v,ldv)||!gmma_aligned(x,ldx)||!gmma_aligned(g,ldg)||
    (cg>1&&(!part||part_words<size_t(cg)*ldg*q||!gmma_aligned(part,(long long)ldg*q))))return false;
 const long long sp=(long long)ldg*q;const int ks=gmma_w_slice(rows,cg),rem=rows-cg*ks;
 if(gmma_x3<float>()){
  if(gmma_detail::GemmW3::can_implement(gmma_w3_args(2*cg,v,ldv,x,ldx,g,ldg,rows,h,q,nullptr,6,part,sp))!=cutlass::Status::kSuccess)return false;
  return !rem||gmma_detail::GemmW3::can_implement(gmma_w3_args(2,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,part+size_t(cg-1)*sp,ldg,rem,h,q,nullptr,6,nullptr,0,1.f))==cutlass::Status::kSuccess;
 }
 if(gmma_detail::GemmD::can_implement(gmma_w_args(2*cg,v,ldv,x,ldx,g,ldg,rows,h,q,nullptr,6,part,sp))!=cutlass::Status::kSuccess)return false;
 if(!rem)return true;auto tail=gmma_w_args(2,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,part+size_t(cg-1)*sp,ldg,rem,h,q,nullptr,6,nullptr,0);tail.epilogue.thread.beta=1.f;
 return gmma_detail::GemmD::can_implement(tail)==cutlass::Status::kSuccess;
}
inline void launch_gmma_g(int cg,const float*v,int ldv,const float*x,int ldx,float*g,int ldg,
                          int rows,int h,int q,float*part,cudaStream_t st,Witness*wit){
 const long long sp=(long long)ldg*q;const int ks=gmma_w_slice(rows,cg),rem=rows-cg*ks;
 if(gmma_x3<float>()){
  gmma_detail::GemmW3 gemm;auto a=gmma_w3_args(2*cg,v,ldv,x,ldx,g,ldg,rows,h,q,wit,6,part,sp);gmma_detail::run(gemm,a,st);
  if(cg>1&&rem){auto tail=gmma_w3_args(2,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,part+size_t(cg-1)*sp,ldg,rem,h,q,nullptr,6,nullptr,0,1.f);
   gmma_detail::GemmW3 gt;gmma_detail::run(gt,tail,st);}
 }else{
  gmma_detail::GemmD gemm;auto a=gmma_w_args(2*cg,v,ldv,x,ldx,g,ldg,rows,h,q,wit,6,part,sp);gmma_detail::run(gemm,a,st);
  if(cg>1&&rem){auto tail=gmma_w_args(2,v+size_t(cg)*ks,ldv,x+size_t(cg)*ks,ldx,part+size_t(cg-1)*sp,ldg,rem,h,q,nullptr,6,nullptr,0);tail.epilogue.thread.beta=1.f;
   gmma_detail::GemmD gt;gmma_detail::run(gt,tail,st);}
 }
 if(cg>1){carrier_g_combine<float><<<std::min(1024,ceildiv(h*q,256)),256,0,st>>>(part,sp,cg,g,ldg,0,q,h,1,false);CU(cudaGetLastError());}
 ++gmma_detail::gmma_g_launches();
}
}
