#pragma once
// Single-SM and two-SM register panels for the GE, TT and TS factors.
// Ports of the single-SM and two-SM register panels of gau.nernst's GPU MODE QR-v2 submission 844219 into the 2.5D engine's GE / TT / TS factors.
//
// Kept exactly: a warp OWNS VEC consecutive columns and holds every row of them in registers; reflectors are
// published through shared memory behind one mbarrier per reflector (32 arrivals, release / acquire.cta); a warp
// consumes every reflector to its left in order (pairs of columns per reduction, as in the original's fma_f32x2 pairs) and then
// generates its own; the two-SM variant splits the columns over the two CTAs of a cluster and rank 0 broadcasts each
// reflector and its tau to rank 1 with cp.async.bulk shared::cluster + an expect_tx mbarrier. Changed, for accuracy
// and range on arbitrary inputs (the original benchmark did not need them):
// * column-major tile packets of the engine instead of row-major matrices, fp32 AND fp64;
// * sum of squares in fp64, beta / tau / v formed in fp64 from the stored values (LAPACK dlarfg); fp64 storage
//   renormalises by the column max when the sum of squares would underflow;
// * IEEE sqrt/division instead of sqrt.approx / rcp.approx; scalar FMA instead of B200 packed f32x2; no fp16 V.
// V (unit diagonal, zeros above) goes to the engine's packed V layout, tau to a per-packet vector; T is built from the reflector overlaps
// (see gau_fast.cuh).
#include "gau_ge.cuh"
namespace tqr {

__device__ __forceinline__ unsigned gau_smem_addr(const void*p){return unsigned(__cvta_generic_to_shared(p));}
__device__ __forceinline__ void gau_mbar_init(unsigned a,int count){asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(a),"r"(count));}
__device__ __forceinline__ void gau_mbar_arrive(unsigned a){asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"::"r"(a):"memory");}
__device__ __forceinline__ void gau_mbar_wait(unsigned a,int phase){
 asm volatile("{\n\t.reg .pred ready;\n\tGAU_WAIT_%=:\n\t"
              "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready, [%0], %1, %2;\n\t"
              "@!ready bra.uni GAU_WAIT_%=;\n\t}"::"r"(a),"r"(phase),"r"(0x989680):"memory");}
__device__ __forceinline__ void gau_mbar_wait_cluster(unsigned a,int phase){
 asm volatile("{\n\t.reg .pred ready;\n\tGAU_WAITC_%=:\n\t"
              "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [%0], %1, %2;\n\t"
              "@!ready bra.uni GAU_WAITC_%=;\n\t}"::"r"(a),"r"(phase),"r"(0x989680):"memory");}

// One Householder reflector from column i of the warp's registers (rows r = it*32 + lane), in fp64 from the stored
// values (dlarfg). Returns tau; v (unit at the pivot, zeros above) replaces the column below the pivot, beta sits on it.
template<class T,int ITEMS>
__device__ __forceinline__ T gau_warp_reflector(T(&col)[ITEMS],int c,int rows,int lane,T(&v)[ITEMS],int*status){
 double tl=0,mx=0;T xc=T(0);
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;const double x=double(col[it]);
  if(r>c&&r<rows){tl+=x*x;if constexpr(sizeof(T)==8)mx=fmax(mx,fabs(x));}
  if(r==c)xc=col[it];}
 tl=gau_warp_sum(tl);
 double x0=double(__shfl_sync(0xffffffffu,xc,c&31)),tail=tl,scl=1.0;bool has=tail>0.0;
 if constexpr(sizeof(T)==8){
  #pragma unroll
  for(int o=16;o;o>>=1)mx=fmax(mx,__shfl_xor_sync(0xffffffffu,mx,o));
  has=mx>0.0;
  if(has&&tail<1e-280){const double m=fmax(mx,fabs(x0));double t2=0;
   #pragma unroll
   for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>c&&r<rows){const double y=double(col[it])/m;t2+=y*y;}}
   tail=gau_warp_sum(t2);x0/=m;scl=m;}}
 double beta=x0,taud=0,inv=0;
 if(has){const double nrm=sqrt(x0*x0+tail);beta=x0<0?nrm:-nrm;taud=(beta-x0)/beta;inv=1.0/(x0-beta);}
 beta*=scl;
 if(lane==0&&!isfinite(T(beta)))atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;const T x=col[it];
  v[it]=r==c?T(1):(has&&r>c&&r<rows)?T((scl==1.0?double(x):double(x)/scl)*inv):T(0);
  if(r==c)col[it]=T(beta);else if(r>c)col[it]=has?v[it]:x;}
 return T(taud);
}
// Apply one published reflector (v from shared, tau) to the warp's VEC columns: pairs of columns per reduction pass.
template<class T,int ITEMS,int VEC>
__device__ __forceinline__ void gau_warp_apply(T(&col)[ITEMS][VEC],const T*vs,T tk,int lane){
 T v[ITEMS];
 #pragma unroll
 for(int it=0;it<ITEMS;++it)v[it]=vs[it*32+lane];
 #pragma unroll
 for(int pr=0;pr<VEC;pr+=2){T d0=T(0),d1=T(0);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){d0+=col[it][pr]*v[it];if(pr+1<VEC)d1+=col[it][pr+1]*v[it];}
  d0=gau_warp_sum(d0)*tk;if(pr+1<VEC)d1=gau_warp_sum(d1)*tk;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){col[it][pr]-=v[it]*d0;if(pr+1<VEC)col[it][pr+1]-=v[it]*d1;}}
}

// ---------------------------------------------------------------- single SM ----------------------------------------
// gau.nernst register_panel_kernel: one CTA per packet, WARPS warps x VEC columns, all rows (ROWS = ITEMS*32) in
// registers, reflectors [COLS x ROWS] + taus in shared memory, mbarrier per reflector.
template<class T,int ITEMS,int VEC,int WARPS>
__global__ __launch_bounds__(WARPS*32,1) void gau_sm1_kernel(T*A,int lda,const TilePacket*packets,T*Vout,int ldv,size_t vstride,
                                                            T*tau_out,int fstride,int kind,int*status,Witness*w){
 constexpr int ROWS=ITEMS*32,COLS=VEC*WARPS;
 extern __shared__ __align__(16) unsigned char gau_smem[];
 T*refl=reinterpret_cast<T*>(gau_smem);                       // [COLS][ROWS]
 T*taus=refl+size_t(COLS)*ROWS;                                // [COLS]
 unsigned long long*mb=reinterpret_cast<unsigned long long*>(taus+COLS+(COLS&1));
 const int id=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 const TilePacket pk=packets[id];const int rows=pk.rows,h=pk.h,c0=warp*VEC;
 T*a=A+pk.row+size_t(pk.col)*lda;
 if(warp==0&&lane==0){for(int i=0;i<COLS;++i)gau_mbar_init(gau_smem_addr(mb+i),32);asm volatile("fence.mbarrier_init.release.cluster;");}
 __syncthreads();
 T col[ITEMS][VEC];
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int j=0;j<VEC;++j)col[it][j]=(r<rows&&c0+j<h)?a[r+size_t(c0+j)*lda]:T(0);}
 // consume every reflector of the warps to the left, in order
 const int kend=min(c0,h);
 for(int k=0;k<kend;++k){gau_mbar_wait(gau_smem_addr(mb+k),0);gau_warp_apply<T,ITEMS,VEC>(col,refl+size_t(k)*ROWS,taus[k],lane);}
 // generate and publish this warp's reflectors
 #pragma unroll
 for(int i=0;i<VEC;++i){
  const int c=c0+i;if(c>=h)break;
  T ci[ITEMS],v[ITEMS];
  #pragma unroll
  for(int it=0;it<ITEMS;++it)ci[it]=col[it][i];
  const T tk=gau_warp_reflector<T,ITEMS>(ci,c,rows,lane,v,status);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){col[it][i]=ci[it];refl[size_t(c)*ROWS+it*32+lane]=v[it];}
  if(lane==0)taus[c]=tk;
  gau_mbar_arrive(gau_smem_addr(mb+c));
  #pragma unroll
  for(int j=i+1;j<VEC;++j){T d=T(0);
   #pragma unroll
   for(int it=0;it<ITEMS;++it)d+=col[it][j]*v[it];
   d=gau_warp_sum(d)*tk;
   #pragma unroll
   for(int it=0;it<ITEMS;++it)col[it][j]-=v[it]*d;}
 }
 // commit: R above, beta on, v below the diagonal; V (unit diagonal, zeros above) and tau for the T builder
 bool bad=false;
 if(c0<h){
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>=rows)continue;
   #pragma unroll
   for(int j=0;j<VEC;++j)if(c0+j<h){const T y=col[it][j];bad|=!isfinite(y);a[r+size_t(c0+j)*lda]=y;}}}
 if(bad)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 __syncthreads();
 T*vo=Vout+size_t(id)*vstride;
 for(int e=tid;e<ldv*h;e+=blockDim.x){const int r=e%ldv,c=e/ldv;vo[e%ldv+size_t(c)*ldv]=r<ROWS?refl[size_t(c)*ROWS+r]:T(0);}
 for(int c=tid;c<h;c+=blockDim.x)tau_out[size_t(id)*fstride+c]=taus[c];
 if(tid==0){atomicAdd(kind==2?&w->tt:kind==1?&w->ts:&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);
  atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}

// ---------------------------------------------------------------- two SMs ------------------------------------------
// gau.nernst register_2sm_panel_kernel: a cluster of 2 CTAs per packet; CTA rank r owns columns [r*COLS/2, (r+1)*COLS/2)
// (WARPS warps x VEC). Rank 0's reflectors are copied into rank 1's shared memory by cp.async.bulk (plus tau by
// st.async) completing an expect_tx mbarrier; rank 1 consumes the whole remote prefix, then its local chain.
template<class T,int ITEMS,int VEC,int WARPS>
__global__ __cluster_dims__(2,1,1) __launch_bounds__(WARPS*32,1) void gau_sm2_kernel(T*A,int lda,const TilePacket*packets,T*Vout,int ldv,size_t vstride,
                                                                                    T*tau_out,int fstride,int kind,int*status,Witness*w){
 constexpr int ROWS=ITEMS*32,LOCAL=VEC*WARPS,COLS=2*LOCAL;
 extern __shared__ __align__(16) unsigned char gau_smem[];
 T*refl=reinterpret_cast<T*>(gau_smem);                       // [COLS][ROWS]: rank 1 also keeps rank 0's reflectors
 T*taus=refl+size_t(COLS)*ROWS;                                // [COLS]
 unsigned long long*mb=reinterpret_cast<unsigned long long*>(taus+COLS+(COLS&1));
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
 unsigned rank;asm volatile("mov.u32 %0, %%cluster_ctarank;":"=r"(rank));
 const int id=blockIdx.x>>1;
 const TilePacket pk=packets[id];const int rows=pk.rows,h=pk.h,c0=int(rank)*LOCAL+warp*VEC;
 T*a=A+pk.row+size_t(pk.col)*lda;
 if(warp==0&&lane==0){for(int i=0;i<COLS;++i)gau_mbar_init(gau_smem_addr(mb+i),rank==0||i>=LOCAL?32:1);
  asm volatile("fence.mbarrier_init.release.cluster;");}
 asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;":::"memory");
 // rank 1: arm the remote mbarriers for rank 0's reflectors (ROWS values + tau per reflector)
 if(rank==1&&warp==0&&lane==0){for(int i=0;i<min(LOCAL,h);++i)
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"::"r"(gau_smem_addr(mb+i)),"r"(int((ROWS+1)*sizeof(T))):"memory");}
 T col[ITEMS][VEC];
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int j=0;j<VEC;++j)col[it][j]=(r<rows&&c0+j<h)?a[r+size_t(c0+j)*lda]:T(0);}
 const int kend=min(c0,h);
 for(int k=0;k<kend;++k){
  if(rank==1&&k<LOCAL)gau_mbar_wait_cluster(gau_smem_addr(mb+k),0);else gau_mbar_wait(gau_smem_addr(mb+k),0);
  gau_warp_apply<T,ITEMS,VEC>(col,refl+size_t(k)*ROWS,taus[k],lane);}
 #pragma unroll
 for(int i=0;i<VEC;++i){
  const int c=c0+i;if(c>=h)break;
  T ci[ITEMS],v[ITEMS];
  #pragma unroll
  for(int it=0;it<ITEMS;++it)ci[it]=col[it][i];
  const T tk=gau_warp_reflector<T,ITEMS>(ci,c,rows,lane,v,status);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){col[it][i]=ci[it];refl[size_t(c)*ROWS+it*32+lane]=v[it];}
  if(lane==0)taus[c]=tk;
  __syncwarp();
  if(rank==0){
   // make this reflector visible to the async proxy, then broadcast it (and tau) into rank 1's shared memory
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   __syncwarp();
   if(lane==0){
    unsigned src=gau_smem_addr(refl+size_t(c)*ROWS),dst,remb,dtau;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(dst):"r"(src));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(remb):"r"(gau_smem_addr(mb+c)));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(dtau):"r"(gau_smem_addr(taus+c)));
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst),"r"(src),"r"(int(ROWS*sizeof(T))),"r"(remb):"memory");
    if constexpr(sizeof(T)==4)asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"::"r"(dtau),"r"(__float_as_uint(float(tk))),"r"(remb):"memory");
    else asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"::"r"(dtau),"l"(__double_as_longlong(double(tk))),"r"(remb):"memory");}
  }
  gau_mbar_arrive(gau_smem_addr(mb+c));
  #pragma unroll
  for(int j=i+1;j<VEC;++j){T d=T(0);
   #pragma unroll
   for(int it=0;it<ITEMS;++it)d+=col[it][j]*v[it];
   d=gau_warp_sum(d)*tk;
   #pragma unroll
   for(int it=0;it<ITEMS;++it)col[it][j]-=v[it]*d;}
 }
 bool bad=false;
 if(c0<h){
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>=rows)continue;
   #pragma unroll
   for(int j=0;j<VEC;++j)if(c0+j<h){const T y=col[it][j];bad|=!isfinite(y);a[r+size_t(c0+j)*lda]=y;}}}
 if(bad)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 __syncthreads();
 // each rank writes the V columns and taus it produced
 T*vo=Vout+size_t(id)*vstride;const int lb=int(rank)*LOCAL,le=min(h,lb+LOCAL);
 for(int e=tid;e<ldv*(le-lb);e+=blockDim.x){const int r=e%ldv,c=lb+e/ldv;vo[r+size_t(c)*ldv]=r<ROWS?refl[size_t(c)*ROWS+r]:T(0);}
 for(int c=lb+tid;c<le;c+=blockDim.x)tau_out[size_t(id)*fstride+c]=taus[c];
 if(rank==0&&tid==0){atomicAdd(kind==2?&w->tt:kind==1?&w->ts:&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);}
 if(tid==0){atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
 // no CTA may exit while its peer can still write into its shared memory
 asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;":::"memory");
}

// ---------------------------------------------------------------- dispatch -----------------------------------------
struct GauSmConfig{int sms=0,items=0,vec=0,warps=0;size_t smem=0;};
template<class T> inline size_t gau_sm_smem(int items,int cols){return (size_t(cols)*items*32+cols+2)*sizeof(T)+size_t(cols)*8+16;}
inline int gau_sm_items(int rows){return rows<=128?4:rows<=256?8:rows<=512?16:rows<=1024?32:0;}
// sms = 1: all h columns on one SM; sms = 2: a cluster of two CTAs, h/2 columns each. The owned columns must fit the
// register budget of the CTA size and the reflectors the shared capacity.
template<class T> inline GauSmConfig gau_sm_config(int sms,int rows,int h){
 GauSmConfig g;const int items=gau_sm_items(rows);if(!items)return g;const int wpt=int(sizeof(T)/4);
 for(int vec:{8,4}){
  const int local=sms==1?h:(h+1)/2,warps=(local+vec-1)/vec;if(warps<1||warps>16)continue;
  const int threads=warps*32,budget=std::min(255,65536/threads)-24;
  if(items*vec*wpt>budget)continue;
  const size_t smem=gau_sm_smem<T>(items,sms==1?warps*vec:2*warps*vec);
  if(smem>size_t(227)*1024)continue;
  g.sms=sms;g.items=items;g.vec=vec;g.warps=warps;g.smem=smem;return g;}
 return g;}
template<class T,int ITEMS,int VEC,int WARPS,int SMS>
inline void gau_sm_launch_one(const GauSmConfig&g,T*A,int lda,const TilePacket*pk,int count,T*V,int ldv,size_t vstride,T*tau,int fstride,int kind,int*status,Witness*w,cudaStream_t st,bool dry){
 auto kernel=SMS==1?gau_sm1_kernel<T,ITEMS,VEC,WARPS>:gau_sm2_kernel<T,ITEMS,VEC,WARPS>;
 static bool ok=false,spill=false;static size_t cap=0,set=0;
 if(!ok){cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,kernel));spill=attr.localSizeBytes>0;
  int device,optin=0;CU(cudaGetDevice(&device));CU(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
  cap=size_t(optin)-attr.sharedSizeBytes;ok=true;}
 if(spill)throw std::runtime_error("gau_sm_register_spill_before_modify");
 if(g.smem>cap)throw std::runtime_error("gau_sm_shared_capacity_before_modify");
 if(dry)return;
 if(g.smem>set){CU(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(g.smem)));set=g.smem;}
 kernel<<<count*SMS,WARPS*32,g.smem,st>>>(A,lda,pk,V,ldv,vstride,tau,fstride,kind,status,w);CU(cudaGetLastError());
}
// Returns the executed config (sms = 0: this shape has no single/two-SM register panel; the caller uses the gmem one).
template<class T>
inline GauSmConfig launch_gau_sm(int sms,T*A,int lda,const TilePacket*pk,int count,int rows,int h,T*V,int ldv,size_t vstride,T*tau,int fstride,int kind,int*status,Witness*w,cudaStream_t st,bool dry){
 const GauSmConfig g=gau_sm_config<T>(sms,rows,h);
 if(!g.sms)throw std::runtime_error("gau_sm_shape_before_modify");
#define TQR_GAU_SM(S,I,Vv,W) if(g.sms==S&&g.items==I&&g.vec==Vv&&g.warps==W){gau_sm_launch_one<T,I,Vv,W,S>(g,A,lda,pk,count,V,ldv,vstride,tau,fstride,kind,status,w,st,dry);return g;}
#define TQR_GAU_SM_W(S,I,Vv) TQR_GAU_SM(S,I,Vv,2)TQR_GAU_SM(S,I,Vv,4)TQR_GAU_SM(S,I,Vv,6)TQR_GAU_SM(S,I,Vv,8)TQR_GAU_SM(S,I,Vv,12)TQR_GAU_SM(S,I,Vv,16)
 TQR_GAU_SM_W(1,4,8)TQR_GAU_SM_W(1,8,8)TQR_GAU_SM_W(1,16,8)TQR_GAU_SM_W(1,16,4)TQR_GAU_SM_W(1,32,4)
 TQR_GAU_SM_W(2,4,8)TQR_GAU_SM_W(2,8,8)TQR_GAU_SM_W(2,16,8)TQR_GAU_SM_W(2,16,4)TQR_GAU_SM_W(2,32,4)
#undef TQR_GAU_SM_W
#undef TQR_GAU_SM
 throw std::runtime_error("gau_sm_config_not_instantiated_before_modify");
}
}
