#pragma once
// Owner-chain panel kernel for the GE factor.
// Adapted from gau.nernst's GPU MODE QR-v2 submission 844219 (`gmem_panel_kernel`). Its ownership and progress rule is kept exactly: a CTA
// OWNS C consecutive panel columns and holds every row of them in registers; it first CONSUMES, in order, every
// reflector published by the owners to its left (x <- x - v tau v^T x), then GENERATES its own C reflectors one by
// one, publishing each (v column + tau) to global memory behind a release flag that the owners to its right acquire.
// There is no grid-wide barrier in the reflector chain: the panel runs as a wavefront, so owner o applies reflector k
// while owner k/C is already producing k+1.
//
// Paper mapping. The GE of one tile packet is still one Householder elimination of the ordered list, the same
// reflectors in the same order. Its reductions (column norm, v^T a_j) cut [K] = rows over the CTA's threads and
// combine additively over those disjoint row sets (block-level carrier, c = THREADS); the GPU level splits the OUTPUT
// columns J over the owners (p_j = ceil(h/C)) with the reflector operand REPLICATED to every later owner (Replicate
// move). Owners commit their own columns exactly once. Nothing here is Gram-Schmidt or Cholesky QR: every column is
// transformed only by Householder reflections, and the T builder below uses overlaps v_i^T v_j of ALREADY-FORMED
// reflectors (the same quantity LAPACK dlarft forms), never A^T A.
//
// * column-major tile packets of the engine instead of row-major matrices;
// * sum of squares accumulated in fp64 for fp32 storage (fp32^2 cannot under/overflow fp64) and combined in fp64;
//   IEEE sqrt/division, not sqrt.approx/rcp.approx; no fp16 V; no B200 packed f32x2 FMA;
// * the reflector V is written straight into the engine's packed V buffer (unit diagonal, zeros above), so the
//   separate pack_ge_v pass is not needed;
// * T is built after the chain exactly as LAPACK dlarft does inside dgeqrf: from the reflector overlaps
//   O = V^T V (V unit lower triangular, |v_ij| <= 1, so O is bounded independently of A and of kappa(A)) and tau,
//   by the dlarft recurrence on 8-column blocks and Eq compose (T12 = -T1 O12 T2) above them; division free,
//   correct for tau = 0. This is NOT a Gram matrix of A: A^T A is never formed, nothing is Cholesky-factored,
//   and the factorization is backward stable with no dependence on the conditioning of A.
#include "tiled_kernels.cuh"
#include "gau_math.cuh"
#include <cooperative_groups.h>
#include <array>
#include <vector>
namespace tqr {

__device__ __forceinline__ void gau_store_release(int*p,int v){asm volatile("st.release.gpu.global.b32 [%0], %1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ int gau_load_acquire(const int*p){int v;asm volatile("ld.acquire.gpu.global.b32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;}
#ifdef GAU_DEBUG_TIMELINE
__device__ unsigned long long*gau_dbg=nullptr;   // [owner][4]: start, consumed, generated, committed (packet 0)
#define GAU_TL(slot) do{if(gau_dbg&&id==0&&tid==0)gau_dbg[owner*4+(slot)]=clock64();}while(0)
#else
#define GAU_TL(slot) do{}while(0)
#endif
template<class A> __device__ __forceinline__ A gau_warp_sum(A v){
 #pragma unroll
 for(int o=16;o;o>>=1)v+=__shfl_xor_sync(0xffffffffu,v,o);
 return v;}

// Block all-reduce of C per-thread partials over the CTA's WARPS warps: recursive halving inside each warp (lane L ends
// with column L >> (5 - log2 C)), one shared-memory hop, then every warp sums the WARPS partials of each column with a
// short shuffle tree and broadcasts the C sums to all lanes. Fixed summation order; 2 barrier-free stages instead of C
// independent 5-level butterflies per stage. part: [C][WARPS] (callers double-buffer it across barriers).
template<class T,int C,int WARPS>
__device__ __forceinline__ void gau_block_allreduce(T(&d)[C],T*part,int lane,int warp){
 constexpr int SH=5-GfLog<C>::v,G=32/C;            // lanes per column group in stage 2
 const T s=gf_reduce_scatter<T,C>(d,lane);
 if((lane&((1<<SH)-1))==0)part[(lane>>SH)*WARPS+warp]=s;
 __syncthreads();
 const int q=lane%C;T t=T(0);
 #pragma unroll
 for(int wv=lane/C;wv<WARPS;wv+=G)t+=part[q*WARPS+wv];
 #pragma unroll
 for(int o=C;o<32;o<<=1)t+=__shfl_xor_sync(0xffffffffu,t,o);
 #pragma unroll
 for(int j=0;j<C;++j)d[j]=__shfl_sync(0xffffffffu,t,j);
}
// One launch = a batch of independent tile packets (GE leaves of a panel, or the TT/TS stacks of one tree level): packet
// id = blockIdx.x / owners gets its own owners, V column block (V + id*vstride, leading dimension ldv), tau and flag words
// (+ id*fstride). flags[k] == epoch  <=>  column k of that packet's V and tau[k] are published for this launch; the host
// bumps epoch per launch, so flag words are never reset. kind: 0 GE, 1 TS, 2 TT (witness only; the algebra is the same
// ordered Householder chain on the packet as stored -- structural zeros of a TT/TS stack stay exact zeros).
template<class T,int C,int ITEMS,int THREADS>
__global__ __launch_bounds__(THREADS,1) void gau_ge_kernel(T*A,int lda,const TilePacket*packets,int owners,T*V,int ldv,size_t vstride,
                                                          T*tau,int*flags,int fstride,int epoch,int kind,int*status,Witness*w){
 constexpr int WARPS=THREADS/32;
 // Keep the consumed reflector in registers only while it fits beside the owned columns; otherwise the update
 // re-reads it (an L1/L2 hit: the same CTA read it one reduction earlier).
 constexpr bool KEEP=ITEMS<=16;
 const int id=blockIdx.x/owners,owner=blockIdx.x-id*owners;
 const TilePacket pk=packets[id];
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rows=pk.rows,h=pk.h,c0=owner*C;
 if(c0>=h)return;                                   // idle owner of a narrower packet (CTA-uniform)
 V+=size_t(id)*vstride;tau+=size_t(id)*fstride;flags+=size_t(id)*fstride;
 T*a=A+pk.row+size_t(pk.col)*lda;
 __shared__ T part[2][C*WARPS];
 __shared__ T tailt[2][WARPS];
 __shared__ int nzw[2][WARPS];
 __shared__ T diag[2];
 __shared__ double dred[WARPS];
 T col[ITEMS][C];
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;
  #pragma unroll
  for(int j=0;j<C;++j)col[it][j]=(r<rows&&c0+j<h)?a[r+size_t(c0+j)*lda]:T(0);}
 int par=0;
 GAU_TL(0);
 // ---- consume every reflector to the left, in order ----
 for(int k=0;k<c0;++k){
  if(tid==0){while(gau_load_acquire(flags+k)!=epoch)__nanosleep(20);}
  __syncthreads();
  const T tk=tau[k];const T*vk=V+size_t(k)*ldv;
  T d[C];
  #pragma unroll
  for(int j=0;j<C;++j)d[j]=T(0);
  T vr[KEEP?ITEMS:1];
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;const T v=(r<rows&&r>=k)?vk[r]:T(0);if constexpr(KEEP)vr[it]=v;
   #pragma unroll
   for(int j=0;j<C;++j)d[j]+=v*col[it][j];}
  gau_block_allreduce<T,C,WARPS>(d,part[par],lane,warp);
  #pragma unroll
  for(int j=0;j<C;++j)d[j]*=tk;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;T v;if constexpr(KEEP)v=vr[it];else v=(r<rows&&r>=k)?vk[r]:T(0);
   #pragma unroll
   for(int j=0;j<C;++j)col[it][j]-=v*d[j];}
  par^=1;
 }
 GAU_TL(1);
 // ---- generate and publish this owner's reflectors ----
 #pragma unroll
 for(int i=0;i<C;++i){
  const int c=c0+i;if(c>=h)break;
  // dlarfg over the block: range-checked fast path in the working precision (IEEE-grade rsqrt/rcp with one Newton
  // step), LAPACK-scaled path otherwise (fp64-accumulated for fp32 storage, power-of-two scaled for fp64). Every
  // quantity below is block-uniform, so the branches are uniform.
  T tl=T(0);unsigned nz=0;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;const T y=col[it][i];
   if(r>c&&r<rows){tl=gf_fma(y,y,tl);nz|=gf_nz(y);}
   if(r==c)diag[par]=y;}
  tl=gau_warp_sum(tl);{const bool wz=__any_sync(0xffffffffu,nz!=0u);if(lane==0){tailt[par][warp]=tl;nzw[par][warp]=wz;}}
  __syncthreads();
  const T tail=gau_warp_sum(lane<WARPS?tailt[par][lane]:T(0));
  const bool has=__any_sync(0xffffffffu,lane<WARPS&&nzw[par][lane]!=0);
  const T x0=diag[par];
  T beta=x0,tk=T(0),inv=T(0);int e=0;bool slow=false;double invd=0;
  if(has){
   if(tail>=GfRange<T>::lo&&tail<=GfRange<T>::hi&&fabs(x0)<=GfRange<T>::xmax){
    const T q=gf_fma(x0,x0,tail),y=gf_rsqrt(q),nrm=q*y,sa=fabs(x0)+nrm,rs=gf_rcp(sa);
    beta=x0<T(0)?nrm:-nrm;tk=sa*y;inv=x0<T(0)?-rs:rs;
   }else{
    slow=true;
    if constexpr(sizeof(T)==8){
     double m=fabs(double(x0));
     #pragma unroll
     for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;if(r>c&&r<rows)m=fmax(m,fabs(double(col[it][i])));}
     #pragma unroll
     for(int o=16;o;o>>=1)m=fmax(m,__shfl_xor_sync(0xffffffffu,m,o));
     if(lane==0)dred[warp]=m;__syncthreads();
     m=lane<WARPS?dred[lane]:0.0;
     #pragma unroll
     for(int o=16;o;o>>=1)m=fmax(m,__shfl_xor_sync(0xffffffffu,m,o));
     __syncthreads();frexp(m,&e);}
    double t2=0;
    #pragma unroll
    for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;if(r>c&&r<rows){const double y=scalbn(double(col[it][i]),-e);t2=fma(y,y,t2);}}
    t2=gau_warp_sum(t2);if(lane==0)dred[warp]=t2;__syncthreads();
    t2=gau_warp_sum(lane<WARPS?dred[lane]:0.0);__syncthreads();
    const double a=scalbn(double(x0),-e),nrm=sqrt(fma(a,a,t2)),bs=a<0?nrm:-nrm;
    beta=T(scalbn(bs,e));tk=T((bs-a)/bs);invd=1.0/(a-bs);}
  }
  if(tid==0){tau[c]=tk;if(!isfinite(beta))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;const T x=col[it][i];
   const T v=r==c?T(1):(has&&r>c&&r<rows)?(slow?T(scalbn(double(x),-e)*invd):x*inv):T(0);
   if(r==c)col[it][i]=beta;else if(r>c)col[it][i]=has?v:x;
   if(r<ldv)V[r+size_t(c)*ldv]=r<rows?v:T(0);}
  for(int r=ITEMS*THREADS+tid;r<ldv;r+=THREADS)V[r+size_t(c)*ldv]=T(0);
  // Every row writer of V column c and tau[c] precedes this barrier; the release below is cumulative over it.
  T d[C];
  #pragma unroll
  for(int j=0;j<C;++j)d[j]=T(0);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;const T v=r==c?T(1):(r>c&&r<rows)?col[it][i]:T(0);
   #pragma unroll
   for(int j=0;j<C;++j)if(j>i)d[j]+=v*col[it][j];}
  if(i+1<C){
   const T s=gf_reduce_scatter<T,C>(d,lane);
   constexpr int SH=5-GfLog<C>::v;
   if((lane&((1<<SH)-1))==0)part[par][(lane>>SH)*WARPS+warp]=s;}
  __syncthreads();
  if(tid==0)gau_store_release(flags+c,epoch);
  if(i+1<C){
   constexpr int G=32/C;const int q=lane%C;T t=T(0);
   #pragma unroll
   for(int wv=lane/C;wv<WARPS;wv+=G)t+=part[par][q*WARPS+wv];
   #pragma unroll
   for(int o=C;o<32;o<<=1)t+=__shfl_xor_sync(0xffffffffu,t,o);
   #pragma unroll
   for(int j=0;j<C;++j)d[j]=__shfl_sync(0xffffffffu,t,j)*tk;
   #pragma unroll
   for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;const T v=r==c?T(1):(r>c&&r<rows)?col[it][i]:T(0);
    #pragma unroll
    for(int j=0;j<C;++j)if(j>i)col[it][j]-=v*d[j];}}
  par^=1;
 }
 GAU_TL(2);
 // ---- the owner commits its columns once: R above, beta on, v below the diagonal ----
 bool bad=false;
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*THREADS+tid;if(r>=rows)continue;
  #pragma unroll
  for(int j=0;j<C;++j)if(c0+j<h){const T y=col[it][j];bad|=!isfinite(y);a[r+size_t(c0+j)*lda]=y;}}
 if(bad)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 if(tid==0){
  const int wk=kind?1:0;
  if(owner==0){atomicAdd(kind==2?&w->tt:kind==1?&w->ts:&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);
   atomicAdd(&w->device_qrv2_products[wk],1ULL);witness_levels(w,wk,THREADS,1,1,1);}
  atomicAdd(&w->device_qrv2_owners[wk],1ULL);
  atomicAdd(&w->device_block_peers[wk],(unsigned long long)THREADS);
  atomicAdd(&w->device_peer_partials[wk],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}

// T of Q = I - V T V^T for the h reflectors, from the reflector overlaps O = V^T V (upper part used; argument G) and
// tau, as in LAPACK dlarft. O is bounded by the dimensions alone (unit lower triangular V, |v_ij| <= 1).
// Level 0: every 8-column block by the dlarft recurrence T[r,c] = -tau_c sum_{s=r}^{c-1} T[r,s] G[s,c] (one warp per
// block). Level L: Eq compose T12 = -T11 (G12 T22) for block pairs of width s = 8*2^L. One CTA, h <= 128.
// ---- reflector overlaps O = V^T V (upper triangle incl. diagonal), exact T arithmetic in every precision mode ----
// Split move over [K] = rows: slice z of S owns rows [floor(rows z/S), floor(rows (z+1)/S)) (disjoint), computes its
// partial P_z with plain FMA, and gau_overlap_combine sums P_0..P_{S-1} in that fixed order and commits O once
// (Combine: additive over disjoint K_z). O decides the orthogonality of Q through T, so it is never computed in TF32.
constexpr int GAU_OV_CHUNK=32;
template<class T>
__global__ __launch_bounds__(256) void gau_overlap_partial(const T*V,int ldv,int rows,int h,int S,T*P){
 extern __shared__ __align__(16) unsigned char gau_smem[];
 T*Vs=reinterpret_cast<T*>(gau_smem);            // GAU_OV_CHUNK x h, row-major (ld h+1) so a row is contiguous
 const int z=blockIdx.x,r0=int((long long)rows*z/S),r1=int((long long)rows*(z+1)/S),tid=threadIdx.x,ld=h+1;
 const int nt4=(h+3)/4;                           // 4x4 tiles over the h x h output, upper ones only
 constexpr int U=3;                               // 528 upper tiles at h = 128 over 256 threads
 T acc[U][4][4];
 int ti[U],tj[U];
 #pragma unroll
 for(int u=0;u<U;++u){const int t=tid+u*256;ti[u]=-1;tj[u]=-1;
  // enumerate upper tiles (bi <= bj) in row-major order of the tile triangle
  int k=t;for(int bi=0;bi<nt4;++bi){const int n=nt4-bi;if(k<n){ti[u]=bi;tj[u]=bi+k;break;}k-=n;}
  #pragma unroll
  for(int x=0;x<4;++x)
   #pragma unroll
   for(int y=0;y<4;++y)acc[u][x][y]=T(0);}
 for(int c0=r0;c0<r1;c0+=GAU_OV_CHUNK){
  const int cn=min(GAU_OV_CHUNK,r1-c0);
  __syncthreads();
  for(int e=tid;e<cn*h;e+=256){const int r=e%cn,c=e/cn;Vs[r*ld+c]=V[(c0+r)+size_t(c)*ldv];}
  __syncthreads();
  #pragma unroll
  for(int u=0;u<U;++u)if(ti[u]>=0){const int i0=ti[u]*4,j0=tj[u]*4;
   for(int r=0;r<cn;++r){T x[4],y[4];
    #pragma unroll
    for(int q=0;q<4;++q){x[q]=i0+q<h?Vs[r*ld+i0+q]:T(0);y[q]=j0+q<h?Vs[r*ld+j0+q]:T(0);}
    #pragma unroll
    for(int q=0;q<4;++q)
     #pragma unroll
     for(int w=0;w<4;++w)acc[u][q][w]+=x[q]*y[w];}}
 }
 T*Pz=P+size_t(z)*h*h;
 #pragma unroll
 for(int u=0;u<U;++u)if(ti[u]>=0){const int i0=ti[u]*4,j0=tj[u]*4;
  #pragma unroll
  for(int q=0;q<4;++q)
   #pragma unroll
   for(int w=0;w<4;++w)if(i0+q<h&&j0+w<h)Pz[(i0+q)+size_t(j0+w)*h]=acc[u][q][w];}
}
template<class T>
__global__ void gau_overlap_combine(const T*P,int S,int h,T*O){
 for(int e=blockIdx.x*blockDim.x+threadIdx.x;e<h*h;e+=gridDim.x*blockDim.x){const int i=e%h,j=e/h;
  if(i>j){O[e]=T(0);continue;}
  const int ti=i/4,tj=j/4;if(ti>tj){O[e]=T(0);continue;}   // lower part of a diagonal 4x4 tile is also written
  T s=T(0);for(int z=0;z<S;++z)s+=P[size_t(z)*h*h+e];O[e]=s;}
}

// T of Q = I - V T V^T for the h reflectors, from the reflector overlaps O = V^T V (upper part used; argument G) and
// tau, as in LAPACK dlarft. O is bounded by the dimensions alone (unit lower triangular V, |v_ij| <= 1).
template<class T>
__global__ __launch_bounds__(512) void gau_build_t_kernel(const T*G,int ldg,const T*tau,int h,T*tri,int b){
 // Shared: S (h x h, ld h+1): T in the upper triangle, the overlaps O(i,j), i<j, stored transposed at S[j,i];
 // M: compose scratch (at most h*h/4). Everything after the coalesced load runs from shared memory.
 extern __shared__ __align__(16) unsigned char gau_smem[];
 const int ld=h+1;T*S=reinterpret_cast<T*>(gau_smem);T*M=S+size_t(h)*ld;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,nt=blockDim.x;
 for(int e=tid;e<h*h;e+=nt){const int i=e%h,j=e/h;   // coalesced along i
  const T g=G[i+size_t(j)*ldg];
  if(i<j)S[j+i*ld]=g;else if(i==j)S[i+i*ld]=tau[i];}
 __syncthreads();
 auto Tv=[&](int i,int j)->T&{return S[i+j*ld];};      // i <= j
 auto Ov=[&](int i,int j)->T{return S[j+i*ld];};        // i <  j
 // level 0: dlarft inside every 8-column block, one warp per block, lanes over rows
 const int nb=(h+7)/8;
 for(int blk=warp;blk<nb;blk+=nt/32){const int b0=blk*8,b1=min(h,b0+8);
  for(int c=b0+1;c<b1;++c){
   T v=T(0);const int r=b0+lane;
   if(r<c){for(int q=r;q<c;++q)v+=Tv(r,q)*Ov(q,c);v*=-Tv(c,c);}
   __syncwarp();
   if(r<c)Tv(r,c)=v;
   __syncwarp();}}
 __syncthreads();
 // Eq compose, level s: T12 = -T11 (O12 T22). 4x4 register tiles; triangular factors masked by index.
 for(int s=8;s<h;s*=2){
  const int pairs=(h+2*s-1)/(2*s),ts=s/4,tiles=pairs*ts*ts;
  for(int t=tid;t<tiles;t+=nt){
   const int pr=t/(ts*ts),rem=t-pr*ts*ts,i0=(rem%ts)*4,j0=(rem/ts)*4,a0=2*s*pr,a1=a0+s,w2=min(s,h-a1);
   if(a1>=h||j0>=w2)continue;
   T acc[4][4]={};
   for(int q=0;q<min(s,j0+4);++q){T x[4],y[4];
    #pragma unroll
    for(int u=0;u<4;++u){x[u]=Ov(a0+i0+u,a1+q);const int j=j0+u;y[u]=(j<w2&&q<=j)?Tv(a1+q,a1+j):T(0);}
    #pragma unroll
    for(int u=0;u<4;++u)
     #pragma unroll
     for(int v=0;v<4;++v)acc[u][v]+=x[u]*y[v];}
   #pragma unroll
   for(int u=0;u<4;++u)
    #pragma unroll
    for(int v=0;v<4;++v)M[size_t(pr)*s*s+(i0+u)+(j0+v)*s]=acc[u][v];}
  __syncthreads();
  for(int t=tid;t<tiles;t+=nt){
   const int pr=t/(ts*ts),rem=t-pr*ts*ts,i0=(rem%ts)*4,j0=(rem/ts)*4,a0=2*s*pr,a1=a0+s,w2=min(s,h-a1);
   if(a1>=h||j0>=w2)continue;
   T acc[4][4]={};
   for(int q=i0;q<s;++q){T x[4],y[4];
    #pragma unroll
    for(int u=0;u<4;++u){const int i=i0+u;x[u]=q>=i?Tv(a0+i,a0+q):T(0);y[u]=M[size_t(pr)*s*s+q+(j0+u)*s];}
    #pragma unroll
    for(int u=0;u<4;++u)
     #pragma unroll
     for(int v=0;v<4;++v)acc[u][v]+=x[u]*y[v];}
   #pragma unroll
   for(int u=0;u<4;++u)
    #pragma unroll
    for(int v=0;v<4;++v)if(j0+v<w2)Tv(a0+i0+u,a1+j0+v)=-acc[u][v];}
  __syncthreads();
 }
 for(int e=tid;e<b*b;e+=nt){const int i=e%b,j=e/b;tri[e]=(i<h&&j<h&&i<=j)?Tv(i,j):T(0);}
}

// Width-specialised T builder (same algebra as gau_build_t_kernel: dlarft on 8-column blocks, then Eq compose
// T12 = -T11 (O12 T22) level by level). H is a compile-time bound (h <= H), so every loop is unrolled; columns h..H-1
// carry tau = 0 and zero overlaps, hence exact zero T entries that are never stored. Shared: S (H x H, ld H+1) holds T
// in the upper triangle and O(i,j), i<j, transposed in the lower; M holds O12 T22 for every pair of the level.
template<class T,int H,int S>
__device__ __forceinline__ void gau_t_compose_level(T*Sm,T*M,int tid){
 constexpr int LD=H+1,P=H/(2*S),OUT=P*S*S,NT=512;
 constexpr int PER=OUT>=NT?OUT/NT:1;                  // outputs per thread (power of two)
 constexpr int TC=PER>=4?4:PER,TR=PER/TC;              // TR x TC register tile (rows x columns)
 constexpr int TILES_R=S/TR,TILES_C=S/TC,TILES=P*TILES_R*TILES_C;
 // phase A: M = O12 T22  (T22 upper: q <= j)
 for(int t=tid;t<TILES;t+=NT){
  const int pr=t/(TILES_R*TILES_C),rem=t-pr*TILES_R*TILES_C,i0=(rem%TILES_R)*TR,j0=(rem/TILES_R)*TC,a0=2*S*pr,a1=a0+S;
  T acc[TR][TC];
  #pragma unroll
  for(int u=0;u<TR;++u)
   #pragma unroll
   for(int v=0;v<TC;++v)acc[u][v]=T(0);
  #pragma unroll 8
  for(int q=0;q<S;++q){T x[TR],y[TC];
   #pragma unroll
   for(int u=0;u<TR;++u)x[u]=Sm[(a1+q)+(a0+i0+u)*LD];            // O(a0+i, a1+q), stored transposed
   #pragma unroll
   for(int v=0;v<TC;++v)y[v]=q<=j0+v?Sm[(a1+q)+(a1+j0+v)*LD]:T(0);  // T(a1+q, a1+j)
   #pragma unroll
   for(int u=0;u<TR;++u)
    #pragma unroll
    for(int v=0;v<TC;++v)acc[u][v]+=x[u]*y[v];}
  #pragma unroll
  for(int u=0;u<TR;++u)
   #pragma unroll
   for(int v=0;v<TC;++v)M[pr*S*S+(i0+u)+(j0+v)*S]=acc[u][v];}
 __syncthreads();
 // phase B: T12 = -T11 M  (T11 upper: q >= i)
 for(int t=tid;t<TILES;t+=NT){
  const int pr=t/(TILES_R*TILES_C),rem=t-pr*TILES_R*TILES_C,i0=(rem%TILES_R)*TR,j0=(rem/TILES_R)*TC,a0=2*S*pr,a1=a0+S;
  T acc[TR][TC];
  #pragma unroll
  for(int u=0;u<TR;++u)
   #pragma unroll
   for(int v=0;v<TC;++v)acc[u][v]=T(0);
  #pragma unroll 8
  for(int q=0;q<S;++q){T x[TR],y[TC];
   #pragma unroll
   for(int u=0;u<TR;++u)x[u]=q>=i0+u?Sm[(a0+i0+u)+(a0+q)*LD]:T(0);  // T(a0+i, a0+q)
   #pragma unroll
   for(int v=0;v<TC;++v)y[v]=M[pr*S*S+q+(j0+v)*S];
   #pragma unroll
   for(int u=0;u<TR;++u)
    #pragma unroll
    for(int v=0;v<TC;++v)acc[u][v]+=x[u]*y[v];}
  #pragma unroll
  for(int u=0;u<TR;++u)
   #pragma unroll
   for(int v=0;v<TC;++v)Sm[(a0+i0+u)+(a1+j0+v)*LD]=-acc[u][v];}
 __syncthreads();
}
template<class T,int H>
__global__ __launch_bounds__(512) void gau_build_t_h(const T*G,int ldg,size_t gstride,const T*tau,int fstride,const TilePacket*packets,T*Ts,int b){
 extern __shared__ __align__(16) unsigned char gau_smem[];
 constexpr int LD=H+1;T*Sm=reinterpret_cast<T*>(gau_smem);T*M=Sm+H*LD;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,id=blockIdx.x;
 const TilePacket pk=packets[id];const int h=pk.h;
 G+=size_t(id)*gstride;tau+=size_t(id)*fstride;T*tri=Ts+size_t(pk.tile)*b*b;
 for(int e=tid;e<H*H;e+=512){const int i=e%H,j=e/H;
  const T g=(i<h&&j<h)?G[i+size_t(j)*ldg]:T(0);
  if(i<j)Sm[j+i*LD]=g;else if(i==j)Sm[i+i*LD]=i<h?tau[i]:T(0);}
 __syncthreads();
 // level 0: dlarft inside each 8-column block; one warp per block, lanes 0..7 own the rows
 if(warp<H/8){const int b0=warp*8,r=b0+lane;
  #pragma unroll
  for(int c=1;c<8;++c){
   T v=T(0);
   if(lane<c){
    #pragma unroll
    for(int q=0;q<8;++q)if(q>=lane&&q<c)v+=Sm[r+(b0+q)*LD]*Sm[(b0+c)+(b0+q)*LD];
    v*=-Sm[(b0+c)+(b0+c)*LD];}
   __syncwarp();
   if(lane<c)Sm[r+(b0+c)*LD]=v;
   __syncwarp();}}
 __syncthreads();
 if constexpr(H>8)gau_t_compose_level<T,H,8>(Sm,M,tid);
 if constexpr(H>16)gau_t_compose_level<T,H,16>(Sm,M,tid);
 if constexpr(H>32)gau_t_compose_level<T,H,32>(Sm,M,tid);
 if constexpr(H>64)gau_t_compose_level<T,H,64>(Sm,M,tid);
 for(int e=tid;e<b*b;e+=512){const int i=e%b,j=e/b;tri[e]=(i<h&&j<h&&i<=j)?Sm[i+j*LD]:T(0);}
}
template<class T,int H> inline size_t gau_build_t_h_smem(){return (size_t(H)*(H+1)+size_t(H)*H/4+8)*sizeof(T);}
template<class T,int H> inline void launch_gau_build_t_h(const T*G,int ldg,size_t gstride,const T*tau,int fstride,const TilePacket*pk,int count,T*Ts,int b,cudaStream_t st){
 static bool set=false;if(!set){CU(cudaFuncSetAttribute(gau_build_t_h<T,H>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(gau_build_t_h_smem<T,H>())));set=true;}
 gau_build_t_h<T,H><<<count,512,gau_build_t_h_smem<T,H>(),st>>>(G,ldg,gstride,tau,fstride,pk,Ts,b);CU(cudaGetLastError());}
// T for every packet of the batch (hmax = widest packet): packet id reads O at G + id*gstride (ld ldg), tau at
// tau + id*fstride, and writes its b x b T block at Ts + packets[id].tile*b*b.
template<class T> inline void launch_gau_build_t_batch(const T*G,int ldg,size_t gstride,const T*tau,int fstride,const TilePacket*pk,int count,int hmax,T*Ts,int b,cudaStream_t st){
 if(hmax<=32)launch_gau_build_t_h<T,32>(G,ldg,gstride,tau,fstride,pk,count,Ts,b,st);
 else if(hmax<=64)launch_gau_build_t_h<T,64>(G,ldg,gstride,tau,fstride,pk,count,Ts,b,st);
 else if(hmax<=128)launch_gau_build_t_h<T,128>(G,ldg,gstride,tau,fstride,pk,count,Ts,b,st);
 else throw std::runtime_error("gau_build_t_width");}

// Register capacity: C*ITEMS <= 64 fp32 words (32 fp64) per thread for the owned columns, THREADS <= 512.
struct GauConfig{int C=0,items=0,threads=0;};
// {C, ITEMS, THREADS}, smallest first; every entry keeps C*ITEMS <= 64 registers of owned columns.
inline const std::vector<std::array<int,3>>&gau_menu(size_t word){
 // Register budget per thread is 65536/THREADS (<= 255): 256-thread owners carry 16 fp32 rows x 8 columns.
 static const std::vector<std::array<int,3>> f32{{8,1,128},{8,2,128},{8,2,256},{8,4,256},{8,8,256},{8,16,256},{4,16,512},{2,32,512},{1,32,1024}};
 static const std::vector<std::array<int,3>> f64{{8,1,128},{8,2,128},{8,2,256},{8,4,256},{8,8,256},{4,16,256},{2,16,512},{1,16,1024}};
 return word==4?f32:f64;}
template<class T> inline GauConfig gau_config(int rows){GauConfig g;
 for(auto&m:gau_menu(sizeof(T)))if(rows<=m[1]*m[2]){g.C=m[0];g.items=m[1];g.threads=m[2];return g;}
 return g;}
inline int gau_max_rows(size_t word){const auto&m=gau_menu(word).back();return m[1]*m[2];}

// Capability/residency/spill checks run once per (T, C, ITEMS, THREADS) and are cached: they are host queries
// (device properties, occupancy, function attributes) that must never sit inside a timed panel.
template<class T,int C,int ITEMS,int THREADS>
inline void gau_ge_launch_one(T*A,int lda,const TilePacket*pk,int count,int hmax,T*V,int ldv,size_t vstride,T*tau,int*flags,int fstride,int epoch,int kind,int*status,Witness*w,cudaStream_t st,bool dry){
 auto kernel=gau_ge_kernel<T,C,ITEMS,THREADS>;
 const int owners=(hmax+C-1)/C;
 static int resident=-1;static bool spill=false;static bool coop=false;
 if(resident<0){int device;CU(cudaGetDevice(&device));int sms=0,cl=0;
  CU(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));CU(cudaDeviceGetAttribute(&cl,cudaDevAttrCooperativeLaunch,device));
  int occ=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kernel,THREADS,0));
  cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,kernel));spill=attr.localSizeBytes>64;coop=cl!=0;resident=occ*sms;}
 if(!coop||owners>resident)throw std::runtime_error("gau_ge_residency_before_modify");
 if(spill)throw std::runtime_error("gau_ge_register_spill_before_modify");
 if(dry)return;
 // Packets are independent, so the batch is issued in chunks that are co-resident as a whole (cooperative launch =
 // residency guarantee, so a waiting owner never excludes an unscheduled producer of its own packet).
 const int per=std::max(1,resident/owners);
 for(int off=0;off<count;off+=per){
  const int n=std::min(per,count-off);
  const TilePacket*p=pk+off;T*v=V+size_t(off)*vstride;T*t=tau+size_t(off)*fstride;int*f=flags+size_t(off)*fstride;int ow=owners;
  void*args[]={&A,&lda,&p,&ow,&v,&ldv,&vstride,&t,&f,&fstride,&epoch,&kind,&status,&w};
  CU(cudaLaunchCooperativeKernel((const void*)kernel,dim3(n*owners),dim3(THREADS),args,0,st));}
}
// Returns the executed GauConfig (C = columns per owner). Throws before any write if the packets do not fit.
template<class T>
inline GauConfig launch_gau_ge(T*A,int lda,const TilePacket*pk,int count,int rows,int h,T*V,int ldv,size_t vstride,T*tau,int*flags,int fstride,int epoch,int kind,int*status,Witness*w,cudaStream_t st,bool dry,bool allow_wide=false){
 if(h<1||h>(allow_wide?512:128)||rows<h||count<1)throw std::runtime_error("gau_ge_shape_before_modify");
 const GauConfig g=gau_config<T>(rows);
 if(!g.C)throw std::runtime_error("gau_ge_rows_exceed_register_capacity_before_modify");
#define TQR_GAU_CASE(Cc,I,Th) if(g.C==Cc&&g.items==I&&g.threads==Th){gau_ge_launch_one<T,Cc,I,Th>(A,lda,pk,count,h,V,ldv,vstride,tau,flags,fstride,epoch,kind,status,w,st,dry);return g;}
 if constexpr(sizeof(T)==4){
  TQR_GAU_CASE(8,1,128)TQR_GAU_CASE(8,2,128)TQR_GAU_CASE(8,2,256)TQR_GAU_CASE(8,4,256)TQR_GAU_CASE(8,8,256)TQR_GAU_CASE(8,16,256)
  TQR_GAU_CASE(4,16,512)TQR_GAU_CASE(2,32,512)TQR_GAU_CASE(1,32,1024)
 }else{
  TQR_GAU_CASE(8,1,128)TQR_GAU_CASE(8,2,128)TQR_GAU_CASE(8,2,256)TQR_GAU_CASE(8,4,256)TQR_GAU_CASE(8,8,256)
  TQR_GAU_CASE(4,16,256)TQR_GAU_CASE(2,16,512)TQR_GAU_CASE(1,16,1024)
 }
#undef TQR_GAU_CASE
 throw std::runtime_error("gau_ge_config_not_instantiated_before_modify");
}
inline int gau_overlap_slices(int rows){return std::max(1,std::min(128,(rows+255)/256));}
template<class T> inline size_t gau_overlap_workspace(int leaf,int b){return size_t(gau_overlap_slices(leaf))*b*b+size_t(b)*b;}
// O = V^T V into O (h x h, ld h) using workspace P (gau_overlap_workspace words). Returns the slice count S (= c).
template<class T> inline int launch_gau_overlaps(const T*V,int ldv,int rows,int h,T*P,T*O,cudaStream_t st){
 const int S=gau_overlap_slices(rows);const size_t sm=size_t(GAU_OV_CHUNK)*(h+1)*sizeof(T);
 gau_overlap_partial<T><<<S,256,sm,st>>>(V,ldv,rows,h,S,P);
 gau_overlap_combine<T><<<std::max(1,(h*h+255)/256),256,0,st>>>(P,S,h,O);CU(cudaGetLastError());return S;}
template<class T> inline size_t gau_build_t_smem(int h){return (size_t(h)*(h+1)+size_t(h)*h/4+64)*sizeof(T);}
template<class T> inline void launch_gau_build_t(const T*G,int ldg,const T*tau,int h,T*tri,int b,cudaStream_t st){
 const size_t sm=gau_build_t_smem<T>(h);static size_t set=0;
 if(sm>set){CU(cudaFuncSetAttribute(gau_build_t_kernel<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(gau_build_t_smem<T>(128))));set=gau_build_t_smem<T>(128);}
 gau_build_t_kernel<T><<<1,512,sm,st>>>(G,ldg,tau,h,tri,b);CU(cudaGetLastError());
}
}
