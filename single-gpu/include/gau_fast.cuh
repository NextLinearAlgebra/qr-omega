#pragma once
// Register-resident panel kernels for the GE and TT factors of the last panels.
// Adapted from gau.nernst's submission 844219 to the GPU MODE QR-v2 leaderboard (774): register_panel_kernel (line 773) and register_2sm_panel_kernel
// (line 923), ported in gau_sm.cuh. Changes:
// 1. dlarfg with a range-checked fast path: beta, tau and 1/(alpha - beta) in the working precision with IEEE sqrt and
//    division when the unscaled sum of squares is a normal number well inside the range (there it equals the scaled
//    one to working accuracy); otherwise the fp64-accumulated (fp32 storage) or max-scaled (fp64 storage) LAPACK
//    path. The branch is warp-uniform. Same reflectors as LAPACK, never approximate sqrt/rcp.
// 2. Batched warp reductions: the VEC dot products of one reflector with a warp's columns are all-reduced by recursive
//    halving (fixed order) instead of VEC independent 5-level butterflies.
// 3. The reflector overlaps O(k,c) = v_k^T v_c (k < c: the quantity LAPACK dlarft forms, never A^T A) are formed IN
//    the chain: once a warp has published its reflectors it keeps them in registers and forms its VEC overlaps with
//    every later reflector as it appears. O(k,c) is packed into rows k < c of reflector c's shared slot -- entries a
//    unit lower triangular v_c leaves unused (consumers only read rows >= c).
// 4. T of Q = I - V T V^T is built in the same kernel from O and tau (dlarft on 8-column blocks, then Eq compose
//    tile. One launch per batch of packets replaces chain + overlap product + T builder.
// Paper mapping: the elimination list, its reflectors and their order are unchanged (one GEQRT / TTQRT of the list
// per packet). Inside it the columns [J] are split over warps (p_j = h / VEC) with each reflector REPLICATED to the
// later owners through shared memory (DSM across the cluster); the rows [K] are split over the 32 lanes, whose
// partial dots and sums of squares over disjoint rows are joined by the additive combine (the shuffle tree): c = 32
// at the warp level. Owners commit their columns exactly once.
#include "gau_sm.cuh"
#include <cooperative_groups.h>
namespace tqr {

// Optional per-warp timeline (clock64) for probes: [warp][0..3] = start, consumed, own chain done, overlaps done;
__device__ unsigned long long*gf_dbg=nullptr;
// dlarfg on column i of a warp's registers (row r = it*32 + lane), pivot row c. On return the column holds beta at the
// pivot and v (unit pivot implied) below it; returns tau (0 and the column untouched when the tail is zero: H = I).
// Fast path (the unscaled sum of squares q is a normal number well inside the range, so it equals the scaled one to
// working accuracy): with y = 1/sqrt(q), s = |alpha| + ||x|| (no cancellation): beta = -sign(alpha) ||x||,
// tau = (beta - alpha)/beta = s y, 1/(alpha - beta) = sign(alpha)/s. Otherwise the fp64-accumulated (fp32 storage) or
// power-of-two scaled (fp64 storage) LAPACK path. Every branch is warp-uniform.
template<class T,int ITEMS,int VEC>
__device__ __forceinline__ T gf_reflector(T(&col)[ITEMS][VEC],int i,int c,int rows,int lane,int*status){
 T tl=T(0),xc=T(0);unsigned nz=0;
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;const T y=col[it][i];
  if(r>c&&r<rows){tl=gf_fma(y,y,tl);nz|=gf_nz(y);}
  if(r==c)xc=y;}
 const bool has=__any_sync(0xffffffffu,nz!=0u);
 tl=gau_warp_sum(tl);
 const T x0=__shfl_sync(0xffffffffu,xc,c&31);
 if(!has)return T(0);
 T beta,tau;
 if(tl>=GfRange<T>::lo&&tl<=GfRange<T>::hi&&fabs(x0)<=GfRange<T>::xmax){
  const T q=gf_fma(x0,x0,tl),y=gf_rsqrt(q),nrm=q*y,sa=fabs(x0)+nrm,rs=gf_rcp(sa);
  beta=x0<T(0)?nrm:-nrm;tau=sa*y;const T inv=x0<T(0)?-rs:rs;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r==c)col[it][i]=beta;else if(r>c&&r<rows)col[it][i]*=inv;}
 }else if constexpr(sizeof(T)==4){
  // fp32 storage: fp32 squares can neither overflow nor underflow fp64, so the fp64 sum is the scaled one
  double t2=0.0;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>c&&r<rows){const double y=double(col[it][i]);t2=fma(y,y,t2);}}
  t2=gau_warp_sum(t2);
  const double a=double(x0),nrm=sqrt(fma(a,a,t2)),b=a<0?nrm:-nrm,inv=1.0/(a-b);
  beta=T(b);tau=T((b-a)/b);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r==c)col[it][i]=beta;else if(r>c&&r<rows)col[it][i]=T(double(col[it][i])*inv);}
 }else{
  // fp64 storage: scale by the power of two of the largest magnitude (LAPACK dnrm2 / dlapy2 range), exactly
  double m=fabs(double(x0));
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>c&&r<rows)m=fmax(m,fabs(double(col[it][i])));}
  #pragma unroll
  for(int o=16;o;o>>=1)m=fmax(m,__shfl_xor_sync(0xffffffffu,m,o));
  int e;frexp(m,&e);double t2=0.0;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>c&&r<rows){const double y=scalbn(double(col[it][i]),-e);t2=fma(y,y,t2);}}
  t2=gau_warp_sum(t2);
  const double a=scalbn(double(x0),-e),nrm=sqrt(fma(a,a,t2)),bs=a<0?nrm:-nrm,inv=1.0/(a-bs);
  beta=T(scalbn(bs,e));tau=T((bs-a)/bs);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r==c)col[it][i]=beta;else if(r>c&&r<rows)col[it][i]=T(scalbn(double(col[it][i]),-e)*inv);}
 }
 if(lane==0&&!isfinite(beta))atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 return tau;
}
// x_q <- x_q - v tau (v^T x_q) for the warp's columns q >= first (all rows in registers, v in registers).
template<class T,int ITEMS,int VEC>
__device__ __forceinline__ void gf_apply(T(&col)[ITEMS][VEC],const T(&v)[ITEMS],T tk,int first,int lane){
 T d[VEC];
 #pragma unroll
 for(int q=0;q<VEC;++q){T s=T(0);
  if(q>=first){
   #pragma unroll
   for(int it=0;it<ITEMS;++it)s=gf_fma(v[it],col[it][q],s);}
  d[q]=s;}
 gf_allreduce<T,VEC>(d,lane);
 #pragma unroll
 for(int q=0;q<VEC;++q)if(q>=first){const T f=-(d[q]*tk);
  #pragma unroll
  for(int it=0;it<ITEMS;++it)col[it][q]=gf_fma(v[it],f,col[it][q]);}
}

// ------------------------------------------------------------------ T from O and tau inside one CTA ----------------
// S (H x (H+1), ld H+1): tau on the diagonal, O(i,j) (i < j) stored transposed at S[j + i*LD]; on return T (upper) sits
// in S[i + j*LD], i <= j. M: compose scratch of H*H/4 words. Columns h..H-1 carry tau = 0 and zero overlaps, so their
// T entries are exactly zero. dlarft on 8-column blocks (one warp per block), then Eq compose level by level.
template<class T,int H,int SS>
__device__ __forceinline__ void gf_t_compose(T*Sm,T*M,int tid,int nt){
 constexpr int LD=H+1,P=H/(2*SS),TS=SS/4,TILES=P*TS*TS;
 // phase A: M = O12 T22 (T22 upper: q <= j)
 for(int t=tid;t<TILES;t+=nt){
  const int pr=t/(TS*TS),rem=t-pr*TS*TS,i0=(rem%TS)*4,j0=(rem/TS)*4,a0=2*SS*pr,a1=a0+SS;
  T acc[4][4];
  #pragma unroll
  for(int u=0;u<4;++u)
   #pragma unroll
   for(int v=0;v<4;++v)acc[u][v]=T(0);
  for(int q=0;q<j0+4;++q){T x[4],y[4];
   #pragma unroll
   for(int u=0;u<4;++u)x[u]=Sm[(a1+q)+(a0+i0+u)*LD];
   #pragma unroll
   for(int v=0;v<4;++v)y[v]=q<=j0+v?Sm[(a1+q)+(a1+j0+v)*LD]:T(0);
   #pragma unroll
   for(int u=0;u<4;++u)
    #pragma unroll
    for(int v=0;v<4;++v)acc[u][v]=gf_fma(x[u],y[v],acc[u][v]);}
  #pragma unroll
  for(int u=0;u<4;++u)
   #pragma unroll
   for(int v=0;v<4;++v)M[pr*SS*SS+(i0+u)+(j0+v)*SS]=acc[u][v];}
 __syncthreads();
 // phase B: T12 = -T11 M (T11 upper: q >= i)
 for(int t=tid;t<TILES;t+=nt){
  const int pr=t/(TS*TS),rem=t-pr*TS*TS,i0=(rem%TS)*4,j0=(rem/TS)*4,a0=2*SS*pr,a1=a0+SS;
  T acc[4][4];
  #pragma unroll
  for(int u=0;u<4;++u)
   #pragma unroll
   for(int v=0;v<4;++v)acc[u][v]=T(0);
  for(int q=i0;q<SS;++q){T x[4],y[4];
   #pragma unroll
   for(int u=0;u<4;++u)x[u]=q>=i0+u?Sm[(a0+i0+u)+(a0+q)*LD]:T(0);
   #pragma unroll
   for(int v=0;v<4;++v)y[v]=M[pr*SS*SS+q+(j0+v)*SS];
   #pragma unroll
   for(int u=0;u<4;++u)
    #pragma unroll
    for(int v=0;v<4;++v)acc[u][v]=gf_fma(x[u],y[v],acc[u][v]);}
  #pragma unroll
  for(int u=0;u<4;++u)
   #pragma unroll
   for(int v=0;v<4;++v)Sm[(a0+i0+u)+(a1+j0+v)*LD]=-acc[u][v];}
 __syncthreads();
}
template<class T,int H>
__device__ __forceinline__ void gf_t_build(T*Sm,T*M,int tid,int nt){
 constexpr int LD=H+1;const int lane=tid&31,warp=tid>>5;
 for(int blk=warp;blk<H/8;blk+=nt>>5){const int b0=blk*8,r=b0+lane;
  #pragma unroll
  for(int c=1;c<8;++c){
   T v=T(0);
   if(lane<c){
    #pragma unroll
    for(int q=0;q<8;++q)if(q>=lane&&q<c)v=gf_fma(Sm[r+(b0+q)*LD],Sm[(b0+c)+(b0+q)*LD],v);
    v*=-Sm[(b0+c)+(b0+c)*LD];}
   __syncwarp();
   if(lane<c)Sm[r+(b0+c)*LD]=v;
   __syncwarp();}}
 __syncthreads();
 unsigned long long*dbg=gf_dbg&&blockIdx.x<=1&&tid==0?gf_dbg:nullptr;
 if(dbg)dbg[184]=clock64();
 if constexpr(H>8)gf_t_compose<T,H,8>(Sm,M,tid,nt);
 if(dbg)dbg[185]=clock64();
 if constexpr(H>16)gf_t_compose<T,H,16>(Sm,M,tid,nt);
 if(dbg)dbg[186]=clock64();
 if constexpr(H>32)gf_t_compose<T,H,32>(Sm,M,tid,nt);
 if(dbg)dbg[187]=clock64();
 if constexpr(H>64)gf_t_compose<T,H,64>(Sm,M,tid,nt);
 if(dbg)dbg[188]=clock64();
}
__host__ __device__ inline int gf_t_width(int h){return h<=32?32:h<=64?64:128;}
template<class T> __host__ __device__ inline size_t gf_t_words(int h){const int H=gf_t_width(h);return size_t(H)*(H+1)+size_t(H)*H/4;}
// Fill S from an overlap accessor, build T, write the b x b tile (T upper, zeros elsewhere).
template<class T,class Get>
__device__ __forceinline__ void gf_t_finish(T*Sm,const T*taus,int h,Get get,T*tri,int b,int tid,int nt){
 const int H=gf_t_width(h),LD=H+1;T*M=Sm+size_t(H)*LD;
 unsigned long long*dbg=gf_dbg&&blockIdx.x<=1&&tid==0?gf_dbg:nullptr;
 if(dbg)dbg[180]=clock64();
 for(int e=tid;e<H*H;e+=nt){const int i=e%H,j=e/H;
  if(i<j)Sm[j+i*LD]=j<h?get(i,j):T(0);else if(i==j)Sm[i+i*LD]=i<h?taus[i]:T(0);}
 __syncthreads();
 if(dbg)dbg[181]=clock64();
 if(H==32)gf_t_build<T,32>(Sm,M,tid,nt);else if(H==64)gf_t_build<T,64>(Sm,M,tid,nt);else gf_t_build<T,128>(Sm,M,tid,nt);
 if(dbg)dbg[182]=clock64();
 for(int e=tid;e<b*b;e+=nt){const int i=e%b,j=e/b;tri[e]=(i<=j&&j<h)?Sm[i+j*LD]:T(0);}
 if(dbg)dbg[183]=clock64();
}

// ------------------------------------------------------------------ single SM ------------------------------------
// One CTA per packet; warp w owns columns [w VEC, (w+1) VEC) with all ROWS = 32 ITEMS rows in registers, and the
// same rows of T. Shared: refl [COLS][SL] (slot c: v_c on rows >= c; O(k,c) = v_k^T v_c on rows k < c; the slot
// stride SL = ROWS + 16 B keeps bulk alignment and moves consecutive slots to different banks), taus, mbarriers
// (reflector published / overlaps of a block complete / diagonal T block ready), packed upper T, per-warp scratch.
// T is built by LAPACK dlarft's left-looking blocked recurrence while the chain runs: when block bb's overlaps are
// complete, its owner forms the diagonal block T22 (dlarft inside the block) and every earlier warp w forms its rows
// of T12 = -T11 O12 T22 as T(rows_w, block) = -Y T22 with Y = T(rows_w, :) O(:, block) (Eq compose).
template<class T> __host__ __device__ constexpr int gf_slot(int rows){return rows+16/int(sizeof(T));}
template<class T> inline size_t gf_sm1_smem(int items,int vec,int warps){
 const size_t cols=size_t(warps)*vec,sl=size_t(gf_slot<T>(items*32));
 return (cols*sl+cols+(cols&1))*sizeof(T)+(cols+2*size_t(warps))*8+(cols*(cols+1)/2+size_t(warps)*vec*vec)*sizeof(T)+64;}
template<class T,int ITEMS,int VEC,int MAXW>
__global__ __launch_bounds__(MAXW*32,1) void gf_sm1_kernel(T*A,int lda,const TilePacket*packets,T*Vout,int ldv,size_t vstride,
                                                           T*tau_out,int fstride,T*Ts,int b,int stage,int kind,int*status,Witness*w){
 constexpr int ROWS=ITEMS*32,SL=gf_slot<T>(ROWS);
 const int nt=blockDim.x,WARPS=nt>>5,COLS=WARPS*VEC;
 extern __shared__ __align__(16) unsigned char gau_smem[];
 const int id=blockIdx.x;const TilePacket pk=packets[id];const int rows=pk.rows,h=pk.h;
 T*refl=reinterpret_cast<T*>(gau_smem);
 T*taus=refl+size_t(COLS)*SL;
 unsigned long long*mb=reinterpret_cast<unsigned long long*>(taus+COLS+(COLS&1));
 unsigned long long*oc=mb+COLS,*td=oc+WARPS;
 T*Tp=reinterpret_cast<T*>(td+WARPS);                        // T(i,j), i <= j, at j(j+1)/2 + i
 T*ysc=Tp+size_t(COLS)*(COLS+1)/2;                            // [WARPS][VEC][VEC]
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,c0=warp*VEC,nb=(h+VEC-1)/VEC;
 T*a=A+pk.row+size_t(pk.col)*lda;T*vo=Vout+size_t(id)*vstride;
 if(tid==0){
  for(int i=0;i<COLS;++i)gau_mbar_init(gau_smem_addr(mb+i),32);
  for(int bb=0;bb<WARPS;++bb){gau_mbar_init(gau_smem_addr(oc+bb),bb+1);gau_mbar_init(gau_smem_addr(td+bb),1);}
  asm volatile("fence.mbarrier_init.release.cluster;");}
 __syncthreads();
 T col[ITEMS][VEC];
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int q=0;q<VEC;++q)col[it][q]=(r<rows&&c0+q<h)?a[r+size_t(c0+q)*lda]:T(0);}
 unsigned long long*dbg=gf_dbg&&id==0?gf_dbg:nullptr;
 if(dbg&&lane==0)dbg[warp*4+0]=clock64();
 // consume every reflector of the warps to the left, in order
 const int kend=min(c0,h);
 for(int k=0;k<kend;++k){
  gau_mbar_wait(gau_smem_addr(mb+k),0);
  T v[ITEMS];
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;v[it]=r>=k?refl[size_t(k)*SL+r]:T(0);}
  gf_apply<T,ITEMS,VEC>(col,v,taus[k],0,lane);
 }
 if(dbg&&lane==0)dbg[warp*4+1]=clock64();
 // generate, publish (shared slot + V column + tau in global memory) and apply this warp's reflectors
 #pragma unroll
 for(int i=0;i<VEC;++i){
  const int c=c0+i;if(c>=h)break;
  const T tk=gf_reflector<T,ITEMS,VEC>(col,i,c,rows,lane,status);
  T v[ITEMS];
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;v[it]=r==c?T(1):r>c?col[it][i]:T(0);if(r>=c)refl[size_t(c)*SL+r]=v[it];}
  if(lane==0)taus[c]=tk;
  gau_mbar_arrive(gau_smem_addr(mb+c));
  if(i+1<VEC)gf_apply<T,ITEMS,VEC>(col,v,tk,i+1,lane);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r<ldv)vo[r+size_t(c)*ldv]=v[it];}
  for(int r=ROWS+lane;r<ldv;r+=32)vo[r+size_t(c)*ldv]=T(0);
  if(lane==0)tau_out[size_t(id)*fstride+c]=tk;
 }
 if(dbg&&lane==0)dbg[warp*4+2]=clock64();
 // commit: R above, beta on, v below the diagonal (the owner writes its columns once)
 bool bad=false;
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r<rows){
  #pragma unroll
  for(int q=0;q<VEC;++q)if(c0+q<h){const T y=col[it][q];bad|=!isfinite(y);a[r+size_t(c0+q)*lda]=y;}}}
 if(bad)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int q=0;q<VEC;++q){const int c=c0+q;col[it][q]=(c<h&&r>=c)?(r==c?T(1):col[it][q]):T(0);}}
 constexpr int SH=5-GfLog<VEC>::v;
 constexpr int NP=VEC*VEC,PPL=NP>=32?NP/32:1,LPP=NP>=32?1:32/NP;   // (row, column) pairs per lane / lanes per pair
 T*ys=ysc+size_t(warp)*NP;
 if(c0<h)for(int bb=warp;bb<nb;++bb){
  const int jb=bb*VEC,j0=max(jb,c0+1),j1=min(jb+VEC,h);
  // overlaps O(c0+q, j) of this warp's reflectors with block bb's
  for(int j=j0;j<j1;++j){
   if(bb>warp)gau_mbar_wait(gau_smem_addr(mb+j),0);
   T d[VEC];
   #pragma unroll
   for(int q=0;q<VEC;++q)d[q]=T(0);
   #pragma unroll
   for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>=j){const T y=refl[size_t(j)*SL+r];
    #pragma unroll
    for(int q=0;q<VEC;++q)d[q]=gf_fma(col[it][q],y,d[q]);}}
   const T s=gf_reduce_scatter<T,VEC>(d,lane);
   const int q=lane>>SH;
   if((lane&((1<<SH)-1))==0&&c0+q<j)refl[size_t(j)*SL+c0+q]=s;
  }
  __syncwarp();
  if(lane==0)gau_mbar_arrive(gau_smem_addr(oc+bb));
  if(bb==warp){
   // diagonal block: dlarft inside the block, lane = row
   if(lane<VEC&&c0+lane<h){const int r=c0+lane;T t[VEC];
    #pragma unroll
    for(int cq=0;cq<VEC;++cq){const int c=c0+cq;t[cq]=T(0);
     if(c<h){
      if(cq==lane)t[cq]=taus[c];
      else if(cq>lane){T acc=T(0);
       #pragma unroll
       for(int p=0;p<cq;++p)if(p>=lane)acc=gf_fma(t[p],refl[size_t(c)*SL+c0+p],acc);
       t[cq]=-taus[c]*acc;}}}
    #pragma unroll
    for(int cq=0;cq<VEC;++cq){const int c=c0+cq;if(c<h&&cq>=lane)Tp[size_t(c)*(c+1)/2+r]=t[cq];}}
   __syncwarp();
   if(lane==0)gau_mbar_arrive(gau_smem_addr(td+warp));
  }else{
   // rows c0.. of T12 for block bb: Y = T(rows, jb-1 ..) O(.., block), then T(rows, block) = -Y T22
   gau_mbar_wait(gau_smem_addr(oc+bb),0);gau_mbar_wait(gau_smem_addr(td+bb),0);
   #pragma unroll
   for(int m=0;m<PPL;++m){const int idx=(lane/LPP)+m*(32/LPP),ri=idx%VEC,p=idx/VEC,part=lane%LPP,i=c0+ri;
    T y=T(0);
    if(i<h)for(int qq=i+part;qq<jb;qq+=LPP)y=gf_fma(Tp[size_t(qq)*(qq+1)/2+i],refl[size_t(jb+p)*SL+qq],y);
    #pragma unroll
    for(int o=LPP/2;o;o>>=1)y+=__shfl_xor_sync(0xffffffffu,y,o);
    if(part==0)ys[p*VEC+ri]=y;}
   __syncwarp();
   #pragma unroll
   for(int m=0;m<PPL;++m){const int idx=(lane/LPP)+m*(32/LPP),ri=idx%VEC,jq=idx/VEC,part=lane%LPP,i=c0+ri,j=jb+jq;
    if(part==0&&i<h&&j<h){T acc=T(0);
     #pragma unroll
     for(int p=0;p<VEC;++p)if(p<=jq)acc=gf_fma(ys[p*VEC+ri],Tp[size_t(j)*(j+1)/2+jb+p],acc);
     Tp[size_t(j)*(j+1)/2+i]=-acc;}}
   __syncwarp();
  }
 }
 if(dbg&&lane==0)dbg[warp*4+3]=clock64();
 __syncthreads();
 T*tri=Ts+size_t(pk.tile)*b*b;
 for(int j=warp;j<b;j+=WARPS)for(int i=lane;i<b;i+=32)tri[i+size_t(j)*b]=(j<h&&i<=j)?Tp[size_t(j)*(j+1)/2+i]:T(0);
 if(dbg&&tid==0){dbg[32*4+0]=dbg[warp*4+3];dbg[32*4+1]=clock64();}
 if(tid==0){atomicAdd(kind==2?&w->tt:kind==1?&w->ts:&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);
  const int wk=kind?1:0,owners=(h+VEC-1)/VEC;
  atomicAdd(&w->device_qrv2_products[wk],1ULL);atomicAdd(&w->device_qrv2_owners[wk],(unsigned long long)owners);
  atomicAdd(&w->device_block_peers[wk],(unsigned long long)(32*owners));witness_levels(w,wk,32,1,1,1);
  atomicAdd(&w->device_peer_partials[wk],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}

// ------------------------------------------------------------------ two SMs --------------------------------------
// A cluster of two CTAs per packet; rank r owns columns [r L, (r+1) L), L = warps*VEC. Shared per rank: refl [L][ROWS]
// (rank 1's slot s first receives rank 0's reflector s, then -- after a CTA barrier -- holds its own reflector L+s,
// with O(k, L+s) packed on rows k < L+s), taus [2L], mbarriers [2L], obuf [L][L] (rank 0: O(k,c), k < c < L), S / M.
// Rank 0's mbarriers L..2L-1 are notified remotely by rank 1's producers, so rank 0's owners can form their overlaps
// with rank 1's reflectors through DSM loads and store them into rank 1's slots. Rank 1 builds T.
template<class T,int ITEMS,int VEC,int MAXW>
__global__ __cluster_dims__(2,1,1) __launch_bounds__(MAXW*32,1) void gf_sm2_kernel(T*A,int lda,const TilePacket*packets,T*Vout,int ldv,size_t vstride,
                                                                                  T*tau_out,int fstride,T*Ts,int b,int stage,int kind,int*status,Witness*w){
 namespace cg=cooperative_groups;
 constexpr int ROWS=ITEMS*32;
 const int nt=blockDim.x,WARPS=nt>>5,L=WARPS*VEC,COLS=2*L;
 extern __shared__ __align__(16) unsigned char gau_smem[];
 const int id=blockIdx.x>>1;const TilePacket pk=packets[id];const int rows=pk.rows,h=pk.h;
 const size_t rw=stage?std::max(size_t(L)*ROWS,gf_t_words<T>(h)):size_t(L)*ROWS;
 T*refl=reinterpret_cast<T*>(gau_smem);
 T*taus=refl+rw;
 unsigned long long*mb=reinterpret_cast<unsigned long long*>(taus+COLS);
 T*obuf=reinterpret_cast<T*>(mb+COLS);
 T*Sm=stage?refl:obuf+size_t(L)*L;
 cg::cluster_group cl=cg::this_cluster();
 const unsigned rank=cl.block_rank();
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,c0=int(rank)*L+warp*VEC;
 T*a=A+pk.row+size_t(pk.col)*lda;
 if(tid==0){
  for(int i=0;i<COLS;++i)gau_mbar_init(gau_smem_addr(mb+i),(rank==0)==(i<L)?32:1);
  // rank 1: arm the mbarriers of rank 0's reflectors (ROWS values + tau each) before the peer can stream into them
  if(rank==1)for(int i=0;i<min(L,h);++i)
   asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"::"r"(gau_smem_addr(mb+i)),"r"(int((ROWS+1)*sizeof(T))):"memory");
  asm volatile("fence.mbarrier_init.release.cluster;");}
 cl.sync();
 T*peer_refl=cl.map_shared_rank(refl,rank^1u);
 T*peer_obuf=cl.map_shared_rank(obuf,rank^1u);
 T col[ITEMS][VEC];
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int q=0;q<VEC;++q)col[it][q]=(r<rows&&c0+q<h)?a[r+size_t(c0+q)*lda]:T(0);}
 if(rank==1){
  // rank 0's reflectors, streamed into slots 0..L-1
  const int kend=min(L,h);
  for(int k=0;k<kend;++k){
   gau_mbar_wait_cluster(gau_smem_addr(mb+k),0);
   T v[ITEMS];
   #pragma unroll
   for(int it=0;it<ITEMS;++it){const int r=it*32+lane;v[it]=r>=k?refl[size_t(k)*ROWS+r]:T(0);}
   gf_apply<T,ITEMS,VEC>(col,v,taus[k],0,lane);}
  __syncthreads();   // every warp is done with the received slots before they are overwritten
 }
 // this rank's reflectors to the left of the warp
 const int lb=int(rank)*L,kend=min(c0,h);
 for(int k=lb;k<kend;++k){
  gau_mbar_wait(gau_smem_addr(mb+k),0);
  T v[ITEMS];
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;v[it]=r>=k?refl[size_t(k-lb)*ROWS+r]:T(0);}
  gf_apply<T,ITEMS,VEC>(col,v,taus[k],0,lane);
 }
 #pragma unroll
 for(int i=0;i<VEC;++i){
  const int c=c0+i;if(c>=h)break;
  const T tk=gf_reflector<T,ITEMS,VEC>(col,i,c,rows,lane,status);
  T v[ITEMS];T*slot=refl+size_t(c-lb)*ROWS;
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;v[it]=r==c?T(1):r>c?col[it][i]:T(0);if(r>=c)slot[r]=v[it];}
  if(lane==0)taus[c]=tk;
  if(rank==0){
   // stream slot c and tau into rank 1 (its slot c / taus[c]) against rank 1's expect_tx mbarrier c
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   __syncwarp();
   if(lane==0){
    unsigned src=gau_smem_addr(slot),dst,remb,dtau;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(dst):"r"(src));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(remb):"r"(gau_smem_addr(mb+c)));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 1;":"=r"(dtau):"r"(gau_smem_addr(taus+c)));
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(dst),"r"(src),"r"(int(ROWS*sizeof(T))),"r"(remb):"memory");
    if constexpr(sizeof(T)==4)asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"::"r"(dtau),"r"(__float_as_uint(float(tk))),"r"(remb):"memory");
    else asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b64 [%0], %1, [%2];"::"r"(dtau),"l"(__double_as_longlong(double(tk))),"r"(remb):"memory");}
  }else{
   // notify rank 0 that reflector c is readable in this rank's slot (rank 0 forms its overlaps with it via DSM)
   __syncwarp();
   if(lane==0){unsigned remb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;":"=r"(remb):"r"(gau_smem_addr(mb+c)));
    asm volatile("fence.acq_rel.cluster;\n\tmbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"::"r"(remb):"memory");}
  }
  gau_mbar_arrive(gau_smem_addr(mb+c));
  if(i+1<VEC)gf_apply<T,ITEMS,VEC>(col,v,tk,i+1,lane);
 }
 bool bad=false;
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r<rows){
  #pragma unroll
  for(int q=0;q<VEC;++q)if(c0+q<h){const T y=col[it][q];bad|=!isfinite(y);a[r+size_t(c0+q)*lda]=y;}}}
 if(bad)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 #pragma unroll
 for(int it=0;it<ITEMS;++it){const int r=it*32+lane;
  #pragma unroll
  for(int q=0;q<VEC;++q){const int c=c0+q;col[it][q]=(c<h&&r>=c)?(r==c?T(1):col[it][q]):T(0);}}
 // overlaps O(c0+q, j) with every later reflector j: rank 0 -> obuf (j < L) or rank 1's slot j-L via DSM (j >= L)
 constexpr int SH=5-GfLog<VEC>::v;
 for(int j=c0+1;j<h;++j){
  const bool remote=rank==0&&j>=L;
  if(j>=c0+VEC){if(remote)gau_mbar_wait_cluster(gau_smem_addr(mb+j),0);else gau_mbar_wait(gau_smem_addr(mb+j),0);}
  const T*src=remote?peer_refl+size_t(j-L)*ROWS:refl+size_t(j-lb)*ROWS;
  T d[VEC];
  #pragma unroll
  for(int q=0;q<VEC;++q)d[q]=T(0);
  #pragma unroll
  for(int it=0;it<ITEMS;++it){const int r=it*32+lane;if(r>=j){const T y=src[r];
   #pragma unroll
   for(int q=0;q<VEC;++q)d[q]=gf_fma(col[it][q],y,d[q]);}}
  const T s=gf_reduce_scatter<T,VEC>(d,lane);
  const int q=lane>>SH;
  if((lane&((1<<SH)-1))==0&&c0+q<j){
   if(rank==1)refl[size_t(j-L)*ROWS+c0+q]=s;
   else if(remote)peer_refl[size_t(j-L)*ROWS+c0+q]=s;
   else obuf[size_t(j)*L+c0+q]=s;}
 }
 cl.sync();   // all chains, overlap stores (local and DSM) complete and visible in the cluster
 T*vo=Vout+size_t(id)*vstride;
 const int le=min(h,lb+L);
 for(int c=lb+warp;c<le;c+=WARPS)for(int r=lane;r<ldv;r+=32)vo[r+size_t(c)*ldv]=(r>=c&&r<ROWS)?refl[size_t(c-lb)*ROWS+r]:T(0);
 for(int c=lb+tid;c<le;c+=nt)tau_out[size_t(id)*fstride+c]=taus[c];
 if(rank==1){
  T*tri=Ts+size_t(pk.tile)*b*b;
  auto get=[&](int i,int j)->T{return j<L?peer_obuf[size_t(j)*L+i]:refl[size_t(j-L)*ROWS+i];};
  if(stage){
   for(int c=warp;c<h;c+=WARPS)for(int k=lane;k<c;k+=32)tri[c+size_t(k)*b]=get(k,c);
   __syncthreads();
   gf_t_finish<T>(Sm,taus,h,[&](int i,int j){return tri[j+size_t(i)*b];},tri,b,tid,nt);
  }else gf_t_finish<T>(Sm,taus,h,get,tri,b,tid,nt);
  if(tid==0){atomicAdd(kind==2?&w->tt:kind==1?&w->ts:&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);}
 }
 if(tid==0){atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
 cl.sync();   // rank 0's shared memory (obuf) stays alive until rank 1 has read it
}

// ------------------------------------------------------------------ dispatch -------------------------------------
struct GfConfig{int sms=0,items=0,vec=0,warps=0,maxw=0,stage=0;size_t smem=0;};
template<class T> inline size_t gf_smem(int sms,int items,int vec,int warps,int h,int stage){
 const size_t rows=size_t(items)*32,L=size_t(warps)*vec,cols=sms==1?L:2*L;
 const size_t refl=(sms==1?cols:L)*rows,taus=cols+(cols&1),mbw=cols*8/sizeof(T),ob=sms==2?L*L:0,tw=gf_t_words<T>(h);
 return (stage?std::max(refl,tw)+taus+mbw+ob:refl+taus+mbw+ob+tw)*sizeof(T)+64;}
// Geometry of one (sms, ITEMS, VEC, MAXW) candidate for a rows x h packet; sms = 0 when it does not fit.
template<class T> inline GfConfig gf_shape(int sms,int items,int vec,int maxw,int rows,int h,size_t cap){
 GfConfig g;if(items*32<rows)return g;
 const int local=sms==1?h:(h+1)/2,warps=(local+vec-1)/vec;
 if(warps<1||warps>maxw)return g;
 if(sms==2&&warps*vec>=h)return g;          // rank 1 must own columns
 if(sms==1){const size_t s=gf_sm1_smem<T>(items,vec,warps);if(s<=cap){g.sms=1;g.items=items;g.vec=vec;g.warps=warps;g.maxw=maxw;g.smem=s;}return g;}
 for(int stage=0;stage<2;++stage){const size_t s=gf_smem<T>(sms,items,vec,warps,h,stage);
  if(s>cap)continue;
  g.sms=sms;g.items=items;g.vec=vec;g.warps=warps;g.maxw=maxw;g.stage=stage;g.smem=s;return g;}
 return g;}
inline size_t gf_shared_cap(){static size_t cap=0;if(!cap){int d,o=0;CU(cudaGetDevice(&d));CU(cudaDeviceGetAttribute(&o,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));cap=size_t(o);}return cap;}
// Launches (or, dry, only checks) one instantiation; false when it spills registers or exceeds shared capacity.
template<class T,int ITEMS,int VEC,int MAXW,int SMS>
inline bool gf_try(GfConfig&out,int rows,int h,T*A,int lda,const TilePacket*pk,int count,T*V,int ldv,size_t vstride,T*tau,int fstride,T*Ts,int b,int kind,int*status,Witness*w,cudaStream_t st,bool dry){
 auto kernel=SMS==1?gf_sm1_kernel<T,ITEMS,VEC,MAXW>:gf_sm2_kernel<T,ITEMS,VEC,MAXW>;
 static bool ok=false,spill=false;static size_t cap=0,set=0;
 if(!ok){cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,kernel));spill=attr.localSizeBytes>64;cap=gf_shared_cap()-attr.sharedSizeBytes-1024;ok=true;}
 if(spill)return false;
 const GfConfig g=gf_shape<T>(SMS,ITEMS,VEC,MAXW,rows,h,cap);
 if(!g.sms)return false;
 out=g;if(dry)return true;
 if(g.smem>set){CU(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,int(g.smem)));set=g.smem;}
 kernel<<<count*SMS,g.warps*32,g.smem,st>>>(A,lda,pk,V,ldv,vstride,tau,fstride,Ts,b,g.stage,kind,status,w);CU(cudaGetLastError());
 return true;
}
// One launch for `count` packets (rows <= `rows`, h columns): chain + overlaps + T. Candidates in order of padding
// (smallest ITEMS first), then VEC 8 before 4. Returns the executed config; throws before any write when no
// single/two-SM register panel fits this shape (the caller then uses the gmem chain).
template<class T>
inline GfConfig launch_gf_sm(int sms,T*A,int lda,const TilePacket*pk,int count,int rows,int h,T*V,int ldv,size_t vstride,T*tau,int fstride,T*Ts,int b,int kind,int*status,Witness*w,cudaStream_t st,bool dry){
 if(h<1||h>128||rows<h||count<1||b<h||(sms!=1&&sms!=2))throw std::runtime_error("gf_shape_before_modify");
 GfConfig g;
#define TQR_GF(I,Vv,W) if(sms==1?gf_try<T,I,Vv,W,1>(g,rows,h,A,lda,pk,count,V,ldv,vstride,tau,fstride,Ts,b,kind,status,w,st,dry):gf_try<T,I,Vv,W,2>(g,rows,h,A,lda,pk,count,V,ldv,vstride,tau,fstride,Ts,b,kind,status,w,st,dry))return g;
 if constexpr(sizeof(T)==4){
  TQR_GF(4,8,16)TQR_GF(8,8,16)TQR_GF(12,8,8)TQR_GF(12,4,16)TQR_GF(16,8,8)TQR_GF(16,4,16)TQR_GF(20,8,8)TQR_GF(20,4,16)TQR_GF(24,4,12)TQR_GF(32,4,12)
 }else{
  TQR_GF(4,8,16)TQR_GF(4,4,16)TQR_GF(8,8,8)TQR_GF(8,4,16)TQR_GF(12,4,12)TQR_GF(16,4,12)
 }
#undef TQR_GF
 throw std::runtime_error("gf_no_register_panel_before_modify");
}
}
