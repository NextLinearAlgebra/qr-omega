#pragma once
// Panel factorization (GE) split across thread blocks, with minipanel-blocked WY updates.
#include "cooperative_packet.cuh"
namespace tqr {
// Minipanel-staged carrier -- "minipanels of 16, blocked WY fused at the minipanel boundary, T
// composed via Eq compose".
//
// WHY. That term is why b=128 loses even with the cheaper panel (8h's negative: the panel grows
// faster than the trailing apply shrinks), and the trailing apply is 56.5% of the factorization with
// no other lever.
//
// WHAT. The elimination advances in minipanels of PW columns. What changes is WHICH products carry a
// per-column carrier: inside a minipanel the dots and rank-1 updates are confined to the minipanel's
// own PW columns, so the per-column cost is O(PW) instead of O(h); the deferred work on the rest of
// the panel is discharged once per minipanel as ONE rank-PW block update, which is the same algebra
// in GEMM-like form (each A element is read once for PW fused multiply-adds instead of once per
// reflector).
//
// T (Eq compose). The diagonal block T2 of each minipanel is still built column by column by the
// same recursion. The off-diagonal block follows the standard blocked identity Q1 Q2 = I - [V1 V2]
// [T1, -T1 (V1^T V2) T2; 0, T2] [V1 V2]^T, i.e. T12 = -T1 (V1^T V2) T2 -- and V1^T V2 needs no
// product of its own, because it is exactly the j < m0 part of the SAME cross-CTA product P =
// V_cur^T A[:, j outside the minipanel] that the trailing update already forms. One product, both
// consumers.
//
// UNIT DIAGONAL. The block update needs V with its implicit unit diagonal, so the reflector diagonal
// is stored as 1 during the factorization and the R diagonal beta -- which every CTA already
// computes redundantly -- is held in `betas` and substituted at writeback. Storage is otherwise
// identical.
//
// COST OF THE BOUNDARY. Two extra grid barriers per minipanel (publish/reduce and reduce/read); at
// PW=16 that is 0.125 barrier per column against the one barrier per column the protocol already
// pays. The reduction is distributed -- each CTA combines a disjoint slice of P and commits it once
// -- because a full P is PW*h*groups partials and having every CTA walk all of them would cost more
// than the update it feeds.
//
// This kernel requires count==1 (one packet spanning the panel height), which is the only case the
// engine dispatches a carrier for and the only case where section 10's crossover table says it wins.
//
// WINDOW (`w0`, `sw`). The kernel stages and eliminates the column WINDOW [w0, w0+sw) of the panel
// rather than the whole panel. With a window the footprint is ceil(rows/groups)*(sw+1) and c is free
// again. The work that leaves the kernel -- the deferred update of the panel columns beyond the
// window and the off-diagonal block of T against earlier windows -- is Eq apply and Eq compose ONE
// LEVEL UP, issued by the host between windows (tiled_engine.cuh, cooperative_ge_batch). sw==h
// reproduces the unwindowed kernel exactly, which is the bit-identity gate.
inline size_t coop_mini_shared_bytes(int local_height,int h,int pw,size_t word){
 return (size_t(local_height)*(size_t(h)+1)+size_t(pw)*size_t(h))*word;
}
inline size_t coop_mini_partial_slots(int groups,int h,int pw){
 // Two generations of (2 + 2*PW) slots, but the column-scaling prologue
 // publishes h slots per group in the same region, so the region is whichever
 // is larger. (A literal here is what 8e and the coop_partial bug both were.)
 const size_t generations=size_t(2)*(size_t(2)+size_t(2)*pw);
 return size_t(groups)*(generations>size_t(h)?generations:size_t(h));
}
inline size_t coop_mini_boundary_slots(int groups,int h,int pw){
 return size_t(groups)*size_t(pw)*size_t(h)+size_t(pw)*size_t(h);
}
inline size_t coop_mini_total_slots(int groups,int h,int pw){
 return coop_mini_partial_slots(groups,h,pw)+coop_mini_boundary_slots(groups,h,pw);
}
// Same scaled-norm/dot preparation as cooperative_prepare, restricted to the [j0, j0+nw) column window. Slot
// layout per group: [0] max, [1] sum of squares, [2+jj] dot, [2+nw+jj] pivot-row entry. TQR_PANEL_INLINE --
// the per-column preparation is on every column's critical path between two grid
//   barriers; a non-inlined call adds the ABI round trip and blocks scheduling across it.
// TQR_PANEL_LIVE -- after column k's barrier, the cross-CTA dot combines for already-factored columns
//   j < k are consumed only by part 0 (the T recursion); other parts skip them, and the live columns
//   j > k are dealt to warps starting at k+1 so no warp carries a dead combine ahead of a live one.
// Both are below the moves: the same partials, the same combine over the same disjoint row sets, in the same
// order, for every value that is consumed (bit-identical results).
#ifndef TQR_PANEL_INLINE
#define TQR_PANEL_INLINE 0
#endif
#ifndef TQR_PANEL_LIVE
#define TQR_PANEL_LIVE 0
#endif
//   (scaled-norm partials sigma/ssq, the pivot-row entries, this warp's dot partials) together, instead of
//   one dependent L2 round trip after each block reduction.
// TQR_PANEL_UNROLL -- the minipanel boundary's partial P = V_cur^T A (one warp reduction per entry) runs
//   four independent entries per warp iteration, so their load/FMA/shuffle chains overlap.
#ifndef TQR_PANEL_PREFETCH
#define TQR_PANEL_PREFETCH 1
#endif
// TQR_PANEL_PCOLS -- the same boundary partial, one warp per A column j with the nw V columns inner: each
//   entry P[c][j] keeps its lane-strided sum and shuffle tree (bit-identical); A[:, j] is read once for nw
//   entries and a warp's serial chain is sw/warps columns instead of nw sw/warps entries.
#ifndef TQR_PANEL_UNROLL
#define TQR_PANEL_UNROLL 0
#endif
// TQR_PANEL_WARPNORM -- the scaled-norm combine (max of the sigma partials, then the rescaled ssq sum) is done
//  by warp 0 alone, replaying tile_reduce's exact trees (per-32 warp trees, then the tree over warp sums
//  with exact zeros for the empty warps), then ONE broadcast barrier instead of four. Bit-identical.
// TQR_PANEL_PLEAN -- the boundary partial P is instruction-bound (SASS: a runtime idx/sw division and a
//  per-row diagonal-mask compare with predicated loads around ~4 FMAs per entry). Step (c, j) without the
//  division, and drop the mask on CTAs whose rows all lie below the minipanel (every CTA but the diagonal
//  one): the same products summed in the same order (bit-identical).
#ifndef TQR_PANEL_PLEAN
#define TQR_PANEL_PLEAN 1
#endif
#ifndef TQR_PANEL_WARPNORM
#define TQR_PANEL_WARPNORM 0
#endif
// JT target columns per warp in the register-blocked boundary partial (0 = off, the PLEAN loop).
#ifndef TQR_PANEL_PBLK
#define TQR_PANEL_PBLK 0
#endif
// The blocked partial's 32-value transposed butterfly + lane-parallel stores + dead-group skip (needs PBLK).
#ifndef TQR_PANEL_PXPOSE
#define TQR_PANEL_PXPOSE 0
#endif
// prepare's fp32 single-pass dots divide every staged value by the local norm inside each warp's dot loop (the
// probe's heaviest instruction line: 16 warps x 12 IEEE divisions per lane per column). XSPRE divides each value once
// after the block norm (one extra CTA barrier); the dots read the same quotient: bit-identical.
#ifndef TQR_PANEL_XSPRE
#define TQR_PANEL_XSPRE 0
#endif
// TDEFER -- the diagonal block T2 of a minipanel is built at its boundary by one warp of every CTA from the combined
// dots every CTA already forms, instead of part 0 extending it column by column behind a CTA barrier (block 0 then
// reached every grid barrier last). Same recursion, same order: bit-identical T. Part 0 still commits the block to
// tri once.
#ifndef TQR_PANEL_TDEFER
#define TQR_PANEL_TDEFER 0
#endif
// AVP -- A[:, j>=m1] -= V_cur P with a warp's 4-column block of P held in registers and lanes over rows (row
// slices when there are fewer column blocks than warps); the same FMA order per element: bit-identical.
#ifndef TQR_PANEL_AVP
#define TQR_PANEL_AVP 0
#endif
#ifndef TQR_PANEL_PCOLS
#define TQR_PANEL_PCOLS 0
#endif
// Development phase timers: block 0, thread 0 accumulates clock64 per phase.
#ifdef TQR_PANEL_PHASES
__device__ unsigned long long tqr_panel_phase[16];
#define TQR_PH(i) do{if(part==0&&tid==0){const long long _n=clock64();atomicAdd(&tqr_panel_phase[i],(unsigned long long)(_n-_ph));_ph=_n;}}while(0)
#define TQR_PH_START long long _ph=clock64()
#define TQR_PB(i,t0) do{if(part==0&&tid==0)atomicAdd(&tqr_panel_phase[i],(unsigned long long)(clock64()-(t0)));}while(0)
#define TQR_CLK clock64()
#else
#define TQR_PB(i,t0) do{}while(0)
#define TQR_CLK 0LL
#define TQR_PH(i) do{}while(0)
#define TQR_PH_START do{}while(0)
#endif
#if TQR_PANEL_INLINE
#define TQR_PANEL_FN __device__ __forceinline__
#else
#define TQR_PANEL_FN __device__ __noinline__
#endif
// single-pass column norm for fp32 storage. sigma_i = sum of x^2 over the CTA's rows is accumulated in DOUBLE: an
// fp32 value squared can neither overflow nor underflow fp64, so the scaled-norm combine of the paper (partials
// combined with their scale factors) holds with scale 1 and the max pass disappears. The double partial travels as
// raw bits in the two existing float slots (sig = low word, ssq = high word), so the partial buffer layout is
// unchanged. Dots use the unscaled column (xs = x) and ratio = 1.
#ifndef TQR_PANEL_SSQ64
#define TQR_PANEL_SSQ64 1
#endif
#if TQR_PANEL_FUSEDLA && TQR_PANEL_SSQ64
#error "FUSEDLA needs scaled-dot handoff after the SSQ64 underflow repair"
#endif
// Initial column normalization does not protect tiny later tails.
#ifndef TQR_PANEL_SSQ64_FP64
#define TQR_PANEL_SSQ64_FP64 0
#endif
template<class T> constexpr bool coop_ssq64(){return TQR_PANEL_SSQ64&&(std::is_same_v<T,float>||(TQR_PANEL_SSQ64_FP64&&std::is_same_v<T,double>));}
#ifndef TQR_PANEL_FUSEDLA
#define TQR_PANEL_FUSEDLA 0
#endif
__device__ __forceinline__ double coop_block_sum_d(double v,double*redd){
 const int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;
 #pragma unroll
 for(int o=16;o;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
 if(lane==0)redd[warp]=v;__syncthreads();
 double t=0;if(warp==0){t=lane<warps?redd[lane]:0.0;
  #pragma unroll
  for(int o=16;o;o>>=1)t+=__shfl_down_sync(0xffffffffu,t,o);
  if(lane==0)redd[32]=t;}
 __syncthreads();t=redd[32];__syncthreads();return t;}
__device__ __forceinline__ void coop_put_d(float*lo,float*hi,double d){const unsigned long long u=__double_as_longlong(d);*lo=__uint_as_float(unsigned(u));*hi=__uint_as_float(unsigned(u>>32));}
__device__ __forceinline__ double coop_get_d(float lo,float hi){return __longlong_as_double((long long)((unsigned long long)__float_as_uint(hi)<<32|__float_as_uint(lo)));}
// The fp64 norm partial: raw bits in two float slots for fp32 storage, the double itself for fp64 storage.
template<class T> __device__ __forceinline__ void coop_put_norm(T*lo,T*hi,double d){if constexpr(std::is_same_v<T,double>){*lo=d;*hi=0;}else coop_put_d((float*)lo,(float*)hi,d);}
template<class T> __device__ __forceinline__ double coop_get_norm(T lo,T hi){if constexpr(std::is_same_v<T,double>)return double(lo);else return coop_get_d(lo,hi);}
// A well-ranged raw sum produces (sqrt(sum), 1). Tiny/huge/nonfinite sums recompute with max scaling BEFORE
// making a reflector. 1.0-1.8% faster whole QR).
#ifndef TQR_PANEL_FP64_FAST_SCALED_NORM
#define TQR_PANEL_FP64_FAST_SCALED_NORM 1
#endif
template<class T> TQR_PANEL_FN void cooperative_prepare_window(
 const T*a,int ld,int local_rows,int first,int w0,int j0,int nw,int k,T*out,int outs,T*red,T*xs){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 // The pivot's MATRIX row is w0+k; its column inside the staged window is k.
 const int krow=w0+k;int start=max(0,krow+1-first);T mx=0;
 if constexpr(std::is_same_v<T,double>&&TQR_PANEL_FP64_FAST_SCALED_NORM&&!TQR_PANEL_SSQ64_FP64){
  __shared__ double redd[33];double d=0;
  for(int r=start+tid;r<local_rows;r+=blockDim.x){const double x=a[r+size_t(k)*ld];xs[r]=x;d+=x*x;}
  d=coop_block_sum_d(d,redd);
  // Conservative normal range, leaving ample room for the later scaled
  // combine. The fallback also distinguishes an all-zero tail from squared
  // underflow. Every branch is uniform within this contraction peer.
  if(d>=0x1p-970&&d<=0x1p970){
   mx=sqrt(d);
   for(int r=start+tid;r<local_rows;r+=blockDim.x)xs[r]/=mx;
   __syncthreads();
   if(tid==0){out[0]=mx;out[outs]=1;}
  }else{
   for(int r=start+tid;r<local_rows;r+=blockDim.x)mx=max(mx,absval(xs[r]));
   mx=tile_reduce<T,true>(mx,red);T sum=0;
   if(mx)for(int r=start+tid;r<local_rows;r+=blockDim.x){T x=xs[r]/mx;xs[r]=x;sum+=x*x;}
   sum=tile_reduce<T,false>(sum,red);
   if(tid==0){out[0]=mx;out[outs]=sum;}
  }
 }else if constexpr(coop_ssq64<T>()){
  __shared__ double redd[33];double d=0;
  for(int r=start+tid;r<local_rows;r+=blockDim.x){const T x=a[r+size_t(k)*ld];xs[r]=x;d+=double(x)*double(x);}
  d=coop_block_sum_d(d,redd);mx=T(sqrt(d));if(tid==0)coop_put_norm<T>(&out[0],&out[outs],d);
#if TQR_PANEL_XSPRE
  if(mx){for(int r=start+tid;r<local_rows;r+=blockDim.x)xs[r]=xs[r]/mx;__syncthreads();}
#endif
 }else{
 for(int r=start+tid;r<local_rows;r+=blockDim.x)mx=max(mx,absval(a[r+size_t(k)*ld]));
 mx=tile_reduce<T,true>(mx,red);T sum=0;
 if(mx)for(int r=start+tid;r<local_rows;r+=blockDim.x){T x=a[r+size_t(k)*ld]/mx;xs[r]=x;sum+=x*x;}
 sum=tile_reduce<T,false>(sum,red);if(tid==0){out[0]=mx;out[outs]=sum;}}
 for(int jj=warp;jj<nw;jj+=warps){int j=j0+jj;T dot=0;
  if(j!=k&&mx)for(int r=start+lane;r<local_rows;r+=32)dot+=((coop_ssq64<T>()&&!TQR_PANEL_XSPRE)?xs[r]/mx:xs[r])*a[r+size_t(j)*ld];
  dot=warp_add(dot);if(lane==0){out[size_t(2+jj)*outs]=dot;out[size_t(2+nw+jj)*outs]=(krow>=first&&krow<first+local_rows)?a[krow-first+size_t(j)*ld]:T(0);}
 }
}
// These boundary reductions have only a lane-zero consumer. Preserve the exact shuffle-add tree and
// omit the final broadcast to inactive lanes.
#ifndef TQR_PANEL_BOUNDARY_NOBCAST
#define TQR_PANEL_BOUNDARY_NOBCAST 1
#endif
template<class T> __device__ __forceinline__ T panel_boundary_sum(T x){
#if TQR_PANEL_BOUNDARY_NOBCAST
 for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffffu,x,d);
 return x;
#else
 return warp_add(x);
#endif
}
template<class T,int PW> __global__ void cooperative_ge_mini(T*A,int lda,const TilePacket*packets,
 int groups,int local_height,int sw,int w0,T*Ts,int b,T*partial,T*bpart,T*bfull,int*status,Witness*w){
 auto grid=cooperative_groups::this_grid();const int part=blockIdx.x;const long long tq_k0=TQR_CLK;(void)tq_k0;
 auto p=packets[0];const int first=part*local_height,rows=max(0,min(local_height,p.rows-first));
 const int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 T*global=A+p.row+size_t(p.col+w0)*lda,*tri=Ts+size_t(p.tile)*b*b;
 extern __shared__ __align__(16) unsigned char memory[];T*a=reinterpret_cast<T*>(memory);
 T*xs=a+size_t(local_height)*sw;T*P=xs+local_height;
 __shared__ T red[32],scales[128],g[PW],betas[128],ratio[coop_partition_limit],t2[PW*PW];
#if TQR_PANEL_TDEFER
 __shared__ T gm[PW*PW],taus_s[PW];   // combined dots v_j^T v_k (j < k) and tau of the current minipanel
#endif
 const size_t stride=2+2*PW,generation=size_t(groups)*stride;
 T*base=partial,*mine=base+part;
 for(int i=tid;i<rows*sw;i+=blockDim.x)a[i%rows+size_t(i/rows)*local_height]=global[first+i%rows+size_t(i/rows)*lda];
 // Only the FIRST window clears T: later windows must keep the blocks their
 // predecessors wrote (the diagonal blocks and the composed off-diagonals).
 if(!w0&&part==0)for(int i=tid;i<b*b;i+=blockDim.x)tri[i]=0;
 __syncthreads();
 for(int j=warp;j<sw;j+=warps){T mx=0;for(int r=lane;r<rows;r+=32)mx=max(mx,absval(a[r+size_t(j)*local_height]));mx=warp_maximum(mx);if(lane==0)mine[size_t(j)*groups]=mx;}
 grid.sync();
 for(int j=warp;j<sw;j+=warps){T mx=0;for(int i=lane;i<groups;i+=32)mx=max(mx,base[size_t(j)*groups+i]);mx=warp_maximum(mx);if(lane==0)scales[j]=mx;for(int r=lane;r<rows;r+=32)if(mx)a[r+size_t(j)*local_height]/=mx;}
 // Every reader of the scaling generation retires before it becomes partials.
 grid.sync();
 TQR_PB(7,tq_k0);
 for(int m0=0;m0<sw;m0+=PW){
  const int m1=min(m0+PW,sw),nw=m1-m0;
  cooperative_prepare_window(a,local_height,rows,first,w0,m0,nw,m0,mine,groups,red,xs);
  for(int k=m0;k<m1;++k){
   const int kk=k-m0,krow=w0+k;
   TQR_PH_START;
   grid.sync();
   TQR_PH(0);
   T*read=base+size_t(kk%2)*generation;const T*sig=read,*ssq=read+groups;
   const int owner=krow/local_height;
#if TQR_PANEL_PREFETCH
   // All of this column's combine inputs in flight at once (the loads the code below would issue
   // one reduction apart). One partial per thread when groups <= threads (the menu's case).
   const bool pf=groups<=int(blockDim.x)&&groups<=128;
   T pf_sig=0,pf_ssq=0,pf_dot[4]={0,0,0,0},pf_row=0;
   if(pf){if(tid<groups){pf_sig=sig[tid];pf_ssq=ssq[tid];}
    if(warp<nw){
     #pragma unroll
     for(int m=0;m<4;++m){const int i=lane+32*m;if(i<groups)pf_dot[m]=read[size_t(2+warp)*groups+i];}
     pf_row=read[size_t(2+nw+warp)*groups+owner];}}
   const T alpha=read[size_t(2+nw+kk)*groups+owner];
#if TQR_PANEL_WARPNORM
   T tail=0,s=0;
   if(pf){
    if(warp==0){T sg[4],sq[4];
     #pragma unroll
     for(int m=0;m<4;++m){const int i=lane+32*m;sg[m]=i<groups?sig[i]:T(0);sq[m]=i<groups?ssq[i]:T(0);}
     // max: tile_reduce<max> = warp trees over tid groups of 32, then the tree over the warp maxima
     T wm[4];
     #pragma unroll
     for(int m=0;m<4;++m)wm[m]=warp_maximum(max(T(0),sg[m]));
     T x=lane<int(blockDim.x/32)?(lane<4?wm[lane]:T(0)):T(0);
     x=(lane==0?wm[0]:lane==1?wm[1]:lane==2?wm[2]:lane==3?wm[3]:x);
     const T tl=warp_maximum(lane<int(blockDim.x/32)?x:T(0));
     const T sc=max(tl,absval(alpha));T ws[4];
     #pragma unroll
     for(int m=0;m<4;++m){const int i=lane+32*m;T term=0;if(tl&&i<groups){T r=sg[m]/sc;ratio[i]=r;term=sq[m]*r*r;}ws[m]=warp_add(term);}
     T y=(lane==0?ws[0]:lane==1?ws[1]:lane==2?ws[2]:lane==3?ws[3]:T(0));
     const T sm=warp_add(lane<int(blockDim.x/32)?y:T(0));
     if(lane==0){red[0]=tl;red[1]=sm;}}
    __syncthreads();tail=red[0];s=red[1];__syncthreads();   // second barrier: red is reused by the next reduction
   }else{
    for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,sig[i]);
    tail=tile_reduce<T,true>(tail,red);
    const T scale0=max(tail,absval(alpha));
    if(tail)for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale0;ratio[i]=r;s+=ssq[i]*r*r;}
    s=tile_reduce<T,false>(s,red);}
   const T scale=max(tail,absval(alpha));
#else
#if TQR_PANEL_SSQ64
   T tail=0,scale=1,s=0;double sd=0;(void)s;
   if constexpr(coop_ssq64<T>()){
    __shared__ double redd[33];
    if(pf){if(tid<groups){sd=coop_get_norm<T>(pf_sig,pf_ssq);ratio[tid]=T(1);}}
    else for(int i=tid;i<groups;i+=blockDim.x){sd+=coop_get_norm<T>(sig[i],ssq[i]);ratio[i]=T(1);}
    sd=coop_block_sum_d(sd,redd);tail=sd>0?T(1):T(0);
    // The norm needs no max pass, but dots still need a scaled column:
    // unscaled fp32 products underflow for nearly dependent columns. Match
    // prepare's local norm and combine the scaled dots in the original law.
    scale=max(T(sqrt(sd)),absval(alpha));
    for(int i=tid;i<groups;i+=blockDim.x){const double d=coop_get_norm<T>(sig[i],ssq[i]);
     ratio[i]=scale?T(sqrt(d))/scale:T(0);}
    __syncthreads();
   }else{
    for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,pf&&i==tid?pf_sig:sig[i]);
    tail=tile_reduce<T,true>(tail,red);scale=max(tail,absval(alpha));
    if(tail)for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale;ratio[i]=r;s+=ssq[i]*r*r;}}
#else
   T tail=0;if(pf){tail=max(tail,pf_sig);}else for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,sig[i]);
   tail=tile_reduce<T,true>(tail,red);
   const T scale=max(tail,absval(alpha));
   T s=0;if(tail){if(pf){if(tid<groups){T r=pf_sig/scale;ratio[tid]=r;s+=pf_ssq*r*r;}}
    else for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale;ratio[i]=r;s+=ssq[i]*r*r;}}
#endif
#endif
#else
   T tail=0;for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,sig[i]);
   tail=tile_reduce<T,true>(tail,red);
   const T alpha=read[size_t(2+nw+kk)*groups+owner];
   const T scale=max(tail,absval(alpha));
   T s=0;if(tail)for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale;ratio[i]=r;s+=ssq[i]*r*r;}
#endif
#if !(TQR_PANEL_PREFETCH && TQR_PANEL_WARPNORM)
#if TQR_PANEL_SSQ64
   if constexpr(!coop_ssq64<T>())
#endif
   s=tile_reduce<T,false>(s,red);
#endif
   TQR_PH(1);
   T beta=alpha,tau=0,den=1;
#if TQR_PANEL_SSQ64
   if constexpr(coop_ssq64<T>()){
    if(tail){const double an=double(alpha);double bn=-sqrt(an*an+sd);if(an<0)bn=-bn;beta=T(bn);den=T((an-bn)/double(scale));tau=T(1.0-an/bn);}
   }else
#endif
   if(tail){const T an=alpha/scale;T sum=an*an+s,bn=-sqrt(sum);if(an<0)bn=-bn;beta=scale*bn;den=an-bn;tau=1-an/bn;}
   if(tid==0)betas[k]=beta;
#if TQR_PANEL_TDEFER
   if(tid==0)taus_s[kk]=tau;
#endif
   if(tau)for(int r=max(0,krow+1-first)+tid;r<rows;r+=blockDim.x)a[r+size_t(k)*local_height]=(a[r+size_t(k)*local_height]/scale)/den;
   __syncthreads();
   TQR_PH(2);
#if TQR_PANEL_FUSEDLA && !TQR_PANEL_LIVE
   // fused look-ahead column update.
   // (a) combine every column's value; (b) update column k+1 FIRST and form its fp64 sigma and unscaled copy;
   // (c) each warp updates its column j > k+1 and accumulates dot(x_{k+1}, a_j) in the same pass (columns j <= k only
   // take their dot). The partial slots written are exactly cooperative_prepare_window's, so the separate "prepare
   // next column" pass disappears. Requires the single-pass norm (TQR_PANEL_SSQ64, fp32 storage).
   if constexpr(coop_ssq64<T>()){
    __shared__ T vals[PW];__shared__ double reddf[33];
    for(int jj=warp;jj<nw;jj+=warps){const int j=m0+jj;if(j==k)continue;T dot=0;
#if TQR_PANEL_PREFETCH
     const bool mine=pf&&jj==warp;
     if(tau){if(mine){
       #pragma unroll
       for(int m=0;m<4;++m){const int i=lane+32*m;if(i<groups)dot+=pf_dot[m]*ratio[i];}}
      else for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+jj)*groups+i]*ratio[i];}
     dot=warp_add(dot);T value=(mine?pf_row:read[size_t(2+nw+jj)*groups+owner])+(tau?dot/den:T(0));
#else
     if(tau)for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+jj)*groups+i]*ratio[i];
     dot=warp_add(dot);T value=read[size_t(2+nw+jj)*groups+owner]+(tau?dot/den:T(0));
#endif
     if(lane==0){vals[jj]=value;if(j<k&&part==0)g[jj]=value;}}
    __syncthreads();
    const int kn=k+1;const bool nxt=kn<m1;const int startk=max(0,krow+1-first),startn=max(0,krow+2-first),krn=krow+1;
    const bool pivn=krn>=first&&krn<first+rows;
    T*outn=base+part+size_t((kk+1)%2)*generation;
    if(nxt){const T upd=tau*vals[kk+1];double d=0;
     for(int r=startk+tid;r<rows;r+=blockDim.x){const T v=tau?a[r+size_t(k)*local_height]:T(0);const T x=a[r+size_t(kn)*local_height]-v*upd;
      a[r+size_t(kn)*local_height]=x;if(r>=startn){xs[r]=x;d+=double(x)*double(x);}}
     if(part==owner&&tid==0)a[krow-first+size_t(kn)*local_height]-=upd;
     d=coop_block_sum_d(d,reddf);
     if(tid==0){coop_put_norm<T>(&outn[0],&outn[groups],d);outn[size_t(2+nw+kk+1)*groups]=pivn?a[krn-first+size_t(kn)*local_height]:T(0);}}
    for(int jj=warp;jj<nw;jj+=warps){const int j=m0+jj;if(j==kn)continue;T dot=0;
     if(j>kn){const T upd=tau*vals[jj];
      for(int r=startk+lane;r<rows;r+=32){const T v=tau?a[r+size_t(k)*local_height]:T(0);const T x=a[r+size_t(j)*local_height]-v*upd;
       a[r+size_t(j)*local_height]=x;if(nxt&&r>=startn)dot+=xs[r]*x;}
      if(part==owner&&lane==0)a[krow-first+size_t(j)*local_height]-=upd;}
     else if(nxt){for(int r=startn+lane;r<rows;r+=32)dot+=xs[r]*a[r+size_t(j)*local_height];}
     if(nxt){dot=warp_add(dot);__syncwarp();
      if(lane==0){outn[size_t(2+jj)*groups]=dot;outn[size_t(2+nw+jj)*groups]=pivn?a[krn-first+size_t(j)*local_height]:T(0);}}}
    __syncthreads();
    TQR_PH(3);
    if(part==0){if(tid<kk){T sum=0;for(int jj=tid;jj<kk;++jj)sum+=tri[w0+m0+tid+size_t(w0+m0+jj)*b]*g[jj];tri[w0+m0+tid+size_t(krow)*b]=-tau*sum;}if(tid==0)tri[krow+size_t(krow)*b]=tau;}
    if(part==owner&&tid==0)a[krow-first+size_t(k)*local_height]=T(1);
    __syncthreads();
    TQR_PH(4);TQR_PH(5);
    continue;
   }
#endif
#if TQR_PANEL_LIVE
   // Live combines only: part 0 needs every j != k (T recursion for j < k); the others need j > k.
   const int jj_first=part==0?0:kk+1;
   for(int jj=jj_first+warp;jj<nw;jj+=warps){const int j=m0+jj;if(j==k)continue;T dot=0;
#else
   for(int jj=warp;jj<nw;jj+=warps){const int j=m0+jj;
#if TQR_PANEL_TDEFER==2
    // The warp of the column being factored has no update: it extends T2 by column kk-1 (the recursion's exact loop;
    // column kk-1's dots and tau are complete since the previous step's barrier). The last column is left to the
    // boundary: one step instead of nw.
    if(j==k){if(kk>0){const int c=kk-1;const T tc=taus_s[c];T sum=0;
      if(lane<c){for(int q=lane;q<c;++q)sum+=t2[lane+size_t(q)*nw]*gm[c*PW+q];}
      if(lane<nw)t2[lane+size_t(c)*nw]=lane<c?-tc*sum:(lane==c?tc:T(0));}
     continue;}
#else
    if(j==k)continue;
#endif
    T dot=0;
#endif
#if TQR_PANEL_PREFETCH
    const bool mine=pf&&jj==warp;
    if(tau){if(mine){
      #pragma unroll
      for(int m=0;m<4;++m){const int i=lane+32*m;if(i<groups)dot+=pf_dot[m]*ratio[i];}}
     else for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+jj)*groups+i]*ratio[i];}
    dot=warp_add(dot);T value=(mine?pf_row:read[size_t(2+nw+jj)*groups+owner])+(tau?dot/den:T(0));
#else
    if(tau)for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+jj)*groups+i]*ratio[i];
    dot=warp_add(dot);T value=read[size_t(2+nw+jj)*groups+owner]+(tau?dot/den:T(0));
#endif
#if TQR_PANEL_TDEFER
    if(j<k){if(lane==0)gm[kk*PW+jj]=value;}
#else
    if(j<k){if(part==0&&lane==0)g[jj]=value;}
#endif
    else{T update=tau*value;for(int r=max(0,krow+1-first)+lane;r<rows;r+=32){T v=tau?a[r+size_t(k)*local_height]:T(0);a[r+size_t(j)*local_height]-=v*update;}
     if(part==owner&&lane==0)a[krow-first+size_t(j)*local_height]-=update;
    }
   }__syncthreads();
   TQR_PH(3);
#if TQR_PANEL_TDEFER
   // The reflector diagonal is the unit of V; beta rejoins A at writeback. Nothing below reads it before the next
   // barrier (prepare's dots start below row krow+1), so no CTA barrier here.
   if(part==owner&&tid==0)a[krow-first+size_t(k)*local_height]=T(1);
#else
   if(part==0){if(tid<kk){T sum=0;for(int jj=tid;jj<kk;++jj)sum+=tri[w0+m0+tid+size_t(w0+m0+jj)*b]*g[jj];tri[w0+m0+tid+size_t(krow)*b]=-tau*sum;}if(tid==0)tri[krow+size_t(krow)*b]=tau;}
   // The reflector diagonal is the unit of V; beta rejoins A at writeback.
   if(part==owner&&tid==0)a[krow-first+size_t(k)*local_height]=T(1);
   __syncthreads();
#endif
   TQR_PH(4);
   if(k+1<m1)cooperative_prepare_window(a,local_height,rows,first,w0,m0,nw,k+1,mine+size_t((kk+1)%2)*generation,groups,red,xs);
   TQR_PH(5);
  }
#if TQR_PANEL_TDEFER
  // T2 of this minipanel: warp 0 of every CTA, the per-column recursion's exact loop (lane i owns row i).
  if(warp==0){
   for(int kk=(TQR_PANEL_TDEFER==2?nw-1:0);kk<nw;++kk){const T tk=taus_s[kk];T sum=0;
    if(lane<kk){for(int jj=lane;jj<kk;++jj)sum+=t2[lane+size_t(jj)*nw]*gm[kk*PW+jj];}
    __syncwarp();
    if(lane<nw)t2[lane+size_t(kk)*nw]=lane<kk?-tk*sum:(lane==kk?tk:T(0));
    __syncwarp();}
   if(part==0)for(int i=lane;i<nw*nw;i+=32)tri[w0+m0+i%nw+size_t(w0+m0+i/nw)*b]=t2[i];}
#endif
  if(nw>=sw)break;
  const long long tq_b0=TQR_CLK;(void)tq_b0;
  // ---- minipanel boundary: one product, two consumers ----
  // P = V_cur^T A[:, j outside the minipanel], summed over the disjoint row
  // sets. Slot-major over `groups` so each reader's walk is contiguous
  // (section 19 change (1)); the B side needs no diagonal mask because every
  // row where V_cur is nonzero lies strictly below every j < m0.
#if TQR_PANEL_PBLK
  // register-blocked boundary partial. The probe put the one-entry-per-warp-iteration loop at 73 us of every 104 us
  // boundary (R=24000, 64 groups: 49% of the panel) -- 128 dependent dot chains per warp. Bit-identical: every entry
  // keeps its lane-strided FMA order over r and the same shuffle tree; dead entries (j inside the minipanel) are
  // still published as zeros.
  {constexpr int JT0=TQR_PANEL_PBLK*16/PW/(sizeof(T)==8?2:1),JT=JT0>0?JT0:1;const bool below=first>=w0+m1;   // PW*JT accumulators: 64 fp32 / 32 fp64 at PBLK=4
   for(int j0=warp*JT;j0<sw;j0+=warps*JT){
    T acc[PW][JT];bool live[JT];bool any=false;
    #pragma unroll
    for(int jt=0;jt<JT;++jt){const int j=j0+jt;live[jt]=j<sw&&(j<m0||j>=m1);any|=live[jt];
     #pragma unroll
     for(int c=0;c<PW;++c)acc[c][jt]=0;}
#if TQR_PANEL_PXPOSE
    // A group of JT columns wholly inside the minipanel is dead: publish its zeros without the row walk.
    if(any)
#endif
    for(int r=lane;r<rows;r+=32){const int gr=first+r;T x[JT];
     #pragma unroll
     for(int jt=0;jt<JT;++jt)x[jt]=live[jt]?a[r+size_t(j0+jt)*local_height]:T(0);
     #pragma unroll
     for(int c=0;c<PW;++c)if(c<nw){const T v=(below||gr>=w0+m0+c)?a[r+size_t(m0+c)*local_height]:T(0);
      #pragma unroll
      for(int jt=0;jt<JT;++jt)acc[c][jt]+=v*x[jt];}}
#if TQR_PANEL_PXPOSE
    // transposed butterfly (recursive halving): 32 partials per lane -> lane l holds the lane-sum of partial l after
    // 31 shuffles, instead of a 5-shuffle tree per value with a lane-0 store each (the probe's top instruction
    // lines). A fixed tree over fixed data (deterministic); not bit-identical to the shfl_down tree.
    static_assert((PW*JT)%32==0,"whole 32-value groups");
    T*flat=&acc[0][0];
    #pragma unroll
    for(int g0=0;g0<PW*JT;g0+=32){T*v=flat+g0;
     #pragma unroll
     for(int s=16;s>=1;s>>=1){const bool up=(lane&s)!=0;
      #pragma unroll
      for(int i=0;i<s;++i){const T send=up?v[i]:v[i+s],keep=up?v[i+s]:v[i];v[i]=keep+__shfl_xor_sync(0xffffffffu,send,s);}}
     const int e=g0+lane,c=e/JT,jt=e%JT,j=j0+jt;
     if(c<nw&&j<sw)bpart[size_t(c*sw+j)*groups+part]=(j<m0||j>=m1)?v[0]:T(0);}
#else
    #pragma unroll
    for(int c=0;c<PW;++c)if(c<nw){
     #pragma unroll
     for(int jt=0;jt<JT;++jt)if(j0+jt<sw){const T s=live[jt]?panel_boundary_sum(acc[c][jt]):T(0);
      if(lane==0)bpart[size_t(c*sw+j0+jt)*groups+part]=s;}}
#endif
   }}
#elif TQR_PANEL_PLEAN
  {const bool below=first>=w0+m1;   // every local row lies below every minipanel column: no diagonal mask
   int c=warp/sw,j=warp%sw;
   for(int idx=warp;idx<nw*sw;idx+=warps){
    T acc=0;
    if(j<m0||j>=m1){const int gcol=m0+c,grow=w0+gcol;
     const T*vc=a+size_t(gcol)*local_height,*ac=a+size_t(j)*local_height;
     if(below){for(int r=lane;r<rows;r+=32)acc+=vc[r]*ac[r];}
     else for(int r=lane;r<rows;r+=32){const int gr=first+r;T v=(gr>=grow)?vc[r]:T(0);acc+=v*ac[r];}
    }
    acc=panel_boundary_sum(acc);if(lane==0)bpart[size_t(idx)*groups+part]=acc;
    j+=warps;while(j>=sw){j-=sw;++c;}
   }}
#elif TQR_PANEL_PCOLS
  for(int j=warp;j<sw;j+=warps){
   const bool live=(j<m0||j>=m1);T acc[PW];
   #pragma unroll
   for(int c=0;c<PW;++c)acc[c]=0;
   if(live)for(int r=lane;r<rows;r+=32){const int gr=first+r;const T x=a[r+size_t(j)*local_height];
    #pragma unroll
    for(int c=0;c<PW;++c)if(c<nw){const int gcol=m0+c;const T v=(gr>=w0+gcol)?a[r+size_t(gcol)*local_height]:T(0);acc[c]+=v*x;}}
   #pragma unroll
   for(int c=0;c<PW;++c)if(c<nw){const T x=panel_boundary_sum(acc[c]);if(lane==0)bpart[size_t(c*sw+j)*groups+part]=x;}
  }
#elif TQR_PANEL_UNROLL
  {constexpr int U=4;
   for(int ib=warp;ib<nw*sw;ib+=warps*U){
    T acc[U];int cu[U],ju[U];bool lv[U];
    #pragma unroll
    for(int u=0;u<U;++u){const int idx=ib+u*warps;acc[u]=0;cu[u]=0;ju[u]=0;lv[u]=false;
     if(idx<nw*sw){cu[u]=idx/sw;ju[u]=idx-cu[u]*sw;lv[u]=(ju[u]<m0||ju[u]>=m1);}}
    for(int r=lane;r<rows;r+=32){const int gr=first+r;
     #pragma unroll
     for(int u=0;u<U;++u)if(lv[u]){const int gcol=m0+cu[u];
      T v=(gr>=w0+gcol)?a[r+size_t(gcol)*local_height]:T(0);acc[u]+=v*a[r+size_t(ju[u])*local_height];}}
    #pragma unroll
    for(int u=0;u<U;++u){const int idx=ib+u*warps;if(idx<nw*sw){const T x=panel_boundary_sum(acc[u]);if(lane==0)bpart[size_t(idx)*groups+part]=x;}}
   }}
#else
  for(int idx=warp;idx<nw*sw;idx+=warps){
   const int c=idx/sw,j=idx-c*sw;T acc=0;
   if(j<m0||j>=m1){const int gcol=m0+c,grow=w0+gcol;
    for(int r=lane;r<rows;r+=32){const int gr=first+r;
     T v=(gr>=grow)?a[r+size_t(gcol)*local_height]:T(0);
     acc+=v*a[r+size_t(j)*local_height];}
   }
   acc=panel_boundary_sum(acc);if(lane==0)bpart[size_t(idx)*groups+part]=acc;
  }
#endif
  TQR_PB(8,tq_b0);
  grid.sync();
  // Each CTA owns a disjoint slice of the combine and commits it once.
  for(int idx=part+warp*groups;idx<nw*sw;idx+=warps*groups){
   T acc=0;for(int i=lane;i<groups;i+=32)acc+=bpart[size_t(idx)*groups+i];
   acc=panel_boundary_sum(acc);if(lane==0)bfull[idx]=acc;
  }
  grid.sync();
  TQR_PB(9,tq_b0);
  for(int i=tid;i<nw*sw;i+=blockDim.x)P[i]=bfull[i];
#if !TQR_PANEL_TDEFER
  for(int i=tid;i<nw*nw;i+=blockDim.x)t2[i]=tri[w0+m0+i%nw+size_t(w0+m0+i/nw)*b];
#endif
  __syncthreads();
  // P <- T2^T P, in registers so the triangular sweep needs no extra barrier.
  for(int j=tid;j<sw;j+=blockDim.x){
   if(j>=m0&&j<m1)continue;
   T col[PW];
   #pragma unroll
   for(int c=0;c<PW;++c)col[c]=(c<nw)?P[size_t(c)*sw+j]:T(0);
   // Unrolled over the compile-time width so col[] keeps static indices and
   // stays in registers; a dynamic bound here put it in local memory.
   for(int c=nw-1;c>=0;--c){T acc=0;
    #pragma unroll
    for(int cp=0;cp<PW;++cp)if(cp<=c)acc+=t2[cp+size_t(c)*nw]*col[cp];
    P[size_t(c)*sw+j]=acc;}
  }
  __syncthreads();
  TQR_PB(10,tq_b0);
  // A[:, j>=m1] -= V_cur P.  Four trailing columns per thread so each V entry
  // read from shared feeds four fused multiply-adds instead of one.
  const int ntr=sw-m1;
#if TQR_PANEL_AVP
  if constexpr(sizeof(T)==4&&PW<=16)   // fp32: 4 x PW P values in registers (fp64 / PW=32 keep the per-idx loop below)
  if(ntr>0){const int ng=(ntr+3)/4;int gg=ng,ww=warps;while(ww){const int t=gg%ww;gg=ww;ww=t;}   // gg = gcd(ng, warps)
   const int rs=min(8,warps/gg),items=ng*rs;   // row slices so that items is a multiple of warps (balanced)
   for(int it=warp;it<items;it+=warps){const int jg=it%ng,sl=it/ng,j0=m1+jg*4;
    T pr[PW][4];
    #pragma unroll
    for(int c=0;c<PW;++c){
     #pragma unroll
     for(int u=0;u<4;++u)pr[c][u]=(c<nw&&j0+u<sw)?P[size_t(c)*sw+j0+u]:T(0);}
    for(int r=sl*32+lane;r<rows;r+=32*rs){const int gr=first+r;T acc0=0,acc1=0,acc2=0,acc3=0;
     #pragma unroll
     for(int c=0;c<PW;++c)if(c<nw){const T vc=(gr>=w0+m0+c)?a[r+size_t(m0+c)*local_height]:T(0);
      acc0+=vc*pr[c][0];acc1+=vc*pr[c][1];acc2+=vc*pr[c][2];acc3+=vc*pr[c][3];}
     a[r+size_t(j0)*local_height]-=acc0;
     if(j0+1<sw)a[r+size_t(j0+1)*local_height]-=acc1;
     if(j0+2<sw)a[r+size_t(j0+2)*local_height]-=acc2;
     if(j0+3<sw)a[r+size_t(j0+3)*local_height]-=acc3;}
   }}
  if constexpr(!(sizeof(T)==4&&PW<=16))
#endif
  {
  if(ntr>0){const int jt=(ntr+3)/4;
   for(int idx=tid;idx<rows*jt;idx+=blockDim.x){
    const int r=idx%rows,jg=idx/rows,j0=m1+jg*4,gr=first+r;
    T v[PW];
    #pragma unroll
    for(int c=0;c<PW;++c)v[c]=(c<nw&&gr>=w0+m0+c)?a[r+size_t(m0+c)*local_height]:T(0);
    T acc0=0,acc1=0,acc2=0,acc3=0;
    #pragma unroll
    for(int c=0;c<PW;++c)if(c<nw){const T vc=v[c];const T*pc=P+size_t(c)*sw+j0;
     acc0+=vc*pc[0];
     if(j0+1<sw)acc1+=vc*pc[1];
     if(j0+2<sw)acc2+=vc*pc[2];
     if(j0+3<sw)acc3+=vc*pc[3];}
    a[r+size_t(j0)*local_height]-=acc0;
    if(j0+1<sw)a[r+size_t(j0+1)*local_height]-=acc1;
    if(j0+2<sw)a[r+size_t(j0+2)*local_height]-=acc2;
    if(j0+3<sw)a[r+size_t(j0+3)*local_height]-=acc3;
   }
  }
  }
  TQR_PB(11,tq_b0);
  // T12 = -T1 (V1^T V2) T2, and (V1^T V2) T2 is the j < m0 part of P.
  for(int gi=part+tid*groups;gi<m0*nw;gi+=size_t(blockDim.x)*groups){
   const int i=gi/nw,c=gi-i*nw;T acc=0;
   for(int j=i;j<m0;++j)acc+=tri[w0+i+size_t(w0+j)*b]*P[size_t(c)*sw+j];
   tri[w0+i+size_t(w0+m0+c)*b]=-acc;
  }
  __syncthreads();
  TQR_PB(6,tq_b0);
 }
 const long long tq_w0=TQR_CLK;(void)tq_w0;
 for(int i=tid;i<rows*sw;i+=blockDim.x){int r=i%rows,j=i/rows,gj=w0+j;
  T v=(first+r==gj)?betas[j]:a[r+size_t(j)*local_height];
  if(first+r<=gj){v*=scales[j];if(!isfinite(v))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
  global[first+r+size_t(j)*lda]=v;}
 // One GE per PANEL, sw reflectors per window: the totals are what the
 // unwindowed kernel reports, so the witness keys stay comparable.
 TQR_PB(7,tq_w0);
 if(part==0&&tid==0){if(!w0)atomicAdd(&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)sw);}
 // The host counts groups peers per member per window (window suboperations are recorded separately
 // from the one logical product per member), so nwin window launches report nwin*groups against the
 // host's groups*count*nwin.
 if(tid==0){atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}
// `sw` is the staged column WINDOW and `w0` its first panel column. sw==h is the unwindowed kernel.
// `dry` runs every capability, capacity and residency check and returns without launching. The
// window loop needs the verdict BEFORE it starts, because a refusal after window 0 has already
// rewritten A cannot fall back to another menu entry; keeping one copy of the rules is what a
// separate predicate would have broken (the coop_partial literal and 8e were both second copies).
template<class T,int PW> void launch_cooperative_ge_mini_pw(T*A,int ld,const TilePacket*packets,
 int rows,int h,int w0,int sw,int groups,int threads,T*tri,int b,T*partial,int*status,Witness*w,cudaStream_t stream,bool dry=false,int max_h=128){
 int device;CU(cudaGetDevice(&device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
 if(!prop.cooperativeLaunch||groups<1||groups>coop_partition_limit||h<1||h>max_h||max_h>512||rows<h||threads<32||threads>1024||threads%32)throw std::runtime_error("cooperative_minipanel_capability_before_modify");
 if(sw<1||sw>128||sw>h||w0<0||w0+sw>h)throw std::runtime_error("cooperative_minipanel_window_before_modify");
 int local_height=ceildiv(rows,groups);size_t dynamic=coop_mini_shared_bytes(local_height,sw,PW,sizeof(T));
 cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,cooperative_ge_mini<T,PW>));
 if(dynamic>prop.sharedMemPerBlockOptin-attr.sharedSizeBytes)throw std::runtime_error("cooperative_minipanel_shared_capacity_before_modify");
 CU(cudaFuncSetAttribute(cooperative_ge_mini<T,PW>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dynamic)));
 int occupancy=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cooperative_ge_mini<T,PW>,threads,dynamic));
 if(size_t(groups)>size_t(occupancy)*prop.multiProcessorCount)throw std::runtime_error("cooperative_minipanel_residency_before_modify");
 if(dry)return;
 T*bpart=partial+coop_mini_partial_slots(groups,sw,PW),*bfull=bpart+size_t(groups)*PW*sw;
 void*args[]={&A,&ld,&packets,&groups,&local_height,&sw,&w0,&tri,&b,&partial,&bpart,&bfull,&status,&w};
 CU(cudaLaunchCooperativeKernel((const void*)cooperative_ge_mini<T,PW>,dim3(groups),dim3(threads),args,dynamic,stream));
}
template<class T> void launch_cooperative_ge_mini(T*A,int ld,const TilePacket*packets,
 int rows,int h,int w0,int sw,int pw,int groups,int threads,T*tri,int b,T*partial,int*status,Witness*w,cudaStream_t stream,bool dry=false,int max_h=128){
 if(pw==8)launch_cooperative_ge_mini_pw<T,8>(A,ld,packets,rows,h,w0,sw,groups,threads,tri,b,partial,status,w,stream,dry,max_h);
 else if(pw==16)launch_cooperative_ge_mini_pw<T,16>(A,ld,packets,rows,h,w0,sw,groups,threads,tri,b,partial,status,w,stream,dry,max_h);
 else if(pw==32)launch_cooperative_ge_mini_pw<T,32>(A,ld,packets,rows,h,w0,sw,groups,threads,tri,b,partial,status,w,stream,dry,max_h);
 else throw std::runtime_error("cooperative_minipanel_width_before_modify");
}
}
