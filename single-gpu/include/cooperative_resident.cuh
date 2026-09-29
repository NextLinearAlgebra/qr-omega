#pragma once
// Resident panel carrier: the thread blocks of a panel stay resident across its columns.
#include "cooperative_minipanel.cuh"
namespace tqr {
// RESIDENT panel carrier.
//
// WHY. Under look-ahead the panel of elimination k+1 runs on the high-priority lane while window B of elimination k
// (low priority) holds the SMs. Stream priority does not preempt, so every launch of the panel chain waits for
// window-B CTAs to retire before it gets its SMs. The windowed carrier (cooperative_ge_mini) is h/sw window launches
// plus a host window-boundary apply (Eq apply + Eq compose, several kernels) between windows: up to ~36 launches per
// 128-column panel at sw = 16.
//
// WHAT. Identical algebra to cooperative_ge_mini with sw = h (the unwindowed carrier):
// - Per column: typed partials (norm, dots, pivot row) over the disjoint row sets, fixed-order combine, one grid
//   barrier per column, confined to the current minipanel's PW columns (cooperative_prepare_window + the column loop,
//   unchanged).
// - Minipanel boundary: ONE product P = V_cur^T A[:, j outside the minipanel] summed over the disjoint row sets in a
//   fixed order (Combine), then P <- T2^T P, the trailing update A[:, j >= m1] -= V_cur P, and
//   T12 = -T1 (V1^T V2) T2 from the j < m0 part of P (Eq compose).
// The only difference is WHERE the panel lives: each CTA stages just the current minipanel of its rows in shared
// memory (local_height x PW words, instead of local_height x (sw+1)); the other panel columns stay in global memory
// (L2 at these sizes) and are read and updated in place at each minipanel boundary.
//
// Numerics: the same reflectors, the same T, the same fixed-order combines as the unwindowed carrier; the trailing
// columns receive the PW-blocked updates in the same order. Column scaling is applied per minipanel at staging (the
// windowed carrier's law with sw = PW): V is scale-free, the R part is restored at writeback.
#ifdef TQR_RES_PHASES
__device__ unsigned long long tqr_res_phase[16];
#define RPH(i) do{if(part==0&&tid==0){const long long _n=clock64();atomicAdd(&tqr_res_phase[i],(unsigned long long)(_n-_rph));_rph=_n;}}while(0)
#define RPH_START long long _rph=clock64()
#else
#define RPH(i) do{}while(0)
#define RPH_START do{}while(0)
#endif
inline size_t coop_res_shared_bytes(int local_height,int h,int pw,size_t word){
 return (size_t(local_height)*(size_t(pw)+1)+size_t(pw)*size_t(h))*word;
}
template<class T,int PW> __global__ void cooperative_ge_resident(T*A,int lda,const TilePacket*packets,
 int groups,int local_height,int h,T*Ts,int b,T*partial,T*bpart,T*bfull,int*status,Witness*w,T*packed_v,int ldv){
 auto grid=cooperative_groups::this_grid();const int part=blockIdx.x;
 auto p=packets[0];const int first=part*local_height,rows=max(0,min(local_height,p.rows-first));
 const int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 T*panel=A+p.row+size_t(p.col)*lda,*tri=Ts+size_t(p.tile)*b*b;
 extern __shared__ __align__(16) unsigned char memory[];T*a=reinterpret_cast<T*>(memory);
 T*xs=a+size_t(local_height)*PW;T*P=xs+local_height;   // P: PW x h, column c of V_cur against panel column j at c*h+j
 __shared__ T red[32],scales[PW],g[PW],betas[PW],ratio[coop_partition_limit],t2[PW*PW];
#if TQR_PANEL_TDEFER
 __shared__ T gm[PW*PW],taus_s[PW];   // see cooperative_minipanel.cuh (TDEFER)
#endif
 const size_t stride=2+2*PW,generation=size_t(groups)*stride;
 T*base=partial,*mine=base+part;
 if(part==0)for(int i=tid;i<b*b;i+=blockDim.x)tri[i]=0;
 RPH_START;
 for(int m0=0;m0<h;m0+=PW){
  const int nw=min(PW,h-m0),m1=m0+nw;
  // ---- stage the minipanel (its columns are current: earlier boundaries updated them in place) ----
  for(int i=tid;i<rows*nw;i+=blockDim.x)a[i%rows+size_t(i/rows)*local_height]=panel[first+i%rows+size_t(m0+i/rows)*lda];
  __syncthreads();
  RPH(0);
  for(int j=warp;j<nw;j+=warps){T mx=0;for(int r=lane;r<rows;r+=32)mx=max(mx,absval(a[r+size_t(j)*local_height]));mx=warp_maximum(mx);if(lane==0)mine[size_t(j)*groups]=mx;}
  grid.sync();
  for(int j=warp;j<nw;j+=warps){T mx=0;for(int i=lane;i<groups;i+=32)mx=max(mx,base[size_t(j)*groups+i]);mx=warp_maximum(mx);if(lane==0)scales[j]=mx;for(int r=lane;r<rows;r+=32)if(mx)a[r+size_t(j)*local_height]/=mx;}
  // Every reader of the scaling generation retires before it becomes partials.
  grid.sync();
  RPH(1);
  // ---- the minipanel's columns: cooperative_ge_mini's column loop with (w0, m0, sw) = (m0, 0, nw) ----
  cooperative_prepare_window(a,local_height,rows,first,m0,0,nw,0,mine,groups,red,xs);
  for(int k=0;k<nw;++k){
   const int kk=k,krow=m0+k;
   grid.sync();
   T*read=base+size_t(kk%2)*generation;const T*sig=read,*ssq=read+groups;
   const int owner=krow/local_height;
   const bool pf=groups<=int(blockDim.x)&&groups<=128;
   T pf_sig=0,pf_ssq=0,pf_dot[4]={0,0,0,0},pf_row=0;
   if(pf){if(tid<groups){pf_sig=sig[tid];pf_ssq=ssq[tid];}
    if(warp<nw){
     #pragma unroll
     for(int m=0;m<4;++m){const int i=lane+32*m;if(i<groups)pf_dot[m]=read[size_t(2+warp)*groups+i];}
     pf_row=read[size_t(2+nw+warp)*groups+owner];}}
   const T alpha=read[size_t(2+nw+kk)*groups+owner];
   T tail=0,scale=1,s=0;double sd=0;
   if constexpr(coop_ssq64<T>()){
    __shared__ double redd[33];
    if(pf){if(tid<groups){sd=coop_get_norm<T>(pf_sig,pf_ssq);ratio[tid]=T(1);}}
    else for(int i=tid;i<groups;i+=blockDim.x){sd+=coop_get_norm<T>(sig[i],ssq[i]);ratio[i]=T(1);}
    sd=coop_block_sum_d(sd,redd);tail=sd>0?T(1):T(0);
    scale=max(T(sqrt(sd)),absval(alpha));
    for(int i=tid;i<groups;i+=blockDim.x){const double d=coop_get_norm<T>(sig[i],ssq[i]);
     ratio[i]=scale?T(sqrt(d))/scale:T(0);}
    __syncthreads();
   }else{
    for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,pf&&i==tid?pf_sig:sig[i]);
    tail=tile_reduce<T,true>(tail,red);scale=max(tail,absval(alpha));
    if(tail)for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale;ratio[i]=r;s+=ssq[i]*r*r;}}
   if constexpr(!coop_ssq64<T>())
   s=tile_reduce<T,false>(s,red);
   T beta=alpha,tau=0,den=1;
   if constexpr(coop_ssq64<T>()){
    if(tail){const double an=double(alpha);double bn=-sqrt(an*an+sd);if(an<0)bn=-bn;beta=T(bn);den=T((an-bn)/double(scale));tau=T(1.0-an/bn);}
   }else
   if(tail){const T an=alpha/scale;T sum=an*an+s,bn=-sqrt(sum);if(an<0)bn=-bn;beta=scale*bn;den=an-bn;tau=1-an/bn;}
   if(tid==0)betas[k]=beta;
#if TQR_PANEL_TDEFER
   if(tid==0)taus_s[kk]=tau;
#endif
   if(tau)for(int r=max(0,krow+1-first)+tid;r<rows;r+=blockDim.x)a[r+size_t(k)*local_height]=(a[r+size_t(k)*local_height]/scale)/den;
   __syncthreads();
   for(int jj=warp;jj<nw;jj+=warps){const int j=jj;
#if TQR_PANEL_TDEFER==2
    if(j==k){if(kk>0){const int c=kk-1;const T tc=taus_s[c];T sum=0;   // idle warp extends T2 (see minipanel)
      if(lane<c){for(int q=lane;q<c;++q)sum+=t2[lane+size_t(q)*nw]*gm[c*PW+q];}
      if(lane<nw)t2[lane+size_t(c)*nw]=lane<c?-tc*sum:(lane==c?tc:T(0));}
     continue;}
#else
    if(j==k)continue;
#endif
    T dot=0;
    const bool mine_=pf&&jj==warp;
    if(tau){if(mine_){
      #pragma unroll
      for(int m=0;m<4;++m){const int i=lane+32*m;if(i<groups)dot+=pf_dot[m]*ratio[i];}}
     else for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+jj)*groups+i]*ratio[i];}
    dot=warp_add(dot);T value=(mine_?pf_row:read[size_t(2+nw+jj)*groups+owner])+(tau?dot/den:T(0));
#if TQR_PANEL_TDEFER
    if(j<k){if(lane==0)gm[kk*PW+jj]=value;}
#else
    if(j<k){if(part==0&&lane==0)g[jj]=value;}
#endif
    else{T update=tau*value;for(int r=max(0,krow+1-first)+lane;r<rows;r+=32){T v=tau?a[r+size_t(k)*local_height]:T(0);a[r+size_t(j)*local_height]-=v*update;}
     if(part==owner&&lane==0)a[krow-first+size_t(j)*local_height]-=update;
    }
   }__syncthreads();
#if TQR_PANEL_TDEFER
   // The reflector diagonal is the unit of V; beta rejoins A at writeback (no reader before the next barrier).
   if(part==owner&&tid==0)a[krow-first+size_t(k)*local_height]=T(1);
#else
   if(part==0){if(tid<kk){T sum=0;for(int jj=tid;jj<kk;++jj)sum+=tri[m0+tid+size_t(m0+jj)*b]*g[jj];tri[m0+tid+size_t(krow)*b]=-tau*sum;}if(tid==0)tri[krow+size_t(krow)*b]=tau;}
   // The reflector diagonal is the unit of V; beta rejoins A at writeback.
   if(part==owner&&tid==0)a[krow-first+size_t(k)*local_height]=T(1);
   __syncthreads();
#endif
   if(k+1<nw)cooperative_prepare_window(a,local_height,rows,first,m0,0,nw,k+1,mine+size_t((kk+1)%2)*generation,groups,red,xs);
  }
  RPH(2);
#if TQR_PANEL_TDEFER
  // T2 of this minipanel (warp 0 of every CTA, the per-column recursion's exact loop); part 0 commits it to tri.
  if(warp==0){
   for(int kk=(TQR_PANEL_TDEFER==2?nw-1:0);kk<nw;++kk){const T tk=taus_s[kk];T sum=0;
    if(lane<kk){for(int jj=lane;jj<kk;++jj)sum+=t2[lane+size_t(jj)*nw]*gm[kk*PW+jj];}
    __syncwarp();
    if(lane<nw)t2[lane+size_t(kk)*nw]=lane<kk?-tk*sum:(lane==kk?tk:T(0));
    __syncwarp();}
   if(part==0)for(int i=lane;i<nw*nw;i+=32)tri[m0+i%nw+size_t(m0+i/nw)*b]=t2[i];}
#endif
  if(nw<h){
   // ---- minipanel boundary: one product, two consumers, the panel's other columns in global memory ---- P[c][j] =
   // V_cur[:,c]^T A[:,j] over this CTA's rows, for every panel column j outside [m0,m1). One warp per column j reads
   // A[:,j] ONCE for all nw reflectors; j < m0 are earlier reflectors (strictly-lower V entries only: every row where
   // V_cur is nonzero lies below them), j >= m1 the trailing columns.
   {const bool below=first>=m1;   // every local row lies below every minipanel pivot: no mask on V_cur
    // TWO target columns per warp pass and two rows per lane per step: four independent global loads in flight per
    // warp instead of one -- the one-column loop was latency-bound (125 us/boundary at 16384 rows, 32 groups). Each
    // accumulator keeps the sequential row order (r, then r+32), so the values are the same as the one-column
    // loop's.
    const int nj=h-nw;   // columns outside the minipanel; t -> panel column (t < m0 ? t : t + nw)
    for(int t=2*warp;t<nj;t+=2*warps){
     const int j0=t<m0?t:t+nw;const bool two=t+1<nj;const int j1=two?((t+1)<m0?t+1:t+1+nw):j0;
     const T*a0=panel+first+size_t(j0)*lda,*a1=panel+first+size_t(j1)*lda;
     T acc0[PW],acc1[PW];
     #pragma unroll
     for(int c=0;c<PW;++c){acc0[c]=0;acc1[c]=0;}
     int r=lane;
     for(;r+32<rows;r+=64){
      const T x00=a0[r],x01=a0[r+32],x10=a1[r],x11=a1[r+32];const int gr0=first+r,gr1=gr0+32;
      #pragma unroll
      for(int c=0;c<PW;++c)if(c<nw){
       const T v0=(below||gr0>=m0+c)?a[r+size_t(c)*local_height]:T(0);
       const T v1=(below||gr1>=m0+c)?a[r+32+size_t(c)*local_height]:T(0);
       acc0[c]+=v0*x00;acc1[c]+=v0*x10;acc0[c]+=v1*x01;acc1[c]+=v1*x11;}
     }
     for(;r<rows;r+=32){const T x0=a0[r],x1=a1[r];const int gr=first+r;
      #pragma unroll
      for(int c=0;c<PW;++c)if(c<nw){const T v=(below||gr>=m0+c)?a[r+size_t(c)*local_height]:T(0);acc0[c]+=v*x0;acc1[c]+=v*x1;}}
#if TQR_PANEL_PXPOSE
     if constexpr(PW==16){   // 32 partials (acc0 | acc1) -> transposed butterfly, lane l stores partial l
      T v[32];
      #pragma unroll
      for(int c=0;c<16;++c){v[c]=acc0[c];v[16+c]=acc1[c];}
      #pragma unroll
      for(int s=16;s>=1;s>>=1){const bool up=(lane&s)!=0;
       #pragma unroll
       for(int i=0;i<s;++i){const T send=up?v[i]:v[i+s],keep=up?v[i+s]:v[i];v[i]=keep+__shfl_xor_sync(0xffffffffu,send,s);}}
      const int c=lane&15;const bool second=lane>=16;
      if(c<nw&&(!second||two))bpart[(size_t(c)*h+(second?j1:j0))*groups+part]=v[0];
     }else
#endif
     {
     #pragma unroll
     for(int c=0;c<PW;++c)if(c<nw){const T s0=panel_boundary_sum(acc0[c]),s1=panel_boundary_sum(acc1[c]);
      if(lane==0){bpart[(size_t(c)*h+j0)*groups+part]=s0;if(two)bpart[(size_t(c)*h+j1)*groups+part]=s1;}}
     }
    }}
   RPH(3);
   grid.sync();
   RPH(4);
   // Each CTA owns a disjoint slice of the combine and commits it once (fixed order over the groups).
   for(int idx=part+warp*groups;idx<nw*h;idx+=warps*groups){
    const int j=idx%h;if(j>=m0&&j<m1)continue;
    T acc=0;for(int i=lane;i<groups;i+=32)acc+=bpart[size_t(idx)*groups+i];
    acc=panel_boundary_sum(acc);if(lane==0)bfull[idx]=acc;
   }
   grid.sync();
   RPH(5);
   for(int i=tid;i<nw*h;i+=blockDim.x)P[i]=bfull[i];
#if !TQR_PANEL_TDEFER
   for(int i=tid;i<nw*nw;i+=blockDim.x)t2[i]=tri[m0+i%nw+size_t(m0+i/nw)*b];
#endif
   __syncthreads();
   // P <- T2^T P, in registers so the triangular sweep needs no extra barrier.
   for(int j=tid;j<h;j+=blockDim.x){
    if(j>=m0&&j<m1)continue;
    T col[PW];
    #pragma unroll
    for(int c=0;c<PW;++c)col[c]=(c<nw)?P[size_t(c)*h+j]:T(0);
    for(int c=nw-1;c>=0;--c){T acc=0;
     #pragma unroll
     for(int cp=0;cp<PW;++cp)if(cp<=c)acc+=t2[cp+size_t(c)*nw]*col[cp];
     P[size_t(c)*h+j]=acc;}
   }
   __syncthreads();
   RPH(6);
   // A[:, j>=m1] -= V_cur P, in place in global memory, EIGHT trailing columns per thread with their eight loads
   // issued before the arithmetic (memory-level parallelism; the four-column loop was latency-bound).
   const int ntr=h-m1;
   if(ntr>0){const int jt=(ntr+7)/8;
    for(int idx=tid;idx<rows*jt;idx+=blockDim.x){
     const int r=idx%rows,jg=idx/rows,j0=m1+jg*8,gr=first+r;
     T v[PW];
     #pragma unroll
     for(int c=0;c<PW;++c)v[c]=(c<nw&&gr>=m0+c)?a[r+size_t(c)*local_height]:T(0);
     T*x=panel+first+r+size_t(j0)*lda;T xv[8];
     #pragma unroll
     for(int u=0;u<8;++u)xv[u]=(j0+u<h)?x[u*size_t(lda)]:T(0);
     #pragma unroll
     for(int u=0;u<8;++u){T acc=0;
      #pragma unroll
      for(int c=0;c<PW;++c)if(c<nw)acc+=v[c]*P[size_t(c)*h+j0+u];
      xv[u]-=acc;}
     #pragma unroll
     for(int u=0;u<8;++u)if(j0+u<h)x[u*size_t(lda)]=xv[u];
    }
   }
   RPH(7);
   // T12 = -T1 (V1^T V2) T2, and (V1^T V2) T2 is the j < m0 part of P.
   for(int gi=part+tid*groups;gi<m0*nw;gi+=size_t(blockDim.x)*groups){
    const int i=gi/nw,c=gi-i*nw;T acc=0;
    for(int j=i;j<m0;++j)acc+=tri[i+size_t(j)*b]*P[size_t(c)*h+j];
    tri[i+size_t(m0+c)*b]=-acc;
   }
   __syncthreads();
   RPH(8);
  }
  // ---- write the minipanel back: V below the diagonal, beta on it, R above (R and beta rescaled) ----
  for(int i=tid;i<rows*nw;i+=blockDim.x){int r=i%rows,j=i/rows,gj=m0+j;
   // export the immutable reflector while it is still in shared memory. Consumers use only p.rows x
   // h; rows above its pivot are explicit zeros.
   if(packed_v)packed_v[first+r+size_t(gj)*ldv]=first+r<gj?T(0):
     (first+r==gj?T(1):a[r+size_t(j)*local_height]);
   T v=(first+r==gj)?betas[j]:a[r+size_t(j)*local_height];
   if(first+r<=gj){v*=scales[j];if(!isfinite(v))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
   panel[first+r+size_t(gj)*lda]=v;}
  __syncthreads();
  RPH(9);
 }
 if(part==0&&tid==0){atomicAdd(&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);}
 if(tid==0){atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}
// Same capability / capacity / residency checks as the windowed launcher; `dry` checks without launching.
template<class T,int PW> void launch_cooperative_ge_resident_pw(T*A,int ld,const TilePacket*packets,
 int rows,int h,int groups,int threads,T*tri,int b,T*partial,int*status,Witness*w,cudaStream_t stream,bool dry=false,T*packed_v=nullptr,int ldv=0){
 int device;CU(cudaGetDevice(&device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
 if(!prop.cooperativeLaunch||groups<1||groups>coop_partition_limit||h<1||h>128||rows<h||threads<32||threads>1024||threads%32)throw std::runtime_error("cooperative_resident_capability_before_modify");
 int local_height=ceildiv(rows,groups);size_t dynamic=coop_res_shared_bytes(local_height,h,PW,sizeof(T));
 cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,cooperative_ge_resident<T,PW>));
 if(dynamic>prop.sharedMemPerBlockOptin-attr.sharedSizeBytes)throw std::runtime_error("cooperative_resident_shared_capacity_before_modify");
 CU(cudaFuncSetAttribute(cooperative_ge_resident<T,PW>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dynamic)));
 int occupancy=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cooperative_ge_resident<T,PW>,threads,dynamic));
 if(size_t(groups)>size_t(occupancy)*prop.multiProcessorCount)throw std::runtime_error("cooperative_resident_residency_before_modify");
 if(dry)return;
 T*bpart=partial+coop_mini_partial_slots(groups,h,PW),*bfull=bpart+size_t(groups)*PW*h;
 void*args[]={&A,&ld,&packets,&groups,&local_height,&h,&tri,&b,&partial,&bpart,&bfull,&status,&w,&packed_v,&ldv};
 CU(cudaLaunchCooperativeKernel((const void*)cooperative_ge_resident<T,PW>,dim3(groups),dim3(threads),args,dynamic,stream));
}
template<class T> void launch_cooperative_ge_resident(T*A,int ld,const TilePacket*packets,
 int rows,int h,int pw,int groups,int threads,T*tri,int b,T*partial,int*status,Witness*w,cudaStream_t stream,bool dry=false,T*packed_v=nullptr,int ldv=0){
 if(pw==8)launch_cooperative_ge_resident_pw<T,8>(A,ld,packets,rows,h,groups,threads,tri,b,partial,status,w,stream,dry,packed_v,ldv);
 else if(pw==16)launch_cooperative_ge_resident_pw<T,16>(A,ld,packets,rows,h,groups,threads,tri,b,partial,status,w,stream,dry,packed_v,ldv);
 else if(pw==32)launch_cooperative_ge_resident_pw<T,32>(A,ld,packets,rows,h,groups,threads,tri,b,partial,status,w,stream,dry,packed_v,ldv);
 else throw std::runtime_error("cooperative_resident_width_before_modify");
}
}
