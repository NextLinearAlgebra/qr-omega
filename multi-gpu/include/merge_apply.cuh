#pragma once
#include "tiled_kernels.cuh"
namespace tqr{
// Structured pentagonal merge apply.
//
// Section 4 states that a merge's reflectors are "nonzero only on the matrix rows of the two tile
// rows it combines (its support rows)". For a k-way TT merge the compact WY factor is therefore V =
// [I; V_1; ...; V_{k-1}] with every V_c upper triangular -- exactly what tile_merge_v materialises.
// Eq apply then reads, with no approximation and no reordering of the products,
//  W = X_0 + sum_{c>=1} V_c^T X_c,   Z = op(T) W,
//  X_0 -= Z,                         X_c -= V_c Z,
// so all three products run on the support rows in place. The dense (k*b x q) staging block never
// exists: the gather and the scatter round trips disappear (two reads and two writes of the stacked
// block per strip), V is read straight from the factor matrix under the same triangular mask
// tile_merge_v applies, and the identity block never enters a product.
//
// Same elimination list, same trees, same carriers, same depth: only where the three products read
// and write changes. Each child's rows are written exactly once, preserving owner-commits-once.
//
// Layout notes. VW[s*b+r] and VD[r*b+s] hold the same V_c under the two access orders the two
// phases need, so the thread index is the fastest varying shared index in both and neither phase
// takes a bank conflict. Columns are carried NREG at a time in registers so one V_c load serves
// blockDim.y*NREG columns. F/fld/fbegin address the retained factors, which are a different matrix
// from X whenever a stored Q is applied.
template<class T,int NREG>
__global__ __launch_bounds__(256) void tile_merge_apply(
    T*A,int ld,int begin,const T*F,int fld,int fbegin,
    const TileMerge*es,int count,const T*Tri,long long tri_stride,
    int b,int col,int q,int radix,bool transpose){
 const int t=blockIdx.y;if(t>=count)return;const TileMerge&e=es[t];
 extern __shared__ __align__(16) unsigned char merge_apply_smem[];
 // VW ([s*b+r]) and VD ([r*b+s]) are the two access orders the W and D
 // phases need. The phases never hold V at the same time -- each reloads its
 // own child -- so they alias one buffer and only the staging area is extra.
 T*VW=reinterpret_cast<T*>(merge_apply_smem);      // [s*b+r] = V_c[s][r]
 T*VD=VW;                                          // [r*b+s] = V_c[s][r]
 T*S=VW+size_t(b)*b;                               // staging blockDim.y*NREG*b
 const int r=threadIdx.x,y=threadIdx.y,yc=blockDim.y,h=e.h,k=e.k;
 const int tid=threadIdx.y*blockDim.x+threadIdx.x,nthr=blockDim.x*blockDim.y;
 const T*Tsrc=Tri+tri_stride*t;
 const int per=yc*NREG,base=blockIdx.x*per,step=gridDim.x*per;
 const int row0=e.row[0]-begin,h0=min(e.height[0],h);
 T w[NREG],z[NREG];
 for(int c0=base;c0<q;c0+=step){
  #define MA_J(u) (c0+y+(u)*yc)
  // --- W = X_0 + sum_{c>=1} V_c^T X_c ---------------------------------
  #pragma unroll
  for(int u=0;u<NREG;++u)w[u]=(MA_J(u)<q&&r<h0)?A[row0+r+size_t(col+MA_J(u))*ld]:T(0);
  for(int c=1;c<k;++c){
   const int rowc=e.row[c]-fbegin,xrowc=e.row[c]-begin,hc=e.height[c];
   for(int i=tid;i<b*b;i+=nthr){int ss=i%b,rr=i/b;
    VW[size_t(ss)*b+rr]=(ss<hc&&rr<h&&ss<=rr)?F[rowc+ss+size_t(e.col+rr)*fld]:T(0);}
   #pragma unroll
   for(int u=0;u<NREG;++u)S[(size_t(y)*NREG+u)*b+r]=(MA_J(u)<q&&r<hc)?A[xrowc+r+size_t(col+MA_J(u))*ld]:T(0);
   __syncthreads();
   // s outer, columns inner: VW[s*b+r] is invariant in u, so it is read once
   // per s into a register instead of once per (s,u), and S[..*b+s] is a warp
   // broadcast (blockDim.x = b, so y is constant across a warp). This is what
   // takes the inner loop off the shared-memory bandwidth bound.
   for(int s=0;s<b;++s){const T v=VW[size_t(s)*b+r];const T*xs=S+s;
    #pragma unroll
    for(int u=0;u<NREG;++u)w[u]+=v*xs[(size_t(y)*NREG+u)*b];}
   __syncthreads();
  }
  // --- Z = op(T) W -----------------------------------------------------
  for(int i=tid;i<b*b;i+=nthr){int ss=i%b,rr=i/b;
   VW[size_t(ss)*b+rr]=(ss<h&&rr<h)?(transpose?Tsrc[ss+size_t(rr)*b]:Tsrc[rr+size_t(ss)*b]):T(0);}
  #pragma unroll
  for(int u=0;u<NREG;++u)S[(size_t(y)*NREG+u)*b+r]=w[u];
  __syncthreads();
  #pragma unroll
  for(int u=0;u<NREG;++u)z[u]=0;
  for(int s=0;s<b;++s){const T v=VW[size_t(s)*b+r];const T*ws=S+s;
   #pragma unroll
   for(int u=0;u<NREG;++u)z[u]+=v*ws[(size_t(y)*NREG+u)*b];}
  __syncthreads();
  // --- X_0 -= Z (V_0 = I), X_c -= V_c Z --------------------------------
  #pragma unroll
  for(int u=0;u<NREG;++u)if(MA_J(u)<q&&r<h0)A[row0+r+size_t(col+MA_J(u))*ld]-=z[u];
  #pragma unroll
  for(int u=0;u<NREG;++u)S[(size_t(y)*NREG+u)*b+r]=z[u];
  __syncthreads();
  for(int c=1;c<k;++c){
   const int rowc=e.row[c]-fbegin,xrowc=e.row[c]-begin,hc=e.height[c];
   for(int i=tid;i<b*b;i+=nthr){int ss=i%b,rr=i/b;
    VD[size_t(rr)*b+ss]=(ss<hc&&rr<h&&ss<=rr)?F[rowc+ss+size_t(e.col+rr)*fld]:T(0);}
   __syncthreads();
   T acc[NREG];
   #pragma unroll
   for(int u=0;u<NREG;++u)acc[u]=0;
   for(int rr=0;rr<b;++rr){const T v=VD[size_t(rr)*b+r];const T*zs=S+rr;
    #pragma unroll
    for(int u=0;u<NREG;++u)acc[u]+=v*zs[(size_t(y)*NREG+u)*b];}
   #pragma unroll
   for(int u=0;u<NREG;++u)if(MA_J(u)<q&&r<hc)A[xrowc+r+size_t(col+MA_J(u))*ld]-=acc[u];
   __syncthreads();
  }
 }
 #undef MA_J
}
// Shared bytes one CTA needs.
inline size_t merge_apply_shared(int b,int yc,int nreg,size_t word){
 return (size_t(b)*b+size_t(yc)*nreg*b)*word;
}
}
