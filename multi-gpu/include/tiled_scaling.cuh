#pragma once
// Column scaling before the factorization (FP32 range safety); R is rescaled afterwards.
#include "tiled_kernels.cuh"
namespace tqr {
// Positive diagonal column scaling commutes with a fixed unpivoted QR:
// A D^-1 = Q R', A = Q (R' D). V/T are dimensionless and are never rescaled.
// Finite scan precedes this operation. All reductions and arithmetic are GPU
// instructions in T; no numerical memory or switch reduction is used.
template<class T> __global__ void tiled_column_max(const T*a,int nr,int n,int ld,T*scale){
 __shared__ T red[256];int col=blockIdx.x;T mx=0;
 for(int row=threadIdx.x;row<nr;row+=blockDim.x)mx=max(mx,absval(a[row+size_t(col)*ld]));
 mx=block_max(mx,red);if(threadIdx.x==0)scale[col]=mx;
}
template<class T> __global__ void tiled_scale_max(T*scale,const T*other,int n){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)scale[i]=max(scale[i],other[i]);
}
// The old flat index paid a 64-bit i%nr, i/nr per element and ran instruction-bound at ~1 TB/s (0.16 s per fp64
// n=65536 factorization over the four whole-matrix passes). Same arithmetic per element, so results are
// bit-identical. UpperOnly visits only the rows it may touch in the contiguous layout (global row <= col); CheckAll
// additionally checks every other element of the column for finiteness (fusing the separate tiled_finite_output pass
// into the restore).
template<class T,bool Restore,bool UpperOnly=false,bool CheckAll=false> __global__ void tiled_column_scale(T*a,int nr,int n,int ld,int begin,const T*scale,int*status,int blk=0,int p=1,int rank=0){
 bool bad=false;
 const int stride=blockDim.x*gridDim.x,r0=blockIdx.x*blockDim.x+threadIdx.x;
 for(int col=blockIdx.y;col<n;col+=gridDim.y){
  T*c=a+size_t(col)*ld;const T s=scale[col];
  int lim=nr;
  if constexpr(UpperOnly){if(!blk){const long long top=(long long)col-begin+1;lim=int(top<0?0:(top<nr?top:nr));}}
  for(int row=r0;row<lim;row+=stride){
   if constexpr(UpperOnly){if(blk){const long long grow=(long long)(row/blk)*blk*p+(long long)rank*blk+row%blk;if(grow>col){if constexpr(CheckAll)bad|=!isfinite(c[row]);continue;}}}
   const T x=c[row];T y;
   if constexpr(Restore)y=x*s;else y=s?x/s:T(0);
   c[row]=y;bad|=!isfinite(y);
  }
  if constexpr(CheckAll)for(int row=lim+r0;row<nr;row+=stride)bad|=!isfinite(c[row]);
 }
 if(__syncthreads_or(bad)&&threadIdx.x==0)atomicCAS(status,0,UNREPRESENTABLE_RESULT);
}
// Launch geometry for the 2-D column passes: rows over up to 8 blocks of 256 (x), columns grid-strided (y).
inline dim3 column_pass_grid(int nr,int n){return dim3(std::max(1,std::min(8,ceildiv(std::max(nr,1),1024))),std::max(1,std::min(n,8192)));}
template<class T> __global__ void finite_scan_2d(const T* a,int rows,int n,int ld,int* status){
 bool bad=false;const int stride=blockDim.x*gridDim.x,r0=blockIdx.x*blockDim.x+threadIdx.x;
 for(int col=blockIdx.y;col<n;col+=gridDim.y){const T*c=a+size_t(col)*ld;for(int row=r0;row<rows;row+=stride)bad|=!isfinite(c[row]);}
 if(__syncthreads_or(bad)&&threadIdx.x==0)atomicCAS(status,0,NONFINITE_INPUT);
}
}
