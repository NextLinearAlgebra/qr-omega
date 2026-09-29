#pragma once
// Single-block factorization of small panels in shared memory.
#include "tiled_kernels.cuh"
namespace tqr {
// A complete native GE packet. The matrix is staged once in shared memory; each warp updates a
// different trailing column. Retained A/tau has the same ordered reflector convention as
// scalar_inplace. No vendor QR is called. The compiled capability is m,n<=128, threads=32/128/256.
template<class T> __global__ void small_shared_packet(T*global,int m,int n,int ld,T*taus,int*status,Witness*w){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 extern __shared__ __align__(16) unsigned char storage[];
 T*a=reinterpret_cast<T*>(storage);
 __shared__ T scale[128],red[8],tau,beta,den,normscale;
 if(tid==0){*status=OK;*w=Witness{};}__syncthreads();
 for(int i=tid;i<m*n;i+=blockDim.x){T x=global[i%m+size_t(i/m)*ld];a[i]=x;if(!isfinite(x))atomicCAS(status,OK,NONFINITE_INPUT);}
 __syncthreads();if(*status)return; // caller A is still byte-identical
 for(int j=warp;j<n;j+=warps){T mx=0;for(int r=lane;r<m;r+=32)mx=max(mx,absval(a[r+j*m]));mx=warp_maximum(mx);if(lane==0)scale[j]=mx;
  for(int r=lane;r<m;r+=32)a[r+j*m]=mx?a[r+j*m]/mx:T(0);
 }__syncthreads();
 for(int k=0;k<min(m,n);++k){
  T tail=0;for(int r=k+1+tid;r<m;r+=blockDim.x)tail=max(tail,absval(a[r+k*m]));tail=tile_reduce<T,true>(tail,red);
  if(tid==0){normscale=max(tail,absval(a[k+k*m]));tau=0;beta=a[k+k*m];den=1;}__syncthreads();
  if(tail){T sum=0;for(int r=k+tid;r<m;r+=blockDim.x){T x=a[r+k*m]/normscale;sum+=x*x;}sum=tile_reduce<T,false>(sum,red);
   if(tid==0){T alpha=a[k+k*m]/normscale,bn=-sqrt(sum);if(alpha<0)bn=-bn;beta=normscale*bn;den=alpha-bn;tau=1-alpha/bn;}__syncthreads();
   for(int r=k+1+tid;r<m;r+=blockDim.x)a[r+k*m]=(a[r+k*m]/normscale)/den;
  }
  if(tid==0){a[k+k*m]=beta;taus[k]=tau;}__syncthreads();
  for(int j=k+1+warp;j<n;j+=warps){T dot=lane==0?a[k+j*m]:T(0);for(int r=k+1+lane;r<m;r+=32)dot+=a[r+k*m]*a[r+j*m];dot=warp_add(dot)*tau;
   if(lane==0)a[k+j*m]-=dot;for(int r=k+1+lane;r<m;r+=32)a[r+j*m]-=a[r+k*m]*dot;
  }__syncthreads();
 }
 for(int i=tid;i<m*n;i+=blockDim.x){int r=i%m,j=i/m;T x=a[i];if(r<=j)x*=scale[j];global[r+size_t(j)*ld]=x;if(!isfinite(x))atomicCAS(status,OK,UNREPRESENTABLE_RESULT);}
 if(tid==0){w->ge=1;w->reflectors=min(m,n);}
}
}
