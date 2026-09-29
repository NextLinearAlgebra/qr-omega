#pragma once
// Warp-level building blocks shared by the QR-v2 panels (gau_ge.cuh owner-CTA chain, gau_fast.cuh register panels):
// range-checked reflector arithmetic and recursive-halving warp reductions.
#include <cuda_runtime.h>
namespace tqr {
template<class T> struct GfRange;
template<> struct GfRange<float>{static constexpr float lo=0x1p-100f,hi=0x1p100f,xmax=0x1p50f;};
template<> struct GfRange<double>{static constexpr double lo=0x1p-900,hi=0x1p900,xmax=0x1p450;};
__device__ __forceinline__ float gf_sqrt(float x){return __fsqrt_rn(x);}
__device__ __forceinline__ double gf_sqrt(double x){return __dsqrt_rn(x);}
__device__ __forceinline__ float gf_div(float a,float b){return __fdiv_rn(a,b);}
__device__ __forceinline__ double gf_div(double a,double b){return __ddiv_rn(a,b);}
__device__ __forceinline__ float gf_fma(float a,float b,float c){return __fmaf_rn(a,b,c);}
__device__ __forceinline__ double gf_fma(double a,double b,double c){return __fma_rn(a,b,c);}

template<int N> struct GfLog{static constexpr int v=N<=1?0:N<=2?1:N<=4?2:N<=8?3:4;};
// Recursive halving: at the level with xor offset O each lane keeps half of its N live partials and receives the
// partner's matching half.
template<class T,int M,int N,int O>
__device__ __forceinline__ void gf_halve(T(&cur)[M],int lane){
 if constexpr(N>1){
  constexpr int H=N/2;const bool hi=(lane&O)!=0;
  #pragma unroll
  for(int q=0;q<H;++q){const T send=hi?cur[q]:cur[q+H];const T keep=hi?cur[q+H]:cur[q];
   cur[q]=keep+__shfl_xor_sync(0xffffffffu,send,O);}
  gf_halve<T,M,H,O/2>(cur,lane);}
}
// Warp sum of N (1,2,4,8,16) partials per lane. Afterwards lane L holds the full sum of entry L >> (5 - log2 N);
// d is destroyed. 2N - 1 + (5 - log2 N) shuffles, fixed summation order.
template<class T,int N>
__device__ __forceinline__ T gf_reduce_scatter(T(&d)[N],int lane){
 gf_halve<T,N,N,16>(d,lane);
 T s=d[0];
 #pragma unroll
 for(int o=16>>GfLog<N>::v;o;o>>=1)s+=__shfl_xor_sync(0xffffffffu,s,o);
 return s;}
// Every lane receives all N sums.
template<class T,int N>
__device__ __forceinline__ void gf_allreduce(T(&d)[N],int lane){
 const T s=gf_reduce_scatter<T,N>(d,lane);
 if constexpr(N==1)d[0]=s;
 else{
  #pragma unroll
  for(int q=0;q<N;++q)d[q]=__shfl_sync(0xffffffffu,s,q<<(5-GfLog<N>::v));}
}

// Nonzero test on the bit pattern (the sign of -0 dropped): no predicate juggling on the critical path.
__device__ __forceinline__ unsigned gf_nz(float y){return __float_as_uint(y)<<1;}
__device__ __forceinline__ unsigned gf_nz(double y){const unsigned long long b=(unsigned long long)__double_as_longlong(y)<<1;return unsigned(b>>32)|unsigned(b);}
// 1/sqrt(q) and 1/s for q, s well inside the normal range (the fast-path bounds): hardware estimate + one Newton step
// (fp32: <= 1 ulp more than correctly rounded; fp64: CUDA's rsqrt / IEEE reciprocal). LAPACK's own dlapy2 / dscal
// sequence is not correctly rounded either; the Householder backward error analysis only needs O(u) relative errors.
__device__ __forceinline__ float gf_rsqrt(float q){const float y=rsqrtf(q),e=__fmaf_rn(-q*y,y,1.0f);return __fmaf_rn(0.5f*y,e,y);}
__device__ __forceinline__ double gf_rsqrt(double q){return rsqrt(q);}
__device__ __forceinline__ float gf_rcp(float s){float r;asm("rcp.approx.ftz.f32 %0, %1;":"=f"(r):"f"(s));return __fmaf_rn(r,__fmaf_rn(-s,r,1.0f),r);}
__device__ __forceinline__ double gf_rcp(double s){return __drcp_rn(s);}
}
