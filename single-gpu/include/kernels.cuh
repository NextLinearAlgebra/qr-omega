#pragma once
// Core device kernels of the engine and its execution counters.
#include "execution_format.hpp"
#include "common.hpp"
namespace tqr {
enum DeviceStatus { OK=0, NONFINITE_INPUT=1, UNREPRESENTABLE_RESULT=2, HISTORY_ERROR=3, PROTOCOL_ERROR=4 };
// Device-side execution counters. The first line is the base set (issued
// eliminations, reflectors, partials, joins, owner commits). The second is the
// execution_format 24 addition: per-product-kind counts written by kernels that
// run AFTER the products they certify, and the simultaneous-credit high-water
// mark, which is the only evidence that a pipeline depth d>1 was physically
// held rather than merely planned. See include/execution_evidence.hpp.
// Execution_format 26 addition: per-kind PEER-PARTIAL counts written by the
// arithmetic kernels that produce the partials (one increment per executed
// peer, derived from executed block coordinates, never from a host-supplied
// c); device_combines (additive-combine kernel launches); device_physical_
// commits (launches that NUMERICALLY write X: commit_update, packed_commit,
// tile_carried_D slices -- history_commit markers are counted separately in
// device_history_marks and are NOT physical commits); device_membership_
// reports (one per executed slice/block that reported its own participation).
// cuBLAS-intermediary peer products run vendor-internal kernels and report
// nothing here: their block execution is labeled unknown in the receipt.
struct Witness { unsigned long long ge,ts,tt,reflectors,partials,joins,commits,replica_bytes,errors;
 unsigned long long device_products[execution_product_kinds],credits_live,credits_max,evidence_overflow;
 unsigned long long device_peer_partials[execution_product_kinds],device_combines,device_physical_commits,device_history_marks,device_membership_reports;
 // Cooperative panels report GPU peers only (per-block thread corroboration there is owed
 // instrumentation, stated in the receipt).
 unsigned long long device_block_peers[execution_product_kinds];
 // QR-v2 factor carriers split output columns, independently of the K cut.
 unsigned long long device_qrv2_products[execution_product_kinds],device_qrv2_owners[execution_product_kinds];
 // Levels: 0 block (warp groups/ slices of a CTA, SMEM combine), 1 cluster (CTAs of a cluster,
 // DSM), 2 gpu (CTAs via HBM), 3 node (GPUs). c buckets: 1,2,3,4,8,16,32,64,128,other.
 unsigned long long device_level_c[execution_product_kinds][execution_levels][execution_c_buckets]; };
__host__ __device__ inline int witness_c_bucket(int c){return c==1?0:c==2?1:c==3?2:c==4?3:c==8?4:c==16?5:c==32?6:c==64?7:c==128?8:9;}
__device__ inline void witness_levels(Witness*w,int kind,int b,int cl,int g,int nd){
 atomicAdd(&w->device_level_c[kind][0][witness_c_bucket(b)],1ULL);atomicAdd(&w->device_level_c[kind][1][witness_c_bucket(cl)],1ULL);
 atomicAdd(&w->device_level_c[kind][2][witness_c_bucket(g)],1ULL);atomicAdd(&w->device_level_c[kind][3][witness_c_bucket(nd)],1ULL);}
template<class T> __device__ T absval(T x){return x<T(0)?-x:x;}
template<class T> __device__ T block_sum(T x,T* s) {
 int t=threadIdx.x;s[t]=x;__syncthreads();
 for(int d=blockDim.x/2;d;d/=2){if(t<d)s[t]+=s[t+d];__syncthreads();}return s[0];
}
template<class T> __device__ T block_max(T x,T* s) {
 int t=threadIdx.x;s[t]=x;__syncthreads();
 for(int d=blockDim.x/2;d;d/=2){if(t<d)s[t]=max(s[t],s[t+d]);__syncthreads();}return s[0];
}
template<class T> __global__ void finite_scan(const T* a,int rows,int n,int ld,int* status){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(rows)*n;i+=size_t(blockDim.x)*gridDim.x)
  if(!isfinite(a[(i/rows)*ld+i%rows]))atomicCAS(status,0,NONFINITE_INPUT);
}
__device__ inline uint64_t mix64(uint64_t x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
template<class T> __global__ void make_input(T* a,int rows,int n,int ld,int begin,int m,int kind,T scale) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(ld)*n;i+=size_t(blockDim.x)*gridDim.x){
  int r=i%ld,j=i/ld; if(r>=rows){a[i]=T(12345);continue;}
  int g=begin+r; int jj=(kind==2&&j>0)?0:j;
  T v=T(int(mix64(uint64_t(g)+uint64_t(jj)*std::max(m,1)+73)%2001)-1000)/T(1000);
  if(kind==1)v=T(0);if(kind==3)v=g==j?T(1):T(0);if(kind==4&&g>j)v=T(0);
  if(kind==5&&g==0&&j==0)v=std::numeric_limits<T>::infinity();
  if(kind==6&&g==0&&j==0)v=std::numeric_limits<T>::quiet_NaN();
  a[i]=v*scale;
 }
}
template<class T> __global__ void identity(T* a,int rows,int n,int ld,int begin) {
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(rows)*n;i+=size_t(blockDim.x)*gridDim.x)
  a[(i/rows)*ld+i%rows]=(begin+int(i%rows)==int(i/rows))?T(1):T(0);
}
template<class T> __global__ void gather_rows(const T* a,int ld,int begin,const int* rows,int count,int col,int width,T* out,int outld,int pos=0) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*width;i+=blockDim.x*gridDim.x)
  out[pos+i%count+(i/count)*outld]=a[rows[i%count]-begin+size_t(col+i/count)*ld];
}
template<class T> __global__ void scatter_rows(const T* in,int inld,int pos,T* a,int ld,int begin,const int* rows,int count,int col,int width) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*width;i+=blockDim.x*gridDim.x)
  a[rows[i%count]-begin+size_t(col+i/count)*ld]=in[pos+i%count+(i/count)*inld];
}
// One complete GE, TS or TT packet. Scalar math, accumulation and storage all use T. No triangular
// pivot division.
template<class T> __global__ void native_hh(T* a,int s,int h,T* v,T* tri,int kind,int* status,Witness* w) {
 __shared__ T red[256];__shared__ T tau,beta,den,scale;
 int tid=threadIdx.x;
 for(int i=tid;i<s*h;i+=blockDim.x)v[i]=T(0);
 for(int i=tid;i<h*h;i+=blockDim.x)tri[i]=T(0);
 __syncthreads();
 for(int k=0;k<h;++k){
  T tail=T(0);for(int r=k+1+tid;r<s;r+=blockDim.x)tail=max(tail,absval(a[r+k*s]));
  tail=block_max(tail,red);
  if(tid==0){scale=max(tail,absval(a[k+k*s]));tau=T(0);beta=a[k+k*s];den=T(1);}
  __syncthreads();
  if(tail!=T(0)){
   T sum=T(0);for(int r=k+tid;r<s;r+=blockDim.x){T x=a[r+k*s]/scale;sum+=x*x;}
   sum=block_sum(sum,red);
   if(tid==0){T alpha=a[k+k*s]/scale;T bn=-sqrt(sum);if(alpha<T(0))bn=-bn;
    beta=scale*bn;den=alpha-bn;tau=T(1)-alpha/bn;
    if(!isfinite(beta))atomicCAS(status,0,UNREPRESENTABLE_RESULT);
   }__syncthreads();
   for(int r=k+1+tid;r<s;r+=blockDim.x)v[r+k*s]=(a[r+k*s]/scale)/den;
  }
  if(tid==0){v[k+k*s]=T(1);tri[k+k*h]=tau;}
  __syncthreads();
  // Safe scaled application to the unfactored columns of the same packet.
  for(int j=k+1;j<h;++j){
   T mx=T(0);for(int r=k+tid;r<s;r+=blockDim.x)mx=max(mx,absval(a[r+j*s]));mx=block_max(mx,red);
   T dot=T(0);if(mx!=T(0))for(int r=k+tid;r<s;r+=blockDim.x)dot+=v[r+k*s]*(a[r+j*s]/mx);
   dot=block_sum(dot,red)*tau;
   for(int r=k+tid;r<s;r+=blockDim.x){T z=mx==T(0)?T(0):(a[r+j*s]/mx-v[r+k*s]*dot)*mx;a[r+j*s]=z;if(!isfinite(z))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
   __syncthreads();
  }
  for(int r=k+tid;r<s;r+=blockDim.x)a[r+k*s]=(r==k)?beta:T(0);
  __syncthreads();
 }
 // Forward compact WY: T(0:k,k)=-tau_k*T_previous*(V_previous^T*v_k).
 // One GPU thread composes this small triangular factor in the selected precision.
 if(tid==0){
  T g[32];
  for(int k=1;k<h;++k){for(int j=0;j<k;++j){T dot=0;for(int r=0;r<s;++r)dot+=v[r+j*s]*v[r+k*s];g[j]=dot;}
   for(int i=0;i<k;++i){T sum=0;for(int j=i;j<k;++j)sum+=tri[i+j*h]*g[j];tri[i+k*h]=-tri[k+k*h]*sum;}
  }
  if(kind==0)atomicAdd(&w->ge,1ULL);else if(kind==1)atomicAdd(&w->ts,1ULL);else atomicAdd(&w->tt,1ULL);
  atomicAdd(&w->reflectors,(unsigned long long)h);
 }
}
template<class T> __global__ void column_scales(const T* x,int s,int q,T* scale){
 __shared__ T red[256];int j=blockIdx.x;T mx=0;for(int r=threadIdx.x;r<s;r+=blockDim.x)mx=max(mx,absval(x[r+j*s]));
 mx=block_max(mx,red);if(threadIdx.x==0)scale[j]=mx;
}
template<class T> __global__ void w_partial(const T* v,const T* x,const T* scale,int s,int h,int q,int lo,int hi,T* out,Witness* w){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<h*q;i+=blockDim.x*gridDim.x){int a=i%h,j=i/h;T z=0;
  if(scale[j]!=T(0))for(int r=lo;r<hi;++r)z+=v[r+a*s]*(x[r+j*s]/scale[j]);out[i]=z;}
 // One peer partial per launch: this kernel forms exactly the [lo,hi)
 // interval of W. APPLY_W is kind 2 (must match CarrierWitness::Kind order).
 if(blockIdx.x==0&&threadIdx.x==0){atomicAdd(&w->partials,1ULL);atomicAdd(&w->device_peer_partials[2],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}
template<class T> __global__ void sum_partials(const T* p,T* out,int count,int layers,Witness* w){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){T z=0;for(int c=0;c<layers;++c)z+=p[c*count+i];out[i]=z;}
 if(blockIdx.x==0&&threadIdx.x==0){atomicAdd(&w->joins,1ULL);atomicAdd(&w->device_combines,1ULL);}
}
template<class T> __global__ void triangular_apply(const T* tri,const T* w,T* z,int h,int q,bool transpose){
 for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<h*q;idx+=blockDim.x*gridDim.x){int i=idx%h,j=idx/h;T sum=0;
  if(transpose){for(int k=0;k<=i;++k)sum+=tri[k+i*h]*w[k+j*h];}
  else{for(int k=i;k<h;++k)sum+=tri[i+k*h]*w[k+j*h];}z[idx]=sum;}
}
template<class T> __global__ void commit_update(const T* v,const T* x,const T* z,const T* scales,int s,int h,int q,T* a,int ld,int begin,const int* rows,int pos,int count,int col,int* status,Witness* witness){
 for(int idx=blockIdx.x*blockDim.x+threadIdx.x;idx<count*q;idx+=blockDim.x*gridDim.x){int rr=idx%count+pos,j=idx/count;T d=0;for(int k=0;k<h;++k)d+=v[rr+k*s]*z[k+j*h];
  T y=scales[j]==T(0)?T(0):(x[rr+j*s]/scales[j]-d)*scales[j];
  // A NUMERICAL write to X: this launch is one physical owner commit.
  a[rows[rr]-begin+size_t(col+j)*ld]=y;if(!isfinite(y))atomicCAS(status,0,UNREPRESENTABLE_RESULT);
 }
 if(blockIdx.x==0&&threadIdx.x==0){atomicAdd(&witness->commits,1ULL);atomicAdd(&witness->device_physical_commits,1ULL);}
}
__global__ inline void record_replica(Witness* w,size_t bytes){if(threadIdx.x==0)atomicAdd(&w->replica_bytes,(unsigned long long)bytes);}
__global__ inline void history_commit(uint64_t* histories,const int* rows,int count,int tiles,int tile,uint64_t expected,uint64_t next,int* status){
 // Used for a homogeneous predecessor group; heterogeneous histories are split.
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){auto p=(unsigned long long*)(histories+size_t(rows[i])*tiles+tile);
  if(atomicCAS(p,(unsigned long long)expected,(unsigned long long)next)!=expected)atomicCAS(status,0,HISTORY_ERROR);}
}
template<class T> __global__ void norm_probe(const T* x,int n,T* out){
 __shared__ T red[256];T mx=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)mx=max(mx,absval(x[i]));mx=block_max(mx,red);
 T z=0;if(mx)for(int i=threadIdx.x;i<n;i+=blockDim.x){T y=x[i]/mx;z+=y*y;}z=block_sum(z,red);if(threadIdx.x==0)*out=mx*sqrt(z);
}
__global__ inline void empty_kernel(){}
__global__ inline void versions(uint64_t* hist,int nr,int begin,const int* rows,const uint64_t* expected,int pos,int count,int col,int q,uint64_t next,int* status){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*q;i+=blockDim.x*gridDim.x){int r=pos+i%count,j=col+i/count;
  auto p=(unsigned long long*)(hist+rows[r]-begin+size_t(j)*nr);
  if(atomicCAS(p,(unsigned long long)expected[r],(unsigned long long)next)!=expected[r])atomicCAS(status,0,HISTORY_ERROR);
 }
}
template<class T> __global__ void packed_commit(const T* v,const T* z,const T* scales,T* x,int s,int h,int q,int* status,Witness* w){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<s*q;i+=blockDim.x*gridDim.x){int r=i%s,j=i/s;T d=0;for(int k=0;k<h;++k)d+=v[r+k*s]*z[k+j*h];T y=scales[j]==0?T(0):(x[i]/scales[j]-d)*scales[j];x[i]=y;if(!isfinite(y))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
 // A NUMERICAL write to X: one physical owner commit (history_commit markers
 // below are NOT physical commits and are counted separately).
 if(blockIdx.x==0&&threadIdx.x==0){atomicAdd(&w->commits,1ULL);atomicAdd(&w->device_physical_commits,1ULL);}
}
template<class T> __global__ void relative_error(const T* a,const T* b,int nr,int n,int lda,int ldb,T* out,bool identity_ref=false,int begin=0){
 __shared__ T red[256];T mx=0;
 for(size_t i=threadIdx.x;i<size_t(nr)*n;i+=blockDim.x){T x=identity_ref?T(begin+int(i%nr)==int(i/nr)):b[i%nr+(i/nr)*ldb];mx=max(mx,absval(x));mx=max(mx,absval(a[i%nr+(i/nr)*lda]));}
 mx=block_max(mx,red);T diff=0,den=0;
 if(mx)for(size_t i=threadIdx.x;i<size_t(nr)*n;i+=blockDim.x){T x=identity_ref?T(begin+int(i%nr)==int(i/nr)):b[i%nr+(i/nr)*ldb];T v=a[i%nr+(i/nr)*lda]/mx;T y=x/mx;diff+=(v-y)*(v-y);den+=y*y;}
 diff=block_sum(diff,red);T saved=diff;den=block_sum(den,red);
 if(threadIdx.x==0){out[0]=sqrt(saved);out[1]=sqrt(den);out[2]=mx;}
}
// Storage-minimal member: the original vectors live below A's diagonal. Only
// k tau scalars and a block reduction are additional numerical state. This is
// the serial degeneration of GE/HH, not a vendor or CPU fallback.
template<class T,bool Initialize=false> __global__ void scalar_inplace(T*a,int m,int n,int ld,T*tau,int*status,Witness*w){
 __shared__ T red[256];__shared__ T scale,den,beta,t;
 if constexpr(Initialize){
  if(threadIdx.x==0){*status=OK;*w=Witness{};}__syncthreads();
  for(size_t i=threadIdx.x;i<size_t(m)*n;i+=blockDim.x)if(!isfinite(a[i%m+size_t(i/m)*ld]))atomicCAS(status,OK,NONFINITE_INPUT);
  __syncthreads();if(*status)return;
 }
 for(int k=0;k<min(m,n);++k){T tail=0;for(int r=k+1+threadIdx.x;r<m;r+=blockDim.x)tail=max(tail,absval(a[r+size_t(k)*ld]));tail=block_max(tail,red);
  if(threadIdx.x==0){scale=max(tail,absval(a[k+size_t(k)*ld]));beta=a[k+size_t(k)*ld];t=0;den=1;}__syncthreads();
  if(tail){T sum=0;for(int r=k+threadIdx.x;r<m;r+=blockDim.x){T x=a[r+size_t(k)*ld]/scale;sum+=x*x;}sum=block_sum(sum,red);
   if(threadIdx.x==0){T alpha=a[k+size_t(k)*ld]/scale,b=-sqrt(sum);if(alpha<0)b=-b;beta=scale*b;den=alpha-b;t=1-alpha/b;if(!isfinite(beta))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}__syncthreads();
   for(int r=k+1+threadIdx.x;r<m;r+=blockDim.x)a[r+size_t(k)*ld]=(a[r+size_t(k)*ld]/scale)/den;
  }if(threadIdx.x==0){a[k+size_t(k)*ld]=beta;tau[k]=t;}__syncthreads();
  for(int j=k+1;j<n;++j){T mx=0;for(int r=k+threadIdx.x;r<m;r+=blockDim.x)mx=max(mx,absval(a[r+size_t(j)*ld]));mx=block_max(mx,red);T dot=0;
   if(mx)for(int r=k+threadIdx.x;r<m;r+=blockDim.x)dot+=(r==k?T(1):a[r+size_t(k)*ld])*(a[r+size_t(j)*ld]/mx);dot=block_sum(dot,red)*t;
   for(int r=k+threadIdx.x;r<m;r+=blockDim.x){T y=mx?(a[r+size_t(j)*ld]/mx-(r==k?T(1):a[r+size_t(k)*ld])*dot)*mx:T(0);a[r+size_t(j)*ld]=y;if(!isfinite(y))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}__syncthreads();
  }
 }if(threadIdx.x==0){atomicAdd(&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)min(m,n));}
}
template<class T> __global__ void scalar_apply(const T*v,int ldv,const T*tau,int m,int k,T*x,int n,int ldx,bool transpose,int*status){
 __shared__ T red[256];int j=blockIdx.x;
 for(int step=0;step<k;++step){int i=transpose?step:k-1-step;T mx=0;for(int r=i+threadIdx.x;r<m;r+=blockDim.x)mx=max(mx,absval(x[r+size_t(j)*ldx]));mx=block_max(mx,red);T dot=0;
  if(mx)for(int r=i+threadIdx.x;r<m;r+=blockDim.x)dot+=(r==i?T(1):v[r+size_t(i)*ldv])*(x[r+size_t(j)*ldx]/mx);dot=block_sum(dot,red)*tau[i];
  for(int r=i+threadIdx.x;r<m;r+=blockDim.x){T y=mx?(x[r+size_t(j)*ldx]/mx-(r==i?T(1):v[r+size_t(i)*ldv])*dot)*mx:T(0);x[r+size_t(j)*ldx]=y;if(!isfinite(y))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}__syncthreads();
 }
}
template<class T> __global__ void mask_R(T*a,int m,int n,int ld){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(m)*n;i+=size_t(blockDim.x)*gridDim.x)if(int(i%m)>int(i/m))a[i%m+(i/m)*ld]=0;}
template<class T> __global__ void compare_bits(const T*a,const T*b,int nr,int n,int lda,int ldb,int*status){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(nr)*n;i+=size_t(blockDim.x)*gridDim.x){const auto*x=(const unsigned char*)(a+i%nr+(i/nr)*lda);const auto*y=(const unsigned char*)(b+i%nr+(i/nr)*ldb);for(int j=0;j<sizeof(T);++j)if(x[j]!=y[j])atomicCAS(status,0,1);}
}

}
