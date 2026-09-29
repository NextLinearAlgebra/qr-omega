#pragma once
// Cooperative panel factorization: the rows of a panel are split among resident thread blocks, which
// combine scaled norm and dot-product partials in a fixed order.
#include "tiled_kernels.cuh"
#include <cooperative_groups.h>
namespace tqr {
// A GE packet's row support is split among resident CTAs, with typed scaled norm/dot partials in two HBM generations.
// Each CTA reads immutable partials; no numerical atomic or custom spin barrier is used. The next
// generation is prepared before the one cooperative grid barrier per column. This first carrier
// prototype stages a complete local panel. Minipanel WY boundaries and persistent GE/TT tree
// scheduling remain separate work.
//
//  stalls: wait 24.4% / long_scoreboard 23.1% / barrier 22.9% / short 10.9%
//  global loads: 4,982,144 requests carrying 137,169,283 sectors
//                = 27.5 sectors per request, i.e. one sector per lane
//  FP64 issue: 1.69e9 thread instructions against ~14k useful FMAs per
//              CTA-column, so over 90% of it was redundant division
// (1) PARTIAL LAYOUT IS SLOT-MAJOR. Every reader walks the `groups` partials
//    of ONE slot; the old part-major stride put 2+2h doubles between them,
//    so each lane landed in its own 32 B sector. Slot-major makes that walk
//    contiguous. Same buffer, same size, same order of accumulation.
// (2) THE SCALED COLUMN IS STAGED ONCE. x_r = a_rk/mx was recomputed inside
//    each of the h dot products, and v_r = (a_rk/scale)/den inside each of
//    the h-k column updates. Both are now computed once per column -- x_r
//    into shared beside the panel, v_r by normalising the reflector column
//    in place before the update reads it, which is the same expression the
//    old epilogue evaluated, just moved ahead of its consumers.
// (3) THE REFLECTOR SCALARS LIVE IN REGISTERS. tile_reduce already broadcasts
//    to every thread, so the shared publish and its barrier bought nothing;
//    the barrier budget per column is unchanged because the normalisation in
//    (2) needs exactly one barrier before the update.
inline constexpr int coop_partition_limit=128;
// Panel slab plus the staged scaled column. Shared by the launcher, the selector's feasibility test
// and the standalone probe, so no copy of this expression can outlive the layout it was written
// against.
inline size_t coop_shared_bytes(int local_height,int h,size_t word){return size_t(local_height)*(size_t(h)+1)*word;}
template<class T> __device__ __noinline__ void cooperative_prepare(
 const T*a,int ld,int local_rows,int first,int h,int k,T*out,int outs,T*red,T*xs){
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 int start=max(0,k+1-first);T mx=0;
 for(int r=start+tid;r<local_rows;r+=blockDim.x)mx=max(mx,absval(a[r+size_t(k)*ld]));
 mx=tile_reduce<T,true>(mx,red);T sum=0;
 // The scaled column this loop already forms is what every dot below needs.
 if(mx)for(int r=start+tid;r<local_rows;r+=blockDim.x){T x=a[r+size_t(k)*ld]/mx;xs[r]=x;sum+=x*x;}
 sum=tile_reduce<T,false>(sum,red);if(tid==0){out[0]=mx;out[outs]=sum;}
 for(int j=warp;j<h;j+=warps){T dot=0;
  if(j!=k&&mx)for(int r=start+lane;r<local_rows;r+=32)dot+=xs[r]*a[r+size_t(j)*ld];
  dot=warp_add(dot);if(lane==0){out[size_t(2+j)*outs]=dot;out[size_t(2+h+j)*outs]=(k>=first&&k<first+local_rows)?a[k-first+size_t(j)*ld]:T(0);}
 }
}
template<class T> __global__ void cooperative_ge(T*A,int lda,const TilePacket*packets,
 int count,int groups,int local_height,int maximum_h,T*Ts,int b,T*partial,int*status,Witness*w){
 auto grid=cooperative_groups::this_grid();int packet=blockIdx.x/groups,part=blockIdx.x%groups;
 auto p=packets[packet];int h=p.h,first=part*local_height,rows=max(0,min(local_height,p.rows-first));
 int tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;
 T*global=A+p.row+size_t(p.col)*lda,*tri=Ts+size_t(p.tile)*b*b;
 extern __shared__ __align__(16) unsigned char memory[];T*a=reinterpret_cast<T*>(memory);
 T*xs=a+size_t(local_height)*h;
 __shared__ T red[32],scales[128],g[128],ratio[coop_partition_limit];
 const size_t stride=2+2*maximum_h,generation=size_t(count)*groups*stride;
 T*base=partial+size_t(packet)*groups*stride,*mine=base+part;
 for(int i=tid;i<rows*h;i+=blockDim.x)a[i%rows+size_t(i/rows)*local_height]=global[first+i%rows+size_t(i/rows)*lda];
 if(part==0)for(int i=tid;i<b*b;i+=blockDim.x)tri[i]=0;
 __syncthreads();
 for(int j=warp;j<h;j+=warps){T mx=0;for(int r=lane;r<rows;r+=32)mx=max(mx,absval(a[r+size_t(j)*local_height]));mx=warp_maximum(mx);if(lane==0)mine[size_t(j)*groups]=mx;}
 grid.sync();
 for(int j=warp;j<h;j+=warps){T mx=0;for(int i=lane;i<groups;i+=32)mx=max(mx,base[size_t(j)*groups+i]);mx=warp_maximum(mx);if(lane==0)scales[j]=mx;for(int r=lane;r<rows;r+=32)if(mx)a[r+size_t(j)*local_height]/=mx;}
 // Every reader of the scaling generation retires before it becomes partials.
 grid.sync();
 cooperative_prepare(a,local_height,rows,first,h,0,mine,groups,red,xs);
 for(int k=0;k<maximum_h;++k){
  grid.sync();
  if(k<h){T*read=base+size_t(k%2)*generation;const T*sig=read,*ssq=read+groups;
   // The cross-CTA combine of the two scaled-norm generations used to run on thread 0 as two serial
   // loops of length `groups`, i.e. 2*groups DEPENDENT global loads on one lane while every other
   // thread waited at the barrier. At groups=32 that is 64 serial L2-latency round trips per column
   // and it dominated the column step. Reducing across the block instead costs two extra CTA
   // barriers, which are ~0.1us, and leaves the algebra identical term for term. The reduction
   // order changes from ascending-i to tile_reduce's fixed tree; it is still deterministic for a
   // given (threads, groups).
   T tail=0;for(int i=tid;i<groups;i+=blockDim.x)tail=max(tail,sig[i]);
   tail=tile_reduce<T,true>(tail,red);
   const int owner=k/local_height;const T alpha=read[size_t(2+h+k)*groups+owner];
   const T scale=max(tail,absval(alpha));
   // sigma_i/scale is the same ratio the dot combine below needs for every j.
   T s=0;if(tail)for(int i=tid;i<groups;i+=blockDim.x){T r=sig[i]/scale;ratio[i]=r;s+=ssq[i]*r*r;}
   s=tile_reduce<T,false>(s,red);
   T beta=alpha,tau=0,den=1;
   if(tail){const T an=alpha/scale;T sum=an*an+s,bn=-sqrt(sum);if(an<0)bn=-bn;beta=scale*bn;den=an-bn;tau=1-an/bn;}
   if(tau)for(int r=max(0,k+1-first)+tid;r<rows;r+=blockDim.x)a[r+size_t(k)*local_height]=(a[r+size_t(k)*local_height]/scale)/den;
   __syncthreads();
   for(int j=warp;j<h;j+=warps){if(j==k)continue;T dot=0;
    if(tau)for(int i=lane;i<groups;i+=32)dot+=read[size_t(2+j)*groups+i]*ratio[i];
    dot=warp_add(dot);T value=read[size_t(2+h+j)*groups+owner]+(tau?dot/den:T(0));
    if(j<k){if(part==0&&lane==0)g[j]=value;}
    else{T update=tau*value;for(int r=max(0,k+1-first)+lane;r<rows;r+=32){T v=tau?a[r+size_t(k)*local_height]:T(0);a[r+size_t(j)*local_height]-=v*update;}
     if(part==owner&&lane==0)a[k-first+size_t(j)*local_height]-=update;
    }
   }__syncthreads();
   if(part==0){if(tid<k){T sum=0;for(int j=tid;j<k;++j)sum+=tri[tid+size_t(j)*b]*g[j];tri[tid+size_t(k)*b]=-tau*sum;}if(tid==0)tri[k+size_t(k)*b]=tau;}
   if(part==owner&&tid==0)a[k-first+size_t(k)*local_height]=beta;
   __syncthreads();
   if(k+1<h)cooperative_prepare(a,local_height,rows,first,h,k+1,mine+size_t((k+1)%2)*generation,groups,red,xs);
  }
 }
 for(int i=tid;i<rows*h;i+=blockDim.x){int r=i%rows,j=i/rows;T v=a[r+size_t(j)*local_height];if(first+r<=j){v*=scales[j];if(!isfinite(v))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}global[first+r+size_t(j)*lda]=v;}
 if(part==0&&tid==0){atomicAdd(&w->ge,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);}
 // Host counts groups peers per member; the two agree exactly when the launched grid is
 // count*groups.
 if(tid==0){atomicAdd(&w->device_peer_partials[0],1ULL);atomicAdd(&w->device_membership_reports,1ULL);}
}
template<class T> void launch_cooperative_ge(T*A,int ld,const TilePacket*packets,int count,
 int rows,int h,int groups,int threads,T*tri,int b,T*partial,int*status,Witness*w,cudaStream_t stream){
 int device;CU(cudaGetDevice(&device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
 if(!prop.cooperativeLaunch||count<1||groups<1||groups>coop_partition_limit||h<1||h>128||rows<h||threads<32||threads>1024||threads%32)throw std::runtime_error("cooperative_packet_capability_before_modify");
 int local_height=ceildiv(rows,groups);size_t dynamic=coop_shared_bytes(local_height,h,sizeof(T));cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,cooperative_ge<T>));
 if(dynamic>prop.sharedMemPerBlockOptin-attr.sharedSizeBytes)throw std::runtime_error("cooperative_panel_shared_capacity_before_modify");
 CU(cudaFuncSetAttribute(cooperative_ge<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dynamic)));
 int occupancy=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cooperative_ge<T>,threads,dynamic));
 if(size_t(count)*groups>size_t(occupancy)*prop.multiProcessorCount)throw std::runtime_error("cooperative_grid_residency_before_modify");
 void*args[]={&A,&ld,&packets,&count,&groups,&local_height,&h,&tri,&b,&partial,&status,&w};
 CU(cudaLaunchCooperativeKernel((const void*)cooperative_ge<T>,dim3(count*groups),dim3(threads),args,dynamic,stream));
}
}
