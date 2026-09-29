#pragma once
// Tile descriptors and kernels of the tiled factorization: GE, TS and TT packets and merges.
#include "kernels.cuh"
namespace tqr {
struct TilePacket {int row,col,rows,h,tile;};
// Child 0 owns the node: it keeps the merged R, every other child keeps its own reflectors. Radix 2
// is the k=2 case and lays the stack out exactly as the earlier pairwise descriptor did
// (row[0]/row[1] are the former top/bottom, and so on).
inline constexpr int TILE_MAX_RADIX=8;
struct TileMerge {
 int row[TILE_MAX_RADIX],rank[TILE_MAX_RADIX],height[TILE_MAX_RADIX],child[TILE_MAX_RADIX];
 int k,col,h,tile;
 __host__ __device__ int top()const{return row[0];}
 __host__ __device__ int bottom()const{return row[1];}
 __host__ __device__ int top_rank()const{return rank[0];}
 __host__ __device__ int bottom_rank()const{return rank[1];}
 __host__ __device__ int ht()const{return height[0];}
 __host__ __device__ int hb()const{return height[1];}
 __host__ __device__ int top_child()const{return child[0];}
 __host__ __device__ int bottom_child()const{return child[1];}
 __host__ __device__ int rows()const{int s=0;for(int i=0;i<k;++i)s+=height[i];return s;}
};
template<class T> __global__ void tiled_finite_output(const T*a,int nr,int n,int ld,int*status){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(nr)*n;i+=size_t(blockDim.x)*gridDim.x)if(!isfinite(a[i%nr+size_t(i/nr)*ld]))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}
template<class T> __device__ T warp_add(T x){for(int d=16;d;d/=2)x+=__shfl_down_sync(0xffffffff,x,d);return __shfl_sync(0xffffffff,x,0);}
template<class T> __device__ T warp_maximum(T x){for(int d=16;d;d/=2)x=max(x,__shfl_down_sync(0xffffffff,x,d));return __shfl_sync(0xffffffff,x,0);}
template<class T,bool Maximum> __device__ T tile_reduce(T x,T*scratch){
 if(blockDim.x==32)return Maximum?warp_maximum(x):warp_add(x);
 int lane=threadIdx.x%32,warp=threadIdx.x/32,warps=blockDim.x/32;x=Maximum?warp_maximum(x):warp_add(x);if(lane==0)scratch[warp]=x;__syncthreads();
 if(warp==0){x=lane<warps?scratch[lane]:T(0);x=Maximum?warp_maximum(x):warp_add(x);if(lane==0)scratch[0]=x;}__syncthreads();return scratch[0];
}
// Re-derived after reading qr-omega@8e783b3 panel-optimization.md: restrict per-column reductions
// to their consumers and inline only that routine. All 32 lanes must participate. The caller
// publishes the result with a CTA barrier.
template<class T> __device__ __forceinline__ T packet_warp_reflector(T*a,int ld,int s,int k){
 int lane=threadIdx.x%32;T tail=0;
 for(int r=k+1+lane;r<s;r+=32)tail=max(tail,absval(a[r+size_t(k)*ld]));
 tail=warp_maximum(tail);T alpha=a[k+size_t(k)*ld],tau=0,beta=alpha;
 if(tail){T scale=max(tail,absval(alpha)),sum=0;
  for(int r=k+lane;r<s;r+=32){T x=a[r+size_t(k)*ld]/scale;sum+=x*x;}
  sum=warp_add(sum);T an=alpha/scale,bn=-sqrt(sum);if(an<0)bn=-bn;
  beta=scale*bn;T den=an-bn;tau=1-an/bn;
  for(int r=k+1+lane;r<s;r+=32)a[r+size_t(k)*ld]=(a[r+size_t(k)*ld]/scale)/den;
 }
 if(lane==0)a[k+size_t(k)*ld]=beta;return tau;
}
// Each CTA is one independent actual GE packet or an ordered packed TT stack.
// Triangular T is retained; GE V aliases A's strictly lower packet triangle.
// Packet columns are scaled positively before QR and only R is rescaled.
template<class T,bool Shared=false,bool WarpNorm=false,bool Reciprocal=false> __global__ void tiled_ge(T*A,int ld,const TilePacket*packets,int count,T*Ts,int b,int kind,int*status,Witness*w,uint64_t*hist=nullptr,int hist_next=0){
 int id=blockIdx.x;if(id>=count)return;auto p=packets[id];int s=p.rows,h=p.h,tid=threadIdx.x,lane=tid%32,warp=tid/32,warps=blockDim.x/32;T*global=A+p.row+size_t(p.col)*ld,*a=global,*tri=Ts+size_t(p.tile)*b*b;int global_ld=ld;
 extern __shared__ __align__(16) unsigned char panel_memory[];
 if constexpr(Shared){a=reinterpret_cast<T*>(panel_memory);for(int i=tid;i<s*h;i+=blockDim.x)a[i]=global[i%s+size_t(i/s)*ld];ld=s;__syncthreads();}
 __shared__ T scales[128],g[128],red[32],tau,beta,den,normscale,norminv,deninv;  // one slot per warp, up to 1024 threads
 for(int i=tid;i<b*b;i+=blockDim.x)tri[i]=0;
 for(int j=warp;j<h;j+=warps){T mx=0;for(int r=lane;r<s;r+=32)mx=max(mx,absval(a[r+size_t(j)*ld]));mx=warp_maximum(mx);if(lane==0)scales[j]=mx;
  T inv=0;if constexpr(Reciprocal){if(lane==0&&mx){inv=T(1)/mx;if(!isfinite(inv))inv=0;}inv=__shfl_sync(0xffffffff,inv,0);}
  for(int r=lane;r<s;r+=32)if(mx){if constexpr(Reciprocal){if(inv)a[r+size_t(j)*ld]*=inv;else a[r+size_t(j)*ld]/=mx;}else a[r+size_t(j)*ld]/=mx;}}
 __syncthreads();
 for(int k=0;k<h;++k){
  if constexpr(WarpNorm){
   if(warp==0){T value=packet_warp_reflector(a,ld,s,k);if(lane==0){tau=value;tri[k+size_t(k)*b]=value;}}
   __syncthreads();
  }else{
  T tail=0;for(int r=k+1+tid;r<s;r+=blockDim.x)tail=max(tail,absval(a[r+size_t(k)*ld]));
  tail=tile_reduce<T,true>(tail,red);
  const T alpha=a[k+size_t(k)*ld],ns=max(tail,absval(alpha));T bet=alpha,tu=0,dn=1;
  if(tail){T sum=0;for(int r=k+tid;r<s;r+=blockDim.x){T x=a[r+size_t(k)*ld]/ns;sum+=x*x;}
   sum=tile_reduce<T,false>(sum,red);   // its barrier also orders the alpha read before the beta write
   const T an=alpha/ns;T bn=-sqrt(sum);if(an<0)bn=-bn;bet=ns*bn;dn=an-bn;tu=1-an/bn;
  }
  if(tid==0){tau=tu;if(tail){a[k+size_t(k)*ld]=bet;tri[k+size_t(k)*b]=tu;}}
  if(tail)for(int r=k+1+tid;r<s;r+=blockDim.x)a[r+size_t(k)*ld]=(a[r+size_t(k)*ld]/ns)/dn;
  __syncthreads();
  }
  for(int j=warp;j<k;j+=warps){T dot=lane==0?a[k+size_t(j)*ld]:T(0);
#pragma unroll 4
   for(int r=k+1+lane;r<s;r+=32)dot+=a[r+size_t(j)*ld]*a[r+size_t(k)*ld];dot=warp_add(dot);if(lane==0)g[j]=dot;}
  __syncthreads();if(tid<k){T sum=0;for(int j=tid;j<k;++j)sum+=tri[tid+size_t(j)*b]*g[j];tri[tid+size_t(k)*b]=-tau*sum;}
  for(int j=k+1+warp;j<h;j+=warps){T dot=lane==0?a[k+size_t(j)*ld]:T(0);
#pragma unroll 4
   for(int r=k+1+lane;r<s;r+=32)dot+=a[r+size_t(k)*ld]*a[r+size_t(j)*ld];dot=warp_add(dot)*tau;
   if(lane==0)a[k+size_t(j)*ld]-=dot;
#pragma unroll 4
   for(int r=k+1+lane;r<s;r+=32)a[r+size_t(j)*ld]-=a[r+size_t(k)*ld]*dot;
  }__syncthreads();
 }
 for(int i=tid;i<h*h;i+=blockDim.x){int r=i%h,j=i/h;if(r<=j){T y=a[r+size_t(j)*ld]*scales[j];a[r+size_t(j)*ld]=y;if(!isfinite(y))atomicCAS(status,0,UNREPRESENTABLE_RESULT);}}
  if constexpr(Shared){__syncthreads();for(int i=tid;i<s*h;i+=blockDim.x)global[i%s+size_t(i/s)*global_ld]=a[i];}
  if(tid==0){if(kind==0)atomicAdd(&w->ge,1ULL);else atomicAdd(&w->tt,1ULL);atomicAdd(&w->reflectors,(unsigned long long)h);}
  // The count comes from executed launch geometry (blockDim.x of THIS block), not from any
  // host-supplied c. kind mapping matches hh()'s host record (0 -> PANEL_GE, else TT_GE). Blocks
  // with id>=count return before this point, so only participating blocks report: this IS the
  // membership. GPU level: one partial per block (the unreplicated product itself). Block level:
  // blockDim.x peers (kept distinct per the nesting rule).
  if(tid==0){atomicAdd(&w->device_peer_partials[kind==0?0:1],1ULL);
   atomicAdd(&w->device_block_peers[kind==0?0:1],(unsigned long long)blockDim.x);
   atomicAdd(&w->device_membership_reports,1ULL);}
  // Fused ordered-QR history publication: the producing block publishes its own entry exactly once,
  // with the same CAS semantics as tile_history_begin. No barrier is needed: nothing reads hist
  // concurrently (consumers are later kernels: children checks, advance CAS chain, end check), and
  // kernel completion already orders all of this block's global writes before any later kernel on
  // the single engine stream observes hist.
  if(hist&&tid==0){if(atomicCAS((unsigned long long*)(hist+p.tile),0ULL,(unsigned long long)hist_next)!=0ULL)atomicCAS(status,0,HISTORY_ERROR);}
}
template<class T> __global__ void pack_ge_v(const T*A,int ld,const TilePacket*packets,int count,T*V,int leaf,int b){
 int packet=blockIdx.y;if(packet>=count)return;auto p=packets[packet];T*out=V+size_t(packet)*leaf*b;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<leaf*b;i+=blockDim.x*gridDim.x){int r=i%leaf,j=i/leaf;out[i]=(r>=p.rows||j>=p.h||r<j)?T(0):r==j?T(1):A[p.row+r+size_t(p.col+j)*ld];}
}
// Launch width of the V packs: enough 128-thread CTAs to cover leaf x w elements at ~4 per thread,
// capped at 2048 (16 per SM). The pack is a copy of leaf x w words; 32 CTAs ran it at
// ~250 GB/s. The element-to-thread map is a grid-stride loop, so the bytes written are identical at
// any width.
inline int pack_v_grid(int leaf,int w){const long long e=(long long)leaf*std::max(1,w);return int(std::max(32LL,std::min(2048LL,(e+511)/512)));}
// Same pack restricted to the panel columns [j0,j1) -- the window carrier packs
// one window's V as soon as that window is factored, so the deferred update of
// the columns beyond it (Eq apply, one level up) has an operand with its unit
// diagonal and its zeros already in place.
template<class T> __global__ void pack_ge_v_cols(const T*A,int ld,const TilePacket*packets,int count,T*V,int leaf,int b,int j0,int j1){
 int packet=blockIdx.y;if(packet>=count)return;auto p=packets[packet];T*out=V+size_t(packet)*leaf*b;
 const int w=j1-j0;if(w<=0)return;
 // (column, row-chunk) items with one 32-bit divide per item instead of a 64-bit i%leaf, i/leaf per element
 // (instruction-bound at ~1 TB/s). Every element gets the same value at the same address, so the packed V is
 // bit-identical.
 const int CH=int(blockDim.x)*8,rch=(leaf+CH-1)/CH;
 for(int it=blockIdx.x;it<w*rch;it+=gridDim.x){
  const int jj=it/rch,rc=it-jj*rch,j=j0+jj,r1=min(leaf,(rc+1)*CH);
  const T*src=A+p.row+size_t(p.col+j)*ld;T*dst=out+size_t(j)*leaf;
  for(int r=rc*CH+int(threadIdx.x);r<r1;r+=blockDim.x)dst[r]=(r>=p.rows||j>=p.h||r<j)?T(0):r==j?T(1):src[r];}
}
// Eq compose, one level up: the off-diagonal block of T that joins the windows
// already factored (V1, T1) to the window just factored (V2, T2),
// T12 = -T1 (V1^T V2) T2, with G = V1^T V2 formed by the carried product that
// precedes this call. One block per column of the block; S = G T2 is held in
// shared so the triangular sweep needs no second launch.
template<class T> __global__ void tile_compose_offdiag(T*tri,int b,const T*G,int w0,int sw){
 const int c=blockIdx.x;if(c>=sw)return;
 extern __shared__ __align__(16) unsigned char raw[];T*S=reinterpret_cast<T*>(raw);
 for(int i=threadIdx.x;i<w0;i+=blockDim.x){T acc=0;
  for(int j=0;j<=c;++j)acc+=G[size_t(i)+size_t(j)*w0]*tri[w0+j+size_t(w0+c)*b];
  S[i]=acc;}
 __syncthreads();
 for(int i=threadIdx.x;i<w0;i+=blockDim.x){T acc=0;
  for(int j=i;j<w0;++j)acc+=tri[i+size_t(j)*b]*S[j];
  tri[i+size_t(w0+c)*b]=-acc;}
}
// V_g of g consecutive eliminations k..k+g-1 is the unit-lower pack of the g*b-wide column block
// [col, col+H) from row `row`: eliminations j>0 are zero above their own diagonal, which is exactly r
// < column. rows <= ldv; rows beyond are never read.
template<class T> __global__ void pack_v_block(const T*A,int ld,int row,int col,int rows,int H,T*V,int ldv){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<(long long)rows*H;i+=(long long)blockDim.x*gridDim.x){
  const int r=int(i%rows),j=int(i/rows);V[size_t(r)+size_t(j)*ldv]=r<j?T(0):r==j?T(1):A[row+r+size_t(col+j)*ld];}
}
// V_g is packed in that order; T_g is permuted symmetrically (P^T T_g P), so Q = I - V_g T_g V_g^T is
// unchanged.
template<class T> __global__ void pack_v_block_perm(const T*A,int ld,int row,int col,int rows,int H,const int*perm,T*V,int ldv){
 // (column, row-chunk) items, one perm load and one 32-bit divide per item (was a 64-bit i%rows,
 // i/rows per element: instruction-bound). Same value to the same address: bit-identical V_g.
 const int CH=int(blockDim.x)*8,rch=(rows+CH-1)/CH;
 for(int it=blockIdx.x;it<H*rch;it+=gridDim.x){
  const int p=it/rch,rc=it-p*rch,j=perm[p],r1=min(rows,(rc+1)*CH);
  const T*src=A+row+size_t(col+j)*ld;T*dst=V+size_t(p)*ldv;
  for(int r=rc*CH+int(threadIdx.x);r<r1;r+=blockDim.x)dst[r]=r<j?T(0):r==j?T(1):src[r];}
}
template<class T> __global__ void permute_t(const T*Tn,int H,const int*perm,T*Tp){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H*H;i+=blockDim.x*gridDim.x){const int r=i%H,c=i/H;Tp[i]=Tn[perm[r]+size_t(perm[c])*H];}
}
// T_g = [[T1, -T1 G T2],[0, T2]] (ld 2b) for Q1 Q2 = I - [V1 V2] T_g [V1 V2]^T, G = V1^T V2. T1, T2
// upper triangular in b x b blocks of ld b (only their upper parts are read; T_g's lower parts are
// written as explicit zeros). One block per column of T_g.
template<class T> __global__ void compose_tg(T*Tg,const T*T1,const T*T2,const T*G,int b){
 const int c=blockIdx.x,ldg=2*b;if(c>=ldg)return;
 extern __shared__ __align__(16) unsigned char raw[];T*S=reinterpret_cast<T*>(raw);
 if(c<b){for(int i=threadIdx.x;i<ldg;i+=blockDim.x)Tg[i+size_t(c)*ldg]=(i<=c)?T1[i+size_t(c)*b]:T(0);return;}
 const int cc=c-b;
 for(int i=threadIdx.x;i<b;i+=blockDim.x){T acc=0;for(int j=0;j<=cc;++j)acc+=G[size_t(i)+size_t(j)*b]*T2[j+size_t(cc)*b];S[i]=acc;}
 __syncthreads();
 for(int i=threadIdx.x;i<b;i+=blockDim.x){T acc=0;for(int j=i;j<b;++j)acc+=T1[i+size_t(j)*b]*S[j];
  Tg[i+size_t(c)*ldg]=-acc;Tg[b+i+size_t(c)*ldg]=(i<=cc)?T2[i+size_t(cc)*b]:T(0);}
}
// T_g[jb:jb+b, jb+c] = upper(T_{j+1})[:,c]. T_g must be zeroed before step 0. One block per column
// c < b.
template<class T> __global__ void compose_tg_step(T*Tg,int H,int j,int b,const T*G,const T*Tn,Witness*wit=nullptr){
 // This launch follows all j G products on the same stream. Window compose
 // products remain counted by their enclosing panel interval.
 if(wit&&blockIdx.x==0&&threadIdx.x==0)atomicAdd(&wit->device_products[6],(unsigned long long)j);
 const int c=blockIdx.x;if(c>=b)return;const int jb=j*b,col=jb+c;
 extern __shared__ __align__(16) unsigned char raw[];T*S=reinterpret_cast<T*>(raw);
 for(int i=threadIdx.x;i<b;i+=blockDim.x)Tg[jb+i+size_t(col)*H]=(i<=c)?Tn[i+size_t(c)*b]:T(0);
 if(!j)return;
 for(int i=threadIdx.x;i<jb;i+=blockDim.x){const T*gi=G+size_t(i/b)*b*b+(i%b);T acc=0;for(int l=0;l<=c;++l)acc+=gi[size_t(l)*b]*Tn[l+size_t(c)*b];S[i]=acc;}
 __syncthreads();
 for(int i=threadIdx.x;i<jb;i+=blockDim.x){T acc=0;for(int l=i;l<jb;++l)acc+=Tg[i+size_t(l)*H]*S[l];Tg[i+size_t(col)*H]=-acc;}
}
// TILED Eq-compose step (same algebra as compose_tg_step). compose_tg_step gave one CTA per output column; each CTA
// re-read the whole composed T_prev (jb x jb) from L2 in a latency-bound triangular dot per row. nsys tf32 65536^2
// (aggregate 4): j=1/2/3 steps 30 / 139 / 322 us median, 56.5 ms per factorization on the main stream at every group
// boundary, growing ~j^2 (aggregate 8 worse). Here the step is two 32x32-tiled products:
//  S = G_j Tn            (jb x b, G_j = V_prev^T V_j stacked as j b x b blocks, Tn upper triangular: l <= c)
//  T_g[0:jb, jb+c] = -T_prev S   (T_prev upper triangular: l >= i)
// plus the diagonal block copy. Tiles read each operand once per tile; summation per element is in 32-wide chunks of
// ascending l (deterministic, fixed order). Same values up to rounding order.
constexpr int kComposeTile=32;
template<class T> __global__ void compose_s_tiled(const T* __restrict__ G,int b,int jb,const T* __restrict__ Tn,T* __restrict__ S){
 constexpr int CT=kComposeTile;__shared__ T Gs[CT][CT+1],Ts[CT][CT+1];
 const int i0=blockIdx.x*CT,c0=blockIdx.y*CT,tx=threadIdx.x,ty=threadIdx.y;T acc[4]={T(0),T(0),T(0),T(0)};
 const int lmax=min(b,c0+CT);   // Tn upper triangular: l <= c < c0+CT
 for(int l0=0;l0<lmax;l0+=CT){
  for(int r=ty;r<CT;r+=8){const int l=l0+r,i=i0+tx;Gs[r][tx]=(i<jb&&l<b)?G[size_t(i/b)*b*b+(i%b)+size_t(l)*b]:T(0);   // Gs[l][i]
   const int c=c0+r,ll=l0+tx;Ts[r][tx]=(c<b&&ll<b&&ll<=c)?Tn[ll+size_t(c)*b]:T(0);}                                  // Ts[c][l]
  __syncthreads();
  #pragma unroll 8
  for(int k=0;k<CT;++k){const T g=Gs[k][tx];
   #pragma unroll
   for(int q=0;q<4;++q)acc[q]+=g*Ts[ty+8*q][k];}
  __syncthreads();}
 const int i=i0+tx;
 #pragma unroll
 for(int q=0;q<4;++q){const int c=c0+ty+8*q;if(i<jb&&c<b)S[i+size_t(c)*jb]=acc[q];}
}
template<class T> __global__ void compose_n_tiled(T* __restrict__ Tg,int H,int jb,int b,const T* __restrict__ S){
 constexpr int CT=kComposeTile;__shared__ T Ps[CT][CT+1],Ss[CT][CT+1];
 const int i0=blockIdx.x*CT,c0=blockIdx.y*CT,tx=threadIdx.x,ty=threadIdx.y;T acc[4]={T(0),T(0),T(0),T(0)};
 const int i=i0+tx;
 for(int l0=i0;l0<jb;l0+=CT){   // T_prev upper triangular: l >= i >= i0
  for(int r=ty;r<CT;r+=8){const int l=l0+r,ii=i0+tx;Ps[r][tx]=(ii<jb&&l<jb&&l>=ii)?Tg[ii+size_t(l)*H]:T(0);   // Ps[l][i]
   const int c=c0+r,ll=l0+tx;Ss[r][tx]=(c<b&&ll<jb)?S[ll+size_t(c)*jb]:T(0);}                                    // Ss[c][l]
  __syncthreads();
  #pragma unroll 8
  for(int k=0;k<CT;++k){const T p=Ps[k][tx];
   #pragma unroll
   for(int q=0;q<4;++q)acc[q]+=p*Ss[ty+8*q][k];}
  __syncthreads();}
 #pragma unroll
 for(int q=0;q<4;++q){const int c=c0+ty+8*q;if(i<jb&&c<b)Tg[i+size_t(jb+c)*H]=-acc[q];}
}
template<class T> __global__ void compose_diag_copy(T*Tg,int H,int j,int b,const T*Tn,Witness*wit=nullptr){
 if(wit&&blockIdx.x==0&&threadIdx.x==0)atomicAdd(&wit->device_products[6],(unsigned long long)j);   // as compose_tg_step
 const int jb=j*b;
 for(int e=blockIdx.x*blockDim.x+threadIdx.x;e<b*b;e+=gridDim.x*blockDim.x){const int i=e%b,c=e/b;
  Tg[jb+i+size_t(jb+c)*H]=(i<=c)?Tn[i+size_t(c)*b]:T(0);}
}
template<class T> __global__ void extract_r(const T*A,int ld,int row,int col,int rows,int h,T*R,int ldr){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*h;i+=blockDim.x*gridDim.x){int r=i%rows,j=i/rows;R[r+size_t(j)*ldr]=r<=j?A[row+r+size_t(col+j)*ld]:T(0);}
}
template<class T> __global__ void scatter_upper(const T*R,int ldr,T*A,int ld,int row,int col,int rows,int h){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*h;i+=blockDim.x*gridDim.x){int r=i%rows,j=i/rows;if(r<=j)A[row+r+size_t(col+j)*ld]=R[r+size_t(j)*ldr];}
}
// One k-child row map per block, staged in shared memory: the descriptor grew
// with the radix, so it is never copied per thread. mbase[c]+r is the source
// row of stack row r, moff[c] its row inside child c, mbound[c] the first row
// after child c.
#define TQR_TILE_ROWMAP(DESC) \
 __shared__ int mk,mh,mcol,mbase[TILE_MAX_RADIX],mbound[TILE_MAX_RADIX],moff[TILE_MAX_RADIX]; \
 if(threadIdx.x==0){const TileMerge&e_=(DESC);mk=e_.k;mh=e_.h;mcol=e_.col;int off_=0; \
  for(int c=0;c<e_.k;++c){mbase[c]=e_.row[c]-begin-off_;moff[c]=off_;off_+=e_.height[c];mbound[c]=off_;}} \
 __syncthreads();
template<class T> __global__ void tile_stack(const T*A,int ld,int begin,const TileMerge*es,int count,T*stack,int b,int radix){
 int t=blockIdx.y;if(t>=count)return;T*out=stack+size_t(t)*radix*b*b;const int m=radix*b;
 TQR_TILE_ROWMAP(es[t]);
 const int stride=blockDim.x*gridDim.x,mask=m-1,sh=__ffs(m)-1;int i0=blockIdx.x*blockDim.x+threadIdx.x;
 if(!(m&mask)&&!(stride&mask)){const int r=i0&mask;int src=-1,lr=0;
  for(int c=0;c<mk;++c)if(r<mbound[c]){src=mbase[c]+r;lr=r-moff[c];break;}
  for(int j=i0>>sh;j<b;j+=stride>>sh)out[size_t(j)*m+r]=(j<mh&&src>=0&&lr<=j)?A[src+size_t(mcol+j)*ld]:T(0);
 }else for(int i=i0;i<m*b;i+=stride){int r=i%m,j=i/m;T v=0;
  if(j<mh)for(int c=0;c<mk;++c)if(r<mbound[c]){if(r-moff[c]<=j)v=A[mbase[c]+r+size_t(mcol+j)*ld];break;}
  out[i]=v;}
}
template<class T> __global__ void tile_scatter_factors(T*A,int ld,int begin,const TileMerge*es,int count,const T*stack,int b,int radix){
 int t=blockIdx.y;if(t>=count)return;const TileMerge&e=es[t];const T*in=stack+size_t(t)*radix*b*b;const int m=radix*b;
 // Child 0 receives the merged R over its full h rows exactly as the pairwise
 // scatter did; every later child receives only the rows it contributed.
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<e.h*e.h;i+=blockDim.x*gridDim.x){int r=i%e.h,j=i/e.h;if(r<=j){
  A[e.row[0]-begin+r+size_t(e.col+j)*ld]=in[r+size_t(j)*m];int off=e.height[0];
  for(int c=1;c<e.k;++c){if(r<e.height[c])A[e.row[c]-begin+r+size_t(e.col+j)*ld]=in[off+r+size_t(j)*m];off+=e.height[c];}}}
}
template<class T> __global__ void tile_merge_v(const T*A,int ld,int begin,const TileMerge*es,int count,T*V,int b,int radix){
 int t=blockIdx.y;if(t>=count)return;T*out=V+size_t(t)*radix*b*b;const int m=radix*b;
 // V = [I; V_1; ...; V_{k-1}]: the stacked children are upper triangular, so
 // column j's reflector is zero below the pivot inside child 0 and upper
 // triangular inside every later child. Identical to [I;V_b] at k=2.
 TQR_TILE_ROWMAP(es[t]);
 const int stride=blockDim.x*gridDim.x,mask=m-1,sh=__ffs(m)-1;int i0=blockIdx.x*blockDim.x+threadIdx.x;
 if(!(m&mask)&&!(stride&mask)){const int r=i0&mask;int src=-1,lr=0,cc=-1;
  for(int c=0;c<mk;++c)if(r<mbound[c]){cc=c;src=mbase[c]+r;lr=r-moff[c];break;}
  for(int j=i0>>sh;j<b;j+=stride>>sh){T v=0;
   if(j<mh&&cc>=0)v=cc?((lr<=j)?A[src+size_t(mcol+j)*ld]:T(0)):(r==j?T(1):T(0));
   out[size_t(j)*m+r]=v;}
 }else for(int i=i0;i<m*b;i+=stride){int r=i%m,j=i/m;T v=0;
  if(j<mh)for(int c=0;c<mk;++c)if(r<mbound[c]){v=c?((r-moff[c]<=j)?A[mbase[c]+r+size_t(mcol+j)*ld]:T(0)):(r==j?T(1):T(0));break;}
  out[i]=v;}
}
template<class T,bool Scatter> __global__ void tile_merge_x(T*A,int ld,int begin,const TileMerge*es,int count,T*X,int b,int col,int q,int radix){
 int t=blockIdx.y;if(t>=count)return;T*out=X+size_t(t)*radix*b*q;const int m=radix*b;
 // b comes from the admitted catalog {16,32,64,128} and radix from {2,4,8}, so
 // radix*b is a power of two and the row split is a mask and a shift. Every
 // admitted launch also has a grid stride that is a multiple of radix*b, which
 // makes the row -- and therefore the child lookup -- loop invariant: the k-way
 // search runs once per thread, not once per element.
 TQR_TILE_ROWMAP(es[t]);
 const int stride=blockDim.x*gridDim.x,mask=m-1,sh=__ffs(m)-1;int i0=blockIdx.x*blockDim.x+threadIdx.x;
 if(!(m&mask)&&!(stride&mask)){const int r=i0&mask;int src=-1;
  for(int c=0;c<mk;++c)if(r<mbound[c]){src=mbase[c]+r;break;}
  for(int j=i0>>sh;j<q;j+=stride>>sh){
   if(src>=0){T*ptr=A+src+size_t(col+j)*ld;if constexpr(Scatter)*ptr=out[size_t(j)*m+r];else out[size_t(j)*m+r]=*ptr;}
   else if constexpr(!Scatter)out[size_t(j)*m+r]=0;}
 }else for(int i=i0;i<m*q;i+=stride){int r=i%m,j=i/m,src=-1;
  for(int c=0;c<mk;++c)if(r<mbound[c]){src=mbase[c]+r;break;}
  if(src>=0){T*ptr=A+src+size_t(col+j)*ld;if constexpr(Scatter)*ptr=out[i];else out[i]=*ptr;}else if constexpr(!Scatter)out[i]=0;}
}
template<class T> __global__ void tile_bottom_v(const T*A,int ld,int row,int col,int hb,int h,T*V,int b){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<b*b;i+=blockDim.x*gridDim.x){int r=i%b,j=i/b;V[i]=(r<hb&&j<h&&r<=j)?A[row+r+size_t(col+j)*ld]:T(0);}
}
// Structural maps, not contractions: one copy/subtraction per output entry. The kernel that
// performs the actual operation reports its singleton map and (for D) its physical owner commit.
__device__ inline void tile_identity_witness(Witness*w,int kind,bool commit){
 if(w&&blockIdx.x==0&&threadIdx.x==0){
  atomicAdd(&w->device_peer_partials[kind],1ULL);
  atomicAdd(&w->device_block_peers[kind],1ULL);
  atomicAdd(&w->device_membership_reports,1ULL);
  witness_levels(w,kind,1,1,1,1);
  if(commit)atomicAdd(&w->device_physical_commits,1ULL);
 }
}
template<class T> __global__ void tile_copy_identity(const T*A,int ld,int row,int col,int rows,int q,T*W,int ldw,Witness*wit){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*q;i+=blockDim.x*gridDim.x){int r=i%rows,j=i/rows;W[r+size_t(j)*ldw]=A[row+r+size_t(col+j)*ld];}
 tile_identity_witness(wit,2,false);
}
template<class T> __global__ void tile_add_identity(T*A,int ld,int row,int col,int rows,int q,const T*Z,int ldz,Witness*wit=nullptr){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*q;i+=blockDim.x*gridDim.x){int r=i%rows,j=i/rows;A[row+r+size_t(col+j)*ld]-=Z[r+size_t(j)*ldz];}
 tile_identity_witness(wit,4,true);
}
template<class T> __global__ void tile_add_identity_t(T*A,int ld,int row,int col,int rows,int q,const T*Zt,int ldzt,Witness*wit=nullptr){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*q;i+=blockDim.x*gridDim.x){int r=i%rows,j=i/rows;A[row+r+size_t(col+j)*ld]-=Zt[j+size_t(r)*ldzt];}
 tile_identity_witness(wit,4,true);
}
__global__ inline void tile_history_begin(uint64_t*hist,const TilePacket*p,int count,int next,int*status){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count&&atomicCAS((unsigned long long*)(hist+p[i].tile),0ULL,(unsigned long long)next)!=0ULL)atomicCAS(status,0,HISTORY_ERROR);}
__global__ inline void tile_history_commit(uint64_t*hist,const TilePacket*p,int count,int expected,int next,int*status,Witness*w){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){if(atomicCAS((unsigned long long*)(hist+p[i].tile),(unsigned long long)expected,(unsigned long long)next)!=(unsigned long long)expected)atomicCAS(status,0,HISTORY_ERROR);atomicAdd(&w->commits,1ULL);
// The per-thread commits above count one mark per tile; device_history_marks carries the same
// number once per launch so the receipt can separate "the cursor advanced" from "X was written"
// (device_physical_commits). A repeated numerical write is invisible to commits alone.
if(threadIdx.x==0&&blockIdx.x==0)atomicAdd(&w->device_history_marks,(unsigned long long)count);}}
__global__ inline void tile_history_children(const uint64_t*hist,const TileMerge*e,int count,int n,int*status){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;for(int c=0;c<e[i].k;++c)if(hist[e[i].child[c]]!=uint64_t(n)){atomicCAS(status,0,HISTORY_ERROR);return;}}
__global__ inline void tile_history_one(const uint64_t*hist,int child,int n,int*status){if(threadIdx.x==0&&hist[child]!=uint64_t(n))atomicCAS(status,0,HISTORY_ERROR);}
// The child must have committed AT LEAST through n (its far window may already advance concurrently on another lane).
__global__ inline void tile_history_atleast(const uint64_t*hist,int child,int n,int*status){if(threadIdx.x==0&&hist[child]<uint64_t(n))atomicCAS(status,0,HISTORY_ERROR);}
__global__ inline void tile_history_end(const uint64_t*hist,size_t count,int n,int*status){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=size_t(blockDim.x)*gridDim.x)if(hist[i]!=uint64_t(n))atomicCAS(status,0,HISTORY_ERROR);}
}
