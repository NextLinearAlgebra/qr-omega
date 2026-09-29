#pragma once
// Common types and utilities: error checks, device buffers, streams and events, JSON, row distributions.
#include "precision_mode.hpp"
#include <cuda_runtime.h>
#include <json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tqr {
using json = nlohmann::json;
inline void cuda_check(cudaError_t e, const char* where) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
#define CU(x) ::tqr::cuda_check((x), #x)
inline double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline size_t checked_mul(size_t a,size_t b) {
  if (b && a>SIZE_MAX/b) throw std::runtime_error("descriptor_overflow"); return a*b;
}
inline size_t checked_add(size_t a,size_t b) {
  if (a>SIZE_MAX-b) throw std::runtime_error("descriptor_overflow"); return a+b;
}
inline int ceildiv(int a,int b) { return a/b+(a%b!=0); }
inline std::string digest(const std::string& s) {
  // Noncryptographic content key; release manifest supplies SHA-256 binary identity.
  uint64_t h=1469598103934665603ULL; for(unsigned char c:s) {h^=c;h*=1099511628211ULL;}
  return std::to_string(h);
}
template<class T> inline std::string precision() { return sizeof(T)==4?"fp32":"fp64"; }
inline json read_json(const std::string& path) { std::ifstream f(path); if(!f) throw std::runtime_error("cannot read "+path); json j;f>>j;return j; }
inline void write_json(const std::string& path,const json& j) { std::ofstream f(path);if(!f)throw std::runtime_error("cannot write "+path);f<<j.dump(2)<<'\n'; }
struct Stream {
  cudaStream_t s=nullptr;
  Stream() {CU(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
  // high: the device's greatest stream priority.
  explicit Stream(bool high){if(!high){CU(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));return;}
   int lo=0,hi=0;CU(cudaDeviceGetStreamPriorityRange(&lo,&hi));CU(cudaStreamCreateWithPriority(&s,cudaStreamNonBlocking,hi));}
  // An explicit priority (middle-priority near lane; main-group side slots at the greatest priority).
  Stream(bool,int prio){CU(cudaStreamCreateWithPriority(&s,cudaStreamNonBlocking,prio));}
  ~Stream(){if(s)cudaStreamDestroy(s);}
  operator cudaStream_t() const {return s;}
  void sync() const {CU(cudaStreamSynchronize(s));}
  Stream(const Stream&)=delete;
};
struct Event {
  cudaEvent_t e=nullptr;
  Event(){CU(cudaEventCreate(&e));} ~Event(){if(e)cudaEventDestroy(e);}
  void record(cudaStream_t s){CU(cudaEventRecord(e,s));}
  void wait(cudaStream_t s){CU(cudaStreamWaitEvent(s,e,0));}
};
template<class T> struct Buffer {
  T* p=nullptr;size_t n=0;bool owns=true;
  Buffer()=default; explicit Buffer(size_t count){alloc(count);}
  Buffer(T* view,size_t count):p(view),n(count),owns(false){}
  ~Buffer(){if(p&&owns)cudaFree(p);}
  Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
  Buffer(Buffer&& x) noexcept:p(x.p),n(x.n),owns(x.owns){x.p=nullptr;x.n=0;}
  Buffer& operator=(Buffer&& x) noexcept {if(p&&owns)cudaFree(p);p=x.p;n=x.n;owns=x.owns;x.p=nullptr;x.n=0;return *this;}
  void alloc(size_t count){if(p)throw std::runtime_error("double allocation");n=count;if(n)CU(cudaMalloc(&p,checked_mul(n,sizeof(T))));}
  void zero(cudaStream_t s){if(n)CU(cudaMemsetAsync(p,0,n*sizeof(T),s));}
  void upload(const std::vector<T>& v,cudaStream_t s){if(v.size()>n)throw std::runtime_error("upload extent");if(!v.empty())CU(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s));}
  std::vector<T> download() const {std::vector<T> v(n);if(n)CU(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
// The isolated core set a measurement must run on. Defaults to this node's
// 12-15; TQR_REQUIRED_CPUS declares a different one on another machine, whose
// allocator hands out different cores. The DISCIPLINE is unchanged -- a
// measurement still has to run on a declared, isolated set and say which --
// only the hardcoded identity of that set moves into the environment.
inline std::string required_cpu_set(){
 const char* e=std::getenv("TQR_REQUIRED_CPUS");
 return (e&&*e)?std::string(e):std::string("12-15");
}
inline int descriptor_dimension(int x){if(x<0)throw std::runtime_error("negative_dimension");return x;}
inline int descriptor_parts(int x){if(x<1)throw std::runtime_error("invalid_parts");return x;}
inline int descriptor_ld(int nr,int pad){if(pad<0||int64_t(nr)+pad>INT_MAX)throw std::runtime_error("invalid_ld");return std::max(1,nr+pad);}
inline int owner_of(int row,int m,int p) {return m?std::min(p-1,int((int64_t(row+1)*p-1)/m)):0;}
inline int row_begin(int m,int p,int r){return int(int64_t(m)*r/p);}
struct RowMap {
 int m=0,p=1,blk=0;
 RowMap()=default;
 RowMap(int m_,int p_,int blk_):m(m_),p(p_),blk(blk_){}
 bool cyclic()const{return blk>0&&p>1;}
 int owner(int g)const{return cyclic()?int((int64_t(g)/blk)%p):owner_of(g,m,p);}
 // rows of rank r whose global index is < g
 int lb(int r,int g)const{
  if(!cyclic()){const int b0=row_begin(m,p,r),n0=row_begin(m,p,r+1)-b0;return std::min(n0,std::max(0,g-b0));}
  const int64_t cyc=int64_t(blk)*p;return int((int64_t(g)/cyc)*blk+std::min<int64_t>(blk,std::max<int64_t>(0,int64_t(g)%cyc-int64_t(r)*blk)));}
 int local_rows(int r)const{return cyclic()?lb(r,m):row_begin(m,p,r+1)-row_begin(m,p,r);}
 int to_local(int g)const{return cyclic()?int((int64_t(g)/(int64_t(blk)*p))*blk+g%blk):g-row_begin(m,p,owner_of(g,m,p));}
 int to_global(int r,int l)const{return cyclic()?int((int64_t(l)/blk)*blk*p+int64_t(r)*blk+l%blk):row_begin(m,p,r)+l;}
 // The affine offset the engine's kernels subtract from descriptor rows: 0 under cyclic (descriptor rows are local).
 int begin(int r)const{return cyclic()?0:row_begin(m,p,r);}
 // One past the last global row of g's contiguous run on its owner (a panel's diagonal block may not cross it).
 int run_end(int g)const{return cyclic()?int(std::min<int64_t>(m,(int64_t(g)/blk+1)*blk)):row_begin(m,p,owner_of(g,m,p)+1);}
 // Descriptor row key of local row l on rank r: global under contiguous, local under cyclic.
 int key(int r,int l)const{return cyclic()?l:row_begin(m,p,r)+l;}
};
// Development harness only. Production instantiation/replay takes an explicit row_block. TQR_ROW_BLOCK=blk (0 =
// contiguous).
inline int tiled_row_block(){static int v=[]{const char*e=std::getenv("TQR_ROW_BLOCK");return (e&&*e)?std::max(0,std::atoi(e)):0;}();return v;}
inline int tiled_row_block_for(int b){const int k=tiled_row_block();return (k>0&&b>0)?std::max(b,k/b*b):0;}
struct RunOptions {
 int m=0,n=0,active=0,b=0,leaf=0,c=0,d=0,threads=0,strip=0,reps=1,pad=0,radix=0,gradix=0;
 int zc=0,dc=0;
 // Forced BLOCK-level apply carriers. 0 = let the selector choose from measurement.
 int wcb=0,zcb=0,dcb=0;
 // Forced panel-carrier staged column window. 0 = let the selector choose; h/0 both mean
 // unwindowed.
 int csw=0;
 std::string tree="",input="random",profile="",output="",plan_file="",cases_file="",mode="run",dtype="fp64";
 size_t budget=SIZE_MAX; int search_budget=192; bool validate=true,reference=true,trace=true;
};
}
