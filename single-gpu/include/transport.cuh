#pragma once
// Inter-GPU transport: one-sided gathers and publications on NCCL LLBuffer primitives over a symmetric
// window, NVSHMEM signal credits, and the all-reduce of W.
#include "kernels.cuh"
#include <thread>
#ifdef TQR_MULTI
#include <mpi.h>
#define NVSHMEMI_HOST_ONLY
#include <nvshmem_host.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/ll_buffer.h>
#include <nccl_device/impl/ll_buffer__funcs.h>
#include <unistd.h>
#endif
namespace tqr {
struct ColumnTeam {
 int size=0,rank[8]={};
 __host__ __device__ int index(int r)const{for(int z=0;z<size;++z)if(rank[z]==r)return z;return -1;}
 __host__ __device__ int begin(int z,int q)const{return int((long long)q*z/size);}
 __host__ __device__ int owner(int j,int q)const{return int(((long long)(j+1)*size-1)/q);}
};

struct Context {
 int rank=0,size=1,device=0;std::string hostname;bool mpi=false;json mapping;
 Context(){
#ifdef TQR_MULTI
  int argc=0;char**argv=nullptr;int provided;
  if(MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided)!=MPI_SUCCESS)throw std::runtime_error("MPI init");mpi=true;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
  char name[MPI_MAX_PROCESSOR_NAME]={0};int len;MPI_Get_processor_name(name,&len);hostname=name;
  std::vector<char> names(size*MPI_MAX_PROCESSOR_NAME);MPI_Allgather(name,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,names.data(),MPI_MAX_PROCESSOR_NAME,MPI_CHAR,MPI_COMM_WORLD);
  for(int r=0;r<size;++r)if(hostname!=std::string(names.data()+r*MPI_MAX_PROCESSOR_NAME))throw std::runtime_error("multi_node_not_authorized");
  MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);int localrank;MPI_Comm_rank(local,&localrank);MPI_Comm_free(&local);
  int count;CU(cudaGetDeviceCount(&count));device=count==1?0:localrank;if(device>=count)throw std::runtime_error("rank_device_mapping");
#else
  hostname="single-process";
#endif
  CU(cudaSetDevice(device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
  std::string uuid;const char* hex="0123456789abcdef";for(auto b:prop.uuid.bytes){uuid+=hex[(unsigned char)b>>4];uuid+=hex[(unsigned char)b&15];}
  mapping={{"rank",rank},{"pe",rank},{"nccl_rank",rank},{"visible_ordinal",device},{"uuid",uuid},{"host",hostname},{"gpu",prop.name}};
#ifdef TQR_MULTI
  char u[33]={};std::copy(uuid.begin(),uuid.end(),u);std::vector<char> us(size*33);MPI_Allgather(u,33,MPI_CHAR,us.data(),33,MPI_CHAR,MPI_COMM_WORLD);
  for(int r=0;r<size;++r)for(int t=r+1;t<size;++t)if(std::string(us.data()+r*33)==std::string(us.data()+t*33))throw std::runtime_error("duplicate_gpu_mapping");
#endif
 }
 void barrier()const{
#ifdef TQR_MULTI
 if(size>1)MPI_Barrier(MPI_COMM_WORLD);
#endif
 }
 int max_status(int value)const{
#ifdef TQR_MULTI
 int out;MPI_Allreduce(&value,&out,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);return out;
#else
 return value;
#endif
 }
 json all_records(const json& value)const{
#ifdef TQR_MULTI
  std::string s=value.dump();int len=s.size();std::vector<int> lengths(size),offsets(size);MPI_Allgather(&len,1,MPI_INT,lengths.data(),1,MPI_INT,MPI_COMM_WORLD);
  for(int i=1;i<size;++i)offsets[i]=offsets[i-1]+lengths[i-1];std::vector<char> data(offsets.back()+lengths.back());
  MPI_Allgatherv(s.data(),len,MPI_CHAR,data.data(),lengths.data(),offsets.data(),MPI_CHAR,MPI_COMM_WORLD);json out=json::array();for(int i=0;i<size;++i)out.push_back(json::parse(data.data()+offsets[i],data.data()+offsets[i]+lengths[i]));return out;
#else
  return json::array({value});
#endif
 }
 ~Context(){
#ifdef TQR_MULTI
 // Do not abort or finalize while an exception unwinds: main's handler prints its message and calls MPI_Abort.
 if(std::uncaught_exceptions())return;
 if(mpi)MPI_Finalize();
#endif
 }
};
#ifdef TQR_MULTI
inline void nc_check(ncclResult_t e,const char* s){if(e!=ncclSuccess)throw std::runtime_error(std::string(s)+": "+ncclGetErrorString(e));}
#define NC(x) ::tqr::nc_check((x),#x)
// DIAGNOSTIC ONLY, compiled out unless -DTQR_TRANSPORT_TRACE is given. Every episode prints its
// identity (rank, kind, root, count, sequence, epoch, grid) and the phase it has reached, so a hang
// can be attributed to an exact episode instead of to a stack frame.
#ifdef TQR_TRANSPORT_TRACE
#define TQR_TRACE(...) do{fprintf(stderr,"TRACE " __VA_ARGS__);fflush(stderr);}while(0)
#else
#define TQR_TRACE(...) do{}while(0)
#endif
// The default publication protocol is stream ordered.
#ifndef TQR_STREAM_ORDERED_PUBLISH
#define TQR_STREAM_ORDERED_PUBLISH 1
#endif
// Pinned-paper API adapter: explicit LL flags for every stage, unicast only. One slot per
// source/element; CTAs own disjoint element indices. Reduce's non-root CTAs only produce; copy's root
// only produces (then reads its own slots). No producer waits for a remote numerical consumer or
// another CTA. Device acknowledgments fence both reuse and epoch clearing, including wrap. `seq` is
// the per-rank collective kernel counter, identical across ranks, starting at 1) -- the window's
// capacity*ranks*4 bytes are exactly two ncclLL sub-buffers, so nothing new is allocated -- and the
// per-episode host drain + MPI_Barrier are replaced by device-side consumer PROGRESS: every rank,
// when its last CTA of episode `seq` retires, stores `seq` into slot [its rank] of every peer's
// progress array (NVSHMEM symmetric heap, written through nvshmem_ptr peer addresses); a sender at
// episode seq first waits, on the device, until each receiver it writes to has published progress >=
// seq-2, i.e. has finished reading every earlier use of this sub-buffer. Waits at episode e depend
// only on episodes <= e-2 of other ranks (deadlock-free by induction) and never on the host. progress
// == nullptr keeps the single-buffered, host-rendezvous protocol.
__device__ inline unsigned long long transport_ld_acquire(const unsigned long long* p){unsigned long long v;asm volatile("ld.acquire.sys.global.u64 %0,[%1];":"=l"(v):"l"(p):"memory");return v;}
__device__ inline void transport_st_release(unsigned long long* p,unsigned long long v){asm volatile("st.release.sys.global.u64 [%0],%1;"::"l"(p),"l"(v):"memory");}
// One CTA waits only for work already issued by every participating rank.
__global__ inline void transport_wait_progress(const unsigned long long* progress,int peers,unsigned long long target){
 if(threadIdx.x==0)for(int p=0;p<peers;++p)while(transport_ld_acquire(progress+p)<target){}
}
// Called after this rank's window memset. The second half of each progress
// array is a separate monotone clear-generation namespace, never an LL credit.
__global__ inline void transport_clear_ack(unsigned long long* const* peers,int ranks,int rank,unsigned long long generation){
 if(threadIdx.x==0){__threadfence_system();for(int p=0;p<ranks;++p)transport_st_release(peers[p]+ranks+rank,generation);}
}
template<class T,bool Reduce> __global__ void paper_ll(ncclDevComm comm,ncclWindow_t win,const T* input,T* output,int count,int root,uint8_t epoch,int pitch,int participants,
  unsigned long long seq,unsigned long long* const* peer_progress,const unsigned long long* my_progress,unsigned* done){
 if(comm.rank>=participants)return;
 const bool multi=peer_progress!=nullptr;
 if(multi){
  if(threadIdx.x==0&&seq>=2){
   if constexpr(Reduce){while(transport_ld_acquire(my_progress+root)+2<seq){}}
   else if(comm.rank==root){for(int p=0;p<participants;++p)while(transport_ld_acquire(my_progress+p)+2<seq){}}
  }
  __syncthreads();
 }
 ncclLLBuffer<ncclLL,false> ll(ncclSymPtr<char>(win,0),pitch,0,uint8_t(multi?2:1),ncclMultimemHandle{});ll.setEpochValue(epoch);
 if(multi)ll.setSubBuffer(uint32_t(seq&1));
 auto team=ncclTeamLsa(comm);
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
  if constexpr(Reduce){ll.send<T>(team,root,comm.rank*count+i,input[i]);
   if(comm.rank==root)output[i]=ll.template recvReduce<4,T,false>(i,participants,count,[] __device__(T x){return x;},[] __device__(T a,T b){return a+b;});
  }else{if(comm.rank==root)for(int p=0;p<participants;++p)ll.send<T>(team,p,i,input[i]);output[i]=ll.template recv<T,false>(i);}
 }
 if(multi){
  __syncthreads();
  if(threadIdx.x==0){
   __threadfence_system(); // each CTA completes its peer writes before retiring
   unsigned* d=done+(seq&3);
   if(atomicAdd(d,1u)==gridDim.x-1){                 // last CTA of this rank's episode
    *d=0;__threadfence_system();
    for(int p=0;p<participants;++p)transport_st_release(peer_progress[p]+comm.rank,seq);
   }
  }
 }
}
// M2: column-split W join and Z gather. Only node members move tensor data.
// Nonmembers retire a one-CTA control episode so the shared parity/epoch
// protocol remains monotone when successive tree nodes have different teams.
// They never touch input/output or an LL payload slot.
template<class T,bool Reduce> __global__ void paper_ll_columns(ncclDevComm comm,ncclWindow_t win,
 const T* input,T* output,int rows,int columns,bool transposed,ColumnTeam members,uint8_t epoch,int pitch,
 unsigned long long seq,unsigned long long* const* peer_progress,const unsigned long long* my_progress,unsigned* done){
 const int mine=members.index(comm.rank),count=rows*columns;
 const bool multi=peer_progress!=nullptr;
 if(mine>=0){
  if(multi){if(threadIdx.x==0)for(int z=0;z<members.size;++z)
   while(transport_ld_acquire(my_progress+members.rank[z])+2<seq){}
   __syncthreads();}
  ncclLLBuffer<ncclLL,false> ll(ncclSymPtr<char>(win,0),pitch,0,uint8_t(multi?2:1),ncclMultimemHandle{});
  ll.setEpochValue(epoch);if(multi)ll.setSubBuffer(uint32_t(seq&1));
  auto team=ncclTeamLsa(comm);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x){
   const int row=i%rows,j=i/rows,z=members.owner(j,columns);
   const int first=members.begin(z,columns),width=members.begin(z+1,columns)-first;
   if constexpr(Reduce){
    ll.send<T>(team,members.rank[z],comm.rank*count+i,input[i]);
    if(mine==z){
     // Never use an atomic floating-point sum.
     T sum=T(0);bool initial=true;
     for(int r=0;r<comm.nRanks;++r)if(members.index(r)>=0){
      T v=ll.template recv<T,false>(r*count+i);sum=initial?v:sum+v;initial=false;}
     output[row+(j-first)*rows]=sum;
    }
   }else{
    if(mine==z){const T value=input[transposed?j-first+row*width:row+(j-first)*rows];
     for(int peer=0;peer<members.size;++peer)ll.send<T>(team,members.rank[peer],i,value);}
    output[transposed?j+row*columns:i]=ll.template recv<T,false>(i);
   }
  }
 }
 if(multi){
  __syncthreads();
  if(threadIdx.x==0){__threadfence_system();unsigned* d=done+(seq&3);
   if(atomicAdd(d,1u)==gridDim.x-1){*d=0;__threadfence_system();
    for(int p=0;p<comm.nRanks;++p)transport_st_release(peer_progress[p]+comm.rank,seq);}
  }
 }
}
struct PublicationHeader {uint64_t generation,expression,bytes,kind,sender,receiver;};
__host__ __device__ inline bool publication_matches(const PublicationHeader&a,const PublicationHeader&b){return a.generation==b.generation&&a.expression==b.expression&&a.bytes==b.bytes&&a.kind==b.kind&&a.sender==b.sender&&a.receiver==b.receiver;}
// The received header is checked ON THE DEVICE, in stream order, instead of by
// a blocking D2H read. The check is the same comparison; only the observer
// moves, so a violated publication still fails the run -- at the next
// checkpoint (end of a factorization, an epoch wrap, or an explicit drain)
// rather than inside the episode that raised it.
__global__ inline void stage_publication_header(PublicationHeader* dst,PublicationHeader h){if(threadIdx.x==0)*dst=h;}
__global__ inline void verify_publication(const PublicationHeader*got,PublicationHeader want,int*fault){if(!publication_matches(*got,want))*fault=1;}
#endif
// Host-side phase accounting for every transport episode. Every field is a
// wall-clock interval around code that already blocks the host, so the timer
// itself is far below the thing it measures. This exists because the
// multi-GPU cost was attributed only by subtraction ("unexplained 86.5%");
// a schedule change must be argued from where the time is, not from a
// residual.
struct TransportAccount {
 double agree=0,stage=0,wait=0,kernel=0,drain=0,barrier=0;uint64_t episodes=0;
 json record()const{return {{"episodes",episodes},{"agree_s",agree},{"stage_s",stage},{"wait_s",wait},{"kernel_s",kernel},{"drain_s",drain},{"barrier_s",barrier},{"total_s",agree+stage+wait+kernel+drain+barrier}};}
};
class Transport {
 Context& ctx;size_t capacity=0,pubcap=0;Stream own;cudaStream_t stream=nullptr;uint64_t sequence=0;int participants;
 int delayed_rank=-1,delay_ms=0;
 TransportAccount acct_publish,acct_paper;TransportAccount* acct=nullptr;
 struct Tick {double t0;explicit Tick():t0(seconds()){} double lap(){double now=seconds(),d=now-t0;t0=now;return d;}};
 // Eager mode (protocol qualification) keeps the per-episode check. The per-episode Allreduce was
 // never the ordering mechanism: paper() keeps its own post-kernel barrier and publish() its
 // barriers (or stream credits), so removing it adds no hazard.
 uint64_t signature_hash=0x9E3779B97F4A7C15ULL,signature_episodes=0;bool eager_agreement=false;
 static uint64_t sig_mix(uint64_t h,uint64_t v){h^=v+0x9E3779B97F4A7C15ULL+(h<<6)+(h>>2);h^=h>>31;h*=0xBF58476D1CE4E5B9ULL;h^=h>>27;return h;}
 uint64_t agree(uint64_t op,int sender,int receiver,uint64_t count,uint64_t word,uint64_t expr,uint64_t kind){
  uint64_t gen=++sequence;
#ifdef TQR_MULTI
  if(ctx.size>1){
   uint64_t sig[9]={gen,op,uint64_t(sender),uint64_t(receiver),count,word,expr,kind,uint64_t(participants)};
   Tick tk;
   if(eager_agreement){
    // Both backend entry points use exactly the same guard, even for zero
    // counts and self copies. A mismatched operation cannot skip agreement.
    uint64_t lo[9],hi[9];
    MPI_Allreduce(sig,lo,9,MPI_UINT64_T,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(sig,hi,9,MPI_UINT64_T,MPI_MAX,MPI_COMM_WORLD);
    for(int i=0;i<9;++i)if(lo[i]!=hi[i])throw std::runtime_error(op==0?"publication_signature_before_issue":"collective_signature");
   }else{for(int i=0;i<9;++i)signature_hash=sig_mix(signature_hash,sig[i]);++signature_episodes;}
   if(acct)acct->agree+=tk.lap();
  }
#endif
  return gen;
 }
#ifdef TQR_MULTI
 ncclComm_t comm=nullptr;ncclDevComm dc{};ncclWindow_t window=nullptr;
 void* llmem=nullptr;void* ncinput=nullptr;void* ncoutput=nullptr;
 char* inbox=nullptr;char* sendbuf=nullptr;uint64_t* signals=nullptr;uint64_t* credits=nullptr;
 bool shmem=false;size_t llbytes=0;size_t free_before=0,total_memory=0;
 // Epoch reuse happens every 254 sequence numbers; the window is cleared then and on first use,
 // never per episode.
 uint64_t next_clear_sequence=0,llclears=0;
 int* fault=nullptr;size_t pubslot=0;
 unsigned long long* progress=nullptr;unsigned long long** peer_progress=nullptr;unsigned* progress_done=nullptr;bool progress_protocol=false,stream_ordered_publish=false;int ll_ctas=32,ll_elems=32;
 // Publications also advance `sequence`, so `sequence` must NOT drive the progress protocol: a
 // sender would wait for a progress value no collective ever publishes (found in review before the
 // first run).
 unsigned long long collective_kernels=0;
 // credit of this rank's last publication as a sender, awaited before sendbuf is re-staged.
 uint64_t pending_credit_gen=0;int pending_credit_receiver=0;uint64_t deferred_credit_waits=0;
#endif
public:
 json accounting()const{return {{"publish",acct_publish.record()},{"paper",acct_paper.record()},{"scope","host wall time inside Transport, by phase: agree = the two signature Allreduce; stage = source staging plus its drain; wait = destination wait/copy/credit; kernel = the LL kernel plus its drain; drain = trailing completion wait; barrier = MPI_Barrier. Sums over all ranks are NOT comparable -- read per rank."}};}
 json evidence={{"paper_kernels",json::array()},{"publications",0},{"paper_copies",0},{"paper_reductions",0},{"staging_bytes",0},{"source_completions",0},{"reader_acknowledgments",0}};
 // `pubbytes` is the largest PUBLISHED payload; 0 means "the same as the
 // collective capacity", which is what every caller that publishes operands
 // wants. The engine publishes only b x h blocks while its collectives carry
 // b x strip, so separating them keeps the per-sender inboxes small enough to
 // give each sender its own region -- which is what lets an episode stop
 // needing a global barrier to keep two senders off one buffer.
 Transport(Context& c,size_t maxbytes,int active=0,size_t pubbytes=0):ctx(c),capacity(std::max(maxbytes,size_t(64))),pubcap(std::max(pubbytes?pubbytes:maxbytes,size_t(64))),participants(active?active:c.size){
  stream=own.s;
#ifdef TQR_MULTI
  if(ctx.size>1){
   CU(cudaMemGetInfo(&free_before,&total_memory));
   nvshmemx_init_attr_t attr=NVSHMEMX_INIT_ATTR_INITIALIZER;MPI_Comm mc=MPI_COMM_WORLD;attr.mpi_comm=&mc;
   if(nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr)!=0)throw std::runtime_error("nvshmem host-library init");shmem=true;
   if(nvshmem_my_pe()!=ctx.rank||nvshmem_n_pes()!=ctx.size)throw std::runtime_error("PE mapping mismatch");
   // A ragged fp32 capacity need not be 8-byte aligned. Every sender's
   // header region must nevertheless retain PublicationHeader alignment.
   pubslot=checked_add(pubcap,sizeof(PublicationHeader));
   pubslot=checked_mul(checked_add(pubslot,alignof(PublicationHeader)-1)/alignof(PublicationHeader),alignof(PublicationHeader));
   inbox=(char*)nvshmem_malloc(pubslot*size_t(ctx.size));sendbuf=(char*)nvshmem_malloc(pubslot);
   CU(cudaMalloc(&fault,sizeof(int)));CU(cudaMemset(fault,0,sizeof(int)));
   signals=(uint64_t*)nvshmem_calloc(ctx.size,sizeof(uint64_t));credits=(uint64_t*)nvshmem_calloc(ctx.size,sizeof(uint64_t));
   progress=(unsigned long long*)nvshmem_calloc(2*ctx.size,sizeof(unsigned long long));
   if(!progress)throw std::runtime_error("nvshmem progress allocation");
   {std::vector<unsigned long long*> peers(ctx.size);bool direct=true;
    for(int p=0;p<ctx.size;++p){peers[p]=(unsigned long long*)nvshmem_ptr(progress,p);if(!peers[p])direct=false;}
    peer_progress=(unsigned long long**)nvshmem_malloc(sizeof(unsigned long long*)*ctx.size);
    progress_done=(unsigned*)nvshmem_calloc(4,sizeof(unsigned));
    if(!peer_progress||!progress_done)throw std::runtime_error("nvshmem progress metadata allocation");
    CU(cudaMemcpy(peer_progress,peers.data(),sizeof(unsigned long long*)*ctx.size,cudaMemcpyHostToDevice));
    const char* env=std::getenv("TQR_TRANSPORT_HOST_RENDEZVOUS");
    const bool host_rendezvous=env&&*env&&*env!='0';
    if(!direct&&!host_rendezvous)throw std::runtime_error("transport_peer_progress_unavailable");
    progress_protocol=direct&&!host_rendezvous;
    const char* pubenv=std::getenv("TQR_STREAM_ORDERED_PUBLISH");
    stream_ordered_publish=progress_protocol&&(pubenv?(std::string(pubenv)!="0"):TQR_STREAM_ORDERED_PUBLISH!=0);
    const char* ctasenv=std::getenv("TQR_LL_CTAS");
    if(ctasenv)ll_ctas=std::stoi(ctasenv);
    if(ll_ctas!=4&&ll_ctas!=8&&ll_ctas!=16&&ll_ctas!=32)throw std::runtime_error("transport_ll_ctas_must_be_4_8_16_32");
    const char* elemsenv=std::getenv("TQR_LL_ELEMS");
    if(elemsenv)ll_elems=std::stoi(elemsenv);
    if(ll_elems<1||ll_elems>(1<<20))throw std::runtime_error("transport_ll_elems_1_to_1048576");
    evidence["stream_ordered_publish"]=stream_ordered_publish;evidence["ll_cta_limit"]=ll_ctas;evidence["ll_elements_per_thread_target"]=ll_elems;
    evidence["collective_sync_protocol"]=progress_protocol?"device progress: two LL sub-buffers, sender waits on the device for receiver progress >= seq-2, no per-episode host drain or barrier":
      std::string("host rendezvous: one drain + MPI_Barrier per episode")+(direct?" (forced by TQR_TRANSPORT_HOST_RENDEZVOUS)":" (a peer is not load/store accessible)");}
   nvshmem_barrier_all();
   if(!inbox||!sendbuf||!signals||!credits)throw std::runtime_error("nvshmem arena allocation");
   ncclUniqueId id;if(ctx.rank==0)NC(ncclGetUniqueId(&id));MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);NC(ncclCommInitRank(&comm,ctx.size,id,ctx.rank));
   llbytes=checked_mul(capacity,size_t(ctx.size)*4);NC(ncclMemAlloc(&llmem,llbytes));NC(ncclMemAlloc(&ncinput,capacity));NC(ncclMemAlloc(&ncoutput,capacity));
   // Initial clearing is setup, before any publication/LL episode can run.
   CU(cudaMemsetAsync(llmem,0,llbytes,stream));CU(cudaStreamSynchronize(stream));ctx.barrier();next_clear_sequence=254;
   NC(ncclCommWindowRegister(comm,llmem,llbytes,&window,NCCL_WIN_COLL_SYMMETRIC));
   ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;req.lsaBarrierCount=0;req.lsaMultimem=false;
   NC(ncclDevCommCreate(comm,&req,&dc));auto team=ncclTeamLsa(comm);if(team.nRanks!=ctx.size)throw std::runtime_error("paper LSA full group unavailable");
   evidence["compiled_paper_kernels"]={"tqr::paper_ll<float,false>","tqr::paper_ll<float,true>","tqr::paper_ll<double,false>","tqr::paper_ll<double,true>"};
   evidence["backend"]="pinned LLBuffer application adapter; explicit LL flags, unicast, GPU sum; no legacy dispatch";
   evidence["owned_arena_bytes"]=llbytes+2*capacity+pubslot*size_t(ctx.size+1)+4*ctx.size*sizeof(uint64_t)+ctx.size*sizeof(unsigned long long*)+4*sizeof(unsigned)+sizeof(int);
   size_t free_after;CU(cudaMemGetInfo(&free_after,&total_memory));evidence["runtime_allocation_observed_bytes"]=free_before>free_after?free_before-free_after:0;evidence["symmetric_heap_reserved_bytes"]=64*1024*1024;
  }
#endif
 }
 // Bind the transport to the caller's stream. An episode is then issued into
 // the SAME ordered queue as the arithmetic that produces its operands and
 // consumes its results, which is what removes the host drains: the ordering
 // that used to be enforced by draining the device is now enforced by the
 // stream. Unbound, the transport keeps its own stream and behaves as before
 // for callers that own no stream (protocol qualification).
 void bind(cudaStream_t s){stream=s?s:own.s;}
 cudaStream_t bound()const{return stream;}
 // Complete every issued episode and surface any deferred protocol fault. The
 // device-side header check records into `fault`; this is where it is read.
 void drain(){CU(cudaStreamSynchronize(stream));checkpoint();agreement_checkpoint();}
 void set_eager_agreement(bool on){eager_agreement=on;}
 // Collective: every rank must call it at the same point (end of factor/apply_q, drain). Compares
 // the rolling signature hash and the episode count of every rank; one MIN/MAX Allreduce pair.
 uint64_t agreement_checks=0,agreed_episodes=0;
 void agreement_checkpoint(){
#ifdef TQR_MULTI
  if(ctx.size<=1)return;
  uint64_t v[2]={signature_hash,signature_episodes},lo[2],hi[2];
  MPI_Allreduce(v,lo,2,MPI_UINT64_T,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(v,hi,2,MPI_UINT64_T,MPI_MAX,MPI_COMM_WORLD);
  if(lo[0]!=hi[0]||lo[1]!=hi[1])throw std::runtime_error("collective_signature");
  ++agreement_checks;agreed_episodes=signature_episodes;
  evidence["agreement_policy"]=eager_agreement?"eager: two MPI_Allreduce per episode":"deferred (plan P3 I5): rolling signature hash per rank, compared by one MIN/MAX Allreduce pair at each factorization/application end and every drain";
  evidence["agreement_checks"]=agreement_checks;evidence["agreed_episodes"]=agreed_episodes;
#endif
 }
 void checkpoint(){
#ifdef TQR_MULTI
  if(!fault)return;int f=0;CU(cudaMemcpy(&f,fault,sizeof(int),cudaMemcpyDeviceToHost));
  if(f){CU(cudaMemset(fault,0,sizeof(int)));throw std::runtime_error("publication_header");}
#endif
 }
 void delay_next_paper_producer(int rank,int milliseconds){delayed_rank=rank;delay_ms=milliseconds;}
 void publish(int sender,int receiver,const void* source,void* destination,size_t bytes,uint64_t expr,uint64_t kind){
  acct=&acct_publish;++acct_publish.episodes;
  uint64_t gen=agree(0,sender,receiver,bytes,1,expr,kind);
  if(sender<0||sender>=ctx.size||receiver<0||receiver>=ctx.size)throw std::runtime_error("publication_rank");
  if(bytes>pubcap)throw std::runtime_error("publication_capacity");
  if(sender==receiver){if(ctx.rank==sender&&bytes)CU(cudaMemcpyAsync(destination,source,bytes,cudaMemcpyDeviceToDevice,stream));return;}
#ifdef TQR_MULTI
  PublicationHeader h{gen,expr,bytes,kind,uint64_t(sender),uint64_t(receiver)};
  Tick tk;
  // STREAM-ORDERED EPISODE. Every step below is issued on the caller's stream and consumed by a
  // later stream-ordered operation on the same stream, so the host never drains the device to make
  // one publication happen. What used to be four host drains and one global barrier per publication
  // is now: stage, put+signal, device-side wait, device-side header check, copy out, credit. The
  // buffer credit is explicit and is what makes the barrier unnecessary -- the sender's NEXT
  // episode cannot overwrite `sendbuf` before the receiver has released this one, because its
  // staging copy is issued after this credit wait on the same stream, and two DIFFERENT senders
  // never share a destination region.
  char*slot=inbox+size_t(sender)*pubslot;
  if(stream_ordered_publish){
   // The sender's credit wait is DEFERRED to the next reuse of its staging buffer. The credit of this
   // episode is only needed before `sendbuf` (and the receiver's per-sender slot) is written again,
   // i.e. at this sender's next publication; waiting for it here stalled the sender's stream on the
   // receiver's progress after every publication. The wait below is for the PREVIOUS episode this
   // rank sent (any receiver), still device-side and in stream order.
   if(ctx.rank==sender){
    if(pending_credit_gen){nvshmemx_signal_wait_until_on_stream(credits+pending_credit_receiver,NVSHMEM_CMP_EQ,pending_credit_gen,stream);++deferred_credit_waits;}
    stage_publication_header<<<1,1,0,stream>>>((PublicationHeader*)sendbuf,h);CU(cudaGetLastError());if(bytes)CU(cudaMemcpyAsync(sendbuf+sizeof(h),source,bytes,cudaMemcpyDeviceToDevice,stream));
    nvshmemx_putmem_signal_on_stream(slot,sendbuf,sizeof(h)+bytes,signals+sender,gen,NVSHMEM_SIGNAL_SET,receiver,stream);
    pending_credit_gen=gen;pending_credit_receiver=receiver;}
   acct_publish.stage+=tk.lap();
   if(ctx.rank==receiver){nvshmemx_signal_wait_until_on_stream(signals+sender,NVSHMEM_CMP_EQ,gen,stream);
    verify_publication<<<1,1,0,stream>>>((const PublicationHeader*)slot,h,fault);CU(cudaGetLastError());
    if(bytes)CU(cudaMemcpyAsync(destination,slot+sizeof(h),bytes,cudaMemcpyDeviceToDevice,stream));
    nvshmemx_signal_op_on_stream(credits+receiver,gen,NVSHMEM_SIGNAL_SET,sender,stream);}
   acct_publish.wait+=tk.lap();
   evidence["deferred_credit_waits"]=deferred_credit_waits;evidence["publication_credit_policy"]="deferred: a sender waits on the device for the credit of its previous publication immediately before re-staging sendbuf";
  }else{
   // SAFE PATH, SIGNAL-FREE BY CONSTRUCTION. No NVSHMEM signal is used and no device-side wait is
   // ever issued, so no stream can be left parked on a value that never arrives -- which is the
   // hazard that hung the arm above. Ordering comes from barriers that EVERY rank executes
   // unconditionally, so the ranks cannot take divergent paths into a cycle. Cost is three barriers
   // and two drains per publication; publications carry only b x h blocks.
   if(ctx.rank==sender){CU(cudaMemcpyAsync(sendbuf,&h,sizeof(h),cudaMemcpyHostToDevice,stream));if(bytes)CU(cudaMemcpyAsync(sendbuf+sizeof(h),source,bytes,cudaMemcpyDeviceToDevice,stream));
    CU(cudaStreamSynchronize(stream));}
   acct_publish.stage+=tk.lap();
   ctx.barrier();
   if(ctx.rank==sender){nvshmem_putmem(slot,sendbuf,sizeof(h)+bytes,receiver);nvshmem_quiet();}
   ctx.barrier();
   acct_publish.wait+=tk.lap();
   if(ctx.rank==receiver){verify_publication<<<1,1,0,stream>>>((const PublicationHeader*)slot,h,fault);CU(cudaGetLastError());
    if(bytes)CU(cudaMemcpyAsync(destination,slot+sizeof(h),bytes,cudaMemcpyDeviceToDevice,stream));
    CU(cudaStreamSynchronize(stream));checkpoint();}
   ctx.barrier();
   TQR_TRACE("publish_done r=%d s=%d d=%d bytes=%zu seq=%llu\n",ctx.rank,sender,receiver,bytes,(unsigned long long)gen);
   acct_publish.drain+=tk.lap();
  }evidence["publications"]=evidence["publications"].get<int>()+1;
  evidence["source_completions"]=evidence["source_completions"].get<int>()+1;evidence["reader_acknowledgments"]=evidence["reader_acknowledgments"].get<int>()+1;
  evidence["staging_bytes"]=evidence["staging_bytes"].get<size_t>()+2*bytes;
#else
  throw std::runtime_error("standalone_remote_request");
#endif
 }
 template<class T,bool Reduce> void paper(int root,const T* src,T* dst,int count,uint64_t expression){
  acct=&acct_paper;++acct_paper.episodes;
  agree(Reduce?2:1,root,root,uint64_t(count),sizeof(T),expression,0);
  if(count<0||root<0||root>=participants)throw std::runtime_error("paper_descriptor");
  size_t bytes=size_t(count)*sizeof(T);if(bytes>capacity)throw std::runtime_error("paper_capacity");if(!count)return;
  if(ctx.size==1){CU(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));return;}
#ifdef TQR_MULTI
  Tick tk;
  // On an epoch wrap, first wait for every prior LL reader, then clear our
  // window, acknowledge that clear to every rank, and wait for their clears.
  // Both sides are device ordered. In particular, a fast sender can never
  // write a new epoch into a peer window that will subsequently be erased.
  if(sequence>=next_clear_sequence){
   if(progress_protocol){
    transport_wait_progress<<<1,1,0,stream>>>(progress,participants,collective_kernels);CU(cudaGetLastError());
    CU(cudaMemsetAsync(llmem,0,llbytes,stream));
    transport_clear_ack<<<1,1,0,stream>>>(peer_progress,ctx.size,ctx.rank,sequence);CU(cudaGetLastError());
    transport_wait_progress<<<1,1,0,stream>>>(progress+ctx.size,ctx.size,sequence);CU(cudaGetLastError());
   }else{
    CU(cudaStreamSynchronize(stream));ctx.barrier();CU(cudaMemsetAsync(llmem,0,llbytes,stream));CU(cudaStreamSynchronize(stream));ctx.barrier();
   }
   next_clear_sequence=sequence+254;++llclears;
  }
   if(ctx.rank<participants&&(Reduce||ctx.rank==root))CU(cudaMemcpyAsync(ncinput,src,bytes,cudaMemcpyDeviceToDevice,stream));acct_paper.stage+=tk.lap();
  uint8_t epoch=uint8_t(2+(sequence%254));
  // Launch geometry from the DEVICE, not from a literal. The 32-CTA cap this
  // replaces predates the strip widths the selector now picks: at fp64
  // n=16384 one reduce carries 2.08e6 elements, so 32 CTAs x 128 threads gave
  // every thread 508 dependent LL slots and left 128 of 132 SMs empty. CTAs
  // own disjoint element indices (the slot index is elt, never blockIdx), so
  // the partition argument is unchanged by the count; only the occupancy is.
  // The bound is the RESIDENT grid for this exact instantiation, because the
  // kernel has no cross-CTA dependency and a resident grid is what the LL
  // poll loop needs to make progress without oversubscription.
  static int resident=0;
  if(!resident){int per_sm=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,paper_ll<T,Reduce>,128,0));
   cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,ctx.device));resident=std::max(1,per_sm*prop.multiProcessorCount);}
  // The LL grid bound is size-aware. Target ll_elems elements per thread above the cap, never beyond the
  // resident grid.
  const int sized=int(std::min<long long>(resident,std::max<long long>(ll_ctas,((long long)count+128LL*ll_elems-1)/(128LL*ll_elems))));
  int blocks=std::max(1,std::min(progress_protocol?sized:resident,ceildiv(count,128)));
  bool delayed=delayed_rank>=0;
  if(ctx.rank==delayed_rank)std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  double launch_time=delayed?seconds():0;
  TQR_TRACE("paper_issue r=%d reduce=%d root=%d count=%d seq=%llu epoch=%u blocks=%d pitch=%d parts=%d\n",ctx.rank,int(Reduce),root,count,(unsigned long long)sequence,unsigned(epoch),blocks,int(capacity*ctx.size),participants);
  paper_ll<T,Reduce><<<blocks,128,0,stream>>>(dc,window,(T*)ncinput,(T*)ncoutput,count,root,epoch,int(capacity*ctx.size),participants,
    ++collective_kernels,progress_protocol?peer_progress:nullptr,progress,progress_done);CU(cudaGetLastError());
  if(!progress_protocol){
  CU(cudaStreamSynchronize(stream));acct_paper.kernel+=tk.lap();TQR_TRACE("paper_kernel_done r=%d reduce=%d count=%d seq=%llu\n",ctx.rank,int(Reduce),count,(unsigned long long)sequence);ctx.barrier();acct_paper.barrier+=tk.lap();TQR_TRACE("paper_barrier_done r=%d reduce=%d count=%d seq=%llu\n",ctx.rank,int(Reduce),count,(unsigned long long)sequence);
  }else acct_paper.kernel+=tk.lap();
  double completion_time=delayed?seconds():0;
  // The receiver output is ready only after this kernel completion. No NVSHMEM
  // operation is used as a substitute for completion of the NCCL device API.
  if(ctx.rank<participants&&(!Reduce||ctx.rank==root))CU(cudaMemcpyAsync(dst,ncoutput,bytes,cudaMemcpyDeviceToDevice,stream));acct_paper.drain+=tk.lap();
  auto name=Reduce?"paper_reductions":"paper_copies";evidence[name]=evidence[name].get<int>()+1;
  std::string kernel="tqr::paper_ll<"+std::string(sizeof(T)==4?"float,":"double,")+(Reduce?"true>":"false>");
  if(std::find(evidence["paper_kernels"].begin(),evidence["paper_kernels"].end(),kernel)==evidence["paper_kernels"].end())evidence["paper_kernels"].push_back(kernel);
  evidence["last_expression"]=expression;evidence["last_protocol_sequence"]=sequence;evidence["epoch_policy"]=progress_protocol?"2..255; initial clear in setup; epoch reuse fenced on device by all-reader progress before memset and all-peer clear acknowledgment after memset":"host ablation: two-sided drain/barrier on epoch reuse";evidence["ll_window_clears"]=llclears;evidence["ll_window_bytes"]=llbytes;
  evidence["episode_sync_policy"]=progress_protocol?"two LL sub-buffers; device receiver progress >= seq-2; no host drain/barrier":"host ablation: drain/barrier after each LL episode";
  evidence["last_ctas"]=blocks;evidence["cta_policy"]=progress_protocol?"size-aware LL grid: max(ll_cta_limit, ceil(count/(128 ll_elements_per_thread_target))) CTAs, at most the resident grid":"host ablation: resident grid";evidence["resident_ctas"]=resident;
  evidence["maximum_ctas"]=std::max(blocks,evidence.value("maximum_ctas",0));
  if(delayed){evidence["delayed_producer_episodes"].push_back({{"rank",delayed_rank},{"delay_ms",delay_ms},{"root",root},{"reduce",Reduce},{"count",count},{"timing",ctx.all_records({{"rank",ctx.rank},{"host_launch",launch_time},{"host_completion",completion_time}})}});delayed_rank=-1;delay_ms=0;}
  evidence["staging_bytes"]=evidence["staging_bytes"].get<size_t>()+2*bytes;
#endif
 }
 // Direct LL loads/stores use the existing registered window. No ncinput /
 // ncoutput staging and no extra device allocation; arbitrary ordered subsets
 // and ragged column slices use the same two sub-buffers as paper().
 template<class T,bool Reduce> void columns(const int* ranks,int nranks,const T* src,T* dst,
     int rows,int columns,bool transposed,uint64_t expression){
  if(nranks<1||nranks>8||rows<1||columns<0)throw std::runtime_error("column_team_descriptor");
  ColumnTeam members;members.size=nranks;uint64_t team_key=0;
  for(int z=0;z<nranks;++z){
   if(ranks[z]<0||ranks[z]>=participants)throw std::runtime_error("column_team_rank");
   for(int k=0;k<z;++k)if(ranks[k]==ranks[z])throw std::runtime_error("column_team_duplicate");
   members.rank[z]=ranks[z];team_key=sig_mix(team_key,uint64_t(ranks[z]+1));
  }
  const size_t count=checked_mul(size_t(rows),size_t(columns)),bytes=checked_mul(count,sizeof(T));
  if(bytes>capacity||count>size_t(INT_MAX))throw std::runtime_error("column_team_capacity");
  acct=&acct_paper;++acct_paper.episodes;
  agree(Reduce?3:4,rows,columns,count,sizeof(T),expression,sig_mix(team_key,uint64_t(transposed)));
  if(!count)return;
  if(ctx.size==1){CU(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));return;}
#ifdef TQR_MULTI
  if(sequence>=next_clear_sequence){
   if(progress_protocol){
    transport_wait_progress<<<1,1,0,stream>>>(progress,participants,collective_kernels);CU(cudaGetLastError());
    CU(cudaMemsetAsync(llmem,0,llbytes,stream));
    transport_clear_ack<<<1,1,0,stream>>>(peer_progress,ctx.size,ctx.rank,sequence);CU(cudaGetLastError());
    transport_wait_progress<<<1,1,0,stream>>>(progress+ctx.size,ctx.size,sequence);CU(cudaGetLastError());
   }else{CU(cudaStreamSynchronize(stream));ctx.barrier();CU(cudaMemsetAsync(llmem,0,llbytes,stream));CU(cudaStreamSynchronize(stream));ctx.barrier();}
   next_clear_sequence=sequence+254;++llclears;
  }
  int resident=0,per_sm=0;cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,ctx.device));
  CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,paper_ll_columns<T,Reduce>,128,0));resident=std::max(1,per_sm*prop.multiProcessorCount);
  const int sized=int(std::min<size_t>(resident,std::max<size_t>(ll_ctas,(count+128ULL*ll_elems-1)/(128ULL*ll_elems))));
  const bool member=members.index(ctx.rank)>=0;
  const int blocks=member?std::max(1,std::min(sized,ceildiv(int(count),128))):1;
  if(ctx.rank==delayed_rank)std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  paper_ll_columns<T,Reduce><<<blocks,128,0,stream>>>(dc,window,src,dst,rows,columns,transposed,members,
   uint8_t(2+(sequence%254)),int(capacity*ctx.size),++collective_kernels,
   progress_protocol?peer_progress:nullptr,progress,progress_done);CU(cudaGetLastError());
  if(!progress_protocol){CU(cudaStreamSynchronize(stream));ctx.barrier();}
  delayed_rank=-1;delay_ms=0;
  const char* key=Reduce?"member_reduce_scatters":"member_all_gathers";
  evidence[key]=evidence.value(key,uint64_t(0))+1;
  evidence["member_payload_episodes"]=evidence.value("member_payload_episodes",uint64_t(0))+(member?1:0);
  evidence["nonmember_control_episodes"]=evidence.value("nonmember_control_episodes",uint64_t(0))+(member?0:1);
  const int mine=members.index(ctx.rank);
  const size_t sent=member?(Reduce?bytes:size_t(rows)*(members.begin(mine+1,columns)-members.begin(mine,columns))*sizeof(T)*nranks):0;
  evidence["member_payload_bytes_sent"]=evidence.value("member_payload_bytes_sent",size_t(0))+sent;
  evidence["member_staging_bytes"]=0;
  evidence["member_protocol"]="ordered LL reduce-scatter / all-gather; tensor payloads only among listed members; control-only progress on nonmembers";
  evidence["ll_window_clears"]=llclears;
  std::string kernel="tqr::paper_ll_columns<"+std::string(sizeof(T)==4?"float,":"double,")+(Reduce?"true>":"false>");
  if(std::find(evidence["paper_kernels"].begin(),evidence["paper_kernels"].end(),kernel)==evidence["paper_kernels"].end())evidence["paper_kernels"].push_back(kernel);
#endif
 }
#ifdef TQR_MULTI
 ncclComm_t reference_comm()const{return comm;}
#endif
 ~Transport(){
#ifdef TQR_MULTI
  // Same reason as Context's destructor: aborting here pre-empts main's handler
  // and loses the message. Skip cleanup and let the unwind reach it.
  if(std::uncaught_exceptions())return;
  if(comm){cudaDeviceSynchronize();ncclDevCommDestroy(comm,&dc);ncclCommWindowDeregister(comm,window);ncclMemFree(llmem);ncclMemFree(ncinput);ncclMemFree(ncoutput);ncclCommDestroy(comm);}
  if(fault)cudaFree(fault);
  if(shmem){if(peer_progress)nvshmem_free(peer_progress);if(progress_done)nvshmem_free(progress_done);if(progress)nvshmem_free(progress);nvshmem_free(credits);nvshmem_free(signals);nvshmem_free(sendbuf);nvshmem_free(inbox);nvshmemx_hostlib_finalize();}
#endif
 }
};
}

