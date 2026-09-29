#pragma once
// The QR-Omega engine: executes the elimination list with carried updates, look-ahead and aggregation
// (Algorithms 1 and 2 of the paper).
#include "tiled_plan.hpp"
#include "transport.cuh"
#include "tiled_scaling.cuh"
#include "small_packet.cuh"
#include "cooperative_packet.cuh"
#include "cooperative_minipanel.cuh"
#include "cooperative_resident.cuh"
#include "tt_panel.cuh"
#include "gau_ge.cuh"
#include "gau_fast.cuh"
#include "multi_wide_t.cuh"
#include "merge_apply.cuh"
#include "carrier_witness.hpp"
#include "execution_evidence.hpp"
#ifndef TQR_COOP_MIN_ROWS
#define TQR_COOP_MIN_ROWS 1024
#endif
// Fault injection for tests; normal builds leave TQR_C1_FAULT_AFTER_FIRST_WINDOW undefined.
#ifdef TQR_C1_FAULT_AFTER_FIRST_WINDOW
inline int tqr_c1_fault_after_first_window = 1;
#else
inline int tqr_c1_fault_after_first_window = 0;
#endif
// Admitted minipanel widths. Production builds take the full menu.
#ifndef TQR_COOP_WIDTHS
#define TQR_COOP_WIDTHS 0,8,16,32
#endif
#ifndef TQR_COOP_MAX_B
#define TQR_COOP_MAX_B 128
#endif
namespace tqr {
// Shared by the engine and the selector so the DP prices the kernel that will actually run. The
// carrier is dispatched only when one packet spans the whole panel (count==1, no tree left) and the
// panel is tall enough to pay for the grid barriers. The threshold was 4096, from a probe that
// compared the carrier against a block GE with a MATCHING leaf.
inline constexpr int coop_min_rows=TQR_COOP_MIN_ROWS;
// Width cap on the carrier. This was 32 while b=64 FP64 appeared to be mispriced by +79%. That was
// not a model error: coop_partial was allocated for a literal 32 groups, and the size check in
// cooperative_ge_batch was a silent `continue`, so every group count the buffer could not hold was
// skipped without a word and the tallest panels fell onto the block path.
inline constexpr int coop_max_b=TQR_COOP_MAX_B;
// The carrier was probed at 256 and 512 threads only; 1024 is usually non-resident for it and 128
// starves the row sweeps.
inline constexpr int coop_threads_for(int threads){return threads==256?256:512;}
// Group count the engine will request. Shared by the selector so it can apply
// the launcher's own shared-capacity feasibility test.
inline constexpr int coop_max_groups=coop_partition_limit;
// Candidate panel-group counts; the panel's contraction replication is chosen among them by measurement.
inline std::vector<int> coop_group_menu(){return {32,64,128};}
// Candidate replications of the trailing apply. Only insufficient capacity selects c=1 (or d=1); unreplicated
// products run the native kernels (tile_native_W/Z/D), and batched BLAS is used only where c>1.
inline std::vector<int> apply_rep_menu(){return {2,4,8};}
// Z and D each get their OWN contraction replication (Alg 2: "D <- V_e Z with
// its own carrier"). Their [K] is the reflector index h, not the row index, so
// the candidates are bounded by h=b<=128 rather than by the panel height. The selector chooses among
// them by measurement; c=1 is the capacity fall-back they are compared against (selection.zd_carrier).
inline std::vector<int> z_rep_menu(){return {2,4};}
inline std::vector<int> d_rep_menu(){return {2,4};}
inline std::string apply_z_rep_op(int c){return "apply_Z_c"+std::to_string(c);}
inline std::string apply_d_rep_op(int c){return "apply_D_c"+std::to_string(c);}
// THE BLOCK-LEVEL APPLY CARRIER. The peers are the blockDim.y warp-groups of one CTA, local memory
// is registers, global memory is shared: each group accumulates its disjoint K_z partial in
// registers, shared memory sums them, the CTA commits once. No HBM partials, so CB costs no
// admitted capacity and carrier_frame_words is unchanged. One menu serves all three products (each
// cuts its own [K]: rows for W, reflectors for Z and D); the selector weighs each product's CB
// separately. Starts at the menu front, never at 1, like every other carrier menu.
inline std::vector<int> block_rep_menu(){return {2,4,8};}
inline std::string apply_w_block_op(int c){return "apply_W_b"+std::to_string(c);}
inline std::string apply_z_block_op(int c){return "apply_Z_b"+std::to_string(c);}
inline std::string apply_d_block_op(int c){return "apply_D_b"+std::to_string(c);}
// The pipeline pace sample names the WHOLE carrier of the arm it measures:
// depth, W replication, Z replication and D replication. Keying it on (d,c)
// alone let one arm be priced on another arm's measurement, which is how the
// depth came to be credited with nothing at every shape.
inline std::string apply_pace_op(int d,int c,int zc,int dc){
 return "apply_pipe_d"+std::to_string(d)+"_c"+std::to_string(c)+"_z"+std::to_string(zc)+"_x"+std::to_string(dc);}
// The depth menu holds strictly >1 candidates under the same rule. At r=1 the
// depth is inert (Eq pipe gives C(d)=t_Sigma for every d) but d slots of W and
// Z credits are still ALLOCATED and charged; that is the capacity term the
// selector weighs among d>1, and d=1 arises only from the capacity fall-back
// or the --d forcing flag.
inline std::vector<int> pipeline_depth_menu(){return {2,4};}
// THE [J] WIDTH OF EQ APPLY IS CHOSEN AS A PACKET COUNT, NOT AS AN ABSOLUTE WIDTH. Deriving the
// strip from r fixes both: every candidate splits the elimination into at least two packets by
// construction, the menu stays small (four points instead of eight, so planning does not grow), and
// the selected r is what the carrier_dispatch witness reports back. Candidates are strictly above
// one, per the standing rule that only capacity or structure may give 1.
inline std::vector<int> packets_menu(){return {2,4,8,16};}
// The width that splits the widest elimination into r packets. Rounded up to a
// multiple of 64 for the apply GEMM's n dimension, but only when the rounding
// still yields exactly r packets -- an alignment that changed r would defeat
// the point of choosing r.
inline int strip_for_packets(int m,int n,int b,int r){
 const int cols=std::min(m,n);if(cols<=0||r<1)return 0;
 const int h=std::min(b,cols),trailing=n-h;if(trailing<=0)return 0;
 int s=ceildiv(trailing,r);
 // The width is rounded UP to the coarsest granularity that still yields exactly r packets. Coarse
 // first, then finer, so the largest aligned width that preserves r wins.
 for(int g:{512,256,128,64}){const int rs=((s+g-1)/g)*g;
  if(rs<trailing&&ceildiv(trailing,rs)==r){s=rs;break;}}
 // The descriptor validator refuses strip>tiled_max_strip, so a derived width
 // above it would throw at instantiation instead of simply not being chosen.
 // Clamping raises r rather than failing, which is the same thing the capacity
 // rule does everywhere else.
 return std::min(std::min(s,trailing),tiled_max_strip);
}
// The derived strip menu at this (m,n,b): one width per admitted packet count, deduplicated, with
// any width that would collapse the elimination to a single packet dropped (it is the structure
// fall-back's business, not a candidate). This is the same discipline the pace term already applies
// ("a shape with no pace sample is credited with nothing"), now applied to the packet width.
inline std::vector<int> strip_candidates(int m,int n,int b,int floor_q){
 std::vector<int> v;
 for(int r:packets_menu()){int s=strip_for_packets(m,n,b,r);
  if(s<=0||s<floor_q)continue;
  const int h=std::min(b,std::min(m,n)),trailing=n-h;
  if(trailing>0&&ceildiv(trailing,s)<2)continue;
  if(std::find(v.begin(),v.end(),s)==v.end())v.push_back(s);}
 // THE SINGLE-PACKET WIDTH IS AN ORDINARY CANDIDATE, not only a last resort. No fixed r is right;
 // the apply, Z and D carriers keep their own menus either way, because the width of a packet has
 // nothing to do with how a product cuts [K].
 {const int full=strip_for_packets(m,n,b,1);
  if(full>0&&std::find(v.begin(),v.end(),full)==v.end())v.push_back(full);}
 return v;
}
// THE PANEL-TALL LEAF IS ALWAYS A CANDIDATE. The block-level cooperative carrier is dispatched only
// when one packet spans the whole panel (count==1), and count = ceil(rows/leaf), so a leaf menu
// that stops at 16384 makes the carrier structurally impossible at every n above 16384. Every leaf
// at or above the panel height is the same schedule, so exactly one is added: the panel height
// itself.
inline std::vector<int> leaf_candidates(int m,int ranks){
 std::vector<int> v{128,256,512,1024,4096,16384};
 // The TALLEST rank slab, taken from row_begin exactly as tiled_instantiate
 // splits the rows, not ceil(m/ranks): an uneven split would leave the tallest
 // rank at count==2 and its panels back on the tree.
 int rows=0;const int p=std::max(1,ranks);
 for(int r=0;r<p;++r)rows=std::max(rows,row_begin(m,p,r+1)-row_begin(m,p,r));
 if(rows>0&&rows<=tiled_max_leaf&&std::find(v.begin(),v.end(),rows)==v.end())v.push_back(rows);
 return v;
}
inline int packets_at_widest_elimination(int m,int n,int b,int strip){
 const int cols=std::min(m,n);if(cols<=0||strip<=0)return 0;
 const int h=std::min(b,cols),trailing=n-h;return trailing>0?ceildiv(trailing,strip):0;}
// The ACROSS-GPU carrier's contraction replication: the arity of the global TT tree. A node of k
// ranks reduces W over k disjoint row sets, so k IS the c of that product and is selected from
// measurement, never fixed.
inline std::vector<int> global_radix_menu(int ranks){std::vector<int> v;for(int k=2;k<=std::min(ranks,TILE_MAX_RADIX);++k)v.push_back(k);if(v.empty())v.push_back(2);return v;}
// Packets pushed through the slot machinery when measuring the pipeline's
// pace. The model's pace formula divides by (R-1), so the two must agree;
// it lives here so they cannot drift apart.
inline constexpr int tiled_pipe_pace_packets=8;
// Sample name for the GPU-level replicated W. The GPU-level unreplicated stage
// runs the block carrier and is sampled per CB as apply_W_b<CB> (likewise Z/D).
inline std::string apply_rep_op(int c){return "apply_W_c"+std::to_string(c);}
// Minipanel width menu. 0 means the full-panel carrier (no within-elimination
// blocking); the positive widths block Eq apply at the minipanel boundary.
inline std::vector<int> coop_width_menu(){return {TQR_COOP_WIDTHS};}
// The panel carrier's STAGED COLUMN WINDOW. The peer stages local_height*(sw+1)+pw*sw words, so the
// window is the only handle that shrinks the footprint without more groups -- and more groups is
// capped by cooperative residency at 128 on this machine. FP32, whose word is half as wide, keeps
// the carrier at 0.99 carried. Sampling it is what lets the selector CHOOSE it.
inline std::vector<int> coop_window_menu(){return {16,32,64};}
// Unwindowed keeps its old name, so existing samples keep their meaning.
inline std::string coop_op(int groups,int pw,int sw=0){
 std::string k="coop_GE:g"+std::to_string(groups)+":p"+std::to_string(pw);
 if(sw>0)k+=":w"+std::to_string(sw);
 return k;}
// One expression for the carrier workspace, shared by the admission check and
// the allocation (a second copy of a size rule is what 8e and the coop_partial
// literal both were).
inline size_t coop_partial_slots_for(const TiledPlan&plan){
 size_t need=size_t(2)*std::max<size_t>(plan.max_batch,1)*size_t(coop_max_groups)*(2+2*size_t(plan.b));
 for(int g:coop_group_menu())for(int pw:coop_width_menu())if(pw&&pw<=plan.b)
  need=std::max(need,coop_mini_total_slots(g,plan.b,pw));
 return need;
}
// Same carrier: the batch's groups (c), minipanel width and threads; the record keeps the planned batch detail and
// the evidence counts resident panels. A batch whose resident launch is not admitted (capacity/occupancy) runs its
// planned windows.
inline bool coop_resident_env(){static const bool v=[]{const char*e=std::getenv("TQR_RESIDENT");return e&&std::atoi(e)==1;}();return v;}
inline int coop_groups_for(int rows){int g=rows/64;g=std::min(coop_max_groups,std::max(32,g));
 int p2=32;while(p2*2<=g)p2*=2;return p2;}
}
#ifndef TQR_MERGE_APPLY_MAX_B
#define TQR_MERGE_APPLY_MAX_B 32
#endif
#include <cublas_v2.h>
// The Eq-apply primitives and the three carrier-aware stages live in
// apply_dispatch.cuh: ONE implementation, shared by this engine's local apply,
// by its across-GPU apply, and by the probe that prices them.
#include "apply_dispatch.cuh"
#include "carrier_gmma.cuh"
namespace tqr {
struct TiledLocalShape{int nr,begin;};
template<class T> struct TiledMatrix {
 int m,n,nr,begin,ld;Buffer<T>a;
 TiledMatrix(int m_,int n_,int p,int rank,int pad=0):m(descriptor_dimension(m_)),n(descriptor_dimension(n_)),nr(row_begin(m,p,rank+1)-row_begin(m,p,rank)),begin(row_begin(m,p,rank)),ld(descriptor_ld(nr,pad)),a(checked_mul(size_t(ld),size_t(n))){}
 // An explicit local shape (the engine's internal block-cyclic copy: nr = RowMap::local_rows, begin = 0).
 TiledMatrix(int m_,int n_,TiledLocalShape s,int pad=0):m(descriptor_dimension(m_)),n(descriptor_dimension(n_)),nr(s.nr),begin(s.begin),ld(descriptor_ld(nr,pad)),a(checked_mul(size_t(ld),size_t(n))){}
};
// One round moves columns [c0, c0+q): every source row goes to exactly one destination row. The sender packs its
// rows destination-major (region d holds cnt[d] rows x q columns, column-major, ld cnt[d], at element offset
// base[d] q); the receiver unpacks by its own tables.
template<class T> __global__ void rd_pack(const T* __restrict__ A,int ld,int nr,int c0,int q,const int* __restrict__ dest,const int* __restrict__ pos,
                                          const int* __restrict__ cnt,const int* __restrict__ base,T* __restrict__ stage){
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(nr)*q;i+=size_t(gridDim.x)*blockDim.x){
  const int r=int(i%nr),j=int(i/nr),d=dest[r];stage[size_t(base[d])*q+pos[r]+size_t(j)*cnt[d]]=A[r+size_t(c0+j)*ld];}}
template<class T> __global__ void rd_unpack(T* __restrict__ A,int ld,int nr,int c0,int q,const int* __restrict__ src,const int* __restrict__ spos,
                                            const int* __restrict__ rcnt,const int* __restrict__ rbase,const T* __restrict__ stage){
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(nr)*q;i+=size_t(gridDim.x)*blockDim.x){
  const int r=int(i%nr),j=int(i/nr),s=src[r];A[r+size_t(c0+j)*ld]=stage[size_t(rbase[s])*q+spos[r]+size_t(j)*rcnt[s]];}}
template<class T> class TiledEngine {
 Context&ctx;TiledPlan plan;Stream stream;cublasHandle_t blas=nullptr;
 Buffer<T>tri,V,stack,X,W,Z,remoteR,remoteV,remoteW,remoteZ;Buffer<int>status;Buffer<Witness>witness;Buffer<uint64_t>hist;
 Buffer<T>d_partials;
 size_t d_lane_words=0,d_max_rows=0;
 json d_products=json::array();
 Buffer<TilePacket>all_packets;Buffer<TileMerge>all_merges;
 const TilePacket*packets=nullptr;const TileMerge*merges=nullptr;
  // Slot 0 is the main stream, so d==1 allocates
 std::vector<std::unique_ptr<Stream>>pipe_stream;std::vector<cublasHandle_t>pipe_blas;std::vector<Buffer<unsigned char>>pipe_workspace;
 // concurrent contraction peers of W (peer_stage_W): per lane slot, side streams at the slot's priority, one handle
 // each (no workspace: the non-split DMMA kernels need none, so the owned-bytes inventory is unchanged), fork/join
 // events.
 static constexpr int kPeerLanes=7;
 struct PeerSet{std::vector<cudaStream_t>s;std::vector<cublasHandle_t>h;cudaEvent_t fork=nullptr;std::vector<cudaEvent_t>join;};
 std::vector<PeerSet>peer_sets;
 PeerLanes peer_lanes(int slot){
  if(peer_sets.size()<=size_t(slot))peer_sets.resize(slot+1);
  PeerSet&ps=peer_sets[slot];
  if(ps.s.empty()){int prio=0;CU(cudaStreamGetPriority(slot_stream(slot),&prio));
   ps.s.resize(kPeerLanes);ps.h.resize(kPeerLanes);ps.join.resize(kPeerLanes);CU(cudaEventCreateWithFlags(&ps.fork,cudaEventDisableTiming));
   for(int z=0;z<kPeerLanes;++z){CU(cudaStreamCreateWithPriority(&ps.s[z],cudaStreamNonBlocking,prio));
    tiled_blas_check(cublasCreate(&ps.h[z]));tiled_blas_check(cublasSetStream(ps.h[z],ps.s[z]));
    tiled_blas_check(cublasSetMathMode(ps.h[z],tiled_blas_math()));tiled_blas_check(cublasSetAtomicsMode(ps.h[z],CUBLAS_ATOMICS_NOT_ALLOWED));
    tiled_blas_check(cublasSetWorkspace(ps.h[z],nullptr,0));CU(cudaEventCreateWithFlags(&ps.join[z],cudaEventDisableTiming));
    // TQR_PEER_SMS = cuBLAS SM-count target per W peer (0 = whole device). The tall-cell nsys showed every group's
    // first cooperative panel waiting ~360 us for 7-8 concurrent one-wave peer GEMMs (~100 long-lived CTAs each) to
    // drain; a per-peer target leaves SMs for the panel (the carrier, K_z and combine are unchanged).
    static const int peer_sms=[]{const char*e=std::getenv("TQR_PEER_SMS");return e?std::atoi(e):0;}();
    if(peer_sms>0)tiled_blas_check(cublasSetSmCountTarget(ps.h[z],peer_sms));}}
  return PeerLanes{ps.s.data(),ps.h.data(),ps.fork,ps.join.data(),kPeerLanes};}
 std::vector<std::unique_ptr<Event>>pipe_event;Event pipe_fork;
 // Handoff for Alg 1's cross-panel Pipeline: recorded when panel k's bulk
 // trailing window retires on lane 1, awaited by the main stream just before
 // panel k+1 touches those columns. Separate from pipe_event so the two lane
 // groups never share an event object.
 Event trail_done;
 Event global_tt_event;bool global_tt_recorded=false;   // global TT factor done (main) -> far global apply (lane B)
 // The windowed panel's T compose (compose_G + T12) runs on its own greatest-priority stream: it only writes T's
 // off-diagonal blocks and its private scratch (win_g, GPART), which no later window reads; joined once per panel.
 Stream compose_stream{true};Event compose_fork,compose_join;bool compose_pending=false;
 Event winb_fork,wina_done;   // window B(G) forks after compose(G) + retire(B(G-1)); its first commit follows window A(G)
 Buffer<unsigned char>blas_workspace;
 // Device pointer arrays for the carried products, one region per pipeline
 // slot. A slot's stream is ordered, so a region is safe to refill for the
 // next product on that slot.
 Buffer<T*>carrier_ptr;
 // What the carrier and the pipeline actually did, counted on the host as it is issued. Reported in
 // evidence() so that c>1 and d>1 can be checked by measurement rather than by reading the source.
 // The previous eight scalars here were declared and never incremented, which is precisely the
 // "recorded yes, executed no" gap; CarrierWitness replaces them with per-product-kind histograms
 // and attributed refusals.
  mutable CarrierWitness cw;
  // C0 PHASE SCOPING. evidence() freezes the factorization receipt: the
  // frozen copy is immutable, and the live counters restart (only after
  // their producing streams have completed) so validation accumulates its
  // own receipt instead of extending the factorization's. factor() opens a
  // new scope. Async issue order inside the factorization is untouched: the
  // only synchronization is the snapshot boundary, which already synced.
  json frozen_factor_receipt_; bool factor_frozen_=false; uint64_t frozen_invocation_=0;
 // PHYSICAL EXECUTION EVIDENCE (execution_format 24). The host counters above
 // say what was ISSUED; these say what the DEVICE ran. One interval per issued
 // packet and per panel factor, opened and closed by one-thread kernels on the
 // packet's own stream (include/execution_evidence.hpp). The log is a fixed
 // ring sized from the descriptor: an overflow is COUNTED and reported as a
 // dropped interval, never estimated away.
 Buffer<ExecInterval>exec_log;unsigned exec_cap=0,exec_next=0;
 unsigned exec_elimination=0;
 // Snapshot of the per-kind issued totals, so an interval can carry the
 // products of ITS packet rather than a running total.
 std::array<uint64_t,CarrierWitness::KINDS> exec_mark{};
 void exec_snapshot(){for(int k=0;k<CarrierWitness::KINDS;++k){uint64_t t=0;for(auto&e:cw.primary_products(k))t+=e.second;exec_mark[k]=t;}}

 // CERTIFY A REFUSAL (execution_format 24). Every product that runs at c=1 and every elimination
 // that issues one packet or uses one slot has to say, with a number, why. This is the single place
 // that decides which of the two applies to a given reason, so a new refusal cannot be added
 // without one.
 using Cert=CarrierWitness::Certificate;
 // A shape whose contraction index cannot be cut into `c` nonempty disjoint
 // parts. The gate re-checks the inequality; it does not take our word.
 void certify_structural(int kind,int reason,int rows,int h,int q,int count,
                         int contraction,int smallest_c,const char*statement,uint64_t n=1){
  if(!(contraction<smallest_c))return;
  Cert c;c.kind=kind;c.reason=reason;c.basis="structural";c.rows=rows;c.h=h;c.q=q;c.count=count;
  c.contraction=contraction;c.smallest_admitted_c=smallest_c;c.statement=statement;cw.certify(c,n);
 }
 // Both times travel with it.
 void certify_priced(int kind,int reason,int rows,int h,int q,int count,
                     double refused_s,double taken_s,const char*statement,uint64_t n=1){
  // A certificate is only issued when it can be SUBSTANTIATED.
  if(!(std::isfinite(refused_s)&&std::isfinite(taken_s)&&refused_s>0&&taken_s>0))return;
  Cert c;c.kind=kind;c.reason=reason;c.basis="priced";c.rows=rows;c.h=h;c.q=q;c.count=count;
  c.refused_s=refused_s;c.taken_s=taken_s;c.statement=statement;cw.certify(c,n);
 }
 void certify_capacity(int kind,int reason,int rows,int h,int q,int count,int smallest_c,
                       uint64_t required,uint64_t available,const char*memory,const char*statement,uint64_t n=1){
  // An unbounded budget (no admission limit was given) or a missing byte count is not a capacity
  // excuse, it is an absent one.
  if(!available||available==SIZE_MAX||!(required>available))return;
  Cert c;c.kind=kind;c.reason=reason;c.basis="capacity";c.rows=rows;c.h=h;c.q=q;c.count=count;
  c.smallest_admitted_c=smallest_c;c.bytes_required=required;c.bytes_available=available;
  c.memory=memory;c.statement=statement;cw.certify(c,n);
 }

 // Recorded truthfully (integrity passes) and always an open gap. Replaces priced certificates that
 // compared a kernel which cannot run this product (TT stacks) and the uncertified compose_G.
 void certify_unimplemented(int kind,int reason,int rows,int h,int q,int count,const char*missing,const char*statement,uint64_t n=1){
  Cert c;c.kind=kind;c.reason=reason;c.basis="unimplemented";c.rows=rows;c.h=h;c.q=q;c.count=count;
  c.missing=missing;c.statement=statement;cw.certify(c,n);
 }

 // The panel product's refusals.
 void certify_panel(int reason,const TileBatch&batch,int count){
  const int rows=batch.ge.empty()?0:batch.ge[0].rows;
  switch(reason){
   case CarrierWitness::R_PANEL_MULTI:
    certify_structural(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,1,2,
      "the panel is split into count>1 leaves, so no single packet spans it: the block-level carrier replaces the leaf GE AND the local tree, and with a tree still present there is nothing for it to replace (measured 0.69x at rows=512, the typical leaf)",count);break;
   case CarrierWitness::R_PANEL_NOT_FULL:
    certify_structural(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,1,2,
      "a ragged short leaf: the last leaf is shorter than the panel, so one packet cannot span the panel height",count);break;
   case CarrierWitness::R_PANEL_H_NE_B:
    certify_structural(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,batch.h,2,
      "the ragged tail panel is narrower than b, a shape the carrier's V/T layout is not defined for",count);break;
   case CarrierWitness::R_PANEL_B_MAX:
    certify_capacity(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,2,
      (uint64_t)batch.cshared_need,(uint64_t)batch.cshared_cap,"block shared memory S_mu",
      "b exceeds the widest panel the carrier's shared footprint admits",count);break;
   case CarrierWitness::R_PANEL_CC_ZERO: case CarrierWitness::R_PANEL_LAUNCH:
    if(batch.ccoop_s<=0&&batch.cshared_need>batch.cshared_cap)
     certify_capacity(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,2,
       (uint64_t)batch.cshared_need,(uint64_t)batch.cshared_cap,"block shared memory S_mu",
       "no admitted carrier peer fits the measured block capacity at this panel shape, which is Sec 3's one excuse for c=1",count);
    else
     certify_priced(CarrierWitness::PANEL_GE,reason,rows,batch.h,1,count,batch.ccoop_s,batch.cblock_s,
       "the weighed block-level carrier was priced against the block GE the engine would run on the SAME packet and lost at this panel shape",count);
    break;
   default: break;
  }
 }
 void certify_apply(int kind,int reason,int rows,int h,int q,int count){
  const int cW=std::max(1,plan.apply_c),cZ=std::max(1,plan.zc),cD=std::max(1,plan.dc);
  switch(reason){
   case CarrierWitness::R_ROWS_LT_C:
    certify_structural(kind,reason,rows,h,q,count,rows,cW,
      "W contracts over the row index: rows < c admits no c disjoint nonempty ranges (Sec 3 Split on [K])",count);break;
   case CarrierWitness::R_H_LT_CZ:
    certify_structural(kind,reason,rows,h,q,count,h,cZ,
      "Z contracts over the reflector index: h < c admits no c disjoint nonempty ranges (Alg 2, Z has its own carrier)",count);break;
   case CarrierWitness::R_H_LT_CD:
    certify_structural(kind,reason,rows,h,q,count,h,cD,
      "D contracts over the reflector index: h < c admits no c disjoint nonempty ranges (Alg 2, D has its own carrier)",count);break;
   case CarrierWitness::R_COUNT_NE_1:{
    // Real capacity, in the memory where the peers' pointer array lives. The replication is the
    // refused product's own: W, Z and D size their peer rows differently, so sharing one c here
    // would price another product's refusal.
    const int ck=kind==CarrierWitness::APPLY_Z?cZ:kind==CarrierWitness::APPLY_D?cD:cW;
    certify_capacity(kind,reason,rows,h,q,count,ck,
      (uint64_t)3*count*ck*sizeof(T*),(uint64_t)3*std::max<size_t>(plan.max_batch,1)*ck*sizeof(T*),
      "HBM peer pointer region, one per pipeline slot",
      "the batch is wider than the plan's max_batch, so the descriptor-sized peer row does not hold count*c peers of this product",count);break;}
   case CarrierWitness::R_ZC_LE_1:
    if(kind==CarrierWitness::APPLY_Z&&plan.z_priced){
     certify_priced(kind,reason,rows,h,q,count,plan.z_refused_s,plan.z_taken_s,
      "Z_replication=1 weighed faster than every Z_c>1 candidate on measured time at the selected shape; the c>1 refusal is priced, not capacity",count);break;}
    certify_capacity(kind,reason,rows,h,q,count,2,
      (uint64_t)plan.owned[ctx.rank],(uint64_t)plan.budget_bytes,
      "rank HBM admission budget",
      "the selected plan carries c=1 for this product: no candidate above one fitted the admission budget, or the arm was forced for an ablation",count);break;
   case CarrierWitness::R_DC_LE_1:
    if(kind==CarrierWitness::APPLY_D&&plan.d_priced){
     certify_priced(kind,reason,rows,h,q,count,plan.d_refused_s,plan.d_taken_s,
      "D_replication=1 weighed faster than every D_c>1 candidate on measured time at the selected shape; the c>1 refusal is priced, not capacity",count);break;}
    certify_capacity(kind,reason,rows,h,q,count,2,
      (uint64_t)plan.owned[ctx.rank],(uint64_t)plan.budget_bytes,
      "rank HBM admission budget",
      "the selected plan carries c=1 for this product: no candidate above one fitted the admission budget, or the arm was forced for an ablation",count);break;
   case CarrierWitness::R_APPLY_C_LE_1:
    certify_capacity(kind,reason,rows,h,q,count,2,
      (uint64_t)plan.owned[ctx.rank],(uint64_t)plan.budget_bytes,
      "rank HBM admission budget",
      "the selected plan carries c=1 for this product: no candidate above one fitted the admission budget, or the arm was forced for an ablation",count);break;
   case CarrierWitness::R_Z_SLICE_MIN_H:
    certify_unimplemented(kind,reason,rows,h,q,count,
      "native carried Z for h below one k-group per warp slice (h < BK-WK+1)",
      "h admits a c-way cut of the reflector index, but the native sliced Z gives each warp slice WK-wide sub-slabs of a BK-wide k-tile (fp64 BK=8c, WK=8; fp32 c=4 BK=128, WK=32), so a slice would own no reflector; the block-carried c=1 kernel runs instead -- an open implementation gap",count);break;
   case CarrierWitness::R_W_SLICE_MIN_ROWS:
    certify_unimplemented(kind,reason,rows,h,q,count,
      "native carried W for rows below one k-tile per cluster CTA ((c/2)*32 rows)",
      "rows admit a c-way cut of the row index, but the native W gives each of its c/2 cluster CTAs a contiguous range that must hold a whole k-tile for both warp slices to own rows; the block-carried c=1 kernel runs instead -- an open implementation gap",count);break;
   case CarrierWitness::R_D_SLICE_MIN_H:
    certify_unimplemented(kind,reason,rows,h,q,count,
      "native carried D for h below one warp-slice k-group per peer (c=2: h<9, c=4: h<25)",
      "h admits a c-way cut of the reflector index, but the native sliced D gives each warp slice WK=8-wide sub-slabs of a BK=8c k-tile, so a slice would own no reflector; the block-carried c=1 kernel runs instead -- an open implementation gap",count);break;
   case CarrierWitness::R_FUSED_MERGE:
    certify_priced(kind,reason,rows,h,q,count,plan.merge_carried_s,plan.merge_fused_s,
      "the fused pentagonal merge apply performs W, Z and D in ONE kernel over the support rows, so no [K] cut exists inside it; priced against the carried three-product sequence at this shape",count);break;
   default: break;
  }
 }
 // Record the block-level carrier a carried apply kernel just executed. The kernel launches dim3(q-tiles,
 // m-tiles, count) blocks of dim3(64, CB) threads, so peers live on axis y and cut K_z = [floor(K*z/CB),
 // floor(K*(z+1)/CB)). cb==0 means the GPU-level cuBLAS path ran (vendor-internal block geometry, left
 // empty) or nothing ran; nothing is recorded then. cb==1 is the clamped degenerate tail (one group owns
 // K_0 = [K]); it is recorded truthfully and stays a coverage gap, not a carrier. The device reports c peer
 // partials, c block peers, one membership report and one physical commit per member.
 static std::string sliced_partition(int bk,int wk,int cl){
  const std::string inner="r = |K_z| mod "+std::to_string(bk)+"; K_w = {k < r : floor(k/"+std::to_string(wk)+
   ") = w} u {k >= r : floor(((k - r) mod "+std::to_string(bk)+")/"+std::to_string(wk)+") = w} for w in [0,c), k relative to K_z";
  if(cl<=1)return "K_z = [K] (one CTA per tile); "+inner;
  return "cluster CTA z of "+std::to_string(cl)+" owns K_z = [floor(K*z/"+std::to_string(cl)+"), floor(K*(z+1)/"+std::to_string(cl)+")); "+inner;
 }
 void record_block_sliced(int kind,int c,int slice_threads,const char*index,uint64_t n=1,int bk=0,int wk=0,int cl=1,int gpu=1){
  if(c<2||!n)throw std::runtime_error("record_block_sliced_invalid");
  cw.levels(kind,c,std::max(1,cl),std::max(1,gpu),1,n,true);
  cw.note_block_peers(kind,c,n);cw.note_membership(n);
  if(kind==CarrierWitness::APPLY_D)cw.native_commits_issued+=n;
  if(!bk)carrier_d_law<T>(c,bk,wk);
  const std::string part=sliced_partition(bk,wk,cl);
  cw.block_carrier_geometry(kind,c,slice_threads,c,1,"y",index,part.c_str(),
   cl>1?"register partials parked in shared memory, summed through distributed shared memory in fixed order (z=0..CL-1, w=0..c-1); single commit"
       :"register partials parked in shared memory, summed in fixed order w=0..c-1; single coalesced commit",n);
 }
#ifdef TQR_MULTI_FIXED_DRIVER
 uint64_t multi_w_mma8_products=0;
 bool launch_multi_w_mma8(int rep,const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
   T*w,int ldw,long long sw,int rows,int h,int q,int count,cudaStream_t st){
  if constexpr(std::is_same_v<T,double>){
   static const bool enabled=[](){const char*p=std::getenv("TQR_MULTI_W_MMA8");return p&&std::string(p)=="1";}();
   static const bool paired=[](){const char*p=std::getenv("TQR_MULTI_W_MMA8_PAIR");return p&&std::string(p)=="1";}();
   if((!enabled&&!paired)||ctx.size<2||rep!=2||rows<32||h<1||q<1||count<1)return false;
   CarrierGArgs<double>a{x,ldx,sx,v,ldv,sv,w,ldw,sw,q,h,rows,witness.p,CarrierWitness::APPLY_W};
   if(paired){
    if(carrier_w_big<double>(rep,q,count))launch_carrier_g<MultiPairedWBig,1>(a,count,st);
    else launch_carrier_g<MultiPairedWSmall,1>(a,count,st);
   }
   else if(carrier_w_big<double>(rep,q,count))launch_carrier_g<MultiMma8WBig,1>(a,count,st);
   else launch_carrier_g<MultiMma8WSmall,1>(a,count,st);
   record_native_w(rep,q,count);
   multi_w_mma8_products+=count;
   return true;
  }
  return false;
 }
 void record_gmma_physical(int kind,const char*index,uint64_t n=1,int gpu=1){
  constexpr int threads=gmma_detail::GemmD::GemmKernel::MaxThreadsPerBlock;
  static_assert(threads==384&&gmma_detail::GemmW3::GemmKernel::MaxThreadsPerBlock==384&&
    gmma_detail::GemmW3a::GemmKernel::MaxThreadsPerBlock==384&&
    gmma_detail::GemmW3b::GemmKernel::MaxThreadsPerBlock==384,"audited physical GMMA geometry");
  cw.levels(kind,2,1,gpu,1,n,true);cw.note_block_peers(kind,2,n);cw.note_membership(n);
  if(kind==CarrierWitness::APPLY_D)cw.native_commits_issued+=n;
  cw.block_thread_group_carrier(kind,2,threads,128,128,
    "w = floor((threadIdx.x mod 256)/128) for threadIdx.x in [128,384); [0,128) are producer threads",
    "gmma_ksplit",index,
    "within each GPU K slice: K_w = {k in [K] : floor((k mod 32)/16) = w}; padded coordinates contribute zero",
    "two full-tile register partials; fixed-order p0+p1 through shared memory; owned half passed to epilogue",n);
 }
 void record_kami_physical(int threads,uint64_t n){
  const int kind=CarrierWitness::APPLY_D;
  cw.levels(kind,2,1,1,1,n,true);cw.note_block_peers(kind,2,n);cw.note_membership(n);cw.native_commits_issued+=n;
  cw.block_thread_group_carrier(kind,2,threads,threads/2,0,
    "w = floor(threadIdx.x / threads_per_peer)","kami_dnp","reflectors",
    "s = 8*ceil(ceil(K/8)/2); K_w = [min(K,w*s), min(K,(w+1)*s)) for w in {0,1}",
    "peer 1 parks its full partial; peer 0 adds p0+p1 and commits X-D exactly once",n);
 }
 bool launch_multi_kami_w(int rep,const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,
   T*w,int ldw,long long sw,int rows,int h,int q,int count,cudaStream_t st){
  if constexpr(std::is_same_v<T,double>){
   static const bool enabled=[](){const char*p=std::getenv("TQR_MULTI_KAMI_W");return p&&std::string(p)=="1";}();
   if(!enabled||ctx.size<2||rep!=2||h<1||q<1||count<1||ldw<h||
      !kami::w_admits(v,ldv,sv,x,ldx,sx,rows,count))return false;
   static const bool wide_rows=[](){const char*p=std::getenv("TQR_MULTI_KAMI_W_TILE");return p&&std::string(p)=="128x64";}();
   if(wide_rows)kami::launch_w<kami::MultiW128x64>(v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count,st,witness.p,CarrierWitness::APPLY_W);
   else kami::launch_w(v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count,st,witness.p,CarrierWitness::APPLY_W);
   const int kind=CarrierWitness::APPLY_W;
   cw.levels(kind,2,1,1,1,count,true);cw.note_block_peers(kind,2,count);cw.note_membership(count);
   cw.block_thread_group_carrier(kind,2,kami::W64x128::Threads,kami::W64x128::LayerThreads,0,
     "w = floor(threadIdx.x / threads_per_peer)","kami_w","rows",
     "n = ceil(K/16); K_w = [min(K,16*floor(n*w/2)), min(K,16*floor(n*(w+1)/2))) for w in {0,1}",
     "peer 1 parks its full partial; peer 0 adds p0+p1 and commits W exactly once",count);
   return true;
  }
  return false;
 }
#endif
 void record_native_w(int rep,int q,int count,int kind=CarrierWitness::APPLY_W,int gpu=1){
  int bk,wk;carrier_w_law<T>(bk,wk);
  record_block_sliced(kind,carrier_w_sk,carrier_w_slice_threads<T>(rep,q,count),"rows",count,bk,wk,rep/carrier_w_sk,gpu);
 }
 void record_native_z(int zrep,int count,int h){
  if(short_carrier_path<T>(zrep,h,false)){record_block_apply(CarrierWitness::APPLY_Z,zrep,"reflectors",count);return;}
  int bk,wk,sk,cl;carrier_z_law<T>(zrep,bk,wk);carrier_z_split<T>(zrep,sk,cl);
  record_block_sliced(CarrierWitness::APPLY_Z,sk,carrier_z_slice_threads<T>(zrep),"reflectors",count,bk,wk,cl);
 }
 void record_block_apply(int kind,int cb,const char*index,uint64_t n=1,int gpu_c=1){
  if(n)cw.levels(kind,cb>0?cb:0,1,cb>0?1:std::max(1,gpu_c),1,n,cb>0);
  if(cb<1||!n)return;
  // The launched kernel runs gridDim.z==count slices (launch helpers), each slice one member with
  // CB executed peers (blockDim.y, already clamped to the contraction). Membership: one report per
  // slice. cb==0 (cuBLAS path) records nothing here: library partials are counted in product() and
  // labeled vendor-unknown in the receipt.
  cw.note_block_peers(kind,cb,n);cw.note_membership(n);
  if(kind==CarrierWitness::APPLY_D)cw.native_commits_issued+=n;
  if(cb==1){
   cw.block_carrier_geometry(kind,1,64,1,1,"y",index,
    "K_0 = [K]; the contraction admits no cut at the selected CB",
    "single group; no cross-group partials",n);return;}
  cw.block_carrier_geometry(kind,cb,64,cb,1,"y",index,
   "K_z = [floor(K*z/CB), floor(K*(z+1)/CB)) for z in [0,CB)",
   "register partials summed in shared scratch; single commit",n);
 }

 void certify_elimination(int packets,int slots,int trailing,const char*where,uint64_t n=1){
  // An elimination with NO packets issued nothing, so CarrierWitness::elimination records it as an
  // empty terminal and never touches slots_per_elimination. The two must agree on what counts as an
  // elimination.
  if(packets<=0)return;
  const double nan=std::numeric_limits<double>::quiet_NaN();
  if(packets==1){
   // Where a narrower strip WAS admitted, the refusal is a measurement: the selector priced the r>1
   // candidates and r=1 won, so the certificate carries both numbers.
   if(plan.widest_packets>1)
    // A TAIL of a genuinely multi-packet schedule. Nothing circular: the r=1 here is the shape of
    // the trailing edge, not a strip chosen to produce it.
    certify_depth("structural",-1,trailing,plan.strip+1,nan,nan,
      "the trailing edge: this elimination has at most one strip of columns left, while the widest elimination of the same schedule issues more than one packet, so r=1 here is the shape of the matrix and not of the strip",n);
   else if(plan.strip_floor_q>0&&trailing>plan.strip_floor_q)
    // The WHOLE schedule is single-packet and a narrower strip was admitted, so
    // "the trailing width is at most one strip" would be citing a strip chosen
    // to make it true. The refusal has to be the measurement instead.
    certify_depth("priced",-1,trailing,2,plan.depth_price_rmulti_s,plan.depth_price_r1_s,
      "the whole schedule is single-packet and a narrower strip was admitted, so r=1 is not structural: the selector priced the best r>1 schedule against the best single-packet one over the whole elimination list and the single-packet schedule won",n);
   else
    certify_depth("structural",-1,trailing,std::max(plan.strip_floor_q,plan.strip)+1,nan,nan,
      "no admitted strip splits this elimination: its trailing width is at most the narrowest strip the selector would consider, so it issues r=1 packets and Eq pipe gives C(d)=t_Sigma for every depth",n);
  }
  else if(slots==1&&plan.d<2)
   certify_depth("capacity",-1,0,2,nan,nan,
     "the selected plan carries d=1: no depth above one fitted the admission budget's W/Z credits, or the arm was forced for an ablation",n,
     (uint64_t)plan.owned[ctx.rank],(uint64_t)plan.budget_bytes);
  else if(slots==1)
   // One ORDERED EPISODE STREAM exists at this level, and a second lane would
   // need a second one. That is the number: 1 available against 2 required.
   certify_depth("structural",-1,1,2,nan,nan,where,n);
 }
 void certify_depth(const char*basis,int reason,int contraction,int smallest,
                    double refused_s,double taken_s,const char*statement,uint64_t n=1,
                    uint64_t required=0,uint64_t available=0){
  Cert c;c.basis=basis;c.reason=reason;c.contraction=contraction;c.smallest_admitted_c=smallest;
  c.refused_s=refused_s;c.taken_s=taken_s;c.statement=statement;
  c.bytes_required=required;c.bytes_available=available;
  const std::string b(basis);
  if(b=="structural"&&!(contraction<smallest))return;
  if(b=="capacity"&&(!available||available==SIZE_MAX||!(required>available)))return;
  if(b=="priced"&&!(std::isfinite(refused_s)&&std::isfinite(taken_s)&&refused_s>0&&taken_s>0))return;
  cw.certify_depth(c,n);
 }
 unsigned exec_begin(int slot,int kind,unsigned packet,cudaStream_t st){
  if(!exec_cap)return ~0u;const unsigned idx=exec_next++;exec_snapshot();
  exec_open<<<1,1,0,st>>>(exec_log.p,exec_cap,idx,(unsigned)slot,(unsigned)kind,exec_elimination,packet,witness.p);
  return idx;}
 void exec_end(unsigned idx,cudaStream_t st){
  if(!exec_cap||idx==~0u)return;unsigned d[CarrierWitness::KINDS];
  for(int k=0;k<CarrierWitness::KINDS;++k){uint64_t t=0;for(auto&e:cw.primary_products(k))t+=e.second;
   d[k]=(unsigned)(t>=exec_mark[k]?t-exec_mark[k]:0);}
  exec_close<<<1,1,0,st>>>(exec_log.p,exec_cap,idx,witness.p,d[0],d[1],d[2],d[3],d[4],d[5],d[6]);}
 double merge_total_s=0,merge_presync_s=0,merge_transport_s=0,local_panel_s=0;uint64_t merge_calls=0;
  Buffer<T>column_scale,scale_temp;
  Buffer<int>gau_flags;Buffer<T>gau_tau,gau_O,gau_vs;size_t gau_count=0;int gau_epoch=0;uint64_t gau_panels=0,gau_refusals=0,gau_tt_levels=0,gau_sm1_packets=0;
  static bool gau_enabled(){static const int e=[]{const char*v=getenv("TQR_GAU_GE");return v?atoi(v):1;}();return e!=0;}
  // Fused single-SM register panel (gau_fast.cuh: chain + reflector overlaps + T in one launch) for packets that fit
  // one SM; TQR_GAU_SM=0 keeps the owner-CTA chain + compose_G + T builder for every packet.
  static bool gau_sm_enabled(){static const int e=[]{const char*v=getenv("TQR_GAU_SM");return v?atoi(v):1;}();return e!=0;}
  Buffer<T>coop_partial;
  // G = V_prev^T V_window, the operand of Eq compose one level up. b x b is an
  // upper bound for every window split of a panel.
  Buffer<T>win_g;
  // GPU-level carrier partial slices (compose_G): <= 128 groups of a b x b output.
  Buffer<T>GPART,GPART_N;   // GPART_N = the near slot's compose_G partial slices
 // per-slot row-major copy of the product's V for the TF32 GMMA D (fp32 plans only), leaf x (b agg)
 // words per lane slot; charged by gmma_vr_words in the inventory.
 Buffer<T>VR;
 size_t vr_ld()const{return size_t(plan.b)*size_t(std::max(1,plan.agg));}
 // The slot holds VR3 (leaf x 3 vr_ld: three pre-split K segments) and Z3 (3 vr_ld x strip). 0 = none.
 int la_free_sms()const{return plan.la!=0?std::max(0,plan.la_free):0;}
 int la_gmma_cap()const{const int f=la_free_sms();if(!f)return 0;int dev=0,sms=0;CU(cudaGetDevice(&dev));
   CU(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));return std::max(1,sms-f);}
 unsigned long long la_capped_products=0,np_products=0;
 static bool x3_math(){if constexpr(std::is_same_v<T,float>)return fp32_math()==Fp32Math::X3;else return false;}
 size_t vr_slot_words()const{return x3_math()?3*size_t(plan.leaf)*vr_ld()+3*vr_ld()*size_t(plan.strip):size_t(plan.leaf)*vr_ld();}
  // aggregation g, composed V_g (leaf x g b), T_g (g b x g b), G blocks.
  int agg=1;Buffer<T>VG,TG,TGP,AGG_G,AGG_S;Buffer<int>AGG_PERM;int agg_perm_off[17]={};bool agg_layered[17]={};
  int agg_gens=1,agg_gen=0;size_t vg_words=0,tg_words=0;
  // VR pack cache for the composed far update. product() is called once per strip, and the TF32/x3 GMMA D re-packed
  // the SAME V_g into the slot's row-major VR for every strip (nsys tf32 65536^2: 1058 pack_vr launches, 311 ms on
  // the far streams). A slot's VR is rewritten only by products on that slot's (ordered) stream, and V_g generation
  // g is rewritten only by composed_end, which bumps vg_epoch[g]. Key: source, shape, generation epoch. Pure data
  // movement (the same bytes): bit-identical. TQR_VR_CACHE=0 disables.
  unsigned long long vg_epoch[2]={1,1};
  struct VrKey{const T*v=nullptr;int ldv=0,rows=0,h=0,hs=0;bool x3=false;unsigned long long ep=0;bool ok=false;
   bool operator==(const VrKey&o)const{return ok&&o.ok&&v==o.v&&ldv==o.ldv&&rows==o.rows&&h==o.h&&hs==o.hs&&x3==o.x3&&ep==o.ep;}};
  std::vector<VrKey> vr_cache;unsigned long long vr_packs=0,vr_pack_hits=0,compose_tiled_steps=0;
  static bool vr_cache_on(){static const bool on=[]{const char*e=std::getenv("TQR_VR_CACHE");return !(e&&std::string(e)=="0");}();return on;}
  VrKey vr_key(const T*v,int ldv,int rows,int h,int hs,bool x3)const{VrKey k;k.v=v;k.ldv=ldv;k.rows=rows;k.h=h;k.hs=hs;k.x3=x3;
   if(vr_cache_on()&&VG.n&&vg_words&&v>=VG.p&&v<VG.p+VG.n){const size_t g=size_t(v-VG.p)/vg_words;if(g<2){k.ep=vg_epoch[g];k.ok=true;}}
   return k;}
  // true when slot's VR must be (re)packed; records the new content either way
  bool vr_need_pack(int slot,const VrKey&k){if(vr_cache.size()<=size_t(slot))vr_cache.resize(slot+1);
   if(k.ok&&vr_cache[slot]==k){++vr_pack_hits;return false;}vr_cache[slot]=k;++vr_packs;return true;}
  int depth_cap=0;   // ge_batch strips in flight when > 0 (the near slot is a single slot)
  size_t near_deferred=0,near_split_a=0;   // deferred near applies / split windows A on the near slot (evidence)
  size_t far_deferred=0,resident_packed=0; // deferred far issues and fused V exports
  Event near_fork,near_done,near_compose_done;
  T*VGp()const{return VG.p+size_t(agg_gen)*vg_words;}
  T*TGp()const{return TG.p+size_t(agg_gen)*tg_words;}
  T*TGPp()const{return TGP.p+size_t(agg_gen)*tg_words;}
  bool balanced_aggregate_layers()const{return plan.qr_schedule.value("aggregate_layer_policy",std::string("whole_constituents"))=="balanced_reflectors";}
  // D-5 (ii) layer order for every group size g' in [2, agg], computed once from the D carrier law of
  // this executor (precision mode fixed at setup) and uploaded before the first factorization.
  void p7_setup_layers(){
   const int b=plan.b,dc=std::max(1,plan.dc);std::vector<int>all;
   for(int g=2;g<=agg;++g){const int H=g*b;agg_perm_off[g]=int(all.size());std::vector<int>perm(H);for(int q=0;q<H;++q)perm[q]=q;
    agg_layered[g]=false;
    if(dc>=2&&native_d_path<T>(dc,H)){int bk,wk;carrier_d_law<T>(dc,bk,wk);
     const bool whole=dc>=g?dc%g==0:g%dc==0;
     // Paper Split partitions the product's reflector index K. It need not
     // stop at a constituent panel boundary: an odd group can give each peer
     // H/c reflectors, including part of one constituent. P is still bijective
     // and (VP)(P^T T P)(VP)^T = V T V^T. Keep the narrower whole-constituent
     // receipt false for this case, and retain it as the default restriction.
     if(bk==dc*wk&&H%bk==0&&(whole||balanced_aggregate_layers())){agg_layered[g]=whole;
      for(int q=0;q<H;++q){const int sl=(q%bk)/wk;perm[q]=sl*(H/dc)+(q/bk)*wk+(q%wk);}
      std::vector<int>seen(H,0);for(int v:perm){if(v<0||v>=H||seen[v]++)throw std::runtime_error("p7_layer_perm");}}}
    all.insert(all.end(),perm.begin(),perm.end());}
   if(!all.empty()){AGG_PERM.alloc(all.size());CU(cudaMemcpy(AGG_PERM.p,all.data(),all.size()*sizeof(int),cudaMemcpyHostToDevice));}
  }
  // True when the cooperative window loop just packed every V column into
  // vbuf itself (one pack_ge_v_cols per window, including the last). The
  // post-batch full pack_ge_v then recomputes identical bytes from the same
  // A and is skipped. Set only on the pw path with nwin>1; reset on entry.
  bool coop_packed_v=false;
  Buffer<T>scalar_tau;int*host_status=nullptr;
  std::unique_ptr<Transport>transport;TiledMatrix<T>*factor_A=nullptr;uint64_t invocation=0,expression=0;bool factored=false;
  void hh(T*a,int ld,int count,int kind,bool shared,size_t dynamic,uint64_t*hist=nullptr,int hist_next=0){cw.block_carrier(kind==0?CarrierWitness::PANEL_GE:CarrierWitness::TT_GE,plan.threads,"rows","warp shuffle then shared scratch (tile_reduce)",count);
   // The device reports blockDim.x peers per executed block; the host notes the same number here.
   // One membership report per member (one block per member: grid==count).
   cw.note_block_peers(kind==0?CarrierWitness::PANEL_GE:CarrierWitness::TT_GE,plan.threads,count);
   cw.note_membership(count);
   if(shared)tiled_ge<T,true><<<count,plan.threads,dynamic,stream>>>(a,ld,packets,count,tri.p,plan.b,kind,status.p,witness.p,hist,hist_next);else tiled_ge<T,false><<<count,plan.threads,0,stream>>>(a,ld,packets,count,tri.p,plan.b,kind,status.p,witness.p,hist,hist_next);}
  // Tries groups 32/16/8 and threads 512/256; falls back to hh on any refusal. Group menu ordered by measurement: the
  // optimum tracks rows/64 -- rows=4096 wants 64 groups, rows=16384 wants 128, and rows<=1024 wants 32. Ordered
  // best-first with the others as fallbacks, since launch_cooperative_ge refuses on residency or shared. Dispatches
  // the carrier the SELECTOR priced -- (cc, cpw, cthreads) from TileBatch -- and nothing else. Remaining menu entries
  // are tried only as FALLBACKS after a launcher refusal, and every refusal is logged rather than swallowed. Returns
  // the group count (the carrier's contraction replication c) that was ACTUALLY launched, or 0 on refusal. It returns
  // the real value rather than a bool because the menu below may fall back to a different (groups,width) than the one
  // the selector priced, and the witness must report the c that executed, not the c that was chosen. The deferred
  // half of a WINDOWED panel, issued between two cooperative launches.
  //  Eq compose  T12 = -T1 (V1^T V2) T2 against the windows already factored,
  //              whose (V1^T V2) is a carried product in its own right;
  //  Eq apply    the panel columns beyond the window, with the window's own
  //              Z and D theirs, and every one of them lands in the witness.
  // 28 of them per group = 11.5 ms serialized on the main stream while every far stream idled. Carrier: SK = 2 warp
  // slices x CL = c/2 cluster CTAs x CG GPU groups over [K] = rows (16-aligned balanced ranges), register partials
  // combined through DSM in fixed order, GPU partial slices (GPART) summed by carrier_g_combine in fixed order, one
  // commit. Returns false (caller keeps its path) when the precision, switch, c or alignment does not admit it.
  bool ieee_simt_w(int rep,const T*v,int ldv,long long sv,const T*x,int ldx,long long sx,T*w,int ldw,long long sw,
                   int rows,int h,int q,int count,cudaStream_t st,int kind,T*part=nullptr){
   if(!part)part=GPART.p;
   if constexpr(std::is_same_v<T,float>){
    if(fp32_math()!=Fp32Math::IEEE||!simt_compose_enabled()||rep<2||!rows||!h||!q||!count)return false;
    const int c=rep>=8?8:rep>=4?4:2;
    if(!simt_w_admits(c,v,ldv,x,ldx,w,ldw))return false;
    const size_t per=size_t(count)*size_t(ldw)*size_t(q);
    int cg=simt_w_gpu_split(c,q,h,rows,count);
    cg=int(std::max<size_t>(1,std::min<size_t>(size_t(cg),GPART.n/std::max<size_t>(1,per))));
    launch_simt_w(c,v,ldv,sv,x,ldx,sx,w,ldw,sw,rows,h,q,count,st,witness.p,kind,cg,part);
    cw.product(kind,c*cg,count,'N');
    record_block_sliced(kind,simt::SK,simt::SLICE,"rows",count,simt::BK,simt::WK,c/simt::SK,cg);
    return true;
   }
   return false;
  }
  void window_boundary(TiledMatrix<T>&x,const TileBatch&batch,int w0,int sw,int lane0){
   const int b=plan.b,leaf=plan.leaf,h=batch.h,rows=batch.ge[0].rows,w1=w0+sw,tile=batch.ge[0].tile;
   cudaStream_t st=slot_stream(lane0);cublasHandle_t bh=lane0?pipe_blas[lane0]:blas;
   pack_ge_v_cols<<<dim3(pack_v_grid(leaf,w1-w0),1),128,0,st>>>(factor_A->a.p,factor_A->ld,packets,1,vbuf(),leaf,b,w0,w1);
   if(w0>0){
    // G = V1^T V2 over the whole contraction [K] = the packet's rows. V2 is
    // zero above matrix row w0, so the rows below it contribute nothing and
    // the operands start there.
    // compose_G with its own carrier (Alg 2; paper Eq compose): the native W_ONLY kernel cuts the
    // contraction [K] = rows over cluster CTAs x warp slices, fixed-order DSM combine, one commit;
    // the device reports kind compose_G. IEEE fp32 (no native W) keeps the c>1 library
    // intermediary; c=1 only where the rows admit no cut, with its certificate.
    int gwhy=-1;const int grep=carrier_c(rows-w0,w0,sw,1,&gwhy);
    static const bool compose_side=[]{const char*e=std::getenv("TQR_MULTI_COMPOSE_SIDE");return !(e&&std::string(e)=="0");}();
    cudaStream_t cst=st;
    if(grep>1&&native_w_path<T>(grep,rows-w0)){
     if(compose_side){compose_fork.record(st);compose_fork.wait(compose_stream.s);cst=compose_stream.s;compose_pending=true;}
     // GPU level: the output is tiny (w0 x sw) and K = rows is long, so groups of CTAs cut the rows
     // further (HBM partial slices, fixed-order combine) -- otherwise one or two CTAs run all rows.
     const int gcg=carrier_w_gpu_split<T>(grep,sw,w0,rows-w0,1);
     launch_carrier_w<T>(grep,vbuf()+w0,leaf,0,vbuf()+size_t(w0)*leaf+w0,leaf,0,win_g.p,w0,size_t(w0)*sw,rows-w0,w0,sw,1,cst,witness.p,CarrierWitness::COMPOSE_G,gcg,GPART.p);
     cw.product(CarrierWitness::COMPOSE_G,grep*gcg,1,'N');record_native_w(grep,sw,1,CarrierWitness::COMPOSE_G,gcg);
    }else if(grep>1&&ieee_simt_w(grep,vbuf()+w0,leaf,0,vbuf()+size_t(w0)*leaf+w0,leaf,0,win_g.p,w0,size_t(w0)*sw,rows-w0,w0,sw,1,st,CarrierWitness::COMPOSE_G)){
    }else if(grep>1){
     cw.product(CarrierWitness::COMPOSE_G,grep,1,'L');
     record_block_apply(CarrierWitness::COMPOSE_G,apply_stage_W<T>(bh,st,grep,vbuf()+w0,leaf,0,vbuf()+size_t(w0)*leaf+w0,leaf,0,win_g.p,w0,size_t(w0)*sw,rows-w0,w0,sw,1,carrier_ptr.p+size_t(lane0)*carrier_ptr_slot(),&cw.remainder_peers,plan.wcb,witness.p),"rows",1,grep);
    }else{
     cw.product_uncarried(CarrierWitness::COMPOSE_G,gwhy);
     cw.native_partials[CarrierWitness::COMPOSE_G]-=1;cw.library_partials[CarrierWitness::COMPOSE_G]+=1;
     cw.levels(CarrierWitness::COMPOSE_G,0,1,1,1,1,false);certify_apply(CarrierWitness::COMPOSE_G,gwhy,rows-w0,w0,sw,1);
     tile_gemm<T>(bh,CUBLAS_OP_T,w0,sw,rows-w0,T(1),vbuf()+w0,leaf,0,vbuf()+size_t(w0)*leaf+w0,leaf,0,T(0),win_g.p,w0,0,1);}
    tile_compose_offdiag<T><<<sw,128,size_t(w0)*sizeof(T),cst>>>(tri.p+size_t(tile)*b*b,b,win_g.p,w0,sw);
   }
   for(int col=batch.col+w1;col<batch.col+h;){
    const int q=std::min(plan.strip,batch.col+h-col);
    product(x.a.p+batch.ge[0].row+size_t(col)*x.ld,x.ld,leaf,vbuf()+size_t(w0)*leaf,leaf,size_t(leaf)*b,
            tri.p+size_t(tile)*b*b+w0+size_t(w0)*b,size_t(b)*b,rows,sw,q,1,true,lane0);
    col+=q;
   }
  }
   // A valid capacity-driven alternative is selected and recorded BEFORE mutation (the stamped
   // batch.cc/cpw/cthreads/csw + certificates); an unsupported implementation path remains a failed qualification
   // (throw, never a silent block-path pass after partial work). Writes R/V into A once, V (unit diagonal, zeros
   // above) straight into vbuf, then T from the reflector overlaps O = V^T V (compose_G carrier, the same product
   // window_boundary issues) by dlarft + Eq compose. Returns the owner count p_j, 0 on a preflight refusal (before
   // any write). Reflector overlaps O_t = V_t^T V_t for `count` packets (V_t at V + t*vstride, ld ldv, K = rows),
   // into gau_O (h x h each, stride b*b), with the engine's compose_G carrier: the product, carrier selection and
   // witness window_boundary uses for T12 (c > 1 over [K] = rows, fixed-order combine, one commit), in the precision
   // of the selected mode.
   void gau_overlaps(const T*V,int ldv,size_t vstride,int rows,int h,int count,cudaStream_t st,cublasHandle_t bh,int lane0){
    T*O=gau_O.p;const size_t os=size_t(plan.b)*plan.b;
    int gwhy=-1;const int grep=carrier_c(rows,h,h,count,&gwhy);
    if(grep>1&&native_w_path<T>(grep,rows)){
     const int gcg=qrv2_overlap_gpu_groups(GPART.n,h,size_t(count),
                                         carrier_w_gpu_split<T>(grep,h,h,rows,count));
     launch_carrier_w<T>(grep,V,ldv,vstride,V,ldv,vstride,O,h,os,rows,h,h,count,st,witness.p,CarrierWitness::COMPOSE_G,gcg,GPART.p);
     cw.product(CarrierWitness::COMPOSE_G,grep*gcg,count,'N');record_native_w(grep,h,count,CarrierWitness::COMPOSE_G,gcg);
    }else if(grep>1&&ieee_simt_w(grep,V,ldv,(long long)vstride,V,ldv,(long long)vstride,O,h,(long long)os,rows,h,h,count,st,CarrierWitness::COMPOSE_G)){
    }else if(grep>1){
     cw.product(CarrierWitness::COMPOSE_G,grep,count,'L');
     record_block_apply(CarrierWitness::COMPOSE_G,apply_stage_W<T>(bh,st,grep,V,ldv,vstride,V,ldv,vstride,O,h,os,rows,h,h,count,carrier_ptr.p+size_t(lane0)*carrier_ptr_slot(),&cw.remainder_peers,plan.wcb,witness.p),"rows",count,grep);
    }else{
     cw.product_uncarried(CarrierWitness::COMPOSE_G,gwhy,count);
     cw.native_partials[CarrierWitness::COMPOSE_G]-=count;cw.library_partials[CarrierWitness::COMPOSE_G]+=count;
     cw.levels(CarrierWitness::COMPOSE_G,0,1,1,1,count,false);certify_apply(CarrierWitness::COMPOSE_G,gwhy,rows,h,h,count);
     tile_gemm<T>(bh,CUBLAS_OP_T,h,h,rows,T(1),V,ldv,vstride,V,ldv,vstride,T(0),O,h,os,count);}
   }
   // Returns the owner count p_j, or 0 on a preflight refusal (before any write).
   int gau_ge_batch(TiledMatrix<T>&x,const TileBatch&batch,int lane0){
    const int count=batch.ge.size(),h=batch.h,b=plan.b,leaf=plan.leaf;
    if(!gau_enabled()||count<1||size_t(count)>gau_count||h<1||h>b||!gau_flags.p)return 0;
    int rows=0;for(auto&g:batch.ge){if(g.h!=h||g.rows<h)return 0;rows=std::max(rows,g.rows);}
    static const int gau_cap=[]{const char*e=getenv("TQR_GAU_MAX_ROWS");return e?atoi(e):0;}();
    if(gau_cap>0&&rows>gau_cap)return 0;
    cudaStream_t st=slot_stream(lane0);cublasHandle_t bh=lane0?pipe_blas[lane0]:blas;
    if(gau_sm_enabled()){GfConfig f;bool fits=false;
     try{f=launch_gf_sm<T>(1,x.a.p,x.ld,packets,count,rows,h,vbuf(),leaf,size_t(leaf)*b,gau_tau.p,b,tri.p,b,0,status.p,witness.p,st,true);fits=f.items<=8;}
     catch(const std::exception&){}
     if(fits){
      launch_gf_sm<T>(1,x.a.p,x.ld,packets,count,rows,h,vbuf(),leaf,size_t(leaf)*b,gau_tau.p,b,tri.p,b,0,status.p,witness.p,st,false);
      coop_packed_v=true;gau_panels+=count;gau_sm1_packets+=count;
      const int owners=(h+f.vec-1)/f.vec;
      auto receipt=cw.qrv2_factor(CarrierWitness::PANEL_GE,true,rows,h,f.vec,f.warps*32,count);
      {std::string bkey="p"+std::to_string(batch.col)+"r"+std::to_string(batch.rank);
       cw.note_panel_batch_executed(bkey,1);cw.note_panel_batch_detail(bkey,1,f.vec,f.warps*32,0);
       cw.panel_batch_executed_detail[bkey]["qrv2"]=receipt;}
      return owners;}
    }
    GauConfig g;
    try{g=launch_gau_ge<T>(x.a.p,x.ld,packets,count,rows,h,vbuf(),leaf,size_t(leaf)*b,gau_tau.p,gau_flags.p,b,gau_epoch+1,0,status.p,witness.p,st,true);}
    catch(const std::exception&e){++gau_refusals;cw.note_preflight("gau_ge_refused");return 0;}
    ++gau_epoch;
    launch_gau_ge<T>(x.a.p,x.ld,packets,count,rows,h,vbuf(),leaf,size_t(leaf)*b,gau_tau.p,gau_flags.p,b,gau_epoch,0,status.p,witness.p,st,false);
    gau_overlaps(vbuf(),leaf,size_t(leaf)*b,rows,h,count,st,bh,lane0);
    launch_gau_build_t_batch<T>(gau_O.p,h,size_t(b)*b,gau_tau.p,b,packets,count,h,tri.p,b,st);
    coop_packed_v=true;gau_panels+=count;
    const int owners=(h+g.C-1)/g.C;
    auto receipt=cw.qrv2_factor(CarrierWitness::PANEL_GE,false,rows,h,g.C,g.threads,count);
    {std::string bkey="p"+std::to_string(batch.col)+"r"+std::to_string(batch.rank);
     cw.note_panel_batch_executed(bkey,1);cw.note_panel_batch_detail(bkey,1,g.C,g.threads,0);
     cw.panel_batch_executed_detail[bkey]["qrv2"]=receipt;}
    return owners;
   }
   // V is staged in gau_vs, and the structural zeros of the stacked triangles stay exact zeros. Returns false on a
   // preflight refusal (before any write).
   bool multi_wide_panel()const{return ctx.size>1&&plan.qr_schedule.value("multi_wide_panel",false);}
   bool gau_tt_level(int count,int h,int kind,int stack_ld=0){
    // Global nodes can have a different arity from local tree nodes. Their
    // packet still describes the actual (possibly ragged) rows; stack_ld is
    // the physical leading dimension, including zero padding.
    const int b=plan.b,m=stack_ld?stack_ld:plan.radix*b;
    if(!gau_enabled()||count<1||size_t(count)>gau_count||!gau_vs.p)return false;
    if(size_t(count)*size_t(m)*b>gau_vs.n)return false;
    if(multi_wide_panel()&&h>128){
     // A wide triangular stack is still one ordered Householder chain.
     // Preflight both its owner-CTA wavefront and the bounded WY builder.
     if(count!=1||win_g.n<size_t(b)*b)throw std::runtime_error("wide_global_factor_inventory_before_modify");
     const GauConfig g=launch_gau_ge<T>(stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,gau_flags.p,b,gau_epoch+1,kind,status.p,witness.p,stream,true,true);
     multi_wide_t_preflight<T>(h,b,count);
     ++gau_epoch;
     launch_gau_ge<T>(stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,gau_flags.p,b,gau_epoch,kind,status.p,witness.p,stream,false,true);
     gau_overlaps(gau_vs.p,m,size_t(m)*b,m,h,count,stream,blas,0);
     launch_multi_wide_t(gau_O.p,h,gau_tau.p,packets,h,tri.p,b,win_g.p,stream);
     ++gau_tt_levels;cw.qrv2_factor(CarrierWitness::TT_GE,false,m,h,g.C,g.threads,count);
     return true;
    }
    if(gau_sm_enabled()){GfConfig f;bool fits=false;
     try{f=launch_gf_sm<T>(1,stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,b,tri.p,b,kind,status.p,witness.p,stream,true);fits=f.items<=8;}
     catch(const std::exception&){}
     if(fits){
      launch_gf_sm<T>(1,stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,b,tri.p,b,kind,status.p,witness.p,stream,false);
      ++gau_tt_levels;gau_sm1_packets+=count;
      cw.qrv2_factor(CarrierWitness::TT_GE,true,m,h,f.vec,f.warps*32,count);
      return true;}
    }
    try{launch_gau_ge<T>(stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,gau_flags.p,b,gau_epoch+1,kind,status.p,witness.p,stream,true);}
    catch(const std::exception&e){++gau_refusals;cw.note_preflight("gau_tt_refused");return false;}
    ++gau_epoch;
    const GauConfig g=launch_gau_ge<T>(stack.p,m,packets,count,m,h,gau_vs.p,m,size_t(m)*b,gau_tau.p,gau_flags.p,b,gau_epoch,kind,status.p,witness.p,stream,false);
    gau_overlaps(gau_vs.p,m,size_t(m)*b,m,h,count,stream,blas,0);
    launch_gau_build_t_batch<T>(gau_O.p,h,size_t(b)*b,gau_tau.p,b,packets,count,h,tri.p,b,stream);
    ++gau_tt_levels;
    cw.qrv2_factor(CarrierWitness::TT_GE,false,m,h,g.C,g.threads,count);
    return true;
   }
   size_t resident_panels=0;
   int cooperative_ge_batch(TiledMatrix<T>&x,const TileBatch&batch,int count,int lane0){
    T*a=x.a.p;const int ld=x.ld,max_rows=batch.ge[0].rows,h=batch.h;
    const int cc=batch.cc,cpw=batch.cpw,cthreads=batch.cthreads;
    coop_packed_v=false;
   if(h<1||h>(multi_wide_panel()?512:128)||max_rows<h||count<1||cc<1) return 0;
   const int csw=(batch.csw>0&&batch.csw<h)?batch.csw:h;
    // The fallback menu below survives only for qualification fixtures, whose stamps are forced by
    // tests (profile_id "qualification-only").
    const bool selected_stamp=plan.profile_id!="qualification-only";
    std::vector<std::pair<int,int>> menu{{cc,cpw}};
    if(!selected_stamp)for(int g:coop_group_menu())for(int pw:coop_width_menu())if(g!=cc||pw!=cpw)menu.push_back({g,pw});
    const int t0=cthreads?cthreads:coop_threads_for(plan.threads),t1=t0==512?256:512;
    std::vector<int> thread_menu{t0};if(!selected_stamp)thread_menu.push_back(t1);
    // Once any menu entry has performed a numerical write, no later entry
    // may be tried: a partially modified panel cannot be refactored silently.
    bool mutated=false;
    for(auto&gp:menu) for(int threads:thread_menu){
     const int groups=gp.first,pw=gp.second;
     if(pw>h){cw.note_preflight("coop_entry_pw_gt_h");continue;}
     const int sw0=std::min(csw,h),wpw=pw?std::min(pw,sw0):0;
     int nwin_out=1;bool resident=false;
     // PREFLIGHT PHASE (no numerical write): dry-check the whole window
     // sequence, scratch layout, and launch capability. Refusals here may
     // try the next menu entry and are counted as preflight.
     try{
      size_t need=pw?coop_mini_total_slots(groups,h,pw):size_t(2)*count*groups*(2+2*h);
      if(coop_partial.n<need){fprintf(stderr,"coop_batch_partial_too_small count=%d rows=%d h=%d g=%d pw=%d need=%zu have=%zu\n",count,max_rows,h,groups,pw,need,coop_partial.n);cw.note_preflight("coop_entry_partial_too_small");continue;}
      if(pw){
       if(count!=1){fprintf(stderr,"coop_minipanel_requires_single_packet count=%d\n",count);cw.note_preflight("coop_entry_minipanel_count");continue;}
       for(int w0=0;w0<h;w0+=sw0)
        launch_cooperative_ge_mini(a,ld,packets,max_rows,h,w0,std::min(sw0,h-w0),wpw,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream,true,multi_wide_panel()?512:128);
       if(coop_resident_env()){try{launch_cooperative_ge_resident<T>(a,ld,packets,max_rows,h,pw,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream,true);resident=true;}catch(...){resident=false;}}
       // Scratch for the window boundary (win_g is b*b; checked here so a
       // missing allocation fails BEFORE the first window mutates A).
       if(win_g.n<size_t(plan.b)*plan.b)throw std::runtime_error("cooperative_window_scratch_before_modify");
      } else {
       // Unwindowed preflight without launching: same rules as the launcher,
       // no second copy of the numbers (dry run via the mini launcher's dry
       // path is not applicable; check capability/capacity/residency here).
       int device;CU(cudaGetDevice(&device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,device));
       if(!prop.cooperativeLaunch||groups<1||groups>coop_partition_limit)throw std::runtime_error("cooperative_packet_capability_before_modify");
       {int local_height=ceildiv(max_rows,groups);size_t dynamic=coop_shared_bytes(local_height,h,sizeof(T));cudaFuncAttributes attr;CU(cudaFuncGetAttributes(&attr,cooperative_ge<T>));
        if(dynamic>prop.sharedMemPerBlockOptin-attr.sharedSizeBytes)throw std::runtime_error("cooperative_panel_shared_capacity_before_modify");
        CU(cudaFuncSetAttribute(cooperative_ge<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(dynamic)));
        int occupancy=0;CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cooperative_ge<T>,threads,dynamic));
        if(size_t(count)*groups>size_t(occupancy)*prop.multiProcessorCount)throw std::runtime_error("cooperative_grid_residency_before_modify");}
      }
     }catch(const std::exception&e){
      if(mutated)throw;
      if(selected_stamp)throw std::runtime_error(std::string("selected_panel_carrier_refused_before_modify p")+std::to_string(batch.col)+"r"+std::to_string(batch.rank)+" rows="+std::to_string(max_rows)+" h="+std::to_string(h)+" g="+std::to_string(groups)+" pw="+std::to_string(pw)+" t="+std::to_string(threads)+" sw="+std::to_string(csw)+": "+e.what());
      fprintf(stderr,"coop_batch_preflight count=%d rows=%d h=%d g=%d pw=%d t=%d: %s\n",count,max_rows,h,groups,pw,threads,e.what());cw.note_preflight("coop_entry_launcher_exception");continue;
     }catch(...){ if(mutated||selected_stamp)throw; cw.note_preflight("coop_entry_launcher_exception");continue; }
     // NUMERICAL PHASE: from here any failure FAILS the invocation. No
     // catch-and-continue may try another carrier after partial mutation.
     try{
      mutated=true;
      if(pw&&resident){
       static const bool pack=[]{const char*e=std::getenv("TQR_RES_PACK");return e&&std::string(e)=="1";}();
       launch_cooperative_ge_resident<T>(a,ld,packets,max_rows,h,pw,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream,false,pack?vbuf():nullptr,plan.leaf);
       nwin_out=1;coop_packed_v=pack;++resident_panels;if(pack)++resident_packed;
      }
      else if(pw){
       int last_w0=0,last_sw=h,nwin=0;
       for(int w0=0;w0<h;w0+=sw0){const int sw=std::min(sw0,h-w0);
        launch_cooperative_ge_mini(a,ld,packets,max_rows,h,w0,sw,wpw,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream,false,multi_wide_panel()?512:128);
        // Diagnostic fault injection: after the first window has modified A, an injected fault must
        // fail the invocation rather than silently refactoring onto another carrier.
#ifdef TQR_C1_FAULT_AFTER_FIRST_WINDOW
        if(nwin==0&&tqr_c1_fault_after_first_window)throw std::runtime_error("c1_fault_injected_after_first_window");
#endif
        if(w0+sw<h)window_boundary(x,batch,w0,sw,lane0);
        last_w0=w0;last_sw=sw;++nwin;
       }
       if(nwin>1)window_boundary(x,batch,last_w0,last_sw,lane0);
       // The composed T of this panel is complete once the side compose stream drains: join it before any Apply.
       if(compose_pending){compose_join.record(compose_stream.s);compose_join.wait(stream.s);if(slot_stream(lane0)!=stream.s)compose_join.wait(slot_stream(lane0));compose_pending=false;}
       if(nwin>1)coop_packed_v=true;
       nwin_out=nwin;
      }
      else launch_cooperative_ge(a,ld,packets,count,max_rows,h,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream);
      cw.block_carrier(CarrierWitness::PANEL_GE,threads,"rows within the CTA row slab","warp shuffle then shared scratch",count);
      cw.note_suboperations(CarrierWitness::PANEL_GE,groups,uint64_t(nwin_out-1)*uint64_t(count));
      cw.note_membership(uint64_t(nwin_out)*uint64_t(groups)*uint64_t(count));
      // Per-batch executed record for selected-vs-executed equality.
      {std::string bkey="p"+std::to_string(batch.col)+"r"+std::to_string(batch.rank);
       cw.note_panel_batch_executed(bkey,groups);
       cw.note_panel_batch_detail(bkey,groups,wpw,threads,pw?(sw0<h?sw0:0):0);}
      return groups;
     }catch(const std::exception&e){
      fprintf(stderr,"coop_batch_failed_after_mutation count=%d rows=%d h=%d g=%d pw=%d t=%d: %s\n",count,max_rows,h,groups,pw,threads,e.what());
      throw;
     }catch(...){ throw; }
    }
   return 0;
  }
  bool cooperative_ge_packet(T*a,int ld,const TilePacket*pdev,int rows,int h,int tile){
   if(h<1||h>128||rows<h) return false;
   int groups=32;
   // Cooperative kernel has higher register pressure than tiled_ge; the probe shows threads=512
   // best and 1024 often non-resident (occupancy 0). Try 512 then 256, else let the caller fall
   // back to hh.
   for(int threads:{512,256}){
    try{
     size_t need=size_t(2)*groups*(2+2*h);
     if(coop_partial.n<need) return false;
     launch_cooperative_ge(a,ld,pdev,1,rows,h,groups,threads,tri.p,plan.b,coop_partial.p,status.p,witness.p,stream);
     return true;
    }catch(const std::exception&e){ fprintf(stderr,"cooperative_ge_fallback rows=%d h=%d threads=%d: %s\n",rows,h,threads,e.what()); continue; }
    catch(...){ continue; }
   }
   return false;
  }
 void select_packets(int first){packets=all_packets.p+first;merges=all_merges.p+first;}
 void normalize_columns(TiledMatrix<T>&x,T*scales){
  if(!x.n)return;
  tiled_column_max<<<x.n,256,0,stream>>>(x.a.p,x.nr,x.n,x.ld,scales);
  if(ctx.size>1)for(int col=0;col<x.n;col+=plan.strip){int q=std::min(plan.strip,x.n-col);
   // The sender's current maximum is frozen for each copy. The following
   // maximum is an ordinary GPU instruction, never a network reduction.
   for(int root=0;root<ctx.size;++root){transport->template paper<T,false>(root,scales+col,scale_temp.p,q,++expression);
    tiled_scale_max<<<ceildiv(q,128),128,0,stream>>>(scales+col,scale_temp.p,q);
   }
  }
  if(x.nr)tiled_column_scale<T,false><<<column_pass_grid(x.nr,x.n),256,0,stream>>>(x.a.p,x.nr,x.n,x.ld,x.begin,scales,status.p);
 }
 void history_start(int count,int col){tile_history_begin<<<ceildiv(count,128),128,0,stream>>>(hist.p,packets,count,col,status.p);}
 void history_advance(int count,int col,int end,cudaStream_t st=nullptr){tile_history_commit<<<ceildiv(count,128),128,0,st?st:stream.s>>>(hist.p,packets,count,col,end,status.p,witness.p);}
 // Eq apply on one strip, carried in pipeline slot `slot`: W, then Z, then D.
 // The slot picks both the stream and the W/Z credit, so two strips in flight
 // never share a buffer.
 void product(T*dst,int ldx,long long sx,const T*v,int ldv,long long sv,const T*t,long long st,int rows,int h,int q,int count,bool transpose,int slot=0){
  const int b=std::max(plan.b,h);   // ld of T, W, Z: b, or g b for a composed Apply
  const bool window_b=plan.la!=0&&slot>=std::max(1,plan.d)&&slot<2*std::max(1,plan.d);
  // TQR_GMMA_NP=1 (window B) / 2 (all): one GMMA CTA per tile instead of a persistent grid (carrier_gmma.cuh).
  const bool np=gmma_detail::gmma_np_mode()==2||(gmma_detail::gmma_np_mode()==1&&window_b);
  const int la_cap=np?gmma_detail::gmma_np_cap:(window_b?la_gmma_cap():0);
  gmma_detail::GmmaSmCapScope cap_scope(la_cap);if(la_cap>0)++la_capped_products;if(np)++np_products;
  cublasHandle_t bh=slot?pipe_blas[slot]:blas;cudaStream_t sst=slot_stream(slot);
  // W holds the c partials side by side (stride carries apply_c); Z is one
  // b×strip block per batch and its stride must NOT carry apply_c -- using
  // the W stride for Z overruns the slot for every apply_c>1 (silent
  // corruption at ac=2/4, a fault at ac=8).
  T*w=W.p+size_t(slot)*credit_words(),*z=Z.p+size_t(slot)*z_credit_words();
  T**pp=carrier_ptr.p+size_t(slot)*carrier_ptr_slot();
  // W = V^T X with contraction replication c. There is no path to c=1 here that does not carry its
  // attribution.
  int why=-1;const int rep=carrier_c(rows,h,q,count,&why);
  // fp64 W's c>1 peers as default-math vendor GEMMs over disjoint K_z (peer_stage_W); TQR_W_PEERS=0 keeps the
  // native CUTLASS carrier.
  static const bool w_peers=[]{const char*e=std::getenv("TQR_W_PEERS");return !(e&&std::string(e)=="0");}();
  static const int w_peers_minq=[]{const char*e=std::getenv("TQR_W_PEERS_MINQ");return e?std::atoi(e):1024;}();
  // TQR_W_PEERS_MINROWS (8192), TQR_W_PEERS_MINROWS_PEER (4096).
  static const int w_peers_minrows=[]{const char*e=std::getenv("TQR_W_PEERS_MINROWS");return e?std::atoi(e):8192;}();
  static const int w_peers_minrp=[]{const char*e=std::getenv("TQR_W_PEERS_MINROWS_PEER");return e?std::atoi(e):4096;}();
  // TQR_IEEE_W_PEERS=1 extends the concurrent vendor W peers to IEEE fp32 (default-math SGEMM, TF32 never; same
  // disjoint K_z, fixed-order combine, one commit). TF32/x3 keep GMMA W.
  static const bool ieee_w_peers=[]{const char*e=std::getenv("TQR_IEEE_W_PEERS");return e&&std::string(e)=="1";}();
  bool peers_precision=std::is_same_v<T,double>;
  if constexpr(std::is_same_v<T,float>)peers_precision=ieee_w_peers&&fp32_math()==Fp32Math::IEEE;
  const bool vendor_peers_w=peers_precision&&w_peers&&rep>1&&rows>=rep&&q>=w_peers_minq&&rep-1<=kPeerLanes&&
    rows>=w_peers_minrows&&rows/rep>=w_peers_minrp;
  const bool native_w=!vendor_peers_w&&native_w_path<T>(rep,rows);
  // A SKINNY native W (few output tiles -- the near window of a composed group, lookahead's window A, the
  // trailing edge) left most SMs idle: at fp64 n=65536 a q=128 W over 59k rows ran on 4 CTAs for 0.86 ms (2
  // TF) on the critical path between panels. The GPU level now cuts [K] = rows further into wcg groups,
  // exactly as compose_G already does: group g's partial goes to the slot's OWN W credit after the committed W
  // (so two strips in flight never share it), and the fixed-order combine g = 0..wcg-1 commits W once. The
  // executed product c is rep*wcg. Default ON. The strict gate still expects the schedule's single
  // apply_replication for every apply_W product and reports the difference. TQR_SKINNY_W_SPLIT=0 disables it.
  static const bool skinny_split=[]{const char*e=std::getenv("TQR_SKINNY_W_SPLIT");return !(e&&std::string(e)=="0");}();
  int wcg=1;
  if constexpr(native_wz_precision<T>()){
   if(skinny_split&&native_w&&!(std::is_same_v<T,float>&&gmma_enabled<T>())){
    wcg=carrier_w_gpu_split<T>(rep,q,h,rows,count);
    const size_t per=std::max<size_t>(1,size_t(count)*size_t(b)*size_t(q)),room=credit_words()/per;
    wcg=int(std::max<size_t>(1,std::min<size_t>(size_t(wcg),room>1?room-1:1)));}}
  if(rep>1)cw.product(CarrierWitness::APPLY_W,rep*wcg,count,native_w?'N':'L');
  else {cw.product_uncarried(CarrierWitness::APPLY_W,why,count);certify_apply(CarrierWitness::APPLY_W,why,rows,h,q,count);}
  // The GPU-level c=1 arm runs the NATIVE block-carried kernel. TF32 products on Hopper GMMA where the
  // shape admits the law (block 2 x GPU CG, rows in whole k-tile slices) and every operand is
  // TMA-addressable (16-byte base and leading dimensions). The same carrier in 3xTF32 (x3 math):
  // in-kernel split of each peer's K_z columns.
  bool gw=false,gw3=false,gw3rs=false;
  if constexpr(std::is_same_v<T,float>){
   if(native_w&&gmma_tf32<T>())gw=gmma_w_admits(rep,v,ldv,dst,ldx,w,b,rows,h,q,count,w+size_t(b)*q,(long long)b*q);
   else if(native_w&&gmma_x3<T>()){
    T*vs0=VR.n?VR.p+size_t(slot)*vr_slot_words():nullptr;
    if(vs0&&gmma_vsplit_words(rows,h)>3*size_t(plan.leaf)*vr_ld())vs0=nullptr;
    gw3rs=gmma_w3rs_admits(rep,v,ldv,dst,ldx,w,b,rows,h,q,count,vs0);
    if(!gw3rs)gw3=gmma_w3_admits(rep,v,ldv,dst,ldx,w,b,rows,h,q,count,w+size_t(b)*q,(long long)b*q);}}
  if(gw3rs){if constexpr(std::is_same_v<T,float>){
    T*vs=VR.p+size_t(slot)*vr_slot_words();
    if(vr_cache.size()>size_t(slot))vr_cache[slot].ok=false;   // the split overwrites the slot's VR region
    launch_gmma_w3rs(rep,v,ldv,dst,ldx,w,b,rows,h,q,sst,witness.p,CarrierWitness::APPLY_W,vs);
    // Executed decomposition: block 1 (warp groups split M), cluster 1, GPU c = rep contiguous row slices.
    cw.levels(CarrierWitness::APPLY_W,1,1,rep,1,count,true);cw.note_block_peers(CarrierWitness::APPLY_W,1,count);cw.note_membership(count);
    cw.block_carrier_geometry(CarrierWitness::APPLY_W,1,256,1,1,"y","rows",
     "K_0 = [K] within each of the c GPU-level contiguous row slices (the two warp groups split M, not K)",
     "one warp group per output element; the c GPU slices are summed by the fixed-order combine g=0..c-1",count);}}
  else if(gw||gw3){if constexpr(std::is_same_v<T,float>){
    if(gw)launch_gmma_w(rep,v,ldv,dst,ldx,w,b,rows,h,q,sst,witness.p,CarrierWitness::APPLY_W,w+size_t(b)*q,(long long)b*q);
    else{   // V_big / V_small in the slot's VR3 region (D packs VR3 only after W and Z, same stream)
     T*vs=VR.n?VR.p+size_t(slot)*vr_slot_words():nullptr;
     if(vs&&gmma_vsplit_words(rows,h)>3*size_t(plan.leaf)*vr_ld())vs=nullptr;
     if(ctx.size>1&&gmma_detail::multi_x3_in_kernel_w())vs=nullptr;
     // W3 stores its column-major high/low V split in the SAME slot that D3 uses for row-major VR3.
     // Its write invalidates any cached D packing even when V_g and its generation have not
     // changed.
     if(vs&&vr_cache.size()>size_t(slot))vr_cache[slot].ok=false;
     launch_gmma_w3(rep,v,ldv,dst,ldx,w,b,rows,h,q,sst,witness.p,CarrierWitness::APPLY_W,w+size_t(b)*q,(long long)b*q,vs);}
#ifdef TQR_MULTI_FIXED_DRIVER
    record_gmma_physical(CarrierWitness::APPLY_W,"rows",count,rep/2);
#else
    record_block_sliced(CarrierWitness::APPLY_W,2,gmma_wg_threads,"rows",count,32,16,1,rep/2);
#endif
   }}
  else if(native_w){
#ifdef TQR_MULTI_FIXED_DRIVER
   if(wcg==1&&launch_multi_w_mma8(rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,sst)){}
   else if(wcg==1&&launch_multi_kami_w(rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,sst)){}
   else
#endif
   {
   if(wcg>1)launch_carrier_w<T>(rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,sst,witness.p,CarrierWitness::APPLY_W,wcg,w+size_t(count)*b*q);
   else apply_stage_W<T>(bh,sst,rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,pp,&cw.remainder_peers,plan.wcb,witness.p);
   record_native_w(rep,q,count,CarrierWitness::APPLY_W,wcg);}}
  else if(vendor_peers_w)
  record_block_apply(CarrierWitness::APPLY_W,
   peer_stage_W<T>(bh,sst,rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,peer_lanes(slot),witness.p),
   "rows",count,rep);
  else
  record_block_apply(CarrierWitness::APPLY_W,
   apply_stage_W<T>(bh,sst,rep,v,ldv,sv,dst,ldx,sx,w,b,size_t(b)*q,rows,h,q,count,pp,&cw.remainder_peers,plan.wcb,witness.p),
   "rows",count,rep);
  int dwhy=-1;const int drep=carrier_c_d(h,count,&dwhy);
  const bool native_d=native_d_path<T>(drep,h);
  // The GMMA D reads Z column-major (K-major B) and V row-major (VR, packed per product below).
  T*vr=VR.n?VR.p+size_t(slot)*vr_slot_words():nullptr;
  // D runs the TF32 K-split kernel over the pre-split operands VR3 (rows x 3hs) and Z3 (3hs x q, after VR3
  // in the slot) at K' = 3hs, hs = h rounded up to whole k-tiles.
  bool gd=false;int hs=h,ldvr=int(vr_ld()),ldzg=b;T*zg=z;
  if constexpr(std::is_same_v<T,float>){
   if(gmma_x3<T>()){hs=gmma_x3_seg(h);ldvr=3*hs;ldzg=3*hs;zg=vr?vr+3*size_t(plan.leaf)*vr_ld():nullptr;
    gd=native_d&&vr&&size_t(hs)<=vr_ld()&&rows<=plan.leaf&&q<=plan.strip&&gmma_d_admits(vr,ldvr,zg,ldzg,dst,ldx,rows,3*hs,q,drep,count);}
   else gd=native_d&&vr&&gmma_tf32<T>()&&size_t(h)<=vr_ld()&&rows<=plan.leaf&&gmma_d_admits(vr,int(vr_ld()),z,b,dst,ldx,rows,h,q,drep,count);}
  const bool zt=native_d&&!gd;
  // Z = T^T W with ITS OWN carrier (Alg 2). [K] is the reflector index h.
  int zwhy=-1;const int zrep=carrier_c_z(h,count,&zwhy);
  const bool native_z=native_z_path_q<T>(zrep,h,q);
  if(zrep>1)cw.product(CarrierWitness::APPLY_Z,zrep,count,native_z?'N':'L');
  else {cw.product_uncarried(CarrierWitness::APPLY_Z,zwhy,count);certify_apply(CarrierWitness::APPLY_Z,zwhy,rows,h,q,count);}
  bool gz=false;
  if constexpr(std::is_same_v<T,float>)gz=native_z&&gmma_z_admits(t,b,w,b,z,b,h,q,zrep,count,transpose,zt);
  if(gz){if constexpr(std::is_same_v<T,float>){launch_gmma_z(t,b,w,b,z,b,h,q,sst,witness.p);
    record_block_sliced(CarrierWitness::APPLY_Z,2,gmma_wg_threads,"reflectors",count,32,16);}}
  else if(native_z){apply_stage_Z<T>(bh,sst,zrep,t,b,st,w,b,size_t(b)*q,z,zt?q:b,size_t(b)*q,h,q,count,transpose,pp,plan.zcb,witness.p,zt);record_native_z(zrep,count,h);}
  else
  record_block_apply(CarrierWitness::APPLY_Z,
   apply_stage_Z<T>(bh,sst,zrep,t,b,st,w,b,size_t(b)*q,z,zt?q:b,size_t(b)*q,h,q,count,transpose,pp,plan.zcb,witness.p,zt),
   "reflectors",count,zrep);
  // D = V Z with ITS OWN carrier (Alg 2, and Fig. carriers gives D (2,2,1)). [K] = h again. So this
  // cut costs no memory at all -- its whole cost is drep smaller GEMMs instead of one, which is
  // what the measurement weighs.
  const bool vendor_d=std::is_same_v<T,double>&&d_partials.n&&native_d&&zt&&drep==2&&count==1&&
                         h>=d_peers_minh()&&q>=1024&&rows<=plan.leaf&&size_t(rows)<=d_max_rows&&q<=plan.strip;
  if(drep>1)cw.product(CarrierWitness::APPLY_D,drep,count,(native_d&&!vendor_d)?'N':'L');
  else {cw.product_uncarried(CarrierWitness::APPLY_D,dwhy,count);certify_apply(CarrierWitness::APPLY_D,dwhy,rows,h,q,count);}
  if(vendor_d){
   d_products.push_back({{"rows",rows},{"h",h},{"q",q},{"slot",slot},
     {"K_ranges",{{0,h/2},{h/2,h}}},{"gpu_c",2},{"partial_count",2},
     {"owner_commits",1},{"internal_vendor_block_carrier","unobserved"}});
   peer_stage_D<T>(bh,sst,v,ldv,z,q,dst,ldx,rows,h,q,
                       d_partials.p+size_t(slot)*d_lane_words,peer_lanes(slot),witness.p);
   record_block_apply(CarrierWitness::APPLY_D,0,"reflectors",count,2);
   // The vendor peer internals remain unobserved. The native final commit is
   // ours and has a matching device counter; no native peer proof is invented.
   cw.native_commits_issued+=count;
  }else if(gd){if constexpr(std::is_same_v<T,float>){
   if(gmma_x3<T>()){
    if(vr_need_pack(slot,vr_key(v,ldv,rows,h,hs,true)))launch_pack_vr(v,ldv,rows,h,hs,vr,ldvr,true,sst);
    pack_z3<<<pack_vr_grid((long long)q*hs),256,0,sst>>>(z,b,h,hs,q,zg,ldzg);CU(cudaGetLastError());
    launch_gmma_d(vr,ldvr,zg,ldzg,dst,ldx,rows,3*hs,q,sst,witness.p);
   }else{
    if(vr_need_pack(slot,vr_key(v,ldv,rows,h,h,false)))launch_pack_vr(v,ldv,rows,h,h,vr,vr_ld(),false,sst);
    launch_gmma_d(vr,vr_ld(),z,b,dst,ldx,rows,h,q,sst,witness.p);}
#ifdef TQR_MULTI_FIXED_DRIVER
   record_gmma_physical(CarrierWitness::APPLY_D,"reflectors",count);
#else
   record_block_sliced(CarrierWitness::APPLY_D,2,gmma_wg_threads,"reflectors",count,32,16);
#endif
  }
  }else if(native_d){
   const int cexec=apply_stage_D<T>(bh,sst,drep,v,ldv,sv,z,q,size_t(b)*q,dst,ldx,sx,rows,h,q,count,plan.dcb,witness.p,true);
#ifdef TQR_MULTI_FIXED_DRIVER
   if(kami::last_dnp_threads())record_kami_physical(kami::last_dnp_threads(),count);
   else
#endif
   if(const int dcl=last_d_cluster();dcl>1){
    // Device and host both report (block 1, cluster c).
    cw.levels(CarrierWitness::APPLY_D,1,dcl,1,1,count,true);
    cw.note_block_peers(CarrierWitness::APPLY_D,1,count);cw.note_membership(count);cw.native_commits_issued+=count;
    cw.block_carrier_geometry(CarrierWitness::APPLY_D,1,128,1,1,"serial","reflectors",
     "cluster CTA z owns the balanced contraction range K_z (units of 2 BK); the CTA's warps split rows x columns only",
     "register partials of the peer's column half pushed through distributed shared memory (st.async + mbarrier), summed in fixed order P_0+P_1 by the owner CTA; single commit",count);
   }
   else if(short_carrier_path<T>(cexec,h,true))record_block_apply(CarrierWitness::APPLY_D,cexec,"reflectors",count);
   else record_block_sliced(CarrierWitness::APPLY_D,cexec,carrier_d_slice_threads<T>(cexec,rows,q,count),"reflectors",count);
  }else
  record_block_apply(CarrierWitness::APPLY_D,
   apply_stage_D<T>(bh,sst,drep,v,ldv,sv,z,b,size_t(b)*q,dst,ldx,sx,rows,h,q,count,plan.dcb,witness.p),
   "reflectors",count,drep);
 }
 // `why` receives the CarrierWitness reason on a refusal. Every refusal runs the native kernels
 // (c=1 regime), never cuBLAS.
 int carrier_c(int rows,int h,int q,int count,int*why=nullptr)const{
  auto no=[&](int r){if(why)*why=r;return 1;};
  if(plan.apply_c<2)return no(CarrierWitness::R_APPLY_C_LE_1);
  // [K] IS THE ROW INDEX, so a nonempty balanced cut needs rows>=c, not rows>=c*h.
  if(rows<plan.apply_c)return no(CarrierWitness::R_ROWS_LT_C);
  // The native W is the apply_c>1 arm; rows that cannot give every cluster CTA a full k-tile are an
  // implementation gap (certified), never a silent switch to library peers.
  if(native_wz_enabled<T>()&&!carrier_w_supported(plan.apply_c,rows))return no(CarrierWitness::R_W_SLICE_MIN_ROWS);
  if(count>int(std::max<size_t>(plan.max_batch,1)))return no(CarrierWitness::R_COUNT_NE_1);
  return plan.apply_c;
 }
 // The merge path's staging block, one per pipeline slot: two strips of one
 // merge elimination are disjoint in X but were sharing this buffer, which
 // is the only thing that kept the tree-level apply on a single lane.
 size_t x_slot_words()const{return std::max<size_t>(plan.max_batch,1)*size_t(plan.radix)*size_t(plan.b)*size_t(plan.strip);}
 size_t credit_words()const{return size_t(std::max(1,plan.apply_c))*std::max<size_t>(plan.max_batch,1)*size_t(plan.b)*agg*size_t(plan.strip);}
 // Three device pointers per peer, max_batch*max(apply_c,zc) peers, per
 // pipeline slot. A slot's stream is ordered, so a region is safe to refill
 // for the next product on that slot.
 size_t carrier_ptr_slot()const{return size_t(3)*std::max<size_t>(plan.max_batch,1)*size_t(std::max({1,plan.apply_c,plan.zc}));}
 // Z holds zc partials side by side, exactly as W holds apply_c of them.
 size_t z_credit_words()const{return size_t(std::max(1,plan.zc))*std::max<size_t>(plan.max_batch,1)*size_t(plan.b)*agg*size_t(plan.strip);}
 // Z's carrier: [K] is the reflector index, so it is h that must admit the cut.
 int carrier_c_z(int h,int count,int*why=nullptr)const{
  auto no=[&](int r){if(why)*why=r;return 1;};
  if(plan.zc<2)return no(CarrierWitness::R_ZC_LE_1);
  if(h<plan.zc)return no(CarrierWitness::R_H_LT_CZ);
  if(native_wz_enabled<T>()&&!native_z_path<T>(plan.zc,h))return no(CarrierWitness::R_Z_SLICE_MIN_H);
  if(count>int(std::max<size_t>(plan.max_batch,1)))return no(CarrierWitness::R_COUNT_NE_1);
  return plan.zc;
 }
 // D's carrier accumulates in place, so it needs no pointer region and no
 // capacity beyond X itself; only h has to admit the cut.
 int carrier_c_d(int h,int count,int*why=nullptr)const{
  auto no=[&](int r){if(why)*why=r;return 1;};
  if(plan.dc<2)return no(CarrierWitness::R_DC_LE_1);
  if(h<plan.dc)return no(CarrierWitness::R_H_LT_CD);
  // where the native carried D exists (fp64) it IS the dc>1 arm; a short h that cannot give every
  // warp slice a non-empty K part is an implementation gap (certified 'unimplemented'), never a
  // silent switch to the sequential library passes.
  if(native_d_precision<T>()&&!native_d_path<T>(plan.dc,h))return no(CarrierWitness::R_D_SLICE_MIN_H);
  return plan.dc;
 }
 cudaStream_t slot_stream(int slot){return slot?pipe_stream[slot]->s:stream.s;}
 // V is staged per panel by pack_ge_v and per level by tile_merge_v, and both
 // the factor of panel k+1 and the trailing update of panel k want it at once
 // once those two overlap. `vbase` selects the half by panel parity; it is 0
 // whenever the overlap is not in use, which is bit-for-bit the old buffer.
 size_t vbase=0,vhalf=0;
 T* vbuf()const{return V.p+vbase;}
 // Credits are issued before the first strip (so every slot sees V and T published) and redeemed
 // after the last (so the caller sees every strip). `base` is the lane the group forks from and
 // joins back to: lane 0 is the main stream, and a second group based on lane d is what carries the
 // bulk trailing update while the main stream factors the next panel.
 void pipeline_fork(int d,int base=0){if(d<2)return;pipe_fork.record(slot_stream(base));for(int i=1;i<d;++i)pipe_fork.wait(slot_stream(base+i));}
 void pipeline_join(int d,int base=0){if(d<2)return;for(int i=1;i<d;++i){pipe_event[base+i]->record(slot_stream(base+i));pipe_event[base+i]->wait(slot_stream(base));}}
   // issued != nullptr: report {packets, trailing columns} instead of recording the elimination, so
   // the caller records it once with the composed far strips included.
   void ge_batch(const TileBatch& batch,TiledMatrix<T>&x,bool factor,bool transpose,int lo=-1,int hi=-1,bool emit_factor=true,int lane0=0,int*issued=nullptr){
   int count=batch.ge.size();if(!count)return;int b=plan.b,leaf=plan.leaf;select_packets(batch.ge.front().tile);
   // C2: coop_packed_v ("vbuf stages the current batch's complete V") is set
   // by the window loop of the factor call that just ran. It must never leak
   // into another batch's call: factor=false Q-application for other panels
   // skipped their mandatory V pack on a stale true and applied the wrong
   // panel's V (two-panel-256-sw64 resid 2.156 with correct stored V/R).
   // Reset at entry; the factor branch below re-arms it when it truly packed.
   coop_packed_v=false;
   if(factor&&emit_factor){
    // Fused history publication: full leaves publish their entries from inside hh's epilogue (same
    // CAS as history_start), ordered after the factor writes, saving one launch per GE batch.
    // Ragged short leaves keep the explicit start AFTER their extra-R product, since publication
    // must follow the R transformation (see comment below).
    bool full=batch.ge.back().h==batch.h;
    // The guard is spelled out one condition at a time so that a fall-back onto the block path NAMES
    // its cause.
    int why=-1;
    if(!full)why=CarrierWitness::R_PANEL_NOT_FULL;
    else if(count!=1)why=CarrierWitness::R_PANEL_MULTI;
    else if(batch.cc<1)why=CarrierWitness::R_PANEL_CC_ZERO;
    else if(batch.h>plan.b)why=CarrierWitness::R_PANEL_H_NE_B;
    else if(plan.b>coop_max_b&&!multi_wide_panel())why=CarrierWitness::R_PANEL_B_MAX;
    // Reset per panel: the flag is set by the window loop below, and a
    // block-path panel (which never enters that function) must never inherit
    // the previous panel's true.
    coop_packed_v=false;
    // Alg 1's last line ("prioritize applies releasing panel k+1"): the factor is bracketed so the
    // host can say, from the device clock, how long after the previous panel's trailing finished
    // this factor actually started.
    const unsigned pev=exec_begin(lane0,EXEC_PANEL_FACTOR,0,slot_stream(lane0));
    // GPU level = J split over owners (p_j) with the reflector operand replicated, contraction c = 1 at that level
    // (reported as such); block level c = threads.
    const int gau=full?gau_ge_batch(x,batch,lane0):0;
    if(gau>0){why=-1;}
    int coop=0;
    if(!gau&&multi_wide_panel()&&count==1&&batch.ge[0].h>128){
     TileBatch actual=batch;actual.h=batch.ge[0].h;
     coop=cooperative_ge_batch(x,actual,count,lane0);
     if(coop>0)why=-1;
     // A ragged leaf has fewer V columns than the full panel. Repack the
     // complete padded V after its last window so unused columns are zero.
     if(actual.h!=batch.h)coop_packed_v=false;
    }else if(!gau&&why<0)coop=cooperative_ge_batch(x,batch,count,lane0);
    if(gau==0&&why<0&&!coop)why=CarrierWitness::R_PANEL_LAUNCH;
    if(gau>0)cw.product(CarrierWitness::PANEL_GE,1,count,'N');
    else if(coop>0)cw.product(CarrierWitness::PANEL_GE,coop,count,'N');
    else {cw.product_uncarried(CarrierWitness::PANEL_GE,why,count);certify_panel(why,batch,count);}
    if((coop||gau)&&full)history_start(count,batch.col+batch.h);
    else if(!coop&&!gau&&full){ hh(x.a.p,x.ld,count,0,plan.ge_shared,size_t(batch.ge[0].rows)*batch.h*sizeof(T),hist.p,batch.col+batch.h); }
    else if(!coop&&!gau)hh(x.a.p,x.ld,count,0,plan.ge_shared,size_t(batch.ge[0].rows)*batch.h*sizeof(T));
    exec_end(pev,slot_stream(lane0));
   }
  // V is produced by the FACTOR, so only the call that factored needs to pack it. Alg 1's
  // cross-panel split calls ge_batch three times per panel (factor with an empty window, window A,
  // window B) and the last two re-packed a V that had not changed: 256 -> 762 launches and 26.35 ->
  // 154.03 ms at fp64 n=16384. The factor==false calls are the Q-application passes, which pack
  // from the stored A and still must. Skipped when the cooperative window loop already packed every
  // V column (coop_packed_v): the bytes are identical, recomputed from the same A.
  if((!factor||emit_factor)&&!coop_packed_v)
   pack_ge_v<<<dim3(pack_v_grid(leaf,b),count),128,0,slot_stream(lane0)>>>(factor_A->a.p,factor_A->ld,packets,count,vbuf(),leaf,b);
  // A leaf shorter than the panel has fewer reflectors, but its remaining
  // panel columns still need that GE transformation before the R-stack merge.
  // They are R entries, not V backing; apply them before publishing history.
   if(factor&&emit_factor){auto&last=batch.ge.back();for(int col=batch.col+last.h;col<batch.col+batch.h;){int q=std::min(plan.strip,batch.col+batch.h-col);
     const unsigned rev=exec_begin(lane0,EXEC_PACKET,0,slot_stream(lane0));
     product(x.a.p+last.row+size_t(col)*x.ld,x.ld,leaf,vbuf()+size_t(count-1)*leaf*b,leaf,size_t(leaf)*b,tri.p+size_t(last.tile)*b*b,size_t(b)*b,last.rows,last.h,q,1,true,lane0);
     exec_end(rev,slot_stream(lane0));col+=q;}if(last.h!=batch.h)history_start(count,batch.col+batch.h);}
  int start=factor?batch.col+batch.h:0;if(lo>start)start=lo;const int stop=hi<0?x.n:std::min(hi,x.n);
  // Ordered publication across slots: tile_history_commit CAS-checks the strip cursor, so commit k
  // must not execute before commit k-1 even though the strips run concurrently. One event per strip
  // chains the commits in issue order; the W/Z/D products still overlap and only the cursor CAS
  // serializes. Without the chain, slot k+1's commit can win the race and flag HISTORY_ERROR (seen
  // at depth=2 in the carrier suite's forced (2,1) cell).
  const int depth=depth_cap?depth_cap:std::max(1,plan.d);int packet=0;
  // STAMP GRANULARITY. A packet interval exists to witness two packets of one elimination
  // overlapping. At depth 1 the elimination is bracketed once instead: the device product counts
  // and the closed interval survive, and only a resolution that could not have shown anything is
  // lost.
  const bool per_packet=depth>1;
  pipeline_fork(depth,lane0);
  const unsigned elim_ev=per_packet?~0u:exec_begin(lane0,EXEC_PACKET,0,slot_stream(lane0));
  std::vector<std::unique_ptr<Event>> commit_done, slot_free;
  // Buffer credits: slot s owns one W/Z credit, so strip k on slot s must wait for slot s's
  // previous strip before overwriting its W/Z. Without this, strip k+d clobbers the credit strip
  // k's D product still reads: every depth>1 carrier cell came back with residual ~1.
  for(int i=0;i<depth;++i){slot_free.push_back(std::make_unique<Event>());slot_free.back()->record(slot_stream(lane0+i));}
  for(int col=start;col<stop;){const int slot=lane0+packet++%depth;int q=std::min(plan.strip,stop-col),full=count-(batch.ge.back().rows<leaf);
   slot_free[slot-lane0]->wait(slot_stream(slot));
   const unsigned ev=per_packet?exec_begin(slot,EXEC_PACKET,(unsigned)(packet-1),slot_stream(slot)):~0u;
   if(full)product(x.a.p+batch.ge[0].row+size_t(col)*x.ld,x.ld,leaf,vbuf(),leaf,size_t(leaf)*b,tri.p+size_t(batch.ge[0].tile)*b*b,size_t(b)*b,leaf,batch.h,q,full,transpose,slot);
   if(full<count){auto&last=batch.ge.back();product(x.a.p+last.row+size_t(col)*x.ld,x.ld,leaf,vbuf()+size_t(full)*leaf*b,leaf,size_t(leaf)*b,tri.p+size_t(last.tile)*b*b,size_t(b)*b,last.rows,batch.h,q,1,transpose,slot);}
   if(factor){if(!commit_done.empty())commit_done.back()->wait(slot_stream(slot));history_advance(count,col,col+q,slot_stream(slot));commit_done.push_back(std::make_unique<Event>());commit_done.back()->record(slot_stream(slot));}
   if(per_packet)exec_end(ev,slot_stream(slot));
   slot_free[slot-lane0]->record(slot_stream(slot));
   col+=q;
  }
  if(!per_packet)exec_end(elim_ev,slot_stream(lane0));
  if(issued){issued[0]=packet;issued[1]=std::max(0,stop-start);}
  else{
  cw.elimination(packet,std::min(depth,packet),count,depth==1?"selected_depth_one":"single_packet_tail");
  certify_elimination(packet,std::min(depth,packet),stop-start,"leaf trailing apply",count);}
  ++exec_elimination;
  pipeline_join(depth,lane0);
 }
  // The far columns [lo,hi) receive ONE Apply of the composed transform of the g consecutive eliminations k..k+g-1
  // (Q_k..Q_{k+g-1} = I - V_g T_g V_g^T, V_g = [V_k .. V_{k+g-1}], T_g by Eq compose, one constituent at a time)
  // instead of g Applies. The elimination list, every Eliminate call and the near columns (panels k+1..k+g-1, which
  // get every elimination individually) are unchanged. W, Z, D of the composed Apply run the ordinary carriers at
  // contraction h = g b; D's c layers are unions of whole constituents (layer order below); A lane owns its stream,
  // cuBLAS handle, pointer slot and GPU-partial buffer (GPART on the main stream, GPART_N on the near slot: window
  // boundaries and QR-v2 overlaps keep using GPART on the main stream).
  struct ComposeLane{cudaStream_t st;cublasHandle_t bh;T*part;T**pp;};
  ComposeLane compose_lane(int lane){return {slot_stream(lane),lane?pipe_blas[lane]:blas,(lane&&GPART_N.n)?GPART_N.p:GPART.p,carrier_ptr.p+size_t(lane)*carrier_ptr_slot()};}
  void composed_check(const std::vector<const TileBatch*>&grp)const{
   const int b=plan.b,leaf=plan.leaf,g=int(grp.size());const TilePacket&pk=grp[0]->ge[0];
   for(int j=1;j<g;++j){const TilePacket&pj=grp[j]->ge[0];
    if(pj.row!=pk.row+j*b||pj.col!=pk.col+j*b||pk.rows!=pj.rows+j*b||pj.h!=b)throw std::runtime_error("p7_composed_geometry");}
   if(pk.h!=b||pk.rows>leaf||g<2||g>agg)throw std::runtime_error("p7_composed_geometry");}
  void composed_begin(int g,const ComposeLane&L){const int H=g*plan.b;CU(cudaMemsetAsync(TGp(),0,size_t(H)*H*sizeof(T),L.st));}
  // Eq compose, one constituent at a time: T_{1..j+1} = [[T_{1..j}, -T_{1..j} G_j T_{j+1}],[0,T_{j+1}]],
  // G_j = V_{1..j}^T V_{j+1}, block i = V_{k+i}^T V_{k+j} over the rows where V_{k+j} is nonzero (natural
  // per-panel V halves, packed by each factor). Each block is a compose_G product with its own carrier:
  // the native W_ONLY kernel (cluster x warp slices over the rows, DSM fixed-order combine, device kind
  // compose_G); IEEE fp32 the SIMT W carrier or the c>1 library intermediary.
  int composed_step(const TileBatch&first,const TileBatch&bj,int g,int j,const ComposeLane&L){
   const int b=plan.b,leaf=plan.leaf,H=g*b,rows=first.ge[0].rows;cudaStream_t st=L.st;int composes=0;
   for(int i=0;i<j;++i){const int kr=rows-j*b;const T*vi=V.p+size_t(i)*vhalf+size_t(j-i)*b,*vj=V.p+size_t(j)*vhalf;T*gout=AGG_G.p+size_t(i)*b*b;   // block i of G_j, b x b, ld b (library partials land in later blocks' space, computed after)
    int why=-1;const int rep=carrier_c(kr,b,b,1,&why);T**pp=L.pp;++composes;
    bool gg=false;const int ggroups=gmma_detail::gmma_g_groups();
    if constexpr(std::is_same_v<T,float>)gg=rep>1&&gmma_g_admits(ggroups,vi,leaf,vj,leaf,gout,b,kr,b,b,L.part,GPART.n);
    if(gg){if constexpr(std::is_same_v<T,float>){
     launch_gmma_g(ggroups,vi,leaf,vj,leaf,gout,b,kr,b,b,L.part,st,witness.p);
     cw.product(CarrierWitness::COMPOSE_G,2*ggroups,1,'N');record_block_sliced(CarrierWitness::COMPOSE_G,2,gmma_wg_threads,"rows",1,32,16,1,ggroups);}}
    else if(rep>1&&native_w_path<T>(rep,kr)){
     const int gcg=carrier_w_gpu_split<T>(rep,b,b,kr,1);
     launch_carrier_w<T>(rep,vi,leaf,0,vj,leaf,0,gout,b,size_t(b)*b,kr,b,b,1,st,witness.p,CarrierWitness::COMPOSE_G,gcg,L.part);
     cw.product(CarrierWitness::COMPOSE_G,rep*gcg,1,'N');record_native_w(rep,b,1,CarrierWitness::COMPOSE_G,gcg);}
    else if(rep>1&&ieee_simt_w(rep,vi,leaf,0,vj,leaf,0,gout,b,(long long)b*b,kr,b,b,1,st,CarrierWitness::COMPOSE_G,L.part)){}
    else if(rep>1){cw.product(CarrierWitness::COMPOSE_G,rep,1,'L');
     record_block_apply(CarrierWitness::COMPOSE_G,apply_stage_W<T>(L.bh,st,rep,vi,leaf,0,vj,leaf,0,gout,b,size_t(b)*b,kr,b,b,1,pp,&cw.remainder_peers,plan.wcb,witness.p),"rows",1,rep);}
    else{tile_gemm<T>(L.bh,CUBLAS_OP_T,b,b,kr,T(1),vi,leaf,0,vj,leaf,0,T(0),gout,b,0,1);
     cw.product_uncarried(CarrierWitness::COMPOSE_G,why);
     cw.native_partials[CarrierWitness::COMPOSE_G]-=1;cw.library_partials[CarrierWitness::COMPOSE_G]+=1;
     cw.levels(CarrierWitness::COMPOSE_G,0,1,1,1,1,false);certify_apply(CarrierWitness::COMPOSE_G,why,kr,b,b,1);}}
   // tiled compose (TQR_COMPOSE_TILED, default on; 0 = the per-column compose_tg_step).
   static const bool tiled=[]{const char*e=std::getenv("TQR_COMPOSE_TILED");return !(e&&std::string(e)=="0");}();
   const T*Tn=tri.p+size_t(bj.ge[0].tile)*b*b;
   if(tiled&&j>0&&b%kComposeTile==0&&AGG_S.n>=size_t(j)*b*b){const int jb=j*b;const dim3 grid(ceildiv(jb,kComposeTile),b/kComposeTile),blk(kComposeTile,8);
    compose_s_tiled<T><<<grid,blk,0,st>>>(AGG_G.p,b,jb,Tn,AGG_S.p);
    compose_n_tiled<T><<<grid,blk,0,st>>>(TGp(),H,jb,b,AGG_S.p);
    compose_diag_copy<T><<<ceildiv(b*b,256),256,0,st>>>(TGp(),H,j,b,Tn,witness.p);CU(cudaGetLastError());++compose_tiled_steps;}
   else compose_tg_step<T><<<b,128,size_t(std::max(1,j)*b)*sizeof(T),st>>>(TGp(),H,j,b,AGG_G.p,Tn,witness.p);
   return composes;}
  // D-5 (ii) layer order. The D carrier that runs (dc, law BK/WK) gives warp slice
  // s = floor((p mod BK)/WK) the composed positions p. Position p holds natural reflector
  // s*(H/dc) + floor(p/BK)*WK + p mod WK, so slice s contracts the contiguous natural range
  // [s H/dc, (s+1) H/dc): exactly dc/g slices per constituent (dc >= g) or g/dc whole constituents
  // per slice (dc < g). V_g is packed in this order and T_g permuted symmetrically (P^T T_g P).
  // Permutations for every g' <= agg were uploaded at setup (p7_setup_layers): no host sync here.
  void composed_end(const TileBatch&first,int g,int composes,const ComposeLane&L){
   const int b=plan.b,leaf=plan.leaf,H=g*b;const TilePacket&pk=first.ge[0];
   const int*perm=AGG_PERM.p+size_t(agg_perm_off[g]);const bool layered=agg_layered[g];
   ++vg_epoch[agg_gen&1];   // V_g generation agg_gen is rewritten -> every cached VR of it is stale
   static const int pack_blocks=[]{const char*e=std::getenv("TQR_COMPOSE_PACK_BLOCKS");const int n=e?std::stoi(e):256;
    if(n<32||n>4096)throw std::runtime_error("invalid_compose_pack_blocks");return n;}();
   pack_v_block_perm<<<pack_blocks,256,0,L.st>>>(factor_A->a.p,factor_A->ld,pk.row,pk.col,pk.rows,H,perm,VGp(),leaf);
   permute_t<T><<<64,256,0,L.st>>>(TGp(),H,perm,TGPp());
   cw.aggregated_applies+=1;cw.aggregate_g[g]+=1;cw.aggregated_compose_G+=composes;
   cw.aggregated_constituent_layers+=layered;}
  void composed_prepare(const std::vector<const TileBatch*>&grp){
   composed_check(grp);const int g=int(grp.size());const ComposeLane L=compose_lane(0);
   composed_begin(g,L);int composes=0;
   for(int j=0;j<g;++j)composes+=composed_step(*grp[0],*grp[j],g,j,L);
   composed_end(*grp[0],g,composes,L);
  }
  // The composed Apply over the far column window [lo,hi) on the lane group based at lane0 (0: the main group; d:
  // lookahead's window-B group). V_g/T_g come from composed_prepare of the SAME group and must not be rebuilt until
  // every window of it retired (the caller waits trail_done first). after: the event recorded after the SAME group's
  // window A; the first strip's history commit waits on it, so the per-constituent strip cursor still advances in
  // column order while window B's W/Z/D products run beside window A.
  int composed_apply(const std::vector<const TileBatch*>&grp,TiledMatrix<T>&x,int lo,int hi,int lane0,Event*after=nullptr){
   const int leaf=plan.leaf,g=int(grp.size()),H=g*plan.b;const TilePacket&pk=grp[0]->ge[0];const int rows=pk.rows;
   const int depth=depth_cap?depth_cap:std::max(1,plan.d);int packet=0;const bool per_packet=depth>1;
   pipeline_fork(depth,lane0);
   const unsigned elim_ev=per_packet?~0u:exec_begin(lane0,EXEC_PACKET,0,slot_stream(lane0));
   std::vector<std::unique_ptr<Event>> commit_done,slot_free;
   for(int i=0;i<depth;++i){slot_free.push_back(std::make_unique<Event>());slot_free.back()->record(slot_stream(lane0+i));}
   for(int col=lo;col<hi;){const int slot=lane0+packet++%depth;const int q=std::min(plan.strip,hi-col);
    slot_free[slot-lane0]->wait(slot_stream(slot));
    const unsigned ev=per_packet?exec_begin(slot,EXEC_PACKET,(unsigned)(packet-1),slot_stream(slot)):~0u;
    product(x.a.p+pk.row+size_t(col)*x.ld,x.ld,0,VGp(),leaf,0,TGPp(),0,rows,H,q,1,true,slot);
    if(!commit_done.empty())commit_done.back()->wait(slot_stream(slot));else if(after)after->wait(slot_stream(slot));
    for(int j=0;j<g;++j){select_packets(grp[j]->ge[0].tile);history_advance(1,col,col+q,slot_stream(slot));}
    commit_done.push_back(std::make_unique<Event>());commit_done.back()->record(slot_stream(slot));
    if(per_packet)exec_end(ev,slot_stream(slot));
    slot_free[slot-lane0]->record(slot_stream(slot));col+=q;}
   if(!per_packet)exec_end(elim_ev,slot_stream(lane0));
   pipeline_join(depth,lane0);
   cw.aggregated_strips+=packet;
   return packet;
  }
  // Largest group size g' in [2, agg] starting at panel pi whose members are all simple one-leaf
  // local panels of full height on consecutive rows, with columns left beyond the group.
  int p7_group(size_t pi,const TiledMatrix<T>&a)const{
   if(agg<2)return 1;int g=1;
   const auto&b0=plan.panels[pi].ranks[ctx.rank];
   auto simple=[&](size_t k){const auto&pn=plan.panels[k];const auto&bt=pn.ranks[ctx.rank];
    return pn.global.empty()&&bt.levels.empty()&&bt.ge.size()==1&&bt.h==plan.b&&bt.ge[0].h==plan.b;};
   if(!simple(pi)||b0.ge[0].rows>plan.leaf)return 1;
   for(size_t k=pi+1;k<plan.panels.size()&&g<agg;++k){
    const auto&bk=plan.panels[k].ranks[ctx.rank];const int j=int(k-pi);
    if(!simple(k)||bk.ge[0].row!=b0.ge[0].row+j*plan.b||b0.ge[0].rows!=bk.ge[0].rows+j*plan.b)break;
    g=j+1;}
   while(g>1){const auto&pl=plan.panels[pi+g-1];if(pl.col+pl.h<a.n)break;--g;}
   // The default keeps whole-constituent layers. The explicitly recorded
   // balanced_reflectors policy instead allows cuts within a constituent.
   const int dc=std::max(1,plan.dc);
   auto layer_ok=[&](int gg){if(dc<2||!native_d_path<T>(dc,gg*plan.b))return true;int bk,wk;carrier_d_law<T>(dc,bk,wk);
    return bk==dc*wk&&(gg*plan.b)%bk==0&&(balanced_aggregate_layers()||(dc>=gg?dc%gg==0:gg%dc==0));};
   while(g>1&&!layer_ok(g))--g;
   return g;
  }
  // Structured pentagonal merge apply (merge_apply.cuh). V = [I;V_1;..] is known here, so Eq apply
  // runs on the support rows in place and the dense (radix*b x q) staging block -- with its gather
  // and scatter round trips -- never exists. NREG is the column blocking. Its optimum is set by how
  // many CTAs the launch produces, not by q alone: NREG is the largest value that still fills the
  // machine with about two waves of 132 SMs, and 4 when nothing does. The grid is NOT capped: a
  // capped grid makes the last grid-stride sweep ragged, which is what made q=16384 prefer a smaller
  // NREG before.
  static constexpr int merge_apply_max_b=TQR_MERGE_APPLY_MAX_B;
  static int merge_apply_nreg(int q,int count,int yc){
   for(int n:{16,8})if((long long)ceildiv(q,yc*n)*count>=256)return n;
   return 4;
  }
  bool merge_apply_usable()const{
   if(plan.b>merge_apply_max_b||256%plan.b)return false;
   return merge_apply_shared(plan.b,256/plan.b,16,sizeof(T))<=size_t(48)*1024;
  }
  // `lane` selects the pipeline slot -- and therefore the stream, the cuBLAS handle and the W/Z
  // credit -- that this window runs on. Lane 0 is the main stream. A window issued on a non-zero
  // lane runs beside the main stream, which is how the bulk trailing update of panel k overlaps the
  // factor of panel k+1 (Alg 1: "prioritize applies releasing panel k+1"). Multi-GPU tree packing:
  // the old fixed 32-CTA grid leaves most SMs idle when only one or two merge nodes remain. Change
  // only the copy grid, never row ownership, numerical arithmetic, carrier partitions or stream
  // order.
  int multi_copy_grid(int rows,int columns,int legacy)const{
   if(ctx.size<2)return legacy;
   static const int requested=[](){const char*v=std::getenv("TQR_MULTI_COPY_CTAS");
    if(!v)return 0;const int n=std::stoi(v);
    if(n!=0&&n!=64&&n!=128&&n!=256&&n!=512)throw std::runtime_error("invalid_multi_copy_ctas");
    return n;}();
   if(!requested)return legacy;
   return std::max(1,std::min(requested,ceildiv(rows*columns,128)));
  }
  void local_trailing(const std::vector<TileMerge>&es,TiledMatrix<T>&x,bool factor,bool transpose,int lo=-1,int hi=-1,int lane=0){
   int count=es.size(),b=plan.b;if(!count)return;select_packets(es.front().tile);
   const int gsmall=4*plan.radix,gwide=16*plan.radix;   // 8 and 32 at radix 2
   const bool fused=merge_apply_usable();
   cudaStream_t st=slot_stream(lane);
   // V.p is shared by the leaf pack and every merge level, so each pass over a
   // column window re-stages it. Both kernels read the reflectors out of A,
   // which the applies never touch, so re-staging is idempotent.
   if(!fused)tile_merge_v<<<dim3(gsmall,count),128,0,st>>>(factor_A->a.p,factor_A->ld,factor_A->begin,merges,count,vbuf(),b,plan.radix);
   int start=factor?es[0].col+es[0].h:0;if(lo>start)start=lo;const int stop=hi<0?x.n:std::min(hi,x.n);
   const int mb=plan.radix*b,yc=256/b;
   // THE TREE-LEVEL APPLY TAKES THE DEPTH TOO. The strips of a merge are as independent as the
   // strips of a leaf: they touch disjoint columns of X. What was genuinely shared is the staging
   // block of the non-fused path, so each slot now stages into its OWN X region, exactly as it
   // already owns its own W and Z credits.
   const int depth=std::max(1,plan.d);int packet=0;
   const bool per_packet=depth>1;          // see the leaf loop: no overlap at depth 1
   pipeline_fork(depth,lane);
   const unsigned elim_ev=per_packet?~0u:exec_begin(lane,EXEC_PACKET,0,st);
   // At depth 1 there is one lane, which is ordered by construction, so no credit event is created
   // and the schedule is bit-for-bit what it was before the tree-level apply took the depth.
   std::vector<std::unique_ptr<Event>> commit_done, slot_free;
   if(depth>1)for(int i=0;i<depth;++i){slot_free.push_back(std::make_unique<Event>());slot_free.back()->record(slot_stream(lane+i));}
   for(int col=start;col<stop;){const int slot=lane+packet++%depth;int q=std::min(plan.strip,stop-col);
    cudaStream_t ss=slot_stream(slot);
    if(depth>1)slot_free[slot-lane]->wait(ss);
    const unsigned ev=per_packet?exec_begin(slot,EXEC_PACKET,(unsigned)(packet-1),ss):~0u;
    if(fused){
     // The fused pentagonal apply performs W, Z and D inside one kernel over the support rows, so
     // all three run unreplicated by construction. Counted with that reason so the apply
     // denominators stay complete and the trade-off (one kernel, no carrier) is visible rather than
     // implied. Each thread computes one output row and sums ALL s serially. The block's b output
     // threads are not b contraction peers.
     for(int kind:{CarrierWitness::APPLY_W,CarrierWitness::APPLY_Z,CarrierWitness::APPLY_D})
      cw.block_carrier_geometry(kind,1,b,yc,1,"serial","reflectors",
       "K_0 = [K]; each output thread sums the full contraction",
       "serial register accumulation; no cross-thread partials",count);
     cw.product_uncarried(CarrierWitness::APPLY_W,CarrierWitness::R_FUSED_MERGE,count);cw.levels(CarrierWitness::APPLY_W,0,1,1,1,count,false);
     cw.product_uncarried(CarrierWitness::APPLY_Z,CarrierWitness::R_FUSED_MERGE,count);cw.levels(CarrierWitness::APPLY_Z,0,1,1,1,count,false);
     cw.product_uncarried(CarrierWitness::APPLY_D,CarrierWitness::R_FUSED_MERGE,count);cw.levels(CarrierWitness::APPLY_D,0,1,1,1,count,false);
     for(int k:{CarrierWitness::APPLY_W,CarrierWitness::APPLY_Z,CarrierWitness::APPLY_D})
      certify_apply(k,CarrierWitness::R_FUSED_MERGE,mb,b,q,count);
     const int nreg=merge_apply_nreg(q,count,yc),per=yc*nreg;
     const size_t shared=merge_apply_shared(b,yc,nreg,sizeof(T));
     const int gx=std::max(1,(q+per-1)/per);
     auto go=[&](auto tag){constexpr int R=decltype(tag)::value;
      tile_merge_apply<T,R><<<dim3(gx,count),dim3(b,yc),shared,ss>>>(
        x.a.p,x.ld,x.begin,factor_A->a.p,factor_A->ld,factor_A->begin,
        merges,count,tri.p+size_t(es[0].tile)*b*b,size_t(b)*b,b,col,q,plan.radix,transpose);};
     if(nreg==4)go(std::integral_constant<int,4>{});
     else if(nreg==8)go(std::integral_constant<int,8>{});
     else go(std::integral_constant<int,16>{});
    }else{
     T*xs=X.p+size_t(slot)*x_slot_words();
     tile_merge_x<T,false><<<dim3(multi_copy_grid(mb,q,gwide),count),128,0,ss>>>(x.a.p,x.ld,x.begin,merges,count,xs,b,col,q,plan.radix);
     product(xs,mb,size_t(mb)*q,vbuf(),mb,size_t(mb)*b,tri.p+size_t(es[0].tile)*b*b,size_t(b)*b,mb,es[0].h,q,count,transpose,slot);
     tile_merge_x<T,true><<<dim3(multi_copy_grid(mb,q,gwide),count),128,0,ss>>>(x.a.p,x.ld,x.begin,merges,count,xs,b,col,q,plan.radix);
    }
    // Ordered publication across slots, exactly as the leaf loop chains it:
    // tile_history_commit CAS-checks the strip cursor, so commit k must not
    // execute before commit k-1 even though the strips run concurrently.
    if(factor){if(depth>1){if(!commit_done.empty())commit_done.back()->wait(ss);history_advance(count,col,col+q,ss);commit_done.push_back(std::make_unique<Event>());commit_done.back()->record(ss);}
               else history_advance(count,col,col+q,ss);}
    if(per_packet)exec_end(ev,ss);
    if(depth>1)slot_free[slot-lane]->record(ss);
    col+=q;
   }
   if(!per_packet)exec_end(elim_ev,st);
   cw.elimination(packet,std::min(depth,packet),count,depth==1?"selected_depth_one":"single_packet_tail");
   certify_elimination(packet,std::min(depth,packet),stop-start,"tree-level merge apply",count);
   pipeline_join(depth,lane);
  ++exec_elimination;
  }
  void local_level(const std::vector<TileMerge>&es,TiledMatrix<T>&x,bool factor,bool transpose,int lo=-1,int hi=-1,bool emit_factor=true,int lane=0){
   int count=es.size(),b=plan.b;if(!count)return;select_packets(es.front().tile);
   // Retrying needs a Sync-templated, live-range-split kernel; the probe + kernel stay as study
   // artifacts (never dispatched).
   if(factor&&emit_factor){const unsigned tev=exec_begin(0,EXEC_TT_FACTOR,0,stream.s);tile_history_children<<<ceildiv(count,128),128,0,stream>>>(hist.p,merges,count,plan.n,status.p);tile_stack<<<dim3(4*plan.radix,count),128,0,stream>>>(x.a.p,x.ld,x.begin,merges,count,stack.p,b,plan.radix);if(gau_tt_level(count,es[0].h,2)){cw.product(CarrierWitness::TT_GE,1,count,'N');}else{cw.product_uncarried(CarrierWitness::TT_GE,CarrierWitness::R_TT_BLOCK_PATH,count);certify_unimplemented(CarrierWitness::TT_GE,CarrierWitness::R_TT_BLOCK_PATH,plan.radix*b,es[0].h,1,count,"cluster TT carrier over the merge stack (plan P6.1)","the merge stack [R_0;R_1;..] is a stack of upper triangles; the cooperative GE cannot run it and no c>1 TT carrier exists yet, so the block GE runs it at c=1 -- an open implementation gap, not a price",count);hh(stack.p,plan.radix*b,count,2,plan.tt_shared,size_t(plan.radix)*es[0].h*es[0].h*sizeof(T));}tile_scatter_factors<<<dim3(4*plan.radix,count),128,0,stream>>>(x.a.p,x.ld,x.begin,merges,count,stack.p,b,plan.radix);history_start(count,es[0].col+es[0].h);exec_end(tev,stream.s);}
   local_trailing(es,x,factor,transpose,lo,hi,lane);
  }
  void record_global_identity(int kind,int reason){
   cw.product_uncarried(kind,reason);
   // Move only the operations whose launched kernels now write witnesses.
   --cw.unobserved_partials[kind];++cw.native_partials[kind];
   if(kind==CarrierWitness::APPLY_D){--cw.unobserved_commits_issued;++cw.native_commits_issued;}
   cw.note_block_peers(kind,1,1);cw.note_membership(1);
   cw.block_carrier_geometry(kind,1,128,1,1,"serial","identity map; no contraction index",
    "one source entry per output entry; no K split",
    "structural copy or one owner subtraction; no additive combine",1);
   cw.levels(kind,1,1,1,1,1,true);
  }
  void global_merge(const TileMerge&e,TiledMatrix<T>&x,bool factor,bool transpose,int lo=-1,int hi=-1,bool emit_factor=true,int lane=0,int*issued=nullptr,int child_ready=-1){
  const int required_history=child_ready<0?plan.n:child_ready;
  if(child_ready>=0&&(!multi_leaf_global_lookahead()||child_ready<e.col+e.h||child_ready>plan.n))
   throw std::runtime_error("global_child_history_window_descriptor");
  const cudaStream_t stage_stream=slot_stream(lane);
  if(lane&&emit_factor)throw std::runtime_error("global_factor_requires_main_lane");
  transport->bind(stage_stream);
  const double merge_t0=seconds();double presync=0,tsport=0;double mark=merge_t0;++merge_calls;
  // No drain: the transport is bound to this stage_stream, so the operand writes
  // above and the episode below are already ordered. PRESYNC now measures only
  // the host time spent issuing the local work, which is what it should be.
  auto PRESYNC=[&]{double now=seconds();presync+=now-mark;mark=now;};
  auto TSPORT=[&]{double now=seconds();tsport+=now-mark;mark=now;};
  // K-ARY ACROSS-GPU MERGE. A node combines K children. Child 0 owns it. The apply forms W =
  // sum_{i<K} V_i^T X_i over K DISJOINT row sets -- {Combine: additive} over disjoint K_z, exactly
  // the paper's condition -- and the reduce that sums them already spans every participant, so a
  // K-way node costs the SAME per episode as a 2-way one while issuing 1/(K-1) as many of them. K=2
  // reproduces the former pairwise node operation for operation.
  const int b=plan.b,h=e.h,K=e.k;const int owner=e.rank[0];const uint64_t expr=++expression;
  const bool column_split=std::getenv("TQR_MEMBER_COLUMNS")&&std::string(std::getenv("TQR_MEMBER_COLUMNS"))=="1";
  cw.global_column_split=column_split;
  int mine=-1;for(int i=0;i<K;++i)if(e.rank[i]==ctx.rank)mine=i;   // -1: holds no block of this node
  if(ctx.rank==owner)select_packets(e.tile);
  if(factor&&emit_factor){
   if(mine==0){if(global_tt_recorded)tile_history_atleast<<<1,1,0,stage_stream>>>(hist.p,e.child[0],required_history,status.p);else tile_history_one<<<1,1,0,stage_stream>>>(hist.p,e.child[0],required_history,status.p);stack.zero(stage_stream);extract_r<<<8,128,0,stage_stream>>>(x.a.p,x.ld,e.row[0]-x.begin,e.col,e.height[0],h,stack.p,K*b);}
   if(mine>0){if(global_tt_recorded)tile_history_atleast<<<1,1,0,stage_stream>>>(hist.p,e.child[mine],required_history,status.p);else tile_history_one<<<1,1,0,stage_stream>>>(hist.p,e.child[mine],required_history,status.p);CU(cudaMemsetAsync(remoteR.p+size_t(mine-1)*b*b,0,size_t(b)*b*sizeof(T),stage_stream));extract_r<<<8,128,0,stage_stream>>>(x.a.p,x.ld,e.row[mine]-x.begin,e.col,e.height[mine],h,remoteR.p+size_t(mine-1)*b*b,b);}
   // Gather the K-1 non-owner R blocks. Each sender owns its own inbox region,
   // so the K-1 publications neither collide nor need a barrier between them.
   for(int i=1;i<K;++i){PRESYNC();transport->publish(e.rank[i],owner,remoteR.p+size_t(i-1)*b*b,remoteR.p+size_t(i-1)*b*b,size_t(b)*h*sizeof(T),expr,100+uint64_t(i));TSPORT();}
   if(mine==0){int off=e.height[0];
    for(int i=1;i<K;++i){CU(cudaMemcpy2DAsync(stack.p+off,K*b*sizeof(T),remoteR.p+size_t(i-1)*b*b,b*sizeof(T),e.height[i]*sizeof(T),h,cudaMemcpyDeviceToDevice,stage_stream));off+=e.height[i];}
    const unsigned gtev=exec_begin(0,EXEC_TT_FACTOR,0,stage_stream);
    // The same Householder TT chain as the local tree, now on the gathered
    // global stack. Reflectors are replicated to column owners; concurrent
    // lanes cut the row contraction and add their partials in fixed order.
    // The GPU-level W join below remains distributed over all K members.
    const char* global_gau=std::getenv("TQR_GLOBAL_GAU_TT");
    if((!global_gau||std::string(global_gau)!="0")&&gau_tt_level(1,h,2,K*b)){
     cw.product(CarrierWitness::TT_GE,1,1,'N');
    }else{
     cw.product_uncarried(CarrierWitness::TT_GE,CarrierWitness::R_TT_BLOCK_PATH,1);
     certify_unimplemented(CarrierWitness::TT_GE,CarrierWitness::R_TT_BLOCK_PATH,K*b,h,1,1,
      "Gau TT carrier over the across-GPU merge stack",
      "global Gau TT is disabled or failed preflight; legacy block TT is an unqualified fallback",1);
     hh(stack.p,K*b,1,2,plan.tt_shared,size_t(e.rows())*h*sizeof(T));
    }
    scatter_upper<<<8,128,0,stage_stream>>>(stack.p,K*b,x.a.p,x.ld,e.row[0]-x.begin,e.col,h,h);
    off=e.height[0];
    for(int i=1;i<K;++i){CU(cudaMemsetAsync(remoteR.p+size_t(i-1)*b*b,0,size_t(b)*b*sizeof(T),stage_stream));CU(cudaMemcpy2DAsync(remoteR.p+size_t(i-1)*b*b,b*sizeof(T),stack.p+off,K*b*sizeof(T),e.height[i]*sizeof(T),h,cudaMemcpyDeviceToDevice,stage_stream));off+=e.height[i];}
    history_start(1,e.col+h);exec_end(gtev,stage_stream);}
   for(int i=1;i<K;++i){PRESYNC();transport->publish(owner,e.rank[i],remoteR.p+size_t(i-1)*b*b,remoteR.p+size_t(i-1)*b*b,size_t(b)*h*sizeof(T),expr,200+uint64_t(i));TSPORT();}
   if(mine>0)scatter_upper<<<8,128,0,stage_stream>>>(remoteR.p+size_t(mine-1)*b*b,b,x.a.p,x.ld,e.row[mine]-x.begin,e.col,e.height[mine],h);
  }
  if(mine>0)tile_bottom_v<<<8,128,0,stage_stream>>>(factor_A->a.p,factor_A->ld,e.row[mine]-factor_A->begin,e.col,e.height[mine],h,remoteV.p,b);
  // T is replicated once per merge application, then each member computes
  // its disjoint column slice of Z. remoteR is dead after factor scattering;
  // reusing its first b*b region preserves the existing inventory.
  if(column_split)for(int i=1;i<K;++i){
   PRESYNC();transport->publish(owner,e.rank[i],tri.p+size_t(e.tile)*b*b,remoteR.p,size_t(b)*b*sizeof(T),expr,400+uint64_t(i));TSPORT();}
  int start=factor?e.col+h:0;if(lo>start)start=lo;const int stop=hi<0?x.n:std::min(hi,x.n);
  // M3(a): host issue remains one ordered sequence on Transport's own communication stage_stream.
  const bool ordered=column_split&&std::getenv("TQR_GLOBAL_PIPELINE")&&std::string(std::getenv("TQR_GLOBAL_PIPELINE"))=="1";
  const int depth=ordered?std::min(std::max(1,plan.d),std::max(1,ceildiv(std::max(0,stop-start),plan.strip))):1;
  cw.global_pipeline=ordered;
  std::vector<std::unique_ptr<Event>> handoff,commit_done;
  if(ordered){
   pipeline_fork(depth,lane);pipe_fork.record(stage_stream);transport->bind(nullptr);pipe_fork.wait(transport->bound());
   for(int slot=0;slot<depth;++slot)handoff.push_back(std::make_unique<Event>());
  }
  // NCCL all-reduce join (TQR_MULTI_NCCL_AR=1): flat node over all ranks, ordered pipeline only.
  static const bool nccl_ar_env=[]{const char*e=std::getenv("TQR_MULTI_NCCL_AR");return e&&std::string(e)=="1";}();
  const bool nccl_ar=nccl_ar_env&&column_split&&ordered&&K==ctx.size&&mine>=0;
  if(nccl_ar)cw.global_nccl_ar=true;
  int gpackets=0;
  for(int col=start;col<stop;){
   const int slot=lane+gpackets%depth;const cudaStream_t gst=slot_stream(slot);
   const cublasHandle_t gbh=slot?pipe_blas[slot]:blas;
   T* pw=ordered?W.p+size_t(slot)*credit_words():remoteW.p;
   T* pr=ordered?X.p+size_t(slot)*x_slot_words():remoteZ.p;
   T* pz=ordered?Z.p+size_t(slot)*z_credit_words():remoteW.p;
   auto to_comm=[&]{if(ordered){handoff[slot-lane]->record(gst);handoff[slot-lane]->wait(transport->bound());}};
   auto to_compute=[&]{if(ordered){handoff[slot-lane]->record(transport->bound());handoff[slot-lane]->wait(gst);}};
   int q=std::min(plan.strip,stop-col);
   if(!ordered){if(mine>=0||!column_split)remoteW.zero(gst);}
   else if(mine>=0)CU(cudaMemsetAsync(pw,0,size_t(std::max(1,plan.apply_c))*b*q*sizeof(T),gst));
   if(ordered&&mine>=0)CU(cudaMemsetAsync(pz,0,size_t(std::max(1,plan.zc))*b*q*sizeof(T),gst));
   ++gpackets;
   const int zfirst=nccl_ar?0:column_split&&mine>=0?int((long long)q*mine/K):0;
   const int zq=nccl_ar?q:column_split?(mine<0?0:int((long long)q*(mine+1)/K)-zfirst):q;
   const bool do_z=nccl_ar?true:column_split?(mine>=0&&zq>0):mine==0;
   // Every participating rank stamps its own packet: a rank's record has to be
   // complete on its own device, not reconstructed from the owner's.
   const unsigned gev=exec_begin(slot,EXEC_GLOBAL_PACKET,(unsigned)(gpackets-1),gst);
   // The across-GPU W join: W = sum_{i<K} V_i^T X_i over K DISJOINT row sets, so the node arity IS
   // the c of this product.
   if(mine==0)++cw.global_owner_packets;
   else if(mine>0)++cw.global_peer_packets;
   else ++cw.transport_only_packets;
   if(do_z){
    if(K>1){
     cw.product(CarrierWitness::JOIN_W,K,1,column_split?'N':'T');
     if(column_split){
      cw.note_block_peers(CarrierWitness::JOIN_W,1);cw.note_membership(1);
      cw.block_carrier_geometry(CarrierWitness::JOIN_W,1,128,1,1,"serial",
       "already formed partials from disjoint node row sets",
       "one output thread consumes one partial from each listed node member",
       "increasing-world-rank additive sum; one write per owned output element",1);
      cw.product_at(CarrierWitness::JOIN_W,CarrierWitness::L_GPU,1);
      cw.levels(CarrierWitness::JOIN_W,1,1,1,K,1,true);
     }
    }
    else cw.product_uncarried(CarrierWitness::JOIN_W,CarrierWitness::R_JOIN_SINGLE);
    if(column_split)cw.note_global_columns(K,mine,q,zfirst,zq);
   }
    // Alg 2 gives Z and D their own carriers at EVERY level). They now go through the SAME dispatch
    // the local apply uses, so the cut, the witness and the price agree.
    //
    // Two of them genuinely admit no cut and say so with a certificate: the owner's V block is the
    // identity, so its W is a COPY and its D a subtraction -- there is no contraction index to
    // split.
    T**gpp=carrier_ptr.p+size_t(slot)*carrier_ptr_slot();
    if(mine==0){
     record_global_identity(CarrierWitness::APPLY_W,CarrierWitness::R_IDENTITY_W);
     certify_structural(CarrierWitness::APPLY_W,CarrierWitness::R_IDENTITY_W,e.height[0],h,q,1,1,2,
       "the node owner's V block is the identity, so W_0 = X_0 is a copy and not a contraction: there is no [K] to cut (Sec 4 Eq apply)");
     record_global_identity(CarrierWitness::APPLY_D,CarrierWitness::R_IDENTITY_D);
     certify_structural(CarrierWitness::APPLY_D,CarrierWitness::R_IDENTITY_D,e.height[0],h,q,1,1,2,
       "the node owner's V block is the identity, so its D commits X_0 -= Z directly: no contraction index exists to split");
    }
    if(mine==0)tile_copy_identity<<<multi_copy_grid(h,q,32),128,0,gst>>>(x.a.p,x.ld,e.row[0]-x.begin,col,h,q,pw,b,witness.p);
     if(mine>0){
      // W_i = V_i^T X_i over this rank's support rows. [K] is the row index.
      int gw=-1;const int grep=carrier_c(e.height[mine],h,q,1,&gw);
      const bool gnw=native_w_path<T>(grep,e.height[mine]);
      if(grep>1)cw.product(CarrierWitness::APPLY_W,grep,1,gnw?'N':'L');
      else {cw.product_uncarried(CarrierWitness::APPLY_W,gw);certify_apply(CarrierWitness::APPLY_W,gw,e.height[mine],h,q,1);}
      if(gnw){
#ifdef TQR_MULTI_FIXED_DRIVER
       if(launch_multi_w_mma8(grep,remoteV.p,b,0,x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,
         pw,b,size_t(b)*q,e.height[mine],h,q,1,gst)){}
       else if(launch_multi_kami_w(grep,remoteV.p,b,0,x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,
         pw,b,size_t(b)*q,e.height[mine],h,q,1,gst)){}
       else
#endif
       {apply_stage_W<T>(gbh,gst,grep,remoteV.p,b,0,x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,
        pw,b,size_t(b)*q,e.height[mine],h,q,1,gpp,&cw.remainder_peers,plan.wcb,witness.p);record_native_w(grep,q,1);}}
      else
      record_block_apply(CarrierWitness::APPLY_W,
       apply_stage_W<T>(gbh,gst,grep,remoteV.p,b,0,x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,
        pw,b,size_t(b)*q,e.height[mine],h,q,1,gpp,&cw.remainder_peers,plan.wcb,witness.p),
       "rows",1,grep);
     }
    PRESYNC();to_comm();
    if(nccl_ar)transport->template allreduce<T>(pw,pr,size_t(b)*q,++expression);
    else if(column_split)transport->template columns<T,true>(e.rank,K,pw,pr,b,q,false,++expression,witness.p);
    else transport->template paper<T,true>(owner,pw,pr,b*q,++expression);
    to_compute();TSPORT();
    const bool gnative=native_d_path<T>(carrier_c_d(h,1),h);
     if(do_z){
      // Z = T^T W on the assigned column slice, cutting [K]=h exactly as the local Z does.
      int gz=-1;const int gzrep=carrier_c_z(h,1,&gz);
      const bool gnz=native_z_path<T>(gzrep,h);
      if(gzrep>1)cw.product(CarrierWitness::APPLY_Z,gzrep,1,gnz?'N':'L');
      else {cw.product_uncarried(CarrierWitness::APPLY_Z,gz);certify_apply(CarrierWitness::APPLY_Z,gz,e.height[0],h,zq,1);}
      if(gnz){apply_stage_Z<T>(gbh,gst,gzrep,(mine==0?tri.p+size_t(e.tile)*b*b:remoteR.p),b,0,pr,b,size_t(b)*zq,
        pz,gnative?zq:b,size_t(b)*zq,h,zq,1,transpose,gpp,plan.zcb,witness.p,gnative);record_native_z(gzrep,1,h);}
      else
      record_block_apply(CarrierWitness::APPLY_Z,
       apply_stage_Z<T>(gbh,gst,gzrep,(mine==0?tri.p+size_t(e.tile)*b*b:remoteR.p),b,0,pr,b,size_t(b)*zq,
        pz,gnative?zq:b,size_t(b)*zq,h,zq,1,transpose,gpp,plan.zcb,witness.p,gnative),
       "reflectors",1,gzrep);
     }
    T* zfull=pr;   // full Z (transposed when gnative) consumed by the owner commit and the members' D
    if(nccl_ar)zfull=pz;   // every rank computed all q columns of Z from the identical all-reduced W
    else{
    PRESYNC();to_comm();
    if(column_split)transport->template columns<T,false>(e.rank,K,pz,pr,b,q,gnative,++expression);
    else transport->template paper<T,false>(owner,pz,pr,b*q,++expression);
    to_compute();TSPORT();}
    if(mine==0){if(gnative)tile_add_identity_t<<<multi_copy_grid(h,q,32),128,0,gst>>>(x.a.p,x.ld,e.row[0]-x.begin,col,h,q,zfull,q,witness.p);
     else tile_add_identity<<<multi_copy_grid(h,q,32),128,0,gst>>>(x.a.p,x.ld,e.row[0]-x.begin,col,h,q,zfull,b,witness.p);if(factor){
      if(ordered&&!commit_done.empty())commit_done.back()->wait(gst);
      history_advance(1,col,col+q,gst);
      if(ordered){commit_done.push_back(std::make_unique<Event>());commit_done.back()->record(gst);}}
    }
     if(mine>0){
      // X_i -= V_i Z, cutting the same [K]=h and accumulating in place.
      int gd=-1;const int gdrep=carrier_c_d(h,1,&gd);
      if(gdrep>1)cw.product(CarrierWitness::APPLY_D,gdrep,1,gnative?'N':'L');
      else {cw.product_uncarried(CarrierWitness::APPLY_D,gd);certify_apply(CarrierWitness::APPLY_D,gd,e.height[mine],h,q,1);}
      if(gnative){
       const int cexec=apply_stage_D<T>(gbh,gst,gdrep,remoteV.p,b,0,zfull,q,size_t(b)*q,
        x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,e.height[mine],h,q,1,plan.dcb,witness.p,true);
#ifdef TQR_MULTI_FIXED_DRIVER
       if(kami::last_dnp_threads())record_kami_physical(kami::last_dnp_threads(),1);
       else
#endif
       if(short_carrier_path<T>(cexec,h,true))record_block_apply(CarrierWitness::APPLY_D,cexec,"reflectors",1);
       else record_block_sliced(CarrierWitness::APPLY_D,cexec,carrier_d_slice_threads<T>(cexec,e.height[mine],q,1),"reflectors",1);
      }else
      record_block_apply(CarrierWitness::APPLY_D,
       apply_stage_D<T>(gbh,gst,gdrep,remoteV.p,b,0,zfull,b,size_t(b)*q,
        x.a.p+e.row[mine]-x.begin+size_t(col)*x.ld,x.ld,0,e.height[mine],h,q,1,plan.dcb,witness.p),
       "reflectors",1,gdrep);
     }
   exec_end(gev,gst);
   col+=q;
  }
  if(ordered){pipeline_join(depth,lane);transport->bind(stage_stream);}
  if(issued){issued[0]=gpackets;issued[1]=std::max(0,stop-start);issued[2]=std::min(depth,gpackets);}
  else if(mine==0){
   cw.elimination(gpackets,std::min(depth,gpackets),1,ordered?(depth==1?"selected_depth_one":"single_packet_tail"):"serial_global_apply");
   certify_elimination(gpackets,std::min(depth,gpackets),stop-start,
    ordered?"across-GPU apply: ordered communication stage_stream and event-linked compute slots":
     "the legacy across-GPU apply runs on one stage_stream");
  }
  ++exec_elimination;
  // The remote history notice follows the owner's LAST commit, so it is issued
  // by whichever pass closes the trailing range, not by every pass.
  PRESYNC();if(factor&&stop>=x.n)for(int i=1;i<K;++i)transport->publish(owner,e.rank[i],nullptr,nullptr,0,expr,300+uint64_t(i));TSPORT();
  merge_total_s+=seconds()-merge_t0;merge_presync_s+=presync;merge_transport_s+=tsport;
 }
#include "multi_global_lookahead.inc"
#include "multi_leaf_lookahead.inc"
 void factor_body(TiledMatrix<T>&a){
  // Reset per factorization, so evidence() reports ONE episode's dispatch and
  // never the sum over a --reps loop. A profile whose denominator held two
  // factorizations is a mistake this workspace has already made once.
  cw=CarrierWitness{};d_products=json::array();near_deferred=0;near_split_a=0;far_deferred=0;resident_packed=0;
  // One episode per record, for the device log too: a denominator holding
  // two factorizations is a mistake this workspace has already made once.
  exec_next=0;exec_elimination=0;if(exec_cap)exec_log.zero(stream);
  witness.zero(stream);hist.zero(stream);
  normalize_columns(a,column_scale.p);
   // Alg 1, last line: "prioritize applies releasing panel k+1 {Pipeline}", and Fig. pipeline: "Because
   // the strips of tile column 1 go first, panel 1 may start at t=7 instead of t=10."
   //
   // The trailing update of panel k is issued as TWO column windows. Window A is exactly the next
   // panel's columns -- everything panel k+1's factor reads -- and runs on the main stream (lane 0).
   // Window B is the rest of the trailing width and runs on the second lane group. The main stream then
   // issues panel k+1's FACTOR without waiting for window B, so the panel factor (the barrier-bound
   // cooperative carrier, whose cost per column barely falls with the number of rows) overlaps the bulk
   // trailing update (bandwidth-bound). Only before window A of panel k+1 does the main stream wait for
   // window B of panel k, because those are the columns window B still owns.
   //
   // Nothing about the arithmetic changes: the same kernels run on the same operands in the same order
   // per column, the history cursor still advances upward (window A holds the lowest trailing columns),
   // and V is staged in two halves by panel parity so the two overlapping panels never write each
   // other's reflector block.
   //
   // TWO restrictions, both structural and both checked here rather than assumed. (1) A panel with a
   // local TREE cannot be split: the level-l+1 factor calls tile_history_children, which requires every
   // child packet to have committed ALL n columns, so the level factors cannot be hoisted past window
   // B.
   // (2) A panel with a GLOBAL merge cannot be split: a merge issues transport
   // episodes on the stream the transport is bound to, and two streams issuing episodes would leave the
   // ranks no single order to agree on. The previous unit's trailing update is issued as window A --
   // exactly the next unit's columns, on the main lane group -- and window B, the rest, on the second
   // lane group (lanes d..2d-1, default priority) while the main group (greatest priority) factors the
   // next unit and applies its near windows, which read only window A's columns. Before the next unit's
   // own far update touches window B's columns (or its composed V_g/T_g are rebuilt), the main stream
   // waits trail_done. Arithmetic, order per column, the elimination list and every carrier are
   // unchanged; only the stream a window is issued on moves.
   if(multi_leaf_global_lookahead())factor_multi_leaf_global(a);
   else if(multi_global_tail_lookahead())factor_multi_global_tail(a);
   else {
    const int depth=std::max(1,plan.d);const bool la=plan.la!=0;const int laneB=la?depth:0;
    const bool nearN=la&&near_enabled();const int laneN=2*depth;bool nearPending=false;
    bool ownB=false;int pendingHalf=-1;   // pendingHalf: V half a single panel's window B still reads
    // TQR_DEFER_B=1 issues window B(G) on the host right AFTER the next unit's first factor instead of right after
    // window A(G). Both become ready when window A(G) retires; issued first, B(G)'s one-wave far CTAs took the SMs
    // and the next (cooperative, higher-priority) panel waited for them to drain (nsys tall fp64: ~147 us per group
    // boundary). B(G) writes columns beyond the next unit and reads the immutable V_g/T_g generation G; its device
    // order (pipe_fork after window A(G)) is unchanged -- only the host issue order moves.
    static const bool deferB=[]{const char*e=std::getenv("TQR_DEFER_B");return e&&std::string(e)=="1";}();
    // pipeline_fork() re-records pipe_fork during the next panel. Retain a
    // separate event for this window's readiness until its deferred issue.
    std::unique_ptr<Event> deferred_ready;if(deferB)deferred_ready=std::make_unique<Event>();
    std::function<void()> pendingB;
    auto flushB=[&]{if(pendingB){auto f=std::move(pendingB);pendingB=nullptr;f();}};
    auto wait_B=[&]{flushB();if(ownB){trail_done.wait(stream.s);ownB=false;pendingHalf=-1;}};
    // Last column (exclusive) of the unit that starts at panel k.
    auto unit_end=[&](size_t k)->int{if(k>=plan.panels.size())return a.n;const int gg=p7_group(k,a);const auto&pl=plan.panels[k+gg-1];return std::min(a.n,pl.col+pl.h);};
    for(size_t pi=0;pi<plan.panels.size();++pi){
     auto&pan=plan.panels[pi];auto&batch=pan.ranks[ctx.rank];
     const int base=pan.col+pan.h;
     const bool last=pi+1>=plan.panels.size();
     double lt0=seconds();
     if(const int g=p7_group(pi,a);g>1){
      // eliminations k..k+g-1. Each panel k+j first gets every earlier constituent on its own
      // columns (near window, individual Applies -- needed before its factor); the far columns
      // [near_end, n) then get the composed Apply once (windows A and B under lookahead). A pending
      // single-panel window B reads a V half this group's factors would overwrite.
      if(pendingHalf>=0)wait_B();
      const auto&pl=plan.panels[pi+g-1];const int near_end=pl.col+pl.h;
      std::vector<const TileBatch*>grp;std::vector<std::array<int,2>>iss(g);
      // compose steps on the near slot (per constituent, right after its factor) and window A split -- only the next
      // unit's first panel columns stay on the main stream.
      const bool near2=nearN&&near2_enabled()&&agg_gens>1&&pipe_mode()==1;
      ComposeLane NL{};int composes=0;const TileBatch&first=plan.panels[pi].ranks[ctx.rank];
      if(near2){std::vector<const TileBatch*>all;for(int j=0;j<g;++j)all.push_back(&plan.panels[pi+j].ranks[ctx.rank]);
       composed_check(all);agg_gen^=1;NL=compose_lane(laneN);}
      for(int j=0;j<g;++j){auto&bj=plan.panels[pi+j].ranks[ctx.rank];grp.push_back(&bj);
       vbase=size_t(j)*vhalf;iss[j]={0,0};
       if(!nearN){
        if(pendingB){
         // Queue the ready factor first, then B of the previous group. This
         // panel's near columns lie in that group's completed window A.
         const int endj=bj.col+bj.h;const unsigned e0=exec_elimination;
         std::array<int,2>f{0,0},e{0,0};
         ge_batch(bj,a,true,true,endj,endj,true,0,f.data());flushB();
         exec_elimination=e0;ge_batch(bj,a,true,true,endj,near_end,false,0,e.data());
         exec_elimination=e0+1;iss[j]={f[0]+e[0],f[1]+e[1]};
        }else ge_batch(bj,a,true,true,-1,near_end,true,0,iss[j].data());
        continue;}
       // Factor j on the main stream (its columns were finished by panel j-1's eager apply, stream-ordered before
       // it); retire the previous deferred apply, which wrote the columns panel j's apply updates next (ordered
       // composition per column); the EAGER apply covers panel j+1's columns (they release its factor); the rest of
       // the group's columns go to the near slot at middle priority, beside panel j+1's factor. One device
       // elimination index for the three calls, as in the single-panel look-ahead split. Arithmetic, carriers and the
       // per-column commit order are unchanged.
       const int endj=plan.panels[pi+j].col+plan.panels[pi+j].h;
       const int eag=(j+1<g)?plan.panels[pi+j+1].col+plan.panels[pi+j+1].h:near_end;
       const unsigned e0=exec_elimination;std::array<int,2>f{0,0},e{0,0},dn{0,0};
       ge_batch(bj,a,true,true,endj,endj,true,0,f.data());flushB();
       if(nearPending){near_done.wait(stream.s);nearPending=false;}
       if(eag>endj){exec_elimination=e0;ge_batch(bj,a,true,true,endj,eag,false,0,e.data());}
       if(eag<near_end){
        near_fork.record(stream.s);near_fork.wait(slot_stream(laneN));
        exec_elimination=e0;depth_cap=1;ge_batch(bj,a,true,true,eag,near_end,false,laneN,dn.data());depth_cap=0;
        near_done.record(slot_stream(laneN));nearPending=true;++near_deferred;}
       exec_elimination=e0+1;
       if(near2){
        if(eag>=near_end){near_fork.record(stream.s);near_fork.wait(slot_stream(laneN));}   // order the slot after factor j
        if(j==0)composed_begin(g,NL);
        composes+=composed_step(first,bj,g,j,NL);}
       iss[j]={f[0]+e[0]+dn[0],f[1]+e[1]+dn[1]};}
      if(near2){composed_end(first,g,composes,NL);near_compose_done.record(slot_stream(laneN));}
      if(nearPending){near_done.wait(stream.s);nearPending=false;}
      vbase=0;
      local_panel_s+=seconds()-lt0;
      // The far strips carry all constituents; their device intervals are stamped with the last
      // constituent's elimination index, restored afterwards.
      const unsigned saved=exec_elimination;exec_elimination=saved-1;
      const int grel=(la&&pi+g<plan.panels.size())?unit_end(pi+g):a.n;
      int nf=0;
      if(near2){
       near_compose_done.wait(stream.s);wait_B();
       const int a1=(pi+g<plan.panels.size())?std::min(grel,near_end+plan.panels[pi+g].h):grel;
       nf=composed_apply(grp,a,near_end,a1,0);
       Event*afterB=nullptr;
       if(a1<grel){near_fork.record(stream.s);near_fork.wait(slot_stream(laneN));
        depth_cap=1;nf+=composed_apply(grp,a,a1,grel,laneN);depth_cap=0;
        near_done.record(slot_stream(laneN));nearPending=true;afterB=&near_done;++near_split_a;}
       if(grel<a.n){pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
        nf+=composed_apply(grp,a,grel,a.n,laneB,afterB);trail_done.record(slot_stream(laneB));ownB=true;}
      }else if(agg_gens>1){
       // (1) compose group G into the generation that window B(G-1) does NOT read, BEFORE retiring it -- V_g/T_g are
       //     immutable once composed and the two generations alternate, so this overlaps the far update;
       // (2) retire B(G-1): it owns columns of both windows below (ordered composition on every column);
       // (3) fork B(G) immediately, beside window A(G): the windows write disjoint columns and read the same immutable
       //     V_g/T_g; the main stream keeps the greatest priority, so A(G) (which releases group G+1) is served first.
       //     Only B(G)'s first history commit waits for A(G)'s last one (per-constituent strip cursor, column order).
       // Arithmetic, carriers, the elimination list and the commit order per column are unchanged.
       agg_gen^=1;
       composed_prepare(grp);
       wait_B();
       if(pipe_mode()<2){
        nf=composed_apply(grp,a,near_end,grel,0);
        if(grel<a.n){
         if(deferB&&pi+g<plan.panels.size()){
          // strips counted now (deterministic geometry), issued after the next unit's first factor
          const int q=plan.strip,nb=(a.n-grel+q-1)/q;nf+=nb;
          deferred_ready->record(stream.s);
          const int generation=agg_gen;const unsigned elimination=exec_elimination;
          pendingB=[this,grp,&a,grel,laneB,&ownB,&deferred_ready,generation,elimination]{
           // composed_apply selects both the V/T addresses and cache epoch
           // through agg_gen. The next unit may already have advanced it.
           const int current_gen=agg_gen;const unsigned current_elim=exec_elimination;
           agg_gen=generation;exec_elimination=elimination;
           deferred_ready->wait(slot_stream(laneB));composed_apply(grp,a,grel,a.n,laneB);
           trail_done.record(slot_stream(laneB));ownB=true;++far_deferred;
           agg_gen=current_gen;exec_elimination=current_elim;};
         }else{pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
         nf+=composed_apply(grp,a,grel,a.n,laneB);trail_done.record(slot_stream(laneB));ownB=true;}}
       }else if(grel<a.n){
        winb_fork.record(stream.s);winb_fork.wait(slot_stream(laneB));
        nf=composed_apply(grp,a,near_end,grel,0);
        wina_done.record(stream.s);
        nf+=composed_apply(grp,a,grel,a.n,laneB,&wina_done);
        trail_done.record(slot_stream(laneB));ownB=true;
       }else nf=composed_apply(grp,a,near_end,grel,0);
      }else{
      // The previous unit's window B owns [near_end, n) and reads V_g/T_g: retire it before either
      // is touched.
      wait_B();
      composed_prepare(grp);
      nf=composed_apply(grp,a,near_end,grel,0);
      if(grel<a.n){
       pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
       nf+=composed_apply(grp,a,grel,a.n,laneB);
       trail_done.record(slot_stream(laneB));ownB=true;}
      }
      exec_elimination=saved;
      // Eq pipe's r per constituent: its near strips plus every composed strip (one owner commit
      // per constituent per strip, device-counted).
      for(int j=0;j<g;++j){const int packets=iss[j][0]+nf;
       cw.elimination(packets,std::min(depth,packets),1,depth==1?"selected_depth_one":"single_packet_tail");
       certify_elimination(packets,std::min(depth,packets),iss[j][1]+(a.n-near_end),"leaf trailing apply (aggregated far columns, P7)",1);}
      pi+=g-1;continue;
     }
     if(nearPending){near_done.wait(stream.s);nearPending=false;}   // A split window A on the near slot owns these columns
     const bool split=la&&!last&&base<a.n&&pan.global.empty()&&batch.levels.empty()&&!batch.ge.empty();
     if(!split){
      wait_B();
      vbase=0;
      ge_batch(batch,a,true,true);
      for(auto&level:batch.levels)local_level(level,a,true,true);
      local_panel_s+=seconds()-lt0;
      for(auto&e:pan.global)global_merge(e,a,true,true);
      continue;
     }
     const int release=unit_end(pi+1);
     vbase=(pi&1)*vhalf;
     // ONE elimination issued as three calls (factor, window A, window B): each call reports its
     // packets instead of recording an elimination, and all three carry the same device
     // elimination index, so the witness holds one elimination with r = packets(A) + packets(B)
     // (a compile-time split would record three).
     const unsigned e0=exec_elimination;std::array<int,2>fi{0,0},wa{0,0},wb{0,0};
     // Panel k's FACTOR first, on the main stream, with an empty apply window.
     // It reads only panel k's own columns, which the previous unit's window A already
     // committed, so it is issued BEFORE the wait below -- that is the overlap.
     ge_batch(batch,a,true,true,base,base,true,0,fi.data());
     wait_B();
     // Window A: the columns that release the next unit.
     exec_elimination=e0;ge_batch(batch,a,true,true,base,release,false,0,wa.data());
     local_panel_s+=seconds()-lt0;
     if(release<a.n){
      lt0=seconds();
      pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
      exec_elimination=e0;ge_batch(batch,a,true,true,release,a.n,false,laneB,wb.data());
      trail_done.record(slot_stream(laneB));ownB=true;pendingHalf=int(pi&1);
      local_panel_s+=seconds()-lt0;
     }
     exec_elimination=e0+1;
     {const int packets=fi[0]+wa[0]+wb[0],count=int(batch.ge.size());
      cw.elimination(packets,std::min(depth,packets),count,depth==1?"selected_depth_one":"single_packet_tail");
      certify_elimination(packets,std::min(depth,packets),fi[1]+wa[1]+wb[1],"leaf trailing apply (lookahead windows A and B)",count);}
    }
    wait_B();
    if(nearPending){near_done.wait(stream.s);nearPending=false;}
    vbase=0;
   }
  if(hist.n)tile_history_end<<<128,128,0,stream>>>(hist.p,hist.n,plan.n,status.p);
  // restore the upper triangle and check every element for finiteness in ONE pass (was restore + a
  // separate tiled_finite_output scan of the whole matrix).
  if(a.nr&&a.n)tiled_column_scale<T,true,true,true><<<column_pass_grid(a.nr,a.n),256,0,stream>>>(a.a.p,a.nr,a.n,a.ld,a.begin,column_scale.p,status.p,rmap.cyclic()?rmap.blk:0,ctx.size,ctx.rank);
 }

public:
 TiledEngine(Context&c,TiledPlan a,size_t budget=SIZE_MAX):ctx(c),plan(std::move(a)),stream(plan.la!=0){
  if(plan.p!=ctx.size)throw std::runtime_error("tiled_subset_not_yet_admitted");
  if(plan.scalar){
   if(ctx.size!=1)throw std::runtime_error("scalar_packet_requires_standalone");size_t before,total,after;CU(cudaMemGetInfo(&before,&total));
   if(plan.small_shared){cudaDeviceProp prop;cudaFuncAttributes attr;CU(cudaGetDeviceProperties(&prop,ctx.device));CU(cudaFuncGetAttributes(&attr,small_shared_packet<T>));
    int available=prop.sharedMemPerBlockOptin-attr.sharedSizeBytes;if(plan.threads<32||size_t(plan.m)*plan.n*sizeof(T)>size_t(std::max(0,available)))throw std::runtime_error("small_packet_shared_capacity_before_modify");
    CU(cudaFuncSetAttribute(small_shared_packet<T>,cudaFuncAttributeMaxDynamicSharedMemorySize,available));}
   if(plan.owned[0]>budget||plan.owned[0]>before)throw std::runtime_error("insufficient_scalar_storage_before_modify");
   status.alloc(1);witness.alloc(1);scalar_tau.alloc(std::min(plan.m,plan.n));CU(cudaMallocHost(&host_status,sizeof(int)));status.zero(stream);witness.zero(stream);stream.sync();CU(cudaMemGetInfo(&after,&total));setup_observed_bytes=before>after?before-after:0;
   size_t matrix=size_t(descriptor_ld(plan.m,plan.pad))*plan.n*sizeof(T);if(checked_add(setup_observed_bytes,matrix)>budget||matrix>after)throw std::runtime_error("observed_scalar_storage_before_modify");return;
  }
  if(plan.ge_shared||plan.tt_shared){cudaDeviceProp prop;cudaFuncAttributes attr;CU(cudaGetDeviceProperties(&prop,ctx.device));CU(cudaFuncGetAttributes(&attr,tiled_ge<T,true>));int available=prop.sharedMemPerBlockOptin-attr.sharedSizeBytes;size_t needed=0;
   for(auto&pan:plan.panels){auto&batch=pan.ranks[ctx.rank];if(plan.ge_shared&&!batch.ge.empty())needed=std::max(needed,size_t(batch.ge[0].rows)*pan.h*sizeof(T));if(plan.tt_shared){for(auto&level:batch.levels)if(!level.empty())needed=std::max(needed,size_t(plan.radix)*pan.h*pan.h*sizeof(T));for(auto&e:pan.global)if(e.top_rank()==ctx.rank)needed=std::max(needed,size_t(e.rows())*e.h*sizeof(T));}}
   if(needed>size_t(std::max(0,available)))throw std::runtime_error("shared_panel_capacity_before_modify");CU(cudaFuncSetAttribute(tiled_ge<T,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,available));}
  size_t before,total;CU(cudaMemGetInfo(&before,&total));if(plan.owned[ctx.rank]>budget||plan.owned[ctx.rank]>before)throw std::runtime_error("insufficient_tiled_storage_before_modify");
  // Cooperative block-level carrier workspace.
  { size_t coop_bytes=coop_partial_slots_for(plan)*sizeof(T);
    if(checked_add(plan.owned[ctx.rank],coop_bytes)>budget) throw std::runtime_error("insufficient_cooperative_storage_before_modify"); }
  status.alloc(1);witness.alloc(1);hist.alloc(plan.retained[ctx.rank]);all_packets.alloc(hist.n);all_merges.alloc(hist.n);status.zero(stream);witness.zero(stream);
  std::vector<TilePacket>hp(hist.n);std::vector<TileMerge>hm(hist.n);std::vector<bool>seen(hist.n,false);
  auto put=[&](TilePacket p){if(size_t(p.tile)>=hp.size()||seen[p.tile])throw std::runtime_error("descriptor_lineage_collision");hp[p.tile]=p;seen[p.tile]=true;};
  for(auto&pan:plan.panels){auto&batch=pan.ranks[ctx.rank];for(auto&p:batch.ge)put(p);for(auto&level:batch.levels)for(size_t i=0;i<level.size();++i){auto&e=level[i];if(e.tile!=level[0].tile+int(i))throw std::runtime_error("noncontiguous_merge_batch");put({0,int(i)*plan.b,e.rows(),e.h,e.tile});hm[e.tile]=e;}for(auto&e:pan.global)if(e.top_rank()==ctx.rank){put({0,0,e.rows(),e.h,e.tile});hm[e.tile]=e;}}
  if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::runtime_error("uninitialized_factor_descriptor");all_packets.upload(hp,stream);all_merges.upload(hm,stream);stream.sync();
  tiled_blas_check(cublasCreate(&blas));tiled_blas_check(cublasSetStream(blas,stream));tiled_blas_check(cublasSetMathMode(blas,tiled_blas_math()));tiled_blas_check(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));
  blas_workspace.alloc(tiled_blas_workspace_bytes);tiled_blas_check(cublasSetWorkspace(blas,blas_workspace.p,blas_workspace.n));
  column_scale.alloc(plan.n);scale_temp.alloc(plan.strip);
  size_t b=plan.b,batch=plan.max_batch;
  // Batched cooperative partials: 2 generations x batch x groups x (2+2b).
  // The group count was a literal 32 here, written when cooperative_ge_batch
  // asked for exactly 32 groups. It now asks coop_groups_for(rows), which is
  // 64 or 128 on a tall panel -- and the size check in cooperative_ge_batch
  // is a SILENT `continue`, so an undersized buffer did not fail, it just
  // skipped every group count the buffer could not hold. The carrier then
  // fell through to g=32, which exceeds shared capacity at h=64, so the
  // TALLEST panels -- the expensive ones -- ran on the block path while the
  // carrier only picked up the short ones. Size it for the largest group
  // count the menu can request.
  // Sized for the whole admitted carrier menu, minipanel boundary included,
  // so no selected (c, width) can be refused for want of workspace -- the
  // silent-skip failure of 8e, which only ever hurt the tallest panels.
  coop_partial.alloc(coop_partial_slots_for(plan));

  win_g.alloc(size_t(std::max(1,plan.apply_c))*plan.b*plan.b);   // c library partials side by side (IEEE fp32 compose_G)
  GPART.alloc(size_t(128)*plan.b*plan.b);
  if(plan.la!=0&&near2_enabled())GPART_N.alloc(size_t(128)*plan.b*plan.b);
  if constexpr(std::is_same_v<T,float>)VR.alloc(size_t(tiled_lane_slots(plan.d,plan.la!=0))*vr_slot_words());   // charged: gmma_vr_words + gmma_x3_words
  agg=plan.agg;   // The schedule's aggregate, charged in the inventory
  if(agg>1){agg_gens=(plan.la!=0&&pipe_enabled())?2:1;vg_words=size_t(plan.leaf)*plan.b*agg;tg_words=size_t(agg)*plan.b*agg*plan.b;
   VG.alloc(vg_words*agg_gens);TG.alloc(tg_words*agg_gens);TGP.alloc(tg_words*agg_gens);AGG_G.alloc(size_t(std::max(1,plan.apply_c)+agg)*plan.b*plan.b);AGG_S.alloc(size_t(agg)*plan.b*plan.b);}
  if(agg>1)p7_setup_layers();
  // Zeroed per factorization in factor_body, so a record always describes ONE episode.
  exec_log.alloc(tiled_evidence_intervals);exec_cap=(unsigned)tiled_evidence_intervals;exec_log.zero(stream);
  size_t rad=size_t(plan.radix);
  // Alg 1's cross-panel Pipeline needs a SECOND lane group: lanes 0..d-1 (the main stream at the
  // device's greatest priority) factor panel k+1 and apply its near window, lanes d..2d-1 carry window
  // B -- the bulk trailing update of panel k -- beside them. Hence 2d slots and 2d W/Z credits, and two
  // halves of V so two panels in flight do not stage over each other. The inventory charges exactly
  // this (tiled_lane_slots(d,la), V at (la?2:1) halves), so the admitted capacity bounds what is
  // allocated.
  vhalf=batch*std::max(size_t(plan.leaf)*b,rad*b*b);
  tri.alloc(plan.retained[ctx.rank]*b*b);V.alloc(size_t(std::max(plan.la?2:1,agg))*vhalf);stack.alloc(std::max(batch*rad,size_t(plan.gradix))*b*b);
  if(gau_enabled()){const auto qw=qrv2_workspace(plan.leaf,b,batch,plan.radix,plan.gradix,plan.apply_c);
   gau_count=qw.count;gau_flags.alloc(qw.flags);CU(cudaMemset(gau_flags.p,0,qw.flags*sizeof(int)));
   gau_tau.alloc(qw.tau);
   // apply_stage_W writes all c partial matrices before combining them into
   // the first slice (IEEE fp32 compose_G). Native W uses GPART instead.
   gau_O.alloc(qw.overlaps);gau_vs.alloc(qw.stack_v);
   // Reflector-overlap partials already use GPART in gau_overlaps(); the
   // previous standalone-probe scratch allocation was never read here.
  }const int lanes=tiled_lane_slots(plan.d,plan.la!=0);X.alloc(size_t(lanes)*batch*rad*b*plan.strip);W.alloc(size_t(lanes)*std::max(1,plan.apply_c)*batch*b*agg*plan.strip);carrier_ptr.alloc(size_t(lanes)*carrier_ptr_slot());Z.alloc(size_t(lanes)*std::max(1,plan.zc)*batch*b*agg*plan.strip);
  { const int dd=tiled_lane_slots(plan.d,plan.la!=0);pipe_stream.resize(dd);pipe_blas.assign(dd,nullptr);pipe_event.resize(dd);pipe_workspace.resize(dd);
     // Main-group side slots 1..d-1 were created at the default (= lowest) priority, level with window B's far
     // update; they now take the greatest (TQR_MAINPRIO=0 restores the default). The near slot 2d takes the middle
     // priority.
     int plo=0,phi=0;CU(cudaDeviceGetStreamPriorityRange(&plo,&phi));
     static const bool mainprio=[]{const char*e=std::getenv("TQR_MAINPRIO");return !(e&&std::string(e)=="0");}();
     for(int i=1;i<dd;++i){
      const bool near_slot=plan.la!=0&&near_enabled()&&i==2*std::max(1,plan.d);
      const bool main_slot=plan.la!=0&&mainprio&&i<std::max(1,plan.d);
      pipe_stream[i]=near_slot?std::make_unique<Stream>(true,(plo+phi)/2):main_slot?std::make_unique<Stream>(true,phi):std::make_unique<Stream>();
      pipe_event[i]=std::make_unique<Event>();
      // A cublas workspace is scratch space, not a constant table: sharing
      // one workspace across handles on concurrent streams races. Each pipe
      // stream owns its workspace (depth is small; the fixed depth is 2).
      pipe_workspace[i].alloc(tiled_blas_workspace_bytes);
      tiled_blas_check(cublasCreate(&pipe_blas[i]));tiled_blas_check(cublasSetStream(pipe_blas[i],pipe_stream[i]->s));
      tiled_blas_check(cublasSetMathMode(pipe_blas[i],tiled_blas_math()));tiled_blas_check(cublasSetAtomicsMode(pipe_blas[i],CUBLAS_ATOMICS_NOT_ALLOWED));
      tiled_blas_check(cublasSetWorkspace(pipe_blas[i],pipe_workspace[i].p,pipe_workspace[i].n));} }
  if constexpr(std::is_same_v<T,double>){if(d_peers_enabled()){
   if(plan.dc!=2)throw std::runtime_error("D_peers_require_c2");
   // multi-GPU partials are sized by this rank's largest possible row count (row-block-cyclic bound or
   // contiguous slab), not by the leaf descriptor bound (262144 in the multi driver's one-leaf-per-rank plans).
   size_t maxr=size_t(plan.leaf);
   if(plan.p>1){const int blk=std::max(1,plan.row_block);maxr=std::min(maxr,size_t(ceildiv(plan.m,blk*plan.p))*size_t(blk)+size_t(blk));}
   d_max_rows=maxr;
   d_lane_words=checked_mul(size_t(2),checked_mul(maxr,size_t(plan.strip)));
   d_partials.alloc(checked_mul(size_t(lanes),d_lane_words));
  }}
  remoteR.alloc(size_t(std::max(1,plan.gradix-1))*b*b);remoteV.alloc(b*b);// remoteW holds the across-GPU carrier's partials: max(apply_c,zc) of a
  // b x strip block, combined in place before the transport reads the first
  // block. remoteZ carries one block (the reduced W, then the broadcast Z).
  remoteW.alloc(size_t(std::max({1,plan.apply_c,plan.zc}))*b*std::max(plan.b,plan.strip));remoteZ.alloc(b*std::max(plan.b,plan.strip));
  // The transport shares the engine's stream, so an episode is issued into the same ordered queue
  // as the arithmetic that produces its operands and consumes its results. Published payloads are
  // only b x h blocks, so the publish arena is charged at b*b -- far below the b x strip the
  // collectives need -- which is what lets every sender own its own destination region.
  transport=std::make_unique<Transport>(ctx,std::max(b*b,b*plan.strip)*sizeof(T),0,size_t(b)*b*sizeof(T));transport->bind(stream.s);
  // M1: the RowMap's cyclic copy, redistribution buffers and permutation/count tables are engine
  // allocations, so they are set up BEFORE the observation and appear in setup_observed_bytes; the caller-matrix term
  // below is then the caller's slab only (charging tiled_rowmap_extra_words there too would count the copy twice).
  rowmap_setup();
  size_t after;CU(cudaMemGetInfo(&after,&total));setup_observed_bytes=before>after?before-after:0;
  // With a native row-block-cyclic caller (tiled_native_input()), the engine's cyclic matrix IS the operand and no
  // caller slab is ever allocated, so there is no slab term to reserve.
  size_t matrix=tiled_native_input()?0:size_t(descriptor_ld(row_begin(plan.m,plan.p,ctx.rank+1)-row_begin(plan.m,plan.p,ctx.rank),plan.pad))*plan.n*sizeof(T);
  if(checked_add(setup_observed_bytes,matrix)>budget||matrix>after)throw std::runtime_error("observed_tiled_storage_before_modify");
 }
 size_t setup_observed_bytes=0;
 // Under block-cyclic rows the engine factors an internal cyclic copy of its local rows: the caller's slab rows
 // are converted on entry to factor() and the factored matrix (R view and V) converted back on exit, both inside
 // the timed factorization (charged, as cuSOLVERMp's slab<->block-cyclic conversion is). The native Q handle (V,
 // T) stays in the cyclic copy; apply_q converts its operand in and out the same way.
 RowMap rmap;std::unique_ptr<TiledMatrix<T>> cyc;Buffer<T> rd_send,rd_recv;
 struct RdTables{std::vector<int> cnt,base,rcnt,rbase;Buffer<int> dest,pos,src,spos,dcnt,dbase,drcnt,drbase;};
 RdTables rd_in,rd_out;uint64_t redistributions=0;double redistribution_host_s=0;
 // Every global row once, in increasing order: sender (s, sl) -> receiver (d, dl).
 template<class SO,class SL,class DO,class DL> void rd_build(RdTables&t,int nsrc,int ndst,SO sown,SL sloc,DO down,DL dloc){
  const int P=ctx.size,me=ctx.rank;t.cnt.assign(P,0);t.rcnt.assign(P,0);std::vector<int> dest(std::max(1,nsrc)),pos(std::max(1,nsrc)),src(std::max(1,ndst)),spos(std::max(1,ndst));
  for(int g=0;g<plan.m;++g){const int so=sown(g),dn=down(g);
   if(so==me){const int l=sloc(g);dest[l]=dn;pos[l]=t.cnt[dn]++;}
   if(dn==me){const int l=dloc(g);src[l]=so;spos[l]=t.rcnt[so]++;}}
  t.base.assign(P,0);t.rbase.assign(P,0);for(int r=1;r<P;++r){t.base[r]=t.base[r-1]+t.cnt[r-1];t.rbase[r]=t.rbase[r-1]+t.rcnt[r-1];}
  if(t.base[P-1]+t.cnt[P-1]!=nsrc||t.rbase[P-1]+t.rcnt[P-1]!=ndst)throw std::runtime_error("rowmap_redistribution_not_a_bijection");
  auto up=[&](Buffer<int>&b,const std::vector<int>&v){b.alloc(v.size());CU(cudaMemcpy(b.p,v.data(),v.size()*sizeof(int),cudaMemcpyHostToDevice));};
  up(t.dest,dest);up(t.pos,pos);up(t.src,src);up(t.spos,spos);up(t.dcnt,t.cnt);up(t.dbase,t.base);up(t.drcnt,t.rcnt);up(t.drbase,t.rbase);}
 void rowmap_setup(){
  rmap=RowMap(plan.m,ctx.size,plan.row_block);if(!rmap.cyclic())return;
  const int me=ctx.rank,slab=row_begin(plan.m,ctx.size,me+1)-row_begin(plan.m,ctx.size,me),cn=rmap.local_rows(me);
  cyc=std::make_unique<TiledMatrix<T>>(plan.m,plan.n,TiledLocalShape{cn,0},plan.pad);
  const size_t words=tiled_native_input()?size_t(1):size_t(std::max(slab,cn))*size_t(tiled_redistribution_cols(plan.n));rd_send.alloc(std::max<size_t>(1,words));rd_recv.alloc(std::max<size_t>(1,words));
  const RowMap&mp=rmap;const int m=plan.m,P=ctx.size;
  auto slab_owner=[m,P](int g){return owner_of(g,m,P);};auto slab_local=[m,P](int g){return g-row_begin(m,P,owner_of(g,m,P));};
  auto cyc_owner=[mp](int g){return mp.owner(g);};auto cyc_local=[mp](int g){return mp.to_local(g);};
  rd_build(rd_in,slab,cn,slab_owner,slab_local,cyc_owner,cyc_local);
  rd_build(rd_out,cn,slab,cyc_owner,cyc_local,slab_owner,slab_local);}
 void redistribute(const TiledMatrix<T>&from,TiledMatrix<T>&to,const RdTables&t,int ncols){
#ifdef TQR_MULTI
  const double t0=seconds();const int cw=tiled_redistribution_cols(plan.n);ncclComm_t comm=transport->reference_comm();
  const ncclDataType_t dt=sizeof(T)==4?ncclFloat:ncclDouble;const int me=ctx.rank;
  for(int c0=0;c0<ncols;c0+=cw){const int q=std::min(cw,ncols-c0);
   if(from.nr)rd_pack<T><<<1024,256,0,stream>>>(from.a.p,from.ld,from.nr,c0,q,t.dest.p,t.pos.p,t.dcnt.p,t.dbase.p,rd_send.p);
   NC(ncclGroupStart());
   for(int r=0;r<ctx.size;++r){if(r==me)continue;
    if(t.cnt[r])NC(ncclSend(rd_send.p+size_t(t.base[r])*q,size_t(t.cnt[r])*q,dt,r,comm,stream));
    if(t.rcnt[r])NC(ncclRecv(rd_recv.p+size_t(t.rbase[r])*q,size_t(t.rcnt[r])*q,dt,r,comm,stream));}
   NC(ncclGroupEnd());
   if(t.cnt[me])CU(cudaMemcpyAsync(rd_recv.p+size_t(t.rbase[me])*q,rd_send.p+size_t(t.base[me])*q,size_t(t.cnt[me])*q*sizeof(T),cudaMemcpyDeviceToDevice,stream));
   if(to.nr)rd_unpack<T><<<1024,256,0,stream>>>(to.a.p,to.ld,to.nr,c0,q,t.src.p,t.spos.p,t.drcnt.p,t.drbase.p,rd_recv.p);
   CU(cudaGetLastError());}
  ++redistributions;redistribution_host_s+=seconds()-t0;
#else
  (void)from;(void)to;(void)t;(void)ncols;throw std::runtime_error("row_block_requires_multi_gpu_build");
#endif
 }
#ifdef TQR_MULTI_FIXED_DRIVER
 // NATIVE row-block-cyclic operand. The engine's cyclic matrix is the caller's A: the driver generates the input
 // directly in this layout (as cuSOLVERMp receives its 2D block-cyclic layout), so the factorization runs without
 // the slab copy and without the two redistributions (memory: one matrix, not two).
 TiledMatrix<T>& native_matrix(){if(!rmap.cyclic()||!cyc)throw std::runtime_error("native_cyclic_requires_row_block");return *cyc;}
 const RowMap& row_map()const{return rmap;}
 int factor_native(){
  TiledMatrix<T>&a=native_matrix();factor_A=&a;factored=false;++invocation;factor_frozen_=false;
  multi_w_mma8_products=0;
  merge_calls=0;merge_total_s=merge_presync_s=merge_transport_s=local_panel_s=0;
  gau_panels=gau_refusals=gau_tt_levels=gau_sm1_packets=0;
  redistributions=0;redistribution_host_s=0;
  if(transport)transport->reset_accounting();
  if(!a.m||!a.n){factored=true;return OK;}
  status.zero(stream);
  if(a.nr&&a.n)finite_scan_2d<<<column_pass_grid(a.nr,a.n),256,0,stream>>>(a.a.p,a.nr,a.n,a.ld,status.p);stream.sync();int st=ctx.max_status(status.download()[0]);if(st)return st;
  factor_body(a);
  stream.sync();if(transport){transport->checkpoint();transport->agreement_checkpoint();}cw.complete_after_fence();st=ctx.max_status(status.download()[0]);factored=st==0;return st;
 }
 int apply_q_native(TiledMatrix<T>&x,bool transpose){if(!factored||x.m!=plan.m||!rmap.cyclic()||x.nr!=rmap.local_rows(ctx.rank))throw std::runtime_error("invalid_native_q_operand");return apply_q_local(x,transpose);}
#endif
  int factor(TiledMatrix<T>&a){if(a.m!=plan.m||a.n!=plan.n||a.begin!=row_begin(plan.m,ctx.size,ctx.rank)||a.nr!=row_begin(plan.m,ctx.size,ctx.rank+1)-a.begin||a.ld!=descriptor_ld(a.nr,plan.pad))throw std::runtime_error("tiled_shape_mismatch");factor_A=&a;factored=false;++invocation;
   factor_frozen_=false;   // a new invocation opens a new observation scope
  if(ctx.size>1){
#ifdef TQR_MULTI_FIXED_DRIVER
   multi_w_mma8_products=0;
#endif
   merge_calls=0;merge_total_s=merge_presync_s=merge_transport_s=local_panel_s=0;
   gau_panels=gau_refusals=gau_tt_levels=gau_sm1_packets=0;
   redistributions=0;redistribution_host_s=0;
   if(transport)transport->reset_accounting();
  }

  if(!a.m||!a.n){factored=true;return OK;}
  if(plan.scalar){if(plan.small_shared)small_shared_packet<<<1,plan.threads,size_t(a.m)*a.n*sizeof(T),stream>>>(a.a.p,a.m,a.n,a.ld,scalar_tau.p,status.p,witness.p);
   else scalar_inplace<T,true><<<1,plan.threads,0,stream>>>(a.a.p,a.m,a.n,a.ld,scalar_tau.p,status.p,witness.p);
   CU(cudaMemcpyAsync(host_status,status.p,sizeof(int),cudaMemcpyDeviceToHost,stream));stream.sync();factored=*host_status==OK;return *host_status;}
  status.zero(stream);
  if(a.nr&&a.n)finite_scan_2d<<<column_pass_grid(a.nr,a.n),256,0,stream>>>(a.a.p,a.nr,a.n,a.ld,status.p);stream.sync();int st=ctx.max_status(status.download()[0]);if(st)return st;
  // Block-cyclic rows factor the internal cyclic copy; the conversions are inside the timed factorization.
  if(rmap.cyclic()){redistribute(a,*cyc,rd_in,a.n);factor_A=cyc.get();factor_body(*cyc);redistribute(*cyc,a,rd_out,a.n);}
  else factor_body(a);
  stream.sync();if(transport){transport->checkpoint();transport->agreement_checkpoint();}cw.complete_after_fence();st=ctx.max_status(status.download()[0]);factored=st==0;return st;
 }
#ifdef TQR_MULTI_FIXED_DRIVER
 // Untimed validation may borrow the redundant slab output. The authoritative
 // R/V factors stay in the distinct cyclic allocation used by the Q handle.
 bool validation_can_borrow_output(const TiledMatrix<T>&a)const{
  return factored&&ctx.size>1&&rmap.cyclic()&&!plan.scalar&&cyc&&factor_A==cyc.get()&&
    a.a.p!=cyc->a.p&&a.m==plan.m&&a.n==plan.n&&a.n>=a.m&&
    a.nr==row_begin(plan.m,ctx.size,ctx.rank+1)-row_begin(plan.m,ctx.size,ctx.rank)&&
    a.ld==descriptor_ld(a.nr,0);
 }
 void validation_restore_output(TiledMatrix<T>&a){
  if(!validation_can_borrow_output(a))throw std::runtime_error("validation_output_borrow_contract");
  redistribute(*cyc,a,rd_out,a.n);stream.sync();
 }
#endif
 int apply_q(TiledMatrix<T>&x,bool transpose){if(!factored||x.m!=plan.m)throw std::runtime_error("invalid_tiled_handle");
  // The Q handle is cyclic; convert the caller's slab operand in, apply, convert back.
  if(rmap.cyclic()&&!plan.scalar){TiledMatrix<T> xc(x.m,x.n,TiledLocalShape{rmap.local_rows(ctx.rank),0});
   status.zero(stream);if(x.nr&&x.n)finite_scan_2d<<<column_pass_grid(x.nr,x.n),256,0,stream>>>(x.a.p,x.nr,x.n,x.ld,status.p);stream.sync();int pf=ctx.max_status(status.download()[0]);if(pf)return pf;
   redistribute(x,xc,rd_in,x.n);const int st=apply_q_local(xc,transpose);redistribute(xc,x,rd_out,x.n);stream.sync();return st;}
  return apply_q_local(x,transpose);}
 int apply_q_local(TiledMatrix<T>&x,bool transpose){
  if(x.a.n&&factor_A->a.n){uintptr_t lo=reinterpret_cast<uintptr_t>(x.a.p),hi=lo+x.a.n*sizeof(T),flo=reinterpret_cast<uintptr_t>(factor_A->a.p),fhi=flo+factor_A->a.n*sizeof(T);if(lo<fhi&&flo<hi)throw std::runtime_error("Q_operand_aliases_retained_factors");}
  if(plan.scalar){status.zero(stream);if(x.nr&&x.n){finite_scan<<<128,128,0,stream>>>(x.a.p,x.nr,x.n,x.ld,status.p);CU(cudaMemcpyAsync(host_status,status.p,sizeof(int),cudaMemcpyDeviceToHost,stream));stream.sync();if(*host_status)return *host_status;
    scalar_apply<<<x.n,plan.threads,0,stream>>>(factor_A->a.p,factor_A->ld,scalar_tau.p,x.m,std::min(plan.m,plan.n),x.a.p,x.n,x.ld,transpose,status.p);
   }CU(cudaMemcpyAsync(host_status,status.p,sizeof(int),cudaMemcpyDeviceToHost,stream));stream.sync();return *host_status;}
  // Caller owns X. This separate, bounded-in-width scale vector is charged
  // to Q application and allocated before modifying X (validation times it).
  Buffer<T>scales(x.n);status.zero(stream);if(x.nr&&x.n)finite_scan_2d<<<column_pass_grid(x.nr,x.n),256,0,stream>>>(x.a.p,x.nr,x.n,x.ld,status.p);stream.sync();int preflight=ctx.max_status(status.download()[0]);if(preflight)return preflight;
  normalize_columns(x,scales.p);
  if(transpose)for(auto&pan:plan.panels){auto&batch=pan.ranks[ctx.rank];ge_batch(batch,x,false,true);for(auto&level:batch.levels)local_level(level,x,false,true);for(auto&e:pan.global)global_merge(e,x,false,true);}
  else for(auto it=plan.panels.rbegin();it!=plan.panels.rend();++it){for(auto e=it->global.rbegin();e!=it->global.rend();++e)global_merge(*e,x,false,false);auto&batch=it->ranks[ctx.rank];for(auto level=batch.levels.rbegin();level!=batch.levels.rend();++level)local_level(*level,x,false,false);ge_batch(batch,x,false,false);}
  // The restore checks every element it writes (the separate finite_output scan is fused into it).
  if(x.nr&&x.n)tiled_column_scale<T,true><<<column_pass_grid(x.nr,x.n),256,0,stream>>>(x.a.p,x.nr,x.n,x.ld,x.begin,scales.p,status.p);
  stream.sync();if(transport){transport->checkpoint();transport->agreement_checkpoint();}cw.complete_after_fence();return ctx.max_status(status.download()[0]);
 }
 // Physical execution evidence is reduced here, after the fence, from the device log.
 void collect_execution_evidence(const Witness&w){
  if(!exec_cap){cw.exec_present=false;return;}
  const size_t used=std::min<size_t>(exec_next,exec_cap);
  std::vector<ExecInterval>log(used);
  if(used)CU(cudaMemcpy(log.data(),exec_log.p,used*sizeof(ExecInterval),cudaMemcpyDeviceToHost));
   cw.exec=exec_reduce(log,used,w.evidence_overflow);
   for(int k=0;k<CarrierWitness::KINDS;++k){cw.device_products[k]=w.device_products[k];cw.device_peer_partials[k]=w.device_peer_partials[k];cw.device_block_peers[k]=w.device_block_peers[k];
    cw.device_qrv2_products[k]=w.device_qrv2_products[k];cw.device_qrv2_owners[k]=w.device_qrv2_owners[k];
    for(int l=0;l<execution_levels;++l)for(int b=0;b<execution_c_buckets;++b)cw.device_level_c[k][l][b]=w.device_level_c[k][l][b];}
   cw.device_combines=w.device_combines;cw.device_physical_commits=w.device_physical_commits;
   cw.device_history_marks=w.device_history_marks;cw.device_membership_reports=w.device_membership_reports;
   cw.credits_max=w.credits_max;cw.evidence_dropped=w.evidence_overflow;cw.exec_present=true;
 }
    // One receipt per invocation and phase (C0): the same reduction serves
    // the frozen factorization receipt and the live validation receipt; only
    // the phase stamp differs. Idempotent across calls (note_selected
    // overwrites, device counters are re-copied from the same download).
    json receipt(const Witness&w,const char*phase,uint64_t inv){collect_execution_evidence(w);
   // GPU-level cuts are (1,1,c) throughout this implementation: only [K] is cut at the GPU level;
   // Idempotent across --reps factorizations.
   const bool manual=plan.selection.value("status",std::string())=="explicit_configuration";
   cw.note_selected(CarrierWitness::PANEL_GE,1,1,std::max(1,plan.c),manual?"explicit multi-GPU candidate; no measured selector":"DP panel-carrier menu (coop weigh) measurement");
   cw.note_selected(CarrierWitness::TT_GE,1,1,1,"Gau/Householder TT backend reported in qrv2_factors; J owners and block K cuts are distinct from GPU K replication");
   cw.note_selected(CarrierWitness::APPLY_W,1,1,std::max(1,plan.apply_c),manual?"explicit multi-GPU candidate; no measured selector":"apply replication menu measurement");
   cw.note_selected(CarrierWitness::APPLY_Z,1,1,std::max(1,plan.zc),manual?"explicit multi-GPU candidate; no measured selector":"Z replication menu measurement");
   cw.note_selected(CarrierWitness::APPLY_D,1,1,std::max(1,plan.dc),manual?"explicit multi-GPU candidate; no measured selector":"D replication menu measurement");
   cw.note_selected(CarrierWitness::JOIN_W,1,1,std::max(2,plan.gradix),manual?"explicit multi-GPU candidate; no measured selector":"global radix menu measurement");
   // Leave selected_c null; its actual block/cluster/GPU cuts are reported by the native kernels.
    auto carrier_record=cw.record();
#ifdef TQR_MULTI_FIXED_DRIVER
    carrier_record["multi_W_mma8"]={{"host_launched_products_since_factor_start",multi_w_mma8_products},
      {"instruction_shape",{16,8,8}},{"block_c",2},{"BK",32},{"WK",16},
      {"scope","host dispatch count; existing native kernel device peer evidence remains authoritative"}};
    carrier_record["physical_launch_geometry"]={{"format",2},
      {"scope","instrumented arithmetic-kernel witness reports; null-witness arithmetic sublaunches are not counted"},
      {"dimensions",{{128,1,1},{256,1,1},{384,1,1}}},{"unknown_dimension_bucket",3},
      {"kami_dnp",{w.physical_launch_geometry[0][0],w.physical_launch_geometry[0][1],w.physical_launch_geometry[0][2],w.physical_launch_geometry[0][3]}},
      {"gmma_W",{w.physical_launch_geometry[1][0],w.physical_launch_geometry[1][1],w.physical_launch_geometry[1][2],w.physical_launch_geometry[1][3]}},
      {"gmma_D",{w.physical_launch_geometry[2][0],w.physical_launch_geometry[2][1],w.physical_launch_geometry[2][2],w.physical_launch_geometry[2][3]}},
      {"kami_W",{w.physical_launch_geometry[3][0],w.physical_launch_geometry[3][1],w.physical_launch_geometry[3][2],w.physical_launch_geometry[3][3]}}};
#endif
    return {{"invocation",inv},{"phase",phase},{"precision_mode",precision_mode_name<T>()},{"GE",w.ge},{"TT",w.tt},{"reflectors",w.reflectors},{"owner_commits",w.commits},{"carrier_dispatch",carrier_record},{ "D_peer_products",d_products},{"qrv2_panel",{{"enabled",gau_enabled()},{"sm1_enabled",gau_sm_enabled()},{"packets",gau_panels},{"sm1_packets",gau_sm1_packets},{"tt_levels",gau_tt_levels},{"refusals",gau_refusals},{"resident_panels",resident_panels},{"resident_env",coop_resident_env()},{"balanced_aggregate_layers",balanced_aggregate_layers()},{"resident_packed",resident_packed},{"far_deferred_issues",far_deferred},{"near_deferred_applies",near_deferred},{"near_split_window_a",near_split_a},{"near2",near2_enabled()},{"near_lane",near_enabled()},{"pipe_mode",pipe_mode()},{"agg_generations",agg_gens},{"source","gau.nernst GPU MODE 774 #844219 register panels (gau_ge.cuh owner-CTA chain; gau_fast.cuh single-SM fused chain + overlaps + T)"}}},{"history_slots",hist.n},{"lookahead_gmma_sm_budget",{{"free_sms",la_free_sms()},{"gmma_sm_cap",la_gmma_cap()},{"window_b_products",la_capped_products},{"gmma_np_mode",gmma_detail::gmma_np_mode()},{"gmma_np_products",np_products},{"finite_gmma_grid",gmma_detail::gmma_finite_record()},{"vr_packs",vr_packs},{"vr_pack_hits",vr_pack_hits},{"compose_tiled_steps",compose_tiled_steps},{"capped_gmma_launches",gmma_detail::gmma_capped_launches()},{"source","schedule lookahead_free_sms (execution format 32)"}}},{"scalar",plan.scalar},{"small_shared",plan.small_shared},{"scalar_tau_bytes",scalar_tau.n*sizeof(T)},{"host_pinned_status_requested_bytes",host_status?sizeof(int):0},{"backend",transport?transport->evidence:json{{"standalone",true},{"publications",0},{"paper_reductions",0},{"paper_copies",0}}},{"backend_accounting",transport?transport->accounting():json(nullptr)},{"across_gpu_accounting",{{"merge_calls",merge_calls},{"merge_total_s",merge_total_s},{"merge_presync_s",merge_presync_s},{"merge_transport_s",merge_transport_s},{"local_panel_issue_s",local_panel_s},{"scope","host wall time on THIS rank; the transport is bound to the engine stream, so presync is host issue time only (no drain) and transport is host time inside Transport (issue plus the signature Allreduce); GPU time is in the stream, not here"}}},{"setup_observed_bytes",setup_observed_bytes},{"row_distribution",rmap.cyclic()?json{{"kind","block_cyclic"},{"blk",rmap.blk},{"local_rows",rmap.local_rows(ctx.rank)},{"redistributions",redistributions},{"redistribution_host_s",redistribution_host_s},{"policy","slab rows converted to the internal cyclic copy on factor/apply entry and back on exit, inside the timed interval (NCCL send/recv, data only)"}}:json{{"kind","contiguous_slabs"}}},{"blas_workspace_bytes",blas_workspace.n},{"column_scaling_bytes",(column_scale.n+scale_temp.n)*sizeof(T)},{"range_policy","normalize complete columns before QR; restore only final R; normalize and restore Q-application operands; V/T remain dimensionless"},{"descriptor_bytes",all_packets.n*sizeof(TilePacket)+all_merges.n*sizeof(TileMerge)},{"history_contract","per-factor ordered strip cursor; children complete before TT; remote history notice after owner commit"},{"arithmetic",std::string("selected precision GPU BLAS (")+(tiled_blas_math()==CUBLAS_PEDANTIC_MATH?"pedantic":"default: fp64 DMMA / fp32 IEEE FFMA, TF32 never")+" math) and native GPU HH; no numerical atomics"}};}
   // Factorization receipt (C0 freeze). The first call per invocation snapshots the live counters,
   // freezes an immutable copy, then restarts the live counters for the validation phase. The
   // restart is enqueued AFTER stream.sync() (plus factor()'s trailing join/sync), so no producing
   // stream is still writing; later calls in the same invocation return the frozen copy even after
   // validation has run. Enforced for selected schedules (a stamped carrier the launcher refuses
   // already threw before modifying A); reported for qualification fixtures, whose forced stamps
   // may legitimately walk the fixture fallback menu.
   json panel_selected_vs_executed()const{
    json mism=json::array();int checked=0,count=0;
    for(const auto&pan:plan.panels){
     if(size_t(ctx.rank)>=pan.ranks.size())continue;const auto&batch=pan.ranks[ctx.rank];if(batch.cc<1)continue;
     const std::string key="p"+std::to_string(batch.col)+"r"+std::to_string(batch.rank);++checked;
     const int sw=(batch.csw>0&&batch.csw<batch.h)?batch.csw:0;
     const json want={{"groups",batch.cc},{"minipanel_width",batch.cpw},
      {"threads",batch.cthreads?batch.cthreads:coop_threads_for(plan.threads)},{"window",batch.cpw?sw:0}};
     auto it=cw.panel_batch_executed_detail.find(key);
     // A hand-picked QR-v2 factor changes the backend inside this exact
     // elimination, not the W/Z/D carriers or the scheduled members.
     if(gau_enabled()&&it!=cw.panel_batch_executed_detail.end()&&it->second.contains("qrv2")){
      const auto&q=it->second.at("qrv2");int rows=0;for(const auto&g:batch.ge)rows=std::max(rows,g.rows);
      if(q.value("h",0)==batch.h&&q.value("rows",0)==rows&&q.value("products",size_t(0))==batch.ge.size()&&
         it->second.value("groups",0)==1&&it->second.value("window",-1)==0)continue;
     }
     if(it==cw.panel_batch_executed_detail.end()||it->second!=want){++count;
      if(mism.size()<16)mism.push_back({{"batch",key},{"scheduled",want},{"executed",it==cw.panel_batch_executed_detail.end()?json(nullptr):it->second}});}
    }
    return {{"enforced",plan.profile_id!="qualification-only"},{"checked",checked},{"mismatch_count",count},{"mismatches",mism},{"pass",count==0}};
   }
   json evidence(){
    if(factor_frozen_)return frozen_factor_receipt_;
    stream.sync();auto w=witness.download()[0];
    json r=receipt(w,"factorization",invocation);
    r["panel_selected_vs_executed"]=panel_selected_vs_executed();
    frozen_factor_receipt_=r;frozen_invocation_=invocation;factor_frozen_=true;
    cw=CarrierWitness{};witness.zero(stream);
    exec_next=0;exec_elimination=0;if(exec_cap)exec_log.zero(stream);
    return r;}
   // Validation-phase receipt (C0): live counters since the factorization
   // freeze, i.e. validation-only when every factor snapshot froze first
   // (which the drivers do). A live read: never frozen, never reset here.
   json validation_evidence(){stream.sync();auto w=witness.download()[0];
    return receipt(w,"validation",factor_frozen_?frozen_invocation_:invocation);}
 ~TiledEngine(){if(host_status)cudaFreeHost(host_status);if(blas)cublasDestroy(blas);for(auto h:pipe_blas)if(h)cublasDestroy(h);
  for(auto&ps:peer_sets){for(auto h:ps.h)if(h)cublasDestroy(h);for(auto s:ps.s)if(s)cudaStreamDestroy(s);for(auto e:ps.join)if(e)cudaEventDestroy(e);if(ps.fork)cudaEventDestroy(ps.fork);}}
};
}
