#pragma once
#include "execution_evidence.hpp"
#include "common.hpp"
#include "p7_workspace.hpp"
#include "qrv2_workspace.hpp"
#include "execution_format.hpp"
#include "tiled_kernels.cuh"
#include "backend_budget.hpp"
namespace tqr {
// All three record formats share one version so old profiles/plans cannot silently qualify a
// changed binary.
inline constexpr const char* tiled_catalog=execution_catalog;
// The contraction replication c and the pipeline depth d are chosen by measurement among candidates above
// 1, like the other schedule parameters (b, leaf, strip, threads, radix, minipanel width). The three
// constants below are defaults (all >1), not the decision.
//  panel product   W and the scaled norms of the panel GE, [K] = matrix row
//  trailing apply  W = V^T X and Z = T^T W of Eq apply, [K] cut into c parts
//                  and summed by an additive Combine over disjoint sets.
//  depth           packets in flight inside one elimination (Eq pipe).
// D = VZ carries its own selected c like every other Eq-apply product (Alg 2): the peers accumulate
// into X in place, so the cut costs no output replica and only the measurement rules among c>1
// (v20+).
#ifndef TQR_FIXED_PANEL_C
#define TQR_FIXED_PANEL_C 128
#endif
#ifndef TQR_FIXED_APPLY_C
#define TQR_FIXED_APPLY_C 2
#endif
#ifndef TQR_FIXED_DEPTH
#define TQR_FIXED_DEPTH 2
#endif
inline constexpr int tqr_fixed_panel_c=TQR_FIXED_PANEL_C;
inline constexpr int tqr_fixed_apply_c=TQR_FIXED_APPLY_C;
inline constexpr int tqr_fixed_depth=TQR_FIXED_DEPTH;
// These are the DEFAULTS a record written before a field existed falls back to,
// not the menus. The menus live in tiled_engine.cuh and hold strictly >1
// candidates under the standing rule (only capacity may make c=1 or d=1);
// among them the selector weighs latency and bandwidth. The --c/--d/--zc/--dc
// forcing flags and the capacity fall-back are the only paths to 1.
static_assert(tqr_fixed_panel_c>1,"the panel carrier default must be a replicated carrier");
static_assert(tqr_fixed_apply_c>=1&&tqr_fixed_depth>=1,"carrier and depth defaults must be feasible");
// Explicit per-handle GPU BLAS workspace, a software catalog allocation.
inline constexpr size_t tiled_blas_workspace_bytes=size_t(32)*1024*1024;
// PHYSICAL EXECUTION EVIDENCE ARENA (execution_format 24). One sizeof(ExecInterval)-byte record
// per issued packet and per panel factor, so the log is charged to the budget
// like every other owned array rather than appearing behind the descriptor's
// back. 256Ki records is above the packet count of every admitted size on the
// standing list; an overflow is COUNTED as a dropped interval, never inferred.
inline constexpr size_t tiled_evidence_intervals=size_t(1)<<18;
inline constexpr size_t tiled_evidence_bytes=tiled_evidence_intervals*sizeof(ExecInterval);
// Descriptor bounds, named once so the validator and the menus cannot drift.
// The leaf bound was 16384, which is what made the panel-tall leaf -- and with
// it the cooperative panel carrier -- unreachable at every n above 16384.
inline constexpr int tiled_max_leaf=262144;
inline constexpr int tiled_max_strip=32768;
// late panel width. Panels that start at or after column late_col (= LATE_FRAC * min(m,n), rounded down to a multiple
// of b) use width LATE_B (a divisor of b and of the row block) instead of b: the elimination list keeps the same
// hierarchical GE/TT structure, only the tile width of the late (latency-bound) panels shrinks. Opt-in:
// TQR_MULTI_LATE_B=256 TQR_MULTI_LATE_FRAC=0.6. Both instantiate and inventory use this one rule.
inline int tiled_late_b(){static const int v=[]{const char*e=std::getenv("TQR_MULTI_LATE_B");return e?std::atoi(e):0;}();return v;}
inline double tiled_late_frac(){static const double v=[]{const char*e=std::getenv("TQR_MULTI_LATE_FRAC");return e?std::atof(e):0.6;}();return v;}
inline int tiled_panel_width(int b,int col,int k){const int lb=tiled_late_b();if(lb<=0||lb>=b||b%lb)return b;
 const int late_col=int(double(k)*tiled_late_frac())/b*b;return col>=late_col?lb:b;}
struct TiledInventory {std::vector<size_t> retained,owned;size_t max_batch=1;};
// Per-batch words of the staging frame that the carrier and the pipeline own: X is radix*b*strip per pipeline slot, W
// is apply_c*b*strip per slot, Z is b*strip per slot. UNCHANGED BY THE BLOCK-LEVEL CARRIER: the CB warp-groups hold
// their K_z partials in registers and sum them in shared scratch, so no HBM partial exists to charge. The admitted
// capacity therefore neither grows with CB nor constrains it; CB is selected on time alone. Flipped by
// -DTQR_CROSS_PANEL=1; the default stays false and bit-identical. With lookahead the engine owns a second lane group
// of d slots: lanes 0..d-1 (the main, high-priority stream group) factor the next panel(s) and apply their near
// windows, lanes d..2d-1 (default priority) carry window B -- the bulk trailing update of the previous elimination
// (or composed group) -- beside them. TQR_NEAR=0 disables.
inline int tiled_lane_slots(int depth,bool la=false){return std::max(1,depth)*(la?2:1)+((la&&near_enabled())?1:0);}
// aggregated far update over g consecutive eliminations. 1 = the unaggregated schedule).
// TQR_LOOKAHEAD=1 makes the development harnesses instantiate lookahead schedules.
inline int tiled_lookahead_env(){static const int v=[]{const char*e=std::getenv("TQR_LOOKAHEAD");return (e&&std::atoi(e)==1)?1:0;}();return v;}
// TQR_LA_FREE_SMS=N makes the development harnesses instantiate lookahead_free_sms = N.
inline int tiled_la_free_env(){static const int v=[]{const char*e=std::getenv("TQR_LA_FREE_SMS");return e?std::max(0,std::atoi(e)):0;}();return v;}
inline int tiled_aggregate_env(){static const int g=[]{const char*e=std::getenv("TQR_AGGREGATE");const int v=e?std::atoi(e):1;return (v>=2&&v<=8)?v:1;}();return g;}
// Alg 1's cross-panel Pipeline adds ONE lane group beyond the depth, so the
// engine owns depth+1 slots: lane 0 factors panel k+1 on the main stream while
// lanes 1..depth carry the bulk trailing update of panel k. Charging depth+1
// keeps the admitted capacity an upper bound on what the engine allocates.
inline size_t carrier_frame_words(int b,int strip,int radix,int depth,int arep,int zrep=1,bool la=false){
 // X (radix), the arep partials of W, and Z, per slot -- plus the device pointer array the carrier
 // needs when it cuts [K] over a BATCH of leaves (three pointers per peer, max(arep,zrep) peers per
 // batch member). Three device pointers per peer, charged in words at the smallest admitted word
 // size (4 bytes) so the charge is an upper bound at fp32 as well as fp64. Re-derive with the arm
 // if it ever ships.
 return size_t(tiled_lane_slots(depth,la))*
  (size_t(radix+arep+std::max(1,zrep))*size_t(b)*size_t(strip)
   + size_t(6)*size_t(std::max({1,arep,zrep})));
}
// extra owned words of the block-cyclic distribution on rank r (0 when contiguous): the engine's internal cyclic
// copy of the local rows (the factored storage and the native Q handle live there), plus the two staging slabs
// of one redistribution round (rows x tiled_redistribution_cols columns, send and receive).
inline int tiled_redistribution_cols(int n){return std::max(1,std::min(n,2048));}
inline size_t tiled_rowmap_extra_words(const RowMap&map,int r,int n,int pad){
 if(!map.cyclic())return 0;
 const size_t slab=size_t(row_begin(map.m,map.p,r+1)-row_begin(map.m,map.p,r)),cyc=size_t(map.local_rows(r));
 return checked_add(checked_mul(size_t(descriptor_ld(int(cyc),pad)),size_t(n)),checked_mul(2*std::max(slab,cyc),size_t(tiled_redistribution_cols(n))));}
inline size_t tiled_rowmap_metadata_bytes(const RowMap&map,int r){
 if(!map.cyclic())return 0;
 const size_t slab=std::max(1,row_begin(map.m,map.p,r+1)-row_begin(map.m,map.p,r));
 const size_t cyc=std::max(1,map.local_rows(r));
 return checked_mul(checked_add(4*checked_add(slab,cyc),size_t(8)*map.p),sizeof(int));
}
// Count the actual ragged tree without allocating its per-factor descriptors.
// This is the same ownership recurrence as tiled_instantiate, not a padded
// upper estimate that can reject a feasible, faster strip at capacity.
inline TiledInventory tiled_inventory(int m,int n,int p,int b,int leaf,int strip,size_t word,int pad=0,int radix=2,int depth=tqr_fixed_depth,int arep=tqr_fixed_apply_c,int gradix=2,int zrep=1,int drep=1,int agg=1,int la=0,int row_block=0){
 if(row_block<0||(row_block&&(p==1||row_block<b||row_block%b)))throw std::runtime_error("invalid_row_distribution");
 TiledInventory z;z.retained.resize(p);const RowMap map(m,p,row_block);
 for(int col=0;col<std::min(m,n);){int root=map.owner(col),h=std::min({tiled_panel_width(b,col,std::min(m,n)),std::min(m,n)-col,map.run_end(col)-col});std::vector<int> roots;
  for(int rank=0;rank<p;++rank){int rows=std::max(0,map.local_rows(rank)-map.lb(rank,col)),count=ceildiv(rows,leaf);if(!count)continue;
   roots.push_back(rank);z.max_batch=std::max(z.max_batch,size_t(count));
   // The local tree is k-ary; this mirrors the grouping in tiled_instantiate exactly, singletons
   // carried forward.
   {size_t nodes=size_t(count),internal=0;
    while(nodes>1){size_t next=0;for(size_t i=0;i<nodes;i+=size_t(radix)){size_t g=std::min(size_t(radix),nodes-i);if(g>1)++internal;++next;}nodes=next;}
    z.retained[rank]+=size_t(count)+internal;}
  }
  // The cross-rank tree is k-ary exactly as the local one is: gradix is the
 // across-GPU carrier's contraction replication c, selected from measurement, not fixed. This mirrors the
 // grouping in tiled_instantiate. The owner of the diagonal block leads the cross-rank tree (child 0 owns every
 // node; R lands in its rows). Contiguous slabs: ranks below the owner hold no rows here, so the owner is already
 // first.
 {auto it=std::find(roots.begin(),roots.end(),root);if(it!=roots.end())std::rotate(roots.begin(),it,roots.end());}
 while(roots.size()>1){std::vector<int> next;for(size_t i=0;i<roots.size();i+=size_t(gradix)){size_t g=std::min(size_t(gradix),roots.size()-i);next.push_back(roots[i]);if(g>1)++z.retained[roots[i]];}roots=std::move(next);}col+=h;
 }
 for(int r=0;r<p;++r){size_t nr=row_begin(m,p,r+1)-row_begin(m,p,r),matrix=checked_add(checked_mul(size_t(descriptor_ld(nr,pad)),size_t(n)),tiled_rowmap_extra_words(map,r,n,pad));
   size_t ts=checked_mul(z.retained[r],size_t(b)*b),frame=z.max_batch*((la?2:1)*std::max(size_t(leaf)*b,size_t(radix)*b*b)+size_t(tiled_lane_slots(depth,la))*size_t(radix)*b*strip+carrier_frame_words(b,strip,radix,depth,arep,zrep,la)),remote=2*size_t(b)*b+(size_t(std::max({1,arep,zrep}))+1)*size_t(b)*std::max(b,strip)+(drep>1?size_t(drep)*size_t(b)*std::max(b,strip):0)+2*size_t(gradix)*size_t(b)*b;
  frame=checked_add(frame,p7_gpu_partial_words(b,arep)+p7_aggregate_words(b,leaf,strip,radix,depth,arep,zrep,agg,z.max_batch,la!=0)+gmma_vr_words(word,depth,la,leaf,b,agg)+gmma_x3_words(word,depth,la,leaf,b,agg,strip));
  z.owned.push_back(checked_mul(checked_add(checked_add(matrix,ts),checked_add(frame,remote+size_t(n)+strip)),word)+z.retained[r]*(sizeof(uint64_t)+sizeof(TilePacket)+sizeof(TileMerge))+sizeof(Witness)+sizeof(int)+tiled_blas_workspace_bytes+tiled_evidence_bytes+tiled_rowmap_metadata_bytes(map,r));
  z.owned.back()=checked_add(z.owned.back(),checked_add(
   qrv2_workspace(leaf,b,z.max_batch,radix,gradix,arep).bytes(word),
   checked_mul(size_t(tiled_lane_slots(depth,la)-1),tiled_blas_workspace_bytes)));
 }return z;
}
// Zero cc means no carrier was priced for this batch and the block path runs.
struct TileBatch {int rank,col,h,leaf;std::vector<TilePacket> ge;std::vector<std::vector<TileMerge>> levels;int cc=0,cpw=0,cthreads=0,csw=0;
 double cblock_s=0,ccoop_s=0;size_t cshared_need=0,cshared_cap=0;};
struct TilePanel {int col,h;std::vector<TileBatch> ranks;std::vector<TileMerge> global;};
struct TiledPlan {
 int m=0,n=0,p=1,b=32,leaf=1024,strip=1024,threads=256,c=tqr_fixed_panel_c,d=tqr_fixed_depth,pad=0,radix=2,gradix=2,pw=0,cthreads=0,apply_c=tqr_fixed_apply_c;
 // aggregate g of the far update (schedule field; 1 = unaggregated).
 int agg=1;
 // Look-ahead depth across panels (0 = serial).
 int la=0;
 // SMs left free for the next unit's panel while look-ahead window B runs. A positive value caps the
 // persistent GMMA grid of window-B products at (device SMs - la_free) (KernelHardwareInfo::sm_count);
 // 0 = no cap. Only meaningful with la = 1.
 int la_free=0;
 // csw_forced = window forced by RunOptions.csw (0 = unforced). Replay restores csw
 // from the schedule and never reads forcing or summary fields.
 int csw_forced=0;
 // Forcing constraints recorded at selection time (RunOptions forced fields).
 // Carried for provenance; NEVER an input to replay (replay reads schedule).
 json forcing=json::object();
 // Digest of the authoritative schedule (tiled_schedule() dump). Stored at
 // selection; verified by every rank on broadcast before any factor runs.
 std::string schedule_digest;
 // Alg 2 gives Z and D their OWN carriers ("Z <- T^T W; D <- V_e Z with its own carrier").
 int zc=1,dc=1;
 // Until a fast carried kernel exists this never fires (carried loses every comparison); it exists
 // so v2 is selectable the moment it wins. Zero/false = not priced.
 bool z_priced=false,d_priced=false;
 double z_refused_s=0,z_taken_s=0,d_refused_s=0,d_taken_s=0;
 // THE BLOCK-LEVEL APPLY CARRIERS. Each CTA's CB warp-groups cut the product's own [K] (rows for W,
 // reflectors for Z and D), hold the partials in registers and sum them in shared memory. Selected
 // from measurement over block_rep_menu starting at the menu front, never at 1. They cost no HBM
 // partials, so they need no capacity fall-back: carrier_frame_words is unchanged by construction,
 // and the comment there says so. The GPU-level rep<=1 arm always runs them; the GPU-level c>1 arm
 // runs cuBLAS peers whose block geometry is vendor-internal.
 int wcb=2,zcb=2,dcb=2;
// The panel carrier's staged column window. 0 = whole panel (the unwindowed kernel).
 int csw=0;
 bool ge_shared=false,tt_shared=false,tt_sparse=false;
 size_t backend_arena_bytes=0,library_allowance_bytes=0;
 bool scalar=false,small_shared=false;
 std::vector<TilePanel> panels;std::vector<size_t> retained,owned;
 size_t packets=0,max_batch=1;std::string profile_id="qualification-only",key;json selection;
 json qr_schedule=json::object();
 // NUMBERS A REFUSAL MUST CARRY (execution_format 24 certificates). Filled by tiled_select, which
 // is where the measurements are. Zero means NOT PRICED, and a zero-priced refusal is a gate
 // failure, not a free pass. The narrowest strip the selector was willing to admit, and the best
 // predicted cost it found with r=1 against the best with r>1.
 int strip_floor_q=0,widest_packets=0;
 int row_block=0;
 double depth_price_r1_s=0,depth_price_rmulti_s=0;
 double tt_block_s=0,tt_coop_s=0;          // merge-stack elimination at its real stack shape
 double merge_fused_s=0,merge_carried_s=0; // fused pentagonal apply vs the carried W/Z/D sequence
 size_t budget_bytes=0;                    // the admission budget the selector actually used
 json record()const{json r_={{"execution_format",execution_format_version},{"catalog",tiled_catalog},{"m",m},{"n",n},{"active",p},{"b",b},{"leaf",leaf},{"strip",strip},{"threads",threads},{"radix",radix},{"global_radix",gradix},{"across_gpu_replication",gradix},{"padding_rows",pad},{"ge_shared",ge_shared},{"tt_shared",tt_shared},{"tt_sparse",tt_sparse},{"backend_arena_bytes_per_rank",backend_arena_bytes},{"library_allowance_bytes_per_rank",library_allowance_bytes},{"blas_workspace_bytes",scalar?0:tiled_blas_workspace_bytes},{"scalar",scalar},{"small_shared",small_shared},{"range_policy","global positive column scaling, GPU maximum via immutable paper copies, rescale only final R"},{"descriptor_lifetime",scalar?"kernel parameters and retained tau":"preuploaded per-factor arrays; no per-call H2D descriptor copy"},{"c",c},{"d",d},{"aggregate",agg},{"carrier_minipanel_width",pw},{"carrier_window",csw},{"carrier_window_forcing",csw_forced},{"apply_replication",apply_c},{"Z_replication",zc},{"D_replication",dc},{"W_block_carrier",wcb},{"Z_block_carrier",zcb},{"D_block_carrier",dcb},{"max_batch",max_batch},{"pipeline_lane_slots",tiled_lane_slots(d,la)},{"lookahead",la},{"lookahead_free_sms",la_free},{"cross_panel_pipeline",la!=0},{"Z_priced",z_priced},{"D_priced",d_priced},{"Z_refused_s",z_refused_s},{"Z_taken_s",z_taken_s},{"D_refused_s",d_refused_s},{"D_taken_s",d_taken_s},{"carrier_threads",cthreads},{"carrier_selection","CHOSEN carrier: contraction replication c and pipeline depth d are selected from measurement among strictly >1 candidates (standing rule: only capacity may make c=1 or d=1). Every product of Eq apply has its OWN c (Alg 2): apply_replication cuts [K]=rows of W, Z_replication cuts [K]=h of Z, D_replication cuts [K]=h of D in place. A c=1/d=1 plan arises only from the capacity fall-back or the --c/--d forcing flags (ablation arm); unreplicated products run native kernels, never cuBLAS"},{"panel_replication",c},{"pipeline_depth",d},{"c_d_source","panel c, apply c, Z and D replication and depth d are selected from measured latency and bandwidth over menus holding strictly >1 candidates. Sec 3 as revised: capacity alone may force c=1/d=1. The capacity fall-back and the ablation forcing flags are recorded in selection (capacity_fallback, forced values); every candidate >1 stays sampled, priced and recorded in selection.menu"},{"tree",scalar?"one native GE packet, ordered HH":"radix-selected local TT; radix-selected k-ary global TT (the across-GPU carrier)"},{"panels",panels.size()},{"packets",packets},{"refusal_prices",{{"TT_block_s",tt_block_s},{"TT_cooperative_GE_s",tt_coop_s},{"strip_floor_q",strip_floor_q},{"widest_packets",widest_packets},{"depth_price_r1_s",depth_price_r1_s},{"depth_price_rmulti_s",depth_price_rmulti_s},{"merge_fused_apply_s",merge_fused_s},{"merge_carried_apply_s",merge_carried_s},{"admission_budget_bytes",budget_bytes},{"basis","the two measured comparisons a refusal has to carry (Sec 3: capacity alone may force c=1; the standing rule admits otherwise only a refusal PRICED with numbers). The engine copies these into the execution witness certificates"}}},{"T_packets_per_rank",retained},{"owned_bytes_per_rank",owned},{"profile_id",profile_id},{"key",key},{"selection",selection},{"forcing",forcing},{"schedule_format",qr_schedule.value("schedule_format",0)},{"schedule_digest",schedule_digest},{"qr_schedule",qr_schedule},{"schedule","compact ordered panel loops; batched independent leaves and tree levels; completion before generation reuse"},{"output","global upper-trapezoid view plus ordered native full-Q handle; V aliases A"},{"cache","conditional service only; no owned L2"},{"physical_bound",nullptr}};
  r_["row_distribution"]={{"kind",row_block?"block_cyclic":"contiguous"},{"blk",row_block}};return r_;}
};
inline TiledPlan tiled_scalar_instantiate(int m,int n,int threads,size_t word,int pad=0){
 if(m<0||n<0||m>128||n>128||(threads!=1&&threads!=32&&threads!=128&&threads!=256))throw std::runtime_error("unsupported_scalar_packet");
 TiledPlan a;a.m=m;a.n=n;a.p=1;a.b=std::max(1,std::min(m,n));a.leaf=std::max(1,m);a.strip=std::max(1,n);a.threads=threads;a.pad=pad;a.scalar=true;a.retained={};a.retained.push_back(0);
 a.owned={checked_mul(checked_add(checked_mul(size_t(descriptor_ld(m,pad)),size_t(n)),size_t(std::min(m,n))),word)+sizeof(int)+sizeof(Witness)};
 a.key=digest(a.record().dump());return a;
}
inline TiledPlan tiled_instantiate(int m,int n,int p,int b,int leaf,int strip,int threads,size_t word,int pad=0,int radix=2,int depth=tqr_fixed_depth,int arep=tqr_fixed_apply_c,int gradix=2,int zrep=1,int drep=1,int wblock=2,int zblock=2,int dblock=2,int agg=1,int la=0,int row_block=0,bool allow_wide_multi=false){
 if(m<0||n<0||p<1||p>4||b<1||b>512||(b>128&&(!allow_wide_multi||p<2))||leaf<b||leaf>tiled_max_leaf||strip<1||strip>tiled_max_strip||(threads!=128&&threads!=256&&threads!=512&&threads!=1024)||pad<0||radix<2||radix>TILE_MAX_RADIX||(radix&(radix-1))||depth<1||depth>16||arep<1||arep>64||gradix<2||gradix>TILE_MAX_RADIX||zrep<1||zrep>64||drep<1||drep>64||(wblock!=2&&wblock!=4&&wblock!=8)||(zblock!=2&&zblock!=4&&zblock!=8)||(dblock!=2&&dblock!=4&&dblock!=8)||agg<1||agg>16||la<0||la>1)throw std::runtime_error("unsupported_tiled_descriptor");
 if(row_block<0||(row_block&&(p==1||row_block<b||row_block%b)))throw std::runtime_error("invalid_row_distribution");
 checked_mul(checked_mul(size_t(m),size_t(n)),word);
 TiledPlan a;a.m=m;a.n=n;a.p=p;a.b=b;a.leaf=leaf;a.strip=strip;a.threads=threads;a.pad=pad;a.radix=radix;a.gradix=gradix;a.d=depth;a.apply_c=arep;a.zc=zrep;a.dc=drep;a.wcb=wblock;a.zcb=zblock;a.dcb=dblock;a.agg=agg;a.la=la;a.retained.resize(p);
 a.widest_packets=std::min(m,n)>0?ceildiv(std::max(0,n-std::min(b,std::min(m,n))),strip):0;
 struct Rep{int row,rank,height,child;};
 auto combine=[&](const Rep*g,int k,int col,int h){TileMerge e{};e.k=k;e.col=col;e.h=h;e.tile=int(a.retained[g[0].rank]++);
  for(int i=0;i<k;++i){e.row[i]=g[i].row;e.rank[i]=g[i].rank;e.height[i]=g[i].height;e.child[i]=g[i].child;}
  ++a.packets;return e;};
 // The block is a whole number of panels (blk a multiple of b, at least b), so a panel's diagonal block never
 // crosses a block boundary. Production descriptors reject a misaligned block rather than rounding it.
 const RowMap map(m,p,row_block);a.row_block=map.cyclic()?map.blk:0;
 for(int col=0;col<std::min(m,n);){int root=map.owner(col),h=std::min({tiled_panel_width(b,col,std::min(m,n)),std::min(m,n)-col,map.run_end(col)-col});TilePanel pan;pan.col=col;pan.h=h;std::vector<Rep> roots;
  for(int rank=0;rank<p;++rank){TileBatch batch;batch.rank=rank;batch.col=col;batch.h=h;batch.leaf=leaf;std::vector<Rep> reps;
   // Local rows of this rank at or below the panel's first global row.
   const int begin=map.lb(rank,col),end=map.local_rows(rank);
   for(int l=begin;l<end;){int rows=std::min(leaf,end-l),hh=std::min(h,rows),id=int(a.retained[rank]++);batch.ge.push_back({l,col,rows,hh,id});reps.push_back({map.key(rank,l),rank,hh,id});++a.packets;l+=rows;}
   a.max_batch=std::max(a.max_batch,batch.ge.size());
   while(reps.size()>1){std::vector<Rep> next;std::vector<TileMerge> level;
    for(size_t i=0;i<reps.size();i+=size_t(radix)){size_t g=std::min(size_t(radix),reps.size()-i);
     if(g==1){next.push_back(reps[i]);continue;}
     auto e=combine(&reps[i],int(g),col,h);level.push_back(e);int total=0;for(size_t j=0;j<g;++j)total+=reps[i+j].height;
     next.push_back({e.row[0],rank,std::min(h,total),e.tile});}
    batch.levels.push_back(std::move(level));reps=std::move(next);}
   if(!reps.empty())roots.push_back(reps[0]);pan.ranks.push_back(std::move(batch));
  }
  // The owner of the diagonal block leads the cross-rank tree (see tiled_inventory).
  {auto it=std::find_if(roots.begin(),roots.end(),[&](const Rep&x){return x.rank==root;});if(it!=roots.end())std::rotate(roots.begin(),it,roots.end());}
  // The cross-rank tree is k-ary. Its arity IS the across-GPU carrier's contraction replication c:
  // a node of k children reduces W = sum_i V_i^T X_i over k DISJOINT row sets, which is {Combine:
  // additive} over disjoint K_z, and k is selected from measurement like every other carrier --
  // never fixed at 2 by the shape of the tree. gradix 2 reproduces the former pairwise tree node
  // for node.
  while(roots.size()>1){std::vector<Rep> next;for(size_t i=0;i<roots.size();i+=size_t(gradix)){size_t g=std::min(size_t(gradix),roots.size()-i);
    if(g==1){next.push_back(roots[i]);continue;}
    auto e=combine(&roots[i],int(g),col,h);pan.global.push_back(e);int total=0;for(size_t j=0;j<g;++j)total+=roots[i+j].height;
    next.push_back({e.row[0],e.rank[0],std::min(h,total),e.tile});}roots=std::move(next);}
  a.panels.push_back(std::move(pan));col+=h;
 }
 // Every term is simultaneously owned. A is caller storage; V aliases its
 // native-factor slots. BLAS and backend observed overhead is added at setup.
 for(int r=0;r<p;++r){size_t nr=row_begin(m,p,r+1)-row_begin(m,p,r),ld=descriptor_ld(nr,pad),batch=a.max_batch;
  size_t matrix=checked_add(checked_mul(ld,size_t(n)),tiled_rowmap_extra_words(map,r,n,pad)),ts=checked_mul(a.retained[r],size_t(b)*b);
  size_t frame=batch*((la?2:1)*std::max(size_t(leaf)*b,size_t(radix)*b*b)+size_t(tiled_lane_slots(depth,la))*size_t(radix)*b*strip+carrier_frame_words(b,strip,radix,depth,arep,zrep,la));
  frame=checked_add(frame,p7_gpu_partial_words(b,arep)+p7_aggregate_words(b,leaf,strip,radix,depth,arep,zrep,agg,batch,la!=0)+gmma_vr_words(word,depth,la,leaf,b,agg)+gmma_x3_words(word,depth,la,leaf,b,agg,strip));
  size_t remote=2*size_t(b)*b+(size_t(std::max({1,arep,zrep}))+1)*size_t(b)*std::max(b,strip)+(drep>1?size_t(drep)*size_t(b)*std::max(b,strip):0)+2*size_t(gradix)*size_t(b)*b,history=a.retained[r]*sizeof(uint64_t),metadata=a.retained[r]*(sizeof(TilePacket)+sizeof(TileMerge));
  size_t owned=checked_mul(checked_add(checked_add(matrix,ts),checked_add(frame,remote+size_t(n)+strip)),word)+history+metadata+sizeof(Witness)+sizeof(int)+tiled_blas_workspace_bytes+tiled_evidence_bytes+tiled_rowmap_metadata_bytes(map,r);
  owned=checked_add(owned,checked_add(qrv2_workspace(leaf,b,batch,radix,gradix,arep).bytes(word),
   checked_mul(size_t(tiled_lane_slots(depth,la)-1),tiled_blas_workspace_bytes)));
  a.owned.push_back(owned);
 }
  auto inventory=tiled_inventory(m,n,p,b,leaf,strip,word,pad,radix,depth,arep,gradix,zrep,drep,agg,la,row_block);if(inventory.retained!=a.retained||inventory.owned!=a.owned||inventory.max_batch!=a.max_batch)throw std::runtime_error("aggregate_descriptor_inventory_disagreement");
 a.key=digest(a.record().dump());return a;
}
inline void tiled_charge_backend(TiledPlan&plan,size_t word,const json&profile){
 if(plan.p==1)return;
 size_t capacity=std::max(size_t(plan.b)*plan.b,size_t(plan.b)*plan.strip)*word;
 size_t arena=backend_arena_budget(capacity,plan.p,profile),allowance=profile.value("library_overhead_allowance_bytes",size_t(0));
 for(auto&bytes:plan.owned)bytes=checked_add(bytes,checked_add(arena,allowance));
 plan.backend_arena_bytes=arena;plan.library_allowance_bytes=allowance;
}

}
