#pragma once
// Execution counters: per product kind, the carrier and replication each product ran with.
#include "common.hpp"
#include "execution_format.hpp"
#include "execution_evidence.hpp"
#include <map>
#include <array>
namespace tqr {
// CARRIER-DISPATCH WITNESS.
//
// The run reports, per product kind, how many products executed at each c; how many packets each
// elimination actually issued (the r of Eq pipe, which is what makes d>1 mean anything at all -- at
// r=1, C(d)=t_Sigma for every d); how many pipeline slots an elimination actually used; and every
// fall-back to c=1 ATTRIBUTED BY REASON.
//
// Issued counts are host observations. Completion is certified only after the existing successful
// stream/transport fence; it is not an independent device counter. Dispatch groups are neither API
// calls nor GPU launches. Unknown physical evidence is emitted as null and cannot satisfy
// qualification.
struct CarrierWitness {
 // The product kinds of one elimination. PANEL_GE and TT_GE are the two
 // elimination products (Alg 1's {GE} on a panel and {TT} on a merge stack);
 // APPLY_W/Z/D are Eq apply's three products (W = V^T X, Z = T^T W, D = V Z);
 // JOIN_W is the across-GPU W reduce, whose c is the arity of the global tree.
 enum Kind{PANEL_GE=0,TT_GE,APPLY_W,APPLY_Z,APPLY_D,JOIN_W,COMPOSE_G,KINDS};
 enum Level{L_BLOCK=0,L_CLUSTER,L_GPU,L_NODE,LEVELS};
 static_assert(KINDS==execution_product_kinds,"device/host product kind mismatch");
 static const char* level_name(int l){
  static const char*n[LEVELS]={"block","cluster","gpu","node"};return n[l];}
 static int primary_level(int k){return k==JOIN_W?L_NODE:L_GPU;}
 // Per-level evidence ADDS information. A block carrier never excuses an
 // uncertified GPU-level c=1. Output parallelism is not a cut of [K].
  // Why a product executed c=1. Standing rule: only capacity (or a shape
  // admitting no cut at all) may make c=1; every such product runs the native
  // kernels, never cuBLAS. Each reason below is a distinct, fixable cause.
  enum Reason{
   R_APPLY_C_LE_1=0,
   R_COUNT_NE_1,
   R_ROWS_LT_C,        // rows<c: the row index admits no c disjoint nonempty parts
   R_ZC_LE_1,
   R_DC_LE_1,
   R_H_LT_CZ,          // h < zc: degenerate reflector count admitting no Z cut
   R_H_LT_CD,          // h < dc: degenerate reflector count admitting no D cut
  R_FUSED_MERGE,      // the fused pentagonal merge apply does W/Z/D in one kernel
  R_PANEL_CC_ZERO,
  R_PANEL_ROWS_MIN,
  R_PANEL_H_NE_B,     // the panel is narrower than b (ragged tail)
  R_PANEL_B_MAX,      // b > coop_max_b
  R_PANEL_NOT_FULL,   // a ragged short leaf: no single packet spans the panel
  R_PANEL_MULTI,      // count>1: a local tree is still present
  R_PANEL_LAUNCH,     // every menu entry was refused by the launcher
  R_TT_BLOCK_PATH,    // a merge-stack elimination goes straight to the block GE
  R_JOIN_SINGLE,      // unreachable: singleton tree nodes are carried forward
  R_GLOBAL_W, R_GLOBAL_Z, R_GLOBAL_D, // unreachable: global products use apply_stage_*
  R_IDENTITY_W, R_IDENTITY_D, R_COMPOSE_UNCARRIED,
  R_D_SLICE_MIN_H,    // h >= c but below the native sliced D's minimum (every warp slice non-empty)
  R_Z_SLICE_MIN_H,    // Same for the native Z
  R_W_SLICE_MIN_ROWS, // rows >= c but below the native W's minimum (every cluster CTA a full k-tile)
  REASONS};
 static const char* kind_name(int k){
  static const char*n[KINDS]={"panel_GE","TT_GE","apply_W","apply_Z","apply_D","join_W","compose_G"};return n[k];}
 static const char* reason_name(int r){
  static const char*n[REASONS]={"apply_c_le_1","count_ne_1","rows_lt_c",
   "Z_replication_1_selected","D_replication_1_selected","h_lt_Z_c","h_lt_D_c","fused_merge_apply","panel_cc_zero",
   "panel_rows_lt_coop_min","panel_h_ne_b","panel_b_gt_coop_max",
   "panel_not_full_leaf","panel_count_gt_1","panel_launcher_refused",
   "TT_merge_stack_block_path","join_single_participant",
   "global_W_carrier_missing","global_Z_carrier_missing","global_D_carrier_missing",
   "global_W_identity_copy","global_D_identity_update","compose_G_unreplicated",
   "native_D_slice_min_h","native_Z_slice_min_h","native_W_slice_min_rows"};return n[r];}
  // Legacy at_c remains the GPU (join_W: node) view.
  std::array<std::array<std::map<int,uint64_t>,LEVELS>,KINDS> products{};
  std::array<std::map<std::string,json>,KINDS> block_partitions{};
  std::array<std::map<std::string,json>,KINDS> qrv2_factors{};
  std::array<uint64_t,KINDS> device_qrv2_products{},device_qrv2_owners{};
  const std::map<int,uint64_t>& primary_products(int k)const{return products[k][primary_level(k)];}
  // One group can contain several products, API calls, and GPU kernels.
  std::array<uint64_t,KINDS> dispatch_groups{}, completed{};
  std::array<uint64_t,REASONS> refusals{};
  // The selector's chosen (pi,pj,c) per product kind and where it was weighed (measurement
  // provenance). Zero means "no selection recorded" (numerical fixtures instantiating directly) and
  // disables the bound rather than passing it vacuously.
  std::array<int,KINDS> selected_pi{}, selected_pj{}, selected_c{};
  std::array<std::string,KINDS> selected_provenance{};
  void note_selected(int k,int pi,int pj,int c,const char* provenance){
   if(k<0||k>=KINDS||pi<1||pj<1||c<1||!provenance||!*provenance)throw std::runtime_error("invalid_carrier_selection");
   selected_pi[k]=pi;selected_pj[k]=pj;selected_c[k]=c;selected_provenance[k]=provenance;}
  uint64_t launch_attempts=0, launch_issued=0;
  std::map<std::string,uint64_t> launch_rejected;
  // Selection/launch-time replacements before any launch (menu-entry refusals, dry-check denials).
  // The engine emits only pre-launch keys; the fault-injection suite proves a post-mutation key
  // would fail.
  uint64_t preflight_substitutions=0;
  std::map<std::string,uint64_t> preflight_by_reason;
  void note_preflight(const char* reason,uint64_t n=1){
   if(!reason||!*reason||!n)throw std::runtime_error("invalid_preflight_note");
   preflight_substitutions+=n;preflight_by_reason[reason]+=n;}
  // NATIVE partials are GPU-level peer partials produced by native kernels that report their own
  // executed peers to device_peer_partials (unreplicated singletons, one per member; cooperative
  // panel groups, groups per member) and must reconcile EXACTLY. LIBRARY partials run inside the
  // cuBLAS/transport intermediary (vendor-internal block execution); they are counted here and
  // labeled unknown in the receipt, never reconciled against a device count that does not exist.
  // SINGLETON no-ops (join nodes with one participant, carried forward with no arithmetic) are not
  // partials at all. BLOCK-level peers live one level down, in block_native_peers vs
  // device_block_peers, and the two levels are never equated (nested carriers have different
  // multiplicities).
  std::array<uint64_t,KINDS> native_partials{}, library_partials{};
  std::array<uint64_t,KINDS> unobserved_partials{};
  uint64_t singleton_noops=0;
  std::array<uint64_t,KINDS> block_native_peers{};
  void note_native_peers(int k,int peers,uint64_t n=1){
   if(k<0||k>=KINDS||peers<1||!n)throw std::runtime_error("invalid_native_peer_note");
   native_partials[k]+=uint64_t(peers)*n;}
  void note_block_peers(int k,int peers,uint64_t n=1){
   if(k<0||k>=KINDS||peers<1||!n)throw std::runtime_error("invalid_block_peer_note");
   block_native_peers[k]+=uint64_t(peers)*n;}
  void note_membership(uint64_t n=1){
   if(!n)throw std::runtime_error("invalid_membership_note");
   membership_expected+=n;}
  uint64_t membership_expected=0;
  // Native: one X-write per uncarried-D member (tile_carried_D slice). Library: carried D performs
  // drep SEQUENTIAL beta=1 writes per member -- counted as drep, not one, because that is what
  // runs. Unobserved: fused/identity D writes with no device counter yet. Production requires
  // library==0 and unobserved==0.
  uint64_t native_commits_issued=0, library_commits_issued=0, unobserved_commits_issued=0;
  // The per-member side lives in combines_logical (one logical combine per batch member, +=n per
  // issue) and combine_output_tiles (one output tile per member written by the combine, +=n per
  // issue). The three are recorded separately so a launch/member/tiling change can never be
  // repaired by redefining another counter: combines_logical reconciles against carried
  // APPLY_W+APPLY_Z products, combines_expected against device_combines.
  uint64_t combines_expected=0, combines_logical=0, combine_output_tiles=0;
  // Extra member-executions inside one logical product (windowed cooperative minis: nwin launches
  // per member). Recorded separately from products so logical products, peer partials and kernel
  // launches stay distinct; native_partials carries their peers.
  std::array<uint64_t,KINDS> panel_suboperations{};
  // Peer partials carried by those suboperations (windowed minis launch
  // nwin times per member): the partial-conservation law reconciles
  // native+library against sum(c*n) over products PLUS these peers.
  std::array<uint64_t,KINDS> subop_peers{};
  void note_suboperations(int k,int c,uint64_t n=1){
   if(k<0||k>=KINDS||c<1)throw std::runtime_error("invalid_suboperation_note");
   if(n){panel_suboperations[k]+=n;native_partials[k]+=uint64_t(c)*n;subop_peers[k]+=uint64_t(c)*n;}}
  // per-batch executed panel carrier. Key "p<col>r<rank>" -> executed groups. Filled by
  // TiledEngine::cooperative_ge_batch at dispatch; checked by tiled_check_selected_vs_executed()
  // and by carrier_gate.py.
  std::map<std::string,int> panel_batch_executed;
  void note_panel_batch_executed(const std::string& key,int groups){
   if(key.empty()||groups<1)throw std::runtime_error("invalid_panel_batch_note");
   auto it=panel_batch_executed.find(key);
   if(it==panel_batch_executed.end())panel_batch_executed[key]=groups;
   else if(it->second!=groups)throw std::runtime_error("panel_batch_reexecuted_with_different_carrier");}
  std::map<std::string,json> panel_batch_executed_detail;
  void note_panel_batch_detail(const std::string& key,int groups,int pw,int threads,int window){
   if(key.empty()||groups<1)throw std::runtime_error("invalid_panel_batch_detail");
   json d={{"groups",groups},{"minipanel_width",pw},{"threads",threads},{"window",window}};
   auto it=panel_batch_executed_detail.find(key);
   if(it==panel_batch_executed_detail.end())panel_batch_executed_detail[key]=d;
   else if(it->second!=d)throw std::runtime_error("panel_batch_reexecuted_with_different_carrier_detail");}
  // Eq pipe's r, per elimination, and the slots the elimination actually used.
 std::map<int,uint64_t> packets_per_elimination,slots_per_elimination;
 // A c that does not divide `rows` leaves a remainder handled by one extra unreplicated peer.
 uint64_t remainder_peers=0, empty_eliminations=0;
 // composed Applies of g consecutive eliminations to far columns; each composed strip is ONE X
 // commit that advances g history cursors (one owner commit per constituent).
 uint64_t aggregated_applies=0,aggregated_strips=0,aggregated_constituent_layers=0,aggregated_compose_G=0;std::map<int,uint64_t> aggregate_g;
 uint64_t global_owner_packets=0, global_peer_packets=0, transport_only_packets=0;
 bool global_column_split=false,global_pipeline=false;
 std::map<std::string,uint64_t> global_column_slices;
 void note_global_columns(int members,int member,int columns,int first,int width){
  if(members<1||member<0||member>=members||width<1||first<0||first+width>columns)
   throw std::runtime_error("invalid_global_column_slice");
  // Each record is a Z output slice (pi=1,pj=members,c=1 at node
  // level) and a W combine over c=members disjoint row contributions.
  std::string key=std::to_string(members)+":"+std::to_string(member)+":"+std::to_string(columns)+":"+std::to_string(first)+":"+std::to_string(width);
  ++global_column_slices[key];
 }
 std::map<std::string,uint64_t> depth_refusals;
 // CERTIFICATES. There is deliberately no third basis: a c=1 that is neither capacity-bound nor
 // structurally priced is an implementation gap and stays a gate failure.
 //
 //  capacity   bytes_required (the smallest admitted c>1, in the memory where
 //  structural the contraction index admits no c disjoint nonempty parts:
 //             contraction < smallest_admitted_c, checked as a number.
 //             are carried, so the refusal is a result and not an opinion.
  struct Certificate {
   int kind=-1,reason=-1;
   const char* basis="";            // "capacity" | "structural" | "priced"
   int rows=0,h=0,q=0,count=0;
   int contraction=0,smallest_admitted_c=0;
   uint64_t bytes_required=0,bytes_available=0;
   const char* memory="";
   double refused_s=std::numeric_limits<double>::quiet_NaN();
   double taken_s=std::numeric_limits<double>::quiet_NaN();
   const char* statement="";
   uint64_t products=0;
   int attempted_c=1, actual_c=1;
   bool searched_smaller=false;
   std::string attempted_desc, actual_desc, proof;
   std::string missing;
   json record()const{
    json j={{"kind",kind<0?json(nullptr):json(kind_name(kind))},
            {"reason",reason<0?json(nullptr):json(reason_name(reason))},
            {"basis",basis},{"products",products},
            {"shape",{{"rows",rows},{"h",h},{"q",q},{"count",count}}},
            {"contraction",contraction},{"smallest_admitted_c",smallest_admitted_c},
            {"statement",statement},
            {"attempted_c",attempted_c},{"actual_c",actual_c},
            {"attempted",attempted_desc.empty()
              ?std::string("c=")+std::to_string(attempted_c)+" cut refused"
              :attempted_desc},
            {"actual",actual_desc.empty()
              ?(reason<0?std::string("c=1 native (no reason)")
                        :std::string("c=1 native (")+reason_name(reason)+")")
              :actual_desc},
            {"searched_smaller",searched_smaller},
            {"proof",proof.empty()?json(nullptr):json(proof)}};
    if(std::string(basis)=="capacity"){j["bytes_required"]=bytes_required;j["bytes_available"]=bytes_available;j["memory"]=memory;}
    if(std::string(basis)=="unimplemented")j["missing"]=missing;
    if(std::string(basis)=="priced"){j["refused_s"]=std::isfinite(refused_s)?json(refused_s):json(nullptr);
                                     j["taken_s"]=std::isfinite(taken_s)?json(taken_s):json(nullptr);}
    return j;}
  };
 // Keyed so one certificate covers a whole refusal population of the same
 // cause and shape class; `products` accumulates the population it covers.
 std::map<std::string,Certificate> certificates;
  void certify(Certificate c,uint64_t n=1){
   // A run with no recorded selection (numerical fixture) keeps attempted_c=1 rather than inventing
   // one.
   if(c.kind>=0&&c.kind<KINDS&&selected_c[c.kind]>1)c.attempted_c=selected_c[c.kind];
   c.actual_c=1;
   std::string key=std::to_string(c.kind)+"/"+std::to_string(c.reason)+"/"+c.basis+"/"+
    std::to_string(c.rows)+"x"+std::to_string(c.h)+"x"+std::to_string(c.q)+"x"+std::to_string(c.count);
   auto it=certificates.find(key);
   if(it==certificates.end()){c.products=n;certificates.emplace(key,c);}
   else it->second.products+=n;
  }
 // Depth certificates use the same machine: an elimination that issued one
 // packet, or used one slot, has to say with a number why.
 std::map<std::string,Certificate> depth_certificates;
 void certify_depth(Certificate c,uint64_t n=1){
  std::string key=std::string(c.basis)+"/"+std::to_string(c.reason)+"/"+c.statement;
  auto it=depth_certificates.find(key);
  if(it==depth_certificates.end()){c.products=n;depth_certificates.emplace(key,c);}
  else it->second.products+=n;
 }
  // Filled by the engine from the device log before record() is called.
  unsigned long long device_products[KINDS]={};
  unsigned long long credits_max=0,evidence_dropped=0;
  unsigned long long device_peer_partials[KINDS]={};
  unsigned long long device_combines=0,device_physical_commits=0,device_history_marks=0,device_membership_reports=0;
  unsigned long long device_block_peers[KINDS]={};
  unsigned long long device_level_c[KINDS][execution_levels][execution_c_buckets]={};
  ExecSummary exec{};
  bool exec_present=false;

 void complete_after_fence(){
  for(int k=0;k<KINDS;++k){completed[k]=0;for(const auto&e:primary_products(k))completed[k]+=e.second;}
 }

  void product_at(int k,int level,int c,uint64_t n=1){
   if(k<0||k>=KINDS||level<0||level>=LEVELS||c<1||!n)throw std::runtime_error("invalid_product_witness");
   products[k][level][c]+=n;
  }
  // path: 'N' native kernels (device-reported peers), 'L' library
  // intermediary (cuBLAS peers, vendor-internal), 'T' transport collective.
  // Native GPU peers for 'N' products are added HERE (c per product); native
  // peers for unreplicated products are added in product_uncarried, except
  // the device-silent paths below. A product() call never invents device
  // evidence beyond these explicit rules.
  void product(int k,int c,uint64_t n=1,char path='L'){
   if(path!='N'&&path!='L'&&path!='T')throw std::runtime_error("invalid_product_path");
   product_at(k,primary_level(k),c,n);++dispatch_groups[k];
   launch_attempts+=n;launch_issued+=n;
    if(path=='N')native_partials[k]+=uint64_t(c)*n;
    if(path=='L'||path=='T')library_partials[k]+=uint64_t(c)*n;
    // One combine LAUNCH per carried W/Z issue over the whole batch (the
    // kernel grid already spans count*sw); logical products above keep
    // counting batch members, so the launch/member distinction survives.
    if((k==APPLY_W||k==APPLY_Z)&&c>1){combines_expected+=1;combines_logical+=n;combine_output_tiles+=n;}
    // compose_G now uses the same carried W kernel and additive-combine counter.
    // Keep the W/Z logical counters scoped to Apply products.
    if(k==COMPOSE_G&&c>1)combines_expected+=1;
   // The LIBRARY carried D performs c SEQUENTIAL beta=1 writes into X per member, not one owner
   // commit: counted as c, because that is what runs. The NATIVE carried D (path 'N',
   // include/carrier_gemm.cuh) commits once per member; its commit is recorded by the caller as a
   // native commit and reconciled with the device's physical-commit counter.
   if(k==APPLY_D&&c>1&&path=='L')library_commits_issued+=uint64_t(c)*n;
  }
 void block_carrier_geometry(int k,int peers,int bx,int by,int bz,const char*axis,
   const char*index,const char*partition,const char*combine,uint64_t n=1){
  if(bx<1||by<1||bz<1||!axis||!index||!*index||!partition||!*partition||!combine||!*combine)
   throw std::runtime_error("invalid_block_partition");
  const std::string a=axis;
  const int derived=a=="x"?bx:(a=="y"?by:(a=="z"?bz:(a=="serial"?1:(a=="warp_lane"&&bx%32==0?32:0))));
  if(peers!=derived)throw std::runtime_error("block_carrier_geometry_mismatch");
  json proof={{"c",peers},{"geometry",{{"block_dim",{bx,by,bz}},{"peer_axis",axis}}},
   {"index",index},{"partition",partition},{"combine",combine}};
  std::string key=proof.dump();auto it=block_partitions[k].find(key);
  if(it==block_partitions[k].end()){proof["products"]=n;block_partitions[k].emplace(key,std::move(proof));}
  else it->second["products"]=it->second["products"].get<uint64_t>()+n;
  product_at(k,L_BLOCK,peers,n);
 }
 void block_carrier(int k,int peers,const char*index,const char*combine,uint64_t n=1){
  block_carrier_geometry(k,peers,peers,1,1,"x",index,
   "K_z = { z + t * blockDim.x | t >= 0 } intersect [K]",combine,n);
 }
 // Factor carriers retain c=1 in the GPU contraction histogram: their
 // parallelism is a Split of [J], with each formed reflector replicated to
 // later column owners. Do not relabel p_j as a contraction replication.
 // product(k,1,n,'N') is issued by the caller; this records its extra owners.
 json qrv2_factor(int k,bool sm1,int rows,int h,int columns,int threads,uint64_t n){
  if((k!=PANEL_GE&&k!=TT_GE)||rows<h||h<1||columns<1||threads<32||threads%32||!n)
   throw std::runtime_error("invalid_qrv2_factor_carrier");
  const int owners=ceildiv(h,columns),gpu_peers=sm1?1:owners,bc=sm1?32:threads;
  json proof={{"backend",sm1?"gau_sm1":"gau_owner_cta"},{"rows",rows},{"h",h},
   {"columns_per_owner",columns},{"owners",owners},{"gpu_peers",gpu_peers},
   {"block_c",bc},{"threads",threads},{"gpu_c",1},
   {"split_index","J"},{"replicated_operand","formed Householder reflector"},
   {"column_partition","J_y = [y*columns_per_owner,min(h,(y+1)*columns_per_owner))"}};
  const std::string key=proof.dump();auto it=qrv2_factors[k].find(key);
  if(it==qrv2_factors[k].end()){auto p=proof;p["products"]=n;qrv2_factors[k].emplace(key,std::move(p));}
  else it->second["products"]=it->second["products"].get<uint64_t>()+n;
  if(gpu_peers>1)note_native_peers(k,gpu_peers-1,n);
  note_membership(uint64_t(gpu_peers)*n);note_block_peers(k,bc*owners,n);
  if(sm1)block_carrier_geometry(k,32,threads,1,1,"warp_lane","rows within each column-owner warp",
   "K_z = { z + 32*t | t >= 0 } intersect [K], z = threadIdx.x mod 32",
   "recursive-halving warp shuffle, fixed order; T by in-chain dlarft",n);
  else block_carrier(k,threads,"rows within each column-owner CTA","warp shuffle then shared scratch, fixed order",n);
  levels(k,bc,1,1,1,n,true);
  proof["products"]=n;return proof;
 }
  // Key "b<c>.cl<c>.g<c>.n<c>" ("b?" = vendor-internal block geometry); ".unobserved" marks paths
  // whose kernels do not report to the device. Observed decompositions, projected per level, must
  // equal the device histogram exactly (carrier_gate.py).
  std::array<std::map<std::string,uint64_t>,KINDS> decomposition{};
  json device_levels_json(int k)const{
   static const char*lab[execution_c_buckets]={"1","2","3","4","8","16","32","64","128","other"};
   json out=json::object();
   for(int l=0;l<execution_levels;++l){json h=json::object();for(int b=0;b<execution_c_buckets;++b)if(device_level_c[k][l][b])h[lab[b]]=device_level_c[k][l][b];out[level_name(l)]=h;}
   return out;}
  static std::string decomposition_key(int b,int cl,int g,int nd,bool observed){
   return "b"+(b?std::to_string(b):std::string("?"))+".cl"+std::to_string(cl)+".g"+std::to_string(g)+".n"+std::to_string(nd)+(observed?"":".unobserved");}
  void levels(int k,int b,int cl,int g,int nd,uint64_t n,bool observed){
   if(k<0||k>=KINDS||b<0||cl<1||g<1||nd<1||!n||(observed&&b<1))throw std::runtime_error("invalid_level_decomposition");
   decomposition[k][decomposition_key(b,cl,g,nd,observed)]+=n;product_at(k,L_CLUSTER,cl,n);}
  void refuse(int r,uint64_t n=1){refusals[r]+=n;}
  // One product at c=1 WITH its reason. The only way to record an unreplicated product, so a silent
  // fall-back cannot be written. GPU-level native partials (one singleton per member) are added
  // here for every reason whose kernels report to the device; the device-silent paths (fused merge,
  // identity copies, compose -- kernels without counters yet) go to unobserved_partials, and
  // carried-forward singleton joins (no arithmetic at all) to singleton_noops. Uncarried D writes
  // to X through the fused or identity kernels are likewise unobserved commits, never native ones.
  void product_uncarried(int k,int r,uint64_t n=1){
   product(k,1,n,'N');refuse(r,n);launch_rejected[reason_name(r)]+=n;
   if(r==R_FUSED_MERGE||r==R_IDENTITY_W||r==R_IDENTITY_D||r==R_COMPOSE_UNCARRIED){
    native_partials[k]-=n;unobserved_partials[k]+=n;
   }else if(r==R_JOIN_SINGLE){
    native_partials[k]-=n;singleton_noops+=n;
   }
   if(k==APPLY_D&&(r==R_FUSED_MERGE||r==R_IDENTITY_D))unobserved_commits_issued+=n;
  }
 void elimination(int packets,int slots,uint64_t count=1,const char* reason=nullptr){
  if(packets==0){empty_eliminations+=count;return;}
  if(packets<0||slots<1||slots>packets)throw std::runtime_error("invalid_elimination_witness");
  packets_per_elimination[packets]+=count;slots_per_elimination[slots]+=count;
  if(slots==1){if(!reason)throw std::runtime_error("missing_depth_refusal");depth_refusals[reason]+=count;}
 }
  void merge(const CarrierWitness&o){
   for(int k=0;k<KINDS;++k){
    for(int l=0;l<LEVELS;++l)for(auto&e:o.products[k][l])products[k][l][e.first]+=e.second;
    for(const auto&e:o.block_partitions[k]){auto it=block_partitions[k].find(e.first);
     if(it==block_partitions[k].end())block_partitions[k].insert(e);
     else it->second["products"]=it->second["products"].get<uint64_t>()+e.second["products"].get<uint64_t>();}
    for(const auto&e:o.qrv2_factors[k]){auto it=qrv2_factors[k].find(e.first);
     if(it==qrv2_factors[k].end())qrv2_factors[k].insert(e);
     else it->second["products"]=it->second["products"].get<uint64_t>()+e.second["products"].get<uint64_t>();}
    device_qrv2_products[k]+=o.device_qrv2_products[k];device_qrv2_owners[k]+=o.device_qrv2_owners[k];
    dispatch_groups[k]+=o.dispatch_groups[k];completed[k]+=o.completed[k];
    native_partials[k]+=o.native_partials[k];library_partials[k]+=o.library_partials[k];
    unobserved_partials[k]+=o.unobserved_partials[k];block_native_peers[k]+=o.block_native_peers[k];
    panel_suboperations[k]+=o.panel_suboperations[k];subop_peers[k]+=o.subop_peers[k];
    // Selections must agree across merged scopes; divergent selections mean
    // two ranks priced different carriers (the family_restore divergence
    // class), which must throw rather than average.
    if(selected_c[k]&&o.selected_c[k]&&
       (selected_c[k]!=o.selected_c[k]||selected_pi[k]!=o.selected_pi[k]||selected_pj[k]!=o.selected_pj[k]))
     throw std::runtime_error("merged_carrier_selection_divergence");
    if(!selected_c[k]&&o.selected_c[k]){selected_pi[k]=o.selected_pi[k];selected_pj[k]=o.selected_pj[k];
     selected_c[k]=o.selected_c[k];selected_provenance[k]=o.selected_provenance[k];}
    device_products[k]+=o.device_products[k];device_peer_partials[k]+=o.device_peer_partials[k];
    for(auto&e:o.decomposition[k])decomposition[k][e.first]+=e.second;
    for(int l=0;l<execution_levels;++l)for(int b=0;b<execution_c_buckets;++b)device_level_c[k][l][b]+=o.device_level_c[k][l][b];}
   for(int r=0;r<REASONS;++r)refusals[r]+=o.refusals[r];
   launch_attempts+=o.launch_attempts;launch_issued+=o.launch_issued;
   for(auto&e:o.launch_rejected)launch_rejected[e.first]+=e.second;
   preflight_substitutions+=o.preflight_substitutions;
   for(auto&e:o.preflight_by_reason)preflight_by_reason[e.first]+=e.second;
   membership_expected+=o.membership_expected;
   for(auto&e:o.panel_batch_executed){
    auto it=panel_batch_executed.find(e.first);
    if(it==panel_batch_executed.end())panel_batch_executed[e.first]=e.second;
    else if(it->second!=e.second)throw std::runtime_error("panel_batch_reexecuted_with_different_carrier");}
   for(auto&e:o.panel_batch_executed_detail){
    auto it=panel_batch_executed_detail.find(e.first);
    if(it==panel_batch_executed_detail.end())panel_batch_executed_detail[e.first]=e.second;
    else if(it->second!=e.second)throw std::runtime_error("panel_batch_reexecuted_with_different_carrier_detail");}
   singleton_noops+=o.singleton_noops;
    native_commits_issued+=o.native_commits_issued;library_commits_issued+=o.library_commits_issued;
    unobserved_commits_issued+=o.unobserved_commits_issued;combines_expected+=o.combines_expected;
    combines_logical+=o.combines_logical;combine_output_tiles+=o.combine_output_tiles;
  for(int r=0;r<REASONS;++r)refusals[r]+=o.refusals[r];
  for(auto&e:o.packets_per_elimination)packets_per_elimination[e.first]+=e.second;
  for(auto&e:o.slots_per_elimination)slots_per_elimination[e.first]+=e.second;
  remainder_peers+=o.remainder_peers;empty_eliminations+=o.empty_eliminations;
  aggregated_applies+=o.aggregated_applies;aggregated_compose_G+=o.aggregated_compose_G;aggregated_constituent_layers+=o.aggregated_constituent_layers;aggregated_strips+=o.aggregated_strips;for(auto&e:o.aggregate_g)aggregate_g[e.first]+=e.second;
  global_owner_packets+=o.global_owner_packets;global_peer_packets+=o.global_peer_packets;
  transport_only_packets+=o.transport_only_packets;
  for(const auto&e:o.depth_refusals)depth_refusals[e.first]+=e.second;
  for(const auto&e:o.certificates){auto it=certificates.find(e.first);if(it==certificates.end())certificates.insert(e);else it->second.products+=e.second.products;}
   for(const auto&e:o.depth_certificates){auto it=depth_certificates.find(e.first);if(it==depth_certificates.end())depth_certificates.insert(e);else it->second.products+=e.second.products;}
   device_combines+=o.device_combines;device_physical_commits+=o.device_physical_commits;
   device_history_marks+=o.device_history_marks;device_membership_reports+=o.device_membership_reports;
   credits_max=std::max(credits_max,o.credits_max);evidence_dropped+=o.evidence_dropped;
  }
 static json histogram(const std::map<int,uint64_t>&h){json j=json::object();for(auto&e:h)j[std::to_string(e.first)]=e.second;return j;}
 json record()const{
  json per=json::object();uint64_t carried=0,uncarried=0;
  for(int k=0;k<KINDS;++k){
   uint64_t tot=0,rep=0,maxc=0;
   for(auto&e:primary_products(k)){tot+=e.second;if(e.first>1){rep+=e.second;maxc=std::max<uint64_t>(maxc,uint64_t(e.first));}}
   carried+=rep;uncarried+=tot-rep;
    per[kind_name(k)]={{"at_c",histogram(primary_products(k))},{"products",tot},{"issued_products",tot},{"completed_after_fence",completed[k]},{"dispatch_groups",dispatch_groups[k]},
     {"api_calls",nullptr},{"gpu_launches",nullptr},
     // Written by kernels that run, on the packet's own stream, AFTER the
     // products they count. Stream order is the certificate: the counter
     // cannot be incremented by work the device has not finished.
     {"device_completed_products",exec_present?json(device_products[k]):json(nullptr)},
     {"carried_products",rep},{"carried_fraction",tot?double(rep)/double(tot):0.0},{"max_c",maxc},
     // Zero selected means no selection recorded (fixture path) and disables the bound.
     {"selected_pi",selected_c[k]?json(selected_pi[k]):json(nullptr)},
     {"selected_pj",selected_c[k]?json(selected_pj[k]):json(nullptr)},
     {"selected_c",selected_c[k]?json(selected_c[k]):json(nullptr)},
     {"selection_provenance",selected_c[k]?json(selected_provenance[k]):json(nullptr)},
     {"native_partials_issued",native_partials[k]},
     {"library_partials_issued",library_partials[k]},
     {"unobserved_partials_issued",unobserved_partials[k]},
     {"block_native_peers_issued",block_native_peers[k]},
     {"panel_suboperations",panel_suboperations[k]},
     {"subop_peers",subop_peers[k]},
     {"device_block_peers",exec_present?json(device_block_peers[k]):json(nullptr)},
     {"device_peer_partials",exec_present?json(device_peer_partials[k]):json(nullptr)},
     {"decomposition",decomposition[k]},
     {"device_level_c",exec_present?device_levels_json(k):json(nullptr)},
     {"qrv2_factors",[&]{json j=json::array();for(const auto&e:qrv2_factors[k])j.push_back(e.second);return j;}()},
     {"device_qrv2_products",exec_present?json(device_qrv2_products[k]):json(nullptr)},
     {"device_qrv2_owners",exec_present?json(device_qrv2_owners[k]):json(nullptr)},
     {"contraction_interval_rule","K_z = [floor(K*z/c), floor(K*(z+1)/c)) for z in [0,c), balanced, nonempty (Sec 3 Split)"}};
   json levels=json::object();
   for(int l=0;l<LEVELS;++l){uint64_t lt=0,lr=0;int lm=0;
    for(const auto&e:products[k][l]){lt+=e.second;if(e.first>1)lr+=e.second;lm=std::max(lm,e.first);}
    levels[level_name(l)]={{"at_c",histogram(products[k][l])},{"products",lt},
     {"carried_products",lr},{"carried_fraction",lt?double(lr)/double(lt):0.0},{"max_c",lm},
     {"index",json::array()},{"combine",json::array()}};
   }
   json proofs=json::array(),indices=json::array(),combines=json::array();
   for(const auto&e:block_partitions[k]){proofs.push_back(e.second);indices.push_back(e.second["index"]);combines.push_back(e.second["combine"]);}
   levels["block"]["partitions"]=proofs;levels["block"]["index"]=indices;levels["block"]["combine"]=combines;
   per[kind_name(k)]["by_level"]=levels;
  }
  json why=json::object();for(int r=0;r<REASONS;++r)if(refusals[r])why[reason_name(r)]=refusals[r];
  json certs=json::array();for(const auto&e:certificates)certs.push_back(e.second.record());
  json dcerts=json::array();for(const auto&e:depth_certificates)dcerts.push_back(e.second.record());
   uint64_t elims=0,packets=0;int rmax=0;
   for(auto&e:packets_per_elimination){elims+=e.second;packets+=uint64_t(e.first)*e.second;rmax=std::max(rmax,e.first);}
   int smax=0;for(auto&e:slots_per_elimination)smax=std::max(smax,e.first);
   json depth_why=json::object();for(const auto&e:depth_refusals)depth_why[e.first]=e.second;
   json launch_rej=json::object();for(const auto&e:launch_rejected)launch_rej[e.first]=e.second;
   json preflight=json::object();for(const auto&e:preflight_by_reason)preflight[e.first]=e.second;
   return {{"execution_format",execution_format_version},{"per_product_kind",per},
    {"empty_terminal_eliminations",empty_eliminations},
    {"global_apply_policy",global_column_split?"member_columns_experimental":"owner_star"},
    {"global_column_slices",global_column_slices},{"global_ordered_pipeline",global_pipeline},
    {"global_owner_packets",global_owner_packets},{"global_peer_packets",global_peer_packets},
    {"transport_only_packets",transport_only_packets},
    {"depth_refusals_by_reason",depth_why},
    {"capacity_certificates",certs},{"depth_certificates",dcerts},
    {"device_completion_scope",exec_present
      ?"device counters written by one-thread kernels ordered after the products on the packet's own stream; reconciled against the host issued counts"
      :"unavailable; completed_after_fence certifies issued work after the existing successful stream/transport fence, not independent device counters"},
    {"max_simultaneous_credits",exec_present?json(credits_max):json(nullptr)},
    {"stage_overlap",exec_present?exec.record():json(nullptr)},
    {"next_panel_release_start",(exec_present&&exec.have_release)?exec.release_record():json(nullptr)},
    {"evidence_dropped_intervals",evidence_dropped},
    {"carried_products",carried},{"uncarried_products",uncarried},
    {"carried_fraction",carried+uncarried?double(carried)/double(carried+uncarried):0.0},
    {"remainder_peers",remainder_peers},
    {"aggregated_applies",aggregated_applies},{"aggregated_strips",aggregated_strips},{"aggregated_compose_G",aggregated_compose_G},{"aggregated_constituent_layers",aggregated_constituent_layers},{"aggregate_g",histogram(aggregate_g)},
    {"packets_per_elimination",histogram(packets_per_elimination)},
    {"slots_per_elimination",histogram(slots_per_elimination)},
    {"eliminations",elims},{"packets_issued",packets},
    {"max_packets_per_elimination",rmax},{"max_slots_per_elimination",smax},
    {"depth_inert",rmax<=1},
    {"refusals_by_reason",why},
    // Host-issued identity rules; per-packet execution receipts below say what the device ran.
    {"product_identity_schema",{{
      {"factorization","this run (one factorization per --reps trace entry)"},
      {"panel","elimination's panel k (exec elimination id in intervals)"},
      {"elimination","exec elimination id; ordered leaf/tree history per panel"},
      {"packet","strip packet within the elimination (r of Eq pipe)"},
      {"subkind",{ {"panel_GE","GE leaf or TT merge-stack factor (kind arg)"},
        {"apply_W","V^T X, transpose"},{ "apply_Z","T^T W, transpose"},
        {"apply_D","V Z accumulated into X in place (subtract)"},
        {"join_W","across-GPU W reduce over disjoint row sets"},
        {"compose_G","-T1 (V1^T V2) T2 transform combine"}}},
      {"operand_versions","expression counter + slot/transport generation per packet"},
      {"output_region","output tile rows x columns owned by the committing peer"}}}},
    {"pipeline_specification",{{
      {"packet_boundaries","elimination-local strip partition of the trailing width (strip menu)"},
      {"slot_ownership","one pipeline slot per in-flight strip; W/Z credits, staging, descriptors, events and generation owned per slot"},
      {"buffer_reservations","d x c W/Z credits sized in carrier_frame_words against S_mu"},
      {"dependency_events","per-strip commit event chain in issue order; exec open/close brackets per packet on its own stream"},
      {"stage_model","W, Z, D stages with measured pace; C(d)=t_sum+(r-1)*pace at r>1, C(d)=t_sum at r=1"}}}},
    {"launch_attempts",launch_attempts},{"launch_issued",launch_issued},
    {"launch_rejected_by_reason",launch_rej},
    {"preflight_substitutions",preflight_substitutions},
    {"preflight_by_reason",preflight},
    // Device clocks (%globaltimer) compare only WITHIN one device; ranks are never correlated
    // across devices without an explicit correlation record (none exists: cross-rank timing
    // comparisons are absent by construction).
    {"device_peer_partials",exec_present?json(device_peer_partials):json(nullptr)},
   {"device_block_peers",exec_present?json(device_block_peers):json(nullptr)},
    {"device_combines",exec_present?json(device_combines):json(nullptr)},
    {"device_physical_commits",exec_present?json(device_physical_commits):json(nullptr)},
    {"device_history_marks",exec_present?json(device_history_marks):json(nullptr)},
    {"device_membership_reports",exec_present?json(device_membership_reports):json(nullptr)},
    {"membership_expected",membership_expected},
   {"panel_batch_executed",[&]{json j=json::object();for(auto&e:panel_batch_executed)j[e.first]=e.second;return j;}()},
   {"panel_batch_executed_detail",[&]{json j=json::object();for(auto&e:panel_batch_executed_detail)j[e.first]=e.second;return j;}()},
   {"native_commits_issued",native_commits_issued},
   {"library_commits_issued",library_commits_issued},
   {"unobserved_commits_issued",unobserved_commits_issued},
   {"singleton_noops",singleton_noops},
    {"combines_expected",combines_expected},
    {"combines_logical",combines_logical},
    {"combine_output_tiles",combine_output_tiles},
    {"clock_scope","device %globaltimer, nanoseconds, comparable across CTAs and streams WITHIN one device only; no cross-device correlation is established or used"},
    // The entries below are labeled unknown and can never close a block-level requirement.
    {"vendor_internal_unknown",json::array({
      "cuBLAS strided-batched/pointer-array peer products (block geometry inside the library)",
      "NCCL/NVSHMEM collective reductions (join partials summed inside transport kernels)",
      "fused pentagonal merge kernel suboperations (no per-product partial identity yet)",
      "identity copy/subtraction kernels (no peer-partial counters yet)"})},
    {"scope","logical products count individual batch members; dispatch_groups count host product invocations, not API calls or GPU launches; global join_W is counted only on its output owner; transport-only ranks perform no arithmetic products"},
    {"depth_contract","packets_per_elimination is the r of Sec 3 Eq pipe. r=1 makes C(d)=t_Sigma for every d, so a depth d>1 with max_packets_per_elimination=1 is INERT regardless of what the plan records"}};
 }
};
}
