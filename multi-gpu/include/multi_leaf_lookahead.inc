// Explicit multi-GPU schedule for one GE leaf per active rank and one global
// merge per panel. Each far column receives GE then the global transform, in
// the original order. Only the next panel's GE factor runs ahead of that tail.
bool multi_leaf_global_lookahead()const{
 return ctx.size>1&&plan.qr_schedule.value("multi_leaf_global_lookahead",false);
}
void factor_multi_leaf_global(TiledMatrix<T>&a){
 const int depth=std::max(1,plan.d),laneB=depth;
 static const bool early_far=[]{const char*e=std::getenv("TQR_MULTI_EARLY_FAR");return !(e&&std::string(e)=="0");}();
 bool pending=false;
 auto retire=[&]{if(pending){trail_done.wait(stream.s);pending=false;}transport->bind(stream);};
 for(size_t pi=0;pi<plan.panels.size();++pi){
  const auto&pan=plan.panels[pi];const auto&batch=pan.ranks[ctx.rank];
  const int base=pan.col+pan.h;
  const bool split=pi+1<plan.panels.size()&&base<a.n;
  const int release=split?std::min(a.n,plan.panels[pi+1].col+plan.panels[pi+1].h):a.n;
  const unsigned ge_id=exec_elimination;
  std::array<int,2> fi{0,0},near_ge{0,0},far_ge{0,0};
  std::array<int,3> near_global{0,0,0},far_global{0,0,0};
  vbase=(pi&1)*vhalf;
  double issue=seconds();
  ge_batch(batch,a,true,true,base,base,true,0,fi.data());
  // Previous far columns include this panel's trailing columns, but do not
  // include its factor columns. Complete them before issuing any new Apply.
  retire();exec_elimination=ge_id;
  ge_batch(batch,a,true,true,base,release,false,0,near_ge.data());
  const unsigned global_id=ge_id+(!batch.ge.empty());
  // The far GE Apply of this panel reads only the local reflectors and columns >= release, which the global TT
  // factor never touches. Fork it BEFORE the TT factor so the bulk update overlaps the owner's TT factorization and
  // its publications. Per-column order is unchanged: far GE, then far global TT Apply (lane B waits
  // global_tt_event).
  const bool early=early_far&&release<a.n;
  if(early){
   pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
   exec_elimination=ge_id;
   ge_batch(batch,a,true,true,release,a.n,false,laneB,far_ge.data());
  }
  global_tt_recorded=early;
  exec_elimination=global_id;
  if(!pan.global.empty()){
   // GE has committed exactly through release on every member. Its R panel
   // is ready; its remaining Apply is not a prerequisite for TT factorization.
   // The far GE Apply will precede the far TT Apply on the same lane group.
   global_merge(pan.global[0],a,true,true,base,release,true,0,near_global.data(),release);
  }
  global_tt_recorded=false;
  if(release<a.n){
   if(early){global_tt_event.record(stream.s);global_tt_event.wait(slot_stream(laneB));}
   else{
   pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
   exec_elimination=ge_id;
   ge_batch(batch,a,true,true,release,a.n,false,laneB,far_ge.data());
   }
   if(!pan.global.empty()){
    exec_elimination=global_id;
    global_merge(pan.global[0],a,true,true,release,a.n,false,laneB,far_global.data());
   }
   trail_done.record(slot_stream(laneB));pending=true;
  }
  if(!batch.ge.empty()){
   const int packets=fi[0]+near_ge[0]+far_ge[0];
   const int slots=std::min(depth,std::max(near_ge[0],far_ge[0]));
   cw.elimination(packets,slots,1,depth==1?"selected_depth_one":"single_packet_tail");
   certify_elimination(packets,slots,fi[1]+near_ge[1]+far_ge[1],
    "GE near/far windows retain per-column ordering before the global transform",1);
  }
  if(!pan.global.empty()&&ctx.rank==pan.global[0].rank[0]){
   const int packets=near_global[0]+far_global[0],slots=std::max(near_global[2],far_global[2]);
   cw.elimination(packets,slots,1,depth==1?"selected_depth_one":"single_packet_tail");
   certify_elimination(packets,slots,near_global[1]+far_global[1],
    "global near/far windows follow matching GE windows; next GE overlaps the far chain",1);
  }
  exec_elimination=global_id+(!pan.global.empty());
  local_panel_s+=seconds()-issue;
 }
 retire();vbase=0;
}
