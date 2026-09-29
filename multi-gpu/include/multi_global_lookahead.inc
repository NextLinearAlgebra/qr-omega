// Member functions of TiledEngine. This schedule is selected explicitly only
// by the multi-GPU driver. All local tree work keeps its original ordering;
// only the last global update has a near/far split. The next GE reads the
// released panel columns while that far window still owns disjoint columns.
bool multi_global_tail_lookahead()const{
 return ctx.size>1&&plan.qr_schedule.value("multi_global_tail_lookahead",false);
}
void factor_multi_global_tail(TiledMatrix<T>&a){
 const int depth=std::max(1,plan.d),laneB=depth;
 if(!plan.la||agg!=1||!rmap.cyclic()||rmap.blk!=plan.b)
  throw std::runtime_error("multi_global_lookahead_descriptor_before_modify");
 bool pending=false;
 auto retire=[&]{if(pending){trail_done.wait(stream.s);pending=false;}transport->bind(stream);};
 for(size_t pi=0;pi<plan.panels.size();++pi){
  const auto&pan=plan.panels[pi];const auto&batch=pan.ranks[ctx.rank];
  const int base=pan.col+pan.h;
  const unsigned ge_id=exec_elimination;
  std::array<int,2> factor_issued{0,0},apply_issued{0,0};
  vbase=(pi&1)*vhalf;
  double issue=seconds();
  // No transport and no shared remote buffers are used by GE. Its V staging
  // half and lane-zero W/Z scratch are separate from the pending global tail.
  ge_batch(batch,a,true,true,base,base,true,0,factor_issued.data());
  retire();
  exec_elimination=ge_id;
  ge_batch(batch,a,true,true,base,a.n,false,0,apply_issued.data());
  if(!batch.ge.empty()){
   const int packets=factor_issued[0]+apply_issued[0],count=int(batch.ge.size());
   cw.elimination(packets,std::min(depth,packets),count,depth==1?"selected_depth_one":"single_packet_tail");
   certify_elimination(packets,std::min(depth,packets),factor_issued[1]+apply_issued[1],
    "leaf apply after next-GE overlap with preceding global tail",count);
  }
  exec_elimination=ge_id+(!batch.ge.empty());
  for(const auto&level:batch.levels)local_level(level,a,true,true);
  local_panel_s+=seconds()-issue;
  for(size_t gi=0;gi<pan.global.size();++gi){
   const auto&e=pan.global[gi];
   const bool split=gi+1==pan.global.size()&&pi+1<plan.panels.size()&&base<a.n;
   if(!split){global_merge(e,a,true,true);continue;}
   const auto&next=plan.panels[pi+1];
   const int release=std::min(a.n,next.col+next.h);
   const unsigned global_id=exec_elimination;
   std::array<int,3> near{0,0,0},far{0,0,0};
   global_merge(e,a,true,true,base,release,true,0,near.data());
   if(release<a.n){
    pipe_fork.record(stream.s);pipe_fork.wait(slot_stream(laneB));
    exec_elimination=global_id;
    global_merge(e,a,true,true,release,a.n,false,laneB,far.data());
    trail_done.record(slot_stream(laneB));pending=true;
   }
   exec_elimination=global_id+1;
   if(ctx.rank==e.rank[0]){
    const int packets=near[0]+far[0],slots=std::max(near[2],far[2]);
    cw.elimination(packets,slots,1,depth==1?"selected_depth_one":"single_packet_tail");
    certify_elimination(packets,slots,near[1]+far[1],
     "global update near/far windows; far tail overlaps the next GE factor",1);
   }
  }
 }
 retire();vbase=0;
}
