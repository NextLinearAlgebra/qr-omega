#pragma once
#include "backend_budget.hpp"
#include "common.hpp"
#include <unordered_map>
namespace tqr {
inline constexpr const char* catalog_id="native-hh-wjoin-v2";
struct FactorEvent {
 int id=0,kind=0,col=0,h=0,owner=0; std::vector<int> rows,children;
 json record()const{return {{"id",id},{"kind",kind==0?"GE":kind==1?"TS":"TT"},{"column",col},{"reflectors",h},{"owner",owner},{"support",rows},{"children",children},{"survivor",std::vector<int>(rows.begin(),rows.begin()+h)}};}
};
struct Plan {
 int m=0,n=0,p=1,b=1,leaf=64,c=1,d=1,threads=128,strip=64;
 std::string tree="binary_tt",profile_id="unprofiled",key,status="heuristic_incumbent";
 std::vector<FactorEvent> events;std::vector<size_t> workspace;
 double predicted=0,selection_s=0;int evaluated=0;bool measured=false;
 json costs,scope;
 json record(bool detail=true)const {
  json j={{"schema",1},{"catalog",catalog_id},{"m",m},{"n",n},{"active",p},{"b",b},{"leaf",leaf},{"c",c},{"d",d},{"threads",threads},{"strip",strip},{"tree",tree},{"profile_id",profile_id},{"key",key},{"selection_status",status},{"predicted_s",measured?json(predicted):json(nullptr)},{"selection_s",selection_s},{"evaluated",evaluated},{"workspace_bytes",workspace},{"costs",costs},{"scope",scope},{"physical_bound",nullptr},{"model_gap",nullptr},{"schedule","panel order; ordered local/global tree; disjoint column strips with CUDA completion credits"},{"replication","immutable V/X copies; disjoint support-row contractions; W join; owner commit"},{"output","distributed upper trapezoid plus ordered full-Q handle"}};
  if(detail){j["events"]=json::array();for(auto&e:events)j["events"].push_back(e.record());}return j;
 }
};
struct Node {std::vector<int> rep;int event=-1;};
inline Plan instantiate(int m,int n,int p,int b,int leaf,int c,int d,int threads,int strip,std::string tree,size_t word,int max_events=1000000,double deadline=std::numeric_limits<double>::infinity()) {
 if(m<0||n<0||p<1||b<1||b>32||leaf<b||(tree!="scalar_inplace"&&leaf>256)||c<1||c>4||d<1||d>2||threads<1||threads>256||(threads&(threads-1))||strip<1||strip>256)throw std::runtime_error("unsupported_descriptor");
 if(checked_mul(size_t(m),size_t(n))>INT_MAX||m>32768||n>32768)throw std::runtime_error("catalog_extent_limit");
 if(tree!="scalar_inplace"&&tree!="flat_ts"&&tree!="binary_tt")throw std::runtime_error("unsupported_tree");
 Plan a;a.m=m;a.n=n;a.p=p;a.b=b;a.leaf=leaf;a.c=c;a.d=d;a.threads=threads;a.strip=strip;a.tree=tree;
 if(tree=="scalar_inplace"){
  if(p!=1||c!=1||d!=1)throw std::runtime_error("scalar_member_requires_p1_c1_d1");
  a.workspace={checked_mul(size_t(m)*n+std::min(m,n),word)+4096};a.scope={{"member","storage-minimal in-place scalar GE"},{"factor_backing","lower triangle of A plus tau"},{"c",1},{"d",1}};return a;
 }
 std::vector<size_t> retained(p,0);int k=std::min(m,n),maxs=1,maxh=1;
 auto emit=[&](int kind,int col,int width,std::vector<int> rows,std::vector<int> children)->Node{
  if(int(a.events.size())>=max_events)throw std::runtime_error("descriptor_budget");
  if(seconds()>deadline)throw std::runtime_error("search_deadline");
  FactorEvent e;e.id=a.events.size();e.kind=kind;e.col=col;e.h=std::min(width,int(rows.size()));e.owner=owner_of(rows[0],m,p);e.rows=std::move(rows);e.children=std::move(children);
  maxs=std::max(maxs,int(e.rows.size()));maxh=std::max(maxh,e.h);
  retained[e.owner]=checked_add(retained[e.owner],checked_mul(size_t(e.rows.size())*e.h+e.h*e.h,word));
  Node v{std::vector<int>(e.rows.begin(),e.rows.begin()+e.h),e.id};a.events.push_back(std::move(e));return v;
 };
 auto merge=[&](Node l,Node r,int col,int width)->Node{std::vector<int> rows=l.rep;rows.insert(rows.end(),r.rep.begin(),r.rep.end());return emit(2,col,width,std::move(rows),{l.event,r.event});};
 for(int col=0;col<k;){
  int root=owner_of(col,m,p),bcur=std::min({b,k-col,row_begin(m,p,root+1)-col});std::vector<Node> global;
  for(int rank=root;rank<p;++rank){
   int start=std::max(col,row_begin(m,p,rank)),end=row_begin(m,p,rank+1);std::vector<Node> leaves;
   for(int row=start;row<end;row+=leaf){int stop=std::min(end,row+leaf);std::vector<int> support(stop-row);std::iota(support.begin(),support.end(),row);
    if(tree=="flat_ts"&&!leaves.empty()){
     Node old=leaves.back();leaves.pop_back();std::vector<int> rows=old.rep;rows.insert(rows.end(),support.begin(),support.end());leaves.push_back(emit(1,col,bcur,std::move(rows),{old.event}));
    }else leaves.push_back(emit(0,col,bcur,std::move(support),{}));
   }
   while(leaves.size()>1){std::vector<Node> next;for(size_t i=0;i<leaves.size();i+=2)next.push_back(i+1<leaves.size()?merge(leaves[i],leaves[i+1],col,bcur):leaves[i]);leaves=std::move(next);}
   if(!leaves.empty())global.push_back(leaves[0]);
  }
  while(global.size()>1){std::vector<Node> next;for(size_t i=0;i<global.size();i+=2)next.push_back(i+1<global.size()?merge(global[i],global[i+1],col,bcur):global[i]);global=std::move(next);}
  col+=bcur;
 }
 // Simultaneous owned allocations: A, input reference, per-entry history, factors,
 // complete d-generation replicated V/X/W/Z/scales, panel, descriptors/status.
 // Validation/reference allocations are charged separately by the runner.
 for(int rank=0;rank<p;++rank){size_t nr=row_begin(m,p,rank+1)-row_begin(m,p,rank);
  size_t matrix=checked_mul(checked_mul(nr,size_t(n)),word+sizeof(uint64_t));
  size_t gen=checked_mul(size_t(c)*(maxs*maxh+maxs*strip+maxh*strip)+2*maxh*strip+strip,word);
  size_t frame=checked_mul(size_t(maxs)*maxh,word)+size_t(maxs)*(sizeof(int)+sizeof(uint64_t));
  size_t backend=p>1?checked_mul(size_t(maxs)*std::max(maxh,strip)+maxh*maxh,size_t(word)*p*16)+8*1024*1024:0;
  a.workspace.push_back(checked_add(checked_add(matrix,retained[rank]),checked_add(frame+size_t(d)*gen+backend,65536)));
 }
 a.scope={{"width_menu",{1,8,16,32}},{"leaf_menu",{32,64,128,256}},{"tree_menu",{"flat_ts","binary_tt"}},{"depth_menu",{1,2}},{"layer_menu",{1,2,4}},{"join","W"},{"backend_order","serialized common episodes"},{"optional_native_tree_certificate","not claimed"},{"cache","measured service only; no owned L2"}};
 return a;
}
inline void inventory(Plan&a,int allocated,size_t word,int pad,const json&profile){
 int s=1,h=1;for(auto&e:a.events){s=std::max(s,int(e.rows.size()));h=std::max(h,e.h);}int cols=std::max(a.m,a.n);size_t cap=std::max(size_t(64),size_t(s)*std::max(h,a.strip)*word);
 size_t gran=profile.value("hardware",json::object()).value("nccl_vmm_granularity",size_t(1));auto rounded=[&](size_t v){return ((v+gran-1)/gran)*gran;};
 a.workspace.clear();a.scope["allocation_lifetimes"]=json::array();
 for(int rank=0;rank<allocated;++rank){int nr=rank<a.p?row_begin(a.m,a.p,rank+1)-row_begin(a.m,a.p,rank):0,public_rows=row_begin(a.m,allocated,rank+1)-row_begin(a.m,allocated,rank);
  size_t input=size_t(std::max(1,public_rows+pad))*a.n*word,retained=0,descriptors=0;
  for(auto&e:a.events){descriptors+=e.rows.size()*sizeof(int);if(e.owner==rank)retained+=(e.rows.size()*e.h+e.h*e.h)*word;}
  size_t generation=(size_t(a.c)*(s*h+s*std::max(h,a.strip)+h*a.strip)+2*h*a.strip+a.strip+h*h)*word;
  size_t history=size_t(nr)*cols*sizeof(uint64_t)+size_t(s)*sizeof(uint64_t);
  size_t frame=2*size_t(s)*std::max(h,a.strip)*word+4+9*8;
  size_t subset=allocated>a.p?size_t(std::max(1,nr))*cols*word:0;
  size_t arena=backend_arena_budget(cap,allocated,profile);
  size_t overhead=allocated>1?profile.value("library_overhead_allowance_bytes",size_t(0)):0;
  if(a.tree=="scalar_inplace"){retained=std::min(a.m,a.n)*word;descriptors=history=generation=subset=0;frame=4+9*8;}
  size_t total=input+retained+descriptors+history+frame+subset+size_t(a.d)*generation+arena+overhead;
  a.workspace.push_back(total);a.scope["allocation_lifetimes"].push_back({{"rank",rank},{"caller_A",input},{"retained_V_T_or_tau",retained},{"row_descriptors",descriptors},{"histories_and_expected",history},{"frame",frame},{"one_generation",generation},{"live_generations",a.d},{"active_layout",subset},{"backend_arenas_rounded",arena},{"library_overhead_allowance",overhead},{"total_requested_and_allowance",total},{"retained_until","handle release; scalar vectors alias caller A"},{"generation_retirement","all local strip readers complete; backend inbox credit and canonical owner history are separate"}});
 }
 a.scope["workspace_claim"]="requested owned bytes, queried VMM rounding and measured library allowance; runtime free-memory delta checked before A modification; no cache allocation";
 a.scope["host_storage"]="plan/factor row lists and optional trace are host metadata, separately from HBM; no host numerical QR";
}
inline double service(const json& profile,const std::string& op,int s,int h,int q,int mode=128){
 if(!profile.contains("samples"))return std::numeric_limits<double>::quiet_NaN();
 static thread_local std::string cached_profile;static thread_local std::unordered_map<std::string,double> cache;
 auto pid=profile.value("id",std::string("unprofiled"));if(pid!=cached_profile||cache.size()>32768){cache.clear();cached_profile=pid;}
 std::string key=op+":"+std::to_string(s)+":"+std::to_string(h)+":"+std::to_string(q)+":"+std::to_string(mode);auto found=cache.find(key);if(found!=cache.end())return found->second;
 double best=std::numeric_limits<double>::infinity(),time=0;
 for(const auto& x:profile["samples"]){if(x.at("op")!=op||x.value("threads",mode)!=mode)continue;
  int xs=x.value("s",1),xh=x.value("h",1),xq=x.value("q",1);
  double dist=abs(log(double(std::max(s,1))/xs))+abs(log(double(std::max(h,1))/xh))+abs(log(double(std::max(q,1))/xq));
  if(dist<best){best=dist;double ratio=double(std::max(s,1))*std::max(h,1)*std::max(q,1)/(double(xs)*xh*xq);
   time=std::max(profile.value("launch_s",0.0),x.at("median_s").get<double>()*ratio);}
 }
 return cache[key]=std::isfinite(best)?time:std::numeric_limits<double>::quiet_NaN();
}
inline double evaluate(Plan& a,const json& p,double deadline=std::numeric_limits<double>::infinity()){
 if(a.tree=="scalar_inplace"){double x=service(p,"scalar_ge",a.m,std::min(a.m,a.n),a.n,a.threads);a.measured=std::isfinite(x);a.predicted=a.measured?x+2*p.value("host_launch_sync_s",0.0)+service(p,"norm",std::max(1,a.m*a.n),1,1,256):0;a.costs={{"scalar_ge_s",a.measured?json(x):json(nullptr)},{"entry_scan_and_host_s",a.measured?json(a.predicted-x):json(nullptr)}};return a.measured?a.predicted:std::numeric_limits<double>::infinity();}
 double factor=0,apply=0,move=0,protocol=0,control=0,redistribution=0;bool valid=true;
 const int allocated=a.scope.value("allocated_ranks",a.p);const size_t word=p.value("precision",std::string("fp64"))=="fp32"?4:8;
 auto get=[&](std::string op,int s,int h,int q,int t){double v=service(p,op,s,h,q,t);if(!std::isfinite(v)){valid=false;return 0.0;}return v;};
 auto sync=p.value("host_launch_sync_s",0.0);auto nv=[&](int count){return get("nv_publish",count,1,1,128);};
 for(const auto&e:a.events){if(seconds()>deadline)throw std::runtime_error("search_deadline");int s=e.rows.size(),h=e.h;
  factor+=get(e.kind==0?"ge":e.kind==1?"ts":"tt",s,h,1,a.threads);control+=3*sync;
  std::vector<int> owned(a.p,0);for(int r:e.rows)++owned[owner_of(r,a.m,a.p)];
  for(int r=0;r<a.p;++r)if(owned[r]){move+=4*get("copy",owned[r]*h,1,1,128);control+=3*sync;if(r!=e.owner)protocol+=2*nv(owned[r]*h*word);}
  std::vector<double> credits(a.d,0.0);double last=0,issue=0;
  for(int j=e.col+h;j<a.n;j+=a.strip){if(seconds()>deadline)throw std::runtime_error("search_deadline");int q=std::min(a.strip,a.n-j),layer=std::min(a.c,s);
   double v=get("apply",s,h,q,a.threads),extra=0;
   // Apply includes one X copy, scale, W, T and VZ. Only omitted movement,
   // joins, histories and explicit protocol episodes are added here.
   extra+=get("copy",s*h,1,1,128)+get("join",h,q,allocated>1?1:layer,128)+2*sync;
   if(allocated==1){extra+=(layer-1)*(get("copy",s*h,1,1,128)+get("copy",s*q,1,1,128));}
   else {
    for(int r=0;r<a.p;++r)if(owned[r]){extra+=2*get("copy",owned[r]*q,1,1,128)+2*sync;if(r!=e.owner)extra+=nv(owned[r]*q*word)*(layer==1?2:1);}
    if(layer>1){extra+=get("paper_copy",s*h*word,1,1,128)+get("paper_copy",h*h*word,1,1,128)+get("paper_copy",s*q*word,1,1,128)+get("paper_reduce",h*q*word,1,1,128)+get("paper_copy",h*q*word,1,1,128);}
   }
   int slot=(j-e.col-h)/a.strip%a.d;double mult=a.d>1?p.value("concurrent_apply_multiplier",double(a.d)):1.0;
   // The identical packet reserved-stage equation is deliberately not applied to CUDA streams.
   double begin=std::max(issue,credits[slot]);credits[slot]=begin+(v+extra)*mult;issue+=p.value("host_enqueue_s",0.0);last=std::max(last,credits[slot]);
  }apply+=last;
  if(allocated>1)for(int r=0;r<a.p;++r)if(r!=e.owner&&owned[r])protocol+=nv(0);
 }
 if(allocated>a.p){int maxs=1;for(auto&e:a.events)maxs=std::max(maxs,int(e.rows.size()));
  for(int src=0;src<allocated;++src)for(int dst=0;dst<a.p;++dst){int lo=std::max(row_begin(a.m,allocated,src),row_begin(a.m,a.p,dst)),hi=std::min(row_begin(a.m,allocated,src+1),row_begin(a.m,a.p,dst+1));
   for(int r=lo;r<hi;r+=maxs)for(int j=0;j<a.n;j+=a.strip){int count=std::min(maxs,hi-r)*std::min(a.strip,a.n-j);redistribution+=4*get("copy",count,1,1,128)+4*sync;if(src!=dst)redistribution+=2*nv(count*word);}
  }
 }
 double entry=get("norm",std::max(1,a.m*a.n/allocated),1,1,256)+2*sync;
 a.costs={{"native_factors_s",factor},{"strip_apply_including_join_protocol_s",apply},{"pack_scatter_s",move},{"ordered_publication_s",protocol},{"host_control_s",control},{"input_output_redistribution_s",redistribution},{"entry_scan_completion_s",entry},{"reused_setup_s",0},{"setup_first_use_s",p.value("backend_setup_s",0.0)},{"fit_method","nearest signature volume interpolation plus enumerated episode counts; predictions with unknown bounds"},{"objective","reused factorization, original input/output placement and usable implicit Q handle"}};
 a.measured=valid;a.predicted=factor+apply+move+protocol+control+redistribution+entry;return valid?a.predicted:std::numeric_limits<double>::infinity();
}

inline Plan select_plan(const RunOptions& o,int ranks,size_t word,const json& profile){
 double start=seconds();int p=o.active?o.active:ranks;if(p<1||p>ranks)throw std::runtime_error("active_subset_out_of_range");
 if(ranks>1&&(o.c>p||o.d>1||o.tree=="scalar_inplace"))throw std::runtime_error("unsupported_distributed_catalog_member");
 // Complete conservative incumbent precedes the bounded menu. Its construction
 // and descriptor limit are charged; A has not been touched by planning.
 bool minimal=ranks==1&&o.tree.empty()&&o.c<=1&&o.d<=1;
 Plan best=instantiate(o.m,o.n,p,o.b?o.b:32,o.leaf?o.leaf:256,o.c?o.c:1,o.d?o.d:1,o.threads?o.threads:128,o.strip?o.strip:64,minimal?"scalar_inplace":o.tree.empty()?"binary_tt":o.tree,word);
 best.scope["allocated_ranks"]=ranks;double cost=evaluate(best,profile);int evaluated=1;
 auto fits=[&](Plan& a){inventory(a,ranks,word,o.pad,profile);return *std::max_element(a.workspace.begin(),a.workspace.end())<=o.budget;};
 if(!fits(best))cost=std::numeric_limits<double>::infinity();
 if(ranks==1&&(o.tree.empty()||o.tree=="scalar_inplace")&&o.c<=1&&o.d<=1){
  Plan a=instantiate(o.m,o.n,1,1,std::max(1,o.m),1,1,o.threads?o.threads:128,64,"scalar_inplace",word);
  double x=evaluate(a,profile);if(fits(a)&&(x<cost||!fits(best)||o.tree=="scalar_inplace")){best=std::move(a);cost=x;}
 }
 const std::vector<int> widths=o.b?std::vector<int>{o.b}:std::vector<int>{8,16,32,1};
 const std::vector<int> leaves=o.leaf?std::vector<int>{o.leaf}:std::vector<int>{64,128,32,256};
 const std::vector<int> layers=o.c?std::vector<int>{o.c}:std::vector<int>{1,2,4};
 const std::vector<int> depths=o.d?std::vector<int>{o.d}:std::vector<int>{1,2};
 const std::vector<int> modes=o.threads?std::vector<int>{o.threads}:std::vector<int>{128,32,1,256};
 const std::vector<std::string> trees=o.tree.empty()?std::vector<std::string>{"binary_tt","flat_ts"}:std::vector<std::string>{o.tree};
 bool exhausted=true;
 std::vector<int> counts={p};if(!o.active&&ranks>1){counts.push_back(1);for(int a=2;a<ranks;++a)counts.push_back(a);}
 for(int b:widths)for(int l:leaves)for(int c:layers)for(int d:depths)for(auto&t:trees)for(int threads:modes)for(int participants:counts){
  if(evaluated>=o.search_budget||seconds()-start>0.2){exhausted=false;goto done;}
  if(l<b||(ranks>1&&(d>1||c>participants))||c>std::max(o.m,1))continue;
  if(t=="scalar_inplace"&&(ranks!=1||c!=1||d!=1))continue;
  if(d>1&&!profile.contains("concurrent_apply_multiplier"))continue;
  try{Plan a=instantiate(o.m,o.n,participants,b,l,c,d,threads,o.strip?o.strip:64,t,word,1000000,start+0.2);a.scope["allocated_ranks"]=ranks;++evaluated;if(!fits(a))continue;
  double x=evaluate(a,profile,start+0.2);if(x<cost){best=std::move(a);cost=x;}}
  catch(const std::runtime_error&e){if(std::string(e.what())=="search_deadline"){exhausted=false;goto done;}throw;}
 }
 done:
 if(!fits(best))throw std::runtime_error("insufficient_storage_before_modify");
 best.evaluated=evaluated;best.selection_s=seconds()-start;best.profile_id=profile.value("id",std::string("unprofiled"));
 // Cost interpolation and floating comparisons are predictive, not certified interval arithmetic.
 best.status=exhausted?"completed_predictive_menu":"heuristic_incumbent";
 best.scope["search_budget_candidates"]=o.search_budget;best.scope["search_budget_s"]=0.2;
 best.scope["allocated_ranks"]=ranks;best.scope["precision"]=word==4?"fp32":"fp64";best.scope["workspace_budget"]=o.budget;best.scope["pad"]=o.pad;
 best.scope["hardware_key"]=profile.contains("hardware")?profile["hardware"]["stable_key"]:json(nullptr);
 best.key=digest(best.record(false).dump()+std::to_string(word));return best;
}
inline Plan load_plan(const RunOptions&o,int ranks,size_t word,const json&profile){
 json j=read_json(o.plan_file),unsigned_record=j;unsigned_record.erase("artifact_digest");
 if(j.at("artifact_digest")!=digest(unsigned_record.dump()))throw std::runtime_error("plan_artifact_integrity");
 const auto&s=j.at("scope");
 if(j.at("catalog")!=catalog_id||j.at("m")!=o.m||j.at("n")!=o.n||j.at("profile_id")!=profile.at("id")||s.at("allocated_ranks")!=ranks||s.at("precision")!=(word==4?"fp32":"fp64")||s.at("hardware_key")!=profile.at("hardware").at("stable_key"))throw std::runtime_error("incompatible_plan");
 Plan a=instantiate(o.m,o.n,j.at("active"),j.at("b"),j.at("leaf"),j.at("c"),j.at("d"),j.at("threads"),j.at("strip"),j.at("tree"),word);
 if(a.record(true).at("events")!=j.at("events"))throw std::runtime_error("plan_schedule_integrity");
 inventory(a,ranks,word,o.pad,profile);if(*std::max_element(a.workspace.begin(),a.workspace.end())>o.budget)throw std::runtime_error("plan_workspace_before_modify");
 a.key=j.at("key");a.profile_id=j.at("profile_id");a.status="compatible_plan_replay";a.scope=s;a.costs=j.at("costs");a.measured=!j.at("predicted_s").is_null();if(a.measured)a.predicted=j.at("predicted_s");return a;
}
inline double pipeline_identical(const std::vector<double>& t,int packets,int depth){
 if(packets==0)return 0;if(t.empty()||depth<1)throw std::runtime_error("pipeline_descriptor");
 double ell=std::accumulate(t.begin(),t.end(),0.0),h=*std::max_element(t.begin(),t.end());
 return ell+(packets-1)*h+((packets-1)/depth)*std::max(ell-depth*h,0.0);
}
}
