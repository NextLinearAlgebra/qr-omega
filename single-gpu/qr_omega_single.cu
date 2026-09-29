// QR-Omega on one GPU: factors a generated matrix with an explicit configuration, times the
// factorization, and validates R and the retained Householder factors.
#define TQR_LIBRARY_ONLY
#include "tiled_main.cu"

struct FixedOptions {
 int m=4096,n=4096,b=128,leaf=8192,strip=4096,threads=256;
 int wc=2,zc=2,dc=2,depth=2,radix=2,gradix=2,aggregate=1,lookahead=0,row_block=0;
 int groups=32,pw=16,panel_threads=512,window=64,reps=3,free_sms=0;
 int groups_late=0,late_rows=0;   // panels with fewer than late_rows rows use groups_late (0 = off)
 bool full_q=false;
 std::string precision="fp64",output,input="random";
};

FixedOptions fixed_options(int argc,char**argv){
 FixedOptions o;
 for(int i=1;i<argc;++i){
  const std::string k=argv[i];
  auto val=[&](){if(++i==argc)throw std::runtime_error("missing_option:"+k);return std::string(argv[i]);};
  if(k=="--m")o.m=std::stoi(val());else if(k=="--n")o.n=std::stoi(val());
  else if(k=="--b")o.b=std::stoi(val());else if(k=="--leaf")o.leaf=std::stoi(val());
  else if(k=="--strip")o.strip=std::stoi(val());else if(k=="--threads")o.threads=std::stoi(val());
  else if(k=="--wc")o.wc=std::stoi(val());else if(k=="--zc")o.zc=std::stoi(val());
  else if(k=="--dc")o.dc=std::stoi(val());else if(k=="--d")o.depth=std::stoi(val());
  else if(k=="--radix")o.radix=std::stoi(val());else if(k=="--gradix")o.gradix=std::stoi(val());
  else if(k=="--aggregate")o.aggregate=std::stoi(val());else if(k=="--lookahead")o.lookahead=std::stoi(val());
  else if(k=="--row-block")o.row_block=std::stoi(val());else if(k=="--free-sms")o.free_sms=std::stoi(val());
  else if(k=="--panel-groups")o.groups=std::stoi(val());else if(k=="--panel-width")o.pw=std::stoi(val());
  else if(k=="--panel-threads")o.panel_threads=std::stoi(val());else if(k=="--panel-window")o.window=std::stoi(val());
  else if(k=="--panel-groups-late")o.groups_late=std::stoi(val());else if(k=="--panel-late-rows")o.late_rows=std::stoi(val());
  else if(k=="--reps")o.reps=std::stoi(val());else if(k=="--precision")o.precision=val();
  else if(k=="--fp32-math")fp32_math()=parse_fp32_math(val());else if(k=="--output")o.output=val();
  else if(k=="--input")o.input=val();else if(k=="--full-q")o.full_q=true;
  else throw std::runtime_error("unknown_option:"+k);
 }
 if(o.m<1||o.n<1||o.b<1||o.b>128||o.leaf<o.b||o.strip<1||o.reps<1||o.depth<1||
    o.wc<2||o.zc<2||o.dc<2||o.aggregate<1||o.aggregate>16||o.lookahead<0||o.lookahead>1||
    (o.precision!="fp32"&&o.precision!="fp64"))throw std::runtime_error("invalid_fixed_schedule_or_c_not_gt_one");
 return o;
}

// Independent full-Q Gram check: form Q from identity using the retained
// Householder factors, then use pedantic BLAS to compute every entry of Q^T Q.
// This is validation only, outside the factorization timer.
template<class T> json fixed_full_q(TiledEngine<T>&engine,int m,double tol){
 if(m>16384)throw std::runtime_error("full_Q_validation_limited_to_16384_rows");
 const double start=seconds();
 TiledMatrix<T>q(m,m,1,0);Buffer<T>qtq(size_t(m)*m);Stream st;
 identity<<<128,128,0,st>>>(q.a.p,m,m,q.ld,0);st.sync();
 const int status=engine.apply_q(q,false);
 cublasHandle_t bh=nullptr;blas_check(cublasCreate(&bh));
 blas_check(cublasSetStream(bh,st));blas_check(cublasSetMathMode(bh,CUBLAS_PEDANTIC_MATH));
 gram(bh,m,q.a.p,qtq.p);st.sync();blas_check(cublasDestroy(bh));
 const double err=relative(qtq.p,(T*)nullptr,m,m,m,1,true);
 return {{"pass",status==0&&std::isfinite(err)&&err<=tol},
         {"columns_checked",m},{"Q_T_Q_minus_I_fro_over_sqrt_m",err},
         {"engineering_tolerance",tol},{"gram_math","pedantic BLAS in storage precision"},
         {"validation_s",seconds()-start}};
}

template<class T> int fixed_run(const FixedOptions&o,Context&ctx){
 if(ctx.size!=1)throw std::runtime_error("single_gpu_only");
 const auto hardware=discover(ctx);
 if(hardware.at("cpu_affinity")!=required_cpu_set())throw std::runtime_error("required_cpu_affinity");
 auto p=tiled_instantiate(o.m,o.n,ctx.size,o.b,o.leaf,o.strip,o.threads,sizeof(T),0,
                         o.radix,o.depth,o.wc,o.gradix,o.zc,o.dc,2,2,2,o.aggregate,o.lookahead,o.row_block);
 p.la_free=o.free_sms;
 p.c=o.groups;p.pw=o.pw;p.cthreads=o.panel_threads;p.csw=o.window;
 json panel_fits=json::array();
 if(o.groups){
  if(o.groups<2||o.pw<1||o.window<0||o.window>o.b)throw std::runtime_error("invalid_panel_carrier");
  for(auto&pan:p.panels)for(auto&batch:pan.ranks)
   if(batch.ge.size()==1&&batch.ge[0].rows>=coop_min_rows&&batch.h>=o.pw){
    // Query the actual compiled kernel's capacity; no borrowed profile or
    // isolated-panel timing decides the footprint under look-ahead.
    const int h=batch.h,rows=batch.ge[0].rows;
    const int groups=(o.groups_late>1&&rows<o.late_rows)?o.groups_late:o.groups;
    const int first=std::min(h,o.window?o.window:h);
    int chosen=0;
    std::vector<int> windows{first};
    for(int w:{64,32,16,8})if(w<first&&w>=o.pw)windows.push_back(w);
    for(int sw:windows){
     try{
      for(int w0=0;w0<h;w0+=sw)
       launch_cooperative_ge_mini<T>(nullptr,rows,nullptr,rows,h,w0,std::min(sw,h-w0),
                                    std::min(o.pw,sw),groups,o.panel_threads,
                                    nullptr,o.b,nullptr,nullptr,nullptr,nullptr,true);
      chosen=sw;break;
     }catch(const std::exception&){}
    }
    if(!chosen)throw std::runtime_error("explicit_panel_footprint_does_not_fit_before_modify");
    batch.cc=groups;batch.cpw=o.pw;batch.cthreads=o.panel_threads;batch.csw=chosen;
    panel_fits.push_back({{"col",pan.col},{"rows",rows},{"h",h},{"groups",groups},{"window",chosen}});
   }
 }
 p.profile_id="manual-fixed-single";
 p.selection={{"status","explicit_configuration"},{"selector_used",false},{"timing_samples_used",false},
              {"scope","This schedule is a candidate; numerical and executed-carrier evidence determine eligibility."}};
 // Query actual placement/granularity; do not borrow timing samples or identities
 // from the old allocation containing a different (clock-capped) GPU.
 const json capacity={{"hardware",hardware},{"library_overhead_allowance_bytes",0}};
 tiled_charge_backend(p,sizeof(T),capacity);
 size_t d_peer_bytes=0;
 if constexpr(std::is_same_v<T,double>){if(d_peers_enabled()){
  if(p.dc!=2)throw std::runtime_error("D_peers_require_c2");
  d_peer_bytes=checked_mul(sizeof(T),checked_mul(size_t(tiled_lane_slots(p.d,p.la!=0)),
                checked_mul(size_t(2),checked_mul(size_t(p.leaf),size_t(p.strip)))));
  for(auto&owned:p.owned)owned=checked_add(owned,d_peer_bytes);
 }}
 size_t budget=SIZE_MAX;
 for(const auto&free:ctx.all_records(hardware.at("hbm_free_observed_bytes")))budget=std::min(budget,free.template get<size_t>());
 p.budget_bytes=budget;
 p.qr_schedule=tiled_schedule(p,sizeof(T));
 if(const char*e=std::getenv("TQR_BALANCED_AGG")){
  const std::string policy=e;
  if(policy!="0"&&policy!="1")throw std::runtime_error("invalid_balanced_aggregate_policy");
  p.qr_schedule["aggregate_layer_policy"]=policy=="1"?"balanced_reflectors":"whole_constituents";
 }
 p.schedule_digest=tiled_schedule_digest(p.qr_schedule);
 p.key=digest(p.qr_schedule.dump());
 const auto schedule_digests=ctx.all_records(p.schedule_digest);
 for(const auto&d:schedule_digests)if(d!=p.schedule_digest)throw std::runtime_error("rank_schedule_mismatch");
 const int kind=o.input=="zero"?1:o.input=="rank_deficient"?2:0;
 if(o.input!="random"&&o.input!="zero"&&o.input!="rank_deficient"&&o.input!="near_large"&&o.input!="near_small")
  throw std::runtime_error("unsupported_input");
 const T scale=o.input=="near_large"?std::numeric_limits<T>::max()/(T(1024)*T(o.m+o.n)):
               o.input=="near_small"?std::numeric_limits<T>::min()*T(128):T(1);
 json record={{"driver","single-fixed-v1"},{"precision",precision<T>()},{"precision_mode",precision_mode_name<T>()},
              {"m",o.m},{"n",o.n},{"gpus",ctx.size},{"input",o.input},{"hardware",hardware},
              {"measurement_class",hardware.at("measurement_class")},{"schedule_digest_per_rank",schedule_digests},
              {"plan",{{"member","compact"},{"member_plan",p.record()}}},{"panel_capacity_choices",panel_fits},
              {"fp64_single_pass_panel_norm",bool(TQR_PANEL_SSQ64_FP64)},
              {"fp64_fast_scaled_panel_norm",bool(TQR_PANEL_FP64_FAST_SCALED_NORM)},
              {"kami_d_tile",kami_d_tile()},{"kami_d_raster",kami_d_raster()},
              {"d_peer_workspace_bytes",d_peer_bytes},{"d_peers_enabled",d_peers_enabled()},
              {"capacity_policy","requested inventory plus actual observed allocation checked before modifying A; no unmeasured library allowance claim"}};
 record["plan"]["member_plan"]["carrier_selection"]="Explicit hand-picked geometry; per-batch kernel capacity checked on the actual GPU; no timing-model selection.";
 record["plan"]["member_plan"]["c_d_source"]="Command-line configuration; executed product receipts remain the evidence for c>1.";
 std::vector<double>times,skews;int status=0;
 {
  const double setup=seconds();TiledEngine<T>engine(ctx,p,budget);
  TiledMatrix<T>a(o.m,o.n,ctx.size,ctx.rank,0);Stream reset_stream;
  record["allocation_backend_s"]=seconds()-setup;
  for(int rep=0;rep<=o.reps;++rep){
   regenerate<<<256,128,0,reset_stream>>>(a.a.p,a.nr,a.n,a.ld,a.begin,a.m,0,kind,scale);reset_stream.sync();
   ctx.barrier();const double start=seconds();
   {PhaseRange range(rep?"tqr:factorization":"tqr:warmup");status=engine.factor(a);}
   ctx.barrier();const double end=seconds();
   const auto ranks=ctx.all_records({{"begin",start},{"end",end}});
   double first=start,last=end,latest=start;
   for(const auto&r:ranks){first=std::min(first,r.at("begin").template get<double>());
    latest=std::max(latest,r.at("begin").template get<double>());last=std::max(last,r.at("end").template get<double>());}
   if(rep){times.push_back(last-first);skews.push_back(latest-first);}else record["first_execution_s"]=last-first;
   if(status)break;
  }
  record["status"]=status;record["rank_evidence"]=ctx.all_records(engine.evidence());
  record["timing"]={{"reused_host_s",times},{"reused_median_s",times.empty()?json(nullptr):json(median(times))},
                    {"launch_skew_s",skews},{"boundary","host release through factor completion; fresh device input to R and retained full-Q factors; reset and validation excluded"}};
  if(!status){PhaseRange range("tqr:validation");record["validation"]=validate_streamed(engine,a,ctx,p.strip,kind,scale,p);
   if(o.full_q){json full=fixed_full_q(engine,o.m,record["validation"]["engineering_tolerance"].get<double>());
    record["validation"]["pass"]=record["validation"]["pass"].get<bool>()&&full["pass"].get<bool>();
    record["validation"]["full_Q_Gram"]=full;}}
 }
 record["eligible_numerically"]=status==0&&record.value("validation",json::object()).value("pass",false);
 record["c_gt_one_admission"]="not inferred from requested c; inspect executed product and panel receipts";
 if(ctx.rank==0){if(o.output.empty())std::cout<<record.dump(2)<<'\n';else write_json(o.output,record);}
 return record.at("eligible_numerically").get<bool>()?0:1;
}

int main(int argc,char**argv){
 try{const auto o=fixed_options(argc,argv);Context ctx;
  return o.precision=="fp64"?fixed_run<double>(o,ctx):fixed_run<float>(o,ctx);
 }catch(const std::exception&e){std::cerr<<"single-fixed: "<<e.what()<<'\n';return 2;}
}
