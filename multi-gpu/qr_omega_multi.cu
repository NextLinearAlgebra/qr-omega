// QR-Omega on the GPUs of one node (one MPI rank per GPU): factors a generated block-cyclic matrix with
// an explicit configuration, times the factorization, and validates the result.
#define TQR_MULTI_FIXED_DRIVER 1
#define TQR_X3_PROMOTE 2
#define TQR_X3_LOCAL_PROMOTE 1
#define TQR_LIBRARY_ONLY
#include "tiled_main.cu"
#include "multi_gram_validation.cuh"

// Native row-block-cyclic input and validation. Global row of local row l on rank r: (l/blk)*blk*p + r*blk + l%blk
// (RowMap::to_global). Values are EXACTLY regenerate()'s function of (global row, col).
template<class T> __global__ void regenerate_cyclic(T*a,int nr,int n,int ld,int blk,int p,int rank,int m,int first_col,int kind,T scale){
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(nr)*n;i+=size_t(blockDim.x)*gridDim.x){
  const int l=int(i%nr),j=first_col+int(i/nr),jj=kind==2?0:j;
  const long long g=(long long)(l/blk)*blk*p+(long long)rank*blk+l%blk;
  T x=T(int(mix64(uint64_t(g)+uint64_t(jj)*std::max(m,1)+73)%2001)-1000)/T(1000);
  if(kind==1)x=0;if(kind==5&&g==0&&j==0)x=std::numeric_limits<T>::infinity();if(kind==6&&g==0&&j==0)x=std::numeric_limits<T>::quiet_NaN();
  a[l+size_t(i/nr)*ld]=x*scale;}
}
template<class T> __global__ void r_columns_cyclic(const T*a,int lda,T*r,int ldr,int nr,int q,int blk,int p,int rank,int first_col){
 for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x;i<size_t(nr)*q;i+=size_t(blockDim.x)*gridDim.x){
  const int l=int(i%nr),j=int(i/nr);const long long g=(long long)(l/blk)*blk*p+(long long)rank*blk+l%blk;
  r[l+size_t(j)*ldr]=g<=first_col+j?a[l+size_t(first_col+j)*lda]:T(0);}
}
// Same checks and tolerance as validate_streamed (tiled_main.cu), on the native cyclic layout: every column block
// of R is multiplied by Q (the retained factors) and compared with the regenerated input; 16 pseudorandom vectors
// go through Q then Q^T.
template<class T,class E> json validate_native(E&engine,Context&ctx,int strip,int kind,T scale,const TiledPlan&plan){
 double start=seconds(),worst_residual=0,worst_inverse=0;int status=0,qmax=std::min(strip,512);
 auto&A=engine.native_matrix();const RowMap&mp=engine.row_map();
 TiledMatrix<T>x(A.m,qmax,TiledLocalShape{A.nr,0}),original(A.m,qmax,TiledLocalShape{A.nr,0});Stream st;size_t checked_columns=0;
 for(int col=0;col<A.n;col+=qmax){const int q=std::min(qmax,A.n-col);x.n=original.n=q;
  r_columns_cyclic<<<256,128,0,st>>>(A.a.p,A.ld,x.a.p,x.ld,x.nr,q,mp.blk,mp.p,ctx.rank,col);
  regenerate_cyclic<<<256,128,0,st>>>(original.a.p,original.nr,q,original.ld,mp.blk,mp.p,ctx.rank,A.m,col,kind,scale);st.sync();
  status=std::max(status,engine.apply_q_native(x,false));
  const double residual=relative(x.a.p,original.a.p,x.nr,q,x.ld,original.ld);
  worst_residual=!std::isfinite(residual)?INFINITY:std::max(worst_residual,residual);checked_columns+=q;}
 x.n=original.n=std::min(qmax,16);
 regenerate_cyclic<<<256,128,0,st>>>(x.a.p,x.nr,x.n,x.ld,mp.blk,mp.p,ctx.rank,A.m,19,0,T(1));
 if(x.nr&&x.n)CU(cudaMemcpyAsync(original.a.p,x.a.p,size_t(x.ld)*x.n*sizeof(T),cudaMemcpyDeviceToDevice,st));st.sync();
 status=std::max(status,engine.apply_q_native(x,false));status=std::max(status,engine.apply_q_native(x,true));
 worst_inverse=relative(x.a.p,original.a.p,x.nr,x.n,x.ld,original.ld);
 auto ranks=ctx.all_records({{"residual_max_column_block_relative",worst_residual},{"random_Q_QT_inverse_relative",worst_inverse},{"status",status}});
 for(auto&r:ranks){worst_residual=std::max(worst_residual,r["residual_max_column_block_relative"].template get<double>());worst_inverse=std::max(worst_inverse,r["random_Q_QT_inverse_relative"].template get<double>());}
 const double u=working_unit_roundoff<T>(),tol=std::min(0.01,32*u*(std::sqrt(double(std::max(1,A.m)))+std::sqrt(double(std::max(1,A.n)))+plan.b)*(1+std::ceil(std::log2(std::max(1,ceildiv(A.m,plan.leaf))))));
 const bool pass=status==0&&std::isfinite(worst_residual)&&std::isfinite(worst_inverse)&&worst_residual<=tol&&worst_inverse<=tol;
 return {{"pass",pass},{"layout","native row-block-cyclic"},{"residual_max_rank_column_block_relative",worst_residual},{"Q_QT_random_inverse_relative",worst_inverse},{"input_columns_checked",checked_columns},{"full_residual_coverage",true},{"orthogonality_evidence","16 deterministic pseudorandom vectors; finite sketch, not a full Gram matrix"},{"engineering_tolerance",tol},{"unit_roundoff",u},{"validation_s",seconds()-start},{"rank_results",ranks}};
}

struct FixedOptions {
 int m=4096,n=4096,b=128,leaf=8192,strip=4096,threads=256;
 int wc=2,zc=2,dc=2,depth=2,radix=2,gradix=2,aggregate=1,lookahead=0,row_block=0;
 int groups=0,pw=16,panel_threads=512,window=64,reps=3,free_sms=0;
 bool capacity_only=false,native_cyclic=false,validate=true;
 double max_factor_seconds=0;
 std::string precision="fp64",output,input="random";
};

FixedOptions fixed_options(int argc,char**argv){
 FixedOptions o;
 for(int i=1;i<argc;++i){
  const std::string k=argv[i];
  auto val=[&](){if(++i==argc)throw std::runtime_error("missing_option:"+k);return std::string(argv[i]);};
  if(k=="--capacity-only")o.capacity_only=true;
  else if(k=="--native-cyclic")o.native_cyclic=true;
  else if(k=="--no-validate")o.validate=false;
  else if(k=="--max-factor-seconds")o.max_factor_seconds=std::stod(val());
  else if(k=="--m")o.m=std::stoi(val());else if(k=="--n")o.n=std::stoi(val());
  else if(k=="--b")o.b=std::stoi(val());else if(k=="--leaf")o.leaf=std::stoi(val());
  else if(k=="--strip")o.strip=std::stoi(val());else if(k=="--threads")o.threads=std::stoi(val());
  else if(k=="--wc")o.wc=std::stoi(val());else if(k=="--zc")o.zc=std::stoi(val());
  else if(k=="--dc")o.dc=std::stoi(val());else if(k=="--d")o.depth=std::stoi(val());
  else if(k=="--radix")o.radix=std::stoi(val());else if(k=="--gradix")o.gradix=std::stoi(val());
  else if(k=="--aggregate")o.aggregate=std::stoi(val());else if(k=="--lookahead")o.lookahead=std::stoi(val());
  else if(k=="--row-block")o.row_block=std::stoi(val());else if(k=="--free-sms")o.free_sms=std::stoi(val());
  else if(k=="--panel-groups")o.groups=std::stoi(val());else if(k=="--panel-width")o.pw=std::stoi(val());
  else if(k=="--panel-threads")o.panel_threads=std::stoi(val());else if(k=="--panel-window")o.window=std::stoi(val());
  else if(k=="--reps")o.reps=std::stoi(val());else if(k=="--precision")o.precision=val();
  else if(k=="--fp32-math")fp32_math()=parse_fp32_math(val());else if(k=="--output")o.output=val();
  else if(k=="--input")o.input=val();else throw std::runtime_error("unknown_option:"+k);
 }
 if(o.m<1||o.n<1||o.b<1||o.b>512||o.leaf<o.b||o.strip<1||o.reps<1||o.depth<1||
    o.wc<2||o.zc<2||o.dc<2||o.aggregate<1||o.aggregate>4||o.lookahead<0||o.lookahead>1||
    !std::isfinite(o.max_factor_seconds)||o.max_factor_seconds<0||
    (o.precision!="fp32"&&o.precision!="fp64"))throw std::runtime_error("invalid_fixed_schedule_or_c_not_gt_one");
 return o;
}

template<class T> int fixed_run(const FixedOptions&o,Context&ctx){
 if(ctx.size<2)throw std::runtime_error("multi_gpu_only");
 const int q_form_width=multi_q_formation_width(o.m); // Validate before allocating or modifying input.
 const char*paired_w=std::getenv("TQR_MULTI_W_MMA8_PAIR");
 const char*direct_w=std::getenv("TQR_MULTI_W_MMA8");
 if(paired_w&&std::string(paired_w)!="0"&&std::string(paired_w)!="1")
  throw std::runtime_error("invalid_multi_W_mma8_pair");
 if(paired_w&&std::string(paired_w)=="1"&&direct_w&&std::string(direct_w)=="1")
  throw std::runtime_error("conflicting_multi_W_mma8_routes");
 const char*x3_d=std::getenv("TQR_MULTI_X3_PROMOTED_D");
 gmma_detail::multi_x3_promoted_d()=x3_d&&std::string(x3_d)=="1";
 const char*x3_w=std::getenv("TQR_MULTI_X3_IN_KERNEL_W");
 gmma_detail::multi_x3_in_kernel_w()=x3_w&&std::string(x3_w)=="1";
 const char*kami_tile=std::getenv("TQR_MULTI_KAMI_D_TILE");
 if(kami_tile&&std::string(kami_tile)!="128"&&std::string(kami_tile)!="64"&&std::string(kami_tile)!="64s2")
  throw std::runtime_error("invalid_multi_kami_D_tile");
 const char*kami_w_tile=std::getenv("TQR_MULTI_KAMI_W_TILE");
 if(kami_w_tile&&std::string(kami_w_tile)!="64x128"&&std::string(kami_w_tile)!="128x64")
  throw std::runtime_error("invalid_multi_kami_W_tile");
 const char*kami_raster=std::getenv("TQR_MULTI_KAMI_D_RASTER");
 if(kami_raster&&std::atoi(kami_raster)!=8&&std::atoi(kami_raster)!=16&&std::atoi(kami_raster)!=32)
  throw std::runtime_error("invalid_multi_kami_D_raster");
 const bool wide=o.b>128;
 if(wide){
  const char*enabled=std::getenv("TQR_MULTI_WIDE_PANEL");
  if(!enabled||std::string(enabled)!="1"||o.aggregate!=1||o.row_block!=o.b||o.groups<2||o.pw<8||o.window<1||o.window>128)
   throw std::runtime_error("wide_multi_panel_requires_explicit_windowed_configuration");
 }

 const auto hardware=discover(ctx);
 if(hardware.at("cpu_affinity")!=required_cpu_set())throw std::runtime_error("required_cpu_affinity");
 auto p=tiled_instantiate(o.m,o.n,ctx.size,o.b,o.leaf,o.strip,o.threads,sizeof(T),0,
                         o.radix,o.depth,o.wc,o.gradix,o.zc,o.dc,2,2,2,o.aggregate,o.lookahead,o.row_block,wide);
 p.la_free=o.free_sms;
 if(o.groups){
  if(o.groups<2||o.pw<1||o.window<0||o.window>o.b)throw std::runtime_error("invalid_panel_carrier");
  for(auto&pan:p.panels)for(auto&batch:pan.ranks)
   if(batch.ge.size()==1&&((batch.ge[0].rows>=coop_min_rows&&batch.h==p.b)||(wide&&batch.ge[0].h>128))){
    batch.cc=o.groups;batch.cpw=o.pw;batch.cthreads=o.panel_threads;batch.csw=o.window;
   }
 }
 if(wide)for(const auto&pan:p.panels){
  if(pan.global.size()>1)throw std::runtime_error("wide_multi_requires_one_global_node");
  for(const auto&batch:pan.ranks)if(!batch.levels.empty()||batch.ge.size()>1)
   throw std::runtime_error("wide_multi_requires_one_leaf_per_active_rank");
 }
 p.profile_id="manual-fixed-multi";
 p.selection={{"status","explicit_configuration"},{"selector_used",false},{"timing_samples_used",false},
              {"scope","This schedule is a candidate; numerical and executed-carrier evidence determine eligibility."}};
 // Query actual placement/granularity; do not borrow timing samples or identities
 // from the old allocation containing a different (clock-capped) GPU.
 const json capacity={{"hardware",hardware},{"library_overhead_allowance_bytes",0}};
 tiled_charge_backend(p,sizeof(T),capacity);
 size_t budget=SIZE_MAX;
 for(const auto&free:ctx.all_records(hardware.at("hbm_free_observed_bytes")))budget=std::min(budget,free.template get<size_t>());
 p.budget_bytes=budget;
 p.qr_schedule=tiled_schedule(p,sizeof(T));
 if(wide){p.qr_schedule["multi_wide_panel"]=true;p.qr_schedule["wide_T_construction"]="128-column dlarft blocks; ordered T12=-(T1(V1^T V2))T2 with two disjoint contraction slices and one owner write";}
 const bool global_la=std::getenv("TQR_MULTI_GLOBAL_LA")&&std::string(std::getenv("TQR_MULTI_GLOBAL_LA"))=="1";
 if(global_la){
  auto enabled=[](const char*key){const char*v=std::getenv(key);return v&&std::string(v)=="1";};
  if(!o.lookahead||o.aggregate!=1||o.row_block!=o.b||!enabled("TQR_MEMBER_COLUMNS")||!enabled("TQR_GLOBAL_PIPELINE"))
   throw std::runtime_error("global_lookahead_requires_cyclic_panel_blocks_and_ordered_member_pipeline");
  p.qr_schedule["multi_global_tail_lookahead"]=true;
 }
 const bool leaf_la=std::getenv("TQR_MULTI_LEAF_LA")&&std::string(std::getenv("TQR_MULTI_LEAF_LA"))=="1";
 if(leaf_la){
  if(!global_la)throw std::runtime_error("leaf_global_lookahead_requires_global_lookahead");
  for(const auto&pan:p.panels){
   if(pan.global.size()>1)throw std::runtime_error("leaf_global_lookahead_requires_one_global_node");
   for(const auto&batch:pan.ranks)
    if(!batch.levels.empty()||batch.ge.size()>1)throw std::runtime_error("leaf_global_lookahead_requires_one_leaf_per_active_rank");
  }
  p.qr_schedule["multi_leaf_global_lookahead"]=true;
  p.qr_schedule["global_child_history_requirement"]="through next-panel release; far GE precedes far global Apply on one lane group";
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
 json record={{"driver","multi-fixed-v1"},{"precision",precision<T>()},{"precision_mode",precision_mode_name<T>()},
              {"m",o.m},{"n",o.n},{"gpus",ctx.size},{"input",o.input},{"hardware",hardware},
              {"measurement_class",hardware.at("measurement_class")},{"schedule_digest_per_rank",schedule_digests},
              {"plan",{{"member","compact"},{"member_plan",p.record()}}},
              {"capacity_policy","requested inventory plus actual observed allocation checked before modifying A; no unmeasured library allowance claim"}};
 record["plan"]["member_plan"]["carrier_selection"]="explicit candidate; no measured selector invoked; actual executed carriers are recorded per rank";
 record["plan"]["member_plan"]["c_d_source"]="multi-fixed command-line configuration; no DP or calibrated model claim";
 if(o.capacity_only){
  const int nr=row_begin(o.m,ctx.size,ctx.rank+1)-row_begin(o.m,ctx.size,ctx.rank),width=std::min(o.m,512);
  const bool reuse=std::getenv("TQR_MULTI_GRAM_REUSE_OUTPUT")&&std::string(std::getenv("TQR_MULTI_GRAM_REUSE_OUTPUT"))=="1";
  const size_t qwords=checked_mul(size_t(descriptor_ld(nr,0)),size_t(o.m));
  const size_t gwords=checked_mul(size_t(o.m),size_t(width));
  const size_t blockwords=checked_mul(size_t(nr+p.b+1),size_t(q_form_width));
  const size_t gram_extra=checked_add(checked_mul(checked_add(checked_add(reuse?0:qwords,gwords),2*blockwords),sizeof(T)),64ULL<<20);
  const size_t owned=p.owned[ctx.rank],free=hardware.at("hbm_free_observed_bytes").template get<size_t>();
  record["capacity_preflight"]=ctx.all_records({{"rank",ctx.rank},{"factor_inventory_bytes",owned},{"full_gram_extra_bytes",gram_extra},
    {"free_bytes",free},{"factor_estimated_fits",owned<=free},{"factor_and_gram_estimated_fits",checked_add(owned,gram_extra)<=free},
    {"Q_formation_block_columns",q_form_width},{"gram_block_columns",width},
    {"scope","production requested inventory before allocation; actual library/setup overhead still must be checked"}});
  record["capacity_only"]=true;record["eligible_numerically"]=false;
  if(ctx.rank==0){if(o.output.empty())std::cout<<record.dump(2)<<'\n';else write_json(o.output,record);}
  return 0;
 }
 record["x3_arithmetic"]={{"gmma_promotion_k_tiles",TQR_X3_PROMOTE},{"promotion_scope","IEEE additions within each K peer; one final fixed-order peer join"},{"gmma_x3_tile_columns",gmma_detail::X3TileN},{"in_kernel_W_split",gmma_detail::multi_x3_in_kernel_w()},{"promoted_three_segment_D",gmma_detail::multi_x3_promoted_d()},{"two_launch_W_owner_policy","first arithmetic term to charged scratch, second reads scratch and commits owner W once; c>2 uses existing partial slices before final combine"},{"scope","multi-GPU fixed driver only; unchanged disjoint K partitions and single output commit"}};
 record["backend_counter_scope"]="transport backend counters are cumulative since engine construction; device carrier evidence and host phase accounting refer to the reported factorization";
 record["execution_controls"]=json::object();
 for(const char*key:{"TQR_MULTI_KAMI_W_TILE","TQR_LL_MULTICAST","TQR_MULTI_W_MMA8_PAIR","TQR_MULTI_Q_FORM_WIDTH"}){
  const char*value=std::getenv(key);record["execution_controls"][key]=value?json(value):json(nullptr);
 }
 for(const char*key:{"TQR_MEMBER_COLUMNS","TQR_GLOBAL_PIPELINE","TQR_GLOBAL_GAU_TT","TQR_SKINNY_W_SPLIT","TQR_W_PEERS","TQR_LL_CTAS","TQR_LL_ELEMS","TQR_MULTI_COPY_CTAS","TQR_MULTI_GLOBAL_LA","TQR_MULTI_LEAF_LA","TQR_MULTI_WIDE_PANEL","TQR_MULTI_X3_PROMOTED_D","TQR_MULTI_X3_IN_KERNEL_W","TQR_MULTI_FULL_GRAM","TQR_MULTI_KAMI_D_TILE","TQR_MULTI_KAMI_D_RASTER","TQR_MULTI_KAMI_W","TQR_MULTI_W_MMA8","TQR_MULTI_GRAM_REUSE_OUTPUT","TQR_MULTI_GRAM_RECHECK_RESTORE","TQR_STREAM_ORDERED_PUBLISH","TQR_TRANSPORT_HOST_RENDEZVOUS"}){
  const char*value=std::getenv(key);record["execution_controls"][key]=value?json(value):json(nullptr);
 }
 std::vector<double>times,skews;int status=0;bool time_limit_exceeded=false;
 if(o.native_cyclic){
  if(o.row_block<1)throw std::runtime_error("native_cyclic_requires_row_block");
  // The caller-side slab is never allocated: remove it from the per-rank inventory the engine admits against.
  for(int r=0;r<ctx.size;++r){const size_t slab=checked_mul(checked_mul(size_t(descriptor_ld(row_begin(o.m,ctx.size,r+1)-row_begin(o.m,ctx.size,r),0)),size_t(o.n)),sizeof(T));
   if(p.owned[r]>slab)p.owned[r]-=slab;}
  record["input_layout"]={{"kind","native row-block-cyclic"},{"row_block",o.row_block},{"scope","input generated directly in the engine's distribution; no slab copy or redistribution in or out of the timed factorization"}};
  tiled_native_input()=true;
 }
 {
  const double setup=seconds();TiledEngine<T>engine(ctx,p,budget);
  std::unique_ptr<TiledMatrix<T>>ap;if(!o.native_cyclic)ap=std::make_unique<TiledMatrix<T>>(o.m,o.n,ctx.size,ctx.rank,0);
  Stream reset_stream;
  record["allocation_backend_s"]=seconds()-setup;
  for(int rep=0;rep<=o.reps;++rep){
   if(o.native_cyclic){auto&A=engine.native_matrix();const RowMap&mp=engine.row_map();
    regenerate_cyclic<<<256,128,0,reset_stream>>>(A.a.p,A.nr,A.n,A.ld,mp.blk,mp.p,ctx.rank,A.m,0,kind,scale);}
   else{auto&a=*ap;regenerate<<<256,128,0,reset_stream>>>(a.a.p,a.nr,a.n,a.ld,a.begin,a.m,0,kind,scale);}
   reset_stream.sync();
   ctx.barrier();const double start=seconds();
   {PhaseRange range(rep?"tqr:timed_factorization":"tqr:warmup_factorization");status=o.native_cyclic?engine.factor_native():engine.factor(*ap);}
   ctx.barrier();const double end=seconds();
   const auto ranks=ctx.all_records({{"begin",start},{"end",end}});
   double first=start,last=end,latest=start;
   for(const auto&r:ranks){first=std::min(first,r.at("begin").template get<double>());
    latest=std::max(latest,r.at("begin").template get<double>());last=std::max(last,r.at("end").template get<double>());}
   if(rep){times.push_back(last-first);skews.push_back(latest-first);}else record["first_execution_s"]=last-first;
   if(o.max_factor_seconds>0&&last-first>o.max_factor_seconds)time_limit_exceeded=true;
   if(status||time_limit_exceeded)break;
  }
  record["status"]=status;record["rank_evidence"]=ctx.all_records(engine.evidence());
  record["timing"]={{"reused_host_s",times},{"reused_median_s",times.empty()?json(nullptr):json(median(times))},
                    {"launch_skew_s",skews},{"requested_max_factor_seconds",o.max_factor_seconds},{"duration_limit_exceeded",time_limit_exceeded},
                    {"duration_policy","stop further factor repetitions if a completed warmup or timed factorization exceeds the requested limit; validate the completed factors separately"},
                    {"boundary",o.native_cyclic?"common host release through all-rank completion; fresh native row-block-cyclic device input to in-place R and retained full-Q factors":"common host release through all-rank completion; fresh balanced-row device input to original-placement R and retained full-Q factors"}};
  if(!status&&!o.validate)record["validation"]={{"pass",nullptr},{"skipped",true},{"scope","timing-only screening run (--no-validate); not numerically qualified"}};
  else if(!status&&o.native_cyclic){PhaseRange range("tqr:validation");record["validation"]=validate_native(engine,ctx,p.strip,kind,scale,p);}
  else if(!status){auto&a=*ap;PhaseRange range("tqr:validation");record["validation"]=validate_streamed(engine,a,ctx,p.strip,kind,scale,p);
   const char*full_gram=std::getenv("TQR_MULTI_FULL_GRAM");
   if(full_gram&&std::string(full_gram)=="1"){
    record["validation"]["full_gram"]=validate_multi_full_gram(engine,a,ctx,p);
    record["validation"]["pass"]=record["validation"]["pass"].template get<bool>()&&record["validation"]["full_gram"]["pass"].template get<bool>();
    const char*recheck=std::getenv("TQR_MULTI_GRAM_RECHECK_RESTORE");
    if(recheck&&std::string(recheck)=="1"){
     record["validation"]["after_gram_restoration"]=validate_streamed(engine,a,ctx,p.strip,kind,scale,p);
     record["validation"]["pass"]=record["validation"]["pass"].template get<bool>()&&record["validation"]["after_gram_restoration"]["pass"].template get<bool>();
    }
   }
  }
 }
 {const json v=record.value("validation",json::object());const json pv=v.value("pass",json(nullptr));
  record["eligible_numerically"]=status==0&&pv.is_boolean()&&pv.get<bool>();}
 record["c_gt_one_admission"]="not inferred from requested c; inspect executed product and panel receipts";
 if(ctx.rank==0){if(o.output.empty())std::cout<<record.dump(2)<<'\n';else write_json(o.output,record);}
 if(!o.validate&&status==0)return time_limit_exceeded?4:0;
 return record.at("eligible_numerically").get<bool>()?(time_limit_exceeded?4:0):1;
}

int main(int argc,char**argv){
 try{const auto o=fixed_options(argc,argv);Context ctx;
  return o.precision=="fp64"?fixed_run<double>(o,ctx):fixed_run<float>(o,ctx);
 }catch(const std::exception&e){std::cerr<<"multi-fixed: "<<e.what()<<'\n';
  int initialized=0,finalized=0;MPI_Initialized(&initialized);if(initialized)MPI_Finalized(&finalized);
  if(initialized&&!finalized)MPI_Abort(MPI_COMM_WORLD,2);return 2;}
}
