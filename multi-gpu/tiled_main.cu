// Driver core shared by the QR-Omega executables: command-line options, input generation, timing and
// JSON reporting.
#include "schedule.hpp"
#include "phase_range.hpp"
using namespace tqr;
RunOptions parse_options(int argc,char**argv){RunOptions o;for(int i=1;i<argc;++i){std::string a=argv[i];auto val=[&](){if(++i==argc)throw std::runtime_error("missing_option");return std::string(argv[i]);};
 if(a=="--mode")o.mode=val();else if(a=="--precision")o.dtype=val();else if(a=="--m")o.m=std::stoi(val());else if(a=="--n")o.n=std::stoi(val());else if(a=="--b")o.b=std::stoi(val());else if(a=="--leaf")o.leaf=std::stoi(val());else if(a=="--strip")o.strip=std::stoi(val());else if(a=="--radix")o.radix=std::stoi(val());else if(a=="--gradix")o.gradix=std::stoi(val());
 // --c forces the apply carrier's contraction replication and --d the within-elimination pipeline
 // depth. Both menus start at 2, so these flags exist to build the ABLATION arm: without a way to
 // force 1 there is no arm to compare the selected >1 against, and "2 is better than 1" stays
 // unverified.
 else if(a=="--c")o.c=std::stoi(val());else if(a=="--d")o.d=std::stoi(val());
 else if(a=="--zc")o.zc=std::stoi(val());else if(a=="--dc")o.dc=std::stoi(val());
 else if(a=="--wcb")o.wcb=std::stoi(val());else if(a=="--zcb")o.zcb=std::stoi(val());else if(a=="--dcb")o.dcb=std::stoi(val());
 else if(a=="--csw")o.csw=std::stoi(val());
 else if(a=="--threads")o.threads=std::stoi(val());else if(a=="--reps")o.reps=std::stoi(val());else if(a=="--pad")o.pad=std::stoi(val());else if(a=="--profile")o.profile=val();else if(a=="--plan")o.plan_file=val();else if(a=="--output")o.output=val();else if(a=="--workspace")o.budget=std::stoull(val());else if(a=="--input")o.input=val();else if(a=="--no-validation")o.validate=false;else if(a=="--no-reference")o.reference=false;else throw std::runtime_error("unknown_option_"+a);
 }if(o.dtype!="fp32"&&o.dtype!="fp64")throw std::runtime_error("precision");if(o.m<0||o.n<0||o.reps<1||o.pad<0)throw std::runtime_error("descriptor");return o;}
template<class T> __global__ void regenerate(T*a,int nr,int n,int ld,int begin,int m,int first_col,int kind,T scale){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(nr)*n;i+=size_t(blockDim.x)*gridDim.x){int r=i%nr,j=first_col+i/nr,jj=kind==2?0:j;T x=T(int(mix64(uint64_t(begin+r)+uint64_t(jj)*std::max(m,1)+73)%2001)-1000)/T(1000);if(kind==1)x=0;if(kind==5&&begin+r==0&&j==0)x=std::numeric_limits<T>::infinity();if(kind==6&&begin+r==0&&j==0)x=std::numeric_limits<T>::quiet_NaN();a[r+size_t(i/nr)*ld]=x*scale;}
}
template<class T> __global__ void r_columns(const T*a,int lda,T*r,int ldr,int nr,int q,int begin,int first_col){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size_t(nr)*q;i+=size_t(blockDim.x)*gridDim.x){int row=i%nr,j=i/nr;r[row+size_t(j)*ldr]=begin+row<=first_col+j?a[row+size_t(first_col+j)*lda]:T(0);}}
template<class T,class E> json validate_streamed(E&engine,TiledMatrix<T>&a,Context&ctx,int strip,int kind,T scale,const TiledPlan&plan){
 double start=seconds(),worst_residual=0,worst_inverse=0;int status=0,qmax=std::min(strip,512);TiledMatrix<T>x(a.m,qmax,ctx.size,ctx.rank),original(a.m,qmax,ctx.size,ctx.rank);Stream st;size_t checked_columns=0;
 for(int col=0;col<a.n;col+=qmax){int q=std::min(qmax,a.n-col);x.n=original.n=q;r_columns<<<256,128,0,st>>>(a.a.p,a.ld,x.a.p,x.ld,x.nr,q,x.begin,col);regenerate<<<256,128,0,st>>>(original.a.p,original.nr,q,original.ld,original.begin,a.m,col,kind,scale);st.sync();status=std::max(status,engine.apply_q(x,false));double residual=relative(x.a.p,original.a.p,x.nr,q,x.ld,original.ld);worst_residual=!std::isfinite(residual)?INFINITY:std::max(worst_residual,residual);checked_columns+=q;}
 x.n=original.n=std::min(qmax,16);regenerate<<<256,128,0,st>>>(x.a.p,x.nr,x.n,x.ld,x.begin,a.m,19,0,T(1));if(x.nr&&x.n)CU(cudaMemcpyAsync(original.a.p,x.a.p,size_t(x.ld)*x.n*sizeof(T),cudaMemcpyDeviceToDevice,st));st.sync();status=std::max(status,engine.apply_q(x,false));status=std::max(status,engine.apply_q(x,true));worst_inverse=relative(x.a.p,original.a.p,x.nr,x.n,x.ld,original.ld);
 auto ranks=ctx.all_records({{"residual_max_column_block_relative",worst_residual},{"random_Q_QT_inverse_relative",worst_inverse},{"status",status}});for(auto&r:ranks){worst_residual=std::max(worst_residual,r["residual_max_column_block_relative"].template get<double>());worst_inverse=std::max(worst_inverse,r["random_Q_QT_inverse_relative"].template get<double>());}
 double u=working_unit_roundoff<T>(),tol=std::min(0.01,32*u*(std::sqrt(double(std::max(1,a.m)))+std::sqrt(double(std::max(1,a.n)))+plan.b)*(1+std::ceil(std::log2(std::max(1,ceildiv(a.m,plan.leaf))))));bool pass=status==0&&std::isfinite(worst_residual)&&std::isfinite(worst_inverse)&&worst_residual<=tol&&worst_inverse<=tol;
 return {{"pass",pass},{"residual_max_rank_column_block_relative",worst_residual},{"Q_QT_random_inverse_relative",worst_inverse},{"input_columns_checked",checked_columns},{"full_residual_coverage",true},{"orthogonality_evidence","16 deterministic pseudorandom vectors; finite sketch, not a full Gram matrix"},{"engineering_tolerance",tol},{"unit_roundoff",u},{"validation_s",seconds()-start},{"rank_results",ranks}};
}
template<class T> class LargeVendorQR {
 cusolverDnHandle_t h=nullptr;cusolverDnParams_t params=nullptr;Buffer<T>tau;Buffer<unsigned char>device;std::vector<unsigned char>host;Buffer<int>info;Stream stream;int m,n,ld;
 static constexpr cudaDataType type=sizeof(T)==4?CUDA_R_32F:CUDA_R_64F;
public:
 LargeVendorQR(int rows,int cols,int stride,T*a):tau(std::min(rows,cols)),info(1),m(rows),n(cols),ld(stride){solver_check(cusolverDnCreate(&h));solver_check(cusolverDnCreateParams(&params));solver_check(cusolverDnSetStream(h,stream));solver_check(cusolverDnSetMathMode(h,CUSOLVER_DEFAULT_MATH));size_t d=0,s=0;if(m&&n)solver_check(cusolverDnXgeqrf_bufferSize(h,params,m,n,type,a,ld,type,tau.p,type,&d,&s));device.alloc(d);host.resize(s);}
 void factor(T*a){if(!m||!n)return;solver_check(cusolverDnXgeqrf(h,params,m,n,type,a,ld,type,tau.p,type,device.p,device.n,host.data(),host.size(),info.p));stream.sync();if(info.download()[0])throw std::runtime_error("reference_info");}
 json memory()const{return {{"device_workspace_bytes",device.n},{"host_workspace_bytes",host.size()},{"tau_bytes",tau.n*sizeof(T)}};}
 ~LargeVendorQR(){if(params)cusolverDnDestroyParams(params);if(h)cusolverDnDestroy(h);}
};
template<class T> json run(const RunOptions&o,Context&ctx,const json&profile){
 double begin=seconds();TiledPlan plan;if(ctx.rank==0){
  if(o.plan_file.empty()){
   plan=tiled_select(o,ctx.size,sizeof(T),profile);
   // rank zero selects once; fill the authoritative schedule here (shared encoder). Summaries are
   // never replay inputs.
   if(!plan.scalar){
    plan.qr_schedule=tiled_schedule(plan,sizeof(T));
    plan.schedule_digest=tiled_schedule_digest(plan.qr_schedule);
   } else {plan.qr_schedule=json::object();plan.schedule_digest="scalar";}
   plan.key=digest(plan.record().dump());
  }
  else {auto saved=read_json(o.plan_file);auto unsigned_plan=saved;unsigned_plan.erase("artifact_digest");if(saved.at("artifact_digest")!=digest(unsigned_plan.dump())||saved.at("m")!=o.m||saved.at("n")!=o.n||saved.at("active")!=ctx.size||saved.at("profile_id")!=profile.at("id")||saved.at("catalog")!=tiled_catalog)throw std::runtime_error("incompatible_tiled_plan");
   if(saved.value("scalar",false)){plan=tiled_scalar_instantiate(o.m,o.n,saved["threads"],sizeof(T),o.pad);plan.selection=saved.value("selection",json::object());plan.selection["replay_status"]="compatible_plan_replay";plan.key=saved.value("key",std::string());plan.profile_id=saved.value("profile_id",std::string());}
   else{
    if(!saved.contains("qr_schedule")||!saved.at("qr_schedule").is_object())throw std::runtime_error("schedule_missing_execution_field_before_modify:qr_schedule");
    if(saved.value("schedule_digest",std::string())!=tiled_schedule_digest(saved.at("qr_schedule")))throw std::runtime_error("schedule_digest_before_modify");
    plan=tiled_restore_from_schedule(saved.at("qr_schedule"),sizeof(T));
    tiled_charge_backend(plan,sizeof(T),profile);
    plan.ge_shared=saved.value("ge_shared",false);plan.tt_shared=saved.value("tt_shared",false);plan.small_shared=saved.value("small_shared",false);plan.tt_sparse=saved.value("tt_sparse",false);
    plan.selection=saved.value("selection",json::object());plan.selection["replay_status"]="compatible_plan_replay";
    plan.forcing=saved.value("forcing",json::object());plan.qr_schedule=saved.at("qr_schedule");plan.schedule_digest=saved.value("schedule_digest",std::string());
    plan.key=saved.value("key",std::string());plan.profile_id=saved.value("profile_id",std::string());
    if(plan.record()["owned_bytes_per_rank"]!=saved["owned_bytes_per_rank"]||*std::max_element(plan.owned.begin(),plan.owned.end())>o.budget)throw std::runtime_error("plan_tiled_inventory");
   }}
 }auto chosen=ctx.all_records(ctx.rank==0?plan.record():json(nullptr))[0];
 if(ctx.rank!=0){
  // Every rank verifies the complete schedule digest and its assigned descriptor subset before any
  // factor modifies A.
  if(chosen.value("scalar",false)){plan=tiled_scalar_instantiate(o.m,o.n,chosen["threads"],sizeof(T),o.pad);plan.selection=chosen.value("selection",json::object());plan.profile_id=chosen.value("profile_id",std::string());plan.key=chosen.value("key",std::string());}
  else{
   if(!chosen.contains("qr_schedule")||!chosen.at("qr_schedule").is_object())throw std::runtime_error("schedule_missing_execution_field_before_modify:qr_schedule");
   if(chosen.value("schedule_digest",std::string())!=tiled_schedule_digest(chosen.at("qr_schedule")))throw std::runtime_error("schedule_digest_before_modify");
   plan=tiled_restore_from_schedule(chosen.at("qr_schedule"),sizeof(T));
   tiled_charge_backend(plan,sizeof(T),profile);
   plan.ge_shared=chosen.value("ge_shared",false);plan.tt_shared=chosen.value("tt_shared",false);plan.small_shared=chosen.value("small_shared",false);plan.tt_sparse=chosen.value("tt_sparse",false);
   plan.selection=chosen.value("selection",json::object());plan.forcing=chosen.value("forcing",json::object());plan.qr_schedule=chosen.at("qr_schedule");plan.schedule_digest=chosen.value("schedule_digest",std::string());
   plan.profile_id=chosen.value("profile_id",std::string());plan.key=chosen.value("key",std::string());
  }
 }
 json record={{"precision",precision<T>()},{"m",o.m},{"n",o.n},{"gpus",ctx.size},{"plan",plan.record()},{"profile_id",profile.at("id")}};int kind=o.input=="zero"?1:o.input=="rank_deficient"?2:o.input=="inf"?5:o.input=="nan"?6:0;T scale=o.input=="near_large"?std::numeric_limits<T>::max()/(T(1024)*T(std::max(1,o.m+o.n))):o.input=="near_small"?std::numeric_limits<T>::min()*T(128):T(1);
 std::unique_ptr<TiledMatrix<T>>a;Stream st;std::vector<double>times;int status=0;
 auto reset=[&](){if(a->nr&&a->n)regenerate<<<256,128,0,st>>>(a->a.p,a->nr,a->n,a->ld,a->begin,a->m,0,kind,scale);st.sync();};
 {double setup=seconds();TiledEngine<T>engine(ctx,plan,o.budget);a=std::make_unique<TiledMatrix<T>>(o.m,o.n,ctx.size,ctx.rank,o.pad);record["allocation_backend_s"]=seconds()-setup;std::vector<double>skews;
     for(int rep=0;rep<=o.reps;++rep){reset();ctx.barrier();double t=seconds();{PhaseRange tqr_rep(rep?"tqr:timed_factorization":"tqr:warmup_factorization");status=engine.factor(*a);}ctx.barrier();double end=seconds();auto rank_times=ctx.all_records({{"begin",t},{"end",end}});double first=t,last=end,latest=t;for(auto&r:rank_times){first=std::min(first,r["begin"].template get<double>());latest=std::max(latest,r["begin"].template get<double>());last=std::max(last,r["end"].template get<double>());}if(rep){times.push_back(last-first);skews.push_back(latest-first);}else record["first_execution_s"]=last-first;if(status)break;}
  record["status"]=status;record["rank_evidence"]=ctx.all_records(engine.evidence());
  {std::string mine = plan.scalar ? std::string("scalar") : plan.schedule_digest;
   record["schedule_digest_per_rank"]=ctx.all_records(mine);
   if(!plan.scalar)record["execution_plan_sha256"]=plan.schedule_digest;}
  record["timing"]={{"reused_host_s",times},{"reused_median_s",median(times)},{"launch_skew_s",skews},{"boundary","common same-node monotonic release through all CUDA completions and MPI barrier; ready distributed A to in-place R/native Q handle"},{"input_reset","regenerated on GPU outside factorization timing"}};
  if(status==NONFINITE_INPUT)record["validation"]={{"pass",kind==5||kind==6},{"expected_error",true},{"input_unchanged_contract","global finite scan completes before any factor modifies A; byte check in contract suite"}};
   else if(o.validate&&status==0){PhaseRange tqr_valid("tqr:validation");record["validation"]=validate_streamed(engine,*a,ctx,plan.strip,kind,scale,plan);}
 }
 if(o.reference&&ctx.size==1&&status==0){LargeVendorQR<T>reference(o.m,o.n,a->ld,a->a.p);std::vector<double>rt;for(int rep=0;rep<=o.reps;++rep){reset();double t=seconds();reference.factor(a->a.p);if(rep)rt.push_back(seconds()-t);}record["reference"]={{"name","cuSOLVER Xgeqrf"},{"math","selected precision, DEFAULT_MATH, NVIDIA_TF32_OVERRIDE=0"},{"raw_s",rt},{"median_s",median(rt)},{"slowdown",median(rt)>0?median(times)/median(rt):0},{"memory",reference.memory()},{"output","R and conventional full-Q Householder handle"}};}
 record["cell_end_to_end_s"]=seconds()-begin;return record;
}
template<class T> int execute(const RunOptions&o,Context&ctx){auto hardware=discover(ctx);if(hardware.value("cpu_affinity",std::string())!=required_cpu_set()&&!(hardware.value("cpu_affinity",std::string())=="0-11"&&hardware.value("measurement_class",std::string())=="contended-development"))throw std::runtime_error("required_cpu_affinity_12_15");
 auto emit=[&](json j){if(ctx.rank==0){if(o.output.empty())std::cout<<j.dump(2)<<'\n';else write_json(o.output,j);}};
 if(o.mode=="discover"){hardware["tiled_catalog"]=tiled_catalog;emit(hardware);return 0;}
 if(o.mode=="resources"){auto r=kernel_resources<T>(ctx.device);r["hardware"]=hardware;emit(r);return 0;}
 if(o.mode=="protocol"){auto j=qualify_protocol<T>(ctx);j["hardware"]=hardware;emit(j);return j.value("pass",false)?0:1;}
 if(o.mode=="probe"){auto p=tiled_probe<T>(ctx,hardware);emit(p);return p["probe_status"].template get<int>()||p["tiled_probe_status"].template get<int>();}
 auto profile=read_json(o.profile);auto unsigned_profile=profile;unsigned_profile.erase("id");if(profile.at("id")!=digest(unsigned_profile.dump()))throw std::runtime_error("profile_integrity");if(profile.at("hardware").at("stable_key")!=hardware.at("stable_key")||profile.at("precision")!=precision<T>()||profile.at("catalog")!=tiled_catalog)throw std::runtime_error("incompatible_profile");
 RunOptions effective=o;auto memories=ctx.all_records(hardware.at("hbm_free_observed_bytes"));for(auto&free:memories)effective.budget=std::min(effective.budget,free.template get<size_t>());auto result=run<T>(effective,ctx,profile);result["measurement_class"]=hardware["measurement_class"];result["hardware"]=hardware;emit(result);return result.at("status").template get<int>()!=0&&!result.value("validation",json::object()).value("expected_error",false)||result.contains("validation")&&!result["validation"].value("pass",false)?1:0;
}
#ifndef TQR_LIBRARY_ONLY
int main(int argc,char**argv){try{auto o=parse_options(argc,argv);if(o.mode=="plan"){auto profile=read_json(o.profile);if(profile["hardware"]["binary_key"]!=binary_key())throw std::runtime_error("plan_binary_mismatch");auto p=tiled_select(o,profile["hardware"]["ranks"],o.dtype=="fp32"?4:8,profile);
 if(!p.scalar&&p.qr_schedule.empty()){p.qr_schedule=tiled_schedule(p,o.dtype=="fp32"?4:8);p.schedule_digest=tiled_schedule_digest(p.qr_schedule);p.key=digest(p.record().dump());}
 auto j=p.record();j["artifact_digest"]=digest(j.dump());if(o.output.empty())std::cout<<j.dump(2)<<'\n';else write_json(o.output,j);return 0;}Context ctx;return o.dtype=="fp32"?execute<float>(o,ctx):execute<double>(o,ctx);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
#endif
