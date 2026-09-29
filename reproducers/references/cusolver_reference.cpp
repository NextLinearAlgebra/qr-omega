// cuSOLVER geqrf benchmark adapter: times the factorization and validates it.
#include "common.hpp"
#include "reference_gpu.hpp"
#include <cusolverDn.h>
#ifdef TQR_REFERENCE_VALIDATE
#include "reference_validation.hpp"
#endif
using namespace tqr;
void solver_check_ref(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cusolver_status_"+std::to_string(s));}
template<class T> int bench(int m,int n,int reps){
 constexpr auto type=sizeof(T)==4?CUDA_R_32F:CUDA_R_64F;
 size_t initial_free,total;CU(cudaMemGetInfo(&initial_free,&total));double setup=seconds();Stream stream;cusolverDnHandle_t handle;cusolverDnParams_t params;
 solver_check_ref(cusolverDnCreate(&handle));solver_check_ref(cusolverDnCreateParams(&params));solver_check_ref(cusolverDnSetStream(handle,stream));const char*mm=std::getenv("TQR_CUSOLVER_MATH");const bool emu=sizeof(T)==4&&mm&&std::string(mm)=="bf16x9";
 solver_check_ref(cusolverDnSetMathMode(handle,emu?CUSOLVER_FP32_EMULATED_BF16X9_MATH:CUSOLVER_DEFAULT_MATH));
 int ld=std::max(1,m);size_t matrix_bytes=checked_mul(checked_mul(size_t(ld),size_t(n)),sizeof(T));
 size_t free_now;CU(cudaMemGetInfo(&free_now,&total));if(matrix_bytes>free_now)throw std::runtime_error("reference_matrix_exceeds_available_memory");
 Buffer<T>a(size_t(ld)*n),tau(std::min(m,n));Buffer<int>info(1);size_t dw=0,hw=0;
 if(m&&n)solver_check_ref(cusolverDnXgeqrf_bufferSize(handle,params,m,n,type,a.p,ld,type,tau.p,type,&dw,&hw));
 CU(cudaMemGetInfo(&free_now,&total));if(dw>free_now)throw std::runtime_error("reference_workspace_exceeds_available_memory");
 Buffer<unsigned char>work(dw);std::vector<unsigned char>host(hw);double setup_s=seconds()-setup,first=0;std::vector<double>times;int status=0;
 for(int rep=0;rep<=reps;++rep){reference_generate(a.p,m,n,ld,m,1024,1,1,0,0,0,sizeof(T));double start=seconds();
  if(m&&n)solver_check_ref(cusolverDnXgeqrf(handle,params,m,n,type,a.p,ld,type,tau.p,type,work.p,dw,host.data(),hw,info.p));stream.sync();double elapsed=seconds()-start;
  if(m&&n)status=info.download()[0];if(rep)times.push_back(elapsed);else first=elapsed;if(status)break;
 }
 auto sorted=times;std::sort(sorted.begin(),sorted.end());
 json validation="info checked; independent full numerical validation separately required";
 bool numerical_pass=true;
#ifdef TQR_REFERENCE_VALIDATE
 if(status==0){
  try{
  work=Buffer<unsigned char>();
  ReferenceOrmqr<T> apply(handle,stream);
  validation=validate_conventional_reference<T>(m,n,1024,
   [&](T*x,int ldx,int col,int q){reference_r(a.p,ld,x,ldx,m,q,1024,1,1,0,0,col,sizeof(T));},
   [&](T*x,int ldx,int q,cublasOperation_t op){apply.apply(m,q,std::min(m,n),a.p,ld,tau.p,x,ldx,op);});
  validation["ormqr_workspace_bytes"]=apply.workspace_bytes();
  validation["ormqr_reflector_block"]=ReferenceOrmqr<T>::reflector_block;
  validation["Q_application"]="conventional cuSOLVER A/tau, blockwise Ormqr; validation excluded from factor timing";
  numerical_pass=validation.value("pass",false);
  }catch(const std::exception&e){validation={{"performed",false},{"completed",false},{"pass",nullptr},{"error",e.what()}};numerical_pass=false;}
 }
#endif
 std::cout<<json{{"reference","cuSOLVER Xgeqrf"},{"precision",precision<T>()},{"m",m},{"n",n},{"gpus",1},{"status",status},{"setup_s",setup_s},{"first_s",first},{"raw_s",times},{"median_s",sorted.empty()?0:sorted[sorted.size()/2]},{"matrix_bytes",matrix_bytes},{"device_workspace_bytes",dw},{"host_workspace_bytes",hw},{"initial_free_bytes",initial_free},{"boundary","ready column-major device input through stream completion; R and conventional Householder Q handle"},{"math",emu?"FP32_EMULATED_BF16X9_MATH (TQR_CUSOLVER_MATH=bf16x9)":"selected precision DEFAULT_MATH; TF32 override0"},{"validation",validation}}.dump()<<'\n';
 solver_check_ref(cusolverDnDestroyParams(params));solver_check_ref(cusolverDnDestroy(handle));return status||!numerical_pass?1:0;
}
int main(int argc,char**argv){try{if(argc!=5)throw std::runtime_error("usage: cusolver_reference fp32|fp64 m n reps");CU(cudaSetDevice(0));int m=std::stoi(argv[2]),n=std::stoi(argv[3]),reps=std::stoi(argv[4]);if(m<0||n<0||reps<1)throw std::runtime_error("descriptor");return std::string(argv[1])=="fp32"?bench<float>(m,n,reps):bench<double>(m,n,reps);}catch(const std::exception&e){std::cout<<json{{"status","failed_or_unsupported"},{"reason",e.what()},{"measured",false}}.dump()<<'\n';return 3;}}
