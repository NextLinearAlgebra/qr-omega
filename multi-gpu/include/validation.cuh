#pragma once
// Numerical validation on the GPU: reconstruction residual and orthogonality checks.
#include "engine.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
namespace tqr {
inline void blas_check(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cublas_"+std::to_string(s));}
inline void solver_check(cusolverStatus_t s){if(s!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("cusolver_"+std::to_string(s));}
template<class T> void gram(cublasHandle_t h,int m,const T*q,T*out){T one=1,zero=0;
 if constexpr(sizeof(T)==4)blas_check(cublasSgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,m,m,m,&one,q,m,q,m,&zero,out,m));
 else blas_check(cublasDgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,m,m,m,&one,q,m,q,m,&zero,out,m));
}
template<class T> class VendorReference {
 cusolverDnHandle_t h=nullptr;Buffer<T> tau,work;Buffer<int> info;int m,n,ld;Stream stream;
public:
 VendorReference(int rows,int cols,T* a):tau(std::min(rows,cols)),info(1),m(rows),n(cols),ld(std::max(1,rows)){
  solver_check(cusolverDnCreate(&h));solver_check(cusolverDnSetStream(h,stream));int size=0;
  if(m&&n){if constexpr(sizeof(T)==4)solver_check(cusolverDnSgeqrf_bufferSize(h,m,n,a,ld,&size));else solver_check(cusolverDnDgeqrf_bufferSize(h,m,n,a,ld,&size));}work.alloc(size);
 }
 void factor(T*a){if(!m||!n)return;
  if constexpr(sizeof(T)==4)solver_check(cusolverDnSgeqrf(h,m,n,a,ld,tau.p,work.p,int(work.n),info.p));
  else solver_check(cusolverDnDgeqrf(h,m,n,a,ld,tau.p,work.p,int(work.n),info.p));stream.sync();if(info.download()[0])throw std::runtime_error("cusolver_info");
 }
 ~VendorReference(){if(h)cusolverDnDestroy(h);}
};
template<class T> double relative(const T*a,const T*b,int m,int n,int lda,int ldb,bool identity=false){
 if(!m||!n)return 0;Buffer<T> out(3);relative_error<<<1,256>>>(a,b,m,n,lda,ldb,out.p,identity);CU(cudaDeviceSynchronize());auto v=out.download();
 return v[1]!=0?double(v[0])/double(v[1]):double(v[0])*double(v[2]);
}
template<class T> json validate(Engine<T>&e,Context&ctx,Matrix<T>&a,const Matrix<T>&orig){
 double start=seconds();Matrix<T> rec(a.m,a.n,ctx.size,ctx.rank);if(a.nr&&a.n)CU(cudaMemcpy2D(rec.a.p,rec.ld*sizeof(T),a.a.p,a.ld*sizeof(T),a.nr*sizeof(T),a.n,cudaMemcpyDeviceToDevice));
 if(e.selected_plan().tree=="scalar_inplace"&&a.m&&a.n){mask_R<<<128,128>>>(rec.a.p,rec.m,rec.n,rec.ld);CU(cudaDeviceSynchronize());}
 int status=e.apply_q(rec,false);auto reconstructed=e.gather_full(rec);auto original=e.gather_full(orig);
 Matrix<T> q(a.m,a.m,ctx.size,ctx.rank);if(q.nr&&q.n)identity<<<128,128>>>(q.a.p,q.nr,q.n,q.ld,q.begin);CU(cudaDeviceSynchronize());status=std::max(status,e.apply_q(q,false));auto fullq=e.gather_full(q);
 status=std::max(status,e.apply_q(q,true));auto identityq=e.gather_full(q);
 json result;
 if(ctx.rank==0){double residual=relative(reconstructed.p,original.p,a.m,a.n,std::max(1,a.m),std::max(1,a.m));double inverse=relative(identityq.p,(T*)nullptr,a.m,a.m,std::max(1,a.m),1,true),orth=0;
  if(a.m){Buffer<T> qtq(size_t(a.m)*a.m);cublasHandle_t bh;blas_check(cublasCreate(&bh));blas_check(cublasSetMathMode(bh,CUBLAS_PEDANTIC_MATH));gram(bh,a.m,fullq.p,qtq.p);CU(cudaDeviceSynchronize());orth=relative(qtq.p,(T*)nullptr,a.m,a.m,a.m,1,true)*std::sqrt(double(a.m));blas_check(cublasDestroy(bh));}
  double u=working_unit_roundoff<T>(),depth=1+std::ceil(std::log2(std::max(1,ceildiv(a.m,e.selected_plan().leaf))));
  double tol=32*u*std::max(1,a.m+a.n+e.selected_plan().b)*depth;
  bool pass=status==0&&std::isfinite(residual)&&std::isfinite(orth)&&std::isfinite(inverse)&&residual<=tol&&inverse<=tol&&orth<=tol*std::sqrt(double(std::max(1,a.m)));
  result={{"pass",pass},{"residual_fro_relative",residual},{"orthogonality_fro",orth},{"full_q_qt_inverse_relative",inverse},{"unit_roundoff",u},{"engineering_tolerance",tol},{"tolerance_derivation","32*u*(m+n+b)*(1+ceil(log2(ceil(m/leaf)))); conservative engineering gate, not a proved uniform stability bound"},{"residual_over_u",residual/u},{"full_Q_tested",true},{"device_status",status}};
 }
 result["validation_s"]=seconds()-start;return result;
}
}
