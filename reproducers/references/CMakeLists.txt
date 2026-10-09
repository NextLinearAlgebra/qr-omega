# Benchmark adapters for the reference libraries (cuSOLVER, MAGMA, SLATE, cuSOLVERMp). Each adapter times the
# library's own QR factorization and validates it with the same checks as QR-Omega. The libraries are installed
# separately; an adapter is built when its prefix is given.
set(MAGMA_ROOT "" CACHE PATH "MAGMA 2.10.0 (ILP64) prefix")
set(SLATE_ROOT "" CACHE PATH "SLATE 2025.05.28 (GCC, OpenMP) prefix")
set(MKL_ROOT "" CACHE PATH "Intel MKL (ILP64) prefix used by MAGMA")
set(CUSOLVERMP_ROOT "" CACHE PATH "cuSOLVERMp 0.8.0 prefix")
set(REFERENCE_NCCL_ROOT "" CACHE PATH "Stock NCCL prefix for cuSOLVERMp")

function(qr_reference target source)
  add_executable(${target} "${source}" "${CMAKE_CURRENT_SOURCE_DIR}/reference_gpu.cu")
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES 90 RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/include/qr_omega"
    "${JSON_INCLUDE_DIR}")
  target_compile_definitions(${target} PRIVATE QR_OMEGA_REFERENCE_VALIDATE=1)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--ftz=false;--prec-div=true;--prec-sqrt=true>)
  target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas)
  install(TARGETS ${target} RUNTIME DESTINATION bin)
endfunction()

qr_reference(qr_reference_cusolver cusolver_reference.cpp)

if(MAGMA_ROOT)
  find_path(QR_MAGMA_INCLUDE magma_v2.h HINTS "${MAGMA_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
  find_library(QR_MAGMA_LIBRARY magma HINTS "${MAGMA_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
  qr_reference(qr_reference_magma magma_reference.cu)
  target_sources(qr_reference_magma PRIVATE magma_nb_tuning.cpp)
  target_compile_definitions(qr_reference_magma PRIVATE MAGMA_ILP64=1 QR_OMEGA_MAGMA_TUNING_HOOK=1)
  target_include_directories(qr_reference_magma PRIVATE "${QR_MAGMA_INCLUDE}")
  target_link_libraries(qr_reference_magma PRIVATE "${QR_MAGMA_LIBRARY}" ${CMAKE_DL_LIBS})
  if(MKL_ROOT)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(qr_reference_magma PRIVATE OpenMP::OpenMP_CXX)
    foreach(component mkl_gf_ilp64 mkl_gnu_thread mkl_core)
      find_library(QR_${component} NAMES ${component} lib${component}.so.3
        HINTS "${MKL_ROOT}/lib" "${MKL_ROOT}/lib/intel64" REQUIRED)
      target_link_libraries(qr_reference_magma PRIVATE "${QR_${component}}")
    endforeach()
  endif()
endif()

if(SLATE_ROOT OR CUSOLVERMP_ROOT)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(SLATE_ROOT)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  find_path(QR_SLATE_INCLUDE slate/slate.hh HINTS "${SLATE_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
  qr_reference(qr_reference_slate slate_reference.cpp)
  target_include_directories(qr_reference_slate PRIVATE "${QR_SLATE_INCLUDE}")
  target_link_libraries(qr_reference_slate PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
  foreach(component slate blaspp lapackpp)
    find_library(QR_${component} ${component} HINTS "${SLATE_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
    target_link_libraries(qr_reference_slate PRIVATE "${QR_${component}}")
  endforeach()
endif()

if(CUSOLVERMP_ROOT)
  find_path(QR_MP_INCLUDE cusolverMp.h HINTS "${CUSOLVERMP_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
  find_library(QR_MP_LIBRARY cusolverMp HINTS "${CUSOLVERMP_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
  find_path(QR_REF_NCCL_INCLUDE nccl.h HINTS "${REFERENCE_NCCL_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
  find_library(QR_REF_NCCL_LIBRARY nccl HINTS "${REFERENCE_NCCL_ROOT}/lib" "${REFERENCE_NCCL_ROOT}/lib64"
    REQUIRED NO_DEFAULT_PATH)
  qr_reference(qr_reference_cusolvermp cusolvermp_reference.cpp)
  target_include_directories(qr_reference_cusolvermp PRIVATE "${QR_MP_INCLUDE}" "${QR_REF_NCCL_INCLUDE}")
  target_link_libraries(qr_reference_cusolvermp PRIVATE MPI::MPI_CXX "${QR_MP_LIBRARY}" "${QR_REF_NCCL_LIBRARY}"
    ${CMAKE_DL_LIBS})
endif()
