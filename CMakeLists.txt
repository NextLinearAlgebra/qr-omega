cmake_minimum_required(VERSION 3.25)
project(qr_omega VERSION 0.1.0 LANGUAGES CXX CUDA)

option(QR_OMEGA_BUILD_MULTI_GPU "Build the single-node MPI/NCCL/NVSHMEM driver" OFF)
option(QR_OMEGA_BUILD_REFERENCES "Build the cuSOLVER, MAGMA, SLATE and cuSOLVERMp benchmark adapters" OFF)
set(CUTLASS_ROOT "" CACHE PATH "CUTLASS c900acd source directory")
set(JSON_INCLUDE_DIR "" CACHE PATH "Directory containing nlohmann json.hpp 3.12.0")
set(NCCL_ROOT "" CACHE PATH "Installed low-latency NCCL (commit 5357eff) prefix")
set(NVSHMEM_ROOT "" CACHE PATH "NVSHMEM 3.6.5 installation prefix")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "The published build requires CUDA 13.0 or newer (tested: 13.0.2).")
endif()
find_package(CUDAToolkit 13.0 REQUIRED)
find_package(Threads REQUIRED)
include(FetchContent)
if(NOT CUTLASS_ROOT)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG c900acd605e907f88b4554d59924893dcc19a3c3)
  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
  set(CUTLASS_ROOT "${cutlass_SOURCE_DIR}")
endif()
if(NOT EXISTS "${CUTLASS_ROOT}/include/cute/tensor.hpp")
  message(FATAL_ERROR "CUTLASS_ROOT must contain include/cute/tensor.hpp")
endif()
if(NOT JSON_INCLUDE_DIR)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_GetProperties(nlohmann_json)
  if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
  endif()
  set(JSON_INCLUDE_DIR "${nlohmann_json_SOURCE_DIR}/single_include/nlohmann")
endif()
if(NOT EXISTS "${JSON_INCLUDE_DIR}/json.hpp")
  message(FATAL_ERROR "JSON_INCLUDE_DIR must contain json.hpp")
endif()

function(qr_omega_driver target directory source)
  add_executable(${target} "${directory}/${source}")
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target} PRIVATE "${directory}/include" "${JSON_INCLUDE_DIR}")
  target_include_directories(${target} SYSTEM PRIVATE "${CUTLASS_ROOT}/include")
  target_compile_definitions(${target} PRIVATE
    TQR_CUTLASS_COMMIT="c900acd605e907f88b4554d59924893dcc19a3c3"
    TQR_PANEL_PBLK=4 TQR_PANEL_PXPOSE=1 TQR_PANEL_XSPRE=1
    TQR_PANEL_TDEFER=2 TQR_PANEL_AVP=1)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3;-DNDEBUG;--generate-code=arch=compute_90a,code=[compute_90a,sm_90a];
    --expt-relaxed-constexpr;--extended-lambda;-lineinfo;
    --ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-pthread>)
  target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cuda_driver Threads::Threads)
  install(TARGETS ${target} RUNTIME DESTINATION bin)
endfunction()

qr_omega_driver(qr_omega_single "${CMAKE_CURRENT_SOURCE_DIR}/single-gpu" qr_omega_single.cu)
if(QR_OMEGA_BUILD_MULTI_GPU)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_path(QR_NCCL_INCLUDE nccl_device/ll_buffer.h HINTS "${NCCL_ROOT}/include" REQUIRED)
  find_library(QR_NCCL_LIBRARY nccl HINTS "${NCCL_ROOT}/lib" "${NCCL_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
  find_path(QR_NVSHMEM_INCLUDE nvshmem_host.h HINTS "${NVSHMEM_ROOT}/include" REQUIRED)
  find_library(QR_NVSHMEM_LIBRARY nvshmem_host HINTS "${NVSHMEM_ROOT}/lib" "${NVSHMEM_ROOT}/lib64" REQUIRED)
  qr_omega_driver(qr_omega_multi "${CMAKE_CURRENT_SOURCE_DIR}/multi-gpu" qr_omega_multi.cu)
  target_compile_definitions(qr_omega_multi PRIVATE TQR_MULTI=1)
  target_include_directories(qr_omega_multi SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
  target_link_libraries(qr_omega_multi PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
  # 3xTF32 multi-GPU measurements use promotion every two contraction tiles.
  qr_omega_driver(qr_omega_multi_x3 "${CMAKE_CURRENT_SOURCE_DIR}/multi-gpu" qr_omega_multi.cu)
  target_compile_definitions(qr_omega_multi_x3 PRIVATE TQR_MULTI=1 TQR_X3RS_PROMOTE=2)
  target_include_directories(qr_omega_multi_x3 SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
  target_link_libraries(qr_omega_multi_x3 PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
endif()

if(QR_OMEGA_BUILD_REFERENCES)
  add_subdirectory(reproducers/references)
endif()
