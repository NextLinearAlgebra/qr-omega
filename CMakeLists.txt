cmake_minimum_required(VERSION 3.25)
project(qr_omega VERSION 0.1.0 LANGUAGES CXX CUDA)

option(QR_OMEGA_BUILD_MULTI_GPU "Build the single-node MPI/NCCL/NVSHMEM driver" OFF)
option(QR_OMEGA_BUILD_REFERENCES "Build the cuSOLVER, MAGMA, SLATE and cuSOLVERMp benchmark adapters" OFF)
option(QR_OMEGA_BUILD_GPU_TESTS "Build and register the Hopper panel correctness tests" OFF)
set(CUTLASS_ROOT "" CACHE PATH "CUTLASS c900acd source directory (fetched when empty)")
set(JSON_INCLUDE_DIR "" CACHE PATH "Directory containing nlohmann json.hpp 3.12.0 (fetched when empty)")
set(NCCL_ROOT "" CACHE PATH "Installed low-latency NCCL (commit 5357eff) prefix")
set(NVSHMEM_ROOT "" CACHE PATH "NVSHMEM 3.6.5 installation prefix")

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "The published build requires CUDA 13.0 or newer (tested: 13.0.2).")
endif()
find_package(CUDAToolkit 13.0 REQUIRED)
find_package(Threads REQUIRED)

# Header-only dependencies at their pinned versions. SOURCE_SUBDIR names a directory without a
# CMakeLists.txt, so FetchContent downloads them without adding their own builds to this one.
include(FetchContent)
if(NOT CUTLASS_ROOT)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG c900acd605e907f88b4554d59924893dcc19a3c3
    SOURCE_SUBDIR include)
  FetchContent_MakeAvailable(cutlass)
  set(CUTLASS_ROOT "${cutlass_SOURCE_DIR}")
endif()
if(NOT JSON_INCLUDE_DIR)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR single_include)
  FetchContent_MakeAvailable(nlohmann_json)
  set(JSON_INCLUDE_DIR "${nlohmann_json_SOURCE_DIR}/single_include/nlohmann")
endif()
if(NOT EXISTS "${CUTLASS_ROOT}/include/cute/tensor.hpp")
  message(FATAL_ERROR "CUTLASS_ROOT must contain include/cute/tensor.hpp")
endif()
if(NOT EXISTS "${JSON_INCLUDE_DIR}/json.hpp")
  message(FATAL_ERROR "JSON_INCLUDE_DIR must contain json.hpp")
endif()

# An executable of QR-Omega for Hopper (sm_90a), compiled with IEEE division and square root and
# without flushing denormals to zero.
function(qr_omega_cuda_executable target source)
  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target} PRIVATE
    "${PROJECT_SOURCE_DIR}/include/qr_omega"
    "${JSON_INCLUDE_DIR}")
  target_include_directories(${target} SYSTEM PRIVATE "${CUTLASS_ROOT}/include")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3;-DNDEBUG;--generate-code=arch=compute_90a,code=[compute_90a,sm_90a];
    --expt-relaxed-constexpr;--extended-lambda;-lineinfo;
    --ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-pthread>)
  target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cuda_driver Threads::Threads)
endfunction()

# An executable that runs on the GPUs of a node, one MPI rank per GPU, with the low-latency NCCL
# and NVSHMEM between them.
function(qr_omega_multi_gpu_executable target source)
  qr_omega_cuda_executable(${target} ${source})
  target_compile_definitions(${target} PRIVATE QR_OMEGA_MULTI_GPU=1)
  target_include_directories(${target} SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
  target_link_libraries(${target} PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
endfunction()

qr_omega_cuda_executable(qr_omega_single benchmarks/qr_omega.cu)
install(TARGETS qr_omega_single RUNTIME DESTINATION bin)

if(QR_OMEGA_BUILD_MULTI_GPU)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_path(QR_NCCL_INCLUDE nccl_device/ll_buffer.h HINTS "${NCCL_ROOT}/include" REQUIRED)
  find_library(QR_NCCL_LIBRARY nccl HINTS "${NCCL_ROOT}/lib" "${NCCL_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
  find_path(QR_NVSHMEM_INCLUDE nvshmem_host.h HINTS "${NVSHMEM_ROOT}/include" REQUIRED)
  find_library(QR_NVSHMEM_LIBRARY nvshmem_host HINTS "${NVSHMEM_ROOT}/lib" "${NVSHMEM_ROOT}/lib64" REQUIRED)
  qr_omega_multi_gpu_executable(qr_omega_multi benchmarks/qr_omega.cu)
  qr_omega_multi_gpu_executable(qr_omega_transport_bench benchmarks/transport.cu)
  if(QR_OMEGA_BUILD_GPU_TESTS)
    qr_omega_multi_gpu_executable(qr_omega_transport_tests tests/transport_gpu.cu)
  endif()
  install(TARGETS qr_omega_multi qr_omega_transport_bench RUNTIME DESTINATION bin)
endif()

if(QR_OMEGA_BUILD_REFERENCES)
  add_subdirectory(reproducers/references)
endif()

include(CTest)
if(NOT BUILD_TESTING)
  return()
endif()

find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
add_test(NAME reproduction_contract
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests -v)
set_tests_properties(reproduction_contract PROPERTIES
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" LABELS "cpu")

if(QR_OMEGA_BUILD_GPU_TESTS)
  # A GPU test with its own executable.
  function(qr_omega_kernel_test name target source)
    qr_omega_cuda_executable(${target} ${source})
    add_test(NAME ${name} COMMAND ${target})
    set_tests_properties(${name} PROPERTIES LABELS "gpu" TIMEOUT 180)
  endfunction()
  # A GPU test that runs the one-GPU driver from a script in tests/; the arguments after the
  # timeout go to the script.
  function(qr_omega_driver_test name script timeout)
    add_test(NAME ${name}
      COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/${script}"
              "$<TARGET_FILE:qr_omega_single>" ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS "gpu" TIMEOUT ${timeout})
  endfunction()

  qr_omega_kernel_test(domain_panel qr_omega_domain_tests tests/domain_panel.cu)
  qr_omega_kernel_test(register_panel qr_omega_register_tests tests/register_panel.cu)
  qr_omega_kernel_test(tree_update qr_omega_tree_update_tests tests/tree_update.cu)
  qr_omega_kernel_test(scaled_norm qr_omega_norm_tests tests/norm_gpu.cu)
  qr_omega_driver_test(hierarchy_updates hierarchy_gpu.py 600)
  qr_omega_driver_test(hierarchy_resident hierarchy_gpu.py 180 --tree-update resident)
  qr_omega_driver_test(hierarchy_reconstructed hierarchy_gpu.py 600 --panel-format wy)
  qr_omega_driver_test(hierarchy_aggregated hierarchy_gpu.py 600 --panel-format wy --aggregate 4)
  qr_omega_driver_test(hierarchy_z_replicated hierarchy_gpu.py 600
    --panel-format wy --aggregate 4 --zc 8)
  qr_omega_driver_test(hierarchy_tall_reconstructed hierarchy_gpu.py 600
    --panel-format wy --aggregate 2 --tree-update resident --tall)
  qr_omega_driver_test(conditioning condition_gpu.py 900)
  qr_omega_driver_test(tree_tail tail_gpu.py 600)
endif()
