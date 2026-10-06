cmake_minimum_required(VERSION 3.25)
project(qr_omega VERSION 0.1.0 LANGUAGES CXX CUDA)

option(QR_OMEGA_BUILD_MULTI_GPU "Build the single-node MPI/NCCL/NVSHMEM driver" OFF)
option(QR_OMEGA_BUILD_REFERENCES "Build the cuSOLVER, MAGMA, SLATE and cuSOLVERMp benchmark adapters" OFF)
option(QR_OMEGA_BUILD_GPU_TESTS "Build and register the Hopper panel correctness tests" OFF)
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

function(qr_omega_cuda_executable target source)
  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_ARCHITECTURES OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/qr_omega"
    "${JSON_INCLUDE_DIR}")
  target_include_directories(${target} SYSTEM PRIVATE "${CUTLASS_ROOT}/include")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3;-DNDEBUG;--generate-code=arch=compute_90a,code=[compute_90a,sm_90a];
    --expt-relaxed-constexpr;--extended-lambda;-lineinfo;
    --ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-pthread>)
  target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cuda_driver Threads::Threads)
endfunction()

qr_omega_cuda_executable(qr_omega_single benchmarks/qr_omega.cu)
install(TARGETS qr_omega_single RUNTIME DESTINATION bin)
if(QR_OMEGA_BUILD_MULTI_GPU)
  find_package(MPI REQUIRED COMPONENTS CXX)
  find_path(QR_NCCL_INCLUDE nccl_device/ll_buffer.h HINTS "${NCCL_ROOT}/include" REQUIRED)
  find_library(QR_NCCL_LIBRARY nccl HINTS "${NCCL_ROOT}/lib" "${NCCL_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
  find_path(QR_NVSHMEM_INCLUDE nvshmem_host.h HINTS "${NVSHMEM_ROOT}/include" REQUIRED)
  find_library(QR_NVSHMEM_LIBRARY nvshmem_host HINTS "${NVSHMEM_ROOT}/lib" "${NVSHMEM_ROOT}/lib64" REQUIRED)
  qr_omega_cuda_executable(qr_omega_multi benchmarks/qr_omega.cu)
  target_compile_definitions(qr_omega_multi PRIVATE TQR_MULTI=1)
  target_include_directories(qr_omega_multi SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
  target_link_libraries(qr_omega_multi PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
  qr_omega_cuda_executable(qr_omega_transport_bench benchmarks/transport.cu)
  target_compile_definitions(qr_omega_transport_bench PRIVATE TQR_MULTI=1)
  target_include_directories(qr_omega_transport_bench SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
  target_link_libraries(qr_omega_transport_bench PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
  if(QR_OMEGA_BUILD_GPU_TESTS)
    qr_omega_cuda_executable(qr_omega_transport_tests tests/transport_gpu.cu)
    target_compile_definitions(qr_omega_transport_tests PRIVATE TQR_MULTI=1)
    target_include_directories(qr_omega_transport_tests SYSTEM PRIVATE "${QR_NCCL_INCLUDE}" "${QR_NVSHMEM_INCLUDE}")
    target_link_libraries(qr_omega_transport_tests PRIVATE MPI::MPI_CXX "${QR_NCCL_LIBRARY}" "${QR_NVSHMEM_LIBRARY}")
  endif()
  install(TARGETS qr_omega_multi qr_omega_transport_bench RUNTIME DESTINATION bin)
endif()

if(QR_OMEGA_BUILD_REFERENCES)
  add_subdirectory(reproducers/references)
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
  add_test(NAME reproduction_contract
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests -v)
  set_tests_properties(reproduction_contract PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" LABELS "cpu")
  if(QR_OMEGA_BUILD_GPU_TESTS)
    qr_omega_cuda_executable(qr_omega_domain_tests tests/domain_panel.cu)
    add_test(NAME domain_panel COMMAND qr_omega_domain_tests)
    set_tests_properties(domain_panel PROPERTIES LABELS "gpu" TIMEOUT 180)
    qr_omega_cuda_executable(qr_omega_register_tests tests/register_panel.cu)
    add_test(NAME register_panel COMMAND qr_omega_register_tests)
    set_tests_properties(register_panel PROPERTIES LABELS "gpu" TIMEOUT 180)
    qr_omega_cuda_executable(qr_omega_tree_update_tests tests/tree_update.cu)
    add_test(NAME tree_update COMMAND qr_omega_tree_update_tests)
    set_tests_properties(tree_update PROPERTIES LABELS "gpu" TIMEOUT 180)
    qr_omega_cuda_executable(qr_omega_norm_tests tests/norm_gpu.cu)
    add_test(NAME scaled_norm COMMAND qr_omega_norm_tests)
    set_tests_properties(scaled_norm PROPERTIES LABELS "gpu" TIMEOUT 180)
    add_test(NAME hierarchy_updates
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>")
    set_tests_properties(hierarchy_updates PROPERTIES LABELS "gpu" TIMEOUT 600)
    add_test(NAME hierarchy_resident
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>" --tree-update resident)
    set_tests_properties(hierarchy_resident PROPERTIES LABELS "gpu" TIMEOUT 180)
    add_test(NAME hierarchy_reconstructed
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>" --panel-format wy)
    set_tests_properties(hierarchy_reconstructed PROPERTIES LABELS "gpu" TIMEOUT 600)
    add_test(NAME hierarchy_aggregated
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>" --panel-format wy --aggregate 4)
    set_tests_properties(hierarchy_aggregated PROPERTIES LABELS "gpu" TIMEOUT 600)
    add_test(NAME hierarchy_z_replicated
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>" --panel-format wy --aggregate 4 --zc 8)
    set_tests_properties(hierarchy_z_replicated PROPERTIES LABELS "gpu" TIMEOUT 600)
    add_test(NAME hierarchy_tall_reconstructed
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/hierarchy_gpu.py"
              "$<TARGET_FILE:qr_omega_single>" --panel-format wy --aggregate 2 --tree-update resident --tall)
    set_tests_properties(hierarchy_tall_reconstructed PROPERTIES LABELS "gpu" TIMEOUT 600)
    add_test(NAME conditioning
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/condition_gpu.py"
              "$<TARGET_FILE:qr_omega_single>")
    set_tests_properties(conditioning PROPERTIES LABELS "gpu" TIMEOUT 900)
    add_test(NAME tree_tail
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/tail_gpu.py"
              "$<TARGET_FILE:qr_omega_single>")
    set_tests_properties(tree_tail PROPERTIES LABELS "gpu" TIMEOUT 600)
  endif()
endif()
