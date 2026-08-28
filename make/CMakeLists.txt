cmake_minimum_required(VERSION 3.26)
set(CMAKE_CUDA_ARCHITECTURES 90a)
project(lrqr_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED MATHDX_ROOT)
  set(MATHDX_ROOT "/home/ftome_local/mathdx/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12")
endif()

find_package(mathdx REQUIRED CONFIG PATHS "${MATHDX_ROOT}/lib/cmake/mathdx" NO_DEFAULT_PATH)
find_package(CUDAToolkit REQUIRED)

# Make locally built tools runnable with the CUDA toolkit selected by CMake.
get_target_property(LRQR_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION_RELEASE)
if(NOT LRQR_CUDART_LIBRARY)
  get_target_property(LRQR_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
endif()
get_filename_component(LRQR_CUDA_LIBRARY_DIR "${LRQR_CUDART_LIBRARY}" DIRECTORY)
set(CMAKE_BUILD_RPATH "${LRQR_CUDA_LIBRARY_DIR}")

# B7: NVTX profiling ranges for φ (exposed-chain-fraction) measurement via nsys.
# Default OFF — macros compile to no-ops. ON pushes named ranges onto the nsys
# timeline (chain_step / update_wave / recon / stitch / lane_merge). The nvtx3
# C API is header-only (inline static, lazy dispatch) — no extra link library.
option(LRQR_NVTX "Enable NVTX ranges" OFF)

add_executable(lrqr_bench bench.cu)
target_include_directories(lrqr_bench PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(lrqr_bench PRIVATE mathdx::cusolverdx_fatbin CUDA::cuda_driver
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand pthread)
# W3' carrier: the wgmma kernel (sm_90a-only: wgmma/setmaxnreg) is instantiated in a
# SEPARATE no-LTO TU (wgmma_carrier.cu) because nvcc 13.0's LTO device-link ptxas pass
# gets empty PTXAS_FLAGS and defaults to sm_90, rejecting wgmma. bench.cu keeps LTO (the
# panel/domino kernels REQUIRE LTO inlining to satisfy __launch_bounds__ regcount). The
# wgmma body is suppressed in bench.cu via `extern template` (lrqr.cuh) so it never enters
# the LTO IR; it lives only in wgmma_carrier.o (sm_90a SASS), linked at device-link.
option(LRQR_LTO "Enable CUDA device LTO for bench.cu (panel kernels need it)" ON)
set_target_properties(lrqr_bench PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
if(LRQR_LTO)
  set_target_properties(lrqr_bench PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
  target_link_options(lrqr_bench PRIVATE $<DEVICE_LINK:-dlto>)
endif()
target_compile_options(lrqr_bench PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>"
)

# wgmma carrier kernel — NON-separable final fatbin (like cusolverdx: a pre-linked
# fatbin host-linked into lrqr_bench, NOT a relocatable entering nvlink's LTO pipeline,
# so it does NOT poison bench.cu's LTO inlining). NO LTO (sm_90a ptxas works without LTO;
# nvcc 13.0 LTO ptxas defaults to sm_90). bench.cu never references the kernel directly —
# it calls the host launcher functions (default visibility, declared in lrqr.cuh), the
# cusolverdx pattern. This dodges both the hidden-stub error AND LTO poisoning.
add_library(wgmma_carrier_lib STATIC wgmma_carrier.cu)
target_include_directories(wgmma_carrier_lib PRIVATE "${MATHDX_ROOT}/include")
set_target_properties(wgmma_carrier_lib PROPERTIES
  CUDA_ARCHITECTURES 90a
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(wgmma_carrier_lib PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")
target_link_libraries(wgmma_carrier_lib PRIVATE CUDA::cuda_driver CUDA::cudart)
target_link_libraries(lrqr_bench PRIVATE wgmma_carrier_lib)
# B7: propagate the NVTX flag to bench.cu (and lrqr.cuh via include).
target_compile_definitions(lrqr_bench PRIVATE $<$<BOOL:${LRQR_NVTX}>:LRQR_NVTX=1>)

add_executable(micro micro.cu)
target_include_directories(micro PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(micro PRIVATE CUDA::cudart CUDA::cublas CUDA::curand)
target_compile_options(micro PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>")

add_executable(fp64_probe fp64_probe.cu)
target_include_directories(fp64_probe PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(fp64_probe PRIVATE mathdx::cusolverdx_fatbin CUDA::cuda_driver
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand pthread)
target_compile_options(fp64_probe PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")
target_link_options(fp64_probe PRIVATE $<DEVICE_LINK:-dlto>)

add_executable(fp64_sweep fp64_sweep.cu)
target_include_directories(fp64_sweep PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(fp64_sweep PRIVATE mathdx::cusolverdx_fatbin CUDA::cuda_driver
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand pthread)
target_compile_options(fp64_sweep PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")
target_link_options(fp64_sweep PRIVATE $<DEVICE_LINK:-dlto>)

# W3 sub-unit (b): standalone CUTLASS consumer test (warp-specialized wgmma TF32)
# Source: w3-consumer-impl 44ad3ea (12/12 >=0.85x cuBLAS). Self-contained — uses
# CUTLASS headers directly, no lrqr.cuh dependency. The W3'' STEP(1) standalone
# 12/12 replication harness.
add_executable(w3_consumer_test w3_consumer_test.cu)
target_include_directories(w3_consumer_test PRIVATE
  "${MATHDX_ROOT}/include"
  /home/ftome_local/cutlass/include
  /home/ftome_local/cutlass/tools/util/include)
target_link_libraries(w3_consumer_test PRIVATE CUDA::cuda_driver
  CUDA::cudart CUDA::cublas CUDA::cusolver pthread)
target_compile_options(w3_consumer_test PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")

# W3 step-2 swizzle probe: find the B32 swizzle formula for scalar-stored K-major k8 tile.
add_executable(probe_g3_sw32 probe_g3_sw32.cu)
target_include_directories(probe_g3_sw32 PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(probe_g3_sw32 PRIVATE CUDA::cuda_driver CUDA::cudart)
target_compile_options(probe_g3_sw32 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")
set_target_properties(probe_g3_sw32 PROPERTIES CUDA_ARCHITECTURES 90a)

# ---------------------------------------------------------------------------
# TQR-Omega: the spec-compliant scheduler and its standalone gate.
# Separate target so the existing bench keeps building and stays the reference
# implementation while the new schedule is proved out against it.
add_executable(tqr_omega_test tqr_omega_test.cu)
target_include_directories(tqr_omega_test PRIVATE "${MATHDX_ROOT}/include")
target_link_libraries(tqr_omega_test PRIVATE mathdx::cusolverdx_fatbin CUDA::cuda_driver
  CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::curand pthread wgmma_carrier_lib)
set_target_properties(tqr_omega_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_link_options(tqr_omega_test PRIVATE $<DEVICE_LINK:-dlto>)
target_compile_options(tqr_omega_test PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-lineinfo>")
