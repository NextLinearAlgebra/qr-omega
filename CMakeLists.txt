cmake_minimum_required(VERSION 3.28)

# Standalone Hopper gate. The historical make/CMakeLists.txt also references
# experiments absent from this source package; it is not used by this build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90a)
endif()
project(tqr_omega_gate LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Threads REQUIRED)

add_executable(tqr_gpu_info src/tqr_gpu_info.cu)
target_link_libraries(tqr_gpu_info PRIVATE CUDA::cudart)

add_executable(tqr_omega_test src/tqr_omega_test.cu)
target_compile_features(tqr_omega_test PRIVATE cxx_std_17 cuda_std_17)
# Omega uses LRQR's in-header primitives, not lrqr::Plan or its external
# WGMMA carrier. No MathDx, CUTLASS, or missing carrier stubs are needed here.
target_link_libraries(tqr_omega_test PRIVATE
  CUDA::cuda_driver CUDA::cudart CUDA::cublas CUDA::cusolver Threads::Threads)
set_target_properties(tqr_omega_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  INTERPROCEDURAL_OPTIMIZATION ON)
target_compile_options(tqr_omega_test PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
target_link_options(tqr_omega_test PRIVATE $<DEVICE_LINK:-dlto>)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME hopper_environment COMMAND tqr_gpu_info)
  set_tests_properties(hopper_environment PROPERTIES FIXTURES_SETUP hopper RUN_SERIAL TRUE)
  foreach(shape IN ITEMS 512x512 2048x512 512x2048)
    string(REPLACE "x" ";" dims "${shape}")
    list(GET dims 0 rows)
    list(GET dims 1 cols)
    add_test(NAME omega_fp64_${shape}
      COMMAND tqr_omega_test --m ${rows} --n ${cols} --b 64 --reps 1 -v)
    set_tests_properties(omega_fp64_${shape} PROPERTIES
      FIXTURES_REQUIRED hopper RUN_SERIAL TRUE TIMEOUT 180
      ENVIRONMENT "TQR_TIER=fp64;TQR_PANEL=domino;TQR_PIPE=1")
  endforeach()
endif()

# The baseline is retained by default. Enable the measured panel optimizations
# explicitly; TQR_PANEL_SYNC=cooperative selects CUDA's standard grid barrier.
option(TQR_PANEL_OPTIMIZE "Inline column regions and skip dead reduction slots" OFF)
if(TQR_PANEL_OPTIMIZE)
  target_compile_definitions(tqr_omega_test PRIVATE TQR_DOM_REGION_INLINE=1 TQR_DOM_LIVE_REDUCE=1)
endif()
