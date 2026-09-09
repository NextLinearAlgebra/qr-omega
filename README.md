# QR-Omega: standalone Hopper tests

This entry point builds `tqr::Omega` and its correctness/performance driver from
the sources in this repository. It needs CMake 3.28+, a CUDA toolkit with Hopper
support, and a Hopper GPU. CUDA 13.0.88 was used for the recorded H100 tests.
The historical `make/CMakeLists.txt` references additional experiments and
dependencies that are not included here; use the root CMake project below.

## Build

```sh
cmake -S . -B build-baseline \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DTQR_PANEL_OPTIMIZE=OFF
cmake --build build-baseline -j 4

cmake -S . -B build-panel \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DTQR_PANEL_OPTIMIZE=ON
cmake --build build-panel -j 4
```

`TQR_PANEL_OPTIMIZE=ON` enables live-slot reduction and per-column inlining.
`TQR_PANEL_SYNC=cooperative` additionally enables cooperative grid synchronization
for non-cluster domino panels. **Both opt-ins are needed for the combined change.**
Defaults remain OFF and `legacy`, respectively. The existing cluster path keeps
priority when selected and successfully launched. No algorithm parameters or
precision modes are changed by these switches.

## Correctness

Select a GPU with `CUDA_VISIBLE_DEVICES`. Run the small smoke tests, then the
existing full suites (19 FP64 + 11 FP32 + 11 TF32 cases):

```sh
TQR_PANEL_SYNC=cooperative ctest --test-dir build-panel --output-on-failure

env TQR_PANEL=domino TQR_PIPE=1 TQR_PANEL_SYNC=cooperative TQR_DETERM=1 \
  TQR_TIER=fp64 build-panel/tqr_omega_test --b 64 --reps 1 -v
env TQR_PANEL=domino TQR_PIPE=1 TQR_PANEL_SYNC=cooperative TQR_DETERM=1 \
  TQR_TIER=fp32 build-panel/tqr_omega_test --b 32 --reps 1 -v
env TQR_PANEL=domino TQR_PIPE=1 TQR_PANEL_SYNC=cooperative TQR_DETERM=1 \
  TQR_TIER=tf32 build-panel/tqr_omega_test --b 32 --reps 1 -v
```

The original correctness CLI uses `--b 32` to select the float suite, which
actually instantiates `Omega<float,128>` and runs all 11 shapes. Check both the
suite result and every `[A6 determinism]` line. Repeat with `build-baseline` and
`TQR_PANEL_SYNC=legacy` for a control. Do not set `TQR_TEST_PANEL_PROBES` when
testing normal calibration and the full shape suite.

## Performance

The explicit performance entry selects the requested precision and shape,
performs two warmups, and reports median CUDA-event timing plus every sample.
Allocation, calibration, input copies and numerical checks are outside QR timing.

```sh
env TQR_PANEL=domino TQR_PIPE=1 TQR_PANEL_SYNC=legacy \
  build-baseline/tqr_omega_test --perf-tier tf32 --m 16384 --n 16384 --b 128 --reps 7
env TQR_PANEL=domino TQR_PIPE=1 TQR_PANEL_SYNC=cooperative \
  build-panel/tqr_omega_test --perf-tier tf32 --m 16384 --n 16384 --b 128 --reps 7
```

Repeat in A-B-B-A order; use `--perf-tier fp64 --b 64` or
`--perf-tier fp32 --b 128` for the other modes. Compare the complete printed plans
as well as numerical results. For a controlled A/B test, copy the six measured
values from a baseline `PERF_PANEL_PROBES values=...` line into
`TQR_TEST_PANEL_PROBES` for **both** arms of that same precision and shape. Do not
reuse a table across GPUs or shapes. Omit this fixture for normal calibration.

TF32 is compared against FP32 cuSOLVER, not an equal-precision reference. Its
sampled performance check uses TF32 epsilon, matching the existing full TF32
gate; FP64, FP32 and 3xTF32 sampled normalization is unchanged. Performance checks
are sampled checks and do not replace the full reconstruction/orthogonality suite.

See [panel optimization notes](docs/panel-optimization.md) for the three changes,
historical H100 results, validation scope and outstanding integration issues.
