# QR-Ω

Householder QR for NVIDIA Hopper GPUs, retaining the configurations behind the
IPDPS 2027 paper. The code supports FP64, IEEE FP32, TF32, and 3×TF32 on one GPU
and on multiple GPUs within a node.

The main results on H200 are:

| Configuration | Paper result |
| --- | --- |
| One GPU, FP64 / FP32 | Up to 1.7× over cuSOLVER; up to 9.3× over MAGMA and SLATE |
| One GPU, n = 131,072 | 47.0 / 47.5 TFLOP/s in FP64 / FP32; 247.4 / 91.9 in TF32 / 3×TF32 |
| Four GPUs, FP64 / FP32 | 1.06–1.14× over cuSOLVERMp |
| Four GPUs, TF32, n = 327,680 | 777 TFLOP/s |

These are the published measurements. Fresh runs check their numerical results
and compare timing and errors with the matching paper preset.

## Build

Requires Hopper (`sm_90a`), CUDA 13.0+ (paper: 13.0.2), GCC 13, CMake 3.25+, and
Python 3.9+. CMake fetches the pinned CUTLASS and nlohmann/json dependencies;
`CUTLASS_ROOT` and `JSON_INCLUDE_DIR` can select existing copies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
```

For multiple GPUs, install Open MPI, NVSHMEM 3.6.5, and the paper's
[low-latency NCCL](https://github.com/ss16118/low-latency-nccl)
(commit `5357eff325eddf978137de7140195a5568fa8a11`, which supplies
`nccl_device/ll_buffer.h`), then configure:

```sh
cmake -S . -B build -DQR_OMEGA_BUILD_MULTI_GPU=ON \
  -DNCCL_ROOT=/path/to/low-latency-nccl/build \
  -DNVSHMEM_ROOT=/path/to/nvshmem -DCMAKE_PREFIX_PATH=/path/to/mpi
cmake --build build --parallel 8
```

Put that NCCL first on `LD_LIBRARY_PATH` when running QR-Ω.

## Reproduce

Run inside an exclusive GPU allocation, with `CUDA_VISIBLE_DEVICES` selecting the
allocated GPUs. One command handles every precision and GPU count:

```sh
python3 reproducers/runner.py --suite smoke                    # small correctness checks
python3 reproducers/runner.py                                 # one-GPU headline cases
python3 reproducers/runner.py --gpus 4                         # four-GPU headline cases
python3 reproducers/runner.py --sizes 16384 --modes fp64 fp32   # specific paper cases
python3 reproducers/runner.py --gpus 4 --plan                  # inspect commands
```

Headline cases cover panel-dominated and large trailing-update workloads. Results
include every-column reconstruction, orthogonality, raw times, and provenance in
`results/`. A run fails if it is more than 4% slower than the paper or either error
exceeds 1.25× the recorded value. These tolerances are configurable; a passing
smoke test makes no performance claim. `--suite paper` runs the complete measured
range. See [the measurement protocol](reproducers/README.md).

For a size without a measured preset, or another GPU, the planner selects the
schedule from a profile of the GPU:

```sh
cmake --build build --target machine_probe                          # once
python3 -m planner --n 20000 --mode fp64 --run                      # probe, select, factor
python3 reproducers/runner.py --auto --sizes 20000 --modes fp64 tf32
```

The probe measures the GPU in a few seconds, and the planner minimizes a cost
model over the schedules that keep every carried product at c ≥ 2 and fit every
memory level. [The profile](machine/README.md) is independent of QR;
[the planner](planner/README.md) documents the model and its validation.

Regenerate the five paper figures and headline table from the preserved data:

```sh
python3 -m venv .venv
.venv/bin/pip install -r plots/requirements.txt
.venv/bin/python plots/reproduce.py
```

## Read the code

| Location | Purpose |
| --- | --- |
| `benchmarks/qr_omega.cu` | The driver: input generation, timing, residual and orthogonality checks |
| `include/qr_omega/engine.cuh`, `plan.hpp` | The elimination list and its schedule on one or several GPUs |
| `include/qr_omega/update.cuh` | The trailing update and the choice of carrier for every product |
| `include/qr_omega/panel_*.cuh` | Panel factorizations: cooperative and register-resident |
| `include/qr_omega/carrier_*.cuh` | The carried products on CUDA cores and tensor cores |
| `include/qr_omega/transport.cuh` | Communication between the GPUs of a node |
| `machine/` | The machine probe, its profiles, and the cost-model terms read from them |
| `planner/` | Schedule selection from a profile: admissible schedules, cost model, calibration |
| `reproducers/` | The runner, the measured presets, optional reference-library adapters |
| `plots/` | The paper's data and the command that regenerates its figures |
| `tests/` | Preset, runner and planner checks that need no GPU |

[The code map](docs/code-map.md) follows the execution path and lists the driver options. Use `.clang-format` for
CUDA/C++ and the Ruff settings in `pyproject.toml` for Python. Run the host checks with
`python3 -m unittest discover -s tests -v`.

Panel kernels adapt gau.nernst's GPU MODE QR-v2 submission 844219; FP64 carried
updates follow KAMI; tensor-core kernels use CUTLASS/CuTe; inter-GPU gathers use
Shen et al.'s low-latency NCCL. See [third-party notices](third_party/README.md).
