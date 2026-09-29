# QR-Ω

QR-Ω is a hierarchical 2.5D Householder QR factorization for NVIDIA Hopper GPUs, on one GPU and on the GPUs of
one node. It keeps the elimination list of hierarchical tiled QR and gives every matrix product of a trailing
update, `W = VᵀX`, `Z = TᵀW` and `D = VZ`, its own *carrier* at each level of the GPU hierarchy: how the product is
split among peers, which operands are replicated, how partial results are combined, and how far updates are
pipelined. Replication is bounded by the memory of each level, so partial sums are combined in registers, shared
memory, HBM, or across GPUs, wherever they fit. The factorization returns R and the Householder factors (V, T) for
applying Q or Qᵀ, in FP64, FP32 (IEEE), TF32, and 3xTF32 arithmetic.

On one H200, n = 131,072:

| Arithmetic | Time (s) | TFLOP/s | Residual | Orthogonality |
| --- | ---: | ---: | ---: | ---: |
| FP64 | 63.9 | 47.0 | 1.1e-14 | 2.6e-15 |
| FP32 | 63.2 | 47.5 | 5.7e-6 | 1.1e-6 |
| TF32 | 12.1 | 247.4 | 1.5e-3 | 1.5e-3 |
| 3xTF32 | 32.7 | 91.9 | 6.8e-6 | 6.1e-6 |

On four H200 GPUs of one node, FP64 factors n = 131,072 in 19.1 s (157 TFLOP/s) and TF32 reaches 777 TFLOP/s at
n = 327,680. The paper compares these results with tuned cuSOLVER, cuSOLVERMp, MAGMA, and SLATE.

## Layout

| Directory | Contents |
| --- | --- |
| `single-gpu/` | Single-GPU engine and driver `qr_omega_single` |
| `multi-gpu/` | Multi-GPU engine (MPI, NCCL, NVSHMEM) and drivers `qr_omega_multi`, `qr_omega_multi_x3` |
| `reproducers/` | Measured configurations of the paper, the runner that executes and validates them, and the reference-library adapters |
| `plots/` | Paper data and the scripts that regenerate its figures |
| `third_party/` | Notices for code adapted from other projects |

## Build

Requirements: a Hopper GPU (`sm_90a`), CUDA 13.0 or newer (tested with 13.0.2), GCC 13, CMake 3.25 or newer, and
Python 3.9 or newer. CMake fetches CUTLASS (commit `c900acd`) and nlohmann/json 3.12.0 unless `CUTLASS_ROOT` and
`JSON_INCLUDE_DIR` point to local copies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8            # build/bin/qr_omega_single
```

The multi-GPU drivers need Open MPI, NVSHMEM 3.6.5, and the low-latency NCCL of Shen et al., which provides
`nccl_device/ll_buffer.h`:

```sh
git clone https://github.com/ss16118/low-latency-nccl.git deps/nccl
git -C deps/nccl checkout 5357eff325eddf978137de7140195a5568fa8a11
make -C deps/nccl -j src.build CUDA_HOME=/path/to/cuda NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
cmake -S . -B build -DQR_OMEGA_BUILD_MULTI_GPU=ON -DNCCL_ROOT=$PWD/deps/nccl/build \
      -DNVSHMEM_ROOT=/path/to/nvshmem -DCMAKE_PREFIX_PATH=/path/to/mpi
cmake --build build --parallel 8            # adds qr_omega_multi and qr_omega_multi_x3
```

At run time, put this NCCL first on `LD_LIBRARY_PATH`.

## Use

A driver generates a matrix, factors it with the configuration given on its command line, and writes a JSON
record with the time (median of the timed repetitions after a warm-up), the configuration, and the numerical
checks. The measured configurations of the paper are stored as presets; the runner selects them by size and
arithmetic, launches the driver, and verifies the result:

```sh
python3 reproducers/single_gpu.py --sizes 16384 --modes fp64 fp32     # one GPU
python3 reproducers/multi_gpu.py --gpus 4 --sizes 131072 --modes fp64  # four GPUs, one MPI rank each
python3 reproducers/single_gpu.py --list                               # measured sizes and modes
python3 reproducers/single_gpu.py --sizes 131072 --plan                # print the exact command lines
```

Run inside your GPU allocation, with `CUDA_VISIBLE_DEVICES` selecting the GPUs. `--smoke` validates all four
arithmetic modes at n = 4,096 in a few seconds. Results go to a new directory under `results/`, with a manifest of
commands, environment, CPU affinity, and hashes.

The drivers can also be called directly (the multi-GPU drivers under `mpirun`, one rank per GPU). The main options:

| Option | Meaning |
| --- | --- |
| `--m`, `--n` | Matrix size |
| `--precision fp64\|fp32`, `--fp32-math ieee\|tf32\|x3` | Storage precision and arithmetic of the products |
| `--b`, `--leaf` | Panel width and the height of the row domain reduced by one GE |
| `--wc`, `--zc`, `--dc` | Contraction replication `c` of the carriers of W, Z, and D (at least 2) |
| `--strip`, `--d` | Width of the trailing column strips and the number in flight (pipeline depth) |
| `--aggregate`, `--lookahead` | Panels composed into one far update, and look-ahead of the next panel (0 or 1) |
| `--panel-groups`, `--panel-width`, `--panel-window` | Thread blocks per panel, minipanel width, and window width of the panel factorization |
| `--row-block`, `--native-cyclic` | Multi-GPU: rows per block of the block-cyclic distribution, generated in place |
| `--reps`, `--output`, `--full-q` | Timed repetitions, JSON output, and an explicit check of Q |

The engine itself is header-only (`single-gpu/include`, `multi-gpu/include`); the drivers show how to build a plan
and run it.

## Reproduce the paper

The figures and quoted numbers are regenerated from the measured data without a GPU:

```sh
python3 -m venv .venv && .venv/bin/pip install -r plots/requirements.txt
.venv/bin/python plots/reproduce.py
```

Fresh measurements use the runner as above; `reproducers/README.md` describes the measurement protocol and how to
build and run the cuSOLVER, cuSOLVERMp, MAGMA, and SLATE baselines with their tuned configurations.

## Credits

The register-resident panel kernels are adapted from gau.nernst's submission 844219 to the GPU MODE QR-v2
leaderboard, with LAPACK reflector arithmetic. The FP64 carried update follows the communication-avoiding GEMM
design of KAMI (Wang et al., SC'25). The tensor-core kernels are built with CUTLASS/CuTe; two headers are modified
copies of CUTLASS files (see `third_party/`). Inter-GPU gathers use the `LLBuffer` primitives of the low-latency
NCCL of Shen et al.
