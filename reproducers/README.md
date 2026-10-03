# Reproducing the measurements

`presets.json` holds every QR-Ω measurement of the paper: the GPU count, arithmetic, matrix order, number of timed
repetitions, the driver options of the schedule, and the published time and numerical checks. `runner.py` is the
only benchmark entry point.

```sh
python3 reproducers/runner.py --sizes 65536 131072              # all measured modes at these sizes
python3 reproducers/runner.py --gpus 4 --sizes 229376 --modes fp64 fp32
python3 reproducers/runner.py --gpus 2 --sizes 229376            # strong-scaling points
python3 reproducers/runner.py --suite smoke                      # all modes at n = 4,096
```

`--list` prints the measured combinations, `--plan` prints the commands without running them, `--build` selects
the build directory, and `--reps` overrides the number of timed repetitions. For timing, reserve the GPUs and CPU
cores, keep clocks fixed, and run nothing else on the node. The published numbers were measured on NVIDIA DGX H200
nodes (driver 570, CUDA 13.0.2).

## Suites and regression checks

The default `--suite headlines` runs FP64/FP32 at n = 256 and 1,024 (the quoted 9.3× and 1.7× speedups), plus all
modes at n = 16,384, 65,536, and 131,072 on one GPU. On four GPUs it runs all modes at n = 131,072, FP64/FP32 at
229,376, and TF32 at 327,680. `--sizes` selects exact measured combinations. `--suite paper` runs all presets for the
GPU count; the two- and three-GPU presets are the strong-scaling points at n = 229,376.

A case fails if the factorization or its checks fail, if the record does not match the requested size, arithmetic
and GPU count, or if its time or errors are worse than the paper's: time / published time must stay within
`--max-slowdown` (default 1.04) and each error / published error within `--max-error-ratio` (default 1.25). Smoke
cases are not compared with the paper. The two- and three-GPU presets recorded timing only, so their errors are
checked but not compared.

The published 3xTF32 points up to n = 2,048 were measured with a two-launch W kernel; this code computes W at every
size with the register-split kernel that the paper describes. Those cells run 10-20% faster than published, with
errors between 0.84 and 1.27 times the published ones, so `--suite paper` reports n = 256 and 512 as error
regressions of 1.27.

The 4% timing allowance covers machine and run variation. Use `--max-slowdown 1.01` for the paper's 1% tie
threshold, with identical hardware, libraries and clocks. A speedup over a vendor library needs a matching
reference run, or is a comparison with the preserved reference measurement.

Each result directory holds the JSON record and log of every case and a manifest with the commands, times and
paper ratios. Failed cases stay available for inspection, the remaining cases still run, and any failure makes
the runner exit nonzero. Existing result directories are never overwritten.

## Protocol

* **Time.** The driver warms up once and reports the median of the timed repetitions (the upper middle one for an
  even count). The interval starts with the input ready on the GPUs and ends when every GPU has finished;
  generation, allocation and validation are outside it. On several GPUs the input is generated in the
  block-cyclic layout, so no redistribution is timed. Throughput counts `(4/3) n³` operations.
* **Residual.** The largest `‖A(:,J) − (QR)(:,J)‖_F / ‖A(:,J)‖_F` over the column blocks `J` of a reconstruction of
  all n columns.
* **Orthogonality.** `‖Q(QᵀX) − X‖_F / ‖X‖_F` for 16 deterministic random vectors `X`, applying Q from the retained
  factors.
* The time and the checks of a data point come from the same configuration. TF32 and 3xTF32 are compared with the
  references' FP32.

## Reference libraries

`references/` contains adapters that time and validate the QR factorizations of cuSOLVER (`Xgeqrf`), cuSOLVERMp
(`pgeqrf`), MAGMA (`geqrf2_gpu`, `geqrf2_mgpu`), and SLATE (`geqrf`) with the same protocol. The tested versions are
cuSOLVER 12.0.4.66 (CUDA 13.0.2), cuSOLVERMp 0.8.0 with stock NCCL 2.30.4, MAGMA 2.10.0 (ILP64, with MKL
2026.0.1), and SLATE 2025.05.28 (GCC, OpenMP). cuSOLVER is always built; the others are built when their prefix
is given:

```sh
cmake -S . -B build -DQR_OMEGA_BUILD_REFERENCES=ON \
      -DMAGMA_ROOT=/path/to/magma -DMKL_ROOT=/path/to/mkl -DSLATE_ROOT=/path/to/slate \
      -DCUSOLVERMP_ROOT=/path/to/cusolvermp -DREFERENCE_NCCL_ROOT=/path/to/stock-nccl
cmake --build build --parallel 8
```

cuSOLVERMp needs stock NCCL: do not put the low-latency NCCL of QR-Ω ahead of it on `LD_LIBRARY_PATH`. Inside a
Slurm step with fewer tasks than GPUs, add `--oversubscribe` to `mpirun` (the runner does this for QR-Ω). Each
adapter prints one JSON record:

| Adapter | Arguments |
| --- | --- |
| `qr_reference_cusolver` | `fp32\|fp64 m n repetitions` |
| `qr_reference_magma` | `fp32\|fp64 m n gpus repetitions` |
| `qr_reference_slate` (MPI) | `fp32\|fp64 m n block_size repetitions` |
| `qr_reference_cusolvermp` (MPI) | `fp32\|fp64 m n block_size repetitions` |

```sh
CUDA_VISIBLE_DEVICES=0 build/bin/qr_reference_cusolver fp64 16384 16384 3
CUDA_VISIBLE_DEVICES=0 TQR_MAGMA_QR_NB=128 build/bin/qr_reference_magma fp64 16384 16384 1 3
CUDA_VISIBLE_DEVICES=0 TQR_SLATE_INNER_BLOCK=32 TQR_SLATE_LOOKAHEAD=1 \
  mpirun -np 1 build/bin/qr_reference_slate fp64 16384 16384 512 3
CUDA_VISIBLE_DEVICES=0,1,2,3 TQR_MP_GRID_ROWS=4 mpirun -np 4 build/bin/qr_reference_cusolvermp fp64 131072 131072 256 2
```

Every library was tuned for every size and precision; `../plots/data/reference-presets.csv` lists the winning
configuration of each measurement. The tuned settings map to the adapters as follows: cuSOLVER's FP32 emulation
`TQR_CUSOLVER_MATH=bf16x9`; MAGMA's block size `TQR_MAGMA_QR_NB` and host threads `OMP_NUM_THREADS`; SLATE's block
size (positional), `TQR_SLATE_INNER_BLOCK`, `TQR_SLATE_LOOKAHEAD`, `TQR_SLATE_PANEL_THREADS`, and
`TQR_SLATE_GRID_ROWS`; cuSOLVERMp's block size (positional) and `TQR_MP_GRID_ROWS`. `TQR_REFERENCE_GEQRF_ONLY=1`
skips the numerical checks for timing-only runs.
