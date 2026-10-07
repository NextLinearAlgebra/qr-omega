# Reproducing the measurements

`presets.json` holds the measured cells: for every matrix order, arithmetic and GPU count of the paper,
the driver options, the number of timed repetitions, the measured time, residual and orthogonality
error, and hashes of the executable and of the two run records the measurement comes from. Eight GPUs
are supported, and combinations that were not measured are rejected rather than assigned another
size's configuration.

```sh
python3 reproducers/runner.py --sizes 65536 131072              # all modes at these sizes, one GPU
python3 reproducers/runner.py --gpus 4 --sizes 229376 --modes fp64 fp32
python3 reproducers/runner.py --gpus 2 --sizes 229376            # strong-scaling points
python3 reproducers/runner.py --gpus 8 --sizes 131072 --plan    # print the 8-GPU commands
python3 reproducers/runner.py --suite smoke                      # all modes at n = 4,096
```

`--list` prints the measured combinations, `--plan` prints the commands without running them, `--build` selects
the build directory, and `--reps` overrides the number of timed repetitions. For timing, reserve the GPUs and CPU
cores, keep clocks fixed, and run nothing else on the node. The measurements were made on NVIDIA DGX H200 nodes
(driver 570, CUDA 13.0.2).

## Suites and checks

The default `--suite headlines` selects FP64/FP32 at n = 256 and 1,024, plus all modes at n = 16,384, 65,536
and 131,072 on one GPU; on four and eight GPUs it selects all modes at n = 131,072, FP64/FP32 at 229,376 and
TF32 at 327,680. `--sizes` selects exact measured combinations. `--suite paper` runs every measured cell of the
GPU count; the two- and three-GPU cells are the strong-scaling points at n = 229,376.

A case fails if the factorization or its checks fail, if the record does not match the requested size,
arithmetic and GPU count, if a product is carried outside the 2.5D bound or nothing is replicated, or if
it is more than `--max-slowdown` (default 1.04) times slower than its recorded measurement. Smoke cases
have no recorded measurement. A speedup over a vendor library needs a matching reference run, or is a
comparison with the reference measurements in `../plots/data/references.csv`.

Each result directory holds the JSON record and log of every case and a manifest with the commands, times and
reproduction ratios. Failed cases stay available for inspection, the remaining cases still run, and any failure makes
the runner exit nonzero. Existing result directories are never overwritten.

## Protocol

* **Time.** The driver warms up once and reports the median of the timed repetitions (the upper middle one for an
  even count). The interval starts with the input ready on the GPUs and ends when every GPU has finished;
  generation, allocation and validation are outside it. On several GPUs the input is generated in the
  block-cyclic layout, so no redistribution is timed. Throughput counts `(4/3) n³` operations.
* **Residual.** The largest `‖A(:,J) − (QR)(:,J)‖_F / ‖A(:,J)‖_F` over the 512-column blocks `J` of a
  reconstruction of all n columns (on several GPUs, over the rows each GPU holds).
* **Orthogonality.** `‖Qᵀ(QX) − X‖_F / ‖X‖_F` for 16 deterministic random vectors `X`, applying Q from the retained
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

On eight GPUs, with the tuned settings of `reference-presets.csv` (MAGMA drives all its GPUs from one process;
SLATE needs one visible GPU per rank):

```sh
TQR_MP_GRID_ROWS=8 mpirun -np 8 build/bin/qr_reference_cusolvermp fp64 131072 131072 256 1
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TQR_MAGMA_QR_NB=64 build/bin/qr_reference_magma fp64 131072 131072 8 1
OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 TQR_SLATE_GRID_ROWS=1 TQR_SLATE_LOOKAHEAD=1 TQR_SLATE_INNER_BLOCK=64 \
  TQR_SLATE_PANEL_THREADS=4 mpirun -np 8 reproducers/references/one-gpu-per-rank.sh \
  build/bin/qr_reference_slate fp64 131072 131072 1024 1
```

Every library was tuned for every size and precision (on eight GPUs: at n = 131,072, then its two best
configurations at every size); `../plots/data/reference-presets.csv` lists the winning configuration of each
measurement. The tuned settings map to the adapters as follows: cuSOLVER's FP32 emulation
`TQR_CUSOLVER_MATH=bf16x9`; MAGMA's block size `TQR_MAGMA_QR_NB` and host threads `OMP_NUM_THREADS`; SLATE's block
size (positional), `TQR_SLATE_INNER_BLOCK`, `TQR_SLATE_LOOKAHEAD`, `TQR_SLATE_PANEL_THREADS`, and
`TQR_SLATE_GRID_ROWS`; cuSOLVERMp's block size (positional) and `TQR_MP_GRID_ROWS`. `TQR_REFERENCE_GEQRF_ONLY=1`
skips the numerical checks for timing-only runs.
