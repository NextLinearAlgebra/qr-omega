# Reproducing the measurements

A cell is one library, GPU count, arithmetic and matrix shape. The measured cells are recorded here:

* `presets.json`: QR-Ω's 183 cells (139 square, 44 tall), each with its driver options, the number
  of timed repetitions, the measured time, residual and orthogonality error, and the protocol and
  hashes of the executable and of the run records the measurement comes from;
* `reference-presets.csv`: the fastest tuned configuration and time of every cell of cuSOLVER,
  cuSOLVERMp, MAGMA and SLATE, and whether that measurement ran the numerical checks (`passed`) or
  was timed only (`timing_only`: the tall shapes and the larger orders on two to four GPUs); all
  their measurements are in `../plots/data/references.csv`.

`runner.py` runs cells again with their recorded configuration and compares each run with its
recorded time. Combinations that were not measured are rejected rather than given another cell's
configuration.

```sh
python3 reproducers/runner.py --list                                  # the measured cells
python3 reproducers/runner.py --sizes 65536 131072                    # QR-Ω, one GPU, every mode
python3 reproducers/runner.py --shapes 1048576x256 --modes fp64 tf32  # a tall shape
python3 reproducers/runner.py --gpus 4 --sizes 229376 --modes fp64 fp32
python3 reproducers/runner.py --gpus 8 --sizes 131072 --plan          # print the eight-GPU commands
python3 reproducers/runner.py --gpus 8 --library MAGMA --sizes 131072 --modes fp32
```

`--library` selects QR-Ω (the default) or a reference, `--gpus` the GPU count (1, 2, 3, 4 or 8),
`--plan` prints the commands without running them, `--build` selects the build directory, and
`--reps` overrides the number of timed repetitions. For timing, reserve the GPUs and the CPU cores
of the node, keep the GPU clocks uncapped, and run nothing else on it. MAGMA and SLATE work on
host cores as well, so their times depend on the cores they get. The measurements were made on
NVIDIA DGX H200 nodes (driver 570, CUDA 13.0.2 through the forward-compatible driver 580.95.05).

## Suites and checks

The default `--suite headlines` selects the cells the paper quotes: on one GPU, FP64 and FP32 at
n = 256 and 1,024 and every mode at n = 16,384, 65,536 and 131,072; on four and eight GPUs, every
mode at n = 131,072, FP64 and FP32 at 229,376 and TF32 at 327,680; on two and three GPUs, the
strong-scaling points at n = 229,376. `--suite tall` selects the tall shapes, `--suite all` every
measured cell of the GPU count, and `--suite smoke` every mode at n = 4,096 with the options of a
measured cell. `--sizes`, `--shapes` and `--modes` select measured cells directly. The references
are compared in FP64 and FP32, so `--modes tf32` selects their FP32 cell.

A QR-Ω run fails if the factorization or its checks fail, if its record does not match the
requested shape, arithmetic and GPU count, if a product is carried outside the 2.5D bound, or if a
square factorization replicates no product (the tuned schedules of some tall shapes carry every
product with c = 1). A reference run fails if the adapter reports a failure or, for a cell measured
with its numerical checks, if they fail; cells measured without them are timed only, and
`--timing-only` times every cell only. Any run
fails if it is more than `--max-slowdown` (default 1.04) times slower than its recorded time; smoke
cases have no recorded time.

Each results directory holds the JSON record and log of every cell and a manifest with the commands,
environments, times and ratios to the recorded times. Failed cells stay available for inspection,
the remaining cells still run, and any failure makes the runner exit nonzero. Existing results
directories are never overwritten.

## Protocol

* **Time.** The driver warms up once and reports the median of the timed repetitions (the upper
  middle one for an even count). The interval starts with the input ready on the GPUs and ends when
  every GPU has finished; generation, allocation and validation are outside it. On several GPUs the
  input is generated in the block-cyclic layout, so no redistribution is timed. Throughput counts
  `(4/3) n³` operations for a square matrix and `2mn² − (2/3) n³` for a tall one.
* **Residual.** The largest `‖A(:,J) − (QR)(:,J)‖_F / ‖A(:,J)‖_F` over the 512-column blocks `J` of a
  reconstruction of all n columns (on several GPUs, over the rows each GPU holds).
* **Orthogonality.** `‖Qᵀ(QX) − X‖_F / ‖X‖_F` for 16 deterministic random vectors `X`, applying Q from
  the retained factors.
* The time and the checks of a data point come from the same configuration. A recorded QR-Ω time is
  the median of the timed repetitions of two runs; a check is the larger of the two runs. TF32 and
  3×TF32 are compared with the references' FP32.

## Reference libraries

`references/` contains adapters that time and validate the QR factorizations of cuSOLVER (`Xgeqrf`),
cuSOLVERMp (`pgeqrf`), MAGMA (`geqrf2_gpu`, `geqrf2_mgpu`) and SLATE (`geqrf`) with the same
protocol. The tested versions are cuSOLVER 12.0.4.66 (CUDA 13.0.2), cuSOLVERMp 0.8.0 with stock
NCCL 2.30.4, MAGMA 2.10.0 (ILP64, with MKL 2026.0.1) and SLATE 2025.05.28 (GCC, OpenMP). cuSOLVER is
always built; the others are built when their prefix is given, for instance through the `paper`
preset:

```sh
export MAGMA_ROOT=/path/to/magma MKLROOT=/path/to/mkl SLATE_ROOT=/path/to/slate \
       CUSOLVERMP_ROOT=/path/to/cusolvermp REFERENCE_NCCL_ROOT=/path/to/stock-nccl
cmake --preset paper && cmake --build --preset paper
```

cuSOLVERMp needs stock NCCL: do not put the low-latency NCCL of QR-Ω ahead of it on
`LD_LIBRARY_PATH`. Inside a Slurm step with fewer tasks than GPUs, add `--oversubscribe` to
`mpirun` (the runner does). Each adapter prints one JSON record:

| Adapter | Arguments |
| --- | --- |
| `qr_reference_cusolver` | `fp32\|fp64 m n repetitions` |
| `qr_reference_magma` | `fp32\|fp64 m n gpus repetitions` |
| `qr_reference_slate` (MPI) | `fp32\|fp64 m n block_size repetitions` |
| `qr_reference_cusolvermp` (MPI) | `fp32\|fp64 m n block_size repetitions` |

The runner passes the tuned configuration of a cell to its adapter through these variables:

| Library | Configuration | Variables |
| --- | --- | --- |
| cuSOLVER | FP32 emulation | `QR_OMEGA_CUSOLVER_MATH=bf16x9` |
| MAGMA | block size, host threads | `QR_OMEGA_MAGMA_NB`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` |
| SLATE | block size (argument), inner blocking, look-ahead, panel threads, grid rows | `QR_OMEGA_SLATE_INNER_BLOCK`, `QR_OMEGA_SLATE_LOOKAHEAD`, `QR_OMEGA_SLATE_PANEL_THREADS`, `QR_OMEGA_SLATE_GRID_ROWS` |
| cuSOLVERMp | block size (argument), grid rows | `QR_OMEGA_CUSOLVERMP_GRID_ROWS` |

On eight GPUs SLATE runs with `OMP_NUM_THREADS=7 MKL_NUM_THREADS=1` per rank, and
`QR_OMEGA_REFERENCE_TIMING_ONLY=1` (`--timing-only`) skips the numerical checks. For example, with
the configurations of `reference-presets.csv`:

```sh
CUDA_VISIBLE_DEVICES=0 build/bin/qr_reference_cusolver fp64 16384 16384 3
CUDA_VISIBLE_DEVICES=0 QR_OMEGA_MAGMA_NB=64 build/bin/qr_reference_magma fp64 16384 16384 1 3
QR_OMEGA_CUSOLVERMP_GRID_ROWS=8 mpirun -np 8 build/bin/qr_reference_cusolvermp fp64 131072 131072 256 3
OMP_NUM_THREADS=7 MKL_NUM_THREADS=1 QR_OMEGA_SLATE_GRID_ROWS=1 QR_OMEGA_SLATE_LOOKAHEAD=1 \
  QR_OMEGA_SLATE_INNER_BLOCK=64 QR_OMEGA_SLATE_PANEL_THREADS=4 \
  mpirun -np 8 reproducers/references/one-gpu-per-rank.sh build/bin/qr_reference_slate fp64 131072 131072 1024 3
```

MAGMA drives all its GPUs from one process; SLATE needs one visible GPU per rank, which
`references/one-gpu-per-rank.sh` provides. Every library was tuned for every precision, and on one
GPU for every size; on eight GPUs it was tuned at n = 131,072 and its two best configurations were
measured at every size.
