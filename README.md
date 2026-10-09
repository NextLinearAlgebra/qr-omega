# QR-Ω

Householder QR for NVIDIA Hopper GPUs, on one GPU and on the GPUs of a node, in FP64, FP32, TF32
and 3×TF32. QR-Ω combines hierarchical Householder QR (HQR) with 2.5D scheduling:

* **Hierarchical panels.** Each GPU factors its rows of a panel as an HQR elimination tree: GEQRT
  on domains, TSQRT chains inside a domain and TTQRT merges between domains, every node held in
  registers. Across GPUs, the GPUs of the panel's grid column merge their triangles.
* **Stable compact WY.** The thin Q of the tree is turned into compact-WY factors by Householder
  reconstruction (modified LU, as in LAPACK `xORHR_COL`) in one fused kernel, so up to 16 panels can
  be aggregated into one wide update. The factorization stays backward stable and its accuracy does
  not depend on the condition number of A; no Gram or Cholesky QR is involved.
* **2.5D carriers.** Every product of an update (`W = VᵀX`, `Z = TᵀW`, `X −= V Z`, and the tree
  updates) runs on a `p_i × p_j` grid of GPUs, thread blocks or warps with a contraction replication
  `c` that satisfies `c² ≤ p_i p_j`. On 8 GPUs a 2×4 grid carries W with `c = 2`; inside a GPU, W, Z
  and the tree updates are replicated over blocks and warps. Every run reports the carriers it used.
* **Tuned schedules.** Tree shape, `c` of every product, GPU grid, aggregation, look-ahead and
  tails are tuned by measurement and kept in `reproducers/presets.json` for every measured matrix
  shape, precision and GPU count.

Ordered traffic and collectives use the low-latency NCCL of Shen et al.,
[Every Microsecond Matters](https://arxiv.org/abs/2607.16100); independent triangle publications use
NVSHMEM. [docs/replication.md](docs/replication.md) lists what each replication factor stores.

## Results

All measurements ran on NVIDIA DGX H200 nodes. QR-Ω is compared with the fastest tuned configuration
of cuSOLVER (one GPU), cuSOLVERMp, MAGMA and SLATE; TF32 and 3×TF32 are compared with the
references' FP32. The figures of the paper are drawn from the recorded measurements:

| Paper | File in `plots/figures/` | Content |
| --- | --- | --- |
| Fig. 2 | `fig02-motivation` | FP64 throughput of the QR libraries, DGEMM and the FP64 peak, one GPU |
| Figs. 6, 7 | `fig06-single-perf`, `fig07-single-acc` | One GPU, n = 256 to 131,072: speedup, throughput, errors |
| Figs. 8, 9 | `fig08-multi8-perf`, `fig09-multi8-acc` | Eight GPUs, n = 131,072 to 327,680 |
| Fig. 10 | `fig10-tall-skinny` | Eleven tall shapes, m up to 4,194,304, one GPU |
| Fig. 11 | `fig11-scaling` | Strong scaling at n = 229,376 on 2, 3, 4 and 8 GPUs |

On one GPU, QR-Ω factors n = 131,072 in 63.6 s in FP64 (47.2 TFLOP/s) and in 12.1 s in TF32
(249 TFLOP/s, 5.3× the fastest FP32 reference); below n ≈ 32,000 it is 1.4–2.2× faster than
cuSOLVER in FP64 and FP32. On eight GPUs it is 1.02–1.06× faster than cuSOLVERMp in FP64 and
1.11–1.12× in FP32, and reaches 317 TFLOP/s in FP64 and 1,327 TFLOP/s in TF32.
`plots/figures/paper-numbers.csv` recomputes every number the paper quotes from the data.

## Requirements

* NVIDIA Hopper GPUs (`sm_90a`); several GPUs must be in one node (measured: NVLink 4 through
  NVSwitch).
* CUDA 13.0 or newer (measured: 13.0.2), GCC 13, CMake 3.25+, Python 3.11+.
* Several GPUs: Open MPI (measured: 5.0.10), NVSHMEM 3.6.5 and the
  [low-latency NCCL](https://github.com/ss16118/low-latency-nccl) at commit
  `5357eff325eddf978137de7140195a5568fa8a11`.
* References (optional): cuSOLVERMp 0.8.0 with stock NCCL, MAGMA 2.10.0 (ILP64, with MKL), and
  SLATE 2025.05.28; cuSOLVER comes with CUDA.
* Figures (no GPU): the matplotlib and numpy versions of `plots/requirements.txt`.

CMake fetches the pinned CUTLASS and nlohmann/json; `CUTLASS_ROOT` and `JSON_INCLUDE_DIR` select
existing copies.

## Build

The presets build into `build/`. One GPU: the driver, its GPU tests and the host tests, then all of
them run:

```sh
cmake --workflow --preset one-gpu
```

The GPUs of a node, with the dependency prefixes in the environment:

```sh
export NCCL_ROOT=/path/to/low-latency-nccl/build NVSHMEM_ROOT=/path/to/nvshmem
cmake --preset node && cmake --build --preset node
```

The `paper` preset also builds the reference adapters from `MAGMA_ROOT`, `MKLROOT`, `SLATE_ROOT`,
`CUSOLVERMP_ROOT` and `REFERENCE_NCCL_ROOT` (stock NCCL); see
[reproducers/README.md](reproducers/README.md). At run time, put the low-latency NCCL first on
`LD_LIBRARY_PATH` for QR-Ω and stock NCCL first for cuSOLVERMp.

## Reproduce

Run timings on reserved GPUs, with nothing else on the node. The times are estimates for H200 GPUs.

1. **Tests** (about 3 minutes on one GPU). `ctest --preset host` checks the recorded data and the
   runner without a GPU; `ctest --preset gpu` checks the panels, the updates, the reconstruction and
   the factorization on one GPU. On several GPUs,
   `python3 tests/hierarchy_gpu.py build/bin/qr_omega_multi --gpus 8` and
   `mpirun -np 8 build/bin/qr_omega_transport_tests` check the distributed hierarchy and the
   transport.
2. **Figures and quoted numbers** (seconds, no GPU). From the recorded measurements:

   ```sh
   python3 -m venv .venv && .venv/bin/pip install -r plots/requirements.txt
   .venv/bin/python plots/reproduce.py
   ```

   This assembles `plots/data/paper.csv`, draws the figures into `plots/figures/` (bit for bit those
   of the paper with the pinned versions) and writes `plots/figures/paper-numbers.csv`, which lists
   every number the paper quotes next to the value recomputed from the data.
3. **Measurements.** `reproducers/runner.py` runs measured cells again with their recorded options
   and checks every run:

   ```sh
   python3 reproducers/runner.py --suite smoke     # every mode at n = 4,096 (under a minute)
   python3 reproducers/runner.py                   # the one-GPU cells the paper quotes (20 minutes)
   python3 reproducers/runner.py --suite tall      # the 44 tall cells (5 minutes)
   python3 reproducers/runner.py --gpus 8          # the eight-GPU cells the paper quotes (20 minutes)
   python3 reproducers/runner.py --gpus 8 --library cuSOLVERMp --sizes 131072
   ```

   `--suite all` runs every measured cell of a GPU count: about 40 minutes on one GPU, 1.5 hours on
   eight, and 2 hours on four; the references take longer (cuSOLVERMp, MAGMA and SLATE about 6 hours
   on eight GPUs). `--list` prints the measured cells and `--plan` the commands.

**Expected results.** Every QR-Ω run passes its residual and orthogonality checks, carries its
products within the 2.5D bound, and finishes within 4% of its recorded time (`--max-slowdown`).
The runner writes the record, log and command of every cell and a manifest with the time over the
recorded time to `results/<UTC time>/`, never overwrites earlier results, and exits nonzero if a
cell fails. [reproducers/README.md](reproducers/README.md) describes the measurement protocol and
the reference adapters.

## Read the code

| Location | Purpose |
| --- | --- |
| `benchmarks/qr_omega.cu` | The driver: input, timing, residual and orthogonality checks |
| `include/qr_omega/engine.cuh`, `plan.hpp` | The elimination list and its schedule on one or several GPUs |
| `include/qr_omega/domains.cuh`, `panel_register.cuh` | GE/TS/TT trees of register-resident nodes |
| `include/qr_omega/householder_reconstruct.cuh` | Thin Q of a tree and its reconstruction into compact WY, in one launch |
| `include/qr_omega/update.cuh`, `tree_update.cuh` | The updates and the carrier of every product |
| `include/qr_omega/carrier*.{hpp,cuh}` | The carried products on CUDA cores and tensor cores, and the 2.5D bound |
| `include/qr_omega/transport.cuh`, `low_latency_collectives.cuh` | Low-latency NCCL and NVSHMEM operations between GPUs |
| `reproducers/` | Measured cells, their runner, and the reference adapters |
| `plots/` | The measurements, the figures and the numbers of the paper |
| `tests/` | Host tests of the data and the runner; GPU tests of panels, updates, reconstruction and transport |

[docs/code-map.md](docs/code-map.md) follows a factorization through the code and lists the driver
options. Panel kernels adapt gau.nernst's GPU MODE QR-v2 submission 844219; FP64 carried updates
follow KAMI; tensor-core kernels use CUTLASS/CuTe. See [third-party notices](third_party/README.md).
