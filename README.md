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
  tails are tuned by measurement and kept in `reproducers/presets.json` for every matrix size,
  precision and GPU count in the paper.

Ordered traffic and collectives use the low-latency NCCL of Shen et al.,
[Every Microsecond Matters](https://arxiv.org/abs/2607.16100); independent triangle publications use
NVSHMEM. [docs/replication.md](docs/replication.md) lists what each replication factor stores.

## Results

NVIDIA H200 GPUs in DGX nodes. Each time is the median of the timed repetitions of two validated runs
of the cell's configuration (`reproducers/presets.json`). The references are cuSOLVER (one GPU),
cuSOLVERMp, MAGMA and SLATE, each tuned for every size (on eight GPUs, tuned at n = 131,072 and its
two best configurations measured at every size); TF32 and 3×TF32 are compared with the references'
FP32. The figures are in `plots/figures`.

| GPUs | Cells | Speedup over the fastest reference |
| --- | --- | --- |
| 1 | 81: n = 256 to 131,072, four modes | FP64 1.01–1.92, FP32 0.98–2.24, TF32 1.75–5.34, 3×TF32 1.52–2.72 |
| 2, 3 | 6: n = 229,376, FP32/TF32/3×TF32 | FP32 1.12, TF32 5.71–6.07, 3×TF32 2.13–2.15 |
| 4 | 25: n = 131,072 to 327,680 | FP64 1.12–1.21, FP32 1.12–1.22, TF32 4.71–5.27, 3×TF32 2.19–2.29 |
| 8 (2×4 grid) | 27: n = 131,072 to 327,680 | FP64 1.02–1.06, FP32 1.11–1.12, TF32 3.45–4.21, 3×TF32 1.89–1.99 |

The largest factorizations run at 47.2 TFLOP/s (FP64, one GPU, n = 131,072), 177.5 (FP64, four GPUs,
n = 229,376), 317 (FP64, eight GPUs, n = 294,912) and 1,327 (TF32, eight GPUs, n = 327,680).

## Build

Requires Hopper (`sm_90a`), CUDA 13.0+ (measured with 13.0.2), GCC 13, CMake 3.25+ and Python 3.11+.
CMake fetches the pinned CUTLASS and nlohmann/json; `CUTLASS_ROOT` and `JSON_INCLUDE_DIR` select
existing copies.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 16
```

For several GPUs, install Open MPI, NVSHMEM 3.6.5 and the
[low-latency NCCL](https://github.com/ss16118/low-latency-nccl) at commit
`5357eff325eddf978137de7140195a5568fa8a11`, put that NCCL first on `LD_LIBRARY_PATH`, and configure:

```sh
cmake -S . -B build -DQR_OMEGA_BUILD_MULTI_GPU=ON -DQR_OMEGA_BUILD_GPU_TESTS=ON \
  -DNCCL_ROOT=/path/to/low-latency-nccl/build -DNVSHMEM_ROOT=/path/to/nvshmem \
  -DCMAKE_PREFIX_PATH=/path/to/mpi
cmake --build build --parallel 16
```

## Reproduce

Run timings on reserved GPUs at fixed clocks, with nothing else on the node.

```sh
ctest --test-dir build -L gpu --output-on-failure            # GPU correctness tests
python3 -m unittest discover -s tests                         # host tests
python3 reproducers/runner.py --gpus 8 --sizes 131072 --plan  # print the 8-GPU commands
```

Every measurement of the paper on one node with eight H200 GPUs, each cell with its recorded options:

```sh
for p in 1 2 3 4 8; do python3 reproducers/runner.py --gpus $p --suite paper --output results/p$p; done
```

That is 139 cells (81 on one GPU, 3 each on two and three, 25 on four, 27 on eight) and takes about
six hours, most of it in the largest four- and eight-GPU matrices; `--sizes` and `--modes` select
subsets, and the default `--suite headlines` runs the cells the paper quotes. Each run checks the
residual `max_J ‖A_J − (QR)_J‖_F / ‖A_J‖_F` over the 512-column blocks J of all n columns and the
orthogonality error `‖Qᵀ(QX) − X‖_F / ‖X‖_F` for 16 random vectors X, reports how every product was
carried, and fails if it is more than 4% slower than its recorded time.

The figures come from the measured cells and the reference measurements
(`plots/data/references.csv`: cuSOLVER, cuSOLVERMp, MAGMA and SLATE, each tuned for every size;
TF32 and 3×TF32 are compared with their FP32):

```sh
python3 -m venv .venv && .venv/bin/pip install -r plots/requirements.txt
python3 plots/collect.py                 # plots/data/paper.csv
.venv/bin/python plots/reproduce.py      # plots/figures/*.pdf and headline-results.csv
```

The schedules are tuned by hand: each preset in `reproducers/presets.json` records the driver
options of its cell, and [docs/code-map.md](docs/code-map.md) lists them. A new size starts from the
preset of a neighbouring size. [reproducers/README.md](reproducers/README.md) covers the reference
adapters and the measurement protocol.

## Read the code

| Location | Purpose |
| --- | --- |
| `benchmarks/qr_omega.cu` | The driver: input, timing, residual and orthogonality checks |
| `include/qr_omega/engine.cuh`, `plan.hpp` | The elimination list and its schedule on one or several GPUs |
| `include/qr_omega/domains.cuh`, `panel_register.cuh` | GE/TS/TT trees of register-resident nodes |
| `include/qr_omega/householder_reconstruct.cuh` | Thin Q of a tree and its reconstruction into compact WY, in one launch |
| `include/qr_omega/update.cuh`, `tree_update.cuh` | The updates and the carrier of every product |
| `include/qr_omega/carrier*.{hpp,cuh}` | The carried products on CUDA cores and tensor cores, and the 2.5D bound |
| `include/qr_omega/transport.cuh`, `paper_collectives.cuh` | Low-latency NCCL and NVSHMEM operations between GPUs |
| `reproducers/` | Measured cells, their runner, and the reference adapters |
| `plots/` | Measurements and the scripts that draw the figures |
| `tests/` | Host contracts, GPU panel, update, reconstruction and transport tests |

Panel kernels adapt gau.nernst's GPU MODE QR-v2 submission 844219; FP64 carried updates follow KAMI;
tensor-core kernels use CUTLASS/CuTe. See [third-party notices](third_party/README.md).
