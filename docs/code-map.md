# Code map

`benchmarks/qr_omega.cu` builds as `qr_omega_single` and, with MPI, low-latency NCCL and NVSHMEM,
`qr_omega_multi`. Multi-GPU execution uses one MPI rank per GPU.

## Factorization path

1. The driver generates a reproducible matrix, times `Engine::factor`, and validates reconstruction
   over every column plus orthogonality on 16 vectors through reverse replay in `Engine::apply_q`.
2. `plan.hpp` distributes rows and columns block-cyclically over a GPU grid. Each panel's owning
   grid column factors local rows; node-level TT merges combine the resulting triangles.
3. `domains.cuh` implements each GPU's hierarchy: GEQRT tiles, TSQRT chains inside domains and
   TTQRT between domains. A persistent kernel schedules the nodes and publishes completed column
   blocks to their consumers. GE/TS/TT reflectors remain in the matrix; scalar taus are retained.
   Every node is factored by the register kernel of `panel_register.cuh` (`gx_body`), which also
   factors node-level TT merges; its warps own four columns each and publish them to consumers.
   Two reusable T/V workspaces serve the panels in flight. Q replay repacks V and rebuilds T by
   dlarft from reflector overlaps, without retaining every panel's full compact WY factors.
   With `--panel-format wy`, `householder_reconstruct.cuh` forms the tree's native-precision
   thin Q down the same GE/TS/TT hierarchy and converts it into standard Householder vectors by
   modified LU and triangular solves, in one launch: a block per domain follows its path from
   the node it enters, the nodes publish their Z = T E, and a dedicated block factors the top
   rows as soon as they exist. This format retains one small T per panel and supports
   composition into wider updates. Both choices start with the same GE/TS/TT hierarchy.
4. `engine.cuh` shares those factors along GPU grid rows and schedules updates in strips. Lookahead
   releases the next panel before starting the far strips. Node TT merges use register-resident
   Householder panels. Node updates sum partial W where the replication bound permits it; other
   GPU grids contract gathered operands.
5. `update.cuh` applies W = V^T X, Z = op(T) W and X -= V Z along the GE/TS/TT tree, reversing the
   order for Q replay. `tree_update.cuh` fuses updates in all four modes through the arithmetic
   adapter in `tree_math.cuh`. Segmented carriers support all modes
   and GPU-level contraction groups. Each actual launch records its `(p_i, p_j, c)` decomposition;
   `carrier.hpp` checks `c^2 <= p_i p_j` at every machine level.
6. `low_latency_collectives.cuh` uses the low-latency NCCL paper's `ncclLLBuffer` protocol for row
   broadcasts, column sum/max/allgather and ordered pair exchanges. NCCL windows contain protocol
   buffers and reuse credits. `transport.cuh` uses NVSHMEM with per-peer generations to publish
   independently produced local triangles into disjoint slots on the merge owner. The merged
   V/T packets and reconstruction factors use ordered NCCL exchanges. CUDA events preserve
   dependencies across streams.

## Options

| Driver option | Meaning |
| --- | --- |
| `--mode` | `fp64`, `fp32`, `tf32`, or `3xtf32` |
| `--m`, `--n` | Matrix dimensions, m >= n >= 1 |
| `--kappa` | Conditioned input of this condition number (square, power-of-two order) |
| `--b` | Panel width, at most 64 |
| `--grid-rows` | Rows of the GPU grid; must divide the GPU count |
| `--domain-tiles` | Tiles per domain: one GE followed by TS eliminations |
| `--fan-in` | TT arity between GPU domains, 2 through 4 |
| `--dc`, `--mc` | Requested domain and merge update replication |
| `--tree-groups` | Requested groups of blocks for the GPU tree contractions |
| `--tree-update` | Input staging: `streaming` or `resident` in shared memory, in all four modes |
| `--panel-format` | Retained `tree` factors or stable reconstructed `wy` factors |
| `--strip`, `--depth` | Columns per update strip and strips in flight |
| `--lookahead`, `--no-lookahead` | Overlap the next panel with the far update, or not |
| `--aggregate` | 1 through 16 reconstructed panels per far update; values above 1 require WY and one GPU grid row |
| `--wc`, `--zc` | Requested W/Z replication for non-fused products |
| `--d-tiles` | 3×TF32 output tiles per thread block of the far D |
| `--tail-from` | One GPU: the column from which a trailing block is factored in the tree format |
| `--tail-<option>` | The schedule of that trailing block (for instance `--tail-strip`, `--tail-no-lookahead`) |
| `--reps`, `--output` | Timed repetitions after one warmup and JSON output path |
| `--no-check` | Time the schedule without the numerical checks |
| `--profile-factor` | CUDA profiler range around the first timed factorization |

## Measurement and validation

The schedules are tuned by hand and kept in `reproducers/presets.json`, one entry per measured
shape, arithmetic and GPU count, with the driver options and the measured time and errors of the
cell. `reproducers/runner.py` runs any of them again and checks the result against the recorded
measurement; `plots/reproduce.py` draws the figures of the paper from the recorded measurements.

`ctest --preset host` runs the host tests of `tests/test_*.py`: the recorded data agree with each
other, the runner selects, runs and judges cells as documented, and the figures and quoted numbers
regenerate from the data. `ctest --preset gpu` covers GE/TS/TT panels, scaled and rank-deficient
inputs, partial panels, updates, reconstruction, tails and Q replay on one GPU. Run
`tests/hierarchy_gpu.py --gpus P` and `qr_omega_transport_tests` explicitly inside a multi-GPU
allocation to validate the distributed hierarchy and both communication backends.
