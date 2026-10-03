# Code map

`reproducers/runner.py` runs the measured presets of `reproducers/presets.json` through the driver
`benchmarks/qr_omega.cu`, which is built twice: `qr_omega_single` for one GPU and, with MPI, NCCL and NVSHMEM,
`qr_omega_multi` for the GPUs of a node (one MPI rank per GPU).

## Factorization path

1. The driver generates the matrix on the GPUs (block-cyclic rows on several GPUs), times `Engine::factor`, and
   checks the result with `Engine::apply_q`: the residual over column blocks and the orthogonality error on 16
   vectors.
2. `plan.hpp` builds the elimination list of Algorithm 1: per panel, one GE of the rows each GPU holds and, across
   GPUs, one TT elimination of arity P that merges the GPUs' triangles.
3. `engine.cuh` runs the list. Panels are factored by `panel_cooperative.cuh` (tall panels, rows split among
   thread blocks, windows and mini-panels) or `panel_register.cuh` (the last panels and the TT merges, columns held
   in registers). Every panel updates the trailing columns in strips; under look-ahead the strips that release the
   next panel go first and the far strips overlap its factorization. On one GPU, `aggregate` panels are composed
   into one transform for the far columns. On several GPUs the TT merge gathers the triangles on the owner of the
   diagonal block, and its update all-reduces W.
4. `update.cuh` computes W = V^T X, Z = T^T W and X -= V Z for one strip and chooses the carrier of each product:
   cuBLAS peers (FP64 and IEEE FP32 W), CUTLASS group carriers (`carrier_cutlass.cuh`), block carriers on CUDA cores
   (`carrier_block.cuh`), the KAMI-style D carriers (`carrier_kami.cuh`), the SIMT Gram carrier
   (`carrier_simt.cuh`), and the wgmma kernels of the TF32 modes (`carrier_wgmma.cuh`). Every product runs with
   c >= 2 peers where its contraction admits the cut; `Tally` counts them.
5. `transport.cuh` moves data between GPUs: one-sided NVSHMEM publications, the low-latency NCCL `LLBuffer`
   exchanges, and `ncclAllReduce` for W.

## Options

| Driver option | Meaning |
| --- | --- |
| `--mode` | `fp64`, `fp32` (IEEE), `tf32`, or `3xtf32` |
| `--m`, `--n` | Matrix dimensions |
| `--b` | Panel width (128 on one GPU, 512 across GPUs) |
| `--strip` | Columns of a strip of the trailing update |
| `--depth` | Strips in flight |
| `--aggregate` | Panels composed into one far update (one GPU) |
| `--lookahead` | Factor the next panel while the far update runs |
| `--wc`, `--zc` | Contraction replication c of W and Z (D always uses 2) |
| `--groups`, `--late-groups`, `--late-rows` | Thread blocks of a cooperative panel, and of panels with fewer rows |
| `--minipanel`, `--window` | Mini-panel and window widths of a cooperative panel (0: the whole panel) |
| `--compose-groups` | TF32: groups of blocks of the Gram products of an aggregated T |
| `--d-tiles` | 3xTF32: output tiles per thread block of the far D |
| `--reps`, `--output` | Timed repetitions after one warm-up, and the JSON record |
