# Schedule selection

`planner/` chooses the schedule of a QR-Ω factorization from the machine profile of the GPU it runs on
(`machine/README.md`). It replaces a hand-measured preset by

    schedule* = argmin { T(schedule; machine, m, n, arithmetic) : schedule admissible } ,

where `T` is a cost model whose machine terms come from the profile, and "admissible" means faithful to the
algorithm and within the capacity of every memory level.

```sh
python3 -m planner --n 20000 --mode fp64                  # schedule, predicted time, command line
python3 -m planner --n 20000 --mode fp64 --run            # factor with it
python3 -m planner --n 20000 --mode fp64 --refine 3       # measure the three best, keep the fastest
python3 reproducers/runner.py --auto --sizes 20000 --modes fp64 tf32
python3 reproducers/runner.py --auto --gpus 4 --sizes 180224 --modes fp64
```

Without `--machine <profile>` the visible GPU is probed once and its profile cached per GPU and driver.

## What is selected

The elimination list is the engine's and never changes: one GE per panel on every GPU and, across GPUs, one TT
elimination of arity P. The planner chooses the moves of every elimination, which are driver options:

| Move | Parameter | Option |
| --- | --- | --- |
| Split, Replicate, Combine of the update | contraction replication `c_W`, `c_Z` of `W = VᵀX` and `Z = TᵀW` (`D = VZ` splits in two) | `wc`, `zc` |
| Split of the panel | blocks `G` that share its rows, mini-panel `p` held in registers, window `s` held in shared memory, and the blocks of the shorter panels | `groups`, `minipanel`, `window`, `late-groups`, `late-rows` |
| Transform combine | panels `a` composed into one far update of rank `a·b` | `aggregate` |
| Pipeline | strip width `q`, strips in flight `d`, look-ahead of the next unit | `strip`, `depth`, `lookahead` |

The panel width `b` is the largest power of two with `b²w ≤ M_shared` (`w` the word size), so that the
triangular factor of a panel stays in one block; on Hopper that is the engine's limit of 128.

## Admissible schedules

1. **Faithful.** `c_W, c_Z ≥ 2` and `G ≥ 2`: every carried product and the panel product split their
   contraction among at least two peers and combine their partials. A schedule with `c = 1` is not in the
   search space, whatever it would cost.
2. **Within capacity.**
   * Device memory: the matrix, the `T` factors, the `c_W + c_Z` partials of `a·b × q` words for each of the
     `d` strips in flight (twice under look-ahead), the composed `V` and `T`, and the validation buffers of the
     driver must fit the free memory of the profile (`schedule.py: memory_bytes` is the inventory of
     `engine.cuh` and `update.cuh`).
   * Shared memory of a block: the slab of a cooperative panel, `⌈r/G⌉(s + 1) + p·s` words for `r` rows, must
     fit `M_shared` less what the kernel declares statically (`kernels.json`). The window is the widest of
     `s`, 64, 32, 16, 8 that fits.
3. **Engine limits.** Panels taller than 8,192 rows use the cooperative carrier, `G ≤ 128`.

## Cost model

The factorization is a sequence of units of `a` panels. The unit whose first panel has `r` rows costs

    M = π Σ P(r_i) + a(a−1)/2 · (U(r, b, b) + C(r))      main lane: panels, near applies, composition
    A = U(r, a·b, a·b)                                    update of the next unit's columns
    F = U(r, a·b, c − a·b)                                update of the remaining c − a·b columns

Without look-ahead a unit takes `M + U(r, a·b, c)`. With look-ahead the main lane of the next unit runs beside
`F`, and a cycle takes `A` plus

    overlap = σ·P_M + O_M + max(0, F − σ·P_M·φ·ε) + δ·L·(1 − φ)·min(S, F) + j·λ_join ,   φ = 1 − G/P_sm .

`P_M` is the panel part of `M` and `O_M` the rest. A panel beside a far update runs `σ` times slower, and the far
update proceeds on the share `φ` of the SMs the panel leaves, with efficiency `ε`. Each of the `L` panel
launches first waits for the far update's kernel in flight (`S` is one strip of it) to release the SMs it needs.
The other main-lane kernels use the whole device.

**Panel.** A register panel is a chain of handshakes between blocks, each a block barrier and a round trip
through L2: `P(r) = b·e(r)·(λ_barrier + λ_L2)`, with `e(r)` tabulated against `log₂ r`. A cooperative panel adds
the work on the window its blocks hold in shared memory and the update of the rest of the panel at every window
boundary, bound by device memory:

    P(r) = b·h·λ_sync + (b/s − 1)·l·λ_launch + ψ_s·(r/G)·b·s·w / B_shared,sm + ψ_b·r·b·(b/s − 1)·w / B_memory,sm .

More blocks shorten the shared-memory term and admit a wider window; under look-ahead they also take SMs from the
far update. A slow far update (FP64) wants few blocks, a fast one (TF32) more.

**Update.** `U(r, k, q)` applies a transform of rank `k` to `q` columns in strips. With the product law of the
profile, `2mnkγ(1 + κ(1/k + ξ(1/m + 1/n)))`, a strip of width `q'` costs

    η·γ·F                                                                    arithmetic
    + ζ·γ·κ·[F_W(c_W/r + ξ(1/k + 1/q')/c_W) + F_Z(1/k + ξ(1/k + 1/q')) + F_D(1/k + ξ(1/r + 1/q'))]
                                                                             short contraction, small output
    + ω·γ·F_W / c_W                                                          share of W that only peers recover
    + ρ·(r/128)·k·q'·w·max(0, 1 − M_L2/(4k·q'·w))·(1/B_memory − 1/B_L2)     reads of Z that leave L2
    + (c_W + c_Z)·k·q'·w / B_copy                                            additive combines

with `F_W = F_D = 2kq'r`, `F_Z = k²q'`, `F` their sum; every further strip adds `s_c·λ_launch`, and a second strip
in flight multiplies a strip by `1 + δ_d`. `C(r)` is the Gram block and the triangular products that add a panel
to the composed transform.

**Kernel constants** (`kernels.json`) are dimensionless: handshakes per column, passes of the data through a
level, efficiencies against the vendor product. They describe the engine's kernels relative to the machine.

| Constant | FP64 | FP32 | TF32 | 3xTF32 | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| `e(r)`, r = 128 … 8,192 | 6.5 … 18.9 | 4.4 … 16.4 | as FP32 | as FP32 | handshakes per column, register panel |
| `h`, `l` | 24.8, 11.2 | 22.8, 0 | | | handshakes per column, launches per window boundary |
| `ψ_s`, `ψ_b` | 4.3, 0.11 | 4.3, 1.4 | | | passes through shared and device memory |
| `π` | 1.08 | 1.14 | 1.02 | 1.07 | panel inside a factorization against a panel alone |
| `η` | 0.97 | 0.69 | 0.00 | 0.39 | arithmetic of the update against the vendor product |
| `ζ` | 3.5 | 3.4 | 2.3 | 0.62 | traffic of the update against the vendor product |
| `ω` | 0 | 1.0 | 0.5 | 4.75 | share of `W` that peers recover |
| `ρ` | 0 | 0 | 1.75 | 0.25 | reads of `Z` per 128 rows that leave L2 |
| `s_c` | 24 | 24 | 0 | 4 | launches per further strip |
| `δ_d` | 0 | 0.02 | 0.06 | 0 | cost of a second strip in flight |
| composition | 16.9, 0.2 | 17.1, 3.4 | 28.8, 0.9 | 16.4, 0.0 | Gram block against the vendor product, launches per panel |
| `σ`, `ε`, `δ`, `j` | 1.25, 0.95, 0.01, 0 | 1.35, 0.95, 0.03, 0 | 1.2, 0.9, 0, 2 | 1.6, 0.95, 0, 2.5 | overlap |

`η` and `ζ` are fitted together and read together: for TF32 the whole fitted cost of the update sits in the
traffic term, as a tensor-core product of this shape is bound by the operands it streams.

The constants were fitted to 962 measurements, panels alone and whole factorizations, on two H200 GPUs of one node,
one at 1.98 GHz and one whose SM clock is capped at 1.50 GHz (`calibration.jsonl`). `python3 -m planner.calibrate --fit planner/calibration.jsonl`
reproduces `kernels.json`; `--measure runs.jsonl` measures the set on the GPU of the allocation (about an hour).
A refit is needed after changing a kernel, not when moving to another GPU the engine supports: there the profile
carries the change.

## Selection

The parameters that interact (aggregation, look-ahead, panel blocks, the shorter panels' blocks, strip) are
ranked exhaustively, the others by coordinate descent from the best points; planning takes a few seconds.
`a ∈ {1, 2, 4, 8}`, the ranks the composed update was measured with. Predictions closer than 1 % are below the
resolution of the model; among them the schedule that replicates and buffers least is chosen. TF32 uses the
64-group Gram carrier from n = 48,000 on, as the presets do. `--refine K` measures the K best schedules and keeps
the fastest.

Across GPUs (`plan.py: node_schedule`) the node-level carriers are fixed by the algorithm (`W`: `(1, 1, P)`,
`D`: `(P, 1, 1)`) and the machine decides the panel carrier inside each GPU: the fewest blocks whose 32-column
window fits the shared memory of a block for the rows one GPU holds.

## Validation

`planner/validation/` holds the planner against the presets on the two GPUs the constants were fitted on;
`python3 -m planner.validate --report <file>` prints these tables. Every case is factored with the preset and with
the planned schedule, each timed as the median of its repetitions; a ratio below 1 means the planned schedule is
faster.

H200 at 1.98 GHz:

| n | FP64 preset (s) | planned (s) | ratio | FP32 preset (s) | planned (s) | ratio | TF32 preset (s) | planned (s) | ratio | 3xTF32 preset (s) | planned (s) | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.002177 | 0.002221 | 1.021 | 0.001965 | 0.002089 | 1.063 | 0.001747 | 0.001749 | 1.001 | 0.001822 | 0.001825 | 1.002 |
| 2,048 | 0.00586 | 0.005774 | 0.985 | 0.006015 | 0.005224 | 0.868 | 0.004356 | 0.004431 | 1.017 | 0.004667 | 0.004645 | 0.995 |
| 4,096 | 0.01583 | 0.01566 | 0.989 | 0.01622 | 0.01602 | 0.988 | 0.01453 | 0.01446 | 0.995 | 0.01564 | 0.01525 | 0.975 |
| 8,192 | 0.04388 | 0.04306 | 0.981 | 0.0434 | 0.04478 | 1.032 | 0.03667 | 0.03611 | 0.985 | 0.0418 | 0.04033 | 0.965 |
| 12,000 | 0.1067 | 0.1013 | 0.950 | 0.1105 | 0.1025 | 0.928 | 0.07847 | 0.07387 | 0.941 | 0.08979 | 0.08267 | 0.921 |
| 16,384 | 0.2007 | 0.2069 | 1.031 | 0.2025 | 0.1982 | 0.979 | 0.1179 | 0.1213 | 1.029 | 0.1509 | 0.1471 | 0.975 |
| 20,000 | – | 0.3253 | – | – | 0.3302 | – | – | 0.17 | – | – | 0.2249 | – |
| 24,000 | 0.5117 | 0.5189 | 1.014 | 0.5261 | 0.5305 | 1.008 | 0.2322 | 0.2285 | 0.984 | 0.3271 | 0.3331 | 1.018 |
| 32,768 | 1.185 | 1.204 | 1.016 | 1.198 | 1.196 | 0.998 | 0.3925 | 0.409 | 1.042 | 0.6674 | 0.7223 | 1.082 |
| 40,000 | – | 2.098 | – | – | 2.073 | – | – | 0.6007 | – | – | 1.153 | – |
| 48,000 | 3.421 | 3.499 | 1.023 | 3.522 | 3.464 | 0.983 | 0.8797 | 0.8978 | 1.021 | 1.842 | 1.871 | 1.016 |
| 65,536 | 8.207 | 8.59 | 1.047 | 8.363 | 8.526 | 1.019 | 1.827 | 1.873 | 1.025 | 4.285 | 4.432 | 1.034 |
| 131,072 | 63.82 | 64.29 | 1.007 | 65.35 | 63.1 | 0.965 | 12.08 | 12.38 | 1.024 | 32.07 | 32.41 | 1.011 |

H200 at 1.50 GHz:

| n | FP64 preset (s) | planned (s) | ratio | FP32 preset (s) | planned (s) | ratio | TF32 preset (s) | planned (s) | ratio | 3xTF32 preset (s) | planned (s) | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.007272 | 0.007135 | 0.981 | 0.007658 | 0.006499 | 0.849 | 0.005485 | 0.005614 | 1.023 | 0.005878 | 0.005873 | 0.999 |
| 4,096 | 0.01932 | 0.01922 | 0.995 | 0.01946 | 0.01912 | 0.982 | 0.01757 | 0.01748 | 0.995 | 0.01906 | 0.01851 | 0.971 |
| 8,192 | 0.05424 | 0.05383 | 0.993 | 0.05367 | 0.0553 | 1.030 | 0.04512 | 0.04446 | 0.985 | 0.05176 | 0.04875 | 0.942 |
| 16,384 | 0.2471 | 0.2428 | 0.983 | 0.2587 | 0.2528 | 0.977 | 0.1459 | 0.1502 | 1.030 | 0.184 | 0.1779 | 0.967 |
| 24,000 | 0.6062 | 0.6179 | 1.019 | 0.6777 | 0.6851 | 1.011 | 0.2853 | 0.2822 | 0.989 | 0.3848 | 0.3933 | 1.022 |
| 32,768 | 1.384 | 1.389 | 1.004 | 1.556 | 1.574 | 1.011 | 0.4786 | 0.5022 | 1.049 | 0.7688 | 0.7729 | 1.005 |
| 48,000 | 3.891 | 4.007 | 1.030 | 4.56 | 4.556 | 0.999 | 1.018 | 1.061 | 1.043 | 2.017 | 2.049 | 1.015 |
| 65,536 | 9.485 | 9.752 | 1.028 | 10.92 | 11.18 | 1.023 | 2.016 | 2.282 | 1.132 | 4.578 | 4.785 | 1.045 |

Over the 44 cases with a preset at 1.98 GHz, planned over preset runs from 0.868 to 1.082 with a geometric mean of
0.998; over the 32 at 1.50 GHz, from 0.849 to 1.132 with a geometric mean of 1.003. The planned schedules are as fast
as the measured ones. They are more than 3 % slower in 7 and 5 cases, by the most (8 % and 13 %) in TF32 and
3xTF32 from n = 32,768 on, and faster where the preset is not the best schedule (FP32 at n = 2,048, every arithmetic
at n = 12,000).

On the planned schedules the model's median error is 8.0 % at 1.98 GHz and 5.2 % at 1.50 GHz. It underestimates
TF32 and 3xTF32 from n = 32,768 on, by up to 29 % within the calibrated sizes (n ≤ 65,536) and 37 % at n = 131,072.

Across GPUs, `node_schedule` reproduces 30 of the 31 multi-GPU presets; the other one, FP64 at n = 163,840 on four
GPUs, differs only in its panel blocks (64 against 48).

## Limits

* Measured on H200 at two SM clocks. On another architecture the profile and the admissibility rules apply as
  they are; the kernel constants carry where the kernels are the same, and `planner.calibrate` refits them where
  they are not. The engine itself is built for Hopper.
* Across GPUs only the panel carrier is selected from the profile; the panel width, the block-cyclic block and
  the strip are the published ones, and no model of the node-level terms uses the peer link yet.
* For TF32 and 3xTF32 at large n, where a fast update makes the overlap of panel and update decisive, the model
  is least accurate and its choice up to 13 % slower than the preset; `--refine` settles those by measurement.
