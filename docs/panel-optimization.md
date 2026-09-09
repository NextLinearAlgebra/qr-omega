# Reduce overhead in the domino panel

At minipanel column `jj`, the next step consumes only dot slots `jj+1..15` and
sigma slot `16`. The original implementation nevertheless reduced all 17 slots
at every column. With 512 threads (16 warps), warp zero reduced unused slot zero
before reducing the required sigma, delaying the block barrier.

The opt-in candidate makes three changes:

1. Start the slot assignment at `jj+1+warp`. Per CTA, a complete 16-column
   minipanel performs 136 slot reductions instead of 272. Each required slot
   retains the original lane-stride accumulation, shuffle tree and arithmetic
   precision; all threads still participate in the block barriers.
2. Force-inline only `d_dom_region()`, the per-column routine. Other phase
   bodies retain `__noinline__`, preserving the original register-pressure
   tradeoff elsewhere.
3. Select the existing cooperative-grid kernel variant and launch it through
   `cudaLaunchCooperativeKernel`. Configure its dynamic shared-memory allowance
   and check the grid against the actual kernel's occupancy limit. Synchronization
   points, launch geometry and data dependencies remain unchanged. Successful
   cluster launches continue to use the original cluster path.

## Historical measurements

H100 PCIe 80GB, 114 SMs; CUDA 13.0.88, driver 580.159.04. The September 8, 2026
experiment compared the original algorithm with this combination in an isolated
source snapshot. The new Git branch is a port of that work and **has not yet
been rebuilt or run on a GPU**. The numbers below are historical evidence, not a
claim that this checkout has already passed GPU validation.

ABBA process order; two warmups and seven measured factorizations per process;
median aggregation; identical planner inputs. Times exclude setup and checks.
Gain = original time / candidate time - 1. TFLOP/s uses the effective square QR
work count, 4n^3/3, rather than all operations executed internally.

| Precision | Square n | Original → candidate ms | Candidate TFLOP/s | Gain | cuSOLVER speedup: before → after |
|---|---:|---:|---:|---:|---:|
| FP64 | 8192 | 89.84 → 76.75 | 9.55 | +17.1% | 1.02x → 1.20x |
| FP64 | 16384 | 324.26 → 301.85 | 19.43 | +7.4% | 1.32x → 1.42x |
| FP32 | 8192 | 86.82 → 72.28 | 10.14 | +20.1% | 0.95x → 1.14x |
| FP32 | 16384 | 299.28 → 275.80 | 21.26 | +8.5% | 1.35x → 1.46x |
| TF32 | 8192 | 80.02 → 64.64 | 11.34 | +23.8% | 1.03x → 1.28x |
| TF32 | 16384 | 191.93 → 157.40 | 37.26 | +21.9% | 2.10x → 2.56x |

TF32 uses FP32 cuSOLVER as reference; this is not an equal-precision comparison.
Independent repeats reproduced TF32 gains above 20% at both sizes. FP32 8192
was approximately 19-20% across rounds. At 32768, FP64/FP32/TF32 gains were about
1%/2%/11%; the optimizations do not deliver 20% everywhere.

Individual changes each improved an 8192-row FP64/FP32 panel microbenchmark by
about 7-9%, and the combination by 25-29%. These are panel-only measurements,
not an end-to-end ablation of each individual change.

## Validation and limitations

- The isolated candidate passed 41 full numerical cases and all 41 within-build
  bitwise-repeatability checks. Full-gate thresholds were unchanged. This does
  not assert bitwise equality between baseline and candidate output arrays.
- The imported performance driver adds explicit tier/shape selection, warmups,
  median timing, cuSOLVER error checks and a fixed-probe test fixture. It aligns
  the sampled TF32 check's epsilon with the existing full TF32 gate; it does not
  establish FP32-level accuracy for TF32. Both performance arms use this driver.
- Four fixed-plan panel checks (FP64/FP32, memcheck/synccheck) reported zero errors.
  Unfrozen runs encountered rejected cluster-launch API calls in both original
  and candidate code. That existing cluster fallback issue remains unresolved;
  these results are not an unconditional sanitizer pass or a race-freedom proof.
- The calibration probe still measures the manual grid-barrier path. Normal
  recalibrating TF32 8192/16384 tests retained the performance gain, but consistent
  cooperative calibration remains future work. Printed `carrier=gmem` is the
  planner's label, not evidence that the cooperative runtime path was skipped.
- 3xTF32 is excluded from the optimization evaluation. Its intended backend
  should be confirmed separately. Other GPU models have not been measured.

This port is being submitted after static checks only because cloud GPU
resources are unavailable. CUDA compilation and GPU execution have not been
repeated for this checkout. When a Hopper machine is available, rebuild both
configurations, run the full gates and determinism checks, and repeat
representative controlled and normal-calibration performance comparisons.
