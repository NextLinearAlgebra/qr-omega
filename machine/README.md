# Machine profile

`machine_probe` measures, in a few seconds, the quantities a cost model of a dense linear-algebra kernel needs
from a GPU and its node. The result is a JSON profile. `machine/` reads it and offers the terms of a cost model
and a small tuner; nothing in this directory depends on QR. The QR-Ω schedule is one client (`planner/`).

```sh
cmake --build build --target machine_probe
build/bin/machine_probe --output machine/profiles/mine.json     # inside the GPU allocation, about 7 s
python3 -m machine.collect --show machine/profiles/mine.json    # basic metrics
```

The probe is built for the architecture of the build host (`-DMACHINE_PROBE_ARCHITECTURES=native`); set it to
`all-major` or to a list such as `80;90;100` to build one binary for several. It uses only portable CUDA and
cuBLAS, so it runs wherever the vendor product runs, with whatever tensor instructions that GPU has.

## Model

A machine is a hierarchy of memory levels ℓ, each with a capacity `M_ℓ`, an access latency `λ_ℓ` and a bandwidth
`B_ℓ`, an arithmetic rate `Γ_a` per arithmetic `a`, and the latency of the operations that order work. A kernel
that sends `S_ℓ` messages and `W_ℓ` bytes across level ℓ and performs `F` operations costs

    T = Σ_ℓ ( S_ℓ λ_ℓ + W_ℓ / B_ℓ ) + F / Γ_a .

This is the α–β–γ model of communication-avoiding linear algebra, written once per level of the GPU: registers and
shared memory of an SM, L1, L2, device memory, the link to a peer GPU, and the link to the host. Bandwidth through
a level scales with the SMs that drive it, `B_ℓ(s) = min(B_ℓ, s · B_ℓ,sm)`, which is what an algorithm pays when
it gives SMs to something else.

Two dimensionless numbers place a kernel in one of three regimes:

* `balance_a = Γ_a / B_copy` (operations per byte): a kernel of lower arithmetic intensity is bound by bandwidth;
* `grain_a = Γ_a · λ_launch` (operations): a kernel that performs fewer is bound by latency.

### Product law

Every blocked factorization spends its operations in products `C(m×n) += A(m×k) B(k×n)`. The probe times the
vendor product on three families of shapes (square, rank-k update, inner product) and `Machine.product(a)` fits

    t(m, n, k) = t_call + 2 m n k γ_a ( 1 + κ_a ( 1/k + ξ_a ( 1/m + 1/n ) ) ) .

`γ` is the time per operation of a compute-bound product. `κ` is the contraction length at which the traffic of
the output costs as much as the arithmetic: a rank-k update runs at the fraction `k / (k + κ)` of the peak rate,
and `κ` is 0.4 to 0.7 times `w · Γ / B_copy` on the GPUs measured so far (`w` is the word size). `ξ` weighs the
two operands against the output.

The law was checked on twelve shapes per arithmetic that are not in the fit, from `60000 × 256, k = 256` to
`512 × 512, k = 30000` (`machine_probe --product tf32 m n k` times one product), on both GPUs of the profiles
below. The median error is 1.8–2.4 % in FP64 (largest 6 %), 1.6–2.1 % in FP32 (largest 12 %) and 8–10 % in TF32,
where the vendor product of irregular shapes departs most from a smooth law (largest 32 %, for 3000 × 3000,
k = 3000).

## What the probe measures

| Quantity | Method |
| --- | --- |
| SM clock | cycles (`clock64`) per device nanosecond (`%globaltimer`) over a register loop, before and after the load; a capped or throttled GPU shows here |
| Access latency of shared memory, L1, L2, device memory | one thread chases a full-period cycle (`next[i] = (a·i + c) mod 2^k`) over a footprint swept from 4 KiB to 32 × L2; the plateaus are the latencies and the steps the capacities |
| Bandwidth of device memory: read, write, copy | 16-byte loads and stores, a contiguous chunk per block, grid swept from 1 block to 4 per SM: per-SM bandwidth, device bandwidth and the saturation point |
| Bandwidth of L2, L1, shared memory | the same loads over a footprint resident in the level |
| Arithmetic, scalar | chains of fused multiply-adds in registers: one chain for the latency, eight per thread on every warp for the rate, FP32 and FP64 |
| Arithmetic, products | cuBLAS `GemmEx` in FP64, FP32 (IEEE), TF32 and FP16 on the three shape families; call latency from a 32³ product |
| Launch, block, barrier, stream join | events around empty kernels (one and 256 in a row), the slope of an empty grid, `__syncthreads` loops at 32/256/1024 threads, fork and join of a second stream |
| Peer GPU and host links | `cudaMemcpyPeerAsync` and pinned host copies at three sizes, fitted to `α + β · bytes` |
| Concurrency | two compute-bound kernels on two streams, on half of the SMs each and on all of them |

Latencies are reported in seconds and in SM cycles. Published latencies are in cycles and are nearly constant
across NVIDIA generations (L1 ≈ 30–40, shared ≈ 29, L2 ≈ 200–280), so cycles are the unit in which a kernel
constant carries from one GPU to another.

## Measured profiles and published values

`profiles/h200.json` and `profiles/h200-1500mhz.json` are two H200 GPUs of the same node; the second one runs at a
capped SM clock, which makes it a second machine with the same memory and 0.76 of the arithmetic.

| | H200, this probe | H200 at 1.5 GHz, this probe | Published |
| --- | ---: | ---: | --- |
| SM clock | 1.978 GHz | 1.500 GHz | H800 1.755 GHz, falling to 1.2–1.3 GHz at its power limit [luo2025] |
| Shared memory latency | 28.9 cycles | 28.9 cycles | 29.0 (H800) [luo2025] |
| L1 latency | 39.6 cycles | 39.6 cycles | 32 (H800) [luo2025], 30–40 (H100) [jarmusch2025dissect] |
| L2 latency | 278 cycles | 246 cycles | 264.5 (H800) [luo2025], 273 (H100) [jarmusch2025dissect] |
| Device memory latency | 580 cycles, 293 ns | 480 cycles, 320 ns | 656 (H800, 374 ns) [luo2025], 658.7 (H100) [jarmusch2025dissect] |
| L2 capacity | 60 MiB reported, 32 MiB fully resident | same | 50 MB in two partitions (H100) |
| Device memory read | 4.49 TB/s | 4.48 TB/s | 4.8 TB/s peak, 4.38 TB/s STREAM (H200) [jarmusch2025] |
| Device memory write | 2.03 TB/s | 1.95 TB/s | 2.2 TB/s (H100) [jarmusch2025dissect] |
| FP64 product | 64.8 TFLOP/s | 49.1 TFLOP/s | 18.9 TFLOP/s DGEMM (H200) [jarmusch2025] |
| FP32 product (IEEE) | 50.9 TFLOP/s | 39.4 TFLOP/s | |
| TF32 product | 452 TFLOP/s | 343 TFLOP/s | 364 TFLOP/s `wgmma` (H800, 114 SMs) [luo2025] |
| FP16 product | 865 TFLOP/s | 653 TFLOP/s | 729 TFLOP/s `wgmma` (H800) [luo2025] |
| κ of FP64 / FP32 / TF32 | 51 / 36 / 319 | 45 / 34 / 284 | |
| FMA latency FP32 / FP64 | 4.4 / 8.2 cycles | 4.4 / 8.2 cycles | 4 / 8.04 (H100) [jarmusch2025dissect] |
| Launch, launch and wait, stream join | 2.0, 4.6, 10.8 µs | 2.0, 4.6, 11.0 µs | 2 µs launch, 3 µs synchronization [jarmusch2026model] |
| Block barrier (1,024 threads) | 124 ns, 246 cycles | 164 ns, 246 cycles | 40–50 cycles `mbarrier` (B200) [jarmusch2026model] |
| Peer GPU link | measured when the probe sees a second GPU; these profiles saw one | | |
| Host link | 5.1 µs + bytes / 55.7 GB/s | 5.3 µs + bytes / 55.8 GB/s | 45 GB/s default [jarmusch2026model] |

The latencies agree with the published ones to the cycle where the method is the same, and the latencies that
are a number of SM cycles are the same on both GPUs while the arithmetic follows the clock. The vendor FP64
product of this profile is 3.4 times the published H200 DGEMM figure, so only ratios are taken from that source.
`literature.json` holds these and the published values for B200, RTX 5080, A100, RTX 4090, V100 and MI300A, with
their sources. They are reference points for a new profile, not inputs: a schedule is always computed from the
profile measured on the GPU it runs on.

## Using a profile

```python
from machine import Machine, Tuner

m = Machine.load("machine/profiles/h200.json")
m.capacity("shared"), m.latency("l2"), m.bandwidth("memory", sms=32, kind="copy")
law = m.product("tf32")  # gamma, kappa, xi, call
law.time(m=65536, n=4096, k=1024)  # seconds of one rank-1024 update
m.balance("tf32"), m.grain("tf32")  # regime boundaries
m.summary()  # every basic metric, flat, SI units
```

`Tuner` selects the parameters of an algorithm the way an auto-tuner does, with named parameters, restrictions
and an objective, except that the objective is a cost model on the profile, so the whole space is ranked without
running anything. `refine` then measures only the few configurations the model ranks best.

```python
# Block size of a right-looking Cholesky factorization of order n, as an illustration of the interface.
def seconds(c, n=32768, a="fp64"):
    nb, law = c["nb"], m.product(a)
    t = 0.0
    for j in range(0, n, nb):
        rest = n - j - nb
        t += nb * 30 * (m.barrier(512) + m.latency("l2"))  # panel: a latency chain per column
        t += (
            law.time(rest, nb, nb) + law.time(rest, rest, nb) / 2
        )  # triangular solve and symmetric update
    return t


tuner = Tuner(
    {"nb": [64, 128, 256, 512, 1024, 2048]},
    seconds,
    restrictions=[lambda c: 8 * c["nb"] ** 2 <= m.capacity("l2")],
)
predicted, best = tuner.best()
# measured, best, table = tuner.refine(run_my_cholesky, top=3, cache="cholesky-h200.json")
```

## Projections

`Machine.scaled(name, clock=, arithmetic=, memory_bandwidth=, sms=)` multiplies the measured quantities of a
profile by published or hypothetical ratios. It answers "what would be selected on such a GPU" before one is
available; it is not a measurement, and the engine's kernels still have to exist for that architecture.

```python
b200 = h200.scaled(
    "B200 (projected)",
    sms=148,
    memory_bandwidth=1.71,  # ratios of [jarmusch2025]
    arithmetic={"fp64": 1.92, "fp32": 1.27, "tf32": 1.27, "fp16": 1.27},
)
```

On this projection the product law moves the way the ratios say (FP64: 124 TFLOP/s and κ 58 against 65 and 51;
TF32: 574 TFLOP/s and κ 237 against 452 and 319), and so does the QR-Ω schedule: a faster FP64 update leaves the
panel more SMs (G 32 → 48 at n = 16,384), and a TF32 update that is less bound by bandwidth needs half the
aggregation (a 8 → 4 at n = 65,536).

## Profile format

`schema: machine-profile/1`. Top-level keys: `device` (name, UUID, SM count, register file, shared memory, L2,
memory, measured clock), `node` (GPU count, peer access, CPU), `latency`, `access` (per level, with the raw
footprint sweep), `bandwidth` (per level and link, with the SM scaling curve), `arithmetic.simt`,
`arithmetic.gemm` (per arithmetic: samples of the three shape families and the rank-k fit), `concurrency`.
Every derived value is stored next to the samples it was derived from.
