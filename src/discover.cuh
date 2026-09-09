#pragma once
// discover.cuh -- populate a tqr::machine by interrogating and measuring the hardware
// it is running on. Single node, multi-GPU NVIDIA.
//
// The hierarchy discovered here has five regions and four boundaries:
//
//   region 0  "registers"  one warp's share of an SM's register file
//   region 1  "smem/L1"    one SM's shared-memory carve-out
//   region 2  "L2"         one GPU's L2
//   region 3  "hbm"        one GPU's global memory
//   region 4  "host ram"   the node's physical memory
//
//   bnd 0  registers <-> smem   P_0 = warps per SM
//   bnd 1  smem      <-> L2     P_1 = SMs per GPU
//   bnd 2  L2        <-> hbm    P_2 = 1
//   bnd 3  hbm       <-> host   P_3 = GPUs in the node
//
// NO HARDCODED HARDWARE VALUES. Capacities and peer counts come from CUDA device
// attributes; every bandwidth and every latency is measured on the spot. The only
// literals here are measurement parameters -- how large a buffer to stream, how many
// times to repeat a timing, how far apart to space a pointer chase -- and each is
// named in discovery_options and justified where it is used. Not one number
// describing this machine's speed or size is written down; they are asked for or timed.
//
// The file is in two halves, and the split is deliberate:
//
//   assemble()  pure host arithmetic. Turns a discovery_report into a machine: bytes
//               to words, aggregate throughput to per-domain bandwidth, the modelling
//               choices below. Compiles and runs without a GPU, so it can be tested
//               against numbers you supply by hand.
//   discover()  the probes. Queries the device attributes and measures everything
//               else, fills a report, and hands it to assemble(). Requires nvcc.
//
// Two modelling choices are made explicitly rather than silently, because neither is
// forced by the hardware. Both live in assemble(), flagged at the point of use:
//
//   (a) The level-0 domain is a WARP. The register file is physically partitioned per
//       thread, so M_0 * P_0 is exactly the SM's register file and no capacity is
//       invented or lost. Lowering occupancy trades P_0 for M_0 at constant product,
//       which is a scheduling degree of freedom the theory is entitled to optimize;
//       full occupancy is the partition reported here. A warpgroup-level (sm_90
//       wgmma) reading would divide P_0 by 4 and multiply M_0 by 4.
//
//   (b) B_3 is the bandwidth one GPU sustains LEAVING ITSELF, taken over the peer path
//       when peers exist and over the host path otherwise. Definition 2.1 folds peer
//       traffic and parent traffic into the single boundary 3, and Theorem 10.3 says a
//       resident boundary -- which is where the 2.5D layer lives, and where a single
//       node normally operates -- carries all-peer-to-peer traffic. Both paths are
//       measured and both are reported; only the governing one reaches the model.
//
// Every B_l is stated PER LEVEL-l DOMAIN, as Definition 2.1 requires: aggregate
// throughput divided by the number of level-l domains contending for it. That is what
// makes Q_l/B_l in the time model (1) dimensionally right, since Q_l is counted at one
// (maximally loaded) domain.

#include "machine.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
// Opt-in: build with -DTQR_WITH_NCCL and link -lnccl to measure boundary 3's peer path
// with the collective library the schedule will actually call. alpha_3 is the exposed
// latency of one dependency-bearing round; if those rounds are NCCL collectives then
// alpha_3 must be NCCL's latency, not cudaMemcpyPeer's. Measured on 4x H200: memcpyPeer
// round 6.7 us, NCCL 2.29.2 AllReduce 8.9 us. The low-latency kernels of arXiv 2607.16100
// reach 2.37 us, but they are a device-side API (NCCL_DEVICE_LL_BUFFER_*) requiring
// custom collective kernels, not an NCCL_ALGO selection -- see the note in
// nccl_peer_probe below.
#if defined(TQR_WITH_NCCL)
#include <nccl.h>
#include <nccl_device.h>
#include <atomic>
#include <thread>
// Requires --extended-lambda: ncclLLA2ASession::recvReduce takes device lambdas.
#endif
#if defined(TQR_WITH_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unistd.h>       // usleep, for the bootstrap-file handshake
#endif
#if defined(__linux__)
#include <unistd.h>
#endif
#endif

namespace tqr {

// ---------------------------------------------------------------------------------
// Options. Every field is a measurement parameter -- a statement about how hard to
// probe, never a statement about the hardware.
// ---------------------------------------------------------------------------------
struct discovery_options {
    // Bytes per matrix element. 8 = fp64, 4 = fp32. Sets machine::word_bytes and is
    // the divisor that turns every measured byte figure into the model's words.
    int word_bytes = 8;

    // Which devices form the machine. count = 0 means "every visible device".
    int first_device = 0;
    int device_count = 0;

    // Every probe is repeated and the MEDIAN is kept, with the min-max spread recorded
    // alongside it. Median rather than best-of: a best-of estimator reports the luckiest
    // sample, which hides instability instead of exposing it, and an earlier best-of
    // version of this file did exactly that. The spread in the report is what tells you
    // whether a number is trustworthy.
    int repeats = 7;          // window size: how many recent samples must agree
    int warmups = 2;
    int max_repeats = 40;     // give up sampling after this many

    // A probe stops as soon as its last `repeats` samples agree to within this. If it
    // never does, the number is still reported but rep.converged is false and the
    // offending probe is named in rep.warning -- an unconverged number is not silently
    // passed off as a measurement.
    double target_spread = 0.05;

    // Streaming buffers are sized relative to the capacity being characterised, so
    // they scale with the machine instead of assuming one.
    //   - HBM: the buffer must not fit in L2, so it is `hbm_l2_multiple` x L2, capped
    //     at `max_free_fraction` of free device memory.
    //   - L2:  the working set must fit in L2 with room to spare, so src and dst are
    //     each L2 / `l2_working_divisor`.
    int    hbm_l2_multiple    = 32;
    int    l2_working_divisor = 8;
    double max_free_fraction  = 0.25;

    // Pointer-chase geometry. Each chase makes one FULL traversal of a random cycle, so
    // every line is touched exactly once before any is revisited, and the footprint is
    // exactly nodes x stride. `chase_stride_bytes` must exceed a cache line so that
    // consecutive hops land on distinct lines.
    //
    // The footprint is what decides which level answers, and the two levels want
    // opposite things:
    //   far  (HBM): footprint = chase_far_l2_multiple x L2, so every hop misses. 2 is
    //               enough on measured hardware and 4 leaves margin for an L2 that
    //               retains more of the set between launches. Measured identical at 2x,
    //               4x and 8x on an idle device, so the choice costs accuracy nothing
    //               and runtime everything -- the far chase is the long pole here.
    //   near (L2):  footprint = L2 / chase_near_l2_divisor, and the chase runs one
    //               UNTIMED warm lap first. Without it the probe is wrong: L2 is cold
    //               at kernel launch and a single traversal touching each line once
    //               would miss on every hop and report the memory latency again.
    int chase_stride_bytes    = 128;
    int chase_far_l2_multiple = 4;
    int chase_near_l2_divisor = 4;

    // Latency probes walk hundreds of MiB of dependent misses and dominate the runtime,
    // so they repeat fewer times. They can afford to: with in-kernel clock64 timing and
    // a full traversal the spread is around 0.1%, against several percent for bandwidth.
    int latency_repeats = 5;
    int latency_warmups = 1;
    int latency_max_repeats = 24;

    // L2 partition discovery. Hopper splits its L2 into partitions behind a crossbar,
    // and a chase spread over both reports the average of a near-hit and a far-hit --
    // a number that describes no actual access. The probe runs one independent chase
    // per SM over its own disjoint region (total footprint L2/l2_probe_divisor, so
    // every chase is an L2 hit) and clusters the resulting per-SM latencies. Two
    // clusters mean two partitions. `mode_gap` is the relative gap, as a fraction of
    // the median, that separates one cluster from the next; `mode_floor` is the share
    // of samples a cluster needs before it counts as real rather than an outlier.
    int    l2_probe_divisor = 4;
    double mode_gap   = 0.15;      // retained for the old gap test; unused by the GMM
    double mode_floor = 0.10;

    // DGNA's stride rule: step = max(DRAM-to-L2 transaction size, L1 line), ~64 B on
    // recent NVIDIA parts. Nodes sit at random multiples of it rather than on a fixed
    // grid, because a uniform stride can alias onto one partition's address bits and
    // hide the very split we are testing for.
    int l2_line_bytes = 64;

    // How far apart two mixture means must be, in pooled standard deviations, before
    // they count as separate partitions rather than two lobes of one gradient. 3 keeps
    // the A100-style split (385 vs 546 cycles) while rejecting the L40-style continuum.
    double l2_mode_separation_sigmas = 3.0;

    // Force the partition count instead of discovering it. 0 = discover. Provided
    // because the probe reports what it can see, and on measured H200 silicon it sees
    // ONE mode on both axes: all SMs against one shared region give 274-307 cycles in a
    // smooth gradient, and 256 addresses swept across 512 MiB give 287-368, largest gap
    // 2.3% -- no step, no second cluster, with L1 bypassed in both cases. Published
    // near-hit figures for Hopper (~258-264 cycles) agree with the single mode measured
    // here (~285); the companion "up to 743" is a far-hit/MISS, and an actual miss
    // measures 690 cycles on this part, so it is the miss that number describes. Set
    // this if you have better information about a device than the probe can extract.
    int l2_partitions_override = 0;

    // Small transfers per link-latency sample. Each one is synchronized individually --
    // that is the point, since pipelined copies would measure bandwidth rather than the
    // exposed cost of a round something waits on.
    int link_rounds = 200;

    // Sweep every peer pair for the one-sided latency rather than trusting PE 0 <-> PE 1.
    // Costs np*(np-1)/2 ping-pongs; off falls back to the single pair.
    bool   sweep_peer_pairs = true;
    // Relative spread below which the fabric is reported as uniform.
    double pair_uniform_tol = 0.10;

    // Rank and size when the run is multi-process. -1 means "read them from the
    // environment" (SLURM_PROCID/SLURM_NTASKS, then OMPI_COMM_WORLD_*), falling back to
    // 0/1 for a single-process run.
    //
    // NVSHMEM requires one process per GPU, so the one-sided leg of a peer boundary can
    // only be probed when the whole discovery is multi-process. Built with
    // TQR_WITH_NVSHMEM and launched under `srun -n <gpus>`, discover() initialises
    // NVSHMEM itself and measures that leg like any other; there is no path for feeding
    // the numbers in by hand, because a number nobody measured is not a measurement.
    //
    // Both legs matter and they are not interchangeable: alpha is technology dependent,
    // and which moves need which is fixed by Definition 4.3 rather than by preference --
    // Combine(a)'s ordered stack-Householder merge cannot be a collective at all.
    int rank   = -1;
    int nranks = -1;

    // Where rank 0 leaves the NVSHMEM/NCCL bootstrap ids for the other ranks.
    const char* bootstrap_file = "/tmp/tqr_discover_bootstrap";

    // Link probing is the slow part of discovery; turn it off for an on-device-only
    // machine. With it off there is nothing to fill boundary 3 with, so discover()
    // fails rather than guessing -- ask for it, or supply the numbers to assemble().
    bool measure_links = true;

    // The model needs a uniform branching factor, so a mixed-GPU node is rejected
    // unless this is set, in which case device `first_device` speaks for all of them.
    bool allow_heterogeneous = false;

    bool verbose = false;
};

// ---------------------------------------------------------------------------------
// Report. Carries the raw measurements, their provenance and the cross-checks, so
// that nothing that went into the machine is hidden behind a single number. It is
// also the input to assemble(), which is how a machine can be built from figures
// obtained some other way.
// ---------------------------------------------------------------------------------
struct discovery_report {
    char machine_name[288] = {};      // machine::name points here; must outlive it
    char device_name[256]  = {};      // cudaDeviceProp::name is char[256]
    char error[256]        = {};      // empty on success
    int  cuda_status       = 0;       // cudaError_t of the first failing call

    int  devices       = 0;           // P_3
    int  sm_count      = 0;           // P_1
    int  warps_per_sm  = 0;           // P_0
    int  warp_size     = 0;
    int  cc_major = 0, cc_minor = 0;

    // Queried capacities, in bytes, before conversion to words.
    size_t regfile_bytes_per_sm = 0;
    size_t smem_bytes_per_sm    = 0;
    size_t l2_bytes             = 0;
    size_t hbm_bytes            = 0;
    size_t host_ram_bytes       = 0;

    // Measured aggregate throughput, bytes/s, before division by the peer count.
    double smem_bw_aggregate = 0.0;   // whole GPU
    double l2_bw_aggregate   = 0.0;   // whole GPU
    double hbm_bw            = 0.0;   // whole GPU
    double peer_bw_out       = 0.0;   // one GPU to all its peers at once
    double host_bw           = 0.0;   // one GPU to pinned host memory

    // (max - min) / median across repeats, for every number above and below. This is
    // the honest error bar on the machine: a boundary whose spread is large has a
    // meaningless Lhat and therefore a meaningless optimal tile size.
    double smem_bw_spread = 0.0, l2_bw_spread = 0.0, hbm_bw_spread = 0.0;
    double peer_bw_spread = 0.0, host_bw_spread = 0.0;
    double smem_lat_spread = 0.0, l2_lat_spread = 0.0, hbm_lat_spread = 0.0;
    double peer_lat_spread = 0.0, host_lat_spread = 0.0;

    // SM clock measured during the far chase, in Hz: in-kernel cycles divided by the
    // wall time of a kernel long enough (about a second) that launch overhead is parts
    // per million. Latencies are counted in cycles and converted with this, so they do
    // not inherit host-side timing jitter. Compare it with the advertised boost clock.
    double sm_clock_hz = 0.0;

    // Latency in cycles, which is the quantity the hardware actually holds fixed.
    double smem_lat_cycles = 0.0, l2_lat_cycles = 0.0, hbm_lat_cycles = 0.0;

    // Measured dependent-access latency, seconds.
    double smem_lat = 0.0;
    double l2_lat   = 0.0;
    double hbm_lat  = 0.0;
    double peer_lat = 0.0;
    double host_lat = 0.0;

    // L2 partitioning, discovered by clustering per-SM chase latency. partitions = 1
    // means the probe found a single latency mode, so L2 is modelled as one region.
    int    l2_partitions = 1;
    double l2_near_cycles = 0.0;   // fastest per-SM sample
    double l2_far_cycles  = 0.0;   // slowest per-SM sample
    double l2_median_cycles = 0.0;
    int    l2_probe_blocks = 0;
    int    l2_gmm_components = 0;  // mixture components the BIC selected
    double l2_mode_mu[4] = {0};    // their means, sorted

    // Distinct bytes each pointer chase touched. These are the evidence that the two
    // latencies mean what they say: far must exceed L2, near must not.
    size_t chase_far_footprint  = 0;
    size_t chase_near_footprint = 0;

    // Cross-check only, never used to fill the machine: the bus-width x clock product
    // the device advertises. hbm_bw / hbm_bw_theoretical is the fraction of peak the
    // streaming kernel actually sustained; a low one means the measured number, not
    // the model, is what should be doubted.
    double hbm_bw_theoretical = 0.0;

    // False if any probe failed to reach discovery_options::target_spread; `warning`
    // names the ones that did not. A machine built from an unconverged probe has an
    // Lhat, and therefore an optimal tile size, that means nothing.
    bool converged = true;
    char warning[256] = {};

    // Collective leg, measured in-kernel (device-initiated) rather than through the host
    // API. The host API costs 8.8 us per small AllReduce here against 1.14 us in-kernel,
    // and the in-kernel figure is what alpha_l means: Combine(+) happens inside
    // Algorithm 1's fused sweep, not as a separately launched collective.
    double peer_lat_device = 0.0;     // seconds, device-initiated one-shot LL AllReduce
    double peer_lat_host   = 0.0;     // seconds, ncclAllReduce host API, for comparison
    bool   peer_device_initiated = false;
    bool   peer_bw_symmetric = false;   // B_3 measured over symmetric-memory windows

    // Speed-of-light decomposition, arXiv 2607.16100 section VI. Lets alpha_3 be judged
    // against this machine's own floor instead of a number from other silicon.
    double sol_l2_rtt = 0.0;        // seconds, one __threadfence
    double sol_ping_pong = 0.0;     // seconds, a value bounced between two GPUs
    double sol_remote_store = 0.0;  // (ping_pong - 2*l2_rtt)/2
    double sol_allreduce = 0.0;     // remote_store + 2*l2_rtt

    // One-sided leg, measured with NVSHMEM when the run is multi-process.
    double onesided_bw       = 0.0;   // bytes/s
    double onesided_lat      = 0.0;   // seconds, half a ping-pong round trip
    double onesided_put_lat  = 0.0;   // seconds, put + quiet, for comparison
    bool   onesided_measured = false;

    // Every ordered pair, not just PE 0 <-> PE 1. On an all-to-all fabric these agree and
    // the sweep costs a few hundred milliseconds; on anything with non-uniform peer
    // distance -- multiple baseboards, a partly-populated switch, a rail-optimised
    // topology -- they do not, and using one pair's number for alpha_3 would understate
    // the boundary by whatever the worst pair costs.
    //
    // alpha_3 takes the WORST pair, because alpha multiplies Scrit in the time model (1)
    // and Scrit is a longest-directed-chain count (Theorem 11.4): a chain is only as fast
    // as the slowest hop it must take. The mean would describe a schedule nobody runs.
    static constexpr int kMaxPeers = 16;
    double onesided_pair_lat[kMaxPeers * kMaxPeers] = {};   // seconds, [i*np + j]
    double onesided_lat_min = 0.0, onesided_lat_max = 0.0, onesided_lat_med = 0.0;
    bool   onesided_pairs_swept = false;
    bool   onesided_uniform = false;    // max within `pair_uniform_tol` of min

    int  rank = 0, nranks = 1;
    bool multi_process = false;

    bool peer_via_nccl  = false;      // peer numbers came from NCCL, not cudaMemcpyPeer
    bool p2p_all_pairs  = false;      // every ordered pair had peer access
    bool links_measured = false;
    bool heterogeneous  = false;
    bool used_peer_path = false;      // which path governs boundary 3; see choice (b)

    bool ok() const { return error[0] == '\0'; }
    void print(FILE* f = stdout) const;
};

namespace detail {

// A converging estimator. Probes push per-repeat results in and it reports the median
// of the most recent `window` samples, together with their spread -- but only once that
// window agrees to within `target`. A fixed repeat count cannot do this: a probe whose
// samples are bimodal returns the median of two different states and looks precise
// while being wrong. That is not hypothetical: a co-tenant process on the same device
// evicts the near chase's working set, and the L2 latency samples then straddle the hit
// and miss values, giving a plausible number that is neither. Sampling until a window
// agrees, and naming the probe when it never does, is what separates a measurement from
// a number that merely reproduced.
struct converger {
    static constexpr int kMax = 128;
    double v[kMax] = {};
    int    n = 0;
    int    window = 5;
    double target = 0.05;

    void add(double x) { if (x > 0.0 && n < kMax) v[n++] = x; }
    bool  full() const { return n >= kMax; }
    bool  ok() const   { return n > 0; }

    // The last `window` samples (or all of them, if fewer), sorted into `out`.
    int tail(double* out) const {
        const int k = n < window ? n : window;
        for (int i = 0; i < k; ++i) out[i] = v[n - k + i];
        std::sort(out, out + k);
        return k;
    }
    double median() const {
        double t[kMax]; const int k = tail(t);
        if (!k) return 0.0;
        return (k & 1) ? t[k / 2] : 0.5 * (t[k / 2 - 1] + t[k / 2]);
    }
    double spread() const {
        double t[kMax]; const int k = tail(t);
        const double m = median();
        return (k > 1 && m > 0.0) ? (t[k - 1] - t[0]) / m : 0.0;
    }
    // True once a full window agrees to within `target`.
    bool settled() const { return n >= window && spread() <= target; }
};

// One-dimensional Gaussian mixture by EM, with k chosen by BIC. This replaces a
// gap-threshold heuristic because the partition count is data, not a constant: DGNA finds
// two well-separated components on A100 (385 and 546 cycles) while the L40 study finds a
// single continuous gradient with no discrete gap at all. Both shapes are real, so the
// probe has to be able to return either.
//
// Returns the selected component count and fills `mu` with the sorted means.
inline int gmm_components(const double* x, int n, int kmax, double* mu, double* wt,
                          double sep_sigmas) {
    if (n < 8) return 1;
    double best_bic = 1e300;
    int best_k = 1;
    double best_mu[8] = {0}, best_w[8] = {0}, best_v[8] = {0};
    if (kmax > 8) kmax = 8;

    for (int k = 1; k <= kmax; ++k) {
        double m[8], v[8], w[8];
        double lo = x[0], hi = x[0];
        for (int i = 1; i < n; ++i) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
        if (!(hi > lo)) return 1;
        for (int j = 0; j < k; ++j) {
            m[j] = lo + (hi - lo) * (j + 0.5) / k;
            v[j] = (hi - lo) * (hi - lo) / (4.0 * k * k) + 1e-9;
            w[j] = 1.0 / k;
        }
        double* resp = (double*)std::malloc((size_t)n * k * sizeof(double));
        if (!resp) return 1;
        for (int it = 0; it < 200; ++it) {
            for (int i = 0; i < n; ++i) {                       // E step
                double tot = 0.0;
                for (int j = 0; j < k; ++j) {
                    const double d = x[i] - m[j];
                    double pj = w[j] * std::exp(-0.5 * d * d / v[j]) / std::sqrt(v[j]);
                    if (!(pj > 0.0)) pj = 1e-300;
                    resp[(size_t)i * k + j] = pj; tot += pj;
                }
                for (int j = 0; j < k; ++j) resp[(size_t)i * k + j] /= tot;
            }
            for (int j = 0; j < k; ++j) {                       // M step
                double sw = 0, sm_ = 0, sv = 0;
                for (int i = 0; i < n; ++i) { const double r = resp[(size_t)i*k+j]; sw += r; sm_ += r*x[i]; }
                if (sw < 1e-12) { w[j] = 1e-12; continue; }
                m[j] = sm_ / sw;
                for (int i = 0; i < n; ++i) {
                    const double d = x[i] - m[j];
                    sv += resp[(size_t)i*k+j] * d * d;
                }
                v[j] = sv / sw + 1e-9;
                w[j] = sw / n;
            }
        }
        double ll = 0.0;
        for (int i = 0; i < n; ++i) {
            double tot = 0.0;
            for (int j = 0; j < k; ++j) {
                const double d = x[i] - m[j];
                tot += w[j] * std::exp(-0.5*d*d/v[j]) / std::sqrt(2.0*3.14159265358979*v[j]);
            }
            ll += std::log(tot > 0.0 ? tot : 1e-300);
        }
        std::free(resp);
        const double params = 3.0 * k - 1.0;
        const double bic = -2.0 * ll + params * std::log((double)n);
        if (bic < best_bic) {
            best_bic = bic; best_k = k;
            for (int j = 0; j < k; ++j) { best_mu[j] = m[j]; best_w[j] = w[j]; best_v[j] = v[j]; }
        }
    }

    // BIC will happily split a continuous gradient into several overlapping components,
    // so a separation test decides whether the split is real. It must be measured against
    // the components' OWN spread, not the spread of all the data: a genuine bimodal
    // sample has a large overall sigma precisely BECAUSE it is bimodal, so testing the
    // gap against the overall sigma makes a real split defeat its own test. (An earlier
    // version did exactly that and reported one component for DGNA's A100 shape.)
    const int kk = best_k < 8 ? best_k : 8;
    for (int i = 1; i < kk; ++i) {                 // insertion sort, carrying v and w
        const double km = best_mu[i], kv = best_v[i], kw = best_w[i];
        int j = i - 1;
        while (j >= 0 && best_mu[j] > km) {
            best_mu[j+1] = best_mu[j]; best_v[j+1] = best_v[j]; best_w[j+1] = best_w[j];
            --j;
        }
        best_mu[j+1] = km; best_v[j+1] = kv; best_w[j+1] = kw;
    }
    // Two conditions, both required, because either alone is fooled:
    //   (1) means separated by sep_sigmas of the components' OWN pooled sigma;
    //   (2) an actual density trough between them.
    // (2) is what distinguishes a partition from a gradient. A GMM will cheerfully fit
    // two Gaussians to a uniform ramp, and those halves pass (1) -- but a ramp has no
    // trough, while a genuinely partitioned cache has almost no samples at the midpoint.
    // This is Hartigan's dip idea reduced to the one comparison we need.
    int kept = 1;
    for (int j = 1; j < kk; ++j) {
        const double within = std::sqrt(0.5 * (best_v[j-1] + best_v[j]));
        if (!(within > 0.0)) continue;
        if ((best_mu[j] - best_mu[j-1]) <= sep_sigmas * within) continue;

        const double mid = 0.5 * (best_mu[j-1] + best_mu[j]);
        const double hw  = 0.25 * (best_mu[j] - best_mu[j-1]);
        int c_mid = 0, c_lo = 0, c_hi = 0;
        for (int i = 0; i < n; ++i) {
            if (x[i] > mid - hw          && x[i] < mid + hw)          ++c_mid;
            if (x[i] > best_mu[j-1] - hw && x[i] < best_mu[j-1] + hw) ++c_lo;
            if (x[i] > best_mu[j]   - hw && x[i] < best_mu[j]   + hw) ++c_hi;
        }
        const int peak = c_lo < c_hi ? c_lo : c_hi;
        if (c_mid * 2 < peak) ++kept;          // a real trough: under half the peak density
    }
    for (int j = 0; j < kk; ++j) { mu[j] = best_mu[j]; wt[j] = best_w[j]; }
    return kept;
}

inline bool report_error(discovery_report& r, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(r.error, sizeof(r.error), fmt, ap);
    va_end(ap);
    return false;
}

// Records a converged result, or flags the probe by name when it never settled.
inline void take(discovery_report& r, const converger& c, const char* what,
                 double* value, double* spread) {
    if (!c.ok()) return;
    *value  = c.median();
    *spread = c.spread();
    if (!c.settled()) {
        r.converged = false;
        const size_t k = std::strlen(r.warning);
        std::snprintf(r.warning + k, sizeof(r.warning) - k, "%s%s +-%.0f%%",
                      k ? ", " : "", what, c.spread() * 100.0);
    }
}

}  // namespace detail

// ---------------------------------------------------------------------------------
// assemble -- the modelling half. Pure host arithmetic, no CUDA.
// ---------------------------------------------------------------------------------
// Turns a filled discovery_report into a machine. Every conversion that happens here
// is one of: bytes -> words, aggregate throughput -> per-domain bandwidth, or one of
// the two modelling choices declared at the top of this file. Nothing is measured and
// nothing is guessed; supply a report by hand and this will build the corresponding
// machine, which is how the arithmetic gets tested without a GPU.
//
// `rep` must outlive `out`: machine::name points at rep.machine_name.
inline bool assemble(machine& out, discovery_report& rep,
                     const discovery_options& opt = discovery_options()) {
    if (opt.word_bytes < 1)
        return detail::report_error(rep, "word_bytes must be at least 1");

    // Peer counts must all be present: they are the shape of the hierarchy, and a
    // zero anywhere would silently collapse a level.
    if (rep.warps_per_sm < 1 || rep.sm_count < 1 || rep.devices < 1)
        return detail::report_error(rep,
            "incomplete peer counts (warps/SM %d, SMs %d, devices %d)",
            rep.warps_per_sm, rep.sm_count, rep.devices);

    if (rep.regfile_bytes_per_sm == 0 || rep.smem_bytes_per_sm == 0 ||
        rep.l2_bytes == 0 || rep.hbm_bytes == 0 || rep.host_ram_bytes == 0)
        return detail::report_error(rep, "a region capacity is missing");

    // Choice (b), decided here so that the report records which way it went.
    rep.used_peer_path = (rep.devices > 1) && (rep.peer_bw_out > 0.0);
    const double b3_bytes = rep.used_peer_path ? rep.peer_bw_out : rep.host_bw;
    const double a3       = rep.used_peer_path ? rep.peer_lat    : rep.host_lat;

    if (!(rep.smem_bw_aggregate > 0.0) || !(rep.l2_bw_aggregate > 0.0) ||
        !(rep.hbm_bw > 0.0) || !(b3_bytes > 0.0))
        return detail::report_error(rep,
            "a bandwidth is missing (smem %.3g, L2 %.3g, HBM %.3g, out %.3g B/s)",
            rep.smem_bw_aggregate, rep.l2_bw_aggregate, rep.hbm_bw, b3_bytes);

    const double wb    = (double)opt.word_bytes;
    const int    warps = rep.warps_per_sm;

    // L2 partitioning, from the probe. A partitioned L2 is not one region of 60 MiB but
    // `parts` peer regions of 60/parts MiB each, and the SMs divide between them: domains
    // are nested, so one level-3 domain (HBM) holds P_2 = parts level-2 domains (L2
    // partitions), each of which holds P_1 = SMs/parts level-1 domains (SMs). The
    // aggregate P_2*M_2 is unchanged at 60 MiB and Pbar_1 is unchanged at the SM count;
    // what changes is that boundary 2 acquires a real collective depth D_2 = 1 where the
    // unified reading had D_2 = 0, and alpha_1 becomes the near-hit rather than an
    // average over near and far.
    //
    // The nesting is an idealisation: the crossbar lets any SM reach either partition,
    // so "SMs per partition" is a modelling choice, not a wiring diagram. It is the one
    // the pebble game needs, and the far-hit latency below is what it costs to leave it.
    const int parts = rep.l2_partitions > 0 ? rep.l2_partitions : 1;
    if (parts > 1 && rep.sm_count % parts != 0)
        return detail::report_error(rep,
            "%d SMs do not divide evenly into %d L2 partitions", rep.sm_count, parts);
    const int sms = rep.sm_count / parts;      // P_1: SMs sharing one L2 partition

    // Capacities. Choice (a): the level-0 domain is a warp, so the SM's register file
    // divides exactly into P_0 shares.
    const double m0 = (double)rep.regfile_bytes_per_sm / (double)warps / wb;
    const double m1 = (double)rep.smem_bytes_per_sm / wb;
    const double m2 = (double)rep.l2_bytes / (double)parts / wb;   // one partition
    const double m3 = (double)rep.hbm_bytes / wb;
    const double m4 = (double)rep.host_ram_bytes / wb;

    // Bandwidths, per level-l domain: aggregate throughput divided by the number of
    // level-l domains that contend for it.
    const double b0 = rep.smem_bw_aggregate / wb / ((double)sms * (double)warps);
    // Aggregate L2 bandwidth is shared by every SM on the GPU, so the per-domain share
    // divides by the total SM count, not by the SMs on one partition.
    const double b1 = rep.l2_bw_aggregate / wb / (double)rep.sm_count;
    const double b2 = rep.hbm_bw / wb;          // P_2 = 1; one L2 per GPU
    const double b3 = b3_bytes / wb;

    machine m;
    if (rep.machine_name[0] == '\0')
        std::snprintf(rep.machine_name, sizeof(rep.machine_name), "%dx %s, single node",
                      rep.devices, rep.device_name[0] ? rep.device_name : "nvidia gpu");
    m.name       = rep.machine_name;
    m.word_bytes = opt.word_bytes;

    // r_l, the reduction fan-in: pairwise combine is the only tree shape the move
    // alphabet guarantees at every boundary, so D_l = ceil(log_2 P_l) is what the
    // schedule pays. Raise it per boundary if a wider hardware collective is available.
    m.root({"registers", m0, machine::kRegisterFile});

    // Boundary 0: a warp reaching its SM's shared memory.
    { machine::boundary b; b.peers = warps; b.bandwidth = b0; b.latency = rep.smem_lat;
      m.attach(b, {"smem/L1", m1, machine::kScratchpad}); }

    // Boundary 1: an SM reaching its own L2 partition. With the L2 split, alpha_1 is the
    // near-hit -- the honest cost of the access this boundary actually describes -- and
    // the far-hit belongs to boundary 2's peer path below.
    { machine::boundary b; b.peers = sms; b.bandwidth = b1;
      b.latency = (parts > 1 && rep.sm_clock_hz > 0.0 && rep.l2_near_cycles > 0.0)
                ? rep.l2_near_cycles / rep.sm_clock_hz : rep.l2_lat;
      m.attach(b, {"L2", m2, machine::kCache}); }

    // Boundary 2: L2 to HBM. P_2 is the partition count, so a partitioned L2 carries a
    // depth-1 collective where a unified one carries none. Its PEER path is the
    // crossbar: reaching the other partition costs the far-hit latency. The crossbar's
    // bandwidth is not measured, so peer_bandwidth stays zero and B falls back to the
    // parent while alpha does not -- see boundary::alpha.
    { machine::boundary b; b.peers = parts; b.bandwidth = b2; b.latency = rep.hbm_lat;
      if (parts > 1 && rep.sm_clock_hz > 0.0 && rep.l2_far_cycles > 0.0)
          b.peer_latency = rep.l2_far_cycles / rep.sm_clock_hz;
      m.attach(b, {"hbm", m3, machine::kDeviceMemory}); }

    // Boundary 3: out of the GPU. The parent path is the host link; the peer path is the
    // inter-GPU fabric, which bypasses host memory entirely and is therefore a genuinely
    // separate transport rather than the same one measured twice.
    { machine::boundary b; b.peers = rep.devices; b.bandwidth = b3; b.latency = a3;
      if (rep.devices > 1 && rep.peer_bw_out > 0.0) {
          b.peer_bandwidth = rep.peer_bw_out / wb;
          b.peer_latency   = rep.peer_lat;
          // An NVSwitch-connected set is not restricted to a binary combine tree: the
          // fabric reaches every peer in one stage, so a collective over P_l ranks has
          // depth 1, not ceil(log2 P_l). Since Scrit = K + D - 1 (Theorem 11.4) that is
          // a directly visible latency term, and it is discovered from p2p reachability
          // rather than assumed.
          if (rep.p2p_all_pairs) b.peer_fanin = rep.devices;
      }
      if (rep.onesided_bw > 0.0 && rep.onesided_lat > 0.0) {
          b.onesided_bandwidth = rep.onesided_bw / wb;
          b.onesided_latency   = rep.onesided_lat;
      }
      m.attach(b, {"host ram", m4, machine::kHostMemory}); }

    if (const char* why = m.validate())
        return detail::report_error(rep, "assembled machine is invalid: %s", why);

    out = m;
    return true;
}

// =================================================================================
// discover -- the measurement half. Requires nvcc.
// =================================================================================
#if defined(__CUDACC__)

namespace detail {

#define TQR_CU(expr)                                                                 \
    do {                                                                             \
        cudaError_t e_ = (expr);                                                      \
        if (e_ != cudaSuccess) {                                                      \
            rep.cuda_status = (int)e_;                                                \
            return detail::report_error(rep, "%s: %s", #expr, cudaGetErrorString(e_)); \
        }                                                                            \
    } while (0)

inline int attr(cudaDeviceAttr a, int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, a, dev);
    return v;
}

// Largest power of two not exceeding v. Used to size shared-memory arrays so their
// index arithmetic can wrap with a mask instead of a modulo, which would otherwise
// dominate the very loop we are trying to time.
inline size_t pow2_floor(size_t v) {
    size_t p = 1;
    while ((p << 1) != 0 && (p << 1) <= v) p <<= 1;
    return p;
}

struct timer {
    cudaEvent_t a{}, b{};
    timer()  { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    timer(const timer&) = delete;
    timer& operator=(const timer&) = delete;
    void start(cudaStream_t s = 0) { cudaEventRecord(a, s); }
    double stop(cudaStream_t s = 0) {              // seconds
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        return (double)ms * 1e-3;
    }
};

// Streaming copy, 16 bytes per element. Counts 2 bytes of traffic per byte copied
// (one read, one write) -- the shape of a trailing update, not of a pure read.
static __global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst,
                              size_t n, int passes) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int p = 0; p < passes; ++p)
        for (size_t i = i0; i < n; i += stride) dst[i] = src[i];
}

// Shared-memory read bandwidth. Four independent loads per iteration so the loop is
// throughput-bound rather than latency-bound; the mask wrap keeps addresses inside the
// array without a modulo; consecutive threads touch consecutive banks.
static __global__ void k_smem_bw(double* sink, unsigned n_words, int iters) {
    extern __shared__ double s[];
    for (unsigned i = threadIdx.x; i < n_words; i += blockDim.x) s[i] = (double)(i + 1u);
    __syncthreads();

    const unsigned mask = n_words - 1u;          // n_words is a power of two
    const unsigned nb   = blockDim.x;
    unsigned idx = threadIdx.x;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    for (int it = 0; it < iters; ++it) {
        a0 += s[ idx            & mask];
        a1 += s[(idx +      nb) & mask];
        a2 += s[(idx + 2u * nb) & mask];
        a3 += s[(idx + 3u * nb) & mask];
        idx += 4u * nb;
    }
    // The compiler cannot prove this false, so the loads survive; it is never taken.
    const double acc = a0 + a1 + a2 + a3;
    if (acc == -1.0) sink[blockIdx.x] = acc;
}

// Pointer chase over a random cycle. One thread chases; every hop depends on the value
// the previous hop returned, so the hardware cannot start the next load early and the
// timing is a dependent-access latency rather than a throughput.
//
// Timing is clock64 INSIDE the kernel, bracketing the timed loop alone. That excludes
// the launch, the argument setup, the warm laps and the final store, so no differencing
// of two runs is needed to cancel overhead -- and differencing was itself the bug: two
// runs of different length touch different footprints, so subtracting them subtracted
// two different latencies and the result swung 375-470 ns run to run. The asm is
// volatile with a memory clobber so the clock reads cannot migrate across the loop.
//
// `warm` laps run untimed first, to populate the level being measured. The far probe
// needs none (its footprint exceeds L2, so every hop misses however often you walk it);
// the near probe needs a full lap, because L2 is cold at launch and a single traversal
// touching each line exactly once would otherwise miss on every hop and report the
// memory latency under L2's name.
static __global__ void k_chase_global(const unsigned* __restrict__ next, unsigned start,
                                      int warm, int steps, unsigned* sink,
                                      long long* cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned p = start;
    for (int i = 0; i < warm; ++i) p = next[p];
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < steps; ++i) p = next[p];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    sink[0] = p;
    *cycles = t1 - t0;
}

static __global__ void k_chase_smem(const unsigned* __restrict__ next, unsigned start,
                                    int steps, unsigned n, unsigned* sink,
                                    long long* cycles) {
    extern __shared__ unsigned sn[];
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) sn[i] = next[i];
    __syncthreads();
    if (threadIdx.x != 0) return;
    unsigned p = start;
    for (unsigned i = 0; i < n; ++i) p = sn[p];              // warm lap
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < steps; ++i) p = sn[p];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    sink[0] = p;
    *cycles = t1 - t0;
}

// One independent chase per block, each over its own disjoint sub-cycle at a distinct
// base offset, every one of them small enough that the whole set is an L2 hit. Blocks
// land on different SMs and different regions, so if the L2 is partitioned the returned
// latencies separate into as many clusters as there are partitions. %smid is recorded so
// the clustering can be checked against physical SM placement.
// ld.global.cg caches in L2 and BYPASSES L1. Without it this probe is wrong rather than
// merely imprecise: each block's region is a fraction of L2 and therefore smaller than
// L1, so ordinary loads are answered by L1 and the result describes L1, not the cache
// being partitioned. Measured on H200 the difference is 47 cycles against 285.
__device__ __forceinline__ unsigned ld_l2(const unsigned* p) {
    unsigned v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Per-SM L2 latency, one SM at a time.
//
// Method from arXiv 2606.22588 (L40 non-uniform L2) and arXiv 2607.19922 (DGNA):
//
//  * SERIALIZED. Blocks take turns on a global counter, so exactly one SM is probing at
//    any instant. Running all SMs concurrently -- which an earlier version of this file
//    did -- measures L2 under 132-way contention, which inflates and smears the samples
//    and can bury a real partition gap. This is the correction that matters most.
//  * One block per SM, guaranteed by asking for more than half the SM's shared memory so
//    the occupancy calculator cannot fit two. That also makes the turn counter safe:
//    every block is resident, so no block can be waiting for a turn held by a block that
//    was never scheduled.
//  * RANDOM stride, not a uniform grid: DGNA notes a random stride exposes partitioning
//    that a uniform one hides, since a fixed stride can alias onto one partition's share
//    of the address bits.
//  * %smid recorded, so the latency can be attributed to physical SM placement and fitted
//    with the additive model L = mu + a(sm) + b(slice) of the L40 paper.
static __global__ void k_l2_per_sm(const unsigned* __restrict__ next, unsigned nodes,
                                   unsigned steps, int nblocks, unsigned* turn,
                                   unsigned* sink, long long* cycles, unsigned* smids) {
    extern __shared__ unsigned scratch[];      // forces one block per SM; unused
    if (threadIdx.x != 0) return;
    scratch[0] = 0;

    unsigned sm = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    smids[blockIdx.x] = sm;

    // Wait for our turn. All blocks are resident, so this cannot deadlock.
    while (atomicAdd(turn, 0u) != blockIdx.x) __nanosleep(200);

    const unsigned* mine = next + (size_t)blockIdx.x * nodes;
    unsigned p = 0;
    for (unsigned i = 0; i < nodes; ++i) p = ld_l2(mine + p);      // warm: make it a hit
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (unsigned i = 0; i < steps; ++i) p = ld_l2(mine + p);
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    sink[blockIdx.x]   = p;
    cycles[blockIdx.x] = t1 - t0;

    __threadfence();
    atomicExch(turn, (blockIdx.x + 1) % (unsigned)nblocks);
}

// A cycle over `nodes` line-aligned slots chosen at RANDOM positions within a region of
// `words` words, rather than at a fixed stride. DGNA reports that a random stride exposes
// L2 partitioning that a uniform one hides: a fixed stride can alias onto one partition's
// share of the address bits, so every hop lands on the same side and the split vanishes.
// Indices are relative to the region base, so each block chases only inside its own slice.
inline void build_random_cycle(unsigned* region, unsigned words, unsigned nodes,
                               unsigned line_words, unsigned seed) {
    const unsigned slots = words / line_words;
    if (nodes > slots) nodes = slots;
    unsigned* pick = (unsigned*)std::malloc((size_t)slots * sizeof(unsigned));
    if (!pick) return;
    for (unsigned i = 0; i < slots; ++i) pick[i] = i;
    unsigned st = seed ? seed : 1u;
    // Partial Fisher-Yates: choose `nodes` distinct line slots out of `slots`, then walk
    // them in the order chosen. Both which lines and their order are random.
    for (unsigned i = 0; i < nodes; ++i) {
        st = st * 1664525u + 1013904223u;
        const unsigned j = i + st % (slots - i);
        const unsigned t = pick[i]; pick[i] = pick[j]; pick[j] = t;
    }
    for (unsigned i = 0; i < nodes; ++i)
        region[(size_t)pick[i] * line_words] = pick[(i + 1u) % nodes] * line_words;
    std::free(pick);
}

// A random cycle over `nodes` slots spaced `stride` apart// A random cycle over `nodes` slots spaced `stride` apart, written into `host_next`
// (which must hold nodes*stride unsigned). A cycle, not a permutation with an end, so
// the chase never leaves the buffer however many steps it takes.
inline void build_cycle(unsigned* host_next, unsigned nodes, unsigned stride,
                        unsigned seed = 12345u) {
    unsigned* order = (unsigned*)std::malloc((size_t)nodes * sizeof(unsigned));
    if (!order) return;
    for (unsigned i = 0; i < nodes; ++i) order[i] = i;
    // Fisher-Yates with a self-contained LCG, so discovery pulls in no RNG dependency
    // and repeats identically from run to run.
    unsigned st = seed;
    for (unsigned i = nodes - 1; i > 0; --i) {
        st = st * 1664525u + 1013904223u;
        const unsigned j = st % (i + 1u);
        const unsigned t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (unsigned i = 0; i < nodes; ++i)
        host_next[(size_t)order[i] * stride] = order[(i + 1u) % nodes] * stride;
    std::free(order);
}

// Grid that fills the device for `k`, chosen by the occupancy API rather than by a
// guessed multiple of the SM count.
inline int full_grid(const void* k, int block, size_t dyn_smem, int sm_count) {
    int per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k, block, dyn_smem);
    if (per_sm < 1) per_sm = 1;
    return per_sm * sm_count;
}

#if defined(TQR_WITH_NCCL)
// Device-initiated one-shot LL AllReduce, structured exactly like the fork's own
// ncclSymkRun_AllReduce_AGxLL_R: bcast one pack into every peer's slot band, then poll
// and reduce. Barrier-free -- the LL flags carry the synchronisation, which is the whole
// point of the design in arXiv 2607.16100. Timed with clock64 so the figure is the
// collective's own cost, with no kernel launch folded in.
static __global__ void k_ll_allreduce(ncclDevComm dc, ncclLLA2AHandle_t h,
                                      int iters, int maxElts, float* out,
                                      long long* cycles) {
    ncclCoopCta cta;
    ncclLLA2ASession<ncclCoopCta> s(cta, dc, ncclTeamLsa(dc), h, blockIdx.x, maxElts);
    const int rank = dc.rank, np = dc.nRanks;
    const int t = threadIdx.x, tn = blockDim.x;
    float v = 1.0f + rank;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) {
        s.bcast<float>(tn * rank + t, v);
        v = s.recvReduce</*Unroll=*/8, float>(
              t, np, tn,
              [] __device__ (float x) -> float { return x; },
              [] __device__ (float a, float b) -> float { return a + b; });
        s.endEpoch(cta);
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    if (t == 0) { cycles[blockIdx.x] = t1 - t0; out[blockIdx.x] = v; }
}
#endif

#if defined(TQR_WITH_NVSHMEM)
// The one-sided leg: a bare put and an acknowledgement, no collective and no host.
// This is the transport for the moves Definition 4.3 forbids a collective from carrying,
// above all Combine(a)'s ordered stack-Householder merge (non-commutative,
// Counterexample 16.1), so the panel-merge depth D_l is paid at THIS latency.
static __global__ void k_nvsh_pingpong(float* flag, int me, int peer, int iters,
                                       long long* cyc) {
    if (threadIdx.x || blockIdx.x) return;
    volatile float* f = flag;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 1; i <= iters; ++i) {
        if (me == 0) {
            nvshmem_float_p(flag, (float)i, peer);
            nvshmem_quiet();
            while (*f < (float)i) { }
        } else {
            while (*f < (float)i) { }
            nvshmem_float_p(flag, (float)i, peer);
            nvshmem_quiet();
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    *cyc = t1 - t0;
}

// L_L2_RTT of the paper's section VI: the latency of one __threadfence(), which makes the
// SM wait until outstanding transactions reach L2 -- the point of coherency between GPUs.
// It is the term that must be subtracted twice from a ping-pong to leave the wire cost.
static __global__ void k_l2_rtt(int iters, long long* cyc, float* sink) {
    if (threadIdx.x || blockIdx.x) return;
    float acc = 0.f;
    long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    for (int i = 0; i < iters; ++i) { sink[0] = acc; __threadfence(); acc += 1.f; }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
    *cyc = t1 - t0;
}

static __global__ void k_nvsh_put(float* dst, float* src, size_t n, int peer,
                                  int iters, long long* cyc) {
    long long t0, t1;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
    // Block-scoped put, one slice per block: a single-thread put serialises the payload
    // through one lane and measures 0.2 GB/s instead of the link.
    const size_t per = n / gridDim.x;
    const size_t off = (size_t)blockIdx.x * per;
    for (int i = 0; i < iters; ++i) {
        nvshmemx_float_put_block(dst + off, src + off, per, peer);
        __syncthreads();
        if (threadIdx.x == 0) nvshmem_quiet();
        __syncthreads();
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
        *cyc = t1 - t0;
    }
}
#endif  // TQR_WITH_NVSHMEM

inline int report_pairs_cap() { return discovery_report::kMaxPeers; }

// Rank and size of a multi-process run, from whichever launcher started it.
inline void world_from_env(int* rank, int* nranks) {
    const char* r = std::getenv("SLURM_PROCID");
    const char* n = std::getenv("SLURM_NTASKS");
    if (!r) r = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!n) n = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!r) r = std::getenv("PMI_RANK");
    if (!n) n = std::getenv("PMI_SIZE");
    *rank   = r ? std::atoi(r) : 0;
    *nranks = n ? std::atoi(n) : 1;
    if (*nranks < 1) *nranks = 1;
    if (*rank < 0 || *rank >= *nranks) *rank = 0;
}

}  // namespace detail

// Fills `out` with the hierarchy of the node this process is running on. Returns false
// and fills rep.error on failure, leaving `out` untouched. `rep` must outlive `out`.
inline bool discover(machine& out, discovery_report& rep,
                     const discovery_options& opt = discovery_options()) {
    rep = discovery_report();

    if (opt.word_bytes < 1)
        return detail::report_error(rep, "word_bytes must be at least 1");

    int visible = 0;
    TQR_CU(cudaGetDeviceCount(&visible));
    if (visible < 1) return detail::report_error(rep, "no CUDA device visible");

    // Establish the world first: in a multi-process run each rank owns exactly one GPU,
    // and the machine's P_3 is the rank count rather than the count of devices this
    // process can see.
    int wrank = opt.rank, wsize = opt.nranks;
    if (wrank < 0 || wsize < 0) detail::world_from_env(&wrank, &wsize);
    rep.rank = wrank; rep.nranks = wsize;
    rep.multi_process = (wsize > 1);

    const int dev0 = rep.multi_process ? (opt.first_device + wrank % visible)
                                       : opt.first_device;
    const int ndev = rep.multi_process ? wsize
                   : (opt.device_count > 0 ? opt.device_count : visible - dev0);
    if (dev0 < 0 || dev0 >= visible || ndev < 1)
        return detail::report_error(rep, "rank %d: device %d outside the %d visible",
                                    wrank, dev0, visible);
    if (!rep.multi_process && dev0 + ndev > visible)
        return detail::report_error(rep,
            "device range [%d,%d) outside the %d visible devices",
            dev0, dev0 + ndev, visible);

    int prev_dev = 0;
    TQR_CU(cudaGetDevice(&prev_dev));
    TQR_CU(cudaSetDevice(dev0));

    cudaDeviceProp prop{};
    TQR_CU(cudaGetDeviceProperties(&prop, dev0));

    // ---- queried structure ------------------------------------------------------
    rep.devices      = ndev;
    rep.sm_count     = prop.multiProcessorCount;
    rep.warp_size    = prop.warpSize;
    rep.warps_per_sm = prop.warpSize > 0
                     ? prop.maxThreadsPerMultiProcessor / prop.warpSize : 0;
    rep.cc_major     = prop.major;
    rep.cc_minor     = prop.minor;
    std::snprintf(rep.device_name, sizeof(rep.device_name), "%s", prop.name);

    rep.regfile_bytes_per_sm = (size_t)prop.regsPerMultiprocessor * sizeof(unsigned);
    rep.smem_bytes_per_sm    = (size_t)prop.sharedMemPerMultiprocessor;
    rep.l2_bytes             = (size_t)prop.l2CacheSize;
    rep.hbm_bytes            = prop.totalGlobalMem;

    if (rep.warps_per_sm < 1 || rep.sm_count < 1 || rep.l2_bytes == 0 ||
        rep.regfile_bytes_per_sm == 0 || rep.smem_bytes_per_sm == 0) {
        cudaSetDevice(prev_dev);
        return detail::report_error(rep,
            "device %d reported a zero capacity or peer count", dev0);
    }

    {   // Cross-check only, never entering the machine. cudaDeviceProp lost
        // memoryClockRate and clockRate in CUDA 13; the attribute API still has them.
        const double clk_hz  = (double)detail::attr(cudaDevAttrMemoryClockRate, dev0) * 1e3;
        const double bus_bit = (double)detail::attr(cudaDevAttrGlobalMemoryBusWidth, dev0);
        rep.hbm_bw_theoretical = clk_hz * (bus_bit / 8.0) * 2.0;   // DDR: two per clock
    }

    // Homogeneity: P_l is a single branching factor, so a mixed node has no honest
    // reading in the model.
    // Compare against every device this process can actually see. In a multi-process run
    // dev0 is this rank's own GPU and ndev is the RANK count, so walking dev0..dev0+ndev
    // would query device ids that do not exist -- rank 3 of 4 would ask for device 6, the
    // call would fail, and the ranks that got further would then hang at the first
    // collective waiting for it.
    const int hom_end = rep.multi_process ? visible : dev0 + ndev;
    for (int d = 0; d < hom_end; ++d) {
        if (d == dev0) continue;
        cudaDeviceProp p{};
        TQR_CU(cudaGetDeviceProperties(&p, d));
        if (p.totalGlobalMem != prop.totalGlobalMem ||
            p.multiProcessorCount != prop.multiProcessorCount ||
            p.l2CacheSize != prop.l2CacheSize ||
            p.sharedMemPerMultiprocessor != prop.sharedMemPerMultiprocessor ||
            p.regsPerMultiprocessor != prop.regsPerMultiprocessor) {
            rep.heterogeneous = true;
            if (!opt.allow_heterogeneous) {
                cudaSetDevice(prev_dev);
                return detail::report_error(rep,
                    "device %d differs from device %d; the model needs one branching "
                    "factor (set allow_heterogeneous to override)", d, dev0);
            }
        }
    }

#if defined(__linux__)
    {
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long psize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && psize > 0) rep.host_ram_bytes = (size_t)pages * (size_t)psize;
    }
#endif
    if (rep.host_ram_bytes == 0) {
        // Without a host figure the outermost region has no capacity. Say so rather
        // than inventing one.
        cudaSetDevice(prev_dev);
        return detail::report_error(rep,
            "could not read host physical memory size on this platform");
    }

    // ---- measured: shared-memory bandwidth ---------------------------------------
    {
        const size_t optin = (size_t)detail::attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev0);
        const size_t half  = rep.smem_bytes_per_sm / 2;
        // Half the SM's carve-out, so two blocks stay resident and the SM sees a
        // realistic number of contending warps rather than one block's worth.
        size_t smem_b = detail::pow2_floor(optin < half ? optin : half);
        if (smem_b < 1024) smem_b = 1024;

        const int block = prop.maxThreadsPerBlock;
        TQR_CU(cudaFuncSetAttribute(detail::k_smem_bw,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem_b));
        const int grid = detail::full_grid((const void*)detail::k_smem_bw, block,
                                           smem_b, rep.sm_count);
        const unsigned n_words = (unsigned)(smem_b / sizeof(double));

        double* sink = nullptr;
        TQR_CU(cudaMalloc(&sink, (size_t)grid * sizeof(double)));

        const int iters = 1 << 14;   // enough that the staging prologue is lost in it
        detail::timer t;
        detail::converger bw{{}, 0, opt.repeats, opt.target_spread};
        for (int r = 0; r < opt.warmups + opt.max_repeats && !bw.settled(); ++r) {
            t.start();
            detail::k_smem_bw<<<grid, block, smem_b>>>(sink, n_words, iters);
            const double sec = t.stop();
            if (cudaGetLastError() != cudaSuccess) break;
            if (r < opt.warmups || sec <= 0.0) continue;
            // 4 loads of 8 bytes per iteration per thread.
            const double bytes = 4.0 * sizeof(double) * (double)iters *
                                 (double)block * (double)grid;
            bw.add(bytes / sec);
        }
        cudaFree(sink);
        detail::take(rep, bw, "smem bw", &rep.smem_bw_aggregate, &rep.smem_bw_spread);
    }

    // ---- measured: L2 and HBM streaming bandwidth --------------------------------
    {
        size_t freeb = 0, totalb = 0;
        TQR_CU(cudaMemGetInfo(&freeb, &totalb));

        // HBM: a working set L2 cannot hold, capped so discovery does not evict a
        // caller's allocations.
        size_t hbm_b = rep.l2_bytes * (size_t)opt.hbm_l2_multiple;
        const size_t cap = (size_t)((double)freeb * opt.max_free_fraction);
        if (hbm_b > cap) hbm_b = cap;
        hbm_b &= ~(size_t)(sizeof(uint4) - 1);
        if (hbm_b < rep.l2_bytes * 4) {
            cudaSetDevice(prev_dev);
            return detail::report_error(rep,
                "only %.1f MiB free on device %d; need %.1f MiB to stream past a "
                "%.1f MiB L2", freeb / 1048576.0, dev0,
                rep.l2_bytes * 4 / 1048576.0, rep.l2_bytes / 1048576.0);
        }

        // L2: src and dst together occupy a quarter of L2 at the default divisor, so
        // they stay resident across passes.
        size_t l2_b = rep.l2_bytes / (size_t)opt.l2_working_divisor;
        l2_b &= ~(size_t)(sizeof(uint4) - 1);
        if (l2_b < sizeof(uint4)) l2_b = sizeof(uint4);

        uint4 *src = nullptr, *dst = nullptr;
        TQR_CU(cudaMalloc(&src, hbm_b));
        if (cudaMalloc(&dst, hbm_b) != cudaSuccess) {
            cudaFree(src);
            cudaSetDevice(prev_dev);
            return detail::report_error(rep,
                "could not allocate two %.1f MiB streaming buffers", hbm_b / 1048576.0);
        }
        TQR_CU(cudaMemset(src, 1, hbm_b));

        const int block = 256;
        const int grid  = detail::full_grid((const void*)detail::k_copy, block, 0,
                                            rep.sm_count);
        detail::timer t;

        // Both probes run the same kernel; only the working-set size differs, which is
        // the whole point -- the difference between the two numbers is the difference
        // between hitting L2 and missing it.
        struct probe { size_t bytes; int passes; double* out; double* spread;
                       const char* name; };
        const probe probes[2] = {
            { hbm_b, 1,   &rep.hbm_bw,          &rep.hbm_bw_spread, "hbm bw" },
            { l2_b,  128, &rep.l2_bw_aggregate, &rep.l2_bw_spread,  "L2 bw"  } };

        for (const probe& pr : probes) {
            const size_t n = pr.bytes / sizeof(uint4);
            detail::converger bw{{}, 0, opt.repeats, opt.target_spread};
            for (int r = 0; r < opt.warmups + opt.max_repeats && !bw.settled(); ++r) {
                t.start();
                detail::k_copy<<<grid, block>>>(src, dst, n, pr.passes);
                const double sec = t.stop();
                if (cudaGetLastError() != cudaSuccess) break;
                if (r < opt.warmups || sec <= 0.0) continue;
                bw.add(2.0 * (double)pr.bytes * (double)pr.passes / sec);
            }
            detail::take(rep, bw, pr.name, pr.out, pr.spread);
        }
        cudaFree(src);
        cudaFree(dst);
    }

    // ---- measured: dependent-access latency --------------------------------------
    // Counted in CYCLES by clock64 inside the kernel, then converted with the SM clock
    // measured from the far chase itself -- a kernel that runs for the better part of a
    // second, so dividing its in-kernel cycles by its wall time gives the frequency to
    // parts per million. Cycles are what the hardware holds fixed; seconds are what the
    // cost model wants; measuring the conversion factor rather than assuming the
    // advertised boost clock is what keeps the two consistent.
    {
        const size_t stride_b = (size_t)opt.chase_stride_bytes;
        const unsigned stride = (unsigned)(stride_b / sizeof(unsigned));

        unsigned*  sink   = nullptr;
        long long* dcycle = nullptr;
        TQR_CU(cudaMalloc(&sink, sizeof(unsigned)));
        TQR_CU(cudaMalloc(&dcycle, sizeof(long long)));

        size_t freeb = 0, totalb = 0;
        TQR_CU(cudaMemGetInfo(&freeb, &totalb));
        const size_t fcap = (size_t)((double)freeb * opt.max_free_fraction);

        size_t far_fp = rep.l2_bytes * (size_t)opt.chase_far_l2_multiple;
        if (far_fp > fcap) far_fp = fcap;                       // shrink, never grow
        const size_t near_fp = rep.l2_bytes / (size_t)opt.chase_near_l2_divisor;

        // The far probe runs first: it doubles as the clock calibration, and the near
        // probe's conversion depends on it.
        struct gprobe { size_t footprint; bool warm_lap; double* lat; double* cyc;
                        double* spread; size_t* record; const char* name; };
        const gprobe gp[2] = {
            { far_fp,  false, &rep.hbm_lat, &rep.hbm_lat_cycles, &rep.hbm_lat_spread,
              &rep.chase_far_footprint,  "hbm latency" },
            { near_fp, true,  &rep.l2_lat,  &rep.l2_lat_cycles,  &rep.l2_lat_spread,
              &rep.chase_near_footprint, "L2 latency"  },
        };

        for (const gprobe& g : gp) {
            if (stride == 0) continue;
            const size_t want = g.footprint / stride_b;
            if (want < 8 || want > 0xFFFFFFFFull / stride) continue;
            const unsigned nodes = (unsigned)want;

            // nodes == steps: one full traversal, every line touched exactly once
            // before any is revisited, footprint exactly nodes * stride.
            const size_t slots = (size_t)nodes * stride;
            unsigned* h = (unsigned*)std::calloc(slots, sizeof(unsigned));
            if (!h) continue;
            detail::build_cycle(h, nodes, stride);
            unsigned* d = nullptr;
            if (cudaMalloc(&d, slots * sizeof(unsigned)) != cudaSuccess) {
                std::free(h);
                continue;
            }
            cudaMemcpy(d, h, slots * sizeof(unsigned), cudaMemcpyHostToDevice);
            std::free(h);

            const int warm = g.warm_lap ? (int)nodes : 64;
            detail::timer t;
            detail::converger cyc{{}, 0, opt.latency_repeats, opt.target_spread};
            detail::converger hz {{}, 0, opt.latency_repeats, opt.target_spread};
            for (int r = 0; r < opt.latency_warmups + opt.latency_max_repeats
                            && !cyc.settled(); ++r) {
                long long c = 0;
                t.start();
                detail::k_chase_global<<<1, 1>>>(d, 0u, warm, (int)nodes, sink, dcycle);
                const double sec = t.stop();
                if (cudaGetLastError() != cudaSuccess) break;
                cudaMemcpy(&c, dcycle, sizeof(c), cudaMemcpyDeviceToHost);
                if (r < opt.latency_warmups || c <= 0 || sec <= 0.0) continue;
                cyc.add((double)c / (double)nodes);
                if (!g.warm_lap) hz.add((double)c / sec);       // far probe only
            }
            cudaFree(d);
            if (!cyc.ok()) continue;
            double ignored = 0.0;
            detail::take(rep, cyc, g.name, g.cyc, g.spread);
            if (hz.ok()) detail::take(rep, hz, "sm clock", &rep.sm_clock_hz, &ignored);
            *g.record = slots * sizeof(unsigned);
            if (rep.sm_clock_hz > 0.0) *g.lat = *g.cyc / rep.sm_clock_hz;
        }

        // Shared memory: the staged cycle is walked in full, once warm and once timed.
        {
            const size_t optin = (size_t)detail::attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev0);
            size_t sm_b = detail::pow2_floor(optin);
            if (sm_b > rep.smem_bytes_per_sm) sm_b = detail::pow2_floor(rep.smem_bytes_per_sm);
            const unsigned nodes = (unsigned)(sm_b / sizeof(unsigned));
            if (nodes >= 4) {
                unsigned* h = (unsigned*)std::calloc(nodes, sizeof(unsigned));
                unsigned* d = nullptr;
                if (h && cudaMalloc(&d, (size_t)nodes * sizeof(unsigned)) == cudaSuccess) {
                    detail::build_cycle(h, nodes, 1u);
                    cudaMemcpy(d, h, (size_t)nodes * sizeof(unsigned),
                               cudaMemcpyHostToDevice);
                    cudaFuncSetAttribute(detail::k_chase_smem,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)sm_b);
                    detail::timer t;
                    detail::converger cyc{{}, 0, opt.latency_repeats, opt.target_spread};
                    for (int r = 0; r < opt.latency_warmups + opt.latency_max_repeats
                                    && !cyc.settled(); ++r) {
                        long long c = 0;
                        t.start();
                        detail::k_chase_smem<<<1, 256, sm_b>>>(d, 0u, (int)nodes, nodes,
                                                               sink, dcycle);
                        t.stop();
                        if (cudaGetLastError() != cudaSuccess) break;
                        cudaMemcpy(&c, dcycle, sizeof(c), cudaMemcpyDeviceToHost);
                        if (r < opt.latency_warmups || c <= 0) continue;
                        cyc.add((double)c / (double)nodes);
                    }
                    cudaFree(d);
                    if (cyc.ok()) {
                        detail::take(rep, cyc, "smem latency", &rep.smem_lat_cycles,
                                     &rep.smem_lat_spread);
                        if (rep.sm_clock_hz > 0.0)
                            rep.smem_lat = rep.smem_lat_cycles / rep.sm_clock_hz;
                    }
                }
                std::free(h);
            }
        }
        // L2 partitioning, per the method of arXiv 2606.22588 and arXiv 2607.19922.
        // One SM probes at a time, each over its own L2-resident region reached by a
        // RANDOM stride, and the resulting per-SM latencies are fitted with a Gaussian
        // mixture. Two well-separated components mean a partitioned L2 (DGNA sees this on
        // A100); one component with a wide spread means a continuous placement gradient
        // (the L40 result). Both are real hardware, so the probe returns whichever it
        // finds instead of assuming a 2-way split.
        if (rep.l2_bytes > 0 && rep.sm_count > 0) {
            const int blocks = rep.sm_count;
            // Ask for more than half the SM's shared memory so exactly one block lands
            // per SM. That is what makes every block resident, which the turn counter
            // needs, and what gives one clean sample per physical SM.
            size_t smem_force = rep.smem_bytes_per_sm / 2 + 1024;
            const size_t optin2 =
                (size_t)detail::attr(cudaDevAttrMaxSharedMemoryPerBlockOptin, dev0);
            if (smem_force > optin2) smem_force = optin2;

            // Total working set stays inside L2 so every hop is a hit; per-block region
            // is a slice of it. Nodes are chosen at random line-aligned offsets rather
            // than on a fixed grid.
            const size_t line = (size_t)opt.l2_line_bytes;
            const size_t total = rep.l2_bytes / (size_t)opt.l2_probe_divisor;
            const unsigned per = (unsigned)(total / ((size_t)blocks * line));
            if (per >= 16 && line >= sizeof(unsigned)) {
                const size_t span = (size_t)blocks * per * (line / sizeof(unsigned));
                unsigned*  h   = (unsigned*)std::calloc(span, sizeof(unsigned));
                unsigned*  d   = nullptr;
                long long* dc  = nullptr;
                unsigned  *dsm = nullptr, *dsk = nullptr, *dturn = nullptr;
                if (h && cudaMalloc(&d, span * sizeof(unsigned)) == cudaSuccess &&
                    cudaMalloc(&dc, (size_t)blocks * sizeof(long long)) == cudaSuccess &&
                    cudaMalloc(&dsm, (size_t)blocks * sizeof(unsigned)) == cudaSuccess &&
                    cudaMalloc(&dsk, (size_t)blocks * sizeof(unsigned)) == cudaSuccess &&
                    cudaMalloc(&dturn, sizeof(unsigned)) == cudaSuccess) {
                    const unsigned words_per_block = per * (unsigned)(line / sizeof(unsigned));
                    for (int bl = 0; bl < blocks; ++bl)
                        detail::build_random_cycle(h + (size_t)bl * words_per_block,
                                                   words_per_block, per,
                                                   (unsigned)(line / sizeof(unsigned)),
                                                   9781u + (unsigned)bl);
                    cudaMemcpy(d, h, span * sizeof(unsigned), cudaMemcpyHostToDevice);
                    cudaMemset(dturn, 0, sizeof(unsigned));
                    cudaFuncSetAttribute(detail::k_l2_per_sm,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)smem_force);
                    const unsigned steps = per * 8;      // ~8 laps of dependent hits
                    detail::k_l2_per_sm<<<blocks, 32, smem_force>>>(
                        d, words_per_block, steps, blocks, dturn, dsk, dc, dsm);
                    cudaDeviceSynchronize();
                    if (cudaGetLastError() == cudaSuccess) {
                        long long* hc = (long long*)std::calloc((size_t)blocks, sizeof(long long));
                        double* v = (double*)std::calloc((size_t)blocks, sizeof(double));
                        if (hc && v) {
                            cudaMemcpy(hc, dc, (size_t)blocks * sizeof(long long),
                                       cudaMemcpyDeviceToHost);
                            int k = 0;
                            for (int bl = 0; bl < blocks; ++bl)
                                if (hc[bl] > 0) v[k++] = (double)hc[bl] / (double)steps;
                            if (k >= 8) {
                                std::sort(v, v + k);
                                double mu[8] = {0}, wt[8] = {0};
                                const int comps = detail::gmm_components(
                                    v, k, 4, mu, wt, opt.l2_mode_separation_sigmas);
                                rep.l2_probe_blocks = k;
                                rep.l2_partitions   = comps;
                                rep.l2_near_cycles  = v[0];
                                rep.l2_far_cycles   = v[k - 1];
                                rep.l2_median_cycles = v[k / 2];
                                rep.l2_gmm_components = comps;
                                for (int j = 0; j < comps && j < 4; ++j) rep.l2_mode_mu[j] = mu[j];
                            }
                        }
                        std::free(hc); std::free(v);
                    }
                }
                cudaFree(d); cudaFree(dc); cudaFree(dsm); cudaFree(dsk); cudaFree(dturn);
                std::free(h);
            }
        }

        if (opt.l2_partitions_override > 0) rep.l2_partitions = opt.l2_partitions_override;

        cudaFree(sink);
        cudaFree(dcycle);
    }

    // ---- measured: boundary 3, the path out of a GPU -----------------------------
    if (opt.measure_links) {
        const int peers = ndev - 1;

        // Full peer reachability is a property of the fabric, so it is established here
        // rather than inside whichever probe happens to measure bandwidth. It decides
        // peer_fanin: an all-to-all fabric reaches every rank in one stage, so a
        // collective has depth 1 rather than ceil(log2 P), and Scrit = K + D - 1
        // (Theorem 11.4) sees the difference directly.
        if (peers > 0 && !rep.multi_process) {
            rep.p2p_all_pairs = true;
            for (int j = dev0; j < dev0 + ndev && rep.p2p_all_pairs; ++j)
                for (int k = dev0; k < dev0 + ndev; ++k) {
                    if (j == k) continue;
                    int can = 0;
                    cudaDeviceCanAccessPeer(&can, j, k);
                    if (!can) { rep.p2p_all_pairs = false; break; }
                }
        }
        size_t freeb = 0, totalb = 0;
        TQR_CU(cudaMemGetInfo(&freeb, &totalb));
        // One outbound buffer here plus one inbound buffer on every peer.
        size_t xfer = (size_t)((double)freeb * opt.max_free_fraction) / (size_t)(peers + 1);
        xfer &= ~(size_t)4095;
        if (xfer < 4096) xfer = 4096;

        // Host path, always measured: it IS boundary 3 when there is one GPU, and it
        // is the streaming path whenever the matrix outgrows aggregate HBM.
        {
            void* h = nullptr;
            void* d = nullptr;
            if (cudaMallocHost(&h, xfer) == cudaSuccess) {
                if (cudaMalloc(&d, xfer) == cudaSuccess) {
                    detail::timer t;
                    detail::converger bw{{}, 0, opt.repeats, opt.target_spread};
                    for (int r = 0; r < opt.warmups + opt.max_repeats && !bw.settled(); ++r) {
                        t.start();
                        cudaMemcpyAsync(d, h, xfer, cudaMemcpyHostToDevice, 0);
                        const double sec = t.stop();
                        if (r >= opt.warmups && sec > 0.0) bw.add((double)xfer / sec);
                    }
                    detail::take(rep, bw, "host bw", &rep.host_bw, &rep.host_bw_spread);

                    // One dependency-bearing round: a word crosses and the issuing side
                    // waits for it. Synchronizing every iteration is the point --
                    // pipelined copies would measure bandwidth, not exposed latency.
                    detail::converger lat{{}, 0, opt.repeats, opt.target_spread};
                    for (int r = 0; r < opt.warmups + opt.max_repeats && !lat.settled(); ++r) {
                        t.start();
                        for (int i = 0; i < opt.link_rounds; ++i) {
                            cudaMemcpyAsync(d, h, (size_t)opt.word_bytes,
                                            cudaMemcpyHostToDevice, 0);
                            cudaStreamSynchronize(0);
                        }
                        const double sec = t.stop();
                        if (r >= opt.warmups && opt.link_rounds > 0)
                            lat.add(sec / (double)opt.link_rounds);
                    }
                    detail::take(rep, lat, "host latency", &rep.host_lat,
                                 &rep.host_lat_spread);
                    cudaFree(d);
                }
                cudaFreeHost(h);
            }
        }

#if defined(TQR_WITH_NCCL)
        // Collective peer path, measured with NCCL. One host thread per device: driving
        // N communicators from a single thread serialises N launches per iteration and
        // measures the CPU rather than the fabric (17.5 us against 8.9 us on 4 GPUs).
        // ncclCommInitAll owns every device from one process, so it is the
        // single-process path; a multi-process run needs ncclCommInitRank instead.
        if (peers > 0 && !rep.multi_process) {
            std::vector<int> devs(ndev);
            for (int i = 0; i < ndev; ++i) devs[i] = dev0 + i;
            std::vector<ncclComm_t> comm(ndev);
            if (ncclCommInitAll(comm.data(), ndev, devs.data()) == ncclSuccess) {
                std::vector<cudaStream_t> cs(ndev);
                std::vector<float*> cb(ndev);
                const size_t big = xfer / sizeof(float);
                bool ok2 = true;
                for (int i = 0; i < ndev && ok2; ++i) {
                    if (cudaSetDevice(dev0 + i) != cudaSuccess) { ok2 = false; break; }
                    if (cudaStreamCreate(&cs[i]) != cudaSuccess) { ok2 = false; break; }
                    if (cudaMalloc(&cb[i], big * sizeof(float)) != cudaSuccess) ok2 = false;
                    else cudaMemset(cb[i], 0, big * sizeof(float));
                }
                // One word for latency, the whole buffer for bandwidth.
                const size_t counts[2] = { 1, big };
                double res[2] = {0.0, 0.0};
                for (int w = 0; w < 2 && ok2; ++w) {
                    const size_t cnt = counts[w];
                    const int iters = (w == 0) ? opt.link_rounds : 8;
                    std::atomic<int> arrived{0}, epoch{0};
                    auto bar = [&]() {
                        const int e = epoch.load(std::memory_order_acquire);
                        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == ndev) {
                            arrived.store(0, std::memory_order_release);
                            epoch.store(e + 1, std::memory_order_release);
                        } else {
                            while (epoch.load(std::memory_order_acquire) == e)
                                std::this_thread::yield();
                        }
                    };
                    std::vector<double> best((size_t)ndev, 1e30);
                    std::vector<std::thread> th;
                    for (int i = 0; i < ndev; ++i) th.emplace_back([&, i]() {
                        cudaSetDevice(dev0 + i);
                        for (int r = 0; r < opt.warmups + 8; ++r)
                            ncclAllReduce(cb[i], cb[i], cnt, ncclFloat, ncclSum, comm[i], cs[i]);
                        cudaStreamSynchronize(cs[i]);
                        bar();
                        for (int rep = 0; rep < opt.repeats; ++rep) {
                            cudaEvent_t a, b;
                            cudaEventCreate(&a); cudaEventCreate(&b);
                            bar();
                            cudaEventRecord(a, cs[i]);
                            for (int it = 0; it < iters; ++it)
                                ncclAllReduce(cb[i], cb[i], cnt, ncclFloat, ncclSum,
                                              comm[i], cs[i]);
                            cudaEventRecord(b, cs[i]);
                            cudaStreamSynchronize(cs[i]);
                            float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
                            const double t1 = (double)ms * 1e-3 / iters;
                            if (t1 > 0.0 && t1 < best[i]) best[i] = t1;
                            cudaEventDestroy(a); cudaEventDestroy(b);
                        }
                    });
                    for (auto& t : th) t.join();
                    double slowest = 0.0;
                    for (int i = 0; i < ndev; ++i) if (best[i] < 1e30 && best[i] > slowest)
                        slowest = best[i];
                    res[w] = slowest;
                }
                if (res[0] > 0.0) { rep.peer_lat_host = res[0]; rep.peer_lat = res[0]; }
                if (res[1] > 0.0) {
                    rep.peer_bw_out = (double)xfer / res[1];
                    rep.peer_bw_spread = 0.0;
                }
                rep.peer_via_nccl = true;

                // Now the device-initiated figure, which supersedes the host-API one for
                // alpha_3. One warp per rank, one pack each: the smallest dependency-
                // bearing round the fabric can carry.
                {
                    const int tn = 32, maxElts = tn * ndev;
                    std::vector<double> lat((size_t)ndev, 0.0);
                    std::vector<std::thread> th2;
                    for (int i = 0; i < ndev; ++i) th2.emplace_back([&, i]() {
                        cudaSetDevice(dev0 + i);
                        ncclDevCommRequirements dreq = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
                        ncclLLA2AHandle_t hh{};
                        ncclDevResourceRequirements rr{};
                        if (ncclLLA2ACreateRequirement(1, maxElts, &hh, &rr) != ncclSuccess) return;
                        dreq.resourceRequirementsList = &rr;
                        ncclDevComm dcm{};
                        if (ncclDevCommCreate(comm[i], &dreq, &dcm) != ncclSuccess) return;
                        float* o = nullptr; long long* c = nullptr;
                        if (cudaMalloc(&o, sizeof(float)) != cudaSuccess) return;
                        if (cudaMalloc(&c, sizeof(long long)) != cudaSuccess) { cudaFree(o); return; }
                        cudaStream_t ds;
                        if (cudaStreamCreate(&ds) != cudaSuccess) { cudaFree(o); cudaFree(c); return; }
                        const int it = 2000;
                        detail::k_ll_allreduce<<<1, tn, 0, ds>>>(dcm, hh, 64, maxElts, o, c);
                        cudaStreamSynchronize(ds);
                        detail::k_ll_allreduce<<<1, tn, 0, ds>>>(dcm, hh, it, maxElts, o, c);
                        cudaStreamSynchronize(ds);
                        if (cudaGetLastError() == cudaSuccess && rep.sm_clock_hz > 0.0) {
                            long long hc = 0;
                            cudaMemcpy(&hc, c, sizeof(hc), cudaMemcpyDeviceToHost);
                            if (hc > 0) lat[i] = (double)hc / it / rep.sm_clock_hz;
                        }
                        cudaFree(o); cudaFree(c); cudaStreamDestroy(ds);
                    });
                    for (auto& t : th2) t.join();
                    double worst = 0.0;
                    for (int i = 0; i < ndev; ++i) if (lat[i] > worst) worst = lat[i];
                    if (worst > 0.0) {
                        rep.peer_lat_device      = worst;
                        rep.peer_lat             = worst;      // this is the alpha_3 used
                        rep.peer_lat_spread      = 0.0;
                        rep.peer_device_initiated = true;
                    }
                    cudaSetDevice(dev0);
                    cudaGetLastError();
                }
                for (int i = 0; i < ndev; ++i) {
                    if (cs[i]) { cudaSetDevice(dev0 + i); cudaStreamDestroy(cs[i]); }
                    if (cb[i]) { cudaSetDevice(dev0 + i); cudaFree(cb[i]); }
                }
                for (int i = 0; i < ndev; ++i) ncclCommDestroy(comm[i]);
                cudaSetDevice(dev0);
                cudaGetLastError();
            }
        }
        if (!rep.peer_via_nccl)
#endif
        // Peer path: device dev0 pushing to every peer at once. The aggregate outbound
        // rate is what one level-3 domain sustains across boundary 3; a single pair
        // would understate a GPU with independent links to each neighbour.
        //
        // Single-process only. This reaches into every device from one process, which a
        // multi-process run must not do -- each rank owns one GPU there, and the peer
        // legs are measured by NCCL (ncclCommInitRank) and NVSHMEM instead.
        if (peers > 0 && !rep.multi_process) {

            if (rep.p2p_all_pairs) {
                uint4* srcbuf = nullptr;
                void** dstbuf = (void**)std::calloc((size_t)ndev, sizeof(void*));
                cudaStream_t* st =
                    (cudaStream_t*)std::calloc((size_t)ndev, sizeof(cudaStream_t));
                bool ok = dstbuf && st && cudaMalloc(&srcbuf, xfer) == cudaSuccess;

                for (int j = dev0 + 1; ok && j < dev0 + ndev; ++j) {
                    if (cudaSetDevice(j) != cudaSuccess) { ok = false; break; }
                    // Already-enabled is not an error: a caller may have set it up.
                    const cudaError_t pe = cudaDeviceEnablePeerAccess(dev0, 0);
                    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled)
                        ok = false;
                    cudaGetLastError();
                    if (ok && cudaMalloc(&dstbuf[j - dev0], xfer) != cudaSuccess)
                        ok = false;
                }
                if (ok) ok = (cudaSetDevice(dev0) == cudaSuccess);
                for (int j = dev0 + 1; ok && j < dev0 + ndev; ++j) {
                    const cudaError_t pe = cudaDeviceEnablePeerAccess(j, 0);
                    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled)
                        ok = false;
                    cudaGetLastError();
                    if (ok && cudaStreamCreate(&st[j - dev0]) != cudaSuccess) ok = false;
                }

                if (ok) {
                    detail::timer t;
                    detail::converger bw{{}, 0, opt.repeats, opt.target_spread};
                    for (int r = 0; r < opt.warmups + opt.max_repeats && !bw.settled(); ++r) {
                        t.start();
                        for (int j = dev0 + 1; j < dev0 + ndev; ++j)
                            cudaMemcpyPeerAsync(dstbuf[j - dev0], j, srcbuf, dev0, xfer,
                                                st[j - dev0]);
                        for (int j = dev0 + 1; j < dev0 + ndev; ++j)
                            cudaStreamSynchronize(st[j - dev0]);
                        const double sec = t.stop();
                        if (r >= opt.warmups && sec > 0.0)
                            bw.add((double)xfer * (double)peers / sec);
                    }
                    detail::take(rep, bw, "peer bw", &rep.peer_bw_out,
                                 &rep.peer_bw_spread);

                    detail::converger lat{{}, 0, opt.repeats, opt.target_spread};
                    for (int r = 0; r < opt.warmups + opt.max_repeats && !lat.settled(); ++r) {
                        t.start();
                        for (int i = 0; i < opt.link_rounds; ++i) {
                            cudaMemcpyPeerAsync(dstbuf[1], dev0 + 1, srcbuf, dev0,
                                                (size_t)opt.word_bytes, st[1]);
                            cudaStreamSynchronize(st[1]);
                        }
                        const double sec = t.stop();
                        if (r >= opt.warmups && opt.link_rounds > 0)
                            lat.add(sec / (double)opt.link_rounds);
                    }
                    detail::take(rep, lat, "peer latency", &rep.peer_lat,
                                 &rep.peer_lat_spread);
                }

                for (int j = dev0 + 1; j < dev0 + ndev; ++j) {
                    if (st && st[j - dev0]) cudaStreamDestroy(st[j - dev0]);
                    if (dstbuf && dstbuf[j - dev0]) {
                        cudaSetDevice(j);
                        cudaFree(dstbuf[j - dev0]);
                    }
                }
                cudaSetDevice(dev0);
                cudaFree(srcbuf);
                std::free(dstbuf);
                std::free(st);
                cudaGetLastError();
            }
        }
        rep.links_measured = true;
    }

#if defined(TQR_WITH_NCCL)
    // Collective leg of a MULTI-PROCESS run. ncclCommInitAll cannot be used here (it
    // wants every device in one process), so the communicator is built rank by rank from
    // a shared unique id, and the same device-initiated one-shot LL AllReduce is timed.
    if (opt.measure_links && rep.multi_process && rep.sm_clock_hz > 0.0) {
        ncclUniqueId nid;
        char npath[512];
        std::snprintf(npath, sizeof(npath), "%s.nccl", opt.bootstrap_file);
        bool nboot = true;
        if (wrank == 0) {
            std::remove(npath);
            if (ncclGetUniqueId(&nid) != ncclSuccess) nboot = false;
            if (nboot) {
                char tmp[512];
                std::snprintf(tmp, sizeof(tmp), "%s.tmp", npath);
                FILE* f = std::fopen(tmp, "wb");
                if (!f) nboot = false;
                else {
                    nboot = std::fwrite(&nid, sizeof(nid), 1, f) == 1;
                    std::fclose(f);
                    if (nboot) nboot = (std::rename(tmp, npath) == 0);
                }
            }
        } else {
            nboot = false;
            for (int t = 0; t < 600 && !nboot; ++t) {
                FILE* f = std::fopen(npath, "rb");
                if (f) { nboot = std::fread(&nid, sizeof(nid), 1, f) == 1; std::fclose(f); }
                if (!nboot) usleep(50000);
            }
        }
        ncclComm_t c1 = nullptr;
        if (nboot && ncclCommInitRank(&c1, wsize, nid, wrank) == ncclSuccess) {
            rep.peer_via_nccl = true;
            rep.p2p_all_pairs = true;      // NCCL built a communicator across every rank

            const int tn = 32, maxElts = tn * wsize;
            ncclDevCommRequirements dreq = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            ncclLLA2AHandle_t hh{};
            ncclDevResourceRequirements rr{};
            ncclDevComm dcm{};
            if (ncclLLA2ACreateRequirement(1, maxElts, &hh, &rr) == ncclSuccess) {
                dreq.resourceRequirementsList = &rr;
                if (ncclDevCommCreate(c1, &dreq, &dcm) == ncclSuccess) {
                    float* o = nullptr; long long* cy = nullptr;
                    cudaStream_t ds = nullptr;
                    if (cudaMalloc(&o, sizeof(float)) == cudaSuccess &&
                        cudaMalloc(&cy, sizeof(long long)) == cudaSuccess &&
                        cudaStreamCreate(&ds) == cudaSuccess) {
                        const int it = 2000;
                        detail::k_ll_allreduce<<<1, tn, 0, ds>>>(dcm, hh, 64, maxElts, o, cy);
                        cudaStreamSynchronize(ds);
                        detail::k_ll_allreduce<<<1, tn, 0, ds>>>(dcm, hh, it, maxElts, o, cy);
                        cudaStreamSynchronize(ds);
                        if (cudaGetLastError() == cudaSuccess) {
                            long long hc = 0;
                            cudaMemcpy(&hc, cy, sizeof(hc), cudaMemcpyDeviceToHost);
                            if (hc > 0) {
                                rep.peer_lat_device = (double)hc / it / rep.sm_clock_hz;
                                rep.peer_lat = rep.peer_lat_device;
                                rep.peer_device_initiated = true;
                            }
                        }
                    }
                    if (ds) cudaStreamDestroy(ds);
                    if (o) cudaFree(o);
                    if (cy) cudaFree(cy);
                }
            }
            // Bandwidth from a large AllReduce: the algorithmic rate one rank sustains
            // outbound. The buffer is allocated with ncclMemAlloc and registered as a
            // SYMMETRIC window, which is what dispatches AllReduce to the low-latency
            // symmetric kernels instead of the generic ring path -- worth 121 -> 198 GB/s
            // at 8 MiB on this machine. Plain cudaMalloc here silently measures the slow
            // path and understates B_3.
            {
                const size_t nf = 1u << 22;
                float* b1 = nullptr;
                ncclWindow_t bwin = nullptr;
                cudaStream_t bs = nullptr;
                const bool symok = (ncclMemAlloc((void**)&b1, nf * sizeof(float)) == ncclSuccess)
                    && (ncclCommWindowRegister(c1, b1, nf * sizeof(float), &bwin,
                                               NCCL_WIN_COLL_SYMMETRIC) == ncclSuccess);
                if (!symok && !b1) cudaMalloc(&b1, nf * sizeof(float));
                rep.peer_bw_symmetric = symok;
                if (b1 && cudaStreamCreate(&bs) == cudaSuccess) {
                    cudaMemset(b1, 0, nf * sizeof(float));
                    for (int w = 0; w < 4; ++w)
                        ncclAllReduce(b1, b1, nf, ncclFloat, ncclSum, c1, bs);
                    cudaStreamSynchronize(bs);
                    cudaEvent_t ea, eb;
                    cudaEventCreate(&ea); cudaEventCreate(&eb);
                    const int it = 16;
                    cudaEventRecord(ea, bs);
                    for (int i = 0; i < it; ++i)
                        ncclAllReduce(b1, b1, nf, ncclFloat, ncclSum, c1, bs);
                    cudaEventRecord(eb, bs);
                    cudaStreamSynchronize(bs);
                    float ms = 0.f; cudaEventElapsedTime(&ms, ea, eb);
                    if (ms > 0.f)
                        rep.peer_bw_out = (double)nf * sizeof(float) * it / ((double)ms * 1e-3);
                    cudaEventDestroy(ea); cudaEventDestroy(eb);
                    cudaStreamDestroy(bs);
                }
                if (bwin) ncclCommWindowDeregister(c1, bwin);
                if (b1) { if (symok) ncclMemFree(b1); else cudaFree(b1); }
            }
            ncclCommDestroy(c1);
            if (wrank == 0) std::remove(npath);
        }
        cudaSetDevice(dev0);
        cudaGetLastError();
    }
#endif

#if defined(TQR_WITH_NVSHMEM)
    // The one-sided leg, measured here rather than supplied. Requires the run to be
    // multi-process (one PE per GPU), which is why discover() is rank-aware at all.
    if (opt.measure_links && rep.multi_process && rep.sm_clock_hz > 0.0) {
        nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
        char path[512];
        std::snprintf(path, sizeof(path), "%s.nvshmem", opt.bootstrap_file);
        bool boot = true;
        if (wrank == 0) {
            std::remove(path);
            if (nvshmemx_get_uniqueid(&uid) != 0) boot = false;
            if (boot) {
                char tmp[512];
                std::snprintf(tmp, sizeof(tmp), "%s.tmp", path);
                FILE* f = std::fopen(tmp, "wb");
                if (!f) boot = false;
                else {
                    boot = std::fwrite(&uid, sizeof(uid), 1, f) == 1;
                    std::fclose(f);
                    if (boot) boot = (std::rename(tmp, path) == 0);   // atomic publish
                }
            }
        } else {
            boot = false;
            for (int t = 0; t < 600 && !boot; ++t) {                  // up to 30 s
                FILE* f = std::fopen(path, "rb");
                if (f) {
                    boot = std::fread(&uid, sizeof(uid), 1, f) == 1;
                    std::fclose(f);
                }
                if (!boot) usleep(50000);
            }
        }
        nvshmemx_init_attr_t iattr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        if (boot && nvshmemx_set_attr_uniqueid_args(wrank, wsize, &uid, &iattr) == 0 &&
            nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &iattr) == 0) {
            const int me = nvshmem_my_pe(), np = nvshmem_n_pes();
            const size_t nfloats = 1u << 22;                          // 16 MiB
            float* sym = (float*)nvshmem_malloc(nfloats * sizeof(float));
            float* flg = (float*)nvshmem_malloc(sizeof(float));
            long long* dc = nullptr;
            if (sym && flg && cudaMalloc(&dc, sizeof(long long)) == cudaSuccess) {
                cudaMemset(sym, 0, nfloats * sizeof(float));
                cudaMemset(flg, 0, sizeof(float));
                nvshmem_barrier_all();

                // Latency: a genuine dependency-bearing round is a round trip, so alpha
                // is half of it -- the value goes and the acknowledgement comes back.
                //
                // Swept over every pair. Each pair runs alone, with the whole team
                // synchronised around it so the two participants are never competing with
                // another pair's traffic; the barrier counts are identical on every PE
                // whether or not it takes part, which is what keeps this from deadlocking.
                const int npairs_max = detail::report_pairs_cap();
                if (opt.sweep_peer_pairs && np >= 2 && np <= npairs_max) {
                    double* pmat = (double*)nvshmem_malloc(
                        (size_t)np * np * sizeof(double));
                    if (pmat) {
                        cudaMemset(pmat, 0, (size_t)np * np * sizeof(double));
                        nvshmem_barrier_all();
                        for (int i = 0; i < np; ++i) {
                            for (int j = i + 1; j < np; ++j) {
                                cudaMemset(flg, 0, sizeof(float));
                                nvshmem_barrier_all();
                                if (me == i || me == j) {
                                    const int other = (me == i) ? j : i;
                                    const int side  = (me == i) ? 0 : 1;
                                    long long hp = 0;
                                    detail::k_nvsh_pingpong<<<1,1>>>(flg, side, other, 64, dc);
                                    cudaDeviceSynchronize();
                                    cudaMemset(flg, 0, sizeof(float));
                                    detail::k_nvsh_pingpong<<<1,1>>>(flg, side, other,
                                                                     opt.link_rounds, dc);
                                    cudaDeviceSynchronize();
                                    if (cudaGetLastError() == cudaSuccess) {
                                        cudaMemcpy(&hp, dc, sizeof(hp), cudaMemcpyDeviceToHost);
                                        if (hp > 0 && me == i) {
                                            // half a round trip, published into PE 0's copy
                                            const double half =
                                                (double)hp / opt.link_rounds
                                                / rep.sm_clock_hz / 2.0;
                                            nvshmem_double_put(&pmat[i*np + j], &half, 1, 0);
                                            nvshmem_double_put(&pmat[j*np + i], &half, 1, 0);
                                        }
                                    }
                                }
                                nvshmem_barrier_all();
                            }
                        }
                        nvshmem_barrier_all();
                        if (me == 0) {
                            const size_t cnt = (size_t)np * np;
                            double* hm = (double*)std::calloc(cnt, sizeof(double));
                            if (hm) {
                                cudaMemcpy(hm, pmat, cnt * sizeof(double),
                                           cudaMemcpyDeviceToHost);
                                double vals[discovery_report::kMaxPeers *
                                            discovery_report::kMaxPeers];
                                int k = 0;
                                for (int i = 0; i < np; ++i)
                                    for (int j = i + 1; j < np; ++j) {
                                        const double x = hm[(size_t)i*np + j];
                                        if (x > 0.0) {
                                            vals[k++] = x;
                                            rep.onesided_pair_lat[i*np + j] = x;
                                            rep.onesided_pair_lat[j*np + i] = x;
                                        }
                                    }
                                if (k > 0) {
                                    std::sort(vals, vals + k);
                                    rep.onesided_lat_min = vals[0];
                                    rep.onesided_lat_med = vals[k/2];
                                    rep.onesided_lat_max = vals[k-1];
                                    // alpha takes the worst pair: Scrit is a longest-chain
                                    // count, and a chain runs at its slowest hop.
                                    rep.onesided_lat = vals[k-1];
                                    rep.sol_ping_pong = vals[k-1] * 2.0;
                                    rep.onesided_pairs_swept = true;
                                    rep.onesided_uniform =
                                        (vals[k-1] - vals[0]) <=
                                        opt.pair_uniform_tol * vals[k/2];
                                }
                                std::free(hm);
                            }
                        }
                        nvshmem_free(pmat);
                    }
                } else if (np >= 2 && me < 2) {
                    long long hc = 0;
                    detail::k_nvsh_pingpong<<<1,1>>>(flg, me, me ^ 1, 64, dc);
                    cudaDeviceSynchronize();
                    cudaMemset(flg, 0, sizeof(float));
                    nvshmem_barrier_all();
                    detail::k_nvsh_pingpong<<<1,1>>>(flg, me, me ^ 1, opt.link_rounds, dc);
                    cudaDeviceSynchronize();
                    if (cudaGetLastError() == cudaSuccess) {
                        cudaMemcpy(&hc, dc, sizeof(hc), cudaMemcpyDeviceToHost);
                        if (hc > 0) {
                            rep.sol_ping_pong =
                                (double)hc / opt.link_rounds / rep.sm_clock_hz;
                            rep.onesided_lat = rep.sol_ping_pong / 2.0;
                        }
                    }
                } else { nvshmem_barrier_all(); }
                nvshmem_barrier_all();

                // Bandwidth: block-scoped put across a full grid.
                {
                    long long hc = 0;
                    const int peer = (me + 1) % np, blk = 64, it = 64;
                    detail::k_nvsh_put<<<blk,512>>>(sym, sym, nfloats, peer, 4, dc);
                    cudaDeviceSynchronize();
                    nvshmem_barrier_all();
                    detail::k_nvsh_put<<<blk,512>>>(sym, sym, nfloats, peer, it, dc);
                    cudaDeviceSynchronize();
                    if (cudaGetLastError() == cudaSuccess) {
                        cudaMemcpy(&hc, dc, sizeof(hc), cudaMemcpyDeviceToHost);
                        if (hc > 0) {
                            const double sec = (double)hc / rep.sm_clock_hz;
                            rep.onesided_bw =
                                (double)nfloats * sizeof(float) * it / sec;
                        }
                    }
                }
                // Speed-of-light decomposition. Cheap, and it converts alpha_3 from a
                // bare number into a number with a floor beside it: without the bound
                // there is no way to tell a good measurement from a fast fabric.
                if (rep.sol_ping_pong > 0.0) {
                    float* snk = nullptr;
                    if (cudaMalloc(&snk, sizeof(float)) == cudaSuccess) {
                        long long fc = 0;
                        const int fit = 20000;
                        detail::k_l2_rtt<<<1,1>>>(1000, dc, snk);
                        cudaDeviceSynchronize();
                        detail::k_l2_rtt<<<1,1>>>(fit, dc, snk);
                        cudaDeviceSynchronize();
                        if (cudaGetLastError() == cudaSuccess) {
                            cudaMemcpy(&fc, dc, sizeof(fc), cudaMemcpyDeviceToHost);
                            if (fc > 0) {
                                rep.sol_l2_rtt = (double)fc / fit / rep.sm_clock_hz;
                                rep.sol_remote_store =
                                    (rep.sol_ping_pong - 2.0 * rep.sol_l2_rtt) / 2.0;
                                if (rep.sol_remote_store > 0.0)
                                    rep.sol_allreduce =
                                        rep.sol_remote_store + 2.0 * rep.sol_l2_rtt;
                            }
                        }
                        cudaFree(snk);
                    }
                }
                rep.onesided_measured =
                    (rep.onesided_lat > 0.0 && rep.onesided_bw > 0.0);
                nvshmem_barrier_all();
                cudaFree(dc);
            }
            if (sym) nvshmem_free(sym);
            if (flg) nvshmem_free(flg);
            nvshmem_finalize();
            if (wrank == 0) std::remove(path);
        }
        cudaSetDevice(dev0);
        cudaGetLastError();
    }
#endif
    {
    }

    cudaSetDevice(prev_dev);

    if (!assemble(out, rep, opt)) return false;
    if (opt.verbose) rep.print();
    return true;
}

#undef TQR_CU

// Convenience overload for callers that do not want the raw measurements. The report
// is a function-local static so machine::name stays valid; not thread-safe, and a
// second call renames the first machine.
inline bool discover(machine& out, const discovery_options& opt = discovery_options()) {
    static discovery_report scratch;
    return discover(out, scratch, opt);
}

#endif  // __CUDACC__

inline void discovery_report::print(FILE* f) const {
    if (!ok()) { std::fprintf(f, "discovery failed: %s\n", error); return; }
    std::fprintf(f, "%s  (CC %d.%d, %d SMs, %d warps/SM)\n",
                 machine_name, cc_major, cc_minor, sm_count, warps_per_sm);

    if (!converged)
        std::fprintf(f, "  WARNING: these probes never settled: %s\n", warning);
    std::fprintf(f, "  capacities (queried)\n");
    std::fprintf(f, "    register file / SM  %10.2f KiB\n", regfile_bytes_per_sm / 1024.0);
    std::fprintf(f, "    shared mem    / SM  %10.2f KiB\n", smem_bytes_per_sm / 1024.0);
    std::fprintf(f, "    L2            / GPU %10.2f MiB\n", l2_bytes / 1048576.0);
    std::fprintf(f, "    HBM           / GPU %10.2f GiB\n", hbm_bytes / 1073741824.0);
    std::fprintf(f, "    host ram      / node%10.2f GiB\n", host_ram_bytes / 1073741824.0);

    // Every measured number is a median over repeats, printed with the min-max spread
    // across those repeats. The spread is the number to look at first: it says whether
    // the median means anything.
    std::fprintf(f, "  bandwidth, aggregate (median of a settled window, +- its spread)\n");
    std::fprintf(f, "    shared mem   %10.1f GB/s  +-%5.1f%%\n",
                 smem_bw_aggregate * 1e-9, smem_bw_spread * 100.0);
    std::fprintf(f, "    L2           %10.1f GB/s  +-%5.1f%%\n",
                 l2_bw_aggregate * 1e-9, l2_bw_spread * 100.0);
    std::fprintf(f, "    HBM          %10.1f GB/s  +-%5.1f%%   (%.0f%% of the %.1f GB/s advertised)\n",
                 hbm_bw * 1e-9, hbm_bw_spread * 100.0,
                 hbm_bw_theoretical > 0.0 ? 100.0 * hbm_bw / hbm_bw_theoretical : 0.0,
                 hbm_bw_theoretical * 1e-9);
    if (links_measured) {
        std::fprintf(f, "    peer out     %10.1f GB/s  +-%5.1f%%   (%s)\n",
                     peer_bw_out * 1e-9, peer_bw_spread * 100.0,
                     p2p_all_pairs ? "all pairs have peer access"
                                   : "peer access incomplete, not used");
        std::fprintf(f, "    host         %10.1f GB/s  +-%5.1f%%\n",
                     host_bw * 1e-9, host_bw_spread * 100.0);
        std::fprintf(f, "    boundary 3 takes the %s path\n", used_peer_path ? "peer" : "host");
        std::fprintf(f, "  boundary 3 transports (alpha is technology dependent)\n");
        std::fprintf(f, "    %-22s %9.1f GB/s %10.3f us\n", "parent (host link)",
                     host_bw * 1e-9, host_lat * 1e6);
        if (peer_bw_out > 0.0)
            std::fprintf(f, "    %-22s %9.1f GB/s %10.3f us   %s%s\n",
                         peer_via_nccl ? "peer collective (NCCL)" : "peer (cudaMemcpyPeer)",
                         peer_bw_out * 1e-9, peer_lat * 1e6,
                         peer_device_initiated ? "device-initiated" : "host API",
                         peer_bw_symmetric ? ", symmetric mem" : "");
        if (peer_device_initiated && peer_lat_host > 0.0)
            std::fprintf(f, "    %-22s %9s      %10.3f us   (superseded)\n",
                         "  same, host API", "-", peer_lat_host * 1e6);
        if (onesided_bw > 0.0)
            std::fprintf(f, "    %-22s %9.1f GB/s %10.3f us   %s\n",
                         "peer one-sided", onesided_bw * 1e-9, onesided_lat * 1e6,
                         onesided_measured
                            ? (onesided_pairs_swept ? "NVSHMEM, worst of all pairs"
                                                    : "NVSHMEM, PE0<->PE1 only")
                            : "not measured");
        if (onesided_pairs_swept) {
            std::fprintf(f, "      pair sweep: min %.3f  median %.3f  max %.3f us"
                            "  (spread %.1f%%)  fabric %s\n",
                         onesided_lat_min * 1e6, onesided_lat_med * 1e6,
                         onesided_lat_max * 1e6,
                         onesided_lat_med > 0.0
                            ? 100.0 * (onesided_lat_max - onesided_lat_min) / onesided_lat_med
                            : 0.0,
                         onesided_uniform ? "uniform" : "NON-UNIFORM");
            if (!onesided_uniform && nranks <= 8) {
                std::fprintf(f, "      latency matrix (us), alpha_3 takes the worst:\n");
                for (int i = 0; i < nranks; ++i) {
                    std::fprintf(f, "        ");
                    for (int j = 0; j < nranks; ++j)
                        if (i == j) std::fprintf(f, "    .  ");
                        else std::fprintf(f, "%6.3f ",
                                          onesided_pair_lat[i*nranks + j] * 1e6);
                    std::fprintf(f, "\n");
                }
            }
        }
        else
            std::fprintf(f, "    %-22s %9s %10s   not supplied\n", "peer one-sided", "-", "-");
    }

    std::fprintf(f, "  dependent-access latency (clock64 in-kernel, full traversal)\n");
    std::fprintf(f, "    shared mem   %10.1f ns  +-%5.1f%%   %8.1f cycles\n",
                 smem_lat * 1e9, smem_lat_spread * 100.0, smem_lat_cycles);
    std::fprintf(f, "    L2           %10.1f ns  +-%5.1f%%   %8.1f cycles   (%.1f MiB footprint, fits the %.1f MiB L2)\n",
                 l2_lat * 1e9, l2_lat_spread * 100.0, l2_lat_cycles,
                 chase_near_footprint / 1048576.0, l2_bytes / 1048576.0);
    std::fprintf(f, "    HBM          %10.1f ns  +-%5.1f%%   %8.1f cycles   (%.1f MiB footprint, %.1fx the L2)\n",
                 hbm_lat * 1e9, hbm_lat_spread * 100.0, hbm_lat_cycles,
                 chase_far_footprint / 1048576.0,
                 l2_bytes ? (double)chase_far_footprint / (double)l2_bytes : 0.0);
    std::fprintf(f, "    SM clock measured during the chase: %.0f MHz\n", sm_clock_hz * 1e-6);
    if (l2_probe_blocks > 0) {
        std::fprintf(f, "    L2 structure: %d partition%s from %d serialized per-SM chases"
                        " (GMM/BIC chose %d component%s)\n",
                     l2_partitions, l2_partitions == 1 ? "" : "s", l2_probe_blocks,
                     l2_gmm_components, l2_gmm_components == 1 ? "" : "s");
        std::fprintf(f, "      per-SM latency  min %.0f  median %.0f  max %.0f cycles"
                        "  (spread %.1f%%)\n",
                     l2_near_cycles, l2_median_cycles, l2_far_cycles,
                     l2_median_cycles > 0.0
                        ? 100.0 * (l2_far_cycles - l2_near_cycles) / l2_median_cycles : 0.0);
        if (l2_partitions > 1) {
            std::fprintf(f, "      component means:");
            for (int j = 0; j < l2_partitions && j < 4; ++j)
                std::fprintf(f, " %.0f", l2_mode_mu[j]);
            std::fprintf(f, " cycles -> modelled as %d peer L2 regions\n", l2_partitions);
        } else {
            std::fprintf(f, "      one component: a continuous placement gradient, not a"
                            " partition. Spread is the honest error bar on alpha_1.\n");
        }
    }
    if (sol_allreduce > 0.0) {
        std::fprintf(f, "  speed-of-light (arXiv 2607.16100 section VI)\n");
        std::fprintf(f, "    L2 RTT %.3f us, remote store %.3f us  ->  SoL AllReduce"
                        " %.3f us\n", sol_l2_rtt * 1e6, sol_remote_store * 1e6,
                     sol_allreduce * 1e6);
        if (peer_lat_device > 0.0)
            std::fprintf(f, "    measured one-shot LL %.3f us = %.1f%% over this"
                            " machine's floor\n", peer_lat_device * 1e6,
                         100.0 * (peer_lat_device - sol_allreduce) / sol_allreduce);
    }
    if (links_measured) {
        std::fprintf(f, "    peer         %10.2f us  +-%5.1f%%\n",
                     peer_lat * 1e6, peer_lat_spread * 100.0);
        std::fprintf(f, "    host         %10.2f us  +-%5.1f%%\n",
                     host_lat * 1e6, host_lat_spread * 100.0);
    }
}

}  // namespace tqr
