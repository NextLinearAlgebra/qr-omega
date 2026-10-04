"""The machine model read from a profile.

A machine is a hierarchy of memory levels. Level l has a capacity M_l (bytes), an access latency lambda_l
(seconds for one dependent access or one message) and a bandwidth B_l (bytes per second). On top of the hierarchy
sit the arithmetic rates Gamma_a (operations per second in arithmetic a) and the latencies of the operations that
order work: a kernel launch, a block barrier, a join of two streams. A cost model is a sum of three kinds of terms,

    T = sum_l (S_l lambda_l + W_l / B_l) + F / Gamma,

with S_l messages and W_l bytes crossing level l and F arithmetic operations. Every quantity below comes from the
profile; nothing is specific to an algorithm.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

WORD = {"fp64": 8, "fp32": 4, "tf32": 4, "fp16": 2}


@dataclass(frozen=True)
class Level:
    name: str
    scope: str  # "sm", "gpu" or "node": what shares one instance of the level
    capacity: float  # bytes
    latency: float  # seconds per dependent access (or per message)
    bandwidth: float  # bytes per second with every SM of the device reading
    bandwidth_per_sm: float  # bytes per second driven by one SM


def solve(a, b):
    """Solve the small dense system a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[p][c]) < 1e-300:
            raise ArithmeticError("singular system")
        a[c], a[p] = a[p], a[c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            for k in range(c, n + 1):
                a[r][k] -= f * a[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][k] * x[k] for k in range(r + 1, n))) / a[r][r]
    return x


@dataclass(frozen=True)
class ProductLaw:
    """Time of C(m x n) += A(m x k) B(k x n): 2 m n k gamma (1 + kappa (1/k + xi (1/m + 1/n))) + call.

    gamma is the time per operation of a product long enough in every direction to be compute-bound. kappa is the
    contraction length at which the traffic of the output costs as much as the arithmetic (the rate is half of its
    peak there), and xi weighs the traffic of the two operands against that of the output.
    """

    gamma: float
    kappa: float
    xi: float
    call: float

    def time(self, m, n, k, share=1.0):
        """Seconds for one product on a share of the SMs."""
        if min(m, n, k) <= 0:
            return 0.0
        flops = 2.0 * m * n * k
        return (
            self.call
            + flops
            * self.gamma
            * (1.0 + self.kappa * (1.0 / k + self.xi * (1.0 / m + 1.0 / n)))
            / share
        )

    def rate(self, m, n, k):
        return 2.0 * m * n * k / (self.time(m, n, k) - self.call)


class Machine:
    def __init__(self, data):
        if not str(data.get("schema", "")).startswith("machine-profile/"):
            raise ValueError("not a machine profile")
        self.data = data
        d, a, b, lat = data["device"], data["access"], data["bandwidth"], data["latency"]
        self.name = d["name"]
        self.sms = int(d["sm_count"])
        self.clock = float(d["clock_hz"])
        self.warps_per_sm = int(d["max_threads_per_sm"]) // int(d["warp_size"])
        mem = b["memory"]
        shared = b["shared"]["read_bytes_per_s_per_sm"]
        l1 = b["l1"]["read_bytes_per_s_per_sm"]
        self.levels = {
            "register": Level(
                "register",
                "sm",
                4.0 * d["registers_per_sm"],
                data["arithmetic"]["simt"]["fp32"]["fma_latency_s"],
                math.inf,
                math.inf,
            ),
            "shared": Level(
                "shared",
                "sm",
                float(d["shared_bytes_per_block_optin"]),
                a["shared"]["latency_s"],
                shared * self.sms,
                shared,
            ),
            "l1": Level(
                "l1",
                "sm",
                float(a["l1"]["capacity_bytes_observed"]),
                a["l1"]["latency_s"],
                l1 * self.sms,
                l1,
            ),
            "l2": Level(
                "l2",
                "gpu",
                float(d["l2_bytes"]),
                a["l2"]["latency_s"],
                b["l2"]["read_bytes_per_s"],
                b["l2"]["read_bytes_per_s_per_sm"],
            ),
            "memory": Level(
                "memory",
                "gpu",
                float(d["memory_total_bytes"]),
                a["memory"]["latency_s"],
                mem["read_bytes_per_s"],
                mem["read_bytes_per_s_per_sm"],
            ),
        }
        peer = b.get("peer")
        if peer:
            self.levels["peer"] = Level(
                "peer",
                "node",
                float(d["memory_total_bytes"]),
                peer["latency_s"],
                peer["bandwidth_bytes_per_s"],
                peer["bandwidth_bytes_per_s"],
            )
        host = b.get("host_to_device")
        if host:
            self.levels["host"] = Level(
                "host",
                "node",
                math.inf,
                host["latency_s"],
                host["bandwidth_bytes_per_s"],
                host["bandwidth_bytes_per_s"],
            )
        self.write_bandwidth = mem["write_bytes_per_s"]
        self.copy_bandwidth = mem["copy_bytes_per_s"]  # read plus write, bytes of both counted
        self.copy_bandwidth_per_sm = mem["copy_bytes_per_s_per_sm"]
        self.memory_free = float(d["memory_free_bytes"])
        self.launch = lat["launch_async_s"]  # one more kernel on a stream the host is ahead of
        self.launch_sync = lat["launch_sync_s"]  # issue one kernel and wait for it
        self.join = lat["stream_fork_join_s"]  # fork work to a second stream and join it
        self.block = lat["block_s"]  # one more block in a grid
        self._barrier = sorted((x["threads"], x["seconds"]) for x in lat["barrier"])
        self._laws = {}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def scaled(self, name, clock=1.0, arithmetic=1.0, memory_bandwidth=1.0, sms=None):
        """A projection of this machine: SM clock, arithmetic rates (one factor or one per arithmetic) and memory
        bandwidth multiplied, optionally another SM count. Latencies that are a number of SM cycles follow the
        clock. A projection answers "what would the schedule be on such a GPU"; it is not a measurement."""
        d = json.loads(json.dumps(self.data))
        factor = (
            (lambda a: arithmetic.get(a, 1.0))
            if isinstance(arithmetic, dict)
            else (lambda a: arithmetic)
        )
        ratio = (sms / self.sms) if sms else 1.0
        d["device"].update(
            name=name,
            clock_hz=self.clock * clock,
            sm_count=sms or self.sms,
            projected_from=self.name,
        )
        for unit, v in d["arithmetic"]["simt"].items():
            f = factor(unit)
            v.update(
                flops_per_s=v["flops_per_s"] * f,
                flops_per_s_per_sm=v["flops_per_s_per_sm"] * f / ratio,
                fma_latency_s=v["fma_latency_s"] / clock,
            )
        for a, g in d["arithmetic"]["gemm"].items():
            if not isinstance(g, dict):
                continue
            f = factor(a)
            g["square"]["flops_per_s"] *= f
            g["peak_flops_per_s"] *= f
            # The arithmetic part of every sample scales with f and its traffic part with the memory bandwidth.
            gamma, kappa = g["rank_k"]["gamma_s_per_flop"], g["rank_k"]["k_half"]
            for sample in g["rank_k"]["samples"] + g.get("inner", []):
                k = sample.get("rows", sample["k"])
                traffic = kappa / (kappa + k)
                sample["flops_per_s"] = 1.0 / (
                    (1.0 - traffic) / (sample["flops_per_s"] * f)
                    + traffic / (sample["flops_per_s"] * memory_bandwidth)
                )
            g["rank_k"]["gamma_s_per_flop"] = gamma / f
            g["rank_k"]["k_half"] = kappa * f / memory_bandwidth
        for level in ("shared", "l1", "l2"):
            d["access"][level]["latency_s"] /= clock
        for key in ("read_bytes_per_s_per_sm",):
            d["bandwidth"]["shared"][key] *= clock
            d["bandwidth"]["l1"][key] *= clock
        mem = d["bandwidth"]["memory"]
        for key in ("read_bytes_per_s", "write_bytes_per_s", "copy_bytes_per_s"):
            mem[key] *= memory_bandwidth
            mem[key + "_per_sm"] *= memory_bandwidth / ratio
        for b in d["latency"]["barrier"]:
            b["seconds"] /= clock
        return Machine(d)

    # ---- hierarchy --------------------------------------------------------------------------------------------------
    def capacity(self, level):
        return self.levels[level].capacity

    def latency(self, level):
        return self.levels[level].latency

    def bandwidth(self, level="memory", sms=None, kind="read"):
        """Bytes per second through a level when `sms` SMs drive it (all of them by default)."""
        lv = self.levels[level]
        if level == "memory" and kind != "read":
            device = self.write_bandwidth if kind == "write" else self.copy_bandwidth
            per_sm = self.copy_bandwidth_per_sm if kind == "copy" else device / self.sms
        else:
            device, per_sm = lv.bandwidth, lv.bandwidth_per_sm
        return device if sms is None else min(device, sms * per_sm)

    def stream(self, nbytes, level="memory", sms=None, kind="read"):
        """Seconds to move nbytes through a level."""
        return nbytes / self.bandwidth(level, sms, kind)

    def fits(self, level, nbytes):
        return nbytes <= self.levels[level].capacity

    def tile(self, level, word, operands=3):
        """Side of the largest square tile of which `operands` copies are resident in a level."""
        return int(math.sqrt(self.levels[level].capacity / (operands * word)))

    def barrier(self, threads):
        """Seconds for one block-wide barrier."""
        for t, s in self._barrier:
            if threads <= t:
                return s
        return self._barrier[-1][1]

    # ---- arithmetic -------------------------------------------------------------------------------------------------
    def arithmetics(self):
        return [k for k, v in self.data["arithmetic"]["gemm"].items() if isinstance(v, dict)]

    def scalar_rate(self, arithmetic, sms=None):
        """Operations per second of scalar (non-tensor) fused multiply-adds held in registers."""
        s = self.data["arithmetic"]["simt"]["fp64" if arithmetic == "fp64" else "fp32"]
        return (
            s["flops_per_s"]
            if sms is None
            else min(s["flops_per_s"], sms * s["flops_per_s_per_sm"])
        )

    def peak(self, arithmetic):
        """Operations per second of the best measured matrix product."""
        return self.data["arithmetic"]["gemm"][arithmetic]["peak_flops_per_s"]

    def product(self, arithmetic):
        """The product law of an arithmetic, fitted to the shapes the probe measured."""
        if arithmetic in self._laws:
            return self._laws[arithmetic]
        g = self.data["arithmetic"]["gemm"][arithmetic]
        samples = [
            (g["square"]["n"], g["square"]["n"], g["square"]["n"], g["square"]["flops_per_s"])
        ]
        samples += [
            (g["rank_k"]["m"], g["rank_k"]["n"], s["k"], s["flops_per_s"])
            for s in g["rank_k"]["samples"]
        ]
        samples += [(s["k"], s["q"], s["rows"], s["flops_per_s"]) for s in g.get("inner", [])]
        # 1/rate = gamma + (gamma kappa)/k + (gamma kappa xi)(1/m + 1/n), least squares on relative errors.
        rows = [([r, r / k, r * (1.0 / m + 1.0 / n)], 1.0) for m, n, k, r in samples]
        ata = [[sum(x[i] * x[j] for x, _ in rows) for j in range(3)] for i in range(3)]
        atb = [sum(x[i] * y for x, y in rows) for i in range(3)]
        law = None
        try:
            gamma, gk, gkx = solve(ata, atb)
            if gamma > 0 and gk > 0 and gkx >= 0:
                law = ProductLaw(
                    gamma, gk / gamma, gkx / gk, g.get("call_latency_s", self.launch_sync)
                )
        except ArithmeticError:
            pass
        if law is None:  # too few shapes for the operand term: keep the probe's rank-k fit
            law = ProductLaw(
                g["rank_k"]["gamma_s_per_flop"],
                g["rank_k"]["k_half"],
                0.25,
                g.get("call_latency_s", self.launch_sync),
            )
        self._laws[arithmetic] = law
        return law

    # ---- dimensionless numbers --------------------------------------------------------------------------------------
    def balance(self, arithmetic):
        """Operations the device performs in the time one byte crosses device memory: below this arithmetic
        intensity a kernel is bound by bandwidth, above it by arithmetic."""
        return self.peak(arithmetic) / self.copy_bandwidth

    def grain(self, arithmetic):
        """Operations forgone during one kernel launch: a product smaller than this is bound by latency."""
        return self.peak(arithmetic) * self.launch_sync

    def summary(self):
        """The basic metrics as a flat dictionary (SI units)."""
        out = {
            "name": self.name,
            "sms": self.sms,
            "clock_hz": self.clock,
            "launch_s": self.launch,
            "launch_sync_s": self.launch_sync,
            "join_s": self.join,
            "barrier_s": self.barrier(1024),
            "memory_write_bytes_per_s": self.write_bandwidth,
            "memory_copy_bytes_per_s": self.copy_bandwidth,
        }
        for name, lv in self.levels.items():
            out[f"{name}_capacity_bytes"] = lv.capacity
            out[f"{name}_latency_s"] = lv.latency
            out[f"{name}_bytes_per_s"] = lv.bandwidth
            out[f"{name}_bytes_per_s_per_sm"] = lv.bandwidth_per_sm
        for a in self.arithmetics():
            law = self.product(a)
            out[f"{a}_peak_flops_per_s"] = self.peak(a)
            out[f"{a}_gamma_s_per_flop"] = law.gamma
            out[f"{a}_kappa"] = law.kappa
            out[f"{a}_xi"] = law.xi
            out[f"{a}_balance_flops_per_byte"] = self.balance(a)
            out[f"{a}_grain_flops"] = self.grain(a)
        for a in ("fp32", "fp64"):
            out[f"{a}_scalar_flops_per_s"] = self.scalar_rate(a)
        return out
