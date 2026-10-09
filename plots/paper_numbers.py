"""Every number the paper quotes, recomputed from the measurements.

Each claim pairs a value as the paper prints it with the same quantity computed from the data and
printed the same way. Values are plain ASCII: ranges as "a-b", lists as "a, b", and a shared power of
ten as a suffix ("5-11e-6"). A claim is "exact" when both print alike, "rounding" when every number
of the data is within one unit of the paper's last printed digit, and "differs" otherwise.
"""

import math
import re
import statistics
from dataclasses import dataclass

import measurements as ms

FP64_PEAK_TFLOPS = 67  # H200 FP64 tensor-core peak, as quoted in the paper
CUTOFF_SMALL = 32000  # "below n of about 32,000" on one GPU
STRONG_SCALING_N = 229376
TALL_REFERENCE_ROWS = 131072


def number(x, digits=1):
    return f"{x:.{digits}f}"


def span(values, digits=1):
    low, high = number(min(values), digits), number(max(values), digits)
    return low if low == high else f"{low}-{high}"


def scaled(values, exponent, digits=1):
    """A value or range in units of 10^exponent: scaled([2.9e-15, 4.0e-15], -15) is '2.9-4.0e-15'."""
    return f"{span([v / 10**exponent for v in values], digits)}e{exponent}"


def power(x, digits=1):
    """A value in scientific notation: power(1.31e-14) is '1.3e-14'."""
    exponent = math.floor(math.log10(x))
    if float(number(x / 10**exponent, digits)) >= 10:
        exponent += 1
    return scaled([x], exponent, digits)


def percent(x):
    return f"{round(100 * x)}%"


def _numbers(text):
    """The numbers of a printed value and the unit of each one's last digit."""
    text = text.replace(",", "").replace("%", "")
    match = re.fullmatch(r"(.*?)e(-?\d+)", text)
    body, scale = (match.group(1), 10.0 ** int(match.group(2))) if match else (text, 1.0)
    tokens = re.findall(r"\d+(?:\.\d+)?", body)
    return [(float(t) * scale, 10.0 ** -len(t.partition(".")[2]) * scale) for t in tokens]


@dataclass(frozen=True)
class Claim:
    section: str
    statement: str
    paper: str
    data: str

    @property
    def status(self) -> str:
        if self.paper == self.data:
            return "exact"
        paper, data = _numbers(self.paper), _numbers(self.data)
        close = len(paper) == len(data) and all(
            abs(a - b) <= unit * (1 + 1e-9) for (a, unit), (b, _) in zip(paper, data, strict=True)
        )
        return "rounding" if close else "differs"


class Results:
    """The fastest tuned configuration of every library and size, as the figures use it."""

    def __init__(self, rows):
        square = [row for row in rows if row["m"] == row["n"]]
        self.best = ms.fastest(square)
        self.checked = ms.fastest(square, checked_only=True)
        self.tall = {
            (row["library"], row["precision"], row["m"], row["n"]): row
            for row in rows
            if row["kind"] == "qr" and row["m"] > row["n"]
        }

    def row(self, gpus, mode, n, lib=ms.OWN, kind="qr"):
        return ms.lookup(self.best, gpus, mode, n, lib, kind)

    def sizes(self, gpus, mode="fp64", lib=ms.OWN):
        return ms.available(self.best, gpus, mode, lib)

    def speedup(self, gpus, mode, n, lib):
        """QR-Omega's speedup over a reference (its FP32 for TF32 and 3xTF32); None if unmeasured."""
        own, ref = self.row(gpus, mode, n), self.row(gpus, mode, n, lib)
        return ref["time_s"] / own["time_s"] if own and ref else None

    def speedups(self, gpus, modes, libs, sizes=None):
        values = (
            self.speedup(gpus, mode, n, lib)
            for mode in modes
            for lib in libs
            for n in (sizes if sizes is not None else self.sizes(gpus, mode))
        )
        return [v for v in values if v is not None]

    def fastest_reference(self, gpus, mode, n):
        refs = (self.row(gpus, mode, n, lib) for lib in ms.REFS)
        return min((r for r in refs if r), key=lambda r: r["time_s"])

    def error(self, gpus, mode, n, metric, lib=ms.OWN):
        row = ms.lookup(self.checked, gpus, mode, n, lib)
        return row[metric] if row else None

    def errors(self, gpus, mode, metric, lib=ms.OWN):
        values = (self.error(gpus, mode, n, metric, lib) for n in self.sizes(gpus, mode))
        return [v for v in values if v is not None]

    def most_accurate_reference(self, gpus, mode, n, metric):
        return min(e for e in (self.error(gpus, mode, n, metric, lib) for lib in ms.REFS) if e)

    def gemm(self, mode, n):
        return self.row(1, mode, n, "GEMM", "gemm")

    def gemm_fraction(self, mode, n, row=None):
        return (row or self.row(1, mode, n))["tflops"] / self.gemm(mode, n)["tflops"]


def abstract(r):
    single, big = ("fp64", "fp32"), 131072
    residual = r.error(1, "fp64", big, "residual")
    vendor = min(r.error(1, "fp64", big, "residual", lib) for lib in ("cuSOLVER", "MAGMA"))
    return [
        Claim(
            "Abstract",
            "largest speedup over cuSOLVER, one GPU (FP64, FP32)",
            "2.2",
            number(max(r.speedups(1, single, ["cuSOLVER"]))),
        ),
        Claim(
            "Abstract",
            "largest speedup over MAGMA and SLATE, one GPU (FP64, FP32)",
            "11.9",
            number(max(r.speedups(1, single, ["MAGMA", "SLATE"]))),
        ),
        Claim(
            "Abstract",
            "FP64 residual at n = 131,072 against cuSOLVER and MAGMA (about)",
            "60%",
            f"{10 * round(10 * residual / vendor)}%",
        ),
        Claim(
            "Abstract", "TF32 throughput at n = 131,072 (TFLOP/s)", "249", number(r.row(1, "tf32", big)["tflops"], 0)
        ),
        Claim(
            "Abstract", "3xTF32 throughput at n = 131,072 (TFLOP/s)", "94", number(r.row(1, "3xtf32", big)["tflops"], 0)
        ),
        Claim(
            "Abstract",
            "speedup over cuSOLVERMp on eight GPUs (FP64, FP32)",
            "1.02-1.12",
            span(r.speedups(8, single, ["cuSOLVER"]), 2),
        ),
        Claim(
            "Abstract",
            "TF32 throughput at n = 327,680 on eight GPUs (TFLOP/s)",
            "1327",
            number(r.row(8, "tf32", 327680)["tflops"], 0),
        ),
    ]


def motivation(r):
    cusolver = {n: r.row(1, "fp64", n, "cuSOLVER") for n in r.sizes(1, lib="cuSOLVER")}
    growth = math.log(cusolver[4096]["time_s"] / cusolver[256]["time_s"]) / math.log(4096 / 256)
    transition = [
        r.gemm_fraction("fp64", n, r.fastest_reference(1, "fp64", n))
        for n in cusolver
        if 4096 <= n <= 65536 and r.gemm("fp64", n)
    ]
    dgemm = r.gemm("fp64", 1024)["tflops"]
    return [
        Claim(
            "III-B",
            "cuSOLVER as a fraction of DGEMM at n = 80,000",
            "77%",
            percent(r.gemm_fraction("fp64", 80000, cusolver[80000])),
        ),
        Claim("III-B", "DGEMM at n = 1,024 (TFLOP/s)", "53", number(dgemm, 0)),
        Claim("III-B", "DGEMM at n = 1,024 as a fraction of the FP64 peak", "79%", percent(dgemm / FP64_PEAK_TFLOPS)),
        Claim("III-B", "cuSOLVER at n = 1,024 (TFLOP/s)", "0.35", number(cusolver[1024]["tflops"], 2)),
        Claim("III-B", "growth exponent of cuSOLVER's time up to n = 4,096", "1.3", number(growth)),
        Claim(
            "III-B",
            "fastest library as a fraction of DGEMM, n = 4,096 to 65,536",
            "7-73%",
            f"{span(transition and [100 * f for f in transition], 0)}%",
        ),
    ]


def one_gpu(r):
    single, big = ("fp64", "fp32"), 131072
    small = [n for n in r.sizes(1) if n < CUTOFF_SMALL]
    accuracy = [
        r.error(1, mode, n, metric) / r.most_accurate_reference(1, mode, n, metric)
        for mode in single
        for n in r.sizes(1, mode)
        for metric in ("residual", "orthogonality")
    ]
    x3_accuracy = [
        r.error(1, "3xtf32", n, "residual") / r.most_accurate_reference(1, "3xtf32", n, "residual")
        for n in r.sizes(1, "3xtf32")
    ]
    tf32_errors = r.errors(1, "tf32", "residual") + r.errors(1, "tf32", "orthogonality")
    rows = {mode: r.row(1, mode, big) for mode in ms.MODES}

    def x3_below_vendors_from():
        sizes = r.sizes(1, "3xtf32")
        below = [
            all(
                r.error(1, "3xtf32", n, "residual") < r.error(1, "3xtf32", n, "residual", lib)
                for lib in ("cuSOLVER", "MAGMA")
            )
            for n in sizes
        ]
        return f"{sizes[next(i for i in range(len(sizes)) if all(below[i:]))]:,}"

    def vendor_min(metric):
        return min(r.error(1, "fp64", big, metric, lib) for lib in ("cuSOLVER", "MAGMA"))

    def over_fastest(mode):
        return number(r.fastest_reference(1, mode, big)["time_s"] / rows[mode]["time_s"])

    magma80 = r.row(1, "fp32", 80000, "MAGMA")["time_s"]
    return [
        Claim(
            "V-B",
            "MAGMA's FP32 advantage at n = 80,000",
            "2%",
            percent(r.row(1, "fp32", 80000)["time_s"] / magma80 - 1),
        ),
        Claim(
            "V-B",
            "speedup over cuSOLVER below n = 32,000 (FP64, FP32)",
            "1.4-2.2",
            span(r.speedups(1, single, ["cuSOLVER"], small)),
        ),
        Claim(
            "V-B",
            "speedup over MAGMA and SLATE below n = 32,000 (FP64, FP32)",
            "1.4-11.9",
            span(r.speedups(1, single, ["MAGMA", "SLATE"], small)),
        ),
        Claim(
            "V-B", "fraction of the GEMM rate at n = 80,000 (FP64)", "0.82", number(r.gemm_fraction("fp64", 80000), 2)
        ),
        Claim(
            "V-B", "fraction of the GEMM rate at n = 80,000 (FP32)", "0.88", number(r.gemm_fraction("fp32", 80000), 2)
        ),
        Claim("V-B", "time at n = 131,072, FP64 (s)", "63.6", number(rows["fp64"]["time_s"])),
        Claim("V-B", "time at n = 131,072, FP32 (s)", "64.6", number(rows["fp32"]["time_s"])),
        Claim("V-B", "throughput at n = 131,072, FP64 (TFLOP/s)", "47.2", number(rows["fp64"]["tflops"])),
        Claim("V-B", "throughput at n = 131,072, FP32 (TFLOP/s)", "46.5", number(rows["fp32"]["tflops"])),
        Claim("V-B", "cuSOLVER at n = 131,072, FP64 (s)", "64.2", number(r.row(1, "fp64", big, "cuSOLVER")["time_s"])),
        Claim("V-B", "MAGMA at n = 131,072, FP32 (s)", "64.5", number(r.row(1, "fp32", big, "MAGMA")["time_s"])),
        Claim("V-B", "TF32 time at n = 131,072 (s)", "12.1", number(rows["tf32"]["time_s"])),
        Claim("V-B", "TF32 throughput at n = 131,072 (TFLOP/s)", "249", number(rows["tf32"]["tflops"], 0)),
        Claim("V-B", "TF32 speedup over the fastest FP32 reference at n = 131,072", "5.3", over_fastest("tf32")),
        Claim(
            "V-B",
            "TF32 fraction of the TF32 GEMM rate at n = 80,000",
            "0.65",
            number(r.gemm_fraction("tf32", 80000), 2),
        ),
        Claim("V-B", "3xTF32 throughput at n = 131,072 (TFLOP/s)", "94", number(rows["3xtf32"]["tflops"], 0)),
        Claim("V-B", "3xTF32 speedup over the fastest FP32 reference at n = 131,072", "2.0", over_fastest("3xtf32")),
        Claim(
            "V-B", "3xTF32 GEMM of CUTLASS at n = 80,000 (TFLOP/s)", "63", number(r.gemm("3xtf32", 80000)["tflops"], 0)
        ),
        Claim(
            "V-B",
            "FP64 and FP32 errors within this factor of the most accurate reference",
            "3.7",
            number(max(accuracy)),
        ),
        Claim("V-B", "FP64 residual at n = 131,072", "1.3e-14", power(r.error(1, "fp64", big, "residual"))),
        Claim("V-B", "the lower of cuSOLVER's and MAGMA's FP64 residual", "2.2e-14", power(vendor_min("residual"))),
        Claim(
            "V-B", "FP64 orthogonality error at n = 131,072", "3.1e-15", power(r.error(1, "fp64", big, "orthogonality"))
        ),
        Claim(
            "V-B",
            "the lower of cuSOLVER's and MAGMA's FP64 orthogonality error",
            "5.5e-15",
            power(vendor_min("orthogonality")),
        ),
        Claim(
            "V-B",
            "SLATE's FP64 residual at n = 131,072",
            "4.3e-15",
            power(r.error(1, "fp64", big, "residual", "SLATE")),
        ),
        Claim("V-B", "TF32 errors", "1.0-5.5e-3", scaled(tf32_errors, -3)),
        Claim(
            "V-B",
            "3xTF32 residual within this factor of the most accurate FP32 reference",
            "9",
            number(max(x3_accuracy), 0),
        ),
        Claim("V-B", "3xTF32 residual below cuSOLVER's and MAGMA's from n", "80,000", x3_below_vendors_from()),
    ]


def tall_skinny(r):
    own = {key: row for key, row in r.tall.items() if key[0] == ms.OWN}
    fp64 = {n: [row["tflops"] for (_, mode, _, k), row in own.items() if mode == "fp64" and k == n] for n in (64, 4096)}

    def speedups(libs):
        values = []
        for (lib, precision, m, n), ref in r.tall.items():
            if lib in libs and m == TALL_REFERENCE_ROWS:
                modes = ("fp64",) if precision == "fp64" else ("fp32", "tf32", "3xtf32")
                values += [ref["time_s"] / own[(ms.OWN, mode, m, n)]["time_s"] for mode in modes]
        return values

    others = speedups(["MAGMA", "SLATE"])
    return [
        Claim("V-B", "tall-and-skinny shapes", "11", str(len({(m, n) for (_, _, m, n) in own}))),
        Claim("V-B", "FP64 tall-and-skinny throughput at n = 64 (TFLOP/s)", "1.2-1.6", span(fp64[64])),
        Claim("V-B", "FP64 tall-and-skinny throughput at n = 4,096 (TFLOP/s)", "18-23", span(fp64[4096], 0)),
        Claim(
            "V-B",
            "TF32 highest tall-and-skinny throughput (TFLOP/s)",
            "46",
            number(max(row["tflops"] for (_, mode, _, _), row in own.items() if mode == "tf32"), 0),
        ),
        Claim("V-B", "speedup over cuSOLVER at m = 131,072", "1.1-3.3", span(speedups(["cuSOLVER"]))),
        Claim(
            "V-B",
            "speedup over MAGMA and SLATE at m = 131,072",
            "1.6-83",
            f"{number(min(others))}-{number(max(others), 0)}",
        ),
    ]


def eight_gpus(r):
    def throughput(mode):
        return [r.row(8, mode, n)["tflops"] for n in r.sizes(8, mode)]

    def over_fastest(mode):
        return [r.fastest_reference(8, mode, n)["time_s"] / r.row(8, mode, n)["time_s"] for n in r.sizes(8, mode)]

    fp64_residual = r.errors(8, "fp64", "residual")
    fp64_orthogonality = r.errors(8, "fp64", "orthogonality")
    fp32_residuals = [e for lib in (ms.OWN, "cuSOLVER", "MAGMA") for e in r.errors(8, "fp32", "residual", lib)]
    one_gpu_residual = r.error(1, "fp64", 131072, "residual")
    return [
        Claim("V-C", "speedup over cuSOLVERMp, FP64", "1.02-1.06", span(r.speedups(8, ["fp64"], ["cuSOLVER"]), 2)),
        Claim("V-C", "speedup over cuSOLVERMp, FP32", "1.11-1.12", span(r.speedups(8, ["fp32"], ["cuSOLVER"]), 2)),
        Claim("V-C", "speedup over MAGMA, FP64", "1.7-2.6", span(r.speedups(8, ["fp64"], ["MAGMA"]))),
        Claim("V-C", "speedup over SLATE, FP64", "2.6-4.6", span(r.speedups(8, ["fp64"], ["SLATE"]))),
        Claim("V-C", "FP64 throughput (TFLOP/s)", "273-317", span(throughput("fp64"), 0)),
        Claim("V-C", "FP64 throughput per GPU (TFLOP/s)", "34-40", span([t / 8 for t in throughput("fp64")], 0)),
        Claim("V-C", "TF32 throughput (TFLOP/s)", "903-1327", span(throughput("tf32"), 0)),
        Claim("V-C", "TF32 speedup over the fastest FP32 reference", "3.5-4.2", span(over_fastest("tf32"))),
        Claim("V-C", "3xTF32 throughput (TFLOP/s)", "493-633", span(throughput("3xtf32"), 0)),
        Claim("V-C", "3xTF32 speedup over the fastest FP32 reference", "1.9-2.0", span(over_fastest("3xtf32"))),
        Claim("V-C", "FP64 residual at n = 131,072", "9.4e-15", power(fp64_residual[0])),
        Claim("V-C", "FP64 residual at n = 294,912", "1.4e-14", power(fp64_residual[-1])),
        Claim(
            "V-C",
            "FP64 orthogonality error, n = 131,072 to 294,912",
            "2.9-4.0e-15",
            scaled([fp64_orthogonality[0], fp64_orthogonality[-1]], -15),
        ),
        Claim("V-C", "SLATE's FP64 residual", "4e-15", scaled(r.errors(8, "fp64", "residual", "SLATE"), -15, 0)),
        Claim("V-C", "FP32 residuals of QR-Omega, cuSOLVERMp and MAGMA", "5-11e-6", scaled(fp32_residuals, -6, 0)),
        Claim(
            "V-C",
            "TF32 residual (median over the sizes)",
            "1.5e-3",
            scaled([statistics.median(r.errors(8, "tf32", "residual"))], -3),
        ),
        Claim("V-C", "3xTF32 residual", "5.3-5.7e-6", scaled(r.errors(8, "3xtf32", "residual"), -6)),
        Claim(
            "VII",
            "eight-GPU FP64 residual over the one-GPU one at n = 131,072, at most",
            "2",
            number(math.ceil(max(fp64_residual) / one_gpu_residual), 0),
        ),
    ]


def strong_scaling(r, n=STRONG_SCALING_N):
    def speedup(mode, gpus, lib=ms.OWN):
        return r.row(2, mode, n, lib)["time_s"] / r.row(gpus, mode, n, lib)["time_s"]

    def from_two(mode):
        return ", ".join(number(speedup(mode, p), 2) for p in (3, 4, 8))

    margins = [r.speedup(p, "fp32", n, "cuSOLVER") for p in (2, 3, 4, 8)]
    return [
        Claim("V-C", "FP32 speedup from two to three, four and eight GPUs", "1.48, 2.01, 3.74", from_two("fp32")),
        Claim("V-C", "3xTF32 speedup from two to three, four and eight GPUs", "1.46, 2.01, 3.41", from_two("3xtf32")),
        Claim("V-C", "cuSOLVERMp speedup from two to eight GPUs", "3.74", number(speedup("fp32", 8, "cuSOLVER"), 2)),
        Claim("V-C", "MAGMA speedup from two to eight GPUs", "3.58", number(speedup("fp32", 8, "MAGMA"), 2)),
        Claim("V-C", "FP32 margin over cuSOLVERMp", "1.12-1.14", span(margins, 2)),
        Claim("V-C", "SLATE speedup from two to eight GPUs", "3.15", number(speedup("fp32", 8, "SLATE"), 2)),
        Claim("V-C", "TF32 speedup from two to eight GPUs", "2.52", number(speedup("tf32", 8), 2)),
    ]


def claims(rows):
    """The paper's quoted numbers, in the order of the paper, against the measurements in rows."""
    r = Results(rows)
    return abstract(r) + motivation(r) + one_gpu(r) + tall_skinny(r) + eight_gpus(r) + strong_scaling(r)
