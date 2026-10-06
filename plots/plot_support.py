"""Shared data lookup and axes for the paper figures."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

HERE = Path(__file__).resolve().parent
MODES = ["fp64", "fp32", "tf32", "3xtf32"]
TITLES = {"fp64": "FP64", "fp32": "FP32", "tf32": "TF32", "3xtf32": "3×TF32"}
LIBS = ["TQR", "cuSOLVER", "MAGMA", "SLATE"]
REFS = LIBS[1:]
MARKERS = {"TQR": "o", "cuSOLVER": "s", "MAGMA": "^", "SLATE": "D"}


def read_rows(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in ["gpus", "m", "n"]:
            row[k] = int(row[k])
        for k in ["time_s", "tflops", "residual", "orthogonality"]:
            row[k] = float(row[k]) if row.get(k) else None
    return rows


def best_rows(rows, accuracy=False):
    best = {}
    for row in rows:
        if accuracy and (
            row["kind"] != "qr"
            or row["validation"] != "passed"
            or not row["residual"]
            or not row["orthogonality"]
        ):
            continue
        key = row["kind"], row["gpus"], row["precision"], row["n"], row["library"]
        if key not in best or row["time_s"] < best[key]["time_s"]:
            best[key] = row
    return best


def lookup(best, p, mode, n, lib="TQR", kind="qr"):
    precision = mode if lib == "TQR" or kind == "gemm" else "fp64" if mode == "fp64" else "fp32"
    return best.get((kind, p, precision, n, lib))


def name(lib, p):
    if lib == "TQR":
        return "QR-Ω"
    if lib == "cuSOLVER" and p > 1:
        return "cuSOLVERMp"
    return lib


def fmt_n(n, pos=None):
    return f"{int(n):,}" if n < 1000 else f"{n / 1000:.3g}k"


def available(best, p, mode, lib="TQR", kind="qr"):
    return sorted(
        k[3]
        for k in best
        if k[0] == kind
        and k[1] == p
        and k[2]
        == (mode if lib == "TQR" or kind == "gemm" else "fp64" if mode == "fp64" else "fp32")
        and k[4] == lib
    )


class Renderer:
    def __init__(self, best, acc, highlights, scaling, cfg, out):
        self.best, self.acc = best, acc
        self.highlights, self.scaling, self.cfg, self.out = (
            highlights,
            scaling,
            cfg,
            out,
        )
        self.colors = cfg["colors"]
        self.files, self.chart_data = [], []
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 13,
                "axes.titleweight": "bold",
                "axes.labelsize": 10,
                "xtick.labelsize": 8.7,
                "ytick.labelsize": 9,
                "legend.fontsize": 10,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.edgecolor": "#9AA5AC",
                "text.color": "#1C252C",
                "axes.labelcolor": "#1C252C",
            }
        )

    def style_axis(self, ax, p, mode, metric):
        ax.set_title(TITLES[mode], loc="left", pad=9)
        ax.set_xscale("log", base=2)
        sizes = sorted(set(n for lib in LIBS for n in available(self.best, p, mode, lib)))
        if p == 1:
            ticks = [256, 1024, 4096, 16384, 65536]
        else:
            ticks = [60000, 98304, 131072, 196608, 262144, 327680]
        if sizes:
            ticks = [x for x in ticks if min(sizes) <= x <= max(sizes)]
            if max(sizes) not in ticks and (not ticks or max(sizes) / ticks[-1] > 1.3):
                ticks.append(max(sizes))
            ax.set_xlim(min(sizes) / 1.10, max(sizes) * 1.12)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_n))
        ax.set_xlabel("Matrix order n (n × n)")
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", color="#DEE3E7", lw=0.7)
        ax.set_axisbelow(True)
        labels = {
            "time_s": "QR-Ω speedup over reference (×)",
            "residual": "Relative residual (block maximum)",
            "orthogonality": "Relative Q·Qᵀ inverse error",
        }
        ax.set_ylabel(labels[metric])
        if metric == "time_s":
            # Integer ticks (1, 2, 3, ...) on the log axis; exact tick set
            # is fixed in curve() once the data limits are known.
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

    def curve(self, ax, p, mode, metric, record=True):
        data = self.best if metric == "time_s" else self.acc
        self.style_axis(ax, p, mode, metric)
        if metric == "time_s":
            # Linear speedup curves: ref_time / TQR_time per reference.
            # Linear-friendly (bounded near 1×) where raw seconds span 5 orders.
            vals = []
            for lib in REFS:
                common = sorted(
                    set(available(data, p, mode, "TQR")) & set(available(data, p, mode, lib))
                )
                xs, ys, flags = [], [], []
                for n in common:
                    own = lookup(data, p, mode, n, "TQR")
                    ref = lookup(data, p, mode, n, lib)
                    if not own or not ref or not own.get("time_s") or not ref.get("time_s"):
                        continue
                    if own["time_s"] <= 0 or ref["time_s"] <= 0:
                        continue
                    s = ref["time_s"] / own["time_s"]
                    xs.append(n)
                    ys.append(s)
                    vals.append(s)
                    flags.append(
                        own["validation"] == "timing_only" or ref["validation"] == "timing_only"
                    )
                    if record:
                        d = dict(own)
                        d.update(
                            chart=f"p{p}-speedup",
                            display_precision=mode,
                            plotted_value=s,
                            speedup_vs=lib,
                            reference_time_s=ref["time_s"],
                        )
                        self.chart_data.append(d)
                if not xs:
                    continue
                ax.plot(xs, ys, color=self.colors[lib], lw=1.8)
                for x, y, open_ in zip(xs, ys, flags):
                    ax.plot(
                        x,
                        y,
                        marker=MARKERS[lib],
                        ms=4,
                        markerfacecolor="white" if open_ else self.colors[lib],
                        markeredgecolor=self.colors[lib],
                        markeredgewidth=1,
                        zorder=4,
                    )
            ax.axhline(1.0, color="#9AA5AC", lw=0.9, ls="--", zorder=1)
            if vals:
                ax.set_ylim(
                    min(0.85, min(vals) * 0.92),
                    max(vals) * 1.25 if max(vals) > 1 else 1.3,
                )
                top = ax.get_ylim()[1]
                if top <= 12:
                    # Show every other integer (1, 3, 5, ...) so labels
                    # have room on the log axis; capped at 10.
                    ticks = list(np.arange(1, min(math.floor(top), 10) + 1, 2))
                    if ticks[-1] == 9 and top >= 10:
                        ticks[-1] = 10
                else:
                    step = 2 if top <= 25 else 5
                    ticks = np.arange(1, math.floor(top) + 1, step)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
            return
        vals = []
        for lib in LIBS:
            points = [lookup(data, p, mode, n, lib) for n in available(data, p, mode, lib)]
            points = [r for r in points if r.get(metric) is not None and r[metric] > 0]
            if not points:
                continue
            xs, ys = [r["n"] for r in points], [r[metric] for r in points]
            vals += ys
            ax.plot(
                xs,
                ys,
                color=self.colors[lib],
                lw=2.1 if lib == "TQR" else 1.4,
                linestyle="-" if lib == "TQR" else "--",
                alpha=1 if lib == "TQR" else 0.92,
            )
            for r in points:
                ax.plot(
                    r["n"],
                    r[metric],
                    marker=MARKERS[lib],
                    ms=4.8 if lib == "TQR" else 4,
                    markerfacecolor="white"
                    if r["validation"] == "timing_only"
                    else self.colors[lib],
                    markeredgecolor=self.colors[lib],
                    markeredgewidth=1,
                    zorder=4,
                )
                if record:
                    self.chart_data.append(
                        dict(
                            chart=f"p{p}-{metric}",
                            display_precision=mode,
                            plotted_value=r[metric],
                            **r,
                        )
                    )
        if vals:
            if max(vals) / min(vals) < 5:
                ax.set_ylim(min(vals) / 1.8, max(vals) * 1.8)
            else:
                ax.set_ylim(min(vals) / 1.6, max(vals) * 2)

    def gemm_for(self, p, mode, n):
        # No cross-size extrapolation. The x3 fallback is labelled as an FP32
        # comparator on the bar and in the caption; it is not an x3 ceiling.
        row = lookup(self.best, p, mode, n, "GEMM", "gemm")
        if row is None and p > 1 and mode == "3xtf32":
            row = lookup(self.best, p, "fp32", n, "GEMM", "gemm")
        return row
