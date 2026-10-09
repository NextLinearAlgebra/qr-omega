"""The figures of the paper, drawn from the table of measurements (measurements.py).

Each figure has the width at which the paper prints it (IEEEtran two-column layout) and no text
smaller than 4 pt there. TF32 and 3xTF32 are compared with the references' FP32 throughout.
"""

import math
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import measurements as ms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullFormatter,
    NullLocator,
)

TEXT_WIDTH = 7.16  # IEEEtran two-column \textwidth, inches
COLUMN_WIDTH = 3.45  # IEEEtran \columnwidth, inches
FS = 7  # base font size, points
COLORS = {
    ms.OWN: "#006D8F",
    "cuSOLVER": "#616D7C",
    "MAGMA": "#9968AC",
    "SLATE": "#D78B32",
    "GEMM": "#A4B9C0",
}
MARKERS = {ms.OWN: "o", "cuSOLVER": "s", "MAGMA": "^", "SLATE": "D"}
TITLES = {"fp64": "FP64", "fp32": "FP32", "tf32": "TF32", "3xtf32": "3×TF32"}
LIBS = (ms.OWN, *ms.REFS)
BAR_SIZE = {1: 65536, 8: 131072}  # the order of the throughput bars of each GPU count
STRONG_SCALING_N = 229376
TALL_COLUMNS = (64, 256, 1024, 4096)
TALL_ROWS = {131072: ("131k", 0.45), 1048576: ("1.05M", 0.72), 4194304: ("4.19M", 1.0)}  # label, shade
FP64_PEAK_TFLOPS = 67  # H200 FP64 tensor-core peak
# Ranges of n annotated with the growth of cuSOLVER's time: first n, last n, label position, shade.
MOTIVATION_RANGES = (
    (256, 4096, 1800, "#F6E1DF"),
    (4096, 65536, 16384, "#ECF0F3"),
    (65536, 131072, 128000, "#E2E9F3"),
)
MOTIVATION_XLIM = (200, 300000)


@dataclass
class Data:
    """The fastest square-matrix measurement of every library and size, and of those with checks."""

    best: dict
    checked: dict

    @classmethod
    def of(cls, rows):
        square = [row for row in rows if row["m"] == row["n"]]
        return cls(ms.fastest(square), ms.fastest(square, checked_only=True))


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FS,
            "axes.titlesize": FS + 0.5,
            "axes.titleweight": "bold",
            "axes.labelsize": FS,
            "xtick.labelsize": FS - 1,
            "ytick.labelsize": FS - 1,
            "legend.fontsize": FS,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#9AA5AC",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "text.color": "#1C252C",
            "axes.labelcolor": "#1C252C",
        }
    )


def save(fig, out, stem):
    """Write the figure as PDF and PNG. The PDF has no creation date, so the same data always gives
    the same file."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        metadata = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(out / f"{stem}.{ext}", dpi=220, facecolor="white", metadata=metadata)
    plt.close(fig)


def blend(hex_color, amount):
    """The color mixed with white: amount 1 is the color itself."""
    rgb = matplotlib.colors.to_rgb(hex_color)
    return tuple(1 - amount * (1 - c) for c in rgb)


def size_label(n, pos=None):
    return f"{int(n):,}" if n < 1000 else f"{n / 1000:.3g}k"


def style_axis(data, ax, gpus, mode, metric):
    """Log axes over the measured orders of a GPU count."""
    ax.set_title(TITLES[mode], loc="left", pad=9)
    ax.set_xscale("log", base=2)
    sizes = sorted({n for lib in LIBS for n in ms.available(data.best, gpus, mode, lib)})
    ticks = [256, 1024, 4096, 16384, 65536] if gpus == 1 else [60000, 98304, 131072, 196608, 262144, 327680]
    if sizes:
        ticks = [x for x in ticks if min(sizes) <= x <= max(sizes)]
        if max(sizes) not in ticks and (not ticks or max(sizes) / ticks[-1] > 1.3):
            ticks.append(max(sizes))
        ax.set_xlim(min(sizes) / 1.10, max(sizes) * 1.12)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(size_label))
    ax.set_xlabel("Matrix order n (n × n)")
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#DEE3E7", lw=0.7)
    ax.set_axisbelow(True)
    if metric == "time_s":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    else:
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))


def speedup_curves(data, ax, gpus, mode):
    """QR-Omega's speedup over each reference against n; a hollow marker is a timing-only point."""
    style_axis(data, ax, gpus, mode, "time_s")
    values = []
    for lib in ms.REFS:
        common = sorted(set(ms.available(data.best, gpus, mode)) & set(ms.available(data.best, gpus, mode, lib)))
        xs, ys, hollow = [], [], []
        for n in common:
            own, ref = ms.lookup(data.best, gpus, mode, n), ms.lookup(data.best, gpus, mode, n, lib)
            speedup = ref["time_s"] / own["time_s"]
            xs.append(n)
            ys.append(speedup)
            values.append(speedup)
            hollow.append(own["validation"] == "timing_only" or ref["validation"] == "timing_only")
        if not xs:
            continue
        ax.plot(xs, ys, color=COLORS[lib], lw=1.8)
        for x, y, open_ in zip(xs, ys, hollow, strict=True):
            ax.plot(
                x,
                y,
                marker=MARKERS[lib],
                ms=4,
                markerfacecolor="white" if open_ else COLORS[lib],
                markeredgecolor=COLORS[lib],
                markeredgewidth=1,
                zorder=4,
            )
    ax.axhline(1.0, color="#9AA5AC", lw=0.9, ls="--", zorder=1)
    if values:
        ax.set_ylim(min(0.85, min(values) * 0.92), max(values) * 1.25 if max(values) > 1 else 1.3)
        top = ax.get_ylim()[1]
        if top <= 12:  # odd integers on the log axis, ending at 10
            ticks = list(np.arange(1, min(math.floor(top), 10) + 1, 2))
            if ticks[-1] == 9 and top >= 10:
                ticks[-1] = 10
        else:
            ticks = np.arange(1, math.floor(top) + 1, 2 if top <= 25 else 5)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))


def error_curves(data, ax, gpus, mode, metric):
    """The residual or orthogonality error of every library against n."""
    style_axis(data, ax, gpus, mode, metric)
    values = []
    for lib in LIBS:
        points = [ms.lookup(data.checked, gpus, mode, n, lib) for n in ms.available(data.checked, gpus, mode, lib)]
        points = [r for r in points if r.get(metric) is not None and r[metric] > 0]
        if not points:
            continue
        xs, ys = [r["n"] for r in points], [r[metric] for r in points]
        values += ys
        own = lib == ms.OWN
        ax.plot(
            xs,
            ys,
            color=COLORS[lib],
            lw=2.1 if own else 1.4,
            linestyle="-" if own else "--",
            alpha=1 if own else 0.92,
        )
        for r in points:
            ax.plot(
                r["n"],
                r[metric],
                marker=MARKERS[lib],
                ms=4.8 if own else 4,
                markerfacecolor="white" if r["validation"] == "timing_only" else COLORS[lib],
                markeredgecolor=COLORS[lib],
                markeredgewidth=1,
                zorder=4,
            )
    if values:
        if max(values) / min(values) < 5:
            ax.set_ylim(min(values) / 1.8, max(values) * 1.8)
        else:
            ax.set_ylim(min(values) / 1.6, max(values) * 2)


def compact(ax, size=FS - 1.5):
    """Smaller text, markers and lines for the panels of a two-row figure."""
    for t in ax.texts:
        t.set_fontsize(min(t.get_fontsize(), size))
    ax.title.set_fontsize(FS + 0.5)
    ax.title.set_position((0, 1.0))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(FS - 0.6)
    for line in ax.lines:
        if line.get_marker() not in (None, "None", ""):
            line.set_markersize(min(line.get_markersize(), 3.2))
        line.set_linewidth(min(line.get_linewidth(), 1.4))


def size_ticks(ax, gpus):
    if gpus == 1:
        ticks = [256, 2048, 16384, 131072]
        labels = {256: "$2^{8}$", 2048: "$2^{11}$", 16384: "$2^{14}$", 131072: "$2^{17}$"}
    else:
        ticks = [131072, 196608, 262144, 327680]
        labels = {t: f"{t / 1000:.0f}k" for t in ticks}
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: labels.get(round(v), "")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def gemm_for(data, gpus, mode, n):
    """The GEMM of the same order and arithmetic; FP32 GEMM stands in for a missing 3xTF32 one on
    several GPUs and is labelled so."""
    row = ms.lookup(data.best, gpus, mode, n, "GEMM", "gemm")
    if row is None and gpus > 1 and mode == "3xtf32":
        row = ms.lookup(data.best, gpus, "fp32", n, "GEMM", "gemm")
    return row


def throughput_bars(data, ax, gpus, mode):
    """Throughput at one order: the value on each bar and QR-Omega's speedup under each reference's."""
    n = BAR_SIZE.get(gpus) if ms.lookup(data.best, gpus, mode, BAR_SIZE.get(gpus)) else None
    ax.set_title(f"{TITLES[mode]}, n = {n:,}" if n else TITLES[mode], fontsize=FS, loc="left", pad=4)
    if n is None:
        return
    libs = [*ms.REFS, ms.OWN] + (["GEMM"] if gpus == 1 else [])  # GEMM comparator on one GPU only
    recs = [
        ms.lookup(data.best, gpus, mode, n, lib) if lib != "GEMM" else gemm_for(data, gpus, mode, n) for lib in libs
    ]
    heights = [x["tflops"] if x else 0 for x in recs]
    own, peak = recs[3], max(heights)
    for i, (lib, rec, height) in enumerate(zip(libs, recs, heights, strict=True)):
        if not rec:
            ax.text(i, peak * 0.03, "n/a", ha="center", va="bottom", fontsize=FS - 2, color="#64727B")
            continue
        ax.bar(i, height, width=0.68, color=COLORS[lib], edgecolor="white", lw=0.5, zorder=3)
        label = f"{height:.1f}" if height < 100 else f"{height:.0f}"
        if i < 3 and own:
            label += f"\n{rec['time_s'] / own['time_s']:.2f}×"
        ax.text(
            i,
            height + peak * 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=FS - 2.2,
            linespacing=0.95,
            color="#1C252C" if i >= 3 else COLORS[lib],
            fontweight="bold" if i == 3 else "normal",
        )
    ax.set_ylim(0, peak * 1.42)
    ax.set_xlim(-0.6, len(libs) - 0.4)
    labels = ["cuSOLVER" if gpus == 1 else "cuSOLVERMp", "MAGMA", "SLATE", "QR-Ω"]
    if gpus == 1:
        labels.append("GEMM" if not recs[4] or recs[4]["precision"] == mode else "FP32 GEMM")
    ax.set_xticks(range(len(libs)), labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=FS - 1.6)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis="y", color="#DEE3E7", lw=0.5)
    ax.set_axisbelow(True)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(FS - 0.6)


def performance_figure(data, gpus, out, stem):
    """Speedup over each reference against n (top) and throughput at one order (bottom)."""
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH, 2.8))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.135, top=0.855, hspace=0.72, wspace=0.27)
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[lib],
            marker=MARKERS[lib],
            lw=1.3,
            ms=3.2,
            label=f"QR-Ω vs {ms.display_name(lib, gpus)}",
        )
        for lib in ms.REFS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=3,
        handlelength=1.8,
        columnspacing=1.6,
    )
    for j, mode in enumerate(ms.MODES):
        ax = axes[0, j]
        speedup_curves(data, ax, gpus, mode)
        top = ax.get_ylim()[1]
        if top > 12:  # a 1-2-5 progression keeps wide speedup ranges readable on the log axis
            ax.set_yticks([t for t in (1, 2, 5, 10, 20, 50) if t <= top])
        compact(ax)
        size_ticks(ax, gpus)
        ax.set_ylabel("Speedup (×)" if j == 0 else "", labelpad=1)
        ax.set_xlabel("n", labelpad=0)
        bx = axes[1, j]
        throughput_bars(data, bx, gpus, mode)
        bx.set_ylabel("TFLOP/s" if j == 0 else "", labelpad=1)
    save(fig, out, stem)


def error_ticks(ax):
    """Label 1, 2 and 5 times each power of ten (every digit if that gives fewer than two ticks)
    when the errors span less than about a decade, so every accuracy panel shows values on its
    axis."""
    lo, hi = ax.get_ylim()
    if hi / lo >= 30:
        return
    decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
    for digits in ((1, 2, 5), range(1, 10)):
        ticks = [k * 10.0**e for e in decades for k in digits if lo <= k * 10.0**e <= hi]
        if len(ticks) >= 2:
            break
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0e}".replace("e-0", "e-")))
    ax.yaxis.set_minor_formatter(NullFormatter())


def accuracy_figure(data, gpus, out, stem):
    """Residual (top) and orthogonality error (bottom) of every library against n."""
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH, 2.45))
    fig.subplots_adjust(left=0.068, right=0.995, bottom=0.11, top=0.82, hspace=0.55, wspace=0.30)
    libs = [lib for lib in LIBS if any(key[1] == gpus and key[4] == lib for key in data.checked)]
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[lib],
            marker=MARKERS[lib],
            lw=1.3,
            ms=3.2,
            ls="-" if lib == ms.OWN else "--",
            label=ms.display_name(lib, gpus),
        )
        for lib in libs
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=4,
        handlelength=2.2,
        columnspacing=1.6,
    )
    for j, mode in enumerate(ms.MODES):
        ax = axes[0, j]
        error_curves(data, ax, gpus, mode, "residual")
        error_ticks(ax)
        compact(ax)
        size_ticks(ax, gpus)
        ax.set_ylabel("Residual" if j == 0 else "", labelpad=1)
        ax.set_xlabel("")
        bx = axes[1, j]
        error_curves(data, bx, gpus, mode, "orthogonality")
        error_ticks(bx)
        compact(bx)
        size_ticks(bx, gpus)
        bx.set_title("")
        bx.set_ylabel("Orthogonality" if j == 0 else "", labelpad=1)
        bx.set_xlabel("n", labelpad=0)
    save(fig, out, stem)


def scaling_figure(data, out, stem, n=STRONG_SCALING_N, counts=(2, 3, 4, 8)):
    """Strong scaling at fixed n: throughput of QR-Omega (FP32, TF32, 3xTF32) and of the references'
    fastest FP32 configuration, labelled with QR-Omega's speedup over two GPUs."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.95))
    fig.subplots_adjust(left=0.155, right=0.995, bottom=0.2, top=0.83)
    series = [
        (ms.OWN, "tf32", "QR-Ω TF32", "#00A3C4"),
        (ms.OWN, "3xtf32", "QR-Ω 3xTF32", "#3D8FA8"),
        (ms.OWN, "fp32", "QR-Ω FP32", COLORS[ms.OWN]),
        ("cuSOLVER", "fp32", "cuSOLVERMp", COLORS["cuSOLVER"]),
        ("MAGMA", "fp32", "MAGMA", COLORS["MAGMA"]),
        ("SLATE", "fp32", "SLATE", COLORS["SLATE"]),
    ]
    width = 0.135
    missing = []
    for j, (lib, precision, _, color) in enumerate(series):
        for i, gpus in enumerate(counts):
            rec = data.best.get(("qr", gpus, precision, n, lib))
            x = i + (j - 2.5) * width
            if not rec:
                missing.append((x, color))
                continue
            ax.bar(x, rec["tflops"], width=width * 0.92, color=color, edgecolor="white", lw=0.3, zorder=3)
            two = data.best.get(("qr", 2, precision, n, lib))
            if lib == ms.OWN and gpus > 2 and two:
                ax.text(
                    x,
                    rec["tflops"] * 1.06,
                    f"{two['time_s'] / rec['time_s']:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FS - 2.4,
                    rotation=90,
                    color="#1C252C",
                )
    ax.set_yscale("log")
    ax.set_ylim(top=max(b.get_height() for b in ax.patches) * 1.9)  # headroom for the labels
    ticks = [t for t in (25, 50, 100, 200, 400, 800, 1600) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
    ax.yaxis.set_minor_locator(NullLocator())
    for x, color in missing:
        ax.plot(x, ax.get_ylim()[0] * 1.08, marker="x", ms=3, mew=0.8, color=color, zorder=4, clip_on=False)
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.legend(
        handles=[Patch(facecolor=c, edgecolor="white", label=label) for _, _, label, c in series],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        ncol=3,
        frameon=False,
        fontsize=FS - 1.3,
        handlelength=1.0,
        columnspacing=0.9,
    )
    ax.set_xticks(range(len(counts)), [f"{p} GPUs" for p in counts], fontsize=FS - 1)
    ax.set_ylabel("TFLOP/s", labelpad=1)
    ax.grid(axis="y", color="#DEE3E7", lw=0.5, which="major")
    ax.set_axisbelow(True)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(FS - 0.6)
    save(fig, out, stem)


def tall_figure(rows, out, stem):
    """Tall-and-skinny matrices on one GPU: throughput against the column count, QR-Omega for three
    row counts and the references at the smallest."""

    def tflops(m, n, time):
        return (2.0 * m * n * n - 2.0 * n**3 / 3) / time / 1e12

    data = {
        (r["library"], r["precision"], r["m"], r["n"]): tflops(r["m"], r["n"], r["time_s"])
        for r in rows
        if r["kind"] == "qr" and r["gpus"] == 1 and r["m"] > r["n"]
    }
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.3))
    fig.subplots_adjust(left=0.068, right=0.985, bottom=0.18, top=0.7, wspace=0.3)
    smallest = min(TALL_ROWS)
    for ax, mode in zip(axes, ms.MODES, strict=True):
        for m, (_, shade) in TALL_ROWS.items():
            pts = [(n, data[(ms.OWN, mode, m, n)]) for n in TALL_COLUMNS if (ms.OWN, mode, m, n) in data]
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts], color=blend(COLORS[ms.OWN], shade), marker="o", ms=4, lw=1.7
            )
        for lib in ms.REFS:
            key = (lib, ms.precision_of(lib, mode), smallest)
            pts = [(n, data[(*key, n)]) for n in TALL_COLUMNS if (*key, n) in data]
            if pts:
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=COLORS[lib],
                    marker=MARKERS[lib],
                    ms=3.5,
                    lw=1.2,
                    ls="--",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(TALL_COLUMNS, ["64", "256", "1,024", "4,096"])
        ax.set_xlim(64 / 1.35, 4096 * 1.35)
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(TITLES[mode], loc="left", fontweight="bold")
        ax.tick_params(labelsize=FS - 0.6)
        ax.set_xlabel("columns n")
    axes[0].set_ylabel("TFLOP/s")
    handles = [
        Line2D([], [], color=blend(COLORS[ms.OWN], shade), marker="o", ms=4, lw=1.7, label=f"QR-Ω, m = {label}")
        for label, shade in TALL_ROWS.values()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLORS[lib],
            marker=MARKERS[lib],
            ms=3.5,
            lw=1.2,
            ls="--",
            label=f"{lib}, m = {TALL_ROWS[smallest][0]}",
        )
        for lib in ms.REFS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.4,
        columnspacing=1.6,
    )
    save(fig, out, stem)


def motivation_figure(data, out, stem):
    """FP64 on one GPU: the fastest tuned configuration of every QR library, DGEMM of the same order,
    and, for each range of n, how the time of cuSOLVER grows and its rate as a fraction of DGEMM."""

    def series(lib, kind="qr"):
        return [
            (n, ms.lookup(data.best, 1, "fp64", n, lib, kind)) for n in ms.available(data.best, 1, "fp64", lib, kind)
        ]

    def time(n):
        return ms.lookup(data.best, 1, "fp64", n, "cuSOLVER")["time_s"]

    gemm = dict(series("GEMM", "gemm"))
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.9))
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.21, top=0.86)
    for i, (lo, hi, centre, color) in enumerate(MOTIVATION_RANGES):
        left = MOTIVATION_XLIM[0] if i == 0 else lo
        right = MOTIVATION_XLIM[1] if i == len(MOTIVATION_RANGES) - 1 else hi
        ax.axvspan(left, right, color=color, lw=0, zorder=0)
        exponent = math.log(time(hi) / time(lo)) / math.log(hi / lo)
        fractions = [row["tflops"] / gemm[n]["tflops"] for n, row in series("cuSOLVER") if lo <= n <= hi and n in gemm]
        low, high = round(100 * min(fractions)), round(100 * max(fractions))
        share = f"≤{high}%" if i == 0 else f"{low}–{high}%"
        ax.text(
            centre, 0.0085, f"time ∝ $n^{{{exponent:.1f}}}$\n{share} of GEMM", ha="center", va="bottom", fontsize=FS - 1
        )
    ax.axhline(FP64_PEAK_TFLOPS, color="#1C252C", lw=0.9, ls="--", zorder=2)
    ax.text(
        MOTIVATION_XLIM[1] * 0.93,
        FP64_PEAK_TFLOPS * 1.1,
        f"FP64 peak, {FP64_PEAK_TFLOPS} TFLOP/s",
        ha="right",
        va="bottom",
        fontsize=FS - 1,
    )
    styles = {
        "GEMM": ("o", "-.", "DGEMM"),
        "cuSOLVER": ("s", "-", "cuSOLVER"),
        "MAGMA": ("^", "-", "MAGMA"),
        "SLATE": ("D", "-", "SLATE"),
    }
    for lib, (marker, line, label) in styles.items():
        points = series(lib, "gemm" if lib == "GEMM" else "qr")
        ax.plot(
            [n for n, _ in points],
            [row["tflops"] for _, row in points],
            color=COLORS[lib],
            marker=marker,
            ms=3,
            lw=1.4,
            ls=line,
            label=label,
            zorder=3,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(*MOTIVATION_XLIM)
    ax.set_ylim(0.004, 250)
    ax.set_xticks([256, 1024, 4096, 16384, 65536], ["256", "1K", "4K", "16K", "64K"])
    ax.minorticks_off()
    ax.tick_params(labelsize=FS - 0.6)
    ax.grid(axis="y", color="#DEE3E7", lw=0.5)
    ax.set_xlabel("Matrix order $n$", labelpad=1)
    ax.set_ylabel("TFLOP/s", labelpad=1)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.55, 1.0),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
        fontsize=FS - 0.5,
    )
    save(fig, out, stem)


def draw(rows, out):
    """Draw every figure of the paper into out, named by its number in the paper."""
    style()
    data = Data.of(rows)
    motivation_figure(data, out, "fig02-motivation")
    performance_figure(data, 1, out, "fig06-single-perf")
    accuracy_figure(data, 1, out, "fig07-single-acc")
    performance_figure(data, 8, out, "fig08-multi8-perf")
    accuracy_figure(data, 8, out, "fig09-multi8-acc")
    tall_figure(rows, out, "fig10-tall-skinny")
    scaling_figure(data, out, "fig11-scaling")
