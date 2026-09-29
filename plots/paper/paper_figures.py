#!/usr/bin/env python3
"""Paper-sized versions of the plots/ figures.

Same data model, colours, markers and panel logic as plots/make_plots.py (its Renderer draws every panel); only the
page geometry changes: one figure* per topic, two rows of four precision panels, 7-pt text, no poster titles.
Input: plots/data/paper.csv. Output: plots/paper/figures/*.pdf|png.
"""
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
PLOTS = HERE.parent
sys.path.insert(0, str(PLOTS))
import make_plots as mp  # noqa: E402

W = 7.16  # IEEEtran two-column \textwidth in inches
FS = 7


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "axes.titlesize": FS + .5,
                         "axes.titleweight": "bold", "axes.labelsize": FS, "xtick.labelsize": FS - 1,
                         "ytick.labelsize": FS - 1, "legend.fontsize": FS, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA5AC",
                         "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5,
                         "ytick.minor.size": 1.5, "text.color": "#1C252C", "axes.labelcolor": "#1C252C"})


def shrink(ax, size=FS - 1.5):
    for t in ax.texts:
        t.set_fontsize(min(t.get_fontsize(), size))
    ax.title.set_fontsize(FS + .5)
    ax.title.set_position((0, 1.0))
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(FS - 1.2)
    for line in ax.lines:
        if line.get_marker() not in (None, "None", ""):
            line.set_markersize(min(line.get_markersize(), 3.2))
        line.set_linewidth(min(line.get_linewidth(), 1.4))


def xticks(ax, p):
    if p == 1:
        ticks = [256, 2048, 16384, 131072]
        lab = {256: "$2^{8}$", 2048: "$2^{11}$", 16384: "$2^{14}$", 131072: "$2^{17}$"}
    else:
        ticks = [131072, 196608, 262144, 327680]
        lab = {t: f"{t/1000:.0f}k" for t in ticks}
    lo, hi = ax.get_xlim()
    ticks = [t for t in ticks if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: lab.get(int(round(v)), "")))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def build(curated, cfg_over):
    rows = mp.read_rows(curated)
    best, acc = mp.best_rows(rows), mp.best_rows(rows, accuracy=True)
    cfg = json.loads((PLOTS / "config.json").read_text())
    cfg.update(cfg_over)
    highlights, _ = mp.select_highlights(best, cfg)
    scaling, _ = mp.select_scaling(best, cfg)
    out = HERE / "figures"
    # Four GPUs: the bars use n = 131,072, the smallest size of the multi-GPU range (no GEMM comparator there).
    for mode in mp.MODES:
        if mp.lookup(best, 4, mode, 131072):
            highlights[(4, mode)] = {"n": 131072}
    r = mp.Renderer(best, acc, highlights, scaling, cfg, out)
    style()
    return r, best, acc, highlights, out


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    print("wrote", out / f"{stem}.pdf")


def bars(r, ax, p, mode):
    """Compact throughput bars: value on each bar; QR-Omega's speedup over each reference under its value."""
    n = r.highlights[p, mode]["n"] if (p, mode) in r.highlights else None
    ax.set_title(f"{mp.TITLES[mode]}, n = {n:,}" if n else mp.TITLES[mode], fontsize=FS, loc="left", pad=4)
    if n is None:
        return
    libs = mp.REFS + ["TQR"] + (["GEMM"] if p == 1 else [])   # GEMM comparator on one GPU only
    recs = [mp.lookup(r.best, p, mode, n, lib) if lib != "GEMM" else r.gemm_for(p, mode, n) for lib in libs]
    ys = [x["tflops"] if x else 0 for x in recs]
    own = recs[3]
    peak = max(ys)
    for i, (lib, x, y) in enumerate(zip(libs, recs, ys)):
        if not x:
            ax.text(i, peak * .03, "n/a", ha="center", va="bottom", fontsize=FS - 2, color="#64727B")
            continue
        ax.bar(i, y, width=.68, color=r.colors[lib], edgecolor="white", lw=.5, zorder=3,
               hatch="////" if x["validation"] == "timing_only" else None)
        label = f"{y:.1f}" if y < 100 else f"{y:.0f}"
        if i < 3 and own:
            label += f"\n{x['time_s'] / own['time_s']:.2f}×"
        ax.text(i, y + peak * .03, label, ha="center", va="bottom", fontsize=FS - 2.2, linespacing=.95,
                color="#1C252C" if i >= 3 else r.colors[lib], fontweight="bold" if i == 3 else "normal")
    ax.set_ylim(0, peak * 1.42)
    ax.set_xlim(-.6, len(libs) - .4)
    labels = ["cuSOLVER" if p == 1 else "cuSOLVERMp", "MAGMA", "SLATE", "QR-Ω"]
    if p == 1:
        labels.append("GEMM" if not recs[4] or recs[4]["precision"] == mode else "FP32 GEMM")
    ax.set_xticks(range(len(libs)), labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=FS - 1.6)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.yaxis.set_major_locator(mp.MaxNLocator(4))
    ax.grid(axis="y", color="#DEE3E7", lw=.5)
    ax.set_axisbelow(True)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(FS - 1.2)


def perf_figure(r, p, out, stem):
    fig, axes = plt.subplots(2, 4, figsize=(W, 2.8))
    fig.subplots_adjust(left=.058, right=.995, bottom=.105, top=.855, hspace=.72, wspace=.27)
    handles = [Line2D([], [], color=r.colors[lib], marker=mp.MARKERS[lib], lw=1.3, ms=3.2,
                      label=f"QR-Ω vs {mp.name(lib, p)}") for lib in mp.REFS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0), frameon=False, ncol=3,
               handlelength=1.8, columnspacing=1.6)
    for j, mode in enumerate(mp.MODES):
        ax = axes[0, j]
        r.curve(ax, p, mode, "time_s", record=False)
        top = ax.get_ylim()[1]
        if top > 12:   # a 1-2-5 progression keeps wide speedup ranges readable on the log axis
            ax.set_yticks([t for t in (1, 2, 5, 10, 20, 50) if t <= top])
        shrink(ax)
        xticks(ax, p)
        ax.set_ylabel("Speedup (×)" if j == 0 else "", labelpad=1)
        ax.set_xlabel("n", labelpad=0)
        bx = axes[1, j]
        bars(r, bx, p, mode)
        bx.set_ylabel("TFLOP/s" if j == 0 else "", labelpad=1)
    save(fig, out, stem)


def acc_figure(r, p, out, stem):
    fig, axes = plt.subplots(2, 4, figsize=(W, 2.45))
    fig.subplots_adjust(left=.068, right=.995, bottom=.11, top=.82, hspace=.55, wspace=.30)
    libs = mp.LIBS if p == 1 else ["TQR"]   # on several GPUs only QR-Omega's checks are reported
    handles = [Line2D([], [], color=r.colors[lib], marker=mp.MARKERS[lib], lw=1.3, ms=3.2,
                      ls="-" if lib == "TQR" else "--", label=mp.name(lib, p)) for lib in libs]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.0), frameon=False, ncol=4,
               handlelength=2.2, columnspacing=1.6)
    for j, mode in enumerate(mp.MODES):
        ax = axes[0, j]
        r.curve(ax, p, mode, "residual", record=False)
        shrink(ax)
        xticks(ax, p)
        ax.set_ylabel("Residual" if j == 0 else "", labelpad=1)
        ax.set_xlabel("")
        bx = axes[1, j]
        r.curve(bx, p, mode, "orthogonality", record=False)
        shrink(bx)
        xticks(bx, p)
        bx.set_title("")
        bx.set_ylabel("Orthogonality" if j == 0 else "", labelpad=1)
        bx.set_xlabel("n", labelpad=0)
    save(fig, out, stem)


def scaling_figure(r, out, stem, n=229376):
    """Strong scaling at fixed n on 2, 3, 4 GPUs: throughput of QR-Omega (IEEE FP32, TF32, 3xTF32) and of the
    references' fastest FP32 configuration; missing bars (does not fit / not measured) are labelled."""
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    fig.subplots_adjust(left=.155, right=.995, bottom=.2, top=.83)
    series = [("TQR", "tf32", "QR-Ω TF32", "#00A3C4"), ("TQR", "3xtf32", "QR-Ω 3xTF32", "#3D8FA8"),
              ("TQR", "fp32", "QR-Ω FP32", r.colors["TQR"]), ("cuSOLVER", "fp32", "cuSOLVERMp", r.colors["cuSOLVER"]),
              ("MAGMA", "fp32", "MAGMA", r.colors["MAGMA"]), ("SLATE", "fp32", "SLATE", r.colors["SLATE"])]
    w = .135
    missing = []
    for j, (lib, prec, label, color) in enumerate(series):
        for i, p in enumerate([2, 3, 4]):
            rec = r.best.get(("qr", p, prec, n, lib))
            x = i + (j - 2.5) * w
            if rec:
                ax.bar(x, rec["tflops"], width=w * .92, color=color, edgecolor="white", lw=.3, zorder=3)
                if lib == "TQR" and p > 2 and r.best.get(("qr", 2, prec, n, lib)):
                    s2 = r.best[("qr", 2, prec, n, lib)]["time_s"] / rec["time_s"]
                    ax.text(x, rec["tflops"] * 1.06, f"{s2:.2f}", ha="center", va="bottom", fontsize=FS - 2.8,
                            rotation=90, color="#1C252C")
            else:
                missing.append((x, color))
    ax.set_yscale("log")
    top = max(b.get_height() for b in ax.patches)
    ax.set_ylim(top=top * 1.9)   # headroom for the speedup labels
    ticks = [t for t in (25, 50, 100, 200, 400, 800, 1600) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
    ax.yaxis.set_minor_locator(NullLocator())
    lo = ax.get_ylim()[0]
    for x, color in missing:
        ax.plot(x, lo * 1.08, marker="x", ms=3, mew=.8, color=color, zorder=4, clip_on=False)
    ax.set_xlim(-.6, 2.6)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="white", label=l) for _, _, l, c in series],
              loc="upper center", bbox_to_anchor=(.5, 1.27), ncol=3, frameon=False, fontsize=FS - 1.3,
              handlelength=1.0, columnspacing=.9)
    ax.set_xticks([0, 1, 2], ["2 GPUs", "3 GPUs", "4 GPUs"], fontsize=FS - 1)
    ax.set_ylabel("TFLOP/s", labelpad=1)
    ax.grid(axis="y", color="#DEE3E7", lw=.5, which="major")
    ax.set_axisbelow(True)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(FS - 1.2)
    save(fig, out, stem)


def main():
    curated = PLOTS / "data/paper.csv"
    over = {"highlight_sizes": {"single:fp64": 65536, "single:fp32": 65536, "single:tf32": 65536,
                                "single:3xtf32": 65536}, "highlight_min_n_multi": 131072}
    if len(sys.argv) > 1:
        over = json.loads(sys.argv[1])
    r, best, acc, hl, out = build(curated, over)
    perf_figure(r, 1, out, "single-perf")
    acc_figure(r, 1, out, "single-acc")
    if any(k[1] == 4 for k in best):
        perf_figure(r, 4, out, "multi4-perf")
        acc_figure(r, 4, out, "multi4-acc")
        scaling_figure(r, out, "scaling")


if __name__ == "__main__":
    main()
