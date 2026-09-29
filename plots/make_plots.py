#!/usr/bin/env python3
"""Render portable poster figures from data/paper.csv; no raw logs needed."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
import numpy as np

from data_io import FIELDS, dump_csv

HERE = Path(__file__).resolve().parent
MODES = ["fp64", "fp32", "tf32", "3xtf32"]
TITLES = {"fp64":"FP64", "fp32":"FP32 (IEEE)", "tf32":"TF32", "3xtf32":"3×TF32"}
LIBS = ["TQR", "cuSOLVER", "MAGMA", "SLATE"]
REFS = LIBS[1:]
MARKERS = {"TQR":"o", "cuSOLVER":"s", "MAGMA":"^", "SLATE":"D"}

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
        if accuracy and (row["kind"] != "qr" or row["validation"] != "passed" or not row["residual"] or not row["orthogonality"]):
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
    return f"{int(n):,}" if n < 1000 else f"{n/1000:.3g}k"

def fmt_time(t):
    return f"{t:.3g} s"

def available(best, p, mode, lib="TQR", kind="qr"):
    return sorted(k[3] for k in best if k[0] == kind and k[1] == p and k[2] == (mode if lib == "TQR" or kind == "gemm" else "fp64" if mode == "fp64" else "fp32") and k[4] == lib)

def select_highlights(best, cfg):
    chosen, candidates = {}, []
    for p in [1, 2, 3, 4]:
        for mode in MODES:
            scope = "single" if p == 1 else f"multi-p{p}"
            key = f"{scope}:{mode}"
            all_sizes = available(best, p, mode)
            feasible = []
            for n in all_sizes:
                own = lookup(best, p, mode, n)
                refs = [lookup(best, p, mode, n, lib) for lib in REFS]
                if not all(refs):
                    continue
                speedup = min(r["time_s"] for r in refs) / own["time_s"]
                candidates.append(dict(scope=scope, gpus=p, precision=mode, n=n, speedup_over_fastest_reference=speedup))
                # The selected size must support the requested GEMM bar too.
                # For distributed x3 only an explicitly labelled IEEE comparator exists.
                gm = lookup(best,p,mode,n,"GEMM","gemm")
                if gm is None and p > 1 and mode == "3xtf32":
                    gm = lookup(best,p,"fp32",n,"GEMM","gemm")
                if gm is not None:
                    feasible.append((speedup, n))
            lower = cfg["highlight_min_n_single" if p == 1 else "highlight_min_n_multi"]
            eligible = [x for x in feasible if x[1] >= lower]
            fallback = not eligible
            eligible = eligible or feasible
            override = cfg["highlight_sizes"].get(key)
            if override is not None:
                if override not in [x[1] for x in feasible]:
                    raise ValueError(f"Highlight {key}={override} lacks TQR or a reference measurement")
                selected = next(x for x in feasible if x[1] == override)
            elif eligible:
                selected = max(eligible, key=lambda x:(x[0], x[1]))
            else:
                continue
            chosen[p, mode] = dict(scope=scope, gpus=p, precision=mode, n=selected[1],
                speedup_over_fastest_reference=selected[0],
                criterion="max min(reference_time/TQR_time), three references and same-size GEMM present",
                coverage_fallback=fallback, eligible_sizes=";".join(str(n) for _,n in eligible))
    return chosen, candidates

def select_scaling(best, cfg):
    chosen, candidates = {}, []
    for mode in MODES:
        common = set.intersection(*(set(available(best, p, mode)) for p in [2, 3, 4]))
        for n in sorted(common):
            if not all(lookup(best, p, mode, n, lib) for p in [2,3,4] for lib in LIBS):
                continue
            own2 = lookup(best, 2, mode, n)["time_s"]
            relative = []
            for lib in REFS:
                ref2 = lookup(best, 2, mode, n, lib)["time_s"]
                for p in [3,4]:
                    relative.append((own2/lookup(best,p,mode,n)["time_s"])/(ref2/lookup(best,p,mode,n,lib)["time_s"]))
            score = statistics.geometric_mean(relative)
            candidates.append(dict(precision=mode, n=n, scaling_advantage_geomean=score,
                tqr_speedup_2_to_3=own2/lookup(best,3,mode,n)["time_s"],
                tqr_speedup_2_to_4=own2/lookup(best,4,mode,n)["time_s"]))
        options = [r for r in candidates if r["precision"] == mode]
        override = cfg["scaling_sizes"].get(mode)
        if override is not None:
            options = [r for r in options if r["n"] == override]
            if not options:
                raise ValueError(f"Scaling size {mode}={override} lacks complete 2/3/4 GPU coverage")
        if options:
            chosen[mode] = max(options, key=lambda r:(r["scaling_advantage_geomean"], r["n"]))
    return chosen, candidates

class Renderer:
    def __init__(self, best, acc, highlights, scaling, cfg, out):
        self.best, self.acc = best, acc
        self.highlights, self.scaling, self.cfg, self.out = highlights, scaling, cfg, out
        self.colors = cfg["colors"]
        self.files, self.chart_data = [], []
        plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":13,
            "axes.titleweight":"bold", "axes.labelsize":10, "xtick.labelsize":8.7,
            "ytick.labelsize":9, "legend.fontsize":10, "pdf.fonttype":42, "ps.fonttype":42,
            "svg.fonttype":"none", "axes.spines.top":False, "axes.spines.right":False,
            "axes.edgecolor":"#9AA5AC", "text.color":"#1C252C", "axes.labelcolor":"#1C252C"})

    def save(self, fig, stem, footer=""):
        if footer:
            fig.text(.02, .015, footer, ha="left", va="bottom", fontsize=8, color="#46545F")
        path = self.out / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        for ext in self.cfg["formats"]:
            fig.savefig(path.with_suffix("."+ext), dpi=self.cfg["dpi"], facecolor="white")
        self.files.append(stem)
        plt.close(fig)

    def legend(self, fig, p, gemm=False):
        libs = LIBS + (["GEMM"] if gemm else [])
        handles = [Line2D([], [], color=self.colors[lib], marker=MARKERS.get(lib, ""), lw=2.5,
                   label=name(lib,p) if lib != "GEMM" else "Measured GEMM") for lib in libs]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), frameon=False, ncol=len(libs), handlelength=2)

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
            if max(sizes) not in ticks and (not ticks or max(sizes)/ticks[-1] > 1.3):
                ticks.append(max(sizes))
            ax.set_xlim(min(sizes)/1.10, max(sizes)*1.12)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_n))
        ax.set_xlabel("Matrix order n (n × n)")
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", color="#DEE3E7", lw=.7)
        ax.set_axisbelow(True)
        labels = {"time_s":"QR-Ω speedup over reference (×)", "residual":"Relative residual (block maximum)", "orthogonality":"Relative Q·Qᵀ inverse error"}
        ax.set_ylabel(labels[metric])
        if metric == "time_s":
            # Integer ticks (1, 2, 3, ...) on the log axis; exact tick set
            # is fixed in curve() once the data limits are known.
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

    def curve(self, ax, p, mode, metric, record=True):
        data = self.best if metric == "time_s" else self.acc
        self.style_axis(ax,p,mode,metric)
        if metric == "time_s":
            # Linear speedup curves: ref_time / TQR_time per reference.
            # Linear-friendly (bounded near 1×) where raw seconds span 5 orders.
            vals = []
            for lib in REFS:
                common = sorted(set(available(data,p,mode,"TQR")) & set(available(data,p,mode,lib)))
                xs, ys, flags = [], [], []
                for n in common:
                    own = lookup(data,p,mode,n,"TQR")
                    ref = lookup(data,p,mode,n,lib)
                    if not own or not ref or not own.get("time_s") or not ref.get("time_s"):
                        continue
                    if own["time_s"] <= 0 or ref["time_s"] <= 0:
                        continue
                    s = ref["time_s"] / own["time_s"]
                    xs.append(n)
                    ys.append(s)
                    vals.append(s)
                    flags.append(own["validation"] == "timing_only" or ref["validation"] == "timing_only")
                    if record:
                        d = dict(own)
                        d.update(chart=f"p{p}-speedup", display_precision=mode, plotted_value=s,
                                 speedup_vs=lib, reference_time_s=ref["time_s"])
                        self.chart_data.append(d)
                if not xs:
                    continue
                ax.plot(xs,ys, color=self.colors[lib], lw=1.8)
                for x, y, open_ in zip(xs,ys,flags):
                    ax.plot(x, y, marker=MARKERS[lib], ms=4,
                            markerfacecolor="white" if open_ else self.colors[lib],
                            markeredgecolor=self.colors[lib], markeredgewidth=1, zorder=4)
            ax.axhline(1.0, color="#9AA5AC", lw=.9, ls="--", zorder=1)
            if vals:
                ax.set_ylim(min(0.85, min(vals)*0.92), max(vals)*1.25 if max(vals) > 1 else 1.3)
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
            points = [lookup(data,p,mode,n,lib) for n in available(data,p,mode,lib)]
            points = [r for r in points if r.get(metric) is not None and r[metric] > 0]
            if not points:
                continue
            xs, ys = [r["n"] for r in points], [r[metric] for r in points]
            vals += ys
            ax.plot(xs,ys, color=self.colors[lib], lw=2.1 if lib == "TQR" else 1.4,
                    linestyle="-" if lib == "TQR" else "--", alpha=1 if lib=="TQR" else .92)
            for r in points:
                ax.plot(r["n"], r[metric], marker=MARKERS[lib], ms=4.8 if lib=="TQR" else 4,
                        markerfacecolor="white" if r["validation"]=="timing_only" else self.colors[lib],
                        markeredgecolor=self.colors[lib], markeredgewidth=1, zorder=4)
                if record:
                    self.chart_data.append(dict(chart=f"p{p}-{metric}", display_precision=mode, plotted_value=r[metric], **r))
        if vals:
            if max(vals)/min(vals) < 5:
                ax.set_ylim(min(vals)/1.8, max(vals)*1.8)
            else:
                ax.set_ylim(min(vals)/1.6, max(vals)*2)
        if metric == "time_s" and (p,mode) in self.highlights:
            n = self.highlights[p,mode]["n"]
            own = lookup(self.best,p,mode,n)
            ax.axvline(n, color=self.colors["TQR"], lw=.8, alpha=.45, zorder=1)
            ax.scatter([n], [own[metric]], s=75, facecolors="none", edgecolors=self.colors["TQR"], linewidths=1.4, zorder=5)
            for j,lib in enumerate(REFS):
                ref = lookup(self.best,p,mode,n,lib)
                ax.annotate("", xy=(n,own[metric]), xytext=(n,ref[metric]),
                    arrowprops=dict(arrowstyle="->", lw=1.1, color=self.colors[lib], connectionstyle=f"arc3,rad={(j-1)*.35}"))
            ax.text(.03,.96, f"Selected n = {n:,}\nQR-Ω: {fmt_time(own[metric])}", transform=ax.transAxes,
                    va="top",fontsize=9,color=self.colors["TQR"], bbox=dict(facecolor="white", edgecolor="none", alpha=.90, pad=3))

    def curves(self, p, metric, layout="grid"):
        scope = "single" if p == 1 else f"multi-p{p}"
        if layout == "row":
            fig,axes = plt.subplots(1,4,figsize=(17.0,4.6))
            fig.subplots_adjust(left=.057,right=.987,bottom=.22,top=.72,wspace=.34)
        elif layout == "column":
            fig,axes = plt.subplots(4,1,figsize=(5.0,14.0))
            fig.subplots_adjust(left=.19,right=.96,bottom=.08,top=.90,hspace=.62)
        else:
            fig,axes = plt.subplots(2,2,figsize=(10.8,8.3))
            fig.subplots_adjust(left=.093,right=.976,bottom=.13,top=.84,hspace=.46,wspace=.28)
        label={"time_s":"Speedup over reference", "residual":"Relative residual", "orthogonality":"Orthogonality check"}[metric]
        fig.suptitle(f"{p} H200 GPU{'s' if p>1 else ''} · {label}",y=.987,fontsize=16,fontweight="bold")
        if metric == "time_s":
            handles=[Line2D([], [], color=self.colors[lib], marker=MARKERS[lib], lw=1.8,
                             label=f"vs {name(lib,p)}") for lib in REFS]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.94), frameon=False, ncol=len(handles), handlelength=2)
        else:
            self.legend(fig,p)
        for ax,mode in zip(np.ravel(axes), MODES):
            self.curve(ax,p,mode,metric,record=(layout=="grid"))
        stem={"time_s":"01-time", "residual":"03-residual", "orthogonality":"04-orthogonality"}[metric]
        self.save(fig,f"{scope}/{stem}"+(f"-{layout}" if layout!="grid" else ""))

    def gemm_for(self,p,mode,n):
        # No cross-size extrapolation. The x3 fallback is labelled as an IEEE
        # comparator on the bar and in the caption; it is not an x3 ceiling.
        row=lookup(self.best,p,mode,n,"GEMM","gemm")
        if row is None and p>1 and mode=="3xtf32":
            row=lookup(self.best,p,"fp32",n,"GEMM","gemm")
        return row

    def bars(self, ax,p,mode):
        ax.set_title(TITLES[mode],loc="left",pad=9)
        if (p,mode) not in self.highlights:
            ax.text(.5,.5,"No common reference size yet",ha="center",transform=ax.transAxes)
            return
        n = self.highlights[p,mode]["n"]
        libs = REFS + ["TQR", "GEMM"]
        records = [lookup(self.best,p,mode,n,lib) if lib!="GEMM" else self.gemm_for(p,mode,n) for lib in libs]
        ys = [r["tflops"] if r else 0 for r in records]
        peak = max(ys)
        for i,(lib,r,y) in enumerate(zip(libs,records,ys)):
            if r:
                ax.bar(i,y,width=.66,color=self.colors[lib],edgecolor="white",lw=.8,
                    hatch="//" if r["validation"]=="timing_only" else None,zorder=3)
                ax.text(i,y+peak*.025,f"{y:.1f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
                self.chart_data.append(dict(chart=f"p{p}-histogram", display_precision=mode, plotted_value=y, **r))
            else:
                ax.text(i,peak*.035,"not\nmeasured",ha="center",va="bottom",fontsize=8,color="#64727B")
        own=records[3]
        for j,ref in enumerate(records[:3]):
            h=peak*(1.18+.19*j)
            # Explicit parallel brackets, with arrowheads at the reference bars.
            # Their height is data-scaled, so labels stay readable across precisions.
            ax.plot([3,3,j],[ys[3]+peak*.11,h,h],color=self.colors[libs[j]],lw=.9,zorder=4)
            ax.annotate("",xy=(j,ys[j]+peak*.105),xytext=(j,h),arrowprops=dict(arrowstyle="->",color=self.colors[libs[j]],lw=1.1))
            x_text = (j+3)/2 - (0.35 if libs[j] == "MAGMA" else 0)
            ax.text(x_text,h+peak*.02,f"{ref['time_s']/own['time_s']:.2f}×",ha="center",fontsize=9,color=self.colors[libs[j]])
        ax.set_ylim(0,peak*1.82)
        ax.set_xlim(-.65,4.65)
        labels=[name(x,p).replace("cuSOLVERMp","cuSOLVER\nMp") for x in libs]
        if records[-1] and records[-1]["precision"] != mode:
            labels[-1]="FP32\nGEMM†"
        ax.set_xticks(range(5),labels,fontsize=8.5)
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y",color="#DEE3E7",lw=.7)
        ax.set_axisbelow(True)
        ax.text(.98,.96,f"n = {n:,}",transform=ax.transAxes,ha="right",va="top",fontsize=10)

    def histogram(self,p,layout="grid"):
        if layout == "row":
            fig,axes=plt.subplots(1,4,figsize=(17,4.8))
            fig.subplots_adjust(left=.05,right=.987,bottom=.19,top=.82,wspace=.31)
        else:
            fig,axes=plt.subplots(2,2,figsize=(10.8,8.3))
            fig.subplots_adjust(left=.08,right=.976,bottom=.14,top=.87,hspace=.40,wspace=.25)
        fig.suptitle(f"{p} H200 GPU{'s' if p>1 else ''} · Throughput at the highlighted sizes",fontsize=15,fontweight="bold",y=.98)
        for ax,mode in zip(np.ravel(axes),MODES):
            self.bars(ax,p,mode)
        scope="single" if p==1 else f"multi-p{p}"
        self.save(fig,f"{scope}/02-throughput"+("-row" if layout=="row" else ""))

    def scaling_panel(self,ax,mode):
        ax.set_title(TITLES[mode],loc="left",pad=9)
        if mode not in self.scaling:
            ax.text(.5,.5,"Awaiting complete reference coverage",ha="center",transform=ax.transAxes)
            return
        n=self.scaling[mode]["n"]
        x=np.arange(3)
        maxrate=0
        for j,lib in enumerate(LIBS):
            records=[lookup(self.best,p,mode,n,lib) for p in [2,3,4]]
            ys=[r["tflops"] for r in records]
            maxrate=max(maxrate,max(ys))
            xpos=x+(j-1.5)*.185
            ax.bar(xpos,ys,width=.175,color=self.colors[lib],edgecolor="white",lw=.5,zorder=3)
            speed=[records[0]["time_s"]/r["time_s"] for r in records]
            for r,s in zip(records,speed):
                self.chart_data.append(dict(chart="multi-scaling",display_precision=mode,plotted_value=r["tflops"],scaling_from_2_gpus=s,**r))
        ax.set_ylim(0,maxrate*1.30)
        ax.set_xticks(x,["2","3","4"])
        ax.set_xlabel("Number of H200 GPUs")
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.grid(axis="y",color="#DEE3E7",lw=.7)
        ax.set_axisbelow(True)
        ax.text(.04,.96,f"n = {n:,}",transform=ax.transAxes,va="top",fontsize=10)

    def scaling_plot(self,layout="grid"):
        fig,axes=plt.subplots(2,2,figsize=(11.8,8.4)) if layout=="grid" else plt.subplots(1,4,figsize=(18,4.7))
        fig.subplots_adjust(left=.075,right=.976 if layout=="grid" else .987,bottom=.14 if layout=="grid" else .23,top=.83 if layout=="grid" else .72,hspace=.46,wspace=.30 if layout=="grid" else .31)
        fig.suptitle("Strong scaling · 2, 3 and 4 H200 GPUs",fontsize=16,fontweight="bold",y=.985)
        handles=[Patch(facecolor=self.colors[lib],edgecolor="white",label=name(lib,4)) for lib in LIBS]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.94),frameon=False,ncol=len(handles))
        for ax,mode in zip(np.ravel(axes),MODES):
            self.scaling_panel(ax,mode)
        self.save(fig,"multi/05-scaling"+("-row" if layout=="row" else ""))

    def gallery(self):
        sections=[]
        for stem in self.files:
            if stem.endswith(("-row","-column")):
                continue
            title=html.escape(stem)
            sections.append(f'<article><h2>{title}</h2><a href="{stem}.pdf"><img loading="lazy" src="{stem}.png" alt="{title}"></a><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a></p></article>')
        (self.out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>QR-Ω figures</title><style>body{font:16px system-ui;margin:32px;background:#f3f7f9;color:#173443}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}article{background:white;padding:16px;border-radius:12px}img{width:100%}h2{font-size:18px}a{color:#006d8f}</style><h1>QR-Ω benchmark figures</h1><p>Portable data and selection details are in ../data/. Click a figure for its vector PDF.</p><main>'+''.join(sections)+'</main>')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=HERE/"data/paper.csv")
    ap.add_argument("--config",type=Path,default=HERE/"config.json")
    ap.add_argument("--output",type=Path,default=HERE/"figures")
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text())
    rows=read_rows(args.data)
    best,acc=best_rows(rows),best_rows(rows,accuracy=True)
    highlights, hc=select_highlights(best,cfg)
    scaling,sc=select_scaling(best,cfg)
    dump_csv(args.output/"data/best.csv",best.values())
    dump_csv(args.output/"data/accuracy.csv",acc.values())
    dump_csv(args.output/"data/highlights.csv",highlights.values(),["scope","gpus","precision","n","speedup_over_fastest_reference","criterion","coverage_fallback","eligible_sizes"])
    dump_csv(args.output/"data/highlight_candidates.csv",hc,["scope","gpus","precision","n","speedup_over_fastest_reference"])
    dump_csv(args.output/"data/scaling_selection.csv",scaling.values(),["precision","n","scaling_advantage_geomean","tqr_speedup_2_to_3","tqr_speedup_2_to_4"])
    dump_csv(args.output/"data/scaling_candidates.csv",sc,["precision","n","scaling_advantage_geomean","tqr_speedup_2_to_3","tqr_speedup_2_to_4"])
    coverage=[]
    for p in [1,2,3,4]:
        for mode in MODES:
            for lib in LIBS:
                ns=available(best,p,mode,lib)
                av=available(acc,p,mode,lib)
                coverage.append(dict(gpus=p,display_precision=mode,library=lib,
                    precision=mode if lib=="TQR" else "fp64" if mode=="fp64" else "fp32",
                    timing_sizes=";".join(map(str,ns)),accuracy_sizes=";".join(map(str,av)),
                    max_timing_n=max(ns,default=""),max_accuracy_n=max(av,default="")))
    dump_csv(args.output/"data/coverage.csv",coverage,["gpus","display_precision","library","precision","timing_sizes","accuracy_sizes","max_timing_n","max_accuracy_n"])
    renderer=Renderer(best,acc,highlights,scaling,cfg,args.output)
    for p in [1,2,3,4]:
        for metric in ["time_s","residual","orthogonality"]:
            renderer.curves(p,metric)
            if p==1:
                renderer.curves(p,metric,"row")
        renderer.histogram(p)
        renderer.histogram(p,"row")
        print(f"Rendered {p}-GPU figures",flush=True)
    renderer.scaling_plot()
    renderer.scaling_plot("row")
    renderer.gallery()
    dump_csv(args.output/"data/figure_points.csv",renderer.chart_data,["chart","display_precision","plotted_value","scaling_from_2_gpus","speedup_vs"]+FIELDS)
    (args.output/"data/selection.json").write_text(json.dumps({"highlights":list(highlights.values()),"scaling":scaling},indent=2)+"\n")
    print(f"Generated {len(renderer.files)} figure variants in {args.output}")

if __name__=="__main__":
    main()
