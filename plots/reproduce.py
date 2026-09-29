#!/usr/bin/env python3
"""Regenerate the figures of the paper and the numbers it quotes from plots/data/paper.csv (no GPU needed)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paper"))
import paper_figures as pf  # noqa: E402
import make_plots as mp  # noqa: E402


def checked_rows(path):
    """Read the paper table and check it: one row per cell, positive times, consistent throughput, and numerical
    checks for every QR-Omega measurement."""
    rows = mp.read_rows(path)
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("kind", "gpus", "precision", "m", "n", "library"))
        if key in seen:
            raise ValueError(f"Duplicate cell: {key}")
        seen.add(key)
        if row["m"] != row["n"] or row["n"] <= 0 or not (math.isfinite(row["time_s"]) and row["time_s"] > 0):
            raise ValueError(f"Invalid measurement: {key}")
        expected = (4 / 3 if row["kind"] == "qr" else 2) * row["n"] ** 3 / row["time_s"] / 1e12
        if not math.isclose(row["tflops"], expected, rel_tol=2e-4):
            raise ValueError(f"Inconsistent throughput: {key}")
        if row["kind"] == "qr" and row["library"] == "TQR" and row["gpus"] in (1, 4) and row["validation"] != "passed":
            raise ValueError(f"QR-Omega measurement without numerical checks: {key}")
    return rows


def headline(best, out):
    """QR-Omega against the fastest reference at the sizes the paper quotes."""
    table = []
    for gpus, sizes in ((1, (65536, 131072)), (4, (131072, 229376, 327680))):
        for mode in mp.MODES:
            for n in sizes:
                own = mp.lookup(best, gpus, mode, n)
                if not own:
                    continue
                refs = [r for r in (mp.lookup(best, gpus, mode, n, lib) for lib in mp.REFS) if r]
                fastest = min(refs, key=lambda r: r["time_s"]) if refs else None
                table.append({"gpus": gpus, "precision": mode, "n": n, "time_s": round(own["time_s"], 4),
                              "tflops": round(own["tflops"], 2),
                              "fastest_reference": mp.name(fastest["library"], gpus) if fastest else "",
                              "speedup": round(fastest["time_s"] / own["time_s"], 3) if fastest else "",
                              "residual": own["residual"], "orthogonality": own["orthogonality"]})
    with (out / "headline-results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data/paper.csv")
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    rows = checked_rows(args.data)
    over = {"highlight_sizes": {f"single:{mode}": 65536 for mode in mp.MODES}, "highlight_min_n_multi": 131072}
    renderer, best, _, _, _ = pf.build(args.data, over)
    out = args.output
    pf.perf_figure(renderer, 1, out, "single-perf")
    pf.acc_figure(renderer, 1, out, "single-acc")
    pf.perf_figure(renderer, 4, out, "multi4-perf")
    pf.acc_figure(renderer, 4, out, "multi4-acc")
    pf.scaling_figure(renderer, out, "scaling")
    table = headline(best, out)
    manifest = {"input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "rows": len(rows),
                "figures": sorted(p.name for p in out.glob("*.pdf")), "headline_rows": len(table)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Checked {len(rows)} rows; wrote {len(manifest['figures'])} figures and headline-results.csv to {out}")


if __name__ == "__main__":
    main()
