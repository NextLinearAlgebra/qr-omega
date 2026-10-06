"""Backward stability independent of the condition number: factor S1 H diag(sigma) H S2 / n,
whose singular values sigma_k = kappa^(-k / (n - 1)) span kappa from 10 to far beyond 1 / u, with
equal column norms, so that no column scaling can hide kappa (kappa = 1 is a signed identity).
Every factorization must pass the benchmark's independent residual and orthogonality checks, and
the worst error over kappa must stay within a small factor of the best."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

KAPPAS = {
    "fp64": (1e1, 1e4, 1e8, 1e12, 1e16, 1e20),
    "fp32": (1e1, 1e3, 1e6, 1e9, 1e12),
    "tf32": (1e1, 1e3, 1e6, 1e9, 1e12),
    "3xtf32": (1e1, 1e3, 1e6, 1e9, 1e12),
}
# Errors that grew with kappa (as u kappa for unstable or u kappa^2 for Gram-based panels) would
# spread over orders of magnitude across these kappas.
SPREAD = 50


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 3, 4, 8), default=1)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=tuple(KAPPAS), default=tuple(KAPPAS))
    args = parser.parse_args()
    binary = args.binary.resolve()
    prefix = (
        []
        if args.gpus == 1
        else ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", str(args.gpus)]
    )
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    # The grids whose WY panels stay within the 2.5D bound, and the tree on the tallest grid.
    grids = [(1, "tree", 1), (1, "wy", 8)]
    if args.gpus > 1:
        rows = 2 if args.gpus == 8 else args.gpus
        grids.append((rows, "tree", 1))
        if rows * rows <= args.gpus // rows:
            grids.append((rows, "wy", 8))
    with tempfile.TemporaryDirectory() as temp:
        for mode in args.modes:
            for rows, panel_format, aggregate in grids:
                residuals = []
                for kappa in KAPPAS[mode]:
                    output = (args.output or Path(temp)) / (
                        f"{mode}-r{rows}-{panel_format}-k{kappa:g}.json"
                    )
                    subprocess.run(
                        [
                            *prefix,
                            str(binary),
                            "--mode",
                            mode,
                            "--m",
                            str(args.n),
                            "--n",
                            str(args.n),
                            "--kappa",
                            str(kappa),
                            "--grid-rows",
                            str(rows),
                            "--panel-format",
                            panel_format,
                            "--aggregate",
                            str(aggregate),
                            "--lookahead",
                            "--reps",
                            "1",
                            "--output",
                            str(output),
                        ],
                        check=True,
                        timeout=300,
                    )
                    record = json.loads(output.read_text())
                    assert record["pass"] and record["status"] == 0, record
                    assert record["kappa"] == kappa, record
                    assert record["carriers"]["bounded"] == record["carriers"]["products"], record
                    residuals.append((kappa, record["residual"], record["orthogonality"]))
                errors = [max(r, o) for _, r, o in residuals]
                spread = max(errors) / min(errors)
                print(
                    mode,
                    f"grid-rows {rows}",
                    panel_format,
                    " ".join(f"k={k:g}:{r:.2e}/{o:.2e}" for k, r, o in residuals),
                    f"spread {spread:.2f}",
                    flush=True,
                )
                assert spread <= SPREAD, (mode, rows, panel_format, residuals)


if __name__ == "__main__":
    main()
