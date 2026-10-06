#!/usr/bin/env python3
"""Write plots/data/paper.csv: QR-Omega's rows from the measured cells (reproducers/presets.json)
and the measurements of the reference libraries and GEMM (plots/data/references.csv)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def own_rows(presets, fields):
    """One row per measured cell: the median of the timed repetitions of its two validated runs and
    the larger residual and orthogonality error of the two."""
    for case in presets:
        n, time = case["n"], case["measured_time_s"]
        row = {
            "kind": "qr",
            "library": "TQR",
            "precision": case["mode"],
            "gpus": case["gpus"],
            "m": case.get("m", n),
            "n": n,
            "time_s": time,
            "tflops": 4 / 3 * n**3 / time / 1e12,
            "residual": case["measured_residual"],
            "orthogonality": case["measured_orthogonality"],
            "validation": "passed",
            "repetitions": 2 * case["reps"],
            "timing_statistic": "median of the timed repetitions of two validated runs",
            "configuration": json.dumps(case["options"], sort_keys=True),
        }
        row["id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
        yield {k: row[k] for k in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, default=HERE.parent / "reproducers/presets.json")
    parser.add_argument("--references", type=Path, default=HERE / "data/references.csv")
    parser.add_argument("--output", type=Path, default=HERE / "data/paper.csv")
    args = parser.parse_args()
    with args.references.open() as f:
        reader = csv.DictReader(f)
        fields, references = reader.fieldnames, list(reader)
    rows = list(own_rows(json.loads(args.presets.read_text()), fields)) + references
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({len(rows) - len(references)} QR-Omega) to {args.output}")


if __name__ == "__main__":
    main()
