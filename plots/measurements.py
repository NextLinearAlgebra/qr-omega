"""The measurements behind the paper: QR-Omega's cells and the reference libraries, as one table.

QR-Omega's rows come from the measured cells (reproducers/presets.json); the rows of cuSOLVER,
cuSOLVERMp, MAGMA, SLATE and GEMM come from data/references.csv. Each row is one measurement of a
factorization (kind "qr") or a matrix product (kind "gemm") of an m x n matrix.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESETS = HERE.parent / "reproducers" / "presets.json"
REFERENCES = HERE / "data" / "references.csv"
TABLE = HERE / "data" / "paper.csv"

OWN = "QR-Omega"  # QR-Omega's library tag in the tables
MODES = ("fp64", "fp32", "tf32", "3xtf32")
REFS = ("cuSOLVER", "MAGMA", "SLATE")  # on several GPUs, "cuSOLVER" is cuSOLVERMp
FIELDS = (
    "id",
    "kind",
    "library",
    "precision",
    "gpus",
    "m",
    "n",
    "time_s",
    "tflops",
    "residual",
    "orthogonality",
    "validation",
    "repetitions",
    "timing_statistic",
    "configuration",
)


def qr_flops(m, n):
    """Householder QR of an m x n matrix; 4/3 n^3 exactly as published for square matrices."""
    return 4 / 3 * n**3 if m == n else 2.0 * m * n * n - 2.0 * n**3 / 3


def own_rows(presets):
    """One row per measured cell of QR-Omega: the median of the timed repetitions of its two runs and,
    when the runs were validated, the larger residual and orthogonality error of the two."""
    for case in presets:
        m, n, time = case.get("m", case["n"]), case["n"], case["measured_time_s"]
        checked = case["measured_residual"] is not None
        row = {
            "kind": "qr",
            "library": OWN,
            "precision": case["mode"],
            "gpus": case["gpus"],
            "m": m,
            "n": n,
            "time_s": time,
            "tflops": qr_flops(m, n) / time / 1e12,
            "residual": case["measured_residual"] if checked else "",
            "orthogonality": case["measured_orthogonality"] if checked else "",
            "validation": "passed" if checked else "timing_only",
            "repetitions": 2 * case["reps"],
            "timing_statistic": "median of the timed repetitions of two " + ("validated runs" if checked else "runs"),
            "configuration": json.dumps(case["options"], sort_keys=True),
        }
        row["id"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
        yield {key: row[key] for key in FIELDS}


def assemble(presets=PRESETS, references=REFERENCES, table=TABLE):
    """Write the table of all measurements: QR-Omega's cells followed by the reference rows."""
    with references.open() as source:
        reference_rows = list(csv.DictReader(source))
    rows = list(own_rows(json.loads(presets.read_text()))) + reference_rows
    with table.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows) - len(reference_rows), len(reference_rows)


def read(table=TABLE):
    """The rows of the table with numbers parsed; a missing error is None."""
    with table.open() as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        for key in ("gpus", "m", "n"):
            row[key] = int(row[key])
        for key in ("time_s", "tflops", "residual", "orthogonality"):
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def check(rows):
    """Reject a table with a repeated measurement, an invalid time or throughput, or a square
    QR-Omega cell without numerical checks."""
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in ("kind", "gpus", "precision", "m", "n", "library"))
        if key in seen:
            raise ValueError(f"repeated measurement: {key}")
        seen.add(key)
        if (
            row["m"] < row["n"]
            or row["n"] <= 0
            or (row["kind"] == "gemm" and row["m"] != row["n"])
            or not (math.isfinite(row["time_s"]) and row["time_s"] > 0)
        ):
            raise ValueError(f"invalid measurement: {key}")
        flops = qr_flops(row["m"], row["n"]) if row["kind"] == "qr" else 2 * row["n"] ** 3
        if not math.isclose(row["tflops"], flops / row["time_s"] / 1e12, rel_tol=2e-4):
            raise ValueError(f"throughput inconsistent with the time: {key}")
        square = row["m"] == row["n"]
        if row["library"] == OWN and square and row["validation"] != "passed":
            raise ValueError(f"QR-Omega measurement without numerical checks: {key}")
    return rows


def fastest(rows, checked_only=False):
    """The fastest row of every (kind, GPUs, precision, n, library); with checked_only, among the
    factorizations that passed their numerical checks."""
    best = {}
    for row in rows:
        if checked_only and (
            row["kind"] != "qr" or row["validation"] != "passed" or not row["residual"] or not row["orthogonality"]
        ):
            continue
        key = row["kind"], row["gpus"], row["precision"], row["n"], row["library"]
        if key not in best or row["time_s"] < best[key]["time_s"]:
            best[key] = row
    return best


def precision_of(lib, mode, kind="qr"):
    """The arithmetic a library is compared in: the references' FP32 for TF32 and 3xTF32."""
    if lib == OWN or kind == "gemm":
        return mode
    return "fp64" if mode == "fp64" else "fp32"


def lookup(best, gpus, mode, n, lib=OWN, kind="qr"):
    return best.get((kind, gpus, precision_of(lib, mode, kind), n, lib))


def available(best, gpus, mode, lib=OWN, kind="qr"):
    """The orders n measured for a library on a GPU count, in increasing order."""
    precision = precision_of(lib, mode, kind)
    return sorted(k[3] for k in best if k[0] == kind and k[1] == gpus and k[2] == precision and k[4] == lib)


def display_name(lib, gpus):
    if lib == OWN:
        return "QR-Ω"
    return "cuSOLVERMp" if lib == "cuSOLVER" and gpus > 1 else lib
