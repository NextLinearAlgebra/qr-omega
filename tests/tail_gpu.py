"""Compact-WY panels followed by a trailing block in the tree format (--tail-from), through the
benchmark's independent validation of the whole factorization, Q = Q_head diag(I, Q_tail)."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

# n, panel width, aggregate, first column of the tail, tail schedule
CASES = [
    (2048, 64, 4, 1024, []),
    (1501, 33, 2, 132, ["--tail-strip", "1024", "--tail-dc", "2"]),
    (3001, 64, 1, 1984, ["--tail-tree-update", "streaming", "--tail-depth", "2"]),
    (2048, 64, 4, 1024, ["--tail-no-lookahead"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    failures = []
    with tempfile.TemporaryDirectory() as temp:
        for mode in ("fp64", "fp32", "tf32", "3xtf32"):
            for n, b, aggregate, tail_from, tail in CASES:
                output = Path(temp) / f"{mode}-{n}.json"
                command = [
                    str(args.binary.resolve()),
                    "--mode",
                    mode,
                    "--m",
                    str(n),
                    "--n",
                    str(n),
                    "--reps",
                    "1",
                    "--b",
                    str(b),
                    "--panel-format",
                    "wy",
                    "--aggregate",
                    str(aggregate),
                    "--lookahead",
                    "--tail-from",
                    str(tail_from),
                    *tail,
                    "--output",
                    str(output),
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                record = json.loads(output.read_text()) if output.exists() else {}
                carriers = record.get("carriers", {})
                ok = (
                    result.returncode == 0
                    and record.get("pass") is True
                    and record.get("tail_from") == tail_from
                    and carriers.get("bounded") == carriers.get("products")
                )
                print(
                    f"{mode:6s} n={n} b={b} aggregate={aggregate} tail_from={tail_from}: "
                    f"residual {record.get('residual')} orthogonality {record.get('orthogonality')}"
                    f" {'ok' if ok else 'FAILED'}"
                )
                if not ok:
                    failures.append((mode, n, result.stderr[-400:]))
    if failures:
        raise SystemExit(f"tree-format tails failed: {failures}")


if __name__ == "__main__":
    main()
