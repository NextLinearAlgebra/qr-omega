#!/usr/bin/env python3
"""Regenerate the figures of the paper and the numbers it quotes from the measurements; no GPU needed.

1. Assemble data/paper.csv from QR-Omega's measured cells (reproducers/presets.json) and the
   measurements of the reference libraries and GEMM (data/references.csv), and check it.
2. Draw the figures of the paper into figures/, named by their number in the paper.
3. Write figures/paper-numbers.csv: every number the paper quotes, next to the same quantity
   computed from the data.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import figures
import measurements as ms
import paper_numbers

HERE = Path(__file__).resolve().parent


def write_claims(claims, path):
    with path.open("w", newline="") as sink:
        writer = csv.writer(sink, lineterminator="\n")
        writer.writerow(("section", "statement", "paper", "data", "status"))
        writer.writerows((c.section, c.statement, c.paper, c.data, c.status) for c in claims)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presets", type=Path, default=ms.PRESETS, help="QR-Omega's measured cells")
    parser.add_argument("--references", type=Path, default=ms.REFERENCES, help="reference measurements")
    parser.add_argument("--table", type=Path, default=ms.TABLE, help="the assembled table to write")
    parser.add_argument("--output", type=Path, default=HERE / "figures", help="directory of the figures")
    args = parser.parse_args()

    own, references = ms.assemble(args.presets, args.references, args.table)
    rows = ms.check(ms.read(args.table))
    figures.draw(rows, args.output)
    claims = paper_numbers.claims(rows)
    write_claims(claims, args.output / "paper-numbers.csv")
    manifest = {
        "presets_sha256": sha256(args.presets),
        "references_sha256": sha256(args.references),
        "rows": len(rows),
        "figures": sorted(p.name for p in args.output.glob("*.pdf")),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    status = Counter(c.status for c in claims)
    print(
        f"{own} QR-Omega and {references} reference measurements; {len(manifest['figures'])} figures in {args.output}"
    )
    print(
        f"{len(claims)} numbers quoted in the paper: {status['exact']} reproduced exactly, "
        f"{status['rounding']} within rounding, {status['differs']} different"
    )
    for claim in claims:
        if claim.status != "exact":
            print(f"  [{claim.section}] {claim.statement}: paper {claim.paper}, data {claim.data} ({claim.status})")


if __name__ == "__main__":
    main()
