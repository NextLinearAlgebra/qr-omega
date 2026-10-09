"""End-to-end GE/TS/TT updates and Q replay, through the benchmark's independent validation."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 3, 4, 8), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("fp64", "fp32", "tf32", "3xtf32"),
        default=("fp64", "fp32", "tf32", "3xtf32"),
    )
    parser.add_argument("--tree-update", choices=("streaming", "resident"), default="streaming")
    parser.add_argument("--panel-format", choices=("tree", "wy"), default="tree")
    parser.add_argument("--aggregate", type=int, default=1)
    parser.add_argument("--zc", type=int, default=2, help="replication of Z (groups of blocks)")
    parser.add_argument(
        "--tall",
        action="store_true",
        help="exercise reconstruction across several waves of GPU blocks",
    )
    args = parser.parse_args()
    binary = args.binary.resolve()
    prefix = [] if args.gpus == 1 else ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", str(args.gpus)]
    grids = sorted({1, args.gpus, 2 if args.gpus in (4, 8) else 1})
    if args.panel_format == "wy":
        # A compact-WY panel over grid rows sums its W over them: pr^2 <= pc keeps that bounded.
        grids = [r for r in grids if r == 1 or r * r <= args.gpus // r]
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temp:
        for mode in args.modes:
            for grid in grids:
                # Odd panels exercise the mapped-update fallback. The small rectangular case
                # leaves some grid rows empty and others shorter than the panel width.
                shapes = [(777, 777, 33, 2, 3, 1), (1025, 1025, 64, 4, 4, 2)]
                if args.gpus > 1:
                    shapes.append((131, 129, 64, 2, 4, 1))
                else:
                    shapes += [(777, 777, 33, 2, 3, 2), (1025, 1025, 64, 4, 4, 1)]
                if args.tall:
                    # Both scalar types need several waves of GPU blocks. The second
                    # case also exercises TS chains, odd widths and a partial final domain.
                    shapes += [(65537, 129, 64, 1, 4, 1), (131073, 65, 33, 2, 3, 1)]
                for m, n, b, tiles, fan, groups in shapes:
                    output = (args.output or Path(temp)) / f"{mode}-r{grid}-{m}x{n}-g{groups}.json"
                    subprocess.run(
                        [
                            *prefix,
                            str(binary),
                            "--mode",
                            mode,
                            "--m",
                            str(m),
                            "--n",
                            str(n),
                            "--b",
                            str(b),
                            "--domain-tiles",
                            str(tiles),
                            "--fan-in",
                            str(fan),
                            "--grid-rows",
                            str(grid),
                            "--tree-groups",
                            str(groups),
                            "--tree-update",
                            args.tree_update,
                            "--panel-format",
                            args.panel_format,
                            "--aggregate",
                            str(args.aggregate),
                            "--dc",
                            "4",
                            "--mc",
                            "4",
                            "--zc",
                            str(args.zc),
                            "--lookahead",
                            "--reps",
                            "1",
                            "--output",
                            str(output),
                        ],
                        check=True,
                        timeout=180,
                    )
                    record = json.loads(output.read_text())
                    assert record["pass"] and record["status"] == 0, record
                    assert record["gpus"] == args.gpus and record["grid"][0] == grid, record
                    assert record["panel_algorithm"] == "hqr", record
                    assert record["panel_format"] == args.panel_format, record
                    assert record["aggregate"] == args.aggregate, record
                    assert record["carriers"]["bounded"] == record["carriers"]["products"], record
                    if args.gpus > 1:
                        comm = record["communication"]
                        assert comm["nccl_commit"] == "5357eff325eddf978137de7140195a5568fa8a11", record
                        assert comm["unordered_backend"] == "NVSHMEM", record
                        if grid > 1:
                            assert comm["unordered_publications"] > 0, record
                    print(
                        mode,
                        record["grid"],
                        m,
                        n,
                        groups,
                        "residual",
                        record["residual"],
                        "orthogonality",
                        record["orthogonality"],
                        flush=True,
                    )


if __name__ == "__main__":
    main()
