"""Plan, and optionally run, a QR-Omega factorization on the GPU of this allocation.

    python3 -m planner --n 20000 --mode fp64                # schedule, prediction, command line
    python3 -m planner --n 20000 --mode fp64 --run          # factor with it
    python3 -m planner --n 20000 --mode fp64 --refine 3     # measure the three best, keep the fastest
    python3 -m planner --n 20000 --machine machine/profiles/h200.json   # plan for a stored profile

Without --machine the visible GPU is probed once and its profile cached.
"""

import argparse
import shlex
import sys
from pathlib import Path

from machine import Machine
from machine.collect import cached

from .driver import BUILD, command, measure
from .model import factor_time
from .plan import load_kernels, plan
from .schedule import MODES, memory_bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--m", type=int, help="rows (default: n)")
    parser.add_argument("--mode", choices=list(MODES), default="fp64")
    parser.add_argument("--machine", type=Path, help="stored profile (default: probe the GPU)")
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--top", type=int, default=1, help="print the best N schedules")
    parser.add_argument("--run", action="store_true", help="factor with the selected schedule")
    parser.add_argument(
        "--refine", type=int, metavar="K", help="measure the K best, keep the fastest"
    )
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    args.m = args.m or args.n
    if args.m < args.n:
        parser.error("the matrix needs at least as many rows as columns")
    return args


def refine(args, ranked):
    """Measure the candidates and return the fastest as (predicted seconds, schedule)."""
    measured = []
    for predicted, s in ranked:
        record = measure(args.mode, args.m, args.n, s, args.build, args.reps)
        seconds = record["median_s"] if record else None
        print(f"measured {seconds}  predicted {predicted:.4g}  {s.options()}", file=sys.stderr)
        if record:
            measured.append((seconds, predicted, s))
    if not measured:
        raise SystemExit("every candidate failed")
    seconds, predicted, best = min(measured, key=lambda x: x[0])
    print(f"measured {seconds:.4g} s, {seconds / predicted - 1:+.1%} against the prediction")
    return predicted, best


def main():
    args = parse_args()
    machine = Machine.load(args.machine) if args.machine else Machine(cached(args.build)[0])
    kernels = load_kernels()
    ranked = plan(machine, args.mode, args.m, args.n, kernels, top=max(args.top, args.refine or 1))
    print(
        f"{machine.name}, {machine.sms} SMs at {machine.clock / 1e9:.2f} GHz; "
        f"{args.mode} {args.m} x {args.n}"
    )
    for seconds, s in ranked[: args.top]:
        print(f"  {seconds:10.4g} s  {s.options()}")
    predicted, best = ranked[0]
    if args.run or args.refine:
        predicted, best = refine(args, ranked[: args.refine or 1])
    total, parts = factor_time(machine, kernels, args.mode, args.m, args.n, best, detail=True)
    flops = 2.0 * args.m * args.n**2 - 2.0 * args.n**3 / 3.0
    print(
        f"predicted {total:.4g} s ({flops / total / 1e12:.1f} TFLOP/s): "
        + ", ".join(f"{key} {value:.3g}" for key, value in parts.items())
    )
    need = memory_bytes(args.mode, args.m, args.n, best)
    print(f"memory {need / 2**30:.2f} of {machine.memory_free / 2**30:.2f} GiB free")
    print(
        "command: " + shlex.join(command(args.mode, args.m, args.n, best, args.build, args.reps)[1])
    )


if __name__ == "__main__":
    main()
