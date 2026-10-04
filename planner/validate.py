"""Measure the planner against the published presets on the GPU of this allocation.

    python3 -m planner.validate --sizes 4096 16384 65536 --modes fp64 tf32 --output records.jsonl
    python3 -m planner.validate --report planner/validation/h200.jsonl

For every size and arithmetic it factors with the preset (where the paper measured one) and with
the schedule the planner selects from this GPU's profile, and records both times next to the
model's predictions.
"""

import argparse
import json
import math
from pathlib import Path

from machine import Machine
from machine.collect import cached

from .driver import BUILD, ROOT, measure
from .model import factor_time
from .plan import load_kernels, plan
from .schedule import MODES, Schedule

PRESETS = ROOT / "reproducers" / "presets.json"


def repetitions(n):
    return 5 if n <= 8192 else 3 if n <= 24000 else 2 if n <= 48000 else 1


def seconds(mode, n, s, build, reps):
    record = measure(mode, n, n, s, build, reps)
    return record["median_s"] if record else None


def validate(machine, kernels, mode, n, preset, build):
    """One case: the preset's time and the planned schedule's, each with the model's prediction."""
    reps = repetitions(n)
    record = {"machine": machine.name, "clock_hz": machine.clock, "mode": mode, "n": n}
    if preset:
        s = Schedule.from_options(preset["options"])
        record.update(
            preset_s=seconds(mode, n, s, build, reps),
            preset_model_s=factor_time(machine, kernels, mode, n, n, s),
            published_s=preset["published_time_s"],
        )
    predicted, s = plan(machine, mode, n, n, kernels)[0]
    record.update(
        planned=s.options(), planned_model_s=predicted, planned_s=seconds(mode, n, s, build, reps)
    )
    return record


def report(paths):
    """Markdown table and summary of stored validation records."""
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    by_case = {(r["mode"], r["n"]): r for r in rows}
    print(f"{rows[0]['machine']} at {rows[0]['clock_hz'] / 1e9:.2f} GHz\n")
    print(
        "| n | "
        + " | ".join(
            f"{m} preset (s) | planned (s) | ratio" for m in ("FP64", "FP32", "TF32", "3xTF32")
        )
        + " |"
    )
    print("| ---: |" + " ---: | ---: | ---: |" * 4)
    ratios, errors = [], []
    for n in sorted({r["n"] for r in rows}):
        cells = []
        for mode in MODES:
            r = by_case.get((mode, n), {})
            preset, planned = r.get("preset_s"), r.get("planned_s")
            if preset and planned:
                ratios.append(planned / preset)
            if planned:
                errors.append(abs(r["planned_model_s"] / planned - 1))
            cells += [
                f"{preset:.4g}" if preset else "–",
                f"{planned:.4g}" if planned else "–",
                f"{planned / preset:.3f}" if preset and planned else "–",
            ]
        print(f"| {n:,} | " + " | ".join(cells) + " |")
    ratios.sort()
    errors.sort()
    mean = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
    print(
        f"\n{len(ratios)} cases with a preset: planned over preset from {ratios[0]:.3f} to "
        f"{ratios[-1]:.3f}, median {ratios[len(ratios) // 2]:.3f}, geometric mean {mean:.3f}; "
        f"{sum(x <= 1.01 for x in ratios)} no more than 1 % slower, "
        f"{sum(x > 1.03 for x in ratios)} more than 3 % slower."
    )
    print(
        f"Model against measurement on the planned schedules: median error "
        f"{100 * errors[len(errors) // 2]:.1f} %, largest {100 * errors[-1]:.1f} %."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--report", type=Path, nargs="+", help="print stored records and exit")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--machine", type=Path, help="stored profile (default: probe the GPU)")
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--output", type=Path, help="append one JSON record per case")
    args = parser.parse_args()
    if args.report:
        report(args.report)
        return
    if not args.sizes:
        parser.error("--sizes is required")
    machine = Machine.load(args.machine) if args.machine else Machine(cached(args.build)[0])
    kernels = load_kernels()
    presets = {(c["mode"], c["n"]): c for c in json.loads(PRESETS.read_text()) if c["gpus"] == 1}
    for n in args.sizes:
        for mode in args.modes:
            record = validate(machine, kernels, mode, n, presets.get((mode, n)), args.build)
            if args.output:
                with args.output.open("a") as out:
                    out.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
