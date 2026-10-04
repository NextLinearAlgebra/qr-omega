"""Fit the kernel constants of the cost model to measured factorizations.

The constants are dimensionless: latency events per panel column, passes of the data through a
memory level, efficiencies of the carried products relative to the machine's product law. They
describe the kernels of the engine, not the machine, so they are fitted once and carried to other
GPUs by the profile. Refit them after changing a kernel.

    python3 -m planner.calibrate --measure runs.jsonl            # the calibration set on this GPU
    python3 -m planner.calibrate --fit runs.jsonl [more.jsonl]   # fit and write kernels.json
    python3 -m planner.calibrate --fit planner/calibration.jsonl # refit the shipped constants

The stages, each a least-squares fit of relative errors:
  1. the panel constants, from single-panel factorizations (n = b);
  2. the apply and compose constants, from factorizations without look-ahead, whose time is a sum;
  3. the constants a sum cannot separate, each from the runs in which the parameter it prices
     varies: the share of W that peers recover (wc), the cost of a strip, of a second strip in
     flight (depth);
  4. the overlap constants of each arithmetic, from factorizations with look-ahead.
"""

import argparse
import copy
import json
import math
from pathlib import Path

from machine import Machine
from machine.collect import cached
from machine.profile import solve

from .driver import BUILD, ROOT, measure
from .model import factor_time, overlap, sync_latency, unit_terms
from .plan import KERNELS, load_kernels
from .schedule import MODES, Schedule, panel_window, register_panel

APPLY = ("panel", "eta", "shape", "compose", "compose_launches")
SEPARATE = ("peers", "strip", "reread")
COOPERATIVE = ("handshakes", "launches", "shared", "boundary")


def least_squares(features, targets):
    """Non-negative least squares on relative errors: minimize sum((f.x / t - 1)^2) over x >= 0.
    The systems have a handful of unknowns, so every subset is solved without the constraint and
    the best solution that satisfies it is kept."""
    n = len(features[0])
    rows = [[f[i] / t for i in range(n)] for f, t in zip(features, targets)]
    best, best_x = None, [0.0] * n
    for mask in range(1, 1 << n):
        active = [i for i in range(n) if mask >> i & 1]
        ata = [[sum(r[i] * r[j] for r in rows) for j in active] for i in active]
        atb = [sum(r[i] for r in rows) for i in active]
        try:
            x = solve(ata, atb)
        except ArithmeticError:
            continue
        if min(x) < 0:
            continue
        residual = sum((sum(r[i] * v for i, v in zip(active, x)) - 1.0) ** 2 for r in rows)
        if best is None or residual < best:
            best, best_x = residual, [0.0] * n
            for i, v in zip(active, x):
                best_x[i] = v
    return best_x


def relative_errors(predicted, measured):
    """Root mean square and largest relative error."""
    e = [p / t - 1.0 for p, t in zip(predicted, measured)]
    return math.sqrt(sum(x * x for x in e) / len(e)), max(abs(x) for x in e)


def schedule(record):
    return Schedule.from_options(record["options"])


def predicted(machines, kernels, records, samples=0):
    return [
        factor_time(
            machines[r["profile"]], kernels, r["mode"], r["m"], r["n"], schedule(r), samples=samples
        )
        for r in records
    ]


def fit_panels(records, machines, kernels, report):
    for arithmetic in ("fp64", "fp32"):
        mode = arithmetic
        heights, cooperative = {}, []
        for r in records:
            s = schedule(r)
            if MODES[r["mode"]].panel != arithmetic or r["n"] != s.b:
                continue
            machine, rows = machines[r["profile"]], r["m"]
            sync = sync_latency(machine)
            if register_panel(rows, s.b):
                heights.setdefault(rows, []).append(r["time"] / (s.b * sync))
                continue
            width = panel_window(machine, kernels, mode, rows, s.groups, s.minipanel, s.window, s.b)
            boundaries = s.b / width - 1
            word = MODES[mode].word
            cooperative.append(
                (
                    [
                        s.b * sync,
                        boundaries * machine.launch_sync,
                        (rows / s.groups)
                        * s.b
                        * width
                        * word
                        / machine.levels["shared"].bandwidth_per_sm,
                        rows * s.b * boundaries * word / machine.levels["memory"].bandwidth_per_sm,
                    ],
                    r["time"],
                )
            )
        constants = kernels["panel"][arithmetic]
        if heights:
            constants["register"] = {str(h): sum(v) / len(v) for h, v in sorted(heights.items())}
            spread = max(max(v) / min(v) - 1.0 for v in heights.values())
            report.append(
                f"panel {arithmetic} register: {len(heights)} heights, "
                f"spread between machines {spread:.3f}"
            )
        if len(cooperative) >= len(COOPERATIVE):
            x = least_squares([f for f, _ in cooperative], [t for _, t in cooperative])
            constants["cooperative"] = dict(zip(COOPERATIVE, x))
            rms, worst = relative_errors(
                [sum(a * c for a, c in zip(f, x)) for f, _ in cooperative],
                [t for _, t in cooperative],
            )
            report.append(
                f"panel {arithmetic} cooperative: {len(cooperative)} runs, "
                f"rms {rms:.3f}, max {worst:.3f}"
            )


def factorizations(records, mode, lookahead=None, depth=1):
    """Runs of more than one panel in this arithmetic, optionally with or without look-ahead."""
    selected = []
    for r in records:
        s = schedule(r)
        if (
            r["mode"] == mode
            and r["n"] > s.b
            and s.depth == depth
            and (lookahead is None or s.lookahead == lookahead)
        ):
            selected.append(r)
    return selected


def fit_apply(records, machines, kernels, report, modes=MODES):
    """The apply constants enter the time linearly: each feature is the change of the prediction
    when one constant goes from 0 to 1."""
    for mode in modes:
        data = factorizations(records, mode, lookahead=False)
        if len(data) < len(APPLY):
            continue
        base = copy.deepcopy(kernels)
        kept = {key: kernels["apply"][mode][key] for key in SEPARATE + ("depth",)}
        base["apply"][mode] = {**dict.fromkeys(APPLY, 0.0), **kept}
        features, targets = [], []
        for r in data:
            machine, s = machines[r["profile"]], schedule(r)
            zero = factor_time(machine, base, mode, r["m"], r["n"], s)
            row = []
            for key in APPLY:
                unit = copy.deepcopy(base)
                unit["apply"][mode][key] = 1.0
                row.append(factor_time(machine, unit, mode, r["m"], r["n"], s) - zero)
            features.append(row)
            targets.append(max(r["time"] - zero, 0.05 * r["time"]))
        kernels["apply"][mode] = {**dict(zip(APPLY, least_squares(features, targets))), **kept}
        rms, worst = relative_errors(predicted(machines, kernels, data), [r["time"] for r in data])
        report.append(
            f"apply {mode}: {len(data)} runs without look-ahead, rms {rms:.3f}, max {worst:.3f}"
        )


def fit_overlap(records, machines, kernels, report):
    """Coordinate descent on the three overlap constants of every arithmetic."""
    for mode in MODES:
        data = factorizations(records, mode, lookahead=True)
        if len(data) < 6:
            continue
        # The unit terms do not depend on the overlap constants: tabulate them once.
        tables = []
        for r in data:
            machine, s = machines[r["profile"]], schedule(r)
            panels = -(-r["n"] // s.b)
            units, pending = [], 0.0
            for first in range(0, panels, s.aggregate):
                unit = unit_terms(machine, kernels, mode, r["m"], r["n"], s, first, panels)
                units.append((unit, pending))
                pending = unit.far
            tables.append((machine, units, r["time"]))

        def error(o):
            times = [
                sum(overlap(unit, pending, machine, o) + unit.window for unit, pending in units)
                for machine, units, _ in tables
            ]
            return relative_errors(times, [t for _, _, t in tables])

        grids = {
            "panel_slowdown": [1.0 + 0.05 * i for i in range(61)],
            "efficiency": [0.05 * i for i in range(31)],
            "joins": [0.5 * i for i in range(41)],
            "drain": [0.01 * i for i in range(51)],
        }
        o = {"panel_slowdown": 1.5, "efficiency": 1.0, "joins": 2.0, "drain": 0.0}
        for _ in range(4):
            for key, values in grids.items():
                o[key] = min(values, key=lambda v: error({**o, key: v})[0])
        kernels["overlap"][mode] = o
        rms, worst = error(o)
        report.append(
            f"overlap {mode}: {len(data)} runs with look-ahead, rms {rms:.3f}, max {worst:.3f}"
        )


def fit_separately(records, machines, kernels, report, name, grid, varies):
    """One apply constant that the runs without look-ahead cannot separate from the arithmetic:
    every candidate value refits the apply constants and is judged on the runs in which the
    parameter it prices varies. The arithmetic is never priced below a quarter of the fit without
    the constant."""
    for mode in MODES:
        data = varies(
            [
                r
                for r in records
                if r["mode"] == mode and r["n"] > schedule(r).b and schedule(r).depth == 1
            ]
        )
        if len(data) < 3:
            continue

        def error(value):
            kernels["apply"][mode][name] = value
            fit_apply(records, machines, kernels, [], (mode,))
            times = predicted(machines, kernels, data, samples=64)
            return relative_errors(times, [r["time"] for r in data])[0]

        error(0.0)
        floor = 0.25 * kernels["apply"][mode]["eta"]
        candidates = []
        for value in grid:
            rms = error(value)
            if kernels["apply"][mode]["eta"] >= floor:
                candidates.append((rms, value))
        best = min(candidates)[1]
        report.append(f"{name} {mode}: {len(data)} runs, rms {error(best):.3f}")


def fit_peers(records, machines, kernels, report):
    """The share of W that only more peers recover, from the runs whose wc is not the usual one."""

    def unusual(runs):
        counts = {}
        for r in runs:
            counts[schedule(r).wc] = counts.get(schedule(r).wc, 0) + 1
        usual = max(counts, key=counts.get) if counts else 0
        return [r for r in runs if schedule(r).wc != usual]

    fit_separately(
        records, machines, kernels, report, "peers", [0.25 * i for i in range(41)], unusual
    )


def fit_strips(records, machines, kernels, report):
    """The launches of one more strip and the reads of Z that leave L2, from series of runs that
    differ only in the strip width, each judged by its shape (every time relative to the mean of
    its series) so that only the dependence on the strip sets them."""
    for mode in MODES:
        series = {}
        for r in factorizations(records, mode):
            rest = {**r["options"], "strip": None}
            key = (r["profile"], r["m"], r["n"], json.dumps(rest, sort_keys=True))
            series.setdefault(key, []).append(r)
        series = [v for v in series.values() if len({schedule(r).strip for r in v}) >= 2]
        if len(series) < 2:
            continue

        def error(values):
            kernels["apply"][mode].update(values)
            fit_apply(records, machines, kernels, [], (mode,))
            total = count = 0
            for runs in series:
                times = predicted(machines, kernels, runs)
                measured = [r["time"] for r in runs]
                mean_p, mean_m = sum(times) / len(runs), sum(measured) / len(runs)
                total += sum(
                    ((p / mean_p) / (t / mean_m) - 1.0) ** 2 for p, t in zip(times, measured)
                )
                count += len(runs)
            return math.sqrt(total / count)

        without = error({"strip": 0.0, "reread": 0.0})
        best = {"strip": 0.0, "reread": 0.0}
        grids = {"strip": [2.0 * i for i in range(21)], "reread": [0.25 * i for i in range(13)]}
        for _ in range(2):
            for name, grid in grids.items():
                best[name] = min(grid, key=lambda value: error({**best, name: value}))
        if error(best) > 0.9 * without:  # no clear dependence on the strip
            best = {"strip": 0.0, "reread": 0.0}
        report.append(
            f"strips {mode}: {len(series)} series, shape rms {error(best):.4f} "
            f"(without: {without:.4f})"
        )


def fit_depth(records, machines, kernels, report):
    """The cost of a second strip in flight, from the runs with depth 2. Its measured gains stay
    below the resolution of the model, so it is priced as a cost or as nothing."""
    for mode in MODES:
        data = factorizations(records, mode, depth=2)
        if len(data) < 2:
            continue

        def error(depth):
            kernels["apply"][mode]["depth"] = depth
            times = predicted(machines, kernels, data, samples=64)
            return relative_errors(times, [r["time"] for r in data])[0]

        kernels["apply"][mode]["depth"] = min([0.02 * i for i in range(101)], key=error)
        report.append(
            f"depth {mode}: {len(data)} runs, rms {error(kernels['apply'][mode]['depth']):.3f}"
        )


def fit(records, machines, kernels):
    """Fit every stage. `records` are measurements with a "profile" naming an entry of `machines`;
    `kernels` supplies the constants that are not fitted (the static shared memory of the panel)."""
    kernels = copy.deepcopy(kernels)
    for mode in MODES:
        kernels["apply"][mode] = {**dict.fromkeys(APPLY + SEPARATE + ("depth",), 0.0), "panel": 1.0}
        kernels["overlap"][mode] = {
            "panel_slowdown": 1.0,
            "efficiency": 1.0,
            "joins": 0.0,
            "drain": 0.0,
        }
    report = []
    fit_panels(records, machines, kernels, report)
    fit_apply(records, machines, kernels, report)
    fit_overlap(records, machines, kernels, [])
    fit_peers(records, machines, kernels, report)
    fit_strips(records, machines, kernels, report)
    fit_overlap(records, machines, kernels, report)
    fit_depth(records, machines, kernels, report)
    return kernels, report


def calibration_set(machine, kernels):
    """The factorizations to measure: (mode, m, n, schedule). Every stage needs its contrast."""
    runs, b = [], 128
    for mode in ("fp64", "fp32"):
        for rows in (128, 256, 512, 1024, 2048, 4096, 8192):
            runs.append((mode, rows, b, Schedule(groups=0)))
        for rows in (12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072):
            for groups in (32, 64, 128):
                for minipanel, window in ((16, 64), (8, 32), (16, 0), (8, 0)):
                    s = Schedule(groups=groups, minipanel=minipanel, window=window)
                    if panel_window(machine, kernels, mode, rows, groups, minipanel, window):
                        runs.append((mode, rows, b, s))
    base = Schedule(wc=4, groups=32, minipanel=16, window=64)
    for mode in MODES:
        for n in (1024, 2048, 4096, 8192, 16384, 32768):
            strip = n - b if n <= 16384 else 8192
            for aggregate in (1, 2, 4, 8):
                for lookahead in (False, True):
                    runs.append(
                        (
                            mode,
                            n,
                            n,
                            base.replace(aggregate=aggregate, lookahead=lookahead, strip=strip),
                        )
                    )
        for n in (8192, 16384):
            for wc in (2, 8):
                runs.append(
                    (mode, n, n, base.replace(aggregate=2, lookahead=True, strip=n - b, wc=wc))
                )
        for n, strips in ((8192, (2016, 8064)), (32768, (4096, 8192, 16384, 32640))):
            for strip in strips:
                aggregate = 4 if n > 8192 else 2
                runs.append(
                    (mode, n, n, base.replace(aggregate=aggregate, lookahead=True, strip=strip))
                )
        for n in (16384, 32768):
            runs.append(
                (mode, n, n, base.replace(aggregate=4, lookahead=True, strip=8192, depth=2))
            )
        for n in (32768, 65536):  # the panel blocks against the far update beside them
            for groups in (32, 64, 128):
                for minipanel in (16, 8):
                    runs.append(
                        (
                            mode,
                            n,
                            n,
                            base.replace(
                                aggregate=8 if mode == "tf32" else 4,
                                lookahead=True,
                                strip=8192,
                                groups=groups,
                                minipanel=minipanel,
                                window=0,
                            ),
                        )
                    )
    return runs


def measure_set(path, build, profile_path=None, skip=()):
    if profile_path:
        machine = Machine.load(profile_path)
    else:
        profile, profile_path = cached(build)
        machine = Machine(profile)
    profile_path = Path(profile_path).resolve()
    if profile_path.is_relative_to(ROOT):
        profile_path = profile_path.relative_to(ROOT)
    kernels = load_kernels()
    with path.open("a") as out:
        for mode, m, n, s in calibration_set(machine, kernels):
            if (mode, m, n, json.dumps(s.options(), sort_keys=True)) in skip:
                continue
            record = measure(mode, m, n, s, build, reps=5 if n <= 8192 else 2)
            entry = {
                "profile": str(profile_path),
                "mode": mode,
                "m": m,
                "n": n,
                "options": s.options(),
                "time": record["median_s"] if record else None,
            }
            out.write(json.dumps(entry) + "\n")
            out.flush()
            print(mode, m, n, entry["time"], flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--measure", type=Path, help="measure the calibration set, append here")
    parser.add_argument("--fit", type=Path, nargs="+", help="measurement files to fit")
    parser.add_argument(
        "--machine", type=Path, help="stored profile of this GPU (default: probe it)"
    )
    parser.add_argument("--build", type=Path, default=BUILD)
    parser.add_argument("--output", type=Path, default=KERNELS)
    args = parser.parse_args()
    if args.measure:
        done = set()
        if args.measure.exists():
            for line in args.measure.read_text().splitlines():
                r = json.loads(line)
                done.add((r["mode"], r["m"], r["n"], json.dumps(r["options"], sort_keys=True)))
        measure_set(args.measure, args.build, args.machine, done)
    if args.fit:
        records = [
            json.loads(line)
            for path in args.fit
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        records = [r for r in records if r["time"]]
        machines = {p: Machine.load(ROOT / p) for p in {r["profile"] for r in records}}
        kernels, report = fit(records, machines, load_kernels())
        kernels["fitted_on"] = sorted(
            {f"{m.name} at {m.clock / 1e9:.2f} GHz" for m in machines.values()}
        )
        args.output.write_text(json.dumps(kernels, indent=1) + "\n")
        print("\n".join(report))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
