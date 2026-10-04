#!/usr/bin/env python3
"""Run the QR-Omega configurations measured in the paper and compare each with its published time and errors."""

import argparse
import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "reproducers" / "presets.json"
MODES = ("fp64", "fp32", "tf32", "3xtf32")
SMOKE_N = 4096
ENVIRONMENT = {"CUDA_MODULE_LOADING": "EAGER", "NVIDIA_TF32_OVERRIDE": "0"}


def is_headline(case):
    """The small-matrix speedups and the large throughputs quoted in the paper."""
    gpus, n, mode = case["gpus"], case["n"], case["mode"]
    if gpus == 1:
        return n in (16384, 65536, 131072) or (n in (256, 1024) and mode in ("fp64", "fp32"))
    if gpus == 4:
        return (
            n == 131072
            or (n == 229376 and mode in ("fp64", "fp32"))
            or (n == 327680 and mode == "tf32")
        )
    return n == 229376  # the strong-scaling points on two and three GPUs


def smoke_case(presets, mode, gpus):
    """A quick unpublished case: the measured schedule of n = 4096 on one GPU, a shrunk one otherwise."""
    if gpus == 1:
        base = next(c for c in presets if (c["gpus"], c["mode"], c["n"]) == (1, mode, SMOKE_N))
        options = base["options"]
    else:
        base = next(c for c in presets if c["gpus"] > 1 and c["mode"] == mode)
        options = {**base["options"], "groups": 32}
    return {"gpus": gpus, "mode": mode, "n": SMOKE_N, "reps": base["reps"], "options": options}


def select(presets, args):
    modes = args.modes or MODES
    if args.suite == "smoke" and not args.sizes:
        return [smoke_case(presets, mode, args.gpus) for mode in modes]
    cases = [
        c
        for c in presets
        if c["gpus"] == args.gpus
        and c["mode"] in modes
        and (c["n"] in args.sizes if args.sizes else args.suite == "paper" or is_headline(c))
    ]
    measured = {(c["n"], c["mode"]) for c in cases}
    wanted = {(n, mode) for n in args.sizes or () for mode in args.modes or ()}
    unmeasured_sizes = set(args.sizes or ()) - {n for n, _ in measured}
    if not cases or wanted - measured or unmeasured_sizes:
        raise ValueError("no measured preset for the request; --list shows the measured cases")
    return sorted(cases, key=lambda c: (c["n"], MODES.index(c["mode"])))


def planned(args):
    """Cases of any size whose schedules the planner selects from the profile of the GPU."""
    sys.path.insert(0, str(ROOT))
    from machine import Machine
    from machine.collect import cached
    from planner.plan import case

    machine = Machine.load(args.machine) if args.machine else Machine(cached(args.build)[0])
    return [case(machine, mode, n, args.gpus) for n in args.sizes for mode in args.modes or MODES]


def command(case, args, result):
    binary = args.build / "bin" / ("qr_omega_single" if args.gpus == 1 else "qr_omega_multi")
    rows, columns = str(case.get("m", case["n"])), str(case["n"])
    reps = str(args.reps or case["reps"])
    argv = [str(binary), "--mode", case["mode"], "--m", rows, "--n", columns, "--reps", reps]
    for key, value in case["options"].items():
        argv += [f"--{key}"] if value is True else [f"--{key}", str(value)]
    argv += ["--output", str(result)]
    if args.gpus > 1:
        argv = [args.mpi, "--bind-to", "none", "--oversubscribe", "-np", str(args.gpus), *argv]
    return binary, argv


def load_record(path, case, reps):
    """Read a driver record; reject it unless the factorization ran, was timed and passed its checks."""
    record = json.loads(path.read_text())
    expected = {"status": 0, "pass": True, "m": case.get("m", case["n"]), "n": case["n"]}
    expected.update(mode=case["mode"], gpus=case["gpus"])
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{key} is {record.get(key)!r}, expected {value!r}")
    times = record.get("times_s") or []
    numbers = [record.get(key) for key in ("median_s", "residual", "orthogonality")] + times
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError("missing or invalid time or numerical error")
    if len(times) != reps or min(times) <= 0 or record["median_s"] != sorted(times)[reps // 2]:
        raise ValueError("incomplete or inconsistent timing samples")
    return record


def paper_ratios(case, record):
    """Measured over published time and errors; empty for a smoke case."""
    measured = {"time_s": record["median_s"], **record}
    return {
        key: measured[key] / case[f"published_{key}"]
        for key in ("time_s", "residual", "orthogonality")
        if f"published_{key}" in case
    }


def run_case(case, args, output):
    name = f"p{case['gpus']}-{case['mode']}-{case['n']}"
    result = output / f"{name}.json"
    binary, argv = command(case, args, result)
    print(shlex.join(argv), flush=True)
    if args.plan:
        return None
    if not binary.is_file():
        raise FileNotFoundError(f"build the driver first: {binary}")
    entry = {"case": case, "command": argv, "result": result.name}
    env = {**os.environ, **ENVIRONMENT}
    try:
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                argv,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=True,
            )
        record = load_record(result, case, args.reps or case["reps"])
        ratios = paper_ratios(case, record)
        entry.update(time_s=record["median_s"], ratios=ratios)
        errors = [ratios.get(key, 0) for key in ("residual", "orthogonality")]
        if ratios.get("time_s", 0) > args.max_slowdown or max(errors) > args.max_error_ratio:
            raise ValueError(f"worse than the paper: ratios {ratios}")
        note = ""
        if ratios:
            tflops = (4 / 3) * case["n"] ** 3 / record["median_s"] / 1e12
            note = f"; {tflops:.2f} TFLOP/s; {ratios['time_s']:.3f} x paper time"
        print(f"PASS {name}: {record['median_s']:.6g} s{note}", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        entry["error"] = str(error)
        print(f"FAIL {name}: {error}; log: {output / name}.log", file=sys.stderr, flush=True)
    return entry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--suite",
        choices=("headlines", "paper", "smoke"),
        default="headlines",
        help="headlines (default), every measured case, or a quick n = 4096 check",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="measured matrix orders to run instead"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, help="default: every measured mode")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, help="default: results/<timestamp>")
    parser.add_argument("--reps", type=int, help="timed repetitions instead of the recorded count")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per case")
    parser.add_argument(
        "--max-slowdown", type=float, default=1.04, help="allowed time / paper time"
    )
    parser.add_argument(
        "--max-error-ratio", type=float, default=1.25, help="allowed error / paper error"
    )
    parser.add_argument(
        "--auto", action="store_true", help="select the schedules of --sizes from the machine"
    )
    parser.add_argument("--machine", type=Path, help="machine profile for --auto (default: probe)")
    parser.add_argument("--mpi", default="mpirun", help="Open MPI launcher")
    parser.add_argument("--plan", action="store_true", help="print the commands and run nothing")
    parser.add_argument("--list", action="store_true", help="list the measured cases")
    args = parser.parse_args()
    if (args.reps is not None and args.reps < 1) or args.timeout < 1:
        parser.error("repetitions and timeout must be positive")
    if min(args.max_slowdown, args.max_error_ratio) < 1:
        parser.error("the allowed ratios must be at least 1")
    if args.auto and not args.sizes:
        parser.error("--auto needs --sizes")
    return args


def main():
    args = parse_args()
    presets = json.loads(PRESETS.read_text())
    if args.list:
        for case in presets:
            if case["gpus"] == args.gpus:
                print(case["gpus"], case["mode"], case["n"])
        return
    cases = planned(args) if args.auto else select(presets, args)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "results" / stamp).resolve()
    if not args.plan:
        if args.gpus > 1 and shutil.which(args.mpi) is None:
            raise FileNotFoundError(f"MPI launcher not found: {args.mpi}")
        output.mkdir(parents=True)  # never overwrite earlier results
    entries = []
    for case in cases:
        entries.append(run_case(case, args, output))
        if not args.plan:
            (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    failed = [entry["result"] for entry in entries if entry and "error" in entry]
    if not args.plan:
        print(f"results: {output}")
    if failed:
        raise ValueError(f"{len(failed)} case(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"reproduction failed: {error}", file=sys.stderr)
        raise SystemExit(1)
