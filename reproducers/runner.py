#!/usr/bin/env python3
"""Measure the cells of the paper again and compare each with its recorded measurement.

A cell is one library, GPU count, arithmetic and matrix shape. QR-Omega's cells run the tuned driver
options of presets.json; the reference libraries' cells run their adapters (references/) with the
tuned configuration of reference-presets.csv. Every run must finish within --max-slowdown of the
recorded time and pass the numerical checks its recorded measurement passed; a reference measured
without them (validation "timing_only") is timed only, as it was recorded.

  python3 reproducers/runner.py --list                        # the measured cells
  python3 reproducers/runner.py --suite smoke                 # every mode at n = 4,096, one GPU
  python3 reproducers/runner.py --sizes 65536 131072          # QR-Omega, one GPU, every mode
  python3 reproducers/runner.py --suite tall                  # the tall-and-skinny shapes
  python3 reproducers/runner.py --gpus 8 --sizes 131072 --library cuSOLVERMp
"""

import argparse
import csv
import datetime
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "reproducers"
PRESETS = HERE / "presets.json"
REFERENCE_PRESETS = HERE / "reference-presets.csv"
OWN = "QR-Omega"
LIBRARIES = (OWN, "cuSOLVER", "cuSOLVERMp", "MAGMA", "SLATE")
MODES = ("fp64", "fp32", "tf32", "3xtf32")
GPU_COUNTS = (1, 2, 3, 4, 8)
SMOKE_N = 4096
REFERENCE_REPS = 3
# Host threads of each SLATE rank in the eight-GPU measurements.
SLATE_EIGHT_GPU_THREADS = {"OMP_NUM_THREADS": "7", "MKL_NUM_THREADS": "1"}
# Fixed for every run: eager module loading, and no TF32 in FP32 vendor products.
ENVIRONMENT = {"CUDA_MODULE_LOADING": "EAGER", "NVIDIA_TF32_OVERRIDE": "0"}


@dataclass
class Case:
    """One cell: what to run and, unless it is a smoke case, its recorded time and whether that
    measurement ran the numerical checks."""

    library: str
    gpus: int
    mode: str
    m: int
    n: int
    reps: int
    settings: dict = field(default_factory=dict)  # driver options or tuned reference configuration
    recorded_time_s: float | None = None
    checked: bool = True

    @property
    def square(self):
        return self.m == self.n

    @property
    def name(self):
        shape = str(self.n) if self.square else f"{self.m}x{self.n}"
        return f"{self.library}-p{self.gpus}-{self.mode}-{shape}"


def qr_omega_cases(path=PRESETS):
    return [
        Case(OWN, c["gpus"], c["mode"], c.get("m", c["n"]), c["n"], c["reps"], c["options"], c["measured_time_s"])
        for c in json.loads(path.read_text())
    ]


def parse_configuration(text):
    """'nb=64,threads=16' as a dictionary; cuSOLVER's FP32 math alone ('default', 'bf16x9') as math."""
    if "=" not in text:
        return {"math": text}
    return dict(item.split("=", 1) for item in text.split(","))


def reference_cases(path=REFERENCE_PRESETS):
    with path.open() as source:
        return [
            Case(
                row["library"],
                int(row["gpus"]),
                row["precision"],
                int(row["m"]),
                int(row["n"]),
                REFERENCE_REPS,
                parse_configuration(row["configuration"]),
                float(row["time_s"]),
                row["validation"] == "passed",
            )
            for row in csv.DictReader(source)
        ]


def is_headline(case):
    """The small-matrix speedups and the large throughputs quoted in the paper."""
    gpus, n, mode = case.gpus, case.n, case.mode
    if not case.square:
        return False
    if gpus == 1:
        return n in (16384, 65536, 131072) or (n in (256, 1024) and mode in ("fp64", "fp32"))
    if gpus in (4, 8):
        return n == 131072 or (n == 229376 and mode in ("fp64", "fp32")) or (n == 327680 and mode == "tf32")
    return n == 229376  # the strong-scaling points on two and three GPUs


def smoke_case(cases, mode, gpus):
    """A quick check at n = 4,096 with the options measured at that order on one GPU, and otherwise
    with those of the GPU count's smallest measured order (without its tail)."""
    own = [c for c in cases if c.library == OWN and c.square and c.gpus == gpus and c.mode == mode]
    base = next((c for c in own if c.n == SMOKE_N), None) if gpus == 1 else (own[0] if own else None)
    if base is None:
        raise ValueError(f"no measured {mode} cell on {gpus} GPUs to start a smoke case from")
    options = {k: v for k, v in base.settings.items() if gpus == 1 or not k.startswith("tail-")}
    return Case(OWN, gpus, mode, SMOKE_N, SMOKE_N, base.reps, options)


def parse_shape(text):
    m, _, n = text.lower().partition("x")
    return int(m), int(n or m)


def select(cases, args):
    """The cells of one library and GPU count that the suite, --sizes, --shapes and --modes ask for."""
    modes = args.modes or MODES
    if args.library != OWN:  # the references are compared in FP64 and FP32 only
        modes = sorted({"fp64" if mode == "fp64" else "fp32" for mode in modes}, key=MODES.index)
    if args.suite == "smoke":
        if args.library != OWN:
            raise ValueError("the smoke suite runs QR-Omega only")
        return [smoke_case(cases, mode, args.gpus) for mode in modes]
    pool = [c for c in cases if c.library == args.library and c.gpus == args.gpus and c.mode in modes]
    if args.sizes or args.shapes:
        wanted = {(n, n) for n in args.sizes or ()} | set(args.shapes or ())
        chosen = [c for c in pool if (c.m, c.n) in wanted]
        missing = {(m, n, mode) for m, n in wanted for mode in modes} - {(c.m, c.n, c.mode) for c in chosen}
        if missing:
            raise ValueError(f"not measured: {sorted(missing)}; --list shows the measured cells")
    elif args.suite == "tall":
        chosen = [c for c in pool if not c.square]
    elif args.suite == "all":
        chosen = pool
    else:
        chosen = [c for c in pool if is_headline(c)]
    if not chosen:
        raise ValueError("no measured cell matches the request; --list shows the measured cells")
    return sorted(chosen, key=lambda c: (c.m != c.n, c.m, c.n, MODES.index(c.mode)))


def qr_omega_command(case, args, record):
    binary = args.build / "bin" / ("qr_omega_single" if case.gpus == 1 else "qr_omega_multi")
    argv = [
        str(binary),
        "--mode",
        case.mode,
        "--m",
        str(case.m),
        "--n",
        str(case.n),
        "--reps",
        str(args.reps or case.reps),
    ]
    for key, value in case.settings.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is not False:
            argv += [f"--{key}", str(value)]
    argv += ["--output", str(record)]
    if case.gpus > 1:
        argv = [args.mpi, "--bind-to", "none", "--oversubscribe", "-np", str(case.gpus), *argv]
    return binary, argv, {}


def reference_command(case, args):
    """The adapter's command line and environment for a tuned reference configuration."""
    s, bin_dir = case.settings, args.build / "bin"
    shape = [case.mode, str(case.m), str(case.n)]
    reps = str(args.reps or case.reps)
    if "threads" in s:
        env = {"OMP_NUM_THREADS": s["threads"], "MKL_NUM_THREADS": s["threads"]}
    elif case.library == "SLATE" and case.gpus == 8:
        env = dict(SLATE_EIGHT_GPU_THREADS)
    else:
        env = {}
    if args.timing_only or not case.checked:
        env["QR_OMEGA_REFERENCE_TIMING_ONLY"] = "1"
    mpi = [args.mpi, "--bind-to", "none", "--oversubscribe", "-np", str(case.gpus)]
    if case.library == "cuSOLVER":
        binary = bin_dir / "qr_reference_cusolver"
        if s.get("math") == "bf16x9":
            env["QR_OMEGA_CUSOLVER_MATH"] = "bf16x9"
        argv = [str(binary), *shape, reps]
    elif case.library == "MAGMA":
        binary = bin_dir / "qr_reference_magma"
        env["QR_OMEGA_MAGMA_NB"] = s["nb"]
        argv = [str(binary), *shape, str(case.gpus), reps]
    elif case.library == "SLATE":
        binary = bin_dir / "qr_reference_slate"
        for key, variable in (
            ("ib", "INNER_BLOCK"),
            ("la", "LOOKAHEAD"),
            ("panel_threads", "PANEL_THREADS"),
            ("grid_rows", "GRID_ROWS"),
        ):
            if key in s:
                env[f"QR_OMEGA_SLATE_{variable}"] = s[key]
        argv = [*mpi, str(HERE / "references" / "one-gpu-per-rank.sh"), str(binary), *shape, s["nb"], reps]
    else:  # cuSOLVERMp
        binary = bin_dir / "qr_reference_cusolvermp"
        env["QR_OMEGA_CUSOLVERMP_GRID_ROWS"] = s["grid_rows"]
        argv = [*mpi, str(binary), *shape, s["nb"], reps]
    return binary, argv, env


def check_qr_omega_record(record, case, reps):
    """Reject a driver record unless the factorization ran as asked, was timed and passed its checks,
    and carried its products within the 2.5D bound c^2 <= p_i p_j. A square factorization must also
    replicate a product (c >= 2); the tuned schedules of some tall shapes carry every product with c = 1."""
    expected = {"status": 0, "pass": True, "m": case.m, "n": case.n, "mode": case.mode, "gpus": case.gpus}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{key} is {record.get(key)!r}, expected {value!r}")
    times = record.get("times_s") or []
    numbers = [record.get(key) for key in ("median_s", "residual", "orthogonality")] + times
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError("missing or invalid time or numerical error")
    if len(times) != reps or min(times) <= 0 or record["median_s"] != sorted(times)[reps // 2]:
        raise ValueError("incomplete or inconsistent timing samples")
    carriers = record.get("carriers", {})
    products = carriers.get("products", 0)
    if record.get("panel_algorithm") != "hqr" or carriers.get("bounded") != products:
        raise ValueError("not an HQR factorization with every product within the 2.5D bound")
    if case.square and not carriers.get("replicated", 0):
        raise ValueError("no product was replicated")
    return record["median_s"]


def check_reference_record(record, case, timing_only):
    """Reject an adapter record unless the factorization ran as asked and completed (SLATE's adapter
    reports a status only when it fails) and, unless only timed, passed its numerical checks."""
    if record.get("status", 0) != 0 or not record.get("median_s"):
        raise ValueError(f"the reference failed: {record.get('reason') or record.get('status')}")
    expected = {"precision": case.mode, "m": case.m, "n": case.n, "gpus": case.gpus}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{key} is {record.get(key)!r}, expected {value!r}")
    if not timing_only and (record.get("validation") or {}).get("pass") is not True:
        raise ValueError("the reference failed its numerical checks")
    return record["median_s"]


def run_qr_omega(case, args, argv, env, record_path, log):
    subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=True)
    return check_qr_omega_record(json.loads(record_path.read_text()), case, args.reps or case.reps)


def run_reference(case, args, argv, env, record_path, log):
    """Run an adapter, which prints one JSON record (its last line starting with a brace)."""
    result = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=log, text=True, timeout=args.timeout)
    log.write(result.stdout)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if not lines:
        raise ValueError(f"the reference printed no record (exit status {result.returncode})")
    record_path.write_text(lines[-1] + "\n")
    return check_reference_record(json.loads(lines[-1]), case, args.timing_only or not case.checked)


def run_case(case, args, output):
    """Run one cell; return its manifest entry (with "error" if it failed)."""
    record_path = output / f"{case.name}.json"
    if case.library == OWN:
        binary, argv, env = qr_omega_command(case, args, record_path)
    else:
        binary, argv, env = reference_command(case, args)
    print(" ".join([*(f"{k}={v}" for k, v in env.items()), shlex.join(argv)]), flush=True)
    entry = {"case": case.name, "command": argv, "environment": env, "record": record_path.name}
    if args.plan:
        return entry
    try:
        if not binary.is_file():
            raise FileNotFoundError(f"build it first: {binary}")
        run = run_qr_omega if case.library == OWN else run_reference
        with (output / f"{case.name}.log").open("w") as log:
            time = run(case, args, argv, {**os.environ, **ENVIRONMENT, **env}, record_path, log)
        entry["time_s"] = time
        note = ""
        if case.recorded_time_s:
            ratio = entry["time_over_recorded"] = time / case.recorded_time_s
            note = f"; {ratio:.3f} x recorded"
            if ratio > args.max_slowdown:
                raise ValueError(f"{ratio:.3f} times slower than recorded (allowed: {args.max_slowdown})")
        print(f"PASS {case.name}: {time:.6g} s{note}", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        entry["error"] = str(error)
        print(f"FAIL {case.name}: {error}; log: {output / case.name}.log", file=sys.stderr, flush=True)
    return entry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", choices=LIBRARIES, default=OWN, help=f"default: {OWN}")
    parser.add_argument("--gpus", type=int, choices=GPU_COUNTS, default=1)
    parser.add_argument(
        "--suite",
        choices=("headlines", "all", "tall", "smoke"),
        default="headlines",
        help="the cells the paper quotes (default), every measured cell of the GPU count, the "
        "tall-and-skinny shapes, or a quick check at n = 4,096",
    )
    parser.add_argument("--sizes", nargs="+", type=int, metavar="N", help="square orders to run instead of a suite")
    parser.add_argument("--shapes", nargs="+", type=parse_shape, metavar="MxN", help="shapes to run instead")
    parser.add_argument("--modes", nargs="+", choices=MODES, help="default: every mode")
    parser.add_argument("--build", type=Path, default=ROOT / "build", help="CMake build directory")
    parser.add_argument("--output", type=Path, help="results directory (default: results/<UTC time>)")
    parser.add_argument("--reps", type=int, help="timed repetitions instead of the recorded count")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per cell")
    parser.add_argument("--max-slowdown", type=float, default=1.04, help="allowed time over recorded time")
    parser.add_argument("--timing-only", action="store_true", help="skip the references' numerical checks")
    parser.add_argument("--mpi", default="mpirun", help="MPI launcher for several GPUs")
    parser.add_argument("--plan", action="store_true", help="print the commands and run nothing")
    parser.add_argument("--list", action="store_true", help="list the measured cells and run nothing")
    args = parser.parse_args(argv)
    if (args.reps is not None and args.reps < 1) or args.timeout < 1:
        parser.error("repetitions and timeout must be positive")
    if args.max_slowdown < 1:
        parser.error("the allowed slowdown must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    cases = qr_omega_cases() + reference_cases()
    if args.list:
        for c in sorted(cases, key=lambda c: (LIBRARIES.index(c.library), c.gpus, c.m != c.n, c.m, c.n)):
            print(
                f"{c.library:10s} {c.gpus} GPU{'s' if c.gpus > 1 else ' '} {c.mode:6s} {c.m:>8} x {c.n:<7} "
                f"{c.recorded_time_s:.6g} s"
            )
        return 0
    selected = select(cases, args)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "results" / stamp).resolve()
    if not args.plan:
        if args.gpus > 1 and shutil.which(args.mpi) is None:
            raise FileNotFoundError(f"MPI launcher not found: {args.mpi}")
        output.mkdir(parents=True)  # never overwrite earlier results
    entries = []
    for case in selected:
        entries.append(run_case(case, args, output))
        if not args.plan:
            (output / "manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    if args.plan:
        return 0
    failed = [e["case"] for e in entries if "error" in e]
    ratios = [e["time_over_recorded"] for e in entries if "time_over_recorded" in e]
    summary = f"{len(entries) - len(failed)} of {len(entries)} cells passed"
    if ratios:
        summary += f"; time over recorded {min(ratios):.3f}-{max(ratios):.3f}"
    print(f"{summary}; results in {output}")
    if failed:
        raise ValueError(f"{len(failed)} cell(s) failed: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # the output was piped into a command that stopped reading
        sys.exit(0)
    except (OSError, ValueError) as error:
        print(f"reproduction failed: {error}", file=sys.stderr)
        sys.exit(1)
