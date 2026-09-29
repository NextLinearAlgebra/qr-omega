#!/usr/bin/env python3
"""Run the measured QR-Omega configurations of the paper, keeping logs and numerical checks for every case."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("fp64", "fp32", "tf32", "3xtf32")


def replace_option(args, option, value):
    args[args.index(option) + 1] = str(value)


def cases(kind, sizes, modes, gpus, smoke):
    presets = json.loads((ROOT / "reproducers" / "presets" / f"{kind}.json").read_text())
    if smoke and kind == "multi-gpu":
        selected = []
        for mode in modes:
            template = next((c for c in presets if c["mode"] == mode),
                            next(c for c in presets if c["mode"] == "fp32"))
            cell = copy.deepcopy(template)
            cell = {k: v for k, v in cell.items() if not k.startswith("published_")}
            cell.update(n=4096, mode=mode, gpus=gpus,
                        published_validation="smoke test; not a paper measurement")
            for key, val in (("--m",4096), ("--n",4096), ("--gradix",gpus), ("--panel-groups",32)):
                replace_option(cell["args"], key, val)
            replace_option(cell["args"], "--precision", "fp64" if mode == "fp64" else "fp32")
            replace_option(cell["args"], "--fp32-math", {"fp64":"ieee", "fp32":"ieee", "tf32":"tf32", "3xtf32":"x3"}[mode])
            if mode == "3xtf32":
                cell["env"].update(TQR_X3RS="1", TQR_PACK_VR_TILED="1", TQR_GMMA_NP="1")
            selected.append(cell)
        return selected
    if modes is None:
        selected = [c for c in presets if c["n"] in sizes and c["gpus"] == gpus]
        missing_sizes = set(sizes) - {c["n"] for c in selected}
        if missing_sizes:
            raise ValueError(f"No measured presets for sizes {sorted(missing_sizes)} on {gpus} GPUs; use --list to see coverage")
        return sorted(selected, key=lambda c: (c["n"], MODES.index(c["mode"])))
    selected = [c for c in presets if c["n"] in sizes and c["mode"] in modes and c["gpus"] == gpus]
    missing = {(n, m) for n in sizes for m in modes} - {(c["n"],c["mode"]) for c in selected}
    if missing:
        raise ValueError(f"No measured presets for {sorted(missing)} on {gpus} GPUs; use --list to see coverage")
    return sorted(selected, key=lambda c: (c["n"], MODES.index(c["mode"])))


def cpu_affinity():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Cpus_allowed_list:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Linux CPU affinity information is unavailable")


def environment(cell):
    # Never let another experiment's TQR_* settings silently change a preset.
    env = {k:v for k,v in os.environ.items() if not k.startswith("TQR_")}
    env.update(cell["env"])
    cpus = cpu_affinity()
    env.update(TQR_REQUIRED_CPUS=cpus, TQR_EXPECT_CPUS=cpus,
               TQR_MEASUREMENT_CLASS="artifact-reproduction",
               CUDA_MODULE_LOADING="EAGER", NVIDIA_TF32_OVERRIDE="0",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def validate_result(path, n, *, mode=None, gpus=None, full_q=False):
    data = json.loads(path.read_text())
    check = data.get("validation", {})
    if data.get("status") != 0 or data.get("eligible_numerically") is not True or check.get("pass") is not True:
        raise ValueError(f"Factorization or numerical validation failed: {path}")
    if check.get("full_residual_coverage") is not True or check.get("input_columns_checked") != n:
        raise ValueError(f"Incomplete reconstruction check: {path}")
    if data.get("m") != n or data.get("n") != n:
        raise ValueError(f"Unexpected matrix dimensions: {path}")
    expected_mode = "fp32x3" if mode == "3xtf32" else mode
    if mode is not None and data.get("precision_mode") != expected_mode:
        raise ValueError(f"Unexpected precision mode: {path}")
    if gpus is not None and data.get("gpus") != gpus:
        raise ValueError(f"Unexpected GPU count: {path}")
    full_q_check = check.get("full_Q_Gram", {}).get("pass")
    if full_q_check is False or (full_q and full_q_check is not True):
        raise ValueError(f"Full-Q check failed: {path}")
    timing = data.get("timing", {}).get("reused_median_s")
    if not isinstance(timing, (int, float)) or not math.isfinite(timing) or timing <= 0:
        raise ValueError(f"Missing or invalid factorization time: {path}")
    return data


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, help="Measured matrix orders (default: 131072)")
    parser.add_argument("--modes", nargs="+", choices=MODES,
                        help="Default: available measured modes at each size; all modes for --smoke")
    parser.add_argument("--gpus", type=int, default=1 if kind == "single-gpu" else 4)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reps", type=int, help="Override the recorded repetition count")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-case timeout in seconds, including validation")
    parser.add_argument("--smoke", action="store_true", help="Validate all requested modes at n=4096")
    parser.add_argument("--plan", action="store_true", help="Print commands without launching or writing files")
    parser.add_argument("--list", action="store_true", help="List available measured presets")
    parser.add_argument("--mpi", default="mpirun", help="Open MPI launcher executable")
    args = parser.parse_args()
    if args.list:
        for cell in json.loads((ROOT / "reproducers" / "presets" / f"{kind}.json").read_text()):
            print(cell["gpus"], cell["mode"], cell["n"])
        return
    if args.gpus not in ((1,) if kind == "single-gpu" else (2,3,4)):
        parser.error("single-gpu requires one GPU; multi-gpu supports 2, 3, or 4 GPUs within one node")
    if args.reps is not None and args.reps < 1 or args.timeout < 1:
        parser.error("Repetitions and timeout must be positive")
    if args.smoke and args.sizes:
        parser.error("Use either --smoke or --sizes")
    sizes = [4096] if args.smoke else args.sizes or [131072]
    modes = args.modes if args.modes is not None else (list(MODES) if args.smoke else None)
    selected = cases(kind, sizes, modes, args.gpus, args.smoke)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "results" / f"{kind}-{stamp}").resolve()
    if not args.plan:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite results: {output}")
        if kind == "multi-gpu" and shutil.which(args.mpi) is None:
            raise FileNotFoundError(f"MPI launcher not found: {args.mpi}")
    commands = []
    for cell in selected:
        binary = "qr_omega_single" if kind == "single-gpu" else (
            "qr_omega_multi_x3" if cell["mode"] == "3xtf32" else "qr_omega_multi")
        binary_path = (args.build / "bin" / binary).resolve()
        argv = list(cell["args"])
        if args.reps is not None:
            replace_option(argv, "--reps", args.reps)
        if args.smoke and kind == "single-gpu":
            argv.append("--full-q")
        result = output / f"p{args.gpus}-{cell['mode']}-{cell['n']}.json"
        command = [str(binary_path), *argv, "--output", str(result)]
        if kind == "multi-gpu":
            # One rank per GPU; --oversubscribe lets Open MPI launch them from a single-task allocation step.
            command = [args.mpi, "--bind-to", "none", "--oversubscribe", "-np", str(args.gpus), *command]
        commands.append((cell, binary_path, result, command))
        if not args.plan and not binary_path.is_file():
            raise FileNotFoundError(f"Build the driver first: {binary_path}")
    if not args.plan:
        output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema":1, "kind":kind, "created_utc":stamp, "cases":[]}
    for cell, binary_path, result, command in commands:
        env = environment(cell)
        preset_env = {k: v for k, v in env.items()
                      if k.startswith("TQR_") or k in ("CUDA_MODULE_LOADING", "NVIDIA_TF32_OVERRIDE",
                                                       "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        unset = [item for key in sorted(os.environ) if key.startswith("TQR_") for item in ("-u", key)]
        print(shlex.join(["env", *unset, *(f"{k}={v}" for k,v in preset_env.items()), *command]), flush=True)
        if args.plan:
            continue
        item = {"preset":cell, "command":command, "environment":preset_env,
                "cpu_affinity":env["TQR_REQUIRED_CPUS"],
                "binary_sha256":hashlib.sha256(binary_path.read_bytes()).hexdigest(),
                "cuda_visible_devices":env.get("CUDA_VISIBLE_DEVICES"), "result":result.name}
        manifest["cases"].append(item)
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        try:
            with result.with_suffix(".log").open("w") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout, check=True)
            data = validate_result(result, cell["n"], mode=cell["mode"], gpus=args.gpus,
                                   full_q=args.smoke and kind == "single-gpu")
            item.update(returncode=completed.returncode, validation="passed",
                        time_s=data["timing"]["reused_median_s"],
                        result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
            print(f"PASS {result.name}: {item['time_s']:.6g} s; full-column residual + orthogonality checks", flush=True)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            item.update(validation="failed", error=str(exc))
            raise
        finally:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.plan:
        print(f"Results: {output}")


def entry(kind):
    try:
        main(kind)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"reproduction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
