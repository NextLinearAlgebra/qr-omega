#!/usr/bin/env python3
"""Measure the machine profile of a GPU, or print one.

python3 -m machine.collect --output machine/profiles/mine.json     # run the probe (inside the GPU allocation)
python3 -m machine.collect --show machine/profiles/h200.json       # basic metrics of a stored profile
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .profile import Machine

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "machine" / "profiles" / "cache"


def probe(build=ROOT / "build", device=0, output=None, quick=False):
    """Run machine_probe on a device and return the profile."""
    binary = Path(build) / "bin" / "machine_probe"
    if not binary.is_file():
        raise FileNotFoundError(
            f"Build the probe first (cmake --build {build} --target machine_probe): {binary}"
        )
    command = [str(binary), "--device", str(device)] + (["--quick"] if quick else [])
    data = json.loads(subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(data, indent=2) + "\n")
    return data


def cached(build=ROOT / "build", device=0, refresh=False):
    """The profile of a device, measured once per GPU, CUDA driver and probe version. Returns (profile, path)."""
    binary = Path(build) / "bin" / "machine_probe"
    if not binary.is_file():
        raise FileNotFoundError(
            f"Build the probe first (cmake --build {build} --target machine_probe): {binary}"
        )
    who = json.loads(
        subprocess.run(
            [str(binary), "--device", str(device), "--identify"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    path = CACHE / f"{who['uuid']}-{who['cuda_driver']}.json"
    if path.is_file() and not refresh:
        data = json.loads(path.read_text())
        if data.get("probe") == who["probe"]:
            return data, path
    return probe(build, device, path), path


def show(machine, out=sys.stdout):
    s = machine.summary()
    w = out.write
    w(
        f"{machine.name}: {machine.sms} SMs at {machine.clock / 1e9:.3f} GHz, {machine.warps_per_sm} warps per SM\n\n"
    )
    w(f"{'level':10s} {'capacity':>12s} {'latency':>18s} {'bandwidth':>12s} {'per SM':>12s}\n")
    for name, lv in machine.levels.items():
        cap = "-" if lv.capacity == float("inf") else f"{lv.capacity / 2**20:.3g} MiB"
        band = "-" if lv.bandwidth == float("inf") else f"{lv.bandwidth / 1e9:.4g} GB/s"
        per = (
            "-"
            if lv.bandwidth_per_sm == float("inf") or lv.scope == "node"
            else f"{lv.bandwidth_per_sm / 1e9:.4g} GB/s"
        )
        cycles = (
            f"{lv.latency * 1e9:7.1f} ns {lv.latency * machine.clock:6.0f} cyc"
            if lv.scope != "node"
            else f"{lv.latency * 1e6:7.2f} us"
        )
        w(f"{name:10s} {cap:>12s} {cycles:>18s} {band:>12s} {per:>12s}\n")
    w(
        f"\nmemory write {s['memory_write_bytes_per_s'] / 1e9:.4g} GB/s, copy (read + write) {s['memory_copy_bytes_per_s'] / 1e9:.4g} GB/s\n"
    )
    w(
        f"launch {machine.launch * 1e6:.2f} us (asynchronous), {machine.launch_sync * 1e6:.2f} us (and wait); "
        f"stream join {machine.join * 1e6:.2f} us; barrier {machine.barrier(1024) * 1e9:.0f} ns\n\n"
    )
    w(
        f"{'arithmetic':10s} {'peak':>12s} {'1/gamma':>12s} {'kappa':>8s} {'xi':>6s} {'balance':>14s} {'grain':>12s}\n"
    )
    for a in machine.arithmetics():
        law = machine.product(a)
        w(
            f"{a:10s} {machine.peak(a) / 1e12:8.1f} TF/s {1e-12 / law.gamma:8.1f} TF/s {law.kappa:8.1f} {law.xi:6.2f} "
            f"{machine.balance(a):9.1f} op/B {machine.grain(a):9.2e} op\n"
        )
    w(
        f"scalar     fp32 {machine.scalar_rate('fp32') / 1e12:.1f} TF/s, fp64 {machine.scalar_rate('fp64') / 1e12:.1f} TF/s\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index within CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument("--output", type=Path, help="Write the measured profile here")
    parser.add_argument("--quick", action="store_true", help="Smaller sweeps (about half the time)")
    parser.add_argument(
        "--show", type=Path, help="Print the basic metrics of a stored profile and exit"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the flat metric dictionary as JSON"
    )
    args = parser.parse_args()
    data = (
        json.loads(args.show.read_text())
        if args.show
        else probe(args.build, args.device, args.output, args.quick)
    )
    machine = Machine(data)
    if args.json:
        print(json.dumps(machine.summary(), indent=2))
    else:
        show(machine)


if __name__ == "__main__":
    main()
