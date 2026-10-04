"""Factor with a schedule through the runner and return the driver's record."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT / "reproducers"))
import runner  # noqa: E402


def command(mode, m, n, schedule, build=BUILD, reps=3, gpus=1, output="record.json"):
    """The runner case of a schedule and the command line that factors it."""
    case = {"gpus": gpus, "mode": mode, "m": m, "n": n, "reps": reps, "options": schedule.options()}
    args = argparse.Namespace(build=Path(build), reps=None, gpus=gpus, mpi="mpirun")
    return case, runner.command(case, args, Path(output))[1]


def measure(mode, m, n, schedule, build=BUILD, reps=3, gpus=1, timeout=7200):
    """Factor an m x n matrix `reps` times after a warm-up. Returns the driver record, or None when
    the driver refuses the schedule or a numerical check fails."""
    with tempfile.TemporaryDirectory() as directory:
        result = Path(directory) / "record.json"
        case, argv = command(mode, m, n, schedule, build, reps, gpus, result)
        try:
            subprocess.run(
                argv,
                env={**os.environ, **runner.ENVIRONMENT},
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return runner.load_record(result, case, reps)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
