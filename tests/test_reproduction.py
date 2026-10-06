"""Consistency of the presets with the paper data and behavior of the runner; no GPU required."""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproducers"))
import runner  # noqa: E402

PRESETS = json.loads(runner.PRESETS.read_text())


class PresetTests(unittest.TestCase):
    def test_every_qr_omega_row_of_the_paper_data_is_a_measured_preset(self):
        with (ROOT / "plots/data/paper.csv").open() as source:
            published = {
                (int(row["gpus"]), row["precision"], int(row["n"])): row
                for row in csv.DictReader(source)
                if row["library"] == "TQR"
            }
        seen = set()
        for case in PRESETS:
            key = case["gpus"], case["mode"], case["n"]
            with self.subTest(key=key):
                self.assertNotIn(key, seen)
                seen.add(key)
                row = published[key]
                self.assertEqual(case["measured_time_s"], float(row["time_s"]))
                self.assertEqual(case["measured_residual"], float(row["residual"]))
                self.assertEqual(case["measured_orthogonality"], float(row["orthogonality"]))
        self.assertEqual(seen, set(published))

    def test_unmeasured_combinations_are_rejected(self):
        args = argparse.Namespace(gpus=4, suite="paper", sizes=[327680], modes=["fp64"])
        with self.assertRaisesRegex(ValueError, "no measured preset"):
            runner.select(PRESETS, args)


class RecordTests(unittest.TestCase):
    case = {"gpus": 1, "mode": "fp64", "n": 4096}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "record.json"
        self.record = {
            "mode": "fp64",
            "m": 4096,
            "n": 4096,
            "gpus": 1,
            "status": 0,
            "pass": True,
            "times_s": [1.0, 2.0],
            "median_s": 2.0,
            "residual": 1e-15,
            "orthogonality": 1e-15,
        }

    def load(self, record, reps=2):
        self.path.write_text(json.dumps(record))
        return runner.load_record(self.path, self.case, reps)

    def test_accepts_the_upper_median_of_an_even_number_of_repetitions(self):
        self.assertEqual(self.load(self.record)["median_s"], 2.0)

    def test_rejects_failed_or_mismatched_runs(self):
        for key, value in (
            ("status", 1),
            ("pass", False),
            ("n", 2048),
            ("mode", "tf32"),
            ("gpus", 4),
        ):
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                record[key] = value
                with self.assertRaises(ValueError):
                    self.load(record)

    def test_rejects_invalid_errors_and_timings(self):
        for key, value in (
            ("residual", float("nan")),
            ("orthogonality", float("inf")),
            ("median_s", 1.5),
            ("times_s", []),
            ("times_s", [float("nan"), 2.0]),
        ):
            with self.subTest(key=key, value=value):
                record = copy.deepcopy(self.record)
                record[key] = value
                with self.assertRaises(ValueError):
                    self.load(record)
        with self.assertRaises(ValueError):
            self.load(self.record, reps=3)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        binary = self.root / "build/bin/qr_omega_single"
        binary.parent.mkdir(parents=True)
        # A stand-in driver that reports the measured time of its case scaled by FAKE_TIME_RATIO.
        binary.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "def option(key): return args[args.index(key) + 1]\n"
            "n, mode, reps = int(option('--n')), option('--mode'), int(option('--reps'))\n"
            f"case = next(c for c in json.loads(Path({str(runner.PRESETS)!r}).read_text()) "
            "if (c['gpus'], c['mode'], c['n']) == (1, mode, n))\n"
            "time = case['measured_time_s'] * float(os.getenv('FAKE_TIME_RATIO', '1'))\n"
            "record = {'mode': mode, 'm': n, 'n': n, 'gpus': 1, 'status': 0, 'pass': True,\n"
            "          'times_s': [time] * reps, 'median_s': time,\n"
            "          'residual': case['measured_residual'],\n"
            "          'orthogonality': case['measured_orthogonality'], 'panel_algorithm': 'hqr',\n"
            "          'carriers': {'products': 1, 'bounded': 1, 'replicated': 1}}\n"
            "Path(option('--output')).write_text(json.dumps(record))\n"
        )
        binary.chmod(0o755)

    def run_cli(self, *args, ratio="1"):
        command = [sys.executable, str(runner.ROOT / "reproducers/runner.py")]
        command += [
            "--build",
            str(self.root / "build"),
            "--presets",
            str(runner.PRESETS),
            "--output",
            str(self.root / "results"),
            *args,
        ]
        environment = {**os.environ, "FAKE_TIME_RATIO": ratio}
        return subprocess.run(command, env=environment, capture_output=True, text=True, check=False)

    def manifest(self):
        return json.loads((self.root / "results/manifest.json").read_text())

    def test_a_slow_but_correct_run_fails_and_keeps_its_record(self):
        result = self.run_cli("--sizes", "16384", "--modes", "fp64", ratio="1.045")
        self.assertEqual(result.returncode, 1, result.stderr)
        entry = self.manifest()[0]
        self.assertAlmostEqual(entry["ratios"]["time_s"], 1.045)
        self.assertIn("error", entry)
        self.assertTrue((self.root / "results" / entry["result"]).is_file())

    def test_a_four_percent_slowdown_passes(self):
        result = self.run_cli("--sizes", "16384", "--modes", "fp64", ratio="1.04")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("error", self.manifest()[0])

    def test_smoke_cases_have_no_recorded_time(self):
        result = self.run_cli("--suite", "smoke", "--modes", "fp64", ratio="10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.manifest()[0]["ratios"], {})

    def test_plan_runs_nothing(self):
        result = self.run_cli("--suite", "headlines", "--modes", "fp64", "--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "results").exists())
        for size in ("256", "1024", "131072"):
            self.assertIn(f"--n {size} ", result.stdout)

    def test_existing_results_are_never_overwritten(self):
        self.assertEqual(self.run_cli("--sizes", "16384", "--modes", "fp64").returncode, 0)
        before = (self.root / "results/manifest.json").read_bytes()
        self.assertEqual(self.run_cli("--sizes", "16384", "--modes", "fp64").returncode, 1)
        self.assertEqual((self.root / "results/manifest.json").read_bytes(), before)

    def test_false_switches_are_omitted(self):
        case = {
            "gpus": 1,
            "mode": "fp64",
            "n": 4096,
            "reps": 1,
            "options": {"lookahead": True, "tail-lookahead": False},
        }
        args = argparse.Namespace(build=self.root, gpus=1, reps=None)
        _, command = runner.command(case, args, self.root / "record.json")
        self.assertIn("--lookahead", command)
        self.assertNotIn("--tail-lookahead", command)

    def test_reproduction_compares_with_the_measured_time(self):
        case = {"measured_time_s": 2}
        self.assertEqual(runner.reproduction_ratios(case, {"median_s": 2.02}), {"time_s": 1.01})


if __name__ == "__main__":
    unittest.main()
