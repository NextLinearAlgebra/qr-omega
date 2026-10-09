"""The runner: the cells it selects, the commands it runs and how it judges a run; no GPU needed."""

import copy
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

CASES = runner.qr_omega_cases() + runner.reference_cases()


def request(*argv, **overrides):
    """The runner's arguments for a command line, with attributes replaced by keyword."""
    args = runner.parse_args(list(argv))
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def cell(library, gpus, mode, n, m=None):
    return next(c for c in CASES if (c.library, c.gpus, c.mode, c.m, c.n) == (library, gpus, mode, m or n, n))


class SelectionTests(unittest.TestCase):
    def test_headlines_are_the_cells_the_paper_quotes(self):
        names = [c.name for c in runner.select(CASES, request())]
        self.assertEqual(len(names), 16)
        self.assertIn("QR-Omega-p1-fp64-256", names)
        self.assertNotIn("QR-Omega-p1-tf32-256", names)
        self.assertIn("QR-Omega-p1-3xtf32-131072", names)

    def test_sizes_and_shapes_select_measured_cells(self):
        selected = runner.select(CASES, request("--sizes", "65536", "--shapes", "131072x64", "--modes", "fp64"))
        self.assertEqual([c.name for c in selected], ["QR-Omega-p1-fp64-65536", "QR-Omega-p1-fp64-131072x64"])

    def test_unmeasured_cells_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not measured"):
            runner.select(CASES, request("--gpus", "4", "--sizes", "327680", "--modes", "fp64"))

    def test_the_tall_suite_runs_every_tall_shape(self):
        selected = runner.select(CASES, request("--suite", "tall"))
        self.assertTrue(selected)
        self.assertTrue(all(not c.square and c.gpus == 1 for c in selected))
        self.assertEqual(len({(c.m, c.n) for c in selected}), 11)

    def test_references_are_compared_in_their_fp32(self):
        selected = runner.select(CASES, request("--library", "MAGMA", "--sizes", "16384", "--modes", "tf32"))
        self.assertEqual([c.name for c in selected], ["MAGMA-p1-fp32-16384"])

    def test_a_smoke_case_replays_the_options_measured_at_its_order(self):
        selected = runner.select(CASES, request("--suite", "smoke"))
        self.assertEqual([c.mode for c in selected], list(runner.MODES))
        for case in selected:
            with self.subTest(mode=case.mode):
                self.assertEqual(case.settings, cell(runner.OWN, 1, case.mode, runner.SMOKE_N).settings)
                self.assertIsNone(case.recorded_time_s)

    def test_a_smoke_case_on_several_gpus_drops_the_tail_options(self):
        for case in runner.select(CASES, request("--suite", "smoke", "--gpus", "8")):
            with self.subTest(mode=case.mode):
                self.assertEqual(case.n, runner.SMOKE_N)
                self.assertFalse([key for key in case.settings if key.startswith("tail-")])


class CommandTests(unittest.TestCase):
    def test_qr_omega_replays_the_recorded_options_and_omits_false_switches(self):
        case = runner.Case(runner.OWN, 1, "fp64", 4096, 4096, 3, {"b": 64, "lookahead": True, "tail-lookahead": False})
        _, argv, env = runner.qr_omega_command(case, request(build=Path("/b")), Path("/r.json"))
        self.assertEqual(
            argv,
            [
                "/b/bin/qr_omega_single",
                "--mode",
                "fp64",
                "--m",
                "4096",
                "--n",
                "4096",
                "--reps",
                "3",
                "--b",
                "64",
                "--lookahead",
                "--output",
                "/r.json",
            ],
        )
        self.assertEqual(env, {})

    def test_several_gpus_run_one_rank_per_gpu(self):
        case = cell(runner.OWN, 8, "fp64", 131072)
        _, argv, _ = runner.qr_omega_command(case, request(), Path("r.json"))
        self.assertEqual(argv[:6], ["mpirun", "--bind-to", "none", "--oversubscribe", "-np", "8"])
        self.assertTrue(argv[6].endswith("bin/qr_omega_multi"))

    def test_each_reference_runs_its_tuned_configuration(self):
        magma = cell("MAGMA", 8, "fp64", 131072)
        _, argv, env = runner.reference_command(magma, request())
        self.assertEqual(argv[1:], ["fp64", "131072", "131072", "8", "3"])
        self.assertEqual(env["QR_OMEGA_MAGMA_NB"], magma.settings["nb"])
        self.assertEqual(env["OMP_NUM_THREADS"], magma.settings["threads"])

        slate = cell("SLATE", 8, "fp64", 131072)
        _, argv, env = runner.reference_command(slate, request())
        self.assertTrue(argv[6].endswith("one-gpu-per-rank.sh"))
        self.assertEqual(env["OMP_NUM_THREADS"], "7")
        self.assertEqual(env["QR_OMEGA_SLATE_GRID_ROWS"], slate.settings["grid_rows"])

        mp = cell("cuSOLVERMp", 8, "fp32", 131072)
        _, argv, env = runner.reference_command(mp, request("--timing-only"))
        self.assertEqual(env["QR_OMEGA_CUSOLVERMP_GRID_ROWS"], mp.settings["grid_rows"])
        self.assertEqual(env["QR_OMEGA_REFERENCE_TIMING_ONLY"], "1")

        emulated = next(c for c in CASES if c.library == "cuSOLVER" and c.settings.get("math") == "bf16x9")
        _, argv, env = runner.reference_command(emulated, request())
        self.assertEqual(env["QR_OMEGA_CUSOLVER_MATH"], "bf16x9")

    def test_a_reference_runs_the_numerical_checks_of_its_recorded_measurement(self):
        checked = cell("cuSOLVER", 1, "fp64", 16384)
        timed = cell("cuSOLVER", 1, "fp64", 256, m=4194304)
        self.assertEqual((checked.checked, timed.checked), (True, False))
        self.assertNotIn("QR_OMEGA_REFERENCE_TIMING_ONLY", runner.reference_command(checked, request())[2])
        self.assertEqual(runner.reference_command(timed, request())[2]["QR_OMEGA_REFERENCE_TIMING_ONLY"], "1")


# The record of a driver run that passed: two timed repetitions on one GPU.
PASSED_RUN = {
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
    "panel_algorithm": "hqr",
    "carriers": {"products": 4, "bounded": 4, "replicated": 3},
}


class RecordTests(unittest.TestCase):
    case = runner.Case(runner.OWN, 1, "fp64", 4096, 4096, 2)

    def check(self, changes=None, reps=2):
        return runner.check_qr_omega_record({**copy.deepcopy(PASSED_RUN), **(changes or {})}, self.case, reps)

    def test_accepts_the_upper_median_of_an_even_number_of_repetitions(self):
        self.assertEqual(self.check(), 2.0)

    def test_rejects_failed_or_mismatched_runs(self):
        for key, value in (("status", 1), ("pass", False), ("n", 2048), ("mode", "tf32"), ("gpus", 4)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check({key: value})

    def test_rejects_invalid_errors_and_timings(self):
        for key, value in (
            ("residual", float("nan")),
            ("orthogonality", float("inf")),
            ("median_s", 1.5),
            ("times_s", []),
            ("times_s", [float("nan"), 2.0]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check({key: value})
        with self.assertRaises(ValueError):
            self.check(reps=3)

    def test_products_must_stay_within_the_bound_and_replicate(self):
        for carriers in ({"products": 4, "bounded": 3, "replicated": 3}, {"products": 4, "bounded": 4}):
            with self.subTest(carriers=carriers), self.assertRaises(ValueError):
                self.check({"carriers": carriers})

    def test_a_tall_schedule_may_carry_every_product_without_replication(self):
        tall = runner.Case(runner.OWN, 1, "fp64", 131072, 256, 2)
        record = {**PASSED_RUN, "m": 131072, "n": 256, "carriers": {"products": 76, "bounded": 76, "replicated": 0}}
        self.assertEqual(runner.check_qr_omega_record(record, tall, 2), 2.0)

    def test_a_reference_must_pass_its_checks_unless_only_timed(self):
        failed = {"status": 0, "precision": "fp64", "m": 4096, "n": 4096, "gpus": 1, "median_s": 1.0}
        failed["validation"] = {"pass": False}
        with self.assertRaises(ValueError):
            runner.check_reference_record(failed, self.case, timing_only=False)
        self.assertEqual(runner.check_reference_record(failed, self.case, timing_only=True), 1.0)

    def test_a_reference_reports_a_status_only_when_it_fails(self):
        passed = {"precision": "fp64", "m": 4096, "n": 4096, "gpus": 1, "median_s": 1.0, "validation": {"pass": True}}
        self.assertEqual(runner.check_reference_record(passed, self.case, timing_only=False), 1.0)
        for changes in ({"status": "failed"}, {"n": 2048}, {"precision": "fp32"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runner.check_reference_record({**passed, **changes}, self.case, timing_only=False)


# A stand-in for the one-GPU driver: it reports the recorded time of its cell scaled by TIME_RATIO.
FAKE_DRIVER = """#!{python}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
def option(key): return args[args.index(key) + 1]
m, n, mode, reps = int(option('--m')), int(option('--n')), option('--mode'), int(option('--reps'))
cell = next(c for c in json.loads(Path({presets!r}).read_text())
            if (c['gpus'], c['mode'], c.get('m', c['n']), c['n']) == (1, mode, m, n))
time = cell['measured_time_s'] * float(os.environ['TIME_RATIO'])
record = {{'mode': mode, 'm': m, 'n': n, 'gpus': 1, 'status': 0, 'pass': True,
          'times_s': [time] * reps, 'median_s': time, 'residual': 1e-15, 'orthogonality': 1e-15,
          'panel_algorithm': 'hqr', 'carriers': {{'products': 1, 'bounded': 1, 'replicated': 1}}}}
Path(option('--output')).write_text(json.dumps(record))
"""


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        driver = self.root / "build/bin/qr_omega_single"
        driver.parent.mkdir(parents=True)
        driver.write_text(FAKE_DRIVER.format(python=sys.executable, presets=str(runner.PRESETS)))
        driver.chmod(0o755)

    def run_cli(self, *argv, ratio="1"):
        command = [sys.executable, str(ROOT / "reproducers/runner.py"), "--build", str(self.root / "build")]
        command += ["--output", str(self.root / "results"), *argv]
        environment = {**os.environ, "TIME_RATIO": ratio}
        return subprocess.run(command, env=environment, capture_output=True, text=True, check=False)

    def manifest(self):
        return json.loads((self.root / "results/manifest.json").read_text())

    def test_a_slow_but_correct_run_fails_and_keeps_its_record(self):
        result = self.run_cli("--sizes", "16384", "--modes", "fp64", ratio="1.045")
        self.assertEqual(result.returncode, 1, result.stderr)
        entry = self.manifest()[0]
        self.assertAlmostEqual(entry["time_over_recorded"], 1.045)
        self.assertIn("error", entry)
        self.assertTrue((self.root / "results" / entry["record"]).is_file())

    def test_a_four_percent_slowdown_passes(self):
        result = self.run_cli("--sizes", "16384", "--shapes", "131072x64", "--modes", "fp64", ratio="1.04")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([("error" in e) for e in self.manifest()], [False, False])

    def test_smoke_cases_have_no_recorded_time(self):
        result = self.run_cli("--suite", "smoke", "--modes", "fp64", ratio="10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("time_over_recorded", self.manifest()[0])

    def test_plan_runs_nothing(self):
        result = self.run_cli("--modes", "fp64", "--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "results").exists())
        for size in ("256", "1024", "131072"):
            self.assertIn(f"--n {size} ", result.stdout)

    def test_existing_results_are_never_overwritten(self):
        self.assertEqual(self.run_cli("--sizes", "16384", "--modes", "fp64").returncode, 0)
        before = (self.root / "results/manifest.json").read_bytes()
        self.assertEqual(self.run_cli("--sizes", "16384", "--modes", "fp64").returncode, 1)
        self.assertEqual((self.root / "results/manifest.json").read_bytes(), before)

    def test_list_prints_every_measured_cell(self):
        result = self.run_cli("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), len(CASES))


if __name__ == "__main__":
    unittest.main()
