"""Checks of the machine model and the planner that need no GPU."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from machine import Machine, Tuner  # noqa: E402
from planner.model import factor_time  # noqa: E402
from planner.plan import load_kernels, node_schedule, plan  # noqa: E402
from planner.schedule import MODES, Schedule, admissible, faithful, panel_window  # noqa: E402

PROFILES = [ROOT / "machine" / "profiles" / name for name in ("h200.json", "h200-1500mhz.json")]
PRESETS = json.loads((ROOT / "reproducers" / "presets.json").read_text())


class MachineModel(unittest.TestCase):
    def test_product_law_reproduces_the_probe(self):
        for path in PROFILES:
            machine = Machine.load(path)
            for arithmetic in machine.arithmetics():
                gemm, law = (
                    machine.data["arithmetic"]["gemm"][arithmetic],
                    machine.product(arithmetic),
                )
                n = gemm["rank_k"]["m"]
                for sample in gemm["rank_k"]["samples"]:
                    error = abs(law.rate(n, n, sample["k"]) / sample["flops_per_s"] - 1)
                    self.assertLess(error, 0.25, (path.name, arithmetic, sample["k"]))

    def test_hierarchy_is_ordered(self):
        for path in PROFILES:
            machine = Machine.load(path)
            latency = [machine.latency(level) for level in ("shared", "l1", "l2", "memory")]
            self.assertEqual(latency, sorted(latency))
            self.assertLess(machine.capacity("shared"), machine.capacity("l2"))
            self.assertLess(machine.bandwidth("memory", sms=1), machine.bandwidth("memory"))

    def test_projection_scales_the_balance(self):
        machine = Machine.load(PROFILES[0])
        projected = machine.scaled("projection", arithmetic=2.0)
        self.assertAlmostEqual(projected.peak("fp64") / machine.peak("fp64"), 2.0, places=6)
        self.assertGreater(projected.product("fp64").kappa, machine.product("fp64").kappa)


class Tuning(unittest.TestCase):
    def test_restrictions_rank_descend_refine(self):
        tuner = Tuner(
            {"x": [1, 2, 3, 4], "y": [1, 2, 3]},
            lambda c: (c["x"] - 3) ** 2 + (c["y"] - 2) ** 2 + 1.0,
            ["x + y <= 5", lambda c: c["x"] != 3],
        )
        _, best = tuner.best()
        self.assertEqual((best["x"], best["y"]), (2, 2))
        self.assertEqual(tuner.descend({"x": 1, "y": 1})[1], best)
        measured, _, table = tuner.refine(lambda c: 10.0 - c["x"], top=3)
        self.assertEqual(len(table), 3)
        self.assertEqual(measured, min(row["measured_s"] for row in table))


class Schedules(unittest.TestCase):
    def setUp(self):
        self.machine = Machine.load(PROFILES[0])
        self.kernels = load_kernels()

    def test_every_selected_schedule_is_faithful_and_fits(self):
        for mode in MODES:
            for n in (256, 1000, 4096, 20000, 65536, 131072):
                predicted, s = plan(self.machine, mode, n, n, self.kernels)[0]
                self.assertTrue(faithful(s), (mode, n, s))
                self.assertTrue(admissible(self.machine, self.kernels, mode, n, n, s)[0], (mode, n))
                self.assertGreater(predicted, 0)

    def test_c_equal_one_is_never_admissible(self):
        for change in ({"wc": 1}, {"zc": 1}, {"groups": 1}):
            s = Schedule(**change)
            self.assertFalse(admissible(self.machine, self.kernels, "fp64", 16384, 16384, s)[0])

    def test_capacity_refuses_what_does_not_fit(self):
        ok, why = admissible(
            self.machine, self.kernels, "fp64", 140000, 140000, Schedule(groups=128, minipanel=8)
        )
        self.assertFalse(ok)
        self.assertIn("memory", why)
        self.assertEqual(panel_window(self.machine, self.kernels, "fp64", 131072, 32, 16, 64), 0)
        self.assertEqual(panel_window(self.machine, self.kernels, "fp64", 131072, 128, 8, 64), 16)

    def test_presets_are_faithful_admissible_and_round_trip(self):
        for case in PRESETS:
            s = Schedule.from_options(case["options"])
            key = case["gpus"], case["mode"], case["n"]
            self.assertTrue(faithful(s), key)
            self.assertTrue(
                admissible(
                    self.machine, self.kernels, case["mode"], case["n"], case["n"], s, case["gpus"]
                )[0],
                key,
            )
            self.assertEqual(Schedule.from_options(s.options()), s)

    def test_model_predicts_measured_times(self):
        presets = {(c["mode"], c["n"]): c for c in PRESETS if c["gpus"] == 1}
        errors = []
        for line in (ROOT / "planner" / "validation" / "h200.jsonl").read_text().splitlines():
            record = json.loads(line)
            mode, n = record["mode"], record["n"]
            if record.get("preset_s") and 1024 <= n <= 65536:
                s = Schedule.from_options(presets[(mode, n)]["options"])
                seconds = factor_time(self.machine, self.kernels, mode, n, n, s)
                errors.append(abs(seconds / record["preset_s"] - 1))
        errors.sort()
        self.assertLess(errors[len(errors) // 2], 0.08)
        self.assertLess(errors[-1], 0.25)

    def test_node_schedule_follows_the_multi_gpu_presets(self):
        for case in PRESETS:
            if case["gpus"] > 1:
                mine = node_schedule(
                    self.machine, self.kernels, case["mode"], case["n"], case["gpus"]
                )
                published = Schedule.from_options(case["options"])
                self.assertEqual(mine.replace(groups=published.groups), published)
                self.assertLessEqual(abs(mine.groups - published.groups), 16)


if __name__ == "__main__":
    unittest.main()
