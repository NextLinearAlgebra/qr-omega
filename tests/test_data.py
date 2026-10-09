"""The recorded measurements agree with each other: the table the figures are drawn from and the tuned
configurations the runner replays come from the measured data; no GPU needed."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "reproducers"))
import measurements as ms  # noqa: E402
import runner  # noqa: E402


class TableTests(unittest.TestCase):
    def test_the_table_is_assembled_from_the_recorded_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "paper.csv"
            ms.assemble(table=table)
            self.assertEqual(table.read_bytes(), ms.TABLE.read_bytes())

    def test_the_table_passes_its_checks(self):
        rows = ms.check(ms.read())
        own = [row for row in rows if row["library"] == ms.OWN]
        self.assertEqual(len(own), len(json.loads(ms.PRESETS.read_text())))

    def test_a_repeated_measurement_is_rejected(self):
        rows = ms.read()
        with self.assertRaisesRegex(ValueError, "repeated"):
            ms.check(rows + rows[:1])

    def test_a_throughput_inconsistent_with_its_time_is_rejected(self):
        rows = ms.read()
        rows[0] = {**rows[0], "tflops": rows[0]["tflops"] * 1.01}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ms.check(rows)

    def test_a_square_qr_omega_cell_needs_its_numerical_checks(self):
        rows = ms.read()
        index = next(i for i, row in enumerate(rows) if row["library"] == ms.OWN and row["m"] == row["n"])
        rows[index] = {**rows[index], "validation": "timing_only"}
        with self.assertRaisesRegex(ValueError, "without numerical checks"):
            ms.check(rows)


class PresetTests(unittest.TestCase):
    def test_every_cell_is_measured_once(self):
        cases = runner.qr_omega_cases() + runner.reference_cases()
        keys = [(c.library, c.gpus, c.mode, c.m, c.n) for c in cases]
        self.assertEqual(len(keys), len(set(keys)))
        for case in cases:
            with self.subTest(case=case.name):
                self.assertGreaterEqual(case.m, case.n)
                self.assertGreater(case.recorded_time_s, 0)

    def test_every_square_qr_omega_cell_was_validated(self):
        for cell in json.loads(ms.PRESETS.read_text()):
            if cell.get("m", cell["n"]) == cell["n"]:
                with self.subTest(gpus=cell["gpus"], mode=cell["mode"], n=cell["n"]):
                    self.assertIsNotNone(cell["measured_residual"])
                    self.assertIsNotNone(cell["measured_orthogonality"])

    def test_every_tuned_reference_is_the_fastest_measured_configuration(self):
        fastest = {}
        for row in ms.read():
            if row["kind"] == "qr" and row["library"] != ms.OWN:
                key = row["library"], row["gpus"], row["precision"], row["m"], row["n"]
                if key not in fastest or row["time_s"] < fastest[key]["time_s"]:
                    fastest[key] = row
        with runner.REFERENCE_PRESETS.open() as source:
            tuned = list(csv.DictReader(source))
        self.assertEqual(len(tuned), len(fastest))
        for preset in tuned:
            # the table names cuSOLVERMp by the cuSOLVER family it belongs to
            library = "cuSOLVER" if preset["library"] == "cuSOLVERMp" else preset["library"]
            key = library, int(preset["gpus"]), preset["precision"], int(preset["m"]), int(preset["n"])
            with self.subTest(key=key):
                row = fastest[key]
                self.assertEqual(float(preset["time_s"]), row["time_s"])
                self.assertEqual(preset["validation"], row["validation"])
                self.assertEqual(json.loads(row["configuration"]), {"tuned": preset["configuration"]})


if __name__ == "__main__":
    unittest.main()
