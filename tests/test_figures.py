"""The figures and the numbers of the paper regenerate exactly from the recorded measurements."""

import csv
import importlib.metadata
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
sys.path.insert(0, str(PLOTS))
import measurements as ms  # noqa: E402
import paper_numbers  # noqa: E402


def pinned_plotting_versions():
    """Whether the versions of plots/requirements.txt are installed: they draw the figures bit for bit."""
    pins = dict(line.split("==") for line in (PLOTS / "requirements.txt").read_text().split())
    try:
        return all(importlib.metadata.version(name) == version for name, version in pins.items())
    except importlib.metadata.PackageNotFoundError:
        return False


class PaperNumberTests(unittest.TestCase):
    def test_the_quoted_numbers_are_recomputed_from_the_measurements(self):
        claims = paper_numbers.claims(ms.check(ms.read()))
        with (PLOTS / "figures/paper-numbers.csv").open() as source:
            recorded = list(csv.reader(source))[1:]
        self.assertEqual([[c.section, c.statement, c.paper, c.data, c.status] for c in claims], recorded)


@unittest.skipUnless(pinned_plotting_versions(), "needs the plotting versions of plots/requirements.txt")
class FigureTests(unittest.TestCase):
    def test_the_figures_regenerate_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            subprocess.run(
                [
                    sys.executable,
                    str(PLOTS / "reproduce.py"),
                    "--output",
                    str(output / "figures"),
                    "--table",
                    str(output / "paper.csv"),
                ],
                check=True,
                capture_output=True,
            )
            for path in sorted((PLOTS / "figures").iterdir()):
                with self.subTest(figure=path.name):
                    self.assertEqual((output / "figures" / path.name).read_bytes(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
