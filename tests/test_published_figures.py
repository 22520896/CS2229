import unittest
import re

from src.author_data import ROOT
from src.verify_published import verify


class PublishedFigureTests(unittest.TestCase):
    def test_estimator_outputs_match_reference(self):
        results = verify()
        self.assertTrue(results["passed"])
        self.assertEqual(sum(result["default_lines_checked"] for result in results["tuned_figures"]), 24)

    def test_exactly_one_notebook_per_reproduced_figure(self):
        figures = [int(re.match(r"figure_(\d{2})_", path.name)[1])
                   for path in (ROOT / "notebooks").glob("figure_*.ipynb")]
        expected = [*range(1, 19), *range(20, 25), 33, 34, 39, 40]
        self.assertEqual(sorted(figures), expected)
        self.assertTrue(all((ROOT / "figures" / f"figure_{n:02d}.png").exists() for n in figures))
