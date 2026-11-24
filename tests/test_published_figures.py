import unittest
import re

from src.author_data import ROOT
from src.verify_published import verify


class PublishedFigureTests(unittest.TestCase):
    def test_published_geometry_and_estimator_reanalysis(self):
        results = verify()
        self.assertTrue(results["passed"])
        self.assertEqual(len(results["digitized_figures"]), 13)
        self.assertEqual(sum(result["default_lines_checked"] for result in results["tuned_figures"]), 24)
        residuals = results["digitized_flow_audit"][28]["internal_residuals_pt"]
        self.assertAlmostEqual(residuals["Balancing"], 0.41559, places=5)

    def test_exactly_one_notebook_per_figure(self):
        figures = [int(re.match(r"figure_(\d{2})_", path.name)[1])
                   for path in (ROOT / "notebooks").glob("figure_*.ipynb")]
        self.assertEqual(sorted(figures), list(range(1, 41)))
        self.assertTrue(all((ROOT / "figures" / f"figure_{n:02d}.png").exists() for n in figures))
