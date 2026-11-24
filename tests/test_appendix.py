import unittest

import numpy as np

from src.time_benchmark import time_winners
from src.verify_appendix import verify


class AppendixTests(unittest.TestCase):
    def test_time_curve_selects_by_validation_and_preserves_exact_ties(self):
        result = time_winners(np.array([10.0, 0.0, 1.0, 8.0, 1.0, 2.0, 10.0]),
                              np.array([0.5, 0.5, 0.5, 0.4, 0.3, 0.3, 0.9]),
                              np.array([0.51, 0.52, 0.99, 0.98, 0.97, 0.96, 0.40]))
        self.assertEqual(result["time_bin"].tolist(), [1, 1, 1, 2, 2, 3])
        np.testing.assert_allclose(result["score"], [0.51, 0.52, np.nan, np.nan, np.nan, 0.40],
                                   equal_nan=True)

    def test_saved_outputs_match_the_independent_references(self):
        result = verify()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["time_figures"]), 2)
        self.assertEqual(len(result["curves"]), 19)
        self.assertEqual(result["a4"]["coefficients"], 34)
