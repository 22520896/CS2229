import unittest

import numpy as np
import pandas as pd

from src.analysis import MODEL, TASK, TEST, VALID, random_search, r_random_stream, selected_curve, uniform
from src.verify import verify


class ValidationSelection(unittest.TestCase):
    def test_r_seed_one(self):
        reference = [0.2655086631421, 0.3721238996368, 0.5728533633519,
                     0.9082077899948, 0.2016819310375]
        np.testing.assert_allclose(uniform(r_random_stream(1), 5), reference, atol=1e-12, rtol=0)

    def test_selection_uses_validation_and_first_tie(self):
        ranks = np.arange(4)
        validation = np.array([0.6, 0.7, 0.7, 0.68])
        scores = np.array([0.4, 0.5, 0.95, 0.99])
        actual, _ = selected_curve(ranks, validation, scores, ranks)
        np.testing.assert_array_equal(actual, [0.4, 0.5, 0.5, 0.5])

    def test_duplicate_defaults_keep_weight(self):
        ranks = np.array([0, 0, 2, 3])
        actual, weight = selected_curve(ranks, [0.6, 0.7, 0.8, 0.75],
                                        np.array([0.5, 0.65, 0.7, 0.9]), np.arange(4))
        np.testing.assert_allclose(actual, [0.575, 0.65, 0.7, 0.7])
        self.assertEqual(weight, 2)

    def test_absent_group_stays_missing(self):
        source = pd.DataFrame({MODEL: ["RF", "GBT"], TASK: ["A", "B"],
                               VALID: [0.6, 0.7], TEST: [0.5, 0.65], "hp": ["default", "default"]})
        result = random_search(source, shuffles=1)
        self.assertTrue(result[(result[MODEL] == "RF") & (result[TASK] == "B")]["score"].isna().all())

    def test_saved_notebooks_and_r_parity(self):
        result = verify()
        self.assertEqual([r["rows"] for r in result["figures"]], [12030, 240, 20, 64, 480])


if __name__ == "__main__":
    unittest.main()
