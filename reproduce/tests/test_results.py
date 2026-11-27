"""Real-output regression checks;no synthetic benchmark or training."""

import copy
import importlib.util
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "submission_results", BASE / "scripts/check_results.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ResultsTests(unittest.TestCase):
    def test_all_aggregate_and_numeric_evidence(self):
        self.assertTrue(checks.verify(manifest=False)["passed"])

    def test_altered_test_metric_is_rejected(self):
        sample = copy.deepcopy(checks.read("results/sample_fits.json")[0])
        sample["metrics"]["test"]["accuracy"] += 0.01
        with self.assertRaises(AssertionError):
            checks.sample_metrics(sample)

    def test_altered_numeric_hash_is_rejected(self):
        sample = copy.deepcopy(checks.read("results/sample_fits.json")[0])
        sample["numeric_sha256"] = "0" * 64
        with self.assertRaises(AssertionError):
            checks.sample_metrics(sample)

    def test_manifest_does_not_claim_full_paper_or_full_numeric_reaudit(self):
        status = checks.read("results/run_status.json")
        self.assertFalse(status["full_paper_complete"])
        self.assertEqual(status["native_pending_or_active"], 1464)
        self.assertEqual(
            checks.verify(manifest=False)["sample_numeric_fits_checked"], 3
        )


if __name__ == "__main__":
    unittest.main()
