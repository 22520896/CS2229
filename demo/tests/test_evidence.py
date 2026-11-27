import ast
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from demo import evidence as E
from demo import inventory as I


class EvidenceParity(unittest.TestCase):
    def test_all_inputs_match_their_frozen_hashes(self):
        self.assertGreater(len(E.verify_sources()["files"]), 30)

    def test_changed_input_is_rejected_without_overwriting_user_files(self):
        original = Path.read_bytes

        def changed(path):
            content = original(path)
            return (
                content + b"tampered"
                if path == E.ROOT / "reproduce/results/coverage.csv"
                else content
            )

        with patch.object(Path, "read_bytes", changed):
            with self.assertRaisesRegex(ValueError, "Nguồn dữ liệu không khớp"):
                E.verify_sources()

    def test_windows_and_git_line_endings_verify_the_same_evidence(self):
        original = Path.read_bytes

        def windows_lines(path):
            content = original(path)
            if path.suffix in {".csv", ".json"}:
                return content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            return content

        with patch.object(Path, "read_bytes", windows_lines):
            E.verify_sources()

    def test_paper_series_reconstruct_main_figures_exactly(self):
        references = {
            "target_smoothing": E.csv("figures/figure_02.csv")
            .groupby(["model_name", "transform__2__cov_mult"])
            .score.mean(),
            "noise_features": E.csv("figures/figure_04_4b_caption.csv").set_index(
                ["model_name", "fraction"]
            )["mean"],
        }
        for mechanism, reference in references.items():
            actual = E.mechanism_series(mechanism)
            for r in actual.loc[actual.source.eq("Paper")].itertuples():
                key = r.x / 100 if mechanism == "noise_features" else r.x
                self.assertAlmostEqual(
                    r.score, reference.loc[r.model, key], delta=1e-12
                )
        for fraction in [0.0, 0.5]:
            frame = E.csv("figures/figure_05.csv")
            reference = (
                frame.loc[frame.removed.eq(fraction)]
                .groupby(["model_name", "transform__2__deactivated"])
                .score.mean()
            )
            actual = E.mechanism_series("rotation", fraction=fraction)
            for r in actual.loc[actual.source.eq("Paper")].itertuples():
                self.assertAlmostEqual(
                    r.score, reference.loc[r.model, not bool(r.x)], delta=1e-12
                )

    def test_saved_own_raw_accuracies_are_unchanged(self):
        for mechanism in ["rotation", "noise_features", "target_smoothing"]:
            actual = E.mechanism_series(
                mechanism, dataset="electricity", normalized=False
            )
            raw = E.reference_cells(mechanism)
            raw = raw.loc[raw.dataset.eq("electricity")]
            for r in actual.itertuples():
                model = next(m for m, label in E.NAMES.items() if label == r.model)
                expected = raw.loc[
                    raw.model.eq(model) & raw.x.eq(r.x)
                ].test_accuracy_mean.iloc[0]
                self.assertEqual(r.score, expected)
            self.assertEqual(set(actual.source), {"Ours"})

    def test_hiding_a_model_cannot_change_scores_or_normalization(self):
        all_models = E.mechanism_series("noise_features", dataset="electricity")
        subset = all_models.loc[all_models.model.eq("RandomForest")]
        fresh = E.mechanism_series("noise_features", dataset="electricity")
        pd.testing.assert_frame_equal(
            subset.reset_index(drop=True),
            fresh.loc[fresh.model.eq("RandomForest")].reset_index(drop=True),
        )
        anchors = E.normalize_ours(
            E.reference_cells("noise_features"), "noise_features"
        )
        for r in anchors.itertuples():
            self.assertAlmostEqual(
                r.score,
                (r.test_accuracy_mean - r.q10) / (r.maximum - r.q10),
                delta=1e-12,
            )

    def test_completed_reference_cells_and_auto_repeat_counts(self):
        expected = {
            "rotation": (224, 560),
            "noise_features": (168, 420),
            "target_smoothing": (240, 576),
        }
        all_cells = E.cells()
        for mechanism, pair in expected.items():
            f = all_cells.loc[
                all_cells.mechanism.eq(mechanism) & all_cells.reference_grid
            ]
            self.assertEqual(
                (len(f), sum(len(json.loads(s)) for s in f.successful_seeds)), pair
            )

    def test_independent_predictions_and_missing_samples(self):
        for mechanism, condition in [
            ("rotation", [0.0, False]),
            ("noise_features", [2.0]),
            ("target_smoothing", [0.1, "robust"]),
        ]:
            sample, matrix = E.sample_numeric(mechanism, "electricity", "rf", condition)
            self.assertEqual(
                matrix.tolist(), sample["metrics"]["test"]["confusion_matrix"]
            )
        self.assertIsNone(
            E.sample_numeric("rotation", "electricity", "resnet", [0.0, False])
        )

    def test_benchmark_never_invents_unobserved_hpo_budgets(self):
        for kind, task, budget in [
            ("numerical", "classification", 2),
            ("numerical", "regression", 3),
            ("categorical", "classification", 29),
            ("categorical", "regression", 22),
        ]:
            own = E.benchmark_series("Ours", kind + "_" + task + "_medium")
            self.assertEqual(int(own["rank"].max()), budget)
            self.assertEqual(set(own["rank"]), set(range(budget + 1)))

    def test_inventory_has_all_40_figures_8_hpo_tables_and_separate_units(self):
        inventory = I.paper_inventory()
        self.assertEqual(
            set(inventory.loc[inventory.ID.str.startswith("Figure "), "ID"]),
            {f"Figure {i}" for i in range(1, 41)},
        )
        self.assertEqual(
            set(inventory.loc[inventory.ID.str.startswith("Table "), "ID"]),
            {f"Table {i}" for i in range(1, 9)},
        )
        self.assertEqual(sum(inventory["Phân tích tác giả"].eq("Đã tính lại")), 27)
        self.assertIn("Chưa", inventory.set_index("ID").loc["Figure 19", "Tự chạy"])
        coverage = I.coverage()
        self.assertEqual(len(coverage), 14)
        self.assertTrue(coverage["Độ phủ (%)"].between(0, 100).all())
        self.assertEqual(coverage["Đơn vị"].nunique(), 3)

    def test_no_training_worker_or_process_launcher_is_imported(self):
        forbidden = {
            "torch",
            "sklearn",
            "subprocess",
            "author_v2",
            "paper_extensions",
            "kaggle",
        }
        for filename in ["app.py", "evidence.py", "charts.py", "inventory.py"]:
            tree = ast.parse((E.ROOT / "demo" / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(
                        forbidden & {a.name.split(".")[0] for a in node.names}
                    )
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(
                        node.func.attr, {"fit", "fit_predict", "fit_transform", "Popen"}
                    )


if __name__ == "__main__":
    unittest.main()
