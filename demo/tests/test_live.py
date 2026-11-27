import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from demo import benchmark as B, inventory as I, live as L
from src import analysis as A


def unit_fixture():
    """Test-only generated data; never stored as demo or paper evidence."""
    rng = np.random.default_rng(24)
    frame = pd.DataFrame(rng.normal(size=(320, 6)), columns=[f"x{i}" for i in range(6)])
    frame["class"] = (frame.x0 + frame.x1 * frame.x2 > 0).astype(int)
    return frame


class UpstreamAndLive(unittest.TestCase):
    def test_cached_datasets_are_first_and_explicitly_labelled(self):
        with patch.object(
            L.D,
            "local_path",
            side_effect=lambda name: L.ROOT
            / (
                "README.md"
                if name in L.PREFERRED_DATASETS
                else "not-cached-unit-fixture"
            ),
        ):
            self.assertEqual(L.dataset_options()[:3], list(L.PREFERRED_DATASETS))
            self.assertIn("Có sẵn", L.dataset_label("MagicTelescope"))
            self.assertIn("Chưa tải", L.dataset_label("Higgs"))

    def test_rotation_comparison_keeps_both_cases_and_matches_upstream(self):
        data = L.prepare(unit_fixture(), "class", 160, 80, 0)
        result = L.compare(
            "rotation",
            data,
            ["RandomForest"],
            [0],
            settings={"remove_halves": [False, True]},
        )
        self.assertEqual(len(result["fits"]), 4)
        self.assertEqual(result["output"].variant.nunique(), 2)
        for half, label in [(False, "Giữ toàn bộ feature"), (True, "Bỏ 50% feature")]:
            expected = L.run(
                "rotation", data, ["RandomForest"], [0], settings={"remove_half": half}
            )["output"]
            actual = (
                result["output"]
                .loc[result["output"].variant.eq(label)]
                .drop(columns="variant")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "ít nhất"):
            L.compare(
                "rotation", data, ["RandomForest"], [0], settings={"remove_halves": []}
            )

    def test_smoothing_comparison_separates_covariance_and_label_changes(self):
        data = L.prepare(unit_fixture(), "class", 160, 80, 0)
        result = L.compare(
            "smoothing",
            data,
            ["RandomForest"],
            [0],
            settings={"levels": [0.0, 0.1], "k": 5, "covariance_modes": [False, True]},
        )
        self.assertEqual(len(result["fits"]), 4)
        self.assertEqual(result["label_changes"].variant.nunique(), 2)
        for robust, label in [
            (False, "Covariance thường"),
            (True, "Robust · MinCovDet"),
        ]:
            expected = L.run(
                "smoothing",
                data,
                ["RandomForest"],
                [0],
                settings={"levels": [0.0, 0.1], "k": 5, "robust": robust},
            )["output"][0]
            actual = (
                result["output"]
                .loc[result["output"].variant.eq(label)]
                .drop(columns=["variant", "seconds"])
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(actual, expected.drop(columns="seconds"))

    def test_removal_can_select_only_removed_features_or_both(self):
        data = L.prepare(unit_fixture(), "class", 160, 80, 0)
        result = L.compare(
            "remove",
            data,
            ["RandomForest"],
            [0],
            settings={"levels": [0.5], "train_parts": ["removed"]},
        )
        self.assertEqual(result["output"].trained_on.tolist(), ["removed"])
        both = L.compare(
            "remove",
            data,
            ["RandomForest"],
            [0],
            settings={"levels": [0.5], "train_parts": ["kept", "removed"]},
        )
        self.assertEqual(set(both["output"].trained_on), {"kept", "removed"})

    def test_pinned_original_core_is_unchanged(self):
        source = L.verify_upstream()
        self.assertEqual(source["commit"], "e9fce1eba2fb498b9571b33a331832a3eef63626")

    def test_prepare_matches_original_without_a_new_preprocessing_pipeline(self):
        raw = unit_fixture()
        expected = L.D.prepare(raw, "class", 160, 80, 7)
        actual = L.prepare(raw, "class", 160, 80, 7)
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(expected, name)
            )

    def test_all_four_mechanisms_match_original_outputs_and_keep_test_labels(self):
        data = L.prepare(unit_fixture(), "class", 160, 80, 0)
        unchanged = data.y_test.copy()
        cases = [
            (
                "smoothing",
                {"levels": [0.0, 0.1], "k": 5, "robust": False},
                lambda: L.X.run_smoothing(
                    data, [0.0, 0.1], 5, False, ["RandomForest"], [0]
                ),
            ),
            (
                "add",
                {"levels": [0.0, 0.5]},
                lambda: L.X.run_add_features(data, [0.0, 0.5], ["RandomForest"], [0]),
            ),
            (
                "remove",
                {"levels": [0.5], "on_removed": True},
                lambda: L.X.run_remove_features(
                    data, [0.5], ["RandomForest"], [0], True
                ),
            ),
            (
                "rotation",
                {"remove_half": True},
                lambda: L.X.run_rotation(data, True, ["RandomForest"], [0]),
            ),
        ]
        for kind, settings, original in cases:
            with self.subTest(kind=kind):
                with L.bounded():
                    expected = original()
                actual = L.run(kind, data, ["RandomForest"], [0], settings=settings)
                if kind == "smoothing":
                    pd.testing.assert_frame_equal(actual["output"][1], expected[1])
                    expected, actual_frame = expected[0], actual["output"][0]
                else:
                    actual_frame = actual["output"]
                pd.testing.assert_frame_equal(
                    actual_frame.drop(columns="seconds", errors="ignore"),
                    expected.drop(columns="seconds", errors="ignore"),
                )
                np.testing.assert_array_equal(data.y_test, unchanged)
        L.verify_upstream()

    def test_boundary_fits_real_models_and_rejects_duplicate_axes(self):
        data = L.prepare(unit_fixture(), "class", 160, 80, 0)
        result = L.run(
            "boundary",
            data,
            ["RandomForest", "MLP"],
            [0],
            settings={"features": [0, 1]},
        )
        self.assertEqual(set(result["output"]), {"RandomForest", "MLP"})
        self.assertEqual(result["output"]["RandomForest"]["model"].n_jobs, 1)
        self.assertEqual(len(result["fits"]), 2)
        fig = L.boundary_plot(data, result["output"], (0, 1), (-2, 2), (-2, 2))
        self.assertEqual(len([t for t in fig.data if t.type == "heatmap"]), 2)
        with self.assertRaisesRegex(ValueError, "khác nhau"):
            L.run(
                "boundary", data, ["RandomForest"], [0], settings={"features": [0, 0]}
            )

    def test_only_one_training_action_runs_at_a_time(self):
        self.assertTrue(L.LOCK.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(ValueError, "đang chạy"):
                with L.bounded():
                    self.fail("Concurrent action was allowed")
        finally:
            L.LOCK.release()


class BenchmarkAndScope(unittest.TestCase):
    def test_csv_reanalysis_uses_the_verified_author_engine(self):
        records = []
        for dataset in ["unit-a", "unit-b"]:
            for model in ["RandomForest", "MLP"]:
                for index in range(6):
                    records.append(
                        dict(
                            model_name=model,
                            data__keyword=dataset,
                            mean_val_score=0.6 + index * 0.02,
                            mean_test_score=0.7 - index * 0.01,
                            hp="default" if index == 0 else "random",
                            mean_time=1.0,
                        )
                    )
        frame = B.read_csv(BytesIO(pd.DataFrame(records).to_csv(index=False).encode()))
        actual, has_defaults = B.curves(frame, "custom_classification", 3, 10)
        normalized = A.normalize(frame, quantile=0.1, clip_test=False)
        search = A.random_search(
            normalized, shuffles=3, default_first=True, match_tasks=True
        )
        expected = A.bands(
            A.across_tasks(search, ("rank", A.MODEL, "shuffle"), skip_missing=True),
            ("rank", A.MODEL),
        )
        expected = expected.rename(columns={A.MODEL: "model"})
        expected["datasets"] = 2
        self.assertTrue(has_defaults)
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_bad_upload_is_rejected(self):
        with self.assertRaises(ValueError):
            B.read_csv(b"model_name,mean_test_score\nMLP,0.9\n")

    def test_partial_grids_never_become_100_percent_paper(self):
        from demo.progress_estimates import all_estimates

        scope = I.paper_scope()
        percent = scope["Ước lượng tự train (%)"]
        self.assertTrue(percent.dropna().between(0, 100, inclusive="left").all())
        estimates = all_estimates()
        self.assertAlmostEqual(estimates["Figure 2"]["percent"], 100 / 60)
        self.assertEqual(estimates["Figure 4b"]["percent"], 5.0)
        self.assertEqual(estimates["Figure 5"]["percent"], 5.0)
        self.assertEqual(estimates["Figure 1"]["denominator"], (35 * 7 + 24 * 8) * 400)
        self.assertEqual(estimates["Figure 8–11, 16–17"]["numerator"], 38006)
        self.assertTrue(scope["Cơ sở ước lượng"].str.len().gt(0).all())
        self.assertNotIn("Phân tích (%)", I.paper_inventory())
        self.assertNotIn("percent", I.work_volume())


class LiveAppFlows(unittest.TestCase):
    def test_csv_read_compute_and_navigation_preserve_imported_results(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory(
            prefix="demo-csv-test-", dir=L.ROOT / ".cache"
        ) as folder:
            path = Path(folder) / "unit-results.csv"
            frame = pd.DataFrame(
                {
                    "model_name": ["RandomForest"] * 6,
                    "data__keyword": ["unit-fixture"] * 6,
                    "mean_val_score": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
                    "mean_test_score": [0.62, 0.63, 0.68, 0.71, 0.72, 0.74],
                    "hp": ["default"] + ["random"] * 5,
                }
            )
            frame.to_csv(path, index=False)
            app = AppTest.from_file(
                str(L.ROOT / "demo/app.py"), default_timeout=30
            ).run()
            app.radio(key="bench_source").set_value("CSV của bạn").run()
            app.text_input(key="bench_path").set_value(str(path)).run()
            app.button(key="bench_read").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            app.slider(key="bench_shuffles").set_value(3).run()
            app.button(key="bench_compute").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertGreater(len(app.get("plotly_chart")), 0)
            app.radio(key="page").set_value("Kết quả tự chạy").run()
            app.radio(key="page").set_value("Demo tương tác").run()
            self.assertFalse(app.exception)
            self.assertGreater(len(app.get("plotly_chart")), 0)

    def test_navigation_does_not_train_and_original_five_modules_exist(self):
        from demo.live_ui import MODULES

        with patch.object(L, "run", side_effect=AssertionError("Unexpected fit")):
            app = AppTest.from_file(
                str(L.ROOT / "demo/app.py"), default_timeout=30
            ).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(MODULES), 5)
            for name in MODULES:
                app.radio(key="live_module").set_value(name).run()
                self.assertFalse(app.exception, name + str(app.exception))
            app.radio(key="page").set_value("Tiến độ theo paper").run()
            self.assertFalse(app.exception)
            displayed = app.dataframe[0].value["Ước lượng tự train (%)"]
            self.assertFalse(displayed.str.contains("Chưa định lượng").any())
            self.assertTrue(displayed.str.startswith(("≈ ", "— (")).all())

    def test_live_button_runs_once_and_keeps_result_when_switching_pages(self):
        with (
            patch("demo.live_ui.raw_data", return_value=unit_fixture()),
            patch.object(L, "compare", wraps=L.compare) as run,
        ):
            app = AppTest.from_file(
                str(L.ROOT / "demo/app.py"), default_timeout=30
            ).run()
            app.radio(key="live_module").set_value("Rotation").run()
            app.multiselect(key="live_models").set_value(["RandomForest"]).run()
            app.button(key="live_prepare").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(run.call_count, 0)
            app.button(key="live_run_rotation").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(run.call_count, 1)
            self.assertGreater(len(app.get("plotly_chart")), 0)
            plotted = app.session_state["_live_result_rotation"][1]["output"]
            self.assertEqual(plotted.variant.nunique(), 2)
            self.assertEqual(len(plotted), 4)
            self.assertEqual(len(app.multiselect(key="live_rotation_halves").value), 2)
            app.radio(key="page").set_value("Kết quả tự chạy").run()
            app.radio(key="page").set_value("Demo tương tác").run()
            self.assertFalse(app.exception)
            self.assertEqual(run.call_count, 1)
            self.assertGreater(len(app.get("plotly_chart")), 0)
            app.multiselect(key="live_rotation_halves").set_value([]).run()
            self.assertTrue(app.button(key="live_run_rotation").disabled)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
