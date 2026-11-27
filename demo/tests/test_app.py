import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


class AppFlows(unittest.TestCase):
    def open_app(self):
        app = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=30).run()
        self.assertFalse(app.exception, str(app.exception))
        self.assertFalse(app.error, str(app.error))
        return app

    def test_rotation_filters_raw_scores_and_empty_models(self):
        app = self.open_app()
        app.radio(key="rotation_fraction").set_value("Loại 50%").run()
        app.selectbox(key="rotation_dataset").set_value("electricity").run()
        self.assertFalse(app.error)
        app.selectbox(key="rotation_metric").set_value("Accuracy · Ours").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        app.multiselect(key="rotation_models").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Chọn ít nhất" in item.value for item in app.info))

    def test_all_experiments_and_their_saved_details(self):
        app = self.open_app()
        for name in [
            "Noise features",
            "Target smoothing",
            "Feature removal · đại diện",
            "Benchmark",
        ]:
            app.radio(key="experiment").set_value(name).run()
            self.assertFalse(app.exception, name + str(app.exception))
            self.assertFalse(app.error, name + str(app.error))
        app.radio(key="benchmark_source").set_value("Paper").run()
        app.selectbox(key="benchmark_panel").set_value("Categorical · regression").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

    def test_progress_inventory_search_and_navigation_state(self):
        app = self.open_app()
        app.selectbox(key="rotation_dataset").set_value("electricity").run()
        app.radio(key="page").set_value("Tiến độ & bằng chứng").run()
        self.assertFalse(app.error)
        app.radio(key="progress_mode").set_value("Hình & phụ lục").run()
        app.text_input(key="inventory_search").set_value("Figure 19").run()
        self.assertFalse(app.exception)
        table = app.dataframe[0].value
        self.assertEqual(table.ID.tolist(), ["Figure 19"])
        app.selectbox(key="inventory_status").set_value("Chưa có").run()
        self.assertFalse(app.error)
        app.radio(key="page").set_value("Thực nghiệm").run()
        self.assertEqual(app.selectbox(key="rotation_dataset").value, "electricity")

    def test_bad_evidence_shows_error_without_presenting_scores(self):
        with patch(
            "demo.evidence.verify_sources",
            side_effect=ValueError("Nguồn dữ liệu không khớp"),
        ):
            app = AppTest.from_file(str(ROOT / "demo/app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertEqual(len(app.get("plotly_chart")), 0)

    def test_sample_confusion_and_csv_downloads_have_real_values(self):
        from demo import evidence as E
        import json
        import pandas as pd
        from io import StringIO

        app = self.open_app()
        key = (
            E.cells()
            .loc[
                E.cells().mechanism.eq("rotation")
                & E.cells().dataset.eq("electricity")
                & E.cells().model.eq("rf")
                & E.cells().parsed.map(lambda c: c == [0.0, False]),
                "cell_id",
            ]
            .iloc[0]
        )
        app.selectbox(key="rotation_case_condition").set_value(key).run()
        self.assertFalse(app.error)
        self.assertTrue(
            any("Confusion matrix" in caption.value for caption in app.caption)
        )
        # The download serialization round-trip must preserve scientific keys and values.
        expected = E.mechanism_series("rotation", "electricity")
        restored = pd.read_csv(
            StringIO(expected.to_csv(index=False)), float_precision="round_trip"
        )
        pd.testing.assert_frame_equal(restored, expected, check_dtype=False)
        config = E.configurations()[key]
        self.assertEqual(json.loads(json.dumps(config)), config)
        self.assertGreaterEqual(len(app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
