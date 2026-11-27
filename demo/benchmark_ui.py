"""Preserved benchmark CSV/analysis controls; incomplete self-run curves stay secondary."""

import streamlit as st

from demo import benchmark as B, evidence as E
from demo.charts import benchmark_chart

PANELS = {
    "Numeric · classification": "numerical_classification_medium",
    "Numeric · regression": "numerical_regression_medium",
    "Categorical · classification": "categorical_classification_medium",
    "Categorical · regression": "categorical_regression_medium",
}


def display(frame, key):
    if frame.empty:
        st.info("Không có dữ liệu cho lựa chọn này.")
        return
    options = frame.model.unique().tolist()
    if key + "_models" in st.session_state:
        st.session_state[key + "_models"] = [
            v for v in st.session_state[key + "_models"] if v in options
        ]
    models = st.multiselect(
        "Model",
        frame.model.unique().tolist(),
        default=frame.model.unique().tolist(),
        key=key + "_models",
    )
    visible = frame.loc[frame.model.isin(models)]
    if visible.empty:
        st.info("Chọn ít nhất một model.")
        return
    st.plotly_chart(
        benchmark_chart(visible, log_x=True), width="stretch", key=key + "_chart"
    )
    with st.expander("Số liệu"):
        st.dataframe(visible, hide_index=True, width="stretch")
        st.download_button(
            "CSV đường HPO",
            visible.to_csv(index=False).encode("utf-8-sig"),
            "benchmark-curves.csv",
            "text/csv",
            key=key + "_download",
        )


def render():
    source = st.radio(
        "Nguồn kết quả",
        ["Paper · đã kiểm chứng", "CSV của bạn"],
        horizontal=True,
        key="bench_source",
    )
    panel_name = st.selectbox("Nhóm dữ liệu", list(PANELS), key="bench_panel")
    if source.startswith("Paper"):
        frame = E.benchmark_series("Paper", PANELS[panel_name])
        budget = st.slider(
            "Ngân sách hiển thị", 10, 400, 400, 10, key="bench_paper_budget"
        )
        st.caption(
            f"CSV tác giả · {int(frame.datasets.iloc[0])} dataset · 15 hoán vị · chọn bằng validation"
        )
        display(frame.loc[frame["rank"] <= budget], "bench_paper")
    with st.expander("Đọc CSV / tính lại HPO", expanded=source == "CSV của bạn"):
        path = st.text_input(
            "Đường dẫn CSV / CSV.GZ",
            str(E.ROOT / ".cache/benchmark.csv.gz"),
            key="bench_path",
        )
        uploaded = st.file_uploader(
            "Hoặc upload CSV kết quả", type=["csv"], key="bench_upload"
        )
        if st.button("Đọc CSV", key="bench_read"):
            try:
                content = uploaded.getvalue() if uploaded is not None else path
                with st.spinner("Đang đọc CSV…"):
                    frame = B.read_csv(content)
                st.session_state["_bench_input"] = frame
                st.session_state.pop("_bench_result", None)
            except (ValueError, OSError, UnicodeError):
                st.error(
                    "Không đọc được CSV. Cần model_name, data__keyword, mean_val_score và mean_test_score."
                )
        imported = st.session_state.get("_bench_input")
        if imported is not None:
            panels = (
                sorted(imported.benchmark.dropna().unique())
                if "benchmark" in imported
                else ["custom"]
            )
            panel = st.selectbox("Nhóm trong CSV", panels, key="bench_csv_panel")
            if "benchmark" not in imported:
                task = st.radio(
                    "Loại bài toán CSV",
                    ["Classification", "Regression"],
                    horizontal=True,
                    key="bench_custom_task",
                )
                panel = "custom_" + task.lower()
            a, b = st.columns(2)
            shuffles = a.slider("Số lần xáo", 3, 30, 15, key="bench_shuffles")
            iterations = b.slider(
                "Iteration tối đa", 10, 400, 400, 10, key="bench_iterations"
            )
            identity = (panel, shuffles, iterations)
            if st.button("Tính lại đường HPO", type="primary", key="bench_compute"):
                try:
                    from demo.live import ensure_resources

                    ensure_resources()
                    with st.spinner("Đang chọn bằng validation và tổng hợp…"):
                        result, has_defaults = B.curves(
                            imported, panel, shuffles, iterations
                        )
                    st.session_state["_bench_result"] = (identity, result, has_defaults)
                except ValueError as exc:
                    st.error(str(exc))
            stored = st.session_state.get("_bench_result")
            if stored and stored[0] == identity:
                if not stored[2]:
                    st.info(
                        "CSV không có default; đường cong bắt đầu từ các cấu hình random."
                    )
                display(stored[1], "bench_imported")
        st.caption(
            "Tính lại phân tích CSV, không train. Dùng engine Python đã đối chiếu R; giữ default-first và validation-only."
        )
    with st.expander("Kết quả benchmark tự chạy · phạm vi hạn chế"):
        st.caption(
            "24/57 thiết lập trong bộ đối chứng cân bằng; chưa đủ để đại diện toàn benchmark paper."
        )
        frame = E.benchmark_series("Ours", PANELS[panel_name])
        st.dataframe(
            frame.groupby("model").tail(1)[["model", "rank", "mean", "low", "high"]],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "CSV bằng chứng tự chạy",
            frame.to_csv(index=False).encode("utf-8-sig"),
            "benchmark-ours-partial.csv",
            "text/csv",
            key="bench_own_download",
        )
