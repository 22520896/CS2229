"""Original interactive demo plus audited results and paper-level scope."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402 -- Streamlit script bootstrap above.

from demo import evidence as E  # noqa: E402 -- Streamlit script needs the repository root.
from demo.charts import confusion_chart, mechanism_chart  # noqa: E402

PRETTY = dict(zip(E.NAMES.values(), E.DISPLAY.values()))


def download_table(frame, filename, key):
    st.download_button(
        "Xuất CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        filename,
        "text/csv",
        key=key,
    )


def model_picker(models, key):
    return st.multiselect(
        "Model",
        list(models),
        default=list(models),
        format_func=lambda m: PRETTY.get(m, m),
        key=key,
    )


def show_case(frame, mechanism, key):
    with st.expander("Chi tiết case"):
        c1, c2, c3 = st.columns(3)
        names = sorted(frame.dataset.unique())
        dataset = c1.selectbox(
            "Dataset",
            names,
            index=names.index("electricity") if "electricity" in names else 0,
            key=key + "_case_dataset",
        )
        available = [
            m
            for m in E.MODELS
            if m in set(frame.loc[frame.dataset.eq(dataset), "model"])
        ]
        model = c2.selectbox(
            "Model",
            available,
            format_func=lambda m: E.DISPLAY[m],
            key=key + "_case_model",
        )
        candidates = frame.loc[
            frame.dataset.eq(dataset) & frame.model.eq(model)
        ].sort_values("condition")
        identifier = c3.selectbox(
            "Điều kiện",
            candidates.cell_id.tolist(),
            format_func=lambda v: _condition(
                mechanism, candidates.set_index("cell_id").loc[v, "parsed"]
            ),
            key=key + "_case_condition",
        )
        row = candidates.set_index("cell_id").loc[identifier]
        a, b, c = st.columns(3)
        a.metric("Validation", f"{row.validation_accuracy_mean:.2%}")
        b.metric("Test", f"{row.test_accuracy_mean:.2%}")
        c.metric("Seeds", ", ".join(map(str, json.loads(row.successful_seeds))))
        config = E.configurations()[identifier]
        st.json(config["configuration"], expanded=False)
        st.download_button(
            "Cấu hình JSON",
            json.dumps(config, indent=2, ensure_ascii=False).encode(),
            "case-configuration.json",
            "application/json",
            key=key + "_case_download",
        )
        sample = E.sample_numeric(mechanism, dataset, model, row.parsed)
        if sample is not None:
            meta, matrix = sample
            st.caption(f"Confusion matrix · test · seed {meta['seed']}")
            st.plotly_chart(
                confusion_chart(matrix), width="stretch", key=key + "_confusion"
            )
        st.markdown(
            f"[Nguồn kết quả]({E.link('reproduce/results/mechanism_cells.csv')})"
        )


def _condition(mechanism, condition):
    if mechanism == "rotation":
        return ("Không xoay" if condition[1] else "Có xoay") + (
            " · loại 50%" if condition[0] else ""
        )
    if mechanism == "noise_features":
        return f"Thêm {(condition[0] - 1) * 100:g}% nhiễu"
    if mechanism == "feature_removal":
        return f"Loại {condition[0] * 100:g}% · " + (
            "train phần bị loại" if condition[1] else "train phần giữ lại"
        )
    return f"Covariance {condition[0]:g} · {condition[1]}"


def show_mechanism(mechanism):
    fraction = 0.0
    if mechanism == "rotation":
        fraction = (
            0.5
            if st.radio(
                "Feature trước xoay",
                ["Giữ toàn bộ", "Loại 50%"],
                horizontal=True,
                key="rotation_fraction",
            )
            == "Loại 50%"
            else 0.0
        )
    base = E.reference_cells(mechanism, fraction)
    c1, c2, c3 = st.columns([1.2, 1.2, 1.7])
    with c1:
        scope = st.selectbox(
            "Dataset",
            ["Tổng hợp", *sorted(base.dataset.unique())],
            key=mechanism + "_dataset",
        )
    with c2:
        metric = st.selectbox(
            "Điểm",
            ["Chuẩn hóa · Paper/Ours", "Accuracy · Ours"],
            key=mechanism + "_metric",
        )
    with c3:
        models = model_picker(E.NAMES.values(), mechanism + "_models")
    normalized = metric.startswith("Chuẩn")
    dataset = None if scope == "Tổng hợp" else scope
    # Calculate using frozen author anchors and complete cohorts before display filters.
    series = E.mechanism_series(mechanism, dataset, normalized, fraction)
    series = series.loc[series.model.isin(models)]
    own = base if dataset is None else base.loc[base.dataset.eq(dataset)]
    shown = own.loc[own.model.map(E.NAMES).isin(models)]
    fits = sum(len(json.loads(v)) for v in shown.successful_seeds)
    st.caption(
        f"{shown.dataset.nunique()} dataset · {len(shown)} ô điều kiện · {fits} fit · Cấu hình công bố"
    )
    if series.empty:
        st.info("Chọn ít nhất một model.")
        return
    st.plotly_chart(
        mechanism_chart(series, mechanism, normalized),
        width="stretch",
        key=mechanism + "_chart",
    )
    with st.expander("Số liệu"):
        st.dataframe(
            E.change_table(series),
            hide_index=True,
            width="stretch",
            column_config={
                name: st.column_config.NumberColumn(format="%.4f")
                for name in ["Đầu", "Cuối", "Δ"]
            },
        )
        download_table(series, mechanism + ".csv", mechanism + "_export")
    show_case(own.loc[own.model.map(E.NAMES).isin(models)], mechanism, mechanism)
    with st.expander("Nguồn & phạm vi"):
        st.write(
            "Paper: trung bình HPO/shuffles. Ours: một cấu hình công bố mỗi điều kiện, đủ seed."
        )
        if normalized:
            st.write(
                "Cùng dataset và mốc chuẩn hóa tác giả; không clip điểm, không chọn model theo test."
            )
        notebook = {
            "rotation": "figure_05_rotation_self_trained.ipynb",
            "noise_features": "figure_04_noise_features_self_trained.ipynb",
            "target_smoothing": "figure_02_target_smoothing_self_trained.ipynb",
        }[mechanism]
        st.markdown(
            f"[Notebook]({E.link('reproduce/notebooks/' + notebook)}) · "
            f"[Protocol]({E.link('reproduce/configs/protocol.json')})"
        )


def experiments():
    st.header("Kết quả tự chạy")
    choice = st.radio(
        "Thực nghiệm",
        [
            "Rotation",
            "Noise features",
            "Target smoothing",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="experiment",
    )
    show_mechanism(E.EXPERIMENTS[choice])


def is_control(key):
    names = {
        "experiment",
        "rotation_fraction",
        "removal_part",
        "progress_mode",
        "inventory_search",
        "inventory_status",
        "scope_detail",
        "live_module",
        "live_dataset",
        "live_n_train",
        "live_n_test",
        "live_seed",
        "live_models",
        "live_f1",
        "live_f2",
        "live_xlim",
        "live_ylim",
        "live_smoothing_levels",
        "live_smoothing_k",
        "live_smoothing_robust",
        "live_feature_mode",
        "live_on_removed",
        "live_remove_half",
        "bench_source",
        "bench_panel",
        "bench_path",
        "bench_csv_panel",
        "bench_shuffles",
        "bench_iterations",
        "bench_custom_task",
        "bench_paper_budget",
    }
    return (
        key in names
        or key.startswith(("live_target_", "live_repeats_", "live_levels_"))
        or key.endswith(
            ("_dataset", "_metric", "_models", "_case_condition", "_case_model")
        )
    )


def main():
    st.set_page_config(page_title="Tabular Trees", layout="wide")
    st.html(
        "<style>.stMainBlockContainer{padding-top:1.25rem;padding-bottom:1rem}</style>"
    )
    saved = st.session_state.setdefault("_saved_controls", {})
    for key, value in list(saved.items()):
        if not is_control(key):
            saved.pop(key)
        elif key not in st.session_state:
            st.session_state[key] = value
    st.sidebar.title("Tabular Trees")
    page = st.sidebar.radio(
        "Điều hướng",
        ["Demo tương tác", "Kết quả tự chạy", "Tiến độ theo paper"],
        label_visibility="collapsed",
        key="page",
    )
    st.sidebar.caption("NeurIPS 2022 · Demo & bằng chứng")
    st.sidebar.link_button(
        "PDF tổng hợp", E.link("reproduce/report/bao-cao-tai-hien.pdf")
    )
    try:
        if page == "Demo tương tác":
            from demo import live, live_ui

            live.verify_upstream()
            live_ui.render()
        else:
            E.verify_sources()
            if page.startswith("Tiến độ"):
                from demo.progress_ui import render

                render()
            else:
                experiments()
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(
            "Không thể đọc bộ bằng chứng. Kiểm tra file hoặc khôi phục từ repository."
        )
        with st.expander("Chi tiết lỗi"):
            st.code(str(exc))
    for key in st.session_state:
        if is_control(key):
            saved[key] = st.session_state[key]


if __name__ == "__main__":
    main()
