"""Two-page, local-only evidence viewer. Opening or filtering never trains."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402 -- Streamlit script bootstrap above.

from demo import evidence as E  # noqa: E402 -- Streamlit script needs the repository root.
from demo import inventory as I  # noqa: E402
from demo.charts import benchmark_chart, confusion_chart, mechanism_chart  # noqa: E402

PANEL_NAMES = {
    "Numeric · classification": "numerical_classification_medium",
    "Numeric · regression": "numerical_regression_medium",
    "Categorical · classification": "categorical_classification_medium",
    "Categorical · regression": "categorical_regression_medium",
}
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


def show_removal():
    own = E.cells()
    own = own.loc[own.mechanism.eq("feature_removal") & own.full_repeats]
    a, b = st.columns(2)
    dataset = a.selectbox(
        "Dataset", sorted(own.dataset.unique()), key="removal_dataset"
    )
    kept = (
        b.radio(
            "Train trên",
            ["Feature giữ lại", "Feature bị loại"],
            horizontal=True,
            key="removal_part",
        )
        == "Feature giữ lại"
    )
    series = E.removal_series(dataset, kept)
    models = model_picker(series.model.unique(), "removal_models")
    series = series.loc[series.model.isin(models)]
    st.caption("Đại diện · Accuracy thô · Không có tỷ lệ hoàn thành full paper")
    if series.empty:
        st.info("Không có output cho lựa chọn này.")
        return
    st.plotly_chart(
        mechanism_chart(series, "feature_removal", False),
        width="stretch",
        key="removal_chart",
    )
    with st.expander("Số liệu"):
        st.dataframe(series, hide_index=True, width="stretch")
        download_table(series, "feature-removal.csv", "removal_export")
    subset = own.loc[
        own.dataset.eq(dataset) & own.parsed.map(lambda c: c[1] is not kept)
    ]
    subset = subset.loc[subset.model.map(E.NAMES).isin(models)]
    show_case(subset, "feature_removal", "removal")


def show_benchmark():
    a, b = st.columns([2, 1])
    panel_name = a.selectbox("Nhóm dữ liệu", list(PANEL_NAMES), key="benchmark_panel")
    source = b.radio(
        "Nguồn", ["Ours", "Paper"], horizontal=True, key="benchmark_source"
    )
    panel = PANEL_NAMES[panel_name]
    frame = E.benchmark_series(source, panel)
    chosen = model_picker(frame.model.unique(), "benchmark_models_" + source + panel)
    visible = frame.loc[frame.model.isin(chosen)]
    if visible.empty:
        st.info("Chọn ít nhất một model.")
        return
    n = int(frame.datasets.iloc[0])
    budget = int(frame["rank"].max())
    st.caption(
        f"{source} · {n} dataset · ngân sách {budget} · 15 hoán vị · default-first"
    )
    st.plotly_chart(benchmark_chart(visible), width="stretch", key="benchmark_chart")
    with st.expander("Số liệu"):
        last = visible.sort_values("rank").groupby("model", as_index=False).tail(1)
        st.dataframe(
            last[["model", "rank", "mean", "low", "high"]],
            hide_index=True,
            width="stretch",
        )
        download_table(
            visible, "benchmark-" + source.lower() + ".csv", "benchmark_export"
        )
    with st.expander("Nguồn & phạm vi"):
        st.write(
            "Hai nguồn có tập dataset, ngân sách và mốc chuẩn hóa riêng; không ghép thành phép so trực tiếp."
        )
        st.write(
            "Ours: bộ đối chứng cân bằng 24/57 thiết lập, gồm cả categorical và regression."
        )
        st.markdown(
            f"[Notebook]({E.link('notebooks/figure_01_benchmark.ipynb')}) · "
            f"[Đối chiếu Python–R]({E.link('evidence/python_r_parity.json')})"
        )


def experiments():
    st.header("Thực nghiệm")
    choice = st.radio(
        "Thực nghiệm",
        [
            "Rotation",
            "Noise features",
            "Target smoothing",
            "Benchmark",
            "Feature removal · đại diện",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="experiment",
    )
    if choice == "Benchmark":
        show_benchmark()
    elif choice.startswith("Feature removal"):
        show_removal()
    else:
        show_mechanism(E.EXPERIMENTS[choice])


def progress():
    st.header("Tiến độ & bằng chứng")
    a, b, c = st.columns(3)
    a.metric("Phân tích CSV tác giả", "27 / 40 hình")
    a.caption("67,5%")
    b.metric("Lưới cấu hình công bố", "3 / 3")
    b.caption("632 ô · 1.556 fit")
    c.metric("Thiết lập có output", "57 / 57")
    c.caption("Không đồng nghĩa đủ HPO")
    mode = st.radio(
        "Danh mục",
        ["Độ phủ thực nghiệm", "Hình & phụ lục"],
        horizontal=True,
        label_visibility="collapsed",
        key="progress_mode",
    )
    if mode == "Độ phủ thực nghiệm":
        table = I.coverage()
        st.caption("Snapshot đã audit · Mỗi hàng có mẫu số riêng · Training đã dừng")
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height=540,
            column_config={
                "Độ phủ (%)": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.2f%%"
                )
            },
        )
    else:
        table = I.paper_inventory()
        a, b = st.columns([2, 1])
        text = a.text_input(
            "Tìm Figure / Table",
            placeholder="Ví dụ: rotation, Figure 19, SAINT",
            key="inventory_search",
        )
        status = b.selectbox(
            "Phân tích tác giả", ["Tất cả", "Đã có", "Chưa có"], key="inventory_status"
        )
        if text:
            mask = (
                table.astype(str)
                .apply(
                    lambda column: column.str.contains(text, case=False, regex=False)
                )
                .any(axis=1)
            )
            table = table.loc[mask]
        if status == "Chưa có":
            table = table.loc[
                table["Phân tích tác giả"].str.startswith(("Chưa", "Thiếu"))
            ]
        elif status == "Đã có":
            table = table.loc[
                ~table["Phân tích tác giả"].str.startswith(("Chưa", "Thiếu"))
            ]
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            height="auto",
            column_config={
                "Bằng chứng": st.column_config.LinkColumn(display_text="Mở"),
                "Phân tích (%)": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%.0f%%"
                ),
            },
        )
    download_table(table, "paper-progress.csv", "progress_export")
    with st.expander("Cách đọc độ phủ"):
        st.write(
            "Phân tích tác giả, cấu hình đủ repeats và thực nghiệm đại diện là các đơn vị khác nhau."
        )
        st.write(
            "Phân tích (%) chỉ đo đã tính lại Figure từ CSV tác giả, không đo phần trăm tự training."
        )
        st.write(
            "100% ba lưới công bố không phải toàn bộ HPO hoặc full paper. Không có một % chung."
        )
        st.write(
            "Catalog tự chạy có 57 thiết lập; nhãn Figure1 tác giả có 59. Phạm vi được giữ riêng."
        )
        st.markdown(
            f"[Nguồn kiểm chứng]({E.link('reproduce/results/source_audits.json')}) · "
            f"[Phạm vi phụ lục]({E.link('evidence/appendix_sources.md')})"
        )


def main():
    st.set_page_config(page_title="Tabular Trees", layout="wide")
    st.html(
        "<style>.stMainBlockContainer{padding-top:1.25rem;padding-bottom:1rem}</style>"
    )
    saved = st.session_state.setdefault("_saved_controls", {})
    for key, value in saved.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.sidebar.title("Tabular Trees")
    page = st.sidebar.radio(
        "Điều hướng",
        ["Thực nghiệm", "Tiến độ & bằng chứng"],
        label_visibility="collapsed",
        key="page",
    )
    st.sidebar.caption("NeurIPS 2022 · Kết quả đã lưu")
    st.sidebar.caption("Snapshot 28/11 · Training đã dừng")
    st.sidebar.link_button(
        "PDF tổng hợp", E.link("reproduce/report/bao-cao-tai-hien.pdf")
    )
    try:
        E.verify_sources()
        if page.startswith("Tiến độ"):
            progress()
        else:
            experiments()
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        st.error(
            "Không thể đọc bộ bằng chứng. Kiểm tra file hoặc khôi phục từ repository."
        )
        with st.expander("Chi tiết lỗi"):
            st.code(str(exc))
    controls = {
        "experiment",
        "rotation_fraction",
        "removal_part",
        "progress_mode",
        "inventory_search",
        "inventory_status",
        "benchmark_panel",
        "benchmark_source",
    }
    for key in st.session_state:
        if (
            key in controls
            or key.endswith(
                ("_dataset", "_metric", "_models", "_case_condition", "_case_model")
            )
            or key.startswith("benchmark_models_")
        ):
            saved[key] = st.session_state[key]


if __name__ == "__main__":
    main()
