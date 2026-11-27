"""Five original demo modules, evaluated only on an explicit run action."""

import json

import plotly.graph_objects as go
import streamlit as st

from demo import live as L
from demo.evidence import COLORS

MODULES = [
    "Benchmark",
    "Decision boundary",
    "Target smoothing",
    "Noise / removal",
    "Rotation",
]


@st.cache_data(show_spinner=False)
def cached_raw(name, modified):
    return L.load_dataset(name)


def raw_data(name):
    path = L.D.local_path(name)
    return cached_raw(name, path.stat().st_mtime_ns) if path.exists() else None


def configuration():
    with st.sidebar.expander("Dữ liệu demo", expanded=True):
        name = st.selectbox(
            "Dataset",
            L.dataset_options(),
            format_func=L.dataset_label,
            key="live_dataset",
        )
        raw = raw_data(name)
        if raw is None:
            if name in L.D.LARGE_DATASETS:
                st.caption("Dataset lớn · lần tải đầu có thể lâu.")
            if st.button("Tải dataset", key="live_load"):
                try:
                    with st.spinner("Đang tải dữ liệu công khai…"):
                        L.load_dataset(name)
                    st.rerun()
                except Exception:
                    st.error(
                        "Không tải được dataset. Kiểm tra mạng hoặc chọn dataset nhỏ hơn."
                    )
            st.info("Tải dataset để chọn target và chuẩn bị dữ liệu.")
            return None
        columns = list(raw.columns)
        target = st.selectbox(
            "Target",
            columns,
            index=columns.index(L.D.guess_target(columns)),
            key="live_target_" + name,
        )
        a, b = st.columns(2)
        n_train = a.number_input(
            "Train", 1000, 10000, 1500, step=500, key="live_n_train"
        )
        n_test = b.number_input("Test", 500, 5000, 1000, step=500, key="live_n_test")
        seed = int(st.number_input("Seed", 0, 10000, 0, key="live_seed"))
        models = st.multiselect(
            "Model",
            L.M.MODEL_NAMES,
            default=["GradientBoostingTree", "RandomForest"],
            key="live_models",
            format_func=lambda name: {
                "GradientBoostingTree": "GBT",
                "RandomForest": "RF",
                "MLP": "MLP",
            }[name],
        )
    signature = (name, target, n_train, n_test, seed)
    if st.button("Chuẩn bị dữ liệu", type="primary", key="live_prepare"):
        try:
            with st.spinner("Đang chia tập và Gaussianize feature…"):
                prepared = L.prepare(raw, target, n_train, n_test, seed)
            st.session_state["_live_prepared"] = (signature, prepared)
        except ValueError as exc:
            st.error(str(exc))
    stored = st.session_state.get("_live_prepared")
    if stored is None or stored[0] != signature:
        return None
    data = stored[1]
    st.caption(
        f"{len(data.y_train):,} train · {len(data.y_test):,} test · {data.X_train.shape[1]} feature"
    )
    return signature, data, models, seed


def perform(kind, signature, data, models, seeds, settings):
    identity = json.dumps([signature, models, seeds, settings], sort_keys=True)
    key = "_live_result_" + kind
    if st.button(
        "Chạy demo",
        type="primary",
        key="live_run_" + kind,
        disabled=not models
        or not settings.get("levels", [True])
        or not settings.get("remove_halves", [True])
        or not settings.get("covariance_modes", [True]),
        # Input widgets do not launch work until this action is requested.
    ):
        counter = st.empty()
        try:
            with st.spinner("Đang chạy trên CPU local…"):
                runner = L.run if kind == "boundary" else L.compare
                result = runner(
                    kind,
                    data,
                    models,
                    seeds,
                    settings=settings,
                    on_fit=lambda count, name: counter.caption(
                        f"Đã hoàn tất {count} fit · {name}"
                    ),
                )
            st.session_state[key] = (identity, result)
        except (ValueError, RuntimeError, MemoryError) as exc:
            st.session_state.pop(key, None)
            st.error(str(exc))
        finally:
            counter.empty()
    stored = st.session_state.get(key)
    if stored and stored[0] == identity:
        for notice in stored[1]["warnings"]:
            st.warning(notice)
        return stored[1]
    st.info("Chọn tham số rồi bấm Chạy demo.")
    return None


def results(frame, kind, by, x_title, signature, settings, fits):
    if frame.empty or "model" not in frame:
        st.info("Không có fit cho lựa chọn này. Nhánh bị loại cần tỷ lệ lớn hơn 0.")
        return
    variants = "variant" in frame
    summary = L.X.summarize(frame, ["model", *(["variant"] if variants else []), by])
    figure = go.Figure()
    styles = ["solid", "dash", "dot", "dashdot"]
    variant_names = list(frame.variant.unique()) if variants else [""]
    groups = summary.groupby(
        ["model", "variant"] if variants else ["model"], sort=False
    )
    for identity, group in groups:
        model, variant = identity if variants else (identity[0], "")
        title = model + (" · " + variant if variant else "")
        figure.add_trace(
            go.Scatter(
                x=group[by],
                y=group.mean_acc,
                mode="lines+markers",
                name=title,
                line=dict(
                    color=COLORS.get(model.split(" · ")[0], "#777777"),
                    dash=styles[variant_names.index(variant) % len(styles)],
                ),
                error_y=dict(
                    type="data",
                    array=group.std_acc,
                    visible=bool((group.std_acc > 0).any()),
                ),
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=430,
        xaxis_title=x_title,
        yaxis_title="Test accuracy",
        yaxis_tickformat=".1%",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=20, b=65),
    )
    st.plotly_chart(figure, width="stretch", key="live_plot_" + kind)
    with st.expander("Số liệu & cấu hình"):
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            "CSV kết quả",
            frame.to_csv(index=False).encode("utf-8-sig"),
            kind + "-local.csv",
            "text/csv",
            key="live_csv_" + kind,
        )
        st.download_button(
            "JSON cấu hình",
            json.dumps(
                {
                    "data": signature,
                    "models": sorted(fits.model.unique().tolist()),
                    "seeds": sorted(fits.seed.unique().tolist()),
                    "settings": settings,
                    "upstream_commit": "e9fce1eba2fb498b9571b33a331832a3eef63626",
                    "purpose": "local illustration; excluded from reproduction progress",
                },
                ensure_ascii=False,
                indent=2,
            ),
            kind + "-local.json",
            "application/json",
            key="live_json_" + kind,
        )
        st.dataframe(fits, hide_index=True, width="stretch")


def boundary(signature, data, seed):
    names = data.feature_names
    for key, fallback in [("live_f1", 0), ("live_f2", 1)]:
        if key in st.session_state and st.session_state[key] >= len(names):
            st.session_state[key] = fallback
    if st.button("Chọn 2 feature quan trọng nhất", key="live_top_features"):
        with st.spinner("Đang tính RF importance…"):
            order = L.importance(data, seed)
        st.session_state["live_f1"] = int(order[-1])
        st.session_state["live_f2"] = int(order[-2])
    a, b = st.columns(2)
    f1 = a.selectbox(
        "Feature trục X",
        range(len(names)),
        format_func=lambda i: names[i],
        key="live_f1",
    )
    f2 = b.selectbox(
        "Feature trục Y",
        range(len(names)),
        index=1,
        format_func=lambda i: names[i],
        key="live_f2",
    )
    a, b = st.columns(2)
    xlim = a.slider("Vùng nhìn X", -4.0, 4.0, (-2.5, 2.5), 0.1, key="live_xlim")
    ylim = b.slider("Vùng nhìn Y", -4.0, 4.0, (-2.5, 2.5), 0.1, key="live_ylim")
    if f1 == f2:
        st.info("Chọn hai feature khác nhau.")
        return
    if xlim[0] == xlim[1] or ylim[0] == ylim[1]:
        st.info("Vùng nhìn phải có độ rộng lớn hơn 0.")
        return
    settings = {"features": [f1, f2]}
    result = perform(
        "boundary", signature, data, ["RandomForest", "MLP"], [seed], settings
    )
    if result:
        st.plotly_chart(
            L.boundary_plot(data, result["output"], (f1, f2), xlim, ylim),
            width="stretch",
        )
        st.caption(
            "Minh họa 2D · RF và MLP của demo gốc · Không phải tái hiện Figure 19."
        )
        with st.expander("Số liệu"):
            st.dataframe(result["fits"], hide_index=True, width="stretch")
            st.download_button(
                "CSV kết quả",
                result["fits"].to_csv(index=False).encode("utf-8-sig"),
                "decision-boundary-local.csv",
                "text/csv",
                key="live_boundary_csv",
            )


def render():
    st.header("Demo tương tác")
    st.caption(
        "Demo gốc · Chạy dữ liệu thật trên CPU · Kết quả minh họa không cộng vào tiến độ paper"
    )
    choice = st.radio(
        "Chức năng",
        MODULES,
        horizontal=True,
        label_visibility="collapsed",
        key="live_module",
    )
    if choice == "Benchmark":
        from demo.benchmark_ui import render as show_benchmark

        show_benchmark()
        return
    selected = configuration()
    if selected is None:
        st.info("Chọn tham số ở sidebar rồi bấm Chuẩn bị dữ liệu.")
        return
    signature, data, models, seed = selected
    if choice == "Decision boundary":
        boundary(signature, data, seed)
        return
    repeats = st.slider("Số seed", 1, 5, 1, key="live_repeats_" + choice)
    seeds = list(range(seed, seed + repeats))
    if not models:
        st.info("Chọn ít nhất một model trong Dữ liệu demo.")
        return
    if choice == "Target smoothing":
        a, b = st.columns([2, 1])
        levels = a.multiselect(
            "Hệ số covariance",
            [0.0, 0.01, 0.05, 0.1, 0.25, 0.5],
            default=[0.0, 0.1],
            key="live_smoothing_levels",
        )
        max_features = min(10, data.X_train.shape[1])
        if "live_smoothing_k" in st.session_state:
            st.session_state["live_smoothing_k"] = min(
                max_features, st.session_state["live_smoothing_k"]
            )
        k = b.number_input(
            "Feature quan trọng giữ lại",
            2,
            max_features,
            min(5, max_features),
            key="live_smoothing_k",
        )
        modes = st.multiselect(
            "Covariance · chọn nhiều để so sánh",
            [False, True],
            default=[False],
            format_func=lambda value: "Robust · MinCovDet" if value else "Thường",
            key="live_covariance_modes",
        )
        settings = {"levels": sorted(levels), "k": int(k), "covariance_modes": modes}
        result = perform("smoothing", signature, data, models, seeds, settings)
        if result:
            frame, changes = result["output"], result["label_changes"]
            results(
                frame,
                "smoothing",
                "lengthscale2",
                "Hệ số covariance",
                signature,
                settings,
                result["fits"],
            )
            st.caption("Nhãn train thay đổi; nhãn test giữ nguyên.")
            st.dataframe(changes, hide_index=True, width="stretch")
    elif choice == "Noise / removal":
        mode = st.radio(
            "Biến đổi",
            ["Thêm feature nhiễu", "Bỏ feature ít quan trọng"],
            horizontal=True,
            key="live_feature_mode",
        )
        kind = "add" if mode.startswith("Thêm") else "remove"
        options = (
            [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
            if kind == "add"
            else [i / 10 for i in range(10)]
        )
        levels = st.multiselect(
            "Tỷ lệ feature",
            options,
            default=[0.0, 1.0] if kind == "add" else [0.0, 0.5],
            key="live_levels_" + kind,
        )
        settings = {"levels": sorted(levels)}
        if kind == "remove":
            parts = st.multiselect(
                "Nhánh train · chọn nhiều để so sánh",
                ["kept", "removed"],
                default=["kept", "removed"],
                format_func=lambda part: "Feature giữ lại"
                if part == "kept"
                else "Feature đã bỏ · GBT",
                key="live_removal_parts",
            )
            if not parts:
                st.info("Chọn ít nhất một nhánh train.")
                return
            settings["train_parts"] = parts
            st.caption("Nhánh feature bị loại dùng GBT như demo gốc.")
        result = perform(kind, signature, data, models, seeds, settings)
        if result:
            frame = result["output"].copy()
            if "trained_on" in frame:
                mask = frame.trained_on.eq("removed")
                frame["variant"] = "Feature giữ lại"
                frame.loc[mask, "variant"] = "Feature đã bỏ · GBT"
            frame["fraction"] *= 100
            results(
                frame,
                kind,
                "fraction",
                "Feature thêm / bỏ (%)",
                signature,
                settings,
                result["fits"],
            )
    else:
        halves = st.multiselect(
            "Feature trước khi xoay · chọn nhiều để so sánh",
            [False, True],
            default=[False, True],
            format_func=lambda value: "Bỏ 50% feature ít quan trọng"
            if value
            else "Giữ toàn bộ feature",
            key="live_rotation_halves",
        )
        settings = {"remove_halves": halves}
        result = perform("rotation", signature, data, models, seeds, settings)
        if result:
            results(
                result["output"],
                "rotation",
                "condition",
                "Phép xoay",
                signature,
                settings,
                result["fits"],
            )
    with st.expander("Phạm vi demo"):
        st.write(
            "RF/GBT/MLP scikit-learn của demo gốc; MLP 4×256, không dropout. Không HPO. "
            "Một dataset, mẫu nhỏ, CPU 1 luồng; không thay thế các kết quả tự train V2."
        )
        st.markdown(
            "[Source demo gốc](https://github.com/LoylP/tabular-trees/tree/e9fce1eba2fb498b9571b33a331832a3eef63626)"
        )
