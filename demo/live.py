"""Resource-bounded adapter around the pinned, unchanged upstream scientific core."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import shutil
import threading
from urllib.request import urlopen
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from demo.upstream.core import data as D, experiments as X, models as M, transforms as T

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/demo-live"
D.DATA_DIR = CACHE / "clf_num"
LOCK = threading.Lock()
MAX_DOWNLOAD = 256 * 1024**2
MIN_RAM = 768 * 1024**2
PREFERRED_DATASETS = ("electricity", "MagicTelescope", "bank-marketing")


def dataset_options():
    order = [
        *PREFERRED_DATASETS,
        *[name for name in D.CLF_NUM_DATASETS if name not in PREFERRED_DATASETS],
    ]
    available = [name for name in order if D.local_path(name).is_file()]
    return available + [name for name in order if name not in available]


def dataset_label(name):
    return (
        f"🟢 {name} · Có sẵn"
        if D.local_path(name).is_file()
        else f"↓ {name} · Chưa tải"
    )


def verify_upstream():
    directory = Path(__file__).resolve().parent / "upstream"
    source = json.loads((directory / "source.json").read_bytes())
    for entry in source["files"]:
        if entry["path"].startswith("core/") or entry["path"] in {
            "app.py",
            "README.md",
            "requirements.txt",
        }:
            if (
                sha256((directory / entry["path"]).read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise ValueError("Source demo gốc bị thay đổi: " + entry["path"])
    return source


def ensure_resources():
    if psutil.virtual_memory().available < MIN_RAM:
        raise ValueError(
            "Cần ít nhất 768 MiB RAM trống để chạy demo. Đóng bớt ứng dụng rồi thử lại."
        )


def load_dataset(name):
    if name not in D.CLF_NUM_DATASETS:
        raise ValueError("Dataset không nằm trong danh mục demo gốc.")
    path = D.local_path(name)
    if not path.exists():
        ensure_resources()
        path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path.parent).free < MAX_DOWNLOAD * 2:
            raise ValueError("Cần thêm dung lượng ổ đĩa trước khi tải dataset.")
        temporary = path.with_suffix(".csv.part")
        try:
            with (
                urlopen(D.hf_url(name), timeout=60) as response,
                temporary.open("wb") as output,
            ):
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD:
                        raise ValueError(
                            "Dataset vượt 256 MiB; chọn dataset nhỏ hơn cho demo local."
                        )
                    output.write(chunk)
            # Validate before making the cache visible; no partial CSV is read by the app.
            pd.read_csv(temporary, nrows=5)
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return D.load_raw(name)


def prepare(raw, target, n_train, n_test, seed):
    ensure_resources()
    if not 100 <= n_train <= 10000 or not 50 <= n_test <= 5000:
        raise ValueError("Số mẫu ngoài phạm vi demo local.")
    return D.prepare(raw, target, n_train, n_test, seed)


def _forest(*args, **kwargs):
    kwargs["n_jobs"] = 1
    return RandomForestClassifier(*args, **kwargs)


@contextmanager
def bounded():
    """Only execution resources change: estimator parameters and formulas stay upstream."""
    ensure_resources()
    if not LOCK.acquire(blocking=False):
        raise ValueError("Một demo khác đang chạy. Chờ hoàn tất rồi thử lại.")
    try:
        with (
            threadpool_limits(limits=1),
            patch.object(M, "RandomForestClassifier", _forest),
            patch.object(T, "RandomForestClassifier", _forest),
        ):
            yield
    finally:
        LOCK.release()


def importance(data, seed):
    with bounded():
        return T.rf_importance_order(data.X_train, data.y_train, seed)


def run(kind, data, models, seeds, *, settings=None, on_fit=None):
    """Calls upstream experiments; returns real outputs without changing stored evidence."""
    verify_upstream()
    settings = dict(settings or {})
    if kind not in {"boundary", "smoothing", "add", "remove", "rotation"}:
        raise ValueError("Thí nghiệm không hợp lệ.")
    if not models or not set(models).issubset(M.MODEL_NAMES):
        raise ValueError("Chọn ít nhất một model hợp lệ.")
    if not seeds or len(seeds) > 5:
        raise ValueError("Chọn từ 1 đến 5 seed.")
    fits, notices = [], []
    original = M.fit_score

    def recording(name, a, y, b, yt, seed):
        result = original(name, a, y, b, yt, seed)
        estimator = result["model"]
        if hasattr(estimator, "n_iter_") and estimator.n_iter_ >= estimator.max_iter:
            notices.append(
                f"{name} · seed {seed}: đạt trần {estimator.max_iter} epoch, chưa xác nhận hội tụ."
            )
        fits.append(
            {
                "model": name,
                "seed": seed,
                **{k: result[k] for k in ("train_acc", "test_acc", "seconds")},
            }
        )
        if on_fit:
            on_fit(len(fits), name)
        return result

    with bounded(), patch.object(X, "fit_score", recording):
        if kind == "boundary":
            f1, f2 = settings["features"]
            if f1 == f2 or min(f1, f2) < 0 or max(f1, f2) >= data.X_train.shape[1]:
                raise ValueError("Chọn hai feature khác nhau và hợp lệ.")
            output = X.fit_two_feature_models(data, f1, f2, seeds[0])
        elif kind == "smoothing":
            output = X.run_smoothing(
                data,
                settings["levels"],
                settings["k"],
                settings["robust"],
                models,
                seeds,
            )
        elif kind == "add":
            output = X.run_add_features(data, settings["levels"], models, seeds)
        elif kind == "remove":
            parts = settings.get("train_parts")
            selected_models = models if parts is None or "kept" in parts else []
            on_removed = settings["on_removed"] if parts is None else "removed" in parts
            output = X.run_remove_features(
                data, settings["levels"], selected_models, seeds, on_removed
            )
        else:
            output = X.run_rotation(data, settings["remove_half"], models, seeds)
    return {
        "output": output,
        "fits": pd.DataFrame(fits),
        "warnings": sorted(set(notices)),
    }


def compare(kind, data, models, seeds, *, settings, on_fit=None):
    """Run selected upstream cases on the SAME prepared split and seeds.

    This orchestrates original calls and adds labels; it does not modify their math.
    """
    settings = dict(settings)
    cases = [("Mặc định", dict(settings))]
    if kind == "rotation":
        selected = settings.pop("remove_halves", [settings.pop("remove_half", False)])
        cases = [
            (
                "Bỏ 50% feature" if half else "Giữ toàn bộ feature",
                {**settings, "remove_half": half},
            )
            for half in [False, True]
            if half in selected
        ]
    elif kind == "smoothing":
        selected = settings.pop("covariance_modes", [settings.pop("robust", False)])
        cases = [
            (
                "Robust · MinCovDet" if robust else "Covariance thường",
                {**settings, "robust": robust},
            )
            for robust in [False, True]
            if robust in selected
        ]
    if not cases:
        raise ValueError("Chọn ít nhất một chế độ so sánh.")
    outputs, fits, changes = [], [], []
    notices: list[str] = []
    completed = 0
    for label, case in cases:

        def progress(count, name):
            if on_fit:
                on_fit(completed + count, name + " · " + label)

        result = run(kind, data, models, seeds, settings=case, on_fit=progress)
        completed += len(result["fits"])
        fits.append(result["fits"].assign(variant=label))
        notices.extend(label + ": " + warning for warning in result["warnings"])
        frame = result["output"]
        if kind == "smoothing":
            frame, labels = frame
            changes.append(labels.assign(variant=label))
        outputs.append(frame.assign(variant=label))
    return {
        "output": pd.concat(outputs, ignore_index=True),
        "fits": pd.concat(fits, ignore_index=True),
        "label_changes": pd.concat(changes, ignore_index=True)
        if changes
        else pd.DataFrame(),
        "warnings": notices,
    }


def boundary_plot(data, fitted, features, xlim, ylim):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{name} · train {result['train_acc']:.1%} · test {result['test_acc']:.1%}"
            for name, result in fitted.items()
        ],
    )
    points = data.X_train[:, list(features)]
    inside = (
        (points[:, 0] >= xlim[0])
        & (points[:, 0] <= xlim[1])
        & (points[:, 1] >= ylim[0])
        & (points[:, 1] <= ylim[1])
    )
    visible = np.where(inside)[0][:1500]
    for column, (name, result) in enumerate(fitted.items(), 1):
        gx, gy, values = X.decision_grid(result["model"], xlim, ylim)
        figure.add_trace(
            go.Heatmap(
                x=gx,
                y=gy,
                z=values,
                colorscale="RdBu",
                zmin=0,
                zmax=1,
                showscale=column == 2,
                colorbar=dict(title="P(lớp 1)"),
            ),
            row=1,
            col=column,
        )
        for label, color in [(0, "#B2182B"), (1, "#2166AC")]:
            selected = visible[data.y_train[visible] == label]
            figure.add_trace(
                go.Scatter(
                    x=points[selected, 0],
                    y=points[selected, 1],
                    mode="markers",
                    marker=dict(
                        size=4, color=color, line=dict(width=0.3, color="white")
                    ),
                    name=data.class_names[label],
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(
            title_text=data.feature_names[features[0]], range=xlim, row=1, col=column
        )
        figure.update_yaxes(
            title_text=data.feature_names[features[1]], range=ylim, row=1, col=column
        )
    figure.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(t=40, b=50, l=10, r=10),
        legend=dict(orientation="h", y=-0.25),
    )
    return figure
