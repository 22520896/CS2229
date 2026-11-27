"""Read-only, hash-bound presentation data. Never imports a training worker."""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
REPO_URL = "https://github.com/22520896/CS2229"
MODELS = ("rf", "gbt", "resnet", "ft_transformer")
NAMES = {
    "rf": "RandomForest",
    "gbt": "GradientBoostingTree",
    "resnet": "Resnet",
    "ft_transformer": "FT Transformer",
}
DISPLAY = {
    "rf": "RF",
    "gbt": "GBT",
    "resnet": "ResNet",
    "ft_transformer": "FT-Transformer",
}
COLORS = {
    "RandomForest": "#C9A227",
    "GradientBoostingTree": "#E4572E",
    "Resnet": "#2F6FDB",
    "FT Transformer": "#17BEBB",
    "MLP": "#8E44AD",
    "SAINT": "#E83F6F",
    "XGBoost": "#2E933C",
    "HistGradientBoostingTree": "#76B041",
}
EXPERIMENTS = {
    "Rotation": "rotation",
    "Noise features": "noise_features",
    "Target smoothing": "target_smoothing",
    "Feature removal · đại diện": "feature_removal",
}


def link(path):
    return REPO_URL + "/blob/main/" + path


def csv(path):
    return pd.read_csv(ROOT / path, float_precision="round_trip")


def source_hash(path):
    content = path.read_bytes()
    if path.suffix in {".csv", ".json"}:
        content = content.replace(b"\r\n", b"\n")
    return sha256(content).hexdigest()


def verify_sources():
    manifest = json.loads((DATA / "manifest.json").read_bytes())
    if manifest.get("hash_policy") != "csv/json CRLF-to-LF; other files byte-exact":
        raise ValueError("Chính sách hash không hợp lệ")
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        if not path.is_file() or source_hash(path) != expected:
            raise ValueError("Nguồn dữ liệu không khớp: " + relative)
    return manifest


@lru_cache(maxsize=1)
def cells():
    frame = csv("reproduce/results/mechanism_cells.csv")
    frame["parsed"] = frame.condition.map(json.loads)
    if not frame.cell_id.is_unique:
        raise ValueError("Trùng ô kết quả")
    for row in frame.itertuples():
        if row.full_repeats and json.loads(row.required_seeds) != json.loads(
            row.successful_seeds
        ):
            raise ValueError("Thiếu seed trong ô đã đánh dấu hoàn tất")
    return frame


def reference_cells(mechanism, fraction=0.0):
    frame = cells()
    frame = frame.loc[
        frame.mechanism.eq(mechanism) & frame.reference_grid & frame.full_repeats
    ].copy()
    if mechanism == "rotation":
        frame = frame.loc[frame.parsed.map(lambda c: c[0] == fraction)].copy()
        frame["x"] = frame.parsed.map(lambda c: int(not c[1]))
    elif mechanism == "noise_features":
        frame["x"] = frame.parsed.map(lambda c: (c[0] - 1) * 100)
    elif mechanism == "target_smoothing":
        frame["x"] = frame.parsed.map(lambda c: c[0])
    else:
        raise ValueError("Không có lưới tham chiếu cho thí nghiệm này")
    return frame


def normalize_ours(frame, mechanism, fraction=0.0):
    anchors = csv("demo/data/author_anchors.csv")
    variant = fraction if mechanism == "rotation" else -1.0
    anchors = anchors.loc[anchors.mechanism.eq(mechanism) & anchors.variant.eq(variant)]
    result = frame.merge(
        anchors[["dataset", "q10", "maximum"]], on="dataset", validate="many_to_one"
    )
    if len(result) != len(frame) or result[["q10", "maximum"]].isna().any().any():
        raise ValueError("Thiếu mốc chuẩn hóa của tác giả")
    if not (result.maximum > result.q10).all():
        raise ValueError("Mốc chuẩn hóa không hợp lệ")
    result["score"] = (result.test_accuracy_mean - result.q10) / (
        result.maximum - result.q10
    )
    return result


def author_series(mechanism, fraction=0.0):
    if mechanism == "rotation":
        number = 23 if fraction == 0 else 24
        frame = csv(f"figures/figure_{number}.csv")
        # Appendix exports condition=True for rotation ON (main table stores deactivated).
        frame["x"] = frame.condition.astype(bool).astype(int)
        value = "score"
    elif mechanism == "noise_features":
        frame = csv("figures/figure_21_caption.csv")
        frame["x"] = frame.condition * 100
        value = "mean"
    elif mechanism == "target_smoothing":
        frame = csv("figures/figure_18.csv")
        frame["x"] = frame.condition
        value = "score"
    else:
        raise ValueError(mechanism)
    # Average search reorders within a dataset before giving datasets equal weight.
    frame = (
        frame.groupby(["data__keyword", "model_name", "x"], as_index=False)[value]
        .mean()
        .rename(
            columns={"data__keyword": "dataset", "model_name": "model", value: "score"}
        )
    )
    # Appendix per-dataset tables store RAW accuracy, unlike main-figure summaries.
    # Apply exactly the main figure's fixed per-dataset author anchors before aggregation.
    anchors = csv("demo/data/author_anchors.csv")
    variant = fraction if mechanism == "rotation" else -1.0
    anchors = anchors.loc[anchors.mechanism.eq(mechanism) & anchors.variant.eq(variant)]
    frame = frame.merge(
        anchors[["dataset", "q10", "maximum"]], on="dataset", validate="many_to_one"
    )
    frame["score"] = (frame.score - frame.q10) / (frame.maximum - frame.q10)
    return frame


def mechanism_series(mechanism, dataset=None, normalized=True, fraction=0.0):
    own = reference_cells(mechanism, fraction)
    own = (
        normalize_ours(own, mechanism, fraction)
        if normalized
        else own.assign(score=own.test_accuracy_mean)
    )
    if dataset is not None:
        own = own.loc[own.dataset.eq(dataset)]
    if own.empty:
        return pd.DataFrame(columns=["source", "model", "x", "score"])
    own["model"] = own.model.map(NAMES)
    result = (
        own.groupby(["model", "x"], as_index=False).score.mean().assign(source="Ours")
    )
    if normalized:
        author = author_series(mechanism, fraction)
        author = author.loc[author.dataset.isin(own.dataset.unique())]
        expected = set(own.dataset.unique())
        for _, block in author.groupby(["model", "x"]):
            if set(block.dataset) != expected:
                raise ValueError("Tập dataset đối chiếu không khớp")
        paper = (
            author.groupby(["model", "x"], as_index=False)
            .score.mean()
            .assign(source="Paper")
        )
        result = pd.concat([paper, result], ignore_index=True)
    return result.sort_values(["source", "model", "x"]).reset_index(drop=True)


def removal_series(dataset, kept=True):
    f = cells()
    f = f.loc[
        f.mechanism.eq("feature_removal") & f.full_repeats & f.dataset.eq(dataset)
    ].copy()
    f = f.loc[f.parsed.map(lambda c: c[1] is not kept)].copy()
    f["x"] = f.parsed.map(lambda c: c[0] * 100)
    f["model"] = f.model.map(NAMES)
    return (
        f[["model", "x", "test_accuracy_mean"]]
        .rename(columns={"test_accuracy_mean": "score"})
        .assign(source="Ours")
    )


def change_table(series):
    rows = []
    for (source, model), frame in series.groupby(["source", "model"], sort=False):
        frame = frame.sort_values("x")
        rows.append(
            {
                "Nguồn": source,
                "Model": model,
                "Đầu": frame.score.iloc[0],
                "Cuối": frame.score.iloc[-1],
                "Δ": frame.score.iloc[-1] - frame.score.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def benchmark_series(source, panel):
    if source == "Paper":
        frame = csv("figures/figure_01.csv")
        frame = frame.loc[frame.panel.eq(panel)].copy()
    else:
        kind = "categorical" if panel.startswith("categorical") else "numeric"
        task = "regression" if "regression" in panel else "classification"
        frame = csv("evidence/v2_benchmark.csv")
        frame = frame.loc[frame.group.eq(kind) & frame.task.eq(task)].rename(
            columns={
                "hpo_budget": "rank",
                "model_name": "model",
                "mean_test": "mean",
                "min_test": "low",
                "max_test": "high",
            }
        )
    if "model_name" in frame:
        frame = frame.rename(columns={"model_name": "model"})
    return frame.sort_values(["model", "rank"]).reset_index(drop=True)


def sample_numeric(mechanism, dataset, model, condition):
    samples = json.loads((ROOT / "reproduce/results/sample_fits.json").read_bytes())
    selected = next(
        (
            s
            for s in samples
            if s["id"] == mechanism
            and s["dataset"] == dataset
            and s["model"] == model
            and s["condition"] == condition
        ),
        None,
    )
    if selected is None:
        return None
    path = ROOT / "reproduce" / selected["file"]
    if sha256(path.read_bytes()).hexdigest() != selected["numeric_sha256"]:
        raise ValueError("Numeric evidence không khớp")
    with np.load(path, allow_pickle=False) as data:
        y, prediction = data["y_test"], data["pred_test"]
        accuracy = float(np.mean(y == prediction))
        if abs(accuracy - selected["metrics"]["test"]["accuracy"]) > 1e-12:
            raise ValueError("Metric không khớp dự đoán")
        matrix = np.array(
            [
                [int(((y == i) & (prediction == j)).sum()) for j in [0, 1]]
                for i in [0, 1]
            ]
        )
        if matrix.tolist() != selected["metrics"]["test"]["confusion_matrix"]:
            raise ValueError("Confusion matrix không khớp")
    return selected, matrix


@lru_cache(maxsize=1)
def configurations():
    return {
        r["cell_id"]: r
        for r in json.loads(
            (ROOT / "reproduce/configs/published_conditions.json").read_bytes()
        )
    }
