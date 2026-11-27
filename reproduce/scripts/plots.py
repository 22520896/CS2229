"""Presentation-only aggregation of already audited cell means;no model fitting."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODELS = ("rf", "gbt", "resnet", "ft_transformer")
NAMES = {
    "rf": "RF",
    "gbt": "GBT",
    "resnet": "ResNet",
    "ft_transformer": "FT-Transformer",
}
COLORS = {
    "rf": "#245ca6",
    "gbt": "#26745d",
    "resnet": "#a15a22",
    "ft_transformer": "#7044aa",
}


def cells(mechanism):
    frame = pd.read_csv(BASE / "results/mechanism_cells.csv")
    frame = frame.loc[
        (frame.mechanism == mechanism) & frame.reference_grid & frame.full_repeats
    ].copy()
    frame["condition_value"] = frame.condition.map(json.loads)
    return frame


def means(mechanism, conditions):
    frame = cells(mechanism)
    output = {}
    for model in MODELS:
        values = []
        for condition in conditions:
            chosen = frame.loc[
                (frame.model == model)
                & frame.condition_value.map(lambda v: v == condition)
            ]
            assert chosen.task.is_unique and len(chosen) == (
                15 if mechanism == "target_smoothing" else 14
            )
            values.append(float(chosen.test_accuracy_mean.mean()) * 100)
        output[model] = values
    return output


def plot(mechanism):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 3.7))
    if mechanism == "rotation":
        values = means(
            mechanism, [[0.0, True], [0.0, False], [0.5, True], [0.5, False]]
        )
        for i, model in enumerate(MODELS):
            score = values[model]
            ax.bar(
                np.arange(2) + (i - 1.5) * 0.19,
                [score[1] - score[0], score[3] - score[2]],
                width=0.17,
                color=COLORS[model],
                label=NAMES[model],
            )
        ax.set_xticks([0, 1], ["Giữ toàn bộ biến", "Loại trước 50% biến"])
        ax.set_ylabel("Có xoay − không xoay (điểm %)")
        ax.axhline(0, color="#8793a1", linewidth=0.8)
    else:
        smoothing = mechanism == "target_smoothing"
        conditions = (
            [[x, "robust"] for x in [0, 0.05, 0.1, 0.25]]
            if smoothing
            else [[x] for x in [1, 1.5, 2]]
        )
        x = [0, 0.05, 0.1, 0.25] if smoothing else [0, 50, 100]
        values = means(mechanism, conditions)
        for model in MODELS:
            ax.plot(
                x,
                values[model],
                "o-",
                color=COLORS[model],
                label=NAMES[model],
                linewidth=1.8,
            )
        ax.set_xticks(x)
        ax.set_xlabel(
            "Hệ số covariance làm trơn nhãn train"
            if smoothing
            else "Số biến Gaussian thêm / số biến đầu vào (%)"
        )
        ax.set_ylabel("Test accuracy trung bình (%)")
    ax.grid(axis="y", color="#e5eaf0")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=9)
    fig.tight_layout()
    return fig


def export():
    for mechanism in ["noise_features", "rotation", "target_smoothing"]:
        fig = plot(mechanism)
        for extension in ["png", "svg"]:
            fig.savefig(
                BASE / "figures" / (mechanism + "." + extension),
                dpi=180,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    export()
