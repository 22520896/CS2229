"""Render the notebook charts and export identical assets for the report and slides."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .analysis import MODEL, PANELS, REMOVE, ROTATE, SMOOTH, analyze
from .author_data import ROOT


FIGURES = ROOT / "figures"
EVIDENCE = ROOT / "evidence"
COLORS = {
    "RandomForest": "#236559", "GradientBoostingTree": "#057a9b",
    "HistGradientBoostingTree": "#495796", "XGBoost": "#284a66",
    "FT Transformer": "#c3622b", "Resnet": "#a64969",
    "MLP": "#81569b", "SAINT": "#876137",
}
NAMES = {
    "RandomForest": "RF", "GradientBoostingTree": "GBT", "HistGradientBoostingTree": "HGBT",
    "XGBoost": "XGBoost", "FT Transformer": "FT-Transformer", "Resnet": "ResNet",
    "MLP": "MLP", "SAINT": "SAINT",
}
COLORS["keep"] = "#ac592e"
COLORS["remove"] = "#126c79"
NAMES["keep"] = "Chỉ dùng nhóm bị loại"
NAMES["remove"] = "Dùng nhóm được giữ lại"
INK = "#1d3040"


def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#bac7ce", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "grid.color": "#dbe4e9", "grid.linewidth": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def ribbon(axis, data, x, group, *, default_line=False):
    for tag, block in data.groupby(group, sort=True):
        if tag not in COLORS:
            continue
        color = COLORS[tag]
        block = block.sort_values(x)
        axis.fill_between(block[x], block.low, block.high, color=color, alpha=0.13, linewidth=0)
        baseline = block[block[x] == 0]
        if default_line and len(baseline):
            axis.axhline(baseline["mean"].iloc[0], color=color, alpha=0.48, linewidth=1, linestyle=":")
            block = block[block[x] > 0]
        axis.plot(block[x], block["mean"], color=color, linewidth=1.7, label=NAMES[tag])
    axis.grid(axis="y")
    axis.set_axisbelow(True)


def shared_legend(fig, axes, bottom=0.12):
    entries = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        entries.update(dict(zip(labels, handles)))
    fig.legend(entries.values(), entries.keys(), loc="lower center", frameon=False, ncol=4,
               bbox_to_anchor=(0.5, 0.005), fontsize=9)
    fig.subplots_adjust(bottom=bottom, left=0.11, right=0.97, top=0.94, hspace=0.48, wspace=0.25)


def benchmark_plot(data, project=False):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4))
    titles = ("Đặc trưng số · phân loại", "Đặc trưng số · hồi quy",
              "Có đặc trưng phân loại · phân loại", "Có đặc trưng phân loại · hồi quy")
    for panel, title, axis in zip(PANELS, titles, axes.flat):
        values = data[data.panel == panel]
        ribbon(axis, values, "rank", MODEL, default_line=not project)
        axis.set_title(f"{title} · {values.datasets.iloc[0]} thiết lập", fontsize=10.5)
        axis.set_ylabel("Điểm test chuẩn hóa")
        if project:
            axis.set_xlabel("Số cấu hình random ngoài default")
            axis.set_xticks(np.unique([0, int(values["rank"].max())]))
        else:
            axis.set_xscale("log")
            axis.set_xlim(1, 400)
            axis.set_xlabel("Thứ hạng random search")
    shared_legend(fig, axes.flat, bottom=0.19)
    return fig


def boxes(axis, data, condition, labels, *, legends=True):
    levels = sorted(labels)
    models = [model for model in COLORS if model in data[MODEL].unique()]
    positions = np.arange(len(levels))
    offsets = np.linspace(-0.27, 0.27, len(models))
    for model, offset in zip(models, offsets):
        scores = [
            data.loc[(data[condition] == label) & (data[MODEL] == model), "score"].dropna().to_numpy()
            for label in levels
        ]
        color = COLORS[model]
        axis.boxplot(scores, positions=positions + offset, widths=0.12, manage_ticks=False,
                     patch_artist=True, boxprops={"facecolor": color, "alpha": 0.65, "edgecolor": color},
                     medianprops={"color": INK}, whiskerprops={"color": color},
                     capprops={"color": color}, flierprops={"markersize": 2, "markeredgecolor": color})
        if legends:
            axis.plot([], [], color=color, linewidth=3, label=NAMES[model])
    axis.set_xticks(positions, [labels[v] for v in levels])
    axis.set_xlim(-0.53, len(levels) - 0.47)
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_ylabel("Điểm test chuẩn hóa")


def plot(number, data):
    theme()
    if number == 1:
        return benchmark_plot(data)
    if number == 2:
        fig, axis = plt.subplots(figsize=(9.3, 4.2))
        boxes(axis, data, SMOOTH, {0.0: "0", 0.05: "0,05", 0.1: "0,10", 0.25: "0,25"})
        axis.set_xlabel("Hệ số hiệp phương sai của kernel Gaussian")
        shared_legend(fig, [axis], bottom=0.23)
        return fig
    if number == 3:
        fig, axis = plt.subplots(figsize=(8.8, 4.2))
        ribbon(axis, data, REMOVE, "method")
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlabel("Tỷ lệ đặc trưng bị loại, từ ít quan trọng nhất")
        axis.set_ylabel("Điểm test GBT chuẩn hóa")
        shared_legend(fig, [axis], bottom=0.23)
        return fig
    if number == 4:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.45), sharey=True)
        for axis, values, title in zip(axes, data, ("4a · Loại đặc trưng ít quan trọng", "4b · Thêm đặc trưng nhiễu")):
            ribbon(axis, values, "fraction", MODEL)
            axis.xaxis.set_major_formatter(PercentFormatter(1))
            axis.set_title(title, fontsize=11)
            axis.set_xlabel("Tỷ lệ đặc trưng")
        axes[0].set_ylabel("Điểm test chuẩn hóa")
        shared_legend(fig, axes, bottom=0.23)
        return fig
    if number == 5:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharey=True)
        # Sorting the original deactivation flag would put rotation first.
        data = data.assign(rotation=lambda frame: (~frame[ROTATE].astype(bool)).astype(int))
        for axis, removed in zip(axes, [0.0, 0.5]):
            boxes(axis, data[data.removed == removed], "rotation", {0: "Trước xoay", 1: "Sau xoay"})
            axis.set_title(f"Loại {removed:.0%} đặc trưng trước phép xoay", fontsize=10.5)
        axes[1].set_ylabel("")
        shared_legend(fig, axes, bottom=0.18)
        return fig
    raise ValueError(number)


def project_plot(number):
    theme()
    if number == 1:
        data = pd.read_csv(EVIDENCE / "v2_benchmark.csv").rename(columns={
            "hpo_budget": "rank", "mean_test": "mean", "min_test": "low", "max_test": "high"})
        data["panel"] = data["group"].replace({"numeric": "numerical"}) + "_" + data.task + "_medium"
        return benchmark_plot(data, project=True)
    if number == 2:
        data = pd.read_csv(EVIDENCE / "v2_smoothing.csv").replace({"rf": "RandomForest", "mlp": "MLP"})
        fig, axis = plt.subplots(figsize=(7.4, 3.5))
        for name, block in data.groupby("model"):
            block = block.sort_values("level")
            axis.plot(block.level, block.test_accuracy, marker="o", color=COLORS[name], label=NAMES[name])
        axis.set_xticks([0, 0.05, 0.10, 0.25])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylabel("Accuracy trên eye_movements")
        axis.set_xlabel("Hệ số làm trơn")
        axis.grid(axis="y")
        shared_legend(fig, [axis], bottom=0.24)
        return fig
    if number == 5:
        data = pd.read_csv(EVIDENCE / "v2_rotation.csv")
        labels = dict(rf="RandomForest", gbt="GradientBoostingTree", ft_transformer="FT Transformer", resnet="Resnet")
        means = data.groupby(["model", "remove_fraction"])["rotation_minus_control"].mean().unstack()
        fig, axis = plt.subplots(figsize=(7.6, 3.5))
        for i, (model, row) in enumerate(means.iterrows()):
            axis.scatter(row[0.0], i - 0.1, color=COLORS[labels[model]], marker="o")
            axis.scatter(row[0.5], i + 0.1, color=COLORS[labels[model]], marker="D")
        axis.scatter([], [], color=INK, marker="o", label="Giữ toàn bộ")
        axis.scatter([], [], color=INK, marker="D", label="Loại 50%")
        axis.axvline(0, color="#879da8", linewidth=1)
        axis.set_yticks(np.arange(len(means)), [NAMES[labels[model]] for model in means.index])
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlabel("Accuracy sau xoay trừ trước xoay")
        shared_legend(fig, [axis], bottom=0.26)
        fig.subplots_adjust(left=0.25)
        return fig
    raise ValueError(number)


def export(number, data, destination=FIGURES):
    destination.mkdir(parents=True, exist_ok=True)
    fig = plot(number, data)
    fig.savefig(destination / f"figure_{number:02d}.png", dpi=150)
    files = (data,) if isinstance(data, pd.DataFrame) else data
    for index, table in enumerate(files):
        suffix = ("", "_4b_script", "_4b_caption")[index]
        if number == 4 and index == 0:
            suffix = "_4a"
        table.to_csv(destination / f"figure_{number:02d}{suffix}.csv", index=False)
    return fig


def export_project(number, destination=FIGURES):
    fig = project_plot(number)
    fig.savefig(destination / f"figure_{number:02d}_v2.png", dpi=150)
    return fig
