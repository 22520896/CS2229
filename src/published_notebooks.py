"""Build notebooks for the fixed and tuned estimator comparisons."""

from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import numpy as np

from .appendix_notebooks import BOOT, SUPPLEMENT
from .author_data import ROOT
from .plots import COLORS, FIGURES, NAMES, shared_legend, theme


FILES = {
    39: "figure_39_tuned_estimators_numeric",
    40: "figure_40_tuned_estimators_categorical",
}
TITLES = {
    39: "Cố định hay tune số cây · đặc trưng số",
    40: "Cố định hay tune số cây · có biến phân loại",
}
PDF_DEFAULTS = ROOT / "evidence" / "tuned_estimators_pdf_defaults.json"


def load_figure(number):
    return json.loads(PDF_DEFAULTS.read_text(encoding="utf-8"))["figures"][str(number)]


def series_style(name):
    base, _, suffix = name.partition("_")
    return COLORS[base], "--" if suffix == "tune" else "-", NAMES[base] + (f" · {suffix}" if suffix else "")


def export_tuned(number, analyzed, published):
    theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.5))
    for (task, panel), axis in zip(published["panels"].items(), axes):
        points = analyzed[analyzed["panel"] == task].sort_values("rank")
        for model, subset in points.groupby("model_name", sort=True):
            color, linestyle, label = series_style(model)
            ranked = subset[subset["rank"] > 0]
            axis.fill_between(ranked["rank"], ranked["low"], ranked["high"], color=color, alpha=.11)
            axis.plot(ranked["rank"], ranked["mean"], color=color, linestyle=linestyle, label=label)
            axis.axhline(subset.loc[subset["rank"] == 0, "mean"].iloc[0], color=color,
                         linestyle=":", linewidth=.9, alpha=.52)
        limits = np.asarray(panel["axes"]["bounds"])
        axis.set(xscale="log", xlim=sorted(limits[:, 0]), ylim=sorted(limits[:, 1]),
                 xlabel="Số lượt tìm kiếm",
                 title=f'{"Phân loại" if task == "classification" else "Hồi quy"} · {points.datasets.iloc[0]} bộ')
        axis.grid(axis="y")
    axes[0].set_ylabel("Điểm test chuẩn hóa")
    shared_legend(fig, axes, bottom=.28)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=175, bbox_inches="tight", pad_inches=.13)
    plt.close(fig)
    analyzed.to_csv(path.with_suffix(".csv"), index=False)
    return path


def create(number):
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    cells = [
        markdown(f"# Figure {number} · {TITLES[number]}\n\n"
                 "**Tái lập phân tích từ kết quả công bố.** Nhánh cố định lấy từ commit `5b2d812`. "
                 "Nhánh tune đọc checkpoint `7265a3f`, nơi các lượt dùng nhiều mức số cây. "
                 "Giữ đúng nhóm bộ dữ liệu và mô hình, loại isolet ở hồi quy số như script. "
                 "Chọn theo validation và dựng dải min–max từ 15 hoán vị với seed 0."),
        code(BOOT + "\nfrom src.published_notebooks import export_tuned, load_figure\n"
             "from src.tuned_estimators import analyze, compare_published_defaults\n"
             f"published = load_figure({number})\n"
             f"values = analyze({number})\n"
             f"display(Image(filename=str(export_tuned({number}, values, published))))\n"
             "display(compare_published_defaults(values, published).round(8))"),
        markdown("Bảng kiểm tra hạng 0 không phụ thuộc vào hoán vị. Cột PDF được hiệu chuẩn từ vị trí đường mặc định. "
                 "Các đường HPO trong notebook dùng seed cố định để chạy lại, còn seed vẽ của tác giả chưa xác định. "
                 f"[Hình công bố trang 47]({SUPPLEMENT}#page=47) · "
                 "[Script gốc](https://github.com/LeoGrin/tabular-benchmark/blob/"
                 "9d54cf53d9fd3159e367e70a00005f4fcbf2c79d/analyses/hyperparameters_trees_comparison.R)"),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    })


def build(number):
    path = ROOT / "notebooks" / (FILES[number] + ".ipynb")
    notebook = create(number)
    NotebookClient(notebook, kernel_name="python3", timeout=600).execute(cwd=str(path.parent))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(path)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=tuple(FILES))
    build(parser.parse_args().number)
