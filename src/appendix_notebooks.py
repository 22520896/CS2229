"""Build one executed notebook per appendix figure, plus the appendix tables."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np

from .author_data import MODEL, ROOT, SOURCES, TASK
from .plots import COLORS, FIGURES, NAMES, ribbon, shared_legend, theme
from .time_benchmark import SOURCES_BY_FIGURE


SUPPLEMENT = (
    "https://papers.neurips.cc/paper_files/paper/2022/file/"
    "0378c7692da36807bdec87ab043cdadc-Supplemental-Datasets_and_Benchmarks.pdf"
)
FILES = {
    "6": "figure_06_time_numerical",
    "7": "figure_07_time_categorical",
    "8": "figure_08_train_size_numerical_classification",
    "9": "figure_09_train_size_numerical_regression",
    "10": "figure_10_train_size_categorical_classification",
    "11": "figure_11_train_size_categorical_regression",
    "12": "figure_12_numerical_classification",
    "13": "figure_13_numerical_regression",
    "14": "figure_14_categorical_classification",
    "15": "figure_15_categorical_regression",
    "16": "figure_16_large_numerical",
    "17": "figure_17_large_categorical",
    "18": "figure_18_smoothing_by_dataset",
    "20": "figure_20_feature_importance_by_dataset",
    "21": "figure_21_feature_addition_by_dataset",
    "22": "figure_22_feature_removal_by_dataset",
    "23": "figure_23_rotation_by_dataset",
    "24": "figure_24_rotation_pruned_by_dataset",
    "33": "figure_33_bayesian_classification",
    "34": "figure_34_bayesian_regression",
    "a4": "appendix_a4_regression",
    "catalog": "appendix_a1_a3_catalog",
}
REGRESSIONS = {
    "smoothing": "Làm trơn mục tiêu",
    "remove": "Loại đặc trưng ít thông tin",
    "add": "Thêm đặc trưng không mang thông tin",
    "rotation": "Xoay và loại 50% đặc trưng",
}


def plot_time(number, values):
    theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    task_type = ("Phân loại", "Hồi quy")
    lower_limits = (0.6, 0.6) if number == 6 else (0.4, 0.5)
    for axis, (_, rows), label, minimum in zip(axes, values.groupby("panel", sort=False),
                                               task_type, lower_limits):
        drawable = rows[rows["seconds"].gt(0)]
        ribbon(axis, drawable, "seconds", MODEL)
        axis.set_title(f"{label} · {rows.datasets.iloc[0]} bộ dữ liệu")
        axis.set(xscale="log", ylim=(minimum, 1.0), xlabel="Thời gian tìm kiếm (giây)",
                 ylabel="Điểm test chuẩn hóa")
    shared_legend(fig, axes, bottom=0.27)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=175, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    values.to_csv(path.with_suffix(".csv"), index=False)
    return path


def plot_raw(number, values):
    theme()
    split = "panel" in values
    group_keys = ["panel", "data__keyword"] if split else "data__keyword"
    datasets = list(values.groupby(group_keys, sort=True))
    columns = 4
    rows = (len(datasets) + columns - 1) // columns
    fig, matrix = plt.subplots(rows, columns, figsize=(12, 2.25 * rows + 1.05), squeeze=False)
    axes = matrix.ravel()
    for axis, (dataset, block) in zip(axes, datasets):
        ribbon(axis, block, "rank", MODEL, default_line=True)
        title = (f"{dataset[1]} · {'R²' if '_regression_' in dataset[0] else 'acc.'}"
                 if split else dataset)
        axis.set_title(title, fontsize=10)
        axis.set(xscale="log", xlim=(1, 400))
        axis.tick_params(labelsize=8)
    for axis in axes[len(datasets):]:
        axis.set_visible(False)
    axis_label = "Điểm test thô" if split else ("R² test" if number in (13, 15) else "Accuracy test")
    fig.supylabel(axis_label, x=0.017, fontsize=11)
    fig.supxlabel("Số cấu hình random ngoài default", y=0.074, fontsize=10.5)
    shared_legend(fig, axes, bottom=0.13)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=165, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    values.to_csv(path.with_suffix(".csv.gz"), index=False,
                  compression={"method": "gzip", "mtime": 0})
    return path


def plot_size(number, values):
    theme()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), sharey=True)
    minimum = (0.6, 0.6, 0.5, 0.4)[number - 8]
    for axis, (size, rows) in zip(axes, values.groupby("train_cap", sort=True)):
        ribbon(axis, rows.dropna(subset=["mean"]), "rank", MODEL, default_line=True)
        axis.set_title(f"{size // 1000}K mẫu train · {rows.datasets.iloc[0]} bộ dữ liệu")
        axis.set(xscale="log", xlim=(1, 400), ylim=(minimum, 1), xlabel="Số cấu hình random")
    axes[0].set_ylabel("Điểm test chuẩn hóa")
    shared_legend(fig, axes, bottom=0.27)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=175, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    values.to_csv(path.with_suffix(".csv"), index=False)
    return path


def plot_bo(number, values):
    theme()
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    ribbon(axis, values[values["method"].eq("Random")], "rank", MODEL, default_line=True)
    for model, group in values[values["method"].eq("Bayesian")].groupby(MODEL, sort=True):
        group = group[group["rank"].gt(0)].sort_values("rank")
        axis.plot(group["rank"], group["mean"], color=COLORS[model], linestyle="--", linewidth=1.6)
    axis.set(xscale="log", xlim=(1, 200), ylim=(0.5 if number == 33 else 0.3, 1.0),
             xlabel="Số cấu hình đã thử", ylabel="Điểm test chuẩn hóa")
    axis.set_title("Random (liền) và Bayesian (nét đứt)")
    shared_legend(fig, [axis], bottom=0.22)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=175, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    values.to_csv(path.with_suffix(".csv"), index=False)
    return path


def plot_ablation(number, values):
    theme()
    boxplot = number in (18, 23, 24)
    panels = list(values.groupby(TASK, sort=True))
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, matrix = plt.subplots(rows, cols, figsize=(12, 2.2 * rows + 1), squeeze=False)
    axes = matrix.ravel()
    models = sorted(values[MODEL].unique()) if boxplot else []
    steps = sorted(values["condition"].unique())
    for axis, (task, data) in zip(axes, panels):
        if boxplot:
            offsets = np.linspace(-0.24, 0.24, len(models))
            for model, offset in zip(models, offsets):
                groups = []
                positions = []
                for index, step in enumerate(steps):
                    sample = data[data[MODEL].eq(model) & data["condition"].eq(step)]["score"].dropna()
                    if len(sample):
                        groups.append(sample.to_numpy())
                        positions.append(index + offset)
                if groups:
                    axis.boxplot(groups, positions=positions, widths=0.14, manage_ticks=False,
                                 boxprops={"color": COLORS[model]}, whiskerprops={"color": COLORS[model]},
                                 capprops={"color": COLORS[model]}, medianprops={"color": COLORS[model]},
                                 flierprops={"marker": ".", "markersize": 2, "markeredgecolor": COLORS[model]})
            labels = ["Không", "Có"] if number in (23, 24) else [f"{step:g}" for step in steps]
            axis.set_xticks(range(len(steps)), labels=labels)
            axis.grid(axis="y")
        else:
            ribbon(axis, data, "condition", "method" if number == 20 else MODEL)
            axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.set_title(task, fontsize=9.5)
        axis.tick_params(labelsize=8)
    for axis in axes[len(panels):]:
        axis.set_visible(False)
    xlabel = {18: "Hệ số làm trơn", 20: "Tỷ lệ tách", 21: "Tỷ lệ thêm", 22: "Tỷ lệ loại",
              23: "Phép xoay", 24: "Phép xoay sau khi loại 50%"}[number]
    fig.supylabel("Accuracy test chuẩn hóa" if number == 20 else "Accuracy test", x=0.014, fontsize=10.5)
    fig.supxlabel(xlabel, y=0.074, fontsize=10.5)
    if boxplot:
        handles = [Line2D([], [], color=COLORS[model], label=NAMES[model]) for model in models]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.005), fontsize=9)
        fig.subplots_adjust(bottom=0.14, left=0.09, right=0.98, top=0.95, hspace=0.55, wspace=0.29)
    else:
        shared_legend(fig, axes, bottom=0.14)
    path = FIGURES / f"figure_{number:02d}.png"
    fig.savefig(path, dpi=155, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    values.to_csv(path.with_suffix(".csv"), index=False)
    return path


BOOT = """from pathlib import Path
import sys

root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/analysis.py").is_file())
sys.path.insert(0, str(root))
from IPython.display import Image, Markdown, display
import pandas as pd
"""


def time_cells(number):
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    source = SOURCES[SOURCES_BY_FIGURE[number]]
    heading = "Chỉ đặc trưng số" if number == 6 else "Có đặc trưng phân loại"
    quantile = "50%" if number == 6 else "40%"
    return [
        md(f"# Figure {number} · Ngân sách thời gian\n\n{heading}. Mục A.2.1 trong phụ lục NeurIPS 2022."),
        md("## Quy trình\n\nĐọc điểm và số giây của các lượt công bố trong bản CSV lịch sử. "
           "Đưa cấu hình mặc định lên đầu 15 thứ tự ngẫu nhiên, cộng dồn thời gian và chọn cấu hình "
           "theo validation. Điểm test chuẩn hóa trên từng bộ dữ liệu, dùng phân vị 10% cho phân loại "
           f"và {quantile} cho hồi quy. Đường là trung bình, vùng tô là min–max qua các hoán vị. "
           "Trục thời gian dùng thang log và phản ánh phép đo từ phần cứng của tác giả."),
        py(BOOT + "\nfrom src.time_benchmark import analyze\n"
           "from src.appendix_notebooks import plot_time\n"
           f"values = analyze({number})\n"
           f"display(Image(filename=str(plot_time({number}, values))))"),
        py('display(values.groupby("panel", sort=False).head(4).round(5))'),
        md(f"## Nguồn\n\n[Phụ lục A.2.1]({SUPPLEMENT}#page=17) · [CSV lịch sử]({source.url}) "
           f"\n\nSHA-256 của CSV gốc `{source.sha256}`."),
    ]


def raw_cells(number):
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cohorts = ("Chỉ đặc trưng số · phân loại", "Chỉ đặc trưng số · hồi quy",
               "Có đặc trưng phân loại · phân loại", "Có đặc trưng phân loại · hồi quy",
               "Chỉ đặc trưng số · 50K", "Có đặc trưng phân loại · 50K")
    cohort = cohorts[number - 12]
    clamp = "Điểm R² âm được chặn ở 0 như phần hiển thị của tác giả." if number in (13, 15, 16, 17) else ""
    source = (f"[CSV lịch sử]({SOURCES['large_numerical' if number == 16 else 'large_categorical'].url})"
              if number >= 16 else "[CSV benchmark công bố](https://ndownloader.figshare.com/files/40081681)")
    return [
        md(f"# Figure {number} · Điểm test từng bộ dữ liệu\n\n{cohort}. "
           "Mỗi ô dùng thang điểm riêng. Đường chấm ngang đánh dấu cấu hình mặc định. "
           "Trong từng bộ dữ liệu, ngân sách so sánh được giới hạn theo số lượt chung. "
           "Chọn cấu hình từ validation và tổng hợp qua 15 lần xáo. " + clamp),
        py(BOOT + "\nfrom src.dataset_benchmark import analyze\nfrom src.appendix_notebooks import plot_raw\n"
           f"values = analyze({number})\ndisplay(Image(filename=str(plot_raw({number}, values))))"),
        py('display(values.groupby("data__keyword")["rank"].max().rename("max_rank").to_frame())'),
        md(f"[Biểu đồ điểm thô trong phụ lục]({SUPPLEMENT}#page={20 if number < 16 else 21}) · {source}"),
    ]


def size_cells(number):
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    source = SOURCES["large_numerical" if number < 10 else "large_categorical"]
    return [
        md(f"# Figure {number} · 10K so với 50K mẫu train\n\n"
           "Chọn cùng các bộ dữ liệu có trong nhóm 50K. Mỗi ô chuẩn hóa theo tham số của "
           "script tác giả cho đúng nhóm và kích thước. Đường chấm ngang là cấu hình mặc định. "
           "Đường liền và vùng tô là trung bình và min–max của 15 thứ tự tìm kiếm, chọn bằng validation."),
        py(BOOT + "\nfrom src.dataset_benchmark import analyze\nfrom src.appendix_notebooks import plot_size\n"
           f"values = analyze({number})\ndisplay(Image(filename=str(plot_size({number}, values))))"),
        py('display(values[values["rank"].eq(0)][["train_cap", "model_name", "mean", "datasets"]].round(5))'),
        md(f"[Phụ lục A.2.2]({SUPPLEMENT}#page=18) · [CSV lịch sử]({source.url})"),
    ]


def bo_cells(number):
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    task = "Phân loại" if number == 33 else "Hồi quy"
    return [
        md(f"# Figure {number} · Random search và tối ưu Bayes\n\n{task} với đặc trưng số. "
           "Đường random tổng hợp 15 hoán vị và dải min–max. Chuỗi Bayes giữ thứ tự bước gốc. "
           "Hai phương pháp cùng chọn theo validation và dùng cấu hình mặc định ở hạng 0. "
           "Điểm được chuẩn hóa theo từng bộ dữ liệu từ đầu vào gộp của hai phương pháp."),
        py(BOOT + "\nfrom src.bo_benchmark import analyze\nfrom src.appendix_notebooks import plot_bo\n"
           f"values = analyze({number})\ndisplay(Image(filename=str(plot_bo({number}, values))))"),
        py('display(values[values["rank"].isin([20, 100, 200])].round(5).reset_index(drop=True))'),
        md(f"[Phụ lục NeurIPS 2022]({SUPPLEMENT}#page=42) · "
           "[Script gốc](https://github.com/LeoGrin/tabular-benchmark/blob/"
           "5b2d812329d191f949f9d085616a6c41d3198688/analyses/plot_bo.R)"),
    ]


def ablation_cells(number):
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    descriptions = {
        18: ("Làm trơn theo từng bộ dữ liệu", "Điểm accuracy thô, hạng 60 và 15 hoán vị. "
             "Boxplot cho từng mô hình và hệ số làm trơn."),
        20: ("Tách nhóm giữ lại và nhóm bị loại", "GBT, hạng 20, 30 hoán vị. "
             "Điểm đã chuẩn hóa với mốc 0,5 accuracy và vùng tô là min–max."),
        21: ("Thêm đặc trưng Gaussian", "Accuracy thô ở hạng 20. Script công bố dùng 15 hoán vị. "
             "Phần mô tả hình dùng 30 hoán vị; bảng cuối lưu và đối chiếu bản 30."),
        22: ("Loại đặc trưng ít thông tin", "Accuracy thô, hạng 30 theo script và 30 hoán vị. "
             "Vùng tô là min–max theo hoán vị."),
        23: ("Xoay với toàn bộ đặc trưng", "Accuracy thô ở hạng 20. Boxplot từ 30 hoán vị."),
        24: ("Xoay sau khi loại 50% đặc trưng", "Accuracy thô ở hạng 20. Boxplot từ 30 hoán vị."),
    }
    title, description = descriptions[number]
    cells = [
        md(f"# Figure {number} · {title}\n\n{description} Mỗi ô là một bộ dữ liệu."),
        py(BOOT + "\nfrom src.dataset_ablations import analyze\nfrom src.appendix_notebooks import plot_ablation\n"
           f"values = analyze({number})\ndisplay(Image(filename=str(plot_ablation({number}, values))))"),
        py('display(values.head(8).round(5))'),
    ]
    if number == 21:
        cells.append(py('caption = analyze(21, caption=True)\n'
                        'caption.to_csv(root / "figures/figure_21_caption.csv", index=False)\n'
                        'display(caption.head(8).round(5))'))
    return [*cells, md(f"[Phụ lục A.4]({SUPPLEMENT}#page=28) · [CSV ablation lịch sử]({SOURCES['ablations'].url})")]


def a4_cells():
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md("# Phụ lục A.4 · Bốn hồi quy kiểm chứng\n\nƯớc lượng hệ số cho làm trơn, bỏ hoặc thêm đặc trưng và xoay. "
           "Mỗi hồi quy dùng điểm test chuẩn hóa của cấu hình lấy theo validation. Mô hình có hiệu ứng "
           "cố định theo bộ dữ liệu; các hàng hiệu ứng này được ẩn như trong paper. Seed 0 áp dụng "
           "cho đối chiếu độc lập với R. Cột estimate của tác giả trích từ CSV lịch sử đã làm tròn, "
           "để đọc cạnh kết quả chạy lại. Mốc hồi quy có thể ảnh hưởng đến hệ số chặn."),
        py(BOOT + '\nfrom src.analysis import scores\n'
           'from src.appendix_a4 import analyze, parity\nfrom src.appendix_notebooks import REGRESSIONS\n'
           'coefficients = analyze(scores("ablations"))\n'
           'reference = pd.read_csv(root / "evidence/a4_r_reference.csv", float_precision="round_trip")\n'
           'display(parity(coefficients, reference))\n'
           'published = pd.read_csv(root / "evidence/a4_author_tables.csv")\n'
           'coefficients.to_csv(root / "evidence/a4_coefficients.csv", index=False)'),
        py('for analysis, title in REGRESSIONS.items():\n'
           '    rows = coefficients[coefficients.analysis.eq(analysis)].merge(\n'
           '        published.loc[published.analysis.eq(analysis), ["term", "estimate"]],\n'
           '        on="term", suffixes=("", "_author"), validate="one_to_one")\n'
           '    display(Markdown(f"### {title} · {rows.nobs.iloc[0]:,} quan sát"))\n'
           '    for column, digits in {"estimate": 6, "estimate_author": 3, "std_error": 6, "statistic": 4}.items():\n'
           '        rows[column] = rows[column].map(lambda value: f"{value:.{digits}f}")\n'
           '    rows["p_value"] = rows["p_value"].map(lambda value: f"{value:.2e}")\n'
           '    display(rows[["term", "estimate", "estimate_author", "std_error", "statistic", "p_value"]])'),
        md(f"[Các mô hình hồi quy trong phụ lục A.4]({SUPPLEMENT}#page=28)"),
    ]


def catalog_cells():
    md, py = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md("# Phụ lục A.1 và A.3 · Danh mục dữ liệu và HPO\n\n"
           "Danh mục ghi 57 thiết lập theo nhóm bài toán và loại đặc trưng. Một bộ dữ liệu có thể "
           "xuất hiện trong nhiều thiết lập. Paper mô tả 45 bộ dữ liệu; giá trị này không phải số hàng của danh mục."),
        py(BOOT + '\ndatasets = pd.read_csv(root / "evidence/appendix_datasets.csv")\n'
           'suites = {337: "Số · Phân loại", 336: "Số · Hồi quy",\n'
           '          334: "Có biến phân loại · Phân loại", 335: "Có biến phân loại · Hồi quy"}\n'
           'for suite, label in suites.items():\n'
           '    rows = datasets[datasets.suite.eq(suite)]\n'
           '    display(Markdown(f"### {label} · {len(rows)} thiết lập"))\n'
           '    display(rows[["paper_name", "processed_dataset_id", "task", "rows", "features"]].reset_index(drop=True))'),
        md("## Không gian tìm kiếm\n\nBảng 1–8 ghi phân phối lấy mẫu hoặc giá trị cố định. "
           "Cột mặc định để trống ở các bảng cây khi paper không in riêng. Riêng SAINT, bảng in "
           "mặc định batch 512 và tập random [128, 256]; cả hai được giữ nguyên."),
        py('hpo = pd.read_csv(root / "evidence/appendix_hpo.csv", keep_default_na=False)\n'
           'for number, rows in hpo.groupby("table"):\n'
           '    display(Markdown(f"### Bảng {number} · {rows.model.iloc[0]}"))\n'
           '    display(rows[["parameter", "distribution", "default"]].reset_index(drop=True))'),
        md(f"[Phụ lục A.1]({SUPPLEMENT}#page=15) · [Phụ lục A.3]({SUPPLEMENT}#page=22)"),
    ]


def build(key):
    path = ROOT / "notebooks" / (FILES[key] + ".ipynb")
    if key.isdecimal():
        number = int(key)
        builder = (time_cells if number in (6, 7) else size_cells if number < 12 else
                   raw_cells if number <= 17 else ablation_cells if number <= 24 else bo_cells)
        cells = builder(number)
    else:
        cells = a4_cells() if key == "a4" else catalog_cells()
    metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    notebook = nbf.v4.new_notebook(cells=cells, metadata=metadata)
    NotebookClient(notebook, kernel_name="python3", timeout=600).execute(cwd=str(path.parent))
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("notebook", choices=FILES)
    build(parser.parse_args().notebook)
