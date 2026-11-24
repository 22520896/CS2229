"""One notebook per figure recovered from the published supplement."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

from .appendix_notebooks import BOOT, SUPPLEMENT
from .author_data import ROOT
from .plots import COLORS, FIGURES, NAMES, shared_legend, theme
from .published_figures import CURVE_FIGURES, FLOW_FIGURES, PAGES, load


FILES = {
    19: "figure_19_decision_regions_from_pdf",
    25: "figure_25_filter_medium_numerical_regression",
    26: "figure_26_filter_medium_numerical_classification",
    27: "figure_27_filter_medium_categorical_regression",
    28: "figure_28_filter_medium_categorical_classification",
    29: "figure_29_filter_large_numerical_regression",
    30: "figure_30_filter_large_numerical_classification",
    31: "figure_31_filter_large_categorical_regression",
    32: "figure_32_filter_large_categorical_classification",
    35: "figure_35_estimators_budget_numeric_from_pdf",
    36: "figure_36_estimators_budget_categorical_from_pdf",
    37: "figure_37_estimators_time_numeric_from_pdf",
    38: "figure_38_estimators_time_categorical_from_pdf",
    39: "figure_39_tuned_estimators_numeric",
    40: "figure_40_tuned_estimators_categorical",
}
FLOW_LABELS = {
    "Removing NA": "Bỏ dữ liệu thiếu", "Balancing": "Cân bằng", "Checks": "Kiểm tra",
    "Passed": "Qua kiểm tra", "Used": "Được dùng", "Not used": "Loại",
    "Too easy": "Quá dễ", "Too small": "Quá nhỏ", "High dimensional": "Quá nhiều chiều",
    "Time series": "Chuỗi thời gian", "Too little information": "Ít thông tin",
    "Too many IDs": "Nhiều cột ID", "Redundant": "Trùng lặp", "Artificial": "Nhân tạo",
    "Not Heterogeneous": "Không dị thể", "Deterministic": "Tất định", "Stream": "Dữ liệu luồng",
    "Too many low cardinality": "Nhiều cột ít giá trị", "Too few cat cols": "Ít cột phân loại",
}
TITLES = {
    19: "Ranh giới trên hai đặc trưng của Electricity",
    25: "Lọc dữ liệu medium · số · hồi quy",
    26: "Lọc dữ liệu medium · số · phân loại",
    27: "Lọc dữ liệu medium · có biến phân loại · hồi quy",
    28: "Lọc dữ liệu medium · có biến phân loại · phân loại",
    29: "Lọc dữ liệu large · số · hồi quy",
    30: "Lọc dữ liệu large · số · phân loại",
    31: "Lọc dữ liệu large · có biến phân loại · hồi quy",
    32: "Lọc dữ liệu large · có biến phân loại · phân loại",
    35: "Tăng trần số cây · đặc trưng số",
    36: "Tăng trần số cây · có biến phân loại",
    37: "So thời gian khi tăng trần số cây · đặc trưng số",
    38: "So thời gian khi tăng trần số cây · có biến phân loại",
    39: "Cố định hay tune số cây · đặc trưng số",
    40: "Cố định hay tune số cây · có biến phân loại",
}


def load_figure(number):
    return load()["figures"][str(number)]


def plot_decision(record):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained", sharex=True, sharey=True)
    colors = ListedColormap(["#467da8", "#e8e7e3", "#bb6464"])
    for (model, panel), ax in zip(record["panels"].items(), axes):
        ax.imshow(panel["color_cells"], cmap=colors, vmin=-1, vmax=1, origin="upper",
                  extent=(0, 1, 0, 1), interpolation="nearest", aspect="auto")
        ax.set_title(f'{model} · accuracy paper {panel["published_accuracy_percent"]}%')
        ax.set_xlabel("Đặc trưng ngang · vị trí trong vùng phóng")
    axes[0].set_ylabel("Đặc trưng dọc · vị trí trong vùng phóng")
    return fig


def render_flow(record):
    nodes = record["nodes"]
    y_min = min(n["box"][1] for n in nodes)
    y_max = max(n["box"][3] for n in nodes)
    height = max(2.8, (y_max - y_min) / 33)
    fig, ax = plt.subplots(figsize=(11.5, height), layout="constrained")
    for link in record["links"]:
        upper, lower = link["upper"], link["lower"]
        path = Path([*upper, *lower, upper[0]],
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
                     Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY])
        color = "#89b7a7" if link["target"] == "Used" else "#9aaeb8"
        ax.add_patch(PathPatch(path, facecolor=color, alpha=.54, edgecolor="none"))
    for node in nodes:
        x0, y0, x1, y1 = node["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fc="#17695f" if node["kept"] else "#a64c4a", ec="none"))
        ax.text(x1 + 2.2, (y0 + y1) / 2, FLOW_LABELS.get(node["name"], node["name"]),
                va="center", fontsize=7.3, color="#203441",
                bbox={"facecolor": "white", "alpha": .82, "pad": .6, "edgecolor": "none"})
    ax.set_xlim(min(n["box"][0] for n in nodes) - 8, max(n["box"][2] for n in nodes) + 54)
    ax.set_ylim(y_max + 8, y_min - 8)
    ax.axis("off")
    return fig


def series_style(name):
    base, _, suffix = name.partition("_")
    return (COLORS[base], "--" if suffix in ("40K", "tune") else "-",
            NAMES[base] + (f" · {suffix}" if suffix else ""))


def plot_curves(record):
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.5))
    for (task, panel), ax in zip(record["panels"].items(), axes):
        limits = np.asarray(panel["axes"]["bounds"])
        for name, traces in panel["series"].items():
            color, linestyle, label = series_style(name)
            for band in traces.get("band", []):
                ax.add_patch(Polygon(band, closed=True, facecolor=color, alpha=.11, edgecolor="none"))
            for default in traces.get("default", []):
                x, y = np.asarray(default).T
                ax.plot(x, y, color=color, linestyle=":", linewidth=.9, alpha=.52)
            for index, trace in enumerate(traces["mean"]):
                x, y = np.asarray(trace).T
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.5,
                        label=label if index == 0 else None)
        ax.set(xscale="log", xlim=sorted(limits[:, 0]), ylim=sorted(limits[:, 1]),
               title="Phân loại" if task == "classification" else "Hồi quy",
               xlabel="Thời gian tìm kiếm (giây)" if record["unit"] == "seconds" else "Số lượt tìm kiếm")
        ax.grid(axis="y")
    axes[0].set_ylabel("Điểm test chuẩn hóa")
    shared_legend(fig, axes, bottom=.30)
    return fig


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


def export(number, record):
    theme()
    plot = plot_decision if number == 19 else render_flow if number in FLOW_FIGURES else plot_curves
    figure = plot(record)
    destination = FIGURES / f"figure_{number:02d}.png"
    figure.savefig(destination, dpi=175, bbox_inches="tight", pad_inches=.13)
    plt.close(figure)
    return destination


def summary(record):
    if record["method"] == "pdf_raster_hue":
        return pd.DataFrame([
            {"mô hình": model, "accuracy ghi trên paper (%)": panel["published_accuracy_percent"],
             **{label: int(np.count_nonzero(np.asarray(panel["color_cells"]) == index))
                for index, label in ((-1, "ô xanh"), (0, "ô trung tính"), (1, "ô đỏ"))}}
            for model, panel in record["panels"].items()])
    if record["method"] == "pdf_vector_flow":
        total = record["audit"]["start_width_pt"]
        return pd.DataFrame([
            {"từ": FLOW_LABELS.get(link["source"], link["source"]),
             "đến": FLOW_LABELS.get(link["target"], link["target"]),
             "độ rộng (pt)": link["width_pt"], "so độ rộng đầu vào (%)": 100 * link["width_pt"] / total}
             for link in record["links"]
        ]).round(4)
    rows = []
    for task, panel in record["panels"].items():
        for model, paths in panel["series"].items():
            line = np.concatenate([np.asarray(path) for path in paths["mean"]])
            rows.append({"bài toán": "Phân loại" if task == "classification" else "Hồi quy",
                         "đường cong": model, "số đỉnh vector": len(line),
                         "x cuối trên đường": line[-1, 0], "y cuối trên đường": line[-1, 1]})
    return pd.DataFrame(rows).round(4)


def create(number):
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    if number in (39, 40):
        return create_tuned(number, markdown, code)
    if number == 19:
        description = ("Số hóa hai vùng màu nền từ ảnh nhúng trong PDF bằng lưới 260 × 240 và phân nhóm sắc đỏ, xanh, trung tính. "
                       "Trục biểu thị vị trí trong ô phóng. Màu dùng để đọc hình và tránh suy ra xác suất lớp. "
                       "Accuracy 85% và 80% đọc từ chú thích tác giả. Lượt này không fit lại RF hay MLP.")
    elif number in FLOW_FIGURES:
        description = ("Trích nhãn nút, đường Bézier và độ rộng từng nhánh từ PDF. "
                       "Tỷ lệ ở bảng lấy độ rộng nhánh chia độ rộng đầu vào. Tỷ lệ mô tả sơ đồ; "
                       "metadata từng bộ dữ liệu và số đếm tuyệt đối chưa được xác minh.")
        if number == 28:
            description += (" Figure 28 có chênh lệch khoảng 1,13% tổng độ rộng giữa đầu và cuối. "
                            "Bản dựng giữ nguyên phần chênh trong hình công bố.")
    else:
        description = ("Hiệu chuẩn hai trục từ vạch lưới và nhãn trong PDF, rồi lấy các đỉnh vector của đường trung bình, "
                       "dải và đường mặc định nếu có. Các giá trị là tọa độ số hóa có giới hạn độ chính xác. "
                       "Nhật ký fit từng lượt và quá trình chọn cấu hình chưa được tái chạy cho hình này.")
        if number <= 38:
            description += (" Giữ nhãn 10K/40K theo hình. Phụ lục mô tả nhánh thấp dùng 1K và patience 20 khi tìm kiếm, "
                            "với default 10K. Nhánh cao dùng 40K và patience 100.")
    intro = markdown(f"# Figure {number} · {TITLES[number]}\n\n**Số hóa hình công bố.** {description}")
    cells = [intro, code(BOOT + "\nfrom src.published_notebooks import export, load_figure, summary\n"
                         f"record = load_figure({number})\n"
                         f"display(Image(filename=str(export({number}, record))))\n"
                         "display(summary(record))")]
    if number in FLOW_FIGURES:
        cells.append(code('audit = record["audit"]\n'
                          'display(pd.DataFrame({\n'
                          '  "được dùng (%)": [100 * audit["used_share"]],\n'
                          '  "bị loại (%)": [100 * audit["not_used_share"]],\n'
                          '  "chênh lệch đầu–cuối (pt)": [audit["terminal_width_residual_pt"]],\n'
                          '}).round(5))'))
    cells.append(markdown(f"[Nguồn PDF NeurIPS 2022 · trang {PAGES[number]}]({SUPPLEMENT}#page={PAGES[number]})\n\n"
                          "Đầu vào số hóa được lưu trong `evidence/published_figure_geometry.json.gz`."))
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    })


def create_tuned(number, markdown, code):
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
