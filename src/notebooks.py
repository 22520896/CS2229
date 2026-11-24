"""Build and execute each independent notebook with its saved visual output."""

from argparse import ArgumentParser

from nbclient import NotebookClient
import nbformat as nbf

from .author_data import ROOT


FILES = (
    "figure_01_benchmark", "figure_02_smoothing", "figure_03_feature_importance",
    "figure_04_uninformative_features", "figure_05_rotation",
)
TITLES = (
    "Figure 1 · Cây và neural theo ngân sách HPO",
    "Figure 2 · Làm trơn mục tiêu huấn luyện",
    "Figure 3 · Tách hai nhóm đặc trưng",
    "Figure 4 · Loại hoặc thêm đặc trưng ít thông tin",
    "Figure 5 · Xoay không gian đặc trưng",
)
STEPS = (
    "Bốn nhóm benchmark medium. Dùng default ở hạng 0, sau đó chọn cấu hình theo điểm validation tốt nhất sau mỗi lượt. "
    "Điểm test chuẩn hóa theo từng bộ dữ liệu, bằng phân vị 10% cho phân loại và trung vị cho hồi quy. "
    "Phần âm của điểm test hồi quy được chặn ở 0 như script. Dải màu là min–max qua 15 lần xáo thứ tự.",
    "So sánh bốn mức hệ số hiệp phương sai ở figure chính. Chuẩn hóa theo phân vị 10% trên từng bộ dữ liệu. "
    "Dùng default-first, chọn bằng validation ở hạng 60 và tổng hợp qua 15 lần xáo. "
    "Khoảng trải của boxplot đến từ các lần xáo thứ tự tìm kiếm.",
    "Xếp hạng đặc trưng bằng Random Forest của tác giả. Với GBT, tách hai nhánh dùng nhóm giữ lại và chỉ dùng nhóm bị loại. "
    "Điểm được chuẩn hóa để 0,5 accuracy gốc ứng với 0. Ở hạng 20, lấy trung bình qua 30 lần xáo, "
    "sau đó lấy trung bình và phân vị 10%–90% qua các bộ dữ liệu. Nhánh dùng nhóm bị loại ở mức 0% không có điểm.",
    "Figure 4a đánh giá ở hạng 30, dải min–max từ 30 lần xáo. "
    "Figure 4b dùng script công bố ở hạng 20, dải phân vị 20%–80% từ 15 lần xáo. "
    "Bảng cuối notebook có cả phương án 30 lần xáo và min–max ghi trong chú thích paper. "
    "Các biến thể dùng chung dữ liệu tác giả và cách chọn dựa trên validation.",
    "Tách từng nhánh giữ toàn bộ hoặc loại 50% đặc trưng trước khi Gaussian hóa và xoay. "
    "So sánh trước và sau xoay ở hạng 20 bằng boxplot qua 30 lần xáo, default-first và chọn theo validation. "
    "Hình lấy hai nhánh từ dữ liệu công bố, gồm cả nhánh mà script vẽ chính có chú thích bật thủ công.",
)
V2 = {
    1: "Đối chứng V2 + mở rộng có 24/57 thiết lập đủ dữ liệu. Ngân sách random chung theo bốn nhóm lần lượt là 2, 3, 29 và 22 ngoài default. Bảng và đường cong dùng bản tổng hợp đã kiểm chứng của dự án.",
    2: "Pilot tự train gồm 8 lượt trên eye_movements, một seed, hai mô hình RF và MLP. Biểu đồ dưới dùng accuracy thô.",
    5: "Đối chứng tự train dùng cấu hình mặc định của 16 bộ dữ liệu số. Có 762/768 fit lõi đã kiểm chứng và 381 cặp đo chênh lệch. Biểu đồ dưới hiển thị trung bình chênh lệch accuracy trên các cặp có đầu ra.",
}
RECIPE = {
    1: ("launch_benchmarks.py", "Chọn nhóm OpenML, model và cấu hình công bố; chia train/validation/test; train; lưu điểm từng split."),
    2: ("launch_xps.py · smooth_target", "Gaussian hóa, chọn 5 đặc trưng bằng RF, làm trơn nhãn train, giữ nhãn validation/test, rồi train và chấm điểm."),
    3: ("launch_xps.py · uninformative_features", "Xếp hạng RF trên train; chọn một tỷ lệ công bố; chạy GBT trên phần giữ lại và trên phần bị loại; chấm điểm từng nhánh."),
    4: ("launch_xps.py · remove_features / add_features", "Tạo hai nhóm đầu vào theo tỷ lệ công bố; loại theo thứ hạng RF hoặc thêm nhiễu Gaussian; train cùng họ mô hình và chấm điểm."),
    5: ("launch_xps.py · random_rotation", "Chọn nhánh bỏ 0% hoặc 50% đặc trưng; Gaussian hóa; bật hoặc tắt xoay; train và so sánh điểm trên cùng protocol."),
}
EXAMPLES = {
    2: ("high_frequencies",),
    3: ("show_uninformative_features",),
    4: ("remove_uninformative_features", "add_uninformative_features"),
    5: ("random_rotation",),
}
START = """from pathlib import Path
import sys

root = next(folder for folder in (Path.cwd(), *Path.cwd().parents)
            if (folder / "src/analysis.py").is_file())
sys.path.insert(0, str(root))
from IPython.display import Image, display
import matplotlib.pyplot as plt
from src.analysis import analyze
from src.author_data import SOURCES, prepare
from src.plots import export, export_project
"""


def create(number):
    source = "benchmark" if number == 1 else "ablations"
    sub = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        sub("# " + TITLES[number - 1] + "\n\n**Nguồn chính** · Kết quả từng lượt do tác giả công bố. "
            "Notebook chạy lại phép phân tích và vẽ hình bằng Python. Lần mở đầu cần Internet để tải và đối chiếu SHA-256."),
        sub("## Cách đọc thí nghiệm\n\n" + STEPS[number - 1]),
        code(START + f'\nsource = "{source}"\ncache = prepare(source)\nprint("SHA-256 nguồn:", SOURCES[source].sha256)'),
        code(f'figure_number = {number}\nscores = analyze(figure_number)\nchart = export(figure_number, scores)\n'
             'plt.close(chart)\ndisplay(Image(filename=str(root / "figures" / f"figure_{figure_number:02d}.png")))'),
    ]
    if number == 4:
        cells += [
            sub("## Khác biệt giữa chú thích paper và script Figure 4b\n\n"
                "Cùng một dữ liệu đầu vào. Bảng dưới đối chiếu biên dưới và biên trên của hai cách tính."),
            code('left, script, caption = scores\n'
                 'comparison = script.merge(caption, on=["model_name", "fraction"], suffixes=("_script", "_caption"))\n'
                 'display(comparison[["model_name", "fraction", "low_script", "high_script", '
                 '"low_caption", "high_caption"]].round(4))'),
        ]
    else:
        cells += [sub("## Một số giá trị đứng sau biểu đồ"),
                  code('display(scores.head(8).round(5))')]
    if number in V2:
        cells += [
            sub("## Đối chứng do nhóm tự train\n\n" + V2[number]),
            code('chart = export_project(figure_number)\nplt.close(chart)\n'
                 'display(Image(filename=str(root / "figures" / f"figure_{figure_number:02d}_v2.png")))'),
        ]
    entry, procedure = RECIPE[number]
    cells += [
        sub("## Quy trình train trong repo tác giả\n\n"
            f"**Minh họa quy trình, không có fit mới ở notebook này.** Điểm vào: `src/launch_benchmarks/{entry}`. "
            f"{procedure}\n\n"
            "Nếu tái chạy độc lập, dùng cấu hình đầy đủ của đúng model, task và điều kiện từ CSV nguồn. "
            "Repo gốc có lệnh chuẩn bị môi trường, launcher và `train_model_on_config`; "
            "khóa phiên bản source trước khi chạy. "),
    ]
    if number in EXAMPLES:
        cells += [
            sub("### Ví dụ cấu hình đã công bố\n\n"
                "Các giá trị dưới đây trích từ dòng đầu tiên của GBT trong từng nhánh, theo thứ tự CSV. "
                "File lịch sử dùng tên bộ dữ liệu và loader `real_data`. Launcher pin hiện tại dùng OpenML task ID và "
                "`openml_no_transform`, nên cần đối chiếu đầu vào khi chạy lại ví dụ."),
            code('import json\nimport pandas as pd\n'
                 'published = json.loads((root / "evidence/published_config_examples.json").read_text(encoding="utf-8"))\n'
                 f'names = {EXAMPLES[number]!r}\n'
                 'display(pd.DataFrame({name: published["examples"][name]["settings"] for name in names}).fillna("—"))'),
        ]
    cells += [
        sub("## Nguồn\n\n"
            "- [Paper NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)\n"
            "- [Repo tác giả, commit đã đối chiếu](https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d)\n"),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    })
    return notebook


def build(number):
    path = ROOT / "notebooks" / (FILES[number - 1] + ".ipynb")
    notebook = create(number)
    NotebookClient(notebook, kernel_name="python3", timeout=600).execute(cwd=str(path.parent))
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=range(1, 6))
    build(parser.parse_args().number)
