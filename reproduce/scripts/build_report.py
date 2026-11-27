"""One concise main report;all numbers come from the frozen curated exports."""

from pathlib import Path
import hashlib
import json
import re
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from plots import BASE, MODELS, NAMES, means

INK = colors.HexColor("#172e42")
MUTED = colors.HexColor("#506475")
BLUE = colors.HexColor("#245ca6")
WIDTH = A4[0] - 88
PAPER = "https://papers.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html"
AUTHOR = "https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d"


def pretty(text):
    """Space prose without altering URLs,figure suffixes,units or scientific notation."""
    text = re.sub(r"SkorchV2", "Skorch V2", str(text), flags=re.I)
    token = re.compile(
        r"(<[^>]*>|\d+(?:\.\d+)?e[+-]\d+\b|\b[0-9a-f]{8,}\b|\d+(?:[.,]\d+)?K\b|\bFigure\s*\d+[ab]?\b|\bA\.\d+(?:\.\d+)*|\bV\d+\b|\bCS\d+\b)",
        re.I,
    )
    result = ""
    for piece in token.split(text):
        if not piece:
            continue
        if token.fullmatch(piece):
            piece = re.sub(r"(Figure)\s*(\d)", r"\1 \2", piece, flags=re.I)
        else:
            piece = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", piece)
            piece = re.sub(r"[,;](?=[^\W\d_])", lambda m: m[0] + " ", piece)
            piece = re.sub(r"(?<=[^\W\d_]),(?=\d)|;(?=\S)", lambda m: m[0] + " ", piece)
            piece = re.sub(r"(?<=[^\W\d_])\.(?=[A-Z\d])", ". ", piece)
        if result and result[-1].isalpha() and piece[0].isdigit():
            result += " "
        result += piece
    return result


def read(name):
    return json.loads((BASE / name).read_bytes())


def main():
    font = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, file in [("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font / file)))
    body = ParagraphStyle(
        "body", fontName="Body", fontSize=9.5, leading=14, textColor=INK, spaceAfter=8
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=8.3, leading=12, textColor=MUTED
    )
    title = ParagraphStyle(
        "title", parent=body, fontName="Bold", fontSize=20, leading=25, spaceAfter=14
    )
    subtitle = ParagraphStyle(
        "subtitle",
        parent=body,
        fontName="Bold",
        fontSize=11.5,
        leading=17,
        spaceBefore=8,
    )
    cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=12, spaceAfter=0)
    cov = pd.read_csv(BASE / "results/coverage.csv").set_index("key")
    scores = pd.read_csv(BASE / "results/balanced_scores.csv")
    story = []

    def p(text, style=body):
        story.append(Paragraph(pretty(text), style))

    def heading(text):
        p(text, subtitle)

    def page(number, name, kicker):
        if number > 1:
            story.append(PageBreak())
        p(kicker.upper(), small)
        p(name, title)

    def table(rows, fractions):
        values = [
            [Paragraph(escape(pretty(value)), cell) for value in row] for row in rows
        ]
        item = Table(values, colWidths=[WIDTH * f for f in fractions], hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#bdc9d5")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e0e6ec")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([item, Spacer(1, 10)])

    def image(path, max_height=240):
        from PIL import Image as PILImage

        with PILImage.open(path) as pic:
            width, height = pic.size
        scale = min(WIDTH / width, max_height / height)
        story.extend(
            [
                Image(str(path), width=width * scale, height=height * scale),
                Spacer(1, 10),
            ]
        )

    def fmt(value):
        return f"{int(value):,}".replace(",", ".")

    def percent(value):
        return f"{value:.2f}%".replace(".", ",")

    page(
        1,
        "Vì sao mô hình cây vẫn mạnh trên dữ liệu bảng?",
        "Báo cáo paper và kết quả tái hiện · CS2229 · 28/11/2026",
    )
    p(
        "<b>Paper:</b> Grinsztajn, Oyallon và Varoquaux, NeurIPS 2022. Với dữ liệu bảng cỡ vừa, "
        "mô hình cây tiếp tục cạnh tranh mạnh sau khi tính đến tìm siêu tham số. Nghiên cứu xem xét ba đặc tính: "
        "biến ít thông tin, hướng trục và hàm mục tiêu không trơn. [P]"
    )
    image(BASE.parent / "figures/figure_01.png", 250)
    p(
        "Figure 1 được <b>tính lại từ kết quả từng lượt tác giả công bố</b>, dùng Python đối chiếu R; "
        "không phải các mô hình vừa được nhóm train lại. Các đường dùng chọn cấu hình bằng validation.",
        small,
    )
    table(
        [
            ["Bằng chứng trong báo cáo", "Đã thực hiện", "Cách hiểu"],
            [
                "Phân tích kết quả tác giả",
                "27/40 figure IDs",
                "Chạy lại phân tích và vẽ, không ghi là tự train",
            ],
            [
                "Benchmark tự train V2 + mở rộng",
                "Có output trên 57 thiết lập đã chốt",
                "HPO và các nhóm đủ repeats còn chưa đồng đều",
            ],
            [
                "Ba cơ chế theo tham số công bố",
                "632 ô; 1.556 seed fit",
                "Noise, Rotation và smoothing đủ ô tham chiếu",
            ],
            [
                "Các pilot và đối chứng khác",
                "Giữ trong inventory và CSV",
                "Bằng chứng đại diện, không thay toàn bộ benchmark",
            ],
        ],
        [0.34, 0.28, 0.38],
    )
    p(
        "<b>Chưa full paper.</b> Mỗi tỷ lệ ở các trang sau có mẫu số riêng. Không cộng số job đã giao vào kết quả thành công."
    )

    page(2, "Phương pháp và cách đọc số", "Protocol nguồn và các lựa chọn đã công khai")
    table(
        [
            ["Thành phần", "Paper / code gốc", "Bản thực hiện"],
            [
                "Dữ liệu",
                "45 dataset; medium khoảng10K, nghiên cứu thêm cap50K [P]",
                "57 thiết lập OpenML được pin theo ID/version/checksum; các biến thể numeric/categorical đếm riêng",
            ],
            [
                "Mô hình",
                "RF, GBT/XGB và MLP, ResNet, FT-Transformer, SAINT; có nhánh HGBT",
                "Dùng model/train/evaluate gốc; HGBT được báo riêng ở categorical",
            ],
            [
                "Tiền xử lý",
                "Pipeline generate_dataset, Gaussian hóa cho neural, encoding theo model",
                "Giữ pipeline/transform gốc; không tạo kết quả dự đoán ở frontend",
            ],
            [
                "HPO",
                "Random search, default đầu tiên; chọn bằng validation",
                "HPO tự chạy giữ riêng; ba cơ chế dùng cấu hình công bố đã chọn bằng validation",
            ],
            [
                "Hai loại lặp",
                "15 hoán vị thứ tự HPO khác với các lần train",
                "Mean một ô chỉ có khi đủ original auto seeds; không gọi15hoán vị là15lần train",
            ],
            [
                "Metric",
                "Accuracy hoặc R²; tổng hợp theo chuẩn hóa nguồn",
                "Accuracy thô ở ablation; bảng cân bằng dùng score chuẩn hóa, không trộn hai đơn vị",
            ],
        ],
        [0.2, 0.38, 0.42],
    )
    heading("Đọc đúng nguồn và mẫu số")
    p(
        "Figure 1 nguồn hiện dùng nhãn 16/19 thiết lập numeric và7/17categorical. Catalog tự chạy đã chốt dùng15/18 "
        "numeric và7/17categorical=57. HELOC và abalone numeric được giữ ngoài mẫu số57. Báo cáo không đồng nhất hai phạm vi này."
    )
    p(
        "Ngân sách tham chiếu trong bảng độ phủ là400cấu hình ngoài default/model/thiết lập. Đây là đơn vị theo sổ chạy "
        "của nhóm, không phải một tỷ lệ hoàn thành mọi hình của paper. Cap50K có thể dùng ít hơn50K hàng."
    )
    p(
        "231 file tác giả và18 worker lõi được kiểm hash. Tham số, seed, loss, optimizer, batch và epochs của từng lượt "
        "được giữ. Code nguồn và cấu hình dùng để đọc lại một ví dụ được đóng gói trong training/ và configs/.",
        small,
    )

    page(
        3,
        "Benchmark tự train: đã chạy tới đâu?",
        "Cấu hình đủ original repeats · snapshot02:00",
    )
    labels = [
        ("settings_numeric", "Numeric settings có output"),
        ("settings_categorical", "Categorical settings có output"),
        ("default_numeric_trees", "Default Numeric: RF/GBT/XGB"),
        ("default_categorical_trees", "Default Categorical: cây"),
        ("numeric_hpo", "Numeric HPO: RF/GBT/XGB"),
        ("categorical_hpo", "Categorical HPO: RF/GBT/XGB"),
        ("hgbt", "Categorical HPO: HGBT"),
        ("numeric_neural_hpo", "Numeric HPO: bốn neural"),
        ("categorical_neural_hpo", "Categorical HPO: bốn neural"),
        ("saint", "SAINT defaults"),
        ("50k", "Defaults cap50K"),
    ]
    table(
        [["Hạng mục", "Đã kiểm chứng / tham chiếu", "% theo phạm vi"]]
        + [
            [
                label,
                f"{fmt(cov.loc[key, 'complete'])}/{fmt(cov.loc[key, 'reference'])}",
                percent(cov.loc[key, "percent"]),
            ]
            for key, label in labels
        ],
        [0.49, 0.32, 0.19],
    )
    p(
        "HPO cap50K: <b>38.006 nhóm đủ repeats</b>; chưa gán một % full-paper cho số này. Bốn task pilot đủ50K "
        "hàng có82/96cấu hình. Có output tại một setting không đồng nghĩa mọi model đều đủ ngân sách."
    )
    heading("Đợt hoàn thiện còn tiếp tục")
    p(
        "Đã giao1.489seed fits ở851nhóm thiếu. Tại snapshot: <b>11success,5failed,9timeout</b>; còn1.464fit chờ/đang chạy. "
        "Ghép11success với sibling seeds hợp lệ bổ sung10nhóm Numeric, không phải11nhóm."
    )
    p(
        "Không thay tiêu chuẩn để ép100%. Năm lỗi FT-Transformer/Bioresponse cap50K là GPU OOM; "
        "giữ đầy đủ thất bại và timeout. Lượt tổng hợp không cancel, thử lại hoặc gửi job mới.",
        small,
    )

    page(
        4,
        "So sánh cân bằng từ kết quả tự train",
        "24/57thiết lập · default đầu tiên ·15hoán vị · chốt25/11",
    )
    groups = [
        ("numeric", "classification"),
        ("numeric", "regression"),
        ("categorical", "classification"),
        ("categorical", "regression"),
    ]
    table_rows = [
        [
            "Mô hình",
            "Numeric phân loại",
            "Numeric hồi quy",
            "Categorical phân loại",
            "Categorical hồi quy",
        ],
        ["Thiết lập / catalog", "4/15", "3/18", "5/7", "12/17"],
        ["HPO ngoài default", "2", "3", "29", "22"],
    ]
    names = [
        ("RF", "RandomForest"),
        ("GBT", "GradientBoostingTree"),
        ("XGBoost", "XGBoost"),
        ("HGBT", "HistGradientBoostingTree"),
        ("MLP", "MLP"),
        ("ResNet", "Resnet"),
        ("FT-Transformer", "FT Transformer"),
        ("SAINT", "SAINT"),
    ]
    for label, model in names:
        values = [label]
        for data_kind, task_type in groups:
            chosen = scores.loc[
                (scores.data_kind == data_kind)
                & (scores.task_type == task_type)
                & (scores.model == model)
            ]
            values.append(
                "Không áp dụng"
                if chosen.empty
                else f"{chosen.normalized_test.iloc[0]:.3f}".replace(".", ",")
            )
        table_rows.append(values)
    table(table_rows, [0.24, 0.19, 0.19, 0.19, 0.19])
    p(
        "Đây là <b>test score đã chuẩn hóa</b>, không phải accuracy. Chỉ so mô hình trong cùng cột; ngân sách giữa các cột "
        "khác nhau. HGBT không có ở hai cột numeric."
    )
    heading("Có thể kết luận gì?")
    p(
        "Mô hình cây đạt điểm cao nhất trong cả bốn cột của tập con này. Kết quả phù hợp về hướng với benchmark tác giả, "
        "nhưng chỉ dựa trên24thiết lập và các ngân sách được ghi rõ. Chưa chứng minh neural luôn thua hoặc khác biệt có ý nghĩa thống kê."
    )
    p(
        "Bảng này giữ nguyên từ mốc25/11. Các fit cơ chế và10nhóm HPO vừa bổ sung không tự động thay bảng hay đường cong cân bằng.",
        small,
    )

    page(
        5,
        "Noise features: biến ít thông tin",
        "Figure4b/21 · tự train ·14dataset ·4model ·168ô/420fit",
    )
    p(
        "Giữ nhãn, thêm các biến Gaussian theo nhánh gốc. Multiplier1/1,5/2 tương ứng thêm0%/50%/100% số biến ban đầu. "
        "Đủ cả ba mức trên14dataset cho RF,GBT,ResNet vàFT-Transformer."
    )
    image(BASE / "figures/noise_features.png", 255)
    p(
        "Mỗi điểm là trung bình đều theo dataset, sau khi trung bình các seed trong một ô. "
        "Đây là accuracy thô tự chạy; không phải đường chuẩn hóa của Figure4b nguồn.",
        small,
    )
    noise = means("noise_features", [[1], [1.5], [2]])
    table(
        [["Mô hình", "Không thêm (%)", "Thêm100% (%)", "Chênh lệch điểm %"]]
        + [
            [
                NAMES[m],
                f"{noise[m][0]:.2f}",
                f"{noise[m][2]:.2f}",
                f"{noise[m][2] - noise[m][0]:+.2f}",
            ]
            for m in MODELS
        ],
        [0.28, 0.24, 0.24, 0.24],
    )
    p(
        "Các họ mô hình phản ứng khác nhau khi thêm biến không mang tín hiệu. Xu hướng tổng hợp không áp dụng cho mọi dataset; "
        "người đọc có thể kiểm từng ô trong results/mechanism_cells.csv."
    )
    p(
        "Notebook: figure_04_noise_features_self_trained.ipynb. Cấu hình từng ô và số seed có trong configs/published_conditions.json.",
        small,
    )

    page(6, "Rotation: hướng trục có ý nghĩa", "Figure5/23–24 · tự train ·224ô/560fit")
    p(
        "So cặp có xoay/không xoay ở hai điều kiện: giữ toàn bộ biến, hoặc loại trước50%biến. "
        "Mỗi cặp giữ cấu hình đã công bố và số seed tương ứng. Trong config, deactivated=True nghĩa là <b>không xoay</b>."
    )
    image(BASE / "figures/rotation.png", 265)
    p(
        "Cột là accuracy có xoay trừ accuracy không xoay, trung bình đều14dataset; số âm biểu thị giảm trên phép so này. "
        "Không dùng màu xanh/đỏ để ngụ ý mô hình nào luôn tốt hơn.",
        small,
    )
    rotation = means("rotation", [[0, True], [0, False], [0.5, True], [0.5, False]])
    table(
        [["Mô hình", "Giữ mọi biến: delta điểm %", "Loại50%: delta điểm %"]]
        + [
            [
                NAMES[m],
                f"{rotation[m][1] - rotation[m][0]:+.2f}",
                f"{rotation[m][3] - rotation[m][2]:+.2f}",
            ]
            for m in MODELS
        ],
        [0.3, 0.35, 0.35],
    )
    p(
        "Độ nhạy với phép xoay hỗ trợ phân tích về hướng trục của dữ liệu bảng. Đây là bằng chứng thực nghiệm trong các ô đã chọn, "
        "không phải một định lý về mọi phân phối dữ liệu."
    )
    p(
        "Notebook: figure_05_rotation_self_trained.ipynb. Tám fit còn thiếu ở PDF27/11 đã có terminal output và audit trước khi được cộng vào bản này.",
        small,
    )

    page(
        7,
        "Target smoothing: mục tiêu không trơn",
        "Figure2/18 · tự train ·15dataset ·240ô/576fit",
    )
    p(
        "Nhánh gốc giữ5biến quan trọng, làm trơn <b>nhãn train</b> bằng trung bình có trọng số Gaussian rồi ngưỡng hóa ở0,5 "
        "cho phân loại. Validation/test giữ nhãn gốc. Bốn hệ số covariance:0;0,05;0,1;0,25."
    )
    image(BASE / "figures/target_smoothing.png", 255)
    p(
        "Cùng cách lấy trung bình theo dataset như trang Noise. Hệ số0 là không làm trơn; không nội suy thêm mức chưa chạy.",
        small,
    )
    smooth = means("target_smoothing", [[a, "robust"] for a in [0, 0.05, 0.1, 0.25]])
    table(
        [["Mô hình", "Hệ số0 (%)", "Hệ số0,25 (%)", "Chênh lệch điểm %"]]
        + [
            [
                NAMES[m],
                f"{smooth[m][0]:.2f}",
                f"{smooth[m][3]:.2f}",
                f"{smooth[m][3] - smooth[m][0]:+.2f}",
            ]
            for m in MODELS
        ],
        [0.28, 0.24, 0.24, 0.24],
    )
    p(
        "Làm trơn thay đổi khoảng cách giữa các họ mô hình. Đây là can thiệp lên mục tiêu train để khảo sát inductive bias, "
        "không phải kỹ thuật tăng accuracy test mặc định."
    )
    p(
        "Giữ nguyên hàm remove_high_frequency_from_train và nhánh robust→classic covariance fallback của executable. "
        "Không tự sửa công thức/fallback để làm đẹp kết quả. Notebook: figure_02_target_smoothing_self_trained.ipynb.",
        small,
    )

    page(
        8,
        "Những phần khác đã thực hiện",
        "Không đánh đồng pilot,phân tích từ CSV và thực nghiệm đầy đủ",
    )
    table(
        [
            ["Phần", "Bằng chứng giữ lại", "Giới hạn"],
            [
                "A.1/A.3",
                "57 thiết lập;8bảng;70mục HPO",
                "Catalog đã đối chiếu, không phải đủ mọi lần train",
            ],
            [
                "Figure1–18;20–24;33–34;39–40",
                "27notebook phân tích từ dữ liệu tác giả",
                "Tái chạy phân tích/chọn validation;không tự train toàn bộ",
            ],
            [
                "Figure3/4a: giữ/loại biến",
                "251nhóm ghi nhận trong union;pilot nhiều mức",
                "Không gán % full figure khi thiếu mẫu số đầy đủ",
            ],
            [
                "Một cây quyết định",
                "4/4pilot,mỗi nhóm dữ liệu một case",
                "Một cấu hình mặc định,không full HPO",
            ],
            [
                "A.4 hồi quy",
                "34hệ số đối chiếu Python–R",
                "Phân tích trên kết quả công bố",
            ],
            [
                "Figure19: biên quyết định",
                "RF/MLP hai biến,Electricity,có pilot tự train",
                "Chưa xác minh đúng hai biến/split của hình gốc",
            ],
            [
                "A.5 lọc dữ liệu",
                "8/8panel pilot,bảo toàn số ứng viên",
                "Một số số đếm khác PDF;không gọi tái lập chính xác",
            ],
            [
                "A.6 benchmark khác",
                "Đối chiếu ID ba nhóm",
                "Version/log lọc lịch sử chưa đủ giải thích mọi chênh lệch",
            ],
            [
                "A.7 Bayesian/Random",
                "16/16lượt pilot GBT,trên2task",
                "Phạm vi nhỏ;chưa full benchmark",
            ],
            [
                "A.9 HPO importance",
                "RFRegressor/LinearRegression trên367configGBT",
                "Phân tích đại diện trên pol",
            ],
            [
                "A.9 giới hạn/tune số cây",
                "Có pilot cặp đối chứng",
                "Không tái hiện toàn bộ grid/hình35–38",
            ],
        ],
        [0.25, 0.36, 0.39],
    )
    p(
        "Inventory40figure IDs và trạng thái nguồn có trong results/figure_inventory.csv. Hình19,25–32,35–38 chưa có "
        "tái lập phân tích chính xác đầy đủ; các pilot tương ứng vẫn được giữ và báo đúng phạm vi."
    )
    p(
        "Union cơ chế có2.172fit thành công,899nhóm ghi nhận, gồm632ô của ba grid tham chiếu và các nhóm removal/smoothing bổ sung. "
        "Các nhánh chồng nguồn không được cộng lần hai vào tổng HPO. Cấu hình,source record và seed của từng nhóm được xuất kèm.",
        small,
    )

    page(
        9,
        "Kiểm chứng, giới hạn và cách dùng",
        "Bản nộp gọn · dữ liệu thật · chưa phát triển demo mới",
    )
    table(
        [
            ["Điểm cần công khai", "Ý nghĩa"],
            [
                "Đầu vào lịch sử",
                "Đã pin OpenML ID/version/checksum;chưa chứng minh mọi byte bằng đầu vào lịch sử của tác giả",
            ],
            [
                "Sampler HPO tự chạy",
                "Các mẫu registry NumPy không khôi phục nguyên thứ tự draw W&B lịch sử;không ghi là replay nguyên từng lượt",
            ],
            [
                "Điểm train SkorchV2",
                "Đo sau best-checkpoint reload;executable gốc ghi trước reload.Validation/test giữ riêng",
            ],
            [
                "Phạm vi đang thiếu",
                "Neural HPO, large-cap và nhiều phân tích chưa full. Một config/ô cơ chế không thay toàn bộ HPO/shuffle",
            ],
        ],
        [0.3, 0.7],
    )
    p(
        "Nguồn nghiên cứu: 50 kiểm thử adapter, lint ứng dụng và mypy14 file đạt; full pytest bị chặn do WinError1455 "
        "khi tải CUDA DLL và không được ghi là pass. Trong bộ nộp có thêm kiểm tra offline về hash,899dòng ô,632ô tham chiếu "
        "và ba bộ numeric đại diện ở1e-12. <b>Ba mẫu numeric không thay audit lại toàn bộ lịch sử.</b>"
    )
    heading("Chạy lại phần trình bày, không train")
    p(
        "Cài requirements.txt của reproduce/,rồi chạy scripts/check_results.py và scripts/build_report.py. "
        "Ba notebook tự train đã được thực thi:Run All chỉ đọc output và vẽ.27notebook phân tích tác giả ở thư mục notebooks/ gốc."
    )
    p(
        "Nếu cần thực sự train lại một ví dụ:training/ có source frozen;README hướng dẫn môi trườngPython3.8,clone upstream "
        "đúng commit và lệnh train_sample.py --run.Bước này là tùy chọn thủ công,<b>chưa chạy trong lượt đóng gói</b>."
    )
    heading("Kết luận có căn cứ")
    p(
        "Các kết quả tự chạy đã có hỗ trợ việc mô hình cây là baseline mạnh ở phạm vi tabular đang xét. Ba can thiệp cho thấy "
        "vai trò của biến ít thông tin,hướng trục và độ không trơn của mục tiêu. Kết luận được giới hạn bởi dữ liệu,cấu hình "
        "và ngân sách đã báo;không khẳng định thắng trên mọi bài toán."
    )
    p(
        f'<b>Nguồn [P].</b> <link href="{PAPER}" color="#245ca6">Paper NeurIPS2022</link>; '
        f'<link href="{AUTHOR}" color="#245ca6">repo tác giả@9d54cf53</link>. '
        "Nguồn điểm và CSV:configs/source_versions.json,results/source_audits.json. "
        "Manifest SHA-256 và README giải thích toàn bộ cây thư mục. PDF chính chỉ có một bản trong reproduce/report/.",
        small,
    )

    target = BASE / "report/bao-cao-tai-hien.pdf"

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5dde5"))
        canvas.line(44, 42, A4[0] - 44, 42)
        canvas.setFont("Body", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(
            44,
            28,
            "CS2229 | V2 + mở rộng | Dữ liệu đã audit tại snapshot 02:00 ngày 28/11/2026",
        )
        canvas.drawRightString(A4[0] - 44, 28, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=38,
        bottomMargin=56,
        title="CS2229 · Paper và thực nghiệm tái hiện",
        author="CS2229 project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(
        json.dumps(
            dict(
                pdf=str(target.relative_to(BASE)),
                sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                training_refits=0,
            )
        )
    )


if __name__ == "__main__":
    main()
