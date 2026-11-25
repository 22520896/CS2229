"""Build the Vietnamese submission PDF from the notebook exports."""

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .author_data import ROOT
from .notebooks import STEPS, TITLES


INK = colors.HexColor("#17343d")
SUBDUED = colors.HexColor("#46646c")
ACCENT = colors.HexColor("#16776d")
MUTED = colors.HexColor("#e5efec")
WIDTH = 481
PAPER = "https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf"
REPO = "https://github.com/LeoGrin/tabular-benchmark/tree/9d54cf53d9fd3159e367e70a00005f4fcbf2c79d"
LFS = "https://github.com/LeoGrin/tabular-benchmark/blob/f3fe60384ba7852e3c0b755c20cce478857d9878/analyses/results/random_search_xps.csv"
FINDINGS = (
    "Đường cong cho thấy lợi thế của nhiều họ mô hình cây trên benchmark medium. "
    "Thứ hạng từng mô hình phụ thuộc nhóm dữ liệu và ngân sách tìm kiếm.",
    "Thay đổi độ trơn của mục tiêu làm thay đổi khoảng cách giữa các mô hình. "
    "Pilot một task của nhóm chỉ kiểm tra quy trình ở quy mô nhỏ.",
    "GBT giữ phần lớn điểm khi loại dần các đặc trưng ít quan trọng. "
    "Hai đường có thể cắt nhau ở tỷ lệ loại rất cao, khi nhóm bị loại gồm phần lớn đặc trưng. "
    "Dải màu tóm tắt biến thiên giữa các bộ dữ liệu.",
    "Loại và thêm đặc trưng ít thông tin thay đổi điểm theo từng họ mô hình. "
    "Dùng hai trục tỷ lệ riêng để phân biệt phép loại và phép thêm.",
    "Xoay trực giao giữ khoảng cách Euclid nhưng đổi hướng các trục. "
    "Các mô hình dựa vào phép chia theo trục có thể nhạy với phép biến đổi này.",
)


def fmt(number):
    return f"{number:,}".replace(",", ".")


def styles():
    import matplotlib

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for family, filename in (("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(family, str(fonts / filename)))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold")
    return {
        "cover": ParagraphStyle("cover", fontName="Bold", fontSize=32, leading=37, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=18),
        "head": ParagraphStyle("head", fontName="Bold", fontSize=24, leading=28, textColor=INK, spaceAfter=14),
        "sub": ParagraphStyle("sub", fontName="Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=14,
                              spaceAfter=7),
        "body": ParagraphStyle("body", fontName="Body", fontSize=10.5, leading=17, textColor=INK, spaceAfter=9),
        "small": ParagraphStyle("small", fontName="Body", fontSize=8.5, leading=12, textColor=SUBDUED, spaceAfter=7),
    }


def footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(MUTED)
    canvas.line(57, 50, 538, 50)
    canvas.setFont("Body", 8)
    canvas.setFillColor(SUBDUED)
    canvas.drawString(57, 34, "CS2229 · Benchmark dữ liệu bảng")
    canvas.drawRightString(538, 34, str(document.page))
    canvas.restoreState()


def build():
    style = styles()
    contents = []

    def p(value, kind="body"):
        contents.append(Paragraph(value, style[kind]))

    def figure(filename, caption, width=WIDTH):
        image = Image(str(ROOT / "figures" / filename))
        image.drawHeight = image.imageHeight * width / image.imageWidth
        image.drawWidth = width
        image.hAlign = "CENTER"
        contents.extend((image, Spacer(1, 8)))
        p(caption, "small")

    def table(headers, rows, widths):
        cells = [[Paragraph(str(value), style["small"]) for value in row] for row in [headers, *rows]]
        panel = Table(cells, colWidths=widths, hAlign="LEFT")
        panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), MUTED),
            ("LINEBELOW", (0, 0), (-1, -1), 0.45, MUTED),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        contents.append(panel)

    evidence = json.loads((ROOT / "evidence/project_snapshot.json").read_text(encoding="utf-8"))
    tests = json.loads((ROOT / "evidence/python_r_parity.json").read_text(encoding="utf-8"))
    recovered = json.loads((ROOT / "evidence/published_parity.json").read_text(encoding="utf-8"))
    contents.append(Spacer(1, 46))
    p("ĐỒ ÁN CS2229", "sub")
    p("Vì sao mô hình cây vẫn mạnh trên dữ liệu bảng?", "cover")
    p("Tái lập phân tích từ dữ liệu công bố và kiểm chứng thực nghiệm một phần")
    p("Paper gốc của Léo Grinsztajn, Edouard Oyallon và Gaël Varoquaux, NeurIPS 2022.")
    contents.append(Spacer(1, 22))
    p("Phạm vi", "sub")
    p("Bộ nộp tái lập phân tích cho 27/40 hình từ điểm từng lượt tác giả công bố. "
      "Phần đối chứng dùng kết quả V2 đã tự train và kiểm chứng độc lập.")
    p("Nguồn điểm của từng biểu đồ được ghi dưới hình. Slide HTML trình bày câu hỏi và kết quả chính.")
    contents.append(PageBreak())

    p("Thiết kế benchmark", "head")
    p("Paper mô tả 45 bộ dữ liệu. Benchmark phân theo loại đặc trưng và bài toán. "
      "Nhóm medium dùng trần khoảng 10.000 dòng train. Accuracy dùng cho phân loại và R² cho hồi quy.")
    p("Chọn cấu hình", "sub")
    p("Tại mỗi ngân sách, chọn theo validation trong số lượt đã quan sát, rồi lấy điểm test của cấu hình được chọn. "
      "Mỗi lần xáo thay đổi thứ tự các lượt công bố. Điều này mô phỏng ngân sách tìm kiếm từ cùng dữ liệu đã lưu.")
    p("Tái lập số bằng Python", "sub")
    p("CSV nguồn được kiểm tra SHA-256. Các notebook lọc đúng nhánh, giữ thứ tự dòng, tái lập chuỗi runif của R "
      "với seed 0 cố định và so sánh đầu ra với phân tích tham chiếu chạy bằng R.")
    table(["Figure", "Số hàng đối chiếu", "Sai số tuyệt đối lớn nhất"], [
        (row["figure"], fmt(row["rows"]), f'{row["max_absolute_error"]:.2g}') for row in tests["figures"]
    ], (95, 163, 223))
    p("Các hàng có giá trị thiếu được so sánh riêng. Mã vẽ dùng các số đã đối chiếu.", "small")
    contents.append(PageBreak())

    for number, (title, protocol, finding) in enumerate(zip(TITLES, STEPS, FINDINGS), 1):
        p(title, "head")
        figure(f"figure_{number:02d}.png", "Nguồn điểm là CSV công bố của tác giả. Hình vẽ lại bằng Python.")
        p("Thiết lập", "sub")
        p(protocol)
        p("Diễn giải", "sub")
        p(finding)
        if number == 4:
            p("Script Figure 4b dùng 15 lần xáo và dải 20%–80%. Caption mô tả 30 lần và min–max. "
              "Notebook lưu riêng cả hai phương án. Hình trên sử dụng phương án của script.")
        contents.append(PageBreak())

    p("Đối chứng V2 do nhóm tự chạy", "head")
    counts = evidence["counts"]
    table(["Track", "Thành công", "Lỗi", "Hết giờ"], [
        (name, fmt(part["completed"]), fmt(part["failed"]), fmt(part["timeout"]))
        for name, part in (("V2 chính", counts["main"]), ("V2 mở rộng", counts["extensions"]))
    ], (176, 112, 90, 103))
    p("Kết quả đã lưu qua hai bước", "sub")
    p(f'Tách điểm validation và test, đối chiếu dữ liệu số tại 1e-12 và kiểm tra '
      f'{evidence["original_source_files"]} hash nguồn tác giả cùng {evidence["worker_files"]} hash worker. '
      "24/57 thiết lập A.1 đủ điều kiện so giữa các họ mô hình ở cùng ngân sách.")
    coverage = evidence["groups_at_400"]
    table(["HPO tham chiếu", "Cấu hình đủ số lần lặp"], [
        (label, f"{fmt(done)}/{fmt(total)} ({done / total:.2%})")
        for label, done, total in zip(("Neural numeric", "Cây numeric", "Neural categorical", "Cây categorical"),
                                     coverage[::2], coverage[1::2])
    ], (245, 236))
    p("Nhóm cây ở bảng này gồm RF, GBT và XGBoost. Mẫu số là 400 cấu hình ngoài default cho mỗi cặp task–mô hình.", "small")
    p("Dừng lô và bảo toàn kết quả", "sub")
    p("Hai fit Desktop đang chạy đã bị gián đoạn khi dừng. Dell còn 178 lượt và Desktop còn 589 lượt chưa bắt đầu. "
      "Mười phiên Kaggle cũ thiếu đầu ra có thể xác minh theo đúng phiên bản.")
    contents.append(PageBreak())

    p("Hai thí nghiệm tự chạy tiêu biểu", "head")
    figure("figure_02_v2.png", "Pilot làm trơn eye_movements · 8 fit · RF và MLP · 1 seed.", width=420)
    figure("figure_05_v2.png", "Phép xoay ở cấu hình mặc định · 16 bộ dữ liệu · 381 cặp · 762/768 fit lõi.", width=420)
    p("Hai hình dùng accuracy thô. Notebook Figure 1 có thêm đồ thị V2 cùng ngân sách.", "small")
    contents.append(PageBreak())

    p("Kết luận và cách chạy lại", "head")
    p("Bài báo liên hệ lợi thế của mô hình cây với khả năng xử lý đặc trưng ít thông tin, "
      "học hàm mục tiêu gồ ghề và tận dụng hướng trục của dữ liệu.")
    p("Khoảng cách với thực nghiệm đầy đủ", "sub")
    p("Bộ nộp tái lập phân tích từ điểm công bố. Phần tự train mới bao phủ một phần ma trận HPO và ablation của paper. "
      "Hai nguồn điểm có nhãn riêng trong notebook, PDF và slide.")
    p("Chạy lại", "sub")
    p("Cài môi trường Python theo README, rồi mở từng notebook. Lượt chạy đầu tải CSV công khai và xác minh SHA-256. "
      "Các lượt tiếp theo dùng cache đã thu gọn. Mở slides/index.html để trình chiếu offline.")
    p("Nguồn", "sub")
    for label, link in (("Paper NeurIPS 2022", PAPER), ("Code tác giả", REPO),
                        ("CSV benchmark trên Figshare", "https://ndownloader.figshare.com/files/40081681"),
                        ("CSV ablation lịch sử", LFS)):
        p(f'<link href="{link}" color="#16776d">{label}</link>')

    contents.append(PageBreak())
    p("Phụ lục · So theo thời gian tìm kiếm", "head")
    figure("figure_06.png", "Figure 6 · Đặc trưng số · 15 bộ dữ liệu phân loại và 19 bộ hồi quy.")
    figure("figure_07.png", "Figure 7 · Có đặc trưng phân loại · 7 bộ dữ liệu phân loại và 13 bộ hồi quy.")
    p("Thời gian cộng dồn từ từng lượt trong CSV lịch sử của tác giả. Mô hình được chọn qua validation. "
      "Vùng tô là min–max qua 15 hoán vị. Phép đo phụ thuộc vào phần cứng và môi trường chạy của tác giả.")
    p("Đối chiếu với R trên 624 hàng. Sai số điểm nhỏ hơn 10⁻¹². "
      "Sai số thời gian lớn nhất khoảng 2,9 × 10⁻¹¹ giây; kiểm tra dùng tolerance tương đối và tuyệt đối 10⁻¹².",
      "small")

    contents.append(PageBreak())
    p("Phụ lục · Tăng dữ liệu train lên 50K", "head")
    figure("figure_08.png", "Figure 8 · Đặc trưng số · Phân loại · 4 bộ dữ liệu.")
    figure("figure_09.png", "Figure 9 · Đặc trưng số · Hồi quy · 3 bộ dữ liệu.")
    p("Cùng nhóm bộ dữ liệu trong hai ô. Điểm chuẩn hóa được tính theo script tác giả cho từng kích thước. "
      "Sự thay đổi thứ hạng giữa các mô hình có thể khác theo nhóm bài toán.")
    contents.append(PageBreak())
    p("Phụ lục · 10K và 50K khi có biến phân loại", "head")
    figure("figure_10.png", "Figure 10 · Phân loại · 2 bộ dữ liệu.")
    figure("figure_11.png", "Figure 11 · Hồi quy · 5 bộ dữ liệu.")
    p("Các đường cong đọc lại từ CSV lịch sử có cả nhánh 10K và 50K. "
      "Figure 12–17 trong bộ notebook cho phép tra điểm thô theo từng bộ dữ liệu.")

    contents.append(PageBreak())
    p("Phụ lục · Hồi quy A.4 và danh mục", "head")
    p("Bốn hồi quy kiểm tra mức thay đổi của điểm chuẩn hóa. Mỗi mô hình kiểm soát khác biệt giữa các bộ dữ liệu. "
      "Nhóm tham chiếu là FT-Transformer. Hệ số tương tác cho biết độ dốc khác nhóm tham chiếu bao nhiêu.")
    with (ROOT / "evidence/a4_coefficients.csv").open(encoding="utf-8", newline="") as handle:
        coefficient = {(r["analysis"], r["term"]): float(r["estimate"]) for r in csv.DictReader(handle)}
    examples = (
        ("smoothing", "Làm trơn mục tiêu", "lengthscale", "GradientBoostingTree", "GBT"),
        ("remove", "Loại đặc trưng ít thông tin", "prop_removed", "Resnet", "ResNet"),
        ("add", "Thêm đặc trưng không thông tin", "prop_added", "Resnet", "ResNet"),
        ("rotation", "Xoay không gian đặc trưng", "rotationTRUE", "GradientBoostingTree", "GBT"),
    )
    table(["Thao tác", "Hệ số của FT-Transformer", "Ví dụ hệ số tương tác"], [
        (name, f"{coefficient[analysis, term]:+.4f}",
         f"{label} · {coefficient[analysis, f'model_name{model}:{term}']:+.4f}")
        for analysis, name, term, model, label in examples
    ], (166, 153, 162))
    p("Notebook A.4 lưu 34 hệ số đã đối chiếu với R, cùng số quan sát, sai số chuẩn, thống kê t, "
      "p-value và cột hệ số công bố để so trực tiếp. Hệ số chạy lại dùng seed 0; bảng gốc làm tròn.", "small")
    p("Danh mục công bố", "sub")
    table(["Nguồn", "Đã lập danh mục", "Nội dung"], [
        ("Phụ lục A.1", "57/57 thiết lập", "OpenML task, dataset ID, số mẫu và đặc trưng."),
        ("Phụ lục A.3", "8/8 bảng, 70 tham số", "Phân phối tìm kiếm và giá trị mặc định được in."),
    ], (107, 147, 227))
    p("Các thiết lập A.1 có thể dùng chung bộ dữ liệu giữa các bài toán hoặc xử lý khác nhau. Paper nêu 45 bộ dữ liệu.", "small")
    p("Notebook Figure 18 và 20–24 tách các biến đổi theo từng bộ dữ liệu. Figure 21 ghi hai phương án 15 và "
      "30 hoán vị do script và chú thích mô tả khác nhau.", "small")

    contents.append(PageBreak())
    p("Phụ lục · Đổi phương pháp tìm siêu tham số", "head")
    figure("figure_33.png", "Figure 33 · Phân loại trên dữ liệu chỉ có đặc trưng số.", width=358)
    figure("figure_34.png", "Figure 34 · Hồi quy trên dữ liệu chỉ có đặc trưng số.", width=358)
    p("Random search tổng hợp qua 15 hoán vị. Bayesian search giữ thứ tự công bố. Hai đường đều lấy "
      "điểm test của cấu hình tốt nhất theo validation tại mỗi ngân sách. Số liệu lấy từ CSV tác giả.", "small")

    contents.append(PageBreak())
    p("Phụ lục · Cố định hay tune số cây", "head")
    figure("figure_39.png", "Figure 39 · Đặc trưng số · Nét đứt là nhánh tune số cây.")
    figure("figure_40.png", "Figure 40 · Có biến phân loại · Nét đứt là nhánh tune số cây.")
    tuned_rows = sum(row["rows"] for row in recovered["tuned_figures"])
    p(f"Tái lập từ checkpoint Git LFS cố định và tune số cây. {fmt(tuned_rows)} hàng khớp R ở ngưỡng 10⁻¹². "
      "Đối chiếu 24 đường mặc định với hình tác giả ở ngưỡng 10⁻⁵. Phần hoán vị dùng seed 0.",
      "small")

    contents.append(PageBreak())
    p("Phụ lục · Điều kiện để chạy lại các hình còn thiếu", "head")
    p("Khi có cấu hình, dữ liệu và cách chia gốc, chạy lại fit rồi lưu validation/test và thời gian. "
      "Khi có checkpoint gốc, có thể đo lại điểm trên đúng split. Các notebook phân tích điểm đã lưu "
      "chỉ tính lại bước chọn theo validation và tổng hợp test.")
    table(["Figure", "Đầu vào còn thiếu", "Bước kiểm chứng"], [
        ("19", "Hai đặc trưng, cách chia và cấu hình chính xác của hình Electricity.",
         "Fit hoặc nạp RF/MLP, đo điểm trên split giữ riêng, xuất dự đoán theo lưới."),
        ("25–32", "Danh sách ứng viên trước lọc và metadata tại từng bước.",
         "Áp dụng quy tắc gốc, lưu ID và đếm lượt giữ hoặc loại sau mỗi bước."),
        ("35–38", "Cấu hình từng lượt và kết quả validation/test/thời gian cho nhánh 40K.",
         "Chạy hai nhánh theo protocol của tác giả, rồi dựng đường theo ngân sách và thời gian."),
    ], (55, 213, 213))
    p("Các bản đọc hình trong PDF được lưu ở mục tư liệu của repo. Phạm vi 27/40 hình chỉ tính từ đầu ra từng lượt.",
      "small")
    target = ROOT / "report" / "bao-cao.pdf"
    SimpleDocTemplate(str(target), pagesize=A4, topMargin=64, bottomMargin=65,
                      leftMargin=57, rightMargin=57,
                      title="Vì sao mô hình cây vẫn mạnh trên dữ liệu bảng?",
                      author="CS2229").build(contents, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    build()
