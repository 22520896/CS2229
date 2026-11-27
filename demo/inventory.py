"""Paper inventory: analysis coverage and self-training scope remain separate."""

import pandas as pd

from demo.evidence import csv, link

FIGURES = {
    1: "Benchmark HPO",
    2: "Target smoothing",
    3: "Đặc trưng giữ lại / bị loại",
    4: "Loại và thêm feature",
    5: "Rotation",
    6: "Ngân sách thời gian · numeric",
    7: "Ngân sách thời gian · categorical",
    8: "10K/50K · numeric classification",
    9: "10K/50K · numeric regression",
    10: "10K/50K · categorical classification",
    11: "10K/50K · categorical regression",
    12: "Numeric classification · từng dataset",
    13: "Numeric regression · từng dataset",
    14: "Categorical classification · từng dataset",
    15: "Categorical regression · từng dataset",
    16: "50K · numeric",
    17: "50K · categorical",
    18: "Smoothing · từng dataset",
    19: "Decision boundary",
    20: "Đặc trưng giữ lại / bị loại · từng dataset",
    21: "Noise · từng dataset",
    22: "Feature removal · từng dataset",
    23: "Rotation · từng dataset",
    24: "Rotation sau loại 50% feature",
    25: "Lọc numeric regression",
    26: "Lọc numeric classification",
    27: "Lọc categorical regression",
    28: "Lọc categorical classification",
    29: "Lọc 50K · numeric regression",
    30: "Lọc 50K · numeric classification",
    31: "Lọc 50K · categorical regression",
    32: "Lọc 50K · categorical classification",
    33: "Bayesian search · classification",
    34: "Bayesian search · regression",
    35: "40K estimators / patience · numeric",
    36: "40K estimators / patience · categorical",
    37: "Thời gian 40K estimators · numeric",
    38: "Thời gian 40K estimators · categorical",
    39: "Tune số cây · numeric",
    40: "Tune số cây · categorical",
}
GROUPS = [
    "numeric_classification",
    "numeric_regression",
    "categorical_classification",
    "categorical_regression",
]
COVERAGE_LABELS = {
    "settings_numeric": "Numeric · thiết lập có output",
    "settings_categorical": "Categorical · thiết lập có output",
    "default_numeric_trees": "Numeric · default RF/GBT/XGB",
    "default_categorical_trees": "Categorical · default RF/GBT/XGB",
    "numeric_hpo": "Numeric · HPO RF/GBT/XGB",
    "categorical_hpo": "Categorical · HPO RF/GBT/XGB",
    "hgbt": "Categorical · HPO HGBT",
    "numeric_neural_hpo": "Numeric · HPO 4 neural",
    "categorical_neural_hpo": "Categorical · HPO 4 neural",
    "saint": "SAINT · defaults",
    "50k": "Train cap 50K · defaults",
    "noise_features": "Noise · cấu hình công bố",
    "rotation": "Rotation · cấu hình công bố",
    "target_smoothing": "Smoothing · cấu hình công bố",
}


def coverage():
    frame = csv("reproduce/results/coverage.csv")
    expected = 100 * frame.complete / frame.reference
    if (expected - frame.percent).abs().max() > 1e-10:
        raise ValueError("Tỷ lệ độ phủ không khớp mẫu số")
    frame["Hạng mục"] = frame.key.map(COVERAGE_LABELS)
    frame["Đã đủ / tham chiếu"] = (
        frame.complete.map(lambda n: f"{n:,}")
        + " / "
        + frame.reference.map(lambda n: f"{n:,}")
    )
    frame["Đơn vị"] = frame.unit.map(
        {
            "setting": "Thiết lập",
            "full-repeat configuration": "Cấu hình đủ seed",
            "published-parameter condition cell": "Ô điều kiện công bố",
        }
    )
    frame["Độ phủ (%)"] = frame.percent
    frame["Phạm vi"] = "Chưa replay đầy đủ paper"
    frame.loc[
        frame.key.isin(["noise_features", "rotation", "target_smoothing"]), "Phạm vi"
    ] = "Một cấu hình/điều kiện; đủ repeats"
    return frame[["Hạng mục", "Đã đủ / tham chiếu", "Độ phủ (%)", "Đơn vị", "Phạm vi"]]


def paper_inventory():
    from demo.progress_estimates import all_estimates, figure_estimate

    estimates = all_estimates()
    rows = []
    source = csv("reproduce/results/figure_inventory.csv")
    for r in source.itertuples():
        n = r.figure
        own = "Chưa có thực nghiệm đúng phạm vi"
        if n in [1, 6, 7, *range(8, 18)]:
            own = "Benchmark/HPO một phần"
        if n in [2, 18]:
            own = "Đủ lưới công bố · 15 dataset"
        if n in [3, 20, 22]:
            own = "Đại diện · 8 dataset"
        if n == 4:
            own = "4b đủ lưới; 4a chạy đại diện"
        if n == 21:
            own = "Đủ lưới công bố · 14 dataset"
        if n in [5, 23, 24]:
            own = "Đủ lưới công bố · 14 dataset"
        rows.append(
            {
                "ID": f"Figure {n}",
                "Nội dung": FIGURES[n],
                "Loại": "Hình chính" if n <= 5 else "Hình phụ lục",
                "Phân tích tác giả": "Đã tính lại"
                if r.author_results_analysis
                else "Chưa có đầu vào",
                "Ước lượng tự train (%)": figure_estimate(n, estimates)["percent"],
                "Cơ sở ước lượng": figure_estimate(n, estimates)["basis"],
                "Tự chạy": own,
                "Bằng chứng": link("notebooks/" + r.author_notebook)
                if r.author_results_analysis
                else "",
            }
        )
    hpo = csv("evidence/appendix_hpo.csv")
    for table, group in hpo.groupby("table"):
        rows.append(
            {
                "ID": f"Table {table}",
                "Nội dung": str(group.model.iloc[0]) + f" · {len(group)} tham số",
                "Loại": "Bảng siêu tham số",
                "Phân tích tác giả": "Đã đối chiếu danh mục",
                "Tự chạy": "Dùng cấu hình công bố; HPO có phạm vi riêng",
                "Bằng chứng": link("notebooks/appendix_a1_a3_catalog.ipynb"),
            }
        )
    rows.append(
        {
            "ID": "A.1",
            "Nội dung": "Danh mục benchmark · 57 thiết lập",
            "Loại": "Danh mục",
            "Phân tích tác giả": "Đã đối chiếu OpenML",
            "Tự chạy": "57/57 có output; không phải đủ HPO",
            "Bằng chứng": link("notebooks/appendix_a1_a3_catalog.ipynb"),
        }
    )
    coeff = csv("evidence/a4_coefficients.csv")
    for name, group in coeff.groupby("analysis"):
        rows.append(
            {
                "ID": "A.4 · " + name,
                "Nội dung": f"Hồi quy · {len(group)} hệ số",
                "Loại": "Phân tích định lượng",
                "Phân tích tác giả": "Python/R đã đối chiếu",
                "Tự chạy": "Phân tích từ CSV tác giả",
                "Bằng chứng": link("notebooks/appendix_a4_regression.ipynb"),
            }
        )
    for identifier, title, status in [
        ("A.2", "Chi tiết benchmark và thời gian", "Có phân tích Figure6–17"),
        ("A.3", "Không gian siêu tham số", "Đã đối chiếu 8 bảng / 70 mục"),
        ("A.5", "Quy trình chọn dataset", "Thiếu log ứng viên để tái lập đầy đủ"),
        ("A.6", "Đối chiếu benchmark khác", "Chưa tái lập toàn bộ phép đối chiếu"),
        ("A.7", "Bayesian optimization", "Đã phân tích Figure 33–34"),
        ("A.8", "Thảo luận Kadra et al.", "Thảo luận; không có % training riêng"),
        (
            "A.9",
            "Độ quan trọng siêu tham số",
            "Có Figure 39–40; thiếu nhánh 40K estimators",
        ),
        ("A.10", "Hướng dẫn dùng benchmark", "Đã ghi nguồn và hướng dẫn local"),
    ]:
        rows.append(
            {
                "ID": identifier,
                "Nội dung": title,
                "Loại": "Mục phụ lục",
                "Phân tích tác giả": status,
                "Tự chạy": "Xem phạm vi từng Figure",
                "Bằng chứng": link("evidence/appendix_sources.md"),
            }
        )
    return pd.DataFrame(rows)


def paper_scope():
    """Source HPO-budget estimates, explicitly distinct from protocol parity."""
    from demo.progress_estimates import all_estimates

    estimates = all_estimates()
    rows = []
    descriptions = [
        (
            "Figure 1",
            "Benchmark",
            "4 nhóm; HPO tối đa 400 cấu hình/model/setting, default-first, chọn validation",
            "HPO có phạm vi riêng; bộ đối chứng cân bằng 24/57 thiết lập",
            "Chưa đủ toàn bộ mô hình, ngân sách và quy trình gốc",
        ),
        (
            "Figure 2",
            "Target smoothing",
            "Làm trơn nhãn train; HPO hạng 60, 15 hoán vị theo script",
            "target_smoothing",
            "Một cấu hình công bố/điều kiện; chưa replay HPO và hoán vị",
        ),
        (
            "Figure 3",
            "Feature importance",
            "GBT trên feature giữ lại/bị loại; HPO hạng 20, 30 hoán vị",
            "feature_removal",
            "Case đại diện; chưa đủ grid và HPO gốc",
        ),
        (
            "Figure 4a",
            "Feature removal",
            "Loại biến ít thông tin; HPO hạng 30, 30 hoán vị",
            "feature_removal",
            "Case đại diện; chưa đủ grid và HPO gốc",
        ),
        (
            "Figure 4b",
            "Noise features",
            "Thêm biến nhiễu; HPO hạng 20, 15 hoán vị theo script",
            "noise_features",
            "Một cấu hình công bố/điều kiện; chưa replay HPO và hoán vị",
        ),
        (
            "Figure 5",
            "Rotation",
            "Có/không xoay; loại 0%/50% biến; HPO hạng 20, 30 hoán vị",
            "rotation",
            "Một cấu hình công bố/điều kiện; chưa replay HPO và hoán vị",
        ),
        (
            "Figure 6–7",
            "Ngân sách thời gian",
            "Tổng hợp theo thời gian tìm kiếm ở cả numeric/categorical",
            "Có timing tự chạy; chưa đủ phép tổng hợp gốc",
            "Phạm vi HPO/time bins còn thiếu",
        ),
        (
            "Figure 8–11, 16–17",
            "Train 10K / 50K",
            "So cùng cohort theo từng nhóm và quy mô dữ liệu",
            "Có default và HPO 50K một phần",
            "Chưa đủ paired HPO theo paper",
        ),
        (
            "Figure 12–15",
            "Điểm từng dataset",
            "Đủ model/setting và quy trình HPO trong bốn nhóm",
            "Có kết quả tự chạy một phần",
            "Không đồng nhất 57 setting A.1 với 59 nhãn Figure 1",
        ),
        (
            "Figure 18, 20–24",
            "Cơ chế · từng dataset",
            "Phân rã các thực nghiệm Figure 2–5, giữ cùng quy trình",
            "Đã có ba lưới cấu hình công bố và case removal",
            "Cùng giới hạn HPO của các hình chính; không đếm hai lần",
        ),
        (
            "Figure 19",
            "Decision boundary",
            "Electricity, feature/split/model gốc của thí nghiệm",
            "Chưa tái hiện đúng phạm vi",
            "Demo 2D local được giữ riêng, không tính là Figure 19",
        ),
        (
            "Figure 25–32",
            "Lọc dataset",
            "Các cohort lọc của paper, ở 10K và 50K",
            "Chưa có thực nghiệm đúng phạm vi",
            "Thiếu grid đối chứng và quy trình lọc gốc",
        ),
        (
            "Figure 33–34",
            "Bayesian search",
            "Random search so với Bayesian optimization",
            "Chưa tự train Bayesian benchmark",
            "Hiện mới phân tích CSV công bố",
        ),
        (
            "Figure 35–38",
            "40K estimators / patience",
            "Train với giới hạn số cây/patience dài và đo ngân sách",
            "Chưa có thực nghiệm đúng phạm vi",
            "Không phải 40K dòng train",
        ),
        (
            "Figure 39–40",
            "Tune số cây",
            "Nhánh cố định so với HPO có tune n_estimators",
            "Chưa tự train đủ hai nhánh",
            "Hiện mới phân tích CSV công bố",
        ),
        (
            "A.1–A.3",
            "Dataset / siêu tham số",
            "Danh mục 57 thiết lập và 8 bảng HPO; protocol chi tiết",
            "Đã đối chiếu danh mục; có cấu hình nguồn",
            "Đối chiếu tài liệu, không có % tự train riêng",
        ),
        (
            "A.4",
            "Hồi quy định lượng",
            "Bốn mô hình hồi quy trên kết quả thực nghiệm gốc",
            "Đã chạy phân tích từ CSV tác giả, đối chiếu Python/R",
            "Phân tích kết quả tác giả, không phải tự train benchmark",
        ),
        (
            "A.5–A.10",
            "Các mục phụ lục khác",
            "Lọc dữ liệu, đối chiếu benchmark, Bayes và phân tích bổ sung",
            "Xem danh mục từng Figure / mục phụ lục",
            "Không gộp thành một % hoàn thành paper",
        ),
    ]
    from demo.evidence import cells

    evidence = cells()
    for figure, name, required, own, missing in descriptions:
        mechanism = (
            own
            if own
            in {"target_smoothing", "feature_removal", "noise_features", "rotation"}
            else None
        )
        if mechanism:
            selected = evidence.loc[
                evidence.mechanism.eq(mechanism) & evidence.full_repeats
            ]
            if mechanism != "feature_removal":
                selected = selected.loc[selected.reference_grid]
            own = f"{len(selected)} ô đủ seed · {selected.dataset.nunique()} dataset · cấu hình công bố/đại diện"
        estimated = estimates[figure]
        if mechanism:
            own = f"{estimated['numerator']:,} cấu hình đủ seed khớp phạm vi gốc · một cấu hình/điều kiện"
        rows.append(
            {
                "Paper": figure,
                "Hạng mục": name,
                "Phạm vi gốc": required,
                "Đã thực hiện": own,
                "Ước lượng tự train (%)": estimated["percent"]
                if not figure.startswith("A.")
                else None,
                "Cơ sở ước lượng": estimated["basis"],
                "Cấu hình được tính": estimated["numerator"],
                "Ngân sách tham chiếu": estimated["denominator"],
                "Còn thiếu / giới hạn": missing,
            }
        )
    return pd.DataFrame(rows)


def work_volume():
    frame = csv("reproduce/results/coverage.csv")
    return pd.DataFrame(
        {
            "Hạng mục": frame.key.map(COVERAGE_LABELS),
            "Đã ghi nhận": frame.complete,
            "Đơn vị": frame.unit,
            "Phạm vi": frame.key.map(
                lambda key: "Lưới một cấu hình công bố/điều kiện"
                if key in {"noise_features", "rotation", "target_smoothing"}
                else "Catalog / ngân sách tự chạy đã khóa"
            ),
        }
    )
