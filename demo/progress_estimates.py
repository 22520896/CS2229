"""Explicit, source-budget estimates. Never equate a representative grid to paper."""

import json

from demo import evidence as E

TARGET_PATH = E.DATA / "progress_targets.json"
MECHANISMS = {
    "Figure 2": "target_smoothing",
    "Figure 3": "feature_removal",
    "Figure 4a": "feature_removal",
    "Figure 4b": "noise_features",
    "Figure 5": "rotation",
}


def estimate(
    numerator, denominator, basis, missing="Chưa replay đầy đủ protocol/HPO gốc"
):
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("Tử số/mẫu số ước lượng không hợp lệ")
    return {
        "percent": 100 * numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "basis": basis,
        "missing": missing,
    }


def all_estimates():
    targets = json.loads(TARGET_PATH.read_bytes())["targets"]
    own = E.cells()
    results = {}
    for figure, target in targets.items():
        rows = own.loc[own.mechanism.eq(MECHANISMS[figure]) & own.full_repeats]
        # Only verified original-cohort/model/condition intersections get credit.
        conditions = {tuple(condition) for condition in target["conditions"]}
        matched = rows.loc[
            rows.dataset.isin(target["datasets"])
            & rows.model.map(E.NAMES).isin(target["models"])
            & rows.parsed.map(lambda value: tuple(value) in conditions)
        ]
        matched = matched.assign(
            condition_key=matched.parsed.map(tuple)
        ).drop_duplicates(["dataset", "model", "condition_key"])
        denominator = (
            len(target["datasets"])
            * len(target["models"])
            * len(conditions)
            * target["hpo_rank"]
        )
        results[figure] = estimate(
            len(matched),
            denominator,
            f"{len(matched):,} cấu hình đủ seed / ({len(target['datasets'])} dataset × {len(target['models'])} model × {len(conditions)} điều kiện × ngân sách {target['hpo_rank']})",
        )
    coverage = E.csv("reproduce/results/coverage.csv").set_index("key")
    main = E.csv("figures/figure_01.csv")
    pairs = sum(
        int(group.datasets.iloc[0]) * group.model_name.nunique()
        for _, group in main.groupby("panel")
    )
    denominator = pairs * 400
    numerator = int(
        coverage.loc[
            [
                "numeric_hpo",
                "numeric_neural_hpo",
                "categorical_hpo",
                "hgbt",
                "categorical_neural_hpo",
            ],
            "complete",
        ].sum()
    )
    benchmark = estimate(
        numerator,
        denominator,
        f"{numerator:,} cấu hình đủ seed / ({pairs} cặp dataset-model gốc × 400); gồm 59 setting Figure 1, không rút xuống 57",
        "Ước lượng khối lượng; chưa chứng minh khớp chuỗi sampler và mọi dataset lịch sử",
    )
    for figure in ["Figure 1", "Figure 6–7", "Figure 12–15"]:
        results[figure] = dict(benchmark)
        if figure != "Figure 1":
            results[figure]["basis"] += (
                "; dùng lại ngân sách training Figure 1, không đếm là thực nghiệm mới"
            )
            results[figure]["missing"] += (
                "; biểu đồ tự chạy đúng phép tổng hợp còn thiếu"
            )
    large = int(
        json.loads((E.ROOT / "reproduce/results/run_status.json").read_bytes())[
            "large_cap_hpo_groups"
        ]
    )
    results["Figure 8–11, 16–17"] = estimate(
        large,
        denominator,
        f"{large:,} nhóm HPO 50K đủ repeats trong snapshot / {denominator:,} ngân sách cấu hình tham chiếu như Figure 1; không cộng default",
        "Ước lượng từ tổng nhóm đã audit; bundle chưa đủ chi tiết để khẳng định khớp từng cohort và paired analysis gốc",
    )
    mechanisms = [results[key] for key in MECHANISMS]
    results["Figure 18, 20–24"] = estimate(
        sum(item["numerator"] for item in mechanisms),
        sum(item["denominator"] for item in mechanisms),
        "Tổng cấu hình / ngân sách của Figure 2, 3, 4a, 4b, 5; chỉ là phân rã theo dataset, không cộng lần hai",
    )
    for figure in [
        "Figure 19",
        "Figure 25–32",
        "Figure 33–34",
        "Figure 35–38",
        "Figure 39–40",
    ]:
        results[figure] = estimate(
            0,
            1,
            "Chưa có output tự train đúng nhánh được xác minh trong snapshot; demo minh họa và CSV tác giả không tính",
            "Cần thực nghiệm tự train đúng phạm vi này",
        )
    results["A.1–A.3"] = estimate(
        0,
        1,
        "Mục tài liệu, không cộng vào tỷ lệ tự train; danh mục và HPO được báo riêng",
        "Đối chiếu tài liệu đã có; không phải training",
    )
    results["A.4"] = estimate(
        0,
        1,
        "Hồi quy từ CSV tác giả được báo ở phần phân tích, không tính là tự train",
        "Đã phân tích 4/4 hồi quy và đối chiếu R; không tự train benchmark",
    )
    results["A.5–A.10"] = estimate(
        0,
        1,
        "Training bổ sung chưa có bằng chứng đầy đủ; phân tích CSV và tài liệu được ghi riêng",
        "Xem từng Figure; không gộp thành % full paper",
    )
    return results


def figure_estimate(number, values=None):
    values = all_estimates() if values is None else values
    if number == 4:
        a, b = values["Figure 4a"], values["Figure 4b"]
        return estimate(
            a["numerator"] + b["numerator"],
            a["denominator"] + b["denominator"],
            "Hai nhánh 4a/4b theo tổng ngân sách cấu hình",
        )
    for key, numbers in [
        ("Figure 1", [1]),
        ("Figure 2", [2, 18]),
        ("Figure 3", [3, 20]),
        ("Figure 4a", [22]),
        ("Figure 4b", [21]),
        ("Figure 5", [5, 23, 24]),
        ("Figure 6–7", [6, 7]),
        ("Figure 8–11, 16–17", [8, 9, 10, 11, 16, 17]),
        ("Figure 12–15", [12, 13, 14, 15]),
        ("Figure 19", [19]),
        ("Figure 25–32", list(range(25, 33))),
        ("Figure 33–34", [33, 34]),
        ("Figure 35–38", list(range(35, 39))),
        ("Figure 39–40", [39, 40]),
    ]:
        if number in numbers:
            return values[key]
    raise ValueError("Không có Figure trong paper")
