"""CSV inputs from the original demo, with the already R-verified analysis engine."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from demo.upstream.core.benchmark import read_results
from src import analysis as A


def read_csv(source):
    if isinstance(source, bytes):
        source = BytesIO(source)
    elif isinstance(source, str):
        source = Path(source).expanduser()
        if not source.is_file() or not source.name.endswith((".csv", ".csv.gz")):
            raise ValueError("Chọn file CSV hoặc CSV.GZ tồn tại.")
    frame = read_results(source)
    if frame.empty:
        raise ValueError("CSV không có kết quả.")
    if len(frame) > 500000:
        raise ValueError(
            "CSV vượt 500.000 dòng; chọn một nhóm benchmark cho demo local."
        )
    for column in (A.VALID, A.TEST):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    if frame[[A.MODEL, A.TASK]].isna().any().any():
        raise ValueError("CSV thiếu tên model hoặc dataset.")
    if frame[[A.VALID, A.TEST]].notna().all(axis=1).sum() == 0:
        raise ValueError("CSV không có cặp validation/test hợp lệ.")
    return frame


def curves(frame, panel, shuffles, max_iter):
    if "benchmark" in frame:
        frame = frame.loc[frame.benchmark.eq(panel)].copy()
    else:
        frame = frame.copy()
    if "mean_time" in frame:
        frame = frame.loc[frame.mean_time.notna()].copy()
    frame[A.MODEL] = frame[A.MODEL].replace(A.ALIASES)
    if frame.empty:
        raise ValueError("Nhóm benchmark không có kết quả hợp lệ.")
    if not 3 <= shuffles <= 30 or not 1 <= max_iter <= 400:
        raise ValueError("Ngân sách phân tích ngoài phạm vi demo.")
    regression = "regression" in str(panel)
    if panel == "numerical_regression_medium":
        frame = frame.loc[frame[A.MODEL].ne("HistGradientBoostingTree")].copy()
    # random_search completes the model/task/rank grid; check memory before creating it.
    largest = int(frame.groupby([A.MODEL, A.TASK]).size().max()) + 1
    estimate = (
        frame[A.MODEL].nunique() * frame[A.TASK].nunique() * largest * shuffles * 120
    )
    if estimate > psutil.virtual_memory().available * 0.5:
        raise ValueError(
            "CSV cần nhiều RAM để tổng hợp. Chọn một nhóm nhỏ hơn hoặc giảm số hoán vị."
        )
    has_defaults = "hp" in frame and frame.hp.eq("default").any()
    if "hp" not in frame:
        frame["hp"] = "random"
    elif frame.hp.isna().any():
        raise ValueError(
            "Cột hp có giá trị trống; xác định default/random trước khi phân tích."
        )
    normalized = A.normalize(
        frame, quantile=0.5 if regression else 0.1, clip_test=regression
    )
    if not normalized[[A.VALID, A.TEST]].notna().all(axis=1).any():
        raise ValueError("CSV không đủ biến thiên để chuẩn hóa và tính đường HPO.")
    search = A.random_search(
        normalized, shuffles=shuffles, default_first=has_defaults, match_tasks=True
    )
    averaged = A.across_tasks(search, ("rank", A.MODEL, "shuffle"), skip_missing=True)
    result = A.bands(averaged, ("rank", A.MODEL))
    result = result.loc[result["rank"] <= max_iter].copy()
    result["datasets"] = frame[A.TASK].nunique()
    return result.rename(columns={A.MODEL: "model"}), bool(has_defaults)
