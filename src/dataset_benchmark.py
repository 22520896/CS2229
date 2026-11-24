"""Figures 12–15, the author's raw-score curves by dataset."""

import numpy as np
import pandas as pd

from .analysis import across_tasks, bands, eligible, normalize, random_search, scores
from .author_data import LARGE_PANELS, MODEL, PANELS, TASK, TEST, TIME_PANELS


KINDS = ("numerical", "numerical", "categorical", "categorical")
LARGE_COUNTS = (4, 3, 2, 5)


def raw_curves(subset, clip_test):
    count = subset.groupby([TASK, MODEL], sort=False).size().groupby(level=0).min()
    rows = eligible(subset).replace([np.inf, -np.inf], np.nan)
    if clip_test:
        rows[TEST] = rows[TEST].clip(lower=0)
    selected = random_search(rows, match_tasks=True)
    subset = selected[(selected["rank"] <= 400) & (selected["rank"] < selected[TASK].map(count))]
    keys = (TASK, MODEL, "rank")
    summary = bands(subset, keys).dropna(subset=["mean"])
    weight = subset.loc[subset["score"].notna()].groupby(list(keys))["weight"].sum()
    summary["count"] = weight.reindex(pd.MultiIndex.from_frame(summary[list(keys)])).to_numpy()
    limits = summary.groupby(TASK)["mean"].transform("min")
    summary["low"] = summary["low"].clip(lower=limits)
    return summary.reset_index(drop=True)


def raw_medium(number):
    panel = PANELS[number - 12]
    published = scores("benchmark")
    subset = published[published["benchmark"] == panel]
    if number == 13:
        subset = subset[subset[MODEL].ne("HistGradientBoostingTree")]
    return raw_curves(subset, clip_test=number in (13, 15))


def raw_large(number):
    kind = "numerical" if number == 16 else "categorical"
    data = scores(f"large_{kind}")
    return pd.concat([
        raw_curves(rows, clip_test="_regression_" in key).assign(panel=key)
        for key, rows in data.groupby("benchmark", sort=True)
    ], ignore_index=True)


def train_size(number):
    index = number - 8
    kind = KINDS[index]
    large, medium = scores(f"large_{kind}"), scores(f"time_{kind}")
    large = large[large["benchmark"] == LARGE_PANELS[index]]
    medium = medium[(medium["benchmark"] == TIME_PANELS[index]) & medium[TASK].isin(large[TASK])]
    if index != 0:
        medium = medium[medium[MODEL].isin(large[MODEL])]
    task_count = large[TASK].nunique()
    if task_count != LARGE_COUNTS[index] or set(large[TASK]) != set(medium[TASK]):
        raise ValueError("Large-sized and medium-sized inputs match different task cohorts")
    output = []
    quantiles = ((0.1, 0.1), (0.4, 0.4), (0.1, 0.2), (0.5, 0.5))[index]
    for size, subset, quantile in zip((10000, 50000), (medium, large), quantiles):
        search = random_search(normalize(eligible(subset), quantile=quantile,
                                         clip_test=index in (1, 3)), match_tasks=True)
        average = across_tasks(search, ("rank", MODEL, "shuffle"), skip_missing=True)
        output.append(bands(average, (MODEL, "rank")).query("rank <= 400").assign(
            train_cap=size, datasets=task_count,
        ))
    return pd.concat(output, ignore_index=True)


def analyze(number):
    if number in range(8, 12):
        return train_size(number)
    if number in range(12, 16):
        return raw_medium(number)
    if number in (16, 17):
        return raw_large(number)
    raise ValueError(f"Unsupported dataset benchmark figure {number}")
