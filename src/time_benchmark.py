"""Time-budget curves for Figures 6–7 from the author's historical benchmark."""

import numpy as np
import pandas as pd

from .analysis import normalize, r_random_stream, scores, uniform
from .author_data import MODEL, TASK, TEST, TIME_PANELS, VALID


BREAKS = np.array([10, 20, 30, 50, 100, 150, 200, 400, 500, 800, 1200, 2000,
                   3000, 4000, 5000, 7000, 10000, 15000, 20000, 30000, 40000, 50000, 100000])
SOURCES_BY_FIGURE = {6: "time_numerical", 7: "time_categorical"}
COHORTS = dict(zip(TIME_PANELS, (15, 19, 7, 13)))


def time_winners(duration, validation, test):
    seconds = np.cumsum(duration, dtype=np.longdouble).astype(float)
    bins = np.searchsorted(BREAKS, seconds, side="right")
    valid = ~np.isnan(validation)
    if not valid.any():
        return pd.DataFrame(columns=["time_bin", "seconds", "score"], dtype=float)

    maxima = np.full(len(BREAKS) + 1, -np.inf)
    np.maximum.at(maxima, bins[valid], validation[valid])
    winner = valid & (validation == maxima[bins])
    bins, seconds = bins[winner], seconds[winner]
    validation, test = validation[winner], test[winner]
    past_best = np.r_[-np.inf, np.maximum.accumulate(validation)[:-1]]
    # A validation level can recur after an intervening lower max in a later bin.
    # Keep the earliest elapsed time across all occurrences, including exact ties.
    first_at_level = {}
    for score, elapsed in zip(validation, seconds):
        first_at_level.setdefault(score, elapsed)
    first = seconds == np.array([first_at_level[score] for score in validation])
    values = np.where((validation >= past_best) & first, test, np.nan)
    return pd.DataFrame(dict(time_bin=bins, seconds=seconds, score=values))


def shuffle_summary(groups, model_levels, task_levels, draws, defaults, duration, validation, test):
    winner_frames = []
    for (model, task), rows in groups.items():
        ordered = rows[np.argsort(draws[rows], kind="stable")]
        ordered = np.r_[rows[defaults[rows]], ordered[~defaults[ordered]]]
        winners = time_winners(duration[ordered], validation[ordered], test[ordered])
        if not winners.empty:
            winner_frames.append(winners.assign(**{MODEL: model, TASK: task}))
    winners = pd.concat(winner_frames, ignore_index=True)
    keys = [MODEL, TASK, "time_bin"]
    grid = pd.MultiIndex.from_product([model_levels, task_levels, sorted(winners.time_bin.unique())],
                                      names=keys).to_frame(index=False)
    complete = grid.merge(winners, how="left", on=keys, sort=False)
    complete["score"] = complete.groupby([MODEL, TASK], sort=False)["score"].ffill()
    grouped = complete.groupby([MODEL, "time_bin"], sort=True)
    mean = grouped["score"].mean().where(grouped["score"].count() == grouped.size())
    return pd.DataFrame({"score": mean, "seconds": grouped["seconds"].median()}).reset_index()


def time_curve(data, quantile, clip_test):
    data = data.loc[data["mean_time"].notna()].reset_index(drop=True)
    if data[[MODEL, TASK, "hp"]].isna().any().any() or (data["mean_time"] < 0).any():
        raise ValueError("Missing time-curve key or negative run duration")
    data = normalize(data, quantile=quantile, clip_test=clip_test)
    groups = data.groupby([MODEL, TASK], sort=True).indices
    model_levels = sorted(data[MODEL].unique())
    task_levels = sorted(data[TASK].unique())
    defaults = data["hp"].eq("default").to_numpy()
    duration, validation, test = (data[key].to_numpy(float) for key in ("mean_time", VALID, TEST))
    stream = r_random_stream(0)
    frames = [
        shuffle_summary(groups, model_levels, task_levels, uniform(stream, len(data)),
                        defaults, duration, validation, test)
        for _ in range(15)
    ]
    all_shuffles = pd.concat(frames, ignore_index=True)
    groups = all_shuffles.groupby([MODEL, "time_bin"], sort=True)
    result = groups["score"].agg(mean="mean", low="min", high="max")
    result["seconds"] = groups["seconds"].mean().where(groups["seconds"].count() == groups.size())
    return result[groups["score"].count() == groups.size()].reset_index()


def analyze(number):
    data = scores(SOURCES_BY_FIGURE[number])
    panels = TIME_PANELS[:2] if number == 6 else TIME_PANELS[2:]
    output = []
    for panel, quantile in zip(panels, (0.1, 0.5 if number == 6 else 0.4)):
        subset = data[data["benchmark"] == panel]
        count = subset.loc[subset["mean_time"].notna(), TASK].nunique()
        if count != COHORTS[panel]:
            raise ValueError(f"Expected {COHORTS[panel]} tasks for {panel}, got {count}")
        output.append(time_curve(subset, quantile, clip_test=quantile != 0.1).assign(
            panel=panel, datasets=count,
        ))
    return pd.concat(output, ignore_index=True)
