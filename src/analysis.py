"""Python implementation of the author's validation-first curve summaries."""

import numpy as np
import pandas as pd

from .author_data import (
    ADD, MODEL, PANELS, REMOVE, ROTATE, SMOOTH, TASK, TEST, VALID, prepare,
)


ALIASES = {
    "rf_c": "RandomForest", "rf_r": "RandomForest",
    "gbt_c": "GradientBoostingTree", "gbt_r": "GradientBoostingTree",
    "xgb_c": "XGBoost", "xgb_r": "XGBoost",
    "hgbt_c": "HistGradientBoostingTree", "hgbt_r": "HistGradientBoostingTree",
    "ft_transformer": "FT Transformer", "ft_transformer_regressor": "FT Transformer",
    "rtdl_resnet": "Resnet", "rtdl_resnet_regressor": "Resnet",
    "rtdl_mlp": "MLP", "rtdl_mlp_regressor": "MLP", "saint": "SAINT",
}


def r_random_stream(seed):
    """R's default MT19937 state initialization and one word per runif draw."""
    state = seed & 0xFFFFFFFF
    for _ in range(50):
        state = (69069 * state + 1) & 0xFFFFFFFF
    words = np.empty(625, dtype=np.uint32)
    for index in range(len(words)):
        state = (69069 * state + 1) & 0xFFFFFFFF
        words[index] = state
    random = np.random.RandomState()
    random.set_state(("MT19937", words[1:], 624, 0, 0.0))
    return random


def uniform(stream, size):
    values = stream.randint(0, 2**32, size=size, dtype=np.uint32) / 2**32
    # R excludes the two endpoints of the unit interval.
    return np.where(values == 0, 0.5 / (2**32 - 1), values)


def scores(name):
    return pd.read_csv(prepare(name), float_precision="round_trip", low_memory=False)


def eligible(data):
    data = data.loc[data["mean_time"].notna()].copy()
    data[MODEL] = data[MODEL].replace(ALIASES)
    return data.reset_index(drop=True)


def normalize(data, quantile=None, baseline=None, clip_test=False):
    data = data.copy()
    for column in (VALID, TEST):
        group = data.groupby(TASK, dropna=False)[column]
        floor = group.transform("quantile", q=quantile) if quantile is not None else baseline
        data[column] = (data[column] - floor) / (group.transform("max") - floor)
    if clip_test:
        data[TEST] = data[TEST].clip(lower=0)
    return data


def selected_curve(ranks, validation, test, targets):
    """Replicate cummax, first validation tie, and LOCF on rank-ordered inputs."""
    if len(ranks) == 0:
        return np.full(len(targets), np.nan), 1
    past_best = np.r_[-np.inf, np.maximum.accumulate(validation)[:-1]]
    accepted = (validation > past_best) & ~np.isnan(test)
    retained_ranks, retained_scores = ranks[accepted], test[accepted]
    if not len(retained_ranks):
        return np.full(len(targets), np.nan), 1
    indexes = np.searchsorted(retained_ranks, targets, side="right") - 1
    result = np.where(indexes >= 0, retained_scores[indexes], np.nan)
    rank_zero_weight = 1
    if len(ranks) > 1 and ranks[0] == ranks[1] == 0 and targets[0] == 0:
        # The analysis groups duplicate defaults by rank 0 after forward fill.
        zero = ranks == 0
        marked = np.cumsum(accepted[zero]) - 1
        valid = marked >= 0
        rank_zero_weight = int(valid.sum())
        result[0] = retained_scores[marked[valid]].mean() if rank_zero_weight else np.nan
    return result, rank_zero_weight


def random_search(data, conditions=(), *, seed=0, shuffles=15, default_first=True,
                  match_tasks=False, rank=None, stream=None):
    data = data.reset_index(drop=True)
    keys = (MODEL, TASK, *conditions)
    needed = data[list(keys)]
    if needed.isna().any().any():
        raise ValueError("Missing model, task or experimental condition")
    groups = data.groupby(list(keys), sort=True).indices
    valid = data[VALID].to_numpy(float)
    test = data[TEST].to_numpy(float)
    default = data["hp"].eq("default").to_numpy()
    if default_first and data["hp"].isna().any():
        raise ValueError("Missing default/random indicator")
    limits = {}
    if match_tasks:
        for key, rows in groups.items():
            count = int(np.count_nonzero(~np.isnan(valid[rows])))
            limits[key[1]] = min(limits.get(key[1], count), count)

    # Complete the same model/task/condition grid as tidyr::complete.
    levels = [sorted(data[key].unique()) for key in keys]
    grid = pd.MultiIndex.from_product(levels, names=keys).to_frame(index=False)
    grid_rows = {key: i for i, key in enumerate(grid.itertuples(index=False, name=None))}
    maximum = max(limits.values()) if match_tasks else max(len(g) for g in groups.values())
    targets = np.arange(0 if default_first else 1, maximum + 1) if rank is None else np.array([rank])
    stream = r_random_stream(seed) if stream is None else stream
    frames = []
    for shuffle in range(1, shuffles + 1):
        draws = uniform(stream, len(data))
        result = np.full((len(grid), len(targets)), np.nan)
        weight = np.ones_like(result, dtype=int)
        observed_ranks = set()
        for key, rows in groups.items():
            order = np.argsort(draws[rows], kind="stable")
            ranks = np.empty(len(rows), dtype=int)
            ranks[order] = np.arange(1, len(rows) + 1)
            if default_first:
                ranks[default[rows]] = 0
            order = np.argsort(ranks, kind="stable")
            if match_tasks:
                order = order[ranks[order] <= limits[key[1]]]
            ranks, indexes = ranks[order], rows[order]
            observed_ranks.update(ranks)
            row = grid_rows[key]
            result[row], weight[row, 0] = selected_curve(ranks, valid[indexes], test[indexes], targets)
        present = np.isin(targets, list(observed_ranks))
        if not present.any():
            continue
        table = grid.loc[grid.index.repeat(int(present.sum()))].reset_index(drop=True)
        table["rank"] = np.tile(targets[present], len(grid))
        table["shuffle"] = shuffle
        table["score"] = result[:, present].ravel()
        table["weight"] = weight[:, present].ravel()
        frames.append(table)
    if not frames:
        raise ValueError(f"Requested rank is absent from the published search ({rank})")
    return pd.concat(frames, ignore_index=True)


def bands(sample, by, lower=0, upper=1):
    group = sample.groupby(list(by), sort=True, dropna=False)["score"]
    return group.agg(mean="mean", count="count").join(
        group.quantile([lower, upper]).unstack().set_axis(["low", "high"], axis=1)
    ).reset_index()


def across_tasks(search, by, skip_missing):
    keys = list(by)
    groups = search.groupby(keys, sort=True, dropna=False)
    if "weight" in search:
        weighted = search.assign(weighted_score=search["score"] * search["weight"],
                                 available=search["weight"].where(search["score"].notna(), 0))
        sums = weighted.groupby(keys, sort=True, dropna=False)[["weighted_score", "available"]].sum()
        means = (sums["weighted_score"] / sums["available"]).rename("score")
    else:
        means = groups["score"].mean()
    if not skip_missing:
        means = means.where(groups["score"].count() == groups.size())
    return means.reset_index()


def figure1(data):
    panels = []
    for benchmark in PANELS:
        panel = eligible(data[data["benchmark"] == benchmark])
        regression = benchmark.endswith("regression_medium")
        if benchmark == "numerical_regression_medium":
            panel = panel[panel[MODEL] != "HistGradientBoostingTree"]
        task_count = panel[TASK].nunique()
        panel = normalize(panel, quantile=0.5 if regression else 0.1, clip_test=regression)
        search = random_search(panel, match_tasks=True)
        mean = across_tasks(search, ("rank", MODEL, "shuffle"), skip_missing=True)
        aggregate = bands(mean, ("rank", MODEL))
        aggregate["panel"] = benchmark
        aggregate["datasets"] = task_count
        panels.append(aggregate[aggregate["rank"] <= 400])
    return pd.concat(panels, ignore_index=True)


def figure2(data):
    panel = normalize(eligible(data[data["xp"] == "high_frequencies"]), quantile=0.1)
    search = random_search(panel, (SMOOTH,), rank=60)
    subset = search[(search[TASK] != "poker") & (search[MODEL] != "MLP") & (search[SMOOTH] < 0.5)]
    return across_tasks(subset, (MODEL, "shuffle", SMOOTH), skip_missing=False)


def figure3(data):
    panel = normalize(eligible(data[data["xp"] == "show_uninformative_features"]), baseline=0.5)
    search = random_search(panel, (REMOVE, "method"), shuffles=30, default_first=False, rank=20)
    subset = search[(search[TASK] != "poker") & (search[MODEL] == "GradientBoostingTree")]
    mean = across_tasks(subset, (TASK, REMOVE, "method"), skip_missing=False)
    return bands(mean, (REMOVE, "method"), 0.1, 0.9)


def feature_sweep(data, experiment, condition, rank, shuffles, require_scores):
    panel = eligible(data[data["xp"] == experiment])
    if require_scores:
        panel = panel.dropna(subset=[VALID, TEST, MODEL])
    panel = normalize(panel, quantile=0.1)
    search = random_search(panel, (condition,), shuffles=shuffles, rank=rank)
    subset = search[~search[TASK].isin(["poker", "jannis"])]
    return across_tasks(subset, (MODEL, condition, "shuffle"), skip_missing=not require_scores)


def figure4(data):
    remove = feature_sweep(data, "remove_uninformative_features", REMOVE, 30, 30, False)
    add = feature_sweep(data, "add_uninformative_features", ADD, 20, 15, True)
    caption_add = feature_sweep(data, "add_uninformative_features", ADD, 20, 30, True)
    return (
        bands(remove, (REMOVE, MODEL)).rename(columns={REMOVE: "fraction"}),
        bands(add, (ADD, MODEL), 0.2, 0.8).rename(columns={ADD: "fraction"}).assign(
            fraction=lambda frame: frame["fraction"] - 1
        ),
        bands(caption_add, (ADD, MODEL)).rename(columns={ADD: "fraction"}).assign(
            fraction=lambda frame: frame["fraction"] - 1
        ),
    )


def figure5(data):
    results = []
    stream = r_random_stream(0)
    for fraction in (0.0, 0.5):
        panel = data[(data["xp"] == "random_rotation") & (data[REMOVE] == fraction)]
        search = random_search(normalize(eligible(panel), quantile=0.1), (ROTATE,),
                               shuffles=30, rank=20, stream=stream)
        average = across_tasks(search[search[TASK] != "poker"],
                               (MODEL, "shuffle", ROTATE), skip_missing=True)
        results.append(average.assign(removed=fraction))
    return pd.concat(results, ignore_index=True)


def analyze(number):
    if number not in range(1, 6):
        raise ValueError(f"Expected a paper figure number, got {number}")
    function = (figure1, figure2, figure3, figure4, figure5)[number - 1]
    return function(scores("benchmark" if number == 1 else "ablations"))
