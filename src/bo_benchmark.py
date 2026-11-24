"""Figures 33–34, chronological Bayesian search versus shuffled random search."""

import numpy as np
import pandas as pd

from .analysis import ALIASES, across_tasks, bands, normalize, random_search, scores
from .author_data import MODEL, TASK, TEST, TIME_PANELS, VALID


MODELS = {"Resnet", "FT Transformer", "GradientBoostingTree", "XGBoost"}


def assemble(number):
    nn, tree, nn_new, tree_new = (scores(name) for name in ("bo_nn", "bo_trees", "bo_nn_2", "bo_trees_2"))
    original = pd.concat([nn, tree], ignore_index=True)
    original = original[~original[TASK].isin(tree_new[TASK])]
    bayesian = pd.concat([original, nn_new, tree_new], ignore_index=True)
    regression = number == 34
    bayesian = bayesian[bayesian["data__regression"].eq(int(regression))].copy()
    bayesian[MODEL] = bayesian[MODEL].replace(ALIASES)
    if regression:
        bayesian[VALID] = bayesian["mean_r2_val"]
        bayesian[TEST] = bayesian["mean_r2_test"]
    data = scores("time_numerical")
    random = data[(data["benchmark"] == TIME_PANELS[int(regression)]) & data[MODEL].isin(MODELS)]
    bayesian = pd.concat([bayesian, random[random["hp"].eq("default")].assign(step=0)], ignore_index=True)
    excluded = ["isolet"] if regression else ["california", "house_16H"]
    data = pd.concat([random.assign(bo=False), bayesian.assign(bo=True)], ignore_index=True)
    data = data[data["mean_time"].notna() & ~data[TASK].isin(excluded)]
    return normalize(data, quantile=0.5 if regression else 0.1, clip_test=regression)


def chronological(data):
    keys = [MODEL, TASK]
    data = data.sort_values("step", kind="stable").rename(columns={"step": "rank"})
    # R cummax propagates a missing validation to every subsequent observation.
    groups = data.groupby(keys, sort=False)
    best = groups[VALID].cummax(skipna=False)
    first_rank = data.groupby([*keys, VALID], sort=False)["rank"].transform("min")
    data["score"] = data[TEST].where(data[VALID].eq(best) & data["rank"].eq(first_rank))
    grid = pd.MultiIndex.from_product([sorted(data[k].unique()) for k in [*keys, "rank"]],
                                      names=[*keys, "rank"]).to_frame(index=False)
    data = grid.merge(data[[*keys, "rank", "score"]], how="left", sort=False)
    data["score"] = data.groupby(keys, sort=False)["score"].ffill()
    return data.groupby([MODEL, "rank"], sort=True)["score"].mean().reset_index(name="mean")


def analyze(number):
    data = assemble(number)
    random = random_search(data[~data["bo"]], match_tasks=True)
    random = across_tasks(random, ("rank", MODEL, "shuffle"), skip_missing=True)
    summary = bands(random, (MODEL, "rank")).assign(method="Random")
    bo = chronological(data[data["bo"]]).assign(method="Bayesian")
    return pd.concat([summary, bo], ignore_index=True).query("rank <= 200").replace(
        [np.inf, -np.inf], np.nan
    ).reset_index(drop=True)
