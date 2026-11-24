"""Per-dataset ablations in Appendix A.4."""

import pandas as pd

from .analysis import bands, eligible, normalize, r_random_stream, random_search, scores
from .author_data import ADD, MODEL, REMOVE, ROTATE, SMOOTH, TASK, TEST, VALID


def panel(data, experiment):
    return eligible(data[data["xp"] == experiment])


def summarize(samples, groups, complete=False):
    result = bands(samples, groups)
    if complete:
        result = result[result["count"].eq(samples["shuffle"].nunique())]
    return result.dropna(subset=["mean"]).reset_index(drop=True)


def analyze(number, caption=False):
    frame = scores("ablations")
    if number == 18:
        data = panel(frame, "high_frequencies")
        selected = random_search(data, (SMOOTH,), rank=60)
        return selected.loc[(selected[TASK] != "poker") & (selected[MODEL] != "MLP")
                            & (selected[SMOOTH] < 0.5)].rename(columns={SMOOTH: "condition"})
    if number == 20:
        data = normalize(panel(frame, "show_uninformative_features"), baseline=0.5)
        selected = random_search(data, (REMOVE, "method"), shuffles=30,
                                 default_first=False, rank=20)
        selected = selected[(selected[TASK] != "poker") & (selected[MODEL] == "GradientBoostingTree")]
        return summarize(selected, (TASK, "method", REMOVE), complete=True).rename(
            columns={REMOVE: "condition"})
    if number == 21:
        data = panel(frame, "add_uninformative_features").dropna(subset=[VALID, TEST, MODEL])
        selected = random_search(data, (ADD,), shuffles=30 if caption else 15, rank=20)
        selected = selected[~selected[TASK].isin(["poker", "jannis"])]
        return summarize(selected, (TASK, MODEL, ADD), complete=True).assign(
            condition=lambda result: result[ADD] - 1
        ).drop(columns=ADD)
    if number == 22:
        data = panel(frame, "remove_uninformative_features")
        selected = random_search(data, (REMOVE,), shuffles=30, rank=30)
        selected = selected[~selected[TASK].isin(["poker", "jannis"])]
        return summarize(selected, (TASK, MODEL, REMOVE)).rename(columns={REMOVE: "condition"})
    if number in (23, 24):
        data = panel(frame, "random_rotation")
        stream = r_random_stream(0)
        by_fraction = {
            fraction: random_search(data[data[REMOVE].eq(fraction)], (ROTATE,),
                                    shuffles=30, rank=20, stream=stream)
            for fraction in (0.0, 0.5)
        }
        selected = by_fraction[0.0 if number == 23 else 0.5]
        selected = selected[(selected[TASK] != "poker") & selected[MODEL].isin(
            ["GradientBoostingTree", "RandomForest", "FT Transformer", "Resnet"])]
        return selected.assign(condition=~selected[ROTATE].astype(bool)).drop(columns=ROTATE)
    raise ValueError(f"Unsupported ablation figure {number}")
