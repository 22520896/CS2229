"""Recompute Figures 39–40 from archived fixed and tuned runs."""

import numpy as np
import pandas as pd

from .analysis import across_tasks, bands, eligible, normalize, random_search, scores
from .author_data import MODEL, TASK, TIME_PANELS


MODELS = ("GradientBoostingTree", "RandomForest", "XGBoost")
GROUPS = {39: ("numerical", TIME_PANELS[:2], (15, 19)),
          40: ("categorical", TIME_PANELS[2:], (7, 13))}


def analyze(number):
    kind, benchmarks, expected_counts = GROUPS[number]
    fixed = eligible(scores(f"time_{kind}"))
    tuned = eligible(scores(f"tuned_{kind}"))
    output = []
    for panel, benchmark, expected in zip(("classification", "regression"), benchmarks, expected_counts):
        lhs = fixed[fixed["benchmark"].eq(benchmark) & fixed[MODEL].isin(MODELS)]
        rhs = tuned[tuned["benchmark"].eq(benchmark) & tuned[MODEL].isin(MODELS)]
        if number == 39 and panel == "regression":
            lhs, rhs = lhs[lhs[TASK].ne("isolet")], rhs[rhs[TASK].ne("isolet")]
        if lhs[TASK].nunique() != expected or set(lhs[TASK]) != set(rhs[TASK]):
            raise ValueError(f"Author cohorts differ for {benchmark}")
        inputs = pd.concat([lhs, rhs.assign(model_name=rhs[MODEL] + "_tune")], ignore_index=True)
        regression = panel == "regression"
        inputs = normalize(inputs, quantile=.5 if regression else .1, clip_test=regression)
        selected = random_search(inputs, match_tasks=True)
        aggregate = across_tasks(selected, ("rank", MODEL, "shuffle"), skip_missing=True)
        output.append(bands(aggregate, (MODEL, "rank")).assign(panel=panel, datasets=expected))
    return pd.concat(output, ignore_index=True)


def compare_published_defaults(analysis, record):
    """Compare seed-independent rank-zero points with calibrated PDF geometry."""
    rows = []
    for panel, values in record["panels"].items():
        defaults = analysis.loc[(analysis["rank"] == 0) & (analysis["panel"] == panel)].set_index(MODEL)
        for model, layers in values["series"].items():
            if len(layers["default"]) != 1:
                raise ValueError(f"Invalid published default for {panel} {model}")
            level = float(np.mean(np.asarray(layers["default"][0])[:, 1]))
            computed = float(defaults.loc[model, "mean"])
            rows.append({"panel": panel, MODEL: model, "recomputed": computed,
                         "pdf_coordinate": level, "absolute_difference": abs(computed - level)})
    return pd.DataFrame(rows)
