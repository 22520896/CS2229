"""Appendix A.4 regressions on validation-selected, normalized test scores."""

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import stdtr

from .analysis import eligible, normalize, r_random_stream, random_search
from .author_data import ADD, MODEL, REMOVE, SMOOTH, ROTATE, TASK, TEST, VALID


SWEEPS = (
    ("smoothing", "high_frequencies", SMOOTH, "lengthscale", 60, 15),
    ("remove", "remove_uninformative_features", REMOVE, "prop_removed", 30, 30),
    ("add", "add_uninformative_features", ADD, "prop_added", 20, 15),
)
VALUE_COLUMNS = ("estimate", "std_error", "statistic", "p_value")
META_COLUMNS = ("nobs", "residual_df", "dataset_baseline", "model_baseline")


def design(data, slope, separate_effect=None):
    """Treatment contrasts and the interactions in the paper's lm formulas."""
    datasets = sorted(data[TASK].unique())
    models = sorted(data[MODEL].unique())
    slope_values = data[slope].to_numpy(float)
    terms = {"(Intercept)": np.ones(len(data))}
    terms.update({f"dataset{name}": data[TASK].eq(name) for name in datasets[1:]})
    model_terms = {f"model_name{name}": data[MODEL].eq(name) for name in models[1:]}
    terms.update(model_terms)
    terms[slope] = slope_values
    if separate_effect is not None:
        terms[separate_effect] = data[separate_effect]
    terms.update({f"{name}:{slope}": contrast * slope_values
                  for name, contrast in model_terms.items()})
    if separate_effect is not None:
        terms[f"{slope}:{separate_effect}"] = slope_values * data[separate_effect]
    return pd.DataFrame(terms).to_numpy(float), list(terms), (datasets[0], models[0])


def fit(data, slope, separate_effect=None):
    data = data.dropna(subset=["score"]).reset_index(drop=True)
    matrix, names, baselines = design(data, slope, separate_effect)
    target = data["score"].to_numpy(float)
    q, r = np.linalg.qr(matrix, mode="reduced")
    if np.linalg.matrix_rank(r) != r.shape[1]:
        raise ValueError("A.4 design is rank deficient")
    estimate = solve_triangular(r, q.T @ target)
    residual = target - matrix @ estimate
    df = len(data) - len(estimate)
    inverse_r = solve_triangular(r, np.eye(r.shape[0]))
    std_error = np.sqrt((residual @ residual / df) * np.sum(inverse_r ** 2, axis=1))
    statistic = estimate / std_error
    result = pd.DataFrame(dict(term=names, estimate=estimate, std_error=std_error,
                               statistic=statistic, p_value=2 * stdtr(df, -np.abs(statistic)),
                               nobs=len(data), residual_df=df,
                               dataset_baseline=baselines[0], model_baseline=baselines[1]))
    return result[~result["term"].str.startswith("dataset")].reset_index(drop=True)


def analyze(data):
    results = []
    for label, experiment, condition, slope, rank, shuffles in SWEEPS:
        panel = eligible(data[data["xp"] == experiment])
        if label == "add":
            panel = panel.dropna(subset=[VALID, TEST, MODEL])
        selected = random_search(normalize(panel, quantile=0.1), (condition,),
                                 shuffles=shuffles, rank=rank)
        if label == "smoothing":
            selected = selected[(selected[TASK] != "poker") & (selected[MODEL] != "MLP")]
        else:
            selected = selected[~selected[TASK].isin(["poker", "jannis"])]
        result = fit(selected.rename(columns={condition: slope}), slope)
        results.append(result.assign(analysis=label))

    rotation_data = data[data["xp"] == "random_rotation"]
    stream = r_random_stream(0)
    samples = []
    for fraction in (0.0, 0.5):
        panel = eligible(rotation_data[rotation_data[REMOVE] == fraction])
        selected = random_search(normalize(panel, quantile=0.1), (ROTATE,), shuffles=30,
                                 rank=20, stream=stream)
        samples.append(selected[selected[TASK] != "poker"].assign(
            rotationTRUE=lambda frame: ~frame[ROTATE].astype(bool),
            features_removedTRUE=fraction > 0,
        ))
    result = fit(pd.concat(samples, ignore_index=True), "rotationTRUE", "features_removedTRUE")
    return pd.concat([*results, result.assign(analysis="rotation")], ignore_index=True)


def parity(actual, expected):
    keys = ["analysis", "term"]
    actual = actual.set_index(keys, verify_integrity=True).sort_index()
    expected = expected.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(actual[list(META_COLUMNS)], expected[list(META_COLUMNS)])
    np.testing.assert_allclose(actual[list(VALUE_COLUMNS)], expected[list(VALUE_COLUMNS)],
                               rtol=1e-12, atol=1e-12)
    errors = (actual[list(VALUE_COLUMNS)] - expected[list(VALUE_COLUMNS)]).abs()
    return {"coefficients": len(actual), "max_absolute_error": errors.max().to_dict(),
            "atol": 1e-12, "rtol": 1e-12}
