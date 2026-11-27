"""Bounded paper pilots using the original splits, transforms and evaluators."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("TABULAR_PILOT_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
WORK = Path(os.environ.get("TABULAR_PILOT_WORK", str(ROOT / "runtime/pilot-paper25-20261125"))).resolve()
sys.path[:0] = [str(ROOT / p) for p in ("src", "notebooks/tabular", "scripts", "tools/benchmarks")]
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  WANDB_MODE="disabled", MPLBACKEND="Agg")

import paper_extensions as paper
from author_v2_session import digest, save
from paper_extensions_v3 import evaluate_capture
from tabular_workbench.engine import modules, verify_sources


def meta_for(task):
    file = os.environ.get("TABULAR_PILOT_INPUTS", str(ROOT / "reports/prepared/next-extensions-20261031/inputs.json"))
    return paper.read(file)["tasks"][task]


def verified_lookup(meta):
    import openml

    relative = Path("org/openml/www/datasets") / str(meta["dataset_id"]) / "dataset.arff"
    candidates = [ROOT / "data/openml-cache", *sorted((ROOT / "runtime/local-extension-batches").glob("*/openml-cache"))]
    if "TABULAR_PILOT_CACHE" in os.environ:
        candidates.insert(0, Path(os.environ["TABULAR_PILOT_CACHE"]))
    for cache in candidates:
        raw = cache / relative
        if raw.is_file() and hashlib.md5(raw.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == meta["md5"]:
            openml.config.set_cache_directory(str(cache))
            data = openml.datasets.get_dataset(meta["dataset_id"], download_qualities=False)
            assert (int(data.dataset_id), int(data.version), data.md5_checksum, data.default_target_attribute) == (
                meta["dataset_id"], meta["version"], meta["md5"], meta["default_target_attribute"])
            return data
    raise FileNotFoundError(f"Missing verified ARFF for task {meta['task']}")


def recipes():
    path = ROOT / "upstream/src/launch_benchmarks/launch_xps.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    return {x["name"]: x["config"] for node in parsed.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "xps" for t in node.targets)
            for x in ast.literal_eval(node.value)}


def transform(name, **overrides):
    specification = recipes()[name]
    for key, value in overrides.items():
        assert value in specification[key]["values"], (key, value)
    return {key: overrides[key] if key in overrides else rule.get("value", rule.get("values", [None])[0])
            for key, rule in specification.items()}


def fit(label, task, model, updates=None, recipe=None, device="cpu", boundary=False):
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from threadpoolctl import threadpool_limits

    meta = meta_for(task)
    out = WORK / "runs" / label
    out.mkdir(parents=True, exist_ok=False)
    previous = Path.cwd()
    os.chdir(out)
    selector_columns = []
    try:
        with (out / "worker.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), threadpool_limits(limits=1):
            training, pipeline, _, registry, _ = modules()
            cfg = paper.default_configuration(model if model != "dt" else "rf", meta, 10000)
            if model == "dt":
                cfg["model_name"] = "pilot_dt"
                cfg["model__random_state"] = 0
            if recipe is not None:
                cfg = {k: v for k, v in cfg.items() if not k.startswith("transform__")}
                cfg.update(recipe)
                for key in recipe:
                    if key.endswith("__method_name"):
                        cfg[key.replace("__method_name", "__apply_on")] = "all"
            cfg.update(updates or {})
            if cfg["model_type"] == "skorch":
                cfg["model__device"] = device
            if device == "cuda":
                assert torch.cuda.is_available()
            orig_selector = pipeline.convert_keyword_to_function["select_features_rf"]

            def select(*args, **kwargs):
                *values, cols = orig_selector(*args, **dict(kwargs, return_features=True))
                selector_columns.extend(map(int, cols))
                return tuple(values)

            context = patch.dict(pipeline.convert_keyword_to_function, {"select_features_rf": select}) if boundary else contextlib.nullcontext()
            random.seed(0)
            np.random.seed(0)
            with patch.object(paper, "lookup", verified_lookup), context:
                xx, yy, cat, ids, provenance = paper.generate(meta, cfg, 0)
            if cfg.get("transform__2__method_name") == "remove_high_frequency_from_train" and not meta["regression"]:
                converted = [y.astype(np.int64) for y in yy]
                for before, after in zip(yy, converted):
                    np.testing.assert_array_equal(before, after)
                    assert np.isin(after, (0, 1)).all()
                yy = converted
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.set_num_threads(1)
            if device == "cuda":
                torch.cuda.manual_seed_all(0)
            identifier = digest(dict(label=label, config=cfg))[:16]
            tree = DecisionTreeRegressor if meta["regression"] else DecisionTreeClassifier
            started = time.monotonic()
            with patch.dict(registry.model_keyword_dic, {"pilot_dt": tree}):
                fitted = training.train_model(0, xx[0], yy[0], cat, cfg, identifier)
            fit_seconds = time.monotonic() - started
            scores, metric, capture = evaluate_capture(training, fitted, xx, yy, cfg, identifier, meta["regression"])
            key, independent = ("r2", r2_score) if meta["regression"] else ("accuracy", accuracy_score)
            payload = {}
            split_label_hashes = {}
            for i, part in enumerate(("train", "validation", "test")):
                truth, prediction = yy[i].ravel(), scores[i].ravel()
                np.testing.assert_allclose(independent(truth, prediction), metric[part][key], atol=1e-12, rtol=1e-12)
                payload.update({"y_" + part: truth, "pred_" + part: prediction, "ids_" + part: ids[i]})
                split_label_hashes[part] = hashlib.sha256(truth.tobytes()).hexdigest()
            if boundary:
                assert xx[0].shape[1] == len(selector_columns) == 2
                axis = np.linspace(-2, 2, 101, dtype=np.float32)
                x, y = np.meshgrid(axis, axis)
                payload.update(grid_axis=axis, grid_pred=fitted.predict(np.c_[x.ravel(), y.ravel()]).reshape(x.shape))
            np.savez_compressed(out / "numeric.npz", **payload)
            trained_estimators = getattr(fitted, "n_estimators_", None)
            if trained_estimators is None and hasattr(fitted, "named_steps"):
                trained_estimators = getattr(fitted.named_steps["model"], "n_estimators_", None)
        result = dict(state="completed", label=label, task=task, model=model, device=device,
                      config=cfg, seed=0, provenance=provenance, fit_seconds=fit_seconds,
                      metrics=metric, evaluation_capture=capture, verified_metric_tolerance=1e-12,
                      split_label_hashes=split_label_hashes,
                      fitted_estimators=int(trained_estimators) if trained_estimators is not None else None,
                      selected_feature_indices=selector_columns, numeric_sha256=paper.sha(out / "numeric.npz"))
        save(out / "result.json", result)
        print(json.dumps(dict(label=label, state="completed", fit_seconds=round(fit_seconds, 2), test=metric["test"][key])), flush=True)
        return result
    except Exception as error:
        save(out / "result.json", dict(label=label, state="failed", error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        os.chdir(previous)


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("suite", choices=("features", "features-neural", "tree", "boundary", "long", "estimators", "bayes",
                                       "smoothing-priority", "importance-priority", "noise-priority",
                                       "noise-credit", "smoothing-pol", "smoothing-pol-neural",
                                       "smoothing-pol-ft", "smoothing-credit-ft", "smoothing-credit-rf",
                                       "noise-electricity-levels", "noise-credit-levels",
                                       "importance-credit-low-rank", "importance-electricity-low-rank",
                                       "smoothing-magic"))
    arg.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arg.add_argument("--resume-smoothing", action="store_true")
    spec = arg.parse_args()
    assert not spec.resume_smoothing or spec.suite == "smoothing-priority"
    assert len(verify_sources()["sha256_lf"]) == 231
    log = []
    if spec.suite == "features":
        for kept in (False, True):
            cfg = transform("uninformative_features", transform__0__num_features_to_remove=0.4,
                            transform__0__keep_removed_features=kept)
            log.append(fit("fig03-" + ("removed" if kept else "kept"), "361062", "gbt", recipe=cfg))
        for ratio in (0.2, 0.4):
            cfg = transform("remove_features", transform__0__num_features_to_remove=ratio)
            log.append(fit(f"fig04a-removed-{ratio}", "361062", "rf", recipe=cfg))
        for ratio in (1.0, 1.5):
            cfg = transform("add_features", transform__0__multiplier=ratio)
            log.append(fit(f"fig04b-multiplier-{ratio}", "361062", "gbt", recipe=cfg))
    if spec.suite == "features-neural":
        gaussianize = dict(transform__1__method_name="gaussienize", transform__1__type="quantile")
        for ratio in (0.2, 0.4):
            cfg = dict(transform("remove_features", transform__0__num_features_to_remove=ratio), **gaussianize)
            log.append(fit(f"fig04a-resnet-removed-{ratio}", "361062", "resnet", recipe=cfg, device=spec.device))
        for ratio in (1.0, 1.5):
            cfg = dict(transform("add_features", transform__0__multiplier=ratio), **gaussianize)
            log.append(fit(f"fig04b-resnet-multiplier-{ratio}", "361062", "resnet", recipe=cfg, device=spec.device))
    if spec.suite == "tree":
        for task in ("361062", "361078", "361110", "361099"):
            log.append(fit("fig01-single-tree-" + task, task, "dt"))
    if spec.suite == "boundary":
        cfg = dict(transform__0__method_name="gaussienize", transform__0__type="quantile",
                   transform__1__method_name="select_features_rf", transform__1__num_features=2)
        for m in ("rf", "mlp"):
            log.append(fit("fig19-" + m, "361060", m, recipe=cfg, boundary=True,
                           device=spec.device if m == "mlp" else "cpu"))
    if spec.suite in ("long", "estimators"):
        # One matched pair for each numeric/categorical task type. Keep all other values fixed.
        for task in ("361062", "361078", "361110", "361099"):
            if spec.suite == "long":
                choices = [(1000, 20), (40000, 100)]
            else:
                import numpy as np

                tuned = int(np.round(np.exp(np.random.RandomState(0).uniform(np.log(10.5), np.log(1000.5)))))
                choices = [(1000, 20), (tuned, None)]
            for n, patience in choices:
                updates = dict(model__max_depth=2, model__n_estimators=n, model__n_iter_no_change=patience)
                log.append(fit(f"a9-{spec.suite}-{task}-{n}", task, "gbt", updates=updates))
            assert log[-2]["provenance"]["split_hash"] == log[-1]["provenance"]["split_hash"]
    if spec.suite == "bayes":
        log.extend(run_searches())
    if spec.suite == "smoothing-priority":
        import numpy as np

        def label_hash(row):
            path = WORK / "runs" / row["label"] / "numeric.npz"
            assert paper.sha(path) == row["numeric_sha256"]
            with np.load(path, allow_pickle=False) as found:
                return tuple(hashlib.sha256(found["y_" + part].astype(np.int64).tobytes()).hexdigest()
                             for part in ("train", "validation", "test"))

        for level in (0.0, 0.05, 0.25):
            cfg = transform("smooth_target", transform__2__cov_mult=level)
            for model in ("gbt", "resnet"):
                label = f"priority-fig02-credit-{model}-{level}"
                before = WORK / "runs" / label / "result.json"
                if spec.resume_smoothing and before.is_file():
                    prior = paper.read(before)
                    if prior["state"] == "completed":
                        assert (prior["task"], prior["model"], prior["seed"]) == ("361055", model, 0)
                        assert all(prior["config"][k] == v for k, v in cfg.items())
                        log.append(prior)
                        continue
                    assert "not implemented for 'Int'" in prior["error"]
                    label += "-dtype-repair"
                log.append(fit(label, "361055", model,
                               recipe=cfg, device=spec.device if model == "resnet" else "cpu"))
        assert len({r["provenance"]["split_hash"] for r in log}) == 1
        hashes = [label_hash(r) for r in log]
        assert len({row[1:] for row in hashes}) == 1
        for offset in (0, 2, 4):
            assert hashes[offset] == hashes[offset + 1]
        assert hashes[0][0] != hashes[-1][0]
    if spec.suite == "importance-priority":
        for portion in (0.2, 0.4, 0.6, 0.8):
            for keep_removed in (False, True):
                cfg = transform("uninformative_features", transform__0__num_features_to_remove=portion,
                                transform__0__keep_removed_features=keep_removed)
                suffix = "removed" if keep_removed else "retained"
                log.append(fit(f"priority-fig03-electricity-{portion}-{suffix}", "361060", "gbt", recipe=cfg))
        assert len({r["provenance"]["split_hash"] for r in log}) == 1
        assert len({r["split_label_hashes"]["test"] for r in log}) == 1
    if spec.suite in ("importance-credit-low-rank", "importance-electricity-low-rank"):
        assert spec.device == "cpu"
        dataset, task, models = (("credit", "361055", ("gbt", "rf")) if "credit" in spec.suite
                                 else ("electricity", "361060", ("rf",)))
        for portion in (0.2, 0.4, 0.6, 0.8):
            recipe = transform("uninformative_features", transform__0__num_features_to_remove=portion,
                               transform__0__keep_removed_features=True)
            for model in models:
                log.append(fit(f"priority-fig03-{dataset}-{model}-{portion}-low-rank", task, model, recipe=recipe))
        assert len({r["provenance"]["split_hash"] for r in log}) == 1
        assert len({tuple(sorted(r["split_label_hashes"].items())) for r in log}) == 1
    if spec.suite in ("noise-priority", "noise-credit", "noise-electricity-levels", "noise-credit-levels"):
        dataset, task = ("credit", "361055") if spec.suite.startswith("noise-credit") else ("electricity", "361060")
        original_treatments = {
            "reference": transform("add_features", transform__0__multiplier=1.0),
            "add100": transform("add_features", transform__0__multiplier=2.0),
            "remove40": transform("remove_features", transform__0__num_features_to_remove=0.4),
        }
        followup_treatments = {
            "add50": transform("add_features", transform__0__multiplier=1.5),
            **{f"remove{portion}": transform("remove_features", transform__0__num_features_to_remove=portion / 100)
               for portion in (20, 60, 80)},
        }
        treatments = followup_treatments if spec.suite.endswith("-levels") else original_treatments
        models = ("resnet", "ft_transformer") if spec.device == "cuda" else ("rf", "gbt")
        for model in models:
            for treatment, recipe in treatments.items():
                if spec.device == "cuda":
                    recipe = dict(recipe, transform__1__method_name="gaussienize", transform__1__type="quantile")
                row = fit(f"priority-fig04-{dataset}-{model}-{treatment}", task, model,
                          recipe=recipe, device=spec.device)
                log.append(dict(row, treatment=treatment))
        assert len({r["provenance"]["split_hash"] for r in log}) == 1
        assert len({tuple(sorted(r["split_label_hashes"].items())) for r in log}) == 1
    smoothing = {
        "smoothing-pol": ("pol", "361062", ("gbt", "rf"), "cpu"),
        "smoothing-pol-neural": ("pol", "361062", ("resnet",), "cuda"),
        "smoothing-pol-ft": ("pol", "361062", ("ft_transformer",), "cuda"),
        "smoothing-credit-ft": ("credit", "361055", ("ft_transformer",), "cuda"),
        "smoothing-credit-rf": ("credit", "361055", ("rf",), "cpu"),
        "smoothing-magic": ("magic", "361065", ("rf", "gbt", "resnet", "ft_transformer"), "cuda"),
    }
    if spec.suite in smoothing:
        dataset, task, models, device = smoothing[spec.suite]
        assert spec.device == device
        # Match the levels of the existing paired controls for each dataset.
        levels = (0.0, 0.05, 0.25) if dataset == "credit" else (0.0, 0.05, 0.1, 0.25)
        for level in levels:
            cfg = transform("smooth_target", transform__2__cov_mult=level)
            for model in models:
                fit_device = "cpu" if model in ("rf", "gbt") else device
                row = fit(f"priority-fig02-{dataset}-{model}-{level}", task, model, recipe=cfg, device=fit_device)
                log.append(dict(row, smoothing=level))
        assert len({r["provenance"]["split_hash"] for r in log}) == 1
        assert len({(r["split_label_hashes"]["validation"], r["split_label_hashes"]["test"]) for r in log}) == 1
        for level in levels:
            assert len({r["split_label_hashes"]["train"] for r in log if r["smoothing"] == level}) == 1
        assert len({r["split_label_hashes"]["train"] for r in log}) > 1
    filename = "bayes-isolated-rng-results.json" if spec.suite == "bayes" else f"{spec.suite}-results.json"
    save(WORK / filename, dict(suite=spec.suite, rows=log, sample_only=True))


def run_searches():
    import numpy as np
    sys.path.insert(0, str(WORK / "bo-deps"))
    from sweeps import RunState, SweepRun, next_run

    # Both methods receive the same one-dimensional subset and validation metric.
    # Sweeps 0.2.0 needs two observed trials before making a GP proposal.
    parameter = "model__learning_rate"
    output = []
    for task, metric in (("361062", "accuracy"), ("361078", "r2")):
        for method in ("random", "bayes"):
            config = dict(method=method, parameters={parameter: dict(distribution="log_uniform_values", min=.01, max=.3)},
                          metric=dict(name="validation", goal="maximize"))
            history = []
            search_py_state = random.Random(0).getstate()
            search_rng = np.random.RandomState(0).get_state()
            for i in range(4):
                random.setstate(search_py_state)
                np.random.set_state(search_rng)
                proposal = next_run(config, history)
                search_py_state = random.getstate()
                search_rng = np.random.get_state()
                value = float(proposal.config[parameter]["value"])
                assert .01 <= value <= .3
                result = fit(f"a7-isolated-rng-{task}-{method}-{i}", task, "gbt", updates={parameter: value,
                             "model__max_depth": 2, "model__n_estimators": 1000, "model__n_iter_no_change": 20})
                observed = result["metrics"]["validation"][metric]
                history.append(SweepRun(state=RunState.finished, config=proposal.config,
                                        summaryMetrics=dict(validation=observed)))
                output.append(dict(result, search=method, step=i, prior_validation_trials=i,
                                   gp_proposal=(method == "bayes" and i >= 2)))
            assert len({r.config[parameter]["value"] for r in history}) > 1
    return output


if __name__ == "__main__":
    main()
