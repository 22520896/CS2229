"""Missing paper branches through direct immutable author entrypoints.

SAINT uses tab_survey's actual factory/fit/predict, not the skorch workbench.
Categorical and50K use generate_dataset and original train/evaluate unchanged.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "notebooks/tabular")]
from author_v2_session import digest, save  # noqa:E402

TRACK = "author-paper-extensions-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_configuration(model, meta, train_cap):
    from tabular_workbench.engine import modules

    _, _, _, registry, launcher = modules()
    assert model in ("rf", "gbt", "xgb", "mlp", "resnet", "ft_transformer", "saint")
    specification = registry.total_config[model]["regression" if meta["regression"] else "classif"][
        "default"
    ]
    cfg = {k: v["value"] if "value" in v else v["values"][0] for k, v in specification.items()}
    cfg.update(
        regression=meta["regression"],
        data__regression=meta["regression"],
        data__categorical=meta["categorical"],
        data__keyword=int(meta["task"]),
        data__method_name="openml_no_transform",
        n_iter="auto",
        train_prop=0.7,
        val_test_prop=0.3,
        max_train_samples=train_cap,
        max_val_samples=50000,
        max_test_samples=50000,
    )
    if cfg["model_type"] == "skorch":
        cfg.update(model__device="cuda", model__max_epochs=300)
    # Match the original dictionary-mutation-safe iteration used by existing
    # sampling adapter; values are still set by the author's modify_config.
    from tabular_workbench.benchmark import SnapshotKeys

    return dict(launcher.modify_config(SnapshotKeys(cfg)))


def lookup(meta):
    import numpy as np
    import openml

    if meta["task"] == "361069":
        candidates = [
            p
            for p in Path("/kaggle/input").glob("**/dataset.npz")
            if sha(p) == meta["private_input_npz_sha256"]
        ]
        assert len(candidates) == 1
        with np.load(candidates[0], allow_pickle=False) as cache:
            x, y = cache["x"], cache["y"]
        assert x.shape == (940160, 24) and y.shape == (940160,)
        return SimpleNamespace(
            default_target_attribute="public-cache-target",
            get_data=lambda **kw: (x.copy(), y.copy(), None, []),
        )
    openml.config.set_cache_directory("/kaggle/temp/extension-openml-cache")
    dataset = openml.datasets.get_dataset(meta["dataset_id"], download_qualities=False)
    assert int(dataset.dataset_id) == meta["dataset_id"] and int(dataset.version) == meta["version"]
    if meta.get("md5"):
        assert dataset.md5_checksum == meta["md5"]
    return dataset


def generate(meta, config, seed):
    """Original data function and split/transform RNG, with only transport bound."""
    import numpy as np
    import openml

    from tabular_workbench.engine import modules

    _, pipeline, _, _, _ = modules()
    dataset = lookup(meta)
    x, y, categories, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if not meta["categorical"]:
        assert categories is None or not np.asarray(categories).any()
    task = SimpleNamespace(get_dataset=lambda: dataset)
    rng = np.random.RandomState(seed)
    rng.randn(1)
    with patch.object(openml.tasks, "get_task", lambda *a, **k: task):
        result = pipeline.generate_dataset(config, rng)
    xx, yy, cat = list(result[:3]), list(result[3:6]), result[6]
    # Exact author launch boundary: only skorch regression is cast tofloat32.
    xx = [a.astype(np.float32) for a in xx]
    yy = [
        a.astype(np.float32).reshape(-1, 1)
        if config["model_type"] == "skorch" and meta["regression"]
        else a.reshape(-1)
        for a in yy
    ]
    ids_rng = np.random.RandomState(seed)
    ids_rng.randn(1)
    split = pipeline.data_to_train_test(np.arange(len(x)).reshape(-1, 1), y, config, ids_rng)
    ids = [a.reshape(-1).astype(np.int64) for a in split[:3]]
    assert [len(v) for v in ids] == [len(v) for v in xx]
    assert len(set(np.concatenate(ids))) == sum(map(len, ids))
    return (
        xx,
        yy,
        cat,
        ids,
        dict(
            dataset_id=meta["dataset_id"],
            version=meta["version"],
            rows=len(x),
            features=x.shape[1],
            categorical_columns=int(np.asarray(categories).sum()) if categories is not None else 0,
            raw_x_dtype=str(x.dtype),
            raw_y_dtype=str(y.dtype),
            split_hash=digest([a.tolist() for a in ids]),
        ),
    )


def auto_seeds(meta, config):
    """Compile and execute the verified author's inline auto-repeat branch."""
    import ast
    import copy

    import numpy as np
    import openml

    from tabular_workbench.engine import modules

    dataset = lookup(meta)
    pipeline = modules()[1]
    source = ROOT / "upstream/src/run_experiment.py"
    choices = [
        n
        for n in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Subscript)
        and isinstance(n.test.left.value, ast.Name)
        and n.test.left.value.id == "config"
        and "n_iter" in ast.dump(n.test.left)
    ]
    assert len(choices) == 1
    body = ast.Module(body=[copy.deepcopy(choices[0])], type_ignores=[])
    ast.fix_missing_locations(body)
    namespace = dict(config=config, np=np, generate_dataset=pipeline.generate_dataset)
    with patch.object(
        openml.tasks, "get_task", lambda *a, **k: SimpleNamespace(get_dataset=lambda: dataset)
    ):
        exec(compile(body, str(source), "exec"), namespace)
    assert namespace["n_iter"] in (1, 2, 3, 5)
    return list(range(namespace["n_iter"]))


def worker(job_path, out):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from tabular_workbench.engine import modules, verify_sources
    from tabular_workbench.runner import metrics

    job = read(job_path)
    assert job["execution_track"] == TRACK and job["seed"] in range(5)
    assert job["train_cap"] in (10000, 50000) and job["timeout_seconds"] == 1800
    baseline = verify_sources()
    path = Path(out)
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = default_configuration(job["model"], job["metadata"], job["train_cap"])
        if "configuration" in job:
            # Use already selected original literals; identity pinned in job.
            assert set(config) == set(job["configuration"])
            config = job["configuration"]
        xx, yy, cat, ids, provenance = generate(job["metadata"], config, job["seed"])
        assert len(xx[0]) <= job["train_cap"] and len(xx[1]) <= 50000 and len(xx[2]) <= 50000
        training = modules()[0]
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        torch.set_num_threads(1)
        if job["device"] == "cuda":
            assert torch.cuda.is_available(), "No silent CPU fallback"
            torch.cuda.manual_seed_all(job["seed"])
        model_id = digest(job)[:16]
        with threadpool_limits(limits=1):
            fitted = training.train_model(job["seed"], xx[0], yy[0], cat, config, model_id)
        # Original evaluator reloads best skorch checkpoint. SAINT.fit already
        # reloads/deletes its best checkpoint in the original implementation.
        scores = training.evaluate_model(
            fitted,
            xx[0],
            yy[0],
            xx[1],
            yy[1],
            xx[2],
            yy[2],
            config,
            model_id,
            return_r2=job["metadata"]["regression"],
            delete_checkpoint=False,
        )
        predictions = [np.asarray(fitted.predict(a)).reshape(-1) for a in xx]
        measured = {}
        numeric = {}
        for i, part in enumerate(("train", "validation", "test")):
            y = yy[i].reshape(-1)
            measured[part] = metrics(y, predictions[i], job["metadata"]["regression"])
            primary = "r2" if job["metadata"]["regression"] else "accuracy"
            np.testing.assert_allclose(measured[part][primary], scores[i], rtol=1e-12, atol=1e-12)
            numeric.update(
                {"y_" + part: y, "prediction_" + part: predictions[i], "ids_" + part: ids[i]}
            )
        np.savez_compressed(path / "numeric.npz", **numeric)
        save(
            path / "result.json",
            dict(
                state="completed",
                job=job,
                metrics=measured,
                provenance=provenance,
                config=config,
                source_commit=baseline["commit"],
                source_modified=False,
                numeric_sha256=sha(path / "numeric.npz"),
                elapsed_seconds=time.monotonic() - started,
                model_type=config["model_type"],
                upstream_metric_parity=True,
                warnings=[
                    "Default configuration coverage only;not full HPO or paper",
                    "Fixed training RNG/thread1 disclosed",
                    "Original data/fit/evaluate bodies used directly",
                ],
            ),
        )
    except Exception as error:
        import traceback

        (path / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        save(
            path / "result.json",
            dict(
                state="failed",
                job=job,
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
                elapsed_seconds=time.monotonic() - started,
                source_modified=False,
            ),
        )


def checkpoint(state, destination, output):
    save(output / "paper-extensions.json", state)
    target = output / "paper-extensions-evidence.zip"
    temporary = target.with_suffix(".tmp")
    files = {}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:

        def add(name, raw):
            archive.writestr(name, raw)
            files[name] = hashlib.sha256(raw).hexdigest()

        add("session.json", json.dumps(state, allow_nan=False).encode())
        for r in state["records"]:
            if r["state"] == "completed":
                source = destination / r["id"] / "numeric.npz"
                assert sha(source) == r["result"]["numeric_sha256"]
                add(r["id"] + "/numeric.npz", source.read_bytes())
        archive.writestr(
            "manifest.json",
            json.dumps(
                dict(
                    files=files,
                    raw_features=False,
                    model_pickle=False,
                    credentials=False,
                    plan_hash=state["plan_hash"],
                )
            ),
        )
    os.replace(temporary, target)


def run(plan_path, output, seconds):
    import subprocess

    from tabular_workbench.continuation import terminate_owned

    plan = read(plan_path)
    assert plan["execution_track"] == TRACK and 0 < seconds <= 36000
    output.mkdir(parents=True, exist_ok=True)
    destination = ROOT / "extension-runs"
    destination.mkdir(exist_ok=False)
    state = dict(
        execution_track=TRACK,
        plan_hash=digest(plan),
        state="running",
        records=[],
        completed=0,
        unstarted=[],
        maximum_seconds=36000,
        model_body_modified=False,
    )
    started = time.monotonic()
    checkpoint(state, destination, output)
    for case_index, specification in enumerate(plan["cases"]):
        if seconds - (time.monotonic() - started) < 1890:
            state["unstarted_cases"] = plan["cases"][case_index:]
            break
        meta = plan["inputs"][specification["task"]]
        try:
            config = default_configuration(specification["model"], meta, specification["train_cap"])
            seeds = auto_seeds(meta, config)
        except Exception as error:
            state.setdefault("input_failures", []).append(
                dict(case=specification, error_type=type(error).__name__, error=str(error))
            )
            checkpoint(state, destination, output)
            continue
        for seed in seeds:
            job = dict(
                specification,
                metadata=meta,
                seed=seed,
                configuration=config,
                execution_track=TRACK,
                timeout_seconds=1800,
                required_seeds=seeds,
            )
            identifier = digest(job)[:16]
            if seconds - (time.monotonic() - started) < 1890:
                state["unstarted"].append(job)
                continue
            request = destination / (identifier + ".json")
            save(request, job)
            with (destination / (identifier + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__)),
                        "worker",
                        "--plan",
                        str(request),
                        "--output",
                        str(destination / identifier),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                )
                timed_out = False
                try:
                    process.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    terminate_owned(process)
                    timed_out = True
            result_path = destination / identifier / "result.json"
            result = (
                read(result_path)
                if result_path.exists() and not timed_out
                else dict(
                    state="timeout" if timed_out else "failed",
                    job=job,
                    error_type="FitBudget1800s" if timed_out else "WorkerExitedBeforeFinalOutput",
                )
            )
            state["records"].append(dict(id=identifier, state=result["state"], result=result))
            state["completed"] = sum(r["state"] == "completed" for r in state["records"])
            checkpoint(state, destination, output)
    state.update(
        state="budget_stopped"
        if state["unstarted"] or state.get("unstarted_cases")
        else "all_cases_attempted",
        elapsed_seconds=time.monotonic() - started,
    )
    checkpoint(state, destination, output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("worker", "run"))
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=36000)
    a = p.parse_args()
    worker(a.plan, a.output) if a.action == "worker" else run(a.plan, a.output, a.seconds)
