"""Direct-author cases with predictions captured at the evaluator's real stages.

The frozen v1 worker remains reproducible. No model/evaluator body is rewritten.
This adapter records predict() outputs already requested by the original code.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import paper_extensions as old
from author_v2_session import save

ROOT, TRACK, read, sha, digest = old.ROOT, old.TRACK, old.read, old.sha, old.digest


def evaluate_capture(training, fitted, xx, yy, config, model_id, regression):
    """Keep the original train-before/best-validation/test score semantics."""
    import numpy as np

    from tabular_workbench.runner import metrics

    calls = {id(x): [] for x in xx}
    original_predict = fitted.predict

    def capture(x, *args, **kwargs):
        assert id(x) in calls, "Unexpected evaluator input; do not guess its split"
        prediction = original_predict(x, *args, **kwargs)
        calls[id(x)].append(np.asarray(prediction).reshape(-1).copy())
        return prediction

    with patch.object(fitted, "predict", capture):
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
            return_r2=regression,
            delete_checkpoint=False,
        )
    expected_calls = 2 if config["model_type"] == "skorch" else 1
    assert all(len(calls[id(x)]) == expected_calls for x in xx)
    predictions = [calls[id(xx[0])][0], calls[id(xx[1])][-1], calls[id(xx[2])][-1]]
    measured = {}
    for index, part in enumerate(("train", "validation", "test")):
        measured[part] = metrics(yy[index].reshape(-1), predictions[index], regression)
        np.testing.assert_allclose(
            measured[part]["r2" if regression else "accuracy"],
            scores[index],
            rtol=1e-12,
            atol=1e-12,
        )
    proof = dict(
        original_evaluator_called=True,
        additional_predict_calls=0,
        train_prediction_call=0,
        validation_prediction_call=expected_calls - 1,
        test_prediction_call=expected_calls - 1,
        calls_per_split=expected_calls,
        original_scores=[float(v) for v in scores],
        score_semantics="Original train before reload;validation/test after reload for skorch",
    )
    return predictions, measured, proof


def worker(job_path, output):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from tabular_workbench.engine import modules, verify_sources

    job = read(job_path)
    assert job["execution_track"] == TRACK and job["seed"] in job["required_seeds"]
    assert job["train_cap"] in (10000, 50000) and job["timeout_seconds"] == 1800
    baseline = verify_sources()
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = job["configuration"]
        assert config["data__keyword"] == int(job["task"])
        assert config["max_train_samples"] == job["train_cap"]
        xx, yy, cat, ids, provenance = old.generate(job["metadata"], config, job["seed"])
        training = modules()[0]
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        torch.set_num_threads(1)
        if job["device"] == "cuda":
            assert torch.cuda.is_available(), "No silent CPU fallback"
            torch.cuda.manual_seed_all(job["seed"])
        identifier = digest(job)[:16]
        with threadpool_limits(limits=1):
            fitted = training.train_model(job["seed"], xx[0], yy[0], cat, config, identifier)
            predictions, measured, capture = evaluate_capture(
                training, fitted, xx, yy, config, identifier, job["metadata"]["regression"]
            )
        numeric = {}
        for index, part in enumerate(("train", "validation", "test")):
            numeric.update(
                {
                    "y_" + part: yy[index].reshape(-1),
                    "prediction_" + part: predictions[index],
                    "ids_" + part: ids[index],
                }
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
                evaluation_capture=capture,
                capture_adapter_sha256=sha(Path(__file__)),
                warnings=[
                    "Fixed training RNG/thread1 disclosed",
                    "Original data/fit/evaluate bodies",
                    "Frozen NumPy registry draws are not historical WandB sequence",
                ],
            ),
        )
    except Exception as error:
        import traceback

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


def run(plan_path, output, seconds):
    from tabular_workbench.continuation import terminate_owned

    plan = read(plan_path)
    assert plan["execution_track"] == TRACK and plan["capture_revision"] == 3
    assert 0 < seconds <= 36000
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
        capture_revision=3,
        fit_checkpoint_resume_supported=False,
    )
    started = time.monotonic()
    old.checkpoint(state, destination, output)
    for index, case in enumerate(plan["cases"]):
        if seconds - (time.monotonic() - started) < 1890:
            state["unstarted_cases"] = plan["cases"][index:]
            break
        meta = plan["inputs"][case["task"]]
        config = plan["configurations"][digest(case)]
        try:
            seeds = old.auto_seeds(meta, config)
        except Exception as error:
            state.setdefault("input_failures", []).append(
                dict(case=case, error_type=type(error).__name__, error=str(error))
            )
            old.checkpoint(state, destination, output)
            continue
        for seed in seeds:
            job = dict(
                case,
                metadata=meta,
                seed=seed,
                execution_track=TRACK,
                timeout_seconds=1800,
                required_seeds=seeds,
                configuration=config,
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
            target = destination / identifier / "result.json"
            result = (
                read(target)
                if target.exists() and not timed_out
                else dict(
                    state="timeout" if timed_out else "failed",
                    job=job,
                    error_type="FitBudget1800s" if timed_out else "WorkerExitedBeforeFinalOutput",
                )
            )
            state["records"].append(dict(id=identifier, state=result["state"], result=result))
            state["completed"] = sum(r["state"] == "completed" for r in state["records"])
            old.checkpoint(state, destination, output)
    state.update(
        state="budget_stopped"
        if state["unstarted"] or state.get("unstarted_cases")
        else "all_cases_attempted",
        elapsed_seconds=time.monotonic() - started,
    )
    old.checkpoint(state, destination, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=36000)
    args = parser.parse_args()
    if args.action == "worker":
        worker(args.plan, args.output)
    else:
        run(args.plan, args.output, args.seconds)
