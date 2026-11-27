"""Bounded partial benchmark; original model/split/train/evaluation functions.

Numerical-only suites336/337. Defaults across all tasks come first; then fixed
random configs. Selection uses outer validation only; test is never consulted.
Each fit is isolated and checkpointed; failed/censored fits remain in the ledger.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

import numpy as np

from tabular_workbench import data, runner
from tabular_workbench.engine import COMMIT, NEURAL, default_config, modules
from tabular_workbench.storage import ROOT, digest, folder, read_json, report, write_json

CPU_MODELS = ["rf", "gbt", "hgbt", "xgb"]
GPU_MODELS = ["mlp", "resnet", "ft_transformer"]
CONFIGS = 16
HPO_TASKS = {"361055", "361060", "361062", "361065", "361072", "361073", "361076", "361078"}
SEEDS = [0, 1, 2]
MAX_RUNTIME = 36000


class SnapshotKeys(dict):
    """WandB Config exposes snapshot keys; ordinary dict views cannot mutate.

    Only this container interface is adapted. The original modify_config body,
    _temp conversion and FT token-width calculation still run unchanged.
    """

    def keys(self):
        return list(super().keys())


def discover():
    import openml

    tasks = []
    for suite, task_type in ((337, "classification"), (336, "regression")):
        for task in sorted(openml.study.get_suite(suite).tasks or []):
            tasks.append({"id": str(task), "suite": suite, "task": task_type})
    return tasks


def register(tasks):
    for item in tasks:
        data.CATALOG[item["id"]] = {"suite": item["suite"], "task": item["task"]}


def sample(spec, rng):
    if "value" in spec:
        return spec["value"]
    if "values" in spec:
        index = rng.choice(len(spec["values"]), p=spec.get("probabilities"))
        return spec["values"][index]
    dist = spec["distribution"]
    if dist in ("uniform", "q_uniform"):
        value = rng.uniform(spec["min"], spec["max"])
    elif dist in ("log_uniform_values", "q_log_uniform_values"):
        value = np.exp(rng.uniform(np.log(spec["min"]), np.log(spec["max"])))
    elif dist == "log_uniform":
        value = np.exp(rng.uniform(spec["min"], spec["max"]))
    elif dist in ("normal", "q_normal"):
        value = rng.normal(spec["mu"], spec["sigma"])
    elif dist in ("log_normal", "q_log_normal"):
        value = np.exp(rng.normal(spec["mu"], spec["sigma"]))
    else:
        raise ValueError("Unimplemented upstream sampling distribution: " + dist)
    if dist.startswith("q_"):
        quantum = spec.get("q", 1)
        value = np.round(value / quantum) * quantum
        if quantum == 1:
            return int(value)
    return float(value)


def configuration(model, regression, index):
    if index == 0:
        return default_config(model, regression, "cuda" if model in NEURAL else "cpu", 300, 0)
    _, _, _, registry, upstream_runner = modules()
    seed = int(hashlib.sha256((model + str(regression) + str(index)).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    original = registry.total_config[model]["regression" if regression else "classif"]["random"]
    cfg = {key: sample(spec, rng) for key, spec in original.items()}
    cfg.update(regression=regression, data__regression=regression, data__categorical=False)
    cfg = dict(upstream_runner.modify_config(SnapshotKeys(cfg)))
    if model in NEURAL:
        cfg.update(model__device="cuda", model__max_epochs=300)
    return cfg


def protocol(tasks):
    return {
        "schema": 1,
        "paper": "NeurIPS2022 tabular benchmark",
        "upstream_commit": COMMIT,
        "scope": "Partial numerical-only336/337 benchmark, not all45datasets/full400HPO",
        "tasks": tasks,
        "cpu_models": CPU_MODELS,
        "gpu_models": GPU_MODELS,
        "configs_per_model": CONFIGS,
        "seed_repeats": SEEDS,
        "train_cap": 10000,
        "hpo_tasks": sorted(HPO_TASKS),
        "default_scope": "all35numeric tasks; HPO only8preselectedtasks",
        "holdout_cap": 50000,
        "max_epochs": 300,
        "fit_timeout_seconds": 1800,
        "cpu_versions": 5,
        "gpu_versions": 2,
        "version_seconds": MAX_RUNTIME,
        "selection": "Mean outer validation accuracy/classification or R2/regression; full3seeds required; ties configindex",
        "preprocessing": "Processed authorOpenML datasets; train-only original QuantileTransformer(normal) for neural; no transform for trees",
        "sampler": "Deterministic NumPy interpretation of original distribution specs, not identical WandB random sequence",
        "repeat_choice": "Three fixed repeats instead of original adaptive1/2/3/5; explicit partial-protocol choice",
        "diagnostics": ["baseline", "noise2x", "rotation"],
        "not_included": [
            "mixedcategorical334/335",
            "SAINT",
            "full400HPO",
            "label_smoothing/removal fullfigure recipes",
        ],
        "private_csv_or_credentials_uploaded": False,
        "monitoring": "Manual only; no schedule or auto resubmission",
    }


def jobs(tasks, device, shard):
    models, shards = (CPU_MODELS, 5) if device == "cpu" else (GPU_MODELS, 2)
    register(tasks)
    pairs = [(task, model) for task in tasks for model in models]
    # Defaults across every task/model before extended HPO; seeds synchronized.
    for index in [0, -1] + list(range(1, CONFIGS)):
        if index == -1:
            for task in tasks:
                if task["id"] not in ("361055", "361072"):
                    continue
                for model_index, model in enumerate(models):
                    if model_index % shards != shard:
                        continue
                    for case in ("noise", "rotation"):
                        for seed in SEEDS:
                            yield {
                                "dataset": task["id"],
                                "models": [model],
                                "case": case,
                                "noise_multiplier": 2,
                                "seed": seed,
                                "train_limit": 10000,
                                "holdout_limit": 50000,
                                "epochs": 300,
                                "device": device,
                                "timeout_seconds": 1800,
                                "benchmark_config_index": 0,
                            }
            continue
        for pair_index, (task, model) in enumerate(pairs):
            if pair_index % shards != shard:
                continue
            if index > 0 and task["id"] not in HPO_TASKS:
                continue
            cfg = configuration(model, task["task"] == "regression", index)
            for seed in SEEDS:
                yield {
                    "dataset": task["id"],
                    "models": [model],
                    "case": "baseline",
                    "noise_multiplier": 2,
                    "seed": seed,
                    "train_limit": 10000,
                    "holdout_limit": 50000,
                    "epochs": 300,
                    "device": device,
                    "timeout_seconds": 1800,
                    "benchmark_config": cfg,
                    "benchmark_config_index": index,
                }


def selected(records):
    groups: Dict[Any, Any] = {}
    for item in records:
        if item["state"] != "completed":
            continue
        result = item["result"]
        request, model = result["request"], result["results"][0]
        key = (
            request["dataset"],
            model["model"],
            request["case"],
            request["benchmark_config_index"],
        )
        groups.setdefault(key, []).append(result)
    choices: Dict[Any, Any] = {}
    for key, results in groups.items():
        if {r["request"]["seed"] for r in results} != set(SEEDS):
            continue
        metric = "r2" if results[0]["provenance"]["task"] == "regression" else "accuracy"
        val = float(np.mean([r["results"][0]["metrics"]["validation"][metric] for r in results]))
        candidate = {
            "dataset": key[0],
            "model": key[1],
            "case": key[2],
            "config_index": key[3],
            "validation_score": val,
            "metric": metric,
            "test_score": float(
                np.mean([r["results"][0]["metrics"]["test"][metric] for r in results])
            ),
            "run_ids": [r["id"] for r in results],
        }
        group = key[:3]
        # Selection NEVER reads candidate test_score.
        if group not in choices or (val, -key[3]) > (
            choices[group]["validation_score"],
            -choices[group]["config_index"],
        ):
            choices[group] = candidate
    return list(choices.values())


def run_batch(device, shard, tasks, seconds=MAX_RUNTIME):
    register(tasks)
    destination = ROOT / "artifacts/benchmarks" / (device + "-" + str(shard))
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "result.json"
    config = protocol(tasks)
    state = (
        read_json(path)
        if path.exists()
        else {
            "schema": 1,
            "device": device,
            "shard": shard,
            "protocol": config,
            "protocol_hash": digest(config),
            "records": [],
        }
    )
    if state["protocol_hash"] != digest(config):
        raise ValueError("Refuse checkpoint from a different protocol")
    complete = {r["spec_hash"] for r in state["records"]}
    started = time.monotonic()
    for request in jobs(tasks, device, shard):
        key = digest(request)
        if key in complete:
            continue
        remaining = seconds - (time.monotonic() - started) - 90
        if remaining < 60:
            state["state"] = "budget_stopped"
            break
        identifier = key[:16]
        run_path = folder(identifier)
        run_path.mkdir(parents=True, exist_ok=True)
        write_json(run_path / "request.json", request)
        write_json(
            run_path / "status.json", {"id": identifier, "state": "queued", "request": request}
        )
        command = [
            sys.executable,
            "-m",
            "tabular_workbench.benchmark",
            "worker",
            "--id",
            identifier,
            "--catalog",
            str(destination / "catalog.json"),
        ]
        write_json(destination / "catalog.json", tasks)
        env = os.environ.copy()
        env.update(
            WANDB_MODE="disabled",
            WANDB_ERROR_REPORTING="false",
            MPLBACKEND="Agg",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        with (run_path / "process.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
            try:
                child.wait(timeout=min(1800, remaining))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
                write_json(
                    run_path / "status.json",
                    {"id": identifier, "state": "timeout", "request": request},
                )
        result = report(identifier)
        if result["state"] in ("queued", "running"):
            result.update(
                state="failed",
                error="Worker exited before saving completed output; inspect process.log",
            )
            write_json(run_path / "status.json", result)
        record = {"spec_hash": key, "id": identifier, "state": result["state"], "request": request}
        if result["state"] == "completed":
            result["execution_location"] = (
                "kaggle"
                if os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
                or ROOT.as_posix().startswith("/kaggle/")
                else "local"
            )
            write_json(run_path / "result.json", result)
            record["result"] = result
            compact_numeric(run_path, result)
        else:
            record["error"] = result.get("error", "Incomplete fit; inspect preserved logs")
        state["records"].append(record)
        state.update(
            state="running",
            selected=selected(state["records"]),
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(path, state)
        print(
            json.dumps(
                {
                    "shard": shard,
                    "device": device,
                    "fits_attempted": len(state["records"]),
                    "last_state": record["state"],
                }
            ),
            flush=True,
        )
    else:
        state["state"] = "all_planned_fits_attempted"
    state.update(selected=selected(state["records"]), elapsed_seconds=time.monotonic() - started)
    write_json(path, state)
    return state


def compact_numeric(path, result):
    """Keep independent metric evidence; delete only this harness's bulky files.

    All requested model hashes remain in result metadata. No user upload/model
    is opened or deleted. Public X is re-fetchable via source/version/checksum.
    """
    path.resolve().relative_to((ROOT / "artifacts/runs").resolve())
    assert path.name == digest(result["request"])[:16]
    with np.load(path / "arrays.npz", allow_pickle=False) as saved:
        payload = {
            key: saved[key]
            for key in saved.files
            if key.startswith(("y_", "ids_")) or "_prediction_" in key
        }
    np.savez_compressed(path / "numeric.npz", **payload)
    result["numeric_sha256"] = hashlib.sha256((path / "numeric.npz").read_bytes()).hexdigest()
    write_json(path / "result.json", result)
    for item in path.rglob("*"):
        if item.is_file() and (item.suffix in (".pt", ".joblib") or item.name == "arrays.npz"):
            item.resolve().relative_to(path.resolve())
            item.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["worker", "run"])
    parser.add_argument("--id")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=MAX_RUNTIME)
    args = parser.parse_args()
    tasks = read_json(ROOT / args.catalog)
    register(tasks)
    if args.action == "worker":
        data.fetch(read_json(folder(args.id) / "request.json")["dataset"])
        runner.worker(args.id)
    else:
        run_batch(args.device, args.shard, tasks, args.seconds)
