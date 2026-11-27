"""Read-only presentation/aggregation of audited remote predictions and metrics.

No model, loss or inference formula is recomputed here. Selection considers
validation only. Cross-model HPO comparison uses common complete config prefixes.
"""

from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Dict

import numpy as np
from scipy.stats import rankdata

from tabular_workbench.benchmark import CONFIGS, HPO_TASKS, SEEDS
from tabular_workbench.engine import COMMIT, MODELS
from tabular_workbench.storage import ROOT, read_json

PAPER_MODELS = ["rf", "gbt", "xgb", "mlp", "resnet", "ft_transformer"]
RESULT_PATH = ROOT / "artifacts/benchmark/verified-dashboard.json"


def failure_kind(record):
    error = record.get("error", "")
    if record["state"] == "timeout":
        return "timeout"
    if "CUDA out of memory" in error:
        return "cuda_out_of_memory"
    if "Out of range float values" in error:
        return "nonfinite_json"
    if "Nonfinite" in error:
        return "nonfinite_predictions"
    if "Worker exited" in error:
        return "worker_exit_before_result"
    return "other"


def aggregate(records):
    groups: Dict[Any, Any] = defaultdict(list)
    for record in records:
        if record["state"] != "completed":
            continue
        request = record["request"]
        key = (
            request["dataset"],
            request["models"][0],
            request["case"],
            request["benchmark_config_index"],
        )
        groups[key].append(record)
    output = []
    for (dataset, model, case, config_index), group in groups.items():
        seeds = sorted(r["request"]["seed"] for r in group)
        if seeds != SEEDS:
            continue
        task = group[0]["result"]["provenance"]["task"]
        metric = "r2" if task == "regression" else "accuracy"
        val = [r["result"]["results"][0]["metrics"]["validation"][metric] for r in group]
        test = [r["result"]["results"][0]["metrics"]["test"][metric] for r in group]
        seconds = [r["result"]["results"][0]["fit_seconds"] for r in group]
        output.append(
            {
                "dataset": dataset,
                "dataset_name": group[0]["result"]["provenance"]["name"],
                "model": model,
                "label": MODELS[model],
                "case": case,
                "config_index": config_index,
                "task": task,
                "metric": metric,
                "validation_mean": float(np.mean(val)),
                "test_mean": float(np.mean(test)),
                "test_std": float(np.std(test)),
                "fit_seconds_mean": float(np.mean(seconds)),
                "seeds": seeds,
                "run_ids": [r["id"] for r in group],
            }
        )
    return sorted(output, key=lambda r: (r["dataset"], r["model"], r["case"], r["config_index"]))


def common_hpo(aggregates):
    index = {(r["dataset"], r["model"], r["case"], r["config_index"]): r for r in aggregates}
    output, excluded = [], []
    for dataset in sorted(HPO_TASKS):
        prefix = 0
        while prefix < CONFIGS and all(
            (dataset, model, "baseline", prefix) in index for model in PAPER_MODELS
        ):
            prefix += 1
        if prefix == 0:
            excluded.append(dataset)
            continue
        chosen = []
        for model in PAPER_MODELS:
            candidates = [index[(dataset, model, "baseline", i)] for i in range(prefix)]
            # Tie-break by config index, NEVER test score.
            best = max(candidates, key=lambda r: (r["validation_mean"], -r["config_index"]))
            chosen.append(dict(best, configs_compared=prefix))
        output.append({"dataset": dataset, "common_complete_configs": prefix, "models": chosen})
    return output, excluded


def paired_defaults(aggregates):
    default = {
        (r["dataset"], r["model"]): r
        for r in aggregates
        if r["case"] == "baseline" and r["config_index"] == 0
    }
    candidates = sorted({r["dataset"] for r in aggregates})
    summaries = []
    for task in ("classification", "regression"):
        included = [
            d
            for d in candidates
            if all((d, m) in default for m in PAPER_MODELS)
            and default[(d, PAPER_MODELS[0])]["task"] == task
        ]
        if not included:
            continue
        ranks = np.asarray(
            [rankdata([-default[(d, m)]["test_mean"] for m in PAPER_MODELS]) for d in included]
        )
        summaries.append(
            {
                "task": task,
                "datasets": included,
                "dataset_count": len(included),
                "models": [
                    {"model": m, "label": MODELS[m], "mean_rank": float(ranks[:, i].mean())}
                    for i, m in enumerate(PAPER_MODELS)
                ],
                "note": "Default only, same datasets/splits/seeds; lower rank is better. Not paper HPO conclusion.",
            }
        )
    return summaries


def build_dashboard(states, audit):
    records = [r for state in states for r in state["records"]]
    counts = Counter(r["state"] for r in records)
    tasks = read_json(ROOT / "docs/TASKS.json")
    aggregated = aggregate(records)
    common, excluded = common_hpo(aggregated)
    names = {r["dataset"]: r["dataset_name"] for r in aggregated}
    completed_datasets = {r["request"]["dataset"] for r in records if r["state"] == "completed"}
    coverage = []
    for task in tasks:
        relevant = [r for r in records if r["request"]["dataset"] == task["id"]]
        complete = [r for r in relevant if r["state"] == "completed"]
        coverage.append(
            {
                "id": task["id"],
                "name": names.get(task["id"], "OpenML task " + task["id"]),
                "task": task["task"],
                "attempted": len(relevant),
                "completed": len(complete),
                "failed_or_timeout": len(relevant) - len(complete),
                "model_coverage": [
                    {
                        "model": model,
                        "complete_configs": sorted(
                            r["config_index"]
                            for r in aggregated
                            if r["dataset"] == task["id"]
                            and r["model"] == model
                            and r["case"] == "baseline"
                        ),
                    }
                    for model in MODELS
                ],
            }
        )
    return {
        "schema": 1,
        "verified": audit["passed"],
        "checked_at": audit["checked_at"],
        "upstream_commit": COMMIT,
        "audit": audit,
        "counts": {
            "attempted": len(records),
            "completed": counts["completed"],
            "failed": counts["failed"],
            "timeout": counts["timeout"],
            "datasets_attempted": len(tasks),
            "datasets_completed": len(completed_datasets),
        },
        "shards": [
            {
                "device": s["device"],
                "shard": s["shard"],
                "state": s["state"],
                "attempted": len(s["records"]),
                "completed": sum(r["state"] == "completed" for r in s["records"]),
                "seconds": s.get("notebook_elapsed_seconds"),
            }
            for s in states
        ],
        "failure_counts": dict(
            Counter(failure_kind(r) for r in records if r["state"] != "completed")
        ),
        "coverage": coverage,
        "aggregates": aggregated,
        "common_hpo": common,
        "common_hpo_excluded": excluded,
        "default_ranks": paired_defaults(aggregated),
        "warnings": [
            "Partial reproduction; not45datasets/400HPO/model.",
            "Only groups with all3seeds are aggregated; failures never replaced.",
            "Common HPO uses equal complete config prefixes and validation-only selection.",
            "Raw rotation/noise are diagnostics, not the paper's full ablation recipes.",
            "HGBT is extra; SAINT and categorical suites are not included.",
        ],
    }


@lru_cache(maxsize=2)
def _cached_dashboard(modified):
    return read_json(RESULT_PATH)


def dashboard():
    if not RESULT_PATH.exists():
        return {"verified": False, "note": "No independently verified remote evidence yet"}
    return _cached_dashboard(RESULT_PATH.stat().st_mtime_ns)
