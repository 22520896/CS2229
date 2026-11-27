"""Maintainer-only extraction of paper targets from verified published ablations.

No training, and never called by the web app. Targets are the ORIGINAL cohorts and
condition levels, crossed with the source analysis HPO rank. They are NOT our queue.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from src import analysis as A  # noqa: E402
from src.author_data import ADD, REMOVE, ROTATE, SMOOTH, MODEL, TASK  # noqa: E402


def prepare():
    path = ROOT / ".cache/ablations.csv.gz"
    proof = json.loads((ROOT / ".cache/ablations.json").read_bytes())
    assert (
        proof["source_sha256"]
        == "25314ccb2d138dae43fd23840f729f62fba0b5f5ab559aeb3727403dedb4e429"
    )
    assert sha256(path.read_bytes()).hexdigest() == proof["projection_sha256"]
    raw = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    targets = {}
    for figure, experiment, rank, columns in [
        ("Figure 2", "high_frequencies", 60, [SMOOTH]),
        ("Figure 3", "show_uninformative_features", 20, [REMOVE, "method"]),
        ("Figure 4a", "remove_uninformative_features", 30, [REMOVE]),
        ("Figure 4b", "add_uninformative_features", 20, [ADD]),
        ("Figure 5", "random_rotation", 20, [REMOVE, ROTATE]),
    ]:
        panel = A.eligible(raw.loc[raw.xp.eq(experiment)])
        excluded = (
            ["poker", "jannis"] if figure in {"Figure 4a", "Figure 4b"} else ["poker"]
        )
        panel = panel.loc[~panel[TASK].isin(excluded)]
        if figure == "Figure 2":
            panel = panel.loc[panel[SMOOTH].lt(0.5) & panel[MODEL].ne("MLP")]
        if figure == "Figure 3":
            panel = panel.loc[panel[MODEL].eq("GradientBoostingTree")]
        if figure == "Figure 4b":
            panel = panel.dropna(subset=[A.VALID, A.TEST, MODEL])
        # Use the source cohort/condition grid, including combinations missing in
        # historical output; this does not shrink a denominator to successful rows.
        conditions = []
        for condition in (
            panel[columns].drop_duplicates().itertuples(index=False, name=None)
        ):
            if figure == "Figure 2":
                mapped = [float(condition[0]), "robust"]
            elif figure == "Figure 3":
                mapped = [float(condition[0]), condition[1] == "keep"]
            elif figure in {"Figure 4a", "Figure 4b"}:
                mapped = (
                    [float(condition[0]), False]
                    if figure == "Figure 4a"
                    else [float(condition[0])]
                )
            else:
                mapped = [float(condition[0]), bool(condition[1])]
            conditions.append(mapped)
        targets[figure] = {
            "experiment": experiment,
            "hpo_rank": rank,
            "datasets": sorted(panel[TASK].unique()),
            "models": sorted(panel[MODEL].unique()),
            "conditions": sorted(conditions, key=lambda x: json.dumps(x)),
            "source": "src.analysis.figure"
            + {
                "Figure 2": "2",
                "Figure 3": "3",
                "Figure 4a": "4",
                "Figure 4b": "4",
                "Figure 5": "5",
            }[figure],
        }
    result = {
        "schema": 1,
        "source_sha256": proof["source_sha256"],
        "projection_sha256": proof["projection_sha256"],
        "formula": "100 * verified full-repeat configurations / (source dataset-model-condition cells * HPO rank)",
        "interpretation": "Approximate configuration-work coverage, not protocol parity, runtime or overall paper completion. Source analysis rank is an approximate HPO budget; default/rank gaps are not replayed.",
        "targets": targets,
    }
    (ROOT / "demo/data/progress_targets.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    prepare()
