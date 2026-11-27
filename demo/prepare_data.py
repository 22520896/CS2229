"""Maintainer-only export of author normalization anchors from verified cached CSV.

The app never calls this tool, downloads data, or fits a model.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis import eligible, normalize  # noqa: E402 -- Maintainer CLI bootstrap.
from demo.evidence import source_hash  # noqa: E402


def prepare():
    target = ROOT / "demo/data"
    target.mkdir(exist_ok=True)
    raw_path = ROOT / ".cache/ablations.csv.gz"
    proof = json.loads((ROOT / ".cache/ablations.json").read_bytes())
    assert sha256(raw_path.read_bytes()).hexdigest() == proof["projection_sha256"]
    assert (
        proof["source_sha256"]
        == "25314ccb2d138dae43fd23840f729f62fba0b5f5ab559aeb3727403dedb4e429"
    )
    raw = pd.read_csv(raw_path, float_precision="round_trip", low_memory=False)
    own = pd.read_csv(ROOT / "reproduce/results/mechanism_cells.csv")
    anchors = []
    for mechanism, experiment, variant in [
        ("rotation", "random_rotation", 0.0),
        ("rotation", "random_rotation", 0.5),
        ("noise_features", "add_uninformative_features", -1.0),
        ("target_smoothing", "high_frequencies", -1.0),
    ]:
        panel = eligible(raw.loc[raw.xp.eq(experiment)])
        if mechanism == "rotation":
            panel = panel.loc[panel["transform__0__num_features_to_remove"].eq(variant)]
        if mechanism == "noise_features":
            panel = panel.dropna(
                subset=["mean_val_score", "mean_test_score", "model_name"]
            )
        selected = own.loc[
            own.mechanism.eq(mechanism) & own.reference_grid & own.full_repeats
        ]
        names = set(selected.dataset)
        original = set(panel["data__keyword"]) - (
            {"poker", "jannis"} if mechanism == "noise_features" else {"poker"}
        )
        assert names == original
        grouped = panel.groupby("data__keyword").mean_test_score.agg(
            q10=lambda s: s.quantile(0.1), maximum="max"
        )
        normalized = normalize(panel, quantile=0.1)
        expected = (panel.mean_test_score - panel["data__keyword"].map(grouped.q10)) / (
            panel["data__keyword"].map(grouped.maximum)
            - panel["data__keyword"].map(grouped.q10)
        )
        assert (expected - normalized.mean_test_score).abs().max() < 1e-12
        for name in sorted(names):
            row = grouped.loc[name]
            anchors.append(
                dict(
                    mechanism=mechanism,
                    variant=variant,
                    dataset=name,
                    q10=row.q10,
                    maximum=row.maximum,
                )
            )
    assert len(anchors) == 57
    pd.DataFrame(anchors).to_csv(target / "author_anchors.csv", index=False)
    (target / "normalization.json").write_text(
        json.dumps(
            dict(
                source_sha256=proof["source_sha256"],
                projection_sha256=proof["projection_sha256"],
                formula="(score - author_dataset_q10) / (author_dataset_max - author_dataset_q10)",
                same_author_anchors_for_both_series=True,
                clipping=False,
                tolerance=1e-12,
                original_analysis_function="src.analysis.normalize",
                original_formula_parity=True,
                model_selection_changed=False,
                model_fit_calls=0,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_manifest()


def write_manifest():
    paths = [
        "reproduce/results/" + p
        for p in [
            "mechanism_cells.csv",
            "coverage.csv",
            "balanced_scores.csv",
            "figure_inventory.csv",
            "sample_fits.json",
            "source_audits.json",
            "run_status.json",
        ]
    ]
    paths += [
        "reproduce/results/numeric/" + m + ".npz"
        for m in ["noise_features", "rotation", "target_smoothing"]
    ]
    paths += [
        "reproduce/configs/" + p
        for p in ["published_conditions.json", "protocol.json", "source_versions.json"]
    ]
    paths += [
        "figures/" + p
        for p in [
            "figure_01.csv",
            "figure_02.csv",
            "figure_04_4b_caption.csv",
            "figure_05.csv",
            "figure_18.csv",
            "figure_21_caption.csv",
            "figure_23.csv",
            "figure_24.csv",
        ]
    ]
    paths += [
        "evidence/" + p
        for p in [
            "v2_benchmark.csv",
            "appendix_datasets.csv",
            "appendix_hpo.csv",
            "a4_coefficients.csv",
            "python_r_parity.json",
            "appendix_parity.json",
            "published_parity.json",
            "project_snapshot.json",
        ]
    ]
    paths += [
        "demo/data/" + p
        for p in [
            "author_anchors.csv",
            "normalization.json",
            "execution.json",
            "progress_targets.json",
        ]
    ]
    manifest = dict(
        schema=2,
        hash_policy="csv/json CRLF-to-LF; other files byte-exact",
        result_snapshot="2026-11-28T02:00:00+07:00",
        training_state="stopped_by_user",
        demo_base_repository="https://github.com/LoylP/tabular-trees",
        demo_base_commit="e9fce1eba2fb498b9571b33a331832a3eef63626",
        results_repository_commit="d78cfd6a9e35af50862b2c37fc1c62e52ae0ee48",
        files={p: source_hash(ROOT / p) for p in paths},
    )
    (ROOT / "demo/data/manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    prepare()
