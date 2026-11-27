"""Offline integrity and numeric checks;never imports a model or calls fit()."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((BASE / name).read_bytes())


def sample_metrics(sample, base=BASE):
    path = base / sample["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["numeric_sha256"]
    ids = []
    with np.load(path, allow_pickle=False) as pack:
        assert set(pack.files) == {
            prefix + part
            for prefix in ["y_", "pred_", "ids_"]
            for part in ["train", "validation", "test"]
        }
        for part in ["train", "validation", "test"]:
            truth, prediction, rows = (
                pack[prefix + part] for prefix in ["y_", "pred_", "ids_"]
            )
            assert truth.ndim == prediction.ndim == rows.ndim == 1
            assert (
                len(truth) == len(prediction) == len(rows) and rows.dtype.kind in "iu"
            )
            assert np.isin(truth, [0, 1]).all() and np.isin(prediction, [0, 1]).all()
            measured = float(np.mean(truth == prediction))
            np.testing.assert_allclose(
                measured, sample["metrics"][part]["accuracy"], rtol=1e-12, atol=1e-12
            )
            confusion = np.bincount(
                truth.astype(int) * 2 + prediction.astype(int), minlength=4
            ).reshape(2, 2)
            np.testing.assert_array_equal(
                confusion, sample["metrics"][part]["confusion_matrix"]
            )
            ids.extend(rows.tolist())
    assert len(ids) == len(set(ids))
    return True


def verify(manifest=True):
    if manifest:
        metadata = read("manifest.json")
        for name, pin in metadata["files"].items():
            path = BASE / name
            assert path.is_file() and path.stat().st_size == pin["bytes"], name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], name
        for name, pin in metadata["repository_dependencies"].items():
            path = BASE.parent / name
            path.resolve().relative_to(BASE.parent.resolve())
            assert path.is_file() and path.stat().st_size == pin["bytes"], name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], name
    with (BASE / "results/coverage.csv").open(encoding="utf-8") as f:
        coverage = list(csv.DictReader(f))
    for row in coverage:
        assert 0 <= int(row["complete"]) <= int(row["reference"])
        np.testing.assert_allclose(
            float(row["percent"]),
            100 * int(row["complete"]) / int(row["reference"]),
            rtol=1e-12,
            atol=1e-12,
        )
    with (BASE / "results/mechanism_cells.csv").open(encoding="utf-8") as f:
        cells = list(csv.DictReader(f))
    assert len(cells) == 899 and len({r["cell_id"] for r in cells}) == len(cells)
    configs = {r["cell_id"]: r for r in read("configs/published_conditions.json")}
    assert set(configs) == {r["cell_id"] for r in cells}
    for row in cells:
        required = json.loads(row["required_seeds"])
        have = json.loads(row["successful_seeds"])
        assert len(have) == len(set(have)) and set(have) <= set(required)
        assert (set(have) == set(required)) == (row["full_repeats"] == "True")
        assert configs[row["cell_id"]]["source_record"] == int(row["source_record"])
    grids = {
        key: [
            r for r in cells if r["mechanism"] == key and r["reference_grid"] == "True"
        ]
        for key in ["noise_features", "rotation", "target_smoothing"]
    }
    for key, expected in [
        ("noise_features", 168),
        ("rotation", 224),
        ("target_smoothing", 240),
    ]:
        assert len(grids[key]) == expected and all(
            r["full_repeats"] == "True" for r in grids[key]
        )
    assert (
        sum(
            len(json.loads(r["successful_seeds"]))
            for rows in grids.values()
            for r in rows
        )
        == 1556
    )
    samples = read("results/sample_fits.json")
    for sample in samples:
        sample_metrics(sample)
    version = read("configs/frozen_training_sources.json")
    for name, pin in version["files"].items():
        assert (
            hashlib.sha256((BASE / "training/source" / name).read_bytes()).hexdigest()
            == pin
        ), name
    return dict(
        passed=True,
        cell_rows=len(cells),
        reference_cells=632,
        reference_seed_fits=1556,
        sample_numeric_fits_checked=len(samples),
        tolerance=1e-12,
        training_refits=0,
        all_source_fits_reaudited_in_this_bundle=False,
    )


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
