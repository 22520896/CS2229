"""Cache processed OpenML benchmark tasks; original split/transform functions."""

import csv
import hashlib
import io
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd

from tabular_workbench.engine import modules
from tabular_workbench.storage import ROOT, digest, read_json, write_json

CATALOG: Dict[str, Dict[str, Any]] = {
    "361055": {"suite": 337, "task": "classification"},
    "361060": {"suite": 337, "task": "classification"},
    "361072": {"suite": 336, "task": "regression"},
}

# Only this pinned public task needs a larger download-side row guard. This is
# NOT the training sample limit and never applies to uploaded CSVs.
PUBLIC_LARGE_TASK: Dict[str, Any] = {
    "task": "361281",
    "dataset_id": 45034,
    "version": 3,
    "rows": 5465575,
    "features": 8,
    "md5": "1dfb70692d795e43b86d6bf1c82465ae",
}


def dataset_folder(identifier):
    if identifier not in CATALOG and not (
        identifier.startswith("upload-")
        and len(identifier) == 19
        and all(c in "0123456789abcdef" for c in identifier[7:])
    ):
        raise ValueError("Unknown dataset")
    return ROOT / "data/datasets" / identifier


def validate_arrays(x, y, regression, public=False, public_task=None):
    max_rows, max_features = (2000000, 2000) if public else (200000, 200)
    if public and public_task == PUBLIC_LARGE_TASK["task"]:
        max_rows = PUBLIC_LARGE_TASK["rows"]
    if x.ndim != 2 or len(x) < 100 or len(x) > max_rows or not 2 <= x.shape[1] <= max_features:
        rows = len(x) if x.ndim else 0
        features = x.shape[1] if x.ndim == 2 else "unknown"
        raise ValueError(
            "Numeric dataset requires 100-%d rows and 2-%d features; received %d rows, %s features (%s input)"
            % (max_rows, max_features, rows, features, "public" if public else "upload")
        )
    if y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(
            "Finite numeric X and target required; missing rows are not silently removed"
        )
    if not regression and (set(np.unique(y)) != {0, 1} or min(np.bincount(y.astype(int))) < 20):
        raise ValueError(
            "Classification currently requires two labels encoded0/1 and ≥20rows/class"
        )


def fetch(identifier):
    if identifier not in CATALOG:
        raise ValueError("Only approved public OpenML tasks can be downloaded")
    path = dataset_folder(identifier)
    if (path / "manifest.json").exists():
        return read_json(path / "manifest.json")
    import openml

    openml.config.set_cache_directory(str(ROOT / "data/openml-cache"))
    meta = CATALOG[identifier]
    assert int(identifier) in (openml.study.get_suite(meta["suite"]).tasks or [])
    # We use the original repo's local RandomState split, not OpenML's folds.
    # Old OpenML eagerly downloads/parses its ARFF splits with download_data=True;
    # task361281 has a malformed/missing split, unrelated to the feature table.
    task = openml.tasks.get_task(int(identifier), download_data=False)
    data = task.get_dataset()
    x, y, categories, names = data.get_data(
        dataset_format="array", target=data.default_target_attribute
    )
    if categories is not None and np.asarray(categories).any():
        raise ValueError("Initial numeric-only profile does not silently encode mixed tasks")
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32 if meta["task"] == "regression" else np.int64)
    if identifier == PUBLIC_LARGE_TASK["task"]:
        expected = PUBLIC_LARGE_TASK
        if (int(data.dataset_id), int(data.version), x.shape, y.shape) != (
            expected["dataset_id"],
            expected["version"],
            (expected["rows"], expected["features"]),
            (expected["rows"],),
        ) or data.md5_checksum != expected["md5"]:
            raise ValueError(
                "Pinned public dataset identity/dimensions changed; refuse row-limit exception"
            )
    validate_arrays(x, y, meta["task"] == "regression", public=True, public_task=identifier)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "dataset.npz", x=x, y=y)
    manifest = {
        "id": identifier,
        "name": data.name,
        "task": meta["task"],
        "suite_id": meta["suite"],
        "openml_task_id": int(identifier),
        "openml_dataset_id": data.dataset_id,
        "dataset_version": data.version,
        "source": "Processed authors' benchmark OpenML task",
        "source_url": "https://www.openml.org/t/%s" % identifier,
        "rows": len(x),
        "features": x.shape[1],
        "feature_names": list(names),
        "sha256": hashlib.sha256((path / "dataset.npz").read_bytes()).hexdigest(),
        "schema_hash": digest(list(names)),
        "numeric_only": True,
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def catalogue():
    result = []
    for identifier, meta in CATALOG.items():
        path = dataset_folder(identifier)
        result.append(
            read_json(path / "manifest.json")
            if (path / "manifest.json").exists()
            else {
                "id": identifier,
                "name": "OpenML task " + identifier,
                "task": meta["task"],
                "ready": False,
            }
        )
    for path in sorted((ROOT / "data/datasets").glob("upload-*/manifest.json")):
        result.append(read_json(path))
    return result


def upload(payload, target, regression):
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError("CSV limit10MB")
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    frame = pd.read_csv(io.BytesIO(payload))
    if target not in frame:
        raise ValueError("Target column not found")
    y = frame.pop(target).to_numpy(dtype=np.float32 if regression else float)
    x = frame.to_numpy(dtype=np.float32)
    validate_arrays(x, y, regression)
    if not regression:
        y = y.astype(np.int64)
    identifier = "upload-" + uuid.uuid4().hex[:12]
    path = dataset_folder(identifier)
    path.mkdir(parents=True)
    np.savez_compressed(path / "dataset.npz", x=x, y=y)
    manifest = {
        "id": identifier,
        "name": "Uploaded numeric CSV",
        "task": "regression" if regression else "classification",
        "source": "User-uploaded local numeric data; NOT paper benchmark",
        "rows": len(x),
        "features": x.shape[1],
        "feature_names": frame.columns.tolist(),
        "numeric_only": True,
        "schema_hash": digest(frame.columns.tolist()),
        "sha256": hashlib.sha256((path / "dataset.npz").read_bytes()).hexdigest(),
    }
    write_json(path / "manifest.json", manifest)
    return manifest


def load(identifier):
    path = dataset_folder(identifier)
    if not (path / "manifest.json").exists():
        raise ValueError("Dataset not cached. Download public data first")
    meta = read_json(path / "manifest.json")
    if hashlib.sha256((path / "dataset.npz").read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Cached dataset checksum changed")
    with np.load(path / "dataset.npz", allow_pickle=False) as saved:
        return saved["x"], saved["y"], meta


def prepare(request):
    x, y, meta = load(request["dataset"])
    _, pipeline, _, _, _ = modules()
    rng = np.random.RandomState(request["seed"])
    rng.randn(1)  # Exact upstream run_experiment pre-split RNG draw.
    cfg = {
        "train_prop": 0.7,
        "val_test_prop": 0.3,
        "max_train_samples": request["train_limit"],
        "max_val_samples": request["holdout_limit"],
        "max_test_samples": request["holdout_limit"],
    }
    ids = np.arange(len(x)).reshape(-1, 1)
    parts = pipeline.data_to_train_test(ids, y, cfg, rng)
    row_ids = [a.ravel().astype(int) for a in parts[:3]]
    ys = [np.asarray(a) for a in parts[3:]]
    for values in ys:
        if len(values) < 10 or (meta["task"] == "classification" and len(np.unique(values)) != 2):
            raise ValueError(
                "Original nonstratified split missing labels/too small; choose a larger limit"
            )
    xx = [x[rows].copy() for rows in row_ids]
    provenance = {
        **meta,
        "split_hash": digest([rows.tolist() for rows in row_ids]),
        "split_counts": [len(a) for a in row_ids],
        "split_config": cfg,
        "split_source": "upstream/src/generate_dataset_pipeline.py::data_to_train_test",
        "rng_pre_split_draw": "RandomState(seed).randn(1)",
        "stratified": False,
    }
    return xx, ys, row_ids, rng, provenance
