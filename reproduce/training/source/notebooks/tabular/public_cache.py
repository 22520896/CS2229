"""Verify/copy one pinned public cache before fit clocks; no network or math edits."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(source, pin):
    source = Path(source)
    assert pin["task"] == "361069" and pin["public_only"] is True
    for name in ("manifest.json", "dataset.npz"):
        expected = pin["files"][name]
        path = source / name
        assert path.is_file() and path.stat().st_size == expected["bytes"]
        assert path.stat().st_size <= 200 * 1024**2
        assert sha(path) == expected["sha256"], "Public cache checksum mismatch"
    metadata = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    assert metadata == pin["manifest"]
    assert metadata["sha256"] == pin["files"]["dataset.npz"]["sha256"]
    assert metadata["id"] == "361069" and metadata["suite_id"] == 337
    assert metadata["source"] == "Processed authors' benchmark OpenML task"
    assert metadata["rows"] == 940160 and metadata["features"] == 24
    with np.load(source / "dataset.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"x", "y"}
        x, y = arrays["x"], arrays["y"]
        assert x.shape == (940160, 24) and x.dtype == np.dtype("float32")
        assert y.shape == (940160,) and y.dtype == np.dtype("int64")
        assert np.isfinite(x).all() and set(np.unique(y)) == {0, 1}
    return metadata


def install(source, project, pin):
    """Publish manifest last, refusing to overwrite different user/cache bytes."""
    source, project = Path(source).resolve(), Path(project).resolve()
    metadata = verify(source, pin)
    destination = project / "data/datasets/361069"
    if destination.exists():
        for name in ("manifest.json", "dataset.npz"):
            path = destination / name
            assert not path.exists() or sha(path) == pin["files"][name]["sha256"], (
                "Existing cache differs; preserve it"
            )
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("dataset.npz", "manifest.json"):
        path = destination / name
        if not path.exists():
            temporary = destination / (name + ".verified-cache-tmp")
            assert not temporary.exists(), "Preserve an unfinished import for inspection"
            shutil.copyfile(source / name, temporary)
            assert sha(temporary) == pin["files"][name]["sha256"]
            temporary.replace(path)
    assert verify(destination, pin) == metadata
    return {
        "public_cache_verified": True,
        "task": "361069",
        "rows": 940160,
        "features": 24,
        "network_called": False,
        "formula_changed": False,
    }


def from_kaggle(project, plan_path, input_root=Path("/kaggle/input")):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    pin = plan["public_cache"]
    # Only the exact pinned cache from the explicit notebook attachment. Do not
    # consume arbitrary CSV/uploads or search the user's computer.
    candidates = []
    pattern = (
        "**/" + pin["dataset_reference"].split("/")[1] + "/manifest.json"
        if "dataset_reference" in pin
        else "**/tabular-study/data/datasets/361069/manifest.json"
    )
    for path in Path(input_root).glob(pattern):
        path.resolve().relative_to(Path(input_root).resolve())
        if sha(path) == pin["files"]["manifest.json"]["sha256"]:
            candidates.append(path.parent)
    assert len(candidates) == 1, "Missing or ambiguous pinned Kaggle cache; stop before any fit"
    return install(candidates[0], project, pin)


def split_parity(project, plan_path):
    """Compare actual original splitter outputs against all three frozen seeds."""
    project = Path(project).resolve()
    sys.path.insert(0, str(project / "src"))
    from tabular_workbench import data
    from tabular_workbench.continuation import read_initial
    from tabular_workbench.runner import numpy_hash

    plan, _ = read_initial(Path(plan_path), Path(plan_path).parent / "resume.json")
    seed_requests = {}
    for request in plan["input_pin_checks"]:
        seed_requests.setdefault(request["seed"], request)
    assert set(seed_requests) == {0, 1, 2}
    for request in seed_requests.values():
        xx, yy, _, _, provenance = data.prepare(request)
        expected = request["paper_ablation"]["input_pin"]
        for key in ("rows", "features", "openml_dataset_id", "dataset_version", "split_hash"):
            assert provenance[key] == expected[key]
        assert numpy_hash(xx) == expected["original_input_hash"]
        assert numpy_hash(yy) == expected["labels_hash"]
    return {"original_split_and_input_parity": True, "seeds_checked": sorted(seed_requests)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    result = (
        install(args.source, args.project, plan["public_cache"])
        if args.source
        else from_kaggle(args.project, args.plan)
    )
    result.update(split_parity(args.project, args.plan))
    print(json.dumps(result))
