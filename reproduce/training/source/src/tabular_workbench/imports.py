"""Import numeric evidence only; no uploaded pickle, executable or model loading."""

import hashlib
import io
import json
import uuid
import zipfile

import numpy as np

from tabular_workbench.engine import COMMIT, MODELS, verify_sources
from tabular_workbench.runner import metrics, numpy_hash
from tabular_workbench.storage import digest, folder, write_json


def checked_members(content, allowed):
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Artifact limit20MB")
    archive = zipfile.ZipFile(io.BytesIO(content))
    entries = archive.infolist()
    if len(entries) != len(allowed) or set(archive.namelist()) != set(allowed):
        raise ValueError("Only result.json and arrays.npz accepted; no executable/model files")
    if sum(entry.file_size for entry in entries) > 50 * 1024 * 1024 or any(
        entry.file_size > 40 * 1024 * 1024 for entry in entries
    ):
        raise ValueError("Expanded artifact exceeds safe size limit")
    return archive


def import_evidence(content):
    from tabular_workbench.api import RunRequest

    with checked_members(content, {"result.json", "arrays.npz"}) as archive:
        raw_json, raw_arrays = archive.read("result.json"), archive.read("arrays.npz")
    r = json.loads(raw_json)
    if (
        not isinstance(r, dict)
        or not isinstance(r.get("warnings"), list)
        or any(not isinstance(w, str) for w in r["warnings"])
    ):
        raise ValueError("Malformed result metadata")
    if r.get("execution_location") not in ("local", "kaggle"):
        raise ValueError("Unknown execution location")
    request = RunRequest(**r["request"])
    if (
        r["state"] != "completed"
        or r["upstream_commit"] != COMMIT
        or r["source_hash"] != digest(verify_sources())
    ):
        raise ValueError("Only complete evidence matching pinned source accepted")
    if hashlib.sha256(raw_arrays).hexdigest() != r["arrays_sha256"]:
        raise ValueError("Artifact arrays checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(raw_arrays)) as nested:
        if (
            len(nested.infolist()) > 100
            or sum(entry.file_size for entry in nested.infolist()) > 50 * 1024 * 1024
        ):
            raise ValueError("Expanded arrays exceed safe size limit")
        if len(nested.namelist()) != len(set(nested.namelist())):
            raise ValueError("Duplicate array entries")
        declared_size = 0
        for entry in nested.infolist():
            if (
                not entry.filename.endswith(".npy")
                or "/" in entry.filename
                or "\\" in entry.filename
            ):
                raise ValueError("Only flat numeric NPY entries accepted")
            with nested.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported NPY format")
                if (
                    dtype.kind not in "fiu"
                    or not 1 <= len(shape) <= 2
                    or any(not 1 <= size <= 10000 for size in shape)
                ):
                    raise ValueError("Invalid numeric array header")
                size = int(np.prod(shape)) * dtype.itemsize
                if size > entry.file_size or size > 40 * 1024 * 1024:
                    raise ValueError("Declared array shape exceeds actual data or size limit")
                declared_size += size
        if declared_size > 50 * 1024 * 1024:
            raise ValueError("Declared numeric arrays exceed size limit")
    with np.load(io.BytesIO(raw_arrays), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for value in arrays.values():
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all() or value.ndim > 2:
            raise ValueError("Only finite numeric array evidence accepted")
    regression = r["provenance"]["task"] == "regression"
    if r["provenance"]["task"] not in ("regression", "classification"):
        raise ValueError("Unknown task evidence")
    if len(r["results"]) != len(request.models) or set(m["model"] for m in r["results"]) != set(
        request.models
    ):
        raise ValueError("Model evidence disagrees with request")
    for m in r["results"]:
        if m["model"] not in MODELS or m["device"] not in ("cpu", "cuda"):
            raise ValueError("Unknown model/device evidence")
        if not np.isfinite(m["fit_seconds"]) or m["fit_seconds"] < 0 or len(m["history"]) > 300:
            raise ValueError("Invalid timing/history")
        for h in m["history"]:
            if not (isinstance(h["epoch"], (int, float)) and np.isfinite(h["epoch"])):
                raise ValueError("Invalid epoch")
            if not all(
                (h[key] is None and m.get("history_nonfinite_observed") is True)
                or (isinstance(h[key], (int, float)) and np.isfinite(h[key]))
                for key in ("train_loss", "valid_loss")
            ):
                raise ValueError("Invalid numeric history")
        for part in ("train", "validation", "test"):
            y, x, ids = arrays["y_" + part], arrays["x_" + part], arrays["ids_" + part]
            if (
                x.ndim != 2
                or len(x) != len(y)
                or ids.shape != y.shape
                or len(y) > 10000
                or x.shape[1] > 800
            ):
                raise ValueError("Malformed or oversized partition")
            expected = metrics(y, arrays[m["model"] + "_prediction_" + part], regression)
            for key, value in expected.items():
                np.testing.assert_allclose(m["metrics"][part][key], value, rtol=1e-12, atol=1e-12)
        if (
            numpy_hash(
                [arrays[m["model"] + "_input_" + part] for part in ("train", "validation", "test")]
            )
            != m["input_hash"]
        ):
            raise ValueError("Model input hash mismatch")
    ids = [arrays["ids_" + part] for part in ("train", "validation", "test")]
    if any(v.dtype.kind not in "iu" or (v < 0).any() for v in ids):
        raise ValueError("Invalid integer row IDs")
    if r["provenance"]["split_hash"] != digest([value.tolist() for value in ids]):
        raise ValueError("Split hash mismatch")
    if len(set(np.concatenate(ids).tolist())) != sum(len(v) for v in ids):
        raise ValueError("Partition row IDs overlap")
    if r["provenance"]["split_counts"] != [len(v) for v in ids]:
        raise ValueError("Reported partition counts mismatch")
    if len(r["provenance"]["feature_names"]) != arrays["original_train"].shape[1]:
        raise ValueError("Reported feature names mismatch")
    for prefix, hash_name in (("original_", "original_input_hash"), ("x_", "case_input_hash")):
        if (
            numpy_hash([arrays[prefix + part] for part in ("train", "validation", "test")])
            != r["provenance"][hash_name]
        ):
            raise ValueError("Input hash mismatch")
    # Bind evidence to actual locally cached data, not just self-reported hashes.
    # No fit or model deserialization; source rows/labels are checked directly.
    from tabular_workbench.data import load

    original_x, original_y, source = load(request.dataset)
    for key in ("sha256", "schema_hash", "rows", "features", "task", "feature_names"):
        if r["provenance"][key] != source[key]:
            raise ValueError("Evidence dataset provenance does not match local source")
    for part in ("train", "validation", "test"):
        indices = arrays["ids_" + part]
        if (indices >= len(original_y)).any():
            raise ValueError("Source row index out of range")
        np.testing.assert_array_equal(arrays["original_" + part], original_x[indices])
        np.testing.assert_array_equal(arrays["y_" + part], original_y[indices])
    identifier = uuid.uuid4().hex[:16]
    path = folder(identifier)
    path.mkdir(parents=True)
    r.update(id=identifier, external_run_id=r["id"], imported_evidence=True)
    r["warnings"].append(
        "Imported numeric evidence: checksums/splits/metrics verified, no uploaded model deserialized; reported runtime/source provenance is preserved."
    )
    (path / "arrays.npz").write_bytes(raw_arrays)
    write_json(path / "request.json", request.dict())
    write_json(path / "result.json", r)
    write_json(
        path / "status.json", {"id": identifier, "state": "completed", "request": request.dict()}
    )
    return {
        "id": identifier,
        "state": "completed",
        "metrics_recomputed_from_frozen_predictions": "passed",
        "model_deserialized": False,
    }
