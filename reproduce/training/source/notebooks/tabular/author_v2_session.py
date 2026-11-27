"""Finite v2 CPU session: one isolated original-model worker per fit.

No legacy continuation entrypoint, implicit retry, optimizer resume or remote API.
Only numeric predictions/targets/row IDs leave the temporary working directory.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from author_v2 import TRACK, read, sha, validate_request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def validate(plan):
    assert plan["execution_track"] == TRACK and plan["maximum_session_seconds"] == 36000
    assert plan["fit_timeout_seconds"] == 1800 and plan["jobs"]
    assert len({j["id"] for j in plan["jobs"]}) == len(plan["jobs"])
    for job in plan["jobs"]:
        r = job["request"]
        validate_request(r)
        assert r["models"][0] in ("rf", "gbt", "xgb") and r["device"] == "cpu"
        assert job["scientific_spec_hash"] == digest(r) and job["id"] == digest(r)[:16]
        assert job["attempt"] == 1 and job["reason"] == "new_unstarted_slot"
        assert r["seed"] in plan["required_seeds"][r["dataset"]]
        assert r["required_seed_indices"] == plan["required_seeds"][r["dataset"]]


def mount_cache(mounted, destination, expected_manifest):
    """Hash-checked flat transport ZIP; never extract arbitrary input members."""
    candidates = list(Path(mounted).glob("**/cache-manifest.json"))
    candidates = [p for p in candidates if sha(p) == expected_manifest]
    assert len(candidates) == 1, "Require exactly one pinned original-dtype cache"
    manifest_path = candidates[0]
    manifest = read(manifest_path)
    assert manifest["execution_track"] == TRACK and manifest["public_only"]
    archive_path = manifest_path.parent / "author-v2-inputs.zip"
    allowed = {f["relative"]: f for t in manifest["tasks"].values() for f in t["files"].values()}
    assert len(allowed) == 16

    def publish(load):
        for name, pin in allowed.items():
            raw = load(name)
            assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
            target = (Path(destination) / name).resolve()
            target.relative_to(Path(destination).resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == raw
            else:
                target.write_bytes(raw)

    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            assert set(archive.namelist()) == set(allowed) and len(archive.namelist()) == 16
            assert sum(i.file_size for i in archive.infolist()) < 16 * 1024**2
            publish(archive.read)
    else:
        # Kaggle's Dataset service expands an uploaded ZIP into a same-stem
        # directory. This is transport only; every original file stays SHA-bound.
        expanded = manifest_path.parent / "author-v2-inputs"
        assert {
            p.relative_to(expanded).as_posix() for p in expanded.rglob("*") if p.is_file()
        } == set(allowed)
        publish(lambda name: (expanded / name).read_bytes())
    return Path(destination)


def compact(path, result):
    import numpy as np

    assert sha(path / "arrays.npz") == result["arrays_sha256"]
    model = result["results"][0]["model"]
    names = [
        prefix + part
        for part in ("train", "validation", "test")
        for prefix in ("y_", "ids_", model + "_prediction_")
    ]
    with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
        values = {n: arrays[n] for n in names}
    np.savez_compressed(path / "numeric.npz", **values)
    return sha(path / "numeric.npz")


def export(destination, state, output):
    """Publish an atomic fit-level checkpoint after every terminal attempt."""
    save(output / "tabular-author-v2.json", state)
    target = output / "tabular-author-v2-evidence.zip"
    temporary = target.with_suffix(".tmp")
    files = {}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:

        def add(name, raw):
            archive.writestr(name, raw)
            files[name] = hashlib.sha256(raw).hexdigest()

        add("session.json", json.dumps(state, indent=2, allow_nan=False).encode())
        for record in state["records"]:
            if record["state"] == "completed":
                source = destination / "runs" / record["id"] / "numeric.npz"
                assert sha(source) == record["numeric_sha256"]
                add(record["id"] + "/numeric.npz", source.read_bytes())
        archive.writestr(
            "manifest.json",
            json.dumps(
                dict(
                    schema=2,
                    execution_track=TRACK,
                    plan_hash=state["plan_hash"],
                    files=files,
                    raw_features=False,
                    model_pickle=False,
                    credentials=False,
                )
            ),
        )
    os.replace(temporary, target)


def discard_temporary_model(path, record, output):
    """After numeric checkpoint proof, remove only this fit's redundant temp files.

    Fit-level continuation never uses a model pickle or feature matrix. Keeping
    hundreds of these on Kaggle would needlessly exhaust its temporary disk.
    """
    assert record["state"] == "completed" and path.name == record["id"]
    numeric_name = record["id"] + "/numeric.npz"
    with zipfile.ZipFile(output / "tabular-author-v2-evidence.zip") as archive:
        assert hashlib.sha256(archive.read(numeric_name)).hexdigest() == record["numeric_sha256"]
    result = record["result"]
    model = result["results"][0]
    targets = [
        (path / "arrays.npz", result["arrays_sha256"]),
        (path / model["model"] / "local-model.joblib", model["model_hash"]),
    ]
    for target, pin in targets:
        target.resolve().relative_to(path.resolve())
        assert target.is_file() and not target.is_symlink() and sha(target) == pin
    for target, _ in targets:
        target.unlink()


def run(plan_path, cache, destination, output, seconds, started=None):
    from tabular_workbench.continuation import terminate_owned
    from tabular_workbench.engine import verify_sources

    plan = read(plan_path)
    validate(plan)
    assert 0 < seconds <= 36000
    baseline = verify_sources()
    destination, output = Path(destination), Path(output)
    assert not destination.exists(), "New session only; audit a stopped session before continuation"
    destination.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    state = dict(
        schema=2,
        execution_track=TRACK,
        plan_hash=digest(plan),
        state="running",
        records=[],
        remaining_jobs=len(plan["jobs"]),
        completed_new=0,
        maximum_seconds=36000,
        author_commit=baseline["commit"],
        adapter_sha256=plan["adapter_sha256"],
        resume_level="terminal fit only",
        legacy_results_merged=False,
        validation_only_selection=True,
    )
    started = time.monotonic() if started is None else started
    export(destination, state, output)
    for job in plan["jobs"]:
        # Preserve the full1800sfit allowance plus atomic export reserve.
        if seconds - (time.monotonic() - started) < 1890:
            state["state"] = "budget_stopped"
            break
        r, identifier = job["request"], job["id"]
        request_path = destination / "requests" / (identifier + ".json")
        save(request_path, r)
        run_path = destination / "runs" / identifier
        assert not run_path.exists()
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            WANDB_MODE="disabled",
            WANDB_ERROR_REPORTING="false",
            MPLBACKEND="Agg",
        )
        with (destination / (identifier + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("author_v2.py")),
                    "--request",
                    str(request_path),
                    "--cache-root",
                    str(cache),
                    "--output-root",
                    str(destination / "runs"),
                ],
                env=env,
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
        record = dict(
            id=identifier,
            scientific_spec_hash=job["scientific_spec_hash"],
            request=r,
            attempt=1,
            reason=job["reason"],
            exit_code=process.returncode,
        )
        if timed_out:
            record.update(state="timeout", error_type="FitBudget1800s")
        elif process.returncode != 0 or not (run_path / "execution-receipt.json").exists():
            record.update(state="failed", error_type="WorkerExitedWithoutFinalReceipt")
        else:
            receipt = read(run_path / "execution-receipt.json")
            record.update(state=receipt["state"], execution_receipt=receipt)
            if receipt["state"] == "completed":
                result = read(run_path / "result.json")
                assert result["request"] == r and result["id"] == identifier
                assert receipt["adapter_sha256"] == state["adapter_sha256"]
                assert (
                    not result["source_modified"] and receipt["author_commit"] == baseline["commit"]
                )
                record.update(result=result, numeric_sha256=compact(run_path, result))
            else:
                record["failure"] = read(run_path / "status.json")
        state["records"].append(record)
        state.update(
            remaining_jobs=len(plan["jobs"]) - len(state["records"]),
            completed_new=sum(r["state"] == "completed" for r in state["records"]),
            elapsed_seconds=time.monotonic() - started,
        )
        export(destination, state, output)
        if record["state"] == "completed":
            discard_temporary_model(run_path, record, output)
        print(
            json.dumps(
                dict(
                    attempted=len(state["records"]),
                    remaining=state["remaining_jobs"],
                    state=record["state"],
                )
            ),
            flush=True,
        )
    else:
        state["state"] = "all_planned_attempts_finished"
    state["elapsed_seconds"] = time.monotonic() - started
    export(destination, state, output)
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, required=True)
    a = p.parse_args()
    run(a.plan, a.cache, a.destination, a.output, a.seconds)
