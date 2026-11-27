"""Immutable fit-level resume with explicit retry lineage and numeric checkpointing.

Reuses original benchmark request/config/seed exactly. Additional request metadata
only gives each retry a fresh storage ID; trainer ignores these bookkeeping keys.
No epoch-resume, model approximation, test-guided choices or automatic resubmit.
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
from typing import Any, cast

from tabular_workbench import data, runner
from tabular_workbench.benchmark import compact_numeric, register
from tabular_workbench.engine import COMMIT, verify_sources
from tabular_workbench.storage import ROOT, digest, folder, read_json, report, write_json


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan(plan, resume):
    assert plan["schema"] == 1 and plan["upstream_commit"] == COMMIT
    assert digest(plan["tasks"]) == resume["tasks_hash"]
    assert plan["parent_protocol_hash"] == resume["parent_protocol_hash"]
    assert plan["initial_verified_count"] == len(resume["completed"])
    assert len(plan["jobs"]) == len({j["scientific_spec_hash"] for j in plan["jobs"]})
    completed = set(resume["completed"])
    for job in plan["jobs"]:
        request = job["request"]
        assert (
            digest(request) == job["scientific_spec_hash"]
            and job["scientific_spec_hash"] not in completed
        )
        assert request["device"] in ("cpu", "cuda") and len(request["models"]) == 1
        assert not request["dataset"].startswith("upload-")
        assert request["epochs"] == 300 and request["train_limit"] == 10000
        assert request["holdout_limit"] == 50000 and request["seed"] in (0, 1, 2)
        assert request["timeout_seconds"] == 1800
        assert job["reason"] in ("not_started", "adapter_retry")


def attempt_request(job, attempt, parent_protocol_hash):
    # Original scientific spec remains separately hashed before adding metadata.
    return dict(
        job["request"],
        continuation={
            "scientific_spec_hash": job["scientific_spec_hash"],
            "attempt": attempt,
            "parent_protocol_hash": parent_protocol_hash,
            "reason": job["reason"],
            "scope": "new independent partial Figure5 fit; not a main-benchmark retry"
            if "paper_ablation" in job["request"]
            else "adapter-only continuation; exploratory retry",
        },
    )


def new_ledger(plan_hash, parent_protocol_hash):
    return {
        "schema": 2,
        "plan_hash": plan_hash,
        "parent_protocol_hash": parent_protocol_hash,
        "state": "ready",
        "records": [],
        "resume_level": "fit-level only",
    }


def remaining_jobs(plan, ledger):
    attempted = {r["scientific_spec_hash"] for r in ledger["records"]}
    return [job for job in plan["jobs"] if job["scientific_spec_hash"] not in attempted]


def read_initial(plan_path, resume_path):
    plan, resume = read_json(plan_path), read_json(resume_path)
    validate_plan(plan, resume)
    assert resume["author_source_hash"] == digest(verify_sources())
    register(plan["tasks"])
    return plan, resume


def run_worker(identifier, plan_path, resume_path):
    plan, resume = read_initial(plan_path, resume_path)
    request = read_json(folder(identifier) / "request.json")
    metadata = request["continuation"]
    raw_request = {key: value for key, value in request.items() if key != "continuation"}
    assert digest(raw_request) == metadata["scientific_spec_hash"]
    assert metadata["scientific_spec_hash"] not in resume["completed"]
    assert any(j["scientific_spec_hash"] == metadata["scientific_spec_hash"] for j in plan["jobs"])
    assert identifier == digest(request)[:16]
    data.fetch(request["dataset"])
    runner.worker(identifier)


def run_session(plan_path, resume_path, destination, seconds, session_id, limit=None):
    plan, resume = read_initial(plan_path, resume_path)
    assert 180 <= seconds <= 36000 and session_id.replace("-", "").isalnum()
    plan_hash = digest(plan)
    destination.mkdir(parents=True, exist_ok=True)
    ledger_path = destination / "continuation.json"
    state = (
        read_json(ledger_path)
        if ledger_path.exists()
        else new_ledger(plan_hash, plan["parent_protocol_hash"])
    )
    assert (
        state["plan_hash"] == plan_hash
        and state["parent_protocol_hash"] == plan["parent_protocol_hash"]
    )
    assert len(state["records"]) == len({r["scientific_spec_hash"] for r in state["records"]})
    started = time.monotonic()
    processed = 0
    state.update(state="running", session_id=session_id)
    write_json(ledger_path, state)
    for job in remaining_jobs(plan, state):
        remaining = (
            seconds - (time.monotonic() - started) - plan["config"]["saving_reserve_seconds"]
        )
        minimum_start_budget = (
            job["request"]["timeout_seconds"]
            if plan["config"].get("require_full_fit_budget", False)
            else 60
        )
        if remaining < minimum_start_budget or (limit is not None and processed >= limit):
            state["state"] = (
                "budget_stopped" if remaining < minimum_start_budget else "explicit_fit_limit"
            )
            break
        attempt = job["attempt"]
        request = attempt_request(job, attempt, plan["parent_protocol_hash"])
        identifier = digest(request)[:16]
        run_path = folder(identifier)
        run_path.mkdir(parents=True, exist_ok=True)
        if (run_path / "request.json").exists():
            assert read_json(run_path / "request.json") == request
        else:
            write_json(run_path / "request.json", request)
        recovered = False
        if (run_path / "result.json").exists():
            # Crash recovery after fit save but before appending ledger.
            result = report(identifier)
            assert result["state"] == "completed" and result["request"] == request
            if not (run_path / "numeric.npz").exists():
                compact_numeric(run_path, result)
            recovered = True
        elif (run_path / "status.json").exists() and read_json(run_path / "status.json")[
            "state"
        ] in ("failed", "timeout", "cancelled"):
            result = report(identifier)
            recovered = True
        else:
            write_json(
                run_path / "status.json", {"id": identifier, "state": "queued", "request": request}
            )
            env = os.environ.copy()
            env.update(
                WANDB_MODE="disabled",
                WANDB_ERROR_REPORTING="false",
                MPLBACKEND="Agg",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            command = [
                sys.executable,
                "-m",
                "tabular_workbench.continuation",
                "worker",
                "--id",
                identifier,
                "--plan",
                str(plan_path),
                "--resume",
                str(resume_path),
            ]
            with (run_path / "process.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=log,
                    start_new_session=os.name != "nt",
                )
                try:
                    process.wait(timeout=min(request["timeout_seconds"], remaining))
                except subprocess.TimeoutExpired:
                    terminate_owned(process)
                    write_json(
                        run_path / "status.json",
                        {
                            "id": identifier,
                            "state": "timeout",
                            "request": request,
                            "error": "Continuation fit time budget reached; no partial result accepted",
                        },
                    )
            result = report(identifier)
            if result["state"] in ("queued", "running"):
                result.update(
                    state="failed",
                    error="Worker ended before final result; see preserved process.log",
                )
                write_json(run_path / "status.json", result)
        record = {
            "id": identifier,
            "scientific_spec_hash": job["scientific_spec_hash"],
            "attempt": attempt,
            "reason": job["reason"],
            "state": result["state"],
            "request": request,
            "recovered_fit_checkpoint": recovered,
        }
        if result["state"] == "completed":
            assert result["source_hash"] == resume["author_source_hash"]
            assert result["request"] == request
            result["execution_location"] = (
                "kaggle" if ROOT.as_posix().startswith("/kaggle/") else "local"
            )
            if not (run_path / "numeric.npz").exists():
                compact_numeric(run_path, result)
            result["numeric_sha256"] = file_hash(run_path / "numeric.npz")
            write_json(run_path / "result.json", result)
            record["result"] = result
        else:
            record["error"] = result.get("error", "Incomplete fit; no metric substituted")
        state["records"].append(record)
        processed += 1
        state.update(
            elapsed_seconds=time.monotonic() - started,
            remaining_jobs=len(remaining_jobs(plan, state)),
            completed_new=sum(r["state"] == "completed" for r in state["records"]),
            initial_completed_skipped=plan["initial_verified_count"],
        )
        write_json(ledger_path, state)
        print(
            json.dumps(
                {
                    "completed_new": state["completed_new"],
                    "attempted": len(state["records"]),
                    "remaining": state["remaining_jobs"],
                    "last_state": record["state"],
                }
            ),
            flush=True,
        )
    else:
        state["state"] = "all_planned_attempts_finished"
    state.update(
        elapsed_seconds=time.monotonic() - started,
        remaining_jobs=len(remaining_jobs(plan, state)),
        initial_completed_skipped=plan["initial_verified_count"],
    )
    write_json(ledger_path, state)
    return state


def terminate_owned(process):
    if os.name != "nt":
        import signal

        kill_group = cast(Any, os).killpg
        kill_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_group(process.pid, cast(Any, signal).SIGKILL)
            process.wait(timeout=10)
    else:
        import psutil

        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in descendants:
            child.kill()
        parent.kill()
        psutil.wait_procs(descendants + [parent], timeout=10)
        process.wait(timeout=10)


def export_session(destination, output):
    state = read_json(destination / "continuation.json")
    manifest = {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        source = destination / "continuation.json"
        payload = source.read_bytes()
        archive.writestr("continuation.json", payload)
        manifest["continuation.json"] = hashlib.sha256(payload).hexdigest()
        for record in state["records"]:
            if record["state"] != "completed":
                continue
            source = folder(record["id"]) / "numeric.npz"
            payload = source.read_bytes()
            assert hashlib.sha256(payload).hexdigest() == record["result"]["numeric_sha256"]
            name = record["id"] + "/numeric.npz"
            archive.writestr(name, payload)
            manifest[name] = hashlib.sha256(payload).hexdigest()
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": 2,
                    "files": manifest,
                    "raw_features": False,
                    "credentials": False,
                    "model_pickle": False,
                    "parent_protocol_hash": state["parent_protocol_hash"],
                    "plan_hash": state["plan_hash"],
                }
            ),
        )
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": file_hash(output)}


def restore_checkpoint(checkpoint, plan_path, resume_path, destination):
    """Continue next version from exact numeric ledger; no pickle/epoch resume."""
    import io
    import re

    import numpy as np

    plan, resume = read_initial(plan_path, resume_path)
    with zipfile.ZipFile(checkpoint) as archive:
        entries = archive.infolist()
        assert len(entries) <= 1500 and len({i.filename for i in entries}) == len(entries)
        assert sum(i.file_size for i in entries) <= 2 * 1024**3
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema"] == 2 and manifest["plan_hash"] == digest(plan)
        assert manifest["parent_protocol_hash"] == resume["parent_protocol_hash"]
        assert (
            not manifest["credentials"]
            and not manifest["model_pickle"]
            and not manifest["raw_features"]
        )
        assert set(archive.namelist()) == set(manifest["files"]) | {"manifest.json"}
        state = json.loads(archive.read("continuation.json"))
        assert state["plan_hash"] == digest(plan)
        assert len(state["records"]) == len({r["scientific_spec_hash"] for r in state["records"]})
        allowed_jobs = {j["scientific_spec_hash"]: j for j in plan["jobs"]}
        expected = {"continuation.json"}
        for record in state["records"]:
            assert record["scientific_spec_hash"] in allowed_jobs
            job = allowed_jobs[record["scientific_spec_hash"]]
            assert record["request"] == attempt_request(
                job, job["attempt"], plan["parent_protocol_hash"]
            )
            assert record["id"] == digest(record["request"])[:16]
            if record["state"] == "completed":
                expected.add(record["id"] + "/numeric.npz")
        assert set(manifest["files"]) == expected
        payloads = {}
        for name, expected_hash in manifest["files"].items():
            assert name == "continuation.json" or re.fullmatch(r"[0-9a-f]{16}/numeric\.npz", name)
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == expected_hash
            payloads[name] = payload
        for record in state["records"]:
            if record["state"] != "completed":
                continue
            result = record["result"]
            assert (
                result["source_hash"] == resume["author_source_hash"]
                and result["request"] == record["request"]
            )
            payload = payloads[record["id"] + "/numeric.npz"]
            assert result["numeric_sha256"] == hashlib.sha256(payload).hexdigest()
            with zipfile.ZipFile(io.BytesIO(payload)) as numeric:
                assert (
                    len(numeric.infolist()) <= 15
                    and sum(i.file_size for i in numeric.infolist()) <= 30 * 1024**2
                )
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                model = result["results"][0]
                rows = []
                for part in ("train", "validation", "test"):
                    y, prediction, ids = (
                        arrays["y_" + part],
                        arrays[model["model"] + "_prediction_" + part],
                        arrays["ids_" + part],
                    )
                    assert ids.dtype.kind in "iu" and ids.shape == y.shape and len(y) <= 50000
                    recomputed = runner.metrics(
                        y, prediction, result["provenance"]["task"] == "regression"
                    )
                    for key, value in recomputed.items():
                        np.testing.assert_allclose(
                            model["metrics"][part][key], value, rtol=1e-12, atol=1e-12
                        )
                    rows.append(ids)
                assert len(set(np.concatenate(rows))) == sum(len(row) for row in rows)
                assert digest([row.tolist() for row in rows]) == result["provenance"]["split_hash"]
    # Only write after all payloads pass validation, and never overwrite differs.
    destination.mkdir(parents=True, exist_ok=True)
    ledger = destination / "continuation.json"
    if ledger.exists():
        assert read_json(ledger) == state
    else:
        write_json(ledger, state)
    for record in state["records"]:
        if record["state"] == "completed":
            path = folder(record["id"])
            path.mkdir(parents=True, exist_ok=True)
            target = path / "numeric.npz"
            payload = payloads[record["id"] + "/numeric.npz"]
            if target.exists():
                assert target.read_bytes() == payload
            else:
                target.write_bytes(payload)
            for name, value in (
                ("result.json", record["result"]),
                ("request.json", record["request"]),
            ):
                target = path / name
                if target.exists():
                    assert read_json(target) == value
                else:
                    write_json(target, value)
    return {
        "restored_attempts": len(state["records"]),
        "remaining": len(remaining_jobs(plan, state)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("worker", "run", "restore"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=int, default=36000)
    parser.add_argument("--session-id", default="one-notebook")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        run_worker(args.id, args.plan, args.resume)
    elif args.action == "restore":
        print(json.dumps(restore_checkpoint(args.checkpoint, args.plan, args.resume, args.output)))
    else:
        run_session(args.plan, args.resume, args.output, args.seconds, args.session_id, args.limit)
