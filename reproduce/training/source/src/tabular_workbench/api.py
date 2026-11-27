"""Loopback-only web API; no remote execution or credential endpoint."""

import json
import multiprocessing as mp
import threading
import uuid
import zipfile
from typing import List, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tabular_workbench.data import catalogue, fetch, upload
from tabular_workbench.engine import COMMIT, MODELS, hardware, verify_sources
from tabular_workbench.imports import import_evidence
from tabular_workbench.runner import comparison, export_zip, preview, worker
from tabular_workbench.storage import ROOT, folder, read_json, report, write_json

app = FastAPI(title="Tabular Study · local-only")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
process = None
active_id = None
lock = threading.Lock()
UPLOAD_FILE = File(...)
TARGET_FIELD = Form(...)
TASK_FIELD = Form("classification")


class RunRequest(BaseModel):
    dataset: str = "361055"
    models: List[str] = Field(default_factory=lambda: ["rf", "mlp"])
    case: Literal["baseline", "noise", "rotation"] = "baseline"
    noise_multiplier: float = Field(2, ge=1, le=4)
    seed: int = Field(0, ge=0, le=100000)
    train_limit: int = Field(600, ge=100, le=5000)
    holdout_limit: int = Field(300, ge=50, le=2000)
    epochs: int = Field(5, ge=1, le=50)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    timeout_seconds: int = Field(600, ge=30, le=1800)

    @validator("models")
    def valid_models(cls, values):
        if (
            not values
            or len(values) > 7
            or len(values) != len(set(values))
            or any(m not in MODELS for m in values)
        ):
            raise ValueError("Choose1–7 distinct supported models")
        return values

    @validator("dataset")
    def valid_dataset(cls, value):
        from tabular_workbench.data import dataset_folder

        dataset_folder(value)
        return value


class CompareRequest(BaseModel):
    a: str
    b: str


@app.middleware("http")
async def local_boundary(request: Request, call_next):
    origin = request.headers.get("origin")
    if (
        request.method not in ("GET", "HEAD", "OPTIONS")
        and origin
        and origin != str(request.base_url).rstrip("/")
    ):
        return JSONResponse({"detail": "Cross-origin writes are not allowed"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(ValueError)
async def value_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(FileNotFoundError)
async def missing_file(request, error):
    return JSONResponse({"detail": "Local artifact not found"}, status_code=404)


@app.get("/api/health")
def health():
    verify_sources()
    return {
        "status": "ok",
        "mode": "local-only",
        "commit": COMMIT,
        "hardware": hardware(),
        "models": MODELS,
    }


@app.get("/api/datasets")
def datasets():
    return catalogue()


@app.get("/api/benchmark")
def benchmark_status():
    # Local metadata only: never consumes quota, queries Kaggle, or starts jobs.
    from tabular_workbench.benchmark import CONFIGS, CPU_MODELS, GPU_MODELS

    tasks = read_json(ROOT / "docs/TASKS.json") if (ROOT / "docs/TASKS.json").exists() else []
    submission = ROOT / "runtime/benchmark-submission.json"
    return {
        "tasks": len(tasks),
        "models": len(CPU_MODELS) + len(GPU_MODELS),
        "configs": CONFIGS,
        "submission": read_json(submission) if submission.exists() else None,
    }


@app.get("/api/benchmark/results")
def benchmark_results():
    from tabular_workbench.results import dashboard

    return dashboard()


def verified_ablation(expanded=False):
    if expanded:
        from tabular_workbench.rotation_results import load
    else:
        from tabular_workbench.ablation_results import load
    from tabular_workbench.snapshots import SnapshotInvalid, SnapshotUnavailable

    try:
        return load()
    except SnapshotUnavailable as error:
        raise HTTPException(404, str(error)) from None
    except SnapshotInvalid as error:
        raise HTTPException(409, str(error)) from None


@app.get("/api/benchmark/ablations/rotation")
def ablation_results():
    value = verified_ablation()
    return {key: value[key] for key in ("audit", "comparison", "protocol", "report_available")}


@app.get("/api/benchmark/ablations/rotation-expanded")
def rotation_expanded():
    value = verified_ablation(True)
    return {
        key: value[key] for key in ("audit", "comparison", "protocol", "info", "report_available")
    }


@app.get("/api/benchmark/ablations/rotation-expanded/export")
def rotation_expanded_export():
    return FileResponse(
        verified_ablation(True)["json_path"],
        media_type="application/json",
        filename="tabular-rotation-expanded.json",
    )


@app.get("/api/benchmark/ablations/rotation-expanded/report")
def rotation_expanded_report():
    value = verified_ablation(True)
    if not value["report_available"]:
        raise HTTPException(404, "Expanded report not generated yet")
    return FileResponse(
        value["pdf_path"], media_type="application/pdf", filename=value["pdf_path"].name
    )


@app.get("/api/benchmark/ablations/rotation/export")
def ablation_export():
    return FileResponse(
        verified_ablation()["json_path"],
        media_type="application/json",
        filename="tabular-rotation-ablation.json",
    )


@app.get("/api/benchmark/ablations/rotation/report")
def ablation_report():
    value = verified_ablation()
    if not value["report_available"]:
        raise HTTPException(404, "Verified ablation report not generated yet")
    return FileResponse(
        value["pdf_path"], media_type="application/pdf", filename=value["pdf_path"].name
    )


def verified_snapshot(identifier):
    from tabular_workbench.snapshots import SnapshotInvalid, SnapshotUnavailable, load

    try:
        return load(identifier)
    except SnapshotUnavailable as error:
        raise HTTPException(404, str(error)) from None
    except SnapshotInvalid as error:
        raise HTTPException(409, str(error)) from None


@app.get("/api/benchmark/snapshots")
def benchmark_snapshots():
    from tabular_workbench.snapshots import catalog

    return catalog()


@app.get("/api/benchmark/snapshots/{identifier}")
def benchmark_snapshot(identifier: str):
    snapshot = verified_snapshot(identifier)
    return {key: snapshot[key] for key in ("info", "data", "coverage")}


@app.get("/api/benchmark/snapshots/{identifier}/export")
def benchmark_snapshot_export(identifier: str):
    snapshot = verified_snapshot(identifier)
    return FileResponse(
        snapshot["json_path"],
        media_type="application/json",
        filename="tabular-verified-" + identifier + ".json",
    )


@app.get("/api/benchmark/snapshots/{identifier}/report")
def benchmark_snapshot_report(identifier: str):
    snapshot = verified_snapshot(identifier)
    if not snapshot["info"]["report_available"]:
        raise HTTPException(404, "Verified report not generated for this snapshot yet")
    return FileResponse(
        snapshot["pdf_path"], media_type="application/pdf", filename=snapshot["pdf_path"].name
    )


@app.get("/api/benchmark/export")
def benchmark_export():
    from tabular_workbench.results import RESULT_PATH

    if not RESULT_PATH.exists():
        raise HTTPException(404, "No independently verified benchmark yet")
    return FileResponse(
        RESULT_PATH, media_type="application/json", filename="tabular-verified-results.json"
    )


@app.get("/api/benchmark/report")
def benchmark_report():
    path = ROOT / "output/pdf/tabular-ket-qua-kiem-chung.pdf"
    if not path.exists():
        raise HTTPException(404, "Verified report not generated yet")
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.post("/api/datasets/{identifier}/fetch")
def download(identifier: str):
    return fetch(identifier)


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = UPLOAD_FILE, target: str = TARGET_FIELD, task: str = TASK_FIELD
):
    if task not in ("classification", "regression"):
        raise ValueError("Unknown task")
    payload = await file.read(10 * 1024 * 1024 + 1)
    return upload(payload, target, task == "regression")


@app.get("/api/runs")
def history():
    result = []
    for path in sorted(
        (ROOT / "artifacts/runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if path.is_dir() and (path / "status.json").exists():
            r = report(path.name)
            if "benchmark_config_index" in r.get("request", {}):
                # Benchmark uses compact numeric evidence, not demo X previews.
                continue
            result.append(
                {
                    key: r.get(key)
                    for key in ("id", "state", "request", "execution_location", "elapsed_seconds")
                }
            )
    return result


@app.post("/api/runs", status_code=202)
def start_run(request: RunRequest):
    global process, active_id
    with lock:
        if process is not None and process.is_alive():
            if active_id and report(active_id)["state"] in ("completed", "failed", "cancelled"):
                process.join(timeout=2)
            if process.is_alive():
                raise HTTPException(409, "One local worker is already running")
        identifier = uuid.uuid4().hex[:16]
        path = folder(identifier)
        path.mkdir(parents=True)
        cfg = request.dict()
        write_json(path / "request.json", cfg)
        write_json(path / "status.json", {"id": identifier, "state": "queued", "request": cfg})
        process = mp.get_context("spawn").Process(target=worker, args=(identifier,), daemon=False)
        process.start()
        active_id = identifier
        threading.Thread(
            target=watch_budget, args=(process, identifier, cfg["timeout_seconds"]), daemon=True
        ).start()
        return {"id": identifier, "state": "queued", "request": cfg}


def watch_budget(job, identifier, seconds):
    """Process-level timeout for uninterruptible tree fits, not a solver change."""
    job.join(timeout=seconds)
    if job.is_alive():
        job.terminate()
        job.join(timeout=5)
        result = report(identifier)
        if result["state"] in ("queued", "running"):
            result.update(
                state="failed",
                error="Local process time budget reached; incomplete fits are not exported",
            )
            write_json(folder(identifier) / "status.json", result)


@app.get("/api/runs/{identifier}")
def run_status(identifier: str):
    result = report(identifier)
    if (
        identifier == active_id
        and result["state"] in ("queued", "running")
        and process is not None
        and not process.is_alive()
    ):
        result.update(
            state="failed",
            error="Worker ended before a complete artifact was saved; see local training.log",
        )
        write_json(folder(identifier) / "status.json", result)
    return result


@app.post("/api/runs/{identifier}/cancel")
def cancel(identifier: str):
    r = report(identifier)
    if r["state"] not in ("queued", "running"):
        return {"state": r["state"], "cancel_requested": False}
    (folder(identifier) / "cancel").touch()
    return {"cancel_requested": True, "note": "Checked at model/neural batch boundaries"}


@app.get("/api/runs/{identifier}/preview")
def run_preview(identifier: str):
    return preview(identifier)


@app.get("/api/runs/{identifier}/export")
def export(identifier: str):
    return Response(
        export_zip(identifier),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="tabular-%s.zip"' % identifier},
    )


@app.post("/api/compare")
def compare_runs(request: CompareRequest):
    return comparison(request.a, request.b)


SOURCE_FILES = {
    "train": "src/train.py",
    "split": "src/generate_dataset_pipeline.py",
    "transforms": "src/data_transforms.py",
    "mlp": "src/models/tabular/bin/mlp.py",
    "resnet": "src/models/tabular/bin/resnet.py",
    "configs": "src/configs/all_model_configs.py",
}


@app.get("/api/source/{key}", response_class=PlainTextResponse)
def source(key: str):
    if key not in SOURCE_FILES:
        raise ValueError("Unknown source key")
    verify_sources()
    return (ROOT / "upstream" / SOURCE_FILES[key]).read_text(encoding="utf-8")


@app.get("/api/kaggle/status")
def kaggle_status():
    path = ROOT / "runtime/kaggle-preflight.json"
    if not path.exists():
        return {"verified": False, "note": "No read-only preflight yet"}
    r = read_json(path)
    submission = (
        read_json(ROOT / "runtime/kaggle-smoke-submission.json")
        if (ROOT / "runtime/kaggle-smoke-submission.json").exists()
        else {}
    )
    status_path = ROOT / "runtime/kaggle-smoke-status.json"
    status = read_json(status_path) if status_path.exists() else {}
    return {
        "verified": r.get("preflight_passed", False),
        "checked_at": r.get("checked_at"),
        "gpu_quota": r.get("quota", {}).get("gpu"),
        "notebook_private": r.get("notebook", {}).get("is_private"),
        "gpu_enabled": r.get("notebook", {}).get("gpu_enabled"),
        "job_submitted": bool(submission.get("job_submitted_this_call")),
        "smoke_status": status.get("status"),
        "status_checked_at": status.get("checked_at"),
        "smoke_budget_seconds": submission.get("max_runtime_seconds"),
        "note": "Cached read-only check; no token exposed; not proof a GPU is allocated",
    }


@app.post("/api/import")
async def import_result(file: UploadFile = UPLOAD_FILE):
    try:
        return import_evidence(await file.read(20 * 1024 * 1024 + 1))
    except (
        KeyError,
        AssertionError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
        EOFError,
        OverflowError,
        UnicodeError,
    ) as error:
        raise ValueError("Invalid numeric evidence archive: " + type(error).__name__) from None


if (ROOT / "dist").exists():
    app.mount("/assets", StaticFiles(directory=ROOT / "dist/assets"), name="assets")


@app.get("/")
def website():
    return FileResponse(ROOT / "dist/index.html")
