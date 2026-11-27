"""Notebook setup/run helpers. Kaggle execution has no credential dependency."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def env_for(root):
    env = os.environ.copy()
    env.update(
        WANDB_MODE="disabled",
        WANDB_ERROR_REPORTING="false",
        WANDB_SENTRY_DSN="",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        UV_CACHE_DIR="/kaggle/temp/tabular-uv-cache",
        UV_PYTHON_INSTALL_DIR="/kaggle/temp/tabular-python-runtime",
    )
    return env


def immutable_json(path, value):
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == value
    else:
        path.write_text(content, encoding="utf-8")


def setup(root, start, seconds, require_cuda=True):
    env = env_for(root)

    def command(args, timeout):
        remaining = seconds - (time.monotonic() - start) - 180
        if remaining <= 30:
            raise TimeoutError("Setup exceeded finite session budget")
        subprocess.run(args, cwd=root, env=env, check=True, timeout=min(timeout, remaining))

    pin = "9d54cf53d9fd3159e367e70a00005f4fcbf2c79d"
    if not (root / "upstream/.git").exists():
        command(
            [
                "git",
                "clone",
                "https://github.com/LeoGrin/tabular-benchmark.git",
                str(root / "upstream"),
            ],
            120,
        )
        command(["git", "-C", str(root / "upstream"), "checkout", "--detach", pin], 30)
    else:
        assert (
            subprocess.check_output(
                ["git", "-C", str(root / "upstream"), "rev-parse", "HEAD"], text=True
            ).strip()
            == pin
        )
    command([sys.executable, "-m", "pip", "install", "-q", "uv==0.7.10"], 120)
    command(
        [sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--python", "3.8", "--quiet"],
        600,
    )
    python = str(root / ".venv/bin/python")
    command(
        [
            python,
            "-c",
            "import torch;from tabular_workbench.engine import verify_sources;verify_sources();"
            + (
                "assert torch.cuda.is_available(),'CUDA allocation required'"
                if require_cuda
                else "print('CPU-only profile; CUDA not required or allocated')"
            ),
        ],
        45,
    )
    print(
        json.dumps(
            {
                "setup_verified": True,
                "original_source_modified": False,
                "seconds": time.monotonic() - start,
                "formula_changed": False,
            }
        )
    )
    return python


def execute(root, python, start, seconds, session_id):
    destination = root / "artifacts/continuation"
    destination.mkdir(parents=True, exist_ok=True)
    output = Path("/kaggle/working")
    summary = {
        "schema": 2,
        "state": "not_started",
        "session_id": session_id,
        "maximum_seconds": seconds,
        "resume_level": "fit-level only",
    }
    try:
        remaining = int(seconds - (time.monotonic() - start))
        if remaining < 240:
            raise TimeoutError("No fit budget after setup")
        command = [
            python,
            "-m",
            "tabular_workbench.continuation",
            "run",
            "--plan",
            str(root / "continuation-input/plan.json"),
            "--resume",
            str(root / "continuation-input/resume.json"),
            "--output",
            str(destination),
            "--seconds",
            str(remaining),
            "--session-id",
            session_id,
        ]
        subprocess.run(
            command, cwd=root, env=env_for(root), check=True, timeout=max(1, remaining - 90)
        )
        summary = json.loads((destination / "continuation.json").read_text())
    except Exception as error:
        summary.update(state="setup_or_session_failed", error_type=type(error).__name__)
        if (destination / "continuation.json").exists():
            summary["partial"] = json.loads((destination / "continuation.json").read_text())
    finally:
        summary["notebook_elapsed_seconds"] = time.monotonic() - start
        (output / "tabular-continuation.json").write_text(
            json.dumps(summary, allow_nan=False), encoding="utf-8"
        )
        if (destination / "continuation.json").exists():
            code = (
                "from pathlib import Path;from tabular_workbench.continuation import export_session;import json;print(json.dumps(export_session(Path("
                + repr(str(destination))
                + "),Path('/kaggle/working/tabular-continuation-evidence.zip'))))"
            )
            subprocess.run(
                [python, "-c", code], cwd=root, env=env_for(root), check=True, timeout=90
            )
        print(
            json.dumps(
                {
                    "state": summary["state"],
                    "seconds": time.monotonic() - start,
                    "notebook_self_submits": False,
                    "scheduled_monitor": False,
                }
            )
        )


def prepare_public_data(root, python, start, seconds, config):
    """Cache exactly one pinned full public table; failure stops before any fit."""
    remaining = int(seconds - (time.monotonic() - start) - 240)
    code = "from pathlib import Path;from tabular_workbench.continuation import read_initial;from tabular_workbench.data import fetch;read_initial(Path('continuation-input/plan.json'),Path('continuation-input/resume.json'));print(fetch('361281'))"
    try:
        if remaining < 60:
            raise TimeoutError("Insufficient public-data preparation budget")
        subprocess.run(
            [python, "-c", code],
            cwd=root,
            env=env_for(root),
            check=True,
            timeout=min(1800, remaining),
        )
    except Exception as error:
        Path("/kaggle/working/tabular-continuation.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "state": "public_data_setup_failed",
                    "error_type": type(error).__name__,
                    "plan_hash": config["plan_hash"],
                    "initial_completed_skipped": config["initial_completed_skipped"],
                    "completed_new": 0,
                    "remaining_jobs": config["counts"]["adapter_retry"],
                    "notebook_elapsed_seconds": time.monotonic() - start,
                }
            ),
            encoding="utf-8",
        )
        raise


def unpack_checkpoint(archive, destination, expected_plan):
    """Only prior audited output JSON + numeric vectors, never model pickle/code."""
    import re
    import zipfile

    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        assert len(infos) <= 1500 and len({i.filename for i in infos}) == len(infos)
        assert sum(i.file_size for i in infos) <= 2 * 1024**3
        manifest = json.loads(source.read("manifest.json"))
        assert manifest["schema"] == 2 and manifest["plan_hash"] == expected_plan
        assert (
            not manifest["credentials"]
            and not manifest["model_pickle"]
            and not manifest["raw_features"]
        )
        assert set(manifest["files"]) == set(source.namelist()) - {"manifest.json"}
        for name, expected in manifest["files"].items():
            assert name == "continuation.json" or re.fullmatch(r"[0-9a-f]{16}/numeric\.npz", name)
            content = source.read(name)
            assert hashlib.sha256(content).hexdigest() == expected
            path = destination / name
            path.resolve().relative_to(destination.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == content
            else:
                path.write_bytes(content)
