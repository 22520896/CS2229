"""Optional explicit original-worker training;not called by report or notebooks.

Use the pinned Python3.8 training environment,not the Python3.11 analysis kernel.
"""

from pathlib import Path
from hashlib import md5, sha256
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "training/source"


def R(path):
    return json.loads(Path(path).read_bytes())


def frozen():
    pins = R(BASE / "configs/frozen_training_sources.json")
    for name, pin in pins["files"].items():
        assert sha256((SOURCE / name).read_bytes()).hexdigest() == pin, name
    return pins


def prepare():
    frozen()
    version = R(BASE / "configs/source_versions.json")
    upstream = SOURCE / "upstream"
    if not upstream.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                version["upstream_repository"],
                str(upstream),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(upstream),
                "checkout",
                "--detach",
                version["upstream_commit"],
            ],
            check=True,
        )
    head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == version["upstream_commit"], (
        "Keep any existing checkout;do not overwrite it"
    )
    baseline = R(SOURCE / "docs/UPSTREAM_BASELINE.json")
    for name, pin in baseline["sha256_lf"].items():
        assert (
            sha256((upstream / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == pin
        ), name
    sys.path[:0] = [str(SOURCE / "scripts"), str(SOURCE / "tools/benchmarks")]
    spec = importlib.util.spec_from_file_location(
        "download_transport_base", SOURCE / "scripts/download_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from public_arff_transport import download_arff
    import openml

    meta = R(BASE / "configs/sample_inputs.json")["tasks"]["361060"]
    cache = BASE / ".runtime/input-cache"
    cache.mkdir(parents=True, exist_ok=True)
    openml.config.set_cache_directory(str(cache))
    dataset = openml.datasets.get_dataset(
        meta["dataset_id"], download_data=False, download_qualities=False
    )
    assert (int(dataset.dataset_id), int(dataset.version), dataset.md5_checksum) == (
        meta["dataset_id"],
        meta["version"],
        meta["md5"],
    )
    raw = cache / "org/openml/www/datasets" / str(meta["dataset_id"]) / "dataset.arff"
    download_arff(dataset.url, raw, meta["md5"])
    assert md5(raw.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == meta["md5"]
    return cache


def run(kind):
    assert sys.version_info[:2] == (3, 8), (
        "Training uses the pinned Python3.8 source environment"
    )
    cache = prepare()
    plan = R(BASE / "configs/sample_jobs.json")
    indices = [i for i, j in enumerate(plan["jobs"]) if j["group"] == kind]
    assert len(indices) == 1
    folder = BASE / ".runtime" / ("fit-" + kind + "-" + uuid.uuid4().hex[:10])
    folder.mkdir(parents=True)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    shutil.copyfile(
        SOURCE / "tools/benchmarks/demo_pilot_worker.py",
        folder / "repeat_seed_adapter.py",
    )
    env = dict(
        os.environ,
        TABULAR_PILOT_ROOT=str(SOURCE),
        TABULAR_PILOT_WORK=str(folder),
        TABULAR_PILOT_CACHE=str(cache),
        TABULAR_PILOT_INPUTS=str(BASE / "configs/sample_inputs.json"),
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    worker = SOURCE / "tools/benchmarks/published_pilot_worker.py"
    proc = subprocess.Popen(
        [sys.executable, "-B", str(worker), str(folder / "plan.json"), str(indices[0])],
        cwd=folder,
        env=env,
    )
    try:
        code = proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        import psutil

        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        proc.terminate()
        proc.wait()
        raise RuntimeError(
            "Original1800s ceiling reached;new partial artifacts retained"
        )
    assert code == 0, "Fit failed;keep its logs and output,no automatic retry"
    print(
        json.dumps(
            dict(
                output=str(folder), reference_outputs_unchanged=True, fit_retried=False
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--prepare",
        action="store_true",
        help="Verify sources and fetch the one pinned public input;no model fit",
    )
    g.add_argument(
        "--run",
        choices=["noise_features", "rotation", "target_smoothing"],
        help="Explicitly train one representative originalRFfit",
    )
    args = p.parse_args()
    if args.prepare:
        prepare()
        print("Pinned sources and public input verified;no fit started")
    else:
        run(args.run)
