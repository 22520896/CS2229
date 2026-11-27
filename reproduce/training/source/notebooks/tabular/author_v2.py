"""Versioned data-boundary correction around the unchanged original-model worker.

Legacy workers and accepted notebooks stay byte-identical. This track supports
one numeric baseline model per process; unsupported diagnostics/uploads fail
before fitting. It changes no model/loss/optimizer/training body.
"""

import argparse
import ast
import contextlib
import copy
import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TRACK = "author-adapter-v2"
CACHE_ROOT = Path(
    "C:/Users/ADMIN/Downloads/THAC_SI/THUANTOAN/tabular-study/runtime/author-v2-data-20261030"
)
FAMILIES = ("rf", "gbt", "xgb", "hgbt", "mlp", "resnet", "ft_transformer")
NEURAL = ("mlp", "resnet", "ft_transformer")
_BINDING = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(request):
    assert request.get("execution_track") == TRACK, "An explicit new-track identity is required"
    assert request.get("case") == "baseline" and "paper_ablation" not in request
    assert len(request.get("models", [])) == 1 and request["models"][0] in FAMILIES
    assert str(request["dataset"]).isdigit(), "Only pinned public numeric inputs"
    assert request["seed"] in range(5)
    assert request["epochs"] == 300 and request["timeout_seconds"] == 1800
    assert request["train_limit"] == 10000 and request["holdout_limit"] == 50000
    assert request["device"] in ("cpu", "cuda")
    assert re.fullmatch(r"[a-f0-9]{64}", request.get("input_manifest_sha256", ""))


class _Transforms:
    def __init__(self, original, rng):
        self.original, self.rng = original, rng

    def __getattr__(self, name):
        return getattr(self.original, name)

    def gaussienize(self, *args, **kwargs):
        # The author's generate_dataset passes the same RNG after splitting.
        kwargs["rng"] = self.rng
        return self.original.gaussienize(*args, **kwargs)


class Adapter:
    def __init__(self, cache_root=CACHE_ROOT):
        self.cache_root = Path(cache_root).resolve()
        self.rng = None

    def load(self, task):
        import numpy as np

        assert str(task).isdigit()
        base = self.cache_root / "tasks" / str(task)
        meta = read(base / "manifest.json")
        assert meta["execution_track"] == TRACK and meta["id"] == str(task)
        assert meta["numeric_only"] and meta["public_only"]
        assert meta["task"] in ("classification", "regression")
        assert sha(base / "dataset.npz") == meta["sha256"]
        with np.load(base / "dataset.npz", allow_pickle=False) as saved:
            assert set(saved.files) == {"x", "y"}
            x, y = saved["x"], saved["y"]
        assert x.shape == (meta["rows"], meta["features"]) and y.shape == (len(x),)
        assert x.dtype == np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
        expected = np.float64 if meta["task"] == "regression" else np.int64
        assert y.dtype == expected, "Reject legacy float32 regression target caches"
        return x, y, meta

    def prepare(self, request):
        import numpy as np

        from tabular_workbench import data

        validate_request(request)
        assert (
            sha(self.cache_root / "tasks" / str(request["dataset"]) / "manifest.json")
            == request["input_manifest_sha256"]
        ), "Pinned v2 input manifest changed"
        # Only cache I/O is substituted. The frozen adapter still calls the
        # original splitter and pre-split RNG draw, now with original y dtype.
        with patch.object(data, "load", self.load):
            xx, ys, rows, rng, provenance = data.prepare(request)
        if "benchmark_config" in request:
            from tabular_workbench.benchmark import configuration

            expected = configuration(
                request["models"][0],
                provenance["task"] == "regression",
                request["benchmark_config_index"],
            )
            assert request["benchmark_config"] == expected, (
                "Configuration differs from the original registry draw"
            )
        native_dtype = str(ys[0].dtype)
        if provenance["task"] == "regression" and request["models"][0] in NEURAL:
            ys = [v.astype(np.float32) for v in ys]
        self.rng = rng
        provenance.update(
            execution_track=TRACK,
            raw_target_dtype=native_dtype,
            evaluation_target_dtype=str(ys[0].dtype),
            feature_transform_rng="same RandomState after original split",
            legacy_results_reused_as_new=False,
        )
        return xx, ys, rows, rng, provenance

    def author_seeds(self, task, model="rf", index=0):
        """Execute the frozen author's actual n_iter branch,without training/W&B.

        The original inline branch is compiled unchanged. No new approximation
        of the repeat thresholds or original data-generation pipeline is used.
        """
        import numpy as np
        import openml

        from tabular_workbench.benchmark import configuration
        from tabular_workbench.engine import modules, verify_sources

        verify_sources()
        x, y, meta = self.load(task)
        cfg = configuration(model, meta["task"] == "regression", index)
        cfg.update(
            data__method_name="openml_no_transform",
            data__keyword=int(task),
            n_iter="auto",
            train_prop=0.7,
            val_test_prop=0.3,
            max_train_samples=10000,
            max_val_samples=50000,
            max_test_samples=50000,
        )
        source = ROOT / "upstream/src/run_experiment.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        choices = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Subscript)
            and isinstance(node.test.left.value, ast.Name)
            and node.test.left.value.id == "config"
            and "n_iter" in ast.dump(node.test.left)
        ]
        assert len(choices) == 1
        body = ast.Module(body=[copy.deepcopy(choices[0])], type_ignores=[])
        ast.fix_missing_locations(body)
        namespace = dict(config=cfg, np=np, generate_dataset=modules()[1].generate_dataset)
        dataset = SimpleNamespace(
            default_target_attribute="already-separated-public-target",
            get_data=lambda **kwargs: (x.copy(), y.copy(), None, meta["feature_names"]),
        )
        with patch.object(
            openml.tasks, "get_task", lambda *a, **k: SimpleNamespace(get_dataset=lambda: dataset)
        ):
            exec(compile(body, str(source), "exec"), namespace)
        repeats = namespace["n_iter"]
        assert repeats in (1, 2, 3, 5)
        return dict(
            task=str(task),
            seed_indices=list(range(repeats)),
            n_iter=repeats,
            author_branch_source_sha256=sha(source),
            rule="Original executable n_iter=auto branch",
            preliminary_test_rows=int(namespace["x_test"].shape[0]),
            training_started=False,
        )

    def modules(self):
        from tabular_workbench.engine import modules

        assert self.rng is not None, "Prepare the original split before transforms"
        training, pipeline, transforms, registry, launcher = modules()
        return training, pipeline, _Transforms(transforms, self.rng), registry, launcher

    @contextlib.contextmanager
    def bound_worker(self, folders):
        from tabular_workbench import runner

        assert _BINDING.acquire(blocking=False), "One isolated worker per process"
        try:
            with patch.object(runner, "prepare", self.prepare), patch.object(
                runner, "modules", self.modules
            ), patch.object(runner, "folder", folders):
                yield runner
        finally:
            _BINDING.release()


def execute(request, cache_root=CACHE_ROOT, output_root=None):
    """Execute a fresh, explicit v2 request without changing any old run ID."""
    from tabular_workbench.engine import verify_sources
    from tabular_workbench.storage import digest, write_json

    validate_request(request)
    baseline = verify_sources()
    identifier = digest(request)[:16]
    root = Path(output_root or ROOT / "artifacts/author-v2/runs").resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / identifier
    assert not path.exists(), "Never overwrite or repeat a v2 run"
    path.mkdir()
    write_json(path / "request.json", request)
    adapter = Adapter(cache_root)
    with adapter.bound_worker(lambda key: root / key) as worker:
        worker.worker(identifier)
    result = read(path / ("result.json" if (path / "result.json").exists() else "status.json"))
    receipt = dict(
        id=identifier,
        execution_track=TRACK,
        state=result["state"],
        request_sha256=sha(path / "request.json"),
        adapter_sha256=sha(Path(__file__)),
        author_commit=baseline["commit"],
        original_training_body_modified=False,
        fixed_training_rng_disclosed=True,
        historical_results_changed=False,
    )
    write_json(path / "execution-receipt.json", receipt)
    return path, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    path, result = execute(read(args.request), args.cache_root, args.output_root)
    print(json.dumps(dict(directory=str(path), execution_track=TRACK, state=result["state"])))
