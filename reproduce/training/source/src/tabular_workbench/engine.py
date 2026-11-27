"""Direct imports from the immutable authors' repository, no model rewrite."""

import hashlib
import importlib
import os
import sys
from functools import lru_cache
from typing import Any, cast

from tabular_workbench.storage import ROOT, read_json

COMMIT = "9d54cf53d9fd3159e367e70a00005f4fcbf2c79d"
MODELS = {
    "rf": "Random Forest",
    "gbt": "Gradient Boosting",
    "hgbt": "HistGradientBoosting",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "resnet": "ResNet",
    "ft_transformer": "FT-Transformer",
}
NEURAL = {"mlp", "resnet", "ft_transformer"}


def verify_sources():
    baseline = read_json(ROOT / "docs/UPSTREAM_BASELINE.json")
    for name, expected in baseline["sha256_lf"].items():
        actual = hashlib.sha256(
            (ROOT / "upstream" / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        if actual != expected:
            raise RuntimeError("Pinned upstream source drift: " + name)
    return baseline


@lru_cache(maxsize=1)
def modules():
    verify_sources()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "upstream/src"))
    os.environ["PROJECT_DIR"] = str(ROOT)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_ERROR_REPORTING"] = "false"
    os.environ["WANDB_SENTRY_DSN"] = ""
    # Notebook hosts export a Jupyter backend to subprocesses. This isolated
    # script is headless; only plotting runtime changes, not model arithmetic.
    os.environ["MPLBACKEND"] = "Agg"
    training = importlib.import_module("train")
    pipeline = importlib.import_module("generate_dataset_pipeline")
    transforms = importlib.import_module("data_transforms")
    registry = importlib.import_module("configs.all_model_configs")
    original_runner = importlib.import_module("run_experiment")
    os.environ["PROJECT_DIR"] = str(ROOT)
    return training, pipeline, transforms, registry, original_runner


def default_config(model, regression, device, epochs, seed):
    if model not in MODELS:
        raise ValueError("Unknown model")
    _, _, _, registry, runner = modules()
    specification = registry.total_config[model]["regression" if regression else "classif"][
        "default"
    ]
    cfg = {
        name: spec["value"] if "value" in spec else spec["values"][0]
        for name, spec in specification.items()
    }
    cfg.update(regression=regression, data__regression=regression, data__categorical=False)
    if model in NEURAL:
        cfg.update(model__device=device, model__max_epochs=epochs)
    else:
        cfg["model__random_state"] = seed
        if model in ("rf", "xgb"):
            cfg["model__n_jobs"] = 1
    return runner.modify_config(cfg)


def hardware():
    import platform

    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": cast(Any, torch.backends.cudnn).allow_tf32,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
    }
