"""Bounded real-data fits with original model/train/transform bodies and immutable output."""

import contextlib
import hashlib
import io
import platform
import random
import time
import traceback
import zipfile
from typing import Any, Dict, List

import numpy as np
import sklearn
import torch
from skorch.callbacks import Callback, Checkpoint
from threadpoolctl import threadpool_limits

from tabular_workbench.data import prepare
from tabular_workbench.engine import (
    COMMIT,
    NEURAL,
    default_config,
    hardware,
    modules,
    verify_sources,
)
from tabular_workbench.storage import digest, folder, read_json, report, write_json


class Cancelled(RuntimeError):
    pass


def numpy_hash(values):
    value = hashlib.sha256()
    for array in values:
        value.update(array.dtype.str.encode())
        value.update(str(array.shape).encode())
        value.update(np.ascontiguousarray(array).tobytes())
    return value.hexdigest()


def display_loss(value):
    """JSON display only; never changes the value seen by training/checkpointing."""
    value = float(value)
    return value if np.isfinite(value) else None


def metrics(y, prediction, regression):
    if (
        y.ndim != 1
        or not y.size
        or y.shape != prediction.shape
        or not np.isfinite(y).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Nonfinite or malformed model prediction; no result substituted")
    if not regression and (not np.isin(y, [0, 1]).all() or not np.isin(prediction, [0, 1]).all()):
        raise ValueError("Binary classification truth/prediction required")
    if regression:
        from sklearn.metrics import mean_squared_error, r2_score

        return {
            "rmse": float(mean_squared_error(y, prediction) ** 0.5),
            "r2": float(r2_score(y, prediction)),
        }
    from sklearn.metrics import accuracy_score, confusion_matrix

    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def execute(identifier):
    path = folder(identifier)
    request = read_json(path / "request.json")
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []

    def progress(phase, **extra):
        write_json(
            path / "status.json",
            {
                "id": identifier,
                "state": "running",
                "request": request,
                "elapsed_seconds": time.perf_counter() - started,
                "progress": {
                    "phase": phase,
                    "models_done": len(results),
                    "models_total": len(request["models"]),
                    **extra,
                },
            },
        )

    def check_cancel():
        if (path / "cancel").exists():
            raise Cancelled(
                "Cancelled at a model/batch boundary; partial fits are not final results"
            )
        if time.perf_counter() - started > request["timeout_seconds"]:
            raise TimeoutError("Configured local time budget exceeded")

    try:
        progress("Kiểm tra source và dữ liệu")
        baseline = verify_sources()
        original, ys, rows, rng, provenance = prepare(request)
        regression = provenance["task"] == "regression"
        training, _, transforms, _, _ = modules()
        # A local diagnostic calls original transform functions, not a claim to
        # replay full paper's removal/Gaussianization/rotation/smoothing recipes.
        case = [a.copy() for a in original]
        if request["case"] == "noise":
            case = list(
                transforms.add_uninformative_features(
                    *case, *ys, multiplier=request["noise_multiplier"], rng=rng
                )[:3]
            )
        elif request["case"] == "rotation":
            case = list(transforms.apply_random_rotation(*case, *ys, rng=rng)[:3])
        if "paper_ablation" in request:
            from tabular_workbench.ablation import apply_recipe

            case, paper_trace = apply_recipe(original, ys, rng, request, provenance)
        payload = {}
        for index, name in enumerate(("train", "validation", "test")):
            payload["x_" + name] = case[index]
            payload["original_" + name] = original[index]
            payload["y_" + name] = ys[index]
            payload["ids_" + name] = rows[index]
        provenance.update(
            case=request["case"],
            case_input_hash=numpy_hash(case),
            original_input_hash=numpy_hash(original),
            transformed_features=case[0].shape[1],
            case_scope="Partial numerical-feature diagnostic; NOT full paper figure recipe",
        )
        if "paper_ablation" in request:
            provenance.update(paper_ablation=paper_trace, case_scope=paper_trace["scope"])
        for model_name in request["models"]:
            check_cancel()
            progress("Đang fit model thật", model=model_name)
            model_path = path / model_name
            model_path.mkdir(parents=True)
            np.random.seed(request["seed"])
            random.seed(request["seed"])
            torch.manual_seed(request["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request["seed"])
            torch.set_num_threads(1)
            device = "cpu"
            if model_name in NEURAL:
                if request["device"] == "cuda" and not torch.cuda.is_available():
                    raise ValueError(
                        "CUDA explicitly requested but unavailable; no silent CPU fallback"
                    )
                device = (
                    "cuda" if request["device"] != "cpu" and torch.cuda.is_available() else "cpu"
                )
            cfg = default_config(model_name, regression, device, request["epochs"], request["seed"])
            if "benchmark_config" in request:
                # Precommitted original search-space configuration, supplied only
                # by the offline benchmark harness (not a public UI parameter).
                cfg = dict(request["benchmark_config"])
                if model_name in NEURAL:
                    cfg.update(model__device=device, model__max_epochs=request["epochs"])
                else:
                    cfg["model__random_state"] = request["seed"]
                    if model_name in ("rf", "xgb"):
                        cfg["model__n_jobs"] = 1
            cfg.update(data__keyword=request["dataset"])
            if "paper_ablation" in request:
                cfg.update(request["paper_ablation"]["transform_config"])
            xx = [a.copy() for a in case]
            if model_name in NEURAL and "paper_ablation" not in request:
                transform_rng = np.random.RandomState(request["seed"] + 10000)
                xx = list(transforms.gaussienize(*xx, *ys, type="quantile", rng=transform_rng)[:3])
            xx = [a.astype(np.float32) for a in xx]
            yy = [
                a.astype(np.float32).reshape(-1, 1)
                if regression and model_name in NEURAL
                else a.copy()
                for a in ys
            ]

            class ProgressCallback(Callback):
                def __init__(self, label):
                    self.label = label

                def on_batch_end(self, net, **kwargs):
                    check_cancel()

                def on_epoch_end(self, net, **kwargs):
                    check_cancel()
                    progress(
                        "Training neural model",
                        model=self.label,
                        epoch=len(net.history),
                        epoch_limit=request["epochs"],
                        train_loss=display_loss(net.history[-1, "train_loss"]),
                        valid_loss=display_loss(net.history[-1, "valid_loss"]),
                        loss_nonfinite_observed=not all(
                            np.isfinite(net.history[-1, key])
                            for key in ("train_loss", "valid_loss")
                        ),
                    )

            checkpoint_path = model_path / "checkpoint"
            checkpoint_path.mkdir()
            # Original factory is called directly; an extra callback only reports
            # and cancels, does not alter any loss/optimizer/update/checkpoint rule.
            if model_name in NEURAL:
                original_factory = training.create_model

                def factory_with_reporting(
                    *args,
                    _factory=original_factory,
                    _checkpoint=checkpoint_path,
                    _label=model_name,
                    **kwargs,
                ):
                    raw_model = _factory(*args, **kwargs)
                    callbacks = raw_model.callbacks
                    for cb in callbacks:
                        if isinstance(cb, Checkpoint):
                            cb.dirname = str(_checkpoint)
                    raw_model.set_params(
                        callbacks=callbacks + [("local_progress", ProgressCallback(_label))]
                    )
                    return raw_model

                # Scoped constructor decoration only; original train_model still
                # creates its TransformedTargetRegressor/other wrappers unchanged.
                training.create_model = factory_with_reporting
                try:
                    with threadpool_limits(limits=1):
                        model_started = time.perf_counter()
                        fitted = training.train_model(
                            request["seed"], xx[0], yy[0], None, cfg, identifier
                        )
                        fit_seconds = time.perf_counter() - model_started
                finally:
                    training.create_model = original_factory
                net = fitted.regressor_ if regression and cfg["transformed_target"] else fitted
                best_path = checkpoint_path / ("params_%s.pt" % identifier)
                if not best_path.exists():
                    raise RuntimeError(
                        "Original best checkpoint missing; refusing last-epoch substitution"
                    )
                net.load_params(f_params=str(best_path))
                history = []
                for epoch in net.history:
                    history.append(
                        {
                            key: display_loss(epoch[key])
                            for key in ("epoch", "train_loss", "valid_loss")
                        }
                    )
                stop = (
                    "early_stopping" if len(history) < request["epochs"] else "epoch_budget_reached"
                )
                model_hash = numpy_hash(
                    [
                        value.detach().cpu().numpy()
                        for _, value in sorted(net.module_.state_dict().items())
                    ]
                )
                net.save_params(f_params=str(model_path / "weights.pt"))
            else:
                with threadpool_limits(limits=1):
                    model_started = time.perf_counter()
                    fitted = training.train_model(
                        request["seed"], xx[0], yy[0], None, cfg, identifier
                    )
                    fit_seconds = time.perf_counter() - model_started
                history, stop = [], "fit_returned"
                import joblib

                joblib.dump(
                    fitted, model_path / "local-model.joblib"
                )  # Trusted LOCAL save only; never import uploaded pickle.
                model_hash = hashlib.sha256(
                    (model_path / "local-model.joblib").read_bytes()
                ).hexdigest()
            check_cancel()
            predictions = [np.asarray(fitted.predict(a)).ravel() for a in xx]
            model_metrics = {
                key: metrics(y.ravel(), p, regression)
                for key, y, p in zip(("train", "validation", "test"), ys, predictions)
            }
            # Direct upstream evaluator on the same restored model; disable its
            # relative-path checkpoint reload/delete (already restored above).
            evaluation_cfg = {**cfg, "model__use_checkpoints": False}
            upstream_scores = training.evaluate_model(
                fitted,
                xx[0],
                yy[0],
                xx[1],
                yy[1],
                xx[2],
                yy[2],
                evaluation_cfg,
                identifier,
                return_r2=regression,
                delete_checkpoint=False,
            )
            primary = "r2" if regression else "accuracy"
            for key, value in zip(("train", "validation", "test"), upstream_scores):
                np.testing.assert_allclose(
                    model_metrics[key][primary], value, atol=1e-12, rtol=1e-12
                )
            for key, prediction, features in zip(("train", "validation", "test"), predictions, xx):
                payload[model_name + "_prediction_" + key] = prediction
                payload[model_name + "_input_" + key] = features
            results.append(
                {
                    "model": model_name,
                    "device": device,
                    "config": cfg,
                    "metrics": model_metrics,
                    "fit_seconds": fit_seconds,
                    "history": history,
                    "history_nonfinite_observed": any(
                        h[key] is None for h in history for key in ("train_loss", "valid_loss")
                    ),
                    "stop_reason": stop,
                    "model_hash": model_hash,
                    "preprocessing": "Original train-only QuantileTransformer(normal)"
                    if model_name in NEURAL
                    else "Unchanged numerical input",
                    "input_hash": numpy_hash(xx),
                    "upstream_metric_parity": "passed",
                }
            )
            progress("Model đã lưu", model=model_name)
            if "paper_ablation" in request:
                results[-1]["preprocessing"] = (
                    "Original RF removal -> train-only Gaussianization -> original rotation off/on; no extra neural transform"
                )
        np.savez_compressed(path / "arrays.npz", **payload)
        warning = [
            "Case nhỏ / partial replication; không thay thế full45-dataset benchmark.",
            "Model seed cố định là adapter reproducibility; không tuyên bố refit bitwise với code/CPU/GPU khác.",
            "Không chọn model theo test; mọi model được cấu hình trước và giữ cả kết quả thấp.",
        ]
        if request["case"] != "baseline":
            warning.append(
                "Biến đổi dùng hàm repo gốc nhưng recipe demo chưa phải toàn bộ thí nghiệm hình paper."
            )
        if any(r["stop_reason"] == "epoch_budget_reached" for r in results):
            warning.append("Neural model đạt giới hạn epoch demo; không gọi là đã hội tụ.")
        if "paper_ablation" in request:
            warning = [
                "Partial Figure5 recipe:2preselected classification tasks/default configurations/3fixedseeds.",
                "Not full paper HPO/normalized accuracy/search shuffles; report paired raw accuracy, not recreated Figure5 boxes.",
                "All4conditions are new fits; old raw-rotation diagnostics and main benchmark are not interchangeable controls.",
                "Original model/loss/optimizer/train/evaluator bodies unchanged; fixed training seeds are an explicit adapter choice, not bitwise author reproduction.",
                "Epoch/time limits and unsuccessful fits must remain visible; no convergence/global optimum claim.",
            ]
        if any(r["history_nonfinite_observed"] for r in results):
            warning.append(
                "Loss history có NaN/Inf, hiển thị null (không thay bằng0). Training không bị JSON display thay đổi; prediction từ best checkpoint vẫn phải hữu hạn và khớp evaluator gốc."
            )
        write_json(
            path / "result.json",
            {
                "id": identifier,
                "state": "completed",
                "request": request,
                "results": results,
                "provenance": provenance,
                "upstream_commit": COMMIT,
                "source_hash": digest(baseline),
                "source_modified": False,
                "warnings": warning,
                "hardware": hardware(),
                "versions": {
                    "numpy": np.__version__,
                    "sklearn": sklearn.__version__,
                    "python": platform.python_version(),
                },
                "elapsed_seconds": time.perf_counter() - started,
                "arrays_sha256": hashlib.sha256((path / "arrays.npz").read_bytes()).hexdigest(),
                "execution_location": "local",
            },
        )
        write_json(
            path / "status.json", {"id": identifier, "state": "completed", "request": request}
        )
    except Exception as error:
        (path / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(
            path / "status.json",
            {
                "id": identifier,
                "state": "cancelled" if isinstance(error, Cancelled) else "failed",
                "request": request,
                "error": str(error),
                "partial_models": results,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )


def worker(identifier):
    path = folder(identifier)
    if (path / "result.json").exists():
        raise ValueError("Completed run is immutable; choose a fresh run identifier")
    with (path / "training.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(
        log
    ), contextlib.redirect_stderr(log):
        execute(identifier)


def saved_arrays(identifier):
    r = report(identifier)
    if r["state"] != "completed":
        raise ValueError("Completed run required")
    path = folder(identifier) / "arrays.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != r["arrays_sha256"]:
        raise ValueError("Frozen arrays hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def preview(identifier):
    r, a = report(identifier), saved_arrays(identifier)
    regression = r["provenance"]["task"] == "regression"
    labels = r["provenance"]["feature_names"]
    return {
        "id": identifier,
        "feature_names": ["rotated-axis-%d" % i for i in range(a["x_train"].shape[1])]
        if r["request"]["case"] == "rotation"
        else labels + ["noise-%d" % i for i in range(a["x_train"].shape[1] - len(labels))],
        "original_feature_names": labels,
        "features": a["x_test"][:20].tolist(),
        "original_features": a["original_test"][:20].tolist(),
        "row_ids": a["ids_test"][:20].tolist(),
        "truth": a["y_test"][:20].tolist(),
        "predictions": {
            m["model"]: a[m["model"] + "_prediction_test"][:20].tolist() for m in r["results"]
        },
        "scatter": {
            m["model"]: {
                "truth": a["y_test"][:200].tolist(),
                "predicted": a[m["model"] + "_prediction_test"][:200].tolist(),
            }
            for m in r["results"]
        },
        "note": "Test predictions/features from frozen backend arrays; projection is not used for fitting",
        "regression": regression,
    }


def comparison(a_id, b_id):
    a, b = report(a_id), report(b_id)
    if (
        a["state"] != "completed"
        or b["state"] != "completed"
        or a["provenance"]["task"] != b["provenance"]["task"]
    ):
        raise ValueError("Two completed runs with same task required")
    common = sorted(set(r["model"] for r in a["results"]) & set(r["model"] for r in b["results"]))
    primary = "r2" if a["provenance"]["task"] == "regression" else "accuracy"
    deltas = []
    for model in common:
        ra = next(r for r in a["results"] if r["model"] == model)
        rb = next(r for r in b["results"] if r["model"] == model)
        va, vb = ra["metrics"]["test"][primary], rb["metrics"]["test"][primary]
        deltas.append({"model": model, "metric": primary, "a": va, "b": vb, "b_minus_a": vb - va})
    return {
        "a": a_id,
        "b": b_id,
        "values": deltas,
        "same_dataset_split": all(
            a["provenance"][k] == b["provenance"][k] for k in ("sha256", "split_hash")
        ),
        "same_source": a["source_hash"] == b["source_hash"],
        "changes": [
            {"parameter": k, "a": v, "b": b["request"].get(k)}
            for k, v in a["request"].items()
            if b["request"].get(k) != v
        ],
        "note": "Presentation subtraction of frozen metrics only; no model refit or algorithm recomputation",
    }


def export_zip(identifier):
    saved_arrays(identifier)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("result.json", "arrays.npz"):
            archive.writestr(name, (folder(identifier) / name).read_bytes())
    return buffer.getvalue()
