"""Allowlisted local snapshot reader; no Kaggle, model fitting or metric changes."""

import hashlib
import json

from tabular_workbench.storage import ROOT, read_json


class SnapshotUnavailable(ValueError):
    pass


class SnapshotInvalid(ValueError):
    pass


def locations(identifier):
    root = ROOT / "runtime/one-notebook"
    paths = {
        "first": (
            ROOT / "artifacts/benchmark",
            None,
            "verified-dashboard.json",
            ROOT / "output/pdf/tabular-ket-qua-kiem-chung.pdf",
            "Đợt đầu · 7 shard",
        ),
        "v12": (
            root / "output/version-12/verified-offline",
            root / "dispatch.json",
            "dashboard.json",
            ROOT / "output/pdf/tabular-ket-qua-kiem-chung-v12.pdf",
            "Gộp đến version 12 · account đầu",
        ),
        "round-2": (
            root / "round-2/output/version-1/verified-offline",
            root / "round-2/dispatch.json",
            "dashboard.json",
            ROOT / "output/pdf/tabular-ket-qua-kiem-chung-round2.pdf",
            "Phiên tiếp nối · account thứ hai",
        ),
        "round-3": (
            root / "round-3/output/version-2/verified-offline",
            root / "round-3/dispatch.json",
            "dashboard.json",
            ROOT / "output/pdf/tabular-ket-qua-kiem-chung-round3.pdf",
            "Gộp đến version 2 · đã bổ sung dataset còn thiếu",
        ),
        "round-4": (
            root / "round-4/output/version-3/verified-offline",
            root / "round-4/dispatch.json",
            "dashboard.json",
            ROOT / "output/pdf/tabular-benchmark-va-ablation.pdf",
            "Gộp đến version 3 · đã giải quyết timeout",
        ),
    }
    if identifier not in paths:
        raise SnapshotUnavailable("Unknown snapshot; only approved local snapshots are accepted")
    return paths[identifier]


def load(identifier):
    directory, receipt_path, filename, pdf, label = locations(identifier)
    try:
        if receipt_path is None:
            data = read_json(directory / filename)
            audit = read_json(directory / "verification.json")
            if not audit["passed"] or data["audit"] != audit:
                raise SnapshotInvalid("First-batch audit differs")
            coverage = None
        else:
            manifest = read_json(directory / "manifest.json")
            receipt = read_json(receipt_path)
            expected = {"audit.json", "dashboard.json", "coverage.json"}
            if (
                set(manifest["files"]) != expected
                or not manifest["offline_only"]
                or manifest["remote_writes"]
            ):
                raise SnapshotInvalid("Invalid local verified manifest")
            for name, value in manifest["files"].items():
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != value:
                    raise SnapshotInvalid("Verified snapshot hash mismatch; re-audit required")
            if (manifest["reference"], manifest["version"], manifest["plan_hash"]) != (
                receipt["reference"],
                receipt["version"],
                receipt["manifest"]["plan_hash"],
            ) or manifest["receipt_sha256"] != hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest():
                raise SnapshotInvalid("Snapshot account/version does not match its receipt")
            data, coverage = read_json(directory / filename), read_json(directory / "coverage.json")
            audit = read_json(directory / "audit.json")
            if data["audit"] != audit or coverage["snapshot_checked_at"] != audit["checked_at"]:
                raise SnapshotInvalid("Snapshot/audit/coverage identity differs")
        if (
            not data["verified"]
            or not audit["passed"]
            or data["counts"]["completed"] != audit["completed_numeric_runs_verified"]
        ):
            raise SnapshotInvalid("Snapshot successful count is not independently verified")
        info = {
            "id": identifier,
            "label": label,
            "completed": data["counts"]["completed"],
            "checked_at": data["checked_at"],
            "report_available": bool(pdf and pdf.exists()),
            "reference": audit.get("reference"),
            "version": audit.get("version"),
            "merged": receipt_path is not None,
        }
        if identifier == "round-4" and info["report_available"]:
            from tabular_workbench.ablation_results import load as load_ablation

            try:
                info["report_available"] = load_ablation()["report_available"]
            except SnapshotUnavailable:
                info["report_available"] = False
        return {
            "info": info,
            "data": data,
            "coverage": coverage,
            "json_path": directory / filename,
            "pdf_path": pdf,
        }
    except FileNotFoundError:
        raise SnapshotUnavailable("No audited local output for this snapshot yet") from None
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SnapshotInvalid("Malformed verified snapshot: " + type(error).__name__) from None


def catalog():
    available, unavailable = [], []
    for identifier in ("first", "v12", "round-2", "round-3", "round-4"):
        try:
            available.append(load(identifier)["info"])
        except (SnapshotUnavailable, SnapshotInvalid) as error:
            unavailable.append({"id": identifier, "note": str(error)})
    return {
        "snapshots": available,
        "unavailable": unavailable,
        "default": available[-1]["id"] if available else None,
        "local_only": True,
    }
