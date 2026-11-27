"""Read-only, allowlisted ablation evidence; no training, credentials or Kaggle."""

import hashlib
import json

from tabular_workbench.snapshots import SnapshotInvalid, SnapshotUnavailable
from tabular_workbench.storage import ROOT, digest, read_json


def locations():
    base = ROOT / "runtime/one-notebook/round-5"
    return (
        base,
        base / "output/version-4/verified-ablation",
        ROOT / "output/pdf/tabular-benchmark-va-ablation.pdf",
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    base, directory, pdf = locations()
    try:
        manifest = read_json(directory / "manifest.json")
        if set(manifest["files"]) != {"audit.json", "comparison.json"}:
            raise SnapshotInvalid("Unexpected ablation evidence files")
        for name, expected in manifest["files"].items():
            if sha(directory / name) != expected:
                raise SnapshotInvalid("Ablation evidence hash mismatch")
        audit, comparison = (
            read_json(directory / "audit.json"),
            read_json(directory / "comparison.json"),
        )
        receipt = read_json(base / "dispatch.json")
        frozen = read_json((ROOT / receipt["release"]) / "manifest.json")
        if frozen != receipt["manifest"] or not audit["passed"] or not manifest["passed"]:
            raise SnapshotInvalid("Unverified ablation release")
        for key in ("reference", "version", "plan_hash", "receipt_sha256", "evidence_sha256"):
            if audit[key] != manifest[key]:
                raise SnapshotInvalid("Ablation audit identity differs")
        if (
            (
                receipt["reference"],
                receipt["version"],
                frozen["plan_hash"],
                sha(base / "dispatch.json"),
            )
            != (
                audit["reference"],
                audit["version"],
                audit["plan_hash"],
                audit["receipt_sha256"],
            )
            or audit["study_kind"] != frozen["study_kind"]
            or audit["study_kind"] != "partial_figure5_ablation"
        ):
            raise SnapshotInvalid("Ablation account/version/plan mismatch")
        if (
            audit["new_verified"] != comparison["verified_new_fits"]
            or audit["planned"] != comparison["planned_new_fits"]
        ):
            raise SnapshotInvalid("Ablation count mismatch")
        plan = read_json((ROOT / receipt["release"]).parent / "plan.json")
        if (
            digest(plan) != audit["plan_hash"]
            or sha((ROOT / receipt["release"]) / "tabular_resume.ipynb") != frozen["code_hash"]
        ):
            raise SnapshotInvalid("Frozen ablation plan/source changed")
        baseline = plan["main_baseline"]
        if sha(ROOT / baseline["audit_path"]) != baseline["audit_sha256"]:
            raise SnapshotInvalid("Main benchmark baseline changed")
        report_available = False
        if pdf.exists() and (directory / "report.json").exists():
            report = read_json(directory / "report.json")
            if report != {
                "pdf_sha256": sha(pdf),
                "ablation_audit_sha256": sha(directory / "audit.json"),
                "main_audit_sha256": baseline["audit_sha256"],
            }:
                raise SnapshotInvalid("PDF identity does not match both audited studies")
            report_available = True
        return {
            "audit": audit,
            "comparison": comparison,
            "protocol": plan["protocol"],
            "report_available": report_available,
            "json_path": directory / "comparison.json",
            "pdf_path": pdf,
        }
    except FileNotFoundError:
        raise SnapshotUnavailable("Ablation has no independently audited local output") from None
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SnapshotInvalid("Malformed ablation evidence: " + type(error).__name__) from None
