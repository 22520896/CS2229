"""Allowlisted16task result reader. No model, network or credential imports."""

import json

from tabular_workbench.ablation_results import sha
from tabular_workbench.snapshots import SnapshotInvalid, SnapshotUnavailable
from tabular_workbench.storage import ROOT, digest, read_json


def locations():
    base = ROOT / "runtime/one-notebook/round-6"
    return (
        base,
        base / "output/version-5/verified-rotation-expanded",
        ROOT / "output/pdf/tabular-rotation-16-datasets.pdf",
    )


def load():
    base, directory, pdf = locations()
    try:
        manifest = read_json(directory / "manifest.json")
        if set(manifest["files"]) != {"audit.json", "comparison.json"}:
            raise SnapshotInvalid("Unexpected expanded evidence files")
        for name, expected in manifest["files"].items():
            if sha(directory / name) != expected:
                raise SnapshotInvalid("Expanded evidence hash mismatch")
        audit, comparison = (
            read_json(directory / "audit.json"),
            read_json(directory / "comparison.json"),
        )
        receipt = read_json(base / "dispatch.json")
        release = ROOT / receipt["release"]
        release.resolve().relative_to((ROOT / "runtime/one-notebook").resolve())
        frozen, plan = read_json(release / "manifest.json"), read_json(release.parent / "plan.json")
        if frozen != receipt["manifest"] or not audit["passed"] or not manifest["passed"]:
            raise SnapshotInvalid("Expanded result not independently verified")
        for key in ("reference", "version", "plan_hash", "receipt_sha256", "evidence_sha256"):
            if audit[key] != manifest[key]:
                raise SnapshotInvalid("Expanded audit identity mismatch")
        if (audit["reference"], audit["version"], audit["plan_hash"], audit["receipt_sha256"]) != (
            receipt["reference"],
            receipt["version"],
            digest(plan),
            sha(base / "dispatch.json"),
        ):
            raise SnapshotInvalid("Expanded account/version/plan mismatch")
        if (
            audit["plan_hash"] != frozen["plan_hash"]
            or sha(release / "tabular_resume.ipynb") != frozen["code_hash"]
        ):
            raise SnapshotInvalid("Expanded frozen notebook changed")
        if (
            audit["study_kind"] != frozen["study_kind"]
            or audit["study_kind"] != "figure5_numeric16_defaults"
        ):
            raise SnapshotInvalid("Wrong expanded study kind")
        if (audit["core_verified"], audit["core_slots"], audit["inherited_verified"]) != (
            comparison["core_verified_fits"],
            comparison["core_total_slots"],
            comparison["inherited_verified_fits"],
        ):
            raise SnapshotInvalid("Expanded count mismatch")
        baseline = plan["main_baseline"]
        if sha(ROOT / baseline["audit_path"]) != baseline["audit_sha256"]:
            raise SnapshotInvalid("Main baseline changed")
        source = directory.parent
        download = read_json(base / "download-latest.json")
        if download["reference"] != audit["reference"] or download["version"] != audit["version"]:
            raise SnapshotInvalid("Wrong expanded download receipt")
        if (
            sha(source / "tabular-continuation.json")
            != download["files"]["tabular-continuation.json"]["sha256"]
        ):
            raise SnapshotInvalid("Expanded warning source changed")
        state = read_json(source / "tabular-continuation.json")
        warning_runs = [
            {"id": r["id"], "dataset": r["request"]["dataset"], "model": r["request"]["models"][0]}
            for r in state["records"]
            if r["state"] == "completed" and r["result"]["results"][0]["history_nonfinite_observed"]
        ]
        main_dir = ROOT / "runtime/one-notebook/round-4/output/version-3/verified-offline"
        if (
            sha(main_dir / "dashboard.json")
            != read_json(main_dir / "manifest.json")["files"]["dashboard.json"]
        ):
            raise SnapshotInvalid("Main dataset catalogue changed")
        names = {
            r["id"]: r["name"]
            for r in read_json(
                ROOT
                / "runtime/one-notebook/round-4/output/version-3/verified-offline/dashboard.json"
            )["coverage"]
        }
        info = {
            "datasets": [{"id": x, "name": names[x]} for x in plan["protocol"]["datasets"]],
            "models": plan["protocol"]["core_models"],
            "blocked": plan["blocked"],
            "history_nonfinite_runs": warning_runs,
            "warning_count": len(warning_runs),
            "verified_core": audit["core_verified"],
            "core_slots": audit["core_slots"],
            "new_verified": audit["new_verified"],
            "inherited_verified": audit["inherited_verified"],
        }
        available = False
        if pdf.exists() and (directory / "report.json").exists():
            if read_json(directory / "report.json") != {
                "pdf_sha256": sha(pdf),
                "rotation_audit_sha256": sha(directory / "audit.json"),
                "main_audit_sha256": baseline["audit_sha256"],
            }:
                raise SnapshotInvalid("Expanded PDF does not match its audits")
            available = True
        return {
            "audit": audit,
            "comparison": comparison,
            "protocol": plan["protocol"],
            "info": info,
            "report_available": available,
            "json_path": directory / "comparison.json",
            "pdf_path": pdf,
        }
    except FileNotFoundError:
        raise SnapshotUnavailable("No audited expanded rotation output") from None
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise SnapshotInvalid("Malformed expanded evidence: " + type(error).__name__) from None
