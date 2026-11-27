"""Verified existing-notebook address aliases; never infer a new remote target."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "reports/dispatch/author-v2-hpo8-20261030/reference-aliases.json"


def resolve_reference(reference, root=ROOT):
    root = Path(root).resolve()
    registry = root / REGISTRY
    if not registry.exists():
        return reference
    rows = json.loads(registry.read_text(encoding="utf-8"))["aliases"]
    if reference not in rows:
        return reference
    entry = rows[reference]
    proof_path = (root / entry["proof"]).resolve()
    proof_path.relative_to((root / "reports/dispatch").resolve())
    assert hashlib.sha256(proof_path.read_bytes()).hexdigest() == entry["sha256"]
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    assert proof["passed"] and proof["private"] and proof["same_version_history"]
    assert proof["current_source_and_exact_predecessor_output_match"]
    assert not proof["new_notebook_created"] and not proof["duplicate_save"]
    assert proof["requested_reference"] == reference
    actual = proof["actual_reference"]
    assert actual.split("/")[0] == reference.split("/")[0]
    assert actual == entry["actual_reference"]
    return actual
