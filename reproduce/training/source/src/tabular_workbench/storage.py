"""Atomic local artifacts and validated identifiers, no database or cloud."""

import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(100):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.01)


def read_json(path):
    for attempt in range(100):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.01)


def folder(identifier):
    if not re.fullmatch(r"[a-f0-9]{16}", identifier):
        raise ValueError("Invalid run identifier")
    return ROOT / "artifacts/runs" / identifier


def report(identifier):
    path = folder(identifier)
    return read_json(path / ("result.json" if (path / "result.json").exists() else "status.json"))
