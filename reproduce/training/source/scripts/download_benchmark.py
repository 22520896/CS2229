"""Manual download of exact authorized versions; never pushes/restarts anything.

Only predetermined summary/evidence names are downloaded. Credentials and signed
URLs remain in memory. Changed local evidence is never overwritten.
"""

import contextlib
import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import requests
from check_kaggle_config import error_summary, read_values
from kaggle_reference_aliases import resolve_reference

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 200 * 1024 * 1024


def replace_with_retry(source, target):
    """Bounded retry for transient Windows sharing violations, not remote jobs."""
    for attempt in range(6):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2**attempt)


def fetch_file(api, owner, slug, version, name, target):
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    if target.exists():
        return target.read_bytes()
    owner, slug = resolve_reference(owner + "/" + slug).split("/")
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = owner, slug
    request.version_number, request.file_path = version, name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    original_send = requests.sessions.Session.send

    def bounded_send(session, prepared, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (20, 45)
        return original_send(session, prepared, **kwargs)

    with api.build_kaggle_client() as client, patch.object(
        requests.sessions.Session, "send", bounded_send
    ):
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        # The SDK follows redirects and returns a streaming Response. Its
        # official bounded retry/resume helper checks ETag/Last-Modified and
        # length before appending Range bytes. Never promote partial evidence.
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        if length and int(length) > MAX_BYTES:
            response.close()
            raise ValueError("Numeric evidence exceeds the download size cap")
        # Large CDN bodies can stop mid-stream. Fetch bounded byte ranges when
        # the object exposes both length and an identity validator. Range and
        # identity checks are transport only; numeric audit remains mandatory.
        validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
        if length and validator:
            url = response.url
            response.close()
            ranged = target.with_suffix(target.suffix + ".ranges")
            marker = ranged.with_suffix(ranged.suffix + ".json")
            identity = dict(
                owner=owner,
                slug=slug,
                version=version,
                name=name,
                size=int(length),
                validator=validator,
            )
            payload = range_download(url, identity, ranged, marker)
            temporary = ranged
        else:
            api.download_file(
                response, str(temporary), client, quiet=True, resume=True, max_retries=2, timeout=45
            )
            payload = temporary.read_bytes()
    if len(payload) > MAX_BYTES:
        raise ValueError("Numeric evidence exceeds the download size cap")
    replace_with_retry(temporary, target)
    return payload


def range_download(url, identity, target, marker, getter=None, chunk_size=4 * 1024**2):
    """Exact Range+If-Range transport with hashed checkpoints, no URL persisted."""
    getter = requests.get if getter is None else getter
    total = identity["size"]
    assert 0 < total <= MAX_BYTES
    previous = b""
    pending = marker.with_suffix(marker.suffix + ".tmp")
    if target.exists() or marker.exists():
        assert target.exists(), "Preserve incomplete range checkpoint"
        previous = target.read_bytes()
        pin = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
        valid = pin is not None and pin["identity"] == identity and pin["bytes"] == len(previous)
        valid = valid and pin["sha256"] == hashlib.sha256(previous).hexdigest()
        if not valid and pending.exists():
            # A sharing violation can occur after append+hashed journal write
            # but before the journal rename. Recover ONLY that fully verified
            # commit, including its prior committed prefix. Never truncate or
            # accept an unjournaled/tampered suffix.
            candidate = json.loads(pending.read_text(encoding="utf-8"))
            assert candidate["identity"] == identity and candidate["bytes"] == len(previous)
            assert candidate["sha256"] == hashlib.sha256(previous).hexdigest(), (
                "Range checkpoint changed"
            )
            if pin is not None:
                assert pin["identity"] == identity and 0 <= pin["bytes"] <= len(previous)
                assert pin["sha256"] == hashlib.sha256(previous[: pin["bytes"]]).hexdigest(), (
                    "Range checkpoint changed"
                )
            replace_with_retry(pending, marker)
            pin = candidate
        assert pin is not None, "Preserve incomplete range checkpoint"
        assert pin["identity"] == identity and pin["bytes"] == len(previous)
        assert pin["sha256"] == hashlib.sha256(previous).hexdigest(), "Range checkpoint changed"
    assert len(previous) <= total
    offset = len(previous)
    while offset < total:
        end = min(total - 1, offset + chunk_size - 1)
        for attempt in range(3):
            try:
                response = getter(
                    url,
                    headers={
                        "Range": "bytes=%d-%d" % (offset, end),
                        "If-Range": identity["validator"],
                        "Accept-Encoding": "identity",
                    },
                    stream=True,
                    timeout=(20, 45),
                )
                try:
                    response.raise_for_status()
                    assert response.status_code == 206, "Server did not honor exact range"
                    assert response.headers.get("Content-Range") == "bytes %d-%d/%d" % (
                        offset,
                        end,
                        total,
                    )
                    validator = response.headers.get("ETag") or response.headers.get(
                        "Last-Modified"
                    )
                    assert not validator or validator == identity["validator"], (
                        "Remote object changed"
                    )
                    parts, size = [], 0
                    for part in response.iter_content(64 * 1024):
                        size += len(part)
                        assert size <= end - offset + 1
                        parts.append(part)
                    raw = b"".join(parts)
                    assert len(raw) == end - offset + 1, "Incomplete range"
                finally:
                    response.close()
                break
            except (requests.RequestException, OSError):
                if attempt == 2:
                    raise
        previous += raw
        with target.open("ab" if offset else "xb") as f:
            f.write(raw)
        value = dict(
            identity=identity, bytes=len(previous), sha256=hashlib.sha256(previous).hexdigest()
        )
        pending.write_text(json.dumps(value), encoding="utf-8")
        replace_with_retry(pending, marker)
        offset = len(previous)
    return previous


def main():
    values, errors = read_values(ROOT / ".env.kaggle")
    if errors:
        raise ValueError("Invalid Kaggle configuration (values suppressed)")
    for key in list(os.environ):
        if key.startswith("KAGGLE_") or key in ("VERBOSE", "VERBOSE_OUTPUT", "DEBUG"):
            os.environ.pop(key, None)
    os.environ["KAGGLE_API_TOKEN"] = values["KAGGLE_API_TOKEN"]
    os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT / "runtime/kaggle-auth")
    logging.disable(logging.CRITICAL)
    submission = json.loads((ROOT / "runtime/benchmark-submission.json").read_text())
    checked = []
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle

        api = kaggle.api
        owner, slug = values["KAGGLE_NOTEBOOK_ID"].split("/")
        assert owner.lower() == api.get_config_value("username").lower()
        for entry in submission["versions"]:
            item = {key: entry[key] for key in ("version", "device", "shard")}
            try:
                stem = "tabular-%s-%d" % (entry["device"], entry["shard"])
                directory = ROOT / "runtime/benchmark-output" / ("version-%d" % entry["version"])
                raw = fetch_file(
                    api, owner, slug, entry["version"], stem + ".json", directory / (stem + ".json")
                )
                summary = json.loads(raw)
                assert summary["device"] == entry["device"] and summary["shard"] == entry["shard"]
                evidence = fetch_file(
                    api,
                    owner,
                    slug,
                    entry["version"],
                    stem + "-evidence.zip",
                    directory / (stem + "-evidence.zip"),
                )
                item.update(
                    summary_sha256=hashlib.sha256(raw).hexdigest(),
                    evidence_sha256=hashlib.sha256(evidence).hexdigest(),
                    evidence_bytes=len(evidence),
                    state=summary["state"],
                    downloaded=True,
                )
            except Exception as error:
                item.update(downloaded=False, **error_summary(error))
            checked.append(item)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "remote_writes": False,
        "jobs_submitted": 0,
        "versions": checked,
    }
    (ROOT / "runtime/benchmark-download.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record))
    return record


if __name__ == "__main__":
    main()
