"""Check local configuration only. No network, upload, token output or job execution."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT_SUFFIXES = ("",) + tuple("_v%d" % index for index in range(2, 8))
CREDENTIAL_KEYS = {
    prefix + suffix
    for suffix in ACCOUNT_SUFFIXES
    for prefix in ("KAGGLE_API_TOKEN", "KAGGLE_NOTEBOOK_ID")
}
ALLOWED = CREDENTIAL_KEYS | {
    "KAGGLE_ACCELERATOR",
    "KAGGLE_MAX_RUNTIME_SECONDS",
    "KAGGLE_GPU_HOURS_AVAILABLE",
    "KAGGLE_ENABLE_INTERNET",
    "KAGGLE_PRIVATE",
    "KAGGLE_AUTO_RUN",
    "KAGGLE_ALLOW_REPLACE_NOTEBOOK",
}


def read_values(path):
    values, errors = {}, []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errors.append("Line %d: expected NAME=value" % line_number)
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in ALLOWED or key in values:
            errors.append("Line %d: unknown or duplicate variable name" % line_number)
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values, errors


def validate(path):
    values, errors = read_values(path)
    missing = [key for key in ("KAGGLE_API_TOKEN", "KAGGLE_NOTEBOOK_ID") if not values.get(key)]
    notebook = values.get("KAGGLE_NOTEBOOK_ID", "")
    if notebook and not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", notebook):
        errors.append("KAGGLE_NOTEBOOK_ID must be owner/notebook-slug")
    for suffix in ACCOUNT_SUFFIXES[1:]:
        token_key, reference_key = "KAGGLE_API_TOKEN" + suffix, "KAGGLE_NOTEBOOK_ID" + suffix
        pair = bool(values.get(token_key)), bool(values.get(reference_key))
        if any(pair) and not all(pair):
            errors.append("Account%s requires both token and notebook variables" % suffix)
        if values.get(reference_key) and not re.fullmatch(
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", values[reference_key]
        ):
            errors.append(reference_key + " must be owner/notebook-slug")
    for key in (
        "KAGGLE_ENABLE_INTERNET",
        "KAGGLE_PRIVATE",
        "KAGGLE_AUTO_RUN",
        "KAGGLE_ALLOW_REPLACE_NOTEBOOK",
    ):
        if values.get(key) not in ("true", "false"):
            errors.append(key + " must be true or false")
    if values.get("KAGGLE_PRIVATE") != "true":
        errors.append("KAGGLE_PRIVATE must remain true for this project")
    try:
        timeout = int(values.get("KAGGLE_MAX_RUNTIME_SECONDS", ""))
        if not 60 <= timeout <= 43200:
            raise ValueError()
    except ValueError:
        errors.append("KAGGLE_MAX_RUNTIME_SECONDS must be an integer from60 to43200")
    quota = values.get("KAGGLE_GPU_HOURS_AVAILABLE", "")
    if quota:
        try:
            hours = float(quota)
            if not 0 <= hours <= 10000:
                raise ValueError()
        except ValueError:
            errors.append("KAGGLE_GPU_HOURS_AVAILABLE must be a nonnegative finite number")
    if not values.get("KAGGLE_ACCELERATOR"):
        errors.append("KAGGLE_ACCELERATOR is required")
    return {
        "file_exists": True,
        "configured": not missing and not errors,
        "missing_variables": missing,
        "errors": errors,
        "secret_values": "never printed",
        "network_called": False,
        "external_write_or_job_started": False,
    }


def configured_token_bytes(values):
    """All configured secrets for local allowlist-bundle/archive exclusion checks."""
    return [
        values["KAGGLE_API_TOKEN" + suffix].encode()
        for suffix in ACCOUNT_SUFFIXES
        if values.get("KAGGLE_API_TOKEN" + suffix)
    ]


def select_account(values, owner=None):
    """Select credentials by recorded owner, not by version or current default.

    Never returns another account's token; authenticate in a fresh process.
    The env file is not edited or echoed, and owner mismatch is fail-closed.
    """
    pairs = []
    for suffix in ACCOUNT_SUFFIXES:
        token, reference = (
            values.get("KAGGLE_API_TOKEN" + suffix),
            values.get("KAGGLE_NOTEBOOK_ID" + suffix),
        )
        if token or reference:
            if (
                not token
                or not reference
                or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", reference)
            ):
                raise ValueError("Incomplete or invalid configured account pair")
            pairs.append((token, reference))
    if owner is None:
        matches = pairs[:1]
    else:
        matches = [
            (token, reference)
            for token, reference in pairs
            if reference.split("/")[0].lower() == owner.lower()
        ]
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous credentials for recorded account owner")
    result = {k: v for k, v in values.items() if k not in CREDENTIAL_KEYS}
    result.update(KAGGLE_API_TOKEN=matches[0][0], KAGGLE_NOTEBOOK_ID=matches[0][1])
    return result


def error_summary(error):
    response = getattr(error, "response", None)
    return {
        "error_type": type(error).__name__,
        "http_status": getattr(response, "status_code", None),
        "raw_error_suppressed": True,
    }


if __name__ == "__main__":
    try:
        result = validate(ROOT / ".env.kaggle")
    except FileNotFoundError:
        result = {"file_exists": False, "configured": False}
    except Exception:
        # Never echo exceptions derived from a potentially secret-valued line.
        result = {"configured": False, "errors": ["Local configuration could not be read"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["configured"] else 2)
