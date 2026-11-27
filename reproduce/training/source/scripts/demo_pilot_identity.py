"""Scientific identities for fixed-configuration mechanism demonstrations."""
import hashlib
import json


def scientific_key(task, model, seed, configuration):
    cfg = {k: v for k, v in configuration.items() if k not in {"model__device", "use_gpu"}}
    value = dict(task=str(task), model=model, seed=seed, configuration=cfg)
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require_disjoint(shards, reserved):
    seen = set(reserved)
    for name, jobs in shards.items():
        for job in jobs:
            key = scientific_key(job["task"], job["model"], job["seed"], job["configuration"])
            if key in seen:
                raise ValueError("Scientific fit is already owned: " + name)
            seen.add(key)
    return seen
