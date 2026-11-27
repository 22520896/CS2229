"""Explicit repeat adapter around the existing, unchanged pilot fit function.

Only the seven seed call sites and the recorded seed are parameterized. Model,
data transformations, training and evaluation statements are not replaced.
"""
import ast
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import warnings


def seeded_fit(module, seed):
    source = inspect.getsource(module.fit)
    tree = ast.parse(source)
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef) and fn.name == "fit"
    changed = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func) if hasattr(ast, "unparse") else None
            # Compatible with the pinned Python 3.8 runtime.
            if isinstance(node.func, ast.Attribute):
                parts, current = [], node.func
                while isinstance(current, ast.Attribute):
                    parts.insert(0, current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    name = ".".join([current.id] + parts)
            indexes = {"random.seed": 0, "np.random.seed": 0, "torch.manual_seed": 0,
                       "torch.cuda.manual_seed_all": 0, "training.train_model": 0,
                       "paper.generate": 2}
            if name in indexes:
                index = indexes[name]
                assert isinstance(node.args[index], ast.Constant) and node.args[index].value == 0
                node.args[index] = ast.Constant(seed)
                changed.append(name)
        elif isinstance(node, ast.keyword) and node.arg == "seed":
            assert isinstance(node.value, ast.Constant) and node.value.value == 0
            node.value = ast.Constant(seed)
            changed.append("result.seed")
    assert sorted(changed) == sorted(["random.seed", "random.seed", "np.random.seed", "np.random.seed",
                                     "torch.manual_seed", "torch.cuda.manual_seed_all",
                                     "training.train_model", "paper.generate", "result.seed"])
    ast.fix_missing_locations(tree)
    namespace = dict(module.__dict__)
    exec(compile(tree, inspect.getfile(module), "exec"), namespace)
    return namespace["fit"]


def main(plan_path, index):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_bytes())
    job = plan["jobs"][int(index)]
    run = plan_path.parent
    source = Path(os.environ["TABULAR_PILOT_ROOT"])
    sys.path[:0] = [str(run), str(source / "tools/benchmarks"), str(source / "scripts")]
    import representative_pilot as pilot
    from demo_pilot_identity import scientific_key
    assert scientific_key(job["task"], job["model"], job["seed"], job["configuration"]) == job["key"]
    assert job["seed"] in job["required_seeds"] and plan["fit_timeout_seconds"] == 1800
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = seeded_fit(pilot, job["seed"])(
            job["key"], job["task"], job["model"], updates=job["configuration"],
            recipe=job["recipe"], device=job["device"], boundary=False)
    assert scientific_key(result["task"], result["model"], result["seed"], result["config"]) == job["key"]
    record = dict(key=job["key"], job=job, result=result,
                  warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured])
    with (run / "runs" / job["key"] / "demo-result.json").open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
