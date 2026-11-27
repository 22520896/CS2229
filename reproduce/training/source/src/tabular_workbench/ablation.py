"""Partial Figure5 recipe; original transform bodies and post-split RNG only."""

import numpy as np

from tabular_workbench.engine import COMMIT, modules

DATASETS = ("361055", "361060")
MODELS = ("rf", "gbt", "xgb", "mlp", "resnet", "ft_transformer")
CORE_MODELS = ("rf", "gbt", "resnet", "ft_transformer")
EXPANSION_FAMILY = "figure5_numeric16_defaults_v1"
SEARCH_FAMILY = "figure5_numeric16_random1_v1"
CPU_SEARCH_FAMILY = "figure5_numeric16_cpu_random2_v1"
PARALLEL_FAMILIES = {
    **{
        "figure5_five_gpu_c%d_v1" % i: (i, "cuda", ("resnet", "ft_transformer"))
        for i in range(4, 9)
    },
    **{"figure5_five_cpu_c%d_v1" % i: (i, "cpu", ("rf", "gbt")) for i in range(16, 20)},
    "figure5_parallel_gpu_c2_v1": (2, "cuda", ("resnet", "ft_transformer")),
    "figure5_gpu_candidate3_v1": (3, "cuda", ("resnet", "ft_transformer")),
    **{
        "figure5_parallel_cpu_c%d_v1" % index: (index, "cpu", ("rf", "gbt"))
        for index in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    },
}
RECIPE_SOURCE = "src/launch_benchmarks/launch_xps.py::xps[random_rotation]"


def recipe(remove_fraction, rotated):
    if type(remove_fraction) is not float or remove_fraction not in (0.0, 0.5):
        raise ValueError("Approved removal fractions are exactly0.0/0.5")
    if type(rotated) is not bool:
        raise ValueError("Explicit rotation boolean required")
    return {
        "transform__0__method_name": "remove_features_rf",
        "transform__0__num_features_to_remove": remove_fraction,
        "transform__0__model_to_use": "rf_c",
        "transform__1__method_name": "gaussienize",
        "transform__1__type": "quantile",
        "transform__2__method_name": "random_rotation",
        "transform__2__deactivated": not rotated,
    }


def validate_request(request):
    specification = request["paper_ablation"]
    assert specification["schema"] == 1 and specification["upstream_commit"] == COMMIT
    assert specification["recipe_source"] == RECIPE_SOURCE
    assert specification["transform_config"] == recipe(
        specification["remove_fraction"], specification["rotated"]
    )
    if "protocol_family" in specification:
        from tabular_workbench.storage import ROOT, read_json

        assert specification["protocol_family"] in (
            EXPANSION_FAMILY,
            SEARCH_FAMILY,
            CPU_SEARCH_FAMILY,
        ) + tuple(PARALLEL_FAMILIES)
        approved = [
            t["id"]
            for t in read_json(ROOT / "docs/TASKS.json")
            if t["suite"] == 337 and t["task"] == "classification"
        ]
        assert len(approved) == 16
        assert request["dataset"] in approved and request["models"][0] in CORE_MODELS
        if specification["protocol_family"] == CPU_SEARCH_FAMILY:
            assert request["models"][0] in ("rf", "gbt") and request["device"] == "cpu"
        if specification["protocol_family"] in PARALLEL_FAMILIES:
            _, device, allowed_models = PARALLEL_FAMILIES[specification["protocol_family"]]
            assert request["device"] == device and request["models"][0] in allowed_models
    else:
        assert request["dataset"] in DATASETS and request["models"][0] in MODELS
    assert len(request["models"]) == 1 and request["seed"] in (0, 1, 2)
    expected_index = (
        2
        if specification.get("protocol_family") == CPU_SEARCH_FAMILY
        else 1
        if specification.get("protocol_family") == SEARCH_FAMILY
        else 0
    )
    if specification.get("protocol_family") in PARALLEL_FAMILIES:
        expected_index = PARALLEL_FAMILIES[specification["protocol_family"]][0]
    assert request["benchmark_config_index"] == expected_index
    if expected_index:
        from tabular_workbench.storage import digest

        assert specification["search_config_sha256"] == digest(request["benchmark_config"])
        assert not any(k.startswith("transform__") for k in request["benchmark_config"])
    assert request["case"] == "paper_rotation_remove_%d_%s" % (
        int(100 * specification["remove_fraction"]),
        "on" if specification["rotated"] else "off",
    )
    assert (
        request["epochs"],
        request["train_limit"],
        request["holdout_limit"],
        request["timeout_seconds"],
    ) == (300, 10000, 50000, 1800)
    return specification


def apply_recipe(original, ys, rng, request, provenance):
    """Run each original pipeline stage once; capture only hashes/dimensions.

    Calling transform_data one stage at a time exposes real stage outputs. Tests
    require bitwise equality and identical RNG state with one full upstream call.
    No selector is fitted on validation/test, no second neural Gaussianization.
    """
    from tabular_workbench.runner import numpy_hash

    specification = validate_request(request)
    pin = specification["input_pin"]
    assert provenance["task"] == "classification"
    for field in ("openml_dataset_id", "dataset_version", "rows", "features", "split_hash"):
        assert provenance[field] == pin[field], "Pinned public input/split changed: " + field
    assert numpy_hash(original) == pin["original_input_hash"], "Pinned split feature bytes changed"
    assert numpy_hash(ys) == pin["labels_hash"], "Pinned split target bytes changed"
    _, pipeline, _, _, _ = modules()
    state = tuple(a.copy() for a in original) + tuple(ys)
    stages = []
    for index in range(3):
        prefix = "transform__%d__" % index
        config = {
            "transform__0__" + key[len(prefix) :]: value
            for key, value in specification["transform_config"].items()
            if key.startswith(prefix)
        }
        before = numpy_hash(state[:3])
        state = pipeline.transform_data(*state, config=config, rng=rng)
        for actual, expected in zip(state[3:], ys):
            np.testing.assert_array_equal(actual, expected)
        if not all(np.isfinite(x).all() for x in state[:3]):
            raise ValueError("Nonfinite original transform output; no replacement")
        stages.append(
            {
                "stage": index,
                "method": config["transform__0__method_name"],
                "input_hash": before,
                "output_hash": numpy_hash(state[:3]),
                "shapes": [list(x.shape) for x in state[:3]],
                "source": "upstream/src/generate_dataset_pipeline.py::transform_data",
            }
        )
    return list(state[:3]), {
        "recipe_source": RECIPE_SOURCE,
        "transform_config": specification["transform_config"],
        "remove_fraction": specification["remove_fraction"],
        "rotated": specification["rotated"],
        "stages": stages,
        "selector_fit_split": "train only",
        "neural_extra_gaussianization": False,
        "expected_float32_model_input_hash": numpy_hash([x.astype(np.float32) for x in state[:3]]),
        "rng": "original post-split RandomState, shared continuously across all3transforms",
        "scope": "Figure5 recipe across16numeric classification datasets/3fixedseeds; configuration/HPO scope is frozen explicitly in its plan,not full-paper normalization"
        if "protocol_family" in specification
        else "Partial Figure5 recipe,2datasets/defaults/3fixedseeds; not full-paper normalized random-search figure",
    }
