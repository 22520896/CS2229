"""Check appendix charts, R references, and catalog coverage."""

from argparse import ArgumentParser
from base64 import b64decode
from hashlib import sha256
import json
import zipfile

import nbformat
import numpy as np
import pandas as pd

from . import appendix_a4, bo_benchmark, dataset_ablations, dataset_benchmark, time_benchmark
from .analysis import scores
from .appendix_notebooks import FILES
from .author_data import ROOT, SOURCES, TIME_REVISION, sha


EVIDENCE = ROOT / "evidence"
TOLERANCE = 1e-12
CURVE_NUMBERS = (*range(8, 19), *range(20, 25), 33, 34)
CURVE_REFERENCE = EVIDENCE / "appendix_curve_reference.zip"


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def time_parity(number, reanalyze):
    path = ROOT / "figures" / f"figure_{number:02d}.csv"
    actual = time_benchmark.analyze(number) if reanalyze else read(path)
    expected = read(EVIDENCE / f"figure_{number:02d}_r_reference.csv")
    keys = ["panel", "datasets", "model_name", "time_bin"]
    metrics = ["mean", "low", "high", "seconds"]
    actual = actual.set_index(keys, verify_integrity=True).sort_index()[metrics]
    expected = expected.set_index(keys, verify_integrity=True).sort_index()[metrics]
    pd.testing.assert_index_equal(actual.index, expected.index)
    left = actual.to_numpy()
    right = expected.to_numpy()
    np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
    np.testing.assert_allclose(left[:, :3], right[:, :3], rtol=0, atol=TOLERANCE, equal_nan=True)
    np.testing.assert_allclose(left[:, 3], right[:, 3], rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True)
    return {"figure": number, "rows": len(actual), "max_absolute_error": (actual - expected).abs().max().to_dict(),
            "missing_elapsed_cells": int(actual["seconds"].isna().sum())}


def verify_notebooks():
    for key, name in FILES.items():
        notebook = nbformat.read(ROOT / "notebooks" / (name + ".ipynb"), as_version=4)
        nbformat.validate(notebook)
        cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count and cell.outputs for cell in cells)
        outputs = [output for cell in cells for output in cell.outputs]
        assert all(output.output_type != "error" for output in outputs)
        if key.isdecimal():
            images = [sha256(b64decode(out.data["image/png"])).hexdigest()
                      for out in outputs if "image/png" in out.get("data", {})]
            assert images == [sha(ROOT / "figures" / f"figure_{int(key):02d}.png")]


def curve_parity(archive, number, reanalyze, caption=False):
    name = f"figure_{number:02d}" + ("_caption" if caption else "")
    with archive.open(name + ".csv") as stream:
        expected = read(stream)
    if reanalyze:
        if number <= 17:
            actual = dataset_benchmark.analyze(number)
        elif number <= 24:
            actual = dataset_ablations.analyze(number, caption=caption)
        else:
            actual = bo_benchmark.analyze(number)
    else:
        suffix = ".csv.gz" if 12 <= number <= 17 else ".csv"
        actual = read(ROOT / "figures" / (name + suffix))
    keys = [name for name in ("panel", "datasets", "train_cap", "data__keyword",
                             "model_name", "rank", "condition", "shuffle", "method") if name in expected]
    metrics = [name for name in ("mean", "low", "high", "count", "score") if name in expected]
    actual = actual.set_index(keys, verify_integrity=True).sort_index()[metrics]
    expected = expected.set_index(keys, verify_integrity=True).sort_index()[metrics]
    pd.testing.assert_index_equal(actual.index, expected.index, exact=False)
    left, right = actual.to_numpy(float), expected.to_numpy(float)
    np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
    np.testing.assert_allclose(left, right, rtol=0, atol=TOLERANCE, equal_nan=True,
                               err_msg=name)
    return {"variant": name, "rows": len(actual), "max_absolute_error": float(np.nanmax(np.abs(left - right)))}


def verify(reanalyze=False):
    verify_notebooks()
    coefficients = (appendix_a4.analyze(scores("ablations")) if reanalyze
                    else read(EVIDENCE / "a4_coefficients.csv"))
    a4 = appendix_a4.parity(coefficients, read(EVIDENCE / "a4_r_reference.csv"))
    published = read(EVIDENCE / "a4_author_tables.csv")
    assert set(map(tuple, coefficients[["analysis", "term"]].to_numpy())) == set(
        map(tuple, published[["analysis", "term"]].to_numpy()))
    data = read(EVIDENCE / "appendix_datasets.csv")
    assert data.groupby("suite").size().to_dict() == {334: 7, 335: 17, 336: 18, 337: 15}
    assert not data["task"].duplicated().any()
    hpo = read(EVIDENCE / "appendix_hpo.csv")
    assert hpo.groupby("table").size().tolist() == [12, 11, 7, 6, 10, 8, 10, 6]
    with zipfile.ZipFile(CURVE_REFERENCE) as archive:
        curves = [curve_parity(archive, number, reanalyze) for number in CURVE_NUMBERS]
        curves.append(curve_parity(archive, 21, reanalyze, caption=True))
    return {"passed": True, "time_source_commit": TIME_REVISION,
            "time_source_sha256": {name: SOURCES[name].sha256 for name in time_benchmark.SOURCES_BY_FIGURE.values()},
            "time_scores_atol": TOLERANCE, "time_seconds_atol": TOLERANCE, "time_seconds_rtol": TOLERANCE,
            "time_figures": [time_parity(number, reanalyze) for number in (6, 7)],
            "curve_reference_sha256": sha(CURVE_REFERENCE), "curves": curves,
            "a4": a4, "catalog": {"dataset_settings": len(data), "hpo_entries": len(hpo)}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = verify(args.reanalyze)
    if args.save:
        (EVIDENCE / "appendix_parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
