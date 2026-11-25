"""Audit the remaining appendix figures and independent tuned-estimator references."""

from argparse import ArgumentParser
from base64 import b64decode
from hashlib import sha256
import json
from zipfile import ZipFile

import nbformat
import numpy as np
import pandas as pd

from .author_data import ROOT, sha
from .published_notebooks import FILES, PDF_DEFAULTS, load_figure
from .tuned_estimators import analyze, compare_published_defaults


EVIDENCE = ROOT / "evidence"
TUNED_REFERENCE = EVIDENCE / "tuned_estimators_r_reference.zip"
THRESHOLD = 1e-12


def check_notebooks():
    for number, name in FILES.items():
        notebook = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        nbformat.validate(notebook)
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert all(c.execution_count and c.outputs for c in code_cells)
        outputs = [output for c in code_cells for output in c.outputs]
        assert all(o.output_type != "error" for o in outputs)
        hashes = [sha256(b64decode(o.data["image/png"])).hexdigest()
                  for o in outputs if "image/png" in o.get("data", {})]
        assert hashes == [sha(ROOT / "figures" / f"figure_{number:02d}.png")]


def verify_tuned(number, record, reanalyze):
    name = f"figure_{number:02d}"
    if reanalyze:
        table = analyze(number)
    else:
        table = pd.read_csv(ROOT / "figures" / f"{name}.csv", float_precision="round_trip")
    keys = ["panel", "datasets", "model_name", "rank"]
    data = ["mean", "high", "low", "count"]
    with ZipFile(TUNED_REFERENCE) as archive, archive.open(f"{name}.csv") as stream:
        r = pd.read_csv(stream, float_precision="round_trip")
    left = table.set_index(keys, verify_integrity=True).sort_index()[data]
    right = r.set_index(keys, verify_integrity=True).sort_index()[data]
    pd.testing.assert_index_equal(left.index, right.index)
    np.testing.assert_allclose(left.to_numpy(), right.to_numpy(), atol=THRESHOLD, rtol=0, equal_nan=True)
    agreement = compare_published_defaults(table, record)
    assert len(agreement) == 12 and agreement["absolute_difference"].max() < 1e-5
    return {"figure": number, "rows": len(table),
            "max_r_error": float((left - right).abs().max().max()),
            "default_lines_checked": len(agreement),
            "max_pdf_default_error": float(agreement["absolute_difference"].max())}


def verify(reanalyze=False):
    check_notebooks()
    return {
        "passed": True, "pdf_defaults_sha256": sha(PDF_DEFAULTS), "reanalyzed_tuned": reanalyze,
        "tuned_r_reference_sha256": sha(TUNED_REFERENCE), "tuned_reference_tolerance": THRESHOLD,
        "tuned_figures": [verify_tuned(n, load_figure(n), reanalyze) for n in FILES],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report = verify(reanalyze=args.reanalyze)
    if args.save:
        (EVIDENCE / "published_parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Figures 39–40 matched the independent R reference")
