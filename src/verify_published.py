"""Audit the remaining appendix figures and independent tuned-estimator references."""

from argparse import ArgumentParser
from base64 import b64decode
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import nbformat
import numpy as np
import pandas as pd

from .author_data import ROOT, sha
from .published_figures import ARCHIVE, CURVE_FIGURES, DIGEST, FLOW_FIGURES, PAGES, audit_flow, extract, load
from .published_notebooks import FILES
from .tuned_estimators import analyze, compare_published_defaults


EVIDENCE = ROOT / "evidence"
TUNED_REFERENCE = EVIDENCE / "tuned_estimators_r_reference.zip"
DIGITIZED = (19, *FLOW_FIGURES, *range(35, 39))
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


def check_shapes(record):
    max_error = 0.0
    vertices = 0
    series_count = 0
    for panel in record["panels"].values():
        assert panel["axes"]["max_tick_error_pt"] < .012
        max_error = max(max_error, panel["axes"]["max_tick_error_pt"])
        for paths in panel["series"].values():
            series_count += 1
            assert paths["mean"] and paths["band"]
            for trace in paths["mean"]:
                values = np.array(trace)
                assert np.isfinite(values).all()
                assert np.all(values[:, 0] > 0) and np.all(np.diff(values[:, 0]) >= 0)
                vertices += len(values)
    return {"series": series_count, "mean_vertices": vertices, "max_tick_error_pt": max_error}


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


def verify(pdf=None, reanalyze=False):
    check_notebooks()
    saved = load()
    assert saved["sha256"] == DIGEST and {int(n) for n in saved["figures"]} == set(PAGES)
    if pdf is not None:
        current = extract(pdf)
        # JSON round-trip canonicalizes integer figure keys and NumPy scalar values.
        assert json.loads(json.dumps(current)) == saved
    figures = saved["figures"]
    flow = {n: audit_flow(figures[str(n)]) for n in FLOW_FIGURES}
    assert [n for n, result in flow.items() if result["internal_residuals_pt"]] == [28]
    assert set(flow[28]["internal_residuals_pt"]) == {"Balancing"}
    curves = {n: check_shapes(figures[str(n)]) for n in CURVE_FIGURES}
    decision = figures["19"]
    assert len(decision["panels"]) == 2
    for panel in decision["panels"].values():
        cells = np.array(panel["color_cells"])
        assert list(cells.shape[::-1]) == decision["grid_size"]
        assert np.array_equal(np.unique(cells), [-1, 0, 1])
    return {
        "passed": True, "source_pdf_sha256": DIGEST, "geometry_sha256": sha(ARCHIVE),
        "pdf_extraction_rechecked": pdf is not None, "reanalyzed_tuned": reanalyze,
        "digitized_figures": list(DIGITIZED), "digitized_flow_audit": flow, "vector_curves_audit": curves,
        "tuned_r_reference_sha256": sha(TUNED_REFERENCE), "tuned_reference_tolerance": THRESHOLD,
        "tuned_figures": [verify_tuned(n, figures[str(n)], reanalyze) for n in (39, 40)],
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    report = verify(pdf=args.pdf, reanalyze=args.reanalyze)
    if args.save:
        (EVIDENCE / "published_parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("13 digitized figures, 2 recomputed estimator comparisons")
