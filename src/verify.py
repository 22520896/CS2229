"""Compare notebook outputs against the frozen, independently run R analysis."""

import argparse
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd

from .analysis import MODEL, PANELS, REMOVE, ROTATE, SMOOTH, analyze
from .author_data import ROOT, sha


ARCHIVE = ROOT / "evidence/r_reference.zip"
TOLERANCE = 1e-12
CONFIG = {
    1: [("figure_01", ["panel", "rank", MODEL], ["mean", "low", "high"])],
    2: [("figure_02", [MODEL, "shuffle", SMOOTH], ["score"])],
    3: [("figure_03", [REMOVE, "method"], ["mean", "low", "high", "count"])],
    4: [(name, ["fraction", MODEL], ["mean", "low", "high", "count"])
        for name in ("figure_04_4a", "figure_04_4b_script", "figure_04_4b_caption")],
    5: [("figure_05", [MODEL, "shuffle", ROTATE, "removed"], ["score"])],
}


def compare(number, reanalyze=False):
    actual = analyze(number) if reanalyze else None
    if actual is not None and not isinstance(actual, tuple):
        actual = (actual,)
    measures = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for index, (filename, keys, metrics) in enumerate(CONFIG[number]):
            with archive.open(filename + ".csv") as stream:
                reference = pd.read_csv(stream, float_precision="round_trip")
            frame = actual[index] if reanalyze else pd.read_csv(
                ROOT / "figures" / (filename + ".csv"), float_precision="round_trip"
            )
            join = frame.merge(reference, on=keys, how="outer", indicator=True,
                               validate="one_to_one", suffixes=("_python", "_r"))
            if not join["_merge"].eq("both").all():
                raise ValueError(f"Missing or extra rows in {filename}")
            left = join[[key + "_python" for key in metrics]].to_numpy(float)
            right = join[[key + "_r" for key in metrics]].to_numpy(float)
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_allclose(left, right, atol=TOLERANCE, rtol=0, equal_nan=True)
            valid = ~np.isnan(left)
            measures.append(dict(panel=filename, rows=len(join), metric_cells=int(valid.sum()),
                                 missing_cells=int((~valid).sum()),
                                 max_absolute_error=float(np.abs(left[valid] - right[valid]).max())))
    return dict(figure=number, rows=sum(p["rows"] for p in measures),
                max_absolute_error=max(p["max_absolute_error"] for p in measures), panels=measures)


def verify_notebooks():
    import nbformat
    import base64

    from .notebooks import FILES

    for number, filename in enumerate(FILES, 1):
        notebook = nbformat.read(ROOT / "notebooks" / (filename + ".ipynb"), as_version=4)
        nbformat.validate(notebook)
        outputs = [output for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs]
        assert all(output.output_type != "error" for output in outputs)
        pngs = [hashlib.sha256(base64.b64decode(output.data["image/png"])).hexdigest()
                for output in outputs if "image/png" in output.get("data", {})]
        charts = [ROOT / "figures" / f"figure_{number:02d}.png"]
        if number in (1, 2, 5):
            charts.append(ROOT / "figures" / f"figure_{number:02d}_v2.png")
        assert pngs == [sha(chart) for chart in charts]


def verify(reanalyze=False):
    verify_notebooks()
    return dict(passed=True, rng="R MT19937, seed 0", tolerance=TOLERANCE,
                reference_sha256=sha(ARCHIVE), figures=[compare(n, reanalyze) for n in range(1, 6)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reanalyze", action="store_true",
                        help="Recompute all figures from the verified source data")
    parser.add_argument("--save", action="store_true", help="Write the parity receipt")
    args = parser.parse_args()
    results = verify(args.reanalyze)
    if args.save:
        path = ROOT / "evidence/python_r_parity.json"
        path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Figs 1–5: {sum(item['rows'] for item in results['figures']):,} rows matched at {TOLERANCE:g}")
