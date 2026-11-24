"""Digitize visible figure geometry from the pinned NeurIPS supplement."""

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
import gzip
import io
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .appendix_notebooks import SUPPLEMENT
from .author_data import CACHE, ROOT, sha


DIGEST = "6940e2127afe2f9dc60a218a99d3fbf1f15e2e5b4c572e3edaba1765f0035b16"
PAGES = {19: 29, 25: 34, 26: 35, 27: 35, 28: 35, 29: 36, 30: 36, 31: 36,
         32: 37, 35: 44, 36: 44, 37: 45, 38: 45, 39: 47, 40: 47}
FLOW_FIGURES = tuple(range(25, 33))
CURVE_FIGURES = tuple(range(35, 41))
ARCHIVE = ROOT / "evidence" / "published_figure_geometry.json.gz"
PLOT_BOUNDS = {"Random Forest": (126, (0.0873, 0.0594, 0.957, 0.943)),
               "MLP": (127, (0.105, 0.057, 0.974, 0.942))}
COLOR_THRESHOLD = 12
GRID = (260, 240)
VECTOR_TOLERANCE = 0.002


def supplement(path=None):
    if path is None:
        CACHE.mkdir(exist_ok=True)
        path = CACHE / "neurips-2022-supplement.pdf"
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as tmp:
                pending = Path(tmp.name)
                try:
                    request = Request(SUPPLEMENT, headers={"User-Agent": "CS2229-benchmark-reproduction"})
                    with urlopen(request, timeout=60) as response:
                        shutil.copyfileobj(response, tmp)
                    tmp.close()
                    if sha(pending) != DIGEST:
                        raise ValueError("Published PDF differs from the pinned copy")
                    pending.replace(path)
                finally:
                    pending.unlink(missing_ok=True)
    if sha(Path(path)) != DIGEST:
        raise ValueError("Unexpected supplement SHA-256")
    return Path(path)


def page_spans(page):
    return [span for block in page.get_text("dict")["blocks"] if "lines" in block
            for line in block["lines"] for span in line["spans"]]


def caption_band(page, number):
    lines = [("".join(span["text"] for span in line["spans"]), line["bbox"][1])
             for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])]
    captions = sorted((top, line) for line, top in lines if line.startswith("Figure "))
    match = next(i for i, (_, title) in enumerate(captions) if title.startswith(f"Figure {number}:"))
    top = captions[match - 1][0] + 10 if match else 0.0
    return top, captions[match][0]


def inside(rect, band):
    return band[0] < rect.y0 and rect.y1 < band[1]


def points_of(item):
    return [[float(p.x), float(p.y)] for p in item[1:]]


def flow_nodes(shapes, names):
    nodes = []
    for node in shapes:
        rect = node["rect"]
        if node.get("fill") not in ((0., 0., 1.), (1., 0., 0.)) or len(node["items"]) != 1:
            continue
        mid = (rect.y0 + rect.y1) / 2
        closest = min(names, key=lambda label: abs(label["origin"][1] - mid - 2.625)
                      + min(abs(label["bbox"][0] - rect.x1 - .975),
                            abs(label["bbox"][2] - rect.x0 + .975)))
        x_error = min(abs(closest["bbox"][0] - rect.x1 - .975),
                      abs(closest["bbox"][2] - rect.x0 + .975))
        if abs(closest["origin"][1] - mid - 2.625) > .15 or x_error > .15:
            raise ValueError(f"Unresolved Sankey node near {tuple(rect)}")
        nodes.append({"name": closest["text"], "box": list(rect), "kept": node["fill"] == (0., 0., 1.)})
    if len({node["name"] for node in nodes}) != len(nodes):
        raise ValueError("Duplicate Sankey node label")
    return nodes


def flow_node_at(nodes, x, lo, hi, source):
    expected_edge = 2 if source else 0
    fits = [n["name"] for n in nodes if abs(n["box"][expected_edge] - x) < VECTOR_TOLERANCE
            and n["box"][1] - VECTOR_TOLERANCE <= lo <= hi <= n["box"][3] + VECTOR_TOLERANCE]
    if len(fits) != 1:
        raise ValueError(f"Expected one node at link boundary ({x:g}, {lo:g}, {hi:g})")
    return fits[0]


def audit_flow(record):
    nodes, links = record["nodes"], record["links"]
    incoming, outgoing = defaultdict(float), defaultdict(float)
    for link in links:
        if link["width_pt"] <= 0:
            raise ValueError("Nonpositive link width")
        incoming[link["target"]] += link["width_pt"]
        outgoing[link["source"]] += link["width_pt"]
    node_errors = {}
    internal = {}
    for node in nodes:
        name = node["name"]
        h = node["box"][3] - node["box"][1]
        sides = (incoming[name], outgoing[name])
        if not any(sides):
            raise ValueError(f"Isolated displayed node: {name}")
        # Plotly uses a minimum visible node height for very thin edges.
        if h > .151:
            node_errors[name] = abs(max(sides) - h)
        if all(sides):
            internal[name] = sides[0] - sides[1]
    total = outgoing["Removing NA"]
    difference = total - incoming["Used"] - incoming["Not used"]
    if not total or max(node_errors.values()) > VECTOR_TOLERANCE:
        raise ValueError("A Sankey link does not fit its visible node")
    if abs(sum(internal.values()) - difference) > VECTOR_TOLERANCE:
        raise ValueError("Internal and terminal width residuals differ")
    return {"max_node_width_error_pt": max(node_errors.values()), "start_width_pt": total,
            "terminal_width_residual_pt": difference,
            "internal_residuals_pt": {n: v for n, v in internal.items() if abs(v) > VECTOR_TOLERANCE},
            "used_share": incoming["Used"] / total, "not_used_share": incoming["Not used"] / total,
            "nodes": len(nodes), "links": len(links)}


def extract_flow(number, page):
    spans = page_spans(page)
    band = caption_band(page, number)
    shapes = [shape for shape in page.get_drawings() if inside(shape["rect"], band)]
    names = [s for s in spans if s["color"] == 0x2a3f5f and band[0] < s["origin"][1] < band[1]]
    nodes = flow_nodes(shapes, names)
    links = []
    for shape in shapes:
        if shape["type"] not in ("f", "fs") or [item[0] for item in shape["items"]] != ["c", "l", "c", "l"]:
            continue
        if shape.get("fill") != (0., 0., 0.) or abs(shape["fill_opacity"] - .2) > .01:
            continue
        upper, _, lower, _ = [points_of(item) for item in shape["items"]]
        sx, sy = upper[0]
        tx, ty = upper[-1]
        top, bottom = lower[-1][1], lower[0][1]
        dw_source, dw_target = top - sy, bottom - ty
        if abs(dw_source - dw_target) > VECTOR_TOLERANCE:
            raise ValueError("Rendered link changes width between endpoints")
        if dw_source <= 1e-5:
            continue
        name = flow_node_at(nodes, sx, sy, top, source=True)
        dest = flow_node_at(nodes, tx, ty, bottom, source=False)
        links.append({"source": name, "target": dest, "width_pt": (dw_source + dw_target) / 2,
                      "upper": upper, "lower": lower})
    result = {"page": PAGES[number], "method": "pdf_vector_flow", "nodes": nodes, "links": links}
    result["audit"] = audit_flow(result)
    return result


def color_key(rgb):
    return sum(round(channel * 255) << shift for channel, shift in zip(rgb, (16, 8, 0)))


def shape_points(shape):
    if not all(item[0] == "l" for item in shape["items"]):
        raise ValueError("Expected a linear chart path")
    return [points_of(shape["items"][0])[0], *[points_of(item)[-1] for item in shape["items"]]]


def nearest(coordinate, marks):
    location = min(marks, key=lambda tick: abs(tick - coordinate))
    if abs(location - coordinate) > 3.2:
        raise ValueError("Axis label too far from its gridline")
    return location


@dataclass(frozen=True)
class Axes:
    x_linear: tuple
    y_linear: tuple
    clip: tuple
    max_tick_error_pt: float

    def convert(self, vector):
        col = np.asarray(vector, dtype=float)
        x = np.power(10., (col[:, 0] - self.x_linear[1]) / self.x_linear[0])
        y = (col[:, 1] - self.y_linear[1]) / self.y_linear[0]
        return np.column_stack((x, y)).tolist()


def panel_axes(shapes, spans):
    grid = [a["rect"] for a in shapes if a.get("color") and a["type"] == "s"
            and .90 < a["color"][0] < .94 and len(set(a["color"])) == 1 and len(a["items"]) == 1]
    vertical = [a for a in grid if a.width < 1e-4 and a.height > 60]
    horizontal = [a for a in grid if a.height < 1e-4 and a.width > 100]
    clip = (horizontal[0].x0, vertical[0].y0, horizontal[0].x1, vertical[0].y1)
    xmarks = [a.x0 for a in vertical]
    ymarks = [a.y0 for a in horizontal]
    x, y = [], []
    for label in spans:
        if label["color"] != 0x4d4d4d:
            continue
        value = float(label["text"])
        x0, y0, x1, y1 = label["bbox"]
        if label["origin"][1] > clip[3]:
            x.append((np.log10(value), nearest((x0 + x1) / 2, xmarks)))
        else:
            # Helvetica labels sit slightly below the horizontal gridline.
            y.append((value, nearest(label["origin"][1] - 2.535, ymarks)))
    if min(len(x), len(y)) < 2:
        raise ValueError("Axis calibration requires at least two ticks per axis")
    x_map, y_map = np.polyfit(*zip(*x), 1), np.polyfit(*zip(*y), 1)
    residual = max(abs(np.polyval(line, a) - b) for pts, line in ((x, x_map), (y, y_map)) for a, b in pts)
    if residual >= 0.012 or x_map[0] <= 0 or y_map[0] >= 0:
        raise ValueError(f"Gridline calibration failed ({residual:.5f} pt)")
    return Axes(tuple(x_map), tuple(y_map), clip, float(residual))


def extract_panel(rect, shapes, spans):
    shapes = [s for s in shapes if s["rect"].intersects(rect)
              or rect.contains(s["rect"].top_left)]
    spans = [s for s in spans if rect.contains(s["origin"])]
    axes = panel_axes(shapes, spans)
    series = {s["color"]: s["text"] for s in spans if s["color"] not in (0, 0xffffff, 0x4d4d4d)}
    path_data = defaultdict(lambda: defaultdict(list))
    for shape in shapes:
        stroke, fill = shape.get("color"), shape.get("fill")
        if not (stroke or fill):
            continue
        code = color_key(stroke or fill)
        count = len(shape["items"])
        if code not in series or count == 0:
            continue
        if shape["type"] == "f" and count > 4:
            layer = "band"
        elif shape["type"] == "s":
            layer = "mean" if count > 2 else "default"
        else:
            continue
        coordinates = axes.convert(shape_points(shape))
        path_data[series[code]][layer].append(coordinates)
    if set(path_data) != set(series.values()) or any(not trace["mean"] for trace in path_data.values()):
        raise ValueError("A model label failed to match its visible curve")
    limits = axes.convert([axes.clip[:2], axes.clip[2:]])
    return {"axes": {"bounds": limits, "max_tick_error_pt": axes.max_tick_error_pt},
            "series": dict(path_data),
            "colors": {name: f"#{code:06x}" for code, name in series.items()}}


def extract_curves(number, page):
    spans = page_spans(page)
    band = caption_band(page, number)
    shapes = page.get_drawings()
    panes = sorted([s["rect"] for s in shapes if s.get("fill") == (1., 1., 1.)
                    and s["rect"].width > 180 and inside(s["rect"], band)], key=lambda r: r.x0)
    if len(panes) != 2:
        raise ValueError(f"Found {len(panes)} panels for Figure {number}")
    return {"page": PAGES[number], "method": "pdf_vector_curves",
            "unit": "seconds" if number in (37, 38) else "iterations",
            "panels": {label: extract_panel(rect, shapes, spans)
                       for rect, label in zip(panes, ("classification", "regression"))}}


def extract_decision(doc):
    grids = {}
    for label, (xref, bounds) in PLOT_BOUNDS.items():
        raw = doc.extract_image(xref)["image"]
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        crop = tuple(round(n * s) for n, s in zip(bounds, (*image.size, *image.size)))
        reduced = image.crop(crop).resize(GRID, Image.Resampling.BOX)
        rgb = np.asarray(reduced, dtype=np.int16)
        margin = rgb[:, :, 0] - rgb[:, :, 2]
        bands = np.where(margin > COLOR_THRESHOLD, 1, np.where(margin < -COLOR_THRESHOLD, -1, 0))
        grids[label] = {"crop_pixels": list(crop), "source_image_size": list(image.size),
                        "published_accuracy_percent": 85 if label == "Random Forest" else 80,
                        "color_cells": bands.tolist()}
    return {"page": PAGES[19], "method": "pdf_raster_hue",
            "color_threshold_rgb": COLOR_THRESHOLD, "grid_size": list(GRID), "panels": grids}


def extract(path=None):
    import fitz
    with fitz.open(supplement(path)) as doc:
        figures = {19: extract_decision(doc)}
        figures.update({n: extract_flow(n, doc[PAGES[n] - 1]) for n in FLOW_FIGURES})
        figures.update({n: extract_curves(n, doc[PAGES[n] - 1]) for n in CURVE_FIGURES})
    return {"source": SUPPLEMENT, "sha256": DIGEST, "figures": figures}


def save(path=None, output=ARCHIVE):
    values = extract(path)
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    output.write_bytes(gzip.compress(payload, mtime=0))


def load():
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Use a local copy of the pinned public PDF")
    parser.add_argument("--output", type=Path, default=ARCHIVE)
    arguments = parser.parse_args()
    save(arguments.pdf, arguments.output)
