"""Project verified author downloads onto the numeric columns the figures use."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
MODEL = "model_name"
TASK = "data__keyword"
VALID = "mean_val_score"
TEST = "mean_test_score"
COMMON = ("hp", MODEL, TASK, VALID, TEST, "mean_time")
REMOVE = "transform__0__num_features_to_remove"
ADD = "transform__0__multiplier"
SMOOTH = "transform__2__cov_mult"
ROTATE = "transform__2__deactivated"
ABLATIONS = ("high_frequencies", "show_uninformative_features",
             "remove_uninformative_features", "add_uninformative_features", "random_rotation")
PANELS = ("numerical_classification_medium", "numerical_regression_medium",
          "categorical_classification_medium", "categorical_regression_medium")
TIME_REVISION = "5b2d812329d191f949f9d085616a6c41d3198688"
TUNED_REVISION = "7265a3f41ae2d89a4b12e5609bb6591ab8f1161f"
TIME_PANELS = (
    "numerical_classif_medium", "numerical_regression_medium",
    "categorical_classif_medium", "categorical_regression_medium",
)
LARGE_PANELS = tuple(panel.replace("_medium", "_large") for panel in TIME_PANELS)
BO_COLUMNS = (MODEL, TASK, VALID, TEST, "mean_time", "step", "data__categorical",
              "data__regression", "mean_r2_test", "mean_r2_val")
BO_PINS = {
    "bo_nn": "5ba3c8ca0dc2848b5e17d678d69dc4170f1e2d8693b0847972abec3746f1cfc1",
    "bo_nn_2": "a9c3e801ea7a9e268c87abbaa9c4ad0e575cb056c7855660a5bd8508cf10c113",
    "bo_trees": "1361fc25fb1c98d1d59f2432b9165cce42a5a9e65df4399011dc77b23646b017",
    "bo_trees_2": "f8b16766304a356a043c9efde9030fcacfb8ca298ce056423a0f9c66545d034c",
}


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    columns: tuple
    group: str
    included: tuple


SOURCES = {
    "benchmark": Source(
        "https://ndownloader.figshare.com/files/40081681?source=cs2229",
        "6d709ae8e10ece7f0a66260ae0f2d8f5d914aed97ce5ab8e7674e654ed6ec1e4",
        ("benchmark", *COMMON), "benchmark", PANELS,
    ),
    "ablations": Source(
        "https://media.githubusercontent.com/media/LeoGrin/tabular-benchmark/"
        "f3fe60384ba7852e3c0b755c20cce478857d9878/analyses/results/random_search_xps.csv?source=cs2229",
        "25314ccb2d138dae43fd23840f729f62fba0b5f5ab559aeb3727403dedb4e429",
        ("xp", *COMMON, "method", REMOVE, ADD, SMOOTH, ROTATE), "xp", ABLATIONS,
    ),
    "time_numerical": Source(
        f"https://media.githubusercontent.com/media/LeoGrin/tabular-benchmark/"
        f"{TIME_REVISION}/analyses/results/random_search_benchmark_numerical.csv?source=cs2229-time",
        "395ea2c330a1e1e414b8e15ccba9bbf23f497241c265ea675670f6ae0e222e62",
        ("benchmark", *COMMON), "benchmark", TIME_PANELS[:2],
    ),
    "time_categorical": Source(
        f"https://media.githubusercontent.com/media/LeoGrin/tabular-benchmark/"
        f"{TIME_REVISION}/analyses/results/random_search_benchmark_categorical.csv?source=cs2229-time",
        "d035715efaaf54261dd31c107159f28a96804164485a3c53de656f8fc632020b",
        ("benchmark", *COMMON), "benchmark", TIME_PANELS[2:],
    ),
}
SOURCES.update({
    f"large_{kind}": Source(source.url, source.sha256, source.columns, "benchmark", panels)
    for kind, panels in (("numerical", LARGE_PANELS[:2]), ("categorical", LARGE_PANELS[2:]))
    for source in (SOURCES[f"time_{kind}"],)
})
SOURCES.update({
    name: Source(
        "https://media.githubusercontent.com/media/LeoGrin/tabular-benchmark/"
        f"{TIME_REVISION}/analyses/results/{name}.csv?source=cs2229-bo",
        digest, BO_COLUMNS, "data__categorical", ("0.0",),
    )
    for name, digest in BO_PINS.items()
})
SOURCES.update({
    f"tuned_{kind}": Source(
        "https://media.githubusercontent.com/media/LeoGrin/tabular-benchmark/"
        f"{TUNED_REVISION}/analyses/results/random_search_benchmark_{kind}.csv?source=cs2229-tuned",
        digest, ("benchmark", *COMMON), "benchmark", panels,
    )
    for kind, digest, panels in (
        ("numerical", "cf01851eafe29ce6a3cd7dec6e42a5d23f79e03a91e2750ec83abe779d126bbe", TIME_PANELS[:2]),
        ("categorical", "5a48d60bb23e7f983bd757364460abef8ad7e351febbec86d1d1f3c153a7e672", TIME_PANELS[2:]),
    )
})


class HashingReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        size = self.stream.readinto(buffer)
        if size:
            self.digest.update(memoryview(buffer)[:size])
        return size


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(name, raw=None):
    source = SOURCES[name]
    CACHE.mkdir(exist_ok=True)
    target = CACHE / f"{name}.csv.gz"
    proof = CACHE / f"{name}.json"
    if target.is_file() and proof.is_file():
        record = json.loads(proof.read_text(encoding="utf-8"))
        if (record["source_sha256"], record["projection_sha256"], record["columns"]) == (
            source.sha256, sha(target), list(source.columns)
        ):
            return target

    request = urllib.request.Request(source.url, headers={"Accept-Encoding": "identity"})
    download = raw.open("rb") if raw else urllib.request.urlopen(request, timeout=90)
    temporary = tempfile.NamedTemporaryFile(dir=CACHE, prefix=f"{name}-", suffix=".tmp", delete=False)
    temp = Path(temporary.name)
    proof_temp = temp.with_suffix(".json.tmp")
    try:
        with download, temporary, gzip.GzipFile(filename="", mode="wb", fileobj=temporary, mtime=0) as packed, \
                io.TextIOWrapper(packed, encoding="utf-8", newline="") as output:
            hashing = HashingReader(download)
            decoded = io.TextIOWrapper(io.BufferedReader(hashing, 1024 * 1024),
                                       encoding="utf-8-sig", newline="")
            csv.field_size_limit(16 * 1024 * 1024)
            reader = csv.reader(decoded)
            header = next(reader)
            if any(header.count(key) != 1 for key in source.columns):
                raise ValueError(f"Unexpected author CSV schema for {name}")
            columns = [header.index(key) for key in source.columns]
            selector = header.index(source.group)
            writer = csv.writer(output)
            writer.writerow(source.columns)
            rows = {group: 0 for group in source.included}
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(f"Invalid author CSV row {reader.line_num}")
                group = row[selector]
                if group in rows:
                    writer.writerow([row[column] for column in columns])
                    rows[group] += 1
            if hashing.digest.hexdigest() != source.sha256 or not all(rows.values()):
                raise ValueError(f"Downloaded author data failed integrity verification for {name}")
        temp.replace(target)
        record = dict(source_sha256=source.sha256, projection_sha256=sha(target),
                      rows=rows, columns=list(source.columns))
        proof_temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        proof_temp.replace(proof)
    finally:
        temp.unlink(missing_ok=True)
        proof_temp.unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--raw", type=Path, help="Use a local copy of the exact author download")
    args = parser.parse_args()
    print(prepare(args.source, args.raw))
