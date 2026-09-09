"""Pinned public benchmark downloads and deterministic development subsets.

No scaling is performed here: preprocessing belongs inside each fitted learner.
The Year Prediction official test partition is never returned by this loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .players import Players


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_archive(entry: dict, cache: Path, download: bool = False) -> Path:
    path = cache / entry["filename"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}; run python -m pgfs.datasets first")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(entry["url"], timeout=120) as source, temporary.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
            if sha256(temporary) != entry["sha256"]:
                raise ValueError(f"Checksum mismatch for {path.name}; source may have changed")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Checksum mismatch for {path.name}; refusing changed data")
    return path


@dataclass
class PublicData:
    X: np.ndarray
    y: np.ndarray
    players: Players
    task: str
    row_indices: np.ndarray
    provenance: dict


def load_public(name: str, entry: dict, cache: Path, rows: int, seed: int) -> PublicData:
    """Sample original row IDs; validate schema/counts before returning data.

    Superconductivity keeps the first row per exact formula, independent of y,
    to prevent the same formula appearing in multiple CV folds. This is not a
    claim that different formulas are chemically independent.
    """
    if rows < 20:
        raise ValueError("Public subsets require at least 20 rows")
    archive = verified_archive(entry, cache)
    with zipfile.ZipFile(archive) as zipped:
        if name == "superconductivity":
            with zipped.open("unique_m.csv") as source:
                formulas = list(csv.DictReader(io.TextIOWrapper(source)))
            seen = set()
            eligible = []
            for index, row in enumerate(formulas):
                formula = row["material"].strip()
                if formula not in seen:
                    seen.add(formula)
                    eligible.append(index)
            with zipped.open("train.csv") as source:
                reader = io.TextIOWrapper(source)
                header = next(csv.reader([reader.readline()]))
                matrix = np.loadtxt(reader, delimiter=",")
            if header[-1] != "critical_temp" or len(formulas) != len(matrix):
                raise ValueError("Superconductivity target/formula schema changed")
            names = header[:-1]
            X, y = matrix[:, :-1], matrix[:, -1]
        elif name == "miniboone":
            with zipped.open("MiniBooNE_PID.txt") as source:
                counts = [int(value) for value in source.readline().split()]
                X = np.loadtxt(source)
            if len(counts) != 2 or sum(counts) != len(X):
                raise ValueError("MiniBooNE header counts do not match events")
            y = np.r_[np.ones(counts[0]), np.zeros(counts[1])]
            names = [f"pid_{i + 1}" for i in range(X.shape[1])]
            eligible = np.arange(len(X))
        elif name == "yearpredictionmsd":
            # Stream the full file to check its count but retain only sampled
            # development rows. Never materialize official held-out outcomes.
            eligible = np.arange(entry["development_rows"])
            selected = np.sort(np.random.default_rng(seed).choice(
                eligible, min(rows, len(eligible)), replace=False
            ))
            keep = set(selected.tolist())
            sampled = []
            with zipped.open("YearPredictionMSD.txt") as source:
                count = 0
                for index, line in enumerate(source):
                    if index in keep:
                        sampled.append(np.fromstring(line.decode("ascii"), sep=","))
                    count += 1
            if count != entry["rows"]:
                raise ValueError("Year Prediction row count changed")
            matrix = np.array(sampled)
            X, y = matrix[:, 1:], matrix[:, 0]
            names = [f"timbre_{i + 1}" for i in range(X.shape[1])]
        else:
            raise ValueError(f"Unknown dataset: {name}")

    if name != "yearpredictionmsd":
        if X.shape != (entry["rows"], entry["features"]):
            raise ValueError(f"Unexpected full shape for {name}: {X.shape}")
        eligible = np.asarray(eligible)
        if rows >= len(eligible):
            selected = eligible
        elif entry["task"] == "binary":
            selected, _ = train_test_split(
                eligible, train_size=rows, stratify=y[eligible], random_state=seed
            )
        else:
            selected = np.random.default_rng(seed).choice(eligible, rows, replace=False)
        selected = np.sort(selected)
        X, y = X[selected], y[selected]
    if X.shape[1] != entry["features"] or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Invalid numeric data for {name}")
    # MiniBooNE uses -999 for unavailable detector quantities despite the
    # catalog's 'no missing values' label. Imputation is fitted within each fold.
    sentinel_count = int(np.sum(X == -999)) if name == "miniboone" else 0
    if name == "miniboone":
        X[X == -999] = np.nan
    return PublicData(
        X, y, Players.singletons(X.shape[1], names), entry["task"], selected,
        {"source": entry, "eligible_rows": len(eligible),
         "sample_seed": seed, "sample_rows": len(selected),
         "missing_sentinel_values": sentinel_count,
         "selection_policy": entry["selection_policy"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/datasets.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for name, entry in json.loads(args.manifest.read_text()).items():
        print(f"{name}: {verified_archive(entry, args.cache, download=True)}", flush=True)


if __name__ == "__main__":
    main()
