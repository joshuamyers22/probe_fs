"""Checkpointed paired size-rule refits using the complete public evidence."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

from benchmark_archive import extract_verified, verify_archive, write_archive
from public_selection import check_loss, replay_fold
from run_public_application import checkpoint_valid, dataset_record, methods, peak_gib
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from pgfs.datasets import load_public, sha256
from pgfs.learners import FitCounter
from pgfs.study import clean, fingerprint, initialize, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHIVE = ROOT / "benchmarks/public-application-results-v1.zip"
CONFIG = ROOT / "experiments/public-selection.json"


def read(path):
    return json.loads(path.read_text())


def validate_cell(path, identity, original_sha):
    record = read(path)
    content = dict(record)
    if (content.pop("record_sha256") != fingerprint(content) or record["run_fingerprint"] != identity
            or record["original_sha256"] != original_sha):
        raise ValueError("Changed replay checkpoint identity or content")
    return record


def baseline(output):
    return output / "baseline/results/public-application-v1"


def prepare(output):
    config = read(CONFIG)
    if config["status"] != "frozen" or sha256(BASE_ARCHIVE) != config["baseline_archive_sha256"]:
        raise ValueError("Unfrozen or changed replay inputs")
    for path, expected in config["protocol_artifact_sha256"].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f"Changed replay protocol: {path}")
    extract_verified(BASE_ARCHIVE, output/"baseline")
    original = read(baseline(output)/"provenance.json")
    for path, expected in original["identity"]["source_hashes"].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f"Changed original learner source: {path}")
    provenance = initialize(output, config, original["identity"]["datasets"], ROOT)
    path = output/"frozen-config.json"
    if not path.exists():
        write_json(path, config)
    elif read(path) != config:
        raise ValueError("Changed frozen replay config")
    archive = output/"source-snapshot.zip"
    if archive.exists():
        verify_archive(archive)
    else:
        paths = set(config["protocol_artifact_sha256"]) | {str(CONFIG.relative_to(ROOT))}
        write_archive(archive, ROOT, [ROOT/p for p in paths])
    return config, provenance


def run_design(output, cache, design, config, provenance):
    source = baseline(output)
    name = f"{design['dataset']}-n{design['rows']}-seed{design['seed']}"
    original_config, original_provenance = read(source/"frozen-config.json"), read(source/"provenance.json")
    manifest = original_provenance["identity"]["datasets"]
    data = load_public(design["dataset"], manifest[design["dataset"]], cache, design["rows"], design["seed"])
    specs = methods(original_config, data.X.shape[1], design["seed"], data.task)
    shared = read(source/name/"data.json")
    if clean(dataset_record(data, specs["all"], original_config)) != shared:
        raise ValueError("Replay data, row order or nested partitions changed")
    if design["dataset"] == "yearpredictionmsd" and max(data.row_indices) >= manifest[design["dataset"]]["development_rows"]:
        raise ValueError("Official test rows entered replay")
    index = {int(row): i for i, row in enumerate(data.row_indices)}
    destination = output/name
    destination.mkdir(exist_ok=True)
    for method, spec in specs.items():
        if method in ("all", "l1"):
            continue
        original_path = source/name/f"cell-{method}.json"
        original_sha = sha256(original_path)
        original = checkpoint_valid(original_path, original_provenance["fingerprint"], design, method,
                                    sha256(source/name/"data.json"), spec)
        path = destination/f"cell-{method}.json"
        if path.exists():
            validate_cell(path, provenance["fingerprint"], original_sha)
            continue
        if peak_gib() > config["memory_gib_cap"]:
            raise MemoryError("Replay memory cap exceeded")
        record = {"run_fingerprint": provenance["fingerprint"], "design": design, "method": method,
                  "original_sha256": original_sha, "data_sha256": sha256(source/name/"data.json"),
                  "folds": [], "accepted_float_differences": []}
        counter, started = FitCounter(), time.perf_counter()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                if original["status"] != "complete":
                    raise ValueError("Original method incomplete")
                for fold, diagnostic, split in zip(original["report"]["folds"],
                                                    original["report"]["ranking_diagnostics"], shared["folds"], strict=True):
                    train, test = ([index[r] for r in split[k]] for k in ("train_rows", "test_rows"))
                    replay = replay_fold(data.X, data.y, train, test, spec, fold, diagnostic, counter)
                    delta = check_loss(replay["loss"]["one_se"], fold["outer_loss"])
                    if delta:
                        record["accepted_float_differences"].append({"fold": fold["fold"], "difference": delta})
                    for policy, selected in replay["selected"].items():
                        if len(selected) == data.X.shape[1]:
                            check_loss(replay["loss"][policy], fold["outer_loss_all_features"])
                    record["folds"].append(replay)
                record["status"] = "complete"
            except (ValueError, RuntimeError, FloatingPointError, MemoryError, ConvergenceWarning) as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        record.update(new_fit_counts=counter.as_dict(), elapsed_seconds=time.perf_counter()-started,
                      peak_memory_gib=peak_gib(), warnings=[f"{w.category.__name__}: {w.message}" for w in observed])
        record = clean(record)
        record["record_sha256"] = fingerprint(record)
        write_json(path, record)
        print(f"{name} {method}: {record['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--design", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.design:
        config, provenance = read(output/"frozen-config.json"), read(output/"provenance.json")
        for path, expected in config["protocol_artifact_sha256"].items():
            if sha256(ROOT/path) != expected:
                raise ValueError(f"Changed frozen replay source: {path}")
        with threadpool_limits(limits=1):
            run_design(output, args.cache.resolve(), json.loads(args.design), config, provenance)
        return
    config, provenance = prepare(output)
    from analyze_public_application import analyze
    analyze(baseline(output), output/"baseline-analysis")
    original = read(baseline(output)/"frozen-config.json")
    ledger_path = output/"worker-time.json"
    ledger = read(ledger_path) if ledger_path.exists() else {"invocations": []}
    for d, n, s in itertools.product(original["datasets"], original["rows"], original["seeds"]):
        design = {"dataset": d, "rows": n, "seed": s}
        remaining = config["worker_hours_cap"]*3600 - sum(r["seconds"] for r in ledger["invocations"])
        if remaining <= 0:
            raise TimeoutError("Replay worker-hour cap exhausted")
        started = time.perf_counter()
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                            "--cache", str(args.cache.resolve()), "--design", json.dumps(design)],
                           cwd=ROOT, check=True, timeout=remaining)
        finally:
            ledger["invocations"].append({"design": design, "seconds": time.perf_counter()-started})
            write_json(ledger_path, ledger)
    records = [read(p) for p in output.glob("*/cell-*.json")]
    failed = sum(r["status"] != "complete" for r in records)
    print(f"Recorded {len(records)} paired methods, {failed} failures.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
