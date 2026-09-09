"""Checkpointed public application study and isolated 600-row cost calibration."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import resource
import subprocess
import sys
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from public_baseline import evaluate_baseline, folds
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from pgfs.budget import budget_ladder
from pgfs.datasets import load_public, sha256
from pgfs.nested import OUTER_CV_SEED_OFFSET, nested_evaluate
from pgfs.selection import INNER_CV_SEED_OFFSET
from pgfs.study import (
    clean,
    detailed_report,
    fingerprint,
    initialize,
    make_spec,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ["experiments/run_public_application.py", "experiments/public_baseline.py"]


def methods(config, p, seed, task):
    candidates = sorted(set(config["candidate_k"] + ([p] if config["include_all_features"] else [])))
    spec = make_spec(config, seed, candidates, task)
    result = {"all": spec, "l1": spec}
    for pair in budget_ladder(spec, p, config["budgets"]):
        if not pair.feasible:
            raise ValueError("Infeasible public application budget")
        result[f"gated-T{pair.gated.n_contexts}"] = pair.gated
        result[f"ungated-T{pair.gated.n_contexts}"] = pair.ungated
    return result


def peak_gib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)


def dataset_record(data, spec, config):
    rows = data.row_indices
    partitions = []
    for train, test in folds(data.X, data.y, spec, config["outer_folds"], OUTER_CV_SEED_OFFSET):
        inner = folds(data.X[train], data.y[train], spec, config["inner_folds"], INNER_CV_SEED_OFFSET)
        partitions.append({"train_rows": rows[train].tolist(), "test_rows": rows[test].tolist(),
                           "inner": [{"train_rows": rows[train[tr]].tolist(),
                                      "valid_rows": rows[train[va]].tolist()} for tr, va in inner]})
    return {"row_indices": rows.tolist(), "provenance": data.provenance,
            "feature_names": data.players.names, "folds": partitions,
            "X_sha256": hashlib.sha256(np.asarray(data.X, dtype="<f8").tobytes()).hexdigest(),
            "y_sha256": hashlib.sha256(np.asarray(data.y, dtype="<f8").tobytes()).hexdigest()}


def checkpoint_valid(path, run_fingerprint, design, method, data_sha, spec):
    record = json.loads(path.read_text())
    expected = (run_fingerprint, design, method, data_sha, spec.as_dict())
    actual = tuple(record[k] for k in ("run_fingerprint", "design", "method", "data_sha256", "spec"))
    if actual != expected:
        raise ValueError(f"Changed application checkpoint identity: {path}")
    content = dict(record)
    recorded_digest = content.pop("record_sha256")
    if fingerprint(content) != recorded_digest:
        raise ValueError(f"Changed application checkpoint content: {path}")
    return record


def run_design(config, manifest, design, output, cache, provenance):
    started = time.perf_counter()
    data = load_public(design["dataset"], manifest[design["dataset"]], cache, design["rows"], design["seed"])
    if len(data.y) != design["rows"]:
        raise ValueError("Requested subset is larger than eligible development data")
    if design["dataset"] == "yearpredictionmsd" and np.any(data.row_indices >= manifest[design["dataset"]]["development_rows"]):
        raise ValueError("Official test rows entered development")
    specifications = methods(config, data.X.shape[1], design["seed"], data.task)
    output.mkdir(parents=True, exist_ok=True)
    data_record = clean(dataset_record(data, specifications["all"], config))
    data_path = output / "data.json"
    if data_path.exists():
        if json.loads(data_path.read_text()) != data_record:
            raise ValueError("Application data or folds changed")
    else:
        write_json(data_path, data_record)
    data_sha = sha256(data_path)
    elapsed_loading = time.perf_counter()-started
    for method, spec in specifications.items():
        path = output / f"cell-{method}.json"
        if path.exists():
            checkpoint_valid(path, provenance["fingerprint"], design, method, data_sha, spec)
            continue
        if peak_gib() > config["memory_gib_cap"]:
            raise MemoryError("Worker memory cap reached before next method")
        started = time.perf_counter()
        record = {"run_fingerprint": provenance["fingerprint"], "design": design,
                  "method": method, "spec": spec.as_dict(), "data_sha256": data_sha,
                  "scientific_evidence": config["scientific_evidence"]}
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                if method in ("all", "l1"):
                    report = evaluate_baseline(data.X, data.y, spec, config["outer_folds"],
                                               config["inner_folds"], config["l1"], method)
                else:
                    result = nested_evaluate(data.X, data.y, data.players, spec,
                                             n_outer_folds=config["outer_folds"], n_inner_folds=config["inner_folds"])
                    report = detailed_report(result)
                    for fold, detail in zip(result.folds, report["folds"]):
                        detail["selected_indices"] = list(fold.selected)
                if not np.isfinite(report["main_results"]["mean_outer_loss"]):
                    raise ValueError("Nonfinite outer loss")
                record.update(status="complete", report=report)
            except (ValueError, RuntimeError, FloatingPointError, MemoryError, ConvergenceWarning) as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        record.update(elapsed_seconds=time.perf_counter()-started, peak_memory_gib=peak_gib(),
                      warnings=dict(Counter(f"{w.category.__name__}: {w.message}" for w in observed)))
        record = clean(record)
        record["record_sha256"] = fingerprint(record)
        write_json(path, record)
        print(f"{design['dataset']} n={design['rows']} seed={design['seed']} {method}: {record['status']}", flush=True)
        if record["peak_memory_gib"] > config["memory_gib_cap"]:
            raise MemoryError("Worker memory cap reached; completed checkpoint retained")
    timing = output / "execution.json"
    if not timing.exists():
        write_json(timing, {"loading_seconds": elapsed_loading, "peak_memory_gib": peak_gib()})


def freeze(output, config, manifest):
    config = dict(config, driver_sha256={name: sha256(ROOT/name) for name in DRIVERS})
    for name, expected in config["protocol_artifact_sha256"].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f"Changed application protocol: {name}")
    provenance = initialize(output, config, manifest, ROOT)
    if not (output/"frozen-config.json").exists():
        write_json(output/"frozen-config.json", config)
    elif json.loads((output/"frozen-config.json").read_text()) != config:
        raise ValueError("Changed frozen application configuration")
    archive = output/"source-snapshot.zip"
    if archive.exists():
        if sha256(archive) != json.loads(archive.with_suffix(".json").read_text())["sha256"]:
            raise ValueError("Changed application source snapshot")
    else:
        paths = set(provenance["identity"]["source_hashes"]) | set(DRIVERS) | set(config["protocol_artifact_sha256"])
        paths |= {"experiments/datasets.json", "experiments/public-application.json", ".python-version"}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name in sorted(paths):
                zipped.write(ROOT/name, name)
        write_json(archive.with_suffix(".json"), {"sha256": sha256(archive)})
    return config, provenance


def validate_scientific_config(config):
    if config["status"] != "frozen" or not config["protocol_artifact_sha256"]:
        raise ValueError("Freeze the application config and protocol after calibration before scientific fitting")
    name = config.get("cost_preflight")
    if not name or name not in config["protocol_artifact_sha256"]:
        raise ValueError("Scientific fitting requires a checksummed cost preflight")
    if sha256(ROOT/name) != config["protocol_artifact_sha256"][name]:
        raise ValueError("Cost preflight checksum changed")
    preflight = json.loads((ROOT/name).read_text())
    if not preflight["within_compute_gate"] or preflight["calibration_failures"]:
        raise ValueError("Cost calibration did not pass")
    for key, value in preflight["planned_config"].items():
        if key not in ("status", "protocol_artifact_sha256", "cost_preflight") and config[key] != value:
            raise ValueError(f"Scientific design differs from cost preflight: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/public-application.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--design", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output, cache = args.output.resolve(), args.cache.resolve()
    manifest = json.loads((ROOT/"experiments/datasets.json").read_text())
    config = json.loads(args.config.read_text())
    if args.design:
        for name, expected in config["driver_sha256"].items():
            if sha256(ROOT/name) != expected:
                raise ValueError("Application driver changed after freezing")
        provenance = initialize(output.parent, config, manifest, ROOT)
        with threadpool_limits(limits=1):
            run_design(config, manifest, json.loads(args.design), output, cache, provenance)
        return
    if args.calibrate:
        config.update(name=output.name, status="calibration", scientific_evidence=False,
                      rows=[600], seeds=[599], workers=1)
    else:
        validate_scientific_config(config)
    if config["workers"] != 1:
        raise ValueError("This runner executes one isolated worker at a time; set workers=1")
    config, provenance = freeze(output, config, manifest)
    designs = [{"dataset": d, "rows": n, "seed": s} for d, n, s in
               itertools.product(config["datasets"], config["rows"], config["seeds"])]
    ledger_path = output/"worker-time.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"invocations": []}
    for design in designs:
        remaining = config["worker_hours_cap"]*3600 - sum(r["seconds"] for r in ledger["invocations"])
        if remaining <= 0:
            raise TimeoutError("Worker-hour cap exhausted; checkpoints retained")
        directory = output/f"{design['dataset']}-n{design['rows']}-seed{design['seed']}"
        started = time.perf_counter()
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with (directory/"execution.log").open("a") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(output/"frozen-config.json"),
                                "--output", str(directory), "--cache", str(cache), "--design", json.dumps(design)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining)
        finally:
            ledger["invocations"].append({"design": design, "seconds": time.perf_counter()-started})
            write_json(ledger_path, ledger)
        print(f"Completed or recorded failures: {directory.name}", flush=True)
    records = [json.loads(p.read_text()) for p in output.glob("*/cell-*.json")]
    failed = sum(r["status"] != "complete" for r in records)
    print(f"Recorded {len(records)} methods; {failed} failures. Analyze cost before scientific fitting.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
