"""Frozen, checkpointed local/global-shadow and full-conditioning controls."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from pgfs.budget import spec_importance_fits
from pgfs.datasets import sha256
from pgfs.experiment_models import StudyConstraints
from pgfs.experiments import _row, run_spec
from pgfs.metrics import compare_primary_endpoint
from pgfs.study import (
    clean,
    detailed_report,
    fingerprint,
    initialize,
    make_simulation,
    make_spec,
    write_json,
)


def specs(config, seed):
    base = make_spec(config, seed, config["simulation"]["candidate_k"], "regression")
    output = {}
    for rung in config["rungs"]:
        t, splits = rung["contexts"], rung["loco_splits"]
        output[f"local-T{t}"] = base.replace(n_contexts=t, gate="soft", shadow_scope="local", label=f"local-T{t}")
        output[f"global-T{t}"] = base.replace(n_contexts=t, gate="soft", shadow_scope="global", label=f"global-T{t}")
        output[f"loco-matched-T{t}"] = base.replace(context_kind="full", n_importance_splits=splits, gate="soft", label=f"loco-matched-T{t}")
    output["loco-gated-B1"] = base.replace(context_kind="full", n_importance_splits=1, gate="soft", label="loco-gated-B1")
    output["loco-ungated-B1"] = base.replace(context_kind="full", n_importance_splits=1, gate="none", label="loco-ungated-B1")
    return output


def comparison(candidate, comparator, reports, predicted, min_effect):
    a, b = reports[candidate], reports[comparator]
    result = compare_primary_endpoint(a["primary_endpoint"], b["primary_endpoint"], min_effect)
    counts = [r["main_results"]["model_fits_total"] for r in (a, b)]
    measured = abs(counts[0]-counts[1])/max(counts)
    expected = abs(predicted[candidate]-predicted[comparator])/max(predicted[candidate], predicted[comparator])
    matched = measured <= .05 and expected <= .05
    result.update(candidate=candidate, comparator=comparator,
                  endpoint_constraints_met=result["comparison_valid"], budget_matched=matched,
                  measured_budget_imbalance=measured, predicted_budget_imbalance=expected)
    if not matched:
        result.update(comparison_valid=False, reason="budget_not_matched", meets_minimum_effect=False)
    return result


def preflight(config):
    sim = config["simulation"]
    p = sim["n_classes"]*sim["class_size"] + sim["n_null_blocks"]*sim["null_block_size"] + sim["n_independent_nulls"]
    methods = specs(config, config["seeds"][0])
    counts = {name: spec_importance_fits(spec, p) for name, spec in methods.items()}
    for rung in config["rungs"]:
        t = rung["contexts"]
        for alternative in (f"global-T{t}", f"loco-matched-T{t}"):
            a, b = counts[f"local-T{t}"], counts[alternative]
            if abs(a-b)/max(a, b) > .05:
                raise ValueError(f"Prespecified compute match infeasible: {alternative}")
    return counts


def run_seed(config, output, root):
    provenance = initialize(output, config, {}, root)
    sim = config["simulation"]
    constraints = StudyConstraints(**{k: sim[k] for k in ("noninferiority_margin", "k_max", "min_effect")})
    predicted = preflight(config)
    for n, rho, snr, seed in itertools.product(sim["n"], sim["rho"], sim["snr"], config["seeds"]):
        design = {"dataset": "simulation", "n": n, "rho": rho, "snr": snr, "seed": seed}
        data = None
        for method, spec in specs(config, seed).items():
            path = output / f"cell-{fingerprint(dict(design, method=method))[:16]}.json"
            if path.exists():
                cell = json.loads(path.read_text())
                if cell["run_fingerprint"] != provenance["fingerprint"] or cell["spec"] != spec.as_dict():
                    raise ValueError(f"Changed checkpoint identity: {path}")
                print(f"resumed {design} {method}", flush=True)
                continue
            if data is None:
                data = make_simulation(sim, design)
            started = time.perf_counter()
            result = run_spec(data, spec, constraints, config["outer_folds"], config["inner_folds"])
            write_json(path, {
                "run_fingerprint": provenance["fingerprint"], "design": design, "method": method,
                "data": {"params": data.params, "player_names": data.players.names,
                         "signal_classes": [sorted(c) for c in data.signal_classes.classes]},
                "spec": spec.as_dict(), "report": detailed_report(result),
                "row": _row(method, 0, result), "predicted_importance_fits": predicted[method],
                "elapsed_seconds": time.perf_counter()-started,
            })
            print(f"saved {design} {method}", flush=True)


def launch(config_path, output, root):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.log").open("a") as log:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                        "--output", str(output), "--shard"], cwd=root, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/context-controls.json"))
    parser.add_argument("--output", type=Path, default=Path("results/context-controls-v1"))
    parser.add_argument("--shard", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text())
    if args.shard:
        if config["driver_sha256"] != sha256(Path(__file__)):
            raise ValueError("Driver source changed after freezing")
        with threadpool_limits(limits=1):
            run_seed(config, args.output, root)
        return
    config["driver_sha256"] = sha256(Path(__file__))
    predicted = preflight(config)
    protocol_files = []
    for name, expected in config["protocol_artifact_sha256"].items():
        path = root / name
        if sha256(path) != expected:
            raise ValueError(f"Changed protocol artifact: {name}")
        protocol_files.append(path)
    output = args.output.resolve()
    provenance = initialize(output, config, {}, root)
    write_json(output / "frozen-config.json", config)
    write_json(output / "compute-preflight.json", predicted)
    archive = output / "source-snapshot.zip"
    if archive.exists():
        if sha256(archive) != json.loads((output / "source-snapshot.json").read_text())["sha256"]:
            raise ValueError("Source archive checksum mismatch")
    else:
        paths = list((root / "src/pgfs").glob("*.py")) + [root / name for name in
                  ("pyproject.toml", "uv.lock", ".python-version", "README.md")] + [args.config.resolve(), Path(__file__).resolve()] + protocol_files
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in paths:
                zipped.write(path, str(path.relative_to(root)))
        write_json(output / "source-snapshot.json", {"sha256": sha256(archive)})
    jobs = []
    for seed in config["seeds"]:
        shard = dict(config, seeds=[seed], parent_fingerprint=provenance["fingerprint"])
        path = output / f"config-seed-{seed}.json"
        if path.exists() and json.loads(path.read_text()) != clean(shard):
            raise ValueError(f"Changed shard configuration: {path}")
        write_json(path, shard)
        jobs.append((path, output / f"seed-{seed}", root))
    sim = config["simulation"]
    expected = len(jobs)*len(sim["n"])*len(sim["rho"])*len(sim["snr"])*len(predicted)
    started = time.perf_counter()
    print(f"Frozen controls: {expected} nested evaluations, {config['workers']} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        pending = {pool.submit(launch, *job) for job in jobs}
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                future.result()
            count = len(list(output.glob("seed-*/cell-*.json")))
            print(f"Progress {count}/{expected}; elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
    if len(list(output.glob("seed-*/cell-*.json"))) != expected:
        raise ValueError("Completed count differs from frozen design")
    timing = output / ("resume-execution.json" if (output / "execution.json").exists() else "execution.json")
    write_json(timing, {"evaluations": expected, "elapsed_seconds": time.perf_counter()-started})
    print("All context controls completed", flush=True)


if __name__ == "__main__":
    main()
