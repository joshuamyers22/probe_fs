"""Post-hoc endpoint diagnostics on archived folds; never new confirmatory evidence."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pgfs.datasets import sha256
from pgfs.learners import FitCounter, evaluate_player_set
from pgfs.metrics import class_recall
from pgfs.nested import FINAL_FIT_KEY, OUTER_CV_SEED_OFFSET
from pgfs.players import derive_seed
from pgfs.selection import make_cv
from pgfs.simulate import make_primary
from pgfs.study import fingerprint, initialize, make_spec, write_json


def residual_signal_variance(counts, betas, rho):
    """Gaussian noisy-proxy DGP: sum beta_h^2 Var(U_h | r_h measurements)."""
    return sum(beta**2 * (1-rho)/(1-rho+count*rho) for count, beta in zip(counts, betas))


def oracle_allocation(k, betas, rho, class_size=2):
    choices = [counts for counts in itertools.product(range(class_size+1), repeat=len(betas)) if sum(counts) == k]
    return min(choices, key=lambda counts: (residual_signal_variance(counts, betas, rho), counts))


def theory(betas, rho, margin, candidates):
    reference = residual_signal_variance([2]*len(betas), betas, rho)
    rows = []
    for k in range(1, 2*len(betas)+1):
        counts = oracle_allocation(k, betas, rho)
        excess = residual_signal_variance(counts, betas, rho)-reference
        rows.append({"rho": rho, "k": k, "class_counts": counts,
                     "oracle_excess_mse": excess, "within_margin": excess <= margin,
                     "in_original_path": k in candidates})
    return rows


def aggregate_folds(folds, margin, k_max):
    loss = float(np.mean([f["loss"] for f in folds]))
    reference = float(np.mean([f["reference_loss"] for f in folds]))
    return {"mean_loss": loss, "mean_reference_loss": reference,
            "mean_excess_loss": loss-reference,
            "mean_size": float(np.mean([f["size"] for f in folds])),
            "mean_recall": float(np.mean([f["recall"] for f in folds])),
            "constraints_met": bool(loss <= reference+margin and all(f["size"] <= k_max for f in folds))}


def evaluate_subset(data, spec, tr, te, selected, fold, counter, reference):
    cols = data.players.select_columns(selected)
    loss = evaluate_player_set(
        spec.learner, spec.loss_fn, data.X[np.ix_(tr, cols)], data.y[tr],
        data.X[np.ix_(te, cols)], data.y[te],
        seed=derive_seed(spec.seed, fold, FINAL_FIT_KEY), counter=counter, kind="selection",
    )
    return {"fold": fold, "loss": loss, "reference_loss": reference,
            "size": len(selected), "recall": class_recall(selected, data.signal_classes)}


def summarize(output, config):
    import csv
    from collections import Counter

    cells = [json.loads(p.read_text()) for p in sorted(output.glob("audit-*.json"))]
    policies = ["original_one_se", "inner_argmin", "fixed_top5", "fixed_top6"]
    groups = {}
    for cell in cells:
        d = cell["design"]
        for policy in policies:
            pair = cell["policies"][policy]
            for method, result in pair.items():
                key = (d["n"], d["snr"], policy, method)
                groups.setdefault(key, []).append(result)
    rows = []
    for (n, snr, policy, method), items in sorted(groups.items()):
        rows.append({"n": n, "snr": snr, "policy": policy, "method": method,
                     "runs": len(items), "valid": sum(i["constraints_met"] for i in items),
                     **{key: float(np.mean([i[key] for i in items])) for key in
                        ["mean_excess_loss", "mean_size", "mean_recall"]}})
    with (output / "policy-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {}
    for policy in policies:
        paired = [c["policies"][policy] for c in cells]
        valid = [p for p in paired if all(r["constraints_met"] for r in p.values())]
        totals[policy] = {
            "valid_pairs": len(valid),
            "gated_passes": sum(p["gated"]["constraints_met"] for p in paired),
            "ungated_passes": sum(p["ungated"]["constraints_met"] for p in paired),
            "both_perfect_recall": sum(p["gated"]["mean_recall"] == p["ungated"]["mean_recall"] == 1.0 for p in valid),
            "valid_recall_difference_mean": float(np.mean([p["gated"]["mean_recall"]-p["ungated"]["mean_recall"] for p in valid])) if valid else None,
        }
    fold_rows = [f for c in cells for f in c["selection_diagnostics"]]
    shrinkage = {}
    for snr in config["simulation"]["snr"]:
        selected = [f for f in fold_rows if f["snr"] == snr]
        shrinkage[str(snr)] = {
            "folds": len(selected), "shrunk": sum(f["chosen"] < f["argmin"] for f in selected),
            "chosen_counts": dict(Counter(f["chosen"] for f in selected)),
            "argmin_counts": dict(Counter(f["argmin"] for f in selected)),
            "median_se_at_min": float(np.median([f["se_at_min"] for f in selected])),
        }
    oracles = {}
    # Oracle fits are cached across method and budget, and counted once per DGP/seed.
    unique = {}
    for cell in cells:
        d = cell["design"]
        unique[(d["n"], d["rho"], d["snr"], d["seed"])] = cell
    for k in [3, 5, 6]:
        items = [c["oracles"][str(k)] for c in unique.values()]
        oracles[str(k)] = {"datasets": len(items), "passes": sum(i["constraints_met"] for i in items),
                           "mean_excess_loss": float(np.mean([i["mean_excess_loss"] for i in items]))}
    write_json(output / "summary.json", {"paired_policies": totals, "shrinkage": shrinkage,
                                         "oracle_policies": oracles, "cells": len(cells)})
    print(json.dumps({"paired_policies": totals, "oracle_policies": oracles, "shrinkage": shrinkage}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("results/expanded-simulation-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/endpoint-audit-v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((args.parent/"frozen-config.json").read_text())
    manifest = json.loads((root/"experiments/datasets.json").read_text())
    parent = initialize(args.parent, config, manifest, root)
    hashes = json.loads((args.parent/"analysis-provenance.json").read_text())["cell_sha256"]
    for name, digest in hashes.items():
        if sha256(args.parent/name) != digest:
            raise ValueError(f"Changed parent cell: {name}")
    spec_config = {"parent_fingerprint": parent["fingerprint"], "driver_sha256": sha256(Path(__file__)),
                   "policies": ["original_one_se", "inner_argmin", "fixed_top5", "fixed_top6"],
                   "oracle_sizes": [3, 5, 6], "purpose": "Post-hoc diagnosis on reused folds; not a new nested study or matched-compute result."}
    provenance = initialize(args.output, spec_config, manifest, root)
    shutil.copyfile(__file__, args.output/"audit-source.py")
    sim = config["simulation"]
    betas = [1.0-0.15*h for h in range(sim["n_classes"])]
    write_json(args.output/"population-risk.json", [row for rho in sim["rho"] for row in theory(betas, rho, sim["noninferiority_margin"], sim["candidate_k"])])
    paths = sorted(args.parent.glob("seed-*/cell-*.json"))
    if len(paths) != 240 or len(hashes) != len(paths):
        raise ValueError("Expected the completed 240-pair expanded batch")
    oracles_cache = {}
    data_cache = {}
    counter = FitCounter()
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        for index, path in enumerate(paths):
            cell = json.loads(path.read_text())
            design = cell["design"]
            target = args.output/f"audit-{fingerprint(design)[:16]}.json"
            if target.exists():
                if json.loads(target.read_text())["run_fingerprint"] != provenance["fingerprint"]:
                    raise ValueError("Diagnostic checkpoint belongs to another run")
                continue
            key = tuple(design[k] for k in ["n", "rho", "snr", "seed"])
            if key not in data_cache:
                data_cache[key] = make_primary(**cell["data"]["params"])
            data = data_cache[key]
            spec = make_spec(config, design["seed"], sim["candidate_k"], "regression")
            splits = list(make_cv(config["outer_folds"], "regression", spec.seed+OUTER_CV_SEED_OFFSET).split(data.X))
            policies = {p: {} for p in spec_config["policies"]}
            diagnostics = []
            for method in ["gated", "ungated"]:
                report = cell["reports"][method]
                per_policy = {p: [] for p in policies}
                for fold, (tr, te) in enumerate(splits):
                    saved = report["folds"][fold]
                    ranking = report["ranking_diagnostics"][fold]["ranking"]
                    selection = saved["k_selection"]
                    ks = {"original_one_se": saved["k_chosen"], "inner_argmin": selection["k_argmin"],
                          "fixed_top5": 5, "fixed_top6": 6}
                    for policy, k in ks.items():
                        result = evaluate_subset(data, spec, tr, te, ranking[:k], fold, counter, saved["outer_loss_all_features"])
                        if policy == "original_one_se" and not np.isclose(result["loss"], saved["outer_loss"], rtol=1e-10, atol=1e-10):
                            raise ValueError("Original fold loss did not reproduce; diagnostic would not be paired")
                        per_policy[policy].append(result)
                    diagnostics.append({"method": method, "fold": fold, "snr": design["snr"],
                                        "chosen": saved["k_chosen"], "argmin": selection["k_argmin"],
                                        "se_at_min": selection["se_at_min"]})
                for policy, methods in policies.items():
                    methods[method] = aggregate_folds(per_policy[policy], sim["noninferiority_margin"], sim["k_max"])
            if key not in oracles_cache:
                oracles_cache[key] = {}
                for k in [3, 5, 6]:
                    allocation = oracle_allocation(k, betas, design["rho"])
                    selected = [2*h+j for h, count in enumerate(allocation) for j in range(count)]
                    folds = [evaluate_subset(data, spec, tr, te, selected, fold, counter,
                                            cell["reports"]["gated"]["folds"][fold]["outer_loss_all_features"])
                             for fold, (tr, te) in enumerate(splits)]
                    oracles_cache[key][str(k)] = aggregate_folds(folds, sim["noninferiority_margin"], sim["k_max"])
            write_json(target, {"run_fingerprint": provenance["fingerprint"], "design": design,
                                "policies": policies, "oracles": oracles_cache[key], "selection_diagnostics": diagnostics})
            if (index+1) % 20 == 0:
                print(f"Diagnosed {index+1}/240; {counter.total} fits; {time.perf_counter()-started:.1f}s", flush=True)
    write_json(args.output/"execution.json", {"additional_fits_this_invocation": counter.total,
                                             "seconds_this_invocation": time.perf_counter()-started})
    summarize(args.output, config)


if __name__ == "__main__":
    main()
