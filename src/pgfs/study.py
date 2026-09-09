"""Resumable pilot studies: python -m pgfs.study --config experiments/pilot.json.

Each budget/seed/design cell is checkpointed atomically. A run directory is tied
to the exact configuration, source files, dependency lock, and package versions.
Public-data results never receive simulated ground-truth endpoint labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .budget import budget_ladder
from .config import MethodSpec
from .datasets import load_public, sha256
from .experiment_models import StudyConstraints
from .experiments import matched_compute_comparison
from .metrics import SignalClasses
from .nested import nested_evaluate
from .players import Players, derive_rng
from .simulate import make_primary


def clean(value):
    """Strict JSON, with unavailable quantities represented as null."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True).encode()).hexdigest()


def initialize(output: Path, config: dict, manifest: dict, root: Path) -> dict:
    paths = sorted((root / "src/pgfs").glob("*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    identity = {
        "config": config, "datasets": manifest,
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in paths},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]},
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "provenance.json"
    digest = fingerprint(identity)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["fingerprint"] != digest:
            raise ValueError("Run identity changed; use a new output directory to avoid mixing results")
        return existing
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    provenance = {"fingerprint": digest, "identity": identity, "git_commit": commit,
                  "platform": platform.platform(), "machine": platform.machine(),
                  "blas_threads": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(path, provenance)
    return provenance


def make_spec(config: dict, seed: int, candidates: list[int], task: str) -> MethodSpec:
    one_se = config.get("one_se_rule", True)
    if not isinstance(one_se, bool):
        raise TypeError("one_se_rule must be a boolean")
    model = (Ridge(alpha=1.0) if task == "regression" else
             LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed))
    learner = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), model
    )
    return MethodSpec(
        learner=learner, loss="mse" if task == "regression" else "log_loss",
        n_importance_splits=config["importance_splits"], n_shadows=config["shadows"],
        shadow_quantile=config["quantile"], candidate_k=tuple(candidates), seed=seed,
        one_se_rule=one_se,
    )


def detailed_report(result) -> dict:
    report = result.report()
    report["selection_frequency"] = result.selection_frequency()
    report["variability"] = result.variability()
    report["ranking_diagnostics"] = [
        {"fold": fold.fold, "ranking": fold.ranking,
         "raw_score": fold.raw_score, "gated_score": fold.gated_score,
         "context_sd": fold.context_sd, "split_sd": fold.split_sd}
        for fold in result.folds
    ]
    return report


def make_simulation(sim: dict, design: dict):
    """Construct the configured DGP, including explicitly chosen class strengths."""
    data = make_primary(
        **{k: design[k] for k in ["n", "rho", "snr", "seed"]},
        **{k: sim[k] for k in ["n_classes", "class_size", "n_null_blocks",
                               "null_block_size", "n_independent_nulls"]},
        betas=tuple(sim["betas"]) if "betas" in sim else None,
    )
    order = sim.get("feature_order", "original")
    if order == "seeded_permutation":
        permutation = derive_rng(design["seed"], 60_491).permutation(data.p)
        inverse = np.argsort(permutation)
        data.X = data.X[:, permutation]
        data.players = Players.singletons(data.p, [data.players.names[j] for j in permutation])
        data.signal_classes = SignalClasses.from_lists(
            [[int(inverse[j]) for j in group] for group in data.signal_classes.classes],
            data.signal_classes.names,
        )
        data.params.update(feature_order=order, column_permutation=permutation.tolist())
    elif order != "original":
        raise ValueError(f"Unknown simulation feature order: {order}")
    return data


def public_comparison(data, spec, contexts: int, config: dict) -> tuple[list, dict, dict]:
    pair = budget_ladder(spec, len(data.players), (contexts,))[0]
    if not pair.feasible:
        raise ValueError("Infeasible public-data budget; select a smaller prespecified budget")
    rows, reports = [], {}
    for name, arm in [("gated", pair.gated), ("ungated", pair.ungated)]:
        result = nested_evaluate(
            data.X, data.y, data.players, arm,
            n_outer_folds=config["outer_folds"], n_inner_folds=config["inner_folds"],
        )
        report = detailed_report(result)
        reports[name] = report
        main = report["main_results"]
        rows.append({
            "method": name, "mean_outer_loss": main["mean_outer_loss"],
            "mean_outer_loss_all_features": main["mean_outer_loss_all_features"],
            "mean_selected_size": main["mean_selected_size"],
            "model_fits": main["model_fits_total"],
            "wall_clock_seconds": main["wall_clock_seconds"],
            "mean_pairwise_jaccard": report["stability"]["mean_pairwise_jaccard"],
        })
    fits = [row["model_fits"] for row in rows]
    imbalance = abs(fits[0] - fits[1]) / max(fits)
    comparison = {
        "budget_matched": imbalance <= 0.05, "measured_budget_imbalance": imbalance,
        "predicted_budget_imbalance": pair.imbalance,
        "loss_difference_gated_minus_ungated": rows[0]["mean_outer_loss"] - rows[1]["mean_outer_loss"],
        "primary_endpoint_applicable": False,
    }
    return rows, comparison, reports


def summarize(output: Path) -> None:
    """Describe complete cells; do not turn pilot repetitions into confirmation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [json.loads(path.read_text()) for path in sorted(output.glob("cell-*.json"))]
    rows = [dict(cell["design"], **row) for cell in cells for row in cell["rows"]]
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Keep DGP settings separate; average only repeated seeds of the same cell.
    groups = {}
    for row in rows:
        group = tuple((k, row[k]) for k in ["dataset", "n", "rho", "snr", "contexts", "method"] if k in row)
        groups.setdefault(group, []).append(row)
    aggregates = []
    for key, items in groups.items():
        agg = dict(key, repeats=len(items))
        for metric in ["mean_outer_loss", "mean_selected_size", "class_recall", "model_fits", "wall_clock_seconds"]:
            values = [r[metric] for r in items if r.get(metric) is not None]
            if values:
                agg[metric] = float(np.mean(values))
                agg[metric + "_sd_across_seeds"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregates.append(agg)
    write_json(output / "aggregates.json", aggregates)
    lines = ["# Experiment pilot results", "", "Exploratory feasibility runs; no confirmatory verdict or study stopping decision.", "",
             "| Dataset/design | Contexts | Seed | Loss gated / ungated | Recall difference | Budget matched |",
             "| --- | --- | --- | --- | --- | --- |"]
    for cell in cells:
        d, c = cell["design"], cell["comparison"]
        label = d["dataset"]
        if label == "simulation":
            label += f" n={d['n']} rho={d['rho']} snr={d['snr']}"
        loss = " / ".join(f"{r['mean_outer_loss']:.5g}" for r in cell["rows"])
        recall = c.get("difference")
        lines.append(f"| {label} | {d['contexts']} | {d['seed']} | {loss} | {recall if recall is not None else 'N/A'} | {c['budget_matched']} |")
    lines += ["", "See cell JSON for constraint validity, per-fold selections, rankings, split/context variability, and exact row IDs.",
              "Public data have no known signal classes; recall/false-selection endpoints are not computed.",
              "One importance split cannot estimate variability between importance splits.",
              "Year Prediction results use development rows only; its official test set remains untouched."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    panels = {}
    for row in aggregates:
        key = tuple((k, row[k]) for k in ["dataset", "n", "rho", "snr"] if k in row)
        panels.setdefault(key, []).append(row)
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 3 * len(panels)), squeeze=False)
    for axis_pair, (key, items) in zip(axes, panels.items()):
        for axis, metric in zip(axis_pair, ["mean_outer_loss", "mean_selected_size", "class_recall"]):
            if metric == "class_recall" and items[0]["dataset"] != "simulation":
                axis.text(0.5, 0.5, "Class recall unavailable\n(no ground-truth signal classes)", ha="center", va="center")
                axis.set_axis_off()
                continue
            for method in ["gated", "ungated"]:
                points = sorted([r for r in items if r["method"] == method], key=lambda r: r["model_fits"])
                axis.plot([r["model_fits"] for r in points], [r[metric] for r in points], "o-", label=method)
            axis.set(title=", ".join(f"{k}={v}" for k, v in key), xlabel="Measured model fits", ylabel=metric)
            axis.legend()
            axis.grid(alpha=0.2)
    fig.suptitle("Exploratory pilot: means across seeds within each design")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output / "performance-vs-compute.png", dpi=130)
    plt.close(fig)


def run(config_path: Path, manifest_path: Path, cache: Path, output: Path, only: str) -> None:
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    root = Path(__file__).resolve().parents[2]
    provenance = initialize(output, config, manifest, root)
    jobs = []
    sim = config["simulation"]
    if only in ("all", "simulation"):
        for n, rho, snr, seed, contexts in itertools.product(
            sim["n"], sim["rho"], sim["snr"], config["seeds"], config["budgets"]
        ):
            jobs.append({"dataset": "simulation", "n": n, "rho": rho, "snr": snr, "seed": seed, "contexts": contexts})
    public = config["public"]
    if only in ("all", "public"):
        for name, seed, contexts in itertools.product(public["datasets"], public["seeds"], public["budgets"]):
            jobs.append({"dataset": name, "n": public["rows"], "seed": seed, "contexts": contexts})
    loaded = {}
    for index, design in enumerate(jobs):
        path = output / f"cell-{fingerprint(design)[:16]}.json"
        if path.exists():
            cell = json.loads(path.read_text())
            if cell["run_fingerprint"] != provenance["fingerprint"]:
                raise ValueError(f"Checkpoint from a different run: {path}")
            print(f"[{index + 1}/{len(jobs)}] resumed {design}", flush=True)
            continue
        print(f"[{index + 1}/{len(jobs)}] running {design}", flush=True)
        started = time.perf_counter()
        if design["dataset"] == "simulation":
            data = make_simulation(sim, design)
            spec = make_spec(config, design["seed"], sim["candidate_k"], "regression")
            constraints = StudyConstraints(**{k: sim[k] for k in ["noninferiority_margin", "k_max", "min_effect"]})
            table = matched_compute_comparison(
                data, spec, constraints, (design["contexts"],),
                config["outer_folds"], config["inner_folds"],
            )
            rows, comparison = table.rows, table.comparisons[0]
            reports = {name: detailed_report(result) for (name, _), result in table.results.items()}
            data_info = {"params": data.params}
        else:
            name = design["dataset"]
            key = (name, design["seed"])
            if key not in loaded:
                loaded[key] = load_public(name, manifest[name], cache, design["n"], design["seed"])
            data = loaded[key]
            spec = make_spec(config, design["seed"], public["candidate_k"], data.task)
            rows, comparison, reports = public_comparison(data, spec, design["contexts"], config)
            data_info = {"provenance": data.provenance, "row_indices": data.row_indices}
        cell = {"run_fingerprint": provenance["fingerprint"], "design": design,
                "data": data_info, "rows": rows, "comparison": comparison,
                "reports": reports, "elapsed_seconds": time.perf_counter() - started}
        write_json(path, cell)
        print(f"  saved {path.name}; {cell['elapsed_seconds']:.1f}s; budget matched={comparison['budget_matched']}", flush=True)
    summarize(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/pilot.json"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/datasets.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot-v1"))
    parser.add_argument("--only", choices=["all", "simulation", "public"], default="all")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.manifest, args.cache, args.output, args.only)


if __name__ == "__main__":
    main()
