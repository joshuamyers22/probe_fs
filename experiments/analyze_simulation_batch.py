"""Validate and summarize a complete expanded simulation batch without refitting."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import t

from pgfs.datasets import sha256
from pgfs.players import derive_rng
from pgfs.study import fingerprint, write_json


def mean_interval(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "mcse": None, "low": None, "high": None}
    mean = float(np.mean(values))
    if len(values) == 1:
        return {"n": 1, "mean": mean, "mcse": None, "low": None, "high": None}
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    width = float(t.ppf(0.975, len(values) - 1)) * se
    return {"n": len(values), "mean": mean, "mcse": se, "low": mean - width, "high": mean + width}


def aggregate(cells: list[dict]) -> list[dict]:
    groups = {}
    for cell in cells:
        d = cell["design"]
        key = tuple(d[k] for k in ("n", "rho", "snr", "contexts"))
        groups.setdefault(key, []).append(cell)
    output = []
    for key, group in sorted(groups.items()):
        if len({c["design"]["seed"] for c in group}) != len(group):
            raise ValueError("Duplicate seed in design group")
        valid = [c for c in group if c["comparison"]["comparison_valid"]]
        row = dict(zip(("n", "rho", "snr", "contexts"), key))
        row.update(
            seeds=len(group), valid=len(valid), invalid=len(group)-len(valid),
            valid_fraction=len(valid)/len(group),
            compute_matched=sum(c["comparison"]["budget_matched"] for c in group),
            valid_clearing_effect=sum(c["comparison"]["meets_minimum_effect"] for c in valid),
            failure_reasons=dict(Counter(c["comparison"]["reason"] for c in group if not c["comparison"]["comparison_valid"])),
        )
        for method in ("gated", "ungated"):
            row[method + "_constraint_passes"] = sum(
                c["reports"][method]["primary_endpoint"]["constraints_met"] for c in group
            )
            row[method + "_loss_failures"] = sum(
                not c["reports"][method]["primary_endpoint"]["loss_constraint_met"] for c in group
            )
            row[method + "_size_failures"] = sum(
                not c["reports"][method]["primary_endpoint"]["size_constraint_met"] for c in group
            )
        paired_loss, paired_size, raw_recall = [], [], []
        for cell in group:
            arms = {r["method"]: r for r in cell["rows"]}
            paired_loss.append(arms["gated"]["mean_outer_loss"] - arms["ungated"]["mean_outer_loss"])
            paired_size.append(arms["gated"]["mean_selected_size"] - arms["ungated"]["mean_selected_size"])
            raw_recall.append(cell["comparison"]["difference"])
        metrics = {
            "qualified_recall_difference": [c["comparison"]["difference"] for c in valid],
            "diagnostic_raw_recall_difference": raw_recall,
            "loss_difference": paired_loss,
            "size_difference": paired_size,
        }
        for name, values in metrics.items():
            row.update({f"{name}_{stat}": value for stat, value in mean_interval(values).items()})
        output.append(row)
    return output


def load_complete(output: Path) -> tuple[list[dict], dict, list[Path]]:
    config = json.loads((output / "frozen-config.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    if (config != provenance["identity"]["config"] or
            fingerprint(provenance["identity"]) != provenance["fingerprint"]):
        raise ValueError("Frozen configuration or batch identity changed")
    sim = config["simulation"]
    expected = set(itertools.product(sim["n"], sim["rho"], sim["snr"], config["seeds"], config["budgets"]))
    paths = sorted(output.glob("seed-*/cell-*.json"))
    cells, found = [], set()
    for path in paths:
        cell = json.loads(path.read_text())
        shard = json.loads((path.parent / "provenance.json").read_text())
        if fingerprint(shard["identity"]) != shard["fingerprint"]:
            raise ValueError(f"Changed shard identity: {path.parent}")
        if cell["run_fingerprint"] != shard["fingerprint"]:
            raise ValueError(f"Mismatched cell provenance: {path}")
        if shard["identity"]["config"]["parent_fingerprint"] != provenance["fingerprint"]:
            raise ValueError(f"Shard belongs to another batch: {path}")
        d = cell["design"]
        key = tuple(d[k] for k in ("n", "rho", "snr", "seed", "contexts"))
        if d["dataset"] != "simulation" or key not in expected or key in found:
            raise ValueError(f"Unexpected or duplicate design: {path}")
        if "betas" in sim and cell["data"]["params"]["betas"] != sim["betas"]:
            raise ValueError(f"Generated class strengths differ from configuration: {path}")
        if sim.get("feature_order") == "seeded_permutation":
            params = cell["data"]["params"]
            p = sim["n_classes"]*sim["class_size"] + sim["n_null_blocks"]*sim["null_block_size"] + sim["n_independent_nulls"]
            if (params.get("feature_order") != "seeded_permutation" or
                    params.get("column_permutation") != derive_rng(d["seed"], 60_491).permutation(p).tolist()):
                raise ValueError(f"Feature order differs from configuration: {path}")
        for report in cell["reports"].values():
            if "one_se_rule" in config and report["spec"]["one_se_rule"] != config["one_se_rule"]:
                raise ValueError(f"Fitted selection rule differs from configuration: {path}")
            if any(f["k_selection"]["candidate_k"] != sim["candidate_k"] for f in report["folds"]):
                raise ValueError(f"Fitted size path differs from configuration: {path}")
        found.add(key)
        cells.append(cell)
    if expected != found:
        raise ValueError(f"Incomplete batch: {len(found)}/{len(expected)} pairs")
    return cells, config, paths


def tie_diagnostics(cells: list[dict]) -> dict:
    """Descriptive audit of exact score ties at the outer selection boundary."""
    output = {}
    for scope, selected in [("all", cells), ("valid", [c for c in cells if c["comparison"]["comparison_valid"]])]:
        output[scope] = {}
        for method in ("gated", "ungated"):
            counts = Counter(folds=0, boundary_ties=0, selected_zero_score=0, selected_size_five=0)
            for cell in selected:
                report = cell["reports"][method]
                diagnostics = {d["fold"]: d for d in report["ranking_diagnostics"]}
                for fold in report["folds"]:
                    d = diagnostics[fold["fold"]]
                    ranking, scores, k = d["ranking"], d["gated_score"], fold["k_chosen"]
                    counts["folds"] += 1
                    counts["boundary_ties"] += int(k < len(ranking) and scores[ranking[k-1]] == scores[ranking[k]])
                    counts["selected_zero_score"] += int(any(scores[j] == 0 for j in ranking[:k]))
                    counts["selected_size_five"] += int(k == 5)
            output[scope][method] = dict(counts)
    return output


def plots(rows: list[dict], output: Path, name: str = "Expanded simulation") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({(r["rho"], r["snr"]) for r in rows})
    fig, axes = plt.subplots(len(regimes), 3, figsize=(15, 3.4 * len(regimes)), squeeze=False)
    for axes_row, (rho, snr) in zip(axes, regimes):
        for contexts in sorted({r["contexts"] for r in rows}):
            points = sorted([r for r in rows if r["rho"] == rho and r["snr"] == snr and r["contexts"] == contexts], key=lambda r: r["n"])
            label = f"Gated T={contexts}"
            axes_row[0].plot([r["n"] for r in points], [r["valid_fraction"] for r in points], "o-", label=label)
            for axis, metric in zip(axes_row[1:], ["qualified_recall_difference", "loss_difference"]):
                for r in points:
                    value = r[metric + "_mean"]
                    if value is None:
                        continue
                    axis.plot(r["n"], value, "o", color=f"C{0 if contexts == 2 else 1}")
                    low, high = r[metric + "_low"], r[metric + "_high"]
                    if low is not None:
                        axis.vlines(r["n"], low, high, color=f"C{0 if contexts == 2 else 1}", alpha=0.6)
                axis.plot([r["n"] for r in points],
                          [np.nan if r[metric + "_mean"] is None else r[metric + "_mean"] for r in points],
                          "-", label=label)
        axes_row[0].set_ylim(-0.05, 1.05)
        axes_row[1].axhline(0.05, linestyle="--", color="black", linewidth=1, label="Minimum effect +0.05")
        for axis in axes_row[1:]:
            axis.axhline(0, color="grey", linewidth=0.7)
        for axis, ylabel in zip(axes_row, ["Valid endpoint fraction", "Recall difference among valid seeds", "MSE difference across all seeds"]):
            axis.set(xlabel="Sample size", ylabel=ylabel, title=f"rho={rho}, SNR={snr}")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
    fig.suptitle(f"{name}: gated minus ungated; descriptive 95% t intervals\nRecall effects condition on valid endpoints; missing estimates are not zeros")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "expanded-simulation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/expanded-simulation-v1"))
    args = parser.parse_args()
    cells, config, paths = load_complete(args.output)
    rows = aggregate(cells)
    write_json(args.output / "ranking-tie-diagnostics.json", tie_diagnostics(cells))
    write_json(args.output / "paired-summary.json", rows)
    with (args.output / "paired-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**r, "failure_reasons": json.dumps(r["failure_reasons"], sort_keys=True)} for r in rows)
    valid = sum(r["valid"] for r in rows)
    matches = sum(r["compute_matched"] for r in rows)
    clears = sum(r["valid_clearing_effect"] for r in rows)
    fits = sum(r["model_fits"] for c in cells for r in c["rows"])
    lines = [f"# {config['name']} results", "",
             f"Completed {len(cells)} paired comparisons and {fits:,} model fits. {matches} compute matches; {valid} valid endpoint comparisons; {len(cells)-valid} inconclusive comparisons. Valid comparisons clearing the fixed +0.05 recall effect: {clears}.", "",
             "Conditional recall differences include only valid paired endpoints. All configured seeds remain in the validity denominator. Intervals are descriptive, conditional 95% t intervals across simulation seeds; they are not multiplicity-adjusted or a confirmatory decision.", "",
             "| n | rho | SNR | T | Valid / total | Valid clearing +0.05 | Conditional recall difference [95% interval] | All-seed MSE difference |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        mean = row["qualified_recall_difference_mean"]
        low, high = row["qualified_recall_difference_low"], row["qualified_recall_difference_high"]
        effect = "N/A" if mean is None else f"{mean:+.4f}"
        if low is not None:
            effect += f" [{low:+.4f}, {high:+.4f}]"
        elif mean is not None:
            effect += " [insufficient valid seeds]"
        lines.append(f"| {row['n']} | {row['rho']} | {row['snr']} | {row['contexts']} | {row['valid']}/{row['seeds']} | {row['valid_clearing_effect']} | {effect} | {row['loss_difference_mean']:+.5f} |")
    lines += ["", "Positive recall differences favor gating; negative loss differences favor gating. The loss diagnostic includes constraint-failing runs and is not a substitute for the qualified endpoint.",
              "No endpoint margin, seed, design or budget was changed during the batch. Full per-arm failure counts, paired uncertainty and raw diagnostics are in paired-summary.csv/json."]
    (args.output / "PAIRED_REPORT.md").write_text("\n".join(lines) + "\n")
    plots(rows, args.output, config["name"])
    write_json(args.output / "analysis-provenance.json", {
        "analysis_sha256": sha256(Path(__file__)), "config": config,
        "cell_sha256": {str(p.relative_to(args.output)): sha256(p) for p in paths},
    })
    print(f"Analyzed {len(cells)} pairs: {valid} valid; {clears} clear +0.05; {fits:,} fits.")


if __name__ == "__main__":
    main()
