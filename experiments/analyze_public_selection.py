"""Validate and summarize paired public size-rule replays without fitting."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from analyze_public_application import analyze as analyze_baseline
from public_selection import check_loss, policy_sets
from run_public_selection import baseline, read, validate_cell

from pgfs.datasets import sha256
from pgfs.metrics import stability_report
from pgfs.study import fingerprint, write_json


def summarize(values):
    return {"values": values, "mean": float(np.mean(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def plots(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(r["dataset"], r["rows"]) for r in rows})
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    for axis, (dataset, n) in zip(axes.flat, groups):
        group = [r for r in rows if (r["dataset"], r["rows"]) == (dataset, n) and r["status"] == "complete"]
        for i, method in enumerate(sorted({r["method"] for r in group})):
            cells = [r for r in group if r["method"] == method]
            x = [np.mean([r[f"{policy}_size"] for r in cells]) for policy in ("one_se", "argmin")]
            y = [np.mean([r[f"{policy}_loss"] for r in cells]) for policy in ("one_se", "argmin")]
            axis.plot(x, y, color=f"C{i}", alpha=.65)
            axis.scatter(x[0], y[0], color=f"C{i}", marker="o", s=45)
            axis.scatter(x[1], y[1], color=f"C{i}", marker="^", s=55, label=f"{method} ({len(cells)}/5)")
        if group:
            for reference, color in (("all", "black"), ("l1", "gray")):
                axis.axhline(np.mean([r[f"{reference}_loss"] for r in group]), color=color,
                             linestyle="--", linewidth=1, label=reference)
        axis.set(title=f"{dataset}, n={n}", xlabel="Mean selected feature count",
                 ylabel="Log loss" if dataset == "miniboone" else "MSE")
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle("Size-rule sensitivity: circles = one-SE; triangles = argmin\nFive-seed means; lower loss is better")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(destination/"public-selection.png", dpi=150)
    plt.close(fig)


def analyze(output, destination):
    source = baseline(output)
    config, provenance = read(output/"frozen-config.json"), read(output/"provenance.json")
    if (config != provenance["identity"]["config"] or config["status"] != "frozen"
            or fingerprint(provenance["identity"]) != provenance["fingerprint"]):
        raise ValueError("Changed replay provenance")
    destination.mkdir(parents=True, exist_ok=True)
    analyze_baseline(source, destination/"baseline")
    original_config = read(source/"frozen-config.json")
    original_methods = {(r["dataset"], r["rows"], r["seed"], r["method"]): r
                        for r in read(destination/"baseline/methods.json")}
    rows, found, inputs, fold_rows = [], set(), {}, []
    for dataset, n, seed in itertools.product(original_config["datasets"], original_config["rows"], original_config["seeds"]):
        name = f"{dataset}-n{n}-seed{seed}"
        data_sha = sha256(source/name/"data.json")
        for arm, t in itertools.product(("gated", "ungated"), original_config["budgets"]):
            method = f"{arm}-T{t}"
            original_path = source/name/f"cell-{method}.json"
            original = read(original_path)
            path = output/name/f"cell-{method}.json"
            record = validate_cell(path, provenance["fingerprint"], sha256(original_path))
            found.add(path)
            inputs[str(path.relative_to(output))] = sha256(path)
            if (record["design"] != {"dataset": dataset, "rows": n, "seed": seed}
                    or record["method"] != method or record["data_sha256"] != data_sha):
                raise ValueError("Changed replay design")
            original_row = original_methods[dataset, n, seed, method]
            row = {"dataset": dataset, "rows": n, "seed": seed, "method": method, "status": record["status"],
                   "new_model_fits": record["new_fit_counts"]["model_fits_total"],
                   "original_method_fits": original_row["model_fits"],
                   "elapsed_seconds": record["elapsed_seconds"], "peak_memory_gib": record["peak_memory_gib"],
                   "all_loss": original_methods[dataset, n, seed, "all"]["loss"],
                   "l1_loss": original_methods[dataset, n, seed, "l1"]["loss"],
                   "error": record.get("error")}
            for key in ("one_se_loss", "argmin_loss", "one_se_size", "argmin_size", "one_se_jaccard", "argmin_jaccard",
                        "loss_difference", "size_difference", "jaccard_difference", "changed_folds"):
                row[key] = None
            for policy, control in itertools.product(("one_se", "argmin"), ("all", "l1")):
                row[f"{policy}_minus_{control}"] = None
            if record["status"] == "complete":
                if (len(record["folds"]) != original_config["outer_folds"]
                        or row["new_model_fits"] != 2*original_config["outer_folds"]):
                    raise ValueError("Incomplete replay folds or fit count")
                differences = []
                for replay, old, diagnostic in zip(record["folds"], original["report"]["folds"],
                                                    original["report"]["ranking_diagnostics"], strict=True):
                    sets = policy_sets(old, diagnostic, len(diagnostic["ranking"]))
                    if replay["fold"] != old["fold"] or replay["selected"] != sets:
                        raise ValueError("Replayed selection differs from training path")
                    if not np.isfinite(list(replay["loss"].values())).all():
                        raise ValueError("Nonfinite replay loss")
                    delta = check_loss(replay["loss"]["one_se"], old["outer_loss"])
                    if delta:
                        differences.append({"fold": old["fold"], "difference": delta})
                    all_fold = read(source/name/"cell-all.json")["report"]["folds"][old["fold"]]
                    check_loss(old["outer_loss_all_features"], all_fold["outer_loss"])
                    for policy, selected in sets.items():
                        if len(selected) == len(diagnostic["ranking"]):
                            check_loss(replay["loss"][policy], all_fold["outer_loss"])
                    fold_rows.append({"dataset": dataset, "rows": n, "seed": seed, "method": method,
                                      "fold": old["fold"], "one_se_size": len(sets["one_se"]),
                                      "argmin_size": len(sets["argmin"]), **replay["loss"],
                                      "loss_difference": replay["loss"]["argmin"]-replay["loss"]["one_se"]})
                if differences != record["accepted_float_differences"]:
                    raise ValueError("Changed replay roundoff audit")
                for policy in ("one_se", "argmin"):
                    stats = stability_report([f["selected"][policy] for f in record["folds"]],
                                             [f["loss"][policy] for f in record["folds"]])
                    row.update({f"{policy}_loss": stats["mean_outer_loss"],
                                f"{policy}_size": stats["mean_selected_size"],
                                f"{policy}_jaccard": stats["mean_pairwise_jaccard"]})
                    for control in ("all", "l1"):
                        row[f"{policy}_minus_{control}"] = row[f"{policy}_loss"]-row[f"{control}_loss"]
                for metric in ("loss", "size", "jaccard"):
                    row[f"{metric}_difference"] = row[f"argmin_{metric}"]-row[f"one_se_{metric}"]
                row["changed_folds"] = sum(f["selected"]["one_se"] != f["selected"]["argmin"] for f in record["folds"])
            elif record["status"] != "failed":
                raise ValueError("Unknown replay status")
            rows.append(row)
    if set(output.glob("*/cell-*.json")) != found:
        raise ValueError("Unexpected replay checkpoints")
    summaries = []
    for dataset, n, method in sorted({(r["dataset"], r["rows"], r["method"]) for r in rows}):
        group = [r for r in rows if (r["dataset"], r["rows"], r["method"]) == (dataset, n, method)]
        complete = [r for r in group if r["status"] == "complete"]
        summaries.append({"dataset": dataset, "rows": n, "method": method, "planned": len(group), "complete": len(complete),
                          "lower": sum(r["loss_difference"] < 0 for r in complete),
                          "higher": sum(r["loss_difference"] > 0 for r in complete),
                          "ties": sum(r["loss_difference"] == 0 for r in complete),
                          **{metric: summarize([r[metric] for r in complete]) for metric in
                             ("one_se_loss", "argmin_loss", "one_se_size", "argmin_size", "one_se_jaccard", "argmin_jaccard",
                              "loss_difference", "size_difference", "jaccard_difference", "argmin_minus_all", "argmin_minus_l1")}})
    lookup = {(r["dataset"], r["rows"], r["seed"], r["method"]): r for r in rows}
    contrasts = []
    for dataset, n, seed, t in itertools.product(original_config["datasets"], original_config["rows"],
                                                original_config["seeds"], original_config["budgets"]):
        gated, ungated = (lookup[dataset, n, seed, f"{arm}-T{t}"] for arm in ("gated", "ungated"))
        complete = gated["status"] == ungated["status"] == "complete"
        entry = {"dataset": dataset, "rows": n, "seed": seed, "contexts": t, "complete": complete}
        for policy in ("one_se", "argmin"):
            entry[f"{policy}_gated_minus_ungated"] = gated[f"{policy}_loss"]-ungated[f"{policy}_loss"] if complete else None
        entry["gating_effect_change"] = (entry["argmin_gated_minus_ungated"]-entry["one_se_gated_minus_ungated"]
                                         if complete else None)
        contrasts.append(entry)
    for name, records in (("methods", rows), ("folds", fold_rows), ("contrasts", contrasts)):
        write_json(destination/f"{name}.json", records)
        with (destination/f"{name}.csv").open("w", newline="") as stream:
            if records:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    write_json(destination/"summary.json", summaries)
    write_json(destination/"analysis-provenance.json", {"input_sha256": inputs, "run_fingerprint": provenance["fingerprint"],
                                                       "analyzer_sha256": sha256(Path(__file__))})
    plots(rows, destination)
    print(f"Analyzed {len(rows)} paired methods: {sum(r['status']=='failed' for r in rows)} failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output, args.report)


if __name__ == "__main__":
    main()
