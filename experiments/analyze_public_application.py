"""Validate and summarize every planned application method, including failures."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from run_public_application import checkpoint_valid, methods

from pgfs.datasets import sha256
from pgfs.study import fingerprint, write_json


def plots(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset, n in sorted({(r["dataset"], r["rows"]) for r in rows}):
        group = [r for r in rows if (r["dataset"], r["rows"]) == (dataset, n)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, method in enumerate(sorted({r["method"] for r in group})):
            all_rows = [r for r in group if r["method"] == method]
            complete = [r for r in all_rows if r["status"] == "complete"]
            if not complete:
                continue
            loss = [r["loss"] for r in complete]
            for axis, x in zip(axes, ("selected_size", "model_fits")):
                values = [r[x] for r in complete]
                axis.scatter(values, loss, color=f"C{i}", alpha=.35, s=16)
                axis.scatter([np.mean(values)], [np.mean(loss)], color=f"C{i}", marker="D", s=40,
                             label=f"{method} ({len(complete)}/{len(all_rows)})")
        axes[0].set_xlabel("Mean selected feature count")
        axes[1].set_xlabel("All counted model fits")
        axes[1].set_xscale("log")
        for axis in axes:
            axis.set_ylabel("Log loss" if dataset == "miniboone" else "MSE")
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
        fig.suptitle(f"{dataset}, n={n}: individual seeds and means; labels show completed/planned")
        fig.tight_layout()
        fig.savefig(destination/f"{dataset}-n{n}.png", dpi=150)
        plt.close(fig)


def analyze(output, destination):
    provenance = json.loads((output/"provenance.json").read_text())
    config = json.loads((output/"frozen-config.json").read_text())
    if config != provenance["identity"]["config"] or fingerprint(provenance["identity"]) != provenance["fingerprint"]:
        raise ValueError("Changed application provenance")
    if not config["scientific_evidence"] or config["status"] != "frozen":
        raise ValueError("Calibration is not scientific application evidence; use the cost analyzer")
    rows, contrasts, inputs, found = [], [], {}, set()
    for dataset, n, seed in itertools.product(config["datasets"], config["rows"], config["seeds"]):
        design = {"dataset": dataset, "rows": n, "seed": seed}
        directory = output/f"{dataset}-n{n}-seed{seed}"
        data = json.loads((directory/"data.json").read_text())
        source = provenance["identity"]["datasets"][dataset]
        ids = data["row_indices"]
        if len(ids) != n or len(set(ids)) != n or data["provenance"]["source"] != source:
            raise ValueError("Changed dataset rows or source")
        if dataset == "yearpredictionmsd" and max(ids) >= source["development_rows"]:
            raise ValueError("Official Year test rows found in development results")
        if len(data["folds"]) != config["outer_folds"]:
            raise ValueError("Wrong outer fold count")
        tested = []
        for split in data["folds"]:
            train, test = set(split["train_rows"]), set(split["test_rows"])
            if train & test or train | test != set(ids) or len(split["inner"]) != config["inner_folds"]:
                raise ValueError("Invalid shared outer folds")
            tested.extend(split["test_rows"])
            validated = []
            for inner in split["inner"]:
                tr, va = set(inner["train_rows"]), set(inner["valid_rows"])
                if tr & va or tr | va != train:
                    raise ValueError("Invalid shared inner folds")
                validated.extend(inner["valid_rows"])
            if sorted(validated) != sorted(train):
                raise ValueError("Inner folds do not partition outer training rows")
        if sorted(tested) != sorted(ids):
            raise ValueError("Outer folds do not partition development subset")
        data_sha = sha256(directory/"data.json")
        inputs[str((directory/"data.json").relative_to(output))] = data_sha
        records = {}
        for method, spec in methods(config, source["features"], seed, source["task"]).items():
            path = directory/f"cell-{method}.json"
            record = checkpoint_valid(path, provenance["fingerprint"], design, method, data_sha, spec)
            inputs[str(path.relative_to(output))] = sha256(path)
            found.add(path)
            row = {**design, "method": method, "status": record["status"],
                   "elapsed_seconds": record["elapsed_seconds"], "peak_memory_gib": record["peak_memory_gib"],
                   "loss": None, "selected_size": None, "selected_fraction": None,
                   "model_fits": None, "jaccard": None, "error": record.get("error")}
            if record["status"] == "complete":
                report = record["report"]
                if len(report["folds"]) != config["outer_folds"]:
                    raise ValueError("Wrong method outer fold count")
                main = report["main_results"]
                losses = [fold["outer_loss"] for fold in report["folds"]]
                sizes = [len(fold.get("selected_indices", fold["selected"])) for fold in report["folds"]]
                if (not np.isfinite(losses).all() or not np.isclose(np.mean(losses), main["mean_outer_loss"], rtol=1e-12, atol=1e-14)
                        or not np.isclose(np.mean(sizes), main["mean_selected_size"], rtol=0, atol=1e-14)):
                    raise ValueError("Method aggregates differ from folds")
                row.update(loss=main["mean_outer_loss"], selected_size=main["mean_selected_size"],
                           selected_fraction=main["mean_selected_size"]/source["features"],
                           model_fits=main["model_fits_total"], jaccard=report["stability"]["mean_pairwise_jaccard"])
            elif record["status"] != "failed":
                raise ValueError("Unknown method status")
            rows.append(row)
            records[method] = row
        for t in config["budgets"]:
            local, ungated = records[f"gated-T{t}"], records[f"ungated-T{t}"]
            for control in (f"ungated-T{t}", "all", "l1"):
                reference = records[control]
                complete = local["status"] == reference["status"] == "complete"
                imbalance = abs(local["model_fits"]-ungated["model_fits"])/max(local["model_fits"], ungated["model_fits"]) if local["status"] == ungated["status"] == "complete" else None
                contrasts.append({**design, "contexts": t, "control": control, "complete": complete,
                                  "measured_primary_imbalance": imbalance,
                                  "primary_compute_matched": imbalance <= .05 if imbalance is not None else None,
                                  "loss_difference": local["loss"]-reference["loss"] if complete else None,
                                  "size_difference": local["selected_size"]-reference["selected_size"] if complete else None})
    if set(output.glob("*/cell-*.json")) != found:
        raise ValueError("Unexpected application checkpoints")
    grouped = []
    for dataset, n, t, control in sorted({(r["dataset"], r["rows"], r["contexts"], r["control"]) for r in contrasts}):
        group = [r for r in contrasts if (r["dataset"], r["rows"], r["contexts"], r["control"]) == (dataset, n, t, control)]
        summary = {"dataset": dataset, "rows": n, "contexts": t, "control": control,
                   "planned_seeds": len(group), "complete_seeds": sum(r["complete"] for r in group)}
        for metric in ("loss_difference", "size_difference"):
            values = [r[metric] for r in group if r["complete"]]
            summary[metric] = {"values": values, "mean": float(np.mean(values)) if values else None,
                               "sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
                               "min": min(values) if values else None, "max": max(values) if values else None}
        grouped.append(summary)
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in (("methods", rows), ("contrasts", contrasts)):
        write_json(destination/f"{name}.json", records)
        with (destination/f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(destination/"summary.json", grouped)
    write_json(destination/"analysis-provenance.json", {"run_fingerprint": provenance["fingerprint"],
                                                       "input_sha256": inputs, "analyzer_sha256": sha256(Path(__file__))})
    plots(rows, destination)
    print(f"Analyzed {len(rows)} method records, including {sum(r['status']=='failed' for r in rows)} failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Recorded application run")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output, args.report)


if __name__ == "__main__":
    main()
