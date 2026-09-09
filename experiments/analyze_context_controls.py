"""Validate and summarize the frozen context-control grid without refitting."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_simulation_batch import mean_interval
from run_context_controls import comparison, preflight, specs

from pgfs.datasets import sha256
from pgfs.study import clean, fingerprint, make_simulation, write_json

DESIGN_KEYS = ("n", "rho", "snr", "seed")


def load_complete(output):
    config = json.loads((output / "frozen-config.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    identity = provenance["identity"]
    if config != identity["config"] or fingerprint(identity) != provenance["fingerprint"]:
        raise ValueError("Frozen batch identity changed")
    if config["driver_sha256"] != sha256(Path(__file__).with_name("run_context_controls.py")):
        raise ValueError("Analyze with the frozen runner source")
    sim = config["simulation"]
    methods = set(specs(config, config["seeds"][0]))
    designs = set(itertools.product(sim["n"], sim["rho"], sim["snr"], config["seeds"]))
    expected = {(*d, m) for d in designs for m in methods}
    paths = sorted(output.glob("seed-*/cell-*.json"))
    found, groups, reconstructed = set(), {}, {}
    shard_identities = {}
    for path in paths:
        cell = json.loads(path.read_text())
        if path.parent not in shard_identities:
            shard = json.loads((path.parent / "provenance.json").read_text())
            if fingerprint(shard["identity"]) != shard["fingerprint"]:
                raise ValueError("Changed shard identity")
            seed = shard["identity"]["config"]["seeds"]
            if len(seed) != 1 or seed[0] not in config["seeds"]:
                raise ValueError("Unexpected shard seed")
            expected_identity = dict(identity, config=dict(config, seeds=seed, parent_fingerprint=provenance["fingerprint"]))
            if shard["identity"] != expected_identity:
                raise ValueError("Shard source or configuration differs from frozen batch")
            shard_identities[path.parent] = shard
        shard = shard_identities[path.parent]
        d, method = cell["design"], cell["method"]
        key = tuple(d[k] for k in DESIGN_KEYS)
        if ((*key, method) not in expected or (*key, method) in found or d["dataset"] != "simulation" or
                cell["run_fingerprint"] != shard["fingerprint"] or d["seed"] != shard["identity"]["config"]["seeds"][0]):
            raise ValueError(f"Unexpected or duplicate checkpoint: {path}")
        spec = specs(config, d["seed"])[method].as_dict()
        report = cell["report"]
        if cell["spec"] != spec or report["spec"] != spec:
            raise ValueError(f"Fitted specification differs from frozen design: {path}")
        if key not in reconstructed:
            data = make_simulation(sim, d)
            reconstructed[key] = clean({"params": data.params, "player_names": data.players.names,
                                        "signal_classes": [sorted(c) for c in data.signal_classes.classes]})
        if cell["data"] != reconstructed[key]:
            raise ValueError(f"Generated parameters, order or class mapping differ: {path}")
        if len(report["folds"]) != config["outer_folds"]:
            raise ValueError("Wrong outer-fold count")
        for fold in report["folds"]:
            selection = fold["k_selection"]
            if selection["candidate_k"] != sim["candidate_k"]:
                raise ValueError("Wrong fitted size path")
            losses = selection["mean_inner_loss"]
            ks = selection["candidate_k"]
            argmin = ks[int(np.argmin(losses))]
            chosen = min(k for k, loss in zip(ks, losses) if loss <= min(losses)+selection["se_at_min"])
            if (selection["k_argmin"] != argmin or fold["k_chosen"] != chosen or
                    selection["k_chosen"] != chosen or len(fold["selected"]) != chosen):
                raise ValueError("One-SE choice differs from recorded loss path")
        found.add((*key, method))
        groups.setdefault(key, {})[method] = cell
    if found != expected:
        raise ValueError(f"Incomplete grid: {len(found)}/{len(expected)} evaluations")
    return groups, config, paths


def validate_shared_fits(cells, config):
    reference = next(iter(cells.values()))["report"]["folds"]
    for cell in cells.values():
        folds = cell["report"]["folds"]
        if [(f["fold"], f["outer_loss_all_features"]) for f in folds] != [(f["fold"], f["outer_loss_all_features"]) for f in reference]:
            raise ValueError("Methods do not share reference evaluations")
    for rung in config["rungs"]:
        t = rung["contexts"]
        a, b = [cells[f"{scope}-T{t}"] for scope in ("local", "global")]
        if a["row"]["model_fits"] != b["row"]["model_fits"]:
            raise ValueError("Local/global fit counts differ")
        for x, y in zip(a["report"]["ranking_diagnostics"], b["report"]["ranking_diagnostics"]):
            if x["fold"] != y["fold"] or x["raw_score"] != y["raw_score"]:
                raise ValueError("Local/global raw outer scores differ")


def aggregate_comparisons(records):
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ("n", "rho", "snr", "contexts", "control"))
        groups.setdefault(key, []).append(record)
    rows = []
    for key, group in sorted(groups.items()):
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate seed in contrast")
        valid = [r for r in group if r["comparison_valid"]]
        row = dict(zip(("n", "rho", "snr", "contexts", "control"), key))
        row.update(seeds=len(group), valid=len(valid), valid_fraction=len(valid)/len(group),
                   compute_matched=sum(r["budget_matched"] for r in group),
                   local_passes=sum(r["candidate_passes"] for r in group),
                   control_passes=sum(r["control_passes"] for r in group),
                   local_wins=sum(r["difference"] > 1e-10 for r in valid),
                   local_losses=sum(r["difference"] < -1e-10 for r in valid),
                   ties=sum(abs(r["difference"]) <= 1e-10 for r in valid),
                   clears_minimum_effect=sum(r["meets_minimum_effect"] for r in valid),
                   max_measured_imbalance=max(r["measured_budget_imbalance"] for r in group),
                   failure_reasons=dict(Counter(r["reason"] for r in group if not r["comparison_valid"])))
        for name, values in {"qualified_recall_difference": [r["difference"] for r in valid],
                             "loss_difference": [r["loss_difference"] for r in group],
                             "size_difference": [r["size_difference"] for r in group],
                             "jaccard_between_methods": [r["jaccard_between_methods"] for r in group]}.items():
            row.update({name+"_"+s: v for s, v in mean_interval(values).items()})
        rows.append(row)
    return rows


def analyze(groups, config):
    predicted = preflight(config)
    comparisons, method_groups = [], {}
    for key, cells in sorted(groups.items()):
        validate_shared_fits(cells, config)
        d = dict(zip(DESIGN_KEYS, key))
        for name, cell in cells.items():
            method_groups.setdefault((*key[:-1], name), []).append(cell)
        reports = {m: c["report"] for m, c in cells.items()}
        for rung in config["rungs"]:
            t = rung["contexts"]
            for control in ("global", "loco-matched"):
                a, b = f"local-T{t}", f"{control}-T{t}"
                cmp = comparison(a, b, reports, predicted, config["simulation"]["min_effect"])
                rows = [cells[m]["row"] for m in (a, b)]
                sets = [(set(x["selected"]), set(y["selected"])) for x, y in zip(reports[a]["folds"], reports[b]["folds"])]
                cmp.update(d, contexts=t, control=control,
                           candidate_passes=rows[0]["constraints_met"], control_passes=rows[1]["constraints_met"],
                           loss_difference=rows[0]["mean_outer_loss"]-rows[1]["mean_outer_loss"],
                           size_difference=rows[0]["mean_selected_size"]-rows[1]["mean_selected_size"],
                           jaccard_between_methods=float(np.mean([len(x & y)/len(x | y) for x, y in sets])))
                comparisons.append(cmp)
    methods = []
    for key, group in sorted(method_groups.items()):
        row = dict(zip(("n", "rho", "snr", "method"), key))
        row.update(seeds=len(group), constraint_passes=sum(c["row"]["constraints_met"] for c in group),
                   loss_failures=sum(not c["report"]["primary_endpoint"]["loss_constraint_met"] for c in group),
                   size_failures=sum(not c["report"]["primary_endpoint"]["size_constraint_met"] for c in group),
                   boundary_ties=0, folds=0)
        for metric in ("mean_outer_loss", "mean_selected_size", "class_recall", "model_fits", "wall_clock_seconds", "mean_pairwise_jaccard"):
            row.update({metric+"_"+s: v for s, v in mean_interval([c["row"][metric] for c in group]).items()})
        for cell in group:
            diagnostics = {r["fold"]: r for r in cell["report"]["ranking_diagnostics"]}
            for fold in cell["report"]["folds"]:
                diag = diagnostics[fold["fold"]]
                k, ranking, score = fold["k_chosen"], diag["ranking"], diag["gated_score"]
                row["folds"] += 1
                row["boundary_ties"] += int(k < len(ranking) and score[ranking[k-1]] == score[ranking[k]])
        methods.append(row)
    return aggregate_comparisons(comparisons), methods, comparisons


def plots(rows, methods, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({(r["rho"], r["snr"]) for r in rows})
    fig, axes = plt.subplots(len(regimes), 2, figsize=(12, 3.3*len(regimes)), squeeze=False)
    for axes_row, (rho, snr) in zip(axes, regimes):
        for color, control in enumerate(("global", "loco-matched")):
            for contexts, style in ((2, "o-"), (4, "x--")):
                points = sorted([r for r in rows if (r["rho"], r["snr"], r["control"], r["contexts"]) == (rho, snr, control, contexts)], key=lambda r: r["n"])
                label = f"{control}, T={contexts}"
                axes_row[0].plot([r["n"] for r in points], [r["valid_fraction"] for r in points], style, color=f"C{color}", label=label)
                metric = "qualified_recall_difference"
                axes_row[1].plot([r["n"] for r in points], [np.nan if r[metric+"_mean"] is None else r[metric+"_mean"] for r in points], style, color=f"C{color}", label=label)
                for r in points:
                    if r[metric+"_low"] is not None:
                        axes_row[1].vlines(r["n"], r[metric+"_low"], r[metric+"_high"], color=f"C{color}", alpha=.5)
        axes_row[0].set_ylim(-.05, 1.05)
        axes_row[1].axhline(0, color="grey", linewidth=.7)
        axes_row[1].axhline(.05, color="black", linestyle=":", linewidth=.7)
        for axis, ylabel in zip(axes_row, ["Matched, valid endpoint fraction", "Local minus control class recall\n(valid matched seeds only)"]):
            axis.set(xlabel="Sample size", ylabel=ylabel, title=f"rho={rho}, SNR={snr}")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("Context controls: primary one-SE rule\nDescriptive 95% t intervals; +0.05 minimum effect; missing estimates are not zeros")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "context-controls.png", dpi=150)
    plt.close(fig)
    sizes = sorted({r["n"] for r in methods})
    fig, axes = plt.subplots(len(sizes), len(regimes), figsize=(16, 3.3*len(sizes)), squeeze=False)
    for axis_row, n in zip(axes, sizes):
        for axis, (rho, snr) in zip(axis_row, regimes):
            subset = [r for r in methods if (r["n"], r["rho"], r["snr"]) == (n, rho, snr)]
            for color, prefix in enumerate(("local-T", "global-T", "loco-matched-T", "loco-gated-B1", "loco-ungated-B1")):
                points = sorted([r for r in subset if r["method"].startswith(prefix)], key=lambda r: r["model_fits_mean"])
                axis.plot([r["model_fits_mean"] for r in points], [r["mean_outer_loss_mean"] for r in points], "o-", color=f"C{color}", label=prefix.rstrip("-T"))
            axis.set(title=f"n={n}, rho={rho}, SNR={snr}", xlabel="Actual model fits", ylabel="Mean outer MSE (all seeds)")
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    fig.suptitle("Performance versus measured cost\nSingle-split LOCO points are cheaper controls, not matched-compute recall evidence")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output / "context-costs.png", dpi=150)
    plt.close(fig)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in r.items()} for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/context-controls-v1"))
    args = parser.parse_args()
    groups, config, paths = load_complete(args.output)
    rows, methods, comparisons = analyze(groups, config)
    for name, data in [("paired-summary", rows), ("method-summary", methods)]:
        write_json(args.output / (name+".json"), data)
        write_csv(args.output / (name+".csv"), data)
    write_json(args.output / "paired-comparisons.json", comparisons)
    fits = sum(c["row"]["model_fits"] for cells in groups.values() for c in cells.values())
    totals = {control: {k: sum(r[k] for r in rows if r["control"] == control)
                       for k in ("seeds", "valid", "compute_matched", "local_wins", "local_losses", "ties", "clears_minimum_effect")}
              for control in ("global", "loco-matched")}
    for total in totals.values():
        total["paired_designs"] = total.pop("seeds")
    write_json(args.output / "summary.json", {"datasets": len(groups), "evaluations": len(paths), "model_fits": fits, "contrasts": totals})
    lines = ["# Context-control contrasts", "", f"{len(groups)} datasets; {len(paths)} nested evaluations; {fits:,} actual fits.", "",
             "Differences are local minus control. Recall includes only valid matched pairs; loss and size retain all seeds. Conditional intervals are descriptive, not confirmatory. Different contrasts can have different valid samples.", "",
             "| n | rho | SNR | T | Control | Valid / total | Qualified recall difference [95% t interval] | All-seed MSE difference |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m, lo, hi = [r["qualified_recall_difference_"+s] for s in ("mean", "low", "high")]
        effect = "N/A" if m is None else f"{m:+.4f}"
        if lo is not None:
            effect += f" [{lo:+.4f}, {hi:+.4f}]"
        elif m is not None:
            effect += " [interval unavailable]"
        lines.append(f"| {r['n']} | {r['rho']} | {r['snr']} | {r['contexts']} | {r['control']} | {r['valid']}/{r['seeds']} | {effect} | {r['loss_difference_mean']:+.5f} |")
    (args.output / "PAIRED_REPORT.md").write_text("\n".join(lines)+"\n")
    plots(rows, methods, args.output)
    write_json(args.output / "analysis-provenance.json", {
        "analysis_sha256": sha256(Path(__file__)), "config": config,
        "runner_sha256": sha256(Path(__file__).with_name("run_context_controls.py")),
        "interval_source_sha256": sha256(Path(__file__).with_name("analyze_simulation_batch.py")),
        "cell_sha256": {str(p.relative_to(args.output)): sha256(p) for p in paths},
    })
    print(json.dumps({"fits": fits, "contrasts": totals}, sort_keys=True))


if __name__ == "__main__":
    main()
