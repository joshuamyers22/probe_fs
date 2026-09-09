"""Validate and compare two frozen selection policies on the same fresh seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_simulation_batch import load_complete, mean_interval

from pgfs.datasets import sha256
from pgfs.study import write_json

KEYS = ("n", "rho", "snr", "contexts", "seed")


def key(cell):
    return tuple(cell["design"][k] for k in KEYS)


def validate_pair(one, arg):
    if one["design"] != arg["design"] or one["data"] != arg["data"]:
        raise ValueError("Policies do not share the same design and generated data parameters")
    for method in ("gated", "ungated"):
        a, b = one["reports"][method], arg["reports"][method]
        if a["spec"]["one_se_rule"] is not True or b["spec"]["one_se_rule"] is not False:
            raise ValueError("Unexpected fitted selection policy")
        if ({k: v for k, v in a["spec"].items() if k != "one_se_rule"} !=
                {k: v for k, v in b["spec"].items() if k != "one_se_rule"}):
            raise ValueError("Method settings differ beyond selection rule")
        if a["ranking_diagnostics"] != b["ranking_diagnostics"]:
            raise ValueError("Policies have different outer ranking estimates")
        if len(a["folds"]) != len(b["folds"]):
            raise ValueError("Policies have different outer folds")
        for x, y in zip(a["folds"], b["folds"]):
            if x["fold"] != y["fold"] or x["outer_loss_all_features"] != y["outer_loss_all_features"]:
                raise ValueError("Policies have different reference evaluations")
            xs, ys = x["k_selection"], y["k_selection"]
            if {k: v for k, v in xs.items() if k != "k_chosen"} != {k: v for k, v in ys.items() if k != "k_chosen"}:
                raise ValueError("Policies have different inner selection paths")
            ks, means = ys["candidate_k"], ys["mean_inner_loss"]
            minimum = ks[int(np.argmin(means))]
            chosen = min(k for k, loss in zip(ks, means) if loss <= min(means)+xs["se_at_min"])
            if y["k_chosen"] != minimum or ys["k_chosen"] != minimum or ys["k_argmin"] != minimum:
                raise ValueError("Argmin did not select the minimizing size")
            if x["k_chosen"] != chosen or xs["k_chosen"] != chosen:
                raise ValueError("One-SE did not apply its specified tolerance")
            if x["selected"] != y["selected"][:chosen]:
                raise ValueError("Selected sets are not prefixes of the shared ranking")


def paired_summary(one_cells, arg_cells):
    one = {key(c): c for c in one_cells}
    arg = {key(c): c for c in arg_cells}
    if len(one) != len(one_cells) or len(arg) != len(arg_cells) or one.keys() != arg.keys():
        raise ValueError("Policies have missing or duplicate paired designs")
    groups = {}
    for k in sorted(one):
        a, b = one[k], arg[k]
        validate_pair(a, b)
        groups.setdefault(k[:-1], []).append((a, b))
    rows = []
    for design, pairs in groups.items():
        row = dict(zip(KEYS[:-1], design))
        row.update(seeds=len(pairs), both_valid=0, recovered=0, lost=0, neither_valid=0)
        metrics = {name: [] for name in ["validity_difference", "common_one_se_recall_effect",
                                        "common_argmin_recall_effect", "common_recall_effect_change"]}
        for method in ("gated", "ungated"):
            for metric in ("loss", "size"):
                metrics[f"{method}_{metric}_change"] = []
        for a, b in pairs:
            av, bv = a["comparison"]["comparison_valid"], b["comparison"]["comparison_valid"]
            row["both_valid" if av and bv else "recovered" if bv else "lost" if av else "neither_valid"] += 1
            metrics["validity_difference"].append(int(bv)-int(av))
            if av and bv:
                ac, bc = a["comparison"]["difference"], b["comparison"]["difference"]
                metrics["common_one_se_recall_effect"].append(ac)
                metrics["common_argmin_recall_effect"].append(bc)
                metrics["common_recall_effect_change"].append(bc-ac)
            for method in ("gated", "ungated"):
                ar = next(r for r in a["rows"] if r["method"] == method)
                br = next(r for r in b["rows"] if r["method"] == method)
                metrics[f"{method}_loss_change"].append(br["mean_outer_loss"]-ar["mean_outer_loss"])
                metrics[f"{method}_size_change"].append(br["mean_selected_size"]-ar["mean_selected_size"])
        row["one_se_valid"] = row["both_valid"]+row["lost"]
        row["argmin_valid"] = row["both_valid"]+row["recovered"]
        for name, values in metrics.items():
            row.update({f"{name}_{stat}": value for stat, value in mean_interval(values).items()})
        rows.append(row)
    return rows


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = sorted({(r["rho"], r["snr"]) for r in rows})
    fig, axes = plt.subplots(len(regimes), 2, figsize=(11, 3*len(regimes)), squeeze=False)
    for (rho, snr), axis_pair in zip(regimes, axes):
        for j, contexts in enumerate(sorted({r["contexts"] for r in rows})):
            points = sorted([r for r in rows if (r["rho"], r["snr"], r["contexts"]) == (rho, snr, contexts)], key=lambda r: r["n"])
            for axis, metric in zip(axis_pair, ["validity_difference", "common_recall_effect_change"]):
                x = [r["n"] for r in points]
                y = [r[metric+"_mean"] if r[metric+"_mean"] is not None else np.nan for r in points]
                axis.plot(x, y, "o-", label=f"Gated T={contexts}", color=f"C{j}")
                for r in points:
                    if r[metric+"_low"] is not None:
                        axis.vlines(r["n"], r[metric+"_low"], r[metric+"_high"], color=f"C{j}", alpha=.6)
        for axis, label in zip(axis_pair, ["Change in valid fraction", "Change in gating recall effect\n(common valid seeds only)"]):
            axis.set(xlabel="Sample size", ylabel=label, title=f"rho={rho}, SNR={snr}")
            axis.axhline(0, color="grey", linewidth=.7)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("Argmin minus one-SE: paired selection-rule sensitivity\nDescriptive 95% t intervals; missing estimates are not zeros")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "selection-rules.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/selection-rule-v1"))
    args = parser.parse_args()
    one, oc, op = load_complete(args.output / "one-se")
    arg, ac, ap = load_complete(args.output / "argmin")
    if oc["one_se_rule"] is not True or ac["one_se_rule"] is not False:
        raise ValueError("Expected one-SE and argmin configurations")
    if {k: v for k, v in oc.items() if k not in ("name", "one_se_rule")} != {k: v for k, v in ac.items() if k not in ("name", "one_se_rule")}:
        raise ValueError("Policy configurations differ beyond name and selection rule")
    identities = [json.loads((args.output / policy / "provenance.json").read_text())["identity"] for policy in ("one-se", "argmin")]
    if {k: v for k, v in identities[0].items() if k != "config"} != {k: v for k, v in identities[1].items() if k != "config"}:
        raise ValueError("Policies used different source, dependencies or dataset manifests")
    rows = paired_summary(one, arg)
    write_json(args.output / "policy-summary.json", rows)
    with (args.output / "policy-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = {k: sum(r[k] for r in rows) for k in ["seeds", "both_valid", "recovered", "lost", "neither_valid", "one_se_valid", "argmin_valid"]}
    totals["paired_designs"] = totals.pop("seeds")
    write_json(args.output / "policy-counts.json", totals)
    lines = ["# Paired selection-rule sensitivity", "", json.dumps(totals, sort_keys=True), "",
             "Changes are argmin minus one-SE. Recall-effect changes compare gated-minus-ungated effects on the common valid seeds only. All-seed validity, loss and size changes retain failed endpoints. Intervals are descriptive 95% t intervals, not confirmatory decisions.", "",
             "| n | rho | SNR | T | One-SE valid | Argmin valid | Recovered / lost | Common valid | Common recall-effect change |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        m, lo, hi = [r["common_recall_effect_change_"+s] for s in ("mean", "low", "high")]
        effect = "N/A" if m is None else f"{m:+.4f}"
        effect += f" [{lo:+.4f}, {hi:+.4f}]" if lo is not None else " [interval unavailable]"
        lines.append(f"| {r['n']} | {r['rho']} | {r['snr']} | {r['contexts']} | {r['one_se_valid']}/{r['seeds']} | {r['argmin_valid']}/{r['seeds']} | {r['recovered']} / {r['lost']} | {r['both_valid']} | {effect} |")
    (args.output / "POLICY_REPORT.md").write_text("\n".join(lines)+"\n")
    plots(rows, args.output)
    write_json(args.output / "policy-analysis-provenance.json", {
        "analysis_sha256": sha256(Path(__file__)),
        "batch_analysis_sha256": sha256(Path(__file__).with_name("analyze_simulation_batch.py")),
        "cell_sha256": {str(p.relative_to(args.output)): sha256(p) for p in op+ap},
        "configurations": {"one-se": oc, "argmin": ac},
    })
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
