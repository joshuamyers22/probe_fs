"""Draw population-risk and post-hoc feasibility diagnostics from saved JSON."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path("results/endpoint-audit-v1")
    risks = json.loads((output/"population-risk.json").read_text())
    summary = json.loads((output/"summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for rho in [0.5, 0.9]:
        points = sorted([r for r in risks if r["rho"] == rho and r["k"] >= 3], key=lambda r: r["k"])
        axes[0].plot([r["k"] for r in points], [r["oracle_excess_mse"] for r in points], "o-", label=f"rho={rho}")
    axes[0].axhline(0.05, color="black", linestyle="--", label="Loss margin 0.05")
    axes[0].axvline(5, color="grey", linestyle=":", label="k=5 absent from original path")
    axes[0].set(xlabel="Number of true signal features", ylabel="Population excess MSE vs all signal features",
                title="Noisy proxies retain complementary information", xticks=[3, 4, 5, 6])
    axes[0].legend(fontsize=8)
    policies = ["original_one_se", "inner_argmin", "fixed_top5", "fixed_top6"]
    labels = ["Original\none-SE", "Inner-CV\nargmin", "Fixed\ntop 5", "Fixed\ntop 6"]
    values = [summary["paired_policies"][p]["valid_pairs"] for p in policies]
    bars = axes[1].bar(labels, values, color=["#666666", "#3478ab", "#3478ab", "#3478ab"])
    axes[1].bar_label(bars, padding=4)
    axes[1].set(ylim=(0, 240), ylabel="Valid paired endpoints out of 240",
                title="Post-hoc rescores of the same saved rankings")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Endpoint audit: theoretical oracle risk and reused-fold diagnostics")
    fig.tight_layout()
    fig.savefig(output/"endpoint-audit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
