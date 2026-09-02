"""Decision and redundancy policy for completed experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .experiment_models import ComparisonTable, StudyConstraints
from .metrics import jaccard
from .nested_models import NestedResult


def decision_criteria(
    comparison: ComparisonTable,
    constraints: StudyConstraints,
    ablation: ComparisonTable | None = None,
) -> dict[str, Any]:
    """Apply the prespecified Section 16 rules to completed comparisons.

    The result distinguishes discontinuation, adoption of the simpler method,
    exploratory evidence, and confirmatory evidence rather than collapsing the
    study decision into one boolean.
    """
    cmps = comparison.comparisons
    n_budgets = len(cmps)
    # A constraint-breaking arm yields no endpoint comparison. Such a budget
    # cannot support or refute the minimum effect.
    valid = [comparison for comparison in cmps if comparison["comparison_valid"]]
    invalid = [comparison for comparison in cmps if not comparison["comparison_valid"]]
    cleared = [comparison for comparison in valid if comparison["meets_minimum_effect"]]

    verdict: dict[str, Any] = {
        "n_budgets": n_budgets,
        "n_valid_comparisons": len(valid),
        "inconclusive_budgets": {
            comparison["budget"]: comparison["reason"] for comparison in invalid
        },
        "budgets_clearing_min_effect": [comparison["budget"] for comparison in cleared],
        "recall_differences": {
            comparison["budget"]: comparison["difference"] for comparison in cmps
        },
        "min_effect": constraints.min_effect,
    }

    # Discontinue only when valid evidence exists and none of it clears the
    # prespecified minimum effect.
    verdict["discontinue_broad_study"] = bool(valid and not cleared)
    verdict["inconclusive"] = bool(n_budgets > 0 and not valid)
    # An advantage at only some valid budgets is exploratory because it depends
    # on a favorable computational budget.
    verdict["exploratory_only"] = bool(len(valid) > 1 and 0 < len(cleared) < len(valid))
    verdict["confirmatory"] = bool(valid and len(cleared) == len(valid))

    # Prefer full-conditioning LOCO when it matches recall and stays within the
    # predictive noninferiority margin.
    if ablation is not None:
        by_method = {row["method"]: row for row in ablation.rows}
        gated = by_method.get("gated_permutation")
        loco = by_method.get("loco_gated")
        if gated and loco:
            verdict["adopt_full_conditioning"] = bool(
                loco["class_recall"] >= gated["class_recall"]
                and loco["mean_outer_loss"]
                <= gated["mean_outer_loss"] + constraints.noninferiority_margin
            )
            verdict["ablation"] = {
                "gated_permutation_recall": gated["class_recall"],
                "loco_gated_recall": loco["class_recall"],
                "gated_permutation_loss": gated["mean_outer_loss"],
                "loco_gated_loss": loco["mean_outer_loss"],
                "gated_permutation_fits": gated["model_fits"],
                "loco_gated_fits": loco["model_fits"],
            }

    verdict["summary"] = _verdict_sentence(verdict)
    return verdict


def _verdict_sentence(verdict: dict[str, Any]) -> str:
    if verdict.get("inconclusive"):
        return (
            "No budget produced a valid endpoint comparison - at least one arm broke a "
            "Section 11 constraint everywhere ("
            + ", ".join(
                f"C{budget}: {reason}"
                for budget, reason in verdict["inconclusive_budgets"].items()
            )
            + "). Resolve the constraints before applying the Section 16 decision rule."
        )
    if verdict.get("discontinue_broad_study"):
        return (
            "Gating did not clear the minimum effect at any budget: "
            "Section 16 says discontinue the broad methodology study."
        )
    if verdict.get("adopt_full_conditioning"):
        return (
            "Full-conditioning gated LOCO matched or exceeded sampled partial-context "
            "gating: adopt the simpler full-conditioning method as the primary contribution."
        )
    if verdict.get("exploratory_only"):
        return (
            "Gating cleared the minimum effect at some but not all budgets: "
            "treat the advantage as exploratory rather than confirmatory."
        )
    if verdict.get("confirmatory"):
        return "Gating cleared the minimum effect at every budget tested."
    return "No budgets evaluated."


def redundancy_check(
    result_a: NestedResult,
    result_b: NestedResult,
    jaccard_threshold: float = 0.9,
) -> dict[str, Any]:
    """Assess whether two methods select near-duplicate feature sets."""
    overlaps = [
        jaccard(left, right)
        for left, right in zip(result_a.per_fold_sets, result_b.per_fold_sets)
    ]
    mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
    return {
        "mean_fold_jaccard": mean_overlap,
        "near_duplicate": bool(mean_overlap >= jaccard_threshold),
        "loss_advantage": float(
            result_b.mean_outer_loss() - result_a.mean_outer_loss()
        ),
        "stability_advantage": float(
            result_a.stability()["mean_pairwise_jaccard"]
            - result_b.stability()["mean_pairwise_jaccard"]
        ),
        "compute_ratio": float(result_a.counter.total / max(1, result_b.counter.total)),
    }
