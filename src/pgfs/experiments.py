"""The primary experiment, the context ablation, and the decision rules.

* Section 10: soft-gated versus ungated ranking at matched computational budgets,
  reported across a ladder of budgets rather than at one point.
* Section 12: the direct alternatives - full-conditioning LOCO with and without
  the same marginal shadow gate, and global shadow comparison without
  conditioning-position matching.
* Section 16: the decision criteria, written as code so that the discontinuation
  rule is applied to the numbers rather than to the narrative around them.

Section 16 says to run the matched-compute gated-versus-ungated experiment
*first*, and to discontinue the broad methodology study if gating fails to clear
the prespecified minimum effect. :func:`decision_criteria` returns that verdict
explicitly, including the "adopt the simpler full-conditioning method" branch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from .budget import BudgetPair, budget_ladder
from .config import MethodSpec
from .metrics import compare_primary_endpoint
from .nested import NestedResult, nested_evaluate
from .simulate import SimData

__all__ = [
    "StudyConstraints",
    "context_ablation",
    "decision_criteria",
    "direct_alternative_specs",
    "matched_compute_comparison",
    "run_spec",
]


@dataclass(frozen=True)
class StudyConstraints:
    """Section 11's prespecified constants. Fixed before simulation, never tuned.

    ``noninferiority_margin`` (``delta``) and ``k_max`` qualify the endpoint;
    ``min_effect`` is the minimum effect size of interest, an absolute increase of
    0.05 in class recall at a matched computational budget.
    """

    noninferiority_margin: float
    k_max: int
    min_effect: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.noninferiority_margin) or self.noninferiority_margin < 0:
            raise ValueError("noninferiority_margin must be finite and non-negative")
        if not isinstance(self.k_max, Integral) or isinstance(self.k_max, bool) or self.k_max < 1:
            raise ValueError("k_max must be a positive integer")
        if not np.isfinite(self.min_effect) or not 0 <= self.min_effect <= 1:
            raise ValueError("min_effect must lie in [0, 1]")


def run_spec(
    data: SimData,
    spec: MethodSpec,
    constraints: StudyConstraints,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> NestedResult:
    """Nested evaluation of one frozen spec on one simulated dataset."""
    return nested_evaluate(
        data.X, data.y, data.players, spec,
        n_outer_folds=n_outer_folds,
        n_inner_folds=n_inner_folds,
        signal_classes=data.signal_classes,
        noninferiority_margin=constraints.noninferiority_margin,
        k_max=constraints.k_max,
    )


def _row(label: str, budget: int, result: NestedResult) -> dict[str, Any]:
    main = result.report()["main_results"]
    stab = result.stability()
    return {
        "method": label,
        "budget": budget,
        "mean_outer_loss": main["mean_outer_loss"],
        "se_outer_loss": main["se_outer_loss"],
        "mean_outer_loss_all_features": main["mean_outer_loss_all_features"],
        "class_recall": main.get("mean_class_recall", float("nan")),
        "mean_selected_size": main["mean_selected_size"],
        "false_selections": main.get("mean_false_selections", float("nan")),
        "redundant_selections": main.get("mean_redundant_selections", float("nan")),
        "model_fits": main["model_fits_total"],
        "wall_clock_seconds": main["wall_clock_seconds"],
        "mean_pairwise_jaccard": stab["mean_pairwise_jaccard"],
        "constraints_met": bool(result.endpoint["constraints_met"]) if result.endpoint else None,
        "endpoint": result.endpoint["endpoint"] if result.endpoint else float("nan"),
    }


@dataclass
class ComparisonTable:
    """Rows keyed by (method, budget) - the shape Section 10's plot needs."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[BudgetPair] = field(default_factory=list)
    results: dict[tuple[str, int], NestedResult] = field(default_factory=dict)

    def to_markdown(self, columns: Sequence[str] | None = None) -> str:
        cols = list(columns) if columns else [
            "method", "budget", "model_fits", "wall_clock_seconds",
            "mean_outer_loss", "class_recall", "mean_selected_size",
            "false_selections", "mean_pairwise_jaccard", "constraints_met",
        ]
        head = "| " + " | ".join(cols) + " |"
        rule = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [head, rule]
        for r in self.rows:
            vals = []
            for c in cols:
                v = r.get(c)
                vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def matched_compute_comparison(
    data: SimData,
    base_spec: MethodSpec,
    constraints: StudyConstraints,
    gated_contexts: tuple[int, ...] = (2, 4, 8),
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    verbose: bool = False,
) -> ComparisonTable:
    """Section 10's decisive comparison, across a ladder of budgets.

    Both arms share outer folds, inner folds, learner, tuning protocol and seed;
    they differ only in how the budget is spent (shadow fits versus extra
    contexts). Predicted and measured fit counts and wall-clock time are both
    reported, since a predicted match that did not materialise is not a match.
    """
    table = ComparisonTable(pairs=budget_ladder(base_spec, len(data.players), gated_contexts))

    for pair in table.pairs:
        if verbose:
            print(pair.describe(), flush=True)
        gated_res = run_spec(data, pair.gated, constraints, n_outer_folds, n_inner_folds)
        ungated_res = run_spec(data, pair.ungated, constraints, n_outer_folds, n_inner_folds)

        table.results[("gated", pair.budget_index)] = gated_res
        table.results[("ungated", pair.budget_index)] = ungated_res

        g_row = _row("gated", pair.budget_index, gated_res)
        u_row = _row("ungated", pair.budget_index, ungated_res)
        g_row["predicted_importance_fits"] = pair.predicted_fits_gated
        u_row["predicted_importance_fits"] = pair.predicted_fits_ungated
        table.rows.extend([g_row, u_row])

        cmp = compare_primary_endpoint(
            gated_res.endpoint, ungated_res.endpoint, constraints.min_effect
        )
        measured_hi = max(g_row["model_fits"], u_row["model_fits"])
        measured_imbalance = (
            abs(g_row["model_fits"] - u_row["model_fits"]) / measured_hi
            if measured_hi else 0.0
        )
        budget_matched = bool(pair.feasible and measured_imbalance <= 0.05)
        endpoint_valid = bool(cmp["comparison_valid"])
        if not budget_matched:
            cmp["comparison_valid"] = False
            cmp["reason"] = "budget_not_matched"
            cmp["meets_minimum_effect"] = False
        cmp.update(
            budget=pair.budget_index,
            fits_gated=g_row["model_fits"],
            fits_ungated=u_row["model_fits"],
            fit_ratio=(g_row["model_fits"] / max(1, u_row["model_fits"])),
            seconds_gated=g_row["wall_clock_seconds"],
            seconds_ungated=u_row["wall_clock_seconds"],
            loss_gated=g_row["mean_outer_loss"],
            loss_ungated=u_row["mean_outer_loss"],
            endpoint_constraints_met=endpoint_valid,
            budget_matched=budget_matched,
            predicted_budget_imbalance=pair.imbalance,
            measured_budget_imbalance=measured_imbalance,
        )
        table.comparisons.append(cmp)

    return table


def direct_alternative_specs(base_spec: MethodSpec) -> dict[str, MethodSpec]:
    """Section 12's direct alternatives, as spec variants of one frozen base.

    Deriving them from the base spec is what keeps the comparison honest: learner,
    loss, folds, candidate sizes and seed are shared by construction, so a
    difference in results cannot be a difference in protocol.
    """
    return {
        "gated_permutation": base_spec.replace(
            gate="soft", context_kind="permutation", label="gated-permutation"
        ),
        "ungated_permutation": base_spec.replace(
            gate="none", context_kind="permutation", label="ungated-permutation"
        ),
        "loco_ungated": base_spec.replace(
            gate="none", context_kind="full", label="LOCO-ungated"
        ),
        "loco_gated": base_spec.replace(
            gate="soft", context_kind="full", label="LOCO-gated"
        ),
        "global_shadow": base_spec.replace(
            gate="soft", context_kind="permutation", shadow_scope="global",
            label="global-shadow",
        ),
    }


def context_ablation(
    data: SimData,
    base_spec: MethodSpec,
    constraints: StudyConstraints,
    which: Sequence[str] | None = None,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    verbose: bool = False,
) -> ComparisonTable:
    """Sampled partial conditioning versus full-conditioning LOCO (Sections 12, 18.6)."""
    specs = direct_alternative_specs(base_spec)
    names = list(which) if which else list(specs)
    table = ComparisonTable()
    for name in names:
        spec = specs[name]
        if verbose:
            print(spec.describe(), flush=True)
        res = run_spec(data, spec, constraints, n_outer_folds, n_inner_folds)
        table.results[(name, 0)] = res
        table.rows.append(_row(name, 0, res))
    return table


def decision_criteria(
    comparison: ComparisonTable,
    constraints: StudyConstraints,
    ablation: ComparisonTable | None = None,
) -> dict[str, Any]:
    """Section 16, applied to the numbers.

    Returns a verdict dictionary rather than a single boolean, because Section 16
    contains four distinct rules with different consequences: discontinue; adopt
    the simpler method; downgrade to exploratory; and check for redundancy against
    an existing baseline.
    """
    cmps = comparison.comparisons
    n_budgets = len(cmps)
    # A budget where either arm broke a Section 11 constraint yields no endpoint
    # value, so it can neither support nor refute the minimum effect. Folding such
    # budgets into "did not clear" would let a comparator's constraint violation
    # count as evidence against gating.
    valid = [c for c in cmps if c["comparison_valid"]]
    invalid = [c for c in cmps if not c["comparison_valid"]]
    cleared = [c for c in valid if c["meets_minimum_effect"]]

    verdict: dict[str, Any] = {
        "n_budgets": n_budgets,
        "n_valid_comparisons": len(valid),
        "inconclusive_budgets": {c["budget"]: c["reason"] for c in invalid},
        "budgets_clearing_min_effect": [c["budget"] for c in cleared],
        "recall_differences": {c["budget"]: c["difference"] for c in cmps},
        "min_effect": constraints.min_effect,
    }

    # Rule 1: discontinue if gating never clears the minimum effect - but only on
    # the evidence of budgets that produced a valid endpoint comparison.
    verdict["discontinue_broad_study"] = bool(valid and not cleared)
    verdict["inconclusive"] = bool(n_budgets > 0 and not valid)

    # Rule 3: an advantage at some but not all valid budgets is exploratory, not
    # confirmatory - the same logic Section 16 applies to a single favourable
    # tuning choice.
    verdict["exploratory_only"] = bool(len(valid) > 1 and 0 < len(cleared) < len(valid))
    verdict["confirmatory"] = bool(valid and len(cleared) == len(valid))

    # Rule 2: prefer the simpler full-conditioning method when it matches or beats
    # sampled partial-context gating.
    if ablation is not None:
        by_method = {r["method"]: r for r in ablation.rows}
        gated = by_method.get("gated_permutation")
        loco = by_method.get("loco_gated")
        if gated and loco:
            loco_wins = bool(
                loco["class_recall"] >= gated["class_recall"]
                and loco["mean_outer_loss"] <= gated["mean_outer_loss"]
                + constraints.noninferiority_margin
            )
            verdict["adopt_full_conditioning"] = loco_wins
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


def _verdict_sentence(v: dict[str, Any]) -> str:
    if v.get("inconclusive"):
        return (
            "No budget produced a valid endpoint comparison - at least one arm broke a "
            "Section 11 constraint everywhere ("
            + ", ".join(f"C{b}: {r}" for b, r in v["inconclusive_budgets"].items())
            + "). Resolve the constraints before applying the Section 16 decision rule."
        )
    if v.get("discontinue_broad_study"):
        return (
            "Gating did not clear the minimum effect at any budget: "
            "Section 16 says discontinue the broad methodology study."
        )
    if v.get("adopt_full_conditioning"):
        return (
            "Full-conditioning gated LOCO matched or exceeded sampled partial-context "
            "gating: adopt the simpler full-conditioning method as the primary contribution."
        )
    if v.get("exploratory_only"):
        return (
            "Gating cleared the minimum effect at some but not all budgets: "
            "treat the advantage as exploratory rather than confirmatory."
        )
    if v.get("confirmatory"):
        return "Gating cleared the minimum effect at every budget tested."
    return "No budgets evaluated."


def redundancy_check(
    result_a: NestedResult, result_b: NestedResult, jaccard_threshold: float = 0.9
) -> dict[str, Any]:
    """Section 16's last rule: is this method just an existing baseline in disguise?

    If the selected sets nearly coincide, the burden shifts to showing an advantage
    in stability, predictive loss, interpretability or computation before claiming
    a separate contribution.
    """
    from .metrics import jaccard

    overlaps = [
        jaccard(a, b) for a, b in zip(result_a.per_fold_sets, result_b.per_fold_sets)
    ]
    mean_overlap = float(np.mean(overlaps)) if overlaps else float("nan")
    return {
        "mean_fold_jaccard": mean_overlap,
        "near_duplicate": bool(mean_overlap >= jaccard_threshold),
        "loss_advantage": float(result_b.mean_outer_loss() - result_a.mean_outer_loss()),
        "stability_advantage": float(
            result_a.stability()["mean_pairwise_jaccard"]
            - result_b.stability()["mean_pairwise_jaccard"]
        ),
        "compute_ratio": float(
            result_a.counter.total / max(1, result_b.counter.total)
        ),
    }
