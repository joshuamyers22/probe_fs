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
from typing import Any

from .budget import budget_ladder
from .config import MethodSpec
from .experiment_decisions import decision_criteria, redundancy_check
from .experiment_models import ComparisonTable, StudyConstraints
from .metrics import compare_primary_endpoint
from .nested import NestedResult, nested_evaluate
from .simulate import SimData

__all__ = [
    "StudyConstraints",
    "context_ablation",
    "decision_criteria",
    "direct_alternative_specs",
    "matched_compute_comparison",
    "redundancy_check",
    "run_spec",
]


def run_spec(
    data: SimData,
    spec: MethodSpec,
    constraints: StudyConstraints,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
) -> NestedResult:
    """Nested evaluation of one frozen spec on one simulated dataset."""
    return nested_evaluate(
        data.X,
        data.y,
        data.players,
        spec,
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
        "constraints_met": bool(result.endpoint["constraints_met"])
        if result.endpoint
        else None,
        "endpoint": result.endpoint["endpoint"] if result.endpoint else float("nan"),
    }


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
    table = ComparisonTable(
        pairs=budget_ladder(base_spec, len(data.players), gated_contexts)
    )

    for pair in table.pairs:
        if verbose:
            print(pair.describe(), flush=True)
        gated_res = run_spec(
            data, pair.gated, constraints, n_outer_folds, n_inner_folds
        )
        ungated_res = run_spec(
            data, pair.ungated, constraints, n_outer_folds, n_inner_folds
        )

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
            if measured_hi
            else 0.0
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
            gate="soft",
            context_kind="permutation",
            shadow_scope="global",
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
