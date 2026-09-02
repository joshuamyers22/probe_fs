"""Nested evaluation (Spec Section 9) and the Section 17 algorithm, executed.

For each outer fold ``o``:

1. hold out the outer test fold;
2. estimate the gated ranking using only the outer-training data;
3. choose ``k_o`` by inner cross-validation;
4. re-estimate the ranking on all outer-training data under the frozen spec;
5. fit the learner on the top ``k_o`` players;
6. evaluate once on the outer test fold.

Two guardrails from Section 9 are enforced in code rather than left to discipline:

* The outer test fold is touched exactly once per fold, by step 6 (plus the
  all-features reference fit that Section 11's noninferiority constraint requires,
  which is scored on the same untouched fold in the same single pass).
* Per-fold selected sets are returned for stability reporting but are never
  combined into a consensus set and re-scored on these folds. Building a
  deployment set is a separate function, :func:`finalize_for_deployment`, which
  *requires* a genuinely untouched external test set and raises without one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .config import MethodSpec
from .importance import ImportanceResult, estimate_importance
from .learners import FitCounter, evaluate_player_set
from .metrics import (
    SignalClasses,
    primary_endpoint,
)
from .nested_models import NestedResult, OuterFoldResult
from .players import Players, derive_seed
from .selection import choose_k_by_inner_cv, make_cv

__all__ = ["NestedResult", "OuterFoldResult", "finalize_for_deployment", "nested_evaluate"]

OUTER_CV_SEED_OFFSET = 104_729
FINAL_FIT_KEY = 22
DEPLOYMENT_KEY = 9_999


def _aggregate_endpoint(
    folds: Sequence[OuterFoldResult],
    signal_classes: SignalClasses,
    noninferiority_margin: float,
    k_max: int,
) -> dict[str, Any]:
    """Aggregate fold results under the prespecified endpoint constraints."""
    per_fold = [
        primary_endpoint(
            fold.selected,
            signal_classes,
            fold.outer_loss,
            fold.outer_loss_all_features,
            noninferiority_margin,
            k_max,
        )
        for fold in folds
    ]
    mean_loss = float(np.mean([fold.outer_loss for fold in folds]))
    mean_reference_loss = float(np.mean([fold.outer_loss_all_features for fold in folds]))
    class_recall = float(np.mean([endpoint["class_recall"] for endpoint in per_fold]))
    loss_constraint_met = mean_loss <= mean_reference_loss + noninferiority_margin
    size_constraint_met = all(len(fold.selected) <= k_max for fold in folds)
    constraints_met = bool(loss_constraint_met and size_constraint_met)
    return {
        "mean_class_recall": class_recall,
        "class_recall": class_recall,
        "mean_outer_loss": mean_loss,
        "mean_outer_loss_all_features": mean_reference_loss,
        "loss_constraint_met": bool(loss_constraint_met),
        "size_constraint_met": bool(size_constraint_met),
        "noninferiority_margin": float(noninferiority_margin),
        "k_max": int(k_max),
        "per_fold": per_fold,
        "constraints_met": constraints_met,
        "endpoint": class_recall if constraints_met else float("nan"),
    }


def nested_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    players: Players,
    spec: MethodSpec,
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    signal_classes: SignalClasses | None = None,
    noninferiority_margin: float | None = None,
    k_max: int | None = None,
    counter: FitCounter | None = None,
) -> NestedResult:
    """Run the Section 9 nested evaluation for one frozen specification.

    ``noninferiority_margin`` (``delta``) and ``k_max`` are the Section 11
    constraints; both must be fixed before the study, and the primary endpoint is
    only computed when both are supplied.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    counter = counter if counter is not None else FitCounter()
    loss = spec.loss_fn

    outer_cv = make_cv(n_outer_folds, spec.task, seed=int(spec.seed) + OUTER_CV_SEED_OFFSET)
    strat = y if spec.task != "regression" else None

    folds: list[OuterFoldResult] = []
    for o, (tr, te) in enumerate(outer_cv.split(X, strat)):
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]

        # Steps 2-3: choose k_o using outer-training data only. The ranking is
        # re-estimated inside each inner fold from that fold's training rows.
        ksel = choose_k_by_inner_cv(
            X_tr, y_tr, players, spec,
            n_inner_folds=n_inner_folds, key=(o,), counter=counter,
        )

        # Step 4: re-estimate the ranking on all outer-training data.
        imp: ImportanceResult = estimate_importance(
            X_tr, y_tr, players, spec, key=(o, 21), counter=counter
        )
        selected = imp.top_k(ksel.k_chosen)

        # Steps 5-6: one fit on the selected players, one evaluation on the
        # untouched outer test fold.
        final_seed = derive_seed(spec.seed, o, FINAL_FIT_KEY)
        cols = players.select_columns(selected)
        outer_loss = evaluate_player_set(
            spec.learner, loss,
            X_tr[:, cols], y_tr, X_te[:, cols], y_te,
            seed=final_seed, counter=counter, kind="selection",
        )

        # Reference model on all players, for Section 11's noninferiority
        # constraint. Scored on the same fold in the same single pass.
        all_cols = players.select_columns(range(len(players)))
        outer_loss_all = evaluate_player_set(
            spec.learner, loss,
            X_tr[:, all_cols], y_tr, X_te[:, all_cols], y_te,
            seed=final_seed, counter=counter, kind="selection",
        )

        diag = imp.diagnostics()
        folds.append(
            OuterFoldResult(
                fold=o,
                k_chosen=ksel.k_chosen,
                selected=selected,
                selected_names=tuple(players.names[j] for j in selected),
                outer_loss=outer_loss,
                outer_loss_all_features=outer_loss_all,
                k_selection=ksel,
                ranking=imp.ranking,
                gated_score=imp.gated_score,
                raw_score=imp.raw_score,
                context_sd=diag["context_sd"],
                split_sd=diag["split_sd"],
            )
        )

    result = NestedResult(
        spec=spec.as_dict(),
        folds=folds,
        players=players,
        counter=counter,
        signal_classes=signal_classes,
    )

    if signal_classes is not None and noninferiority_margin is not None and k_max is not None:
        result.endpoint = _aggregate_endpoint(
            folds, signal_classes, noninferiority_margin, k_max
        )

    return result


def finalize_for_deployment(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    players: Players,
    spec: MethodSpec,
    X_external: np.ndarray | None = None,
    y_external: np.ndarray | None = None,
    n_inner_folds: int = 3,
    counter: FitCounter | None = None,
) -> dict[str, Any]:
    """Build and score a single deployment set (Spec Section 9, final paragraph).

    Section 9: if a final consensus set is required for deployment, define it
    after nested evaluation, refit on all development data, and evaluate it only
    on a genuinely untouched external test set. Passing no external set is
    therefore an error rather than a silently unscored result - reporting a
    deployment set's performance on the same outer folds that produced it is the
    exact mistake Section 9 rules out.
    """
    if X_external is None or y_external is None:
        raise ValueError(
            "finalize_for_deployment requires a genuinely untouched external test set; "
            "Section 9 forbids re-evaluating a consensus set on the nested outer folds."
        )
    X_dev = np.asarray(X_dev, dtype=float)
    y_dev = np.asarray(y_dev)
    counter = counter if counter is not None else FitCounter()

    ksel = choose_k_by_inner_cv(
        X_dev, y_dev, players, spec,
        n_inner_folds=n_inner_folds, key=(DEPLOYMENT_KEY,), counter=counter,
    )
    imp = estimate_importance(
        X_dev, y_dev, players, spec, key=(DEPLOYMENT_KEY, 21), counter=counter
    )
    selected = imp.top_k(ksel.k_chosen)

    cols = players.select_columns(selected)
    seed = derive_seed(spec.seed, DEPLOYMENT_KEY, FINAL_FIT_KEY)
    external_loss = evaluate_player_set(
        spec.learner, spec.loss_fn,
        X_dev[:, cols], y_dev,
        np.asarray(X_external, dtype=float)[:, cols], np.asarray(y_external),
        seed=seed, counter=counter, kind="selection",
    )
    return {
        "selected": [players.names[j] for j in selected],
        "selected_indices": list(selected),
        "k_chosen": ksel.k_chosen,
        "external_loss": external_loss,
        "spec": spec.as_dict(),
        **counter.as_dict(),
    }
