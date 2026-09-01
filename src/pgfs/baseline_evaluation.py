"""Nested evaluation and reporting for baseline selectors."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MethodSpec
from .learners import FitCounter, evaluate_player_set
from .metrics import SignalClasses, selection_report, stability_report
from .players import Players
from .selection import make_cv

Selector = Callable[[np.ndarray, np.ndarray, Players, MethodSpec, int], tuple]
OUTER_CV_SEED_OFFSET = 104_729


@dataclass
class BaselineResult:
    """Baseline results in the same reporting shape as ``NestedResult``."""

    name: str
    per_fold_sets: list[tuple[int, ...]]
    outer_losses: list[float]
    outer_losses_all: list[float]
    players: Players
    seconds: float
    counter: FitCounter = field(default_factory=FitCounter)
    signal_classes: SignalClasses | None = None

    def report(self) -> dict[str, Any]:
        """Return predictive, selection, stability, and compute metrics."""
        report: dict[str, Any] = {
            "method": self.name,
            "mean_outer_loss": float(np.mean(self.outer_losses)),
            "se_outer_loss": (
                float(
                    np.std(self.outer_losses, ddof=1) / np.sqrt(len(self.outer_losses))
                )
                if len(self.outer_losses) > 1
                else 0.0
            ),
            "mean_outer_loss_all_features": float(np.mean(self.outer_losses_all)),
            "mean_selected_size": float(
                np.mean([len(selected) for selected in self.per_fold_sets])
            ),
            "wall_clock_seconds": round(self.seconds, 4),
            "model_fits_counted": self.counter.total,
            **stability_report(self.per_fold_sets, self.outer_losses),
        }
        if self.signal_classes is not None:
            fold_reports = [
                selection_report(selected, self.signal_classes)
                for selected in self.per_fold_sets
            ]
            report["class_recall"] = float(
                np.mean([fold["class_recall"] for fold in fold_reports])
            )
            report["false_selections"] = float(
                np.mean([fold["n_outside_classes"] for fold in fold_reports])
            )
            report["redundant_selections"] = float(
                np.mean([fold["n_redundant"] for fold in fold_reports])
            )
        return report


def run_baseline_nested(
    X: np.ndarray,
    y: np.ndarray,
    players: Players,
    spec: MethodSpec,
    selector: Selector,
    name: str,
    n_outer_folds: int = 5,
    signal_classes: SignalClasses | None = None,
) -> BaselineResult:
    """Evaluate a selector on the same outer folds as the primary method."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    counter = FitCounter()
    outer_cv = make_cv(
        n_outer_folds, spec.task, seed=int(spec.seed) + OUTER_CV_SEED_OFFSET
    )
    stratification = y if spec.task != "regression" else None
    selections: list[tuple[int, ...]] = []
    losses: list[float] = []
    all_feature_losses: list[float] = []
    started_at = time.perf_counter()

    for fold, (train, test) in enumerate(outer_cv.split(X, stratification)):
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        selected = tuple(
            selector(X_train, y_train, players, spec, int(spec.seed) + fold)
        ) or (0,)
        selections.append(selected)
        selected_columns = players.select_columns(selected)
        losses.append(
            evaluate_player_set(
                spec.learner,
                spec.loss_fn,
                X_train[:, selected_columns],
                y_train,
                X_test[:, selected_columns],
                y_test,
                seed=int(spec.seed) + fold,
                counter=counter,
                kind="selection",
            )
        )
        all_columns = players.select_columns(range(len(players)))
        all_feature_losses.append(
            evaluate_player_set(
                spec.learner,
                spec.loss_fn,
                X_train[:, all_columns],
                y_train,
                X_test[:, all_columns],
                y_test,
                seed=int(spec.seed) + fold,
                counter=counter,
                kind="selection",
            )
        )

    return BaselineResult(
        name=name,
        per_fold_sets=selections,
        outer_losses=losses,
        outer_losses_all=all_feature_losses,
        players=players,
        seconds=time.perf_counter() - started_at,
        counter=counter,
        signal_classes=signal_classes,
    )
