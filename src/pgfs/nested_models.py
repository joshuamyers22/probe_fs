"""Stable result contracts for nested feature-selection evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .learners import FitCounter
from .metrics import (
    SignalClasses,
    selection_frequency,
    selection_report,
    stability_report,
)
from .players import Players
from .selection import KSelection


@dataclass
class OuterFoldResult:
    fold: int
    k_chosen: int
    selected: tuple[int, ...]
    selected_names: tuple[str, ...]
    outer_loss: float
    outer_loss_all_features: float
    k_selection: KSelection
    ranking: tuple[int, ...]
    gated_score: np.ndarray
    raw_score: np.ndarray
    context_sd: np.ndarray
    split_sd: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "k_chosen": self.k_chosen,
            "selected": list(self.selected_names),
            "outer_loss": self.outer_loss,
            "outer_loss_all_features": self.outer_loss_all_features,
            "k_selection": self.k_selection.as_dict(),
        }


@dataclass
class NestedResult:
    """Everything Section 14 asks to be reported, and nothing it forbids."""

    spec: dict[str, Any]
    folds: list[OuterFoldResult]
    players: Players
    counter: FitCounter = field(default_factory=FitCounter)
    signal_classes: SignalClasses | None = None
    endpoint: dict[str, Any] | None = None

    @property
    def outer_losses(self) -> np.ndarray:
        return np.array([fold.outer_loss for fold in self.folds], dtype=float)

    @property
    def outer_losses_all_features(self) -> np.ndarray:
        return np.array(
            [fold.outer_loss_all_features for fold in self.folds], dtype=float
        )

    @property
    def per_fold_sets(self) -> list[tuple[int, ...]]:
        return [fold.selected for fold in self.folds]

    def mean_outer_loss(self) -> float:
        return float(self.outer_losses.mean())

    def se_outer_loss(self) -> float:
        count = len(self.folds)
        return (
            float(self.outer_losses.std(ddof=1) / np.sqrt(count)) if count > 1 else 0.0
        )

    def stability(self) -> dict[str, float]:
        return stability_report(self.per_fold_sets, self.outer_losses)

    def selection_frequency(self) -> dict[str, float]:
        frequencies = selection_frequency(self.per_fold_sets, len(self.players))
        return {
            self.players.names[index]: float(frequencies[index])
            for index in range(len(self.players))
        }

    def variability(self) -> dict[str, np.ndarray]:
        """Keep context and split variability separate."""
        return {
            "context_sd": np.mean([fold.context_sd for fold in self.folds], axis=0),
            "split_sd": np.mean([fold.split_sd for fold in self.folds], axis=0),
        }

    def report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "spec": self.spec,
            "main_results": {
                "mean_outer_loss": self.mean_outer_loss(),
                "se_outer_loss": self.se_outer_loss(),
                "mean_outer_loss_all_features": float(
                    self.outer_losses_all_features.mean()
                ),
                "mean_selected_size": float(
                    np.mean([fold.k_chosen for fold in self.folds])
                ),
                **self.counter.as_dict(),
            },
            "stability": self.stability(),
            "folds": [fold.as_dict() for fold in self.folds],
        }
        if self.signal_classes is not None:
            per_fold = [
                selection_report(fold.selected, self.signal_classes)
                for fold in self.folds
            ]
            report["main_results"]["mean_class_recall"] = float(
                np.mean([item["class_recall"] for item in per_fold])
            )
            report["main_results"]["mean_false_selections"] = float(
                np.mean([item["n_outside_classes"] for item in per_fold])
            )
            report["main_results"]["mean_redundant_selections"] = float(
                np.mean([item["n_redundant"] for item in per_fold])
            )
        if self.endpoint is not None:
            report["primary_endpoint"] = self.endpoint
        return report
