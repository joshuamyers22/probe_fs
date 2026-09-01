"""Study configuration and reporting contracts for experiment workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from .budget import BudgetPair
from .nested import NestedResult


@dataclass(frozen=True)
class StudyConstraints:
    """Prespecified constants fixed before simulation and never tuned."""

    noninferiority_margin: float
    k_max: int
    min_effect: float = 0.05

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.noninferiority_margin)
            or self.noninferiority_margin < 0
        ):
            raise ValueError("noninferiority_margin must be finite and non-negative")
        if (
            not isinstance(self.k_max, Integral)
            or isinstance(self.k_max, bool)
            or self.k_max < 1
        ):
            raise ValueError("k_max must be a positive integer")
        if not np.isfinite(self.min_effect) or not 0 <= self.min_effect <= 1:
            raise ValueError("min_effect must lie in [0, 1]")


@dataclass
class ComparisonTable:
    """Experiment rows keyed by method and computational budget."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    pairs: list[BudgetPair] = field(default_factory=list)
    results: dict[tuple[str, int], NestedResult] = field(default_factory=dict)

    def to_markdown(self, columns: Sequence[str] | None = None) -> str:
        """Render selected result columns as a Markdown table."""
        cols = (
            list(columns)
            if columns
            else [
                "method",
                "budget",
                "model_fits",
                "wall_clock_seconds",
                "mean_outer_loss",
                "class_recall",
                "mean_selected_size",
                "false_selections",
                "mean_pairwise_jaccard",
                "constraints_met",
            ]
        )
        head = "| " + " | ".join(cols) + " |"
        rule = "| " + " | ".join("---" for _ in cols) + " |"
        lines = [head, rule]
        for row in self.rows:
            values = []
            for column in cols:
                value = row.get(column)
                values.append(
                    f"{value:.4g}" if isinstance(value, float) else str(value)
                )
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
