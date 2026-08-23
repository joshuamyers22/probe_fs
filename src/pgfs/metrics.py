"""Redundancy-aware metrics (Spec Sections 11 and 14).

Class recall (Section 11):

    R_class(A) = (1/g) * sum_h 1{ A intersect C_h != empty }

The primary endpoint is class recall *subject to* two prespecified constraints -
a noninferiority margin on outer-test loss and a cap on set size - and the minimum
effect size of interest is an absolute increase of 0.05 in class recall at a
matched computational budget. :func:`primary_endpoint` returns the constraint
verdicts alongside the recall so that a recall number can never be quoted without
the conditions it was earned under; :func:`compare_primary_endpoint` applies the
0.05 rule.

Section 14 also requires stability to be reported, always paired with predictive
performance and set size, "since selecting nothing is trivially stable" -
:func:`stability_report` therefore refuses to return a bare stability number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "SignalClasses",
    "class_recall",
    "compare_primary_endpoint",
    "jaccard",
    "primary_endpoint",
    "selection_report",
    "stability_report",
]


@dataclass(frozen=True)
class SignalClasses:
    """Ground-truth redundant signal classes ``C_1, ..., C_g`` (Section 11).

    Section 13 requires the primary data-generating process to define these
    explicitly, so they are a property of the simulation, never something inferred
    from a fitted model.
    """

    classes: tuple[frozenset[int], ...]
    names: tuple[str, ...] = ()

    @classmethod
    def from_lists(cls, groups: Iterable[Sequence[int]], names: Sequence[str] | None = None):
        cl = tuple(frozenset(int(i) for i in g) for g in groups)
        nm = tuple(names) if names is not None else tuple(f"C{h + 1}" for h in range(len(cl)))
        if any(not group for group in cl):
            raise ValueError("signal classes must be non-empty")
        if any(i < 0 for group in cl for i in group):
            raise ValueError("signal-class members must be non-negative")
        seen: set[int] = set()
        for group in cl:
            if seen.intersection(group):
                raise ValueError("signal classes must not overlap")
            seen.update(group)
        if len(nm) != len(cl):
            raise ValueError("signal-class names must match the number of classes")
        return cls(cl, nm)

    @property
    def g(self) -> int:
        return len(self.classes)

    @property
    def members(self) -> frozenset[int]:
        return frozenset().union(*self.classes) if self.classes else frozenset()


def class_recall(selected: Iterable[int], classes: SignalClasses) -> float:
    """Fraction of signal classes with at least one selected member."""
    A = set(selected)
    if classes.g == 0:
        return float("nan")
    return float(np.mean([1.0 if A & set(C) else 0.0 for C in classes.classes]))


def selection_report(selected: Iterable[int], classes: SignalClasses) -> dict[str, float]:
    """Section 11's required accompanying counts."""
    A = set(selected)
    hits_per_class = [len(A & set(C)) for C in classes.classes]
    return {
        "class_recall": class_recall(A, classes),
        "n_selected": float(len(A)),
        "n_outside_classes": float(len(A - set(classes.members))),
        "n_redundant": float(sum(max(0, h - 1) for h in hits_per_class)),
        "n_classes_hit": float(sum(1 for h in hits_per_class if h > 0)),
    }


def primary_endpoint(
    selected: Iterable[int],
    classes: SignalClasses,
    outer_loss: float,
    outer_loss_all_features: float,
    delta: float,
    k_max: int,
) -> dict[str, object]:
    """Class recall qualified by the two Section 11 constraints.

    ``endpoint`` is ``nan`` when either constraint fails: a recall figure earned
    by a set that is too large, or that gives up more than ``delta`` of predictive
    performance against the all-features model, is not a value of this endpoint.
    """
    A = set(selected)
    rep = selection_report(A, classes)
    loss_ok = bool(outer_loss <= outer_loss_all_features + delta)
    size_ok = bool(len(A) <= int(k_max))
    both = loss_ok and size_ok
    return {
        **rep,
        "outer_loss": float(outer_loss),
        "outer_loss_all_features": float(outer_loss_all_features),
        "noninferiority_margin": float(delta),
        "k_max": int(k_max),
        "loss_constraint_met": loss_ok,
        "size_constraint_met": size_ok,
        "constraints_met": both,
        "endpoint": rep["class_recall"] if both else float("nan"),
    }


def compare_primary_endpoint(
    endpoint_a: dict[str, object],
    endpoint_b: dict[str, object],
    min_effect: float = 0.05,
) -> dict[str, object]:
    """Apply Section 11's minimum effect size of interest (0.05 class recall).

    ``a`` is the candidate method, ``b`` the comparator; both are assumed to have
    been run at a matched computational budget (Section 10).
    """
    a_ok = bool(endpoint_a["constraints_met"])
    b_ok = bool(endpoint_b["constraints_met"])
    a = float(endpoint_a["class_recall"])
    b = float(endpoint_b["class_recall"])
    diff = a - b

    if a_ok and b_ok:
        reason = "ok"
    elif not a_ok and not b_ok:
        reason = "both_failed_constraints"
    elif not a_ok:
        reason = "candidate_failed_constraints"
    else:
        reason = "comparator_failed_constraints"

    return {
        "class_recall_a": a,
        "class_recall_b": b,
        "difference": diff,
        "min_effect": float(min_effect),
        "both_satisfy_constraints": a_ok and b_ok,
        # The endpoint is defined only under the constraints, so a comparison in
        # which either arm fails them is *inconclusive*, not a negative result.
        # Reporting it as a failed effect would let a comparator's constraint
        # violation masquerade as evidence against the method.
        "comparison_valid": a_ok and b_ok,
        "reason": reason,
        "meets_minimum_effect": bool(a_ok and b_ok and diff >= min_effect),
    }


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def stability_report(
    per_fold_sets: Sequence[Sequence[int]],
    outer_losses: Sequence[float],
) -> dict[str, float]:
    """Stability, always paired with loss and set size (Section 14).

    Selection frequency across folds is reported as a *diagnostic* only; Section 9
    forbids combining per-fold sets into a consensus set and re-evaluating it on
    the same outer folds, and this function never builds such a set.
    """
    sets = [set(s) for s in per_fold_sets]
    pairs = [jaccard(a, b) for a, b in combinations(sets, 2)]
    sizes = [len(s) for s in sets]
    return {
        "mean_pairwise_jaccard": float(np.mean(pairs)) if pairs else float("nan"),
        "mean_selected_size": float(np.mean(sizes)) if sizes else float("nan"),
        "sd_selected_size": float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0,
        "mean_outer_loss": float(np.mean(outer_losses)) if len(outer_losses) else float("nan"),
        "n_folds": float(len(sets)),
    }


def selection_frequency(per_fold_sets: Sequence[Sequence[int]], n_players: int) -> np.ndarray:
    """How often each player was selected across outer folds (diagnostic only)."""
    freq = np.zeros(n_players, dtype=float)
    for s in per_fold_sets:
        for j in s:
            freq[j] += 1.0
    return freq / max(1, len(per_fold_sets))
