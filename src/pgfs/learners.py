"""Learner adapter, fit accounting, and the empty-player-set base model.

Spec Section 5 requires that the base and augmented models "use the same training
data, validation observations, learner settings, and - where supported - random
seed". :func:`fit_predict` enforces this by pinning ``random_state`` to a value
supplied by the caller, which is held constant across the base fit, the augmented
fit and all shadow fits within one importance split.

Spec Section 10 and 14 require reporting both model-fit count and wall-clock
time; :class:`FitCounter` is the accounting object threaded through every routine
that fits anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone

from .losses import Loss

__all__ = ["FitCounter", "evaluate_player_set", "fit_predict"]

MAX_RANDOM_SEED = 2**31 - 1


@dataclass
class FitCounter:
    """Counts model fits by category and accumulates wall-clock time.

    ``importance`` counts fits spent estimating rankings (base, augmented and
    shadow fits). ``selection`` counts fits spent walking the selection path and
    evaluating chosen sets. ``cache_hits`` records base fits avoided by the
    nested-prefix cache described in :mod:`pgfs.importance`; it is *not* added to
    the budget, since an avoided fit costs nothing.
    """

    importance: int = 0
    shadow: int = 0
    selection: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    _t0: float | None = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return self.importance + self.selection

    def add(self, kind: str, n: int = 1) -> None:
        if kind == "importance":
            self.importance += n
        elif kind == "shadow":
            # Shadow fits are importance fits; `shadow` is a breakdown counter.
            self.importance += n
            self.shadow += n
        elif kind == "selection":
            self.selection += n
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown fit kind {kind!r}")

    def merge(self, other: FitCounter) -> None:
        self.importance += other.importance
        self.shadow += other.shadow
        self.selection += other.selection
        self.cache_hits += other.cache_hits
        self.seconds += other.seconds

    def as_dict(self) -> dict[str, float]:
        return {
            "model_fits_total": self.total,
            "model_fits_importance": self.importance,
            "model_fits_shadow": self.shadow,
            "model_fits_selection": self.selection,
            "base_fits_reused_from_cache": self.cache_hits,
            "wall_clock_seconds": round(self.seconds, 4),
        }


class _ConstantModel:
    """Base model for the empty player set, and for degenerate training targets.

    Section 4 makes the empty conditioning set reachable: the player drawn first
    in a permutation has ``S = {}``. The intercept-only model is the honest
    counterpart of "no players", and it must be trained on ``D_tr`` like any other
    model so that ``Delta`` for a first-position player is a like-for-like
    comparison.
    """

    def __init__(self, task: str) -> None:
        self.task = task
        self.value_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _ConstantModel:
        self.value_ = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.value_, dtype=float)


def _pin_seed(estimator, seed: int):
    """Set ``random_state`` when the estimator exposes it (Section 5)."""
    params = estimator.get_params(deep=False)
    if "random_state" in params:
        estimator.set_params(random_state=int(seed) % MAX_RANDOM_SEED)
    return estimator


def _predict(model, X_va: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(model.predict(X_va), dtype=float)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X_va), dtype=float)
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 1:
            # Single class seen in training: sklearn returns P(that class)=1.
            only = getattr(model, "classes_", np.array([0.0]))[0]
            return np.full(X_va.shape[0], float(only), dtype=float)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X_va), dtype=float)
        return 1.0 / (1.0 + np.exp(-z))
    return np.asarray(model.predict(X_va), dtype=float)


def fit_predict(
    learner,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    task: str,
    seed: int,
) -> np.ndarray:
    """Fit a fresh clone of ``learner`` and predict on the validation block.

    Falls back to the intercept-only model when there are no columns, or when the
    training targets are degenerate for a classifier (single class present).
    """
    if X_tr.shape[1] == 0:
        return _ConstantModel(task).fit(X_tr, y_tr).predict(X_va)
    if task != "regression" and len(np.unique(y_tr)) < 2:
        return _ConstantModel(task).fit(X_tr, y_tr).predict(X_va)

    model = _pin_seed(clone(learner), seed)
    model.fit(X_tr, y_tr)
    return _predict(model, X_va, task)


def evaluate_player_set(
    learner,
    loss: Loss,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    counter: FitCounter | None = None,
    kind: str = "importance",
) -> float:
    """One fit, one evaluation, one entry in the fit budget."""
    t0 = time.perf_counter()
    pred = fit_predict(learner, X_tr, y_tr, X_va, loss.task, seed)
    value = loss(y_va, pred)
    if counter is not None:
        counter.add(kind)
        counter.seconds += time.perf_counter() - t0
    return value
