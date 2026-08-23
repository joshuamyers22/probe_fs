"""The selection path and the choice of ``k`` (Spec Section 8).

    A_k = {j_(1), ..., j_(k)}

``k`` is chosen by inner cross-validation on predictive loss, under the
one-standard-error rule: the smallest ``k`` whose mean inner-validation loss lies
within one standard error of the minimum.

Section 8 is emphatic that ``k`` is the *only* selection-path parameter, so this
module exposes no importance threshold, no win-rate cutoff and no
selection-frequency cutoff. The ranking is re-estimated inside every inner fold
from that fold's training data only; ranking once on all of the outer-training
data and then choosing ``k`` against inner folds would leak the held-out inner
validation rows into the ranking and bias the chosen ``k`` downward.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .config import MethodSpec
from .importance import estimate_importance
from .learners import FitCounter, evaluate_player_set
from .players import Players, derive_seed

__all__ = ["KSelection", "choose_k_by_inner_cv", "make_cv", "one_se_choice"]

INNER_CV_SEED_OFFSET = 7_919
SELECTION_FIT_KEY = 12


@dataclass
class KSelection:
    """Result of the inner-CV search over candidate set sizes."""

    candidate_k: tuple[int, ...]
    fold_losses: np.ndarray            # (n_inner_folds, n_k)
    mean_loss: np.ndarray              # (n_k,)
    se_at_min: float
    k_min: int                         # argmin of mean loss
    k_chosen: int                      # after the 1-SE rule (or == k_min)
    fold_rankings: tuple[tuple[int, ...], ...] = ()
    counter: FitCounter = field(default_factory=FitCounter)

    def as_dict(self) -> dict:
        return {
            "candidate_k": list(self.candidate_k),
            "mean_inner_loss": [float(v) for v in self.mean_loss],
            "k_argmin": self.k_min,
            "se_at_min": float(self.se_at_min),
            "k_chosen": self.k_chosen,
        }


def make_cv(n_splits: int, task: str, seed: int):
    """Stratified folds for classification, plain K-fold otherwise."""
    if task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def one_se_choice(
    candidate_k: Iterable[int],
    fold_losses: np.ndarray,
    use_one_se: bool = True,
) -> tuple[int, int, float]:
    """Apply Section 8's one-standard-error rule.

    Returns ``(k_chosen, k_argmin, se_at_min)``. The standard error is taken from
    the spread across inner folds *at the minimising k*, which is the usual 1-SE
    convention; with a single inner fold there is no spread to estimate and the
    rule degenerates to the argmin.
    """
    ks = tuple(candidate_k)
    mean = fold_losses.mean(axis=0)
    i_min = int(np.argmin(mean))
    k_min = ks[i_min]

    n_folds = fold_losses.shape[0]
    if n_folds > 1:
        se = float(np.std(fold_losses[:, i_min], ddof=1) / np.sqrt(n_folds))
    else:
        se = 0.0

    if not use_one_se:
        return k_min, k_min, se

    threshold = mean[i_min] + se
    eligible = [i for i in range(len(ks)) if mean[i] <= threshold]
    i_chosen = min(eligible, key=lambda i: (ks[i], i)) if eligible else i_min
    return ks[i_chosen], k_min, se


def choose_k_by_inner_cv(
    X: np.ndarray,
    y: np.ndarray,
    players: Players,
    spec: MethodSpec,
    n_inner_folds: int = 3,
    key: tuple[int, ...] = (),
    counter: FitCounter | None = None,
) -> KSelection:
    """Inner cross-validation over ``k`` (Section 8; Section 9 step 3).

    Within each inner fold: estimate the gated ranking on the inner-training rows
    only, then fit and score ``A_k`` for every candidate ``k`` on that same fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    counter = counter if counter is not None else FitCounter()
    ks = spec.candidates_within(len(players))
    loss = spec.loss_fn

    cv = make_cv(n_inner_folds, spec.task, seed=int(spec.seed) + INNER_CV_SEED_OFFSET)
    fold_losses: list[list[float]] = []
    fold_rankings: list[tuple[int, ...]] = []

    for f, (tr, va) in enumerate(cv.split(X, y if spec.task != "regression" else None)):
        imp = estimate_importance(
            X[tr], y[tr], players, spec, key=(*key, 11, f), counter=counter
        )
        fold_rankings.append(imp.ranking)

        seed_f = derive_seed(spec.seed, *key, SELECTION_FIT_KEY, f)
        row: list[float] = []
        for k in ks:
            cols = players.select_columns(imp.top_k(k))
            row.append(
                evaluate_player_set(
                    spec.learner, loss,
                    X[np.ix_(tr, cols)], y[tr], X[np.ix_(va, cols)], y[va],
                    seed=seed_f, counter=counter, kind="selection",
                )
            )
        fold_losses.append(row)

    losses = np.asarray(fold_losses, dtype=float)
    k_chosen, k_min, se = one_se_choice(ks, losses, spec.one_se_rule)
    return KSelection(
        candidate_k=ks,
        fold_losses=losses,
        mean_loss=losses.mean(axis=0),
        se_at_min=se,
        k_min=k_min,
        k_chosen=k_chosen,
        fold_rankings=tuple(fold_rankings),
        counter=counter,
    )
