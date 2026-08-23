"""External baselines (Spec Section 12).

Section 12 requires: no feature selection; a learner-native selector; recursive
feature elimination with cross-validation; tuned stability selection; Boruta or a
comparable shadow-feature method; and model-X knockoffs "when their assumptions are
supportable".

Two commitments from Section 12 shape this module. First, "each baseline receives a
reasonable prespecified tuning protocol" - so every selector's tuning grid is an
argument fixed before the run, not something adapted to results. Second, "simpler
baselines are not forced to spend unused budget on irrelevant tuning parameters" -
so a baseline that finishes cheaply is *reported* as cheap rather than padded, and
results go on performance-versus-compute curves where that cheapness is visible.

Knockoffs carry a real precondition. The Gaussian model-X construction here assumes
``X`` is multivariate normal with a covariance that can be estimated well from the
training rows; :func:`select_knockoffs` records that assumption in its returned
metadata rather than letting it pass silently, because Section 12 admits the
baseline only when the assumption is supportable and Section 15 forbids inheriting
knockoff-style FDR claims for the gated method.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, lasso_path
from sklearn.preprocessing import StandardScaler

from .config import MethodSpec
from .learners import FitCounter, evaluate_player_set
from .metrics import SignalClasses, selection_report, stability_report
from .players import Players, derive_rng
from .selection import make_cv

__all__ = [
    "DEFAULT_BASELINES",
    "BaselineResult",
    "Selector",
    "run_baseline_nested",
    "select_all",
    "select_boruta",
    "select_knockoffs",
    "select_learner_native",
    "select_rfecv",
    "select_stability_selection",
]

# A selector maps outer-training data to a set of player indices.
Selector = Callable[[np.ndarray, np.ndarray, Players, MethodSpec, int], tuple]
OUTER_CV_SEED_OFFSET = 104_729
STABILITY_RNG_KEY = 31
BORUTA_RNG_KEY = 32
KNOCKOFF_RNG_KEY = 33


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _players_from_columns(players: Players, selected_cols: Sequence[int]) -> tuple[int, ...]:
    """A block player counts as selected if any of its columns was selected."""
    chosen = {int(column) for column in selected_cols}
    return tuple(j for j in range(len(players)) if chosen.intersection(players.columns[j]))


def _l1_model(task: str, seed: int, alpha: float | None = None):
    if task == "regression":
        return Lasso(alpha=alpha if alpha else 0.05, random_state=seed, max_iter=5000)
    return LogisticRegression(
        penalty="l1", solver="liblinear",
        C=1.0 / max(alpha or 0.05, 1e-6), random_state=seed, max_iter=2000,
    )


def _coef_vector(model) -> np.ndarray:
    coef = np.asarray(getattr(model, "coef_", np.zeros(1)), dtype=float)
    return np.abs(coef).ravel() if coef.ndim == 1 else np.abs(coef).max(axis=0)


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------
def select_all(X, y, players: Players, spec: MethodSpec, seed: int) -> tuple:
    """No feature selection (Section 12). The reference every constraint uses."""
    return tuple(range(len(players)))


def select_learner_native(X, y, players: Players, spec: MethodSpec, seed: int) -> tuple:
    """Regularized regression as a learner-native selector.

    The penalty is chosen by the selector's own cross-validation - its
    prespecified tuning protocol - and any player with a nonzero coefficient on
    at least one of its columns is selected.
    """
    Xs = StandardScaler().fit_transform(X)
    if spec.task == "regression":
        model = LassoCV(cv=3, random_state=seed, max_iter=5000).fit(Xs, y)
    else:
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, random_state=seed, max_iter=2000
        ).fit(Xs, y)
    nz = np.flatnonzero(_coef_vector(model) > 1e-10)
    return _players_from_columns(players, nz) or tuple(range(len(players)))


def select_rfecv(X, y, players: Players, spec: MethodSpec, seed: int, min_features: int = 1) -> tuple:
    """Recursive feature elimination with cross-validation (Section 12)."""
    Xs = StandardScaler().fit_transform(X)
    estimator = _l1_model(spec.task, seed, alpha=0.01)
    scoring = "neg_mean_squared_error" if spec.task == "regression" else "neg_log_loss"
    rfe = RFECV(
        estimator=estimator,
        step=1,
        cv=make_cv(3, spec.task, seed),
        scoring=scoring,
        min_features_to_select=min_features,
    )
    rfe.fit(Xs, y)
    return _players_from_columns(players, np.flatnonzero(rfe.support_)) or (0,)


def select_stability_selection(
    X, y, players: Players, spec: MethodSpec, seed: int,
    n_subsamples: int = 50,
    subsample_fraction: float = 0.5,
    alpha: float = 0.05,
    thresholds: Sequence[float] = (0.6, 0.7, 0.8, 0.9),
) -> tuple:
    """Tuned stability selection (Section 12).

    Selection frequencies come from L1 fits on random half-samples; the frequency
    threshold is tuned over a prespecified grid by cross-validated predictive loss
    on the training data, which is the "tuned" part Section 12 asks for.
    """
    rng = derive_rng(seed, STABILITY_RNG_KEY)
    Xs = StandardScaler().fit_transform(X)
    n = len(y)
    n_sub = max(10, int(subsample_fraction * n))
    freq = np.zeros(X.shape[1], dtype=float)

    for _ in range(n_subsamples):
        idx = rng.choice(n, size=n_sub, replace=False)
        if spec.task != "regression" and len(np.unique(y[idx])) < 2:
            continue
        model = _l1_model(spec.task, seed, alpha=alpha).fit(Xs[idx], y[idx])
        freq[_coef_vector(model) > 1e-10] += 1.0
    freq /= max(1, n_subsamples)

    # Tune the threshold on training data only.
    best, best_loss = None, np.inf
    cv = make_cv(3, spec.task, seed)
    for thr in thresholds:
        cols = np.flatnonzero(freq >= thr)
        if cols.size == 0:
            continue
        losses = []
        for tr, va in cv.split(X, y if spec.task != "regression" else None):
            losses.append(
                evaluate_player_set(
                    spec.learner, spec.loss_fn,
                    X[np.ix_(tr, cols)], y[tr], X[np.ix_(va, cols)], y[va], seed=seed,
                )
            )
        mean = float(np.mean(losses))
        if mean < best_loss:
            best, best_loss = cols, mean
    if best is None:
        best = np.array([int(np.argmax(freq))])
    return _players_from_columns(players, best)


def select_boruta(
    X, y, players: Players, spec: MethodSpec, seed: int,
    n_iter: int = 20, alpha: float = 0.05,
) -> tuple:
    """A Boruta-style shadow-feature baseline (Section 12).

    Each iteration appends a permuted copy of every column and keeps a "hit" for
    any real column whose importance exceeds the maximum shadow importance. Hits
    are tested against Binomial(n_iter, 1/2) with a Bonferroni correction.

    This is the natural comparator for the gated method because it is also a
    shadow method - but note the contrast Section 12 is drawing: Boruta's shadow
    reference is *global* (the maximum over all shadows in that iteration), not
    matched to a conditioning position.
    """
    from scipy.stats import binom
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    rng = derive_rng(seed, BORUTA_RNG_KEY)
    p_cols = X.shape[1]
    hits = np.zeros(p_cols, dtype=int)

    forest_cls = RandomForestRegressor if spec.task == "regression" else RandomForestClassifier
    for it in range(n_iter):
        shadow = np.column_stack([rng.permutation(X[:, c]) for c in range(p_cols)])
        Z = np.hstack([X, shadow])
        model = forest_cls(n_estimators=100, random_state=seed + it, n_jobs=1).fit(Z, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        hits += (imp[:p_cols] > imp[p_cols:].max()).astype(int)

    cutoff = binom.ppf(1.0 - alpha / max(1, p_cols), n_iter, 0.5)
    cols = np.flatnonzero(hits >= cutoff)
    if cols.size == 0:  # fall back to the strongest evidence rather than nothing
        cols = np.array([int(np.argmax(hits))])
    return _players_from_columns(players, cols)


def select_knockoffs(
    X, y, players: Players, spec: MethodSpec, seed: int,
    target_fdr: float = 0.1, return_meta: bool = False,
) -> tuple:
    """Model-X knockoffs with equicorrelated Gaussian construction (Section 12).

    Assumptions, stated because Section 12 admits this baseline only when they are
    supportable: ``X`` is multivariate Gaussian, its covariance is estimated
    accurately enough from the training rows, and the knockoff statistic is
    computed without reusing ``y`` in the construction. Under those conditions the
    knockoff filter controls FDR at ``target_fdr``. None of that transfers to the
    gated method (Section 15).
    """
    rng = derive_rng(seed, KNOCKOFF_RNG_KEY)
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape

    Sigma = np.corrcoef(Xs, rowvar=False)
    Sigma = np.atleast_2d(Sigma) + 1e-6 * np.eye(d)
    eig_min = float(np.linalg.eigvalsh(Sigma).min())
    s = np.full(d, min(1.0, max(2.0 * eig_min - 1e-8, 1e-8)))

    Sigma_inv = np.linalg.pinv(Sigma)
    D = np.diag(s)
    mu_k = Xs - Xs @ Sigma_inv @ D
    V = 2.0 * D - D @ Sigma_inv @ D
    # Nearest PSD square root, guarding the numerically ragged tail.
    w, Q = np.linalg.eigh((V + V.T) / 2.0)
    V_half = Q @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ Q.T
    X_tilde = mu_k + rng.normal(size=(n, d)) @ V_half

    Z = np.hstack([Xs, X_tilde])
    y_num = np.asarray(y, dtype=float)
    alphas, coefs, _ = lasso_path(Z, y_num, n_alphas=60, eps=1e-3)
    entry = np.zeros(2 * d, dtype=float)
    nonzero = np.abs(coefs) > 1e-10
    for j in range(2 * d):
        idx = np.flatnonzero(nonzero[j])
        entry[j] = alphas[idx[0]] if idx.size else 0.0

    W = entry[:d] - entry[d:]
    thresholds = np.sort(np.abs(W[W != 0]))
    T = np.inf
    for t in thresholds:
        num = 1 + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        if num / den <= target_fdr:
            T = t
            break
    cols = np.flatnonzero(W >= T) if np.isfinite(T) else np.array([], dtype=int)
    chosen = _players_from_columns(players, cols)
    if return_meta:
        return chosen, {"threshold": float(T), "min_eigenvalue": eig_min, "target_fdr": target_fdr}
    return chosen


DEFAULT_BASELINES: dict[str, Selector] = {
    "no_selection": select_all,
    "learner_native": select_learner_native,
    "rfecv": select_rfecv,
    "stability_selection": select_stability_selection,
    "boruta": select_boruta,
    "knockoffs": select_knockoffs,
}


# --------------------------------------------------------------------------
# nested runner
# --------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """Baseline results in the same reporting shape as :class:`NestedResult`."""

    name: str
    per_fold_sets: list[tuple[int, ...]]
    outer_losses: list[float]
    outer_losses_all: list[float]
    players: Players
    seconds: float
    counter: FitCounter = field(default_factory=FitCounter)
    signal_classes: SignalClasses | None = None

    def report(self) -> dict[str, Any]:
        rep: dict[str, Any] = {
            "method": self.name,
            "mean_outer_loss": float(np.mean(self.outer_losses)),
            "se_outer_loss": float(
                np.std(self.outer_losses, ddof=1) / np.sqrt(len(self.outer_losses))
            ) if len(self.outer_losses) > 1 else 0.0,
            "mean_outer_loss_all_features": float(np.mean(self.outer_losses_all)),
            "mean_selected_size": float(np.mean([len(s) for s in self.per_fold_sets])),
            "wall_clock_seconds": round(self.seconds, 4),
            "model_fits_counted": self.counter.total,
            **stability_report(self.per_fold_sets, self.outer_losses),
        }
        if self.signal_classes is not None:
            per = [selection_report(s, self.signal_classes) for s in self.per_fold_sets]
            rep["class_recall"] = float(np.mean([r["class_recall"] for r in per]))
            rep["false_selections"] = float(np.mean([r["n_outside_classes"] for r in per]))
            rep["redundant_selections"] = float(np.mean([r["n_redundant"] for r in per]))
        return rep


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
    """Run a baseline on the *same* outer folds as the primary method.

    The outer folds are generated from the same seed and splitter as
    :func:`pgfs.nested.nested_evaluate`, so baseline and method results are paired
    fold by fold rather than merely averaged over comparable designs.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    counter = FitCounter()
    outer_cv = make_cv(
        n_outer_folds, spec.task, seed=int(spec.seed) + OUTER_CV_SEED_OFFSET
    )
    strat = y if spec.task != "regression" else None

    sets: list[tuple[int, ...]] = []
    losses: list[float] = []
    losses_all: list[float] = []
    t0 = time.perf_counter()

    for o, (tr, te) in enumerate(outer_cv.split(X, strat)):
        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        selected = tuple(selector(X_tr, y_tr, players, spec, int(spec.seed) + o))
        if not selected:
            selected = (0,)
        sets.append(selected)

        cols = players.select_columns(selected)
        losses.append(
            evaluate_player_set(
                spec.learner, spec.loss_fn,
                X_tr[:, cols], y_tr, X_te[:, cols], y_te,
                seed=int(spec.seed) + o, counter=counter, kind="selection",
            )
        )
        all_cols = players.select_columns(range(len(players)))
        losses_all.append(
            evaluate_player_set(
                spec.learner, spec.loss_fn,
                X_tr[:, all_cols], y_tr, X_te[:, all_cols], y_te,
                seed=int(spec.seed) + o, counter=counter, kind="selection",
            )
        )

    return BaselineResult(
        name=name,
        per_fold_sets=sets,
        outer_losses=losses,
        outer_losses_all=losses_all,
        players=players,
        seconds=time.perf_counter() - t0,
        counter=counter,
        signal_classes=signal_classes,
    )
