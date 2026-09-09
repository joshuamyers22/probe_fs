"""Counted L1 selection with preprocessing and tuning confined to training folds."""

from __future__ import annotations

import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pgfs.learners import FitCounter, evaluate_player_set
from pgfs.metrics import stability_report
from pgfs.nested import FINAL_FIT_KEY, OUTER_CV_SEED_OFFSET
from pgfs.players import derive_seed
from pgfs.selection import INNER_CV_SEED_OFFSET, make_cv


def folds(X, y, spec, count, offset):
    return list(make_cv(count, spec.task, spec.seed + offset).split(
        X, y if spec.task != "regression" else None))


def fit_selector(X, y, task, penalty, seed, settings, counter):
    l1_options = ({"l1_ratio": 1.0} if LogisticRegression().get_params()["penalty"] == "deprecated"
                  else {"penalty": "l1"})
    model = (Lasso(alpha=penalty, max_iter=settings["max_iter"], tol=settings["tol"],
                   random_state=seed) if task == "regression" else
             LogisticRegression(C=1.0/penalty, **l1_options, solver="liblinear",
                                max_iter=settings["max_iter"], tol=settings["tol"], random_state=seed))
    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), model)
    started = time.perf_counter()
    counter.add("importance")  # Includes every tuning selector and outer refit.
    pipeline.fit(X, y)
    counter.seconds += time.perf_counter()-started
    weights = np.abs(np.asarray(model.coef_)).reshape(-1)
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite L1 coefficients")
    selected = tuple(np.flatnonzero(weights > settings["coefficient_tolerance"]).tolist())
    # Empty selection is an honest intercept-only prediction, not a forced feature.
    return selected, weights


def tune_l1(X, y, spec, inner_folds, settings, counter):
    penalties = settings["penalties"]
    if (not penalties or any(not np.isfinite(v) or v <= 0 for v in penalties)
            or penalties != sorted(set(penalties), reverse=True)):
        raise ValueError("Penalties must be finite, positive, unique and decreasing")
    losses, selections = [], []
    for f, (train, valid) in enumerate(folds(X, y, spec, inner_folds, INNER_CV_SEED_OFFSET)):
        row, sets = [], []
        seed = derive_seed(spec.seed, 81_001, f)
        for penalty in penalties:
            selected, _ = fit_selector(X[train], y[train], spec.task, penalty, seed, settings, counter)
            row.append(evaluate_player_set(
                spec.learner, spec.loss_fn, X[train][:, selected], y[train],
                X[valid][:, selected], y[valid], seed, counter, "selection"))
            sets.append(selected)
        losses.append(row)
        selections.append(sets)
    losses = np.asarray(losses)
    if not np.isfinite(losses).all():
        raise ValueError("Nonfinite L1 tuning losses")
    # Penalties decrease, so exact mean-loss ties choose the strongest penalty.
    chosen = int(np.argmin(losses.mean(axis=0)))
    return penalties[chosen], {"penalties": penalties, "chosen_penalty": penalties[chosen],
                              "fold_losses": losses.tolist(), "fold_selected": selections}


def evaluate_baseline(X, y, spec, outer_folds, inner_folds, settings, method):
    if method not in ("all", "l1"):
        raise ValueError("Unknown public baseline")
    counter, records = FitCounter(), []
    for f, (train, test) in enumerate(folds(X, y, spec, outer_folds, OUTER_CV_SEED_OFFSET)):
        seed = derive_seed(spec.seed, f, FINAL_FIT_KEY)
        if method == "l1":
            penalty, tuning = tune_l1(X[train], y[train], spec, inner_folds, settings, counter)
            selected, weights = fit_selector(X[train], y[train], spec.task, penalty, seed, settings, counter)
            ranking = np.argsort(-weights, kind="stable").tolist()
        else:
            selected, ranking, tuning = tuple(range(X.shape[1])), list(range(X.shape[1])), None
        loss = evaluate_player_set(spec.learner, spec.loss_fn, X[train][:, selected], y[train],
                                   X[test][:, selected], y[test], seed, counter, "selection")
        if not np.isfinite(loss):
            raise ValueError("Nonfinite baseline outer loss")
        records.append({"fold": f, "selected": selected, "ranking": ranking,
                        "outer_loss": loss, "tuning": tuning})
    losses = [r["outer_loss"] for r in records]
    return {"spec": spec.as_dict(), "folds": records,
            "main_results": {"mean_outer_loss": float(np.mean(losses)),
                             "mean_selected_size": float(np.mean([len(r["selected"]) for r in records])),
                             **counter.as_dict()},
            "stability": stability_report([r["selected"] for r in records], losses)}
