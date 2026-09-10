"""Replay a frozen training-only size path, without recomputing its rankings."""

from __future__ import annotations

import numpy as np

from pgfs.learners import FitCounter, evaluate_player_set
from pgfs.nested import FINAL_FIT_KEY
from pgfs.players import derive_seed


def policy_sets(fold, diagnostic, p):
    path = fold["k_selection"]
    ks, losses = path["candidate_k"], np.asarray(path["mean_inner_loss"])
    ranking = diagnostic["ranking"]
    if (ks != sorted(set(ks)) or not ks or min(ks) < 1 or max(ks) > p
            or losses.shape != (len(ks),) or not np.isfinite(losses).all()
            or sorted(ranking) != list(range(p)) or diagnostic["fold"] != fold["fold"]
            or not np.isfinite(path["se_at_min"]) or path["se_at_min"] < 0):
        raise ValueError("Invalid frozen selection path or ranking")
    argmin = ks[int(np.argmin(losses))]
    one_se = min(k for k, loss in zip(ks, losses) if loss <= min(losses) + path["se_at_min"])
    if (path["k_argmin"] != argmin or path["k_chosen"] != one_se or fold["k_chosen"] != one_se
            or fold["selected_indices"] != ranking[:one_se]):
        raise ValueError("Recorded selection differs from frozen path")
    return {"one_se": ranking[:one_se], "argmin": ranking[:argmin]}


def replay_fold(X, y, train, test, spec, fold, diagnostic, counter=None):
    sets = policy_sets(fold, diagnostic, X.shape[1])
    counter = FitCounter() if counter is None else counter
    result = {"fold": fold["fold"], "selected": sets, "loss": {}}
    X_train, X_test = X[train], X[test]
    for policy, selected in sets.items():
        # Public players are singleton columns. Match Players.select_columns and
        # nested_evaluate's row-then-column indexing, including column order.
        columns = sorted(selected)
        loss = evaluate_player_set(
            spec.learner, spec.loss_fn, X_train[:, columns], y[train],
            X_test[:, columns], y[test], derive_seed(spec.seed, fold["fold"], FINAL_FIT_KEY),
            counter, "selection",
        )
        if not np.isfinite(loss):
            raise ValueError("Nonfinite replay loss")
        result["loss"][policy] = float(loss)
    return result


def check_loss(actual, expected):
    if not np.isfinite([actual, expected]).all() or not np.isclose(actual, expected, rtol=1e-12, atol=1e-14):
        raise ValueError("Replay differs from recorded loss")
    return actual - expected
