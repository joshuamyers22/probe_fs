"""Pure ranking, gating, and refit-compatibility policy for importance results."""

from __future__ import annotations

import numpy as np

from .config import MethodSpec
from .shadows import shadow_threshold


def rank_players(score: np.ndarray) -> tuple[int, ...]:
    """Return decreasing score order with exact ties broken by player index."""

    return tuple(
        sorted(range(len(score)), key=lambda player: (-float(score[player]), player))
    )


def fit_signature(spec: MethodSpec) -> tuple[tuple[str, str], ...]:
    """Identify fields that determine splits, contexts, fits, and shadow draws."""

    values = {
        "learner": repr(spec.learner),
        "loss": spec.loss,
        "context_kind": spec.context_kind,
        "n_contexts": spec.effective_n_contexts,
        "n_importance_splits": spec.n_importance_splits,
        "importance_val_fraction": spec.importance_val_fraction,
        # Keep the configured value even for gate="none", so a gated result may
        # be regated to the raw comparator without appearing to change design.
        "n_shadows": spec.n_shadows,
        "seed": spec.seed,
    }
    return tuple(sorted((key, repr(value)) for key, value in values.items()))


def apply_gate(delta: np.ndarray, tau: np.ndarray, gate: str) -> np.ndarray:
    """Apply the configured soft, hard, or disabled importance gate.

    The soft gate is ``max(0, delta - max(0, tau))``. A negative shadow
    threshold never credits a player above its raw contribution, and a player
    that merely matches its probe contributes zero.
    """

    floor = np.maximum(0.0, np.asarray(tau, dtype=float))
    delta = np.asarray(delta, dtype=float)
    if gate == "none":
        return delta
    if gate == "soft":
        return np.maximum(0.0, delta - floor)
    if gate == "hard":
        return np.where(delta > floor, delta, 0.0)
    raise ValueError(f"unknown gate {gate!r}")


def gate_all(
    delta: np.ndarray, shadow_delta: np.ndarray, spec: MethodSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Compute thresholds at the configured scope, then gate without fitting."""

    if not spec.uses_shadows or shadow_delta.size == 0:
        tau = np.zeros_like(delta)
        return tau, apply_gate(delta, tau, spec.gate)

    quantile, method = spec.shadow_quantile, spec.quantile_method
    if spec.shadow_scope == "local":
        # One threshold per player, context, and split: position matched.
        tau = np.quantile(shadow_delta, quantile, axis=3, method=method)
    elif spec.shadow_scope == "global":
        # One threshold per split, pooled across players and contexts.
        players, contexts, splits, _ = shadow_delta.shape
        tau = np.empty((players, contexts, splits), dtype=float)
        for split in range(splits):
            tau[:, :, split] = shadow_threshold(
                shadow_delta[:, :, split, :].ravel(), quantile, method
            )
    else:  # pragma: no cover - validated in MethodSpec
        raise ValueError(f"unknown shadow_scope {spec.shadow_scope!r}")

    return tau, apply_gate(delta, tau, spec.gate)
