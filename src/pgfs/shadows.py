"""Marginally permuted, dimension-matched shadows (Spec Section 6).

For each ``(j, t, b)`` a shadow is built by *independently* permuting the training
rows of player ``j`` and *independently* permuting its validation rows. Two
details matter and are enforced here:

1. Train and validation row permutations are drawn separately, so the shadow
   carries player ``j``'s marginal distribution into both blocks while destroying
   its row-level link to ``Y`` in each.
2. A block player receives **one joint row permutation for the whole block**, so
   dimension and within-block dependence survive; permuting block columns
   independently would create a probe that is easier to beat than the real player.

Section 6 also fixes the interpretation: shadows are negative controls, not
conditionally exchangeable null variables. ``tau`` carries no null probability, so
nothing in this module returns anything resembling a p-value.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QUANTILE_METHODS", "shadow_blocks", "shadow_threshold"]

# Section 6: "The exact empirical-quantile convention must be specified in the
# implementation." The default is `higher`, the conservative choice: with m
# shadows it returns an order statistic actually attained by a shadow draw, and
# never interpolates below one.
QUANTILE_METHODS = ("higher", "linear", "lower", "nearest", "midpoint")


def shadow_blocks(
    X: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    cols: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one dimension-matched marginal shadow of a player.

    Returns
    -------
    (Z_tr, Z_va):
        Shadow columns for the training and validation blocks, each with the same
        number of columns as the player. One row permutation per block, shared
        across the player's columns.
    """
    perm_tr = rng.permutation(len(tr_idx))
    perm_va = rng.permutation(len(va_idx))
    Z_tr = X[np.ix_(tr_idx[perm_tr], cols)]
    Z_va = X[np.ix_(va_idx[perm_va], cols)]
    return Z_tr, Z_va


def shadow_threshold(
    shadow_deltas: np.ndarray,
    q: float,
    method: str = "higher",
) -> float:
    """Local shadow threshold ``tau_jtb = Q_q(Delta^(1), ..., Delta^(m))``."""
    if method not in QUANTILE_METHODS:
        raise ValueError(f"unknown quantile method {method!r}; available: {QUANTILE_METHODS}")
    arr = np.asarray(shadow_deltas, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q, method=method))
