"""Loss functions.

Spec Section 3: the loss function is part of the frozen method specification and
must be fixed before evaluation. Every loss here is *lower-is-better*, which is
what Section 5 assumes when it defines

    Delta_jtb = L(S) - L(S + j)

so that positive Delta means adding player j improved validation performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["Loss", "LOSSES", "get_loss"]

_EPS = 1e-12


@dataclass(frozen=True)
class Loss:
    """A lower-is-better validation loss.

    Attributes
    ----------
    name:
        Registry key, stored in the method fingerprint.
    task:
        ``"regression"`` or ``"binary"``. Determines whether the learner adapter
        produces point predictions or class-1 probabilities.
    fn:
        ``fn(y_true, pred) -> float``.
    """

    name: str
    task: str
    fn: Callable[[np.ndarray, np.ndarray], float]

    def __call__(self, y_true: np.ndarray, pred: np.ndarray) -> float:
        return float(self.fn(np.asarray(y_true), np.asarray(pred)))


def _mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _error_rate(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(float) != y))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, 0.0, 1.0) - y) ** 2))


LOSSES: dict[str, Loss] = {
    "mse": Loss("mse", "regression", _mse),
    "mae": Loss("mae", "regression", _mae),
    "log_loss": Loss("log_loss", "binary", _log_loss),
    "error_rate": Loss("error_rate", "binary", _error_rate),
    "brier": Loss("brier", "binary", _brier),
}


def get_loss(name: str) -> Loss:
    try:
        return LOSSES[name]
    except KeyError:  # pragma: no cover - defensive
        raise KeyError(
            f"unknown loss {name!r}; available: {sorted(LOSSES)}"
        ) from None
