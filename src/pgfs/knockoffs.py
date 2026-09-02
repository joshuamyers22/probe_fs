"""Gaussian model-X knockoff construction and selection policy."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

from .players import derive_rng

KNOCKOFF_RNG_KEY = 33


def select_knockoff_columns(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    target_fdr: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select columns using equicorrelated Gaussian model-X knockoffs."""
    rng = derive_rng(seed, KNOCKOFF_RNG_KEY)
    Xs = StandardScaler().fit_transform(X)
    n, d = Xs.shape

    covariance = np.corrcoef(Xs, rowvar=False)
    covariance = np.atleast_2d(covariance) + 1e-6 * np.eye(d)
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance).min())
    s = np.full(d, min(1.0, max(2.0 * minimum_eigenvalue - 1e-8, 1e-8)))

    covariance_inverse = np.linalg.pinv(covariance)
    diagonal = np.diag(s)
    knockoff_mean = Xs - Xs @ covariance_inverse @ diagonal
    knockoff_covariance = 2.0 * diagonal - diagonal @ covariance_inverse @ diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(
        (knockoff_covariance + knockoff_covariance.T) / 2.0
    )
    covariance_root = (
        eigenvectors
        @ np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None)))
        @ eigenvectors.T
    )
    knockoffs = knockoff_mean + rng.normal(size=(n, d)) @ covariance_root

    augmented = np.hstack([Xs, knockoffs])
    alphas, coefficients, _ = lasso_path(
        augmented, np.asarray(y, dtype=float), n_alphas=60, eps=1e-3
    )
    entry = np.zeros(2 * d, dtype=float)
    nonzero = np.abs(coefficients) > 1e-10
    for column in range(2 * d):
        indices = np.flatnonzero(nonzero[column])
        entry[column] = alphas[indices[0]] if indices.size else 0.0

    statistics = entry[:d] - entry[d:]
    candidates = np.sort(np.abs(statistics[statistics != 0]))
    threshold = np.inf
    for candidate in candidates:
        false_discoveries = 1 + np.sum(statistics <= -candidate)
        discoveries = max(1, np.sum(statistics >= candidate))
        if false_discoveries / discoveries <= target_fdr:
            threshold = candidate
            break
    selected = (
        np.flatnonzero(statistics >= threshold)
        if np.isfinite(threshold)
        else np.array([], dtype=int)
    )
    metadata = {
        "threshold": float(threshold),
        "min_eigenvalue": minimum_eigenvalue,
        "target_fdr": target_fdr,
    }
    return selected, metadata
