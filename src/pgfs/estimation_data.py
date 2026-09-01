"""Domain validation for feature-importance estimation inputs."""

import numpy as np

from .players import Players


def validate_estimation_data(X: np.ndarray, y: np.ndarray, players: Players) -> None:
    """Reject arrays and player definitions that cannot be estimated safely."""
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if len(y) < 2:
        raise ValueError("importance estimation needs at least two observations")
    if len(players) < 1:
        raise ValueError("at least one player is required")
    columns = (column for block in players.columns for column in block)
    if any(column >= X.shape[1] for column in columns):
        raise ValueError("player columns must be valid columns of X")
