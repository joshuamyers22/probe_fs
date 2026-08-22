"""Players and deterministic RNG derivation.

Spec Section 3: "A player may be a single feature or a predefined block of
features."  Everything downstream (contexts, shadows, gating, ranking, selection)
operates on players, never on raw columns, so that a block is added, permuted and
selected as one unit (Section 6: "for a feature block, use one joint row
permutation for the entire block").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["Players", "derive_rng", "derive_seed"]

MAX_RANDOM_SEED = 2**31 - 1


@dataclass(frozen=True)
class Players:
    """An ordered partition (or sub-selection) of design-matrix columns.

    ``columns[j]`` holds the column indices belonging to player ``j``. Player
    order is the tie-breaking order required by Section 7 ("resolve exact ties
    with a fixed deterministic rule, such as original column order").
    """

    columns: tuple[tuple[int, ...], ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.columns) != len(self.names):
            raise ValueError("columns and names must have equal length")
        seen: set[int] = set()
        for cols in self.columns:
            if len(cols) == 0:
                raise ValueError("a player must own at least one column")
            if len(set(cols)) != len(cols):
                raise ValueError("a column cannot occur twice within one player")
            if any(not isinstance(c, (int, np.integer)) or c < 0 for c in cols):
                raise ValueError("player columns must be non-negative integers")
            overlap = seen.intersection(cols)
            if overlap:
                raise ValueError(f"columns assigned to more than one player: {sorted(overlap)}")
            seen.update(cols)

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self):
        return iter(range(len(self.columns)))

    @property
    def n_players(self) -> int:
        return len(self.columns)

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(len(c) for c in self.columns)

    @classmethod
    def singletons(cls, n_columns: int, names: Sequence[str] | None = None) -> "Players":
        """One player per column - the default in Sections 4-8."""
        if not isinstance(n_columns, (int, np.integer)) or n_columns < 1:
            raise ValueError("n_columns must be a positive integer")
        if names is None:
            names = [f"x{j}" for j in range(n_columns)]
        if len(names) != n_columns:
            raise ValueError("names length must equal n_columns")
        return cls(tuple((j,) for j in range(n_columns)), tuple(names))

    @classmethod
    def from_blocks(
        cls,
        blocks: Iterable[Sequence[int]],
        names: Sequence[str] | None = None,
    ) -> "Players":
        """Explicit blocks; each block is one indivisible player."""
        cols = tuple(tuple(int(c) for c in b) for b in blocks)
        if names is None:
            names = [f"block{j}" for j in range(len(cols))]
        return cls(cols, tuple(names))

    def select_columns(self, player_set: Iterable[int]) -> np.ndarray:
        """Column indices spanned by a set of players, in player order."""
        out: list[int] = []
        for j in sorted(player_set):
            out.extend(self.columns[j])
        return np.asarray(out, dtype=int)


def derive_rng(seed: int, *key: int) -> np.random.Generator:
    """Deterministic child generator addressed by an integer key path.

    Reproducibility requirement of Sections 3 and 10 ("paired random seeds"):
    randomness is addressed by *position in the design* (outer fold, inner fold,
    split, context, player, shadow), never by call order. Two methods that share
    a seed therefore see identical splits, identical permutations and identical
    shadow draws wherever their designs coincide - which is exactly what Section 4
    demands ("the same permutations must be used for gated and ungated methods in
    paired comparisons").
    """
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(int(k) for k in key))
    return np.random.default_rng(ss)


def derive_seed(seed: int, *key: int) -> int:
    """Return a deterministic estimator-compatible seed for an address path."""
    return int(derive_rng(seed, *key).integers(0, MAX_RANDOM_SEED))
