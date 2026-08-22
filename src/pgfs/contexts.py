"""Conditioning-context distributions (Spec Section 4).

The primary method's sole context distribution is the uniform permutation
distribution: for repetition ``t`` draw a permutation ``pi_t`` of the ``p``
players and condition each player on its predecessors,

    S_jt = { l : pi_t(l) < pi_t(j) }.

Full-conditioning LOCO (``S_j = all players except j``) is the principal
alternative of Section 2 and the ablation of Section 13; it is exposed here as a
second sampler so that the *identical* scoring, gating, ranking and selection code
paths run for both, which is what makes the Section 6 ablation a clean comparison.

A context repetition is materialised as a list of ``(player, conditioning set)``
pairs. Emitting whole repetitions - rather than one player at a time - is what
lets :mod:`pgfs.importance` exploit the nesting of permutation prefixes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CONTEXT_KINDS", "sample_context", "n_contexts_for_kind"]

CONTEXT_KINDS = ("permutation", "full")

# A context repetition: ordered list of (player j, conditioning set S_j) pairs.
Context = list[tuple[int, tuple[int, ...]]]


def sample_context(kind: str, n_players: int, rng: np.random.Generator) -> Context:
    """Draw one conditioning-context repetition.

    Parameters
    ----------
    kind:
        ``"permutation"`` (primary, Section 4) or ``"full"`` (full-conditioning
        LOCO, Section 12).
    rng:
        Generator addressed by ``(seed, ..., context index)`` so that two methods
        sharing a seed draw identical permutations for identical context indices.
    """
    if kind == "permutation":
        order = rng.permutation(n_players)
        out: Context = []
        prefix: list[int] = []
        for j in order:
            out.append((int(j), tuple(prefix)))
            prefix.append(int(j))
        return out
    if kind == "full":
        allp = list(range(n_players))
        return [(j, tuple(x for x in allp if x != j)) for j in allp]
    raise ValueError(f"unknown context kind {kind!r}; available: {CONTEXT_KINDS}")


def n_contexts_for_kind(kind: str, requested: int) -> int:
    """Full conditioning is deterministic, so repeating it buys nothing."""
    if kind == "full":
        return 1
    return int(requested)
