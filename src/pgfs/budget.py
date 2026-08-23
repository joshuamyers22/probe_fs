"""Matched-compute accounting (Spec Section 10).

Section 10 makes the decisive comparison gated versus ungated *at matched
computational budgets*, with the trade stated explicitly: the gated method takes
fewer contexts and spends the difference on ``m`` shadow fits per feature-context
pair; the ungated method spends that budget on additional contexts instead.

Getting the match right requires counting fits the way :mod:`pgfs.importance`
actually spends them, which means accounting for the base-loss cache. Within one
importance split, the base and augmented sets a permutation touches are its
``p+1`` nested prefixes, and the prefix of size ``s`` is a *uniform* random
``s``-subset. Across ``T`` independent permutations the expected number of
distinct subsets of size ``s`` is

    C(p, s) * (1 - (1 - 1/C(p, s))^T)

so the expected base fits per split are that quantity summed over ``s = 0..p``,
and shadow fits add ``T * p * m`` on top.

This matters because the saving is not symmetric between the arms. The ungated
arm buys many more contexts with the same budget, and extra contexts hit the
cache more and more often - so the naive "extra contexts cost ``p-1`` fits each"
rule overcharges it, and matching on that rule hands the gated arm roughly ten
percent more compute. The consequence at the limit is worth naming: repeated
contexts saturate at ``2^p`` distinct subsets, so beyond that point the ungated
arm cannot spend more budget on new contexts at all, and an exact match is
infeasible rather than merely awkward. :func:`match_budget` reports the residual
imbalance instead of hiding it, and every experiment additionally reports measured
fit counts and wall-clock time, as Section 10 requires: the prediction is a
planning tool, the measurement is the evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import MethodSpec

__all__ = [
    "BudgetPair",
    "budget_ladder",
    "expected_base_fits",
    "expected_importance_fits",
    "match_budget",
    "matched_ungated_contexts",
    "spec_importance_fits",
]


def expected_base_fits(n_players: int, n_contexts: int, context_kind: str = "permutation") -> float:
    """Expected distinct base/augmented fits per importance split.

    Exact expectation under the permutation context distribution, given the
    ``frozenset`` cache. For full conditioning the sets are deterministic: ``p``
    leave-one-out sets plus the full set.
    """
    p, T = int(n_players), int(n_contexts)
    if p <= 0:
        return 0.0
    if context_kind == "full":
        return float(p + 1)
    if p == 1:
        return 2.0

    total = 0.0
    for s in range(p + 1):
        c = math.comb(p, s)
        if c == 1:
            total += 1.0
        else:
            # c * (1 - (1 - 1/c)^T), computed in log space for large c.
            total += c * (1.0 - math.exp(T * math.log1p(-1.0 / c)))
    return total


def expected_importance_fits(
    n_players: int,
    n_contexts: int,
    n_splits: int,
    n_shadows: int,
    context_kind: str = "permutation",
) -> float:
    """Predicted model fits for one call to ``estimate_importance``."""
    T = 1 if context_kind == "full" else int(n_contexts)
    base = expected_base_fits(n_players, T, context_kind)
    return float(n_splits) * (base + T * int(n_players) * int(n_shadows))


def spec_importance_fits(spec: MethodSpec, n_players: int) -> float:
    return expected_importance_fits(
        n_players,
        spec.effective_n_contexts,
        spec.n_importance_splits,
        spec.effective_n_shadows,
        spec.context_kind,
    )


def matched_ungated_contexts(
    n_players: int,
    n_contexts_gated: int,
    n_shadows: int,
    max_contexts: int = 100_000,
) -> int:
    """Contexts the ungated arm may buy with the gated arm's shadow budget.

    Chosen by search over ``T`` rather than in closed form, because the cache makes
    the cost of a context depend on how many contexts came before it.
    """
    p, T, m = int(n_players), int(n_contexts_gated), int(n_shadows)
    if p <= 1 or m <= 0:
        return max(1, T)

    target = expected_importance_fits(p, T, 1, m)
    best_T, best_gap = T, abs(expected_importance_fits(p, T, 1, 0) - target)

    lo, hi = T, max(T * 2, 8)
    while expected_importance_fits(p, hi, 1, 0) < target and hi < max_contexts:
        hi *= 2
    hi = min(hi, max_contexts)

    while lo <= hi:
        mid = (lo + hi) // 2
        cost = expected_importance_fits(p, mid, 1, 0)
        gap = abs(cost - target)
        if gap < best_gap or (gap == best_gap and mid < best_T):
            best_T, best_gap = mid, gap
        if cost < target:
            lo = mid + 1
        else:
            hi = mid - 1

    # Check the neighbours of the crossing point; the search above can stop one
    # short of the closer of the two.
    for cand in (best_T - 1, best_T + 1):
        if 1 <= cand <= max_contexts:
            gap = abs(expected_importance_fits(p, cand, 1, 0) - target)
            if gap < best_gap:
                best_T, best_gap = cand, gap
    return max(1, best_T)


@dataclass(frozen=True)
class BudgetPair:
    """One rung of the Section 10 budget ladder: a paired gated/ungated spec."""

    budget_index: int
    gated: MethodSpec
    ungated: MethodSpec
    predicted_fits_gated: float
    predicted_fits_ungated: float

    @property
    def imbalance(self) -> float:
        """Relative gap between the two predicted budgets (0 is a perfect match)."""
        hi = max(self.predicted_fits_gated, self.predicted_fits_ungated)
        lo = min(self.predicted_fits_gated, self.predicted_fits_ungated)
        return (hi - lo) / hi if hi else 0.0

    @property
    def feasible(self) -> bool:
        """False when context saturation prevents a real match (see module docs)."""
        return self.imbalance <= 0.05

    def describe(self) -> str:
        flag = "" if self.feasible else "  [!] budget match infeasible at this size"
        return (
            f"budget {self.budget_index}: "
            f"gated T={self.gated.effective_n_contexts} m={self.gated.effective_n_shadows} "
            f"(~{self.predicted_fits_gated:.0f} fits) vs "
            f"ungated T={self.ungated.effective_n_contexts} "
            f"(~{self.predicted_fits_ungated:.0f} fits), "
            f"imbalance {self.imbalance:.1%}{flag}"
        )


def match_budget(base_spec: MethodSpec, n_players: int, n_contexts_gated: int, index: int = 1):
    """Build one paired gated/ungated rung at a given gated context count."""
    gated = base_spec.replace(
        gate="soft", n_contexts=n_contexts_gated,
        label=f"{base_spec.label or 'pgfs'}-gated@C{index}",
    )
    T_ung = matched_ungated_contexts(n_players, n_contexts_gated, base_spec.n_shadows)
    ungated = base_spec.replace(
        gate="none", n_contexts=T_ung,
        label=f"{base_spec.label or 'pgfs'}-ungated@C{index}",
    )
    return BudgetPair(
        budget_index=index,
        gated=gated,
        ungated=ungated,
        predicted_fits_gated=spec_importance_fits(gated, n_players),
        predicted_fits_ungated=spec_importance_fits(ungated, n_players),
    )


def budget_ladder(
    base_spec: MethodSpec,
    n_players: int,
    gated_contexts: tuple[int, ...],
) -> list[BudgetPair]:
    """Build paired specs across budgets ``C_1 < ... < C_H`` (Section 10).

    Section 10 also insists results be plotted against computational budget rather
    than reported at a single matched point, which is why this returns a ladder
    rather than one pair. Both specs in a pair share the base spec's seed, so they
    see identical splits and identical permutations for identical context indices
    (Section 4).
    """
    return [
        match_budget(base_spec, n_players, T, index=h + 1)
        for h, T in enumerate(sorted(gated_contexts))
    ]
