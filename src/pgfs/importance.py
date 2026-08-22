"""Raw contributions, shadow thresholds, gating and ranking (Spec Sections 5-7).

For split ``b``, context ``t`` and player ``j``:

    Delta_jtb = L(S_jt) - L(S_jt + j)                                 (Sec. 5)
    tau_jtb   = Q_q(Delta^(1)_jtb, ..., Delta^(m)_jtb)                (Sec. 6)
    G_jtb     = max(0, Delta_jtb - max(0, tau_jtb))                   (Sec. 7)
    I_j^gate  = mean over b, t of G_jtb                               (Sec. 7)

Fits happen in one pass; thresholding and gating happen in a second, vectorized
pass over the stored shadow contributions. Separating them is what lets the
Section 12 comparator "global shadow comparison without conditioning-position
matching" reuse the identical fits: local and global scope differ only in which
shadow contributions enter a quantile, never in what was fitted.

Two further implementation points deserve stating.

**Base-loss caching.** Within one importance split the conditioning sets drawn by
a permutation are nested: the augmented set of the player at position ``i`` is the
base set of the player at position ``i+1``. Caching losses on ``frozenset(S)``
therefore turns ``2p`` base/augmented fits per permutation into ``p+1``. This is a
pure bookkeeping win - it changes no number the method reports - but it changes the
compute budget, so cache hits are counted separately from fits and the matched-
compute accounting in :mod:`pgfs.budget` is built on the same arithmetic.

**Fairness of the pairing.** Splits, permutations and learner seeds are addressed
by position, never by call order, so a gated and an ungated spec sharing a seed
see the same splits and the same permutations for the same context indices
(Sections 4 and 10). The ungated spec spends no shadow fits at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MethodSpec
from .contexts import sample_context
from .learners import FitCounter, evaluate_player_set
from .players import Players, derive_rng, derive_seed
from .shadows import shadow_blocks, shadow_threshold

__all__ = ["ImportanceResult", "estimate_importance", "rank_players", "apply_gate"]

# RNG address-space tags, so different draws never collide.
_TAG_SPLIT, _TAG_CONTEXT, _TAG_SHADOW, _TAG_LEARNER = 1, 2, 3, 4


@dataclass
class ImportanceResult:
    """Per-player scores plus the ``(p, T, B)`` arrays needed for diagnostics."""

    players: Players
    gated_score: np.ndarray            # I_j^gate  (Section 7)
    raw_score: np.ndarray              # I_j^raw   (Section 7, diagnostics only)
    ranking: tuple[int, ...]           # decreasing gated score, ties by player index
    delta: np.ndarray                  # (p, T, B)
    tau: np.ndarray                    # (p, T, B); all-zero for an ungated run
    gated: np.ndarray                  # (p, T, B)
    shadow_delta: np.ndarray           # (p, T, B, m); empty when ungated
    spec_fingerprint: str = ""
    fit_signature: tuple[tuple[str, str], ...] = ()
    counter: FitCounter = field(default_factory=FitCounter)

    @property
    def n_players(self) -> int:
        return len(self.players)

    def top_k(self, k: int) -> tuple[int, ...]:
        """``A_k = {j_(1), ..., j_(k)}`` (Section 8)."""
        return tuple(self.ranking[: int(k)])

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Section 14 diagnostics.

        Context variability and split variability are returned as separate
        arrays, as Section 14 requires; collapsing them into a single spread
        would hide which source of noise dominates a ranking.
        """
        per_context = self.gated.mean(axis=2)   # (p, T): averaged over splits
        per_split = self.gated.mean(axis=1)     # (p, B): averaged over contexts
        beat = self.delta - np.maximum(0.0, self.tau)
        return {
            "context_sd": per_context.std(axis=1, ddof=1)
            if per_context.shape[1] > 1 else np.zeros(self.n_players),
            "split_sd": per_split.std(axis=1, ddof=1)
            if per_split.shape[1] > 1 else np.zeros(self.n_players),
            # Diagnostic only. Section 8 forbids thresholding on win rate.
            "win_rate": (beat > 0).mean(axis=(1, 2)),
            "mean_tau": self.tau.mean(axis=(1, 2)),
            "mean_delta": self.delta.mean(axis=(1, 2)),
        }

    def regate(self, spec: MethodSpec) -> "ImportanceResult":
        """Re-derive scores under a different gate or shadow scope, without refitting.

        Valid only for a spec differing from the one that produced this result in
        gating parameters alone (``gate``, ``shadow_quantile``, ``quantile_method``,
        ``shadow_scope``). This is how the Section 13 sensitivity analyses over
        ``q`` and the Section 12 global-shadow comparator avoid paying for the same
        fits twice. Sensitivity over ``m`` is *not* available this way: fewer
        shadows is a different set of draws, not a re-reading of these ones.
        """
        requested = _fit_signature(spec)
        if self.fit_signature and requested != self.fit_signature:
            raise ValueError(
                "regate may only change gate, shadow quantile/method, or shadow scope; "
                "the requested spec changes the fitted design"
            )
        if spec.uses_shadows and self.shadow_delta.shape[3] != spec.effective_n_shadows:
            raise ValueError("regate cannot change the number of fitted shadows")
        tau, gated = _gate_all(self.delta, self.shadow_delta, spec)
        score = gated.mean(axis=(1, 2))
        return ImportanceResult(
            players=self.players,
            gated_score=score,
            raw_score=self.raw_score,
            ranking=rank_players(score),
            delta=self.delta,
            tau=tau,
            gated=gated,
            shadow_delta=self.shadow_delta,
            spec_fingerprint=spec.fingerprint(),
            fit_signature=requested,
            counter=self.counter,
        )

    def to_table(self) -> list[dict[str, Any]]:
        diag = self.diagnostics()
        rank_of = {j: r for r, j in enumerate(self.ranking)}
        return [
            {
                "player": self.players.names[j],
                "index": j,
                "rank": rank_of[j],
                "gated_score": float(self.gated_score[j]),
                "raw_score": float(self.raw_score[j]),
                "context_sd": float(diag["context_sd"][j]),
                "split_sd": float(diag["split_sd"][j]),
                "win_rate": float(diag["win_rate"][j]),
            }
            for j in sorted(range(self.n_players), key=lambda j: rank_of[j])
        ]


def rank_players(score: np.ndarray) -> tuple[int, ...]:
    """Decreasing score, exact ties broken by original player order (Section 7)."""
    return tuple(sorted(range(len(score)), key=lambda j: (-float(score[j]), j)))


def _fit_signature(spec: MethodSpec) -> tuple[tuple[str, str], ...]:
    """Fields that determine splits, contexts, learner fits, and shadow draws."""
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
    """Section 7's soft gate, plus the Section 13 sensitivity variants.

    soft: ``max(0, Delta - max(0, tau))``. The inner ``max(0, tau)`` means a
    *negative* threshold - shadows that actively hurt - never credits a player
    with more than its own raw contribution; the outer ``max(0, .)`` means a
    player that merely matches its probe contributes nothing rather than a
    negative amount.
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


def _gate_all(
    delta: np.ndarray, shadow_delta: np.ndarray, spec: MethodSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``tau`` at the configured scope, then gate. No model fits here."""
    if not spec.uses_shadows or shadow_delta.size == 0:
        tau = np.zeros_like(delta)
        return tau, apply_gate(delta, tau, spec.gate)

    q, method = spec.shadow_quantile, spec.quantile_method
    if spec.shadow_scope == "local":
        # One threshold per (j, t, b): conditioning-position matched (Section 6).
        tau = np.quantile(shadow_delta, q, axis=3, method=method)
    elif spec.shadow_scope == "global":
        # Section 12 comparator: one threshold per split, pooled over players and
        # contexts, so a player's probe is no longer matched to its position.
        p, T, B, _ = shadow_delta.shape
        tau = np.empty((p, T, B), dtype=float)
        for b in range(B):
            tau[:, :, b] = shadow_threshold(shadow_delta[:, :, b, :].ravel(), q, method)
    else:  # pragma: no cover - validated in MethodSpec
        raise ValueError(f"unknown shadow_scope {spec.shadow_scope!r}")

    return tau, apply_gate(delta, tau, spec.gate)


def _split_indices(
    n: int, val_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _validate_estimation_data(X: np.ndarray, y: np.ndarray, players: Players) -> None:
    """Fail early with domain-specific messages at the estimation boundary."""
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


def estimate_importance(
    X: np.ndarray,
    y: np.ndarray,
    players: Players,
    spec: MethodSpec,
    key: tuple[int, ...] = (),
    counter: FitCounter | None = None,
) -> ImportanceResult:
    """Estimate gated (and raw) importance on one dataset.

    ``X, y`` are an *inner-training* dataset in the sense of Section 7 ("within an
    inner-training dataset, the primary ranking score is ..."): the caller is
    responsible for never passing data that the current evaluation level is meant
    to hold out.

    ``key`` addresses this call's position in the nested design (e.g.
    ``(outer_fold, inner_fold)``), so that randomness is reproducible and so that
    two specs sharing a seed are paired wherever their designs coincide.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    _validate_estimation_data(X, y, players)
    p = len(players)
    T = spec.effective_n_contexts
    B = int(spec.n_importance_splits)
    m = spec.effective_n_shadows
    loss = spec.loss_fn
    counter = counter if counter is not None else FitCounter()

    delta = np.zeros((p, T, B), dtype=float)
    shadow_delta = np.zeros((p, T, B, m), dtype=float)

    for b in range(B):
        split_rng = derive_rng(spec.seed, *key, _TAG_SPLIT, b)
        tr_idx, va_idx = _split_indices(len(y), spec.importance_val_fraction, split_rng)
        X_tr_all, y_tr = X[tr_idx], y[tr_idx]
        X_va_all, y_va = X[va_idx], y[va_idx]

        # One learner seed for the whole split: base, augmented and shadow fits
        # share it, as Section 5 requires.
        learner_seed = derive_seed(spec.seed, *key, _TAG_LEARNER, b)

        cache: dict[frozenset, float] = {}

        def base_loss(player_set: frozenset) -> float:
            hit = cache.get(player_set)
            if hit is not None:
                counter.cache_hits += 1
                return hit
            cols = players.select_columns(player_set)
            value = evaluate_player_set(
                spec.learner, loss,
                X_tr_all[:, cols], y_tr, X_va_all[:, cols], y_va,
                seed=learner_seed, counter=counter, kind="importance",
            )
            cache[player_set] = value
            return value

        for t in range(T):
            ctx_rng = derive_rng(spec.seed, *key, _TAG_CONTEXT, b, t)
            context = sample_context(spec.context_kind, p, ctx_rng)

            for j, S in context:
                S_set = frozenset(S)
                L_base = base_loss(S_set)
                L_aug = base_loss(S_set | {j})
                delta[j, t, b] = L_base - L_aug

                if m == 0:
                    continue

                base_cols = players.select_columns(S_set)
                player_cols = np.asarray(players.columns[j], dtype=int)
                for r in range(m):
                    s_rng = derive_rng(spec.seed, *key, _TAG_SHADOW, b, t, j, r)
                    Z_tr, Z_va = shadow_blocks(X, tr_idx, va_idx, player_cols, s_rng)
                    Xs_tr = np.hstack([X_tr_all[:, base_cols], Z_tr])
                    Xs_va = np.hstack([X_va_all[:, base_cols], Z_va])
                    L_shadow = evaluate_player_set(
                        spec.learner, loss,
                        Xs_tr, y_tr, Xs_va, y_va,
                        seed=learner_seed, counter=counter, kind="shadow",
                    )
                    shadow_delta[j, t, b, r] = L_base - L_shadow

    tau, gated = _gate_all(delta, shadow_delta, spec)
    gated_score = gated.mean(axis=(1, 2))
    return ImportanceResult(
        players=players,
        gated_score=gated_score,
        raw_score=delta.mean(axis=(1, 2)),
        ranking=rank_players(gated_score),
        delta=delta,
        tau=tau,
        gated=gated,
        shadow_delta=shadow_delta,
        spec_fingerprint=spec.fingerprint(),
        fit_signature=_fit_signature(spec),
        counter=counter,
    )
