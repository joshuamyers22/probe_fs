"""The frozen method specification (Spec Section 3).

Section 3 lists what must be fixed *before evaluation*; Section 9 step 4 requires
re-estimating the ranking "using the frozen method specification". Both are served
by one immutable object that every routine receives and none can modify, plus a
:meth:`MethodSpec.fingerprint` string that goes into result records so a reported
number can be traced back to the specification that produced it.

Note what is deliberately *not* here: no gated-importance threshold, no win-rate
cutoff, no selection-frequency cutoff. Section 8: "Selected-set size k is the only
selection-path parameter."
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Sequence

from .contexts import CONTEXT_KINDS
from .losses import LOSSES, Loss, get_loss
from .shadows import QUANTILE_METHODS

__all__ = ["MethodSpec", "GATES", "SHADOW_SCOPES"]

# "soft" is the primary method (Section 7). "hard" and "none" exist for the
# Section 13 sensitivity analyses and the Section 10 ungated comparison.
GATES = ("soft", "hard", "none")

# Section 12 requires "global shadow comparison without conditioning-position
# matching" as a direct alternative. "local" is the primary method: one threshold
# per (j, t, b). "global" pools shadow contributions across players and contexts
# within a split, which is precisely the conditioning-position matching removed.
SHADOW_SCOPES = ("local", "global")


@dataclass(frozen=True)
class MethodSpec:
    """Immutable method specification.

    Parameters
    ----------
    learner:
        Unfitted scikit-learn-style estimator, cloned for every fit.
    loss:
        Key into :data:`pgfs.losses.LOSSES`; also fixes the task type.
    n_contexts:
        ``T``, contexts per importance split (Section 4). Forced to 1 when
        ``context_kind="full"``, which is deterministic.
    n_importance_splits:
        ``B``, train/validation splits used inside one importance estimation
        (Sections 5-7). These are *not* the inner CV folds that choose ``k``.
    importance_val_fraction:
        Validation share of each importance split.
    n_shadows:
        ``m`` shadows per ``(j, t, b)`` (Section 6). Ignored when
        ``gate="none"``, where no shadow fits are spent at all - which is exactly
        what frees budget for extra contexts in the Section 10 comparison.
    shadow_quantile:
        ``q``; the primary specification fixes ``q = 0.9`` (Section 6).
    quantile_method:
        Empirical-quantile convention, which Section 6 requires be stated.
    shadow_scope:
        ``"local"`` (primary: one tau per ``(j, t, b)``) or ``"global"``
        (Section 12's unmatched comparator: one tau per split, pooled over
        players and contexts). Costs no extra fits either way.
    candidate_k:
        ``K``, candidate selected-set sizes (Sections 3 and 8).
    one_se_rule:
        Section 8's one-standard-error rule. ``False`` selects the argmin
        instead, and is a deviation from the primary method.
    gate:
        ``"soft"`` (primary), ``"hard"`` (sensitivity), ``"none"`` (ungated
        comparator of Section 10).
    context_kind:
        ``"permutation"`` (primary) or ``"full"`` (LOCO alternative).
    seed:
        Root seed. Two specs sharing a seed see identical splits and identical
        permutations wherever their designs coincide (Sections 4 and 10).
    """

    learner: Any
    loss: str = "mse"
    n_contexts: int = 8
    n_importance_splits: int = 2
    importance_val_fraction: float = 0.3
    n_shadows: int = 5
    shadow_quantile: float = 0.9
    quantile_method: str = "higher"
    shadow_scope: str = "local"
    candidate_k: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20)
    one_se_rule: bool = True
    gate: str = "soft"
    context_kind: str = "permutation"
    seed: int = 0
    label: str = ""
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; available: {sorted(LOSSES)}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}")
        if self.context_kind not in CONTEXT_KINDS:
            raise ValueError(f"context_kind must be one of {CONTEXT_KINDS}")
        if self.shadow_scope not in SHADOW_SCOPES:
            raise ValueError(f"shadow_scope must be one of {SHADOW_SCOPES}")
        if self.quantile_method not in QUANTILE_METHODS:
            raise ValueError(f"quantile_method must be one of {QUANTILE_METHODS}")
        if not 0.0 < self.shadow_quantile < 1.0:
            raise ValueError("shadow_quantile must lie strictly inside (0, 1)")
        if not 0.0 < self.importance_val_fraction < 1.0:
            raise ValueError("importance_val_fraction must lie strictly inside (0, 1)")
        count_fields = {
            "n_contexts": self.n_contexts,
            "n_importance_splits": self.n_importance_splits,
            "n_shadows": self.n_shadows,
        }
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in count_fields.values()):
            raise ValueError("context, split, and shadow counts must be integers")
        if self.n_contexts < 1 or self.n_importance_splits < 1:
            raise ValueError("n_contexts and n_importance_splits must be >= 1")
        if self.n_shadows < 0:
            raise ValueError("n_shadows must be >= 0")
        if self.gate != "none" and self.n_shadows < 1:
            raise ValueError("a gated spec needs n_shadows >= 1")
        if (
            not self.candidate_k
            or any(not isinstance(k, Integral) or isinstance(k, bool) or k < 1 for k in self.candidate_k)
        ):
            raise ValueError("candidate_k must be a non-empty set of positive sizes")
        object.__setattr__(self, "candidate_k", tuple(sorted(set(int(k) for k in self.candidate_k))))

    # -- derived -----------------------------------------------------------
    @property
    def loss_fn(self) -> Loss:
        return get_loss(self.loss)

    @property
    def task(self) -> str:
        return self.loss_fn.task

    @property
    def uses_shadows(self) -> bool:
        return self.gate != "none"

    @property
    def effective_n_contexts(self) -> int:
        return 1 if self.context_kind == "full" else int(self.n_contexts)

    @property
    def effective_n_shadows(self) -> int:
        return int(self.n_shadows) if self.uses_shadows else 0

    def candidates_within(self, n_players: int) -> tuple[int, ...]:
        """Candidate sizes clipped to the number of available players."""
        ks = tuple(k for k in self.candidate_k if k <= n_players)
        return ks if ks else (n_players,)

    def replace(self, **changes: Any) -> "MethodSpec":
        """Return a modified copy (the original stays frozen)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)

    # -- provenance --------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "learner": repr(self.learner),
            "loss": self.loss,
            "task": self.task,
            "context_kind": self.context_kind,
            "n_contexts": self.effective_n_contexts,
            "n_importance_splits": self.n_importance_splits,
            "importance_val_fraction": self.importance_val_fraction,
            "gate": self.gate,
            "n_shadows": self.effective_n_shadows,
            "shadow_quantile": self.shadow_quantile if self.uses_shadows else None,
            "quantile_method": self.quantile_method if self.uses_shadows else None,
            "shadow_scope": self.shadow_scope if self.uses_shadows else None,
            "candidate_k": list(self.candidate_k),
            "one_se_rule": self.one_se_rule,
            "seed": self.seed,
        }

    def fingerprint(self) -> str:
        """Stable short hash of the frozen specification."""
        blob = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        d = self.as_dict()
        gate = d["gate"]
        shadow = f", m={d['n_shadows']}, q={d['shadow_quantile']}" if self.uses_shadows else ""
        return (
            f"{self.label or 'spec'}[{self.fingerprint()}] "
            f"gate={gate}, contexts={d['context_kind']}xT{d['n_contexts']}, "
            f"B={d['n_importance_splits']}{shadow}, loss={d['loss']}"
        )
