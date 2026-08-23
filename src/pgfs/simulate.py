"""Data-generating processes (Spec Section 13).

Section 13's primary regime "contains correlated, redundant signal groups", and
Section 11 requires the ground-truth classes ``C_1, ..., C_g`` to be defined
*before* running the study, since the primary endpoint is class recall. The
primary DGP therefore returns its classes as data, not as an afterthought:

    y = sum_h beta_h * U_h + noise

where ``U_h`` is a latent driver for class ``h`` and every observed member of that
class is a noisy view of ``U_h``. Any single member supplies the class's predictive
information, which is exactly the redundancy that makes exact-support recovery
the wrong target (Section 15) and class recall the right one.

Null predictors are generated in correlated blocks as well as independently. Nulls
that are correlated with each other - but with nothing in ``y`` - are the ones that
make a shadow gate earn its keep or fail visibly, so a regime containing only
independent nulls would flatter the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metrics import SignalClasses
from .players import Players

__all__ = ["SECONDARY_REGIMES", "SimData", "make_primary", "make_secondary"]

SECONDARY_REGIMES = (
    "independent",     # independent null and signal features
    "weak",            # weak signals near the selection boundary
    "nonlinear",       # nonlinear main effects
    "interaction",     # interactions with weak marginal effects
    "mixed",           # mixed continuous and categorical predictors
    "p_ge_n",          # p >= n
)


@dataclass
class SimData:
    """A simulated dataset plus the ground truth the endpoint needs."""

    X: np.ndarray
    y: np.ndarray
    players: Players
    signal_classes: SignalClasses
    task: str
    regime: str
    params: dict

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def p(self) -> int:
        return self.X.shape[1]

    def describe(self) -> str:
        return (
            f"{self.regime}: n={self.n}, p={self.p}, "
            f"g={self.signal_classes.g} classes, "
            f"{len(self.signal_classes.members)} signal members, "
            f"{self.p - len(self.signal_classes.members)} nulls"
        )


def _binarize(eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    prob = 1.0 / (1.0 + np.exp(-eta))
    return (rng.random(len(prob)) < prob).astype(float)


def _scale_noise(signal: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    sd = np.sqrt(np.var(signal) / max(snr, 1e-12))
    return signal + rng.normal(0.0, sd, size=len(signal))


def make_primary(
    n: int = 400,
    n_classes: int = 4,
    class_size: int = 3,
    n_null_blocks: int = 3,
    null_block_size: int = 4,
    n_independent_nulls: int = 8,
    rho: float = 0.8,
    null_rho: float = 0.6,
    snr: float = 2.0,
    betas: tuple[float, ...] | None = None,
    task: Literal["regression", "binary"] = "regression",
    seed: int = 0,
) -> SimData:
    """Primary regime: correlated, redundant signal classes (Sections 11 and 13).

    Parameters
    ----------
    n_classes, class_size:
        ``g`` classes of ``class_size`` interchangeable members each.
    rho:
        Within-class correlation between a member and its class latent driver's
        realisation; members of a class correlate at roughly ``rho`` with one
        another.
    null_rho:
        Within-block correlation among null predictors. Correlated nulls are the
        hard case for any shadow calibration, so they are in the primary grid, not
        relegated to sensitivity analysis.
    snr:
        Ratio of signal variance to noise variance for the regression target.
    """
    rng = np.random.default_rng(seed)
    betas = tuple(betas) if betas is not None else tuple(
        1.0 - 0.15 * h for h in range(n_classes)
    )
    if len(betas) != n_classes:
        raise ValueError("betas must have length n_classes")

    cols: list[np.ndarray] = []
    names: list[str] = []
    classes: list[list[int]] = []
    eta = np.zeros(n, dtype=float)

    a, b = np.sqrt(rho), np.sqrt(1.0 - rho)
    for h in range(n_classes):
        U = rng.normal(size=n)
        eta += betas[h] * U
        members: list[int] = []
        for c in range(class_size):
            members.append(len(cols))
            cols.append(a * U + b * rng.normal(size=n))
            names.append(f"sig{h + 1}_{c + 1}")
        classes.append(members)

    an, bn = np.sqrt(null_rho), np.sqrt(1.0 - null_rho)
    for blk in range(n_null_blocks):
        V = rng.normal(size=n)  # correlated with its block, unrelated to y
        for c in range(null_block_size):
            names.append(f"nullblk{blk + 1}_{c + 1}")
            cols.append(an * V + bn * rng.normal(size=n))

    for i in range(n_independent_nulls):
        names.append(f"null{i + 1}")
        cols.append(rng.normal(size=n))

    X = np.column_stack(cols)
    if task == "regression":
        y = _scale_noise(eta, snr, rng)
    else:
        y = _binarize(eta * np.sqrt(snr) / max(np.std(eta), 1e-12), rng)

    return SimData(
        X=X,
        y=y,
        players=Players.singletons(X.shape[1], names),
        signal_classes=SignalClasses.from_lists(classes),
        task=task,
        regime="primary",
        params={
            "n": n, "n_classes": n_classes, "class_size": class_size,
            "rho": rho, "null_rho": null_rho, "snr": snr, "betas": list(betas),
            "n_null_blocks": n_null_blocks, "null_block_size": null_block_size,
            "n_independent_nulls": n_independent_nulls, "seed": seed, "task": task,
        },
    )


def make_secondary(
    regime: str,
    n: int = 300,
    seed: int = 0,
    task: Literal["regression", "binary"] = "regression",
    **kwargs,
) -> SimData:
    """Section 13's secondary regimes.

    Section 13 is explicit that these run *after* the primary grid, so this is a
    separate entry point rather than an option on :func:`make_primary`. Where a
    regime has no redundancy (``independent``), each signal feature is its own
    singleton class, and class recall degenerates to ordinary recall.
    """
    if regime not in SECONDARY_REGIMES:
        raise ValueError(f"regime must be one of {SECONDARY_REGIMES}")
    rng = np.random.default_rng(seed)

    if regime == "independent":
        p_signal, p_null = kwargs.get("p_signal", 5), kwargs.get("p_null", 25)
        X = rng.normal(size=(n, p_signal + p_null))
        beta = np.linspace(1.0, 0.5, p_signal)
        eta = X[:, :p_signal] @ beta
        classes = [[j] for j in range(p_signal)]

    elif regime == "weak":
        p_signal, p_null = kwargs.get("p_signal", 4), kwargs.get("p_null", 30)
        X = rng.normal(size=(n, p_signal + p_null))
        beta = np.full(p_signal, kwargs.get("beta", 0.18))
        eta = X[:, :p_signal] @ beta
        classes = [[j] for j in range(p_signal)]

    elif regime == "nonlinear":
        p_null = kwargs.get("p_null", 25)
        X = rng.normal(size=(n, 3 + p_null))
        eta = np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 - np.abs(X[:, 2])
        classes = [[0], [1], [2]]

    elif regime == "interaction":
        p_null = kwargs.get("p_null", 25)
        X = rng.normal(size=(n, 2 + p_null))
        # Product term only: each factor has ~zero marginal association with y.
        eta = 1.5 * X[:, 0] * X[:, 1]
        classes = [[0], [1]]

    elif regime == "mixed":
        p_null = kwargs.get("p_null", 20)
        cont = rng.normal(size=(n, 2))
        cat = rng.integers(0, 3, size=(n, 2)).astype(float)
        nulls_c = rng.normal(size=(n, p_null // 2))
        nulls_k = rng.integers(0, 4, size=(n, p_null - p_null // 2)).astype(float)
        X = np.column_stack([cont, cat, nulls_c, nulls_k])
        eta = 1.0 * cont[:, 0] + 0.8 * (cat[:, 0] == 1) - 0.8 * (cat[:, 0] == 2)
        classes = [[0], [2]]

    else:  # p_ge_n
        p = kwargs.get("p", max(2 * n, 60))
        n_classes, class_size = kwargs.get("n_classes", 3), kwargs.get("class_size", 2)
        X = rng.normal(size=(n, p))
        eta = np.zeros(n)
        classes = []
        col = 0
        for h in range(n_classes):
            U = rng.normal(size=n)
            eta += (1.0 - 0.2 * h) * U
            members = []
            for _ in range(class_size):
                X[:, col] = 0.9 * U + np.sqrt(1 - 0.81) * rng.normal(size=n)
                members.append(col)
                col += 1
            classes.append(members)

    snr = kwargs.get("snr", 2.0)
    y = _scale_noise(eta, snr, rng) if task == "regression" else _binarize(
        eta * np.sqrt(snr) / max(np.std(eta), 1e-12), rng
    )
    names = [f"x{j}" for j in range(X.shape[1])]
    for h, members in enumerate(classes):
        for c, j in enumerate(members):
            names[j] = f"sig{h + 1}_{c + 1}"

    return SimData(
        X=X,
        y=y,
        players=Players.singletons(X.shape[1], names),
        signal_classes=SignalClasses.from_lists(classes),
        task=task,
        regime=regime,
        params={"n": n, "seed": seed, "task": task, "snr": snr, **kwargs},
    )
