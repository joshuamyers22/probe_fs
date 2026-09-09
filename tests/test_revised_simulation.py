"""Regression checks for configured strengths and endpoint feasibility."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from pgfs.simulate import make_primary
from pgfs.study import make_simulation

preflight = runpy.run_path("experiments/check_simulation_feasibility.py")["check"]


def config():
    return json.loads(Path("experiments/revised-simulation.json").read_text())


def test_runner_uses_explicit_strengths_without_changing_covariates():
    sim = config()["simulation"]
    design = {"n": 100, "rho": 0.5, "snr": 0.25, "seed": 201}
    actual = make_simulation(sim, design)
    expected = make_primary(**design, n_classes=3, class_size=2, n_null_blocks=1,
                            null_block_size=3, n_independent_nulls=6, betas=(1, .3, .15))
    original = make_simulation({k: v for k, v in sim.items() if k != "betas"}, design)
    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.y, expected.y)
    np.testing.assert_array_equal(actual.X, original.X)
    assert not np.array_equal(actual.y, original.y)
    assert tuple(actual.params["betas"]) == (1, .3, .15)
    assert tuple(original.params["betas"]) == (1, .85, .7)


def test_revised_endpoint_has_partial_and_full_coverage_headroom():
    for regime in preflight(config())["regimes"]:
        assert regime["feasible_recall_values"] == [2/3, 1]
        rows = {tuple(a["counts"]): a for a in regime["allocations"]}
        assert rows[(2, 1, 0)]["within_margin"]
        assert rows[(2, 1, 1)]["within_margin"]
        assert not rows[(2, 0, 0)]["within_margin"]
        # Independently compute the conditional covariance for (2, 1, 0).
        rho = regime["rho"]
        covariance = np.array([[1, rho, 0], [rho, 1, 0], [0, 0, 1]])
        cross = np.sqrt(rho)*np.array([1, 1, .3])
        residual = 1+.3**2+.15**2-cross@np.linalg.solve(covariance, cross)
        reference = (1+.3**2+.15**2)*(1-rho)/(1+rho)
        assert rows[(2, 1, 0)]["excess_mse"] == pytest.approx(residual-reference)


def test_preflight_rejects_original_recall_ceiling():
    c = config()
    c["simulation"]["betas"] = [1, .85, .7]
    with pytest.raises(ValueError, match="No feasible partial-coverage"):
        preflight(c)
