"""Independent covariance check of the noisy-proxy population risk diagnosis."""

import runpy

import numpy as np
import pytest
from scipy.linalg import block_diag

audit = runpy.run_path("experiments/audit_endpoint.py")


@pytest.mark.parametrize("rho", [0.5, 0.9])
@pytest.mark.parametrize("counts", [(0, 1, 2), (1, 1, 1), (2, 2, 2)])
def test_residual_variance_matches_covariance_schur_complement(rho, counts):
    betas = np.array([1.0, 0.85, 0.7])
    covariance = block_diag(*[(1-rho)*np.eye(n)+rho*np.ones((n, n)) for n in counts if n])
    cross = np.concatenate([np.repeat(b*np.sqrt(rho), n) for b, n in zip(betas, counts)])
    expected = float(betas @ betas - cross @ np.linalg.solve(covariance, cross))
    assert audit["residual_signal_variance"](counts, betas, rho) == pytest.approx(expected)


def test_missing_five_excludes_a_feasible_population_size_at_high_correlation():
    rows = audit["theory"]([1, 0.85, 0.7], 0.9, 0.05, [1, 2, 3, 4, 6])
    by_k = {r["k"]: r for r in rows}
    assert not by_k[4]["within_margin"]
    assert by_k[5]["within_margin"] and not by_k[5]["in_original_path"]
    assert by_k[3]["oracle_excess_mse"] == pytest.approx(0.10480263157894736)


def test_at_lower_correlation_even_best_five_exceeds_original_margin():
    rows = audit["theory"]([1, 0.85, 0.7], 0.5, 0.05, [1, 2, 3, 4, 6])
    assert [r["k"] for r in rows if r["within_margin"]] == [6]
