import numpy as np

from pgfs.knockoffs import select_knockoff_columns


def test_knockoff_selection_is_deterministic_and_reports_assumptions() -> None:
    rng = np.random.default_rng(7)
    design = rng.normal(size=(80, 4))
    outcome = 2 * design[:, 0] + rng.normal(scale=0.2, size=80)

    first = select_knockoff_columns(design, outcome, seed=11)
    second = select_knockoff_columns(design, outcome, seed=11)

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[1]["target_fdr"] == 0.1
    assert np.isfinite(first[1]["min_eigenvalue"])
