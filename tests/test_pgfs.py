"""Tests for the properties the specification actually pins down.

Each test names the section it is protecting. The interesting ones are the
leakage test (Section 9) and the pairing test (Sections 4 and 10) - those two
guard claims that are easy to make and hard to notice breaking.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from pgfs import (
    MethodSpec,
    Players,
    SignalClasses,
    estimate_importance,
    make_primary,
    nested_evaluate,
)
from pgfs.budget import (
    expected_importance_fits,
    match_budget,
    matched_ungated_contexts,
    spec_importance_fits,
)
from pgfs.contexts import sample_context
from pgfs.importance import apply_gate, rank_players
from pgfs.metrics import class_recall, primary_endpoint, selection_report
from pgfs.nested import OUTER_CV_SEED_OFFSET, finalize_for_deployment
from pgfs.selection import one_se_choice
from pgfs.shadows import shadow_blocks, shadow_threshold


def small_data(n=120, seed=0):
    return make_primary(
        n=n, n_classes=2, class_size=2, n_null_blocks=1, null_block_size=2,
        n_independent_nulls=2, snr=4.0, seed=seed,
    )


def base_spec(**kw):
    defaults = {
        "learner": Ridge(alpha=1.0),
        "loss": "mse",
        "n_contexts": 2,
        "n_importance_splits": 1,
        "n_shadows": 2,
        "candidate_k": (1, 2, 3, 4),
        "seed": 7,
    }
    defaults.update(kw)
    return MethodSpec(**defaults)


# -- Section 7: the gate -----------------------------------------------------
def test_soft_gate_formula():
    delta = np.array([1.0, 1.0, -0.5, 0.2])
    tau = np.array([0.4, -0.3, 0.1, 0.9])
    got = apply_gate(delta, tau, "soft")
    # max(0, delta - max(0, tau)); a negative tau must not credit the player.
    assert np.allclose(got, [0.6, 1.0, 0.0, 0.0])


def test_hard_and_none_gates():
    delta = np.array([1.0, 0.05])
    tau = np.array([0.5, 0.5])
    assert np.allclose(apply_gate(delta, tau, "hard"), [1.0, 0.0])
    assert np.allclose(apply_gate(delta, tau, "none"), delta)


def test_gate_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown gate"):
        apply_gate(np.array([1.0]), np.array([0.5]), "experimental")


def test_ranking_breaks_ties_by_player_order():
    assert rank_players(np.array([0.5, 0.5, 0.9, 0.5])) == (2, 0, 1, 3)


# -- Section 4: contexts -----------------------------------------------------
def test_permutation_contexts_are_nested_prefixes():
    ctx = sample_context("permutation", 5, np.random.default_rng(0))
    assert [len(S) for _, S in ctx] == [0, 1, 2, 3, 4]
    for i, (j, S) in enumerate(ctx):
        assert j not in S
        assert set(S) == {p for p, _ in ctx[:i]}


def test_full_conditioning_leaves_one_out():
    ctx = sample_context("full", 4, np.random.default_rng(0))
    assert len(ctx) == 4
    for j, S in ctx:
        assert set(S) == set(range(4)) - {j}


def test_full_conditioning_forces_single_context():
    assert base_spec(context_kind="full", n_contexts=9).effective_n_contexts == 1


# -- Section 6: shadows ------------------------------------------------------
def test_shadow_is_dimension_matched_and_row_permuted():
    X = np.arange(40, dtype=float).reshape(10, 4)
    tr, va = np.arange(0, 7), np.arange(7, 10)
    cols = np.array([1, 2])
    Z_tr, Z_va = shadow_blocks(X, tr, va, cols, np.random.default_rng(3))
    assert Z_tr.shape == (7, 2) and Z_va.shape == (3, 2)
    # Same multiset of rows, reordered.
    assert sorted(map(tuple, Z_tr)) == sorted(map(tuple, X[np.ix_(tr, cols)]))


def test_block_shadow_uses_one_joint_row_permutation():
    # Within-block dependence must survive: a block permuted column-by-column
    # would break the row alignment that carries it.
    X = np.column_stack([np.arange(20.0), np.arange(20.0) * 10])
    Z_tr, _ = shadow_blocks(X, np.arange(16), np.arange(16, 20), np.array([0, 1]),
                            np.random.default_rng(1))
    assert np.allclose(Z_tr[:, 1], Z_tr[:, 0] * 10)


def test_shadow_threshold_quantile_convention():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert shadow_threshold(d, 0.9, "higher") == 4.0
    assert shadow_threshold(d, 0.5, "higher") == 2.0


# -- Sections 4 and 10: paired randomness -----------------------------------
def test_gated_and_ungated_share_splits_and_permutations():
    data = small_data()
    gated = base_spec(gate="soft", n_contexts=3)
    ungated = base_spec(gate="none", n_contexts=3)
    g = estimate_importance(data.X, data.y, data.players, gated)
    u = estimate_importance(data.X, data.y, data.players, ungated)
    # Identical seeds and context indices => identical raw contributions.
    assert np.allclose(g.delta, u.delta)
    assert np.allclose(g.raw_score, u.raw_score)
    # And the ungated arm spent nothing on shadows.
    assert u.counter.shadow == 0 and g.counter.shadow > 0


def test_ungated_extra_contexts_extend_rather_than_replace():
    """A larger T reuses the smaller T's contexts, so budgets nest cleanly."""
    data = small_data()
    a = estimate_importance(data.X, data.y, data.players, base_spec(gate="none", n_contexts=2))
    b = estimate_importance(data.X, data.y, data.players, base_spec(gate="none", n_contexts=5))
    assert np.allclose(a.delta, b.delta[:, :2, :])


def test_estimation_is_deterministic():
    data = small_data()
    spec = base_spec()
    a = estimate_importance(data.X, data.y, data.players, spec)
    b = estimate_importance(data.X, data.y, data.players, spec)
    assert a.ranking == b.ranking and np.allclose(a.gated_score, b.gated_score)


# -- Section 12: global shadow scope reuses the same fits --------------------
def test_regate_changes_scores_without_refitting():
    data = small_data()
    spec = base_spec(n_contexts=2, n_shadows=3)
    res = estimate_importance(data.X, data.y, data.players, spec)
    fits_before = res.counter.importance

    same = res.regate(spec)
    assert np.allclose(same.gated_score, res.gated_score)

    glob = res.regate(spec.replace(shadow_scope="global"))
    assert glob.counter.importance == fits_before  # no new fits
    for b in range(glob.tau.shape[2]):
        assert np.allclose(glob.tau[:, :, b], glob.tau[0, 0, b])  # one tau per split


# -- Section 8: the 1-SE rule ------------------------------------------------
def test_one_se_rule_prefers_the_smaller_k():
    ks = (1, 2, 3, 4)
    # k=3 minimises, but k=2 is well within one standard error of it.
    losses = np.array([
        [1.00, 0.62, 0.50, 0.61],
        [1.02, 0.63, 0.70, 0.64],
        [0.98, 0.61, 0.60, 0.62],
    ])
    chosen, argmin, se = one_se_choice(ks, losses, use_one_se=True)
    assert argmin == 3 and chosen == 2 and se > 0
    assert one_se_choice(ks, losses, use_one_se=False)[0] == 3


# -- Section 10: budget arithmetic ------------------------------------------
@pytest.mark.parametrize("p,T,m", [(14, 2, 3), (20, 4, 5), (60, 8, 3)])
def test_matched_budgets_are_close_in_predicted_fits(p, T, m):
    T_u = matched_ungated_contexts(p, T, m)
    gated = expected_importance_fits(p, T, 1, m)
    ungated = expected_importance_fits(p, T_u, 1, 0)
    assert T_u > T  # the ungated arm really does buy more contexts
    assert abs(gated - ungated) / max(gated, ungated) < 0.05


def test_predicted_fit_count_matches_measured_for_both_arms():
    data = small_data()
    for gate, shadows in (("soft", 2), ("none", 0)):
        spec = base_spec(gate=gate, n_contexts=6, n_shadows=max(shadows, 1),
                         n_importance_splits=2)
        res = estimate_importance(data.X, data.y, data.players, spec)
        predicted = spec_importance_fits(spec, len(data.players))
        # The prediction is an expectation over permutations, so allow slack -
        # but it must not be systematically wrong, which is what the old
        # "every context costs p-1" rule was for the cache-heavy ungated arm.
        assert 0.85 * predicted <= res.counter.importance <= 1.15 * predicted


def test_measured_budgets_match_at_the_paired_rung():
    """The match must survive contact with the actual fit counter (Section 10)."""
    data = small_data(n=150)
    pair = match_budget(base_spec(n_shadows=3), len(data.players), n_contexts_gated=3)
    g = estimate_importance(data.X, data.y, data.players, pair.gated)
    u = estimate_importance(data.X, data.y, data.players, pair.ungated)
    gf, uf = g.counter.importance, u.counter.importance
    assert abs(gf - uf) / max(gf, uf) < 0.15


# -- Section 9: no leakage from the outer test fold --------------------------
def test_outer_test_fold_does_not_influence_selection():
    data = small_data(n=140, seed=2)
    spec = base_spec(n_contexts=2, n_shadows=2, candidate_k=(1, 2, 3))
    kwargs = {
        "n_outer_folds": 3,
        "n_inner_folds": 2,
        "signal_classes": data.signal_classes,
    }

    first = nested_evaluate(data.X, data.y, data.players, spec, **kwargs)
    again = nested_evaluate(data.X, data.y, data.players, spec, **kwargs)
    assert first.per_fold_sets == again.per_fold_sets

    from pgfs.selection import make_cv

    # Corrupt one fold's held-out rows at a time. Those rows are training rows for
    # the *other* folds, so only fold o's own selection is expected to be immune -
    # and it must be exactly immune, since Section 9 lets the outer test fold enter
    # nothing but the final evaluation.
    cv = make_cv(3, spec.task, seed=int(spec.seed) + OUTER_CV_SEED_OFFSET)
    rng = np.random.default_rng(0)
    for o, (_, te) in enumerate(cv.split(data.X, None)):
        y_corrupt = data.y.copy()
        y_corrupt[te] = rng.normal(size=len(te)) * 50.0
        corrupted = nested_evaluate(data.X, y_corrupt, data.players, spec, **kwargs)
        assert corrupted.folds[o].selected == first.folds[o].selected
        assert corrupted.folds[o].k_chosen == first.folds[o].k_chosen
        assert corrupted.folds[o].ranking == first.folds[o].ranking
        # The evaluation, by contrast, sees the corruption.
        assert corrupted.folds[o].outer_loss > first.folds[o].outer_loss


def test_deployment_finalization_requires_external_data():
    data = small_data(n=80)
    with pytest.raises(ValueError, match="external test set"):
        finalize_for_deployment(data.X, data.y, data.players, base_spec())


# -- Section 11: endpoint and constraints -----------------------------------
def test_class_recall_counts_classes_not_members():
    classes = SignalClasses.from_lists([[0, 1, 2], [3, 4], [5]])
    assert class_recall([0, 1, 2], classes) == pytest.approx(1 / 3)
    assert class_recall([0, 3, 5], classes) == 1.0


def test_endpoint_is_nan_when_a_constraint_fails():
    classes = SignalClasses.from_lists([[0, 1], [2, 3]])
    ok = primary_endpoint([0, 2], classes, outer_loss=1.0,
                          outer_loss_all_features=1.0, delta=0.05, k_max=4)
    assert ok["constraints_met"] and ok["endpoint"] == 1.0

    too_lossy = primary_endpoint([0, 2], classes, outer_loss=2.0,
                                 outer_loss_all_features=1.0, delta=0.05, k_max=4)
    assert not too_lossy["loss_constraint_met"] and np.isnan(too_lossy["endpoint"])

    too_big = primary_endpoint([0, 1, 2, 3], classes, outer_loss=1.0,
                               outer_loss_all_features=1.0, delta=0.05, k_max=3)
    assert not too_big["size_constraint_met"] and np.isnan(too_big["endpoint"])


def test_selection_report_separates_false_and_redundant():
    classes = SignalClasses.from_lists([[0, 1], [2, 3]])
    rep = selection_report([0, 1, 2, 9], classes)
    assert rep["n_outside_classes"] == 1 and rep["n_redundant"] == 1


# -- Section 3: players and blocks ------------------------------------------
def test_block_players_are_selected_as_units():
    rng = np.random.default_rng(0)
    n = 150
    U = rng.normal(size=n)
    X = np.column_stack([U + 0.1 * rng.normal(size=n), U + 0.1 * rng.normal(size=n),
                         rng.normal(size=n), rng.normal(size=n)])
    y = 2.0 * U + 0.3 * rng.normal(size=n)
    players = Players.from_blocks([[0, 1], [2], [3]], names=["signal_block", "n1", "n2"])
    spec = base_spec(learner=LinearRegression(), n_contexts=3, n_shadows=3)
    res = estimate_importance(X, y, players, spec)
    assert res.ranking[0] == 0
    assert res.gated_score[0] > res.gated_score[1]


def test_players_reject_overlapping_columns():
    with pytest.raises(ValueError, match="more than one player"):
        Players.from_blocks([[0, 1], [1, 2]])


def test_players_reject_duplicate_negative_and_empty_designs():
    with pytest.raises(ValueError, match="twice"):
        Players.from_blocks([[0, 0]])
    with pytest.raises(ValueError, match="non-negative"):
        Players.from_blocks([[-1]])
    with pytest.raises(ValueError, match="positive"):
        Players.singletons(0)


def test_importance_rejects_malformed_data_and_player_indices():
    spec = base_spec()
    with pytest.raises(ValueError, match="same number of rows"):
        estimate_importance(np.zeros((4, 2)), np.zeros(3), Players.singletons(2), spec)
    with pytest.raises(ValueError, match="at least two"):
        estimate_importance(np.zeros((1, 2)), np.zeros(1), Players.singletons(2), spec)
    with pytest.raises(ValueError, match="valid columns"):
        estimate_importance(np.zeros((4, 2)), np.zeros(4), Players.from_blocks([[2]]), spec)


def test_regate_rejects_changes_that_require_refitting():
    data = small_data(n=50)
    spec = base_spec()
    result = estimate_importance(data.X, data.y, data.players, spec)
    # Threshold-only changes are safe and reuse exactly the same fitted shadows.
    result.regate(spec.replace(shadow_quantile=0.5, shadow_scope="global"))
    with pytest.raises(ValueError, match="fitted design"):
        result.regate(spec.replace(seed=999))
    with pytest.raises(ValueError, match="fitted design"):
        result.regate(spec.replace(n_shadows=3))


def test_signal_classes_reject_ambiguous_ground_truth():
    with pytest.raises(ValueError, match="overlap"):
        SignalClasses.from_lists([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="non-empty"):
        SignalClasses.from_lists([[]])


def test_spec_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        base_spec(shadow_quantile=1.0)
    with pytest.raises(ValueError):
        base_spec(gate="squishy")
    with pytest.raises(ValueError):
        base_spec(shadow_scope="regional")
    with pytest.raises(ValueError, match="integers"):
        base_spec(n_contexts=1.5)
    with pytest.raises(ValueError, match="positive sizes"):
        base_spec(candidate_k=(1.5,))


def test_study_constraints_reject_nonsensical_values():
    from pgfs.experiments import StudyConstraints

    with pytest.raises(ValueError, match="non-negative"):
        StudyConstraints(-0.1, 3)
    with pytest.raises(ValueError, match="positive integer"):
        StudyConstraints(0.1, 0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StudyConstraints(0.1, 3, 1.1)


# -- Section 16: decision rules ---------------------------------------------
def _fake_comparison(entries):
    """entries: list of (budget, recall_gated, recall_ungated, valid)."""
    from pgfs.experiments import ComparisonTable

    table = ComparisonTable()
    for budget, a, b, valid in entries:
        diff = a - b
        table.comparisons.append({
            "budget": budget,
            "class_recall_a": a,
            "class_recall_b": b,
            "difference": diff,
            "min_effect": 0.05,
            "both_satisfy_constraints": valid,
            "comparison_valid": valid,
            "reason": "ok" if valid else "comparator_failed_constraints",
            "meets_minimum_effect": bool(valid and diff >= 0.05),
        })
    return table


def _constraints():
    from pgfs.experiments import StudyConstraints

    return StudyConstraints(noninferiority_margin=0.05, k_max=6, min_effect=0.05)


def test_decision_discontinues_only_on_valid_evidence():
    from pgfs.experiments import decision_criteria

    v = decision_criteria(_fake_comparison([(1, 0.5, 0.5, True), (2, 0.5, 0.52, True)]),
                          _constraints())
    assert v["discontinue_broad_study"] and not v["inconclusive"]


def test_decision_is_inconclusive_when_constraints_broke():
    from pgfs.experiments import decision_criteria

    # A large recall advantage that never reached a valid endpoint comparison must
    # not be recorded as a failure of gating.
    v = decision_criteria(_fake_comparison([(1, 0.9, 0.5, False), (2, 0.9, 0.5, False)]),
                          _constraints())
    assert v["inconclusive"] and not v["discontinue_broad_study"]
    assert not v["confirmatory"]


def test_decision_marks_partial_advantage_exploratory():
    from pgfs.experiments import decision_criteria

    v = decision_criteria(_fake_comparison([(1, 0.9, 0.5, True), (2, 0.5, 0.5, True)]),
                          _constraints())
    assert v["exploratory_only"] and not v["confirmatory"]

    v2 = decision_criteria(_fake_comparison([(1, 0.9, 0.5, True), (2, 0.8, 0.6, True)]),
                           _constraints())
    assert v2["confirmatory"] and not v2["exploratory_only"]


def test_direct_alternatives_share_the_protocol():
    from pgfs.experiments import direct_alternative_specs

    base = base_spec()
    specs = direct_alternative_specs(base)
    for s in specs.values():
        assert s.seed == base.seed
        assert s.loss == base.loss
        assert s.candidate_k == base.candidate_k
        assert s.n_importance_splits == base.n_importance_splits
    assert specs["loco_gated"].effective_n_contexts == 1
    assert specs["ungated_permutation"].effective_n_shadows == 0


def test_infeasible_budget_cannot_support_primary_claim():
    from pgfs.experiments import matched_compute_comparison

    data = small_data(n=50)
    table = matched_compute_comparison(
        data, base_spec(n_shadows=1), _constraints(), gated_contexts=(1,),
        n_outer_folds=2, n_inner_folds=2,
    )
    assert not table.pairs[0].feasible
    assert not table.comparisons[0]["comparison_valid"]
    assert table.comparisons[0]["reason"] == "budget_not_matched"
