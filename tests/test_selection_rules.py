"""Check policy propagation and truthful class mapping after feature reordering."""

import json
from pathlib import Path

import numpy as np
import pytest

from pgfs.metrics import class_recall
from pgfs.nested import nested_evaluate
from pgfs.selection import one_se_choice
from pgfs.study import make_simulation, make_spec


def config():
    return json.loads(Path("experiments/selection-one-se.json").read_text())


def test_shuffled_data_preserves_outcomes_feature_names_and_truth():
    sim = config()["simulation"]
    design = {"n": 80, "rho": .5, "snr": .25, "seed": 301}
    original = make_simulation(dict(sim, feature_order="original"), design)
    shuffled = make_simulation(sim, design)
    repeat = make_simulation(sim, design)
    permutation = shuffled.params["column_permutation"]
    assert permutation != list(range(original.p))
    np.testing.assert_array_equal(shuffled.X, original.X[:, permutation])
    np.testing.assert_array_equal(shuffled.y, original.y)
    np.testing.assert_array_equal(shuffled.X, repeat.X)
    assert shuffled.players.names == tuple(original.players.names[j] for j in permutation)
    for old, new in zip(original.signal_classes.classes, shuffled.signal_classes.classes):
        assert {original.players.names[j] for j in old} == {shuffled.players.names[j] for j in new}
        assert class_recall(new, shuffled.signal_classes) == pytest.approx(1/3)
    assert class_recall(set(range(shuffled.p)) - shuffled.signal_classes.members, shuffled.signal_classes) == 0


def test_configured_policy_changes_selection_on_known_loss_path():
    c = config()
    losses = np.array([[1.1, .5], [1.1, 1.5]])
    for flag, expected in [(True, 1), (False, 2)]:
        spec = make_spec(dict(c, one_se_rule=flag), 301, [1, 2], "regression")
        assert one_se_choice(spec.candidate_k, losses, spec.one_se_rule)[0] == expected
    with pytest.raises(TypeError, match="boolean"):
        make_spec(dict(c, one_se_rule="false"), 301, [1, 2], "regression")


def test_nested_policies_share_rankings_paths_and_reference_but_apply_own_rule():
    c = config()
    design = {"n": 80, "rho": .5, "snr": .25, "seed": 301}
    data = make_simulation(c["simulation"], design)
    results = []
    for flag in (True, False):
        spec = make_spec(dict(c, one_se_rule=flag), 301, list(range(1, 7)), "regression")
        spec = spec.replace(n_contexts=1, n_shadows=1)
        results.append(nested_evaluate(data.X, data.y, data.players, spec, n_outer_folds=2, n_inner_folds=2))
    for one, arg in zip(results[0].folds, results[1].folds):
        assert one.ranking == arg.ranking
        np.testing.assert_array_equal(one.k_selection.fold_losses, arg.k_selection.fold_losses)
        assert one.outer_loss_all_features == arg.outer_loss_all_features
        assert arg.k_chosen == arg.k_selection.k_min
        assert one.k_chosen <= arg.k_chosen
