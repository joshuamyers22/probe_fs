"""Policy comparisons must use paired inputs and common valid recall endpoints."""

import copy
import runpy

import pytest


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return runpy.run_path("experiments/analyze_selection_rules.py")


def pair(seed, valid_one=True, valid_arg=True, effect_one=.1, effect_arg=.2):
    cells = []
    for flag, valid, effect, k in [(True, valid_one, effect_one, 1), (False, valid_arg, effect_arg, 2)]:
        reports = {}
        for method in ("gated", "ungated"):
            reports[method] = {
                "spec": {"one_se_rule": flag, "seed": seed},
                "ranking_diagnostics": [{"fold": 0, "ranking": [0, 1], "gated_score": [2, 1]}],
                "folds": [{"fold": 0, "outer_loss_all_features": 1,
                           "k_chosen": k, "selected": ["s0", "s1"][:k],
                           "k_selection": {"candidate_k": [1, 2], "mean_inner_loss": [1.1, 1],
                                           "se_at_min": .5, "k_chosen": k, "k_argmin": 2}}],
            }
        cells.append({
            "design": {"dataset": "simulation", "n": 100, "rho": .5, "snr": 2, "contexts": 2, "seed": seed},
            "data": {"params": {"seed": seed}}, "reports": reports,
            "comparison": {"comparison_valid": valid, "difference": effect},
            "rows": [{"method": m, "mean_outer_loss": 2 if flag else 1,
                      "mean_selected_size": k} for m in ("gated", "ungated")],
        })
    return cells


def test_recovered_and_lost_endpoints_do_not_enter_common_recall_effect(analysis):
    pairs = [pair(1), pair(2, False, True, .9, -.9),
             pair(3, True, False, -.8, .8), pair(4, False, False)]
    row = analysis["paired_summary"]([p[0] for p in pairs], [p[1] for p in pairs])[0]
    assert (row["both_valid"], row["recovered"], row["lost"], row["neither_valid"]) == (1, 1, 1, 1)
    assert row["common_recall_effect_change_n"] == 1
    assert row["common_recall_effect_change_mean"] == pytest.approx(.1)
    assert row["common_recall_effect_change_low"] is None
    assert row["validity_difference_mean"] == 0
    assert row["gated_loss_change_n"] == 4
    assert row["gated_loss_change_mean"] == -1
    assert row["ungated_size_change_mean"] == 1


def test_no_common_valid_seeds_means_no_policy_recall_estimate(analysis):
    a, b = pair(1, False, True)
    row = analysis["paired_summary"]([a], [b])[0]
    assert row["common_recall_effect_change_mean"] is None
    assert row["common_recall_effect_change_n"] == 0
    assert row["recovered"] == 1


@pytest.mark.parametrize("corruption", ["ranking", "path", "chosen", "policy", "data"])
def test_policy_validation_rejects_unpaired_or_misapplied_results(analysis, corruption):
    a, b = pair(1)
    b = copy.deepcopy(b)
    report = b["reports"]["gated"]
    if corruption == "ranking":
        report["ranking_diagnostics"][0]["ranking"] = [1, 0]
    elif corruption == "path":
        report["folds"][0]["k_selection"]["mean_inner_loss"] = [2, 1]
    elif corruption == "chosen":
        report["folds"][0]["k_chosen"] = 1
    elif corruption == "policy":
        report["spec"]["one_se_rule"] = True
    else:
        b["data"]["params"]["seed"] = 2
    with pytest.raises(ValueError):
        analysis["validate_pair"](a, b)


def test_duplicate_or_missing_design_cannot_inflate_paired_sample(analysis):
    a, b = pair(1)
    with pytest.raises(ValueError, match="missing or duplicate"):
        analysis["paired_summary"]([a, a], [b])
    with pytest.raises(ValueError, match="missing or duplicate"):
        analysis["paired_summary"]([a], [])
