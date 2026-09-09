"""Budget and fit-sharing invariants for the context-control study."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from pgfs.experiment_models import StudyConstraints
from pgfs.experiments import run_spec
from pgfs.study import detailed_report, make_simulation

runner = runpy.run_path("experiments/run_context_controls.py")


def config():
    return json.loads(Path("experiments/context-controls.json").read_text())


def test_loco_spends_matched_budget_on_splits_and_retains_cheap_controls():
    c = config()
    specs = runner["specs"](c, 77)
    costs = runner["preflight"](c)
    assert len(specs) == 8
    for t in (2, 4):
        a, b = costs[f"local-T{t}"], costs[f"loco-matched-T{t}"]
        assert abs(a-b)/max(a, b) <= .05
        assert specs[f"loco-matched-T{t}"].effective_n_contexts == 1
        assert specs[f"loco-matched-T{t}"].n_importance_splits == t
        assert costs[f"global-T{t}"] == a
    assert costs["loco-gated-B1"] == 61
    assert costs["loco-ungated-B1"] == 16
    c["rungs"][0]["loco_splits"] = 1
    with pytest.raises(ValueError, match="infeasible"):
        runner["preflight"](c)


def test_budget_mismatch_cannot_be_qualified_even_if_both_endpoints_pass():
    endpoint = {"constraints_met": True, "class_recall": 1}
    reports = {"a": {"primary_endpoint": endpoint, "main_results": {"model_fits_total": 100}},
               "b": {"primary_endpoint": endpoint, "main_results": {"model_fits_total": 50}}}
    result = runner["comparison"]("a", "b", reports, {"a": 100, "b": 50}, .05)
    assert result["endpoint_constraints_met"]
    assert not result["comparison_valid"]
    assert not result["meets_minimum_effect"]
    assert result["reason"] == "budget_not_matched"


def test_global_scope_preserves_raw_scores_fits_and_outer_reference():
    c = config()
    c.update(outer_folds=2, inner_folds=2, shadows=1)
    data = make_simulation(c["simulation"], {"n": 60, "rho": .5, "snr": .25, "seed": 77})
    specs = runner["specs"](c, 77)
    results = [run_spec(data, specs[name], StudyConstraints(.05, 6), 2, 2)
               for name in ("local-T2", "global-T2")]
    reports = [detailed_report(r) for r in results]
    assert reports[0]["main_results"]["model_fits_total"] == reports[1]["main_results"]["model_fits_total"]
    for a, b in zip(results[0].folds, results[1].folds):
        np.testing.assert_array_equal(a.raw_score, b.raw_score)
        assert a.outer_loss_all_features == b.outer_loss_all_features
    altered = copy.deepcopy(reports[0]["spec"])
    altered.update(label=reports[1]["spec"]["label"], shadow_scope="global")
    assert altered == reports[1]["spec"]
