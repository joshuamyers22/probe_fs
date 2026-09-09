"""Constraint, pairing, and provenance checks for context-control summaries."""

import copy
import json
import runpy
from pathlib import Path

import pytest

from pgfs.datasets import sha256
from pgfs.study import clean, fingerprint, make_simulation, write_json


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return runpy.run_path("experiments/analyze_context_controls.py")


def record(seed, valid, difference, matched=True):
    return {"n": 300, "rho": .5, "snr": .25, "contexts": 2, "control": "global", "seed": seed,
            "comparison_valid": valid, "budget_matched": matched, "candidate_passes": valid,
            "control_passes": valid, "difference": difference, "meets_minimum_effect": valid and difference >= .05,
            "measured_budget_imbalance": 0 if matched else .2,
            "reason": "ok" if valid else "failed", "loss_difference": -1, "size_difference": 1,
            "jaccard_between_methods": .5}


def test_failed_endpoints_and_mismatched_budgets_do_not_enter_recall_mean(analysis):
    rows = [record(1, True, .1), record(2, False, .9), record(3, False, -.9, False)]
    r = analysis["aggregate_comparisons"](rows)[0]
    assert (r["seeds"], r["valid"], r["compute_matched"]) == (3, 1, 2)
    assert r["qualified_recall_difference_mean"] == .1
    assert r["qualified_recall_difference_mcse"] is None
    assert r["loss_difference_n"] == 3
    assert r["loss_difference_mean"] == -1
    with pytest.raises(ValueError, match="Duplicate"):
        analysis["aggregate_comparisons"]([rows[0], rows[0]])


def test_empty_valid_subset_is_unavailable_not_zero(analysis):
    r = analysis["aggregate_comparisons"]([record(1, False, .8)])[0]
    assert r["qualified_recall_difference_mean"] is None
    assert r["local_wins"] == 0


@pytest.mark.parametrize("corruption", ["reference", "raw", "fits"])
def test_shared_fit_validation_rejects_unpaired_controls(analysis, corruption):
    c = {"row": {"model_fits": 100}, "report": {
        "folds": [{"fold": 0, "outer_loss_all_features": 1}],
        "ranking_diagnostics": [{"fold": 0, "raw_score": [1, 2]}]}}
    cells = {m: copy.deepcopy(c) for m in ("local-T2", "global-T2")}
    config = {"rungs": [{"contexts": 2}]}
    analysis["validate_shared_fits"](cells, config)
    if corruption == "reference":
        cells["global-T2"]["report"]["folds"][0]["outer_loss_all_features"] = 2
    elif corruption == "raw":
        cells["global-T2"]["report"]["ranking_diagnostics"][0]["raw_score"][0] = 3
    else:
        cells["global-T2"]["row"]["model_fits"] = 99
    with pytest.raises(ValueError):
        analysis["validate_shared_fits"](cells, config)


@pytest.mark.parametrize("corruption", ["truth", "spec", "shard", "missing"])
def test_loader_checks_truth_actual_method_and_completeness(analysis, tmp_path, corruption):
    config = json.loads(Path("experiments/context-controls.json").read_text())
    config.update(seeds=[71], driver_sha256=sha256(Path("experiments/run_context_controls.py")))
    config["simulation"].update(n=[50], rho=[.5], snr=[.25])
    identity = {"config": config, "source_hashes": {"test": "test"}}
    parent = fingerprint(identity)
    write_json(tmp_path / "frozen-config.json", config)
    write_json(tmp_path / "provenance.json", {"identity": identity, "fingerprint": parent})
    shard_config = dict(config, parent_fingerprint=parent)
    shard_identity = dict(identity, config=shard_config)
    digest = fingerprint(shard_identity)
    directory = tmp_path / "seed-71"
    directory.mkdir()
    write_json(directory / "provenance.json", {"identity": shard_identity, "fingerprint": digest})
    design = {"dataset": "simulation", "n": 50, "rho": .5, "snr": .25, "seed": 71}
    data = make_simulation(config["simulation"], design)
    metadata = clean({"params": data.params, "player_names": data.players.names,
                      "signal_classes": [sorted(c) for c in data.signal_classes.classes]})
    paths = []
    for name, spec in analysis["specs"](config, 71).items():
        cell = {"design": design, "method": name, "run_fingerprint": digest, "data": metadata,
                "spec": spec.as_dict(), "report": {"spec": spec.as_dict(), "folds": [
                    {"fold": f, "k_chosen": 1, "selected": [data.players.names[0]],
                     "k_selection": {"candidate_k": [1, 2, 3, 4, 5, 6], "mean_inner_loss": [1, 2, 3, 4, 5, 6],
                                     "k_argmin": 1, "k_chosen": 1, "se_at_min": 0}}
                    for f in range(3)]}}
        path = directory / f"cell-{name}.json"
        write_json(path, cell)
        paths.append(path)
    assert len(analysis["load_complete"](tmp_path)[0]) == 1
    if corruption == "missing":
        paths[0].rename(directory / "hidden.json")
    elif corruption == "shard":
        shard_identity["source_hashes"] = {"test": "changed"}
        write_json(directory / "provenance.json", {"identity": shard_identity, "fingerprint": fingerprint(shard_identity)})
    else:
        cell = json.loads(paths[0].read_text())
        if corruption == "truth":
            cell["data"]["signal_classes"][0] = [0]
        else:
            cell["report"]["spec"]["one_se_rule"] = False
        write_json(paths[0], cell)
    with pytest.raises(ValueError):
        analysis["load_complete"](tmp_path)
