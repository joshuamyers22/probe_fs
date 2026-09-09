"""Failed constraints must never be silently converted into endpoint evidence."""

import runpy

import pytest

from pgfs.study import fingerprint, write_json

analysis = runpy.run_path("experiments/analyze_simulation_batch.py")


def cell(seed, valid, difference):
    endpoint = {"constraints_met": valid, "loss_constraint_met": valid,
                "size_constraint_met": True}
    return {
        "design": {"n": 300, "rho": 0.5, "snr": 2, "contexts": 2, "seed": seed},
        "comparison": {"comparison_valid": valid, "budget_matched": True,
                       "difference": difference, "meets_minimum_effect": valid and difference >= 0.05,
                       "reason": "ok" if valid else "both_failed_constraints"},
        "reports": {name: {"primary_endpoint": endpoint} for name in ["gated", "ungated"]},
        "rows": [{"method": "gated", "mean_outer_loss": 2, "mean_selected_size": 3},
                 {"method": "ungated", "mean_outer_loss": 3, "mean_selected_size": 4}],
    }


def test_qualified_effect_excludes_invalid_seeds_but_denominator_keeps_them():
    result = analysis["aggregate"]([cell(1, True, 0.1), cell(2, False, 0.9)])[0]
    assert result["seeds"] == 2
    assert result["valid_fraction"] == 0.5
    assert result["qualified_recall_difference_mean"] == 0.1
    assert result["qualified_recall_difference_mcse"] is None
    assert result["diagnostic_raw_recall_difference_mean"] == 0.5
    assert result["loss_difference_n"] == 2
    assert result["gated_loss_failures"] == 1


def test_zero_valid_seeds_is_unavailable_not_zero_effect():
    result = analysis["aggregate"]([cell(1, False, 0.9)])[0]
    assert result["qualified_recall_difference_mean"] is None
    assert result["valid_clearing_effect"] == 0


def test_duplicate_seed_cannot_inflate_replication_count():
    with pytest.raises(ValueError, match="Duplicate seed"):
        analysis["aggregate"]([cell(1, True, 0), cell(1, True, 0)])


def test_mcse_uses_independent_seed_count():
    result = analysis["mean_interval"]([0.0, 0.2])
    assert result["mean"] == pytest.approx(0.1)
    assert result["mcse"] == pytest.approx(0.1)
    assert result["low"] < 0 < result["high"]


def test_tie_audit_counts_only_ties_crossing_the_selected_boundary():
    c = cell(1, True, 0)
    for method, k in [("gated", 1), ("ungated", 2)]:
        c["reports"][method]["folds"] = [{"fold": 0, "k_chosen": k}]
        c["reports"][method]["ranking_diagnostics"] = [
            {"fold": 0, "ranking": [0, 1, 2], "gated_score": [1, 1, 0]}]
    result = analysis["tie_diagnostics"]([c])
    assert result["valid"]["gated"]["boundary_ties"] == 1
    assert result["valid"]["ungated"]["boundary_ties"] == 0
    assert result["all"]["gated"]["selected_zero_score"] == 0


@pytest.mark.parametrize("corruption", ["config", "betas", "path", "shard"])
def test_complete_loader_rejects_mismatched_frozen_or_fitted_inputs(tmp_path, corruption):
    config = {"seeds": [1], "budgets": [2], "simulation": {
        "n": [300], "rho": [0.5], "snr": [2], "betas": [1, .3, .15], "candidate_k": [1, 2, 3]}}
    identity = {"config": config}
    parent = fingerprint(identity)
    write_json(tmp_path / "provenance.json", {"identity": identity, "fingerprint": parent})
    write_json(tmp_path / "frozen-config.json", config)
    directory = tmp_path / "seed-1"
    directory.mkdir()
    shard_identity = {"config": {"parent_fingerprint": parent}}
    digest = fingerprint(shard_identity)
    write_json(directory / "provenance.json", {"identity": shard_identity, "fingerprint": digest})
    c = cell(1, True, 0)
    c["design"]["dataset"] = "simulation"
    c["run_fingerprint"] = digest
    c["data"] = {"params": {"betas": [1, .3, .15]}}
    for report in c["reports"].values():
        report["folds"] = [{"k_selection": {"candidate_k": [1, 2, 3]}}]
    path = directory / "cell-test.json"
    write_json(path, c)
    assert len(analysis["load_complete"](tmp_path)[0]) == 1
    if corruption == "config":
        config["simulation"]["betas"] = [1, .85, .7]
        write_json(tmp_path / "frozen-config.json", config)
    elif corruption == "shard":
        shard_identity["config"]["extra"] = "changed"
        write_json(directory / "provenance.json", {"identity": shard_identity, "fingerprint": digest})
    else:
        if corruption == "betas":
            c["data"]["params"]["betas"] = [1, .85, .7]
        else:
            c["reports"]["gated"]["folds"][0]["k_selection"]["candidate_k"] = [1, 3]
        write_json(path, c)
    with pytest.raises(ValueError):
        analysis["load_complete"](tmp_path)
