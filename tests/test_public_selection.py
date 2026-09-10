"""Replay must equal full nested execution and preserve frozen selections."""

import copy
import importlib

import numpy as np
import pytest

from pgfs.datasets import PublicData
from pgfs.learners import FitCounter
from pgfs.nested import OUTER_CV_SEED_OFFSET, nested_evaluate
from pgfs.players import Players
from pgfs.selection import make_cv
from pgfs.study import (
    clean,
    detailed_report,
    fingerprint,
    initialize,
    make_spec,
    write_json,
)


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return importlib.import_module("public_selection")


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize("gate", ["soft", "none"])
def test_replay_equals_full_nested_policies(replay, task, gate):
    rng = np.random.default_rng(71)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] + .3*X[:, 1] + rng.normal(size=60)
    if task == "binary":
        y = (y > np.median(y)).astype(int)
    X[::9, 2] = np.nan
    config = {"importance_splits": 1, "shadows": 1, "quantile": .9, "one_se_rule": True}
    spec = make_spec(config, 37, [1, 2, 4], task).replace(n_contexts=1, gate=gate)
    players = Players.singletons(4)
    one = nested_evaluate(X, y, players, spec, n_outer_folds=2, n_inner_folds=2)
    arg = nested_evaluate(X, y, players, spec.replace(one_se_rule=False), n_outer_folds=2, n_inner_folds=2)
    report = clean(detailed_report(one))
    counter = FitCounter()
    cv = make_cv(2, task, spec.seed+OUTER_CV_SEED_OFFSET)
    for f, (train, test) in enumerate(cv.split(X, y if task == "binary" else None)):
        old = dict(report["folds"][f], selected_indices=list(one.folds[f].selected))
        actual = replay.replay_fold(X, y, train, test, spec, old, report["ranking_diagnostics"][f], counter)
        for policy, expected in (("one_se", one), ("argmin", arg)):
            assert actual["selected"][policy] == list(expected.folds[f].selected)
            assert actual["loss"][policy] == expected.folds[f].outer_loss
        assert one.folds[f].ranking == arg.folds[f].ranking
        np.testing.assert_array_equal(one.folds[f].k_selection.fold_losses, arg.folds[f].k_selection.fold_losses)
        changed = y.copy()
        changed[test] = 0 if task == "binary" else 1e6
        perturbed = replay.replay_fold(X, changed, train, test, spec, old, report["ranking_diagnostics"][f])
        assert perturbed["selected"] == actual["selected"]
        assert perturbed["loss"] != actual["loss"]
    assert counter.total == 4


def test_size_choice_rejects_tampering_and_preserves_first_minimum(replay):
    fold = {"fold": 0, "k_chosen": 1, "selected_indices": [2],
            "k_selection": {"candidate_k": [1, 2, 3], "mean_inner_loss": [1.1, 1., 1.],
                            "se_at_min": .2, "k_argmin": 2, "k_chosen": 1}}
    diagnostic = {"fold": 0, "ranking": [2, 0, 1]}
    assert replay.policy_sets(fold, diagnostic, 3) == {"one_se": [2], "argmin": [2, 0]}
    for key, value in (("k_argmin", 3), ("se_at_min", -1), ("mean_inner_loss", [1., float("nan"), 2.])):
        changed = copy.deepcopy(fold)
        changed["k_selection"][key] = value
        with pytest.raises(ValueError):
            replay.policy_sets(changed, diagnostic, 3)
    with pytest.raises(ValueError):
        replay.policy_sets(fold, dict(diagnostic, ranking=[2, 2, 1]), 3)


def test_loss_validation_rejects_nonfinite_and_material_differences(replay):
    assert replay.check_loss(1., 1.) == 0
    assert replay.check_loss(1.+1e-14, 1.) != 0
    for loss in (float("nan"), float("inf"), 1.01):
        with pytest.raises(ValueError, match="Replay differs"):
            replay.check_loss(loss, 1.)


def test_checkpoint_resume_rejects_identity_and_content_changes(replay, tmp_path):
    runner = importlib.import_module("run_public_selection")
    record = {"run_fingerprint": "run", "original_sha256": "original", "status": "complete"}
    record["record_sha256"] = fingerprint(record)
    path = tmp_path/"cell.json"
    write_json(path, record)
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert runner.validate_cell(path, "run", "original") == record
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    for run, original in (("other", "original"), ("run", "changed")):
        with pytest.raises(ValueError, match="checkpoint"):
            runner.validate_cell(path, run, original)
    record["status"] = "failed"
    write_json(path, record)
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_cell(path, "run", "original")


def test_run_resume_and_analysis_preserve_all_planned_denominators(replay, monkeypatch, tmp_path):
    original_runner = importlib.import_module("run_public_application")
    runner = importlib.import_module("run_public_selection")
    analyzer = importlib.import_module("analyze_public_selection")
    rng = np.random.default_rng(31)
    X = rng.normal(size=(40, 4))
    y = X[:, 0] + rng.normal(size=40)
    manifest = {"fixture": {"features": 4, "task": "regression"}}
    data = PublicData(X, y, Players.singletons(4), "regression", np.arange(40), {"source": manifest["fixture"]})
    for module in (original_runner, runner):
        monkeypatch.setattr(module, "load_public", lambda *a: data)
    config = {"candidate_k": [1, 2], "include_all_features": True, "budgets": [1],
              "importance_splits": 1, "shadows": 1, "quantile": .9, "one_se_rule": True,
              "inner_folds": 2, "outer_folds": 2, "memory_gib_cap": 8, "scientific_evidence": True,
              "status": "frozen", "protocol_artifact_sha256": {}, "datasets": ["fixture"], "rows": [40], "seeds": [31],
              "l1": {"penalties": [1., .1], "max_iter": 20000, "tol": 1e-4, "coefficient_tolerance": 1e-10}}
    design = {"dataset": "fixture", "rows": 40, "seed": 31}
    original = runner.baseline(tmp_path)
    frozen, provenance = original_runner.freeze(original, config, manifest)
    original_runner.run_design(frozen, manifest, design, original/"fixture-n40-seed31", tmp_path, provenance)
    replay_config = {"status": "frozen", "memory_gib_cap": 8}
    provenance = initialize(tmp_path, replay_config, manifest, runner.ROOT)
    write_json(tmp_path/"frozen-config.json", replay_config)
    runner.run_design(tmp_path, tmp_path, design, replay_config, provenance)
    paths = list(tmp_path.glob("*/cell-*.json"))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    runner.run_design(tmp_path, tmp_path, design, replay_config, provenance)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    analyzer.analyze(tmp_path, tmp_path/"analysis")
    rows = runner.read(tmp_path/"analysis/methods.json")
    assert len(rows) == 2 and all(r["new_model_fits"] == 4 for r in rows)
    assert all(r["size_difference"] >= 0 for r in rows)
    record = runner.read(paths[0])
    record.update(status="failed", error="deliberate fixture failure")
    record.pop("record_sha256")
    record["record_sha256"] = fingerprint(record)
    write_json(paths[0], record)
    analyzer.analyze(tmp_path, tmp_path/"failed-analysis")
    rows = runner.read(tmp_path/"failed-analysis/methods.json")
    assert len(rows) == 2 and sum(r["status"] == "failed" for r in rows) == 1
    assert next(r for r in rows if r["status"] == "failed")["argmin_loss"] is None
    assert runner.read(tmp_path/"failed-analysis/contrasts.json")[0]["complete"] is False
