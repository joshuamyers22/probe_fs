"""Public baselines and checkpoints must preserve training and data boundaries."""

import importlib
import json

import numpy as np
import pytest

from pgfs.datasets import PublicData
from pgfs.players import Players
from pgfs.study import make_spec


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return importlib.import_module("public_baseline"), importlib.import_module("run_public_application")


@pytest.fixture
def config():
    return {"candidate_k": [1, 3], "include_all_features": True, "budgets": [1],
            "importance_splits": 1, "shadows": 1, "quantile": .9, "one_se_rule": True,
            "inner_folds": 2, "outer_folds": 2, "memory_gib_cap": 8, "scientific_evidence": False,
            "l1": {"penalties": [1., .01], "max_iter": 20000, "tol": 1e-4, "coefficient_tolerance": 1e-10}}


def data(task="regression"):
    rng = np.random.default_rng(53)
    X = rng.normal(size=(60, 4))
    y = X[:, 0]+rng.normal(size=60)*.4
    if task != "regression":
        y = (y > np.median(y)).astype(int)
    X[::7, 1] = np.nan
    return X, y


def test_selector_preprocessing_is_refit_on_each_inner_training_partition(modules, config, monkeypatch):
    baseline, _ = modules
    X, y = data()
    spec = make_spec(config, 12, [1, 3], "regression")
    pipelines = []
    original = baseline.make_pipeline

    def capture(*args):
        pipeline = original(*args)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(baseline, "make_pipeline", capture)
    baseline.tune_l1(X, y, spec, 2, config["l1"], baseline.FitCounter())
    expected = baseline.folds(X, y, spec, 2, baseline.INNER_CV_SEED_OFFSET)
    assert len(pipelines) == 4
    for i, (train, _) in enumerate(expected):
        for pipeline in pipelines[i*2:(i+1)*2]:
            imputer, scaler = pipeline.steps[0][1], pipeline.steps[1][1]
            np.testing.assert_allclose(imputer.statistics_, np.nanmedian(X[train], axis=0))
            np.testing.assert_allclose(scaler.mean_, imputer.transform(X[train]).mean(axis=0))
            assert scaler.n_samples_seen_ == len(train)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_baseline_counts_selector_tuning_refits_and_predictions(modules, config, task):
    baseline, _ = modules
    X, y = data(task)
    spec = make_spec(config, 12, [1, 3], task)
    result = baseline.evaluate_baseline(X, y, spec, 2, 2, config["l1"], "l1")
    # Per outer fold: 2 penalties x 2 inner folds x (selector + prediction),
    # then one outer selector refit and one final prediction fit.
    assert result["main_results"]["model_fits_total"] == 20
    assert result["main_results"]["model_fits_importance"] == 10
    assert result["main_results"]["model_fits_selection"] == 10
    all_result = baseline.evaluate_baseline(X, y, spec, 2, 2, config["l1"], "all")
    assert all_result["main_results"]["model_fits_total"] == 2
    assert np.isfinite(result["main_results"]["mean_outer_loss"])


def test_outer_holdout_cannot_change_its_selection_or_tuning(modules, config):
    baseline, _ = modules
    X, y = data()
    spec = make_spec(config, 12, [1, 3], "regression")
    first = baseline.evaluate_baseline(X, y, spec, 2, 2, config["l1"], "l1")
    _, test = baseline.folds(X, y, spec, 2, baseline.OUTER_CV_SEED_OFFSET)[0]
    changed_X, changed_y = X.copy(), y.copy()
    changed_X[test], changed_y[test] = 1e4, -1e4
    second = baseline.evaluate_baseline(changed_X, changed_y, spec, 2, 2, config["l1"], "l1")
    for key in ("selected", "ranking", "tuning"):
        assert first["folds"][0][key] == second["folds"][0][key]
    assert first["folds"][0]["outer_loss"] != second["folds"][0]["outer_loss"]


def test_empty_selection_predicts_intercept_and_ties_choose_strongest_penalty(modules, config):
    baseline, _ = modules
    X = np.zeros((40, 4))
    y = np.linspace(-1, 1, 40)
    spec = make_spec(config, 12, [1, 3], "regression")
    result = baseline.evaluate_baseline(X, y, spec, 2, 2, config["l1"], "l1")
    splits = baseline.folds(X, y, spec, 2, baseline.OUTER_CV_SEED_OFFSET)
    for fold, (train, test) in zip(result["folds"], splits):
        assert fold["selected"] == ()
        assert fold["tuning"]["chosen_penalty"] == 1.
        assert fold["outer_loss"] == pytest.approx(np.mean((y[test]-y[train].mean())**2))


def test_shared_folds_resume_integrity_and_original_row_ids(modules, config, monkeypatch, tmp_path):
    _, runner = modules
    X, y = data()
    public = PublicData(X, y, Players.singletons(4), "regression", np.arange(100, 160), {})
    monkeypatch.setattr(runner, "load_public", lambda *a: public)
    design = {"dataset": "fixture", "rows": 60, "seed": 12}
    provenance = {"fingerprint": "fixture-run"}
    runner.run_design(config, {"fixture": {}}, design, tmp_path, tmp_path, provenance)
    checkpoints = sorted(tmp_path.glob("cell-*.json"))
    assert len(checkpoints) == 4
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in checkpoints}
    runner.run_design(config, {"fixture": {}}, design, tmp_path, tmp_path, provenance)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in checkpoints}
    shared = json.loads((tmp_path/"data.json").read_text())
    records = {json.loads(p.read_text())["method"]: json.loads(p.read_text()) for p in checkpoints}
    assert all(r["status"] == "complete" for r in records.values())
    assert len({r["data_sha256"] for r in records.values()}) == 1
    for f, split in enumerate(shared["folds"]):
        assert set(split["train_rows"]).isdisjoint(split["test_rows"])
        assert set(split["train_rows"]) | set(split["test_rows"]) == set(range(100, 160))
        for inner in split["inner"]:
            assert set(inner["train_rows"]).isdisjoint(inner["valid_rows"])
            assert set(inner["train_rows"]) | set(inner["valid_rows"]) == set(split["train_rows"])
        assert records["all"]["report"]["folds"][f]["outer_loss"] == pytest.approx(
            records["gated-T1"]["report"]["folds"][f]["outer_loss_all_features"])
    with pytest.raises(ValueError, match="checkpoint identity"):
        runner.run_design(config, {"fixture": {}}, design, tmp_path, tmp_path, {"fingerprint": "changed"})
    tampered = json.loads(checkpoints[0].read_text())
    tampered["report"]["main_results"]["mean_outer_loss"] += 1
    checkpoints[0].write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="checkpoint content"):
        runner.run_design(config, {"fixture": {}}, design, tmp_path, tmp_path, provenance)


def test_runner_rejects_official_year_test_rows(modules, config, monkeypatch, tmp_path):
    _, runner = modules
    X, y = data()
    public = PublicData(X, y, Players.singletons(4), "regression", np.arange(60), {})
    monkeypatch.setattr(runner, "load_public", lambda *a: public)
    with pytest.raises(ValueError, match="Official test rows"):
        runner.run_design(config, {"yearpredictionmsd": {"development_rows": 50}},
                          {"dataset": "yearpredictionmsd", "rows": 60, "seed": 12},
                          tmp_path, tmp_path, {"fingerprint": "test"})


def test_freeze_rejects_changed_penalty_grid(modules, config, tmp_path):
    _, runner = modules
    config["protocol_artifact_sha256"] = {}
    runner.freeze(tmp_path, config, {})
    changed = json.loads(json.dumps(config))
    changed["l1"]["penalties"] = [1., .1]
    with pytest.raises(ValueError, match="Run identity changed"):
        runner.freeze(tmp_path, changed, {})


def test_application_analysis_keeps_failures_and_rejects_calibration(modules, config, monkeypatch, tmp_path):
    _, runner = modules
    analyzer = importlib.import_module("analyze_public_application")
    config.update(datasets=["fixture"], rows=[60], seeds=[12], status="frozen",
                  scientific_evidence=True, protocol_artifact_sha256={})
    source = {"features": 4, "task": "regression"}
    manifest = {"fixture": source}
    X, y = data()
    public = PublicData(X, y, Players.singletons(4), "regression", np.arange(60), {"source": source})
    monkeypatch.setattr(runner, "load_public", lambda *a: public)
    original = runner.evaluate_baseline

    def fail_l1(*args):
        if args[-1] == "l1":
            raise ValueError("deliberate numerical failure")
        return original(*args)

    monkeypatch.setattr(runner, "evaluate_baseline", fail_l1)
    config, provenance = runner.freeze(tmp_path/"run", config, manifest)
    runner.run_design(config, manifest, {"dataset": "fixture", "rows": 60, "seed": 12},
                      tmp_path/"run/fixture-n60-seed12", tmp_path, provenance)
    analyzer.analyze(tmp_path/"run", tmp_path/"analysis")
    rows = json.loads((tmp_path/"analysis/methods.json").read_text())
    assert len(rows) == 4
    failed = next(r for r in rows if r["method"] == "l1")
    assert failed["status"] == "failed" and failed["loss"] is None
    summaries = json.loads((tmp_path/"analysis/summary.json").read_text())
    assert next(r for r in summaries if r["control"] == "l1")["complete_seeds"] == 0
    config.update(status="calibration", scientific_evidence=False)
    runner.freeze(tmp_path/"calibration", config, manifest)
    with pytest.raises(ValueError, match="Calibration is not scientific"):
        analyzer.analyze(tmp_path/"calibration", tmp_path/"not-written")


def test_scientific_config_cannot_bypass_failed_or_mismatched_cost_gate(modules, config, monkeypatch, tmp_path):
    _, runner = modules
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    planned = dict(config, rows=[3000, 5000], seeds=[501], status="draft", protocol_artifact_sha256={})
    preflight = {"within_compute_gate": False, "calibration_failures": 1, "planned_config": planned}
    path = tmp_path/"cost.json"
    runner.write_json(path, preflight)
    frozen = dict(planned, status="frozen", cost_preflight="cost.json", protocol_artifact_sha256={"cost.json": runner.sha256(path)})
    with pytest.raises(ValueError, match="did not pass"):
        runner.validate_scientific_config(frozen)
    preflight.update(within_compute_gate=True, calibration_failures=0)
    runner.write_json(path, preflight)
    frozen["protocol_artifact_sha256"]["cost.json"] = runner.sha256(path)
    runner.validate_scientific_config(frozen)
    frozen["rows"] = [10000]
    with pytest.raises(ValueError, match="differs from cost preflight: rows"):
        runner.validate_scientific_config(frozen)
