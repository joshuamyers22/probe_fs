"""Executable dependency rules for the PGFS domain model."""

import ast
from pathlib import Path


def test_configuration_and_metrics_do_not_import_experiment_runner() -> None:
    for name in ("config", "metrics"):
        path = Path(f"src/pgfs/{name}.py")
        tree = ast.parse(path.read_text())
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not any("experiments" in module for module in modules), path


def test_baseline_evaluation_does_not_depend_on_selector_implementations() -> None:
    path = Path("src/pgfs/baseline_evaluation.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "baselines" not in modules


def test_experiment_models_do_not_depend_on_experiment_runner() -> None:
    path = Path("src/pgfs/experiment_models.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "experiments" not in modules


def test_experiment_decisions_do_not_depend_on_fitting_or_simulation_runners() -> None:
    path = Path("src/pgfs/experiment_decisions.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules.isdisjoint({"experiments", "importance", "nested", "simulate"})


def test_experiment_runner_does_not_define_decision_policy() -> None:
    tree = ast.parse(Path("src/pgfs/experiments.py").read_text())
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert functions.isdisjoint(
        {"decision_criteria", "redundancy_check", "_verdict_sentence"}
    )


def test_estimation_validation_does_not_depend_on_fitting_adapters() -> None:
    path = Path("src/pgfs/estimation_data.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules.isdisjoint({"learners", "importance"})


def test_importance_policy_does_not_depend_on_fitting_or_sampling_adapters() -> None:
    path = Path("src/pgfs/importance_policy.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules.isdisjoint({"contexts", "estimation_data", "importance", "learners"})


def test_nested_result_models_do_not_depend_on_evaluation_runner() -> None:
    path = Path("src/pgfs/nested_models.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "nested" not in modules


def test_knockoff_policy_does_not_depend_on_baseline_or_evaluation_workflows() -> None:
    path = Path("src/pgfs/knockoffs.py")
    tree = ast.parse(path.read_text())
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert modules.isdisjoint({"baselines", "baseline_evaluation", "learners"})
