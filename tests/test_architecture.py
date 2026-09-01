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
