"""Integrity, data-boundary, and pilot reporting checks without network access."""

import json
import zipfile

import numpy as np
import pytest

from pgfs.datasets import load_public, sha256, verified_archive
from pgfs.study import clean, initialize, write_json


def archive(tmp_path, files, rows, features, task="regression", **extra):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        for name, text in files.items():
            zipped.writestr(name, text)
    return {"filename": path.name, "sha256": sha256(path), "rows": rows,
            "features": features, "task": task, "selection_policy": "test", **extra}


def test_checksum_rejects_modified_cache(tmp_path):
    entry = archive(tmp_path, {"x": "original"}, 1, 1)
    (tmp_path / entry["filename"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verified_archive(entry, tmp_path)


def test_year_subset_never_contains_official_test_rows(tmp_path):
    text = "".join(f"{1900+i},{i},{i+1}\n" for i in range(50))
    entry = archive(tmp_path, {"YearPredictionMSD.txt": text}, 50, 2, development_rows=30)
    first = load_public("yearpredictionmsd", entry, tmp_path, 20, 11)
    second = load_public("yearpredictionmsd", entry, tmp_path, 20, 11)
    np.testing.assert_array_equal(first.row_indices, second.row_indices)
    assert len(first.y) == 20
    assert first.row_indices.max() < 30
    np.testing.assert_array_equal(first.y, first.row_indices + 1900)
    np.testing.assert_array_equal(first.X[:, 0], first.row_indices)


def test_miniboone_header_labels_and_sentinel_handling(tmp_path):
    text = "20 20\n" + "".join(f"{i} {-999 if i == 0 else i+1}\n" for i in range(40))
    entry = archive(tmp_path, {"MiniBooNE_PID.txt": text}, 40, 2, task="binary")
    data = load_public("miniboone", entry, tmp_path, 40, 11)
    np.testing.assert_array_equal(data.y, np.r_[np.ones(20), np.zeros(20)])
    assert np.isnan(data.X[0, 1])
    subset = load_public("miniboone", entry, tmp_path, 20, 11)
    assert subset.y.sum() == 10
    np.testing.assert_array_equal(subset.y, (subset.row_indices < 20).astype(float))


def test_superconductivity_keeps_first_formula_without_target_selection(tmp_path):
    files = {
        "unique_m.csv": "material\n" + "".join(f"formula{i//2}\n" for i in range(40)),
        "train.csv": "descriptor,critical_temp\n" + "".join(f"{i},{1000-i}\n" for i in range(40)),
    }
    entry = archive(tmp_path, files, 40, 1)
    data = load_public("superconductivity", entry, tmp_path, 20, 11)
    np.testing.assert_array_equal(data.row_indices, np.arange(0, 40, 2))
    np.testing.assert_array_equal(data.y, 1000 - data.row_indices)


def test_strict_json_uses_null_for_unavailable_endpoints(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"endpoint": np.nan, "array": np.array([1.0, np.inf])})
    assert json.loads(path.read_text()) == {"endpoint": None, "array": [1.0, None]}
    assert clean(np.int64(4)) == 4


def test_resume_rejects_changed_configuration(tmp_path):
    pytest.importorskip("matplotlib", reason="experiment provenance includes plotting dependencies")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    first = initialize(tmp_path, {"seed": 1}, {}, root)
    assert initialize(tmp_path, {"seed": 1}, {}, root) == first
    with pytest.raises(ValueError, match="Run identity changed"):
        initialize(tmp_path, {"seed": 2}, {}, root)


@pytest.mark.parametrize("git_missing", [False, True])
def test_provenance_works_in_source_archives_without_git(tmp_path, monkeypatch, git_missing):
    import subprocess
    from pathlib import Path

    def no_revision(*args, **kwargs):
        if git_missing:
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(args[0], 128, stdout="", stderr="not a git repository")

    monkeypatch.setattr("pgfs.study.subprocess.run", no_revision)
    root = Path(__file__).resolve().parents[1]
    result = initialize(tmp_path, {"seed": 1}, {}, root)
    assert result["git_commit"] is None
    assert result["identity"]["source_hashes"]["src/pgfs/study.py"] == sha256(root / "src/pgfs/study.py")
    assert initialize(tmp_path, {"seed": 1}, {}, root) == result


def test_public_comparison_has_no_fabricated_ground_truth():
    from pgfs.datasets import PublicData
    from pgfs.players import Players
    from pgfs.study import make_spec, public_comparison

    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 8))
    data = PublicData(X, X[:, 0] + rng.normal(size=50), Players.singletons(8),
                      "regression", np.arange(50), {})
    config = {"importance_splits": 1, "shadows": 3, "quantile": 0.9,
              "outer_folds": 2, "inner_folds": 2}
    spec = make_spec(config, 11, [1, 3], "regression")
    rows, comparison, reports = public_comparison(data, spec, 1, config)
    assert len(rows) == 2
    assert comparison["primary_endpoint_applicable"] is False
    assert comparison["budget_matched"]
    for report in reports.values():
        assert "primary_endpoint" not in report
        assert "mean_class_recall" not in report["main_results"]
        assert len(report["folds"]) == 2


@pytest.mark.parametrize("p", [15, 50, 81, 90, 300])
def test_one_context_budget_is_exact_at_public_dataset_widths(p):
    from pgfs.budget import expected_base_fits
    # A single permutation has exactly p+1 distinct prefixes, at every width.
    assert expected_base_fits(p, 1) == pytest.approx(p + 1, abs=1e-10)


def test_wide_budget_matches_high_precision_occupancy_expectation():
    import math
    from decimal import Decimal, localcontext

    from pgfs.budget import expected_base_fits

    with localcontext() as context:
        context.prec = 100
        p, repetitions = 90, 8
        exact = sum(Decimal(math.comb(p, s)) * (
            1 - (1 - 1 / Decimal(math.comb(p, s))) ** repetitions
        ) for s in range(p + 1))
    assert expected_base_fits(p, repetitions) == pytest.approx(float(exact), rel=1e-12)
