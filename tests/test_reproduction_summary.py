"""Cross-platform roundoff must not hide changed scientific results or counts."""

import math
import runpy

import pytest


@pytest.fixture
def check(monkeypatch):
    monkeypatch.syspath_prepend("experiments")
    return runpy.run_path("experiments/reproduce.py")["check_summary"]


def test_observed_linux_interval_roundoff_requires_opt_in_and_is_reported(check):
    actual = {"low": 0.0272074009569747, "valid": 140}
    recorded = {"low": 0.027207400956974673, "valid": 140}
    with pytest.raises(ValueError, match=r"revised.low: actual="):
        check(actual, recorded, "revised")
    differences = check(actual, recorded, "revised", allow_roundoff=True)
    assert differences == [{"path": "revised.low", "actual": actual["low"],
                            "recorded": recorded["low"],
                            "absolute_difference": abs(actual["low"]-recorded["low"])}]


@pytest.mark.parametrize("actual,recorded", [
    ({"valid": 140}, {"valid": 141}),
    ({"valid": 10**15+1}, {"valid": 10**15}),
    ({"valid": 140.0}, {"valid": 140}),
    ({"valid": True}, {"valid": 1}),
    ({"loss": 0.50001}, {"loss": 0.5}),
    ({"loss": math.nan}, {"loss": 0.5}),
    ({"loss": math.inf}, {"loss": 0.5}),
    ({"loss": 0.0}, {"loss": None}),
    ({"status": "valid"}, {"status": "invalid"}),
    ({"loss": 0.5, "extra": 1}, {"loss": 0.5}),
    ([0.5], [0.5, 0.5]),
])
def test_roundoff_mode_rejects_substantive_or_structural_changes(check, actual, recorded):
    with pytest.raises(ValueError, match="Regenerated statistics differ"):
        check(actual, recorded, "summary", allow_roundoff=True)


def test_exact_summary_has_no_roundoff_entries(check):
    summary = {"valid": 140, "interval": [None, 0.5], "matched": True}
    assert check(summary, summary, "summary") == []
    assert check(summary, summary, "summary", allow_roundoff=True) == []
