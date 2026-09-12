"""The real-data validation command.

Its most important job is refusing to let a synthetic result be mistaken for
a real one, so that is what most of these tests check.
"""
import numpy as np
import pytest

from backend.validate import (
    format_report, is_valid_run, paired_difference, power_note, run,
)


def test_paired_difference_arithmetic():
    base = {str(i): 1.0 for i in range(10)}
    other = {str(i): 1.5 for i in range(10)}
    out = paired_difference(base, other)
    assert out["matched_trades"] == 10
    assert out["difference_r"] == pytest.approx(0.5)
    assert out["base_mean_r"] == pytest.approx(1.0)


def test_paired_difference_needs_enough_overlap():
    assert paired_difference({"1": 1.0}, {"1": 2.0}) is None
    assert paired_difference({}, {}) is None


def test_paired_difference_only_uses_shared_trades():
    """Two policies take different trade sequences; only the overlap compares."""
    base = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "9": -9.0}
    other = {"1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0, "8": +9.0}
    out = paired_difference(base, other)
    assert out["matched_trades"] == 5
    assert out["difference_r"] == pytest.approx(0.0)


def test_power_note_scales_with_sample_size():
    assert "far too few" in power_note(10)
    assert "0.00 and 2.78" in power_note(60)
    assert "large effect" in power_note(200)
    assert "usable" in power_note(1000)


def test_report_banner_marks_synthetic_data_loudly():
    report = {"symbol": "NAS100_USD", "provider": "synthetic", "status": "synthetic",
              "policies": {}, "comparisons": {}, "warnings": ["SYNTHETIC DATA: ..."]}
    text = format_report(report)
    assert "NOT A TEST OF THE STRATEGY" in text
    assert "SYNTHETIC" in text
    assert is_valid_run(report) is False


def test_report_banner_marks_real_data():
    report = {"symbol": "NAS100_USD", "provider": "oanda", "status": "real",
              "policies": {}, "comparisons": {}, "warnings": []}
    assert "REAL MARKET DATA" in format_report(report)
    assert is_valid_run(report) is True


def test_a_token_without_data_is_not_a_real_run():
    """Regression: the banner was derived from the provider name, so setting a
    token and fetching nothing printed REAL MARKET DATA over zero bars and
    exited zero. Reaching the broker is not the same as receiving data."""
    report = {"symbol": "NAS100_USD", "provider": "oanda", "status": "no_data",
              "bars_available": 0, "policies": {}, "comparisons": {},
              "warnings": ["seeding failed: 403"]}
    text = format_report(report)
    assert "NOTHING WAS TESTED" in text
    assert "REAL MARKET DATA" not in text
    assert is_valid_run(report) is False


def test_an_unknown_status_is_never_treated_as_valid():
    assert is_valid_run({"status": "unknown"}) is False
    assert is_valid_run({}) is False
    assert "NOTHING WAS TESTED" in format_report(
        {"symbol": "X", "provider": "oanda", "policies": {}, "comparisons": {},
         "warnings": []}
    )


def test_report_always_states_what_is_not_modelled():
    report = {"symbol": "X", "provider": "oanda", "data_is_real": True,
              "policies": {}, "comparisons": {}, "warnings": []}
    text = format_report(report)
    assert "Slippage" in text and "gap risk" in text


def test_validate_runs_on_a_fresh_database(tmp_path, monkeypatch):
    """Regression: validate crashed on a machine that had never created the
    database, which is exactly the state of a fresh checkout."""
    db = str(tmp_path / "fresh.db")
    monkeypatch.setattr("backend.store.DB_PATH", db)
    monkeypatch.setenv("DATA_PROVIDER", "synthetic")
    report = run(symbol="NAS100_USD", days=2, step_minutes=60, do_seed=False)
    assert "warnings" in report
    assert is_valid_run(report) is False


def test_validate_flags_thin_history(tmp_path, monkeypatch):
    db = str(tmp_path / "thin.db")
    monkeypatch.setattr("backend.store.DB_PATH", db)
    monkeypatch.setenv("DATA_PROVIDER", "synthetic")
    report = run(symbol="NOTHING_HERE", days=2, step_minutes=60, do_seed=False)
    assert any("seed more history" in w for w in report["warnings"])
    assert report["policies"] == {}
    assert report["status"] == "no_data"
    assert is_valid_run(report) is False
