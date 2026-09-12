"""Storage, resampling and feed assembly."""
import os
import tempfile

import pandas as pd
import pytest

from backend.data import feed
from backend.data.providers import SyntheticProvider, parse_oanda_candles
from backend.store import (
    count_rows, init_db, known_symbols, latest_ts, load_1m, resample_ohlcv, upsert_1m,
)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "candles.db")
    monkeypatch.setattr("backend.store.DB_PATH", path)
    init_db(path)
    return path


def minute_frame(start=1_717_200_000, n=120, price=20_000.0):
    rows = []
    for i in range(n):
        base = price + i
        rows.append({"t": start + i * 60, "o": base, "h": base + 2,
                     "l": base - 2, "c": base + 1, "v": 10.0})
    return pd.DataFrame(rows)


def test_roundtrip_through_the_store(db_path, monkeypatch):
    monkeypatch.setattr("backend.store.DB_PATH", db_path)
    df = minute_frame()
    assert upsert_1m("NAS100_USD", df, db_path) == 120
    assert count_rows("NAS100_USD", db_path) == 120
    assert latest_ts("NAS100_USD", db_path) == int(df["t"].iloc[-1])
    assert "NAS100_USD" in known_symbols(db_path)

    loaded = load_1m("NAS100_USD", 0, 9_999_999_999, db_path)
    assert len(loaded) == 120
    assert loaded["t"].is_monotonic_increasing


def test_upsert_is_idempotent(db_path):
    df = minute_frame()
    upsert_1m("NAS100_USD", df, db_path)
    upsert_1m("NAS100_USD", df, db_path)
    assert count_rows("NAS100_USD", db_path) == 120


def test_resample_aggregates_correctly():
    df = minute_frame(n=60)
    out = resample_ohlcv(df, "15m")
    assert len(out) == 4
    first = out.iloc[0]
    assert first["o"] == df["o"].iloc[0]
    assert first["c"] == df["c"].iloc[14]
    assert first["h"] == df["h"].iloc[:15].max()
    assert first["l"] == df["l"].iloc[:15].min()
    assert first["v"] == df["v"].iloc[:15].sum()


def test_resample_timestamps_are_epoch_seconds():
    """Guards the pandas 2 vs 3 datetime-resolution difference: a wrong
    conversion silently yields timestamps a thousand times too small."""
    df = minute_frame(start=1_717_200_000, n=60)
    out = resample_ohlcv(df, "15m")
    assert out["t"].iloc[0] == 1_717_200_000
    assert int(out["t"].iloc[1] - out["t"].iloc[0]) == 900
    as_dt = pd.to_datetime(out["t"], unit="s", utc=True)
    assert as_dt.dt.year.iloc[0] == 2024


def test_resample_rejects_an_unknown_timeframe():
    with pytest.raises(ValueError):
        resample_ohlcv(minute_frame(), "7m")


def test_resample_of_empty_frame_keeps_the_schema():
    out = resample_ohlcv(pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"]), "5m")
    assert list(out.columns) == ["t", "o", "h", "l", "c", "v"]
    assert out.empty


def test_synthetic_provider_is_deterministic_across_processes():
    """Regression: the seed was derived from Python's built-in hash(), which
    is randomised per process. Two backtests run as separate processes then
    received different data, so their results were never comparable."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from backend.data.providers import SyntheticProvider;"
        "d=SyntheticProvider().fetch_1m('NAS100_USD',count=50,to_ts=1717200000);"
        "print(round(float(d.c.iloc[-1]),6))"
    )
    root = Path(__file__).resolve().parent.parent
    runs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=root).stdout.strip()
        for _ in range(3)
    }
    assert len(runs) == 1, f"synthetic feed differs between processes: {runs}"
    assert runs != {""}, "subprocess produced no output"


def test_synthetic_provider_is_deterministic():
    a = SyntheticProvider().fetch_1m("NAS100_USD", count=500, to_ts=1_717_200_000)
    b = SyntheticProvider().fetch_1m("NAS100_USD", count=500, to_ts=1_717_200_000)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_provider_never_runs_past_the_requested_end():
    end = 1_717_200_000
    df = SyntheticProvider().fetch_1m("NAS100_USD", count=2000, to_ts=end)
    assert int(df["t"].iloc[-1]) <= end
    assert df["t"].is_monotonic_increasing
    assert not df["t"].duplicated().any()


def test_synthetic_bars_are_internally_consistent():
    df = SyntheticProvider().fetch_1m("NAS100_USD", count=1000)
    assert (df["h"] >= df[["o", "c"]].max(axis=1) - 1e-9).all()
    assert (df["l"] <= df[["o", "c"]].min(axis=1) + 1e-9).all()
    assert (df["v"] >= 0).all()


def test_synthetic_feed_skips_weekends():
    from backend.ict.sessions import is_weekend
    df = SyntheticProvider().fetch_1m("NAS100_USD", count=5000)
    assert not any(is_weekend(int(t)) for t in df["t"])


def test_parse_oanda_candles_skips_incomplete_bars():
    payload = [
        {"time": "2024-06-03T13:30:00.000000000Z", "complete": True,
         "mid": {"o": "1.0", "h": "2.0", "l": "0.5", "c": "1.5"}, "volume": 10},
        {"time": "2024-06-03T13:31:00.000000000Z", "complete": False,
         "mid": {"o": "1.5", "h": "2.5", "l": "1.0", "c": "2.0"}, "volume": 5},
    ]
    df = parse_oanda_candles(payload)
    assert len(df) == 1
    assert df["c"].iloc[0] == 1.5


def test_parse_oanda_candles_tolerates_malformed_rows():
    payload = [{"time": "2024-06-03T13:30:00Z", "complete": True, "mid": {"o": "1.0"}}]
    assert parse_oanda_candles(payload).empty


def test_load_frames_drops_the_unclosed_bar(db_path, monkeypatch):
    monkeypatch.setattr("backend.store.DB_PATH", db_path)
    provider = SyntheticProvider()
    df = provider.fetch_1m("NAS100_USD", count=6000)
    upsert_1m("NAS100_USD", df, db_path)

    # Ask mid-bar: the 15m frame must not include the bar still forming.
    now = int(df["t"].iloc[-1]) - 300
    frames = feed.load_frames("NAS100_USD", now, ["5m", "15m"], bars=50,
                              provider=provider, allow_fetch=False)
    for tf, seconds in (("5m", 300), ("15m", 900)):
        if not frames[tf].empty:
            assert int(frames[tf]["t"].iloc[-1]) + seconds <= now


def test_build_context_assembles_all_timeframes(db_path, monkeypatch):
    monkeypatch.setattr("backend.store.DB_PATH", db_path)
    provider = SyntheticProvider()
    upsert_1m("NAS100_USD", provider.fetch_1m("NAS100_USD", count=8000), db_path)

    ctx = feed.build_context("NAS100_USD", provider=provider, allow_fetch=False,
                             config={"ltf": "5m", "mtf": "15m", "htf": "1h"})
    assert set(ctx.frames) == {"5m", "15m", "1h"}
    assert ctx.price > 0
    assert ctx.atr("5m") > 0
