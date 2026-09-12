"""The agents that implement the study book's teachings."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backend.agents.base import LONG, NEUTRAL, SHORT, MarketContext
from backend.agents.book_agents import (
    DeliveryResistanceAgent,
    FirstPresentedGapAgent,
    InversionAgent,
    ObsidianWickAgent,
    OpeningGapAgent,
    SessionRangeAgent,
)
from backend.ict import sessions
from backend.store import resample_ohlcv

NY = ZoneInfo("America/New_York")


def ctx_at(df_1m, now_ts, **cfg):
    frames = {tf: resample_ohlcv(df_1m, tf) for tf in ("5m", "15m", "1h")}
    base = {"max_staleness_seconds": 10 ** 9, "require_killzone": False}
    base.update(cfg)
    return MarketContext(symbol="NAS100_USD", now_ts=now_ts, frames=frames, config=base)


def day_of_minutes(day: datetime, price_fn):
    rows = []
    start = int(day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    for i in range(24 * 60):
        t = start + i * 60
        p = price_fn(i)
        rows.append({"t": t, "o": p, "h": p + 2, "l": p - 2, "c": p, "v": 10.0})
    return rows


@pytest.fixture(scope="module")
def session_day():
    """A Monday of one-minute bars with a distinct premarket and opening range."""
    monday = datetime(2024, 6, 3, tzinfo=NY)
    friday = datetime(2024, 5, 31, tzinfo=NY)
    rows = day_of_minutes(friday, lambda i: 20_000.0)
    rows += day_of_minutes(monday, lambda i: 20_050.0 + (i % 120) * 0.1)
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def test_session_range_waits_for_the_window_to_close(session_day):
    """An opening-range extreme formed at 09:57 was not available at 09:40."""
    mid_window = int(datetime(2024, 6, 3, 9, 40, tzinfo=NY).timestamp())
    signal = SessionRangeAgent().run(ctx_at(session_day, mid_window))
    # The premarket has closed by 09:40, so that range may be used; the
    # opening range must not be, and the evidence must say so.
    assert signal.evidence["opening_range"]["complete"] is False


def test_session_range_grades_only_after_the_window(session_day):
    after = int(datetime(2024, 6, 3, 10, 30, tzinfo=NY).timestamp())
    signal = SessionRangeAgent().run(ctx_at(session_day, after))
    assert signal.error is None
    assert signal.evidence["opening_range"]["complete"] is True
    assert signal.evidence["premarket"]["complete"] is True
    if "graded_octants" in signal.evidence:
        assert len(signal.evidence["graded_octants"]) == 9   # 0/8 through 8/8


def test_premarket_and_opening_ranges_are_kept_separate(session_day):
    """They can have different midpoints on the same day."""
    after = int(datetime(2024, 6, 3, 10, 30, tzinfo=NY).timestamp())
    pm = sessions.premarket_range(session_day, after)
    orng = sessions.opening_range(session_day, after)
    assert pm.start_ts != orng.start_ts
    assert pm.as_dict()["name"] != orng.as_dict()["name"]


def test_opening_gap_measures_prior_close_to_open(session_day):
    after = int(datetime(2024, 6, 3, 10, 30, tzinfo=NY).timestamp())
    gap = sessions.opening_gap(session_day, after)
    assert gap["complete"] is True
    assert gap["prior_close"] == pytest.approx(20_000.0)
    assert gap["direction"] == "gap_up"
    assert gap["midpoint"] == pytest.approx((gap["open"] + gap["prior_close"]) / 2)


def test_opening_gap_agent_reports_the_reference(session_day):
    after = int(datetime(2024, 6, 3, 10, 30, tzinfo=NY).timestamp())
    signal = OpeningGapAgent().run(ctx_at(session_day, after))
    assert signal.error is None
    assert "opening_gap" in signal.evidence


def test_first_presented_gap_agent_waits_for_the_open(session_day):
    before_open = int(datetime(2024, 6, 3, 8, 30, tzinfo=NY).timestamp())
    signal = FirstPresentedGapAgent().run(ctx_at(session_day, before_open))
    assert signal.direction == NEUTRAL
    assert "not opened" in signal.rationale


def test_every_book_agent_survives_real_looking_data(synthetic_1m):
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    ctx = ctx_at(synthetic_1m, now)
    for agent in (SessionRangeAgent(), OpeningGapAgent(), InversionAgent(),
                  FirstPresentedGapAgent(), ObsidianWickAgent(), DeliveryResistanceAgent()):
        signal = agent.run(ctx)
        assert signal.error is None, f"{agent.name}: {signal.error}"
        assert signal.direction in (LONG, SHORT, NEUTRAL)
        assert -1.0 <= signal.score <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.rationale


def test_book_agents_abstain_on_empty_frames():
    empty = {tf: pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"]) for tf in ("5m", "15m", "1h")}
    ctx = MarketContext(symbol="NAS100_USD", now_ts=1_717_424_100, frames=empty)
    for agent in (SessionRangeAgent(), OpeningGapAgent(), InversionAgent(),
                  FirstPresentedGapAgent(), ObsidianWickAgent(), DeliveryResistanceAgent()):
        signal = agent.run(ctx)
        assert signal.error is None
        assert signal.direction == NEUTRAL


def test_obsidian_agent_states_its_own_limits(synthetic_1m):
    """The source leaves the qualifying event unresolved, so the agent must
    not present the measurement as a complete setup."""
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    signal = ObsidianWickAgent().run(ctx_at(synthetic_1m, now))
    text = (signal.rationale + str(signal.evidence)).lower()
    if signal.confidence > 0:
        assert "not established" in text or "do not establish" in text
    assert ObsidianWickAgent().default_weight < 1.0, "weighted low by design"


def test_delivery_resistance_classifies_the_route(synthetic_1m):
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    signal = DeliveryResistanceAgent().run(ctx_at(synthetic_1m, now))
    assert "low_resistance" in signal.evidence
    assert isinstance(signal.evidence["low_resistance"], bool)
    assert 0.0 <= signal.evidence["quality"] <= 1.0


def test_silver_bullet_london_window_exists():
    """The 2023 lesson gives 03:00-04:00 New York as a window."""
    at_three = int(datetime(2024, 6, 3, 3, 30, tzinfo=NY).timestamp())
    names = [s.name for s in sessions.active_sessions(at_three)]
    assert "silver_bullet_london" in names
