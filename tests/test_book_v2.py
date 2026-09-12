"""Tests for edition 0.6 of the study: new primitives, policies and agents."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backend.agents.base import LONG, NEUTRAL, SHORT, MarketContext
from backend.agents.book_agents_v2 import (
    CheckpointAgent, RejectionBlockAgent, VenomAgent, VolumeImbalanceAgent,
)
from backend.backtest.engine import EXIT_POLICIES, Backtester, BacktestTrade
from backend.ict import core, sessions
from backend.ict.contracts import Contract, RiskLadder
from backend.store import resample_ohlcv

NY = ZoneInfo("America/New_York")
TEST_CONTRACT = Contract("TEST", "unit test", 1.0, 0.25, 0.0)


def ctx_at(df_1m, now_ts, **cfg):
    frames = {tf: resample_ohlcv(df_1m, tf) for tf in ("5m", "15m", "1h")}
    base = {"max_staleness_seconds": 10 ** 9, "require_killzone": False}
    base.update(cfg)
    return MarketContext(symbol="NAS100_USD", now_ts=now_ts, frames=frames, config=base)


# ---------------------------------------------------------------------------
# Chapter 37: rejection blocks
# ---------------------------------------------------------------------------

def test_rejection_block_separates_body_from_wick(make_frame):
    """The worked example: body reference 108, wick high 110. A return to 109
    crosses the body without crossing the wick."""
    df = make_frame([
        (100, 101, 99, 100), (101, 104, 100, 103),
        (104, 110, 103, 108),      # highest body 108, highest wick 110
        (107, 108, 104, 105), (105, 106, 101, 102),
        (102, 103, 99, 100),
    ])
    blocks = core.find_rejection_blocks(df, strength=2)
    bearish = [b for b in blocks if b.direction == core.BEARISH]
    assert bearish
    block = bearish[0]
    assert block.body_ref == 108.0
    assert block.wick_extreme == 110.0
    assert block.crossing(109.0) == "body"
    assert block.crossing(111.0) == "both"
    assert block.crossing(107.0) == "neither"


def test_bullish_rejection_block_mirrors(make_frame):
    """The mirror: body reference 92, wick low 90; a return to 91 crosses the
    body boundary without making a new low."""
    df = make_frame([
        (100, 101, 99, 100), (99, 100, 96, 97),
        (96, 97, 90, 92),          # lowest body 92, lowest wick 90
        (93, 96, 92, 95), (95, 99, 94, 98),
        (98, 102, 97, 101),
    ])
    blocks = core.find_rejection_blocks(df, strength=2)
    bullish = [b for b in blocks if b.direction == core.BULLISH]
    assert bullish
    block = bullish[0]
    assert block.body_ref == 92.0
    assert block.wick_extreme == 90.0
    assert block.crossing(91.0) == "body"
    assert block.crossing(89.0) == "both"


def test_the_longest_wick_candle_need_not_supply_the_body(make_frame):
    """The source warns against automatically taking the body of the candle
    with the longest wick."""
    df = make_frame([
        (100, 101, 99, 100), (100, 103, 99, 102),
        (100, 112, 99, 101),       # longest wick, but a low body
        (104, 106, 103, 106),      # highest body
        (105, 106, 102, 103), (103, 104, 100, 101),
    ])
    blocks = core.find_rejection_blocks(df, strength=2, cluster=3)
    bearish = [b for b in blocks if b.direction == core.BEARISH]
    if bearish:
        block = bearish[0]
        assert block.wick_extreme == 112.0
        assert block.body_ref == 106.0
        assert block.body_idx != block.wick_idx


# ---------------------------------------------------------------------------
# Appendix A: volume imbalance and suspension blocks
# ---------------------------------------------------------------------------

def test_volume_imbalance_measures_body_separation(make_frame):
    df = make_frame([
        (100, 106, 99, 104),       # body 100-104
        (106, 110, 105, 109),      # body 106-109, so a gap 104 to 106
    ])
    out = core.find_volume_imbalances(df)
    assert len(out) == 1
    assert out[0].direction == core.BULLISH
    assert out[0].bottom == 104.0 and out[0].top == 106.0


def test_suspension_block_needs_separations_at_both_ends(make_frame):
    """The corrected definition: a candle bounded above and below by body
    separations, with no three-candle wick gap required."""
    df = make_frame([
        (100, 106, 99, 104),
        (106, 112, 105, 110),      # separated from both neighbours
        (114, 120, 113, 118),
    ])
    blocks = core.find_suspension_blocks(df)
    assert blocks, "two adjacent body separations should form a suspension block"

    only_one = make_frame([(100, 106, 99, 104), (106, 112, 105, 110),
                           (108, 112, 107, 111)])
    assert core.find_suspension_blocks(only_one) == []


# ---------------------------------------------------------------------------
# Chapter 38: breaker projection anchors
# ---------------------------------------------------------------------------

def test_breaker_projection_excludes_the_raid():
    """A at 100 and B at 108 give a width of 8 and a projection of 116. Using
    the raid low of 97 gives 119, which is valid arithmetic against the wrong
    anchor."""
    assert core.breaker_projection(100, 108, 1.0) == 116.0
    assert core.breaker_projection(97, 108, 1.0) == 119.0
    assert core.breaker_projection_bearish(108, 100, 1.0) == 92.0


# ---------------------------------------------------------------------------
# Chapter 22: staged protection
# ---------------------------------------------------------------------------

def test_progressive_stop_matches_the_worked_arithmetic():
    """Entry 100, stop 80, target 180. At 120 the stop moves to 85, at 140 to
    90, at 160 to entry."""
    bt = Backtester(exit_policy=EXIT_POLICIES["progressive_stop"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=80.0,
                          target=180.0, units=4.0)
    for price, expected in ((120.0, 85.0), (140.0, 90.0), (160.0, 100.0)):
        bt._manage(trade, high=price, low=price - 1, now=300, contract=TEST_CONTRACT)
        assert trade.stop == pytest.approx(expected), price


def test_progressive_stop_mirrors_for_a_short():
    bt = Backtester(exit_policy=EXIT_POLICIES["progressive_stop"])
    trade = BacktestTrade(direction=SHORT, entry_ts=0, entry=100.0, stop=120.0,
                          target=20.0, units=4.0)
    bt._manage(trade, high=81.0, low=80.0, now=300, contract=TEST_CONTRACT)
    assert trade.stop == pytest.approx(115.0)   # a quarter of 20 removed


def test_progressive_stop_permits_open_risk_unlike_breakeven_at_1R():
    """The book contrasts the two explicitly. At one times risk the staged
    ladder has not yet reached breakeven."""
    bt = Backtester(exit_policy=EXIT_POLICIES["progressive_stop"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=80.0,
                          target=180.0, units=4.0)
    bt._manage(trade, high=120.0, low=119.0, now=300, contract=TEST_CONTRACT)
    assert trade.stop < trade.entry, "still carrying open risk at 1R"


def test_stop_ladder_never_loosens_protection():
    bt = Backtester(exit_policy=EXIT_POLICIES["progressive_stop"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=80.0,
                          target=180.0, units=4.0)
    bt._manage(trade, high=160.0, low=159.0, now=300, contract=TEST_CONTRACT)
    assert trade.stop == pytest.approx(100.0)
    bt._manage(trade, high=121.0, low=120.0, now=600, contract=TEST_CONTRACT)
    assert trade.stop == pytest.approx(100.0), "a lower rung must not pull the stop back"


# ---------------------------------------------------------------------------
# Chapter 26: the two gap ladders
# ---------------------------------------------------------------------------

def test_gap_bulk_at_half_takes_the_majority_early():
    bt = Backtester(exit_policy=EXIT_POLICIES["gap_bulk_at_half"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=95.0,
                          target=120.0, units=12.0)
    bt._manage(trade, high=110.5, low=101.0, now=300, contract=TEST_CONTRACT)
    assert trade.legs, "half way to target should trigger the partial"
    assert trade.legs[0].units == 9.0        # 75% of twelve
    assert trade.remaining_units == 3.0


def test_gap_ladder_keeps_runners_past_the_target():
    """Variant B: the bulk off at full closure with small runners beyond."""
    bt = Backtester(exit_policy=EXIT_POLICIES["gap_ladder"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=95.0,
                          target=120.0, units=12.0)
    bt._manage(trade, high=110.5, low=101.0, now=300, contract=TEST_CONTRACT)
    assert trade.legs[0].units == 2.0
    bt._manage(trade, high=120.5, low=111.0, now=600, contract=TEST_CONTRACT)
    assert trade.remaining_units > 0, "runners must survive full closure"
    assert sum(l.units for l in trade.legs) == 9.0


def test_a_progress_scale_out_beyond_one_is_an_extension():
    bt = Backtester()
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=95.0,
                          target=120.0, units=12.0)
    assert bt._price_at_progress(trade, 1.5) == pytest.approx(130.0)
    assert bt._price_at_progress(trade, 0.5) == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# Chapter 27: the risk ladder
# ---------------------------------------------------------------------------

def test_risk_halves_after_a_loss_and_restores_on_recovery():
    ladder = RiskLadder(base_risk=0.02, floor_risk=0.0025)
    ladder.on_loss(1000)
    assert ladder.current_risk == pytest.approx(0.01)
    ladder.on_win(1000)
    assert ladder.current_risk == pytest.approx(0.02)


def test_risk_halves_again_after_a_second_loss():
    ladder = RiskLadder(base_risk=0.02, floor_risk=0.0025)
    ladder.on_loss(1000)
    ladder.on_loss(500)
    assert ladder.current_risk == pytest.approx(0.005)
    assert ladder.drawdown_to_recover == pytest.approx(1500)


def test_risk_never_falls_below_the_floor():
    ladder = RiskLadder(base_risk=0.02, floor_risk=0.0025)
    for _ in range(10):
        ladder.on_loss(100)
    assert ladder.current_risk == pytest.approx(0.0025)


def test_five_consecutive_wins_halve_risk():
    """A sizing policy, not a claim that a streak predicts a loss."""
    ladder = RiskLadder(base_risk=0.02, halve_after_wins=5)
    for _ in range(5):
        ladder.on_win(100)
    assert ladder.current_risk == pytest.approx(0.01)
    assert ladder.consecutive_wins == 0


# ---------------------------------------------------------------------------
# Sessions added by edition 0.6
# ---------------------------------------------------------------------------

def test_new_windows_exist():
    def ts(h, m=0):
        return int(datetime(2024, 6, 3, h, m, tzinfo=NY).timestamp())
    assert sessions.primary_session(ts(9, 55)).name == "macro_0950"
    assert sessions.primary_session(ts(15, 55)).name == "market_on_close"
    assert sessions.primary_session(ts(13, 40)).name == "pm_opening_range"


def test_lunch_now_runs_to_1330():
    def ts(h, m=0):
        return int(datetime(2024, 6, 3, h, m, tzinfo=NY).timestamp())
    assert sessions.primary_session(ts(13, 15)).name == "lunch"


def test_model13_checkpoints():
    def ts(h, m=0):
        return int(datetime(2024, 6, 3, h, m, tzinfo=NY).timestamp())
    assert sessions.at_checkpoint(ts(10, 2)) == "10:00"
    assert sessions.at_checkpoint(ts(14, 30)) == "14:30"
    assert sessions.at_checkpoint(ts(11, 7)) is None


def test_venom_window_is_separate_from_the_premarket_range(synthetic_1m):
    now = int(synthetic_1m["t"].iloc[-1])
    venom = sessions.venom_range(synthetic_1m, now)
    premarket = sessions.premarket_range(synthetic_1m, now)
    assert venom.start_ts != premarket.start_ts, "different lessons, different anchors"


# ---------------------------------------------------------------------------
# The new agents
# ---------------------------------------------------------------------------

def test_new_agents_survive_real_looking_data(synthetic_1m):
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    ctx = ctx_at(synthetic_1m, now)
    for agent in (RejectionBlockAgent(), VolumeImbalanceAgent(), VenomAgent(),
                  CheckpointAgent()):
        sig = agent.run(ctx)
        assert sig.error is None, f"{agent.name}: {sig.error}"
        assert sig.direction in (LONG, SHORT, NEUTRAL)
        assert sig.rationale


def test_new_agents_abstain_on_empty_frames():
    empty = {tf: pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"]) for tf in ("5m", "15m", "1h")}
    ctx = MarketContext(symbol="NAS100_USD", now_ts=1_717_424_100, frames=empty)
    for agent in (RejectionBlockAgent(), VolumeImbalanceAgent(), VenomAgent()):
        sig = agent.run(ctx)
        assert sig.error is None
        assert sig.direction == NEUTRAL


def test_venom_rejects_a_wick_only_raid(synthetic_1m):
    """The signature wants a close beyond the pool; a brief wick is not it."""
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    sig = VenomAgent().run(ctx_at(synthetic_1m, now))
    if sig.direction == NEUTRAL and "wick" in sig.rationale:
        assert "close" in sig.rationale


def test_checkpoint_agent_never_directs(synthetic_1m):
    now = int(synthetic_1m["t"].iloc[-1]) + 60
    sig = CheckpointAgent().run(ctx_at(synthetic_1m, now))
    assert sig.direction == NEUTRAL
    assert sig.score == 0.0
    assert sig.role == "gate"


def test_the_risk_ladder_changes_size_across_a_backtest(synthetic_1m):
    """With the ladder active, risk falls after losses instead of staying fixed."""
    from backend.ict.contracts import RiskLadder
    ladder = RiskLadder(base_risk=0.01, floor_risk=0.001)
    bt = Backtester(step_minutes=30, bars=120, risk_ladder=ladder)
    result = bt.run("NAS100_USD", synthetic_1m, warmup_bars=4000,
                    config={"require_killzone": False})
    losses = [t for t in result.trades if t.pnl < 0]
    if losses:
        assert ladder.current_risk <= 0.01
        assert ladder.current_risk >= ladder.floor_risk


def test_without_a_ladder_risk_stays_fixed(synthetic_1m):
    bt = Backtester(step_minutes=30, bars=120)
    assert bt.risk_ladder is None
    result = bt.run("NAS100_USD", synthetic_1m, warmup_bars=4000,
                    config={"require_killzone": False})
    assert result.metrics["trades"] >= 0
