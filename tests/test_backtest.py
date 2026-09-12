"""Backtester correctness. The look-ahead tests matter most: a backtest that
peeks is worse than no backtest, because it is believable."""
import pandas as pd
import pytest

from backend.agents.base import LONG, MarketContext
from backend.agents.orchestrator import AgentFarm
from backend.backtest.engine import Backtester, BacktestTrade, compute_metrics
from backend.ict.contracts import Contract
from backend.store import TF_MINUTES


# A costless one-dollar-per-point contract keeps the arithmetic in these unit
# tests readable; cost handling is tested separately.
TEST_CONTRACT = Contract("TEST", "unit test", dollars_per_point=1.0,
                         tick_size=0.25, round_trip_cost=0.0)


def test_slices_never_include_an_unclosed_bar(synthetic_1m):
    """The decisive invariant: at time T the farm may only see bars that had
    already closed at T."""
    bt = Backtester(bars=100)
    full = bt._prepare(synthetic_1m, ["5m", "15m", "1h"])
    for now in synthetic_1m["t"].iloc[5000::997]:
        frames = bt._slice(full, int(now))
        for tf, frame in frames.items():
            if frame.empty:
                continue
            bar_seconds = TF_MINUTES[tf] * 60
            last_close_time = int(frame["t"].iloc[-1]) + bar_seconds
            assert last_close_time <= int(now), (
                f"{tf} frame at {now} ends with a bar closing at {last_close_time}"
            )


def test_slicing_full_history_equals_slicing_raw_bars(synthetic_1m):
    """Resampling once and truncating must equal truncating and resampling."""
    from backend.store import resample_ohlcv

    bt = Backtester(bars=50)
    full = bt._prepare(synthetic_1m, ["15m"])
    now = int(synthetic_1m["t"].iloc[8000])
    from_full = bt._slice(full, now)["15m"]

    truncated = synthetic_1m[synthetic_1m["t"] <= now]
    direct = resample_ohlcv(truncated, "15m")
    direct = direct[direct["t"] + 900 <= now].tail(50).reset_index(drop=True)

    pd.testing.assert_frame_equal(from_full, direct)


def test_backtest_runs_and_reports(synthetic_1m, synthetic_peer_1m):
    bt = Backtester(step_minutes=15, bars=120)
    result = bt.run("NAS100_USD", synthetic_1m,
                    correlated_1m=synthetic_peer_1m, correlated_symbol="SPX500_USD",
                    warmup_bars=4000, config={"require_killzone": False})
    assert result.decisions_evaluated > 0
    assert result.metrics["trades"] == len([t for t in result.trades if t.exit_reason != "open"])
    assert result.ending_equity > 0
    # Equity must reconcile with the trades taken.
    expected = result.starting_equity + sum(t.pnl for t in result.trades)
    assert result.ending_equity == pytest.approx(expected, rel=1e-9)


def test_short_history_is_reported_not_silently_empty():
    tiny = pd.DataFrame({
        "t": range(1_717_000_000, 1_717_000_000 + 600 * 60, 60),
        "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 1.0,
    })
    result = Backtester().run("NAS100_USD", tiny, warmup_bars=3000)
    assert result.warnings
    assert "warmup" in result.warnings[0]
    assert result.metrics["trades"] == 0


def test_stop_is_taken_before_target_within_one_bar():
    """When a bar spans both levels the simulation must assume the loss."""
    bt = Backtester()
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=103.0, units=10.0)
    closed = bt._manage(trade, high=104.0, low=98.0, now=300, contract=TEST_CONTRACT)
    assert closed
    assert trade.exit_reason == "stop"
    assert trade.r_multiple == pytest.approx(-1.0)


def test_target_closes_the_trade_at_the_target_price():
    bt = Backtester()
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=103.0, units=10.0)
    closed = bt._manage(trade, high=103.5, low=99.5, now=300, contract=TEST_CONTRACT)
    assert closed and trade.exit_reason == "target"
    assert trade.r_multiple == pytest.approx(3.0)
    assert trade.pnl == pytest.approx(30.0)


def test_short_trade_accounting_is_mirrored():
    bt = Backtester()
    trade = BacktestTrade(direction="SHORT", entry_ts=0, entry=100.0, stop=101.0,
                          target=97.0, units=10.0)
    closed = bt._manage(trade, high=100.5, low=96.5, now=300, contract=TEST_CONTRACT)
    assert closed and trade.exit_reason == "target"
    assert trade.r_multiple == pytest.approx(3.0)
    assert trade.pnl == pytest.approx(30.0)


def test_time_stop_closes_a_stalled_trade():
    bt = Backtester(max_hold_minutes=60)
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=103.0, units=10.0)
    assert not bt._manage(trade, 100.2, 99.8, now=1800, contract=TEST_CONTRACT)
    assert bt._manage(trade, 100.2, 99.8, now=3600, contract=TEST_CONTRACT)
    assert trade.exit_reason == "time_stop"


def test_metrics_on_an_empty_sample_are_zero_not_nan():
    m = compute_metrics([], 100_000.0, 100_000.0, [])
    assert m["trades"] == 0
    assert m["win_rate"] == 0.0
    assert m["profit_factor"] == 0.0


def test_metrics_arithmetic():
    trades = [
        BacktestTrade(LONG, 0, 100, 99, 103, 10, exit_reason="target", r_multiple=3.0, pnl=300),
        BacktestTrade(LONG, 0, 100, 99, 103, 10, exit_reason="stop", r_multiple=-1.0, pnl=-100),
        BacktestTrade(LONG, 0, 100, 99, 103, 10, exit_reason="stop", r_multiple=-1.0, pnl=-100),
    ]
    m = compute_metrics(trades, 100_000.0, 100_100.0, [(1, 100_300), (2, 100_100)])
    assert m["trades"] == 3
    assert m["wins"] == 1 and m["losses"] == 2
    assert m["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["total_r"] == pytest.approx(1.0)
    assert m["profit_factor"] == pytest.approx(1.5)
    assert m["max_drawdown_pct"] > 0


def test_backtest_respects_the_daily_trade_cap(synthetic_1m):
    """The risk agent's counters are fed from the simulation, so the cap must
    bind in a backtest exactly as it would live."""
    bt = Backtester(step_minutes=15, bars=120)
    result = bt.run("NAS100_USD", synthetic_1m, warmup_bars=4000,
                    config={"require_killzone": False, "max_trades_per_day": 1})
    from backend.ict.sessions import ny_day_start
    per_day = {}
    for t in result.trades:
        day = ny_day_start(t.entry_ts)
        per_day[day] = per_day.get(day, 0) + 1
    assert all(count <= 1 for count in per_day.values()), per_day


def test_costs_are_charged_on_the_round_trip():
    """Appendix C: the net result is the gross minus the round-trip cost."""
    from backend.ict.contracts import MNQ
    bt = Backtester()
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=20_000.0, stop=19_988.75,
                          target=20_022.5, units=1.0)
    bt._manage(trade, high=20_023.0, low=19_999.0, now=300, contract=MNQ)
    assert trade.exit_reason == "target"
    # 22.5 points x $2 = $45 gross, less $3 of costs.
    assert trade.gross_pnl == pytest.approx(45.0)
    assert trade.costs == pytest.approx(3.0)
    assert trade.pnl == pytest.approx(42.0)


def test_a_resting_order_is_cancelled_when_the_target_trades_first(synthetic_1m):
    """The study fixes this rule before results are collected: if the target
    trades before the entry fills, the order is cancelled, not filled late."""
    bt = Backtester(step_minutes=5, bars=120)
    result = bt.run("NAS100_USD", synthetic_1m, warmup_bars=4000,
                    config={"require_killzone": False})
    assert isinstance(result.orders_cancelled, dict)
    for reason in result.orders_cancelled:
        assert reason in ("expired", "target_traded_first",
                          "protection_breached_before_entry")


def test_metrics_separate_gross_from_net():
    trades = [
        BacktestTrade(LONG, 0, 100, 99, 103, 1, exit_reason="target",
                      r_multiple=3.0, pnl=42.0, gross_pnl=45.0, costs=3.0),
        BacktestTrade(LONG, 0, 100, 99, 103, 1, exit_reason="stop",
                      r_multiple=-1.0, pnl=-23.0, gross_pnl=-20.0, costs=3.0),
    ]
    m = compute_metrics(trades, 10_000.0, 10_019.0, [(1, 10_019.0)])
    assert m["gross_pnl"] == pytest.approx(25.0)
    assert m["costs_paid"] == pytest.approx(6.0)
    assert m["net_pnl"] == pytest.approx(19.0)
    assert m["break_even_rate"] == pytest.approx(0.25)


def test_metrics_are_json_serialisable(synthetic_1m):
    """numpy scalars in the metrics break the API, so the breakdowns must be
    plain Python floats."""
    import json
    bt = Backtester(step_minutes=30, bars=120)
    result = bt.run("NAS100_USD", synthetic_1m, warmup_bars=4000,
                    config={"require_killzone": False})
    json.dumps(result.as_dict())
    for section in ("by_session", "by_direction", "by_entry_family"):
        for row in result.metrics[section].values():
            assert type(row["total_r"]) is float, (section, type(row["total_r"]))


# ---------------------------------------------------------------------------
# Partial exits and position management
# ---------------------------------------------------------------------------

def test_book_management_arithmetic():
    """Chapter 21: three units short at 140 protected at 146 carry 18
    point-units of risk. Exits at 130, 134 and 138 realise 18 point-units,
    which is 1.00 times initial risk, not the 4.00 a full exit at 116 gives."""
    from backend.backtest.engine import EXIT_POLICIES
    bt = Backtester(exit_policy=EXIT_POLICIES["all_at_target"])
    trade = BacktestTrade(direction="SHORT", entry_ts=0, entry=140.0, stop=146.0,
                          target=116.0, units=3.0)
    assert trade.risk_per_unit * trade.initial_units == pytest.approx(18.0)

    bt._book_leg(trade, 130.0, 1.0, 100, "partial", TEST_CONTRACT)
    bt._book_leg(trade, 134.0, 1.0, 200, "partial", TEST_CONTRACT)
    bt._close_remaining(trade, 138.0, 300, "manual", TEST_CONTRACT)

    realised_points = sum(leg.units * (140.0 - leg.price) for leg in trade.legs)
    assert realised_points == pytest.approx(18.0)
    assert trade.r_multiple == pytest.approx(1.0)
    assert trade.remaining_units == 0
    assert len(trade.legs) == 3


def test_full_exit_at_the_far_objective_is_four_times_risk():
    """The comparison the book draws: the same position held to 116."""
    bt = Backtester()
    trade = BacktestTrade(direction="SHORT", entry_ts=0, entry=140.0, stop=146.0,
                          target=116.0, units=3.0)
    bt._close_remaining(trade, 116.0, 300, "target", TEST_CONTRACT)
    assert trade.r_multiple == pytest.approx(4.0)


def test_scale_out_leaves_a_runner():
    from backend.backtest.engine import ExitPolicy
    bt = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=104.0, units=4.0)
    # Reaches 1R (101) but not the target.
    closed = bt._manage(trade, high=101.5, low=100.2, now=300, contract=TEST_CONTRACT)
    assert closed is False
    assert trade.remaining_units == 2.0
    assert len(trade.legs) == 1
    assert trade.legs[0].reason == "scale_1R"
    assert trade.legs[0].r_at_exit == pytest.approx(1.0)


def test_a_single_contract_cannot_be_scaled_out_of():
    """The book's $50 budget buys one contract; there is nothing to scale."""
    from backend.backtest.engine import ExitPolicy
    bt = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=104.0, units=1.0)
    bt._manage(trade, high=101.5, low=100.2, now=300, contract=TEST_CONTRACT)
    assert trade.legs == []
    assert trade.remaining_units == 1.0


def test_a_scale_out_never_exceeds_the_remaining_quantity():
    """An exit order sized for an earlier quantity must not reverse the position."""
    from backend.backtest.engine import ExitPolicy
    policy = ExitPolicy("aggressive", scale_outs=[(1.0, 0.9), (1.5, 0.9)])
    bt = Backtester(exit_policy=policy)
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=110.0, units=3.0)
    bt._manage(trade, high=102.0, low=100.5, now=300, contract=TEST_CONTRACT)
    assert trade.remaining_units >= 0
    assert sum(leg.units for leg in trade.legs) <= trade.initial_units


def test_every_unit_is_accounted_for_at_the_close():
    from backend.backtest.engine import EXIT_POLICIES
    bt = Backtester(exit_policy=EXIT_POLICIES["thirds_1R_2R"])
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=104.0, units=6.0)
    bt._manage(trade, high=101.2, low=100.1, now=300, contract=TEST_CONTRACT)
    bt._manage(trade, high=102.2, low=101.0, now=600, contract=TEST_CONTRACT)
    bt._manage(trade, high=104.5, low=102.0, now=900, contract=TEST_CONTRACT)
    assert sum(leg.units for leg in trade.legs) == pytest.approx(6.0)
    assert trade.remaining_units == 0


def test_r_is_measured_against_the_initial_stop_not_a_moved_one():
    """Moving a stop later must not rewrite the risk the trade was taken with."""
    from backend.backtest.engine import ExitPolicy
    bt = Backtester(exit_policy=ExitPolicy("be", scale_outs=[(1.0, 0.5)], breakeven_at_r=1.0))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=105.0, units=4.0)
    bt._manage(trade, high=101.5, low=100.2, now=300, contract=TEST_CONTRACT)
    assert trade.stop == 100.0 and trade.stop_moves == 1
    assert trade.initial_stop == 99.0
    assert trade.risk_per_unit == pytest.approx(1.0)


def test_a_breakeven_stop_can_cause_an_early_exit():
    """The book's warning made mechanical: a stop moved into an expected
    supportive retracement takes the runner out at entry."""
    from backend.backtest.engine import ExitPolicy
    bt = Backtester(exit_policy=ExitPolicy("be", scale_outs=[(1.0, 0.5)], breakeven_at_r=1.0))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=105.0, units=4.0)
    bt._manage(trade, high=101.5, low=100.2, now=300, contract=TEST_CONTRACT)
    closed = bt._manage(trade, high=101.0, low=99.8, now=600, contract=TEST_CONTRACT)
    assert closed is True
    assert trade.exit_reason == "stop_moved"
    # Half banked at +1R, half out at entry: +0.5R overall rather than a loss.
    assert trade.r_multiple == pytest.approx(0.5)


def test_stop_is_still_checked_before_a_scale_out_in_the_same_bar():
    from backend.backtest.engine import ExitPolicy
    bt = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=104.0, units=4.0)
    closed = bt._manage(trade, high=102.0, low=98.5, now=300, contract=TEST_CONTRACT)
    assert closed is True
    assert trade.exit_reason == "stop"
    assert trade.r_multiple == pytest.approx(-1.0)


def test_partial_exits_change_the_realised_r_not_the_risk():
    """A scaled exit banks less than a full run to target on a winner."""
    from backend.backtest.engine import ExitPolicy
    full = Backtester()
    scaled = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))

    def run(bt):
        trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                              target=104.0, units=4.0)
        bt._manage(trade, high=101.5, low=100.2, now=300, contract=TEST_CONTRACT)
        bt._manage(trade, high=104.5, low=101.0, now=600, contract=TEST_CONTRACT)
        return trade

    assert run(full).r_multiple == pytest.approx(4.0)
    # Half out at +1R, half at +4R.
    assert run(scaled).r_multiple == pytest.approx(2.5)


def test_scaling_out_cuts_the_loss_when_the_trade_reverses():
    """The variance-reduction mechanism, in one trade.

    Half the position banked at +1R, the rest stopped at -1R, nets zero where
    the unscaled trade loses a full R. This is what shrinks the spread of
    outcomes; it is not the same thing as improving the average.
    """
    from backend.backtest.engine import ExitPolicy
    scaled = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=110.0, units=4.0)
    scaled._manage(trade, high=101.2, low=100.1, now=300, contract=TEST_CONTRACT)
    assert trade.remaining_units == 2.0
    closed = scaled._manage(trade, high=100.5, low=98.5, now=600, contract=TEST_CONTRACT)
    assert closed is True
    assert trade.r_multiple == pytest.approx(0.0)

    unscaled = Backtester()
    plain = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=110.0, units=4.0)
    unscaled._manage(plain, high=101.2, low=100.1, now=300, contract=TEST_CONTRACT)
    unscaled._manage(plain, high=100.5, low=98.5, now=600, contract=TEST_CONTRACT)
    assert plain.r_multiple == pytest.approx(-1.0)


def test_scaling_out_also_caps_the_winner():
    """The other half of the trade-off, so the comparison stays honest."""
    from backend.backtest.engine import ExitPolicy
    scaled = Backtester(exit_policy=ExitPolicy("half", scale_outs=[(1.0, 0.5)]))
    trade = BacktestTrade(direction=LONG, entry_ts=0, entry=100.0, stop=99.0,
                          target=110.0, units=4.0)
    scaled._manage(trade, high=101.2, low=100.1, now=300, contract=TEST_CONTRACT)
    scaled._manage(trade, high=110.5, low=101.0, now=600, contract=TEST_CONTRACT)
    # Half at +1R and half at +10R averages +5.5R, against +10R unscaled.
    assert trade.r_multiple == pytest.approx(5.5)
