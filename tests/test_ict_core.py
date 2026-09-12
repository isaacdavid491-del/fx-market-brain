"""Tests for the ICT primitives: these encode what each concept means."""
import pandas as pd
import pytest

from backend.ict import core


def test_find_swings_locates_obvious_pivots(make_frame):
    # A clean peak at index 3 and a clean trough at index 7.
    df = make_frame([
        (10, 11, 9, 10), (11, 12, 10, 11), (12, 14, 11, 13), (13, 20, 12, 19),
        (19, 19, 15, 16), (16, 17, 13, 14), (14, 15, 11, 12), (12, 13, 5, 6),
        (6, 9, 6, 8), (8, 11, 7, 10),
    ])
    swings = core.find_swings(df, strength=2)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    assert any(s.idx == 3 and s.price == 20 for s in highs)
    assert any(s.idx == 7 and s.price == 5 for s in lows)


def test_swings_need_confirmation_on_both_sides(make_frame):
    df = make_frame([(10, 11, 9, 10)] * 4)
    assert core.find_swings(df, strength=2) == []


def test_structure_bos_then_choch(make_frame):
    """Up leg, pullback, higher break (BOS), then a break of the low (CHoCH)."""
    bars = [
        (100, 102, 99, 101), (101, 104, 100, 103), (103, 108, 102, 107),
        (107, 110, 106, 109),                                   # swing high 110
        (109, 109, 104, 105), (105, 106, 101, 102),             # pullback, swing low 101
        (102, 104, 101, 103), (103, 107, 102, 106),
        (106, 112, 105, 111),                                   # closes above 110 -> BOS
        (111, 113, 110, 112), (112, 112, 108, 109),
        (109, 110, 105, 106), (106, 107, 99, 100),              # closes below the low -> CHoCH
        (100, 101, 96, 97), (97, 98, 94, 95),
    ]
    df = make_frame(bars)
    events, state = core.market_structure(df, strength=2)
    kinds = [(e.kind, e.direction) for e in events]
    assert ("BOS", core.BULLISH) in kinds
    assert ("CHoCH", core.BEARISH) in kinds
    # The change of character must come after the continuation break.
    bos_i = next(i for i, e in enumerate(events) if e.kind == "BOS" and e.direction == core.BULLISH)
    choch_i = next(i for i, e in enumerate(events) if e.kind == "CHoCH" and e.direction == core.BEARISH)
    assert choch_i > bos_i
    assert state["trend"] == core.BEARISH


def test_structure_breaks_on_close_not_wick(make_frame):
    """A wick through a swing high is a raid, not a break of structure."""
    bars = [
        (100, 101, 99, 100), (100, 103, 99, 102), (102, 110, 101, 104),   # swing high 110
        (104, 105, 102, 103), (103, 104, 100, 101),
        (101, 109, 100, 102), (102, 112, 101, 103),   # wick above 110, closes below
        (103, 104, 101, 102), (102, 103, 100, 101),
    ]
    df = make_frame(bars)
    events, _ = core.market_structure(df, strength=2)
    bullish = [e for e in events if e.direction == core.BULLISH and e.level == 110]
    assert bullish == [], "a wick above the swing high must not register as a break"


def test_bullish_fvg_detected_and_filled(make_frame):
    # Bar 2's low (20) sits above bar 0's high (12): a bullish imbalance.
    df = make_frame([
        (10, 12, 9, 11), (12, 22, 11, 21), (21, 25, 20, 24),
        (24, 26, 21, 25),                     # overlaps bar 1, so leaves no second gap
        (25, 25, 19, 20),                     # dips into the gap without filling it
    ])
    gaps = core.find_fvgs(df)
    bull = [g for g in gaps if g.direction == core.BULLISH]
    assert len(bull) == 1
    gap = bull[0]
    assert gap.bottom == 12 and gap.top == 20
    assert gap.touched is True
    assert gap.filled is False


def test_bearish_fvg_and_full_fill(make_frame):
    df = make_frame([
        (30, 31, 28, 29), (28, 29, 19, 20), (20, 21, 18, 19),
        (19, 29, 18, 28),   # closes back above the gap top -> filled
    ])
    gaps = core.find_fvgs(df)
    bear = [g for g in gaps if g.direction == core.BEARISH]
    assert len(bear) == 1
    assert bear[0].bottom == 21 and bear[0].top == 28
    assert bear[0].filled is True


def test_no_fvg_when_bars_overlap(make_frame):
    df = make_frame([(10, 15, 9, 14), (14, 18, 13, 17), (17, 20, 12, 19)])
    assert core.find_fvgs(df) == []


def test_order_block_is_last_opposing_candle_before_the_break(make_frame):
    bars = [
        (100, 102, 99, 101), (101, 103, 100, 102), (102, 106, 101, 105),
        (105, 107, 104, 106),                          # swing high 107
        (106, 106, 102, 103), (103, 104, 100, 101),
        (101, 102, 99, 100),                           # down candle: the order block
        (100, 112, 100, 111),                          # displacement through 107
        (111, 113, 110, 112), (112, 114, 111, 113),
    ]
    df = make_frame(bars)
    events, _ = core.market_structure(df, strength=2)
    blocks = core.find_order_blocks(df, events)
    bull = [b for b in blocks if b.direction == core.BULLISH]
    assert bull, "a bullish break should leave a bullish order block"
    assert bull[0].idx == 6
    assert bull[0].bottom == 99 and bull[0].top == 102


def test_liquidity_pool_clusters_equal_highs(make_frame):
    bars = []
    for _ in range(3):
        bars += [
            (100, 101, 99, 100), (100, 102, 99, 101), (101, 110, 100, 105),  # high ~110
            (105, 106, 103, 104), (104, 105, 95, 96),
        ]
    df = make_frame(bars)
    pools = core.liquidity_pools(df, tolerance=1.0)
    buyside = [p for p in pools if p.side == "buyside"]
    assert buyside, "equal highs should form a buyside pool"
    assert max(p.touches for p in buyside) >= 2


def test_sweep_requires_running_the_level(make_frame):
    bars = [
        (100, 101, 99, 100), (100, 103, 99, 102), (102, 110, 101, 104),   # swing high 110
        (104, 105, 102, 103), (103, 104, 101, 102),
        (102, 115, 101, 103),   # runs above 110 and closes back below: a sweep
        (103, 104, 101, 102),
    ]
    df = make_frame(bars)
    sweeps = core.recent_sweeps(df, lookback=10, buffer=0.5)
    buyside = [s for s in sweeps if s.side == "buyside"]
    assert buyside
    assert buyside[0].reclaimed is True
    assert buyside[0].implied_direction == core.BEARISH


def test_dealing_range_and_ote_band(make_frame):
    df = make_frame([(100, 120, 80, 110)] + [(110, 111, 109, 110)] * 5)
    rng = core.dealing_range(df, lookback=10)
    assert rng.high == 120 and rng.low == 80
    assert rng.equilibrium == 100
    assert rng.zone(110) == "premium"
    assert rng.zone(90) == "discount"
    low, high = core.ote_zone(rng, core.BULLISH)
    # 0.62-0.79 retracement of a 40-point range measured down from the high.
    assert low == pytest.approx(120 - 0.79 * 40)
    assert high == pytest.approx(120 - 0.62 * 40)


def test_atr_is_positive_on_real_looking_data(synthetic_1m):
    value = core.atr(synthetic_1m.tail(500)).value
    assert value > 0


def test_primitives_tolerate_empty_and_tiny_frames():
    empty = pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    assert core.find_swings(empty) == []
    assert core.find_fvgs(empty) == []
    assert core.find_order_blocks(empty) == []
    assert core.liquidity_pools(empty) == []
    assert core.recent_sweeps(empty) == []
    assert core.dealing_range(empty) is None
    events, state = core.market_structure(empty)
    assert events == [] and state["trend"] == "neutral"


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        core.find_swings(pd.DataFrame({"t": [1], "close": [2]}))
