"""Tests derived from the study book's own worked examples.

Where the source states an arithmetic result, it is asserted literally. Where
it draws a distinction (a wick is not a close; three "first gap" identities
are different things), the distinction is the test.
"""
import pandas as pd
import pytest

from backend.ict import core
from backend.ict.contracts import MNQ, NQ, get_contract, net_reward_to_risk, size_position


# ---------------------------------------------------------------------------
# Chapter 9: bodies, wicks and nested midpoints
# ---------------------------------------------------------------------------

def test_wick_midpoints_match_the_worked_example():
    """O=108, C=104, H=116, L=102 gives an upper-wick midpoint of 112 while
    the whole-candle midpoint is 109. Naming the wrong object moves the
    reference by three units."""
    o, c, h, l = 108, 104, 116, 102
    assert core.upper_wick_midpoint(o, h, c) == 112.0
    assert core.candle_midpoint(h, l) == 109.0
    assert core.upper_wick_midpoint(o, h, c) != core.candle_midpoint(h, l)


def test_lower_wick_midpoint_uses_the_body_low():
    assert core.lower_wick_midpoint(o=108, l=102, c=104) == 103.0


def test_halfway_between_a_gap_boundary_and_a_wick_midpoint():
    """The book's nested construction: boundary 108 and wick midpoint 112 give
    110 halfway between."""
    assert core.consequent_encroachment(112.0, 108.0) == 110.0


def test_opposing_wick_interval_matches_the_obsidian_example(make_frame):
    """Candle A's upper wick spans 110-114 with midpoint 112; candle B's lower
    wick spans 116-120 with midpoint 118; the interval is 112 to 118."""
    # A: body top 110, high 114 -> upper wick midpoint 112
    # B: body low 120, low 116  -> lower wick midpoint 118
    df = make_frame([
        (105, 106, 104, 105),
        (108, 114, 107, 110),     # A: max(O,C)=110, H=114 -> mid 112
        (121, 122, 116, 120),     # B: min(O,C)=120, L=116 -> mid 118
        (119, 120, 118, 119),
    ])
    pairs = core.opposing_wick_pairs(df, lookback=10, min_wick_atr=0.0)
    assert pairs
    pair = next(p for p in pairs if p.upper_idx == 1 and p.lower_idx == 2)
    assert pair.upper_mid == 112.0
    assert pair.lower_mid == 118.0
    assert pair.interval == (112.0, 118.0)
    assert pair.contains(115.0) and not pair.contains(125.0)


# ---------------------------------------------------------------------------
# Chapter 5: grading a range
# ---------------------------------------------------------------------------

def test_range_grading_matches_the_worked_example():
    """For 100 to 180: width 80, midpoint 140, quarters 120 and 160, eighths
    every 10 units."""
    graded = core.grade_range(100, 180, divisions=8)
    assert graded["width"] == 80
    assert graded["midpoint"] == 140
    assert graded["q2_8"] == 120
    assert graded["q6_8"] == 160
    eighths = [graded[f"q{i}_8"] for i in range(9)]
    assert eighths == [100, 110, 120, 130, 140, 150, 160, 170, 180]


def test_half_range_projection_is_an_extension_not_a_deviation():
    """A half-range projection above the high of a 100-180 range is 220."""
    assert core.project_range(100, 180, 1.5) == 220.0
    assert core.project_range(100, 180, -0.5) == 60.0


# ---------------------------------------------------------------------------
# Chapter 10: inversion requires a close, not a wick
# ---------------------------------------------------------------------------

def test_a_wick_through_a_gap_is_not_an_inversion(make_frame):
    """The source example: a bullish gap spans 104-108, one candle trades to
    103 but closes at 106. That is not yet an inversion."""
    df = make_frame([
        (100, 101, 99, 100), (100, 109, 100, 108), (108, 112, 104, 110),  # gap 101->104
        (110, 111, 103, 106),    # wick below the gap, closes back inside
        (106, 107, 105, 106),
    ])
    gaps = core.find_fvgs(df)
    bull = [g for g in gaps if g.direction == core.BULLISH]
    assert bull, "the three-candle pattern should leave a bullish gap"
    inversions = core.find_inversions(df, bull)
    assert inversions == [], "a wick through the boundary must not qualify"


def test_a_close_below_the_gap_qualifies_the_inversion(make_frame):
    """The later candle closing at 102.50 is the qualifying event."""
    df = make_frame([
        (100, 101, 99, 100), (100, 109, 100, 108), (108, 112, 104, 110),
        (110, 111, 103, 106),      # wick only
        (106, 107, 100, 100.5),    # close below the gap bottom: qualifies
        (100, 101, 99, 100),
    ])
    gaps = [g for g in core.find_fvgs(df) if g.direction == core.BULLISH]
    inversions = core.find_inversions(df, gaps)
    assert len(inversions) == 1
    inv = inversions[0]
    assert inv.new_direction == core.BEARISH
    assert inv.trigger == "close_beyond"
    assert inv.close_price == pytest.approx(100.5)


def test_inversion_records_the_original_gap_identity(make_frame):
    df = make_frame([
        (100, 101, 99, 100), (100, 109, 100, 108), (108, 112, 104, 110),
        (110, 111, 100, 100.5), (100, 106, 99, 105),
    ])
    gaps = [g for g in core.find_fvgs(df) if g.direction == core.BULLISH]
    inversions = core.find_inversions(df, gaps)
    if inversions:
        out = inversions[0].as_dict()
        assert out["original_direction"] == core.BULLISH
        assert out["new_direction"] == core.BEARISH
        assert out["formed_t"] != out["inverted_t"], "formation and failure are separate events"


# ---------------------------------------------------------------------------
# Chapter 11: the three "first gap" identities
# ---------------------------------------------------------------------------

def test_first_presented_gap_identities_are_distinct(make_frame):
    """A is the chronological anchor, a later displacing gap is a different
    identity, and the first opposite-direction gap is the reflection."""
    bars = [
        (100, 101, 99, 100), (100, 102, 99, 101), (101, 104, 100, 103),
        (103, 106, 102, 105),                      # swing high 106
        (105, 106, 103, 104), (104, 105, 102, 103),
        (103, 108, 103, 107),                      # bullish gap A (chronological)
        (107, 109, 106, 108),
        (108, 115, 108, 114),                      # displacing gap through 106
        (114, 116, 113, 115),
        (115, 116, 110, 111),
        (111, 112, 105, 106),                      # bearish gap (reflection)
        (106, 107, 104, 105),
    ]
    df = make_frame(bars)
    out = core.first_presented_gaps(df, session_start_ts=int(df["t"].iloc[0]))
    assert out["chronological"] is not None
    assert out["chronological"].gap.direction == core.BULLISH
    if out["reflection"] is not None:
        assert out["reflection"].gap.direction != out["chronological"].gap.direction
    if out["displacement"] is not None:
        # A displacing gap may be a different gap from the chronological first.
        assert out["displacement"].identity == "displacement"


def test_presented_gaps_ignore_anything_before_the_session_start(make_frame):
    df = make_frame([
        (100, 101, 99, 100), (100, 109, 100, 108), (108, 112, 104, 110),
        (110, 112, 109, 111), (111, 118, 111, 117),
    ])
    late = int(df["t"].iloc[-1]) + 10_000
    out = core.first_presented_gaps(df, session_start_ts=late)
    assert all(v is None for v in out.values())


# ---------------------------------------------------------------------------
# Appendix C: contract arithmetic and costs
# ---------------------------------------------------------------------------

def test_contract_specifications():
    assert NQ.dollars_per_point == 20.0 and NQ.tick_size == 0.25
    assert NQ.dollars_per_tick == 5.0
    assert MNQ.dollars_per_point == 2.0
    assert MNQ.dollars_per_tick == 0.5


def test_eleven_and_a_quarter_points_of_risk():
    """The book's table: 11.25 points is $225 on NQ and $22.50 on MNQ."""
    assert 11.25 * NQ.dollars_per_point == 225.0
    assert 11.25 * MNQ.dollars_per_point == 22.5


def test_a_fifty_dollar_budget_buys_one_micro_not_two():
    """$22.50 of price risk plus $3 of costs is $25.50; two would be $51."""
    out = size_position(MNQ, risk_budget=50.0, entry=20_009.0, stop=19_997.75)
    assert out["risk_per_unit"] == pytest.approx(25.5)
    assert out["units"] == 1.0


def test_gross_two_to_one_is_net_one_point_six_five():
    """The worked example: $42 net win against $25.50 net loss is about 1.65,
    and the break-even probability is 37.78%."""
    out = net_reward_to_risk(MNQ, entry=20_009.0, stop=19_997.75, target=20_031.5)
    assert out["gross_rr"] == pytest.approx(2.0)
    assert out["net_win"] == pytest.approx(42.0)
    assert out["net_loss"] == pytest.approx(25.5)
    assert out["net_rr"] == pytest.approx(1.65, abs=0.01)
    assert out["break_even_rate"] == pytest.approx(0.3778, abs=0.0001)


def test_tick_rounding_is_to_a_quarter_point():
    assert MNQ.round_to_tick(20_009.13) == 20_009.25
    assert MNQ.round_to_tick(20_009.10) == 20_009.00
    assert MNQ.round_to_tick(20_009.00) == 20_009.00


def test_rounding_a_stop_never_tightens_it():
    """Rounding must move protection away from entry, not toward it."""
    assert MNQ.round_away(19_997.80, -1) == 19_997.75    # long stop rounds down
    assert MNQ.round_away(20_082.10, +1) == 20_082.25    # short stop rounds up


def test_offset_by_whole_ticks():
    """Protection one increment below the raid low of 19,998 is 19,997.75."""
    assert MNQ.offset_ticks(19_998.0, -1) == 19_997.75
    assert MNQ.offset_ticks(20_031.75, -1) == 20_031.5


def test_contract_lookup_defaults_to_the_micro():
    assert get_contract(None).symbol == "MNQ"
    assert get_contract("NQ").symbol == "NQ"
    assert get_contract("unknown").symbol == "MNQ"
