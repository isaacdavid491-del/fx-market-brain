"""Killzone and session tests, including daylight saving."""
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.ict import sessions

NY = ZoneInfo("America/New_York")


def ts(y, m, d, hh, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=NY).timestamp())


def test_silver_bullet_window():
    assert sessions.primary_session(ts(2024, 6, 3, 10, 30)).name == "silver_bullet_am"
    assert sessions.session_weight(ts(2024, 6, 3, 10, 30)) == 1.0


def test_new_york_open_window():
    assert sessions.primary_session(ts(2024, 6, 3, 9, 35)).name == "ny_open_killzone"


def test_lunch_is_low_weight():
    assert sessions.session_weight(ts(2024, 6, 3, 12, 0)) < 0.2


def test_weekend_detection():
    assert sessions.is_weekend(ts(2024, 6, 1, 12))          # Saturday
    assert sessions.is_weekend(ts(2024, 5, 31, 18))         # Friday after the close
    assert not sessions.is_weekend(ts(2024, 6, 2, 19))      # Sunday reopen
    assert not sessions.is_weekend(ts(2024, 6, 3, 10))      # Monday


def test_regular_trading_hours():
    assert sessions.in_rth(ts(2024, 6, 3, 10))
    assert not sessions.in_rth(ts(2024, 6, 3, 8))
    assert not sessions.in_rth(ts(2024, 6, 3, 16, 30))
    assert not sessions.in_rth(ts(2024, 6, 1, 10))          # Saturday


def test_daylight_saving_is_handled_by_the_zone():
    """10:30 New York is the silver bullet in both winter and summer, even
    though the UTC offset differs."""
    winter = ts(2024, 1, 10, 10, 30)     # EST, UTC-5
    summer = ts(2024, 7, 10, 10, 30)     # EDT, UTC-4
    assert sessions.primary_session(winter).name == "silver_bullet_am"
    assert sessions.primary_session(summer).name == "silver_bullet_am"
    assert sessions.to_ny(winter).utcoffset().total_seconds() == -5 * 3600
    assert sessions.to_ny(summer).utcoffset().total_seconds() == -4 * 3600


def test_asian_session_wraps_midnight():
    assert sessions.primary_session(ts(2024, 6, 3, 21)).name == "asia"
    assert sessions.primary_session(ts(2024, 6, 3, 23, 30)).name == "asia"


def test_session_bounds_are_ordered():
    start, end = sessions.session_bounds(ts(2024, 6, 3, 12), "silver_bullet_am")
    assert end - start == 3600
    assert sessions.to_ny(start).hour == 10


def test_ny_day_start_is_local_midnight():
    day = sessions.ny_day_start(ts(2024, 6, 3, 14, 12))
    local = sessions.to_ny(day)
    assert (local.hour, local.minute) == (0, 0)
    assert local.date() == datetime(2024, 6, 3).date()


def test_previous_day_levels(make_frame):
    import pandas as pd
    day2 = sessions.ny_day_start(ts(2024, 6, 4, 12))
    day1 = sessions.ny_day_start(ts(2024, 6, 3, 12))
    rows = []
    for i in range(24):
        rows.append({"t": day1 + i * 3600, "o": 100, "h": 100 + i, "l": 100 - i, "c": 100, "v": 1})
    for i in range(6):
        rows.append({"t": day2 + i * 3600, "o": 105, "h": 106, "l": 104, "c": 105, "v": 1})
    df = pd.DataFrame(rows)
    levels = sessions.previous_day_levels(df, ts(2024, 6, 4, 6))
    assert levels["pdh"] == 123
    assert levels["pdl"] == 77
    assert levels["day_open"] == 105
