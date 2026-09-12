"""Killzones, sessions and day anchors, in New York time.

NASDAQ index futures trade nearly around the clock, but the ICT model treats
only a few windows as high-probability. Everything here is expressed in
America/New_York because that is the reference clock for the equity session,
and the zone handles daylight saving for us.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Session:
    name: str
    start: time
    end: time
    weight: float      # how much the farm trusts signals inside this window
    note: str

    def contains(self, moment: datetime) -> bool:
        local = moment.astimezone(NY).time()
        if self.start <= self.end:
            return self.start <= local < self.end
        # Window wraps past midnight (Asia).
        return local >= self.start or local < self.end


# Ordered most- to least-specific; the first match wins for the primary label.
SESSIONS: Tuple[Session, ...] = (
    Session("silver_bullet_am", time(10, 0), time(11, 0), 1.00,
            "NY AM Silver Bullet: the highest-conviction one-hour window."),
    Session("ny_open_killzone", time(9, 30), time(10, 0), 0.95,
            "NY equity open: the day's manipulation leg usually resolves here."),
    Session("ny_am_killzone", time(7, 0), time(9, 30), 0.80,
            "NY AM killzone: pre-open positioning and the Judas swing."),
    Session("silver_bullet_pm", time(14, 0), time(15, 0), 0.70,
            "PM Silver Bullet: the afternoon continuation window."),
    Session("ny_pm_killzone", time(13, 30), time(16, 0), 0.60,
            "NY PM: weaker, prone to reversals into the close."),
    Session("silver_bullet_london", time(3, 0), time(4, 0), 0.65,
            "London Silver Bullet, the 03:00-04:00 window of the 2023 lesson."),
    Session("london_killzone", time(2, 0), time(5, 0), 0.55,
            "London killzone: sets the session high or low NASDAQ later raids."),
    Session("lunch", time(11, 30), time(13, 0), 0.15,
            "NY lunch: low participation, avoid initiating."),
    Session("asia", time(20, 0), time(0, 0), 0.25,
            "Asian range: consolidation that frames the London raid."),
)

# Regular trading hours for the cash index.
RTH_START, RTH_END = time(9, 30), time(16, 0)


def to_ny(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(NY)


def active_sessions(ts: int) -> List[Session]:
    moment = to_ny(ts)
    return [s for s in SESSIONS if s.contains(moment)]


def primary_session(ts: int) -> Optional[Session]:
    found = active_sessions(ts)
    return found[0] if found else None


def session_weight(ts: int) -> float:
    """0.0 outside every named window, rising toward 1.0 in the best ones."""
    found = active_sessions(ts)
    return max((s.weight for s in found), default=0.0)


def in_rth(ts: int) -> bool:
    moment = to_ny(ts)
    if moment.weekday() >= 5:
        return False
    return RTH_START <= moment.time() < RTH_END


def is_weekend(ts: int) -> bool:
    moment = to_ny(ts)
    if moment.weekday() == 5:
        return True
    if moment.weekday() == 6:
        return moment.time() < time(18, 0)   # futures reopen Sunday 18:00 NY
    if moment.weekday() == 4:
        return moment.time() >= time(17, 0)
    return False


def ny_day_start(ts: int) -> int:
    """Epoch seconds of midnight NY for the day containing `ts`."""
    moment = to_ny(ts)
    midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(midnight.astimezone(timezone.utc).timestamp())


def session_bounds(ts: int, session_name: str) -> Optional[Tuple[int, int]]:
    """UTC epoch bounds of a named session on the NY day containing `ts`."""
    session = next((s for s in SESSIONS if s.name == session_name), None)
    if session is None:
        return None
    day = to_ny(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    start = day.replace(hour=session.start.hour, minute=session.start.minute)
    end = day.replace(hour=session.end.hour, minute=session.end.minute)
    if session.end <= session.start:
        end = end + timedelta(days=1)
    return (int(start.astimezone(timezone.utc).timestamp()),
            int(end.astimezone(timezone.utc).timestamp()))


def slice_session(df: pd.DataFrame, ts: int, session_name: str) -> pd.DataFrame:
    bounds = session_bounds(ts, session_name)
    if bounds is None or df.empty:
        return df.iloc[0:0]
    start, end = bounds
    return df[(df["t"] >= start) & (df["t"] < end)]


def previous_day_levels(df: pd.DataFrame, ts: int) -> Dict[str, Optional[float]]:
    """Previous NY day's high, low and close: PDH/PDL are standard draws."""
    if df.empty:
        return {"pdh": None, "pdl": None, "pdc": None, "day_open": None}
    today_start = ny_day_start(ts)
    prev_start = ny_day_start(today_start - 3600)

    prev = df[(df["t"] >= prev_start) & (df["t"] < today_start)]
    today = df[df["t"] >= today_start]
    return {
        "pdh": float(prev["h"].max()) if not prev.empty else None,
        "pdl": float(prev["l"].min()) if not prev.empty else None,
        "pdc": float(prev["c"].iloc[-1]) if not prev.empty else None,
        "day_open": float(today["o"].iloc[0]) if not today.empty else None,
    }


# ---------------------------------------------------------------------------
# Named intraday ranges
# ---------------------------------------------------------------------------
# The study book keeps these strictly apart: a premarket range, an opening
# range and an opening gap are three different measurements that can have
# three different midpoints on the same day. Each is frozen once its window
# closes, and an anchor is never moved afterwards.

PREMARKET_START, PREMARKET_END = time(7, 0), time(9, 0)
OPENING_RANGE_START, OPENING_RANGE_END = time(9, 30), time(10, 0)
# The instructor's chart convention: the RTH gap runs from the prior session's
# final one-minute close, timed at 16:14, to the next 09:30 open. These are
# charting conventions rather than exchange settlement definitions.
RTH_LAST_MINUTE = time(16, 14)
# The new-week gap runs from Friday's final one-minute close at 16:59 to the
# Sunday 18:00 reopen.
WEEK_LAST_MINUTE = time(16, 59)
WEEK_REOPEN = time(18, 0)


@dataclass(frozen=True)
class NamedRange:
    """A frozen measurement: endpoints, when they became known, and a midpoint.

    `complete` records whether the window had actually closed at the moment of
    the request. A premarket extreme set at 08:57 was not available at 08:40,
    and a finished chart hides that.
    """
    name: str
    high: Optional[float]
    low: Optional[float]
    start_ts: int
    end_ts: int
    complete: bool

    @property
    def midpoint(self) -> Optional[float]:
        if self.high is None or self.low is None:
            return None
        return (self.high + self.low) / 2.0

    @property
    def width(self) -> float:
        if self.high is None or self.low is None:
            return 0.0
        return max(self.high - self.low, 0.0)

    def level(self, q: float) -> Optional[float]:
        """Grade the range: level at fraction q = low + q x (high - low).

        q may lie outside [0, 1] for a projection beyond an endpoint. This is
        exact arithmetic on chosen anchors, not a statistical estimate.
        """
        if self.low is None or self.high is None:
            return None
        return self.low + float(q) * (self.high - self.low)

    def position(self, price: float) -> Optional[float]:
        if not self.width:
            return None
        return (float(price) - self.low) / self.width

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "high": round(self.high, 2) if self.high is not None else None,
            "low": round(self.low, 2) if self.low is not None else None,
            "midpoint": round(self.midpoint, 2) if self.midpoint is not None else None,
            "width": round(self.width, 2),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "complete": self.complete,
        }


def _window_bounds(ts: int, start: time, end: time) -> Tuple[int, int]:
    day = to_ny(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    begin = day.replace(hour=start.hour, minute=start.minute)
    finish = day.replace(hour=end.hour, minute=end.minute)
    if end <= start:
        finish = finish + timedelta(days=1)
    return (int(begin.astimezone(timezone.utc).timestamp()),
            int(finish.astimezone(timezone.utc).timestamp()))


def named_range(df: pd.DataFrame, ts: int, name: str,
                start: time, end: time) -> NamedRange:
    """Freeze a window's extremes using only bars that closed by `ts`."""
    begin, finish = _window_bounds(ts, start, end)
    if df is None or df.empty:
        return NamedRange(name, None, None, begin, finish, complete=ts >= finish)
    window = df[(df["t"] >= begin) & (df["t"] < finish) & (df["t"] <= ts)]
    if window.empty:
        return NamedRange(name, None, None, begin, finish, complete=ts >= finish)
    return NamedRange(name, float(window["h"].max()), float(window["l"].min()),
                      begin, finish, complete=ts >= finish)


def premarket_range(df: pd.DataFrame, ts: int) -> NamedRange:
    """07:00-09:00 New York. Graded only after the window closes."""
    return named_range(df, ts, "premarket", PREMARKET_START, PREMARKET_END)


def opening_range(df: pd.DataFrame, ts: int) -> NamedRange:
    """09:30-10:00 New York, a different measurement from the opening gap."""
    return named_range(df, ts, "opening_range", OPENING_RANGE_START, OPENING_RANGE_END)


def opening_gap(df: pd.DataFrame, ts: int) -> Dict[str, Any]:
    """The RTH opening gap: prior session's 16:14 close to today's 09:30 open.

    Returned as an explicit pair of endpoints with the midpoint, because the
    book insists a gap keeps its identity and its dates rather than becoming
    an anonymous shaded rectangle.
    """
    out: Dict[str, Any] = {"name": "opening_gap", "prior_close": None,
                           "open": None, "midpoint": None, "width": 0.0,
                           "direction": None, "complete": False}
    if df is None or df.empty:
        return out

    day = to_ny(ts).replace(hour=0, minute=0, second=0, microsecond=0)
    today_open_ts = int(day.replace(hour=9, minute=30).astimezone(timezone.utc).timestamp())

    # Walk back to the most recent prior session's final minute.
    prior_day = day - timedelta(days=1)
    for _ in range(5):
        if prior_day.weekday() < 5:
            break
        prior_day = prior_day - timedelta(days=1)
    prior_close_ts = int(
        prior_day.replace(hour=RTH_LAST_MINUTE.hour, minute=RTH_LAST_MINUTE.minute)
        .astimezone(timezone.utc).timestamp()
    )

    prior = df[(df["t"] <= min(prior_close_ts, ts))]
    if prior.empty:
        return out
    out["prior_close"] = float(prior["c"].iloc[-1])

    today = df[(df["t"] >= today_open_ts) & (df["t"] <= ts)]
    if today.empty:
        return out
    out["open"] = float(today["o"].iloc[0])
    out["complete"] = True
    out["width"] = round(abs(out["open"] - out["prior_close"]), 2)
    out["midpoint"] = round((out["open"] + out["prior_close"]) / 2.0, 2)
    out["direction"] = "gap_up" if out["open"] > out["prior_close"] else "gap_down"
    out["prior_close"] = round(out["prior_close"], 2)
    out["open"] = round(out["open"], 2)
    return out


def new_week_opening_gap(df: pd.DataFrame, ts: int) -> Dict[str, Any]:
    """Friday's 16:59 close to the Sunday 18:00 reopen.

    A standing reference that stays relevant for weeks, which is why the
    source keeps at least five of them on the chart.
    """
    out: Dict[str, Any] = {"name": "new_week_opening_gap", "friday_close": None,
                           "sunday_open": None, "midpoint": None, "width": 0.0,
                           "complete": False}
    if df is None or df.empty:
        return out

    local = to_ny(ts)
    # Most recent Friday at or before `ts`.
    friday = local.replace(hour=WEEK_LAST_MINUTE.hour, minute=WEEK_LAST_MINUTE.minute,
                           second=0, microsecond=0)
    while friday.weekday() != 4 or friday > local:
        friday = friday - timedelta(days=1)
    friday_ts = int(friday.astimezone(timezone.utc).timestamp())
    sunday_ts = int((friday.replace(hour=WEEK_REOPEN.hour, minute=0) + timedelta(days=2))
                    .astimezone(timezone.utc).timestamp())

    before = df[df["t"] <= min(friday_ts, ts)]
    if before.empty:
        return out
    out["friday_close"] = round(float(before["c"].iloc[-1]), 2)

    after = df[(df["t"] >= sunday_ts) & (df["t"] <= ts)]
    if after.empty:
        return out
    out["sunday_open"] = round(float(after["o"].iloc[0]), 2)
    out["complete"] = True
    out["width"] = round(abs(out["sunday_open"] - out["friday_close"]), 2)
    out["midpoint"] = round((out["sunday_open"] + out["friday_close"]) / 2.0, 2)
    return out
