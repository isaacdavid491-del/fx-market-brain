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
