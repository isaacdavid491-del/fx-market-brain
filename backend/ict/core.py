"""Core ICT market primitives.

Every function here is pure: it takes an OHLCV frame (columns t,o,h,l,c,v with
`t` in epoch seconds, ascending) and returns plain dataclasses. Keeping the
primitives side-effect free means the live agents and the backtester run the
identical code path, so a backtest result describes the live system.

Vocabulary, briefly, since the naming follows ICT rather than classical TA:
  swing            a fractal pivot high or low
  BOS              break of structure, trend continuation
  CHoCH            change of character, the first break against the trend
  FVG              fair value gap, a 3-bar imbalance left by fast price
  order block      last opposing candle before a displacement leg
  liquidity pool   a cluster of equal highs/lows where stops accumulate
  sweep            a raid of that pool followed by rejection
  dealing range    the swing low to swing high the market is trading inside
  OTE              optimal trade entry, the 0.62-0.79 retracement band
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BULLISH = "bullish"
BEARISH = "bearish"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Swing:
    idx: int
    t: int
    price: float
    kind: str  # "high" | "low"

    def as_dict(self) -> Dict[str, Any]:
        return {"idx": self.idx, "t": self.t, "price": round(self.price, 5), "kind": self.kind}


@dataclass
class StructureEvent:
    idx: int
    t: int
    kind: str        # "BOS" | "CHoCH"
    direction: str   # bullish | bearish
    level: float     # the swing level that was broken
    swing_idx: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx, "t": self.t, "kind": self.kind,
            "direction": self.direction, "level": round(self.level, 5),
        }


@dataclass
class FVG:
    idx: int          # index of the third candle of the pattern
    t: int
    direction: str
    top: float
    bottom: float
    size: float
    touched: bool = False
    filled: bool = False
    filled_idx: Optional[int] = None

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t, "direction": self.direction,
            "top": round(self.top, 5), "bottom": round(self.bottom, 5),
            "midpoint": round(self.midpoint, 5), "size": round(self.size, 5),
            "touched": self.touched, "filled": self.filled,
        }


@dataclass
class OrderBlock:
    idx: int
    t: int
    direction: str   # bullish OB = demand, bearish OB = supply
    top: float
    bottom: float
    displacement_idx: int
    has_fvg: bool = False
    mitigated: bool = False
    mitigated_idx: Optional[int] = None
    broken: bool = False   # price closed through it: a breaker candidate

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t, "direction": self.direction,
            "top": round(self.top, 5), "bottom": round(self.bottom, 5),
            "has_fvg": self.has_fvg, "mitigated": self.mitigated, "broken": self.broken,
        }


@dataclass
class LiquidityPool:
    price: float
    side: str        # "buyside" (above price, resting buy stops) | "sellside"
    touches: int
    first_t: int
    last_t: int
    swept: bool = False
    swept_t: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "price": round(self.price, 5), "side": self.side, "touches": self.touches,
            "last_t": self.last_t, "swept": self.swept,
        }


@dataclass
class Sweep:
    idx: int
    t: int
    side: str        # buyside sweep = raid of highs (bearish), sellside = bullish
    level: float
    extreme: float   # how far past the level price ran
    reclaimed: bool  # closed back inside the range

    @property
    def implied_direction(self) -> str:
        return BEARISH if self.side == "buyside" else BULLISH

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t, "side": self.side, "level": round(self.level, 5),
            "extreme": round(self.extreme, 5), "reclaimed": self.reclaimed,
            "implied_direction": self.implied_direction,
        }


@dataclass
class DealingRange:
    high: float
    low: float
    high_t: int
    low_t: int
    direction: str   # the leg direction: bullish if low came first

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def span(self) -> float:
        return max(self.high - self.low, 0.0)

    def position(self, price: float) -> float:
        """Where price sits in the range: 0.0 at the low, 1.0 at the high."""
        if self.span <= 0:
            return 0.5
        return float(np.clip((price - self.low) / self.span, -1.0, 2.0))

    def zone(self, price: float) -> str:
        pos = self.position(price)
        if pos > 0.5:
            return "premium"
        if pos < 0.5:
            return "discount"
        return "equilibrium"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "high": round(self.high, 5), "low": round(self.low, 5),
            "equilibrium": round(self.equilibrium, 5), "direction": self.direction,
        }


@dataclass
class ATR:
    value: float
    series: pd.Series = field(repr=False, default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(df: pd.DataFrame) -> None:
    missing = {"t", "o", "h", "l", "c"} - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")


def atr(df: pd.DataFrame, period: int = 14) -> ATR:
    """Average true range. Falls back to mean bar range on short frames."""
    _require(df)
    if df.empty:
        return ATR(0.0, pd.Series(dtype=float))
    high, low, close = df["h"], df["l"], df["c"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    series = tr.rolling(period, min_periods=1).mean()
    value = float(series.iloc[-1]) if len(series) else 0.0
    if not np.isfinite(value) or value <= 0:
        value = float((high - low).mean() or 0.0)
    return ATR(value, series)


def find_swings(df: pd.DataFrame, strength: int = 2) -> List[Swing]:
    """Fractal pivots: a high with `strength` lower highs on each side, and the
    mirror for lows. `strength` bars at the right edge are unconfirmed, which is
    handled by the consumers rather than hidden here."""
    _require(df)
    n = len(df)
    out: List[Swing] = []
    if n < 2 * strength + 1:
        return out
    high = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    for i in range(strength, n - strength):
        left_h, right_h = high[i - strength:i], high[i + 1:i + strength + 1]
        if high[i] > left_h.max() and high[i] >= right_h.max():
            out.append(Swing(i, int(ts[i]), float(high[i]), "high"))
        left_l, right_l = low[i - strength:i], low[i + 1:i + strength + 1]
        if low[i] < left_l.min() and low[i] <= right_l.min():
            out.append(Swing(i, int(ts[i]), float(low[i]), "low"))
    out.sort(key=lambda s: s.idx)
    return out


def market_structure(
    df: pd.DataFrame, strength: int = 2, swings: Optional[List[Swing]] = None
) -> Tuple[List[StructureEvent], Dict[str, Any]]:
    """Walk the bars and record every break of a confirmed swing.

    A break in the direction of the prevailing trend is a BOS; the first break
    against it is a CHoCH. Breaks are judged on closes, so a wick through a
    level is a liquidity raid rather than a structural break, which is the
    distinction the whole methodology rests on.
    """
    _require(df)
    swings = swings if swings is not None else find_swings(df, strength)
    events: List[StructureEvent] = []
    state = {
        "trend": "neutral",
        "last_event": None,
        "protected_high": None,
        "protected_low": None,
    }
    if df.empty:
        return events, state

    close = df["c"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    pending_highs: List[Swing] = []
    pending_lows: List[Swing] = []
    ptr = 0
    trend = "neutral"

    for i in range(len(df)):
        # A swing is only tradable once its right-hand bars have printed.
        while ptr < len(swings) and swings[ptr].idx + strength <= i:
            s = swings[ptr]
            (pending_highs if s.kind == "high" else pending_lows).append(s)
            ptr += 1

        price = close[i]

        if pending_highs and price > pending_highs[-1].price:
            broken = pending_highs[-1]
            while pending_highs and price > pending_highs[-1].price:
                broken = pending_highs.pop()
            kind = "CHoCH" if trend == BEARISH else "BOS"
            trend = BULLISH
            events.append(StructureEvent(i, int(ts[i]), kind, BULLISH, broken.price, broken.idx))
            pending_lows = [s for s in pending_lows if s.idx > broken.idx] or pending_lows[-1:]

        elif pending_lows and price < pending_lows[-1].price:
            broken = pending_lows[-1]
            while pending_lows and price < pending_lows[-1].price:
                broken = pending_lows.pop()
            kind = "CHoCH" if trend == BULLISH else "BOS"
            trend = BEARISH
            events.append(StructureEvent(i, int(ts[i]), kind, BEARISH, broken.price, broken.idx))
            pending_highs = [s for s in pending_highs if s.idx > broken.idx] or pending_highs[-1:]

    state["trend"] = trend
    state["last_event"] = events[-1].as_dict() if events else None
    state["protected_high"] = pending_highs[-1].as_dict() if pending_highs else None
    state["protected_low"] = pending_lows[-1].as_dict() if pending_lows else None
    return events, state


def find_fvgs(df: pd.DataFrame, min_size: float = 0.0) -> List[FVG]:
    """Three-bar imbalances, with mitigation tracked forward through the frame.

    Bullish gap: bar i's low sits above bar i-2's high, so the middle bar
    traded through a band nobody offered into. Bearish is the mirror.
    """
    _require(df)
    n = len(df)
    out: List[FVG] = []
    if n < 3:
        return out
    high = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    for i in range(2, n):
        if low[i] > high[i - 2]:
            size = low[i] - high[i - 2]
            if size > min_size:
                out.append(FVG(i, int(ts[i]), BULLISH, top=float(low[i]),
                               bottom=float(high[i - 2]), size=float(size)))
        elif high[i] < low[i - 2]:
            size = low[i - 2] - high[i]
            if size > min_size:
                out.append(FVG(i, int(ts[i]), BEARISH, top=float(low[i - 2]),
                               bottom=float(high[i]), size=float(size)))

    for gap in out:
        for j in range(gap.idx + 1, n):
            if low[j] <= gap.top and high[j] >= gap.bottom:
                gap.touched = True
            if gap.direction == BULLISH and low[j] <= gap.bottom:
                gap.filled, gap.filled_idx = True, j
                break
            if gap.direction == BEARISH and high[j] >= gap.top:
                gap.filled, gap.filled_idx = True, j
                break
    return out


def displacement_bars(df: pd.DataFrame, atr_value: float, multiple: float = 1.5,
                      body_ratio: float = 0.5) -> List[int]:
    """Indices of energetic, one-sided bars: the footprint of institutional
    repricing rather than drift."""
    _require(df)
    if df.empty or atr_value <= 0:
        return []
    rng = (df["h"] - df["l"]).to_numpy(dtype=float)
    body = (df["c"] - df["o"]).abs().to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rng > 0, body / rng, 0.0)
    mask = (rng >= multiple * atr_value) & (ratio >= body_ratio)
    return [int(i) for i in np.flatnonzero(mask)]


def find_order_blocks(
    df: pd.DataFrame,
    events: Optional[List[StructureEvent]] = None,
    strength: int = 2,
    max_lookback: int = 12,
) -> List[OrderBlock]:
    """The last opposing candle before the leg that broke structure.

    An order block only earns the name if the leg that followed it actually
    displaced and broke a level, so structure events are the anchor rather
    than raw candle colour.
    """
    _require(df)
    n = len(df)
    if n < 3:
        return []
    if events is None:
        events, _ = market_structure(df, strength)

    o = df["o"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    c = df["c"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    blocks: List[OrderBlock] = []
    seen: set[Tuple[int, str]] = set()

    for ev in events:
        want_down = ev.direction == BULLISH  # bullish break => find last down candle
        start = ev.idx
        found = None
        for j in range(start, max(start - max_lookback, -1), -1):
            is_down = c[j] < o[j]
            if is_down == want_down:
                found = j
                break
        if found is None or (found, ev.direction) in seen:
            continue
        seen.add((found, ev.direction))

        has_fvg = False
        if found + 2 < n:
            # bool() because comparing numpy scalars yields numpy.bool_, which
            # is not JSON-serialisable once it reaches the API.
            has_fvg = bool(low[found + 2] > h[found]) if ev.direction == BULLISH \
                else bool(h[found + 2] < low[found])

        blocks.append(
            OrderBlock(
                idx=found, t=int(ts[found]), direction=ev.direction,
                top=float(h[found]), bottom=float(low[found]),
                displacement_idx=ev.idx, has_fvg=has_fvg,
            )
        )

    for ob in blocks:
        for j in range(ob.displacement_idx + 1, n):
            if low[j] <= ob.top and h[j] >= ob.bottom:
                ob.mitigated = True
                if ob.mitigated_idx is None:
                    ob.mitigated_idx = j
            if ob.direction == BULLISH and c[j] < ob.bottom:
                ob.broken = True
                break
            if ob.direction == BEARISH and c[j] > ob.top:
                ob.broken = True
                break
    return blocks


def liquidity_pools(
    df: pd.DataFrame,
    swings: Optional[List[Swing]] = None,
    tolerance: float = 0.0,
    strength: int = 2,
) -> List[LiquidityPool]:
    """Cluster near-equal swing highs and lows: where stop orders pile up.

    `tolerance` is an absolute price distance; callers normally pass a fraction
    of ATR so the clustering scales with volatility.
    """
    _require(df)
    swings = swings if swings is not None else find_swings(df, strength)
    if not swings:
        return []
    if tolerance <= 0:
        tolerance = atr(df).value * 0.15

    high_val = df["h"].to_numpy(dtype=float)
    low_val = df["l"].to_numpy(dtype=float)
    n = len(df)
    pools: List[LiquidityPool] = []

    for kind, side in (("high", "buyside"), ("low", "sellside")):
        group = [s for s in swings if s.kind == kind]
        if not group:
            continue
        group.sort(key=lambda s: s.price)
        cluster: List[Swing] = [group[0]]
        for s in group[1:]:
            if abs(s.price - cluster[-1].price) <= tolerance:
                cluster.append(s)
            else:
                pools.append(_pool_from_cluster(cluster, side))
                cluster = [s]
        pools.append(_pool_from_cluster(cluster, side))

    # A pool is spent once price has traded decisively beyond it.
    for pool in pools:
        last_idx = max((s for s in swings if s.t == pool.last_t), key=lambda s: s.idx, default=None)
        start = (last_idx.idx + strength + 1) if last_idx else 0
        for j in range(start, n):
            if pool.side == "buyside" and high_val[j] > pool.price + tolerance:
                pool.swept, pool.swept_t = True, int(df["t"].iloc[j])
                break
            if pool.side == "sellside" and low_val[j] < pool.price - tolerance:
                pool.swept, pool.swept_t = True, int(df["t"].iloc[j])
                break
    pools.sort(key=lambda p: (-p.touches, -p.last_t))
    return pools


def _pool_from_cluster(cluster: List[Swing], side: str) -> LiquidityPool:
    prices = [s.price for s in cluster]
    times = [s.t for s in cluster]
    return LiquidityPool(
        price=float(np.mean(prices)), side=side, touches=len(cluster),
        first_t=int(min(times)), last_t=int(max(times)),
    )


def recent_sweeps(
    df: pd.DataFrame,
    swings: Optional[List[Swing]] = None,
    lookback: int = 30,
    strength: int = 2,
    buffer: float = 0.0,
) -> List[Sweep]:
    """Stop raids in the recent window: price runs a prior swing then rejects.

    The reclaim test is what separates a sweep from a genuine breakout, so it
    is reported rather than assumed.
    """
    _require(df)
    n = len(df)
    if n < 3:
        return []
    swings = swings if swings is not None else find_swings(df, strength)
    if not swings:
        return []
    if buffer <= 0:
        buffer = atr(df).value * 0.05

    high = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    close = df["c"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    out: List[Sweep] = []
    start = max(n - lookback, 1)
    for i in range(start, n):
        prior_highs = [s for s in swings if s.kind == "high" and s.idx + strength <= i and s.idx < i]
        prior_lows = [s for s in swings if s.kind == "low" and s.idx + strength <= i and s.idx < i]
        if prior_highs:
            ref = prior_highs[-1]
            if high[i] > ref.price + buffer:
                out.append(Sweep(i, int(ts[i]), "buyside", ref.price, float(high[i]),
                                 reclaimed=bool(close[i] < ref.price)))
        if prior_lows:
            ref = prior_lows[-1]
            if low[i] < ref.price - buffer:
                out.append(Sweep(i, int(ts[i]), "sellside", ref.price, float(low[i]),
                                 reclaimed=bool(close[i] > ref.price)))
    return out


def dealing_range(df: pd.DataFrame, lookback: int = 60) -> Optional[DealingRange]:
    """The swing high and low price is currently working between."""
    _require(df)
    if df.empty:
        return None
    window = df.tail(lookback)
    hi_pos = int(window["h"].to_numpy(dtype=float).argmax())
    lo_pos = int(window["l"].to_numpy(dtype=float).argmin())
    hi_t = int(window["t"].iloc[hi_pos])
    lo_t = int(window["t"].iloc[lo_pos])
    return DealingRange(
        high=float(window["h"].iloc[hi_pos]), low=float(window["l"].iloc[lo_pos]),
        high_t=hi_t, low_t=lo_t,
        direction=BULLISH if lo_pos < hi_pos else BEARISH,
    )


def ote_zone(rng: DealingRange, direction: str) -> Tuple[float, float]:
    """Optimal trade entry: the 0.62-0.79 retracement of the dealing range.

    Longs want the discount retracement of an up leg; shorts want the premium
    retracement of a down leg.
    """
    span = rng.span
    if span <= 0:
        return (rng.low, rng.high)
    if direction == BULLISH:
        return (rng.high - 0.79 * span, rng.high - 0.62 * span)
    return (rng.low + 0.62 * span, rng.low + 0.79 * span)


# ---------------------------------------------------------------------------
# Measurements the study book specifies precisely
# ---------------------------------------------------------------------------
# Chapter 9 is emphatic that naming the wrong object moves the reference: for
# O=108, C=104, H=116, L=102 the upper-wick midpoint is 112 while the
# whole-candle midpoint is 109. Three units of difference is the whole trade.

def upper_wick_midpoint(o: float, h: float, c: float) -> float:
    """Midpoint of the upper wick: [max(O, C) + H] / 2."""
    return (max(float(o), float(c)) + float(h)) / 2.0


def lower_wick_midpoint(o: float, l: float, c: float) -> float:
    """Midpoint of the lower wick: [L + min(O, C)] / 2."""
    return (float(l) + min(float(o), float(c))) / 2.0


def candle_midpoint(h: float, l: float) -> float:
    """Whole-candle midpoint: (H + L) / 2. Not the same object as a wick midpoint."""
    return (float(h) + float(l)) / 2.0


def consequent_encroachment(top: float, bottom: float) -> float:
    """Midpoint of an identified gap, wick or inefficiency."""
    return (float(top) + float(bottom)) / 2.0


@dataclass
class WickPair:
    """Two opposing wick midpoints and the interval between them.

    The Obsidian review measures an upper wick on one candle and a lower wick
    on another, then studies the interval between their midpoints. The book is
    careful that any two opposing wicks somewhere on a chart are *not*
    sufficient to establish the method, so this reports the measurement and
    leaves qualification to the caller.
    """
    upper_idx: int
    lower_idx: int
    upper_t: int
    lower_t: int
    upper_mid: float
    lower_mid: float

    @property
    def interval(self) -> Tuple[float, float]:
        return (min(self.upper_mid, self.lower_mid), max(self.upper_mid, self.lower_mid))

    @property
    def width(self) -> float:
        low, high = self.interval
        return high - low

    def contains(self, price: float) -> bool:
        low, high = self.interval
        return low <= price <= high

    def as_dict(self) -> Dict[str, Any]:
        low, high = self.interval
        return {
            "upper_t": self.upper_t, "lower_t": self.lower_t,
            "upper_mid": round(self.upper_mid, 2), "lower_mid": round(self.lower_mid, 2),
            "interval_low": round(low, 2), "interval_high": round(high, 2),
            "width": round(self.width, 2),
        }


def opposing_wick_pairs(df: pd.DataFrame, lookback: int = 20,
                        min_wick_atr: float = 0.15,
                        atr_value: Optional[float] = None,
                        max_separation: int = 6) -> List[WickPair]:
    """Recent candle pairs with opposing wicks, best first.

    A pair is one candle carrying a qualifying upper wick and another, within
    `max_separation` bars, carrying a qualifying lower wick. Pairs are ranked
    by how recent they are and how pronounced both wicks are.

    Only the geometry is established here. The book's verification table lists
    the qualifying event, the surrounding narrative and the protection rules
    as unresolved, so nothing here claims a pair is tradable.
    """
    _require(df)
    n = len(df)
    if n < 2:
        return []
    if atr_value is None:
        atr_value = atr(df).value
    threshold = max(atr_value * min_wick_atr, 0.0)

    o = df["o"].to_numpy(dtype=float)
    h = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    c = df["c"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    start = max(n - lookback, 0)
    upper_len = {i: h[i] - max(o[i], c[i]) for i in range(start, n)}
    lower_len = {i: min(o[i], c[i]) - low[i] for i in range(start, n)}
    uppers = [i for i in range(start, n) if upper_len[i] >= threshold]
    lowers = [i for i in range(start, n) if lower_len[i] >= threshold]

    scored: List[Tuple[float, WickPair]] = []
    for ui in uppers:
        for li in lowers:
            if ui == li or abs(ui - li) > max_separation:
                continue
            recency = 1.0 - (n - 1 - max(ui, li)) / float(max(lookback, 1))
            size = upper_len[ui] + lower_len[li]
            scored.append((
                recency * 2.0 + (size / atr_value if atr_value > 0 else 0.0),
                WickPair(
                    upper_idx=ui, lower_idx=li,
                    upper_t=int(ts[ui]), lower_t=int(ts[li]),
                    upper_mid=upper_wick_midpoint(o[ui], h[ui], c[ui]),
                    lower_mid=lower_wick_midpoint(o[li], low[li], c[li]),
                ),
            ))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [pair for _, pair in scored[:8]]


@dataclass
class Inversion:
    """A gap that failed on a *close* and is now used from the other side.

    Chapter 10 draws the line explicitly: a wick through a bullish gap is not
    a bearish inversion, a close below is required. It also notes this is not
    one universal rule across every lesson, so the trigger is recorded rather
    than assumed.
    """
    gap: "FVG"
    inverted_idx: int
    inverted_t: int
    new_direction: str
    trigger: str          # "close_beyond"
    close_price: float
    retested: bool = False
    retest_idx: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original_direction": self.gap.direction,
            "new_direction": self.new_direction,
            "top": round(self.gap.top, 2), "bottom": round(self.gap.bottom, 2),
            "midpoint": round(self.gap.midpoint, 2),
            "formed_t": self.gap.t, "inverted_t": self.inverted_t,
            "trigger": self.trigger, "close_price": round(self.close_price, 2),
            "retested": self.retested,
        }


def find_inversions(df: pd.DataFrame, gaps: Optional[List[FVG]] = None) -> List[Inversion]:
    """Gaps whose failure was confirmed by a close beyond the far boundary.

    A wick through the boundary is deliberately not enough. The two events are
    separated because a stop can be taken out by the wick long before the
    close qualifies the label, and classification never retroactively protects
    a position.
    """
    _require(df)
    n = len(df)
    if n < 4:
        return []
    gaps = gaps if gaps is not None else find_fvgs(df)

    close = df["c"].to_numpy(dtype=float)
    high = df["h"].to_numpy(dtype=float)
    low = df["l"].to_numpy(dtype=float)
    ts = df["t"].to_numpy(dtype="int64")

    out: List[Inversion] = []
    for gap in gaps:
        for j in range(gap.idx + 1, n):
            failed = (gap.direction == BULLISH and close[j] < gap.bottom) or \
                     (gap.direction == BEARISH and close[j] > gap.top)
            if not failed:
                continue
            inv = Inversion(
                gap=gap, inverted_idx=j, inverted_t=int(ts[j]),
                new_direction=BEARISH if gap.direction == BULLISH else BULLISH,
                trigger="close_beyond", close_price=float(close[j]),
            )
            # An inverted gap is normally traded on its retest, so record it.
            for k in range(j + 1, n):
                if low[k] <= gap.top and high[k] >= gap.bottom:
                    inv.retested, inv.retest_idx = True, k
                    break
            out.append(inv)
            break
    return out


@dataclass
class PresentedGap:
    """A gap carrying its chronological identity.

    Chapter 11 keeps three identities apart: the first gap after the open, the
    first gap that displaced beyond a relevant swing, and the first subsequent
    gap opposite in direction to the chronological first, which the source
    calls the first-presented reflection. A later displacing gap can take
    emphasis without changing which gap was chronologically first.
    """
    gap: FVG
    identity: str          # "chronological" | "displacement" | "reflection"
    knowable_t: int        # when the completed three-candle pattern became visible

    def as_dict(self) -> Dict[str, Any]:
        out = self.gap.as_dict()
        out["identity"] = self.identity
        out["knowable_t"] = self.knowable_t
        return out


def first_presented_gaps(df: pd.DataFrame, session_start_ts: int,
                         swings: Optional[List[Swing]] = None,
                         strength: int = 2) -> Dict[str, Optional[PresentedGap]]:
    """Classify the session's first gaps by the three distinct identities.

    The middle candle's label time is not the moment the pattern is knowable:
    the third candle has to close first, so `knowable_t` is recorded
    separately from the gap's own timestamp.
    """
    _require(df)
    out: Dict[str, Optional[PresentedGap]] = {
        "chronological": None, "displacement": None, "reflection": None,
    }
    if df.empty:
        return out

    gaps = [g for g in find_fvgs(df) if g.t >= session_start_ts]
    if not gaps:
        return out
    swings = swings if swings is not None else find_swings(df, strength)

    first = gaps[0]
    out["chronological"] = PresentedGap(first, "chronological", first.t)

    # First gap whose leg displaced beyond a swing formed before it.
    close = df["c"].to_numpy(dtype=float)
    for gap in gaps:
        prior_highs = [s for s in swings if s.kind == "high" and s.idx + strength <= gap.idx]
        prior_lows = [s for s in swings if s.kind == "low" and s.idx + strength <= gap.idx]
        if gap.direction == BULLISH and prior_highs and close[gap.idx] > prior_highs[-1].price:
            out["displacement"] = PresentedGap(gap, "displacement", gap.t)
            break
        if gap.direction == BEARISH and prior_lows and close[gap.idx] < prior_lows[-1].price:
            out["displacement"] = PresentedGap(gap, "displacement", gap.t)
            break

    # First later gap opposite in direction to the chronological first.
    for gap in gaps[1:]:
        if gap.direction != first.direction:
            out["reflection"] = PresentedGap(gap, "reflection", gap.t)
            break
    return out


def grade_range(low: float, high: float, divisions: int = 8) -> Dict[str, float]:
    """Subdivide a range into equal parts: quarters, octants and so on.

    Chapter 5's arithmetic exactly: for 100 to 180 the width is 80, the
    midpoint 140, quarters 120 and 160, eighths every 10 units. The
    subdivision is exact; whether the chosen range is *useful* is a separate
    judgement the book insists on keeping separate.
    """
    low, high = float(low), float(high)
    width = high - low
    out = {"low": low, "high": high, "width": width, "midpoint": low + 0.5 * width}
    for i in range(divisions + 1):
        q = i / divisions
        out[f"q{i}_{divisions}"] = low + q * width
    return out


def project_range(low: float, high: float, q: float) -> float:
    """Project a level outside the range: low + q x (high - low).

    A half-range projection above the high of a 100-180 range is
    180 + 0.5 x 80 = 220. This is a range extension, not a standard deviation.
    """
    return float(low) + float(q) * (float(high) - float(low))
