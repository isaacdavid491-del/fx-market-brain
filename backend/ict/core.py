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
