"""Liquidity agents: where price is drawn to, and where stops were just taken."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.agents.base import LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext, direction_from_score
from backend.ict import core
from backend.ict.sessions import previous_day_levels, session_weight


class LiquidityDrawAgent(BaseAgent):
    """Finds the nearest unswept pools on each side and picks the likely draw.

    Price moves from one pool of resting orders to the next. When untouched
    equal highs sit far closer than the nearest equal lows, the asymmetry is
    the trade: the objective above is cheaper for the market to reach.
    """

    name = "liquidity_draw"
    role = "analyst"
    description = "Nearest unswept buyside/sellside liquidity as the directional draw."
    default_weight = 1.5

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.mtf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        price = ctx.price
        atr_val = ctx.atr(tf) or 1e-9
        pools = [p for p in ctx.pools(tf) if not p.swept]

        above = [p for p in pools if p.price > price and p.side == "buyside"]
        below = [p for p in pools if p.price < price and p.side == "sellside"]

        # Previous-day extremes are standing liquidity in their own right.
        pdl_levels = previous_day_levels(df, ctx.now_ts)
        extra: List[Dict[str, Any]] = []
        if pdl_levels.get("pdh") and pdl_levels["pdh"] > price:
            extra.append({"price": pdl_levels["pdh"], "side": "buyside", "label": "PDH"})
        if pdl_levels.get("pdl") and pdl_levels["pdl"] < price:
            extra.append({"price": pdl_levels["pdl"], "side": "sellside", "label": "PDL"})

        nearest_up = min([p.price for p in above] + [e["price"] for e in extra if e["side"] == "buyside"],
                         default=None)
        nearest_down = max([p.price for p in below] + [e["price"] for e in extra if e["side"] == "sellside"],
                           default=None)

        if nearest_up is None and nearest_down is None:
            return self.neutral("no unswept liquidity identified", timeframe=tf)

        dist_up = (nearest_up - price) / atr_val if nearest_up is not None else None
        dist_down = (price - nearest_down) / atr_val if nearest_down is not None else None

        if dist_up is not None and dist_down is not None:
            total = dist_up + dist_down
            # Closer side wins; score is the normalised asymmetry.
            score = ((dist_down - dist_up) / total) if total > 0 else 0.0
            confidence = min(0.75, 0.3 + 0.1 * abs(score) * 5)
        elif dist_up is not None:
            score, confidence = 0.55, 0.4
        else:
            score, confidence = -0.55, 0.4

        # Thick pools (many equal touches) are stronger magnets.
        touch_up = max([p.touches for p in above], default=1)
        touch_down = max([p.touches for p in below], default=1)
        if touch_up > touch_down:
            score += 0.1
        elif touch_down > touch_up:
            score -= 0.1

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"Nearest unswept buyside {('%.2f' % nearest_up) if nearest_up else 'n/a'} "
                f"({dist_up:.1f} ATR)" if dist_up is not None else "no buyside pool"
            ) + (
                f" vs sellside {('%.2f' % nearest_down) if nearest_down else 'n/a'} "
                f"({dist_down:.1f} ATR)." if dist_down is not None else " vs no sellside pool."
            ),
            evidence={
                "timeframe": tf,
                "nearest_buyside": nearest_up,
                "nearest_sellside": nearest_down,
                "atr_to_buyside": round(dist_up, 2) if dist_up is not None else None,
                "atr_to_sellside": round(dist_down, 2) if dist_down is not None else None,
                "pools_above": [p.as_dict() for p in above[:3]],
                "pools_below": [p.as_dict() for p in below[:3]],
                "prev_day": {k: (round(v, 2) if v is not None else None) for k, v in pdl_levels.items()},
            },
            levels={k: v for k, v in {"draw_up": nearest_up, "draw_down": nearest_down}.items() if v},
        )


class SweepAgent(BaseAgent):
    """Turtle soup: a raid of a prior swing that price immediately rejects.

    This is the farm's highest-conviction entry trigger. A stop run that closes
    back inside the range means the move was engineered to fill orders, not to
    continue, so the trade is against the raid.
    """

    name = "liquidity_sweep"
    role = "analyst"
    description = "Stop raids and failed breakouts (turtle soup) on the entry timeframe."
    default_weight = 2.2

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        lookback = int(self.params.get("lookback", 24))
        sweeps = ctx.sweeps(tf, lookback=lookback)
        if not sweeps:
            return self.neutral(f"no sweeps in the last {lookback} {tf} bars", timeframe=tf)

        n = len(df)
        best: Optional[core.Sweep] = None
        best_score = 0.0
        decay = float(self.params.get("decay_bars", 10))

        for sw in sweeps:
            bars_since = n - 1 - sw.idx
            freshness = max(0.0, 1.0 - bars_since / decay)
            if freshness <= 0:
                continue
            # An unreclaimed raid may just be a breakout: score it far lower.
            quality = 1.0 if sw.reclaimed else 0.25
            magnitude = min(1.0, abs(sw.extreme - sw.level) / max(ctx.atr(tf), 1e-9))
            value = freshness * quality * (0.6 + 0.4 * magnitude)
            if value > best_score:
                best_score, best = value, sw

        if best is None:
            return self.neutral(f"sweeps on {tf} are stale", timeframe=tf)

        bars_since = n - 1 - best.idx
        direction_sign = 1.0 if best.implied_direction == core.BULLISH else -1.0
        score = direction_sign * min(1.0, 0.55 + best_score * 0.5)
        confidence = min(0.95, 0.35 + 0.6 * best_score)

        # A raid inside a killzone is the textbook case; outside, discount it.
        kz = session_weight(best.t)
        confidence *= (0.6 + 0.4 * kz)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"{best.side} liquidity raided at {best.level:.2f} "
                f"(ran to {best.extreme:.2f}, "
                f"{'reclaimed' if best.reclaimed else 'not reclaimed'}), "
                f"{bars_since} {tf} bars ago."
            ),
            evidence={
                "timeframe": tf, "sweep": best.as_dict(), "bars_since": bars_since,
                "killzone_weight_at_sweep": round(kz, 2),
                "sweeps_seen": [s.as_dict() for s in sweeps[-4:]],
            },
            levels={"sweep_extreme": best.extreme, "sweep_level": best.level},
        )
