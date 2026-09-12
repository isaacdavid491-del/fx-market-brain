"""Imbalance agents: fair value gaps and order blocks — the entry zones."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.agents.base import AgentSignal, BaseAgent, MarketContext, direction_from_score
from backend.ict import core


class FairValueGapAgent(BaseAgent):
    """Scores unfilled 3-bar imbalances near price.

    An unfilled gap is unfinished business: price tends to return to it. A gap
    price is currently trading inside is treated as an active entry zone, one
    below price as support, one above as a magnet.
    """

    name = "fair_value_gap"
    role = "analyst"
    description = "Unfilled fair value gaps acting as entry zones and draws."
    default_weight = 1.6

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        n = len(df)
        max_age = int(self.params.get("max_age_bars", 60))
        near = float(self.params.get("near_atr", 2.0))

        live = [g for g in ctx.fvgs(tf) if not g.filled and (n - 1 - g.idx) <= max_age]
        if not live:
            return self.neutral(f"no live fair value gaps on {tf}", timeframe=tf)

        inside: List[core.FVG] = [g for g in live if g.contains(price)]
        score, confidence = 0.0, 0.0
        chosen: Optional[core.FVG] = None
        why = ""

        if inside:
            # Prefer the freshest gap price is actually reacting to.
            chosen = max(inside, key=lambda g: g.idx)
            sign = 1.0 if chosen.direction == core.BULLISH else -1.0
            score = sign * 0.8
            confidence = 0.6 if not chosen.touched else 0.45
            why = (f"price is inside a {chosen.direction} FVG "
                   f"{chosen.bottom:.2f}-{chosen.top:.2f}")
        else:
            below = [g for g in live if g.top < price and g.direction == core.BULLISH]
            above = [g for g in live if g.bottom > price and g.direction == core.BEARISH]
            nearest_below = max(below, key=lambda g: g.top, default=None)
            nearest_above = min(above, key=lambda g: g.bottom, default=None)

            d_below = (price - nearest_below.top) / atr_val if nearest_below else None
            d_above = (nearest_above.bottom - price) / atr_val if nearest_above else None

            if d_below is not None and d_below <= near and (d_above is None or d_below < d_above):
                chosen = nearest_below
                score, confidence = 0.5, 0.4
                why = f"unfilled bullish FVG {d_below:.1f} ATR below price"
            elif d_above is not None and d_above <= near:
                chosen = nearest_above
                score, confidence = -0.5, 0.4
                why = f"unfilled bearish FVG {d_above:.1f} ATR above price"
            else:
                return self.neutral(f"live FVGs on {tf} are too far from price",
                                    timeframe=tf, live_gaps=len(live))

        bull = sum(1 for g in live if g.direction == core.BULLISH)
        bear = len(live) - bull
        # A lopsided book of open gaps corroborates the direction.
        if bull != bear:
            score += 0.15 * (1.0 if bull > bear else -1.0)

        levels = {}
        if chosen is not None:
            levels = {"fvg_top": chosen.top, "fvg_bottom": chosen.bottom,
                      "fvg_mid": chosen.midpoint}

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=f"{why}; {bull} bullish / {bear} bearish gaps open on {tf}.",
            evidence={
                "timeframe": tf, "selected": chosen.as_dict() if chosen else None,
                "bullish_open": bull, "bearish_open": bear,
                "recent": [g.as_dict() for g in live[-4:]],
            },
            levels=levels,
        )


class OrderBlockAgent(BaseAgent):
    """Scores unmitigated order blocks, and flags breakers.

    An order block is only interesting until it is traded back through. Once
    price closes beyond it, it inverts: the same zone becomes a breaker that
    should hold from the other side.
    """

    name = "order_block"
    role = "analyst"
    description = "Unmitigated order blocks and breaker blocks near price."
    default_weight = 1.6

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.mtf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        n = len(df)
        max_age = int(self.params.get("max_age_bars", 80))
        near = float(self.params.get("near_atr", 2.5))

        blocks = [b for b in ctx.order_blocks(tf) if (n - 1 - b.idx) <= max_age]
        if not blocks:
            return self.neutral(f"no order blocks on {tf}", timeframe=tf)

        fresh = [b for b in blocks if not b.broken]
        breakers = [b for b in blocks if b.broken]

        best: Optional[core.OrderBlock] = None
        best_score, best_conf, why = 0.0, 0.0, ""

        for ob in fresh:
            if ob.contains(price):
                distance = 0.0
            elif ob.direction == core.BULLISH and price > ob.top:
                distance = (price - ob.top) / atr_val
            elif ob.direction == core.BEARISH and price < ob.bottom:
                distance = (ob.bottom - price) / atr_val
            else:
                continue  # price is on the wrong side: the block is spent
            if distance > near:
                continue
            sign = 1.0 if ob.direction == core.BULLISH else -1.0
            proximity = 1.0 - min(distance / near, 1.0)
            quality = 1.0 if ob.has_fvg else 0.75      # displacement left a gap
            freshness = 1.0 if not ob.mitigated else 0.6
            value = proximity * quality * freshness
            if value > abs(best_score):
                best_score = sign * (0.45 + 0.45 * value)
                best_conf = 0.3 + 0.5 * value
                best = ob
                why = (f"{'inside' if distance == 0 else f'{distance:.1f} ATR from'} "
                       f"an unmitigated {ob.direction} order block "
                       f"{ob.bottom:.2f}-{ob.top:.2f}")

        # A broken block flips polarity and can still be traded from the far side.
        for ob in breakers:
            if not ob.contains(price):
                continue
            sign = -1.0 if ob.direction == core.BULLISH else 1.0
            value = 0.5 if ob.has_fvg else 0.4
            if abs(sign * (0.4 + value * 0.4)) > abs(best_score):
                best_score = sign * (0.4 + value * 0.4)
                best_conf = 0.35
                best = ob
                why = (f"price is retesting a broken {ob.direction} order block "
                       f"(breaker) {ob.bottom:.2f}-{ob.top:.2f}")

        if best is None:
            return self.neutral(f"no order block near price on {tf}",
                                timeframe=tf, candidates=len(blocks))

        return self.signal(
            direction=direction_from_score(best_score), score=best_score, confidence=best_conf,
            rationale=why + ".",
            evidence={
                "timeframe": tf, "selected": best.as_dict(),
                "unmitigated": len([b for b in fresh if not b.mitigated]),
                "breakers": len(breakers),
                "recent": [b.as_dict() for b in blocks[-4:]],
            },
            levels={"ob_top": best.top, "ob_bottom": best.bottom, "ob_mid": best.midpoint},
        )
