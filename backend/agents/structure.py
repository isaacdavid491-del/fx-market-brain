"""Structure agents: higher-timeframe bias and mid-timeframe structure shifts."""
from __future__ import annotations

from typing import Any, Dict

from backend.agents.base import LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext, direction_from_score
from backend.ict import core
from backend.ict.sessions import previous_day_levels


class HTFBiasAgent(BaseAgent):
    """Sets the directional frame from the higher timeframe.

    Nothing else in the farm is allowed to define "the trend". This agent owns
    that, combining the HTF structural trend with where price sits inside the
    HTF dealing range, so an extended market is trusted less than one that has
    just turned from a discount.
    """

    name = "htf_bias"
    role = "analyst"
    description = "Higher-timeframe directional bias from structure and range position."
    default_weight = 2.0

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.htf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        events, state = ctx.structure(tf)
        trend = state.get("trend", "neutral")
        rng = ctx.dealing_range(tf, lookback=self.params.get("range_lookback", 60))
        price = ctx.price

        score = 0.0
        if trend == core.BULLISH:
            score = 0.7
        elif trend == core.BEARISH:
            score = -0.7

        zone, pos = "unknown", 0.5
        if rng is not None:
            pos = rng.position(price)
            zone = rng.zone(price)
            # A bullish trend deep in premium is late; in discount it is fresh.
            if trend == core.BULLISH:
                score += 0.3 * (0.5 - pos) * 2.0
            elif trend == core.BEARISH:
                score -= 0.3 * (pos - 0.5) * 2.0

        recent = events[-1] if events else None
        bars_since = (len(df) - 1 - recent.idx) if recent else 999
        # Conviction decays as the last structural event recedes.
        freshness = max(0.0, 1.0 - bars_since / float(self.params.get("decay_bars", 40)))
        confidence = 0.35 + 0.5 * freshness if trend != "neutral" else 0.1

        levels = previous_day_levels(ctx.frame(ctx.mtf), ctx.now_ts)
        evidence: Dict[str, Any] = {
            "timeframe": tf,
            "trend": trend,
            "bars_since_event": bars_since,
            "last_event": state.get("last_event"),
            "range": rng.as_dict() if rng else None,
            "range_zone": zone,
            "range_position": round(float(pos), 3),
            "prev_day": {k: (round(v, 2) if v is not None else None) for k, v in levels.items()},
        }
        rationale = (
            f"{tf} structure is {trend}; price sits in {zone} "
            f"({pos:.0%} of range), {bars_since} bars since the last "
            f"{(recent.kind if recent else 'n/a')}."
        )
        return self.signal(direction=direction_from_score(score), score=score,
                           confidence=confidence, rationale=rationale, evidence=evidence)


class MarketStructureAgent(BaseAgent):
    """Reads BOS and CHoCH on the trading timeframe.

    A change of character is the first evidence that the prior leg is finished,
    so it is scored as a stronger, higher-confidence event than a continuation
    break, but only while it is fresh.
    """

    name = "market_structure"
    role = "analyst"
    description = "Break of structure / change of character on the trading timeframe."
    default_weight = 1.8

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.mtf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        events, state = ctx.structure(tf)
        if not events:
            return self.neutral(f"no structure breaks on {tf}", timeframe=tf)

        last = events[-1]
        bars_since = len(df) - 1 - last.idx
        decay = float(self.params.get("decay_bars", 20))
        freshness = max(0.0, 1.0 - bars_since / decay)
        if freshness <= 0:
            return self.neutral(
                f"last {tf} {last.kind} is {bars_since} bars stale",
                timeframe=tf, last_event=last.as_dict(),
            )

        base = 0.9 if last.kind == "CHoCH" else 0.65
        score = base * (1.0 if last.direction == core.BULLISH else -1.0)
        confidence = (0.45 + 0.45 * freshness) * (1.0 if last.kind == "CHoCH" else 0.85)

        # Consecutive same-direction breaks are a trending leg worth respecting.
        streak = 1
        for ev in reversed(events[:-1]):
            if ev.direction == last.direction:
                streak += 1
            else:
                break
        if streak >= 2 and last.kind == "BOS":
            score *= 1.1
            confidence = min(1.0, confidence + 0.05)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"{tf} {last.kind} {last.direction} through {last.level:.2f}, "
                f"{bars_since} bars ago (streak {streak})."
            ),
            evidence={
                "timeframe": tf, "last_event": last.as_dict(), "bars_since": bars_since,
                "streak": streak, "trend": state.get("trend"),
                "protected_high": state.get("protected_high"),
                "protected_low": state.get("protected_low"),
                "recent_events": [e.as_dict() for e in events[-5:]],
            },
            levels={
                "structure_level": last.level,
                **({"protected_low": state["protected_low"]["price"]} if state.get("protected_low") else {}),
                **({"protected_high": state["protected_high"]["price"]} if state.get("protected_high") else {}),
            },
        )
