"""Timing agents: killzone gating and the daily accumulation-manipulation-distribution cycle."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from backend.agents.base import AgentSignal, BaseAgent, MarketContext, direction_from_score
from backend.ict import core
from backend.ict.sessions import (
    active_sessions,
    in_rth,
    is_weekend,
    ny_day_start,
    primary_session,
    session_weight,
    slice_session,
    to_ny,
)


class KillzoneAgent(BaseAgent):
    """Gates the farm on time of day.

    ICT setups are time-dependent: the same pattern that works at the New York
    open is noise at lunch. This agent takes no direction. It returns a
    multiplier the orchestrator applies to the whole farm's conviction, and it
    vetoes outright when the market is closed for the weekend.
    """

    name = "killzone"
    role = "gate"
    description = "Session and killzone gating in New York time."
    default_weight = 0.0   # gates scale conviction, they never vote on direction

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        ts = ctx.now_ts
        ny = to_ny(ts)
        sessions = active_sessions(ts)
        primary = primary_session(ts)
        weight = session_weight(ts)
        rth = in_rth(ts)
        weekend = is_weekend(ts)

        require_kz = bool(ctx.config.get("require_killzone", True))
        floor = float(ctx.config.get("killzone_floor", 0.35))

        evidence: Dict[str, Any] = {
            "ny_time": ny.strftime("%Y-%m-%d %H:%M %Z"),
            "weekday": ny.strftime("%A"),
            "sessions": [s.name for s in sessions],
            "primary_session": primary.name if primary else None,
            "session_note": primary.note if primary else None,
            "session_weight": round(weight, 3),
            "regular_trading_hours": rth,
        }

        if weekend:
            return self.signal(
                direction="NEUTRAL", score=0.0, confidence=0.0, multiplier=0.0,
                veto=True, veto_reason="market closed for the weekend",
                rationale=f"{ny.strftime('%A %H:%M %Z')}: NASDAQ futures are closed.",
                evidence=evidence,
            )

        if require_kz and weight < floor:
            return self.signal(
                direction="NEUTRAL", score=0.0, confidence=0.0, multiplier=weight,
                veto=True,
                veto_reason=f"outside killzones (session weight {weight:.2f} < {floor:.2f})",
                rationale=(
                    f"{ny.strftime('%H:%M %Z')} is "
                    f"{primary.name if primary else 'outside every killzone'}; "
                    "the farm stands down."
                ),
                evidence=evidence,
            )

        # Inside a window: scale conviction by how good the window is.
        multiplier = 0.5 + 0.5 * weight
        return self.signal(
            direction="NEUTRAL", score=0.0, confidence=0.0, multiplier=multiplier,
            rationale=(
                f"{ny.strftime('%H:%M %Z')} — "
                f"{primary.name if primary else 'no named killzone'} "
                f"(weight {weight:.2f}, conviction x{multiplier:.2f})."
            ),
            evidence=evidence,
        )


class PowerOfThreeAgent(BaseAgent):
    """Reads the daily accumulation / manipulation / distribution cycle.

    The model: the session builds a range, then runs one side of it to trap
    breakout traders (the Judas swing), then expands the other way. A raid of
    the pre-open range that price reclaims says the manipulation leg is done
    and names the direction of the distribution leg.
    """

    name = "power_of_three"
    role = "analyst"
    description = "Daily open, Judas swing and the accumulation-manipulation-distribution cycle."
    default_weight = 1.4

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        ts = ctx.now_ts
        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        day_start = ny_day_start(ts)

        today = df[df["t"] >= day_start]
        if today.empty:
            return self.neutral("no bars yet for the current NY day")

        day_open = float(today["o"].iloc[0])
        day_high = float(today["h"].max())
        day_low = float(today["l"].min())

        # The Asian range is the accumulation the London/NY raid works against.
        asia = slice_session(df, day_start - 3600, "asia")
        asia_high = float(asia["h"].max()) if not asia.empty else None
        asia_low = float(asia["l"].min()) if not asia.empty else None

        score, confidence = 0.0, 0.0
        phase = "accumulation"
        notes = []

        if asia_high is not None and asia_low is not None:
            swept_high = day_high > asia_high
            swept_low = day_low < asia_low
            if swept_low and not swept_high and price > asia_low:
                phase = "distribution"
                score, confidence = 0.75, 0.55
                notes.append(f"Asian low {asia_low:.2f} raided and reclaimed")
            elif swept_high and not swept_low and price < asia_high:
                phase = "distribution"
                score, confidence = -0.75, 0.55
                notes.append(f"Asian high {asia_high:.2f} raided and reclaimed")
            elif swept_high and swept_low:
                phase = "expansion"
                # Both sides taken: follow the side price is holding.
                score = 0.4 if price > (asia_high + asia_low) / 2 else -0.4
                confidence = 0.3
                notes.append("both sides of the Asian range taken")
            elif swept_high or swept_low:
                phase = "manipulation"
                score = 0.5 if swept_high else -0.5
                confidence = 0.3
                notes.append("range broken and holding, expansion underway")
            else:
                notes.append("price still inside the Asian range")

        # Displacement away from the daily open corroborates the intended leg.
        drift = (price - day_open) / atr_val
        if abs(drift) > 0.5:
            score += 0.2 * (1.0 if drift > 0 else -1.0)
            confidence = max(confidence, 0.25)
            notes.append(f"trading {drift:+.1f} ATR from the {day_open:.2f} daily open")

        if confidence == 0.0:
            return self.neutral(
                "no readable power-of-three signature yet",
                day_open=round(day_open, 2), asia_high=asia_high, asia_low=asia_low,
            )

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=f"Phase {phase}: " + "; ".join(notes) + ".",
            evidence={
                "timeframe": tf, "phase": phase,
                "day_open": round(day_open, 2),
                "day_high": round(day_high, 2), "day_low": round(day_low, 2),
                "asia_high": round(asia_high, 2) if asia_high else None,
                "asia_low": round(asia_low, 2) if asia_low else None,
                "atr_from_open": round(drift, 2),
            },
            levels={k: v for k, v in {
                "day_open": day_open, "asia_high": asia_high, "asia_low": asia_low,
            }.items() if v is not None},
        )


class PremiumDiscountAgent(BaseAgent):
    """Only buy at a discount, only sell at a premium.

    Scores where price sits inside the dealing range, with a bonus when it is
    in the 0.62-0.79 optimal trade entry band, the retracement depth the model
    expects a continuation to begin from.
    """

    name = "premium_discount"
    role = "analyst"
    description = "Dealing-range position, equilibrium and the optimal trade entry band."
    default_weight = 1.3

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.mtf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        rng = ctx.dealing_range(tf, lookback=int(self.params.get("lookback", 60)))
        if rng is None or rng.span <= 0:
            return self.neutral(f"no dealing range on {tf}")

        price = ctx.price
        pos = rng.position(price)
        zone = rng.zone(price)

        # Discount favours longs, premium favours shorts; the edges are strongest.
        score = (0.5 - pos) * 2.0 * 0.8
        confidence = 0.25 + 0.4 * min(1.0, abs(pos - 0.5) * 2.0)

        long_ote = core.ote_zone(rng, core.BULLISH)
        short_ote = core.ote_zone(rng, core.BEARISH)
        in_long_ote = long_ote[0] <= price <= long_ote[1]
        in_short_ote = short_ote[0] <= price <= short_ote[1]

        if in_long_ote and rng.direction == core.BULLISH:
            score = max(score, 0.7)
            confidence = min(1.0, confidence + 0.2)
        elif in_short_ote and rng.direction == core.BEARISH:
            score = min(score, -0.7)
            confidence = min(1.0, confidence + 0.2)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"Price at {pos:.0%} of the {tf} dealing range ({zone}), "
                f"equilibrium {rng.equilibrium:.2f}"
                + (", inside the long OTE band" if in_long_ote else "")
                + (", inside the short OTE band" if in_short_ote else "")
                + "."
            ),
            evidence={
                "timeframe": tf, "range": rng.as_dict(), "zone": zone,
                "position": round(float(pos), 3),
                "long_ote": [round(long_ote[0], 2), round(long_ote[1], 2)],
                "short_ote": [round(short_ote[0], 2), round(short_ote[1], 2)],
                "in_long_ote": in_long_ote, "in_short_ote": in_short_ote,
            },
            levels={"equilibrium": rng.equilibrium, "range_high": rng.high, "range_low": rng.low},
        )
