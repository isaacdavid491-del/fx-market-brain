"""Agents from edition 0.6 of the advanced day-trading study.

Chapter 41 sets the precedence rule these follow: the most recent explicit
explanation of the same decision governs, earlier teaching supplies missing
background, and windows from different lessons stay separate rather than being
merged into one clock. Where the source leaves a condition unresolved, the
agent measures what it can and says so.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from backend.agents.base import (
    LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext, direction_from_score,
)
from backend.ict import core
from backend.ict.sessions import (
    at_checkpoint, noon_hour_range, primary_session, to_ny, venom_range,
)


class RejectionBlockAgent(BaseAgent):
    """Which boundary is price actually crossing: the body or the wick?

    Chapter 37. A bearish rejection block runs from a swing cluster's highest
    open-or-close to its highest wick. Price can cross the body reference
    while the wick extreme stays untouched, and the sweep language hides that
    difference. The agent reports the crossing explicitly.

    The source favours bearish rejection blocks in declining contexts and
    bullish ones in rising contexts, so the higher-timeframe trend gates how
    much confidence a block earns.
    """

    name = "rejection_block"
    role = "analyst"
    description = "Rejection blocks: the body reference against the wick extreme."
    default_weight = 1.5

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.mtf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        blocks = core.find_rejection_blocks(df, ctx.swings(tf))
        if not blocks:
            return self.neutral(f"no rejection blocks on {tf}", timeframe=tf)

        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        _, state = ctx.structure(ctx.htf)
        trend = state.get("trend", "neutral")

        best, best_value, crossing = None, 0.0, "neither"
        for block in blocks:
            where = block.crossing(price)
            if where == "neither":
                continue
            # A return that has already run the wick extreme is a different
            # event from one that stopped at the body reference.
            value = 1.0 if where == "body" else 0.45
            distance = abs(price - block.body_ref) / atr_val
            if distance > 2.0:
                continue
            value *= 1.0 - min(distance / 2.0, 1.0) * 0.5
            if value > best_value:
                best_value, best, crossing = value, block, where

        if best is None:
            return self.neutral(
                f"price has not crossed any rejection block boundary on {tf}",
                timeframe=tf, blocks=len(blocks),
            )

        sign = -1.0 if best.direction == core.BEARISH else 1.0
        score = sign * (0.45 + 0.4 * best_value)
        confidence = 0.25 + 0.4 * best_value
        # The lesson prefers bearish blocks in downtrends and the mirror.
        aligned = ((best.direction == core.BEARISH and trend == core.BEARISH)
                   or (best.direction == core.BULLISH and trend == core.BULLISH))
        if aligned:
            confidence = min(1.0, confidence + 0.15)
        else:
            confidence *= 0.7

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"Price crossed the {best.direction} rejection block's "
                f"{'body reference' if crossing == 'body' else 'wick extreme'} "
                f"(body {best.body_ref:.2f}, wick {best.wick_extreme:.2f}); "
                f"{ctx.htf} trend is {trend}."
            ),
            evidence={
                "timeframe": tf, "selected": best.as_dict(), "crossing": crossing,
                "htf_trend": trend, "aligned_with_trend": aligned,
                "note": "body and wick boundaries are separate observations",
            },
            levels={"rejection_body": best.body_ref,
                    "rejection_wick": best.wick_extreme},
        )


class VolumeImbalanceAgent(BaseAgent):
    """Body separations and the candles bounded by them.

    Appendix A names two objects this agent measures. A volume imbalance is
    the separation between adjacent candle *bodies* and has nothing to do with
    traded volume. A suspension block is a candle bounded by body separations
    at both ends, and the corrected definition does not require a conventional
    three-candle wick gap: neighbouring wick ranges may overlap.
    """

    name = "volume_imbalance"
    role = "analyst"
    description = "Body separations and suspension blocks near price."
    default_weight = 1.1

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        atr_val = max(ctx.atr(tf), 1e-9)
        imbalances = core.find_volume_imbalances(df, min_width=atr_val * 0.05)
        suspensions = core.find_suspension_blocks(df, min_width=atr_val * 0.05)
        if not imbalances:
            return self.neutral(f"no body separations on {tf}", timeframe=tf)

        n = len(df)
        price = ctx.price
        recent = [vi for vi in imbalances if n - 1 - vi.idx <= 40]
        if not recent:
            return self.neutral(f"body separations on {tf} are stale", timeframe=tf)

        near = min(recent, key=lambda vi: min(abs(price - vi.top), abs(price - vi.bottom)))
        distance = 0.0 if near.bottom <= price <= near.top else \
            min(abs(price - near.top), abs(price - near.bottom)) / atr_val
        if distance > 2.0:
            return self.neutral(
                f"nearest body separation is {distance:.1f} ATR away",
                timeframe=tf, count=len(recent),
            )

        sign = 1.0 if near.direction == core.BULLISH else -1.0
        proximity = 1.0 - min(distance / 2.0, 1.0)
        score = sign * (0.35 + 0.3 * proximity)
        confidence = 0.2 + 0.3 * proximity
        if suspensions:
            confidence = min(1.0, confidence + 0.1)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"A {near.direction} body separation {near.bottom:.2f}-{near.top:.2f} sits "
                f"{'at price' if distance == 0 else f'{distance:.1f} ATR away'}; "
                f"{len(suspensions)} suspension block(s) on {tf}."
            ),
            evidence={
                "timeframe": tf, "selected": near.as_dict(),
                "recent_count": len(recent),
                "suspension_blocks": suspensions[-2:],
                "note": "body separation, not traded volume",
            },
            levels={"vi_top": near.top, "vi_bottom": near.bottom},
        )


class VenomAgent(BaseAgent):
    """The 08:00-09:30 references and the paired arrival-and-departure signature.

    Chapter 24. The tutorial marks the high and low formed from 08:00 through
    09:30 New York as the session's initial references. Its bearish signature
    is a candle *closing* above the raided high pool, with inefficient
    delivery into that area and an inefficient departure away from it. The
    paired arrival and departure is the identifying feature: a brief wick
    above a high does not recover the description.

    The model needs a directional predisposition rather than its shape being
    hunted in a churned middle range, so an absent raid is reported as absent
    rather than relabelled.
    """

    name = "venom"
    role = "analyst"
    description = "Venom 08:00-09:30 references and the arrival/departure signature."
    default_weight = 1.3

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        window = venom_range(df, ctx.now_ts)
        if not window.complete or window.width <= 0:
            return self.neutral(
                "the 08:00-09:30 window has not closed, so its references are not final",
                window=window.as_dict(),
            )

        session = df[df["t"] >= window.end_ts]
        if len(session) < 4:
            return self.neutral("not enough bars since the window closed",
                                window=window.as_dict())

        atr_val = max(ctx.atr(tf), 1e-9)
        gaps = [g for g in ctx.fvgs(tf) if g.t >= window.start_ts]
        high = session["h"].to_numpy(dtype=float)
        low = session["l"].to_numpy(dtype=float)
        close = session["c"].to_numpy(dtype=float)

        evidence: Dict[str, Any] = {
            "window": window.as_dict(),
            "bars_since_window": len(session),
            "signature": "close beyond the raided pool, plus inefficient arrival and departure",
        }

        # Bearish: the high pool is raided and a candle CLOSES above it.
        closed_above = bool((close > window.high).any())
        wicked_above = bool((high > window.high).any())
        closed_below = bool((close < window.low).any())
        wicked_below = bool((low < window.low).any())
        evidence.update({"closed_above_high": closed_above, "wicked_above_high": wicked_above,
                         "closed_below_low": closed_below, "wicked_below_low": wicked_below})

        if not (wicked_above or wicked_below):
            return self.neutral(
                "neither Venom reference has been raided; the model wants a raid first",
                **evidence,
            )

        # Inefficient delivery into and away from the area, which is the
        # paired feature the source identifies.
        near_pool = [g for g in gaps
                     if abs(g.midpoint - (window.high if wicked_above else window.low))
                     <= 3.0 * atr_val]
        arrival_departure = len(near_pool) >= 2
        evidence["inefficiencies_near_pool"] = len(near_pool)
        evidence["paired_arrival_and_departure"] = arrival_departure

        price = ctx.price
        if closed_above and price < window.high:
            score, confidence = -0.75, 0.45
            why = f"a candle closed above the {window.high:.2f} reference and price is back below it"
        elif closed_below and price > window.low:
            score, confidence = 0.75, 0.45
            why = f"a candle closed below the {window.low:.2f} reference and price is back above it"
        elif wicked_above and not closed_above:
            return self.neutral(
                f"only a wick above {window.high:.2f}; the signature wants a close",
                **evidence,
            )
        elif wicked_below and not closed_below:
            return self.neutral(
                f"only a wick below {window.low:.2f}; the signature wants a close",
                **evidence,
            )
        else:
            return self.neutral("raid present but price has not returned through it",
                                **evidence)

        if arrival_departure:
            confidence = min(1.0, confidence + 0.2)
        else:
            confidence *= 0.7
            why += ", though the paired arrival and departure is not clearly present"

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=why + ".",
            evidence=evidence,
            levels={"venom_high": window.high, "venom_low": window.low},
        )


class CheckpointAgent(BaseAgent):
    """The Model 13 search checkpoints.

    Chapter 23 gives an index-futures morning window of 08:30-11:00 New York
    with checkpoints at 08:30, 09:30, 10:00 and 10:30, and afternoon
    checkpoints from 13:30 to 15:30. The source is explicit that these are
    opportunities to look for the model, not orders to trade at each one, so
    this agent takes no direction and only nudges conviction.
    """

    name = "checkpoint"
    role = "gate"
    description = "Model 13 search checkpoints; raises conviction slightly, never directs it."
    default_weight = 0.0

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        checkpoint = at_checkpoint(ctx.now_ts)
        local = to_ny(ctx.now_ts)
        minutes = local.hour * 60 + local.minute
        in_morning_window = 8 * 60 + 30 <= minutes < 11 * 60

        multiplier = 1.0
        if checkpoint:
            multiplier = 1.1
        elif not in_morning_window and minutes < 13 * 60 + 30:
            multiplier = 0.95

        return self.signal(
            direction=NEUTRAL, score=0.0, confidence=0.0, multiplier=multiplier,
            rationale=(
                f"{local.strftime('%H:%M %Z')}"
                + (f" is the {checkpoint} checkpoint" if checkpoint
                   else " is not a named checkpoint")
                + f"; conviction x{multiplier:.2f}."
            ),
            evidence={
                "checkpoint": checkpoint,
                "in_model13_morning_window": in_morning_window,
                "note": "checkpoints are opportunities to look, not to trade",
            },
        )
