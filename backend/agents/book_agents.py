"""Agents implementing the specific teachings of the advanced day-trading study.

Each agent below traces to a numbered chapter, and the docstrings say which.
Where the source is explicit (a close is required for inversion; anchors are
frozen once a window closes; three "first gap" identities are distinct) the
rule is implemented literally. Where the book itself records a condition as
unresolved, the agent reports the measurement and abstains from claiming more
than the evidence supports.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from backend.agents.base import (
    LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext, direction_from_score,
)
from backend.ict import core
from backend.ict.sessions import in_rth, ny_day_start, to_ny


class SessionRangeAgent(BaseAgent):
    """The 07:00-09:00 premarket and 09:30-10:00 opening ranges, graded.

    Chapters 4 and 5. Two rules matter and both are enforced here: the ranges
    are kept separate because they can have different midpoints on the same
    day, and a range is only graded once its window has closed. An opening
    range extreme formed at 09:57 was not available at 09:40, so an incomplete
    window produces no signal rather than a provisional one.
    """

    name = "session_range"
    role = "analyst"
    description = "Premarket and opening ranges, frozen after their window and graded into octants."
    default_weight = 1.7

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        premarket = ctx.premarket(tf)
        opening = ctx.opening_range(tf)
        price = ctx.price
        evidence: Dict[str, Any] = {
            "premarket": premarket.as_dict(),
            "opening_range": opening.as_dict(),
            "note": "a range is graded only after its window closes",
        }

        # Prefer the opening range once it is complete, else the premarket.
        chosen = None
        if opening.complete and opening.width > 0:
            chosen = opening
        elif premarket.complete and premarket.width > 0:
            chosen = premarket

        if chosen is None:
            active = "opening range" if not opening.complete else "premarket"
            return self.neutral(
                f"the {active} window has not closed yet; anchors are not final",
                **evidence,
            )

        position = chosen.position(price)
        if position is None:
            return self.neutral("range has no width", **evidence)

        graded = core.grade_range(chosen.low, chosen.high, 8)
        evidence["graded_octants"] = {k: round(v, 2) for k, v in graded.items()
                                      if k.startswith("q")}
        evidence["position_in_range"] = round(position, 3)
        evidence["range_used"] = chosen.name

        # Outside the range is an expansion away from it; inside, the octant
        # location is read as premium or discount relative to its midpoint.
        if position > 1.0:
            score, confidence = 0.6, 0.45
            why = f"trading above the {chosen.name} high {chosen.high:.2f}"
        elif position < 0.0:
            score, confidence = -0.6, 0.45
            why = f"trading below the {chosen.name} low {chosen.low:.2f}"
        else:
            score = (0.5 - position) * 2.0 * 0.7
            confidence = 0.2 + 0.35 * min(1.0, abs(position - 0.5) * 2.0)
            octant = int(min(position, 0.999) * 8)
            why = (f"inside the {chosen.name} range at octant {octant + 1} of 8, "
                   f"midpoint {chosen.midpoint:.2f}")

        levels = {f"{chosen.name}_high": chosen.high, f"{chosen.name}_low": chosen.low,
                  f"{chosen.name}_mid": chosen.midpoint}
        # A half-range projection is a stated objective, not a statistic.
        levels["projection_up"] = core.project_range(chosen.low, chosen.high, 1.5)
        levels["projection_down"] = core.project_range(chosen.low, chosen.high, -0.5)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=why + ".", evidence=evidence, levels=levels,
        )


class OpeningGapAgent(BaseAgent):
    """The RTH opening gap and the new-week opening gap.

    Chapter 4 for the definitions: the RTH gap runs from the prior session's
    final one-minute close to the 09:30 open, and the new-week gap from
    Friday's final close to the Sunday reopen. Chapter 18 notes the roughly
    70% midpoint-visit figure is something to measure rather than an
    established result, so this agent treats the midpoint as a candidate draw
    and nothing more.
    """

    name = "opening_gap"
    role = "analyst"
    description = "RTH and new-week opening gaps as references and candidate draws."
    default_weight = 1.4

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        gap = ctx.opening_gap(tf)
        nwog = ctx.new_week_gap(tf)
        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        evidence: Dict[str, Any] = {"opening_gap": gap, "new_week_opening_gap": nwog}

        if not gap.get("complete") or not gap.get("width"):
            return self.neutral("no measurable RTH opening gap for this session", **evidence)

        midpoint = float(gap["midpoint"])
        distance = (midpoint - price) / atr_val
        levels = {"opening_gap_mid": midpoint,
                  "opening_gap_high": max(gap["open"], gap["prior_close"]),
                  "opening_gap_low": min(gap["open"], gap["prior_close"])}
        if nwog.get("complete") and nwog.get("midpoint"):
            levels["nwog_mid"] = float(nwog["midpoint"])

        # An unvisited midpoint above price is a draw upward, and vice versa.
        reach = float(self.params.get("reach_atr", 4.0))
        if abs(distance) > reach:
            return self.neutral(
                f"opening-gap midpoint {midpoint:.2f} is {abs(distance):.1f} ATR away, "
                "too far to organise this move",
                **evidence,
            )

        score = float(np.clip(distance / reach, -1.0, 1.0)) * 0.7
        confidence = 0.25 + 0.3 * (1.0 - min(abs(distance) / reach, 1.0))
        if not in_rth(ctx.now_ts):
            confidence *= 0.6   # the reference belongs to the cash session

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"{gap['direction'].replace('_', ' ')} of {gap['width']:.2f} points; "
                f"midpoint {midpoint:.2f} sits {abs(distance):.1f} ATR "
                f"{'above' if distance > 0 else 'below'} price."
            ),
            evidence=evidence, levels=levels,
        )


class InversionAgent(BaseAgent):
    """Gaps that failed on a close and now act from the other side.

    Chapter 10 is the whole point of this agent. A wick through a bullish gap
    is not a bearish inversion; a close below is required. The two events are
    kept separate because a stop can be taken out by the wick well before the
    label qualifies, and classifying it later does not retroactively protect
    anything.
    """

    name = "inversion"
    role = "analyst"
    description = "Inverted fair value gaps, qualified by a close beyond the boundary."
    default_weight = 1.8

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        inversions = ctx.inversions(tf)
        if not inversions:
            return self.neutral(f"no close-confirmed inversions on {tf}", timeframe=tf)

        n = len(df)
        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        decay = float(self.params.get("decay_bars", 30))

        best, best_value = None, 0.0
        for inv in inversions:
            bars_since = n - 1 - inv.inverted_idx
            freshness = max(0.0, 1.0 - bars_since / decay)
            if freshness <= 0:
                continue
            # An inverted gap is normally engaged on its retest.
            in_zone = inv.gap.bottom <= price <= inv.gap.top
            distance = 0.0 if in_zone else min(abs(price - inv.gap.top),
                                               abs(price - inv.gap.bottom)) / atr_val
            if distance > 2.0:
                continue
            proximity = 1.0 - min(distance / 2.0, 1.0)
            value = freshness * (0.6 + 0.4 * proximity)
            if value > best_value:
                best_value, best = value, inv

        if best is None:
            return self.neutral(f"inversions on {tf} are stale or far from price",
                                timeframe=tf, count=len(inversions))

        sign = 1.0 if best.new_direction == core.BULLISH else -1.0
        score = sign * min(1.0, 0.55 + 0.4 * best_value)
        confidence = min(0.9, 0.35 + 0.5 * best_value)

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"A {best.gap.direction} gap {best.gap.bottom:.2f}-{best.gap.top:.2f} "
                f"failed on a close at {best.close_price:.2f}, so it now reads "
                f"{best.new_direction}"
                + (" and has been retested." if best.retested else ", awaiting its retest.")
            ),
            evidence={
                "timeframe": tf, "selected": best.as_dict(),
                "qualifying_event": "close beyond the boundary, not a wick",
                "total_inversions": len(inversions),
            },
            levels={"inversion_top": best.gap.top, "inversion_bottom": best.gap.bottom,
                    "inversion_ce": best.gap.midpoint},
        )


class FirstPresentedGapAgent(BaseAgent):
    """The session's first gaps, by their three distinct identities.

    Chapter 11. The first chronological gap after the open, the first gap that
    displaced beyond a relevant swing, and the first later gap opposite to the
    chronological first, which the source calls the first-presented reflection.
    A later displacing gap can take emphasis without changing which gap was
    chronologically first, so all three are reported rather than collapsed.
    """

    name = "first_presented_gap"
    role = "analyst"
    description = "Chronological, displacement and reflection gap identities after the open."
    default_weight = 1.5

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        local = to_ny(ctx.now_ts)
        if local.hour < 9 or (local.hour == 9 and local.minute < 30):
            return self.neutral("the cash session has not opened yet",
                                ny_time=local.strftime("%H:%M"))

        presented = ctx.presented_gaps(tf)
        found = {k: v for k, v in presented.items() if v is not None}
        if not found:
            return self.neutral("no gaps have formed since the open", timeframe=tf)

        price = ctx.price
        atr_val = max(ctx.atr(tf), 1e-9)
        evidence: Dict[str, Any] = {
            "timeframe": tf,
            "identities": {k: v.as_dict() for k, v in found.items()},
            "note": "chronological, displacement and reflection are distinct identities",
        }

        # The displacing gap carries the most weight, then the reflection,
        # then the chronological anchor.
        for key, weight in (("displacement", 1.0), ("reflection", 0.8), ("chronological", 0.6)):
            candidate = found.get(key)
            if candidate is None:
                continue
            gap = candidate.gap
            if gap.filled:
                continue
            distance = 0.0 if gap.contains(price) else \
                min(abs(price - gap.top), abs(price - gap.bottom)) / atr_val
            if distance > 2.5:
                continue
            sign = 1.0 if gap.direction == core.BULLISH else -1.0
            # The reflection gap points against the chronological first.
            proximity = 1.0 - min(distance / 2.5, 1.0)
            score = sign * weight * (0.45 + 0.35 * proximity)
            confidence = 0.25 + 0.4 * proximity * weight
            return self.signal(
                direction=direction_from_score(score), score=score, confidence=confidence,
                rationale=(
                    f"The first {key} gap since the open is {gap.direction}, "
                    f"{gap.bottom:.2f}-{gap.top:.2f}, "
                    f"{'containing price' if distance == 0 else f'{distance:.1f} ATR away'}."
                ),
                evidence=evidence,
                levels={"presented_top": gap.top, "presented_bottom": gap.bottom,
                        "presented_ce": gap.midpoint},
            )

        return self.neutral("the session's first gaps are filled or far from price", **evidence)


class ObsidianWickAgent(BaseAgent):
    """Opposing wick midpoints and the interval between them.

    Chapter 8. The measurement is exact and implemented literally: the upper
    wick midpoint is [max(O,C) + H] / 2 and the lower is [L + min(O,C)] / 2.
    The book is equally clear that any two opposing wicks are *not* sufficient
    to establish the method, and that the qualifying event and protection
    rules remain unresolved, so this agent keeps its confidence low and says
    so in the rationale.
    """

    name = "obsidian_wicks"
    role = "analyst"
    description = "Opposing wick midpoint intervals (measurement only, qualification unresolved)."
    default_weight = 0.9

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        if len(df) < self.min_bars:
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        pairs = ctx.wick_pairs(tf, lookback=int(self.params.get("lookback", 20)))
        if not pairs:
            return self.neutral(f"no opposing wick pairs on {tf}", timeframe=tf)

        price = ctx.price
        inside = [p for p in pairs if p.contains(price)]
        if not inside:
            return self.neutral(
                "price is outside every measured wick interval",
                timeframe=tf, pairs=[p.as_dict() for p in pairs[:3]],
            )

        pair = inside[0]
        low, high = pair.interval
        # Position inside the interval is the only directional read taken, and
        # it is deliberately a weak one.
        position = (price - low) / (high - low) if high > low else 0.5
        score = (0.5 - position) * 2.0 * 0.5
        confidence = 0.2

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"Price sits inside an opposing wick interval {low:.2f}-{high:.2f}. "
                "Geometry only: the qualifying event for this setup is not "
                "established by the source."
            ),
            evidence={
                "timeframe": tf, "selected": pair.as_dict(),
                "caveat": "opposing wicks alone do not establish the setup",
            },
            levels={"wick_interval_low": low, "wick_interval_high": high},
        )


class DeliveryResistanceAgent(BaseAgent):
    """Low-resistance versus high-resistance delivery along the route.

    Chapter 7. Low resistance combines supportive same-direction closes with
    gaps left partly unfilled. Repeated full fills and back-and-forth movement
    indicate resistance. Chapter 7 also insists resistance is measured through
    the route rather than by the destination: a favourable ending does not
    erase a path that would have stopped the trade.

    This agent scores the quality of delivery, not a direction of its own, so
    it reports a modest score in the direction price is actually being
    delivered and leaves the thesis to the other agents.
    """

    name = "delivery_resistance"
    role = "analyst"
    description = "Whether recent delivery is low- or high-resistance along the route."
    default_weight = 1.2

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        lookback = int(self.params.get("lookback", 30))
        if len(df) < max(self.min_bars, lookback):
            return self.neutral(f"not enough {tf} bars ({len(df)})")

        window = df.tail(lookback)
        closes = window["c"].to_numpy(dtype=float)
        opens = window["o"].to_numpy(dtype=float)

        up_closes = int((closes > opens).sum())
        down_closes = int((closes < opens).sum())
        net = float(closes[-1] - closes[0])
        direction = core.BULLISH if net > 0 else core.BEARISH

        # Supportive closes: those agreeing with the net move.
        supportive = up_closes if direction == core.BULLISH else down_closes
        support_ratio = supportive / max(len(window), 1)

        # Gaps left partly unfilled are the low-resistance signature.
        recent_gaps = [g for g in ctx.fvgs(tf) if g.idx >= len(df) - lookback]
        agreeing = [g for g in recent_gaps if g.direction == direction]
        unfilled = [g for g in agreeing if not g.filled]
        fill_ratio = (len(unfilled) / len(agreeing)) if agreeing else 0.0

        # Back-and-forth: path length relative to net displacement.
        path = float(np.abs(np.diff(closes)).sum())
        efficiency = float(abs(net) / path) if path > 0 else 0.0

        quality = float(0.4 * support_ratio + 0.3 * fill_ratio + 0.3 * min(efficiency * 3.0, 1.0))
        # bool() because a numpy comparison yields numpy.bool_, which is not
        # JSON-serialisable once the evidence reaches the API.
        low_resistance = bool(quality >= float(self.params.get("threshold", 0.5)))

        sign = 1.0 if direction == core.BULLISH else -1.0
        score = sign * (quality - 0.5) * 1.6
        confidence = 0.2 + 0.4 * abs(quality - 0.5) * 2

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=(
                f"{'Low' if low_resistance else 'High'}-resistance {direction} delivery: "
                f"{supportive}/{len(window)} supportive closes, "
                f"{len(unfilled)}/{len(agreeing) or 0} agreeing gaps still unfilled, "
                f"path efficiency {efficiency:.2f}."
            ),
            evidence={
                "timeframe": tf, "delivery_direction": direction,
                "low_resistance": low_resistance,
                "quality": round(quality, 3),
                "support_ratio": round(support_ratio, 3),
                "unfilled_gap_ratio": round(fill_ratio, 3),
                "path_efficiency": round(efficiency, 3),
            },
        )
