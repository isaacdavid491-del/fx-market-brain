"""SMT divergence: NASDAQ against its correlated index."""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from backend.agents.base import AgentSignal, BaseAgent, MarketContext, direction_from_score
from backend.ict import core


class SMTDivergenceAgent(BaseAgent):
    """Compares NASDAQ's swing extremes with a correlated index (S&P / Dow).

    Two indices that normally move together should make their highs and lows
    together. When NASDAQ makes a lower low and the S&P does not, the low was
    a stop raid rather than genuine weakness: one of the two was lying, and the
    one that failed to confirm is telling the truth.

    Abstains, loudly, when no correlated series is available rather than
    inventing a reading.
    """

    name = "smt_divergence"
    role = "analyst"
    description = "Smart-money-technique divergence against a correlated index."
    default_weight = 1.5

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        tf = ctx.ltf
        df = ctx.frame(tf)
        peer = ctx.correlated_frames.get(tf)

        if peer is None or peer.empty:
            return self.neutral(
                "no correlated index series supplied; SMT cannot be evaluated",
                correlated_symbol=ctx.correlated_symbol,
            )
        if len(df) < self.min_bars or len(peer) < self.min_bars:
            return self.neutral("not enough overlapping bars for SMT")

        lookback = int(self.params.get("lookback", 24))
        # Align on timestamp so a gap in either feed cannot shift the comparison.
        merged = df[["t", "h", "l"]].merge(
            peer[["t", "h", "l"]], on="t", how="inner", suffixes=("", "_peer")
        )
        if len(merged) < lookback + 2:
            return self.neutral(
                f"only {len(merged)} aligned bars, need {lookback + 2}",
                correlated_symbol=ctx.correlated_symbol,
            )

        window = merged.tail(lookback)
        half = max(len(window) // 2, 2)
        first, second = window.iloc[:half], window.iloc[half:]

        nq_ll = float(second["l"].min()) < float(first["l"].min())
        pe_ll = float(second["l_peer"].min()) < float(first["l_peer"].min())
        nq_hh = float(second["h"].max()) > float(first["h"].max())
        pe_hh = float(second["h_peer"].max()) > float(first["h_peer"].max())

        score, confidence, why = 0.0, 0.0, ""
        kind = "none"

        if nq_ll and not pe_ll:
            kind, score, confidence = "bullish_smt", 0.8, 0.55
            why = f"{ctx.symbol} made a lower low that {ctx.correlated_symbol} did not confirm"
        elif pe_ll and not nq_ll:
            kind, score, confidence = "bullish_smt_peer", 0.6, 0.45
            why = f"{ctx.correlated_symbol} made a lower low that {ctx.symbol} refused to follow"
        elif nq_hh and not pe_hh:
            kind, score, confidence = "bearish_smt", -0.8, 0.55
            why = f"{ctx.symbol} made a higher high that {ctx.correlated_symbol} did not confirm"
        elif pe_hh and not nq_hh:
            kind, score, confidence = "bearish_smt_peer", -0.6, 0.45
            why = f"{ctx.correlated_symbol} made a higher high that {ctx.symbol} refused to follow"
        else:
            return self.neutral(
                f"{ctx.symbol} and {ctx.correlated_symbol} are in agreement, no divergence",
                aligned_bars=len(merged), timeframe=tf,
            )

        # Weak correlation makes any divergence meaningless.
        corr = float(merged["l"].pct_change().corr(merged["l_peer"].pct_change()) or 0.0)
        if corr < float(self.params.get("min_correlation", 0.3)):
            return self.neutral(
                f"correlation with {ctx.correlated_symbol} is only {corr:.2f}; "
                "divergence is not meaningful",
                correlation=round(corr, 3),
            )
        confidence *= min(1.0, max(0.3, corr))

        return self.signal(
            direction=direction_from_score(score), score=score, confidence=confidence,
            rationale=why + f" (correlation {corr:.2f}).",
            evidence={
                "timeframe": tf, "kind": kind,
                "peer": ctx.correlated_symbol, "correlation": round(corr, 3),
                "aligned_bars": len(merged),
                "nasdaq_lower_low": nq_ll, "peer_lower_low": pe_ll,
                "nasdaq_higher_high": nq_hh, "peer_higher_high": pe_hh,
            },
        )
