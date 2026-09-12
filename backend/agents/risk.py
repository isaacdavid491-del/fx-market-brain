"""Risk agent: the farm's veto holder and position sizer."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.agents.base import AgentSignal, BaseAgent, MarketContext
from backend.ict.sessions import is_weekend, to_ny


class RiskManagerAgent(BaseAgent):
    """Checks conditions under which the farm should not trade at all.

    This agent never expresses a view on direction. It exists so that no
    combination of enthusiastic analysts can push a trade through when the
    account, the data, or the market state says no. Vetoes here are absolute.
    """

    name = "risk_manager"
    role = "risk"
    description = "Account, data quality and volatility guardrails with veto power."
    default_weight = 0.0

    def evaluate(self, ctx: MarketContext) -> AgentSignal:
        cfg = ctx.config
        evidence: Dict[str, Any] = {}
        ltf = ctx.frame(ctx.ltf)

        # -- data quality --------------------------------------------------
        if ltf.empty:
            return self._veto("no price data for the entry timeframe", evidence)

        staleness = int(ctx.now_ts) - int(ltf["t"].iloc[-1])
        max_stale = int(cfg.get("max_staleness_seconds", 900))
        evidence["data_staleness_seconds"] = staleness
        if staleness > max_stale:
            return self._veto(
                f"price data is {staleness}s stale (limit {max_stale}s)", evidence
            )

        # -- calendar ------------------------------------------------------
        if is_weekend(ctx.now_ts):
            return self._veto("weekend: market closed", evidence)

        # -- account state -------------------------------------------------
        equity = float(ctx.equity)
        start_equity = float(cfg.get("starting_equity", equity) or equity)
        day_pnl_pct = float(cfg.get("day_pnl_pct", 0.0))
        daily_loss_limit = float(cfg.get("daily_loss_limit_pct", 0.03))
        evidence["equity"] = round(equity, 2)
        evidence["day_pnl_pct"] = round(day_pnl_pct, 4)
        evidence["daily_loss_limit_pct"] = daily_loss_limit

        if day_pnl_pct <= -abs(daily_loss_limit):
            return self._veto(
                f"daily loss limit hit ({day_pnl_pct:.2%} <= -{daily_loss_limit:.2%})",
                evidence,
            )

        open_positions = int(cfg.get("open_positions", 0))
        max_positions = int(cfg.get("max_open_positions", 1))
        evidence["open_positions"] = open_positions
        if open_positions >= max_positions:
            return self._veto(
                f"already holding {open_positions} position(s), limit {max_positions}",
                evidence,
            )

        trades_today = int(cfg.get("trades_today", 0))
        max_trades = int(cfg.get("max_trades_per_day", 3))
        evidence["trades_today"] = trades_today
        if trades_today >= max_trades:
            return self._veto(
                f"{trades_today} trades already taken today, limit {max_trades}", evidence
            )

        # -- volatility regime ---------------------------------------------
        atr_val = ctx.atr(ctx.ltf)
        price = ctx.price
        atr_pct = (atr_val / price) if price else 0.0
        evidence["atr"] = round(atr_val, 4)
        evidence["atr_pct_of_price"] = round(atr_pct, 5)

        min_atr_pct = float(cfg.get("min_atr_pct", 0.0002))
        max_atr_pct = float(cfg.get("max_atr_pct", 0.02))
        if atr_pct < min_atr_pct:
            return self._veto(
                f"volatility too low to pay for the spread (ATR {atr_pct:.4%} of price)",
                evidence,
            )
        if atr_pct > max_atr_pct:
            return self._veto(
                f"volatility abnormally high (ATR {atr_pct:.2%} of price); standing aside",
                evidence,
            )

        # Size down in unusually fast conditions rather than refusing outright.
        multiplier = 1.0
        typical = float(np.nanmedian(ctx.frame(ctx.ltf)["h"] - ctx.frame(ctx.ltf)["l"]) or atr_val)
        if typical > 0 and atr_val > 1.8 * typical:
            multiplier = 0.6
            evidence["volatility_note"] = "elevated ATR versus typical bar range, size reduced"

        risk_pct = float(ctx.risk_per_trade)
        evidence["risk_per_trade_pct"] = risk_pct
        evidence["risk_budget_currency"] = round(equity * risk_pct * multiplier, 2)

        return self.signal(
            direction="NEUTRAL", score=0.0, confidence=0.0, multiplier=multiplier,
            rationale=(
                f"Risk checks pass: {risk_pct:.2%} of {equity:,.0f} equity at stake, "
                f"ATR {atr_pct:.3%} of price, conviction x{multiplier:.2f}."
            ),
            evidence=evidence,
        )

    def _veto(self, reason: str, evidence: Dict[str, Any]) -> AgentSignal:
        return self.signal(
            direction="NEUTRAL", score=0.0, confidence=0.0, multiplier=0.0,
            veto=True, veto_reason=reason,
            rationale=f"Risk veto: {reason}.", evidence=evidence,
        )


def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  contract_value: float = 1.0, max_units: Optional[float] = None) -> Dict[str, float]:
    """Units such that a stop-out costs exactly the risk budget.

    `contract_value` is the currency value of a one-point move in one unit, so
    the same maths covers a CFD, a micro future and a cash index.
    """
    stop_distance = abs(float(entry) - float(stop))
    if stop_distance <= 0 or equity <= 0 or risk_pct <= 0:
        return {"units": 0.0, "risk_currency": 0.0, "stop_distance": stop_distance}
    risk_currency = equity * risk_pct
    units = risk_currency / (stop_distance * max(contract_value, 1e-9))
    if max_units is not None:
        units = min(units, max_units)
    return {
        "units": round(units, 4),
        "risk_currency": round(risk_currency, 2),
        "stop_distance": round(stop_distance, 4),
    }
