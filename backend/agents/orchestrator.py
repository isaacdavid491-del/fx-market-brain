"""The agent farm: runs the specialists and turns their votes into a trade plan."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agents.base import LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext
from backend.agents.imbalance import FairValueGapAgent, OrderBlockAgent
from backend.agents.liquidity import LiquidityDrawAgent, SweepAgent
from backend.agents.risk import RiskManagerAgent, position_size
from backend.agents.smt import SMTDivergenceAgent
from backend.agents.structure import HTFBiasAgent, MarketStructureAgent
from backend.agents.timing import KillzoneAgent, PowerOfThreeAgent, PremiumDiscountAgent

log = logging.getLogger("ict.farm")

DEFAULT_CONFIG: Dict[str, Any] = {
    # timeframe roles
    "htf": "1h",
    "mtf": "15m",
    "ltf": "5m",
    # decision thresholds
    "entry_threshold": 0.22,      # normalised net conviction needed to act
    "min_agreement": 0.55,        # share of directional weight that must agree
    "min_rr": 1.8,                # reject plans that do not pay enough
    "target_rr": 3.0,
    # gating
    "require_killzone": True,
    "killzone_floor": 0.35,
    # risk
    "risk_per_trade": 0.005,
    "max_trades_per_day": 3,
    "max_open_positions": 1,
    "daily_loss_limit_pct": 0.03,
    "stop_buffer_atr": 0.35,
    "min_stop_atr": 0.6,          # a stop inside one bar's range is noise, not a stop
    "max_stop_atr": 4.0,
    "contract_value": 1.0,
}


def default_agents() -> List[BaseAgent]:
    """The roster. Weights encode how much the model trusts each concept."""
    return [
        HTFBiasAgent(),
        MarketStructureAgent(),
        SweepAgent(),
        LiquidityDrawAgent(),
        FairValueGapAgent(),
        OrderBlockAgent(),
        PremiumDiscountAgent(),
        PowerOfThreeAgent(),
        SMTDivergenceAgent(),
        KillzoneAgent(),
        RiskManagerAgent(),
    ]


@dataclass
class TradePlan:
    direction: str
    entry: float
    stop: float
    take_profit: float
    secondary_target: Optional[float]
    risk_reward: float
    units: float
    risk_currency: float
    stop_distance: float
    entry_type: str
    reasoning: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": round(self.entry, 2),
            "stop": round(self.stop, 2),
            "take_profit": round(self.take_profit, 2),
            "secondary_target": round(self.secondary_target, 2) if self.secondary_target else None,
            "risk_reward": round(self.risk_reward, 2),
            "units": self.units,
            "risk_currency": self.risk_currency,
            "stop_distance": round(self.stop_distance, 2),
            "entry_type": self.entry_type,
            "reasoning": self.reasoning,
        }


@dataclass
class Decision:
    symbol: str
    timestamp: int
    price: float
    action: str                    # LONG | SHORT | STAND_ASIDE
    net_score: float
    agreement: float
    conviction: float
    plan: Optional[TradePlan]
    signals: List[AgentSignal]
    vetoes: List[Dict[str, str]]
    narrative: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": round(self.price, 2),
            "action": self.action,
            "net_score": round(self.net_score, 4),
            "agreement": round(self.agreement, 4),
            "conviction": round(self.conviction, 4),
            "plan": self.plan.as_dict() if self.plan else None,
            "narrative": self.narrative,
            "vetoes": self.vetoes,
            "agents": [s.as_dict() for s in self.signals],
        }


class AgentFarm:
    """Runs every agent, combines the directional votes, and writes the plan.

    Aggregation rules, in order:
      1. Analysts vote. Each contributes score x confidence x weight.
      2. The net vote is normalised by total analyst weight, so adding an agent
         cannot inflate conviction by itself.
      3. Gate and risk agents multiply the result; any veto ends it.
      4. A plan is only produced if conviction, agreement and reward-to-risk
         all clear their thresholds.
    """

    def __init__(self, agents: Optional[List[BaseAgent]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.agents = agents if agents is not None else default_agents()
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    # -- introspection -----------------------------------------------------
    def roster(self) -> List[Dict[str, Any]]:
        return [a.info() for a in self.agents]

    # -- main entry point --------------------------------------------------
    def evaluate(self, ctx: MarketContext) -> Decision:
        ctx.config = {**self.config, **(ctx.config or {})}
        signals = [agent.run(ctx) for agent in self.agents]

        analysts = [s for s in signals if s.role == "analyst"]
        gates = [s for s in signals if s.role in ("gate", "risk")]

        total_weight = sum(a.weight for a in analysts) or 1.0
        net = sum(a.contribution for a in analysts) / total_weight

        # Agreement: of the analysts that actually expressed a view, how much
        # weight backs the majority side? A 0.6 net score from one loud agent
        # is not the same as the same score from six.
        directional = [a for a in analysts if a.direction != NEUTRAL and a.confidence > 0]
        voted_weight = sum(a.weight * a.confidence for a in directional)
        side = LONG if net > 0 else SHORT
        agree_weight = sum(
            a.weight * a.confidence for a in directional if a.direction == side
        )
        agreement = (agree_weight / voted_weight) if voted_weight > 0 else 0.0

        multiplier = 1.0
        for g in gates:
            multiplier *= float(g.multiplier)
        conviction = net * multiplier

        vetoes = [
            {"agent": g.agent, "reason": g.veto_reason}
            for g in signals if g.veto
        ]

        decision = Decision(
            symbol=ctx.symbol, timestamp=ctx.now_ts, price=ctx.price,
            action="STAND_ASIDE", net_score=net, agreement=agreement,
            conviction=conviction, plan=None, signals=signals, vetoes=vetoes,
            narrative="",
        )

        if vetoes:
            decision.narrative = "Standing aside: " + "; ".join(v["reason"] for v in vetoes) + "."
            return decision

        threshold = float(self.config["entry_threshold"])
        min_agreement = float(self.config["min_agreement"])

        if abs(conviction) < threshold:
            decision.narrative = (
                f"No trade: conviction {conviction:+.3f} is inside the "
                f"+/-{threshold:.2f} band. {self._top_voices(analysts)}"
            )
            return decision

        if agreement < min_agreement:
            decision.narrative = (
                f"No trade: the farm is split ({agreement:.0%} of voting weight "
                f"agrees, {min_agreement:.0%} required). {self._top_voices(analysts)}"
            )
            return decision

        direction = LONG if conviction > 0 else SHORT
        plan = self._build_plan(ctx, direction, signals)
        if plan is None:
            decision.narrative = (
                f"{direction} conviction {conviction:+.3f} but no plan clears "
                f"the {self.config['min_rr']}:1 reward-to-risk floor."
            )
            return decision

        decision.action = direction
        decision.plan = plan
        decision.narrative = self._narrative(direction, conviction, agreement, plan, analysts)
        return decision

    # -- plan construction -------------------------------------------------
    def _build_plan(self, ctx: MarketContext, direction: str,
                    signals: List[AgentSignal]) -> Optional[TradePlan]:
        price = ctx.price
        atr_val = max(ctx.atr(ctx.ltf), 1e-9)
        levels = self._collect_levels(signals)
        reasoning: List[str] = []

        entry, entry_type = price, "market"
        # Prefer entering from a named zone rather than chasing price.
        zone_mid = self._entry_zone(direction, levels, price, atr_val)
        if zone_mid is not None:
            entry, entry_type = zone_mid, "limit_at_zone"
            reasoning.append(f"Entry staged at the {entry:.2f} confluence zone rather than at market.")

        stop = self._stop_level(direction, levels, entry, atr_val)
        if stop is None:
            return None
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return None

        max_stop = float(self.config["max_stop_atr"]) * atr_val
        if stop_distance > max_stop:
            return None
        reasoning.append(
            f"Stop {stop:.2f} sits beyond structural invalidation "
            f"({stop_distance / atr_val:.1f} ATR away)."
        )

        target, secondary = self._targets(direction, levels, entry, stop, atr_val)
        rr = abs(target - entry) / stop_distance
        if rr < float(self.config["min_rr"]):
            return None
        reasoning.append(f"First target {target:.2f} pays {rr:.1f} to 1.")

        sizing = position_size(
            equity=ctx.equity,
            risk_pct=float(ctx.risk_per_trade),
            entry=entry, stop=stop,
            contract_value=float(self.config.get("contract_value", 1.0)),
        )
        return TradePlan(
            direction=direction, entry=entry, stop=stop, take_profit=target,
            secondary_target=secondary, risk_reward=rr,
            units=sizing["units"], risk_currency=sizing["risk_currency"],
            stop_distance=stop_distance, entry_type=entry_type, reasoning=reasoning,
        )

    @staticmethod
    def _collect_levels(signals: List[AgentSignal]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in signals:
            for k, v in (s.levels or {}).items():
                out.setdefault(f"{s.agent}.{k}", float(v))
                out.setdefault(k, float(v))
        return out

    def _entry_zone(self, direction: str, levels: Dict[str, float],
                    price: float, atr_val: float) -> Optional[float]:
        keys = ("ob_mid", "fvg_mid")
        candidates = [levels[k] for k in keys if k in levels]
        if not candidates:
            return None
        # Only stage an entry that is a retracement, and close enough to fill.
        if direction == LONG:
            valid = [c for c in candidates if c < price and (price - c) <= 1.5 * atr_val]
            return max(valid) if valid else None
        valid = [c for c in candidates if c > price and (c - price) <= 1.5 * atr_val]
        return min(valid) if valid else None

    def _stop_level(self, direction: str, levels: Dict[str, float],
                    entry: float, atr_val: float) -> Optional[float]:
        """Beyond structural invalidation, but never inside the noise band.

        The anchor is the level that proves the idea wrong: the swept extreme,
        the far side of the order block, the protected swing. A buffer is added
        past it, and the result is then pushed out to `min_stop_atr` if the
        anchor sits so close to entry that ordinary bar noise would take it
        out before the idea had a chance to be wrong.
        """
        buffer = float(self.config["stop_buffer_atr"]) * atr_val
        floor = float(self.config.get("min_stop_atr", 0.0)) * atr_val
        if direction == LONG:
            anchors = [
                levels.get("sweep_extreme") if levels.get("sweep_extreme", entry + 1) < entry else None,
                levels.get("ob_bottom"),
                levels.get("fvg_bottom"),
                levels.get("protected_low"),
                levels.get("range_low"),
            ]
            valid = [a for a in anchors if a is not None and a < entry]
            anchor = max(valid) if valid else entry - 1.5 * atr_val
            return min(anchor - buffer, entry - floor)
        anchors = [
            levels.get("sweep_extreme") if levels.get("sweep_extreme", entry - 1) > entry else None,
            levels.get("ob_top"),
            levels.get("fvg_top"),
            levels.get("protected_high"),
            levels.get("range_high"),
        ]
        valid = [a for a in anchors if a is not None and a > entry]
        anchor = min(valid) if valid else entry + 1.5 * atr_val
        return max(anchor + buffer, entry + floor)

    def _targets(self, direction: str, levels: Dict[str, float], entry: float,
                 stop: float, atr_val: float):
        risk = abs(entry - stop)
        target_rr = float(self.config["target_rr"])
        # Targets are liquidity, not arithmetic. The nearest pool on the far
        # side is the first objective; if that objective does not pay the
        # minimum reward-to-risk, the caller rejects the plan rather than
        # stretching the target to a level nothing is drawing price toward.
        if direction == LONG:
            draws = [v for k, v in levels.items()
                     if k in ("draw_up", "range_high", "liquidity_draw.draw_up") and v > entry]
            primary = min(draws) if draws else entry + target_rr * risk
            secondary = max(draws) if draws else entry + target_rr * risk
            return primary, max(secondary, primary)
        draws = [v for k, v in levels.items()
                 if k in ("draw_down", "range_low", "liquidity_draw.draw_down") and v < entry]
        primary = max(draws) if draws else entry - target_rr * risk
        secondary = min(draws) if draws else entry - target_rr * risk
        return primary, min(secondary, primary)

    # -- reporting ---------------------------------------------------------
    @staticmethod
    def _top_voices(analysts: List[AgentSignal], n: int = 3) -> str:
        ranked = sorted(analysts, key=lambda a: abs(a.contribution), reverse=True)[:n]
        parts = [f"{a.agent} {a.direction.lower()} ({a.contribution:+.2f})"
                 for a in ranked if abs(a.contribution) > 0.01]
        return ("Loudest voices: " + ", ".join(parts) + ".") if parts else "No agent has a view."

    def _narrative(self, direction: str, conviction: float, agreement: float,
                   plan: TradePlan, analysts: List[AgentSignal]) -> str:
        supporting = [a for a in analysts if a.direction == direction and a.confidence > 0]
        supporting.sort(key=lambda a: abs(a.contribution), reverse=True)
        reasons = " ".join(a.rationale for a in supporting[:3])
        return (
            f"{direction} {plan.units} units at {plan.entry:.2f}, stop {plan.stop:.2f}, "
            f"target {plan.take_profit:.2f} ({plan.risk_reward:.1f}R). "
            f"Conviction {conviction:+.2f} with {agreement:.0%} of voting weight in agreement. "
            f"{reasons}"
        )
