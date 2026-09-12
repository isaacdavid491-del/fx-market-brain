"""Contract specifications, tick rounding and trading costs.

Appendix C of the study book is blunt about this: a gross chart ratio is not
the ratio you trade. Its worked example has 11.25 points of risk and 22.50 of
reward, a gross 2.00, which becomes roughly 1.65 net once a modest fee and
adverse-execution allowance are included. Break-even probability moves with it.

So the farm prices every plan in an actual contract, rounds every level to a
real tradable increment, and reports gross and net separately.

Specifications are from CME product pages as cited by the book. Re-check them
against the exchange before trading; they change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Contract:
    symbol: str
    name: str
    dollars_per_point: float
    tick_size: float
    # Round-trip commission plus an allowance for adverse execution, per unit.
    round_trip_cost: float

    @property
    def dollars_per_tick(self) -> float:
        return self.dollars_per_point * self.tick_size

    def round_to_tick(self, price: float) -> float:
        """Snap a price to a tradable increment.

        A level that cannot be quoted cannot be an order, and a backtest that
        fills at unquotable prices flatters itself by a fraction of a tick on
        every trade.
        """
        if self.tick_size <= 0:
            return float(price)
        ticks = round(float(price) / self.tick_size)
        return round(ticks * self.tick_size, 10)

    def round_away(self, price: float, direction: int) -> float:
        """Round to a tick in the conservative direction.

        `direction` is +1 to round up, -1 to round down. Used for protective
        stops so rounding never quietly tightens the stop.
        """
        if self.tick_size <= 0:
            return float(price)
        fn = math.ceil if direction > 0 else math.floor
        return round(fn(float(price) / self.tick_size) * self.tick_size, 10)

    def offset_ticks(self, price: float, ticks: int) -> float:
        """A price offset by whole ticks, as the book's worked examples do:
        protection one increment beyond the raid low, target one increment
        before the reference."""
        return self.round_to_tick(price + ticks * self.tick_size)

    def pnl(self, entry: float, exit_price: float, units: float, direction: str) -> float:
        sign = 1.0 if direction == "LONG" else -1.0
        gross = sign * (exit_price - entry) * units * self.dollars_per_point
        return gross - self.costs(units)

    def gross_pnl(self, entry: float, exit_price: float, units: float, direction: str) -> float:
        sign = 1.0 if direction == "LONG" else -1.0
        return sign * (exit_price - entry) * units * self.dollars_per_point

    def costs(self, units: float) -> float:
        return abs(units) * self.round_trip_cost

    def risk_dollars(self, entry: float, stop: float, units: float) -> float:
        return abs(entry - stop) * units * self.dollars_per_point


# CME E-mini and Micro E-mini Nasdaq-100, per the specifications the book cites.
NQ = Contract("NQ", "E-mini Nasdaq-100", dollars_per_point=20.0,
              tick_size=0.25, round_trip_cost=5.00)
MNQ = Contract("MNQ", "Micro E-mini Nasdaq-100", dollars_per_point=2.0,
               tick_size=0.25, round_trip_cost=3.00)

# The OANDA CFD the live feed provides. One unit is one index point of
# exposure, and the spread stands in for commission.
NAS100_CFD = Contract("NAS100_USD", "Nasdaq 100 CFD", dollars_per_point=1.0,
                      tick_size=0.1, round_trip_cost=0.0)

CONTRACTS: Dict[str, Contract] = {c.symbol: c for c in (NQ, MNQ, NAS100_CFD)}


def get_contract(name: Optional[str]) -> Contract:
    """Resolve a contract by symbol, defaulting to the micro.

    The micro is the default deliberately: it is the size at which the book's
    own arithmetic exercise permits one unit on a $50 risk budget.
    """
    if not name:
        return MNQ
    return CONTRACTS.get(str(name).upper(), MNQ)


def size_position(contract: Contract, risk_budget: float, entry: float, stop: float,
                  include_costs: bool = True) -> Dict[str, float]:
    """Whole units affordable within a risk budget, costs included.

    The book's example is exact: 11.25 points on MNQ is $22.50, plus $3 of
    costs is $25.50, so a $50 budget buys one contract and not two, because
    two would be $51. Fractional contracts do not exist, so this floors.
    """
    stop_distance = abs(float(entry) - float(stop))
    out = {"units": 0.0, "stop_distance": round(stop_distance, 4),
           "risk_per_unit": 0.0, "risk_total": 0.0, "budget": round(risk_budget, 2)}
    if stop_distance <= 0 or risk_budget <= 0:
        return out

    per_unit = stop_distance * contract.dollars_per_point
    if include_costs:
        per_unit += contract.round_trip_cost
    out["risk_per_unit"] = round(per_unit, 2)
    if per_unit <= 0:
        return out

    units = math.floor(risk_budget / per_unit)
    out["units"] = float(max(units, 0))
    out["risk_total"] = round(out["units"] * per_unit, 2)
    return out


def net_reward_to_risk(contract: Contract, entry: float, stop: float, target: float,
                       units: float = 1.0) -> Dict[str, float]:
    """Gross and net reward-to-risk, and the break-even hit rate.

    Reproduces Appendix C: gross 2.00 becomes net 1.65, and the break-even
    probability is risk / (reward + risk) = 37.78%.
    """
    risk_points = abs(entry - stop)
    reward_points = abs(target - entry)
    if risk_points <= 0:
        return {"gross_rr": 0.0, "net_rr": 0.0, "break_even_rate": 1.0,
                "net_win": 0.0, "net_loss": 0.0}

    costs = contract.costs(units)
    gross_win = reward_points * units * contract.dollars_per_point
    gross_loss = risk_points * units * contract.dollars_per_point
    net_win = gross_win - costs
    net_loss = gross_loss + costs

    net_rr = (net_win / net_loss) if net_loss > 0 else 0.0
    break_even = (net_loss / (net_win + net_loss)) if (net_win + net_loss) > 0 else 1.0
    return {
        "gross_rr": round(reward_points / risk_points, 3),
        "net_rr": round(net_rr, 3),
        "break_even_rate": round(break_even, 4),
        "net_win": round(net_win, 2),
        "net_loss": round(net_loss, 2),
    }


@dataclass
class RiskLadder:
    """Dynamic risk sizing after losses and winning streaks.

    Chapter 27. The older trade plans halve risk after a full planned loss and
    permit restoration once half that loss is recovered, halving again after a
    further loss. Model 13's illustrative progression runs 2%, 1%, 0.5%,
    0.25%, holding at the smallest level until the recovery condition is met.
    They also prescribe halving after five consecutive wins.

    The book is explicit that these examples do not authorise halving forever
    or restoring full risk automatically after any winning trade, so the floor
    and the restoration condition are both set explicitly here rather than
    left implied.
    """
    base_risk: float = 0.02
    floor_risk: float = 0.0025
    halve_after_wins: int = 5
    current_risk: float = 0.0
    consecutive_wins: int = 0
    drawdown_to_recover: float = 0.0

    def __post_init__(self) -> None:
        if not self.current_risk:
            self.current_risk = self.base_risk

    def on_loss(self, loss_amount: float) -> None:
        """Halve risk, and remember what must be recovered to restore it."""
        self.consecutive_wins = 0
        self.drawdown_to_recover += abs(float(loss_amount))
        self.current_risk = max(self.current_risk / 2.0, self.floor_risk)

    def on_win(self, win_amount: float) -> None:
        """Recovering half the outstanding loss restores one step of risk.

        A winning streak reduces risk rather than increasing it; that is a
        sizing policy in the source, not a claim that a streak makes the next
        trade more likely to lose.
        """
        self.consecutive_wins += 1
        if self.drawdown_to_recover > 0:
            self.drawdown_to_recover -= abs(float(win_amount))
            if self.drawdown_to_recover <= 0:
                self.drawdown_to_recover = 0.0
                self.current_risk = min(self.current_risk * 2.0, self.base_risk)
        if self.halve_after_wins and self.consecutive_wins >= self.halve_after_wins:
            self.current_risk = max(self.current_risk / 2.0, self.floor_risk)
            self.consecutive_wins = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "base_risk": self.base_risk,
            "current_risk": round(self.current_risk, 5),
            "floor_risk": self.floor_risk,
            "consecutive_wins": self.consecutive_wins,
            "drawdown_to_recover": round(self.drawdown_to_recover, 2),
        }
