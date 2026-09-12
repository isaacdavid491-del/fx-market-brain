"""Walk-forward backtester for the agent farm.

The farm evaluates on exactly the frames it would have had at that moment,
built by truncating history rather than by any separate historical code path.
Fills are simulated on 1-minute bars, so a stop and a target inside the same
5-minute decision bar resolve in a defined order instead of being guessed at.

Where the simulation is optimistic, it is documented. Nothing here models
slippage on gaps, overnight financing, or the fact that a real limit order can
be queued behind size at the same price.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.agents.base import MarketContext
from backend.agents.orchestrator import AgentFarm, Decision
from backend.ict.contracts import Contract, get_contract
from backend.ict.sessions import ny_day_start, primary_session
from backend.store import TF_MINUTES, resample_ohlcv

log = logging.getLogger("ict.backtest")


@dataclass
class BacktestTrade:
    direction: str
    entry_ts: int
    entry: float
    stop: float
    target: float
    units: float
    exit_ts: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = "open"
    r_multiple: float = 0.0
    pnl: float = 0.0          # net of costs
    gross_pnl: float = 0.0
    costs: float = 0.0
    entry_family: str = "confirmed"
    mae_r: float = 0.0        # worst excursion against the trade, in R
    mfe_r: float = 0.0        # best excursion in favour, in R
    session: Optional[str] = None
    narrative: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry_ts": self.entry_ts,
            "entry": round(self.entry, 2),
            "stop": round(self.stop, 2),
            "target": round(self.target, 2),
            "units": self.units,
            "exit_ts": self.exit_ts,
            "exit_price": round(self.exit_price, 2) if self.exit_price is not None else None,
            "exit_reason": self.exit_reason,
            "r_multiple": round(self.r_multiple, 3),
            "pnl": round(self.pnl, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "costs": round(self.costs, 2),
            "entry_family": self.entry_family,
            "mae_r": round(self.mae_r, 2),
            "mfe_r": round(self.mfe_r, 2),
            "session": self.session,
            "narrative": self.narrative,
        }


@dataclass
class BacktestResult:
    symbol: str
    start_ts: int
    end_ts: int
    starting_equity: float
    ending_equity: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    decisions_evaluated: int = 0
    signals_generated: int = 0
    # The study insists unfilled orders and rejected candidates are part of
    # the sample: keeping only the fills that happened flatters the record.
    orders_cancelled: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self, include_trades: bool = True) -> Dict[str, Any]:
        out = {
            "symbol": self.symbol,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "decisions_evaluated": self.decisions_evaluated,
            "signals_generated": self.signals_generated,
            "orders_cancelled": self.orders_cancelled,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "equity_curve": [[int(t), round(float(e), 2)] for t, e in self.equity_curve],
        }
        if include_trades:
            out["trades"] = [t.as_dict() for t in self.trades]
        return out


class Backtester:
    """Replays the farm bar by bar over stored history."""

    def __init__(self, farm: Optional[AgentFarm] = None,
                 starting_equity: float = 100_000.0,
                 risk_per_trade: float = 0.005,
                 step_minutes: int = 5,
                 max_hold_minutes: int = 240,
                 limit_expiry_minutes: int = 30,
                 bars: int = 250):
        self.farm = farm or AgentFarm()
        self.starting_equity = float(starting_equity)
        self.risk_per_trade = float(risk_per_trade)
        self.step_minutes = int(step_minutes)
        self.max_hold_minutes = int(max_hold_minutes)
        self.limit_expiry_minutes = int(limit_expiry_minutes)
        self.bars = int(bars)

    # -- frame preparation -------------------------------------------------
    def _prepare(self, df_1m: pd.DataFrame, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Resample once for the whole history; slices are taken per decision.

        Resampling the full series and then truncating is equivalent to
        resampling a truncated series, provided only closed bars are used,
        which `_slice` enforces.
        """
        return {tf: resample_ohlcv(df_1m, tf) for tf in timeframes}

    def _slice(self, full: Dict[str, pd.DataFrame], now_ts: int) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for tf, frame in full.items():
            if frame.empty:
                out[tf] = frame
                continue
            bar_seconds = TF_MINUTES.get(tf, 1) * 60
            closed = frame[frame["t"] + bar_seconds <= now_ts]
            out[tf] = closed.tail(self.bars).reset_index(drop=True)
        return out

    # -- main loop ---------------------------------------------------------
    def run(self, symbol: str, df_1m: pd.DataFrame,
            correlated_1m: Optional[pd.DataFrame] = None,
            correlated_symbol: Optional[str] = None,
            warmup_bars: int = 3000,
            config: Optional[Dict[str, Any]] = None,
            progress_every: int = 0) -> BacktestResult:
        cfg = {**self.farm.config, **(config or {})}
        timeframes = list(dict.fromkeys([cfg.get("ltf", "5m"), cfg.get("mtf", "15m"), cfg.get("htf", "1h")]))

        df_1m = df_1m.sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
        result = BacktestResult(
            symbol=symbol,
            start_ts=int(df_1m["t"].iloc[0]) if len(df_1m) else 0,
            end_ts=int(df_1m["t"].iloc[-1]) if len(df_1m) else 0,
            starting_equity=self.starting_equity,
            ending_equity=self.starting_equity,
        )
        if len(df_1m) <= warmup_bars + 10:
            result.warnings.append(
                f"only {len(df_1m)} 1m bars supplied, need more than the "
                f"{warmup_bars}-bar warmup; nothing was simulated"
            )
            # Keep the metrics shape identical on every path so callers never
            # have to special-case an aborted run.
            result.metrics = compute_metrics([], self.starting_equity,
                                             self.starting_equity, [])
            return result

        full = self._prepare(df_1m, timeframes)
        corr_full = (self._prepare(correlated_1m, timeframes)
                     if correlated_1m is not None and not correlated_1m.empty else {})

        equity = self.starting_equity
        open_trade: Optional[BacktestTrade] = None
        pending: Optional[Dict[str, Any]] = None
        trades_today = 0
        day_start_equity = equity
        current_day = None

        step_seconds = self.step_minutes * 60
        t_arr = df_1m["t"].to_numpy(dtype="int64")
        o_arr = df_1m["o"].to_numpy(dtype=float)
        h_arr = df_1m["h"].to_numpy(dtype=float)
        l_arr = df_1m["l"].to_numpy(dtype=float)
        c_arr = df_1m["c"].to_numpy(dtype=float)

        contract = get_contract(cfg.get("contract"))

        for i in range(warmup_bars, len(df_1m)):
            now = int(t_arr[i])
            day = ny_day_start(now)
            if day != current_day:
                current_day = day
                trades_today = 0
                day_start_equity = equity

            # -- manage an open position on this bar -----------------------
            if open_trade is not None:
                closed = self._manage(open_trade, h_arr[i], l_arr[i], now, contract)
                if closed:
                    equity += open_trade.pnl
                    result.equity_curve.append((now, equity))
                    result.trades.append(open_trade)
                    open_trade = None
                continue

            # -- work a resting limit order --------------------------------
            if pending is not None:
                # Chapter 19 fixes these cancellation rules before results are
                # collected: cancel if the target trades first, if protection
                # is breached before entry, or if the session deadline passes.
                cancel: Optional[str] = None
                if now > pending["expires"]:
                    cancel = "expired"
                elif pending["direction"] == "LONG":
                    if h_arr[i] >= pending["target"]:
                        cancel = "target_traded_first"
                    elif l_arr[i] <= pending["stop"]:
                        cancel = "protection_breached_before_entry"
                else:
                    if l_arr[i] <= pending["target"]:
                        cancel = "target_traded_first"
                    elif h_arr[i] >= pending["stop"]:
                        cancel = "protection_breached_before_entry"

                if cancel:
                    result.orders_cancelled[cancel] = result.orders_cancelled.get(cancel, 0) + 1
                    pending = None
                    continue

                entry = pending["entry"]
                if l_arr[i] <= entry <= h_arr[i]:
                    open_trade = BacktestTrade(
                        direction=pending["direction"], entry_ts=now, entry=entry,
                        stop=pending["stop"], target=pending["target"],
                        units=pending["units"],
                        session=(primary_session(now).name if primary_session(now) else None),
                        narrative=pending["narrative"],
                        entry_family=pending.get("entry_family", "retracement"),
                    )
                    trades_today += 1
                    pending = None
                continue

            # -- decision points -------------------------------------------
            if now % step_seconds != 0:
                continue

            frames = self._slice(full, now)
            if any(len(frames[tf]) < 40 for tf in timeframes):
                continue

            day_pnl_pct = (equity - day_start_equity) / day_start_equity if day_start_equity else 0.0
            ctx = MarketContext(
                symbol=symbol, now_ts=now, frames=frames,
                correlated_symbol=correlated_symbol,
                correlated_frames=self._slice(corr_full, now) if corr_full else {},
                equity=equity, risk_per_trade=self.risk_per_trade,
                config={
                    **cfg,
                    "trades_today": trades_today,
                    "day_pnl_pct": day_pnl_pct,
                    "open_positions": 0,
                    "starting_equity": self.starting_equity,
                    # The live feed's staleness check is meaningless in replay.
                    "max_staleness_seconds": 10 ** 9,
                },
            )
            decision = self.farm.evaluate(ctx)
            result.decisions_evaluated += 1
            if progress_every and result.decisions_evaluated % progress_every == 0:
                log.info("backtest: %s decisions, equity %.0f",
                         result.decisions_evaluated, equity)

            plan = decision.plan
            if plan is None or decision.action not in ("LONG", "SHORT"):
                continue
            result.signals_generated += 1

            if plan.entry_type == "market":
                # Fill at the next bar's open: the decision used this bar's close.
                if i + 1 >= len(df_1m):
                    continue
                fill = float(o_arr[i + 1])
                open_trade = BacktestTrade(
                    direction=plan.direction, entry_ts=int(t_arr[i + 1]), entry=fill,
                    stop=plan.stop, target=plan.take_profit, units=plan.units,
                    session=(primary_session(now).name if primary_session(now) else None),
                    narrative=decision.narrative,
                    entry_family=plan.entry_family,
                )
                trades_today += 1
            else:
                pending = {
                    "direction": plan.direction, "entry": plan.entry, "stop": plan.stop,
                    "target": plan.take_profit, "units": plan.units,
                    "expires": now + self.limit_expiry_minutes * 60,
                    "narrative": decision.narrative,
                    "entry_family": plan.entry_family,
                }

        # Close anything still open at the end of the sample, at the last price.
        if open_trade is not None:
            self._close(open_trade, float(c_arr[-1]), int(t_arr[-1]), "end_of_sample", contract)
            equity += open_trade.pnl
            result.trades.append(open_trade)
            result.equity_curve.append((int(t_arr[-1]), equity))

        result.ending_equity = equity
        result.metrics = compute_metrics(result.trades, self.starting_equity, equity,
                                         result.equity_curve)
        return result

    # -- position management ----------------------------------------------
    def _manage(self, trade: BacktestTrade, high: float, low: float,
                now: int, contract: Contract) -> bool:
        risk = abs(trade.entry - trade.stop) or 1e-9
        if trade.direction == "LONG":
            trade.mae_r = min(trade.mae_r, (low - trade.entry) / risk)
            trade.mfe_r = max(trade.mfe_r, (high - trade.entry) / risk)
            # Stop first when both levels are inside one bar: the pessimistic
            # assumption, since 1m bars do not say which came first.
            if low <= trade.stop:
                self._close(trade, trade.stop, now, "stop", contract)
                return True
            if high >= trade.target:
                self._close(trade, trade.target, now, "target", contract)
                return True
        else:
            trade.mae_r = min(trade.mae_r, (trade.entry - high) / risk)
            trade.mfe_r = max(trade.mfe_r, (trade.entry - low) / risk)
            if high >= trade.stop:
                self._close(trade, trade.stop, now, "stop", contract)
                return True
            if low <= trade.target:
                self._close(trade, trade.target, now, "target", contract)
                return True

        if now - trade.entry_ts >= self.max_hold_minutes * 60:
            price = (high + low) / 2.0
            self._close(trade, price, now, "time_stop", contract)
            return True
        return False

    @staticmethod
    def _close(trade: BacktestTrade, price: float, ts: int, reason: str,
               contract: Contract) -> None:
        """Close the position and book both gross and net results.

        Costs are charged on the round trip. The study's Appendix C makes the
        point plainly: a gross 2.00 reward-to-risk is a net 1.65, and ignoring
        that difference makes every backtest look better than the account
        would have.
        """
        sign = 1.0 if trade.direction == "LONG" else -1.0
        risk = abs(trade.entry - trade.stop) or 1e-9
        trade.exit_price = float(price)
        trade.exit_ts = int(ts)
        trade.exit_reason = reason
        trade.r_multiple = sign * (price - trade.entry) / risk
        trade.gross_pnl = contract.gross_pnl(trade.entry, price, trade.units, trade.direction)
        trade.costs = contract.costs(trade.units)
        trade.pnl = trade.gross_pnl - trade.costs


def compute_metrics(trades: List[BacktestTrade], starting_equity: float,
                    ending_equity: float,
                    equity_curve: List[Tuple[int, float]]) -> Dict[str, Any]:
    """Summary statistics. Returns zeros rather than NaN on an empty sample."""
    closed = [t for t in trades if t.exit_reason != "open"]
    n = len(closed)
    base = {
        "trades": n,
        "gross_pnl": 0.0, "costs_paid": 0.0, "net_pnl": 0.0,
        "by_entry_family": {},
        "wins": 0, "losses": 0, "win_rate": 0.0,
        "total_r": 0.0, "avg_r": 0.0, "expectancy_r": 0.0,
        "profit_factor": 0.0, "max_drawdown_pct": 0.0,
        "return_pct": round((ending_equity / starting_equity - 1) * 100, 3) if starting_equity else 0.0,
        "avg_win_r": 0.0, "avg_loss_r": 0.0,
        "by_exit_reason": {}, "by_session": {}, "by_direction": {},
    }
    if n == 0:
        return base

    rs = np.array([t.r_multiple for t in closed], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs <= 0]

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    peak, max_dd = starting_equity, 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)

    by_reason: Dict[str, int] = {}
    for t in closed:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    by_session: Dict[str, Dict[str, Any]] = {}
    for t in closed:
        key = t.session or "none"
        row = by_session.setdefault(key, {"trades": 0, "total_r": 0.0})
        row["trades"] += 1
        row["total_r"] = round(float(row["total_r"]) + float(t.r_multiple), 3)

    by_direction: Dict[str, Dict[str, Any]] = {}
    for t in closed:
        row = by_direction.setdefault(t.direction, {"trades": 0, "total_r": 0.0})
        row["trades"] += 1
        row["total_r"] = round(float(row["total_r"]) + float(t.r_multiple), 3)

    gross_total = float(sum(t.gross_pnl for t in closed))
    cost_total = float(sum(t.costs for t in closed))
    by_family: Dict[str, Dict[str, Any]] = {}
    for t in closed:
        row = by_family.setdefault(t.entry_family, {"trades": 0, "total_r": 0.0})
        row["trades"] += 1
        row["total_r"] = round(float(row["total_r"]) + float(t.r_multiple), 3)

    base.update({
        # Gross and net are reported side by side because the difference is
        # the whole point of the study's cost arithmetic.
        "gross_pnl": round(gross_total, 2),
        "costs_paid": round(cost_total, 2),
        "net_pnl": round(gross_total - cost_total, 2),
        "by_entry_family": by_family,
        "wins": int(len(wins)), "losses": int(len(losses)),
        "win_rate": round(len(wins) / n, 4),
        "total_r": round(float(rs.sum()), 3),
        "avg_r": round(float(rs.mean()), 3),
        "expectancy_r": round(float(rs.mean()), 3),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else 0.0,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 3),
        "avg_mae_r": round(float(np.mean([t.mae_r for t in closed])), 3),
        "avg_mfe_r": round(float(np.mean([t.mfe_r for t in closed])), 3),
        "by_exit_reason": by_reason,
        "by_session": by_session,
        "by_direction": by_direction,
        # The hit rate this sample's average win and loss would have needed
        # just to break even, which is the honest yardstick for the win rate.
        "break_even_rate": round(
            abs(float(losses.mean())) / (float(wins.mean()) + abs(float(losses.mean()))), 4
        ) if len(wins) and len(losses) else 0.0,
    })
    return base
