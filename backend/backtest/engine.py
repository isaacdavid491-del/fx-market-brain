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
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.agents.base import MarketContext
from backend.agents.orchestrator import AgentFarm, Decision
from backend.ict.contracts import Contract, RiskLadder, get_contract
from backend.ict.sessions import ny_day_start, primary_session
from backend.store import TF_MINUTES, resample_ohlcv

log = logging.getLogger("ict.backtest")


@dataclass
class ExitLeg:
    """One partial fill out of a position."""
    ts: int
    price: float
    units: float
    reason: str
    r_at_exit: float          # R on this leg, measured on the initial stop distance

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts, "price": round(self.price, 2), "units": self.units,
            "reason": self.reason, "r_at_exit": round(self.r_at_exit, 3),
        }


@dataclass
class ExitPolicy:
    """How a position is reduced once it is working.

    Chapter 15 distinguishes three reasons to reduce: an objective was
    reached, expected behaviour weakened, or an operational consideration
    changed commitment. Only the first is mechanical, so only the first is
    modelled here; the other two are discretionary and would need a rule the
    source does not supply.

    `scale_outs` is a list of (R multiple, fraction of the initial position).
    Fractions are floored to whole contracts, because a position of one
    contract cannot be scaled out of at all.
    """
    name: str = "all_at_target"
    scale_outs: List[Tuple[float, float]] = field(default_factory=list)
    # Scale-outs expressed as progress toward the objective rather than in R.
    # The opening-gap ladders of chapter 26 are stated this way: a partial at
    # half gap, the bulk at full closure, runners at extensions beyond it.
    # A progress value above 1.0 is an extension past the target.
    scale_outs_by_progress: List[Tuple[float, float]] = field(default_factory=list)
    breakeven_at_r: Optional[float] = None     # move the stop to entry at this R
    # Staged protection tied to progress toward the objective, as
    # (progress fraction, fraction of the ORIGINAL stop distance removed).
    # Chapter 22: a quarter of the way, take a quarter off the distance; half
    # way, half off; three quarters of the way, breakeven. The book contrasts
    # this explicitly with moving every trade to breakeven at one times risk,
    # because it permits some open risk while partials are banked.
    stop_ladder: List[Tuple[float, float]] = field(default_factory=list)
    trail_after_r: Optional[float] = None      # begin trailing at this R
    trail_distance_r: float = 1.0
    source: str = ""                           # which chapter this came from

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scale_outs": [[r, f] for r, f in self.scale_outs],
            "scale_outs_by_progress": [[p, f] for p, f in self.scale_outs_by_progress],
            "stop_ladder": [[p, f] for p, f in self.stop_ladder],
            "breakeven_at_r": self.breakeven_at_r,
            "trail_after_r": self.trail_after_r,
            "trail_distance_r": self.trail_distance_r,
            "source": self.source,
        }


# Named policies for comparison. The book's own management example is a
# three-unit short scaled out in thirds, so that shape is included.
EXIT_POLICIES: Dict[str, ExitPolicy] = {
    "all_at_target": ExitPolicy("all_at_target"),
    "half_at_1R": ExitPolicy("half_at_1R", scale_outs=[(1.0, 0.5)]),
    "half_at_1R_breakeven": ExitPolicy("half_at_1R_breakeven",
                                       scale_outs=[(1.0, 0.5)], breakeven_at_r=1.0),
    "half_at_2R": ExitPolicy("half_at_2R", scale_outs=[(2.0, 0.5)]),
    "thirds_1R_2R": ExitPolicy("thirds_1R_2R",
                               scale_outs=[(1.0, 1 / 3), (2.0, 1 / 3)]),
    "thirds_1R_2R_breakeven": ExitPolicy("thirds_1R_2R_breakeven",
                                         scale_outs=[(1.0, 1 / 3), (2.0, 1 / 3)],
                                         breakeven_at_r=1.0),
    "half_at_1R_trail": ExitPolicy("half_at_1R_trail", scale_outs=[(1.0, 0.5)],
                                   trail_after_r=1.5, trail_distance_r=1.0),

    # --- edition 0.6 -----------------------------------------------------
    # Chapter 22's staged protection, with no scaling of its own, so the
    # effect of the stop ladder can be seen on its own terms.
    "progressive_stop": ExitPolicy(
        "progressive_stop",
        stop_ladder=[(0.25, 0.25), (0.50, 0.50), (0.75, 1.00)],
        source="ch22 Model 6 buy-side trade plan",
    ),
    # The same ladder alongside a partial, which is how the plan describes it:
    # partials are banked while the position still carries some open risk.
    "progressive_stop_half_at_1R": ExitPolicy(
        "progressive_stop_half_at_1R",
        scale_outs=[(1.0, 0.5)],
        stop_ladder=[(0.25, 0.25), (0.50, 0.50), (0.75, 1.00)],
        source="ch22 combined with a partial",
    ),
    # Chapter 26 variant A: the great majority off at the half-gap objective.
    "gap_bulk_at_half": ExitPolicy(
        "gap_bulk_at_half",
        scale_outs_by_progress=[(0.5, 0.75)],
        source="ch26 variant A, 75-80% at half gap",
    ),
    # Chapter 26 variant B: a small partial at half gap, the bulk at full
    # closure, and small runners left for the extension objectives. The
    # twelve-unit illustration is two, then seven, then one at each extension.
    "gap_ladder": ExitPolicy(
        "gap_ladder",
        scale_outs_by_progress=[(0.5, 2 / 12), (1.0, 7 / 12),
                                (1.2, 1 / 12), (1.5, 1 / 12), (2.0, 1 / 12)],
        source="ch26 variant B, 2 at half gap, 7 at closure, 3 runners",
    ),
}


@dataclass
class BacktestTrade:
    direction: str
    entry_ts: int
    entry: float
    stop: float
    target: float
    units: float
    initial_units: float = 0.0
    remaining_units: float = 0.0
    initial_stop: float = 0.0
    legs: List[ExitLeg] = field(default_factory=list)
    stop_moves: int = 0
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

    def __post_init__(self) -> None:
        if not self.initial_units:
            self.initial_units = self.units
        if not self.remaining_units:
            self.remaining_units = self.units
        if not self.initial_stop:
            self.initial_stop = self.stop

    @property
    def risk_per_unit(self) -> float:
        """Distance to the *initial* stop.

        Every R figure is measured against this, so moving a stop later never
        rewrites the risk the trade was taken with.
        """
        return abs(self.entry - self.initial_stop) or 1e-9

    def r_at(self, price: float) -> float:
        sign = 1.0 if self.direction == "LONG" else -1.0
        return sign * (float(price) - self.entry) / self.risk_per_unit

    def as_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry_ts": self.entry_ts,
            "entry": round(self.entry, 2),
            "stop": round(self.stop, 2),
            "initial_stop": round(self.initial_stop, 2),
            "target": round(self.target, 2),
            "units": self.units,
            "initial_units": self.initial_units,
            "legs": [leg.as_dict() for leg in self.legs],
            "stop_moves": self.stop_moves,
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
                 bars: int = 250,
                 exit_policy: Optional[ExitPolicy] = None,
                 risk_ladder: Optional[RiskLadder] = None):
        self.farm = farm or AgentFarm()
        self.starting_equity = float(starting_equity)
        self.risk_per_trade = float(risk_per_trade)
        self.step_minutes = int(step_minutes)
        self.max_hold_minutes = int(max_hold_minutes)
        self.limit_expiry_minutes = int(limit_expiry_minutes)
        self.bars = int(bars)
        self.exit_policy = exit_policy or EXIT_POLICIES["all_at_target"]
        # Chapter 27's dynamic sizing. Off by default so the exit-policy
        # comparison is not confounded by changing position size.
        self.risk_ladder = risk_ladder

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
                    if self.risk_ladder is not None:
                        if open_trade.pnl < 0:
                            self.risk_ladder.on_loss(open_trade.pnl)
                        else:
                            self.risk_ladder.on_win(open_trade.pnl)
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
                equity=equity,
                risk_per_trade=(self.risk_ladder.current_risk
                                if self.risk_ladder is not None else self.risk_per_trade),
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
            self._close_remaining(open_trade, float(c_arr[-1]), int(t_arr[-1]),
                                  "end_of_sample", contract)
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
        """Advance one bar. Returns True when the position is fully closed.

        Order of checks within the bar is deliberate and pessimistic: the stop
        is tested before any favourable level, because a one-minute bar does
        not say which came first and Appendix D forbids defaulting to the
        favourable sequence.
        """
        risk = trade.risk_per_unit
        if trade.direction == "LONG":
            trade.mae_r = min(trade.mae_r, (low - trade.entry) / risk)
            trade.mfe_r = max(trade.mfe_r, (high - trade.entry) / risk)
            stop_hit = low <= trade.stop
            target_hit = high >= trade.target
        else:
            trade.mae_r = min(trade.mae_r, (trade.entry - high) / risk)
            trade.mfe_r = max(trade.mfe_r, (trade.entry - low) / risk)
            stop_hit = high >= trade.stop
            target_hit = low <= trade.target

        if stop_hit:
            self._close_remaining(trade, trade.stop, now,
                                  "stop" if trade.stop_moves == 0 else "stop_moved",
                                  contract)
            return True

        # Scale-outs in R, in order, each only once.
        for level_r, fraction in self.exit_policy.scale_outs:
            if any(leg.reason == f"scale_{level_r:g}R" for leg in trade.legs):
                continue
            level = self._price_at_r(trade, level_r)
            reached = high >= level if trade.direction == "LONG" else low <= level
            if not reached:
                continue
            units = self._scale_units(trade, fraction)
            if units <= 0:
                # A one-contract position cannot be scaled out of. The book's
                # own $50-budget example buys exactly one contract.
                continue
            self._book_leg(trade, level, units, now, f"scale_{level_r:g}R", contract)
            if trade.remaining_units <= 0:
                self._finalise(trade, now, contract)
                return True

        # Scale-outs by progress toward the objective. A progress above 1.0
        # is an extension beyond the target, which is how chapter 26 keeps
        # small runners past full gap closure.
        for progress, fraction in self.exit_policy.scale_outs_by_progress:
            tag = f"scale_{progress:g}x"
            if any(leg.reason == tag for leg in trade.legs):
                continue
            level = self._price_at_progress(trade, progress)
            reached = high >= level if trade.direction == "LONG" else low <= level
            if not reached:
                continue
            units = self._scale_units(trade, fraction)
            if units <= 0:
                continue
            self._book_leg(trade, level, units, now, tag, contract)
            if trade.remaining_units <= 0:
                self._finalise(trade, now, contract)
                return True

        # Staged protection tied to progress toward the objective (chapter 22).
        for progress, reduction in self.exit_policy.stop_ladder:
            level = self._price_at_progress(trade, progress)
            reached = high >= level if trade.direction == "LONG" else low <= level
            if not reached:
                continue
            # The reduction is a fraction of the ORIGINAL entry-to-stop
            # distance, not of whatever the stop has since become.
            new_stop = self._stop_after_reduction(trade, reduction)
            if trade.direction == "LONG" and new_stop > trade.stop:
                trade.stop = new_stop
                trade.stop_moves += 1
            elif trade.direction == "SHORT" and new_stop < trade.stop:
                trade.stop = new_stop
                trade.stop_moves += 1

        # Protective stop moves, applied after any scale-out on the same bar.
        current_r = trade.mfe_r
        if (self.exit_policy.breakeven_at_r is not None
                and trade.stop_moves == 0
                and current_r >= self.exit_policy.breakeven_at_r):
            # Moving the stop to entry can cause an early exit in an expected
            # supportive retracement; that cost is exactly what the comparison
            # is meant to expose.
            trade.stop = trade.entry
            trade.stop_moves += 1

        if self.exit_policy.trail_after_r is not None and current_r >= self.exit_policy.trail_after_r:
            trail_r = current_r - self.exit_policy.trail_distance_r
            trailed = self._price_at_r(trade, trail_r)
            if trade.direction == "LONG" and trailed > trade.stop:
                trade.stop = trailed
                trade.stop_moves += 1
            elif trade.direction == "SHORT" and trailed < trade.stop:
                trade.stop = trailed
                trade.stop_moves += 1

        if target_hit and not self._runs_past_target():
            self._close_remaining(trade, trade.target, now, "target", contract)
            return True

        if now - trade.entry_ts >= self.max_hold_minutes * 60:
            self._close_remaining(trade, (high + low) / 2.0, now, "time_stop", contract)
            return True
        return False

    @staticmethod
    def _price_at_r(trade: BacktestTrade, r: float) -> float:
        sign = 1.0 if trade.direction == "LONG" else -1.0
        return trade.entry + sign * r * trade.risk_per_unit

    def _runs_past_target(self) -> bool:
        """Whether this policy deliberately keeps runners beyond the objective.

        Chapter 26's second variant takes the bulk at full closure and leaves
        small runners for the extension objectives, so reaching the target
        must not flatten the position. The source is clear about the cost:
        keeping the final runner can surrender profit against a perfect target
        exit, and a stopped runner keeps its actual result.
        """
        return any(progress > 1.0 for progress, _ in self.exit_policy.scale_outs_by_progress)

    @staticmethod
    def _price_at_progress(trade: BacktestTrade, progress: float) -> float:
        """Price at a fraction of the way from entry to the objective."""
        return trade.entry + float(progress) * (trade.target - trade.entry)

    @staticmethod
    def _stop_after_reduction(trade: BacktestTrade, reduction: float) -> float:
        """Protection after removing `reduction` of the ORIGINAL stop distance.

        Chapter 22's arithmetic: entry 100, stop 80, target 180. A quarter of
        the way (120) removes a quarter of the 20-unit distance, moving the
        stop to 85. Half way (140) moves it to 90. Three quarters (160) puts
        it at entry.
        """
        sign = 1.0 if trade.direction == "LONG" else -1.0
        remaining = trade.risk_per_unit * (1.0 - float(reduction))
        return trade.entry - sign * remaining

    @staticmethod
    def _scale_units(trade: BacktestTrade, fraction: float) -> float:
        """Whole contracts to release, never more than remain.

        Floored, because a fraction of a contract cannot be traded, and capped
        so an exit order can never exceed the remaining quantity. Chapter 15
        warns that an uncancelled order sized for an earlier quantity can
        create an extra exit or reverse the position.
        """
        wanted = math.floor(trade.initial_units * fraction)
        if wanted <= 0:
            return 0.0
        # Never close the whole position on a scale-out; a runner must survive.
        return float(min(wanted, max(trade.remaining_units - 1, 0)))

    def _book_leg(self, trade: BacktestTrade, price: float, units: float,
                  ts: int, reason: str, contract: Contract) -> None:
        units = min(units, trade.remaining_units)
        if units <= 0:
            return
        trade.legs.append(ExitLeg(ts=int(ts), price=float(price), units=float(units),
                                  reason=reason, r_at_exit=trade.r_at(price)))
        trade.remaining_units -= units
        trade.gross_pnl += contract.gross_pnl(trade.entry, price, units, trade.direction)
        trade.costs += contract.costs(units)

    def _close_remaining(self, trade: BacktestTrade, price: float, ts: int,
                         reason: str, contract: Contract) -> None:
        self._book_leg(trade, price, trade.remaining_units, ts, reason, contract)
        self._finalise(trade, ts, contract, reason)

    def _finalise(self, trade: BacktestTrade, ts: int, contract: Contract,
                  reason: Optional[str] = None) -> None:
        """Close the books on a position, in R measured on the initial risk.

        This reproduces the book's management arithmetic: three units entered
        at 140 protected at 146 carry 18 point-units of risk, and exits at
        130, 134 and 138 realise 18 point-units, which is 1.00 times initial
        risk rather than the 4.00 a full exit at the far objective would give.
        """
        trade.pnl = trade.gross_pnl - trade.costs
        trade.exit_ts = int(ts)
        trade.exit_reason = reason or (trade.legs[-1].reason if trade.legs else "closed")
        trade.remaining_units = 0.0
        if trade.legs:
            total_units = sum(leg.units for leg in trade.legs)
            trade.exit_price = (
                sum(leg.price * leg.units for leg in trade.legs) / total_units
                if total_units else trade.entry
            )
            # R on the whole position: each leg's R weighted by the share of
            # the initial position it closed.
            trade.r_multiple = sum(
                leg.r_at_exit * (leg.units / trade.initial_units) for leg in trade.legs
            ) if trade.initial_units else 0.0
        trade.units = trade.initial_units


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
