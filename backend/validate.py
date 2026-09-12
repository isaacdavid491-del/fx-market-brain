"""End-to-end validation of the farm against real market history.

Everything measured so far in this repository used a synthetic random walk,
which cannot answer whether the strategy has an edge: ICT describes behaviour
that exists because real participants exist, and a random walk has none of it.
This module runs the same comparison against real bars and, above all, refuses
to let a synthetic result be mistaken for a real one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.agents.orchestrator import AgentFarm
from backend.backtest.engine import EXIT_POLICIES, Backtester
from backend.data.feed import seed_history
from backend.data.providers import get_provider
from backend.store import count_rows, init_db, latest_ts, load_1m

# The policies worth separating: a baseline, banking a partial, and the
# staged-protection ladder that looked best on synthetic data.
DEFAULT_POLICIES = ("all_at_target", "half_at_1R", "progressive_stop_half_at_1R")


@dataclass
class PolicyResult:
    name: str
    trades: int
    win_rate: float
    expectancy: float
    total_r: float
    profit_factor: float
    max_drawdown_pct: float
    net_pnl: float
    per_trade: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        out.pop("per_trade")
        return out


def paired_difference(base: Dict[str, float], other: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Compare two policies on the trades they both took.

    Aggregate totals are not comparable: a different exit changes when a
    position closes, which changes which later signals can be acted on, so the
    two runs trade different sequences.
    """
    keys = sorted(set(base) & set(other))
    if len(keys) < 5:
        return None
    x = np.array([base[k] for k in keys], dtype=float)
    y = np.array([other[k] for k in keys], dtype=float)
    diff = y - x
    se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return {
        "matched_trades": len(keys),
        "base_mean_r": round(float(x.mean()), 3),
        "policy_mean_r": round(float(y.mean()), 3),
        "difference_r": round(float(diff.mean()), 3),
        "standard_error": round(se, 3),
        "t": round(float(diff.mean() / se), 2) if se else 0.0,
        "std_change_pct": round(float(y.std(ddof=1) / x.std(ddof=1) - 1) * 100, 1)
        if x.std(ddof=1) else 0.0,
    }


def power_note(trades: int) -> str:
    """How much to trust a sample this size.

    Calibrated on a measured fact rather than a rule of thumb: the identical
    paired comparison on two samples of roughly 57 trades returned t values of
    0.00 and 2.78.
    """
    if trades < 40:
        return ("far too few trades to conclude anything; expect this to swing "
                "wildly between samples")
    if trades < 150:
        return ("too few trades to separate a real effect from noise; the same "
                "comparison on two ~57-trade samples gave t of 0.00 and 2.78")
    if trades < 400:
        return "enough to see a large effect, not enough to trust a small one"
    return "a usable sample, though still one market regime"


def run(symbol: Optional[str] = None, days: int = 60, step_minutes: int = 5,
        policies: Tuple[str, ...] = DEFAULT_POLICIES,
        equity: float = 100_000.0, contract: str = "MNQ",
        require_killzone: bool = True,
        correlated_symbol: Optional[str] = None,
        do_seed: bool = True) -> Dict[str, Any]:
    from backend.service import get_service

    # A fresh checkout has no database yet, and validate is the first command
    # a new user runs.
    init_db()

    service = get_service()
    symbol = symbol or service.symbol
    correlated_symbol = correlated_symbol or service.correlated_symbol
    provider = get_provider()
    is_real = provider.name != "synthetic"

    report: Dict[str, Any] = {
        "symbol": symbol,
        "provider": provider.name,
        # `status` is decided at the end, from what actually arrived. Setting a
        # token is not evidence of data: an earlier version of this function
        # derived "real" from the provider name alone and cheerfully printed
        # REAL MARKET DATA over zero bars, which is the one mistake the whole
        # command exists to prevent.
        "status": "unknown",
        "data_is_real": False,
        "days_requested": days,
        "policies": {},
        "comparisons": {},
        "warnings": [],
    }
    if not is_real:
        report["warnings"].append(
            "SYNTHETIC DATA: this is a random walk and cannot tell you whether "
            "the strategy has an edge. Set OANDA_TOKEN for a real answer."
        )

    if do_seed:
        for sym in filter(None, (symbol, correlated_symbol)):
            try:
                seed_history(sym, days=days, provider=provider)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                report["warnings"].append(f"seeding {sym} failed: {exc}")

    end = int(time.time())
    start = end - days * 86400
    df = load_1m(symbol, start, end)
    report["bars_available"] = len(df)
    report["bars_stored_total"] = count_rows(symbol)
    report["latest_ts"] = latest_ts(symbol)

    if len(df) < 5000:
        report["status"] = "no_data"
        report["warnings"].append(
            f"only {len(df)} one-minute bars available; seed more history before "
            "drawing any conclusion"
        )
        return report

    # Real bars actually reached the backtester, so the result means something.
    report["status"] = "real" if is_real else "synthetic"
    report["data_is_real"] = is_real

    peer = None
    if correlated_symbol:
        peer_df = load_1m(correlated_symbol, start, end)
        peer = peer_df if len(peer_df) > 1000 else None
    if peer is None:
        report["warnings"].append(
            f"no correlated series for {correlated_symbol}; the SMT agent will abstain"
        )

    cfg = {"require_killzone": require_killzone, "contract": contract}
    warmup = min(max(len(df) // 4, 2000), max(len(df) - 500, 1))

    results: Dict[str, PolicyResult] = {}
    for name in policies:
        policy = EXIT_POLICIES.get(name)
        if policy is None:
            report["warnings"].append(f"unknown policy {name}")
            continue
        bt = Backtester(AgentFarm(config=cfg), starting_equity=equity,
                        step_minutes=step_minutes, bars=200, exit_policy=policy)
        res = bt.run(symbol, df, correlated_1m=peer,
                     correlated_symbol=correlated_symbol if peer is not None else None,
                     warmup_bars=warmup, config=cfg)
        m = res.metrics
        results[name] = PolicyResult(
            name=name, trades=m["trades"], win_rate=m["win_rate"],
            expectancy=m["expectancy_r"], total_r=m["total_r"],
            profit_factor=m["profit_factor"], max_drawdown_pct=m["max_drawdown_pct"],
            net_pnl=m["net_pnl"],
            per_trade={str(t.entry_ts): t.r_multiple for t in res.trades},
        )
        report["policies"][name] = results[name].as_dict()
        report["decisions_evaluated"] = res.decisions_evaluated

    baseline = policies[0]
    if baseline in results:
        report["power"] = power_note(results[baseline].trades)
        for name in policies[1:]:
            if name in results:
                report["comparisons"][f"{name}_vs_{baseline}"] = paired_difference(
                    results[baseline].per_trade, results[name].per_trade
                )
    return report


BANNERS = {
    "real": "REAL MARKET DATA",
    "synthetic": "SYNTHETIC DATA - NOT A TEST OF THE STRATEGY",
    "no_data": "NO USABLE DATA - NOTHING WAS TESTED",
    "unknown": "NO USABLE DATA - NOTHING WAS TESTED",
}


def is_valid_run(report: Dict[str, Any]) -> bool:
    """True only when real bars were actually replayed."""
    return report.get("status") == "real"


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    status = report.get("status", "unknown")
    banner = BANNERS.get(status, BANNERS["unknown"])
    lines.append("=" * 78)
    lines.append(f"  {banner}")
    lines.append("=" * 78)
    lines.append(f"instrument {report['symbol']}   provider {report['provider']}   "
                 f"bars {report.get('bars_available', 0)}")
    if report.get("decisions_evaluated"):
        lines.append(f"decision points evaluated: {report['decisions_evaluated']}")
    lines.append("")

    if report.get("policies"):
        lines.append(f"{'policy':<30}{'trades':>7}{'win%':>7}{'expect':>9}"
                     f"{'totalR':>8}{'PF':>6}{'maxDD%':>8}{'net$':>10}")
        lines.append("-" * 85)
        for name, p in report["policies"].items():
            lines.append(f"{name:<30}{p['trades']:>7}{p['win_rate']*100:>7.0f}"
                         f"{p['expectancy']:>+9.3f}{p['total_r']:>+8.2f}"
                         f"{p['profit_factor']:>6.2f}{p['max_drawdown_pct']:>8.2f}"
                         f"{p['net_pnl']:>+10.0f}")
        lines.append("")

    if report.get("comparisons"):
        lines.append("Paired on matched trades (aggregate totals are not comparable):")
        for label, c in report["comparisons"].items():
            if c is None:
                lines.append(f"  {label}: too few matched trades")
                continue
            lines.append(
                f"  {label}: {c['matched_trades']} trades, "
                f"{c['base_mean_r']:+.3f} -> {c['policy_mean_r']:+.3f} R, "
                f"difference {c['difference_r']:+.3f} (se {c['standard_error']:.3f}, "
                f"t {c['t']:+.2f}), spread {c['std_change_pct']:+.1f}%"
            )
        lines.append("")

    if report.get("power"):
        lines.append(f"Sample size: {report['power']}.")
    for w in report.get("warnings", []):
        lines.append(f"  ! {w}")
    lines.append("")
    lines.append("Costs and tick rounding are included. Slippage beyond the modelled")
    lines.append("allowance, gap risk and queue position are not.")
    return "\n".join(lines)
