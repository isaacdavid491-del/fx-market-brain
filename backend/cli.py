"""Command line entry point for the agent farm.

    python -m backend.cli decide
    python -m backend.cli backtest --days 14
    python -m backend.cli seed --days 30
    python -m backend.cli agents

Analysis only: no command here places an order.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Dict

from backend.service import get_service

GREEN, RED, GREY, BOLD, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[1m", "\033[0m"


def colour(direction: str, text: str, enabled: bool) -> str:
    if not enabled:
        return text
    if direction == "LONG":
        return f"{GREEN}{text}{RESET}"
    if direction == "SHORT":
        return f"{RED}{text}{RESET}"
    return f"{GREY}{text}{RESET}"


def cmd_decide(args: argparse.Namespace) -> int:
    service = get_service()
    decision = service.decide(symbol=args.symbol,
                              require_killzone=None if args.killzone else False,
                              equity=args.equity)
    if args.json:
        print(json.dumps(decision.as_dict(), indent=2))
        return 0

    tty = sys.stdout.isatty()
    out = decision.as_dict()
    print(f"\n{BOLD if tty else ''}{out['symbol']} @ {out['price']}{RESET if tty else ''}")
    print(f"Verdict: {colour(out['action'], out['action'], tty)}  "
          f"(conviction {out['conviction']:+.3f}, agreement {out['agreement']:.0%})")
    print(f"{out['narrative']}\n")

    print(f"{'AGENT':<18} {'VIEW':<8} {'SCORE':>7} {'CONF':>6} {'CONTRIB':>8}  REASONING")
    print("-" * 100)
    for agent in out["agents"]:
        view = "VETO" if agent["veto"] else agent["direction"]
        reason = (agent["veto_reason"] if agent["veto"] else agent["rationale"])[:52]
        print(f"{agent['agent']:<18} {colour(agent['direction'], f'{view:<8}', tty)} "
              f"{agent['score']:>+7.2f} {agent['confidence']:>6.2f} "
              f"{agent['contribution']:>+8.2f}  {reason}")

    if out["plan"]:
        plan = out["plan"]
        print(f"\n{BOLD if tty else ''}PLAN{RESET if tty else ''}  "
              f"{plan['direction']} {plan['units']} units")
        print(f"  entry {plan['entry']} ({plan['entry_type']})   "
              f"stop {plan['stop']}   target {plan['take_profit']}   "
              f"R:R {plan['risk_reward']}")
        print(f"  risking {plan['risk_currency']} over {plan['stop_distance']} points")
        for line in plan["reasoning"]:
            print(f"  - {line}")
    print(f"\n{GREY if tty else ''}Model output, not financial advice. "
          f"No order is sent to a broker.{RESET if tty else ''}\n")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    service = get_service()
    out = service.backtest(symbol=args.symbol, days=args.days,
                           step_minutes=args.step,
                           require_killzone=None if args.killzone else False,
                           include_trades=args.json,
                           exit_policy=args.exit_policy)
    if not out.get("ok"):
        print(f"backtest failed: {out.get('error')}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    metrics = out["metrics"]
    print(f"\n{out['symbol']}  {args.days} days  provider={out['provider']}"
          f"  exits={out.get('exit_policy', {}).get('name', 'n/a')}")
    print(f"decisions {out['decisions_evaluated']}, signals {out['signals_generated']}")
    rows = [
        ("trades", metrics["trades"]),
        ("win rate", f"{metrics['win_rate']:.0%}"),
        ("total R", metrics["total_r"]),
        ("expectancy", f"{metrics['expectancy_r']} R"),
        ("profit factor", metrics["profit_factor"]),
        ("max drawdown", f"{metrics['max_drawdown_pct']}%"),
        ("return", f"{metrics['return_pct']}%"),
    ]
    for key, value in rows:
        print(f"  {key:<16} {value}")
    if metrics.get("by_session"):
        print("  by session:")
        for name, row in metrics["by_session"].items():
            print(f"    {name:<20} {row['trades']:>3} trades  {row['total_r']:>+7.2f} R")
    for warning in out.get("warnings", []):
        print(f"  ! {warning}")
    print()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the real-data validation and print a report."""
    from backend.validate import format_report, is_valid_run, run

    report = run(symbol=args.symbol, days=args.days, step_minutes=args.step,
                 equity=args.equity or 100_000.0, contract=args.contract,
                 require_killzone=not args.ignore_killzone,
                 do_seed=not args.no_seed)
    if args.report_json:
        # Written alongside the human report so a long run never has to be
        # repeated just to change output format.
        pathlib.Path(args.report_json).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    # Only a run that actually replayed real bars counts as a validation, and
    # the exit code says so. Synthetic data or no data both fail.
    return 0 if is_valid_run(report) else 2


def cmd_seed(args: argparse.Namespace) -> int:
    print(json.dumps(get_service().seed(days=args.days, symbol=args.symbol), indent=2))
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    service = get_service()
    for agent in service.farm.roster():
        print(f"{agent['name']:<18} {agent['role']:<9} w={agent['weight']:<5} {agent['description']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="backend.cli",
                                     description="NASDAQ ICT agent farm")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    decide = sub.add_parser("decide", help="run the farm now")
    decide.add_argument("--symbol")
    decide.add_argument("--equity", type=float)
    decide.add_argument("--killzone", action="store_true",
                        help="enforce killzone gating (default: ignore it so a "
                             "decision is always produced)")
    decide.add_argument("--json", action="store_true")
    decide.set_defaults(func=cmd_decide)

    back = sub.add_parser("backtest", help="replay the farm over stored history")
    back.add_argument("--symbol")
    back.add_argument("--days", type=int, default=14)
    back.add_argument("--step", type=int, default=5, help="minutes between decisions")
    back.add_argument("--killzone", action="store_true")
    back.add_argument("--exit-policy", dest="exit_policy",
                      help="all_at_target (default), half_at_1R, thirds_1R_2R, ...")
    back.add_argument("--json", action="store_true")
    back.set_defaults(func=cmd_backtest)

    seed = sub.add_parser("seed", help="download and store history")
    seed.add_argument("--symbol")
    seed.add_argument("--days", type=int, default=30)
    seed.set_defaults(func=cmd_seed)

    validate = sub.add_parser(
        "validate",
        help="seed real history and measure the farm against it")
    validate.add_argument("--symbol")
    validate.add_argument("--days", type=int, default=60)
    validate.add_argument("--step", type=int, default=5)
    validate.add_argument("--equity", type=float)
    validate.add_argument("--contract", default="MNQ")
    validate.add_argument("--ignore-killzone", action="store_true")
    validate.add_argument("--no-seed", action="store_true",
                          help="use stored history instead of downloading")
    validate.add_argument("--json", action="store_true")
    validate.add_argument("--report-json", dest="report_json", metavar="PATH",
                          help="also write the full report as JSON to this path")
    validate.set_defaults(func=cmd_validate)

    agents = sub.add_parser("agents", help="list the roster")
    agents.set_defaults(func=cmd_agents)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
