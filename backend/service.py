"""Service layer: the farm as the API sees it.

Holds the configured farm, resolves symbols and keeps the ingestion loop for
the NASDAQ instruments. Kept separate from `app.py` so the API stays thin and
the farm can be driven from a script or a notebook just as easily.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.agents.orchestrator import DEFAULT_CONFIG, AgentFarm, Decision
from backend.backtest.engine import Backtester
from backend.data.feed import build_context, ingest_latest, seed_history
from backend.data.providers import CORRELATED_SYMBOL, NASDAQ_SYMBOL, get_provider
from backend.store import count_rows, latest_ts, load_1m

log = logging.getLogger("ict.service")

_lock = threading.Lock()


def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def farm_config() -> Dict[str, Any]:
    """Farm settings, overridable by environment variable."""
    return {
        **DEFAULT_CONFIG,
        "htf": os.getenv("ICT_HTF", DEFAULT_CONFIG["htf"]),
        "mtf": os.getenv("ICT_MTF", DEFAULT_CONFIG["mtf"]),
        "ltf": os.getenv("ICT_LTF", DEFAULT_CONFIG["ltf"]),
        "entry_threshold": env_float("ICT_ENTRY_THRESHOLD", DEFAULT_CONFIG["entry_threshold"]),
        "min_agreement": env_float("ICT_MIN_AGREEMENT", DEFAULT_CONFIG["min_agreement"]),
        "min_rr": env_float("ICT_MIN_RR", DEFAULT_CONFIG["min_rr"]),
        "require_killzone": env_bool("ICT_REQUIRE_KILLZONE", DEFAULT_CONFIG["require_killzone"]),
        "risk_per_trade": env_float("ICT_RISK_PER_TRADE", DEFAULT_CONFIG["risk_per_trade"]),
        "max_trades_per_day": env_int("ICT_MAX_TRADES_PER_DAY", DEFAULT_CONFIG["max_trades_per_day"]),
        "daily_loss_limit_pct": env_float("ICT_DAILY_LOSS_LIMIT", DEFAULT_CONFIG["daily_loss_limit_pct"]),
        "contract_value": env_float("ICT_CONTRACT_VALUE", DEFAULT_CONFIG["contract_value"]),
    }


class FarmService:
    """Everything the API needs, in one place."""

    def __init__(self) -> None:
        self.symbol = NASDAQ_SYMBOL
        self.correlated_symbol = os.getenv("CORRELATED_SYMBOL", CORRELATED_SYMBOL)
        self.equity = env_float("ICT_EQUITY", 100_000.0)
        self.config = farm_config()
        self.farm = AgentFarm(config=self.config)
        self.provider = get_provider()
        self._last_decision: Optional[Decision] = None
        self._last_decision_ts: float = 0.0

    # -- data ------------------------------------------------------------
    def symbols(self) -> List[str]:
        out = [self.symbol]
        if self.correlated_symbol and self.correlated_symbol != self.symbol:
            out.append(self.correlated_symbol)
        return out

    def ingest(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        results = []
        for sym in ([symbol] if symbol else self.symbols()):
            try:
                results.append(ingest_latest(sym, self.provider))
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                log.warning("ingest %s failed: %s", sym, exc)
                results.append({"ok": False, "symbol": sym, "error": str(exc)})
        return {"results": results}

    def seed(self, days: int = 30, symbol: Optional[str] = None) -> Dict[str, Any]:
        results = []
        for sym in ([symbol] if symbol else self.symbols()):
            try:
                results.append(seed_history(sym, days=days, provider=self.provider))
            except Exception as exc:  # noqa: BLE001
                log.warning("seed %s failed: %s", sym, exc)
                results.append({"ok": False, "symbol": sym, "error": str(exc)})
        return {"results": results}

    def data_status(self) -> Dict[str, Any]:
        out = {}
        for sym in self.symbols():
            latest = latest_ts(sym)
            out[sym] = {
                "bars_stored": count_rows(sym),
                "latest_ts": latest,
                "age_seconds": int(time.time() - latest) if latest else None,
            }
        return {"provider": self.provider.name, "symbols": out}

    # -- analysis ---------------------------------------------------------
    def decide(self, symbol: Optional[str] = None, now_ts: Optional[int] = None,
               require_killzone: Optional[bool] = None,
               equity: Optional[float] = None) -> Decision:
        sym = symbol or self.symbol
        config = dict(self.config)
        if require_killzone is not None:
            config["require_killzone"] = bool(require_killzone)
        ctx = build_context(
            symbol=sym, now_ts=now_ts, config=config,
            equity=equity if equity is not None else self.equity,
            risk_per_trade=float(config.get("risk_per_trade", 0.005)),
            correlated_symbol=self.correlated_symbol,
            provider=self.provider,
        )
        farm = AgentFarm(config=config)
        decision = farm.evaluate(ctx)
        with _lock:
            self._last_decision = decision
            self._last_decision_ts = time.time()
        return decision

    def last_decision(self) -> Optional[Decision]:
        with _lock:
            return self._last_decision

    # -- backtest ---------------------------------------------------------
    def backtest(self, symbol: Optional[str] = None, days: int = 14,
                 step_minutes: int = 5, equity: Optional[float] = None,
                 require_killzone: Optional[bool] = None,
                 include_trades: bool = True) -> Dict[str, Any]:
        sym = symbol or self.symbol
        end = int(time.time())
        start = end - days * 86400
        df = load_1m(sym, start, end)
        if df.empty:
            return {"ok": False, "error": f"no stored history for {sym}; seed it first"}

        config = dict(self.config)
        if require_killzone is not None:
            config["require_killzone"] = bool(require_killzone)

        peer = None
        if self.correlated_symbol:
            peer_df = load_1m(self.correlated_symbol, start, end)
            peer = peer_df if not peer_df.empty else None

        bt = Backtester(
            farm=AgentFarm(config=config),
            starting_equity=equity if equity is not None else self.equity,
            risk_per_trade=float(config.get("risk_per_trade", 0.005)),
            step_minutes=step_minutes,
        )
        # Leave enough history for the highest timeframe to be meaningful.
        warmup = min(max(len(df) // 4, 2000), max(len(df) - 500, 1))
        result = bt.run(
            sym, df, correlated_1m=peer,
            correlated_symbol=self.correlated_symbol if peer is not None else None,
            warmup_bars=warmup, config=config,
        )
        out = result.as_dict(include_trades=include_trades)
        out["ok"] = True
        out["provider"] = self.provider.name
        out["config"] = {k: config[k] for k in
                         ("htf", "mtf", "ltf", "entry_threshold", "min_agreement",
                          "min_rr", "require_killzone", "risk_per_trade")}
        return out


_service: Optional[FarmService] = None


def get_service() -> FarmService:
    global _service
    if _service is None:
        _service = FarmService()
    return _service
