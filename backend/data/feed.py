"""Feed assembly: candles in, `MarketContext` out.

The farm never touches the database or the network directly. Everything it
reads comes through here, which is what lets the backtester hand it sliced
history and get identical behaviour.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from backend.agents.base import MarketContext
from backend.data.providers import CandleProvider, get_provider
from backend.store import TF_MINUTES, load_1m, resample_ohlcv, upsert_1m, latest_ts

log = logging.getLogger("ict.feed")

DEFAULT_TIMEFRAMES = ("5m", "15m", "1h")
DEFAULT_BARS = 300


def ingest_latest(symbol: str, provider: Optional[CandleProvider] = None,
                  count: int = 500) -> Dict[str, Any]:
    """Pull the most recent 1-minute candles and store them."""
    provider = provider or get_provider()
    df = provider.fetch_1m(symbol, count=count)
    written = upsert_1m(symbol, df)
    return {
        "ok": True, "symbol": symbol, "provider": provider.name,
        "written": written, "latest_ts": latest_ts(symbol),
    }


def seed_history(symbol: str, days: int = 30, provider: Optional[CandleProvider] = None,
                 max_pages: int = 200) -> Dict[str, Any]:
    """Page backwards until `days` of 1-minute history is stored."""
    provider = provider or get_provider()
    target_start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    cursor = int(time.time())
    total = 0

    for _ in range(max_pages):
        df = provider.fetch_1m(symbol, count=500, to_ts=cursor)
        if df is None or df.empty:
            break
        total += upsert_1m(symbol, df)
        oldest = int(df["t"].min())
        if oldest <= target_start:
            break
        cursor = oldest - 60
        if provider.name == "oanda":
            time.sleep(0.2)   # stay well inside the rate limit

    return {"ok": True, "symbol": symbol, "provider": provider.name,
            "written": total, "days": days}


def load_frames(symbol: str, now_ts: Optional[int] = None,
                timeframes: Optional[List[str]] = None,
                bars: int = DEFAULT_BARS,
                provider: Optional[CandleProvider] = None,
                allow_fetch: bool = True) -> Dict[str, pd.DataFrame]:
    """Build `{timeframe: frame}` ending at `now_ts`.

    Reads the stored 1-minute series and aggregates up. If the store is short,
    it backfills from the provider once rather than returning a frame too thin
    for the agents to read.
    """
    now_ts = int(now_ts or time.time())
    timeframes = list(timeframes or DEFAULT_TIMEFRAMES)
    minutes_needed = max(bars * TF_MINUTES.get(tf, 1) for tf in timeframes)
    # Overnight and weekend gaps mean wall-clock span must exceed bar count.
    span = int(minutes_needed * 60 * 2.2)
    start_ts = now_ts - span

    df1 = load_1m(symbol, start_ts, now_ts)
    if allow_fetch and len(df1) < minutes_needed * 0.4:
        provider = provider or get_provider()
        try:
            fetched = provider.fetch_1m(symbol, count=min(minutes_needed, 5000), to_ts=now_ts)
            if not fetched.empty:
                upsert_1m(symbol, fetched)
                df1 = load_1m(symbol, start_ts, now_ts)
        except Exception as exc:  # noqa: BLE001 - a failed backfill is not fatal
            log.warning("backfill for %s failed: %s", symbol, exc)

    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        frame = resample_ohlcv(df1, tf)
        # Never let a partially-formed final bar reach the agents.
        if not frame.empty:
            bar_seconds = TF_MINUTES.get(tf, 1) * 60
            if frame["t"].iloc[-1] + bar_seconds > now_ts:
                frame = frame.iloc[:-1]
        out[tf] = frame.tail(bars).reset_index(drop=True)
    return out


def build_context(symbol: str, now_ts: Optional[int] = None,
                  config: Optional[Dict[str, Any]] = None,
                  equity: float = 100_000.0,
                  risk_per_trade: float = 0.005,
                  correlated_symbol: Optional[str] = None,
                  bars: int = DEFAULT_BARS,
                  provider: Optional[CandleProvider] = None,
                  allow_fetch: bool = True) -> MarketContext:
    config = dict(config or {})
    now_ts = int(now_ts or time.time())
    timeframes = [config.get("ltf", "5m"), config.get("mtf", "15m"), config.get("htf", "1h")]
    timeframes = list(dict.fromkeys(timeframes))

    frames = load_frames(symbol, now_ts, timeframes, bars, provider, allow_fetch)
    correlated_frames: Dict[str, pd.DataFrame] = {}
    if correlated_symbol:
        try:
            correlated_frames = load_frames(
                correlated_symbol, now_ts, timeframes, bars, provider, allow_fetch
            )
        except Exception as exc:  # noqa: BLE001 - SMT degrades to abstaining
            log.warning("correlated feed %s failed: %s", correlated_symbol, exc)

    return MarketContext(
        symbol=symbol, now_ts=now_ts, frames=frames,
        correlated_symbol=correlated_symbol, correlated_frames=correlated_frames,
        equity=equity, risk_per_trade=risk_per_trade, config=config,
    )
