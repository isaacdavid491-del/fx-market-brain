"""Candle providers.

Two implementations: OANDA for real NASDAQ CFD data (NAS100_USD), and a
deterministic synthetic generator so the farm, the API and the backtester can
be exercised without credentials. The synthetic feed is clearly labelled
everywhere it surfaces; it exists for development and tests, never to stand in
for real prices in a decision.
"""
from __future__ import annotations

import logging
import os
import time
import zlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import requests

from backend.ict.sessions import NY, is_weekend

log = logging.getLogger("ict.data")

OANDA_API_BASE = os.getenv("OANDA_API_BASE", "https://api-fxpractice.oanda.com/v3")
OHLCV = ["t", "o", "h", "l", "c", "v"]

# OANDA's CFD tickers for the indices the farm cares about.
NASDAQ_SYMBOL = os.getenv("NASDAQ_SYMBOL", "NAS100_USD")
CORRELATED_SYMBOL = os.getenv("CORRELATED_SYMBOL", "SPX500_USD")


class CandleProvider(Protocol):
    name: str

    def fetch_1m(self, symbol: str, count: int = 500,
                 to_ts: Optional[int] = None) -> pd.DataFrame: ...


class OandaProvider:
    """Live/practice OANDA v3 candles."""

    name = "oanda"

    def __init__(self, token: Optional[str] = None, api_base: Optional[str] = None):
        self.token = token if token is not None else os.getenv("OANDA_TOKEN", "")
        self.api_base = api_base or OANDA_API_BASE

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _headers(self) -> Dict[str, str]:
        if not self.token:
            raise RuntimeError("OANDA_TOKEN is not set")
        return {"Authorization": f"Bearer {self.token}"}

    def fetch_1m(self, symbol: str, count: int = 500,
                 to_ts: Optional[int] = None) -> pd.DataFrame:
        params: Dict[str, str] = {
            "granularity": "M1", "price": "M", "count": str(min(int(count), 5000)),
        }
        if to_ts:
            params["to"] = datetime.fromtimestamp(int(to_ts), tz=timezone.utc).isoformat()
        url = f"{self.api_base}/instruments/{symbol}/candles"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"OANDA error {resp.status_code}: {resp.text[:300]}")
        return parse_oanda_candles(resp.json().get("candles", []))


def parse_oanda_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in candles:
        if not item.get("complete"):
            continue
        dt = datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
        mid = item.get("mid") or {}
        try:
            rows.append({
                "t": int(dt.timestamp()),
                "o": float(mid["o"]), "h": float(mid["h"]),
                "l": float(mid["l"]), "c": float(mid["c"]),
                "v": float(item.get("volume", 0.0)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=OHLCV)
    return pd.DataFrame(rows).drop_duplicates(subset=["t"]).sort_values("t").reset_index(drop=True)


class SyntheticProvider:
    """Deterministic NASDAQ-shaped 1-minute bars for development and tests.

    The walk is shaped so the ICT primitives have something real to find:
    volatility follows the New York session profile, the market is closed at
    weekends, and periodic displacement legs leave gaps and order blocks
    behind rather than producing featureless noise.
    """

    name = "synthetic"

    def __init__(self, base_price: float = 20_000.0, seed: int = 7,
                 minute_vol: float = 0.00035):
        self.base_price = float(base_price)
        self.seed = int(seed)
        self.minute_vol = float(minute_vol)

    @staticmethod
    def _session_multiplier(ts: int) -> float:
        """Intraday volatility profile in NY time."""
        local = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(NY)
        minutes = local.hour * 60 + local.minute
        if 570 <= minutes < 660:      # 09:30-11:00 the cash open
            return 2.4
        if 660 <= minutes < 780:      # lunch
            return 0.7
        if 780 <= minutes < 960:      # afternoon into the close
            return 1.4
        if 120 <= minutes < 300:      # London
            return 1.3
        if 420 <= minutes < 570:      # pre-open
            return 1.1
        return 0.5                    # Asia / overnight

    def fetch_1m(self, symbol: str, count: int = 500,
                 to_ts: Optional[int] = None) -> pd.DataFrame:
        end = int(to_ts or time.time())
        end -= end % 60
        count = int(count)
        # Seeded on the symbol so repeated calls agree with each other and
        # with the backtester. zlib.crc32 rather than the built-in hash():
        # Python randomises string hashing per process, so hash() made the
        # series differ between runs and quietly destroyed the comparability
        # of any two backtests executed as separate processes.
        seed = (zlib.crc32(symbol.encode("utf-8")) + self.seed) % (2**32)
        rng = np.random.default_rng(seed)

        # Build backwards from `end` so the series never runs past the
        # requested boundary: walking forwards and skipping weekends
        # overshoots by exactly the closed time it stepped over.
        stamps: List[int] = []
        cursor = end
        guard = count * 10 + 20_000
        while len(stamps) < count and guard > 0:
            guard -= 1
            cursor -= 60
            if not is_weekend(cursor):
                stamps.append(cursor)
        stamps.reverse()

        n = len(stamps)
        # Occasional displacement legs: the fast one-sided moves that leave
        # fair value gaps and order blocks behind.
        shocks = np.zeros(n)
        for idx in rng.choice(n, size=max(1, n // 240), replace=False):
            shocks[idx] = rng.normal(0, 1) * 0.004
        mults = np.array([self._session_multiplier(t) for t in stamps])
        steps = rng.normal(0, self.minute_vol, n) * mults + shocks

        # A plain walk at this volatility already trends enough for structure
        # to break. Anything added on top compounds and the series wanders off
        # to prices the instrument would never print.
        close = self.base_price * np.exp(np.cumsum(steps))
        open_ = np.concatenate([[self.base_price], close[:-1]])
        spread = np.abs(rng.normal(0, self.minute_vol * 0.8, n)) * mults * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = (rng.gamma(2.0, 200.0, n) * mults).round()

        return pd.DataFrame({
            "t": stamps, "o": open_, "h": high, "l": low, "c": close, "v": volume,
        })[OHLCV]


def get_provider(name: Optional[str] = None) -> CandleProvider:
    """OANDA when a token is present, synthetic otherwise.

    Set DATA_PROVIDER=synthetic to force the offline feed.
    """
    choice = (name or os.getenv("DATA_PROVIDER", "auto")).lower()
    if choice == "synthetic":
        return SyntheticProvider()
    oanda = OandaProvider()
    if choice == "oanda":
        if not oanda.available:
            raise RuntimeError("DATA_PROVIDER=oanda but OANDA_TOKEN is not set")
        return oanda
    if oanda.available:
        return oanda
    log.warning("No OANDA_TOKEN: falling back to the synthetic feed. "
                "Signals are for development only.")
    return SyntheticProvider()
