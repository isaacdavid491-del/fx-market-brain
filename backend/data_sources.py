"""
data_sources.py — free real market data via Yahoo Finance.
No API key, no account, no signup required.
Supports NAS100, any stock, forex pair, crypto, commodity, index.
"""
import threading
import time
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

# ─── Symbol aliases → Yahoo Finance tickers ───────────────────────────────────
SYMBOL_MAP = {
    # US indices (futures — most liquid, real-time)
    "NAS100":  "NQ=F",  "NASDAQ": "NQ=F",  "US100": "NQ=F",  "NQ": "NQ=F",
    "SP500":   "ES=F",  "US500":  "ES=F",  "SPX":   "ES=F",  "ES": "ES=F",
    "DOW30":   "YM=F",  "US30":   "YM=F",  "DOW":   "YM=F",
    "RUSSELL": "RTY=F",
    # European / Asian indices
    "DAX":     "^GDAXI", "GER40":  "^GDAXI",
    "FTSE":    "^FTSE",  "UK100":  "^FTSE",
    "NIKKEI":  "^N225",  "JP225":  "^N225",
    "HSI":     "^HSI",
    # Commodities (futures)
    "GOLD":    "GC=F",   "XAUUSD": "GC=F",   "XAU": "GC=F",
    "SILVER":  "SI=F",   "XAGUSD": "SI=F",
    "OIL":     "CL=F",   "USOIL":  "CL=F",   "WTI": "CL=F",
    "BRENT":   "BZ=F",
    "NATGAS":  "NG=F",
    # Forex (=X suffix means spot rate)
    "EURUSD":  "EURUSD=X",  "EUR_USD": "EURUSD=X",
    "GBPUSD":  "GBPUSD=X",  "GBP_USD": "GBPUSD=X",
    "USDJPY":  "USDJPY=X",  "USD_JPY": "USDJPY=X",
    "USDCHF":  "USDCHF=X",
    "AUDUSD":  "AUDUSD=X",
    "NZDUSD":  "NZDUSD=X",
    "USDCAD":  "USDCAD=X",
    "GBPJPY":  "GBPJPY=X",
    "EURJPY":  "EURJPY=X",
    # Crypto
    "BTCUSD":  "BTC-USD",  "BTC":  "BTC-USD",  "BITCOIN": "BTC-USD",
    "ETHUSD":  "ETH-USD",  "ETH":  "ETH-USD",
    "SOLUSD":  "SOL-USD",  "SOL":  "SOL-USD",
    "XRPUSD":  "XRP-USD",
}

# yfinance interval/period per timeframe
_TF_CONFIG: Dict[str, tuple] = {
    "1m":  ("1m",  "5d"),    # 5 days of 1-min candles
    "5m":  ("5m",  "30d"),   # 30 days of 5-min candles
    "15m": ("15m", "60d"),   # 60 days of 15-min candles
    "1h":  ("60m", "2y"),    # 2 years of 1-hour candles
    "1d":  ("1d",  "5y"),    # 5 years of daily candles
}

# In-memory cache (60-second TTL) to avoid hammering Yahoo Finance
_cache: Dict[str, dict] = {}
_cache_ttl = 60
_lock = threading.Lock()


def resolve(symbol: str) -> str:
    """Map a friendly symbol name to its Yahoo Finance ticker."""
    key = symbol.upper().replace("/", "").replace("_", "").replace("-", "")
    return SYMBOL_MAP.get(key, symbol.upper())


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance DataFrame to our standard t,o,h,l,c,v format."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])

    df = raw.copy()

    # yfinance sometimes returns MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    rename = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
    df = df.rename(columns=rename)

    for col in ["o", "h", "l", "c"]:
        if col not in df.columns:
            return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    if "v" not in df.columns:
        df["v"] = 0.0

    # Convert DatetimeIndex → unix timestamp (seconds)
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        df["t"] = idx.astype("int64") // 10 ** 9
    else:
        df["t"] = pd.to_datetime(idx, utc=True).astype("int64") // 10 ** 9

    df = (
        df[["t", "o", "h", "l", "c", "v"]]
        .dropna(subset=["o", "h", "l", "c"])
        .drop_duplicates("t")
        .sort_values("t")
        .reset_index(drop=True)
    )
    return df.astype({"t": int, "o": float, "h": float, "l": float, "c": float, "v": float})


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    d = df_1h.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt")
    out = pd.DataFrame({
        "o": d["o"].resample("4h").first(),
        "h": d["h"].resample("4h").max(),
        "l": d["l"].resample("4h").min(),
        "c": d["c"].resample("4h").last(),
        "v": d["v"].resample("4h").sum(),
    }).dropna()
    out["t"] = out.index.astype("int64") // 10 ** 9
    return out.reset_index(drop=True)[["t", "o", "h", "l", "c", "v"]]


def fetch_all(symbol: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all timeframes (1m, 5m, 15m, 1h, 4h, 1d).
    Results are cached for 60 seconds to avoid rate-limiting.
    Uses Yahoo Finance — no API key, no account needed.
    """
    key = symbol.upper()
    now = time.time()

    with _lock:
        cached = _cache.get(key)
        if cached and now - cached["ts"] < _cache_ttl:
            return cached["data"]

    ticker = resolve(symbol)
    data: Dict[str, pd.DataFrame] = {}

    for tf, (interval, period) in _TF_CONFIG.items():
        try:
            raw = yf.download(
                ticker,
                interval=interval,
                period=period,
                progress=False,
                auto_adjust=True,
            )
            df = _normalize(raw)
            if not df.empty:
                data[tf] = df
        except Exception as exc:
            print(f"[data_sources] {ticker} {tf}: {exc}")

    # 4h = resample from 1h (Yahoo doesn't provide 4h natively)
    if "1h" in data and not data["1h"].empty:
        r4h = _resample_4h(data["1h"])
        if not r4h.empty:
            data["4h"] = r4h

    with _lock:
        _cache[key] = {"data": data, "ts": now}

    return data


def latest_price(symbol: str) -> Optional[float]:
    """Return the most recent close price for a symbol."""
    try:
        info = yf.Ticker(resolve(symbol)).fast_info
        p = getattr(info, "last_price", None)
        if p:
            return float(p)
    except Exception:
        pass
    # Fallback: use cached 1m data
    with _lock:
        cached = _cache.get(symbol.upper())
        if cached:
            df = cached["data"].get("1m")
            if df is not None and not df.empty:
                return float(df["c"].iloc[-1])
    return None
