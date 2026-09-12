"""SQLite candle store and timeframe resampling.

Shared by the FX signal model and the NASDAQ ICT agent farm so both read
exactly the same bars. Pure storage concerns only: no strategy logic here.
"""
from __future__ import annotations

import os
import sqlite3
from typing import List, Optional

import pandas as pd

DB_PATH = os.getenv("DB_PATH", "/tmp/fx.db")

OHLCV_COLUMNS = ["t", "o", "h", "l", "c", "v"]

# Pandas offset aliases. "T"/"H" were deprecated in pandas 2.2 and removed in
# pandas 3.0; the lowercase forms below are valid in both.
TF_MAP = {
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

# Minutes per timeframe, used for lookback maths.
TF_MINUTES = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "15m": 15,
    "30m": 30, "1h": 60, "4h": 240, "1d": 1440,
}


def db(path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection, creating the parent directory when needed."""
    target = path or DB_PATH
    db_dir = os.path.dirname(target)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(path: Optional[str] = None) -> None:
    conn = db(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles_1m (
          symbol TEXT NOT NULL,
          t INTEGER NOT NULL,
          o REAL NOT NULL,
          h REAL NOT NULL,
          l REAL NOT NULL,
          c REAL NOT NULL,
          v REAL NOT NULL,
          PRIMARY KEY(symbol, t)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_t ON candles_1m(symbol, t);")
    conn.commit()
    conn.close()


def upsert_1m(symbol: str, df: pd.DataFrame, path: Optional[str] = None) -> int:
    if df is None or df.empty:
        return 0
    rows = [
        (symbol, int(r.t), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v))
        for r in df.itertuples(index=False)
    ]
    conn = db(path)
    conn.executemany(
        "INSERT OR REPLACE INTO candles_1m(symbol,t,o,h,l,c,v) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def latest_ts(symbol: str, path: Optional[str] = None) -> Optional[int]:
    conn = db(path)
    cur = conn.cursor()
    cur.execute("SELECT MAX(t) FROM candles_1m WHERE symbol = ?", (symbol,))
    out = cur.fetchone()[0]
    conn.close()
    return int(out) if out is not None else None


def count_rows(symbol: str, path: Optional[str] = None) -> int:
    conn = db(path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM candles_1m WHERE symbol = ?", (symbol,))
    out = cur.fetchone()[0]
    conn.close()
    return int(out or 0)


def known_symbols(path: Optional[str] = None) -> List[str]:
    conn = db(path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM candles_1m ORDER BY symbol")
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def load_1m(symbol: str, start_ts: int, end_ts: int, path: Optional[str] = None) -> pd.DataFrame:
    conn = db(path)
    query = (
        "SELECT t,o,h,l,c,v FROM candles_1m "
        "WHERE symbol = ? AND t BETWEEN ? AND ? ORDER BY t ASC"
    )
    df = pd.read_sql_query(query, conn, params=(symbol, int(start_ts), int(end_ts)))
    conn.close()
    return df


def resample_ohlcv(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate 1-minute bars up to `tf`. Returns the same column layout."""
    if tf not in TF_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}")
    if df_1m is None or df_1m.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    if tf == "1m":
        return df_1m.reset_index(drop=True)[OHLCV_COLUMNS].copy()

    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt").sort_index()
    rule = TF_MAP[tf]
    out = pd.DataFrame(
        {
            "o": d["o"].resample(rule).first(),
            "h": d["h"].resample(rule).max(),
            "l": d["l"].resample(rule).min(),
            "c": d["c"].resample(rule).last(),
            "v": d["v"].resample(rule).sum(),
        }
    ).dropna(subset=["o", "h", "l", "c"])
    if out.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    # Epoch seconds without assuming the index resolution: pandas 2.x indexes
    # are nanosecond-backed, pandas 3.x may be microsecond-backed, so neither
    # `.view("int64")` nor a fixed 10**9 divisor is portable.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    out["t"] = ((out.index - epoch) // pd.Timedelta(seconds=1)).astype("int64")
    return out.reset_index(drop=True)[OHLCV_COLUMNS]
