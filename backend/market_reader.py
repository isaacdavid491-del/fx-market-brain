"""
market_reader.py — formats raw OHLCV data for Claude to read directly.
No pattern labels, no pre-defined concepts. Just clean numbers.
"""
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd


# How many candles to show Claude per timeframe
TF_CANDLE_COUNTS = {
    "1d":  30,   # 30 days
    "4h":  36,   # 6 days
    "1h":  48,   # 2 days
    "15m": 48,   # 12 hours
    "5m":  60,   # 5 hours
    "1m":  60,   # 1 hour
}


def _ts_label(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def format_candles(df: pd.DataFrame, tf: str, n: Optional[int] = None) -> str:
    """
    Format last n candles of a DataFrame as a plain text table.
    Includes open, high, low, close, volume, and two derived values
    (body size as % of range, and close-to-close change) so Claude has
    numbers to work with — it can discover what they mean itself.
    """
    n = n or TF_CANDLE_COUNTS.get(tf, 50)
    recent = df.tail(n).copy().reset_index(drop=True)
    if recent.empty:
        return f"\n[{tf}] — no data\n"

    lines = [
        f"\n═══ {tf.upper()} ({len(recent)} candles, oldest → newest) ═══",
        f"{'Time (UTC)':<18} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Vol':>8} {'Body%':>6} {'Chg':>7}",
        "─" * 75,
    ]

    prev_close = None
    for _, row in recent.iterrows():
        o, h, l, c, v = float(row['o']), float(row['h']), float(row['l']), float(row['c']), float(row['v'])
        rng     = h - l
        body_pct = round(abs(c - o) / rng * 100) if rng > 1e-8 else 0
        chg      = round((c - prev_close) / prev_close * 10000, 1) if prev_close else 0.0  # in pips × 1e4
        ts_lbl   = _ts_label(int(row['t']))
        lines.append(
            f"{ts_lbl:<18} {o:>8.5f} {h:>8.5f} {l:>8.5f} {c:>8.5f} {v:>8.0f} {body_pct:>5}% {chg:>+7.1f}"
        )
        prev_close = c

    lines.append(f"(Body% = candle body as % of total range. Chg = close-to-close change in 0.0001 units.)")
    return "\n".join(lines)


def format_all(df_map: Dict[str, pd.DataFrame]) -> str:
    """Concatenate all TF tables into one block for Claude to read."""
    parts = []
    for tf in ["1d", "4h", "1h", "15m", "5m", "1m"]:
        df = df_map.get(tf)
        if df is not None and not df.empty and len(df) >= 5:
            parts.append(format_candles(df, tf))
    return "\n".join(parts) if parts else "No data available."


def current_price_summary(df_map: Dict[str, pd.DataFrame]) -> Dict:
    """Return the latest OHLC values per TF for quick reference."""
    summary = {}
    for tf, df in df_map.items():
        if df is not None and not df.empty:
            last = df.iloc[-1]
            summary[tf] = {
                "o": round(float(last['o']), 5),
                "h": round(float(last['h']), 5),
                "l": round(float(last['l']), 5),
                "c": round(float(last['c']), 5),
            }
    return summary
