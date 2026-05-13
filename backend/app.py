import os
import json
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import HistGradientBoostingClassifier

# =============================
# Config
# =============================

# FIX 1: Use /tmp by default (Render always writable). You can still override via DB_PATH env var.
DB_PATH = os.getenv("DB_PATH", "/tmp/fx.db")

OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
DEFAULT_INSTRUMENT = os.getenv("DEFAULT_INSTRUMENT", "EUR_USD")
INGEST_EVERY_SECONDS = int(os.getenv("INGEST_EVERY_SECONDS", "60"))
HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "30"))

OANDA_API_BASE = "https://api-fxpractice.oanda.com/v3"

app = FastAPI(title="FX Market Brain")

# =============================
# DB
# =============================

def db() -> sqlite3.Connection:
    # FIX 2: Ensure DB directory exists before connecting (important on Render)
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)  # keep safe; if dirname is "", skip
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    # FIX 2 (again): ensure directory exists at startup too
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = db()
    conn.execute("""
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
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_t ON candles_1m(symbol, t);")
    conn.commit()
    conn.close()

# =============================
# OANDA client
# =============================

def oanda_headers() -> Dict[str, str]:
    if not OANDA_TOKEN:
        raise RuntimeError("Missing OANDA_TOKEN")
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def oanda_get_candles(symbol: str, granularity: str, count: int = 500, to_rfc3339: Optional[str] = None) -> List[Dict[str, Any]]:
    # granularity: M1, M5, M15, H1...
    params = {"granularity": granularity, "price": "M", "count": str(count)}
    if to_rfc3339:
        params["to"] = to_rfc3339
    url = f"{OANDA_API_BASE}/instruments/{symbol}/candles"
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA error {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data.get("candles", [])

def parse_oanda_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for x in candles:
        if not x.get("complete"):
            continue
        t = x["time"]
        dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        ts = int(dt.timestamp())
        mid = x["mid"]
        rows.append({
            "t": ts,
            "o": float(mid["o"]),
            "h": float(mid["h"]),
            "l": float(mid["l"]),
            "c": float(mid["c"]),
            "v": float(x.get("volume", 0.0)),
        })
    if not rows:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["t"]).sort_values("t")
    return df

def upsert_1m(symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = db()
    cur = conn.cursor()
    n = 0
    for r in df.itertuples(index=False):
        try:
            cur.execute(
                "INSERT OR REPLACE INTO candles_1m(symbol,t,o,h,l,c,v) VALUES (?,?,?,?,?,?,?)",
                (symbol, int(r.t), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)),
            )
            n += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return n

def latest_ts(symbol: str) -> Optional[int]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT MAX(t) FROM candles_1m WHERE symbol = ?", (symbol,))
    out = cur.fetchone()[0]
    conn.close()
    return int(out) if out is not None else None

def load_1m(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    conn = db()
    q = """
      SELECT t,o,h,l,c,v FROM candles_1m
      WHERE symbol = ? AND t BETWEEN ? AND ?
      ORDER BY t ASC
    """
    df = pd.read_sql_query(q, conn, params=(symbol, start_ts, end_ts))
    conn.close()
    return df

# =============================
# Multi-timeframe resample (derived from 1m)
# =============================

TF_MAP = {
    "1m": "1T",
    "5m": "5T",
    "15m": "15T",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

def resample_ohlcv(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    if tf not in TF_MAP:
        raise ValueError("Unsupported tf")
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt")
    rule = TF_MAP[tf]
    out = pd.DataFrame()
    out["o"] = d["o"].resample(rule).first()
    out["h"] = d["h"].resample(rule).max()
    out["l"] = d["l"].resample(rule).min()
    out["c"] = d["c"].resample(rule).last()
    out["v"] = d["v"].resample(rule).sum()
    out = out.dropna()
    out["t"] = (out.index.view("int64") // 10**9).astype(int)
    return out.reset_index(drop=True)[["t","o","h","l","c","v"]]

# =============================
# AI baseline (learns from data, multi-TF features)
# =============================

_model: Optional[HistGradientBoostingClassifier] = None
_model_meta: Dict[str, Any] = {}

def build_features(df_1m: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Purely data-driven features across multiple timeframes:
    - returns, rolling vol, range position, momentum across 1m/5m/15m/1h/4h/1d
    """
    if df_1m.shape[0] < 2000:
        return None

    def add_tf_feats(tf_name: str, window: int) -> pd.DataFrame:
        d = resample_ohlcv(df_1m, tf_name)
        if d.shape[0] < window + 50:
            return pd.DataFrame()
        d["ret1"] = d["c"].pct_change()
        d["vol"] = d["ret1"].rolling(window).std()
        d["mom"] = d["c"].pct_change(window)
        d["rng"] = (d["h"] - d["l"]) / d["c"].replace(0, np.nan)
        roll_hi = d["h"].rolling(window).max()
        roll_lo = d["l"].rolling(window).min()
        d["pos"] = (d["c"] - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
        cols = ["t", "ret1", "vol", "mom", "rng", "pos"]
        d = d[cols].dropna()
        d = d.rename(columns={c: f"{tf_name}_{c}" for c in cols if c != "t"})
        return d

    feats = None
    specs = [("1m", 60), ("5m", 60), ("15m", 60), ("1h", 60), ("4h", 60), ("1d", 30)]
    for tf_name, w in specs:
        f = add_tf_feats(tf_name, w)
        if f.empty:
            continue
        feats = f if feats is None else feats.merge(f, on="t", how="inner")

    if feats is None or feats.shape[0] < 500:
        return None

    H = 60  # 60 minutes ahead
    base = df_1m[["t","c"]].copy()
    base["fut_c"] = base["c"].shift(-H)
    base["y"] = (base["fut_c"] > base["c"]).astype(int)
    lab = base[["t","y"]].dropna()

    feats = feats.merge(lab, on="t", how="inner").dropna()
    return feats

def train_model(symbol: str) -> Dict[str, Any]:
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df = load_1m(symbol, start_ts, end_ts)
    feats = build_features(df)
    if feats is None:
        return {"ok": False, "reason": "Not enough data yet. Let ingestion run longer."}

    X = feats.drop(columns=["y"])
    y = feats["y"].astype(int)

    split = int(len(feats) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split:], y.iloc[split:]

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300)
    model.fit(X_train.drop(columns=["t"]), y_train)

    acc = float(model.score(X_test.drop(columns=["t"]), y_test))
    global _model, _model_meta
    _model = model
    _model_meta = {
        "symbol": symbol,
        "trained_at": int(time.time()),
        "rows": int(len(feats)),
        "test_acc": acc,
        "horizon_minutes": 60,
    }
    return {"ok": True, **_model_meta}

def infer_signal(symbol: str) -> Dict[str, Any]:
    global _model
    if _model is None:
        train_model(symbol)

    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df = load_1m(symbol, start_ts, end_ts)
    feats = build_features(df)
    if feats is None or feats.empty:
        return {"side": "NEUTRAL", "confidence": 0.0, "reason": "Not enough data yet."}

    latest = feats.sort_values("t").iloc[-1:]
    X = latest.drop(columns=["y"])
    proba_up = float(_model.predict_proba(X.drop(columns=["t"]))[0, 1]) if _model else 0.5

    if proba_up >= 0.55:
        side = "LONG"
        conf = (proba_up - 0.5) * 2.0
    elif proba_up <= 0.45:
        side = "SHORT"
        conf = (0.5 - proba_up) * 2.0
    else:
        side = "NEUTRAL"
        conf = 0.0

    df_recent = df.tail(2000).copy()
    df_recent["ret1"] = df_recent["c"].pct_change()
    vol = float(df_recent["ret1"].rolling(300).std().iloc[-1] or 0.0005)
    last = float(df_recent["c"].iloc[-1])

    sl_dist = max(2.5 * vol * last, 0.0008 * last)
    tp_dist = max(4.0 * vol * last, 0.0012 * last)

    if side == "LONG":
        entry = last
        sl = last - sl_dist
        tp = last + tp_dist
    elif side == "SHORT":
        entry = last
        sl = last + sl_dist
        tp = last - tp_dist
    else:
        entry = last
        sl = None
        tp = None

    return {
        "symbol": symbol,
        "timestamp": int(df_recent["t"].iloc[-1]),
        "side": side,
        "confidence": round(float(conf), 4),
        "prob_up": round(proba_up, 4),
        "entry": round(entry, 6),
        "sl": round(sl, 6) if sl is not None else None,
        "tp": round(tp, 6) if tp is not None else None,
        "horizon_minutes": _model_meta.get("horizon_minutes", 60),
        "model": _model_meta,
    }

# =============================
# ICT Analysis Engine
# =============================

_NY_TZ = ZoneInfo("America/New_York")

_KILL_ZONES: Dict[str, tuple] = {
    "london": ( 2 * 60,  5 * 60),        # 02:00–05:00 ET
    "ny_am":  ( 7 * 60, 11 * 60),        # 07:00–11:00 ET
    "ny_pm":  (13 * 60 + 30, 16 * 60),   # 13:30–16:00 ET
}

def _kz_name(ts: int) -> Optional[str]:
    dt = datetime.fromtimestamp(ts, tz=_NY_TZ)
    hm = dt.hour * 60 + dt.minute
    for name, (start, end) in _KILL_ZONES.items():
        if start <= hm < end:
            return name
    return None


def compute_ict(df: pd.DataFrame, swing_len: int = 5, min_rr: float = 2.0) -> Dict[str, Any]:
    """
    Port of Pine Script ICT Live Analysis Engine.
    Detects kill zones, pivot-based liquidity pools, sweeps, FVGs, MSS, and trade signals.
    """
    if df.shape[0] < swing_len * 2 + 10:
        return {"error": "Not enough data for ICT analysis"}

    d = df.copy().reset_index(drop=True)
    n = len(d)

    # ── Kill zones ────────────────────────────────────────────────────────────
    d["kz"] = d["t"].apply(_kz_name)
    d["in_kz"] = d["kz"].notna()

    # ── Pivot highs / lows (vectorised via centered rolling window) ───────────
    w = 2 * swing_len + 1
    roll_max = d["h"].rolling(w, center=True, min_periods=w).max()
    roll_min = d["l"].rolling(w, center=True, min_periods=w).min()
    d["pivot_high"] = np.where(d["h"] == roll_max, d["h"], np.nan)
    d["pivot_low"]  = np.where(d["l"] == roll_min, d["l"], np.nan)

    # Forward-fill – mirrors Pine Script's `var float last_ph/pl`
    d["last_ph"] = pd.Series(d["pivot_high"]).ffill()
    d["last_pl"] = pd.Series(d["pivot_low"]).ffill()

    # ── Liquidity sweeps ──────────────────────────────────────────────────────
    d["bsl_sweep"] = d["last_ph"].notna() & (d["h"] > d["last_ph"]) & (d["c"] < d["last_ph"])
    d["ssl_sweep"] = d["last_pl"].notna() & (d["l"] < d["last_pl"]) & (d["c"] > d["last_pl"])

    # ── Fair Value Gaps ───────────────────────────────────────────────────────
    # Bullish FVG (SIBI): low[current] > high[2 bars ago]
    d["bull_fvg_lo"] = d["h"].shift(2)   # lower boundary = high of 2-bar-ago candle
    d["bull_fvg_hi"] = d["l"]            # upper boundary = low of current candle
    d["bull_fvg"]    = d["bull_fvg_hi"] > d["bull_fvg_lo"]

    # Bearish FVG (BISI): high[current] < low[2 bars ago]
    d["bear_fvg_lo"] = d["h"]            # lower boundary = high of current candle
    d["bear_fvg_hi"] = d["l"].shift(2)   # upper boundary = low of 2-bar-ago candle
    d["bear_fvg"]    = d["bear_fvg_lo"] < d["bear_fvg_hi"]

    # ── Market Structure Shifts ───────────────────────────────────────────────
    d["bull_mss"] = (
        d["ssl_sweep"].shift(1).fillna(False)
        & d["last_ph"].notna()
        & (d["c"] > d["last_ph"])
    )
    d["bear_mss"] = (
        d["bsl_sweep"].shift(1).fillna(False)
        & d["last_pl"].notna()
        & (d["c"] < d["last_pl"])
    )

    # ── Trade signals ─────────────────────────────────────────────────────────
    d["long_sig"]  = d["bull_mss"] & d["bull_fvg"] & d["in_kz"]
    d["short_sig"] = d["bear_mss"] & d["bear_fvg"] & d["in_kz"]

    # ── Collect results (last 200 bars) ───────────────────────────────────────
    recent = d.tail(200)

    signals: List[Dict[str, Any]] = []
    for row in recent.itertuples(index=False):
        if row.long_sig:
            lo, hi = float(row.bull_fvg_lo), float(row.bull_fvg_hi)
            ep = (lo + hi) / 2
            sl = lo - (hi - lo)       # one FVG-width below zone
            tp = ep + (ep - sl) * min_rr
            signals.append({"t": int(row.t), "side": "LONG",
                             "entry": round(ep, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                             "kz": row.kz, "fvg_lo": round(lo, 6), "fvg_hi": round(hi, 6)})
        elif row.short_sig:
            lo, hi = float(row.bear_fvg_lo), float(row.bear_fvg_hi)
            ep = (lo + hi) / 2
            # Bug-fix vs original Pine Script: SL must be ABOVE the FVG zone for a short
            sl = hi + (hi - lo)       # one FVG-width above zone
            tp = ep - (sl - ep) * min_rr
            signals.append({"t": int(row.t), "side": "SHORT",
                             "entry": round(ep, 6), "sl": round(sl, 6), "tp": round(tp, 6),
                             "kz": row.kz, "fvg_lo": round(lo, 6), "fvg_hi": round(hi, 6)})

    fvgs: List[Dict[str, Any]] = []
    for row in d.tail(100).itertuples(index=False):
        if row.bull_fvg and not pd.isna(row.bull_fvg_lo):
            fvgs.append({"t": int(row.t), "type": "bullish",
                         "zone_hi": round(float(row.bull_fvg_hi), 6),
                         "zone_lo": round(float(row.bull_fvg_lo), 6)})
        if row.bear_fvg and not pd.isna(row.bear_fvg_hi):
            fvgs.append({"t": int(row.t), "type": "bearish",
                         "zone_hi": round(float(row.bear_fvg_hi), 6),
                         "zone_lo": round(float(row.bear_fvg_lo), 6)})

    sweeps: List[Dict[str, Any]] = []
    for row in d.tail(100).itertuples(index=False):
        if row.bsl_sweep and not pd.isna(row.last_ph):
            sweeps.append({"t": int(row.t), "type": "BSL", "level": round(float(row.last_ph), 6)})
        if row.ssl_sweep and not pd.isna(row.last_pl):
            sweeps.append({"t": int(row.t), "type": "SSL", "level": round(float(row.last_pl), 6)})

    mss_list: List[Dict[str, Any]] = []
    for row in d.tail(100).itertuples(index=False):
        if row.bull_mss:
            mss_list.append({"t": int(row.t), "type": "bullish"})
        if row.bear_mss:
            mss_list.append({"t": int(row.t), "type": "bearish"})

    latest = d.iloc[-1]
    return {
        "symbol": DEFAULT_INSTRUMENT,
        "timestamp": int(latest["t"]),
        "kill_zone": latest["kz"],
        "in_kill_zone": bool(latest["in_kz"]),
        "last_pivot_high": round(float(latest["last_ph"]), 6) if pd.notna(latest["last_ph"]) else None,
        "last_pivot_low":  round(float(latest["last_pl"]), 6) if pd.notna(latest["last_pl"]) else None,
        "recent_signals": signals[-10:],
        "recent_fvgs":    fvgs[-20:],
        "recent_sweeps":  sweeps[-20:],
        "recent_mss":     mss_list[-20:],
    }


# =============================
# Ingestion scheduler
# =============================

def ingest_once(symbol: str) -> Dict[str, Any]:
    candles = oanda_get_candles(symbol, "M1", count=500)
    df = parse_oanda_candles(candles)
    n = upsert_1m(symbol, df)
    return {"ok": True, "inserted": n, "latest_ts": latest_ts(symbol)}

def ensure_seed_history(symbol: str) -> Dict[str, Any]:
    target_start = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    to_time = datetime.now(timezone.utc)

    total = 0
    for _ in range(200):
        candles = oanda_get_candles(symbol, "M1", count=500, to_rfc3339=to_time.isoformat())
        df = parse_oanda_candles(candles)
        if df.empty:
            break
        total += upsert_1m(symbol, df)
        oldest = int(df["t"].min())
        if datetime.fromtimestamp(oldest, tz=timezone.utc) <= target_start:
            break
        to_time = datetime.fromtimestamp(oldest - 60, tz=timezone.utc)
        time.sleep(0.2)
    return {"ok": True, "seeded": total}

scheduler = BackgroundScheduler(daemon=True)

def scheduled_ingest():
    try:
        ingest_once(DEFAULT_INSTRUMENT)
    except Exception:
        pass

# =============================
# API
# =============================

@app.on_event("startup")
def _startup():
    init_db()
    try:
        ensure_seed_history(DEFAULT_INSTRUMENT)
    except Exception:
        pass
    scheduler.add_job(scheduled_ingest, "interval", seconds=INGEST_EVERY_SECONDS, id="ingest")
    scheduler.start()

@app.get("/api/health")
def health():
    return {"ok": True, "instrument": DEFAULT_INSTRUMENT, "latest_ts": latest_ts(DEFAULT_INSTRUMENT)}

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_INDEX_HTML)

@app.get("/api/history")
def history(
    symbol: str = Query(DEFAULT_INSTRUMENT),
    tf: str = Query("1m"),
    minutes: int = Query(24*60, ge=60, le=60*24*30),
):
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - minutes * 60
    df1 = load_1m(symbol, start_ts, end_ts)
    if df1.empty:
        return {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}

    df = resample_ohlcv(df1, tf) if tf != "1m" else df1

    return {
        "t": df["t"].astype(int).tolist(),
        "o": df["o"].astype(float).tolist(),
        "h": df["h"].astype(float).tolist(),
        "l": df["l"].astype(float).tolist(),
        "c": df["c"].astype(float).tolist(),
        "v": df["v"].astype(float).tolist(),
    }

@app.get("/api/signal")
def signal(symbol: str = Query(DEFAULT_INSTRUMENT)):
    return infer_signal(symbol)

@app.post("/api/admin/ingest")
def admin_ingest(x_admin_key: Optional[str] = Header(default=None), symbol: str = Query(DEFAULT_INSTRUMENT)):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return ingest_once(symbol)

@app.post("/api/admin/train")
def admin_train(x_admin_key: Optional[str] = Header(default=None), symbol: str = Query(DEFAULT_INSTRUMENT)):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return train_model(symbol)

@app.get("/api/ict")
def ict(
    symbol: str = Query(DEFAULT_INSTRUMENT),
    swing_len: int = Query(5, ge=2, le=20, description="Pivot swing length"),
    min_rr: float = Query(2.0, ge=1.0, description="Minimum risk:reward"),
):
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())
    df = load_1m(symbol, start_ts, end_ts)
    if df.empty:
        raise HTTPException(status_code=503, detail="No data yet — wait for ingestion to complete.")
    return compute_ict(df, swing_len=swing_len, min_rr=min_rr)

# =============================
# Frontend (Lightweight Charts via CDN)
# =============================

_INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>FX Market Brain</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:#fafafa; color:#222; }
    #top  { padding:10px 14px; display:flex; gap:8px; align-items:center; flex-wrap:wrap;
            background:#fff; border-bottom:1px solid #e5e5e5; }
    #chart { height: 62vh; }
    .pill { padding:5px 10px; border:1px solid #ddd; border-radius:999px; font-size:13px; }
    button { padding:7px 12px; border-radius:8px; border:1px solid #ddd; background:#fff; cursor:pointer; }
    button:hover { background:#f5f5f5; }
    #ict-panel { display:flex; gap:10px; flex-wrap:wrap; padding:10px 14px; background:#fff;
                 border-top:1px solid #e5e5e5; }
    .card { padding:10px 14px; border:1px solid #eee; border-radius:10px; background:#fafafa;
            min-width:180px; font-size:12px; line-height:1.6; }
    .card h4 { margin:0 0 4px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#888; }
    .bull { color:#2e7d32; } .bear { color:#c62828; } .neu { color:#555; }
  </style>
</head>
<body>
  <div id="top">
    <div class="pill"><b>FX Market Brain</b></div>
    <label>TF:
      <select id="tf">
        <option>1m</option><option>5m</option><option>15m</option><option>1h</option><option>4h</option><option>1d</option>
      </select>
    </label>
    <label>Lookback:
      <select id="mins">
        <option value="240">4h</option>
        <option value="1440" selected>1d</option>
        <option value="4320">3d</option>
        <option value="10080">7d</option>
        <option value="43200">30d</option>
      </select>
    </label>
    <button id="refresh">Refresh</button>
    <div id="sig"  class="pill">ML Signal: …</div>
    <div id="kz-badge" class="pill neu">KZ: —</div>
    <div id="ict-sig-badge" class="pill neu">ICT: —</div>
  </div>

  <div id="chart"></div>

  <div id="ict-panel">
    <div class="card" id="card-signal"><h4>Latest ICT Signal</h4><span class="neu">—</span></div>
    <div class="card" id="card-sweeps"><h4>Recent Sweeps</h4><span class="neu">—</span></div>
    <div class="card" id="card-mss"   ><h4>Recent MSS</h4>   <span class="neu">—</span></div>
    <div class="card" id="card-fvg"   ><h4>Recent FVGs</h4>  <span class="neu">—</span></div>
    <div class="card" id="card-pivot" ><h4>Pivot Levels</h4> <span class="neu">—</span></div>
  </div>

<script>
const el = document.getElementById('chart');
const chart = LightweightCharts.createChart(el, {
  width: el.clientWidth, height: el.clientHeight,
  layout: { background: { color: '#fff' }, textColor: '#333' },
  grid: { vertLines: { color: '#f0f0f0' }, horzLines: { color: '#f0f0f0' } },
});
const series  = chart.addCandlestickSeries();
const slLine  = chart.addLineSeries({ color: '#ef5350', lineWidth: 1, lineStyle: 2 });
const tpLine  = chart.addLineSeries({ color: '#26a69a', lineWidth: 1, lineStyle: 2 });
let fvgLines  = [];

function toBars(data) {
  return data.t.map((t,i) => ({ time: t, open: data.o[i], high: data.h[i], low: data.l[i], close: data.c[i] }));
}
function fmt(ts) { return new Date(ts*1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}); }

async function load() {
  const tf      = document.getElementById('tf').value;
  const minutes = document.getElementById('mins').value;
  const [h, s]  = await Promise.all([
    fetch(`/api/history?tf=${tf}&minutes=${minutes}`).then(r => r.json()),
    fetch('/api/signal').then(r => r.json()),
  ]);

  if (h.t && h.t.length) {
    series.setData(toBars(h));
    const t0 = h.t[0], tN = h.t[h.t.length - 1];
    slLine.setData(s.sl ? [{time:t0,value:s.sl},{time:tN,value:s.sl}] : []);
    tpLine.setData(s.tp ? [{time:t0,value:s.tp},{time:tN,value:s.tp}] : []);
  }

  const sig = document.getElementById('sig');
  const cls = s.side === 'LONG' ? 'bull' : s.side === 'SHORT' ? 'bear' : 'neu';
  sig.innerHTML = `ML: <b class="${cls}">${s.side}</b> conf=${s.confidence} entry=${s.entry}`;
}

async function loadIct() {
  let ict;
  try { ict = await fetch('/api/ict').then(r => r.json()); }
  catch (_) { return; }
  if (ict.error) return;

  // ── KZ badge ──────────────────────────────────────────────────────────────
  const kzEl = document.getElementById('kz-badge');
  const kzNames = { london: 'London KZ', ny_am: 'NY AM KZ', ny_pm: 'NY PM KZ' };
  if (ict.in_kill_zone) {
    kzEl.textContent = kzNames[ict.kill_zone] || ict.kill_zone;
    kzEl.style.background = '#e8f5e9'; kzEl.style.borderColor = '#a5d6a7';
  } else {
    kzEl.textContent = 'Off-session'; kzEl.style.background = ''; kzEl.style.borderColor = '#ddd';
  }

  // ── ICT signal badge ───────────────────────────────────────────────────────
  const ictBadge = document.getElementById('ict-sig-badge');
  const lastSig = ict.recent_signals && ict.recent_signals.length
    ? ict.recent_signals[ict.recent_signals.length - 1] : null;
  if (lastSig) {
    const cls = lastSig.side === 'LONG' ? 'bull' : 'bear';
    ictBadge.innerHTML = `ICT: <b class="${cls}">${lastSig.side}</b> @ ${fmt(lastSig.t)}`;
    ictBadge.style.background = lastSig.side === 'LONG' ? '#e8f5e9' : '#ffebee';
  } else {
    ictBadge.textContent = 'ICT: no signal'; ictBadge.style.background = '';
  }

  // ── FVG price lines on chart ───────────────────────────────────────────────
  fvgLines.forEach(l => { try { series.removePriceLine(l); } catch(_) {} });
  fvgLines = [];
  (ict.recent_fvgs || []).slice(-6).forEach(f => {
    const col = f.type === 'bullish' ? 'rgba(46,125,50,0.55)' : 'rgba(198,40,40,0.55)';
    const lbl = f.type === 'bullish' ? 'FVG ↑' : 'FVG ↓';
    fvgLines.push(series.createPriceLine({ price: f.zone_hi, color: col, lineWidth: 1, lineStyle: 2, title: lbl }));
    fvgLines.push(series.createPriceLine({ price: f.zone_lo, color: col, lineWidth: 1, lineStyle: 2, title: '' }));
  });

  // ── Latest signal card ─────────────────────────────────────────────────────
  const cSig = document.getElementById('card-signal');
  if (lastSig) {
    const cls = lastSig.side === 'LONG' ? 'bull' : 'bear';
    cSig.innerHTML = `<h4>Latest ICT Signal</h4>
      <b class="${cls}">${lastSig.side}</b> @ ${fmt(lastSig.t)} [${lastSig.kz||'—'}]<br>
      Entry: ${lastSig.entry}<br>SL: ${lastSig.sl} &nbsp; TP: ${lastSig.tp}`;
  } else {
    cSig.innerHTML = '<h4>Latest ICT Signal</h4><span class="neu">No signal in recent 200 bars</span>';
  }

  // ── Sweeps card ────────────────────────────────────────────────────────────
  const cSweep = document.getElementById('card-sweeps');
  const sweeps = (ict.recent_sweeps || []).slice(-4);
  cSweep.innerHTML = '<h4>Recent Sweeps</h4>' + (sweeps.length
    ? sweeps.map(s => `<span class="${s.type==='BSL'?'bear':'bull'}">${s.type}</span> @ ${fmt(s.t)} (${s.level})`).join('<br>')
    : '<span class="neu">none</span>');

  // ── MSS card ───────────────────────────────────────────────────────────────
  const cMss = document.getElementById('card-mss');
  const mssList = (ict.recent_mss || []).slice(-4);
  cMss.innerHTML = '<h4>Recent MSS</h4>' + (mssList.length
    ? mssList.map(m => `<span class="${m.type==='bullish'?'bull':'bear'}">${m.type==='bullish'?'▲':'▼'} MSS</span> @ ${fmt(m.t)}`).join('<br>')
    : '<span class="neu">none</span>');

  // ── FVG card ───────────────────────────────────────────────────────────────
  const cFvg = document.getElementById('card-fvg');
  const fvgs = (ict.recent_fvgs || []).slice(-4);
  cFvg.innerHTML = '<h4>Recent FVGs</h4>' + (fvgs.length
    ? fvgs.map(f => `<span class="${f.type==='bullish'?'bull':'bear'}">${f.type==='bullish'?'▲':'▼'}</span> ${f.zone_lo}–${f.zone_hi} @ ${fmt(f.t)}`).join('<br>')
    : '<span class="neu">none</span>');

  // ── Pivot levels card ──────────────────────────────────────────────────────
  const cPivot = document.getElementById('card-pivot');
  cPivot.innerHTML = `<h4>Pivot Levels</h4>
    <span class="bear">Last PH:</span> ${ict.last_pivot_high ?? '—'}<br>
    <span class="bull">Last PL:</span> ${ict.last_pivot_low  ?? '—'}`;
}

async function refresh() {
  await Promise.all([load(), loadIct()]);
}

document.getElementById('refresh').onclick = refresh;
window.addEventListener('resize', () => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""