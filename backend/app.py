import os
import json
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.price_action import full_analysis as pa_full_analysis
from backend import ai_analyst

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

DB_PATH              = os.getenv("DB_PATH", "/tmp/fx.db")
OANDA_TOKEN          = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT_ID     = os.getenv("OANDA_ACCOUNT_ID", "")
ADMIN_KEY            = os.getenv("ADMIN_KEY", "")
DEFAULT_INSTRUMENT   = os.getenv("DEFAULT_INSTRUMENT", "EUR_USD")
INGEST_EVERY_SECONDS = int(os.getenv("INGEST_EVERY_SECONDS", "60"))
HISTORY_DAYS         = int(os.getenv("HISTORY_DAYS", "30"))

OANDA_API_BASE = "https://api-fxpractice.oanda.com/v3"

app = FastAPI(title="FX Market Brain — Price Action AI")

# ═══════════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════════

def db() -> sqlite3.Connection:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
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

# ═══════════════════════════════════════════════════════════════════
# OANDA client
# ═══════════════════════════════════════════════════════════════════

def oanda_headers() -> Dict[str, str]:
    if not OANDA_TOKEN:
        raise RuntimeError("Missing OANDA_TOKEN")
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def oanda_get_candles(symbol: str, granularity: str, count: int = 500,
                      to_rfc3339: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"granularity": granularity, "price": "M", "count": str(count)}
    if to_rfc3339:
        params["to"] = to_rfc3339
    url = f"{OANDA_API_BASE}/instruments/{symbol}/candles"
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA error {r.status_code}: {r.text[:500]}")
    return r.json().get("candles", [])

def parse_oanda_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for x in candles:
        if not x.get("complete"):
            continue
        dt = datetime.fromisoformat(x["time"].replace("Z", "+00:00"))
        mid = x["mid"]
        rows.append({
            "t": int(dt.timestamp()),
            "o": float(mid["o"]), "h": float(mid["h"]),
            "l": float(mid["l"]), "c": float(mid["c"]),
            "v": float(x.get("volume", 0.0)),
        })
    if not rows:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    return pd.DataFrame(rows).drop_duplicates(subset=["t"]).sort_values("t")

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
    df = pd.read_sql_query(
        "SELECT t,o,h,l,c,v FROM candles_1m WHERE symbol=? AND t BETWEEN ? AND ? ORDER BY t ASC",
        conn, params=(symbol, start_ts, end_ts),
    )
    conn.close()
    return df

# ═══════════════════════════════════════════════════════════════════
# Multi-timeframe resampling (derived from 1m base data)
# ═══════════════════════════════════════════════════════════════════

TF_MAP = {
    "1m": "1min", "5m": "5min", "15m": "15min",
    "1h": "1h",   "4h": "4h",   "1d":  "1D",
}

def resample_ohlcv(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df_1m.empty or tf not in TF_MAP:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    d = df_1m.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt")
    rule = TF_MAP[tf]
    out = pd.DataFrame({
        "o": d["o"].resample(rule).first(),
        "h": d["h"].resample(rule).max(),
        "l": d["l"].resample(rule).min(),
        "c": d["c"].resample(rule).last(),
        "v": d["v"].resample(rule).sum(),
    }).dropna()
    out["t"] = out.index.astype("int64") // 10**9
    return out.reset_index(drop=True)[["t","o","h","l","c","v"]]

def _build_df_map(symbol: str) -> Dict[str, pd.DataFrame]:
    """Load 1m data and resample into all timeframes."""
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df_1m    = load_1m(symbol, start_ts, end_ts)
    df_map: Dict[str, pd.DataFrame] = {}
    for tf in TF_MAP:
        df_map[tf] = df_1m if tf == "1m" else resample_ohlcv(df_1m, tf)
    return df_map

# ═══════════════════════════════════════════════════════════════════
# Legacy ML signal (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════

_model: Optional[HistGradientBoostingClassifier] = None
_model_meta: Dict[str, Any] = {}

def build_features(df_1m: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df_1m.shape[0] < 2000:
        return None

    def add_tf_feats(tf_name: str, window: int) -> pd.DataFrame:
        d = resample_ohlcv(df_1m, tf_name)
        if d.shape[0] < window + 50:
            return pd.DataFrame()
        d["ret1"] = d["c"].pct_change()
        d["vol"]  = d["ret1"].rolling(window).std()
        d["mom"]  = d["c"].pct_change(window)
        d["rng"]  = (d["h"] - d["l"]) / d["c"].replace(0, np.nan)
        roll_hi = d["h"].rolling(window).max()
        roll_lo = d["l"].rolling(window).min()
        d["pos"]  = (d["c"] - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
        d = d[["t","ret1","vol","mom","rng","pos"]].dropna()
        return d.rename(columns={c: f"{tf_name}_{c}" for c in d.columns if c != "t"})

    feats = None
    for tf_name, w in [("1m",60),("5m",60),("15m",60),("1h",60),("4h",60),("1d",30)]:
        f = add_tf_feats(tf_name, w)
        if f.empty:
            continue
        feats = f if feats is None else feats.merge(f, on="t", how="inner")

    if feats is None or feats.shape[0] < 500:
        return None

    H = 60
    base = df_1m[["t","c"]].copy()
    base["fut_c"] = base["c"].shift(-H)
    base["y"] = (base["fut_c"] > base["c"]).astype(int)
    return feats.merge(base[["t","y"]].dropna(), on="t", how="inner").dropna()

def train_model(symbol: str) -> Dict[str, Any]:
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df = load_1m(symbol, start_ts, end_ts)
    feats = build_features(df)
    if feats is None:
        return {"ok": False, "reason": "Not enough data yet."}

    X = feats.drop(columns=["y"])
    y = feats["y"].astype(int)
    split = int(len(feats) * 0.8)
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300)
    model.fit(X.iloc[:split].drop(columns=["t"]), y.iloc[:split])
    acc = float(model.score(X.iloc[split:].drop(columns=["t"]), y.iloc[split:]))

    global _model, _model_meta
    _model = model
    _model_meta = {
        "symbol": symbol, "trained_at": int(time.time()),
        "rows": int(len(feats)), "test_acc": acc, "horizon_minutes": 60,
    }
    return {"ok": True, **_model_meta}

def infer_signal(symbol: str) -> Dict[str, Any]:
    global _model
    if _model is None:
        train_model(symbol)

    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df = load_1m(symbol, start_ts, end_ts)
    feats = build_features(df)
    if feats is None or feats.empty:
        return {"side": "NEUTRAL", "confidence": 0.0, "reason": "Not enough data yet."}

    latest   = feats.sort_values("t").iloc[-1:]
    proba_up = float(_model.predict_proba(latest.drop(columns=["y","t"]))[0, 1]) if _model else 0.5

    if proba_up >= 0.55:
        side, conf = "LONG",    (proba_up - 0.5) * 2.0
    elif proba_up <= 0.45:
        side, conf = "SHORT",   (0.5 - proba_up) * 2.0
    else:
        side, conf = "NEUTRAL", 0.0

    df_r = df.tail(2000).copy()
    df_r["ret1"] = df_r["c"].pct_change()
    vol  = float(df_r["ret1"].rolling(300).std().iloc[-1] or 0.0005)
    last = float(df_r["c"].iloc[-1])
    sl_d = max(2.5 * vol * last, 0.0008 * last)
    tp_d = max(4.0 * vol * last, 0.0012 * last)

    return {
        "symbol": symbol, "timestamp": int(df_r["t"].iloc[-1]),
        "side": side, "confidence": round(conf, 4), "prob_up": round(proba_up, 4),
        "entry": round(last, 6),
        "sl": round(last - sl_d if side=="LONG" else last + sl_d, 6) if side!="NEUTRAL" else None,
        "tp": round(last + tp_d if side=="LONG" else last - tp_d, 6) if side!="NEUTRAL" else None,
        "horizon_minutes": _model_meta.get("horizon_minutes", 60),
        "model": _model_meta,
    }

# ═══════════════════════════════════════════════════════════════════
# Ingestion scheduler
# ═══════════════════════════════════════════════════════════════════

def ingest_once(symbol: str) -> Dict[str, Any]:
    candles = oanda_get_candles(symbol, "M1", count=500)
    df = parse_oanda_candles(candles)
    n  = upsert_1m(symbol, df)
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

# ═══════════════════════════════════════════════════════════════════
# API routes
# ═══════════════════════════════════════════════════════════════════

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

@app.get("/api/history")
def history(
    symbol: str = Query(DEFAULT_INSTRUMENT),
    tf: str     = Query("15m"),
    minutes: int = Query(4320, ge=60, le=60*24*30),
):
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = end_ts - minutes * 60
    df_1m    = load_1m(symbol, start_ts, end_ts)
    if df_1m.empty:
        return {"t":[],"o":[],"h":[],"l":[],"c":[],"v":[]}
    df = resample_ohlcv(df_1m, tf) if tf != "1m" else df_1m
    return {
        "t": df["t"].astype(int).tolist(),
        "o": df["o"].astype(float).tolist(),
        "h": df["h"].astype(float).tolist(),
        "l": df["l"].astype(float).tolist(),
        "c": df["c"].astype(float).tolist(),
        "v": df["v"].astype(float).tolist(),
    }

@app.get("/api/price-action")
def price_action(symbol: str = Query(DEFAULT_INSTRUMENT)):
    """Compute pure price action analysis across all timeframes (no indicators)."""
    df_map = _build_df_map(symbol)
    if all(df.empty for df in df_map.values()):
        return {"error": "no_data"}
    return pa_full_analysis(df_map)

@app.get("/api/ai-analysis")
def ai_analysis(symbol: str = Query(DEFAULT_INSTRUMENT)):
    """Run Claude AI market analysis based purely on price action across all timeframes."""
    df_map = _build_df_map(symbol)
    if all(df.empty for df in df_map.values()):
        return {"error": "no_data", "bias": "neutral", "confidence": 0,
                "summary": "No price data available. Ensure OANDA_TOKEN is configured."}
    pa_data = pa_full_analysis(df_map)
    return ai_analyst.analyze(symbol, pa_data)

@app.get("/api/signal")
def signal(symbol: str = Query(DEFAULT_INSTRUMENT)):
    """Legacy ML signal endpoint."""
    return infer_signal(symbol)

@app.post("/api/admin/ingest")
def admin_ingest(x_admin_key: Optional[str] = Header(default=None),
                 symbol: str = Query(DEFAULT_INSTRUMENT)):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return ingest_once(symbol)

@app.post("/api/admin/train")
def admin_train(x_admin_key: Optional[str] = Header(default=None),
                symbol: str = Query(DEFAULT_INSTRUMENT)):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return train_model(symbol)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_INDEX_HTML)

# ═══════════════════════════════════════════════════════════════════
# Frontend — professional price action trading dashboard
# ═══════════════════════════════════════════════════════════════════

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FX Market Brain — Price Action AI</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:     #0d1117;
      --bg2:    #161b22;
      --bg3:    #21262d;
      --border: #30363d;
      --text:   #e6edf3;
      --muted:  #8b949e;
      --green:  #3fb950;
      --red:    #f85149;
      --blue:   #58a6ff;
      --yellow: #d29922;
      --purple: #bc8cff;
      --orange: #f0883e;
    }

    html, body {
      height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 13px;
      overflow: hidden;
    }

    #app { display: flex; flex-direction: column; height: 100vh; }

    /* ── Header ── */
    #header {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 14px;
      background: var(--bg2);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0; flex-wrap: wrap;
    }

    #logo { font-weight: 800; font-size: 13px; color: var(--blue); letter-spacing: -0.3px; white-space: nowrap; }

    .sep { width: 1px; height: 18px; background: var(--border); flex-shrink: 0; }

    #sym {
      background: var(--bg3); border: 1px solid var(--border);
      color: var(--text); padding: 4px 8px; border-radius: 6px;
      font-size: 12px; width: 88px; font-family: monospace; text-transform: uppercase;
    }
    #sym:focus { outline: none; border-color: var(--blue); }

    .tf-btns { display: flex; gap: 3px; }
    .tf-btn {
      background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
      padding: 4px 9px; border-radius: 5px; cursor: pointer;
      font-size: 11px; font-weight: 600; transition: all .12s;
    }
    .tf-btn:hover  { border-color: var(--blue); color: var(--blue); }
    .tf-btn.active { background: var(--blue); border-color: var(--blue); color: #fff; }

    .btn {
      background: var(--bg3); border: 1px solid var(--border); color: var(--text);
      padding: 5px 12px; border-radius: 6px; cursor: pointer;
      font-size: 12px; font-weight: 500; transition: all .12s; white-space: nowrap;
    }
    .btn:hover { border-color: var(--blue); color: var(--blue); }
    .btn:disabled { opacity: .45; cursor: not-allowed; }
    .btn.primary { background: var(--blue); border-color: var(--blue); color: #fff; font-weight: 600; }
    .btn.primary:hover { background: #79b8ff; border-color: #79b8ff; color: #fff; }

    #live-badge {
      display: flex; align-items: center; gap: 5px;
      font-size: 11px; color: var(--muted);
    }
    #live-dot {
      width: 7px; height: 7px; border-radius: 50%; background: var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    #last-up { margin-left: auto; color: var(--muted); font-size: 11px; white-space: nowrap; }

    /* ── Multi-TF structure strip ── */
    #tf-strip {
      display: flex; align-items: center; gap: 6px;
      padding: 5px 14px;
      background: var(--bg2); border-bottom: 1px solid var(--border);
      flex-shrink: 0; flex-wrap: wrap;
    }
    #tf-strip-label { color: var(--muted); font-size: 11px; font-weight: 600; letter-spacing: .5px; margin-right: 2px; }

    .tbadge {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;
      border: 1px solid; cursor: default;
    }
    .tbadge.uptrend     { background: rgba(63,185,80,.12);  border-color: var(--green);  color: var(--green); }
    .tbadge.downtrend   { background: rgba(248,81,73,.12);  border-color: var(--red);    color: var(--red); }
    .tbadge.range       { background: var(--bg3);           border-color: var(--border); color: var(--muted); }
    .tbadge.contraction { background: rgba(188,140,255,.12);border-color: var(--purple); color: var(--purple); }
    .tbadge.expansion   { background: rgba(210,153,34,.12); border-color: var(--yellow); color: var(--yellow); }
    .tbadge.unknown     { background: var(--bg3);           border-color: var(--border); color: var(--muted); }

    /* ── Main layout ── */
    #main { display: flex; flex: 1; overflow: hidden; min-height: 0; }

    /* ── Chart panel ── */
    #chart-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
    #chart { flex: 1; min-height: 0; }

    /* ── Analysis panel ── */
    #panel {
      width: 310px; min-width: 260px; max-width: 380px;
      border-left: 1px solid var(--border);
      overflow-y: auto; display: flex; flex-direction: column;
      background: var(--bg);
    }

    /* Panel sections */
    .sec { border-bottom: 1px solid var(--border); padding: 11px 12px; }
    .sec-title {
      font-size: 10px; font-weight: 700; letter-spacing: 1px;
      text-transform: uppercase; color: var(--muted); margin-bottom: 8px;
    }

    /* Bias card */
    #bias-card { padding: 14px 12px; border-bottom: 1px solid var(--border); text-align: center; }
    #bias-lbl  { font-size: 22px; font-weight: 900; letter-spacing: 3px; }
    .c-bull { color: var(--green); }
    .c-bear { color: var(--red); }
    .c-neut { color: var(--muted); }

    #conf-wrap { background: var(--bg3); border-radius: 4px; height: 5px; margin: 8px 0; overflow: hidden; }
    #conf-bar  { height: 100%; border-radius: 4px; transition: width .4s, background .4s; width: 0%; }
    #conf-num  { color: var(--muted); font-size: 11px; }

    #bias-sum {
      text-align: left; margin-top: 8px;
      color: var(--muted); font-size: 12px; line-height: 1.55;
    }

    /* Key levels */
    .lvl-row {
      display: flex; align-items: center; gap: 7px;
      padding: 4px 0; border-bottom: 1px solid rgba(48,54,61,.5);
      font-size: 12px;
    }
    .lvl-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
    .lvl-p   { font-family: monospace; font-size: 12px; min-width: 64px; }
    .lvl-r   { color: var(--muted); font-size: 11px; }
    .lvl-why { padding: 0 0 5px 15px; color: var(--muted); font-size: 10px; line-height: 1.4; }
    .lvl-s   { margin-left: auto; font-size: 10px; color: var(--muted); flex-shrink: 0; }

    /* Prediction */
    #pred-box  { background: var(--bg3); border-radius: 6px; padding: 10px; }
    #pred-move { line-height: 1.55; font-size: 12px; margin-bottom: 6px; }
    #pred-inv  { color: var(--muted); font-size: 11px; line-height: 1.4; }
    #pred-tgt  { margin-bottom: 4px; }
    .tgt-tag   { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 700; font-family: monospace; }

    /* Education */
    #edu-box {
      background: rgba(88,166,255,.07);
      border: 1px solid rgba(88,166,255,.2);
      border-radius: 6px; padding: 10px;
    }
    #edu-concept { color: var(--blue); font-weight: 700; font-size: 12px; margin-bottom: 5px; }
    #edu-text    { color: var(--text); font-size: 12px; line-height: 1.6; }

    /* Zones */
    .zone-row {
      display: flex; align-items: center; gap: 6px;
      padding: 3px 0; font-size: 11px;
    }
    .zone-tag {
      padding: 1px 6px; border-radius: 3px;
      font-size: 10px; font-weight: 700; white-space: nowrap; flex-shrink: 0;
    }
    .t-bull-ob { background: rgba(63,185,80,.2);  color: var(--green); }
    .t-bear-ob { background: rgba(248,81,73,.2);  color: var(--red); }
    .t-bull-fv { background: rgba(88,166,255,.2); color: var(--blue); }
    .t-bear-fv { background: rgba(210,153,34,.2); color: var(--yellow); }
    .zone-p    { font-family: monospace; font-size: 11px; }

    /* TF breakdown tabs */
    #tf-tabs { display: flex; gap: 2px; flex-wrap: wrap; margin-bottom: 8px; }
    .tf-tab {
      padding: 3px 7px; border-radius: 4px; font-size: 10px; font-weight: 600;
      cursor: pointer; background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
      transition: all .1s;
    }
    .tf-tab:hover  { border-color: var(--blue); color: var(--blue); }
    .tf-tab.active { background: var(--bg3); border-color: var(--blue); color: var(--blue); }
    #tf-content    { color: var(--text); font-size: 12px; line-height: 1.6; min-height: 40px; }

    /* Detailed analysis */
    #det-text { color: var(--muted); font-size: 12px; line-height: 1.7; }

    /* Scrollbar */
    ::-webkit-scrollbar       { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    .loading { color: var(--muted); font-style: italic; font-size: 12px; }

    @media (max-width: 700px) {
      #panel { width: 100%; max-width: unset; border-left: none; border-top: 1px solid var(--border); }
      #main  { flex-direction: column; }
      #chart { min-height: 50vh; }
    }
  </style>
</head>
<body>
<div id="app">

  <!-- Header -->
  <div id="header">
    <span id="logo">▪ FX MARKET BRAIN</span>
    <div class="sep"></div>
    <input id="sym" type="text" value="EUR_USD" placeholder="EUR_USD" title="Press Enter or click Analyze"/>
    <div class="sep"></div>
    <div class="tf-btns">
      <button class="tf-btn" data-tf="1m">1M</button>
      <button class="tf-btn" data-tf="5m">5M</button>
      <button class="tf-btn active" data-tf="15m">15M</button>
      <button class="tf-btn" data-tf="1h">1H</button>
      <button class="tf-btn" data-tf="4h">4H</button>
      <button class="tf-btn" data-tf="1d">1D</button>
    </div>
    <div class="sep"></div>
    <button class="btn primary" id="btn-analyze">🧠 Analyze</button>
    <button class="btn" id="btn-refresh">↺ Chart</button>
    <div id="live-badge"><div id="live-dot"></div><span>LIVE</span></div>
    <span id="last-up">—</span>
  </div>

  <!-- Multi-TF structure strip -->
  <div id="tf-strip">
    <span id="tf-strip-label">STRUCTURE</span>
    <span class="loading">Load data to see multi-TF structure...</span>
  </div>

  <!-- Main area -->
  <div id="main">

    <!-- Chart -->
    <div id="chart-panel">
      <div id="chart"></div>
    </div>

    <!-- Analysis panel -->
    <div id="panel">

      <!-- Bias card -->
      <div id="bias-card">
        <div id="bias-lbl" class="c-neut">—</div>
        <div id="conf-wrap"><div id="conf-bar"></div></div>
        <div id="conf-num">Confidence: —</div>
        <div id="bias-sum" class="loading">Click "Analyze" to run AI price action analysis across all timeframes...</div>
      </div>

      <!-- Key levels -->
      <div class="sec">
        <div class="sec-title">Key Levels</div>
        <div id="levels-list"><span class="loading">—</span></div>
      </div>

      <!-- Prediction -->
      <div class="sec">
        <div class="sec-title">Prediction</div>
        <div id="pred-box">
          <div id="pred-tgt"></div>
          <div id="pred-move" class="loading">—</div>
          <div id="pred-inv"></div>
        </div>
      </div>

      <!-- Education -->
      <div class="sec">
        <div class="sec-title">📚 Learn Price Action</div>
        <div id="edu-box">
          <div id="edu-concept">Waiting for analysis...</div>
          <div id="edu-text" style="color:var(--muted);">The AI will explain what price is doing right now and WHY — teaching you to read the market yourself, without any indicators.</div>
        </div>
      </div>

      <!-- TF breakdown -->
      <div class="sec">
        <div class="sec-title">Timeframe Breakdown</div>
        <div id="tf-tabs">
          <div class="tf-tab active" data-tf="1d">1D</div>
          <div class="tf-tab" data-tf="4h">4H</div>
          <div class="tf-tab" data-tf="1h">1H</div>
          <div class="tf-tab" data-tf="15m">15M</div>
          <div class="tf-tab" data-tf="5m">5M</div>
          <div class="tf-tab" data-tf="1m">1M</div>
        </div>
        <div id="tf-content" class="loading">—</div>
      </div>

      <!-- Active zones (OBs + FVGs) -->
      <div class="sec">
        <div class="sec-title">Active Zones on <span id="zones-tf">—</span></div>
        <div id="zones-list"><span class="loading">—</span></div>
      </div>

      <!-- Detailed analysis -->
      <div class="sec">
        <div class="sec-title">Detailed Analysis</div>
        <div id="det-text" class="loading">—</div>
      </div>

    </div>
  </div>
</div>

<script>
// ════════════════════════════════════════════════════════════════
// Chart setup
// ════════════════════════════════════════════════════════════════
const chartEl = document.getElementById('chart');
let chart, candleSeries;
let priceLinesActive = [];

function initChart() {
  chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout: { background: { color: '#0d1117' }, textColor: '#e6edf3' },
    grid:   { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    crosshair: { mode: 1 },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });

  candleSeries = chart.addCandlestickSeries({
    upColor:        '#3fb950', downColor:      '#f85149',
    borderUpColor:  '#3fb950', borderDownColor:'#f85149',
    wickUpColor:    '#3fb950', wickDownColor:  '#f85149',
  });
}

function clearOverlays() {
  priceLinesActive.forEach(pl => { try { candleSeries.removePriceLine(pl); } catch(e) {} });
  priceLinesActive = [];
  candleSeries.setMarkers([]);
}

function addLine(price, color, title, style = 1, width = 1) {
  if (!price || isNaN(price)) return;
  try {
    const pl = candleSeries.createPriceLine({ price, color, lineStyle: style, lineWidth: width, axisLabelVisible: true, title });
    priceLinesActive.push(pl);
  } catch(e) {}
}

// ════════════════════════════════════════════════════════════════
// State
// ════════════════════════════════════════════════════════════════
let currentTf  = '15m';
let currentSym = 'EUR_USD';
let paData     = null;
let aiData     = null;
let activeTfTab = '1d';
let tfBreakdown = {};

// ════════════════════════════════════════════════════════════════
// Chart data loading
// ════════════════════════════════════════════════════════════════
const minutesMap = { '1m':240, '5m':1440, '15m':4320, '1h':10080, '4h':43200, '1d':43200 };

async function loadChart() {
  const sym  = getSym();
  const mins = minutesMap[currentTf] || 4320;
  try {
    const data = await fetch(`/api/history?symbol=${sym}&tf=${currentTf}&minutes=${mins}`).then(r => r.json());
    if (!data.t || !data.t.length) return;
    const bars = data.t.map((t,i) => ({ time:t, open:data.o[i], high:data.h[i], low:data.l[i], close:data.c[i] }));
    candleSeries.setData(bars);
    chart.timeScale().fitContent();
    document.getElementById('last-up').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { console.error('Chart load:', e); }
}

// ════════════════════════════════════════════════════════════════
// Price action data
// ════════════════════════════════════════════════════════════════
async function loadPA() {
  const sym = getSym();
  try {
    paData = await fetch(`/api/price-action?symbol=${sym}`).then(r => r.json());
    renderTfStrip(paData);
    renderZones(paData, currentTf);
    renderZonesPanel(paData, currentTf);
    renderOverlays(paData, currentTf);
  } catch(e) { console.error('PA load:', e); }
}

// ════════════════════════════════════════════════════════════════
// AI analysis
// ════════════════════════════════════════════════════════════════
async function runAnalysis() {
  const sym = getSym();
  const btn = document.getElementById('btn-analyze');
  btn.disabled = true; btn.textContent = '⏳ Analyzing…';
  document.getElementById('bias-sum').textContent = 'Running AI analysis across all timeframes…';

  try {
    await loadPA();
    aiData = await fetch(`/api/ai-analysis?symbol=${sym}`).then(r => r.json());
    renderAIPanel(aiData);
  } catch(e) {
    document.getElementById('bias-sum').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '🧠 Analyze';
  }
}

// ════════════════════════════════════════════════════════════════
// Render: multi-TF structure strip
// ════════════════════════════════════════════════════════════════
const TREND_ICON  = { uptrend:'▲', downtrend:'▼', range:'↔', contraction:'◆', expansion:'↕', unknown:'?' };
const TREND_LABEL = { uptrend:'UP', downtrend:'DN', range:'RNG', contraction:'SQZ', expansion:'EXP', unknown:'?' };

function renderTfStrip(pa) {
  const strip = document.getElementById('tf-strip');
  const tfs   = ['1d','4h','1h','15m','5m','1m'];
  let html = '<span id="tf-strip-label">STRUCTURE</span>';
  for (const tf of tfs) {
    const d = pa[tf];
    if (!d || d.error) { html += `<span class="tbadge unknown">${tf.toUpperCase()} ?</span>`; continue; }
    const trend = d.structure?.trend || 'unknown';
    html += `<span class="tbadge ${trend}" title="${d.structure?.desc || ''}">${tf.toUpperCase()} ${TREND_ICON[trend]||'?'} ${TREND_LABEL[trend]||'?'}</span>`;
  }
  strip.innerHTML = html;
}

// ════════════════════════════════════════════════════════════════
// Render: chart overlays (S/R lines, patterns, liquidity)
// ════════════════════════════════════════════════════════════════
function renderOverlays(pa, tf) {
  clearOverlays();
  if (!pa) return;
  const d = pa[tf];
  if (!d || d.error) return;

  // S/R zones — solid green=support, solid red=resistance; dashed if moderate
  (d.sr_zones || []).forEach(z => {
    const col   = z.role === 'resistance' ? '#f85149' : '#3fb950';
    const style = z.strength === 'strong' ? 0 : 2;
    addLine(z.price, col, `${z.strength === 'strong' ? '●' : '○'} ${z.role.toUpperCase()}`, style, 1);
  });

  // OB zones — two dashed lines per zone
  (d.order_blocks || []).forEach(ob => {
    const col = ob.kind === 'bullish_ob' ? '#3fb95066' : '#f8514966';
    addLine(ob.top,    col, ob.kind === 'bullish_ob' ? 'OB↑ top' : 'OB↓ top', 2, 1);
    addLine(ob.bottom, col, ob.kind === 'bullish_ob' ? 'OB↑ bot' : 'OB↓ bot', 2, 1);
  });

  // FVG boundaries — dotted blue/yellow lines
  (d.fvgs || []).forEach(fvg => {
    const col = fvg.kind === 'bullish_fvg' ? '#58a6ff55' : '#d2992255';
    addLine(fvg.top,    col, fvg.kind === 'bullish_fvg' ? 'FVG↑ top' : 'FVG↓ top', 3, 1);
    addLine(fvg.bottom, col, fvg.kind === 'bullish_fvg' ? 'FVG↑ bot' : 'FVG↓ bot', 3, 1);
  });

  // Liquidity — purple dotted
  const liq = d.liquidity || {};
  if (liq.bsl) addLine(liq.bsl, '#bc8cff77', 'BSL', 3, 1);
  if (liq.ssl) addLine(liq.ssl, '#bc8cff77', 'SSL', 3, 1);

  // Candlestick pattern markers
  const patterns = (d.patterns || []);
  if (patterns.length) {
    const markers = patterns.map(p => ({
      time:     p.t,
      position: p.dir === 'bearish' ? 'aboveBar' : 'belowBar',
      color:    p.dir === 'bullish' ? '#3fb950' : p.dir === 'bearish' ? '#f85149' : '#8b949e',
      shape:    p.dir === 'bullish' ? 'arrowUp'  : p.dir === 'bearish' ? 'arrowDown' : 'circle',
      text:     p.name,
    }));
    try { candleSeries.setMarkers(markers.sort((a,b) => a.time - b.time)); } catch(e) {}
  }
}

// ════════════════════════════════════════════════════════════════
// Render: zones panel (OBs + FVGs for selected TF)
// ════════════════════════════════════════════════════════════════
function renderZones(pa, tf) { renderZonesPanel(pa, tf); }

function renderZonesPanel(pa, tf) {
  document.getElementById('zones-tf').textContent = tf.toUpperCase();
  if (!pa) return;
  const d = pa[tf];
  if (!d || d.error) { document.getElementById('zones-list').innerHTML = '<span class="loading">No data</span>'; return; }

  const obs  = d.order_blocks || [];
  const fvgs = d.fvgs || [];
  let html = '';

  obs.forEach(ob => {
    const cls = ob.kind === 'bullish_ob' ? 't-bull-ob' : 't-bear-ob';
    const lbl = ob.kind === 'bullish_ob' ? 'Bull OB' : 'Bear OB';
    html += `<div class="zone-row">
      <span class="zone-tag ${cls}">${lbl}</span>
      <span class="zone-p">${ob.bottom.toFixed(5)}–${ob.top.toFixed(5)}</span>
    </div>`;
  });

  fvgs.forEach(fvg => {
    const cls = fvg.kind === 'bullish_fvg' ? 't-bull-fv' : 't-bear-fv';
    const lbl = fvg.kind === 'bullish_fvg' ? 'Bull FVG' : 'Bear FVG';
    html += `<div class="zone-row">
      <span class="zone-tag ${cls}">${lbl}</span>
      <span class="zone-p">${fvg.bottom.toFixed(5)}–${fvg.top.toFixed(5)}</span>
    </div>`;
  });

  document.getElementById('zones-list').innerHTML = html || '<span class="loading">No active zones on this TF</span>';
}

// ════════════════════════════════════════════════════════════════
// Render: AI analysis panel
// ════════════════════════════════════════════════════════════════
function renderAIPanel(ai) {
  if (!ai) return;

  const bias = (ai.bias || 'neutral').toLowerCase();
  const conf = ai.confidence || 0;

  // Bias label
  const lbl = document.getElementById('bias-lbl');
  lbl.textContent = bias.toUpperCase();
  lbl.className = bias === 'bullish' ? 'c-bull' : bias === 'bearish' ? 'c-bear' : 'c-neut';

  // Confidence bar
  const bar = document.getElementById('conf-bar');
  bar.style.width = conf + '%';
  bar.style.background = bias === 'bullish' ? '#3fb950' : bias === 'bearish' ? '#f85149' : '#8b949e';
  document.getElementById('conf-num').textContent = `Confidence: ${conf}%`;
  document.getElementById('bias-sum').textContent = ai.summary || '';

  // Key levels
  const levels = ai.key_levels || [];
  let lvlHtml = '';
  levels.forEach(l => {
    const col = l.role === 'resistance' ? '#f85149' : '#3fb950';
    lvlHtml += `
      <div class="lvl-row">
        <div class="lvl-dot" style="background:${col}"></div>
        <span class="lvl-p">${typeof l.price === 'number' ? l.price.toFixed(5) : l.price}</span>
        <span class="lvl-r">${l.role}</span>
        <span class="lvl-s">${l.strength || ''}</span>
      </div>
      <div class="lvl-why">${l.reason || ''}</div>`;
    // Draw on chart too
    try { addLine(l.price, col, l.role.toUpperCase(), l.strength === 'strong' ? 0 : 2, l.strength === 'strong' ? 2 : 1); } catch(e) {}
  });
  document.getElementById('levels-list').innerHTML = lvlHtml || '<span class="loading">No key levels identified</span>';

  // Prediction
  const pred = ai.prediction || {};
  const tgt  = pred.target;
  const tgtHtml = tgt
    ? `<div id="pred-tgt"><span class="tgt-tag" style="background:${bias==='bullish'?'rgba(63,185,80,.2)':'rgba(248,81,73,.2)'}; color:${bias==='bullish'?'#3fb950':'#f85149'}">Target: ${typeof tgt==='number'?tgt.toFixed(5):tgt}</span></div>`
    : '';
  document.getElementById('pred-tgt').outerHTML = tgtHtml || '<div id="pred-tgt"></div>';
  document.getElementById('pred-move').textContent = pred.next_move || '—';
  document.getElementById('pred-inv').textContent  = pred.invalidation ? '⚠ Invalidation: ' + pred.invalidation : '';

  // Education
  const edu = ai.education || {};
  document.getElementById('edu-concept').textContent = edu.concept || 'Price Action Concept';
  document.getElementById('edu-text').textContent    = edu.explanation || '';
  document.getElementById('edu-text').style.color    = 'var(--text)';

  // TF breakdown
  tfBreakdown = ai.timeframe_breakdown || {};
  renderTfContent(activeTfTab);

  // Detailed analysis
  document.getElementById('det-text').textContent = ai.detailed_analysis || '—';
  document.getElementById('det-text').style.color = 'var(--muted)';
}

function renderTfContent(tf) {
  activeTfTab = tf;
  document.querySelectorAll('.tf-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tf === tf);
  });
  const text = tfBreakdown[tf] || '—';
  document.getElementById('tf-content').textContent = text;
  document.getElementById('tf-content').classList.remove('loading');
}

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════
function getSym() {
  return document.getElementById('sym').value.trim().toUpperCase() || 'EUR_USD';
}

function setActiveTf(tf) {
  currentTf = tf;
  document.querySelectorAll('.tf-btn').forEach(b => b.classList.toggle('active', b.dataset.tf === tf));
  loadChart().then(() => {
    if (paData) { renderOverlays(paData, tf); renderZonesPanel(paData, tf); }
  });
}

// ════════════════════════════════════════════════════════════════
// Event listeners
// ════════════════════════════════════════════════════════════════
document.getElementById('btn-analyze').addEventListener('click', runAnalysis);
document.getElementById('btn-refresh').addEventListener('click', () => {
  loadChart();
  loadPA().then(() => { if (paData) renderOverlays(paData, currentTf); });
});

document.getElementById('sym').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAnalysis();
});

document.querySelectorAll('.tf-btn').forEach(btn => {
  btn.addEventListener('click', () => setActiveTf(btn.dataset.tf));
});

document.querySelectorAll('.tf-tab').forEach(tab => {
  tab.addEventListener('click', () => renderTfContent(tab.dataset.tf));
});

// ════════════════════════════════════════════════════════════════
// Auto-refresh (chart + PA every 60s)
// ════════════════════════════════════════════════════════════════
async function autoRefresh() {
  await loadChart();
  await loadPA();
  if (paData) renderOverlays(paData, currentTf);
}

// Init
initChart();
autoRefresh();
setInterval(autoRefresh, 60000);
</script>
</body>
</html>"""
