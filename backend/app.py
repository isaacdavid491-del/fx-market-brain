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
    body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    #top { padding:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    #chart { height: 70vh; }
    .pill { padding:6px 10px; border:1px solid #ddd; border-radius:999px; }
    button { padding:8px 12px; border-radius:10px; border:1px solid #ddd; background:#fff; }
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
    <div id="sig" class="pill">Signal: …</div>
  </div>

  <div id="chart"></div>

<script>
const el = document.getElementById('chart');
const chart = LightweightCharts.createChart(el, { width: el.clientWidth, height: el.clientHeight });
const series = chart.addCandlestickSeries();
const slLine = chart.addLineSeries();
const tpLine = chart.addLineSeries();

function toBars(data){
  const bars = [];
  for (let i=0;i<data.t.length;i++){
    bars.push({ time: data.t[i], open: data.o[i], high: data.h[i], low: data.l[i], close: data.c[i] });
  }
  return bars;
}

async function load(){
  const tf = document.getElementById('tf').value;
  const minutes = document.getElementById('mins').value;
  const h = await fetch(`/api/history?tf=${tf}&minutes=${minutes}`).then(r=>r.json());
  series.setData(toBars(h));

  const s = await fetch(`/api/signal`).then(r=>r.json());
  const sig = document.getElementById('sig');
  sig.innerHTML = `Signal: <b>${s.side}</b> | conf=${s.confidence} | entry=${s.entry}` + (s.sl?` | SL=${s.sl}`:'') + (s.tp?` | TP=${s.tp}`:'');
  const lastTime = h.t[h.t.length-1];
  if (s.sl){
    slLine.setData([{time: h.t[0], value: s.sl}, {time: lastTime, value: s.sl}]);
  } else {
    slLine.setData([]);
  }
  if (s.tp){
    tpLine.setData([{time: h.t[0], value: s.tp}, {time: lastTime, value: s.tp}]);
  } else {
    tpLine.setData([]);
  }
}

document.getElementById('refresh').onclick = load;
window.addEventListener('resize', () => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
load();
setInterval(load, 60000);
</script>
</body>
</html>
"""