import logging
import os
import json
import time
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
# DB and resampling
# =============================
# Storage and timeframe aggregation live in backend.store so the FX model and
# the NASDAQ ICT agent farm read exactly the same bars.

from backend.store import (  # noqa: E402
    TF_MAP,
    db,
    init_db,
    latest_ts,
    load_1m,
    resample_ohlcv,
    upsert_1m,
)

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

# Candle fetching and history seeding are shared with the agent farm.
from backend.data.feed import ingest_latest, seed_history  # noqa: E402
from backend.data.providers import get_provider  # noqa: E402


def ingest_once(symbol: str) -> Dict[str, Any]:
    out = ingest_latest(symbol, get_provider())
    return {"ok": True, "inserted": out["written"], "latest_ts": out["latest_ts"]}


def ensure_seed_history(symbol: str) -> Dict[str, Any]:
    out = seed_history(symbol, days=HISTORY_DAYS, provider=get_provider())
    return {"ok": True, "seeded": out["written"]}


scheduler = BackgroundScheduler(daemon=True)


def ingest_symbols() -> List[str]:
    """Every instrument the scheduler keeps fresh: the FX pair plus the
    NASDAQ instruments the agent farm trades."""
    from backend.service import get_service

    service = get_service()
    symbols = [DEFAULT_INSTRUMENT, *service.symbols()]
    return list(dict.fromkeys(s for s in symbols if s))


def scheduled_ingest():
    for symbol in ingest_symbols():
        try:
            ingest_once(symbol)
        except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the rest
            logging.getLogger("ingest").warning("ingest %s failed: %s", symbol, exc)

# =============================
# API
# =============================

@app.on_event("startup")
def _startup():
    init_db()
    for symbol in ingest_symbols():
        try:
            ensure_seed_history(symbol)
        except Exception as exc:  # noqa: BLE001 - start up even with a cold feed
            logging.getLogger("startup").warning("seeding %s failed: %s", symbol, exc)
    scheduler.add_job(scheduled_ingest, "interval", seconds=INGEST_EVERY_SECONDS, id="ingest")
    scheduler.start()

# The NASDAQ ICT agent farm lives under /api/ict with a dashboard at /ict.
from backend.api_ict import dashboard_router as ict_dashboard_router  # noqa: E402
from backend.api_ict import router as ict_router  # noqa: E402

app.include_router(ict_router)
app.include_router(ict_dashboard_router)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "instrument": DEFAULT_INSTRUMENT,
        "latest_ts": latest_ts(DEFAULT_INSTRUMENT),
        "ict_farm": "/ict",
    }

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
    <a class="pill" href="/ict" style="text-decoration:none;color:inherit">NASDAQ ICT Agent Farm &rarr;</a>
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