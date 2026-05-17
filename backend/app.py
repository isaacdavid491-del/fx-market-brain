import os
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

from backend.market_reader import format_all, current_price_summary
from backend import ai_brain

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

app = FastAPI(title="FX Market Brain")

# ═══════════════════════════════════════════════════════════════════
# Database
# ═══════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
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
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles_1m (
            symbol TEXT NOT NULL,
            t      INTEGER NOT NULL,
            o      REAL NOT NULL,
            h      REAL NOT NULL,
            l      REAL NOT NULL,
            c      REAL NOT NULL,
            v      REAL NOT NULL,
            PRIMARY KEY(symbol, t)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_c1m ON candles_1m(symbol, t)")
    conn.commit()
    conn.close()
    ai_brain.init_journal_table()

# ═══════════════════════════════════════════════════════════════════
# OANDA client
# ═══════════════════════════════════════════════════════════════════

def oanda_headers() -> Dict[str, str]:
    if not OANDA_TOKEN:
        raise RuntimeError("OANDA_TOKEN not set")
    return {"Authorization": f"Bearer {OANDA_TOKEN}"}

def oanda_get_candles(symbol: str, granularity: str, count: int = 500,
                      to_rfc3339: Optional[str] = None) -> List[Dict]:
    params = {"granularity": granularity, "price": "M", "count": str(count)}
    if to_rfc3339:
        params["to"] = to_rfc3339
    r = requests.get(f"{OANDA_API_BASE}/instruments/{symbol}/candles",
                     headers=oanda_headers(), params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"OANDA {r.status_code}: {r.text[:300]}")
    return r.json().get("candles", [])

def parse_oanda_candles(candles: List[Dict]) -> pd.DataFrame:
    rows = []
    for x in candles:
        if not x.get("complete"):
            continue
        dt  = datetime.fromisoformat(x["time"].replace("Z", "+00:00"))
        mid = x["mid"]
        rows.append({"t": int(dt.timestamp()),
                     "o": float(mid["o"]), "h": float(mid["h"]),
                     "l": float(mid["l"]), "c": float(mid["c"]),
                     "v": float(x.get("volume", 0))})
    if not rows:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])
    return pd.DataFrame(rows).drop_duplicates("t").sort_values("t")

def upsert_1m(symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    conn = _db()
    cur  = conn.cursor()
    n = 0
    for r in df.itertuples(index=False):
        try:
            cur.execute("INSERT OR REPLACE INTO candles_1m VALUES (?,?,?,?,?,?,?)",
                        (symbol, int(r.t), float(r.o), float(r.h),
                         float(r.l), float(r.c), float(r.v)))
            n += 1
        except Exception:
            continue
    conn.commit()
    conn.close()
    return n

def latest_ts(symbol: str) -> Optional[int]:
    conn = _db()
    row  = conn.execute("SELECT MAX(t) FROM candles_1m WHERE symbol=?", (symbol,)).fetchone()
    conn.close()
    return int(row[0]) if row and row[0] else None

def load_1m(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    conn = _db()
    df   = pd.read_sql_query(
        "SELECT t,o,h,l,c,v FROM candles_1m WHERE symbol=? AND t BETWEEN ? AND ? ORDER BY t",
        conn, params=(symbol, start_ts, end_ts))
    conn.close()
    return df

# ═══════════════════════════════════════════════════════════════════
# Multi-timeframe resampling
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
    end_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp())
    df_1m    = load_1m(symbol, start_ts, end_ts)
    df_map: Dict[str, pd.DataFrame] = {}
    for tf in TF_MAP:
        df_map[tf] = df_1m if tf == "1m" else resample_ohlcv(df_1m, tf)
    return df_map

# ═══════════════════════════════════════════════════════════════════
# Ingestion
# ═══════════════════════════════════════════════════════════════════

def ingest_once(symbol: str) -> Dict:
    df = parse_oanda_candles(oanda_get_candles(symbol, "M1", count=500))
    n  = upsert_1m(symbol, df)
    return {"ok": True, "inserted": n, "latest_ts": latest_ts(symbol)}

def ensure_seed_history(symbol: str) -> Dict:
    target = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    to     = datetime.now(timezone.utc)
    total  = 0
    for _ in range(200):
        candles = oanda_get_candles(symbol, "M1", count=500, to_rfc3339=to.isoformat())
        df = parse_oanda_candles(candles)
        if df.empty:
            break
        total += upsert_1m(symbol, df)
        oldest = int(df["t"].min())
        if datetime.fromtimestamp(oldest, tz=timezone.utc) <= target:
            break
        to = datetime.fromtimestamp(oldest - 60, tz=timezone.utc)
        time.sleep(0.2)
    return {"ok": True, "seeded": total}

scheduler = BackgroundScheduler(daemon=True)

def _scheduled_ingest():
    try:
        ingest_once(DEFAULT_INSTRUMENT)
    except Exception:
        pass

def _scheduled_verify():
    """Verify elapsed predictions every 5 minutes."""
    try:
        def get_price():
            df = load_1m(DEFAULT_INSTRUMENT,
                         int(datetime.now(timezone.utc).timestamp()) - 120,
                         int(datetime.now(timezone.utc).timestamp()))
            return float(df["c"].iloc[-1]) if not df.empty else None
        ai_brain.verify_pending_predictions(DEFAULT_INSTRUMENT, get_price)
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
    scheduler.add_job(_scheduled_ingest, "interval", seconds=INGEST_EVERY_SECONDS, id="ingest")
    scheduler.add_job(_scheduled_verify, "interval", seconds=300, id="verify")
    scheduler.start()

@app.get("/api/health")
def health():
    return {"ok": True, "instrument": DEFAULT_INSTRUMENT, "latest_ts": latest_ts(DEFAULT_INSTRUMENT)}

@app.get("/api/history")
def history(
    symbol:  str = Query(DEFAULT_INSTRUMENT),
    tf:      str = Query("15m"),
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

@app.get("/api/observe")
def observe(symbol: str = Query(DEFAULT_INSTRUMENT)):
    """
    Main AI endpoint. Feed raw price data across all timeframes to Claude.
    Claude observes the numbers, learns from past predictions, makes a new prediction.
    No pre-defined rules — the AI discovers everything itself.
    """
    df_map  = _build_df_map(symbol)
    if all(df.empty for df in df_map.values()):
        return {"error": "no_data", "summary": "No price data. Check OANDA_TOKEN."}

    raw_text    = format_all(df_map)
    prices      = current_price_summary(df_map)
    current_c   = prices.get("1m", prices.get("5m", prices.get("15m", {}))).get("c", 0.0)

    # Verify any elapsed predictions before generating a new one
    def get_price():
        return current_c
    ai_brain.verify_pending_predictions(symbol, get_price)

    result = ai_brain.observe_and_predict(symbol, raw_text, current_c)
    result["current_prices"] = prices
    result["stats"] = ai_brain.journal_stats(symbol)
    return result

@app.get("/api/journal")
def journal(symbol: str = Query(DEFAULT_INSTRUMENT), limit: int = Query(20, ge=1, le=100)):
    """Return the AI's observation journal — past predictions and outcomes."""
    entries = ai_brain.load_recent_journal(symbol, limit=limit)
    result  = []
    for e in entries:
        an   = e.get("analysis", {})
        pred = an.get("prediction", {})
        result.append({
            "id":            e["id"],
            "observed_at":   e["observed_at"],
            "price_at_obs":  e["price_at_obs"],
            "direction":     pred.get("direction"),
            "target":        pred.get("target_price"),
            "horizon_min":   pred.get("horizon_minutes"),
            "confidence":    pred.get("confidence"),
            "summary":       an.get("summary", ""),
            "observations":  an.get("observations", []),
            "reasoning":     pred.get("reasoning", ""),
            "verified":      bool(e["verified"]),
            "was_correct":   e.get("was_correct"),
            "actual_price":  e.get("actual_price"),
        })
    return {"entries": result, "stats": ai_brain.journal_stats(symbol)}

@app.get("/api/stats")
def stats(symbol: str = Query(DEFAULT_INSTRUMENT)):
    return ai_brain.journal_stats(symbol)

@app.post("/api/admin/ingest")
def admin_ingest(x_admin_key: Optional[str] = Header(default=None),
                 symbol: str = Query(DEFAULT_INSTRUMENT)):
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Unauthorized")
    return ingest_once(symbol)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(_INDEX_HTML)

# ═══════════════════════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════════════════════

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>FX Market Brain</title>
  <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:   #0d1117; --bg2: #161b22; --bg3: #21262d;
      --bd:   #30363d; --txt: #e6edf3; --dim: #8b949e;
      --grn:  #3fb950; --red: #f85149; --blu: #58a6ff;
      --ylw:  #d29922; --pur: #bc8cff;
    }
    html,body { height:100%; background:var(--bg); color:var(--txt);
      font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;
      font-size:13px; overflow:hidden; }

    #app { display:flex; flex-direction:column; height:100vh; }

    /* Header */
    #hdr {
      display:flex; align-items:center; gap:10px; padding:8px 14px;
      background:var(--bg2); border-bottom:1px solid var(--bd);
      flex-shrink:0; flex-wrap:wrap;
    }
    #logo { font-weight:900; font-size:14px; color:var(--blu); letter-spacing:-0.5px; }
    .sep  { width:1px; height:18px; background:var(--bd); }

    #sym {
      background:var(--bg3); border:1px solid var(--bd); color:var(--txt);
      padding:4px 9px; border-radius:6px; font-size:12px;
      width:90px; font-family:monospace; text-transform:uppercase;
    }
    #sym:focus { outline:none; border-color:var(--blu); }

    .tfbtns { display:flex; gap:3px; }
    .tfb {
      background:var(--bg3); border:1px solid var(--bd); color:var(--dim);
      padding:4px 9px; border-radius:5px; cursor:pointer;
      font-size:11px; font-weight:700; transition:all .1s;
    }
    .tfb:hover  { border-color:var(--blu); color:var(--blu); }
    .tfb.on     { background:var(--blu); border-color:var(--blu); color:#fff; }

    .btn {
      background:var(--bg3); border:1px solid var(--bd); color:var(--txt);
      padding:5px 13px; border-radius:6px; cursor:pointer;
      font-size:12px; font-weight:600; transition:all .1s; white-space:nowrap;
    }
    .btn:hover    { border-color:var(--blu); }
    .btn:disabled { opacity:.4; cursor:not-allowed; }
    .btn.primary  { background:var(--blu); border-color:var(--blu); color:#fff; }
    .btn.primary:hover { background:#79b8ff; }

    #live { display:flex; align-items:center; gap:5px; font-size:11px; color:var(--dim); }
    #dot  { width:7px; height:7px; border-radius:50%; background:var(--grn); animation:blink 2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

    #upd { margin-left:auto; color:var(--dim); font-size:11px; }

    /* Price strip */
    #pstrip {
      display:flex; align-items:center; gap:8px; padding:5px 14px;
      background:var(--bg2); border-bottom:1px solid var(--bd);
      flex-shrink:0; font-family:monospace; font-size:12px; overflow-x:auto;
    }
    .ptf { display:flex; gap:4px; align-items:center; }
    .ptflbl { color:var(--dim); font-size:10px; font-weight:700; }
    .ptfc   { color:var(--txt); }
    .pu  { color:var(--grn); }
    .pd  { color:var(--red); }

    /* Main */
    #main { display:flex; flex:1; overflow:hidden; min-height:0; }

    /* Chart panel */
    #cpanel { flex:1; display:flex; flex-direction:column; min-width:0; }
    #chart  { flex:1; min-height:0; }

    /* Right panel */
    #rpanel {
      width:320px; min-width:270px;
      border-left:1px solid var(--bd);
      overflow-y:auto; display:flex; flex-direction:column;
    }
    .sec { border-bottom:1px solid var(--bd); padding:12px; }
    .stitle { font-size:10px; font-weight:700; letter-spacing:1px;
              text-transform:uppercase; color:var(--dim); margin-bottom:8px; }

    /* Brain status */
    #bstatus {
      padding:14px 12px; border-bottom:1px solid var(--bd);
      text-align:center;
    }
    #bias-lbl { font-size:20px; font-weight:900; letter-spacing:3px; margin-bottom:6px; }
    .gbull { color:var(--grn); } .gbear { color:var(--red); } .gneut { color:var(--dim); }

    #cbar-wrap { background:var(--bg3); border-radius:3px; height:4px; margin:6px 0; overflow:hidden; }
    #cbar      { height:100%; border-radius:3px; transition:width .4s,background .4s; width:0; }
    #cnum      { color:var(--dim); font-size:11px; }

    /* Observations */
    .obs-item {
      padding:5px 0; border-bottom:1px solid rgba(48,54,61,.4);
      font-size:12px; line-height:1.55; color:var(--txt);
    }
    .obs-item::before { content:"→ "; color:var(--blu); font-weight:700; }

    /* Prediction card */
    #pred-card { background:var(--bg3); border-radius:7px; padding:11px; }
    .pred-dir  { font-size:15px; font-weight:900; letter-spacing:2px; margin-bottom:6px; }
    .pred-tgt  { font-family:monospace; font-size:12px; color:var(--dim); margin-bottom:6px; }
    .pred-rsn  { font-size:12px; line-height:1.6; color:var(--txt); margin-bottom:6px; }
    .pred-wtch { font-size:11px; color:var(--pur); line-height:1.5; }

    /* Hypothesis */
    #hyp-box {
      background:rgba(88,166,255,.06); border:1px solid rgba(88,166,255,.18);
      border-radius:6px; padding:10px;
      font-size:12px; line-height:1.65; color:var(--txt);
    }

    /* Journal */
    .jentry {
      padding:8px 0; border-bottom:1px solid rgba(48,54,61,.4); font-size:11px;
    }
    .jtime   { color:var(--dim); font-size:10px; margin-bottom:3px; }
    .jpred   { font-weight:700; margin-bottom:3px; }
    .jup     { color:var(--grn); } .jdn { color:var(--red); } .jsw { color:var(--dim); }
    .jrsn    { color:var(--dim); line-height:1.45; margin-bottom:3px; }
    .jresult { font-size:10px; font-weight:700; }
    .jok     { color:var(--grn); } .jno { color:var(--red); } .jpnd { color:var(--ylw); }

    /* Stats bar */
    #stats-bar {
      padding:8px 12px; background:var(--bg2); border-bottom:1px solid var(--bd);
      display:flex; gap:16px; font-size:11px; align-items:center;
      flex-shrink:0;
    }
    .stat-item { display:flex; gap:4px; align-items:center; }
    .stat-lbl  { color:var(--dim); }
    .stat-val  { font-weight:700; font-family:monospace; }

    /* Scrollbar */
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:var(--bg2); }
    ::-webkit-scrollbar-thumb { background:var(--bd); border-radius:3px; }

    .dim { color:var(--dim); }
    .loading { color:var(--dim); font-style:italic; font-size:12px; }
  </style>
</head>
<body>
<div id="app">

  <!-- Header -->
  <div id="hdr">
    <span id="logo">▪ FX BRAIN</span>
    <div class="sep"></div>
    <input id="sym" type="text" value="EUR_USD" placeholder="EUR_USD"/>
    <div class="sep"></div>
    <div class="tfbtns">
      <button class="tfb" data-tf="1m">1M</button>
      <button class="tfb" data-tf="5m">5M</button>
      <button class="tfb on" data-tf="15m">15M</button>
      <button class="tfb" data-tf="1h">1H</button>
      <button class="tfb" data-tf="4h">4H</button>
      <button class="tfb" data-tf="1d">1D</button>
    </div>
    <div class="sep"></div>
    <button class="btn primary" id="btn-observe">👁 Watch &amp; Predict</button>
    <button class="btn" id="btn-refresh">↺</button>
    <div id="live"><div id="dot"></div><span>LIVE</span></div>
    <span id="upd">—</span>
  </div>

  <!-- Price strip -->
  <div id="pstrip">
    <span class="dim" style="font-size:10px;font-weight:700;">PRICE</span>
    <!-- populated by JS -->
  </div>

  <!-- Stats bar -->
  <div id="stats-bar">
    <div class="stat-item">
      <span class="stat-lbl">Observations:</span>
      <span class="stat-val" id="st-total">—</span>
    </div>
    <div class="stat-item">
      <span class="stat-lbl">Verified:</span>
      <span class="stat-val" id="st-ver">—</span>
    </div>
    <div class="stat-item">
      <span class="stat-lbl">Accuracy:</span>
      <span class="stat-val" id="st-acc">—</span>
    </div>
    <span class="dim" style="margin-left:auto;font-size:11px;" id="st-note">
      The AI learns from whether its predictions were correct
    </span>
  </div>

  <!-- Main -->
  <div id="main">

    <!-- Chart -->
    <div id="cpanel">
      <div id="chart"></div>
    </div>

    <!-- Analysis panel -->
    <div id="rpanel">

      <!-- Bias -->
      <div id="bstatus">
        <div id="bias-lbl" class="gneut">—</div>
        <div id="cbar-wrap"><div id="cbar"></div></div>
        <div id="cnum">Confidence: —</div>
        <div id="bias-sum" class="loading" style="margin-top:8px;text-align:left;">
          Click "Watch &amp; Predict" — the AI reads raw price data across all timeframes and builds its own understanding of how this market moves.
        </div>
      </div>

      <!-- Observations -->
      <div class="sec">
        <div class="stitle">What the AI sees right now</div>
        <div id="obs-list"><span class="loading">—</span></div>
      </div>

      <!-- Prediction -->
      <div class="sec">
        <div class="stitle">Prediction</div>
        <div id="pred-card">
          <div id="pred-dir" class="pred-dir gneut">—</div>
          <div id="pred-tgt" class="pred-tgt"></div>
          <div id="pred-rsn" class="pred-rsn loading">—</div>
          <div id="pred-wtch" class="pred-wtch"></div>
        </div>
      </div>

      <!-- Hypothesis (the AI's current model) -->
      <div class="sec">
        <div class="stitle">AI's current model of this market</div>
        <div id="hyp-box" class="loading">—</div>
      </div>

      <!-- Journal -->
      <div class="sec">
        <div class="stitle">Learning journal</div>
        <div id="journal-list"><span class="loading">—</span></div>
      </div>

    </div>
  </div>
</div>

<script>
// ── Chart ────────────────────────────────────────────────────────────────────
const chartEl = document.getElementById('chart');
let chart, candles;
let plines = [];

function initChart() {
  chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout:  { background:{color:'#0d1117'}, textColor:'#e6edf3' },
    grid:    { vertLines:{color:'#1c2128'}, horzLines:{color:'#1c2128'} },
    crosshair: { mode:1 },
    rightPriceScale: { borderColor:'#30363d' },
    timeScale: { borderColor:'#30363d', timeVisible:true, secondsVisible:false },
  });
  candles = chart.addCandlestickSeries({
    upColor:'#3fb950',   downColor:'#f85149',
    borderUpColor:'#3fb950', borderDownColor:'#f85149',
    wickUpColor:'#3fb950',   wickDownColor:'#f85149',
  });
}

function clearLines() {
  plines.forEach(pl => { try { candles.removePriceLine(pl); } catch(e){} });
  plines = [];
}

function addLine(price, color, title, style=2) {
  if (!price || isNaN(price)) return;
  try {
    plines.push(candles.createPriceLine({price,color,lineStyle:style,lineWidth:1,axisLabelVisible:true,title}));
  } catch(e){}
}

// ── State ─────────────────────────────────────────────────────────────────────
let currentTf  = '15m';
let lastResult = null;

function getSym() { return document.getElementById('sym').value.trim().toUpperCase() || 'EUR_USD'; }

// ── Chart loading ─────────────────────────────────────────────────────────────
const minsMap = { '1m':240,'5m':1440,'15m':4320,'1h':10080,'4h':43200,'1d':43200 };

async function loadChart() {
  const sym  = getSym();
  const mins = minsMap[currentTf] || 4320;
  try {
    const d = await fetch(`/api/history?symbol=${sym}&tf=${currentTf}&minutes=${mins}`).then(r=>r.json());
    if (!d.t || !d.t.length) return;
    candles.setData(d.t.map((t,i) => ({time:t,open:d.o[i],high:d.h[i],low:d.l[i],close:d.c[i]})));
    chart.timeScale().fitContent();
    document.getElementById('upd').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

// ── Price strip ───────────────────────────────────────────────────────────────
function renderPriceStrip(prices) {
  const strip = document.getElementById('pstrip');
  const tfs   = ['1m','5m','15m','1h','4h','1d'];
  let html = '<span class="dim" style="font-size:10px;font-weight:700;">PRICE</span>';
  for (const tf of tfs) {
    const p = prices[tf];
    if (!p) continue;
    const chg = p.c - p.o;
    const cls = chg > 0 ? 'pu' : chg < 0 ? 'pd' : '';
    html += `<div class="ptf">
      <span class="ptflbl">${tf.toUpperCase()}</span>
      <span class="ptfc ${cls}">${p.c.toFixed(5)}</span>
    </div>`;
  }
  strip.innerHTML = html;
}

// ── Stats bar ─────────────────────────────────────────────────────────────────
function renderStats(s) {
  if (!s) return;
  document.getElementById('st-total').textContent = s.total_observations ?? '—';
  document.getElementById('st-ver').textContent   = s.verified ?? '—';
  const acc = s.accuracy_pct;
  const el  = document.getElementById('st-acc');
  el.textContent = acc != null ? acc + '%' : '—';
  el.style.color = acc == null ? '' : acc >= 60 ? '#3fb950' : acc >= 45 ? '#d29922' : '#f85149';
}

// ── Main observe ──────────────────────────────────────────────────────────────
async function observe() {
  const sym = getSym();
  const btn = document.getElementById('btn-observe');
  btn.disabled = true; btn.textContent = '⏳ Reading charts…';
  document.getElementById('bias-sum').textContent = 'Reading raw price data across all timeframes…';

  try {
    const result = await fetch(`/api/observe?symbol=${sym}`).then(r=>r.json());
    lastResult = result;

    if (result.current_prices) renderPriceStrip(result.current_prices);
    if (result.stats)          renderStats(result.stats);
    renderBrain(result);
    drawPredLine(result);
    await loadJournal(sym);
  } catch(e) {
    document.getElementById('bias-sum').textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = '👁 Watch & Predict';
  }
}

// ── Render brain panel ────────────────────────────────────────────────────────
function renderBrain(r) {
  if (!r) return;
  const pred = r.prediction || {};
  const dir  = (pred.direction || 'sideways').toLowerCase();
  const conf = pred.confidence || 0;

  // Bias
  const lbl = document.getElementById('bias-lbl');
  lbl.textContent = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';
  lbl.className   = dir === 'up' ? 'gbull' : dir === 'down' ? 'gbear' : 'gneut';

  // Confidence bar
  const bar = document.getElementById('cbar');
  bar.style.width      = conf + '%';
  bar.style.background = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#8b949e';
  document.getElementById('cnum').textContent = 'Confidence: ' + conf + '%';
  document.getElementById('bias-sum').textContent = r.summary || '';

  // Observations
  const obs = r.observations || [];
  document.getElementById('obs-list').innerHTML = obs.length
    ? obs.map(o => `<div class="obs-item">${escHtml(o)}</div>`).join('')
    : '<span class="loading">No observations</span>';

  // Prediction card
  const pdEl = document.getElementById('pred-dir');
  pdEl.textContent = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';
  pdEl.className   = 'pred-dir ' + (dir==='up'?'gbull':dir==='down'?'gbear':'gneut');

  const tgt = pred.target_price;
  document.getElementById('pred-tgt').textContent =
    (tgt ? `Target: ${Number(tgt).toFixed(5)}  ` : '') +
    (pred.horizon_minutes ? `Horizon: ${pred.horizon_minutes} min` : '');

  document.getElementById('pred-rsn').textContent  = pred.reasoning  || '—';
  document.getElementById('pred-rsn').classList.remove('loading');
  document.getElementById('pred-wtch').textContent = r.what_to_watch ? '👁 ' + r.what_to_watch : '';

  // Hypothesis
  const hyp = document.getElementById('hyp-box');
  hyp.textContent = r.pattern_hypothesis || '—';
  hyp.classList.remove('loading');
}

// ── Draw prediction line on chart ─────────────────────────────────────────────
function drawPredLine(r) {
  clearLines();
  if (!r) return;
  const pred = r.prediction || {};
  const tgt  = pred.target_price;
  if (!tgt) return;
  const dir = (pred.direction || '').toLowerCase();
  const col = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#8b949e';
  addLine(Number(tgt), col, `AI TARGET`, 2);
}

// ── Journal ───────────────────────────────────────────────────────────────────
async function loadJournal(sym) {
  try {
    const data = await fetch(`/api/journal?symbol=${sym}&limit=15`).then(r=>r.json());
    renderStats(data.stats);
    renderJournal(data.entries || []);
  } catch(e) {}
}

function renderJournal(entries) {
  const el = document.getElementById('journal-list');
  if (!entries.length) { el.innerHTML = '<span class="loading">No observations yet</span>'; return; }

  el.innerHTML = entries.map(e => {
    const dt  = new Date(e.observed_at * 1000).toLocaleString();
    const dir = (e.direction || 'sideways').toLowerCase();
    const dircls = dir === 'up' ? 'jup' : dir === 'dn' ? 'jdn' : dir === 'down' ? 'jdn' : 'jsw';
    const dirlbl = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';

    let resultHtml = '';
    if (e.verified) {
      const ok = e.was_correct;
      resultHtml = `<div class="jresult ${ok?'jok':'jno'}">${ok ? '✓ CORRECT' : '✗ WRONG'} — actual: ${e.actual_price ? Number(e.actual_price).toFixed(5) : '?'}</div>`;
    } else {
      resultHtml = `<div class="jresult jpnd">⏳ Pending verification</div>`;
    }

    const rsn = e.reasoning ? escHtml(e.reasoning.slice(0, 150)) + (e.reasoning.length > 150 ? '…' : '') : '';

    return `<div class="jentry">
      <div class="jtime">${dt} — price ${Number(e.price_at_obs).toFixed(5)}</div>
      <div class="jpred"><span class="${dircls}">${dirlbl}</span> → ${e.target ? Number(e.target).toFixed(5) : '?'} in ${e.horizon_min||'?'}min (${e.confidence||'?'}%)</div>
      ${rsn ? `<div class="jrsn">${rsn}</div>` : ''}
      ${resultHtml}
    </div>`;
  }).join('');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setTf(tf) {
  currentTf = tf;
  document.querySelectorAll('.tfb').forEach(b => b.classList.toggle('on', b.dataset.tf === tf));
  loadChart();
}

// ── Events ────────────────────────────────────────────────────────────────────
document.getElementById('btn-observe').addEventListener('click', observe);
document.getElementById('btn-refresh').addEventListener('click', () => loadChart());
document.getElementById('sym').addEventListener('keydown', e => { if (e.key==='Enter') observe(); });
document.querySelectorAll('.tfb').forEach(b => b.addEventListener('click', () => setTf(b.dataset.tf)));

// ── Auto-refresh ──────────────────────────────────────────────────────────────
async function tick() {
  await loadChart();
  const sym = getSym();
  const j   = await fetch(`/api/journal?symbol=${sym}&limit=1`).then(r=>r.json()).catch(()=>({}));
  if (j.stats) renderStats(j.stats);
}

initChart();
loadChart();
loadJournal(getSym());
setInterval(tick, 60000);
</script>
</body>
</html>"""
