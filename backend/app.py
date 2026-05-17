import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler

from backend.data_sources import fetch_all, latest_price
from backend.market_reader import format_all, current_price_summary
from backend import ai_brain

# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

DEFAULT_INSTRUMENT = os.getenv("DEFAULT_INSTRUMENT", "NAS100")
ADMIN_KEY          = os.getenv("ADMIN_KEY", "")

app       = FastAPI(title="FX Market Brain")
scheduler = BackgroundScheduler(daemon=True)

# ═══════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
def _startup():
    ai_brain.init_journal_table()

    def _verify_job():
        try:
            p = latest_price(DEFAULT_INSTRUMENT)
            if p:
                ai_brain.verify_pending_predictions(DEFAULT_INSTRUMENT, lambda: p)
        except Exception:
            pass

    scheduler.add_job(_verify_job, "interval", seconds=300, id="verify")
    scheduler.start()

# ═══════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    p = latest_price(DEFAULT_INSTRUMENT)
    return {"ok": True, "instrument": DEFAULT_INSTRUMENT, "last_price": p}


@app.get("/api/history")
def history(
    symbol: str = Query(DEFAULT_INSTRUMENT),
    tf:     str = Query("15m"),
):
    """Return OHLCV candles for the chart. Data from Yahoo Finance, no key needed."""
    data = fetch_all(symbol)
    df   = data.get(tf)
    if df is None or df.empty:
        return {"t": [], "o": [], "h": [], "l": [], "c": [], "v": []}
    return {
        "t": df["t"].tolist(),
        "o": df["o"].tolist(),
        "h": df["h"].tolist(),
        "l": df["l"].tolist(),
        "c": df["c"].tolist(),
        "v": df["v"].tolist(),
    }


@app.get("/api/observe")
def observe(symbol: str = Query(DEFAULT_INSTRUMENT)):
    """
    Feed raw multi-TF price data to the AI brain.
    Claude reads the numbers directly, finds its own patterns, makes a prediction.
    Result is saved to the learning journal automatically.
    """
    df_map = fetch_all(symbol)
    if not df_map:
        return {
            "error": "no_data",
            "summary": f"Could not fetch market data for {symbol}. Check your internet connection.",
        }

    raw_text  = format_all(df_map)
    prices    = current_price_summary(df_map)
    current_c = (
        prices.get("1m") or prices.get("5m") or prices.get("15m") or {}
    ).get("c", 0.0)

    # Verify any predictions whose horizon has elapsed before making a new one
    ai_brain.verify_pending_predictions(symbol, lambda: current_c)

    result = ai_brain.observe_and_predict(symbol, raw_text, float(current_c))
    result["current_prices"] = prices
    result["stats"]          = ai_brain.journal_stats(symbol)
    return result


@app.get("/api/journal")
def journal(
    symbol: str = Query(DEFAULT_INSTRUMENT),
    limit:  int = Query(20, ge=1, le=100),
):
    """The AI's learning journal — every observation, prediction, and outcome."""
    entries = ai_brain.load_recent_journal(symbol, limit=limit)
    result  = []
    for e in entries:
        an   = e.get("analysis", {})
        pred = an.get("prediction", {})
        result.append({
            "id":           e["id"],
            "observed_at":  e["observed_at"],
            "price_at_obs": e["price_at_obs"],
            "direction":    pred.get("direction"),
            "target":       pred.get("target_price"),
            "horizon_min":  pred.get("horizon_minutes"),
            "confidence":   pred.get("confidence"),
            "summary":      an.get("summary", ""),
            "observations": an.get("observations", []),
            "reasoning":    pred.get("reasoning", ""),
            "verified":     bool(e["verified"]),
            "was_correct":  e.get("was_correct"),
            "actual_price": e.get("actual_price"),
        })
    return {"entries": result, "stats": ai_brain.journal_stats(symbol)}


@app.get("/api/stats")
def stats(symbol: str = Query(DEFAULT_INSTRUMENT)):
    return ai_brain.journal_stats(symbol)


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
      --bg:  #0d1117; --bg2: #161b22; --bg3: #21262d;
      --bd:  #30363d; --txt: #e6edf3; --dim: #8b949e;
      --grn: #3fb950; --red: #f85149; --blu: #58a6ff;
      --ylw: #d29922; --pur: #bc8cff;
    }
    html, body {
      height: 100%; background: var(--bg); color: var(--txt);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 13px; overflow: hidden;
    }
    #app { display: flex; flex-direction: column; height: 100vh; }

    /* Header */
    #hdr {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 14px; background: var(--bg2);
      border-bottom: 1px solid var(--bd); flex-shrink: 0; flex-wrap: wrap;
    }
    #logo { font-weight: 900; font-size: 14px; color: var(--blu); letter-spacing: -.4px; }
    .sep  { width: 1px; height: 18px; background: var(--bd); flex-shrink: 0; }

    #sym {
      background: var(--bg3); border: 1px solid var(--bd); color: var(--txt);
      padding: 4px 9px; border-radius: 6px; font-size: 12px;
      width: 100px; font-family: monospace; text-transform: uppercase;
    }
    #sym:focus { outline: none; border-color: var(--blu); }

    .tfbtns { display: flex; gap: 3px; }
    .tfb {
      background: var(--bg3); border: 1px solid var(--bd); color: var(--dim);
      padding: 4px 9px; border-radius: 5px; cursor: pointer;
      font-size: 11px; font-weight: 700; transition: all .1s;
    }
    .tfb:hover { border-color: var(--blu); color: var(--blu); }
    .tfb.on   { background: var(--blu); border-color: var(--blu); color: #fff; }

    .btn {
      background: var(--bg3); border: 1px solid var(--bd); color: var(--txt);
      padding: 5px 13px; border-radius: 6px; cursor: pointer;
      font-size: 12px; font-weight: 600; transition: all .1s; white-space: nowrap;
    }
    .btn:hover   { border-color: var(--blu); }
    .btn:disabled { opacity: .4; cursor: not-allowed; }
    .btn.pri { background: var(--blu); border-color: var(--blu); color: #fff; }
    .btn.pri:hover { background: #79b8ff; border-color: #79b8ff; }

    #live { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--dim); }
    #dot  { width: 7px; height: 7px; border-radius: 50%; background: var(--grn);
            animation: blink 2s infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
    #upd { margin-left: auto; color: var(--dim); font-size: 11px; }

    /* Price strip */
    #pstrip {
      display: flex; align-items: center; gap: 12px;
      padding: 5px 14px; background: var(--bg2);
      border-bottom: 1px solid var(--bd); flex-shrink: 0;
      overflow-x: auto; font-family: monospace; font-size: 12px;
    }
    .ptf  { display: flex; gap: 5px; align-items: center; }
    .ptfl { color: var(--dim); font-size: 10px; font-weight: 700; }
    .pu   { color: var(--grn); } .pd { color: var(--red); }

    /* Stats */
    #sbar {
      display: flex; align-items: center; gap: 16px;
      padding: 5px 14px; background: var(--bg2);
      border-bottom: 1px solid var(--bd); flex-shrink: 0; font-size: 11px;
    }
    .si  { display: flex; gap: 4px; align-items: center; }
    .sl  { color: var(--dim); }
    .sv  { font-weight: 700; font-family: monospace; }
    #snt { margin-left: auto; color: var(--dim); font-size: 11px; }

    /* Main */
    #main { display: flex; flex: 1; overflow: hidden; min-height: 0; }
    #cpanel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
    #chart  { flex: 1; min-height: 0; }

    /* Right panel */
    #rpanel {
      width: 320px; min-width: 270px;
      border-left: 1px solid var(--bd);
      overflow-y: auto; display: flex; flex-direction: column;
    }
    .sec    { border-bottom: 1px solid var(--bd); padding: 11px 12px; }
    .stitle { font-size: 10px; font-weight: 700; letter-spacing: 1px;
              text-transform: uppercase; color: var(--dim); margin-bottom: 8px; }

    /* Bias */
    #bcard { padding: 14px 12px; border-bottom: 1px solid var(--bd); text-align: center; }
    #blbl  { font-size: 20px; font-weight: 900; letter-spacing: 3px; margin-bottom: 6px; }
    .gb { color: var(--grn); } .gr { color: var(--red); } .gn { color: var(--dim); }
    #cbw { background: var(--bg3); border-radius: 3px; height: 4px; margin: 6px 0; overflow: hidden; }
    #cb  { height: 100%; border-radius: 3px; transition: width .4s, background .4s; width: 0; }
    #cnum { color: var(--dim); font-size: 11px; }
    #bsum { text-align: left; margin-top: 8px; color: var(--dim); font-size: 12px; line-height: 1.55; }

    /* Observations */
    .oi { padding: 5px 0; border-bottom: 1px solid rgba(48,54,61,.4);
          font-size: 12px; line-height: 1.55; }
    .oi::before { content: "→ "; color: var(--blu); font-weight: 700; }

    /* Prediction card */
    #pcard { background: var(--bg3); border-radius: 7px; padding: 11px; }
    .pdir  { font-size: 15px; font-weight: 900; letter-spacing: 2px; margin-bottom: 6px; }
    .ptgt  { font-family: monospace; font-size: 12px; color: var(--dim); margin-bottom: 6px; }
    .prsn  { font-size: 12px; line-height: 1.6; margin-bottom: 6px; }
    .pwtch { font-size: 11px; color: var(--pur); line-height: 1.5; }

    /* Hypothesis */
    #hbox {
      background: rgba(88,166,255,.06); border: 1px solid rgba(88,166,255,.18);
      border-radius: 6px; padding: 10px;
      font-size: 12px; line-height: 1.65;
    }

    /* Journal entries */
    .je  { padding: 8px 0; border-bottom: 1px solid rgba(48,54,61,.4); font-size: 11px; }
    .jt  { color: var(--dim); font-size: 10px; margin-bottom: 3px; }
    .jp  { font-weight: 700; margin-bottom: 3px; }
    .jup { color: var(--grn); } .jdn { color: var(--red); } .jsw { color: var(--dim); }
    .jr  { color: var(--dim); line-height: 1.45; margin-bottom: 3px; }
    .jres { font-size: 10px; font-weight: 700; }
    .jok { color: var(--grn); } .jno { color: var(--red); } .jpnd { color: var(--ylw); }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--bd); border-radius: 3px; }
    .dim { color: var(--dim); }
    .lod { color: var(--dim); font-style: italic; font-size: 12px; }

    @media (max-width: 700px) {
      #rpanel { width: 100%; border-left: none; border-top: 1px solid var(--bd); }
      #main { flex-direction: column; }
    }
  </style>
</head>
<body>
<div id="app">

  <div id="hdr">
    <span id="logo">▪ FX BRAIN</span>
    <div class="sep"></div>
    <input id="sym" type="text" value="NAS100" placeholder="NAS100" title="Try: NAS100, EURUSD, GOLD, BTC, SP500..."/>
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
    <button class="btn pri" id="btn-obs">👁 Watch &amp; Predict</button>
    <button class="btn" id="btn-ref">↺</button>
    <div id="live"><div id="dot"></div><span>LIVE</span></div>
    <span id="upd">—</span>
  </div>

  <div id="pstrip">
    <span class="dim" style="font-size:10px;font-weight:700;">PRICE</span>
  </div>

  <div id="sbar">
    <div class="si"><span class="sl">Observations:</span><span class="sv" id="st-tot">—</span></div>
    <div class="si"><span class="sl">Verified:</span><span class="sv" id="st-ver">—</span></div>
    <div class="si"><span class="sl">Accuracy:</span><span class="sv" id="st-acc">—</span></div>
    <span id="snt">AI learns from whether its predictions were correct</span>
  </div>

  <div id="main">
    <div id="cpanel"><div id="chart"></div></div>

    <div id="rpanel">

      <div id="bcard">
        <div id="blbl" class="gn">—</div>
        <div id="cbw"><div id="cb"></div></div>
        <div id="cnum">Confidence: —</div>
        <div id="bsum" class="lod">
          Click "Watch &amp; Predict" to start.<br/>
          The AI reads raw NAS100 price data (no indicators) and discovers patterns on its own.
          Every prediction is logged and verified — the AI learns from being right or wrong.
        </div>
      </div>

      <div class="sec">
        <div class="stitle">What the AI notices right now</div>
        <div id="obs-list"><span class="lod">—</span></div>
      </div>

      <div class="sec">
        <div class="stitle">Prediction</div>
        <div id="pcard">
          <div id="pdir" class="pdir gn">—</div>
          <div id="ptgt" class="ptgt"></div>
          <div id="prsn" class="prsn lod">—</div>
          <div id="pwtch" class="pwtch"></div>
        </div>
      </div>

      <div class="sec">
        <div class="stitle">AI's current model of this market</div>
        <div id="hbox" class="lod">—</div>
      </div>

      <div class="sec">
        <div class="stitle">Learning journal</div>
        <div id="jlist"><span class="lod">No observations yet. Click Watch &amp; Predict.</span></div>
      </div>

    </div>
  </div>
</div>

<script>
// ── Chart ─────────────────────────────────────────────────────────────────────
const chartEl = document.getElementById('chart');
let chart, cser;
let plines = [];

function initChart() {
  chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout:  { background: { color: '#0d1117' }, textColor: '#e6edf3' },
    grid:    { vertLines: { color: '#1c2128' }, horzLines: { color: '#1c2128' } },
    crosshair: { mode: 1 },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
  });
  cser = chart.addCandlestickSeries({
    upColor: '#3fb950', downColor: '#f85149',
    borderUpColor: '#3fb950', borderDownColor: '#f85149',
    wickUpColor: '#3fb950', wickDownColor: '#f85149',
  });
}

function clearLines() {
  plines.forEach(pl => { try { cser.removePriceLine(pl); } catch(e) {} });
  plines = [];
}

function addLine(price, color, title, style = 2) {
  if (!price || isNaN(price)) return;
  try {
    plines.push(cser.createPriceLine({ price, color, lineStyle: style, lineWidth: 1,
                                       axisLabelVisible: true, title }));
  } catch(e) {}
}

// ── State ─────────────────────────────────────────────────────────────────────
let tf = '15m';
function getSym() { return document.getElementById('sym').value.trim().toUpperCase() || 'NAS100'; }

// ── Price format (adaptive decimals) ─────────────────────────────────────────
function fmt(n) {
  if (n == null) return '—';
  n = Number(n);
  if (n > 1000)  return n.toFixed(2);
  if (n > 10)    return n.toFixed(3);
  if (n > 1)     return n.toFixed(5);
  return n.toFixed(6);
}

// ── Chart loading ─────────────────────────────────────────────────────────────
async function loadChart() {
  const sym = getSym();
  try {
    const d = await fetch(`/api/history?symbol=${sym}&tf=${tf}`).then(r => r.json());
    if (!d.t || !d.t.length) return;
    cser.setData(d.t.map((t,i) => ({ time:t, open:d.o[i], high:d.h[i], low:d.l[i], close:d.c[i] })));
    chart.timeScale().fitContent();
    document.getElementById('upd').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

// ── Price strip ───────────────────────────────────────────────────────────────
function renderStrip(prices) {
  const strip = document.getElementById('pstrip');
  const tfs   = ['1m','5m','15m','1h','4h','1d'];
  let html = '<span class="dim" style="font-size:10px;font-weight:700">PRICE</span>';
  for (const t of tfs) {
    const p = prices[t];
    if (!p) continue;
    const chg = p.c - p.o;
    const cls = chg > 0 ? 'pu' : chg < 0 ? 'pd' : '';
    html += `<div class="ptf"><span class="ptfl">${t.toUpperCase()}</span><span class="${cls}">${fmt(p.c)}</span></div>`;
  }
  strip.innerHTML = html;
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function renderStats(s) {
  if (!s) return;
  document.getElementById('st-tot').textContent = s.total_observations ?? '—';
  document.getElementById('st-ver').textContent = s.verified ?? '—';
  const acc = s.accuracy_pct;
  const el  = document.getElementById('st-acc');
  el.textContent = acc != null ? acc + '%' : '—';
  el.style.color = acc == null ? '' : acc >= 60 ? '#3fb950' : acc >= 45 ? '#d29922' : '#f85149';
}

// ── Observe ───────────────────────────────────────────────────────────────────
async function observe() {
  const sym = getSym();
  const btn = document.getElementById('btn-obs');
  btn.disabled = true; btn.textContent = '⏳ Reading…';
  document.getElementById('bsum').textContent = 'Fetching ' + sym + ' data across all timeframes…';

  try {
    const res = await fetch(`/api/observe?symbol=${sym}`).then(r => r.json());
    if (res.current_prices) renderStrip(res.current_prices);
    if (res.stats)          renderStats(res.stats);
    renderBrain(res);
    await loadJournal(sym);
    await loadChart();
  } catch(e) {
    document.getElementById('bsum').textContent = 'Error: ' + e.message;
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

  const lbl = document.getElementById('blbl');
  lbl.textContent = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';
  lbl.className   = dir === 'up' ? 'gb' : dir === 'down' ? 'gr' : 'gn';

  const bar = document.getElementById('cb');
  bar.style.width      = conf + '%';
  bar.style.background = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#8b949e';
  document.getElementById('cnum').textContent = 'Confidence: ' + conf + '%';
  document.getElementById('bsum').textContent = r.summary || '';

  // Observations
  const obs = r.observations || [];
  document.getElementById('obs-list').innerHTML = obs.length
    ? obs.map(o => `<div class="oi">${esc(o)}</div>`).join('')
    : '<span class="lod">No observations</span>';

  // Prediction
  const pd = document.getElementById('pdir');
  pd.textContent = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';
  pd.className   = 'pdir ' + (dir==='up'?'gb':dir==='down'?'gr':'gn');

  const tgt = pred.target_price;
  document.getElementById('ptgt').textContent =
    (tgt ? 'Target: ' + fmt(tgt) + '   ' : '') +
    (pred.horizon_minutes ? 'Horizon: ' + pred.horizon_minutes + ' min' : '');

  const rsnEl = document.getElementById('prsn');
  rsnEl.textContent = pred.reasoning || '—';
  rsnEl.classList.remove('lod');

  document.getElementById('pwtch').textContent = r.what_to_watch ? '👁 ' + r.what_to_watch : '';

  // Hypothesis
  const h = document.getElementById('hbox');
  h.textContent = r.pattern_hypothesis || '—';
  h.classList.remove('lod');

  // Draw target on chart
  clearLines();
  if (tgt) {
    const col = dir === 'up' ? '#3fb950' : dir === 'down' ? '#f85149' : '#8b949e';
    addLine(Number(tgt), col, 'AI TARGET', 2);
  }
}

// ── Journal ───────────────────────────────────────────────────────────────────
async function loadJournal(sym) {
  try {
    const data = await fetch(`/api/journal?symbol=${sym}&limit=20`).then(r => r.json());
    if (data.stats) renderStats(data.stats);
    renderJournal(data.entries || []);
  } catch(e) {}
}

function renderJournal(entries) {
  const el = document.getElementById('jlist');
  if (!entries.length) {
    el.innerHTML = '<span class="lod">No observations yet.</span>';
    return;
  }
  el.innerHTML = entries.map(e => {
    const dt  = new Date(e.observed_at * 1000).toLocaleString();
    const dir = (e.direction || 'sideways').toLowerCase();
    const dcls = dir === 'up' ? 'jup' : dir === 'down' ? 'jdn' : 'jsw';
    const dlbl = dir === 'up' ? '▲ UP' : dir === 'down' ? '▼ DOWN' : '→ SIDEWAYS';

    let res = '';
    if (e.verified) {
      res = `<div class="jres ${e.was_correct ? 'jok' : 'jno'}">${e.was_correct ? '✓ CORRECT' : '✗ WRONG'} — actual: ${fmt(e.actual_price)}</div>`;
    } else {
      res = `<div class="jres jpnd">⏳ Pending (${e.horizon_min || 60} min horizon)</div>`;
    }

    const rsn = e.reasoning ? esc(e.reasoning.slice(0, 160)) + (e.reasoning.length > 160 ? '…' : '') : '';
    return `<div class="je">
      <div class="jt">${dt} — price ${fmt(e.price_at_obs)}</div>
      <div class="jp"><span class="${dcls}">${dlbl}</span> → ${fmt(e.target)} in ${e.horizon_min || '?'} min (${e.confidence || '?'}%)</div>
      ${rsn ? `<div class="jr">${rsn}</div>` : ''}
      ${res}
    </div>`;
  }).join('');
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Events ────────────────────────────────────────────────────────────────────
document.getElementById('btn-obs').addEventListener('click', observe);
document.getElementById('btn-ref').addEventListener('click', loadChart);
document.getElementById('sym').addEventListener('keydown', e => { if (e.key === 'Enter') observe(); });
document.querySelectorAll('.tfb').forEach(b => b.addEventListener('click', () => {
  tf = b.dataset.tf;
  document.querySelectorAll('.tfb').forEach(x => x.classList.toggle('on', x.dataset.tf === tf));
  loadChart();
}));

// ── Auto-refresh chart every 60s ──────────────────────────────────────────────
initChart();
loadChart();
loadJournal(getSym());
setInterval(loadChart, 60000);
</script>
</body>
</html>"""
