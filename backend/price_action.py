"""
Pure price action analysis — zero indicators.
Detects: market structure (HH/HL/LH/LL), candlestick patterns, order blocks,
fair value gaps, support/resistance, and liquidity levels.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple


# ─── Candle helpers ───────────────────────────────────────────────────────────

def _body(r) -> float: return abs(float(r['c']) - float(r['o']))
def _uw(r) -> float:   return float(r['h']) - max(float(r['o']), float(r['c']))
def _lw(r) -> float:   return min(float(r['o']), float(r['c'])) - float(r['l'])
def _rng(r) -> float:  return float(r['h']) - float(r['l'])
def _bull(r) -> bool:  return float(r['c']) >= float(r['o'])
def _bear(r) -> bool:  return float(r['c']) < float(r['o'])


# ─── 1. Candlestick patterns ──────────────────────────────────────────────────

def detect_patterns(df: pd.DataFrame) -> List[Dict]:
    """Identify named candlestick patterns with educational descriptions."""
    patterns = []
    if len(df) < 3:
        return patterns

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        t    = int(row['t'])
        r    = _rng(row)
        if r < 1e-10:
            continue

        body = _body(row) / r
        uw   = _uw(row)   / r
        lw   = _lw(row)   / r

        # Doji — indecision
        if body < 0.1:
            patterns.append(dict(t=t, name="Doji", dir="neutral",
                desc="Open ≈ Close — buyers and sellers are in perfect balance. "
                     "The next candle decides direction; do NOT trade the doji itself."))
            continue

        # Bullish pin bar (hammer / dragonfly) — rejection of lows
        if lw > 0.60 and body < 0.30 and uw < 0.15:
            patterns.append(dict(t=t, name="Bullish Pin Bar", dir="bullish",
                desc="Sellers pushed price down hard but buyers rejected it even harder. "
                     "The long lower wick = smart money absorbing supply and reversing. "
                     "Most powerful when it appears at a key support or demand zone."))

        # Bearish pin bar (shooting star / gravestone) — rejection of highs
        if uw > 0.60 and body < 0.30 and lw < 0.15:
            patterns.append(dict(t=t, name="Bearish Pin Bar", dir="bearish",
                desc="Buyers pushed price up aggressively but sellers rejected every pip of it. "
                     "The long upper wick = smart money distributing at supply. "
                     "Most reliable when found at a key resistance or supply zone."))

        # Inside bar — compression before breakout
        if float(row['h']) <= float(prev['h']) and float(row['l']) >= float(prev['l']):
            patterns.append(dict(t=t, name="Inside Bar", dir="neutral",
                desc="Price compressed inside the 'mother bar'. Market is coiling under tension. "
                     "Trade the breakout: above mother bar high = bullish, below mother bar low = bearish. "
                     "The bigger the mother bar, the bigger the expected breakout."))

        # Marubozu — total dominance, no wicks
        if body > 0.90:
            d = "bullish" if _bull(row) else "bearish"
            patterns.append(dict(t=t, name=f"{'Bull' if d=='bullish' else 'Bear'} Marubozu", dir=d,
                desc="Near-perfect candle with no wicks — one side dominated with zero opposition. "
                     "Extremely powerful momentum signal. Buyers (or sellers) were in complete control "
                     "from open to close. Trend continuation is highly probable."))

        # Bullish engulfing — buyers overwhelm sellers
        if (_bull(row) and _bear(prev)
                and float(row['c']) > float(prev['o'])
                and float(row['o']) < float(prev['c'])):
            patterns.append(dict(t=t, name="Bullish Engulfing", dir="bullish",
                desc="A bullish candle completely swallows the previous bearish candle. "
                     "Buyers didn't just match sellers — they overwhelmed them entirely. "
                     "This is a reversal signal; the bigger the engulfing candle, the stronger the signal."))

        # Bearish engulfing — sellers overwhelm buyers
        if (_bear(row) and _bull(prev)
                and float(row['c']) < float(prev['o'])
                and float(row['o']) > float(prev['c'])):
            patterns.append(dict(t=t, name="Bearish Engulfing", dir="bearish",
                desc="A bearish candle completely swallows the previous bullish candle. "
                     "Sellers dominated with overwhelming force — high-probability reversal. "
                     "Watch for this at supply zones, resistance, or after exhaustion moves."))

    return patterns[-15:]


# ─── 2. Market structure (swing high/low analysis) ────────────────────────────

def find_swings(df: pd.DataFrame, lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """
    Find swing highs and lows using the fractal method.
    A swing high: highest point within a lookback window on each side.
    A swing low:  lowest point within a lookback window on each side.
    """
    n  = len(df)
    lb = max(1, min(lookback, n // 4))
    highs: List[Dict] = []
    lows:  List[Dict] = []

    for i in range(lb, n - lb):
        h_win = df['h'].iloc[i - lb: i + lb + 1]
        l_win = df['l'].iloc[i - lb: i + lb + 1]

        if float(df['h'].iloc[i]) >= float(h_win.max()):
            highs.append(dict(t=int(df['t'].iloc[i]), price=float(df['h'].iloc[i])))
        if float(df['l'].iloc[i]) <= float(l_win.min()):
            lows.append(dict(t=int(df['t'].iloc[i]), price=float(df['l'].iloc[i])))

    return highs, lows


def classify_structure(highs: List[Dict], lows: List[Dict]) -> Dict:
    """
    Classify market structure from swing sequence.
    HH + HL = uptrend | LH + LL = downtrend | mixed = range/contraction/expansion
    """
    if len(highs) < 2 or len(lows) < 2:
        return dict(trend="unknown", desc="Insufficient swing points. More data needed.")

    rh = sorted(highs, key=lambda x: x['t'])[-3:]
    rl = sorted(lows,  key=lambda x: x['t'])[-3:]

    hh = len(rh) >= 2 and all(rh[i]['price'] > rh[i-1]['price'] for i in range(1, len(rh)))
    hl = len(rl) >= 2 and all(rl[i]['price'] > rl[i-1]['price'] for i in range(1, len(rl)))
    lh = len(rh) >= 2 and all(rh[i]['price'] < rh[i-1]['price'] for i in range(1, len(rh)))
    ll = len(rl) >= 2 and all(rl[i]['price'] < rl[i-1]['price'] for i in range(1, len(rl)))

    if hh and hl:
        trend, desc = "uptrend",     "HH + HL confirmed — Bullish structure. Each pullback to a Higher Low is a buy opportunity."
    elif lh and ll:
        trend, desc = "downtrend",   "LH + LL confirmed — Bearish structure. Each rally to a Lower High is a sell opportunity."
    elif hh and ll:
        trend, desc = "expansion",   "Expanding range — both highs AND lows are extending. Volatile; avoid trading the middle."
    elif lh and hl:
        trend, desc = "contraction", "Contracting range — both highs and lows compressing. Big move incoming; wait for the breakout."
    else:
        trend, desc = "range",       "Mixed structure — ranging market. Trade bounces off the high of range (resistance) and low of range (support)."

    return dict(trend=trend, desc=desc, recent_highs=rh, recent_lows=rl)


# ─── 3. Order blocks ──────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame) -> List[Dict]:
    """
    Bullish OB: last bearish candle immediately before a strong bullish impulse.
    Bearish OB: last bullish candle immediately before a strong bearish impulse.

    These zones mark where large institutions placed pending orders.
    When price returns to an OB, it often bounces — that's the institutional fill.
    """
    obs: List[Dict] = []
    n = len(df)
    if n < 6:
        return obs

    avg_rng = float((df['h'] - df['l']).mean())
    last_c  = float(df['c'].iloc[-1])

    for i in range(1, n - 3):
        r       = df.iloc[i]
        fwd_max = float(df['h'].iloc[i+1:i+4].max())
        fwd_min = float(df['l'].iloc[i+1:i+4].min())

        # Bullish OB: bearish candle followed by a strong up move
        if _bear(r) and (fwd_max - float(r['h'])) > avg_rng * 1.5:
            mitigated = last_c < float(r['c'])
            obs.append(dict(
                t=int(r['t']), kind="bullish_ob",
                top=round(float(r['o']), 6), bottom=round(float(r['c']), 6),
                mitigated=mitigated,
                desc="Bullish OB — institutions placed heavy buy orders here. "
                     "Price returning to this zone = opportunity to enter long with institutions."
            ))

        # Bearish OB: bullish candle followed by a strong down move
        if _bull(r) and (float(r['l']) - fwd_min) > avg_rng * 1.5:
            mitigated = last_c > float(r['c'])
            obs.append(dict(
                t=int(r['t']), kind="bearish_ob",
                top=round(float(r['c']), 6), bottom=round(float(r['o']), 6),
                mitigated=mitigated,
                desc="Bearish OB — institutions placed heavy sell orders here. "
                     "Price returning to this zone = opportunity to enter short with institutions."
            ))

    active = [o for o in obs if not o['mitigated']]
    return active[-8:]


# ─── 4. Fair value gaps (imbalances) ─────────────────────────────────────────

def detect_fvg(df: pd.DataFrame) -> List[Dict]:
    """
    Bullish FVG: high[i] < low[i+2] — price gapped up, leaving a void below.
    Bearish FVG: low[i] > high[i+2] — price gapped down, leaving a void above.

    Price moves too fast for the market to transact fairly, leaving an imbalance.
    Markets are self-correcting — price tends to return to fill these gaps.
    """
    fvgs: List[Dict] = []
    n = len(df)
    if n < 3:
        return fvgs

    last_c = float(df['c'].iloc[-1])

    for i in range(n - 2):
        c1    = df.iloc[i]
        c3    = df.iloc[i + 2]
        mid_t = int(df['t'].iloc[i + 1])

        # Bullish FVG
        if float(c3['l']) > float(c1['h']):
            filled = float(c1['h']) < last_c < float(c3['l'])
            fvgs.append(dict(
                t=mid_t, kind="bullish_fvg",
                top=round(float(c3['l']), 6),
                bottom=round(float(c1['h']), 6),
                size=round(float(c3['l']) - float(c1['h']), 6),
                filled=filled,
                desc="Bullish FVG — price moved up too fast, leaving a void. "
                     "This zone acts as a magnet; price will likely dip back to fill it before continuing higher."
            ))

        # Bearish FVG
        if float(c3['h']) < float(c1['l']):
            filled = float(c3['h']) < last_c < float(c1['l'])
            fvgs.append(dict(
                t=mid_t, kind="bearish_fvg",
                top=round(float(c1['l']), 6),
                bottom=round(float(c3['h']), 6),
                size=round(float(c1['l']) - float(c3['h']), 6),
                filled=filled,
                desc="Bearish FVG — price dropped too fast, leaving a void above. "
                     "This gap acts like a ceiling; when price retraces up, it often stalls here."
            ))

    unfilled = [f for f in fvgs if not f['filled']]
    return unfilled[-8:]


# ─── 5. Support and resistance ────────────────────────────────────────────────

def find_sr(df: pd.DataFrame, tol_pct: float = 0.03) -> List[Dict]:
    """
    Cluster swing highs and lows into S/R zones.
    More price reactions at a level = stronger zone.
    tol_pct: tolerance % for clustering (0.03 = 3 pips on 1.xxxx pair).
    """
    if len(df) < 20:
        return []

    highs, lows = find_swings(df, lookback=min(5, len(df) // 4))
    prices = sorted([h['price'] for h in highs] + [l['price'] for l in lows])
    if not prices:
        return []

    clusters: List[List[float]] = []
    grp: List[float] = [prices[0]]
    for p in prices[1:]:
        if grp[-1] > 0 and (p - grp[-1]) <= (tol_pct / 100) * grp[-1]:
            grp.append(p)
        else:
            clusters.append(grp)
            grp = [p]
    clusters.append(grp)

    zones: List[Dict] = []
    for g in clusters:
        if len(g) < 2:
            continue
        center = sum(g) / len(g)
        zones.append(dict(
            price=round(center, 6),
            touches=len(g),
            strength="strong" if len(g) >= 3 else "moderate",
        ))

    last_c = float(df['c'].iloc[-1])
    for z in zones:
        z['role'] = 'resistance' if z['price'] > last_c else 'support'

    return sorted(zones, key=lambda x: abs(x['price'] - last_c))[:8]


# ─── 6. Liquidity pools ───────────────────────────────────────────────────────

def find_liquidity(df: pd.DataFrame) -> Dict:
    """
    Identify where retail stop losses are clustered — smart money hunts these.
    BSL (buy-side liquidity):  above swing highs — short sellers' stops.
    SSL (sell-side liquidity): below swing lows  — long buyers' stops.
    Equal highs/lows: stop clusters at obvious double tops/bottoms.
    """
    if len(df) < 20:
        return {}

    highs, lows = find_swings(df, lookback=5)
    rh = [h['price'] for h in sorted(highs, key=lambda x: x['t'])[-5:]]
    rl = [l['price'] for l in sorted(lows,  key=lambda x: x['t'])[-5:]]

    eq_h: List[float] = []
    eq_l: List[float] = []
    for i in range(1, len(rh)):
        if rh[i-1] > 0 and abs(rh[i] - rh[i-1]) / rh[i-1] < 0.0001:
            eq_h.append(round((rh[i] + rh[i-1]) / 2, 6))
    for i in range(1, len(rl)):
        if rl[i-1] > 0 and abs(rl[i] - rl[i-1]) / rl[i-1] < 0.0001:
            eq_l.append(round((rl[i] + rl[i-1]) / 2, 6))

    return dict(
        bsl=round(max(rh), 6) if rh else None,
        ssl=round(min(rl), 6) if rl else None,
        equal_highs=list(set(eq_h)),
        equal_lows=list(set(eq_l)),
        desc="BSL above recent highs = short stops (smart money target to trigger short covering). "
             "SSL below recent lows = long stops (smart money sweeps these before reversing up)."
    )


# ─── Full multi-TF analysis ───────────────────────────────────────────────────

def full_analysis(df_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Run complete price action analysis across all timeframes.
    df_map: {'1m': df, '5m': df, '15m': df, '1h': df, '4h': df, '1d': df}
    Returns structured data consumed by the API and the AI analyst.
    """
    result: Dict[str, Any] = {}
    for tf, df in df_map.items():
        if df is None or df.empty or len(df) < 10:
            result[tf] = {"error": "insufficient_data"}
            continue
        try:
            df = df.reset_index(drop=True)
            highs, lows = find_swings(df, lookback=min(5, len(df) // 4))
            last = df.iloc[-1]
            result[tf] = {
                "ohlc": dict(
                    o=round(float(last['o']), 6), h=round(float(last['h']), 6),
                    l=round(float(last['l']), 6), c=round(float(last['c']), 6),
                ),
                "structure":    classify_structure(highs, lows),
                "patterns":     detect_patterns(df)[-5:],
                "order_blocks": detect_order_blocks(df),
                "fvgs":         detect_fvg(df),
                "sr_zones":     find_sr(df),
                "liquidity":    find_liquidity(df),
            }
        except Exception as exc:
            result[tf] = {"error": str(exc)}
    return result
