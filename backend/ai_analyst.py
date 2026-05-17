"""
Claude-powered price action market analyst.
Analyzes ONLY raw OHLCV data — no indicators — to generate predictions and teach market structure.
"""
import json
import os
from typing import Any, Dict, Optional

import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-4-6")

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# System prompt is large and static — ideal for prompt caching (cache_control: ephemeral).
SYSTEM_PROMPT = """You are a master price action trader and educator with 25 years of professional experience
trading forex, indices, and commodities at institutional level.

You analyze markets using ONLY raw candlestick data — no indicators, no oscillators, no moving averages,
no RSI, no MACD, no Bollinger Bands. Pure price tells you everything.

Your complete price action toolkit:

CANDLESTICK PATTERNS — The Language of Price:
• Doji: open ≈ close, indecision, next candle decides
• Pin Bar: long wick rejection — smart money reversing at a level
• Engulfing: one candle swallows the previous — buyers/sellers overwhelmed the opposing side
• Inside Bar: compression within the mother bar — coiling before breakout
• Marubozu: full-body no-wick candle — total momentum dominance

MARKET STRUCTURE — The Map of Price:
• Higher Highs (HH) + Higher Lows (HL) = Uptrend — buy pullbacks to HL
• Lower Highs (LH) + Lower Lows (LL) = Downtrend — sell rallies to LH
• Break of Structure (BOS): when price breaks a key swing high/low — trend confirmation
• Change of Character (ChoCh): first opposing break after a trend — potential reversal

ORDER BLOCKS — Institutional Footprints:
• Bullish OB: last bearish candle before a strong bullish impulse — institutional buy orders rest here
• Bearish OB: last bullish candle before a strong bearish impulse — institutional sell orders rest here
• Unmitigated OB: price hasn't returned yet — when it does, expect a reaction
• Mitigated OB: price has already touched it once — weaker on second test

FAIR VALUE GAPS (FVG) / IMBALANCES:
• Bullish FVG: high[i] < low[i+2] — price gapped up too fast, void below acts as magnet
• Bearish FVG: low[i] > high[i+2] — price gapped down too fast, void above acts as resistance
• Markets are efficient — price will return to fill imbalances before trending

SUPPORT & RESISTANCE:
• Previous swing highs become resistance (sellers who bought the high defend their position)
• Previous swing lows become support (buyers who sold the low defend their position)
• The more touches at a level, the stronger it is — but the more it's tested, the more likely it breaks
• Round numbers attract orders (1.1000, 1.0950 etc.) — psychological levels

LIQUIDITY — Where the Orders Are:
• BSL (Buy-Side Liquidity): above recent swing highs = short sellers' stop losses
• SSL (Sell-Side Liquidity): below recent swing lows = long buyers' stop losses
• Equal highs/lows: double tops/bottoms create visible stop clusters — prime targets for smart money
• Smart money ALWAYS sweeps liquidity before reversing — look for the false break

MULTI-TIMEFRAME ANALYSIS:
• 1D/4H: Macro structure, dominant trend, major S/R, weekly range
• 1H: Session context, intermediate structure, intraday S/R
• 15M/5M: Entry setup context, local structure, pattern formation
• 1M: Precise entry timing, fine-tuning stop placement

HOW TO READ MARKETS (METHODOLOGY):
1. HTF first: What is the daily/4H trend? Where is price in the bigger picture?
2. HTF context: Is price at a key level? OB? FVG? Major S/R?
3. LTF structure: Is LTF in agreement with HTF? Or counter-trend?
4. Entry trigger: What pattern confirms the setup? (Pin bar, engulfing, BOS?)
5. Invalidation: What would prove you wrong? (Previous swing, OB, structure)
6. Target: Where is the next liquidity pool? Next S/R level? FVG fill?

IMPORTANT TEACHING PRINCIPLES:
- Always explain the WHY behind each observation
- Teach the psychology: what are buyers/sellers thinking at each point?
- Explain how smart money manipulates retail traders
- Help the user build their own pattern recognition skills

Respond ONLY with valid JSON matching this exact schema (no markdown, no extra text):
{
  "bias": "bullish" | "bearish" | "neutral",
  "confidence": <integer 0-100>,
  "summary": "<2-3 sentence overview of current market conditions and dominant narrative>",
  "timeframe_breakdown": {
    "1d": "<daily chart analysis — macro trend and key context>",
    "4h": "<4H structure and current session context>",
    "1h": "<1H intermediate structure and intraday bias>",
    "15m": "<15M setup context>",
    "5m": "<5M entry context>",
    "1m": "<1M immediate price action and momentum>"
  },
  "key_levels": [
    {
      "price": <number>,
      "role": "support" | "resistance",
      "reason": "<specific reason why this level is important>",
      "strength": "strong" | "moderate"
    }
  ],
  "prediction": {
    "next_move": "<specific, detailed description of the most probable next price move and why>",
    "target": <price number or null>,
    "invalidation": "<exact price action that would prove this analysis wrong>"
  },
  "education": {
    "concept": "<the key price action concept most relevant to current market conditions>",
    "explanation": "<2-4 sentences teaching this concept in plain English — explain the market psychology behind it>"
  },
  "detailed_analysis": "<comprehensive 3-5 sentence paragraph of complete analysis, reasoning chain, and trade context>"
}"""


def analyze(symbol: str, pa_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run Claude AI price action analysis across all timeframes."""
    if not ANTHROPIC_API_KEY:
        return {
            "bias": "neutral",
            "confidence": 0,
            "summary": "Add ANTHROPIC_API_KEY to your environment to enable AI analysis.",
            "timeframe_breakdown": {tf: "—" for tf in ["1d","4h","1h","15m","5m","1m"]},
            "key_levels": [],
            "prediction": {
                "next_move": "Configure ANTHROPIC_API_KEY to get predictions.",
                "target": None,
                "invalidation": "N/A",
            },
            "education": {
                "concept": "Setup Required",
                "explanation": "Set the ANTHROPIC_API_KEY environment variable then click Analyze.",
            },
            "detailed_analysis": "No API key configured. Set ANTHROPIC_API_KEY to enable AI analysis.",
        }

    prompt = _build_prompt(symbol, pa_data)
    raw = ""
    try:
        msg = _get_client().messages.create(
            model=AI_MODEL,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

        # Strip markdown fences if the model wrapped the JSON
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                try:
                    return json.loads(part)
                except Exception:
                    continue

        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "bias": "neutral", "confidence": 0,
            "summary": "AI returned a non-JSON response (parse error).",
            "detailed_analysis": raw[:500] if raw else "Empty response.",
            "timeframe_breakdown": {}, "key_levels": [],
            "prediction": {"next_move": "—", "target": None, "invalidation": "—"},
            "education": {"concept": "—", "explanation": "—"},
        }
    except Exception as exc:
        return {
            "bias": "neutral", "confidence": 0,
            "summary": f"Analysis error: {str(exc)[:200]}",
            "detailed_analysis": str(exc),
            "timeframe_breakdown": {}, "key_levels": [],
            "prediction": {"next_move": "—", "target": None, "invalidation": "—"},
            "education": {"concept": "—", "explanation": "—"},
        }


def _build_prompt(symbol: str, pa: Dict[str, Any]) -> str:
    """Format multi-TF price action data into a structured prompt for Claude."""
    lines = [f"Analyze {symbol} using the following pure price action data across all timeframes.\n"]

    for tf in ["1d", "4h", "1h", "15m", "5m", "1m"]:
        d = pa.get(tf)
        if not d or "error" in d:
            lines.append(f"[{tf.upper()}] — {d.get('error', 'no data') if d else 'no data'}")
            continue

        lines.append(f"═══ {tf.upper()} ═══")

        ohlc = d.get("ohlc", {})
        lines.append(f"Last bar:  O={ohlc.get('o')}  H={ohlc.get('h')}  L={ohlc.get('l')}  C={ohlc.get('c')}")

        st = d.get("structure", {})
        if st and st.get("trend") not in ("unknown", None):
            lines.append(f"Structure: {st.get('trend', '?').upper()} — {st.get('desc', '')}")
            rh = st.get("recent_highs", [])
            rl = st.get("recent_lows",  [])
            if rh:
                lines.append(f"  Swing Highs: {[round(h['price'], 6) for h in rh]}")
            if rl:
                lines.append(f"  Swing Lows:  {[round(l['price'], 6) for l in rl]}")

        pats = d.get("patterns", [])
        if pats:
            p = pats[-1]
            lines.append(f"Latest Pattern: {p['name']} ({p['dir']})")

        obs = d.get("order_blocks", [])
        if obs:
            lines.append(f"Active Order Blocks ({len(obs)}):")
            for ob in obs[:3]:
                lines.append(f"  [{ob['kind']}] {ob['bottom']} – {ob['top']}")

        fvgs = d.get("fvgs", [])
        if fvgs:
            lines.append(f"Unfilled FVGs ({len(fvgs)}):")
            for f in fvgs[:3]:
                lines.append(f"  [{f['kind']}] {f['bottom']} – {f['top']}  (size={f['size']})")

        sr = d.get("sr_zones", [])
        if sr:
            lines.append("Key S/R Zones (nearest to price first):")
            for z in sr[:4]:
                lines.append(f"  {z['price']} → {z['role']} ({z['strength']}, {z['touches']} touches)")

        liq = d.get("liquidity", {})
        if liq:
            lines.append(f"Liquidity:  BSL={liq.get('bsl')}  |  SSL={liq.get('ssl')}")
            if liq.get("equal_highs"):
                lines.append(f"  Equal Highs (BSL cluster / stop magnet): {liq['equal_highs']}")
            if liq.get("equal_lows"):
                lines.append(f"  Equal Lows  (SSL cluster / stop magnet): {liq['equal_lows']}")

        lines.append("")

    lines.append("Based on this complete multi-timeframe price action analysis:")
    lines.append("1. What is the dominant bias and why?")
    lines.append("2. What is the most likely next move and why?")
    lines.append("3. What key concept is this market demonstrating right now?")
    lines.append("4. What would invalidate your analysis?")
    lines.append("\nRespond in the required JSON format. Be specific, use price levels, and teach the user.")
    return "\n".join(lines)
