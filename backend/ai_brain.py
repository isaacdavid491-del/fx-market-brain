"""
ai_brain.py — Claude watches raw price data and discovers how the market works on its own.

No pre-defined rules. No borrowed concepts.
Claude reads the numbers, forms its own hypotheses, makes predictions,
learns from outcomes, and builds a growing model of this specific market.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import anthropic

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL          = os.getenv("AI_MODEL", "claude-sonnet-4-6")
DB_PATH           = os.getenv("DB_PATH", "/tmp/fx.db")

_client: Optional[anthropic.Anthropic] = None


def _client_() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ─── System prompt: pure discovery mindset ────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent observer watching raw financial price data.

Your entire task is to figure out how this market works by reading the numbers directly.
You have no trading rules, no indicators, no strategies given to you.
You start from zero and build your own understanding.

The data you receive is:
- Rows of candlestick data: timestamp, open price, high price, low price, close price, volume
- Body% = how much of the candle range was covered by the open-to-close move
- Chg = how much the close changed from the previous close (in 0.0001 units)
- Multiple timeframes of the same instrument simultaneously

Your approach:
1. READ the numbers carefully across all timeframes
2. NOTICE mathematical relationships — what precedes a rise? A fall? A pause?
3. FORM your own hypothesis — name things in your own words if needed
4. PREDICT specifically what you think price will do next and over what timeframe
5. LEARN from your past predictions — what were you right about? What were you wrong about? Why?

When past observations are shown to you with their outcomes, study them seriously.
Update your internal model. Refine your understanding.

The market has patterns — your job is to find them yourself through observation, not rules.

You respond ONLY with valid JSON:
{
  "observations": [
    "<thing you notice in the current numbers, in your own words>",
    "<another observation>",
    "<up to 5 observations total>"
  ],
  "pattern_hypothesis": "<your current best theory about what drives price moves on this instrument — built from what you've seen, refined from past predictions>",
  "what_i_learned": "<if you have past predictions with outcomes, what do they teach you? If first time, write 'First observation — no prior data yet'>",
  "prediction": {
    "direction": "up" | "down" | "sideways",
    "target_price": <number or null>,
    "horizon_minutes": <how many minutes from now>,
    "confidence": <0-100 integer>,
    "reasoning": "<your specific reasoning from the numbers you see>"
  },
  "what_to_watch": "<what specific price behavior in the next period would confirm or deny your current hypothesis>",
  "summary": "<2-3 sentences summarizing what you see and what you expect>"
}"""


# ─── Journal DB helpers ───────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_journal_table() -> None:
    conn = _db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_journal (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol                 TEXT    NOT NULL,
            observed_at            INTEGER NOT NULL,
            price_at_obs           REAL    NOT NULL,
            analysis_json          TEXT    NOT NULL,
            prediction_dir         TEXT,
            prediction_target      REAL,
            prediction_horizon_min INTEGER DEFAULT 60,
            prediction_confidence  INTEGER,
            verified               INTEGER DEFAULT 0,
            actual_price           REAL,
            was_correct            INTEGER,
            verified_at            INTEGER
        )
    """)
    conn.commit()
    conn.close()


def save_observation(symbol: str, price: float, analysis: Dict, horizon_min: int = 60) -> int:
    pred = analysis.get("prediction", {})
    conn = _db()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO ai_journal
          (symbol, observed_at, price_at_obs, analysis_json,
           prediction_dir, prediction_target, prediction_horizon_min, prediction_confidence)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        symbol,
        int(datetime.now(timezone.utc).timestamp()),
        price,
        json.dumps(analysis),
        pred.get("direction"),
        pred.get("target_price"),
        pred.get("horizon_minutes", horizon_min),
        pred.get("confidence"),
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def load_recent_journal(symbol: str, limit: int = 8) -> List[Dict]:
    """Load recent journal entries (with outcomes where available)."""
    conn = _db()
    rows = conn.execute("""
        SELECT * FROM ai_journal
        WHERE symbol = ?
        ORDER BY observed_at DESC
        LIMIT ?
    """, (symbol, limit)).fetchall()
    conn.close()
    result = []
    for r in rows:
        entry = dict(r)
        try:
            entry["analysis"] = json.loads(entry["analysis_json"])
        except Exception:
            entry["analysis"] = {}
        result.append(entry)
    return result


def verify_pending_predictions(symbol: str, get_current_price_fn) -> int:
    """
    Check journal entries whose prediction horizon has elapsed.
    Mark them correct/incorrect based on actual price movement.
    Returns count of newly verified entries.
    """
    now  = int(datetime.now(timezone.utc).timestamp())
    conn = _db()
    pending = conn.execute("""
        SELECT id, price_at_obs, prediction_dir, prediction_target, prediction_horizon_min, observed_at
        FROM ai_journal
        WHERE symbol = ? AND verified = 0
          AND (observed_at + prediction_horizon_min * 60) <= ?
    """, (symbol, now)).fetchall()

    verified = 0
    for row in pending:
        try:
            actual = get_current_price_fn()
            if actual is None:
                continue

            price_at_obs = float(row["price_at_obs"])
            direction    = row["prediction_dir"]
            actual_f     = float(actual)
            move         = actual_f - price_at_obs
            threshold    = price_at_obs * 0.00005   # ~0.5 pip minimum move to count

            if direction == "up":
                correct = 1 if move > threshold else 0
            elif direction == "down":
                correct = 1 if move < -threshold else 0
            else:
                correct = 1 if abs(move) <= threshold * 4 else 0

            conn.execute("""
                UPDATE ai_journal
                SET verified=1, actual_price=?, was_correct=?, verified_at=?
                WHERE id=?
            """, (actual_f, correct, now, row["id"]))
            verified += 1
        except Exception:
            continue

    conn.commit()
    conn.close()
    return verified


def journal_stats(symbol: str) -> Dict:
    """Accuracy stats for the AI's predictions on this symbol."""
    conn = _db()
    row  = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN verified=1 THEN 1 ELSE 0 END) as verified,
            SUM(CASE WHEN was_correct=1 THEN 1 ELSE 0 END) as correct
        FROM ai_journal WHERE symbol = ?
    """, (symbol,)).fetchone()
    conn.close()
    total    = row["total"]    or 0
    ver      = row["verified"] or 0
    correct  = row["correct"]  or 0
    return {
        "total_observations": total,
        "verified": ver,
        "correct": correct,
        "accuracy_pct": round(correct / ver * 100, 1) if ver > 0 else None,
    }


# ─── Build prompt ─────────────────────────────────────────────────────────────

def _build_prompt(symbol: str, raw_data_text: str, journal: List[Dict]) -> str:
    lines = [f"INSTRUMENT: {symbol}"]
    lines.append(f"CURRENT TIME (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("═══ RAW PRICE DATA ═══")
    lines.append(raw_data_text)

    if journal:
        lines.append("\n═══ YOUR PAST OBSERVATIONS AND OUTCOMES ═══")
        lines.append("(Study these. Learn from what you got right and wrong.)\n")
        for entry in reversed(journal):   # show oldest first so narrative builds
            obs_time = datetime.fromtimestamp(entry["observed_at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            an       = entry.get("analysis", {})
            pred     = an.get("prediction", {})
            lines.append(f"── Observation at {obs_time} (price was {entry['price_at_obs']:.5f}) ──")

            # What Claude said at the time
            for obs in (an.get("observations") or [])[:3]:
                lines.append(f"  You noticed: {obs}")

            if pred:
                lines.append(f"  You predicted: {pred.get('direction','?').upper()} → target {pred.get('target_price','?')} in {pred.get('horizon_minutes','?')} min (confidence: {pred.get('confidence','?')}%)")
                lines.append(f"  Your reasoning: {pred.get('reasoning','—')[:200]}")

            # Outcome
            if entry.get("verified"):
                result_txt = "✓ CORRECT" if entry.get("was_correct") else "✗ WRONG"
                lines.append(f"  OUTCOME: {result_txt} — actual price reached {entry.get('actual_price','?'):.5f}")
            else:
                lines.append(f"  OUTCOME: Not yet verified (horizon not elapsed)")
            lines.append("")

    lines.append("═══ YOUR TASK ═══")
    lines.append("Read the raw price data above carefully.")
    lines.append("Study your past observations and outcomes.")
    lines.append("What do you observe? What patterns do you see in these numbers?")
    lines.append("What do you think will happen next? Be specific.")
    lines.append("Respond with your JSON analysis.")
    return "\n".join(lines)


# ─── Main observation function ────────────────────────────────────────────────

def observe_and_predict(symbol: str, raw_data_text: str, current_price: float) -> Dict[str, Any]:
    """
    Main entry point. Feed raw price data + past journal to Claude.
    Claude observes, learns, predicts. Result is saved to journal.
    """
    if not ANTHROPIC_API_KEY:
        return {
            "observations": ["ANTHROPIC_API_KEY not set. Add it to enable AI analysis."],
            "pattern_hypothesis": "—",
            "what_i_learned": "—",
            "prediction": {"direction": "sideways", "target_price": None, "horizon_minutes": 60, "confidence": 0, "reasoning": "No API key."},
            "what_to_watch": "—",
            "summary": "Set ANTHROPIC_API_KEY to enable the AI brain.",
            "_error": "no_api_key",
        }

    journal  = load_recent_journal(symbol, limit=6)
    prompt   = _build_prompt(symbol, raw_data_text, journal)

    try:
        msg = _client_().messages.create(
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

        # Strip markdown fences if present
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                try:
                    result = json.loads(part)
                    break
                except Exception:
                    continue
            else:
                result = {"summary": raw[:300], "observations": [], "prediction": {}, "_raw": raw}
        else:
            result = json.loads(raw)

    except json.JSONDecodeError as e:
        result = {
            "summary": "JSON parse error — model returned non-JSON.",
            "observations": [], "prediction": {}, "pattern_hypothesis": "",
            "what_i_learned": "", "what_to_watch": "", "_parse_error": str(e),
        }
    except Exception as e:
        return {
            "summary": f"API error: {str(e)[:200]}",
            "observations": [], "prediction": {}, "_error": str(e),
        }

    # Save to journal
    try:
        save_observation(symbol, current_price, result)
    except Exception:
        pass

    return result
