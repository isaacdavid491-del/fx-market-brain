"""Agent framework: the market context every agent reads and the signal it returns.

The farm is a set of narrow specialists. Each one looks at the same
`MarketContext`, answers a single question about the market, and returns an
`AgentSignal`. The orchestrator does all the combining, so no agent needs to
know that any other agent exists.

The context computes ICT primitives lazily and caches them per timeframe. That
matters: nine agents each recomputing swings and structure on four timeframes
would be the dominant cost in a backtest.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from backend.ict import core
from backend.ict.sessions import session_weight

log = logging.getLogger("ict.agents")

LONG, SHORT, NEUTRAL = "LONG", "SHORT", "NEUTRAL"


def direction_from_score(score: float, threshold: float = 0.15) -> str:
    if score >= threshold:
        return LONG
    if score <= -threshold:
        return SHORT
    return NEUTRAL


def jsonable(value: Any) -> Any:
    """Convert numpy scalars and containers to plain Python types.

    Evidence dicts are free-form, so a numpy bool or float slipping in would
    otherwise only fail at the API boundary, far from its cause.
    """
    import numpy as _np

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, _np.bool_):
        return bool(value)
    if isinstance(value, _np.integer):
        return int(value)
    if isinstance(value, _np.floating):
        out = float(value)
        return out if _np.isfinite(out) else None
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    return value


@dataclass
class AgentSignal:
    """One specialist's read on the market.

    `score` is signed in [-1, 1]: positive is bullish. `confidence` in [0, 1]
    says how much evidence backs it. They are kept separate so the orchestrator
    can distinguish "mildly bullish and sure" from "very bullish and guessing".
    """
    agent: str
    role: str
    direction: str = NEUTRAL
    score: float = 0.0
    confidence: float = 0.0
    weight: float = 1.0
    rationale: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    multiplier: float = 1.0   # gate/risk agents scale the farm's conviction
    veto: bool = False
    veto_reason: str = ""
    levels: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def contribution(self) -> float:
        return float(self.score) * float(self.confidence) * float(self.weight)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "role": self.role,
            "direction": self.direction,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "weight": round(float(self.weight), 4),
            "contribution": round(self.contribution, 4),
            "multiplier": round(float(self.multiplier), 4),
            "rationale": self.rationale,
            "evidence": jsonable(self.evidence),
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "levels": {k: round(float(v), 5) for k, v in self.levels.items()},
            "error": self.error,
        }


@dataclass
class MarketContext:
    """Everything the agents are allowed to look at, and nothing else.

    `frames` maps a timeframe label to an ascending OHLCV frame that ends at
    or before `now_ts`. The backtester builds these by slicing history, which
    is what keeps live and historical behaviour identical.
    """
    symbol: str
    now_ts: int
    frames: Dict[str, pd.DataFrame]
    correlated_symbol: Optional[str] = None
    correlated_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    equity: float = 100_000.0
    risk_per_trade: float = 0.005
    config: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[Tuple[str, str], Any] = field(default_factory=dict, repr=False)

    # -- timeframe roles ---------------------------------------------------
    @property
    def htf(self) -> str:
        return self.config.get("htf", "1h")

    @property
    def mtf(self) -> str:
        return self.config.get("mtf", "15m")

    @property
    def ltf(self) -> str:
        return self.config.get("ltf", "5m")

    def frame(self, tf: str) -> pd.DataFrame:
        df = self.frames.get(tf)
        if df is None:
            return pd.DataFrame(columns=core_columns())
        return df

    def has(self, tf: str, minimum: int = 30) -> bool:
        return len(self.frame(tf)) >= minimum

    @property
    def price(self) -> float:
        for tf in (self.ltf, self.mtf, self.htf):
            df = self.frame(tf)
            if not df.empty:
                return float(df["c"].iloc[-1])
        return 0.0

    @property
    def session_weight(self) -> float:
        return session_weight(self.now_ts)

    # -- cached primitives -------------------------------------------------
    def _cached(self, key: str, tf: str, builder):
        ck = (key, tf)
        if ck not in self._cache:
            self._cache[ck] = builder(self.frame(tf))
        return self._cache[ck]

    def atr(self, tf: str, period: int = 14) -> float:
        return float(self._cached(f"atr{period}", tf, lambda d: core.atr(d, period)).value)

    def swings(self, tf: str, strength: int = 2) -> List[core.Swing]:
        return self._cached(f"swings{strength}", tf, lambda d: core.find_swings(d, strength))

    def structure(self, tf: str, strength: int = 2):
        return self._cached(
            f"structure{strength}", tf,
            lambda d: core.market_structure(d, strength, self.swings(tf, strength)),
        )

    def fvgs(self, tf: str) -> List[core.FVG]:
        return self._cached("fvgs", tf, lambda d: core.find_fvgs(d, min_size=self.atr(tf) * 0.05))

    def order_blocks(self, tf: str) -> List[core.OrderBlock]:
        def build(d):
            events, _ = self.structure(tf)
            return core.find_order_blocks(d, events)
        return self._cached("obs", tf, build)

    def pools(self, tf: str) -> List[core.LiquidityPool]:
        return self._cached(
            "pools", tf,
            lambda d: core.liquidity_pools(d, self.swings(tf), tolerance=self.atr(tf) * 0.15),
        )

    def sweeps(self, tf: str, lookback: int = 30) -> List[core.Sweep]:
        return self._cached(
            f"sweeps{lookback}", tf,
            lambda d: core.recent_sweeps(d, self.swings(tf), lookback=lookback,
                                         buffer=self.atr(tf) * 0.05),
        )

    def dealing_range(self, tf: str, lookback: int = 60) -> Optional[core.DealingRange]:
        return self._cached(f"range{lookback}", tf, lambda d: core.dealing_range(d, lookback))


def core_columns() -> List[str]:
    return ["t", "o", "h", "l", "c", "v"]


class BaseAgent:
    """One specialist. Subclasses implement `evaluate` only."""

    name: str = "base"
    role: str = "analyst"
    description: str = ""
    default_weight: float = 1.0
    min_bars: int = 30

    def __init__(self, weight: Optional[float] = None, **params: Any):
        self.weight = self.default_weight if weight is None else float(weight)
        self.params = params

    # Subclasses override this.
    def evaluate(self, ctx: MarketContext) -> AgentSignal:  # pragma: no cover - abstract
        raise NotImplementedError

    def signal(self, **kwargs: Any) -> AgentSignal:
        kwargs.setdefault("agent", self.name)
        kwargs.setdefault("role", self.role)
        kwargs.setdefault("weight", self.weight)
        return AgentSignal(**kwargs)

    def neutral(self, rationale: str, **evidence: Any) -> AgentSignal:
        return self.signal(direction=NEUTRAL, score=0.0, confidence=0.0,
                           rationale=rationale, evidence=evidence)

    def run(self, ctx: MarketContext) -> AgentSignal:
        """Never raises: a broken specialist abstains instead of taking the
        farm down, and the failure is reported on the signal."""
        try:
            out = self.evaluate(ctx)
            if out is None:
                return self.neutral("agent returned nothing")
            out.score = float(max(-1.0, min(1.0, out.score)))
            out.confidence = float(max(0.0, min(1.0, out.confidence)))
            return out
        except Exception as exc:  # noqa: BLE001 - deliberate isolation boundary
            log.warning("agent %s failed: %s", self.name, exc)
            sig = self.neutral(f"agent error: {exc}")
            sig.error = traceback.format_exc(limit=3)
            return sig

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name, "role": self.role,
            "description": self.description, "weight": self.weight,
        }
