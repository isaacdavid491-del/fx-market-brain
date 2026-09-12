"""HTTP API for the NASDAQ ICT agent farm.

Analysis and paper-trade planning only. Nothing here places an order with a
broker, and the plan objects are advisory: they describe the trade the farm
would take, for a human to accept or reject.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from backend.service import get_service
from backend.ict.sessions import SESSIONS, primary_session, session_weight, to_ny

router = APIRouter(prefix="/api/ict", tags=["ict"])

TEMPLATE_DIR = Path(__file__).parent / "templates"

DISCLAIMER = (
    "Educational and research software. Signals are model output, not financial "
    "advice, and no order is ever sent to a broker. Trading index CFDs and "
    "futures carries substantial risk of loss."
)


def _check_admin(key: Optional[str]) -> None:
    admin = os.getenv("ADMIN_KEY", "")
    if not admin or key != admin:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/health")
def health() -> Dict[str, Any]:
    service = get_service()
    return {
        "ok": True,
        "symbol": service.symbol,
        "correlated_symbol": service.correlated_symbol,
        "data": service.data_status(),
        "disclaimer": DISCLAIMER,
    }


@router.get("/agents")
def agents() -> Dict[str, Any]:
    service = get_service()
    return {
        "count": len(service.farm.agents),
        "agents": service.farm.roster(),
        "config": service.config,
    }


@router.get("/sessions")
def sessions(ts: Optional[int] = Query(None, description="epoch seconds, default now")) -> Dict[str, Any]:
    now = int(ts or time.time())
    current = primary_session(now)
    return {
        "now_ts": now,
        "new_york_time": to_ny(now).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "current_session": current.name if current else None,
        "session_weight": round(session_weight(now), 3),
        "windows": [
            {
                "name": s.name,
                "start_ny": s.start.strftime("%H:%M"),
                "end_ny": s.end.strftime("%H:%M"),
                "weight": s.weight,
                "note": s.note,
            }
            for s in SESSIONS
        ],
    }


@router.get("/decision")
def decision(
    symbol: Optional[str] = Query(None),
    ts: Optional[int] = Query(None, description="evaluate as of this epoch second"),
    require_killzone: Optional[bool] = Query(None),
    equity: Optional[float] = Query(None, gt=0),
) -> Dict[str, Any]:
    """The farm's current read: every agent's vote, and the plan if any."""
    service = get_service()
    result = service.decide(symbol=symbol, now_ts=ts,
                            require_killzone=require_killzone, equity=equity)
    out = result.as_dict()
    out["disclaimer"] = DISCLAIMER
    return out


@router.get("/plan")
def plan(symbol: Optional[str] = Query(None),
         require_killzone: Optional[bool] = Query(None)) -> Dict[str, Any]:
    """Just the trade plan, for a thin client or an alerting hook."""
    service = get_service()
    result = service.decide(symbol=symbol, require_killzone=require_killzone)
    return {
        "symbol": result.symbol,
        "timestamp": result.timestamp,
        "action": result.action,
        "plan": result.plan.as_dict() if result.plan else None,
        "narrative": result.narrative,
        "vetoes": result.vetoes,
        "disclaimer": DISCLAIMER,
    }


@router.get("/backtest")
def backtest(
    symbol: Optional[str] = Query(None),
    days: int = Query(14, ge=1, le=120),
    step_minutes: int = Query(5, ge=1, le=60),
    require_killzone: Optional[bool] = Query(None),
    include_trades: bool = Query(True),
) -> Dict[str, Any]:
    """Replay the farm over stored history.

    Long windows are slow: every step runs all eleven agents.
    """
    service = get_service()
    out = service.backtest(symbol=symbol, days=days, step_minutes=step_minutes,
                           require_killzone=require_killzone,
                           include_trades=include_trades)
    out["disclaimer"] = DISCLAIMER
    return out


@router.post("/admin/ingest")
def admin_ingest(x_admin_key: Optional[str] = Header(default=None),
                 symbol: Optional[str] = Query(None)) -> Dict[str, Any]:
    _check_admin(x_admin_key)
    return get_service().ingest(symbol)


@router.post("/admin/seed")
def admin_seed(x_admin_key: Optional[str] = Header(default=None),
               days: int = Query(30, ge=1, le=120),
               symbol: Optional[str] = Query(None)) -> Dict[str, Any]:
    _check_admin(x_admin_key)
    return get_service().seed(days=days, symbol=symbol)


def dashboard_html() -> str:
    path = TEMPLATE_DIR / "ict_dashboard.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<h1>NASDAQ ICT agent farm</h1><p>Dashboard template missing.</p>"


dashboard_router = APIRouter(tags=["ict"])


@dashboard_router.get("/ict", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(dashboard_html())
