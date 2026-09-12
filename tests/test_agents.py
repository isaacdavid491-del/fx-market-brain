"""Agent behaviour: each specialist must answer its own question, and none of
them may take the farm down."""
import pandas as pd
import pytest

from backend.agents.base import LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext
from backend.agents.orchestrator import AgentFarm, default_agents
from backend.agents.risk import RiskManagerAgent, position_size
from backend.agents.timing import KillzoneAgent
from backend.store import TF_MINUTES, resample_ohlcv


def context_from(df_1m, now_ts=None, **kw):
    frames = {tf: resample_ohlcv(df_1m, tf) for tf in ("5m", "15m", "1h")}
    now = now_ts or int(df_1m["t"].iloc[-1]) + 60
    cfg = kw.pop("config", {})
    cfg.setdefault("max_staleness_seconds", 10 ** 9)
    return MarketContext(symbol="NAS100_USD", now_ts=now, frames=frames, config=cfg, **kw)


def test_every_agent_returns_a_valid_signal(synthetic_1m):
    ctx = context_from(synthetic_1m)
    for agent in default_agents():
        sig = agent.run(ctx)
        assert isinstance(sig, AgentSignal)
        assert sig.direction in (LONG, SHORT, NEUTRAL)
        assert -1.0 <= sig.score <= 1.0
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.error is None, f"{agent.name} raised: {sig.error}"
        assert sig.rationale, f"{agent.name} gave no rationale"


def test_a_broken_agent_abstains_instead_of_raising(synthetic_1m):
    class Exploding(BaseAgent):
        name = "exploding"

        def evaluate(self, ctx):
            raise RuntimeError("boom")

    sig = Exploding().run(context_from(synthetic_1m))
    assert sig.direction == NEUTRAL
    assert sig.confidence == 0.0
    assert "boom" in sig.rationale
    assert sig.error is not None


def test_agents_abstain_on_empty_data():
    empty = {tf: pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"]) for tf in ("5m", "15m", "1h")}
    ctx = MarketContext(symbol="NAS100_USD", now_ts=1_717_000_000, frames=empty)
    for agent in default_agents():
        sig = agent.run(ctx)
        assert sig.error is None
        if sig.role == "analyst":
            assert sig.direction == NEUTRAL


def test_killzone_vetoes_at_the_weekend(synthetic_1m):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    saturday = int(datetime(2024, 6, 1, 12, tzinfo=ZoneInfo("America/New_York")).timestamp())
    sig = KillzoneAgent().run(context_from(synthetic_1m, now_ts=saturday))
    assert sig.veto is True
    assert sig.multiplier == 0.0


def test_killzone_vetoes_outside_the_windows(synthetic_1m):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    dead_hour = int(datetime(2024, 6, 3, 12, 15, tzinfo=ZoneInfo("America/New_York")).timestamp())
    sig = KillzoneAgent().run(context_from(synthetic_1m, now_ts=dead_hour,
                                           config={"require_killzone": True}))
    assert sig.veto is True


def test_killzone_passes_in_the_silver_bullet(synthetic_1m):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    sb = int(datetime(2024, 6, 3, 10, 15, tzinfo=ZoneInfo("America/New_York")).timestamp())
    sig = KillzoneAgent().run(context_from(synthetic_1m, now_ts=sb))
    assert sig.veto is False
    assert sig.multiplier == pytest.approx(1.0)


def test_risk_agent_vetoes_past_the_daily_loss_limit(synthetic_1m):
    ctx = context_from(synthetic_1m, config={"day_pnl_pct": -0.05, "daily_loss_limit_pct": 0.03})
    sig = RiskManagerAgent().run(ctx)
    assert sig.veto is True
    assert "daily loss limit" in sig.veto_reason


def test_risk_agent_vetoes_on_the_trade_count(synthetic_1m):
    ctx = context_from(synthetic_1m, config={"trades_today": 3, "max_trades_per_day": 3})
    sig = RiskManagerAgent().run(ctx)
    assert sig.veto is True


def test_risk_agent_vetoes_stale_data(synthetic_1m):
    ctx = context_from(synthetic_1m, config={"max_staleness_seconds": 60})
    ctx.now_ts = int(synthetic_1m["t"].iloc[-1]) + 100_000
    sig = RiskManagerAgent().run(ctx)
    assert sig.veto is True
    assert "stale" in sig.veto_reason


def test_risk_agent_passes_in_normal_conditions(synthetic_1m):
    sig = RiskManagerAgent().run(context_from(synthetic_1m))
    assert sig.veto is False
    assert sig.multiplier > 0


def test_position_size_risks_exactly_the_budget():
    out = position_size(equity=50_000, risk_pct=0.01, entry=20_000, stop=19_900)
    assert out["risk_currency"] == 500.0
    assert out["units"] == pytest.approx(5.0)
    # Losing at the stop must cost the budget.
    assert out["units"] * (20_000 - 19_900) == pytest.approx(500.0)


def test_position_size_refuses_a_zero_stop():
    assert position_size(100_000, 0.01, 20_000, 20_000)["units"] == 0.0


def test_smt_abstains_without_a_peer(synthetic_1m):
    from backend.agents.smt import SMTDivergenceAgent
    sig = SMTDivergenceAgent().run(context_from(synthetic_1m))
    assert sig.direction == NEUTRAL
    assert "no correlated" in sig.rationale


def test_smt_reads_a_peer_series(synthetic_1m, synthetic_peer_1m):
    from backend.agents.smt import SMTDivergenceAgent
    ctx = context_from(synthetic_1m)
    ctx.correlated_symbol = "SPX500_USD"
    ctx.correlated_frames = {tf: resample_ohlcv(synthetic_peer_1m, tf)
                             for tf in ("5m", "15m", "1h")}
    sig = SMTDivergenceAgent().run(ctx)
    assert sig.error is None
    assert sig.direction in (LONG, SHORT, NEUTRAL)


def test_context_caches_expensive_primitives(synthetic_1m):
    ctx = context_from(synthetic_1m)
    first = ctx.swings("15m")
    second = ctx.swings("15m")
    assert first is second, "swings should be computed once per timeframe"
