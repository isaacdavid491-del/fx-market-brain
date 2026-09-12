"""Orchestration rules: vetoes, thresholds, plan quality and symmetry."""
import pandas as pd
import pytest

from backend.agents.base import LONG, NEUTRAL, SHORT, AgentSignal, BaseAgent, MarketContext
from backend.agents.orchestrator import AgentFarm
from backend.store import resample_ohlcv


class Stub(BaseAgent):
    """A fixed-opinion agent for testing aggregation in isolation."""

    def __init__(self, name, role="analyst", score=0.0, confidence=1.0, weight=1.0,
                 veto=False, multiplier=1.0, levels=None):
        super().__init__(weight=weight)
        self.name = name
        self.role = role
        self._score = score
        self._confidence = confidence
        self._veto = veto
        self._multiplier = multiplier
        self._levels = levels or {}

    def evaluate(self, ctx):
        return self.signal(
            direction=LONG if self._score > 0 else SHORT if self._score < 0 else NEUTRAL,
            score=self._score, confidence=self._confidence, multiplier=self._multiplier,
            veto=self._veto, veto_reason="stub veto" if self._veto else "",
            rationale="stub", levels=self._levels,
        )


def ctx_from(df_1m, **cfg):
    frames = {tf: resample_ohlcv(df_1m, tf) for tf in ("5m", "15m", "1h")}
    base = {"max_staleness_seconds": 10 ** 9, "require_killzone": False}
    base.update(cfg)
    return MarketContext(symbol="NAS100_USD", now_ts=int(df_1m["t"].iloc[-1]) + 60,
                         frames=frames, config=base)


def test_a_single_veto_blocks_the_trade(synthetic_1m):
    farm = AgentFarm(agents=[
        Stub("bull1", score=1.0, weight=2.0),
        Stub("bull2", score=1.0, weight=2.0),
        Stub("gate", role="gate", veto=True, multiplier=0.0),
    ])
    decision = farm.evaluate(ctx_from(synthetic_1m))
    assert decision.action == "STAND_ASIDE"
    assert decision.plan is None
    assert decision.vetoes and decision.vetoes[0]["reason"] == "stub veto"


def test_weak_conviction_does_not_trade(synthetic_1m):
    farm = AgentFarm(agents=[Stub("weak", score=0.1, confidence=0.5)])
    decision = farm.evaluate(ctx_from(synthetic_1m))
    assert decision.action == "STAND_ASIDE"
    assert "conviction" in decision.narrative


def test_a_split_farm_does_not_trade(synthetic_1m):
    """Equal weight on both sides must not produce a trade even if the net
    score happens to clear the threshold."""
    farm = AgentFarm(
        agents=[
            Stub("bull", score=1.0, weight=3.0),
            Stub("bear1", score=-1.0, weight=1.2),
            Stub("bear2", score=-1.0, weight=1.2),
        ],
        config={"min_agreement": 0.9, "entry_threshold": 0.05},
    )
    decision = farm.evaluate(ctx_from(synthetic_1m))
    assert decision.action == "STAND_ASIDE"
    assert "split" in decision.narrative


def test_normalisation_means_extra_agents_do_not_inflate_conviction(synthetic_1m):
    one = AgentFarm(agents=[Stub("a", score=0.8, weight=1.0)])
    many = AgentFarm(agents=[Stub(f"a{i}", score=0.8, weight=1.0) for i in range(5)])
    ctx_a, ctx_b = ctx_from(synthetic_1m), ctx_from(synthetic_1m)
    assert one.evaluate(ctx_a).net_score == pytest.approx(many.evaluate(ctx_b).net_score)


def test_a_produced_plan_always_clears_the_reward_floor(synthetic_1m):
    farm = AgentFarm(
        agents=[Stub("bull", score=1.0, weight=3.0)],
        config={"require_killzone": False, "min_rr": 2.5},
    )
    decision = farm.evaluate(ctx_from(synthetic_1m))
    if decision.plan is not None:
        assert decision.plan.risk_reward >= 2.5


def test_plan_geometry_is_coherent(synthetic_1m):
    farm = AgentFarm(agents=[Stub("bull", score=1.0, weight=3.0)],
                     config={"require_killzone": False})
    decision = farm.evaluate(ctx_from(synthetic_1m))
    if decision.plan:
        plan = decision.plan
        if plan.direction == LONG:
            assert plan.stop < plan.entry < plan.take_profit
        else:
            assert plan.take_profit < plan.entry < plan.stop
        assert plan.units > 0
        assert plan.stop_distance > 0


def test_risk_on_a_plan_stays_inside_the_budget(synthetic_1m):
    """Sizing is in whole contracts including costs, so the realised risk sits
    at or below the budget and within one contract of it."""
    from backend.ict.contracts import get_contract
    farm = AgentFarm(agents=[Stub("bull", score=1.0, weight=3.0)],
                     config={"require_killzone": False, "contract": "MNQ"})
    ctx = ctx_from(synthetic_1m)
    ctx.equity, ctx.risk_per_trade = 100_000.0, 0.005
    decision = farm.evaluate(ctx)
    if decision.plan:
        plan = decision.plan
        contract = get_contract(plan.contract)
        budget = 500.0
        risk_per_unit = plan.stop_distance * contract.dollars_per_point + contract.round_trip_cost
        realised = plan.units * risk_per_unit
        assert realised <= budget + 1e-6
        assert realised + risk_per_unit > budget, "should size up to the budget"
        assert plan.units == int(plan.units), "contracts are whole units"


def test_plan_levels_sit_on_tradable_increments(synthetic_1m):
    """A level that cannot be quoted cannot be an order."""
    from backend.ict.contracts import get_contract
    farm = AgentFarm(agents=[Stub("bull", score=1.0, weight=3.0)],
                     config={"require_killzone": False, "contract": "MNQ"})
    decision = farm.evaluate(ctx_from(synthetic_1m))
    if decision.plan:
        contract = get_contract(decision.plan.contract)
        for level in (decision.plan.entry, decision.plan.stop, decision.plan.take_profit):
            assert level == pytest.approx(contract.round_to_tick(level)), level


def test_net_reward_to_risk_is_reported_and_below_gross(synthetic_1m):
    farm = AgentFarm(agents=[Stub("bull", score=1.0, weight=3.0)],
                     config={"require_killzone": False, "contract": "MNQ"})
    decision = farm.evaluate(ctx_from(synthetic_1m))
    if decision.plan:
        assert decision.plan.net_risk_reward > 0
        assert decision.plan.net_risk_reward < decision.plan.risk_reward
        assert 0 < decision.plan.break_even_rate < 1


def test_pyramiding_is_limited_to_equilibrium_or_better():
    """The October lesson limits long additions to equilibrium or below."""
    farm = AgentFarm()
    assert farm.pyramid_allowed(LONG, price=110.0, equilibrium=120.0) is True
    assert farm.pyramid_allowed(LONG, price=130.0, equilibrium=120.0) is False
    assert farm.pyramid_allowed(SHORT, price=130.0, equilibrium=120.0) is True
    assert farm.pyramid_allowed(SHORT, price=110.0, equilibrium=120.0) is False


def test_direction_is_symmetric_under_price_mirroring(synthetic_1m):
    """Mirroring every price about a constant must flip the decision, not
    change whether the farm trades. Catches sign bugs in the aggregation."""
    df = synthetic_1m.tail(6000).reset_index(drop=True)
    pivot = 2 * float(df["c"].mean())
    mirrored = df.copy()
    mirrored["o"] = pivot - df["o"]
    mirrored["c"] = pivot - df["c"]
    mirrored["h"] = pivot - df["l"]
    mirrored["l"] = pivot - df["h"]

    farm = AgentFarm(config={"require_killzone": False})
    normal = farm.evaluate(ctx_from(df))
    flipped = farm.evaluate(ctx_from(mirrored))

    opposite = {LONG: SHORT, SHORT: LONG, "STAND_ASIDE": "STAND_ASIDE"}
    assert flipped.action == opposite[normal.action], (
        f"{normal.action} did not mirror to {opposite[normal.action]}, "
        f"got {flipped.action}"
    )


def test_decision_serialises_completely(synthetic_1m):
    decision = AgentFarm(config={"require_killzone": False}).evaluate(ctx_from(synthetic_1m))
    out = decision.as_dict()
    assert set(out) >= {"symbol", "action", "net_score", "agreement", "conviction",
                        "plan", "narrative", "vetoes", "agents"}
    assert len(out["agents"]) == 21
    assert all("rationale" in a for a in out["agents"])
    import json
    json.dumps(out)   # must be JSON-serialisable for the API


def test_roster_is_exposed():
    roster = AgentFarm().roster()
    assert len(roster) == 21
    names = {a["name"] for a in roster}
    assert {"htf_bias", "liquidity_sweep", "killzone", "risk_manager"} <= names


def test_stop_is_never_inside_the_noise_band(synthetic_1m):
    """A stop closer than `min_stop_atr` would be taken out by ordinary bar
    noise before the idea had a chance to be wrong."""
    farm = AgentFarm(agents=[Stub("bull", score=1.0, weight=3.0)],
                     config={"require_killzone": False, "min_stop_atr": 0.8,
                             "min_rr": 0.1})
    ctx = ctx_from(synthetic_1m)
    decision = farm.evaluate(ctx)
    assert decision.plan is not None
    assert decision.plan.stop_distance >= 0.8 * ctx.atr(ctx.ltf) - 1e-6


def test_a_tight_structural_anchor_is_widened_not_ignored():
    """The anchor still sets the side of the stop; only its distance is floored."""
    farm = AgentFarm(config={"min_stop_atr": 1.0, "stop_buffer_atr": 0.1})
    entry, atr_val = 100.0, 10.0
    stop = farm._stop_level(LONG, {"ob_bottom": 99.0}, entry, atr_val)
    assert stop < entry                      # still below entry for a long
    assert entry - stop >= 1.0 * atr_val     # widened to the floor

    stop_short = farm._stop_level(SHORT, {"ob_top": 101.0}, entry, atr_val)
    assert stop_short > entry
    assert stop_short - entry >= 1.0 * atr_val


def test_a_wide_anchor_is_left_alone():
    farm = AgentFarm(config={"min_stop_atr": 0.5, "stop_buffer_atr": 0.1})
    stop = farm._stop_level(LONG, {"ob_bottom": 80.0}, entry=100.0, atr_val=10.0)
    assert stop == pytest.approx(80.0 - 1.0)   # anchor minus the buffer, untouched


def test_abstaining_agents_do_not_dilute_the_vote(synthetic_1m):
    """Regression: normalising by the whole roster let agents that correctly
    abstain throttle everyone else, so installing more specialists quietly
    made the farm stop trading."""
    ctx_a, ctx_b = ctx_from(synthetic_1m), ctx_from(synthetic_1m)
    lone = AgentFarm(agents=[Stub("bull", score=0.8, confidence=1.0, weight=2.0)])
    with_abstainers = AgentFarm(agents=[
        Stub("bull", score=0.8, confidence=1.0, weight=2.0),
        *[Stub(f"quiet{i}", score=0.0, confidence=0.0, weight=2.0) for i in range(6)],
    ])
    assert lone.evaluate(ctx_a).net_score == pytest.approx(
        with_abstainers.evaluate(ctx_b).net_score
    )


def test_participation_is_reported_and_gates_a_lone_voice(synthetic_1m):
    """A weighted mean can hit full conviction on one voice, so participation
    has to be measured and floored."""
    farm = AgentFarm(
        agents=[
            Stub("loud", score=1.0, confidence=1.0, weight=1.0),
            *[Stub(f"quiet{i}", score=0.0, confidence=0.0, weight=3.0) for i in range(4)],
        ],
        config={"require_killzone": False, "min_participation": 0.5},
    )
    decision = farm.evaluate(ctx_from(synthetic_1m))
    assert decision.participation < 0.5
    assert decision.action == "STAND_ASIDE"
    assert "has a view" in decision.narrative


def test_participation_passes_when_the_farm_is_engaged(synthetic_1m):
    farm = AgentFarm(
        agents=[Stub(f"bull{i}", score=0.9, confidence=0.9, weight=2.0) for i in range(4)],
        config={"require_killzone": False, "min_participation": 0.5},
    )
    decision = farm.evaluate(ctx_from(synthetic_1m))
    assert decision.participation == pytest.approx(0.9)
