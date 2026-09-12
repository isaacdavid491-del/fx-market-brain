"""API surface tests. These run against the synthetic feed."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATA_PROVIDER", "synthetic")


@pytest.fixture(scope="module")
def client():
    from backend.app import app
    # Skip the startup hook: it seeds history over the network.
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def seeded():
    """Put enough NASDAQ history in the store for the farm to read."""
    from backend.data.providers import SyntheticProvider
    from backend.store import init_db, upsert_1m
    init_db()
    for symbol, seed, price in (("NAS100_USD", 7, 20_000.0), ("SPX500_USD", 11, 5_200.0)):
        df = SyntheticProvider(base_price=price, seed=seed).fetch_1m(symbol, count=12_000)
        upsert_1m(symbol, df)
    return True


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert res.json()["ict_farm"] == "/ict"


def test_ict_health(client, seeded):
    res = client.get("/api/ict/health")
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == "NAS100_USD"
    assert "disclaimer" in body
    assert body["data"]["symbols"]["NAS100_USD"]["bars_stored"] > 0


def test_agents_endpoint_lists_the_roster(client):
    body = client.get("/api/ict/agents").json()
    assert body["count"] == 21
    names = {a["name"] for a in body["agents"]}
    assert "liquidity_sweep" in names and "risk_manager" in names


def test_sessions_endpoint(client):
    body = client.get("/api/ict/sessions").json()
    assert "windows" in body and len(body["windows"]) >= 6
    assert "new_york_time" in body
    assert any(w["name"] == "silver_bullet_am" for w in body["windows"])


def test_decision_endpoint_returns_every_agent(client, seeded):
    res = client.get("/api/ict/decision", params={"require_killzone": "false"})
    assert res.status_code == 200
    body = res.json()
    assert body["action"] in ("LONG", "SHORT", "STAND_ASIDE")
    assert len(body["agents"]) == 21
    assert all("rationale" in a and "contribution" in a for a in body["agents"])
    assert "disclaimer" in body


def test_decision_accepts_a_historical_timestamp(client, seeded):
    from backend.store import latest_ts
    ts = latest_ts("NAS100_USD")
    body = client.get("/api/ict/decision",
                      params={"ts": ts - 7200, "require_killzone": "false"}).json()
    assert body["timestamp"] == ts - 7200


def test_plan_endpoint_is_a_thin_view(client, seeded):
    body = client.get("/api/ict/plan", params={"require_killzone": "false"}).json()
    assert set(body) >= {"symbol", "action", "plan", "narrative", "vetoes"}


def test_admin_routes_require_the_key(client):
    assert client.post("/api/ict/admin/ingest").status_code == 401
    assert client.post("/api/ict/admin/seed").status_code == 401
    assert client.post("/api/ict/admin/ingest",
                       headers={"x-admin-key": "wrong"}).status_code == 401


def test_admin_ingest_with_the_key(client, monkeypatch):
    monkeypatch.setenv("ADMIN_KEY", "s3cret")
    res = client.post("/api/ict/admin/ingest",
                      params={"symbol": "NAS100_USD"},
                      headers={"x-admin-key": "s3cret"})
    assert res.status_code == 200
    assert res.json()["results"][0]["ok"] is True


def test_dashboard_renders(client):
    res = client.get("/ict")
    assert res.status_code == 200
    assert "NASDAQ ICT Agent Farm" in res.text
    assert "/api/ict/decision" in res.text


def test_backtest_endpoint(client, seeded):
    res = client.get("/api/ict/backtest",
                     params={"days": 3, "step_minutes": 30,
                             "require_killzone": "false", "include_trades": "true"})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "metrics" in body and "trades" in body["metrics"]
    assert body["metrics"]["trades"] >= 0


def test_backtest_reports_missing_history(client):
    res = client.get("/api/ict/backtest", params={"symbol": "NOT_A_SYMBOL", "days": 3})
    body = res.json()
    assert body["ok"] is False
    assert "seed it first" in body["error"]


def test_validate_endpoint_reports_status(client, seeded):
    res = client.get("/api/ict/validate",
                     params={"days": 10, "step_minutes": 60, "seed": "false"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("real", "synthetic", "no_data")
    assert "headline" in body and "text" in body
    # The synthetic feed must never be reported as a valid validation.
    assert body["valid"] is False
    assert "NOT A TEST" in body["headline"] or "NOTHING WAS TESTED" in body["headline"]
