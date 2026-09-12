# FX Market Brain

Two market models behind one FastAPI service:

- **FX signal model** — a gradient-boosted classifier over multi-timeframe
  features for a currency pair. Served at `/` and `/api/signal`.
- **NASDAQ ICT agent farm** — eleven specialist agents that read NASDAQ price
  action through the ICT (Inner Circle Trader) lens and argue their way to a
  single trade plan. Served at `/ict` and `/api/ict/*`.

Both read the same 1-minute candle store, so their timeframes always agree.

> **Analysis only.** Nothing here connects to a broker or places an order.
> Output is model output, not financial advice. Trading index CFDs and futures
> carries substantial risk of loss.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest                       # 89 tests
uvicorn backend.app:app --reload       # open http://localhost:8000/ict
```

Without an `OANDA_TOKEN` the service runs on a deterministic **synthetic**
feed so everything is explorable offline. Every response says which provider
produced it.

```bash
python -m backend.cli agents           # list the eleven agents
python -m backend.cli decide           # run the farm now, with full reasoning
python -m backend.cli backtest --days 14
python -m backend.cli seed --days 30   # download real history (needs a token)
```

## The agent farm in one paragraph

Nine analysts each answer one question — where is higher-timeframe structure
pointing, did structure just break or change character, were stops just raided
and rejected, which liquidity pool is price drawn to, is there an unfilled
imbalance or an unmitigated order block nearby, is price at a premium or a
discount, where are we in the daily accumulation-manipulation-distribution
cycle, and does the S&P confirm this high or low. Their votes are weighted and
normalised. A killzone agent then scales the result by how good the trading
window is, and a risk agent can veto outright. A trade is only planned when
conviction, agreement and reward-to-risk all clear their thresholds, which
works out to roughly one or two setups a day.

Full documentation: **[docs/ict-agent-farm.md](docs/ict-agent-farm.md)**.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | FX dashboard |
| `GET /ict` | Agent farm dashboard: every vote and the resulting plan |
| `GET /api/signal` | FX model signal |
| `GET /api/ict/decision` | Farm decision with all agent reasoning |
| `GET /api/ict/plan` | Just the trade plan |
| `GET /api/ict/backtest` | Walk-forward replay over stored history |
| `GET /api/health` | Service health |

Admin routes (`/api/admin/*`, `/api/ict/admin/*`) require the `x-admin-key`
header to match `ADMIN_KEY`.

## Layout

```
backend/
  store.py             SQLite candle store and timeframe resampling
  app.py               FastAPI app: FX model plus route registration
  ict/                 ICT primitives and New York session windows
  agents/              the eleven agents and the orchestrator
  data/                OANDA and synthetic providers, context assembly
  backtest/            walk-forward engine and metrics
  service.py, api_ict.py, cli.py
tests/                 89 tests
docs/ict-agent-farm.md
```

## Configuration

`OANDA_TOKEN`, `OANDA_ACCOUNT_ID`, `ADMIN_KEY`, `DB_PATH`,
`DEFAULT_INSTRUMENT`, `INGEST_EVERY_SECONDS`, `HISTORY_DAYS`, plus the
`NASDAQ_SYMBOL`, `CORRELATED_SYMBOL` and `ICT_*` settings documented in
[docs/ict-agent-farm.md](docs/ict-agent-farm.md#configuration). `render.yaml`
lists them all for deployment.
