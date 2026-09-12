# FX Market Brain

Two market models behind one FastAPI service:

- **FX signal model** — a gradient-boosted classifier over multi-timeframe
  features for a currency pair. Served at `/` and `/api/signal`.
- **NASDAQ ICT agent farm** — twenty-one specialist agents that read NASDAQ
  price action through the ICT (Inner Circle Trader) lens and argue their way
  to a single trade plan, priced in a real contract with costs. Served at
  `/ict` and `/api/ict/*`.

Both read the same 1-minute candle store, so their timeframes always agree.

> **Analysis only.** Nothing here connects to a broker or places an order.
> Output is model output, not financial advice. Trading index CFDs and futures
> carries substantial risk of loss.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest                       # 169 tests
uvicorn backend.app:app --reload       # open http://localhost:8000/ict
```

Without an `OANDA_TOKEN` the service runs on a deterministic **synthetic**
feed so everything is explorable offline. Every response says which provider
produced it.

```bash
python -m backend.cli agents           # list the twenty-one agents
python -m backend.cli decide           # run the farm now, with full reasoning
python -m backend.cli backtest --days 14
python -m backend.cli seed --days 30   # download real history (needs a token)
```

## The agent farm in one paragraph

Eighteen analysts each answer one question — where higher-timeframe structure
points, whether structure just broke or changed character, whether stops were
just raided and rejected, which liquidity pool price is drawn to, whether
there is an unfilled imbalance, an unmitigated order block or a gap that has
failed on a close, where price sits in the frozen premarket and opening
ranges, whether delivery is meeting resistance, and whether the S&P confirms
this high or low. Their votes form a confidence-weighted mean, so an agent
that abstains neither argues for a trade nor against one. A killzone agent
then scales the result by how good the trading window is, and a risk agent can
veto outright. A trade is only planned when conviction, participation,
agreement and *net* reward-to-risk all clear their thresholds.

Every plan is priced in a real contract: levels snapped to 0.25-point
increments, whole-contract sizing, commission and adverse-execution costs
charged, and the break-even hit rate reported alongside the gross ratio.

Documentation: **[docs/ict-agent-farm.md](docs/ict-agent-farm.md)** for the
system, **[docs/book-teachings.md](docs/book-teachings.md)** for how an
advanced ICT study book's rules map onto the code.

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
  ict/                 ICT primitives, session windows, contract specs
  agents/              the twenty-one agents and the orchestrator
  data/                OANDA and synthetic providers, context assembly
  backtest/            walk-forward engine and metrics
  service.py, api_ict.py, cli.py
tests/                 169 tests
docs/ict-agent-farm.md
docs/book-teachings.md
```

## Configuration

`OANDA_TOKEN`, `OANDA_ACCOUNT_ID`, `ADMIN_KEY`, `DB_PATH`,
`DEFAULT_INSTRUMENT`, `INGEST_EVERY_SECONDS`, `HISTORY_DAYS`, plus the
`NASDAQ_SYMBOL`, `CORRELATED_SYMBOL` and `ICT_*` settings documented in
[docs/ict-agent-farm.md](docs/ict-agent-farm.md#configuration). `render.yaml`
lists them all for deployment.
