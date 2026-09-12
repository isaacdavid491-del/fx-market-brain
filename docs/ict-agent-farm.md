# NASDAQ ICT Agent Farm

A committee of narrow specialists that reads NASDAQ price action through the
ICT (Inner Circle Trader) lens, argues about direction, and produces a single
trade plan with a stop, a target and a position size.

**This is analysis software. It never places an order.** Every plan is
advisory output for a human to accept or reject, and every number it produces
is model output rather than financial advice.

---

## Why a farm rather than one model

ICT is not one signal. It is a stack of independent readings — structure,
liquidity, imbalance, time of day, relative strength — that are supposed to
line up before a trade is worth taking. Encoding that as a single scoring
function buries the reasoning. Encoding it as separate agents means each one
can be read, tested and re-weighted on its own, and the dashboard can show you
*why* the farm stood aside.

Each agent answers exactly one question and has no idea the others exist. The
orchestrator does all the combining.

## The roster

| Agent | Role | Weight | Question it answers |
|---|---|---|---|
| `htf_bias` | analyst | 2.0 | Which way is the higher timeframe pointing, and is price early or late in its range? |
| `market_structure` | analyst | 1.8 | Has structure just broken (BOS) or changed character (CHoCH)? |
| `liquidity_sweep` | analyst | 2.2 | Were stops just raided and rejected (turtle soup)? |
| `liquidity_draw` | analyst | 1.5 | Which pool of resting orders is price most likely drawn to? |
| `fair_value_gap` | analyst | 1.6 | Is price in, or approaching, an unfilled 3-bar imbalance? |
| `order_block` | analyst | 1.6 | Is price at an unmitigated order block, or retesting a breaker? |
| `premium_discount` | analyst | 1.3 | Is price cheap or expensive inside its dealing range, and in the OTE band? |
| `power_of_three` | analyst | 1.4 | Where are we in the daily accumulation → manipulation → distribution cycle? |
| `smt_divergence` | analyst | 1.5 | Does the correlated index confirm this high or low, or refuse to? |
| `killzone` | gate | — | Is this a window worth trading at all? |
| `risk_manager` | risk | — | Is there any reason not to trade right now? |

Analysts vote. Gate and risk agents do not vote: they multiply the farm's
conviction and can veto outright.

## How a decision is made

1. **Analysts vote.** Each contributes `score x confidence x weight`, where
   score is signed in `[-1, 1]` and confidence is separate so that "mildly
   bullish and certain" is distinguishable from "very bullish and guessing".
2. **Normalise.** The net vote is divided by total analyst weight, so adding
   an agent cannot inflate conviction by itself.
3. **Gate.** The killzone and risk agents multiply the result. Any veto ends
   the decision immediately.
4. **Check agreement.** Of the analysts that expressed a view, at least
   `min_agreement` (default 55%) of the voting weight must back the majority
   side. A loud single agent cannot carry a trade.
5. **Build a plan.** Entry prefers a named confluence zone over chasing price.
   The stop goes beyond structural invalidation. The first target is the
   nearest opposing liquidity pool — and if that target does not pay
   `min_rr` (default 1.8:1), *the trade is rejected rather than the target
   stretched*.

A trade requires all of: no veto, conviction past the threshold, enough
agreement, and a plan that pays. In practice that is roughly one to two setups
per day, which is the intended selectivity.

## Killzones

All windows are New York time, so daylight saving is handled by the zone
rather than by an offset.

| Window | NY time | Weight |
|---|---|---|
| Silver bullet (AM) | 10:00–11:00 | 1.00 |
| New York open | 09:30–10:00 | 0.95 |
| NY AM killzone | 07:00–09:30 | 0.80 |
| Silver bullet (PM) | 14:00–15:00 | 0.70 |
| NY PM | 13:30–16:00 | 0.60 |
| London | 02:00–05:00 | 0.55 |
| Asia | 20:00–00:00 | 0.25 |
| Lunch | 11:30–13:00 | 0.15 |

With `ICT_REQUIRE_KILLZONE=true` (the default) the farm stands aside whenever
the session weight is below 0.35, and always at the weekend.

## Running it

```bash
pip install -r requirements-dev.txt

python -m backend.cli agents                  # list the roster
python -m backend.cli decide                  # run the farm now
python -m backend.cli decide --killzone       # enforce session gating
python -m backend.cli seed --days 30          # download history
python -m backend.cli backtest --days 14      # replay over stored history
python -m backend.cli backtest --exit-policy half_at_1R

uvicorn backend.app:app --reload              # then open /ict
```

### HTTP API

| Endpoint | Purpose |
|---|---|
| `GET /ict` | Dashboard: every agent's vote and the resulting plan |
| `GET /api/ict/decision` | Full decision including all agent reasoning |
| `GET /api/ict/plan` | Just the trade plan |
| `GET /api/ict/agents` | Roster and active configuration |
| `GET /api/ict/sessions` | Killzone windows and the current session |
| `GET /api/ict/backtest` | Walk-forward replay over stored history |
| `GET /api/ict/exit-policies` | Position-management policies the replay can use |
| `POST /api/ict/admin/ingest` | Pull latest candles (needs `x-admin-key`) |
| `POST /api/ict/admin/seed` | Backfill history (needs `x-admin-key`) |

`GET /api/ict/decision?ts=<epoch>` evaluates the farm as of any past moment,
which is the quickest way to inspect a setup after the fact.

## Data

`NAS100_USD` from OANDA, with `SPX500_USD` as the correlation partner for the
SMT agent. One-minute candles are stored in SQLite and aggregated up, so every
timeframe comes from one source of truth.

Without an `OANDA_TOKEN` the system falls back to a **synthetic** feed: a
deterministic random walk shaped to the New York volatility profile, with
weekends closed and periodic displacement legs so the primitives have real
structure to find. It exists for development, tests and demos. Every response
reports which provider produced it — check that field before believing a
number.

## Backtesting, and what it does not model

The backtester replays the farm bar by bar. At each step it hands the agents
frames truncated to that moment, built by the same code path as live, and
fills are simulated on 1-minute bars.

Deliberately conservative:
- Market entries fill at the *next* bar's open, never the close the decision used.
- When one bar spans both the stop and the target, the **stop** is assumed first.
- Limit entries must actually be touched within 30 minutes or they expire.
- The daily trade cap and loss limit are fed back into the risk agent, so the
  guardrails bind in replay exactly as they would live.

Not modelled, and therefore optimistic: spread, commission, slippage, gap
risk, overnight financing, and queue position on limit orders. Treat backtest
output as a check on the logic, not as a forecast of returns.

Because a market entry fills at the next bar's open rather than at the close
the decision was made on, a trade's realised reward-to-risk is usually a
little worse than the plan's. That gap is the simulation being honest, not an
accounting error.

### What the synthetic backtest does and does not tell you

The synthetic feed is a random walk, so no strategy can have a real edge on
it. A result near break-even is therefore the *expected* outcome and the only
one worth checking for. A 30-day replay at the default settings:

| | |
|---|---|
| decision points | 6,877 |
| plans produced | 21 |
| trades taken | 17 |
| win rate | 29% |
| expectancy | +0.10 R |
| profit factor | 1.15 |
| max drawdown | 3.4% |

That is fair odds: a 29% hit rate against a median 2.3:1 reward-to-risk is
close to a coin flip, which is exactly what a random walk should produce. The
21 plans from nearly 7,000 decision points also confirm the selectivity is
working out to one or two setups a day.

This measurement does earn its keep. An earlier version of the stop logic
placed stops as close as 0.35 ATR from entry, inside a single bar's range, and
the same replay returned -13 R with a 7% win rate — noise was taking out
positions before the idea could be right or wrong. That is what the
`min_stop_atr` floor fixes, and the backtest is how the flaw was found.

What the synthetic run cannot tell you is whether the ICT model has an edge on
real NASDAQ prices. That needs real history: seed a token-backed feed and
replay against that. Resist tuning thresholds against the synthetic numbers —
on a random walk you would only be fitting noise.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `NASDAQ_SYMBOL` | `NAS100_USD` | Instrument the farm trades |
| `CORRELATED_SYMBOL` | `SPX500_USD` | Partner series for SMT divergence |
| `DATA_PROVIDER` | `auto` | `oanda`, `synthetic`, or auto-detect by token |
| `ICT_HTF` / `ICT_MTF` / `ICT_LTF` | `1h` / `15m` / `5m` | Bias / structure / entry timeframes |
| `ICT_ENTRY_THRESHOLD` | `0.22` | Conviction needed to act |
| `ICT_MIN_AGREEMENT` | `0.55` | Share of voting weight that must agree |
| `ICT_MIN_RR` | `1.8` | Reward-to-risk floor |
| `ICT_REQUIRE_KILLZONE` | `true` | Stand aside outside the session windows |
| `ICT_RISK_PER_TRADE` | `0.005` | Fraction of equity risked per trade |
| `ICT_MAX_TRADES_PER_DAY` | `3` | Hard daily cap |
| `ICT_DAILY_LOSS_LIMIT` | `0.03` | Stop for the day past this drawdown |
| `ICT_CONTRACT_VALUE` | `1.0` | Currency value of a one-point move per unit |
| `ICT_EQUITY` | `100000` | Account size used for sizing |

## Layout

```
backend/
  ict/core.py          swings, structure, FVGs, order blocks, liquidity, ranges
  ict/sessions.py      killzones and day anchors in New York time
  agents/base.py       MarketContext (cached primitives) and BaseAgent
  agents/*.py          the nine analysts, the gate and the risk manager
  agents/orchestrator.py  voting, gating and plan construction
  data/providers.py    OANDA and synthetic feeds
  data/feed.py         store -> MarketContext assembly
  backtest/engine.py   walk-forward replay and metrics
  service.py           the farm as the API sees it
  api_ict.py           HTTP routes
  cli.py               command line entry point
```

## Extending it

Add an agent by subclassing `BaseAgent`, implementing `evaluate(ctx)`, and
appending it to `default_agents()`. Read what you need off the context —
`ctx.swings("15m")`, `ctx.fvgs("5m")`, `ctx.pools("15m")` — all cached per
timeframe. Return a signed score and a confidence, and say why in the
rationale: that text is what appears on the dashboard and in the CLI.

An agent that raises is caught, reported on its own signal, and treated as an
abstention. One broken specialist cannot take the farm down.
