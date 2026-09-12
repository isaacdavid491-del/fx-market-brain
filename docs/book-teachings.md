# What the study book taught the agents

Source: *ICT Advanced Day Trading — an illustrated study of his public
execution methods*, independent research edition, 10 September 2026.

The book is a research reconstruction, not a rulebook. It repeatedly marks
which conditions its sources actually specify and which remain unresolved.
That distinction is carried into the code: where the source states an exact
rule or arithmetic result, the implementation is literal and tested against
the book's own worked example. Where the book says a condition is
unestablished, the agent measures what it can and is weighted low.

---

## Chapter by chapter

### Ch. 4-5 — Keep ranges separate, and grade without moving anchors

Three measurements that are routinely conflated are kept apart: the
07:00-09:00 premarket range, the 09:30-10:00 opening range, and the RTH
opening gap running from the prior session's final one-minute close to the
09:30 open. They can have three different midpoints on the same day.

Implemented in `backend/ict/sessions.py` as `premarket_range`,
`opening_range`, `opening_gap` and `new_week_opening_gap`, each returning
explicit endpoints and the time they became known.

The rule that matters most is the one about information: **a range is not
graded until its window has closed.** An opening-range extreme formed at
09:57 was not available at 09:40, and a finished chart hides that. Every
`NamedRange` carries a `complete` flag, and `SessionRangeAgent` refuses to
grade an open window.

Grading is exact arithmetic: level at fraction `q` = `low + q x (high - low)`.
`grade_range` and `project_range` reproduce the book's example — a 100 to 180
range has width 80, midpoint 140, quarters at 120 and 160, eighths every 10
units, and a half-range projection above the high at 220.

### Ch. 7 — Which gaps and order blocks receive attention

Low-resistance delivery combines supportive same-direction closes with gaps
left partly unfilled; repeated full fills and back-and-forth movement mean
resistance. Crucially, resistance is measured **through the route**, not by
the destination: a favourable ending does not erase a path that would have
stopped the trade.

`DeliveryResistanceAgent` scores support ratio, unfilled-gap ratio and path
efficiency over the recent window.

The chapter also supplies the "management only" classification: a late
supportive gap can justify holding a position while being an unattractive new
entry. That is why the agents separate entry zones from reference levels.

### Ch. 8-9 — Wick midpoints and nested references

The measurements are exact and implemented literally:

| Object | Formula |
|---|---|
| Upper wick midpoint | `[max(O, C) + H] / 2` |
| Lower wick midpoint | `[L + min(O, C)] / 2` |
| Whole-candle midpoint | `(H + L) / 2` |
| Consequent encroachment | midpoint of the identified gap or wick |

The book's own check: for `O=108, C=104, H=116, L=102` the upper-wick midpoint
is 112 while the whole-candle midpoint is 109. Naming the wrong object moves
the reference by three units. That example is a test.

`ObsidianWickAgent` measures opposing wick pairs and the interval between
their midpoints. It is deliberately the **lowest-weighted agent in the farm**
(0.9) because the book's verification table lists the qualifying event, the
exact candles and the protection rules as unresolved. Any two opposing wicks
somewhere on a chart do not establish the method, and the agent's rationale
says so.

### Ch. 10 — Inversion requires a defined event

The sharpest rule in the book, and the one the first implementation got
wrong. **A wick through a bullish gap is not a bearish inversion; a close
below is required.**

`find_inversions` qualifies only on a close beyond the far boundary and
records the trigger. The two events stay separate because, as the book puts
it, a long protected at 103.50 could already be stopped out before the
inversion label qualifies — classification does not retroactively protect a
position.

`InversionAgent` carries a high weight (1.8) precisely because this condition
*is* specified exactly.

### Ch. 11 — First displacement and reflection gaps

Three identities that are not interchangeable:

1. **Chronological** — the first gap after 09:30.
2. **Displacement** — the first gap whose leg displaced beyond a relevant swing.
3. **Reflection** — the first later gap opposite in direction to the chronological first.

A later displacing gap can take emphasis without changing which gap was
chronologically first. `first_presented_gaps` returns all three, and records
`knowable_t` separately from the gap's own timestamp, because the middle
candle's label time is not the moment the completed three-candle pattern
became visible.

### Ch. 12 — Entry families

Anticipatory, confirmed and retracement entries carry different information
and must be evaluated separately. The book's warning is explicit: never give
the early entry the price of anticipation and the success filter of later
confirmation.

Every plan carries an `entry_family`, and the backtester reports results
broken down by it.

### Ch. 14 — Pyramiding and total risk

Long additions are limited to equilibrium or below; higher references serve
management rather than new entries. `AgentFarm.pyramid_allowed` enforces this,
mirrored for shorts.

### Ch. 19-20 — Worked entries

The offsets are taken from the worked examples: protection one increment
beyond the invalidation point, target one increment before the reference. The
plan builder applies both through `Contract.offset_ticks`, and rounds
protection *away* from entry so snapping to a tick never quietly tightens a
stop.

The cancellation rules are fixed before results are collected, as the book
insists: cancel an unfilled entry if the target trades first, if protection is
breached before entry, or if the session deadline passes. All three are
implemented, and each cancellation is counted in the result.

### Appendix C — Contract arithmetic and costs

The single largest correction to the original system, which modelled no costs
at all.

| | NQ | MNQ |
|---|---|---|
| Dollars per point | $20 | $2 |
| Tick size | 0.25 | 0.25 |
| Dollars per tick | $5 | $0.50 |
| 11.25-point risk | $225 | $22.50 |

The worked example is reproduced exactly in `backend/ict/contracts.py` and
asserted in tests: 11.25 points of risk on MNQ is $22.50, plus $3 of costs is
$25.50, so a $50 budget buys **one** contract and not two. A 22.50-point
target earns $45 gross and $42 net, making the net reward-to-risk 1.65 rather
than the gross 2.00 read off the chart, and the break-even hit rate 37.78%.

Consequences in the code:
- Every plan level is snapped to a real 0.25 increment. A level that cannot be
  quoted cannot be an order.
- Sizing is in whole contracts, floored, with costs included per unit.
- Plans are accepted or rejected on **net** reward-to-risk, and every plan
  reports the break-even hit rate it needs.
- The backtester books gross and net separately and reports costs paid.

### Appendix D — Preserve failures in the ledger

"Attractive winners alone cannot reveal setup frequency or repeatable
results." The backtest result now counts cancelled and expired orders by
reason alongside filled trades, and reports the break-even rate the realised
sample would have needed.

The appendix also states the rule the engine already followed: if stop and
target fall within one candle the intrabar order is unknown, so use finer data
or an explicit conservative treatment, and never default to the favourable
sequence. Fills simulate on one-minute bars and assume the stop.

---

## What was deliberately not implemented

The book is careful about the limits of its own evidence, and so is this.

| Teaching | Why it is not a rule here |
|---|---|
| Obsidian entry model | The qualifying event, exact candles and stop rules are listed as unresolved. Only the wick geometry is implemented, weighted low. |
| One universal closing rule | The book shows the sources disagree: gap inversion needs a close, but a 2022 structure lesson permits swing violation without one. Only the gap-inversion close requirement is enforced. |
| Conflicting-array override | "A mechanical study must define or disallow the override." The farm disallows it: a local warning simply votes against the thesis. |
| The ~70% opening-gap midpoint figure | The book calls it something to measure, not an established result. It is treated as a candidate draw, and is measurable from the backtest. |
| Stage and phase counting | "Do not equate every second pullback with the favored second stage." No pullback counter is implemented. |

The book's closing warning applies to this repository too: a model name is
less specific than an executable plan, and any probability claim needs a
denominator, instrument, dates, entry, protection and costs.


---

## The backtest

Thirty days of synthetic one-minute data, decisions every five minutes,
killzone gating on, sized on MNQ at 0.5% of a $100,000 account, costs charged
at $3 per contract round trip. The same data and settings for both runs; the
only difference is the six book-derived agents.

| | Before | After |
|---|---|---|
| Decision points | 6,624 | 6,602 |
| Plans produced | 99 | 92 |
| Trades filled | 68 | 66 |
| Win rate | 26% | 29% |
| Break-even rate needed | 23% | 25% |
| Expectancy | +0.14 R | +0.14 R |
| Profit factor | 1.20 | 1.19 |
| Max drawdown | 7.2% | 5.6% |
| Gross | +$4,537 | +$5,079 |
| Costs | $2,478 | $2,070 |
| Net | +$2,059 | +$3,009 |

**Read this as a check on the mechanics, not as evidence about the method.**
The data is a random walk. Both runs land near break-even, both sit within a
couple of points of the hit rate they need just to cover costs, and 66 trades
cannot separate a real difference from noise. The book's own closing warning
applies: any probability claim needs a denominator, instrument, dates, entry,
protection and costs, and this has only the last two.

What the run does establish is that the machinery works: costs are charged,
levels are tradable, orders are cancelled under the stated rules, and the
ledger keeps the trades that did not happen.

### The entry-family split, and why it is a trap

The breakdown by entry family looks decisive:

| Entry family | Trades | Total R |
|---|---|---|
| Retracement (staged limit) | 43 | +13.5 |
| Confirmed (market) | 23 | -4.6 |

It would be easy to conclude that staged limit entries are simply better. The
book warns against exactly this inference, and the cancellation ledger shows
why:

| Cancellation reason | Count |
|---|---|
| Target traded first | 15 |
| Expired unfilled | 10 |
| Protection breached before entry | 1 |

Fifteen times the market reached the target without ever coming back to fill
the limit. Those are moves the retracement family called correctly and earned
nothing from, and they are absent from its R total. The confirmed entry took
every one of those trades at a worse price and carries their losses as well as
their wins.

Comparing the two families on filled trades alone gives the early entry the
price of anticipation and the success filter of confirmation, which is the
error the book names directly. The honest comparison has to include the
twenty-six unfilled orders, which is why the engine counts them.

### A bug the backtest found

The first run of this comparison produced **one trade in 7,028 decisions**
after the six agents were added. The agents were not at fault. Conviction was
being normalised by the total weight of the roster, so an agent that correctly
abstained still sat in the denominator and throttled every other agent's vote.
Installing more specialists made the farm quieter, which is the opposite of
what a committee should do.

The vote is now a confidence-weighted mean: an abstaining agent leaves both
the numerator and the denominator, so it neither argues for a trade nor
against one. Because a mean can reach full conviction on a single voice, a
`participation` measure and floor were added alongside it, and both are
reported on every decision.
