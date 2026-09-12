# What the study book taught the agents

Sources: *ICT Advanced Day Trading — an illustrated study of his public
execution methods*, independent research editions 0.2 and 0.6, 10 September
2026. Edition 0.6 adds nineteen chapters; its additions are marked below.

Chapter 41 of edition 0.6 sets the precedence rule this repository follows:
the most recent explicit explanation of the same decision governs, earlier
teaching supplies missing background rather than silently replacing a later
refinement, and windows from different lessons stay separate instead of being
merged into one clock.

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


---

## Partial exits: do they improve expectancy?

Chapter 15 describes reducing a position as it works — a favourable partial, a
further reduction when the expected behaviour weakens, tightened protection on
the last unit. Chapter 21 works the arithmetic: three units short at 140
protected at 146 carry 18 point-units of risk, and exits at 130, 134 and 138
realise 18 point-units, which is 1.00 times initial risk rather than the 4.00
a full exit at the far objective would have given.

That arithmetic is implemented in `ExitPolicy` and asserted as a test. R is
always measured against the **initial** stop distance, so moving protection
later never rewrites the risk the trade was taken with, and each leg
contributes its R weighted by the share of the position it closed.

Seven policies were replayed over the same 30 days: all out at target, half at
1R, half at 2R, thirds at 1R and 2R, each of the scaling variants with the
stop moved to breakeven, and one with a trailing stop.

### The aggregate table, which is misleading

| Policy | Trades | Win rate | Expectancy | Max drawdown |
|---|---|---|---|---|
| half_at_1R | 63 | 25% | +0.147 R | 2.1% |
| thirds_1R_2R | 63 | 41% | +0.141 R | 2.7% |
| half_at_1R_trail | 67 | 46% | +0.138 R | 1.9% |
| half_at_2R | 63 | 41% | +0.130 R | 4.2% |
| half_at_1R_breakeven | 67 | 63% | +0.083 R | 2.7% |
| thirds_1R_2R_breakeven | 67 | 63% | +0.069 R | 2.4% |
| all_at_target | 63 | 25% | +0.067 R | 5.2% |

Read naively this says partial exits roughly double expectancy. **That reading
is wrong**, and the error is worth recording because it is easy to make.

A scale-out changes *when* a position closes. The farm holds one position at a
time, so closing earlier changes when the next signal can be acted on, which
reshuffles every trade after it. The policies are not managing the same trades;
they are trading different sequences. Comparing their aggregate expectancy
compares two different samples.

### The paired comparison, which is not

Restricting to the 55 trades every policy actually took, with the same entry
timestamp:

| Policy | Mean R | Std dev of R | Best trade |
|---|---|---|---|
| all_at_target | +0.200 | 2.17 | +11.39 R |
| half_at_1R | +0.201 | 1.41 | +6.19 R |
| thirds_1R_2R | +0.186 | 1.40 | +5.46 R |
| half_at_1R_breakeven | +0.234 | 1.33 | +6.19 R |

Against the baseline, on matched trades:

| Policy | Difference | Standard error | t |
|---|---|---|---|
| half_at_1R | +0.001 R | 0.126 | +0.00 |
| thirds_1R_2R | -0.015 R | 0.137 | -0.11 |
| half_at_1R_breakeven | +0.033 R | 0.167 | +0.20 |

**Partial exits did not improve expectancy.** Every difference is a small
fraction of its own standard error. What they did do is cut the standard
deviation of per-trade returns by 35 to 39%, and roughly halve maximum
drawdown.

This is the textbook result and the theoretically expected one: on a random
walk no exit rule can change the expected value of a position, because the
price process is a martingale and stopping it at a different time does not
alter its mean. Scaling out trades upside for consistency. The mechanism is
visible in one trade and is asserted as a test: half banked at +1R against a
later stop nets zero where the unscaled trade loses a full R, and half banked
at +1R against a +10R run returns 5.5R rather than 10R.

### On breakeven stops

The book warns that moving a stop into an expected supportive retracement can
cause an early exit. The breakeven variants show the shape of that cost: the
win rate jumps from 25% to 63%, which looks like a large improvement and is
not one. Thirty-five of those trades exit at the moved stop rather than at
target, and in the aggregate run the policy's expectancy was *lower* than
plain scaling. On matched trades the difference was again inside the noise.

The lesson is about the metric rather than the policy: a 63% win rate with
worse expectancy than a 25% win rate is the clearest possible demonstration
that win rate does not measure whether a rule is any good.

### What this does and does not establish

The data is synthetic. A random walk is precisely the case where exit policy
provably cannot matter to the mean, so finding that it does not matter is a
check that the accounting is sound, not a finding about markets. On real data
with genuine trend persistence the answer could differ in either direction,
and the book's own comment on its worked example applies exactly: one path
"cannot establish the best policy across a sample."

What is robust and worth carrying forward: the variance reduction is
mechanical rather than statistical, the whole-contract constraint means a
one-contract position cannot be scaled at all, and any future comparison of
exit policies must be paired on matched trades rather than read off aggregate
totals.


---

# Edition 0.6 additions

## Ch. 22 — Staged protection tied to progress

The headline new management rule, and the reason it matters here is that the
book contrasts it directly with the policy tested in the previous round:

> reduce protection by twenty-five percent at quarter progress toward the
> expected objective and by fifty percent at half progress; at three-quarter
> progress require breakeven … This model permits some open risk while banking
> partials; it is different from moving every trade to breakeven at one times
> risk.

The arithmetic is exact and is a test. Entry 100, stop 80, target 180: at 120
the stop moves to 85, at 140 to 90, at 160 to entry. The reduction is always a
fraction of the **original** entry-to-stop distance, and progress is measured
from entry to the chosen objective.

Implemented as `ExitPolicy.stop_ladder`, with `progressive_stop` and
`progressive_stop_half_at_1R` in the policy roster.

## Ch. 23 — Model 13 search windows

An index-futures morning search window of 08:30-11:00 New York, with
checkpoints at 08:30, 09:30, 10:00 and 10:30, and afternoon checkpoints from
13:30 through 15:30. The source is explicit that these are opportunities to
look for the model, not instructions to trade at each one, so `CheckpointAgent`
takes no direction at all and only nudges conviction.

For the afternoon it prefers morning equal highs or lows where present,
otherwise the extremes of the noon-to-13:00 hour, which is that lesson's
reference window and not the broader lunch period.

## Ch. 24 — Venom and the paired signature

The April 2025 tutorial marks the high and low formed from 08:00 through 09:30
New York. That is a specific ninety-minute construction belonging to that
lesson, kept separate from the 07:00-09:00 premarket range of the 2026
material.

The bearish signature is a candle **closing** above the raided high pool, with
inefficient delivery into the area and an inefficient departure away from it.
The paired arrival and departure is the identifying feature; a brief wick above
a high does not recover the description. `VenomAgent` requires the close and
reports the wick-only case as a non-signal.

The lesson also names three participation times — anticipatory, reactive and
deferred — which cannot all be credited with the best entry price.

## Ch. 26 — Two opening-gap exit ladders

The source describes two variants **in the same lesson** and warns that
combining them into one mandatory allocation would misstate it, so both are
implemented and labelled separately:

| Policy | Allocation |
|---|---|
| `gap_bulk_at_half` | 75-80% off at the half-gap objective |
| `gap_ladder` | 2 units at half gap, 7 at full closure, 3 runners at extensions |

The twelve-unit illustration of the second is the book's own arithmetic
choice, not reported fills. Extensions sit at 0.2, 0.5 and 1.0 gap widths
beyond, which the engine expresses as progress values above 1.0. A policy with
extension rungs deliberately does **not** flatten at the target, and the source
is clear about the cost: keeping the final runner can surrender profit against
a perfect target exit, and a stopped runner keeps its actual result.

## Ch. 27 — The risk ladder

Halve risk after a full planned loss, restore it once half that loss is
recovered, halve again after a further loss; Model 13's illustrative
progression runs 2%, 1%, 0.5%, 0.25%. Also halve after five consecutive wins,
which is a sizing policy and not a claim that a streak predicts a loss.

`RiskLadder` implements this with an explicit floor and an explicit recovery
condition, because the book states plainly that the examples do not authorise
halving forever or restoring full risk automatically after any winning trade.
It is off by default so it cannot confound an exit-policy comparison.

## Ch. 37 — Rejection blocks and the body reference

The chapter answers a question the sweep language hides: *which boundary is
price actually crossing?* A bearish rejection block runs from a swing
cluster's highest open-or-close to its highest wick. With a body reference of
108 and a wick high of 110, a return to 109 crosses the body without crossing
the wick, and "liquidity was swept" would conceal which happened.

`RejectionBlockAgent` reports the crossing as `body`, `both` or `neither`. The
candle with the longest wick is not automatically the one with the highest
body, and the two are recorded separately.

## Ch. 38 — Breaker projection anchors

For a bullish breaker, measure the pre-raid low A to the following high B and
project beyond B. The later raid is **excluded** from the measurement. With A
at 100 and B at 108 the width is 8 and one width beyond B is 116; using the
raid low of 97 gives 119, which is arithmetically valid against the wrong
anchor. Both are asserted in tests so the anchor rule cannot drift.

## Appendix A — Corrected suspension block

Edition 0.6 corrects the narrower definition used earlier: a suspension block
is a candle bounded by body separations at both ends, and a conventional
three-candle wick gap is **not** required — neighbouring wick ranges may
overlap. One adjacent body separation alone is still insufficient.

A volume imbalance is the separation between adjacent candle bodies and has
nothing to do with traded volume, which `VolumeImbalanceAgent` says in its own
evidence to stop the name misleading a reader.

## New session windows

| Window | New York time | Source |
|---|---|---|
| `macro_0950` | 09:50-10:10 | ch. 28, 43 |
| `pm_opening_range` | 13:30-14:00 | ch. 24, 31 |
| `market_on_close` | 15:50-16:00 | ch. 32 |
| `venom_0800_0930` | 08:00-09:30 | ch. 24 |
| `overnight` | 00:00-07:00 | ch. 28 |
| `noon_hour` | 12:00-13:00 | ch. 23 |

Lunch was widened from 11:30-13:00 to 11:30-13:30 to match the later teaching.
The market-on-close window keeps both labels in the docs: the earlier lesson
says 15:45-16:00 and the later one 15:50-16:00, and the material does not
establish whether the latter supersedes the former or names a narrower core
inside it.

## Still not implemented

| Teaching | Why |
|---|---|
| Model 13 stop placement | The chapter records a material conflict: the wording says the first candle's low, the verbal explanation selects its high. Different risk, unresolved. |
| Enigma | Presented as a personal approach; the lecture does not release the algorithm and the book refuses to publish a formula. |
| Lunch retracement entry | Its prerequisites are specific and conditional, and the book is explicit it "is not a rule to buy every down morning at noon". |
| The 70% opening-gap and 90% gap-revisit figures | Both are described as claims to measure, not established results. |


---

# Did edition 0.6 make the farm better?

Two separate questions, measured separately.

## The four new agents: no measurable effect

Twenty-one agents against seventeen, same data, same exit policy:

| | 17 agents | 21 agents |
|---|---|---|
| Plans produced | 105 | 107 |
| Trades | 59 | 57 |
| Expectancy | -0.198 R | -0.200 R |

Paired on the 42 trades both configurations took, the difference is **exactly
zero**, with the standard deviation unchanged. That is not a coincidence and
it is worth understanding: the new agents vote on *whether* to trade, but none
of them contributes a level the plan builder anchors a stop or target to. A
trade entered at the same moment therefore gets an identical plan and an
identical outcome. All four can do is change which trades are taken, and on
this sample they barely did.

## The chapter 22 stop ladder: consistently positive, in this simulator

The exit policies were compared paired, on matched trades, across five
independent synthetic markets. One sample would not have settled it — the
first run of this comparison gave a paired difference of +0.001 R with
t = 0.00, and the second, on different data, gave +0.249 R with t = 2.78 for
the same comparison.

Paired difference in mean R against holding to target:

| Seed | Baseline R | `half_at_1R` | `progressive_stop_half_at_1R` |
|---|---|---|---|
| 7 | -0.200 | +0.249 | +0.272 |
| 23 | +0.022 | +0.177 | +0.154 |
| 41 | +0.043 | +0.064 | +0.251 |
| 59 | +0.155 | -0.010 | +0.007 |
| 83 | -0.216 | +0.167 | +0.173 |
| **mean** | | **+0.129** (t 2.83) | **+0.171** (t 3.66) |

Chapter 22's ladder combined with a partial is positive in **all five**
markets. Banking half at one times risk is positive in four of five. Both cut
the standard deviation of per-trade returns by about a third, in every single
market.

The ladder on its own, with no partial, is the weakest of the new policies and
has the lowest win rate of any policy tested. That is consistent with the
source, which describes it as operating while partials are already being
banked: tightening protection without taking anything off simply gets the
position stopped more often.

## Why this is not yet a finding about markets

The effect is consistent and the across-seed t-statistic is strong, but the
measurement lives inside a simulator whose frictions could produce it
mechanically:

- The synthetic feed is a geometric random walk, so price carries a small
  upward drift even though the log steps have none.
- Trades carry a four-hour time stop that exits at the bar midpoint, and the
  median target is around 3.5 R, so most positions never reach their
  objective. Banking at 1 R captures value from the many trades that touch
  1 R and then stall, which is a property of that exit grid rather than of
  the market.
- Within a bar the stop is assumed to fill before any favourable level.

So the honest statement is narrow: **inside this backtester, on synthetic
data, scaling out with a progressive stop consistently improves mean R and
reliably cuts variance.** Whether it survives on real NASDAQ prices is
untested, and the book's own warning stands: one path, or one simulator,
cannot establish the best policy across a sample.

## A correction to the previous round

The earlier conclusion in this document was that partial exits do not improve
expectancy, based on a single paired sample with t = 0.00. That was
under-powered. Five markets give a consistently positive point estimate. The
variance claim from that round survives unchanged and is the more solid of the
two.

## The bug that made all of this measurable

The synthetic feed seeded its generator from Python's built-in `hash()`, which
is randomised per process. Every backtest run as a separate process therefore
received a **different price series**, so no two runs in this repository's
history were ever comparable, and the swing between a +0.147 R expectancy in
one session and -0.218 R in the next was measuring different markets rather
than different strategies. Seeding from `zlib.crc32` fixes it, and a test now
spawns subprocesses to prove determinism holds across process boundaries.

Every number in this section was produced after that fix.
