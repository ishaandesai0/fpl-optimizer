# FPL Optimizer

A four-stage pipeline for Fantasy Premier League 2026/27: fit team ratings from
results, project player points from measured per-90 rates, solve the squad as an
integer program, and keep score of how well the projections did.

Built for the 2026/27 season. All data comes from the public FPL API — no auth,
no scraping, no third-party services.

---

## Why this exists

FPL publishes its own expected-points figure (`ep_next`), so the bar for any
model is: does it beat the free number already sitting in the API?

On the opening gameweek of 2026/27, `ep_next` separated the best player in the
game from the median by **2.09 points**. This pipeline separates them by **4.83**,
with the ordering driven by measured rates rather than price. Whether that
translates into better decisions is what `backtest.py` exists to answer — and
until it has a few gameweeks in it, treat every claim here as unproven.

---

## Install

```bash
pip install requests pulp
```

`requests` for the API, `pulp` for the MILP solver (it bundles CBC). Everything
else is standard library — no numpy, no pandas, no scipy.

Optional but recommended: last season's results as a CSV from
[football-data.co.uk](https://www.football-data.co.uk/englandm.php) (the `E0.csv`
for 2025/26). This takes the team ratings from ~14% fitted to ~71% fitted.

---

## The weekly loop

Run these in order, before each deadline:

```bash
python ratings.py     --out ratings.json --fresh --results-csv E0.csv --history-weight 0.5
python projections.py --horizon 5 --deep 600 --ratings ratings.json --overrides overrides.json --out xp.csv --top 30
python backtest.py    snapshot --gw <N> --xp xp.csv
python fpl_opt.py     --entry <YOUR_ID> --projections xp.csv --horizon 5
```

And once the gameweek has finished:

```bash
python backtest.py score --gw <N>
python backtest.py report
```

**Order matters.** Ratings feed projections feed the optimizer. Snapshot before
you look at the optimizer output, so you're freezing a projection you haven't
yet reacted to — but after you've settled on the parameters you'll actually act
on. Never snapshot after kickoff.

Find your entry id by logging into the FPL site and clicking "Points" — the URL
reads `/entry/1234567/event/1`.

---

## Files

### `ratings.py` — team attack and defence coefficients

Fits a Poisson model to actual results:

```
goals_home ~ Poisson(mu * gamma * attack[home] * defence[away])
goals_away ~ Poisson(mu * attack[away] * defence[home])
```

Attack and defence are each normalised to mean 1.0, so attack 1.30 means "30%
more goals than average against an average opponent" and defence 0.80 means
"concedes 20% fewer". `gamma` is home advantage, `mu` the base scoring rate.

This replaces FPL's `strength_overall_*` fields, which are **1-5 integer tiers** —
twenty clubs collapsed into about five values, with attack and defence forced to
move inversely. Under tiers, a side that scores freely and defends badly cannot
be represented at all. Fitted ratings show exactly that shape (Bournemouth at
1.174 attack / 1.063 defence in 2026/27).

Fitting is coordinate ascent — the standard iterative scaling for this model — so
no scipy. Every rating is shrunk toward a tier-derived prior expressed as
`--prior-weight` pseudo-matches per team, which decays in influence as real
results accumulate.

| Flag | Purpose |
|---|---|
| `--fresh` | bypass the 15-minute cache |
| `--results-csv` | football-data.co.uk style CSV of past results |
| `--history-weight` | weight on historical matches vs this season (default 0.5) |
| `--prior-weight` | pseudo-matches of tier prior per team (default 6.0) |
| `--results` | JSON list of `{h,a,hg,ag}` with FPL team ids |

**Read the fitted-on line.** `"71% data, 29% prior"` tells you how much to trust
the table. Below ~40%, the ordering is mostly the prior wearing a continuous coat.

Team names in external CSVs are matched fuzzily, but **"United" and "City" are
never stripped** — doing so collapses Man Utd and Man City onto the same key and
silently swaps two clubs' ratings. Unmapped clubs are reported and skipped;
promoted sides legitimately have no history and stay on the tier prior.

---

### `projections.py` — expected points per player per gameweek

Decomposes FPL scoring into components and models each separately:

```
xP = appearance + goals + assists + clean sheet + conceded
     + saves + defensive contributions + bonus + cards
```

Writes `id,gw,ep` — one row per player per gameweek.

**Three ideas do most of the work.**

*Empirical-Bayes shrinkage.* Every rate is pulled toward a prior, with the weight
of observed evidence growing as minutes accumulate. `PRIOR_MINUTES = 810` means
one 90-minute game carries ~10% weight.

*Exposure-scaled shrinkage for rare events.* Holding the prior's *minutes*
constant is wrong for low-rate stats: a defender's 0.12 xG/90 prior carries only
~1 pseudo-event over 810 minutes, so one freak set-piece header swamps it, while
a striker's 0.90/90 prior carries ~8 and barely moves. Poisson-Gamma says hold
the pseudo-*event* count constant instead (`PRIOR_EVENTS = 4.0`), which makes
required exposure scale inversely with the rate. Applies to `xg`, `xa`, `defcon`.

*Threshold events integrated over the minutes mixture.* DefCon is
`P(count >= threshold)`, which is convex in lambda. Most players either start and
play ~80 minutes or don't play at all — very few reliably play the average.
Evaluating the threshold at *expected* minutes and scaling by `p(appear)` is not
an approximation, it's an underestimate, and a large one. `defcon_points()`
conditions on each scenario instead.

**The minutes model** ranks its evidence: last season's `starts` first, then
goalkeeper depth-chart rank, then price. Price is a good prior for *talent* and a
bad one for *minutes* — keepers and budget defenders are the most nailed players
in the game and among the cheapest.

**Coverage weighting** discounts thin prior seasons. A player with 900 minutes
before an injury has a real but unrepresentative sample; his rates are blended
toward the price prior in proportion to `_minutes / 2400`. Discounted, not
discarded.

| Flag | Purpose |
|---|---|
| `--horizon` | gameweeks to project (default 5) |
| `--deep N` | fetch last-season rates for the N priciest players. **Use 600** — a smaller N corrects expensive players only, biasing mid-price comparisons |
| `--ratings` | `ratings.json` from `ratings.py`; replaces the coarse tiers |
| `--overrides` | manual context (see below) |
| `--prior-minutes` | evidence weight; lower trusts this season sooner |
| `--top N` | how many players to print as a sanity check |

`--deep 600` caches for 24 hours, so the first run each day is slow (~2 min) and
the rest are instant.

---

### `fpl_opt.py` — squad and transfer optimizer

Exact MILP over the full FPL rule set: 15 players (2/5/5/3), ≤3 per club, within
budget, with a legal XI (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD) and captain chosen
**separately for each gameweek** in the horizon. Objective is XI + captain +
`BENCH_WEIGHT` × bench, decayed at `DECAY` per week.

Falls back to a local-search heuristic if PuLP is missing.

| Flag | Purpose |
|---|---|
| `--entry ID` | pull your live squad and bank from the API |
| `--projections` | `xp.csv`; auto-detects the `gw` column and skips its own FDR scaling |
| `--free-transfers` | how many you have banked (max 5 in 2026/27) |
| `--allow-hits` | permit transfers beyond the free ones at −4 each |
| `--lock` / `--ban` | force player ids in or out |
| `--explain ID` | dump a player's ep vector with gameweek labels |
| `--budget` | for a fresh squad or wildcard, instead of `--entry` |
| `--pool-size` | candidates kept per position (default 45); lower it if CBC drags |

**Squad value uses selling price, not `now_cost`.** FPL returns only half of any
price rise, rounded down. Valuing at `now_cost` invents budget that doesn't exist.

**Hits are off by default**, and this is deliberate. The optimizer solves one
decision point in isolation — it cannot see that you get another free transfer
next week, so it systematically overvalues acting now. The printed edge is
measured against rolling zero times, never against *rolling now and moving later*.

**Read the edge as an upper bound.** Under about +2 over a five-week horizon,
roll — the unmodelled option value of banking is worth roughly that on its own.

---

### `backtest.py` — did any of this work?

Point-in-time backtesting is impossible here: `bootstrap-static` is a live
snapshot with no history, so you can't reconstruct what the model would have said
in GW3 once GW9 has been played. Rates, prices and availability have all moved.

So this works **forward**. Snapshot before each deadline, score after the
gameweek resolves, accumulate a genuine out-of-sample record.

```bash
python backtest.py snapshot --gw 2 --xp xp.csv   # before the deadline
python backtest.py score --gw 2                  # after it finishes
python backtest.py report                        # pooled across all weeks
```

Reports Pearson and Spearman correlation, MAE, RMSE, signed bias, calibration by
xP bucket, per-position breakdown, and the biggest misses in both directions.

**Two design choices worth understanding.**

The baseline is FPL's own `ep_next`, frozen at the same moment. That's the free
alternative — if the model can't beat it, the machinery isn't earning its keep,
and `report` says so outright when the Spearman delta is under 0.02.

The headline metric is the **top 60 by xP**, not all players. Overall correlation
is misleading: 400 fringe players correctly projected near zero will inflate it.
Only the top ~60 are ever selectable.

The **calibration table is what tunes the parameters.** If the 4-5 bucket averages
3.2 actual, the model is inflating premiums, and that points at `PRIOR_EVENTS` or
the fixture multiplier. Those constants are currently estimates; calibration is
how they stop being guesses.

Snapshots live in `history/` and should be committed — they're the evaluation
record.

---

### `diagnose.py` — why is this player rated like that?

```bash
python diagnose.py Sangare --ratings ratings.json --overrides overrides.json
python diagnose.py 565 --ratings ratings.json --overrides overrides.json
```

Prints raw season inputs, which prior source fired, price prior vs last season vs
posterior side by side, the minutes breakdown, fixture lambdas, and the xP split
by component. For DefCon players it shows the threshold, projected action count,
and hit probability conditional on starting.

Name lookup is accent-insensitive (`Sangare` finds `M.Sangaré`, `Sesko` finds
`Šeško`). Ambiguous names list the candidates with ids.

**Always pass `--overrides` if you use them.** Without it the tool explains a
model you aren't running — it will warn you, but the warning exists because this
already caused one wrong decision. The total is also cross-checked against
`score_fixture` directly and prints `MISMATCH` if the display breakdown has
drifted from the model.

This is the most useful tool here. Every real bug found so far was caught by
decomposing a number that looked surprising, not by reading code.

---

### `overrides.json` — context the API can't express

FPL's `status` and `chance_of_playing_next_round` are blunt and lag the news. A
player back from a long lay-off is flagged available long before he's playing 90
minutes. A defender promoted by injuries ahead of him carries last season's
backup minutes. None of that is visible in the data.

```json
{
  "players": {
    "11": {
      "p_start": 0.85,
      "note": "Saliba out for months, Timber not back. Role changed."
    }
  }
}
```

| Field | Effect |
|---|---|
| `exclude` | drop from the pool entirely |
| `avail` | scales the chance of featuring at all |
| `p_start` | replaces the start probability outright |
| `minutes_mult` | scales minutes **per start** (a managed return plays 55 minutes, not 55% of matches) |
| `xg90` `xa90` `defcon90` `bps90` | override a per-90 rate |
| `note` | free text, echoed when applied |

The three minutes knobs act on different things and compose without compounding.

**Prefer ids.** A name key matching more than one player is *refused*, not guessed
at — `Sangare` matches two players at different clubs, and binding to the wrong
one is silent and worse than not binding at all.

**An override rebalances every comparison involving that player.** Correcting one
Arsenal defender's minutes and not his direct competitor's made the model
recommend selling the wrong one. When you override someone, check who he competes
with for the same squad slot and make sure that comparison is still fair.

**Delete entries once stale.** An override that outlives its situation is worse
than none. Review weekly.

---

### Tests

```bash
for t in test_proj test_minutes test_priors test_ratings test_backtest test_fpl; do python $t.py; done
```

`test_ratings.py` is the most useful: it simulates seasons from known attack and
defence coefficients and checks the fitter recovers them (r ≈ 0.88 attack, 0.93
defence over a full season, from a deliberately flat prior). `test_backtest.py`
verifies a null model scores Spearman ≈ 0, so the harness can't flatter a broken
model.

---

## Model constants

Set by judgement, not fitted. The backtest exists to replace these with measured
values.

| Constant | File | Value | Meaning |
|---|---|---|---|
| `PRIOR_MINUTES` | projections | 810 | one game ≈ 10% weight |
| `PRIOR_EVENTS` | projections | 4.0 | pseudo-events for rare-rate shrinkage |
| `FULL_SEASON_MINUTES` | projections | 2400 | yardstick for prior coverage |
| `PRICE_ELASTICITY` | projections | xg 0.85, defcon −0.35 | how price informs each prior |
| `k_games` | projections | 4.0 | matches of prior on start probability |
| `BENCH_WEIGHT` | fpl_opt | 0.12 | bench points vs starters |
| `DECAY` | fpl_opt | 0.84 | per-gameweek discount |
| `GAMMA_PRIOR` | ratings | 1.25 | long-run home advantage |
| `prior_weight` | ratings | 6.0 | pseudo-matches of tier prior |

**`PRICE_ELASTICITY["defcon"]` is negative on purpose.** Expensive defenders are
attacking full-backs, who clear and tackle *less* than cheap centre-backs.
Positive scaling there would be actively wrong.

---

## Known limitations

**No multi-week transfer planning.** The optimizer treats each week in isolation.
It can't reason about banking transfers, hit timing, or arriving at a double
gameweek with four in hand. This is the largest remaining gap.

**No chip logic.** Wildcard, Free Hit, Bench Boost and Triple Captain are ignored
entirely. Two sets in 2026/27, first expiring at the GW19 deadline.

**Bonus is the weakest component.** `expected_bonus` is a hand-fit linear map off
BPS. Bonus requires a top-3 finish *within a match*, which depends on teammates.
The BPS also changed for 2026/27 — no penalty for being tackled, CBI now one BPS
per three actions instead of two — so historical rates are biased for full-backs
and wingers.

**Poisson is wrong for DefCon counts.** Defensive actions are overdispersed —
game state drives them, and a centre-back in a team chasing a lead gets far fewer.
Negative binomial is the right family; Poisson understates the tails.

**The minutes prior takes last season's role at face value.** Role changes from
injuries ahead in the pecking order, transfers, or a new manager are invisible.
`overrides.json` is the only fix.

**New signings and promoted-club players have no priors.** Roughly 270 of 610
players fall back to the price heuristic, and a cheap player with a strong real
rate stays underrated for four or five gameweeks before the data catches up.

---

## A note on debugging this

Every real bug found so far produced output that looked completely reasonable:

- The transfer objective double-counted future value, so it always wanted hits.
- Price-based minutes priors collapsed every keeper and budget defender to a third
  of their value.
- The DefCon threshold was evaluated at mean minutes, understating it badly.
- An override on one player silently reordered a decision about a different one.

None of these threw an error. All of them returned confident, plausible
recommendations. What surfaced them was decomposing a number that looked
surprising — which is what `diagnose.py` is for.

Be especially suspicious of a recommendation that survives many model revisions.
If none of those revisions touched the flawed component, stability isn't
confirmation.
