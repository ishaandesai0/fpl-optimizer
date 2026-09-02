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
python watch.py       --entry <YOUR_ID> --xp xp.csv
python ratings.py     --out ratings.json --fresh --results-csv E0.csv --history-weight 0.5
python projections.py --horizon 5 --deep 600 --ratings ratings.json --out xp.csv --top 30
python backtest.py    snapshot --gw <N> --xp xp.csv
python fpl_opt.py     --entry <YOUR_ID> --projections xp.csv --horizon 5
```

And once the gameweek has finished:

```bash
python backtest.py score --gw <N>
python backtest.py report
```

**Order matters.** `watch.py` first — it reports what changed and who the model
is guessing about, using last week's `xp.csv`. Then ratings feed projections feed
the optimizer. Snapshot before
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
| `--red-weight` | multiplier on matches containing a red card (default 0.45); 1.0 disables |
| `--results` | JSON list of `{h,a,hg,ag}` with FPL team ids |

**Matches with red cards are downweighted.** A dismissal makes the scoreline a
poor sample of eleven-a-side football — Brighton 4-0 Villa in GW1 2026/27 had all
four goals inside 31 minutes after a red. Detected from FPL's fixture stats for
the current season and from `HR`/`AR` columns in a historical CSV. Downweighted,
not discarded.

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

*Prior strength scales with prior quality.* A measured season is real evidence
and should resist a couple of noisy games; a price-derived guess is a placeholder
and should surrender to actual data fast. Giving both the same shrinkage constant
held ~280 players with no Premier League history at their price prior for months
while evidence piled up against it — a midfielder who started both opening games
and led his position in defensive contributions still read as a 53% starter.
`NO_PRIOR_STRENGTH = 0.35` scales the shrinkage constant by how much the prior is
worth, interpolating for partial seasons. It cuts both ways: a no-prior player
with no starts drops faster too. The goalkeeper depth chart is exempt — it's a
structural signal about role, not a price guess, so it keeps full weight.

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
| `--deep` | fetch last-season rates for **every** player. Pass with no number; `--deep N` caps at the N priciest, which corrects expensive players only and biases mid-price comparisons |
| `--ratings` | `ratings.json` from `ratings.py`; replaces the coarse tiers |
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
python diagnose.py Sangare --ratings ratings.json
python diagnose.py 565 --ratings ratings.json
```

Prints raw season inputs, which prior source fired, price prior vs last season vs
posterior side by side, the minutes breakdown, fixture lambdas, and the xP split
by component. For DefCon players it shows the threshold, projected action count,
and hit probability conditional on starting.

Name lookup is accent-insensitive (`Sangare` finds `M.Sangaré`, `Sesko` finds
`Šeško`). Ambiguous names list the candidates with ids.

The total is cross-checked against `score_fixture` directly and prints
`MISMATCH` if the display breakdown has drifted from the model — a diagnostic
that silently disagrees with the model it explains is worse than none.

This is the most useful tool here. Every real bug found so far was caught by
decomposing a number that looked surprising, not by reading code.

---

### `watch.py` — what changed, and who should I check?

```bash
python watch.py --entry <YOUR_ID> --xp xp.csv
```

Run it first each week, before the rest of the pipeline.

**Roster diff.** Caches the player list each run and reports what moved: club
changes, players added or removed from the game, availability flips in both
directions, and price moves. Squad members are marked.

This matters most while the real transfer window is open (through 1 September).
A player who leaves the league keeps his id and flips to status `u`. A player who
changes club keeps his last-season rates — earned in a different system, under a
different manager, in a different role — and **nothing else in the pipeline
notices**. Coverage weighting won't catch it either: the sample is fine, it's
just no longer applicable. A club change is the strongest signal that an override
is needed.

**Arrivals threatening incumbents.** When a player joins a club, everyone already
at that club in that position gets flagged — squad members and anyone carrying an
override especially. This is the gap that let Arsenal's Konsa signing past: he
appears only as "new to the game", and the incumbent's own row is completely
unchanged — same status, same news, same history. A signing is the most common
reason a minutes assumption goes stale and the least visible.

**Review queue.** Ranks players by how much a wrong assumption about them would
cost — impact (horizon xP, or squad membership) times uncertainty (no prior
season, status flag, news text, minutes without starts). Players with an override
already set are demoted rather than dropped, so you still see them but lower.

This reports; it doesn't act. There is no mechanism for feeding a correction back
into the model — see "Why there are no manual overrides" below. What it gives you
is a list of players whose projections rest on thin evidence, so you can weigh
them accordingly.

---

### `draft.py` — FPL Draft companion

```bash
python draft.py lineup  --entry <ID> --xp xp.csv
python draft.py waivers --entry <ID> --xp xp.csv --horizon 5
python draft.py matchup --entry <ID> --xp xp.csv
```

Reuses `xp.csv` but solves a different problem. Draft has no budget, no prices,
no 3-per-club limit and no captain, so `fpl_opt.py` is enforcing constraints that
don't exist. Element ids mostly match the classic game but not always — they're
remapped by name, team and position, since a silent mismatch would attribute one
player's projection to another.

**`lineup`** — highest-xP legal XI. A pure sort once there's no captain.

**`waivers`** — ranks add/drop pairs by their effect on your **starting XI**, not
by raw xP difference against the worst player in your squad. Upgrading a bench
defender who never starts changes nothing; comparing against him inflates every
gain. Reports both the weekly gain and a decayed horizon gain, and the horizon
one is what matters: transactions are unlimited, so you keep whoever you claim
and the player you drop is gone. The two columns routinely disagree.

Also flags **dead squad slots** — long-term injuries returning nothing. These
never appear in the ranking, because a benched player doesn't affect your XI so
swapping him scores zero gain, but they're the cheapest move available.

**`matchup`** — projects your opponent's XI too and reports P(win) under a normal
approximation. With 3 points for a win, a 40-point loss and a 1-point loss cost
the same, so expected points is the wrong objective at the tails:

| P(win) | play for |
|---|---|
| ≥ 60% | the floor — nailed starters, don't chase ceiling you don't need |
| ≤ 40% | variance — a volatile pick at lower xP can raise P(win) above a steady one |
| 40-60% | just field the best XI |

It lists bench swaps that *lower* expected points but *raise* P(win).

`player_sd` is the crudest thing in the project — a hand-fit base and slope per
position. It feeds P(win) directly, so an inflated value makes every matchup look
closer to a coin flip than it is. The residuals in `history/gw*_scored.csv` are
what should replace it.

Not modelled: positional scarcity. A mediocre goalkeeper is worth more when every
other keeper is rostered than his xP suggests, and pricing that needs a
replacement-level calculation per position across the league.

---

### `draft_probe.py` — endpoint discovery

The Draft API isn't documented and differs from the classic game's. This tries the
plausible paths and reports which respond, with their top-level keys. Useful when
something 404s.

```bash
python draft_probe.py --entry <ID> --gw <a gameweek whose deadline has passed>
```

Confirmed working: `entry/{id}/event/{gw}` for picks (note `.../picks` 404s),
`entry/{id}/public` with `league_set` nested inside the `entry` object,
`league/{id}/details`, `league/{id}/element-status`. `watchlist/{id}` is 403.

---

### Why there are no manual overrides

An earlier version let you hand-correct a player's minutes or rates through an
`overrides.json` file, for cases the API can't express — a defender promoted by
injuries ahead of him, a striker easing back from a lay-off, a new signing with
no prior season.

It was removed. Not because the inputs were wrong, but because the mechanism
produced four bugs in a week, three of which returned confident wrong answers
rather than errors:

- Overriding one player's minutes and not his direct competitor's silently
  rebalanced every comparison between them, and made the optimizer recommend
  selling the wrong one of two players at the same club.
- A name key bound to two different players at two different clubs.
- `minutes_mult` compounded through `p_start`, so a 0.6 multiplier produced 0.36
  of the minutes.
- `diagnose.py` reported the unoverridden model while `xp.csv` held the
  overridden one, and the two disagreed with nothing to flag it.

The underlying problem is that "the model is wrong about this player" isn't
checkable, while "the model has no data on this player" is. An override encodes
a belief; nothing downstream can tell whether it was justified, and it goes
stale silently.

What replaced it: nothing, deliberately. The model is allowed to be wrong about
players it has thin evidence on, and `watch.py` reports who those are so you can
discount accordingly. New signings are underrated for roughly four or five
gameweeks until real minutes accumulate — a known, bounded cost, and a smaller
one than a mechanism that can quietly corrupt comparisons you didn't know it
touched.

If this comes back, the defensible version is narrow: permit a correction *only*
where no prior season exists, and refuse it on any player the model already has
measured data for. That's enforceable in code rather than resting on discipline.

---

### Tests

```bash
for t in test_proj test_minutes test_priors test_ratings test_backtest test_fpl test_watch test_draft; do python $t.py; done
```

On Windows PowerShell:

```powershell
foreach ($t in "test_proj","test_minutes","test_priors","test_ratings","test_backtest","test_fpl","test_watch","test_draft") { python "$t.py" > $null; if ($?) { "$t PASS" } else { "$t FAIL" } }
```

`test_draft.py` checks that variance helps when you're the underdog and hurts
when you're favoured — the core H2H claim. `test_ratings.py` is the most useful: it simulates seasons from known attack and
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
| `NO_PRIOR_STRENGTH` | projections | 0.35 | weight of a price guess vs a measured season |
| `BONUS_SLOPE` / `BONUS_CAP` | projections | 0.075 / 1.40 | bonus curve; run `backtest.py calibrate-bonus` |
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
injuries ahead in the pecking order, transfers, or a new manager are invisible
until real minutes accumulate. `k_games = 4`, so each start moves the estimate
meaningfully and it self-corrects within a few gameweeks.

**No concept of pre-season disruption.** After a major tournament, players from
nations that went deep return late and are eased in through August, but their
priors come from a full season at full fitness. The model over-projects all of
them for the first few gameweeks. It self-corrects by ~GW5 as real minutes
accumulate. There is no workaround; discount those players yourself.

**`mins_if_start` can't fully separate substitute minutes.** `bootstrap-static`
has no appearance count, so start-minutes are bounded above by `starts * 90`.
Exact when a player has no starts; caps rather than eliminates the error
otherwise.

**New signings and promoted-club players have no priors.** Roughly 280 of 626
players fall back to the price heuristic. `NO_PRIOR_STRENGTH` makes that prior
yield quickly to real minutes, so the underrating now clears in two or three
gameweeks rather than lingering — but a player with no history and no minutes is
genuinely unknowable and the model says so.

**Mid-season club changes invalidate priors silently.** A player's rates carry
over from his old club even though role and system have changed. `watch.py`
flags the move; the correction is manual.

---

## Status

Live for the 2026/27 season. Projections are snapshotted before each deadline and
scored afterwards, so `history/` accumulates a real out-of-sample record rather
than an in-sample fit. The model constants below are currently judgement calls;
the point of the backtest is to replace them with measured values.

Anyone evaluating this should look at `history/` before the code. A model with a
scorecard is a different thing from a model with claims.

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
