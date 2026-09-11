# Weekly runbook

Working directory: `C:\Users\ishaa\FPL`

If using a venv, activate first: `.\.venv\Scripts\Activate.ps1`

---

## A. Before the deadline

Run in this order. Ratings feed projections feed every decision.

```powershell
cd C:\Users\ishaa\FPL

# 1. what changed since last run
python watch.py --entry 8447496 --xp xp.csv

# 2. refit team attack/defence from results
python ratings.py --out ratings.json --fresh --results-csv E0.csv --history-weight 0.5

# 3. project every player, 5 gameweeks out
python projections.py --horizon 5 --deep --ratings ratings.json --out xp.csv --top 30

# 4. freeze the projection BEFORE looking at decisions
python backtest.py snapshot --gw <GW> --xp xp.csv

# 5. classic squad
python fpl_opt.py --entry 8447496 --projections xp.csv --horizon 5 --free-transfers <N>

# 6. draft leagues
python draft.py lineup  --entry 274930 --xp xp.csv
python draft.py matchup --entry 274930 --xp xp.csv
python draft.py waivers --entry 274930 --xp xp.csv --horizon 5

python draft.py lineup  --entry 239566 --xp xp.csv
python draft.py matchup --entry 239566 --xp xp.csv
python draft.py waivers --entry 239566 --xp xp.csv --horizon 5
```

**If you've claimed since the last deadline**, Draft's API still returns your
previous squad — add the pending moves so lineups and matchups reflect what you
actually own:

```powershell
python draft.py lineup --entry 274930 --xp xp.csv --add Tavernier --drop Sangare
python draft.py lineup --entry 239566 --xp xp.csv --add Richards --drop Konsa
```

Names or ids, comma-separated, and counts must match since a draft transaction
is a swap. Identical names are refused — use ids (there are two Fofanas). The
same flags work on `matchup` and `waivers`, and a pending claim is excluded
from the free-agent pool.
```

Snapshot goes at step 4, before you see any recommendation — that keeps the
record honest. Re-snapshot only if you regenerate `xp.csv`; never after kickoff.

### What to read

**watch.py** — CLUB CHANGES first. A player who moved keeps rates earned in a
different system and nothing else in the pipeline notices. Then ARRIVALS
THREATENING INCUMBENTS (only fires for players you own), then the review queue.

**ratings.py** — the "% data, % prior" line. Below ~40% the table is mostly
prior. Also check the unmapped-teams list is only relegated clubs.

**projections.py** — sanity-check the top 30 against players you'd recognise.
A big week-over-week jump in the top numbers is a bug until proven otherwise.

**fpl_opt.py** — the roll/move edge. Under about +2 over five gameweeks, roll:
banking is worth roughly that and the optimizer can't see it.

**draft.py matchup** — FAVOURED means take the floor, UNDERDOG means take
variance. With 3-1-0, a 40-point loss costs the same as a 1-point loss.

---

## B. After the gameweek finishes

Wait for every fixture. Scoring mid-flight reports 0 for players who haven't
kicked off, which lands as huge over-projections.

```powershell
python backtest.py score --gw <GW>
python backtest.py report
```

`report` prints a bootstrap 95% interval on the model-vs-`ep_next` gap. **If it
spans zero, that is not evidence of an edge in either direction** — a model with
no edge produces gaps of ±0.15 at one gameweek. Expect it to span zero until
roughly GW7.

The calibration bias line is meaningful at small n, since it averages over every
player rather than resting on a rank ordering.

---

## C. Periodically

```powershell
# component calibration — uses outcomes, so every gameweek counts
python backtest.py calibrate-bonus
python backtest.py calibrate-variance

# record a gameweek's outcomes with no projection attached
python backtest.py record --gw <GW>
```

`calibrate-variance` sets `SD_BASE` / `SD_SLOPE` in `draft.py`. It fed P(win)
directly and was over 100% wrong on forwards before being measured.

`calibrate-bonus` sets `BONUS_FLOOR` / `BONUS_SLOPE` / `BONUS_CAP` in
`projections.py`. Judge it on the overall mean (~0.27), not the tail buckets —
those map observed match BPS to bonus, while the model applies a shrunk *rate*.
Different functions.

---

## D. Investigating anything odd

```powershell
python diagnose.py <name or id> --ratings ratings.json
python fpl_opt.py --entry 8447496 --projections xp.csv --horizon 5 --explain <id>
python -c "import csv; print([r for r in csv.DictReader(open('xp.csv',encoding='utf-8')) if r['id']=='<id>'])"
```

If `diagnose.py` and `xp.csv` disagree for the same player and gameweek, that is
a bug, not a rounding difference. It has been three times so far.

---

## E. Housekeeping

```powershell
foreach ($t in "test_proj","test_minutes","test_priors","test_ratings","test_backtest","test_fpl","test_watch","test_draft") { python "$t.py" > $null; if ($?) { "$t PASS" } else { "$t FAIL" } }

git add -A
git commit -m "<what changed>"
git push
```

Commit `history/` — the snapshots and outcomes are the evaluation record and
can't be regenerated. `xp.csv` and `ratings.json` are gitignored.

---

## Flags worth remembering

| flag | where | why |
|---|---|---|
| `--free-transfers N` | fpl_opt | defaults to 1; you bank up to 5 |
| `--ban <id>` / `--lock <id>` | fpl_opt | force a player out or in |
| `--explain <id>` | fpl_opt | dump a player's ep vector by gameweek |
| `--allow-hits` | fpl_opt | permit −4 transfers; off by default |
| `--partial` | backtest score | score only finished fixtures |
| `--fresh` | ratings, watch | bypass the 15-minute cache |
| `--deep` | projections | last-season rates for everyone; never pass a number |
| `--horizon N` | most | 5 is the default working horizon |

---

## Standing reminders

- **Snapshot every gameweek**, including ones where you do nothing. GW1 has no
  snapshot and never can — that week is permanently off the scorecard.
- A recommendation that survives many model revisions is not thereby confirmed.
  If none of those revisions touched the flawed component, stability proves
  nothing. This happened with the Thiaw transfer.
- Numbers that jump week over week are bugs until explained. The bonus curve and
  the `games_played` undercount were both found this way.
- New signings and promoted-club players have no priors and read low for two or
  three gameweeks. `NO_PRIOR_STRENGTH` speeds this up but doesn't eliminate it.
