# Weekly runbook

Working directory: `C:\Users\ishaa\FPL`
Activate first: `.\.venv\Scripts\Activate.ps1`

---

## A. After the gameweek finishes

Wait for every fixture, then check bonus has settled before scoring — the
scored file is the permanent record and a premature write corrupts it.

```powershell
python -c "import requests; d=requests.get('https://fantasy.premierleague.com/api/event/<GW>/live/').json(); print(sum(1 for e in d['elements'] if e['stats']['bonus']>0), 'players with bonus')"
```

About 30 means final — three per match across ten fixtures. Then:

```powershell
python backtest.py score  --gw <GW>
python backtest.py report
python backtest.py record --gw <GW>
```

Occasionally, to re-check the fitted components:

```powershell
python backtest.py calibrate-bonus
python backtest.py calibrate-variance
python backtest.py calibrate-gk
```

`report` prints a bootstrap 95% interval on the model-vs-`ep_next` gap. **If it
spans zero that is not evidence of an edge in either direction.** It has spanned
zero since GW5; the honest read is that the edge is unproven.

---

## B. Before the deadline

```powershell
python watch.py       --entry 8447496 --xp xp.csv
python ratings.py     --out ratings.json --fresh --results-csv E0.csv --history-weight 0.5
python projections.py --horizon 5 --deep --ratings ratings.json --priors priors.json --out xp.csv --top 30
python backtest.py    snapshot --gw <GW> --xp xp.csv
python congestion.py  --entry 8447496
python fpl_opt.py     --entry 8447496 --projections xp.csv --horizon 5 --free-transfers <N>
python report.py      --entry 8447496 --out report.html
```

Snapshot **before** looking at any recommendation — that keeps the record
honest. Re-snapshot only if you regenerate `xp.csv`, never after kickoff.

### Draft

```powershell
python draft.py lineup  --entry 274930 --xp xp.csv
python draft.py matchup --entry 274930 --xp xp.csv
python draft.py waivers --entry 274930 --xp xp.csv --horizon 5 --top 20
python draft.py lineup  --entry 239566 --xp xp.csv
python draft.py matchup --entry 239566 --xp xp.csv
python draft.py waivers --entry 239566 --xp xp.csv --horizon 5 --top 20
```

Draft publishes picks only once a deadline has passed, so claims made this week
won't show. Patch them in:

```powershell
python draft.py lineup --entry 274930 --xp xp.csv --add <in> --drop <out>
```

Names or ids, comma-separated, counts must match. Identical names are refused —
use ids. Draft ids differ from classic ids for some players (Sangaré is 565 in
classic, 556 in draft; Barcola 627 vs 628), so check rather than assume.

Same problem in classic between making a transfer and the deadline:

```powershell
python mysquad.py --entry 8447496 --out <sold> --in <bought> --bank <£m>
python fpl_opt.py --squad squad.json --projections xp.csv --horizon 1 --free-transfers 0
```

---

## C. Research and planning

```powershell
python fixtures.py --all --next 6  --ratings ratings.json      # every club's run
python fixtures.py --all --next 14 --ratings ratings.json      # to the wildcard horizon
python fixtures.py --team ARS --next 6 --ratings ratings.json
python rank.py MID --sort value --top 20
python rank.py DEF --max-price 5.5 --top 20
python rank.py MID --no-prior                                  # price-guess players
python diagnose.py <name|id> --ratings ratings.json --priors priors.json
python noprior.py --min-price 5.5 --xp xp.csv --top 25 --template
```

Wildcard build from scratch:

```powershell
python fpl_opt.py --budget 100.8 --projections xp.csv --horizon 5
python fpl_opt.py --budget 100.8 --projections xp.csv --horizon 5 --ban <id>,<id> --lock <id>,<id>
```

`--lock` and `--ban` take **ids only**, not names.

---

## D. Maintenance

**`european.json`** — midweek fixtures the FPL API cannot see. Update when UEFA
publishes a round, roughly monthly. Add Carabao Cup rounds too; cup rotation is
heavier than European rotation. Only fill in rounds near an upcoming deadline —
a stale entry reads as fact.

**`priors.json`** — foreign-league seasons for players with no Premier League
history. The code refuses an entry for anyone who already has FPL data. Pull
numbers from FBref or Transfermarkt; never invent them.

**`advanced.csv`** — Understat npxG, xA, shots, key passes.

```powershell
python advanced.py --out advanced.csv
python advanced.py --offline --raw understat_raw.csv --out advanced.csv
```

Nothing in the model consumes it. Two tests found no measurable gain over
FPL's own xG+xA, so it exists for research and for re-testing once the sample
is larger.

---

## E. Tests

```powershell
foreach ($t in "test_proj","test_minutes","test_priors","test_ratings","test_backtest","test_fpl","test_watch","test_draft") { python "$t.py" > $null; if ($?) { "$t PASS" } else { "$t FAIL" } }
```

Model-question tests, to re-run as the sample grows:

```powershell
python test_defcon.py --ratings ratings.json    # fixture effect on DefCon — POSITIVE, now in the model
python test_form.py                             # form vs xP — form was WORSE
python test_advanced.py                         # Understat vs FPL xG/xA — no difference
python test_minutes_cal.py                      # p_start calibration — clean
```

```powershell
git add -A
git commit -m "<what changed>"
git push
```

Commit `history/` — the snapshots and outcomes are the evaluation record and
cannot be regenerated. `xp.csv`, `ratings.json`, `squad.json` and `report.html`
are gitignored.

---

## Flags worth remembering

| flag | where | why |
|---|---|---|
| `--free-transfers N` | fpl_opt | defaults to 1; you bank up to 5 |
| `--horizon 1` | fpl_opt | when only this week matters (wildcard next week) |
| `--ban` / `--lock` | fpl_opt | ids only |
| `--budget` | fpl_opt | build from scratch instead of from your squad |
| `--squad` | fpl_opt, draft | a squad the API hasn't published yet |
| `--add` / `--drop` | draft | pending claims |
| `--allow-hits` | fpl_opt | permit −4 transfers; off by default |
| `--deep` | projections | last-season rates for everyone; never pass a number |
| `--priors` | projections, diagnose | pass to both or they disagree |
| `--fresh` | ratings, watch | bypass the 15-minute cache |
| `--partial` | backtest score | score only finished fixtures |

---

## Standing reminders

- **Snapshot every gameweek**, including ones where you do nothing. GW1 has no
  snapshot and never can.
- **Your football reads have beaten the model repeatedly** — Cherki's role under
  Maresca, Szoboszlai as a six with a penalty goal, Barry's sustained xG,
  De Cuyper as an attacking wing-back, dropping Bruno. Override when you know
  something the model cannot see. Don't override a number you have no new
  information about: the late Ajer/Verbruggen switch in GW5 cost 9 points.
- **Defenders are the worst-predicted position** all season. The DefCon fixture
  adjustment went in at GW7 and helped in its first week — one week is not proof.
- **Numbers that jump week over week are bugs until explained.** The bonus curve
  and the `games_played` undercount were both found this way.
- **If `diagnose.py` and `xp.csv` disagree for the same player and gameweek, it
  is a bug.** Three times so far, always a flag passed to one and not the other.
- **A recommendation that survives many model revisions is not thereby
  confirmed** if none of those revisions touched the flawed component.
- Players with no prior season run on a price guess — about half the pool.
  `watch.py` flags them. Their underlying can be strong (Kostoulas, Belloumi);
  check Understat before dismissing them.
