#!/usr/bin/env python3
"""
projections.py — expected-points model for Fantasy Premier League 2026/27.

Builds a per-gameweek xP for every player by decomposing FPL scoring into its
components and modelling each one separately:

    xP = appearance + goals + assists + clean sheet + conceded
         + saves + defensive contributions + bonus + cards

Rather than trusting FPL's own `ep_next`, every rate is estimated with
empirical-Bayes shrinkage toward a position prior, so a player with 90 minutes
played is mostly prior and a player with 1800 minutes is mostly signal. Team
goals for/against come from a Poisson model driven by FPL's own strength
ratings, which is what makes clean sheets and DefCon tractable.

Usage
-----
    python projections.py --horizon 5 --out xp.csv
    python fpl_opt.py --entry 8447496 --projections xp.csv --horizon 5

Output is `id,gw,ep` — one row per player per gameweek. fpl_opt.py detects the
`gw` column and uses the values directly instead of applying its own crude
fixture-difficulty scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import unicodedata
from typing import NamedTuple

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://fantasy.premierleague.com/api"
CACHE_DIR = os.path.expanduser("~/.fpl_cache")
CACHE_TTL = 900

GKP, DEF, MID, FWD = 1, 2, 3, 4
POS = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}

# ---------------------------------------------------------------------------
# scoring table (2026/27 — DefCon and chips confirmed unchanged from 2025/26)
# ---------------------------------------------------------------------------
GOAL_PTS = {GKP: 10, DEF: 6, MID: 5, FWD: 4}  # GK raised to 10 in 2026/27
CS_PTS = {GKP: 4, DEF: 4, MID: 1, FWD: 0}
ASSIST_PTS = 3
DEFCON_PTS = 2
DEFCON_THRESHOLD = {DEF: 10, MID: 12, FWD: 12}  # GKs don't earn DefCon
YELLOW_PTS = -1

# league-average goals per game, by venue
AVG_HOME_GOALS = 1.55
AVG_AWAY_GOALS = 1.25
SOT_PER_GOAL = 3.2  # shots on target faced per goal conceded, for save volume

# Bonus curve: BPS-per-start mapped to expected bonus points. A player around
# 18 BPS rarely places top-3; one around 32 usually does. Hand-fit — the
# backtest records actual bonus so this can be regressed properly later.
BONUS_FLOOR = 17.0
BONUS_SLOPE = 0.16
BONUS_CAP = 2.6

# ---------------------------------------------------------------------------
# priors: what we believe about a player before we've seen them play.
# Rates are per 90 and describe a *median-priced* player in that position.
# PRIOR_MINUTES is how much evidence it takes to move halfway to the observed
# rate, so 810 means one 90-minute game carries ~10% weight.
# ---------------------------------------------------------------------------
PRIOR_MINUTES = 810  # ~9 full matches

# For rare events (a defender's xG, an assist rate), holding the prior's
# *minutes* constant is wrong: a 0.12/90 prior carries only ~1 pseudo-event
# over 810 minutes, so one freak set-piece header swamps it, while a 0.90/90
# striker's prior carries ~8 and barely moves. Poisson-Gamma says hold the
# pseudo-EVENT count constant instead, which makes the required exposure scale
# inversely with the rate. Clamped so neither tail goes silly.
# Minutes that count as a "full" prior season for confidence weighting.
# Deliberately below 38x90: a regular starter who misses a few games should
# still get essentially full weight.
FULL_SEASON_MINUTES = 2400.0

PRIOR_EVENTS = 4.0
EVENT_SHRINK_BAND = (450.0, 1800.0)  # minutes: 5 to 20 matches
RARE_EVENT_KEYS = ("xg", "xa")
PRIORS = {
    #        xG90   xA90   defcon90  bps90   yellow90
    GKP: dict(xg=0.00, xa=0.01, defcon=0.4, bps=17.0, yellow=0.04),
    DEF: dict(xg=0.06, xa=0.07, defcon=7.2, bps=17.0, yellow=0.17),
    MID: dict(xg=0.15, xa=0.15, defcon=6.5, bps=15.0, yellow=0.16),
    FWD: dict(xg=0.38, xa=0.15, defcon=3.0, bps=16.0, yellow=0.12),
}

# How strongly price informs each prior, as an exponent on (price / median).
# Attacking output scales steeply with price; defensive work does not — the
# expensive defenders are attacking full-backs, who clear and tackle *less*
# than the cheap centre-backs. Hence the negative exponent on defcon.
PRICE_ELASTICITY = dict(xg=0.85, xa=0.70, bps=0.45, defcon=-0.35, yellow=0.0)
PRICE_MULT_CAP = (0.40, 2.6)  # keep the tails sane


def price_priors(pos: int, price: float, median_price: float) -> dict:
    """Scale the positional prior by how expensive the player is.

    FPL prices players on expected output, so price is a much stronger prior
    than position alone — it's the one piece of information that's fully
    available before a ball is kicked.
    """
    base = PRIORS[pos]
    ratio = max(0.5, price / max(median_price, 4.0))
    out = {}
    for key, val in base.items():
        e = PRICE_ELASTICITY.get(key, 0.0)
        mult = ratio**e if e else 1.0
        out[key] = val * min(PRICE_MULT_CAP[1], max(PRICE_MULT_CAP[0], mult))
    return out


def load_overrides(path: str | None) -> dict:
    """Manual context the API can't express.

    FPL's `status` and `chance_of_playing_next_round` are blunt and often lag
    the news. A player back from a long lay-off is flagged available long
    before he's playing 90 minutes; a rotation risk before a European tie
    isn't flagged at all. This is the seam for judgement the model can't have.

    Keys are player ids or names (accents optional). Recognised fields:
        exclude       drop entirely (suspended, sold, long-term out)
        minutes_mult  scale expected minutes, e.g. 0.5 for a managed return
        p_start       override start probability outright
        xg90/xa90/defcon90/bps90   override a rate directly
        avail         override availability, 0..1
        note          free text, echoed in output
    """
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("players", data)


def resolve_overrides(elements: list[dict], overrides: dict) -> dict[int, dict]:
    """Bind override keys to exactly one player each.

    A name key is a convenience, not an identifier: "Sangare" matches both
    M.Sangare and I.Sangare, who play for different clubs. Applying an
    override to the wrong player is silent and worse than not applying it, so
    an ambiguous key is REFUSED rather than guessed at. Ids are unambiguous
    and always win.
    """
    resolved: dict[int, dict] = {}
    by_id = {e["id"]: e for e in elements}

    for key, val in overrides.items():
        if key.isdigit():
            pid = int(key)
            if pid in by_id:
                resolved[pid] = val
            else:
                print(f"  ! override key {key!r} is not a valid player id — "
                      f"skipped")
            continue

        k = strip_accents(key)
        exact = [e for e in elements if strip_accents(e["web_name"]) == k]
        hits = exact or ([e for e in elements
                          if len(k) >= 4 and k in strip_accents(e["web_name"])]
                         if len(k) >= 4 else [])

        if len(hits) == 1:
            resolved.setdefault(hits[0]["id"], val)
        elif not hits:
            print(f"  ! override key {key!r} matched no player — skipped")
        else:
            names = ", ".join(f"{e['web_name']} (id {e['id']})" for e in hits[:6])
            print(f"  ! override key {key!r} is AMBIGUOUS — matches {names}. "
                  f"Skipped; use a player id instead.")
    return resolved


def strip_accents(s: str) -> str:
    s = s.replace("ø", "o").replace("Ø", "O").replace("ß", "ss")
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c)).lower()


def _get(path: str, fname: str, ttl: int = CACHE_TTL):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, fname)
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < ttl:
        with open(cache) as f:
            return json.load(f)
    r = requests.get(f"{API}/{path}", timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


def fnum(d: dict, key: str, default: float = 0.0) -> float:
    """FPL returns most numerics as strings, and nulls for unplayed players."""
    try:
        v = d.get(key)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. rates, with empirical-Bayes shrinkage
# ---------------------------------------------------------------------------
def shrink(events: float, minutes: float, prior_rate: float,
           k: float = PRIOR_MINUTES) -> float:
    """Posterior per-90 rate. With zero minutes this returns the prior; as
    minutes grow it converges on the observed rate."""
    prior_events = prior_rate * k / 90.0
    return (events + prior_events) / ((minutes + k) / 90.0)


def event_weight(prior_rate: float, target_events: float = PRIOR_EVENTS) -> float:
    """Minutes of prior needed to carry `target_events` pseudo-observations."""
    if prior_rate <= 0:
        return EVENT_SHRINK_BAND[1]
    k = target_events / prior_rate * 90.0
    return min(EVENT_SHRINK_BAND[1], max(EVENT_SHRINK_BAND[0], k))


def player_rates(e: dict, median_price: float = 5.5,
                 prior_minutes: float = PRIOR_MINUTES,
                 last: dict | None = None) -> dict:
    pos = e["element_type"]
    pri = price_priors(pos, e["now_cost"] / 10.0, median_price)
    if last:
        # A prior season only beats the price heuristic to the extent it
        # actually covers a season. A player who managed 900 minutes before
        # breaking his leg has a real but thin sample: blend toward price
        # rather than promoting a partial season to ground truth.
        # Default to full coverage when _minutes is absent: a missing key
        # should not silently discard a measured prior. Only an explicitly
        # thin season gets discounted.
        cover = min(1.0, last.get("_minutes", FULL_SEASON_MINUTES)
                    / FULL_SEASON_MINUTES)
        pri = {k: (cover * last[k] + (1 - cover) * pri[k]) if k in last else pri[k]
               for k in pri}
    mins = fnum(e, "minutes")

    # totals this season; the *_per_90 fields are unreliable at low minutes
    xg = fnum(e, "expected_goals")
    xa = fnum(e, "expected_assists")
    defcon = fnum(e, "defensive_contribution")
    bps = fnum(e, "bps")
    yellow = fnum(e, "yellow_cards")
    saves = fnum(e, "saves")

    k = prior_minutes
    # rare events get exposure-scaled shrinkage; high-rate ones don't need it
    k_xg = event_weight(pri["xg"])
    k_xa = event_weight(pri["xa"])
    return dict(
        xg=shrink(xg, mins, pri["xg"], k_xg),
        xa=shrink(xa, mins, pri["xa"], k_xa),
        defcon=shrink(defcon, mins, pri["defcon"], event_weight(pri["defcon"])),
        bps=shrink(bps, mins, pri["bps"], k),
        yellow=shrink(yellow, mins, pri["yellow"], k),
        saves=shrink(saves, mins, (last or {}).get("saves", 2.8), k)
        if pos == GKP else 0.0,
    )


def last_season_rates(element_id: int) -> dict | None:
    """Per-90 rates from the most recent completed season, if FPL has them.

    This is what the price heuristic is standing in for, so use it wherever
    it exists. Cached for a day since it never changes mid-season.
    """
    try:
        summary = _get(f"element-summary/{element_id}/",
                       f"elsum_{element_id}.json", ttl=86400)
    except requests.HTTPError:
        return None
    past = summary.get("history_past") or []
    if not past:
        return None
    p = past[-1]
    mins = fnum(p, "minutes")
    if mins < 450:  # too little to beat the price prior
        return None
    per90 = lambda key: fnum(p, key) / (mins / 90.0)
    out = dict(
        xg=per90("expected_goals"),
        xa=per90("expected_assists"),
        bps=per90("bps"),
        yellow=per90("yellow_cards"),
        saves=per90("saves"),
    )
    out["_minutes"] = mins
    starts = fnum(p, "starts")
    out["_starts"] = starts if starts > 0 else mins / 85.0
    dc = fnum(p, "defensive_contribution")
    if dc > 0:
        out["defcon"] = dc / (mins / 90.0)
    # pre-2022 seasons have no xG; fall back to actual returns
    if out["xg"] == 0 and fnum(p, "goals_scored") > 0:
        out["xg"] = per90("goals_scored")
    if out["xa"] == 0 and fnum(p, "assists") > 0:
        out["xa"] = per90("assists")
    return {k: v for k, v in out.items() if v > 0 or k.startswith("_")}


# ---------------------------------------------------------------------------
# 2. minutes model
# ---------------------------------------------------------------------------
class Minutes(NamedTuple):
    """Minutes are a mixture, not a scalar. Most players either start and play
    ~80 minutes or don't play at all; very few reliably play the average of
    those two. Threshold events (DefCon) are convex in minutes, so collapsing
    the mixture to its mean understates them severely — hence the components."""
    p_appear: float
    p_60: float
    exp_minutes: float
    p_start: float
    mins_if_start: float
    p_cameo: float
    cameo_mins: float


def minutes_model(e: dict, games_played: int, last: dict | None = None,
                  gk_rank: int | None = None) -> Minutes:
    """Return (p_appear, p_60, expected_minutes).

    Priority of evidence for the start prior:
      1. last season's starts — by far the best predictor of this season's
      2. depth-chart rank, for goalkeepers (a binary, near-deterministic role)
      3. price, which tracks talent well but squad status only loosely

    Price alone badly underrates the cheap-but-nailed tier: keepers and budget
    defenders are the most certain starters in the game and among the cheapest.
    """
    status = e.get("status", "a")
    if status in ("i", "s", "u", "n"):
        return Minutes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    avail = 1.0
    if status == "d":
        chance = e.get("chance_of_playing_next_round")
        avail = (chance / 100.0) if chance is not None else 0.75

    starts = fnum(e, "starts")
    mins = fnum(e, "minutes")
    g = max(games_played, 1)

    price = e["now_cost"] / 10.0
    if last and last.get("_starts"):
        # last season's start rate, nudged toward the mean for regression
        prior_start = min(0.95, max(0.05, last["_starts"] / 38.0)) * 0.88 + 0.06
    elif e["element_type"] == GKP and gk_rank is not None:
        # keepers are a depth chart, not a price curve
        prior_start = {0: 0.88, 1: 0.10}.get(gk_rank, 0.02)
    else:
        # price fallback: £4.0m fringe -> ~0.25, £7.0m -> ~0.75, £11m+ -> ~0.92
        prior_start = min(0.93, max(0.15, 0.15 + 0.09 * (price - 4.0)))

    # shrink observed start rate toward that prior; k in *matches*
    k_games = 4.0
    p_start = (starts + prior_start * k_games) / (g + k_games)
    p_start = min(1.0, p_start) * avail

    prior_mins_if_start = 78.0
    if last and last.get("_starts", 0) > 0:
        prior_mins_if_start = max(45.0, min(90.0,
                                            last["_minutes"] / last["_starts"]))
    # Minutes played as a substitute say nothing about how long he lasts when
    # he STARTS, so they must not enter this estimate. Bootstrap has no
    # appearance count, so bound start-minutes above by starts * 90: with zero
    # starts this correctly contributes nothing, and elsewhere it caps the
    # error rather than letting cameos inflate the numerator without limit.
    start_minutes = min(mins, starts * 90.0)
    mins_if_start = ((start_minutes + prior_mins_if_start * k_games)
                     / (starts + k_games))
    mins_if_start = max(20.0, min(90.0, mins_if_start))

    # cameo appearances off the bench
    p_bench_cameo = min(0.45, max(0.0, (0.55 - p_start))) * avail
    p_appear = min(1.0, p_start + p_bench_cameo)

    # P(60+) is essentially P(start) discounted for early subs
    p_60 = p_start * min(1.0, max(0.35, (mins_if_start - 12) / 70.0))
    cameo_mins = 18.0
    exp_minutes = p_start * mins_if_start + p_bench_cameo * cameo_mins
    return Minutes(p_appear, p_60, exp_minutes, p_start, mins_if_start,
                   p_bench_cameo, cameo_mins)


# ---------------------------------------------------------------------------
# 3. team Poisson model
# ---------------------------------------------------------------------------
def team_strength(teams: dict) -> dict:
    """Normalise FPL's strength ratings so 1.0 is league average.

    FPL doesn't always populate the granular attack/defence ratings — early in
    a season they can be zero. Degrade through progressively coarser sources
    rather than dividing by zero.
    """
    fine = ["strength_attack_home", "strength_attack_away",
            "strength_defence_home", "strength_defence_away"]

    def mean(key):
        vals = [float(t.get(key) or 0) for t in teams.values()]
        return sum(vals) / len(vals) if vals else 0.0

    # 1. the granular ratings, if they're actually populated
    if all(mean(k) > 0 for k in fine):
        means = {k: mean(k) for k in fine}
        out = {tid: {k: float(t[k]) / means[k] for k in fine}
               for tid, t in teams.items()}
        return _tag(out, "strength_attack/defence")

    # 2. overall home/away ratings, used for both attack and defence
    oh, oa = "strength_overall_home", "strength_overall_away"
    if mean(oh) > 0 and mean(oa) > 0:
        mh, ma = mean(oh), mean(oa)
        out = {}
        for tid, t in teams.items():
            h, a = float(t[oh]) / mh, float(t[oa]) / ma
            out[tid] = {"strength_attack_home": h, "strength_attack_away": a,
                        "strength_defence_home": h, "strength_defence_away": a}
        return _tag(out, "strength_overall (attack=defence)")

    # 3. the coarse 1-5 tier
    if mean("strength") > 0:
        ms = mean("strength")
        out = {}
        for tid, t in teams.items():
            s = float(t["strength"]) / ms
            out[tid] = dict.fromkeys(fine, s)
        return _tag(out, "coarse 1-5 strength tier")

    # 4. nothing usable — treat every side as league average
    out = {tid: dict.fromkeys(fine, 1.0) for tid in teams}
    return _tag(out, "FLAT — no strength data, fixtures carry no signal")


def _tag(out: dict, source: str) -> dict:
    print(f"Team strength source: {source}")
    return out


def fixture_lambdas(strength: dict, home: int, away: int,
                    fitted: dict | None = None) -> tuple[float, float]:
    """Expected goals for (home, away).

    With fitted ratings this is the Poisson model directly. Without them it
    falls back to FPL's tier ratings, where a stronger defence divides.
    """
    if fitted:
        mu, gamma = fitted["mu"], fitted["gamma"]
        atk, dfn = fitted["attack"], fitted["defence"]
        lh = mu * gamma * atk[str(home)] * dfn[str(away)]
        la = mu * atk[str(away)] * dfn[str(home)]
        return max(0.15, lh), max(0.15, la)
    lh = (AVG_HOME_GOALS * strength[home]["strength_attack_home"]
          / strength[away]["strength_defence_away"])
    la = (AVG_AWAY_GOALS * strength[away]["strength_attack_away"]
          / strength[home]["strength_defence_home"])
    return max(0.15, lh), max(0.15, la)


def team_baseline(fixtures: list, strength: dict, fitted: dict | None,
                  team_id: int) -> float:
    """A team's expected goals against an average opponent, over the WHOLE
    season schedule.

    This is the correct denominator for the fixture multiplier. A player's
    per-90 rate was earned across a full season's mix of opponents, so the
    adjustment for one fixture is `this fixture / that average`.

    Averaging over the projection horizon instead would be self-defeating: a
    team with five easy games would have a high average, every fixture would
    divide down to ~1.0, and the easy run would produce no lift whatsoever.
    """
    lams = []
    for f in fixtures:
        if f.get("event") is None:
            continue
        if f["team_h"] == team_id:
            lams.append(fixture_lambdas(strength, f["team_h"], f["team_a"], fitted)[0])
        elif f["team_a"] == team_id:
            lams.append(fixture_lambdas(strength, f["team_h"], f["team_a"], fitted)[1])
    return sum(lams) / len(lams) if lams else 1.4


def pois_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def p_at_least(threshold: int, lam: float) -> float:
    """P(X >= threshold) for X ~ Poisson(lam). Used for DefCon."""
    if lam <= 0:
        return 0.0
    below = sum(pois_pmf(k, lam) for k in range(threshold))
    return max(0.0, 1.0 - below)


def expected_conceded_penalty(lam: float, cap: int = 12) -> float:
    """E[floor(goals_conceded / 2)] — the -1 per two conceded rule."""
    return sum((k // 2) * pois_pmf(k, lam) for k in range(cap))


def expected_save_points(lam_against: float, saves90: float,
                         minutes_frac: float, cap: int = 20) -> float:
    """1 point per 3 saves. Save volume scales with how much the opponent
    attacks, so a keeper behind a bad defence is worth more, not less."""
    exp_saves = saves90 * minutes_frac * max(0.4, lam_against / 1.35)
    if exp_saves <= 0:
        return 0.0
    return sum((k // 3) * pois_pmf(k, exp_saves) for k in range(cap))


def expected_bonus(bps90: float, mins: "Minutes") -> float:
    """Bonus needs a top-3 BPS finish IN A MATCH, so it must be conditioned on
    starting — averaging in the weeks a player doesn't feature pushes his
    per-match BPS below the threshold and zeroes the component entirely.

    Still the weakest part of the model: bonus depends on teammates' BPS as
    much as the player's own, and the 2026/27 BPS changes (no penalty for being
    tackled, CBI now one point per three actions) make historical rates biased
    for full-backs and wingers. The curve below is hand-fit and should be
    replaced by a regression on actual bonus once the backtest has enough
    gameweeks — `bps` and `bonus` are recorded in the scored output for exactly
    that purpose.
    """
    if mins.p_start <= 0:
        return 0.0
    per_start = bps90 * (mins.mins_if_start / 90.0)
    if per_start < BONUS_FLOOR:
        return 0.0
    haul = min(BONUS_CAP, BONUS_SLOPE * (per_start - BONUS_FLOOR))
    return mins.p_start * haul


# ---------------------------------------------------------------------------
# 4. assemble
# ---------------------------------------------------------------------------
def defcon_points(rate90: float, threshold: int, mins: "Minutes") -> float:
    """E[DefCon points], integrating the threshold over the minutes mixture.

    P(count >= threshold) is convex in lambda across the relevant range, so
    evaluating it at E[minutes] and scaling by p(appear) is not an
    approximation — it's an underestimate, and a large one for players who
    either start or sit. Condition on each scenario instead.
    """
    if rate90 <= 0:
        return 0.0
    p_start_hit = p_at_least(threshold, rate90 * mins.mins_if_start / 90.0)
    p_cameo_hit = p_at_least(threshold, rate90 * mins.cameo_mins / 90.0)
    return (mins.p_start * p_start_hit
            + mins.p_cameo * p_cameo_hit) * DEFCON_PTS


def score_fixture(e: dict, rates: dict, mins: Minutes, lam_for: float,
                  lam_against: float, team_avg_for: float) -> float:
    pos = e["element_type"]
    p_appear, p_60, exp_minutes = mins.p_appear, mins.p_60, mins.exp_minutes
    if exp_minutes <= 0:
        return 0.0
    frac = exp_minutes / 90.0

    # appearance: 1 for playing, 2 for 60+
    pts = p_appear * 1.0 + p_60 * 1.0

    # attacking returns, scaled by how good this fixture is for the team
    attack_mult = lam_for / max(team_avg_for, 0.3)
    pts += rates["xg"] * frac * attack_mult * GOAL_PTS[pos]
    pts += rates["xa"] * frac * attack_mult * ASSIST_PTS

    # clean sheet requires 60 minutes
    if CS_PTS[pos]:
        pts += p_60 * math.exp(-lam_against) * CS_PTS[pos]

    # goals conceded hurts keepers and defenders
    if pos in (GKP, DEF):
        pts -= p_60 * expected_conceded_penalty(lam_against)

    if pos == GKP:
        pts += expected_save_points(lam_against, rates["saves"], frac)

    # DefCon is a threshold event — model the count over the minutes mixture
    if pos in DEFCON_THRESHOLD:
        pts += defcon_points(rates["defcon"], DEFCON_THRESHOLD[pos], mins)

    pts += expected_bonus(rates["bps"], mins)
    pts += rates["yellow"] * frac * YELLOW_PTS
    return max(0.0, pts)


def build(horizon: int, start_gw: int | None = None, deep: int = 0,
          prior_minutes: float = PRIOR_MINUTES,
          ratings_file: str | None = None,
          overrides_file: str | None = None) -> tuple[list[dict], int]:
    boot = _get("bootstrap-static/", "bootstrap.json")
    fixtures = _get("fixtures/", "fixtures.json")
    teams = {t["id"]: t for t in boot["teams"]}

    fitted = None
    if ratings_file:
        with open(ratings_file, encoding="utf-8") as f:
            fitted = json.load(f)
        n = fitted.get("matches", 0)
        print(f"Team strength source: fitted ratings from {n} matches "
              f"(mu={fitted['mu']:.3f}, gamma={fitted['gamma']:.3f})")
        strength = {}
    else:
        strength = team_strength(teams)

    if start_gw is None:
        start_gw = next((ev["id"] for ev in boot["events"] if ev.get("is_next")), None)
        if start_gw is None:
            start_gw = next((ev["id"] for ev in boot["events"]
                             if not ev.get("finished")), 1)
    gws = list(range(start_gw, start_gw + horizon))

    games_played = sum(1 for ev in boot["events"] if ev.get("finished"))

    # team -> gw -> list of (lam_for, lam_against); a list handles doubles,
    # an empty list handles blanks
    sched: dict[int, dict[int, list]] = {t: {g: [] for g in gws} for t in teams}
    for f in fixtures:
        g = f.get("event")
        if g not in gws:
            continue
        h, a = f["team_h"], f["team_a"]
        lh, la = fixture_lambdas(strength, h, a, fitted)
        sched[h][g].append((lh, la))
        sched[a][g].append((la, lh))

    # each team's baseline attacking output, measured over the full season
    # schedule rather than the horizon (see team_baseline for why)
    team_avg_for = {t: team_baseline(fixtures, strength, fitted, t) for t in teams}

    # median price per position, so the price prior is self-calibrating
    med_price = {}
    for pos in POS:
        prices = sorted(e["now_cost"] / 10.0 for e in boot["elements"]
                        if e["element_type"] == pos)
        med_price[pos] = prices[len(prices) // 2] if prices else 5.5

    last_rates: dict[int, dict] = {}
    if deep:
        pool = sorted(boot["elements"], key=lambda e: -e["now_cost"])[:deep]
        print(f"Fetching last-season rates for {len(pool)} players "
              f"(cached; first run is slow)...")
        for i, e in enumerate(pool, 1):
            r = last_season_rates(e["id"])
            if r:
                last_rates[e["id"]] = r
            if i % 25 == 0:
                print(f"  {i}/{len(pool)}")
        print(f"  got real priors for {len(last_rates)} players.")

    # goalkeeper depth chart: within each club, rank keepers by price then by
    # ownership, which together identify the first choice almost perfectly
    gk_rank: dict[int, int] = {}
    for tid in teams:
        keepers = sorted(
            (e for e in boot["elements"]
             if e["element_type"] == GKP and e["team"] == tid),
            key=lambda e: (-e["now_cost"], -fnum(e, "selected_by_percent")))
        for i, e in enumerate(keepers):
            gk_rank[e["id"]] = i

    raw_overrides = load_overrides(overrides_file)
    if raw_overrides:
        print(f"\nResolving {len(raw_overrides)} override key(s)...")
    overrides = resolve_overrides(boot["elements"], raw_overrides)
    applied = []

    rows = []
    for e in boot["elements"]:
        ov = overrides.get(e["id"])
        if ov and ov.get("exclude"):
            applied.append(f"{e['web_name']}: EXCLUDED"
                           + (f" ({ov['note']})" if ov.get("note") else ""))
            continue

        rates = player_rates(e, med_price[e["element_type"]], prior_minutes,
                             last_rates.get(e["id"]))
        mins = minutes_model(e, games_played, last_rates.get(e["id"]),
                             gk_rank.get(e["id"]))

        ramp = list(ov.get("minutes_ramp", [])) if ov else []

        if ov:
            for key, field in (("xg90", "xg"), ("xa90", "xa"),
                               ("defcon90", "defcon"), ("bps90", "bps")):
                if key in ov:
                    rates[field] = float(ov[key])
            if "p_start" in ov or "minutes_mult" in ov or "avail" in ov:
                # Each knob acts on exactly one thing, so they compose without
                # compounding:
                #   p_start      -> replaces the start probability
                #   avail        -> scales the chance of featuring at all
                #   minutes_mult -> scales minutes PER START (a managed return
                #                   plays 55 minutes, not 55% of matches)
                ps = float(ov.get("p_start", mins.p_start))
                ps = max(0.0, min(1.0, ps * float(ov.get("avail", 1.0))))
                mis = mins.mins_if_start * float(ov.get("minutes_mult", 1.0))
                mis = max(10.0, min(90.0, mis))
                p_cameo = mins.p_cameo * float(ov.get("avail", 1.0))
                mins = Minutes(
                    p_appear=min(1.0, ps + p_cameo),
                    p_60=ps * min(1.0, max(0.35, (mis - 12) / 70.0)),
                    exp_minutes=ps * mis + p_cameo * mins.cameo_mins,
                    p_start=ps, mins_if_start=mis,
                    p_cameo=p_cameo, cameo_mins=mins.cameo_mins)
            desc = f"{e['web_name']}: E[min] {mins.exp_minutes:.0f}"
            if ramp:
                path = ", ".join(
                    f"{mins.p_start * float(ramp[min(i, len(ramp)-1)]):.2f}"
                    for i in range(len(gws)))
                desc += f"  p_start by GW: [{path}]"
            applied.append(desc + (f" — {ov['note']}" if ov.get("note") else ""))
        for gi, g in enumerate(gws):
            # A returning player isn't a fixed probability — he's benched now
            # and starting in three weeks. minutes_ramp expresses that as a
            # multiplier per horizon week; the last value carries forward.
            m_g = mins
            if ramp:
                mult = float(ramp[min(gi, len(ramp) - 1)])
                ps = max(0.0, min(1.0, mins.p_start * mult))
                mis = mins.mins_if_start
                pc = mins.p_cameo
                m_g = Minutes(
                    p_appear=min(1.0, ps + pc),
                    p_60=ps * min(1.0, max(0.35, (mis - 12) / 70.0)),
                    exp_minutes=ps * mis + pc * mins.cameo_mins,
                    p_start=ps, mins_if_start=mis,
                    p_cameo=pc, cameo_mins=mins.cameo_mins)
            ep = sum(
                score_fixture(e, rates, m_g, lf, la, team_avg_for[e["team"]])
                for lf, la in sched[e["team"]][g]
            )
            rows.append({"id": e["id"], "gw": g, "ep": round(ep, 3),
                         "name": e["web_name"], "pos": POS[e["element_type"]],
                         "team": teams[e["team"]]["short_name"],
                         "cost": e["now_cost"] / 10.0})
    if applied:
        print(f"\nApplied {len(applied)} manual override(s):")
        for a in applied:
            print(f"  {a}")
    return rows, start_gw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--gw", type=int, help="start gameweek (default: next)")
    ap.add_argument("--out", default="xp.csv")
    ap.add_argument("--top", type=int, default=25,
                    help="how many players to print as a sanity check")
    ap.add_argument("--deep", type=int, default=0, metavar="N",
                    help="fetch last-season rates for the N priciest players "
                         "and use them as priors instead of the price heuristic")
    ap.add_argument("--prior-minutes", type=float, default=PRIOR_MINUTES,
                    help="evidence weight; lower trusts this season sooner")
    ap.add_argument("--ratings", help="ratings.json from ratings.py — replaces "
                                      "FPL's coarse tiers with fitted values")
    ap.add_argument("--overrides", help="overrides.json of manual context the "
                                        "API can't express (injuries, returns)")
    args = ap.parse_args()

    rows, start_gw = build(args.horizon, args.gw, args.deep, args.prior_minutes,
                           args.ratings, args.overrides)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gw", "ep"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out} "
          f"(GW{start_gw}-{start_gw + args.horizon - 1}).")

    # sanity check: totals over the horizon
    tot: dict[int, dict] = {}
    for r in rows:
        t = tot.setdefault(r["id"], {**r, "total": 0.0})
        t["total"] += r["ep"]
    ranked = sorted(tot.values(), key=lambda r: -r["total"])

    print(f"\n{'PLAYER':<20}{'TEAM':>6}{'POS':>5}{'£':>7}{'TOTAL':>8}{'PER GW':>8}")
    print("-" * 54)
    for r in ranked[:args.top]:
        print(f"{r['name']:<20}{r['team']:>6}{r['pos']:>5}{r['cost']:>7.1f}"
              f"{r['total']:>8.2f}{r['total']/args.horizon:>8.2f}")

    gw1 = sorted((r["ep"] for r in rows if r["gw"] == start_gw), reverse=True)
    print(f"\nGW{start_gw} spread: top {gw1[0]:.2f}, "
          f"median {gw1[len(gw1)//2]:.2f}, delta {gw1[0]-gw1[len(gw1)//2]:.2f}")


if __name__ == "__main__":
    main()
