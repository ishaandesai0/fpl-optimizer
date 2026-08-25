#!/usr/bin/env python3
"""
ratings.py — continuous attack/defence ratings for Premier League teams.

FPL's `strength_overall` fields are 1-5 integer tiers: twenty clubs squeezed
into about five distinct values, with attack and defence forced to be equal.
That's the binding constraint on the projection model — it clusters whole
squads together and distorts captaincy.

This fits a Poisson attack/defence model to actual results instead:

    goals_home ~ Poisson(mu * gamma * attack[home] * defence[away])
    goals_away ~ Poisson(mu * attack[away] * defence[home])

attack and defence are each normalised to mean 1.0, so attack 1.30 means "30%
more goals than an average side against an average opponent" and defence 0.80
means "concedes 20% fewer". gamma is home advantage.

Early in a season there aren't enough matches to fit twenty teams' worth of
parameters, so every rating is shrunk toward a prior built from FPL's tiers.
The prior is expressed as a number of pseudo-matches (--prior-weight), which
decays naturally in influence as real results accumulate.

Fitting is by coordinate ascent — the standard iterative scaling for this
model. No scipy needed.

Usage
-----
    python ratings.py --out ratings.json
    python projections.py --horizon 5 --deep 600 --ratings ratings.json --out xp.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://fantasy.premierleague.com/api"

# Long-run Premier League home advantage, and how many matches of evidence it
# takes to move meaningfully off it. Half a season is the right order: gamma is
# a single parameter estimated from every match at once, but it's a ratio of
# two noisy totals, so early-season swings are almost entirely sampling noise.
GAMMA_PRIOR = 1.25
GAMMA_PRIOR_MATCHES = 200.0
GAMMA_BAND = (1.02, 1.45)
CACHE_DIR = os.path.expanduser("~/.fpl_cache")


def _get(path: str, fname: str, ttl: int = 900):
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


# "United" and "City" are NOT noise — stripping them collapses Man Utd and
# Man City onto the same key, which silently swaps two clubs' ratings. Only
# genuinely empty tokens get removed; everything else is handled by aliases.
ALIASES = {
    "manutd": "manchesterunited", "manunited": "manchesterunited",
    "manu": "manchesterunited", "mufc": "manchesterunited",
    "mancity": "manchestercity", "mcfc": "manchestercity",
    "spurs": "tottenham", "tottenhamhotspur": "tottenham",
    "nottmforest": "nottinghamforest", "forest": "nottinghamforest",
    "wolverhamptonwanderers": "wolves", "wolverhampton": "wolves",
    "newcastleunited": "newcastle", "leedsunited": "leeds",
    "brightonandhovealbion": "brighton", "brightonhovealbion": "brighton",
    "westham": "westhamunited", "westhamutd": "westhamunited",
    "sheffieldutd": "sheffieldunited", "nottingham": "nottinghamforest",
    "cpalace": "crystalpalace", "palace": "crystalpalace",
    "bournemouth": "bournemouth", "afcbournemouth": "bournemouth",
    "coventrycity": "coventry", "hullcity": "hull", "ipswichtown": "ipswich",
    "lutontown": "luton", "norwichcity": "norwich", "stokecity": "stoke",
}


def norm(s: str) -> str:
    s = s.lower().replace("&", "and")
    s = re.sub(r"\b(fc|afc)\b", " ", s)
    s = re.sub(r"[^a-z]", "", s)
    return ALIASES.get(s, s)


def map_teams(names: set[str], teams: dict) -> dict[str, int]:
    """Match external team names (football-data.co.uk etc.) to FPL ids.

    Sources spell clubs inconsistently — "Man United", "Man Utd", "Nott'm
    Forest", "Spurs". Strip the noise words and match on what's left, then
    report anything that didn't land so it can be fixed by hand.
    """
    by_norm = {}
    for tid, t in teams.items():
        for key in (t.get("name", ""), t.get("short_name", "")):
            if key:
                by_norm.setdefault(norm(key), tid)
    out = {}
    for nme in names:
        n = norm(nme)
        if n in by_norm:
            out[nme] = by_norm[n]
            continue
        # prefix fallback, but only when it identifies exactly one club —
        # an ambiguous prefix is left unmapped rather than guessed at
        cand = {tid for k, tid in by_norm.items()
                if k and len(n) >= 4 and (k.startswith(n) or n.startswith(k))}
        if len(cand) == 1:
            out[nme] = cand.pop()
    return out


def load_results_csv(path: str, teams: dict, weight: float) -> list[dict]:
    """Read a football-data.co.uk style CSV (HomeTeam/AwayTeam/FTHG/FTAG)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return []
    cols = {c.lower(): c for c in rows[0]}
    h = cols.get("hometeam") or cols.get("home")
    a = cols.get("awayteam") or cols.get("away")
    hg = cols.get("fthg") or cols.get("homegoals") or cols.get("hg")
    ag = cols.get("ftag") or cols.get("awaygoals") or cols.get("ag")
    if not all((h, a, hg, ag)):
        sys.exit(f"{path}: need HomeTeam/AwayTeam/FTHG/FTAG columns, "
                 f"got {list(rows[0])}")

    names = {r[h] for r in rows} | {r[a] for r in rows}
    mapping = map_teams(names, teams)
    missing = sorted(names - set(mapping))
    if missing:
        print(f"  unmapped teams (skipped): {', '.join(missing)}")
        print("  ^ relegated/promoted sides are expected here; anything else "
              "needs a manual fix.")

    out = []
    for r in rows:
        if r[h] not in mapping or r[a] not in mapping:
            continue
        try:
            out.append({"h": mapping[r[h]], "a": mapping[r[a]],
                        "hg": int(float(r[hg])), "ag": int(float(r[ag])),
                        "w": weight})
        except (ValueError, TypeError):
            continue
    print(f"  loaded {len(out)} historical matches at weight {weight}")
    return out


def tier_prior(teams: dict) -> dict[int, tuple[float, float]]:
    """Seed attack/defence from FPL's 1-5 tiers.

    Coarse, but it's the only pre-season information available. A tier-5 side
    is assumed to both score more and concede less than a tier-2 side; the
    exponents control how much the tiers are allowed to say.
    """
    vals = []
    for t in teams.values():
        h = float(t.get("strength_overall_home") or 0)
        a = float(t.get("strength_overall_away") or 0)
        vals.append((h + a) / 2 if (h or a) else 3.0)
    mean = sum(vals) / len(vals)

    prior = {}
    for (tid, t), v in zip(teams.items(), vals):
        rel = v / mean if mean else 1.0
        # a top-tier side scores ~35% more and concedes ~25% fewer
        prior[tid] = (rel**0.9, rel**-0.75)
    return prior


def fit(results: list[dict], team_ids: list[int], prior: dict,
        prior_weight: float = 6.0, iters: int = 300, tol: float = 1e-9):
    """Coordinate-ascent MLE with the prior entered as pseudo-matches.

    Each team gets `prior_weight` imaginary matches against an average
    opponent in which it performed exactly as the prior expects. With zero
    real results the fit returns the prior; with a full season it's dominated
    by actual goals.
    """
    atk = {t: prior[t][0] for t in team_ids}
    dfn = {t: prior[t][1] for t in team_ids}

    # home advantage straight from the aggregate split
    for r in results:
        r.setdefault("w", 1.0)
    hg = sum(r["w"] * r["hg"] for r in results)
    ag = sum(r["w"] * r["ag"] for r in results)
    n = max(sum(r["w"] for r in results), 1.0)
    mu = ((hg + ag) / (2 * n)) if results else 1.40
    # Home advantage is ONE league-wide parameter with a strong empirical
    # prior (~1.25 in the Premier League). A handful of matches carries almost
    # no information about it — nine games can easily show a 4:1 home split by
    # chance — so shrink hard, in log space, and clamp to a plausible band.
    # An inflated gamma silently overrates every home fixture downstream.
    gamma_obs = (hg / ag) if (results and ag > 0) else GAMMA_PRIOR
    gamma_obs = min(2.2, max(0.7, gamma_obs))
    w_gamma = n / (n + GAMMA_PRIOR_MATCHES)
    log_g = (w_gamma * math.log(gamma_obs)
             + (1 - w_gamma) * math.log(GAMMA_PRIOR))
    gamma = min(GAMMA_BAND[1], max(GAMMA_BAND[0], math.exp(log_g)))

    for _ in range(iters):
        delta = 0.0

        for t in team_ids:
            scored = sum(r["w"] * r["hg"] for r in results if r["h"] == t) + \
                     sum(r["w"] * r["ag"] for r in results if r["a"] == t)
            denom = sum(r["w"] * mu * gamma * dfn[r["a"]]
                        for r in results if r["h"] == t) + \
                    sum(r["w"] * mu * dfn[r["h"]]
                        for r in results if r["a"] == t)
            new = ((scored + prior_weight * prior[t][0] * mu)
                   / (denom + prior_weight * mu))
            delta = max(delta, abs(new - atk[t]))
            atk[t] = new

        for t in team_ids:
            conceded = sum(r["w"] * r["ag"] for r in results if r["h"] == t) + \
                       sum(r["w"] * r["hg"] for r in results if r["a"] == t)
            denom = sum(r["w"] * mu * atk[r["a"]]
                        for r in results if r["h"] == t) + \
                    sum(r["w"] * mu * gamma * atk[r["h"]]
                        for r in results if r["a"] == t)
            new = ((conceded + prior_weight * prior[t][1] * mu)
                   / (denom + prior_weight * mu))
            delta = max(delta, abs(new - dfn[t]))
            dfn[t] = new

        # identifiability: attack and defence are only defined up to scale
        ma = sum(atk.values()) / len(atk)
        md = sum(dfn.values()) / len(dfn)
        for t in team_ids:
            atk[t] /= ma
            dfn[t] /= md
        mu *= ma * md

        if delta < tol:
            break

    return atk, dfn, mu, gamma


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ratings.json")
    ap.add_argument("--prior-weight", type=float, default=6.0,
                    help="pseudo-matches of tier prior per team; lower trusts "
                         "actual results sooner")
    ap.add_argument("--results", help="optional JSON of extra historical results: "
                                      "[{h,a,hg,ag}, ...] with FPL team ids")
    ap.add_argument("--fresh", action="store_true",
                    help="bypass the cache and refetch from the API")
    ap.add_argument("--results-csv", help="football-data.co.uk style CSV of "
                                          "past results (HomeTeam,AwayTeam,"
                                          "FTHG,FTAG)")
    ap.add_argument("--history-weight", type=float, default=0.5,
                    help="weight on historical matches vs this season's")
    args = ap.parse_args()

    ttl = 0 if args.fresh else 900
    boot = _get("bootstrap-static/", "bootstrap.json", ttl)
    fixtures = _get("fixtures/", "fixtures.json", ttl)
    teams = {t["id"]: t for t in boot["teams"]}
    names = {t["id"]: t["short_name"] for t in teams.values()}

    # a fixture counts once it has a scoreline; the `finished` flag lags and
    # `finished_provisional` flips earlier, so trust the scores themselves
    results = [
        {"h": f["team_h"], "a": f["team_a"],
         "hg": f["team_h_score"], "ag": f["team_a_score"]}
        for f in fixtures
        if f.get("team_h_score") is not None and f.get("team_a_score") is not None
    ]
    with_flag = sum(1 for f in fixtures if f.get("finished"))
    if len(results) != with_flag:
        print(f"note: {len(results)} fixtures have scores, {with_flag} carry the "
              f"`finished` flag — using scores.")
    if not results:
        played = sum(1 for f in fixtures if f.get("started"))
        print(f"WARNING: no scorelines found ({played} fixtures marked started). "
              f"Ratings will be 100% prior. If matches have been played, your "
              f"cached fixtures.json may be stale — delete {CACHE_DIR}.")
    if args.results:
        with open(args.results, encoding="utf-8") as f:
            extra = json.load(f)
        for r in extra:
            r.setdefault("w", args.history_weight)
        results = extra + results
        print(f"Added {len(extra)} historical results.")
    if args.results_csv:
        results = load_results_csv(args.results_csv, teams,
                                   args.history_weight) + results

    prior = tier_prior(teams)
    atk, dfn, mu, gamma = fit(results, list(teams), prior, args.prior_weight)

    weight = sum(r.get("w", 1.0) for r in results) * 2 / max(len(teams), 1)
    conf = weight / (weight + args.prior_weight)
    print(f"Fitted on {len(results)} matches "
          f"({weight:.1f} per team). Prior weight {args.prior_weight:.1f} "
          f"-> ratings are {conf:.0%} data, {1-conf:.0%} prior.")
    print(f"Base rate mu={mu:.3f}, home advantage gamma={gamma:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mu": mu, "gamma": gamma,
                   "attack": {str(t): atk[t] for t in atk},
                   "defence": {str(t): dfn[t] for t in dfn},
                   "matches": len(results)}, f, indent=1)
    print(f"Wrote {args.out}")

    print(f"\n{'TEAM':<6}{'ATTACK':>9}{'DEFENCE':>9}{'NET':>8}")
    print("-" * 32)
    for t in sorted(teams, key=lambda t: -(atk[t] / dfn[t])):
        print(f"{names[t]:<6}{atk[t]:>9.3f}{dfn[t]:>9.3f}{atk[t]/dfn[t]:>8.2f}")


if __name__ == "__main__":
    main()
