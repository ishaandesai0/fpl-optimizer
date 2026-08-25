#!/usr/bin/env python3
"""
diagnose.py — why does the model rate this player the way it does?

Dumps every input feeding a player's projection and decomposes the resulting
xP into its scoring components, so you can see whether a surprising number
comes from a bad prior, a bad minutes estimate, or a genuine model view.

Usage
-----
    python diagnose.py Sangare
    python diagnose.py Haaland --ratings ratings.json
    python diagnose.py 411 --ratings ratings.json --deep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
import unicodedata

import projections as P


# Letters NFD won't decompose — these are distinct codepoints, not a base
# letter plus a combining mark, so they need mapping by hand. Without this,
# "odegaard" fails to find Ødegaard.
_SPECIAL = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "å": "a", "Å": "A",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ł": "l", "Ł": "L",
    "đ": "d", "Đ": "D", "ß": "ss", "œ": "oe", "Œ": "OE", "ı": "i",
})


def fold(s: str) -> str:
    """Strip accents so 'sangare' matches 'M.Sangaré' and 'odegaard' matches
    'Ødegaard'. FPL web_names are full of diacritics; a plain .lower()
    comparison misses a large slice of the league."""
    s = s.translate(_SPECIAL)
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def find(elements: list[dict], needle: str) -> dict | None:
    if needle.isdigit():
        return next((e for e in elements if e["id"] == int(needle)), None)
    n = fold(needle)
    hits = [e for e in elements
            if n in fold(e["web_name"])
            or n in fold(f"{e.get('first_name','')} {e.get('second_name','')}")]
    if len(hits) > 1:
        print("Multiple matches — pick one by id:")
        for e in hits[:12]:
            print(f"  {e['id']:>4}  {e['web_name']:<20} £{e['now_cost']/10:.1f}m")
        sys.exit(0)
    if not hits:
        # offer near misses rather than just failing
        close = [e for e in elements if fold(e["web_name"])[:3] == n[:3]]
        if close:
            print(f"No exact match for {needle!r}. Did you mean:")
            for e in close[:10]:
                print(f"  {e['id']:>4}  {e['web_name']:<20} £{e['now_cost']/10:.1f}m")
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("player", help="name fragment or FPL element id")
    ap.add_argument("--ratings", help="ratings.json from ratings.py")
    ap.add_argument("--gw", type=int, help="gameweek to decompose (default: next)")
    ap.add_argument("--overrides", help="overrides.json — MUST match what you "
                                        "pass to projections.py, or this tool "
                                        "explains a model you aren't using")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    teams = {t["id"]: t for t in boot["teams"]}

    e = find(boot["elements"], args.player)
    if not e:
        sys.exit(f"No player matching {args.player!r}")

    ov = None
    if args.overrides:
        raw = P.load_overrides(args.overrides)
        ov = P.resolve_overrides(boot["elements"], raw).get(e["id"])
    else:
        # Silent divergence between this tool and xp.csv is exactly how the
        # Mosquera/Calafiori comparison went wrong: 1.74 here, 3.47 there.
        raw = P.load_overrides("overrides.json") \
            if os.path.exists("overrides.json") else {}
        if raw and P.resolve_overrides(boot["elements"], raw).get(e["id"]):
            print("\n  ! overrides.json contains an entry for this player but "
                  "--overrides was not passed.\n    This output will NOT match "
                  "xp.csv. Re-run with --overrides overrides.json.")

    pos = e["element_type"]
    games_played = len({f["event"] for f in fixtures
                        if f.get("team_h_score") is not None})

    print(f"\n{e['web_name']}  ({teams[e['team']]['short_name']} "
          f"{P.POS[pos]}, £{e['now_cost']/10:.1f}m, id {e['id']})")
    print("=" * 62)

    # --- raw inputs --------------------------------------------------------
    print("RAW INPUTS THIS SEASON")
    for k in ("minutes", "starts", "total_points", "expected_goals",
              "expected_assists", "defensive_contribution", "bps",
              "yellow_cards", "saves"):
        print(f"  {k:<26}{P.fnum(e, k):>10.2f}")
    print(f"  {'status':<26}{e.get('status'):>10}")
    print(f"  {'selected_by_percent':<26}{P.fnum(e,'selected_by_percent'):>10.1f}")
    if e.get("news"):
        print(f"  news: {e['news']}")

    # --- priors ------------------------------------------------------------
    last = P.last_season_rates(e["id"])
    prices = sorted(x["now_cost"] / 10.0 for x in boot["elements"]
                    if x["element_type"] == pos)
    med = prices[len(prices) // 2]

    print(f"\nPRIOR SOURCE: ", end="")
    if last:
        print(f"last season ({last.get('_minutes',0):.0f} min, "
              f"{last.get('_starts',0):.0f} starts)")
    else:
        print(f"PRICE HEURISTIC ONLY — no usable prior season")
        print("  ! rates are inferred from cost, not measured. A cheap player")
        print("    with a strong real rate will be underrated for weeks.")

    price_pri = P.price_priors(pos, e["now_cost"] / 10.0, med)
    rates = P.player_rates(e, med, P.PRIOR_MINUTES, last)
    if ov:
        for key, field in (("xg90", "xg"), ("xa90", "xa"),
                           ("defcon90", "defcon"), ("bps90", "bps")):
            if key in ov:
                rates[field] = float(ov[key])
    print(f"\n{'RATE (per 90)':<18}{'PRICE PRIOR':>13}{'LAST SEASON':>13}{'POSTERIOR':>12}")
    print("-" * 56)
    for k in ("xg", "xa", "defcon", "bps", "yellow"):
        ls = f"{last[k]:.2f}" if last and k in last else "—"
        print(f"{k:<18}{price_pri[k]:>13.2f}{ls:>13}{rates[k]:>12.2f}")

    # --- minutes -----------------------------------------------------------
    gk_rank = None
    if pos == P.GKP:
        keepers = sorted((x for x in boot["elements"]
                          if x["element_type"] == P.GKP and x["team"] == e["team"]),
                         key=lambda x: (-x["now_cost"],
                                        -P.fnum(x, "selected_by_percent")))
        gk_rank = next(i for i, x in enumerate(keepers) if x["id"] == e["id"])
        print(f"\nGK depth chart rank at club: {gk_rank} "
              f"({'first choice' if gk_rank == 0 else 'backup'})")

    mm = P.minutes_model(e, games_played, last, gk_rank)
    if ov and ("p_start" in ov or "minutes_mult" in ov or "avail" in ov):
        ps = float(ov.get("p_start", mm.p_start))
        ps = max(0.0, min(1.0, ps * float(ov.get("avail", 1.0))))
        mis = max(10.0, min(90.0,
                            mm.mins_if_start * float(ov.get("minutes_mult", 1.0))))
        p_cameo = mm.p_cameo * float(ov.get("avail", 1.0))
        mm = P.Minutes(
            p_appear=min(1.0, ps + p_cameo),
            p_60=ps * min(1.0, max(0.35, (mis - 12) / 70.0)),
            exp_minutes=ps * mis + p_cameo * mm.cameo_mins,
            p_start=ps, mins_if_start=mis,
            p_cameo=p_cameo, cameo_mins=mm.cameo_mins)
    if ov:
        print(f"\nOVERRIDE APPLIED: {', '.join(f'{k}={v}' for k, v in ov.items() if k != 'note')}")
        if ov.get("note"):
            print(f"  note: {ov['note']}")
    p_app, p_60, exp_min = mm.p_appear, mm.p_60, mm.exp_minutes
    if P.fnum(e, "minutes") > 45 and P.fnum(e, "bps") == 0:
        print("\n  ! bps is 0 despite real minutes — likely a stale field in "
              "FPL's feed. Bonus is being under-projected for this player.")

    print(f"\nMINUTES   p(appear) {p_app:.2f}   p(60+) {p_60:.2f}   "
          f"E[min] {exp_min:.1f}")
    print(f"          p(start) {mm.p_start:.2f} x {mm.mins_if_start:.0f}min  |  "
          f"p(cameo) {mm.p_cameo:.2f} x {mm.cameo_mins:.0f}min")

    # --- fixture -----------------------------------------------------------
    fitted = None
    if args.ratings:
        with open(args.ratings) as f:
            fitted = json.load(f)
    strength = {} if fitted else P.team_strength(teams)

    gw = args.gw or next((ev["id"] for ev in boot["events"] if ev.get("is_next")), 1)
    mine = [f for f in fixtures if f.get("event") == gw
            and e["team"] in (f["team_h"], f["team_a"])]
    if not mine:
        print(f"\nNo fixture in GW{gw} (blank).")
        return

    total = 0.0
    for f in mine:
        home = f["team_h"] == e["team"]
        opp = f["team_a"] if home else f["team_h"]
        lh, la = P.fixture_lambdas(strength, f["team_h"], f["team_a"], fitted)
        lam_for, lam_against = (lh, la) if home else (la, lh)
        print(f"\nGW{gw} vs {teams[opp]['short_name']} "
              f"({'H' if home else 'A'})   "
              f"xG for {lam_for:.2f}, against {lam_against:.2f}   "
              f"P(clean sheet) {math.exp(-lam_against):.2f}")

        # decompose — must use the SAME baseline as projections.py or the
        # numbers here won't reconcile with xp.csv
        frac = exp_min / 90.0
        team_avg = P.team_baseline(fixtures, strength, fitted, e["team"])
        mult = lam_for / team_avg
        print(f"  season baseline for {teams[e['team']]['short_name']}: "
              f"{team_avg:.2f} xG/match -> fixture multiplier {mult:.2f}")
        parts = {
            "appearance": p_app * 1.0 + p_60 * 1.0,
            "goals": rates["xg"] * frac * mult * P.GOAL_PTS[pos],
            "assists": rates["xa"] * frac * mult * P.ASSIST_PTS,
            "clean sheet": (p_60 * math.exp(-lam_against) * P.CS_PTS[pos]
                            if P.CS_PTS[pos] else 0.0),
            "conceded": (-p_60 * P.expected_conceded_penalty(lam_against)
                         if pos in (P.GKP, P.DEF) else 0.0),
            "saves": (P.expected_save_points(lam_against, rates["saves"], frac)
                      if pos == P.GKP else 0.0),
            "bonus": P.expected_bonus(rates["bps"], frac),
            "cards": rates["yellow"] * frac * P.YELLOW_PTS,
        }
        if pos in P.DEFCON_THRESHOLD:
            thr = P.DEFCON_THRESHOLD[pos]
            lam_start = rates["defcon"] * mm.mins_if_start / 90.0
            p_hit_start = P.p_at_least(thr, lam_start)
            parts["defcon"] = P.defcon_points(rates["defcon"], thr, mm)
            print(f"  DefCon: needs {thr}. If he starts, projects "
                  f"{lam_start:.1f} actions -> P(hit) {p_hit_start:.0%}; "
                  f"weighted by p(start) {mm.p_start:.2f} -> "
                  f"{parts['defcon']:.2f} pts")

        print(f"\n  {'COMPONENT':<16}{'xP':>8}")
        print("  " + "-" * 24)
        for k, v in sorted(parts.items(), key=lambda kv: -abs(kv[1])):
            if abs(v) > 0.005:
                print(f"  {k:<16}{v:>8.2f}")
        sub = max(0.0, sum(parts.values()))
        # The model is score_fixture. The breakdown above is a re-derivation
        # for display, so cross-check it — a diagnostic that silently disagrees
        # with the model it explains is worse than no diagnostic.
        truth = P.score_fixture(e, rates, mm, lam_for, lam_against, team_avg)
        print("  " + "-" * 24)
        print(f"  {'TOTAL':<16}{sub:>8.2f}")
        if abs(truth - sub) > 0.02:
            print(f"  {'MODEL SAYS':<16}{truth:>8.2f}   <-- MISMATCH "
                  f"({truth - sub:+.2f}); trust this number, the breakdown "
                  f"above has drifted")
        total += truth

    if len(mine) > 1:
        print(f"\nDouble gameweek — combined xP {total:.2f}")


if __name__ == "__main__":
    main()
