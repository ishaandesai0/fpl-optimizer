#!/usr/bin/env python3
"""
congestion.py — who played midweek, and how little rest do they have?

The model reads Premier League fixtures only. It has no idea that Arsenal
played in Europe on Tuesday night, so a player with 48 hours' rest and one
with a full week look identical to it. The OpenFPL paper names this as the
single documented gap between open models and commercial ones: the FPL API
encodes injuries and suspensions but never says whether a manager intends to
rest someone.

Closing that properly means scraping predicted lineups, which is fragile. This
is the cheap version: keep a short list of midweek fixture dates, and before
each deadline see which of your players are on short rest. You then make the
call yourself, the way you already do for role and form.

    python congestion.py --entry 8447496
    python congestion.py --squad squad.json --min-rest 4
    python congestion.py --all                 # every club, not just yours

Dates live in european.json, which you update when UEFA publishes a round —
a few minutes every month or so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import projections as P

TEMPLATE = {
    "_readme": [
        "Midweek fixtures the FPL API cannot see: Champions League, Europa,",
        "Conference, and domestic cups. Add a row per matchday with the clubs",
        "involved, using FPL short names (ARS, MCI, LIV, ...).",
        "",
        "Dates are the kickoff date in YYYY-MM-DD. Teams in the same round",
        "often play on different days, so give each day its own row.",
        "",
        "You only need rounds close to an upcoming deadline — there is no",
        "value in filling in the whole season ahead of time, and a stale",
        "entry is worse than a missing one because it reads as fact."
    ],
    "midweek": [
        {"date": "2026-01-01", "comp": "EXAMPLE — delete this row",
         "teams": []}
    ]
}


def load(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(TEMPLATE, f, indent=2)
        print(f"Created {path} with a template. Add midweek fixtures and "
              f"re-run.")
        return []
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for row in raw.get("midweek", []):
        if not row.get("teams"):
            continue
        try:
            d = dt.date.fromisoformat(row["date"])
        except (KeyError, ValueError):
            print(f"  ! skipping a row with a bad date: {row.get('date')!r}")
            continue
        out.append((d, row.get("comp", "?"), [t.upper() for t in row["teams"]]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int)
    ap.add_argument("--squad", help="squad.json")
    ap.add_argument("--european", default="european.json")
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--min-rest", type=float, default=4.0,
                    help="flag anything below this many days between the "
                         "midweek game and the Premier League kickoff")
    ap.add_argument("--all", action="store_true",
                    help="every club, not only the ones you own")
    args = ap.parse_args()

    midweek = load(args.european)
    if not midweek:
        return

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
    by_short = {v: k for k, v in short.items()}
    meta = {e["id"]: e for e in boot["elements"]}

    gw = P.next_gameweek(boot, fixtures)

    # kickoff per team for the upcoming gameweek
    kick: dict = {}
    for f in fixtures:
        if f.get("event") != gw or not f.get("kickoff_time"):
            continue
        try:
            k = dt.datetime.fromisoformat(
                f["kickoff_time"].replace("Z", "+00:00")).date()
        except ValueError:
            continue
        kick[f["team_h"]] = k
        kick[f["team_a"]] = k

    if not kick:
        sys.exit(f"No kickoff times for GW{gw} yet.")

    # most recent midweek game before each team's kickoff
    rest: dict = {}
    for tid, k in kick.items():
        sn = short[tid]
        best = None
        for d, comp, teams in midweek:
            if sn not in teams or d >= k:
                continue
            gap = (k - d).days
            if best is None or gap < best[0]:
                best = (gap, comp, d)
        if best:
            rest[tid] = best

    if not rest:
        print(f"No midweek fixtures on record before GW{gw}. "
              f"Add them to {args.european} if there were.")
        return

    # which players do we care about
    owned: set = set()
    if args.squad or args.entry:
        if args.squad:
            with open(args.squad, encoding="utf-8") as f:
                raw = json.load(f)
            owned = set(raw["squad"] if isinstance(raw, dict) else raw)
        else:
            try:
                import fpl_opt as F
                ids, _, _ = F.fetch_entry(args.entry, gw)
                owned = set(ids)
            except Exception as e:
                print(f"  ! could not load entry: {e}")

    xp: dict = {}
    if os.path.exists(args.xp):
        import csv
        for r in csv.DictReader(open(args.xp, encoding="utf-8")):
            if int(r["gw"]) == gw:
                xp[int(r["id"])] = float(r["ep"])

    print(f"\nGameweek {gw} — rest since midweek football\n")
    print(f"{'TEAM':<6}{'COMP':<8}{'PLAYED':<12}{'KICKOFF':<12}{'REST':>6}")
    print("-" * 46)
    for tid, (gap, comp, d) in sorted(rest.items(), key=lambda x: x[1][0]):
        mark = "  <-- short" if gap < args.min_rest else ""
        print(f"{short[tid]:<6}{comp:<8}{d.isoformat():<12}"
              f"{kick[tid].isoformat():<12}{gap:>6}{mark}")

    if owned:
        affected = [(pid, meta[pid]) for pid in owned
                    if pid in meta and meta[pid]["team"] in rest]
        if not affected:
            print(f"\nNone of your {len(owned)} players are at a club that "
                  f"played midweek.")
        else:
            print(f"\nYour players on short rest:\n")
            print(f"{'PLAYER':<20}{'TEAM':>5}{'REST':>6}{'GW xP':>8}")
            print("-" * 39)
            rowsort = sorted(affected,
                             key=lambda a: (rest[a[1]['team']][0],
                                            -xp.get(a[0], 0)))
            for pid, e in rowsort:
                gap = rest[e["team"]][0]
                if not args.all and gap >= args.min_rest:
                    continue
                print(f"{e['web_name']:<20}{short[e['team']]:>5}{gap:>6}"
                      f"{xp.get(pid, 0):>8.2f}")

    print("\nThe model cannot see any of this. Check team news before kickoff.")


if __name__ == "__main__":
    main()
