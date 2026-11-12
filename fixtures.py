#!/usr/bin/env python3
"""
fixtures.py — who plays whom, with the model's view of each side.

    python fixtures.py                 # next gameweek
    python fixtures.py --gw 7
    python fixtures.py --gw 5 --ratings ratings.json
    python fixtures.py --team MCI --next 6

Exists because the equivalent PowerShell one-liner needs nested quotes and
gets mangled every time.
"""

from __future__ import annotations

import argparse
import json

import projections as P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gw", type=int, help="gameweek (default: next)")
    ap.add_argument("--ratings", help="ratings.json, to show expected goals")
    ap.add_argument("--team", help="short name, e.g. MCI — shows this team's run")
    ap.add_argument("--next", type=int, default=5,
                    help="gameweeks to show with --team or --all")
    ap.add_argument("--all", action="store_true",
                    help="every team's run, ranked — the wildcard planning view")
    ap.add_argument("--sort", choices=("against", "for", "net"),
                    default="against",
                    help="rank by xG conceded (defence targets), xG scored "
                         "(attack targets), or the ratio")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
    by_short = {v: k for k, v in short.items()}

    fitted = None
    if args.ratings:
        with open(args.ratings, encoding="utf-8") as f:
            fitted = json.load(f)
    strength = {} if fitted else P.team_strength(
        {t["id"]: t for t in boot["teams"]})

    def lams(fx):
        return P.fixture_lambdas(strength, fx["team_h"], fx["team_a"], fitted)

    if args.all:
        start = args.gw or P.next_gameweek(boot, fixtures)
        gws = set(range(start, start + args.next))
        rows = []
        for tid, name in short.items():
            lf = la = 0.0
            n = 0
            for fx in fixtures:
                if fx.get("event") not in gws:
                    continue
                if tid not in (fx["team_h"], fx["team_a"]):
                    continue
                home = fx["team_h"] == tid
                h, a = lams(fx)
                lf += h if home else a
                la += a if home else h
                n += 1
            if not n:
                continue
            rows.append({"team": name, "n": n,
                         "for": lf / n, "agn": la / n,
                         "net": (lf / n) / max(la / n, 0.1)})
        key = {"against": lambda r: r["agn"],
               "for": lambda r: -r["for"],
               "net": lambda r: -r["net"]}[args.sort]
        rows.sort(key=key)

        print(f"\nGW{start}-{start + args.next - 1}, averaged per match. "
              f"Sorted by {args.sort}.")
        print(f"{'TEAM':<6}{'GMS':>5}{'xG FOR':>9}{'xG AGN':>9}{'RATIO':>8}")
        print("-" * 37)
        for r in rows:
            flag = "  <- blank/double" if r["n"] != args.next else ""
            print(f"{r['team']:<6}{r['n']:>5}{r['for']:>9.2f}"
                  f"{r['agn']:>9.2f}{r['net']:>8.2f}{flag}")
        print()
        return

    if args.team:
        tid = by_short.get(args.team.upper())
        if not tid:
            raise SystemExit(f"unknown team {args.team!r}. "
                             f"Options: {' '.join(sorted(by_short))}")
        start = args.gw or P.next_gameweek(boot, fixtures)
        gws = range(start, start + args.next)
        print(f"\n{args.team.upper()}, GW{start}-{start + args.next - 1}")
        print(f"{'GW':>4}{'OPPONENT':>12}{'H/A':>5}{'xG FOR':>9}{'xG AGN':>9}")
        print("-" * 39)
        tot_f = tot_a = 0.0
        for g in gws:
            played = [f for f in fixtures
                      if f.get("event") == g and tid in (f["team_h"], f["team_a"])]
            if not played:
                print(f"{g:>4}{'BLANK':>12}")
                continue
            for fx in played:
                home = fx["team_h"] == tid
                opp = short[fx["team_a"] if home else fx["team_h"]]
                lh, la = lams(fx)
                lf, lag = (lh, la) if home else (la, lh)
                tot_f += lf
                tot_a += lag
                print(f"{g:>4}{opp:>12}{'H' if home else 'A':>5}"
                      f"{lf:>9.2f}{lag:>9.2f}")
        n = max(args.next, 1)
        print("-" * 39)
        print(f"{'avg':>4}{'':>12}{'':>5}{tot_f/n:>9.2f}{tot_a/n:>9.2f}")
        return

    gw = args.gw or P.next_gameweek(boot, fixtures)
    this = [f for f in fixtures if f.get("event") == gw]
    if not this:
        raise SystemExit(f"no fixtures found for GW{gw}")

    print(f"\nGameweek {gw}")
    if fitted:
        print(f"{'HOME':>6}{'':3}{'AWAY':<6}{'xG H':>8}{'xG A':>8}"
              f"{'CS H':>7}{'CS A':>7}")
        print("-" * 46)
        import math
        for fx in sorted(this, key=lambda f: f.get("kickoff_time") or ""):
            lh, la = lams(fx)
            print(f"{short[fx['team_h']]:>6}{' v ':3}{short[fx['team_a']]:<6}"
                  f"{lh:>8.2f}{la:>8.2f}"
                  f"{math.exp(-la):>7.0%}{math.exp(-lh):>7.0%}")

    else:
        for fx in this:
            print(f"{short[fx['team_h']]:>6} v {short[fx['team_a']]}")
        print("\n--ratings ratings.json adds xG and clean-sheet odds.")


if __name__ == "__main__":
    main()
