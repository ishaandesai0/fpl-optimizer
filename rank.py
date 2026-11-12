#!/usr/bin/env python3
"""
rank.py — the shortlist view, for building a squad rather than judging a player.

`diagnose.py` explains ONE player in depth. This ranks many, so you can pick
who deserves the deep look.

    python rank.py MID
    python rank.py DEF --max-price 5.5
    python rank.py MID --sort value
    python rank.py FWD --min-price 8 --top 15

Ranks on the HORIZON total, not this gameweek. A single fixture is the wrong
basis for a wildcard: several players who top a weekly list are one-week
clean-sheet or multiplier plays whose runs are mediocre.
"""

from __future__ import annotations

import argparse
import csv
import os

import projections as P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pos", nargs="?", default="ALL",
                    choices=["GKP", "DEF", "MID", "FWD", "ALL"])
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-price", type=float, default=0.0)
    ap.add_argument("--max-price", type=float, default=99.0)
    ap.add_argument("--sort", choices=("total", "value", "gw"), default="total",
                    help="total = horizon xP (default), value = xP per GBPm, "
                         "gw = this gameweek only")
    ap.add_argument("--team", help="restrict to one club, e.g. ARS")
    ap.add_argument("--no-prior", action="store_true",
                    help="only players with no measured season — the ones "
                         "running on a price guess")
    args = ap.parse_args()

    if not os.path.exists(args.xp):
        raise SystemExit(f"{args.xp} not found — run projections.py first")

    boot = P._get("bootstrap-static/", "bootstrap.json")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    meta = {e["id"]: e for e in boot["elements"]}

    tot: dict[int, float] = {}
    first: dict[int, float] = {}
    gws: set[int] = set()
    with open(args.xp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid, g, ep = int(row["id"]), int(row["gw"]), float(row["ep"])
            gws.add(g)
            tot[pid] = tot.get(pid, 0.0) + ep
            if g == min(gws) or pid not in first:
                first[pid] = ep
    gw0 = min(gws)
    with open(args.xp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gw"]) == gw0:
                first[int(row["id"])] = float(row["ep"])

    rows = []
    for pid, t in tot.items():
        e = meta.get(pid)
        if not e:
            continue
        p = P.POS[e["element_type"]]
        if args.pos != "ALL" and p != args.pos:
            continue
        price = e["now_cost"] / 10.0
        if not (args.min_price <= price <= args.max_price):
            continue
        club = teams[e["team"]]
        if args.team and club != args.team.upper():
            continue
        if args.no_prior:
            try:
                if P.last_season_rates(pid) is not None:
                    continue
            except Exception:
                pass
        rows.append({
            "name": e["web_name"], "team": club, "pos": p, "price": price,
            "total": t, "per": t / len(gws), "gw": first.get(pid, 0.0),
            "value": t / price, "own": P.fnum(e, "selected_by_percent"),
            "status": e.get("status", "a"),
        })

    key = {"total": lambda r: -r["total"],
           "value": lambda r: -r["value"],
           "gw": lambda r: -r["gw"]}[args.sort]
    rows.sort(key=key)

    lbl = args.pos if args.pos != "ALL" else "all positions"
    print(f"\n{lbl}, GW{min(gws)}-{max(gws)}, sorted by {args.sort}"
          + (f", {args.team.upper()} only" if args.team else "")
          + (", NO PRIOR SEASON only" if args.no_prior else ""))
    print(f"{'':2}{'PLAYER':<19}{'TEAM':>5}{'£':>6}{'GW':>6}{'TOTAL':>7}"
          f"{'PER GW':>8}{'xP/£m':>7}{'OWN%':>7}")
    print("-" * 68)
    for r in rows[:args.top]:
        flag = " !" if r["status"] != "a" else "  "
        print(f"{flag}{r['name']:<19}{r['team']:>5}{r['price']:>6.1f}"
              f"{r['gw']:>6.2f}{r['total']:>7.1f}{r['per']:>8.2f}"
              f"{r['value']:>7.2f}{r['own']:>7.1f}")

    print("\n! = flagged.")


if __name__ == "__main__":
    main()
