#!/usr/bin/env python3
"""
noprior.py — who does the model have nothing on?

Lists every player without a measured Premier League season, ranked by how
much it matters. These are the players running on the price heuristic, and
they're the only ones a foreign-league prior can legitimately be supplied for
(projections.py refuses entries for anyone with real PL data).

    python noprior.py --min-price 5.5
    python noprior.py --xp xp.csv --top 40
    python noprior.py --template          # emits priors.json scaffolding

Rank is price x horizon xP where available: an expensive player the model is
guessing about is a bigger problem than a cheap one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import projections as P


def load_xp(path):
    if not path or not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = out.get(int(row["id"]), 0.0) + float(row["ep"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-price", type=float, default=5.0,
                    help="ignore players below this price (£m)")
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--priors", default="priors.json")
    ap.add_argument("--template", action="store_true",
                    help="print priors.json scaffolding for the listed players")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    teams = {t["id"]: t for t in boot["teams"]}
    xp = load_xp(args.xp)

    existing = set()
    if os.path.exists(args.priors):
        raw = P.load_foreign_priors(args.priors)
        existing = {P.strip_accents(k) for k in raw if not k.isdigit()} | \
                   {k for k in raw if k.isdigit()}

    print(f"Checking {len(boot['elements'])} players for Premier League "
          f"history (cached; first run is slow)...")

    rows = []
    for e in boot["elements"]:
        price = e["now_cost"] / 10.0
        if price < args.min_price:
            continue
        try:
            if P.last_season_rates(e["id"]) is not None:
                continue          # has PL data — not eligible
        except Exception:
            pass
        have = (P.strip_accents(e["web_name"]) in existing
                or str(e["id"]) in existing)
        rows.append({
            "id": e["id"], "name": e["web_name"],
            "team": teams[e["team"]]["short_name"],
            "pos": P.POS[e["element_type"]], "price": price,
            "xp": xp.get(e["id"], 0.0),
            "mins": P.fnum(e, "minutes"),
            "have": have,
        })

    # rank by price and projected output together: an expensive player the
    # model is guessing about costs more than a cheap one
    rows.sort(key=lambda r: -(r["price"] * 2 + r["xp"]))
    shown = rows[:args.top]

    print(f"\n{len(rows)} players at £{args.min_price}m+ have NO Premier "
          f"League season.\n")
    print(f"{'':3}{'PLAYER':<20}{'TEAM':>5}{'POS':>5}{'£':>7}{'HORIZON xP':>12}"
          f"{'MIN':>6}")
    print("-" * 58)
    for r in shown:
        mark = " ok" if r["have"] else "  •"
        print(f"{mark}{r['name']:<20}{r['team']:>5}{r['pos']:>5}"
              f"{r['price']:>7.1f}{r['xp']:>12.1f}{r['mins']:>6.0f}")
    print("\n  ok = already in priors.json     • = running on the price guess")

    if args.template:
        todo = [r for r in shown if not r["have"]]
        scaffold = {
            str(r["id"]): {
                "league": "FILL_ME",
                "season": "2025/26",
                "minutes": 0,
                "starts": 0,
                "goals": 0,
                "assists": 0,
                "note": f"{r['name']} ({r['team']} {r['pos']}, £{r['price']:.1f}m)"
            } for r in todo
        }
        print("\n" + "=" * 58)
        print("SCAFFOLD — paste into priors.json under \"players\" and fill in")
        print("=" * 58)
        print(json.dumps(scaffold, indent=2, ensure_ascii=False))
        print("\nLeagues: ligue1 la_liga bundesliga serie_a eredivisie")
        print("         primeira_liga belgian_pro super_lig championship mls other")
        print("Leave an entry out entirely rather than guessing at its numbers —")
        print("a made-up season is worse than the price heuristic it replaces.")


if __name__ == "__main__":
    main()
