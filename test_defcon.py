#!/usr/bin/env python3
"""
test_defcon.py — is DefCon really fixture-independent?

`score_fixture` applies a fixture multiplier to goals, assists and clean
sheets, and NONE to defensive contributions. The reasoning was that tackles
and interceptions are a function of the player, not the opponent — which is
why cheap high-DefCon defenders were recommended as a floor that survives a
bad run.

That may be wrong. A side expected to dominate spends the match with the ball,
so its defenders have less to defend. Arsenal at home to Hull should produce
fewer defensive actions from Arsenal centre-backs than the same players away
at City. If so, the model is mis-ranking exactly the position it predicts
worst — defenders have been the weakest position in all six scored gameweeks.

    python test_defcon.py --ratings ratings.json

Method: for every defender-gameweek on record, pair the observed DefCon count
with the fixture's expected goal difference. A negative correlation means
dominance suppresses defensive actions. The bootstrap interval decides, not
the point estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import projections as P

HIST_DIR = "history"


def pearson(x, y):
    n = len(x)
    if n < 5:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def bootstrap_r(x, y, trials=2000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(y)
    out = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        r = pearson([x[i] for i in idx], [y[i] for i in idx])
        if r == r:
            out.append(r)
    if len(out) < 100:
        return None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", default="ratings.json")
    ap.add_argument("--min-minutes", type=int, default=60)
    args = ap.parse_args()

    fitted = None
    if os.path.exists(args.ratings):
        with open(args.ratings, encoding="utf-8") as f:
            fitted = json.load(f)
    else:
        sys.exit(f"{args.ratings} not found — run ratings.py first")

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    meta = {e["id"]: e for e in boot["elements"]}

    # fixture lambdas per (gameweek, team)
    lam: dict = {}
    for f in fixtures:
        g = f.get("event")
        if not g:
            continue
        lh, la = P.fixture_lambdas({}, f["team_h"], f["team_a"], fitted)
        lam[(g, f["team_h"])] = (lh, la)
        lam[(g, f["team_a"])] = (la, lh)

    rows = []
    for fn in sorted(os.listdir(HIST_DIR)):
        if not fn.endswith("_outcomes.csv"):
            continue
        g = int(fn[2:4])
        for r in csv.DictReader(open(os.path.join(HIST_DIR, fn), encoding="utf-8")):
            if float(r["minutes"]) < args.min_minutes:
                continue
            e = meta.get(int(r["id"]))
            if not e:
                continue
            key = (g, e["team"])
            if key not in lam:
                continue
            lf, la = lam[key]
            rows.append({
                "pos": r["pos"],
                "defcon": float(r.get("defcon") or 0),
                "mins": float(r["minutes"]),
                "lam_for": lf,
                "lam_against": la,
                "dominance": lf - la,      # expected goal difference
                "gw": g,
            })

    if len(rows) < 100:
        sys.exit(f"Only {len(rows)} rows — need more gameweeks.")

    print(f"{len(rows)} player-gameweeks with {args.min_minutes}+ minutes.\n")

    for pos in ("DEF", "MID", "ALL"):
        sub = [r for r in rows if pos == "ALL" or r["pos"] == pos]
        if len(sub) < 60:
            continue
        # per-90 so a substitute's short cameo doesn't drag the count down
        d90 = [r["defcon"] / (r["mins"] / 90.0) for r in sub]
        dom = [r["dominance"] for r in sub]
        ag = [r["lam_against"] for r in sub]

        r_dom = pearson(dom, d90)
        r_ag = pearson(ag, d90)
        ci = bootstrap_r(dom, d90)

        print(f"{pos} (n={len(sub)})")
        print(f"  DefCon/90 vs expected goal difference : {r_dom:+.3f}")
        print(f"  DefCon/90 vs expected goals AGAINST   : {r_ag:+.3f}")
        if ci:
            lo, hi = ci
            verdict = ("dominance SUPPRESSES DefCon" if hi < 0 else
                       "dominance RAISES DefCon" if lo > 0 else
                       "no measurable effect")
            print(f"  95% CI on the first [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
        print()

    # practical size: what would a fixture adjustment actually be worth?
    defs = [r for r in rows if r["pos"] == "DEF"]
    if len(defs) >= 60:
        defs.sort(key=lambda r: r["dominance"])
        third = len(defs) // 3
        low = defs[:third]            # expected to be dominated
        high = defs[-third:]          # expected to dominate
        f90 = lambda g: sum(r["defcon"] / (r["mins"] / 90.0) for r in g) / len(g)
        print("=" * 58)
        print("Practical size, defenders split by expected dominance:")
        print(f"  most dominated third   DefCon/90 {f90(low):.2f}  "
              f"(mean GD {sum(r['dominance'] for r in low)/len(low):+.2f})")
        print(f"  most dominant third    DefCon/90 {f90(high):.2f}  "
              f"(mean GD {sum(r['dominance'] for r in high)/len(high):+.2f})")
        gap = f90(low) - f90(high)
        print(f"  difference: {gap:+.2f} actions per 90")
        print(f"\nThe DefCon threshold is 10 for defenders, so a gap of {gap:.2f}")
        print("moves the hit probability materially only if it is a couple of")
        print("actions or more. Below that it is not worth modelling.")


if __name__ == "__main__":
    main()
