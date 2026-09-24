#!/usr/bin/env python3
"""
test_advanced.py — do shots and key passes add anything over xG and xA?

The argument for them is statistical rather than a preference between
providers: goals are rare, so xG is noisy at five gameweeks, while shots are
several times more frequent and should stabilise sooner. If that's true, a
blend of the two should predict realised points better than xG alone.

This checks it against the scored gameweeks rather than assuming it.

    python test_advanced.py
    python test_advanced.py --advanced advanced.csv --min-minutes 180

Method: for each attacking player with enough minutes, correlate several
candidate predictors against actual points per 90 across all scored
gameweeks. A predictor only earns its place if it beats what's already there.
"""

from __future__ import annotations

import argparse
import csv
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
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx and sy else float("nan")


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x, y):
    return pearson(rank(x), rank(y))


def bootstrap_diff(x_a, x_b, y, trials=2000, seed=0):
    """CI on corr(a,y) - corr(b,y). Two predictors on the same players are
    highly correlated with each other, so the naive difference looks more
    decisive than it is; resampling the players gives an honest interval."""
    import random
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        a = [x_a[i] for i in idx]
        b = [x_b[i] for i in idx]
        t = [y[i] for i in idx]
        da, db = spearman(a, t), spearman(b, t)
        if da == da and db == db:
            diffs.append(da - db)
    if len(diffs) < 100:
        return None
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--advanced", default="advanced.csv")
    ap.add_argument("--min-minutes", type=int, default=180)
    args = ap.parse_args()

    if not os.path.exists(args.advanced):
        sys.exit(f"{args.advanced} not found — run advanced.py first")

    adv = {int(r["id"]): r for r in
           csv.DictReader(open(args.advanced, encoding="utf-8"))}

    # realised points per 90 across every scored gameweek
    tot_pts: dict = {}
    tot_min: dict = {}
    files = sorted(f for f in os.listdir(HIST_DIR)
                   if f.endswith("_outcomes.csv") or f.endswith("_scored.csv")) \
        if os.path.isdir(HIST_DIR) else []
    seen = set()
    use = []
    for fn in sorted(files, key=lambda f: (f[2:4], not f.endswith("_outcomes.csv"))):
        if fn[2:4] in seen:
            continue
        seen.add(fn[2:4])
        use.append(fn)
    for fn in use:
        for r in csv.DictReader(open(os.path.join(HIST_DIR, fn), encoding="utf-8")):
            pid = int(r["id"])
            tot_pts[pid] = tot_pts.get(pid, 0.0) + float(r["actual"])
            tot_min[pid] = tot_min.get(pid, 0.0) + float(r["minutes"])

    boot = P._get("bootstrap-static/", "bootstrap.json")
    meta = {e["id"]: e for e in boot["elements"]}

    rows = []
    for pid, a in adv.items():
        e = meta.get(pid)
        if not e or tot_min.get(pid, 0) < args.min_minutes:
            continue
        if e["element_type"] not in (3, 4):      # attacking returns only
            continue
        mins = tot_min[pid]
        fpl_min = float(e["minutes"] or 0) or mins
        rows.append({
            "name": e["web_name"],
            "pts90": tot_pts[pid] / (mins / 90.0),
            "fpl_xg": float(e["expected_goals"] or 0) / (fpl_min / 90.0),
            "fpl_xa": float(e["expected_assists"] or 0) / (fpl_min / 90.0),
            "np_xg": float(a["np_xg90"]),
            "und_xa": float(a["xa90"]),
            "shots": float(a["shots90"]),
            "kp": float(a["key_passes90"]),
            "chain": float(a["chain90"]),
        })

    if len(rows) < 30:
        sys.exit(f"Only {len(rows)} players qualify — need more data.")

    y = [r["pts90"] for r in rows]
    print(f"{len(rows)} midfielders and forwards with {args.min_minutes}+ "
          f"minutes, across gameweeks {', '.join(sorted(seen))}.")
    print("Target: realised points per 90.\n")

    # candidate predictors, including the blends worth testing
    cands = {
        "FPL xG+xA":            [r["fpl_xg"] + r["fpl_xa"] for r in rows],
        "Understat npxG+xA":    [r["np_xg"] + r["und_xa"] for r in rows],
        "shots only":           [r["shots"] for r in rows],
        "key passes only":      [r["kp"] for r in rows],
        "shots + KP":           [r["shots"] + r["kp"] for r in rows],
        "xG+xA, half shots+KP": [r["fpl_xg"] + r["fpl_xa"]
                                 + 0.5 * (r["shots"] + r["kp"]) / 4.0
                                 for r in rows],
        "xg_chain":             [r["chain"] for r in rows],
    }

    base_name = "FPL xG+xA"
    base = cands[base_name]

    print(f"{'PREDICTOR':<24}{'PEARSON':>9}{'SPEARMAN':>10}{'vs base':>10}")
    print("-" * 53)
    results = {}
    for name, x in cands.items():
        p, s = pearson(x, y), spearman(x, y)
        results[name] = s
        delta = "" if name == base_name else f"{s - results[base_name]:+.3f}"
        print(f"{name:<24}{p:>9.3f}{s:>10.3f}{delta:>10}")

    print(f"\nBaseline is what the model already uses ({base_name}).")
    print("A predictor only earns its place if it beats that by more than")
    print("noise — so the interval below, not the point estimate, decides.\n")

    for name in ("shots + KP", "xG+xA, half shots+KP", "Understat npxG+xA"):
        ci = bootstrap_diff(cands[name], base, y)
        if not ci:
            continue
        lo, hi = ci
        verdict = ("BETTER" if lo > 0 else
                   "WORSE" if hi < 0 else
                   "no different")
        print(f"  {name:<24} 95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    print("\nIf every interval spans zero, the extra data adds nothing")
    print("measurable and the right call is to leave the model alone.")


if __name__ == "__main__":
    main()
