#!/usr/bin/env python3
"""
test_form.py — does recent form beat the model's xP?

The claim: the model under-rates players who have been producing, because
shrinkage pulls every observed rate back toward a prior. De Cuyper is the
example — 3.39 projected, 15 scored, and his underlying numbers said he was an
attacking wing-back the whole time.

The counter-claim is just as plausible: chasing points-form is the classic FPL
error, because finishing regresses and the player you buy at his peak is the
one about to cool off. Barry cuts that way — the model shrank him toward 0.39
xG and he has sustained 0.82.

Both are testable against the record rather than argued about.

    python test_form.py
    python test_form.py --min-prior-minutes 180

Method is walk-forward: to predict gameweek g, use only gameweeks before g.
No peeking. Each predictor is z-scored within its gameweek before blending, so
an expected-points figure and a points-per-90 figure can be combined without
one swamping the other by scale.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

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


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return out


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def zscore(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((a - m) ** 2 for a in v) / max(n - 1, 1))
    return [(a - m) / sd for a in v] if sd else [0.0] * n


def bootstrap_diff(a, b, y, trials=2000, seed=0):
    import random
    rng = random.Random(seed)
    n = len(y)
    d = []
    for _ in range(trials):
        idx = [rng.randrange(n) for _ in range(n)]
        da = spearman([a[i] for i in idx], [y[i] for i in idx])
        db = spearman([b[i] for i in idx], [y[i] for i in idx])
        if da == da and db == db:
            d.append(da - db)
    if len(d) < 100:
        return None
    d.sort()
    return d[int(0.025 * len(d))], d[int(0.975 * len(d))]


def read_gw(gw, suffix):
    path = os.path.join(HIST_DIR, f"gw{gw:02d}_{suffix}.csv")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-prior-minutes", type=int, default=135,
                    help="minutes needed BEFORE the predicted gameweek")
    args = ap.parse_args()

    gws = sorted(int(f[2:4]) for f in os.listdir(HIST_DIR)
                 if f.endswith("_scored.csv")) if os.path.isdir(HIST_DIR) else []
    if len(gws) < 3:
        sys.exit("Need at least 3 scored gameweeks.")

    samples = []          # (xp, form_pts90, form_bps90, actual, gw, name)
    for g in gws[1:]:     # need >=2 prior gameweeks to form an estimate
        prior_pts, prior_min, prior_bps = {}, {}, {}
        for pg in range(1, g):
            rows = read_gw(pg, "outcomes") or read_gw(pg, "scored")
            if not rows:
                continue
            for r in rows:
                pid = int(r["id"])
                prior_pts[pid] = prior_pts.get(pid, 0.0) + float(r["actual"])
                prior_min[pid] = prior_min.get(pid, 0.0) + float(r["minutes"])
                prior_bps[pid] = prior_bps.get(pid, 0.0) + float(r.get("bps") or 0)

        cur = read_gw(g, "scored")
        if not cur:
            continue
        for r in cur:
            pid = int(r["id"])
            if float(r["minutes"]) <= 0:
                continue
            pm = prior_min.get(pid, 0.0)
            if pm < args.min_prior_minutes:
                continue
            samples.append({
                "xp": float(r["xp"]),
                "form": prior_pts[pid] / (pm / 90.0),
                "bps": prior_bps[pid] / (pm / 90.0),
                "actual": float(r["actual"]),
                "gw": g,
                "name": r["name"],
            })

    if len(samples) < 100:
        sys.exit(f"Only {len(samples)} samples — need more gameweeks.")

    # z-score within gameweek so the scales are comparable before blending
    by_gw: dict = {}
    for s in samples:
        by_gw.setdefault(s["gw"], []).append(s)
    for g, rows in by_gw.items():
        for key in ("xp", "form", "bps"):
            z = zscore([r[key] for r in rows])
            for r, v in zip(rows, z):
                r["z_" + key] = v

    y = [s["actual"] for s in samples]
    n_gw = len(by_gw)
    print(f"{len(samples)} player-gameweeks across GW{min(by_gw)}-{max(by_gw)}, "
          f"walk-forward.\n"
          f"Each prediction uses only gameweeks before the one it predicts.\n")

    cands = {
        "model xP":              [s["z_xp"] for s in samples],
        "form (pts/90)":         [s["z_form"] for s in samples],
        "BPS form":              [s["z_bps"] for s in samples],
        "xP + 0.25 form":        [s["z_xp"] + 0.25 * s["z_form"] for s in samples],
        "xP + 0.50 form":        [s["z_xp"] + 0.50 * s["z_form"] for s in samples],
        "xP + 1.00 form":        [s["z_xp"] + 1.00 * s["z_form"] for s in samples],
        "xP + 0.25 BPS form":    [s["z_xp"] + 0.25 * s["z_bps"] for s in samples],
    }

    print(f"{'PREDICTOR':<22}{'PEARSON':>9}{'SPEARMAN':>10}{'vs xP':>9}")
    print("-" * 50)
    base = cands["model xP"]
    base_s = spearman(base, y)
    for name, x in cands.items():
        p, s = pearson(x, y), spearman(x, y)
        delta = "" if name == "model xP" else f"{s - base_s:+.3f}"
        print(f"{name:<22}{p:>9.3f}{s:>10.3f}{delta:>9}")

    print("\nThe point estimate is not the answer — these predictors are highly")
    print("correlated with each other, so a gap can look decisive and be noise.\n")
    for name in ("form (pts/90)", "xP + 0.25 form", "xP + 0.50 form",
                 "xP + 1.00 form"):
        ci = bootstrap_diff(cands[name], base, y)
        if not ci:
            continue
        lo, hi = ci
        verdict = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "no different"
        print(f"  {name:<22} 95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

    # The direct version of the claim: is the model biased against players who
    # have been outproducing it?
    print("\n" + "=" * 62)
    print("Is the model systematically wrong about form players?")
    print("=" * 62)
    hot = [s for s in samples if s["z_form"] - s["z_xp"] > 0.75]
    cold = [s for s in samples if s["z_form"] - s["z_xp"] < -0.75]
    mid = [s for s in samples if abs(s["z_form"] - s["z_xp"]) <= 0.75]
    for label, grp in (("form well ABOVE xP", hot),
                       ("form in line with xP", mid),
                       ("form well BELOW xP", cold)):
        if len(grp) < 20:
            continue
        bias = sum(s["xp"] - s["actual"] for s in grp) / len(grp)
        print(f"  {label:<24} n={len(grp):<5} mean xP {sum(s['xp'] for s in grp)/len(grp):.2f}"
              f"  actual {sum(s['actual'] for s in grp)/len(grp):.2f}"
              f"  bias {bias:+.2f}")
    print("\nIf the 'above' group is much more negative than the others, the")
    print("model really is under-rating in-form players and a form term is")
    print("justified. If the biases are similar, the apparent pattern is just")
    print("the model being under-projected everywhere, which it already is.")


if __name__ == "__main__":
    main()
