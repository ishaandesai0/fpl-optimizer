#!/usr/bin/env python3
"""
backtest.py — is the model actually any good?

Point-in-time backtesting isn't possible from the FPL API: `bootstrap-static`
is a live snapshot with no history, so you cannot reconstruct what the model
would have said in GW3 once GW9 has been played. The rates, prices and
availability flags have all moved.

So this works forward instead. Before each deadline you snapshot the
projections; after the gameweek resolves you score them against what actually
happened. From GW2 onward you accumulate a real out-of-sample record.

    # before the deadline
    python backtest.py snapshot --gw 2 --xp xp.csv

    # after the gameweek finishes
    python backtest.py score --gw 2

    # once a few weeks have accumulated
    python backtest.py report

Metrics
-------
Correlation (Pearson and Spearman) against realized points; MAE and RMSE;
calibration by decile — do players projected at 5 xP actually average 5? — and
a head-to-head against FPL's own `ep_next`, which is the free alternative and
therefore the bar this model has to clear.

The decision-relevant subset is the top ~60 players by xP, since those are the
only ones you'd ever pick. A model can look great overall purely by correctly
predicting that 400 fringe players will score nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

import projections as P

HIST_DIR = "history"


# ---------------------------------------------------------------------------
# stats helpers (no scipy)
# ---------------------------------------------------------------------------
def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (sx * sy) if sx and sy else float("nan")


def rank(vals: list[float]) -> list[float]:
    """Average ranks, so ties don't distort Spearman."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rank(x), rank(y))


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
def snapshot(gw: int, xp_file: str):
    """Freeze this week's projections plus FPL's own ep_next as a baseline."""
    os.makedirs(HIST_DIR, exist_ok=True)
    boot = P._get("bootstrap-static/", "bootstrap.json")

    mine: dict[int, float] = {}
    with open(xp_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("gw", gw)) == gw:
                mine[int(row["id"])] = float(row["ep"])
    if not mine:
        sys.exit(f"No rows for GW{gw} in {xp_file}")

    rows = []
    for e in boot["elements"]:
        rows.append({
            "id": e["id"],
            "name": e["web_name"],
            "pos": P.POS[e["element_type"]],
            "cost": e["now_cost"] / 10.0,
            "xp": round(mine.get(e["id"], 0.0), 3),
            # the baseline: FPL's own estimate, frozen at the same moment
            "ep_next": P.fnum(e, "ep_next"),
            "selected": P.fnum(e, "selected_by_percent"),
            "status": e.get("status", "a"),
        })

    path = os.path.join(HIST_DIR, f"gw{gw:02d}_snapshot.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    live = sum(1 for r in rows if r["xp"] > 0)
    print(f"Snapshotted GW{gw}: {len(rows)} players ({live} with non-zero xP) "
          f"-> {path}")
    print("Run `backtest.py score --gw %d` once the gameweek finishes." % gw)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def actual_points(gw: int) -> dict[int, dict]:
    """One call gets every player's realized stats for a gameweek."""
    live = P._get(f"event/{gw}/live/", f"live_{gw}.json", ttl=3600)
    out = {}
    for el in live.get("elements", []):
        s = el.get("stats", {})
        out[el["id"]] = {
            "points": s.get("total_points", 0),
            "minutes": s.get("minutes", 0),
            "defcon": s.get("defensive_contribution", 0),
            # bps and bonus are recorded so the hand-fit bonus curve in
            # projections.expected_bonus can eventually be replaced by a
            # regression on what actually happened
            "bps": s.get("bps", 0),
            "bonus": s.get("bonus", 0),
            "goals": s.get("goals_scored", 0),
            "assists": s.get("assists", 0),
            # a keeper's score decomposes into these; without them the
            # goalkeeper model can only be judged on its total, which says
            # nothing about WHICH component is short
            "saves": s.get("saves", 0),
            "conceded": s.get("goals_conceded", 0),
            "pen_saves": s.get("penalties_saved", 0),
            "clean_sheet": s.get("clean_sheets", 0),
        }
    return out


def unfinished_teams(gw: int) -> tuple[set[int], int, int]:
    """Teams whose GW fixture hasn't produced a scoreline yet.

    Scoring a gameweek mid-flight is silently wrong: the live endpoint reports
    0 minutes and 0 points for anyone who hasn't kicked off, so they land in
    the metrics as enormous over-projections. Exclude them rather than let
    them poison the calibration.
    """
    fixtures = P._get("fixtures/", "fixtures.json", ttl=300)
    this_gw = [f for f in fixtures if f.get("event") == gw]
    pending = {f["team_h"] for f in this_gw if f.get("team_h_score") is None} | \
              {f["team_a"] for f in this_gw if f.get("team_a_score") is None}
    done = sum(1 for f in this_gw if f.get("team_h_score") is not None)
    return pending, done, len(this_gw)


def record(gw: int):
    """Write a gameweek's OUTCOMES with no projection attached.

    Two different jobs get confused here. Scoring a projection needs it frozen
    before kickoff — otherwise the model has read the answer sheet. But
    calibrating a component (what bonus does a 30-BPS performance actually
    earn? how much do points vary by position?) needs only outcomes, and works
    on any gameweek regardless of whether a projection was ever saved.

    GW1 2026/27 has no snapshot, so it can never be scored. Its results are
    still perfectly good calibration data, and are already feeding the model's
    priors. This makes them available to the calibration tools without letting
    them near `report`, which reads only *_scored.csv.
    """
    os.makedirs(HIST_DIR, exist_ok=True)
    real = actual_points(gw)
    if not real:
        sys.exit(f"No live data for GW{gw}.")
    pending, done, total = unfinished_teams(gw)
    if pending:
        sys.exit(f"GW{gw} is still in progress ({done}/{total} fixtures).")

    boot = P._get("bootstrap-static/", "bootstrap.json")
    teams = {t["id"]: t for t in boot["teams"]}
    rows = []
    for e in boot["elements"]:
        r = real.get(e["id"])
        if not r or r["minutes"] == 0:
            continue
        rows.append({"id": e["id"], "name": e["web_name"],
                     "pos": P.POS[e["element_type"]],
                     "team": teams[e["team"]]["short_name"],
                     "actual": r["points"], "minutes": r["minutes"],
                     "defcon": r["defcon"], "bps": r["bps"],
                     "bonus": r["bonus"], "goals": r["goals"],
                     "assists": r["assists"], "saves": r["saves"],
                     "conceded": r["conceded"], "pen_saves": r["pen_saves"],
                     "clean_sheet": r["clean_sheet"]})

    path = os.path.join(HIST_DIR, f"gw{gw:02d}_outcomes.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Recorded {len(rows)} appearances from GW{gw} -> {path}")
    print("Outcomes only — no projection attached, so this feeds calibration "
          "but never `report`.")


def score(gw: int, min_price: float = 0.0, partial: bool = False):
    path = os.path.join(HIST_DIR, f"gw{gw:02d}_snapshot.csv")
    if not os.path.exists(path):
        sys.exit(f"No snapshot for GW{gw}. You must snapshot before the deadline.")

    snap = list(csv.DictReader(open(path, encoding="utf-8")))
    real = actual_points(gw)
    if not real:
        sys.exit(f"No live data for GW{gw} yet.")

    pending, done, total = unfinished_teams(gw)
    if pending and not partial:
        sys.exit(f"GW{gw} is still in progress — {done}/{total} fixtures have "
                 f"scorelines.\nPlayers who haven't kicked off report 0 points, "
                 f"which would land in the metrics as huge over-projections.\n"
                 f"Wait for the last match, or pass --partial to score only the "
                 f"{total-len(pending)//2 if total else 0} finished fixtures.")

    boot = P._get("bootstrap-static/", "bootstrap.json")
    team_of = {e["id"]: e["team"] for e in boot["elements"]}
    excluded = 0

    rows = []
    for r in snap:
        pid = int(r["id"])
        if pid not in real:
            continue
        if team_of.get(pid) in pending:
            excluded += 1
            continue
        rows.append({
            "id": pid, "name": r["name"], "pos": r["pos"],
            "cost": float(r["cost"]),
            "xp": float(r["xp"]), "ep_next": float(r["ep_next"]),
            "actual": real[pid]["points"],
            "minutes": real[pid]["minutes"],
            "defcon": real[pid]["defcon"],
            "bps": real[pid]["bps"],
            "bonus": real[pid]["bonus"],
            "saves": real[pid]["saves"],
            "conceded": real[pid]["conceded"],
            "pen_saves": real[pid]["pen_saves"],
            "clean_sheet": real[pid]["clean_sheet"],
        })

    out = os.path.join(HIST_DIR, f"gw{gw:02d}_scored.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'='*64}\nGW{gw} — {len(rows)} players scored\n{'='*64}")
    if excluded:
        print(f"PARTIAL: {done}/{total} fixtures complete. Excluded {excluded} "
              f"players whose match hasn't finished.\nRe-run without --partial "
              f"once the gameweek closes to overwrite with the full result.\n")
    report_gw(rows)
    print(f"\nWrote {out}")


def metrics(rows: list[dict], key: str = "xp") -> dict:
    x = [r[key] for r in rows]
    y = [float(r["actual"]) for r in rows]
    n = len(rows)
    return {
        "n": n,
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "mae": sum(abs(a - b) for a, b in zip(x, y)) / n,
        "rmse": math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)) / n),
        "bias": sum(a - b for a, b in zip(x, y)) / n,
    }


def line(label: str, m: dict) -> str:
    return (f"{label:<22}{m['n']:>6}{m['pearson']:>9.3f}{m['spearman']:>10.3f}"
            f"{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['bias']:>+8.2f}")


def report_gw(rows: list[dict]):
    hdr = (f"{'SUBSET':<22}{'N':>6}{'PEARSON':>9}{'SPEARMAN':>10}"
           f"{'MAE':>8}{'RMSE':>8}{'BIAS':>8}")
    print(hdr)
    print("-" * len(hdr))

    played = [r for r in rows if r["minutes"] > 0]
    top = sorted(rows, key=lambda r: -r["xp"])[:60]

    print(line("all players", metrics(rows)))
    print(line("played only", metrics(played)))
    print(line("top 60 by xP", metrics(top)))
    print(line("  [baseline ep_next]", metrics(top, "ep_next")))

    print(f"\n{'POSITION':<22}{'N':>6}{'PEARSON':>9}{'SPEARMAN':>10}"
          f"{'MAE':>8}{'RMSE':>8}{'BIAS':>8}")
    print("-" * len(hdr))
    for pos in ("GKP", "DEF", "MID", "FWD"):
        sub = [r for r in played if r["pos"] == pos]
        if len(sub) > 5:
            print(line(pos, metrics(sub)))

    # calibration: bucket by projection, compare mean xP to mean actual
    print(f"\nCALIBRATION (played only)")
    print(f"{'xP BUCKET':<14}{'N':>6}{'MEAN xP':>10}{'MEAN ACTUAL':>13}{'GAP':>8}")
    print("-" * 51)
    buckets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 99)]
    for lo, hi in buckets:
        sub = [r for r in played if lo <= r["xp"] < hi]
        if not sub:
            continue
        mx = sum(r["xp"] for r in sub) / len(sub)
        ma = sum(r["actual"] for r in sub) / len(sub)
        label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"{label:<14}{len(sub):>6}{mx:>10.2f}{ma:>13.2f}{ma-mx:>+8.2f}")

    # biggest misses, both directions — these are where to look for bugs
    played.sort(key=lambda r: r["actual"] - r["xp"])
    print(f"\n{'BIGGEST OVER-PROJECTIONS':<26}{'xP':>7}{'ACTUAL':>8}{'MIN':>6}")
    for r in played[:6]:
        print(f"  {r['name']:<24}{r['xp']:>7.2f}{r['actual']:>8}{r['minutes']:>6}")
    print(f"\n{'BIGGEST UNDER-PROJECTIONS':<26}{'xP':>7}{'ACTUAL':>8}{'MIN':>6}")
    for r in played[-6:][::-1]:
        print(f"  {r['name']:<24}{r['xp']:>7.2f}{r['actual']:>8}{r['minutes']:>6}")


# ---------------------------------------------------------------------------
# multi-week report
# ---------------------------------------------------------------------------
def bootstrap_gap(rows: list[dict], trials: int = 2000, seed: int = 0):
    """Confidence interval on (model Spearman - baseline Spearman).

    A single gameweek's comparison on ~60 players is mostly noise: a model
    with no edge whatsoever produces gaps of +/-0.15 by chance. Resampling the
    scored appearances gives an honest interval, so a positive-looking number
    isn't mistaken for evidence before there's enough of it.
    """
    import random
    rng = random.Random(seed)
    n = len(rows)
    gaps = []
    for _ in range(trials):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        try:
            g = (metrics(sample)["spearman"]
                 - metrics(sample, "ep_next")["spearman"])
        except Exception:
            continue
        if g == g:  # skip nan
            gaps.append(g)
    if len(gaps) < 100:
        return None
    gaps.sort()
    return (gaps[int(0.025 * len(gaps))], gaps[int(0.5 * len(gaps))],
            gaps[int(0.975 * len(gaps))])


def report():
    files = sorted(f for f in os.listdir(HIST_DIR) if f.endswith("_scored.csv")) \
        if os.path.isdir(HIST_DIR) else []
    if not files:
        sys.exit("No scored gameweeks yet.")

    allrows = []
    hdr = (f"{'GAMEWEEK':<22}{'N':>6}{'PEARSON':>9}{'SPEARMAN':>10}"
           f"{'MAE':>8}{'RMSE':>8}{'BIAS':>8}")
    print(hdr)
    print("-" * len(hdr))
    for fn in files:
        rows = [dict(r, xp=float(r["xp"]), ep_next=float(r["ep_next"]),
                     actual=float(r["actual"]), minutes=int(r["minutes"]),
                     cost=float(r["cost"]))
                for r in csv.DictReader(open(os.path.join(HIST_DIR, fn),
                                          encoding="utf-8"))]
        played = [r for r in rows if r["minutes"] > 0]
        gw = fn[2:4]
        print(line(f"GW{gw}", metrics(played)))
        allrows += played

    print("-" * len(hdr))
    print(line("POOLED", metrics(allrows)))
    top = sorted(allrows, key=lambda r: -r["xp"])[:60 * len(files)]
    print(line("POOLED top-60/wk", metrics(top)))
    print(line("  [baseline ep_next]", metrics(top, "ep_next")))

    m_model, m_base = metrics(top), metrics(top, "ep_next")
    delta = m_model["spearman"] - m_base["spearman"]
    print(f"\nModel vs FPL's ep_next on the decision-relevant subset: "
          f"{delta:+.3f} Spearman")

    ci = bootstrap_gap(top)
    if ci:
        lo, mid, hi = ci
        print(f"  95% CI [{lo:+.3f}, {hi:+.3f}] from {len(top)} appearances "
              f"across {len(files)} gameweek(s)")
        if lo > 0:
            print("  -> The edge is distinguishable from noise. The model is "
                  "beating the free baseline.")
        elif hi < 0:
            print("  -> The baseline is beating the model. Something is "
                  "wrong; check calibration and the biggest misses.")
        else:
            need = max(0, 5 - len(files))
            print(f"  -> Interval spans zero: this is NOT yet evidence of an "
                  f"edge either way.\n     A model with no edge at all "
                  f"produces gaps this size by chance at this sample size.\n"
                  f"     Roughly {need or 'a few'} more gameweek(s) before the "
                  f"comparison means anything.")

    # allrows is PLAYED-only, so this bias is conditioned on having featured.
    # The model assigns most players p_start < 1, so conditional on playing its
    # expectation is low BY CONSTRUCTION. Reporting it as overall calibration
    # invites exactly the wrong correction. The honest figure is the
    # all-players bias in `score` output, which includes those who didn't play.
    print(f"\nBias among players who FEATURED: {metrics(allrows)['bias']:+.3f}")
    print("  A selection effect, not miscalibration: conditioning on players")
    print("  who played favours those the model gave a low p_start. For real")
    print("  calibration read the 'all players' BIAS line in `score` output,")
    print("  which includes everyone who didn't feature.")

    print("\nPOOLED CALIBRATION")
    print(f"{'xP BUCKET':<14}{'N':>6}{'MEAN xP':>10}{'MEAN ACTUAL':>13}{'GAP':>8}")
    print("-" * 51)
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 99)]:
        sub = [r for r in allrows if lo <= r["xp"] < hi]
        if not sub:
            continue
        mx = sum(r["xp"] for r in sub) / len(sub)
        ma = sum(r["actual"] for r in sub) / len(sub)
        print(f"{(f'{lo}-{hi}' if hi<99 else f'{lo}+'):<14}{len(sub):>6}"
              f"{mx:>10.2f}{ma:>13.2f}{ma-mx:>+8.2f}")


def calibrate_bonus():
    """Fit the bonus curve to what actually happened.

    Reads every scored gameweek, buckets appearances by BPS, and reports mean
    actual bonus per bucket alongside what projections.expected_bonus predicts.
    This is what should set BONUS_FLOOR / BONUS_SLOPE / BONUS_CAP — the current
    values are reasoned from the format's physical limits (6 bonus points per
    match across ~22 starters), not fitted.
    """
    # calibration needs outcomes, not projections, so it reads both kinds
    files = sorted(f for f in os.listdir(HIST_DIR)
                   if f.endswith("_scored.csv") or f.endswith("_outcomes.csv")) \
        if os.path.isdir(HIST_DIR) else []
    if not files:
        sys.exit("No gameweek data yet. Run `score` or `record` first.")
    # if a gameweek has both, prefer the scored file and don't double-count
    seen_gw = set()
    keep = []
    for fn in sorted(files, key=lambda f: (f[2:4], not f.endswith("_scored.csv"))):
        gw = fn[2:4]
        if gw in seen_gw:
            continue
        seen_gw.add(gw)
        keep.append(fn)
    files = keep

    rows = []
    for fn in files:
        for r in csv.DictReader(open(os.path.join(HIST_DIR, fn), encoding="utf-8")):
            if int(r.get("minutes", 0)) < 45:
                continue
            if "bps" not in r or "bonus" not in r:
                continue
            rows.append((float(r["bps"]), float(r["bonus"])))
    if not rows:
        sys.exit("Scored files have no bps/bonus columns — re-score after "
                 "updating backtest.py.")

    print(f"{len(rows)} appearances of 45+ minutes across {len(files)} "
          f"gameweek(s): {', '.join(sorted(seen_gw))}")
    overall = sum(b for _, b in rows) / len(rows)
    print(f"Mean bonus per appearance: {overall:.3f} "
          f"(format implies ~0.27 across all starters)\n")

    print(f"{'BPS BUCKET':<14}{'N':>6}{'MEAN BPS':>10}{'ACTUAL':>9}{'MODEL':>8}")
    print("-" * 47)
    edges = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 45), (45, 999)]
    for lo, hi in edges:
        sub = [(b, bo) for b, bo in rows if lo <= b < hi]
        if not sub:
            continue
        mb = sum(b for b, _ in sub) / len(sub)
        ma = sum(bo for _, bo in sub) / len(sub)
        pred = P.expected_bonus(mb, P.Minutes(1.0, 1.0, 90.0, 1.0, 90.0, 0.0, 18.0))
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"{label:<14}{len(sub):>6}{mb:>10.1f}{ma:>9.2f}{pred:>8.2f}")

    print("\nMODEL is expected_bonus() at that bucket's mean BPS, assuming a "
          "start.")

    # The table above maps a SINGLE MATCH's BPS to that match's bonus. The
    # model applies its curve to a shrunk per-90 RATE. Different functions:
    # bonus is convex in BPS across 20-40, so a player whose RATE is 30 earns
    # more on average than a single 30-BPS match returns. Fitting to the
    # realization curve therefore understates. Grouping by PLAYER gives
    # E[bonus | rate], which is what expected_bonus actually consumes.
    by_player: dict = {}
    for fn in files:
        for r in csv.DictReader(open(os.path.join(HIST_DIR, fn), encoding="utf-8")):
            if int(r.get("minutes", 0)) < 45 or "bps" not in r:
                continue
            p = by_player.setdefault(r["id"], {"bps": [], "bonus": []})
            p["bps"].append(float(r["bps"]))
            p["bonus"].append(float(r["bonus"]))

    multi = {k: v for k, v in by_player.items() if len(v["bps"]) >= 2}
    if len(multi) < 30:
        print("\nNot enough players with 2+ appearances for the per-player "
              "view yet.")
        return

    print(f"\nPER-PLAYER view — {len(multi)} players with 2+ appearances.")
    print("This is the correct target: a player's mean BPS vs the bonus he "
          "actually averages.")
    print(f"{'BPS RATE':<14}{'N':>6}{'MEAN BPS':>10}{'ACTUAL':>9}{'MODEL':>8}")
    print("-" * 47)
    M = P.Minutes(1.0, 1.0, 90.0, 1.0, 90.0, 0.0, 18.0)
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 999)]:
        sub = [v for v in multi.values()
               if lo <= sum(v["bps"]) / len(v["bps"]) < hi]
        if not sub:
            continue
        mb = sum(sum(v["bps"]) / len(v["bps"]) for v in sub) / len(sub)
        ma = sum(sum(v["bonus"]) / len(v["bonus"]) for v in sub) / len(sub)
        pred = P.expected_bonus(mb, M)
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        print(f"{label:<14}{len(sub):>6}{mb:>10.1f}{ma:>9.2f}{pred:>8.2f}")

    print("\nFit BONUS_FLOOR / BONUS_SLOPE / BONUS_CAP to THIS table, not the "
          "per-match one.")


def calibrate_gk():
    """Where do goalkeeper points actually come from, and what does the model
    think?

    GK has been the worst-predicted position all season — Pearson at or below
    zero in three of five gameweeks, bias around -1.3. A total that's wrong
    tells you nothing about WHICH term is short, so this decomposes the real
    points into the components the scoring rules define and puts the model's
    own average beside each.
    """
    files = _gameweek_files()
    if not files:
        sys.exit("No gameweek data yet.")

    # _gameweek_files prefers scored files, but those predate the saves
    # columns. Take whichever file for a gameweek actually carries them.
    by_gw: dict = {}
    for fn in sorted(os.listdir(HIST_DIR)):
        if not (fn.endswith("_scored.csv") or fn.endswith("_outcomes.csv")):
            continue
        path = os.path.join(HIST_DIR, fn)
        with open(path, encoding="utf-8") as fh:
            header = (fh.readline() or "").split(",")
        if "saves" not in [h.strip() for h in header]:
            continue
        by_gw.setdefault(fn[2:4], path)      # first match wins; both are fine
    files = list(by_gw.values())

    rows = []
    for path in files:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("pos") != "GKP" or int(r.get("minutes", 0)) < 60:
                continue
            rows.append(r)

    print(f"Using {len(files)} gameweek file(s) with save data: "
          f"{', '.join(sorted(by_gw))}")

    if len(rows) < 20:
        sys.exit("Not enough goalkeeper data with saves recorded.\n"
                 "Re-run `record --gw N` for each gameweek after updating "
                 "backtest.py — the older files predate these columns.")

    n = len(rows)
    f = lambda r, k: float(r.get(k) or 0)

    app = 2.0                                     # every 60+ minute start
    cs = sum(4.0 if f(r, "clean_sheet") else 0.0 for r in rows) / n
    conc = sum(-(int(f(r, "conceded")) // 2) for r in rows) / n
    sav = sum(int(f(r, "saves")) // 3 for r in rows) / n
    pens = sum(5.0 * f(r, "pen_saves") for r in rows) / n
    bon = sum(f(r, "bonus") for r in rows) / n
    actual = sum(f(r, "actual") for r in rows) / n

    print(f"\n{n} goalkeeper appearances of 60+ minutes.\n")
    print(f"{'COMPONENT':<18}{'ACTUAL/GAME':>13}")
    print("-" * 31)
    print(f"{'appearance':<18}{app:>13.2f}")
    print(f"{'clean sheet':<18}{cs:>13.2f}")
    print(f"{'saves':<18}{sav:>13.2f}")
    print(f"{'bonus':<18}{bon:>13.2f}")
    print(f"{'penalty saves':<18}{pens:>13.2f}")
    print(f"{'conceded':<18}{conc:>13.2f}")
    print("-" * 31)
    print(f"{'SUM':<18}{app + cs + sav + bon + pens + conc:>13.2f}")
    print(f"{'ACTUAL TOTAL':<18}{actual:>13.2f}")

    raw_saves = sum(f(r, "saves") for r in rows) / n
    cs_rate = sum(1 for r in rows if f(r, "clean_sheet")) / n
    print(f"\nRaw rates: {raw_saves:.2f} saves/game, "
          f"{cs_rate:.0%} clean sheet rate, "
          f"{sum(f(r,'conceded') for r in rows)/n:.2f} conceded/game")
    print(f"Penalty saves: {sum(f(r,'pen_saves') for r in rows):.0f} in {n} "
          f"appearances")

    # what a model with no penalty-save term is structurally missing
    print(f"\nThe model has NO penalty-save term at all, so it is short "
          f"{pens:.2f}/game there by construction.")
    print("Compare `saves` above against what projections.expected_save_points")
    print("returns for a keeper at the raw rate — if the model is much lower,")
    print("the floor-of-3 rule is being evaluated at a mean rather than")
    print("integrated over the distribution of saves.")

    import projections as PJ
    M = PJ.Minutes(1.0, 1.0, 90.0, 1.0, 90.0, 0.0, 18.0)
    model_saves = PJ.expected_save_points(1.35, raw_saves, 1.0, 1.35)
    print(f"\n  model save points at {raw_saves:.2f} saves/90, neutral "
          f"fixture: {model_saves:.2f}")
    print(f"  actual save points per appearance:                  {sav:.2f}")
    if sav > model_saves * 1.25:
        print("  -> the model is materially under-crediting saves.")


def _gameweek_files():
    """Scored files preferred, outcomes as fallback, one per gameweek."""
    if not os.path.isdir(HIST_DIR):
        return []
    files = sorted(f for f in os.listdir(HIST_DIR)
                   if f.endswith("_scored.csv") or f.endswith("_outcomes.csv"))
    seen, keep = set(), []
    for fn in sorted(files, key=lambda f: (f[2:4], not f.endswith("_scored.csv"))):
        if fn[2:4] in seen:
            continue
        seen.add(fn[2:4])
        keep.append(fn)
    return keep


def calibrate_variance():
    """Observed spread of gameweek points by position.

    `draft.player_sd` is a hand-fit base-plus-slope guess that feeds P(win) in
    head-to-head matchups directly — an inflated value makes every fixture look
    like a coin flip and blunts the favoured/underdog advice. This measures it
    instead. Uses outcomes only, so every gameweek counts.
    """
    files = sorted(f for f in os.listdir(HIST_DIR)
                   if f.endswith("_scored.csv") or f.endswith("_outcomes.csv")) \
        if os.path.isdir(HIST_DIR) else []
    if not files:
        sys.exit("No gameweek data yet.")
    seen, keep = set(), []
    for fn in sorted(files, key=lambda f: (f[2:4], not f.endswith("_scored.csv"))):
        if fn[2:4] in seen:
            continue
        seen.add(fn[2:4]); keep.append(fn)

    by_pos: dict[str, list[float]] = {}
    for fn in keep:
        for r in csv.DictReader(open(os.path.join(HIST_DIR, fn), encoding="utf-8")):
            if int(r.get("minutes", 0)) < 60:
                continue
            by_pos.setdefault(r["pos"], []).append(float(r["actual"]))

    print(f"Appearances of 60+ minutes across gameweeks {', '.join(sorted(seen))}\n")
    print(f"{'POS':<6}{'N':>6}{'MEAN':>8}{'SD':>8}{'P90':>7}{'MAX':>6}")
    print("-" * 41)
    for pos in ("GKP", "DEF", "MID", "FWD"):
        v = sorted(by_pos.get(pos, []))
        if len(v) < 10:
            continue
        n = len(v)
        mean = sum(v) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / max(n - 1, 1))
        print(f"{pos:<6}{n:>6}{mean:>8.2f}{sd:>8.2f}{v[int(.9*n)]:>7.0f}{v[-1]:>6.0f}")
    print("\nCompare SD here against draft.player_sd(mean_xp, pos). If the "
          "model's sd is much\nlarger, P(win) is being pulled toward 50% and "
          "the H2H advice is too timid.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="freeze projections before a deadline")
    s.add_argument("--gw", type=int, required=True)
    s.add_argument("--xp", default="xp.csv")

    c = sub.add_parser("score", help="score a finished gameweek")
    c.add_argument("--gw", type=int, required=True)
    c.add_argument("--partial", action="store_true",
                   help="score only the fixtures that have finished")

    sub.add_parser("report", help="pooled results across all scored gameweeks")
    rec = sub.add_parser("record",
                         help="save a gameweek's outcomes with no projection "
                              "attached — for calibration, never for scoring")
    rec.add_argument("--gw", type=int, required=True)

    sub.add_parser("calibrate-bonus",
                   help="fit the bonus curve against recorded bps/bonus")
    sub.add_parser("calibrate-variance",
                   help="per-position points variance, for draft.player_sd")
    sub.add_parser("calibrate-gk",
                   help="decompose goalkeeper points — the worst-predicted "
                        "position all season")

    args = ap.parse_args()
    if args.cmd == "snapshot":
        snapshot(args.gw, args.xp)
    elif args.cmd == "score":
        score(args.gw, partial=args.partial)
    elif args.cmd == "record":
        record(args.gw)
    elif args.cmd == "calibrate-bonus":
        calibrate_bonus()
    elif args.cmd == "calibrate-variance":
        calibrate_variance()
    elif args.cmd == "calibrate-gk":
        calibrate_gk()
    else:
        report()


if __name__ == "__main__":
    main()
