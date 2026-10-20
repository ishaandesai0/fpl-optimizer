#!/usr/bin/env python3
"""
test_minutes_cal.py — is p_start actually calibrated?

Every projection is p_start x points-if-playing, so if p_start is biased the
whole model is scaled wrong. The backtest's -0.8 bias among players who
featured is NOT evidence of that — it is the definition of a conditional
average, and the all-players bias of -0.01 says the model is right in
aggregate. But "right in aggregate" can hide offsetting errors: too low on
nailed starters and too high on fringe players would net out to zero.

This checks the only thing that matters: of the players the model gave a 0.8
chance of starting, did 80% start?

    python test_minutes_cal.py

Needs snapshot files (the frozen projections) and outcomes files (what
happened). It reconstructs p_start from the same inputs projections.py used,
so it measures the model as it actually ran rather than as it runs today.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import projections as P

HIST_DIR = "history"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-minutes", type=int, default=60,
                    help="minutes that count as having started")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    meta = {e["id"]: e for e in boot["elements"]}

    gws = sorted(int(f[2:4]) for f in os.listdir(HIST_DIR)
                 if f.endswith("_outcomes.csv")) if os.path.isdir(HIST_DIR) else []
    if len(gws) < 3:
        sys.exit("Need at least 3 gameweeks of outcomes.")

    # Reconstruct p_start as the model saw it BEFORE each gameweek: rates and
    # minutes built only from earlier gameweeks. Using today's totals would
    # leak the outcome back into the prediction.
    rows = []
    for g in gws:
        prior_min, prior_starts = {}, {}
        for pg in gws:
            if pg >= g:
                continue
            path = os.path.join(HIST_DIR, f"gw{pg:02d}_outcomes.csv")
            if not os.path.exists(path):
                continue
            for r in csv.DictReader(open(path, encoding="utf-8")):
                pid = int(r["id"])
                m = float(r["minutes"])
                prior_min[pid] = prior_min.get(pid, 0.0) + m
                if m >= args.start_minutes:
                    prior_starts[pid] = prior_starts.get(pid, 0) + 1

        games_played = len([x for x in gws if x < g])
        if games_played < 2:
            continue

        cur = os.path.join(HIST_DIR, f"gw{g:02d}_outcomes.csv")
        played = {int(r["id"]): float(r["minutes"])
                  for r in csv.DictReader(open(cur, encoding="utf-8"))}

        for pid, e in meta.items():
            if prior_min.get(pid, 0) <= 0:
                continue          # no prior evidence at all; model can't rank him
            # rebuild the element dict as it would have looked
            shim = dict(e)
            shim["minutes"] = prior_min[pid]
            shim["starts"] = prior_starts.get(pid, 0)
            shim["status"] = "a"
            shim["chance_of_playing_next_round"] = None
            try:
                last = P.last_season_rates(pid)
            except Exception:
                last = None
            try:
                mm = P.minutes_model(shim, games_played, last, None)
            except Exception:
                continue
            rows.append({
                "p_start": mm.p_start,
                "started": 1 if played.get(pid, 0.0) >= args.start_minutes else 0,
                "minutes": played.get(pid, 0.0),
                "exp_min": mm.exp_minutes,
                "name": e["web_name"],
                "gw": g,
            })

    if len(rows) < 200:
        sys.exit(f"Only {len(rows)} player-gameweeks — need more data.")

    print(f"{len(rows)} player-gameweeks across GW{min(r['gw'] for r in rows)}"
          f"-{max(r['gw'] for r in rows)}.")
    print(f"'Started' means {args.start_minutes}+ minutes.\n")

    bands = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.55),
             (0.55, 0.7), (0.7, 0.85), (0.85, 1.01)]
    print(f"{'p_start BAND':<16}{'N':>6}{'MEAN p_start':>14}{'ACTUAL':>9}"
          f"{'GAP':>8}")
    print("-" * 53)
    tot_pred = tot_act = 0.0
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r["p_start"] < hi]
        if len(sub) < 15:
            continue
        pred = sum(r["p_start"] for r in sub) / len(sub)
        act = sum(r["started"] for r in sub) / len(sub)
        tot_pred += sum(r["p_start"] for r in sub)
        tot_act += sum(r["started"] for r in sub)
        flag = "  <--" if abs(pred - act) > 0.08 else ""
        print(f"{lo:.2f}-{hi:<11.2f}{len(sub):>6}{pred:>14.3f}{act:>9.3f}"
              f"{act - pred:>+8.3f}{flag}")

    print("-" * 53)
    print(f"{'OVERALL':<16}{len(rows):>6}{tot_pred/len(rows):>14.3f}"
          f"{tot_act/len(rows):>9.3f}{(tot_act - tot_pred)/len(rows):>+8.3f}")

    # minutes, not just the binary
    pred_m = sum(r["exp_min"] for r in rows) / len(rows)
    act_m = sum(r["minutes"] for r in rows) / len(rows)
    print(f"\nExpected minutes {pred_m:.1f} vs actual {act_m:.1f} "
          f"({act_m - pred_m:+.1f})")

    print("\nA band whose ACTUAL sits well above its MEAN p_start means the")
    print("model under-rates those players' chance of starting. If that is")
    print("concentrated in the high bands, nailed starters are being")
    print("discounted and every projection they drive is too low.")
    print("\nA uniform offset would NOT change rankings — it scales everything")
    print("equally — but it would shift the transfer-edge thresholds, since a")
    print("'+2.0 over five gameweeks' would really be larger.")
    print("Offsetting errors across bands WOULD change rankings, and that is")
    print("the case worth acting on.")


if __name__ == "__main__":
    main()
