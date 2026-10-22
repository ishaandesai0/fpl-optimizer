#!/usr/bin/env python3
"""
absences.py — who else is the minutes model reading as "dropped"?

The minutes model divides starts by league gameweeks played. A player who
missed five weeks injured and has started both games since looks like "2
starts in 7" — a rotation risk — when he is actually nailed. Doku was the case
that surfaced this; he will not be the only one.

The signature is detectable without any injury feed: a run of consecutive
gameweeks with zero minutes, followed by a return to playing. Nothing else
produces that pattern. A player who is simply out of favour appears as
scattered zeros and cameos, not a clean block.

    python absences.py --min-price 5.0
    python absences.py --xp xp.csv --top 20 --template

Emits availability.json scaffolding for the players worth correcting. It only
proposes COMPLETED absences — a run of zeros at the end of the record means he
is still unavailable, and the model is right to discount him.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import projections as P


def absence_evidence(history, window):
    """Is this gap injury-shaped or bench-shaped?

    Nothing in the FPL data states a reason, so this is evidence rather than
    proof. But the two cases do not look the same to the market: an injury is
    announced, managers dump the player, and his price falls. A player quietly
    left out drifts instead — ownership erodes slowly and the price holds.

    Returns (transfers_out during the gap per gameweek, the same figure for
    gameweeks he played, price change across the gap in GBPm).
    """
    by_round: dict = {}
    for h in history:
        r = h["round"]
        d = by_round.setdefault(r, {"mins": 0, "out": 0, "value": None})
        d["mins"] += h.get("minutes", 0)
        d["out"] += h.get("transfers_out", 0) or 0
        if d["value"] is None:
            d["value"] = h.get("value")

    gap = [by_round[r] for r in window if r in by_round]
    rest = [d for r, d in by_round.items() if r not in set(window)
            and d["mins"] > 0]
    if not gap:
        return 0.0, 0.0, 0.0

    out_gap = sum(d["out"] for d in gap) / len(gap)
    out_rest = (sum(d["out"] for d in rest) / len(rest)) if rest else 0.0

    vals = [d["value"] for d in gap if d["value"] is not None]
    dv = (vals[-1] - vals[0]) / 10.0 if len(vals) > 1 else 0.0
    return out_gap, out_rest, dv


def runs_of_zero(history):
    """Maximal runs of consecutive zero-minute gameweeks, as (start, end)."""
    rounds = sorted({h["round"] for h in history})
    mins = {}
    for h in history:
        mins[h["round"]] = mins.get(h["round"], 0) + h.get("minutes", 0)
    out, run = [], []
    for r in rounds:
        if mins.get(r, 0) == 0:
            run.append(r)
        else:
            if run:
                out.append((run[0], run[-1]))
            run = []
    if run:
        out.append((run[0], run[-1]))
    return out, rounds, mins


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--min-run", type=int, default=3,
                    help="consecutive blank gameweeks that count as an absence")
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--availability", default="availability.json")
    ap.add_argument("--template", action="store_true",
                    help="emit availability.json scaffolding")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    gw_now = P.next_gameweek(boot, fixtures)

    xp: dict = {}
    if os.path.exists(args.xp):
        for r in csv.DictReader(open(args.xp, encoding="utf-8")):
            xp[int(r["id"])] = xp.get(int(r["id"]), 0.0) + float(r["ep"])

    have = set()
    if os.path.exists(args.availability):
        raw = P.load_availability(args.availability)
        have = {int(k) for k in raw if str(k).isdigit()}

    pool = [e for e in boot["elements"]
            if e["now_cost"] / 10.0 >= args.min_price
            and P.fnum(e, "minutes") > 0]
    print(f"Checking {len(pool)} players at £{args.min_price}m+ with minutes "
          f"(cached; first run is slow)...")

    found = []
    for e in pool:
        try:
            hist = P._get(f"element-summary/{e['id']}/",
                          f"element-{e['id']}.json").get("history", [])
        except Exception:
            continue
        if not hist:
            continue
        zeros, rounds, mins = runs_of_zero(hist)
        if not rounds:
            continue
        last_round = max(rounds)

        # only completed absences: a run that ends before the latest gameweek
        completed = [(a, b) for a, b in zeros
                     if b - a + 1 >= args.min_run and b < last_round]
        if not completed:
            continue

        missed = []
        for a, b in completed:
            missed += list(range(a, b + 1))

        played = len(rounds)
        starts = P.fnum(e, "starts")
        avail = max(1, played - len(missed))
        naive = starts / max(played, 1)
        corrected = starts / avail

        out_gap, out_rest, dv = absence_evidence(hist, missed)
        # a transfer-out rate well above his baseline, or a price fall during
        # the gap, is what an announced injury looks like
        ratio = out_gap / out_rest if out_rest > 0 else 0.0
        shape = ("injury" if (ratio >= 1.8 or dv <= -0.2) else
                 "unclear" if ratio >= 1.2 or dv < 0 else
                 "bench?")

        found.append({
            "e": e, "missed": missed, "played": played,
            "ratio": ratio, "dv": dv, "shape": shape,
            "starts": starts, "naive": naive, "corrected": corrected,
            "lift": corrected - naive,
            "xp": xp.get(e["id"], 0.0),
            "since": sum(1 for r in rounds if r > max(m for m in missed)
                         and mins.get(r, 0) > 0),
            "have": e["id"] in have,
        })

    if not found:
        print("\nNo completed absences of "
              f"{args.min_run}+ consecutive gameweeks found.")
        return

    found.sort(key=lambda f: -f["lift"])
    shown = found[:args.top]

    print(f"\n{len(found)} players have a completed absence. "
          f"Showing the {len(shown)} most distorted.\n")
    print(f"{'':3}{'PLAYER':<17}{'TEAM':>4}{'MISSED':>8}{'NAIVE':>7}"
          f"{'FIXED':>7}{'TRF-OUT':>9}{'Δ£':>7}{'LOOKS':>9}{'HZN':>7}")
    print("-" * 80)
    for f in shown:
        e = f["e"]
        mark = " ok" if f["have"] else "  •"
        rng = (f"{f['missed'][0]}-{f['missed'][-1]}"
               if len(f["missed"]) > 1 else str(f["missed"][0]))
        print(f"{mark}{e['web_name']:<17}{teams[e['team']]:>4}{rng:>8}"
              f"{f['naive']:>7.2f}{f['corrected']:>7.2f}"
              f"{f['ratio']:>8.1f}x{f['dv']:>7.1f}{f['shape']:>9}"
              f"{f['xp']:>7.1f}")

    print("\n  ok = already in availability.json     • = uncorrected")
    print("\nTRF-OUT is how much faster managers sold him during the gap than")
    print("when he was playing. Δ£ is the price move across it. An announced")
    print("injury shows both; a quiet benching shows neither — that is the")
    print("only way the data distinguishes them, and it is evidence rather")
    print("than proof. 'bench?' rows are the ones most likely to be a healthy")
    print("player left out, and correcting those would flatter him wrongly.")
    print("\nNAIVE is starts divided by gameweeks played, which is what the")
    print("model currently uses. FIXED divides by gameweeks he was available")
    print("for. The bigger the gap, the more the model is reading an injury")
    print("as a manager's decision.")
    print("\nA correction only shrinks the denominator — it cannot invent")
    print("starts, so a player who genuinely is not trusted stays low.")

    if args.template:
        todo = [f for f in shown if not f["have"] and f["shape"] != "bench?"]
        if not todo:
            print("\nNothing to scaffold — every uncorrected row looks like a "
                  "benching\nrather than an absence.")
            return
        scaffold = {
            str(f["e"]["id"]): {
                "missed": f["missed"],
                "note": (f"{f['e']['web_name']} ({teams[f['e']['team']]}) — "
                         f"blank in GW{f['missed'][0]}-{f['missed'][-1]}, "
                         f"{f['since']} appearance(s) since. Sold "
                         f"{f['ratio']:.1f}x faster than usual during the gap, "
                         f"price {f['dv']:+.1f} — looks like {f['shape']}. "
                         f"VERIFY before keeping.")
            } for f in todo
        }
        print("\n" + "=" * 62)
        print("SCAFFOLD — paste into availability.json under \"players\"")
        print("=" * 62)
        print(json.dumps(scaffold, indent=2, ensure_ascii=False))
        print("\nCHECK EACH ONE before keeping it. A run of zeros is strong")
        print("evidence of absence but not proof — a player can be dropped for")
        print("a month. projections.py refuses any gameweek he actually")
        print("played in, but it cannot tell injury from exile.")


if __name__ == "__main__":
    main()
