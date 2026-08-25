#!/usr/bin/env python3
"""
watch.py — what changed, and who should I check before this deadline?

Two jobs.

1. ROSTER DIFF. The real-world transfer window stays open until 1 September,
   and FPL adds and removes players as it moves. A player who leaves the league
   keeps his id and flips to status 'u'. A player who changes club keeps his
   last-season rates — which were earned in a different system, under a
   different manager, in a different role — and nothing in the model notices.
   This caches the roster each run and reports what moved.

2. REVIEW QUEUE. The model is confidently wrong in predictable places: players
   with no prior season, players whose role changed, players carrying news the
   API expresses only as a blunt flag. Rather than curating injury notes by
   hand, this ranks who is worth five minutes of your attention, weighted by
   how much they'd actually affect your decisions.

    python watch.py --entry 8447496 --xp xp.csv --overrides overrides.json

Nothing here edits overrides.json. It tells you where to look; the judgement
stays yours.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import projections as P

HIST_DIR = "history"
ROSTER = os.path.join(HIST_DIR, "roster.json")


def load_xp(path: str | None) -> dict[int, float]:
    """Horizon totals per player, so 'impact' means something."""
    if not path or not os.path.exists(path):
        return {}
    out: dict[int, float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = out.get(int(row["id"]), 0.0) + float(row["ep"])
    return out


def snapshot_roster(elements: list[dict], teams: dict) -> dict:
    return {str(e["id"]): {"name": e["web_name"], "team": e["team"],
                           "cost": e["now_cost"], "status": e.get("status", "a"),
                           "news": (e.get("news") or "").strip()}
            for e in elements}


def diff_roster(old: dict, new: dict, teams: dict, xp: dict, squad: set) -> list[str]:
    lines = []

    added = [k for k in new if k not in old]
    gone = [k for k in old if k not in new]
    moved, priced, flagged, cleared = [], [], [], []

    for k, cur in new.items():
        prev = old.get(k)
        if not prev:
            continue
        if prev["team"] != cur["team"]:
            moved.append((k, prev, cur))
        if prev["cost"] != cur["cost"]:
            priced.append((k, prev, cur))
        was_ok = prev["status"] == "a"
        now_ok = cur["status"] == "a"
        if was_ok and not now_ok:
            flagged.append((k, cur))
        elif not was_ok and now_ok:
            cleared.append((k, cur))

    def label(k, d):
        t = teams.get(d["team"], {}).get("short_name", "???")
        mark = "  <-- IN YOUR SQUAD" if int(k) in squad else ""
        return f"{d['name']:<20}{t:>5}  £{d['cost']/10:>4.1f}m{mark}"

    if moved:
        lines.append("\nCLUB CHANGES — last-season rates were earned elsewhere")
        for k, prev, cur in moved:
            a = teams.get(prev["team"], {}).get("short_name", "???")
            b = teams.get(cur["team"], {}).get("short_name", "???")
            mark = "  <-- IN YOUR SQUAD" if int(k) in squad else ""
            lines.append(f"  {cur['name']:<20} {a} -> {b}   "
                         f"(id {k}){mark}")
        lines.append("  ^ new fixtures apply automatically, but his xG/DefCon "
                     "rates came from a different role. Consider an override.")

    if added:
        big = sorted(added, key=lambda k: -new[k]["cost"])[:12]
        lines.append(f"\nNEW TO THE GAME ({len(added)}) — no priors, on the "
                     f"price heuristic")
        for k in big:
            lines.append(f"  {label(k, new[k])}")
        if len(added) > len(big):
            lines.append(f"  ... and {len(added)-len(big)} cheaper")

    if gone:
        lines.append(f"\nREMOVED FROM THE GAME ({len(gone)})")
        for k in gone:
            mark = "  <-- WAS IN YOUR SQUAD" if int(k) in squad else ""
            lines.append(f"  {old[k]['name']}{mark}")

    if flagged:
        lines.append("\nNEWLY UNAVAILABLE")
        for k, cur in sorted(flagged, key=lambda kv: -xp.get(int(kv[0]), 0))[:12]:
            lines.append(f"  {label(k, cur)}")
            if cur["news"]:
                lines.append(f"      {cur['news']}")

    if cleared:
        lines.append("\nBACK TO AVAILABLE — check whether minutes will be managed")
        for k, cur in sorted(cleared, key=lambda kv: -xp.get(int(kv[0]), 0))[:12]:
            lines.append(f"  {label(k, cur)}")

    if priced:
        rises = sorted((p for p in priced if p[2]["cost"] > p[1]["cost"]),
                       key=lambda p: -(p[2]["cost"] - p[1]["cost"]))[:6]
        falls = sorted((p for p in priced if p[2]["cost"] < p[1]["cost"]),
                       key=lambda p: (p[2]["cost"] - p[1]["cost"]))[:6]
        if rises or falls:
            lines.append("\nPRICE MOVES")
            for k, prev, cur in rises + falls:
                d = (cur["cost"] - prev["cost"]) / 10
                mark = "  <-- IN YOUR SQUAD" if int(k) in squad else ""
                lines.append(f"  {cur['name']:<20}{d:+.1f}m  "
                             f"-> £{cur['cost']/10:.1f}m{mark}")
    return lines


def review_queue(elements, teams, xp, squad, overrides, deep_ids, top_n):
    """Rank players by how much a wrong assumption about them would cost.

    Impact is the horizon projection (or squad membership, which forces a
    look regardless). Uncertainty is whether the model is working from
    measured data or a price guess.
    """
    rows = []
    for e in elements:
        pid = e["id"]
        proj = xp.get(pid, 0.0)
        in_squad = pid in squad
        if proj < 8.0 and not in_squad:
            continue

        reasons = []
        weight = 0.0

        if pid not in deep_ids:
            reasons.append("no prior season — rates inferred from price")
            weight += 3.0
        if e.get("status", "a") != "a":
            chance = e.get("chance_of_playing_next_round")
            reasons.append(f"status '{e.get('status')}'"
                           + (f", {chance}% chance" if chance is not None else ""))
            weight += 4.0
        if (e.get("news") or "").strip():
            reasons.append(f"news: {e['news'].strip()}")
            weight += 2.0
        if P.fnum(e, "starts") == 0 and P.fnum(e, "minutes") > 0:
            reasons.append("has minutes but no starts — bench role?")
            weight += 2.0

        if not reasons:
            continue
        if str(pid) in overrides:
            reasons.append("(override already set)")
            weight -= 5.0
        if in_squad:
            weight += 3.0

        rows.append({"id": pid, "name": e["web_name"],
                     "team": teams[e["team"]]["short_name"],
                     "pos": P.POS[e["element_type"]],
                     "cost": e["now_cost"] / 10.0, "xp": proj,
                     "in_squad": in_squad,
                     "score": weight + proj / 10.0, "reasons": reasons})

    rows.sort(key=lambda r: -r["score"])
    return rows[:top_n]


def stale_overrides(elements, overrides, games_played):
    """An override that outlives its situation is worse than none."""
    by_id = {e["id"]: e for e in elements}
    out = []
    for key, val in overrides.items():
        if not key.isdigit():
            continue
        e = by_id.get(int(key))
        if not e:
            out.append((key, "player no longer in the game — DELETE"))
            continue
        starts = P.fnum(e, "starts")
        mins = P.fnum(e, "minutes")
        if starts >= 5 or mins >= 450:
            out.append((f"{e['web_name']} ({key})",
                        f"{int(starts)} starts / {int(mins)} min this season — "
                        f"the model can see this now. Consider removing."))
        elif e.get("status", "a") != "a" and "avail" not in val:
            out.append((f"{e['web_name']} ({key})",
                        f"now flagged status '{e.get('status')}' — override may "
                        f"be asserting minutes he won't get."))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, help="your FPL team id")
    ap.add_argument("--xp", default="xp.csv", help="projections, for ranking")
    ap.add_argument("--overrides", default="overrides.json")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--fresh", action="store_true", help="bypass the cache")
    ap.add_argument("--no-save", action="store_true",
                    help="report without updating the cached roster")
    args = ap.parse_args()

    ttl = 0 if args.fresh else 900
    boot = P._get("bootstrap-static/", "bootstrap.json", ttl)
    teams = {t["id"]: t for t in boot["teams"]}
    elements = boot["elements"]
    games_played = sum(1 for ev in boot["events"] if ev.get("finished"))

    xp = load_xp(args.xp)
    squad: set[int] = set()
    if args.entry:
        try:
            summary = P._get(f"entry/{args.entry}/", f"entry_{args.entry}.json")
            for gw in range(40, 0, -1):
                try:
                    picks = P._get(f"entry/{args.entry}/event/{gw}/picks/",
                                   f"picks_{args.entry}_{gw}.json")
                except Exception:
                    continue
                squad = {p["element"] for p in picks["picks"]}
                break
        except Exception as exc:
            print(f"(couldn't load your squad: {exc})")

    overrides = P.load_overrides(args.overrides) \
        if os.path.exists(args.overrides) else {}

    # which players actually have measured priors
    deep_ids = set()
    for e in elements:
        try:
            if P.last_season_rates(e["id"]) is not None:
                deep_ids.add(e["id"])
        except Exception:
            pass

    new_roster = snapshot_roster(elements, teams)
    print(f"{len(elements)} players in the game, {len(deep_ids)} with measured "
          f"priors, {len(elements)-len(deep_ids)} on the price heuristic.")

    if os.path.exists(ROSTER):
        with open(ROSTER, encoding="utf-8") as f:
            old = json.load(f)
        lines = diff_roster(old, new_roster, teams, xp, squad)
        if lines:
            print("\n" + "=" * 66)
            print("ROSTER CHANGES SINCE LAST RUN")
            print("=" * 66)
            print("\n".join(lines))
        else:
            print("\nNo roster changes since last run.")
    else:
        print("\nNo previous roster cached — this run establishes the baseline. "
              "Diffs start next time.")

    stale = stale_overrides(elements, overrides, games_played)
    if stale:
        print("\n" + "=" * 66)
        print("STALE OVERRIDES")
        print("=" * 66)
        for who, why in stale:
            print(f"  {who}: {why}")

    queue = review_queue(elements, teams, xp, squad, overrides, deep_ids, args.top)
    if queue:
        print("\n" + "=" * 66)
        print("REVIEW QUEUE — check the news on these before the deadline")
        print("=" * 66)
        for r in queue:
            mark = " [SQUAD]" if r["in_squad"] else ""
            print(f"\n  {r['name']} ({r['team']} {r['pos']}, £{r['cost']:.1f}m, "
                  f"id {r['id']}) — {r['xp']:.1f} xP over horizon{mark}")
            for why in r["reasons"]:
                print(f"      - {why}")

    if not args.no_save:
        os.makedirs(HIST_DIR, exist_ok=True)
        with open(ROSTER, "w", encoding="utf-8") as f:
            json.dump(new_roster, f, indent=1, ensure_ascii=False)
        print(f"\nRoster cached to {ROSTER}")


if __name__ == "__main__":
    main()
