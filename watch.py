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

    python watch.py --entry 8447496 --xp xp.csv

This reports; it doesn't act. There is no mechanism for feeding a correction
back into the model — that was tried and removed (see README). What it tells
you is where the model is working from thin evidence, so you can weigh its
output accordingly rather than patching it.
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

# Roughly what share of each position's xP comes from the player's own measured
# rates, as opposed to team-level inputs (appearance, clean sheet, conceded)
# that the model knows without any player history. Used to weight how much a
# missing prior season actually matters. Heuristic, not fitted — worth
# revisiting once the backtest has per-position calibration.
RATE_DEPENDENCE = {1: 0.25, 2: 0.55, 3: 0.85, 4: 0.95}


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


def threatened_incumbents(old: dict, new: dict, elements: list[dict], teams: dict,
                          squad: set, xp: dict) -> list[str]:
    """A signing threatens whoever already plays that position at that club.

    This is the gap that let the Konsa signing slip past: he shows up only as
    "new to the game", with nothing connecting him to the Arsenal defender you
    already own. A new arrival is the most common reason the model's minutes
    estimate for an incumbent goes stale, and it is invisible from that
    player's own row in the API — his status, news and history are unchanged.
    """
    by_id = {e["id"]: e for e in elements}
    arrivals = [k for k in new if k not in old]
    if not arrivals:
        return []

    lines = []
    for k in arrivals:
        a = by_id.get(int(k))
        if not a:
            continue
        pos, team = a["element_type"], a["team"]
        # Only players you actually hold or have made an assumption about.
        # The xP threshold that used to be here caught every squad player in
        # the league and turned this into a scouting report. Note also that
        # FPL prices new signings low regardless of real fee — Konsa cost £51m
        # and lists at £4.5m — so the arrival's price says nothing about
        # whether he's a threat. Judgement stays with you; this just makes
        # sure the question reaches you.
        rivals = [e for e in elements
                  if e["element_type"] == pos and e["team"] == team
                  and e["id"] != a["id"]
                  and e["id"] in squad]
        if not rivals:
            continue
        club = teams.get(team, {}).get("short_name", "???")
        lines.append(f"\n  {a['web_name']} ({P.POS[pos]}, {club}) has arrived — "
                     f"recheck your assumptions on:")
        for e in sorted(rivals, key=lambda e: -xp.get(e["id"], 0)):
            lines.append(f"      {e['web_name']:<20} "
                         f"{xp.get(e['id'], 0):>5.1f} xP  [IN YOUR SQUAD]")
    if lines:
        return ["\nARRIVALS THREATENING INCUMBENTS"] + lines
    return []


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
                     "rates came from a different role — discount his "
                     "projection until he has minutes at the new club.")

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


def review_queue(elements, teams, xp, squad, deep_ids, top_n):
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
        pos = e["element_type"]

        if pid not in deep_ids:
            # "No prior season" matters in proportion to how much of a
            # player's xP depends on his PERSONAL rates rather than on
            # team-level inputs. A first-choice keeper's projection is mostly
            # appearance points plus his team's clean-sheet probability —
            # both known without him. A forward's is mostly his own xG.
            # Without this, every promoted-club keeper crowds out the players
            # whose ratings are actually uncertain.
            rate_dep = RATE_DEPENDENCE[pos]
            reasons.append(f"no prior season — rates inferred from price "
                           f"({int(rate_dep*100)}% of his xP depends on them)")
            weight += 3.0 * rate_dep
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, help="your FPL team id")
    ap.add_argument("--xp", default="xp.csv", help="projections, for ranking")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--fresh", action="store_true", help="bypass the cache")
    ap.add_argument("--no-save", action="store_true",
                    help="report without updating the cached roster")
    args = ap.parse_args()

    ttl = 0 if args.fresh else 900
    boot = P._get("bootstrap-static/", "bootstrap.json", ttl)
    teams = {t["id"]: t for t in boot["teams"]}
    elements = boot["elements"]

    xp = load_xp(args.xp)
    squad: set[int] = set()
    if args.entry:
        # walk back from the most recent gameweek, not from 40 — picks only
        # exist for deadlines that have passed, and 404s aren't cached, so
        # starting high means dozens of wasted requests every run
        next_gw = next((ev["id"] for ev in boot["events"] if ev.get("is_next")), 1)
        for gw in range(next_gw - 1, 0, -1):
            try:
                picks = P._get(f"entry/{args.entry}/event/{gw}/picks/",
                               f"picks_{args.entry}_{gw}.json")
            except Exception:
                continue
            squad = {p["element"] for p in picks["picks"]}
            print(f"Loaded your GW{gw} squad ({len(squad)} players).")
            break
        if not squad:
            print("! Could not load your squad — squad flags below will be "
                  "missing. Check the entry id.")

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
        lines += threatened_incumbents(old, new_roster, elements, teams,
                                       squad, xp)
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

    queue = review_queue(elements, teams, xp, squad, deep_ids, args.top)
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
