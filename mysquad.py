#!/usr/bin/env python3
"""
mysquad.py — write squad.json reflecting transfers you've already made.

FPL only publishes picks for gameweeks whose deadline has passed. Between
making a transfer and the deadline, `entry/{id}/event/{gw}` 404s and the
pipeline falls back to your previous squad — so the optimizer plans around
players you've already sold and can't see the ones you bought.

This takes your last published squad, applies the swaps by name, and writes a
file `fpl_opt.py --squad` can read.

    python mysquad.py --entry 8447496 --out Mosquera,Mbeumo --in Thiaw,Cherki

Names are matched the same way diagnose.py matches them: accents optional,
initials optional, ambiguous names refused rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import sys

import projections as P
import fpl_opt as F


def resolve(elements, needle):
    n = P.strip_accents(needle)
    exact = [e for e in elements if P.strip_accents(e["web_name"]) == n]
    hits = exact or [e for e in elements
                     if len(n) >= 4 and n in P.strip_accents(e["web_name"])]
    if not hits:
        sys.exit(f"No player matching {needle!r}")
    if len(hits) > 1:
        opts = ", ".join(f"{e['web_name']} (id {e['id']})" for e in hits[:8])
        sys.exit(f"{needle!r} is ambiguous — {opts}. Use the id instead.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, required=True)
    ap.add_argument("--out", default="", metavar="NAMES",
                    help="comma-separated players sold (names or ids)")
    ap.add_argument("--in", dest="incoming", default="", metavar="NAMES",
                    help="comma-separated players bought (names or ids)")
    ap.add_argument("--bank", type=float, default=None,
                    help="money in the bank after the transfers, in £m")
    ap.add_argument("--file", default="squad.json")
    args = ap.parse_args()

    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    elements = boot["elements"]
    by_id = {e["id"]: e for e in elements}
    gw = P.next_gameweek(boot, fixtures)

    ids, bank, _ = F.fetch_entry(args.entry, gw)
    squad = list(ids)

    def pick(token):
        token = token.strip()
        if not token:
            return None
        if token.isdigit():
            e = by_id.get(int(token))
            if not e:
                sys.exit(f"id {token} is not a player")
            return e
        return resolve(elements, token)

    outs = [pick(t) for t in args.out.split(",") if t.strip()]
    ins = [pick(t) for t in args.incoming.split(",") if t.strip()]
    if len(outs) != len(ins):
        sys.exit(f"{len(outs)} out but {len(ins)} in — a transfer is a swap, "
                 f"so these must match.")

    for e in outs:
        if e["id"] not in squad:
            sys.exit(f"{e['web_name']} (id {e['id']}) isn't in your published "
                     f"GW{gw-1} squad — already applied?")
        squad.remove(e["id"])
    for e in ins:
        if e["id"] in squad:
            sys.exit(f"{e['web_name']} is already in the squad")
        squad.append(e["id"])

    if len(squad) != 15:
        sys.exit(f"ended up with {len(squad)} players, not 15")
    counts: dict[int, int] = {}
    for pid in squad:
        counts[by_id[pid]["element_type"]] = \
            counts.get(by_id[pid]["element_type"], 0) + 1
    if counts != {1: 2, 2: 5, 3: 5, 4: 3}:
        print(f"  ! squad shape is {counts}, expected 2/5/5/3 — check the swaps")
    clubs: dict[int, int] = {}
    for pid in squad:
        clubs[by_id[pid]["team"]] = clubs.get(by_id[pid]["team"], 0) + 1
    over = [t for t, c in clubs.items() if c > 3]
    if over:
        print(f"  ! more than 3 players from a club — check the swaps")

    for e in outs:
        print(f"  OUT  {e['web_name']}")
    for e in ins:
        print(f"  IN   {e['web_name']}")

    payload = {"squad": squad, "gw": gw}
    if args.bank is not None:
        payload["bank"] = args.bank
    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nWrote {args.file} for GW{gw}. Use it with:")
    bankflag = f" --bank {args.bank}" if args.bank is not None else ""
    print(f"  python fpl_opt.py --squad {args.file} --projections xp.csv "
          f"--horizon 5 --free-transfers 0{bankflag}")


if __name__ == "__main__":
    main()
