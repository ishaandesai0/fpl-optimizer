#!/usr/bin/env python3
"""
chips.py — when to play the chips you have left.

Every chip is worth its INCREMENT, not its headline. Triple Captain on a
player who scores 9 pays 27 instead of 18 — it is worth 9, not 27. Bench Boost
is worth the bench minus whatever the automatic substitutions would have given
you anyway. Free Hit is worth the best legal squad you could field minus the
one you would have fielded. Valuing them any other way makes every chip look
enormous and every week look like the right week.

    python chips.py --entry 8447496
    python chips.py --entry 8447496 --through 19
    python chips.py --entry 8447496 --no-freehit     # skip the slow part

Reads which chips are already gone from the entry's own history, so it cannot
disagree with the app. The first set expires at the Gameweek 19 deadline and
does not carry over, so the planner treats that as a hard wall.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import projections as P

# The first set of chips expires at this deadline and cannot be carried into
# the second half; a fresh set unlocks from GW20.
FIRST_HALF_LAST_GW = 19

CHIP_NAMES = {
    "wildcard": "Wildcard",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
    "freehit": "Free Hit",
}


def chip_state(entry_id: int) -> dict:
    """Which chips this entry has already played, and in which gameweek."""
    hist = P._get(f"entry/{entry_id}/history/", f"entry_{entry_id}_hist.json")
    used = {}
    for c in hist.get("chips", []):
        name = c.get("name")
        used[name] = c.get("event")
    return used


def fixture_counts(fixtures, teams) -> dict:
    """Fixtures per team per gameweek. 0 is a blank, 2+ a double."""
    counts: dict = {}
    for f in fixtures:
        gw = f.get("event")
        if gw is None:
            continue              # postponed, not yet rescheduled
        for side in ("team_h", "team_a"):
            counts.setdefault(gw, {}).setdefault(f[side], 0)
            counts[gw][f[side]] += 1
    # a team absent from a gameweek entirely has a blank
    for gw in counts:
        for t in teams:
            counts[gw].setdefault(t, 0)
    return counts


def load_xp(path: str) -> dict:
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["id"]), {})[int(r["gw"])] = float(r["ep"])
    return out


def best_xi_split(squad_ids, xp, meta, gw):
    """Return (xi_ids, bench_ids, xi_total) maximising this gameweek."""
    rows = []
    for pid in squad_ids:
        e = meta.get(pid)
        if not e:
            continue
        rows.append((pid, e["element_type"], xp.get(pid, {}).get(gw, 0.0)))

    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 11 - 1 - d - m
            if not (1 <= f <= 3):
                continue
            pick, ok = [], True
            for t, need in ((1, 1), (2, d), (3, m), (4, f)):
                cand = sorted((r for r in rows if r[1] == t),
                              key=lambda r: -r[2])
                if len(cand) < need:
                    ok = False
                    break
                pick += cand[:need]
            if not ok:
                continue
            tot = sum(r[2] for r in pick)
            if best is None or tot > best[0]:
                best = (tot, pick)
    if best is None:
        return [], list(squad_ids), 0.0
    xi = {r[0] for r in best[1]}
    return list(xi), [p for p in squad_ids if p not in xi], best[0]


def autosub_credit(xi_ids, bench_ids, xp, meta, gw, p_appear):
    """What the automatic substitutions are already expected to be worth.

    Bench Boost does not add the whole bench — a starter who plays nothing is
    replaced by a bench player anyway. Only the part the autosubs would NOT
    have delivered is new. This is the expected number of blanking starters
    times the value of the bench players who would come on.
    """
    exp_blanks = sum(1.0 - p_appear.get(pid, 0.9) for pid in xi_ids)
    bench_vals = sorted((xp.get(p, {}).get(gw, 0.0) for p in bench_ids),
                        reverse=True)
    credit = 0.0
    for i in range(min(int(exp_blanks) + 1, len(bench_vals))):
        share = min(1.0, max(0.0, exp_blanks - i))
        credit += share * bench_vals[i]
    return credit


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, required=True)
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--through", type=int, default=FIRST_HALF_LAST_GW,
                    help=f"last gameweek to plan for (default {FIRST_HALF_LAST_GW}, "
                         f"when the first set expires)")
    ap.add_argument("--no-freehit", action="store_true",
                    help="skip the Free Hit valuation, which runs one MILP "
                         "per gameweek and is the slow part")
    args = ap.parse_args()

    if not os.path.exists(args.xp):
        sys.exit(f"{args.xp} not found — run projections.py first")

    xp = load_xp(args.xp)
    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    meta = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t for t in boot["teams"]}
    gw_now = P.next_gameweek(boot, fixtures)

    covered = sorted({g for d in xp.values() for g in d})
    want = list(range(gw_now, args.through + 1))
    missing = [g for g in want if g not in covered]
    if missing:
        need = args.through - gw_now + 1
        print(f"xp.csv covers GW{min(covered)}-{max(covered)}; planning to "
              f"GW{args.through} needs {need} gameweeks.\n"
              f"Run:\n"
              f"  python projections.py --horizon {need} --deep --out xp.csv\n")
        if len(missing) == len(want):
            sys.exit(1)
        print(f"Continuing with GW{gw_now}-{max(covered)} only.\n")
        want = [g for g in want if g in covered]

    used = chip_state(args.entry)
    first_half_used = {k: v for k, v in used.items()
                       if v and v <= FIRST_HALF_LAST_GW}
    remaining = [c for c in ("3xc", "bboost", "freehit")
                 if c not in first_half_used]

    print("CHIPS")
    for key, label in CHIP_NAMES.items():
        gw = first_half_used.get(key)
        if key == "wildcard":
            note = f"played GW{gw}" if gw else "unused"
        elif gw:
            note = f"played GW{gw}"
        else:
            note = "AVAILABLE"
        print(f"  {label:<16}{note}")
    weeks_left = args.through - gw_now + 1
    print(f"\n{len(remaining)} chip(s) left, {weeks_left} gameweek(s) to "
          f"GW{args.through}. One chip per gameweek.")
    if len(remaining) > weeks_left:
        print("  ! more chips than gameweeks — one of these will expire.")

    if not remaining:
        print("\nNothing left to plan in the first half.")
        return

    # ---- blanks and doubles --------------------------------------------
    counts = fixture_counts(fixtures, teams)
    print("\nFIXTURE SHAPE")
    odd = False
    for gw in want:
        c = counts.get(gw, {})
        blanks = [teams[t]["short_name"] for t, n in c.items() if n == 0]
        doubles = [teams[t]["short_name"] for t, n in c.items() if n >= 2]
        if blanks or doubles:
            odd = True
            bits = []
            if doubles:
                bits.append(f"DOUBLE: {', '.join(sorted(doubles))}")
            if blanks:
                bits.append(f"blank: {', '.join(sorted(blanks))}")
            print(f"  GW{gw}  {' | '.join(bits)}")
    if not odd:
        print(f"  GW{min(want)}-{max(want)}: every club plays exactly once.")
        print("  Blanks and doubles are usually announced only a few weeks "
              "ahead,\n  so re-run this as the cup rounds are scheduled.")

    # ---- squad ----------------------------------------------------------
    import fpl_opt as F
    squad_ids, bank, _ = F.fetch_entry(args.entry, gw_now)
    # fpl_opt works in tenths of a million throughout — p.cost is now_cost
    # as the API gives it, so the budget must be in the same units.
    budget = round(sum(meta[p]["now_cost"] for p in squad_ids if p in meta)
                   + bank * 10)

    p_appear: dict = {}
    for pid in squad_ids:
        e = meta.get(pid)
        if not e:
            continue
        ch = e.get("chance_of_playing_next_round")
        p_appear[pid] = 0.95 if ch is None else ch / 100.0

    # ---- value each chip, each week -------------------------------------
    rows = []
    for gw in want:
        xi, bench, xi_total = best_xi_split(squad_ids, xp, meta, gw)
        entry = {"gw": gw, "xi": xi_total}

        if "3xc" in remaining:
            entry["3xc"] = max((xp.get(p, {}).get(gw, 0.0) for p in xi),
                               default=0.0)

        if "bboost" in remaining:
            raw = sum(xp.get(p, {}).get(gw, 0.0) for p in bench)
            credit = autosub_credit(xi, bench, xp, meta, gw, p_appear)
            entry["bboost"] = max(0.0, raw - credit)
            entry["bb_raw"] = raw

        rows.append(entry)

    if "freehit" in remaining and not args.no_freehit:
        print(f"\nValuing Free Hit against a £{budget/10:.1f}m budget — "
              f"{len(rows)} optimizations...")
        for entry in rows:
            gw = entry["gw"]
            # Rebuild for this gameweek alone. project() is what fills in .ep,
            # and asking for a one-week horizon starting at gw puts the week
            # we care about at index 0, which is what solve_milp optimizes.
            players, pteams, _ = F.load_players(args.xp)
            F.project(players, pteams, gw, 1)
            pool = [p for p in players.values() if p.ep]
            if not pool:
                print(f"  ! GW{gw}: no projections")
                continue
            try:
                squad, _ = F.solve_milp(pool, budget, current=None,
                                        free_transfers=99, horizon=1)
                # best_xi returns (score, xi, captain) and its score already
                # includes the captain bonus. entry["xi"] does not, so compare
                # the raw XI sums or the chip looks a captain's worth better
                # than it is.
                _, fh_xi, _ = F.best_xi(squad, 0)
                entry["freehit"] = max(0.0,
                                       sum(p.ep[0] for p in fh_xi)
                                       - entry["xi"])
                print(f"  GW{gw}: +{entry['freehit']:.1f}")
            except Exception as e:
                print(f"  ! GW{gw}: {type(e).__name__}: {e}")

    # ---- report ----------------------------------------------------------
    cols = [c for c in ("3xc", "bboost", "freehit") if c in remaining]
    print(f"\nINCREMENTAL VALUE — what each chip ADDS, not what it scores\n")
    hdr = f"{'GW':>4}{'YOUR XI':>9}"
    for c in cols:
        hdr += f"{CHIP_NAMES[c]:>15}"
    print(hdr)
    print("-" * len(hdr))
    for e in rows:
        line = f"{e['gw']:>4}{e['xi']:>9.1f}"
        for c in cols:
            v = e.get(c)
            line += f"{v:>15.1f}" if v is not None else f"{'—':>15}"
        print(line)

    print("\nBEST WEEK FOR EACH")
    plan = {}
    for c in cols:
        cand = [(e.get(c, -1), e["gw"]) for e in rows if e.get(c) is not None]
        if not cand:
            continue
        v, gw = max(cand)
        plan[c] = (gw, v)
        print(f"  {CHIP_NAMES[c]:<16}GW{gw:<4}+{v:.1f}")

    # one chip per gameweek
    weeks = [gw for gw, _ in plan.values()]
    if len(weeks) != len(set(weeks)):
        print("\n  ! two chips want the same gameweek. Only one can be "
              "played;\n    give that week to whichever gains more and take "
              "the next-best\n    week for the other.")

    if "bboost" in plan:
        gw, v = plan["bboost"]
        raw = next(e["bb_raw"] for e in rows if e["gw"] == gw)
        print(f"\nBench Boost at GW{gw}: bench projects {raw:.1f}, of which "
              f"{raw - v:.1f} the\nautomatic substitutions would have given "
              f"you anyway. A bench slot that\nnever plays contributes "
              f"nothing — check for a non-playing second keeper.")

    print(f"\nThese are projections, and a blank or double not yet scheduled "
          f"would\nchange all of it. Re-run weekly. GW{args.through} is the "
          f"wall: an unplayed\nchip is lost, so a mediocre week beats no week "
          f"at all.")


if __name__ == "__main__":
    main()
