#!/usr/bin/env python3
"""
draft.py — FPL Draft companion.

Reuses the classic pipeline's projections (`xp.csv`) but solves a different
problem. Draft has no budget, no prices, no 3-per-club limit and no captain.
The weekly decisions are:

  1. which XI to field from your 15
  2. which free agent to claim, and who to drop for him
  3. how to play a head-to-head matchup you're favoured or not favoured to win

That third one is why this isn't just fpl_opt with constraints removed. In a
points league you maximise expected points. In H2H you're beating ONE opponent,
and 3 points for a win means a 40-point loss and a 1-point loss cost the same.
When you're the underdog, variance is your friend — a boom-or-bust forward with
lower xP can raise P(win) above a steady midfielder with higher xP. When you're
favoured, the reverse: take the safe floor and protect the lead.

Usage
-----
    python draft.py lineup  --entry 274930 --xp xp.csv
    python draft.py waivers --entry 274930 --xp xp.csv --top 15
    python draft.py matchup --entry 274930 --xp xp.csv

Element ids mostly match the classic game but not always, so ids are remapped
by name, team and position where they diverge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import projections as P

DRAFT_API = "https://draft.premierleague.com/api"

SQUAD_LIMITS = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
FORMATIONS = [(d, m, f)
              for d in range(XI_MIN[2], XI_MAX[2] + 1)
              for m in range(XI_MIN[3], XI_MAX[3] + 1)
              for f in range(XI_MIN[4], XI_MAX[4] + 1)
              if d + m + f == 10]


def dget(path: str, fname: str, ttl: int = 600):
    """Draft lives on its own host with its own cache namespace."""
    import time
    import requests
    os.makedirs(P.CACHE_DIR, exist_ok=True)
    cache = os.path.join(P.CACHE_DIR, f"draft_{fname}")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < ttl:
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    r = requests.get(f"{DRAFT_API}/{path}", timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


# ---------------------------------------------------------------------------
# id mapping
# ---------------------------------------------------------------------------
def build_id_map(draft_boot: dict, classic_boot: dict) -> dict[int, int]:
    """draft element id -> classic element id.

    Most ids coincide, but not all, and a silent mismatch would attribute one
    player's projection to another. Match on identity (name + team + position)
    rather than trusting the id, and fall back only when that's unambiguous.
    """
    def key(e, teams):
        return (P.strip_accents(e["web_name"]),
                teams[e["team"]]["short_name"], e["element_type"])

    d_teams = {t["id"]: t for t in draft_boot["teams"]}
    c_teams = {t["id"]: t for t in classic_boot["teams"]}
    classic_by_key: dict = {}
    for e in classic_boot["elements"]:
        classic_by_key.setdefault(key(e, c_teams), e["id"])

    out, unmatched = {}, []
    for e in draft_boot["elements"]:
        k = key(e, d_teams)
        if k in classic_by_key:
            out[e["id"]] = classic_by_key[k]
        else:
            # last resort: same name and position, any club (recent transfer)
            cands = {cid for ck, cid in classic_by_key.items()
                     if ck[0] == k[0] and ck[2] == k[2]}
            if len(cands) == 1:
                out[e["id"]] = cands.pop()
            else:
                unmatched.append(e["web_name"])
    if unmatched:
        print(f"  {len(unmatched)} draft players had no classic match "
              f"(no projection): {', '.join(unmatched[:8])}"
              + (" ..." if len(unmatched) > 8 else ""))
    return out


DECAY = 0.88  # per-gameweek discount when valuing a player you'll keep


def load_xp(path: str, gw: int) -> dict[int, float]:
    """This gameweek only — the right basis for setting an XI."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["gw"]) == gw:
                out[int(row["id"])] = float(row["ep"])
    return out


def load_xp_horizon(path: str, gw: int, horizon: int) -> dict[int, float]:
    """Decayed total over the horizon — the right basis for an add/drop.

    Transactions are unlimited in Draft, so the cost of a move isn't a
    transfer, it's permanent: you keep whoever you claim and the player you
    drop becomes available to everyone else. Judging that on one gameweek's
    projection values an asset you'll hold for months by what it does on
    Saturday.
    """
    out: dict[int, float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = int(row["gw"])
            if gw <= g < gw + horizon:
                out[int(row["id"])] = out.get(int(row["id"]), 0.0) + \
                    float(row["ep"]) * DECAY ** (g - gw)
    return out


# ---------------------------------------------------------------------------
# variance
# ---------------------------------------------------------------------------
# Levels anchored on measured GW1-2 spread of actual points for 60+ minute
# appearances: GKP 2.73, DEF 3.47, MID 3.57, FWD 3.19. The previous hand-fit
# values ran 31% high for defenders and over 100% high for forwards, which
# inflated team variance and dragged every head-to-head P(win) toward 50% —
# precisely blunting the favoured/underdog advice this exists to give.
#
# The SLOPE is still unfitted. Separating "how does spread grow with expected
# points" from the pooled level needs projections paired with outcomes, which
# only `*_scored.csv` provides. Kept deliberately shallow so the data-anchored
# level dominates until there's enough to fit it.
# Re-check with: python backtest.py calibrate-variance
SD_BASE = {1: 1.95, 2: 2.28, 3: 1.85, 4: 1.35}
SD_SLOPE = {1: 0.25, 2: 0.35, 3: 0.40, 4: 0.45}


def player_sd(xp: float, pos: int) -> float:
    """Standard deviation of a player's gameweek score.

    FPL points are a near-certain appearance floor plus lumpy returns, so
    spread grows with expected points but far less than proportionally.
    """
    return SD_BASE[pos] + SD_SLOPE[pos] * max(0.0, xp)


def win_probability(my_xp: float, my_var: float,
                    opp_xp: float, opp_var: float) -> float:
    """P(I outscore my opponent), normal approximation.

    Summing eleven independent-ish players makes the team total roughly normal
    by the CLT, which is good enough for ranking decisions even though single
    players are anything but.
    """
    sd = math.sqrt(max(my_var + opp_var, 1e-9))
    z = (my_xp - opp_xp) / sd
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# lineup
# ---------------------------------------------------------------------------
def best_xi(squad: list[dict]):
    """Highest-xP legal XI. No captain in Draft, so this is a pure sort."""
    by_pos = {k: sorted([p for p in squad if p["pos"] == k],
                        key=lambda p: -p["xp"]) for k in P.POS}
    best = (-1.0, None)
    for d, m, f in FORMATIONS:
        if len(by_pos[2]) < d or len(by_pos[3]) < m or len(by_pos[4]) < f \
                or not by_pos[1]:
            continue
        xi = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        tot = sum(p["xp"] for p in xi)
        if tot > best[0]:
            best = (tot, xi)
    return best


def fetch_picks(entry: int, gw: int):
    """Picks only exist for gameweeks whose deadline has passed, so walk back
    from the target until one resolves — asking for the upcoming gameweek 404s.
    """
    last_err = None
    for g in range(gw, 0, -1):
        for tmpl in (f"entry/{entry}/event/{g}",
                     f"entry/{entry}/event/{g}/picks",
                     f"draft/entry/{entry}/event/{g}"):
            try:
                data = dget(tmpl, f"picks_{entry}_{g}.json", ttl=300)
            except Exception as exc:
                last_err = exc
                continue
            if data.get("picks"):
                if g != gw:
                    print(f"  (GW{gw} picks not published yet — using your "
                          f"GW{g} squad)")
                return data
    raise SystemExit(f"Could not fetch picks for entry {entry}: {last_err}\n"
                     f"Run: python draft_probe.py --entry {entry} --gw {max(gw-1,1)}")


def resolve_names(elements, tokens):
    """Turn names or ids into draft elements, refusing ambiguous names.

    Same rule as elsewhere: a name that matches more than one player is
    refused rather than guessed at — there are two Fofanas and two Richardses
    in this game, and binding to the wrong one is silent.
    """
    out = []
    by_id = {e["id"]: e for e in elements}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            e = by_id.get(int(tok))
            if not e:
                sys.exit(f"id {tok} is not a player")
            out.append(e)
            continue
        n = P.strip_accents(tok)
        exact = [e for e in elements if P.strip_accents(e["web_name"]) == n]
        hits = exact or [e for e in elements
                         if len(n) >= 4 and n in P.strip_accents(e["web_name"])]
        if not hits:
            sys.exit(f"no player matching {tok!r}")
        if len(hits) > 1:
            opts = ", ".join(f"{e['web_name']} (id {e['id']})" for e in hits[:8])
            sys.exit(f"{tok!r} is ambiguous — {opts}. Use an id.")
        out.append(hits[0])
    return out


def squad_rows(entry: int, gw: int, idmap, xp, draft_boot,
               add=None, drop=None, squad_file=None):
    """Your 15, with pending moves applied.

    Draft publishes picks only once a deadline has passed, so between making a
    claim and the deadline the API still returns last week's squad — the same
    problem `mysquad.py` solves for classic. --add/--drop patch the fetched
    squad so lineups and matchups reflect what you actually own.
    """
    d_teams = {t["id"]: t for t in draft_boot["teams"]}
    by_id = {e["id"]: e for e in draft_boot["elements"]}

    if squad_file:
        with open(squad_file, encoding="utf-8") as f:
            raw = json.load(f)
        ids = raw["squad"] if isinstance(raw, dict) else raw
        return [make_row(by_id[i], d_teams, idmap, xp) for i in ids
                if i in by_id]

    picks = fetch_picks(entry, gw)
    ids = [pk["element"] for pk in picks.get("picks", []) if pk["element"] in by_id]

    outs = resolve_names(draft_boot["elements"],
                         (drop or "").split(",")) if drop else []
    ins = resolve_names(draft_boot["elements"],
                        (add or "").split(",")) if add else []
    if outs or ins:
        if len(outs) != len(ins):
            sys.exit(f"{len(outs)} dropped but {len(ins)} added — a draft "
                     f"transaction is a swap, so these must match.")
        for e in outs:
            if e["id"] not in ids:
                sys.exit(f"{e['web_name']} isn't in your published squad — "
                         f"already applied?")
            ids.remove(e["id"])
        for e in ins:
            if e["id"] in ids:
                sys.exit(f"{e['web_name']} is already in the squad")
            ids.append(e["id"])
        for e in outs:
            print(f"  DROP {e['web_name']}")
        for e in ins:
            print(f"  ADD  {e['web_name']}")
        if len(ids) != 15:
            print(f"  ! squad is now {len(ids)} players, not 15")

    return [make_row(by_id[i], d_teams, idmap, xp) for i in ids]


def make_row(e, d_teams, idmap, xp):
    cid = idmap.get(e["id"])
    return {"id": e["id"], "cid": cid, "name": e["web_name"],
            "pos": e["element_type"], "team": d_teams[e["team"]]["short_name"],
            "xp": xp.get(cid, 0.0) if cid else 0.0,
            "status": e.get("status", "a"), "news": (e.get("news") or "").strip()}


def show_xi(rows):
    tot, xi = best_xi(rows)
    if not xi:
        print("Could not build a legal XI — check the squad loaded correctly.")
        return 0.0, [], []
    bench = sorted([p for p in rows if p not in xi],
                   key=lambda p: (p["pos"] != 1, -p["xp"]))
    d = sum(1 for p in xi if p["pos"] == 2)
    m = sum(1 for p in xi if p["pos"] == 3)
    f = sum(1 for p in xi if p["pos"] == 4)
    print(f"\n{'STARTING XI':<24}{'CLUB':>6}{'POS':>6}{'xP':>8}   ({d}-{m}-{f})")
    print("-" * 52)
    for p in sorted(xi, key=lambda p: (p["pos"], -p["xp"])):
        flag = "  !" if p["status"] != "a" else ""
        print(f"{p['name']:<24}{p['team']:>6}{P.POS[p['pos']]:>6}"
              f"{p['xp']:>8.2f}{flag}")
    print("-" * 52)
    print("BENCH")
    for p in bench:
        flag = "  !" if p["status"] != "a" else ""
        print(f"{p['name']:<24}{p['team']:>6}{P.POS[p['pos']]:>6}"
              f"{p['xp']:>8.2f}{flag}")
    var = sum(player_sd(p["xp"], p["pos"]) ** 2 for p in xi)
    print(f"\nProjected {tot:.1f} points  (sd {math.sqrt(var):.1f})")
    flagged = [p for p in xi if p["news"]]
    if flagged:
        print("\nNews on your starters:")
        for p in flagged:
            print(f"  {p['name']}: {p['news']}")
    return tot, xi, bench


# ---------------------------------------------------------------------------
# waivers
# ---------------------------------------------------------------------------
def dead_slots(mine, horizon_xp):
    """Squad slots returning nothing.

    The XI-impact ranking can't surface these: a long-term injury sitting on
    your bench doesn't change the eleven you field, so replacing him shows a
    gain of zero. But he's a free roster spot, and in a format with unlimited
    transactions that's the cheapest move available.
    """
    out = []
    for p in mine:
        why = None
        if p["status"] in ("i", "s", "u", "n"):
            why = f"status '{p['status']}'" + (f" — {p['news']}" if p["news"] else "")
        elif horizon_xp.get(p["cid"], 0.0) < 1.5:
            why = f"{horizon_xp.get(p['cid'], 0.0):.1f} xP over the horizon"
        if why:
            out.append((p, why))
    return out


def waivers(entry, gw, idmap, xp, draft_boot, league_id, top,
            horizon_xp=None, horizon=5, patch=None):
    d_teams = {t["id"]: t for t in draft_boot["teams"]}
    by_id = {e["id"]: e for e in draft_boot["elements"]}

    try:
        status = dget(f"league/{league_id}/element-status",
                      f"status_{league_id}.json", ttl=300)
        owned = {s["element"] for s in status["element_status"]
                 if s.get("owner") is not None}
    except Exception as exc:
        print(f"Couldn't read ownership ({exc}) — cannot tell free agents from "
              f"owned players. Waiver list unavailable.")
        return

    mine = squad_rows(entry, gw, idmap, xp, draft_boot, **(patch or {}))
    owned = owned | {p["id"] for p in mine}   # a pending claim isn't a free agent
    free = [make_row(by_id[e], d_teams, idmap, xp) for e in by_id
            if e not in owned and by_id[e].get("status") == "a"]

    base_tot, base_xi = best_xi(mine)
    print(f"\n{len(free)} free agents available. "
          f"Your XI currently projects {base_tot:.1f} this week.")

    dead = dead_slots(mine, horizon_xp or {})
    if dead:
        print(f"\nDEAD SQUAD SLOTS — returning nothing, replace them free")
        for p, why in dead:
            print(f"  {p['name']:<20}{P.POS[p['pos']]:>5}  {why}")
        print("  ^ not in the ranking below — a benched player scores zero gain.")

    # What a claim is worth is its effect on the ELEVEN YOU FIELD, not on the
    # worst player in your squad. Upgrading a bench defender who never starts
    # changes nothing; comparing against him inflates every gain. So evaluate
    # each (add, drop) pair by re-solving the XI.
    moves = []
    for cand in sorted(free, key=lambda p: -p["xp"])[:120]:
        for drop in mine:
            if drop["pos"] != cand["pos"]:
                continue
            trial = [p for p in mine if p is not drop] + [cand]
            tot, _ = best_xi(trial)
            if tot > base_tot + 1e-9:
                moves.append((tot - base_tot, cand, drop, tot))
    if not moves:
        print("\nNo available player improves your starting XI. Hold the pick.")
        return

    # keep the best drop for each incoming player
    best_by_cand = {}
    for gain, cand, drop, tot in moves:
        cur = best_by_cand.get(cand["id"])
        if cur is None or gain > cur[0]:
            best_by_cand[cand["id"]] = (gain, cand, drop, tot)
    ranked = sorted(best_by_cand.values(), key=lambda m: -m[0])[:top]

    hz = horizon_xp or {}
    print(f"\n{'CLAIM':<19}{'POS':>4}{'GW':>6}{'HORIZON':>9}   "
          f"{'DROP':<17}{'GW GAIN':>8}{'HZ GAIN':>9}")
    print("-" * 74)
    for gain, cand, drop, tot in ranked:
        hz_gain = hz.get(cand["cid"], 0.0) - hz.get(drop["cid"], 0.0)
        print(f"{cand['name']:<19}{P.POS[cand['pos']]:>4}{cand['xp']:>6.2f}"
              f"{hz.get(cand['cid'], 0.0):>9.1f}   {drop['name']:<17}"
              f"{gain:>+8.2f}{hz_gain:>+9.1f}")

    print("\nHZ GAIN is the one that matters. Transactions are free.")


# ---------------------------------------------------------------------------
# matchup
# ---------------------------------------------------------------------------
def matchup(entry, gw, idmap, xp, draft_boot, league_id, patch=None):
    try:
        details = dget(f"league/{league_id}/details",
                       f"league_{league_id}.json", ttl=600)
    except Exception as exc:
        print(f"Couldn't read league details ({exc}).")
        return

    entries = {le["entry_id"]: le for le in details.get("league_entries", [])
               if le.get("entry_id")}
    fixture = None
    for mt in details.get("matches", []):
        if mt.get("event") != gw:
            continue
        if mt.get("league_entry_1") in {le["id"] for le in
                                        details["league_entries"]
                                        if le.get("entry_id") == entry} or \
           mt.get("league_entry_2") in {le["id"] for le in
                                        details["league_entries"]
                                        if le.get("entry_id") == entry}:
            fixture = mt
            break
    if not fixture:
        print(f"No GW{gw} fixture found for entry {entry}.")
        return

    le_by_id = {le["id"]: le for le in details["league_entries"]}
    mine_le = next(le["id"] for le in details["league_entries"]
                   if le.get("entry_id") == entry)
    opp_le = (fixture["league_entry_2"] if fixture["league_entry_1"] == mine_le
              else fixture["league_entry_1"])
    opp_entry = le_by_id[opp_le].get("entry_id")
    opp_name = le_by_id[opp_le].get("entry_name", "opponent")

    my_rows = squad_rows(entry, gw, idmap, xp, draft_boot, **(patch or {}))
    my_tot, my_xi = best_xi(my_rows)
    my_var = sum(player_sd(p["xp"], p["pos"]) ** 2 for p in my_xi)

    if not opp_entry:
        print(f"Opponent {opp_name} has no public entry id — cannot project "
              f"their side.")
        return
    opp_rows = squad_rows(opp_entry, gw, idmap, xp, draft_boot)
    opp_tot, opp_xi = best_xi(opp_rows)
    opp_var = sum(player_sd(p["xp"], p["pos"]) ** 2 for p in opp_xi)

    pw = win_probability(my_tot, my_var, opp_tot, opp_var)
    print(f"\n{'='*58}")
    print(f"GW{gw}: you {my_tot:.1f} (sd {math.sqrt(my_var):.1f})  vs  "
          f"{opp_name} {opp_tot:.1f} (sd {math.sqrt(opp_var):.1f})")
    print(f"P(win) {pw:.0%}   —   margin {my_tot-opp_tot:+.1f}")
    print("=" * 58)

    if pw >= 0.60:
        print("FAVOURED. Take the safe floor: prefer nailed starters over "
              "boom-or-bust picks, and don't chase upside you don't need.\n"
              "A 40-point win and a 1-point win are both worth 3.")
    elif pw <= 0.40:
        print("UNDERDOG. Expected points are the wrong objective here — you "
              "need variance.\nPrefer the volatile option even at slightly "
              "lower xP: a differential forward over a steady defender, an "
              "attacking returner over appearance points.\nLosing by 30 costs "
              "exactly what losing by 1 does.")
    else:
        print("COIN FLIP. Just field your highest-xP legal XI; neither floor "
              "nor ceiling is worth distorting for.")

    # which bench swaps would most raise P(win), as opposed to xP
    bench = [p for p in my_rows if p not in my_xi]
    swaps = []
    for out_p in my_xi:
        for in_p in bench:
            if in_p["pos"] != out_p["pos"] or in_p["xp"] <= 0:
                continue
            t = my_tot - out_p["xp"] + in_p["xp"]
            v = (my_var - player_sd(out_p["xp"], out_p["pos"]) ** 2
                 + player_sd(in_p["xp"], in_p["pos"]) ** 2)
            new_pw = win_probability(t, v, opp_tot, opp_var)
            if new_pw > pw + 0.005:
                swaps.append((new_pw - pw, in_p, out_p, t, new_pw))
    if swaps:
        swaps.sort(key=lambda s: -s[0])
        print("\nSwaps that raise P(win) despite lowering expected points:")
        for delta, in_p, out_p, t, new_pw in swaps[:4]:
            print(f"  {in_p['name']} for {out_p['name']}: "
                  f"xP {t:.1f} ({t-my_tot:+.1f}) but P(win) {new_pw:.0%} "
                  f"({delta*100:+.1f}pp)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lineup", "waivers", "matchup"):
        s = sub.add_parser(name)
        s.add_argument("--entry", type=int, required=True)
        s.add_argument("--xp", default="xp.csv")
        s.add_argument("--gw", type=int)
        s.add_argument("--league", type=int, help="league id (auto-detected)")
        s.add_argument("--add", default="", metavar="NAMES",
                       help="players claimed since the last deadline "
                            "(names or ids, comma-separated)")
        s.add_argument("--drop", default="", metavar="NAMES",
                       help="players dropped since the last deadline")
        s.add_argument("--squad", help="JSON file of 15 draft element ids, "
                                       "instead of fetching")
        if name == "waivers":
            s.add_argument("--top", type=int, default=8)
            s.add_argument("--horizon", type=int, default=5,
                           help="gameweeks to value a claim over")
    args = ap.parse_args()

    draft_boot = dget("bootstrap-static", "bootstrap.json")
    classic_boot = P._get("bootstrap-static/", "bootstrap.json")

    gw = args.gw
    if gw is None:
        gw = next((ev["id"] for ev in classic_boot["events"]
                   if ev.get("is_next")), 1)

    idmap = build_id_map(draft_boot, classic_boot)
    xp = load_xp(args.xp, gw)
    print(f"GW{gw}: {len(xp)} projections loaded, "
          f"{len(idmap)} draft ids mapped.")

    league_id = args.league
    if league_id is None:
        try:
            pub = dget(f"entry/{args.entry}/public", f"pub_{args.entry}.json")
            # the payload shape isn't documented and has changed before, so
            # look anywhere a league id might plausibly live
            cands = []
            for key in ("league_set", "leagues"):
                v = pub.get(key)
                if isinstance(v, list):
                    cands += [x if isinstance(x, int) else x.get("id")
                              for x in v if isinstance(x, (int, dict))]
            entry_blk = pub.get("entry")
            if isinstance(entry_blk, dict):
                for key in ("league_set", "leagues", "league"):
                    v = entry_blk.get(key)
                    if isinstance(v, list):
                        cands += [x if isinstance(x, int) else x.get("id")
                                  for x in v if isinstance(x, (int, dict))]
                    elif isinstance(v, int):
                        cands.append(v)
            cands = [c for c in cands if c]
            league_id = cands[0] if cands else None
            if not league_id:
                print(f"(no league id in entry/{args.entry}/public — keys were: "
                      f"{list(pub)[:10]})")
        except Exception as exc:
            print(f"(couldn't auto-detect league: {exc})")

    patch = dict(add=args.add, drop=args.drop, squad_file=args.squad)
    if args.cmd == "lineup":
        rows = squad_rows(args.entry, gw, idmap, xp, draft_boot, **patch)
        show_xi(rows)
    elif args.cmd == "waivers":
        if not league_id:
            sys.exit("Need --league for waivers. Find it in the Draft URL "
                     "(/league/<ID>/standings), or run draft_probe.py.")
        hz = load_xp_horizon(args.xp, gw, args.horizon)
        waivers(args.entry, gw, idmap, xp, draft_boot, league_id, args.top,
                hz, args.horizon, patch)
    else:
        if not league_id:
            sys.exit("Need --league for matchup. Find it in the Draft URL "
                     "(/league/<ID>/standings), or run draft_probe.py.")
        matchup(args.entry, gw, idmap, xp, draft_boot, league_id, patch)


if __name__ == "__main__":
    main()
