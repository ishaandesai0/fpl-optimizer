#!/usr/bin/env python3
"""
fpl_opt.py — Fantasy Premier League squad optimizer.

Pulls live data from the public FPL API, projects points, and solves for the
optimal 15-man squad (or the optimal transfers from an existing squad) under
FPL's constraints:

    2 GK / 5 DEF / 5 MID / 3 FWD, <= 3 players per club, <= budget.

Objective = expected points of the starting XI + captain (doubled)
          + BENCH_WEIGHT * bench, summed over the projection horizon
          with geometric decay, minus 4 pts per transfer beyond the free ones.

Usage
-----
    pip install requests pulp

    # Fresh squad (wildcard / GW1)
    python fpl_opt.py --budget 100.0 --horizon 3

    # Weekly run: best moves from your current squad
    python fpl_opt.py --entry 1234567 --free-transfers 1

    # Plug in your own projections instead of FPL's
    python fpl_opt.py --projections my_ep.csv     # cols: id,ep  (or web_name,ep)

    # Save the resulting squad for next week
    python fpl_opt.py --horizon 3 --save my_squad.json

Notes
-----
* No auth needed; everything used here is public.
* Falls back to a local-search heuristic if PuLP isn't installed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://fantasy.premierleague.com/api"
CACHE_DIR = os.path.expanduser("~/.fpl_cache")
CACHE_TTL = 900  # seconds

POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_LIMITS = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
BENCH_WEIGHT = 0.12  # bench points are worth ~12% of a starter's
DECAY = 0.84  # each future gameweek is discounted by this factor
TRANSFER_HIT = 4.0


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _get(path: str, fname: str) -> dict | list:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, fname)
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < CACHE_TTL:
        with open(cache) as f:
            return json.load(f)
    r = requests.get(f"{API}/{path}", timeout=20,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    data = r.json()
    with open(cache, "w") as f:
        json.dump(data, f)
    return data


@dataclass
class Player:
    id: int
    name: str
    pos: int
    team: int
    team_name: str
    cost: int          # tenths of a million
    avail: float       # 0..1 probability of being available
    base_ep: float     # per-match expected points before fixture adjustment
    news: str
    selected: float
    ep: list[float] = field(default_factory=list)  # per-gameweek projection
    fixed_ep: dict = field(default_factory=dict)   # gw -> xP, from --projections

    @property
    def total_ep(self) -> float:
        return sum(e * DECAY**i for i, e in enumerate(self.ep))

    @property
    def gw1_ep(self) -> float:
        return self.ep[0] if self.ep else 0.0


def availability(e: dict) -> float:
    status = e.get("status", "a")
    if status == "a":
        return 1.0
    chance = e.get("chance_of_playing_next_round")
    if chance is not None:
        return chance / 100.0
    return {"d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}.get(status, 0.0)


def base_ep(e: dict) -> float:
    """FPL's own expected points, with sane fallbacks in preseason."""
    for key in ("ep_next", "ep_this"):
        try:
            v = float(e.get(key) or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            return v
    # fallback: last season's rate, dampened
    try:
        ppg = float(e.get("points_per_game") or 0)
    except (TypeError, ValueError):
        ppg = 0.0
    if ppg > 0:
        return ppg * 0.9
    # last resort: price is a decent prior for how good FPL thinks a player is
    return max(0.0, (e["now_cost"] - 38) / 10.0)


def fetch_entry(entry_id: int, next_gw: int) -> tuple[list[int], float, dict]:
    """Pull a manager's current 15 and bank from the public entry endpoints.

    Picks are only published once a gameweek's deadline has passed, so we walk
    backwards from the most recent gameweek until we find one that exists.
    """
    summary = _get(f"entry/{entry_id}/", f"entry_{entry_id}.json")
    bank = (summary.get("last_deadline_bank") or 0) / 10.0

    for gw in range(next_gw - 1, 0, -1):
        try:
            picks = _get(f"entry/{entry_id}/event/{gw}/picks/",
                         f"picks_{entry_id}_{gw}.json")
        except requests.HTTPError:
            continue
        ids = [p["element"] for p in picks["picks"]]
        # FPL returns only half your profit on a sale, rounded down to £0.1m.
        # Valuing the squad at now_cost invents budget that doesn't exist.
        sell = {p["element"]: p.get("selling_price")
                for p in picks["picks"] if p.get("selling_price")}
        if len(ids) == 15:
            print(f"Loaded squad from GW{gw} "
                  f"({summary.get('player_first_name', '')} "
                  f"{summary.get('player_last_name', '')}, "
                  f"team '{summary.get('name', '')}').")
            return ids, bank, sell
    raise SystemExit(f"No published picks found for entry {entry_id}. "
                     "Picks appear only after a deadline has passed.")


def load_players(proj_file: str | None) -> tuple[dict[int, Player], dict[int, dict], int]:
    boot = _get("bootstrap-static/", "bootstrap.json")
    teams = {t["id"]: t for t in boot["teams"]}

    next_gw = None
    for ev in boot["events"]:
        if ev.get("is_next"):
            next_gw = ev["id"]
    if next_gw is None:
        for ev in boot["events"]:
            if not ev.get("finished"):
                next_gw = ev["id"]
                break
    next_gw = next_gw or 1

    override: dict[int, float] = {}
    by_name: dict[str, float] = {}
    per_gw: dict[int, dict[int, float]] = {}
    if proj_file:
        with open(proj_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = float(row.get("ep") or row.get("xP") or 0)
                if row.get("gw") and row.get("id"):
                    per_gw.setdefault(int(row["id"]), {})[int(row["gw"])] = val
                elif row.get("id"):
                    override[int(row["id"])] = val
                elif row.get("web_name"):
                    by_name[row["web_name"].strip().lower()] = val

    players: dict[int, Player] = {}
    for e in boot["elements"]:
        t = teams[e["team"]]
        p = Player(
            id=e["id"],
            name=e["web_name"],
            pos=e["element_type"],
            team=e["team"],
            team_name=t["short_name"],
            cost=e["now_cost"],
            avail=availability(e),
            base_ep=base_ep(e),
            news=(e.get("news") or "").strip(),
            selected=float(e.get("selected_by_percent") or 0),
        )
        if p.id in override:
            p.base_ep = override[p.id]
        elif p.name.lower() in by_name:
            p.base_ep = by_name[p.name.lower()]
        if p.id in per_gw:
            p.fixed_ep = per_gw[p.id]
        players[p.id] = p
    return players, teams, next_gw


def project(players: dict[int, Player], teams: dict, start_gw: int, horizon: int) -> None:
    """Spread base_ep across the horizon, adjusting for fixture difficulty.

    Handles double and blank gameweeks: a team playing twice gets two fixture
    contributions in that week, a team with no fixture gets zero.
    """
    fixtures = _get("fixtures/", "fixtures.json")
    gws = list(range(start_gw, start_gw + horizon))

    # team -> gw -> list of difficulty ratings (1 easy .. 5 hard)
    diff: dict[int, dict[int, list[int]]] = {tid: {g: [] for g in gws} for tid in teams}
    for f in fixtures:
        g = f.get("event")
        if g not in diff.get(f["team_h"], {}):
            continue
        diff[f["team_h"]][g].append(f.get("team_h_difficulty") or 3)
        diff[f["team_a"]][g].append(f.get("team_a_difficulty") or 3)

    n_fixed = 0
    for p in players.values():
        if p.fixed_ep:
            # already fixture- and minutes-adjusted upstream; use as-is
            p.ep = [p.fixed_ep.get(g, 0.0) for g in gws]
            n_fixed += 1
            continue
        p.ep = []
        for g in gws:
            week = 0.0
            for d in diff[p.team][g]:
                # difficulty 3 is neutral; each step is worth ~9%
                week += p.base_ep * (1.0 + (3 - d) * 0.09)
            p.ep.append(week * p.avail)
    if n_fixed:
        print(f"Using external per-gameweek projections for {n_fixed} players "
              f"(FDR scaling bypassed).")


# --------------------------------------------------------------------------
# squad valuation
# --------------------------------------------------------------------------
FORMATIONS = [
    (d, m, f)
    for d in range(XI_MIN[2], XI_MAX[2] + 1)
    for m in range(XI_MIN[3], XI_MAX[3] + 1)
    for f in range(XI_MIN[4], XI_MAX[4] + 1)
    if d + m + f == 10
]


def best_xi(squad: list[Player], gw_index: int = 0):
    """Pick the highest-scoring legal XI + captain from a 15. Returns (score, xi, cap)."""
    by_pos = {k: sorted([p for p in squad if p.pos == k],
                        key=lambda p: -p.ep[gw_index]) for k in POS}
    best = (-1.0, None, None)
    for d, m, f in FORMATIONS:
        if len(by_pos[2]) < d or len(by_pos[3]) < m or len(by_pos[4]) < f or not by_pos[1]:
            continue
        xi = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        cap = max(xi, key=lambda p: p.ep[gw_index])
        score = sum(p.ep[gw_index] for p in xi) + cap.ep[gw_index]
        if score > best[0]:
            best = (score, xi, cap)
    return best


def squad_value(squad: list[Player]) -> float:
    """Total objective: XI + captain each week, decayed, plus bench weight."""
    total = 0.0
    for i in range(len(squad[0].ep)):
        score, xi, _ = best_xi(squad, i)
        bench = sum(p.ep[i] for p in squad if p not in xi)
        total += (score + BENCH_WEIGHT * bench) * DECAY**i
    return total


# --------------------------------------------------------------------------
# optimizer: MILP
# --------------------------------------------------------------------------
def solve_milp(pool, budget, current=None, free_transfers=1, locked=(), banned=(),
               allow_hits=False, horizon=None):
    """Exact MILP. Picks the 15, then picks a fresh XI + captain for every
    gameweek in the horizon, so future fixtures are valued the way they'll
    actually be used rather than through a flat fudge factor."""
    import pulp

    cur = set(current or [])
    n_gw = min(horizon or len(pool[0].ep), len(pool[0].ep))
    gws = range(n_gw)
    prob = pulp.LpProblem("fpl", pulp.LpMaximize)

    pick = {p.id: pulp.LpVariable(f"s{p.id}", cat="Binary") for p in pool}
    start = {(p.id, g): pulp.LpVariable(f"x{p.id}_{g}", cat="Binary")
             for p in pool for g in gws}
    capt = {(p.id, g): pulp.LpVariable(f"c{p.id}_{g}", cat="Binary")
            for p in pool for g in gws}

    obj = pulp.lpSum(
        DECAY**g * (
            (start[p.id, g] + capt[p.id, g]) * p.ep[g]
            + BENCH_WEIGHT * (pick[p.id] - start[p.id, g]) * p.ep[g]
        )
        for p in pool for g in gws
    )

    hits = None
    if cur:
        bought = pulp.lpSum(pick[p.id] for p in pool if p.id not in cur)
        if allow_hits:
            hits = pulp.LpVariable("hits", lowBound=0)
            prob += hits >= bought - free_transfers
            obj -= TRANSFER_HIT * hits
        else:
            prob += bought <= free_transfers

    prob += obj

    prob += pulp.lpSum(pick.values()) == 15
    prob += pulp.lpSum(p.cost * pick[p.id] for p in pool) <= budget

    for pos, n in SQUAD_LIMITS.items():
        prob += pulp.lpSum(pick[p.id] for p in pool if p.pos == pos) == n

    for t in {p.team for p in pool}:
        prob += pulp.lpSum(pick[p.id] for p in pool if p.team == t) <= MAX_PER_CLUB

    for g in gws:
        prob += pulp.lpSum(start[p.id, g] for p in pool) == 11
        prob += pulp.lpSum(capt[p.id, g] for p in pool) == 1
        for pos in POS:
            in_pos = pulp.lpSum(start[p.id, g] for p in pool if p.pos == pos)
            prob += in_pos >= XI_MIN[pos]
            prob += in_pos <= XI_MAX[pos]
        for p in pool:
            prob += start[p.id, g] <= pick[p.id]
            prob += capt[p.id, g] <= start[p.id, g]

    for pid in locked:
        if pid in pick:
            prob += pick[pid] == 1
    for pid in banned:
        if pid in pick:
            prob += pick[pid] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"solver status: {pulp.LpStatus[prob.status]}")
    squad = [p for p in pool if pick[p.id].value() > 0.5]
    return squad, pulp.value(prob.objective)


# --------------------------------------------------------------------------
# optimizer: heuristic fallback (no external solver needed)
# --------------------------------------------------------------------------
def solve_local(pool, budget, current=None, free_transfers=1, locked=(), banned=(),
                allow_hits=False, horizon=None, iters=60, seed=0):
    rng = random.Random(seed)
    pool = [p for p in pool if p.id not in set(banned)]
    players_by_id = {p.id: p for p in pool}
    by_pos = {k: sorted([p for p in pool if p.pos == k], key=lambda p: -p.total_ep)
              for k in POS}
    lock = [p for p in pool if p.id in set(locked)]
    cur = set(current or [])

    def legal(sq):
        if sum(p.cost for p in sq) > budget:
            return False
        counts: dict[int, int] = {}
        for p in sq:
            counts[p.team] = counts.get(p.team, 0) + 1
            if counts[p.team] > MAX_PER_CLUB:
                return False
        return all(sum(1 for p in sq if p.pos == k) == n for k, n in SQUAD_LIMITS.items())

    def score(sq):
        v = squad_value(sq)
        if cur:
            n_new = len([p for p in sq if p.id not in cur])
            if not allow_hits and n_new > free_transfers:
                return -1e9
            v -= TRANSFER_HIT * max(0, n_new - free_transfers)
        return v

    # cheapest available player per position, used to reserve funds
    floor = {k: min((p.cost for p in by_pos[k]), default=40) for k in POS}

    def seed_squad(noise=0.0):
        sq = list(lock)
        # fill the expensive positions first so the budget binds early
        for pos in (4, 3, 2, 1):
            n = SQUAD_LIMITS[pos]
            while sum(1 for p in sq if p.pos == pos) < n:
                spent = sum(p.cost for p in sq)
                # money that must stay back for every slot still unfilled
                reserve = sum(floor[k] * (SQUAD_LIMITS[k] - sum(1 for p in sq if p.pos == k))
                              for k in POS) - floor[pos]
                ceiling = budget - spent - reserve
                cands = [p for p in by_pos[pos]
                         if p not in sq and p.cost <= ceiling
                         and sum(1 for q in sq if q.team == p.team) < MAX_PER_CLUB]
                if not cands:
                    return None
                key = lambda p: -(p.total_ep / max(p.cost - 35, 5)) * (
                    1 + noise * rng.gauss(0, 1))
                sq.append(min(cands, key=key))
        return sq if legal(sq) else None

    best, best_s = None, float("-inf")
    for attempt in range(6):
        # in transfer mode the current squad is the only legal starting point
        sq = ([players_by_id[i] for i in current] if cur
              else seed_squad(noise=0.0 if attempt == 0 else 0.25))
        if sq is None:
            continue
        improved = True
        rounds = 0
        while improved and rounds < iters:
            improved, rounds = False, rounds + 1
            for out in list(sq):
                if out.id in set(locked):
                    continue
                spare = budget - sum(p.cost for p in sq) + out.cost
                base = score(sq)
                for inn in by_pos[out.pos]:
                    if inn in sq or inn.cost > spare:
                        continue
                    trial = [p for p in sq if p is not out] + [inn]
                    if not legal(trial):
                        continue
                    s = score(trial)
                    if s > base + 1e-9:
                        sq, base, improved = trial, s, True
                        break
        s = score(sq)
        if s > best_s:
            best, best_s = sq, s
    return best, best_s


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def report(squad, budget, current=None, players=None, gw=None):
    order = {1: 0, 2: 1, 3: 2, 4: 3}
    score, xi, cap = best_xi(squad, 0)
    bench = [p for p in squad if p not in xi]
    vc = max((p for p in xi if p is not cap), key=lambda p: p.gw1_ep)

    print("\n" + "=" * 62)
    label = f"GW{gw} xP" if gw else "xP"
    print(f"{'STARTING XI':<24}{'CLUB':>6}{'POS':>6}{'£':>8}{label:>8}")
    print("-" * 62)
    for p in sorted(xi, key=lambda p: (order[p.pos], -p.gw1_ep)):
        tag = " (C)" if p is cap else (" (V)" if p is vc else "")
        print(f"{p.name + tag:<24}{p.team_name:>6}{POS[p.pos]:>6}"
              f"{p.cost/10:>8.1f}{p.gw1_ep:>8.2f}")
    print("-" * 62)
    print("BENCH")
    ranked = sorted(bench, key=lambda p: (p.pos != 1, -p.gw1_ep))
    n = 0
    for p in ranked:
        if p.pos == 1:
            label = "GK"
        else:
            n += 1
            label = str(n)
        print(f"{label + '. ' + p.name:<24}{p.team_name:>6}{POS[p.pos]:>6}"
              f"{p.cost/10:>8.1f}{p.gw1_ep:>8.2f}")
    print("=" * 62)
    spend = sum(p.cost for p in squad)
    print(f"Spend £{spend/10:.1f}m   |   ITB £{(budget - spend)/10:.1f}m"
          f"   |   GW xP {score:.1f} (incl. captain)")
    print(f"Horizon xP (decayed): {squad_value(squad):.1f}")

    if current:
        cur = set(current)
        out = [p for p in current if p not in {q.id for q in squad}]
        inn = [p for p in squad if p.id not in cur]
        if inn:
            print(f"\nTransfers ({len(inn)}):")
            for p in inn:
                print(f"  IN   {p.name:<20} {p.team_name:>4}  £{p.cost/10:.1f}m  "
                      f"xP {p.gw1_ep:.2f}")
            for pid in out:
                q = (players or {}).get(pid)
                label = f"{q.name:<20} {q.team_name:>4}  £{q.cost/10:.1f}m" if q else f"id {pid}"
                print(f"  OUT  {label}")
        else:
            print("\nNo transfer improves the squad. Roll it.")

    flagged = [p for p in squad if p.news]
    if flagged:
        print("\nNews:")
        for p in flagged:
            print(f"  {p.name}: {p.news}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=100.0, help="squad budget in £m")
    ap.add_argument("--bank", type=float, default=0.0, help="money in the bank (£m)")
    ap.add_argument("--horizon", type=int, default=3, help="gameweeks to project")
    ap.add_argument("--squad", help="JSON file of your current 15 player ids")
    ap.add_argument("--entry", type=int,
                    help="your FPL team id — pulls your live squad and bank")
    ap.add_argument("--free-transfers", type=int, default=1)
    ap.add_argument("--allow-hits", action="store_true",
                    help="permit transfers beyond the free ones at -4 each")
    ap.add_argument("--pool-size", type=int, default=45,
                    help="candidates kept per position (speeds up the solve)")
    ap.add_argument("--projections", help="CSV of your own xP: id,ep or web_name,ep")
    ap.add_argument("--lock", default="", help="comma-separated player ids to force in")
    ap.add_argument("--ban", default="", help="comma-separated player ids to exclude")
    ap.add_argument("--min-ep", type=float, default=0.0,
                    help="drop players below this xP to shrink the search")
    ap.add_argument("--save", help="write the resulting squad ids to this JSON file")
    ap.add_argument("--no-solver", action="store_true", help="force the heuristic")
    ap.add_argument("--explain", type=int, metavar="ID",
                    help="dump a player's full ep vector with gameweek labels")
    args = ap.parse_args()

    players, teams, gw = load_players(args.projections)
    project(players, teams, gw, max(1, args.horizon))
    print(f"Optimizing for GW{gw} over a {args.horizon}-week horizon "
          f"({len(players)} players).")

    current = None
    bank = args.bank
    sell_prices: dict[int, int] = {}
    if args.entry:
        current, bank, sell_prices = fetch_entry(args.entry, gw)
        if args.bank:
            bank = args.bank
    elif args.squad:
        with open(args.squad, encoding="utf-8") as f:
            raw = json.load(f)
        current = raw["squad"] if isinstance(raw, dict) else raw

    if current:
        # value held players at what they'd actually sell for
        held = sum(sell_prices.get(i, players[i].cost) for i in current)
        budget = round(held + bank * 10)
        lost = sum(players[i].cost for i in current) - held
        print(f"Current squad sells for £{held/10:.1f}m "
              f"+ £{bank:.1f}m ITB = £{budget/10:.1f}m to spend.")
        if lost > 0:
            print(f"  (£{lost/10:.1f}m of paper profit is not recoverable — "
                  f"FPL returns half of any rise, rounded down.)")
    else:
        budget = round(args.budget * 10)

    keep = set(current or [])
    live = [p for p in players.values() if p.avail > 0 or p.id in keep]
    pool = []
    for pos in POS:
        ranked = sorted([p for p in live if p.pos == pos], key=lambda p: -p.total_ep)
        pool += [p for p in ranked[:args.pool_size] if p.total_ep >= args.min_ep]
    pool += [players[i] for i in keep if players[i] not in pool]

    eps = sorted((p.gw1_ep for p in pool), reverse=True)
    spread = eps[0] - eps[len(eps) // 2] if eps else 0
    print(f"Projection spread: top {eps[0]:.2f}, median {eps[len(eps)//2]:.2f} "
          f"(delta {spread:.2f}).")
    if spread < 2.0:
        print("  ! Projections are flat — FPL's ep_next has little signal this "
              "early. Treat the output as a constraint check, not a ranking.")
    locked = [int(x) for x in args.lock.split(",") if x.strip()]
    banned = [int(x) for x in args.ban.split(",") if x.strip()]

    kw = dict(current=current, free_transfers=args.free_transfers,
              locked=locked, banned=banned, allow_hits=args.allow_hits,
              horizon=args.horizon)
    solver = solve_local if args.no_solver else solve_milp
    try:
        squad, value = solver(pool, budget, **kw)
    except ImportError:
        print("PuLP not found — using the local-search fallback "
              "(`pip install pulp` for the exact optimum).")
        squad, value = solve_local(pool, budget, **kw)

    report(squad, budget, current, players, gw)

    if args.explain:
        p = players.get(args.explain)
        if p:
            print(f"\n{p.name} ep vector (index -> gameweek -> xP):")
            for i, v in enumerate(p.ep):
                mark = "  <- shown in table" if i == 0 else ""
                print(f"  ep[{i}]  GW{gw + i}  {v:.3f}{mark}")
            if p.fixed_ep:
                print(f"  raw from CSV: "
                      f"{ {g: round(v, 3) for g, v in sorted(p.fixed_ep.items())} }")
            if p.ep and p.fixed_ep:
                want = p.fixed_ep.get(gw)
                if want is not None and abs(want - p.ep[0]) > 1e-6:
                    print(f"  !! MISALIGNED: table shows {p.ep[0]:.3f} but "
                          f"GW{gw} in the CSV is {want:.3f}")

    if current:
        # what does doing nothing score? that's the bar any move has to clear
        roll = [players[i] for i in current]
        base = sum(
            DECAY**g * (best_xi(roll, g)[0]
                        + BENCH_WEIGHT * sum(p.ep[g] for p in roll
                                             if p not in best_xi(roll, g)[1]))
            for g in range(args.horizon)
        )
        n_new = len([p for p in squad if p.id not in set(current)])
        print(f"\nRoll (0 transfers): {base:.2f}   |   "
              f"This move ({n_new}): {value:.2f}   |   "
              f"Edge: {value - base:+.2f} over {args.horizon} GWs")
        if value - base < 1.0:
            print("  Thin. Rolling the transfer is the better play.")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"squad": [p.id for p in squad],
                       "gw": gw,
                       "spend": sum(p.cost for p in squad)}, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
