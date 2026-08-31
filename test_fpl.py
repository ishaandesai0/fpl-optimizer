import random, json, os, sys
sys.argv = ["fpl_opt.py"]
import fpl_opt as F

rng = random.Random(7)
teams = [{"id": i, "short_name": f"T{i:02d}"} for i in range(1, 21)]
elements = []
pid = 0
for t in teams:
    for pos, n in [(1,3),(2,7),(3,9),(4,4)]:
        for _ in range(n):
            pid += 1
            cost = rng.choice([40,45,50,55,60,65,70,75,80,90,110,140])
            ep = max(0.5, cost/22 + rng.gauss(0,1.0))
            elements.append({"id":pid,"web_name":f"P{pid}","element_type":pos,
                "team":t["id"],"now_cost":cost,"ep_next":f"{ep:.1f}","ep_this":"0",
                "status": "a" if rng.random()>0.06 else "i",
                "chance_of_playing_next_round": None,
                "news":"", "selected_by_percent":"5.0","points_per_game":"3.0"})
boot = {"teams":teams,"elements":elements,
        "events":[{"id":1,"is_next":True,"finished":False}]}
fixtures = []
for gw in range(1,6):
    ids = list(range(1,21)); rng.shuffle(ids)
    for a,b in zip(ids[::2], ids[1::2]):
        fixtures.append({"event":gw,"team_h":a,"team_a":b,
                         "team_h_difficulty":rng.randint(2,5),
                         "team_a_difficulty":rng.randint(2,5)})

F._get = lambda path, fname: boot if "bootstrap" in path else fixtures

players, tmap, gw = F.load_players(None)
F.project(players, {t["id"]:t for t in teams}, gw, 3)
print("players:", len(players), "gw:", gw, "eps sample:", players[1].ep)

pool = [p for p in players.values() if p.avail > 0]
sq, _ = F.solve_local(pool, 1000)
assert len(sq)==15, len(sq)
cost = sum(p.cost for p in sq); assert cost <= 1000, cost
from collections import Counter
print("pos counts:", Counter(F.POS[p.pos] for p in sq))
print("max per club:", max(Counter(p.team for p in sq).values()))
F.report(sq, 1000)

# transfer mode
cur = [p.id for p in sq[:15]]
# degrade: swap two good players for junk to see if it recommends transfers
worst = sorted(pool, key=lambda p:p.total_ep)
cur2 = [p.id for p in sq[:13]] + [w.id for w in worst if w.pos in (3,3)][:0]
budget2 = sum(players[i].cost for i in cur) + 20
sq2, _ = F.solve_local(pool, budget2, current=cur, free_transfers=1)
print("\n--- transfer mode ---")
F.report(sq2, budget2, current=cur, players=players)
