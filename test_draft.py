"""Validate the id mapping, lineup legality, and the H2H variance logic."""
import math, draft as D, projections as P

# --- id mapping ------------------------------------------------------------
dteams = [{"id":1,"short_name":"ARS"},{"id":2,"short_name":"CHE"}]
cteams = [{"id":9,"short_name":"ARS"},{"id":4,"short_name":"CHE"}]
def e(i,n,t,pos): return {"id":i,"web_name":n,"team":t,"element_type":pos,"status":"a","news":""}
draft_boot={"teams":dteams,"elements":[e(1,"Saka",1,3), e(2,"Palmer",2,3),
                                       e(3,"Guéhi",1,2), e(9,"Ghost",1,4)]}
classic_boot={"teams":cteams,"elements":[e(101,"Saka",9,3), e(2,"Palmer",4,3),
                                         e(303,"Guehi",9,2)]}
m = D.build_id_map(draft_boot, classic_boot)
print("id map:", m)
assert m[1]==101, "different ids, same player -> must map by identity"
assert m[2]==2,   "matching id must still map"
assert m[3]==303, "accents must not block a match"
assert 9 not in m, "a player with no classic counterpart must be left out"

# --- lineup legality -------------------------------------------------------
def row(n,pos,xp,team="ARS"):
    return {"id":0,"cid":0,"name":n,"pos":pos,"team":team,"xp":xp,
            "status":"a","news":""}
squad = ([row(f"g{i}",1,2.0+i) for i in range(2)] +
         [row(f"d{i}",2,1.0+i) for i in range(5)] +
         [row(f"m{i}",3,2.0+i) for i in range(5)] +
         [row(f"f{i}",4,3.0+i) for i in range(3)])
tot, xi = D.best_xi(squad)
assert len(xi)==11
assert sum(1 for p in xi if p["pos"]==1)==1
assert 3 <= sum(1 for p in xi if p["pos"]==2) <= 5
assert 2 <= sum(1 for p in xi if p["pos"]==3) <= 5
assert 1 <= sum(1 for p in xi if p["pos"]==4) <= 3
# must pick the best available, i.e. beat any random legal XI
alt = squad[:1]+squad[2:5]+squad[7:11]+squad[12:15]
assert tot >= sum(p["xp"] for p in alt)
print(f"best XI {tot:.1f}, formation "
      f"{sum(1 for p in xi if p['pos']==2)}-{sum(1 for p in xi if p['pos']==3)}"
      f"-{sum(1 for p in xi if p['pos']==4)}")

# --- variance / win probability -------------------------------------------
assert abs(D.win_probability(50,100,50,100) - 0.5) < 1e-9, "level -> 50%"
assert D.win_probability(60,100,50,100) > 0.5
assert D.win_probability(40,100,50,100) < 0.5
# as an underdog, MORE variance must help
behind_lowvar = D.win_probability(45, 60, 55, 100)
behind_hivar  = D.win_probability(45, 200, 55, 100)
print(f"underdog: low-var {behind_lowvar:.3f} vs high-var {behind_hivar:.3f}")
assert behind_hivar > behind_lowvar, "variance should help when behind"
# when favoured, LESS variance must help
ahead_lowvar = D.win_probability(55, 60, 45, 100)
ahead_hivar  = D.win_probability(55, 200, 45, 100)
print(f"favoured: low-var {ahead_lowvar:.3f} vs high-var {ahead_hivar:.3f}")
assert ahead_lowvar > ahead_hivar, "variance should hurt when ahead"

# sd must rise with xP and be higher for attackers
assert D.player_sd(5,4) > D.player_sd(5,1)
assert D.player_sd(8,3) > D.player_sd(2,3)
print("passed")
