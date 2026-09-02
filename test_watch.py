"""Exercise the roster diff and review queue against synthetic transfer-window churn."""
import json, os, shutil
import watch as W

teams = {1:{"id":1,"short_name":"ARS"},2:{"id":2,"short_name":"NEW"},3:{"id":3,"short_name":"CHE"}}

def el(pid, name, team, cost, status="a", news="", starts=1, minutes=90):
    return {"id":pid,"web_name":name,"team":team,"now_cost":cost,"status":status,
            "news":news,"element_type":2,"starts":starts,"minutes":minutes,
            "chance_of_playing_next_round":None}

before = [el(1,"Mosquera",1,55), el(2,"Thiaw",2,50), el(3,"Leaver",3,60),
          el(4,"Crocked",1,70), el(5,"Returning",2,65,status="i")]
old = W.snapshot_roster(before, teams)

# window churn: Thiaw moves club, Leaver removed, Newbie added,
# Crocked flagged, Returning cleared, Mosquera rises
after = [el(1,"Mosquera",1,56), el(2,"Thiaw",3,50), el(4,"Crocked",1,70,status="i",
             news="Knee injury - 75% chance of playing"),
         el(5,"Returning",2,65,status="a"), el(6,"Newbie",2,75)]
new = W.snapshot_roster(after, teams)

xp = {1:19.0, 2:19.7, 4:16.0, 5:14.0, 6:12.0}
squad = {1, 2}

lines = W.diff_roster(old, new, teams, xp, squad)
out = "\n".join(lines)
print(out)

assert "CLUB CHANGES" in out and "Thiaw" in out, "club move must be flagged"
assert "NEW TO THE GAME" in out and "Newbie" in out
assert "REMOVED FROM THE GAME" in out and "Leaver" in out
assert "NEWLY UNAVAILABLE" in out and "Crocked" in out
assert "BACK TO AVAILABLE" in out and "Returning" in out
assert "PRICE MOVES" in out and "Mosquera" in out
assert "IN YOUR SQUAD" in out, "squad members must be marked"
print("\n--- diff covers every churn type ---")

# review queue: no-prior + high xP should outrank a well-understood player
deep = {1, 4}          # only these have measured priors
q = W.review_queue(after, teams, xp, squad, deep, top_n=10)
names = [r["name"] for r in q]
print("\nreview queue:", names)
assert "Newbie" in names, "a new signing with real xP must be queued"
assert "Thiaw" in names, "squad member without a prior must be queued"
assert "Crocked" in names, "status flag must be queued"

print("\npassed")
