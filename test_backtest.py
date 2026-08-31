"""Validate the metrics against cases with known answers."""
import math, random, os, csv, shutil
import backtest as B

# --- correlation helpers ---------------------------------------------------
assert abs(B.pearson([1,2,3,4,5],[2,4,6,8,10]) - 1.0) < 1e-9, "perfect +"
assert abs(B.pearson([1,2,3,4,5],[10,8,6,4,2]) + 1.0) < 1e-9, "perfect -"
# spearman must be 1.0 for a monotone-but-nonlinear relationship
assert abs(B.spearman([1,2,3,4,5],[1,4,9,16,25]) - 1.0) < 1e-9, "monotone"
assert B.pearson([1,2,3,4,5],[1,4,9,16,25]) < 0.99, "pearson penalises curvature"
# ties handled via average ranks
assert abs(sum(B.rank([5,5,5,5])) - 10.0) < 1e-9
print("correlation helpers OK")

# --- metrics ---------------------------------------------------------------
rows = [{"xp": 3.0, "ep_next": 2.0, "actual": 5.0},
        {"xp": 1.0, "ep_next": 1.0, "actual": 1.0},
        {"xp": 2.0, "ep_next": 3.0, "actual": 0.0}]
m = B.metrics(rows)
assert abs(m["mae"] - (2+0+2)/3) < 1e-9
assert abs(m["bias"] - ((3-5)+(1-1)+(2-0))/3) < 1e-9, "bias must be signed"
print(f"metrics OK (mae {m['mae']:.2f}, bias {m['bias']:+.2f})")

# --- a model that IS good should show it -----------------------------------
rng = random.Random(3)
good, noise = [], []
for _ in range(400):
    truth = max(0.0, rng.gauss(2.5, 2.0))
    actual = max(0, round(truth + rng.gauss(0, 2.0)))
    good.append({"xp": truth, "ep_next": truth*0.5 + rng.gauss(1.2, 1.4),
                 "actual": actual, "minutes": 90, "pos": "MID", "cost": 6.0,
                 "name": "x", "id": 1, "defcon": 0})
mg, mb = B.metrics(good), B.metrics(good, "ep_next")
print(f"informative model spearman {mg['spearman']:.3f} vs baseline {mb['spearman']:.3f}")
assert mg["spearman"] > mb["spearman"], "better model must score better"
assert mg["spearman"] > 0.5

# --- a useless model should NOT look good ----------------------------------
junk = [{"xp": rng.random()*5, "ep_next": 2.0, "actual": max(0, round(rng.gauss(2,2))),
         "minutes": 90, "pos": "MID", "cost": 6.0, "name": "x", "id": 1, "defcon": 0}
        for _ in range(400)]
mj = B.metrics(junk)
print(f"noise model spearman {mj['spearman']:.3f} (should be ~0)")
assert abs(mj["spearman"]) < 0.15, "random projections must not look predictive"

# --- calibration detects a systematically biased model ----------------------
biased = [dict(r, xp=r["xp"]*1.5) for r in good]
assert B.metrics(biased)["bias"] > B.metrics(good)["bias"], "inflation shows as bias"
print("bias detection OK")

# --- end-to-end on a temp history dir --------------------------------------
B.HIST_DIR = "test_history"
shutil.rmtree(B.HIST_DIR, ignore_errors=True)
os.makedirs(B.HIST_DIR)
with open(os.path.join(B.HIST_DIR, "gw02_scored.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id","name","pos","cost","xp","ep_next","actual","minutes","defcon"])
    w.writeheader()
    for i, r in enumerate(good[:120]):
        w.writerow({"id":i,"name":f"P{i}","pos":"MID","cost":6.0,"xp":round(r["xp"],3),
                    "ep_next":round(r["ep_next"],3),"actual":r["actual"],"minutes":90,"defcon":0})
print("\n--- report() end-to-end ---")
B.report()
shutil.rmtree(B.HIST_DIR, ignore_errors=True)
print("\npassed")
