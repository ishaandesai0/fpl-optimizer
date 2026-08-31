"""Recover known ratings from simulated seasons — the real test of a fitter."""
import random, math, ratings as R

rng = random.Random(11)
TIDS = list(range(1, 21))
# ground truth: attack/defence spread across the league
true_atk = {t: 0.55 + 0.09 * (t % 10) + 0.30 * (t <= 4) for t in TIDS}
true_dfn = {t: 1.45 - 0.08 * (t % 10) - 0.25 * (t <= 4) for t in TIDS}
ma = sum(true_atk.values())/20; md = sum(true_dfn.values())/20
true_atk = {t: v/ma for t, v in true_atk.items()}
true_dfn = {t: v/md for t, v in true_dfn.items()}
MU, GAMMA = 1.35, 1.22

def pois(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L: return k
        k += 1

def season(n_rounds):
    out = []
    for _ in range(n_rounds):
        ids = TIDS[:]; rng.shuffle(ids)
        for h, a in zip(ids[::2], ids[1::2]):
            out.append({"h": h, "a": a,
                        "hg": pois(MU*GAMMA*true_atk[h]*true_dfn[a]),
                        "ag": pois(MU*true_atk[a]*true_dfn[h])})
    return out

# a deliberately bad prior: everyone average
flat_prior = {t: (1.0, 1.0) for t in TIDS}

def corr(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(sx*sy) if sx and sy else 0.0

print(f"{'MATCHES/TEAM':>13}{'corr(atk)':>11}{'corr(def)':>11}{'gamma':>8}")
print("-"*43)
prev = -1
for rounds in (1, 5, 19, 38):
    res = season(rounds)
    atk, dfn, mu, gamma = R.fit(res, TIDS, flat_prior, prior_weight=6.0)
    ca = corr([atk[t] for t in TIDS], [true_atk[t] for t in TIDS])
    cd = corr([dfn[t] for t in TIDS], [true_dfn[t] for t in TIDS])
    print(f"{rounds:>13}{ca:>11.3f}{cd:>11.3f}{gamma:>8.3f}")
    if rounds == 38:
        assert ca > 0.85 and cd > 0.80, "should recover truth over a full season"
        assert abs(gamma - GAMMA) < 0.15, f"home advantage off: {gamma:.3f}"
        assert abs(mu - MU) < 0.25, f"base rate off: {mu:.3f}"
    prev = ca

# normalisation invariants
atk, dfn, mu, gamma = R.fit(season(19), TIDS, flat_prior)
assert abs(sum(atk.values())/20 - 1.0) < 1e-6, "attack must normalise to mean 1"
assert abs(sum(dfn.values())/20 - 1.0) < 1e-6, "defence must normalise to mean 1"

# zero results -> pure prior, no crash, no div-by-zero
pr = {t: (1.2 if t == 1 else 0.95, 0.8 if t == 1 else 1.01) for t in TIDS}
a0, d0, mu0, g0 = R.fit([], TIDS, pr, prior_weight=6.0)
assert abs(a0[1]/ (sum(a0.values())/20) - pr[1][0]/(sum(p[0] for p in pr.values())/20)) < 1e-6
print("\nzero-match fit returns the prior cleanly; mu=%.2f gamma=%.2f" % (mu0, g0))

# one round only: prior must dominate, not overfit
res1 = season(1)
atk1, _, _, _ = R.fit(res1, TIDS, pr, prior_weight=6.0)
spread1 = max(atk1.values()) - min(atk1.values())
atk38, _, _, _ = R.fit(season(38), TIDS, pr, prior_weight=6.0)
spread38 = max(atk38.values()) - min(atk38.values())
print(f"attack spread: 1 round {spread1:.3f} vs 38 rounds {spread38:.3f}")
assert spread1 < spread38, "one round must not produce a wider spread than a season"
# gamma must survive an extreme small sample — the real GW1 case was 20-5
extreme = [{"h": 1, "a": 2, "hg": 3, "ag": 0}] * 9
_, _, _, gx = R.fit(extreme, TIDS, flat_prior, prior_weight=6.0)
print(f"gamma on a 27-0 home sweep (9 matches): {gx:.3f}")
assert gx < 1.32, f"gamma must not chase an extreme small sample: {gx:.3f}"

_, _, _, g1 = R.fit(season(1), TIDS, flat_prior, prior_weight=6.0)
print(f"gamma after 1 round: {g1:.3f} (must stay near 1.25)")
assert 1.10 < g1 < 1.35, f"gamma unstable on small sample: {g1:.3f}"

# but it must still converge on the truth given a full season
_, _, _, g38 = R.fit(season(38), TIDS, flat_prior, prior_weight=6.0)
print(f"gamma after 38 rounds: {g38:.3f} (truth {GAMMA})")
assert abs(g38 - GAMMA) < 0.12, f"gamma should converge: {g38:.3f}"
print("\npassed")
