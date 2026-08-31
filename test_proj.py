"""Unit-test the projection math against hand-checkable cases."""
import math, projections as P

# --- Poisson helpers -------------------------------------------------------
assert abs(sum(P.pois_pmf(k, 1.4) for k in range(30)) - 1.0) < 1e-9, "pmf must sum to 1"
assert abs(P.p_at_least(0, 1.0) - 1.0) < 1e-9
# DefCon: a DEF averaging 7.2 actions/90 clearing a threshold of 10
p = P.p_at_least(10, 7.2)
print(f"P(DEF hits 10 CBIT | lam=7.2) = {p:.3f}  -> {p*2:.2f} pts")
# a high-volume CB at 11/90 should be much more likely
p2 = P.p_at_least(10, 11.0)
print(f"P(DEF hits 10 CBIT | lam=11.0) = {p2:.3f} -> {p2*2:.2f} pts")
assert p2 > p and p2 > 0.5

# conceded penalty must rise with lambda and be ~0 for a dominant defence
c_lo, c_hi = P.expected_conceded_penalty(0.6), P.expected_conceded_penalty(2.4)
print(f"E[conceded penalty] lam=0.6 -> {c_lo:.3f} | lam=2.4 -> {c_hi:.3f}")
assert c_hi > c_lo and c_lo < 0.25

# --- shrinkage -------------------------------------------------------------
assert abs(P.shrink(0, 0, 0.38) - 0.38) < 1e-9, "no minutes -> pure prior"
far = P.shrink(20 * 0.9, 3600, 0.10)   # 3600 mins at 0.45/90 vs 0.10 prior
near = P.shrink(1 * 0.9, 90, 0.10)
print(f"shrink: 3600min -> {far:.3f} (obs 0.45) | 90min -> {near:.3f} (prior 0.10)")
assert far > near and far > 0.35
# posterior weight on observed data must equal minutes/(minutes+k)
w = 90 / (90 + P.PRIOR_MINUTES)
assert abs(near - (w * 0.9 + (1 - w) * 0.10)) < 1e-9, "shrinkage weight is wrong"
print(f"  one game carries {w:.0%} weight — as intended")

# --- team model ------------------------------------------------------------
teams = {i: {"strength_attack_home": 1100 + (i % 5) * 80,
             "strength_attack_away": 1080 + (i % 5) * 80,
             "strength_defence_home": 1100 + (i % 4) * 70,
             "strength_defence_away": 1080 + (i % 4) * 70} for i in range(1, 21)}
st = P.team_strength(teams)
lh, la = P.fixture_lambdas(st, 5, 1)
print(f"lambdas: home {lh:.2f} away {la:.2f}")
assert 0.4 < lh < 4.0 and 0.4 < la < 4.0
assert lh > la, "home advantage should hold for comparable sides"

# --- full player scoring ---------------------------------------------------
def mk(pos, cost, mins=900, starts=10, **kw):
    d = {"element_type": pos, "now_cost": cost, "minutes": mins, "starts": starts,
         "status": "a", "chance_of_playing_next_round": None,
         "expected_goals": 0, "expected_assists": 0, "defensive_contribution": 0,
         "bps": 0, "yellow_cards": 0, "saves": 0}
    d.update(kw); return d

cases = {
  "premium FWD": mk(P.FWD, 155, expected_goals=9.0, expected_assists=2.0, bps=280),
  "mid FWD":     mk(P.FWD, 60,  expected_goals=2.5, expected_assists=1.0, bps=150),
  "attack MID":  mk(P.MID, 120, expected_goals=4.0, expected_assists=5.0, bps=300),
  "defmid":      mk(P.MID, 55,  defensive_contribution=140, bps=190),
  "CB defcon":   mk(P.DEF, 55,  defensive_contribution=130, bps=200),
  "fringe DEF":  mk(P.DEF, 40,  mins=180, starts=2),
  "keeper":      mk(P.GKP, 55,  saves=32, bps=190),
}
print(f"\n{'CASE':<14}{'xP good fix':>12}{'xP hard fix':>12}")
print("-"*38)
out = {}
for name, e in cases.items():
    r = P.player_rates(e); m = P.minutes_model(e, 10)
    good = P.score_fixture(e, r, m, lam_for=2.1, lam_against=0.7, team_avg_for=1.5)
    hard = P.score_fixture(e, r, m, lam_for=0.9, lam_against=2.2, team_avg_for=1.5)
    out[name] = good
    print(f"{name:<14}{good:>12.2f}{hard:>12.2f}")
    assert good > hard, f"{name}: good fixture must beat hard one"

assert out["premium FWD"] > out["mid FWD"], "premium must out-project mid-price"
assert out["fringe DEF"] < 2.5, "fringe player should project low"
assert out["CB defcon"] > out["fringe DEF"]
print("\nall assertions passed")

# --- team_strength fallback chain ------------------------------------------
print("\n--- fallback chain ---")
base = {"strength": 3, "strength_overall_home": 1200, "strength_overall_away": 1180,
        "strength_attack_home": 1150, "strength_attack_away": 1130,
        "strength_defence_home": 1170, "strength_defence_away": 1140}

def variant(**zero):
    t = {i: dict(base) for i in range(1, 21)}
    for i in t:
        for k in zero: t[i][k] = 0
        t[i]["strength_attack_home"] = t[i]["strength_attack_home"] + (i % 5) * 60 if "strength_attack_home" not in zero else 0
        t[i]["strength_overall_home"] = t[i]["strength_overall_home"] + (i % 5) * 60 if "strength_overall_home" not in zero else 0
    return t

# fine ratings present
s = P.team_strength(variant())
assert abs(sum(v["strength_attack_home"] for v in s.values())/20 - 1.0) < 1e-6

# fine ratings zeroed -> overall
s = P.team_strength(variant(strength_attack_home=0, strength_attack_away=0,
                            strength_defence_home=0, strength_defence_away=0))
lh, la = P.fixture_lambdas(s, 5, 1)
assert lh > 0 and la > 0
print(f"  overall fallback lambdas: {lh:.2f} / {la:.2f}")

# everything zeroed except coarse tier
s = P.team_strength(variant(strength_attack_home=0, strength_attack_away=0,
                            strength_defence_home=0, strength_defence_away=0,
                            strength_overall_home=0, strength_overall_away=0))
lh, la = P.fixture_lambdas(s, 5, 1)
assert abs(lh - P.AVG_HOME_GOALS) < 1e-6, "uniform tier -> league average"
print(f"  coarse tier lambdas: {lh:.2f} / {la:.2f}")

# total blackout
s = P.team_strength(variant(strength=0, strength_attack_home=0, strength_attack_away=0,
                            strength_defence_home=0, strength_defence_away=0,
                            strength_overall_home=0, strength_overall_away=0))
lh, la = P.fixture_lambdas(s, 5, 1)
assert abs(lh - P.AVG_HOME_GOALS) < 1e-6
print("  flat fallback OK — no crash")
print("\nfallback chain passed")
