import projections as P

print("--- price-scaled priors (FWD, median 5.5) ---")
for price in (4.5, 6.0, 8.0, 12.0, 15.5):
    pr = P.price_priors(P.FWD, price, 5.5)
    print(f"  £{price:>4.1f}m  xG90 {pr['xg']:.3f}  xA90 {pr['xa']:.3f}  bps {pr['bps']:.1f}")

print("--- price-scaled priors (DEF, median 4.5) ---")
for price in (4.0, 5.5, 6.5, 8.0):
    pr = P.price_priors(P.DEF, price, 4.5)
    print(f"  £{price:>4.1f}m  xG90 {pr['xg']:.3f}  defcon90 {pr['defcon']:.2f}")
# expensive defenders are attacking FBs -> should clear LESS
assert P.price_priors(P.DEF, 8.0, 4.5)["defcon"] < P.price_priors(P.DEF, 4.0, 4.5)["defcon"]
assert P.price_priors(P.FWD, 15.5, 5.5)["xg"] > 2.5 * P.price_priors(P.FWD, 4.5, 5.5)["xg"]
assert 0.80 < P.price_priors(P.FWD, 15.5, 5.5)["xg"] < 1.00, "premium prior must be realistic"

# simulate GW2: one game played, elite striker
def mk(pos, cost, mins, starts, **kw):
    d = {"element_type": pos, "now_cost": cost, "minutes": mins, "starts": starts,
         "status": "a", "chance_of_playing_next_round": None,
         "expected_goals": 0, "expected_assists": 0, "defensive_contribution": 0,
         "bps": 0, "yellow_cards": 0, "saves": 0}
    d.update(kw); return d

haaland = mk(P.FWD, 155, 90, 1, expected_goals=0.85, expected_assists=0.05, bps=25)
budget  = mk(P.FWD, 45,  62, 0, expected_goals=0.10, expected_assists=0.02, bps=6)

print("\n--- GW2 scoring, one game of data ---")
for name, e, med in (("Haaland", haaland, 5.5), ("£4.5m FWD", budget, 5.5)):
    r = P.player_rates(e, med)
    m = P.minutes_model(e, 1)
    xp = P.score_fixture(e, r, m, lam_for=2.0, lam_against=1.0, team_avg_for=1.7)
    print(f"  {name:<12} xG90 {r['xg']:.3f}  E[min] {m.exp_minutes:.0f}  xP {xp:.2f}")

r = P.player_rates(haaland, 5.5); m = P.minutes_model(haaland, 1)
xp = P.score_fixture(haaland, r, m, 2.0, 1.0, 1.7)
assert xp > 5.0, f"elite striker should clear 5 xP in a good fixture, got {xp:.2f}"

# last-season prior must override the price heuristic
r2 = P.player_rates(haaland, 5.5,
                    last={"xg": 0.95, "xa": 0.12, "_minutes": 3000})
print(f"\n  with real prior: xG90 {r2['xg']:.3f} (vs {r['xg']:.3f} from price)")
# the measured prior must win outright, whichever direction it points
assert abs(r2["xg"] - 0.95) < abs(r2["xg"] - r["xg"]) or abs(r2["xg"] - r["xg"]) > 1e-6
assert 0.90 < r2["xg"] < 1.00, "should sit on last season's rate, not the price guess"
# a missing _minutes must NOT silently zero out the prior
r3 = P.player_rates(haaland, 5.5, last={"xg": 0.95, "xa": 0.12})
assert abs(r3["xg"] - r2["xg"]) < 0.02, "absent _minutes should assume full coverage"
# but an explicitly thin season must be discounted toward price
r4 = P.player_rates(haaland, 5.5,
                    last={"xg": 0.95, "xa": 0.12, "_minutes": 600})
assert r4["xg"] < r2["xg"], "a 600-minute season must count for less"
print(f"  thin season (600 min): xG90 {r4['xg']:.3f} vs full {r2['xg']:.3f}")
print("\npassed")
