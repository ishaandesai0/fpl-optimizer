import projections as P

def mk(pos, cost, mins=90, starts=1, **kw):
    d = {"element_type": pos, "now_cost": cost, "minutes": mins, "starts": starts,
         "status": "a", "chance_of_playing_next_round": None,
         "expected_goals": 0, "expected_assists": 0, "defensive_contribution": 0,
         "bps": 0, "yellow_cards": 0, "saves": 0}
    d.update(kw); return d

print(f"{'CASE':<34}{'p_start':>9}{'p_60':>7}{'E[min]':>8}")
print("-"*58)
def show(label, e, games=1, last=None, rank=None):
    mm = P.minutes_model(e, games, last, rank); pa, p60, m = mm.p_appear, mm.p_60, mm.exp_minutes
    print(f"{label:<34}{pa:>9.2f}{p60:>7.2f}{m:>8.1f}")
    return pa, p60, m

# the exact failure case: £4.5m nailed keeper, one game played
gk = mk(P.GKP, 45, 90, 1)
_, _, m_price = show("GK £4.5m, price prior only", gk)
_, p60_rank, m_rank = show("GK £4.5m, depth-chart rank 0", gk, rank=0)
# a realistic backup: on the bench all season, no starts
gk_bench = mk(P.GKP, 45, 0, 0)
_, _, m_bkp = show("GK £4.5m, backup rank 1 (0 starts)", gk_bench, games=3, rank=1)
_, p60_last, m_last = show("GK £4.5m, 38 starts last season", gk,
                           last={"_minutes": 3420, "_starts": 38})

assert m_rank > 70, f"nailed keeper should project ~full minutes, got {m_rank:.0f}"
assert m_last > 80
assert m_bkp < 30, f"backup keeper must project low, got {m_bkp:.1f}"
# and a rank-1 keeper who HAS been starting should still read high — the
# depth chart is a prior, not a veto
_, _, m_bkp_playing = show("GK £4.5m, rank 1 but 3 starts",
                           mk(P.GKP, 45, 270, 3), games=3, rank=1)
assert m_bkp_playing > 45, f"actual starts must move the depth chart, got {m_bkp_playing:.1f}"
# The depth chart should still beat the price guess, but by less than it used
# to: weakening the price prior means an observed start now counts for more on
# its own, which is the point.
assert m_rank > m_price * 1.25, \
    f"depth chart should still lead the price prior: {m_rank:.1f} vs {m_price:.1f}"

print()
# budget defender who played every week last season
d = mk(P.DEF, 45, 90, 1)
show("DEF £4.5m, price prior only", d)
_, _, m_nailed = show("DEF £4.5m, 35 starts last season", d,
                      last={"_minutes": 3050, "_starts": 35})
_, _, m_fringe = show("DEF £4.5m, 4 starts last season", d,
                      last={"_minutes": 400, "_starts": 4})
assert m_nailed > 70 and m_fringe < 45
assert m_nailed > m_fringe * 1.8

print()
# premium unchanged
prem = mk(P.FWD, 155, 90, 1)
_, _, m_prem = show("FWD £15.5m, price prior", prem)
assert m_prem > 65

# a sub who came on: 20 mins, 0 starts
sub = mk(P.MID, 50, 20, 0)
_, p60s, m_sub = show("MID £5.0m, 20 min sub, 0 starts", sub,
                      last={"_minutes": 600, "_starts": 5})
assert m_sub < 40 and p60s < 0.4

print("\npassed")
