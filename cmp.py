import csv, json, urllib.request as u
b = json.load(u.urlopen("https://fantasy.premierleague.com/api/bootstrap-static/"))
m = {e["id"]: e for e in b["elements"]}
rows = list(csv.DictReader(open("advanced.csv", encoding="utf-8")))

pairs = []
for r in rows:
    e = m.get(int(r["id"]))
    if not e:
        continue
    mins = float(e["minutes"] or 0)
    if mins < 180:
        continue
    fpl_xa = float(e["expected_assists"] or 0) / (mins / 90)
    fpl_xg = float(e["expected_goals"] or 0) / (mins / 90)
    und_xa = float(r["xa90"])
    und_xg = float(r["np_xg90"])
    pairs.append((e["web_name"], fpl_xg, und_xg, fpl_xa, und_xa))

print(f'{"PLAYER":<18}{"FPL xG":>8}{"UND npxG":>10}{"FPL xA":>8}{"UND xA":>8}')
print("-" * 52)
for n, fg, ug, fa, ua in sorted(pairs, key=lambda p: -p[2])[:20]:
    print(f"{n:<18}{fg:>8.3f}{ug:>10.3f}{fa:>8.3f}{ua:>8.3f}")

def ratio(idx_f, idx_u):
    vals = [p[idx_u] / p[idx_f] for p in pairs if p[idx_f] > 0.05]
    vals.sort()
    return sum(vals) / len(vals), vals[len(vals) // 2], len(vals)

mg, medg, ng = ratio(1, 2)
ma, meda, na = ratio(3, 4)
print(f"\nnpxG / FPL xG : mean {mg:.3f}, median {medg:.3f}  (n={ng})")
print(f"xA   / FPL xA : mean {ma:.3f}, median {meda:.3f}  (n={na})")
print("\nA median near 1.0 means the providers agree and there is nothing to")
print("gain. Well below 1.0 on xG is the penalty component FPL includes.")
print("Anything far from 1.0 on xA is a genuine provider disagreement.")
