import csv, json, urllib.request as u
b = json.load(u.urlopen("https://fantasy.premierleague.com/api/bootstrap-static/"))
m = {e["id"]: e for e in b["elements"]}
t = {x["id"]: x["short_name"] for x in b["teams"]}
rows = list(csv.DictReader(open("advanced.csv", encoding="utf-8")))

defs = []
for r in rows:
    e = m.get(int(r["id"]))
    if not e or e["element_type"] != 2:
        continue
    if int(r["minutes"]) < 270:
        continue
    defs.append((e["web_name"], t[e["team"]], float(r["np_xg90"]),
                 float(r["xa90"]), float(r["key_passes90"]),
                 float(r["np_xg90"]) + float(r["xa90"]), int(r["minutes"])))

defs.sort(key=lambda d: -d[5])
print(f'{"PLAYER":<18}{"TM":>4}{"npxG":>8}{"xA":>8}{"KP/90":>8}{"xG+xA":>8}{"MIN":>6}')
print("-" * 60)
for d in defs[:20]:
    print(f"{d[0]:<18}{d[1]:>4}{d[2]:>8.3f}{d[3]:>8.3f}{d[4]:>8.2f}{d[5]:>8.3f}{d[6]:>6}")

med = sorted(d[5] for d in defs)[len(defs)//2]
print(f"\nmedian defender xG+xA per 90: {med:.3f}  (n={len(defs)})")
print("Anyone far above that is doing a job the DEF bracket does not describe.")
