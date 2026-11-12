import projections as P
h = P._get("element-summary/400/", "element-400.json")["history"]
print(f'{"GW":>4}{"MIN":>6}{"STARTS":>8}{"PTS":>6}')
print("-" * 24)
for x in h:
    print(f'{x["round"]:>4}{x["minutes"]:>6}{x.get("starts", 0):>8}{x["total_points"]:>6}')
mins = sum(x["minutes"] for x in h)
st = sum(x.get("starts", 0) for x in h)
print(f'\ntotal {mins} minutes, {st} starts across {len(h)} gameweeks')
recent = h[-3:]
print(f'last 3: {[x["minutes"] for x in recent]} minutes, '
      f'{sum(x.get("starts",0) for x in recent)} starts')
