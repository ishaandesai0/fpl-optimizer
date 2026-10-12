import projections as P, json
names = ["Raya","Bentley","Calafiori","De Cuyper","Egan","Muharemović","Van Hecke",
         "Rogers","Groß","Belloumi","Gibbs-White","Saka","Haaland","Kostoulas","João Pedro"]
boot = P._get("bootstrap-static/", "bootstrap.json")
ids = []
for n in names:
    hits = [e for e in boot["elements"] if P.strip_accents(e["web_name"]) == P.strip_accents(n)]
    if len(hits) != 1:
        print(f"CHECK {n}: {[(e['id'], e['web_name']) for e in hits]}")
    else:
        ids.append(hits[0]["id"])
        print(f"{n:<16}{hits[0]['id']}")
json.dump({"squad": ids, "bank": 0.0}, open("squad.json","w"))
print(f"\n{len(ids)} of 15 resolved")
