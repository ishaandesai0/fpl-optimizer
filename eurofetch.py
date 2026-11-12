#!/usr/bin/env python3
"""
eurofetch.py — fill in the Champions League rows of european.json automatically.

Maintaining midweek fixtures by hand works but is easy to forget, and a
forgotten round is worse than no file at all: congestion.py reports stale rest
figures as fact. This pulls the Champions League schedule from
football-data.org, whose free tier covers it.

Europa and Conference are NOT on the free tier, so Brighton, Bournemouth,
Sunderland and Palace stay manual. Those rows are preserved untouched.

    # one-time: get a free key at football-data.org/client/register
    setx FOOTBALL_DATA_KEY your-key-here

    python eurofetch.py                      # merge CL into european.json
    python eurofetch.py --dry-run            # show what would change
    python eurofetch.py --season 2026

Auto-fetched rows carry "src": "auto". Anything without that marker is yours
and is never modified or deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.football-data.org/v4"

# football-data.org names to FPL short names. Only English clubs matter —
# everyone else in the competition is irrelevant to a Premier League deadline.
NAME_MAP = {
    "arsenal": "ARS", "manchester city": "MCI", "manchester united": "MUN",
    "liverpool": "LIV", "aston villa": "AVL", "chelsea": "CHE",
    "tottenham hotspur": "TOT", "newcastle united": "NEW",
    "brighton": "BHA", "brighton & hove albion": "BHA",
    "afc bournemouth": "BOU", "bournemouth": "BOU",
    "crystal palace": "CRY", "sunderland": "SUN", "nottingham forest": "NFO",
    "everton": "EVE", "fulham": "FUL", "brentford": "BRE",
    "leeds united": "LEE", "west ham united": "WHU", "wolverhampton": "WOL",
}


def to_short(name: str) -> str | None:
    n = name.lower().replace(" fc", "").replace("afc ", "").strip()
    if n in NAME_MAP:
        return NAME_MAP[n]
    for key, short in NAME_MAP.items():
        if key in n or n in key:
            return short
    return None


def fetch(key: str, comp: str, season: int):
    url = f"{API}/competitions/{comp}/matches?season={season}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r).get("matches", [])
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:200]
        if e.code == 403:
            sys.exit(f"403 from football-data.org — your key does not cover "
                     f"{comp}. The free tier includes the Champions League "
                     f"(CL) but not the Europa or Conference League.\n{body}")
        if e.code == 429:
            sys.exit("429 — rate limited. The free tier allows 10 requests "
                     "a minute; wait and retry.")
        sys.exit(f"HTTP {e.code} from football-data.org: {body}")
    except Exception as e:
        sys.exit(f"could not reach football-data.org: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", default=os.environ.get("FOOTBALL_DATA_KEY"),
                    help="football-data.org API key, or set FOOTBALL_DATA_KEY")
    ap.add_argument("--comp", default="CL",
                    help="competition code; CL is the only European one on "
                         "the free tier")
    ap.add_argument("--season", type=int,
                    help="starting year; defaults to the current campaign")
    ap.add_argument("--file", default="european.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.key:
        sys.exit("No API key. Register free at "
                 "football-data.org/client/register, then:\n"
                 "  setx FOOTBALL_DATA_KEY your-key-here\n"
                 "(open a new terminal afterwards)")

    today = dt.date.today()
    season = args.season or (today.year if today.month >= 7 else today.year - 1)

    matches = fetch(args.key, args.comp, season)
    if not matches:
        sys.exit(f"No {args.comp} matches returned for season {season}. "
                 f"Check the season year.")

    # one row per date, listing the English clubs playing that day
    by_date: dict = {}
    for m in matches:
        utc = m.get("utcDate")
        if not utc:
            continue
        d = dt.datetime.fromisoformat(utc.replace("Z", "+00:00")).date()
        for side in ("homeTeam", "awayTeam"):
            nm = (m.get(side) or {}).get("name") or ""
            short = to_short(nm)
            if short:
                by_date.setdefault(d.isoformat(), set()).add(short)

    if not by_date:
        sys.exit("No English clubs found in the fixture list — the name "
                 "mapping may need updating.")

    fresh = [{"date": d, "comp": args.comp, "teams": sorted(t), "src": "auto"}
             for d, t in sorted(by_date.items())]

    existing = {"midweek": []}
    if os.path.exists(args.file):
        with open(args.file, encoding="utf-8") as f:
            existing = json.load(f)

    manual = [r for r in existing.get("midweek", [])
              if r.get("src") != "auto" or r.get("comp") != args.comp]
    replaced = len(existing.get("midweek", [])) - len(manual)

    merged = sorted(manual + fresh, key=lambda r: (r["date"], r.get("comp", "")))

    print(f"{args.comp} season {season}: {len(matches)} matches, "
          f"{len(fresh)} dates involving English clubs.")
    print(f"Keeping {len(manual)} manual row(s), replacing {replaced} "
          f"previous auto row(s).\n")
    for r in fresh:
        mark = "auto" if r.get("src") == "auto" else ""
        print(f"  {r['date']}  {r['comp']:<6}{', '.join(r['teams']):<28}{mark}")

    missing = {"BHA", "BOU", "SUN", "CRY"} - {t for r in merged
                                              for t in r["teams"]}
    if missing:
        print(f"\nNot on the free tier: {', '.join(sorted(missing))} — "
              f"add by hand (Thursdays).")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    existing["midweek"] = merged
    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.file} ({len(merged)} rows).")



if __name__ == "__main__":
    main()
