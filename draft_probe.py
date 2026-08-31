#!/usr/bin/env python3
"""
draft_probe.py — find out what the Draft API actually exposes.

The Draft endpoints aren't publicly documented and differ from the classic
game's. Rather than guess paths one at a time, try the plausible set and report
which respond, with their top-level keys and a peek at anything that looks like
a league id or a picks list.

    python draft_probe.py --entry 274930
"""

import argparse
import json
import sys

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://draft.premierleague.com/api"
H = {"User-Agent": "Mozilla/5.0"}


def probe(path):
    url = f"{API}/{path}"
    try:
        r = requests.get(url, timeout=15, headers=H)
    except Exception as exc:
        return path, None, f"error: {exc}"
    if r.status_code != 200:
        return path, r.status_code, None
    try:
        return path, 200, r.json()
    except Exception:
        return path, 200, "(not json)"


def summarise(data, depth=0):
    if isinstance(data, dict):
        return "{" + ", ".join(list(data)[:14]) + ("}" if len(data) <= 14 else ", ...}")
    if isinstance(data, list):
        inner = summarise(data[0], depth + 1) if data else ""
        return f"[{len(data)} items] {inner}"[:200]
    return str(data)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, required=True)
    ap.add_argument("--gw", type=int, default=2,
                    help="a gameweek whose deadline has PASSED")
    args = ap.parse_args()
    e, gw = args.entry, args.gw

    paths = [
        "game",
        "bootstrap-static",
        "bootstrap-dynamic",
        f"entry/{e}/public",
        f"entry/{e}/history",
        f"entry/{e}/event/{gw}",
        f"entry/{e}/event/{gw}/picks",
        f"draft/entry/{e}/event/{gw}",
        f"watchlist/{e}",
    ]

    found_leagues = set()
    print(f"Probing with entry={e}, gw={gw}\n" + "=" * 66)
    for p in paths:
        path, code, data = probe(p)
        if code != 200:
            print(f"  {code or 'ERR':>4}  {path}")
            continue
        print(f"   200  {path}")
        print(f"         {summarise(data)}")
        if isinstance(data, dict):
            # hunt for league ids anywhere in the payload
            for key in ("leagues", "league_set", "entry"):
                v = data.get(key)
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, int):
                            found_leagues.add(item)
                        elif isinstance(item, dict) and "id" in item:
                            found_leagues.add(item["id"])
                elif isinstance(v, dict):
                    print(f"         {key}: {summarise(v)}")
            if "picks" in data:
                print(f"         picks: {summarise(data['picks'])}")

    if found_leagues:
        print(f"\nLeague ids found: {sorted(found_leagues)}")
        for lid in sorted(found_leagues)[:2]:
            for p in (f"league/{lid}/details", f"league/{lid}/element-status"):
                path, code, data = probe(p)
                print(f"  {code or 'ERR':>4}  {path}")
                if code == 200:
                    print(f"         {summarise(data)}")
    else:
        print("\nNo league ids found in any payload. Open a Draft league in "
              "your browser — the URL contains it: /league/<ID>/standings")


if __name__ == "__main__":
    main()
