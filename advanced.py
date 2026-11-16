#!/usr/bin/env python3
"""
advanced.py — non-penalty xG and creation volume from Understat.

FPL's own `expected_goals` INCLUDES penalties. That is the single clearest
measurement error in the pipeline: a midfielder whose one goal was a spot-kick
carries an inflated xG rate for weeks, and the model treats it as open-play
scoring ability. Understat publishes `np_xg` separately, along with shots and
key passes — both of which stabilise far faster than goals or assists at small
samples, which is exactly the regime early season sits in.

    pip install soccerdata
    python advanced.py --season 2026-27 --out advanced.csv
    python advanced.py --unmatched          # names that failed to resolve

Writes `id,np_xg90,xa90,shots90,key_passes90,chain90,buildup90,minutes`.
`projections.py` reads it the same way it reads priors.json.

Understat only — deliberately not FBref. soccerdata's FBref path hits a
CAPTCHA and auto-installs PyAutoGUI to click through it, which is fragile and
against Sports-Reference's terms. Understat is a clean JSON endpoint and
carries everything in the priority list.

STATUS: Understat has been down for maintenance since 2026-11-03. Every URL,
including the getLeagueData endpoints, returns a "We will come back soon" page
with HTTP 200, so soccerdata fails with JSONDecodeError on a non-JSON body.
That is the site, not this script — do not spend time debugging it. Other
projects have moved to FotMob. If a source returns, the better experiment is a
walk-forward rewrite of test_advanced.py: the current test correlates season
aggregates, so n is the player count (~115) no matter how many gameweeks pass.
Predicting each player-gameweek from rates known beforehand would give ~900
rows from the same season.

Two tests of this data have already returned null results against FPL's own
xG+xA, so nothing here is load-bearing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import unicodedata

import projections as P

MIN_MINUTES = 180          # below this the per-90 rates are noise


def norm(s: str) -> str:
    s = s.replace("ø", "o").replace("Ø", "O").replace("ß", "ss")
    s = "".join(c for c in unicodedata.normalize("NFKD", s)
                if not unicodedata.combining(c))
    return "".join(ch for ch in s.lower() if ch.isalpha() or ch == " ").strip()


def surname(full: str) -> str:
    parts = norm(full).split()
    return parts[-1] if parts else ""


def fetch(season: str, raw_out: str | None = None):
    try:
        import soccerdata as sd
    except ImportError:
        sys.exit("pip install soccerdata")
    u = sd.Understat("ENG-Premier League", season)
    df = u.read_player_season_stats().reset_index()
    if raw_out:
        df.to_csv(raw_out, index=False)
    return df.to_dict("records")


def load_raw(path: str):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def staleness(rows, boot) -> tuple[int, int, str | None]:
    """Compare the minutes in a scrape against how far the season has run.

    A stale file loads perfectly and produces a plausible-looking table, which
    is how a GW5-era advanced.csv once got compared against ten gameweeks of
    outcomes and returned a confident wrong answer. Nothing downstream can
    detect that, so it has to be caught here.
    """
    try:
        scraped = max(float(r.get("minutes") or 0) for r in rows)
    except ValueError:
        return 0, 0, "no minutes column"
    live = max((float(e.get("minutes") or 0) for e in boot["elements"]),
               default=0.0)
    if live <= 0:
        return int(scraped), 0, None
    gw_scraped = round(scraped / 90.0)
    gw_live = round(live / 90.0)
    if gw_live - gw_scraped >= 2:
        return gw_scraped, gw_live, (
            f"this scrape covers about {gw_scraped} gameweek(s); the season "
            f"has run {gw_live}")
    return gw_scraped, gw_live, None


def match(rows, elements, teams):
    """Understat full names to FPL element ids.

    Understat gives "Bryan Mbeumo"; FPL gives "Mbeumo". Match on surname
    within the same club first — that resolves nearly everything and makes a
    wrong match almost impossible, since two players at one club rarely share
    a surname. Fall back to a league-wide unique surname, and refuse anything
    still ambiguous rather than guessing.
    """
    by_club: dict = {}
    by_surname: dict = {}
    for e in elements:
        club = teams[e["team"]]["name"]
        sn = surname(e["web_name"])
        by_club.setdefault((norm(club), sn), []).append(e)
        by_surname.setdefault(sn, []).append(e)
        # web_name is sometimes an initial+surname ("Bruno G.", "M.Sangaré");
        # also index the full name so those still resolve
        full_sn = surname(f"{e.get('first_name','')} {e.get('second_name','')}")
        if full_sn and full_sn != sn:
            by_club.setdefault((norm(club), full_sn), []).append(e)
            by_surname.setdefault(full_sn, []).append(e)

    matched, unmatched, ambiguous = {}, [], []
    for r in rows:
        try:
            mins = float(r["minutes"])
        except (TypeError, ValueError):
            continue
        if mins < MIN_MINUTES:
            continue

        sn = surname(r["player"])
        club = norm(r["team"])
        hits = by_club.get((club, sn), [])
        if not hits:
            hits = by_surname.get(sn, [])
        hits = list({e["id"]: e for e in hits}.values())

        if len(hits) == 1:
            matched[hits[0]["id"]] = r
        elif not hits:
            unmatched.append(r)
        else:
            ambiguous.append((r, hits))
    return matched, unmatched, ambiguous


def per90(r, key, mins):
    try:
        return float(r[key]) / (mins / 90.0)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--out", default="advanced.csv")
    ap.add_argument("--raw", default="understat_raw.csv",
                    help="cache the scraped table here, and reuse it with "
                         "--offline")
    ap.add_argument("--offline", action="store_true",
                    help="read --raw instead of scraping")
    ap.add_argument("--allow-stale", action="store_true",
                    help="write even when the scrape lags the season — a "
                         "stale file loads fine and silently invalidates "
                         "anything built on it")
    ap.add_argument("--unmatched", action="store_true",
                    help="list players that failed to resolve, and stop")
    args = ap.parse_args()

    rows = load_raw(args.raw) if args.offline else fetch(args.season, args.raw)

    boot = P._get("bootstrap-static/", "bootstrap.json")
    teams = {t["id"]: t for t in boot["teams"]}

    gw_s, gw_l, stale = staleness(rows, boot)
    if stale:
        print(f"\n  ! STALE: {stale}.")
        if args.offline:
            print(f"    {args.raw} is a snapshot, not current data. Anything "
                  f"built from it\n    will be compared against outcomes it "
                  f"could not have predicted.")
        if not args.allow_stale:
            sys.exit("    Refusing to write. Re-scrape, or pass "
                     "--allow-stale if you know why.")
        print("    --allow-stale given; continuing.\n")

    matched, unmatched, ambiguous = match(rows, boot["elements"], teams)

    print(f"{len(rows)} Understat rows, "
          f"{sum(1 for r in rows if float(r.get('minutes') or 0) >= MIN_MINUTES)} "
          f"with {MIN_MINUTES}+ minutes.")
    print(f"Matched {len(matched)} to FPL ids. "
          f"{len(unmatched)} unmatched, {len(ambiguous)} ambiguous.")

    if args.unmatched or unmatched or ambiguous:
        if unmatched:
            print(f"\nUNMATCHED ({len(unmatched)}) — no FPL player with that "
                  f"surname:")
            for r in sorted(unmatched, key=lambda r: -float(r["minutes"]))[:25]:
                print(f"  {r['player']:<26}{r['team']:<20}"
                      f"{float(r['minutes']):>6.0f} min")
        if ambiguous:
            print(f"\nAMBIGUOUS ({len(ambiguous)}) — refused rather than "
                  f"guessed:")
            for r, hits in ambiguous[:15]:
                opts = ", ".join(f"{e['web_name']} ({e['id']})" for e in hits[:4])
                print(f"  {r['player']:<26}{r['team']:<20} -> {opts}")
    if args.unmatched:
        return

    out = []
    for pid, r in matched.items():
        mins = float(r["minutes"])
        out.append({
            "id": pid,
            "np_xg90": round(per90(r, "np_xg", mins), 4),
            "xa90": round(per90(r, "xa", mins), 4),
            "shots90": round(per90(r, "shots", mins), 3),
            "key_passes90": round(per90(r, "key_passes", mins), 3),
            "chain90": round(per90(r, "xg_chain", mins), 4),
            "buildup90": round(per90(r, "xg_buildup", mins), 4),
            "minutes": int(mins),
        })
    out.sort(key=lambda r: -r["np_xg90"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {len(out)} rows to {args.out} "
          f"(covers ~{gw_s} gameweek(s))")

    meta = {e["id"]: e for e in boot["elements"]}
    print(f"\nTop non-penalty xG/90 ({MIN_MINUTES}+ minutes):")
    print(f"{'PLAYER':<20}{'npxG/90':>9}{'xA/90':>8}{'Sh/90':>7}{'KP/90':>7}"
          f"{'MIN':>6}")
    print("-" * 57)
    for r in out[:15]:
        e = meta[r["id"]]
        print(f"{e['web_name']:<20}{r['np_xg90']:>9.3f}{r['xa90']:>8.3f}"
              f"{r['shots90']:>7.2f}{r['key_passes90']:>7.2f}"
              f"{r['minutes']:>6}")




if __name__ == "__main__":
    main()
