#!/usr/bin/env python3
"""
report.py — the whole week on one page.

The weekly loop produces useful output across a dozen commands, and reading it
means scrolling back through a terminal. This renders the same information as
a single self-contained HTML file: squad and recommended XI, the transfer
edge, anyone flagged, the best players by position with season price movement,
and each club's fixture run.

    python report.py --entry 8447496 --out report.html
    python report.py --squad squad.json --out report.html
    python report.py --out report.html            # no squad, just the market

No new dependencies and no server — it writes a file you open in a browser.
Regenerate it after projections.py and it reflects the current model.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import sys

import projections as P

CSS = """
:root{--bg:#0f1117;--card:#181b24;--line:#272b36;--txt:#e6e8ee;--dim:#9aa0ad;
--up:#4ade80;--down:#f87171;--accent:#7dd3fc;--warn:#fbbf24}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--txt);
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:28px 0 10px;color:var(--accent);
border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.wrap{max-width:1180px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:14px 16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:500;padding:5px 8px;
border-bottom:1px solid var(--line);font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td{padding:5px 8px;border-bottom:1px solid rgba(255,255,255,.04)}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.up{color:var(--up)}.down{color:var(--down)}.dim{color:var(--dim)}
.warn{color:var(--warn)}
.pill{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;
background:#2a2f3d;color:var(--dim);margin-left:5px}
.cap{background:#1d4ed8;color:#fff}
.vice{background:#334155;color:#cbd5e1}
.bench td{opacity:.6}
.big{font-size:22px;font-weight:600}
.note{color:var(--dim);font-size:12px;margin-top:8px;line-height:1.5}
"""


def esc(s):
    return html.escape(str(s))


def load_xp(path):
    """Per-gameweek projections, keyed by player then gameweek."""
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["id"]), {})[int(r["gw"])] = float(r["ep"])
    return out


def squad_ids(args, boot):
    if args.squad:
        with open(args.squad, encoding="utf-8") as f:
            raw = json.load(f)
        return (raw["squad"] if isinstance(raw, dict) else raw), None
    if args.entry:
        try:
            import fpl_opt as F
            fixtures = P._get("fixtures/", "fixtures.json")
            gw = P.next_gameweek(boot, fixtures)
            ids, bank, _ = F.fetch_entry(args.entry, gw)
            return ids, bank
        except Exception as e:
            print(f"  ! could not load entry {args.entry}: {e}")
    return [], None


def pos_name(t):
    return P.POS[t]


def price_cell(e):
    """Season price movement. FPL stores it in tenths of a million."""
    d = e.get("cost_change_start", 0) / 10.0
    if d > 0:
        return f'<span class="up">+{d:.1f}</span>'
    if d < 0:
        return f'<span class="down">{d:.1f}</span>'
    return '<span class="dim">0.0</span>'


def flag_cell(e):
    st = e.get("status", "a")
    if st == "a":
        return ""
    chance = e.get("chance_of_playing_next_round")
    label = {"d": f"{chance}%" if chance is not None else "doubt",
             "i": "injured", "s": "susp", "u": "out"}.get(st, st)
    return f'<span class="pill warn">{esc(label)}</span>'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", type=int, help="FPL entry id")
    ap.add_argument("--squad", help="squad.json, for a squad not yet published")
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--ratings", default="ratings.json")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--top", type=int, default=12,
                    help="players per position in the market tables")
    args = ap.parse_args()

    if not os.path.exists(args.xp):
        sys.exit(f"{args.xp} not found — run projections.py first")

    xp = load_xp(args.xp)
    boot = P._get("bootstrap-static/", "bootstrap.json")
    fixtures = P._get("fixtures/", "fixtures.json")
    meta = {e["id"]: e for e in boot["elements"]}
    teams = {t["id"]: t for t in boot["teams"]}

    gws = sorted({g for d in xp.values() for g in d})
    gw0, gwN = gws[0], gws[-1]

    fitted = None
    if os.path.exists(args.ratings):
        with open(args.ratings, encoding="utf-8") as f:
            fitted = json.load(f)

    ids, bank = squad_ids(args, boot)

    def ep(pid, g=None):
        d = xp.get(pid, {})
        return d.get(g, 0.0) if g else sum(d.values())

    parts = []
    parts.append(f"<h1>Fantasy Premier League — Gameweek {gw0}</h1>")
    parts.append(f'<div class="sub">Generated {dt.datetime.now():%d %b %Y, %H:%M} '
                 f'&middot; projections cover GW{gw0}–{gwN} '
                 f'&middot; {len(meta)} players</div>')

    # ---- squad -----------------------------------------------------------
    if ids:
        rows = []
        for pid in ids:
            e = meta.get(pid)
            if not e:
                continue
            rows.append({
                "e": e, "pid": pid,
                "gw": ep(pid, gw0), "tot": ep(pid),
                "pos": e["element_type"],
            })
        # best legal XI, same rule the optimizer uses
        best, formation = None, None
        for d in range(3, 6):
            for m in range(2, 6):
                f_ = 11 - 1 - d - m
                if not (1 <= f_ <= 3):
                    continue
                pick = []
                for t, need in ((1, 1), (2, d), (3, m), (4, f_)):
                    cand = sorted((r for r in rows if r["pos"] == t),
                                  key=lambda r: -r["gw"])
                    if len(cand) < need:
                        pick = None
                        break
                    pick += cand[:need]
                if pick is None:
                    continue
                tot = sum(r["gw"] for r in pick)
                if best is None or tot > best[0]:
                    best, formation = (tot, pick), (d, m, f_)
        xi = {r["pid"] for r in best[1]} if best else set()
        cap = max(best[1], key=lambda r: r["gw"]) if best else None
        vice = sorted(best[1], key=lambda r: -r["gw"])[1] if best else None

        value = sum(e["now_cost"] for e in
                    (meta[i] for i in ids if i in meta)) / 10.0

        parts.append("<h2>Your squad</h2>")
        parts.append('<div class="grid">')
        parts.append('<div class="card">')
        parts.append(f'<div class="big">{best[0]:.1f}</div>'
                     f'<div class="dim">projected GW{gw0}, before captaincy'
                     f' &middot; {formation[0]}-{formation[1]}-{formation[2]}</div>'
                     f'<div class="note">Squad value £{value:.1f}m'
                     + (f" &middot; £{bank:.1f}m in the bank" if bank is not None else "")
                     + "</div>")
        parts.append("</div>")

        parts.append('<div class="card" style="grid-column:1/-1">')
        parts.append("<table><tr><th>Player</th><th>Team</th><th>Pos</th>"
                     f"<th class='num'>£</th><th class='num'>Δ season</th>"
                     f"<th class='num'>GW{gw0}</th>"
                     f"<th class='num'>GW{gw0}–{gwN}</th></tr>")
        order = sorted(rows, key=lambda r: (r["pid"] not in xi, r["pos"], -r["gw"]))
        for r in order:
            e = r["e"]
            tag = ""
            if cap and r["pid"] == cap["pid"]:
                tag = '<span class="pill cap">C</span>'
            elif vice and r["pid"] == vice["pid"]:
                tag = '<span class="pill vice">V</span>'
            cls = "" if r["pid"] in xi else ' class="bench"'
            parts.append(
                f"<tr{cls}><td>{esc(e['web_name'])}{tag}{flag_cell(e)}</td>"
                f"<td class='dim'>{esc(teams[e['team']]['short_name'])}</td>"
                f"<td class='dim'>{pos_name(e['element_type'])}</td>"
                f"<td class='num'>{e['now_cost']/10:.1f}</td>"
                f"<td class='num'>{price_cell(e)}</td>"
                f"<td class='num'>{r['gw']:.2f}</td>"
                f"<td class='num'>{r['tot']:.1f}</td></tr>")
        parts.append("</table>")
        parts.append('<div class="note">Dimmed rows are the bench.</div>')
        parts.append("</div></div>")

        flagged = [r for r in rows if r["e"].get("status", "a") != "a"]
        if flagged:
            parts.append("<h2>Flagged in your squad</h2>")
            parts.append('<div class="card"><table>')
            for r in sorted(flagged, key=lambda r: -r["tot"]):
                e = r["e"]
                parts.append(
                    f"<tr><td>{esc(e['web_name'])}</td>"
                    f"<td class='dim'>{esc(teams[e['team']]['short_name'])}</td>"
                    f"<td class='warn'>{esc(e.get('news') or 'flagged')}</td>"
                    f"<td class='num'>{r['tot']:.1f}</td></tr>")
            parts.append("</table></div>")

    # ---- market ----------------------------------------------------------
    parts.append(f"<h2>Best by position, GW{gw0}–{gwN}</h2>")
    parts.append('<div class="grid">')
    for t in (1, 2, 3, 4):
        cands = []
        for pid, d in xp.items():
            e = meta.get(pid)
            if not e or e["element_type"] != t:
                continue
            cands.append((sum(d.values()), d.get(gw0, 0.0), e))
        cands.sort(key=lambda c: -c[0])
        parts.append('<div class="card">')
        parts.append(f"<h3 style='margin:0 0 8px;font-size:13px'>"
                     f"{pos_name(t)}</h3>")
        parts.append("<table><tr><th>Player</th><th class='num'>£</th>"
                     "<th class='num'>Δ</th>"
                     f"<th class='num'>GW{gw0}</th><th class='num'>Total</th>"
                     "<th class='num'>Own%</th></tr>")
        for tot, g, e in cands[:args.top]:
            mine = ' style="color:#7dd3fc"' if e["id"] in set(ids) else ""
            parts.append(
                f"<tr><td{mine}>{esc(e['web_name'])}{flag_cell(e)}</td>"
                f"<td class='num'>{e['now_cost']/10:.1f}</td>"
                f"<td class='num'>{price_cell(e)}</td>"
                f"<td class='num'>{g:.2f}</td>"
                f"<td class='num'>{tot:.1f}</td>"
                f"<td class='num dim'>{P.fnum(e,'selected_by_percent'):.1f}</td>"
                "</tr>")
        parts.append("</table></div>")
    parts.append("</div>")
    parts.append('<div class="note">Δ = price change this season. '
                 'Yours are highlighted.</div>')

    # ---- fixture runs ----------------------------------------------------
    if fitted:
        n_ahead = min(6, len(gws))
        rows = []
        for tid, t in teams.items():
            lf = la = 0.0
            n = 0
            for f in fixtures:
                if f.get("event") not in set(gws[:n_ahead]):
                    continue
                if tid not in (f["team_h"], f["team_a"]):
                    continue
                h, a = P.fixture_lambdas({}, f["team_h"], f["team_a"], fitted)
                home = f["team_h"] == tid
                lf += h if home else a
                la += a if home else h
                n += 1
            if n:
                rows.append((t["short_name"], lf / n, la / n, n))
        rows.sort(key=lambda r: r[2])
        parts.append(f"<h2>Fixture runs, next {n_ahead} gameweeks</h2>")
        parts.append('<div class="card"><table>'
                     "<tr><th>Team</th><th class='num'>xG for</th>"
                     "<th class='num'>xG against</th><th class='num'>Ratio</th>"
                     "<th class='num'>Games</th></tr>")
        for nm, f_, a_, n in rows:
            parts.append(f"<tr><td>{esc(nm)}</td>"
                         f"<td class='num'>{f_:.2f}</td>"
                         f"<td class='num'>{a_:.2f}</td>"
                         f"<td class='num'>{f_/max(a_,.01):.2f}</td>"
                         f"<td class='num dim'>{n}</td></tr>")
        parts.append("</table>")
        parts.append('<div class="note">A game count below the horizon '
                     'means a blank gameweek.</div>')
        parts.append("</div>")

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>FPL GW{gw0}</title><style>{CSS}</style></head>"
           f"<body><div class='wrap'>{''.join(parts)}</div></body></html>")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Wrote {args.out} ({len(doc)//1024} KB) — open it in a browser.")


if __name__ == "__main__":
    main()
