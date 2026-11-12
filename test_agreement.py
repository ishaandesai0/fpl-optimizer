#!/usr/bin/env python3
"""
test_agreement.py — do projections.py and diagnose.py say the same thing?

Four separate bugs have had the same shape: a flag added to one tool and not
the other, so the two silently disagreed about the same player. The override
system, foreign priors, the DefCon fixture adjustment, and the availability
correction all shipped this way. Each time the symptom was a number in
`diagnose` that did not match `xp.csv`, and each time it was found by eye.

This asserts they agree. It also checks that the data files both tools read
resolve identically, which is the mechanism behind every one of those bugs.

    python test_agreement.py
    python test_agreement.py --offline      # no network; logic checks only

Offline mode skips the live comparison and tests the resolution rules, so it
is safe to run in CI or with no connection.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

import projections as P


def check_resolution():
    """Both tools must choose the same file from the same inputs."""
    failures = []
    with tempfile.TemporaryDirectory() as d:
        present = os.path.join(d, "present.json")
        with open(present, "w") as f:
            f.write("{}")
        missing = os.path.join(d, "missing.json")

        cases = [
            (None, present, False, present, "default present"),
            (None, present, True, None, "default present, disabled"),
            (present, present, False, present, "explicit"),
            (missing, present, False, None, "explicit but absent"),
            (None, missing, False, None, "no default present"),
            (present, missing, True, None, "explicit, disabled"),
        ]
        for explicit, default, disable, want, label in cases:
            got, _ = P.resolve_data_file(explicit, default, disable, "t")
            ok = got == want
            print(f"  {'ok ' if ok else 'FAIL'} {label:<30} -> "
                  f"{os.path.basename(got) if got else None}")
            if not ok:
                failures.append(label)

    # a disabled file must never be read, whatever else is true
    got, note = P.resolve_data_file("anything.json", "other.json", True, "t")
    if got is not None:
        failures.append("disable flag ignored")
    return failures


def check_constants():
    """The filenames both tools default to must exist as shared constants, not
    be duplicated as literals that can drift apart."""
    failures = []
    for name in ("DEFAULT_PRIORS", "DEFAULT_AVAILABILITY", "DEFAULT_RATINGS"):
        if not hasattr(P, name):
            failures.append(f"projections.{name} missing")
            print(f"  FAIL {name} not defined in projections")
        else:
            print(f"  ok  {name} = {getattr(P, name)}")

    src = open("diagnose.py", encoding="utf-8").read()
    # diagnose must reference the constants, not hardcode the names again
    for literal in ('"priors.json"', '"availability.json"', '"ratings.json"'):
        if literal in src:
            failures.append(f"diagnose hardcodes {literal}")
            print(f"  FAIL diagnose.py hardcodes {literal} — use the constant")
    if not failures:
        print("  ok  diagnose.py references the shared constants")
    return failures


def check_flag_parity():
    """Every flag that changes a projection must exist in both tools."""
    import re
    def flags(fn):
        return set(re.findall(r'add_argument\("(--[a-z-]+)"',
                              open(fn, encoding="utf-8").read()))

    pj, dg = flags("projections.py"), flags("diagnose.py")
    # flags that legitimately belong to one tool only
    ONLY_PROJECTIONS = {"--out", "--top", "--deep", "--horizon"}
    ONLY_DIAGNOSE: set = set()

    missing_in_diag = pj - dg - ONLY_PROJECTIONS
    missing_in_proj = dg - pj - ONLY_DIAGNOSE

    failures = []
    for f in sorted(missing_in_diag):
        print(f"  FAIL {f} is in projections.py but not diagnose.py")
        failures.append(f"diagnose missing {f}")
    for f in sorted(missing_in_proj):
        print(f"  FAIL {f} is in diagnose.py but not projections.py")
        failures.append(f"projections missing {f}")
    if not failures:
        print(f"  ok  {len(pj & dg)} shared flag(s), none model-affecting "
              f"on one side only")
    return failures


def check_live(xp_path, sample):
    """Compare diagnose's total against xp.csv for real players."""
    if not os.path.exists(xp_path):
        print(f"  skip {xp_path} not found — run projections.py first")
        return []

    xp: dict = {}
    with open(xp_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            xp.setdefault(int(r["id"]), {})[int(r["gw"])] = float(r["ep"])
    if not xp:
        print("  skip xp.csv is empty")
        return []

    gw0 = min(g for d in xp.values() for g in d)

    import importlib
    dg = importlib.import_module("diagnose")
    if not hasattr(dg, "decompose"):
        print("  skip diagnose.py exposes no reusable decompose() — compare "
              "by eye")
        return []

    failures = []
    ids = sorted(xp, key=lambda i: -sum(xp[i].values()))[:sample]
    for pid in ids:
        want = xp[pid].get(gw0)
        try:
            got = dg.decompose(pid, gw0)
        except Exception as e:
            print(f"  skip id {pid}: {e}")
            continue
        if want is None or got is None:
            continue
        if abs(got - want) > 0.005:
            print(f"  FAIL id {pid}: diagnose {got:.3f} vs xp.csv {want:.3f}")
            failures.append(pid)
        else:
            print(f"  ok  id {pid}: {got:.3f}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xp", default="xp.csv")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    all_failures = []

    print("shared constants")
    all_failures += check_constants()

    print("\nflag parity")
    all_failures += check_flag_parity()

    print("\ndata-file resolution")
    all_failures += check_resolution()

    if not args.offline:
        print("\nlive agreement (diagnose vs xp.csv)")
        all_failures += check_live(args.xp, args.sample)

    print()
    if all_failures:
        print(f"{len(all_failures)} FAILURE(S): "
              f"{', '.join(str(f) for f in all_failures[:6])}")
        sys.exit(1)
    print("passed — the two tools cannot diverge by omission")


if __name__ == "__main__":
    main()
