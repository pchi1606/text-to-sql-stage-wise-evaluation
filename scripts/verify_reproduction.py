#!/usr/bin/env python3
# Author: Prabhu Rangaraju
"""Verify the reproduction: check the paper's headline numbers against the
regenerated outputs.

Exits 0 with 'ALL CHECKS PASSED' on success, 1 otherwise.
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def check(name, actual, expected, tol=0):
    if isinstance(expected, float) or isinstance(tol, float):
        ok = actual is not None and abs(actual - expected) <= tol
    else:
        ok = actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {actual}, expected {expected}")
    if not ok:
        failures.append(name)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    # ---- 1. Synthetic arm (n=12): 5/12 -> 12/12, rp=7, rmm=0 ----
    rep = open(os.path.join(BASE, "synth", "REPORT.md")).read()
    m = re.search(r"Baseline EX: (\d+)/(\d+)", rep)
    check("synthetic baseline EX 5/12",
          (int(m.group(1)), int(m.group(2))) if m else None, (5, 12))
    m = re.search(r"\| Qfix \(repaired\) \| (\d+) \|", rep)
    check("synthetic Qfix (rp) = 7", int(m.group(1)) if m else None, 7)
    m = re.search(r"\| Qbreak \(regressed\) \| (\d+) \|", rep)
    check("synthetic Qbreak (rmm) = 0", int(m.group(1)) if m else None, 0)

    # ---- 2. Spider arm (n=88): 10/88 -> 15/88, rp=5, rmm=0 ----
    s = load_json(os.path.join(BASE, "spider", "results.json"))
    check("spider n = 88", s["n"], 88)
    check("spider baseline EX 10/88", s["baseline_ex"], 10)
    check("spider after EX 15/88", s["after_ex"], 15)
    v = s["verification"]
    check("spider Qfix (rp) = 5", v["Qfix"], 5)
    check("spider Qnet = +5", v["Qnet"], 5)
    check("spider broken (rmm) = []", v["broken"], [])

    # ---- 3. BIRD arm (n=100): 0/100 -> 6/100, rp=6, rmm=0 ----
    b0 = load_json(os.path.join(BASE, "bird", "results_v0.json"))
    b1 = load_json(os.path.join(BASE, "bird", "results_v1.json"))
    check("bird n = 100", len(b1), 100)
    check("bird baseline EX 0/100",
          sum(1 for q in b0.values() if q.get("ex") == 1), 0)
    check("bird after EX 6/100",
          sum(1 for q in b1.values() if q.get("ex") == 1), 6)

    # ---- 4. Blind-repair ablation (Spider): localized +5 vs blind -5 ----
    a1 = load_json(os.path.join(BASE, "strengthening", "raw", "a1.json"))
    check("blind ablation: baseline v10 EX = 10", a1["v10_ex"], 10)
    check("blind ablation: localized v11 EX = 15", a1["v11_ex"], 15)
    check("blind ablation: blind EX = 5 (net -5)", a1["blind_ex"], 5)

    # ---- 5. Held-out generalization: rp=4, rmm=0 on each slice ----
    hs = load_json(os.path.join(BASE, "strengthening", "heldout",
                                "spider_heldout_results.json"))["heldout"]
    check("spider held-out R+ (rp) = 4", hs["R+"], 4)
    check("spider held-out R- (rmm) = 0", hs["R-"], 0)
    check("spider held-out Wilson 95% CI",
          [round(x, 4) for x in hs["wilson95_R+_rate"]], [0.0359, 0.2116])
    hb = load_json(os.path.join(BASE, "strengthening", "heldout",
                                "bird_heldout_results.json"))["heldout"]
    check("bird held-out R+ (rp) = 4", hb["R+"], 4)
    check("bird held-out R- (rmm) = 0", hb["R-"], 0)
    check("bird held-out Wilson 95% CI",
          [round(x, 4) for x in hb["wilson95_R+_rate"]], [0.0315, 0.1884])

    # ---- 6. Validation tests ran (tests.py writes no artifact; check raw/) ----
    raw = os.path.join(BASE, "strengthening", "raw")
    n_raw = len([f for f in os.listdir(raw) if f.endswith(".json")]) if os.path.isdir(raw) else 0
    ok = n_raw >= 10
    print(f"[{'PASS' if ok else 'FAIL'}] strengthening raw artifacts present: {n_raw} JSONs")
    if not ok:
        failures.append("raw artifacts")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
