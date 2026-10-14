# Author: Prabhu Rangaraju
"""A6. BOOTSTRAP CONFIDENCE INTERVALS for EX.

95% percentile bootstrap CIs (10,000 resamples, seed 42) for EX on:
synthetic n=12 (v1.0, v1.1), Spider v1.0, Spider v1.1, Spider Agent-B,
BIRD n=100 (v1.0, v1.1).
Wilson score intervals additionally for the n=12 case.
EX vectors are read from the raw outputs of A1/A2/A5 (executed code);
the BIRD vectors are read from real_benchmark/bird/results_v0.json and
results_v1.json (added 2026-09-29 to give the BIRD CIs archived provenance).
"""

import json
import os
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402


def load(name):
    with open(os.path.join(common.RAW_DIR, name)) as fh:
        return json.load(fh)


def main():
    a1 = load("a1.json")
    a2s = load("a2_synth.json")
    a5 = load("a5.json")

    bird_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "bird")
    with open(os.path.join(bird_dir, "results_v0.json")) as fh:
        bird_v0 = json.load(fh)
    with open(os.path.join(bird_dir, "results_v1.json")) as fh:
        bird_v1 = json.load(fh)
    bird_v0_vec = [bird_v0[k]["ex"] for k in sorted(bird_v0)]
    bird_v1_vec = [bird_v1[k]["ex"] for k in sorted(bird_v1)]

    cases = [
        ("synthetic n=12, agent v1.0", a2s["agents"]["v1.0"]["ex_vec"], True),
        ("synthetic n=12, agent v1.1", a2s["agents"]["v1.1"]["ex_vec"], True),
        ("Spider n=88, agent v1.0", a1["v10_vec"], False),
        ("Spider n=88, agent v1.1", a1["v11_vec"], False),
        ("Spider n=88, Agent-B", a5["agent_b"]["ex_vec"], False),
        ("BIRD n=100, agent v1.0", bird_v0_vec, False),
        ("BIRD n=100, agent v1.1", bird_v1_vec, False),
    ]
    rows = []
    for label, vec, is_small in cases:
        n = len(vec)
        k = sum(vec)
        est = k / n
        lo, hi = common.bootstrap_ci(vec, n_resamples=10000, seed=42)
        row = {"case": label, "n": n, "k": k,
               "estimate": round(est, 4),
               "boot_lo": round(lo, 4), "boot_hi": round(hi, 4)}
        if is_small:
            wlo, whi = common.wilson(k, n)
            row["wilson_lo"] = round(wlo, 4)
            row["wilson_hi"] = round(whi, 4)
        rows.append(row)
    out = {"method": ("percentile bootstrap, 10,000 resamples, seed 42; "
                      "Wilson score interval (z=1.96) for n=12"),
           "rows": rows}
    path = common.write_raw("a6.json", out)
    print("wrote", path)
    print("| case | est | 95% boot CI | Wilson |")
    for r in rows:
        w = ("[%.3f, %.3f]" % (r["wilson_lo"], r["wilson_hi"])
             if "wilson_lo" in r else "-")
        print("| %s | %.3f | [%.3f, %.3f] | %s |" % (
            r["case"], r["estimate"], r["boot_lo"], r["boot_hi"], w))


if __name__ == "__main__":
    main()
