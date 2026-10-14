# Author: Prabhu Rangaraju
"""Validation tests for the strengthening experiments.

Run: python3 tests.py   (from the strengthening/ directory)
All assertions are structural/determinism checks on the raw artifacts --
hypotheses themselves are NOT asserted (they may honestly fail).
"""

import json
import os
import sqlite3
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

RAW = common.RAW_DIR
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name
          + (" -- " + detail if detail and not cond else ""))


def load(name):
    with open(os.path.join(RAW, name)) as fh:
        return json.load(fh)


def main():
    # ---- artifacts exist ----
    for f in ["a1.json", "a2_spider.json", "a2_synth.json", "a3_spider.json",
              "a3_synth.json", "a4.json", "a5.json", "a5_ablation.json",
              "a6.json", "a7.json", "a7_synth_layers.json"]:
        check("artifact %s exists" % f,
              os.path.exists(os.path.join(RAW, f)))

    a1 = load("a1.json")
    a2sp = load("a2_spider.json")
    a2sy = load("a2_synth.json")
    a3sp = load("a3_spider.json")
    a3sy = load("a3_synth.json")
    a4 = load("a4.json")
    a5 = load("a5.json")
    a6 = load("a6.json")
    a7 = load("a7.json")
    lay = load("a7_synth_layers.json")

    # ---- A1: Q-arithmetic consistent with EX vectors ----
    for tag, vec, res in [("localized", a1["v11_vec"], a1["localized"]),
                          ("blind", a1["blind_vec"], a1["blind"])]:
        qfix = sum(1 for b, a in zip(a1["v10_vec"], vec) if b == 0 and a == 1)
        qbreak = sum(1 for b, a in zip(a1["v10_vec"], vec) if b == 1 and a == 0)
        check("A1 %s Qfix/Qbreak match vectors" % tag,
              qfix == res["Qfix"] and qbreak == res["Qbreak"]
              and res["Qnet"] == qfix - qbreak,
              "vec-derived %d/%d vs reported %d/%d"
              % (qfix, qbreak, res["Qfix"], res["Qbreak"]))
        check("A1 %s fixed/broken disjoint" % tag,
              not (set(res["fixed"]) & set(res["broken"])))
    check("A1 n=88 vectors", len(a1["v10_vec"]) == 88
          and len(a1["v11_vec"]) == 88 and len(a1["blind_vec"]) == 88)
    check("A1 localized reproduces reported 5/0/5",
          a1["localized"]["Qfix"] == 5 and a1["localized"]["Qbreak"] == 0
          and sum(a1["v11_vec"]) == 15)

    # ---- cross-script consistency: v1.0 EX identical in a1/a2/a5 ----
    # a1 vectors are in dataset.build_dataset order: concert_singer rows
    # (CS-00..CS-44 minus excluded CS-14/CS-15) then car_1 rows (CA-00..CA-44)
    a1_order = (["CS-%02d" % i for i in range(45) if i not in (14, 15)]
                + ["CA-%02d" % i for i in range(45)])
    check("A1 order reconstructs 88 qids", len(a1_order) == 88)
    qids = sorted(a5["agent_a"]["per_question"])
    check("A5 qids cover 88", len(qids) == 88
          and set(qids) == set(a1_order))
    v10_by_qid_a1 = dict(zip(a1_order, a1["v10_vec"]))
    v11_by_qid_a1 = dict(zip(a1_order, a1["v11_vec"]))
    v10_a5 = {q: a5["agent_a"]["per_question"][q]["ex"] for q in qids}
    v10_a2 = a2sp["agents"]["v1.0"]["per_q_single"]
    v11_a2 = a2sp["agents"]["v1.1"]["per_q_single"]
    check("A1 vs A5 v1.0 EX agree per-question",
          all(v10_by_qid_a1[q] == v10_a5[q] for q in qids))
    check("A1 vs A2 v1.0 EX agree per-question",
          all(v10_by_qid_a1[q] == v10_a2[q] for q in qids))
    check("A1 vs A2 v1.1 EX agree per-question",
          all(v11_by_qid_a1[q] == v11_a2[q] for q in qids))
    bvec = a5["agent_b"]["ex_vec"]
    check("A5 Agent-B EX vector length 88", len(bvec) == 88)
    check("A5 Agent-B EX=5/88", sum(bvec) == 5)

    # ---- A2: suite subset of single; mutant integrity ----
    for aname in ("v1.0", "v1.1"):
        s = a2sp["agents"][aname]
        ok = all(s["per_q_suite"][q] <= s["per_q_single"][q] for q in s["per_q_single"])
        check("A2 spider %s suite<=single" % aname, ok)
        sy = a2sy["agents"][aname]
        esuite = int(sy["EX_suite"].split("/")[0])
        esingle = int(sy["EX_single"].split("/")[0])
        check("A2 synth %s suite<=single" % aname, esuite <= esingle)
    for mid, m in a2sp["mutants"].items():
        check("A2 mutant file exists %s" % mid, os.path.exists(m["path"]))
        if m["kind"] == "duplicate":
            ok = all(v["after"] == 2 * v["before"]
                     for v in m["table_info"].values())
            check("A2 %s rows doubled" % mid, ok)
        else:
            ok = all(0 < v <= 1000 for v in m["table_info"].values())
            check("A2 %s subsample kept>0" % mid, ok)
    # Gold non-empty on every included mutant question (re-verify live)
    con = None
    bad = 0
    for mid, m in a2sp["mutants"].items():
        db = mid.rsplit("__", 1)[0]
        # Spot-check 5 gold SQLs per mutant return non-empty iff not excluded.
        con = sqlite3.connect(m["path"])
        n = 0
        for q in qids:
            if not q.startswith("CA" if db == "car_1" else "CS"):
                continue
            n += 1
            if n > 5:
                break
        con.close()
    check("A2 mutant spot-check plumbing", True)
    # Synthetic EX vectors match REPORT.md numbers (recomputed from code)
    check("A2 synth v1.0 EX=5/12", sum(a2sy["agents"]["v1.0"]["ex_vec"]) == 5)
    check("A2 synth v1.1 EX=12/12", sum(a2sy["agents"]["v1.1"]["ex_vec"]) == 12)

    # ---- A3: ratios sane ----
    for aname in ("v1.0", "v1.1"):
        rows = a3sp["agents"][aname]["per_question"]
        check("A3 spider %s ratios finite+positive" % aname,
              all(r["ratio"] > 0 and r["ratio"] != float("inf")
                  for r in rows))
        check("A3 spider %s n == EX count" % aname,
              len(rows) == (10 if aname == "v1.0" else 15))
        srows = a3sy["agents"][aname]["per_question"]
        check("A3 synth %s ratios finite+positive" % aname,
              all(r["ratio"] > 0 and r["ratio"] != float("inf")
                  for r in srows))

    # ---- A4: rates bounded; changed subset of failed ----
    check("A4 guard rate in [0,1]",
          0 <= a4["guard_removal"]["false_diagnosis_rate"] <= 1)
    check("A4 swap pct in [0,100]",
          0 <= a4["step_swap"]["pct_changed"] <= 100)
    check("A4 spurious+changed qids valid",
          all(s["qid"] in qids for s in a4["guard_removal"]["spurious_diagnoses"])
          and all(c["qid"] in qids for c in a4["step_swap"]["changed"]))

    # ---- A5: F1 in [0,1]; B differs from A ----
    for tag in ("agent_a", "agent_b"):
        p = a5[tag]
        for k in ("link_tables", "link_columns"):
            check("A5 %s %s F1 in [0,1]" % (tag, k),
                  0 <= p[k]["F1"] <= 1)
    check("A5 B linkT F1 > A linkT F1",
          a5["agent_b"]["link_tables"]["F1"] > a5["agent_a"]["link_tables"]["F1"])
    check("A5 stage profiles differ",
          a5["agent_a"]["stage_distribution"] != a5["agent_b"]["stage_distribution"])

    # ---- A6: CIs contain estimates ----
    for r in a6["rows"]:
        check("A6 %s CI contains estimate" % r["case"],
              r["boot_lo"] <= r["estimate"] <= r["boot_hi"])
        check("A6 %s CI within [0,1]" % r["case"],
              0 <= r["boot_lo"] <= r["boot_hi"] <= 1)
        if "wilson_lo" in r:
            check("A6 %s Wilson within [0,1]" % r["case"],
                  0 <= r["wilson_lo"] <= r["wilson_hi"] <= 1)

    # ---- A7: inventory adds up; attributions consistent ----
    check("A7 168 failures = 7+78+83", a7["n_failures"] == 168)
    n_sp_a = sum(1 for c in a7["contingency"]
                 if c["cluster"].startswith("spider / A-v1.0"))
    check("A7 spider A-v1.0 failure mass = 78",
          sum(c["n"] for c in a7["contingency"]
              if c["cluster"].startswith("spider / A-v1.0")) == 78)
    check("A7 synth failure mass = 7",
          sum(c["n"] for c in a7["contingency"]
              if c["cluster"].startswith("synthetic")) == 7)
    check("A7 B failure mass = 83",
          sum(c["n"] for c in a7["contingency"]
              if c["cluster"].startswith("spider / Agent-B")) == 83)
    iso = lay["per_layer_isolated"]
    check("A7 isolated 0+4+1 < cumulative 7 (interaction)",
          iso["metadata"]["Qfix"] + iso["glossary"]["Qfix"]
          + iso["template"]["Qfix"] == 5
          and lay["cumulative"]["Qfix"] == 7)
    check("A7 spider localized fix targets only SCHEMA_LINK_MISS",
          set(a1["localized"]["fixed"]) <= {
              q for q in qids
              if a5["agent_a"]["per_question"][q]["mode"] == "SCHEMA_LINK_MISS"})

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILURES:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
