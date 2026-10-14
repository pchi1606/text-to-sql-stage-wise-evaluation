# Author: Prabhu Rangaraju
"""A2 (synthetic). Test-suite accuracy on retail.db (n=12).

Same mutant protocol as the Spider arm: 70% row-subsample (seed 42) and
row-duplicated variant; suite-pass = EX=1 on original AND all valid mutants.
Also saves the v1.0/v1.1 EX vectors for the bootstrap experiment (A6).
"""

import os
import random
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

sys.path.insert(0, common.SYNTH_DIR)
os.chdir(common.SYNTH_DIR)

import dataset  # noqa: E402
import fixer  # noqa: E402
import verifier  # noqa: E402
from agent import BaselineSQLAgent  # noqa: E402


def main():
    random.seed(42)
    db_path = os.path.join(common.SYNTH_DIR, "retail.db")
    data = dataset.DATASET
    assert len(data) == 12

    sub = os.path.join(common.MUTANT_DIR, "retail__sub70.sqlite")
    dup = os.path.join(common.MUTANT_DIR, "retail__dup.sqlite")
    kept = common.build_subsample_mutant(db_path, sub, frac=0.70, seed=42)
    counts = common.build_duplicate_mutant(db_path, dup)
    variants = {"orig": db_path, "sub70": sub, "dup": dup}

    agent_v10 = BaselineSQLAgent()
    v0 = verifier.evaluate_suite(agent_v10, data, db_path)
    diagnoses = {qid: (v0[qid]["failure_mode"], v0[qid]["failure_stage"],
                       v0[qid]["evidence"]) for qid in v0}
    fixes = fixer.propose_fixes(diagnoses)
    agent_v11 = BaselineSQLAgent()
    for f in fixes:
        fixer.apply_fix(agent_v11, f)
    v1 = verifier.evaluate_suite(agent_v11, data, db_path)
    agents = {"v1.0": agent_v10, "v1.1": agent_v11}

    out = {"n": 12, "subsample_kept": kept, "duplicate_counts": counts,
           "agents": {}}
    for aname, agent in agents.items():
        per_variant = {}
        for vname, vpath in variants.items():
            per_q, gold_ok = {}, {}
            for e in data:
                grows = common.gold_rows(vpath, e["gold_sql"])
                ok = len(grows) > 0
                gold_ok[e["id"]] = ok
                if ok:
                    trace = agent.answer(e["question"], vpath)
                    arows = (trace["s5"]["rows"]
                             if trace["s5"]["ok"] else [])
                    per_q[e["id"]] = common.multiset_equal(arows, grows)
                else:
                    per_q[e["id"]] = None
            per_variant[vname] = per_q
            out.setdefault("gold_empty_exclusions", {})[
                "%s/%s" % (aname, vname)] = sum(
                    1 for v in gold_ok.values() if not v)
        suite, single, accidental = {}, {}, []
        for e in data:
            qid = e["id"]
            single[qid] = per_variant["orig"][qid]
            s = single[qid] == 1
            for vname in ("sub70", "dup"):
                exv = per_variant[vname][qid]
                if exv is None:
                    continue
                if exv != 1:
                    s = False
            suite[qid] = 1 if s else 0
            if single[qid] == 1 and not s:
                accidental.append(qid)
        ex_single = sum(single.values())
        ex_suite = sum(suite.values())
        out["agents"][aname] = {
            "EX_single": "%d/12" % ex_single,
            "EX_suite": "%d/12" % ex_suite,
            "accidentally_right": accidental,
            "n_accidentally_right": len(accidental),
            "ex_vec": [single[e["id"]] for e in data],
        }
    path = common.write_raw("a2_synth.json", out)
    print("wrote", path)
    for aname in agents:
        a = out["agents"][aname]
        print("%s: EX_single=%s EX_suite=%s accidentally_right=%d %s" % (
            aname, a["EX_single"], a["EX_suite"],
            a["n_accidentally_right"], a["accidentally_right"]))
    print("gold-empty exclusions:", out["gold_empty_exclusions"])


if __name__ == "__main__":
    main()
