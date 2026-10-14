# Author: Prabhu Rangaraju
"""A3 (synthetic). VES-style efficiency gate on retail.db (n=12)."""

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

    agent_v10 = BaselineSQLAgent()
    v0 = verifier.evaluate_suite(agent_v10, data, db_path)
    diagnoses = {qid: (v0[qid]["failure_mode"], v0[qid]["failure_stage"],
                       v0[qid]["evidence"]) for qid in v0}
    agent_v11 = BaselineSQLAgent()
    for f in fixer.propose_fixes(diagnoses):
        fixer.apply_fix(agent_v11, f)

    out = {"agents": {}}
    for aname, agent in (("v1.0", agent_v10), ("v1.1", agent_v11)):
        rows = []
        for e in data:
            trace = agent.answer(e["question"], db_path)
            grows = common.gold_rows(db_path, e["gold_sql"])
            arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
            if common.multiset_equal(arows, grows) != 1:
                continue
            t_agent = common.median_ms(trace["s4"]["sql"], db_path, runs=5)
            t_gold = common.median_ms(e["gold_sql"], db_path, runs=5)
            ratio = t_agent / t_gold if t_gold > 0 else float("inf")
            rows.append({"qid": e["id"], "t_agent_ms": round(t_agent, 4),
                         "t_gold_ms": round(t_gold, 4),
                         "ratio": round(ratio, 4)})
        ratios = [r["ratio"] for r in rows if r["ratio"] != float("inf")]
        summary = common.dist_summary(ratios)
        summary["share_ratio_le_2"] = (
            sum(1 for x in ratios if x <= 2) / len(ratios) if ratios else 0.0)
        out["agents"][aname] = {"per_question": rows, "ratio_summary": summary}
    path = common.write_raw("a3_synth.json", out)
    print("wrote", path)
    for aname in ("v1.0", "v1.1"):
        s = out["agents"][aname]["ratio_summary"]
        print("%s: n=%d median=%.3f p90=%.3f max=%.3f share<=2=%.3f" % (
            aname, s["n"], s["median"], s["p90"], s["max"],
            s["share_ratio_le_2"]))


if __name__ == "__main__":
    main()
