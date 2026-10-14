# Author: Prabhu Rangaraju
"""A3 (Spider). VES-style efficiency gate.

For every question with EX=1 (agent v1.0 and v1.1): time the agent SQL and
the gold SQL (median of 5 runs each, fresh sqlite connection per run, same
discipline for both), ratio = t_agent / t_gold. Report the ratio
distribution (median, p90, max) and the share of EX=1 answers with ratio <= 2
("efficient"). Framed as a gate: correctness alone vs correctness+efficiency.
"""

import os
import random
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

sys.path.insert(0, common.SPIDER_DIR)
os.chdir(common.SPIDER_DIR)

import dataset  # noqa: E402
import fixer  # noqa: E402
import verifier  # noqa: E402
from agent import SpiderBaselineAgent  # noqa: E402
from goldparse import parse_gold  # noqa: E402
from schema import SpiderSchema  # noqa: E402


def main():
    random.seed(42)
    db_dir = os.path.join(common.SPIDER_DIR, "data", "db_extract",
                          "spider_data", "database")
    schemas = {
        "concert_singer": SpiderSchema(
            "concert_singer",
            os.path.join(db_dir, "concert_singer", "concert_singer.sqlite")),
        "car_1": SpiderSchema(
            "car_1", os.path.join(db_dir, "car_1", "car_1.sqlite")),
    }
    data, _ = dataset.build_dataset(schemas, parse_gold)

    agent_v10 = SpiderBaselineAgent(schemas)
    v0 = verifier.evaluate_suite(agent_v10, data, schemas)
    diagnoses = {qid: (v0[qid]["failure_mode"], v0[qid]["failure_stage"],
                       v0[qid]["evidence"]) for qid in v0}
    agent_v11 = SpiderBaselineAgent(schemas)
    for f in fixer.propose_fixes(diagnoses):
        fixer.apply_fix(agent_v11, f)

    out = {"agents": {}, "notes": (
        "median of 5 runs each; fresh sqlite3 connection per run for both "
        "agent and gold SQL; times in ms; DBs are tiny so absolute times "
        "are noisy -- the ratio is the reported quantity.")}
    for aname, agent in (("v1.0", agent_v10), ("v1.1", agent_v11)):
        rows = []
        for e in data:
            trace = agent.answer(e["question"], e["db_id"])
            grows = common.gold_rows(schemas[e["db_id"]].db_path,
                                     e["gold_sql"])
            arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
            if common.multiset_equal(arows, grows) != 1:
                continue
            db_path = schemas[e["db_id"]].db_path
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
        summary["n_inf"] = len(rows) - len(ratios)
        out["agents"][aname] = {"per_question": rows, "ratio_summary": summary}
    path = common.write_raw("a3_spider.json", out)
    print("wrote", path)
    for aname in ("v1.0", "v1.1"):
        s = out["agents"][aname]["ratio_summary"]
        print("%s: n=%d median=%.3f p90=%.3f max=%.3f share<=2=%.3f" % (
            aname, s["n"], s["median"], s["p90"], s["max"],
            s["share_ratio_le_2"]))


if __name__ == "__main__":
    main()
