# Author: Prabhu Rangaraju
"""Diagnose WHY Agent-B failures land on S5 EXECUTION_ERROR (mechanism check)."""

import os
import random
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

sys.path.insert(0, common.SPIDER_DIR)
os.chdir(common.SPIDER_DIR)

import importlib.util

import dataset  # noqa: E402
from diagnoser import localize  # noqa: E402
from goldparse import parse_gold  # noqa: E402
from schema import SpiderSchema  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "agentb_mod", os.path.join(STRENGTH_DIR, "exp_a5_agentb.py"))
agentb_mod = importlib.util.module_from_spec(spec)
sys.modules["agentb_mod"] = agentb_mod
spec.loader.exec_module(agentb_mod)


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
    agent = agentb_mod.AgentB(schemas)
    shown = 0
    for e in data:
        schema = schemas[e["db_id"]]
        trace = agent.answer(e["question"], e["db_id"])
        grows = common.gold_rows(schema.db_path, e["gold_sql"])
        arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
        ex = common.multiset_equal(arows, grows)
        mode, stage, _ev = localize(e, trace, {"ex": ex}, agent, schema)
        if mode == "EXECUTION_ERROR" and shown < 3:
            shown += 1
            print("=====", e["id"], e["question"][:90])
            print("s2 tables:", trace["s2"]["tables"])
            print("s3 plan tables:", trace["s3"]["tables"],
                  "pruned:", trace["s3"]["pruned"])
            print("gold tables:", e["gold_tables"])
            print("SQL:", trace["s4"]["sql"][:300])
            print("error:", trace["s5"]["error"][:160])


if __name__ == "__main__":
    main()
