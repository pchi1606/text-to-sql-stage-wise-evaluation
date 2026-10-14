# Author: Prabhu Rangaraju
"""Ablation: how much of Agent-B's S2 F1 gain comes from value matching
vs stemming+synonyms alone? Runs B with the literal-matching step disabled.
"""

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
from goldparse import parse_gold  # noqa: E402
from metrics import schema_linking_pr  # noqa: E402
from schema import SpiderSchema  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "agentb_mod", os.path.join(STRENGTH_DIR, "exp_a5_agentb.py"))
agentb_mod = importlib.util.module_from_spec(spec)
sys.modules["agentb_mod"] = agentb_mod
spec.loader.exec_module(agentb_mod)


class AgentBNoValue(agentb_mod.AgentB):
    def s2_schema_linking(self, question, intent, schema):
        # Skip the literal-matching augmentation: pure stemming+synonyms S2.
        from agent import SpiderBaselineAgent
        return SpiderBaselineAgent.s2_schema_linking(
            self, question, intent, schema)


def f1(agent, data, schemas):
    t, c = [], []
    for e in data:
        trace = agent.answer(e["question"], e["db_id"])
        link = schema_linking_pr(trace, e["gold_tables"], e["gold_columns"])
        t.append(link["tables"]["f1"])
        c.append(link["columns"]["f1"])
    return sum(t) / len(t), sum(c) / len(c)


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
    ft, fc = f1(agentb_mod.AgentB(schemas), data, schemas)
    nt, nc = f1(AgentBNoValue(schemas), data, schemas)
    print("Agent-B            linkT F1=%.4f linkC F1=%.4f" % (ft, fc))
    print("Agent-B no-literal linkT F1=%.4f linkC F1=%.4f" % (nt, nc))
    print("delta from value matching: dT=%+.4f dC=%+.4f" % (ft - nt, fc - nc))
    common.write_raw("a5_ablation.json", {
        "agent_b": {"linkT_F1": round(ft, 4), "linkC_F1": round(fc, 4)},
        "agent_b_no_literal": {"linkT_F1": round(nt, 4),
                               "linkC_F1": round(nc, 4)},
    })


if __name__ == "__main__":
    main()
