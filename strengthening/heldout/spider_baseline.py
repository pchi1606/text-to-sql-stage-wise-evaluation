# Author: Prabhu Rangaraju
"""Held-out repair evaluation -- Spider STEP A: baseline + dev-only diagnosis.

Runs v1.0 baseline on dev and held-out separately; diagnoses DEV failures
only. Writes spider_stageA.json (baseline EX per split + dev failure
evidence, including S2 gap tokens) for inspection BEFORE any fix is authored.
Held-out baseline EX is recorded but held-out failures are NOT diagnosed.

Run: cd <spider_dir> && venv/bin/python <heldout>/spider_baseline.py
"""

import ast
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "spider"))

sys.path.insert(0, SPIDER_DIR)
os.chdir(SPIDER_DIR)

import dataset as spider_dataset
import verifier
from agent import SpiderBaselineAgent
from goldparse import parse_gold
from schema import SpiderSchema


def build():
    schemas = {
        "concert_singer": SpiderSchema(
            "concert_singer",
            os.path.join(SPIDER_DIR, "data", "db_extract", "spider_data",
                         "database", "concert_singer", "concert_singer.sqlite")),
        "car_1": SpiderSchema(
            "car_1",
            os.path.join(SPIDER_DIR, "data", "db_extract", "spider_data",
                         "database", "car_1", "car_1.sqlite")),
    }
    data, _excl = spider_dataset.build_dataset(schemas, parse_gold)
    return schemas, data


def parse_gap_tokens(evidence):
    m = re.search(r"unresolved term\(s\) (\[.*?\])", evidence)
    if not m:
        return []
    try:
        return list(ast.literal_eval(m.group(1)))
    except Exception:
        return []


def main():
    split = json.load(open(os.path.join(HERE, "spider_split.json")))
    dev_ids = set(split["dev"])
    held_ids = set(split["heldout"])

    schemas, data = build()
    dev_data = [e for e in data if e["id"] in dev_ids]
    held_data = [e for e in data if e["id"] in held_ids]
    assert len(dev_data) == 44 and len(held_data) == 44

    agent = SpiderBaselineAgent(schemas)  # v1.0
    v0_dev = verifier.evaluate_suite(agent, dev_data, schemas)
    v0_held = verifier.evaluate_suite(agent, held_data, schemas)

    ex_dev = sum(r["ex"] for r in v0_dev.values())
    ex_held = sum(r["ex"] for r in v0_held.values())

    # DEV-ONLY diagnosis.
    dev_stage = Counter(r["failure_stage"] for r in v0_dev.values()
                        if r["failure_stage"])
    dev_mode = Counter(r["failure_mode"] for r in v0_dev.values())
    dev_s2_gaps = Counter()
    dev_s2_qids = []
    for qid, r in sorted(v0_dev.items()):
        if r["failure_mode"] == "SCHEMA_LINK_MISS":
            dev_s2_qids.append(qid)
            for t in parse_gap_tokens(r["evidence"]):
                dev_s2_gaps[t] += 1

    out = {
        "split": {"dev": sorted(dev_ids), "heldout": sorted(held_ids)},
        "baseline_ex_dev": ex_dev,
        "baseline_ex_heldout": ex_held,
        "n_dev": 44, "n_heldout": 44,
        "dev_failure_stage_dist": dict(dev_stage),
        "dev_failure_mode_dist": dict(dev_mode),
        "dev_s2_failure_qids": dev_s2_qids,
        "dev_s2_gap_tokens": dict(dev_s2_gaps),
        "dev_s2_evidence": {qid: v0_dev[qid]["evidence"]
                            for qid in dev_s2_qids},
    }
    with open(os.path.join(HERE, "spider_stageA.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("baseline EX dev: %d/44" % ex_dev)
    print("baseline EX heldout: %d/44" % ex_held)
    print("dev failure stages:", dict(dev_stage))
    print("dev S2 failures: %d" % len(dev_s2_qids))
    print("dev S2 gap tokens (freq):")
    for tok, c in dev_s2_gaps.most_common():
        print("  %-18s %d" % (tok, c))


if __name__ == "__main__":
    main()
