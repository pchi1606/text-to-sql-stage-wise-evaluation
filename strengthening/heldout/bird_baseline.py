# Author: Prabhu Rangaraju
"""Held-out repair evaluation -- BIRD STEP A: baseline + dev-only diagnosis.

Runs v1.0 baseline on dev and held-out separately; diagnoses DEV failures
only. Writes bird_stageA.json (baseline EX per split + dev failure evidence:
S2 gap tokens, S3 missing-concept names) for inspection BEFORE any fix is
authored. Held-out baseline EX is recorded but held-out failures are NOT
diagnosed.

Run: cd <bird_dir> && venv/bin/python <heldout>/bird_baseline.py
"""

import ast
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BIRD_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "bird"))
SRC = os.path.join(BIRD_DIR, "src")

sys.path.insert(0, SRC)
os.chdir(BIRD_DIR)

from schema_inventory import db_path
from gold_parse import derive_gold_tables, derive_gold_columns
from bird_glossary import gold_definitions
from bird_agent import BirdSQLAgent
import bird_verifier as verifier


def load_dataset():
    inv = json.load(open(os.path.join(BIRD_DIR, "schema_inventory.json")))
    sample = json.load(open(os.path.join(BIRD_DIR, "sample.json")))
    dataset = []
    for row in sorted(sample, key=lambda r: r["question_id"]):
        db = inv[row["db_id"]]
        gtables = derive_gold_tables(row["SQL"], db)
        dataset.append({
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row["evidence"],
            "difficulty": row["difficulty"],
            "gold_sql": row["SQL"],
            "gold_tables": gtables,
            "gold_columns": derive_gold_columns(row["SQL"], db, gtables),
            "gold_definitions": gold_definitions(row["SQL"], row["db_id"]),
        })
    return dataset, inv


def parse_gap_tokens(evidence):
    m = re.search(r"unresolved term\(s\) (\[.*?\])", evidence)
    if not m:
        return []
    try:
        return list(ast.literal_eval(m.group(1)))
    except Exception:
        return []


def parse_missing_concepts(evidence):
    return re.findall(r"concept '([A-Z_]+)' has no governed definition",
                      evidence)


def main():
    split = json.load(open(os.path.join(HERE, "bird_split.json")))
    dev_ids = set(split["dev"])
    held_ids = set(split["heldout"])

    dataset, inv = load_dataset()
    db_path_of = lambda db_id: db_path(db_id)  # noqa: E731
    qid_of = lambda e: "Q%d" % e["question_id"]
    dev_data = [e for e in dataset if qid_of(e) in dev_ids]
    held_data = [e for e in dataset if qid_of(e) in held_ids]
    assert len(dev_data) == 50 and len(held_data) == 50
    ev_map = {qid_of(e): (e["evidence"] or "") for e in dev_data}

    agent = BirdSQLAgent(inv)  # v1.0
    v0_dev = verifier.evaluate_suite(agent, dev_data, inv, db_path_of)
    v0_held = verifier.evaluate_suite(agent, held_data, inv, db_path_of)

    ex_dev = sum(r["ex"] for r in v0_dev.values())
    ex_held = sum(r["ex"] for r in v0_held.values())

    # DEV-ONLY diagnosis.
    dev_stage = Counter(r["failure_stage"] for r in v0_dev.values()
                        if r["failure_stage"])
    dev_mode = Counter(r["failure_mode"] for r in v0_dev.values())
    dev_s2_gaps = Counter()
    dev_s2_qids = []
    dev_missing_concepts = Counter()
    dev_s3_qids = []
    dev_s6_qids = []
    dev_s4_qids = []
    dev_s2_evidence = {}
    dev_s3_evidence = {}
    for qid, r in sorted(v0_dev.items()):
        mode = r["failure_mode"]
        if mode == "SCHEMA_LINK_MISS":
            dev_s2_qids.append(qid)
            dev_s2_evidence[qid] = r["evidence"]
            for t in parse_gap_tokens(r["evidence"]):
                dev_s2_gaps[t] += 1
        elif mode == "SEMANTIC_DEFINITION_VIOLATION":
            dev_s3_qids.append(qid)
            dev_s3_evidence[qid] = r["evidence"]
            for c in parse_missing_concepts(r["evidence"]):
                dev_missing_concepts[c] += 1
        elif mode == "AGG_FILTER_ERROR":
            dev_s4_qids.append(qid)
        elif mode in ("WRONG_RESULT", "EMPTY_RESULT"):
            dev_s6_qids.append(qid)

    # Which dev S6 questions carry BIRD evidence definitions?
    dev_s6_with_evidence = [qid for qid in dev_s6_qids
                            if ev_map.get(qid, "").strip()]

    out = {
        "split": {"dev": sorted(dev_ids), "heldout": sorted(held_ids)},
        "baseline_ex_dev": ex_dev,
        "baseline_ex_heldout": ex_held,
        "n_dev": 50, "n_heldout": 50,
        "dev_failure_stage_dist": dict(dev_stage),
        "dev_failure_mode_dist": dict(dev_mode),
        "dev_s2_failure_qids": dev_s2_qids,
        "dev_s2_gap_tokens": dict(dev_s2_gaps),
        "dev_s2_evidence": dev_s2_evidence,
        "dev_s3_qids": dev_s3_qids,
        "dev_s3_missing_concepts": dict(dev_missing_concepts),
        "dev_s3_evidence": dev_s3_evidence,
        "dev_s4_qids": dev_s4_qids,
        "dev_s6_qids": dev_s6_qids,
        "dev_s6_with_evidence_count": len(dev_s6_with_evidence),
    }
    with open(os.path.join(HERE, "bird_stageA.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print("baseline EX dev: %d/50" % ex_dev)
    print("baseline EX heldout: %d/50" % ex_held)
    print("dev failure stages:", dict(dev_stage))
    print("dev failure modes:", dict(dev_mode))
    print("dev S2 failures: %d" % len(dev_s2_qids))
    print("dev S2 gap tokens (freq):")
    for tok, c in dev_s2_gaps.most_common(40):
        print("  %-18s %d" % (tok, c))
    print("dev S3 missing concepts:", dict(dev_missing_concepts))
    print("dev S6 with evidence: %d/%d" % (len(dev_s6_with_evidence),
                                           len(dev_s6_qids)))


if __name__ == "__main__":
    main()
