# Author: Prabhu Rangaraju
"""Suite evaluation and fix verification (Qfix / Qbreak / Qnet) for the
BIRD arm. Mirrors the prototype's verifier.py: evaluate_suite runs the full
6-stage pipeline per question (routing each question to its own database),
and verify_fix authenticates the fix with the same assertions (no question in
both sets; Qnet arithmetic). Follows SQLENS (Qfix/Qbreak/Qnet).
"""

import sqlite3

import bird_diagnoser as diagnoser
from bird_metrics import (schema_linking_pr, execution_accuracy,
                          efficiency_ms, semantic_conformance)


def gold_rows(db_path, sql):
    con = sqlite3.connect(db_path)
    try:
        return [tuple(r) for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def evaluate_suite(agent, dataset, inventory, db_path_of):
    results = {}
    for entry in dataset:
        qid = "Q%d" % entry["question_id"]
        db_path = db_path_of(entry["db_id"])
        trace = agent.answer(entry["question"], entry["evidence"],
                             entry["db_id"], db_path)
        grows = gold_rows(db_path, entry["gold_sql"])
        arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
        ex = execution_accuracy(arows, grows)
        link = schema_linking_pr(trace, entry["gold_tables"],
                                 entry["gold_columns"])
        conf = semantic_conformance(trace["s4"]["sql"],
                                    entry["gold_definitions"],
                                    agent.glossary)
        mres = {"ex": ex, "conformance": conf}
        mode, stage, evidence = diagnoser.localize(
            entry, trace, mres, agent, inventory[entry["db_id"]]["tables"])
        results[qid] = {
            "question_id": entry["question_id"],
            "db_id": entry["db_id"],
            "question": entry["question"],
            "gold_sql": entry["gold_sql"],
            "agent_sql": trace["s4"]["sql"],
            "ex": ex,
            "link_f1_tables": link["tables"]["f1"],
            "link_f1_cols": link["columns"]["f1"],
            "conformance": conf["score"],
            "runtime_ms": efficiency_ms(trace),
            "failure_mode": mode,
            "failure_stage": stage,
            "evidence": evidence,
            "agent_version": trace["agent_version"],
        }
    return results


def verify_fix(before, after):
    fixed, broken = [], []
    for qid in before:
        b, a = before[qid]["ex"], after[qid]["ex"]
        if b == 0 and a == 1:
            fixed.append(qid)
        elif b == 1 and a == 0:
            broken.append(qid)
    assert not (set(fixed) & set(broken)), \
        "a question cannot be both fixed and broken"
    qfix, qbreak = len(fixed), len(broken)
    qnet = qfix - qbreak
    assert qnet == qfix - qbreak
    return {
        "Qfix": qfix,
        "Qbreak": qbreak,
        "Qnet": qnet,
        "fixed": sorted(fixed),
        "broken": sorted(broken),
    }
