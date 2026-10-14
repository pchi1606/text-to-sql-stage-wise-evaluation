# Author: Prabhu Rangaraju
"""Suite evaluation and fix verification (Qfix / Qbreak / Qnet).

evaluate_suite(agent, dataset, schemas) runs the full pipeline per question
and records per-stage metrics plus the localized diagnosis.

verify_fix(before, after) authenticates the fix: a question counts as fixed
only if it went from EX=0 to EX=1, and as broken only if it regressed from
EX=1 to EX=0. Qnet = Qfix - Qbreak. A fix that repairs nothing net is rejected
by this accounting -- following SQLENS (Qfix/Qbreak/Qnet).
"""

import sqlite3

from diagnoser import localize
from metrics import schema_linking_pr, execution_accuracy, efficiency_ms


def gold_rows(db_path, sql):
    con = sqlite3.connect(db_path)
    try:
        return [tuple(r) for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def evaluate_suite(agent, dataset, schemas):
    results = {}
    for entry in dataset:
        qid = entry["id"]
        schema = schemas[entry["db_id"]]
        trace = agent.answer(entry["question"], entry["db_id"])
        grows = gold_rows(schema.db_path, entry["gold_sql"])
        arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
        ex = execution_accuracy(arows, grows)
        link = schema_linking_pr(trace, entry["gold_tables"],
                                 entry["gold_columns"])
        mres = {"ex": ex}
        mode, stage, evidence = localize(entry, trace, mres, agent, schema)
        results[qid] = {
            "question": entry["question"],
            "db_id": entry["db_id"],
            "gold_sql": entry["gold_sql"],
            "gold_tables": entry["gold_tables"],
            "gold_columns": entry["gold_columns"],
            "agent_sql": trace["s4"]["sql"],
            "ex": ex,
            "link_f1_tables": link["tables"]["f1"],
            "link_f1_cols": link["columns"]["f1"],
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
    assert not (set(fixed) & set(broken)), "a question cannot be both fixed and broken"
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
