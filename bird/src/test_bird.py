# Author: Prabhu Rangaraju
"""Required validation tests for the BIRD arm.

Plain-function tests (no pytest dependency); run with:
    ./venv/bin/python src/test_bird.py

  1. test_gold_sql_clean_nonempty -- every sampled gold SQL executes
     without error and returns at least one row (sample validity).
  2. test_perfect_agent_ex_one -- an agent whose S4 emits the gold SQL
     scores EX=1 on all 100 questions (metric sanity: EX=1 is reachable
     and the gold SQLs are executable).
  3. test_two_run_determinism -- two full evaluate_suite runs of the
     fixed agent are byte-identical after zeroing wall-clock runtime_ms
     (documented deviation: runtime is real wall-clock and nondeterministic).
  4. test_q_assertions -- verify_fix on the shipped results: no question
     in both sets, Qnet == Qfix - Qbreak, and the validated values
     Qfix=5 / Qbreak=0 / Qnet=5.
"""

import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from bird_agent import BirdSQLAgent  # noqa: E402
from bird_verifier import evaluate_suite, verify_fix  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_bird", os.path.join(HERE, "run_bird.py"))
_rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rb)

EXPECTED_QFIX, EXPECTED_QBREAK, EXPECTED_QNET = 6, 0, 6


def db_path_of(db_id):
    return os.path.join(ROOT, "data", "minidev", "MINIDEV",
                        "dev_databases", db_id, db_id + ".sqlite")


def test_gold_sql_clean_nonempty():
    dataset, _inv = _rb.load_dataset()
    assert len(dataset) == 100, "sample must be n=100, got %d" % len(dataset)
    bad, empty = [], []
    for e in dataset:
        con = sqlite3.connect(db_path_of(e["db_id"]))
        try:
            rows = con.execute(e["gold_sql"]).fetchall()
        except Exception as exc:  # noqa: BLE001
            bad.append((e["question_id"], str(exc)[:80]))
            continue
        finally:
            con.close()
        if not rows:
            empty.append(e["question_id"])
    assert not bad, "gold SQL errors: %s" % bad
    assert not empty, "gold SQL with empty results: %s" % empty
    print("PASS test_gold_sql_clean_nonempty (100/100 clean, non-empty)")


def test_perfect_agent_ex_one():
    dataset, inv = _rb.load_dataset()
    gold = {e["question"]: e["gold_sql"] for e in dataset}
    agent = BirdSQLAgent(inv)
    agent.s4_sql_generation = (
        lambda question, intent, plan, db_id, evidence="":
        gold[question])
    results = evaluate_suite(agent, dataset, inv, db_path_of)
    bad = sorted(qid for qid, r in results.items() if r["ex"] != 1)
    assert not bad, "perfect agent EX!=1 on: %s" % bad
    print("PASS test_perfect_agent_ex_one (100/100 EX=1)")


def _canonical(results):
    canon = {}
    for qid in sorted(results):
        r = dict(results[qid])
        r["runtime_ms"] = 0  # wall-clock; nondeterministic by nature
        canon[qid] = r
    return json.dumps(canon, sort_keys=True)


def _fixed_agent(inv):
    agent = BirdSQLAgent(inv)
    v0 = json.load(open(os.path.join(ROOT, "results_v0.json")))
    diagnoses = {qid: (r["failure_mode"], r["failure_stage"],
                       r["evidence"]) for qid, r in v0.items()}
    from bird_fixer import propose_fixes, apply_fix
    for fix in propose_fixes(diagnoses):
        apply_fix(agent, fix)
    return agent


def test_two_run_determinism():
    dataset, inv = _rb.load_dataset()
    r1 = _canonical(evaluate_suite(_fixed_agent(inv), dataset, inv,
                                   db_path_of))
    r2 = _canonical(evaluate_suite(_fixed_agent(inv), dataset, inv,
                                   db_path_of))
    assert r1 == r2, "two full runs differ (excluding runtime_ms)"
    print("PASS test_two_run_determinism (byte-identical, runtime excluded)")


def test_q_assertions():
    v0 = json.load(open(os.path.join(ROOT, "results_v0.json")))
    v1 = json.load(open(os.path.join(ROOT, "results_v1.json")))
    ver = verify_fix(v0, v1)  # asserts no qid in both sets, Qnet arithmetic
    assert ver["Qnet"] == ver["Qfix"] - ver["Qbreak"]
    assert not (set(ver["fixed"]) & set(ver["broken"]))
    assert (ver["Qfix"], ver["Qbreak"], ver["Qnet"]) == (
        EXPECTED_QFIX, EXPECTED_QBREAK, EXPECTED_QNET), ver
    print("PASS test_q_assertions (Qfix=%d Qbreak=%d Qnet=%d)" %
          (ver["Qfix"], ver["Qbreak"], ver["Qnet"]))


if __name__ == "__main__":
    test_gold_sql_clean_nonempty()
    test_perfect_agent_ex_one()
    test_two_run_determinism()
    test_q_assertions()
    print("ALL BIRD TESTS PASSED")
