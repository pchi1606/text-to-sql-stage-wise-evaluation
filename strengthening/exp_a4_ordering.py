# Author: Prabhu Rangaraju
"""A4. DIAGNOSER ORDERING ABLATION (Spider v1.0 failures).

(a) Guard value: remove the "EX==1 -> no failure" guard (implemented by
    calling diagnoser.localize with mres ex=0 -- the guard is the only use
    of mres["ex"] in localize) and count spurious diagnoses on the 10
    EX=1 questions. Reports the false-diagnosis rate the guard prevents.

(b) Step-swap: the Spider diagnoser has NO missing-definition/conformance
    steps (no governed glossary for Spider DBs); its actual adjacent pair
    after the EX guard is lexical-gap->S2 vs join-coverage->S3. The swap
    implemented here is join-coverage BEFORE lexical-gap. Report the % of
    the 78 v1.0 failures whose attributed (mode, stage) changes.
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
import verifier  # noqa: E402
from diagnoser import localize, _gap_tokens  # noqa: E402
from agent import SpiderBaselineAgent  # noqa: E402
from goldparse import parse_gold  # noqa: E402
from schema import SpiderSchema  # noqa: E402


def localize_swapped(entry, trace, mres, agent, schema):
    """diagnoser.localize with the join-coverage check moved BEFORE the
    lexical-gap check. Everything else identical."""
    s5 = trace["s5"]
    intent = trace["s1"]
    sql = trace["s4"]["sql"].lower()

    if s5["error"]:
        return ("EXECUTION_ERROR", "S5", "SQLite raised %s" % s5["error"])

    if mres["ex"] == 1:
        return ("no failure", None, "EX=1: agent result matches gold")

    # SWAPPED: join coverage first ...
    plan_tables = set(trace["s3"]["tables"])
    missing_tables = sorted(set(entry["gold_tables"]) - plan_tables)
    if missing_tables:
        return (
            "JOIN_ERROR", "S3",
            "plan tables %s omit required %s"
            % (sorted(plan_tables), missing_tables),
        )

    # ... then lexical gaps.
    gaps = _gap_tokens(entry["question"], agent, schema)
    if gaps:
        return (
            "SCHEMA_LINK_MISS", "S2",
            "unresolved term(s) %s: matched no table or column name token "
            "(exact overlap, no stemming/synonyms)" % sorted(set(gaps)),
        )

    problems = []
    for func, (t, c) in intent.get("aggs", []):
        if func.lower() not in sql or c.lower() not in sql:
            problems.append("agg %s(%s) not realized in SQL" % (func, c))
    for (t, c), op, _v in intent.get("filters", []):
        if c.lower() not in sql:
            problems.append("filter column %s.%s missing from SQL" % (t, c))
        elif op in (">", "<", ">=", "<=", "=") and op not in sql \
                and "like" not in sql:
            problems.append("filter comparator for %s.%s not realized"
                            % (t, c))
    if intent.get("group_word") and "_group_col" in intent:
        g = intent["_group_col"]
        if "group by" not in sql:
            problems.append("group-by on %s.%s missing" % (g[0], g[1]))
    if problems:
        return ("AGG_FILTER_ERROR", "S4", "; ".join(problems))

    if s5["n_rows"] == 0:
        return ("EMPTY_RESULT", "S6",
                "agent returned 0 rows while gold is non-empty")

    return ("WRONG_RESULT", "S6",
            "result mismatch with no earlier-stage evidence")


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
    agent = SpiderBaselineAgent(schemas)

    # Traces for every question (deterministic agent)
    traces, ex = {}, {}
    for e in data:
        trace = agent.answer(e["question"], e["db_id"])
        traces[e["id"]] = (e, trace)
        grows = common.gold_rows(schemas[e["db_id"]].db_path, e["gold_sql"])
        arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
        ex[e["id"]] = common.multiset_equal(arows, grows)

    correct = sorted(q for q in ex if ex[q] == 1)
    failed = sorted(q for q in ex if ex[q] == 0)
    assert len(correct) == 10 and len(failed) == 78

    # (a) guard removed: localize with ex forced to 0.
    spurious = []
    for qid in correct:
        e, trace = traces[qid]
        mode, stage, _ev = localize(e, trace, {"ex": 0}, agent,
                                   schemas[e["db_id"]])
        if stage is not None:
            spurious.append({"qid": qid, "mode": mode, "stage": stage})

    # (b) swapped order on the 78 failures.
    changed = []
    for qid in failed:
        e, trace = traces[qid]
        schema = schemas[e["db_id"]]
        m0, s0, _ = localize(e, trace, {"ex": 0}, agent, schema)
        m1, s1, _ = localize_swapped(e, trace, {"ex": 0}, agent, schema)
        if (m0, s0) != (m1, s1):
            changed.append({"qid": qid,
                            "orig": [m0, s0], "swapped": [m1, s1]})

    out = {
        "n_correct": len(correct),
        "n_failed": len(failed),
        "guard_removal": {
            "spurious_diagnoses": spurious,
            "n_spurious": len(spurious),
            "false_diagnosis_rate": len(spurious) / len(correct),
            "method": ("diagnoser.localize called with mres ex=0 on the 10 "
                       "EX=1 questions; mres['ex'] is only used by the guard, "
                       "so this exactly removes the guard."),
        },
        "step_swap": {
            "swapped": "join-coverage (S3) check moved BEFORE lexical-gap (S2) check",
            "note": ("Spider diagnoser has no missing-definition/conformance "
                     "steps; the actual adjacent pair is lexical-gap<->"
                     "join-coverage."),
            "n_changed": len(changed),
            "pct_changed": 100.0 * len(changed) / len(failed),
            "changed": changed,
        },
    }
    path = common.write_raw("a4.json", out)
    print("wrote", path)
    print("(a) guard removed: %d/%d spurious diagnoses (rate %.2f)" % (
        len(spurious), len(correct), len(spurious) / len(correct)))
    for s in spurious:
        print("    %s -> %s/%s" % (s["qid"], s["mode"], s["stage"]))
    print("(b) step swap: %d/%d attributions changed (%.1f%%)" % (
        len(changed), len(failed), 100.0 * len(changed) / len(failed)))
    for c in changed:
        print("    %s: %s/%s -> %s/%s" % (
            c["qid"], c["orig"][0], c["orig"][1],
            c["swapped"][0], c["swapped"][1]))


if __name__ == "__main__":
    main()
