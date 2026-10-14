# Author: Prabhu Rangaraju
"""A2 (Spider). TEST-SUITE ACCURACY via mutated DB instances.

Does single-DB EX overstate correctness? For each Spider DB, generate 2
mutated instances:
  (i)  row-subsampled at 70% (seed 42). For every question, VERIFY the gold
       SQL still returns non-empty on the mutant; questions failing that
       check are excluded from that mutant's suite (documented).
  (ii) row-duplicated variant (every row doubled; INTEGER PKs get fresh
       values). Tests DISTINCT/multiset robustness.

suite-pass(q) = EX=1 on the original DB AND on every mutant instance where
the gold SQL returns non-empty. Reported for agent v1.0 and v1.1:
EX_single vs EX_suite, and the count of "accidentally right" questions
(EX_single=1 but suite-fail).
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

DB_IDS = ["concert_singer", "car_1"]


def build_schemas(db_paths):
    return {db_id: SpiderSchema(db_id, path)
            for db_id, path in db_paths.items()}


def main():
    random.seed(42)
    db_dir = os.path.join(common.SPIDER_DIR, "data", "db_extract",
                          "spider_data", "database")
    orig_paths = {
        "concert_singer": os.path.join(
            db_dir, "concert_singer", "concert_singer.sqlite"),
        "car_1": os.path.join(db_dir, "car_1", "car_1.sqlite"),
    }
    schemas = build_schemas(orig_paths)
    data, exclusions = dataset.build_dataset(schemas, parse_gold)
    assert len(data) == 88 and len(exclusions) == 2

    # ---- build mutants (deterministic) ----
    mutant_specs = {}   # mutant_db_id -> (kind, path, build_info)
    for db_id in DB_IDS:
        sub = os.path.join(common.MUTANT_DIR, db_id + "__sub70.sqlite")
        dup = os.path.join(common.MUTANT_DIR, db_id + "__dup.sqlite")
        kept = common.build_subsample_mutant(orig_paths[db_id], sub,
                                             frac=0.70, seed=42)
        counts = common.build_duplicate_mutant(orig_paths[db_id], dup)
        mutant_specs[db_id + "__sub70"] = ("subsample70", sub, kept)
        mutant_specs[db_id + "__dup"] = ("duplicate", dup, counts)
    mutant_schemas = build_schemas(
        {mid: spec[1] for mid, spec in mutant_specs.items()})
    all_schemas = dict(schemas)
    all_schemas.update(mutant_schemas)

    # ---- agents ----
    agent_v10 = SpiderBaselineAgent(all_schemas)
    v0_full = verifier.evaluate_suite(agent_v10, data, schemas)
    diagnoses = {qid: (v0_full[qid]["failure_mode"],
                       v0_full[qid]["failure_stage"],
                       v0_full[qid]["evidence"]) for qid in v0_full}
    agent_v11 = SpiderBaselineAgent(all_schemas)
    for f in fixer.propose_fixes(diagnoses):
        fixer.apply_fix(agent_v11, f)
    agents = {"v1.0": agent_v10, "v1.1": agent_v11}

    # ---- evaluate on original + mutants ----
    # db_variant: 'orig' | mutant id; per (agent, variant): per-q ex + gold_ok.
    results = {}
    gold_ok_counts = {}
    for aname, agent in agents.items():
        results[aname] = {}
        for variant, db_id in [("orig", None)] + [
                (mid, mid.rsplit("__", 1)[0]) for mid in mutant_specs]:
            per_q, gold_ok = {}, {}
            for e in data:
                real_db = e["db_id"] if variant == "orig" else None
                if variant == "orig":
                    schema = schemas[e["db_id"]]
                    mid = e["db_id"]
                else:
                    if not variant.startswith(e["db_id"] + "__"):
                        continue
                    schema = mutant_schemas[variant]
                    mid = variant
                grows = common.gold_rows(schema.db_path, e["gold_sql"])
                ok = len(grows) > 0
                gold_ok[e["id"]] = ok
                if ok:
                    trace = agent.answer(e["question"], mid)
                    arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
                    per_q[e["id"]] = common.multiset_equal(arows, grows)
                else:
                    per_q[e["id"]] = None  # excluded from this variant
            results[aname][variant] = per_q
            gold_ok_counts.setdefault(aname, {})[variant] = sum(
                1 for v in gold_ok.values() if v)

    # ---- suite pass ----
    out = {"n": 88, "mutants": {}, "agents": {}}
    for mid, (kind, path, info) in mutant_specs.items():
        out["mutants"][mid] = {"kind": kind, "path": path,
                               "table_info": info}
    for aname in agents:
        per_q_suite, per_q_single, accidental = {}, {}, []
        variants = ["orig"] + list(mutant_specs)
        for e in data:
            qid = e["id"]
            single = results[aname]["orig"][qid]
            per_q_single[qid] = single
            suite = single == 1
            for v in variants[1:]:
                if not v.startswith(e["db_id"] + "__"):
                    continue
                exv = results[aname][v][qid]
                if exv is None:
                    continue  # gold empty on this mutant: not counted
                if exv != 1:
                    suite = False
            per_q_suite[qid] = 1 if suite else 0
            if single == 1 and not suite:
                accidental.append(qid)
        ex_single = sum(per_q_single.values())
        ex_suite = sum(per_q_suite.values())
        out["agents"][aname] = {
            "EX_single": "%d/88" % ex_single,
            "EX_suite": "%d/88" % ex_suite,
            "accidentally_right": accidental,
            "n_accidentally_right": len(accidental),
            "per_q_single": per_q_single,
            "per_q_suite": per_q_suite,
        }
        # Per-mutant exclusion documentation (gold-empty counts)
        excl = {}
        for v in variants[1:]:
            db = v.rsplit("__", 1)[0]
            n_ex = sum(1 for e in data if e["db_id"] == db
                       and results[aname][v][e["id"]] is None)
            excl[v] = n_ex
        out["agents"][aname]["gold_empty_exclusions_per_mutant"] = excl
    out["notes"] = (
        "suite-pass(q) = EX=1 on original AND on every mutant where that "
        "question's gold SQL returns non-empty. Subsample mutants use "
        "random.Random(42), 70% row retention per table. Duplicate mutants "
        "double every row (fresh INTEGER PK values).")
    path = common.write_raw("a2_spider.json", out)
    print("wrote", path)
    for aname in agents:
        a = out["agents"][aname]
        print("%s: EX_single=%s EX_suite=%s accidentally_right=%d %s" % (
            aname, a["EX_single"], a["EX_suite"],
            a["n_accidentally_right"], a["accidentally_right"]))
        print("   gold-empty exclusions:", a["gold_empty_exclusions_per_mutant"])


if __name__ == "__main__":
    main()
