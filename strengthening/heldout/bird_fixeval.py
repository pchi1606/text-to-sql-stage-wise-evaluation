# Author: Prabhu Rangaraju
"""Held-out repair evaluation -- BIRD STEP B: dev-derived fix -> re-eval.

Dev-derived fix (authored from bird_stageA.json ONLY; held-out blinded).
Dev baseline: S2=27, S3=11, S4=7, S6=5 (2 EMPTY_RESULT + 3 WRONG_RESULT).

  metadata: dev S2 gap tokens re-derive only
      diagnosed->diagnosis, women->female.
      EXCLUDED (absent from dev gaps): pupils, enrollees, enrolees,
      testtakers, examined, customers, woman.
      No NEW synonym elements: remaining dev gaps are values/typos/formula
      words with no unambiguous schema referent.
  glossary: dev S3 evidence names missing concepts LOAN_RUNNING_OK (x2) and
      AVG_SALARY (x2); dev S4 failures present -> enable enforcement.
      EXCLUDED: LOAN_PAID_OK (no dev failure evidences it).
  template: 5/5 dev S6 questions carry BIRD evidence definitions that the
      baseline never parses -> enable evidence-predicate parsing (general
      mechanism, same as original).

Applies the fix to a FRESH agent; re-runs on dev AND held-out; verifies
Qfix/Qbreak/Qnet per split. PRIMARY: held-out R+/R-/RΔ. Also runs dev
single-fix ablations (metadata/glossary/template alone).

Run: cd <bird_dir> && venv/bin/python <heldout>/bird_fixeval.py
"""

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIRD_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "bird"))
SRC = os.path.join(BIRD_DIR, "src")

sys.path.insert(0, SRC)
os.chdir(BIRD_DIR)

from schema_inventory import db_path
from gold_parse import derive_gold_tables, derive_gold_columns
from bird_glossary import gold_definitions, CATALOG
from bird_agent import BirdSQLAgent
import bird_verifier as verifier

# ---- dev-derived fix spec (from bird_stageA.json only) --------------------
DEV_FIX = {
    "synonyms": {
        "diagnosed": "diagnosis",   # re-derived (dev S2 gap x1)
        "women": "female",           # re-derived (dev S2 gap x1)
        # EXCLUDED: pupils, enrollees, enrolees, testtakers, examined,
        # customers, woman -- none occur as gap tokens in dev failures.
    },
    "glossary_concepts": ["LOAN_RUNNING_OK", "AVG_SALARY"],
    # EXCLUDED: LOAN_PAID_OK -- no dev failure evidences it.
    "use_glossary_enforcement": True,
    "use_evidence_predicates": True,
    "version": "1.3-dev",
}


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


def fresh_agent(inv, metadata=True, glossary=True, template=True):
    ag = BirdSQLAgent(inv)
    if metadata:
        ag.synonyms.update(DEV_FIX["synonyms"])
    if glossary:
        for c in DEV_FIX["glossary_concepts"]:
            ag.glossary[c] = CATALOG[c]
        ag.glossary_version = "1.1-dev"
        ag.use_glossary_enforcement = DEV_FIX["use_glossary_enforcement"]
    if template:
        ag.use_evidence_predicates = DEV_FIX["use_evidence_predicates"]
    ag.version = DEV_FIX["version"]
    return ag


def canonical(results):
    canon = {}
    for qid in sorted(results):
        r = dict(results[qid])
        r["runtime_ms"] = 0  # wall-clock; nondeterministic by nature
        canon[qid] = json.loads(json.dumps(r, sort_keys=True, default=str))
    return json.dumps(canon, sort_keys=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


def run_once(dataset, inv, dev_data, held_data):
    db_path_of = lambda db_id: db_path(db_id)  # noqa: E731
    base = BirdSQLAgent(inv)  # v1.0
    v0_dev = verifier.evaluate_suite(base, dev_data, inv, db_path_of)
    v0_held = verifier.evaluate_suite(base, held_data, inv, db_path_of)

    full = fresh_agent(inv)
    v1_dev = verifier.evaluate_suite(full, dev_data, inv, db_path_of)
    v1_held = verifier.evaluate_suite(full, held_data, inv, db_path_of)
    ver_dev = verifier.verify_fix(v0_dev, v1_dev)
    ver_held = verifier.verify_fix(v0_held, v1_held)

    # Dev single-fix ablations (marginals, as in the original report)
    ablations = {}
    for name, kw in (("metadata_only", dict(glossary=False, template=False)),
                     ("glossary_only", dict(metadata=False, template=False)),
                     ("template_only", dict(metadata=False, glossary=False))):
        ag = fresh_agent(inv, **kw)
        v = verifier.evaluate_suite(ag, dev_data, inv, db_path_of)
        vv = verifier.verify_fix(v0_dev, v)
        ablations[name] = {"R+": vv["Qfix"], "R-": vv["Qbreak"],
                           "RΔ": vv["Qnet"], "fixed": vv["fixed"],
                           "broken": vv["broken"]}
    return {
        "canon": (canonical(v0_dev), canonical(v1_dev),
                  canonical(v0_held), canonical(v1_held)),
        "ver_dev": ver_dev, "ver_held": ver_held, "ablations": ablations,
        "ex": {
            "v0_dev": sum(r["ex"] for r in v0_dev.values()),
            "v1_dev": sum(r["ex"] for r in v1_dev.values()),
            "v0_held": sum(r["ex"] for r in v0_held.values()),
            "v1_held": sum(r["ex"] for r in v1_held.values()),
        },
    }


def main():
    split = json.load(open(os.path.join(HERE, "bird_split.json")))
    dev_ids, held_ids = set(split["dev"]), set(split["heldout"])
    dataset, inv = load_dataset()
    qid_of = lambda e: "Q%d" % e["question_id"]  # noqa: E731
    dev_data = [e for e in dataset if qid_of(e) in dev_ids]
    held_data = [e for e in dataset if qid_of(e) in held_ids]
    assert len(dev_data) == 50 and len(held_data) == 50

    r1 = run_once(dataset, inv, dev_data, held_data)
    r2 = run_once(dataset, inv, dev_data, held_data)
    s1 = json.dumps([r1["canon"], r1["ver_dev"], r1["ver_held"],
                     r1["ablations"]], sort_keys=True)
    s2 = json.dumps([r2["canon"], r2["ver_dev"], r2["ver_held"],
                     r2["ablations"]], sort_keys=True)
    assert s1 == s2, "NON-DETERMINISTIC held-out pipeline"
    h = hashlib.sha256(s1.encode()).hexdigest()
    print("determinism: PASS (two full runs byte-identical excl. runtime_ms)")

    vd, vh = r1["ver_dev"], r1["ver_held"]
    print("dev:      R+=%d R-=%d RΔ=%d  fixed=%s broken=%s"
          % (vd["Qfix"], vd["Qbreak"], vd["Qnet"], vd["fixed"], vd["broken"]))
    print("heldout:  R+=%d R-=%d RΔ=%d  fixed=%s broken=%s"
          % (vh["Qfix"], vh["Qbreak"], vh["Qnet"], vh["fixed"], vh["broken"]))
    for name, a in r1["ablations"].items():
        print("dev ablation %-14s R+=%d R-=%d RΔ=%d fixed=%s"
              % (name, a["R+"], a["R-"], a["RΔ"], a["fixed"]))
    lo, hi = wilson(vh["Qfix"], 50)
    print("held-out R+ rate: %d/50 = %.3f  Wilson95%% [%.3f, %.3f]"
          % (vh["Qfix"], vh["Qfix"] / 50, lo, hi))

    # Perfect-agent sanity on the splits.
    db_path_of = lambda db_id: db_path(db_id)  # noqa: E731
    gold = {e["question"]: e["gold_sql"] for e in dataset}
    for name, sub in (("dev", dev_data), ("heldout", held_data)):
        agent = BirdSQLAgent(inv)
        agent.s4_sql_generation = (
            lambda question, intent, plan, db_id, evidence="":
            gold[question])
        res = verifier.evaluate_suite(agent, sub, inv, db_path_of)
        bad = sorted(q for q in res if res[q]["ex"] != 1)
        assert not bad, "perfect-agent failed on %s split: %s" % (name, bad)
        print("perfect-agent sanity (%s): PASS (%d/%d EX=1)"
              % (name, len(res), len(res)))

    out = {
        "dev_fix_spec": {
            "synonyms": DEV_FIX["synonyms"],
            "glossary_concepts": DEV_FIX["glossary_concepts"],
            "use_glossary_enforcement": DEV_FIX["use_glossary_enforcement"],
            "use_evidence_predicates": DEV_FIX["use_evidence_predicates"],
            "version": DEV_FIX["version"],
        },
        "n_dev": 50, "n_heldout": 50,
        "baseline_ex": r1["ex"],
        "dev": {"R+": vd["Qfix"], "R-": vd["Qbreak"], "RΔ": vd["Qnet"],
                "fixed": vd["fixed"], "broken": vd["broken"]},
        "heldout": {"R+": vh["Qfix"], "R-": vh["Qbreak"], "RΔ": vh["Qnet"],
                    "fixed": vh["fixed"], "broken": vh["broken"],
                    "wilson95_R+_rate": [round(lo, 4), round(hi, 4)]},
        "dev_ablations": r1["ablations"],
        "tests": {"determinism": "PASS sha256 %s" % h,
                  "q_assertions": "PASS (verify_fix asserts held)",
                  "perfect_agent_dev": "PASS 50/50",
                  "perfect_agent_heldout": "PASS 50/50"},
    }
    with open(os.path.join(HERE, "bird_heldout_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote bird_heldout_results.json")


if __name__ == "__main__":
    main()
