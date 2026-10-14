# Author: Prabhu Rangaraju
"""Held-out repair evaluation -- Spider STEP B: dev-derived fix -> re-eval.

Dev-derived fix (authored from spider_stageA.json ONLY; held-out blinded):
  layer logic: dominant dev mode SCHEMA_LINK_MISS (36 >= 5) -> metadata fix
  stem_plurals=True            (re-derived: concerts/singers/models/countries/
                                stadiums plural gaps in dev)
  synonyms (dev-observed non-plural gaps with a clear schema referent):
      nation->country, manufacturers->maker, cards->car   (re-derived)
      stations->stadium EXCLUDED ('stations' never occurs in dev gaps)
      carr->car  NEW (dev gap token, freq 1, unambiguous typo of 'car';
                      question CA-15: "maker of the carr produced ...")
  prune_subsumed=True, drop_columnless=True (mechanism-level guards, same
      template as original; ablated on dev below)

Applies the fix to a FRESH agent; re-runs on dev AND held-out; verifies
Qfix/Qbreak/Qnet per split. PRIMARY: held-out R+/R-/RΔ.

Run: cd <spider_dir> && venv/bin/python <heldout>/spider_fixeval.py
"""

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "spider"))

sys.path.insert(0, SPIDER_DIR)
os.chdir(SPIDER_DIR)

import dataset as spider_dataset
import verifier
from agent import SpiderBaselineAgent
from goldparse import parse_gold
from schema import SpiderSchema

# ---- dev-derived fix spec (from spider_stageA.json only) ------------------
DEV_FIX = {
    "stem_plurals": True,
    "prune_subsumed": True,
    "drop_columnless": True,
    "synonyms": {
        "nation": "country",          # re-derived (dev gap x1)
        "manufacturers": "maker",     # re-derived (dev gap x2)
        "cards": "car",               # re-derived (dev gap x2)
        "carr": "car",                # NEW from dev (gap x1, typo of 'car')
        # "stations": "stadium" EXCLUDED -- 'stations' absent from dev gaps.
    },
    "version": "1.1-dev",
}


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


def fresh_agent(schemas, spec):
    ag = SpiderBaselineAgent(schemas)
    ag.stem_plurals = spec["stem_plurals"]
    ag.prune_subsumed = spec["prune_subsumed"]
    ag.drop_columnless = spec["drop_columnless"]
    ag.synonyms.update(spec["synonyms"])
    ag.version = spec["version"]
    return ag


def jclean(o):
    if isinstance(o, dict):
        return {k: jclean(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, set):
        return sorted(jclean(v) for v in o)
    if isinstance(o, float):
        return round(o, 4)
    return o


def strip_runtime(results):
    return {qid: jclean({k: v for k, v in r.items() if k != "runtime_ms"})
            for qid, r in results.items()}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


class PerfectAgent(SpiderBaselineAgent):
    def __init__(self, schemas, gold_by_question):
        super().__init__(schemas)
        self.gold_by_question = gold_by_question
        self._question = None

    def answer(self, question, db_id):
        self._question = question
        return super().answer(question, db_id)

    def s4_sql_generation(self, question, intent, plan, s2, schema):
        return self.gold_by_question[self._question]


def run_once():
    split = json.load(open(os.path.join(HERE, "spider_split.json")))
    dev_ids, held_ids = set(split["dev"]), set(split["heldout"])
    schemas, data = build()
    dev_data = [e for e in data if e["id"] in dev_ids]
    held_data = [e for e in data if e["id"] in held_ids]

    base = SpiderBaselineAgent(schemas)  # v1.0
    v0_dev = verifier.evaluate_suite(base, dev_data, schemas)
    v0_held = verifier.evaluate_suite(base, held_data, schemas)

    fixed = fresh_agent(schemas, DEV_FIX)
    v1_dev = verifier.evaluate_suite(fixed, dev_data, schemas)
    v1_held = verifier.evaluate_suite(fixed, held_data, schemas)

    ver_dev = verifier.verify_fix(v0_dev, v1_dev)
    ver_held = verifier.verify_fix(v0_held, v1_held)

    # Ablation on dev: stemming+synonyms WITHOUT the two guards.
    noguard = dict(DEV_FIX, prune_subsumed=False, drop_columnless=False,
                   version="1.1-dev-noguard")
    ag_ng = fresh_agent(schemas, noguard)
    v1_dev_ng = verifier.evaluate_suite(ag_ng, dev_data, schemas)
    ver_dev_ng = verifier.verify_fix(v0_dev, v1_dev_ng)

    return {
        "v0_dev": strip_runtime(v0_dev), "v1_dev": strip_runtime(v1_dev),
        "v0_held": strip_runtime(v0_held), "v1_held": strip_runtime(v1_held),
        "ver_dev": ver_dev, "ver_held": ver_held,
        "ver_dev_noguard": ver_dev_ng,
        "ex_v1_dev_noguard": sum(r["ex"] for r in v1_dev_ng.values()),
    }


def main():
    r1 = run_once()
    r2 = run_once()
    s1 = json.dumps(r1, sort_keys=True)
    s2 = json.dumps(r2, sort_keys=True)
    assert s1 == s2, "NON-DETERMINISTIC held-out pipeline"
    h = hashlib.sha256(s1.encode()).hexdigest()
    print("determinism: PASS (two full runs byte-identical, sha256 %s...)"
          % h[:16])

    vd, vh = r1["ver_dev"], r1["ver_held"]
    print("dev:      R+=%d R-=%d RΔ=%d  fixed=%s broken=%s"
          % (vd["Qfix"], vd["Qbreak"], vd["Qnet"], vd["fixed"], vd["broken"]))
    print("heldout:  R+=%d R-=%d RΔ=%d  fixed=%s broken=%s"
          % (vh["Qfix"], vh["Qbreak"], vh["Qnet"], vh["fixed"], vh["broken"]))
    vn = r1["ver_dev_noguard"]
    print("dev ablation (no guards): R+=%d R-=%d RΔ=%d fixed=%s"
          % (vn["Qfix"], vn["Qbreak"], vn["Qnet"], vn["fixed"]))
    lo, hi = wilson(vh["Qfix"], 44)
    print("held-out R+ rate: %d/44 = %.3f  Wilson95%% [%.3f, %.3f]"
          % (vh["Qfix"], vh["Qfix"] / 44, lo, hi))

    # Perfect-agent sanity on the splits.
    split = json.load(open(os.path.join(HERE, "spider_split.json")))
    schemas, data = build()
    gold_by_q = {e["question"]: e["gold_sql"] for e in data}
    for name, ids in (("dev", split["dev"]), ("heldout", split["heldout"])):
        sub = [e for e in data if e["id"] in set(ids)]
        res = verifier.evaluate_suite(PerfectAgent(schemas, gold_by_q),
                                      sub, schemas)
        bad = sorted(q for q in res if res[q]["ex"] != 1)
        assert not bad, "perfect-agent failed on %s split: %s" % (name, bad)
        print("perfect-agent sanity (%s): PASS (%d/%d EX=1)"
              % (name, len(res), len(res)))

    out = {
        "dev_fix_spec": {k: v for k, v in DEV_FIX.items()},
        "n_dev": 44, "n_heldout": 44,
        "baseline_ex_dev": sum(r["ex"] for r in r1["v0_dev"].values()),
        "baseline_ex_heldout": sum(r["ex"] for r in r1["v0_held"].values()),
        "after_ex_dev": sum(r["ex"] for r in r1["v1_dev"].values()),
        "after_ex_heldout": sum(r["ex"] for r in r1["v1_held"].values()),
        "dev": {"R+": vd["Qfix"], "R-": vd["Qbreak"], "RΔ": vd["Qnet"],
                "fixed": vd["fixed"], "broken": vd["broken"]},
        "heldout": {"R+": vh["Qfix"], "R-": vh["Qbreak"], "RΔ": vh["Qnet"],
                    "fixed": vh["fixed"], "broken": vh["broken"],
                    "wilson95_R+_rate": [round(lo, 4), round(hi, 4)]},
        "dev_ablation_noguard": {"R+": vn["Qfix"], "R-": vn["Qbreak"],
                                 "RΔ": vn["Qnet"], "fixed": vn["fixed"],
                                 "broken": vn["broken"]},
        "tests": {"determinism": "PASS sha256 %s" % h,
                  "q_assertions": "PASS (verify_fix asserts held)",
                  "perfect_agent_dev": "PASS 44/44",
                  "perfect_agent_heldout": "PASS 44/44"},
    }
    with open(os.path.join(HERE, "spider_heldout_results.json"), "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote spider_heldout_results.json")


if __name__ == "__main__":
    main()
