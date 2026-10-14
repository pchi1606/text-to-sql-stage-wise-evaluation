# Author: Prabhu Rangaraju
"""A7 helper (synthetic). Per-layer ISOLATED fix attribution.

Applies each synthetic fix layer ALONE to a fresh v1.0 agent and measures
which v1.0 failures it repairs (Qfix per layer, with question IDs). Also
records the v1.0 failure (mode, stage) inventory for the contingency table.
"""

import os
import random
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

sys.path.insert(0, common.SYNTH_DIR)
os.chdir(common.SYNTH_DIR)

import dataset  # noqa: E402
import fixer  # noqa: E402
import verifier  # noqa: E402
from agent import BaselineSQLAgent  # noqa: E402


def main():
    random.seed(42)
    db_path = os.path.join(common.SYNTH_DIR, "retail.db")
    data = dataset.DATASET

    agent_v10 = BaselineSQLAgent()
    v0 = verifier.evaluate_suite(agent_v10, data, db_path)
    diagnoses = {qid: (v0[qid]["failure_mode"], v0[qid]["failure_stage"],
                       v0[qid]["evidence"]) for qid in v0}
    failures = {qid: {"mode": v0[qid]["failure_mode"],
                      "stage": v0[qid]["failure_stage"]}
                for qid in v0 if v0[qid]["failure_stage"]}

    fixes = fixer.propose_fixes(diagnoses)
    layers = {}
    for f in fixes:
        ag = BaselineSQLAgent()  # fresh v1.0
        fixer.apply_fix(ag, f)
        v1 = verifier.evaluate_suite(ag, data, db_path)
        ver = verifier.verify_fix(v0, v1)
        layers[f["layer"]] = {
            "title": f["title"],
            "targets": f["targets"],
            "Qfix": ver["Qfix"],
            "Qbreak": ver["Qbreak"],
            "Qnet": ver["Qnet"],
            "fixed": ver["fixed"],
            "broken": ver["broken"],
            "fixed_modes": {qid: failures[qid]["mode"]
                            for qid in ver["fixed"]},
        }

    # All three layers together (validates the cumulative 7/0/7)
    ag_all = BaselineSQLAgent()
    for f in fixes:
        fixer.apply_fix(ag_all, f)
    v_all = verifier.evaluate_suite(ag_all, data, db_path)
    ver_all = verifier.verify_fix(v0, v_all)

    out = {"n": 12,
           "v10_failures": failures,
           "per_layer_isolated": layers,
           "cumulative": {"Qfix": ver_all["Qfix"],
                          "Qbreak": ver_all["Qbreak"],
                          "Qnet": ver_all["Qnet"],
                          "fixed": ver_all["fixed"]},
           "note": ("Each layer applied ALONE to a fresh v1.0 agent; "
                    "Qfix counts questions going EX 0->1.")}
    path = common.write_raw("a7_synth_layers.json", out)
    print("wrote", path)
    print("v1.0 failures:", failures)
    for layer, r in layers.items():
        print("%s: Qfix=%d Qbreak=%d fixed=%s modes=%s" % (
            layer, r["Qfix"], r["Qbreak"], r["fixed"], r["fixed_modes"]))
    print("cumulative: Qfix=%d Qbreak=%d Qnet=%d" % (
        ver_all["Qfix"], ver_all["Qbreak"], ver_all["Qnet"]))


if __name__ == "__main__":
    main()
