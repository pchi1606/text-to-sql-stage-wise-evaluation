# Author: Prabhu Rangaraju
"""Main entry point: build DB -> baseline eval -> diagnose -> fix -> re-eval
-> verify -> REPORT.md. Deterministic: no timestamps, sorted output,
random.seed(42) in db.build_db. Run with:  python3 run.py
"""

import os

import db
import dataset
import verifier
import fixer
from agent import BaselineSQLAgent

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(WORKDIR, "retail.db")
REPORT_PATH = os.path.join(WORKDIR, "REPORT.md")

STAGE_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6"]


def failure_distribution(results):
    dist = {}
    for qid in sorted(results):
        r = results[qid]
        if r["failure_stage"]:
            dist[r["failure_stage"]] = dist.get(r["failure_stage"], 0) + 1
    return dist


def _qrow(qid, r):
    stage = r["failure_stage"] or "-"
    return (
        "| %s | %d | %.2f | %.2f | %.0f | %s | %s |\n"
        % (qid, r["ex"], r["link_f1_tables"], r["link_f1_cols"],
           r["conformance"], r["failure_mode"], stage)
    )


def build_report(v0, v1, verification, fixes, dist0):
    L = []
    A = L.append
    A("# SQL Agent Evaluation Framework -- Prototype Results\n")
    A("Closed-loop prototype: stage-wise evaluation of a 6-stage text-to-SQL "
      "agent (S1 intent, S2 schema linking, S3 planning, S4 SQL generation, "
      "S5 execution, S6 interpretation), failure localization to the "
      "responsible stage, fix proposal at the owning layer "
      "(metadata/glossary/template), and fix authentication via "
      "Qfix/Qbreak/Qnet.\n")
    A("## Dataset\n")
    A("12 natural-language questions over a synthetic retail database "
      "(retail.db: 40 customers, 20 products, 200 orders, ~12% cancelled). "
      "Every gold SQL was validated to execute cleanly and return non-empty "
      "results.\n")
    A("## Baseline evaluation (agent v1.0)\n")
    A("| qid | EX | link F1 (tables) | link F1 (cols) | conformance | "
      "failure mode | stage |\n")
    A("|-----|----|------------------|----------------|-------------|"
      "--------------|-------|\n")
    for qid in sorted(v0):
        A(_qrow(qid, v0[qid]))
    A("\nBaseline EX: %d/12.\n" % sum(r["ex"] for r in v0.values()))
    A("## Failure distribution by stage (baseline)\n")
    for s in STAGE_ORDER:
        if s in dist0:
            qids = sorted(q for q in v0 if v0[q]["failure_stage"] == s)
            A("- %s: %d (%s)\n" % (s, dist0[s], ", ".join(qids)))
    A("## Fixes applied (one per layer)\n")
    for i, f in enumerate(fixes, 1):
        A("### Fix %d [%s layer]: %s\n" % (i, f["layer"], f["title"]))
        A("%s\n" % f["description"])
        A("Targets: %s. Agent version -> %s.\n" % (f["targets"], f["version"]))
    A("## Fix verification (agent v1.1)\n")
    A("| qid | EX before | EX after |\n")
    A("|-----|-----------|----------|\n")
    for qid in sorted(v0):
        A("| %s | %d | %d |\n" % (qid, v0[qid]["ex"], v1[qid]["ex"]))
    A("\n| metric | value |\n|--------|-------|\n")
    A("| Qfix (repaired) | %d |\n" % verification["Qfix"])
    A("| Qbreak (regressed) | %d |\n" % verification["Qbreak"])
    A("| Qnet = Qfix - Qbreak | %d |\n" % verification["Qnet"])
    A("\nFixed: %s. Broken: %s.\n"
      % (", ".join(verification["fixed"]) or "none",
         ", ".join(verification["broken"]) or "none"))
    A("## Deliberate baseline weaknesses (by design)\n")
    A("- W1: S2 has no synonym handling ('turnover'/'sales' unmapped).\n")
    A("- W2: S3 prunes the lowest-confidence table on 3-table plans.\n")
    A("- W3: S4 omits the governed status='completed' predicate for revenue.\n")
    A("- W4: S4 region template references non-existent column c.regio.\n")
    A("- W5: S4 guesses for AVG_ORDER_VALUE (per-item AVG) and 'repeat' "
      "(COUNT(*)).\n")
    A("## Limitations\n")
    A("- The agent under evaluation is rule-based, not an LLM: the prototype "
      "demonstrates the evaluation/diagnosis/fix machinery, not frontier "
      "model behavior.\n")
    A("- n=12 questions on one synthetic schema; no statistical claims.\n")
    A("- Diagnosis is heuristic and ordered (S5 error > EX > lexical gap > "
      "missing definition > conformance > join coverage > result shape); "
      "ambiguous multi-cause failures are attributed to the first match.\n")
    A("- Semantic conformance is substring-based, not a true logical "
      "equivalence check.\n")
    A("## What this prototype does not do\n")
    A("- No longitudinal trend mining across runs (single before/after "
      "comparison only).\n")
    A("- No capture of manual analyst workflows as ground-truth trajectories.\n")
    A("- No efficiency (VES-style) scoring gates; runtime is recorded only.\n")
    A("- No LLM-as-judge for semantic equivalence.\n")
    A("## Reproducibility\n")
    A("Python 3 stdlib only. `python3 run.py` rebuilds retail.db with "
      "random.seed(42) and regenerates this file byte-identically on repeat "
      "runs (no timestamps).\n")
    return "".join(L)


def main():
    db.build_db(DB_PATH)
    data = dataset.DATASET

    agent = BaselineSQLAgent()
    v0 = verifier.evaluate_suite(agent, data, DB_PATH)
    dist0 = failure_distribution(v0)

    diagnoses = {qid: (v0[qid]["failure_mode"], v0[qid]["failure_stage"],
                       v0[qid]["evidence"]) for qid in v0}
    fixes = fixer.propose_fixes(diagnoses)
    for f in fixes:
        fixer.apply_fix(agent, f)

    v1 = verifier.evaluate_suite(agent, data, DB_PATH)
    verification = verifier.verify_fix(v0, v1)

    report = build_report(v0, v1, verification, fixes, dist0)
    with open(REPORT_PATH, "w") as fh:
        fh.write(report)

    print("=== SQL Agent Evaluation Framework (prototype) ===")
    print("Baseline EX: %d/12   Agent v%s" % (
        sum(r["ex"] for r in v0.values()), "1.0"))
    print("Failure distribution by stage:")
    for s in STAGE_ORDER:
        if s in dist0:
            qids = sorted(q for q in v0 if v0[q]["failure_stage"] == s)
            print("  %s: %d  (%s)" % (s, dist0[s], ", ".join(qids)))
    print("Fixes applied: %d (%s)" % (
        len(fixes), ", ".join(f["layer"] for f in fixes)))
    print("Qfix=%d  Qbreak=%d  Qnet=%d" % (
        verification["Qfix"], verification["Qbreak"], verification["Qnet"]))
    print("Fixed: %s" % (", ".join(verification["fixed"]) or "none"))
    print("Broken: %s" % (", ".join(verification["broken"]) or "none"))
    print("REPORT.md written to %s" % REPORT_PATH)


if __name__ == "__main__":
    main()
