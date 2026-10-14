# Author: Prabhu Rangaraju
"""A1. LOCALIZED vs BLIND REPAIR ABLATION.

Question: does failure localization beat blind repair?

Localized fix (v1.1, from the diagnosed dominant mode SCHEMA_LINK_MISS):
plural stemming for TABLE matching only + 4 curated synonyms
(stations->stadium, nation->country, manufacturers->maker, cards->car)
+ subsumption pruning + columnless-table guard.

Blind repair (no diagnosis consulted -- "fix everything that looks like a
lexical gap"): plural stemming for BOTH table and column matching
(deliberately aggressive: the v1.1 fix kept column matching exact because
stemming columns adds false positives, e.g. 'names' pulling Name into every
SELECT) + an automatically generated broad synonym map (every question token
sharing a >=4-char prefix or a substring relation with any schema
table/column token, mapped to the alphabetically-first such schema token --
NO failure analysis used to pick these) + NO subsumption-pruning guard and
NO columnless-table guard.

Both evaluated on the same n=88 with the same EX semantics and the same
verifier.verify_fix assertions (Qfix/Qbreak/Qnet).
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


def build_schemas():
    db_dir = os.path.join(common.SPIDER_DIR, "data", "db_extract",
                          "spider_data", "database")
    return {
        "concert_singer": SpiderSchema(
            "concert_singer",
            os.path.join(db_dir, "concert_singer", "concert_singer.sqlite")),
        "car_1": SpiderSchema(
            "car_1", os.path.join(db_dir, "car_1", "car_1.sqlite")),
    }


def build_broad_synonyms(schemas, data):
    """Generate the blind synonym map: NO diagnosis used.

    Rule: for each question token t (len>3) and each schema table/column
    NAME token s (both DBs): if t != s and (common prefix >= 4 chars, or t
    is a substring of s / s is a substring of t with both len>3), record
    t -> alphabetically-first such s. Deterministic.
    """
    schema_toks = set()
    for schema in schemas.values():
        for t, cols in schema.tables.items():
            schema_toks |= set(schema.table_tokens(t))
            for c in cols:
                schema_toks |= set(schema.column_tokens(c))
    schema_toks = sorted(schema_toks)

    def common_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x == y:
                n += 1
            else:
                break
        return n

    qtokens = set()
    for e in data:
        qtokens |= set(SpiderBaselineAgent._tokens(e["question"]))
    mapping = {}
    for t in sorted(qtokens):
        if len(t) <= 3:
            continue
        cands = []
        for s in schema_toks:
            if t == s or len(s) <= 3:
                continue
            if common_prefix(t, s) >= 4 or t in s or s in t:
                cands.append(s)
        if cands:
            mapping[t] = cands[0]
    return mapping


class BlindRepairAgent(SpiderBaselineAgent):
    """Blind repair: aggressive, undiagnosed, no precision guards."""

    def __init__(self, schemas, broad_synonyms):
        super().__init__(schemas)
        self.stem_plurals = True       # tables AND columns (see s2 override)
        self.prune_subsumed = False    # NO subsumption-pruning guard
        self.drop_columnless = False   # NO columnless-table guard
        self.synonyms = dict(broad_synonyms)
        self.version = "1.1-blind"

    def s2_schema_linking(self, question, intent, schema):
        # Same as the parent, except column matching ALSO stems: the parent
        # deliberately disables stemming for columns (false-positive risk).
        # Blind repair stems everything.
        t_content = set(self._content(question))
        t_tok = {t: {self._norm(x) for x in schema.table_tokens(t)}
                 for t in schema.tables}
        c_content = set(self._content(question))
        c_tok = {}
        for t in schema.tables:
            for c in schema.tables[t]:
                c_tok[(t, c)] = {self._norm(x)
                                 for x in schema.column_tokens(c)}
        tscores = {t: len(t_content & t_tok[t]) for t in schema.tables}
        ranked = [t for t in schema.tables if tscores[t] > 0]
        ranked.sort(key=lambda t: (-tscores[t], t))
        linked = ranked[:4]
        columns = {}
        for t in linked:
            scored = []
            for c in schema.tables[t]:
                ov = len(c_content & c_tok[(t, c)])
                if ov > 0:
                    scored.append((ov, c))
            scored.sort(key=lambda x: (-x[0], x[1]))
            columns[t] = [c for _, c in scored[:6]]
        return {"tables": linked, "columns": columns,
                "table_scores": tscores}


def ex_vector(agent, data, schemas):
    vec, per_q = [], {}
    for e in data:
        ex, _trace = common.agent_ex(
            agent, e["question"], e["db_id"],
            schemas[e["db_id"]].db_path, e["gold_sql"])
        vec.append(ex)
        per_q[e["id"]] = ex
    return vec, per_q


def main():
    random.seed(42)
    schemas = build_schemas()
    data, exclusions = dataset.build_dataset(schemas, parse_gold)
    assert len(data) == 88, "expected n=88, got %d" % len(data)
    assert len(exclusions) == 2

    # V1.0 baseline.
    agent_v10 = SpiderBaselineAgent(schemas)
    v0_vec, v0_perq = ex_vector(agent_v10, data, schemas)

    # V1.1 localized fix (same propose/apply path as run.py)
    agent_v11 = SpiderBaselineAgent(schemas)
    diagnoses = {}
    # Diagnoses needed by propose_fixes: recompute via verifier.evaluate_suite.
    v0_full = verifier.evaluate_suite(agent_v10, data, schemas)
    for qid in v0_full:
        diagnoses[qid] = (v0_full[qid]["failure_mode"],
                          v0_full[qid]["failure_stage"],
                          v0_full[qid]["evidence"])
    fixes = fixer.propose_fixes(diagnoses)
    assert len(fixes) == 1 and fixes[0]["layer"] == "metadata"
    for f in fixes:
        fixer.apply_fix(agent_v11, f)
    v1_vec, v1_perq = ex_vector(agent_v11, data, schemas)

    # Blind repair.
    broad = build_broad_synonyms(schemas, data)
    agent_blind = BlindRepairAgent(schemas, broad)
    vb_vec, vb_perq = ex_vector(agent_blind, data, schemas)
    # Determinism check: second blind run must be identical.
    agent_blind2 = BlindRepairAgent(schemas, broad)
    vb_vec2, _ = ex_vector(agent_blind2, data, schemas)
    assert vb_vec2 == vb_vec, "blind run not deterministic"

    before = {qid: {"ex": e} for qid, e in v0_perq.items()}
    loc_after = {qid: {"ex": e} for qid, e in v1_perq.items()}
    blind_after = {qid: {"ex": e} for qid, e in vb_perq.items()}
    ver_loc = verifier.verify_fix(before, loc_after)
    ver_blind = verifier.verify_fix(before, blind_after)

    n = len(data)
    payload = {
        "n": n,
        "broad_synonyms": broad,
        "n_broad_synonyms": len(broad),
        "v10_ex": sum(v0_vec),
        "v11_ex": sum(v1_vec),
        "blind_ex": sum(vb_vec),
        "v10_vec": v0_vec,
        "v11_vec": v1_vec,
        "blind_vec": vb_vec,
        "localized": {"Qfix": ver_loc["Qfix"], "Qbreak": ver_loc["Qbreak"],
                      "Qnet": ver_loc["Qnet"], "fixed": ver_loc["fixed"],
                      "broken": ver_loc["broken"]},
        "blind": {"Qfix": ver_blind["Qfix"], "Qbreak": ver_blind["Qbreak"],
                  "Qnet": ver_blind["Qnet"], "fixed": ver_blind["fixed"],
                  "broken": ver_blind["broken"]},
        "comparison": {
            "localized": {"Qfix": ver_loc["Qfix"],
                          "Qbreak": ver_loc["Qbreak"],
                          "Qnet": ver_loc["Qnet"],
                          "final_EX": "%d/%d" % (sum(v1_vec), n)},
            "blind": {"Qfix": ver_blind["Qfix"],
                      "Qbreak": ver_blind["Qbreak"],
                      "Qnet": ver_blind["Qnet"],
                      "final_EX": "%d/%d" % (sum(vb_vec), n)},
        },
        "notes": ("Blind = aggressive undiagnosed repair: stemming on BOTH "
                  "table and column matching + programmatic broad synonym "
                  "map (%d pairs, no diagnosis used) + no subsumption "
                  "pruning + no columnless-table guard." % len(broad)),
    }
    path = common.write_raw("a1.json", payload)
    print("wrote", path)
    print("localized: Qfix=%d Qbreak=%d Qnet=%d final EX %d/%d" % (
        ver_loc["Qfix"], ver_loc["Qbreak"], ver_loc["Qnet"],
        sum(v1_vec), n))
    print("blind:     Qfix=%d Qbreak=%d Qnet=%d final EX %d/%d" % (
        ver_blind["Qfix"], ver_blind["Qbreak"], ver_blind["Qnet"],
        sum(vb_vec), n))
    print("broad synonyms: %d" % len(broad))
    print("blind fixed:", ver_blind["fixed"])
    print("blind broken:", ver_blind["broken"])


if __name__ == "__main__":
    main()
