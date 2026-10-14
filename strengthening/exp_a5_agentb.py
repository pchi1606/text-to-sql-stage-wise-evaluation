# Author: Prabhu Rangaraju
"""A5. SECOND AGENT VARIANT (Agent-B) -- framework discrimination check.

Agent-B vs Agent-A (v1.0 baseline):
  S2 IMPROVED: column-value literal matching (quoted literals and
      capitalized question tokens matched against sampled DISTINCT column
      values, case-insensitive) PLUS the v1.1 synonym set
      (stations->stadium, nation->country, manufacturers->maker, cards->car).
      Also stems table names (as v1.1 does). NO subsumption pruning, NO
      columnless-table guard (kept off to isolate the S3 weakening).
  S3 DELIBERATELY WEAKENED: greedy join in S2 table order using ONLY direct
      FK edges between consecutive tables -- no FK-graph BFS multi-hop
      paths. Tables that cannot directly join the previous table fall back
      to the 1=1 cross join and are dropped as disconnected (same pruning
      rule as the parent), which surfaces as JOIN_ERROR.

Full evaluate->diagnose loop (no fix cycle). Report EX, schema-linking
P/R/F1, and the failure-stage distribution for B vs A. Success criterion:
stage profiles differ in the predicted direction (B higher S2 F1; B's
failures concentrated in S3 JOIN_ERROR vs A's in S2). Whatever is observed
is reported honestly.
"""

import os
import random
import re
import sqlite3
import sys

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402

sys.path.insert(0, common.SPIDER_DIR)
os.chdir(common.SPIDER_DIR)

import dataset  # noqa: E402
import verifier  # noqa: E402
from diagnoser import localize  # noqa: E402
from metrics import schema_linking_pr  # noqa: E402
from agent import SpiderBaselineAgent  # noqa: E402
from goldparse import parse_gold  # noqa: E402
from schema import SpiderSchema  # noqa: E402

V11_SYNONYMS = {
    "stations": "stadium",
    "nation": "country",
    "manufacturers": "maker",
    "cards": "car",
}


def build_value_index(schema):
    """value (lowercased/stripped) -> (table, column), sampled from TEXT
    columns (up to 200 distinct values each). Deterministic order."""
    index = {}
    con = sqlite3.connect(schema.db_path)
    try:
        for t in sorted(schema.tables):
            for c in schema.tables[t]:
                ctype = [r[2] for r in con.execute(
                    'PRAGMA table_info("%s")' % t) if r[1] == c]
                if not ctype or not any(
                        k in (ctype[0] or "").upper()
                        for k in ("CHAR", "TEXT", "CLOB")):
                    continue
                try:
                    vals = con.execute(
                        'SELECT DISTINCT "%s" FROM "%s" LIMIT 200' % (c, t))
                    for (v,) in vals:
                        if v is None:
                            continue
                        key = str(v).strip().lower()
                        if key and key not in index:
                            index[key] = (t, c)
                except Exception:  # noqa: BLE001 - skip unreadable columns
                    continue
    finally:
        con.close()
    return index


class AgentB(SpiderBaselineAgent):
    def __init__(self, schemas):
        super().__init__(schemas)
        self.stem_plurals = True
        self.synonyms.update(V11_SYNONYMS)
        self.prune_subsumed = False
        self.drop_columnless = False
        self.version = "B"
        self._value_index = {db: build_value_index(s)
                             for db, s in schemas.items()}

    def s2_schema_linking(self, question, intent, schema):
        base = super().s2_schema_linking(question, intent, schema)
        # Literal matching: quoted literals + capitalized tokens vs sampled
        # column values (case-insensitive)
        lits = re.findall(r"'([^']+)'", question)
        caps = re.findall(r"[A-Z][A-Za-z]+", question)
        idx = self._value_index[schema.db_id]
        for tok in lits + caps:
            hit = idx.get(tok.strip().lower())
            if not hit:
                continue
            t, c = hit
            if t not in base["tables"] and len(base["tables"]) < 6:
                base["tables"].append(t)
            if t in base["tables"]:
                cols = base["columns"].setdefault(t, [])
                if c not in cols and len(cols) < 8:
                    cols.append(c)
        return base

    def s3_planning(self, s2, intent, schema):
        tables = list(s2["tables"])
        need = set()
        for _f, col in intent["aggs"]:
            need.add(col[0])
        for col, _op, _v in intent["filters"]:
            need.add(col[0])
        if intent["group_word"]:
            g = self._col_lookup(intent["group_word"], schema)
            if g:
                need.add(g[0])
                intent["_group_col"] = g
        if intent["order_word"]:
            o = self._col_lookup(intent["order_word"], schema)
            if o:
                need.add(o[0])
                intent["_order_col"] = o
        for t in sorted(need):
            if t not in tables:
                tables.append(t)
        # GREEDY: each table joins via a DIRECT FK edge to an already-kept
        # table only (most recent first = mention-order greediness). No BFS
        # multi-hop path search (the deliberate weakening). Tables with no
        # direct edge to the kept set fall back to 1=1 and are dropped as
        # disconnected (same pruning rule as the parent).
        ordered, joins = [], []
        for t in tables:
            if not ordered:
                ordered.append(t)
                continue
            cond = None
            for prev in reversed(ordered):
                for nb, c in schema.neighbors(prev):
                    if nb == t:
                        cond = c
                        break
                if cond is not None:
                    break
            if cond is None:
                joins.append((None, t, "1=1"))
            else:
                joins.append((prev, t, cond))
                ordered.append(t)
        pruned = [t for _, t, c in joins if c == "1=1"]
        if pruned:
            ordered = [t for t in ordered if t not in pruned]
            joins = [j for j in joins if j[2] != "1=1"]
        return {"tables": ordered, "joins": joins, "pruned": pruned,
                "intent": intent}


def profile(agent, data, schemas, label):
    link_tables, link_cols = [], []
    stages, modes = {}, {}
    per_q, ex_vec = {}, []
    for e in data:
        schema = schemas[e["db_id"]]
        trace = agent.answer(e["question"], e["db_id"])
        grows = common.gold_rows(schema.db_path, e["gold_sql"])
        arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
        ex = common.multiset_equal(arows, grows)
        ex_vec.append(ex)
        link = schema_linking_pr(trace, e["gold_tables"], e["gold_columns"])
        link_tables.append(link["tables"])
        link_cols.append(link["columns"])
        mode, stage, _ev = localize(e, trace, {"ex": ex}, agent, schema)
        per_q[e["id"]] = {"ex": ex, "mode": mode, "stage": stage,
                          "link_f1_tables": round(link["tables"]["f1"], 4),
                          "link_f1_cols": round(link["columns"]["f1"], 4)}
        if stage:
            stages[stage] = stages.get(stage, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1

    def mean(key, rows):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "label": label,
        "EX": "%d/%d" % (sum(ex_vec), len(ex_vec)),
        "ex_vec": ex_vec,
        "link_tables": {"P": round(mean("precision", link_tables), 4),
                        "R": round(mean("recall", link_tables), 4),
                        "F1": round(mean("f1", link_tables), 4)},
        "link_columns": {"P": round(mean("precision", link_cols), 4),
                         "R": round(mean("recall", link_cols), 4),
                         "F1": round(mean("f1", link_cols), 4)},
        "stage_distribution": stages,
        "mode_distribution": modes,
        "per_question": per_q,
    }


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

    agent_a = SpiderBaselineAgent(schemas)
    agent_b = AgentB(schemas)

    prof_a = profile(agent_a, data, schemas, "Agent-A (v1.0)")
    prof_b = profile(agent_b, data, schemas, "Agent-B")

    out = {"n": len(data), "agent_a": prof_a, "agent_b": prof_b,
           "design": ("B: S2 improved via column-value literal matching + "
                      "v1.1 synonyms + table stemming; S3 weakened to greedy "
                      "direct-edge joins, no FK BFS."),
           "success_criterion": ("B higher S2 F1 than A; B failures "
                                 "concentrated in S3 JOIN_ERROR vs A in S2.")}
    path = common.write_raw("a5.json", out)
    print("wrote", path)
    for p in (prof_a, prof_b):
        print("%s: EX=%s linkT F1=%.3f linkC F1=%.3f stages=%s" % (
            p["label"], p["EX"], p["link_tables"]["F1"],
            p["link_columns"]["F1"], p["stage_distribution"]))
        print("   modes:", p["mode_distribution"])


if __name__ == "__main__":
    main()
