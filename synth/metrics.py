# Author: Prabhu Rangaraju
"""Per-stage metric functions for the SQL-eval-framework prototype.

  schema_linking_pr : precision/recall/F1 of S2 linked tables and of linked
                      'table.column' pairs against gold.
  execution_accuracy: 1 iff the multisets of result row-tuples are equal
                      (order-insensitive; floats rounded to 4dp).
  efficiency_ms     : S5 wall-clock runtime.
  semantic_conformance: 1 iff every gold glossary definition's required
                      predicates appear in the agent SQL and the aggregation
                      shape matches; reports the missing pieces and whether
                      the shape itself was wrong.
"""

from glossary import CONCEPT_TO_DEF


def _prf(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def schema_linking_pr(trace, gold_tables, gold_columns):
    linked_tables = trace["s2"]["tables"]
    linked_cols = [
        "%s.%s" % (t, c)
        for t, cols in trace["s2"]["columns"].items()
        for c in cols
    ]
    return {
        "tables": _prf(linked_tables, gold_tables),
        "columns": _prf(linked_cols, gold_columns),
    }


def _norm_row(row):
    return tuple(round(v, 4) if isinstance(v, float) else v for v in row)


def execution_accuracy(agent_rows, gold_rows):
    a = sorted(_norm_row(r) for r in agent_rows)
    g = sorted(_norm_row(r) for r in gold_rows)
    if not g:
        return 0
    return 1 if a == g else 0


def efficiency_ms(trace):
    return trace["s5"]["runtime_ms"]


def semantic_conformance(agent_sql, gold_definitions, glossary):
    sql = agent_sql.lower()
    missing = []
    shape_ok = True
    for d in gold_definitions:
        spec = glossary[d]
        for p in spec.get("required_predicates", []):
            if p.lower() not in sql:
                missing.append("%s: missing predicate '%s'" % (d, p))
        shape = spec.get("agg_shape")
        if shape and shape.lower() not in sql:
            missing.append("%s: missing agg shape '%s'" % (d, shape))
            shape_ok = False
    return {
        "score": 1.0 if not missing else 0.0,
        "missing": missing,
        "shape_ok": shape_ok,
    }


def concept_definition_key(concept):
    return CONCEPT_TO_DEF.get(concept)
