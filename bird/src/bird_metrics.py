# Author: Prabhu Rangaraju
"""Per-stage metric functions for the BIRD arm. Mirrors the prototype's
metrics.py: schema_linking_pr (tables + table.column vs gold), EX via
multiset row-tuple equality (order-insensitive, floats rounded to 4dp),
efficiency_ms, and substring-based semantic_conformance against the glossary.
"""

from bird_glossary import CATALOG


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


def _sort_key(row):
    # Type-aware sort key: real result sets mix None/str/numbers.
    key = []
    for v in row:
        if v is None:
            key.append((0, 0, ""))
        elif isinstance(v, bool):
            key.append((1, int(v), ""))
        elif isinstance(v, (int, float)):
            key.append((1, v, ""))
        else:
            key.append((2, 0, str(v)))
    return tuple(key)


def execution_accuracy(agent_rows, gold_rows):
    a = sorted((_norm_row(r) for r in agent_rows), key=_sort_key)
    g = sorted((_norm_row(r) for r in gold_rows), key=_sort_key)
    if not g:
        return 0
    return 1 if a == g else 0


def efficiency_ms(trace):
    return trace["s5"]["runtime_ms"]


def semantic_conformance(agent_sql, gold_definitions, glossary):
    """Substring conformance; definitions missing from the glossary are
    reported separately (the diagnoser attributes those to S3)."""
    sql = agent_sql.lower()
    missing, missing_defs, shape_ok = [], [], True
    for d in gold_definitions:
        spec = glossary.get(d)
        if spec is None:
            missing_defs.append(d)
            continue
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
        "missing_definitions": sorted(set(missing_defs)),
        "shape_ok": shape_ok,
    }
