# Author: Prabhu Rangaraju
"""Per-stage metric functions for the Spider arm.

  schema_linking_pr : precision/recall/F1 of S2 linked tables and of linked
                      'table.column' pairs against gold (heuristic-derived).
  execution_accuracy: 1 iff the multisets of result row-tuples are equal
                      (order-insensitive; floats rounded to 4dp).
  efficiency_ms     : S5 wall-clock runtime (measured but EXCLUDED from the
                      deterministic results artifact -- see run.py).

Deviation from the prototype: no semantic_conformance (no governed glossary
exists for these Spider DBs; recorded as not-applicable).
"""


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
    def norm(v):
        if isinstance(v, float):
            return round(v, 4)
        return v
    return tuple(norm(v) for v in row)


def execution_accuracy(agent_rows, gold_rows):
    a = sorted(_norm_row(r) for r in agent_rows)
    g = sorted(_norm_row(r) for r in gold_rows)
    if not g:
        return 0
    return 1 if a == g else 0


def efficiency_ms(trace):
    return trace["s5"]["runtime_ms"]
