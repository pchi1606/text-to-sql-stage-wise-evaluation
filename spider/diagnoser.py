# Author: Prabhu Rangaraju
"""Failure diagnosis for the Spider arm: localize a wrong answer to a stage.

Taxonomy (mode -> stage):
  SCHEMA_LINK_MISS            -> S2  (a content term matched no table/column)
  JOIN_ERROR                  -> S3  (plan dropped a required table)
  AGG_FILTER_ERROR            -> S4  (wrong aggregation shape / filter)
  EXECUTION_ERROR             -> S5  (SQLite raised)
  EMPTY_RESULT / WRONG_RESULT -> S6  (interpretation stage: bad/empty answer)

SEMANTIC_DEFINITION_VIOLATION is NOT APPLICABLE here: no governed domain
glossary exists for the Spider databases (no natural governed business
concept was identified), so it is never emitted.

Decision order is deliberate and documented (same as the prototype): a DB
error always wins (S5); a correct end-to-end answer (EX=1) is never diagnosed
as a failure even if intermediate retrieval was imperfect; lexical gaps are
attributed to S2; then join coverage; then aggregation/filter shape; finally
result-shape checks.
"""

import re

from agent import STOPWORDS


def _lexicon(schema, synonyms):
    lex = set(STOPWORDS)
    for t, cols in schema.tables.items():
        lex |= set(schema.table_tokens(t))
        for c in cols:
            lex |= set(schema.column_tokens(c))
    for k, v in synonyms.items():
        lex.add(k)
        lex.add(v)
    # Generic question words that carry no schema content.
    lex |= {"distinct", "different", "unique", "between", "each", "every",
            "average", "total", "many", "much", "number", "oldest",
            "youngest", "highest", "lowest", "smallest", "largest",
            "minimum", "maximum", "earliest", "latest", "first", "last"}
    return lex


def _content_tokens(question, agent):
    toks = re.findall(r"[a-z0-9]+", question.lower())
    toks = [agent._norm(t) for t in toks]
    return [t for t in toks if t not in STOPWORDS and not t.isdigit()]


def _gap_tokens(question, agent, schema):
    # Faithful to S2: a token is "resolved" if it is in the lexicon, or --
    # when the agent stems -- if its stem matches a (stemmed) table-name
    # token. Column matching in S2 is always exact.
    from agent import _stem
    lex = _lexicon(schema, agent.synonyms)
    table_lex = set()
    for t in schema.tables:
        for w in schema.table_tokens(t):
            table_lex.add(_stem(w) if agent.stem_plurals else w)
    gaps = []
    for tok in _content_tokens(question, agent):
        if tok in lex:
            continue
        if agent.stem_plurals and _stem(tok) in table_lex:
            continue
        gaps.append(tok)
    return gaps


def localize(entry, trace, mres, agent, schema):
    """Return (failure_mode, stage, evidence). stage is None when no failure."""
    s5 = trace["s5"]
    intent = trace["s1"]
    sql = trace["s4"]["sql"].lower()

    if s5["error"]:
        return ("EXECUTION_ERROR", "S5", "SQLite raised %s" % s5["error"])

    if mres["ex"] == 1:
        return ("no failure", None, "EX=1: agent result matches gold")

    gaps = _gap_tokens(entry["question"], agent, schema)
    if gaps:
        return (
            "SCHEMA_LINK_MISS", "S2",
            "unresolved term(s) %s: matched no table or column name token "
            "(exact overlap, no stemming/synonyms)" % sorted(set(gaps)),
        )

    plan_tables = set(trace["s3"]["tables"])
    missing_tables = sorted(set(entry["gold_tables"]) - plan_tables)
    if missing_tables:
        return (
            "JOIN_ERROR", "S3",
            "plan tables %s omit required %s"
            % (sorted(plan_tables), missing_tables),
        )

    # Aggregation / filter shape check (S4)
    problems = []
    for func, (t, c) in intent.get("aggs", []):
        if func.lower() not in sql or c.lower() not in sql:
            problems.append("agg %s(%s) not realized in SQL" % (func, c))
    for (t, c), op, _v in intent.get("filters", []):
        if c.lower() not in sql:
            problems.append("filter column %s.%s missing from SQL" % (t, c))
        elif op in (">", "<", ">=", "<=", "=") and op not in sql and "like" not in sql:
            problems.append("filter comparator for %s.%s not realized" % (t, c))
    if intent.get("group_word") and "_group_col" in intent:
        g = intent["_group_col"]
        if "group by" not in sql:
            problems.append("group-by on %s.%s missing" % (g[0], g[1]))
    if problems:
        return ("AGG_FILTER_ERROR", "S4", "; ".join(problems))

    if s5["n_rows"] == 0:
        return ("EMPTY_RESULT", "S6",
                "agent returned 0 rows while gold is non-empty")

    return ("WRONG_RESULT", "S6",
            "result mismatch with no earlier-stage evidence")
