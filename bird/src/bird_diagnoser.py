# Author: Prabhu Rangaraju
"""Failure diagnosis for the BIRD arm: localize a wrong answer to the
responsible pipeline stage.

Taxonomy (mode -> stage), adapted from the prototype:
  SCHEMA_LINK_MISS            -> S2  (a content term matched no table/column)
  SEMANTIC_DEFINITION_VIOLATION -> S3 (needed concept has no governed
                                       definition, or the plan/SQL did not
                                       enforce one)
  JOIN_ERROR                  -> S3  (plan dropped a required table)
  AGG_FILTER_ERROR            -> S4  (wrong aggregation shape / filter vs gold)
  EXECUTION_ERROR             -> S5  (SQLite raised)
  EMPTY_RESULT / WRONG_RESULT -> S6  (interpretation stage)

Decision order (kept from the prototype, with one BIRD adaptation):
S5 error first; EX==1 -> "no failure" (never diagnose a correct answer);
required-schema miss with lexical-gap explanation -> S2; missing governed
definition -> S3; conformance -> S3/S4;
join coverage -> S3; *aggregate-shape vs gold -> S4* (adaptation: BIRD golds
are aggregation-heavy, so a direct agent-vs-gold aggregate comparison is
checked before result-shape); result shape -> S6.
"""

import re

from bird_agent import STOPWORDS as _STOP, BirdSQLAgent
from bird_glossary import CATALOG, CONCEPT_TRIGGERS

AGG_TOKENS = ("avg(", "sum(", "count(", "min(", "max(")

# Words owned by the S1 intent stage (formula/shape words), not by schema
# linking: a question can fail on 'percentage' without any S2 miss.
FORMULA_WORDS = {
    "percentage", "percent", "ratio", "proportion", "rate", "average",
    "total", "sum", "number", "count", "highest", "lowest", "most",
    "least", "oldest", "youngest", "biggest", "smallest", "largest",
    "maximum", "minimum", "top", "ascending", "descending",
}
# Generic measure fillers ('GPT level'): the operative token is the measure
# name, which S2 scores separately.
MEASURE_FILLERS = {"level", "levels", "value", "values"}


def _stem(w):
    return BirdSQLAgent._stem(w)


def _content_tokens(question, synonyms):
    toks = re.findall(r"[a-z0-9]+", question.lower())
    toks = [synonyms.get(t, t) for t in toks]
    return [t for t in toks
            if t not in _STOP and not t.isdigit() and len(t) > 1]


def _value_tokens(question):
    """Tokens that denote filter *values*, not schema concepts: quoted
    literals, mid-sentence proper nouns, and ALL-CAPS codes ('SLE')."""
    vals = set()
    for m in re.finditer(r"'([^']+)'", question):
        vals.update(re.findall(r"[a-z0-9]+", m.group(1).lower()))
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", question):
        prev = question[:m.start()].rstrip()
        if prev and prev[-1] not in ".!?":
            vals.add(m.group(1).lower())
    for m in re.finditer(r"\b([A-Z]{2,})\b", question):
        vals.add(m.group(1).lower())
    return vals


def _lexicon(db_tables):
    lex = set()
    for t, tinfo in db_tables.items():
        lex |= set(_stem(w) for w in tinfo["name_tokens"])
        for col, ctoks in tinfo["column_index"].items():
            lex |= set(_stem(w) for w in ctoks)
    return lex


def _gap_tokens(question, intent, synonyms, db_tables, db_id):
    lex = _lexicon(db_tables)
    values = _value_tokens(question)
    skip = set(_stem(w) for w in FORMULA_WORDS | MEASURE_FILLERS) | values
    trig_words = set()
    for trigs, _c in CONCEPT_TRIGGERS.get(db_id, []):
        for w in trigs:
            trig_words.update(_stem(x)
                              for x in re.findall(r"[a-z0-9]+", w))
    gaps = []
    for t in _content_tokens(question, synonyms):
        st = _stem(t)
        if st in lex or st in trig_words or st in skip:
            continue
        gaps.append(t)
    return gaps


def _agg_signature(sql):
    s = sql.lower()
    fns = sorted(f for f in ("avg", "sum", "count", "min", "max")
                 if f + "(" in s)
    return (fns, "group by" in s)


def localize(entry, trace, mres, agent, db_tables):
    """Return (failure_mode, stage, evidence). stage None => no failure."""
    s5 = trace["s5"]
    intent = trace["s1"]
    db_id = entry["db_id"]

    if s5["error"]:
        return ("EXECUTION_ERROR", "S5", "SQLite raised %s" % s5["error"])

    if mres["ex"] == 1:
        return ("no failure", None, "EX=1: agent result matches gold")

    gaps = _gap_tokens(entry["question"], intent, agent.synonyms,
                       db_tables, db_id)
    # S2 gate (corrected): SCHEMA_LINK_MISS requires an actual missed
    # required gold table or column -- a lexical gap token alone is not
    # enough (unfamiliar question words like 'born'/'age'/'opened' are not
    # evidence of a linking failure when the plan covered the schema).
    # Lexical gaps only explain a verified required-schema miss.
    missed_tables = sorted(set(entry["gold_tables"])
                           - set(trace["s2"]["tables"]))
    linked_cols = {"%s.%s" % (t, c)
                   for t, cols in trace["s2"]["columns"].items()
                   for c in cols}
    missed_cols = sorted(set(entry["gold_columns"]) - linked_cols)
    if (missed_tables or missed_cols) and gaps:
        return (
            "SCHEMA_LINK_MISS", "S2",
            "plan missed required table(s) %s / column(s) %s; unexplained "
            "by known tokens, consistent with unresolved term(s) %s"
            % (missed_tables or "-", missed_cols or "-",
               sorted(set(gaps))),
        )

    for concept in sorted(intent.get("concepts", [])):
        if concept in CATALOG and concept not in agent.glossary:
            return (
                "SEMANTIC_DEFINITION_VIOLATION", "S3",
                "concept '%s' has no governed definition (glossary v%s); "
                "the plan cannot be faithful to an undefined domain "
                "mapping" % (concept, agent.glossary_version),
            )

    if entry["gold_definitions"]:
        conf = mres["conformance"]
        if conf["missing_definitions"]:
            return (
                "SEMANTIC_DEFINITION_VIOLATION", "S3",
                "gold relies on governed definition(s) %s absent from "
                "glossary v%s" % (conf["missing_definitions"],
                                  agent.glossary_version),
            )
        if conf["score"] == 0:
            detail = "; ".join(conf["missing"])
            if not conf["shape_ok"]:
                return ("AGG_FILTER_ERROR", "S4",
                        "wrong aggregation shape. " + detail)
            return ("SEMANTIC_DEFINITION_VIOLATION", "S3",
                    "plan did not enforce governed definition(s) %s. %s"
                    % (entry["gold_definitions"], detail))

    plan_tables = set(trace["s3"]["tables"])
    missing_tables = sorted(set(entry["gold_tables"]) - plan_tables)
    if missing_tables:
        return (
            "JOIN_ERROR", "S3",
            "plan tables %s omit required %s"
            % (sorted(plan_tables), missing_tables),
        )

    if _agg_signature(trace["s4"]["sql"]) != _agg_signature(entry["gold_sql"]):
        return (
            "AGG_FILTER_ERROR", "S4",
            "agent aggregate signature %s != gold %s" % (
                _agg_signature(trace["s4"]["sql"]),
                _agg_signature(entry["gold_sql"])),
        )

    if s5["n_rows"] == 0:
        return ("EMPTY_RESULT", "S6",
                "agent returned 0 rows while gold is non-empty")

    return ("WRONG_RESULT", "S6",
            "result mismatch with no earlier-stage evidence")
