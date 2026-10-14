# Author: Prabhu Rangaraju
"""Failure diagnosis: localize a wrong answer to the responsible pipeline stage.

Taxonomy (mode -> stage):
  SCHEMA_LINK_MISS            -> S2  (a business term matched no table/column)
  SEMANTIC_DEFINITION_VIOLATION -> S3 (known concept has no governed definition,
                                       or the plan did not enforce one)
  JOIN_ERROR                  -> S3  (plan dropped a required table)
  AGG_FILTER_ERROR            -> S4  (wrong aggregation shape / filter)
  EXECUTION_ERROR             -> S5  (SQLite raised)
  EMPTY_RESULT / WRONG_RESULT -> S6  (interpretation stage: bad/empty answer)

Decision order is deliberate and documented: a DB error always wins (S5);
a correct end-to-end answer (EX=1) is never diagnosed as a failure even if
intermediate retrieval was imperfect; lexical gaps are attributed to S2;
missing governed definitions to S3; then conformance, join coverage, and
finally result-shape checks.
"""

import re

from agent import TABLE_KEYWORDS, COLUMN_HINTS, STOPWORDS
from glossary import CONCEPT_TO_DEF, SEMANTIC_CONCEPTS
from metrics import concept_definition_key

# Single-word concept tokens the intent stage can detect (used to keep known
# concepts out of the S2 "unresolved term" check).
CONCEPT_WORDS = {"revenue", "repeat"}


def _content_tokens(question, synonyms):
    toks = re.findall(r"[a-z0-9]+", question.lower())
    toks = [synonyms.get(t, t) for t in toks]
    return [t for t in toks if t not in STOPWORDS and not t.isdigit()]


def _gap_tokens(question, intent, synonyms):
    lexicon = set()
    for kw in TABLE_KEYWORDS.values():
        lexicon |= kw
    for colmap in COLUMN_HINTS.values():
        for hints in colmap.values():
            lexicon |= hints
    return [
        t for t in _content_tokens(question, synonyms)
        if t not in lexicon and t not in CONCEPT_WORDS
    ]


def localize(entry, trace, mres, agent):
    """Return (failure_mode, stage, evidence). stage is None when no failure."""
    s5 = trace["s5"]
    intent = trace["s1"]

    if s5["error"]:
        return ("EXECUTION_ERROR", "S5", "SQLite raised %s" % s5["error"])

    if mres["ex"] == 1:
        return ("no failure", None, "EX=1: agent result matches gold")

    gaps = _gap_tokens(entry["question"], intent, agent.synonyms)
    if gaps:
        return (
            "SCHEMA_LINK_MISS", "S2",
            "unresolved business term(s) %s: matched no table or column "
            "(no synonym handling)" % sorted(set(gaps)),
        )

    for concept in sorted(intent.get("concepts", [])):
        if concept in SEMANTIC_CONCEPTS:
            dkey = concept_definition_key(concept)
            if dkey and dkey not in agent.glossary:
                return (
                    "SEMANTIC_DEFINITION_VIOLATION", "S3",
                    "concept '%s' has no governed definition "
                    "(glossary v%s); the plan cannot be faithful to an "
                    "undefined metric" % (concept, agent.glossary_version),
                )

    if entry["gold_definitions"]:
        conf = mres["conformance"]
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
            "plan tables %s omit required %s (planner pruned a table)"
            % (sorted(plan_tables), missing_tables),
        )

    if s5["n_rows"] == 0:
        return ("EMPTY_RESULT", "S6",
                "agent returned 0 rows while gold is non-empty")

    return ("WRONG_RESULT", "S6",
            "result mismatch with no earlier-stage evidence")
