# Author: Prabhu Rangaraju
"""General parser for BIRD evidence semantic definitions.

BIRD questions ship with an `evidence` string that defines terms as
predicates, e.g. "Ig G higher than normal refers to IGG >= 2000" or
"normal anti-SM refers to SM IN('-', '+-')".  The baseline agent (W2/W3)
deliberately never parses these; this module is the S4-layer fix that
does, gated behind agent.use_evidence_predicates.

Only exactly-parseable predicate shapes are honored; anything else
(formulas, prose, aggregations) is skipped, never guessed:
  COL OP number            (>, <, >=, <=, =, <>, !=; '< =' tolerated)
  COL = 'literal'          (single or double quotes)
  COL IN ('a', 'b', ...)
  COL IS [NOT] NULL
  COL BETWEEN a AND b
  year(COL) ...            (mapped to strftime('%Y', COL))
  conjunctions/disjunctions of the above with AND/OR
Value translations of the form "SM = 'negative' means '-'" are honored by
mapping the shorthand back to the full value inside IN-lists/equalities.

A mapping fires only when at least half of its (stemmed, stopword-free)
phrase tokens appear in the question.
"""

import re

from bird_agent import BirdSQLAgent

_STEM = BirdSQLAgent._stem
_STOP = None  # lazy import to avoid circulars


def _stopwords():
    global _STOP
    if _STOP is None:
        from bird_agent import STOPWORDS
        _STOP = STOPWORDS
    return _STOP


def _phrase_tokens(phrase):
    toks = re.findall(r"[a-z0-9]+", phrase.lower())
    return set(_STEM(t) for t in toks
               if t not in _stopwords() and len(t) > 1)


_CMP_RE = re.compile(
    r"^((?:year\([^)]*\)|[A-Za-z_][\w]*))\s*(>=|<=|<>|!=|=|>|<)\s*"
    r"(-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\")$", re.IGNORECASE)
_BETWEEN_RE = re.compile(
    r"^((?:year\([^)]*\)|[A-Za-z_][\w]*))\s+BETWEEN\s+"
    r"(-?\d+(?:\.\d+)?|'[^']*')\s+AND\s+"
    r"(-?\d+(?:\.\d+)?|'[^']*')$", re.IGNORECASE)
_IN_RE = re.compile(
    r"^([A-Za-z_][\w]*)\s+IN\s*\(([^)]*)\)$", re.IGNORECASE)
_NULL_RE = re.compile(
    r"^([A-Za-z_][\w]*)\s+IS\s+(NOT\s+)?NULL$", re.IGNORECASE)
_YEAR_FN_RE = re.compile(r"^year\(\s*`?([A-Za-z_][\w ]*?)`?\s*\)$",
                         re.IGNORECASE)


def _split_top(rhs, word):
    return [p.strip() for p in
            re.split(r"(?<![a-zA-Z])%s(?![a-zA-Z])" % word, rhs,
                     flags=re.IGNORECASE) if p.strip()]


def _norm_op(rhs):
    return re.sub(r"([<>!])\s+(=)", r"\1\2", rhs)


def _parse_simple(text):
    """Parse one predicate; return a pred tuple or None."""
    text = _norm_op(text.strip().replace("`", ""))
    m = _BETWEEN_RE.match(text)
    if m:
        return ("between", m.group(1), m.group(2), m.group(3))
    m = _IN_RE.match(text)
    if m:
        vals = [v.strip().strip("'\"") for v in m.group(2).split(",")]
        vals = [v for v in vals if v != ""]
        if vals:
            return ("in", m.group(1), vals)
        return None
    m = _NULL_RE.match(text)
    if m:
        return ("null", m.group(1), bool(m.group(2)))
    m = _CMP_RE.match(text)
    if m:
        col, op, val = m.group(1), m.group(2), m.group(3)
        return ("cmp", col, op, val)
    return None


def _parse_rhs(rhs):
    or_parts = _split_top(rhs, "or")
    if len(or_parts) > 1:
        branches = []
        for part in or_parts:
            ands = [_parse_simple(p) for p in _split_top(part, "and")]
            if any(a is None for a in ands):
                return []
            branches.append(("and", ands) if len(ands) > 1 else ands[0])
        return [("or", branches)]
    ands = [_parse_simple(p) for p in _split_top(rhs, "and")]
    if any(a is None for a in ands):
        return []
    return ands


_MEANS_RE = re.compile(r"(\w+)\s*=\s*'([^']+)'\s+means\s+'([^']+)'")
_MEANS_BARE_RE = re.compile(r"'([^']+)'\s+means\s+'([^']+)'")
_CLAUSE_RE = re.compile(r"(.+?)\s+(?:refers to|means)\s+(.+)$",
                        re.IGNORECASE | re.DOTALL)


_NEG_RE = re.compile(r"\b(not|no|never|without|n't)\b", re.IGNORECASE)


def parse_evidence(evidence):
    """Return (mappings, translations, distinct_hint).

    mappings: [{"trigger": set, "preds": [...], "phrase": str, "neg": bool}]
    translations: {COL_UPPER: {shorthand: full_value}}
    distinct_hint: evidence explicitly asks for DISTINCT in the result.
    """
    mappings, translations = [], {}
    if not evidence:
        return mappings, translations
    taken = []
    for m in _MEANS_RE.finditer(evidence):
        translations.setdefault(m.group(1).upper(), {})[m.group(3)] = \
            m.group(2)
        taken.append(m.span())
    for m in _MEANS_BARE_RE.finditer(evidence):
        # Skip bare matches already consumed by the COL = 'x' means 'y'
        # form (they would store the inverted direction)
        if any(s <= m.start() < e for s, e in taken):
            continue
        translations.setdefault("*", {})[m.group(1)] = m.group(2)
    for clause in evidence.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        m = _CLAUSE_RE.match(clause)
        if not m:
            continue
        phrase = m.group(1).strip().strip("'\"")
        rhs = m.group(2).strip()
        preds = _parse_rhs(rhs)
        if not preds:
            continue
        trigger = _phrase_tokens(phrase)
        if not trigger:
            continue
        mappings.append({"trigger": trigger, "preds": preds,
                         "phrase": phrase,
                         "neg": bool(_NEG_RE.search(phrase))})
    distinct_hint = bool(re.search(r"distinct", evidence, re.IGNORECASE))
    return mappings, translations, distinct_hint
