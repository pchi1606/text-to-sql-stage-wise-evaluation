# Author: Prabhu Rangaraju
"""BirdSQLAgent: a deliberately weak RULE-BASED 6-stage text-to-SQL agent over
real BIRD benchmark databases.

The framework under test evaluates *any* agent; this baseline is intentionally
weak so the evaluation/diagnosis/fix machinery has real failures to find.
Genuine (not sabotaged) weaknesses, each documented where implemented:
  W1. S2 has no synonym handling (agent.synonyms starts empty): paraphrases
      such as 'pupils'/'enrollees' match no column.
  W2. S2/S4 cannot resolve evidence-defined *value* mappings ('normal range
      of LDH', 'underage', 'severe thrombosis'): evidence text is used for
      token overlap in S2 only; S4 never parses evidence formulas.
  W3. S4 guesses the measure column by token overlap and applies one template
      per qtype: ratio/percentage/evidence formulas (DIVIDE(...), year
      arithmetic, CASE value-maps) are inexpressible and fall back to a
      plain aggregate.
  W4. S4 filter literals are mapped to columns by crude preceding-token
      overlap; filters with no overlap are silently dropped.
  W5. S4 cannot build subqueries; ordinal top-n ('10th and 11th') and
      multi-scalar gold shapes are out of reach.
  W6. S3 joins unreachable tables with a comma fallback (cartesian risk).

Fix hooks: agent.synonyms (metadata layer), agent.glossary /
agent.use_glossary_enforcement (glossary layer). Each stage returns structured
output recorded in the trace dict. Architecture mirrors the prototype's
agent.py (S1 intent, S2 schema linking, S3 planning, S4 SQL generation,
S5 execution, S6 interpretation).
"""

import re
import sqlite3
import time

from bird_glossary import (GLOSSARY, GLOSSARY_VERSION, CATALOG,
                            detect_concepts)
from schema_inventory import fk_graph

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "in", "on", "by", "and",
    "or", "to", "how", "many", "much", "were", "was", "are", "do", "does",
    "did", "we", "our", "each", "all", "show", "list", "which", "who",
    "whom", "whose", "why", "per", "with", "than", "that", "this", "it",
    "as", "at", "from", "be", "have", "has", "had", "there", "their",
    "they", "them", "he", "she", "his", "her", "its", "between", "among",
    "over", "under", "more", "most", "least", "less", "top", "please",
    "state", "give", "name", "names", "provide", "identify", "indicate",
    "also", "if", "so", "such", "no", "not", "but", "into", "out", "up",
    "down", "when", "where", "been", "being", "both", "either", "neither",
    "whether", "while", "during", "after", "before",
    # Handled by the year/date extractors, not schema linking.
    "year", "years",
    # Comparative/formula words owned by S1 intent, not S2 linking.
    "greater", "higher", "lower", "fewer", "difference", "deviation",
    "increase", "decrease",
    # Generic filler nouns/demonstratives/number-words (literals, not schema)
    "communication", "done", "ever", "those", "full", "just",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
}

AGG_WORDS = {
    "average": "AVG", "avg": "AVG", "total": "SUM", "sum": "SUM",
    "maximum": "MAX", "max": "MAX", "minimum": "MIN", "min": "MIN",
    "highest": "MAX", "lowest": "MIN", "ratio": "AVG", "percentage": "AVG",
    "percent": "AVG", "rate": "AVG",
}

COUNT_PHRASES = ("how many", "number of", "count of")

NUMERIC_TYPES = {"INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE"}


def _quote_ident(name):
    return '"%s"' % name.replace('"', '""')


def _quote_lit(value):
    return "'%s'" % str(value).replace("'", "''")


class BirdSQLAgent:
    def __init__(self, inventory):
        self.inv = inventory                       # db_id -> schema
        self.synonyms = {}                         # metadata hook (W1)
        self.glossary = dict(GLOSSARY)             # glossary hook
        self.glossary_version = GLOSSARY_VERSION
        self.use_glossary_enforcement = False      # glossary hook
        self.use_evidence_predicates = False       # S4 evidence hook (W2/W3)
        self.version = "1.0"

    # ------------------------------------------------------------------ utils.
    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _stem(w):
        # Lightweight stemmer so plural/verb variants ('patients', 'scores',
        # 'diagnosed') meet their lexicon forms. Conservative: short words
        # and ss/us/is/as/os endings are left alone; 'es' is stripped only
        # after sibilants (boxes->box) so 'scores'->'score', not 'scor'.
        if len(w) > 4:
            if w.endswith("ies"):
                return w[:-3] + "y"
            if w.endswith(("ses", "xes", "zes", "ches", "shes")):
                return w[:-2]
            if w.endswith("s") and not w.endswith(
                    ("ss", "us", "is", "as", "os")):
                return w[:-1]
        if len(w) > 5:
            if w.endswith("ed"):
                return w[:-2]
            if w.endswith("ing"):
                return w[:-3]
        return w

    def _expanded(self, text):
        return [self.synonyms.get(t, t) for t in self._tokens(text)]

    def _content(self, text):
        # Stemmed content tokens: the vocabulary S2 matches against.
        # Single-character tokens (possessive "'s" fragments) are noise.
        return [self._stem(t) for t in self._expanded(text)
                if t not in STOPWORDS and not t.isdigit()
                and len(t) > 1]

    # ------------------------------------------------------------- S1: intent.
    def s1_intent(self, question, db_id):
        q = question.lower()
        tokens = self._expanded(question)
        tset = set(t for t in tokens if t not in STOPWORDS)

        qtype = self._classify_qtype(q, tset)
        concepts = detect_concepts(question, db_id)
        aggfn = self._aggfn(q, tset, qtype)
        topn, direction = self._topn(q)
        group_phrase = self._group_phrase(q)
        literals, comparisons, betweens, years = self._extract_filters(
            question)

        return {
            "qtype": qtype,
            "concepts": concepts,
            "aggfn": aggfn,
            "topn": topn,
            "direction": direction,
            "group_phrase": group_phrase,
            "literals": literals,
            "comparisons": comparisons,
            "betweens": betweens,
            "years": years,
            "keywords": sorted(tset),
        }

    @staticmethod
    def _classify_qtype(q, tset):
        if any(p in q for p in COUNT_PHRASES):
            return "count"
        if re.search(r"\btop (\d+)\b", q) or \
                re.search(r"\b(\d+)(st|nd|rd|th)\b", q):
            return "topn"
        if q.startswith(("which ", "who ")) and \
                ({"highest", "lowest", "most", "least", "biggest",
                   "smallest", "largest"} & tset):
            return "topn"
        if AGG_WORDS.keys() & tset:
            return "aggregation"
        if q.startswith(("what is the", "what was the", "what are the",
                         "what is a", "what is an")):
            return "scalar"
        return "list"

    @staticmethod
    def _aggfn(q, tset, qtype):
        for w, fn in AGG_WORDS.items():
            if w in tset:
                return fn
        if qtype == "topn":
            if {"lowest", "least", "smallest"} & tset:
                return "MIN"
            return "MAX"
        return None

    @staticmethod
    def _topn(q):
        m = re.search(r"\btop (\d+)\b", q)
        if m:
            n = int(m.group(1))
        else:
            m2 = re.search(r"\b(\d+)(st|nd|rd|th)\b", q)
            n = int(m2.group(1)) if m2 else 1
        direction = "ASC" if re.search(
            r"\b(lowest|least|smallest|minimum)\b", q) else "DESC"
        return n, direction

    @staticmethod
    def _group_phrase(q):
        m = re.search(
            r"\b(?:by|per|each|of each|for each)\s+([a-z][a-z \-']{2,40}?)"
            r"(?:,|\?|\.|$| with | where | and | that | who )", q)
        return m.group(1).strip() if m else None

    def _extract_filters(self, question):
        """Return (literals, comparisons, betweens, years).

        Each literal: {raw, before} where before = the 5 tokens preceding it.
        Comparisons: {op, value, before}. Betweens: {lo, hi, before}.
        Years: {year, before}.
        """
        toks = self._tokens(question)
        literals, comparisons, betweens, years = [], [], [], []

        def before(pos, n=5):
            return toks[max(0, pos - n):pos]

        for m in re.finditer(r"'([^']{1,60})'", question):
            literals.append({"raw": m.group(1), "before": before(
                len(self._tokens(question[:m.start()])), ), "kind": "string"})

        for m in re.finditer(
                r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{4}/\d{1,2}/\d{1,2})\b", question):
            literals.append({"raw": m.group(1),
                             "before": before(len(self._tokens(
                                 question[:m.start()]))), "kind": "date"})

        for m in re.finditer(
                r"\b(greater than|more than|over|less than|fewer than|under|"
                r"below|above|at least|at most)\s+(\d+(?:\.\d+)?)", question,
                re.IGNORECASE):
            word = m.group(1).lower()
            op = (">" if word in ("greater than", "more than", "over",
                                  "above") else
                  "<" if word in ("less than", "fewer than", "under",
                                  "below") else
                  ">=" if word == "at least" else "<=")
            comparisons.append({"op": op, "value": m.group(2),
                                "before": before(len(self._tokens(
                                    question[:m.start()])))})
        for m in re.finditer(
                r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=[.?,;]|$)", question,
                re.IGNORECASE):
            nums = re.findall(r"\d[\d/]*", m.group(1) + " " + m.group(2))
            if len(nums) >= 2:
                betweens.append({"lo": nums[0], "hi": nums[1],
                                 "before": before(len(self._tokens(
                                     question[:m.start()])))})
        for m in re.finditer(r"\bin\s+((?:19|20)\d{2})\b", question,
                             re.IGNORECASE):
            years.append({"year": m.group(1), "before": before(
                len(self._tokens(question[:m.start()])))})
        # Bare numbers that are not part of dates (year-like or thresholds).
        for m in re.finditer(r"(?<![\w/])(\d+(?:\.\d+)?)(?![\w/])", question):
            raw = m.group(1)
            pos = len(self._tokens(question[:m.start()]))
            if any(c["value"] == raw for c in comparisons):
                continue
            literals.append({"raw": raw, "before": before(pos),
                             "kind": "number"})
        # Capitalized phrases (proper nouns) not at sentence start.
        # {2,}: single short capitals like 'Ig' are abbreviations, not
        # literals (baseline correction: they produced spurious filters).
        for m in re.finditer(
                r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+){0,3})\b",
                question):
            raw = m.group(1)
            if raw.lower() in STOPWORDS:
                continue
            pos = len(self._tokens(question[:m.start()]))
            literals.append({"raw": raw, "before": before(pos),
                             "kind": "string"})
        return literals, comparisons, betweens, years

    # ---------------------------------------------------- S2: schema linking.
    def s2_schema_linking(self, question, evidence, intent, db_id):
        content = self._content(question + " " + evidence)
        cset = set(content)
        db = self.inv[db_id]
        table_scores = {}
        for t, tinfo in db["tables"].items():
            toks = set(self._stem(w) for w in tinfo["name_tokens"])
            for col, ctoks in tinfo["column_index"].items():
                toks |= set(self._stem(w) for w in ctoks)
            table_scores[t] = len(cset & toks)
        # Count-entity boost: the noun after "how many"/"number of".
        count_entity = None
        m = re.search(r"(?:how many|number of)\s+([a-z ]+?)(?:\s|$)",
                      question.lower())
        if m:
            for w in self._tokens(m.group(1)):
                sw = self._stem(w)
                for t, tinfo in db["tables"].items():
                    names = set(self._stem(x)
                                for x in tinfo["name_tokens"])
                    if sw in names:
                        table_scores[t] += 2
                        count_entity = count_entity or t
        linked = [t for t in table_scores if table_scores[t] > 0]
        if not linked:  # fallback: nothing matched; take alphabetically first
            linked = [sorted(table_scores)[0]]
        linked.sort(key=lambda t: (-table_scores[t], t))
        linked = linked[:4]
        columns = {}
        for t in linked:
            scored = []
            ctoks_all = db["tables"][t]["column_index"]
            for col, ctoks in ctoks_all.items():
                s = len(cset & set(self._stem(w) for w in ctoks))
                if s > 0:
                    scored.append((s, col))
            scored.sort(key=lambda x: (-x[0], x[1]))
            cols = [c for _, c in scored[:6]]
            pk = db["tables"][t]["pk"]
            if pk and pk not in cols:
                cols.append(pk)
            columns[t] = cols
        return {"tables": linked, "columns": columns,
                "table_scores": table_scores,
                "count_entity": count_entity}

    # ---------------------------------------------------------- S3: planning.
    def s3_planning(self, s2, intent, db_id):
        tables = list(s2["tables"])
        graph = fk_graph(self.inv[db_id])
        # BFS from the top table to cover every linked table.
        seen = {tables[0]}
        edges = []          # (left, right, left_col, right_col)
        order = [tables[0]]
        for target in tables[1:]:
            path = self._bfs(graph, seen, target)
            if path is None:
                continue    # unreachable: comma fallback below (W6)
            for a, b, cond in path:
                m = re.match(r'"([^"]+)"\."([^"]+)" = "([^"]+)"\."([^"]+)"',
                             cond)
                lt, lc, rt, rc = m.groups()
                edges.append((lt, rt, lc, rc))
                if b not in seen:
                    seen.add(b)
                    order.append(b)
        comma = [t for t in tables if t not in seen]
        return {"tables": order + comma, "joins": edges, "comma": comma,
                "intent": intent}

    @staticmethod
    def _bfs(graph, seen, target):
        # BFS from any seen node to target; returns list of (a,b,cond).
        from collections import deque
        if target in seen:
            return []
        q = deque()
        prev = {}
        for s in sorted(seen):
            q.append(s)
            prev[s] = None
        while q:
            cur = q.popleft()
            for nb, cond in graph.get(cur, []):
                if nb in prev:
                    continue
                prev[nb] = (cur, cond)
                if nb == target:
                    path = []
                    node = nb
                    while prev[node] is not None:
                        par, c = prev[node]
                        path.append((par, node, c))
                        node = par
                    return path[::-1]
                q.append(nb)
        return None

    # --------------------------------------------------- S4: SQL generation.
    def _resolve_column(self, plan, db_id, keywords, types=None,
                        exclude_ids=True):
        """Best-overlap column across plan tables; None if no overlap."""
        db = self.inv[db_id]
        kset = set(self._stem(w) for w in keywords)
        best, best_s = None, 0
        for t in plan["tables"]:
            for col in db["tables"][t]["column_index"]:
                ctype = next(c["type"] for c in db["tables"][t]["columns"]
                             if c["name"] == col)
                if types and ctype not in types:
                    continue
                ctoks = set(self._stem(w) for w in
                            db["tables"][t]["column_index"][col])
                if exclude_ids and col.lower().endswith("id") and \
                        len(kset & ctoks) == 0:
                    continue
                s = len(kset & ctoks)
                if s > best_s:
                    best, best_s = (t, col), s
        return best

    def _from_sql(self, plan):
        alias = {t: "a%d" % i for i, t in enumerate(plan["tables"])}
        first = plan["tables"][0]
        sql = "%s %s" % (_quote_ident(first), alias[first])
        seen = {first}
        for lt, rt, lc, rc in plan["joins"]:
            nxt, prv = (rt, lt) if lt in seen else (lt, rt)
            ncol, pcol = (rc, lc) if lt in seen else (lc, rc)
            sql += " JOIN %s %s ON %s.%s = %s.%s" % (
                _quote_ident(nxt), alias[nxt], alias[nxt],
                _quote_ident(ncol), alias[prv], _quote_ident(pcol))
            seen.add(nxt)
        for t in plan["tables"]:
            if t not in seen:  # W6: comma fallback for unreachable tables
                sql += ", %s %s" % (_quote_ident(t), alias[t])
                seen.add(t)
        return sql, alias

    def _where_sql(self, intent, plan, alias, db_id, question="",
                   evidence=""):
        db = self.inv[db_id]
        conds = []
        text_types = {"TEXT", "VARCHAR", "CHAR", "DATE", "DATETIME"}
        for lit in intent["literals"]:
            before = [w for w in lit["before"] if w not in STOPWORDS]
            if lit["kind"] == "number":
                col = self._resolve_column(plan, db_id, before,
                                           types=NUMERIC_TYPES)
                if col:
                    t, c = col
                    conds.append("%s.%s = %s" % (
                        alias[t], _quote_ident(c), lit["raw"]))
            else:
                col = self._resolve_column(plan, db_id, before,
                                           types=text_types)
                if col:
                    t, c = col
                    conds.append("%s.%s = %s" % (
                        alias[t], _quote_ident(c),
                        _quote_lit(lit["raw"])))
        for y in intent["years"]:
            before = [w for w in y["before"] if w not in STOPWORDS]
            col = self._resolve_column(plan, db_id, before,
                                       types=text_types)
            if col:
                t, c = col
                conds.append("strftime('%%Y', %s.%s) = '%s'" % (
                    alias[t], _quote_ident(c), y["year"]))
        for cp in intent["comparisons"]:
            before = [w for w in cp["before"] if w not in STOPWORDS]
            col = self._resolve_column(plan, db_id, before,
                                       types=NUMERIC_TYPES)
            if col:
                t, c = col
                conds.append("%s.%s %s %s" % (
                    alias[t], _quote_ident(c), cp["op"], cp["value"]))
        for b in intent["betweens"]:
            before = [w for w in b["before"] if w not in STOPWORDS]
            col = self._resolve_column(
                plan, db_id, before,
                types=NUMERIC_TYPES | text_types)
            if col:
                t, c = col
                lo, hi = b["lo"], b["hi"]
                is_num = re.fullmatch(r"\d+(\.\d+)?", lo) and \
                    re.fullmatch(r"\d+(\.\d+)?", hi)
                conds.append("%s.%s BETWEEN %s AND %s" % (
                    alias[t], _quote_ident(c),
                    lo if is_num else _quote_lit(lo),
                    hi if is_num else _quote_lit(hi)))
        if self.use_glossary_enforcement:
            for concept in intent["concepts"]:
                spec = self.glossary.get(concept)
                if not spec:
                    continue
                preds = spec["required_predicates"]
                col_pred = next((p for p in preds
                                 if not (p.startswith("'") and
                                         p.endswith("'"))), None)
                val_pred = next((p for p in preds
                                 if p.startswith("'") and p.endswith("'")),
                                None)
                if not (col_pred and val_pred):
                    continue  # column-mapping concepts handled in _measure
                col = self._resolve_column(
                    plan, db_id, self._tokens(col_pred), types=None,
                    exclude_ids=False)
                if col:
                    t, c = col
                    frag = "%s.%s = %s" % (alias[t], _quote_ident(c),
                                           val_pred)
                    if frag not in conds:
                        conds.append(frag)
        # S4 evidence-predicate fix (off at baseline): parsed BIRD evidence
        # definitions become WHERE predicates.
        conds.extend(self._evidence_conds(question, evidence, plan, alias,
                                          db_id))
        # De-duplicate, keep order.
        seen_c, uniq = set(), []
        for c in conds:
            if c not in seen_c:
                seen_c.add(c)
                uniq.append(c)
        return " AND ".join(uniq)

    # --------------------------------- S4 fix: evidence predicate parsing.
    def _ev_col_expr(self, col, plan, alias, db_id):
        """Bind an evidence column name to an aliased expression.

        Handles year(COL) -> strftime('%Y', ...). Returns None when the
        column cannot be bound to a plan table (predicate skipped).
        """
        ym = re.match(r"(?i)^year\(\s*`?([A-Za-z_][\w ]*?)`?\s*\)$", col)
        inner = ym.group(1) if ym else col
        db = self.inv[db_id]
        hit = None
        cu = inner.upper()
        for t in plan["tables"]:
            for c in db["tables"][t]["columns"]:
                if c["name"].upper() == cu:
                    hit = (t, c["name"])
                    break
            if hit:
                break
        if hit is None:
            hit = self._resolve_column(plan, db_id, [inner], types=None,
                                       exclude_ids=False)
        if hit is None:
            return None
        t, c = hit
        expr = "%s.%s" % (alias[t], _quote_ident(c))
        if ym:
            expr = "strftime('%%Y', %s)" % expr
        return expr

    def _ev_val(self, val, col, translations):
        v = val.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", v):
            return v
        bare = v.strip("'\"")
        up = col.upper() if not col.upper().startswith("YEAR(") else col
        ym = re.match(r"(?i)^year\(\s*`?([A-Za-z_][\w ]*?)`?\s*\)$", col)
        key = ym.group(1).upper() if ym else col.upper()
        tmap = translations.get(key, {})
        starred = translations.get("*", {})
        return "'%s'" % tmap.get(bare, starred.get(bare, bare))

    def _ev_pred_sql(self, pred, plan, alias, db_id, translations):
        kind = pred[0]
        if kind == "and":
            parts = [self._ev_pred_sql(p, plan, alias, db_id,
                                       translations) for p in pred[1]]
            if any(p is None for p in parts):
                return None
            return " AND ".join(parts)
        if kind == "or":
            parts = [self._ev_pred_sql(p, plan, alias, db_id,
                                       translations) for p in pred[1]]
            if any(p is None for p in parts):
                return None
            return "(" + " OR ".join(parts) + ")"
        col = pred[1]
        expr = self._ev_col_expr(col, plan, alias, db_id)
        if expr is None:
            return None
        if kind == "cmp":
            return "%s %s %s" % (expr, pred[2],
                                 self._ev_val(pred[3], col, translations))
        if kind == "between":
            return "%s BETWEEN %s AND %s" % (
                expr, self._ev_val(pred[2], col, translations),
                self._ev_val(pred[3], col, translations))
        if kind == "in":
            vals = [self._ev_val("'%s'" % v, col, translations)
                    for v in pred[2]]
            return "%s IN (%s)" % (expr, ", ".join(vals))
        if kind == "null":
            return "%s IS %sNULL" % (expr, "NOT " if pred[2] else "")
        return None

    def _evidence_conds(self, question, evidence, plan, alias, db_id):
        """WHERE fragments from parsed BIRD evidence definitions.

        A mapping fires only when >= half its phrase tokens occur in the
        question; negated phrases require a negation in the question.
        """
        if not self.use_evidence_predicates or not evidence:
            return []
        from bird_evidence import parse_evidence, _NEG_RE
        mappings, translations, _distinct = parse_evidence(evidence)
        qtoks = set(self._content(question))
        qneg = bool(_NEG_RE.search(question))
        conds = []
        for mp in mappings:
            trig = mp["trigger"]
            # Synonym-aware: the agent's S2 vocabulary maps paraphrases
            # (e.g. women->female) before matching.
            trig_x = set(trig)
            for t in trig:
                trig_x.add(self.synonyms.get(t, t))
            hit = len(trig_x & qtoks)
            if hit == 0 or hit * 2 < len(trig):
                continue
            if mp["neg"] and not qneg:
                continue
            for pred in mp["preds"]:
                frag = self._ev_pred_sql(pred, plan, alias, db_id,
                                         translations)
                if frag:
                    conds.append(frag)
        return conds

    def _measure(self, intent, plan, db_id):
        # Glossary concept can pin the measure (e.g. AVG_SALARY -> A11).
        if self.use_glossary_enforcement:
            for concept in intent["concepts"]:
                if concept == "AVG_SALARY":
                    col = self._resolve_column(
                        plan, db_id, ["A11"], types=None,
                        exclude_ids=False)
                    if col:
                        return col
        kws = [w for w in intent["keywords"]
               if w not in AGG_WORDS and w not in STOPWORDS]
        return self._resolve_column(plan, db_id, kws, types=NUMERIC_TYPES)

    def _count_sql(self, plan, alias, db_id, evidence, from_sql, wc):
        # Evidence may explicitly require DISTINCT ("Should consider
        # DISTINCT in the final result"): count distinct keys of the
        # counted entity instead of joined rows.
        if self.use_evidence_predicates and evidence:
            from bird_evidence import parse_evidence
            _m, _t, distinct = parse_evidence(evidence)
            ent = plan.get("count_entity")
            if distinct and ent and ent in alias:
                pk = self.inv[db_id]["tables"][ent]["pk"]
                if pk:
                    return ("SELECT COUNT(DISTINCT %s.%s) AS n FROM %s%s"
                            % (alias[ent], _quote_ident(pk), from_sql,
                               wc))
        return "SELECT COUNT(*) AS n FROM %s%s" % (from_sql, wc)

    def s4_sql_generation(self, question, intent, plan, db_id, evidence=""):
        from_sql, alias = self._from_sql(plan)
        where = self._where_sql(intent, plan, alias, db_id, question,
                                evidence)
        wc = (" WHERE " + where) if where else ""
        qtype = intent["qtype"]

        if qtype == "count":
            return self._count_sql(plan, alias, db_id, evidence,
                                   from_sql, wc)

        if qtype == "list":
            cols = []
            for t in plan["tables"]:
                for c in self.s2_columns_for(plan, t)[:4]:
                    cols.append("%s.%s" % (alias[t], _quote_ident(c)))
            sel = ", ".join(cols) if cols else "*"
            return "SELECT %s FROM %s%s" % (sel, from_sql, wc)

        measure = self._measure(intent, plan, db_id)
        if measure:
            mt, mc = measure
            mexpr = "%s.%s" % (alias[mt], _quote_ident(mc))
        else:
            mexpr = None  # resolved per qtype below; never AVG(*)

        group = None
        if intent["group_phrase"]:
            gkws = [w for w in self._tokens(intent["group_phrase"])
                    if w not in STOPWORDS]
            group = self._resolve_column(plan, db_id, gkws, types=None)

        aggfn = intent["aggfn"] or "AVG"  # W3: fallback guess
        if qtype == "aggregation":
            if mexpr is None:
                # W3: no measurable column resolved; honest COUNT fallback.
                return "SELECT COUNT(*) AS v FROM %s%s" % (from_sql, wc)
            if group:
                gt, gc = group
                gexpr = "%s.%s" % (alias[gt], _quote_ident(gc))
                return ("SELECT %s, %s(%s) AS v FROM %s%s "
                        "GROUP BY %s" % (gexpr, aggfn, mexpr, from_sql, wc,
                                         gexpr))
            return "SELECT %s(%s) AS v FROM %s%s" % (
                aggfn, mexpr, from_sql, wc)

        if qtype == "topn":
            n = intent["topn"] or 1
            if group:
                gt, gc = group
                gexpr = "%s.%s" % (alias[gt], _quote_ident(gc))
                vexpr = "%s(%s)" % (aggfn, mexpr) if mexpr else "COUNT(*)"
                return ("SELECT %s, %s AS v FROM %s%s GROUP BY %s "
                        "ORDER BY v %s LIMIT %d" % (
                            gexpr, vexpr, from_sql, wc, gexpr,
                            intent["direction"], n))
            oexpr = mexpr or "%s.%s" % (
                alias[plan["tables"][0]],
                _quote_ident(self.s2_columns_for(
                    plan, plan["tables"][0])[0]))
            return ("SELECT %s FROM %s%s ORDER BY %s %s LIMIT %d" % (
                oexpr, from_sql, wc, oexpr, intent["direction"], n))

        # Scalar.
        if measure:
            return "SELECT %s AS v FROM %s%s LIMIT 1" % (
                mexpr, from_sql, wc)
        cols = []
        for t in plan["tables"]:
            for c in self.s2_columns_for(plan, t)[:2]:
                cols.append("%s.%s" % (alias[t], _quote_ident(c)))
        sel = cols[0] if cols else "*"
        return "SELECT %s FROM %s%s LIMIT 1" % (sel, from_sql, wc)

    def s2_columns_for(self, plan, table):
        cols = plan.get("s2_columns", {}).get(table, [])
        if cols:
            return cols
        db = self.inv[plan["db_id"]]
        return [c["name"] for c in db["tables"][table]["columns"][:4]]

    # -------------------------------------------------------- S5: execution.
    @staticmethod
    def s5_execution(sql, db_path):
        t0 = time.perf_counter()
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(sql)
            rows = [tuple(r) for r in cur.fetchall()]
            cols = [d[0] for d in cur.description] if cur.description else []
            con.close()
            ms = (time.perf_counter() - t0) * 1000.0
            return {"ok": True, "rows": rows, "columns": cols,
                    "n_rows": len(rows), "runtime_ms": ms, "error": None}
        except Exception as e:  # noqa: BLE001 - record any DB failure
            ms = (time.perf_counter() - t0) * 1000.0
            return {"ok": False, "rows": [], "columns": [], "n_rows": 0,
                    "runtime_ms": ms, "error": "%s: %s" % (type(e).__name__, e)}

    # ----------------------------------------------------- S6: interpretation.
    @staticmethod
    def s6_interpretation(question, intent, s5):
        flags = []
        if not s5["ok"]:
            flags.append("EXECUTION_FAILED")
        elif s5["n_rows"] == 0:
            flags.append("EMPTY_RESULT")
        if s5["ok"] and s5["n_rows"] > 0:
            answer = "Query returned %d row(s); first row: %s" % (
                s5["n_rows"], s5["rows"][0])
        else:
            answer = "Query returned %d row(s)." % s5["n_rows"]
        return {"flags": flags, "answer": answer,
                "n_rows": s5["n_rows"], "n_cols": len(s5["columns"])}

    # ------------------------------------------------------------------ run.
    def answer(self, question, evidence, db_id, db_path):
        intent = self.s1_intent(question, db_id)
        s2 = self.s2_schema_linking(question, evidence, intent, db_id)
        plan = self.s3_planning(s2, intent, db_id)
        plan["s2_columns"] = s2["columns"]
        plan["db_id"] = db_id
        plan["count_entity"] = s2.get("count_entity")
        sql = self.s4_sql_generation(question, intent, plan, db_id,
                                    evidence)
        s5 = self.s5_execution(sql, db_path)
        s6 = self.s6_interpretation(question, intent, s5)
        return {"s1": intent, "s2": s2, "s3": plan, "s4": {"sql": sql},
                "s5": s5, "s6": s6, "agent_version": self.version}
