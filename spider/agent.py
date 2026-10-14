# Author: Prabhu Rangaraju
"""SpiderBaselineAgent: a deliberately WEAK rule-based 6-stage text-to-SQL agent.

The agent is evaluated against REAL Spider dev questions on REAL Spider sqlite
databases. The FRAMEWORK (evaluation/diagnosis/fix machinery) is what's under
test; this agent exists to give it real failures to find.

Known weaknesses (honest, observed -- no sabotage):
  W1. S2 schema linking is exact token overlap between question tokens and
      table/column NAME tokens -- no plural stemming, no synonyms. "singers"
      does not match table "singer"; "stations" does not match "stadium".
  W2. S1 filter extraction handles only a single comparison / single equality
      / BETWEEN / LIKE per question. OR-ed predicates, NOT IN, INTERSECT,
      EXCEPT and scalar-subquery filters are unreachable.
  W3. S1 value extraction is heuristic: quoted literals, capitalized tokens,
      year numbers. Case-mismatched DB values (e.g. 'france' vs 'France') are
      not normalized, so equality predicates often match zero rows.
  W4. S1 aggregate<->column pairing is proximity-based and frequently wrong;
      S4 can only emit flat SELECTs (no subqueries, no set operations).
  W5. Column disambiguation across tables is by S2 link score only; the wrong
      table's column is often chosen (e.g. CountryName vs Country).

Fix hooks: agent.synonyms (metadata layer), agent.stem_plurals (metadata),
agent.relaxed_filters (template layer). Each stage returns structured output
recorded in the trace dict.
"""

import re
import sqlite3
import time

from schema import SpiderSchema, join_path

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "in", "on", "by", "and",
    "or", "to", "how", "many", "much", "were", "was", "are", "do", "does",
    "we", "our", "each", "all", "show", "list", "which", "who", "whom",
    "per", "with", "than", "that", "this", "it", "as", "at", "from", "be",
    "been", "have", "has", "had", "more", "most", "top", "between", "its",
    "their", "there", "here", "s", "t", "d", "ll", "m", "re", "ve",
}

AGG_FUNCS = {
    "average": "AVG", "mean": "AVG",
    "minimum": "MIN", "min": "MIN",
    "maximum": "MAX", "max": "MAX",
    "total": "SUM", "sum": "SUM",
}

SUPERLATIVES = {
    "most", "fewest", "highest", "lowest", "smallest", "largest",
    "minimum", "maximum", "earliest", "latest", "youngest", "oldest",
    "best", "worst",
}


def _stem(tok):
    # Naive plural stemmer used ONLY when agent.stem_plurals is on (fix hook)
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


class SpiderBaselineAgent:
    def __init__(self, schemas):
        self.schemas = schemas            # db_id -> SpiderSchema
        self.synonyms = {}                # metadata-layer hook (W1)
        self.stem_plurals = False         # metadata-layer hook (W1)
        self.prune_subsumed = False       # metadata-layer hook (junction-table FP)
        self.drop_columnless = False      # metadata-layer hook (S3 precision)
        self.relaxed_filters = False      # template-layer hook (W3)
        self.version = "1.0"

    # ------------------------------------------------------------------ utils.
    @staticmethod
    def _tokens(question):
        return re.findall(r"[a-z0-9]+", question.lower())

    def _norm(self, tok):
        tok = self.synonyms.get(tok, tok)
        if self.stem_plurals:
            return _stem(tok)
        return tok

    def _content(self, question):
        return [t for t in (self._norm(x) for x in self._tokens(question))
                if t not in STOPWORDS and not t.isdigit()]

    # ------------------------------------------------------------- S1: intent.
    def s1_intent(self, question, schema):
        q = question.lower()
        toks = self._tokens(question)
        tset = set(toks)

        intent = {
            "qtype": "list", "distinct": False, "aggs": [],
            "group_word": None, "order_word": None, "order_dir": None,
            "limit": None, "filters": [], "select_words": [],
        }

        if "distinct" in tset or "different" in tset or "unique" in tset:
            intent["distinct"] = True

        # ---- qtype.
        if re.search(r"\bhow many\b|\btotal number\b|\bnumber of\b", q):
            intent["qtype"] = "count"
        elif (re.search(r"\btop\s+\d+", q) or
              (SUPERLATIVES & tset and
               re.search(r"\b(singer|car|model|stadium|concert|country|year)\b", q))):
            intent["qtype"] = "topn"
        elif AGG_FUNCS.keys() & tset:
            intent["qtype"] = "aggregation"

        # ---- aggregation pairs (func -> nearby column word)
        for i, tok in enumerate(toks):
            if tok in AGG_FUNCS:
                col = self._near_column(toks, i, schema)
                if col:
                    intent["aggs"].append((AGG_FUNCS[tok], col))

        # ---- group-by cue.
        m = re.search(r"\bfor each (\w+)|\beach (\w+)|\bper (\w+)|\bin each (\w+)",
                      q)
        if m:
            intent["group_word"] = next(g for g in m.groups() if g)

        # ---- order-by cue.
        m = re.search(r"\bordered by (\w+)|\bsorted by (\w+)", q)
        if m:
            intent["order_word"] = next(g for g in m.groups() if g)
            if "descending" in tset or "desc" in tset:
                intent["order_dir"] = "DESC"
            elif "ascending" in tset or "asc" in tset:
                intent["order_dir"] = "ASC"
            else:
                intent["order_dir"] = "ASC"
        elif "oldest to the youngest" in q or "oldest to youngest" in q:
            intent["order_word"] = "age"
            intent["order_dir"] = "DESC"
        elif "youngest to the oldest" in q:
            intent["order_word"] = "age"
            intent["order_dir"] = "ASC"

        # ---- limit.
        m = re.search(r"\btop\s+(\d+)", q)
        if m:
            intent["limit"] = int(m.group(1))
        elif intent["qtype"] == "topn":
            intent["limit"] = 1

        # ---- select-column cue words: nouns near "show"/"list"/"name"
        for i, tok in enumerate(toks):
            if tok in ("name", "names", "country", "countries", "age",
                       "location", "capacity", "theme", "model", "models",
                       "make", "year", "song"):
                intent["select_words"].append(tok)

        # ---- filters.
        intent["filters"] = self._extract_filters(question, toks, schema)
        return intent

    def _col_lookup(self, word, schema):
        """Map a question word to a (table, column) via name-token overlap."""
        word = self._norm(word)
        best = None
        for t, cols in schema.tables.items():
            for c in cols:
                if word in schema.column_tokens(c):
                    score = 1 + (0.5 if word in schema.table_tokens(t) else 0)
                    if best is None or score > best[0]:
                        best = (score, (t, c))
        return best[1] if best else None

    def _near_column(self, toks, i, schema):
        for j in range(max(0, i - 3), min(len(toks), i + 4)):
            if j == i:
                continue
            w = toks[j]
            if w in STOPWORDS or w in AGG_FUNCS or w in SUPERLATIVES:
                continue
            hit = self._col_lookup(w, schema)
            if hit:
                return hit
        return None

    def _extract_filters(self, question, toks, schema):
        q = question.lower()
        filters = []

        def add(col, op, val):
            if col and not any(f[0] == col for f in filters):
                filters.append((col, op, val))

        # BETWEEN.
        m = re.search(r"between\s+(\d+)\s+and\s+(\d+)", q)
        if m:
            idx = q.index(m.group(0))
            before = self._tokens(q[max(0, idx - 40):idx])
            col = self._near_column(before, len(before) - 1, schema) \
                if before else None
            if col:
                add(col, "BETWEEN", (int(m.group(1)), int(m.group(2))))

        # Numeric comparisons: "<colword> above|below|... <num>" or "<cue> <colword> <num>"
        for pat, op in [
            (r"(\w+)\s+(?:above|over|older than|greater than|more than)\s+(?:the\s+)?(\w+\s+)?(\d+)", ">"),
            (r"(\w+)\s+(?:below|under|less than|younger than|smaller than)\s+(?:the\s+)?(\w+\s+)?(\d+)", "<"),
            (r"(\w+)\s+at least\s+(\d+)", ">="),
        ]:
            for m in re.finditer(pat, q):
                cands = [m.group(1)]
                g2 = m.group(2) if m.lastindex and m.lastindex >= 2 else None
                if g2 and not g2.strip().isdigit():
                    cands.append(g2.strip())
                col, num = None, None
                for cw in cands:
                    hit = self._col_lookup(cw, schema)
                    if hit:
                        col = hit
                        break
                nums = re.findall(r"\d+", m.group(0))
                num = int(nums[-1]) if nums else None
                if col and num is not None:
                    add(col, op, num)

        # Year equality / ranges.
        for m in re.finditer(r"\bin\s+(?:the\s+)?year(?:s)?\s+of\s+both\s+(\d{4})\s+and\s+(\d{4})", q):
            col = self._col_lookup("year", schema)
            if col:
                add(col, "IN", (int(m.group(1)), int(m.group(2))))
        if not any(f[1] == "IN" for f in filters):
            for m in re.finditer(r"(\d{4})\s+or\s+(\d{4})", q):
                col = self._col_lookup("year", schema)
                if col:
                    add(col, "IN", (int(m.group(1)), int(m.group(2))))
        for pat, op in [
            (r"in\s+(?:the\s+)?year\s+(\d{4})\s+or\s+after", ">="),
            (r"after\s+(\d{4})", ">"),
            (r"\bin\s+(\d{4})\b", "="),
            (r"year\s*(?:of\s+|was\s+)?(\d{4})", "="),
        ]:
            for m in re.finditer(pat, q):
                col = self._col_lookup("year", schema)
                if col:
                    add(col, op, int(m.group(1)))

        # Quoted literal -> LIKE on a name-ish column.
        for m in re.finditer(r"'([^']+)'", question):
            val = m.group(1)
            if val.isdigit():
                continue
            col = self._col_lookup("song", schema) or \
                self._col_lookup("name", schema)
            if col:
                add(col, "LIKE", val)

        # Proper-noun equality: "from France", "in France"
        for m in re.finditer(r"\b(?:from|in|of the)\s+([A-Z][A-Za-z]+)", question):
            val = m.group(1)
            if val.lower() in STOPWORDS:
                continue
            window = toks[max(0, toks.index(val.lower()) - 5):
                          toks.index(val.lower()) + 1]
            col = self._near_column(window, len(window) - 1, schema)
            if col:
                add(col, "=", val)
        return filters

    # ---------------------------------------------------- S2: schema linking.
    def s2_schema_linking(self, question, intent, schema):
        # Table matching uses stemming when agent.stem_plurals is on (fix
        # hook); column matching is ALWAYS exact -- stemming columns adds
        # false positives (e.g. 'names' pulling Name into every SELECT).
        save = self.stem_plurals
        # Table matching stems iff the flag is on (both sides consistently)
        t_content = set(self._content(question))
        t_tok = {t: {self._norm(x) for x in schema.table_tokens(t)}
                 for t in schema.tables}
        # Column matching is ALWAYS exact: temporarily disable stemming.
        self.stem_plurals = False
        c_content = set(self._content(question))
        c_tok = {}
        for t in schema.tables:
            for c in schema.tables[t]:
                c_tok[(t, c)] = {self._norm(x)
                                 for x in schema.column_tokens(c)}
        self.stem_plurals = save
        tscores = {t: len(t_content & t_tok[t]) for t in schema.tables}
        ranked = [t for t in schema.tables if tscores[t] > 0]
        ranked.sort(key=lambda t: (-tscores[t], t))
        linked = ranked[:4]
        if self.prune_subsumed:
            # Fix hook (metadata layer): a lower-ranked table is kept only
            # if it contributes a NEW matching token beyond the tables
            # already kept. Without this, junction tables such as
            # singer_in_concert link on a shared token ('singer') and the
            # extra JOIN silently changes COUNT(*) semantics.
            linked, covered = [], set()
            for t in ranked:
                matched = t_content & t_tok[t]
                if matched - covered:
                    linked.append(t)
                    covered |= matched
            linked = linked[:4]
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

    # ---------------------------------------------------------- S3: planning.
    def s3_planning(self, s2, intent, schema):
        tables = list(s2["tables"])
        # Intent-required tables (from filters / aggs / group / order words)
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
        if self.drop_columnless:
            # Fix hook (metadata layer): precision guard. Drop an S2-linked
            # table that has no linked columns and that no intent signal
            # requires -- typically a false positive from an ambiguous stem
            # (e.g. 'cars' matching car_makers). If nothing survives, fall
            # back to the intent-required tables, else the top-ranked table.
            kept = [t for t in tables
                    if s2["columns"].get(t) or t in need]
            tables = kept or sorted(need) or tables[:1]
        for t in sorted(need):
            if t not in tables:
                tables.append(t)
        ordered, joins = join_path(schema, tables)
        # Drop disconnected leftovers recorded by the cross-join fallback.
        pruned = [t for _, t, c in joins if c == "1=1"]
        if pruned:
            ordered = [t for t in ordered if t not in pruned]
            joins = [j for j in joins if j[2] != "1=1"]
        return {"tables": ordered, "joins": joins, "pruned": pruned,
                "intent": intent}

    # --------------------------------------------------- S4: SQL generation.
    @staticmethod
    def _q(ident):
        return '"%s"' % ident.replace('"', '""')

    def _from_sql(self, plan):
        if not plan["tables"]:
            return None
        first = plan["tables"][0]
        sql = "%s %s" % (self._q(first), self._q(first))
        for _l, nxt, cond in plan["joins"]:
            sql += " JOIN %s %s ON %s" % (self._q(nxt), self._q(nxt), cond)
        return sql

    def _where_sql(self, intent):
        parts = []
        for (t, c), op, val in intent["filters"]:
            col = "%s.%s" % (self._q(t), self._q(c))
            if op == "BETWEEN":
                parts.append("%s BETWEEN %s AND %s" % (col, val[0], val[1]))
            elif op == "IN":
                parts.append("%s IN (%s)" % (
                    col, ", ".join(str(v) for v in val)))
            elif op == "LIKE":
                v = val.replace("'", "''")
                parts.append("%s LIKE '%%%s%%'" % (col, v))
            elif op == "=" and isinstance(val, str):
                v = val.replace("'", "''")
                if self.relaxed_filters:
                    parts.append("LOWER(%s) = LOWER('%s')" % (col, v))
                else:
                    parts.append("%s = '%s'" % (col, v))
            else:
                parts.append("%s %s %s" % (col, op, val))
        return " AND ".join(parts)

    def _select_cols(self, plan, intent, schema, for_topn=False):
        cols = []
        seen = set()
        if intent["qtype"] == "aggregation" and intent["aggs"]:
            for func, (t, c) in intent["aggs"]:
                if t in plan["tables"]:
                    key = (t, c, func)
                    if key not in seen:
                        cols.append("%s(%s.%s)" % (
                            func, self._q(t), self._q(c)))
                        seen.add(key)
            if cols:
                return cols
        # Group column first (matches gold shape col, count(*))
        g = intent.get("_group_col")
        if g and g[0] in plan["tables"]:
            cols.append("%s.%s" % (self._q(g[0]), self._q(g[1])))
            seen.add((g[0], g[1]))
        # Linked columns.
        for t in plan["tables"]:
            for c in plan["s2cols"].get(t, []):
                if (t, c) not in seen:
                    cols.append("%s.%s" % (self._q(t), self._q(c)))
                    seen.add((t, c))
                if len(cols) >= 4:
                    break
            if len(cols) >= 4:
                break
        if not cols:
            cols = ["*"]
        return cols

    def s4_sql_generation(self, question, intent, plan, s2, schema):
        frm = self._from_sql(plan)
        if frm is None:
            return "SELECT 1 WHERE 0"  # no tables linked: valid, empty
        plan["s2cols"] = s2["columns"]
        where = self._where_sql(intent)
        qt = intent["qtype"]

        if qt == "count":
            sel = []
            g = intent.get("_group_col")
            if g and g[0] in plan["tables"]:
                sel.append("%s.%s" % (self._q(g[0]), self._q(g[1])))
            sel.append("COUNT(*)")
            sql = "SELECT %s FROM %s" % (", ".join(sel), frm)
            if where:
                sql += " WHERE " + where
            if g and g[0] in plan["tables"]:
                sql += " GROUP BY %s.%s" % (self._q(g[0]), self._q(g[1]))
            return sql

        if qt == "topn":
            cols = self._select_cols(plan, intent, schema, for_topn=True)
            sql = "SELECT %s%s FROM %s" % (
                "DISTINCT " if intent["distinct"] else "",
                ", ".join(cols), frm)
            if where:
                sql += " WHERE " + where
            o = intent.get("_order_col")
            if o and o[0] in plan["tables"]:
                sql += " ORDER BY %s.%s %s" % (
                    self._q(o[0]), self._q(o[1]),
                    intent["order_dir"] or "ASC")
            if intent["limit"]:
                sql += " LIMIT %d" % intent["limit"]
            return sql

        # List / aggregation.
        cols = self._select_cols(plan, intent, schema)
        sql = "SELECT %s%s FROM %s" % (
            "DISTINCT " if intent["distinct"] else "",
            ", ".join(cols), frm)
        if where:
            sql += " WHERE " + where
        g = intent.get("_group_col")
        if g and g[0] in plan["tables"] and qt != "count":
            if not any(x.startswith("COUNT(") for x in cols):
                cols.append("COUNT(*)")
                sql = "SELECT %s%s FROM %s" % (
                    "DISTINCT " if intent["distinct"] else "",
                    ", ".join(cols), frm)
                if where:
                    sql += " WHERE " + where
            sql += " GROUP BY %s.%s" % (self._q(g[0]), self._q(g[1]))
        o = intent.get("_order_col")
        if o and o[0] in plan["tables"]:
            sql += " ORDER BY %s.%s %s" % (
                self._q(o[0]), self._q(o[1]), intent["order_dir"] or "ASC")
        if intent["limit"]:
            sql += " LIMIT %d" % intent["limit"]
        return sql

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
    def answer(self, question, db_id):
        schema = self.schemas[db_id]
        intent = self.s1_intent(question, schema)
        s2 = self.s2_schema_linking(question, intent, schema)
        plan = self.s3_planning(s2, intent, schema)
        sql = self.s4_sql_generation(question, intent, plan, s2, schema)
        s5 = self.s5_execution(sql, schema.db_path)
        s6 = self.s6_interpretation(question, intent, s5)
        return {"s1": intent, "s2": s2, "s3": plan, "s4": {"sql": sql},
                "s5": s5, "s6": s6, "agent_version": self.version,
                "db_id": db_id}
