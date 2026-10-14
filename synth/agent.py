# Author: Prabhu Rangaraju
"""BaselineSQLAgent: a deliberately imperfect RULE-BASED 6-stage text-to-SQL agent.

The framework under test evaluates *any* agent; this baseline is intentionally
weak so the evaluation/diagnosis/fix machinery has real failures to find.
Deliberate weaknesses (each documented where implemented):
  W1. S2 has no synonym handling (agent.synonyms starts empty): business
      synonyms such as 'turnover'/'sales' match no table or column.
  W2. S3 prunes the lowest-confidence table whenever a plan has >2 non-hub
      tables (drops 'customers' on Q06).
  W3. S4 computes revenue WITHOUT the governed status='completed' predicate
      unless glossary enforcement is enabled.
  W4. S4's region-grouped revenue template references a non-existent column
      (c.regio) -- forces an S5 execution error on Q11.
  W5. S4 has no AVG_ORDER_VALUE / REPEAT_BUYER_RATE builders: it guesses
      (per-item AVG; COUNT(*) for 'repeat').

Fix hooks: agent.synonyms (metadata layer), agent.glossary /
agent.use_glossary_enforcement (glossary layer), agent.fix_joins /
agent.fix_typos (template layer). Each stage returns structured output recorded
in the trace dict.
"""

import re
import sqlite3
import time

from glossary import GLOSSARY, GLOSSARY_VERSION, CONCEPT_TO_DEF, SEMANTIC_CONCEPTS

ALIASES = {"customers": "c", "orders": "o", "order_items": "oi", "products": "p"}

TABLE_KEYWORDS = {
    "customers": {"customer", "customers", "client", "region", "name", "buyer", "buyers"},
    "orders": {
        "order", "orders", "purchase", "status", "cancelled", "completed",
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "year", "date", "month",
    },
    "order_items": {
        "item", "items", "quantity", "price", "amount", "total",
        "revenue", "value", "average",
    },
    "products": {"product", "products", "category", "electronics", "furniture",
                 "clothing", "groceries"},
}

COLUMN_HINTS = {
    "customers": {
        "customer_id": {"customer", "id"},
        "name": {"name"},
        "region": {"region"},
        "signup_date": {"date", "signup"},
    },
    "orders": {
        "order_id": {"order", "id"},
        "customer_id": {"customer", "id"},
        "order_date": {"date", "order", "month", "year"},
        "status": {"status", "cancelled", "completed"},
    },
    "order_items": {
        "item_id": {"item", "id"},
        "order_id": {"order", "id"},
        "product_id": {"product", "id"},
        "quantity": {"quantity"},
        "unit_price": {"price", "unit", "value", "amount"},
    },
    "products": {
        "product_id": {"product", "id"},
        "name": {"name"},
        "category": {"category"},
    },
}

# Foreign-key graph used by the S3 planner to build join paths.
FK_EDGES = [
    ("customers", "orders", "{c}.customer_id = {o}.customer_id"),
    ("orders", "order_items", "{o}.order_id = {oi}.order_id"),
    ("order_items", "products", "{oi}.product_id = {p}.product_id"),
]

STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "in", "on", "by", "and",
    "or", "to", "how", "many", "much", "were", "was", "are", "do", "we",
    "our", "each", "all", "show", "list", "which", "who", "per", "with",
    "than", "that", "this", "it", "as", "at", "from", "be", "have", "has",
    "more", "most", "top", "highest", "number", "fraction", "total",
    "average", "sum", "count", "names",
}

# Narrower stopword set used by S2 schema linking: only true function words,
# so content words such as 'total'/'average' still score table keywords.
S2_STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "in", "on", "by", "and",
    "or", "to", "how", "were", "was", "are", "do", "we", "our", "each",
    "all", "which", "who", "per", "with", "than", "that", "this", "it",
    "as", "at", "from", "be", "have", "has",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


class BaselineSQLAgent:
    def __init__(self):
        self.synonyms = {}                       # metadata-layer hook (W1)
        self.glossary = dict(GLOSSARY)           # glossary-layer hook
        self.glossary_version = GLOSSARY_VERSION
        self.use_glossary_enforcement = False    # glossary-layer hook (W3)
        self.fix_joins = False                   # template-layer hook (W2)
        self.fix_typos = False                   # template-layer hook (W4)
        self.version = "1.0"

    # ------------------------------------------------------------------ utils.
    @staticmethod
    def _tokens(question):
        return re.findall(r"[a-z0-9]+", question.lower())

    def _expanded(self, question):
        return [self.synonyms.get(t, t) for t in self._tokens(question)]

    # ------------------------------------------------------------- S1: intent.
    def s1_intent(self, question):
        q = question.lower()
        tokens = self._expanded(question)
        tset = set(tokens)

        concepts = set()
        if "revenue" in tset:
            concepts.add("revenue")
        if "average" in tset and "order" in tset and "value" in tset:
            concepts.add("avg_order_value")
        if "repeat" in tset:
            concepts.add("repeat")

        if "how many" in q:
            qtype = "count"
        elif "per order" in q or "each order" in q or "for each" in q:
            qtype = "list"
        elif "top" in tset or "most" in tset or "highest" in tset:
            qtype = "topn"
        elif {"average", "total", "sum", "fraction"} & tset:
            qtype = "aggregation"
        else:
            qtype = "list"

        group_dim = None
        if qtype in ("aggregation", "topn"):
            if "categor" in q:
                group_dim = "category"
            elif "region" in tset:
                group_dim = "region"
            elif ("customer" in tset or "customers" in tset) and qtype == "topn":
                group_dim = "customer_name"

        filt = None
        m = re.search(r"\b(north|south|east|west) region\b", q)
        if m:
            filt = ("region", m.group(1).capitalize())
        m2 = re.search(r"in the ([A-Z][a-zA-Z]+) category", question)
        if m2:
            filt = ("category", m2.group(1))
        if "cancelled" in tset:
            filt = ("status", "cancelled")

        time_window = None
        m3 = re.search(
            r"(january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+(\d{4})", q)
        if m3:
            mo, yr = MONTHS[m3.group(1)], int(m3.group(2))
            start = "%d-%02d-01" % (yr, mo)
            end = "%d-%02d-01" % (yr + (mo == 12), mo % 12 + 1)
            time_window = (start, end)

        topn = None
        m4 = re.search(r"top (\d+)", q)
        if m4:
            topn = int(m4.group(1))
        else:
            m5 = re.search(r"(\d+)\s+customers", q)
            if m5 and ("most" in tset or "highest" in tset):
                topn = int(m5.group(1))
        if qtype == "topn" and topn is None:
            topn = 1

        table_focus = None
        if "customer" in q:
            table_focus = "customers"
        elif "product" in q:
            table_focus = "products"
        elif "order" in q:
            table_focus = "orders"

        grain = "order" if ("per order" in q or "each order" in q or "for each" in q) else None

        return {
            "qtype": qtype,
            "concepts": sorted(concepts),
            "concept": sorted(concepts)[0] if concepts else None,
            "group_dim": group_dim,
            "filter": filt,
            "time_window": time_window,
            "topn": topn,
            "table_focus": table_focus,
            "grain": grain,
        }

    # ---------------------------------------------------- S2: schema linking.
    def s2_schema_linking(self, question, intent):
        tokens = self._expanded(question)
        content = [t for t in tokens if t not in S2_STOPWORDS and not t.isdigit()]
        cset = set(content)
        table_scores = {
            t: len(cset & kw) for t, kw in TABLE_KEYWORDS.items()
        }
        linked = [t for t in TABLE_KEYWORDS if table_scores[t] > 0]
        linked.sort(key=lambda t: -table_scores[t])
        linked = linked[:4]
        columns = {}
        for t in linked:
            scored = []
            for col, hints in COLUMN_HINTS[t].items():
                if cset & hints:
                    scored.append(col)
            columns[t] = scored[:6]
        return {"tables": linked, "columns": columns,
                "table_scores": table_scores}

    # ---------------------------------------------------------- S3: planning.
    def _join_path(self, tables):
        if len(tables) <= 1:
            return []
        adj = {}
        for a, b, cond in FK_EDGES:
            cond = cond.format(c="c", o="o", oi="oi", p="p")
            adj.setdefault(a, []).append((b, cond))
            adj.setdefault(b, []).append((a, cond))
        seen = {tables[0]}
        edges = []
        while len(seen) < len(tables):
            progressed = False
            for t in list(seen):
                for nb, cond in adj.get(t, []):
                    if nb in tables and nb not in seen:
                        edges.append((t, nb, cond))
                        seen.add(nb)
                        progressed = True
            if not progressed:
                break
        return edges

    def s3_planning(self, s2, intent):
        tables = list(s2["tables"])
        hub_added = []
        if "order_items" in tables and "orders" not in tables:
            tables.append("orders")          # FK-hub completion
            hub_added.append("orders")
        pruned = []
        non_hub = [t for t in tables if t not in hub_added]
        if len(non_hub) > 2 and not self.fix_joins:
            # W2 (deliberate): prune the lowest-confidence table.
            drop = sorted(non_hub,
                          key=lambda t: (s2["table_scores"][t], t))[0]
            tables.remove(drop)
            pruned.append(drop)
        joins = self._join_path(tables)
        return {"tables": tables, "hub_added": hub_added, "pruned": pruned,
                "joins": joins, "intent": intent}

    # --------------------------------------------------- S4: SQL generation.
    def _from_with_joins(self, plan):
        tables = plan["tables"]
        first = tables[0]
        sql = "%s %s" % (first, ALIASES[first])
        seen = {first}
        for left, right, cond in plan["joins"]:
            nxt = right if left in seen else left
            sql += " JOIN %s %s ON %s" % (nxt, ALIASES[nxt], cond)
            seen.add(nxt)
        for t in tables:  # disconnected fallback (should not happen)
            if t not in seen:
                sql += ", %s %s" % (t, ALIASES[t])
                seen.add(t)
        return sql

    @staticmethod
    def _where_sql(intent, alias):
        filt = intent["filter"]
        if not filt:
            return ""
        col, val = filt
        return "%s.%s = '%s'" % (alias, col, val)

    def _build_revenue(self, intent, plan):
        gd = intent["group_dim"]
        if gd == "region" and not self.fix_typos:
            gcol = "c.regio"          # W4 (deliberate): typo -> S5 error
        else:
            gcol = {"category": "p.category", "region": "c.region"}.get(gd)
        sel = ("SUM(oi.quantity * oi.unit_price) AS revenue"
               if not gcol else "%s, SUM(oi.quantity * oi.unit_price) AS revenue" % gcol)
        sql = "SELECT %s FROM %s" % (sel, self._from_with_joins(plan))
        wheres = []
        if self.use_glossary_enforcement:      # W3 fixed by glossary layer
            wheres.append("o.status = 'completed'")
        if intent["time_window"]:
            s, e = intent["time_window"]
            wheres.append("o.order_date >= '%s' AND o.order_date < '%s'" % (s, e))
        if wheres:
            sql += " WHERE " + " AND ".join(wheres)
        if gcol:
            sql += " GROUP BY " + gcol
        return sql

    def _build_aggregation(self, intent, plan):
        concept = intent["concept"]
        if concept == "revenue":
            return self._build_revenue(intent, plan)
        if concept == "avg_order_value":
            # W5 (deliberate): per-item AVG instead of the governed fragment.
            if self.use_glossary_enforcement:
                frag = self.glossary["AVG_ORDER_VALUE"]["sql_fragment"]
                return ("SELECT %s AS aov FROM order_items oi "
                        "JOIN orders o ON oi.order_id = o.order_id "
                        "WHERE o.status = 'completed'" % frag)
            return ("SELECT AVG(oi.quantity * oi.unit_price) AS aov "
                    "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id")
        if concept == "repeat":
            # W5 (deliberate): naive guess; no governed definition in v1.0.
            if self.use_glossary_enforcement and "REPEAT_BUYER_RATE" in self.glossary:
                frag = self.glossary["REPEAT_BUYER_RATE"]["sql_fragment"]
                return "SELECT %s AS repeat_rate" % frag
            return "SELECT COUNT(*) FROM customers"
        # Unknown concept (e.g. 'turnover' pre-fix): naive single-table guess.
        return "SELECT SUM(quantity * unit_price) AS total FROM order_items"

    def _build_topn(self, intent, plan):
        n = intent["topn"] or 1
        if intent["group_dim"] == "customer_name" and "orders" in plan["tables"]:
            return ("SELECT c.name, COUNT(*) AS n FROM customers c "
                    "JOIN orders o ON c.customer_id = o.customer_id "
                    "GROUP BY c.name ORDER BY n DESC LIMIT %d" % n)
        return ("SELECT region, COUNT(*) AS n FROM customers "
                "GROUP BY region ORDER BY n DESC LIMIT %d" % n)

    def _build_order_grain(self, intent, plan):
        tables = plan["tables"]
        sel = ["o.order_id"]
        if "customers" in tables:
            sel.append("c.name")
        sel.append("SUM(oi.quantity * oi.unit_price) AS total")
        group = "o.order_id" + (", c.name" if "customers" in tables else "")
        return ("SELECT %s FROM %s WHERE o.status = 'completed' "
                "GROUP BY %s ORDER BY o.order_id LIMIT 20"
                % (", ".join(sel), self._from_with_joins(plan), group))

    def s4_sql_generation(self, question, intent, plan):
        qtype = intent["qtype"]
        if qtype == "count":
            t = intent["table_focus"] or plan["tables"][0]
            a = ALIASES[t]
            w = self._where_sql(intent, a)
            sql = "SELECT COUNT(*) FROM %s %s" % (t, a)
            return sql + (" WHERE " + w if w else "")
        if qtype == "list" and intent.get("grain") == "order":
            return self._build_order_grain(intent, plan)
        if qtype == "list":
            t = intent["table_focus"] or plan["tables"][0]
            a = ALIASES[t]
            cols = ["name"] if "name" in question.lower() else ["*"]
            w = self._where_sql(intent, a)
            sql = "SELECT %s FROM %s %s" % (", ".join(cols), t, a)
            return sql + (" WHERE " + w if w else "")
        if qtype == "topn":
            return self._build_topn(intent, plan)
        if qtype == "aggregation":
            return self._build_aggregation(intent, plan)
        return self._build_topn(intent, plan)

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
    def answer(self, question, db_path):
        intent = self.s1_intent(question)
        s2 = self.s2_schema_linking(question, intent)
        plan = self.s3_planning(s2, intent)
        sql = self.s4_sql_generation(question, intent, plan)
        s5 = self.s5_execution(sql, db_path)
        s6 = self.s6_interpretation(question, intent, s5)
        return {"s1": intent, "s2": s2, "s3": plan, "s4": {"sql": sql},
                "s5": s5, "s6": s6, "agent_version": self.version}
