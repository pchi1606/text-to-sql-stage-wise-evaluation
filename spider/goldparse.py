# Author: Prabhu Rangaraju
"""Heuristic derivation of gold tables/columns from Spider gold SQL.

Method (documented heuristic, not a full SQL parser):
  1. Mask string literals so quoted text can't match identifiers.
  2. Find FROM/JOIN <identifier> occurrences; resolve each to a schema table
     (case-insensitive). Capture the alias (AS t1 or bare t1) when present.
  3. Split each top-level UNION/INTERSECT/EXCEPT branch and union the results.
  4. For every other identifier in the query: if it is `prefix.col`, resolve
     `prefix` through the alias map (or a direct table name) to a table; if it
     is a bare name matching a schema column, attribute it to the first gold
     table (in FROM order) owning that column. Skip SQL keywords, function
     names, numbers, and aliases themselves.
  5. `t.*` expands to all columns of t; bare `*` expands to all columns of all
     gold tables; the `*` inside COUNT(*) is ignored (aggregate, no columns).

Limitations: no FROM-subquery handling; correlated subqueries are parsed for
their table/column references only at the outer level (their FROM/JOINs are
still picked up by step 2, which is what we want for tables).
"""

import re

KEYWORDS = {
    "select", "from", "where", "join", "on", "as", "and", "or", "not",
    "in", "is", "null", "group", "by", "order", "having", "limit",
    "distinct", "asc", "desc", "union", "intersect", "except", "all",
    "count", "sum", "avg", "min", "max", "like", "between", "exists",
    "case", "when", "then", "else", "end", "cast", "inner", "left",
    "right", "outer", "full", "cross", "natural",
}

SET_OPS = re.compile(r"\bUNION\b|\bINTERSECT\b|\bEXCEPT\b", re.I)


def _mask_literals(sql):
    return re.sub(r"'[^']*'", "''", sql)


def _split_setops(sql):
    parts, depth, start = [], 0, 0
    for m in SET_OPS.finditer(sql):
        # Only split at paren depth 0.
        seg = sql[start:m.start()]
        if seg.count("(") - seg.count(")") == 0:
            parts.append(seg)
            start = m.end()
    parts.append(sql[start:])
    return parts


def _canon_table(name, schema):
    for t in schema.tables:
        if t.lower() == name.lower():
            return t
    return None


def _canon_column(table, col, schema):
    for c in schema.tables[table]:
        if c.lower() == col.lower():
            return c
    return None


def parse_branch(sql, schema):
    """Parse one set-op branch. Returns (tables, alias_map)."""
    work = _mask_literals(sql)
    tables, amap = [], {}
    for m in re.finditer(
            r"\b(?:from|join)\b\s+"
            r"(?:\"([^\"]+)\"|`([^`]+)|([A-Za-z_][\w]*))"
            r"(?:\s+(?:as\s+)?([A-Za-z_][\w]*))?",
            work, re.I):
        raw = m.group(1) or m.group(2) or m.group(3)
        alias = m.group(4)
        if raw.lower() in KEYWORDS or raw == "(":
            continue
        t = _canon_table(raw, schema)
        if t is None:
            continue
        if t not in tables:
            tables.append(t)
        if alias and alias.lower() not in KEYWORDS:
            amap[alias.lower()] = t
    return tables, amap


def parse_gold(sql, schema):
    """Returns (gold_tables, gold_columns) as sorted canonical lists."""
    tables, columns = [], set()
    branches = _split_setops(_mask_literals(sql))
    for br in branches:
        btables, amap = parse_branch(br, schema)
        for t in btables:
            if t not in tables:
                tables.append(t)
        # Direct table-name prefixes also resolve.
        for t in btables:
            amap[t.lower()] = t

        # Expand t.*
        for m in re.finditer(
                r"([A-Za-z_][\w]*)\s*\.\s*\*", br):
            t = amap.get(m.group(1).lower())
            if t:
                for c in schema.tables[t]:
                    columns.add("%s.%s" % (t, c))
        # Bare * (but not inside count(*))
        br2 = re.sub(r"(?i)\bcount\s*\(\s*\*\s*\)", "count()", br)
        if re.search(r"(?<![\w.])\*(?![\w])", br2):
            for t in btables:
                for c in schema.tables[t]:
                    columns.add("%s.%s" % (t, c))

        # Prefixed columns: prefix.col.
        for m in re.finditer(
                r"([A-Za-z_][\w]*)\s*\.\s*"
                r"(?:\"([^\"]+)\"|`([^`]+)|([A-Za-z_][\w]*))", br):
            prefix, col = m.group(1), m.group(2) or m.group(3) or m.group(4)
            if col == "*":
                continue
            t = amap.get(prefix.lower())
            if t and t in schema.tables:
                cc = _canon_column(t, col, schema)
                if cc:
                    columns.add("%s.%s" % (t, cc))

        # Bare column names.
        for m in re.finditer(r"(?<![\w.\"`])([A-Za-z_][\w]*)", br):
            w = m.group(1)
            if w.lower() in KEYWORDS or w.lower() in amap:
                continue
            owners = schema.column_table.get(w.lower(), [])
            if not owners:
                continue
            # Attribute to first gold table (FROM order) owning it.
            for t in tables:
                if t in owners:
                    cc = _canon_column(t, w, schema)
                    columns.add("%s.%s" % (t, cc))
                    break
    return sorted(tables), sorted(columns)
