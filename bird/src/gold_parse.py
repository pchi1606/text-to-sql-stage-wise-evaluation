# Author: Prabhu Rangaraju
"""Derive gold tables/columns per question from gold SQL via regex.

HEURISTIC (documented): tables from FROM/JOIN clauses (backtick/quote aware);
columns matched against the schema inventory -- alias-qualified references
resolved through the SQL's own alias map (`FROM t AS a`), plus bare
single-token references (quoted identifier spans removed first so that e.g.
`District Code` does not count as a hit for a `District` column), plus exact
substring matches for multi-word column names. Subqueries are handled via the
alias map; `SELECT *` is not expanded. Hand-verified on 10 random sampled
questions (see BIRD_RESULTS.md).
"""

import re


def _strip_literals(sql):
    return re.sub(r"'[^']*'", "''", sql)


def _alias_map(sql):
    """Map alias-or-name (lowercased) -> table name from FROM/JOIN clauses."""
    amap = {}
    for m in re.finditer(
            r"\b(?:from|join)\s+(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]"
            r"|([A-Za-z_][\w$]*))(?:\s+(?:as\s+)?([A-Za-z_][\w$]*))?",
            sql, re.IGNORECASE):
        table = next(g for g in m.groups()[:4] if g)
        alias = m.group(5)
        amap[table.lower()] = table
        if alias and alias.lower() not in ("inner", "left", "right", "full",
                                           "cross", "where", "on", "join"):
            amap[alias.lower()] = table
    return amap


def derive_gold_tables(sql, inventory_db):
    amap = _alias_map(sql)
    tables = []
    for _alias, table in amap.items():
        for t in inventory_db["tables"]:
            if t.lower() == table.lower() and t not in tables:
                tables.append(t)
    return tables


def derive_gold_columns(sql, inventory_db, gold_tables):
    amap = _alias_map(sql)
    # Canonical table name (inventory casing) per alias.
    canon = {}
    for a, t in amap.items():
        for gt in gold_tables:
            if gt.lower() == t.lower():
                canon[a] = gt
    work = _strip_literals(sql)
    # Quoted identifier spans, for exact multi-word matching and for
    # exclusion from the bare single-token check.
    quoted = re.findall(r"`([^`]+)`|\"([^\"]+)\"", work)
    quoted = [q for pair in quoted for q in pair if q]
    bare_work = re.sub(r"`[^`]*`|\"[^\"]*\"", " ", work)
    found = []

    def add(t, c):
        key = "%s.%s" % (t, c)
        if key not in found:
            found.append(key)

    for t in gold_tables:
        for col in inventory_db["tables"][t]["column_index"]:
            hit = False
            if " " in col or "(" in col or "-" in col:
                # Multi-word / special-char column: exact substring.
                if re.search(re.escape(col), work, re.IGNORECASE):
                    hit = True
            else:
                # Alias-qualified: T.col / "T".col / `T`.col.
                for m in re.finditer(
                        r"([A-Za-z_][\w$]*|\"[^\"]+\"|`[^`]+`)\s*\.\s*"
                        r"([\"`]?%s[\"`]?)\b" % re.escape(col),
                        work, re.IGNORECASE):
                    qual = m.group(1).strip("\"`").lower()
                    if qual in canon and canon[qual] == t:
                        hit = True
                        break
                # Bare single-token reference (quoted spans removed)
                if not hit and re.search(
                        r"(?<![\w.])%s(?![\w])" % re.escape(col),
                        bare_work, re.IGNORECASE):
                    hit = True
            if hit:
                add(t, col)
    return sorted(found)
