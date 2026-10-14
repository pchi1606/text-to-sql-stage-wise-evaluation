# Author: Prabhu Rangaraju
"""Schema extraction from real Spider sqlite files via PRAGMA.

No tables.json is needed: table names, column names and FK edges are derived
directly from the database files.
"""

import re
import sqlite3


def _name_tokens(name):
    # Split snake_case and camelCase: Song_release_year -> {song, release, year}
    # CountryName -> {country, name}
    parts = re.split(r"[_\s]+", name)
    toks = []
    for p in parts:
        toks += re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p)
    return [t.lower() for t in toks if t]


class SpiderSchema:
    def __init__(self, db_id, db_path):
        self.db_id = db_id
        self.db_path = db_path
        self.tables = {}        # table -> [columns in order]
        self.column_table = {}  # lower(column) -> [tables] owning it
        self.fk_edges = []      # (table1, col1, table2, col2), undirected BFS graph
        self._load()

    def _load(self):
        con = sqlite3.connect(self.db_path)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for t in tables:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
                self.tables[t] = cols
                for c in cols:
                    self.column_table.setdefault(c.lower(), []).append(t)
            for t in tables:
                for r in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                    # R: id, seq, ref_table, from_col, to_col, ...
                    ref, frm, to = r[2], r[3], r[4]
                    self.fk_edges.append((t, frm, ref, to))
        finally:
            con.close()

    def table_tokens(self, table):
        return _name_tokens(table)

    def column_tokens(self, column):
        return _name_tokens(column)

    def neighbors(self, table):
        """Undirected adjacency for join-path BFS: table -> [(table, cond)]."""
        out = []
        for t1, c1, t2, c2 in self.fk_edges:
            if t1 == table:
                out.append((t2, f'"{t1}"."{c1}" = "{t2}"."{c2}"'))
            elif t2 == table:
                out.append((t1, f'"{t1}"."{c1}" = "{t2}"."{c2}"'))
        return out

    def summary(self):
        return {t: list(c) for t, c in self.tables.items()}


def join_path(schema, tables):
    """BFS shortest-path join edges covering `tables` (undirected FK graph).

    Returns (ordered_tables, join_conds). Tables unreachable from the seed are
    appended with a cross-join fallback (documented weakness)."""
    tables = list(dict.fromkeys(tables))
    if len(tables) <= 1:
        return tables, []
    seed = tables[0]
    ordered, conds = [seed], []
    seen = {seed}
    remaining = [t for t in tables[1:]]
    # Repeatedly attach the closest remaining table to the covered set.
    while remaining:
        best = None  # (dist, target, path_edges)
        for target in remaining:
            # BFS from covered set to target.
            dist, path = _bfs(schema, seen, target)
            if dist is not None and (best is None or dist < best[0]):
                best = (dist, target, path)
        if best is None:
            # Disconnected: cross-join fallback.
            t = remaining.pop(0)
            ordered.append(t)
            conds.append((None, t, "1=1"))
            seen.add(t)
            continue
        _dist, target, path = best
        for edge in path:
            nxt, cond = edge
            if nxt not in seen:
                ordered.append(nxt)
                conds.append((ordered[-2], nxt, cond))
                seen.add(nxt)
        remaining.remove(target)
    return ordered, conds


def _bfs(schema, sources, target):
    from collections import deque
    if target in sources:
        return 0, []
    prev = {s: None for s in sources}
    dq = deque(sources)
    while dq:
        cur = dq.popleft()
        for nb, cond in schema.neighbors(cur):
            if nb not in prev:
                prev[nb] = (cur, cond)
                dq.append(nb)
    if target not in prev:
        return None, None
    path = []
    cur = target
    while cur not in sources:
        p, cond = prev[cur]
        path.append((cur, cond))
        cur = p
    path.reverse()
    return len(path), path
