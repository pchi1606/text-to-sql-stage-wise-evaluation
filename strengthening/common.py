# Author: Prabhu Rangaraju
"""Shared helpers for the strengthening experiments (pure stdlib).

No agent/diagnoser imports here: experiment scripts set sys.path for the
Spider or the synthetic framework and import those modules separately, so
this module must stay import-context free.
"""

import json
import math
import os
import random
import shutil
import sqlite3
import time

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
# Package-relative paths (patched for the public reproduction package;
# the original used absolute ~/workspace paths).
PKG_ROOT = os.path.dirname(STRENGTH_DIR)
SPIDER_DIR = os.path.join(PKG_ROOT, "spider")
SYNTH_DIR = os.path.join(PKG_ROOT, "synth")
RAW_DIR = os.path.join(STRENGTH_DIR, "raw")
MUTANT_DIR = os.path.join(STRENGTH_DIR, "mutants")


def write_raw(name, payload):
    path = os.path.join(RAW_DIR, name)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


# ------------------------------------------------------------------ stats.
def bootstrap_ci(xs, n_resamples=10000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for the mean of a 0/1 vector."""
    xs = list(xs)
    n = len(xs)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        s = sum(xs[rng.randrange(n)] for _ in range(n)) / n
        stats.append(s)
    stats.sort()
    lo = stats[int((alpha / 2) * n_resamples)]
    hi = stats[int((1 - alpha / 2) * n_resamples) - 1]
    return lo, hi


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def dist_summary(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return {"n": 0}
    def q(p):
        return xs[min(n - 1, int(p * n))]
    return {
        "n": n,
        "min": xs[0],
        "median": (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2),
        "p90": q(0.90),
        "max": xs[-1],
        "mean": sum(xs) / n,
    }


# ------------------------------------------------------------------ EX.
def _norm_row(row):
    def norm(v):
        if isinstance(v, float):
            return round(v, 4)
        return v
    return tuple(norm(v) for v in row)


def _safe_sort_key(row):
    key = []
    for v in row:
        if v is None:
            key.append(("None", ""))
        elif isinstance(v, float):
            key.append(("float", round(v, 4)))
        elif isinstance(v, bool):
            key.append(("bool", int(v)))
        elif isinstance(v, int):
            key.append(("int", v))
        else:
            key.append(("other", str(v)))
    return key


def _sorted_rows(rows):
    normed = [_norm_row(r) for r in rows]
    try:
        return sorted(normed)  # identical to the framework's EX ordering
    except TypeError:
        # Mutants can surface NULLs the original data never had (mixed
        # None/int in one column); fall back to a deterministic type-tagged
        # order instead of crashing. Only reached where the original would
        # have raised.
        return sorted(normed, key=_safe_sort_key)


def multiset_equal(a_rows, g_rows):
    a = _sorted_rows(a_rows)
    g = _sorted_rows(g_rows)
    if not g:
        return 0
    return 1 if a == g else 0


def gold_rows(db_path, sql):
    con = sqlite3.connect(db_path)
    try:
        return [tuple(r) for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def ex_of_sql(db_path, sql):
    """EX of an arbitrary SQL string against gold handled by caller."""
    raise NotImplementedError  # placeholder guard; not used


def agent_ex(agent, question, db_id, db_path, gold_sql):
    """Run agent.answer and score EX vs gold (multiset rule)."""
    trace = agent.answer(question, db_id)
    grows = gold_rows(db_path, gold_sql)
    arows = trace["s5"]["rows"] if trace["s5"]["ok"] else []
    return multiset_equal(arows, grows), trace


# ------------------------------------------------------------------ timing.
def median_ms(sql, db_path, runs=5):
    """Median wall-clock ms of executing sql (fresh connection per run)."""
    ts = []
    for _ in range(runs):
        con = sqlite3.connect(db_path)
        try:
            t0 = time.perf_counter()
            con.execute(sql).fetchall()
            t1 = time.perf_counter()
        finally:
            con.close()
        ts.append((t1 - t0) * 1000.0)
    ts.sort()
    return ts[runs // 2]


# ------------------------------------------------------------------ mutants.
def _tables(con):
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def _pk_int_col(con, table):
    for r in con.execute('PRAGMA table_info("%s")' % table):
        # R: cid, name, type, notnull, dflt, pk.
        if r[5] == 1 and "INT" in (r[2] or "").upper():
            return r[1]
    return None


def _col_names(con, table):
    return [r[1] for r in con.execute('PRAGMA table_info("%s")' % table)]


def build_subsample_mutant(src_path, dst_path, frac=0.70, seed=42):
    """Copy src DB; per table keep a random `frac` of rows (seed 42).

    Returns the per-table kept counts for documentation.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.copy(src_path, dst_path)
    rng = random.Random(seed)
    kept = {}
    con = sqlite3.connect(dst_path)
    try:
        for t in _tables(con):
            rowids = [r[0] for r in con.execute(
                'SELECT rowid FROM "%s"' % t)]
            k = int(round(len(rowids) * frac))
            keep = set(rng.sample(rowids, k)) if rowids else set()
            if len(keep) < len(rowids):
                con.execute(
                    'DELETE FROM "%s" WHERE rowid NOT IN (%s)' % (
                        t, ",".join("?" * len(keep)) or "NULL"),
                    tuple(keep))
            kept[t] = len(keep)
        con.commit()
    finally:
        con.close()
    return kept


def _unique_nonpk_cols(con, table):
    """Columns covered by a UNIQUE (non-PK) index: duplication must avoid
    violating them."""
    cols = []
    for r in con.execute('PRAGMA index_list("%s")' % table):
        # Seq, name, unique, origin, partial.
        if r[2] == 1 and r[3] == "u":
            for c in con.execute('PRAGMA index_info("%s")' % r[1]):
                cols.append(c[2])  # (seqno, cid, name)
    return cols


def build_duplicate_mutant(src_path, dst_path):
    """Copy src DB; double every row (INTEGER PKs get fresh values).

    Tables with a non-PK UNIQUE constraint cannot hold two identical rows:
    the duplicated copies get mangled values in the unique column(s)
    (TEXT -> value || '__DUP2', numbers -> value + 10**9, NULL stays NULL).
    This is a documented deviation from pure row-doubling; both agent and
    gold SQL run on the same mutant so the multiset comparison stays valid.

    Returns the per-table (before, after) counts plus any mangled columns.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.copy(src_path, dst_path)
    counts = {}
    con = sqlite3.connect(dst_path)
    try:
        for t in _tables(con):
            before = con.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
            pk = _pk_int_col(con, t)
            cols = _col_names(con, t)
            ins_cols = [c for c in cols if c != pk]
            uniq = _unique_nonpk_cols(con, t)
            select_exprs = []
            for c in ins_cols:
                if c in uniq:
                    ctype = [r[2] for r in con.execute(
                        'PRAGMA table_info("%s")' % t) if r[1] == c][0].upper()
                    if "CHAR" in ctype or "TEXT" in ctype:
                        select_exprs.append(
                            '"%s" || \'__DUP2\'' % c)
                    elif "INT" in ctype or "REAL" in ctype or "NUM" in ctype:
                        select_exprs.append(
                            'CASE WHEN "%s" IS NULL THEN NULL '
                            'ELSE "%s" + 1000000000 END' % (c, c))
                    else:
                        raise ValueError(
                            "cannot mangle unique column %s.%s of type %s"
                            % (t, c, ctype))
                else:
                    select_exprs.append('"%s"' % c)
            collist = ",".join('"%s"' % c for c in ins_cols)
            con.execute(
                'INSERT INTO "%s" (%s) SELECT %s FROM "%s"'
                % (t, collist, ",".join(select_exprs), t))
            after = con.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0]
            assert after == 2 * before, (t, before, after)
            counts[t] = {"before": before, "after": after,
                         "mangled_unique_cols": uniq}
        con.commit()
    finally:
        con.close()
    return counts
