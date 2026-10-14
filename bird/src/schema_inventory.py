# Author: Prabhu Rangaraju
"""Schema inventory builder for the BIRD arm.

Per database: tables -> columns (name, declared type, pk flag), foreign keys
from PRAGMA foreign_key_list, plus a token index per table/column built from
column names (camel-case split) and the database_description CSVs
(column_name / column_description / value_description). Cached to
schema_inventory.json (deterministic: everything sorted).
"""

import csv
import json
import os
import re
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "minidev", "MINIDEV")
DB_DIR = os.path.join(DATA, "dev_databases")
OUT = os.path.join(ROOT, "schema_inventory.json")


def _tokens(text):
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return re.findall(r"[a-z0-9]+", text.lower())


# Formula-fragment noise from malformed description CSVs (e.g. frpm.csv
# embeds "commonsense evidence: eligible free rate = ..."): not schema
# vocabulary, so it must not enter the token index.
DESC_NOISE = {"commonsense", "evidence"}


def db_path(db_id):
    return os.path.join(DB_DIR, db_id, db_id + ".sqlite")


def build_inventory(db_ids):
    inv = {}
    for db_id in sorted(db_ids):
        con = sqlite3.connect(db_path(db_id))
        cur = con.cursor()
        tables = sorted(
            r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'").fetchall())
        desc = _load_descriptions(db_id)
        tinfo = {}
        for t in tables:
            cols = []
            pk = None
            for cid, name, ctype, _nn, _dv, ispk in cur.execute(
                    "PRAGMA table_info(\"%s\")" % t.replace('"', '""')):
                cols.append({"name": name, "type": (ctype or "").upper()})
                if ispk:
                    pk = name
            fks = []
            for (_id, _seq, ref_table, from_col, to_col, *_rest) in cur.execute(
                    "PRAGMA foreign_key_list(\"%s\")" % t.replace('"', '""')):
                fks.append({"from_col": from_col, "ref_table": ref_table,
                            "ref_col": to_col})
            col_index = {}
            for c in cols:
                toks = set(_tokens(c["name"]))
                d = desc.get((t, c["name"]), {})
                for field in ("column_name", "column_description",
                              "value_description"):
                    toks |= set(_tokens(d.get(field, "")))
                toks -= DESC_NOISE
                col_index[c["name"]] = sorted(toks)
            tinfo[t] = {
                "columns": cols,
                "pk": pk,
                "fks": sorted(fks, key=lambda f: (f["from_col"],
                                                  f["ref_table"])),
                "name_tokens": sorted(set(_tokens(t))),
                "column_index": col_index,
                "description": desc.get((t, None), {}).get("table", ""),
            }
        con.close()
        inv[db_id] = {"tables": tinfo}
    return inv


def _load_descriptions(db_id):
    """Map (table, column-or-None) -> dict of description fields."""
    out = {}
    ddir = os.path.join(DB_DIR, db_id, "database_description")
    if not os.path.isdir(ddir):
        return out
    for fname in sorted(os.listdir(ddir)):
        if not fname.endswith(".csv"):
            continue
        table = fname[:-4]
        with open(os.path.join(ddir, fname), newline="",
                  encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                orig = (row.get("original_column_name") or "").strip()
                if orig:
                    # Hygiene: some official CSVs (e.g. financial/account.csv)
                    # carry a trailing comma in the header, so part of the
                    # value_description lands under a phantom '' column.
                    vd = row.get("value_description", "") or ""
                    extra = row.get("", "") or ""
                    if extra and extra not in vd:
                        vd = (vd + "\n" + extra).strip()
                    out[(table, orig)] = {
                        "column_name": row.get("column_name", ""),
                        "column_description": row.get("column_description",
                                                      ""),
                        "value_description": vd,
                    }
    return out


def fk_graph(inv_db):
    """Undirected adjacency: table -> [(neighbor, cond_tmpl)] with real cols."""
    adj = {}
    for t, tinfo in inv_db["tables"].items():
        for fk in tinfo["fks"]:
            rt = fk["ref_table"]
            cond = '"%s"."%s" = "%s"."%s"' % (t, fk["from_col"],
                                              rt, fk["ref_col"])
            adj.setdefault(t, []).append((rt, cond))
            adj.setdefault(rt, []).append((t, cond))
    for t in adj:
        adj[t] = sorted(set(adj[t]))
    return adj


if __name__ == "__main__":
    inv = build_inventory(["california_schools", "financial",
                           "thrombosis_prediction"])
    with open(OUT, "w") as fh:
        json.dump(inv, fh, indent=1, sort_keys=True)
    print("wrote", OUT)
    for db_id, db in inv.items():
        print(db_id, "tables:", sorted(db["tables"]))
