# Author: Prabhu Rangaraju
"""Evaluation dataset: sampled real Spider dev questions on 2 databases.

Sampling rule (deterministic): first 45 questions of concert_singer (all of
them, parquet rows 0-44) plus first 45 questions of car_1 (parquet rows
87-131), in file order. random.seed(42) is set as the canonical determinism
seed; the slice itself is order-based so no RNG is consumed.

Exclusions (checked live in build_dataset): gold SQL must execute cleanly and
return non-empty rows, else the question is excluded with a reason.
"""

import random
import sqlite3

import pandas as pd

PARQUET = "data/validation-00000-of-00001.parquet"
DB_DIR = "data/db_extract/spider_data/database"

SAMPLE_SPEC = [("concert_singer", 45), ("car_1", 45)]

# Qid prefix per db.
PREFIX = {"concert_singer": "CS", "car_1": "CA"}


def check_gold(db_path, sql):
    """Returns (ok, rows_or_error). ok requires clean exec + non-empty."""
    try:
        con = sqlite3.connect(db_path)
        try:
            rows = con.execute(sql).fetchall()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001
        return False, "EXEC_ERROR: %s: %s" % (type(e).__name__, e)
    if not rows:
        return False, "EMPTY_RESULT: gold SQL returns 0 rows on the real DB"
    return True, rows


def build_dataset(schemas, gold_parser):
    random.seed(42)
    df = pd.read_parquet(PARQUET)
    dataset, exclusions = [], []
    for db_id, n in SAMPLE_SPEC:
        sub = df[df["db_id"] == db_id].head(n)
        schema = schemas[db_id]
        for pos, (_idx, row) in enumerate(sub.iterrows()):
            qid = "%s-%02d" % (PREFIX[db_id], pos)
            ok, info = check_gold(schema.db_path, row["query"])
            if not ok:
                exclusions.append({
                    "id": qid, "db_id": db_id,
                    "question": row["question"], "gold_sql": row["query"],
                    "reason": info,
                })
                continue
            gt, gc = gold_parser(row["query"], schema)
            dataset.append({
                "id": qid,
                "db_id": db_id,
                "question": row["question"],
                "gold_sql": row["query"],
                "gold_tables": gt,
                "gold_columns": gc,
            })
    return dataset, exclusions
