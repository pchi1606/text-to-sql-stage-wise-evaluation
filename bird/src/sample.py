# Author: Prabhu Rangaraju
"""Sampling for the BIRD arm (deterministic, seed 42).

RULE (recorded): take ALL questions from california_schools (30) and
financial (32); from thrombosis_prediction (50) take a seed-42 shuffle of
its question_id-sorted list and keep the first 38, for n=100 total.
Then verify every sampled gold SQL executes on its sqlite DB and returns
non-empty rows; exclusions listed with reasons.
"""
import json, random, sqlite3, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "minidev", "MINIDEV")
DB_DIR = os.path.join(DATA, "dev_databases")

def db_path(db_id):
    return os.path.join(DB_DIR, db_id, db_id + ".sqlite")

def main():
    d = json.load(open(os.path.join(DATA, "mini_dev_sqlite.json")))
    by_db = {}
    for row in d:
        by_db.setdefault(row["db_id"], []).append(row)
    cs = sorted(by_db["california_schools"], key=lambda r: r["question_id"])
    fin = sorted(by_db["financial"], key=lambda r: r["question_id"])
    tp = sorted(by_db["thrombosis_prediction"], key=lambda r: r["question_id"])
    rng = random.Random(42)
    rng.shuffle(tp)
    sample = cs + fin + tp[:38]
    assert len(sample) == 100, len(sample)
    print("sampled:", len(sample), "  cs:", len(cs), " fin:", len(fin), " tp:", len(tp[:38]))

    ok, excluded = [], []
    for row in sample:
        qid = row["question_id"]
        try:
            con = sqlite3.connect(db_path(row["db_id"]))
            cur = con.cursor()
            cur.execute(row["SQL"])
            rows = cur.fetchall()
            con.close()
        except Exception as e:
            excluded.append({"question_id": qid, "db_id": row["db_id"],
                             "reason": "gold SQL raised %s: %s" % (type(e).__name__, e),
                             "question": row["question"], "SQL": row["SQL"]})
            continue
        if not rows:
            excluded.append({"question_id": qid, "db_id": row["db_id"],
                             "reason": "gold SQL returned 0 rows",
                             "question": row["question"], "SQL": row["SQL"]})
            continue
        ok.append(row)

    json.dump(ok, open(os.path.join(ROOT, "sample.json"), "w"), indent=1)
    json.dump(excluded, open(os.path.join(ROOT, "exclusions.json"), "w"), indent=1)
    print("kept:", len(ok), " excluded:", len(excluded))
    for e in excluded:
        print("EXCL", e["question_id"], e["db_id"], "-", e["reason"][:100])

if __name__ == "__main__":
    main()
