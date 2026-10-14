# Author: Prabhu Rangaraju
"""Held-out repair evaluation -- STEP 1: deterministic dev/held-out splits.

Split rule (seed 42), recorded BEFORE any diagnosis:
  Spider (n=88): stratified by DB. random.Random(42); shuffle each DB's ID
      list; dev = first 22 of each shuffled list (CS: 22 dev / 21 held,
      CA: 22 dev / 23 held -> 44/44).
  BIRD (n=100): stratified by DB. random.Random(42); shuffle each DB's ID
      list; dev = first 15 california_schools + first 16 financial +
      first 19 thrombosis_prediction -> 50/50.

Outputs: spider_split.json, bird_split.json in this directory.
Run with the spider venv python (needs pandas for dataset building):
    cd <spider_dir> && venv/bin/python <heldout>/split.py
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "spider"))
BIRD_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "bird"))

sys.path.insert(0, SPIDER_DIR)
os.chdir(SPIDER_DIR)  # dataset.py uses relative data paths

import dataset as spider_dataset
from goldparse import parse_gold
from schema import SpiderSchema


def stratified_split(id_lists, dev_counts, seed=42):
    """id_lists: dict group -> [ids]; dev_counts: group -> n_dev."""
    rng = random.Random(seed)
    dev, held = {}, {}
    for group, ids in id_lists.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_dev = dev_counts[group]
        dev[group] = ids[:n_dev]
        held[group] = ids[n_dev:]
    return dev, held


def spider_ids():
    schemas = {
        "concert_singer": SpiderSchema(
            "concert_singer",
            os.path.join(SPIDER_DIR, "data", "db_extract", "spider_data",
                         "database", "concert_singer", "concert_singer.sqlite")),
        "car_1": SpiderSchema(
            "car_1",
            os.path.join(SPIDER_DIR, "data", "db_extract", "spider_data",
                         "database", "car_1", "car_1.sqlite")),
    }
    data, exclusions = spider_dataset.build_dataset(schemas, parse_gold)
    print("spider: n=%d, exclusions=%d" % (len(data), len(exclusions)))
    assert len(data) == 88, "expected n=88"
    assert sorted(e["id"] for e in exclusions) == ["CS-14", "CS-15"]
    by_db = {"concert_singer": [], "car_1": []}
    for e in data:
        by_db[e["db_id"]].append(e["id"])
    return by_db


def bird_ids():
    sample = json.load(open(os.path.join(BIRD_DIR, "sample.json")))
    by_db = {}
    for row in sample:
        by_db.setdefault(row["db_id"], []).append(
            "Q%d" % row["question_id"])  # canonical paper qid notation
    print("bird: n=%d" % len(sample))
    assert len(sample) == 100
    return by_db


def main():
    # --- Spider ---
    s_ids = spider_ids()
    dev, held = stratified_split(
        s_ids, {"concert_singer": 22, "car_1": 22}, seed=42)
    spider_split = {
        "seed": 42,
        "rule": ("stratified by db_id: random.Random(42); shuffle each DB's "
                 "sorted ID list; dev = first 22 of each shuffled list "
                 "(CS 22/21, CA 22/23) -> 44 dev / 44 held-out"),
        "dev": sorted(sum(dev.values(), [])),
        "heldout": sorted(sum(held.values(), [])),
        "dev_by_db": {k: sorted(v) for k, v in dev.items()},
        "heldout_by_db": {k: sorted(v) for k, v in held.items()},
    }
    assert len(spider_split["dev"]) == 44 and len(spider_split["heldout"]) == 44
    with open(os.path.join(HERE, "spider_split.json"), "w") as fh:
        json.dump(spider_split, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("spider dev/heldout: %d/%d" % (len(spider_split["dev"]),
                                         len(spider_split["heldout"])))

    # --- BIRD ---
    b_ids = bird_ids()
    dev, held = stratified_split(
        b_ids,
        {"california_schools": 15, "financial": 16,
         "thrombosis_prediction": 19},
        seed=42)
    bird_split = {
        "seed": 42,
        "rule": ("stratified by db_id: random.Random(42); shuffle each DB's "
                 "sorted ID list; dev = first 15 california_schools + first "
                 "16 financial + first 19 thrombosis_prediction -> 50 dev / "
                 "50 held-out"),
        "dev": sorted(sum(dev.values(), [])),
        "heldout": sorted(sum(held.values(), [])),
        "dev_by_db": {k: sorted(v) for k, v in dev.items()},
        "heldout_by_db": {k: sorted(v) for k, v in held.items()},
    }
    assert len(bird_split["dev"]) == 50 and len(bird_split["heldout"]) == 50
    with open(os.path.join(HERE, "bird_split.json"), "w") as fh:
        json.dump(bird_split, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("bird dev/heldout: %d/%d" % (len(bird_split["dev"]),
                                       len(bird_split["heldout"])))


if __name__ == "__main__":
    main()
