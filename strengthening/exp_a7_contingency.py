# Author: Prabhu Rangaraju
"""A7. FAILURE-CLUSTER x REPAIR-LAYER CONTINGENCY (preliminary).

Failure inventory (dataset, agent, stage, mode) from executed runs:
  synthetic v1.0 (7), Spider v1.0 (78), Spider Agent-B (83).
Repair attributions (layer, Qfix, Qbreak, which questions):
  synthetic per-layer ISOLATED runs (metadata 0, glossary 4, template 1;
    cumulative 7 -- two failures need multi-layer repairs),
  Spider v1.1 localized metadata fix (Qfix=5, all SCHEMA_LINK_MISS),
  Spider blind metadata repair (Qfix=1, Qbreak=6).
Contingency: failure-mode cluster -> repair layer that actually repaired.
BIRD data to be added later.
"""

import json
import os
import sys
from collections import Counter

STRENGTH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STRENGTH_DIR)
import common  # noqa: E402


def load(name):
    with open(os.path.join(common.RAW_DIR, name)) as fh:
        return json.load(fh)


def main():
    a1 = load("a1.json")
    a5 = load("a5.json")
    lay = load("a7_synth_layers.json")

    # ---- failure inventory ----
    clusters = []  # (dataset, agent, qid, stage, mode)
    for qid, f in lay["v10_failures"].items():
        clusters.append(("synthetic", "A-v1.0", qid, f["stage"], f["mode"]))
    for qid, r in a5["agent_a"]["per_question"].items():
        if r["stage"]:
            clusters.append(("spider", "A-v1.0", qid, r["stage"], r["mode"]))
    for qid, r in a5["agent_b"]["per_question"].items():
        if r["stage"]:
            clusters.append(("spider", "Agent-B", qid, r["stage"], r["mode"]))

    # ---- repair attributions ----
    # fix_id -> {layer, scope, fixed:{qid: mode}, Qfix, Qbreak}
    fixes = {}
    for layer, r in lay["per_layer_isolated"].items():
        fixes["synth/%s (isolated)" % layer] = {
            "layer": layer, "scope": "synthetic",
            "fixed": r["fixed_modes"], "Qfix": r["Qfix"],
            "Qbreak": r["Qbreak"]}
    fixes["spider/metadata-localized (v1.1)"] = {
        "layer": "metadata", "scope": "spider",
        "fixed": {qid: "SCHEMA_LINK_MISS"
                  for qid in a1["localized"]["fixed"]},
        "Qfix": a1["localized"]["Qfix"],
        "Qbreak": a1["localized"]["Qbreak"]}
    fixes["spider/metadata-blind"] = {
        "layer": "metadata", "scope": "spider",
        "fixed": {qid: "SCHEMA_LINK_MISS"
                  for qid in a1["blind"]["fixed"]},
        "Qfix": a1["blind"]["Qfix"],
        "Qbreak": a1["blind"]["Qbreak"]}

    # Map qid -> cluster key for repair lookup.
    qid_cluster = {(d, a, q): (s, m) for d, a, q, s, m in clusters}

    # ---- contingency: cluster (dataset, agent, stage, mode) x fix ----
    cluster_keys = sorted(set((d, a, s, m) for d, a, _q, s, m in clusters))
    table = []
    for ck in cluster_keys:
        d, a, s, m = ck
        members = [q for (dd, aa, q, ss, mm) in clusters
                   if (dd, aa, ss, mm) == ck]
        row = {"cluster": "%s / %s / %s / %s" % ck, "n": len(members),
               "members": sorted(members), "repairs": {}}
        for fid, f in fixes.items():
            if f["scope"] == "synthetic" and d != "synthetic":
                continue
            if f["scope"] == "spider" and (d != "spider" or a != "A-v1.0"):
                continue  # spider fixes were derived/applied on A-v1.0 only
            rep = sum(1 for q in f["fixed"] if q in members)
            row["repairs"][fid] = "%d/%d" % (rep, len(members))
        table.append(row)

    out = {
        "n_failures": len(clusters),
        "n_failure_keys": len(cluster_keys),
        "fixes": {fid: {"layer": f["layer"], "Qfix": f["Qfix"],
                        "Qbreak": f["Qbreak"],
                        "fixed_qids": sorted(f["fixed"])}
                  for fid, f in fixes.items()},
        "contingency": table,
        "interaction_note": (
            "Synthetic isolated per-layer Qfix sums to 5 (0+4+1) but the "
            "cumulative all-layers Qfix is 7: Q09 (S2/SCHEMA_LINK_MISS) "
            "needs metadata+glossary and Q11 (S5/EXECUTION_ERROR) needs "
            "glossary+template. The diagnosed stage does not uniquely "
            "determine the sufficient repair layer."),
        "scope_note": ("Spider fixes were proposed from and applied to "
                       "Agent-A v1.0 only; Agent-B failures have no repair "
                       "attempt (no fix cycle run for B). BIRD data pending."),
    }
    path = common.write_raw("a7.json", out)
    print("wrote", path)
    print("failures inventoried:", len(clusters))
    for row in table:
        print("%-52s n=%3d  %s" % (
            row["cluster"], row["n"],
            " | ".join("%s: %s" % (k.split("/")[0], v)
                       for k, v in row["repairs"].items())))


if __name__ == "__main__":
    main()
