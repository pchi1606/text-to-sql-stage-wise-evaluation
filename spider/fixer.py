# Author: Prabhu Rangaraju
"""Fix proposal and application for the Spider arm (one fix per layer).

Layers (same as the prototype):
  metadata -- agent.synonyms / stemming / S2-S3 table-selection knowledge
  glossary -- NOT APPLICABLE (no governed domain glossary exists for Spider)
  template -- S3/S4 generation templates (not triggered by this run)

propose_fixes(diagnoses) maps the DOMINANT observed failure mode to the layer
that owns the root cause. apply_fix(agent, fix) mutates the agent and bumps
its version string.
"""

from collections import Counter

# Synonym mappings derived from the observed SCHEMA_LINK_MISS gap tokens:
# question words with no stemmed match on any table-name token.
FIX_SYNONYMS = {
    "stations": "stadium",        # paraphrase of 'stadiums'
    "nation": "country",          # 'name and nation of the singer'
    "manufacturers": "maker",     # car_1 Maker column
    "cards": "car",               # typo-variant of 'cars'
}


def propose_fixes(diagnoses):
    """diagnoses: dict qid -> (mode, stage, evidence). Returns [fix dict]."""
    modes = Counter(mode for mode, _stage, _ev in diagnoses.values())
    total_fail = sum(c for m, c in modes.items() if m != "no failure")
    fixes = []

    dominant, count = modes.most_common(1)[0]
    if dominant == "SCHEMA_LINK_MISS" and count >= 5:
        def _apply_metadata(ag):
            ag.stem_plurals = True      # stem for TABLE-name matching only
            ag.prune_subsumed = True    # drop tables contributing no new token
            ag.drop_columnless = True   # drop linked tables w/o column evidence
            ag.synonyms.update(FIX_SYNONYMS)
        fixes.append({
            "layer": "metadata",
            "title": "Plural stemming + synonym map + table-selection guards",
            "description": (
                "S2 table matching had no plural handling, so 'singers', "
                "'concerts', 'stadiums', 'models', 'countries' matched no "
                "table. Enable naive plural stemming for table-name matching "
                "(column matching stays exact), register observed non-plural "
                "gaps as synonyms (stations->stadium, nation->country, "
                "manufacturers->maker, cards->car), prune linked tables that "
                "contribute no new matching token (junction-table false "
                "positives), and drop S2-linked tables with no linked columns "
                "that no intent signal requires (ambiguous-stem false "
                "positives)."
            ),
            "targets": "SCHEMA_LINK_MISS (S2)",
            "version": "1.1",
            "apply": _apply_metadata,
        })

    # No glossary-layer fix: SEMANTIC_DEFINITION_VIOLATION is not applicable
    # (no governed domain glossary for Spider). No template-layer fix was
    # triggered: JOIN_ERROR/AGG_FILTER_ERROR/EXECUTION_ERROR were not dominant.
    return fixes


def apply_fix(agent, fix):
    fix["apply"](agent)
    agent.version = fix["version"]
    return agent
