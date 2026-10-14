# Author: Prabhu Rangaraju
"""Fix proposal and application for the BIRD arm, one fix per layer.

Layers (mirroring the prototype):
  metadata -- agent.synonyms (business vocabulary / schema-linking knowledge)
  glossary -- governed semantic definitions + S4 enforcement of them
  template -- S4 generation templates (here: evidence predicate parsing)

propose_fixes(diagnoses) maps observed failure modes to the layer that owns
the root cause and returns at most one fix per layer. Fixes are authored from
the observed baseline failures (see BIRD_RESULTS.md); apply_fix mutates the
agent and bumps its version string.
"""

from bird_glossary import CATALOG, GLOSSARY_VERSION


def propose_fixes(diagnoses):
    """diagnoses: dict qid -> (mode, stage, evidence). Returns [fix dict]."""
    modes = {mode for (mode, _stage, _ev) in diagnoses.values()}
    fixes = []

    if "SCHEMA_LINK_MISS" in modes:
        # Gap tokens observed across S2 failures: enrollment paraphrases
        # ('pupils'/'enrollees'), clinical paraphrases ('diagnosed',
        # 'examined'), and entity paraphrases ('customers'->client,
        # 'women'->female). Map each to a token the schema descriptions
        # or glossary triggers actually contain.
        def _apply_metadata(ag):
            ag.synonyms.update({
                "pupils": "enrollment",
                "enrollees": "enrollment",
                "enrolees": "enrollment",
                "testtakers": "takers",
                "diagnosed": "diagnosis",
                "examined": "examination",
                "customers": "client",
                "women": "female",
                "woman": "female",
            })
        fixes.append({
            "layer": "metadata",
            "title": "Add enrollment/clinical/entity synonyms",
            "description": (
                "S2 schema linking had no lexical mapping for paraphrases "
                "observed as gap tokens in SCHEMA_LINK_MISS failures: "
                "'pupils'/'enrollees' (enrollment columns), 'diagnosed'/"
                "'examined' (Diagnosis/Examination), 'customers' (client "
                "table), 'women' (female concept trigger). Register them "
                "as synonyms in agent.synonyms."
            ),
            "targets": "SCHEMA_LINK_MISS (S2)",
            "version": "1.1",
            "apply": _apply_metadata,
        })

    if modes & {"SEMANTIC_DEFINITION_VIOLATION", "AGG_FILTER_ERROR"}:
        def _apply_glossary(ag):
            for concept in ("LOAN_PAID_OK", "LOAN_RUNNING_OK", "AVG_SALARY"):
                ag.glossary[concept] = CATALOG[concept]
            ag.glossary_version = "1.1"
            # S4 now enforces governed predicates: gender/admission/loan
            # status predicates are appended to WHERE, and AVG_SALARY pins
            # the measure column to district.A11.
            ag.use_glossary_enforcement = True
        fixes.append({
            "layer": "glossary",
            "title": "Add financial governed definitions + enforcement",
            "description": (
                "Add the missing LOAN_PAID_OK / LOAN_RUNNING_OK / AVG_SALARY "
                "definitions (glossary v1.0 -> v1.1) and enable glossary "
                "enforcement in S4: governed status/gender/admission "
                "predicates become mandatory WHERE clauses and AVG_SALARY "
                "pins the measure to district.A11."
            ),
            "targets": "SEMANTIC_DEFINITION_VIOLATION (S3), AGG_FILTER_ERROR (S4)",
            "version": "1.2",
            "apply": _apply_glossary,
        })

    if modes & {"WRONG_RESULT", "EMPTY_RESULT"}:
        # S6 result mismatches with no earlier-stage cause: the agent's
        # WHERE clause does not capture the question's semantic predicates.
        # 68/100 sample questions carry BIRD evidence definitions of the
        # form "X refers to <predicate>" which the baseline (W2/W3) never
        # parses. Enable the general evidence-predicate parser in S4.
        def _apply_evidence(ag):
            ag.use_evidence_predicates = True
        fixes.append({
            "layer": "template",
            "title": "Parse BIRD evidence definitions into S4 predicates",
            "description": (
                "Enable agent.use_evidence_predicates: S4 parses BIRD "
                "evidence clauses of the form 'X refers to <predicate>' "
                "(comparisons, IN-lists, IS NULL, BETWEEN, AND/OR "
                "combinations, 'A means B' value translations) and adds "
                "them as WHERE predicates when the defined phrase occurs "
                "in the question. Only exactly-parseable shapes are "
                "honored; formulas and prose are skipped."
            ),
            "targets": "WRONG_RESULT / EMPTY_RESULT (S6)",
            "version": "1.3",
            "apply": _apply_evidence,
        })

    if modes & {"JOIN_ERROR", "EXECUTION_ERROR"}:
        # Placeholder: only populated if the baseline actually shows
        # join/execution failures; kept for layer parity with the prototype.
        pass

    return fixes


def apply_fix(agent, fix):
    fix["apply"](agent)
    agent.version = fix["version"]
    return agent
