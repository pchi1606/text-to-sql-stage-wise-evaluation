# Author: Prabhu Rangaraju
"""Governed semantic definitions for the BIRD arm.

Every concept here is a genuine domain mapping taken from the benchmark's own
expert-annotated `evidence` field (the oracle knowledge BIRD ships per
question) -- e.g. "in-patient refers to Admission = '+'", "status = 'A'
means contract finished, no problems", "A11 pertains to average salary".
In an enterprise deployment these are exactly the glossary entries a data
governance team would own; here the benchmark authors already wrote them.

CATALOG holds every genuine concept (DB-scoped). GLOSSARY v1.0 ships only the
thrombosis_prediction family; the financial family (loan status codes,
average-salary column) is deliberately absent in v1.0 so the fix cycle has a
real glossary-layer gap to close -- mirroring the prototype, where
REPEAT_BUYER_RATE was added by the glossary fix.

Each entry declares machine-checkable conformance criteria consumed by
metrics.semantic_conformance: required_predicates are substrings that must
appear in generated SQL; agg_shape is an optional required substring.
"""

import re

GLOSSARY_VERSION = "1.0"

# Full catalog of genuine concepts. "dbs" scopes each concept to the database
# whose schema actually carries the governed column.
CATALOG = {
    "MALE_PATIENT": {
        "dbs": ["thrombosis_prediction"],
        "definition": "male patient: SEX = 'M'",
        "required_predicates": ["SEX", "'M'"],
    },
    "FEMALE_PATIENT": {
        "dbs": ["thrombosis_prediction"],
        "definition": "female patient: SEX = 'F'",
        "required_predicates": ["SEX", "'F'"],
    },
    "INPATIENT": {
        "dbs": ["thrombosis_prediction"],
        "definition": "in-patient admission: Admission = '+'",
        "required_predicates": ["Admission", "'+'"],
    },
    "OUTPATIENT": {
        "dbs": ["thrombosis_prediction"],
        "definition": "outpatient follow-up: Admission = '-'",
        "required_predicates": ["Admission", "'-'"],
    },
    "LOAN_PAID_OK": {
        "dbs": ["financial"],
        "definition": "loan fully paid with no issue: status = 'A' "
                      "(contract finished, no problems)",
        "required_predicates": ["status", "'A'"],
    },
    "LOAN_RUNNING_OK": {
        "dbs": ["financial"],
        "definition": "loan still running with no issue: status = 'C' "
                      "(running contract, ok so far)",
        "required_predicates": ["status", "'C'"],
    },
    "AVG_SALARY": {
        "dbs": ["financial"],
        "definition": "average salary is district.A11",
        "required_predicates": ["A11"],
    },
}

# V1.0 glossary: thrombosis family only (financial family added by fix).
GLOSSARY = {k: v for k, v in CATALOG.items()
            if k in ("MALE_PATIENT", "FEMALE_PATIENT", "INPATIENT",
                     "OUTPATIENT")}

# Question-token -> concept (checked against the question's db scope).
CONCEPT_TRIGGERS = {
    "thrombosis_prediction": [
        (("male", "males"), "MALE_PATIENT"),
        (("female", "females"), "FEMALE_PATIENT"),
        (("in-patient", "inpatient", "inpatients"), "INPATIENT"),
        (("outpatient", "outpatients", "out-patient"), "OUTPATIENT"),
    ],
    "financial": [
        (("fully paid", "no issue", "no issues", "no problems"), "LOAN_PAID_OK"),
        (("still running", "running contract", "running contracts"),
         "LOAN_RUNNING_OK"),
        (("average salary", "avg salary", "salary"), "AVG_SALARY"),
    ],
    "california_schools": [],
}


def detect_concepts(question, db_id):
    # Word-boundary matching: substring matching misfires ('male' inside
    # 'female' fired MALE_PATIENT on female-only questions, which the
    # glossary fix's S4 enforcement then turned into contradictory
    # SEX='F' AND SEX='M' predicates).
    q = question.lower()
    toks = set(re.findall(r"[a-z0-9]+", q))
    found = []
    for triggers, concept in CONCEPT_TRIGGERS.get(db_id, []):
        hit = False
        for t in triggers:
            if " " in t:
                if re.search(r"\b%s\b" % re.escape(t), q):
                    hit = True
                    break
            elif t in toks:
                hit = True
                break
        if hit:
            found.append(concept)
    return sorted(set(found))


def gold_definitions(gold_sql, db_id):
    """Concepts whose governed predicates all appear in the gold SQL."""
    sql = gold_sql.lower()
    defs = []
    for concept, spec in CATALOG.items():
        if db_id not in spec["dbs"]:
            continue
        if all(p.lower() in sql for p in spec["required_predicates"]):
            defs.append(concept)
    return sorted(defs)
