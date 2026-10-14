# Author: Prabhu Rangaraju
"""Governed semantic definitions -- the prototype's 'semantic layer'.

An enterprise SQL agent must answer against governed business definitions, not
raw columns. The glossary is versioned; the fixer can add definitions to it
(the glossary-layer fix). Each entry declares machine-checkable conformance
criteria used by metrics.semantic_conformance:
  required_predicates: substrings that must appear in generated SQL
  agg_shape:           substring the aggregation must contain
  sql_fragment:        optional governed SELECT expression the fixed agent uses
"""

GLOSSARY_VERSION = "1.0"

GLOSSARY = {
    "REVENUE": {
        "definition": (
            "SUM(quantity * unit_price) over order_items joined to orders "
            "with status = 'completed'"
        ),
        "required_predicates": ["status", "completed"],
        "agg_shape": "sum(",
    },
    "AVG_ORDER_VALUE": {
        "definition": (
            "SUM(quantity * unit_price) / COUNT(DISTINCT order_id) "
            "over completed orders"
        ),
        "required_predicates": ["status", "completed"],
        "agg_shape": "count(distinct",
        "sql_fragment": "SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id)",
    },
    "ACTIVE_CUSTOMER": {
        "definition": (
            "customer with >= 1 completed order in the last 90 days of "
            "max(order_date)"
        ),
        "required_predicates": ["status", "completed"],
    },
}

# Concept word detected by the agent -> governing glossary entry.
# 'repeat' intentionally has NO entry in v1.0: Q12 exercises the
# glossary-addition fix.
CONCEPT_TO_DEF = {
    "revenue": "REVENUE",
    "avg_order_value": "AVG_ORDER_VALUE",
    "repeat": "REPEAT_BUYER_RATE",
}

# Business concepts the agent's intent stage can detect (vs. unknown synonyms
# like 'turnover', which schema linking cannot resolve at all).
SEMANTIC_CONCEPTS = {"revenue", "avg_order_value", "repeat"}
