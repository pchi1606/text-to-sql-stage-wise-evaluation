# Author: Prabhu Rangaraju
"""Fix proposal and application, one fix per layer.

Layers:
  metadata -- agent.synonyms (business vocabulary / schema-linking knowledge)
  glossary -- governed semantic definitions + S4 enforcement of them
  template -- S3/S4 generation templates (join planning, column references)

propose_fixes(diagnoses) maps observed failure modes to the layer that owns
the root cause and returns at most one fix per layer. apply_fix(agent, fix)
mutates the agent and bumps its version string.
"""

from glossary import GLOSSARY_VERSION


def propose_fixes(diagnoses):
    """diagnoses: dict qid -> (mode, stage, evidence). Returns [fix dict]."""
    modes = {mode for (mode, _stage, _ev) in diagnoses.values()}
    fixes = []

    if "SCHEMA_LINK_MISS" in modes:
        def _apply_metadata(ag):
            ag.synonyms.update({"turnover": "revenue", "sales": "revenue"})
        fixes.append({
            "layer": "metadata",
            "title": "Add business synonyms (turnover/sales -> revenue)",
            "description": (
                "S2 schema linking had no lexical mapping for the business "
                "synonyms 'turnover'/'sales', so the revenue concept was never "
                "detected. Register both as synonyms of 'revenue' in "
                "agent.synonyms."
            ),
            "targets": "SCHEMA_LINK_MISS (S2)",
            "version": "1.1",
            "apply": _apply_metadata,
        })

    if modes & {"SEMANTIC_DEFINITION_VIOLATION", "AGG_FILTER_ERROR"}:
        def _apply_glossary(ag):
            ag.glossary["REPEAT_BUYER_RATE"] = {
                "definition": "fraction of customers with > 1 completed order",
                "required_predicates": ["status", "completed"],
                "sql_fragment": (
                    "(SELECT CAST(SUM(CASE WHEN t.cnt > 1 THEN 1 ELSE 0 END) AS REAL) "
                    "/ (SELECT COUNT(*) FROM customers) FROM "
                    "(SELECT customer_id, COUNT(*) AS cnt FROM orders "
                    "WHERE status = 'completed' GROUP BY customer_id) t)"
                ),
            }
            ag.glossary_version = "1.1"
            # S4 now consults the glossary: enforce required predicates and
            # use governed SQL fragments instead of naive guesses.
            ag.use_glossary_enforcement = True
        fixes.append({
            "layer": "glossary",
            "title": "Glossary-driven generation",
            "description": (
                "Add the missing REPEAT_BUYER_RATE definition (glossary "
                "v1.0 -> v1.1) and enable glossary enforcement in S4: the "
                "REVENUE status='completed' predicate is now mandatory and "
                "AVG_ORDER_VALUE / REPEAT_BUYER_RATE use their governed SQL "
                "fragments instead of naive aggregations."
            ),
            "targets": "SEMANTIC_DEFINITION_VIOLATION (S3), AGG_FILTER_ERROR (S4)",
            "version": "1.1",
            "apply": _apply_glossary,
        })

    if modes & {"JOIN_ERROR", "EXECUTION_ERROR"}:
        def _apply_template(ag):
            ag.fix_joins = True   # S3 keeps every linked table
            ag.fix_typos = True   # S4 region template uses c.region
        fixes.append({
            "layer": "template",
            "title": "Repair S3/S4 templates",
            "description": (
                "S3 planner no longer prunes the lowest-confidence table on "
                "3-table plans (fixes JOIN_ERROR); S4 region-grouped revenue "
                "template references c.region instead of the non-existent "
                "c.regio (fixes EXECUTION_ERROR)."
            ),
            "targets": "JOIN_ERROR (S3), EXECUTION_ERROR (S5)",
            "version": "1.1",
            "apply": _apply_template,
        })

    return fixes


def apply_fix(agent, fix):
    fix["apply"](agent)
    agent.version = fix["version"]
    return agent
