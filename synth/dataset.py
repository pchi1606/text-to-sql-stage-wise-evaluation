# Author: Prabhu Rangaraju
"""Evaluation dataset: 12 natural-language questions over retail.db.

Each entry:
  id               Q01..Q12
  question         natural language
  gold_sql         reference SQL -- must execute cleanly and return non-empty
  gold_tables      tables the correct query touches
  gold_columns     content columns (non-key) the correct query needs, as
                   'table.column' -- used for schema-linking precision/recall
  gold_definitions glossary keys the correct answer depends on (may be empty)
  stage_tags       pipeline stages the question stresses

Coverage: simple filter+aggregation (Q01, Q02, Q07), 3-table joins (Q06, Q11),
revenue-by-category (Q03), synonym trap 'turnover' (Q09), time window (Q05),
top-N (Q04, Q10), average order value (Q08), and Q12 whose correct answer needs
REPEAT_BUYER_RATE -- a definition absent from glossary v1.0, exercising the
glossary-addition fix.
"""

DATASET = [
    {
        "id": "Q01",
        "question": "How many customers are in the West region?",
        "gold_sql": "SELECT COUNT(*) FROM customers WHERE region = 'West'",
        "gold_tables": ["customers"],
        "gold_columns": ["customers.region"],
        "gold_definitions": [],
        "stage_tags": ["S1", "S2", "S4", "S5"],
    },
    {
        "id": "Q02",
        "question": "List the names of all products in the Electronics category.",
        "gold_sql": "SELECT name FROM products WHERE category = 'Electronics'",
        "gold_tables": ["products"],
        "gold_columns": ["products.name", "products.category"],
        "gold_definitions": [],
        "stage_tags": ["S1", "S2", "S4", "S5"],
    },
    {
        "id": "Q03",
        "question": "What is the total revenue by product category?",
        "gold_sql": (
            "SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "JOIN products p ON oi.product_id = p.product_id "
            "WHERE o.status = 'completed' GROUP BY p.category"
        ),
        "gold_tables": ["order_items", "orders", "products"],
        "gold_columns": [
            "products.category",
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
        ],
        "gold_definitions": ["REVENUE"],
        "stage_tags": ["S2", "S3", "S4"],
    },
    {
        "id": "Q04",
        "question": "Which 3 customers placed the most orders?",
        "gold_sql": (
            "SELECT c.name, COUNT(*) AS n_orders FROM customers c "
            "JOIN orders o ON c.customer_id = o.customer_id "
            "GROUP BY c.name ORDER BY n_orders DESC LIMIT 3"
        ),
        "gold_tables": ["customers", "orders"],
        "gold_columns": ["customers.name"],
        "gold_definitions": [],
        "stage_tags": ["S1", "S2", "S3", "S4"],
    },
    {
        "id": "Q05",
        "question": "What was the total revenue in March 2025?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.status = 'completed' "
            "AND o.order_date >= '2025-03-01' AND o.order_date < '2025-04-01'"
        ),
        "gold_tables": ["order_items", "orders"],
        "gold_columns": [
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
            "orders.order_date",
        ],
        "gold_definitions": ["REVENUE"],
        "stage_tags": ["S1", "S2", "S3", "S4"],
    },
    {
        "id": "Q06",
        "question": (
            "For each completed order, show the order id, customer name, "
            "and total amount."
        ),
        "gold_sql": (
            "SELECT o.order_id, c.name, SUM(oi.quantity * oi.unit_price) AS total "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "WHERE o.status = 'completed' "
            "GROUP BY o.order_id, c.name ORDER BY o.order_id LIMIT 20"
        ),
        "gold_tables": ["customers", "orders", "order_items"],
        "gold_columns": [
            "orders.order_id",
            "customers.name",
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
        ],
        "gold_definitions": [],
        "stage_tags": ["S2", "S3", "S4"],
    },
    {
        "id": "Q07",
        "question": "How many orders were cancelled?",
        "gold_sql": "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
        "gold_tables": ["orders"],
        "gold_columns": ["orders.status"],
        "gold_definitions": [],
        "stage_tags": ["S1", "S2", "S4", "S5"],
    },
    {
        "id": "Q08",
        "question": "What is the average order value for completed orders?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) / COUNT(DISTINCT o.order_id) AS aov "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.status = 'completed'"
        ),
        "gold_tables": ["order_items", "orders"],
        "gold_columns": [
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
        ],
        "gold_definitions": ["AVG_ORDER_VALUE"],
        "stage_tags": ["S1", "S2", "S4"],
    },
    {
        "id": "Q09",
        "question": "What is our total turnover?",
        "gold_sql": (
            "SELECT SUM(oi.quantity * oi.unit_price) AS turnover "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "WHERE o.status = 'completed'"
        ),
        "gold_tables": ["order_items", "orders"],
        "gold_columns": [
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
        ],
        "gold_definitions": ["REVENUE"],
        "stage_tags": ["S1", "S2"],
    },
    {
        "id": "Q10",
        "question": "Which region has the highest number of customers?",
        "gold_sql": (
            "SELECT region, COUNT(*) AS n FROM customers "
            "GROUP BY region ORDER BY n DESC LIMIT 1"
        ),
        "gold_tables": ["customers"],
        "gold_columns": ["customers.region"],
        "gold_definitions": [],
        "stage_tags": ["S1", "S2", "S4"],
    },
    {
        "id": "Q11",
        "question": "What is the total revenue by region?",
        "gold_sql": (
            "SELECT c.region, SUM(oi.quantity * oi.unit_price) AS revenue "
            "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE o.status = 'completed' GROUP BY c.region"
        ),
        "gold_tables": ["order_items", "orders", "customers"],
        "gold_columns": [
            "customers.region",
            "order_items.quantity",
            "order_items.unit_price",
            "orders.status",
        ],
        "gold_definitions": ["REVENUE"],
        "stage_tags": ["S4", "S5"],
    },
    {
        "id": "Q12",
        "question": "What fraction of customers are repeat buyers?",
        "gold_sql": (
            "SELECT CAST(SUM(CASE WHEN t.cnt > 1 THEN 1 ELSE 0 END) AS REAL) "
            "/ (SELECT COUNT(*) FROM customers) AS repeat_rate "
            "FROM (SELECT customer_id, COUNT(*) AS cnt FROM orders "
            "WHERE status = 'completed' GROUP BY customer_id) t"
        ),
        "gold_tables": ["orders", "customers"],
        "gold_columns": ["orders.status", "orders.customer_id"],
        "gold_definitions": [],  # REPEAT_BUYER_RATE is NOT in glossary v1.0
        "stage_tags": ["S1", "S3"],
    },
]
