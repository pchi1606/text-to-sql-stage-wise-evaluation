# Author: Prabhu Rangaraju
"""Build the deterministic retail.db used by the SQL-eval-framework prototype.

Schema:
  customers(customer_id, name, region, signup_date)
  orders(order_id, customer_id, order_date, status)          -- status in {completed, cancelled}
  order_items(item_id, order_id, product_id, quantity, unit_price)
  products(product_id, name, category)

40 customers, 20 products, ~200 orders, 1-4 items per order. ~12% of orders
are cancelled -- this matters for the governed REVENUE definition, which must
exclude cancelled orders. Product 20 is never ordered (reserved for anti-join
style questions). Deterministic: random.seed(42); rebuilding is idempotent.
"""

import os
import random
import sqlite3
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retail.db")

REGIONS = ["North", "South", "East", "West"]

CATEGORIES = {
    "Electronics": ["Laptop Pro", "Phone X", "Tablet S", "Headphones", "Smartwatch"],
    "Furniture": ["Desk", "Chair", "Sofa", "Bookshelf", "Lamp"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket", "Sneakers", "Hat"],
    "Groceries": ["Coffee", "Tea", "Rice", "Pasta", "Olive Oil"],
}


def build_db(path=DB_PATH):
    random.seed(42)
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(
        """
        CREATE TABLE customers(
            customer_id INTEGER PRIMARY KEY, name TEXT, region TEXT, signup_date TEXT);
        CREATE TABLE products(
            product_id INTEGER PRIMARY KEY, name TEXT, category TEXT);
        CREATE TABLE orders(
            order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
        CREATE TABLE order_items(
            item_id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER,
            quantity INTEGER, unit_price REAL);
        """
    )

    customers = [
        (i, "Customer %02d" % i, REGIONS[(i - 1) % 4], "2024-%02d-15" % (((i - 1) % 12) + 1))
        for i in range(1, 41)
    ]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

    products, prices, pid = [], {}, 1
    for cat, names in CATEGORIES.items():
        for nm in names:
            price = round(random.uniform(5, 1200), 2)
            products.append((pid, nm, cat))
            prices[pid] = price
            pid += 1
    cur.executemany("INSERT INTO products VALUES (?,?,?)", products)

    base = date(2025, 1, 1)
    orders = []
    for oid in range(1, 201):
        cid = random.randint(1, 40)
        d = base + timedelta(days=random.randint(0, 263))
        status = "cancelled" if random.random() < 0.12 else "completed"
        orders.append((oid, cid, d.isoformat(), status))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)

    items, iid = [], 1
    for (oid, _cid, _d, _s) in orders:
        for _ in range(random.randint(1, 4)):
            prod = random.randint(1, 19)  # product 20 is never ordered
            qty = random.randint(1, 5)
            items.append((iid, oid, prod, qty, prices[prod]))
            iid += 1
    cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print(build_db())
