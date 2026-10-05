"""SQLite storage layer. All SQL lives here, so swapping in PostgreSQL later means
changing this one module (e.g. psycopg + the same method names)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models.preprocessing import FEATURE_COLUMNS, ID_COLUMN, TARGET

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    gender TEXT, SeniorCitizen INTEGER, Partner TEXT, Dependents TEXT, tenure INTEGER,
    PhoneService TEXT, MultipleLines TEXT, InternetService TEXT, OnlineSecurity TEXT,
    OnlineBackup TEXT, DeviceProtection TEXT, TechSupport TEXT, StreamingTV TEXT,
    StreamingMovies TEXT, Contract TEXT, PaperlessBilling TEXT, PaymentMethod TEXT,
    MonthlyCharges REAL, TotalCharges REAL,
    churned            INTEGER NOT NULL,
    churn_probability  REAL
);
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    input_json       TEXT NOT NULL,
    prediction       TEXT NOT NULL,
    churn_probability REAL NOT NULL,
    risk_category    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""


class Database:
    def __init__(self, url: str):
        if not url.startswith("sqlite:///"):
            raise NotImplementedError(
                "Only SQLite is implemented. To use PostgreSQL, add a backend in this class.")
        self.path = url[len("sqlite:///"):]
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self):
        with self.connect() as c:
            c.executescript(SCHEMA)

    def get_meta(self, key: str):
        with self.connect() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str):
        with self.connect() as c:
            c.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))

    # ------------------------------------------------------------------ customers
    def customer_count(self) -> int:
        with self.connect() as c:
            return c.execute("SELECT COUNT(*) FROM customers").fetchone()[0]

    def replace_customers(self, df: pd.DataFrame, probabilities) -> int:
        """Replace all customer rows. `probabilities` is aligned with df rows."""
        rows = df[[ID_COLUMN] + FEATURE_COLUMNS + [TARGET]].copy()
        rows["churn_probability"] = list(probabilities)
        cols = ["customer_id"] + FEATURE_COLUMNS + ["churned", "churn_probability"]
        rows.columns = cols
        with self.connect() as c:
            c.execute("DELETE FROM customers")
            c.executemany(
                f"INSERT INTO customers ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                rows.astype(object).where(rows.notna(), None).values.tolist(),
            )
        return len(rows)

    def load_customers(self) -> pd.DataFrame:
        with self.connect() as c:
            return pd.read_sql_query("SELECT * FROM customers", c)

    # ---------------------------------------------------------------- predictions
    def save_prediction(self, features: dict, label: str, probability: float, risk: str) -> int:
        with self.connect() as c:
            cur = c.execute(
                "INSERT INTO predictions (created_at, input_json, prediction, churn_probability, risk_category)"
                " VALUES (?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 json.dumps(features), label, float(probability), risk),
            )
            return cur.lastrowid

    def recent_predictions(self, limit: int = 10) -> list[dict]:
        with self.connect() as c:
            rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["input"] = json.loads(d.pop("input_json"))
            out.append(d)
        return out

    def prediction_count(self) -> int:
        with self.connect() as c:
            return c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
