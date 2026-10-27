"""SQLite schema and data-access helpers. All SQL uses parameterised queries."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg

SCHEMA = """
CREATE TABLE IF NOT EXISTS product_batches (
    batch_id        TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,
    quantity        INTEGER NOT NULL CHECK (quantity > 0),
    supplier_id     TEXT NOT NULL,
    origin_location TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id      TEXT PRIMARY KEY,
    batch_id         TEXT REFERENCES product_batches(batch_id),
    order_date       TEXT NOT NULL,
    shipping_mode    TEXT NOT NULL,
    scheduled_days   INTEGER NOT NULL CHECK (scheduled_days >= 0),
    region           TEXT NOT NULL,
    market           TEXT NOT NULL,
    category         TEXT NOT NULL,
    quantity         INTEGER NOT NULL CHECK (quantity >= 1),
    unit_price       REAL NOT NULL CHECK (unit_price >= 0),
    discount_rate    REAL NOT NULL CHECK (discount_rate BETWEEN 0 AND 1),
    customer_segment TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('Delivered','In Transit')),
    late_delivery    INTEGER CHECK (late_delivery IN (0,1)),
    actual_days      REAL,
    risk_probability REAL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_level       TEXT CHECK (risk_level IN ('Low','Medium','High')),
    anomaly_score    REAL,
    is_anomaly       INTEGER NOT NULL DEFAULT 0 CHECK (is_anomaly IN (0,1)),
    data_source      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS supply_chain_events (
    event_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id       TEXT NOT NULL REFERENCES product_batches(batch_id),
    event_type     TEXT NOT NULL CHECK (event_type IN ('MANUFACTURED','DISPATCHED','RECEIVED','INSPECTED','DELIVERED')),
    participant_id TEXT NOT NULL CHECK (length(trim(participant_id)) > 0),
    timestamp      TEXT NOT NULL,
    location       TEXT NOT NULL CHECK (length(trim(location)) > 0),
    quantity       INTEGER NOT NULL CHECK (quantity >= 0)
);
CREATE TABLE IF NOT EXISTS ledger_blocks (
    block_id     INTEGER PRIMARY KEY,
    event_id     INTEGER REFERENCES supply_chain_events(event_id),
    payload_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL UNIQUE,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_count   INTEGER NOT NULL,
    last_block_id INTEGER NOT NULL,
    last_hash     TEXT NOT NULL,
    note          TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_history (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id      TEXT NOT NULL,
    source           TEXT NOT NULL CHECK (source IN ('manual','batch_upload')),
    input_json       TEXT NOT NULL,
    risk_probability REAL NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_level       TEXT NOT NULL,
    anomaly_score    REAL,
    is_anomaly       INTEGER NOT NULL DEFAULT 0,
    model_version    TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id      TEXT NOT NULL UNIQUE REFERENCES shipments(shipment_id),
    status           TEXT NOT NULL CHECK (status IN ('Open','In Review','Escalated','Closed - Confirmed Issue','Closed - False Alarm')),
    priority         TEXT NOT NULL CHECK (priority IN ('Low','Medium','High')),
    notes            TEXT NOT NULL DEFAULT '',
    outcome          TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_batch ON supply_chain_events(batch_id);
CREATE INDEX IF NOT EXISTS idx_ship_risk ON shipments(risk_level, status);
"""

TABLES = ["investigations", "prediction_history", "ledger_checkpoints", "ledger_blocks",
          "supply_chain_events", "shipments", "product_batches"]  # child tables first


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_conn(path=None) -> sqlite3.Connection:
    path = path or cfg.DB_PATH
    cfg.DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn, reset: bool = False):
    if reset:
        for t in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {t}")  # table names are constants, not user input
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def session():
    conn = get_conn()
    try:
        init_db(conn)
        yield conn
    finally:
        conn.close()


def db_ready() -> bool:
    if not cfg.DB_PATH.exists():
        return False
    with session() as conn:
        return conn.execute("SELECT COUNT(*) FROM shipments").fetchone()[0] > 0


# ---- shipments ------------------------------------------------------------
_SHIP_COLS = ["shipment_id", "batch_id", "order_date", "shipping_mode", "scheduled_days", "region", "market",
              "category", "quantity", "unit_price", "discount_rate", "customer_segment", "status", "late_delivery",
              "actual_days", "risk_probability", "risk_level", "anomaly_score", "is_anomaly", "data_source"]


def insert_shipments(conn, df: pd.DataFrame):
    rows = df[_SHIP_COLS].astype(object).where(df[_SHIP_COLS].notna(), None).values.tolist()
    q = f"INSERT OR REPLACE INTO shipments ({','.join(_SHIP_COLS)}) VALUES ({','.join('?' * len(_SHIP_COLS))})"
    with conn:
        conn.executemany(q, rows)


def fetch_shipments(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM shipments", conn)


def distinct_values(conn, column: str) -> list:
    if column not in cfg.CATEGORICAL:
        raise ValueError("unsupported column")
    return [r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM shipments ORDER BY 1")]


# ---- batches & events -----------------------------------------------------
def create_batch(conn, batch_id, product_name, category, quantity, supplier_id, origin_location):
    with conn:
        conn.execute("INSERT INTO product_batches VALUES (?,?,?,?,?,?,?)",
                     (batch_id.strip(), product_name.strip(), category.strip(), int(quantity),
                      supplier_id.strip(), origin_location.strip(), now_utc()))


def list_batches(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM product_batches ORDER BY created_at DESC, batch_id", conn)


def get_batch(conn, batch_id):
    return conn.execute("SELECT * FROM product_batches WHERE batch_id = ?", (batch_id,)).fetchone()


def events_for_batch(conn, batch_id) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT e.event_id, e.event_type, e.participant_id, e.timestamp, e.location, e.quantity, b.block_id
           FROM supply_chain_events e LEFT JOIN ledger_blocks b ON b.event_id = e.event_id
           WHERE e.batch_id = ? ORDER BY e.timestamp, e.event_id""", conn, params=(batch_id,))


# ---- predictions ----------------------------------------------------------
def log_predictions(conn, df: pd.DataFrame, source: str):
    import json
    from .config import MODEL_VERSION, RAW_INPUTS
    ts = now_utc()
    rows = [(r["shipment_id"], source, json.dumps({c: r[c] for c in RAW_INPUTS}, default=str),
             float(r["risk_probability"]), r["risk_level"], float(r["anomaly_score"]), int(r["is_anomaly"]),
             MODEL_VERSION, ts) for _, r in df.iterrows()]
    with conn:
        conn.executemany("""INSERT INTO prediction_history (shipment_id, source, input_json, risk_probability,
                            risk_level, anomaly_score, is_anomaly, model_version, created_at)
                            VALUES (?,?,?,?,?,?,?,?,?)""", rows)


def recent_predictions(conn, limit=20) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT created_at, shipment_id, source, risk_probability, risk_level, anomaly_score, is_anomaly
           FROM prediction_history ORDER BY prediction_id DESC LIMIT ?""", conn, params=(limit,))


# ---- investigations -------------------------------------------------------
def upsert_investigation(conn, shipment_id, status, priority, notes, outcome):
    ts = now_utc()
    with conn:
        conn.execute(
            """INSERT INTO investigations (shipment_id, status, priority, notes, outcome, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(shipment_id) DO UPDATE SET status=excluded.status, priority=excluded.priority,
               notes=excluded.notes, outcome=excluded.outcome, updated_at=excluded.updated_at""",
            (shipment_id, status, priority, notes.strip(), outcome.strip(), ts, ts))


def list_investigations(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT i.investigation_id, i.shipment_id, i.status, i.priority, i.notes, i.outcome, i.updated_at,
                  s.risk_level, s.risk_probability, s.is_anomaly
           FROM investigations i JOIN shipments s USING (shipment_id) ORDER BY i.updated_at DESC""", conn)
