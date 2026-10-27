import sqlite3
import pytest
from src import database as db
from src.blockchain import Ledger


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    db.init_db(c)
    yield c
    c.close()


@pytest.fixture
def ledger(conn):
    db.create_batch(conn, "B1", "Widget", "Electronics", 100, "SUP-1", "Shenzhen")
    led = Ledger(conn)
    for t in ["MANUFACTURED", "DISPATCHED", "RECEIVED", "INSPECTED", "DELIVERED"]:
        led.add_event("B1", t, "P-1", "Somewhere", 100)
    led.save_checkpoint("test")
    return led
