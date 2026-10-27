import sqlite3
import pytest
from src import database as db


def test_unknown_batch_rejected_by_foreign_key(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO supply_chain_events (batch_id,event_type,participant_id,timestamp,location,quantity)"
                     " VALUES ('NOPE','RECEIVED','P','t','L',1)")


def test_duplicate_batch_rejected(conn):
    db.create_batch(conn, "B1", "x", "c", 1, "s", "o")
    with pytest.raises(sqlite3.IntegrityError):
        db.create_batch(conn, "B1", "x", "c", 1, "s", "o")


def test_bad_event_type_rejected(conn):
    db.create_batch(conn, "B1", "x", "c", 1, "s", "o")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO supply_chain_events (batch_id,event_type,participant_id,timestamp,location,quantity)"
                     " VALUES ('B1','HACKED','P','t','L',1)")


def test_sql_injection_string_is_just_data(conn):
    db.create_batch(conn, "B1'; DROP TABLE product_batches;--", "x", "c", 1, "s", "o")
    assert len(db.list_batches(conn)) == 1
