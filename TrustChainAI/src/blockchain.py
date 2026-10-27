"""Tamper-evident, hash-linked supply-chain ledger stored in SQLite.

IMPORTANT - what this is and is not
  * It is a LOCAL hash-linked ledger (each block stores the SHA-256 of its content plus the
    previous block's hash). It is NOT a decentralised blockchain: one operator controls the
    database, so there is no consensus and no protection against a rewrite of the entire chain.
  * It shows that records were not changed AFTER being written (tamper-evidence). It does NOT
    prove that the original data was true.
  * Deleting the newest blocks cannot be detected from the chain alone; it is detectable only
    against a trusted checkpoint (block count + last hash) stored/exported elsewhere.
"""
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field

from . import config as cfg
from .database import now_utc

GENESIS_PREV_HASH = "0" * 64


def canonical_json(payload: dict) -> str:
    """Deterministic serialisation: sorted keys, no whitespace, ASCII only."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_hash(block_id: int, prev_hash: str, payload_json: str) -> str:
    return hashlib.sha256(f"{block_id}|{prev_hash}|{payload_json}".encode("utf-8")).hexdigest()


def event_payload(row) -> dict:
    return {"batch_id": row["batch_id"], "event_type": row["event_type"], "participant_id": row["participant_id"],
            "timestamp": row["timestamp"], "location": row["location"], "quantity": int(row["quantity"])}


@dataclass
class VerificationResult:
    valid: bool
    blocks_checked: int
    issues: list = field(default_factory=list)  # dicts: type, block_id, detail
    checkpoint_used: bool = False


class Ledger:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        self.ensure_genesis()

    # ---- writing ----------------------------------------------------------
    def ensure_genesis(self):
        if self.conn.execute("SELECT COUNT(*) FROM ledger_blocks").fetchone()[0] == 0:
            payload = canonical_json({"type": "GENESIS", "note": "TrustChain AI ledger start"})
            with self.conn:
                self.conn.execute("INSERT INTO ledger_blocks (block_id, event_id, payload_json, prev_hash, hash, created_at)"
                                  " VALUES (0, NULL, ?, ?, ?, ?)",
                                  (payload, GENESIS_PREV_HASH, compute_hash(0, GENESIS_PREV_HASH, payload), now_utc()))

    def last_block(self):
        return self.conn.execute("SELECT * FROM ledger_blocks ORDER BY block_id DESC LIMIT 1").fetchone()

    def add_event(self, batch_id, event_type, participant_id, location, quantity, timestamp=None) -> int:
        """Insert an event AND its ledger block in one transaction. Returns the new block_id."""
        if event_type not in cfg.EVENT_TYPES:
            raise ValueError(f"event_type must be one of {cfg.EVENT_TYPES}")
        if int(quantity) < 0:
            raise ValueError("quantity cannot be negative")
        if not str(participant_id).strip() or not str(location).strip():
            raise ValueError("participant_id and location are required")
        timestamp = timestamp or now_utc()
        with self.conn:  # atomic: both inserts succeed or neither does
            cur = self.conn.execute(
                "INSERT INTO supply_chain_events (batch_id, event_type, participant_id, timestamp, location, quantity)"
                " VALUES (?,?,?,?,?,?)",
                (batch_id, event_type, str(participant_id).strip(), timestamp, str(location).strip(), int(quantity)))
            event_id = cur.lastrowid
            ev = self.conn.execute("SELECT * FROM supply_chain_events WHERE event_id=?", (event_id,)).fetchone()
            payload_json = canonical_json(event_payload(ev))
            last = self.last_block()
            block_id, prev = last["block_id"] + 1, last["hash"]
            self.conn.execute(
                "INSERT INTO ledger_blocks (block_id, event_id, payload_json, prev_hash, hash, created_at) VALUES (?,?,?,?,?,?)",
                (block_id, event_id, payload_json, prev, compute_hash(block_id, prev, payload_json), now_utc()))
        return block_id

    # ---- checkpoints ------------------------------------------------------
    def save_checkpoint(self, note="manual checkpoint") -> dict:
        last = self.last_block()
        count = self.conn.execute("SELECT COUNT(*) FROM ledger_blocks").fetchone()[0]
        with self.conn:
            self.conn.execute("INSERT INTO ledger_checkpoints (block_count, last_block_id, last_hash, note, created_at)"
                              " VALUES (?,?,?,?,?)", (count, last["block_id"], last["hash"], note, now_utc()))
        return {"block_count": count, "last_block_id": last["block_id"], "last_hash": last["hash"]}

    def latest_checkpoint(self):
        return self.conn.execute("SELECT * FROM ledger_checkpoints ORDER BY checkpoint_id DESC LIMIT 1").fetchone()

    # ---- verification -----------------------------------------------------
    def verify(self, use_checkpoint: bool = True) -> VerificationResult:
        issues = []
        blocks = self.conn.execute("SELECT * FROM ledger_blocks ORDER BY block_id").fetchall()
        prev = None
        for b in blocks:
            bid = b["block_id"]
            if prev is None:
                if bid != 0:
                    issues.append({"type": "missing_block", "block_id": 0, "detail": "Genesis block (0) is missing."})
                elif b["prev_hash"] != GENESIS_PREV_HASH:
                    issues.append({"type": "broken_link", "block_id": bid, "detail": "Genesis block has a wrong previous hash."})
            else:
                if bid != prev["block_id"] + 1:
                    issues.append({"type": "missing_block", "block_id": bid,
                                   "detail": f"Block(s) {prev['block_id'] + 1}..{bid - 1} are missing from the chain."})
                if b["prev_hash"] != prev["hash"]:
                    issues.append({"type": "broken_link", "block_id": bid,
                                   "detail": "Stored previous-hash does not match the preceding block's hash."})
            if compute_hash(bid, b["prev_hash"], b["payload_json"]) != b["hash"]:
                issues.append({"type": "altered_block", "block_id": bid,
                               "detail": "Recomputed SHA-256 differs from the stored hash (block content was changed)."})
            if b["event_id"] is not None:
                ev = self.conn.execute("SELECT * FROM supply_chain_events WHERE event_id=?", (b["event_id"],)).fetchone()
                if ev is None:
                    issues.append({"type": "event_mismatch", "block_id": bid, "detail": "The event this block anchors no longer exists."})
                elif canonical_json(event_payload(ev)) != b["payload_json"]:
                    issues.append({"type": "event_mismatch", "block_id": bid,
                                   "detail": "The event table row no longer matches the data that was hashed."})
            prev = b
        for r in self.conn.execute("""SELECT e.event_id FROM supply_chain_events e
                                      LEFT JOIN ledger_blocks b ON b.event_id = e.event_id WHERE b.block_id IS NULL"""):
            issues.append({"type": "unledgered_event", "block_id": None,
                           "detail": f"Event {r['event_id']} has no ledger block (block deleted or never written)."})
        cp = self.latest_checkpoint() if use_checkpoint else None
        if cp is not None:
            if len(blocks) < cp["block_count"]:
                issues.append({"type": "truncated_chain", "block_id": None,
                               "detail": f"Checkpoint recorded {cp['block_count']} blocks but only {len(blocks)} exist."})
            anchor = next((b for b in blocks if b["block_id"] == cp["last_block_id"]), None)
            if anchor is None or anchor["hash"] != cp["last_hash"]:
                issues.append({"type": "checkpoint_mismatch", "block_id": cp["last_block_id"],
                               "detail": "The block at the checkpoint position differs from the trusted checkpoint hash."})
        return VerificationResult(not issues, len(blocks), issues, cp is not None)


# ---- tamper demonstration (operates on an in-memory COPY only) -------------
TAMPER_MODES = {
    "alter_data": "Change the quantity inside a block's data",
    "break_link": "Overwrite a block's previous-hash link",
    "delete_middle": "Delete a block from the middle of the chain",
    "delete_last": "Delete the newest block (caught only by the checkpoint)",
    "edit_event_table": "Edit an event row in the events table",
    "rewrite_chain": "Alter a block AND recompute every later hash (caught only by the checkpoint)",
}


def clone_to_memory(conn: sqlite3.Connection) -> sqlite3.Connection:
    mem = sqlite3.connect(":memory:")
    conn.backup(mem)
    mem.row_factory = sqlite3.Row
    return mem


def run_tamper_demo(conn: sqlite3.Connection, mode: str) -> VerificationResult:
    """Copy the ledger to memory, tamper with the COPY, verify. The real database is never modified."""
    if mode not in TAMPER_MODES:
        raise ValueError("unknown tamper mode")
    mem = clone_to_memory(conn)
    try:
        ids = [r[0] for r in mem.execute("SELECT block_id FROM ledger_blocks WHERE block_id > 0 ORDER BY block_id")]
        if len(ids) < 3:
            raise ValueError("Need at least 3 recorded events to run the demonstration.")
        mid = ids[len(ids) // 2]
        if mode == "alter_data":
            p = json.loads(mem.execute("SELECT payload_json FROM ledger_blocks WHERE block_id=?", (mid,)).fetchone()[0])
            p["quantity"] += 999
            mem.execute("UPDATE ledger_blocks SET payload_json=? WHERE block_id=?", (canonical_json(p), mid))
        elif mode == "break_link":
            mem.execute("UPDATE ledger_blocks SET prev_hash=? WHERE block_id=?", ("f" * 64, mid))
        elif mode == "delete_middle":
            mem.execute("DELETE FROM ledger_blocks WHERE block_id=?", (mid,))
        elif mode == "delete_last":
            mem.execute("DELETE FROM ledger_blocks WHERE block_id=?", (ids[-1],))
        elif mode == "edit_event_table":
            mem.execute("UPDATE supply_chain_events SET quantity = quantity + 999 WHERE event_id = "
                        "(SELECT event_id FROM ledger_blocks WHERE block_id=?)", (mid,))
        elif mode == "rewrite_chain":
            p = json.loads(mem.execute("SELECT payload_json FROM ledger_blocks WHERE block_id=?", (mid,)).fetchone()[0])
            p["quantity"] += 999
            mem.execute("UPDATE ledger_blocks SET payload_json=? WHERE block_id=?", (canonical_json(p), mid))
            prev_hash = mem.execute("SELECT prev_hash FROM ledger_blocks WHERE block_id=?", (mid,)).fetchone()[0]
            for bid in [mid] + [i for i in ids if i > mid]:
                row = mem.execute("SELECT payload_json FROM ledger_blocks WHERE block_id=?", (bid,)).fetchone()
                h = compute_hash(bid, prev_hash, row[0])
                mem.execute("UPDATE ledger_blocks SET prev_hash=?, hash=? WHERE block_id=?", (prev_hash, h, bid))
                prev_hash = h
            # the attacker also fixes the event row so only the external checkpoint can expose it
            mem.execute("UPDATE supply_chain_events SET quantity = quantity + 999 WHERE event_id = "
                        "(SELECT event_id FROM ledger_blocks WHERE block_id=?)", (mid,))
        mem.commit()
        return Ledger(mem).verify()
    finally:
        mem.close()
