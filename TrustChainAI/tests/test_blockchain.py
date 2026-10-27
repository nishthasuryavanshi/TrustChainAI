import pytest
from src import blockchain as bc


def types(res):
    return {i["type"] for i in res.issues}


def test_clean_ledger_is_valid(ledger):
    assert ledger.verify().valid


def test_altered_block_detected(ledger):
    ledger.conn.execute("UPDATE ledger_blocks SET payload_json = replace(payload_json, '100', '999') WHERE block_id = 3")
    assert "altered_block" in types(ledger.verify())


def test_broken_link_detected(ledger):
    ledger.conn.execute("UPDATE ledger_blocks SET prev_hash = ? WHERE block_id = 2", ("a" * 64,))
    assert "broken_link" in types(ledger.verify())


def test_missing_middle_block_detected(ledger):
    ledger.conn.execute("DELETE FROM ledger_blocks WHERE block_id = 3")
    t = types(ledger.verify())
    assert "missing_block" in t and "broken_link" in t


def test_deleted_last_block_needs_checkpoint(ledger):
    ledger.conn.execute("DELETE FROM ledger_blocks WHERE block_id = 5")
    assert "truncated_chain" in types(ledger.verify(use_checkpoint=True))
    # the events table still has the event, so it is also caught without a checkpoint
    assert "unledgered_event" in types(ledger.verify(use_checkpoint=False))


def test_event_table_edit_detected(ledger):
    ledger.conn.execute("UPDATE supply_chain_events SET quantity = 5 WHERE event_id = 2")
    assert "event_mismatch" in types(ledger.verify())


def test_full_rewrite_only_caught_by_checkpoint(ledger):
    res = bc.run_tamper_demo(ledger.conn, "rewrite_chain")
    assert "checkpoint_mismatch" in types(res)


@pytest.mark.parametrize("mode", list(bc.TAMPER_MODES))
def test_every_demo_mode_is_detected_and_real_db_untouched(ledger, mode):
    assert not bc.run_tamper_demo(ledger.conn, mode).valid
    assert ledger.verify().valid


def test_canonical_json_is_deterministic():
    assert bc.canonical_json({"b": 1, "a": 2}) == bc.canonical_json({"a": 2, "b": 1}) == '{"a":2,"b":1}'


def test_invalid_event_rejected(ledger):
    with pytest.raises(ValueError):
        ledger.add_event("B1", "STOLEN", "P", "X", 1)
    with pytest.raises(ValueError):
        ledger.add_event("B1", "RECEIVED", "P", "X", -5)
