"""Page 4: Ledger entries, integrity verification, tamper demonstration (test copy only)."""
import pandas as pd
import streamlit as st

from src import blockchain as bc
from src import database as db
from src import utils

utils.page_setup("Blockchain Verification")
st.title("Blockchain Verification")
utils.require_ready()
st.info("**What this is:** a local, hash-linked SHA-256 ledger in SQLite. Changes made after an event was written are detectable. "
        "**What it is not:** a decentralised blockchain (one operator controls it, there is no consensus), and it cannot prove the "
        "original data was true. Deleting the newest blocks is only detectable against a saved checkpoint.")

with db.session() as conn:
    ledger = bc.Ledger(conn)
    res = ledger.verify()
    cp = ledger.latest_checkpoint()
    blocks = pd.read_sql_query(
        """SELECT b.block_id, e.batch_id, e.event_type, e.participant_id, e.timestamp, e.location, e.quantity,
                  b.prev_hash, b.hash FROM ledger_blocks b LEFT JOIN supply_chain_events e ON e.event_id = b.event_id
           ORDER BY b.block_id""", conn)

if res.valid:
    st.success(f"Ledger VALID - {res.blocks_checked} blocks verified" + (" against the latest checkpoint." if res.checkpoint_used else "."))
else:
    st.error(f"Ledger INVALID - {len(res.issues)} problem(s) found.")
    st.dataframe(pd.DataFrame(res.issues), hide_index=True)

c1, c2, c3 = st.columns(3)
c1.metric("Blocks", res.blocks_checked)
c2.metric("Latest block hash", utils.short_hash(blocks.hash.iloc[-1]))
c3.metric("Latest checkpoint", f"{cp['block_count']} blocks" if cp else "none", cp["created_at"] if cp else None, delta_color="off")
if st.button("Save a trusted checkpoint now"):
    if res.valid:
        with db.session() as conn:
            bc.Ledger(conn).save_checkpoint("Saved from dashboard")
        st.rerun()
    else:
        st.error("Refusing to checkpoint an invalid ledger.")
st.caption("A checkpoint stores the block count and last hash. For real use, keep a copy somewhere the ledger operator cannot edit.")

st.subheader("Ledger entries")
q = st.text_input("Filter by batch ID or event type")
view = blocks if not q else blocks[blocks.apply(lambda r: q.lower() in f"{r.batch_id} {r.event_type}".lower(), axis=1)]
show = view.assign(prev_hash=view.prev_hash.map(utils.short_hash), hash=view.hash.map(utils.short_hash))
st.dataframe(show, hide_index=True)
with st.expander("Full hashes (copy/export)"):
    st.dataframe(view[["block_id", "prev_hash", "hash"]], hide_index=True)
    st.download_button("Export ledger (CSV)", blocks.to_csv(index=False), "ledger_export.csv", "text/csv")

st.subheader("Tampering demonstration (test copy only)")
st.caption("This copies the ledger into memory, tampers with the COPY, and runs the verifier. Your real data is never modified.")
mode = st.selectbox("Simulated attack", list(bc.TAMPER_MODES), format_func=lambda k: bc.TAMPER_MODES[k])
if st.button("Run tamper test"):
    try:
        with db.session() as conn:
            demo = bc.run_tamper_demo(conn, mode)
        if demo.valid:
            st.warning("The verifier did NOT detect this change.")
        else:
            st.error(f"Tampering detected - {len(demo.issues)} issue(s):")
            st.dataframe(pd.DataFrame(demo.issues), hide_index=True)
        if mode == "rewrite_chain":
            st.caption("Recomputing every hash makes the chain internally consistent; only the external checkpoint exposes it. "
                       "This is why real systems anchor hashes somewhere the operator cannot rewrite (e.g. a public chain).")
    except ValueError as e:
        st.error(str(e))
