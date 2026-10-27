"""Page 5: Investigation center - triage high-risk / unusual shipments and record outcomes."""
import pandas as pd
import streamlit as st

from src import config as cfg
from src import database as db
from src import utils

utils.page_setup("Investigation Center")
st.title("Investigation Center")
st.caption("Predictions and anomaly flags are leads for human review - they are not findings of fraud or wrongdoing.")
utils.require_ready()

with db.session() as conn:
    ships = db.fetch_shipments(conn)
    inv = db.list_investigations(conn)

k1, k2, k3 = st.columns(3)
k1.metric("Open investigations", int(inv.status.isin(["Open", "In Review", "Escalated"]).sum()) if len(inv) else 0)
k2.metric("Closed - confirmed issue", int((inv.status == "Closed - Confirmed Issue").sum()) if len(inv) else 0)
k3.metric("Closed - false alarm", int((inv.status == "Closed - False Alarm").sum()) if len(inv) else 0)

t1, t2, t3 = st.tabs(["High-risk queue", "Unusual patterns", "Investigations"])
cols = ["shipment_id", "status", "shipping_mode", "region", "category", "quantity", "unit_price",
        "risk_probability", "risk_level", "anomaly_score"]
with t1:
    only_transit = st.checkbox("Only shipments still in transit", True)
    q = ships[ships.risk_level == "High"]
    q = q[q.status == "In Transit"] if only_transit else q
    st.caption(f"{len(q):,} high-risk shipments")
    st.dataframe(q[cols].sort_values("risk_probability", ascending=False).head(500), hide_index=True)
with t2:
    a = ships[ships.is_anomaly == 1].sort_values("anomaly_score", ascending=False)
    st.caption(f"{len(a):,} shipments flagged as statistically unusual (Isolation Forest)")
    st.dataframe(a[cols + ["discount_rate"]].head(500), hide_index=True)

with t3:
    st.subheader("Open or update an investigation")
    pool = ships[(ships.risk_level == "High") | (ships.is_anomaly == 1)].sort_values("risk_probability", ascending=False)
    typed = st.text_input("Search shipment ID", placeholder="SHP-000123")
    pool = pool[pool.shipment_id.str.contains(typed.strip(), case=False, regex=False)] if typed.strip() else pool
    if pool.empty:
        st.info("No matching high-risk or unusual shipments.")
    else:
        sid = st.selectbox("Shipment", pool.shipment_id.head(500).tolist())
        s = ships[ships.shipment_id == sid].iloc[0]
        st.markdown(f"{utils.risk_badge(s.risk_level)} {utils.badge('Unusual pattern', 'warn') if s.is_anomaly else ''} "
                    f"risk **{s.risk_probability:.0%}** · {s.shipping_mode} · {s.region} · {s.category} · status {s.status}",
                    unsafe_allow_html=True)
        with db.session() as conn:
            hist = db.recent_predictions(conn, 500)
            cur = conn.execute("SELECT * FROM investigations WHERE shipment_id=?", (sid,)).fetchone()
            batch_events = db.events_for_batch(conn, s.batch_id) if s.batch_id else pd.DataFrame()
        h = hist[hist.shipment_id == sid]
        with st.expander("Shipment history"):
            st.write(f"Linked batch: **{s.batch_id or 'none'}**")
            if len(batch_events):
                st.dataframe(batch_events[["event_type", "timestamp", "location", "participant_id"]], hide_index=True)
            st.write("Prediction history:")
            st.dataframe(h, hide_index=True) if len(h) else st.caption("No manual/batch predictions logged for this shipment.")
        with st.form("inv"):
            c1, c2 = st.columns(2)
            status = c1.selectbox("Status", cfg.INVESTIGATION_STATUSES, index=cfg.INVESTIGATION_STATUSES.index(cur["status"]) if cur else 0)
            prio = c2.selectbox("Priority", cfg.PRIORITIES, index=cfg.PRIORITIES.index(cur["priority"]) if cur else 1)
            notes = st.text_area("Notes (avoid personal data)", cur["notes"] if cur else "")
            outcome = st.text_area("Outcome", cur["outcome"] if cur else "")
            if st.form_submit_button("Save investigation", type="primary"):
                if status.startswith("Closed") and not outcome.strip():
                    st.error("Please describe the outcome before closing.")
                else:
                    with db.session() as conn:
                        db.upsert_investigation(conn, sid, status, prio, notes, outcome)
                    st.success("Saved.")
                    st.rerun()
    st.subheader("All investigations")
    if inv.empty:
        st.info("No investigations yet. Pick a shipment above to start one.")
    else:
        fs = st.multiselect("Filter by status", cfg.INVESTIGATION_STATUSES)
        st.dataframe(inv[inv.status.isin(fs)] if fs else inv, hide_index=True)
