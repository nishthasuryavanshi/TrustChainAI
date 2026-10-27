"""Page 3: Product traceability - batches, event timeline, QR lookup."""
import uuid

import plotly.express as px
import streamlit as st

from src import config as cfg
from src import database as db
from src import utils
from src.blockchain import Ledger

utils.page_setup("Product Traceability")
st.title("Product Traceability")
st.caption("Follow a product batch from manufacture to delivery. Every event is written to the hash-linked ledger.")
utils.require_ready()

with db.session() as conn:
    batches = db.list_batches(conn)
    ids = batches.batch_id.tolist()

with st.expander("Create a new batch"):
    with st.form("new_batch", clear_on_submit=True):
        c1, c2 = st.columns(2)
        bid = c1.text_input("Batch ID", f"BATCH-{uuid.uuid4().hex[:6].upper()}")
        name = c1.text_input("Product name")
        cat = c1.text_input("Category")
        qty = c2.number_input("Quantity", 1, 10_000_000, 100)
        sup = c2.text_input("Supplier / manufacturer ID")
        origin = c2.text_input("Manufacturing location")
        if st.form_submit_button("Create batch & record MANUFACTURED event", type="primary"):
            if not all(x.strip() for x in (bid, name, cat, sup, origin)):
                st.error("All fields are required.")
            else:
                try:
                    with db.session() as conn:
                        db.create_batch(conn, bid, name, cat, qty, sup, origin)
                        Ledger(conn).add_event(bid.strip(), "MANUFACTURED", sup, origin, qty)
                    st.success(f"Batch {bid} created.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not create batch (duplicate ID?): {e}")

if not ids:
    st.info("No batches yet. Create one above.")
    st.stop()

preset = st.query_params.get("batch")
typed = st.text_input("Search batch ID", value=preset or "", placeholder="BATCH-2026001")
matches = [i for i in ids if typed.strip().lower() in i.lower()] if typed.strip() else ids
if not matches:
    st.warning(f"No batch matches '{typed}'.")
    st.stop()
batch_id = st.selectbox("Batch", matches)

with db.session() as conn:
    b = db.get_batch(conn, batch_id)
    ev = db.events_for_batch(conn, batch_id)
    n_ship = conn.execute("SELECT COUNT(*) FROM shipments WHERE batch_id=?", (batch_id,)).fetchone()[0]

left, right = st.columns([3, 1])
with left:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product", b["product_name"])
    c2.metric("Quantity", f"{b['quantity']:,}")
    c3.metric("Supplier", b["supplier_id"])
    c4.metric("Linked shipments", n_ship)
    last = ev.iloc[-1] if len(ev) else None
    if last is not None:
        done = last.event_type == "DELIVERED"
        st.markdown(f"**Current stage:** {utils.badge(last.event_type, 'ok' if done else 'info')} at {last.location} "
                    f"· {last.timestamp}", unsafe_allow_html=True)
with right:
    url = f"{cfg.BASE_URL}/Product_Traceability?batch={batch_id}"
    st.image(utils.qr_png(url), caption="Scan to open this batch", width=150)
    st.download_button("Download QR", utils.qr_png(url), f"{batch_id}_qr.png", "image/png")

st.subheader("Product journey")
if ev.empty:
    st.info("No events recorded for this batch yet.")
else:
    ev["stage"] = ev.event_type.str.title()
    fig = px.line(ev, x="timestamp", y="stage", markers=True, text="location", hover_data=["participant_id", "quantity"],
                  category_orders={"stage": [e.title() for e in cfg.EVENT_TYPES]})
    fig.update_traces(textposition="top center", marker_size=12, line_color=utils.TEAL)
    fig.update_yaxes(title="")
    fig.update_xaxes(title="")
    st.plotly_chart(utils.style_fig(fig, 320))
    st.dataframe(ev[["block_id", "event_type", "timestamp", "location", "participant_id", "quantity"]], hide_index=True)

st.subheader("Record a new event")
with st.form("new_event", clear_on_submit=True):
    c1, c2, c3 = st.columns(3)
    et = c1.selectbox("Event type", cfg.EVENT_TYPES)
    who = c2.text_input("Participant ID", b["supplier_id"])
    loc = c3.text_input("Location")
    q = c1.number_input("Quantity", 0, 10_000_000, int(b["quantity"]))
    if st.form_submit_button("Record event on ledger", type="primary"):
        try:
            with db.session() as conn:
                blk = Ledger(conn).add_event(batch_id, et, who, loc, q)
            st.success(f"Event recorded in ledger block #{blk}.")
            st.rerun()
        except ValueError as e:
            st.error(str(e))
