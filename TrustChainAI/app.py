"""TrustChain AI - Page 1: Executive Dashboard.  Run with:  streamlit run app.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config as cfg
from src import database as db
from src import utils

utils.page_setup("Executive Dashboard")
st.title("Executive Dashboard")
st.caption("Delivery performance, predicted risk and anomaly alerts across all shipments.")
utils.require_ready()

with db.session() as conn:
    df = db.fetch_shipments(conn)

source = df["data_source"].iloc[0]
if source == cfg.SOURCE_SYNTHETIC:
    st.warning("Showing **synthetic demo data**. Figures illustrate the system only and are not real supply-chain results.")

with st.sidebar:
    st.subheader("Filters")
    modes = st.multiselect("Shipping mode", sorted(df.shipping_mode.unique()), default=sorted(df.shipping_mode.unique()))
    regions = st.multiselect("Region", sorted(df.region.unique()), default=sorted(df.region.unique()))
f = df[df.shipping_mode.isin(modes) & df.region.isin(regions)]
if f.empty:
    st.info("No shipments match the current filters.")
    st.stop()

delivered = f[f.status == "Delivered"]
in_transit = f[f.status == "In Transit"]
delayed = int((delivered.late_delivery == 1).sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total shipments", f"{len(f):,}", f"{len(in_transit):,} in transit", delta_color="off")
c2.metric("Delayed shipments", f"{delayed:,}", f"{delayed / max(len(delivered), 1):.1%} of delivered", delta_color="off")
c3.metric("High-risk shipments", f"{int((f.risk_level == 'High').sum()):,}",
          f"{int(((f.risk_level == 'High') & (f.status == 'In Transit')).sum())} still in transit", delta_color="off")
c4.metric("Anomaly alerts", f"{int(f.is_anomaly.sum()):,}", "statistical outliers, not fraud", delta_color="off")

st.divider()
a, b = st.columns(2)
by_mode = delivered.groupby("shipping_mode").late_delivery.agg(["mean", "count"]).reset_index()
fig = px.bar(by_mode, x="shipping_mode", y="mean", text=by_mode["mean"].map("{:.0%}".format),
             title="Late-delivery rate by shipping mode (delivered shipments)", labels={"mean": "Late rate", "shipping_mode": ""})
fig.update_yaxes(tickformat=".0%")
a.plotly_chart(utils.style_fig(fig))

lvl = f.risk_level.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
lvl.columns = ["risk_level", "count"]
fig = px.pie(lvl, names="risk_level", values="count", hole=.55, color="risk_level",
             color_discrete_map=utils.RISK_COLORS, title="Predicted risk distribution")
b.plotly_chart(utils.style_fig(fig))

a, b = st.columns(2)
d = delivered.assign(month=pd.to_datetime(delivered.order_date).dt.to_period("M").dt.to_timestamp())
trend = d.groupby("month").agg(late_rate=("late_delivery", "mean"), shipments=("shipment_id", "count")).reset_index()
fig = px.line(trend, x="month", y="late_rate", markers=True, title="Monthly late-delivery rate", labels={"late_rate": "Late rate", "month": ""})
fig.update_yaxes(tickformat=".0%")
a.plotly_chart(utils.style_fig(fig))

reg = delivered.groupby("region").late_delivery.mean().sort_values().reset_index()
fig = px.bar(reg, x="late_delivery", y="region", orientation="h", title="Late-delivery rate by region",
             labels={"late_delivery": "Late rate", "region": ""})
fig.update_xaxes(tickformat=".0%")
b.plotly_chart(utils.style_fig(fig))

heat = f.pivot_table(index="region", columns="shipping_mode", values="risk_probability", aggfunc="mean")
fig = px.imshow(heat, aspect="auto", color_continuous_scale="Tealrose", text_auto=".0%",
                title="Average predicted risk: region × shipping mode")
st.plotly_chart(utils.style_fig(fig, 380))

st.subheader("Shipments")
q1, q2, q3 = st.columns([2, 1, 1])
text = q1.text_input("Search shipment ID", placeholder="e.g. SHP-000123")
lv = q2.multiselect("Risk level", ["Low", "Medium", "High"])
stt = q3.multiselect("Status", ["Delivered", "In Transit"])
t = f
if text:
    t = t[t.shipment_id.str.contains(text.strip(), case=False, regex=False)]
if lv:
    t = t[t.risk_level.isin(lv)]
if stt:
    t = t[t.status.isin(stt)]
cols = ["shipment_id", "status", "order_date", "shipping_mode", "region", "category", "scheduled_days",
        "risk_probability", "risk_level", "anomaly_score", "is_anomaly", "late_delivery"]
st.caption(f"{len(t):,} shipments match")
st.dataframe(t[cols].sort_values("risk_probability", ascending=False).head(1000), hide_index=True,
             column_config={"risk_probability": st.column_config.ProgressColumn("Risk probability", min_value=0, max_value=1, format="%.2f")})

with st.expander("Model information"):
    m = utils.load_metrics()
    st.write(f"Data source: **{m.get('data_source')}** · trained {m.get('trained_at')} · "
             f"{m.get('rows_train'):,} train / {m.get('rows_test'):,} test rows")
    st.caption(m.get("note", ""))
