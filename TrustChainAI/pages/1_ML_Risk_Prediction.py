"""Page 2: ML risk prediction (single form + CSV batch)."""
import io
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as cfg
from src import database as db
from src import predict_risk as pr
from src import utils
from src.data_preprocessing import validate_input_frame

utils.page_setup("ML Risk Prediction")
st.title("ML Risk Prediction")
st.caption("Estimate late-delivery risk from information known **before** delivery. Outcome columns are never used.")
utils.require_ready()

m = utils.load_metrics()
with st.expander("Model performance (held-out test set)"):
    rf, lr = m["random_forest"], m["logistic_regression_baseline"]
    st.dataframe(pd.DataFrame({"Random Forest": rf, "Logistic Regression (baseline)": lr}).drop("confusion_matrix"))
    st.caption(f"Data source: {m['data_source']}. {m['note']}")
    st.write("Random Forest confusion matrix [[TN, FP], [FN, TP]]:", rf["confusion_matrix"])

with db.session() as conn:
    opts = {c: db.distinct_values(conn, c) for c in cfg.CATEGORICAL}

tab1, tab2 = st.tabs(["Single shipment", "Batch CSV upload"])

with tab1:
    with st.form("single"):
        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Shipping mode", opts["shipping_mode"])
        sched = c1.number_input("Scheduled days", 0, 60, 4)
        odate = c1.date_input("Order date", date.today())
        region = c2.selectbox("Region", opts["region"])
        market = c2.selectbox("Market", opts["market"])
        cat = c2.selectbox("Category", opts["category"])
        qty = c3.number_input("Quantity", 1, 10000, 2)
        price = c3.number_input("Unit price", 0.0, 1_000_000.0, 120.0)
        disc = c3.slider("Discount rate", 0.0, 1.0, 0.1, 0.01)
        seg = st.selectbox("Customer segment", opts["customer_segment"])
        go_btn = st.form_submit_button("Predict risk", type="primary")
    if go_btn:
        raw = pd.DataFrame([{"shipment_id": f"MANUAL-{pd.Timestamp.now():%Y%m%d%H%M%S}", "order_date": str(odate),
                             "shipping_mode": mode, "scheduled_days": sched, "region": region, "market": market,
                             "category": cat, "quantity": qty, "unit_price": price, "discount_rate": disc,
                             "customer_segment": seg}])
        clean, errs, warns = validate_input_frame(raw)
        if errs:
            st.error(" ".join(errs))
        else:
            try:
                res = pr.predict_frame(clean).iloc[0]
                factors = pr.explain_single(clean.iloc[0].to_dict())
            except pr.ModelsNotFoundError as e:
                st.error(str(e))
                st.stop()
            with db.session() as conn:
                db.log_predictions(conn, res.to_frame().T, "manual")
            left, right = st.columns([1, 1])
            fig = go.Figure(go.Indicator(mode="gauge+number", value=res.risk_probability * 100,
                                         number={"suffix": "%"}, title={"text": "Late-delivery probability"},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": utils.RISK_COLORS[res.risk_level]},
                                                "steps": [{"range": [0, 40], "color": "#12332f"},
                                                          {"range": [40, 70], "color": "#3a2f12"},
                                                          {"range": [70, 100], "color": "#3a1616"}]}))
            left.plotly_chart(utils.style_fig(fig, 300))
            right.markdown(f"**Classification:** {utils.risk_badge(res.risk_level)}", unsafe_allow_html=True)
            if res.is_anomaly:
                right.markdown(utils.badge("Unusual shipment pattern", "warn") +
                               f" &nbsp; anomaly score {res.anomaly_score:.3f}", unsafe_allow_html=True)
                right.caption("Statistically unusual vs. training data. This is not evidence of fraud.")
            else:
                right.markdown(utils.badge("Normal pattern", "ok") + f" &nbsp; anomaly score {res.anomaly_score:.3f}",
                               unsafe_allow_html=True)
            st.subheader("Main factors")
            fdf = pd.DataFrame(factors)
            fdf["label"] = fdf.feature.str.replace("_", " ").str.title() + " = " + fdf.value.astype(str)
            fdf["effect"] = fdf.impact.map(lambda v: "raises risk" if v > 0 else "lowers risk")
            import plotly.express as px
            fig = px.bar(fdf.iloc[::-1], x="impact", y="label", orientation="h", color="effect",
                         color_discrete_map={"raises risk": utils.RED, "lowers risk": utils.TEAL},
                         labels={"impact": "Change in risk probability vs. a typical value", "label": ""})
            fig.update_xaxes(tickformat="+.0%")
            st.plotly_chart(utils.style_fig(fig, 300))
            st.caption("Approximation: each bar shows how much the probability changes if that input were replaced "
                       "by its typical training value. It describes the model, not causes in the real world.")

with tab2:
    template = pd.DataFrame([{"shipment_id": "DEMO-1", "order_date": "2026-10-01", "shipping_mode": opts["shipping_mode"][0],
                              "scheduled_days": 4, "region": opts["region"][0], "market": opts["market"][0],
                              "category": opts["category"][0], "quantity": 2, "unit_price": 99.5, "discount_rate": 0.1,
                              "customer_segment": opts["customer_segment"][0]}])
    st.download_button("Download CSV template", template.to_csv(index=False), "shipment_template.csv", "text/csv")
    up = st.file_uploader("Upload shipments CSV", type=["csv"])
    if up is not None:
        try:
            raw = pd.read_csv(io.BytesIO(up.getvalue()))
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            st.stop()
        clean, errs, warns = validate_input_frame(raw)
        for w in warns:
            st.warning(w)
        if errs:
            for e in errs:
                st.error(e)
        else:
            unseen = [f"{c}: {sorted(set(clean[c]) - set(opts[c]))[:5]}" for c in cfg.CATEGORICAL if set(clean[c]) - set(opts[c])]
            if unseen:
                st.info("Values not seen in training (treated as neutral): " + "; ".join(unseen))
            bar = st.progress(0.0, text="Scoring…")
            try:
                res = pr.predict_frame(clean)
            except pr.ModelsNotFoundError as e:
                st.error(str(e))
                st.stop()
            bar.progress(1.0, text="Done")
            with db.session() as conn:
                db.log_predictions(conn, res, "batch_upload")
            k1, k2, k3 = st.columns(3)
            k1.metric("Rows scored", f"{len(res):,}")
            k2.metric("High risk", int((res.risk_level == "High").sum()))
            k3.metric("Unusual patterns", int(res.is_anomaly.sum()))
            st.dataframe(res.sort_values("risk_probability", ascending=False), hide_index=True)
            st.download_button("Download predictions (CSV)", res.to_csv(index=False), "trustchain_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV with the columns in the template to score many shipments at once.")

with st.expander("Recent prediction history"):
    with db.session() as conn:
        h = db.recent_predictions(conn)
    st.dataframe(h, hide_index=True) if len(h) else st.caption("No predictions yet.")
