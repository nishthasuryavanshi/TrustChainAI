"""Shared Streamlit helpers: theme, badges, charts styling, setup guard, QR codes."""
import html
import io
import json

import qrcode
import streamlit as st

from . import config as cfg
from . import database as db
from .predict_risk import models_available

TEAL, BLUE, AMBER, RED, SLATE = "#14b8a6", "#3b82f6", "#f59e0b", "#ef4444", "#64748b"
RISK_COLORS = {"Low": TEAL, "Medium": AMBER, "High": RED}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
[data-testid="stMetric"] {background:#111a2e; border:1px solid #1e2a44; border-left:3px solid #14b8a6;
    padding:14px 16px; border-radius:10px;}
[data-testid="stMetricLabel"] {color:#94a3b8;}
[data-testid="stSidebar"] {border-right:1px solid #1e2a44;}
.tc-brand {font-weight:700; font-size:1.2rem; letter-spacing:.3px; color:#e2e8f0;}
.tc-brand span {color:#14b8a6;}
.tc-sub {color:#94a3b8; font-size:.8rem; margin-bottom:.5rem;}
.tc-badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid;}
.tc-ok {color:#14b8a6; border-color:#14b8a6; background:#14b8a618;}
.tc-warn {color:#f59e0b; border-color:#f59e0b; background:#f59e0b18;}
.tc-bad {color:#ef4444; border-color:#ef4444; background:#ef444418;}
.tc-info {color:#3b82f6; border-color:#3b82f6; background:#3b82f618;}
.tc-hash {font-family:ui-monospace,Menlo,monospace; font-size:.8rem; color:#94a3b8;}
</style>
"""


def page_setup(title: str, icon: str = "🛡️"):
    st.set_page_config(page_title=f"{title} · TrustChain AI", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="tc-brand">Trust<span>Chain</span> AI</div>'
                    '<div class="tc-sub">Supply-chain risk &amp; traceability</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="tc-badge tc-{kind}">{html.escape(str(text))}</span>'


def risk_badge(level: str) -> str:
    return badge(f"{level} risk", {"Low": "ok", "Medium": "warn", "High": "bad"}.get(level, "info"))


def style_fig(fig, height: int = 340):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      colorway=[TEAL, BLUE, AMBER, RED, SLATE], legend_title_text="")
    return fig


def short_hash(h, n: int = 10) -> str:
    return f"{h[:n]}…{h[-4:]}" if h else ""


def load_metrics() -> dict:
    p = cfg.MODELS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def qr_png(text: str) -> bytes:
    buf = io.BytesIO()
    qrcode.make(text).save(buf, format="PNG")
    return buf.getvalue()


def require_ready():
    """Stop the page with a friendly empty state until data + models exist."""
    if db.db_ready() and models_available():
        return
    from . import bootstrap
    st.title("Welcome to TrustChain AI")
    st.info("No database or trained models found yet. Generate the demo environment "
            "(takes about 30-60 seconds): it loads data, trains the models and seeds the ledger.")
    st.caption("Using the real DataCo dataset? Put `DataCoSupplyChainDataset.csv` in the `data/` folder first. "
               "Otherwise a clearly labelled SYNTHETIC dataset is generated.")
    if st.button("Generate demo data and train models", type="primary"):
        with st.status("Setting up…", expanded=True) as status:
            bootstrap.run_setup(progress=st.write)
            status.update(label="Setup complete", state="complete")
        st.rerun()
    st.stop()
