"""Central configuration: paths, feature lists and constants shared by every module."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DB_PATH = Path(os.environ.get("TRUSTCHAIN_DB", DATA_DIR / "trustchain.db"))
DATACO_CSV = DATA_DIR / "DataCoSupplyChainDataset.csv"
SYNTHETIC_CSV = DATA_DIR / "synthetic_shipments.csv"
BASE_URL = os.environ.get("TRUSTCHAIN_BASE_URL", "http://localhost:8501")

SOURCE_DATACO = "DataCo Smart Supply Chain"
SOURCE_SYNTHETIC = "SYNTHETIC demo data"
MODEL_VERSION = "1.0"

# ---- Features -------------------------------------------------------------
# Only information known at order/dispatch time is used. Outcomes such as the
# real shipping days, delivery status or shipping date are NEVER model inputs.
TARGET = "late_delivery"
RAW_INPUTS = ["order_date", "shipping_mode", "scheduled_days", "region", "market",
              "category", "quantity", "unit_price", "discount_rate", "customer_segment"]
CATEGORICAL = ["shipping_mode", "region", "market", "category", "customer_segment"]
NUMERIC = ["scheduled_days", "quantity", "unit_price", "discount_rate",
           "order_value", "order_month", "order_dow"]
FEATURES = CATEGORICAL + NUMERIC
LEAKAGE_COLUMNS = ["late_delivery", "actual_days", "delivery_status", "shipping_date"]

RISK_BANDS = (0.40, 0.70)  # <0.40 Low, 0.40-0.70 Medium, >=0.70 High
MAX_UPLOAD_ROWS = 50_000

EVENT_TYPES = ["MANUFACTURED", "DISPATCHED", "RECEIVED", "INSPECTED", "DELIVERED"]
INVESTIGATION_STATUSES = ["Open", "In Review", "Escalated",
                          "Closed - Confirmed Issue", "Closed - False Alarm"]
PRIORITIES = ["Low", "Medium", "High"]
