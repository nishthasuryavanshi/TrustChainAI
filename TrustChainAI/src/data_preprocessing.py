"""Loading, cleaning and validating shipment data."""
from pathlib import Path

import pandas as pd

from . import config as cfg

DATACO_MAP = {
    "Order Item Id": "shipment_id", "order date (DateOrders)": "order_date",
    "Shipping Mode": "shipping_mode", "Days for shipment (scheduled)": "scheduled_days",
    "Order Region": "region", "Market": "market", "Category Name": "category",
    "Order Item Quantity": "quantity", "Order Item Product Price": "unit_price",
    "Order Item Discount Rate": "discount_rate", "Customer Segment": "customer_segment",
    "Late_delivery_risk": "late_delivery", "Days for shipping (real)": "actual_days",
}
_NUM = ["scheduled_days", "quantity", "unit_price", "discount_rate"]


class ValidationError(ValueError):
    """Raised when a dataset or upload does not match the expected schema."""


def load_dataco(path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in DATACO_MAP if c not in raw.columns]
    if missing:
        raise ValidationError(f"Not a DataCo file - missing columns: {missing}")
    return raw[list(DATACO_MAP)].rename(columns=DATACO_MAP)


def clean_training_data(df: pd.DataFrame):
    """Remove duplicates/invalid rows, impute gaps. Returns (clean_df, report)."""
    report = {"rows_in": int(len(df))}
    df = df.drop_duplicates().copy()
    report["duplicates_removed"] = report["rows_in"] - len(df)
    df["shipment_id"] = df["shipment_id"].astype(str).str.strip()
    df = df.drop_duplicates("shipment_id")
    for c in _NUM + ["actual_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df[cfg.TARGET] = pd.to_numeric(df[cfg.TARGET], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["order_date", cfg.TARGET])
    df = df[df[cfg.TARGET].isin([0, 1])]
    report["rows_dropped_missing_target_or_date"] = before - len(df)
    for c in cfg.CATEGORICAL:
        df[c] = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str).str.strip()
    report["values_imputed"] = int(df[_NUM].isna().sum().sum())
    for c in _NUM:
        df[c] = df[c].fillna(df[c].median())
    ok = (df.quantity >= 1) & (df.unit_price >= 0) & df.discount_rate.between(0, 1) & (df.scheduled_days >= 0)
    report["rows_dropped_invalid_ranges"] = int((~ok).sum())
    df = df[ok].copy()
    df["quantity"] = df["quantity"].astype(int)
    df["scheduled_days"] = df["scheduled_days"].astype(int)
    df[cfg.TARGET] = df[cfg.TARGET].astype(int)
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report


def load_dataset(path=None, n_synthetic: int = 15000):
    """Returns (clean_df, source_label, cleaning_report). Falls back to synthetic data."""
    path = Path(path) if path else (cfg.DATACO_CSV if cfg.DATACO_CSV.exists() else None)
    if path is not None:
        df, report = clean_training_data(load_dataco(path))
        return df, cfg.SOURCE_DATACO, report
    from .synthetic_data import generate_shipments
    cfg.DATA_DIR.mkdir(exist_ok=True)
    raw = generate_shipments(n_synthetic)
    raw.to_csv(cfg.SYNTHETIC_CSV, index=False)
    df, report = clean_training_data(raw)
    return df, cfg.SOURCE_SYNTHETIC, report


def validate_input_frame(df: pd.DataFrame):
    """Validate a user-supplied frame (CSV upload / form). Returns (clean_df, errors, warnings).

    Invalid rows are dropped and reported; clean_df is None when nothing usable remains.
    Outcome columns are ignored so they can never leak into predictions.
    """
    errors, warnings = [], []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) == 0:
        return None, ["The file contains no rows."], warnings
    if len(df) > cfg.MAX_UPLOAD_ROWS:
        return None, [f"Too many rows ({len(df):,}); the limit is {cfg.MAX_UPLOAD_ROWS:,}."], warnings
    missing = [c for c in cfg.RAW_INPUTS if c not in df.columns]
    if missing:
        return None, [f"Missing required columns: {', '.join(missing)}"], warnings
    ignored = [c for c in cfg.LEAKAGE_COLUMNS if c in df.columns]
    if ignored:
        warnings.append(f"Outcome columns ignored (not allowed as inputs): {', '.join(ignored)}")
    if "shipment_id" not in df.columns:
        df["shipment_id"] = [f"UPLOAD-{i+1:06d}" for i in range(len(df))]
    df["shipment_id"] = df["shipment_id"].astype(str).str.strip()
    out = df[["shipment_id"] + cfg.RAW_INPUTS].copy()
    for c in _NUM:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce")
    for c in cfg.CATEGORICAL:
        out[c] = out[c].astype("object").where(out[c].notna(), "").astype(str).str.strip()
    ok = (out[_NUM].notna().all(axis=1) & out.order_date.notna() & (out.quantity >= 1)
          & (out.unit_price >= 0) & out.discount_rate.between(0, 1) & (out.scheduled_days >= 0)
          & (out[cfg.CATEGORICAL] != "").all(axis=1))
    if (~ok).sum():
        warnings.append(f"{int((~ok).sum())} invalid row(s) were skipped "
                        "(missing/negative/out-of-range values or unparseable dates).")
    out = out[ok].copy()
    if out.empty:
        return None, ["No valid rows remain after validation."], warnings
    out["order_date"] = out["order_date"].dt.strftime("%Y-%m-%d")
    out["quantity"] = out["quantity"].astype(int)
    out["scheduled_days"] = out["scheduled_days"].astype(int)
    return out.reset_index(drop=True), errors, warnings
