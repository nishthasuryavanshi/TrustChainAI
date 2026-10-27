"""Derived features and the shared scikit-learn preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame (only information known before delivery)."""
    out = df.copy()
    d = pd.to_datetime(out["order_date"], errors="coerce")
    out["order_month"] = d.dt.month.fillna(1).astype(int)
    out["order_dow"] = d.dt.dayofweek.fillna(0).astype(int)
    out["order_value"] = out["quantity"] * out["unit_price"] * (1 - out["discount_rate"])
    return out[cfg.FEATURES]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.CATEGORICAL),
        ("num", StandardScaler(), cfg.NUMERIC),
    ])
