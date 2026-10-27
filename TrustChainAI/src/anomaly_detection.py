"""Isolation Forest anomaly detection.

An anomaly is only a statistically unusual shipment - NOT confirmed fraud.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .feature_engineering import prepare


def build_anomaly_model(contamination: float = 0.02, seed: int = 42) -> Pipeline:
    # numeric features only: unusual quantities / prices / discounts / lead times
    prep = ColumnTransformer([("num", StandardScaler(), cfg.NUMERIC)])
    return Pipeline([("prep", prep),
                     ("iforest", IsolationForest(n_estimators=200, contamination=contamination,
                                                 random_state=seed, n_jobs=-1))])


def score_anomalies(model: Pipeline, df: pd.DataFrame):
    """Returns (score, flag). Higher score = more unusual."""
    X = prepare(df)
    return -model.score_samples(X), model.predict(X) == -1


def inject_synthetic_anomalies(df: pd.DataFrame, frac: float = 0.02, seed: int = 7) -> pd.DataFrame:
    """SIMULATED suspicious activity (extreme quantity/price/discount) for demonstration only."""
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    out["is_injected_anomaly"] = 0
    idx = rng.choice(len(out), max(1, int(len(out) * frac)), replace=False)
    out.loc[idx, "quantity"] = (out.loc[idx, "quantity"] * rng.integers(15, 40, len(idx))).astype(int)
    out.loc[idx, "unit_price"] = out.loc[idx, "unit_price"] * rng.uniform(6, 15, len(idx))
    out.loc[idx, "discount_rate"] = rng.uniform(0.6, 0.9, len(idx))
    out.loc[idx, "is_injected_anomaly"] = 1
    return out


def evaluate_on_simulated(model: Pipeline, df: pd.DataFrame) -> dict:
    sim = inject_synthetic_anomalies(df)
    _, flag = score_anomalies(model, sim)
    y = sim["is_injected_anomaly"].to_numpy() == 1
    tp = int((flag & y).sum())
    return {"label": "SIMULATED anomalies injected into held-out rows - demonstration only, not real fraud",
            "injected": int(y.sum()), "flagged": int(flag.sum()), "true_positives": tp,
            "recall": round(float(tp / max(y.sum(), 1)), 3), "precision": round(float(tp / max(flag.sum(), 1)), 3)}
