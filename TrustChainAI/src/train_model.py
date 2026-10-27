"""Train Logistic Regression (baseline), Random Forest, and Isolation Forest.

Run:  python -m src.train_model [--data path/to/DataCoSupplyChainDataset.csv]
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config as cfg
from .anomaly_detection import build_anomaly_model, evaluate_on_simulated
from .data_preprocessing import load_dataset
from .feature_engineering import build_preprocessor, prepare


def _evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def _reference(train_df: pd.DataFrame) -> dict:
    """Typical value per raw input; used for the 'what drives this prediction' explanation."""
    ref = {c: train_df[c].mode().iloc[0] for c in cfg.CATEGORICAL}
    for c in ["scheduled_days", "quantity", "unit_price", "discount_rate"]:
        ref[c] = float(train_df[c].median())
    dates = pd.to_datetime(train_df["order_date"]).sort_values()
    ref["order_date"] = str(dates.iloc[len(dates) // 2].date())
    return ref


def train(df: pd.DataFrame, source: str, models_dir=cfg.MODELS_DIR, cleaning_report=None):
    """Train and save everything. Returns (metrics, held_out_test_df)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[cfg.TARGET], random_state=42)
    Xtr, ytr = prepare(train_df), train_df[cfg.TARGET]
    Xte, yte = prepare(test_df), test_df[cfg.TARGET]

    lr = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    rf = Pipeline([("prep", build_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample",
                                                  n_jobs=-1, random_state=42))])
    lr.fit(Xtr, ytr)
    rf.fit(Xtr, ytr)
    iso = build_anomaly_model().fit(Xtr)

    sample = Xte.sample(min(2000, len(Xte)), random_state=0)
    imp = permutation_importance(rf, sample, yte.loc[sample.index], n_repeats=3, random_state=0, scoring="roc_auc")
    importances = dict(sorted(zip(cfg.FEATURES, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": source, "model_version": cfg.MODEL_VERSION,
        "rows_train": int(len(train_df)), "rows_test": int(len(test_df)),
        "late_rate_test": round(float(yte.mean()), 4),
        "features_used": cfg.FEATURES, "excluded_leakage_columns": cfg.LEAKAGE_COLUMNS,
        "logistic_regression_baseline": _evaluate(lr, Xte, yte),
        "random_forest": _evaluate(rf, Xte, yte),
        "permutation_importance_auc_drop": importances,
        "anomaly_simulation": evaluate_on_simulated(iso, test_df),
        "cleaning_report": cleaning_report or {},
        "note": ("Metrics are computed on a held-out 20% split. If data_source is SYNTHETIC, the scores only "
                 "demonstrate the pipeline and must not be quoted as real-world performance."),
    }
    joblib.dump({"pipeline": rf, "reference": _reference(train_df), "version": cfg.MODEL_VERSION},
                models_dir / "risk_model.joblib")
    joblib.dump(lr, models_dir / "baseline_lr.joblib")
    joblib.dump(iso, models_dir / "anomaly_model.joblib")
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from . import predict_risk
    predict_risk.clear_cache()
    return metrics, test_df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Path to DataCoSupplyChainDataset.csv (optional)")
    args = ap.parse_args()
    data, src, rep = load_dataset(args.data)
    m, _ = train(data, src, cleaning_report=rep)
    print(f"Data source: {src}\nLogistic Regression :", m["logistic_regression_baseline"],
          "\nRandom Forest       :", m["random_forest"], "\nSaved to", cfg.MODELS_DIR)
