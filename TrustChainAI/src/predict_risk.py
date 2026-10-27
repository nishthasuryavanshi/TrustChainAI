"""Load saved models and score shipments."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config as cfg
from .anomaly_detection import score_anomalies
from .feature_engineering import prepare

_FILES = ("risk_model.joblib", "anomaly_model.joblib", "metrics.json")


class ModelsNotFoundError(RuntimeError):
    """Raised when trained model files are missing or unreadable."""


def models_available(models_dir=cfg.MODELS_DIR) -> bool:
    return all((Path(models_dir) / f).exists() for f in _FILES)


@lru_cache(maxsize=4)
def _load(models_dir_str: str):
    d = Path(models_dir_str)
    try:
        return joblib.load(d / "risk_model.joblib"), joblib.load(d / "anomaly_model.joblib")
    except Exception as exc:  # missing / corrupt / incompatible files
        raise ModelsNotFoundError(
            f"Could not load models from {d}. Run `python -m src.bootstrap`. ({exc})") from exc


def clear_cache():
    _load.cache_clear()


def risk_level(p: float) -> str:
    return "High" if p >= cfg.RISK_BANDS[1] else "Medium" if p >= cfg.RISK_BANDS[0] else "Low"


def predict_frame(df: pd.DataFrame, models_dir=cfg.MODELS_DIR) -> pd.DataFrame:
    """Add risk_probability, risk_level, anomaly_score, is_anomaly to a validated frame."""
    bundle, iso = _load(str(models_dir))
    out = df.copy()
    out["risk_probability"] = bundle["pipeline"].predict_proba(prepare(out))[:, 1].round(4)
    out["risk_level"] = out["risk_probability"].map(risk_level)
    score, flag = score_anomalies(iso, out)
    out["anomaly_score"] = score.round(4)
    out["is_anomaly"] = flag.astype(int)
    return out


def explain_single(row: dict, top_k: int = 5, models_dir=cfg.MODELS_DIR) -> list:
    """Approximate factor contributions: how much the risk changes if one input is replaced
    by its typical (reference) value. Positive = pushes risk up. Not a causal claim."""
    bundle, _ = _load(str(models_dir))
    ref = bundle["reference"]
    base = pd.DataFrame([{c: row[c] for c in cfg.RAW_INPUTS}])
    variants = [base]
    for c in cfg.RAW_INPUTS:
        v = base.copy()
        v[c] = ref[c]
        variants.append(v)
    probs = bundle["pipeline"].predict_proba(prepare(pd.concat(variants, ignore_index=True)))[:, 1]
    res = [{"feature": c, "value": row[c], "typical": ref[c], "impact": float(probs[0] - probs[i + 1])}
           for i, c in enumerate(cfg.RAW_INPUTS)]
    return sorted(res, key=lambda r: -abs(r["impact"]))[:top_k]
