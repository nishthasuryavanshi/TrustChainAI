import pandas as pd
import pytest
from src import config as cfg
from src import predict_risk as pr
from src.data_preprocessing import clean_training_data, validate_input_frame
from src.synthetic_data import generate_shipments
from src.train_model import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    df, _ = clean_training_data(generate_shipments(2500))
    metrics, test = train(df, "SYNTHETIC demo data", models_dir=d)
    return d, metrics, test


def test_no_leakage_columns_in_features():
    assert not set(cfg.FEATURES) & set(cfg.LEAKAGE_COLUMNS)
    assert not set(cfg.RAW_INPUTS) & set(cfg.LEAKAGE_COLUMNS)


def test_synthetic_is_deterministic():
    assert generate_shipments(50).equals(generate_shipments(50))


def test_training_saves_models_and_honest_metrics(trained):
    d, m, _ = trained
    assert pr.models_available(d)
    assert 0.5 < m["random_forest"]["roc_auc"] <= 1 and "logistic_regression_baseline" in m


def test_predictions_in_range(trained):
    d, _, test = trained
    inp = validate_input_frame(test.head(20)[["shipment_id"] + cfg.RAW_INPUTS])[0]
    out = pr.predict_frame(inp, d)
    assert out.risk_probability.between(0, 1).all() and set(out.risk_level) <= {"Low", "Medium", "High"}


def test_outcome_columns_are_ignored_in_uploads(trained):
    _, _, test = trained
    clean, errs, warns = validate_input_frame(test.head(5))
    assert clean is not None and "late_delivery" not in clean.columns and any("ignored" in w for w in warns)


def test_validation_rejects_bad_input():
    assert validate_input_frame(pd.DataFrame({"a": [1]}))[0] is None
    row = {c: "x" for c in cfg.CATEGORICAL}
    row.update(order_date="2026-01-01", scheduled_days=2, quantity=-3, unit_price=5, discount_rate=0.1)
    clean, errs, _ = validate_input_frame(pd.DataFrame([row]))
    assert clean is None and errs


def test_missing_model_raises_clean_error(tmp_path):
    with pytest.raises(pr.ModelsNotFoundError):
        pr.predict_frame(pd.DataFrame(), tmp_path)


def test_explain_returns_ranked_factors(trained):
    d, _, test = trained
    row = validate_input_frame(test.head(1)[["shipment_id"] + cfg.RAW_INPUTS])[0].iloc[0].to_dict()
    f = pr.explain_single(row, models_dir=d)
    assert len(f) == 5 and abs(f[0]["impact"]) >= abs(f[-1]["impact"])
