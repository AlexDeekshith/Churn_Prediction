import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from config import Config
from models.data_loader import load_raw
from models.preprocessing import (FEATURE_COLUMNS, TARGET, FeatureEngineer, build_pipeline,
                                  clean_dataframe, detect_outliers, split_data, validate_customer)

CATS = {"gender": ["Female", "Male"], "Partner": ["No", "Yes"], "Dependents": ["No", "Yes"],
        "PhoneService": ["No", "Yes"], "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        **{c: ["No", "No internet service", "Yes"] for c in
           ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]},
        "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": ["No", "Yes"],
        "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]}


@pytest.fixture(scope="module")
def clean():
    return clean_dataframe(load_raw(Config.DATA_PATH))


def test_cleaning_converts_types_and_fills_new_customers(clean):
    assert clean["TotalCharges"].dtype.kind == "f"
    assert clean["TotalCharges"].notna().all()           # blank strings were handled
    assert set(clean[TARGET].unique()) == {0, 1}
    assert clean["customerID"].is_unique


def test_cleaning_strips_whitespace_and_blank_totals():
    raw = pd.read_csv(Config.DATA_PATH).head(5)
    raw.loc[0, "TotalCharges"] = " "
    raw.loc[0, "tenure"] = 0
    out = clean_dataframe(raw)
    assert out.loc[0, "TotalCharges"] == 0.0


def test_split_is_reproducible_and_stratified(clean):
    a = split_data(clean, 0.2, 42)
    b = split_data(clean, 0.2, 42)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert abs(a[3].mean() - a[2].mean()) < 0.01        # similar churn rate in both splits
    assert set(a[0].index).isdisjoint(a[1].index)        # no row in both splits


def test_pipeline_output_is_numeric_and_handles_unseen_category(clean):
    X_train, X_test, y_train, _ = split_data(clean, 0.2, 42)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X_train, y_train)
    odd = X_test.head(3).copy()
    odd["PaymentMethod"] = "Crypto"                      # never seen in training
    assert np.isfinite(pipe.predict_proba(odd)).all()


def test_feature_engineering_is_row_wise(clean):
    X = clean[FEATURE_COLUMNS].head(10)
    full = FeatureEngineer().transform(X)
    single = FeatureEngineer().transform(X.iloc[[3]])
    assert full.iloc[3]["n_addon_services"] == single.iloc[0]["n_addon_services"]


def test_outlier_report_has_all_numeric_columns(clean):
    rep = detect_outliers(clean)
    assert set(rep) == {"tenure", "MonthlyCharges", "TotalCharges"}


def test_validation_accepts_valid_and_estimates_total():
    payload = {"gender": "female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No", "tenure": "10",
               "PhoneService": "No", "MultipleLines": "Yes", "InternetService": "No",
               "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
               "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
               "PaymentMethod": "Mailed check", "MonthlyCharges": "25.5", "TotalCharges": ""}
    clean, errors, notes = validate_customer(payload, CATS)
    assert not errors and notes
    assert clean["gender"] == "Female" and clean["TotalCharges"] == 255.0
    assert clean["MultipleLines"] == "No phone service" and clean["OnlineSecurity"] == "No internet service"


@pytest.mark.parametrize("field,value", [("tenure", -1), ("tenure", 2.5), ("tenure", "abc"), ("MonthlyCharges", 9999),
                                         ("MonthlyCharges", float("nan")), ("Contract", "Weekly"), ("gender", ""),
                                         ("SeniorCitizen", "maybe")])
def test_validation_rejects_bad_values(field, value):
    from tests.conftest import VALID
    _, errors, _ = validate_customer({**VALID, field: value}, CATS)
    assert field in errors
