"""Cleaning, feature engineering, validation and the sklearn preprocessing pipeline.

The SAME FeatureEngineer + ColumnTransformer live inside the saved Pipeline, so
training and prediction cannot drift apart. All statistics (medians, scaling,
category lists) are learned from the training split only -> no data leakage.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customerID"
TARGET = "Churn"

FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
NUMERIC_RAW = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_RAW = [c for c in FEATURE_COLUMNS if c not in NUMERIC_RAW + ["SeniorCitizen"]]
ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
NUMERIC_MODEL = NUMERIC_RAW + ["SeniorCitizen", "n_addon_services", "avg_monthly_spend"]

NUMERIC_LIMITS = {"tenure": (0, 120), "MonthlyCharges": (0, 500), "TotalCharges": (0, 100_000)}


# --------------------------------------------------------------------------- cleaning
def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy and attach a small report in `.attrs['report']`."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    report = {"rows_raw": len(df)}

    for col in df.columns[~df.dtypes.map(pd.api.types.is_numeric_dtype)]:
        df[col] = df[col].str.strip()

    report["duplicate_ids_removed"] = int(df.duplicated(subset=ID_COLUMN).sum())
    df = df.drop_duplicates(subset=ID_COLUMN)

    # TotalCharges arrives as text with blank strings for brand-new customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    report["total_charges_missing"] = int(df["TotalCharges"].isna().sum())
    # Tenure 0 means "not billed yet", so 0 is the true total (not a guess).
    new_customers = df["tenure"].eq(0) & df["TotalCharges"].isna()
    df.loc[new_customers, "TotalCharges"] = 0.0
    report["filled_zero_for_new_customers"] = int(new_customers.sum())

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    before = len(df)
    df = df.dropna(subset=[TARGET])
    report["rows_missing_target_dropped"] = before - len(df)
    df[TARGET] = df[TARGET].astype(int)

    report["rows_clean"] = len(df)
    df.attrs["report"] = report
    return df.reset_index(drop=True)


def detect_outliers(df: pd.DataFrame) -> dict:
    """IQR rule (1.5 x IQR) per numeric column plus impossible-value checks."""
    out = {}
    for col in NUMERIC_RAW:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        out[col] = {"lower_fence": round(float(lo), 2), "upper_fence": round(float(hi), 2),
                    "n_outliers": n, "n_negative": int((s < 0).sum())}
    return out


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """Reproducible, stratified train/test split (keeps the churn ratio equal)."""
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# --------------------------------------------------------------------------- features
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise features only (no statistics from other rows), so it cannot leak."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["n_addon_services"] = (X[ADDON_SERVICES] == "Yes").sum(axis=1)
        X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        return X


def build_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([("num", numeric, NUMERIC_MODEL), ("cat", categorical, CATEGORICAL_RAW)])
    return Pipeline([("features", FeatureEngineer()), ("prep", prep), ("model", estimator)])


# --------------------------------------------------------------------------- validation
def _to_flag(value) -> int:
    if isinstance(value, bool):
        return int(value)
    s = str(value).strip().lower()
    if s in {"1", "yes", "true", "y"}:
        return 1
    if s in {"0", "no", "false", "n"}:
        return 0
    raise ValueError("must be Yes/No (or 1/0)")


def validate_customer(payload: dict, categories: dict[str, list[str]]):
    """Validate one customer. Returns (clean_dict, errors{field: message}, notes[list])."""
    clean, errors, notes = {}, {}, []
    if not isinstance(payload, dict):
        return {}, {"_": "Request body must be a JSON object."}, notes

    def blank(v):
        return v is None or (isinstance(v, str) and not v.strip())

    for col in FEATURE_COLUMNS:
        v = payload.get(col)
        if col == "TotalCharges" and blank(v):
            continue  # estimated below once tenure and monthly charges are known
        if blank(v):
            errors[col] = "This field is required."
            continue
        try:
            if col == "SeniorCitizen":
                clean[col] = _to_flag(v)
            elif col in NUMERIC_RAW:
                num = float(v)
                if math.isnan(num) or math.isinf(num):
                    raise ValueError("must be a finite number")
                lo, hi = NUMERIC_LIMITS[col]
                if not lo <= num <= hi:
                    raise ValueError(f"must be between {lo} and {hi}")
                if col == "tenure":
                    if num != int(num):
                        raise ValueError("must be a whole number of months")
                    num = int(num)
                clean[col] = num
            else:
                lookup = {c.lower(): c for c in categories[col]}
                key = str(v).strip().lower()
                if key not in lookup:
                    raise ValueError("must be one of: " + ", ".join(categories[col]))
                clean[col] = lookup[key]
        except (ValueError, TypeError) as exc:
            msg = str(exc)
            errors[col] = msg if "must" in msg else "must be a number"

    if errors:
        return clean, errors, notes

    if "TotalCharges" not in clean:
        clean["TotalCharges"] = round(clean["tenure"] * clean["MonthlyCharges"], 2)
        notes.append("Total charges was blank, so it was estimated as tenure x monthly charges.")

    # Services that cannot exist without internet / phone service.
    if clean["InternetService"] == "No":
        for col in ADDON_SERVICES:
            clean[col] = "No internet service"
    else:
        for col in ADDON_SERVICES:
            if clean[col] == "No internet service":
                errors[col] = "Customer has internet service, so choose Yes or No."
    if clean["PhoneService"] == "No":
        clean["MultipleLines"] = "No phone service"
    elif clean["MultipleLines"] == "No phone service":
        errors["MultipleLines"] = "Customer has phone service, so choose Yes or No."

    return {c: clean[c] for c in FEATURE_COLUMNS}, errors, notes
