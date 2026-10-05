"""Load the saved pipeline and turn customer features into a prediction + explanation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.preprocessing import ADDON_SERVICES, FEATURE_COLUMNS, validate_customer

FEATURE_LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "tenure": "Tenure", "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines", "InternetService": "Internet service",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
    "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}


def risk_level(probability: float, low_max: float, high_min: float) -> str:
    """Business rule: thresholds come from configuration, not from the model."""
    if probability >= high_min:
        return "High"
    if probability >= low_max:
        return "Medium"
    return "Low"


def _display_value(feature: str, row: dict) -> str:
    v = row[feature]
    if feature == "tenure":
        return f"{v} month{'s' if v != 1 else ''}"
    if feature == "MonthlyCharges":
        return f"${v:,.2f}"
    if feature == "SeniorCitizen":
        return "Yes" if v else "No"
    return str(v)


class ChurnPredictor:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.pipeline = joblib.load(model_dir / "churn_model.joblib")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.background = pd.read_csv(model_dir / "background.csv")
        self.threshold = self.metadata["threshold"]

    @property
    def categories(self) -> dict:
        return self.metadata["categories"]

    def validate(self, payload):
        return validate_customer(payload, self.categories)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]

    # ------------------------------------------------------------------ explanation
    def _counterfactuals(self, row: dict) -> dict[str, pd.DataFrame]:
        """For each feature, rebuild the customer many times with that feature swapped for
        values seen in real training customers. Features that only make sense together
        (internet block, phone block, tenure+total charges) are swapped together so the
        counterfactual customers stay realistic."""
        bg = self.background
        n = len(bg)
        base = pd.DataFrame([row] * n)
        has_net, has_phone = row["InternetService"] != "No", row["PhoneService"] == "Yes"
        ratio = 1.0
        if row["tenure"] > 0 and row["MonthlyCharges"] > 0:
            ratio = row["TotalCharges"] / (row["tenure"] * row["MonthlyCharges"])
        out = {}
        for f in ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract",
                  "PaperlessBilling", "PaymentMethod"]:
            d = base.copy()
            d[f] = bg[f].values
            out[f] = d
        block = ["InternetService"] + ADDON_SERVICES
        d = base.copy()
        d[block] = bg[block].values
        out["InternetService"] = d
        d = base.copy()
        d[["PhoneService", "MultipleLines"]] = bg[["PhoneService", "MultipleLines"]].values
        out["PhoneService"] = d
        if has_net:
            sub = bg[bg["InternetService"] != "No"]
            for f in ADDON_SERVICES:
                d = pd.DataFrame([row] * len(sub))
                d[f] = sub[f].values
                out[f] = d
        if has_phone:
            sub = bg[bg["PhoneService"] == "Yes"]
            d = pd.DataFrame([row] * len(sub))
            d["MultipleLines"] = sub["MultipleLines"].values
            out["MultipleLines"] = d
        d = base.copy()
        d["tenure"] = bg["tenure"].values
        d["TotalCharges"] = (bg["tenure"] * row["MonthlyCharges"] * ratio).round(2).values
        out["tenure"] = d
        d = base.copy()
        d["MonthlyCharges"] = bg["MonthlyCharges"].values
        d["TotalCharges"] = (row["tenure"] * bg["MonthlyCharges"] * ratio).round(2).values
        out["MonthlyCharges"] = d
        return out

    def explain(self, row: dict, p_full: float, top_n: int = 4, min_points: float = 1.0) -> dict:
        """Impact of a feature = P(churn) for this customer minus the average P(churn) when
        that feature is replaced by typical customer values. Positive = pushes risk up."""
        variants = self._counterfactuals(row)
        frame = pd.concat(variants.values(), ignore_index=True)
        probs = self.predict_proba(frame)
        impacts, start = [], 0
        for feature, d in variants.items():
            mean_p = probs[start:start + len(d)].mean()
            start += len(d)
            impacts.append({
                "feature": feature, "label": FEATURE_LABELS[feature],
                "value": _display_value(feature, row),
                "impact_points": round(float(p_full - mean_p) * 100, 1),
            })
        up = sorted([i for i in impacts if i["impact_points"] >= min_points],
                    key=lambda i: -i["impact_points"])[:top_n]
        down = sorted([i for i in impacts if i["impact_points"] <= -min_points],
                      key=lambda i: i["impact_points"])[:top_n]
        return {"increase_risk": up, "reduce_risk": down}

    # ------------------------------------------------------------------ main entry
    def predict_one(self, row: dict, low_max: float, high_min: float, explain: bool = True) -> dict:
        p = float(self.predict_proba(pd.DataFrame([row]))[0])
        result = {
            "prediction": "Likely to Churn" if p >= self.threshold else "Likely to Stay",
            "churn": bool(p >= self.threshold),
            "probability": round(p, 4),
            "risk_level": risk_level(p, low_max, high_min),
            "thresholds": {"low_max": low_max, "high_min": high_min, "decision": self.threshold},
        }
        if explain:
            result["factors"] = self.explain(row, p)
        return result
