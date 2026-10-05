"""Pandas aggregations behind the dashboard, customer analysis and risk pages.
Every number is computed from the customers table at request time."""
from __future__ import annotations

import numpy as np
import pandas as pd

TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
CHARGE_LABELS = ["Under $35", "$35-$70", "$70-$90", "$90 and over"]

ANALYSIS_DIMENSIONS = {
    "Contract": "Contract", "InternetService": "Internet service", "PaymentMethod": "Payment method",
    "tenure_group": "Tenure group", "charge_band": "Monthly charge band", "TechSupport": "Tech support",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "PaperlessBilling": "Paperless billing",
    "gender": "Gender", "senior": "Senior citizen", "Partner": "Partner", "Dependents": "Dependents",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add display-only helper columns (never used for training)."""
    d = df.copy()
    d["tenure_group"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 10_000], labels=TENURE_LABELS).astype(str)
    d["charge_band"] = pd.cut(d["MonthlyCharges"], [-1, 35, 70, 90, 10_000], labels=CHARGE_LABELS).astype(str)
    d["senior"] = np.where(d["SeniorCitizen"] == 1, "Senior", "Non-senior")
    return d


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    if f.get("contract"):
        df = df[df["Contract"] == f["contract"]]
    if f.get("internet"):
        df = df[df["InternetService"] == f["internet"]]
    if f.get("gender"):
        df = df[df["gender"] == f["gender"]]
    if f.get("senior") in ("0", "1"):
        df = df[df["SeniorCitizen"] == int(f["senior"])]
    if f.get("tenure_min") is not None:
        df = df[df["tenure"] >= f["tenure_min"]]
    if f.get("tenure_max") is not None:
        df = df[df["tenure"] <= f["tenure_max"]]
    return df


def breakdown(df: pd.DataFrame, col: str, order: list | None = None) -> list[dict]:
    if df.empty:
        return []
    g = df.groupby(col)["churned"].agg(total="count", churned="sum").reset_index()
    g["rate"] = g["churned"] / g["total"]
    if order:
        g["_o"] = g[col].map({k: i for i, k in enumerate(order)})
        g = g.sort_values("_o").drop(columns="_o")
    else:
        g = g.sort_values("rate", ascending=False)
    rename = {"No": "No internet service"} if col == "InternetService" else {}
    return [{"label": rename.get(str(r[col]), str(r[col])), "total": int(r["total"]), "churned": int(r["churned"]),
             "rate": round(float(r["rate"]), 4)} for r in g.to_dict("records")]


def filter_options(df: pd.DataFrame) -> dict:
    return {"contract": sorted(df["Contract"].unique().tolist()),
            "internet": sorted(df["InternetService"].unique().tolist()),
            "gender": sorted(df["gender"].unique().tolist()),
            "tenure_max": int(df["tenure"].max())}


def dashboard_payload(full: pd.DataFrame, filters: dict, high_min: float) -> dict:
    df = prepare(apply_filters(full, filters))
    n = len(df)
    kpis = {
        "total_customers": n,
        "churned_customers": int(df["churned"].sum()),
        "churn_rate": round(float(df["churned"].mean()), 4) if n else 0.0,
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2) if n else 0.0,
        "avg_tenure": round(float(df["tenure"].mean()), 1) if n else 0.0,
        "high_risk_customers": int((df["churn_probability"] >= high_min).sum()),
    }
    demo = []
    for col, name in [("gender", "Gender"), ("senior", "Age group"), ("Partner", "Partner"),
                      ("Dependents", "Dependents")]:
        for row in breakdown(df, col):
            demo.append({**row, "group": name})
    return {
        "kpis": kpis,
        "distribution": {"stayed": n - kpis["churned_customers"], "churned": kpis["churned_customers"]},
        "by_contract": breakdown(df, "Contract"),
        "by_tenure": breakdown(df, "tenure_group", TENURE_LABELS),
        "by_charges": breakdown(df, "charge_band", CHARGE_LABELS),
        "by_internet": breakdown(df, "InternetService"),
        "by_payment": breakdown(df, "PaymentMethod"),
        "by_demographics": demo,
        "filters_applied": {k: v for k, v in filters.items() if v not in (None, "")},
    }


def _histogram(df: pd.DataFrame, col: str, bins: list) -> dict:
    labels = [f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)]
    out = {"labels": labels}
    for name, flag in (("stayed", 0), ("churned", 1)):
        counts, _ = np.histogram(df.loc[df["churned"] == flag, col], bins=bins)
        out[name] = counts.tolist()
    return out


def analysis_payload(full: pd.DataFrame, by: str, filters: dict) -> dict:
    df = prepare(apply_filters(full, filters))
    order = {"tenure_group": TENURE_LABELS, "charge_band": CHARGE_LABELS}.get(by)
    profile = {}
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        profile[col] = {
            name: {"mean": round(float(df.loc[df["churned"] == flag, col].mean()), 2),
                   "median": round(float(df.loc[df["churned"] == flag, col].median()), 2)}
            for name, flag in (("stayed", 0), ("churned", 1))
        } if len(df) and df["churned"].nunique() == 2 else {}
    return {
        "by": by, "by_label": ANALYSIS_DIMENSIONS[by], "segments": breakdown(df, by, order),
        "profile": profile, "n": len(df),
        "tenure_hist": _histogram(df, "tenure", list(range(0, 78, 6))) if len(df) else {},
        "charges_hist": _histogram(df, "MonthlyCharges", list(range(15, 135, 15))) if len(df) else {},
    }


def list_customers(full: pd.DataFrame, low_max: float, high_min: float, search: str = "",
                   risk: str = "", sort: str = "churn_probability", order: str = "desc",
                   page: int = 1, per_page: int = 25) -> dict:
    df = full.copy()
    df["risk"] = np.where(df["churn_probability"] >= high_min, "High",
                          np.where(df["churn_probability"] >= low_max, "Medium", "Low"))
    counts = df["risk"].value_counts().to_dict()
    if search:
        df = df[df["customer_id"].str.contains(search.strip(), case=False, regex=False)]
    if risk in ("Low", "Medium", "High"):
        df = df[df["risk"] == risk]
    sortable = {"churn_probability", "tenure", "MonthlyCharges", "customer_id", "Contract"}
    df = df.sort_values(sort if sort in sortable else "churn_probability", ascending=(order == "asc"))
    total = len(df)
    page = max(1, page)
    per_page = min(max(1, per_page), 100)
    rows = df.iloc[(page - 1) * per_page: page * per_page]
    cols = ["customer_id", "churn_probability", "risk", "Contract", "tenure", "MonthlyCharges",
            "InternetService", "PaymentMethod", "churned"]
    return {"total": total, "page": page, "per_page": per_page,
            "pages": max(1, -(-total // per_page)),
            "risk_counts": {k: int(counts.get(k, 0)) for k in ("High", "Medium", "Low")},
            "customers": rows[cols].round({"churn_probability": 4}).to_dict("records")}
