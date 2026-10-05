"""Fallback generator: builds a SYNTHETIC demo dataset with the Telco schema.

Only used when the real IBM Telco CSV cannot be downloaded. Churn is drawn from a
logistic model of contract, tenure, charges, support, etc., so patterns are
realistic - but every number is simulated and the app labels it as sample data.
"""
import numpy as np
import pandas as pd


def generate(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda opts, p=None: rng.choice(opts, n, p=p)
    df = pd.DataFrame({"customerID": [f"{rng.integers(1000, 9999)}-S{i:05d}" for i in range(n)]})
    df["gender"] = pick(["Female", "Male"])
    df["SeniorCitizen"] = rng.binomial(1, 0.16, n)
    df["Partner"] = pick(["Yes", "No"])
    df["Dependents"] = pick(["Yes", "No"], [0.3, 0.7])
    df["Contract"] = pick(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])
    df["tenure"] = np.where(df.Contract == "Month-to-month", rng.integers(1, 40, n), rng.integers(6, 73, n))
    df["PhoneService"] = pick(["Yes", "No"], [0.9, 0.1])
    df["MultipleLines"] = np.where(df.PhoneService == "No", "No phone service", pick(["Yes", "No"]))
    df["InternetService"] = pick(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])
    net = df.InternetService != "No"
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = np.where(net, pick(["Yes", "No"]), "No internet service")
    df["PaperlessBilling"] = pick(["Yes", "No"], [0.6, 0.4])
    df["PaymentMethod"] = pick(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    base = {"DSL": 45, "Fiber optic": 80, "No": 20}
    df["MonthlyCharges"] = (df.InternetService.map(base) + rng.normal(0, 8, n)
                            + 5 * (df[["StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)).clip(18, 120).round(2)
    df["TotalCharges"] = (df.MonthlyCharges * df.tenure * rng.uniform(0.95, 1.05, n)).round(2)
    z = (-1.0 + 1.3 * (df.Contract == "Month-to-month") - 1.0 * (df.Contract == "Two year")
         - 0.035 * df.tenure + 0.015 * (df.MonthlyCharges - 65) + 0.5 * (df.InternetService == "Fiber optic")
         + 0.4 * (df.PaymentMethod == "Electronic check") - 0.5 * (df.TechSupport == "Yes")
         - 0.4 * (df.OnlineSecurity == "Yes") + 0.3 * df.SeniorCitizen)
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return df


if __name__ == "__main__":
    generate().to_csv("customer_churn.csv", index=False)
