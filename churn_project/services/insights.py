"""Turns the customer table into plain-English findings and suggested actions.

Findings are computed from the data (numbers shown are the calculated values).
Actions are rule-based suggestions that only appear when a finding crosses a
stated threshold; they are business judgement, not model output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(s: pd.Series) -> float:
    return float(s.mean()) if len(s) else float("nan")


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _group_rates(df, col, min_n=30):
    g = df.groupby(col)["churned"].agg(["mean", "count"])
    return g[g["count"] >= min_n].sort_values("mean", ascending=False)


def generate_insights(df: pd.DataFrame, high_min: float) -> dict:
    n, overall = len(df), _rate(df["churned"])
    findings, actions = [], []

    def add(fid, title, text, evidence):
        findings.append({"id": fid, "title": title, "text": text, "evidence": evidence})

    add("overall", "Overall churn",
        f"{int(df['churned'].sum()):,} of {n:,} customers churned ({_pct(overall)}).",
        {"churn_rate": round(overall, 4), "customers": n})

    # Contract
    cr = _group_rates(df, "Contract")
    top, bot = cr.index[0], cr.index[-1]
    ratio = cr.loc[top, "mean"] / cr.loc[bot, "mean"] if cr.loc[bot, "mean"] else float("inf")
    add("contract", "Contract type",
        f"{top} customers churn the most ({_pct(cr.loc[top, 'mean'])}, {int(cr.loc[top, 'count']):,} customers), "
        f"about {ratio:.1f}x the rate of {bot} customers ({_pct(cr.loc[bot, 'mean'])}).",
        {k: round(float(v), 4) for k, v in cr["mean"].items()})

    # Tenure
    corr_t = float(df["tenure"].corr(df["churned"]))
    med_c, med_s = df.loc[df.churned == 1, "tenure"].median(), df.loc[df.churned == 0, "tenure"].median()
    first_year = _rate(df.loc[df.tenure <= 12, "churned"])
    later = _rate(df.loc[df.tenure > 12, "churned"])
    direction = "Shorter tenure goes with higher churn" if corr_t < 0 else "Longer tenure goes with higher churn"
    add("tenure", "Customer tenure",
        f"{direction} (correlation {corr_t:+.2f}). Churned customers had a median tenure of {med_c:.0f} months "
        f"versus {med_s:.0f} for retained customers. Customers in their first 12 months churn at "
        f"{_pct(first_year)} compared with {_pct(later)} afterwards.",
        {"correlation": round(corr_t, 3), "first_12_months": round(first_year, 4), "after_12_months": round(later, 4)})

    # Monthly charges
    corr_m = float(df["MonthlyCharges"].corr(df["churned"]))
    q = pd.qcut(df["MonthlyCharges"], 4, labels=False, duplicates="drop")
    q_low, q_high = _rate(df.loc[q == q.min(), "churned"]), _rate(df.loc[q == q.max(), "churned"])
    mc_c, mc_s = df.loc[df.churned == 1, "MonthlyCharges"].mean(), df.loc[df.churned == 0, "MonthlyCharges"].mean()
    rel = "higher" if corr_m > 0 else "lower"
    add("charges", "Monthly charges",
        f"Churned customers pay ${mc_c:.2f} per month on average versus ${mc_s:.2f} for retained customers "
        f"({rel} charges go with churn; correlation {corr_m:+.2f}). The most expensive quarter of customers churns at "
        f"{_pct(q_high)}, the cheapest quarter at {_pct(q_low)}.",
        {"correlation": round(corr_m, 3), "top_quartile": round(q_high, 4), "bottom_quartile": round(q_low, 4)})

    # Internet service
    ir = _group_rates(df, "InternetService")
    nm = lambda v: "No internet service" if v == "No" else f"{v} internet"
    add("internet", "Internet service",
        f"{nm(ir.index[0])} customers have the highest churn ({_pct(ir.iloc[0]['mean'])}); {nm(ir.index[-1])} "
        f"customers have the lowest ({_pct(ir.iloc[-1]['mean'])}).", {k: round(float(v), 4) for k, v in ir["mean"].items()})

    # Payment method
    pr = _group_rates(df, "PaymentMethod")
    add("payment", "Payment method",
        f"{pr.index[0]} customers churn at {_pct(pr.iloc[0]['mean'])}, versus {_pct(pr.iloc[-1]['mean'])} for "
        f"{pr.index[-1]}.", {k: round(float(v), 4) for k, v in pr["mean"].items()})

    # Support services among internet customers
    net = df[df.InternetService != "No"]
    svc = []
    for col, name in [("TechSupport", "tech support"), ("OnlineSecurity", "online security"),
                      ("OnlineBackup", "online backup"), ("DeviceProtection", "device protection"),
                      ("StreamingTV", "streaming TV"), ("StreamingMovies", "streaming movies")]:
        yes, no = _rate(net.loc[net[col] == "Yes", "churned"]), _rate(net.loc[net[col] == "No", "churned"])
        svc.append((name, col, yes, no, no - yes))
    svc.sort(key=lambda r: r[4], reverse=True)
    best, worst = svc[0], svc[-1]
    add("services", "Add-on services (internet customers)",
        f"Customers without {best[0]} churn at {_pct(best[3])} versus {_pct(best[2])} with it. "
        f"The weakest association is {worst[0]}: {_pct(worst[2])} with versus {_pct(worst[3])} without.",
        {s[1]: {"with": round(s[2], 4), "without": round(s[3], 4)} for s in svc})

    # Demographics
    sr = df.groupby("SeniorCitizen")["churned"].mean()
    add("senior", "Senior citizens",
        f"Senior citizens churn at {_pct(sr.get(1, np.nan))} compared with {_pct(sr.get(0, np.nan))} for others.",
        {"senior": round(float(sr.get(1, np.nan)), 4), "non_senior": round(float(sr.get(0, np.nan)), 4)})
    pa, de = df.groupby("Partner")["churned"].mean(), df.groupby("Dependents")["churned"].mean()
    add("household", "Partners and dependents",
        f"Customers with a partner churn at {_pct(pa.get('Yes', np.nan))} (without: {_pct(pa.get('No', np.nan))}); "
        f"customers with dependents churn at {_pct(de.get('Yes', np.nan))} (without: {_pct(de.get('No', np.nan))}).",
        {"partner_yes": round(float(pa.get("Yes", np.nan)), 4), "partner_no": round(float(pa.get("No", np.nan)), 4),
         "dependents_yes": round(float(de.get("Yes", np.nan)), 4), "dependents_no": round(float(de.get("No", np.nan)), 4)})

    # Highest-risk combination
    combo = df.groupby(["Contract", "InternetService"])["churned"].agg(["mean", "count"])
    combo = combo[combo["count"] >= 100].sort_values("mean", ascending=False)
    if len(combo):
        k = combo.index[0]
        add("segment", "Highest-churn segment",
            f"Among segments with at least 100 customers, {k[0]} contracts with {k[1]} internet churn the most: "
            f"{_pct(combo.iloc[0]['mean'])} across {int(combo.iloc[0]['count']):,} customers.",
            {"contract": k[0], "internet": k[1], "rate": round(float(combo.iloc[0]["mean"]), 4),
             "customers": int(combo.iloc[0]["count"])})

    # ------------------------------------------------------------------ actions
    hi = df["churn_probability"] >= high_min

    def act(title, finding, why, target):
        actions.append({"title": title, "based_on": finding, "rationale": why, "target_customers": int(target)})

    if cr.loc[top, "mean"] >= 1.5 * overall:
        m = (df.Contract == top) & hi
        act("Contract upgrade offers", "contract",
            f"{top} customers churn at {_pct(cr.loc[top, 'mean'])} vs {_pct(overall)} overall. Offer a discount or "
            f"perk for moving to a longer contract.", m.sum())
    if first_year >= 1.2 * overall:
        m = (df.tenure <= 12) & hi
        act("Loyalty offers for new customers", "tenure",
            f"First-year churn is {_pct(first_year)}. A welcome check-in and early loyalty reward could target this window.",
            m.sum())
    if best[4] >= 0.05:
        m = (df.InternetService != "No") & (df[best[1]] == "No") & hi
        act(f"Incentivise {best[0]}", "services",
            f"Internet customers without {best[0]} churn {best[4] * 100:.1f} points more than those with it. "
            f"Offer a free trial or bundle (association, not proven cause).", m.sum())
    if pr.iloc[0]["mean"] - overall >= 0.03:
        m = (df.PaymentMethod == pr.index[0]) & hi
        act("Move customers to automatic payment", "payment",
            f"{pr.index[0]} customers churn at {_pct(pr.iloc[0]['mean'])}. Encourage automatic payment with a small credit.",
            m.sum())
    if q_high - q_low >= 0.05:
        m = (df.MonthlyCharges >= df["MonthlyCharges"].quantile(0.75)) & hi
        act("Personalised pricing plans", "charges",
            f"The highest-charge quarter churns at {_pct(q_high)} vs {_pct(q_low)} for the lowest. "
            f"Review plan fit and offer right-sized bundles.", m.sum())
    act("Targeted retention campaign for high-risk customers", "model",
        f"{int(hi.sum()):,} customers ({_pct(hi.mean())}) score at or above the high-risk threshold of "
        f"{high_min:.0%}. Prioritise them for outreach.", hi.sum())

    return {"findings": findings, "actions": actions, "based_on_customers": n}
