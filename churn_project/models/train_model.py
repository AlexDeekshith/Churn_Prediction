"""Train, compare, select and save the churn model.

Run from the project root:   python -m models.train_model

Selection uses 5-fold cross-validation on the TRAINING split only. The held-out
test set is used once, to report honest metrics for every candidate.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from config import Config
from models.data_loader import ensure_dataset, load_raw
from models.preprocessing import (FEATURE_COLUMNS, CATEGORICAL_RAW, ID_COLUMN, TARGET,
                                  build_pipeline, clean_dataframe, detect_outliers, split_data)

log = logging.getLogger("train")



def candidate_models(seed: int) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30,
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_depth=12,
                                                n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        max_depth=3, subsample=0.8, random_state=seed),
    }
    try:  # optional extra model if the library is installed
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          subsample=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1)
    except ImportError:
        pass
    return models


def fit_pipeline(pipe, X, y):
    return clone(pipe).fit(X, y)


def best_threshold(y, p) -> float:
    """Cutoff that maximises F1 on out-of-fold training predictions. Probabilities stay
    calibrated (no class re-weighting); only the churn/stay LABEL cutoff is tuned."""
    grid = np.arange(0.10, 0.71, 0.01)
    return round(float(grid[int(np.argmax([f1_score(y, (p >= t).astype(int)) for t in grid]))]), 2)


def cross_validate(pipe, X, y, seed, folds=5):
    """Out-of-fold probabilities from stratified K-fold (every row scored by a model that never saw it)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        oof[va] = fit_pipeline(pipe, X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def evaluate(model, X_test, y_test, threshold) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)
    idx = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()},
        "classification_report": classification_report(
            y_test, pred, target_names=["Stayed", "Churned"], output_dict=True),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config
    source = ensure_dataset(cfg.DATA_PATH)
    raw = load_raw(cfg.DATA_PATH)
    df = clean_dataframe(raw)
    log.info("Dataset: %s | %s rows | churn rate %.1f%%", source, len(df), 100 * df[TARGET].mean())

    X_train, X_test, y_train, y_test = split_data(df, cfg.TEST_SIZE, cfg.RANDOM_STATE)
    log.info("Train %d rows / test %d rows", len(X_train), len(X_test))

    results, fitted = {}, {}
    for name, est in candidate_models(cfg.RANDOM_STATE).items():
        pipe = build_pipeline(est)
        oof_train = cross_validate(pipe, X_train, y_train, cfg.RANDOM_STATE)
        thr = best_threshold(y_train, oof_train)
        cv_pred = (oof_train >= thr).astype(int)
        model = fit_pipeline(pipe, X_train, y_train)
        res = evaluate(model, X_test, y_test, thr)
        res.update(threshold=thr, cv_f1=float(f1_score(y_train, cv_pred)), cv_recall=float(recall_score(y_train, cv_pred)))
        results[name], fitted[name] = res, model
        log.info("%-20s cutoff %.2f CV-F1 %.3f | test acc %.3f prec %.3f rec %.3f F1 %.3f AUC %.3f", name, thr, res["cv_f1"],
                 res["accuracy"], res["precision"], res["recall"], res["f1"], res["roc_auc"])

    # Select on cross-validated F1 (training data only); recall breaks ties.
    best = max(results, key=lambda n: (round(results[n]["cv_f1"], 3), results[n]["cv_recall"]))
    log.info("Selected model: %s", best)
    model = fitted[best]

    # Calibration check on the test set: does "40-50%" really mean about 45% churn?
    cal = pd.DataFrame({"p": model.predict_proba(X_test)[:, 1], "y": y_test.values})
    cal["bin"] = pd.cut(cal["p"], [0, .1, .2, .3, .4, .5, .6, .8, 1.0], include_lowest=True)
    calibration = [{"range": f"{int(round(b.left * 100))}-{int(round(b.right * 100))}%", "n": int(len(g)),
                    "predicted": round(float(g["p"].mean()), 3), "actual": round(float(g["y"].mean()), 3)}
                   for b, g in cal.groupby("bin", observed=True)]

    # Permutation importance on the held-out test set, per RAW feature (model-agnostic).
    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                  random_state=cfg.RANDOM_STATE, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": float(m), "std": float(s)}
         for f, m, s in zip(FEATURE_COLUMNS, perm.importances_mean, perm.importances_std)),
        key=lambda d: d["importance"], reverse=True)

    # Honest scores for every customer: out-of-fold predictions from the selected model.
    X_all, y_all = df[FEATURE_COLUMNS], df[TARGET]
    oof = cross_validate(build_pipeline(candidate_models(cfg.RANDOM_STATE)[best]),
                         X_all, y_all, cfg.RANDOM_STATE)

    # Refit the selected model on the training split (what was evaluated) and save it.
    out = cfg.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "churn_model.joblib")
    pd.DataFrame({"customer_id": df[ID_COLUMN], "churn_probability": oof.round(6)}).to_csv(
        out / "oof_scores.csv", index=False)
    X_train.sample(150, random_state=cfg.RANDOM_STATE).to_csv(out / "background.csv", index=False)

    metadata = {
        "selected_model": best, "threshold": results[best]["threshold"], "dataset_source": source,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "churn_rate": float(y_all.mean()), "random_state": cfg.RANDOM_STATE,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_RAW},
        "cleaning_report": df.attrs.get("report", {}),
        "outliers": detect_outliers(df),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "selection_rule": "highest 5-fold cross-validated F1 on the training split, at a cutoff tuned on that split (recall breaks ties)",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"selected_model": best, "models": results, "feature_importance": importance,
         "calibration": calibration}, indent=2))
    log.info("Saved model, metrics and metadata to %s", out)
    return metadata


if __name__ == "__main__":
    main()
