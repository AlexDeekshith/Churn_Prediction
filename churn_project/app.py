"""Flask application: pages + JSON API for the Customer Churn Analytics System.

Run:  python app.py        (or: flask --app app run)
"""
from __future__ import annotations

import json
import logging

from flask import Flask, jsonify, redirect, render_template, request, url_for

from config import Config
from database.database import Database
from models.data_loader import ensure_dataset, load_raw
from models.predict import ChurnPredictor
from models.preprocessing import clean_dataframe
from services import analytics, insights

log = logging.getLogger("app")


def create_app(overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config.update(overrides or {})
    cfg = app.config

    # --- model (train automatically on first run if no saved model exists) -----------
    if not (cfg["MODEL_DIR"] / "churn_model.joblib").exists():
        log.warning("No saved model found - training one now.")
        from models import train_model
        train_model.main()
    predictor = ChurnPredictor(cfg["MODEL_DIR"])

    # --- database (seed once, and again whenever the model is retrained) -------------
    db = Database(cfg["DATABASE_URL"])
    db.init_schema()
    stamp = predictor.metadata["trained_at"]
    if db.customer_count() == 0 or db.get_meta("model_trained_at") != stamp:
        ensure_dataset(cfg["DATA_PATH"])
        clean = clean_dataframe(load_raw(cfg["DATA_PATH"]))
        import pandas as pd
        scores = pd.read_csv(cfg["MODEL_DIR"] / "oof_scores.csv").set_index("customer_id")["churn_probability"]
        db.replace_customers(clean, clean["customerID"].map(scores).values)
        db.set_meta("model_trained_at", stamp)

    app.extensions["predictor"], app.extensions["db"] = predictor, db
    metrics = json.loads((cfg["MODEL_DIR"] / "metrics.json").read_text())

    # --- helpers -------------------------------------------------------------------
    def thresholds():
        low, high = cfg["RISK_LOW_MAX"], cfg["RISK_HIGH_MIN"]
        try:
            low = float(request.args.get("low_max", low))
            high = float(request.args.get("high_min", high))
        except ValueError:
            raise ValueError("Risk thresholds must be numbers.")
        if not 0 < low < high < 1:
            raise ValueError("Thresholds must satisfy 0 < low_max < high_min < 1.")
        return low, high

    def filters():
        a = request.args

        def num(k):
            v = a.get(k)
            if v in (None, ""):
                return None
            if not v.lstrip("-").isdigit():
                raise ValueError(f"{k} must be a whole number.")
            return int(v)
        return {"contract": a.get("contract", ""), "internet": a.get("internet", ""),
                "gender": a.get("gender", ""), "senior": a.get("senior", ""),
                "tenure_min": num("tenure_min"), "tenure_max": num("tenure_max")}

    @app.errorhandler(ValueError)
    def bad_value(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(404)
    def not_found(e):
        if request.path.startswith("/api/"):
            return jsonify(error="Not found"), 404
        return render_template("index.html", active="about", error_404=True), 404

    @app.errorhandler(500)
    def server_error(e):
        if request.path.startswith("/api/"):
            return jsonify(error="Internal server error"), 500
        return "Internal server error", 500

    @app.context_processor
    def inject():
        return {"model_name": predictor.metadata["selected_model"],
                "data_source": predictor.metadata["dataset_source"]}

    # --- pages -----------------------------------------------------------------------
    @app.get("/")
    def home():
        return redirect(url_for("dashboard"))

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html", active="dashboard")

    @app.get("/analysis")
    def analysis():
        return render_template("analysis.html", active="analysis", dimensions=analytics.ANALYSIS_DIMENSIONS)

    @app.get("/predict")
    def prediction():
        return render_template("prediction.html", active="prediction")

    @app.get("/customers")
    def customers():
        return render_template("customers.html", active="customers")

    @app.get("/performance")
    def performance():
        return render_template("performance.html", active="performance")

    @app.get("/insights")
    def insights_page():
        return render_template("insights.html", active="insights")

    @app.get("/about")
    def about():
        return render_template("index.html", active="about", error_404=False)

    # --- API -------------------------------------------------------------------------
    @app.get("/health")
    def health():
        return jsonify(status="ok", customers=db.customer_count(), model=predictor.metadata["selected_model"])

    @app.get("/api/meta")
    def api_meta():
        full = db.load_customers()
        return jsonify(categories=predictor.categories, thresholds={
            "low_max": cfg["RISK_LOW_MAX"], "high_min": cfg["RISK_HIGH_MIN"]},
            model=predictor.metadata["selected_model"], dataset_source=predictor.metadata["dataset_source"],
            filter_options=analytics.filter_options(full), trained_at=predictor.metadata["trained_at"],
            n_rows=predictor.metadata["n_rows"], n_test=predictor.metadata["n_test"],
            selection_rule=predictor.metadata["selection_rule"], cleaning=predictor.metadata["cleaning_report"],
            versions=predictor.metadata["versions"], predictions_stored=db.prediction_count())

    @app.get("/api/dashboard")
    def api_dashboard():
        _, high = thresholds()
        return jsonify(analytics.dashboard_payload(db.load_customers(), filters(), high))

    @app.get("/api/analysis")
    def api_analysis():
        by = request.args.get("by", "Contract")
        if by not in analytics.ANALYSIS_DIMENSIONS:
            raise ValueError("Unknown dimension: " + by)
        return jsonify(analytics.analysis_payload(db.load_customers(), by, filters()))

    @app.get("/api/customers")
    def api_customers():
        low, high = thresholds()
        a = request.args
        try:
            page, per_page = int(a.get("page", 1)), int(a.get("per_page", 25))
        except ValueError:
            raise ValueError("page and per_page must be whole numbers.")
        return jsonify(analytics.list_customers(
            db.load_customers(), low, high, a.get("search", ""), a.get("risk", ""),
            a.get("sort", "churn_probability"), a.get("order", "desc"), page, per_page) | {
            "thresholds": {"low_max": low, "high_min": high}})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(error="Send a JSON body with the customer fields."), 400
        clean, errors, notes = predictor.validate(payload)
        if errors:
            return jsonify(error="Some fields are invalid.", details=errors), 400
        low, high = cfg["RISK_LOW_MAX"], cfg["RISK_HIGH_MIN"]
        result = predictor.predict_one(clean, low, high)
        result["notes"] = notes
        result["id"] = db.save_prediction(clean, result["prediction"], result["probability"], result["risk_level"])
        return jsonify(result)

    @app.get("/api/predictions")
    def api_predictions():
        return jsonify(predictions=db.recent_predictions(10))

    @app.get("/api/model-performance")
    def api_performance():
        return jsonify(metrics | {"threshold": predictor.threshold, "n_test": predictor.metadata["n_test"],
                                  "selection_rule": predictor.metadata["selection_rule"]})

    @app.get("/api/insights")
    def api_insights():
        _, high = thresholds()
        return jsonify(insights.generate_insights(db.load_customers(), high))

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host="127.0.0.1", port=5000, debug=Config.DEBUG)
