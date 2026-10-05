import json

import numpy as np
import pytest

from config import Config
from database.database import Database
from models.predict import ChurnPredictor, risk_level


def test_model_loads_and_outputs_probabilities(app):
    predictor = app.extensions["predictor"]
    assert predictor.metadata["selected_model"]
    probs = predictor.predict_proba(predictor.background)
    assert probs.shape[0] == len(predictor.background)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert probs.std() > 0.05                      # not a constant / hard-coded value


def test_probability_responds_to_inputs(app, valid_payload):
    predictor = app.extensions["predictor"]
    risky = predictor.predict_one(valid_payload, 0.3, 0.6, explain=False)["probability"]
    safe_payload = {**valid_payload, "tenure": 70, "Contract": "Two year", "TechSupport": "Yes",
                    "OnlineSecurity": "Yes", "TotalCharges": 6000}
    safe = predictor.predict_one(safe_payload, 0.3, 0.6, explain=False)["probability"]
    assert risky > safe + 0.3


def test_saved_model_loads_from_disk():
    p = ChurnPredictor(Config.MODEL_DIR)
    assert p.threshold == pytest.approx(p.metadata["threshold"])


@pytest.mark.parametrize("p,expected", [(0.05, "Low"), (0.30, "Medium"), (0.59, "Medium"), (0.60, "High"), (0.99, "High")])
def test_risk_levels_follow_configurable_thresholds(p, expected):
    assert risk_level(p, 0.30, 0.60) == expected
    assert risk_level(0.5, 0.2, 0.4) == "High"       # same probability, different business rule


def test_predict_endpoint_returns_full_result(client, valid_payload):
    r = client.post("/api/predict", json=valid_payload)
    body = r.get_json()
    assert r.status_code == 200
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] in ("Likely to Churn", "Likely to Stay")
    assert body["risk_level"] in ("Low", "Medium", "High")
    assert "increase_risk" in body["factors"] and "reduce_risk" in body["factors"]
    assert all(f["impact_points"] > 0 for f in body["factors"]["increase_risk"])


def test_predict_is_deterministic(client, valid_payload):
    a = client.post("/api/predict", json=valid_payload).get_json()["probability"]
    b = client.post("/api/predict", json=valid_payload).get_json()["probability"]
    assert a == b


def test_predict_rejects_invalid_input(client, valid_payload):
    r = client.post("/api/predict", json={**valid_payload, "tenure": -5, "Contract": "Weekly"})
    assert r.status_code == 400
    assert set(r.get_json()["details"]) == {"tenure", "Contract"}


def test_predict_rejects_missing_and_non_json_bodies(client):
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.post("/api/predict", data="not json", content_type="text/plain").status_code == 400
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 400


def test_predictions_are_saved_to_database(client, app, valid_payload):
    db = app.extensions["db"]
    before = db.prediction_count()
    client.post("/api/predict", json=valid_payload)
    assert db.prediction_count() == before + 1
    latest = db.recent_predictions(1)[0]
    assert latest["risk_category"] in ("Low", "Medium", "High") and latest["created_at"]


def test_dashboard_numbers_match_data_and_respond_to_filters(client, app):
    full = app.extensions["db"].load_customers()
    d = client.get("/api/dashboard").get_json()
    assert d["kpis"]["total_customers"] == len(full)
    assert d["kpis"]["churned_customers"] == int(full["churned"].sum())
    f = client.get("/api/dashboard?contract=Two year").get_json()
    assert f["kpis"]["total_customers"] == int((full["Contract"] == "Two year").sum())
    assert client.get("/api/dashboard?tenure_min=abc").status_code == 400


def test_customers_endpoint_search_sort_and_thresholds(client):
    d = client.get("/api/customers?per_page=10&sort=churn_probability&order=desc").get_json()
    probs = [c["churn_probability"] for c in d["customers"]]
    assert probs == sorted(probs, reverse=True)
    one = client.get("/api/customers?search=7590-VHVEG").get_json()
    assert one["total"] == 1
    assert client.get("/api/customers?low_max=0.8&high_min=0.2").status_code == 400
    strict = client.get("/api/customers?low_max=0.1&high_min=0.2").get_json()["risk_counts"]["High"]
    lenient = client.get("/api/customers?low_max=0.5&high_min=0.9").get_json()["risk_counts"]["High"]
    assert strict > lenient


def test_performance_and_insights_endpoints(client):
    m = client.get("/api/model-performance").get_json()
    assert {"Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"} <= set(m["models"])
    sel = m["models"][m["selected_model"]]
    cm = sel["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == m["n_test"]
    i = client.get("/api/insights").get_json()
    assert i["findings"] and i["actions"]


def test_all_pages_render(client):
    for path in ["/dashboard", "/analysis", "/predict", "/customers", "/performance", "/insights", "/about"]:
        assert client.get(path).status_code == 200
    assert client.get("/missing").status_code == 404
    assert client.get("/api/missing").get_json()["error"] == "Not found"


# ---------------------------------------------------------------- database
def test_database_round_trip(tmp_path):
    from models.data_loader import load_raw
    from models.preprocessing import clean_dataframe
    db = Database(f"sqlite:///{tmp_path / 'x.db'}")
    db.init_schema()
    df = clean_dataframe(load_raw(Config.DATA_PATH)).head(20)
    assert db.replace_customers(df, np.linspace(0, 1, 20)) == 20
    assert db.customer_count() == 20
    loaded = db.load_customers()
    assert loaded["churn_probability"].between(0, 1).all()
    pid = db.save_prediction({"tenure": 1}, "Likely to Stay", 0.12, "Low")
    assert pid == 1 and db.recent_predictions()[0]["input"] == {"tenure": 1}
    db.set_meta("k", "v")
    assert db.get_meta("k") == "v" and db.get_meta("missing") is None


def test_database_rejects_unsupported_backend():
    with pytest.raises(NotImplementedError):
        Database("postgresql://user:pw@host/db")
