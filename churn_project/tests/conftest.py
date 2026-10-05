import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402

VALID = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 90.0,
    "TotalCharges": 270.0,
}


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "test.db"
    return create_app({"TESTING": True, "DATABASE_URL": f"sqlite:///{db}"})


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_payload():
    return dict(VALID)
