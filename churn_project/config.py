"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # --- secrets / deployment ------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
    DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"

    # --- storage -------------------------------------------------------------
    # sqlite:///path/to/file.db today; the Database class is the only place that
    # needs to change to support postgresql://user:pass@host/db later.
    DATABASE_URL = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'churn.db'}"
    )
    DATA_PATH = Path(os.environ.get("DATA_PATH", BASE_DIR / "data" / "customer_churn.csv"))
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models" / "saved_model"))

    # --- business rules (NOT model parameters) -------------------------------
    # probability < RISK_LOW_MAX            -> Low risk
    # RISK_LOW_MAX <= probability < RISK_HIGH_MIN -> Medium risk
    # probability >= RISK_HIGH_MIN          -> High risk
    RISK_LOW_MAX = float(os.environ.get("RISK_LOW_MAX", 0.30))
    RISK_HIGH_MIN = float(os.environ.get("RISK_HIGH_MIN", 0.60))

    # --- ML ------------------------------------------------------------------
    RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))
    TEST_SIZE = float(os.environ.get("TEST_SIZE", 0.2))
