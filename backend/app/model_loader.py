import json
from pathlib import Path

import joblib
import pandas as pd

#file paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "match_predictor.pkl"
TEAM_FEATURES_PATH = MODELS_DIR / "team_features.csv"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_team_features():
    if not TEAM_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Team features file not found: {TEAM_FEATURES_PATH}")

    return pd.read_csv(TEAM_FEATURES_PATH)


def load_model_metrics():
    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(FEATURE_IMPORTANCE_PATH)