from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from app.model_loader import (
    load_feature_importance,
    load_model,
    load_model_metrics,
    load_team_features,
)
from app.schemas import PredictionRequest
from app.utils import (
    FEATURE_COLUMNS,
    build_prediction_features,
    build_team_comparison,
    calculate_expected_goals,
    generate_key_factors,
)

#api confirguation and setup
app = FastAPI(
    title="Women's Soccer AI Match Predictor",
    description="Predicts women's soccer match outcomes using European league and NWSL data.",
    version="1.0.0",
)

#CORS Middleware setup
#allows cross-origin requests from web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#global resource loading
model = load_model()
team_features = load_team_features()
model_metrics = load_model_metrics()
feature_importance = load_feature_importance()


@app.get("/")
#API health chech and endpoint discovery
def root():
    return {
        "message": "Women's Soccer AI Match Predictor API is running",
        "endpoints": [
            "/teams",
            "/predict",
            "/model-metrics",
            "/feature-importance",
        ],
    }


@app.get("/teams")
# returns a sorted list of all unique teams avaiable in dataset
def get_teams():
    teams = sorted(team_features["team"].dropna().unique().tolist())

    return {
        "count": len(teams),
        "teams": teams,
    }


@app.get("/model-metrics")
#returns performance metrics of the trained model
def get_model_metrics():
    return model_metrics


@app.get("/feature-importance")
#returns ranked list of features that most influence the model's predictions
def get_feature_importance():
    if feature_importance.empty:
        return {"features": []}

    return {
        "features": feature_importance.to_dict(orient="records")
    }


@app.post("/predict")
#predicts the outcome of a match between two teams. calculates probalilities, expected goals, and qualitive insights
def predict_match(request: PredictionRequest):
    home_team = request.home_team
    away_team = request.away_team

    #teams MUST be distinct
    if home_team == away_team:
        raise HTTPException(
            status_code=400,
            detail="Home team and away team must be different."
        )

    #fetch and engineer features for the specific team matchup
    try:
        features, home, away = build_prediction_features(
            home_team,
            away_team,
            team_features,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    #reorder columns to match model training signature and handle NaNs
    features = features[FEATURE_COLUMNS].fillna(0)

    #perfom inference
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    #map probabilities to class labels and format for output
    class_labels = list(model.classes_)

    probability_output = {}

    for label, probability in zip(class_labels, probabilities):
        probability_output[label] = round(float(probability) * 100, 2)

    #calculate final response metrics
    confidence = round(float(np.max(probabilities)) * 100, 2)

    expected_goals = calculate_expected_goals(home, away)
    key_factors = generate_key_factors(home_team, away_team, home, away)
    team_comparison = build_team_comparison(home, away)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probability_output,
        "expected_goals": expected_goals,
        "key_factors": key_factors,
        "team_comparison": team_comparison,
    }