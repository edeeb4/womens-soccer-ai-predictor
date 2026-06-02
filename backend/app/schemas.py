from pydantic import BaseModel


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str


class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    prediction: str
    confidence: float
    probabilities: dict
    expected_goals: dict
    key_factors: list[str]
    team_comparison: dict