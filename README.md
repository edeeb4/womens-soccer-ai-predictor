# womens-soccer-ai-predictor
full stack machine learning app that predicts women's soccer match outcomes using European women's league and NWSL performance data.

## Live Links

**Live Demo:** https://womens-soccer-ai-predictor.vercel.app/  
**API Docs:** https://womens-soccer-ai-predictor.onrender.com/docs  
**Backend API:** https://womens-soccer-ai-predictor.onrender.com

> Note: The backend is currently hosted on a free Render instance, so the first request may take 30–60 seconds to wake up after inactivity.

## Project Overview

This project allows the users to select two women's soccer teams and generate a machine learning prediction for match outcome. 

This app predicts:
- Home win, draw, and away win probabilities
- Expected goals for each team
- Model confidence score
- Key factors affecting the prediction
- Team comparsion metrics
- Model performance metrics
- Top feature importance values


## Tech Stack

### Frontend
- React
- Vite
- JavaScript
- CSS
- Vercel

### Backend
- Python
- FastAPI
- Uvicorn
- Render

### Machine Learning
- Pandas
- NumPy
- scikit-learn
- Random Forest Classifier
- Joblib

### Data
- European women's league and NWSL performance data
- Goals, expected goals, win rates, goal differential, and defensive metrics


## Features
- Interactive team selection dashboard
- Win/draw/loss probability outputs
- Expected goals prediction
- Confidence score
- Key prediction factors
- Team comparison cards
- Model performance display
- Feature importance display
- Full-stack deployment with Vercel and Render



## Example Prediction Output

```json
{
  "home_team": "Barcelona",
  "away_team": "Real Madrid",
  "prediction": "HOME_WIN",
  "confidence": 87.31,
  "probabilities": {
    "AWAY_WIN": 6.5,
    "DRAW": 6.19,
    "HOME_WIN": 87.31
  },
  "expected_goals": {
    "home_team": 2.0,
    "away_team": 1.11
  }
}

