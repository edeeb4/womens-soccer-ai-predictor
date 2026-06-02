import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

#file and directory paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
MODEL_PATH = MODELS_DIR / "match_predictor.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"

#feature selection
#lists all the features used for training the model to predict match outcomes
FEATURE_COLUMNS = [
    "home_avg_goals_for",
    "home_avg_goals_against",
    "home_avg_xg_for",
    "home_avg_xg_against",
    "home_win_rate",
    "home_draw_rate",
    "home_loss_rate",
    "home_matches_played",
    "home_goal_difference_per_match",
    "home_xg_difference_per_match",

    "away_avg_goals_for",
    "away_avg_goals_against",
    "away_avg_xg_for",
    "away_avg_xg_against",
    "away_win_rate",
    "away_draw_rate",
    "away_loss_rate",
    "away_matches_played",
    "away_goal_difference_per_match",
    "away_xg_difference_per_match",

    "goal_difference_gap",
    "xg_difference_gap",
    "win_rate_gap",
    "attack_strength_gap",
    "defensive_strength_gap",
]


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    #data loading and prep
    df = pd.read_csv(TRAINING_DATA_PATH)

    print("Training data shape:", df.shape)
    print("Result distribution:")
    print(df["result"].value_counts())

    X = df[FEATURE_COLUMNS]
    y = df["result"]

    #fills missing feature value with 0 to help prevent model errors
    X = X.fillna(0)

    #train/test split with stratification to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #model initialization
    #the balanced weights will help handle the lower frequency of draws in the dataset
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8
    )

    #training and evaluation
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    #calculate standard performance metrics for classification and print results
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    #saves performance summary as JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "classes": list(model.classes_)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    #save feature ranking to identify most predicitve variables
    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    #serialize the trained model to file for later use in predictions
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved feature importance to: {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()