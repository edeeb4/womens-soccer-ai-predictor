import pandas as pd
from pathlib import Path

#file paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CLEAN_MATCHES_PATH = DATA_DIR / "clean_matches.csv"
TEAM_FEATURES_PATH = MODELS_DIR / "team_features.csv"
TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"


def calculate_team_features(matches_df):
    team_rows = []

    #flatten match data: one row for each team perspective per match
    for _, row in matches_df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_score = row["home_score"]
        away_score = row["away_score"]

        home_xg = row["home_xg"]
        away_xg = row["away_xg"]

        #home team perspective
        team_rows.append({
            "team": home_team,
            "league": row["league"],
            "source": row["source"],
            "goals_for": home_score,
            "goals_against": away_score,
            "xg_for": home_xg,
            "xg_against": away_xg,
            "win": 1 if home_score > away_score else 0,
            "draw": 1 if home_score == away_score else 0,
            "loss": 1 if home_score < away_score else 0,
            "match_played": 1
        })

        #away team perspective
        team_rows.append({
            "team": away_team,
            "league": row["league"],
            "source": row["source"],
            "goals_for": away_score,
            "goals_against": home_score,
            "xg_for": away_xg,
            "xg_against": home_xg,
            "win": 1 if away_score > home_score else 0,
            "draw": 1 if away_score == home_score else 0,
            "loss": 1 if away_score < home_score else 0,
            "match_played": 1
        })

    #group by team and calculate average features
    team_match_df = pd.DataFrame(team_rows)

    team_features = team_match_df.groupby("team").agg({
        "goals_for": "mean",
        "goals_against": "mean",
        "xg_for": "mean",
        "xg_against": "mean",
        "win": "mean",
        "draw": "mean",
        "loss": "mean",
        "match_played": "sum"
    }).reset_index()

    #rename columns for better clarity
    team_features = team_features.rename(columns={
        "goals_for": "avg_goals_for",
        "goals_against": "avg_goals_against",
        "xg_for": "avg_xg_for",
        "xg_against": "avg_xg_against",
        "win": "win_rate",
        "draw": "draw_rate",
        "loss": "loss_rate",
        "match_played": "matches_played"
    })

    #calculate additional features
    team_features["goal_difference_per_match"] = (
        team_features["avg_goals_for"] - team_features["avg_goals_against"]
    )

    team_features["xg_difference_per_match"] = (
        team_features["avg_xg_for"] - team_features["avg_xg_against"]
    )

    return team_features


def create_training_data(matches_df, team_features):
    df = matches_df.copy()

    #Add home team features
    df = df.merge(
        team_features,
        left_on="home_team",
        right_on="team",
        how="left"
    )

    df = df.rename(columns={
        "avg_goals_for": "home_avg_goals_for",
        "avg_goals_against": "home_avg_goals_against",
        "avg_xg_for": "home_avg_xg_for",
        "avg_xg_against": "home_avg_xg_against",
        "win_rate": "home_win_rate",
        "draw_rate": "home_draw_rate",
        "loss_rate": "home_loss_rate",
        "matches_played": "home_matches_played",
        "goal_difference_per_match": "home_goal_difference_per_match",
        "xg_difference_per_match": "home_xg_difference_per_match"
    })

    df = df.drop(columns=["team"])

    #Add away team features
    df = df.merge(
        team_features,
        left_on="away_team",
        right_on="team",
        how="left"
    )

    df = df.rename(columns={
        "avg_goals_for": "away_avg_goals_for",
        "avg_goals_against": "away_avg_goals_against",
        "avg_xg_for": "away_avg_xg_for",
        "avg_xg_against": "away_avg_xg_against",
        "win_rate": "away_win_rate",
        "draw_rate": "away_draw_rate",
        "loss_rate": "away_loss_rate",
        "matches_played": "away_matches_played",
        "goal_difference_per_match": "away_goal_difference_per_match",
        "xg_difference_per_match": "away_xg_difference_per_match"
    })

    df = df.drop(columns=["team"])

    #Comparison features
    df["goal_difference_gap"] = (
        df["home_goal_difference_per_match"] - df["away_goal_difference_per_match"]
    )

    df["xg_difference_gap"] = (
        df["home_xg_difference_per_match"] - df["away_xg_difference_per_match"]
    )

    df["win_rate_gap"] = df["home_win_rate"] - df["away_win_rate"]

    df["attack_strength_gap"] = df["home_avg_goals_for"] - df["away_avg_goals_for"]

    df["defensive_strength_gap"] = (
        df["away_avg_goals_against"] - df["home_avg_goals_against"]
    )

    return df


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    matches_df = pd.read_csv(CLEAN_MATCHES_PATH)

    print("Clean matches shape:", matches_df.shape)

    #Make sure xG columns are numeric.
    matches_df["home_xg"] = pd.to_numeric(matches_df["home_xg"], errors="coerce")
    matches_df["away_xg"] = pd.to_numeric(matches_df["away_xg"], errors="coerce")

    #Fill missing xG with average goals.
    matches_df["home_xg"] = matches_df["home_xg"].fillna(matches_df["home_score"])
    matches_df["away_xg"] = matches_df["away_xg"].fillna(matches_df["away_score"])

    team_features = calculate_team_features(matches_df)

    print("Team features shape:", team_features.shape)
    print(team_features.head())

    training_data = create_training_data(matches_df, team_features)

    print("Training data shape:", training_data.shape)
    print(training_data.head())

    team_features.to_csv(TEAM_FEATURES_PATH, index=False)
    training_data.to_csv(TRAINING_DATA_PATH, index=False)

    print(f"\nSaved team features to: {TEAM_FEATURES_PATH}")
    print(f"Saved training data to: {TRAINING_DATA_PATH}")


if __name__ == "__main__":
    main()