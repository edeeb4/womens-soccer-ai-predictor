import pandas as pd

#list of features that are required for the trained model to make predictions
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

#extract past features for two teams and build a feature vector for the model to make a prediction on the match outcome
def build_prediction_features(home_team, away_team, team_features):
    #look up past stats for both teams
    home_data = team_features[team_features["team"] == home_team]
    away_data = team_features[team_features["team"] == away_team]

    #make sure both teams are found in the dataset if not presents an error
    if home_data.empty:
        raise ValueError(f"Home team not found: {home_team}")

    if away_data.empty:
        raise ValueError(f"Away team not found: {away_team}")

    home = home_data.iloc[0]
    away = away_data.iloc[0]

    #combines raw stas and calculated competitive gaps
    features = {
        "home_avg_goals_for": home["avg_goals_for"],
        "home_avg_goals_against": home["avg_goals_against"],
        "home_avg_xg_for": home["avg_xg_for"],
        "home_avg_xg_against": home["avg_xg_against"],
        "home_win_rate": home["win_rate"],
        "home_draw_rate": home["draw_rate"],
        "home_loss_rate": home["loss_rate"],
        "home_matches_played": home["matches_played"],
        "home_goal_difference_per_match": home["goal_difference_per_match"],
        "home_xg_difference_per_match": home["xg_difference_per_match"],

        "away_avg_goals_for": away["avg_goals_for"],
        "away_avg_goals_against": away["avg_goals_against"],
        "away_avg_xg_for": away["avg_xg_for"],
        "away_avg_xg_against": away["avg_xg_against"],
        "away_win_rate": away["win_rate"],
        "away_draw_rate": away["draw_rate"],
        "away_loss_rate": away["loss_rate"],
        "away_matches_played": away["matches_played"],
        "away_goal_difference_per_match": away["goal_difference_per_match"],
        "away_xg_difference_per_match": away["xg_difference_per_match"],

        #compares offensive/defensive relative strengths
        "goal_difference_gap": home["goal_difference_per_match"] - away["goal_difference_per_match"],
        "xg_difference_gap": home["xg_difference_per_match"] - away["xg_difference_per_match"],
        "win_rate_gap": home["win_rate"] - away["win_rate"],
        "attack_strength_gap": home["avg_goals_for"] - away["avg_goals_for"],
        "defensive_strength_gap": away["avg_goals_against"] - home["avg_goals_against"],
    }

    return pd.DataFrame([features]), home, away

#estimates expected goals for a specific mathchup based on averaging the team offensive strength and opponent defensive vulnerability
def calculate_expected_goals(home, away):
    home_expected_goals = (home["avg_xg_for"] + away["avg_xg_against"]) / 2
    away_expected_goals = (away["avg_xg_for"] + home["avg_xg_against"]) / 2

    return {
        "home_team": round(float(home_expected_goals), 2),
        "away_team": round(float(away_expected_goals), 2),
    }


#compares team stats to generate human readable insights about the key factors that may influence the match outcome based on historical performance data
def generate_key_factors(home_team, away_team, home, away):
    key_factors = []

    #win rate insight
    if home["win_rate"] > away["win_rate"]:
        key_factors.append(f"{home_team} has a higher overall win rate.")
    elif home["win_rate"] < away["win_rate"]:
        key_factors.append(f"{away_team} has a higher overall win rate.")
    else:
        key_factors.append("Both teams have a similar overall win rate.")

    #offensive threat insight
    if home["avg_xg_for"] > away["avg_xg_for"]:
        key_factors.append(f"{home_team} creates more expected goals per match.")
    elif home["avg_xg_for"] < away["avg_xg_for"]:
        key_factors.append(f"{away_team} creates more expected goals per match.")
    else:
        key_factors.append("Both teams generate similar expected goals.")

    #defensive vulnerability insight
    if home["avg_goals_against"] < away["avg_goals_against"]:
        key_factors.append(f"{home_team} has allowed fewer goals per match.")
    elif home["avg_goals_against"] > away["avg_goals_against"]:
        key_factors.append(f"{away_team} has allowed fewer goals per match.")
    else:
        key_factors.append("Both teams have allowed a similar number of goals.")

    #goal difference insight
    if home["goal_difference_per_match"] > away["goal_difference_per_match"]:
        key_factors.append(f"{home_team} has the stronger goal difference.")
    elif home["goal_difference_per_match"] < away["goal_difference_per_match"]:
        key_factors.append(f"{away_team} has the stronger goal difference.")
    else:
        key_factors.append("Both teams have a similar goal difference.")

    return key_factors


#formats raw team performance metrics into a clean dictionary for UI display
def build_team_comparison(home, away):
    return {
        "home": {
            "avg_goals_for": round(float(home["avg_goals_for"]), 2),
            "avg_goals_against": round(float(home["avg_goals_against"]), 2),
            "avg_xg_for": round(float(home["avg_xg_for"]), 2),
            "avg_xg_against": round(float(home["avg_xg_against"]), 2),
            "win_rate": round(float(home["win_rate"]) * 100, 2),
            "matches_played": int(home["matches_played"]),
        },
        "away": {
            "avg_goals_for": round(float(away["avg_goals_for"]), 2),
            "avg_goals_against": round(float(away["avg_goals_against"]), 2),
            "avg_xg_for": round(float(away["avg_xg_for"]), 2),
            "avg_xg_against": round(float(away["avg_xg_against"]), 2),
            "win_rate": round(float(away["win_rate"]) * 100, 2),
            "matches_played": int(away["matches_played"]),
        },
    }