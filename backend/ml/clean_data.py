#file will clean the data and save it to a new file
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


#file paths for input and output datasets
EUROPEAN_MATCHES_PATH = DATA_DIR / "european_matches.csv"
NWSL_MATCHES_PATH = DATA_DIR / "nwsl_team_stats.csv"
OUTPUT_PATH = DATA_DIR / "clean_matches.csv"


def add_result_column(df):
    #determine match outcome based on home/away results
    #returns the data with a new result column
    def get_result(row):
        if row["home_score"] > row["away_score"]:
            return "HOME_WIN"
        elif row["home_score"] < row["away_score"]:
            return "AWAY_WIN"
        else:
            return "DRAW"

    df["result"] = df.apply(get_result, axis=1)
    return df


#EUROPEAN MATCHES
def clean_european_matches():
    #renames columns, handles missing values, correct data types\
    df = pd.read_csv(EUROPEAN_MATCHES_PATH)
    print("Original European Matches Shape:", df.shape)

    #standardize column names
    column_mapping = {
        "Date": "date",
        "League_id": "league",
        "Home": "home_team",
        "Away": "away_team",
        "ScoreHome": "home_score",
        "ScoreAway": "away_score",
        "xGHome": "home_xg",
        "xGAway": "away_xg",
    }
    df = df.rename(columns=column_mapping)

    #filter for relevant features
    useful_columns =[
        "date",
        "league",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_xg",
        "away_xg",
    ]
    df = df[useful_columns].copy()
    df["source"] = "european_leagues"

    #remove rows that are missing critical information
    df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"])

    #drop rows where the scores could not be converted
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")

    df = df.dropna(subset=["home_score", "away_score"])

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df = add_result_column(df)

    print("Cleaned European matches shape:", df.shape)

    return df


#nwsl stats
def clean_nwsl_matches():
    df = pd.read_csv(NWSL_MATCHES_PATH)

    print("Original NWSL shape:", df.shape)

    #Keep only rows where team was the home team to prevent counting each match twice
    df = df[df["venue"] == "Home"].copy()

    clean_df = pd.DataFrame()

    clean_df["date"] = df["date"]
    clean_df["league"] = "NWSL"
    clean_df["home_team"] = df["team"]
    clean_df["away_team"] = df["opp"]
    clean_df["home_score"] = df["gf"]
    clean_df["away_score"] = df["ga"]
    clean_df["home_xg"] = df["tm_exp_xG"]
    clean_df["away_xg"] = df["opp_exp_xG"]
    clean_df["source"] = "nwsl"

    clean_df = clean_df.dropna(
        subset=[
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )

    clean_df["home_score"] = pd.to_numeric(clean_df["home_score"], errors="coerce")
    clean_df["away_score"] = pd.to_numeric(clean_df["away_score"], errors="coerce")
    clean_df["home_xg"] = pd.to_numeric(clean_df["home_xg"], errors="coerce")
    clean_df["away_xg"] = pd.to_numeric(clean_df["away_xg"], errors="coerce")

    clean_df = clean_df.dropna(subset=["home_score", "away_score"])

    clean_df["home_score"] = clean_df["home_score"].astype(int)
    clean_df["away_score"] = clean_df["away_score"].astype(int)

    clean_df = add_result_column(clean_df)

    print("Cleaned NWSL shape:", clean_df.shape)

    return clean_df


def main():
    european_df = clean_european_matches()
    nwsl_df = clean_nwsl_matches()

    #combine the two datasets into one clean dataset and sort by chronological order
    clean_df = pd.concat([european_df, nwsl_df], ignore_index=True)
    clean_df = clean_df.sort_values(by=["source", "date"], ascending=True)

    #export final cleaned CSV
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal clean dataset shape:", clean_df.shape)
    print(clean_df.head())
    print(f"\nSaved clean dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()