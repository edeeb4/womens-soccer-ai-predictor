#helper file to inspect data
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

files = [
    "european_matches.csv",
    "european_team_stats.csv",
    "nwsl_team_stats.csv",
]

for file_name in files:
    file_path = DATA_DIR / file_name

    print("\n" + "="*80)
    print(f"FILE: {file_name}")
    print("="*80)

    if not file_path.exists():
        print(f"Missing File: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print("\nShape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())