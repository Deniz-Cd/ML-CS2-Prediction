"""Clean a raw dataset and compute its feature matrix.

Reads data/LARGE_DATA_<player>.xlsx, drops broken rows and outliers,
runs the feature engineering from compute_stats.py and writes
data/LARGE_DATA_<player>_COMPUTED.xlsx ready for training.

Usage: python clean.py <player_name>
"""

import sys

import pandas as pd

import compute_stats as cs

player = sys.argv[1] if len(sys.argv) > 1 else "Buco"

data = pd.read_excel(f"data/LARGE_DATA_{player}.xlsx", sheet_name="Sheet1")

# Zeroes in these columns mean the API request failed for that match.
for col in ["kills", "deaths", "rounds", "time", "enemy_team_elo", "adr"]:
    data = data[data[col] != 0]

# Trim extreme games: kill counts outside [7, 25] are rare outliers
# that hurt training much more than they help.
data = data[(data["kills"] >= 7) & (data["kills"] <= 25)]
data = data.reset_index(drop=True)

feature_matrix = cs.compute(data)

# The first rows have NaNs from the rolling/EWMA computations.
feature_matrix = feature_matrix.iloc[3:, :]

output_path = f"data/LARGE_DATA_{player}_COMPUTED.xlsx"
feature_matrix.to_excel(output_path, index=False)
print(f"Feature matrix saved to {output_path}")
