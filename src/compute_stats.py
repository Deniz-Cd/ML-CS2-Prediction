"""Build the model's feature matrix from raw match data.

Takes the raw per-match data produced by get_data.py and turns it into
the features the model trains on. All "history" features (EWMAs, slopes,
streaks, winrate) are computed strictly from matches *before* the one
being predicted, to avoid leaking the target.

The target is kpr_to_predict (kills per round); predictions are scaled
back to kills using the 'rounds' column, which is carried through the
matrix but dropped before training.
"""

import numpy as np
import pandas as pd

import functions as fn

MAPS = [
    "de_mirage", "de_inferno", "de_dust2", "de_nuke",
    "de_train", "de_anubis", "de_ancient", "de_overpass",
]


def compute(data):
    """Return the feature matrix DataFrame for raw match data.

    Expected columns in `data`: kills, enemy_team_elo, time, rounds,
    deaths, adr, map, won, player_elo. Rows are ordered newest first.
    """
    kpr = data["kills"] / data["rounds"]
    dpr = data["deaths"] / data["rounds"]

    # EWMAs of past performance. shift(1) excludes the current match,
    # since its stats are exactly what we are trying to predict.
    ewma_kpr = kpr.shift(1).ewm(alpha=0.1).mean()
    ewma_dpr = dpr.shift(1).ewm(alpha=0.1).mean()
    ewma_adr = data["adr"].shift(1).ewm(alpha=0.1).mean()

    # Short-term form: slope of the last 3 matches.
    kpr_slope = fn.rolling_slope(kpr.shift(1), window=3)
    dpr_slope = fn.rolling_slope(dpr.shift(1), window=3)
    adr_slope = fn.rolling_slope(data["adr"].shift(1), window=3)

    # Winrate over the previous 5 matches, smoothed.
    winrate = pd.DataFrame(fn.get_winrate(data["won"]))
    ewma_winrate = winrate.ewm(alpha=0.33).mean().values.ravel()

    # Win/loss streak going into the match, split into direction and
    # length (capped at 5 - beyond that a longer streak adds no signal).
    current_streak = fn.get_current_streak(data["won"])
    streak_sign = np.where(current_streak >= 0, 1, -1)
    streak_count = np.minimum(np.abs(current_streak), 5)

    features = pd.DataFrame({
        # Strength of the opposition relative to the player.
        "elo_diff": np.abs(data["enemy_team_elo"] - data["player_elo"]).astype(float),
        # Time of day is circular (23:59 is close to 00:01), so encode
        # it with a sine transform instead of raw seconds.
        "time": np.sin(2 * np.pi * data["time"] / 86400),
        "ewma_deaths": ewma_dpr,
        "avg_adr": ewma_adr,
        "ewma_winrate": ewma_winrate,
        "streak_sign": streak_sign,
        "streak_count": streak_count,
        "ewma_kpr": ewma_kpr,
        "kpr_slope": kpr_slope,
        "dpr_slope": dpr_slope,
        "adr_slope": adr_slope,
    })

    # One-hot encoding of the map; the model learns per-map importance itself.
    for map_name in MAPS:
        features[map_name] = (data["map"] == map_name).astype(float)

    features["rounds"] = data["rounds"].values
    features["kpr_to_predict"] = kpr

    return features
