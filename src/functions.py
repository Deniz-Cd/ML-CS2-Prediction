"""Small numeric helpers used during feature engineering."""

import numpy as np


def standardize(x):
    """Return x standardized to zero mean and unit variance (z-scores)."""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def get_winrate(results):
    """Return the winrate (%) over the previous 5 matches for each match.

    `results` is a series of 0/1 match outcomes ordered from most recent
    to oldest, as returned by the FACEIT history endpoint.
    """
    winrate = np.zeros(len(results))

    for i in range(len(results)):
        if i >= 5:
            winrate[i - 5] = np.sum(results[i - 5 : i]) / 5 * 100
        # Near the end of the series there are fewer than 5 past matches.
        if len(results) - i < 5:
            winrate[i] = np.sum(results[i:]) / (5 - (len(results) - i)) * 100

    return winrate


def get_current_streak(results):
    """Return the win/loss streak going into each match.

    Positive values are win streaks, negative values are loss streaks.
    The streak for match i is computed from the matches played before it
    (rows i+1 onwards, since rows are ordered newest first).
    """
    streak = np.zeros(len(results))

    for i in range(len(results) - 1):
        last_result = results[i + 1]
        for j in range(i + 1, len(results)):
            if results[j] != last_result:
                break
            streak[i] += 1 if last_result == 1 else -1

    return streak


def rolling_slope(series, window=5):
    """Return the least-squares slope of each trailing `window` of the series.

    Used to capture short-term form trends (e.g. is the player's KPR
    going up or down over the last few matches). The first `window - 1`
    entries are 0 since there is not enough history.
    """
    slopes = []

    for i in range(len(series)):
        if i < window - 1:
            slopes.append(0)
            continue
        y = series[i - window + 1 : i + 1]
        x = np.arange(window)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        slopes.append(slope)

    return np.array(slopes)
