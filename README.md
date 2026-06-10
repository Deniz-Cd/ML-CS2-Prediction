# CS2 Kill Prediction

A statistics project that predicts how many kills a player will get in their next Counter-Strike 2 match, using their FACEIT match history and an XGBoost regression model.

## How it works

Rather than predicting raw kills, the model predicts **kills per round (KPR)** and scales the result back to kills using the match's round count. This removes the noise caused by matches of very different lengths.

The features describe the player's recent form and the context of the upcoming match — all computed strictly from matches *before* the one being predicted, so no target information leaks into training:

| Feature | Description |
|---|---|
| `elo_diff` | Absolute elo difference between the player and the enemy team |
| `time` | Time of day, sine-encoded (23:59 and 00:01 are "close") |
| `ewma_kpr`, `ewma_deaths`, `avg_adr` | Exponentially weighted averages of past kills/deaths per round and ADR |
| `kpr_slope`, `dpr_slope`, `adr_slope` | Trend (least-squares slope) of recent form over the last 3 matches |
| `ewma_winrate` | Smoothed winrate over the previous 5 matches |
| `streak_sign`, `streak_count` | Direction and length (capped at 5) of the current win/loss streak |
| `de_*` | One-hot encoding of the map |

The model is evaluated with **5-fold time-series cross-validation** (always validated on matches that come after its training data) using two kinds of metrics:

- **Within-k accuracy** — how often the prediction lands within ±1/2/3/5 kills of the actual result, compared against a baseline that always predicts the player's mean kills.
- **Above/below average classification** — accuracy, precision, recall and F1 for the simpler question "will the player perform above or below their average?"

## Pipeline

```
get_data.py  →  clean.py  →  main.py / tune.py  →  plot.py
 (fetch raw      (filter +      (train +            (visualize
  match data)     features)      evaluate)           results)
```

| Script | Role |
|---|---|
| `src/get_data.py` | Fetches a player's match history from the FACEIT Data API into `data/LARGE_DATA_<player>.xlsx` |
| `src/getmatches.py` | FACEIT API helper functions (matches, stats, elo, maps, results) |
| `src/clean.py` | Removes broken rows and outliers, builds the feature matrix via `compute_stats.py` |
| `src/compute_stats.py` | Feature engineering (EWMAs, slopes, streaks, encodings) |
| `src/functions.py` | Small numeric helpers (winrate, streaks, rolling slope) |
| `src/main.py` | Trains the tuned XGBoost model and reports cross-validated metrics |
| `src/tune.py` | Random search over XGBoost hyperparameters with a saved leaderboard |
| `src/plot.py` | Predicted vs. actual kills scatter plot with a ±5 kill margin |
| `src/demos.py` | (Experimental) bulk demo download + parsing for future within-match features |

## Setup

```bash
pip install -r requirements.txt
```

You need a free [FACEIT Data API key](https://developers.faceit.com/). Set it as an environment variable:

```bash
export FACEIT_API_KEY="your-key-here"   # Windows: set FACEIT_API_KEY=your-key-here
```

## Usage

Run everything from the repository root:

```bash
# 1. Fetch a player's match history (interactive: asks for name and match count)
python src/get_data.py

# 2. Clean the data and build the feature matrix
python src/clean.py <player_name>

# 3. Train and evaluate the model
python src/main.py <player_name>

# 4. Optional: re-run the hyperparameter search (slow — 2000 trials)
python src/tune.py <player_name>

# 5. Optional: plot predicted vs. actual kills
python src/plot.py
```

Outputs (predictions, feature importances, tuning leaderboard) are written to `results/`. Raw and computed datasets live in `data/`. Both folders are git-ignored — the data belongs to the individual players it was collected from.

## License

MIT — see [LICENSE](LICENSE).
