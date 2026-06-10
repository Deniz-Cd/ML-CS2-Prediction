"""Train and evaluate the kill-prediction model.

An XGBoost regressor predicts a player's kills-per-round (KPR) for the
next match from the feature matrix built by clean.py. Predictions are
scaled back to kills with the match's round count and evaluated with
5-fold time-series cross-validation (no shuffling - the model is always
validated on matches that come after its training data).

The hyperparameters below come from the random search in tune.py.

Usage: python main.py [player_name]
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 5
RESULTS_DIR = "results"
MAP_FEATURES = [
    "de_mirage", "de_inferno", "de_dust2", "de_nuke",
    "de_train", "de_anubis", "de_ancient", "de_overpass",
]

# Best parameters found by the random search in tune.py.
XGB_PARAMS = dict(
    n_estimators=1087,
    learning_rate=0.05937510582556151,
    max_depth=8,
    colsample_bytree=0.6,
    min_child_weight=0.6749527504397954,
    subsample=0.7950500242521765,
    gamma=0.0013687903259109243,
    reg_alpha=0.0008704486696257267,
    reg_lambda=1.4805419766456205,
    n_jobs=-1,
    tree_method="hist",
    eval_metric="rmse",
)


def load_data(player):
    """Load a computed feature matrix and split it into X and y."""
    data = pd.read_excel(f"data/LARGE_DATA_{player}_COMPUTED.xlsx", sheet_name="Sheet1")
    y = data["kpr_to_predict"]
    X = data.drop(columns=["kpr_to_predict"]).apply(pd.to_numeric, errors="coerce")
    return X, y


def classification_metrics(kills_actual, kills_predicted):
    """Score the model as an above/below-average-kills classifier.

    A prediction counts as correct when it lands on the same side of the
    mean kill count as the actual result. This is the headline metric:
    'will the player perform above or below their average?'
    """
    threshold = np.mean(kills_actual)
    y_true = (kills_actual > threshold).astype(int)
    y_pred = (kills_predicted > threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    confusion = np.array([[tn, fp], [fn, tp]])

    return accuracy, precision, recall, f1, confusion


def within_k_accuracy(kills_actual, kills_predicted, k):
    """Fraction of predictions within +/- k kills of the actual value.

    Kills have very high natural variance, so exact prediction is not a
    realistic goal; being within a few kills is what matters in practice.
    """
    return np.mean(np.abs(kills_actual - kills_predicted) <= k)


def evaluate_fold(X_train, X_test, y_train, y_test, test_rounds):
    """Train on one fold and return its metrics plus the predictions."""
    regressor = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=30)
    regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predicted_kpr = regressor.predict(X_test)

    # Scale KPR back to kills using each match's actual round count.
    kills_actual = y_test.to_numpy() * test_rounds
    kills_predicted = predicted_kpr * test_rounds

    accuracy, precision, recall, f1, confusion = classification_metrics(
        kills_actual, kills_predicted
    )
    print(
        f"Accuracy: {accuracy*100:.2f}%, Recall: {recall*100:.2f}%, "
        f"Precision: {precision*100:.2f}%, F1: {f1*100:.2f}%"
    )
    print("Confusion Matrix:")
    print(confusion)

    metrics = {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "mape": np.mean(np.abs((y_test.to_numpy() - predicted_kpr) / y_test.to_numpy())) * 100,
        "mae": mean_absolute_error(kills_actual, kills_predicted),
        "rmse": np.sqrt(mean_squared_error(kills_actual, kills_predicted)),
        "r2": r2_score(kills_actual, kills_predicted),
    }

    # Within-k accuracies, compared against a baseline that always
    # predicts the mean kill count - the model has to beat this to be
    # worth anything.
    mean_kills = np.mean(kills_actual)
    for k in [1, 2, 3, 5]:
        metrics[f"acc_{k}"] = within_k_accuracy(kills_actual, kills_predicted, k) * 100
        metrics[f"base_{k}"] = within_k_accuracy(kills_actual, mean_kills, k) * 100
        print(f"Within {k} kills accuracy: {metrics[f'acc_{k}']:.2f}%")

    return regressor, metrics, kills_actual, kills_predicted


def print_feature_importances(regressor):
    """Print per-feature importance, with the map one-hots grouped together."""
    importances = regressor.feature_importances_
    percentages = importances / importances.sum() * 100
    names = regressor.feature_names_in_

    print("Feature importances (%):")
    map_total = 0.0
    for name, pct in zip(names, percentages):
        if name in MAP_FEATURES:
            map_total += pct
        else:
            print(f"{name}: {pct:.2f}%")
    print(f"MAP: {map_total:.2f}%")


def main():
    player = sys.argv[1] if len(sys.argv) > 1 else "Buco"
    X, y = load_data(player)

    print(f"Target std: {np.std(y):.4f}, mean: {np.mean(y):.4f}")

    splitter = TimeSeriesSplit(n_splits=N_SPLITS)
    totals = None

    for train_index, test_index in splitter.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Drop training rows with NaNs (early matches lack full history).
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]

        # 'rounds' is only needed to convert KPR back to kills - the
        # model must not train on it.
        test_rounds = X_test["rounds"].to_numpy()
        X_train = X_train.drop(columns=["rounds"])
        X_test = X_test.drop(columns=["rounds"])

        regressor, metrics, kills_actual, kills_predicted = evaluate_fold(
            X_train, X_test, y_train, y_test, test_rounds
        )

        if totals is None:
            totals = {key: 0.0 for key in metrics}
        for key, value in metrics.items():
            totals[key] += value

    averages = {key: value / N_SPLITS for key, value in totals.items()}
    print(
        f"\nAverage Accuracy: {averages['accuracy']:.2f}%, "
        f"Average Recall: {averages['recall']:.2f}%, "
        f"Average Precision: {averages['precision']:.2f}%, "
        f"Average F1: {averages['f1']:.2f}%"
    )
    for k in [5, 3, 2, 1]:
        print(f"Average {k}-Kill Accuracy: {averages[f'acc_{k}']:.2f}%")
        print(f"Average {k}-Kill Baseline (mean kills): {averages[f'base_{k}']:.2f}%")
    print(
        f"Average MAE: {averages['mae']:.4f}, RMSE: {averages['rmse']:.4f}, "
        f"R2: {averages['r2']:.4f}, MAPE: {averages['mape']:.2f}%"
    )

    # Save the last fold's predictions for plotting/inspection.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = X_test.copy()
    results["actual_kills"] = kills_actual
    results["predicted_kills"] = kills_predicted
    results.to_excel(f"{RESULTS_DIR}/predictions.xlsx", index=False)
    print(f"\nPredictions saved to {RESULTS_DIR}/predictions.xlsx")

    print_feature_importances(regressor)


if __name__ == "__main__":
    main()
