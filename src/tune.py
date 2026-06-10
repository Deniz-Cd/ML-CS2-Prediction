"""Hyperparameter random search for the kill-prediction model.

Samples N_ITER random XGBoost configurations and scores each one with
5-fold time-series cross-validation, optimizing the above/below-average
classification accuracy (see main.py). Results are saved as a sortable
leaderboard, and a final model is fit and evaluated with the best
parameters found.

Usage: python tune.py [player_name]
"""

import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

PLAYER = sys.argv[1] if len(sys.argv) > 1 else "Cicciks"
DATA_PATH = f"data/LARGE_DATA_{PLAYER}_COMPUTED.xlsx"
SHEET = "Sheet1"
ROUNDS_COL = "rounds"
RESULTS_DIR = "results"

RANDOM_STATE = 0
N_SPLITS = 5
N_ITER = 2000
EARLY_STOP = 50
PRIMARY_OBJECTIVE = "clf_accuracy"   # maximize accuracy of above-mean-kills classification
VERBOSE_FIT = False

def sample_params(rng: np.random.RandomState) -> Dict[str, Any]:
    return {
        "n_estimators": int(rng.randint(300, 1600)),
        "learning_rate": float(10 ** rng.uniform(-2.2, -0.7)),
        "max_depth": int(rng.randint(3, 9)),
        "min_child_weight": float(10 ** rng.uniform(-1, 1.2)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "gamma": float(10 ** rng.uniform(-3, 0)),
        "reg_alpha": float(10 ** rng.uniform(-5, 1)),
        "reg_lambda": float(10 ** rng.uniform(-3, 2)),
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }

def mean_wo_nan(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan

def clf_metrics_from_kills(kills_actual: np.ndarray, kills_pred: np.ndarray) -> Dict[str, float]:
    thr = np.mean(kills_actual)
    y_true = (kills_actual > thr).astype(int)
    y_pred = (kills_pred   > thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)

def within_kills_acc(kills_actual: np.ndarray, kills_pred: np.ndarray, k: int) -> float:
    return float(np.mean(np.abs(kills_actual - kills_pred) <= k))

@dataclass
class FoldReport:
    mae_kills: float
    rmse_kills: float
    r2_kills: float
    acc_5kills: float
    acc_1kill: float
    acc_2kills: float
    acc_3kills: float
    clf_accuracy: float
    clf_precision: float
    clf_recall: float
    clf_f1: float
    mape_kpr: float

@dataclass
class TrialReport:
    params: Dict[str, Any]
    mean_metrics: FoldReport
    time_sec: float

def evaluate_params_once(params: Dict[str, Any],
                         X_df: pd.DataFrame,
                         y_series: pd.Series,
                         rounds_col: str,
                         n_splits: int) -> TrialReport:
    t0 = time.time()
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[FoldReport] = []

    for tr_idx, va_idx in tss.split(X_df):
        X_tr, X_va = X_df.iloc[tr_idx].copy(), X_df.iloc[va_idx].copy()
        y_tr, y_va = y_series.iloc[tr_idx].copy(), y_series.iloc[va_idx].copy()

        rounds_tr = X_tr[rounds_col].to_numpy()
        rounds_va = X_va[rounds_col].to_numpy()

        X_tr = X_tr.drop(columns=[rounds_col])
        X_va = X_va.drop(columns=[rounds_col])

        train_mask = ~X_tr.isna().any(axis=1)
        X_tr_clean = X_tr.loc[train_mask]
        y_tr_clean = y_tr.loc[X_tr_clean.index]

        reg = xgb.XGBRegressor(**params, early_stopping_rounds=EARLY_STOP)
        reg.fit(X_tr_clean, y_tr_clean, eval_set=[(X_va, y_va)], verbose=VERBOSE_FIT)

        y_pred_kpr = reg.predict(X_va)
        kills_actual = y_va.to_numpy() * rounds_va
        kills_pred   = y_pred_kpr * rounds_va

        mae_kills  = float(np.mean(np.abs(kills_actual - kills_pred)))
        rmse_kills = float(np.sqrt(np.mean((kills_actual - kills_pred) ** 2)))
        denom = np.sum((kills_actual - np.mean(kills_actual))**2)
        r2_kills   = float(1 - np.sum((kills_actual - kills_pred)**2) / (denom if denom != 0 else 1e-12))

        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.abs((y_va.to_numpy() - y_pred_kpr) /
                          np.where(y_va.to_numpy() == 0, np.nan, y_va.to_numpy())) * 100
        mape_kpr = float(np.nanmean(mape))

        acc_1 = within_kills_acc(kills_actual, kills_pred, 1)
        acc_2 = within_kills_acc(kills_actual, kills_pred, 2)
        acc_3 = within_kills_acc(kills_actual, kills_pred, 3)
        acc_5 = within_kills_acc(kills_actual, kills_pred, 5)

        clf = clf_metrics_from_kills(kills_actual, kills_pred)

        fold_metrics.append(FoldReport(
            mae_kills=mae_kills,
            rmse_kills=rmse_kills,
            r2_kills=r2_kills,
            acc_5kills=acc_5,
            acc_1kill=acc_1,
            acc_2kills=acc_2,
            acc_3kills=acc_3,
            clf_accuracy=clf["accuracy"],
            clf_precision=clf["precision"],
            clf_recall=clf["recall"],
            clf_f1=clf["f1"],
            mape_kpr=mape_kpr
        ))

    def avg(field: str) -> float:
        return mean_wo_nan([getattr(fm, field) for fm in fold_metrics])

    mean_report = FoldReport(
        mae_kills=avg("mae_kills"),
        rmse_kills=avg("rmse_kills"),
        r2_kills=avg("r2_kills"),
        acc_5kills=avg("acc_5kills"),
        acc_1kill=avg("acc_1kill"),
        acc_2kills=avg("acc_2kills"),
        acc_3kills=avg("acc_3kills"),
        clf_accuracy=avg("clf_accuracy"),
        clf_precision=avg("clf_precision"),
        clf_recall=avg("clf_recall"),
        clf_f1=avg("clf_f1"),
        mape_kpr=avg("mape_kpr")
    )

    return TrialReport(params=params, mean_metrics=mean_report, time_sec=time.time() - t0)

def run_random_search(X_df: pd.DataFrame,
                      y_series: pd.Series,
                      rounds_col: str,
                      n_splits: int,
                      n_iter: int,
                      random_state: int,
                      primary_objective: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.RandomState(random_state)
    trials: List[TrialReport] = []

    print(f"Random search: {n_iter} trials, {n_splits}-fold TimeSeriesSplit")
    for i in range(1, n_iter + 1):
        params = sample_params(rng)
        tr = evaluate_params_once(params, X_df, y_series, rounds_col, n_splits)
        trials.append(tr)
        print(f"[{i:02d}/{n_iter}] {primary_objective}={getattr(tr.mean_metrics, primary_objective):.4f} "
              f"time={tr.time_sec:.1f}s  "
              f"params={{n_estimators:{params['n_estimators']}, lr:{params['learning_rate']:.4f}, "
              f"max_depth:{params['max_depth']}, mcw:{params['min_child_weight']:.3f}}}")

    rows = []
    for tr in trials:
        row = {
            **{f"param_{k}": v for k, v in tr.params.items()},
            **{f"metric_{k}": v for k, v in asdict(tr.mean_metrics).items()},
            "time_sec": tr.time_sec
        }
        rows.append(row)
    leaderboard = pd.DataFrame(rows)

    ascending = False  # higher is better for clf_accuracy
    leaderboard = leaderboard.sort_values(by=f"metric_{primary_objective}", ascending=ascending).reset_index(drop=True)
    best_params_cols = [c for c in leaderboard.columns if c.startswith("param_")]
    best_params = {c.replace("param_", ""): leaderboard.loc[0, c] for c in best_params_cols}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    leaderboard_path = f"{RESULTS_DIR}/xgb_random_search_leaderboard.xlsx"
    leaderboard.to_excel(leaderboard_path, index=False)
    print(f"Leaderboard saved to: {leaderboard_path}")

    return leaderboard, best_params

def main():
    FeatureMatrix = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    kpr_to_predict = FeatureMatrix["kpr_to_predict"]
    FeatureMatrix = FeatureMatrix.drop(columns=["kpr_to_predict"])

    FeatureMatrix_df = pd.DataFrame(FeatureMatrix).copy()
    for c in FeatureMatrix_df.columns:
        FeatureMatrix_df[c] = pd.to_numeric(FeatureMatrix_df[c], errors="coerce")
    if ROUNDS_COL not in FeatureMatrix_df.columns:
        raise ValueError(f"'{ROUNDS_COL}' column missing from FeatureMatrix.")

    leaderboard, best_params = run_random_search(
        X_df=FeatureMatrix_df,
        y_series=kpr_to_predict,
        rounds_col=ROUNDS_COL,
        n_splits=N_SPLITS,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        primary_objective=PRIMARY_OBJECTIVE
    )

    print("\nBest params (max clf_accuracy):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    rounds_all = FeatureMatrix_df[ROUNDS_COL].to_numpy()
    X_all = FeatureMatrix_df.drop(columns=[ROUNDS_COL])
    train_mask_all = ~X_all.isna().any(axis=1)
    X_all_clean = X_all.loc[train_mask_all]
    y_all_clean = kpr_to_predict.loc[X_all_clean.index]

    final_params = dict(best_params)
    final_params.update({
        "early_stopping_rounds": EARLY_STOP,
        "eval_metric": "rmse",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    })
    final_model = xgb.XGBRegressor(**final_params)

    split_idx = int(len(X_all_clean) * 0.9)
    X_tr_all, X_va_all = X_all_clean.iloc[:split_idx], X_all_clean.iloc[split_idx:]
    y_tr_all, y_va_all = y_all_clean.iloc[:split_idx], y_all_clean.iloc[split_idx:]

    final_model.fit(X_tr_all, y_tr_all, eval_set=[(X_va_all, y_va_all)], verbose=VERBOSE_FIT)

    y_va_pred_kpr = final_model.predict(X_va_all)
    rounds_va = FeatureMatrix_df.loc[X_va_all.index, ROUNDS_COL].to_numpy()
    kills_va_actual = y_va_all.to_numpy() * rounds_va
    kills_va_pred = y_va_pred_kpr * rounds_va

    preds_df = X_va_all.copy()
    preds_df["actual_kills"] = kills_va_actual
    preds_df["predicted_kills"] = kills_va_pred

    os.makedirs(RESULTS_DIR, exist_ok=True)
    preds_path = f"{RESULTS_DIR}/predictions_best_params.xlsx"
    preds_df.to_excel(preds_path, index=False)
    print(f"Validation predictions saved to: {preds_path}")

    importances = final_model.feature_importances_
    percentages = (importances / (importances.sum() + 1e-12)) * 100
    feat_names = getattr(final_model, "feature_names_in_", X_va_all.columns.to_numpy())

    imp_df = pd.DataFrame({"feature": feat_names, "importance_%": percentages}).sort_values("importance_%", ascending=False)
    imp_path = f"{RESULTS_DIR}/feature_importances_best_params.xlsx"
    imp_df.to_excel(imp_path, index=False)
    print(f"Feature importances saved to: {imp_path}")

    map_cols = [c for c in feat_names if str(c).startswith("de_")]
    map_imp = float(imp_df.loc[imp_df["feature"].isin(map_cols), "importance_%"].sum())
    print(f"MAP group importance: {map_imp:.2f}%")


if __name__ == "__main__":
    main()
