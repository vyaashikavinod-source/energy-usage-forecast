from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import PATHS
from src.data import download_dataset, load_raw, make_daily
from src.features import make_supervised, FEATURE_COLS, TARGET_COL
from src.validate import validate_daily_table, validate_supervised_table
from modeling.baselines import baseline_last_week, baseline_ma7

@dataclass
class FoldResult:
    fold: int
    mae: float
    rmse: float
    base_last_week_mae: float
    base_last_week_rmse: float
    base_ma7_mae: float
    base_ma7_rmse: float

def rmse(y_true, y_pred) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)

def make_folds(df: pd.DataFrame, n_folds: int = 5, test_size: int = 30):
    df = df.sort_values("day").reset_index(drop=True)
    total = len(df)
    start_test = total - (test_size * n_folds)
    folds = []
    for i in range(n_folds):
        test_start = start_test + i * test_size
        test_end = test_start + test_size
        train = df.iloc[:test_start].copy()
        test = df.iloc[test_start:test_end].copy()
        if len(test) == 0 or len(train) < 30:
            continue
        folds.append((i + 1, train, test))
    return folds

def main():
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    PATHS.dataset_dir.mkdir(parents=True, exist_ok=True)

    files = download_dataset(PATHS.dataset_dir)
    raw = load_raw(files.csv_path)
    daily = make_daily(raw)
    validate_daily_table(daily)

    sup = make_supervised(daily)
    validate_supervised_table(sup, FEATURE_COLS, TARGET_COL)

    folds = make_folds(sup, n_folds=4, test_size=30)
    if not folds:
        raise RuntimeError("Could not create folds. Dataset too small after preprocessing.")

    results: list[FoldResult] = []

    for fold_id, train_df, test_df in folds:
        X_train = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COL]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COL]

        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        pred_last_week = baseline_last_week(test_df)
        pred_ma = baseline_ma7(test_df)

        results.append(
            FoldResult(
                fold=fold_id,
                mae=float(mean_absolute_error(y_test, pred)),
                rmse=rmse(y_test, pred),
                base_last_week_mae=float(mean_absolute_error(y_test, pred_last_week)),
                base_last_week_rmse=rmse(y_test, pred_last_week),
                base_ma7_mae=float(mean_absolute_error(y_test, pred_ma)),
                base_ma7_rmse=rmse(y_test, pred_ma),
            )
        )

    mae_vals = np.array([r.mae for r in results], dtype=float)
    rmse_vals = np.array([r.rmse for r in results], dtype=float)

    out = {
        "folds": [r.__dict__ for r in results],
        "mean": {"mae": float(mae_vals.mean()), "rmse": float(rmse_vals.mean())},
        "std": {"mae": float(mae_vals.std(ddof=0)), "rmse": float(rmse_vals.std(ddof=0))},
    }

    (PATHS.reports_dir / "cv_metrics.json").write_text(json.dumps(out, indent=2))
    print("Saved reports/cv_metrics.json")

if __name__ == "__main__":
    main()
