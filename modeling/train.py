from __future__ import annotations

import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import PATHS
from src.data import download_dataset, load_raw, make_daily
from src.features import make_supervised, FEATURE_COLS, TARGET_COL
from src.validate import validate_daily_table, validate_supervised_table
from modeling.baselines import baseline_last_week, baseline_ma7

def rmse(y_true, y_pred) -> float:
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)

def split_recent(df: pd.DataFrame, test_days: int = 30):
    df = df.sort_values("day").reset_index(drop=True)
    cutoff = df["day"].max() - pd.Timedelta(days=test_days)
    train = df[df["day"] <= cutoff].copy()
    test = df[df["day"] > cutoff].copy()
    return train, test

def main():
    PATHS.dataset_dir.mkdir(parents=True, exist_ok=True)
    PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    PATHS.images_dir.mkdir(parents=True, exist_ok=True)
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)

    files = download_dataset(PATHS.dataset_dir)
    raw = load_raw(files.csv_path)
    daily = make_daily(raw)
    validate_daily_table(daily)

    sup = make_supervised(daily)
    validate_supervised_table(sup, FEATURE_COLS, TARGET_COL)

    train_df, test_df = split_recent(sup, test_days=30)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    pred_last_week = baseline_last_week(test_df)
    pred_ma = baseline_ma7(test_df)

    metrics = {
        "test_window_days": 30,
        "model": {
            "name": "HistGradientBoostingRegressor",
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": rmse(y_test, pred),
        },
        "baseline_last_week": {
            "name": "same_day_last_week",
            "mae": float(mean_absolute_error(y_test, pred_last_week)),
            "rmse": rmse(y_test, pred_last_week),
        },
        "baseline_ma7": {
            "name": "last_7_day_average",
            "mae": float(mean_absolute_error(y_test, pred_ma)),
            "rmse": rmse(y_test, pred_ma),
        },
    }

    (PATHS.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "train_end_day": str(train_df["day"].max().date()),
    }
    joblib.dump(bundle, PATHS.artifacts_dir / "model.joblib")

    plt.figure()
    plt.plot(test_df["day"], y_test.to_numpy(), label="actual")
    plt.plot(test_df["day"], pred_last_week, label="baseline_last_week")
    plt.plot(test_df["day"], pred_ma, label="baseline_ma7")
    plt.plot(test_df["day"], pred, label="model")
    plt.xlabel("date")
    plt.ylabel("daily_kwh")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATHS.images_dir / "test_predictions.png", dpi=160)
    plt.close()

    print("Saved reports/metrics.json")
    print("Saved artifacts/model.joblib")
    print("Saved images/test_predictions.png")

if __name__ == "__main__":
    main()
