from __future__ import annotations
import pandas as pd

CORE_FEATURE_COLS = [
    "dow",
    "month",
    "dom",
    "is_weekend",
    "lag_1",
    "lag_7",
    "ma7",
    "ma14",
]

TEMP_FEATURE_COLS = [
    "daily_mean_temp",
    "temp_lag_1",
    "temp_ma7",
]

TARGET_COL = "daily_kwh"

def make_supervised(daily: pd.DataFrame, use_temp: bool = False) -> pd.DataFrame:
    if "day" not in daily.columns:
        raise ValueError("Expected a 'day' column")
    if TARGET_COL not in daily.columns:
        raise ValueError(f"Expected a '{TARGET_COL}' column")

    df = daily.copy()
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values("day").reset_index(drop=True)

    df["dow"] = df["day"].dt.dayofweek.astype(int)
    df["month"] = df["day"].dt.month.astype(int)
    df["dom"] = df["day"].dt.day.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["lag_1"] = df[TARGET_COL].shift(1)
    df["lag_7"] = df[TARGET_COL].shift(7)
    df["ma7"] = df[TARGET_COL].shift(1).rolling(7).mean()
    df["ma14"] = df[TARGET_COL].shift(1).rolling(14).mean()

    if use_temp:
        if "daily_mean_temp" not in df.columns:
            raise ValueError("use_temp=True but 'daily_mean_temp' is missing")
        df["temp_lag_1"] = df["daily_mean_temp"].shift(1)
        df["temp_ma7"] = df["daily_mean_temp"].shift(1).rolling(7).mean()

    df = df.dropna().reset_index(drop=True)
    return df
