from __future__ import annotations
import pandas as pd

FEATURE_COLS = [
    "daily_mean_temp",
    "temp_lag_1",
    "temp_ma7",
    "dow",
    "month",
    "dom",
    "is_weekend",
    "lag_1",
    "lag_7",
    "ma7",
    "ma14",
]
TARGET_COL = "daily_kwh"

def make_supervised(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("day").reset_index(drop=True).copy()

    df["lag_1"] = df["daily_kwh"].shift(1)
    df["lag_7"] = df["daily_kwh"].shift(7)
    df["ma7"] = df["daily_kwh"].shift(1).rolling(7).mean()
    df["ma14"] = df["daily_kwh"].shift(1).rolling(14).mean()

    df["temp_lag_1"] = df["daily_mean_temp"].shift(1)
    df["temp_ma7"] = df["daily_mean_temp"].shift(1).rolling(7).mean()

    df = df.dropna().reset_index(drop=True)
    return df
