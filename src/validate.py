from __future__ import annotations
import pandas as pd

def validate_daily_table(daily: pd.DataFrame) -> None:
    required = {"day", "daily_kwh", "daily_mean_temp", "dow", "month", "dom", "is_weekend"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if daily["daily_kwh"].isna().any():
        raise ValueError("daily_kwh has missing values.")
    if (daily["daily_kwh"] < 0).any():
        raise ValueError("daily_kwh has negative values.")

    daily = daily.sort_values("day")
    diffs = daily["day"].diff().dropna()
    if (diffs.dt.days < 1).any():
        raise ValueError("Found duplicate or out-of-order days.")

def validate_supervised_table(sup: pd.DataFrame, feature_cols: list[str], target_col: str) -> None:
    if sup[feature_cols + [target_col]].isna().any().any():
        raise ValueError("Supervised table contains missing values after feature creation.")
