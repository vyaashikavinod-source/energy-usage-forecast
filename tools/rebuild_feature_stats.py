from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "processed" / "daily_energy.csv"
OUT_PATH = ROOT / "app" / "model_assets" / "feature_stats.json"

df = pd.read_csv(DATA_PATH)
df["day"] = pd.to_datetime(df["day"])
df = df.sort_values("day").reset_index(drop=True)

feature_order = [
    "daily_mean_temp",
    "dow",
    "month",
    "dom",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "roll7_mean",
    "roll7_std",
    "roll14_mean",
]

kwh = df["daily_kwh"].astype(float).to_numpy()
temp = df["daily_mean_temp"].astype(float).to_numpy()
day = pd.to_datetime(df["day"])

dow = day.dt.dayofweek.astype(float).to_numpy()
month = day.dt.month.astype(float).to_numpy()
dom = day.dt.day.astype(float).to_numpy()
is_weekend = (dow >= 5).astype(float)

def lag(arr, n):
    out = np.full_like(arr, np.nan, dtype=float)
    out[n:] = arr[:-n]
    return out

def roll_mean(arr, w):
    return pd.Series(arr).rolling(w).mean().to_numpy(dtype=float)

def roll_std(arr, w):
    return pd.Series(arr).rolling(w).std(ddof=0).to_numpy(dtype=float)

feat = {
    "daily_mean_temp": temp,
    "dow": dow,
    "month": month,
    "dom": dom,
    "is_weekend": is_weekend,
    "lag_1": lag(kwh, 1),
    "lag_2": lag(kwh, 2),
    "lag_7": lag(kwh, 7),
    "lag_14": lag(kwh, 14),
    "roll7_mean": roll_mean(kwh, 7),
    "roll7_std": roll_std(kwh, 7),
    "roll14_mean": roll_mean(kwh, 14),
}

X = np.column_stack([feat[f] for f in feature_order]).astype(float)

mask = ~np.isnan(X).any(axis=1)
X = X[mask]

means = X.mean(axis=0)
stds = X.std(axis=0)
stds = np.where(stds == 0, 1.0, stds)

out = {
    "feature_order": feature_order,
    "means": means.tolist(),
    "stds": stds.tolist(),
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote: {OUT_PATH}")
print(f"Features: {len(feature_order)}")