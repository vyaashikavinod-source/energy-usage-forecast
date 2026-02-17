from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"

@dataclass(frozen=True)
class DataFiles:
    zip_path: Path
    csv_path: Path

def download_dataset(target_dir: Path) -> DataFiles:
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / "uci_appliances_energy_prediction.zip"
    if not zip_path.exists():
        urlretrieve(UCI_ZIP_URL, zip_path)

    extract_dir = target_dir / "uci_raw"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    csv_candidates = list(extract_dir.rglob("energydata_complete.csv"))
    if not csv_candidates:
        raise FileNotFoundError("Could not find energydata_complete.csv after extraction.")
    return DataFiles(zip_path=zip_path, csv_path=csv_candidates[0])

def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def make_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["day"] = df["date"].dt.floor("D")

    temp_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    if not temp_cols:
        raise ValueError("Temperature columns T1..T9 not found.")

    df["temp_mean"] = df[temp_cols].mean(axis=1)

    daily = (
        df.groupby("day", as_index=False)
          .agg(
              daily_wh=("Appliances", "sum"),
              daily_mean_temp=("temp_mean", "mean"),
          )
    )

    daily["daily_kwh"] = daily["daily_wh"] / 1000.0
    daily = daily.drop(columns=["daily_wh"])

    daily["dow"] = daily["day"].dt.dayofweek
    daily["month"] = daily["day"].dt.month
    daily["dom"] = daily["day"].dt.day
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)

    return daily.sort_values("day").reset_index(drop=True)
