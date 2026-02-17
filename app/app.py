from __future__ import annotations

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# make repo root importable (so `import src...` works when running Streamlit)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PATHS
from src.data import download_dataset, load_raw, make_daily
from src.features import make_supervised, FEATURE_COLS, TARGET_COL


def load_bundle():
    bundle_path = PATHS.artifacts_dir / "model.joblib"
    if not bundle_path.exists():
        return None
    return joblib.load(bundle_path)


def load_metrics():
    metrics_path = PATHS.reports_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def pick_default_strategy(metrics: dict) -> str:
    # choose the best MAE among model vs baselines for the latest 30-day window
    candidates = {
        "model": metrics["model"]["mae"],
        "baseline_last_week": metrics["baseline_last_week"]["mae"],
        "baseline_ma7": metrics["baseline_ma7"]["mae"],
    }
    return min(candidates, key=candidates.get)


def forecast_next_days(
    daily: pd.DataFrame,
    bundle: dict | None,
    strategy: str,
    horizon: int,
    assumed_temp: float,
) -> pd.DataFrame:
    # Use last observed history and roll forward day-by-day.
    # If strategy is baseline_ma7: prediction = mean(last 7 values)
    # If strategy is baseline_last_week: prediction = value from 7 days ago
    # If strategy is model: use saved sklearn model + engineered features (temp is assumed constant)
    hist = daily.sort_values("day").reset_index(drop=True).copy()
    history_kwh = hist["daily_kwh"].tolist()
    history_temp = hist["daily_mean_temp"].tolist()

    model = bundle["model"] if (bundle and strategy == "model") else None

    future_rows = []
    last_day = pd.to_datetime(hist["day"].iloc[-1])

    for i in range(1, horizon + 1):
        day = last_day + pd.Timedelta(days=i)
        dow = int(day.dayofweek)
        month = int(day.month)
        dom = int(day.day)
        is_weekend = 1 if dow >= 5 else 0

        # temperature assumptions for future
        daily_mean_temp = float(assumed_temp)

        # lags/rolling from history (actual + predicted)
        lag_1 = float(history_kwh[-1])
        lag_7 = float(history_kwh[-7]) if len(history_kwh) >= 7 else float(np.mean(history_kwh))
        ma7 = float(np.mean(history_kwh[-7:])) if len(history_kwh) >= 7 else float(np.mean(history_kwh))
        ma14 = float(np.mean(history_kwh[-14:])) if len(history_kwh) >= 14 else float(np.mean(history_kwh))

        temp_lag_1 = float(history_temp[-1])
        temp_ma7 = float(np.mean(history_temp[-7:])) if len(history_temp) >= 7 else float(np.mean(history_temp))

        if strategy == "baseline_ma7":
            pred = ma7
        elif strategy == "baseline_last_week":
            pred = lag_7
        else:
            # model
            x = pd.DataFrame([{
                "daily_mean_temp": daily_mean_temp,
                "temp_lag_1": temp_lag_1,
                "temp_ma7": temp_ma7,
                "dow": dow,
                "month": month,
                "dom": dom,
                "is_weekend": is_weekend,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "ma7": ma7,
                "ma14": ma14,
            }])[FEATURE_COLS]
            pred = float(model.predict(x)[0])

        future_rows.append({"day": day, "pred_daily_kwh": pred})

        # push predicted into history for next step
        history_kwh.append(pred)
        history_temp.append(daily_mean_temp)

    return pd.DataFrame(future_rows)


st.set_page_config(page_title="Energy Forecast App", layout="wide")

st.title("Energy Forecast App")

metrics = load_metrics()
bundle = load_bundle()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("What this app does")
    st.write(
        "Loads the dataset, builds a daily table, then shows a short forecast. "
        "It also shows the saved evaluation metrics so it's not just 'trust me'."
    )

with col2:
    st.subheader("Current saved results (latest 30-day test)")
    if metrics is None:
        st.warning("No reports/metrics.json found yet. Run:  python -m modeling.train")
    else:
        st.json(metrics)

st.divider()

# Load data
files = download_dataset(PATHS.dataset_dir)
raw = load_raw(files.csv_path)
daily = make_daily(raw)

st.subheader("Daily data preview")
st.dataframe(daily.tail(10), use_container_width=True)

st.divider()

default_strategy = "baseline_ma7"
if metrics:
    default_strategy = pick_default_strategy(metrics)

strategy = st.selectbox(
    "Forecast method (default picks the best from your last test window)",
    options=["baseline_ma7", "baseline_last_week", "model"],
    index=["baseline_ma7", "baseline_last_week", "model"].index(default_strategy),
)

horizon = st.slider("Forecast horizon (days)", min_value=3, max_value=21, value=7, step=1)
assumed_temp = float(daily["daily_mean_temp"].tail(7).mean())
assumed_temp = st.number_input("Assumed mean temperature for future days", value=float(assumed_temp))

future = forecast_next_days(daily, bundle, strategy, horizon, assumed_temp)

st.subheader("Forecast")
st.dataframe(future, use_container_width=True)

fig = plt.figure()
plt.plot(future["day"], future["pred_daily_kwh"])
plt.xlabel("day")
plt.ylabel("predicted daily_kwh")
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig)

st.divider()

st.subheader("Saved test window plot")
plot_path = PATHS.images_dir / "test_predictions.png"
if plot_path.exists():
    st.image(str(plot_path), use_container_width=True)
else:
    st.info("No images/test_predictions.png yet. Run:  python -m modeling.train")
