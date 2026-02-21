from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import PATHS
from src.data import download_dataset, load_raw, make_daily

MODEL_ASSETS_DIR = Path(__file__).parent / "model_assets"
TFLITE_PATH = MODEL_ASSETS_DIR / "energy_model.tflite"
STATS_PATH = MODEL_ASSETS_DIR / "feature_stats.json"
METRICS_TF_PATH = MODEL_ASSETS_DIR / "metrics_tf.json"

FEATURE_ORDER = [
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

def _read_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())

def load_tf_metrics():
    return _read_json(METRICS_TF_PATH)

def load_feature_stats():
    d = _read_json(STATS_PATH)
    if d is None:
        return None
    mu = np.array([float(d["mean"][k]) for k in FEATURE_ORDER], dtype=np.float32)
    sd = np.array([float(d["std"][k]) for k in FEATURE_ORDER], dtype=np.float32)
    sd = np.where(sd == 0.0, 1.0, sd).astype(np.float32)
    return mu, sd

def pick_recommended_forecaster(metrics: dict) -> str:
    model_mae = float(metrics["model"]["mae"])
    ma7_mae = float(metrics["baseline_ma7"]["mae"])
    return "tflite" if model_mae < ma7_mae else "ma7"

@st.cache_resource
def load_tflite_interpreter():
    if not TFLITE_PATH.exists():
        return None
    itp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    itp.allocate_tensors()
    return itp

def _tflite_predict(itp, x_row: np.ndarray) -> float:
    in_det = itp.get_input_details()[0]
    out_det = itp.get_output_details()[0]
    x_row = x_row.astype(np.float32).reshape(1, -1)
    itp.set_tensor(in_det["index"], x_row)
    itp.invoke()
    yhat = itp.get_tensor(out_det["index"]).reshape(-1)[0]
    return float(yhat)

def _safe_mean(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.mean(a))

def _safe_std(a):
    a = np.asarray(a, dtype=float)
    if len(a) <= 1:
        return 0.0
    return float(np.std(a, ddof=1))

def compute_features_for_day(day: pd.Timestamp, history_kwh: list[float], assumed_temp: float) -> dict:
    dow = int(day.dayofweek)
    month = int(day.month)
    dom = int(day.day)
    is_weekend = 1 if dow >= 5 else 0

    lag_1 = float(history_kwh[-1])
    lag_2 = float(history_kwh[-2]) if len(history_kwh) >= 2 else float(history_kwh[-1])
    lag_7 = float(history_kwh[-7]) if len(history_kwh) >= 7 else float(_safe_mean(history_kwh))
    lag_14 = float(history_kwh[-14]) if len(history_kwh) >= 14 else float(_safe_mean(history_kwh))

    prev = history_kwh[:-1] if len(history_kwh) >= 2 else history_kwh
    roll7 = prev[-7:] if len(prev) >= 7 else prev
    roll14 = prev[-14:] if len(prev) >= 14 else prev

    roll7_mean = float(_safe_mean(roll7))
    roll7_std = float(_safe_std(roll7))
    roll14_mean = float(_safe_mean(roll14))

    return {
        "daily_mean_temp": float(assumed_temp),
        "dow": float(dow),
        "month": float(month),
        "dom": float(dom),
        "is_weekend": float(is_weekend),
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_7": float(lag_7),
        "lag_14": float(lag_14),
        "roll7_mean": float(roll7_mean),
        "roll7_std": float(roll7_std),
        "roll14_mean": float(roll14_mean),
    }

def forecast_ma7(history_kwh: list[float]) -> float:
    if len(history_kwh) >= 7:
        return float(np.mean(history_kwh[-7:]))
    return float(np.mean(history_kwh))

def forecast_tflite_next_days(
    daily: pd.DataFrame,
    horizon: int,
    assumed_temp: float,
    itp,
    mu: np.ndarray,
    sd: np.ndarray,
) -> pd.DataFrame:
    hist = daily.sort_values("day").reset_index(drop=True).copy()
    history_kwh = hist["daily_kwh"].astype(float).tolist()
    last_day = pd.to_datetime(hist["day"].iloc[-1])

    rows = []
    for i in range(1, horizon + 1):
        d = last_day + pd.Timedelta(days=i)
        feats = compute_features_for_day(d, history_kwh, assumed_temp)
        x = np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
        x = (x - mu) / sd
        pred = _tflite_predict(itp, x)
        rows.append({"day": d, "pred_daily_kwh": pred})
        history_kwh.append(pred)

    return pd.DataFrame(rows)

def forecast_ma7_next_days(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    hist = daily.sort_values("day").reset_index(drop=True).copy()
    history_kwh = hist["daily_kwh"].astype(float).tolist()
    last_day = pd.to_datetime(hist["day"].iloc[-1])

    rows = []
    for i in range(1, horizon + 1):
        d = last_day + pd.Timedelta(days=i)
        pred = forecast_ma7(history_kwh)
        rows.append({"day": d, "pred_daily_kwh": pred})
        history_kwh.append(pred)

    return pd.DataFrame(rows)

st.set_page_config(page_title="Energy Forecast App", layout="wide")
st.title("Energy Forecast App")

tf_metrics = load_tf_metrics()
stats = load_feature_stats()
itp = load_tflite_interpreter()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("What this app does")
    st.write("Offline forecast bundle + two forecasters (MA7 baseline and TFLite model) with an automatic recommendation based on saved metrics.")

with col2:
    st.subheader("Model bundle status")
    st.write(f"TFLite found: {TFLITE_PATH.exists()}")
    st.write(f"Stats found: {STATS_PATH.exists()}")
    st.write(f"TF metrics found: {METRICS_TF_PATH.exists()}")
    if tf_metrics is not None:
        st.json(tf_metrics)

st.divider()

files = download_dataset(PATHS.dataset_dir)
raw = load_raw(files.csv_path)
daily = make_daily(raw)

st.subheader("Daily data preview")
st.dataframe(daily.tail(10), use_container_width=True)

horizon = st.slider("Forecast horizon (days)", min_value=3, max_value=21, value=7, step=1)
assumed_temp_default = float(daily["daily_mean_temp"].tail(7).mean())
assumed_temp = st.number_input("Assumed mean temperature for future days", value=float(assumed_temp_default))

recommended = "ma7"
if tf_metrics is not None:
    recommended = pick_recommended_forecaster(tf_metrics)

choice = st.selectbox(
    "Default view (based on metrics_tf.json)",
    options=["recommended", "ma7", "tflite"],
    index=0,
)

st.divider()

ma7_future = forecast_ma7_next_days(daily, horizon)

tflite_future = None
if (itp is not None) and (stats is not None):
    mu, sd = stats
    tflite_future = forecast_tflite_next_days(daily, horizon, assumed_temp, itp, mu, sd)

show = recommended if choice == "recommended" else choice

st.subheader("Forecast outputs")
left, right = st.columns([1, 1])

with left:
    st.write("MA7 baseline")
    st.dataframe(ma7_future, use_container_width=True)

with right:
    st.write("TFLite model")
    if tflite_future is None:
        st.warning("TFLite forecast not available. Check your model_assets files.")
    else:
        st.dataframe(tflite_future, use_container_width=True)

st.divider()

st.subheader("Plot (both, with selected highlighted)")
fig = plt.figure()
plt.plot(ma7_future["day"], ma7_future["pred_daily_kwh"], label="ma7")
if tflite_future is not None:
    plt.plot(tflite_future["day"], tflite_future["pred_daily_kwh"], label="tflite")
plt.title(f"Selected: {show} | Recommended: {recommended}")
plt.xlabel("day")
plt.ylabel("predicted daily_kwh")
plt.xticks(rotation=30)
plt.legend()
plt.tight_layout()
st.pyplot(fig)

st.divider()

st.subheader("Saved test window plot")
plot_path = PATHS.images_dir / "test_predictions.png"
if plot_path.exists():
    st.image(str(plot_path), use_container_width=True)
else:
    st.info("No images/test_predictions.png yet.")
