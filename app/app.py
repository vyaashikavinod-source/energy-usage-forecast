from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import tensorflow as tf
except Exception:
    tf = None


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

DATA_PATH = ROOT_DIR / "dataset" / "processed" / "daily_energy.csv"
ASSETS_DIR = APP_DIR / "model_assets"

TFLITE_PATH = ASSETS_DIR / "energy_model.tflite"
FEATURE_STATS_PATH = ASSETS_DIR / "feature_stats.json"
TF_METRICS_PATH = ASSETS_DIR / "metrics_tf.json"


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_daily() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Missing daily dataset at {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    if "day" not in df.columns:
        st.error("daily_energy.csv must contain a 'day' column")
        st.stop()

    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values("day").reset_index(drop=True)

    required = {"daily_kwh", "daily_mean_temp"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"daily_energy.csv is missing columns: {sorted(missing)}")
        st.stop()

    return df


def load_feature_stats():
    stats = _read_json(FEATURE_STATS_PATH)
    if not stats:
        return None

    feature_order = stats.get("feature_order") or stats.get("features") or stats.get("columns")
    means = stats.get("means")
    stds = stats.get("stds")

    if feature_order is None or means is None or stds is None:
        return None

    if isinstance(feature_order, dict):
        feature_order = list(feature_order.values())

    if isinstance(means, dict):
        means_vec = [float(means[f]) for f in feature_order]
    else:
        means_vec = [float(x) for x in means]

    if isinstance(stds, dict):
        stds_vec = [float(stds[f]) for f in feature_order]
    else:
        stds_vec = [float(x) for x in stds]

    if len(means_vec) != len(feature_order) or len(stds_vec) != len(feature_order):
        return None

    stds_vec = [s if s != 0 else 1.0 for s in stds_vec]

    return {
        "feature_order": feature_order,
        "means": np.array(means_vec, dtype=np.float32),
        "stds": np.array(stds_vec, dtype=np.float32),
    }


def load_tf_metrics():
    m = _read_json(TF_METRICS_PATH)
    if not m:
        return None

    def _get_mae(obj):
        if obj is None:
            return None
        v = obj.get("mae")
        if v is None:
            return None
        return float(v)

    ai_mae = _get_mae(m.get("model"))
    simple_mae = _get_mae(m.get("baseline_ma7")) or _get_mae(m.get("baseline"))

    return {"raw": m, "ai_mae": ai_mae, "simple_mae": simple_mae}


@st.cache_resource
def load_interpreter():
    if tf is None:
        return None
    if not TFLITE_PATH.exists():
        return None
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    return interpreter


def build_features_for_day(
    day: pd.Timestamp,
    history_kwh: list[float],
    assumed_temp: float,
    feature_order: list[str],
) -> dict:
    dow = int(day.dayofweek)
    month = int(day.month)
    dom = int(day.day)
    is_weekend = 1.0 if dow >= 5 else 0.0

    lag_1 = float(history_kwh[-1])
    lag_2 = float(history_kwh[-2]) if len(history_kwh) >= 2 else float(history_kwh[-1])
    lag_7 = float(history_kwh[-7]) if len(history_kwh) >= 7 else float(np.mean(history_kwh))
    lag_14 = float(history_kwh[-14]) if len(history_kwh) >= 14 else float(np.mean(history_kwh))

    roll7_mean = float(np.mean(history_kwh[-7:])) if len(history_kwh) >= 7 else float(np.mean(history_kwh))
    roll7_std = float(np.std(history_kwh[-7:])) if len(history_kwh) >= 7 else float(np.std(history_kwh))
    roll14_mean = float(np.mean(history_kwh[-14:])) if len(history_kwh) >= 14 else float(np.mean(history_kwh))
    roll14_std = float(np.std(history_kwh[-14:])) if len(history_kwh) >= 14 else float(np.std(history_kwh))

    base = {
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
        "roll14_std": float(roll14_std),
    }

    out = {}
    for f in feature_order:
        if f in base:
            out[f] = float(base[f])
        else:
            out[f] = 0.0
    return out


def predict_ai(
    daily: pd.DataFrame,
    horizon: int,
    assumed_temp: float,
    stats: dict,
    interpreter,
) -> pd.DataFrame | None:
    if interpreter is None or stats is None:
        return None

    feature_order = stats["feature_order"]
    mu = stats["means"]
    sigma = stats["stds"]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if not input_details:
        return None

    expected = int(np.prod(input_details[0]["shape"][1:])) if len(input_details[0]["shape"]) > 1 else int(input_details[0]["shape"][0])
    if expected != len(feature_order):
        return None

    history_kwh = daily["daily_kwh"].astype(float).tolist()
    last_day = pd.to_datetime(daily["day"].iloc[-1])

    rows = []
    for i in range(1, horizon + 1):
        d = last_day + pd.Timedelta(days=i)
        feats = build_features_for_day(d, history_kwh, assumed_temp, feature_order)
        x = np.array([feats[f] for f in feature_order], dtype=np.float32)
        x = (x - mu) / sigma
        x = x.reshape(1, -1).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        pred = float(np.ravel(y)[0])

        rows.append({"day": d, "pred_kwh": pred})
        history_kwh.append(pred)

    return pd.DataFrame(rows)


def predict_simple(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    hist = daily["daily_kwh"].astype(float).tolist()
    last_day = pd.to_datetime(daily["day"].iloc[-1])

    rows = []
    for i in range(1, horizon + 1):
        d = last_day + pd.Timedelta(days=i)
        if len(hist) >= 7:
            pred = float(np.mean(hist[-7:]))
        else:
            pred = float(np.mean(hist))
        rows.append({"day": d, "pred_kwh": pred})
        hist.append(pred)

    return pd.DataFrame(rows)


def build_insights(daily: pd.DataFrame, forecast_df: pd.DataFrame, horizon: int, assumed_temp: float) -> list[str]:
    last7 = daily.tail(7)["daily_kwh"].astype(float).to_numpy()
    prev7 = daily.tail(14).head(7)["daily_kwh"].astype(float).to_numpy() if len(daily) >= 14 else None

    expected = float(forecast_df["pred_kwh"].sum())
    last_week = float(np.sum(last7))

    if prev7 is not None and len(prev7) == 7 and float(np.sum(prev7)) != 0:
        change = (last_week - float(np.sum(prev7))) / float(np.sum(prev7)) * 100.0
        change_txt = f"Compared to last week: {change:+.0f}% change in expected usage"
    else:
        change_txt = "Compared to last week: not enough history to compute change"

    temp_txt = "Peak day driver: temperature assumption"
    return [change_txt, temp_txt]


def plot_forecast(
    daily: pd.DataFrame,
    simple_df: pd.DataFrame,
    ai_df: pd.DataFrame | None,
    compare: bool,
):
    hist = daily.tail(35).copy()
    hist = hist[["day", "daily_kwh"]].rename(columns={"daily_kwh": "kwh"})

    def _to_series(df, name):
        x = pd.DataFrame({"day": pd.to_datetime(df["day"]), "kwh": df["pred_kwh"].astype(float), "series": name})
        return x

    series = [pd.DataFrame({"day": hist["day"], "kwh": hist["kwh"], "series": "Historical"})]
    series.append(_to_series(simple_df, "Simple forecast"))

    if compare and ai_df is not None:
        series.append(_to_series(ai_df, "AI forecast"))

    plot_df = pd.concat(series, ignore_index=True)

    if go is None:
        st.line_chart(plot_df, x="day", y="kwh", color="series")
        return

    fig = go.Figure()
    for name in plot_df["series"].unique():
        sdf = plot_df[plot_df["series"] == name]
        fig.add_trace(go.Scatter(x=sdf["day"], y=sdf["kwh"], mode="lines+markers", name=name))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="",
        yaxis_title="kWh",
        legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def to_forecast_csv(df: pd.DataFrame, rate: float) -> bytes:
    out = df.copy()
    out["estimated_cost_usd"] = out["pred_kwh"].astype(float) * float(rate)
    out = out[["day", "pred_kwh", "estimated_cost_usd"]]
    out["day"] = pd.to_datetime(out["day"]).dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")


st.set_page_config(page_title="Electricity Usage Forecast", layout="centered")
st.markdown(
    """
    <style>
    /* Make only the Compare forecasts toggle red */
    div[data-testid="stToggle"] label:has(span:contains("Compare forecasts")) + div [data-baseweb="switch"] > div {
        background-color: #dc2626 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stAppViewContainer"] { background: #0b0f19; }
[data-testid="stAppViewContainer"] { color: #e5e7eb; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
h1, h2, h3, h4, p, div, span, label { color: #e5e7eb !important; }
hr { border-color: rgba(255,255,255,0.08); }

.block-container { padding-top: 26px; max-width: 980px; }

.card {
  background: #101827;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 16px 16px;
}
.kpi-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.kpi-title { font-size: 13px; opacity: 0.85; margin: 0; }
.kpi-value { font-size: 28px; margin: 6px 0 0 0; }
.kpi-sub { font-size: 12px; opacity: 0.75; margin: 6px 0 0 0; }

.icon {
  width: 34px; height: 34px;
  display: inline-flex; align-items: center; justify-content: center;
  border-radius: 10px;
  margin-right: 10px;
  font-size: 18px;
}
.blue { background: rgba(59,130,246,0.22); color: #93c5fd; }
.green { background: rgba(34,197,94,0.22); color: #86efac; }
.yellow { background: rgba(234,179,8,0.22); color: #fde68a; }

.section-title { font-size: 18px; margin: 18px 0 10px 0; }

div[data-testid="stToggle"] label div[role="switch"][aria-checked="true"] { background: #ef4444 !important; }
div[data-testid="stToggle"] label div[role="switch"] { background: rgba(255,255,255,0.18) !important; }

.stButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
  <div class="icon yellow">⚡</div>
  <div style="font-size:34px;line-height:1.1;">Electricity Usage Forecast</div>
</div>
<div style="opacity:0.82;margin-bottom:14px;">
Forecast daily electricity usage and estimated cost for the upcoming days. Adjust scenario and export results below.
</div>
""",
    unsafe_allow_html=True,
)

daily = load_daily()
stats = load_feature_stats()
tfm = load_tf_metrics()
interpreter = load_interpreter()

st.markdown('<div class="section-title">Adjust forecast scenario</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    horizon = st.selectbox("Forecast days", [7, 10, 14, 21], index=2, key="horizon_days")
with c2:
    temp_default = float(daily["daily_mean_temp"].tail(7).mean())
    temp_choice = st.selectbox("Expected avg. temperature (°C)", [round(temp_default - 5, 1), round(temp_default, 1), round(temp_default + 5, 1)], index=1, key="temp_choice")
with c3:
    rate = st.selectbox("Electricity rate ($/kWh)", [0.10, 0.20, 0.25, 0.30], index=1, key="rate_choice")
with c4:
    style = st.selectbox("Forecast style", ["Recommended", "Simple", "AI"], index=0, key="style_choice")

compare = st.toggle("Compare forecasts", value=False, key="compare_forecasts_toggle")

simple_df = predict_simple(daily, int(horizon))
ai_df = None
ai_ok = True
ai_reason = ""

if tf is None:
    ai_ok = False
    ai_reason = "AI forecast is unavailable right now. TensorFlow is not installed."
elif interpreter is None:
    ai_ok = False
    ai_reason = "AI forecast is unavailable right now. The model file was not found."
elif stats is None:
    ai_ok = False
    ai_reason = "AI forecast is unavailable right now. feature_stats.json has an unsupported means/stds format."
else:
    ai_df = predict_ai(daily, int(horizon), float(temp_choice), stats, interpreter)
    if ai_df is None:
        ai_ok = False
        ai_reason = "AI forecast is unavailable right now. The model input shape does not match feature_order."

use_ai = False
if style == "AI":
    use_ai = ai_ok
elif style == "Simple":
    use_ai = False
else:
    if tfm and tfm["ai_mae"] is not None and tfm["simple_mae"] is not None:
        use_ai = bool(tfm["ai_mae"] < tfm["simple_mae"]) and ai_ok
    else:
        use_ai = ai_ok

chosen_df = ai_df if (use_ai and ai_df is not None) else simple_df
chosen_name = "AI" if (use_ai and ai_df is not None) else "Simple"

tomorrow_kwh = float(chosen_df.iloc[0]["pred_kwh"])
est_cost = float(chosen_df["pred_kwh"].sum()) * float(rate)
peak_idx = int(chosen_df["pred_kwh"].astype(float).values.argmax())
peak_day = pd.to_datetime(chosen_df.iloc[peak_idx]["day"])
peak_val = float(chosen_df.iloc[peak_idx]["pred_kwh"])

st.markdown(
    f"""
<div class="kpi-row" style="margin-top:12px;">
  <div class="card">
    <div style="display:flex;align-items:center;">
      <div class="icon blue">⚡</div>
      <div>
        <div class="kpi-title">Tomorrow's Usage</div>
        <div class="kpi-value">{tomorrow_kwh:.1f} kWh</div>
        <div class="kpi-sub">Forecast style: {chosen_name}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;align-items:center;">
      <div class="icon green">$</div>
      <div>
        <div class="kpi-title">Estimated {int(horizon)}-Day Cost</div>
        <div class="kpi-value">${est_cost:.2f}</div>
        <div class="kpi-sub">Using ${float(rate):.2f} per kWh</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div style="display:flex;align-items:center;">
      <div class="icon yellow">🗓</div>
      <div>
        <div class="kpi-title">Peak Usage Date</div>
        <div class="kpi-value">{peak_day.strftime("%b %d, %Y")}</div>
        <div class="kpi-sub">{peak_val:.1f} kWh</div>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Forecast</div>', unsafe_allow_html=True)
plot_forecast(daily, simple_df, ai_df, compare=bool(compare))

st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
ins = build_insights(daily, chosen_df, int(horizon), float(temp_choice))
for line in ins:
    st.write(f"• {line}")

csv_bytes = to_forecast_csv(chosen_df, float(rate))
st.download_button(
    "Download forecast CSV",
    data=csv_bytes,
    file_name="forecast.csv",
    mime="text/csv",
)

if (compare or style == "AI") and not ai_ok:
    st.info(ai_reason)

if tfm and tfm["ai_mae"] is not None and tfm["simple_mae"] is not None:
    ai_mae = float(tfm["ai_mae"])
    simple_mae = float(tfm["simple_mae"])
    improvement = (simple_mae - ai_mae) / simple_mae * 100.0 if simple_mae != 0 else 0.0
    st.markdown(
        f"""
<div style="margin-top:14px;opacity:0.85;">
Model quality: Validation MAE (AI) {ai_mae:.3f} | Validation MAE (Simple) {simple_mae:.3f} | Improvement {improvement:+.0f}%
</div>
""",
        unsafe_allow_html=True,
    )