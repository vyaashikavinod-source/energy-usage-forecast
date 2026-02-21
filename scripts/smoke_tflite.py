import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    stats = json.loads(Path("artifacts/feature_stats.json").read_text())
    feature_order = stats["feature_order"]
    mu = np.array([stats["mean"][k] for k in feature_order], dtype=np.float32)
    sd = np.array([stats["std"][k] for k in feature_order], dtype=np.float32)
    sd = np.where(sd == 0, 1.0, sd).astype(np.float32)

    df = pd.read_csv(args.data)
    if "day" not in df.columns or "daily_kwh" not in df.columns:
        raise ValueError("Expected columns: day, daily_kwh (and optionally daily_mean_temp).")

    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values("day").reset_index(drop=True)

    df["dow"] = df["day"].dt.dayofweek.astype(int)
    df["month"] = df["day"].dt.month.astype(int)
    df["dom"] = df["day"].dt.day.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    for lag in [1, 2, 7, 14]:
        df[f"lag_{lag}"] = df["daily_kwh"].shift(lag)

    df["roll7_mean"] = df["daily_kwh"].shift(1).rolling(7).mean()
    df["roll7_std"] = df["daily_kwh"].shift(1).rolling(7).std()
    df["roll14_mean"] = df["daily_kwh"].shift(1).rolling(14).mean()

    if "daily_mean_temp" not in df.columns:
        df["daily_mean_temp"] = 0.0

    df = df.dropna().reset_index(drop=True)

    X = df[feature_order].astype(np.float32).values
    X = (X - mu) / sd

    interpreter = tf.lite.Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    n = min(args.n, len(X))
    preds = []
    for i in range(n):
        x = X[i:i+1]
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        yhat = interpreter.get_tensor(out["index"]).reshape(-1)[0]
        preds.append(float(yhat))

    print("Sample predictions:", preds)

if __name__ == "__main__":
    main()
