import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Missing required column. Looked for one of: {candidates}. Found: {sorted(df.columns)}")

def build_features(df: pd.DataFrame, date_col: str, y_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if "daily_mean_temp" not in df.columns:
        if "temp" in df.columns:
            df["daily_mean_temp"] = df["temp"]

    df["dow"] = df[date_col].dt.dayofweek.astype(int)
    df["month"] = df[date_col].dt.month.astype(int)
    df["dom"] = df[date_col].dt.day.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    for lag in [1, 2, 7, 14]:
        df[f"lag_{lag}"] = df[y_col].shift(lag)

    df["roll7_mean"] = df[y_col].shift(1).rolling(7).mean()
    df["roll7_std"] = df[y_col].shift(1).rolling(7).std()
    df["roll14_mean"] = df[y_col].shift(1).rolling(14).mean()

    y = df[y_col].copy()
    drop_cols = [y_col]
    x = df.drop(columns=drop_cols)

    keep = [c for c in x.columns if c != date_col]
    x = x[keep]

    m = x.notna().all(axis=1) & y.notna()
    x = x[m].reset_index(drop=True)
    y = y[m].reset_index(drop=True)

    return x, y

def baseline_last_week(y: pd.Series) -> pd.Series:
    return y.shift(7)

def baseline_ma7(y: pd.Series) -> pd.Series:
    return y.shift(1).rolling(7).mean()

def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--test_window_days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    df = pd.read_csv(data_path)
    date_col = _pick_col(df, ["day", "date", "timestamp", "ds"])
    y_col = _pick_col(df, ["daily_kwh", "kwh", "energy_kwh", "y"])

    x, y = build_features(df, date_col=date_col, y_col=y_col)

    n = len(x)
    if n < args.test_window_days + 60:
        raise ValueError(f"Not enough rows after feature building. Have {n}, need at least {args.test_window_days + 60}.")

    test_n = args.test_window_days
    train_end = n - test_n

    x_train_full = x.iloc[:train_end].copy()
    y_train_full = y.iloc[:train_end].copy()
    x_test = x.iloc[train_end:].copy()
    y_test = y.iloc[train_end:].copy()

    val_n = min(60, max(14, int(0.1 * len(x_train_full))))
    x_train = x_train_full.iloc[:-val_n].copy()
    y_train = y_train_full.iloc[:-val_n].copy()
    x_val = x_train_full.iloc[-val_n:].copy()
    y_val = y_train_full.iloc[-val_n:].copy()

    feature_order = list(x_train.columns)
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0).replace(0, 1.0)

    def norm(df_: pd.DataFrame) -> np.ndarray:
        z = (df_[feature_order] - mu) / sd
        return z.astype(np.float32).values

    Xtr = norm(x_train)
    Xva = norm(x_val)
    Xte = norm(x_test)

    ytr = y_train.astype(np.float32).values.reshape(-1, 1)
    yva = y_val.astype(np.float32).values.reshape(-1, 1)
    yte = y_test.astype(np.float32).values.reshape(-1, 1)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=[tf.keras.metrics.MeanAbsoluteError()])

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=200,
        batch_size=32,
        verbose=1,
        callbacks=cb
    )

    pred = model.predict(Xte, verbose=0).reshape(-1)

    y_test_np = y_test.values.reshape(-1)
    b1 = baseline_last_week(y).iloc[train_end:].values.reshape(-1)
    b2 = baseline_ma7(y).iloc[train_end:].values.reshape(-1)

    metrics = {
        "test_window_days": int(args.test_window_days),
        "model": {"name": "tf_mlp_regressor", "mae": mae(y_test_np, pred), "rmse": rmse(y_test_np, pred)},
        "baseline_last_week": {"name": "same_day_last_week", "mae": mae(y_test_np, b1), "rmse": rmse(y_test_np, b1)},
        "baseline_ma7": {"name": "last_7_day_average", "mae": mae(y_test_np, b2), "rmse": rmse(y_test_np, b2)},
        "n_train": int(len(x_train_full)),
        "n_test": int(len(x_test)),
        "features": feature_order
    }

    artifacts_dir = Path("artifacts")
    reports_dir = Path("reports")
    _ensure_dir(artifacts_dir)
    _ensure_dir(reports_dir)

    sm_dir = artifacts_dir / "tf_savedmodel"
    if sm_dir.exists():
        for p in sm_dir.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(sm_dir.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        sm_dir.rmdir()
    model.export(str(sm_dir))


    (reports_dir / "metrics_tf.json").write_text(json.dumps(metrics, indent=2))

    stats = {
        "feature_order": feature_order,
        "mean": {k: float(mu[k]) for k in feature_order},
        "std": {k: float(sd[k]) for k in feature_order}
    }
    (artifacts_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
