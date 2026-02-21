import argparse
from pathlib import Path
import pandas as pd

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise FileNotFoundError(str(raw_path))

    df = pd.read_csv(raw_path)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the raw file.")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.floor("D")

    if "Appliances" not in df.columns:
        raise ValueError("Expected an 'Appliances' column (target) in the raw file.")

    df["kwh_10min"] = df["Appliances"].astype(float) / 1000.0 / 6.0
    daily_kwh = df.groupby("day", as_index=False)["kwh_10min"].sum().rename(columns={"kwh_10min": "daily_kwh"})

    temp_candidates = [c for c in ["T_out", "T6", "T2", "T1"] if c in df.columns]
    if temp_candidates:
        temp_col = temp_candidates[0]
        daily_temp = df.groupby("day", as_index=False)[temp_col].mean().rename(columns={temp_col: "daily_mean_temp"})
        out_df = daily_kwh.merge(daily_temp, on="day", how="left")
    else:
        out_df = daily_kwh.copy()

    out_path = Path(args.out)
    _ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
