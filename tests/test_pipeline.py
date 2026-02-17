from src.config import PATHS
from src.data import download_dataset, load_raw, make_daily
from src.features import make_supervised, FEATURE_COLS, TARGET_COL


def test_daily_table_builds():
    files = download_dataset(PATHS.dataset_dir)
    raw = load_raw(files.csv_path)
    daily = make_daily(raw)

    assert len(daily) > 50
    assert "day" in daily.columns
    assert "daily_kwh" in daily.columns
    assert daily["daily_kwh"].isna().sum() == 0


def test_supervised_has_features_and_target():
    files = download_dataset(PATHS.dataset_dir)
    raw = load_raw(files.csv_path)
    daily = make_daily(raw)
    sup = make_supervised(daily)

    for c in FEATURE_COLS:
        assert c in sup.columns
    assert TARGET_COL in sup.columns
    assert sup[FEATURE_COLS + [TARGET_COL]].isna().sum().sum() == 0
