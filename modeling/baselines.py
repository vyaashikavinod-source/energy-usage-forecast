from __future__ import annotations
import numpy as np
import pandas as pd

def baseline_last_week(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["lag_7"].to_numpy()

def baseline_ma7(test_df: pd.DataFrame) -> np.ndarray:
    return test_df["ma7"].to_numpy()
