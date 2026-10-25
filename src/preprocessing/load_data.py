"""Step 1: Data Collection — UCI Heart Disease dataset (920 samples, 13 features).

Combines all four source locations (Cleveland, Hungary, Switzerland, VA Long
Beach) rather than only the 303-row Cleveland subset that ucimlrepo's
processed endpoint exposes on its own.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "datasets" / "uci" / "heart_disease_raw.csv"

_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
_SOURCES = ["cleveland", "hungarian", "switzerland", "va"]
_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def fetch_uci_heart_disease(force_refresh: bool = False) -> pd.DataFrame:
    """Load and combine all four UCI Heart Disease source files. Downloads
    once, then caches to disk so later runs work offline.
    """
    if RAW_PATH.exists() and not force_refresh:
        return pd.read_csv(RAW_PATH)

    frames = []
    for source in _SOURCES:
        url = f"{_BASE_URL}/processed.{source}.data"
        df = pd.read_csv(url, header=None, names=_COLUMNS, na_values="?")
        df["source"] = source
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["num"] = combined["num"].astype(int)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RAW_PATH, index=False)
    return combined


if __name__ == "__main__":
    data = fetch_uci_heart_disease()
    print(f"Loaded {len(data)} rows, {data.shape[1]} columns")
    print(data["source"].value_counts())
    print(f"Cached at {RAW_PATH}")
