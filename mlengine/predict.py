"""Classify a window of accelerometer data and assess fall risk.

Usage:  python -m mlengine.predict readings.csv
CSV needs columns x,y,z (m/s^2, ~20 Hz); at least 80 rows. The last 80 rows are used.
"""
import sys

import joblib
import numpy as np
import pandas as pd

from .features import WINDOW, fall_risk, window_features


def predict_window(w: np.ndarray, model_path: str = "models/activity_model.joblib") -> dict:
    if w.shape != (WINDOW, 3):
        raise ValueError(f"expected shape ({WINDOW}, 3), got {w.shape}")
    bundle = joblib.load(model_path)
    X = pd.DataFrame([window_features(w)])[bundle["features"]]
    m = bundle["model"]
    proba = m.predict_proba(X)[0]
    i = int(proba.argmax())
    return {"activity": m.classes_[i], "confidence": float(proba[i]), **fall_risk(w)}


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    print(predict_window(df[["x", "y", "z"]].to_numpy()[-WINDOW:]))
