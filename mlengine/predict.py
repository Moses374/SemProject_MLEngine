"""Classify a window of accelerometer data and assess fall risk.

Usage:  python -m mlengine.predict readings.csv
CSV needs columns x,y,z (m/s^2, ~20 Hz); at least 80 rows. The last 80 rows are used.
"""
import sys

import joblib
import numpy as np

from .features import WINDOW, fall_risk, window_features


def predict_window(w: np.ndarray, model_path: str = "models/activity_model.joblib") -> dict:
    if w.shape != (WINDOW, 3):
        raise ValueError(f"expected shape ({WINDOW}, 3), got {w.shape}")
    bundle = joblib.load(model_path)
    f = window_features(w)
    X = np.array([[f[name] for name in bundle["features"]]])
    m = bundle["model"]
    if hasattr(m, "n_jobs"):
        m.n_jobs = 1  # one window at a time: threading overhead would make it about 3x slower
    proba = m.predict_proba(X)[0]
    i = int(proba.argmax())
    return {"activity": str(m.classes_[i]), "confidence": float(proba[i]), **fall_risk(w)}


if __name__ == "__main__":
    cols = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
    window = np.column_stack((cols["x"], cols["y"], cols["z"]))[-WINDOW:]
    print(predict_window(window))
