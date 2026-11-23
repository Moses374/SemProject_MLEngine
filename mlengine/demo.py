"""Live demo: a local web page that classifies 4-second windows of accelerometer data.

    python -m mlengine.train      # once, creates models/demo_model.joblib
    python -m mlengine.demo       # then open http://localhost:8000

Windows come only from the held-out test subjects, whom the demo model was never trained on.
"""
import argparse
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

from .data import load_raw
from .features import G, WINDOW, fall_risk, make_windows, window_features

PAGE = os.path.join(os.path.dirname(__file__), "demo_page.html")
ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]


class Engine:
    def __init__(self, model_path="models/demo_model.joblib", data_path="WISDM_ar_v1.1_raw.txt"):
        bundle = joblib.load(model_path)
        self.model, self.features = bundle["model"], bundle["features"]
        self.test_users = bundle["test_users"]
        df = load_raw(data_path)
        df = df[df["user"].isin(self.test_users)]
        self.pool = {a: [] for a in ACTIVITIES}
        # non-overlapping windows from held-out subjects only
        for user, act, w in make_windows(df, WINDOW, WINDOW):
            self.pool[act].append((user, w))
        self.rng = random.Random()

    def classify(self, w: np.ndarray) -> dict:
        X = pd.DataFrame([window_features(w)])[self.features]
        proba = self.model.predict_proba(X)[0]
        probs = {c: float(p) for c, p in zip(self.model.classes_, proba)}
        top = max(probs, key=probs.get)
        return {"x": w[:, 0].round(3).tolist(), "y": w[:, 1].round(3).tolist(),
                "z": w[:, 2].round(3).tolist(), "predicted": top, "confidence": probs[top],
                "probs": probs, "fall": fall_risk(w), "g": G, "hz": 20}

    def real_window(self, activity: str) -> dict:
        user, w = self.rng.choice(self.pool[activity])
        out = self.classify(w)
        out.update(actual=activity, subject=int(user), kind="real recording")
        return out

    def simulated_fall(self) -> dict:
        """Walking, a hard impact, then a motionless phone. A staged signal, not a recording."""
        user, walk = self.rng.choice(self.pool["Walking"])
        w = walk.copy()
        rng = np.random.default_rng()
        rest = np.array([0.0, G, 0.0]) + rng.normal(0, 0.15, (WINDOW, 3))
        impact = 28
        w[impact + 1:] = rest[impact + 1:]
        w[impact] = [6.0, 33.0, 11.0]
        out = self.classify(w)
        out.update(actual="(staged fall)", subject=None, kind="simulated fall")
        return out


def make_handler(engine: Engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console clean on stage
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/window":
                act = parse_qs(u.query).get("activity", ["Walking"])[0]
                if act not in engine.pool or not engine.pool[act]:
                    return self._send(400, b'{"error":"unknown activity"}', "application/json")
                return self._send(200, json.dumps(engine.real_window(act)).encode(), "application/json")
            if u.path == "/api/fall":
                return self._send(200, json.dumps(engine.simulated_fall()).encode(), "application/json")
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            self._send(404, b"not found", "text/plain")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if not os.path.exists("models/demo_model.joblib"):
        raise SystemExit("models/demo_model.joblib not found. Run: python -m mlengine.train")
    print("Loading data and model (about 15 seconds)...")
    engine = Engine()
    n = {a: len(v) for a, v in engine.pool.items()}
    print(f"Held-out subjects: {engine.test_users}. Windows available: {n}")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine))
    print(f"Demo running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
