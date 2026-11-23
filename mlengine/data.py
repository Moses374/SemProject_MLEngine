"""Load and clean the raw WISDM accelerometer file (standard library and numpy only)."""
import os
import urllib.request

import numpy as np

URL = "https://huggingface.co/datasets/stark0908/WISDM/resolve/main/WISDM_ar_v1.1_raw.txt"
DEFAULT_PATH = "WISDM_ar_v1.1_raw.txt"


def load_raw(path: str = DEFAULT_PATH) -> dict:
    """Parse and clean the raw file.

    Returns a dict of equal-length numpy arrays (user, activity, timestamp, x, y, z), sorted by
    user then time. Dropped: malformed lines, non-numeric or missing values, zero timestamps
    (the phone clock had not started) and duplicate (user, timestamp) readings.
    """
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    user, act, ts, xyz = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().rstrip(";").split(",")
            if len(parts) != 6:
                continue
            try:
                u, t = float(parts[0]), float(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            if not np.isfinite([u, t, x, y, z]).all() or t <= 0:
                continue
            user.append(u); act.append(parts[1]); ts.append(t); xyz.append((x, y, z))
    user = np.array(user, dtype=np.int64)
    act = np.array(act)
    ts = np.array(ts, dtype=np.int64)
    xyz = np.array(xyz)

    # keep the first of any duplicate (user, timestamp) pair, in file order
    _, first = np.unique(np.column_stack((user, ts)), axis=0, return_index=True)
    keep = np.sort(first)
    user, act, ts, xyz = user[keep], act[keep], ts[keep], xyz[keep]

    order = np.lexsort((ts, user))  # by user, then time
    return {"user": user[order], "activity": act[order], "timestamp": ts[order],
            "x": xyz[order, 0], "y": xyz[order, 1], "z": xyz[order, 2]}
