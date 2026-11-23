"""Load and clean the raw WISDM accelerometer file."""
import os
import urllib.request

import pandas as pd

URL = "https://huggingface.co/datasets/stark0908/WISDM/resolve/main/WISDM_ar_v1.1_raw.txt"
COLUMNS = ["user", "activity", "timestamp", "x", "y", "z"]
DEFAULT_PATH = "WISDM_ar_v1.1_raw.txt"


def load_raw(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Parse the raw file, dropping malformed rows, NaNs, and zero timestamps."""
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 6:
                rows.append(parts)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ["user", "timestamp", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df["timestamp"] > 0]  # some recordings start with 0 timestamps
    df = df.drop_duplicates(subset=["user", "timestamp"])
    df["user"] = df["user"].astype(int)
    return df.sort_values(["user", "timestamp"]).reset_index(drop=True)
