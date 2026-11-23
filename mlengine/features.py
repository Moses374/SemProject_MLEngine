"""Sliding-window segmentation and feature extraction."""
import numpy as np

WINDOW = 80      # ~4 s at the 20 Hz WISDM sampling rate
STEP = 40        # 50% overlap
FALL_G_THRESHOLD = 2.5  # impact threshold (multiples of gravity) for the fall-risk flag
G = 9.81


def make_windows(data: dict, window: int = WINDOW, step: int = STEP):
    """Yield (user, activity, array[window,3]) windows, each from one user doing one activity.

    `data` is the dict returned by data.load_raw. Groups are visited in order of first appearance.
    """
    users, acts = data["user"], data["activity"]
    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    names, codes = np.unique(acts, return_inverse=True)
    key = users.astype(np.int64) * len(names) + codes
    _, first = np.unique(key, return_index=True)
    for k in key[np.sort(first)]:
        rows = xyz[key == k]
        user, act = int(k // len(names)), str(names[k % len(names)])
        for s in range(0, len(rows) - window + 1, step):
            yield user, act, rows[s:s + window]


def _corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def window_features(w: np.ndarray) -> dict:
    """Statistical + signal-magnitude features for one [window,3] block."""
    f = {}
    mag = np.linalg.norm(w, axis=1)
    for i, ax in enumerate("xyz"):
        a = w[:, i]
        f[f"{ax}_mean"] = a.mean()
        f[f"{ax}_std"] = a.std()
        f[f"{ax}_min"] = a.min()
        f[f"{ax}_max"] = a.max()
        f[f"{ax}_mad"] = np.abs(a - np.median(a)).mean()
        f[f"{ax}_zcr"] = np.mean(np.diff(np.sign(a - a.mean())) != 0)
        spec = np.abs(np.fft.rfft(a - a.mean()))
        f[f"{ax}_dom_freq"] = spec[1:].argmax() + 1
        f[f"{ax}_energy"] = (spec ** 2).sum() / len(a)
    f["mag_mean"] = mag.mean()
    f["mag_std"] = mag.std()
    f["mag_max"] = mag.max()
    f["mag_min"] = mag.min()
    f["mag_range"] = mag.max() - mag.min()
    f["xy_corr"] = _corr(w[:, 0], w[:, 1])
    f["xz_corr"] = _corr(w[:, 0], w[:, 2])
    f["yz_corr"] = _corr(w[:, 1], w[:, 2])
    return f


def build_dataset(data: dict, window: int = WINDOW, step: int = STEP):
    """Return (X, y, users, feature_names): one row of 32 features per labelled window."""
    rows, y, users = [], [], []
    for user, act, w in make_windows(data, window, step):
        rows.append(window_features(w))
        y.append(act)
        users.append(user)
    names = list(rows[0])
    X = np.array([[r[n] for n in names] for r in rows])
    return X, np.array(y), np.array(users), names


def fall_risk(w: np.ndarray) -> dict:
    """Heuristic fall flag: a high-impact spike followed by near-stillness.

    WISDM contains no labelled falls, so this is a rule layered on top of the
    activity classifier, not a trained detector. It must be validated on a
    dataset with real falls (e.g. SisFall, MobiAct) before any real use.
    """
    mag = np.linalg.norm(w, axis=1) / G
    peak = int(mag.argmax())
    after = mag[peak + 1:]
    impact = bool(mag[peak] >= FALL_G_THRESHOLD)
    still_after = bool(len(after) >= 10 and abs(after[-10:].mean() - 1.0) < 0.15
                       and after[-10:].std() < 0.1)
    return {"impact_g": float(mag[peak]), "impact": impact,
            "post_impact_still": still_after, "fall_suspected": impact and still_after}
