"""Facts and figures that explain OUR OWN random forest (mlengine/models.py): a real tree, real tree votes,
train vs test, speed, training time and file size.

Run from the project root:  python -m reports.build_ml_figs_own
Writes reports/own/fig/learn_vs_use.png, tree.png, votes.png and reports/own/ml_facts.json.
The scikit-learn version of this script is reports/build_ml_figs.py.
"""
import json
import os
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mlengine.data import load_raw
from mlengine.evaluation import accuracy as accuracy_score, group_shuffle_split
from mlengine.features import WINDOW, STEP, build_dataset, make_windows, window_features
from mlengine.models import RandomForest

OUT = os.path.join("reports", "own")
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "text.color": "black"})
SEED = 42
data = load_raw()
Xa, ya, ga, feat_names = build_dataset(data)
X, y, groups = pd.DataFrame(Xa, columns=feat_names), pd.Series(ya), pd.Series(ga)
tr, te = group_shuffle_split(groups.to_numpy(), test_size=0.25, seed=SEED)
rf = RandomForest(n_estimators=300, random_state=SEED, class_weight="balanced")
t0 = time.perf_counter()
rf.fit(X.iloc[tr], y.iloc[tr])
fit_seconds = time.perf_counter() - t0
with tempfile.TemporaryDirectory() as d:  # file size of the model as train.py saves it
    joblib.dump({"model": rf, "features": feat_names}, os.path.join(d, "m.joblib"), compress=3)
    model_mb = os.path.getsize(os.path.join(d, "m.joblib")) / 1e6

facts = {
    "train_windows": int(len(tr)), "test_windows": int(len(te)),
    "train_accuracy": float(accuracy_score(y.iloc[tr], rf.predict(X.iloc[tr]))),
    "test_accuracy": float(accuracy_score(y.iloc[te], rf.predict(X.iloc[te]))),
    "n_trees": len(rf.trees_), "n_features": X.shape[1],
    "avg_depth": float(np.mean([t.depth_ for t in rf.trees_])),
    "avg_leaves": float(np.mean([t.n_leaves for t in rf.trees_])),
    "total_nodes": int(sum(t.node_count for t in rf.trees_)),
    "fit_seconds_300_trees": fit_seconds,
    "model_file_mb_27_subjects": model_mb,
}

# speed: raw window -> features -> prediction
user, act, w = next(make_windows(data))
def time_it():
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        f = window_features(w)
        rf.predict_proba(np.array([[f[n] for n in feat_names]]))
    return (time.perf_counter() - t0) / N * 1000
facts["ms_per_prediction"] = time_it()  # our forest always predicts on one thread

# --- fig: learn vs use
def box(ax, xy, w_, h, text, fc="white", fs=10):
    ax.add_patch(FancyBboxPatch(xy, w_, h, boxstyle="round,pad=0.02", fc=fc, ec="black", lw=1))
    ax.text(xy[0] + w_ / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)
def arr(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1, color="black"))
fig, ax = plt.subplots(figsize=(9, 4.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
ax.text(0.1, 4.35, f"1. LEARNING (done once, on a computer, about {round(facts['fit_seconds_300_trees'], -1):.0f} seconds)", fontsize=11, fontweight="bold")
box(ax, (0.1, 2.9), 2.3, 1.1, "25,230 labelled\nwindows\n(\"this was Walking\")", "#eeeeee")
arr(ax, (2.45, 3.45), (3.15, 3.45))
box(ax, (3.2, 2.9), 2.3, 1.1, "Our training algorithm\n(mlengine/models.py)\nfinds the patterns")
arr(ax, (5.55, 3.45), (6.25, 3.45))
box(ax, (6.3, 2.9), 3.5, 1.1, f"Saved model file\nmodels/activity_model.joblib\n(300 trees, about {round(model_mb * 36 / 27):.0f} MB)", "#cccccc")
ax.plot([0.1, 9.9], [2.45, 2.45], color="black", lw=0.8, ls=":")
ax.text(0.1, 2.1, f"2. USING (every time new data arrives, about {round(facts['ms_per_prediction'], -1):.0f} ms on a laptop)", fontsize=11, fontweight="bold")
box(ax, (0.1, 0.4), 2.3, 1.1, "New 4-second\nwindow from the\nphone (no label)", "#eeeeee")
arr(ax, (2.45, 0.95), (3.15, 0.95))
box(ax, (3.2, 0.4), 2.3, 1.1, "Same 32 features\ncomputed")
arr(ax, (5.55, 0.95), (6.25, 0.95))
box(ax, (6.3, 0.4), 3.5, 1.1, "Load saved model\n=> e.g. \"Walking, 92% sure\"\n(+ fall rule check)", "#cccccc")
arr(ax, (8.05, 2.9), (8.05, 1.55))
ax.text(8.15, 2.2, "reused", fontsize=9, style="italic")
fig.savefig(f"{FIG}/learn_vs_use.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# --- fig: a real tree (first tree, top 2 levels), redrawn cleanly from its actual splits
t = rf.trees_[0]
names = list(X.columns)
Xtr = X.iloc[tr].to_numpy()
def n_reaching(n):
    """How many training windows pass through node n (the drawing only goes 2 levels deep)."""
    node = np.zeros(len(Xtr), dtype=np.int64)
    hit = node == n
    for _ in range(3):
        internal = t.left[node] != -1
        go_left = Xtr[np.arange(len(Xtr)), np.maximum(t.feature[node], 0)] <= t.threshold[node]
        node = np.where(internal, np.where(go_left, t.left[node], t.right[node]), node)
        hit |= node == n
    return int(hit.sum())
def question(n):
    return f"{names[t.feature[n]]} <= {t.threshold[n]:.2f} ?"
def outcome(n):
    if t.left[n] == -1:
        cls = rf.classes_[int(np.argmax(t.value[n]))]
        return f"{cls}\n(end point: {n_reaching(n):,} windows)"
    cls = rf.classes_[int(np.argmax(t.value[n]))]
    share = t.value[n].max()  # class-balanced share, as the tree itself weighs it
    return f"mostly {cls}\n({share:.0%} of {n_reaching(n):,} windows)"
fig, ax = plt.subplots(figsize=(9.5, 4.4)); ax.set_xlim(-0.4, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")
root, L, R = 0, t.left[0], t.right[0]
pos = {root: (5, 3.9), L: (2.5, 2.3), R: (7.5, 2.3)}
kids = {L: (t.left[L], t.right[L]), R: (t.left[R], t.right[R])}
xs = {kids[L][0]: 1.2, kids[L][1]: 3.8, kids[R][0]: 6.2, kids[R][1]: 8.8}
def nbox(xy, txt, fc="white", w_=2.5, h=0.75):
    ax.add_patch(FancyBboxPatch((xy[0] - w_ / 2, xy[1] - h / 2), w_, h, boxstyle="round,pad=0.02", fc=fc, ec="black", lw=1))
    ax.text(xy[0], xy[1], txt, ha="center", va="center", fontsize=9)
nbox(pos[root], question(root), w_=3.0)
for n in (L, R):
    nbox(pos[n], question(n), w_=3.0)
    ax.annotate("", xy=(pos[n][0], pos[n][1] + 0.4), xytext=(pos[root][0], pos[root][1] - 0.4), arrowprops=dict(arrowstyle="-|>", color="black"))
    ax.text((pos[n][0] + pos[root][0]) / 2 + (-0.35 if n == L else 0.3), (pos[n][1] + pos[root][1]) / 2 + 0.1, "yes" if n == L else "no", fontsize=9, style="italic")
    for k, yes in zip(kids[n], ("yes", "no")):
        nbox((xs[k], 0.75), outcome(k), fc="#eeeeee", w_=2.35, h=1.0)
        ax.annotate("", xy=(xs[k], 1.28), xytext=(pos[n][0], pos[n][1] - 0.4), arrowprops=dict(arrowstyle="-|>", color="black"))
        ax.text((xs[k] + pos[n][0]) / 2 + (-0.3 if yes == "yes" else 0.2), (1.28 + pos[n][1] - 0.4) / 2, yes, fontsize=9, style="italic")
facts["tree_root"] = question(root)
facts["tree_top"] = {"root": question(root), "left": question(L), "right": question(R),
                     "leaves": [outcome(k).replace("\n", " ") for n in (L, R) for k in kids[n]]}
fig.savefig(f"{FIG}/tree.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# --- fig: how the 300 trees vote on two real, unseen windows
pred = rf.predict(X.iloc[te])
yt = y.iloc[te].to_numpy()
good = next(i for i in range(len(te)) if yt[i] == "Walking" and pred[i] == "Walking")
bad = next(i for i in range(len(te)) if yt[i] == "Downstairs" and pred[i] != "Downstairs")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
facts["votes"] = {}
for ax, i, ttl in [(axes[0], good, "Real Walking window"), (axes[1], bad, "Real Downstairs window")]:
    x1 = X.iloc[[te[i]]]
    votes = [rf.classes_[int(tree.predict_proba_encoded(x1.to_numpy()).argmax())] for tree in rf.trees_]
    cnt = pd.Series(votes).value_counts().reindex(rf.classes_).fillna(0)
    ax.bar(range(len(cnt)), cnt.values, color=["#555555" if c == cnt.idxmax() else "#bbbbbb" for c in cnt.index], edgecolor="black")
    ax.set_xticks(range(len(cnt))); ax.set_xticklabels(cnt.index, rotation=35, ha="right")
    for k, v in enumerate(cnt.values):
        ax.text(k, v + 3, int(v), ha="center", fontsize=9)
    ax.set_title(f"{ttl}\n(actual: {yt[i]}, model says: {pred[i]})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    facts["votes"][ttl] = {k: int(v) for k, v in cnt.items()}
axes[0].set_ylabel("Trees voting (out of 300)")
fig.tight_layout(); fig.savefig(f"{FIG}/votes.png", dpi=200, bbox_inches="tight"); plt.close(fig)

json.dump(facts, open(f"{OUT}/ml_facts.json", "w"), indent=2)
print(json.dumps(facts, indent=2))
