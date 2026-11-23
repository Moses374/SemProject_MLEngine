"""Build MLEngine_Colab.ipynb (a step-by-step, explained run of the whole project).

Run from the project root:  python -m reports.build_colab
The notebook clones the GitHub repo when it runs in Google Colab and uses the repo's own code (mlengine/).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md("""
# Activity Recognition and Fall-Risk Detection: step-by-step in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moses374/SemProject_MLEngine/blob/master/MLEngine_Colab.ipynb)

ICS 3202 Semester Project. Moses Kibet (183638), Collins Jared (184587), Kathy Auma (192231).

This notebook runs our whole machine learning engine from start to finish and **explains every step**. It uses the same code that is in the project's `mlengine/` folder on GitHub, so what you see here is exactly what the project does.

**The models are our own code.** The majority baseline, softmax (logistic) regression, decision tree and random forest are written from scratch with NumPy in `mlengine/models.py`. scikit-learn's versions are trained only as a reference to check ours against, and scikit-learn also provides the train/test split and the scoring functions.

**How to use it:** choose *Runtime > Run all*, or run the cells one at a time while presenting. Each section has a short explanation, then the code, then its result. Training our own forest takes a few minutes.

| Step | What happens |
|---|---|
| 1 | Get the code and check the library versions |
| 2 | Load and clean the WISDM accelerometer data |
| 3 | Cut it into 4-second windows |
| 4 | Turn each window into 32 summary numbers (features) |
| 5 | Split the people into train and test groups |
| 6 | Train our three models, and check them against scikit-learn |
| 7 | Evaluate honestly and look at the mistakes |
| 8 | Look inside the trained model: how our trees learn and vote |
| 9 | Save the models and make a prediction |
| 10 | The fall rule |
| 11 | The live demo (two phone screens) |
""")

# ---------------------------------------------------------------- 1 setup
md("""
## 1. Get the code and check the versions

In Colab we download (clone) our project from GitHub. On a laptop that already has the project, this cell does nothing.
""")
code("""
import os, sys, subprocess

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("SemProject_MLEngine"):
        subprocess.run(["git", "clone", "-q", "https://github.com/Moses374/SemProject_MLEngine.git"], check=True)
    os.chdir("SemProject_MLEngine")
print("Working folder:", os.getcwd())
print("Project files:", [f for f in sorted(os.listdir(".")) if not f.startswith(".")])
""")
md("""
Our models need only **numpy** (arrays and maths) plus **joblib** (to train trees in parallel and to save models). **scikit-learn** is used for
the person-wise split, the scoring functions and the reference models. We pin the versions we used so the numbers match our report.
""")
code("""
import numpy, sklearn

TESTED = ("2.3.3", "1.7.2")
have = (numpy.__version__, sklearn.__version__)
print("numpy", have[0], "| scikit-learn", have[1])
if have == TESTED:
    print("\\nVersions match the report, so the numbers below will match the slides.")
else:
    print("\\nThese are not the versions our report was produced with, so results may differ slightly.")
    print("To reproduce exactly: run  !pip install -q numpy==2.3.3 scikit-learn==1.7.2")
    print("then choose Runtime > Restart session, and run all cells again.")
""")

# ---------------------------------------------------------------- 2 data
md("""
## 2. Load and clean the data

The WISDM dataset has one line per sensor reading: *user, activity, timestamp, x, y, z*. The file is messy
(broken lines, a missing value, zero timestamps, duplicates), so `load_raw` cleans it. The first run downloads about 50 MB.

Here is the real code of that function:
""")
code("""
import inspect
from mlengine.data import load_raw
print(inspect.getsource(load_raw))
""")
code("""
import numpy as np

data = load_raw()                       # a dictionary of numpy arrays
print("Readings after cleaning:", f"{len(data['user']):,}")
print("People:", len(np.unique(data["user"])))
print("\\nReadings per activity:")
for act, n in zip(*np.unique(data["activity"], return_counts=True)):
    print(f"  {act:<11} {n:>8,}")
""")
md("""
*Say this:* "The raw file has 1,086,466 usable lines. After removing zero timestamps and duplicates we keep 1,019,496 readings
from 36 people. Walking and jogging dominate, so plain accuracy can be misleading. We also report macro F1."
""")

# ---------------------------------------------------------------- 3 windows
md("""
## 3. Cut the stream into 4-second windows

One reading says almost nothing. A **window** of 80 readings (4 seconds at 20 per second) shows the pattern of movement.
Each window starts 2 seconds after the previous one (50% overlap), and every window comes from one person doing one activity,
so it has one correct label.
""")
code("""
import matplotlib.pyplot as plt
from mlengine.features import make_windows, WINDOW, STEP

print(f"Window = {WINDOW} readings (4 s), step = {STEP} readings (2 s)")
print(inspect.getsource(make_windows))
""")
code("""
# One example window for two very different activities
examples = {}
for user, act, w in make_windows(data):
    if act in ("Walking", "Sitting") and act not in examples and w[:, 1].std() > 0:
        examples[act] = w
    if len(examples) == 2:
        break

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, (act, w) in zip(axes, examples.items()):
    ax.plot(np.linalg.norm(w, axis=1), color="black")
    ax.set_title(f"{act}: total acceleration")
    ax.set_xlabel("reading (20 per second)")
axes[0].set_ylabel("m/s²")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 4 features
md("""
## 4. Turn each window into 32 features

80 readings × 3 axes = 240 numbers is too many and too noisy to learn from directly. `window_features` summarises each window
into 32 numbers: averages (phone orientation), spread (how vigorous), zero crossings and a Fourier transform (how fast it repeats),
magnitude statistics and axis correlations.
""")
code("""
from mlengine.features import window_features
print(inspect.getsource(window_features))
""")
code("""
f = window_features(examples["Walking"])
print("A few of the 32 features for the Walking window:")
for k in ["z_mean", "y_mean", "y_std", "y_mad", "y_energy", "mag_max"]:
    print(f"  {k:<10} {f[k]:10.3f}")
f2 = window_features(examples["Sitting"])
print("\\nSame features for the Sitting window:")
for k in ["z_mean", "y_mean", "y_std", "y_mad", "y_energy", "mag_max"]:
    print(f"  {k:<10} {f2[k]:10.3f}")
""")
md("Now build the whole dataset: one row of 32 features per window. This takes about a minute.")
code("""
from mlengine.features import build_dataset

X, y, users, names = build_dataset(data)
print("Feature table:", X.shape, "(windows × features)")
print("Labels:", y.shape, "| people (groups):", users.shape)
""")

# ---------------------------------------------------------------- 5 split
md("""
## 5. Split by person, not by window

If we shuffled windows randomly, near-identical windows from the same person would land in both train and test, and the score
would be inflated. A real user is a **new person**, so we hold out whole people: about 75% for training, 25% for testing.
""")
code("""
from sklearn.model_selection import GroupShuffleSplit
from mlengine.train import SEED

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, users))
train_people, test_people = sorted(set(users[train_idx])), sorted(set(users[test_idx]))
print(f"Training: {len(train_people)} people, {len(train_idx):,} windows")
print(f"Testing:  {len(test_people)} people, {len(test_idx):,} windows")
print("Test people:", test_people)
print("Overlap between the two groups:", set(train_people) & set(test_people) or "none")
""")

# ---------------------------------------------------------------- 6 train
md("""
## 6. Train our three models and compare them

These classes are **our own code** (`mlengine/models.py`, written with NumPy):

* **Baseline (`MajorityClassifier`):** always guesses the most common activity. Any real model must beat it.
* **Logistic regression (`SoftmaxRegression`):** a simple linear model, trained by gradient descent with the Adam optimiser.
* **Random forest (`RandomForest`):** 300 decision trees that vote.

*Training* means showing the model the labelled training windows so it finds the patterns itself.
""")
code("""
import time
from sklearn.metrics import accuracy_score, f1_score
from mlengine.models import MajorityClassifier, SoftmaxRegression, RandomForest

models = {
    "baseline (majority)": MajorityClassifier(),
    "logistic regression": SoftmaxRegression(),
    "random forest": RandomForest(n_estimators=300, random_state=SEED, class_weight="balanced"),
}
results = {}
for name, m in models.items():
    t0 = time.perf_counter()
    m.fit(X[train_idx], y[train_idx])
    secs = time.perf_counter() - t0
    pred = m.predict(X[test_idx])
    results[name] = (accuracy_score(y[test_idx], pred), f1_score(y[test_idx], pred, average="macro"))
    print(f"{name:22s} accuracy={results[name][0]:.3f}   macro F1={results[name][1]:.3f}   (trained in {secs:.0f} s)")
""")
md("""
*Say this:* "Both real models beat the baseline by a wide margin, and the random forest is best, so that is the one we keep.
Macro F1 averages the score over all six activities equally, so small classes such as Standing count as much as Walking."

### Do our models agree with scikit-learn?

Writing models from scratch is only trustworthy if they match a well-tested library. So we train scikit-learn's versions on the **same split**
and compare. This is a check only: scikit-learn's models are never used by the engine.
""")
code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

references = {
    "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, class_weight="balanced"),
}
print(f"{'':22s}{'ours (acc / macro F1)':>26s}{'scikit-learn (acc / macro F1)':>34s}")
for name, ref in references.items():
    ref.fit(X[train_idx], y[train_idx])
    p = ref.predict(X[test_idx])
    ra, rf1 = accuracy_score(y[test_idx], p), f1_score(y[test_idx], p, average="macro")
    print(f"{name:22s}{results[name][0]:>14.3f} / {results[name][1]:.3f}{ra:>20.3f} / {rf1:.3f}")
""")
md("""
The two columns are within a fraction of a percentage point, so our from-scratch models learn essentially the same thing as the library's.
""")

md("""
### How does our random forest learn? (the real code)

Each tree is built one question at a time. At every step the tree:

1. looks at a **random handful of features** (the square root of 32, so 5) and tries **every possible threshold** for each, such as "is `y_mad <= 0.98`?";
2. keeps the question that makes the two resulting groups **purest**, measured by **Gini impurity** (0 means all one activity);
3. repeats on each side until a group is pure, then stops. That final group is a **leaf** that remembers its mix of activities.

The forest trains each tree on a **bootstrap sample** (rows drawn at random with replacement) and gives rare activities more weight (`class_weight="balanced"`).
To classify, every tree gives the activity mix of the leaf the window falls into, and the forest **averages** them.
""")
code("""
from mlengine.models import _fit_tree, DecisionTree
print(inspect.getsource(_fit_tree))
print(inspect.getsource(RandomForest.fit))
print(inspect.getsource(RandomForest.predict_proba))
""")
code("""
# The heart of it: how a tree picks its questions (Gini impurity, every threshold of each candidate feature)
print(inspect.getsource(DecisionTree.fit_encoded))
""")

# ---------------------------------------------------------------- 7 evaluate
md("""
## 7. Evaluate honestly and look at the mistakes
""")
code("""
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

rf = models["random forest"]
pred = rf.predict(X[test_idx])
print(classification_report(y[test_idx], pred))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y[test_idx], pred, ax=ax, xticks_rotation=40, cmap="Greys")
ax.set_title("Random forest on people it never saw")
plt.tight_layout(); plt.show()
""")
md("""
*Read the table:* **precision** = when the model says "Walking", how often is it right? **recall** = of all real Walking windows,
how many did it find? **Downstairs** is the weak class (low recall): it looks very like Upstairs and Walking from a pocket phone.
""")
code("""
# Overfitting check: the forest on its own training windows vs on unseen people
print(f"Accuracy on the windows it trained on : {accuracy_score(y[train_idx], rf.predict(X[train_idx])):.3f}")
print(f"Accuracy on people it has never seen  : {accuracy_score(y[test_idx], pred):.3f}")
""")
md("""
The forest scores 100% on its own training windows because it has effectively **memorised** them (overfitting). On new people it scores
about 85%. That is why we always report the unseen-people number.
""")

# ---------------------------------------------------------------- 8 inside
md("""
## 8. Look inside the trained model

A random forest is 300 decision trees. Each tree is a game of twenty questions about the features. Here are the real questions at
the top of the first tree (`to_text` is a method we wrote on our `DecisionTree` class):
""")
code("""
tree0 = rf.trees_[0]
print(tree0.to_text(names, [str(c) for c in rf.classes_], max_depth=2))
print(f"Trees: {len(rf.trees_)} | average depth: {np.mean([t.depth_ for t in rf.trees_]):.1f} levels"
      f" | average end points per tree: {np.mean([t.n_leaves for t in rf.trees_]):.0f}"
      f" | nodes in total: {sum(t.node_count for t in rf.trees_):,}")
""")
code("""
# How the 300 trees see one unseen Downstairs window (each tree's top choice)
i = next(k for k in test_idx if y[k] == "Downstairs")
votes = [rf.classes_[int(t.predict_proba_encoded(X[[i]]).argmax())] for t in rf.trees_]
vals, cnt = np.unique(votes, return_counts=True)
print("Real window: Downstairs | forest answer:", rf.predict(X[[i]])[0])
for v, c in sorted(zip(vals, cnt), key=lambda p: -p[1]):
    print(f"  {v:<11} {c:>3} trees")
""")
code("""
# Which features matter most? (share of the Gini improvement each feature provides)
order = np.argsort(rf.feature_importances_)[::-1][:10]
for k in order:
    print(f"  {names[k]:<10} {rf.feature_importances_[k]:.3f}")
""")

# ---------------------------------------------------------------- 9 save/predict
md("""
## 9. Save the models and make a prediction

Training happens once; the result is saved as a file. Using the model later is just loading that file.
We save two: one trained on everyone (for deployment) and one trained without the 9 test people (for the honest demo).
""")
code("""
import joblib, time

os.makedirs("models", exist_ok=True)
joblib.dump({"model": rf, "features": names, "test_users": [int(u) for u in test_people]},
            "models/demo_model.joblib", compress=3)            # trained on the training people only
full = RandomForest(n_estimators=300, random_state=SEED, class_weight="balanced").fit(X, y)
joblib.dump({"model": full, "features": names}, "models/activity_model.joblib", compress=3)   # all 36 people
for f in sorted(os.listdir("models")):
    print(f"{f:<26} {os.path.getsize('models/' + f) / 1e6:5.1f} MB")
""")
code("""
from mlengine.describe import summarize
print(summarize("models/demo_model.joblib")[:1500])    # the .joblib is binary; this shows what is inside
""")
code("""
from mlengine.predict import predict_window

window = examples["Walking"]                           # 80 readings x 3 axes
t0 = time.perf_counter()
answer = predict_window(window)
print(answer)
print(f"\\nPrediction took {1000 * (time.perf_counter() - t0):.0f} ms (including loading the model file)")
""")

# ---------------------------------------------------------------- 10 fall
md("""
## 10. The fall rule

WISDM has no recorded falls, so this is a **rule**, not a trained detector: flag when the peak is at least 2.5 g **and** the
last half second is still (near 1 g). It was only tested on a staged signal and must be validated on real fall data (SisFall, MobiAct).
""")
code("""
from mlengine.features import fall_risk, FALL_G_THRESHOLD, G
print(inspect.getsource(fall_risk))
""")
code("""
rng = np.random.default_rng(1)
fall = examples["Walking"].copy()
fall[29:] = np.array([0, G, 0]) + rng.normal(0, 0.15, (WINDOW - 29, 3))   # lying still afterwards
fall[28] = [6, 33, 11]                                                    # the impact

fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, (title, w) in zip(axes, [("Real walking window", examples["Walking"]), ("Staged fall", fall)]):
    r = fall_risk(w)
    ax.plot(np.linalg.norm(w, axis=1) / G, color="black")
    ax.axhline(FALL_G_THRESHOLD, ls="--", color="gray")
    ax.set_title(f"{title}: fall flag = {'YES' if r['fall_suspected'] else 'no'}")
    ax.set_xlabel("reading")
axes[0].set_ylabel("multiples of g")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 11 demo
md("""
## 11. The live demo

This starts our demo server inside Colab and shows the two phone screens below: **Mary's phone** (which senses movement, names the
activity and asks "Are you OK?" after a fall) and the **caregiver's phone** (which gets the alert). The activities are real recordings
from the 9 test people; the fall is staged; calls and notifications are simulated. Tick *Technical view* to see the graphs.

If the embedded view does not appear, run the next cell again, or use the link it prints.
""")
code("""
import time, urllib.request

PORT = 8000
server = subprocess.Popen([sys.executable, "-m", "mlengine.demo", "--port", str(PORT)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(60):                                    # wait up to ~2 minutes for the server to start
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=2)
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("The demo server did not start")
print("Demo server is running.")

if IN_COLAB:
    from google.colab import output
    output.serve_kernel_port_as_iframe(PORT, height=760)
else:
    print(f"Open http://localhost:{PORT} in your browser.")
""")
code("""
# Run this when you have finished with the demo
server.terminate()
print("Demo server stopped.")
""")
md("""
## Summary

* Cleaned 1,019,496 readings into 25,230 labelled 4-second windows with 32 features each.
* The models are our own NumPy code (`mlengine/models.py`) and match scikit-learn's versions within a fraction of a point.
* Tested on people the model had never seen: about **85% accuracy** and **0.778 macro F1**, against 39% and 0.09 for the baseline.
* The weak spot is Downstairs; the fall rule is a transparent, unvalidated first step.
* Reproduce anywhere: `pip install -r requirements.txt`, then `python -m mlengine.train` and `python -m mlengine.demo`.

Code: https://github.com/Moses374/SemProject_MLEngine
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.write(nb, "MLEngine_Colab.ipynb")
print("written MLEngine_Colab.ipynb with", len(cells), "cells")
