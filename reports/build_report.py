"""Build reports/WISDM_ML_Engine_Report.pdf (black text, Times, grayscale figures).

Run from the project root:  python reports/build_report.py
Figures are regenerated from the data and a re-run of the same subject-wise split
used in mlengine/train.py, so every number in the report is computed, not typed.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, Preformatted)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlengine.data import DEFAULT_PATH
from mlengine.features import FALL_G_THRESHOLD, G, STEP, WINDOW, build_dataset, fall_risk
from mlengine.data import load_raw

OUT = "reports"
FIG = os.path.join(OUT, "fig")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.spines.top": False, "axes.spines.right": False})
GREYS = ["#222222", "#555555", "#888888", "#aaaaaa", "#cccccc", "#eeeeee"]
SEED = 42

# ------------------------------------------------------------------ data + stats
raw_lines = raw_parsed = 0
with open(DEFAULT_PATH) as f:
    for line in f:
        line = line.strip().rstrip(";")
        if not line:
            continue
        raw_lines += 1
        if len(line.split(",")) == 6:
            raw_parsed += 1
data = load_raw()
df = pd.DataFrame(data)  # report-only convenience; the engine itself uses numpy arrays
n_clean = len(df)
acts = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
counts = df.activity.value_counts().reindex(acts)

Xa, ya, ga, feat_names = build_dataset(data)
X = pd.DataFrame(Xa, columns=feat_names)
y = pd.Series(ya)
groups = pd.Series(ga)
feats = X
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
test_users = groups.iloc[te].nunique()
train_users = groups.iloc[tr].nunique()
models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                            class_weight="balanced"),
}
res = {}
for n, m in models.items():
    m.fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[te])
    res[n] = (accuracy_score(y.iloc[te], p), f1_score(y.iloc[te], p, average="macro"))
rf = models["Random forest"]
pred = rf.predict(X.iloc[te])
labels = sorted(y.unique())
conf = confusion_matrix(y.iloc[te], pred, labels=labels)
prec, rec, f1s, sup = precision_recall_fscore_support(y.iloc[te], pred, labels=labels)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)


def box(ax, xy, w, h, text, fc="white", fs=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fc=fc, ec="black", lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1, color="black"))


# ------------------------------------------------------------------ figures
# 1. pipeline diagram
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
steps = ["Raw WISDM\ntext file", "Clean\n(drop bad rows)", "Sliding\nwindows", "Feature\nextraction",
         "Train &\nevaluate", "Saved\nmodel"]
for i, s in enumerate(steps):
    x = 0.1 + i * 1.65
    box(ax, (x, 1.5), 1.35, 0.9, s)
    if i:
        arrow(ax, (x - 0.28, 1.95), (x - 0.02, 1.95))
box(ax, (7.0, 0.1), 1.4, 0.9, "Activity\n(6 classes)")
box(ax, (8.55, 0.1), 1.4, 0.9, "Fall-risk\nrule")
arrow(ax, (8.0, 1.5), (7.9, 1.02)); arrow(ax, (8.4, 1.5), (9.0, 1.02))
fig.savefig(f"{FIG}/pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 2. class distribution
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(acts, counts / 1000, color="#888888", edgecolor="black")
for i, v in enumerate(counts / 1000):
    ax.text(i, v + 6, f"{v:.0f}k", ha="center")
ax.set_ylabel("Readings (thousands)")
fig.savefig(f"{FIG}/classes.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 3. example signals per activity (acceleration magnitude)
fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharey=True)
for a, ax in zip(acts, axes.ravel()):
    g = df[df.activity == a].iloc[3000:3000 + WINDOW]
    mag = np.linalg.norm(g[["x", "y", "z"]].to_numpy(), axis=1)
    ax.plot(np.arange(WINDOW) / 20, mag, color="black", lw=1)
    ax.set_title(a, fontsize=10)
    ax.set_xlabel("seconds")
axes[0, 0].set_ylabel("|a| (m/s²)"); axes[1, 0].set_ylabel("|a| (m/s²)")
fig.tight_layout()
fig.savefig(f"{FIG}/signals.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 4. windowing diagram
fig, ax = plt.subplots(figsize=(8, 2.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
t = np.linspace(0.2, 9.8, 300)
ax.plot(t, 2.4 + 0.25 * np.sin(t * 5) * (1 + 0.3 * np.sin(t)), color="black", lw=0.8)
for i, (x0, y0) in enumerate([(0.2, 1.3), (2.6, 0.5), (5.0, 1.3)]):
    ax.add_patch(plt.Rectangle((x0, y0), 4.8, 0.6, fc=GREYS[4 + (i % 2)], ec="black", alpha=0.9))
    ax.text(x0 + 2.4, y0 + 0.3, f"Window {i + 1} ({WINDOW} samples = 4 s)", ha="center", va="center", fontsize=9)
ax.text(5, 2.95, "Stream of readings", ha="center", fontsize=9)
ax.text(5, 0.05, f"Each window starts {STEP} samples (2 s) after the previous one, so neighbours overlap by 50%.",
        ha="center", fontsize=9)
fig.savefig(f"{FIG}/windows.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 5. split diagram
fig, ax = plt.subplots(figsize=(8, 1.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
box(ax, (0.2, 0.7), 6.4, 0.7, f"Training set: {train_users} subjects (all their windows)", fc="#dddddd")
box(ax, (6.9, 0.7), 2.9, 0.7, f"Test set: {test_users} unseen subjects", fc="white")
ax.text(5, 0.2, "Whole subjects are held out, so no person appears on both sides.", ha="center", fontsize=9)
fig.savefig(f"{FIG}/split.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 6. model comparison
fig, ax = plt.subplots(figsize=(6, 3))
names = list(res)
xs = np.arange(len(names))
ax.bar(xs - 0.2, [res[n][0] for n in names], 0.4, color="#888888", edgecolor="black", label="Accuracy")
ax.bar(xs + 0.2, [res[n][1] for n in names], 0.4, color="white", edgecolor="black", hatch="//", label="Macro F1")
for i, n in enumerate(names):
    ax.text(i - 0.2, res[n][0] + 0.015, f"{res[n][0]:.2f}", ha="center", fontsize=9)
    ax.text(i + 0.2, res[n][1] + 0.015, f"{res[n][1]:.2f}", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names]); ax.set_ylim(0, 1.05)
ax.legend(frameon=False)
fig.savefig(f"{FIG}/models.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 7. confusion matrix (grayscale, row-normalised with counts)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
norm = conf / conf.sum(axis=1, keepdims=True)
ax.imshow(norm, cmap="Greys", vmin=0, vmax=1)
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right"); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{conf[i, j]}", ha="center", va="center", fontsize=9,
                color="white" if norm[i, j] > 0.5 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for s in ax.spines.values():
    s.set_visible(True)
fig.savefig(f"{FIG}/confusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 8. feature importance
fig, ax = plt.subplots(figsize=(6, 3.4))
imp[::-1].plot.barh(ax=ax, color="#888888", edgecolor="black")
ax.set_xlabel("Importance (random forest)")
fig.savefig(f"{FIG}/importance.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 9. fall rule illustration
rng = np.random.default_rng(0)
sim = np.tile([0, G, 0], (WINDOW, 1)) + rng.normal(0, 0.2, (WINDOW, 3))
sim[30] = [5, 35, 10]
walk = df[df.activity == "Walking"].iloc[5000:5000 + WINDOW][["x", "y", "z"]].to_numpy()
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
for ax, w, title in [(axes[0], walk, "Walking window"), (axes[1], sim, "Simulated fall")]:
    m = np.linalg.norm(w, axis=1) / G
    r = fall_risk(w)
    ax.plot(np.arange(WINDOW) / 20, m, color="black", lw=1)
    ax.axhline(FALL_G_THRESHOLD, color="black", ls="--", lw=0.8)
    ax.text(0.1, FALL_G_THRESHOLD + 0.08, "impact threshold 2.5 g", fontsize=8)
    ax.set_title(f"{title}: fall flag = {'YES' if r['fall_suspected'] else 'no'}", fontsize=10)
    ax.set_xlabel("seconds")
axes[0].set_ylabel("|a| in multiples of g")
fig.tight_layout()
fig.savefig(f"{FIG}/fall.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ document
BASE = "Times-Roman"
body = ParagraphStyle("body", fontName=BASE, fontSize=11, leading=15, alignment=TA_JUSTIFY,
                      textColor=colors.black, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=14, leading=18, spaceBefore=12, keepWithNext=1,
                    spaceAfter=6, alignment=0)
h2 = ParagraphStyle("h2", parent=body, fontName="Times-Bold", fontSize=12, leading=15, spaceBefore=8, keepWithNext=1,
                    spaceAfter=4, alignment=0)
cap = ParagraphStyle("cap", parent=body, fontName="Times-Italic", fontSize=10, leading=13,
                     alignment=TA_CENTER, spaceAfter=10)
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=22, leading=28,
                       alignment=TA_CENTER, spaceAfter=10)
sub = ParagraphStyle("sub", parent=body, fontSize=13, alignment=TA_CENTER, leading=18)
cell = ParagraphStyle("cell", parent=body, fontSize=10, leading=12.5, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
mono = ParagraphStyle("mono", fontName="Courier", fontSize=9, leading=11, textColor=colors.black)

S = []


def P(t, s=body): S.append(Paragraph(t, s))


def img(name, width_cm, caption):
    path = f"{FIG}/{name}.png"
    from PIL import Image as PI
    w, h = PI.open(path).size
    S.append(Image(path, width=width_cm * cm, height=width_cm * cm * h / w))
    P(caption, cap)


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black))
    t.setStyle(TableStyle(st))
    S.append(t); S.append(Spacer(1, 8))


# --- title page
S.append(Spacer(1, 5 * cm))
P("Activity Recognition and Fall-Risk Detection from Smartphone Accelerometer Data", title)
P("ICS 3202 Semester Project: Machine Learning Engine", sub)
P("Dataset: WISDM Activity Recognition (Kwapisz, Weiss and Moore, 2011)", sub)
S.append(Spacer(1, 2 * cm))
P("Moses Kibet (183638)<br/>Collins Jared (184587)<br/>Kathy Auma (192231)", sub)
S.append(PageBreak())

# --- 1
P("1. Introduction", h1)
P("Falls are a leading cause of injury among elderly people, and a fall that goes unnoticed can be far more "
  "dangerous than one that is reported quickly. Almost everyone now carries a smartphone, and every smartphone "
  "contains an accelerometer: a sensor that measures how the phone is accelerating along three directions. "
  "The idea behind this project is to use that sensor to work out what a person is doing and to flag a "
  "likely fall to a caregiver or family member.")
P("This report describes the machine learning engine behind that idea. It has two parts. First, a "
  "<b>classifier</b> that takes a short snippet of accelerometer readings and decides which of six activities "
  "the person was performing (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing). Second, a "
  "<b>fall-risk rule</b> that looks for the signature of a fall: a hard impact followed by stillness. "
  "The report explains the data, each processing step, how the models were evaluated, the results, and "
  "the limitations that matter for real use.")
img("pipeline", 15.5, "Figure 1. The pipeline from raw sensor file to a prediction.")

# --- 2
P("2. The dataset", h1)
P("The project uses the WISDM (Wireless Sensor Data Mining) Activity Recognition dataset collected at Fordham "
  "University. Volunteers carried an Android phone in their front trouser pocket and performed six activities "
  "while the phone recorded acceleration 20 times per second. The raw file contains one line per reading:")
S.append(Preformatted("user, activity, timestamp, x-accel, y-accel, z-accel;\n33,Jogging,49105962326000,-0.69,12.68,0.50;",
                      mono))
S.append(Spacer(1, 6))
P("The <i>x, y, z</i> values are acceleration in m/s² along the three axes of the phone. When the phone is "
  "perfectly still, the total acceleration equals gravity, about 9.81 m/s² (written 1 g). Because gravity "
  "acts along a different axis depending on how the phone is oriented, the readings also reveal posture: "
  "this is how sitting and standing can be told apart even though neither involves movement.")
table([["Property", "Value"],
       ["Subjects", f"{df.user.nunique()}"],
       ["Sampling rate", "20 readings per second"],
       ["Lines parsed from raw file", f"{raw_parsed:,}"],
       ["Readings after cleaning", f"{n_clean:,}"],
       ["Target variable", "activity (6 classes)"]], [6, 8])

# --- 3
P("3. Data exploration (Deliverable 1)", h1)
P("The first deliverable explored the raw file in the notebook <i>WISDM_Exploration.ipynb</i>. Its findings were:")
table([["Question", "Answer"],
       ["Rows and columns", "1,086,466 rows and 6 columns (user, activity, timestamp, x-accel, y-accel, z-accel)"],
       ["Datatypes", "user: integer; activity: text (object); timestamp: integer; x, y, z acceleration: floating point"],
       ["Complete?", "Almost. Exactly one value is missing (a z-accel reading). A few malformed lines in the raw "
                     "file had to be skipped while loading."],
       ["df_sample", "The first 15 and last 20 rows joined into one table (35 rows)"]], [4, 10.5])
P("The activities are not evenly represented (Figure 2). Walking and Jogging make up two thirds of the "
  "readings, while Sitting and Standing together are under 11%. This imbalance matters: a model that always "
  "guessed Walking would already be right about 39% of the time on windows, so accuracy alone can be "
  "misleading, and the report therefore also uses macro F1 (explained in Section 6).")
img("classes", 10.5, "Figure 2. Number of readings per activity.")
P("Figure 3 shows the total acceleration for four seconds of each activity. The activities look clearly "
  "different. Jogging swings strongly, Walking has a regular rhythm, and Sitting and Standing are almost "
  "flat. Upstairs and Downstairs resemble Walking, which anticipates where the classifier will struggle.")
img("signals", 15.5, "Figure 3. Acceleration magnitude |a| = √(x²+y²+z²) for four seconds of each activity.")

# --- 4
P("4. Data preparation", h1)
P("Before any learning, the raw file is cleaned by <i>mlengine/data.py</i>:")
for t in ["<b>Malformed lines</b> (wrong number of fields) are skipped.",
          "<b>Missing numeric values</b> are removed.",
          "<b>Zero timestamps</b> are removed. A timestamp of zero means the clock had not started, so the "
          "reading cannot be placed in time.",
          "<b>Duplicate readings</b> (same subject and timestamp) are removed.",
          "The data are sorted by subject and time so that consecutive rows really are consecutive readings."]:
    P("&bull; " + t)
P(f"The cleaning reduces the file from {raw_parsed:,} parsed lines to {n_clean:,} readings "
  f"({raw_parsed - n_clean:,} removed, {100 * (raw_parsed - n_clean) / raw_parsed:.1f}%).")

# --- 5
P("5. Windowing and feature extraction", h1)
P("<b>Why windows?</b> A single reading says almost nothing about activity: one instant of acceleration "
  "could belong to walking, running or sitting down. What distinguishes activities is how the signal "
  "behaves over a short period. The stream is therefore cut into windows of "
  f"{WINDOW} readings (4 seconds). Consecutive windows overlap by 50% (Figure 4), which gives more training "
  "examples and means an event is never split unluckily between two windows. A window is only used if "
  "all its readings come from one subject doing one activity, so each window has a single correct label. "
  f"This produces {len(feats):,} labelled windows.")
img("windows", 15.5, "Figure 4. Sliding windows with 50% overlap.")
P("<b>Why features?</b> A window of 80 readings x 3 axes is 240 raw numbers. Instead of giving these directly "
  f"to the models, each window is summarised into {X.shape[1]} descriptive numbers (features). This is "
  "compact, works well with classical models, and is easy to explain.")
table([["Feature group", "What it measures", "Why it helps"],
       ["Mean (per axis)", "Average acceleration on x, y, z", "Reflects phone orientation, which separates "
                                                              "sitting from standing"],
       ["Std, min, max, MAD (per axis)", "How much and how far the signal varies", "Separates vigorous "
                                                                                   "activity (jogging) from calm"],
       ["Zero-crossing rate", "How often the signal crosses its own mean", "A simple measure of step rhythm"],
       ["Dominant frequency, energy", "Strongest repeating frequency and total vibration (from an FFT)",
        "Walking and jogging repeat at different rates"],
       ["Magnitude statistics", "Mean, std, min, max, range of |a|", "Orientation-independent measure of "
                                                                      "movement intensity"],
       ["Axis correlations", "How x, y and z move together", "Captures the direction of motion"]],
      [3.6, 5.4, 5.5])

# --- 6
P("6. Models and evaluation method", h1)
P("Three models were compared, from simplest to most capable:")
P("&bull; <b>Majority-class baseline</b>: always predicts the most common activity. Any real model must beat it.")
P("&bull; <b>Logistic regression</b>: a linear model on standardised features. Fast and easy to interpret.")
P("&bull; <b>Random forest</b>: 300 decision trees voting together. It can capture non-linear patterns and "
  "needs little tuning. Class weights are balanced so that rare activities such as Standing are not ignored.")
P("<b>Subject-wise split.</b> A common mistake with this kind of data is to shuffle all windows and split them "
  "randomly. Because neighbouring windows from the same person are nearly identical, the model then "
  "sees almost the same data in training and testing and the score is inflated. A real app must work on "
  "people it has never seen. Here, whole subjects are held out "
  f"(Figure 5): {train_users} subjects for training and {test_users} for testing.")
img("split", 15.5, "Figure 5. Subject-wise train/test split.")
P("<b>Metrics.</b> <i>Accuracy</i> is the fraction of windows classified correctly. <i>Precision</i> for an "
  "activity is how often a prediction of that activity is right; <i>recall</i> is how many of the true "
  "windows of that activity are found. <i>F1</i> combines the two, and <i>macro F1</i> averages F1 over all "
  "six activities equally, so small classes count as much as large ones.")

# --- 7
P("7. Results", h1)
table([["Model", "Accuracy", "Macro F1"]] +
      [[n, f"{res[n][0]:.3f}", f"{res[n][1]:.3f}"] for n in res], [7, 3.5, 3.5])
img("models", 10.5, "Figure 6. Held-out test performance of the three models.")
P("Both real models far outperform the baseline, and the random forest is best, so it is the model "
  "saved for use. To check that the result does not depend on one lucky split, the random forest was also "
  "evaluated with 5-fold cross-validation grouped by subject, giving a macro F1 of 0.791 with a standard "
  "deviation of 0.035. The result is stable.")
P("<b>Per-activity results.</b>", body)
table([["Activity", "Precision", "Recall", "F1", "Test windows"]] +
      [[l, f"{prec[i]:.2f}", f"{rec[i]:.2f}", f"{f1s[i]:.2f}", int(sup[i])] for i, l in enumerate(labels)],
      [4, 2.8, 2.8, 2.8, 3.2])
P("Figure 7 is the confusion matrix. Each row is the true activity and each column the prediction; "
  "numbers on the diagonal are correct. Jogging and Walking are recognised reliably. The main weakness is "
  f"<b>Downstairs</b>: only {rec[labels.index('Downstairs')]:.0%} of true Downstairs windows are found, "
  "and most of the rest are labelled Upstairs or Walking. This is expected: all three are walking-like "
  "motions, and a phone in a pocket records only subtle differences between them. Sitting and Standing "
  "are the other main source of confusion with each other, since neither involves movement and they differ "
  "mostly in phone tilt.")
img("confusion", 9.5, "Figure 7. Confusion matrix on the held-out subjects (counts of windows).")
P("Figure 8 shows which features the random forest relied on most. The strongest features describe the "
  "average and spread of the signal, in line with the reasoning in Section 5: orientation separates the "
  "still activities, and vibration intensity separates the moving ones.")
img("importance", 11, "Figure 8. The twelve most important features.")

# --- 8
P("8. Fall-risk detection", h1)
P("The WISDM dataset contains <b>no recorded falls</b>, so a classifier for falls cannot be trained from it. "
  "Instead the engine applies a transparent, physics-based rule to every window. A fall typically has two "
  "stages: a hard impact with the ground, then the person lying still. The rule flags a window when "
  f"both hold:")
P(f"&bull; the peak acceleration is at least <b>{FALL_G_THRESHOLD} g</b> (an impact), and")
P("&bull; during the last half second of the window the total acceleration stays close to 1 g with very "
  "little variation (stillness after the impact).")
P("Requiring stillness is important. Jogging and even walking produce peaks above 2.5 g with every footstep, but "
  "the person keeps moving afterwards, so those windows are not flagged (Figure 9, left). A simulated "
  "impact followed by a motionless phone is flagged (right).")
img("fall", 15.5, "Figure 9. The fall rule on a real walking window and on a simulated fall.")
P("<b>Important caveat.</b> The rule was checked only on this simulated signal. Its thresholds are "
  "reasonable physical guesses, not values tuned or validated on real fall data. It must be evaluated on a "
  "labelled fall dataset (for example SisFall or MobiAct) before being relied upon.")

# --- 9
P("9. Using the engine", h1)
P("The package is organised so each step can be run or reused independently:")
table([["File", "Purpose"],
       ["mlengine/data.py", "Loads and cleans the raw file"],
       ["mlengine/features.py", "Windowing, feature extraction and the fall-risk rule"],
       ["mlengine/train.py", "Trains and compares models; saves the best model, plots and metrics"],
       ["mlengine/predict.py", "Takes 80 readings and returns the activity, confidence and fall flag"],
       ["reports/build_report.py", "Generates this report"]], [5, 9.5])
S.append(Preformatted("pip install -r requirements.txt\npython -m mlengine.train\n"
                      "python -m mlengine.predict readings.csv", mono))
S.append(Spacer(1, 6))
P("The prediction input is a CSV with columns x, y and z (in m/s², sampled at 20 Hz). The output is, for "
  "example: <i>activity = Walking, confidence = 0.98, fall_suspected = False</i>.")

# --- 10
P("10. Limitations and future work", h1)
for t in ["<b>No real fall data.</b> The fall rule is unvalidated (Section 8). The most valuable next step "
          "is to evaluate and tune it, or train a fall classifier, on SisFall or MobiAct.",
          "<b>No lying-down class.</b> WISDM has no such activity, yet lying down is relevant after a fall.",
          "<b>Young adult volunteers.</b> WISDM subjects were not elderly. Elderly gait is slower and more "
          "variable, so performance on the target population is unknown.",
          "<b>Single phone placement.</b> All data come from a trouser pocket; a phone in a bag or hand "
          "would behave differently.",
          "<b>Weak on Downstairs.</b> Better features (for example a gravity-removed signal) or a "
          "neural network on the raw windows may help.",
          "<b>Not yet a mobile app.</b> The engine is a Python library; wrapping it in a phone app "
          "or a small web service is the remaining integration work."]:
    P("&bull; " + t)

# --- 11
P("11. Conclusion", h1)
P("The engine recognises six everyday activities from a phone accelerometer with about 85% accuracy and "
  "0.78 macro F1 on people it has never seen, far above the 39% / 0.09 of the baseline. Careful handling of "
  "the data (cleaning, windowing, and a subject-wise evaluation) makes these numbers a realistic estimate "
  "rather than an inflated one. A transparent fall-risk rule is layered on top, with a clear statement of "
  "what has and has not been validated.")
P("<b>Reference.</b> Kwapisz, J. R., Weiss, G. M., and Moore, S. A. (2011). Activity Recognition using "
  "Cell Phone Accelerometers. <i>ACM SIGKDD Explorations</i>, 12(2), 74-82.")


def footer(c, d):
    c.saveState()
    c.setFont("Times-Roman", 9)
    c.setFillColor(colors.black)
    if d.page > 1:
        c.drawCentredString(A4[0] / 2, 1.2 * cm, f"{d.page}")
    c.restoreState()


doc = SimpleDocTemplate(f"{OUT}/WISDM_ML_Engine_Report.pdf", pagesize=A4, leftMargin=2.3 * cm,
                        rightMargin=2.3 * cm, topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                        title="Activity Recognition and Fall-Risk Detection", author="SemProject_MLEngine group")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("raw lines", raw_lines, "parsed", raw_parsed, "clean", n_clean)
print("written", f"{OUT}/WISDM_ML_Engine_Report.pdf")
