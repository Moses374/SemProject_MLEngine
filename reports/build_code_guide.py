"""Build reports/Code_Walkthrough.pdf: how to explain every file of the project, with the real code.

Run from the project root:  python -m reports.build_code_guide
Code excerpts are copied from the actual source files at build time and the numbers are read from
reports/metrics.json and the saved model, so the document cannot drift from the repo.
"""
import json
import time

import joblib
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

from mlengine.features import fall_risk, window_features
from mlengine.predict import predict_window


# ------------------------------------------------------------------ helpers
def grab(path, start, end=None):
    """Source text from the line containing `start` up to (not including) the next line containing `end`."""
    lines = open(path, encoding="utf-8").read().splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    j = len(lines) if end is None else next(k for k in range(i + 1, len(lines)) if end in lines[k])
    return "\n".join(lines[i:j]).rstrip()


def nlines(path):
    return len(open(path, encoding="utf-8").read().splitlines())


MAXCOL = 104


def fit(code):
    out = []
    for l in code.splitlines():
        while len(l) > MAXCOL:
            cut = l.rfind(" ", 0, MAXCOL)
            cut = cut if cut > 40 else MAXCOL
            out.append(l[:cut])
            l = "    " + l[cut:].lstrip()
        out.append(l)
    return "\n".join(out)


body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY,
                      textColor=colors.black, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=10, spaceAfter=6,
                    alignment=0, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Times-Bold", fontSize=12.5, leading=16, spaceBefore=9, spaceAfter=3,
                    alignment=0, keepWithNext=1)
lab = ParagraphStyle("lab", parent=body, fontName="Times-Bold", spaceAfter=1, spaceBefore=3, alignment=0,
                     keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=24, leading=30, alignment=TA_CENTER,
                       spaceAfter=10)
sub = ParagraphStyle("sub", parent=body, fontSize=13, alignment=TA_CENTER, leading=18)
cell = ParagraphStyle("cell", parent=body, fontSize=10, leading=12.5, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
code_style = ParagraphStyle("code", fontName="Courier", fontSize=7.4, leading=9.2, textColor=colors.black,
                            backColor=colors.Color(0.95, 0.95, 0.95), borderColor=colors.black, borderWidth=0.4,
                            borderPadding=4, spaceBefore=6, spaceAfter=10, leftIndent=4, rightIndent=4)
S = []


def P(t, st=body):
    S.append(Paragraph(t, st))


def B(items):
    for t in items:
        P("&bull; " + t)


def CODE(text):
    S.append(Preformatted(fit(text), code_style))


def C(t):
    """Inline code."""
    return f"<font face='Courier'>{t}</font>"


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black)]))
    S.append(t)
    S.append(Spacer(1, 8))


F = "mlengine"
files = {n: f"{F}/{n}" for n in ["data.py", "features.py", "models.py", "evaluation.py", "train.py", "predict.py",
                                 "describe.py", "demo.py", "demo_page.html"]}

# ------------------------------------------------------------------ facts read from the repo
M = json.load(open("reports/metrics.json"))
R = M["results"]
acc = {k: R[k]["accuracy"] for k in ("baseline (majority)", "logistic regression", "random forest")}
f1 = {k: R[k]["macro_f1"] for k in ("baseline (majority)", "logistic regression", "random forest")}
cv_mean = R["random forest 5-fold subject-wise CV"]["macro_f1_mean"]
cv_std = R["random forest 5-fold subject-wise CV"]["macro_f1_std"]
report_rows = {}
for line in M["report"].splitlines():
    p = line.split()
    if len(p) == 5 and p[0] in ("Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"):
        report_rows[p[0]] = (float(p[1]), float(p[2]), float(p[3]), int(p[4]))
bundle = joblib.load("models/activity_model.joblib")
rf = bundle["model"]
n_nodes = sum(t.node_count for t in rf.trees_)
avg_depth = float(np.mean([t.depth_ for t in rf.trees_]))
avg_leaves = float(np.mean([t.n_leaves for t in rf.trees_]))
w_all = np.genfromtxt("reports/samples/sample_walking.csv", delimiter=",", names=True)
win = np.column_stack((w_all["x"], w_all["y"], w_all["z"]))
feat = window_features(win)
Xw = np.array([[feat[n] for n in bundle["features"]]])
rf.predict_proba(Xw)
t0 = time.perf_counter()
for _ in range(20):
    rf.predict_proba(Xw)
ms = (time.perf_counter() - t0) / 20 * 1000
ms_txt = f"about {round(ms, -1):.0f} ms"
ans = predict_window(win)
fall = fall_risk(win)
votes = np.array([t.predict_proba_encoded(Xw)[0].argmax() for t in rf.trees_])
top_class = list(rf.classes_).index(ans["activity"])
n_top = int((votes == top_class).sum())
SK = "scikit-learn"

# ------------------------------------------------------------------ cover
S.append(Spacer(1, 4.5 * cm))
P("Code Walkthrough", title)
P("How to explain every part of the project, with the real code", sub)
P("Activity Recognition and Fall-Risk Detection from Smartphone Accelerometer Data", sub)
S.append(Spacer(1, 1.5 * cm))
P("Moses Kibet (183638)<br/>Collins Jared (184587)<br/>Kathy Auma (192231)", sub)
S.append(Spacer(1, 1.5 * cm))
P("Every code box in this document is copied from the project files when the document is built, and every number is "
  "read from the saved results, so it matches what is on GitHub: github.com/Moses374/SemProject_MLEngine", sub)
P("This version describes the project after the models were rewritten from scratch: no machine-learning library "
  "is used.", sub)
S.append(PageBreak())

# ------------------------------------------------------------------ 1 how to explain code
P("1. How to explain code to people who will not read it", h1)
P("Nobody in the room will read your code line by line, and you should not either. A lecturer wants to know three things "
  "about each piece: <b>what is it for</b>, <b>how does it work in outline</b>, and <b>why did you do it that way</b>. "
  "Use this order every time.")
table([["Step", "What you say", "Example (the data loader)"],
       ["1. Purpose (one sentence)", "What goes in, what comes out.",
        "\"This function reads the raw sensor file and gives back clean arrays of numbers.\""],
       ["2. Walk (3 to 5 points)", "The main stages, in plain words. Point at the one or two key lines.",
        "\"It skips broken lines, throws away zero timestamps and duplicates, then sorts by person and time.\""],
       ["3. Why (the decision)", "The reason, ideally with a number or a consequence.",
        "\"Zero timestamps mean the phone clock had not started, so those readings cannot be placed in time. "
        "Cleaning removed 6.2% of the file.\""]],
      [3.6, 5.6, 7.8])
P("Rules that make explaining easier", h2)
B(["<b>Use one real example.</b> Follow a single window through the code (Section 4) instead of describing code in the abstract.",
   "<b>Show the output, not only the code.</b> Run it and show the printed result; a result is more convincing than a function.",
   "<b>Name the idea before the syntax.</b> Say \"it slides a window along the stream\", then point at the loop.",
   "<b>Say what is yours.</b> Everything in this project is our own code. The classifiers, the splitting and the scoring were "
   f"written from scratch with NumPy; there is no machine-learning library such as {SK}. Say it once, plainly, and then "
   "be ready to explain any line.",
   "<b>If you do not know a line, explain its job.</b> \"That line makes the result repeatable\" is better than guessing the syntax.",
   "<b>Keep the files straight:</b> data.py cleans, features.py describes windows, models.py learns, evaluation.py splits "
   "and scores, train.py runs the experiment, predict.py uses the result, describe.py shows what is inside, demo.py serves the demo."])

P("The whole codebase in 90 seconds", h2)
P("\"Our project is a small Python package called mlengine, and it uses no machine-learning library: only numpy for the "
  "maths and joblib to train trees in parallel and to save models. <b>data.py</b> loads and cleans the raw WISDM file. "
  "<b>features.py</b> cuts the data into four-second windows and turns each window into 32 numbers, and it also holds the "
  "fall rule. <b>models.py</b> is the machine learning: we wrote a majority baseline, a logistic regression and a random "
  "forest of decision trees ourselves. <b>evaluation.py</b> splits the people into training and test groups, so nobody "
  "appears on both sides, and computes the scores. <b>train.py</b> ties these together: it trains the three models, tests "
  "them on people they have never seen, and saves the best one to a file. <b>predict.py</b> loads that file and classifies "
  "a new window, and <b>describe.py</b> prints what is inside the file, because the file itself is binary. <b>demo.py</b> is "
  "a small web server that feeds real test windows to the model and sends the answer to the demo page.\"")

# ------------------------------------------------------------------ 2 map
P("2. The map: files, size and who explains what", h1)
rows = [["File", "Lines", "Its job", "Explains it"],
        ["mlengine/data.py", nlines(files["data.py"]), "Load and clean the raw file", "Moses"],
        ["mlengine/features.py", nlines(files["features.py"]), "Windows, 32 features, fall rule",
         "Collins (windows, features) and Kathy (fall rule)"],
        ["mlengine/models.py", nlines(files["models.py"]), "Baseline, logistic regression, decision tree, random forest",
         "Collins (tree and forest) and Kathy (baseline and logistic regression)"],
        ["mlengine/evaluation.py", nlines(files["evaluation.py"]), "Person-wise split, cross-validation folds, scores",
         "Moses"],
        ["mlengine/train.py", nlines(files["train.py"]), "Run the experiment, compare models, save", "Collins"],
        ["mlengine/predict.py", nlines(files["predict.py"]), "Load a saved model and classify one window", "Moses"],
        ["mlengine/describe.py", nlines(files["describe.py"]), "Show what is inside the binary model files", "Moses"],
        ["mlengine/demo.py", nlines(files["demo.py"]), "Local web server behind the demo", "Kathy"],
        ["mlengine/demo_page.html", nlines(files["demo_page.html"]),
         "The two-phone demo page (HTML, CSS, JavaScript)", "Kathy"]]
table(rows, [4.0, 1.4, 6.4, 5.2])
eng = sum(nlines(files[n]) for n in ["data.py", "features.py", "models.py", "evaluation.py", "train.py", "predict.py",
                                     "describe.py"])
P(f"The engine (the seven Python files) is about {eng} lines, of which models.py is the largest. You can swap who "
  "explains which file; just keep each file with one person.")

# ------------------------------------------------------------------ 3 file by file
P("3. File by file", h1)

# data.py
P("data.py: loading and cleaning (Moses)", h2)
P("<b>Purpose:</b> read the raw WISDM text file and return clean numpy arrays: user, activity, timestamp, x, y, z. "
  "It is the only code that touches the raw file, so every later step can trust the data.")
CODE(grab(files["data.py"], "    user, act, ts, xyz = [], [], [], []"))
P("Walk", lab)
B(["The loop reads the file one line at a time, strips the trailing semicolon and splits on commas; "
   f"{C('len(parts) != 6')} skips any line that does not have exactly 6 pieces.",
   f"{C('except ValueError: continue')} skips a line with an unreadable or missing number. That is how the "
   "one missing z value in the file is dropped.",
   f"{C('np.isfinite(...)')} rejects not-a-number values, and {C('t &lt;= 0')} rejects zero timestamps.",
   f"{C('np.unique(..., return_index=True)')} finds the first occurrence of each (user, timestamp) pair, "
   f"so duplicate readings are dropped; {C('np.sort(first)')} keeps them in file order.",
   f"{C('np.lexsort((ts, user))')} sorts by user and then time, so consecutive rows are consecutive readings."])
P("Say this", lab)
P("\"We loaded 1,086,466 usable lines and kept 1,019,496 readings from 36 people. The lines we removed were zero timestamps "
  "(the phone clock had not started), duplicates and the one missing value.\"")
P("If asked why", lab)
P("Windows must be made from readings that are really consecutive in time. Without sorting and de-duplicating, a window could "
  "contain jumps or repeated readings and the features would be wrong.")

# features: make_windows
P("features.py: windows (Collins)", h2)
P("<b>Purpose:</b> cut each person's recording of each activity into overlapping 80-reading (4-second) windows.")
CODE(grab(files["features.py"], "def make_windows", "def _corr"))
P("Walk", lab)
B([f"{C('np.unique(acts, return_inverse=True)')} turns the six activity names into the numbers 0 to 5.",
   f"{C('key = users * len(names) + codes')} makes one number for every (person, activity) pair. "
   "Think of it as a labelled bucket: \"person 33 jogging\".",
   f"The loop visits each bucket; {C('rows = xyz[key == k]')} pulls out all readings of that bucket, still in time order.",
   f"{C('range(0, len(rows) - window + 1, step)')} slides along: start at 0, move 40 readings each time, "
   "stop when a full 80 no longer fit.",
   f"{C('yield')} hands back one window at a time instead of building a huge list (this is called a generator)."])
P("Say this", lab)
P("\"Windows are cut inside one person doing one activity, so a window never mixes two activities and always has a single correct "
  "label. Each window starts 2 seconds after the last, so neighbouring windows overlap by half. This gives 25,230 windows.\"")

# features: window_features
P("features.py: the 32 features (Collins)", h2)
P("<b>Purpose:</b> summarise one window of 240 numbers (80 readings x 3 axes) into 32 numbers the model can learn from.")
CODE(grab(files["features.py"], "def window_features", '    f["mag_mean"]'))
P("Walk", lab)
B(["For each axis (x, y, z) it computes <b>8 numbers</b>: mean (the phone's orientation), standard deviation and MAD "
   "(how much it varies), min and max, the zero-crossing rate (how often the signal crosses its average, a measure of step rhythm), "
   "the dominant frequency and the energy.",
   f"{C('np.fft.rfft')} is a Fourier transform: it splits the signal into repeating frequencies, so "
   f"{C('spec[1:].argmax() + 1')} is the strongest repeating rate. Walking and jogging repeat at different rates.",
   "After the loop (not shown) come 5 statistics of the total magnitude and 3 correlations between the axes. "
   "<b>8 x 3 + 5 + 3 = 32.</b>"])
P("Say this", lab)
P("\"The same function is used for training and for live prediction, so the model always sees features computed exactly the "
  "same way.\"")
P("If asked why", lab)
P("Raw readings are noisy and there are too many of them. Hand-designed features are compact, fast and explainable: orientation "
  "separates sitting from standing, and vibration strength and rhythm separate walking from jogging.")
CODE(grab(files["features.py"], "def build_dataset", "def fall_risk"))
P(f"<b>build_dataset</b> loops over every window, calls {C('window_features')}, and stacks the results into a table "
  f"{C('X')} (one row per window, 32 columns), with {C('y')} (the correct activity) and "
  f"{C('users')} (which person each window came from; this is what makes fair testing possible).")

# fall_risk
P("features.py: the fall rule (Kathy)", h2)
P("<b>Purpose:</b> decide whether a window looks like a fall: a hard impact followed by stillness.")
CODE(grab(files["features.py"], "def fall_risk"))
P("Walk", lab)
B([f"{C('np.linalg.norm(w, axis=1) / G')} turns each reading into the total strength of movement, measured in g "
   "(1 g is the pull of gravity, so a still phone reads about 1).",
   f"{C('peak = argmax')} finds the strongest moment. <b>impact</b> is true if it is at least 2.5 g.",
   "<b>still_after</b> is true if there are at least 10 readings after the peak and the <b>last 10 readings (the final half second)</b> "
   "average within 0.15 g of 1 g with almost no variation.",
   "<b>fall_suspected</b> needs both. Walking and jogging often exceed 2.5 g on a footstep but keep moving, so they are not flagged."])
P("Say this", lab)
P("\"This is a rule, not machine learning, because WISDM contains no falls to learn from. We tested it only on a staged signal, so "
  "it must be validated on real fall data before real use.\"")

# ---- models.py
P("models.py: the machine learning, written from scratch", h2)
P("<b>Purpose:</b> the three classifiers we compare. All three have the same small interface so the rest of the engine "
  f"treats them alike: {C('fit(X, y)')} learns from labelled windows, {C('predict(X)')} returns activities, and "
  f"{C('predict_proba(X)')} returns a probability for each of the six activities. {C('classes_')} lists the activities in "
  "the order of those probability columns.")
P("Helpers and the baseline (Kathy)", lab)
CODE(grab(files["models.py"], "def _encode", "class SoftmaxRegression"))
B([f"{C('_encode')} turns the six activity names into the numbers 0 to 5 (and remembers the names).",
   f"{C('_balanced_class_weights')} gives each activity a weight of <b>n / (k x count)</b>, where n is the number of windows, "
   "k = 6 and count is how many windows that activity has. Rare activities (Standing) get a big weight and common ones "
   "(Walking) a small one, so every activity counts equally in training. This is the {C('class_weight=\"balanced\"')} setting."
   .replace("{C('class_weight=\"balanced\"')}", C('class_weight="balanced"')),
   f"{C('MajorityClassifier')} counts the activities in the training data and always answers the commonest one "
   f"(Walking). It scores {acc['baseline (majority)']:.1%} accuracy and {f1['baseline (majority)']:.3f} macro F1, which is "
   "the bar any real model must beat."])

P("SoftmaxRegression: logistic regression by gradient descent (Kathy)", h2)
P("<b>Purpose:</b> a simple model that gives every feature a weight for every activity, adds them up into a score per "
  "activity, and turns the six scores into probabilities. It learns the weights by trial and improvement.")
CODE(grab(files["models.py"], "        self.mean_ = X.mean(axis=0)", "        self.n_iter_ = t"))
P("Walk", lab)
B([f"<b>Standardise first:</b> {C('Z = (X - mean) / scale')} rescales every feature to a similar size (mean 0, spread 1), "
   "so a feature measured in the hundreds does not drown out one measured in fractions.",
   f"<b>Scores to probabilities (softmax):</b> {C('_scores = Z @ coef_ + intercept_')} multiplies the features by the weights "
   f"({C('@')} is matrix multiplication). {C('_softmax')} then makes every score positive with {C('exp')} and divides by the "
   "total, so the six answers add up to 1 and can be read as probabilities.",
   f"<b>The loss:</b> {C('-np.log(P[np.arange(n), idx]).mean()')} is the average of minus the log of the probability the model "
   "gave to the <b>correct</b> activity. It is small when the model is confident and right, large when it is confident and wrong. "
   "This is called cross-entropy. The extra term adds a small penalty on large weights (L2 regularisation) to stop it overfitting.",
   f"<b>The gradient:</b> {C('G = (P - Y) / n')} is, for each window, the predicted probabilities minus the true answer "
   f"({C('Y')} is 1 for the right activity and 0 for the rest). {C('Z.T @ G')} turns that into how much each weight should change.",
   "<b>Adam update:</b> instead of stepping by the raw gradient, Adam keeps a running average of recent gradients "
   f"({C('m')}) and of their squares ({C('v')}) and takes a step sized for each weight separately. It usually reaches a good "
   "answer faster and more steadily than plain gradient descent.",
   f"<b>When to stop:</b> after {C('max_iter')} rounds, or earlier once the loss improves by less than {C('tol')}."])
P("Say this", lab)
P(f"\"Logistic regression draws straight-line boundaries between activities. It gets {acc['logistic regression']:.1%} accuracy "
  f"and {f1['logistic regression']:.3f} macro F1, which shows the features carry a lot of the signal even for a simple model. "
  "The random forest does better because it can capture curved, interacting patterns.\"")

P("DecisionTree: how a tree chooses its questions (Collins)", h2)
P("<b>Purpose:</b> build a flowchart of yes/no questions about the features, such as \"is y_mad at most 0.98?\", that ends "
  "in an activity. Nobody picks the questions: the code tries them and keeps the best.")
CODE(grab(files["models.py"], "            n = len(rows)", "            if best is None or parent_impurity"))
P("Walk", lab)
B(["<b>Gini impurity</b> measures how mixed a group of windows is. A group that is all Walking has impurity 0; an even "
   "mix of six activities is high. The line "
   f"{C('parent_impurity = wt - (totals ** 2).sum() / wt')} is the group's weighted Gini ({C('totals')} is the weight of "
   "each activity in the group, {C('wt')} their sum)."
   .replace("{C('wt')}", C("wt")),
   f"<b>Stop conditions:</b> the group is pure ({C('parent_impurity &lt;= 1e-12')}), too small, or too deep. That group becomes a "
   "<b>leaf</b> that remembers its mix of activities.",
   f"<b>Random feature subset:</b> {C('self.rng.choice(n_features, size=mtry, replace=False)')} picks only 5 of the 32 "
   "features to consider at this question (the square root of 32). This is what makes the 300 trees different from each other.",
   f"<b>Try every threshold:</b> for each candidate feature, {C('np.argsort')} sorts the group by that feature, and "
   f"{C('np.cumsum')} gives, for every possible cut point, how much of each activity falls on the left. From that the code "
   f"computes the mixed-ness of the left and right groups ({C('child')}) for <b>all cut points at once</b>, with no inner loop.",
   f"{C('valid')} only allows a cut between two different values (a cut between two equal values separates nothing). "
   f"{C('child.argmin()')} picks the cut that leaves the two groups purest. The threshold is the midpoint between the two values."])
CODE(grab(files["models.py"], "            if best is None or parent_impurity", "        self.feature = np.array(feature)"))
P("Walk (continued)", lab)
B([f"If no cut improves purity, the group stays a leaf. Otherwise {C('importance[f] += parent_impurity - child_impurity')} "
   "records how much purer this question made things: adding this up over the whole forest gives the feature importances "
   "in the report.",
   f"The two halves become new nodes and go on the {C('stack')} to be split in turn. Using a stack instead of the function "
   "calling itself keeps it simple and avoids Python's recursion limit.",
   "The tree is stored as five parallel lists (feature, threshold, left child, right child, class mix), one entry per node. "
   f"A node with {C('left == -1')} is a leaf."])
CODE(grab(files["models.py"], "    def predict_proba_encoded", "def _fit_tree"))
P(f"To classify, every window starts at the root, and {C('go_left = X[...] &lt;= threshold')} sends it left or right until "
  "it reaches a leaf; the answer is that leaf's class mix. The code moves all windows one level at a time "
  f"({C('active')} tracks which are still travelling) rather than one window at a time.")

P("RandomForest: 300 trees that vote (Collins)", h2)
P("<b>Purpose:</b> one tree on all the data tends to memorise it, so we grow many different trees and average them.")
CODE(grab(files["models.py"], "def _fit_tree", "class RandomForest"))
B([f"<b>Bootstrap sample:</b> {C('rng.integers(0, len(idx), len(idx))')} draws n windows <b>with replacement</b>, so some windows "
   "appear twice or more and about a third are left out. {C('counts')} says how many times each was drawn, and that count "
   "becomes the window's weight, which is a compact way to train on the bootstrap sample."
   .replace("{C('counts')}", C("counts")),
   f"{C('counts[rows] * class_w[idx[rows]]')} multiplies that by the class weight, so rare activities still count more.",
   "Each tree gets its own random seed, so the result is repeatable but every tree is different."])
CODE(grab(files["models.py"], "        mtry = max(1, int(np.sqrt(n_features)))", "    def predict_proba"))
CODE(grab(files["models.py"], "        return np.mean([t.predict_proba_encoded(X)", "    def predict(self, X)"))
P("Walk", lab)
B([f"{C('SeedSequence(random_state).spawn(300)')} creates 300 independent seeds from the one seed 42, so training is repeatable.",
   f"{C('Parallel(n_jobs=-1)')} (from joblib) trains the trees on all CPU cores at once, because each tree is independent.",
   f"{C('feature_importances_')} averages each tree's importances: the share of the Gini improvement each feature provides.",
   f"{C('predict_proba')} asks all 300 trees for their leaf's class mix and <b>averages</b> them. The activity with the highest "
   "average wins, and that average is the confidence the app shows."])
P("Say this", lab)
P(f"\"One tree can be wrong, but 300 different trees tend to make different mistakes, and averaging cancels many of them. "
  f"Our forest reaches {acc['random forest']:.1%} accuracy and {f1['random forest']:.3f} macro F1 on people it has never seen. "
  f"It has about {n_nodes:,} decision points in total, on average {avg_depth:.0f} levels deep.\"")
P("If asked why", lab)
P("Trees can capture curved and interacting patterns that a straight-line model cannot, they need no feature scaling, "
  "and we can read their questions. The forest scores 100% on its own training windows (it memorises them) and "
  f"{acc['random forest']:.0%} on new people, which is why we only report the second number.")

# ---- evaluation.py
P("evaluation.py: splitting by person and scoring (Moses)", h2)
P("<b>Purpose:</b> decide which people are used for training and which for testing, and compute the scores.")
CODE(grab(files["evaluation.py"], "def group_shuffle_split", "def accuracy"))
P("Walk", lab)
B([f"{C('np.unique(groups, return_inverse=True)')} lists the 36 people and gives every window the number of its person.",
   f"{C('n_test = ceil(0.25 x 36) = 9')}: nine people are held out. {C('np.random.RandomState(seed).permutation')} shuffles the "
   "people with a fixed seed, so the same nine are chosen every run (9, 13, 14, 17, 22, 27, 31, 32 and 36).",
   f"{C('np.isin(member, ...)')} then collects <b>every window</b> of those people on one side. No person is ever on both sides.",
   f"{C('group_kfold')} makes 5 folds for cross-validation: it deals the people out one by one, biggest first, always to the "
   "fold that is currently smallest, so the folds are similar in size and nobody is in two folds."])
CODE(grab(files["evaluation.py"], "def _per_class", "def classification_report"))
P("Walk (scores)", lab)
B([f"The <b>confusion matrix</b> counts, for each real activity (row), how often the model said each activity (column). "
   f"{C('np.add.at')} adds one to the right cell for every window.",
   "<b>Precision</b> for an activity is the correct answers on the diagonal divided by everything the model labelled with it; "
   "<b>recall</b> is the correct answers divided by all real windows of that activity; <b>F1</b> is their harmonic mean.",
   "<b>Macro F1</b> is the plain average of the six F1 scores, so Standing counts as much as Walking. Accuracy alone would "
   "reward a model that ignores the small classes."])
P("Say this", lab)
P("\"We wrote the split and the scores ourselves and then checked them against scikit-learn's versions on our real data. "
  "They picked exactly the same nine test people and produced identical accuracy, macro F1, confusion matrix and report, "
  "so replacing the library's splitting and scoring with ours changed no result.\"")

# ---- train.py
P("train.py: running the experiment (Collins)", h2)
P("<b>Purpose:</b> window the data, split by person, train and score the three models, pick the best, and save it.")
CODE(grab(files["train.py"], "    # Hold out whole subjects", "    best_name"))
P("Walk", lab)
B([f"{C('group_shuffle_split(groups, ...)')} puts nine people in the test set; the same split every run because of {C('SEED')}.",
   "The dictionary lists the three models. The forest has 300 trees and balanced class weights.",
   f"For each model, {C('fit')} is where the learning happens, and {C('predict')} is only using the result on the "
   "test people. Accuracy and macro F1 are computed from the predictions."])
CODE(grab(files["train.py"], "    cv = []", "    top = {}"))
P(f"<b>Cross-validation:</b> to check the result does not depend on one lucky split, we repeat the experiment five times "
  f"with a different fifth of the people held out each time (a smaller 150-tree forest to save time). The average is "
  f"{cv_mean:.3f} with a spread of {cv_std:.3f}, so the result is stable.")
CODE(grab(files["train.py"], "    # Keep the model trained", '    with open(os.path.join(out_reports, "metrics.json")'))
P("<b>Two files are saved on purpose.</b> "
  f"{C('demo_model.joblib')} is trained on the 27 training people only (and remembers which 9 were held out), so the demo can "
  f"show windows from people it has never seen. Then {C('best.fit(X, y)')} retrains on all 36 people and saves "
  f"{C('activity_model.joblib')} for deployment.")
P("If asked why", lab)
P("Windows from the same person are nearly identical, so a random split would let the model memorise people and inflate the "
  "score. We measured the effect of memorisation directly: the forest scores 100% on its own training windows and "
  f"{acc['random forest']:.0%} on new people.")

# ---- predict
P("predict.py: using the saved model (Moses)", h2)
P("<b>Purpose:</b> take 80 readings, return the activity, a confidence and the fall flag. No learning happens here.")
CODE(grab(files["predict.py"], "def predict_window", "if __name__"))
P("Walk", lab)
B([f"{C('joblib.load')} reads the saved model file from disk.",
   f"{C('window_features(w)')} computes the same 32 features; "
   f"{C('[f[name] for name in bundle[&quot;features&quot;]]')} puts them in the <b>same order</b> the model was trained with, "
   "because the model learned by column position.",
   f"{C('predict_proba')} gives a probability for each of the six activities; "
   f"{C('argmax')} picks the largest, which is also the confidence."])
P("<b>Speed, honestly:</b> classifying one window takes " + ms_txt + " on this laptop, because our forest visits its 300 "
  "trees one after another in Python. Windows arrive every 2 seconds, so that is comfortably fast enough. A library written in "
  "compiled code would be faster; we accept that for code we can explain line by line. The two lines "
  f"{C('if hasattr(m, &quot;n_jobs&quot;): m.n_jobs = 1')} in predict.py and demo.py are left over from when the forest came "
  "from a library: our forest only uses n_jobs while training, so they now have no effect. Say this if asked, "
  "rather than defending the comment next to them.")

# describe
P("describe.py: reading the binary model file (Moses)", h2)
P("<b>Purpose:</b> a .joblib file is binary and cannot be opened in a text editor, so this tool loads it and prints what is inside: "
  "the model type, the six activities, the 32 inputs in order, the number of trees and nodes, the most important features, and the real "
  f"questions at the top of the first tree. Running training also saves this as {C('models/model_summary.txt')}, which is on GitHub.")
CODE(grab(files["describe.py"], "    if hasattr(model, &quot;trees_&quot;)".replace("&quot;", '"'), "    return"))
P(f"For the deployment model it reports {len(rf.trees_)} trees, about {n_nodes:,} nodes in total, an average depth of "
  f"{avg_depth:.1f} levels and about {avg_leaves:.0f} end points per tree. {C('trees[0].to_text(...)')} is a method we wrote "
  "on DecisionTree that prints the top of the tree as indented if/else rules.")

# demo
P("demo.py and the demo page (Kathy)", h2)
P("<b>Purpose:</b> a tiny local web server. The page in the browser asks it for a window; it computes the answer with the saved model "
  f"and replies in JSON. It uses only Python's built-in {C('http.server')}, so there is no extra library.")
CODE(grab(files["demo.py"], "    def classify", "    def real_window"))
CODE(grab(files["demo.py"], "        def do_GET", "    return H"))
P("Walk", lab)
B([f"<b>Engine.__init__</b> (not shown) loads {C('demo_model.joblib')}, loads the data, keeps only the 9 held-out people "
   f"({C('np.isin')}), cuts their data into windows, and files them by activity. So every window the demo shows "
   "comes from someone the model never trained on.",
   "<b>classify</b> is the same idea as predict.py but it also returns the raw readings (for the charts), all six probabilities and the fall rule.",
   f"<b>do_GET</b> is the routing: {C('/')} sends the HTML page, {C('/api/window?activity=Walking')} "
   f"sends a real window and its prediction, and {C('/api/fall')} sends the staged fall.",
   "<b>demo_page.html</b> is the front end (HTML for layout, CSS for style, JavaScript for behaviour): it calls those two URLs, draws the "
   "two phone screens, runs the 10-second \"Are you OK?\" countdown and sends the alert to the caregiver phone. "
   "The phones, the calls and the notifications are simulated; the predictions and the fall rule are real."])
P("Say this", lab)
P("\"The browser never contains the model. It asks the Python server, the server asks the saved model, and the answer comes "
  "back as JSON. On a real phone the same idea would run either on the device or on a small server.\"")

# ------------------------------------------------------------------ 4 worked example
P("4. One window through the whole code (worked example)", h1)
P(f"This uses the sample file {C('reports/samples/sample_walking.csv')}: 80 real readings from a Walking recording.")
table([["Step", "What happens", "Real values from this window"],
       ["1. Input", "80 readings, 3 axes",
        f"shape {win.shape}; first reading x={win[0,0]:.2f}, y={win[0,1]:.2f}, z={win[0,2]:.2f}"],
       ["2. Features", "window_features turns it into 32 numbers",
        f"z_mean = {feat['z_mean']:.2f}, y_mean = {feat['y_mean']:.2f}, y_mad = {feat['y_mad']:.2f}, "
        f"y_energy = {feat['y_energy']:.0f}, mag_max = {feat['mag_max']:.1f}"],
       ["3. Trees", "each of the 300 trees follows its questions to a leaf",
        f"{n_top} of the 300 trees put {ans['activity']} on top; the rest favoured other activities"],
       ["4. Average", "the leaves' class mixes are averaged",
        f"answer = {ans['activity']}, confidence = {ans['confidence']:.0%}"],
       ["5. Fall rule", "peak impact and stillness check",
        f"peak = {fall['impact_g']:.2f} g (threshold 2.5), impact = {fall['impact']}, "
        f"still afterwards = {fall['post_impact_still']}, fall suspected = {fall['fall_suspected']}"]],
      [2.2, 5.0, 9.8])
P(f"Run it yourself in front of the audience: {C('python -m mlengine.predict reports/samples/sample_walking.csv')}. "
  f"The second sample, {C('sample_fall.csv')}, is a staged fall and returns {C('fall_suspected: True')}. "
  "(That saved model was trained on all 36 people, so use this to show how it works, not as accuracy evidence.)")

# ------------------------------------------------------------------ 5 results
P("5. The results, with the numbers to quote", h1)
table([["Model", "Accuracy", "Macro F1"]] +
      [[n.capitalize(), f"{acc[n]:.3f}", f"{f1[n]:.3f}"] for n in ("baseline (majority)", "logistic regression", "random forest")],
      [7, 4, 4])
P(f"Random forest, 5-fold cross-validation by person: macro F1 {cv_mean:.3f} with a spread of {cv_std:.3f}. "
  "Tested on 9 people (6,526 windows) the model had never seen.")
table([["Activity", "Precision", "Recall", "F1", "Test windows"]] +
      [[a, f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}", v[3]] for a, v in report_rows.items()],
      [4, 3, 3, 3, 3.5])
P(f"Downstairs is the weak class: it finds only {report_rows['Downstairs'][1]:.0%} of real Downstairs windows, mostly "
  "confusing them with Upstairs and Walking, which look alike from a pocket phone. During development we also trained "
  f"{SK}'s logistic regression and random forest on the same split as a check: they scored 0.830 / 0.750 and "
  "0.849 / 0.775 (accuracy / macro F1), within 0.003 of ours. After that check we removed the library.")

# ------------------------------------------------------------------ 6 cheat sheet
P("6. Reading the code: a cheat sheet", h1)
table([["You see", "It means"],
       [C("import numpy as np"), "Load the numpy library and call it np. numpy handles arrays of numbers fast."],
       [C("X[tr]"), "Take the rows of X whose positions are listed in tr (here, the training windows)."],
       [C('data["user"]'), "Look up an entry in a dictionary (a set of named values)."],
       [C("A @ B"), "Matrix multiplication: how every feature is multiplied by its weight and added up."],
       [C("np.cumsum(...)"), "Running total. Used to count, for every cut point, how much of each activity lies to its left."],
       [C("np.argsort / argmax / argmin"), "The positions that sort an array / of its largest / of its smallest value."],
       [C("np.bincount(idx)"), "Count how many times each number 0, 1, 2... appears."],
       [C("[f[name] for name in names]"), "A list comprehension: build a list by repeating something for each name."],
       [C('f"{x:.2f}"'), "An f-string: put the value of x into the text with 2 decimals."],
       [C("yield"), "Hand back one item at a time instead of a whole list (a generator)."],
       [C("class Name:"), "Define a model as an object with its own settings and its fit and predict methods."],
       [C("try: ... except ValueError: continue"), "Try something; if it fails with that error, skip to the next item."],
       [C('if __name__ == "__main__":'), "Run this part only when the file is run directly, not when it is imported."],
       [C("python -m mlengine.train"), "Run the file train.py inside the package mlengine."],
       [C("random_state=42 / SEED"), "A fixed starting point for randomness, so the results repeat exactly."],
       [C(".fit(X, y)") + " / " + C(".predict(X)"), "fit = learn from examples (training). predict = answer for new examples."],
       [C("Parallel(n_jobs=-1)"), "joblib runs the same job (growing a tree) on all CPU cores at once."],
       [C("joblib.dump / load"), "Save a trained model to a file / load it back."]],
      [6.2, 10.8])

# ------------------------------------------------------------------ 7 code Q&A
P("7. Questions about the code, with answers", h1)
table([["Question", "Answer", "Who"],
       ["Which libraries do you use, and why those?",
        "Only numpy (arrays, maths, the Fourier transform) and joblib (growing the trees in parallel and saving models). "
        "There is no machine-learning library. File parsing, the web server and downloads use the Python standard library. "
        "pandas, matplotlib and reportlab are used only for the optional notebook and for building the report.", "Moses"],
       ["Did you write the machine learning yourselves, or use a library?",
        f"We wrote it: the baseline, logistic regression, decision tree and random forest in models.py, and the splitting and "
        f"scoring in evaluation.py. We started from {SK} and then replaced it, after checking our versions matched.", "Collins"],
       ["How do you know your own models are correct?",
        f"We trained {SK}'s logistic regression and random forest on the same split during development and compared: theirs "
        f"scored 0.830 / 0.750 and 0.849 / 0.775 (accuracy / macro F1), ours {acc['logistic regression']:.3f} / "
        f"{f1['logistic regression']:.3f} and {acc['random forest']:.3f} / {f1['random forest']:.3f}, within 0.003. We also "
        "checked that our split, folds and scores are identical to the library's on the real data. We have not written automated tests.", "Collins"],
       ["What is Gini impurity?",
        "A measure of how mixed a group is: 0 if every window in it has the same activity, higher the more mixed. A tree chooses "
        "the question that leaves the two resulting groups as pure (low Gini) as possible.", "Collins"],
       ["How does the tree choose where to split?",
        "At each step it looks at 5 randomly chosen features, tries every possible cut point of each, and keeps the cut that makes "
        "the left and right groups purest. It repeats on each side until a group is pure or cannot be split.", "Collins"],
       ["Why 300 trees, and why random subsets of features?",
        "Each tree sees a different bootstrap sample and only 5 of the 32 features at each question, so the trees make different "
        "mistakes and averaging them cancels many errors. 300 is plenty for stable results; we did not tune it.", "Collins"],
       ["What is a bootstrap sample?",
        "A random sample drawn with replacement, the same size as the training set. About two thirds of the distinct windows appear "
        "(some more than once). Each tree trains on its own.", "Collins"],
       ["What is class_weight balanced?",
        "Each activity gets a weight of n / (k x count), so rare activities such as Standing count as much in training as common ones "
        "such as Walking. Without it the model would favour Walking.", "Kathy"],
       ["How does logistic regression learn?",
        "It starts with all weights at zero and repeatedly adjusts them to reduce the cross-entropy loss, using gradient descent "
        "with the Adam optimiser, until the loss stops improving. A softmax turns the six scores into probabilities.", "Kathy"],
       ["What is Adam?",
        "A way of taking the gradient-descent steps: it keeps a running average of recent gradients and their squares and adapts "
        "the step size per weight, so training is faster and steadier than using the raw gradient.", "Kathy"],
       ["What is regularisation, and do you use it?",
        "A small penalty on large weights that discourages the model from fitting noise. Logistic regression uses an L2 penalty "
        "(C = 1). The forest has no depth limit; it relies on averaging many trees instead.", "Kathy"],
       ["How do you stop test people leaking into training?",
        "The split is by person: group_shuffle_split picks 9 of the 36 people and puts every window of each on the test side. The demo "
        "model records its test people and the demo keeps only them. The notebook prints the overlap between the groups and it is empty.", "Moses"],
       ["Why is the random seed fixed, and can you reproduce the numbers?",
        "A fixed seed gives the same split and the same forest every run. We pin numpy 2.3.3 and joblib 1.5.2 and reproduced the "
        "reported numbers in a fresh environment. Different library versions could change results slightly.", "Moses"],
       ["Why not a library for the splitting and scoring?",
        "They are short and we wanted no machine-learning library at all. We verified they give identical results to the library's.", "Moses"],
       ["Why is the same feature function used in training and prediction?",
        "If the two were computed differently the model would see inputs unlike anything it learned from. One function guarantees consistency.", "Collins"],
       ["What does n_jobs do?",
        "It sets how many CPU cores joblib uses to grow trees. Training uses all cores (-1). It has no effect on prediction in our "
        "forest.", "Collins"],
       ["Is the forest fast enough?",
        f"Training takes under a minute. Classifying one window takes {ms_txt}, and windows arrive every 2 seconds, so it is fast enough. "
        "A compiled library would be quicker; our forest loops over its trees in Python.", "Moses"],
       ["What is inside the saved model file?",
        "A Python dictionary saved with joblib: our trained RandomForest object (300 trees stored as arrays) and the 32 feature names "
        "in the order the model expects. It is binary, so python -m mlengine.describe prints its contents.", "Moses"],
       ["How are bad lines in the raw file handled?",
        "Lines without 6 fields, unreadable numbers, missing values, zero timestamps and duplicates are skipped, and the clean count "
        "is 1,019,496 readings from 1,086,466 usable lines.", "Moses"],
       ["Did you tune hyperparameters?",
        "No. We used 300 trees, the square-root rule for features and no depth limit as sensible defaults. Tuning is future work.", "Collins"],
       ["Is the fall rule machine learning?",
        "No. It is a transparent physics rule (impact of at least 2.5 g followed by stillness) because there are no falls in the "
        "training data. It must be validated on real fall data.", "Kathy"],
       ["Does the demo contain the model?",
        "No. The browser page only draws. The Python server holds the model, computes the answer and returns JSON.", "Kathy"],
       ["Do you have tests?",
        "We checked the code by running the full pipeline in a clean environment with no machine-learning library installed, "
        "comparing results to the library's during development, running the predictor on sample files, and exercising the demo end to "
        "end. We have not written automated unit tests; that would be a next step.", "Moses"],
       ["What would you change to use a different window length?",
        "Change WINDOW (and STEP) at the top of features.py, then retrain, because the features and the model depend on it.", "Collins"],
       ["What if a phone samples at a different rate than 20 Hz?",
        "Our windows assume 20 readings per second. A real app would resample the phone's data to 20 Hz before windowing.", "Moses"],
       ["Why Python's built-in web server instead of Flask?",
        "It keeps the engine dependency-free. It is fine for a local demo; a production service would use a proper framework.", "Kathy"],
       ["Could this run on the phone?",
        "Not as written. The model would be exported to a mobile format, or the phone would send windows to a server. Neither is built.", "Kathy"]],
      [4.8, 10.5, 1.7])

# ------------------------------------------------------------------ 8 Colab
P("8. Running everything in Google Colab", h1)
P("The notebook <b>MLEngine_Colab.ipynb</b> runs the whole project step by step with explanations. It clones the code from GitHub "
  "and uses the same mlengine functions, so it is the same code, not a copy. It is also a good <b>plan B for the demo</b>: if the "
  "laptop fails on the day, any computer with a browser can run it.")
P("Open it with this link:", lab)
CODE("https://colab.research.google.com/github/Moses374/SemProject_MLEngine/blob/master/MLEngine_Colab.ipynb")
B(["Choose <b>Runtime, then Run all</b>. The first run downloads the 50 MB dataset. Training our own forest takes under a minute; "
   "the whole notebook took about three minutes on a laptop.",
   "Or run the cells one at a time while presenting. Each section starts with a plain-language explanation, then shows the code, "
   "then its result, so it doubles as a script for explaining the code.",
   "Section 1 prints the numpy and joblib versions and says how to install the exact ones "
   f"({C('pip install numpy==2.3.3 joblib==1.5.2')}, then Runtime, Restart session) if they differ. There is no "
   f"{SK} in the notebook.",
   "Sections 2 to 10 follow this document: the data and cleaning (data.py), windows and features (features.py), the person-wise "
   "split (evaluation.py), the three models and how the forest learns (models.py, shown with the real code), the scores, "
   "the inside of the model, saving and prediction (predict.py), and the fall rule.",
   "Section 11 starts the live demo server inside Colab and shows the two phone screens in the notebook. "
   "<b>Try this once beforehand.</b> The notebook ran cleanly on a laptop, but the embedded demo view uses a Colab feature that "
   "could not be tested outside Colab; if it does not appear, run the cell again.",
   "<b>If Colab says \"Failed to fetch\" or \"Unable to connect to the runtime\",</b> that is Colab failing to reach its "
   "servers, not our code. Click Reconnect, reload the page, try an incognito window with extensions off (ad blockers and "
   "some networks block gstatic.com), or try another network."])


def footer(c, d):
    c.saveState()
    c.setFont("Times-Roman", 9)
    if d.page > 1:
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
    c.restoreState()


doc = SimpleDocTemplate("reports/Code_Walkthrough.pdf", pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm, title="Code Walkthrough")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("written reports/Code_Walkthrough.pdf")
