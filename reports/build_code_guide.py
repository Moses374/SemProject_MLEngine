"""Build reports/Code_Walkthrough.pdf: how to explain every file of the project, with the real code.

Run from the project root:  python -m reports.build_code_guide
Code excerpts are copied from the actual source files at build time, so they cannot drift from the repo.
"""
import os

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

from mlengine.features import window_features, fall_risk
from mlengine.predict import predict_window

# ------------------------------------------------------------------ helpers
def grab(path, start, end=None):
    """Return the source text from the line containing `start` up to (not including) the line containing `end`."""
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
            out.append(l[:cut]); l = "    " + l[cut:].lstrip()
        out.append(l)
    return "\n".join(out)


body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=15, alignment=TA_JUSTIFY, textColor=colors.black, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=10, spaceAfter=6, alignment=0, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Times-Bold", fontSize=12.5, leading=16, spaceBefore=9, spaceAfter=3, alignment=0, keepWithNext=1)
lab = ParagraphStyle("lab", parent=body, fontName="Times-Bold", spaceAfter=1, spaceBefore=3, alignment=0, keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=24, leading=30, alignment=TA_CENTER, spaceAfter=10)
sub = ParagraphStyle("sub", parent=body, fontSize=13, alignment=TA_CENTER, leading=18)
cell = ParagraphStyle("cell", parent=body, fontSize=10, leading=12.5, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")
code_style = ParagraphStyle("code", fontName="Courier", fontSize=7.4, leading=9.2, textColor=colors.black,
                            backColor=colors.Color(0.95, 0.95, 0.95), borderColor=colors.black, borderWidth=0.4,
                            borderPadding=4, spaceBefore=6, spaceAfter=10, leftIndent=4, rightIndent=4)
S = []


def P(t, st=body): S.append(Paragraph(t, st))
def B(items):
    for t in items:
        P("&bull; " + t)
def CODE(text): S.append(Preformatted(fit(text), code_style))
def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black)]))
    S.append(t); S.append(Spacer(1, 8))


F = "mlengine"
files = {n: f"{F}/{n}" for n in ["data.py", "features.py", "train.py", "predict.py", "describe.py", "demo.py", "demo_page.html"]}

# ------------------------------------------------------------------ cover
S.append(Spacer(1, 4.5 * cm))
P("Code Walkthrough", title)
P("How to explain every part of the project, with the real code", sub)
P("Activity Recognition and Fall-Risk Detection from Smartphone Accelerometer Data", sub)
S.append(Spacer(1, 1.5 * cm))
P("Moses Kibet (183638)<br/>Collins Jared (184587)<br/>Kathy Auma (192231)", sub)
S.append(Spacer(1, 1.5 * cm))
P("Every code box in this document is copied from the project files when the document is built, so it matches "
  "what is on GitHub: github.com/Moses374/SemProject_MLEngine", sub)
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
   "<b>Say what you did not write.</b> We did not write the random forest; scikit-learn provides it. We wrote the cleaning, "
   "windowing, features, fair testing and fall rule. That honest split is a strength.",
   "<b>If you do not know a line, explain its job.</b> \"That line makes the result repeatable\" is better than guessing the syntax.",
   "<b>Keep the files straight:</b> data.py cleans, features.py describes windows, train.py learns, predict.py uses, "
   "describe.py shows what is inside, demo.py serves the demo."])

P("The whole codebase in 90 seconds", h2)
P("\"Our project is a small Python package called mlengine. <b>data.py</b> loads and cleans the raw WISDM file. "
  "<b>features.py</b> cuts the data into four-second windows and turns each window into 32 numbers, and it also holds the "
  "fall rule. <b>train.py</b> splits the people into training and test groups, trains three models, tests them on people "
  "they have never seen, and saves the best one to a file. <b>predict.py</b> loads that file and classifies a new window, "
  "and <b>describe.py</b> prints what is inside the file, because the file itself is binary. <b>demo.py</b> is a small web "
  "server that feeds real test windows to the model and sends the answer to the demo page. The whole engine needs only "
  "numpy and scikit-learn.\"")

# ------------------------------------------------------------------ 2 map
P("2. The map: files, size and who explains what", h1)
rows = [["File", "Lines", "Its job", "Explains it"],
        ["mlengine/data.py", nlines(files["data.py"]), "Load and clean the raw file", "Moses"],
        ["mlengine/features.py", nlines(files["features.py"]), "Windows, 32 features, fall rule", "Collins (windows, features) and Kathy (fall rule)"],
        ["mlengine/train.py", nlines(files["train.py"]), "Split by person, train, evaluate, save", "Collins"],
        ["mlengine/predict.py", nlines(files["predict.py"]), "Load a saved model and classify one window", "Moses"],
        ["mlengine/describe.py", nlines(files["describe.py"]), "Show what is inside the binary model files", "Moses"],
        ["mlengine/demo.py", nlines(files["demo.py"]), "Local web server behind the demo", "Kathy"],
        ["mlengine/demo_page.html", nlines(files["demo_page.html"]), "The two-phone demo page (HTML, CSS, JavaScript)", "Kathy"]]
table(rows, [4.2, 1.5, 6.3, 5.0])
P("The engine (the five Python files) is about "
  f"{sum(nlines(files[n]) for n in ['data.py', 'features.py', 'train.py', 'predict.py', 'describe.py'])} lines. "
  "That is small enough that each of you can explain your files completely.")

# ------------------------------------------------------------------ 3 file by file
P("3. File by file", h1)

# data.py
P("data.py: loading and cleaning (Moses)", h2)
P("<b>Purpose:</b> read the raw WISDM text file and return clean numpy arrays: user, activity, timestamp, x, y, z. "
  "It is the only code that touches the raw file, so every later step can trust the data.")
CODE(grab(files["data.py"], "    user, act, ts, xyz = [], [], [], []"))
P("Walk", lab)
B(["The loop reads the file one line at a time, strips the trailing semicolon and splits on commas; "
   "<font face='Courier'>len(parts) != 6</font> skips any line that does not have exactly 6 pieces.",
   "<font face='Courier'>except ValueError: continue</font> skips a line with an unreadable or missing number. That is how the "
   "one missing z value in the file is dropped.",
   "<font face='Courier'>np.isfinite(...)</font> rejects not-a-number values, and <font face='Courier'>t &lt;= 0</font> rejects zero timestamps.",
   "<font face='Courier'>np.unique(..., return_index=True)</font> finds the first occurrence of each (user, timestamp) pair, "
   "so duplicate readings are dropped; <font face='Courier'>np.sort(first)</font> keeps them in file order.",
   "<font face='Courier'>np.lexsort((ts, user))</font> sorts by user and then time, so consecutive rows are consecutive readings."])
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
B(["<font face='Courier'>np.unique(acts, return_inverse=True)</font> turns the six activity names into the numbers 0 to 5.",
   "<font face='Courier'>key = users * len(names) + codes</font> makes one number for every (person, activity) pair. "
   "Think of it as a labelled bucket: \"person 33 jogging\".",
   "The loop visits each bucket; <font face='Courier'>rows = xyz[key == k]</font> pulls out all readings of that bucket, still in time order.",
   "<font face='Courier'>range(0, len(rows) - window + 1, step)</font> slides along: start at 0, move 40 readings each time, "
   "stop when a full 80 no longer fit.",
   "<font face='Courier'>yield</font> hands back one window at a time instead of building a huge list (this is called a generator)."])
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
   "<font face='Courier'>np.fft.rfft</font> is a Fourier transform: it splits the signal into repeating frequencies, so "
   "<font face='Courier'>spec[1:].argmax() + 1</font> is the strongest repeating rate. Walking and jogging repeat at different rates.",
   "After the loop (not shown) come 5 statistics of the total magnitude and 3 correlations between the axes. "
   "<b>8 x 3 + 5 + 3 = 32.</b>"])
P("Say this", lab)
P("\"The same function is used for training and for live prediction, so the model always sees features computed exactly the "
  "same way.\"")
P("If asked why", lab)
P("Raw readings are noisy and there are too many of them. Hand-designed features are compact, fast and explainable: orientation "
  "separates sitting from standing, and vibration strength and rhythm separate walking from jogging.")
CODE(grab(files["features.py"], "def build_dataset", "def fall_risk"))
P("<b>build_dataset</b> loops over every window, calls <font face='Courier'>window_features</font>, and stacks the results into a table "
  "<font face='Courier'>X</font> (one row per window, 32 columns), with <font face='Courier'>y</font> (the correct activity) and "
  "<font face='Courier'>users</font> (which person each window came from; this is what makes fair testing possible).")

# fall_risk
P("features.py: the fall rule (Kathy)", h2)
P("<b>Purpose:</b> decide whether a window looks like a fall: a hard impact followed by stillness.")
CODE(grab(files["features.py"], "def fall_risk"))
P("Walk", lab)
B(["<font face='Courier'>np.linalg.norm(w, axis=1) / G</font> turns each reading into the total strength of movement, measured in g "
   "(1 g is the pull of gravity, so a still phone reads about 1).",
   "<font face='Courier'>peak = argmax</font> finds the strongest moment. <b>impact</b> is true if it is at least 2.5 g.",
   "<b>still_after</b> is true if there are at least 10 readings after the peak and the <b>last 10 readings (the final half second)</b> "
   "average within 0.15 g of 1 g with almost no variation.",
   "<b>fall_suspected</b> needs both. Walking and jogging often exceed 2.5 g on a footstep but keep moving, so they are not flagged."])
P("Say this", lab)
P("\"This is a rule, not machine learning, because WISDM contains no falls to learn from. We tested it only on a staged signal, so "
  "it must be validated on real fall data before real use.\"")

# train
P("train.py: learning and honest testing (Collins)", h2)
P("<b>Purpose:</b> train three models, test them on people they have never seen, and save the best.")
CODE(grab(files["train.py"], "    # Hold out whole subjects", "    best_name"))
P("Walk", lab)
B(["<font face='Courier'>GroupShuffleSplit(..., test_size=0.25)</font> picks about a quarter of the <b>people</b> for testing. "
   "Passing <font face='Courier'>users</font> as the groups guarantees that no person appears on both sides. "
   "<font face='Courier'>random_state=SEED</font> fixes the choice, so it is the same every run.",
   "The dictionary lists the three models. The random forest has 300 trees, "
   "<font face='Courier'>class_weight=\"balanced\"</font> (rare activities such as Standing count more), and "
   "<font face='Courier'>n_jobs=-1</font> (use every CPU core).",
   "For logistic regression, <font face='Courier'>make_pipeline(StandardScaler(), ...)</font> puts the rescaling and the model "
   "together, so the scaling is learned from the training windows only (no test information leaks in).",
   "The loop trains each model on <font face='Courier'>X[tr]</font> and scores it on <font face='Courier'>X[te]</font>: training "
   "(<font face='Courier'>fit</font>) is where the learning happens; <font face='Courier'>predict</font> is only using the result."])
CODE(grab(files["train.py"], "    # Keep the model trained", '    with open(os.path.join(out_reports, "metrics.json")'))
P("<b>Two files are saved on purpose.</b> <font face='Courier'>demo_model.joblib</font> is trained on the 27 training people only "
  "(and remembers which 9 were held out), so the demo can show windows from people it has never seen. Then "
  "<font face='Courier'>best.fit(X, y)</font> retrains on all 36 people and saves <font face='Courier'>activity_model.joblib</font> "
  "for deployment.")
P("If asked why", lab)
P("Windows from the same person are nearly identical, so a random split would let the model memorise people and inflate the "
  "score. We measured the effect of memorisation directly: the forest scores 100% on its own training windows and 85% on new people.")

# predict
P("predict.py: using the saved model (Moses)", h2)
P("<b>Purpose:</b> take 80 readings, return the activity, a confidence and the fall flag. No learning happens here.")
CODE(grab(files["predict.py"], "def predict_window", "if __name__"))
P("Walk", lab)
B(["<font face='Courier'>joblib.load</font> reads the saved model file from disk.",
   "<font face='Courier'>window_features(w)</font> computes the same 32 features; "
   "<font face='Courier'>[f[name] for name in bundle[\"features\"]]</font> puts them in the <b>same order</b> the model was trained with, "
   "because the model learned by column position.",
   "<font face='Courier'>predict_proba</font> gives a probability for each of the six activities; "
   "<font face='Courier'>argmax</font> picks the largest, which is also the confidence.",
   "<font face='Courier'>m.n_jobs = 1</font> switches off parallel threads for a single window. It is about three times faster "
   "(roughly 30 ms instead of 110 ms) because starting threads costs more than the work."])

# describe
P("describe.py: reading the binary model file (Moses)", h2)
P("<b>Purpose:</b> a .joblib file is binary and cannot be opened in a text editor, so this tool loads it and prints what is inside: "
  "the model type, the six activities, the 32 inputs in order, the number of trees and nodes, the most important features, and the real "
  "questions at the top of the first tree. Running training also saves this as <font face='Courier'>models/model_summary.txt</font>, which is on GitHub.")
CODE(grab(files["describe.py"], '        out += ["", "Top of the first tree', "    return"))

# demo
P("demo.py and the demo page (Kathy)", h2)
P("<b>Purpose:</b> a tiny local web server. The page in the browser asks it for a window; it computes the answer with the saved model "
  "and replies in JSON. It uses only Python's built-in <font face='Courier'>http.server</font>, so there is no extra library.")
CODE(grab(files["demo.py"], "    def classify", "    def real_window"))
CODE(grab(files["demo.py"], "        def do_GET", "    return H"))
P("Walk", lab)
B(["<b>Engine.__init__</b> (not shown) loads <font face='Courier'>demo_model.joblib</font>, loads the data, keeps only the 9 held-out people "
   "(<font face='Courier'>np.isin</font>), cuts their data into windows, and files them by activity. So every window the demo shows "
   "comes from someone the model never trained on.",
   "<b>classify</b> is the same idea as predict.py but it also returns the raw readings (for the charts), all six probabilities and the fall rule.",
   "<b>do_GET</b> is the routing: <font face='Courier'>/</font> sends the HTML page, <font face='Courier'>/api/window?activity=Walking</font> "
   "sends a real window and its prediction, and <font face='Courier'>/api/fall</font> sends the staged fall.",
   "<b>demo_page.html</b> is the front end (HTML for layout, CSS for style, JavaScript for behaviour): it calls those two URLs, draws the "
   "two phone screens, runs the 10-second \"Are you OK?\" countdown and sends the alert to the caregiver phone. "
   "The phones, the calls and the notifications are simulated; the predictions and the fall rule are real."])
P("Say this", lab)
P("\"The browser never contains the model. It asks the Python server, the server asks the saved model, and the answer comes "
  "back as JSON. On a real phone the same idea would run either on the device or on a small server.\"")

# ------------------------------------------------------------------ 4 worked example
P("4. One window through the whole code (worked example)", h1)
P("This uses the sample file <font face='Courier'>reports/samples/sample_walking.csv</font>: 80 real readings from a Walking recording.")
w = np.genfromtxt("reports/samples/sample_walking.csv", delimiter=",", names=True)
win = np.column_stack((w["x"], w["y"], w["z"]))
feat = window_features(win)
ans = predict_window(win)
fall = fall_risk(win)
table([["Step", "What happens", "Real values from this window"],
       ["1. Input", "80 readings, 3 axes", f"shape {win.shape}; first reading x={win[0,0]:.2f}, y={win[0,1]:.2f}, z={win[0,2]:.2f}"],
       ["2. Features", "window_features turns it into 32 numbers",
        f"z_mean = {feat['z_mean']:.2f}, y_mean = {feat['y_mean']:.2f}, y_mad = {feat['y_mad']:.2f}, "
        f"y_energy = {feat['y_energy']:.0f}, mag_max = {feat['mag_max']:.1f}"],
       ["3. Model", "300 trees answer; their votes are combined",
        f"answer = {ans['activity']}, confidence = {ans['confidence']:.0%}"],
       ["4. Fall rule", "peak impact and stillness check",
        f"peak = {fall['impact_g']:.2f} g (threshold 2.5), impact = {fall['impact']}, "
        f"still afterwards = {fall['post_impact_still']}, fall suspected = {fall['fall_suspected']}"]],
      [2.2, 5.0, 9.8])
P("Run it yourself in front of the audience: <font face='Courier'>python -m mlengine.predict reports/samples/sample_walking.csv</font>. "
  "The second sample, <font face='Courier'>sample_fall.csv</font>, is a staged fall and returns <font face='Courier'>fall_suspected: True</font>. "
  "(That saved model was trained on all 36 people, so use this to show how it works, not as accuracy evidence.)")

# ------------------------------------------------------------------ 5 cheat sheet
P("5. Reading the code: a cheat sheet", h1)
table([["You see", "It means"],
       ["<font face='Courier'>import numpy as np</font>", "Load the numpy library and call it np. numpy handles arrays of numbers fast."],
       ["<font face='Courier'>X[tr]</font>", "Take the rows of X whose positions are listed in tr (here, the training windows)."],
       ["<font face='Courier'>data[\"user\"]</font>", "Look up an entry in a dictionary (a set of named values)."],
       ["<font face='Courier'>[f[name] for name in names]</font>", "A list comprehension: build a list by repeating something for each name."],
       ["<font face='Courier'>f\"{x:.2f}\"</font>", "An f-string: put the value of x into the text with 2 decimals."],
       ["<font face='Courier'>yield</font>", "Hand back one item at a time instead of a whole list (a generator)."],
       ["<font face='Courier'>def name(arg: type) -> type:</font>", "Define a function. The types are hints for readers; Python does not enforce them."],
       ["<font face='Courier'>try: ... except ValueError: continue</font>", "Try something; if it fails with that error, skip to the next item."],
       ["<font face='Courier'>if __name__ == \"__main__\":</font>", "Run this part only when the file is run directly, not when it is imported."],
       ["<font face='Courier'>python -m mlengine.train</font>", "Run the file train.py inside the package mlengine."],
       ["<font face='Courier'>random_state=42</font>", "A fixed starting point for randomness, so the results repeat exactly."],
       ["<font face='Courier'>.fit(X, y)</font> / <font face='Courier'>.predict(X)</font>", "fit = learn from examples (training). predict = answer for new examples."],
       ["<font face='Courier'>joblib.dump / load</font>", "Save a trained model to a file / load it back."]],
      [6.2, 10.8])

# ------------------------------------------------------------------ 6 code Q&A
P("6. Questions about the code, with answers", h1)
table([["Question", "Answer", "Who"],
       ["Which parts did you write yourselves, and which come from a library?",
        "We wrote the cleaning, windowing, the 32 features, the subject-wise testing, the fall rule, the predictor and the demo server. "
        "scikit-learn provides the models (decision trees, random forest, logistic regression); numpy provides arrays and the Fourier transform.", "Moses"],
       ["Why only numpy and scikit-learn?",
        "Fewer dependencies means fewer things to break and an engine that is easy to explain. File parsing, the web server and downloads "
        "use the Python standard library. pandas, matplotlib and reportlab are only used for the optional notebook and report building.", "Moses"],
       ["How do you stop test people leaking into training?",
        "The split is by person: GroupShuffleSplit gets the users array as groups. The demo model records its test_users, and the demo "
        "keeps only those people (np.isin). The notebook prints the overlap between the two groups and it is empty.", "Collins"],
       ["Why is the random seed fixed?",
        "So the split and the forest are the same every run, and the numbers in the report can be reproduced. We pinned the library "
        "versions too, because newer scikit-learn grows slightly different forests (81.7% instead of 84.9% in our test).", "Collins"],
       ["Why is the same feature function used in training and prediction?",
        "If the two were computed differently the model would see inputs unlike anything it learned from. One function guarantees consistency.", "Collins"],
       ["What does n_jobs do?",
        "It sets how many CPU cores scikit-learn uses. Training uses all cores (-1). Predicting one window uses one core, because "
        "starting threads costs more than the work.", "Collins"],
       ["What is joblib, and what is inside the saved file?",
        "joblib saves Python objects to disk. Our file holds the trained random forest and the list of the 32 feature names in the order "
        "the model expects. It is binary, so python -m mlengine.describe prints its contents.", "Moses"],
       ["How are bad lines in the raw file handled?",
        "Lines without 6 fields, unreadable numbers, missing values, zero timestamps and duplicates are skipped, and the clean count "
        "is 1,019,496 readings from 1,086,466 usable lines.", "Moses"],
       ["Is the fall rule machine learning?",
        "No. It is a transparent physics rule (impact of at least 2.5 g followed by stillness) because there are no falls in the "
        "training data. It must be validated on real fall data.", "Kathy"],
       ["Does the demo contain the model?",
        "No. The browser page only draws. The Python server holds the model, computes the answer and returns JSON.", "Kathy"],
       ["How do you know the code works? Do you have tests?",
        "We checked it by running the full pipeline in a clean environment, comparing results to the report (identical), running "
        "the predictor on sample files, and exercising the demo end to end. We have not written automated unit tests; that would be "
        "a next step.", "Moses"],
       ["What would you change to use a different window length?",
        "Change WINDOW (and STEP) at the top of features.py, then retrain, because the features and the model depend on it.", "Collins"],
       ["What if a phone samples at a different rate than 20 Hz?",
        "Our windows assume 20 readings per second. A real app would resample the phone's data to 20 Hz before windowing.", "Moses"],
       ["Why use Python's built-in web server instead of Flask?",
        "It keeps the engine dependency-free. It is fine for a local demo; a production service would use a proper framework.", "Kathy"],
       ["Could this run on the phone?",
        "Not as written. The model would be exported to a mobile format, or the phone would send windows to a server. Neither is built.", "Kathy"]],
      [5.0, 10.4, 1.6])

# ------------------------------------------------------------------ 7 Colab
P("7. Running everything in Google Colab", h1)
P("The notebook <b>MLEngine_Colab.ipynb</b> runs the whole project step by step with explanations. It clones the code from GitHub "
  "and uses the same mlengine functions, so it is the same code, not a copy. It is also a good <b>plan B for the demo</b>: if the "
  "laptop fails on the day, any computer with a browser can run it.")
P("Open it with this link:", lab)
CODE("https://colab.research.google.com/github/Moses374/SemProject_MLEngine/blob/master/MLEngine_Colab.ipynb")
B(["Choose <b>Runtime, then Run all</b>. The first run downloads the 50 MB dataset; the notebook takes a few minutes.",
   "Or run the cells one at a time while presenting. Each section starts with a plain-language explanation, then shows the code, "
   "then its result, so it doubles as a script for explaining the code.",
   "Section 1 checks the library versions. Colab ships newer ones, so the cell prints how to install the pinned versions "
   "(then <b>Runtime, Restart session</b>, then Run all again). Without this the accuracy may differ by a few points.",
   "Sections 2 to 10 show the data, windows, features, split, training, evaluation, the inside of the model, saving, prediction and the fall rule.",
   "Section 11 starts the live demo server inside Colab and shows the two phone screens in the notebook. "
   "<b>Try this once beforehand.</b> The notebook ran cleanly on a laptop, but the embedded demo view uses a Colab feature that "
   "could not be tested outside Colab; if it does not appear, run the cell again."])


def footer(c, d):
    c.saveState(); c.setFont("Times-Roman", 9)
    if d.page > 1:
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
    c.restoreState()


doc = SimpleDocTemplate("reports/Code_Walkthrough.pdf", pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm, title="Code Walkthrough")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("written reports/Code_Walkthrough.pdf")
