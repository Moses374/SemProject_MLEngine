// Builds reports/WISDM_ML_Engine_Presentation.pptx from the figures in reports/fig.
// Run from the project root (after `python -m reports.build_report` has made the figures):
//   NODE_PATH=<dir containing node_modules> node reports/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const FIG = path.join("reports", "fig");
const OUT = process.env.DECK_OUT || path.join("reports", "WISDM_ML_Engine_Presentation.pptx");

const THEME = {
  name: "Plain Academic",
  headFontFace: "Times New Roman",
  bodyFontFace: "Times New Roman",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "333333", lt2: "EEEEEE",
    accent1: "444444", accent2: "666666", accent3: "888888",
    accent4: "AAAAAA", accent5: "CCCCCC", accent6: "222222",
    hlink: "000000", folHlink: "333333",
  },
};

const SCRIPT = JSON.parse(fs.readFileSync(path.join("reports", "script.json"), "utf-8"));
let slideNo = 0; // slides are created in order, so the nth slide takes the nth script entry
function scriptNote(title) {
  const e = SCRIPT.slides[slideNo++];
  if (!e) throw new Error("No script entry for slide " + slideNo);
  console.log(`slide ${slideNo}: "${title}" <- script "${e.title}"`);
  return `PRESENTER: ${e.presenter}

${e.say}` + (e.handoff ? `

HANDOFF: ${e.handoff}` : "");
}
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Activity Recognition and Fall-Risk Detection";
pres.author = "SemProject_MLEngine group";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  slideNumber: { x: 9.2, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: "666666", align: "right" },
  objects: [{
    placeholder: {
      options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.8, fontSize: 34, bold: true,
                 color: C.text1, align: "left", valign: "middle", margin: 0 },
      text: "",
    },
  }],
});
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: "FFFFFF" },
  objects: [{
    placeholder: {
      options: { name: "title", type: "title", x: 0.8, y: 1.3, w: 8.4, h: 1.5, fontSize: 38, bold: true,
                 color: C.text1, align: "center", valign: "middle", margin: 0 },
      text: "",
    },
  }],
});

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// Place an image so it fits inside the box, centred, preserving aspect ratio.
function fig(slide, name, x, y, w, h, alt) {
  const file = path.join(FIG, name + ".png");
  const s = pngSize(file);
  const k = Math.min(w / s.w, h / s.h);
  const iw = s.w * k, ih = s.h * k;
  slide.addImage({ path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih,
                   altText: alt, objectName: "Figure " + name });
}
function text(slide, str, x, y, w, h, opts = {}) {
  slide.addText(str, Object.assign({ x, y, w, h, fontSize: 16, color: C.text1, align: "left",
                                     valign: "top", margin: 0, isTextBox: true }, opts));
}
function bullets(slide, items, x, y, w, h, fontSize = 16) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1,
                                                            paraSpaceAfter: 8 } }));
  text(slide, runs, x, y, w, h, { fontSize });
}
function caption(slide, str, x, y, w) {
  text(slide, str, x, y, w, 0.3, { fontSize: 11, italic: true, align: "center", color: C.text2 });
}
function content(title, notes, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addNotes(scriptNote(title));
  return s;
}
function table(slide, rows, x, y, w, colW, fontSize = 13, rowH = 0) {
  const data = rows.map((r, i) => r.map((c) => ({
    text: String(c),
    options: { bold: i === 0, fontSize, color: C.text1, fontFace: "Times New Roman",
               fill: i === 0 ? { color: C.background2 } : { color: C.background1 } },
  })));
  slide.addTable(data, { x, y, w, colW, rowH: rowH || undefined, border: { type: "solid", pt: 0.75, color: "888888" },
                         margin: [0.04, 0.08, 0.04, 0.08], valign: "middle" });
}

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  s.addText("Activity Recognition and Fall-Risk Detection from Smartphone Accelerometer Data",
            { placeholder: "title" });
  text(s, "ICS 3202 Semester Project: Machine Learning Engine", 0.8, 3.3, 8.4, 0.4,
       { fontSize: 18, align: "center" });
  text(s, "Dataset: WISDM Activity Recognition (Kwapisz, Weiss and Moore, 2011)", 0.8, 3.75, 8.4, 0.4,
       { fontSize: 14, align: "center" });
  text(s, "Moses Kibet (183638)   |   Collins Jared (184587)   |   Kathy Auma (192231)", 0.8, 4.4, 8.4, 0.4,
       { fontSize: 14, align: "center" });
  s.addNotes(scriptNote("Title"));
}

// ---------------------------------------------------------------- 2 problem
{
  const s = content("The problem and our goal",
    "A fall that goes unnoticed is far more dangerous than one reported quickly. Every smartphone has an " +
    "accelerometer, so the sensor is already in the pocket of the people we want to protect.", "Introduction");
  bullets(s, [
    "Falls are a leading cause of injury in elderly people, and delays in help make them worse.",
    "Every smartphone has an accelerometer, a sensor measuring acceleration along three axes (x, y, z).",
    "Goal: classify what the person is doing, then flag a likely fall to a caregiver.",
  ], 0.5, 1.4, 5.0, 3.6, 18);
  const rows = [["Part", "Job"], ["Classifier", "Walking, Jogging, Upstairs, Downstairs, Sitting or Standing?"],
                ["Fall rule", "Hard impact followed by stillness?"]];
  table(s, rows, 5.8, 1.5, 3.7, [1.1, 2.6], 15);
}

// ---------------------------------------------------------------- how ML works
{
  const s = content("How machine learning works here", null, "Introduction");
  fig(s, "learn_vs_use", 0.5, 1.15, 9, 3.2, "Learning stage and using stage diagram");
  bullets(s, [
    "We never wrote rules for walking. We gave the computer 25,230 labelled examples and it found the patterns.",
    "Training happens once. The saved model is then reused for every new window.",
  ], 0.5, 4.4, 9, 0.9, 14);
}

// ---------------------------------------------------------------- 3 pipeline
{
  const s = content("The pipeline",
    "Walk through left to right: raw file, cleaning, windowing, features, training, saved model. The saved " +
    "model gives the activity; the fall rule runs separately on the same window.", "Introduction");
  fig(s, "pipeline", 0.5, 1.3, 9, 3.0, "Pipeline diagram: raw file, clean, windows, features, train, model");
  text(s, "Everything is a small Python package (mlengine/): data, features, train, predict.",
       0.5, 4.5, 9, 0.5, { fontSize: 16, align: "center" });
}

// ---------------------------------------------------------------- 4 dataset
pres.addSection({ title: "Data" });
{
  const s = content("The WISDM dataset",
    "Volunteers carried an Android phone in the front trouser pocket and did six activities. When still, " +
    "total acceleration equals gravity (about 9.81 m/s2, written 1 g). Gravity falls on different axes " +
    "depending on orientation, which is how sitting and standing differ.", "Data");
  const stats = [["36", "subjects"], ["20 Hz", "sampling rate"], ["1,019,496", "readings after cleaning"],
                 ["6", "activities"]];
  stats.forEach((st, i) => {
    const x = 0.5 + (i % 2) * 2.4, y = 1.4 + Math.floor(i / 2) * 1.6;
    s.addShape(pres.ShapeType.rect, { x, y, w: 2.2, h: 1.4, fill: { color: C.background2 },
                                       line: { color: "888888", width: 0.75 }, objectName: "Stat card " + (i + 1) });
    text(s, st[0], x, y + 0.2, 2.2, 0.6, { fontSize: 26, bold: true, align: "center" });
    text(s, st[1], x + 0.1, y + 0.85, 2.0, 0.45, { fontSize: 14, align: "center" });
  });
  fig(s, "classes", 5.4, 1.3, 4.2, 3.0, "Bar chart of readings per activity");
  caption(s, "Readings per activity: Walking and Jogging are about two thirds of the data", 5.4, 4.35, 4.2);
}

// ---------------------------------------------------------------- 5 exploration
{
  const s = content("Data exploration (Deliverable 1)",
    "These answers come from the notebook WISDM_Exploration.ipynb. One z-accel value is missing, so " +
    "strictly the dataset is not 100% complete.", "Data");
  table(s, [
    ["Question", "Answer"],
    ["Rows and columns", "1,086,466 rows, 6 columns"],
    ["Datatypes", "user and timestamp: integer; activity: text; x, y, z: float"],
    ["Complete?", "Nearly: exactly 1 missing value (a z-accel reading); a few malformed lines skipped"],
    ["df_sample", "First 15 and last 20 rows merged (35 rows)"],
  ], 0.5, 1.4, 9, [2.2, 6.8], 16, 0.5);
  text(s, "Cleaning removes malformed lines, missing values, zero timestamps and duplicates: " +
          "1,086,466 parsed lines become 1,019,496 readings (6.2% removed).", 0.5, 4.2, 9, 0.8, { fontSize: 16 });
}

// ---------------------------------------------------------------- 6 signals
{
  const s = content("What the signals look like",
    "Jogging swings strongly, walking has a regular rhythm, sitting and standing are almost flat. Upstairs " +
    "and downstairs resemble walking, which foreshadows where the classifier will struggle.", "Data");
  fig(s, "signals", 0.5, 1.25, 9, 3.5, "Acceleration magnitude for four seconds of each activity");
  caption(s, "Acceleration magnitude |a| for four seconds of each activity", 0.5, 4.8, 9);
}

// ---------------------------------------------------------------- 7 windows
pres.addSection({ title: "Method" });
{
  const s = content("Windowing: turning a stream into examples",
    "A single reading tells you almost nothing. Behaviour over a few seconds is what distinguishes " +
    "activities. Overlap gives more examples and avoids splitting an event unluckily.", "Method");
  fig(s, "windows", 0.5, 1.25, 9, 2.2, "Sliding windows with 50 percent overlap");
  bullets(s, [
    "Window = 80 readings = 4 seconds; each starts 2 seconds after the last (50% overlap).",
    "Only windows with one subject and one activity are kept, so each has a single label.",
    "Result: 25,230 labelled windows.",
  ], 0.5, 3.6, 9, 1.6, 15);
}

// ---------------------------------------------------------------- 8 features
{
  const s = content("Features: summarising each window",
    "Each window is 240 raw numbers. We summarise it into 32 features, which suits classical models and " +
    "is easy to explain.", "Method");
  table(s, [
    ["Feature group", "What it measures", "Why it helps"],
    ["Mean (per axis)", "Average on x, y, z", "Orientation: sitting vs standing"],
    ["Std, min, max, MAD", "Spread of the signal", "Vigorous vs calm movement"],
    ["Zero-crossing rate", "Crossings of the mean", "Step rhythm"],
    ["Dominant frequency, energy", "Repeat rate and vibration (FFT)", "Walking vs jogging"],
    ["Magnitude statistics", "Statistics of |a|", "Overall movement intensity"],
    ["Axis correlations", "How x, y, z move together", "Direction of motion"],
  ], 0.5, 1.4, 9, [2.7, 3.2, 3.1], 14);
  text(s, "32 features per window in total.", 0.5, 4.75, 9, 0.35, { fontSize: 14, italic: true });
}

// ---------------------------------------------------------------- decision tree
{
  const s = content("A decision tree: twenty questions", null, "Method");
  fig(s, "tree", 0.5, 1.15, 9, 3.3, "Top of a real decision tree from the trained model");
  bullets(s, [
    "Real tree from our model: y_mad is how much the up-down reading wiggles.",
    "Nobody chose these questions. Training picks the ones that best split the labelled examples.",
  ], 0.5, 4.45, 9, 0.9, 14);
}
// ---------------------------------------------------------------- forest
{
  const s = content("A forest: 300 trees vote", null, "Method");
  fig(s, "votes", 0.5, 1.15, 9, 3.0, "Votes of 300 trees on a real Walking window and a real Downstairs window");
  bullets(s, [
    "Many trees, each trained on a random sample, vote. A close vote means low confidence.",
    "On its own training windows the forest scores 100%; on unseen people 85%. We report 85%.",
  ], 0.5, 4.25, 9, 1.0, 14);
}

// ---------------------------------------------------------------- 9 evaluation
{
  const s = content("Evaluating honestly: subject-wise split",
    "If windows are shuffled randomly, near-identical neighbouring windows from the same person land in " +
    "both train and test, inflating the score. A real app must work on new people.", "Method");
  fig(s, "split", 0.5, 1.25, 9, 1.6, "Training set of 27 subjects and test set of 9 unseen subjects");
  bullets(s, [
    "Whole subjects are held out: 27 train, 9 test. No person appears on both sides.",
    "Models: majority-class baseline, logistic regression, random forest (300 trees).",
    "Metrics: accuracy, and macro F1 (averages all six activities equally, so small classes count).",
  ], 0.5, 3.0, 9, 2.0, 15);
}

// ---------------------------------------------------------------- 10 results
pres.addSection({ title: "Results" });
{
  const s = content("Results",
    "Both real models far outperform the baseline. Random forest is best and is the saved model. Five-fold " +
    "cross-validation grouped by subject gives macro F1 0.791 plus or minus 0.035, so the result is stable.",
    "Results");
  fig(s, "models", 0.4, 1.25, 5.0, 3.3, "Accuracy and macro F1 for three models");
  table(s, [
    ["Model", "Accuracy", "Macro F1"],
    ["Baseline", "0.391", "0.094"],
    ["Logistic regression", "0.829", "0.749"],
    ["Random forest", "0.849", "0.775"],
  ], 5.6, 1.5, 4.0, [1.9, 1.0, 1.1], 14);
  text(s, "5-fold subject-wise CV (random forest): macro F1 0.791 ± 0.035", 5.6, 3.3, 4.0, 0.9,
       { fontSize: 14 });
  text(s, "Unseen people: 85%. Its own training windows: 100% (memorised). We report 85%.", 5.6, 4.0, 4.0, 0.9,
       { fontSize: 14, bold: true });
}

// ---------------------------------------------------------------- 11 confusion
{
  const s = content("Where the model succeeds and struggles",
    "Jogging and Walking are reliable. Downstairs recall is only 0.34: most true Downstairs windows are " +
    "labelled Upstairs or Walking, which is expected from a pocket phone. Sitting and Standing confuse " +
    "each other because they differ mostly in tilt.", "Results");
  fig(s, "confusion", 0.4, 1.2, 4.7, 3.9, "Confusion matrix on held-out subjects");
  table(s, [
    ["Activity", "Precision", "Recall", "F1"],
    ["Jogging", "0.96", "0.97", "0.96"],
    ["Walking", "0.92", "0.90", "0.91"],
    ["Standing", "0.78", "0.81", "0.80"],
    ["Sitting", "0.76", "0.80", "0.78"],
    ["Upstairs", "0.61", "0.86", "0.72"],
    ["Downstairs", "0.78", "0.34", "0.48"],
  ], 5.3, 1.35, 4.3, [1.5, 1.0, 0.9, 0.9], 13);
  text(s, "Downstairs is the weak class: it is mostly mistaken for Upstairs and Walking.", 5.3, 4.3, 4.3, 0.8,
       { fontSize: 14 });
}

// ---------------------------------------------------------------- 12 fall rule
{
  const s = content("Fall-risk detection",
    "WISDM has no recorded falls, so a fall classifier cannot be trained from it. The rule uses physics: " +
    "impact, then stillness. Requiring stillness stops footsteps from triggering alarms. The thresholds " +
    "are unvalidated.", "Results");
  fig(s, "fall", 0.5, 1.2, 9, 2.4, "Fall rule on a walking window and a simulated fall");
  bullets(s, [
    "Flag when peak acceleration is at least 2.5 g AND the last 0.5 s is still, near 1 g.",
    "Walking and jogging exceed 2.5 g on every step but keep moving, so they are not flagged.",
    "Caveat: checked only on a simulated signal. Needs validation on real fall data (SisFall, MobiAct).",
  ], 0.5, 3.7, 9, 1.5, 14);
}

// ---------------------------------------------------------------- 13 live demo
{
  const s = content("Live demonstration", null, "Results");
  fig(s, "demo_fall", 0.4, 1.2, 5.4, 3.5, "Screenshot of the demo page showing a fall alert");
  caption(s, "Backup screenshot, in case the live demo cannot run", 0.4, 4.75, 5.4);
  text(s, "What we will do live", 6.1, 1.3, 3.5, 0.4, { fontSize: 18, bold: true });
  const steps = ["Walk and Jog: Mary's phone names the activity, the caregiver sees it",
                 "Stairs down: our known weak spot, shown honestly",
                 "Simulate a fall: \"Are you OK?\" countdown, then a caregiver alert",
                 "Technical view: the evidence behind each decision"];
  const runs = steps.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < steps.length - 1,
                                                          paraSpaceAfter: 10 } }));
  text(s, runs, 6.1, 1.8, 3.5, 3.0, { fontSize: 15 });
}

// ---------------------------------------------------------------- under the hood
{
  const s = content("Under the hood", null, "Results");
  table(s, [
    ["File", "Job"],
    ["data.py", "Load and clean the raw file"],
    ["features.py", "Windows, 32 features, fall rule"],
    ["train.py", "Compare models, test on unseen people, save"],
    ["predict.py", "Load the model, classify one window"],
    ["demo.py + page", "Local server and the app-style page"],
  ], 0.5, 1.4, 5.0, [1.6, 3.4], 13, 0.5);
  text(s, "mlengine/ package", 0.5, 1.1, 5.0, 0.3, { fontSize: 13, italic: true });
  text(s, "The saved model", 5.8, 1.1, 3.7, 0.3, { fontSize: 16, bold: true });
  bullets(s, [
    "models/activity_model.joblib, about 22 MB",
    "300 decision trees, 32 inputs, 6 outputs",
    "Not on GitHub: rebuilt in about a minute with python -m mlengine.train",
    "A prediction takes about 30 ms",
  ], 5.8, 1.5, 3.7, 2.4, 13);
  text(s, "Demo: browser asks the Python server for a window; the server computes features, asks the model, runs the fall rule and replies with JSON.",
       0.5, 4.5, 9, 0.7, { fontSize: 14 });
}

// ---------------------------------------------------------------- 13 limitations
pres.addSection({ title: "Conclusion" });
{
  const s = content("Limitations and future work",
    "Be upfront: the fall rule is not validated, WISDM subjects are not elderly, and there is no lying-down " +
    "class.", "Conclusion");
  table(s, [
    ["Limitation", "Next step"],
    ["No real fall data; rule unvalidated", "Evaluate or train on SisFall / MobiAct"],
    ["No lying-down class", "Add a dataset that includes it"],
    ["Young adult volunteers", "Test on elderly participants"],
    ["Trouser pocket only", "Collect hand-held and bag placements"],
    ["Weak on Downstairs", "Gravity-removed signal; neural network on raw windows"],
    ["Python library, not an app", "Wrap in a phone app or web service"],
  ], 0.5, 1.4, 9, [4.2, 4.8], 16, 0.48);
}

// ---------------------------------------------------------------- 14 conclusion
{
  const s = content("Conclusion",
    "Summarise: careful cleaning, windowing and subject-wise evaluation make these numbers a realistic " +
    "estimate. The fall rule is a clearly labelled first step.", "Conclusion");
  bullets(s, [
    "Recognises six activities at about 85% accuracy (macro F1 0.78) on unseen people, versus 39% / 0.09 for the baseline.",
    "Careful cleaning, windowing and subject-wise testing make the numbers a realistic estimate.",
    "A transparent fall rule sits on top, with a clear statement of what is and is not validated.",
    "Reproduce: python -m mlengine.train",
  ], 0.5, 1.4, 9, 3.4, 18);
  text(s, "Reference: Kwapisz, Weiss and Moore (2011). Activity Recognition using Cell Phone Accelerometers. " +
          "ACM SIGKDD Explorations, 12(2), 74-82.", 0.5, 4.7, 8.6, 0.5, { fontSize: 11, italic: true });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
