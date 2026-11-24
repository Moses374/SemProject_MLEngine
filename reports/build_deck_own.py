"""Build reports/WISDM_ML_Engine_Presentation_OwnModels.pptx: the deck for OUR OWN model implementations.

Starts from the scikit-learn deck (reports/WISDM_ML_Engine_Presentation.pptx) so the design stays identical, then
swaps in the figures from reports/own/fig, updates the model-dependent text and numbers, and rewrites the speaker
notes from reports/own/script.json.

Run from the project root, after reports.build_ml_figs_own and reports.build_report_own:
    python -m reports.build_deck_own
"""
import json
import os

from PIL import Image
from pptx import Presentation

SRC = os.path.join("reports", "WISDM_ML_Engine_Presentation.pptx")
OUT = os.path.join("reports", "WISDM_ML_Engine_Presentation_OwnModels.pptx")
FIG = os.path.join("reports", "own", "fig")
SCRIPT = json.load(open(os.path.join("reports", "own", "script.json"), encoding="utf-8"))
N = json.load(open(os.path.join("reports", "own", "numbers.json"), encoding="utf-8"))
F = json.load(open(os.path.join("reports", "own", "ml_facts.json"), encoding="utf-8"))

res, ref, pc = N["results"], N["reference"], N["per_class"]
cv_mean, cv_std = N["cv_macro_f1"]
deployed_mb = os.path.getsize(os.path.join("models", "activity_model.joblib")) / 1e6 \
    if os.path.exists(os.path.join("models", "activity_model.joblib")) else F["model_file_mb_27_subjects"] * 36 / 27
ms = round(F["ms_per_prediction"], -1)

# Per slide (1-based): exact text to find in a paragraph or table cell, and what replaces it.
TEXT = {
    1: [("ICS 3202 Semester Project: Machine Learning Engine",
         "ICS 3202 Semester Project: ML Engine with Our Own Models")],
    3: [("and it found the patterns.", "and our own training code found the patterns.")],
    4: [("data, features, train, predict.", "data, features, models, train, predict.")],
    10: [("Nobody chose these questions. Training picks", "Nobody chose these questions. Our training code picks")],
    12: [("Models: majority-class baseline, logistic regression, random forest (300 trees).",
          "Models, all written by us: baseline, logistic regression, random forest (300 trees).")],
    17: [("about 22 MB", f"about {deployed_mb:.0f} MB"),
         ("300 decision trees, 32 inputs, 6 outputs", "300 decision trees grown by our code, 32 inputs, 6 outputs"),
         ("rebuilt in about a minute", "rebuilt in a few minutes"),
         ("about 30 ms", f"about {ms:.0f} ms")],
    19: [("Careful cleaning, windowing and subject-wise testing make the numbers a realistic estimate.",
          "Our own models match scikit-learn on the same data; subject-wise testing makes the numbers realistic.")],
}

# Per slide: table rows keyed by their first cell, with the full new row.
def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


TABLES = {
    13: {"Model": ["Model (ours)", "Accuracy", "Macro F1"],
         "Baseline": ["Baseline", f3(res["Majority-class baseline"][0]), f3(res["Majority-class baseline"][1])],
         "Logistic regression": ["Logistic regression", f3(res["Logistic regression"][0]), f3(res["Logistic regression"][1])],
         "Random forest": ["Random forest", f3(res["Random forest"][0]), f3(res["Random forest"][1])]},
    14: {a: [a, f2(pc[a]["precision"]), f2(pc[a]["recall"]), f2(pc[a]["f1"])]
         for a in ["Jogging", "Walking", "Standing", "Sitting", "Upstairs", "Downstairs"]},
    17: {"train.py": ["models.py, train.py", "Our own models; compare, test on unseen people, save"]},
}

# Results slide: the cross-validation line also carries the scikit-learn check.
CV_OLD = "5-fold subject-wise CV (random forest): macro F1"
CV_NEW = [f"5-fold subject-wise CV: macro F1 {cv_mean:.3f} ± {cv_std:.3f}",
          f"scikit-learn's forest, same split: {ref['Random forest'][0]:.3f} / {ref['Random forest'][1]:.3f}"]


def set_paragraph_text(p, new):
    """Replace a paragraph's text, keeping the formatting of its first run."""
    runs = p.runs
    runs[0].text = new
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def replace_text(slide, old, new):
    hits = 0
    for sh in slide.shapes:
        frames = []
        if sh.has_text_frame:
            frames.append(sh.text_frame)
        if getattr(sh, "has_table", False) and sh.has_table:
            frames += [c.text_frame for row in sh.table.rows for c in row.cells]
        for tf in frames:
            for p in tf.paragraphs:
                if old in p.text:
                    set_paragraph_text(p, p.text.replace(old, new))
                    hits += 1
    if hits != 1:
        raise SystemExit(f"expected one match for {old!r}, found {hits}")


def replace_rows(slide, rows):
    table = next(sh.table for sh in slide.shapes if getattr(sh, "has_table", False) and sh.has_table)
    seen = set()
    for row in table.rows:
        key = row.cells[0].text
        if key in rows:
            for cell, value in zip(row.cells, rows[key]):
                set_paragraph_text(cell.text_frame.paragraphs[0], value)
            seen.add(key)
    missing = set(rows) - seen
    if missing:
        raise SystemExit(f"table rows not found: {missing}")


def replace_cv_line(slide):
    box = next(sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.startswith(CV_OLD))
    p = box.text_frame.paragraphs[0]
    set_paragraph_text(p, CV_NEW[0])
    p.add_line_break()
    r = p.add_run()
    r.text = CV_NEW[1]
    src = p.runs[0].font
    r.font.size, r.font.bold, r.font.name = src.size, src.bold, src.name


def replace_figures(slide):
    """Swap each 'Figure <name>' picture for reports/own/fig/<name>.png, fitted inside the same box."""
    for sh in slide.shapes:
        if sh.shape_type != 13 or not sh.name.startswith("Figure "):
            continue
        path = os.path.join(FIG, sh.name[len("Figure "):] + ".png")
        if not os.path.exists(path):  # e.g. the demo screenshot, which does not depend on the model
            continue
        blip = sh._element.blipFill.blip
        part = slide.part.related_part(blip.rEmbed)
        part._blob = open(path, "rb").read()
        w, h = Image.open(path).size
        cx, cy = sh.left + sh.width / 2, sh.top + sh.height / 2
        k = min(sh.width / w, sh.height / h)
        sh.width, sh.height = int(w * k), int(h * k)
        sh.left, sh.top = int(cx - sh.width / 2), int(cy - sh.height / 2)


def note(entry):
    text = f"PRESENTER: {entry['presenter']}\n\n{entry['say']}"
    return text + (f"\n\nHANDOFF: {entry['handoff']}" if entry["handoff"] else "")


pres = Presentation(SRC)
slides = list(pres.slides)
if len(slides) != len(SCRIPT["slides"]):
    raise SystemExit(f"{len(slides)} slides but {len(SCRIPT['slides'])} script entries")
for i, slide in enumerate(slides, 1):
    replace_figures(slide)
    for old, new in TEXT.get(i, []):
        replace_text(slide, old, new)
    if i in TABLES:
        replace_rows(slide, TABLES[i])
    if i == 13:
        replace_cv_line(slide)
    slide.notes_slide.notes_text_frame.text = note(SCRIPT["slides"][i - 1])
pres.core_properties.title = "Activity Recognition and Fall-Risk Detection (own models)"
pres.save(OUT)
print("written", OUT)
