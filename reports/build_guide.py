"""Build reports/Presenter_Guide.pdf from reports/script.json.

Run from the project root:  python -m reports.build_guide
"""
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

D = json.load(open("reports/script.json", encoding="utf-8"))
SLIDES, TERMS, QA, DEMO, BACK = D["slides"], D["terms"], D["qa"], D["demo"], D["backend"]
PEOPLE = ["Moses", "Collins", "Kathy"]

body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11.5, leading=16, alignment=TA_JUSTIFY,
                      textColor=colors.black, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=body, fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=12,
                    spaceAfter=6, alignment=0, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Times-Bold", fontSize=12.5, leading=16, spaceBefore=10,
                    spaceAfter=3, alignment=0, keepWithNext=1)
lab = ParagraphStyle("lab", parent=body, fontName="Times-Bold", spaceAfter=1, alignment=0, keepWithNext=1)
title = ParagraphStyle("title", parent=body, fontName="Times-Bold", fontSize=24, leading=30,
                       alignment=TA_CENTER, spaceAfter=10)
sub = ParagraphStyle("sub", parent=body, fontSize=13, alignment=TA_CENTER, leading=18)
cell = ParagraphStyle("cell", parent=body, fontSize=10.5, leading=13, alignment=0, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Times-Bold")

S = []


def P(t, st=body):
    S.append(Paragraph(t, st))


def table(rows, widths):
    data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LINEBELOW", (0, 0), (-1, 0), 1.2, colors.black)]))
    S.append(t)
    S.append(Spacer(1, 8))


# ---- cover
S.append(Spacer(1, 5 * cm))
P("Presenter's Guide", title)
P("Activity Recognition and Fall-Risk Detection from Smartphone Accelerometer Data", sub)
P("A plain-language explanation of the project, a slide-by-slide script, and how the three of us share the presentation", sub)
S.append(Spacer(1, 2 * cm))
P("Moses Kibet (183638)<br/>Collins Jared (184587)<br/>Kathy Auma (192231)", sub)
S.append(PageBreak())

# ---- 1 whole project simply
P("1. The whole project in plain words", h1)
P("<b>What we built.</b> A program that looks at a few seconds of movement data from a phone and says what the "
  "person was doing: walking, jogging, going upstairs, going downstairs, sitting or standing. On top of that, "
  "a simple rule checks whether the movement looks like a fall.")
P("<b>Where the data comes from.</b> Every smartphone has an accelerometer, a sensor that measures movement in "
  "three directions. A public dataset called WISDM contains recordings from 36 volunteers who carried a phone "
  "in a trouser pocket while doing the six activities. Each recording is labelled with the activity, so a "
  "program can learn from it.")
P("<b>How it works, in five steps.</b>")
for n, t in enumerate([
    "<b>Clean</b> the raw file. Real sensor files contain broken lines and duplicates; we remove them.",
    "<b>Cut into windows.</b> One reading means almost nothing; a 4-second slice shows the pattern of movement.",
    "<b>Summarise each window</b> into 32 numbers (features), such as the average and how much it wiggles.",
    "<b>Train a model.</b> We show the model thousands of labelled windows so it learns which patterns "
    "belong to which activity. We compared three models and kept the best (a random forest).",
    "<b>Test fairly.</b> We test on people the model never saw, since that is what a real user would be."], 1):
    P(f"{n}. {t}")
P("<b>Training versus using.</b> Training is done once: the computer is shown 25,230 labelled windows and works out "
  "the patterns itself, producing a model, which is just a file on disk (models/activity_model.joblib). Using the model "
  "is what happens every time new data arrives: the same 32 features are computed and the saved model answers in about "
  "30 milliseconds. No learning happens at that point.")
P("<b>What we found.</b> The best model is correct about 85% of the time on new people. Always guessing the most "
  "common activity would be right only 39% of the time. The model is excellent at jogging and walking but weak "
  "at downstairs, which it often confuses with upstairs and walking.")
P("<b>The honest caveat on falls.</b> WISDM has no recordings of falls, so we could not train a fall detector. "
  "We wrote a rule from physics (a hard impact, then stillness) and tested it only on a simulated signal. It "
  "needs real fall data before anyone relies on it. Saying this openly is a strength, not a weakness.")

# ---- 2 who presents
P("2. Who presents what", h1)
P("Each person presents a connected part, so every member speaks and can answer questions about what they "
  "explain. Handoff lines are written into the script so the changeover is smooth. You can swap the parts "
  "between you; just keep each block together.")
blocks = {}
for i, s in enumerate(SLIDES, 1):
    blocks.setdefault(s["presenter"], []).append((i, s["title"]))
table([["Presenter", "Slides", "Topic", "Approx. time"],
       ["Moses", "1 to 6, 17 and 19", "Introduction, problem, how machine learning works here, pipeline, dataset, "
        "data exploration; then the backend (\"Under the hood\") and the conclusion", "8 to 9 min"],
       ["Collins", "7 to 12, and runs the demo", "Signals, windowing, features, how a decision tree and a random forest "
        "decide, fair testing; operates the live demo on slide 16", "8 to 9 min"],
       ["Kathy", "13 to 16, and 18", "Results, where the model struggles, fall detection, narrates the demo, "
        "limitations", "7 to 8 min"]],
      [2.4, 3.0, 8.0, 2.2])
P("The whole talk takes about 24 to 28 minutes including the demo, at roughly one minute per slide, leaving time for questions. "
  "<b>Questions:</b> whoever owns that part of the work answers it (the question list in Section 7 shows who). "
  "If someone is unsure, a teammate can add to the answer. If a question goes beyond our data, say so "
  "plainly and give your best reasoned expectation.")

# ---- 3 slide by slide
P("3. Slide-by-slide script", h1)
P("For each slide: what is on it, who speaks, and what to say. The wording is plain on purpose. Say it in your "
  "own words; do not read it out. The same text is in the speaker notes of the PowerPoint.")
for i, s in enumerate(SLIDES, 1):
    blk = [Paragraph(f"Slide {i}: {s['title']}  ({s['presenter']})", h2),
           Paragraph("On the slide", lab), Paragraph(s["on_slide"], body),
           Paragraph("What to say", lab), Paragraph(s["say"], body)]
    if s["handoff"]:
        blk += [Paragraph("Handoff", lab), Paragraph(s["handoff"], body)]
    S.append(KeepTogether(blk))

# ---- 4 demo
S.append(PageBreak())
P("4. The live demonstration", h1)
P("The demo is an app-style web page that runs on one laptop. It shows two phone screens: <b>Mary's phone</b>, which "
  "senses her movement, names the activity and asks \"Are you OK?\" after a fall, and the <b>caregiver's phone</b>, which "
  "shows her status in plain language and receives the alert. The activities are real four-second recordings from people "
  "the model never trained on; the fall is staged; calls and notifications are simulated. A \"Technical view\" "
  "switch shows the graphs and numbers behind each decision. <b>Collins operates it and Kathy explains</b>, so the "
  "audience hears one voice while the other drives. Moses watches the clock and steps in if something goes wrong.")
P("Before the day: set it up", h2)
for i, t in enumerate(DEMO["setup"]):
    P(("" if i == 0 else f"{i}. ") + t)
P("During the presentation: what to click and what to say", h2)
table([["Step", "Action", "What it shows / what to say"]] + [[i + 1, a, w] for i, (a, w) in enumerate(DEMO["run"])],
      [1.2, 4.2, 11.6])
P("If something goes wrong", h2)
table([["Problem", "Fix"]] + DEMO["trouble"], [5.2, 11.8])
P("<b>Be honest on stage.</b> If the model gets an activity wrong, do not hide it or click again quickly. "
  "Say: \"That is our known weak spot, and the scoreboard records it.\" Markers value that honesty.")

# ---- 5 backend
S.append(PageBreak())
P("5. If they ask to see the backend", h1)
P(BACK["intro"])
P("What is where", h2)
table([["Path", "What it is"]] + BACK["layout"], [6.2, 10.8])
P("What to show, in order", h2)
table([["Step", "Show", "What to do", "What to say"]] +
      [[i + 1, a, b, c] for i, (a, b, c) in enumerate(BACK["steps"])], [1.3, 3.1, 6.5, 6.1])
P("<b>Rule for all of it:</b> explain the idea first (a window goes in, 32 features are computed, the saved "
  "model answers), then show the file that does it. Never scroll through code you cannot explain.")

# ---- 6 terms
S.append(PageBreak())
P("6. Key terms in one line each", h1)
P("Learn these. If you can say each one in your own words, you can answer most questions.")
table([["Term", "Meaning"]] + TERMS, [3.6, 13.4])

# ---- 5 Q&A
P("7. Questions you are likely to be asked", h1)
table([["Question", "Answer", "Who answers"]] + [[q, a, o] for q, a, o in QA], [4.4, 10.2, 2.4])

# ---- 6 tips
P("9. Colab and the code walkthrough", h1)
P("<b>Colab notebook</b> (runs the whole project step by step and the live demo in a browser, nothing to install):")
S.append(Preformatted("https://colab.research.google.com/github/Moses374/SemProject_MLEngine/blob/master/MLEngine_Colab.ipynb",
                      ParagraphStyle("u", fontName="Courier", fontSize=8, leading=10)))
S.append(Spacer(1, 6))
P("Choose Runtime, then Run all. Try it once before the day. It is the best plan B if the laptop fails. "
  "<b>Code walkthrough:</b> reports/Code_Walkthrough.pdf explains every file with the real code, who explains it, "
  "a worked example and questions about the code.")

P("8. Preparing and delivering", h1)
for t in ["<b>Rehearse your own part out loud</b> at least twice, then once all together to practise the handoffs and the demo.",
          "<b>Look at the audience, not the screen.</b> Glance at the slide only to point to something.",
          "<b>Point at the figures.</b> For each graph, say what the horizontal axis is, what the vertical axis is, "
          "and the one thing to notice.",
          "<b>Use the numbers that matter:</b> 36 subjects, 25,230 windows, 85% accuracy, 39% baseline, "
          "34% downstairs recall. You do not need to recite every figure.",
          "<b>Be upfront about limits.</b> Markers respond well to honesty about what was and was not validated.",
          "<b>Check the files</b> before the day: open the PowerPoint, run through all slides, and keep a "
          "copy of the PDF report as a backup.",
          "<b>If a number escapes you,</b> say it is in the report and give the nearest round figure."]:
    P("&bull; " + t)


def footer(c, d):
    c.saveState()
    c.setFont("Times-Roman", 9)
    if d.page > 1:
        c.drawCentredString(A4[0] / 2, 1.2 * cm, str(d.page))
    c.restoreState()


doc = SimpleDocTemplate("reports/Presenter_Guide.pdf", pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2.2 * cm, bottomMargin=2.2 * cm, title="Presenter's Guide")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("written reports/Presenter_Guide.pdf")
