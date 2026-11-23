# SemProject_MLEngine

ICS 3202 Semester Project, Deliverable 1: Dataset Exploration.

## Research area

Fall detection and activity monitoring for elderly care, using smartphone accelerometer data. The eventual application reads motion sensor data from a phone, classifies the activity a person is doing, and flags a likely fall to a caregiver or family member.

## Dataset

**WISDM (Wireless Sensor Data Mining) - Activity Recognition using Cell Phone Accelerometers**

- Source: Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011), WISDM Lab, Fordham University.
- Public mirror used: https://huggingface.co/datasets/stark0908/WISDM
- Raw file: `WISDM_ar_v1.1_raw.txt` (triaxial accelerometer readings from 36 subjects across 6 activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
- Expected output variable: `activity`

## Notebook

[WISDM_Exploration.ipynb](WISDM_Exploration.ipynb) downloads the raw dataset and answers:

a. Row and column counts
b. Column datatypes
c. Whether the dataset is complete (missing values)
d. First 15 and last 20 rows merged into `df_sample`

## Running it

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace WISDM_Exploration.ipynb
```

The notebook downloads the ~50MB raw data file on first run and caches it locally as `WISDM_ar_v1.1_raw.txt` (not tracked in git).

## ML engine

The `mlengine/` package turns the raw accelerometer stream into an activity classifier plus a fall-risk flag.

| Module | Role |
|---|---|
| `mlengine/data.py` | Load the raw file; drop malformed rows, NaNs, zero timestamps and duplicates |
| `mlengine/features.py` | 4 s sliding windows (80 samples, 50% overlap), 32 time/frequency features per window, fall-risk rule |
| `mlengine/train.py` | Train baseline, logistic regression and random forest; evaluate; save model and reports |
| `mlengine/predict.py` | Classify one 80-sample window and report fall risk |
| `mlengine/demo.py` | Local web demo (page in `mlengine/demo_page.html`) |

```
python -m mlengine.train                 # writes models/ and reports/
python -m mlengine.predict readings.csv  # CSV with x,y,z columns (m/s^2, 20 Hz)
```

### Live demo

```
python -m mlengine.train   # once; also saves models/demo_model.joblib
python -m mlengine.demo    # then open http://localhost:8000
```

An app-style demo with two phone screens. Mary's phone names her current activity (with an honest "likely" when the model is unsure), draws live motion and a daily activity bar, and after a suspected fall asks "Are you OK?" with a 10-second countdown. The caregiver's phone shows her status and history and receives an urgent alert if she does not answer or asks for help. Activities are real windows from the 9 held-out subjects (people the demo model never trained on); the fall is staged because WISDM has no falls; calls and notifications are simulated. A "Technical view" switch shows the sensor graphs, class confidences and the fall rule's numbers. See `reports/Presenter_Guide.pdf` for the run sheet.

### Results

Evaluation holds out whole subjects (25% of the 36), so scores reflect performance on people the model has never seen. Windows from one person never appear in both train and test.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.391 | 0.094 |
| Logistic regression | 0.829 | 0.749 |
| **Random forest** | **0.849** | **0.775** |

Random forest 5-fold subject-wise CV macro-F1: 0.791 +/- 0.035. Plots are in `reports/` (`confusion_matrix.png`, `feature_importance.png`) and metrics in `reports/metrics.json`.

Jogging and Walking are classified well (F1 0.96 and 0.91). Downstairs is the weak class (recall 0.34) and is mostly confused with Upstairs and Walking, which is typical for waist/pocket accelerometer data. Sitting and Standing are separated mainly by the phone's orientation (gravity on each axis).

### Limitations

- **The fall flag is a heuristic, not a trained model.** WISDM has no fall recordings. The rule flags a spike of at least 2.5 g followed by about 0.5 s of stillness near 1 g. It was only checked on a synthetic signal. Before real use it needs validation on a labelled fall dataset such as SisFall or MobiAct.
- WISDM has no "lying down" class, so that activity cannot be recognised.
- Walking and Jogging windows often exceed 2.5 g on impact, so the stillness condition matters for avoiding false alarms.
- The model file (`models/activity_model.joblib`, about 22 MB) is not tracked in git. Run `python -m mlengine.train` to regenerate it.

## Group

- Moses Kibet - 183638
- Collins Jared - 184587
- Kathy Auma - 192231
