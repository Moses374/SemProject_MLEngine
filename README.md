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

## Group

- Moses Kibet - 183638
- Collins Jared - 184587
- Kathy Auma - 192231
