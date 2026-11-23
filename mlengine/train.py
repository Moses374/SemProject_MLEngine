"""Train and evaluate activity classifiers with a subject-wise split.

Needs only numpy and joblib: the models are in models.py and the splitting and scoring in evaluation.py.
Plots are made by reports/build_report.py.
"""
import json
import os

import joblib
import numpy as np

from .data import load_raw
from .describe import main as describe_models
from .evaluation import accuracy, classification_report, group_kfold, group_shuffle_split, macro_f1
from .features import build_dataset
from .models import MajorityClassifier, RandomForest, SoftmaxRegression

SEED = 42


def main(out_models="models", out_reports="reports"):
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)
    print("Loading and windowing...")
    X, y, groups, names = build_dataset(load_raw())
    print(f"{len(X)} windows, {X.shape[1]} features, {len(set(groups))} subjects")

    # Hold out whole subjects so we measure generalisation to unseen people.
    tr, te = group_shuffle_split(groups, test_size=0.25, seed=SEED)
    # Our own implementations (mlengine/models.py); the best of these is saved and deployed.
    models = {
        "baseline (majority)": MajorityClassifier(),
        "logistic regression": SoftmaxRegression(),
        "random forest": RandomForest(n_estimators=300, random_state=SEED, class_weight="balanced"),
    }

    results = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        results[name] = {"accuracy": accuracy(y[te], pred), "macro_f1": macro_f1(y[te], pred)}
        print(f"{name:22s} acc={results[name]['accuracy']:.3f} macroF1={results[name]['macro_f1']:.3f}")

    best_name = max(models, key=lambda k: results[k]["macro_f1"])
    best = models[best_name]
    report = classification_report(y[te], best.predict(X[te]))
    print(f"\nBest: {best_name}\n{report}")

    cv = []
    for cv_tr, cv_te in group_kfold(groups, n_splits=5):
        rf = RandomForest(n_estimators=150, random_state=SEED, class_weight="balanced").fit(X[cv_tr], y[cv_tr])
        cv.append(macro_f1(y[cv_te], rf.predict(X[cv_te])))
    cv = np.array(cv)
    results["random forest 5-fold subject-wise CV"] = {"macro_f1_mean": cv.mean(), "macro_f1_std": cv.std()}
    print(f"RF 5-fold subject-wise CV macro-F1: {cv.mean():.3f} +/- {cv.std():.3f}")

    top = {}
    if hasattr(best, "feature_importances_"):
        order = np.argsort(best.feature_importances_)[::-1][:15]
        top = {names[i]: float(best.feature_importances_[i]) for i in order}

    # Keep the model trained on the training subjects only, plus the held-out subject ids, so the
    # live demo can show windows from people this model has never seen.
    joblib.dump({"model": best, "features": names, "test_users": sorted(int(u) for u in set(groups[te]))},
                os.path.join(out_models, "demo_model.joblib"), compress=3)

    # Refit the winner on all subjects for deployment.
    best.fit(X, y)
    joblib.dump({"model": best, "features": names}, os.path.join(out_models, "activity_model.joblib"), compress=3)
    with open(os.path.join(out_reports, "metrics.json"), "w") as f:
        json.dump({"best": best_name, "results": results, "report": report, "top_features": top},
                  f, indent=2, default=float)
    print("Saved models and metrics.\n")
    describe_models()


if __name__ == "__main__":
    main()
