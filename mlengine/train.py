"""Train and evaluate activity classifiers with a subject-wise split.

Needs only numpy and scikit-learn (joblib comes with scikit-learn). Plots are made by reports/build_report.py.
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_raw
from .describe import main as describe_models
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
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    # Our own implementations (mlengine/models.py); the best of these is saved and deployed.
    models = {
        "baseline (majority)": MajorityClassifier(),
        "logistic regression": SoftmaxRegression(),
        "random forest": RandomForest(n_estimators=300, random_state=SEED, class_weight="balanced"),
    }
    # The same models from scikit-learn, trained only as a reference to check ours against.
    references = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                                class_weight="balanced"),
    }

    def evaluate(m):
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        return {"accuracy": accuracy_score(y[te], pred), "macro_f1": f1_score(y[te], pred, average="macro")}

    results = {}
    for name, m in models.items():
        results[name] = evaluate(m)
        print(f"{name:45s} acc={results[name]['accuracy']:.3f} macroF1={results[name]['macro_f1']:.3f}")
    for name, m in references.items():
        key = f"{name} (scikit-learn reference)"
        results[key] = evaluate(m)
        print(f"{key:45s} acc={results[key]['accuracy']:.3f} macroF1={results[key]['macro_f1']:.3f}")

    best_name = max(models, key=lambda k: results[k]["macro_f1"])
    best = models[best_name]
    report = classification_report(y[te], best.predict(X[te]))
    print(f"\nBest: {best_name}\n{report}")

    cv = []
    for cv_tr, cv_te in GroupKFold(n_splits=5).split(X, y, groups):
        rf = RandomForest(n_estimators=150, random_state=SEED, class_weight="balanced").fit(X[cv_tr], y[cv_tr])
        cv.append(f1_score(y[cv_te], rf.predict(X[cv_te]), average="macro"))
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
