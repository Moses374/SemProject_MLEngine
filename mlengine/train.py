"""Train and evaluate activity classifiers with a subject-wise split."""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_raw
from .features import build_dataset

SEED = 42


def main(out_models="models", out_reports="reports"):
    os.makedirs(out_models, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)
    print("Loading and windowing...")
    feats = build_dataset(load_raw())
    X = feats.drop(columns=["user", "activity"])
    y = feats["activity"]
    groups = feats["user"]
    print(f"{len(feats)} windows, {X.shape[1]} features, {groups.nunique()} subjects")

    # Hold out whole subjects so we measure generalisation to unseen people.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr, te = next(gss.split(X, y, groups))
    models = {
        "baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED,
                                                class_weight="balanced"),
    }
    results = {}
    for name, m in models.items():
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        results[name] = {"accuracy": accuracy_score(y.iloc[te], pred),
                         "macro_f1": f1_score(y.iloc[te], pred, average="macro")}
        print(f"{name:22s} acc={results[name]['accuracy']:.3f} macroF1={results[name]['macro_f1']:.3f}")

    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = models[best_name]
    pred = best.predict(X.iloc[te])
    report = classification_report(y.iloc[te], pred)
    print(f"\nBest: {best_name}\n{report}")

    cv = cross_val_score(
        RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=SEED, class_weight="balanced"),
        X, y, groups=groups, cv=GroupKFold(n_splits=5), scoring="f1_macro")
    results["random forest 5-fold subject-wise CV"] = {"macro_f1_mean": cv.mean(), "macro_f1_std": cv.std()}
    print(f"RF 5-fold subject-wise CV macro-F1: {cv.mean():.3f} +/- {cv.std():.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y.iloc[te], pred, ax=ax, xticks_rotation=45, cmap="Blues")
    ax.set_title(f"{best_name}: held-out subjects")
    fig.tight_layout()
    fig.savefig(os.path.join(out_reports, "confusion_matrix.png"), dpi=130)

    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        imp[::-1].plot.barh(ax=ax)
        ax.set_title("Top 15 feature importances")
        fig.tight_layout()
        fig.savefig(os.path.join(out_reports, "feature_importance.png"), dpi=130)

    # Keep the model trained on the training subjects only, plus the held-out subject ids, so the
    # live demo can show windows from people this model has never seen.
    joblib.dump({"model": best, "features": list(X.columns),
                 "test_users": sorted(int(u) for u in groups.iloc[te].unique())},
                os.path.join(out_models, "demo_model.joblib"), compress=3)

    # Refit the winner on all subjects for deployment.
    best.fit(X, y)
    joblib.dump({"model": best, "features": list(X.columns)},
                os.path.join(out_models, "activity_model.joblib"), compress=3)
    with open(os.path.join(out_reports, "metrics.json"), "w") as f:
        json.dump({"best": best_name, "results": results, "report": report}, f, indent=2, default=float)
    print("Saved model and reports.")


if __name__ == "__main__":
    main()
