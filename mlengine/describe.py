"""Print a human-readable summary of a saved model.

A .joblib file is binary, so it cannot be opened in a text editor. This reads it and shows what is inside.

    python -m mlengine.describe                          # both models in models/
    python -m mlengine.describe models/demo_model.joblib
"""
import os
import sys

import joblib
import numpy as np
from sklearn.tree import export_text

DEFAULT_MODELS = ["models/activity_model.joblib", "models/demo_model.joblib"]


def summarize(path: str) -> str:
    bundle = joblib.load(path)
    model, names = bundle["model"], bundle["features"]
    out = [f"MODEL FILE: {path}  ({os.path.getsize(path) / 1e6:.1f} MB)",
           f"Type: {type(model).__name__}",
           f"Activities it can output ({len(model.classes_)}): {', '.join(model.classes_)}",
           f"Inputs: {len(names)} features per 4-second window, in this order:",
           "  " + ", ".join(names)]
    if "test_users" in bundle:
        out.append(f"Trained WITHOUT these subjects (kept for honest testing): {bundle['test_users']}")
    else:
        out.append("Trained on all 36 subjects (the deployment model).")
    if hasattr(model, "estimators_"):
        trees = model.estimators_
        out += ["",
                f"Random forest of {len(trees)} decision trees",
                f"  average depth {np.mean([t.get_depth() for t in trees]):.1f} levels, "
                f"average {np.mean([t.get_n_leaves() for t in trees]):.0f} end points per tree",
                f"  {sum(t.tree_.node_count for t in trees):,} nodes in total (this is the 'knowledge' in the file)",
                "", "Most important features (share of the forest's decisions):"]
        order = np.argsort(model.feature_importances_)[::-1][:10]
        out += [f"  {names[i]:<12} {model.feature_importances_[i]:.3f}" for i in order]
        out += ["", "Top of the first tree (the real questions the model asks, 3 levels deep):",
                export_text(trees[0], feature_names=list(names), class_names=[str(c) for c in model.classes_], max_depth=3)]
    return "\n".join(out)


def main():
    paths = sys.argv[1:] or [p for p in DEFAULT_MODELS if os.path.exists(p)]
    if not paths:
        raise SystemExit("No model found. Run: python -m mlengine.train")
    text = ("\n\n" + "=" * 70 + "\n\n").join(summarize(p) for p in paths)
    print(text)
    if not sys.argv[1:]:
        with open("models/model_summary.txt", "w") as f:
            f.write(text + "\n")
        print("\n(saved as models/model_summary.txt, a normal text file you can open)")


if __name__ == "__main__":
    main()
