"""Train/test splitting and scoring, written with NumPy only.

These reproduce the choices scikit-learn's GroupShuffleSplit and GroupKFold make (same people in each fold for the
same seed) and its accuracy, macro F1, classification report and confusion matrix, so earlier results stay comparable.
"""
import numpy as np


def group_shuffle_split(groups, test_size=0.25, seed=42):
    """One random split that keeps every person (group) entirely on one side. Returns (train_idx, test_idx)."""
    people, member = np.unique(groups, return_inverse=True)
    n_test = int(np.ceil(test_size * len(people)))
    order = np.random.RandomState(seed).permutation(len(people))
    test_people, train_people = order[:n_test], order[n_test:]
    return np.flatnonzero(np.isin(member, train_people)), np.flatnonzero(np.isin(member, test_people))


def group_kfold(groups, n_splits=5):
    """Yield (train_idx, test_idx) for each fold; no person appears in two folds and folds are kept similar in size.

    Deterministic: the biggest people are placed first, each into the fold that currently has the fewest windows.
    """
    people, member = np.unique(groups, return_inverse=True)
    if n_splits > len(people):
        raise ValueError("more folds than people")
    size = np.bincount(member)
    order = np.argsort(size, kind="stable")[::-1]
    fold_size = np.zeros(n_splits)
    person_fold = np.zeros(len(people), dtype=int)
    for p in order:
        f = int(np.argmin(fold_size))
        fold_size[f] += size[p]
        person_fold[p] = f
    fold_of_row = person_fold[member]
    for f in range(n_splits):
        yield np.flatnonzero(fold_of_row != f), np.flatnonzero(fold_of_row == f)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, labels=None):
    """Rows are the actual activity, columns the predicted one. Returns (matrix, labels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    pos = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(m, ([pos[c] for c in y_true], [pos[c] for c in y_pred]), 1)
    return m, labels


def _per_class(y_true, y_pred):
    m, labels = confusion_matrix(y_true, y_pred)
    tp = np.diag(m).astype(float)
    predicted, actual = m.sum(axis=0), m.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return labels, precision, recall, f1, actual


def macro_f1(y_true, y_pred):
    """F1 averaged over the activities with equal weight, so small ones count as much as big ones."""
    return float(_per_class(y_true, y_pred)[3].mean())


def classification_report(y_true, y_pred):
    """Precision, recall, F1 and count for each activity, as a text table."""
    labels, p, r, f1, n = _per_class(y_true, y_pred)
    names = [str(c) for c in labels]
    width = max(max(len(s) for s in names), len("weighted avg"), 2)
    row = "{:>{w}s} " + " {:>9.2f}" * 3 + " {:>9}\n"
    out = "{:>{w}s} ".format("", w=width) + " {:>9}" * 4
    out = out.format("precision", "recall", "f1-score", "support") + "\n\n"
    for i, s in enumerate(names):
        out += row.format(s, p[i], r[i], f1[i], int(n[i]), w=width)
    total = int(n.sum())
    out += "\n"
    out += "{:>{w}s} ".format("accuracy", w=width) + " {:>9}".format("") * 2
    out += " {:>9.2f} {:>9}\n".format(accuracy(y_true, y_pred), total)
    out += row.format("macro avg", p.mean(), r.mean(), f1.mean(), total, w=width)
    out += row.format("weighted avg", np.average(p, weights=n), np.average(r, weights=n),
                      np.average(f1, weights=n), total, w=width)
    return out
