"""Classifiers implemented from scratch with NumPy.

Each model has the same small interface used by the rest of the engine:
``fit(X, y)``, ``predict(X)``, ``predict_proba(X)`` and a ``classes_`` array whose order matches the
columns of ``predict_proba``. ``X`` may be a NumPy array or a pandas DataFrame.
"""
import numpy as np
from joblib import Parallel, delayed


def _as_array(X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)


def _encode(y):
    """Sorted class labels and each sample's class index into them."""
    classes, idx = np.unique(np.asarray(y), return_inverse=True)
    return classes, idx


def _balanced_class_weights(idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Weight each class by n / (k * count), so every class carries the same total weight."""
    counts = np.bincount(idx, minlength=n_classes)
    return len(idx) / (n_classes * counts)


class MajorityClassifier:
    """Baseline: always predicts the most frequent training class."""

    def fit(self, X, y):
        self.classes_, idx = _encode(y)
        self._proba = np.bincount(idx, minlength=len(self.classes_)) / len(idx)
        return self

    def predict_proba(self, X):
        return np.tile(self._proba, (len(X), 1))

    def predict(self, X):
        return np.full(len(X), self.classes_[self._proba.argmax()])


class SoftmaxRegression:
    """Multinomial logistic regression trained by full-batch gradient descent (Adam).

    Features are standardised internally. The loss is the mean cross-entropy plus an L2 penalty
    ``||W||^2 / (2 * C * n)``, which is the same objective scikit-learn's LogisticRegression uses.
    """

    def __init__(self, C=1.0, lr=0.05, max_iter=3000, tol=1e-6):
        self.C, self.lr, self.max_iter, self.tol = C, lr, max_iter, tol

    def _scores(self, Z):
        return Z @ self.coef_ + self.intercept_

    @staticmethod
    def _softmax(S):
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = _as_array(X)
        self.classes_, idx = _encode(y)
        n, d = X.shape
        k = len(self.classes_)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        Y = np.eye(k)[idx]
        lam = 1.0 / (self.C * n)

        self.coef_ = np.zeros((d, k))
        self.intercept_ = np.zeros(k)
        params = [self.coef_, self.intercept_]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        prev_loss = np.inf
        for t in range(1, self.max_iter + 1):
            P = self._softmax(self._scores(Z))
            loss = (-np.log(P[np.arange(n), idx] + 1e-12).mean()
                    + 0.5 * lam * (self.coef_ ** 2).sum())
            if prev_loss - loss < self.tol and t > 1:
                break
            prev_loss = loss
            G = (P - Y) / n
            grads = [Z.T @ G + lam * self.coef_, G.sum(axis=0)]
            for p, g, mi, vi in zip(params, grads, m, v):
                mi *= b1
                mi += (1 - b1) * g
                vi *= b2
                vi += (1 - b2) * g * g
                p -= self.lr * (mi / (1 - b1 ** t)) / (np.sqrt(vi / (1 - b2 ** t)) + eps)
        self.n_iter_ = t
        return self

    def predict_proba(self, X):
        return self._softmax(self._scores((_as_array(X) - self.mean_) / self.scale_))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DecisionTree:
    """CART classification tree using Gini impurity and per-sample weights.

    The fitted tree is stored as parallel arrays indexed by node id. A node is a leaf exactly when
    ``left[node] == -1``; internal nodes send a sample left when ``x[feature] <= threshold``.
    """

    def __init__(self, max_features=None, max_depth=None, min_samples_leaf=1, rng=None):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit_encoded(self, X: np.ndarray, idx: np.ndarray, weights: np.ndarray, n_classes: int):
        """Fit on numeric class indices; ``weights`` must be positive for every row given."""
        n_features = X.shape[1]
        mtry = n_features if self.max_features is None else self.max_features
        max_depth = np.inf if self.max_depth is None else self.max_depth
        msl = self.min_samples_leaf
        W_all = np.zeros((len(idx), n_classes))
        W_all[np.arange(len(idx)), idx] = weights

        feature, threshold, left, right, value = [], [], [], [], []
        importance = np.zeros(n_features)

        def new_node(rows):
            totals = W_all[rows].sum(axis=0)
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(totals / totals.sum())
            return len(value) - 1, totals

        root, root_totals = new_node(np.arange(len(idx)))
        stack = [(root, np.arange(len(idx)), root_totals, 0)]
        self.depth_ = 0
        while stack:
            node, rows, totals, depth = stack.pop()
            self.depth_ = max(self.depth_, depth)
            n = len(rows)
            wt = totals.sum()
            parent_impurity = wt - (totals ** 2).sum() / wt  # weighted Gini: wt * gini
            if depth >= max_depth or n < 2 * msl or parent_impurity <= 1e-12:
                continue

            best = None  # (child_impurity, feature, threshold, sorted rows, split position)
            for f in self.rng.choice(n_features, size=mtry, replace=False):
                xs = X[rows, f]
                order = np.argsort(xs, kind="stable")
                xs = xs[order]
                cl = np.cumsum(W_all[rows[order]], axis=0)[:-1]
                cr = totals - cl
                wl = cl.sum(axis=1)
                wr = cr.sum(axis=1)
                child = (wl - (cl ** 2).sum(axis=1) / wl) + (wr - (cr ** 2).sum(axis=1) / wr)
                pos = np.arange(1, n)  # left side holds `pos` samples
                valid = (xs[:-1] < xs[1:]) & (pos >= msl) & (n - pos >= msl)
                if not valid.any():
                    continue
                child = np.where(valid, child, np.inf)
                i = int(child.argmin())
                if best is None or child[i] < best[0]:
                    best = (child[i], f, (xs[i] + xs[i + 1]) / 2, rows[order], i + 1)

            if best is None or parent_impurity - best[0] <= 1e-12:
                continue
            child_impurity, f, thr, sorted_rows, split = best
            importance[f] += parent_impurity - child_impurity
            lrows, rrows = sorted_rows[:split], sorted_rows[split:]
            lnode, ltot = new_node(lrows)
            rnode, rtot = new_node(rrows)
            feature[node], threshold[node], left[node], right[node] = f, thr, lnode, rnode
            stack.append((lnode, lrows, ltot, depth + 1))
            stack.append((rnode, rrows, rtot, depth + 1))

        self.feature = np.array(feature)
        self.threshold = np.array(threshold)
        self.left = np.array(left)
        self.right = np.array(right)
        self.value = np.array(value)
        total = importance.sum()
        self.importance = importance / total if total > 0 else importance
        return self

    @property
    def node_count(self) -> int:
        return len(self.left)

    @property
    def n_leaves(self) -> int:
        return int((self.left == -1).sum())

    def to_text(self, feature_names, class_names, max_depth=3) -> str:
        """The tree's top levels as indented if/else rules, one line per branch or leaf."""
        lines = []

        def walk(node, depth):
            pad = "|   " * depth
            if self.left[node] == -1:
                lines.append(f"{pad}|--- class: {class_names[int(self.value[node].argmax())]}")
                return
            if depth >= max_depth:
                lines.append(f"{pad}|--- (deeper levels truncated)")
                return
            name, thr = feature_names[self.feature[node]], self.threshold[node]
            lines.append(f"{pad}|--- {name} <= {thr:.2f}")
            walk(self.left[node], depth + 1)
            lines.append(f"{pad}|--- {name} >  {thr:.2f}")
            walk(self.right[node], depth + 1)

        walk(0, 0)
        return "\n".join(lines)

    def predict_proba_encoded(self, X: np.ndarray) -> np.ndarray:
        node = np.zeros(len(X), dtype=np.int64)
        active = self.left[node] != -1
        while active.any():
            a = np.flatnonzero(active)
            na = node[a]
            go_left = X[a, self.feature[na]] <= self.threshold[na]
            node[a] = np.where(go_left, self.left[na], self.right[na])
            active[a] = self.left[node[a]] != -1
        return self.value[node]


def _fit_tree(X, idx, class_w, n_classes, max_features, max_depth, min_samples_leaf, seed):
    rng = np.random.default_rng(seed)
    # Bootstrap: draw n rows with replacement; a row drawn c times gets weight c.
    counts = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
    rows = np.flatnonzero(counts)
    tree = DecisionTree(max_features, max_depth, min_samples_leaf, rng)
    return tree.fit_encoded(X[rows], idx[rows], counts[rows] * class_w[idx[rows]], n_classes)


class RandomForest:
    """Bagged CART trees with a random feature subset tried at every split.

    ``class_weight="balanced"`` reweights classes by their frequency in the full training set.
    Probabilities are the average of the trees' leaf class distributions.
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None, min_samples_leaf=1,
                 class_weight=None, random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = _as_array(X)
        self.classes_, idx = _encode(y)
        k = len(self.classes_)
        n_features = X.shape[1]
        mtry = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else self.max_features
        class_w = (_balanced_class_weights(idx, k) if self.class_weight == "balanced"
                   else np.ones(k))
        seeds = np.random.SeedSequence(self.random_state).spawn(self.n_estimators)
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_tree)(X, idx, class_w, k, mtry, self.max_depth, self.min_samples_leaf, s)
            for s in seeds)
        self.feature_importances_ = np.mean([t.importance for t in self.trees_], axis=0)
        return self

    def predict_proba(self, X):
        X = _as_array(X)
        return np.mean([t.predict_proba_encoded(X) for t in self.trees_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
