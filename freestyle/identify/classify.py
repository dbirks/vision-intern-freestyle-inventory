"""Classifiers, Leave-One-Group-Out evaluation, and unknown-threshold calibration."""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

import numpy as np

from freestyle.identify.embed import l2_normalize


def _as_2d_float(X: Any) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D feature matrix, got shape {arr.shape}")
    return arr


def _as_1d_labels(y: Sequence[Any] | np.ndarray, expected_len: int) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f"Expected 1D labels of length {expected_len}, got shape {arr.shape}")
    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _format_predictions(
    classes: np.ndarray,
    pred_indices: np.ndarray,
    scores: np.ndarray,
    unknown_thr: float | None,
    unknown_label: Any,
) -> np.ndarray:
    chosen = classes[pred_indices]
    if unknown_thr is None:
        return chosen
    reject = scores < float(unknown_thr)
    if not np.any(reject):
        return chosen
    if np.issubdtype(classes.dtype, np.str_) and isinstance(unknown_label, str):
        width = max(int(classes.dtype.itemsize // 4), len(unknown_label), 1)
        out = chosen.astype(f"<U{width}")
        out[reject] = unknown_label
        return out
    out_obj = chosen.astype(object)
    out_obj[reject] = unknown_label
    return out_obj


def balance_classes(
    X: Any,
    y: Sequence[Any] | np.ndarray,
    max_per_class: int | None = None,
    weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Cap each class's gallery rows and optionally return inverse-frequency weights.

    ``max_per_class`` keeps that many rows per label, spread evenly over the rows a
    gallery already has (deterministic, no RNG), so a class with 60 crops cannot
    outvote one with a single crop the way plain k-NN voting lets it.

    Returns ``(X, y, sample_weights)``; ``sample_weights`` is ``None`` unless
    ``weights`` is set, in which case each retained row weighs ``1/class_count``
    normalised to average one.
    """
    X_arr = _as_2d_float(X)
    y_arr = _as_1d_labels(y, X_arr.shape[0])
    if max_per_class is not None and int(max_per_class) <= 0:
        raise ValueError(f"max_per_class must be positive when set, got {max_per_class}")

    keep: list[int] = []
    for label in dict.fromkeys(y_arr.tolist()):
        rows = np.flatnonzero(y_arr == label)
        if max_per_class is not None and len(rows) > int(max_per_class):
            stride = len(rows) / int(max_per_class)
            rows = rows[[int(row * stride) for row in range(int(max_per_class))]]
        keep.extend(rows.tolist())
    keep.sort()
    kept = np.asarray(keep, dtype=np.int64)

    sample_weights: np.ndarray | None = None
    if weights:
        counts = {label: int(np.sum(y_arr == label)) for label in dict.fromkeys(y_arr.tolist())}
        per_row = np.array([1.0 / max(1, counts[label]) for label in y_arr[kept].tolist()])
        sample_weights = (per_row * (len(per_row) / max(1.0, per_row.sum()))).astype(np.float32)
    return X_arr[kept], y_arr[kept], sample_weights


class _BaseClassifier:
    classes_: np.ndarray

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError(f"{type(self).__name__} must be fitted before calling predict.")

    def predict_proba(self, X: Any) -> np.ndarray:
        raise NotImplementedError

    def predict_scores(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.max(probs, axis=1).astype(np.float32)

    def predict_with_scores(
        self,
        X: Any,
        unknown_thr: float | None = None,
        unknown_label: Any = "unknown",
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        probs = self.predict_proba(X)
        scores = self.predict_scores(X)
        pred_idx = np.argmax(probs, axis=1)
        preds = _format_predictions(self.classes_, pred_idx, scores, unknown_thr, unknown_label)
        return preds, scores

    def predict(
        self,
        X: Any,
        unknown_thr: float | None = None,
        unknown_label: Any = "unknown",
    ) -> np.ndarray:
        preds, _ = self.predict_with_scores(
            X, unknown_thr=unknown_thr, unknown_label=unknown_label
        )
        return preds


class PrototypeClassifier(_BaseClassifier):
    """Per-class L2-normalized mean prototype classifier using cosine similarity.

    One prototype per class makes the gallery size irrelevant to the vote: a class
    with 60 crops and a class with 1 crop each get exactly one candidate vector, so
    a gallery dominated by one product no longer swallows the rare products. Set
    ``max_per_class`` to additionally cap the rows averaged into each prototype.
    """

    def __init__(self, temperature: float = 0.1, max_per_class: int | None = None) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)
        self.max_per_class = max_per_class

    def fit(self, X: Any, y: Sequence[Any] | np.ndarray) -> PrototypeClassifier:
        X_arr, y_arr, _ = balance_classes(X, y, max_per_class=self.max_per_class)
        X_arr = l2_normalize(X_arr)
        classes = np.unique(y_arr)
        protos = np.stack(
            [X_arr[y_arr == cls].mean(axis=0) for cls in classes],
            axis=0,
        )
        self.classes_ = classes
        self.prototypes_ = l2_normalize(protos)
        self.n_per_class_ = {str(cls): int(np.sum(y_arr == cls)) for cls in classes}
        return self

    def similarity(self, X: Any) -> np.ndarray:
        """Cosine similarities ``(N, C)`` against each class prototype."""
        self._check_is_fitted()
        X_arr = l2_normalize(_as_2d_float(X))
        return (X_arr @ self.prototypes_.T).astype(np.float32)

    def predict_proba(self, X: Any) -> np.ndarray:
        sims = self.similarity(X)
        return _softmax(sims / self.temperature).astype(np.float32)

    def predict_scores(self, X: Any) -> np.ndarray:
        sims = self.similarity(X)
        return np.max(sims, axis=1).astype(np.float32)

    def predict_with_scores(
        self,
        X: Any,
        unknown_thr: float | None = None,
        unknown_label: Any = "unknown",
    ) -> tuple[np.ndarray, np.ndarray]:
        sims = self.similarity(X)
        pred_idx = np.argmax(sims, axis=1)
        scores = sims[np.arange(len(sims)), pred_idx].astype(np.float32)
        preds = _format_predictions(self.classes_, pred_idx, scores, unknown_thr, unknown_label)
        return preds, scores


class KNNClassifier(_BaseClassifier):
    """Cosine k-Nearest-Neighbor classifier over gallery embeddings.

    ``max_per_class`` caps how many gallery rows each label contributes (an evenly
    spread deterministic subsample), which stops a 60-crop class from filling every
    neighbour slot; ``balance`` weights each neighbour by ``1/class_count`` so rare
    classes are not outvoted in the top-k.
    """

    def __init__(
        self,
        k: int = 5,
        temperature: float | None = None,
        max_per_class: int | None = None,
        balance: bool = False,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive when set, got {temperature}")
        self.k = int(k)
        self.temperature = float(temperature) if temperature is not None else None
        self.max_per_class = max_per_class
        self.balance = bool(balance)

    def fit(self, X: Any, y: Sequence[Any] | np.ndarray) -> KNNClassifier:
        X_bal, y_bal, _ = balance_classes(X, y, max_per_class=self.max_per_class)
        X_arr = l2_normalize(X_bal)
        y_arr = _as_1d_labels(y_bal, X_arr.shape[0])
        classes, y_idx = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes
        self.X_train_ = X_arr
        self.y_train_idx_ = y_idx.astype(np.int64)
        if self.balance:
            counts = np.bincount(y_idx, minlength=len(classes)).astype(np.float32)
            weights = 1.0 / np.maximum(1.0, counts)
            self.class_weights_ = (weights * float(len(classes)) / float(weights.sum())).astype(np.float32)
        else:
            self.class_weights_ = None
        return self

    def _top_k(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        X_arr = l2_normalize(_as_2d_float(X))
        sims = X_arr @ self.X_train_.T
        k_eff = min(self.k, self.X_train_.shape[0])
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k_eff]
        top_sims = np.take_along_axis(sims, order, axis=1)
        top_cls = self.y_train_idx_[order]
        return top_sims.astype(np.float32), top_cls

    def _probs_and_scores_from_topk(
        self, top_sims: np.ndarray, top_cls: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n, k_eff = top_sims.shape
        n_classes = len(self.classes_)
        probs = np.zeros((n, n_classes), dtype=np.float32)

        if self.temperature is not None:
            weights = np.exp((top_sims - np.max(top_sims, axis=1, keepdims=True)) / self.temperature)
        else:
            weights = np.maximum(top_sims, 0.0)
        if getattr(self, "class_weights_", None) is not None:
            weights = weights * self.class_weights_[top_cls]

        for i in range(n):
            row_w = weights[i]
            if row_w.sum() <= 0.0:
                row_w = np.ones_like(row_w)
            np.add.at(probs[i], top_cls[i], row_w)
            probs[i] /= probs[i].sum()

        pred_idx = np.argmax(probs, axis=1)
        # Confidence combines positive cosine similarity magnitude and consensus among top-k.
        pos_sims = np.maximum(top_sims, 0.0)
        match_mask = top_cls == pred_idx[:, None]
        scores = (np.sum(pos_sims * match_mask, axis=1) / float(k_eff)).astype(np.float32)
        return probs, scores

    def predict_proba(self, X: Any) -> np.ndarray:
        top_sims, top_cls = self._top_k(X)
        probs, _ = self._probs_and_scores_from_topk(top_sims, top_cls)
        return probs

    def predict_scores(self, X: Any) -> np.ndarray:
        top_sims, top_cls = self._top_k(X)
        _, scores = self._probs_and_scores_from_topk(top_sims, top_cls)
        return scores

    def predict_with_scores(
        self,
        X: Any,
        unknown_thr: float | None = None,
        unknown_label: Any = "unknown",
    ) -> tuple[np.ndarray, np.ndarray]:
        top_sims, top_cls = self._top_k(X)
        probs, scores = self._probs_and_scores_from_topk(top_sims, top_cls)
        pred_idx = np.argmax(probs, axis=1)
        preds = _format_predictions(self.classes_, pred_idx, scores, unknown_thr, unknown_label)
        return preds, scores


class _NumpySoftmaxRegression:
    """Pure-NumPy L2-regularized multinomial logistic regression fallback."""

    def __init__(self, C: float = 1.0, max_iter: int = 500, tol: float = 1e-6) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y_idx: np.ndarray, n_classes: int) -> _NumpySoftmaxRegression:
        n, d = X.shape
        X64 = X.astype(np.float64, copy=False)
        W = np.zeros((d, n_classes), dtype=np.float64)
        b = np.zeros((n_classes,), dtype=np.float64)
        one_hot = np.eye(n_classes, dtype=np.float64)[y_idx]
        reg = 1.0 / (self.C * float(n))

        def loss_and_grad(w_mat: np.ndarray, b_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            logits = X64 @ w_mat + b_vec[None, :]
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            ce = -np.mean(np.sum(one_hot * np.log(np.maximum(probs, 1e-15)), axis=1))
            loss = float(ce + 0.5 * reg * np.sum(w_mat * w_mat))
            diff = (probs - one_hot) / float(n)
            grad_w = X64.T @ diff + reg * w_mat
            grad_b = np.sum(diff, axis=0)
            return loss, grad_w, grad_b

        step = 1.0
        for _ in range(self.max_iter):
            loss, gw, gb = loss_and_grad(W, b)
            grad_norm_sq = float(np.sum(gw * gw) + np.sum(gb * gb))
            if grad_norm_sq <= self.tol * self.tol:
                break
            cur_step = step
            accepted = False
            for _ in range(25):
                W_try = W - cur_step * gw
                b_try = b - cur_step * gb
                loss_try, _, _ = loss_and_grad(W_try, b_try)
                if loss_try <= loss - 1e-4 * cur_step * grad_norm_sq:
                    W, b = W_try, b_try
                    step = min(cur_step * 1.5, 10.0)
                    accepted = True
                    break
                cur_step *= 0.5
            if not accepted:
                break

        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        logits = X.astype(np.float32, copy=False) @ self.W + self.b[None, :]
        return _softmax(logits).astype(np.float32)


class LinearProbe(_BaseClassifier):
    """Multinomial logistic regression linear probe (sklearn with NumPy fallback)."""

    def __init__(
        self,
        C: float = 10.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        normalize: bool = True,
        backend: Literal["auto", "sklearn", "numpy"] = "auto",
        random_state: int = 42,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if backend not in ("auto", "sklearn", "numpy"):
            raise ValueError(f"backend must be 'auto', 'sklearn', or 'numpy', got {backend!r}")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.normalize = bool(normalize)
        self.backend = backend
        self.random_state = int(random_state)
        self._impl: Any = None
        self._single_class_mode = False
        self.backend_used_: str = ""

    def fit(self, X: Any, y: Sequence[Any] | np.ndarray) -> LinearProbe:
        X_arr = _as_2d_float(X)
        if self.normalize:
            X_arr = l2_normalize(X_arr)
        y_arr = _as_1d_labels(y, X_arr.shape[0])
        classes, y_idx = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes

        if len(classes) == 1:
            self._single_class_mode = True
            self._impl = None
            self.backend_used_ = "trivial"
            return self
        self._single_class_mode = False

        use_sklearn = False
        LogisticRegression = None
        if self.backend in ("auto", "sklearn"):
            try:
                from sklearn.linear_model import LogisticRegression as SkLogisticRegression

                LogisticRegression = SkLogisticRegression
                use_sklearn = True
            except ImportError:
                if self.backend == "sklearn":
                    raise
                use_sklearn = False

        if use_sklearn and LogisticRegression is not None:
            clf = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            clf.fit(X_arr, y_idx)
            self._impl = clf
            self.backend_used_ = "sklearn"
        else:
            clf_np = _NumpySoftmaxRegression(C=self.C, max_iter=self.max_iter, tol=self.tol)
            clf_np.fit(X_arr, y_idx, n_classes=len(classes))
            self._impl = clf_np
            self.backend_used_ = "numpy"
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _as_2d_float(X)
        if self.normalize:
            X_arr = l2_normalize(X_arr)
        if self._single_class_mode:
            return np.ones((X_arr.shape[0], 1), dtype=np.float32)
        return np.asarray(self._impl.predict_proba(X_arr), dtype=np.float32)


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_label: Any = "unknown",
) -> dict[str, Any]:
    classes = list(np.unique(y_true).tolist())
    pred_unique = list(np.unique(y_pred).tolist())
    pred_labels = list(classes)
    for lbl in pred_unique:
        if lbl not in pred_labels:
            pred_labels.append(lbl)

    row_index = {c: i for i, c in enumerate(classes)}
    col_index = {c: j for j, c in enumerate(pred_labels)}
    cm = [[0 for _ in pred_labels] for _ in classes]

    for yt, yp in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        cm[row_index[yt]][col_index[yp]] += 1

    per_class_recall: dict[Any, float] = {}
    f1_scores: list[float] = []
    for c in classes:
        r = row_index[c]
        c_col = col_index[c]
        tp = cm[r][c_col]
        fn = sum(cm[r]) - tp
        fp = sum(cm[row][c_col] for row in range(len(classes))) - tp
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        per_class_recall[c] = recall
        f1_scores.append(f1)

    n_total = len(y_true)
    correct = np.asarray([yt == yp for yt, yp in zip(y_true.tolist(), y_pred.tolist(), strict=True)])
    covered = np.asarray([yp != unknown_label for yp in y_pred.tolist()])
    n_covered = int(np.sum(covered))

    accuracy = float(np.mean(correct)) if n_total > 0 else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    coverage = float(n_covered / n_total) if n_total > 0 else 0.0
    covered_accuracy = float(np.mean(correct[covered])) if n_covered > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm,
        "classes": classes,
        "pred_labels": pred_labels,
        "coverage": coverage,
        "covered_accuracy": covered_accuracy,
        "n_covered": n_covered,
        "n_samples": int(n_total),
    }


def leave_one_group_out_eval(
    X: Any,
    y: Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    clf_factory: Callable[[], Any],
    *,
    unknown_thr: float | None = None,
    unknown_label: Any = "unknown",
) -> dict[str, Any]:
    """Run Leave-One-Group-Out cross-validation without group leakage.

    Every group is held out once as the query set while all other groups form the
    gallery/training set. Returns pooled accuracy, macro-F1, per-class recall,
    confusion matrix, and per-group accuracy.
    """
    X_arr = _as_2d_float(X)
    n = X_arr.shape[0]
    y_arr = _as_1d_labels(y, n)
    groups_arr = np.asarray(groups)
    if groups_arr.ndim != 1 or groups_arr.shape[0] != n:
        raise ValueError(f"Expected 1D groups of length {n}, got shape {groups_arr.shape}")

    unique_groups = list(dict.fromkeys(groups_arr.tolist()))
    if len(unique_groups) < 2:
        raise ValueError(
            f"Leave-One-Group-Out requires at least 2 distinct groups, got {len(unique_groups)}"
        )

    y_pred_all = np.empty((n,), dtype=object)
    scores_all = np.empty((n,), dtype=np.float32)
    per_group_accuracy: dict[Any, float] = {}

    for grp in unique_groups:
        val_mask = groups_arr == grp
        train_mask = ~val_mask
        if not set(groups_arr[train_mask].tolist()).isdisjoint(set(groups_arr[val_mask].tolist())):
            raise AssertionError(f"Group leakage detected for group {grp!r}")

        clf = clf_factory()
        clf.fit(X_arr[train_mask], y_arr[train_mask])
        if hasattr(clf, "predict_with_scores"):
            preds_g, scores_g = clf.predict_with_scores(
                X_arr[val_mask], unknown_thr=unknown_thr, unknown_label=unknown_label
            )
        else:
            preds_g = clf.predict(X_arr[val_mask], unknown_thr=unknown_thr)
            scores_g = (
                clf.predict_scores(X_arr[val_mask])
                if hasattr(clf, "predict_scores")
                else np.max(clf.predict_proba(X_arr[val_mask]), axis=1)
            )
        y_pred_all[val_mask] = np.asarray(preds_g, dtype=object)
        scores_all[val_mask] = np.asarray(scores_g, dtype=np.float32)
        per_group_accuracy[grp] = float(
            np.mean([yt == yp for yt, yp in zip(y_arr[val_mask].tolist(), preds_g.tolist(), strict=True)])
        )

    if unknown_thr is None and (
        np.issubdtype(y_arr.dtype, np.str_) or np.issubdtype(y_arr.dtype, np.number)
    ):
        try:
            y_pred_typed = y_pred_all.astype(y_arr.dtype)
        except (TypeError, ValueError):
            y_pred_typed = y_pred_all
    else:
        y_pred_typed = y_pred_all

    metrics = _compute_classification_metrics(y_arr, y_pred_typed, unknown_label=unknown_label)
    metrics["n_groups"] = len(unique_groups)
    metrics["groups"] = unique_groups
    metrics["per_group_accuracy"] = per_group_accuracy
    metrics["unknown_thr"] = unknown_thr
    metrics["y_true"] = y_arr.tolist()
    metrics["y_pred"] = y_pred_typed.tolist()
    metrics["scores"] = scores_all.tolist()
    return metrics


def select_unknown_threshold(
    scores: Sequence[float] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    y_true: Sequence[Any] | np.ndarray,
    *,
    target_coverage: float = 0.8,
    unknown_label: Any = "unknown",
    metric: Literal["auto", "covered_accuracy", "overall_accuracy"] = "auto",
) -> dict[str, Any]:
    """Select an unknown rejection threshold maximizing accuracy with coverage >= target."""
    if not 0.0 < float(target_coverage) <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    if metric not in ("auto", "covered_accuracy", "overall_accuracy"):
        raise ValueError(f"Unsupported metric mode: {metric!r}")

    scores_arr = np.asarray(scores, dtype=np.float64)
    n = scores_arr.shape[0]
    if scores_arr.ndim != 1 or n == 0:
        raise ValueError(f"scores must be a non-empty 1D array, got shape {scores_arr.shape}")
    y_pred_arr = _as_1d_labels(y_pred, n)
    y_true_arr = _as_1d_labels(y_true, n)

    has_gt_unknown = any(yt == unknown_label for yt in y_true_arr.tolist())
    opt_metric = (
        ("overall_accuracy" if has_gt_unknown else "covered_accuracy")
        if metric == "auto"
        else metric
    )

    correct_if_accepted = np.asarray(
        [yp == yt for yp, yt in zip(y_pred_arr.tolist(), y_true_arr.tolist(), strict=True)],
        dtype=bool,
    )
    gt_is_unknown = np.asarray(
        [yt == unknown_label for yt in y_true_arr.tolist()],
        dtype=bool,
    )

    unique_scores = np.unique(scores_arr)
    best_key: tuple[float, float, float] | None = None
    best_record: dict[str, Any] | None = None

    for thr_candidate in unique_scores:
        accepted = scores_arr >= thr_candidate
        n_cov = int(np.sum(accepted))
        coverage = float(n_cov / n)
        if coverage + 1e-12 < float(target_coverage):
            continue

        covered_acc = float(np.mean(correct_if_accepted[accepted])) if n_cov > 0 else 0.0
        n_correct_overall = int(np.sum(correct_if_accepted & accepted)) + int(
            np.sum(gt_is_unknown & ~accepted)
        )
        overall_acc = float(n_correct_overall / n)

        primary = overall_acc if opt_metric == "overall_accuracy" else covered_acc
        secondary = covered_acc if opt_metric == "overall_accuracy" else overall_acc
        key = (primary, secondary, coverage)

        if best_key is None or key > best_key:
            rejected_scores = scores_arr[~accepted]
            accepted_min = float(np.min(scores_arr[accepted]))
            if rejected_scores.size > 0:
                rejected_max = float(np.max(rejected_scores))
                calibrated_thr = 0.5 * (accepted_min + rejected_max)
            else:
                calibrated_thr = accepted_min - 1e-6

            best_key = key
            best_record = {
                "threshold": float(calibrated_thr),
                "coverage": coverage,
                "accuracy": float(primary),
                "covered_accuracy": covered_acc,
                "overall_accuracy": overall_acc,
                "n_covered": n_cov,
                "n_total": int(n),
                "target_coverage": float(target_coverage),
                "target_met": bool(coverage + 1e-12 >= float(target_coverage)),
                "metric": opt_metric,
            }

    assert best_record is not None
    return best_record


def calibrate_unknown_threshold(
    clf: Any,
    X_val: Any,
    y_val: Sequence[Any] | np.ndarray,
    *,
    target_coverage: float = 0.8,
    unknown_label: Any = "unknown",
    metric: Literal["auto", "covered_accuracy", "overall_accuracy"] = "auto",
) -> dict[str, Any]:
    """Run ``clf`` on ``X_val`` and select the optimal unknown rejection threshold."""
    if hasattr(clf, "predict_with_scores"):
        y_pred, scores = clf.predict_with_scores(X_val, unknown_thr=None)
    else:
        y_pred = clf.predict(X_val)
        scores = (
            clf.predict_scores(X_val)
            if hasattr(clf, "predict_scores")
            else np.max(clf.predict_proba(X_val), axis=1)
        )
    return select_unknown_threshold(
        scores=scores,
        y_pred=y_pred,
        y_true=y_val,
        target_coverage=target_coverage,
        unknown_label=unknown_label,
        metric=metric,
    )
