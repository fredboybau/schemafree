from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


def _as_int(values: Sequence[int]) -> IntArray:
    return np.asarray(values, dtype=np.int_)


def _as_float(values: Sequence[float]) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = _as_int(y_true)
    pred = _as_int(y_pred)
    classes = np.unique(np.concatenate([truth, pred]))
    scores = []
    for cls in classes:
        tp = float(np.sum((pred == cls) & (truth == cls)))
        fp = float(np.sum((pred == cls) & (truth != cls)))
        fn = float(np.sum((pred != cls) & (truth == cls)))
        denom = 2.0 * tp + fp + fn
        scores.append(0.0 if denom == 0.0 else (2.0 * tp) / denom)
    return float(np.mean(scores))


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    truth = _as_int(y_true)
    pred = _as_int(y_pred)
    classes = np.unique(truth)
    recalls = []
    for cls in classes:
        support = float(np.sum(truth == cls))
        if support == 0.0:
            continue
        hits = float(np.sum((truth == cls) & (pred == cls)))
        recalls.append(hits / support)
    return float(np.mean(recalls)) if recalls else 0.0


def sensitivity(y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> float:
    truth = _as_int(y_true)
    pred = _as_int(y_pred)
    pos = float(np.sum(truth == positive))
    if pos == 0.0:
        return 0.0
    return float(np.sum((truth == positive) & (pred == positive))) / pos


def specificity(y_true: Sequence[int], y_pred: Sequence[int], positive: int = 1) -> float:
    truth = _as_int(y_true)
    pred = _as_int(y_pred)
    neg = float(np.sum(truth != positive))
    if neg == 0.0:
        return 0.0
    return float(np.sum((truth != positive) & (pred != positive))) / neg


def auc(y_true: Sequence[int], scores: Sequence[float], positive: int = 1) -> float:
    truth = _as_int(y_true)
    values = _as_float(scores)
    pos = values[truth == positive]
    neg = values[truth != positive]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    _resolve_ties(values, ranks)
    rank_sum_pos = float(np.sum(ranks[truth == positive]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _resolve_ties(values: FloatArray, ranks: FloatArray) -> None:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    start = 0
    while start < sorted_vals.size:
        stop = start + 1
        while stop < sorted_vals.size and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        if stop - start > 1:
            avg = float(np.mean(ranks[order[start:stop]]))
            for idx in order[start:stop]:
                ranks[idx] = avg
        start = stop
