from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]


def _placements(pos: FloatArray, neg: FloatArray) -> tuple[FloatArray, FloatArray]:
    m = pos.size
    n = neg.size
    v10 = np.empty(m, dtype=np.float64)
    v01 = np.empty(n, dtype=np.float64)
    for i in range(m):
        v10[i] = float(np.mean((pos[i] > neg) + 0.5 * (pos[i] == neg)))
    for j in range(n):
        v01[j] = float(np.mean((pos > neg[j]) + 0.5 * (pos == neg[j])))
    return v10, v01


def delong_variance(
    y_true: Sequence[int], scores: Sequence[float], positive: int = 1
) -> tuple[float, float]:
    truth = np.asarray(y_true, dtype=np.int_)
    values = np.asarray(scores, dtype=np.float64)
    pos = values[truth == positive]
    neg = values[truth != positive]
    if pos.size == 0 or neg.size == 0:
        return 0.5, 0.0
    v10, v01 = _placements(pos, neg)
    area = float(np.mean(v10))
    s10 = float(np.var(v10, ddof=1)) if pos.size > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if neg.size > 1 else 0.0
    variance = s10 / pos.size + s01 / neg.size
    return area, variance


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, data.size, data.size)
        means[b] = float(np.mean(data[draw]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def paired_t(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    result = stats.ttest_rel(np.asarray(a), np.asarray(b))
    return float(result.statistic), float(result.pvalue)


def cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return float(np.mean(diff)) / sd


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


def benjamini_hochberg(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        value = p[idx] * m / (rank + 1)
        running = min(running, value)
        adjusted[idx] = min(running, 1.0)
    return [float(x) for x in adjusted]


def cochran_q(estimates: Sequence[float], variances: Sequence[float]) -> tuple[float, float]:
    theta = np.asarray(estimates, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    weights = 1.0 / np.clip(var, 1e-12, None)
    pooled = float(np.sum(weights * theta) / np.sum(weights))
    q = float(np.sum(weights * (theta - pooled) ** 2))
    df = max(theta.size - 1, 1)
    p_value = float(stats.chi2.sf(q, df))
    return q, p_value


def i_squared(q: float, k: int) -> float:
    df = max(k - 1, 1)
    return float(max(0.0, (q - df) / q)) if q > 0.0 else 0.0
