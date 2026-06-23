from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _probs(values: Sequence[float]) -> FloatArray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], bins: int = 15
) -> float:
    p = _probs(probs)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        confidence = float(np.mean(p[mask]))
        accuracy = float(np.mean(y[mask]))
        total += (np.sum(mask) / p.size) * abs(confidence - accuracy)
    return float(total)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    p = _probs(probs)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def temperature_scale(
    logits: Sequence[float], labels: Sequence[int], steps: int = 200, lr: float = 0.05
) -> float:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    log_t = 0.0
    for _ in range(steps):
        temp = np.exp(log_t)
        scaled = z / temp
        prob = 1.0 / (1.0 + np.exp(-scaled))
        grad_scaled = prob - y
        grad_log_t = float(np.mean(grad_scaled * (-scaled)))
        log_t -= lr * grad_log_t
    return float(np.exp(log_t))


def calibration_slope(
    logits: Sequence[float], labels: Sequence[int], steps: int = 500, lr: float = 0.01
) -> float:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    slope = 1.0
    intercept = 0.0
    for _ in range(steps):
        linear = slope * z + intercept
        prob = 1.0 / (1.0 + np.exp(-linear))
        residual = prob - y
        grad_slope = float(np.mean(residual * z))
        grad_intercept = float(np.mean(residual))
        slope -= lr * grad_slope
        intercept -= lr * grad_intercept
    return float(slope)


def reliability_curve(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> tuple[FloatArray, FloatArray]:
    p = _probs(probs)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    confidences = []
    accuracies = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi)
        if not np.any(mask):
            continue
        confidences.append(float(np.mean(p[mask])))
        accuracies.append(float(np.mean(y[mask])))
    return np.asarray(confidences), np.asarray(accuracies)
