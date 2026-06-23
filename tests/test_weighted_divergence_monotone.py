import numpy as np


def _weighted_divergence(d: np.ndarray, sizes: np.ndarray, beta: float) -> float:
    log_w = -beta * d + np.log(sizes)
    log_w = log_w - log_w.max()
    w = np.exp(log_w)
    w = w / w.sum()
    return float(np.sum(w * d))


def test_weighted_divergence_is_non_increasing_in_beta() -> None:
    rng = np.random.default_rng(11)
    d = rng.uniform(0.0, 2.0, size=6)
    sizes = rng.uniform(500.0, 12000.0, size=6)
    betas = np.linspace(0.0, 8.0, 40)
    values = [_weighted_divergence(d, sizes, b) for b in betas]
    diffs = np.diff(values)
    assert np.all(diffs <= 1e-9)


def test_strictly_decreasing_when_divergences_differ() -> None:
    d = np.array([0.1, 0.9, 1.4, 1.8])
    sizes = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    low = _weighted_divergence(d, sizes, 0.5)
    high = _weighted_divergence(d, sizes, 5.0)
    assert high < low
