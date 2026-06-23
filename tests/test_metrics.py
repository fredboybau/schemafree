import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from schemafree.control.calibration import expected_calibration_error
from schemafree.control.marks import auc, balanced_accuracy, macro_f1
from schemafree.control.significance import delong_variance


def test_macro_f1_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    p = rng.integers(0, 3, size=200)
    assert abs(macro_f1(y.tolist(), p.tolist()) - f1_score(y, p, average="macro")) < 1e-9


def test_balanced_accuracy_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, size=300)
    p = rng.integers(0, 4, size=300)
    assert abs(balanced_accuracy(y.tolist(), p.tolist()) - balanced_accuracy_score(y, p)) < 1e-9


def test_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=400)
    s = rng.normal(size=400) + y * 0.7
    assert abs(auc(y.tolist(), s.tolist()) - roc_auc_score(y, s)) < 1e-9


def test_delong_area_matches_rank_auc() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=250)
    s = rng.normal(size=250) + y * 0.5
    area, variance = delong_variance(y.tolist(), s.tolist())
    assert abs(area - roc_auc_score(y, s)) < 1e-9
    assert variance >= 0.0


def test_ece_is_a_probability_distance() -> None:
    rng = np.random.default_rng(4)
    probs = rng.uniform(size=500)
    labels = (rng.uniform(size=500) < probs).astype(int)
    value = expected_calibration_error(probs.tolist(), labels.tolist())
    assert 0.0 <= value <= 1.0
