import numpy as np
from scipy import optimize


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _objective(z: np.ndarray, d: np.ndarray, prior: np.ndarray, beta: float) -> float:
    w = _softmax(z)
    rel_entropy = -np.sum(w * np.log((w + 1e-12) / prior))
    return float(np.sum(w * d) - rel_entropy / beta)


def _analytic(d: np.ndarray, sizes: np.ndarray, beta: float) -> np.ndarray:
    log_w = -beta * d + np.log(sizes)
    return _softmax(log_w)


def test_formula_matches_entropy_regularized_minimizer() -> None:
    rng = np.random.default_rng(0)
    for _ in range(8):
        k = 4
        d = rng.uniform(0.0, 2.0, size=k)
        sizes = rng.uniform(500.0, 12000.0, size=k)
        prior = sizes / sizes.sum()
        beta = float(rng.uniform(0.5, 4.0))
        analytic = _analytic(d, sizes, beta)

        best = None
        for _ in range(6):
            z0 = rng.normal(size=k)
            res = optimize.minimize(
                _objective,
                z0,
                args=(d, prior, beta),
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000},
            )
            if best is None or res.fun < best.fun:
                best = res
        numeric = _softmax(best.x)

        assert np.allclose(analytic, numeric, atol=1e-3)
        analytic_obj = _objective(np.log(analytic + 1e-12), d, prior, beta)
        assert analytic_obj <= best.fun + 1e-6
