from schemafree.triangulation.budget import compose_epsilon


def test_epsilon_decreases_with_noise() -> None:
    sigmas = [0.62, 0.95, 1.7, 3.1]
    eps = [compose_epsilon(s, 0.01, 100, 1e-5) for s in sigmas]
    for earlier, later in zip(eps, eps[1:]):
        assert later < earlier


def test_operating_points_track_the_swept_budget() -> None:
    bands = {
        0.62: (3.0, 9.0),
        0.95: (0.8, 3.0),
        1.7: (0.15, 1.0),
        3.1: (0.05, 0.4),
    }
    for sigma, (lo, hi) in bands.items():
        eps = compose_epsilon(sigma, 0.01, 100, 1e-5)
        assert lo <= eps <= hi, (sigma, eps)


def test_least_private_point_is_the_smallest_noise() -> None:
    eps = {s: compose_epsilon(s, 0.01, 100, 1e-5) for s in (0.62, 0.95, 1.7, 3.1)}
    assert max(eps, key=lambda s: eps[s]) == 0.62
    assert min(eps, key=lambda s: eps[s]) == 3.1


def test_zero_noise_is_unbounded() -> None:
    assert compose_epsilon(0.0, 0.01, 100, 1e-5) == float("inf")
