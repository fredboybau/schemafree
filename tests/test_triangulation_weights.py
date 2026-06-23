import torch

from schemafree.triangulation.fix import div_aware_aggregate


def _random_updates(k: int, dim: int, seed: int) -> list:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(dim, generator=g) for _ in range(k)]


def test_weights_live_on_the_simplex() -> None:
    updates = _random_updates(4, 32, seed=1)
    sizes = [917, 4049, 963, 11534]
    result = div_aware_aggregate(updates, sizes, beta=2.0)
    assert torch.isclose(result.weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(result.weights >= 0.0)


def test_beta_zero_is_size_proportional() -> None:
    updates = _random_updates(4, 16, seed=2)
    sizes = [917, 4049, 963, 11534]
    result = div_aware_aggregate(updates, sizes, beta=0.0)
    expected = torch.tensor(sizes, dtype=result.weights.dtype)
    expected = expected / expected.sum()
    assert torch.allclose(result.weights, expected, atol=1e-6)


def test_divergence_stays_in_unit_interval_pair() -> None:
    updates = _random_updates(5, 64, seed=3)
    result = div_aware_aggregate(updates, [1] * 5, beta=1.0)
    assert torch.all(result.divergences >= -1e-5)
    assert torch.all(result.divergences <= 2.0 + 1e-5)


def test_identical_updates_have_zero_divergence() -> None:
    base = torch.randn(20, generator=torch.Generator().manual_seed(4))
    updates = [base.clone() for _ in range(3)]
    result = div_aware_aggregate(updates, [10, 20, 30], beta=3.0)
    assert torch.allclose(result.divergences, torch.zeros(3), atol=1e-5)
    expected = torch.tensor([10.0, 20.0, 30.0])
    expected = expected / expected.sum()
    assert torch.allclose(result.weights, expected, atol=1e-5)
