import torch

from schemafree.control.probe import fit_probe, predict
from schemafree.datum.bearings import ProbeSpec


def test_linear_probe_overfits_separable_batch() -> None:
    torch.manual_seed(0)
    n = 64
    half = n // 2
    features = torch.cat(
        [
            torch.randn(half, 8) + 4.0,
            torch.randn(half, 8) - 4.0,
        ],
        dim=0,
    )
    labels = torch.cat([torch.ones(half), torch.zeros(half)]).long()
    probe = fit_probe(features, labels, num_classes=2, spec=ProbeSpec(epochs=300, lr=5e-2))
    pred, _ = predict(probe, features)
    accuracy = (pred == labels).float().mean().item()
    assert accuracy > 0.98


def test_probe_outputs_two_class_scores() -> None:
    torch.manual_seed(1)
    features = torch.randn(20, 6)
    labels = (torch.arange(20) % 2).long()
    probe = fit_probe(features, labels, num_classes=2, spec=ProbeSpec(epochs=10))
    _, prob = predict(probe, features)
    assert prob.shape == (20, 2)
    assert torch.allclose(prob.sum(dim=1), torch.ones(20), atol=1e-5)
