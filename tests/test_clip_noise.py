import torch

from schemafree.instruments.clamp import clip_and_noise, per_sample_grad_norms


def test_norms_match_manual() -> None:
    g = torch.Generator().manual_seed(5)
    grads = torch.randn(8, 3, 4, generator=g)
    norms = per_sample_grad_norms(grads)
    manual = grads.reshape(8, -1).norm(dim=1)
    assert torch.allclose(norms, manual, atol=1e-6)


def test_clipping_bounds_each_contribution() -> None:
    g = torch.Generator().manual_seed(6)
    grads = torch.randn(16, 10, generator=g) * 5.0
    clip = 1.0
    scaled = grads * torch.clamp(clip / grads.norm(dim=1, keepdim=True), max=1.0)
    assert torch.all(scaled.norm(dim=1) <= clip + 1e-5)


def test_zero_noise_is_clipped_mean() -> None:
    g = torch.Generator().manual_seed(7)
    grads = torch.randn(16, 10, generator=g) * 5.0
    clip = 1.0
    out = clip_and_noise(grads, clip, noise_multiplier=0.0)
    scaled = grads * torch.clamp(clip / grads.norm(dim=1, keepdim=True), max=1.0)
    expected = scaled.sum(dim=0) / 16
    assert out.shape == grads.shape[1:]
    assert torch.allclose(out, expected, atol=1e-6)


def test_noise_scales_with_multiplier() -> None:
    grads = torch.zeros(64, 2000)
    gen = torch.Generator().manual_seed(8)
    out = clip_and_noise(grads, clip_norm=1.0, noise_multiplier=4.0, generator=gen)
    empirical = out.std().item() * grads.shape[0]
    assert abs(empirical - 4.0) < 0.6
