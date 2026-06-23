import torch

from schemafree.datum.bearings import EncoderSpec
from schemafree.instruments.theodolite import build_encoder
from schemafree.instruments.vernier import Occupant


def test_encoder_emits_cls_embedding() -> None:
    spec = EncoderSpec()
    encoder = build_encoder(spec)
    out = encoder(torch.zeros(2, 3, spec.image_size, spec.image_size))
    assert out.shape == (2, spec.dim)


def test_token_grid_count() -> None:
    spec = EncoderSpec()
    encoder = build_encoder(spec)
    tokens = encoder(torch.zeros(1, 3, spec.image_size, spec.image_size), return_tokens=True)
    patches = (spec.image_size // spec.patch_size) ** 2
    assert tokens.shape == (1, patches + 1, spec.dim)


def test_projection_dimension() -> None:
    spec = EncoderSpec(image_size=64, dim=96, depth=2, heads=3, projection_dim=512)
    occupant = Occupant(spec)
    crops = [torch.rand(3, 3, 64, 64), torch.rand(3, 3, 32, 32)]
    out = occupant(crops)
    assert out.shape == (6, 512)


def test_local_crop_uses_interpolated_positions() -> None:
    spec = EncoderSpec(image_size=64, dim=48, depth=1, heads=3)
    encoder = build_encoder(spec)
    small = encoder(torch.rand(2, 3, 32, 32))
    assert small.shape == (2, spec.dim)
