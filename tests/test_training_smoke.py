import os

import gin
import torch

from schemafree.datum.bearings import (
    DataSpec,
    DistillSpec,
    EncoderSpec,
    FederationSpec,
    OptimSpec,
    PrivacySpec,
)
from schemafree.datum.origin import set_seed
from schemafree.sheets._common import build_sources, cohort_sizes
from schemafree.triangulation.traverse import federated_pretrain

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SMOKE = os.path.join(_ROOT, "plans", "_smoke.gin")


def test_smoke_federation_produces_usable_encoder() -> None:
    gin.clear_config()
    try:
        gin.parse_config_file(_SMOKE)
        set_seed(7)
        encoder = EncoderSpec()
        data = DataSpec()
        model = federated_pretrain(
            encoder,
            build_sources(data, encoder, "", synthetic_batches=2),
            cohort_sizes(data),
            FederationSpec(),
            DistillSpec(),
            OptimSpec(),
            PrivacySpec(),
        )
        embedding = model.represent(torch.rand(3, 3, encoder.image_size, encoder.image_size))
        assert embedding.shape == (3, encoder.dim)
        assert torch.all(torch.isfinite(embedding))
    finally:
        gin.clear_config()
