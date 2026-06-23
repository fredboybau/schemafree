import copy

import torch

from schemafree.datum.bearings import (
    DistillSpec,
    EncoderSpec,
    FederationSpec,
    OptimSpec,
    PrivacySpec,
)
from schemafree.datum.origin import set_seed
from schemafree.fieldwork.channels import SyntheticField
from schemafree.instruments.vernier import Occupant
from schemafree.triangulation.traverse import federated_pretrain


def _specs() -> tuple:
    encoder = EncoderSpec(
        image_size=32,
        patch_size=16,
        dim=48,
        depth=2,
        heads=3,
        mlp_ratio=2.0,
        projection_dim=64,
        bottleneck_dim=32,
        hidden_dim=96,
    )
    fed = FederationSpec(rounds=1, local_epochs=1, beta=2.0, secure_sum=False)
    distill = DistillSpec(global_crops=2, local_crops=2, warmup_teacher_epochs=1)
    optim = OptimSpec(lr=1e-2, weight_decay=0.0, warmup_epochs=0)
    privacy = PrivacySpec(enabled=False)
    return encoder, fed, distill, optim, privacy


def _sources() -> dict:
    return {
        "sipakmed": SyntheticField(2, 4, 32, 16, n_global=2, n_local=2, signal=0.3, seed=11),
        "mendeley_lbc": SyntheticField(2, 4, 32, 16, n_global=2, n_local=2, signal=0.9, seed=12),
    }


def test_one_round_moves_the_global_encoder() -> None:
    encoder, fed, distill, optim, privacy = _specs()
    set_seed(0)
    reference = copy.deepcopy(Occupant(encoder).encoder.state_dict())
    set_seed(0)
    model = federated_pretrain(
        encoder, _sources(), {"sipakmed": 4049, "mendeley_lbc": 963}, fed, distill, optim, privacy
    )
    moved = [torch.any(model.encoder.state_dict()[k] != reference[k]).item() for k in reference]
    assert any(moved)


def test_global_model_carries_no_distillation_center() -> None:
    encoder, fed, distill, optim, privacy = _specs()
    model = federated_pretrain(
        encoder, _sources(), {"sipakmed": 4049, "mendeley_lbc": 963}, fed, distill, optim, privacy
    )
    assert all("center" not in key for key in model.state_dict())


def test_sources_expose_images_without_labels() -> None:
    for source in _sources().values():
        for crops in source:
            assert all(isinstance(view, torch.Tensor) for view in crops)
            break
