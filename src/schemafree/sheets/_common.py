import os

import torch
from torch import Tensor

from schemafree.datum.bearings import DataSpec, EncoderSpec
from schemafree.fieldwork.channels import FolderField, MultiCropTransform, SyntheticField
from schemafree.fieldwork.parties import COHORTS
from schemafree.fieldwork.stain import StainNormalizer
from schemafree.triangulation.traverse import FieldSource


def training_cohorts(data: DataSpec) -> list[str]:
    return [name for name in data.cohorts if name != data.held_out]


def build_sources(
    data: DataSpec,
    encoder: EncoderSpec,
    data_root: str,
    synthetic_batches: int,
) -> dict[str, FieldSource]:
    sources: dict[str, FieldSource] = {}
    local_size = max(encoder.patch_size * 2, encoder.image_size // 2)
    for offset, name in enumerate(training_cohorts(data)):
        cohort_dir = os.path.join(data_root, name)
        if data_root and os.path.isdir(cohort_dir):
            normalizer = StainNormalizer() if data.stain_normalize else None
            transform = MultiCropTransform(
                image_size=encoder.image_size,
                local_size=local_size,
                normalizer=normalizer,
            )
            sources[name] = FolderField(cohort_dir, transform, data.local_batch)
        else:
            sources[name] = SyntheticField(
                num_batches=synthetic_batches,
                batch_size=min(data.local_batch, 8),
                image_size=encoder.image_size,
                local_size=local_size,
                seed=1000 + offset,
            )
    return sources


def cohort_sizes(data: DataSpec) -> dict[str, int]:
    nominal = {
        "herlev": 917,
        "sipakmed": 4049,
        "mendeley_lbc": 963,
        "cric": 11534,
    }
    return {
        name: nominal.get(name, len(COHORTS[name].native_classes))
        for name in training_cohorts(data)
    }


def labeled_synthetic(
    encoder: EncoderSpec,
    count: int,
    seed: int,
    signal: float = 0.4,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    labels = torch.arange(count).remainder(2).long()
    images = torch.rand(count, 3, encoder.image_size, encoder.image_size, generator=generator)
    bump = labels.reshape(-1, 1, 1, 1).float() * signal
    images = torch.clamp(images + bump, 0.0, 1.0)
    return images, labels
