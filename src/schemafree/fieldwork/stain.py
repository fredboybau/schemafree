from typing import Optional

import torch
from torch import Tensor

_HED_REFERENCE = torch.tensor(
    [
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78],
    ]
)


def _to_optical_density(rgb: Tensor) -> Tensor:
    return -torch.log10(torch.clamp(rgb, min=1e-6))


def _from_optical_density(od: Tensor) -> Tensor:
    return torch.clamp(torch.pow(10.0, -od), 0.0, 1.0)


class StainNormalizer:
    def __init__(self, beta: float = 0.15) -> None:
        self.beta = beta
        self.reference: Tensor = _HED_REFERENCE.clone()
        self.target_concentration: Optional[Tensor] = None

    def fit(self, sample: Tensor) -> "StainNormalizer":
        od = _to_optical_density(sample).reshape(-1, 3)
        mask = od.sum(dim=1) > self.beta
        kept = od[mask] if mask.any() else od
        self.target_concentration = torch.quantile(kept, 0.99, dim=0)
        return self

    def apply(self, image: Tensor) -> Tensor:
        od = _to_optical_density(image)
        c, h, w = od.shape
        flat = od.reshape(c, -1)
        stains = torch.linalg.lstsq(self.reference.T, flat).solution
        if self.target_concentration is not None:
            current = torch.quantile(stains, 0.99, dim=1, keepdim=True)
            scale = self.target_concentration.reshape(-1, 1) / (current + 1e-6)
            stains = stains * scale
        recon = (self.reference.T @ stains).reshape(c, h, w)
        return _from_optical_density(recon)


def stain_jitter(
    image: Tensor,
    strength: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    od = _to_optical_density(image)
    c, h, w = od.shape
    flat = od.reshape(c, -1)
    stains = torch.linalg.lstsq(_HED_REFERENCE.T, flat).solution
    alpha = 1.0 + strength * (
        2.0 * torch.rand(stains.shape[0], 1, generator=generator, device=image.device) - 1.0
    )
    bias = strength * (
        2.0 * torch.rand(stains.shape[0], 1, generator=generator, device=image.device) - 1.0
    )
    stains = stains * alpha + bias
    recon = (_HED_REFERENCE.T @ stains).reshape(c, h, w)
    return _from_optical_density(recon)
