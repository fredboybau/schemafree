from typing import Optional

import torch
from torch import Tensor


def per_sample_grad_norms(per_sample: Tensor) -> Tensor:
    flat = per_sample.reshape(per_sample.shape[0], -1)
    return flat.norm(dim=1)


def clip_and_noise(
    per_sample: Tensor,
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    batch = per_sample.shape[0]
    flat = per_sample.reshape(batch, -1)
    norms = flat.norm(dim=1, keepdim=True)
    scale = torch.clamp(clip_norm / (norms + 1e-12), max=1.0)
    clipped = (flat * scale).sum(dim=0)
    std = noise_multiplier * clip_norm
    if std > 0.0:
        noise = torch.normal(
            mean=0.0,
            std=std,
            size=clipped.shape,
            generator=generator,
            device=clipped.device,
            dtype=clipped.dtype,
        )
        clipped = clipped + noise
    return (clipped / batch).reshape(per_sample.shape[1:])
