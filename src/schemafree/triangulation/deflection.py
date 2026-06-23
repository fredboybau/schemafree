from collections.abc import Sequence

import torch
from torch import Tensor


def consensus_drift(updates: Sequence[Tensor]) -> Tensor:
    stacked = torch.stack([u.reshape(-1) for u in updates], dim=0)
    return stacked.mean(dim=0)


def deflection(update: Tensor, drift: Tensor) -> Tensor:
    flat = update.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(flat, drift, dim=0, eps=1e-12)
    return 1.0 - cos


def cohort_divergences(updates: Sequence[Tensor]) -> list[Tensor]:
    drift = consensus_drift(updates)
    return [deflection(u, drift) for u in updates]
