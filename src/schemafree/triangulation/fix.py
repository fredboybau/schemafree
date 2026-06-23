import dataclasses
from collections.abc import Sequence

import torch
from torch import Tensor

from schemafree.triangulation.deflection import cohort_divergences


@dataclasses.dataclass
class DivAwareResult:
    aggregate: Tensor
    weights: Tensor
    divergences: Tensor


def div_aware_aggregate(
    updates: Sequence[Tensor],
    sizes: Sequence[int],
    beta: float,
) -> DivAwareResult:
    if len(updates) != len(sizes):
        raise ValueError("updates and sizes must align")
    divergences: list[Tensor] = cohort_divergences(updates)
    d = torch.stack(divergences)
    n = torch.tensor(sizes, dtype=d.dtype, device=d.device)
    log_w = torch.log(n).sub(d.mul(beta))
    weights = torch.softmax(log_w, dim=0)
    flat = torch.stack([u.reshape(-1) for u in updates], dim=0)
    aggregate = (weights.unsqueeze(1) * flat).sum(dim=0)
    return DivAwareResult(
        aggregate=aggregate.reshape(updates[0].shape),
        weights=weights,
        divergences=d,
    )
