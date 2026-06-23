from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor

COMM_OVERHEAD = 0.184


def secure_sum(
    vectors: Sequence[Tensor],
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    n = len(vectors)
    if n == 0:
        raise ValueError("secure_sum requires at least one vector")
    masked: list[Tensor] = [v.clone() for v in vectors]
    for i in range(n):
        for j in range(i + 1, n):
            pad = torch.randn(
                vectors[i].shape,
                generator=generator,
                device=vectors[i].device,
                dtype=vectors[i].dtype,
            )
            masked[i] = masked[i] + pad
            masked[j] = masked[j] - pad
    total = masked[0].clone()
    for m in masked[1:]:
        total = total + m
    return total
