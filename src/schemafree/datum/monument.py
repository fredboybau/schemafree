import dataclasses
import os
import tempfile
from collections.abc import Mapping
from typing import Any

import torch


@dataclasses.dataclass
class Monument:
    round_index: int
    seed: int
    state: dict[str, torch.Tensor]
    extra: dict[str, Any]


def stash(path: str, monument: Monument) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        "round_index": monument.round_index,
        "seed": monument.seed,
        "state": monument.state,
        "extra": monument.extra,
    }
    handle, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def restore(path: str, map_location: str = "cpu") -> Monument:
    payload: Mapping[str, Any] = torch.load(path, map_location=map_location)
    return Monument(
        round_index=int(payload["round_index"]),
        seed=int(payload["seed"]),
        state=dict(payload["state"]),
        extra=dict(payload["extra"]),
    )
