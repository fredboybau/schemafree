import torch
from torch import Tensor, nn

from schemafree.datum.bearings import EncoderSpec
from schemafree.instruments.theodolite import VisionTransformer, build_encoder


class UnitRowLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.direction = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.trunc_normal_(self.direction, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        weight = nn.functional.normalize(self.direction, dim=1, p=2)
        return nn.functional.linear(x, weight)


class DinoHead(nn.Module):
    def __init__(self, in_dim: int, spec: EncoderSpec) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.hidden_dim),
            nn.GELU(),
            nn.Linear(spec.hidden_dim, spec.bottleneck_dim),
        ]
        self.mlp = nn.Sequential(*layers)
        self.last = UnitRowLinear(spec.bottleneck_dim, spec.projection_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return self.last(x)


class Occupant(nn.Module):
    def __init__(self, spec: EncoderSpec) -> None:
        super().__init__()
        self.encoder: VisionTransformer = build_encoder(spec)
        self.head = DinoHead(self.encoder.embed_dim, spec)

    def represent(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, crops: list[Tensor]) -> Tensor:
        outputs: list[Tensor] = []
        start = 0
        while start < len(crops):
            stop = start + 1
            while stop < len(crops) and crops[stop].shape[-1] == crops[start].shape[-1]:
                stop += 1
            batch = torch.cat(crops[start:stop], dim=0)
            features = self.encoder(batch)
            outputs.append(self.head(features))
            start = stop
        return torch.cat(outputs, dim=0)
