import torch
from torch import Tensor, nn

from schemafree.datum.bearings import ProbeSpec
from schemafree.instruments.theodolite import VisionTransformer


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc(features)


@torch.no_grad()
def extract_features(encoder: VisionTransformer, images: Tensor) -> Tensor:
    encoder.eval()
    return encoder(images)


def fit_probe(
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    spec: ProbeSpec,
) -> LinearProbe:
    probe = LinearProbe(features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    criterion = nn.CrossEntropyLoss()
    standardized = _standardize(features)
    for _ in range(spec.epochs):
        optimizer.zero_grad()
        logits = probe(standardized)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    return probe


def predict(probe: LinearProbe, features: Tensor) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        logits = probe(_standardize(features))
        prob = torch.softmax(logits, dim=-1)
        return logits.argmax(dim=-1), prob


def _standardize(features: Tensor) -> Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-6
    return (features - mean) / std
