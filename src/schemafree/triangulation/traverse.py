import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Optional, Protocol

import torch
from torch import Tensor, nn

from schemafree.adjustment.selfdist import SelfDistillation, teacher_temp_schedule
from schemafree.datum.bearings import (
    DistillSpec,
    EncoderSpec,
    FederationSpec,
    OptimSpec,
    PrivacySpec,
)
from schemafree.instruments.clamp import clip_and_noise
from schemafree.instruments.vernier import Occupant
from schemafree.triangulation.fix import div_aware_aggregate

TEACHER_MOMENTUM = 0.996


class FieldSource(Protocol):
    n_global: int

    def __iter__(self) -> "Iterator[list[Tensor]]":
        ...

    def __len__(self) -> int:
        ...


def _flatten(module: nn.Module) -> Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _load(module: nn.Module, flat: Tensor) -> None:
    offset = 0
    for p in module.parameters():
        count = p.numel()
        p.data.copy_(flat[offset : offset + count].reshape(p.shape))
        offset += count


def _grad_vector(module: nn.Module) -> Tensor:
    parts: list[Tensor] = []
    for p in module.parameters():
        if p.grad is None:
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts)


def _assign_grad(module: nn.Module, flat: Tensor) -> None:
    offset = 0
    for p in module.parameters():
        count = p.numel()
        p.grad = flat[offset : offset + count].reshape(p.shape).clone()
        offset += count


@torch.no_grad()
def _ema(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(momentum).add_(s.data, alpha=1.0 - momentum)


def _local_step(
    student: Occupant,
    teacher: Occupant,
    selfdist: SelfDistillation,
    optimizer: torch.optim.Optimizer,
    crops: Sequence[Tensor],
    n_global: int,
    teacher_temp: float,
    privacy: PrivacySpec,
    generator: Optional[torch.Generator],
) -> float:
    n_crops = len(crops)
    if not privacy.enabled:
        optimizer.zero_grad()
        student_out = student(list(crops))
        with torch.no_grad():
            teacher_out = teacher(list(crops[:n_global]))
        loss = selfdist(student_out, teacher_out, n_crops, n_global, teacher_temp)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    batch = crops[0].shape[0]
    grads: list[Tensor] = []
    running = 0.0
    for i in range(batch):
        sample = [c[i : i + 1] for c in crops]
        optimizer.zero_grad()
        student_out = student(sample)
        with torch.no_grad():
            teacher_out = teacher(sample[:n_global])
        loss = selfdist(student_out, teacher_out, n_crops, n_global, teacher_temp)
        loss.backward()
        grads.append(_grad_vector(student))
        running += float(loss.detach())
    stacked = torch.stack(grads, dim=0)
    aggregated = clip_and_noise(stacked, privacy.clip_norm, privacy.noise_multiplier, generator)
    optimizer.zero_grad()
    _assign_grad(student, aggregated)
    optimizer.step()
    return running / max(batch, 1)


def _train_cohort(
    global_model: Occupant,
    source: FieldSource,
    distill: DistillSpec,
    optim: OptimSpec,
    privacy: PrivacySpec,
    local_epochs: int,
    projection_dim: int,
    generator: Optional[torch.Generator],
) -> Occupant:
    student = copy.deepcopy(global_model)
    teacher = copy.deepcopy(global_model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    selfdist = SelfDistillation(distill, projection_dim)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=optim.lr,
        betas=optim.betas,
        weight_decay=optim.weight_decay,
    )
    for epoch in range(local_epochs):
        tt = teacher_temp_schedule(distill, epoch)
        for crops in source:
            _local_step(
                student,
                teacher,
                selfdist,
                optimizer,
                crops,
                source.n_global,
                tt,
                privacy,
                generator,
            )
            _ema(teacher, student, TEACHER_MOMENTUM)
    return student


def federated_pretrain(
    encoder_spec: EncoderSpec,
    sources: Mapping[str, FieldSource],
    sizes: Mapping[str, int],
    fed: FederationSpec,
    distill: DistillSpec,
    optim: OptimSpec,
    privacy: PrivacySpec,
    generator: Optional[torch.Generator] = None,
) -> Occupant:
    cohorts = list(sources.keys())
    global_model = Occupant(encoder_spec)
    cohort_sizes = [sizes[name] for name in cohorts]
    for _ in range(fed.rounds):
        enc_updates: list[Tensor] = []
        head_updates: list[Tensor] = []
        base_enc = _flatten(global_model.encoder)
        base_head = _flatten(global_model.head)
        for name in cohorts:
            trained = _train_cohort(
                global_model,
                sources[name],
                distill,
                optim,
                privacy,
                fed.local_epochs,
                encoder_spec.projection_dim,
                generator,
            )
            enc_updates.append(_flatten(trained.encoder) - base_enc)
            head_updates.append(_flatten(trained.head) - base_head)
        result = div_aware_aggregate(enc_updates, cohort_sizes, fed.beta)
        head_flat = torch.stack([h.reshape(-1) for h in head_updates], dim=0)
        head_aggregate = (result.weights.unsqueeze(1) * head_flat).sum(dim=0)
        _load(global_model.encoder, base_enc + result.aggregate.reshape(-1))
        _load(global_model.head, base_head + head_aggregate)
    return global_model
