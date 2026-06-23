import torch

from schemafree.adjustment.selfdist import SelfDistillation, teacher_temp_schedule
from schemafree.datum.bearings import DistillSpec


def test_loss_is_non_negative_and_finite() -> None:
    spec = DistillSpec(global_crops=2, local_crops=2)
    loss_fn = SelfDistillation(spec, projection_dim=64)
    g = torch.Generator().manual_seed(0)
    student = torch.randn(16, 64, generator=g)
    teacher = torch.randn(8, 64, generator=g)
    value = loss_fn(student, teacher, n_student_crops=4, n_teacher_crops=2, teacher_temp=0.04)
    assert torch.isfinite(value)
    assert value.item() >= 0.0


def test_center_tracks_teacher_mean() -> None:
    spec = DistillSpec()
    loss_fn = SelfDistillation(spec, projection_dim=8)
    teacher = torch.full((8, 8), 3.0)
    before = loss_fn.center.clone()
    loss_fn(torch.randn(8, 8), teacher, n_student_crops=2, n_teacher_crops=2, teacher_temp=0.04)
    moved = loss_fn.center - before
    assert torch.all(moved > 0.0)


def test_teacher_temperature_warms_up() -> None:
    spec = DistillSpec(warmup_teacher_temp=0.02, teacher_temp=0.04, warmup_teacher_epochs=10)
    assert teacher_temp_schedule(spec, 0) == 0.02
    assert teacher_temp_schedule(spec, 10) == 0.04
    assert 0.02 < teacher_temp_schedule(spec, 5) < 0.04


def test_gradient_flows_to_student_only() -> None:
    spec = DistillSpec()
    loss_fn = SelfDistillation(spec, projection_dim=16)
    student = torch.randn(8, 16, requires_grad=True)
    teacher = torch.randn(8, 16, requires_grad=True)
    loss = loss_fn(student, teacher, n_student_crops=2, n_teacher_crops=2, teacher_temp=0.04)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
