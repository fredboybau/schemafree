import torch
from torch import Tensor, nn

from schemafree.datum.bearings import DistillSpec


def teacher_temp_schedule(spec: DistillSpec, epoch: int) -> float:
    if epoch >= spec.warmup_teacher_epochs:
        return spec.teacher_temp
    span = spec.teacher_temp - spec.warmup_teacher_temp
    frac = epoch / max(spec.warmup_teacher_epochs, 1)
    return spec.warmup_teacher_temp + span * frac


class SelfDistillation(nn.Module):
    center: Tensor

    def __init__(self, spec: DistillSpec, projection_dim: int) -> None:
        super().__init__()
        self.student_temp = spec.student_temp
        self.center_momentum = spec.center_momentum
        self.register_buffer("center", torch.zeros(1, projection_dim))

    def forward(
        self,
        student: Tensor,
        teacher: Tensor,
        n_student_crops: int,
        n_teacher_crops: int,
        teacher_temp: float,
    ) -> Tensor:
        student_chunks = student.div(self.student_temp).chunk(n_student_crops)
        teacher_probs = torch.softmax((teacher - self.center) / teacher_temp, dim=-1)
        teacher_chunks = teacher_probs.detach().chunk(n_teacher_crops)

        total = torch.zeros((), device=student.device, dtype=student.dtype)
        pairs = 0
        for ti, target in enumerate(teacher_chunks):
            for si, pred in enumerate(student_chunks):
                if si == ti:
                    continue
                log_pred = torch.log_softmax(pred, dim=-1)
                total = total + torch.sum(-target * log_pred, dim=-1).mean()
                pairs += 1
        self._update_center(teacher)
        return total.div(max(pairs, 1))

    @torch.no_grad()
    def _update_center(self, teacher: Tensor) -> None:
        batch_center = teacher.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )
