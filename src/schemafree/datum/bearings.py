import dataclasses

import gin


@gin.configurable
@dataclasses.dataclass(frozen=True)
class EncoderSpec:
    image_size: int = 224
    patch_size: int = 16
    dim: int = 384
    depth: int = 12
    heads: int = 6
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    projection_dim: int = 512
    bottleneck_dim: int = 256
    hidden_dim: int = 2048


@gin.configurable
@dataclasses.dataclass(frozen=True)
class DistillSpec:
    student_temp: float = 0.04
    teacher_temp: float = 0.04
    warmup_teacher_temp: float = 0.02
    warmup_teacher_epochs: int = 10
    center_momentum: float = 0.9
    global_crops: int = 2
    local_crops: int = 6


@gin.configurable
@dataclasses.dataclass(frozen=True)
class FederationSpec:
    rounds: int = 100
    local_epochs: int = 3
    beta: float = 2.0
    aggregate_buffers: bool = False
    secure_sum: bool = True


@gin.configurable
@dataclasses.dataclass(frozen=True)
class PrivacySpec:
    enabled: bool = True
    clip_norm: float = 1.0
    delta: float = 1e-5
    sample_rate: float = 0.01
    noise_multiplier: float = 0.62
    target_epsilon: float = 8.0


@gin.configurable
@dataclasses.dataclass(frozen=True)
class OptimSpec:
    lr: float = 1e-3
    weight_decay: float = 0.04
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    clip_grad: float = 3.0


@gin.configurable
@dataclasses.dataclass(frozen=True)
class DataSpec:
    cohorts: tuple[str, ...] = ("herlev", "sipakmed", "mendeley_lbc", "cric")
    held_out: str = "herlev"
    image_size: int = 224
    local_batch: int = 256
    stain_normalize: bool = True
    label_fraction: float = 1.0
    dirichlet_alpha: float = 100.0
    num_clients: int = 4


@gin.configurable
@dataclasses.dataclass(frozen=True)
class ProbeSpec:
    epochs: int = 100
    lr: float = 1e-2
    weight_decay: float = 0.0
    batch_size: int = 256
    fine_tune: bool = False


@gin.configurable
@dataclasses.dataclass(frozen=True)
class RunSpec:
    seed: int = 42
    world_size: int = 4
    device: str = "cuda"
    amp: bool = False
    out_dir: str = "runs/principal"
    log_every: int = 10
