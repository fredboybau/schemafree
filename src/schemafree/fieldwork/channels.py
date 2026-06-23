import os
from collections.abc import Iterator
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

from schemafree.fieldwork.stain import StainNormalizer, stain_jitter

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class MultiCropTransform:
    def __init__(
        self,
        image_size: int = 224,
        local_size: int = 96,
        n_global: int = 2,
        n_local: int = 6,
        jitter_strength: float = 0.05,
        normalizer: Optional[StainNormalizer] = None,
    ) -> None:
        self.image_size = image_size
        self.local_size = local_size
        self.n_global = n_global
        self.n_local = n_local
        self.jitter_strength = jitter_strength
        self.normalizer = normalizer

    def _one(self, image: Tensor, size: int, generator: Optional[torch.Generator]) -> Tensor:
        c, h, w = image.shape
        side = min(h, w, size)
        top = int(torch.randint(0, h - side + 1, (1,), generator=generator).item())
        left = int(torch.randint(0, w - side + 1, (1,), generator=generator).item())
        crop = image[:, top : top + side, left : left + side].unsqueeze(0)
        crop = torch.nn.functional.interpolate(
            crop, size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)
        if torch.rand(1, generator=generator).item() < 0.5:
            crop = torch.flip(crop, dims=[2])
        crop = stain_jitter(crop, self.jitter_strength, generator)
        if self.normalizer is not None:
            crop = self.normalizer.apply(crop)
        return crop

    def __call__(self, image: Tensor, generator: Optional[torch.Generator] = None) -> list[Tensor]:
        crops = [self._one(image, self.image_size, generator) for _ in range(self.n_global)]
        crops += [self._one(image, self.local_size, generator) for _ in range(self.n_local)]
        return crops


class SyntheticField:
    def __init__(
        self,
        num_batches: int,
        batch_size: int,
        image_size: int,
        local_size: int,
        n_global: int = 2,
        n_local: int = 2,
        signal: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.image_size = image_size
        self.local_size = local_size
        self.n_global = n_global
        self.n_local = n_local
        self.signal = signal
        self.seed = seed

    def __len__(self) -> int:
        return self.num_batches

    def _crop(self, size: int, generator: torch.Generator) -> Tensor:
        base = torch.rand(self.batch_size, 3, size, size, generator=generator)
        bias = torch.rand(self.batch_size, 1, 1, 1, generator=generator) * self.signal
        return base.add(bias).clamp(0.0, 1.0)

    def __iter__(self) -> Iterator[list[Tensor]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for _ in range(self.num_batches):
            crops = [self._crop(self.image_size, generator) for _ in range(self.n_global)]
            crops += [self._crop(self.local_size, generator) for _ in range(self.n_local)]
            yield crops


class FolderField:
    def __init__(
        self,
        root: str,
        transform: MultiCropTransform,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        self.transform = transform
        self.batch_size = batch_size
        self.seed = seed
        self.n_global = transform.n_global
        self.paths = self._scan(root)
        if not self.paths:
            raise FileNotFoundError(f"no images found under {root}")

    @staticmethod
    def _scan(root: str) -> list[str]:
        found: list[str] = []
        for base, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(_IMAGE_SUFFIXES):
                    found.append(os.path.join(base, name))
        return sorted(found)

    def __len__(self) -> int:
        return (len(self.paths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[Tensor]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        for start in range(0, len(self.paths), self.batch_size):
            chunk = self.paths[start : start + self.batch_size]
            per_image: list[list[Tensor]] = []
            for path in chunk:
                with Image.open(path) as handle:
                    image = TF.to_tensor(handle.convert("RGB"))
                per_image.append(self.transform(image, generator))
            n_crops = len(per_image[0])
            yield [torch.stack([img[c] for img in per_image], dim=0) for c in range(n_crops)]
