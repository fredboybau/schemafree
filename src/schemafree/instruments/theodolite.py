import math
from typing import Optional

import torch
from torch import Tensor, nn

from schemafree.datum.bearings import EncoderSpec


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, spec: EncoderSpec) -> None:
        super().__init__()
        self.embed_dim = spec.dim
        self.patch_embed = PatchEmbed(spec.image_size, spec.patch_size, spec.dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spec.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, spec.dim))
        self.blocks = nn.ModuleList(
            [Block(spec.dim, spec.heads, spec.mlp_ratio) for _ in range(spec.depth)]
        )
        self.norm = nn.LayerNorm(spec.dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _interpolate(self, x: Tensor, grid: int) -> Tensor:
        n = self.pos_embed.shape[1] - 1
        if grid * grid == n:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]
        dim = patch_pos.shape[-1]
        side = int(math.sqrt(n))
        patch_pos = patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos, size=(grid, grid), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid * grid, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: Tensor, return_tokens: bool = False) -> Tensor:
        b = x.shape[0]
        grid = x.shape[-1] // self.patch_embed.proj.kernel_size[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self._interpolate(tokens, grid)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        if return_tokens:
            return tokens
        return tokens[:, 0]


def build_encoder(spec: Optional[EncoderSpec] = None) -> VisionTransformer:
    return VisionTransformer(spec if spec is not None else EncoderSpec())
