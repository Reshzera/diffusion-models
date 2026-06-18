from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, hidden_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1, x.shape[-1])


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, num_tokens, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = mx.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = mx.transpose(x, (0, 2, 1, 3)).reshape(batch_size, num_tokens, hidden_dim)
        x = self.proj(x)
        return self.proj_drop(x)


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SmallViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        hidden_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, hidden_dim)
        num_tokens = self.patch_embed.num_patches + 1

        self.cls_token = mx.random.normal((1, 1, hidden_dim)) * 0.02
        self.pos_embed = mx.random.normal((1, num_tokens, hidden_dim)) * 0.02
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = [
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = mx.broadcast_to(self.cls_token, (batch_size, 1, self.cls_token.shape[-1]))
        x = mx.concatenate([cls_token, x], axis=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.head(x[:, 0])


def count_parameters(model: nn.Module) -> int:
    def count(tree) -> int:
        if isinstance(tree, mx.array):
            return math.prod(tree.shape)
        if isinstance(tree, dict):
            return sum(count(value) for value in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(count(value) for value in tree)
        return 0

    return count(model.parameters())
