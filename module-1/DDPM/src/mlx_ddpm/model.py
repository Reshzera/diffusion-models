import math

import mlx.core as mx
import mlx.nn as nn


def timestep_embedding(timesteps: mx.array, dim: int) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(10_000) * mx.arange(half) / max(half - 1, 1))
    args = timesteps.astype(mx.float32)[:, None] * freqs[None]
    emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array, time_emb: mx.array) -> mx.array:
        h = self.norm1(x)
        h = nn.silu(h)
        h = self.conv1(h)

        time_h = nn.silu(time_emb)
        time_h = self.time_proj(time_h)
        time_h = time_h[:, None, None, :]
        h = h + time_h

        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        skip = self.skip(x) if self.skip is not None else x
        return h + skip


class DDPMUNet(nn.Module):
    """Small U-Net noise predictor epsilon_theta(x_t, t).

    Inputs and outputs are NHWC tensors in [-1, 1], matching MLX Conv2d layout.
    """

    def __init__(self, image_channels: int = 3, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        c = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        ##32x32
        self.input = nn.Conv2d(image_channels, c, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(c, c, time_dim)
        ##16x16
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(c * 2, c * 2, time_dim)
        self.attention1 = MultiHeadSelfAttention(c * 2, num_heads=4, dropout=0.1)
        ##8x8
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResidualBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResidualBlock(c * 4, c * 4, time_dim)
        ##16x16
        self.upsample1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(c * 4, c * 2, time_dim)
        self.attention2 = MultiHeadSelfAttention(c * 2, num_heads=4, dropout=0.1)
        ##32x32
        self.upsample2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(c * 2, c, time_dim)
        ##32x32
        self.output_norm = nn.GroupNorm(min(8, c), c)
        self.output = nn.Conv2d(c, image_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        h0 = self.input(x)  # 32x32x64
        h1 = self.down1(h0, time_emb)  # 32x32x64
        h2 = self.downsample1(h1)  # 16x16x128
        h2 = self.down2(h2, time_emb)  # 16x16x128

        h = self.downsample2(h2)  # 8x8x256
        h = self.mid1(h, time_emb)  # 8x8x256
        h = self.mid2(h, time_emb)  # 8x8x256

        h = self.upsample1(h)  # 16x16x128
        h = mx.concatenate([h, h2], axis=-1)  # 16x16x256
        h = self.up1(h, time_emb)  # 16x16x128
        h = self.upsample2(h)  # 32x32x64
        h = mx.concatenate([h, h1], axis=-1)  # 32x32x128
        h = self.up2(h, time_emb)  # 32x32x64

        return self.output(nn.silu(self.output_norm(h)))  # 32x32x3