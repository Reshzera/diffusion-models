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
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_proj(nn.silu(time_emb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
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

        self.input = nn.Conv2d(image_channels, c, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(c, c, time_dim)
        self.downsample1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(c * 2, c * 2, time_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResidualBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResidualBlock(c * 4, c * 4, time_dim)

        self.upsample1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(c * 4, c * 2, time_dim)
        self.upsample2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(c * 2, c, time_dim)

        self.output_norm = nn.GroupNorm(min(8, c), c)
        self.output = nn.Conv2d(c, image_channels, kernel_size=3, padding=1)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        h0 = self.input(x)
        h1 = self.down1(h0, time_emb)
        h2 = self.downsample1(h1)
        h2 = self.down2(h2, time_emb)

        h = self.downsample2(h2)
        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        h = self.upsample1(h)
        h = mx.concatenate([h, h2], axis=-1)
        h = self.up1(h, time_emb)
        h = self.upsample2(h)
        h = mx.concatenate([h, h1], axis=-1)
        h = self.up2(h, time_emb)

        return self.output(nn.silu(self.output_norm(h)))
