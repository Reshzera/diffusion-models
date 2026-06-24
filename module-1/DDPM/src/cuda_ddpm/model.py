import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, hidden_dim)
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        time_h = F.silu(time_emb)
        time_h = self.time_proj(time_h)
        time_h = time_h[:, :, None, None]
        h = h + time_h

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        skip = self.skip(x) if self.skip is not None else x
        return h + skip


class DDPMUNet(nn.Module):
    """Small U-Net noise predictor epsilon_theta(x_t, t).

    Inputs and outputs are NCHW tensors in [-1, 1], matching PyTorch Conv2d layout.
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
        self.attention1 = MultiHeadSelfAttention(c * 2, num_heads=4, dropout=0.1)
        self.downsample2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.mid1 = ResidualBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResidualBlock(c * 4, c * 4, time_dim)
        self.upsample1 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResidualBlock(c * 4, c * 2, time_dim)
        self.attention2 = MultiHeadSelfAttention(c * 2, num_heads=4, dropout=0.1)
        self.upsample2 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.up2 = ResidualBlock(c * 2, c, time_dim)
        self.output_norm = nn.GroupNorm(min(8, c), c)
        self.output = nn.Conv2d(c, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        h0 = self.input(x)
        h1 = self.down1(h0, time_emb)
        h2 = self.downsample1(h1)
        h2 = self.down2(h2, time_emb)

        h = self.downsample2(h2)
        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        h = self.upsample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, time_emb)
        h = self.upsample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, time_emb)

        return self.output(F.silu(self.output_norm(h)))
