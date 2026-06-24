# MLX DDPM

Educational Denoising Diffusion Probabilistic Model implementation in [MLX](https://ml-explore.github.io/mlx/) using `uv`.

Reference paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), Ho, Jain, Abbeel, 2020.

## What This Implements

- Forward process `q(x_t | x_0)` with a linear beta schedule.
- Noise-prediction objective `E[||epsilon - epsilon_theta(x_t, t)||^2]` from the simplified DDPM loss.
- Reverse denoising sampler using the learned mean parameterization.
- Small U-Net epsilon predictor with sinusoidal timestep embeddings.
- NHWC image tensors because MLX convolutions use channels-last layout.

This is intentionally compact for Module 1 study work, not a production image generator.

## Setup

```bash
uv sync
```

## Smoke Test

```bash
uv run mlx-ddpm smoke
```

## Train

Point `--data-dir` at a folder containing images. Images are center-cropped, resized, converted to `[-1, 1]`, and loaded recursively.

```bash
uv run mlx-ddpm train \
  --data-dir /path/to/images \
  --image-size 32 \
  --channels 3 \
  --batch-size 32 \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/ddpm
```

The final checkpoint is saved to `runs/ddpm/ddpm.safetensors`. Intermediate checkpoints are saved to `runs/ddpm/checkpoints/ddpm_step_*.safetensors` when `--save-every` is non-zero. Intermediate samples are saved as `sample_*.png` when `--sample-every` is non-zero.

## CIFAR-10

The training CLI can also download CIFAR-10 automatically and convert the official Python batches to PNG images:

```bash
uv run mlx-ddpm train \
  --dataset cifar-10 \
  --image-size 32 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 32 \
  --timesteps 500 \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/cifar10-ddpm
```

By default this saves the archive and converted images under `data/cifar-10`. Override that with `--data-dir` if needed.

## Oxford Flowers 102

The training CLI can download and extract Oxford Flowers 102 automatically:

```bash
uv run mlx-ddpm train \
  --dataset oxford-flowers \
  --data-dir data/oxford-flowers \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 16 \
  --timesteps 500 \
  --save-every 1000 \
  --steps 10000
```

The dataset archive is saved as `data/oxford-flowers/102flowers.tgz` and images are extracted to `data/oxford-flowers/jpg`.

Generate from a saved checkpoint:

```bash
uv run mlx-ddpm sample \
  --checkpoint runs/ddpm/checkpoints/ddpm_step_001000.safetensors \
  --image-size 64 \
  --channels 3 \
  --timesteps 500 \
  --output runs/ddpm/test_samples.png
```

## Sample

Use the same model-shape arguments used during training.

```bash
uv run mlx-ddpm sample \
  --checkpoint runs/ddpm/ddpm.safetensors \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --output runs/ddpm/samples.png
```

## Useful Smaller Experiment

For quick iteration on a small dataset:

```bash
uv run mlx-ddpm train \
  --data-dir /path/to/images \
  --image-size 32 \
  --base-channels 32 \
  --timesteps 200 \
  --batch-size 16 \
  --steps 1000
```
