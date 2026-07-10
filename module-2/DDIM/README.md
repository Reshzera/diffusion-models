# Module 2 DDIM

Educational Module 2 implementation for faster diffusion sampling and cleaner sampler configuration.

This module keeps the Module 1 U-Net and epsilon-prediction training objective, then updates the diffusion layer with:

- DDIM sampling for fewer denoising steps without retraining.
- An ancestral baseline sampler exposed as `--sampler ancestral`.
- Linear and cosine beta schedules.
- A configurable sampler interface shared by training-time previews, checkpoint sampling, and smoke tests.
- MLX support for Apple silicon and PyTorch/CUDA support for NVIDIA GPUs.

Reference papers:

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502), Song, Meng, Ermon, 2020.
- Ho, Jain, Abbeel, 2020, for the base epsilon-prediction objective.
- Nichol, Dhariwal, 2021, for cosine schedule motivation.

## Setup

```bash
uv sync
```

Use `mlx-ddim` on Apple silicon. Use `cuda-ddim` on NVIDIA GPU machines.

## Smoke Test

Run the default DDIM path with a cosine schedule:

```bash
uv run mlx-ddim smoke --noise-schedule cosine --sample-steps 4
uv run cuda-ddim smoke --device auto --noise-schedule cosine --sample-steps 4
```

Run the ancestral baseline sampler:

```bash
uv run mlx-ddim smoke --sampler ancestral
uv run cuda-ddim smoke --device auto --sampler ancestral
```

## Train

Training is still the Module 1 epsilon-prediction objective. Point `--data-dir` at a folder containing images. Images are center-cropped, resized, converted to `[-1, 1]`, and loaded recursively.

```bash
uv run mlx-ddim train \
  --data-dir /path/to/images \
  --image-size 32 \
  --channels 3 \
  --batch-size 32 \
  --timesteps 1000 \
  --noise-schedule linear \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/ddim
```

CUDA equivalent:

```bash
uv run cuda-ddim train \
  --data-dir /path/to/images \
  --image-size 32 \
  --channels 3 \
  --batch-size 32 \
  --timesteps 1000 \
  --noise-schedule linear \
  --steps 10000 \
  --save-every 1000 \
  --output-dir runs/cuda-ddim \
  --device cuda
```

If you train with `--noise-schedule cosine`, sample the checkpoint with the same schedule.

## Faster Sampling

DDIM uses the same trained U-Net but follows a reduced timestep grid. `--sample-steps` controls the number of denoising evaluations.

```bash
uv run mlx-ddim sample \
  --checkpoint runs/ddim/ddim.safetensors \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --noise-schedule linear \
  --sample-steps 50 \
  --output runs/ddim/ddim_50_steps.png
```

CUDA equivalent:

```bash
uv run cuda-ddim sample \
  --checkpoint runs/cuda-ddim/ddim.pt \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --noise-schedule linear \
  --sample-steps 50 \
  --output runs/cuda-ddim/ddim_50_steps.png \
  --device cuda
```

`--ddim-eta 0.0` is deterministic. Increase `--ddim-eta` above zero to inject stochasticity.

## Compare Samplers

Generate with the full ancestral sampler:

```bash
uv run cuda-ddim sample \
  --checkpoint runs/cuda-ddim/ddim.pt \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --sampler ancestral \
  --output runs/cuda-ddim/ancestral_1000_steps.png \
  --device cuda
```

Generate with a 50-step DDIM sampler from the same checkpoint:

```bash
uv run cuda-ddim sample \
  --checkpoint runs/cuda-ddim/ddim.pt \
  --image-size 32 \
  --channels 3 \
  --timesteps 1000 \
  --sample-steps 50 \
  --output runs/cuda-ddim/ddim_50_steps.png \
  --device cuda
```

## Reuse Previous Weights

You can load a checkpoint from Module 1 by passing its file path explicitly. Keep `--image-size`, `--channels`, `--base-channels`, `--time-dim`, `--timesteps`, and `--noise-schedule` consistent with the training run.

```bash
uv run cuda-ddim sample \
  --checkpoint /path/to/previous-module-checkpoint.pt \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --timesteps 1000 \
  --noise-schedule linear \
  --sample-steps 50 \
  --output runs/cuda-ddim/reused_weights.png \
  --device cuda
```

## Dataset Shortcuts

The training CLI still supports the Module 1 dataset helpers:

```bash
uv run cuda-ddim train \
  --dataset cifar-10 \
  --data-dir data/cifar-10 \
  --image-size 32 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 32 \
  --timesteps 1000 \
  --steps 10000 \
  --device cuda
```

```bash
uv run cuda-ddim train \
  --dataset oxford-flowers \
  --data-dir data/oxford-flowers \
  --image-size 64 \
  --channels 3 \
  --base-channels 64 \
  --batch-size 16 \
  --timesteps 1000 \
  --steps 50000 \
  --device cuda
```

## Notes

- DDIM does not change the model weights or the training loss.
- The noise schedule is part of training and sampling configuration. Keep it consistent for checkpoints.
- `--sample-steps` controls DDIM sampling. The ancestral baseline always uses all `--timesteps` steps.
- This module intentionally stops at DDIM and cosine schedules. Euler and Heun samplers are left as optional EDM experiments.
