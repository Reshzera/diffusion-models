# Small ViT With MLX

Module 0 experiment from the roadmap: a minimal Vision Transformer classifier for CIFAR-10, implemented with Apple MLX.

## Setup

```bash
uv sync
```

## Smoke Test

Runs one epoch on synthetic CIFAR-shaped data to verify the model, gradients, and optimizer.

```bash
uv run small-vit-train --smoke-test --epochs 1
```

## Train On CIFAR-10

The script downloads the original CIFAR-10 Python archive into `data/` on first run.

```bash
uv run small-vit-train --epochs 10 --batch-size 128 --lr 3e-4
```

Training saves the latest weights to `checkpoints/latest.safetensors` after every epoch.

Useful smaller run:

```bash
uv run small-vit-train --epochs 2 --train-limit 5000 --batch-size 128
```

## Manual Testing

Test one image from the CIFAR-10 test split:

```bash
uv run small-vit-predict --cifar-index 0
```

Test a local image. It will be resized to `32x32`, so use an image that belongs to one of the CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```bash
uv run small-vit-predict --image path/to/image.jpg
```

If you trained with non-default architecture flags, pass the same flags to prediction, for example:

```bash
uv run small-vit-predict --cifar-index 0 --hidden-dim 96 --depth 2 --num-heads 3
```

## Architecture

- Image size: `32x32`
- Patch size: `4x4`
- Tokens: `64` image tokens plus one class token
- Hidden dim: `192`
- Transformer depth: `6`
- Attention heads: `3`
- MLP ratio: `4`
- Classes: `10`

This mirrors the roadmap's ViT skeleton:

```text
Image -> Patches -> Linear Patch Projection -> Transformer Blocks -> Prediction Head
```
