import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image
from tqdm import trange

from mlx_ddim.data import download_cifar10, download_oxford_flowers, image_batches, list_images
from mlx_ddim.diffusion import DDIMSchedule
from mlx_ddim.model import DDIMUNet


def save_image_grid(images: mx.array, path: str, columns: int = 4) -> None:
    array = np.array(((images + 1.0) * 127.5).astype(mx.uint8))
    n, height, width, channels = array.shape
    rows = int(np.ceil(n / columns))
    if channels == 1:
        canvas = np.zeros((rows * height, columns * width), dtype=np.uint8)
    else:
        canvas = np.zeros((rows * height, columns * width, channels), dtype=np.uint8)

    for index, image in enumerate(array):
        row, col = divmod(index, columns)
        y, x = row * height, col * width
        canvas[y : y + height, x : x + width] = image[..., 0] if channels == 1 else image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def build_model(args) -> DDIMUNet:
    return DDIMUNet(
        image_channels=args.channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    )


def train(args) -> None:
    data_dir = args.data_dir
    if args.dataset == "oxford-flowers":
        data_dir = download_oxford_flowers(data_dir or "data/oxford-flowers")
    elif args.dataset == "cifar-10":
        data_dir = download_cifar10(data_dir or "data/cifar-10")
    elif data_dir is None:
        raise ValueError("--data-dir is required when --dataset image-folder")

    paths = list_images(data_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Images: {len(paths)}")
    print(f"Image size: {args.image_size}x{args.image_size}, channels: {args.channels}")
    print(f"Batch size: {args.batch_size}, steps: {args.steps}, timesteps: {args.timesteps}")
    print(f"Output directory: {output_dir}")

    model = build_model(args)
    schedule = DDIMSchedule(timesteps=args.timesteps, schedule=args.noise_schedule)
    optimizer = optim.AdamW(learning_rate=args.lr)
    batches = image_batches(paths, args.batch_size, args.image_size, args.channels)

    def loss_fn(model, x0):
        return schedule.training_loss(model, x0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    progress = trange(1, args.steps + 1, desc="training")
    for step in progress:
        x0 = mx.array(next(batches))
        loss, grads = loss_and_grad(model, x0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        progress.set_postfix(loss=f"{float(loss):.4f}")

        if args.sample_every and step % args.sample_every == 0:
            samples = schedule.sample(
                model,
                (args.sample_count, args.image_size, args.image_size, args.channels),
                sampler=args.sampler,
                sample_steps=args.sample_steps,
                eta=args.ddim_eta,
            )
            sample_path = output_dir / f"sample_{step:06d}.png"
            save_image_grid(samples, str(sample_path))
            progress.write(f"Saved sample grid: {sample_path}")

        if args.save_every and step % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"ddim_step_{step:06d}.safetensors"
            model.save_weights(str(checkpoint_path))
            progress.write(f"Saved checkpoint: {checkpoint_path}")

    final_path = output_dir / "ddim.safetensors"
    model.save_weights(str(final_path))
    print(f"Saved final checkpoint: {final_path}")


def sample(args) -> None:
    model = build_model(args)
    model.load_weights(args.checkpoint)
    schedule = DDIMSchedule(timesteps=args.timesteps, schedule=args.noise_schedule)
    images = schedule.sample(
        model,
        (args.count, args.image_size, args.image_size, args.channels),
        sampler=args.sampler,
        sample_steps=args.sample_steps,
        eta=args.ddim_eta,
    )
    save_image_grid(images, args.output)


def smoke(args) -> None:
    model = build_model(args)
    schedule = DDIMSchedule(timesteps=args.timesteps, schedule=args.noise_schedule)
    x0 = mx.random.normal((2, args.image_size, args.image_size, args.channels))
    loss = schedule.training_loss(model, x0)
    samples = schedule.sample(
        model,
        (2, args.image_size, args.image_size, args.channels),
        sampler=args.sampler,
        sample_steps=args.sample_steps,
        eta=args.ddim_eta,
    )
    mx.eval(loss, samples)
    print(f"smoke loss={float(loss):.4f}, samples_shape={samples.shape}")


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    base_channels: int = 64,
    time_dim: int = 256,
    timesteps: int = 1_000,
) -> None:
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3, choices=(1, 3))
    parser.add_argument("--base-channels", type=int, default=base_channels)
    parser.add_argument("--time-dim", type=int, default=time_dim)
    parser.add_argument("--timesteps", type=int, default=timesteps)
    parser.add_argument("--noise-schedule", choices=("linear", "cosine"), default="linear")


def add_sampler_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sampler", choices=("ancestral", "ddim"), default="ddim")
    parser.add_argument("--sample-steps", type=int, help="Number of denoising steps for DDIM sampling")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM stochasticity; 0 is deterministic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and sample a small MLX DDIM study model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_model_args(train_parser)
    train_parser.add_argument("--dataset", choices=("image-folder", "oxford-flowers", "cifar-10"), default="image-folder")
    train_parser.add_argument("--data-dir")
    train_parser.add_argument("--output-dir", default="runs/ddim")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--sample-every", type=int, default=1_000)
    train_parser.add_argument("--sample-count", type=int, default=16)
    train_parser.add_argument("--save-every", type=int, default=1_000)
    add_sampler_args(train_parser)
    train_parser.set_defaults(func=train)

    sample_parser = subparsers.add_parser("sample")
    add_model_args(sample_parser)
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--output", default="runs/ddim/samples.png")
    sample_parser.add_argument("--count", type=int, default=16)
    add_sampler_args(sample_parser)
    sample_parser.set_defaults(func=sample)

    smoke_parser = subparsers.add_parser("smoke")
    add_model_args(smoke_parser, base_channels=16, time_dim=64, timesteps=8)
    add_sampler_args(smoke_parser)
    smoke_parser.set_defaults(func=smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
