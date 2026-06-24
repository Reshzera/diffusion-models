import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from cuda_ddpm.data import download_cifar10, download_oxford_flowers, image_batches, list_images
from cuda_ddpm.diffusion import DDPMSchedule
from cuda_ddpm.model import DDPMUNet


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def save_image_grid(images: torch.Tensor, path: str, columns: int = 4) -> None:
    images = images.detach().cpu().permute(0, 2, 3, 1)
    array = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
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


def build_model(args, device: torch.device) -> DDPMUNet:
    return DDPMUNet(
        image_channels=args.channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
    ).to(device)


def train(args) -> None:
    device = resolve_device(args.device)
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

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print(f"Images: {len(paths)}")
    print(f"Image size: {args.image_size}x{args.image_size}, channels: {args.channels}")
    print(f"Batch size: {args.batch_size}, steps: {args.steps}, timesteps: {args.timesteps}")
    print(f"Output directory: {output_dir}")

    model = build_model(args, device)
    schedule = DDPMSchedule(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batches = image_batches(paths, args.batch_size, args.image_size, args.channels)

    progress = trange(1, args.steps + 1, desc="training")
    for step in progress:
        x0 = torch.from_numpy(next(batches)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = schedule.training_loss(model, x0)
        loss.backward()
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if args.sample_every and step % args.sample_every == 0:
            samples = schedule.sample(model, (args.sample_count, args.channels, args.image_size, args.image_size))
            sample_path = output_dir / f"sample_{step:06d}.png"
            save_image_grid(samples, str(sample_path))
            progress.write(f"Saved sample grid: {sample_path}")

        if args.save_every and step % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"ddpm_step_{step:06d}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            progress.write(f"Saved checkpoint: {checkpoint_path}")

    final_path = output_dir / "ddpm.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final checkpoint: {final_path}")


def sample(args) -> None:
    device = resolve_device(args.device)
    model = build_model(args, device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    schedule = DDPMSchedule(timesteps=args.timesteps, device=device)
    images = schedule.sample(model, (args.count, args.channels, args.image_size, args.image_size))
    save_image_grid(images, args.output)


def smoke(args) -> None:
    device = resolve_device(args.device)
    model = build_model(args, device)
    schedule = DDPMSchedule(timesteps=args.timesteps, device=device)
    x0 = torch.randn((2, args.channels, args.image_size, args.image_size), device=device)
    loss = schedule.training_loss(model, x0)
    samples = schedule.sample(model, (2, args.channels, args.image_size, args.image_size))
    print(f"smoke loss={loss.item():.4f}, samples_shape={tuple(samples.shape)}")


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
    parser.add_argument("--device", default="auto", help="Torch device to use: auto, cuda, cuda:0, or cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and sample a small CUDA/PyTorch DDPM.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_model_args(train_parser)
    train_parser.add_argument("--dataset", choices=("image-folder", "oxford-flowers", "cifar-10"), default="image-folder")
    train_parser.add_argument("--data-dir")
    train_parser.add_argument("--output-dir", default="runs/cuda-ddpm")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--steps", type=int, default=10_000)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--sample-every", type=int, default=1_000)
    train_parser.add_argument("--sample-count", type=int, default=16)
    train_parser.add_argument("--save-every", type=int, default=1_000)
    train_parser.set_defaults(func=train)

    sample_parser = subparsers.add_parser("sample")
    add_model_args(sample_parser)
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--output", default="runs/cuda-ddpm/samples.png")
    sample_parser.add_argument("--count", type=int, default=16)
    sample_parser.set_defaults(func=sample)

    smoke_parser = subparsers.add_parser("smoke")
    add_model_args(smoke_parser, base_channels=16, time_dim=64, timesteps=8)
    smoke_parser.set_defaults(func=smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
