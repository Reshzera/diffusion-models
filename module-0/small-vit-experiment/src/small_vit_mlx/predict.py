from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from .data import CIFAR10_CLASSES, load_cifar10, normalize_images
from .model import SmallViT


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32)[None, ...]


def predict(model: SmallViT, image: np.ndarray) -> tuple[int, float, np.ndarray]:
    model.eval()
    x = mx.array(normalize_images(image))
    logits = model(x)
    probs = mx.softmax(logits, axis=-1)
    mx.eval(probs)
    probs_np = np.array(probs[0])
    class_id = int(probs_np.argmax())
    return class_id, float(probs_np[class_id]), probs_np


def build_model(args: argparse.Namespace) -> SmallViT:
    model = SmallViT(
        image_size=32,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        dropout=0.0,
    )
    model.load_weights(str(args.checkpoint))
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual inference with a trained small MLX ViT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.safetensors"))
    parser.add_argument("--image", type=Path, default=None, help="Path to a local RGB image.")
    parser.add_argument("--cifar-index", type=int, default=None, help="Index from the CIFAR-10 test split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image is None and args.cifar_index is None:
        raise SystemExit("Use --image path/to/image.png or --cifar-index N")
    if args.image is not None and args.cifar_index is not None:
        raise SystemExit("Choose only one input: --image or --cifar-index")

    model = build_model(args)

    expected_label = None
    if args.image is not None:
        image = load_image(args.image)
        source = str(args.image)
    else:
        (_, _), (test_images, test_labels) = load_cifar10(args.data_dir)
        image = test_images[args.cifar_index : args.cifar_index + 1]
        expected_label = int(test_labels[args.cifar_index])
        source = f"CIFAR-10 test index {args.cifar_index}"

    class_id, confidence, probs = predict(model, image)
    top5 = probs.argsort()[-5:][::-1]

    print(f"input: {source}")
    if expected_label is not None:
        print(f"expected: {CIFAR10_CLASSES[expected_label]}")
    print(f"predicted: {CIFAR10_CLASSES[class_id]} ({confidence:.3f})")
    print("top 5:")
    for index in top5:
        print(f"  {CIFAR10_CLASSES[int(index)]}: {probs[int(index)]:.3f}")


if __name__ == "__main__":
    main()
